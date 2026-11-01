# Copyright 2026 NanoCad lab, UCLA
# https://nanocad.ee.ucla.edu/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU Mamba2 inference laws ported from archive fdbb0ea.

Decode arithmetic and fusion order are preserved. Prefill reuses CIM's SSD
work law and is NEW, UNVALIDATED against old goldens. No legacy config aliases.
"""
import math
from typing import Any, Dict, Sequence, Tuple

import llm_util
from train_timing import GemmType
from dataclasses import dataclass

SSM_OP_NAMES = ("layernorm1", "ssm_proj_conv", "ssm_scan", "ssm_gate_out_proj",
                "layernorm2", "ffn1", "gelu", "ffn2")


@dataclass(frozen=True)
class LocalDims:
    D: int
    d_ssm: int
    d_ssm_local: int
    groups: int
    groups_local: int
    heads: int
    heads_local: int
    state_size: int
    conv_kernel: int
    conv_width_local: int
    tp_shard_factor: int

    @property
    def L_local(self):
        return float(self.d_ssm_local) / self.groups_local

    @property
    def g_h_local(self):
        return float(self.heads_local) / self.groups_local

    @property
    def needs_rms_stats(self):
        return self.tp_shard_factor > 1


def local_dims(tc):
    cfg = tc.ssm
    width = cfg.resolve_d_inner(tc.hidden_dim)
    tp = int(tc.tp)
    group_shard = math.gcd(int(cfg.n_groups), tp)
    return LocalDims(tc.hidden_dim, width, width // tp, int(cfg.n_groups),
        int(cfg.n_groups) // group_shard, int(cfg.n_heads), int(cfg.n_heads) // tp,
        cfg.d_state, cfg.d_conv,
        width // tp + 2 * math.ceil(cfg.n_groups * cfg.d_state / tp),
        tp // group_shard)

from timing_model import CollectiveType, DirectionTiming, OperationTiming

ALL_REDUCE = CollectiveType.ALL_REDUCE


def build_ssm_timings(
    tc: Any,
    *,
    batch_size: int,
    seq_len: int,
) -> Tuple[Dict[str, OperationTiming], Dict[str, float]]:
    # NOTE ON FUSION ASSUMPTIONS (IMPORTANT FOR TIMING/ENERGY METADATA)
    #
    # We mirror transformer timing semantics:
    #   - LayerNorm/Residual/GELU/SwiGLU are modeled explicitly (same helpers).
    #   - GEMMs use existing FFN1/FFN2 sharding rules.
    #
    # SSM-specific per-element ops are treated as "trivially fusable" into the
    # producing kernel epilogue, so they do NOT get separate roofline time or
    # mem_bytes. This keeps SSM aggregation consistent with transformer timing
    # while avoiding double-counting pointwise work.
    #
    # Trivially fusable (not modeled as standalone roofline):
    #   - Bias/activation/scale on GEMM outputs (in-proj, gate-proj, out-proj)
    #   - Per-element transforms on conv output (delta/exp/nonlin)
    #   - Per-element transforms on scan-in / scan-out GEMM outputs
    #   - Simple skip/add or gating multiplies when applied to the same tensor
    #
    # Not trivially fusable (modeled explicitly):
    #   - LayerNorm/Residual/GELU/SwiGLU (transformer-style helpers)
    #   - RMSNorm *reduction* (statistics): reduction across hidden dim
    #   - Scan recurrence/state update: sequential dependency across time/state
    #   - RMS stats allreduce: hard synchronization collective when groups are sharded
    #
    # We model TWO explicit rooflines for SSM-specific kernels:
    #   (1) Depthwise causal conv (per-channel 1D conv, not a GEMM)
    #   (2) Selective scan/state update (recurrence), including local RMS stats math
    #
    # If you want a more conservative bandwidth model, reintroduce roofline
    # time for additional pointwise ops and count read+write bytes separately.
    dims = local_dims(tc)
    tokens = float(batch_size) * float(seq_len)
    if tokens <= 0:
        tokens = 0.0
    tokens_int = int(math.ceil(tokens)) if tokens > 0 else 0

    D = dims.D
    d_ssm = dims.d_ssm
    d_ssm_local = dims.d_ssm_local
    groups = dims.groups
    groups_local = dims.groups_local
    heads = dims.heads
    state_size = dims.state_size
    conv_kernel = dims.conv_kernel
    L_local = dims.L_local
    g_h_local = dims.g_h_local
    precision_bytes = float(tc.precision.activations)
    intermediate_size = int(getattr(tc, "intermediate_size", 0) or 0)

    z_width = d_ssm
    conv_width_global = d_ssm + 2 * groups * state_size
    conv_width_local = dims.conv_width_local
    in_proj_out = z_width + conv_width_global + heads

    transformer_timings: Dict[str, OperationTiming] = {}

    # LayerNorm 1 + residual (transformer-style)
    residual1_f = tc.get_residual_f(tensor_shape=(int(tokens_int), int(d_ssm), int(D)))
    layernorm1_f, layernorm1_reduction_f, LN1_comm_bytes_f = tc.get_layernorm_f(
        batch=batch_size,
        seq_len=seq_len,
        d_model=D,
    )
    transformer_timings["layernorm1"] = OperationTiming(
        "layernorm1",
        forward=DirectionTiming(
            compute_time=layernorm1_f + residual1_f,
            comm_time=layernorm1_reduction_f,
            comm_bytes=int(LN1_comm_bytes_f or 0),
        ),
        backward=None,
    )

    # Stage 1: in-projection + depthwise conv
    in_proj_time, in_proj_comm_time, in_proj_comm_bytes, in_proj_flops, in_proj_mem = tc.parallelism_gemm_forward(
        (int(tokens_int), int(D), int(in_proj_out)),
        "ssm_in_proj_f",
        gemm_type=GemmType.FFN1,
    )
    # Depthwise conv proxy (NOT a GEMM):
    # - conv_dim == conv_width channels
    # - Each output element does a 1D dot with kernel length = conv_kernel.
    # - We approximate flops as 2 * conv_kernel per output element.
    conv_k = int(conv_kernel)
    conv_flops = float(2 * conv_k) * tokens * float(conv_width_local)
    # Memory model: read conv_kernel inputs + write 1 output per element.
    # We intentionally do NOT multiply weight bytes per output element (weights are tiny + cached).
    conv_mem_bytes = tokens * float(conv_width_local) * float(conv_k + 1) * precision_bytes
    conv_time = tc.roofline(
        conv_flops,
        conv_mem_bytes,
        name="ssm_conv_dw_f",
        mem_level=tc.num_levels - 1,
    ) + tc.O
    conv_comm_time = 0.0
    conv_comm_bytes = 0

    transformer_timings["ssm_proj_conv"] = OperationTiming(
        "ssm_proj_conv",
        forward=DirectionTiming(
            compute_time=in_proj_time + conv_time,
            comm_time=in_proj_comm_time + conv_comm_time,
            comm_bytes=int((in_proj_comm_bytes or 0) + (conv_comm_bytes or 0)),
            flops=in_proj_flops + conv_flops,
            memory_accesses=dict(tc._mem_levels(tc._combine_mem(in_proj_mem, conv_mem_bytes))),
        ),
        backward=None,
    )

    # Stage 2: scan only
    scan_common_ops_group = (
        (L_local * state_size)
        + (g_h_local * state_size)
        + 3*(L_local * state_size)
        + (L_local * state_size)
    )
    scan_readout_ops_group = (L_local * state_size) + (L_local * (state_size - 1))
    scan_ops_groups = groups_local * (scan_common_ops_group + scan_readout_ops_group)

    # Scan roofline covers the recurrence + local RMS stats math.
    scan_flops = scan_ops_groups * tokens
    # Read + write of state per token.
    state_bytes_per_token = 2.0 * float(d_ssm_local * state_size) * precision_bytes
    scan_mem_bytes = tokens * state_bytes_per_token
    rms_reduce_flops = float(2 * d_ssm_local) * tokens
    rms_reduce_mem_bytes = tokens * float(d_ssm_local) * float(tc.precision.stats)
    scan_flops_total = scan_flops + rms_reduce_flops
    scan_mem_bytes_total = scan_mem_bytes + rms_reduce_mem_bytes
    if seq_len > 1:
        # NEW, UNVALIDATED GPU prefill: same SSD work law as the CIM path.
        # Chunk EACH sequence independently; never flatten batch into time.
        from cim_timing import ssd_chunked_scan_work, ssm_recurrent_scan_work
        kwargs = dict(d_inner=d_ssm_local, d_state=state_size,
                      n_groups=groups_local, n_heads=dims.heads_local,
                      act_bytes=precision_bytes)
        if tc.ssm.chunk_size > 1:
            work = ssd_chunked_scan_work(seq_len, chunk_size=tc.ssm.chunk_size, **kwargs)
        else:
            work = ssm_recurrent_scan_work(seq_len, **kwargs)
        scan_flops_total = batch_size * work.ops + rms_reduce_flops
        scan_mem_bytes_total = batch_size * work.state_bytes + rms_reduce_mem_bytes
    scan_time = tc.roofline(
        scan_flops_total,
        scan_mem_bytes_total,
        name="ssm_scan_f",
        mem_level=tc.num_levels - 1,
    ) + tc.O

    # RMSNorm stats collective (only when TP shards groups)
    rms_comm_time = 0.0
    rms_comm_bytes = 0
    if dims.needs_rms_stats:
        rms_elements = tokens_int * groups_local
        per_rank_bytes = tc.precision.stats * float(rms_elements)
        rms_comm_bytes = int(math.ceil(per_rank_bytes))
        participants = int(dims.tp_shard_factor)
        if rms_comm_bytes > 0:
            rms_comm_time = tc.network_model.collective(
                kind=ALL_REDUCE,
                size_bytes=rms_comm_bytes,
                participants=participants,
                ib=tc.links["tp"].bandwidth,
                ll=tc.links["tp"].latency,
                local_bytes=0,
                debug_label="ssm_rms_stats_allreduce",
                axis="tp",
            )

    transformer_timings["ssm_scan"] = OperationTiming(
        "ssm_scan",
        forward=DirectionTiming(
            compute_time=scan_time,
            comm_time=rms_comm_time,
            comm_bytes=int(rms_comm_bytes or 0),
            flops=scan_flops_total,
            memory_accesses=dict(tc._mem_levels(scan_mem_bytes_total)),
        ),
        backward=None,
    )

    # Stage 3: gate + out projection + residual
    # Elementwise gate multiply (no GEMM).
    gate_flops = 0.0
    gate_mem_bytes = 0.0
    gate_time = 0.0
    if tokens > 0 and d_ssm_local > 0:
        gate_flops = tokens * float(d_ssm_local)
        gate_mem_bytes = tokens * float(d_ssm_local) * precision_bytes * 3.0
        gate_time = tc.roofline(
            gate_flops,
            gate_mem_bytes,
            name="ssm_gate_f",
            mem_level=tc.num_levels - 1,
        ) + tc.O
    gate_comm_time = 0.0
    gate_comm_bytes = 0
    out_proj_time, out_proj_comm_time, out_proj_comm_bytes, out_proj_flops, out_proj_mem = tc.parallelism_gemm_forward(
        (int(tokens_int), int(d_ssm), int(D)),
        "ssm_out_proj_f",
        gemm_type=GemmType.FFN2,
    )
    transformer_timings["ssm_gate_out_proj"] = OperationTiming(
        "ssm_gate_out_proj",
        forward=DirectionTiming(
            compute_time=gate_time + out_proj_time,
            comm_time=gate_comm_time + out_proj_comm_time,
            comm_bytes=int((gate_comm_bytes or 0) + (out_proj_comm_bytes or 0)),
            flops=gate_flops + out_proj_flops,
            memory_accesses=dict(tc._mem_levels(tc._combine_mem(gate_mem_bytes, out_proj_mem))),
        ),
        backward=None,
    )

    # LayerNorm 2 + MLP (only when intermediate_size is enabled)
    if intermediate_size > 0:
        ffn1_dim = int(tc._ffn1_output_dim(intermediate_size))
        gemm_ffn1 = (int(tokens_int), int(D), int(ffn1_dim))
        gemm_ffn2 = (int(tokens_int), int(ffn1_dim), int(D))

        ffn1_gemm_f, ffn1_reduction_f, ffn1_size_f, ffn1_flops_f, ffn1_mem_f = tc.parallelism_gemm_forward(
            gemm_ffn1,
            "ssm_ffn1_f",
            gemm_type=GemmType.FFN1,
        )
        ffn2_gemm_f, ffn2_reduction_f, ffn2_size_f, ffn2_flops_f, ffn2_mem_f = tc.parallelism_gemm_forward(
            gemm_ffn2,
            "ssm_ffn2_f",
            gemm_type=GemmType.FFN2,
        )

        transformer_timings["ffn1"] = OperationTiming(
            "ffn1",
            forward=DirectionTiming(
                compute_time=ffn1_gemm_f,
                comm_time=ffn1_reduction_f,
                comm_bytes=int(ffn1_size_f or 0),
                flops=ffn1_flops_f,
                memory_accesses=dict(tc._mem_levels(ffn1_mem_f)),
            ),
            backward=None,
        )
        transformer_timings["ffn2"] = OperationTiming(
            "ffn2",
            forward=DirectionTiming(
                compute_time=ffn2_gemm_f,
                comm_time=ffn2_reduction_f,
                comm_bytes=int(ffn2_size_f or 0),
                flops=ffn2_flops_f,
                memory_accesses=dict(tc._mem_levels(ffn2_mem_f)),
            ),
            backward=None,
        )

        ffn1_spec = tc._shard_gemm_descriptor(gemm_ffn1, GemmType.FFN1)
        ffn1_activation_shape = (ffn1_spec.shard_m, ffn1_spec.k, ffn1_spec.shard_n)
        if llm_util.is_llama_style(tc.model_type):
            act_f = tc.get_swiglu_f(tensor_shape=ffn1_activation_shape)
        else:
            act_f = tc.get_gelu_f(tensor_shape=ffn1_activation_shape)
        transformer_timings["gelu"] = OperationTiming(
            "gelu",
            forward=DirectionTiming(compute_time=act_f, comm_time=0.0, comm_bytes=0),
            backward=None,
        )

        residual2_f = tc.get_residual_f(tensor_shape=gemm_ffn2)
        layernorm2_f, layernorm2_reduction_f, LN2_comm_bytes_f = tc.get_layernorm_f(
            batch=batch_size,
            seq_len=seq_len,
            d_model=D,
        )
        transformer_timings["layernorm2"] = OperationTiming(
            "layernorm2",
            forward=DirectionTiming(
                compute_time=layernorm2_f + residual2_f,
                comm_time=layernorm2_reduction_f,
                comm_bytes=int(LN2_comm_bytes_f or 0),
            ),
            backward=None,
        )
    else:
        transformer_timings["layernorm2"] = OperationTiming(
            "layernorm2",
            forward=DirectionTiming(compute_time=0.0, comm_time=0.0, comm_bytes=0),
            backward=None,
        )
        transformer_timings["ffn1"] = OperationTiming(
            "ffn1",
            forward=DirectionTiming(compute_time=0.0, comm_time=0.0, comm_bytes=0),
            backward=None,
        )
        transformer_timings["gelu"] = OperationTiming(
            "gelu",
            forward=DirectionTiming(compute_time=0.0, comm_time=0.0, comm_bytes=0),
            backward=None,
        )
        transformer_timings["ffn2"] = OperationTiming(
            "ffn2",
            forward=DirectionTiming(compute_time=0.0, comm_time=0.0, comm_bytes=0),
            backward=None,
        )

    embedding_f, embedding_mem = tc.get_embedding_f(
        vocab_size=tc.vocab_size,
        seq_len=seq_len,
        hidden_dim=D,
    )
    embedding_b = tc.get_embedding_b(
        vocab_size=tc.vocab_size,
        seq_len=seq_len,
        hidden_dim=D,
    )
    if tc.disable_embedding_unembedding:
        embedding_f = 0.0
        embedding_b = 0.0
        embedding_mem = {}
    transformer_timings["embedding"] = OperationTiming(
        "embedding",
        forward=DirectionTiming(
            compute_time=embedding_f,
            comm_time=0.0,
            comm_bytes=0,
            memory_accesses=dict(tc._mem_levels(embedding_mem)),
        ),
        backward=DirectionTiming(
            compute_time=embedding_b,
            comm_time=0.0,
            comm_bytes=0,
        ),
    )

    gemm_linear = (batch_size, seq_len, D, tc.vocab_size)
    linear_softmax_f, linear_softmax_mem = tc.get_linear_softmax_f(gemm=gemm_linear)
    linear_softmax_b = tc.get_linear_softmax_b(gemm=gemm_linear)
    if tc.disable_embedding_unembedding:
        linear_softmax_f = 0.0
        linear_softmax_b = 0.0
        linear_softmax_mem = {}
    transformer_timings["linear_softmax"] = OperationTiming(
        "linear_softmax",
        forward=DirectionTiming(
            compute_time=linear_softmax_f,
            comm_time=0.0,
            comm_bytes=0,
            memory_accesses=dict(tc._mem_levels(linear_softmax_mem)),
        ),
        backward=DirectionTiming(
            compute_time=linear_softmax_b,
            comm_time=0.0,
            comm_bytes=0,
        ),
    )

    transformer_forward = 0.0
    for name in SSM_OP_NAMES:
        op = transformer_timings.get(name)
        if op is not None and op.forward is not None:
            transformer_forward += op.forward.total_time()
    embedding_forward = transformer_timings["embedding"].total_forward_time()
    embedding_backward = transformer_timings["embedding"].total_backward_time()
    linear_softmax_forward = transformer_timings["linear_softmax"].total_forward_time()
    linear_softmax_backward = transformer_timings["linear_softmax"].total_backward_time()
    node_breakdown = {
        "transformer_time_f": transformer_forward,
        "transformer_time_b": 0.0,
        "linear_softmax_f": linear_softmax_forward,
        "linear_softmax_b": linear_softmax_backward,
        "embedding_f": embedding_forward,
        "embedding_b": embedding_backward,
    }

    tc.ssm_kernel_evidence = dict(
        in_proj=in_proj_time, conv=conv_time, scan=scan_time, gate=gate_time,
        out_proj=out_proj_time, rms_bytes=rms_comm_bytes,
        rms_participants=dims.tp_shard_factor,
        conv_flops=conv_flops, conv_mem_bytes=conv_mem_bytes,
        scan_flops=scan_flops_total, scan_mem_bytes=scan_mem_bytes_total,
    )
    return transformer_timings, node_breakdown


def build_memory_data(estimator, *, mode, batch_size, seq_len,
                      gemm_shapes=None, kv_cache_tokens=None):
    """Per-layer/per-device inference memory, replayed by program.memory_sim.

    Archive component formulas are retained except conv history (d_conv-1).
    State is resident for every owned SSM layer, independent of sequence length.
    Attention layers retain the existing transformer's memory data verbatim.
    """
    import copy
    from memory_estimation import MemKind, MemoryEstimator

    tc = estimator.time_calc
    if mode != "inference":
        raise ValueError("GPU SSM memory supports inference only")
    dims = local_dims(tc)
    p = tc.precision
    D, inner, local = dims.D, dims.d_ssm, dims.d_ssm_local
    groups, state, heads = dims.groups_local, dims.state_size, dims.heads_local
    width = dims.conv_width_local
    tokens = batch_size * seq_len
    act = float(p.activations)
    def output(k, n, kind):
        return float(tc._gemm_output_elements((tokens, k, n), kind)) * act

    proj = output(D, 2 * inner + 2 * dims.groups * state + dims.heads, GemmType.FFN1)
    out = output(inner, D, GemmType.FFN2)
    ffn_out = 0.0
    ffn_params = 0.0
    if tc.ffn_per_layer:
        ffn_dim = tc._ffn1_output_dim(tc.intermediate_size)
        ffn_out = output(ffn_dim, D, GemmType.FFN2)
        ffn_params = float(2 * D * ffn_dim) / tc.tp
    # Archive parameter residency convention: grouped B/C projection columns
    # are replicated when TP splits a group; LN/RMS affine vectors included.
    xbc = local + 2 * groups * state
    params = D * (2 * local + 2 * groups * state + heads) + local * D
    params += xbc * dims.conv_kernel + 3 * heads + D + local + ffn_params
    weights = float(params) * float(p.parameters + p.master_parameters)
    state_bytes = float(batch_size * local * state) * act
    conv_bytes = float(batch_size * width * (dims.conv_kernel - 1)) * act
    outputs = {
        MemKind.SSM_PROJ_CONV: max(proj, tokens * width * act),
        MemKind.SSM_SCAN: tokens * local * act + tokens * groups * float(p.stats),
        MemKind.SSM_GATE_OUT_PROJ: out, MemKind.LAYERNORM1: out,
        MemKind.LAYERNORM2: out if tc.ffn_per_layer else 0.0,
        MemKind.MLP: ffn_out, MemKind.EMBEDDING: 0.0,
        MemKind.SOFTMAX: 0.0, MemKind.OPTIMIZER: 0.0,
        MemKind.TRANSFORMER: max(proj, tokens * width * act, out, ffn_out),
    }
    if not tc.disable_embedding_unembedding:
        outputs[MemKind.EMBEDDING] = out
        outputs[MemKind.SOFTMAX] = tokens * math.ceil(tc.vocab_size / tc.tp) * act
    ssm = dict(weight_mem_per_layer=weights, static_mem_per_layer=weights,
        activation_mem_per_layer=outputs[MemKind.TRANSFORMER],
        ssm_state_bytes_per_layer=state_bytes, conv_state_bytes_per_layer=conv_bytes,
        ssm_cache_bytes_per_layer=state_bytes + conv_bytes,
        kv_cache_bytes_per_layer=0.0, persistent_bytes_by_kind={},
        transient_bytes_by_kind=outputs)
    has_attention = any("attention" in kinds for kinds in tc.layer_mixers)
    if has_attention:
        proxy = copy.copy(tc)
        proxy.ssm = None
        baseline = MemoryEstimator(proxy).build_memory_data(
            mode=mode, batch_size=batch_size, seq_len=seq_len,
            gemm_shapes=gemm_shapes, kv_cache_tokens=kv_cache_tokens)
    else:
        emb, head = tc._embedding_softmax_params_per_rank(D, tc.vocab_size)
        from program.layout import RankLayout, cluster_coords
        sizes = dict(tp=tc.tp, cp=tc.cp, ep=tc.ep, pp=tc.pp, dp=tc.dp)
        layout, _ = RankLayout.from_network_layout(tc.hw_config.network_layout,
            sizes, False, extract_optimize_2dmap=False,
            unsupported_axis_context="SSM memory estimation", empty_axis_order_is_none=True)
        extra = {}
        for stage in range(tc.pp):
            for rank in range(tc.tp * tc.cp * tc.ep):
                device = layout.linearize(cluster_coords(layout.axis_order, rank, stage,
                                          sizes=layout.axis_sizes))
                value = float(tc.hw_config.sw_config.const_mem_offset)
                value += (emb if stage == 0 else 0) * float(p.parameters + p.master_parameters)
                value += (head if stage == tc.pp - 1 else 0) * float(p.parameters + p.master_parameters)
                extra[device] = value
        baseline = dict(ssm, extra_static_bytes_per_device=extra)
    by_layer = {}
    for i, mixer in enumerate(tc.layer_mixers):
        if "ssm" in mixer:
            data = dict(ssm)
            if tc.is_moe_layer(i):
                # Replace the dense MLP with the existing expert/router census.
                dense_ffn = float(2 * D * tc._ffn1_output_dim(tc.intermediate_size)) / tc.tp
                census = tc.layer_params_per_rank(D, tc.moe_intermediate_size, moe=True)
                moe_ffn = sum(v for k, v in census.items() if k in ("experts", "router"))
                data["weight_mem_per_layer"] += (moe_ffn - dense_ffn) * float(p.parameters + p.master_parameters)
                data["transient_bytes_by_kind"] = dict(outputs)
                for kind in (MemKind.MLP, MemKind.ROUTER, MemKind.MOE_DISPATCH, MemKind.MOE_COMBINE):
                    data["transient_bytes_by_kind"][kind] = baseline["transient_bytes_by_kind_moe"].get(kind, 0.0)
        else:
            moe = tc.is_moe_layer(i)
            suffix = "moe" if moe else "dense"
            data = dict(weight_mem_per_layer=baseline[f"weight_mem_per_layer_{suffix}"],
                kv_cache_bytes_per_layer=baseline["kv_cache_bytes_per_layer"],
                persistent_bytes_by_kind=baseline[f"persistent_bytes_by_kind_{suffix}"],
                transient_bytes_by_kind=baseline[f"transient_bytes_by_kind_{suffix}"])
        by_layer[i] = data
    baseline["by_layer"] = by_layer
    tc.layer_plan_memory_evidence = by_layer
    return baseline
