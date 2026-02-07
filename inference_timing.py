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

"""LLM inference prefill time-calculation entry points."""

import math
import os
from types import SimpleNamespace
import yaml
from typing import Any, Dict, List, Optional, Tuple, Mapping, Set
from train_timing import (
    LLMExecutionDispatcher,
    TimeCalculationLLM,
    GemmType,
    COMMUNICATION_RULES,
    COMM_RULE_DEFAULT_KEY,
)
from memory_estimation import MemoryEstimator
from simulate_inference_graph import DecodeSample, InferenceConfig, InferenceEngine
import llm_util
import json
from timing_model import CollectiveType, CommSpec, DirectionTiming, OperationTiming, OperationGroup
from tile import AccessBytes, formatBytes

def convert_prefix(value: float) -> float:
    """Assign SI unit prefixes to numerical values."""
    if value > 1:
        prefixes = ["", "k", "M", "G"]
        index = min(int(math.log10(value) // 3), len(prefixes) - 1)
        scaled_value = value / (1000 ** index)
        return f"{scaled_value:.2f}{prefixes[index]}"
    else:
        prefixes = ["", "m", "µ", "n"]
        index = min(int(-(math.log10(value) // 3)), len(prefixes) - 1)
        scaled_value = value * (1000 ** index)
        return f"{scaled_value:.2f}{prefixes[index]}"

class TimeCalculationLLMInference(TimeCalculationLLM):
    """Inference-specialized facade for ``TimeCalculationLLM``."""

    def __init__(self, hw_config, model_config, mode, output_dir: Optional[str] = None):
        super().__init__(hw_config, model_config, mode, output_dir)
        self._raw_model_config = model_config

    def _build_decode_transformer_results(
        self,
        *,
        batch_size: int,
        total_seq_len: int,
        use_moe_layer: bool,
        gemm_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    ) -> Tuple[Dict[str, OperationTiming], Dict[str, float]]:
        """Construct transformer timings + node breakdown for a single decode step."""

        head_dim = getattr(self, "head_dim", None)
        if head_dim is None:
            head_dim = self.hidden_dim // self.num_heads

        token_bytes = llm_util.kv_cache_token_bytes(
            batch_size=batch_size,
            kv_heads=self.kv_heads,
            head_dim=head_dim,
            precision_bytes=self.precision.kv_cache,
        )
        intermediate_size = self.moe_intermediate_size if use_moe_layer else self.intermediate_size
        gemm_ctx = self
        if use_moe_layer != self.use_moe:
            gemm_ctx = SimpleNamespace(
                use_moe=use_moe_layer,
                moe_num_experts=self.moe_num_experts,
                moe_top_k=self.moe_top_k,
                moe_intermediate_size=intermediate_size,
                model_type=self.model_type,
                head_dim=getattr(self, "head_dim", None),
            )
        if gemm_shapes is None or use_moe_layer != self.use_moe:
            gemm_shapes = llm_util.process_decode_gemm_shapes(
                gemm_ctx,
                batch_size=batch_size,
                current_seq_len=total_seq_len,
                d_model=self.hidden_dim,
                num_heads=self.num_heads,
                kv_heads=self.kv_heads,
                intermediate_size=intermediate_size,
                vocab_size=self.vocab_size,
                model_type=self.model_type,
            )

        gemm_qkv_proj = gemm_shapes["qkv_proj"]
        gemm_attention_score = gemm_shapes["attention_score"]
        gemm_attention_output = gemm_shapes["attention_output"]
        gemm_output_proj = gemm_shapes["output_proj"]
        gemm_ffn1 = gemm_shapes["ffn1"]
        gemm_ffn2 = gemm_shapes["ffn2"]
        # FlashAttention is not used during single-token (incremental) decoding.


        output_seq_len = 1

        transformer_timings: Dict[str, OperationTiming] = {}

        def _make_forward(
            op_name: str,
            compute_time: float,
            comm_time: float,
            comm_bytes: float,
            *,
            flops: float = 0.0,
            memory: Optional[Mapping[str, float]] = None,
            memory_profile: Optional[AccessBytes] = None,
        ) -> DirectionTiming:
            bytes_int = int(math.ceil(float(comm_bytes or 0.0)))
            return DirectionTiming(
                compute_time=compute_time,
                comm_time=comm_time,
                comm_bytes=bytes_int,
                flops=flops,
                memory_accesses=dict(memory) if memory else {},
                memory_profile=memory_profile,
            )

        # QKV projection
        qkv_proj_time, qkv_proj_reduction, qkv_proj_size, qkv_proj_flops, qkv_proj_mem, qkv_proj_profile = self.parallelism_gemm_forward(
            gemm_qkv_proj, "decode_qkv_proj_f", gemm_type=GemmType.QKV, return_profile=True
        )
        transformer_timings["qkv_proj"] = OperationTiming(
            "qkv_proj",
            forward=_make_forward(
                "qkv_proj",
                compute_time=qkv_proj_time,
                comm_time=qkv_proj_reduction,
                comm_bytes=qkv_proj_size,
                flops=qkv_proj_flops,
                memory=self._mem_levels(qkv_proj_mem),
                memory_profile=qkv_proj_profile,
            ),
            backward=None,
        )

        # Attention components
        attention_score_time, attention_score_reduction, attention_score_size, attention_score_flops, attention_score_mem, attention_score_profile = self.parallelism_gemm_forward(
            gemm_attention_score, "decode_attention_score_f", gemm_type=GemmType.ATTENTION_SCORE, return_profile=True
        )
        attention_output_time, attention_output_reduction, attention_output_size, attention_output_flops, attention_output_mem, attention_output_profile = self.parallelism_gemm_forward(
            gemm_attention_output, "decode_attention_output_f", gemm_type=GemmType.ATTENTION_OUTPUT, return_profile=True
        )
        attention_scale_softmax_f = self.get_scale_softmax_f(gemm_attention_score)

        attention_reduction = attention_score_reduction + attention_output_reduction
        attention_comm_bytes = attention_score_size + attention_output_size
        attention_forward_compute = attention_score_time + attention_scale_softmax_f + attention_output_time
        attention_forward_time = attention_forward_compute + attention_reduction
        attention_flops = (attention_score_flops or 0.0) + (attention_output_flops or 0.0)
        attention_mem = self._combine_mem(attention_score_mem, attention_output_mem)
        attention_profile = self._combine_profiles(attention_score_profile, attention_output_profile)

        transformer_timings["attention"] = OperationTiming(
            "attention",
            forward=_make_forward(
                "attention",
                compute_time=attention_forward_compute,
                comm_time=attention_reduction,
                comm_bytes=attention_comm_bytes,
                flops=attention_flops,
                memory=attention_mem,
                memory_profile=attention_profile,
            ),
            backward=None,
        )

        attention_scale_softmax_op = OperationTiming(
            "attention_scale_softmax",
            forward=_make_forward(
                "attention_scale_softmax",
                compute_time=attention_scale_softmax_f,
                comm_time=0.0,
                comm_bytes=0.0,
            ),
            backward=None,
        )
        transformer_timings["attention_scale_softmax"] = attention_scale_softmax_op

        # Output projection
        out_proj_time, out_proj_reduction, out_proj_size, out_proj_flops, out_proj_mem, out_proj_profile = self.parallelism_gemm_forward(
            gemm_output_proj, "decode_output_projection_f", gemm_type=GemmType.OUT_PROJ, return_profile=True
        )
        transformer_timings["output_proj"] = OperationTiming(
            "output_proj",
            forward=_make_forward(
                "output_proj",
                compute_time=out_proj_time,
                comm_time=out_proj_reduction,
                comm_bytes=out_proj_size,
                flops=out_proj_flops,
                memory=self._mem_levels(out_proj_mem),
                memory_profile=out_proj_profile,
            ),
            backward=None,
        )

        # FFN layers (dense vs MoE)
        router_time_f = 0.0
        router_comm_f = 0.0
        router_bytes_f = 0.0
        dispatch_fwd_time = 0.0
        dispatch_fwd_bytes = 0.0
        combine_fwd_time = 0.0
        moe_tokens_local = None
        moe_tokens_shared = None
        if not use_moe_layer:
            ffn1_time, ffn1_reduction, ffn1_size, ffn1_flops, ffn1_mem, ffn1_profile = self.parallelism_gemm_forward(
                gemm_ffn1, "decode_ffn1_f", gemm_type=GemmType.FFN1, return_profile=True
            )
            ffn2_time, ffn2_reduction, ffn2_size, ffn2_flops, ffn2_mem, ffn2_profile = self.parallelism_gemm_forward(
                gemm_ffn2, "decode_ffn2_f", gemm_type=GemmType.FFN2, return_profile=True
            )
        else:
            gemm_router = gemm_shapes.get("router")
            if gemm_router is None:
                raise KeyError("Missing decode GEMM shape for 'router'")
            allow_padding = self._moe_allow_padding()
            (
                tokens_owner,
                tokens_dispatched,
                tokens_local,
                _experts_per_rank,
                _tokens_per_expert,
            ) = self._moe_routed_tokens_per_expert(
                batch_size,
                output_seq_len,
                allow_padding=allow_padding,
            )
            moe_tokens_local = tokens_local
            moe_tokens_shared = self._moe_tokens_shared(tokens_owner)
            router_time_f, router_comm_f, router_bytes_f = self.get_router_f(
                gemm_router,
                gemm_ffn1,
                batch_size=batch_size,
                seq_len=output_seq_len,
            )
            moe_group = self._moe_routing_group()
            axis = None
            dispatch_fwd_bytes = int(
                math.ceil(self.precision.activations * tokens_dispatched * self.hidden_dim)
            )
            dispatch_fwd_time = self.network_model.collective(
                kind=CollectiveType.ALL_TO_ALL,
                size_bytes=dispatch_fwd_bytes,
                participants=moe_group,
                ib=self.links["ep"].bandwidth,
                ll=self.links["ep"].latency,
                local_bytes=0.0,
                debug_label="decode_moe_dispatch_f_all_to_all",
                axis=axis,
            )
            ffn1_time, ffn1_reduction, ffn1_size, ffn1_flops, ffn1_mem, ffn1_profile = self.get_moe_ffn_f(
                gemm_ffn1,
                "decode_ffn1_f",
                gemm_type=GemmType.FFN1,
                batch_size=batch_size,
                seq_len=output_seq_len,
                allow_padding=allow_padding,
                return_profile=True,
            )
            ffn2_time, ffn2_reduction, ffn2_size, ffn2_flops, ffn2_mem, ffn2_profile = self.get_moe_ffn_f(
                gemm_ffn2,
                "decode_ffn2_f",
                gemm_type=GemmType.FFN2,
                batch_size=batch_size,
                seq_len=output_seq_len,
                allow_padding=allow_padding,
                return_profile=True,
            )
            combine_fwd_time = self.network_model.collective(
                kind=CollectiveType.ALL_TO_ALL,
                size_bytes=dispatch_fwd_bytes,
                participants=moe_group,
                ib=self.links["ep"].bandwidth,
                ll=self.links["ep"].latency,
                local_bytes=0.0,
                debug_label="decode_moe_combine_f_all_to_all",
                axis=axis,
            )
            transformer_timings["router"] = OperationTiming(
                "router",
                forward=_make_forward(
                    "router",
                    compute_time=router_time_f,
                    comm_time=router_comm_f,
                    comm_bytes=router_bytes_f,
                ),
                backward=None,
            )
            transformer_timings["moe_dispatch"] = OperationTiming(
                "moe_dispatch",
                forward=_make_forward(
                    "moe_dispatch",
                    compute_time=0.0,
                    comm_time=dispatch_fwd_time,
                    comm_bytes=dispatch_fwd_bytes,
                ),
                backward=None,
            )
            transformer_timings["moe_combine"] = OperationTiming(
                "moe_combine",
                forward=_make_forward(
                    "moe_combine",
                    compute_time=0.0,
                    comm_time=combine_fwd_time,
                    comm_bytes=dispatch_fwd_bytes,
                ),
                backward=None,
            )

        transformer_timings["ffn1"] = OperationTiming(
            "ffn1",
            forward=_make_forward(
                "ffn1",
                compute_time=ffn1_time,
                comm_time=ffn1_reduction,
                comm_bytes=ffn1_size,
                flops=ffn1_flops,
                memory=self._mem_levels(ffn1_mem),
                memory_profile=locals().get("ffn1_profile"),
            ),
            backward=None,
        )
        transformer_timings["ffn2"] = OperationTiming(
            "ffn2",
            forward=_make_forward(
                "ffn2",
                compute_time=ffn2_time,
                comm_time=ffn2_reduction,
                comm_bytes=ffn2_size,
                flops=ffn2_flops,
                memory=self._mem_levels(ffn2_mem),
                memory_profile=locals().get("ffn2_profile"),
            ),
            backward=None,
        )

        # GELU/SwiGLU activation
        ffn1_spec = self._shard_gemm_descriptor(gemm_ffn1, GemmType.FFN1)
        ffn1_activation_shape = (ffn1_spec.shard_m, ffn1_spec.k, ffn1_spec.shard_n)
        ffn1_activation_shape_shared = None
        if use_moe_layer and moe_tokens_local is not None:
            use_tp_sharded = bool(getattr(self, "tp_ep", True))
            ffn1_n = ffn1_spec.shard_n if use_tp_sharded else ffn1_spec.n
            ffn1_activation_shape = (moe_tokens_local, ffn1_spec.k, ffn1_n)
            if moe_tokens_shared and moe_tokens_shared > 0:
                ffn1_activation_shape_shared = (moe_tokens_shared, ffn1_spec.k, ffn1_n)
        if llm_util.is_llama_style(self.model_type):
            act_f = self.get_swiglu_f(ffn1_activation_shape)
            if ffn1_activation_shape_shared is not None:
                act_f += self.get_swiglu_f(ffn1_activation_shape_shared)
        else:
            act_f = self.get_gelu_f(ffn1_activation_shape)
            if ffn1_activation_shape_shared is not None:
                act_f += self.get_gelu_f(ffn1_activation_shape_shared)
        transformer_timings["gelu"] = OperationTiming(
            "gelu",
            forward=_make_forward("gelu", compute_time=act_f, comm_time=0.0, comm_bytes=0.0),
            backward=None,
        )

        # Layer norms
        head_dim = getattr(self, "head_dim", None)
        if head_dim is None:
            head_dim = self.hidden_dim // self.num_heads
        q_size = self.num_heads * head_dim
        output_proj_shape = (
            batch_size,
            output_seq_len,
            q_size,
            self.hidden_dim,
        )
        residual1_f = self.get_residual_f(output_proj_shape)
        layernorm1_f, layernorm1_reduction, layernorm1_bytes = self.get_layernorm_f(
            batch=batch_size, seq_len=output_seq_len, d_model=self.hidden_dim
        )
        transformer_timings["layernorm1"] = OperationTiming(
            "layernorm1",
            forward=_make_forward(
                "layernorm1",
                compute_time=layernorm1_f + residual1_f,
                comm_time=layernorm1_reduction,
                comm_bytes=layernorm1_bytes,
            ),
            backward=None,
        )

        ffn2_shape = (
            batch_size,
            output_seq_len,
            intermediate_size,
            self.hidden_dim,
        )
        residual2_f = self.get_residual_f(ffn2_shape)
        layernorm2_f, layernorm2_reduction, layernorm2_bytes = self.get_layernorm_f(
            batch=batch_size, seq_len=output_seq_len, d_model=self.hidden_dim
        )
        transformer_timings["layernorm2"] = OperationTiming(
            "layernorm2",
            forward=_make_forward(
                "layernorm2",
                compute_time=layernorm2_f + residual2_f,
                comm_time=layernorm2_reduction,
                comm_bytes=layernorm2_bytes,
            ),
            backward=None,
        )

        linear_shape = (
            batch_size,
            output_seq_len,
            self.hidden_dim,
            self.vocab_size,
        )
        linear_softmax_f, linear_softmax_mem = self.get_linear_softmax_f(linear_shape)
        transformer_timings["linear_softmax"] = OperationTiming(
            "linear_softmax",
            forward=_make_forward(
                "linear_softmax",
                compute_time=linear_softmax_f,
                comm_time=0.0,
                comm_bytes=0.0,
                memory=self._mem_levels(linear_softmax_mem),
            ),
            backward=None,
        )

        mlp_group = OperationGroup(
            "MLP",
            operations=(
                transformer_timings["ffn1"],
                transformer_timings["gelu"],
                transformer_timings["ffn2"],
            ),
        )

        # Match exact floating-point operation order from original code
        qkv_proj_forward = qkv_proj_time + qkv_proj_reduction
        attention_forward = attention_score_time + attention_scale_softmax_f + attention_output_time + attention_reduction
        out_proj_forward = out_proj_time + out_proj_reduction
        mha_forward = qkv_proj_forward + attention_forward + out_proj_forward

        ffn1_forward = ffn1_time + ffn1_reduction
        ffn2_forward = ffn2_time + ffn2_reduction
        mlp_forward = ffn1_forward + act_f + ffn2_forward
        if use_moe_layer:
            mlp_forward += router_time_f + dispatch_fwd_time + combine_fwd_time

        layernorm1_forward = residual1_f + layernorm1_f
        layernorm2_forward = residual2_f + layernorm2_f

        transformer_forward = (
            mha_forward
            + mlp_forward
            + layernorm1_forward
            + layernorm1_reduction
            + layernorm2_forward
            + layernorm2_reduction
        )

        node_breakdown = {
            "transformer_time_f": transformer_forward,
            "transformer_time_b": 0.0,
            "linear_softmax_f": transformer_timings["linear_softmax"].total_forward_time(),
            "linear_softmax_b": 0.0,
            "embedding_f": 0.0,
            "embedding_b": 0.0,
        }

        return transformer_timings, node_breakdown

    def _energy_per_byte_levels(self) -> Tuple[float, float, float, float]:
        """Return (L0,L1,L2,L3) energy per byte from tech config."""
        tc = self.hw_config.tech_config
        e_l0 = float(getattr(tc.SRAMR, "dynamic_energy_per_bit", 0.0) or 0.0) * 8.0
        e_l1 = float(getattr(tc.SRAML1, "dynamic_energy_per_bit", 0.0) or 0.0) * 8.0
        e_l2 = float(getattr(tc.SRAML2, "dynamic_energy_per_bit", 0.0) or 0.0) * 8.0
        e_l3 = float(getattr(tc.DRAM, "dynamic_energy_per_bit", 0.0) or 0.0) * 8.0
        return e_l0, e_l1, e_l2, e_l3

    def _network_energy_per_byte(self) -> float:
        """Approximate interconnect energy per byte (use first dimension)."""
        e_bit = float(self.network.energies_per_bit[0]) if self.network.energies_per_bit else 0.0
        return e_bit * 8.0

    def _compute_energy_breakdown(self, transformer_timings: Dict[str, OperationTiming]) -> Dict[str, Any]:
        """Compute per-op and total energy breakdown (compute/memory/communication)."""
        e_flop = float(self.core.nominal_energy_per_flop)
        e_levels = self._energy_per_byte_levels()
        e_comm = self._network_energy_per_byte()

        def _memory_energy(forward: DirectionTiming) -> Tuple[float, Dict[str, float]]:
            # Prefer structured profile; fallback to totals dict
            totals: Tuple[float, float, float, float]
            if forward.memory_profile is not None:
                t = forward.memory_profile.totals()
                totals = (float(t[0]), float(t[1]), float(t[2]), float(t[3]))
            else:
                # ensure 4 levels
                lv = [forward.memory_accesses.get(f"L{i}", 0.0) for i in range(4)]
                totals = (float(lv[0]), float(lv[1]), float(lv[2]), float(lv[3]))
            per_level = [totals[i] * e_levels[i] for i in range(4)]
            mem_total = float(sum(per_level))
            return mem_total, {f"L{i}": float(val) for i, val in enumerate(per_level)}

        by_op: Dict[str, Any] = {}
        total_compute = 0.0
        total_memory = 0.0
        total_comm = 0.0
        for name, op in transformer_timings.items():
            fwd = op.forward
            if fwd is None:
                continue
            compute_e = float(fwd.flops or 0.0) * e_flop
            mem_e, per_level = _memory_energy(fwd)
            comm_e = float(fwd.comm_bytes or 0.0) * e_comm
            total_e = compute_e + mem_e + comm_e
            by_op[name] = {
                "compute_j": compute_e,
                "memory_j": mem_e,
                "memory_levels_j": per_level,
                "communication_j": comm_e,
                "total_j": total_e,
            }
            total_compute += compute_e
            total_memory += mem_e
            total_comm += comm_e

        return {
            "ops": by_op,
            "totals": {
                "compute_j": total_compute,
                "memory_j": total_memory,
                "communication_j": total_comm,
                "total_j": total_compute + total_memory + total_comm,
            },
        }


    def prepare_decode_graphs(
        self,
        *,
        batch_size: int,
        total_seq_len: int,
        gemm_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    ):
        moe_layers_active = bool(self.use_moe and any(getattr(self, "moe_layer_mask", []) or []))
        moe_intermediate = self.moe_intermediate_size
        decode_gemm_shapes_moe = gemm_shapes
        if decode_gemm_shapes_moe is None:
            decode_gemm_shapes_moe = llm_util.process_decode_gemm_shapes(
                self,
                batch_size=batch_size,
                current_seq_len=total_seq_len,
                d_model=self.hidden_dim,
                num_heads=self.num_heads,
                kv_heads=self.kv_heads,
                intermediate_size=moe_intermediate if self.use_moe else self.intermediate_size,
                vocab_size=self.vocab_size,
                model_type=self.model_type,
            )
        decode_gemm_shapes_dense = decode_gemm_shapes_moe
        if self.use_moe:
            dense_ctx = SimpleNamespace(
                use_moe=False,
                moe_num_experts=self.moe_num_experts,
                moe_top_k=self.moe_top_k,
                moe_intermediate_size=self.intermediate_size,
                model_type=self.model_type,
                head_dim=getattr(self, "head_dim", None),
            )
            decode_gemm_shapes_dense = llm_util.process_decode_gemm_shapes(
                dense_ctx,
                batch_size=batch_size,
                current_seq_len=total_seq_len,
                d_model=self.hidden_dim,
                num_heads=self.num_heads,
                kv_heads=self.kv_heads,
                intermediate_size=self.intermediate_size,
                vocab_size=self.vocab_size,
                model_type=self.model_type,
            )

        transformer_timings, node_breakdown = self._build_decode_transformer_results(
            batch_size=batch_size,
            total_seq_len=total_seq_len,
            use_moe_layer=False,
            gemm_shapes=decode_gemm_shapes_dense,
        )
        moe_transformer_timings = None
        moe_node_breakdown = None
        if moe_layers_active:
            moe_transformer_timings, moe_node_breakdown = self._build_decode_transformer_results(
                batch_size=batch_size,
                total_seq_len=total_seq_len,
                use_moe_layer=True,
                gemm_shapes=decode_gemm_shapes_moe,
            )

        output_act_bytes = decode_gemm_shapes_dense["qkv_proj"][0] * decode_gemm_shapes_dense["qkv_proj"][1] * self.precision_bytes
        energy = self.calc_energy(transformer_timings, output_act_bytes)
        self._write_forward_memory_report(transformer_timings, "forward_memory_decode.yaml")
        if moe_transformer_timings:
            self._write_forward_memory_report(moe_transformer_timings, "forward_memory_decode_moe.yaml")

        if self._generate_graphs:
            results_path = os.path.join(self.output_dir, "decode_transformer_results.txt")
            with open(results_path, "w", encoding="utf-8") as results_file:
                json.dump(
                    {
                        "transformer_results": {
                            name: timing.to_dict() for name, timing in transformer_timings.items()
                        },
                        "node_breakdown": node_breakdown,
                    },
                    results_file,
                    indent=2,
                    sort_keys=True,
                )

        return self._prepare_execution_graphs(
            node_breakdown=node_breakdown,
            transformer_timings=transformer_timings,
            moe_node_breakdown=moe_node_breakdown,
            moe_transformer_timings=moe_transformer_timings,
            batch_size=batch_size,
            seq_len=1,
            hidden_dim=self.hidden_dim,
            intermediate_size=self.intermediate_size,
            vocab_size=self.vocab_size,
            include_pipeline_backward=False,
            include_transformer_backward=False,
            gemm_shapes=decode_gemm_shapes_moe if self.use_moe else decode_gemm_shapes_dense,
        ), energy
    
    def calc_energy(self, transformer_timings: Dict[str, OperationTiming], cross_layer_comm) -> float:
        """
        Calculate energy consumption based on transformer results.
        """
        if getattr(self, "use_moe", False) and not getattr(self, "_moe_energy_warning_emitted", False):
            warning = (
                "!!! WARNING: MoE energy estimates are not reliable.\n"
                "!!! WARNING: Router/dispatch/combine (and some MoE-specific comms) are not modeled in energy.\n"
                "!!! WARNING: Treat reported energy numbers as lower bounds for MoE runs."
            )
            print(warning)
            self._moe_energy_warning_emitted = True
        # NOTE: MoE energy estimation is incomplete; router/dispatch/combine are not modeled.
        total_flops = 0.0
        total_hbm_bytes = 0.0
        inter_comm_bytes = 0.0  # data parallelism?

        aggregate_groups = {
            "MLP": ("ffn1", "gelu", "ffn2"),
        }
        solo_ops = ("layernorm1", "layernorm2", "embedding", "linear_softmax", "qkv_proj", "attention", "output_proj")

        for members in aggregate_groups.values():
            for name in members:
                op = transformer_timings.get(name)
                if op is None:
                    continue
                total_flops += op.forward.flops
                total_hbm_bytes += op.forward.memory_accesses.get("L3", 0.0)
                inter_comm_bytes += op.forward.comm_bytes

        for name in solo_ops:
            op = transformer_timings.get(name)
            if op is None:
                continue
            total_flops += op.forward.flops
            total_hbm_bytes += op.forward.memory_accesses.get("L3", 0.0)
            inter_comm_bytes += op.forward.comm_bytes

        total_comm_bytes = self.num_layers * inter_comm_bytes + (self.pp - 1) * cross_layer_comm

        energy_per_flop = self.core.nominal_energy_per_flop
        energy_hbm_byte = self.DRAM.dynamic_energy_per_bit * 8
        # TODO: honor per-dimension interconnect energies; currently assumes all comms use dimension 0.
        energy_comm_byte = (self.network.energies_per_bit[0] if self.network.energies_per_bit else 0.0) * 8

        total_energy = (total_flops * energy_per_flop) + \
            (total_hbm_bytes * energy_hbm_byte) + \
                (total_comm_bytes * energy_comm_byte)   
        
        return total_energy

    def _write_forward_memory_report(self, transformer_timings: Dict[str, OperationTiming], filename: str) -> None:
        """
        Emit a YAML artifact summarizing read/write bytes per op for the forward pass.
        """
        if not transformer_timings:
            return
        def _flash_attention_profile(op_name: str) -> Optional[AccessBytes]:
            if op_name != "attention":
                return None
            # Approximate DRAM traffic for standard attention (non-fused score/out tiles).
            # Reads: Q, K, V, attention scores; Writes: attention scores, attention output.
            batch = self._effective_transformer_batch()
            seq_len = self.seq_len
            # crude heuristic: use decode_len to decide whether we're in decode or prefill reports
            if "decode" in filename:
                seq_len = 1  # per-token decode op shapes
            else:
                seq_len = max(1, self.seq_len - self.model.decode_len)
            num_heads = self.num_heads
            kv_heads = self.kv_heads or num_heads
            head_dim = self.hidden_dim // num_heads
            q_size = num_heads * head_dim
            kv_size = kv_heads * head_dim
            bytes_per = self.precision_bytes
            q_bytes = batch * seq_len * q_size * bytes_per
            k_bytes = batch * seq_len * kv_size * bytes_per
            v_bytes = batch * seq_len * kv_size * bytes_per
            score_bytes = batch * kv_heads * seq_len * seq_len * bytes_per
            out_bytes = batch * seq_len * q_size * bytes_per
            dram_reads = q_bytes + k_bytes + v_bytes + score_bytes  # read scores once
            dram_writes = score_bytes + out_bytes
            # Add KV-cache fetch/store during decode: fetch prior K/V, store new K/V.
            if "decode" in filename:
                prefill_len = max(0, self.seq_len - self.model.decode_len)
                kv_fetch = batch * kv_heads * head_dim * bytes_per * 2 * prefill_len
                kv_store = batch * kv_heads * head_dim * bytes_per * 2  # new token
                dram_reads += kv_fetch
                dram_writes += kv_store
            else:
                # Prefill: write K/V for each token of length seq_len
                kv_store = batch * kv_heads * head_dim * bytes_per * 2 * seq_len
                dram_writes += kv_store
            return AccessBytes(
                reads=(0.0, 0.0, 0.0, float(dram_reads)),
                writes=(0.0, 0.0, 0.0, float(dram_writes)),
            )
        def _ffn1_double_profile(op_name: str, seq_len_eff: int) -> Optional[AccessBytes]:
            if op_name != "ffn1":
                return None
            # Build memory profile for a single FFN1 GEMM with intermediate_size,
            # then double it to represent gate + up projections separately.
            try:
                _, _, _, _, mem_accesses, mem_profile = self.get_gemm_time(
                    seq_len_eff,
                    self.hidden_dim,
                    self.intermediate_size,
                    name="ffn1_synth",
                    return_profile=True,
                )
            except Exception:
                return None
            if mem_profile is not None:
                reads = tuple(2.0 * r for r in mem_profile.reads)
                writes = tuple(2.0 * w for w in mem_profile.writes)
            elif mem_accesses:
                reads = tuple(2.0 * float(v) for v in mem_accesses)
                writes = (0.0, 0.0, 0.0, 0.0)
            else:
                return None
            return AccessBytes(reads=reads, writes=writes)

        def _qkv_split_profile(op_name: str, seq_len_eff: int) -> Optional[AccessBytes]:
            if op_name != "qkv_proj":
                return None
            batch = self._effective_transformer_batch()
            num_heads = self.num_heads
            kv_heads = self.kv_heads or num_heads
            head_dim = self.hidden_dim // num_heads
            q_size = num_heads * head_dim
            kv_size = kv_heads * head_dim
            try:
                # Q projection
                _, _, _, _, q_mem_accesses, q_profile = self.get_gemm_time(
                    seq_len_eff,
                    self.hidden_dim,
                    q_size,
                    name="q_proj_synth",
                    return_profile=True,
                )
                # K projection
                _, _, _, _, k_mem_accesses, k_profile = self.get_gemm_time(
                    seq_len_eff,
                    self.hidden_dim,
                    kv_size,
                    name="k_proj_synth",
                    return_profile=True,
                )
                # V projection
                _, _, _, _, v_mem_accesses, v_profile = self.get_gemm_time(
                    seq_len_eff,
                    self.hidden_dim,
                    kv_size,
                    name="v_proj_synth",
                    return_profile=True,
                )
            except Exception:
                return None

            def _reads_writes(profile, fallback_accesses):
                if profile is not None:
                    return profile.reads, profile.writes
                if fallback_accesses:
                    return tuple(float(x) for x in fallback_accesses), (0.0, 0.0, 0.0, 0.0)
                return None, None

            components = []
            for prof, acc in ((q_profile, q_mem_accesses), (k_profile, k_mem_accesses), (v_profile, v_mem_accesses)):
                r, w = _reads_writes(prof, acc)
                if r is None:
                    return None
                components.append((r, w))

            reads = tuple(sum(comp[0][i] for comp in components) for i in range(4))
            writes = tuple(sum(comp[1][i] for comp in components) for i in range(4))
            return AccessBytes(reads=reads, writes=writes)
        report: Dict[str, Any] = {}
        total_bytes = 0
        for name, timing in transformer_timings.items():
            forward = timing.forward
            if forward is None:
                continue
            profile = getattr(forward, "memory_profile", None)
            # Effective sequence length for this report (decode: 1 token, prefill: seq_len - decode_len)
            seq_eff = 1 if "decode" in filename else max(1, self.seq_len - self.model.decode_len)
            synth_ffn1 = _ffn1_double_profile(name, seq_eff)
            synth_qkv = _qkv_split_profile(name, seq_eff)
            if synth_ffn1 is not None:
                profile = synth_ffn1
            if synth_qkv is not None:
                profile = synth_qkv
            # For SwiGLU FFN1 (fused gate+up), approximate as two separate GEMMs
            # by adding an extra read of the input activations (A is read twice).
            if profile is None:
                profile = _flash_attention_profile(name)
            if profile is None:
                # Fallback: build a synthetic profile from memory_accesses if available
                access = getattr(forward, "memory_accesses", {}) or {}
                totals = tuple(float(access.get(f"L{i}", 0.0)) for i in range(4))
                if not any(totals):
                    continue
                reads = totals
                writes = (0.0, 0.0, 0.0, 0.0)
            else:
                totals = profile.totals()
                reads = profile.reads
                writes = profile.writes
            op_total = sum(totals)
            total_bytes += op_total
            reads_map = {f"L{i}": formatBytes(b) for i, b in enumerate(reads)}
            writes_map = {f"L{i}": formatBytes(b) for i, b in enumerate(writes)}
            totals_map = {f"L{i}": formatBytes(b) for i, b in enumerate(totals)}
            report[name] = {
                "reads": reads_map,
                "writes": writes_map,
                "totals": totals_map,
                "total_bytes": int(op_total),
                "total_bytes_hr": formatBytes(int(op_total)),
            }
        report["summary"] = {
            "total_bytes": int(total_bytes),
            "total_bytes_hr": formatBytes(int(total_bytes)),
        }
        if not report["summary"]["total_bytes"]:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as fh:
            yaml.dump(report, fh, sort_keys=True)

    def _combine_profiles(self, *profiles: Optional[AccessBytes]) -> Optional[AccessBytes]:
        combined: Optional[AccessBytes] = None
        for p in profiles:
            if p is None:
                continue
            if combined is None:
                combined = p
                continue
            combined = AccessBytes(
                reads=tuple(cr + pr for cr, pr in zip(combined.reads, p.reads)),
                writes=tuple(cw + pw for cw, pw in zip(combined.writes, p.writes)),
            )
        return combined



    def calc_time(self) -> Tuple[float, float, Optional[Dict[str, Any]]]:
        batch_size = self._effective_transformer_batch()
        vocab_size = self.vocab_size
        hidden_dim = self.hidden_dim
        decode_len = self.model.decode_len
        prefill_len = self.seq_len - decode_len
        num_heads = self.num_heads
        intermediate_size = self.intermediate_size
        kv_heads = self.kv_heads

        total_time = 0.0
        total_energy = 0.0
        prefill_energy_breakdown: Optional[Dict[str, Any]] = None
        prefill_peak_gb = 0.0
        mem_estimator = MemoryEstimator(self)

        if prefill_len <= 0:
            print("Skipping prefill")
            self.pipeline_graph = None
            self.pipeline_root = None
            self.pipeline_interconnect = None
            self.transformer_graph = None
            self.transformer_forward_root = None
            self.transformer_backward_root = None
            self.transformer_graph_moe = None
            self.transformer_forward_root_moe = None
            self.transformer_backward_root_moe = None
            self.transformer_analytical_time_forward = None
            self.transformer_analytical_time_backward = None
            self._last_prefill_timings = None
        else:
            num_SMs = self.hw_config.tech_config.core.num_bundles
            transformer_timings, node_breakdown = self.compute_all_gemm_and_node_times(
                batch_size,
                vocab_size,
                hidden_dim,
                prefill_len,
                num_heads,
                kv_heads,
                intermediate_size,
                num_SMs,
                use_moe_override=False,
                return_profile=True,
            )
            moe_transformer_timings = None
            moe_node_breakdown = None
            if self.use_moe and any(getattr(self, "moe_layer_mask", []) or []):
                moe_transformer_timings, moe_node_breakdown = self.compute_all_gemm_and_node_times(
                    batch_size,
                    vocab_size,
                    hidden_dim,
                    prefill_len,
                    num_heads,
                    kv_heads,
                    self.moe_intermediate_size,
                    num_SMs,
                    use_moe_override=True,
                    return_profile=True,
            )

            output_act_bytes = batch_size * prefill_len * hidden_dim * self.precision_bytes
            total_energy = self.calc_energy(transformer_timings, output_act_bytes)
            prefill_energy_breakdown = self._compute_energy_breakdown(transformer_timings)
            self._write_forward_memory_report(transformer_timings, "forward_memory_prefill.yaml")
            if moe_transformer_timings:
                self._write_forward_memory_report(moe_transformer_timings, "forward_memory_prefill_moe.yaml")

            head_dim = getattr(self, "head_dim", None)
            if head_dim is None:
                head_dim = hidden_dim // num_heads
            token_bytes = llm_util.kv_cache_token_bytes(
                batch_size=batch_size,
                kv_heads=self.kv_heads,
                head_dim=head_dim,
                precision_bytes=self.precision.kv_cache,
            )

            (
                pipeline_graph,
                pipeline_root,
                _,
                _,
                transformer_graph,
                transformer_forward_root,
                transformer_backward_root,
                moe_transformer_graph,
                moe_transformer_forward_root,
                moe_transformer_backward_root,
                interconnect_params,
            ) = self._prepare_execution_graphs(
                node_breakdown=node_breakdown,
                transformer_timings=transformer_timings,
                moe_node_breakdown=moe_node_breakdown,
                moe_transformer_timings=moe_transformer_timings,
                batch_size=batch_size,
                seq_len=prefill_len,
                hidden_dim=hidden_dim,
                intermediate_size=intermediate_size,
                vocab_size=vocab_size,
                include_pipeline_backward=False,
                include_transformer_backward=False,
            )

            self.pipeline_graph = pipeline_graph
            self.pipeline_root = pipeline_root
            self.pipeline_interconnect = interconnect_params
            self.transformer_graph = transformer_graph
            self.transformer_forward_root = transformer_forward_root
            self.transformer_backward_root = None
            self.transformer_graph_moe = moe_transformer_graph
            self.transformer_forward_root_moe = moe_transformer_forward_root
            self.transformer_backward_root_moe = moe_transformer_backward_root
            self.transformer_analytical_time_forward = node_breakdown.get("transformer_time_f")
            self.transformer_analytical_time_backward = None
            # keep a handle for downstream reporting
            self._last_prefill_timings = transformer_timings

            dispatcher = LLMExecutionDispatcher(
                time_calc=self,
                pipeline_graph=self.pipeline_graph,
                pipeline_root=self.pipeline_root,
                interconnect_params=self.pipeline_interconnect,
                transformer_graph=self.transformer_graph,
                transformer_forward_root=self.transformer_forward_root,
                transformer_backward_root=self.transformer_backward_root,
                moe_transformer_graph=self.transformer_graph_moe,
                moe_transformer_forward_root=self.transformer_forward_root_moe,
                moe_transformer_backward_root=self.transformer_backward_root_moe,
            )
            mode = self.execution_mode
            try:
                result = dispatcher.run(mode)
            except NotImplementedError as exc:
                raise NotImplementedError(
                    f"{exc}. Selected execution mode '{mode.value}'."
                ) from exc

            self.pipeline_graph = dispatcher.pipeline_graph
            self.pipeline_root = result.graph_root
            self.pipeline_interconnect = dispatcher.interconnect_params

            total_time = result.total_time

            prefill_memory_data = mem_estimator.build_memory_data(
                mode="inference",
                batch_size=batch_size,
                seq_len=prefill_len,
                kv_cache_tokens=prefill_len,
            )
            prefill_root = dispatcher.build_flattened_root_for_memory()
            _, prefill_peak_gb = mem_estimator.simulate_peak(
                prefill_root,
                prefill_memory_data,
                mode="inference",
                filename="memory_graph_prefill",
            )

        decode_peak_gb = prefill_peak_gb
        if decode_len > 0:
            decode_gemm_shapes = llm_util.process_decode_gemm_shapes(
                self,
                batch_size=batch_size,
                current_seq_len=self.seq_len,
                d_model=hidden_dim,
                num_heads=num_heads,
                kv_heads=kv_heads,
                intermediate_size=self.moe_intermediate_size if self.use_moe else intermediate_size,
                vocab_size=vocab_size,
                model_type=self.model_type,
            )
            (
                decode_pipeline_graph,
                decode_pipeline_root,
                _,
                _,
                decode_transformer_graph,
                decode_transformer_forward_root,
                decode_transformer_backward_root,
                decode_moe_transformer_graph,
                decode_moe_transformer_forward_root,
                decode_moe_transformer_backward_root,
                decode_interconnect_params,
            ), _ = self.prepare_decode_graphs(
                batch_size=batch_size,
                total_seq_len=self.seq_len,
                gemm_shapes=decode_gemm_shapes,
            )
            decode_dispatcher = LLMExecutionDispatcher(
                time_calc=self,
                pipeline_graph=decode_pipeline_graph,
                pipeline_root=decode_pipeline_root,
                interconnect_params=decode_interconnect_params,
                transformer_graph=decode_transformer_graph,
                transformer_forward_root=decode_transformer_forward_root,
                transformer_backward_root=decode_transformer_backward_root,
                moe_transformer_graph=decode_moe_transformer_graph,
                moe_transformer_forward_root=decode_moe_transformer_forward_root,
                moe_transformer_backward_root=decode_moe_transformer_backward_root,
            )
            decode_memory_root = decode_dispatcher.build_flattened_root_for_memory()
            decode_memory_data = mem_estimator.build_memory_data(
                mode="inference",
                batch_size=batch_size,
                seq_len=1,
                gemm_shapes=decode_gemm_shapes,
                kv_cache_tokens=self.seq_len,
            )
            original_pipeline_graph = self.pipeline_graph
            try:
                self.pipeline_graph = decode_pipeline_graph
                _, decode_peak_gb = mem_estimator.simulate_peak(
                    decode_memory_root,
                    decode_memory_data,
                    mode="inference",
                    filename="memory_graph_decode",
                )
            finally:
                self.pipeline_graph = original_pipeline_graph

        max_peak_gb = max(prefill_peak_gb, decode_peak_gb)
        self.memory_peak_gb = max_peak_gb

        hardware_mem_bytes = getattr(self.DRAM, "size", None)
        if hardware_mem_bytes is None and hasattr(self.hw_config, "tech_config"):
            tech_cfg = self.hw_config.tech_config
            if hasattr(tech_cfg, "DRAM"):
                hardware_mem_bytes = getattr(tech_cfg.DRAM, "size", None)

        if hardware_mem_bytes is not None:
            hardware_mem_gib = float(hardware_mem_bytes) / float(1024 ** 3)
            self.memory_capacity_per_device_gb = hardware_mem_gib
            mem_delta = hardware_mem_gib - max_peak_gb
            self.memory_headroom_gb = mem_delta
            self.memory_capacity_exceeded = mem_delta < 0
            self.memory_capacity_violation_gb = abs(mem_delta) if mem_delta < 0 else 0.0

            memory_dir = os.path.join(self.output_dir, "memory-summary")
            os.makedirs(memory_dir, exist_ok=True)
            info_lines = [
                "Simulation mode: inference",
                f"Hardware memory capacity (per gpu): {hardware_mem_gib:.2f} GiB",
                f"Prefill peak memory usage (per gpu): {prefill_peak_gb:.2f} GiB",
                f"Final decode peak memory usage (per gpu): {decode_peak_gb:.2f} GiB",
                f"Max peak memory usage (per gpu): {max_peak_gb:.2f} GiB",
            ]
            if mem_delta < 0:
                info_lines.append(
                    f"[WARN] Peak memory exceeds capacity by {abs(mem_delta):.2f} GiB"
                )
            else:
                info_lines.append(f"Remaining memory headroom: {mem_delta:.2f} GiB")
            info_path = os.path.join(memory_dir, "memory_capacity_comparison.txt")
            with open(info_path, "w", encoding="utf-8") as info_file:
                info_file.write("\n".join(info_lines) + "\n")
        else:
            self.memory_capacity_per_device_gb = None
            self.memory_headroom_gb = None
            self.memory_capacity_exceeded = False
            self.memory_capacity_violation_gb = 0.0

        return total_time, total_energy, prefill_energy_breakdown

    def calc_decode_time(self) -> Tuple[float, List[DecodeSample]]:
        """
        Calculate autoregressive decode phase execution time using sample-based approach.

        Returns:
            float: Total decode phase execution time
        """
        # Get inference sampling configuration
        sample_every = self.model.inference_sample_every
        if sample_every == -1:
            sample_every = 2**31 - 1

        decode_len = self.model.decode_len
        if decode_len == 0:
            print("Skipping decode")
            return 0.0, []

        # Create inference configuration from model parameters
        inference_config = InferenceConfig(
            batch_size=self._effective_transformer_batch(),
            seq_len=self.seq_len - decode_len,
            decode_len=decode_len,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            kv_heads=self.kv_heads,
            intermediate_size=self.intermediate_size,
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            use_moe=self.use_moe,
            num_experts=self.moe_num_experts,
            top_k=self.moe_top_k,
            pp=self.pp,
            tp=self.tp,
            tp_sp=self.tp_sp,
            moe_dp=self.moe_dp,
            sample_every=sample_every,
        )


        # Create inference engine with proper hardware and model configs
        inference_engine = InferenceEngine(
            config=inference_config,
            hw_config=self.hw_config,
            model_config=self._raw_model_config,
            time_calc_cls=TimeCalculationLLMInference,
        )

        # Build decode phase using sample-based approach with real RAPID-LLM integration
        # decode_time, decode_energy, decode_samples = inference_engine._build_decode_graph()
        return inference_engine._build_decode_graph()

    def calc_total_inference_time(self) -> dict:
        """
        Calculate complete inference time including prefill + decode phases.

        Returns:
            dict: Breakdown of inference timing components
        """
        # Calculate prefill time (existing functionality)
        prefill_time, prefill_energy, prefill_breakdown = self.calc_time()

        # Calculate decode time (new functionality)
        decode_time, decode_energy, decode_samples = self.calc_decode_time()
        total_time = prefill_time + decode_time

        time_to_first_token = prefill_time
        if decode_samples:
            time_to_first_token += decode_samples[0].execution_time

        head_dim = getattr(self, "head_dim", None)
        if head_dim is None:
            head_dim = self.hidden_dim // self.num_heads
        token_bytes = llm_util.kv_cache_token_bytes(
            batch_size=self._effective_transformer_batch(),
            kv_heads=self.kv_heads,
            head_dim=head_dim,
            precision_bytes=self.precision.kv_cache,
        )
        prefill_len = self.seq_len - self.model.decode_len
        decode_len = self.model.decode_len
        num_layers = self.num_layers

        prefill_store_bytes = token_bytes * prefill_len * num_layers
        decode_store_bytes = token_bytes * decode_len * num_layers
        decode_fetch_bytes = token_bytes * num_layers * (
            decode_len * (prefill_len + self.seq_len) // 2
        )

        def _to_gib(byte_val: int) -> str:
            gib_val = byte_val / (1024 ** 3)
            if gib_val > 1024:
                tib_val = gib_val / 1024
                return f"{tib_val:.1f} TiB"
            return f"{gib_val:.1f} GiB"

        if decode_samples:
            # do NOT use effective_transformer_batch here
            decode_rates = self._decode_token_rates(decode_samples, decode_len, decode_time, self.batch_size)
        else:
            decode_rates = None

        print(
            f"[prefill] time: {prefill_time:.4f}s, "
            f"[decode] time: {decode_time:.4f}s, "
            f"[total] time: {total_time:.4f}s"
        )
        print(
            f"[kv-cache] prefill_store={_to_gib(prefill_store_bytes)}, "
            f"decode_store={_to_gib(decode_store_bytes)}, "
            f"decode_fetch={_to_gib(decode_fetch_bytes)}"
        )
        
        total_energy = prefill_energy + decode_energy
        print(
            f"[prefill] energy: {convert_prefix(prefill_energy)}J, energy/tok: {convert_prefix(prefill_energy / (self._effective_transformer_batch() * prefill_len))}J",
            f"[decode] energy: {convert_prefix(decode_energy)}J, energy/tok: {convert_prefix(decode_energy / (self._effective_transformer_batch() * decode_len))}J",
            f"[total] energy: {convert_prefix(total_energy)}J, energy/tok: {convert_prefix(total_energy / (self._effective_transformer_batch() * (prefill_len + decode_len)))}J",
        )

        def _sum_memory_levels_from_profile(transformer_profiles: Dict[str, OperationTiming]) -> Dict[str, float]:
            """Aggregate byte accesses per cache level across ops."""
            levels = {f"L{i}": 0.0 for i in range(4)}
            for timing in transformer_profiles.values():
                forward = timing.forward
                if forward is None:
                    continue
                profile = getattr(forward, "memory_profile", None)
                if profile is not None:
                    totals = profile.totals()
                    for i in range(4):
                        levels[f"L{i}"] += float(totals[i])
                else:
                    for i in range(4):
                        levels[f"L{i}"] += float(forward.memory_accesses.get(f"L{i}", 0.0))
            return levels

        # Prefill memory bytes already aggregated over sequence length.
        prefill_mem_levels = _sum_memory_levels_from_profile(
            getattr(self, "_last_prefill_timings", {}) or {}
        )


        # Build energy results YAML with per-op breakdowns (prefill) and decode totals.
        try:
            # For decode per-op, approximate with one-step decode at current seq_len
            decode_gemm_shapes = llm_util.process_decode_gemm_shapes(
                self,
                batch_size=self._effective_transformer_batch(),
                current_seq_len=self.seq_len,
                d_model=self.hidden_dim,
                num_heads=self.num_heads,
                kv_heads=self.kv_heads,
                intermediate_size=self.moe_intermediate_size if self.use_moe else self.intermediate_size,
                vocab_size=self.vocab_size,
                model_type=self.model_type,
            )
            decode_timings_step, _ = self._build_decode_transformer_results(
                batch_size=self._effective_transformer_batch(),
                total_seq_len=self.seq_len,
                use_moe_layer=False,
                gemm_shapes=decode_gemm_shapes,
            )
            decode_breakdown_step = self._compute_energy_breakdown(decode_timings_step)
            decode_mem_levels_step = _sum_memory_levels_from_profile(decode_timings_step)
            # Scale per-op by decode_len to get totals across decode
            def _scale_breakdown(b: Dict[str, Any], factor: float) -> Dict[str, Any]:
                ops_scaled = {}
                for k, v in b["ops"].items():
                    ml = {lvl: val * factor for lvl, val in v["memory_levels_j"].items()}
                    ops_scaled[k] = {
                        "compute_j": v["compute_j"] * factor,
                        "memory_j": v["memory_j"] * factor,
                        "memory_levels_j": ml,
                        "communication_j": v["communication_j"] * factor,
                        "total_j": v["total_j"] * factor,
                    }
                tot = b["totals"]
                totals_scaled = {k: float(tot[k]) * factor for k in tot}
                return {"ops": ops_scaled, "totals": totals_scaled}
            decode_breakdown = _scale_breakdown(decode_breakdown_step, float(decode_len))
        except Exception:
            decode_breakdown = {
                "ops": {},
                "totals": {
                    "compute_j": 0.0,
                    "memory_j": 0.0,
                    "communication_j": 0.0,
                    "total_j": float(decode_energy),
                },
            }
            decode_mem_levels_step = {f"L{i}": 0.0 for i in range(4)}

        energy_yaml = {
            "prefill": prefill_breakdown or {"ops": {}, "totals": {"total_j": float(prefill_energy)}},
            "decode": {
                **decode_breakdown,
                "avg_energy_per_token_j": float(decode_energy) / float(max(1, decode_len)),
            },
            "total": {
                "total_j": float(total_energy),
            },
        }
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, "energy_results.yaml"), "w", encoding="utf-8") as fh:
                yaml.dump(energy_yaml, fh, sort_keys=True)
            # Also emit a CSV-style text for quick inspection
            csv_lines = []
            csv_lines.append("phase,op,compute_j,memory_j,communication_j,total_j")
            for phase in ("prefill", "decode"):
                ops = energy_yaml.get(phase, {}).get("ops", {}) or {}
                totals = energy_yaml.get(phase, {}).get("totals", {}) or {}
                for op_name, vals in ops.items():
                    csv_lines.append(
                        f"{phase},{op_name},{vals.get('compute_j',0.0)},{vals.get('memory_j',0.0)},{vals.get('communication_j',0.0)},{vals.get('total_j',0.0)}"
                    )
                # phase totals row
                if totals:
                    csv_lines.append(
                        f"{phase},_total,{totals.get('compute_j',0.0)},{totals.get('memory_j',0.0)},{totals.get('communication_j',0.0)},{totals.get('total_j',0.0)}"
                    )
            with open(os.path.join(self.output_dir, "energy_results.csv"), "w", encoding="utf-8") as fh_csv:
                fh_csv.write("\n".join(csv_lines) + "\n")
        except Exception:
            pass

        # Memory bytes and memory energy per level for the full pass.
        levels = [f"L{i}" for i in range(4)]

        def _sum_energy_levels(breakdown: Dict[str, Any]) -> Dict[str, float]:
            totals = {lvl: 0.0 for lvl in levels}
            for vals in (breakdown or {}).get("ops", {}).values():
                for lvl, energy in vals.get("memory_levels_j", {}).items():
                    totals[lvl] = totals.get(lvl, 0.0) + float(energy)
            return totals

        decode_mem_levels_total = {lvl: decode_mem_levels_step.get(lvl, 0.0) * float(decode_len) for lvl in levels}
        total_mem_levels = {lvl: prefill_mem_levels.get(lvl, 0.0) + decode_mem_levels_total.get(lvl, 0.0) for lvl in levels}

        prefill_energy_levels = _sum_energy_levels(prefill_breakdown)
        decode_energy_levels = _sum_energy_levels(decode_breakdown)
        total_energy_levels = {lvl: prefill_energy_levels.get(lvl, 0.0) + decode_energy_levels.get(lvl, 0.0) for lvl in levels}

        return {
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_inference_time": total_time,
            "time_to_first_token": time_to_first_token,
            "prefill_energy_j": float(prefill_energy),
            "decode_energy_j": float(decode_energy),
            "total_energy_j": float(total_energy),
            "avg_decode_energy_per_token_j": float(decode_energy) / float(max(1, decode_len)),
            "kv_cache_prefill_store_bytes": prefill_store_bytes,
            "kv_cache_decode_store_bytes": decode_store_bytes,
            "kv_cache_decode_fetch_bytes": decode_fetch_bytes,
            "decode_tokens_per_s": decode_rates,
            "memory_bytes_levels": {
                "prefill": prefill_mem_levels,
                "decode_per_token": decode_mem_levels_step,
                "decode_total": decode_mem_levels_total,
                "total": total_mem_levels,
            },
            "memory_energy_levels": {
                "prefill": prefill_energy_levels,
                "decode": decode_energy_levels,
                "total": total_energy_levels,
            },
        }

    @staticmethod
    def _decode_token_rates(
        samples: List[DecodeSample],
        decode_len: int,
        total_decode_time: float,
        batch_size: int,
    ) -> Dict[str, float]:
        if decode_len <= 0:
            return {}

        def token_time_at(step: int) -> float:
            if not samples:
                return 0.0
            if step <= samples[0].step_id:
                return samples[0].execution_time
            for idx in range(1, len(samples)):
                prev = samples[idx - 1]
                curr = samples[idx]
                if step <= curr.step_id:
                    gap = curr.step_id - prev.step_id
                    if gap <= 0:
                        return curr.execution_time
                    ratio = (step - prev.step_id) / gap
                    return prev.execution_time + ratio * (curr.execution_time - prev.execution_time)
            return samples[-1].execution_time

        def safe_rate(token_time: float) -> float:
            if token_time <= 0.0:
                return 0.0
            return 1.0 / token_time

        last_step = max(decode_len - 1, 0)
        mid_step = decode_len // 2

        start_rate = safe_rate(token_time_at(0))
        mid_rate = safe_rate(token_time_at(mid_step))
        end_rate = safe_rate(token_time_at(last_step))

        overall_rate = 0.0
        if total_decode_time > 0.0:
            overall_rate = decode_len / total_decode_time

        return {
            "start": start_rate,
            "midpoint": mid_rate,
            "end": end_rate,
            "midpoint_step": mid_step,
            "overall": overall_rate,
        }



__all__ = ["TimeCalculationLLMInference"]
