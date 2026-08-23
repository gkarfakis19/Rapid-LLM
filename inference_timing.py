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
from collections import OrderedDict
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Mapping
from train_timing import (
    LLMExecutionDispatcher,
    TimeCalculationLLM,
    GemmType,
)
from memory_estimation import MemoryEstimator
from simulate_inference_graph import DecodeSample, InferenceConfig, InferenceEngine
import llm_util
import json
from timing_model import DirectionTiming, OperationTiming, OperationGroup

def convert_prefix(value: float) -> float:
    """Assign SI unit prefixes to numerical values."""
    if value == 0:
        return "0"
    if value < 0:
        return f"{value:.2f}"
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

        token_bytes = llm_util.attention_kv_cache_token_bytes(
            getattr(self, "attention_type", "mha"),
            batch_size=batch_size,
            kv_heads=self.kv_heads,
            head_dim=head_dim,
            precision_bytes=self.precision.kv_cache,
            kv_lora_rank=getattr(self, "kv_lora_rank", None),
            num_heads=getattr(self, "num_heads", None),
            qk_nope_head_dim=getattr(self, "qk_nope_head_dim", None),
            qk_rope_head_dim=getattr(self, "qk_rope_head_dim", None),
            v_head_dim=getattr(self, "v_head_dim", None),
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
                attention_type=getattr(self, "attention_type", "mha"),
                q_lora_rank=getattr(self, "q_lora_rank", None),
                kv_lora_rank=getattr(self, "kv_lora_rank", None),
                qk_nope_head_dim=getattr(self, "qk_nope_head_dim", None),
                qk_rope_head_dim=getattr(self, "qk_rope_head_dim", None),
                v_head_dim=getattr(self, "v_head_dim", None),
                run_type=str(
                    getattr(
                        self,
                        "run_type",
                        getattr(getattr(self, "model", None), "run_type", "training"),
                    )
                ).lower(),
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
        run_type = str(
            getattr(
                self,
                "run_type",
                getattr(getattr(self, "model", None), "run_type", "training"),
            )
        ).lower()
        mla_decode = str(getattr(self, "attention_type", "mha")).lower() == "mla" and run_type == "inference"


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
        ) -> DirectionTiming:
            bytes_int = int(math.ceil(float(comm_bytes or 0.0)))
            return DirectionTiming(
                compute_time=compute_time,
                comm_time=comm_time,
                comm_bytes=bytes_int,
                flops=flops,
                memory_accesses=dict(memory) if memory else {},
            )

        # QKV projection
        if mla_decode:
            transformer_timings["qkv_proj"] = self._build_mla_qkv_projection_timing(
                gemm_shapes,
                include_backward=False,
                hidden_dim=self.hidden_dim,
                decode=True,
            )
            qkv_op = transformer_timings["qkv_proj"].forward
            qkv_proj_time = qkv_op.compute_time
            qkv_proj_reduction = qkv_op.comm_time
            qkv_proj_size = qkv_op.comm_bytes
            qkv_proj_flops = qkv_op.flops
            qkv_proj_mem = qkv_op.memory_accesses
        else:
            qkv_proj_time, qkv_proj_reduction, qkv_proj_size, qkv_proj_flops, qkv_proj_mem = self.parallelism_gemm_forward(
                gemm_qkv_proj, "decode_qkv_proj_f", gemm_type=GemmType.QKV, decode=True
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
                ),
                backward=None,
            )

        # Attention components
        if mla_decode:
            transformer_timings.update(
                self._build_mla_attention_timings(
                    gemm_shapes,
                    include_backward=False,
                    hidden_dim=self.hidden_dim,
                    decode=True,
                )
            )
            attention_score_op = transformer_timings["attention_score"].forward
            attention_score_time = attention_score_op.compute_time
            attention_score_reduction = attention_score_op.comm_time
            attention_score_size = attention_score_op.comm_bytes
            attention_score_flops = attention_score_op.flops
            attention_score_mem = attention_score_op.memory_accesses
            attention_output_op = transformer_timings["attention_output"].forward
            attention_output_time = attention_output_op.compute_time
            attention_output_reduction = attention_output_op.comm_time
            attention_output_size = attention_output_op.comm_bytes
            attention_output_flops = attention_output_op.flops
            attention_output_mem = attention_output_op.memory_accesses
            attention_scale_softmax_f = transformer_timings["attention_scale_softmax"].forward.compute_time
        else:
            attention_score_time, attention_score_reduction, attention_score_size, attention_score_flops, attention_score_mem = self.parallelism_gemm_forward(
                gemm_attention_score, "decode_attention_score_f", gemm_type=GemmType.ATTENTION_SCORE, decode=True
            )
            attention_output_time, attention_output_reduction, attention_output_size, attention_output_flops, attention_output_mem = self.parallelism_gemm_forward(
                gemm_attention_output, "decode_attention_output_f", gemm_type=GemmType.ATTENTION_OUTPUT, decode=True
            )
            attention_scale_softmax_f = self.get_scale_softmax_f(gemm_attention_score)

            attention_reduction = attention_score_reduction + attention_output_reduction
            attention_comm_bytes = attention_score_size + attention_output_size
            attention_forward_compute = attention_score_time + attention_scale_softmax_f + attention_output_time
            attention_forward_time = attention_forward_compute + attention_reduction
            attention_flops = (attention_score_flops or 0.0) + (attention_output_flops or 0.0)
            attention_mem = self._combine_mem(attention_score_mem, attention_output_mem)

            transformer_timings["attention"] = OperationTiming(
                "attention",
                forward=_make_forward(
                    "attention",
                    compute_time=attention_forward_compute,
                    comm_time=attention_reduction,
                    comm_bytes=attention_comm_bytes,
                    flops=attention_flops,
                    memory=attention_mem,
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

        if mla_decode:
            attention_op = transformer_timings["attention"].forward
            attention_reduction = attention_op.comm_time
            attention_comm_bytes = attention_op.comm_bytes
            attention_forward_compute = attention_op.compute_time
            attention_forward_time = transformer_timings["attention"].total_forward_time()
            attention_flops = attention_op.flops
            attention_mem = attention_op.memory_accesses

        # Output projection
        if mla_decode:
            transformer_timings["output_proj"] = self._build_mla_output_projection_timing(
                gemm_shapes,
                include_backward=False,
                hidden_dim=self.hidden_dim,
            )
            out_proj_op = transformer_timings["output_proj"].forward
            out_proj_time = out_proj_op.compute_time
            out_proj_reduction = out_proj_op.comm_time
            out_proj_size = out_proj_op.comm_bytes
            out_proj_flops = out_proj_op.flops
            out_proj_mem = out_proj_op.memory_accesses
        else:
            out_proj_time, out_proj_reduction, out_proj_size, out_proj_flops, out_proj_mem = self.parallelism_gemm_forward(
                gemm_output_proj, "decode_output_projection_f", gemm_type=GemmType.OUT_PROJ
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
            ffn1_time, ffn1_reduction, ffn1_size, ffn1_flops, ffn1_mem = self.parallelism_gemm_forward(
                gemm_ffn1, "decode_ffn1_f", gemm_type=GemmType.FFN1
            )
            ffn2_time, ffn2_reduction, ffn2_size, ffn2_flops, ffn2_mem = self.parallelism_gemm_forward(
                gemm_ffn2, "decode_ffn2_f", gemm_type=GemmType.FFN2
            )
        else:
            gemm_router = gemm_shapes.get("router")
            if gemm_router is None:
                raise KeyError("Missing decode GEMM shape for 'router'")
            allow_padding = self._moe_allow_padding()
            (
                tokens_owner,
                tokens_dispatched_balanced,
                _tokens_local_balanced,
                experts_per_rank,
                _tokens_per_expert_balanced,
            ) = self._moe_balanced_routed_tokens_per_expert(
                batch_size,
                output_seq_len,
                allow_padding=allow_padding,
            )
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
                math.ceil(self.precision.activations * tokens_dispatched_balanced * self.hidden_dim)
            )
            dispatch_fwd_time = self._moe_comm_time(
                self._moe_comm_decomposition(dispatch_fwd_bytes, experts_per_rank),
                debug_label="decode_moe_dispatch_f",
                axis=axis,
            )
            ffn1_time, ffn1_reduction, ffn1_size, ffn1_flops, ffn1_mem = self.get_moe_ffn_f(
                gemm_ffn1,
                "decode_ffn1_f",
                gemm_type=GemmType.FFN1,
                batch_size=batch_size,
                seq_len=output_seq_len,
                allow_padding=allow_padding,
            )
            ffn2_time, ffn2_reduction, ffn2_size, ffn2_flops, ffn2_mem = self.get_moe_ffn_f(
                gemm_ffn2,
                "decode_ffn2_f",
                gemm_type=GemmType.FFN2,
                batch_size=batch_size,
                seq_len=output_seq_len,
                allow_padding=allow_padding,
            )
            combine_fwd_time = self._moe_comm_time(
                self._moe_comm_decomposition(dispatch_fwd_bytes, experts_per_rank),
                debug_label="decode_moe_combine_f",
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
        linear_softmax_f, linear_softmax_mem = self.get_linear_softmax_f(
            linear_shape, name="decode_linear_softmax_f"
        )
        if getattr(self, "disable_embedding_unembedding", False):
            linear_softmax_f = 0.0
            linear_softmax_mem = {}
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
                attention_type=getattr(self, "attention_type", "mha"),
                q_lora_rank=getattr(self, "q_lora_rank", None),
                kv_lora_rank=getattr(self, "kv_lora_rank", None),
                qk_nope_head_dim=getattr(self, "qk_nope_head_dim", None),
                qk_rope_head_dim=getattr(self, "qk_rope_head_dim", None),
                v_head_dim=getattr(self, "v_head_dim", None),
                run_type=str(
                    getattr(
                        self,
                        "run_type",
                        getattr(getattr(self, "model", None), "run_type", "training"),
                    )
                ).lower(),
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
            if getattr(self, "disable_embedding_unembedding", False) and name in {"embedding", "linear_softmax"}:
                continue
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

    def _write_fws_cim_report(
        self, *, seq_len: int, batch_size: int, sequential_time_s: float
    ) -> None:
        """FWS-CIM spatial pipeline report (consumer of the CimDeviceModel laws).

        Modeling assumptions restated in the output: LN/GELU/adder helper lanes
        are absorbed into their stage (OPTIMA sizing contract, adopted as an
        assumption); energy is PARTIAL (analog + boundary interconnect, plus
        cim_dram KV traffic when that story is configured — no fabric/SRAM
        energy). Chip-boundary transfers are simple p2p over the pp link; MoE
        dispatch/combine are per-MoE-layer boundary transfers over the ep
        link; a transfer longer than the pipeline period warns
        (bandwidth-bound) but does not fail. Chip capacity overflow is a hard
        error; KV capacity overflow is a WARN (the memory-pass precedent).
        The decode section is computed by DIRECT LAW EVALUATION at three
        contexts (prefill+1, midpoint, final) — never plumbed through the
        discarded per-step decode temp instances. Writes
        ``fws_cim_report.json`` into the output directory (output/VIT or
        output/LLM per run mode) and stashes the readable section on
        ``self.fws_cim_report_lines`` for run_perf to append to the results
        txt (which run_perf owns and rewrites after this method runs).
        """
        cim = self.cim_model
        if self.pp != 1:
            raise ValueError(
                "device_class: fws_cim requires parallelism.pp = 1; chip placement "
                f"comes from cim.chip.layers_per_chip (got pp={self.pp})."
            )

        # Capacity is a hard error before any report output.
        cim.validate_capacity()

        # D12: the derived per-macro digital-pool sizing is REPORTED. On stdout
        # only — the report file and the results txt are the frozen closed-form
        # accounting (ADJ-8) and the pool moves no number in them.
        print(cim.report_digital_pool())

        tp = max(1, int(self.tp))
        seq_len = int(seq_len)
        batch_size = int(batch_size)
        num_layers = int(self.num_layers)

        # Layer classes (dense vs MoE) come from the model's MoE mask.
        class_mask = cim.layer_class_mask()
        num_moe_layers = sum(class_mask)
        num_dense_layers = len(class_mask) - num_moe_layers
        is_vit = cim.params.is_vit_shaped
        # Analog stages price the tokens one forward pass pushes through the
        # arrays. The pass-1 ViT convention (M = seq, B folded by nobody at
        # the recorded B=1 points) is kept bit-stable; LLM reports fold the
        # batch: tokens_owner = B * S.
        tokens_owner = None if is_vit else batch_size * seq_len
        tokens_owner_num = seq_len if tokens_owner is None else tokens_owner
        # The LLM wavefront carries B streams, so the attention stage prices
        # the same B streams the analog stages price (streams fold into the
        # SA contraction; ViT keeps the pass-1 one-image wavefront).
        streams = 1 if is_vit else batch_size

        stage_times = (
            cim.layer_stage_times(seq_len, tp, tokens=tokens_owner, streams=streams)
            if num_dense_layers > 0
            else OrderedDict()
        )
        moe_stage_times = (
            cim.moe_layer_stage_times(
                seq_len, tp, tokens_owner=tokens_owner, streams=streams
            )
            if num_moe_layers > 0
            else OrderedDict()
        )
        att = cim.prefill_attention_timing(seq_len=seq_len, tp=tp, streams=streams)
        # Sequential-figure disclosures. Under fabric.model: sa the
        # attention_score and attention_output ops both return the one folded
        # QK+PV fabric run (cim_timing N_mult contract), so the per-op
        # sequential sum counts it twice per layer (gpu_native prices the two
        # ops separately, so no note there). MoE runs additionally serialize
        # the experts in the per-op figure (the caller's per-expert
        # multiplier); the spatial report prices all expert arrays in
        # parallel and is authoritative.
        sequential_notes = []
        if str(cim.fabric.model) == "sa":
            sequential_notes.append(
                "attention_score and attention_output both return the one "
                "folded QK+PV fabric run, so this per-op sum counts it twice "
                "per layer; the spatial pipeline metrics are the "
                "authoritative output"
            )
        if num_moe_layers > 0:
            sequential_notes.append(
                "the per-op sum prices MoE experts serialized (one per-expert "
                "shape times the expert count); the spatial pipeline prices "
                "all expert arrays in parallel and is authoritative"
            )
        sequential_note = "; ".join(sequential_notes) if sequential_notes else None
        endpoint_times = cim.endpoint_stage_times(seq_len, batch_size)
        period_s, bottleneck = cim.pipeline_period(
            seq_len, tp, batch_size, tokens=tokens_owner, streams=streams
        )
        fps = 1.0 / period_s if period_s > 0 else float("inf")
        dense_block_s = sum(stage_times.values())
        moe_block_s = sum(moe_stage_times.values())
        # Pass-1 field semantics: block_latency is the DENSE per-layer block
        # (S1..S5); an all-MoE stack falls back to the MoE block.
        block_latency_s = dense_block_s if num_dense_layers > 0 else moe_block_s

        chip_layers = cim.chip_layer_counts()
        chip_usage = cim.chip_array_usage()
        num_chips = len(chip_layers)
        capacity = int(cim.chip.arrays_per_chip)
        placement_auto = cim.layers_per_chip_is_auto

        # Boundary p2p transfers between consecutive chips over the pp link.
        # The model's own cross-layer term is NOT added on top: it is 0 at the
        # pp == 1 asserted above, and the boundary hops are priced here only.
        boundary_bytes = (
            float(batch_size) * float(seq_len) * float(self.hidden_dim) * float(self.precision_bytes)
        )
        num_boundaries = num_chips - 1
        boundary_time_s = 0.0
        if num_boundaries > 0:
            boundary_time_s = self.network_model._analytical_point_to_point(
                boundary_bytes, *self.links["pp"]
            )
        boundary_times_s = [boundary_time_s] * num_boundaries
        bandwidth_bound = num_boundaries > 0 and boundary_time_s > period_s
        if bandwidth_bound:
            print(
                "[WARNING]: fws_cim: chip-boundary transfer time "
                f"({boundary_time_s * 1e6:.6f} us) exceeds the pipeline period "
                f"({period_s * 1e6:.6f} us); the pipeline is bandwidth-bound."
            )

        # MoE dispatch/combine: per-MoE-layer boundary transfers over the ep
        # link (balanced-A2A sizing; moe_expert_parallel divides the time).
        # One name drives both the link lookup and every report label, so the
        # reported link can never drift from the link actually priced.
        dispatch_link_name = "ep"
        ep_link = self.links[dispatch_link_name]
        dispatch_bytes = 0.0
        dispatch_time_s = 0.0
        dispatch_bound = False
        expert_pool_chips, expert_pool_arrays = cim.moe_expert_pool()
        if num_moe_layers > 0:
            dispatch_bytes = cim.moe_dispatch_bytes(tokens_owner_num, self.precision_bytes)
            dispatch_time_s = cim.moe_dispatch_time(
                tokens_owner_num,
                self.precision_bytes,
                ep_link.bandwidth,
                ep_link.latency,
            )
            dispatch_bound = dispatch_time_s > period_s
            if dispatch_bound:
                print(
                    "[WARNING]: fws_cim: MoE dispatch/combine transfer time "
                    f"({dispatch_time_s * 1e6:.6f} us) exceeds the pipeline period "
                    f"({period_s * 1e6:.6f} us); the pipeline is bandwidth-bound "
                    f"on the {dispatch_link_name} link."
                )

        end_to_end_s = (
            num_dense_layers * dense_block_s
            + num_moe_layers * moe_block_s
            + sum(endpoint_times.values())
            + sum(boundary_times_s)
            + num_moe_layers * 2.0 * dispatch_time_s  # dispatch + combine
        )

        stage_arrays = (
            cim.per_layer_stage_arrays() if num_dense_layers > 0 else {}
        )
        stage_arrays_by_stage = {
            "S1_qkv": stage_arrays.get("qkv", 0),
            "S2_attention": 0,  # digital fabric; no analog arrays
            "S3_o_proj": stage_arrays.get("o_proj", 0),
            "S4_ffn1": stage_arrays.get("ffn1", 0),
            "S5_ffn2": stage_arrays.get("ffn2", 0),
        }
        moe_stage_arrays = (
            cim.moe_layer_stage_arrays() if num_moe_layers > 0 else OrderedDict()
        )
        moe_stage_arrays_by_stage = {
            "S1_qkv": moe_stage_arrays.get("qkv", 0),
            "S2_attention": 0,  # digital fabric; no analog arrays
            "S3_o_proj": moe_stage_arrays.get("o_proj", 0),
            "S4_router": moe_stage_arrays.get("router", 0),
            "S5_ffn1_moe": moe_stage_arrays.get("ffn1_routed", 0)
            + moe_stage_arrays.get("ffn1_shared", 0),
            "S6_ffn2_moe": moe_stage_arrays.get("ffn2_routed", 0)
            + moe_stage_arrays.get("ffn2_shared", 0),
        }
        endpoint_arrays = cim.endpoint_arrays()
        endpoint_chip = {
            "patch_embed": 0,
            "vit_head": num_chips - 1,
            "lm_head": num_chips - 1,
        }
        endpoint_m = {
            "patch_embed": f"M=seq={seq_len}",
            "vit_head": f"M=B={batch_size}",
            "lm_head": f"M=B*S={batch_size * seq_len}",
        }
        endpoint_m_tokens = {
            "patch_embed": seq_len,
            "vit_head": batch_size,
            "lm_head": batch_size * seq_len,
        }

        # --- KV section (LLM stories; ViT has no KV cache) -------------------
        kv_story = str(
            getattr(
                getattr(self.hw_config, "inference_config", None), "kvcache_type", "hbm_only"
            )
        ).strip().lower()
        decode_len = int(getattr(self.model, "decode_len", 0) or 0)
        final_context = int(self.seq_len)
        prefill_len = final_context - decode_len
        kv_precision = float(self.precision.kv_cache)
        kv_disabled = bool(getattr(self, "disable_kv_cache", False))
        kv_section = None
        kv_bw = None
        if not is_vit and not kv_disabled and kv_story in ("cim_sram", "cim_dram"):
            tech_dram = self.hw_config.tech_config.DRAM
            sram_size = float(getattr(tech_dram, "size", 0.0) or 0.0)
            sram_bw = float(getattr(tech_dram, "bandwidth", 0.0) or 0.0)
            kv_bw = cim.kv_story_bandwidth(kv_story, sram_bw)
            kv_capacity = cim.kv_story_capacity(kv_story, sram_size)
            kv_per_stream = cim.kv_bytes_per_stream(final_context, kv_precision, tp)
            kv_total = float(batch_size) * kv_per_stream
            kv_fits = kv_total <= kv_capacity
            if not kv_fits:
                # WARN precedent (memory pass): report, never raise.
                print(
                    "[WARNING]: fws_cim: KV cache at the final context does not "
                    f"fit the {kv_story} capacity: {kv_total / 1024 ** 3:.2f} GiB "
                    f"({batch_size} streams x {kv_per_stream / 1024 ** 3:.2f} GiB "
                    f"at context {final_context}) vs {kv_capacity / 1024 ** 3:.2f} GiB."
                )
            kv_section = {
                "story": kv_story,
                "final_context": final_context,
                "kv_precision_bytes": kv_precision,
                "bandwidth_bytes_per_s": kv_bw,
                "capacity_bytes": kv_capacity,
                "bytes_per_stream": kv_per_stream,
                "total_bytes": kv_total,
                "max_streams": cim.kv_max_streams(
                    kv_capacity, final_context, kv_precision, tp
                ),
                "max_context_at_batch": cim.kv_max_context(
                    kv_capacity, kv_precision, batch_size, tp
                ),
                "fits": kv_fits,
            }
            # calc_time's memory-capacity text surfaces this side check.
            self._fws_kv_summary = dict(kv_section)

        # --- Decode section: DIRECT LAW EVALUATION at three contexts ---------
        decode_section = None
        if decode_len > 0 and kv_section is not None:
            mid_context = prefill_len + max(1, (decode_len + 1) // 2)
            decode_boundary_bytes = (
                float(batch_size) * float(self.hidden_dim) * float(self.precision_bytes)
            )
            decode_boundary_s = 0.0
            if num_boundaries > 0:
                decode_boundary_s = self.network_model._analytical_point_to_point(
                    decode_boundary_bytes, *self.links["pp"]
                )
            decode_dispatch_s = 0.0
            if num_moe_layers > 0:
                decode_dispatch_s = cim.moe_dispatch_time(
                    batch_size, self.precision_bytes, ep_link.bandwidth, ep_link.latency
                )
            endpoint_decode = cim.endpoint_stage_times(batch_size=batch_size, decode=True)
            entries = []
            for label, ctx in (
                ("first", prefill_len + 1),
                ("midpoint", mid_context),
                ("final", final_context),
            ):
                stages_d = cim.decode_all_stage_times(
                    ctx, kv_bw, kv_precision, batch_size, tp
                )
                period_d, bottleneck_d = cim.decode_pipeline_period(
                    ctx, kv_bw, kv_precision, batch_size, tp
                )
                s2 = cim.decode_s2_timing(
                    ctx, kv_bw, kv_precision, batch_size=batch_size, tp=tp
                )
                dense_stages_d = (
                    cim.decode_layer_stage_times(ctx, kv_bw, kv_precision, batch_size, tp)
                    if num_dense_layers > 0
                    else OrderedDict()
                )
                moe_stages_d = (
                    cim.decode_moe_layer_stage_times(
                        ctx, kv_bw, kv_precision, batch_size, tp
                    )
                    if num_moe_layers > 0
                    else OrderedDict()
                )
                # Single-stream step latency: every layer's stages in
                # sequence plus endpoints, chip boundaries, and MoE
                # dispatch/combine at decode token counts.
                step_latency_s = (
                    num_dense_layers * sum(dense_stages_d.values())
                    + num_moe_layers * sum(moe_stages_d.values())
                    + sum(endpoint_decode.values())
                    + num_boundaries * decode_boundary_s
                    + num_moe_layers * 2.0 * decode_dispatch_s
                )
                entries.append(
                    {
                        "label": label,
                        "context": ctx,
                        "stages_us": {k: v * 1e6 for k, v in stages_d.items()},
                        "period_us": period_d * 1e6,
                        "bottleneck_stage": bottleneck_d,
                        "s2": {
                            "sa_time_us": s2.attention.sa_time_s * 1e6,
                            "softmax_time_us": s2.attention.softmax_time_s * 1e6,
                            "kv_read_bytes": s2.kv_read_bytes,
                            "kv_read_time_us": s2.kv_read_time_s * 1e6,
                            "bound": s2.bound,
                            "qk_cycles": s2.attention.qk_cycles,
                            "pv_cycles": s2.attention.pv_cycles,
                            "total_cycles": s2.attention.total_cycles,
                        },
                        "step_latency_us": step_latency_s * 1e6,
                    }
                )
            period_final_s = entries[-1]["period_us"] * 1e-6
            step_final_s = entries[-1]["step_latency_us"] * 1e-6
            # Reconcile the fabric ceiling (B / period, full pipeline
            # occupancy) with the KV stream cap at the final context
            # (CimDeviceModel.decode_sustained_throughput — the law).
            sustained = None
            if period_final_s > 0 and step_final_s > 0:
                sustained = cim.decode_sustained_throughput(
                    step_final_s,
                    period_final_s,
                    int(kv_section["max_streams"]),
                    batch_size,
                    # Aggregate-rate caps: the bottleneck stage passes one
                    # wavefront of B per period, and every resident
                    # wavefront's kv reads draw on the ONE declared KV tier
                    # (per-device bytes: kv_bytes_per_stream at the final
                    # context).
                    kv_read_bandwidth_bytes_per_s=kv_bw,
                    kv_bytes_per_token=float(kv_section["bytes_per_stream"]),
                )
            decode_section = {
                "decode_len": decode_len,
                "prefill_len": prefill_len,
                "batch_size": batch_size,
                "kv_story": kv_story,
                "contexts": entries,
                "aggregate_tokens_per_s_final": (
                    batch_size / period_final_s if period_final_s > 0 else float("inf")
                ),
                "aggregate_note": (
                    "aggregate_tokens_per_s_final is the FABRIC CEILING "
                    "B / period(final): it assumes full spatial-pipeline "
                    "occupancy (>= wavefronts_full wavefronts of B streams "
                    "in flight); sustained_tokens_per_s caps the wavefronts "
                    "by the KV stream capacity."
                ),
                "sustained_tokens_per_s": (
                    None if sustained is None else sustained.tokens_per_s
                ),
                "wavefronts_full": (
                    None if sustained is None else sustained.wavefronts_full
                ),
                "wavefronts_kv": (
                    None if sustained is None else sustained.wavefronts_kv
                ),
                "decode_throughput_limit": (
                    None if sustained is None else sustained.limiting_factor
                ),
                "integration_note": (
                    "per-step decode times are stepwise in context (systolic "
                    "tile quantization), so the integrated decode total's "
                    "trapezoid rule between samples is an approximation; set "
                    "inference_param.sample_every small for paper runs."
                ),
            }

        area_per_array = float(cim.analog.area_mm2_per_array)
        area = None
        if area_per_array > 0:
            area = {
                "transformer_stack": cim.stack_area_mm2(),
                "total": cim.total_area_mm2(),
                "per_array": area_per_array,
            }

        analog_stack_pj = cim.transformer_stack_energy_pj(tokens_owner_num)
        endpoint_energy_pj = cim.endpoint_energy_pj(seq_len, batch_size)
        analog_endpoints_pj = sum(endpoint_energy_pj.values())
        pp_dim = None
        layout = getattr(self.hw_config, "network_layout", None)
        if layout is not None:
            pp_dim = layout.dimension_for_parallelism("pp")
        pp_energy_per_bit_j = (
            float(getattr(pp_dim, "energy_per_bit", 0.0) or 0.0) if pp_dim is not None else 0.0
        )
        interconnect_pj = boundary_bytes * 8.0 * pp_energy_per_bit_j * num_boundaries * 1e12
        # MoE dispatch/combine are boundary transfers too (DESIGN2 1.3/1.5):
        # price their bytes on the ep link's energy_per_bit. The bytes are
        # link-count-invariant: moe_expert_parallel k splits the transfer
        # over k parallel links (dividing TIME by k) but moves the same
        # total bytes.
        dispatch_energy_pj = 0.0
        if num_moe_layers > 0 and dispatch_bytes > 0:
            ep_dim = (
                layout.dimension_for_parallelism(dispatch_link_name)
                if layout is not None
                else None
            )
            ep_energy_per_bit_j = (
                float(getattr(ep_dim, "energy_per_bit", 0.0) or 0.0)
                if ep_dim is not None
                else 0.0
            )
            dispatch_energy_pj = (
                dispatch_bytes * 8.0 * ep_energy_per_bit_j * num_moe_layers * 2.0 * 1e12
            )
        interconnect_pj += dispatch_energy_pj
        # cim_dram KV traffic energy (writes for every cached token + decode
        # reads over the growing context); zero for cim_sram / ViT.
        kv_dram_pj = 0.0
        if kv_section is not None and kv_story == "cim_dram":
            kv_write_bytes = kv_section["total_bytes"]
            per_token_layer = cim.kv_bytes_per_stream_layer(1, kv_precision, tp)
            decode_ctx_sum = 0.0
            if decode_len > 0:
                decode_ctx_sum = (
                    float(decode_len) * float(prefill_len)
                    + float(decode_len) * (decode_len + 1) / 2.0
                )
            kv_read_bytes_total = (
                float(batch_size) * num_layers * per_token_layer * decode_ctx_sum
            )
            kv_dram_pj = cim.kv_dram_energy_pj(kv_write_bytes + kv_read_bytes_total)
        total_partial_pj = analog_stack_pj + analog_endpoints_pj + interconnect_pj + kv_dram_pj
        # Scope disclosure: on decode runs the analog/endpoint/interconnect
        # terms are priced for the prefill pass only, while the cim_dram KV
        # term covers prefill writes plus every decode step's reads — say so
        # rather than mixing step ranges silently.
        energy_scope = (
            "analog arrays + boundary interconnect (+ cim_dram KV traffic) "
            "only; no fabric/SRAM energy"
        )
        if decode_len > 0 and kv_section is not None:
            energy_scope += "; analog + interconnect cover the prefill pass only"
            if kv_dram_pj > 0:
                energy_scope += (
                    ", while kv_dram covers prefill KV writes + every decode "
                    "step's KV reads"
                )

        chips = []
        layer_start = 0
        for idx, (count, used) in enumerate(zip(chip_layers, chip_usage)):
            chips.append(
                {
                    "chip": idx,
                    "num_layers": int(count),
                    "layer_range": [layer_start, layer_start + int(count) - 1],
                    "arrays_used": int(used),
                    "arrays_per_chip": capacity if capacity > 0 else None,
                    "occupancy": (float(used) / capacity) if capacity > 0 else None,
                }
            )
            layer_start += int(count)

        def _attention_detail():
            return {
                "qk_cycles": att.qk_cycles,
                "sv_cycles": att.pv_cycles,
                "fabric_total_cycles": att.total_cycles,
                "sa_time_us": att.sa_time_s * 1e6,
                "softmax_cycles": att.softmax_cycles,
                "softmax_time_us": att.softmax_time_s * 1e6,
            }

        stages_json = {}
        for stage_name, stage_time in stage_times.items():
            entry = {
                "time_us": stage_time * 1e6,
                "arrays_per_layer": stage_arrays_by_stage[stage_name],
            }
            if stage_name == "S2_attention":
                entry.update(_attention_detail())
            stages_json[stage_name] = entry
        moe_stages_json = {}
        for stage_name, stage_time in moe_stage_times.items():
            entry = {
                "time_us": stage_time * 1e6,
                "arrays_per_layer": moe_stage_arrays_by_stage[stage_name],
            }
            if stage_name == "S2_attention":
                entry.update(_attention_detail())
            moe_stages_json[stage_name] = entry
        endpoints_json = {
            name: {
                "time_us": t * 1e6,
                "arrays": int(endpoint_arrays[name]),
                "chip": endpoint_chip[name],
                "m_tokens": endpoint_m_tokens[name],
            }
            for name, t in endpoint_times.items()
        }
        moe_json = None
        if num_moe_layers > 0:
            expert_pool_json = None
            if expert_pool_chips > 0:
                expert_pool_json = {
                    "chips": expert_pool_chips,
                    "arrays_used_per_chip": expert_pool_arrays,
                    "arrays_per_chip": capacity if capacity > 0 else None,
                    "occupancy": (
                        float(expert_pool_arrays) / capacity if capacity > 0 else None
                    ),
                }
            moe_json = {
                "num_moe_layers": num_moe_layers,
                "num_experts": cim.params.num_experts,
                "top_k": cim.params.top_k,
                "n_shared_experts": cim.params.n_shared_experts,
                "expert_imbalance_factor": cim.params.expert_imbalance_factor,
                "moe_intermediate_size": cim.params.moe_intermediate,
                "arrays_per_moe_layer": cim.arrays_per_moe_layer(),
                "stage_arrays": {k: int(v) for k, v in moe_stage_arrays.items()},
                "tokens_hot": cim.moe_tokens_hot(tokens_owner_num),
                "expert_parallel": cim.moe_expert_parallel,
                "dispatch": {
                    "bytes_each_way": dispatch_bytes,
                    "time_us_each_way": dispatch_time_s * 1e6,
                    "count": num_moe_layers * 2,  # dispatch + combine per layer
                    "link": dispatch_link_name,
                    "bandwidth_bound": dispatch_bound,
                },
                "expert_pool": expert_pool_json,
                "block_latency_us": moe_block_s * 1e6,
            }

        report = {
            "device_class": "fws_cim",
            "fabric_model": str(cim.fabric.model),
            "seq_len": seq_len,
            "batch_size": batch_size,
            "tp": tp,
            "num_layers": num_layers,
            "period_us": period_s * 1e6,
            "bottleneck_stage": bottleneck,
            "fps": fps,
            "block_latency_us": block_latency_s * 1e6,
            "end_to_end_latency_us": end_to_end_s * 1e6,
            "sequential_latency_us": sequential_time_s * 1e6,
            "sequential_latency_note": sequential_note,
            "qk_cycles": att.qk_cycles,
            "sv_cycles": att.pv_cycles,
            "layer_classes": {"dense": num_dense_layers, "moe": num_moe_layers},
            "tokens_owner": tokens_owner_num,
            "stages": stages_json,
            "moe_stages": moe_stages_json,
            "moe": moe_json,
            "endpoint_stages": endpoints_json,
            "embedding_note": cim.embedding_note,
            "arrays_per_layer": cim.arrays_per_layer(),
            "arrays_per_moe_layer": (
                cim.arrays_per_moe_layer() if num_moe_layers > 0 else None
            ),
            "arrays_transformer_stack": cim.transformer_stack_arrays(),
            "arrays_total": cim.total_arrays(),
            "area_mm2": area,
            "placement": "auto" if placement_auto else "explicit",
            "derived_layers_per_chip": (
                [int(c) for c in chip_layers] if placement_auto else None
            ),
            "chips": chips,
            "boundary": {
                "bytes_per_boundary": boundary_bytes,
                "count": num_boundaries,
                "times_us": [t * 1e6 for t in boundary_times_s],
                "bandwidth_bound": bandwidth_bound,
            },
            "kv": kv_section,
            "decode": decode_section,
            "energy_partial_pj": {
                "analog_stack": analog_stack_pj,
                "analog_endpoints": analog_endpoints_pj,
                "interconnect": interconnect_pj,
                "interconnect_moe_dispatch": dispatch_energy_pj,
                "kv_dram_traffic": kv_dram_pj,
                "total": total_partial_pj,
                "scope": energy_scope,
            },
            "assumptions": [
                "LN/GELU/adder helper lanes are absorbed into their stage (OPTIMA sizing contract).",
                "QK and PV runs occupy the fabric arrays concurrently (folded-K attention).",
                "shots_per_output affects energy only, never time.",
            ],
        }
        if tp >= 2:
            # tp >= 2 runs a system of tp shard devices (the timing laws
            # shard kv heads / KV bytes per device); the census does not
            # shard weight matrices, so each shard is counted at the full
            # per-device figure — a conservative upper bound. System totals
            # multiply by tp; the per-shard fields above stay unchanged.
            report["tp_shards"] = tp
            report["system_chips"] = tp * (num_chips + expert_pool_chips)
            report["system_area_mm2"] = None if area is None else tp * area["total"]
        report_path = os.path.join(self.output_dir, "fws_cim_report.json")
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2)

        sequential_note_lines = []
        if str(cim.fabric.model) == "sa":
            sequential_note_lines.append(
                "  (counts the one folded QK+PV fabric run once per attention "
                "op - twice per layer; the spatial pipeline is authoritative)"
            )
        if num_moe_layers > 0:
            sequential_note_lines.append(
                "  (prices MoE experts serialized - one per-expert shape times "
                "the expert count; the spatial pipeline prices all expert "
                "arrays in parallel and is authoritative)"
            )
        if num_moe_layers == 0:
            end_to_end_desc = (
                f"({num_layers} layers + endpoints + {num_boundaries} boundary transfers)"
            )
        else:
            end_to_end_desc = (
                f"({num_dense_layers} dense + {num_moe_layers} MoE layers + endpoints + "
                f"{num_boundaries} boundary transfers + dispatch/combine)"
            )
        lines = [
            "",
            "==============================================",
            "FWS-CIM spatial pipeline",
            "==============================================",
            "Weights are stationary in analog arrays; attention runs on the digital "
            f"fabric (cim.fabric.model: {cim.fabric.model}).",
            "LN/GELU/adder helper lanes are absorbed into their stage (OPTIMA sizing "
            "contract, adopted as an assumption).",
            "",
            f"Pipeline period:        {period_s * 1e6:.6f} us  (bottleneck: {bottleneck})",
            # One period completes one B-sized batch: at B > 1 the 1/period
            # figure is batches/s, and per-sequence throughput is B/period
            # (mirrors the decode section's B-folded tok/s).
            (
                f"Throughput:             {fps:.2f} inferences/s"
                if batch_size <= 1
                else (
                    f"Throughput:             {fps:.2f} batches/s "
                    f"(x B={batch_size} streams = {fps * batch_size:.2f} sequences/s)"
                )
            ),
            f"Block latency (S1..S5): {block_latency_s * 1e6:.6f} us",
            f"End-to-end latency:     {end_to_end_s * 1e6:.6f} us  " + end_to_end_desc,
            f"Sequential-execution latency (existing per-op sum, for contrast): "
            f"{sequential_time_s * 1e6:.6f} us",
            *sequential_note_lines,
            "",
        ]

        def _stage_line(stage_name, stage_time, arrays_by_stage):
            if stage_name == "S2_attention":
                return (
                    f"  {stage_name:<14}{stage_time * 1e6:>12.6f} us  fabric  "
                    f"qk={att.qk_cycles} sv={att.pv_cycles} "
                    f"total={att.total_cycles} cycles, softmax={att.softmax_cycles} cycles"
                )
            return (
                f"  {stage_name:<14}{stage_time * 1e6:>12.6f} us  analog  "
                f"arrays/layer={arrays_by_stage[stage_name]}"
            )

        if stage_times:
            if num_moe_layers == 0:
                # tokens is the load-bearing analog M (B-folded for LLM runs);
                # both layer-class header branches print it.
                lines.append(
                    f"Per-layer stages (seq_len={seq_len}, tp={tp}, "
                    f"tokens={tokens_owner_num}):"
                )
            else:
                lines.append(
                    f"Per-layer stages (dense x{num_dense_layers}, seq_len={seq_len}, "
                    f"tp={tp}, tokens={tokens_owner_num}):"
                )
            for stage_name, stage_time in stage_times.items():
                lines.append(_stage_line(stage_name, stage_time, stage_arrays_by_stage))
        if moe_stage_times:
            p = cim.params
            lines.append(
                f"MoE layer stages (x{num_moe_layers}, E={p.num_experts}, "
                f"top_k={p.top_k}, shared={p.n_shared_experts}, "
                f"alpha={p.expert_imbalance_factor}, tokens_owner={tokens_owner_num}, "
                f"tokens_hot={cim.moe_tokens_hot(tokens_owner_num)}):"
            )
            for stage_name, stage_time in moe_stage_times.items():
                lines.append(
                    _stage_line(stage_name, stage_time, moe_stage_arrays_by_stage)
                )
        if endpoint_times:
            lines.append("Endpoint stages:")
            for name, t in endpoint_times.items():
                lines.append(
                    f"  {name:<14}{t * 1e6:>12.6f} us  analog  "
                    f"arrays={endpoint_arrays[name]}  (chip {endpoint_chip[name]}, {endpoint_m[name]})"
                )
        if cim.embedding_note:
            lines.append(f"  note: {cim.embedding_note}")
        lines.append("")
        if placement_auto:
            lines.append(
                "Placement (cim.chip.layers_per_chip: auto -> derived "
                f"{[int(c) for c in chip_layers]}):"
            )
        else:
            lines.append("Placement (cim.chip.layers_per_chip):")
        for entry in chips:
            first, last = entry["layer_range"]
            occ = ""
            if entry["occupancy"] is not None:
                occ = f"/{capacity} ({entry['occupancy'] * 100:.1f}% occupancy)"
            lines.append(
                f"  chip {entry['chip']}: layers {first}..{last} ({entry['num_layers']}), "
                f"arrays {entry['arrays_used']}{occ}"
            )
        if num_boundaries > 0:
            lines.append(
                f"Boundary transfers (p2p over the pp link): {num_boundaries} x "
                f"{boundary_time_s * 1e6:.6f} us ({boundary_bytes:.0f} bytes each)"
            )
            if bandwidth_bound:
                lines.append(
                    "  [WARNING] bandwidth-bound: boundary transfer exceeds the pipeline period"
                )
        else:
            lines.append("Boundary transfers: none (single chip)")
        if num_moe_layers > 0:
            k_par = cim.moe_expert_parallel
            lines.append(
                f"MoE dispatch/combine (p2p over the {dispatch_link_name} link): "
                f"{num_moe_layers} layers x 2 x "
                f"{dispatch_time_s * 1e6:.6f} us ({dispatch_bytes:.0f} bytes each way"
                + (f", spread over {k_par} parallel expert chips" if k_par > 1 else "")
                + ")"
            )
            if dispatch_bound:
                lines.append(
                    f"  [WARNING] bandwidth-bound: {dispatch_link_name} "
                    "dispatch/combine exceeds the pipeline period"
                )
            if expert_pool_chips > 0:
                occ = ""
                if capacity > 0:
                    occ = f"/{capacity} ({expert_pool_arrays / capacity * 100:.1f}% occupancy)"
                lines.append(
                    f"MoE expert pool: {expert_pool_chips} dedicated chips, "
                    f"{expert_pool_arrays} routed arrays each{occ}"
                )
        if num_moe_layers == 0:
            lines.append(
                f"Arrays: {cim.arrays_per_layer()} per layer, "
                f"{cim.transformer_stack_arrays()} transformer stack, {cim.total_arrays()} total"
            )
        else:
            lines.append(
                f"Arrays: {cim.arrays_per_layer()} per dense layer, "
                f"{cim.arrays_per_moe_layer()} per MoE layer, "
                f"{cim.transformer_stack_arrays()} transformer stack, {cim.total_arrays()} total"
            )
        if area is not None:
            lines.append(
                f"Area: {area['transformer_stack']:.4f} mm2 transformer stack, "
                f"{area['total']:.4f} mm2 total ({area_per_array} mm2/array)"
            )
        if tp >= 2:
            lines.append(
                f"TP shards: {tp} — placement/arrays/area above are PER SHARD "
                f"device; system total = {tp} shards x "
                f"{num_chips + expert_pool_chips} chips = "
                f"{tp * (num_chips + expert_pool_chips)} chips"
                + (
                    f", {tp * area['total']:.4f} mm2"
                    if area is not None
                    else ""
                )
                + " (the array census does not shard weight matrices — a"
                " conservative upper bound per shard)."
            )
        if kv_section is not None:
            lines.append("")
            lines.append(
                f"KV cache (story: {kv_story}): "
                f"{kv_section['bytes_per_stream'] / 1024 ** 3:.3f} GiB/stream at context "
                f"{final_context}, {kv_section['total_bytes'] / 1024 ** 3:.3f} GiB total "
                f"(B={batch_size}) vs {kv_section['capacity_bytes'] / 1024 ** 3:.3f} GiB capacity"
            )
            lines.append(
                f"  max streams at context {final_context}: {kv_section['max_streams']}; "
                f"max context at B={batch_size}: {kv_section['max_context_at_batch']}; "
                f"fits: {'yes' if kv_section['fits'] else 'NO'}"
            )
            if not kv_section["fits"]:
                lines.append("  [WARNING] KV capacity exceeded (reported, not fatal)")
        if decode_section is not None:
            lines.append("")
            lines.append(
                f"Decode (direct law evaluation; B={batch_size}, "
                f"decode_len={decode_len}, KV story {kv_story}):"
            )
            for entry in decode_section["contexts"]:
                s2 = entry["s2"]
                lines.append(
                    f"  {entry['label']:<9} ctx={entry['context']:<8} "
                    f"period={entry['period_us']:.6f} us "
                    f"(bottleneck: {entry['bottleneck_stage']}), "
                    f"step latency={entry['step_latency_us']:.6f} us, "
                    f"S2 bound={s2['bound']} (sa={s2['sa_time_us']:.6f} us, "
                    f"softmax={s2['softmax_time_us']:.6f} us, "
                    f"kv_read={s2['kv_read_time_us']:.6f} us)"
                )
            w_full = decode_section["wavefronts_full"]
            w_kv = decode_section["wavefronts_kv"]
            limit = decode_section["decode_throughput_limit"]
            if w_full is None:
                lines.append(
                    "  aggregate decode throughput at final context: "
                    f"{decode_section['aggregate_tokens_per_s_final']:.2f} tok/s (B / period)"
                )
            else:
                lines.append(
                    "  aggregate decode throughput at final context (fabric "
                    f"ceiling, needs >= {w_full * batch_size} streams in flight): "
                    f"{decode_section['aggregate_tokens_per_s_final']:.2f} tok/s (B / period)"
                )
                if limit == "infeasible":
                    lines.append(
                        "  sustained at KV capacity "
                        f"(max {kv_section['max_streams']} streams < B={batch_size}): "
                        "0.00 tok/s [infeasible: KV holds no full wavefront of B]"
                    )
                else:
                    lines.append(
                        "  sustained at KV capacity "
                        f"(max {kv_section['max_streams']} streams => {w_kv} "
                        f"wavefronts of B={batch_size}): "
                        f"{decode_section['sustained_tokens_per_s']:.2f} tok/s "
                        f"[limited by {limit}]"
                    )
                    if limit == "kv_bandwidth":
                        lines.append(
                            "    (KV tier bandwidth cap: "
                            f"{kv_section['bandwidth_bytes_per_s']:.6g} B/s tier / "
                            f"{kv_section['bytes_per_stream']:.0f} B KV read per "
                            "generated token; all resident wavefronts share the "
                            "one KV tier)"
                        )
            lines.append(f"  note: {decode_section['integration_note']}")
        energy_label = (
            "Energy per inference (PARTIAL: analog + boundary interconnect"
            + (" + cim_dram KV traffic" if kv_dram_pj > 0 else "")
            + " only"
        )
        if decode_len > 0 and kv_section is not None:
            energy_label += "; analog covers the prefill pass only"
            if kv_dram_pj > 0:
                energy_label += ", kv_dram covers all decode-step KV reads"
        lines.append(energy_label + "):")
        energy_line = (
            f"  analog stack {analog_stack_pj * 1e-6:.3f} uJ + "
            f"endpoints {analog_endpoints_pj * 1e-6:.3f} uJ + "
            f"interconnect {interconnect_pj * 1e-6:.3f} uJ"
        )
        if dispatch_energy_pj > 0:
            energy_line += (
                f" (incl. MoE dispatch/combine {dispatch_energy_pj * 1e-6:.3f} uJ "
                f"on the {dispatch_link_name} link)"
            )
        if kv_dram_pj > 0:
            energy_line += f" + kv_dram {kv_dram_pj * 1e-6:.3f} uJ"
        energy_line += f" = {total_partial_pj * 1e-6:.3f} uJ"
        lines.append(energy_line)
        self.fws_cim_report_lines = lines

    def calc_time(self) -> Tuple[float, float]:
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
        prefill_peak_gb = 0.0
        mem_estimator = MemoryEstimator(self)

        if prefill_len <= 0:
            print("Skipping prefill")
            self.workload = None
            self.pipeline_interconnect = None
            self.transformer_blocks = None
            self.transformer_analytical_time_forward = None
            self.transformer_analytical_time_backward = None
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
                )

            output_act_bytes = batch_size * prefill_len * hidden_dim * self.precision_bytes
            total_energy = self.calc_energy(transformer_timings, output_act_bytes)

            head_dim = getattr(self, "head_dim", None)
            if head_dim is None:
                head_dim = hidden_dim // num_heads
            token_bytes = llm_util.attention_kv_cache_token_bytes(
                getattr(self, "attention_type", "mha"),
                batch_size=batch_size,
                kv_heads=self.kv_heads,
                head_dim=head_dim,
                precision_bytes=self.precision.kv_cache,
                kv_lora_rank=getattr(self, "kv_lora_rank", None),
                num_heads=getattr(self, "num_heads", None),
                qk_nope_head_dim=getattr(self, "qk_nope_head_dim", None),
                qk_rope_head_dim=getattr(self, "qk_rope_head_dim", None),
                v_head_dim=getattr(self, "v_head_dim", None),
            )
            if getattr(self, "disable_kv_cache", False):
                token_bytes = 0.0

            workload, _ = self._prepare_execution_graphs(
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

            self.workload = workload
            self.pipeline_interconnect = dict(workload.interconnect)
            self.transformer_blocks = workload.blocks
            self.transformer_analytical_time_forward = node_breakdown.get("transformer_time_f")
            self.transformer_analytical_time_backward = None

            dispatcher = LLMExecutionDispatcher(self, workload)
            mode = self.execution_mode
            try:
                result = dispatcher.run(mode)
            except NotImplementedError as exc:
                raise NotImplementedError(
                    f"{exc}. Selected execution mode '{mode.value}'."
                ) from exc

            total_time = result.total_time

            if getattr(self, "cim_model", None) is not None:
                # FWS-CIM spatial pipeline report: computed from the
                # CimDeviceModel stage laws directly (never reconstructed from
                # the per-op sums above). The sequential per-op total stays
                # untouched and is quoted in the report for contrast only.
                self._write_fws_cim_report(
                    seq_len=prefill_len,
                    batch_size=batch_size,
                    sequential_time_s=total_time,
                )

            prefill_memory_data = mem_estimator.build_memory_data(
                mode="inference",
                batch_size=batch_size,
                seq_len=prefill_len,
                kv_cache_tokens=0 if getattr(self, "disable_kv_cache", False) else prefill_len,
            )
            prefill_program = dispatcher.build_flat_program_for_memory()
            _, prefill_peak_gb = mem_estimator.simulate_peak(
                prefill_program,
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
            (decode_workload, _), _ = self.prepare_decode_graphs(
                batch_size=batch_size,
                total_seq_len=self.seq_len,
                gemm_shapes=decode_gemm_shapes,
            )
            decode_dispatcher = LLMExecutionDispatcher(self, decode_workload)
            decode_memory_program = decode_dispatcher.build_flat_program_for_memory()
            decode_memory_data = mem_estimator.build_memory_data(
                mode="inference",
                batch_size=batch_size,
                seq_len=1,
                gemm_shapes=decode_gemm_shapes,
                kv_cache_tokens=self.seq_len,
            )
            _, decode_peak_gb = mem_estimator.simulate_peak(
                decode_memory_program,
                decode_memory_data,
                mode="inference",
                filename="memory_graph_decode",
            )

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
            ]
            if getattr(self, "cim_model", None) is not None:
                # For fws_cim the DRAM stub is the activation buffer and weight
                # bytes are excluded (weights live in the analog arrays), so
                # this capacity comparison is an activation-feasibility check.
                kv_summary = getattr(self, "_fws_kv_summary", None)
                kv_story_label = (kv_summary or {}).get("story")
                if kv_story_label == "cim_sram":
                    info_lines.append(
                        "Device class: fws_cim — weights live in analog arrays and are "
                        "excluded; capacity below is the activation buffer "
                        "(activation-feasibility check; includes the KV cache under "
                        "kvcache_type: cim_sram)."
                    )
                elif kv_story_label == "cim_dram":
                    info_lines.append(
                        "Device class: fws_cim — weights live in analog arrays and are "
                        "excluded; capacity below is the activation buffer "
                        "(activation-feasibility check; the KV cache lives on the "
                        "cim.kv_dram tier and is excluded here)."
                    )
                else:
                    info_lines.append(
                        "Device class: fws_cim — weights live in analog arrays and are "
                        "excluded; capacity below is the activation buffer "
                        "(activation-feasibility check)."
                    )
                if kv_story_label == "cim_dram":
                    # Side capacity check (DESIGN2 1.4): KV at the final
                    # context vs the cim.kv_dram capacity — WARN, never raise.
                    kv_needed_gib = float(kv_summary["total_bytes"]) / float(1024 ** 3)
                    kv_cap_gib = float(kv_summary["capacity_bytes"]) / float(1024 ** 3)
                    info_lines.append(
                        f"KV DRAM tier (kvcache_type: cim_dram): needs "
                        f"{kv_needed_gib:.2f} GiB at context "
                        f"{kv_summary['final_context']} (batch "
                        f"{batch_size}) vs {kv_cap_gib:.2f} GiB capacity"
                    )
                    if kv_summary["fits"]:
                        info_lines.append(
                            f"KV DRAM headroom: {kv_cap_gib - kv_needed_gib:.2f} GiB"
                        )
                    else:
                        info_lines.append(
                            "[WARN] KV DRAM capacity exceeded by "
                            f"{kv_needed_gib - kv_cap_gib:.2f} GiB"
                        )
            info_lines += [
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

        return total_time, total_energy

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
            return 0.0, 0.0, []

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
            use_moe=self.use_moe,
            num_experts=self.moe_num_experts,
            top_k=self.moe_top_k,
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
        prefill_time, prefill_energy = self.calc_time()
        # Calculate decode time (new functionality)
        decode_time, decode_energy, decode_samples = self.calc_decode_time()
        total_time = prefill_time + decode_time

        time_to_first_token = prefill_time
        if decode_samples:
            time_to_first_token += decode_samples[0].execution_time

        head_dim = getattr(self, "head_dim", None)
        if head_dim is None:
            head_dim = self.hidden_dim // self.num_heads
        token_bytes = llm_util.attention_kv_cache_token_bytes(
            getattr(self, "attention_type", "mha"),
            batch_size=self._effective_transformer_batch(),
            kv_heads=self.kv_heads,
            head_dim=head_dim,
            precision_bytes=self.precision.kv_cache,
            kv_lora_rank=getattr(self, "kv_lora_rank", None),
            num_heads=getattr(self, "num_heads", None),
            qk_nope_head_dim=getattr(self, "qk_nope_head_dim", None),
            qk_rope_head_dim=getattr(self, "qk_rope_head_dim", None),
            v_head_dim=getattr(self, "v_head_dim", None),
        )
        if getattr(self, "disable_kv_cache", False):
            token_bytes = 0.0
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
        if not getattr(self, "disable_kv_cache", False):
            print(
                f"[kv-cache] prefill_store={_to_gib(prefill_store_bytes)}, "
                f"decode_store={_to_gib(decode_store_bytes)}, "
                f"decode_fetch={_to_gib(decode_fetch_bytes)}"
            )
        
        total_energy = prefill_energy + decode_energy
        prefill_denom = self._effective_transformer_batch() * prefill_len
        decode_denom = self._effective_transformer_batch() * decode_len
        total_denom = self._effective_transformer_batch() * (prefill_len + decode_len)
        prefill_energy_tok = prefill_energy / prefill_denom if prefill_denom > 0 else 0.0
        decode_energy_tok = decode_energy / decode_denom if decode_denom > 0 else 0.0
        total_energy_tok = total_energy / total_denom if total_denom > 0 else 0.0
        print(
            f"[prefill] energy: {convert_prefix(prefill_energy)}J, energy/tok: {convert_prefix(prefill_energy_tok)}J",
            f"[decode] energy: {convert_prefix(decode_energy)}J, energy/tok: {convert_prefix(decode_energy_tok)}J",
            f"[total] energy: {convert_prefix(total_energy)}J, energy/tok: {convert_prefix(total_energy_tok)}J",
        )


        return {
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_inference_time": total_time,
            "time_to_first_token": time_to_first_token,
            "kv_cache_prefill_store_bytes": prefill_store_bytes,
            "kv_cache_decode_store_bytes": decode_store_bytes,
            "kv_cache_decode_fetch_bytes": decode_fetch_bytes,
            "decode_tokens_per_s": decode_rates,
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
