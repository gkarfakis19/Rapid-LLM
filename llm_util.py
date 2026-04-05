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

import math
import os
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import config

def reshape_gemm_to_3d(arg):
    """
    Reshape a 4-dimensional GEMM [batch_size, M, K, N] into 3 dimensions [M, K, N].

    Parameters:
        arg (list or tuple): A list or tuple containing 4 dimensions [batch_size, M, K, N].

    Returns:
        tuple: A tuple (M, K, N) representing the reshaped GEMM dimensions.
    """
    
    if len(arg) != 4:
        raise ValueError("Input must contain exactly 4 dimensions [batch_size, M, K, N].")
    
    
    batch_size, M, K, N = arg
    if batch_size <= 0:
        raise ValueError("Batch size must be greater than 0.")
    M *= batch_size  # Multiply batch_size into M
        
    return M, K, N


ATTENTION_GEMM_KEYS = {"attention_score", "attention_output"}

_LLAMA_STYLE_MODEL_TYPES = {"llama", "deepseek_v3", "glm4_moe", "glm4", "glm"}
_GLM_MODEL_TYPES = {"glm4_moe", "glm4", "glm"}
_VIT_MODEL_TYPES = {"vit", "vit_dinov3"}


def is_llama_style(model_type) -> bool:
    return str(model_type or "").strip().lower() in _LLAMA_STYLE_MODEL_TYPES


def is_glm_style(model_type) -> bool:
    return str(model_type or "").strip().lower() in _GLM_MODEL_TYPES


def is_vit_style(model_type) -> bool:
    return str(model_type or "").strip().lower() in _VIT_MODEL_TYPES


def uses_gated_mlp(model_type, swiglu_mlp: bool = False) -> bool:
    return (
        is_llama_style(model_type)
        or str(model_type or "").strip().lower() == "vit_dinov3"
        or bool(swiglu_mlp)
    )


def vit_patch_dim(patch_size: Tuple[int, int], in_chans: int) -> int:
    return int(patch_size[0]) * int(patch_size[1]) * int(in_chans)


def vit_output_dim(*, hidden_dim: int, num_classes: int) -> int:
    return int(num_classes) if int(num_classes) > 0 else int(hidden_dim)


def vit_patch_embed_param_count(
    *,
    hidden_dim: int,
    patch_size: Tuple[int, int],
    in_chans: int = 3,
) -> int:
    patch_dim = vit_patch_dim(patch_size, in_chans)
    params = int(patch_dim) * int(hidden_dim)
    params += int(hidden_dim)
    return int(params)


def vit_head_param_count(
    *,
    hidden_dim: int,
    num_classes: int,
) -> int:
    params = 2 * int(hidden_dim)
    if int(num_classes) > 0:
        params += int(hidden_dim) * int(num_classes) + int(num_classes)
    return int(params)


def resolve_head_dim(hidden_dim, num_heads, head_dim=None) -> int:
    if head_dim is None:
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads when head_dim is not provided")
        return hidden_dim // num_heads
    head_dim = int(head_dim)
    if head_dim <= 0:
        raise ValueError("head_dim must be a positive integer")
    return head_dim


def attention_dim_sizes(hidden_dim, num_heads, kv_heads, head_dim=None):
    if kv_heads is None:
        kv_heads = num_heads
    head_dim = resolve_head_dim(hidden_dim, num_heads, head_dim=head_dim)
    q_size = num_heads * head_dim
    kv_size = kv_heads * head_dim
    return head_dim, q_size, kv_size


def mla_dim_sizes(num_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim):
    qk_head_dim = int(qk_nope_head_dim) + int(qk_rope_head_dim)
    q_size = int(num_heads) * qk_head_dim
    v_size = int(num_heads) * int(v_head_dim)
    kv_up_size = int(num_heads) * (int(qk_nope_head_dim) + int(v_head_dim))
    return qk_head_dim, q_size, v_size, kv_up_size


def mla_attention_param_sizes(
    hidden_dim,
    num_heads,
    q_lora_rank,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
):
    groups = mla_attention_param_groups(
        hidden_dim,
        num_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
    )
    qk_head_dim, q_size, v_size, kv_up_size = mla_dim_sizes(
        num_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
    )
    d_proj = groups["d_proj_q"] + groups["d_proj_kv"]
    u_proj_q = groups["u_proj_q_nope"] + groups["u_proj_q_rope"]
    u_proj_kv = groups["u_proj_k_nope"] + groups["u_proj_v"]
    out_proj = groups["out_proj"]
    return {
        "d_proj": d_proj,
        "u_proj_q": u_proj_q,
        "u_proj_kv": u_proj_kv,
        "out_proj": out_proj,
        "qk_head_dim": qk_head_dim,
        "q_size": q_size,
        "v_size": v_size,
        "kv_up_size": kv_up_size,
        "projection_stage": groups["projection_stage"],
        "attention_stage": groups["attention_stage"],
        "output_stage": groups["output_stage"],
        "total": d_proj + u_proj_q + u_proj_kv + out_proj,
    }


def mla_attention_params_per_rank(
    hidden_dim,
    num_heads,
    q_lora_rank,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    *,
    tp=1,
):
    groups = mla_attention_param_groups(
        hidden_dim,
        num_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
    )
    tp = max(1, int(tp))
    sharded = (
        groups["u_proj_q_nope"]
        + groups["u_proj_q_rope"]
        + groups["u_proj_k_nope"]
        + groups["u_proj_v"]
        + groups["out_proj"]
    ) / float(tp)
    replicated = groups["d_proj_q"] + groups["d_proj_kv"]
    return {
        "replicated": replicated,
        "sharded": sharded,
        "total": replicated + sharded,
    }


def mla_attention_param_groups(
    hidden_dim,
    num_heads,
    q_lora_rank,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
):
    """Return exact MLA parameter groups used by the training/inference model."""
    hidden_dim = int(hidden_dim)
    num_heads = int(num_heads)
    q_lora_rank = int(q_lora_rank)
    kv_lora_rank = int(kv_lora_rank)
    qk_nope_head_dim = int(qk_nope_head_dim)
    qk_rope_head_dim = int(qk_rope_head_dim)
    v_head_dim = int(v_head_dim)

    groups = {
        "d_proj_q": hidden_dim * q_lora_rank,
        "d_proj_kv": hidden_dim * (kv_lora_rank + qk_rope_head_dim),
        "u_proj_q_nope": q_lora_rank * num_heads * qk_nope_head_dim,
        "u_proj_q_rope": q_lora_rank * num_heads * qk_rope_head_dim,
        "u_proj_k_nope": kv_lora_rank * num_heads * qk_nope_head_dim,
        "u_proj_v": kv_lora_rank * num_heads * v_head_dim,
        "out_proj": num_heads * v_head_dim * hidden_dim,
    }
    groups["projection_stage"] = (
        groups["d_proj_q"]
        + groups["d_proj_kv"]
        + groups["u_proj_q_nope"]
        + groups["u_proj_q_rope"]
        + groups["u_proj_k_nope"]
        + groups["u_proj_v"]
    )
    groups["attention_stage"] = 0
    groups["output_stage"] = groups["out_proj"]
    groups["total"] = (
        groups["projection_stage"] + groups["attention_stage"] + groups["output_stage"]
    )
    return groups


def mla_kv_cache_token_bytes(
    batch_size,
    qk_rope_head_dim,
    precision_bytes,
    *,
    num_heads,
    qk_nope_head_dim,
    v_head_dim,
):
    """Return per-token MLA KV-cache bytes for the default full-cache decode contract."""
    qk_head_dim = int(qk_nope_head_dim) + int(qk_rope_head_dim)
    return (
        int(batch_size)
        * int(num_heads)
        * (qk_head_dim + int(v_head_dim))
        * float(precision_bytes)
    )


def mla_runtime_proxy_sizes(
    num_heads,
    q_lora_rank,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
):
    """Return proxy stage-output widths for the selected MLA runtime contract."""
    qk_head_dim, q_size, _v_size, kv_up_size = mla_dim_sizes(
        num_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
    )
    qkv_output_size = int(q_lora_rank) + int(kv_lora_rank) + int(qk_rope_head_dim) + int(q_size) + int(kv_up_size)
    attention_output_size = int(v_head_dim)
    output_proj_input_size = int(num_heads) * int(v_head_dim)
    return {
        "qkv_output_size": qkv_output_size,
        "attention_output_size": attention_output_size,
        "output_proj_input_size": output_proj_input_size,
    }


def mla_runtime_op_shapes(
    *,
    batch_size,
    query_seq_len,
    hidden_dim,
    num_heads,
    q_lora_rank,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
):
    """Return MLA component GEMM shapes for the default full-cache runtime path."""
    qk_head_dim = int(qk_nope_head_dim) + int(qk_rope_head_dim)
    return OrderedDict(
        {
            "D_proj_q": (batch_size, query_seq_len, hidden_dim, q_lora_rank),
            "D_proj_kv": (
                batch_size,
                query_seq_len,
                hidden_dim,
                int(kv_lora_rank) + int(qk_rope_head_dim),
            ),
            "U_proj_q": (
                batch_size,
                query_seq_len,
                q_lora_rank,
                int(num_heads) * qk_head_dim,
            ),
            "U_proj_kv": (
                batch_size,
                query_seq_len,
                kv_lora_rank,
                int(num_heads) * (int(qk_nope_head_dim) + int(v_head_dim)),
            ),
        }
    )


def mla_activation_tensor_bytes(
    *,
    batch_size,
    seq_len,
    key_seq_len=None,
    hidden_dim,
    intermediate_size,
    num_heads,
    q_lora_rank,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    precision_bytes,
    model_type="gpt",
    swiglu_mlp=False,
    flash_attention=False,
    full_recomputation=False,
    tp=1,
    cp=1,
):
    """Approximate MLA activation sizing using the selected RAPID MLA runtime tensors.

    The helper intentionally models the *local* per-rank activation footprint:

    - ``seq_len`` is the local query length seen by the rank.
    - ``key_seq_len`` is the visible key length for attention. When omitted, it
      defaults to the full prefill/training context reconstructed from
      ``seq_len * cp``.
    - TP shards head-dependent and low-rank latent tensors. CP shards the query
      token dimension while leaving the visible key length full during
      training/prefill (matching the current CP attention abstraction).
    """
    batch_size = int(batch_size)
    seq_len = int(seq_len)
    tp = max(1, int(tp))
    cp = max(1, int(cp))
    hidden_dim = int(hidden_dim)
    num_heads = int(num_heads)
    q_lora_rank = int(q_lora_rank)
    kv_lora_rank = int(kv_lora_rank)
    qk_nope_head_dim = int(qk_nope_head_dim)
    qk_rope_head_dim = int(qk_rope_head_dim)
    v_head_dim = int(v_head_dim)
    if key_seq_len is None:
        key_seq_len = seq_len * cp
    key_seq_len = int(key_seq_len)
    projected_dim = (
        2 * int(intermediate_size)
        if uses_gated_mlp(model_type, swiglu_mlp=swiglu_mlp)
        else int(intermediate_size)
    )

    local_heads = int(math.ceil(float(num_heads) / float(tp)))
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    hidden_bytes = batch_size * seq_len * hidden_dim * precision_bytes
    q_down_bytes = batch_size * seq_len * q_lora_rank * precision_bytes
    kv_down_bytes = batch_size * seq_len * (kv_lora_rank + qk_rope_head_dim) * precision_bytes
    q_up_bytes = batch_size * seq_len * local_heads * qk_head_dim * precision_bytes
    kv_up_bytes = batch_size * seq_len * local_heads * (qk_nope_head_dim + v_head_dim) * precision_bytes
    qkv_bytes = q_down_bytes + kv_down_bytes + q_up_bytes + kv_up_bytes
    attn_ctx_bytes = batch_size * local_heads * seq_len * v_head_dim * precision_bytes
    attn_score_bytes = 0.0
    if not flash_attention:
        attn_score_bytes = batch_size * local_heads * seq_len * key_seq_len * precision_bytes
    ffn_bytes = (batch_size * seq_len * projected_dim * precision_bytes) / float(max(1, tp))

    # Training stores the hidden-state boundaries plus MLA latent intermediates.
    if full_recomputation:
        training_bytes = (2.0 * hidden_bytes) + qkv_bytes + attn_ctx_bytes
    else:
        training_bytes = (4.0 * hidden_bytes) + qkv_bytes + attn_ctx_bytes + attn_score_bytes + ffn_bytes

    inference_peak_bytes = max(hidden_bytes, qkv_bytes, attn_ctx_bytes, attn_score_bytes, ffn_bytes)
    return {
        "hidden_bytes": float(hidden_bytes),
        "qkv_bytes": float(qkv_bytes),
        "attention_ctx_bytes": float(attn_ctx_bytes),
        "attention_score_bytes": float(attn_score_bytes),
        "ffn_bytes": float(ffn_bytes),
        "training_bytes": float(training_bytes),
        "inference_peak_bytes": float(inference_peak_bytes),
    }


def attention_kv_cache_token_bytes(
    attention_type,
    *,
    batch_size,
    kv_heads,
    head_dim,
    precision_bytes,
    kv_lora_rank=None,
    num_heads=None,
    qk_nope_head_dim=None,
    qk_rope_head_dim=None,
    v_head_dim=None,
):
    if str(attention_type or "").lower() == "mla":
        return mla_kv_cache_token_bytes(
            batch_size=batch_size,
            qk_rope_head_dim=qk_rope_head_dim,
            precision_bytes=precision_bytes,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            v_head_dim=v_head_dim,
        )
    return kv_cache_token_bytes(
        batch_size=batch_size,
        kv_heads=kv_heads,
        head_dim=head_dim,
        precision_bytes=precision_bytes,
    )


def multihead_decoder_gemm(self, batch_size, seq_len, d_model, num_heads, kv_heads, intermediate_size, vocab_size, model_type="gpt"):
    """
    Generate GEMM shapes [M, K, N] for a multi-head Transformer decoder block.

    Parameters:
        batch_size (int): batch size (B)
        seq_len (int): sequence length (S)
        d_model (int): hidden size (D)
        num_heads (int): number of attention heads (H)
        intermediate_size (int): first FFN layer output dimension (typically 4 * D)
        vocab_size (int): vocabulary size (V)
        
        for a standard multi-head attention, kv_heads = num_heads


    """
    assert num_heads % kv_heads == 0, "num_heads must be divisible by kv_heads"
    head_dim, q_size, kv_size = attention_dim_sizes(
        d_model,
        num_heads,
        kv_heads,
        head_dim=getattr(self, "head_dim", None),
    )
    shared_heads = num_heads // kv_heads # how many heads share the same K,V
    gemms = OrderedDict()

    attention_type = str(getattr(self, "attention_type", "mha")).lower()

    if attention_type == "mla":
        qk_head_dim, _q_size_mla, _v_size_mla, _kv_up_size = mla_dim_sizes(
            num_heads,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
        )
        runtime_ops = mla_runtime_op_shapes(
            batch_size=batch_size,
            query_seq_len=seq_len,
            hidden_dim=d_model,
            num_heads=num_heads,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
        )
        gemms.update(runtime_ops)
        proxy_sizes = mla_runtime_proxy_sizes(
            num_heads,
            self.q_lora_rank,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
        )

        gemms["qkv_proj"] = (batch_size, seq_len, d_model, proxy_sizes["qkv_output_size"])
        gemms["attention_score"] = (
            batch_size * num_heads,
            seq_len,
            qk_head_dim,
            seq_len,
        )
        gemms["attention_output"] = (
            batch_size * num_heads,
            seq_len,
            seq_len,
            proxy_sizes["attention_output_size"],
        )
        gemms["output_proj"] = (
            batch_size,
            seq_len,
            proxy_sizes["output_proj_input_size"],
            d_model,
        )
    else:
        gemms["qkv_proj"] = (batch_size, seq_len, d_model, q_size + 2 * kv_size)

        gemms["attention_score"] = (
            batch_size * kv_heads,
            seq_len * shared_heads,
            head_dim,
            seq_len,
        )

        gemms["attention_output"] = (
            batch_size * kv_heads,
            seq_len * shared_heads,
            seq_len,
            head_dim,
        )
        gemms["output_proj"] = (batch_size, seq_len, q_size, d_model)
    if uses_gated_mlp(model_type, swiglu_mlp=getattr(self, "swiglu_mlp", False)):
        projected_dim = 2 * intermediate_size
    else:
        projected_dim = intermediate_size
    if self.use_moe:
        moe_intermediate = int(getattr(self, "moe_intermediate_size", intermediate_size))
        projected_dim = (
            2 * moe_intermediate
            if uses_gated_mlp(model_type, swiglu_mlp=getattr(self, "swiglu_mlp", False))
            else moe_intermediate
        )
        # MoE GEMM shapes use token-owner dimensions; token dispatch is modeled in timing logic.
        gemms["ffn1"] = (batch_size, seq_len, d_model, projected_dim)
        gemms["ffn2"] = (batch_size, seq_len, moe_intermediate, d_model)
    else:
        gemms["ffn1"] = (batch_size, seq_len, d_model, projected_dim)
        gemms["ffn2"] = (batch_size, seq_len, intermediate_size, d_model) 
    gemms["linear"] = (batch_size, seq_len, d_model, vocab_size)
    gemms["router"] = (
        batch_size,
        seq_len,
        d_model,
        getattr(self, "moe_num_experts", getattr(self, "num_experts", 1)),
    )

    return gemms


def process_gemm_shapes(self, batch_size, seq_len, d_model, num_heads, kv_heads, intermediate_size, vocab_size):
    """
    Process GEMM shapes, reshape them into 3d.

    Parameters:
        batch_size (int): Batch size.
        seq_len (int): Sequence length.
        d_model (int): Hidden size.
        num_heads (int): Number of attention heads.
        intermediate_size (int): First FFN layer output dimension.
        vocab_size (int): Vocabulary size.
    """
    # Generate GEMM shapes in 4D
    gemm_shapes_4d = multihead_decoder_gemm(
        self,
        batch_size=batch_size,
        seq_len=seq_len,
        d_model=d_model,
        num_heads=num_heads,
        kv_heads=kv_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        model_type=self.model_type,
    )

    processed = OrderedDict()
    attention_keys = ATTENTION_GEMM_KEYS
    for key, shape in gemm_shapes_4d.items():
        if key in attention_keys:
            processed[key] = tuple(shape)
        else:
            processed[key] = reshape_gemm_to_3d(shape)

    return processed

def get_transformer_mem_layer(
    dp,
    tp,
    cp=1,
    pp=1,
    mb=1,
    batch_size=1,
    hidden_dim=1,
    seq_len=1,
    intermediate_size=1,
    n_heads=1,
    kv_heads=None,
    head_dim=None,
    precision=None,
    zero_stage=0,
    flash_attention=False,
    full_recomputation=False,
    model_type="gpt",
    attention_type="mha",
    q_lora_rank=None,
    kv_lora_rank=None,
    qk_nope_head_dim=None,
    qk_rope_head_dim=None,
    v_head_dim=None,
    run_type="training",
    key_seq_len=None,
    swiglu_mlp=False,
):  #  https://arxiv.org/pdf/2205.05198. https://shjwudp.github.io/blog/2023/gpt-training-memory-estimation-nemo-training-practice/
    """Approximate transformer-layer memory sizing for training/inference."""
    #Activations refer to output activations that need to be stored

    if str(attention_type or "").lower() == "mla":
        mla_activation = mla_activation_tensor_bytes(
            batch_size=batch_size,
            seq_len=seq_len,
            key_seq_len=key_seq_len,
            hidden_dim=hidden_dim,
            intermediate_size=intermediate_size,
            num_heads=n_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            precision_bytes=precision.activations,
            model_type=model_type,
            flash_attention=flash_attention,
            full_recomputation=full_recomputation,
            tp=tp,
            cp=cp,
        )
        act_memory_layer = mla_activation["training_bytes"]
        act_memory_layer_inf = mla_activation["inference_peak_bytes"]
    else:
        if full_recomputation:
            act_memory_layer = seq_len * batch_size * hidden_dim * (2 ) * (precision.activations / 2) #full recompute:
        elif flash_attention:
            act_memory_layer = seq_len * batch_size * hidden_dim * (34 / tp ) * (precision.activations / 2) #assuming selective recompute
        else:
            act_memory_layer = seq_len * batch_size * hidden_dim * (34 / tp + 5 * n_heads * seq_len/(hidden_dim * tp) ) * (precision.activations / 2)

        act_memory_layer_inf = seq_len * batch_size * intermediate_size / tp * precision.activations  #inference max activation memory, no need to store for backpropagation
    ffn_proj_factor = 3 if uses_gated_mlp(model_type, swiglu_mlp=swiglu_mlp) else 2

    if kv_heads is None:
        kv_heads = n_heads
    if str(attention_type or "").lower() == "mla":
        attention_params_per_rank = mla_attention_params_per_rank(
            hidden_dim,
            n_heads,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            tp=tp,
        )["total"]
    else:
        head_dim = resolve_head_dim(hidden_dim, n_heads, head_dim=head_dim)
        q_size = n_heads * head_dim
        kv_size = kv_heads * head_dim
        attention_params_per_rank = ((hidden_dim * (q_size + 2 * kv_size)) + (q_size * hidden_dim)) / tp

    transformer_param_layer = attention_params_per_rank + (
        intermediate_size * ffn_proj_factor * hidden_dim / tp
    )  # weights Wq,Wk,Wv,Wo,ffn per rank

    optimizer_mem = (precision.optimizer_states * 2 + precision.activations) * transformer_param_layer # don't divide by dp for DDP, NeMo ZeRO-1 optimizer style
    tensor_weight_memory_layer = transformer_param_layer * precision.parameters #weight memory
    # master_parameters is set to 0 by default, so this works.
    master_weight_memory_layer = transformer_param_layer * precision.master_parameters
    weight_memory_layer = tensor_weight_memory_layer + master_weight_memory_layer
    
    gradient_mem = transformer_param_layer * precision.gradients  # gradient buffers scaled by precision
    # precision has been replaced with a class that has many different precision types.
    # furthemore, we have added this "master weight" copy for weights that are stored in FP32 optionally.
    # for weight_memory_layer it makes sense to just add them together. But I can see it's only really used in infernece?
    # for training, static_memory_layer needs to be broken up into different equations that use the correct precisions.
    # TODO TODO TODO

    if zero_stage >= 3:
        weight_memory_layer /= dp
    if zero_stage >= 2:
        gradient_mem /= dp
    if zero_stage >= 1:
        optimizer_mem /= dp

    static_memory_layer = optimizer_mem + gradient_mem + weight_memory_layer # optimizer states + gradients + weights
    layer_mem = (act_memory_layer + weight_memory_layer)


    return layer_mem, act_memory_layer, act_memory_layer_inf, static_memory_layer, gradient_mem, optimizer_mem, weight_memory_layer

def get_linear_softmax_mem(batch_size, seq_len, hidden_dim, vocab_size, precision, t):
    # t = 1
    # weights = hidden_dim * vocab_size
    # softmax_act = batch_size * seq_len * vocab_size * precision
    # softmax_wt = (hidden_dim + 1) * vocab_size * precision
    # softmax_point = (2 * batch_size * seq_len * vocab_size + batch_size * seq_len) * precision
    # NOTE: sigmoid and exp could have been combined
    # 1 sigmoids
    # 1 exp
    # 1 pointwise div
    # softmax_mem = (softmax_act + softmax_wt + softmax_point)
    mem = 4 * seq_len * batch_size * hidden_dim / t *(1+vocab_size/hidden_dim) * (precision.activations / 2) #from https://arxiv.org/pdf/2205.05198
    return mem


def get_embedding_act_mem(batch_size, seq_len, hidden_dim, p, t, precision):
    mem = 4 * seq_len * batch_size * hidden_dim * p / t * (precision.activations / 2)  # from https://arxiv.org/pdf/2205.05198

    return mem
    
def get_embedding_weight_mem(
    vocab_size: int,
    hidden_dim: int,
    precision,
    tied_embeddings: bool = True,
    param_replica_factor: int = 1,  # Should be equal to dp. For ZeRO-3 (future work) set to 1.
) -> int:
    """
    Embedding WEIGHT memory per rank:
      - Input embeddings: (V*H*bytes)/vocab_shards * replica_factor
      (vocab shards is WIP)
      - Output head: same as input if untied, else 0.
    """
    per_matrix = vocab_size * hidden_dim * (precision.parameters + precision.master_parameters) / 1.0
    input_embed = per_matrix * param_replica_factor
    output_head = 0 if tied_embeddings else per_matrix * param_replica_factor
    return input_embed + output_head
    
def estimate_inference_memory(exp_hw_config, exp_model_config, **kwargs):
    """Run graph-based memory estimation for prefill + final decode peaks."""
    mode = kwargs.get("mode", "LLM")
    output_dir = kwargs.get(
        "output_dir",
        os.path.join(os.getcwd(), "output", "memory_estimator_smoke"),
    )

    model_cfg = exp_model_config.model_config
    sched_cfg = getattr(exp_hw_config, "sch_config", None)

    def _set_if_present(obj, attr, key):
        if key in kwargs and hasattr(obj, attr):
            setattr(obj, attr, int(kwargs[key]))

    if "batch_size" in kwargs:
        _set_if_present(model_cfg, "batch_size", "batch_size")
        _set_if_present(model_cfg, "global_batch_size", "batch_size")
    if "global_batch_size" in kwargs:
        _set_if_present(model_cfg, "global_batch_size", "global_batch_size")
        _set_if_present(model_cfg, "batch_size", "global_batch_size")

    for key in (
        "seq_len",
        "decode_len",
        "num_layers",
        "hidden_dim",
        "num_heads",
        "kv_heads",
        "intermediate_size",
        "vocab_size",
    ):
        _set_if_present(model_cfg, key, key)

    if sched_cfg is not None:
        for key in ("tp", "cp", "pp", "mb"):
            if key in kwargs and hasattr(sched_cfg, key):
                setattr(sched_cfg, key, int(kwargs[key]))
        if "moe_dp" in kwargs:
            sched_cfg.inference.moe_dp = int(kwargs["moe_dp"])
        if "replica_count" in kwargs:
            sched_cfg.inference.replica_count = int(kwargs["replica_count"])

    from inference_timing import TimeCalculationLLMInference
    from llm_execution import LLMExecutionDispatcher
    from memory_estimation import MemoryEstimator

    tc = TimeCalculationLLMInference(exp_hw_config, exp_model_config, mode, output_dir=output_dir)

    batch_size = tc._effective_transformer_batch()
    vocab_size = tc.vocab_size
    hidden_dim = tc.hidden_dim
    seq_len = tc.seq_len
    num_heads = tc.num_heads
    kv_heads = tc.kv_heads
    intermediate_size = tc.intermediate_size
    decode_len = int(getattr(tc.model, "decode_len", 0) or 0)
    prefill_len = max(0, int(seq_len) - decode_len)

    mem_estimator = MemoryEstimator(tc)
    num_SMs = tc.hw_config.tech_config.core.num_bundles

    prefill_peak_gb = 0.0
    if prefill_len > 0:
        transformer_timings, node_breakdown = tc.compute_all_gemm_and_node_times(
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
        if tc.use_moe and any(getattr(tc, "moe_layer_mask", []) or []):
            moe_transformer_timings, moe_node_breakdown = tc.compute_all_gemm_and_node_times(
                batch_size,
                vocab_size,
                hidden_dim,
                prefill_len,
                num_heads,
                kv_heads,
                tc.moe_intermediate_size,
                num_SMs,
                use_moe_override=True,
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
        ) = tc._prepare_execution_graphs(
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
        dispatcher = LLMExecutionDispatcher(
            time_calc=tc,
            pipeline_graph=pipeline_graph,
            pipeline_root=pipeline_root,
            interconnect_params=interconnect_params,
            transformer_graph=transformer_graph,
            transformer_forward_root=transformer_forward_root,
            transformer_backward_root=transformer_backward_root,
            moe_transformer_graph=moe_transformer_graph,
            moe_transformer_forward_root=moe_transformer_forward_root,
            moe_transformer_backward_root=moe_transformer_backward_root,
        )
        prefill_root = dispatcher.build_flattened_root_for_memory()
        prefill_memory_data = mem_estimator.build_memory_data(
            mode="inference",
            batch_size=batch_size,
            seq_len=prefill_len,
            kv_cache_tokens=0 if getattr(tc, "disable_kv_cache", False) else prefill_len,
        )
        tc.pipeline_graph = pipeline_graph
        _, prefill_peak_gb = mem_estimator.simulate_peak(
            prefill_root,
            prefill_memory_data,
            mode="inference",
            filename="memory_graph_prefill",
        )

    decode_peak_gb = 0.0
    if decode_len > 0:
        decode_gemm_shapes = process_decode_gemm_shapes(
            tc,
            batch_size=batch_size,
            current_seq_len=seq_len,
            d_model=hidden_dim,
            num_heads=num_heads,
            kv_heads=kv_heads,
            intermediate_size=tc.moe_intermediate_size if tc.use_moe else intermediate_size,
            vocab_size=vocab_size,
            model_type=tc.model_type,
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
        ), _ = tc.prepare_decode_graphs(
            batch_size=batch_size,
            total_seq_len=seq_len,
            gemm_shapes=decode_gemm_shapes,
        )
        decode_dispatcher = LLMExecutionDispatcher(
            time_calc=tc,
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
        decode_root = decode_dispatcher.build_flattened_root_for_memory()
        decode_memory_data = mem_estimator.build_memory_data(
            mode="inference",
            batch_size=batch_size,
            seq_len=1,
            gemm_shapes=decode_gemm_shapes,
            kv_cache_tokens=0 if getattr(tc, "disable_kv_cache", False) else seq_len,
        )
        tc.pipeline_graph = decode_pipeline_graph
        _, decode_peak_gb = mem_estimator.simulate_peak(
            decode_root,
            decode_memory_data,
            mode="inference",
            filename="memory_graph_decode",
        )

    max_peak_gb = max(prefill_peak_gb, decode_peak_gb)
    hardware_mem_bytes = getattr(tc.DRAM, "size", None)
    if hardware_mem_bytes is None and hasattr(tc.hw_config, "tech_config"):
        tech_cfg = tc.hw_config.tech_config
        if hasattr(tech_cfg, "DRAM"):
            hardware_mem_bytes = getattr(tech_cfg.DRAM, "size", None)

    capacity_gb = None
    headroom_gb = None
    if hardware_mem_bytes is not None:
        capacity_gb = float(hardware_mem_bytes) / float(1024 ** 3)
        headroom_gb = capacity_gb - max_peak_gb

    return {
        "prefill_peak_gb": prefill_peak_gb,
        "decode_peak_gb": decode_peak_gb,
        "max_peak_gb": max_peak_gb,
        "capacity_gb": capacity_gb,
        "headroom_gb": headroom_gb,
        "output_dir": output_dir,
    }


def _test_mem_req_total(exp_hw_config, exp_model_config, **kwargs):
    return estimate_inference_memory(exp_hw_config, exp_model_config, **kwargs)


# ====================================================================
# DECODE-SPECIFIC UTILITIES FOR AUTOREGRESSIVE INFERENCE
# ====================================================================

def kv_cache_token_bytes(batch_size, kv_heads, head_dim, precision_bytes):
    """Return total bytes to store K+V for a single new token."""
    return batch_size * kv_heads * head_dim * precision_bytes * 2


def autoregressive_decoder_gemm(self, batch_size, current_seq_len, d_model, num_heads, kv_heads, intermediate_size, vocab_size, model_type="gpt"):
    """
    Generate GEMM shapes for a single decode step in autoregressive generation.

    Key differences from training/prefill GEMMs:
    - Sequence length is typically 1 (generating one token at a time)
    - Attention over growing KV-cache (current_seq_len)
    - Always uses KV-cache (one-token query, cached keys/values)

    Parameters:
        batch_size (int): Batch size (B)
        current_seq_len (int): Current sequence length including cache (growing: 1, 2, 3, ...)
        d_model (int): Hidden size (D)
        num_heads (int): Number of attention heads (H)
        kv_heads (int): Number of key/value heads (H_kv)
        intermediate_size (int): First FFN layer output dimension (typically 4 * D)
        vocab_size (int): Vocabulary size (V)

    Returns:
        OrderedDict: GEMM shapes [M, K, N] for decode step operations
    """
    assert num_heads % kv_heads == 0, "num_heads must be divisible by kv_heads"
    head_dim, q_size, kv_size = attention_dim_sizes(
        d_model,
        num_heads,
        kv_heads,
        head_dim=getattr(self, "head_dim", None),
    )
    shared_heads = num_heads // kv_heads
    gemms = OrderedDict()
    attention_type = str(getattr(self, "attention_type", "mha")).lower()
    run_type = str(
        getattr(self, "run_type", getattr(getattr(self, "model", None), "run_type", "training"))
    ).lower()

    # Decode-specific GEMM shapes with KV-cache handling (always enabled)
    if attention_type == "mla" and run_type == "inference":
        qk_head_dim, _q_size_mla, _v_size_mla, _kv_up_size = mla_dim_sizes(
            num_heads,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
        )
        runtime_ops = mla_runtime_op_shapes(
            batch_size=batch_size,
            query_seq_len=1,
            hidden_dim=d_model,
            num_heads=num_heads,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
        )
        gemms.update(runtime_ops)
        proxy_sizes = mla_runtime_proxy_sizes(
            num_heads,
            self.q_lora_rank,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
        )
        gemms["qkv_proj"] = (batch_size, 1, d_model, proxy_sizes["qkv_output_size"])
        gemms["attention_score"] = (
            batch_size * num_heads,
            1,
            qk_head_dim,
            current_seq_len,
        )
        gemms["attention_output"] = (
            batch_size * num_heads,
            1,
            current_seq_len,
            proxy_sizes["attention_output_size"],
        )
        gemms["output_proj"] = (
            batch_size,
            1,
            proxy_sizes["output_proj_input_size"],
            d_model,
        )
    else:
        # QKV Projection: Only the new token (seq_len = 1)
        gemms["qkv_proj"] = (batch_size, 1, d_model, q_size + 2 * kv_size)

        # Attention Score: Q(new) @ K(cached+new)
        gemms["attention_score"] = (
            batch_size * kv_heads,
            1 * shared_heads,
            head_dim,
            current_seq_len,
        )

        # Attention Output: attention_weights @ V(cached+new)
        gemms["attention_output"] = (
            batch_size * kv_heads,
            1 * shared_heads,
            current_seq_len,
            head_dim,
        )

    # === POST-ATTENTION LAYERS (same regardless of cache) ===

    # Output projection & FFNs only process the new token
    if not (attention_type == "mla" and run_type == "inference"):
        gemms["output_proj"] = (batch_size, 1, q_size, d_model)
    projected_dim = (
        2 * intermediate_size
        if uses_gated_mlp(model_type, swiglu_mlp=getattr(self, "swiglu_mlp", False))
        else intermediate_size
    )
    if self.use_moe:
        moe_intermediate = int(getattr(self, "moe_intermediate_size", intermediate_size))
        projected_dim = (
            2 * moe_intermediate
            if uses_gated_mlp(model_type, swiglu_mlp=getattr(self, "swiglu_mlp", False))
            else moe_intermediate
        )
        # MoE GEMM shapes use token-owner dimensions; dispatch/combine modeled separately.
        gemms["ffn1"] = (batch_size, 1, d_model, projected_dim)
        gemms["ffn2"] = (batch_size, 1, moe_intermediate, d_model)
    else:
        gemms["ffn1"] = (batch_size, 1, d_model, projected_dim) 
        gemms["ffn2"] = (batch_size, 1, intermediate_size, d_model)
    gemms["linear"] = (batch_size, 1, d_model, vocab_size)
    gemms["router"] = (batch_size, 1, d_model, getattr(self, "moe_num_experts", 1))

    return gemms


def process_decode_gemm_shapes(
    self,
    batch_size,
    current_seq_len,
    d_model,
    num_heads,
    kv_heads,
    intermediate_size,
    vocab_size,
    model_type="gpt",
):
    """
    Process decode GEMM shapes and reshape them into 3D.

    Similar to process_gemm_shapes but for decode-specific patterns. Handles grouped
    query attention (GQA) by allowing kv_heads != num_heads.

    Integrates with existing RAPID-LLM GEMM processing infrastructure.
    """
    # Generate decode GEMM shapes in 4D
    gemm_shapes_4d = autoregressive_decoder_gemm(
        self,
        batch_size=batch_size,
        current_seq_len=current_seq_len,
        d_model=d_model,
        num_heads=num_heads,
        kv_heads=kv_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        model_type=model_type,
    )

    processed = OrderedDict()
    decode_attention_keys = ATTENTION_GEMM_KEYS
    for key, shape in gemm_shapes_4d.items():
        if key in decode_attention_keys:
            # Keep attention GEMMs in 4D for proper computation
            processed[key] = tuple(shape)
        else:
            # Reshape other GEMMs to 3D using existing infrastructure
            processed[key] = reshape_gemm_to_3d(shape)

    return processed






if __name__ == "__main__":

    
    exp_hw_config_path = "configs/hardware-config/a100_80GB.yaml"
    exp_model_config_path = "configs/model-config/LLM.yaml"
    exp_hw_path = os.path.expandvars(os.path.expanduser(exp_hw_config_path))
    exp_model_path = os.path.expandvars(os.path.expanduser(exp_model_config_path))
    exp_hw_config = config.parse_config(exp_hw_path, config_type="hardware")
    exp_model_config = config.parse_config(exp_model_path, config_type="LLM")
    summary = estimate_inference_memory(
        exp_hw_config,
        exp_model_config,
    )
    print("Memory Estimator Smoke Test")
    print(f"Prefill peak memory (per gpu): {summary['prefill_peak_gb']:.2f} GiB")
    print(f"Final decode peak memory (per gpu): {summary['decode_peak_gb']:.2f} GiB")
    print(f"Max peak memory (per gpu): {summary['max_peak_gb']:.2f} GiB")
    if summary["capacity_gb"] is None:
        print("Hardware memory capacity (per gpu): unknown")
    else:
        print(f"Hardware memory capacity (per gpu): {summary['capacity_gb']:.2f} GiB")
    if summary["headroom_gb"] is None:
        print("Memory headroom: unknown")
    else:
        print(f"Memory headroom: {summary['headroom_gb']:.2f} GiB")
    print(f"Output dir: {summary['output_dir']}")
