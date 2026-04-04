from pathlib import Path

import pytest

import config
import llm_util
from inference_timing import TimeCalculationLLMInference
from train_timing import (
    GemmType,
    SOFTMAX_BACKWARD_FLOPS_PER_ELEMENT,
    SOFTMAX_FORWARD_FLOPS_PER_ELEMENT,
    TimeCalculationLLM,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HW_BASE = PROJECT_ROOT / "validation_scripts" / "validation_configs" / "hardware-config" / "a100_80GB.yaml"


def _build_hw_config(*, tp: int = 1, cp: int = 1, tp_sp: bool = False) -> config.HWConfig:
    hw_config = config.parse_config(str(HW_BASE), config_type="hardware")
    hw_config.sch_config.tp = tp
    hw_config.sch_config.cp = cp
    hw_config.sch_config.pp = 1
    hw_config.sch_config.mb = 1
    hw_config.sch_config.tp_sp = tp_sp
    hw_config.sch_config.train.dp = 1
    hw_config.sch_config.train.ep = 1
    hw_config.sch_config.train.tp_ep = True
    return hw_config


def _build_mla_model(
    *,
    run_type: str = "training",
    use_flashattention: bool = False,
    use_moe: bool = False,
    full_recomputation: bool = False,
) -> config.ModelConfig:
    model_param = {
        "mode": "LLM",
        "run_type": run_type,
        "tied_embeddings": False,
        "model_type": "llama",
        "global_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "full_recomputation": full_recomputation,
        "seq_len": 8,
        "decode_len": 4 if run_type == "inference" else 0,
        "hidden_dim": 64,
        "attention": {
            "attention_type": "mla",
            "num_heads": 4,
            "head_dim": 16,
            "use_flashattention": use_flashattention,
            "attention_tile_size": 64,
            "kv_lora_rank": 4,
            "q_lora_rank": 8,
            "qk_nope_head_dim": 12,
            "qk_rope_head_dim": 4,
            "v_head_dim": 8,
        },
        "intermediate_size": 128,
        "vocab_size": 512,
        "num_layers": 2,
        "moe": {
            "num_experts": 4 if use_moe else 1,
            "top_k": 2 if use_moe else 1,
            "moe_intermediate_size": 128,
            "n_shared_experts": 0,
            "moe_layer_freq": 1,
            "first_k_dense_replace": 0,
        },
    }
    llm_config = config.LLMConfig.from_dict(model_param)
    inference_cfg = config.LLMInferenceConfig(sample_every=-1) if run_type == "inference" else None
    return config.ModelConfig(model_config=llm_config, inference_config=inference_cfg)


def _build_gqa_model() -> config.ModelConfig:
    model_param = {
        "mode": "LLM",
        "run_type": "training",
        "tied_embeddings": False,
        "model_type": "llama",
        "global_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "seq_len": 8,
        "hidden_dim": 64,
        "attention": {
            "attention_type": "gqa",
            "num_heads": 4,
            "kv_heads": 1,
            "head_dim": 16,
            "use_flashattention": False,
            "attention_tile_size": 64,
        },
        "intermediate_size": 128,
        "vocab_size": 512,
        "num_layers": 2,
        "moe": {
            "num_experts": 1,
            "top_k": 1,
            "moe_intermediate_size": 128,
            "n_shared_experts": 0,
            "moe_layer_freq": 1,
            "first_k_dense_replace": 0,
        },
    }
    llm_config = config.LLMConfig.from_dict(model_param)
    return config.ModelConfig(model_config=llm_config, inference_config=None)


def _matmul_flops(gemm):
    if len(gemm) == 4:
        batch, m, k, n = gemm
    elif len(gemm) == 3:
        batch, m, k, n = 1, gemm[0], gemm[1], gemm[2]
    else:
        raise ValueError(f"Unsupported GEMM rank: {len(gemm)}")
    return 2.0 * float(batch) * float(m) * float(k) * float(n)


def _softmax_forward_flops(gemm):
    batch, m, _k, n = gemm
    elements = batch * m * n
    return float(elements) * float(SOFTMAX_FORWARD_FLOPS_PER_ELEMENT + 1)


def _softmax_backward_flops(gemm):
    batch, m, _k, n = gemm
    elements = batch * m * n
    return float(elements) * float(SOFTMAX_BACKWARD_FLOPS_PER_ELEMENT + 1)


def _sharded_output_bytes(tc, gemm, gemm_type, *, precision_bytes, visible_cp=False):
    spec = tc._shard_gemm_descriptor(gemm, gemm_type)
    elements = spec.output_elements()
    if visible_cp and tc.cp > 1:
        elements *= tc.cp
    return float(elements) * float(precision_bytes)


def test_mla_param_groups_sum_to_expected_total():
    groups = llm_util.mla_attention_param_groups(
        hidden_dim=64,
        num_heads=4,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=8,
    )
    sizes = llm_util.mla_attention_param_sizes(
        hidden_dim=64,
        num_heads=4,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=8,
    )

    assert groups["projection_stage"] == 1152
    assert groups["attention_stage"] == 576
    assert groups["output_stage"] == 2176
    assert groups["total"] == 3904
    assert sizes["d_proj"] == 1024
    assert sizes["u_proj_q"] == 512
    assert sizes["u_proj_kv"] == 320
    assert sizes["out_proj"] == 2048
    assert sizes["total"] == groups["total"]


def test_mla_training_shapes_follow_latent_runtime_path():
    hw_config = _build_hw_config()
    model = _build_mla_model()
    config.validate_configs(hw_config, model)
    tc = TimeCalculationLLM(hw_config, model, "LLM")

    gemm_shapes = llm_util.process_gemm_shapes(
        tc,
        batch_size=4,
        seq_len=8,
        d_model=64,
        num_heads=4,
        kv_heads=4,
        intermediate_size=128,
        vocab_size=512,
    )

    assert gemm_shapes["qkv_proj"] == (32, 64, 32)
    assert gemm_shapes["attention_score"] == (16, 8, 16, 8)
    assert gemm_shapes["attention_output"] == (16, 8, 8, 4)
    assert gemm_shapes["output_proj"] == (32, 16, 64)
    assert gemm_shapes["attention_score_1"] == (16, 8, 8, 4)
    assert gemm_shapes["attention_score_2"] == (16, 8, 4, 8)
    assert gemm_shapes["attention_score_rope"] == (16, 8, 4, 8)
    assert gemm_shapes["attention_ctx_latent"] == (16, 8, 8, 4)


def test_mla_inference_prefill_shapes_follow_latent_runtime_path():
    hw_config = _build_hw_config()
    hw_config.inference_config = hw_config.sch_config.inference
    model = _build_mla_model(run_type="inference")
    config.validate_configs(hw_config, model)
    tc = TimeCalculationLLMInference(hw_config, model, "LLM")

    gemm_shapes = llm_util.process_gemm_shapes(
        tc,
        batch_size=4,
        seq_len=8,
        d_model=64,
        num_heads=4,
        kv_heads=4,
        intermediate_size=128,
        vocab_size=512,
    )

    assert gemm_shapes["qkv_proj"] == (32, 64, 32)
    assert gemm_shapes["K_rope_proj"] == (32, 64, 4)
    assert gemm_shapes["attention_output"] == (16, 8, 8, 4)
    assert gemm_shapes["output_proj"] == (32, 16, 64)
    assert gemm_shapes["attention_score_1"] == (16, 8, 8, 4)


def test_mla_inference_decode_shapes_follow_latent_runtime_path():
    hw_config = _build_hw_config()
    hw_config.inference_config = hw_config.sch_config.inference
    model = _build_mla_model(run_type="inference")
    config.validate_configs(hw_config, model)
    tc = TimeCalculationLLMInference(hw_config, model, "LLM")

    gemm_shapes = llm_util.process_decode_gemm_shapes(
        tc,
        batch_size=4,
        current_seq_len=8,
        d_model=64,
        num_heads=4,
        kv_heads=4,
        intermediate_size=128,
        vocab_size=512,
        model_type=tc.model_type,
    )

    assert gemm_shapes["qkv_proj"] == (4, 64, 32)
    assert gemm_shapes["K_rope_proj"] == (4, 64, 4)
    assert gemm_shapes["attention_score"] == (16, 1, 16, 8)
    assert gemm_shapes["attention_output"] == (16, 1, 8, 4)
    assert gemm_shapes["output_proj"] == (4, 16, 64)
    assert gemm_shapes["attention_score_1"] == (16, 1, 8, 4)


def test_mla_activation_tensor_bytes_match_expected_formula():
    activation = llm_util.mla_activation_tensor_bytes(
        batch_size=4,
        seq_len=8,
        key_seq_len=8,
        hidden_dim=64,
        intermediate_size=128,
        num_heads=4,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_rope_head_dim=4,
        precision_bytes=2,
        model_type="llama",
        flash_attention=False,
        full_recomputation=False,
        tp=1,
        cp=1,
    )

    assert activation["hidden_bytes"] == 4096.0
    assert activation["qkv_bytes"] == 2048.0
    assert activation["attention_ctx_bytes"] == 1024.0
    assert activation["attention_score_bytes"] == 2048.0
    assert activation["ffn_bytes"] == 16384.0
    assert activation["training_bytes"] == 37888.0
    assert activation["inference_peak_bytes"] == 16384.0


def test_mla_activation_tensor_bytes_track_tp_cp_and_flashattention():
    activation = llm_util.mla_activation_tensor_bytes(
        batch_size=4,
        seq_len=4,
        key_seq_len=8,
        hidden_dim=64,
        intermediate_size=128,
        num_heads=4,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_rope_head_dim=4,
        precision_bytes=2,
        model_type="llama",
        flash_attention=True,
        full_recomputation=False,
        tp=2,
        cp=2,
    )

    assert activation["hidden_bytes"] == 1024.0
    assert activation["qkv_bytes"] == 512.0
    assert activation["attention_ctx_bytes"] == 256.0
    assert activation["attention_score_bytes"] == 0.0
    assert activation["ffn_bytes"] == 4096.0
    assert activation["training_bytes"] == 8960.0
    assert activation["inference_peak_bytes"] == 4096.0


def test_mla_training_composite_flops_match_runtime_components():
    hw_config = _build_hw_config()
    model = _build_mla_model()
    config.validate_configs(hw_config, model)
    tc = TimeCalculationLLM(hw_config, model, "LLM")

    batch_size = tc._effective_transformer_batch()
    gemm_shapes = llm_util.process_gemm_shapes(
        tc,
        batch_size=batch_size,
        seq_len=tc.seq_len,
        d_model=tc.hidden_dim,
        num_heads=tc.num_heads,
        kv_heads=tc.kv_heads,
        intermediate_size=tc.intermediate_size,
        vocab_size=tc.vocab_size,
    )
    timings, _ = tc.compute_all_gemm_and_node_times(
        batch_size,
        tc.vocab_size,
        tc.hidden_dim,
        tc.seq_len,
        tc.num_heads,
        tc.kv_heads,
        tc.intermediate_size,
        tc.hw_config.tech_config.core.num_bundles,
    )

    expected_qkv_flops = sum(
        _matmul_flops(gemm_shapes[key]) for key in ("D_proj_q", "D_proj_kv", "U_proj_q_rope", "K_rope_proj")
    )
    expected_attn_score_flops = sum(
        _matmul_flops(gemm_shapes[key])
        for key in ("attention_score_1", "attention_score_2", "attention_score_rope", "attention_ctx_latent")
    )
    expected_softmax_f = _softmax_forward_flops(gemm_shapes["attention_score"])
    expected_softmax_b = _softmax_backward_flops(gemm_shapes["attention_score"])
    expected_out_flops = _matmul_flops(gemm_shapes["O_proj_absorbed"])

    assert timings["qkv_proj"].forward.flops == pytest.approx(expected_qkv_flops)
    assert timings["qkv_proj"].backward.flops == pytest.approx(2.0 * expected_qkv_flops)
    assert timings["attention_score"].forward.flops == pytest.approx(expected_attn_score_flops)
    assert timings["attention_score"].backward.flops == pytest.approx(2.0 * expected_attn_score_flops)
    assert timings["attention_scale_softmax"].forward.flops == pytest.approx(expected_softmax_f)
    assert timings["attention_scale_softmax"].backward.flops == pytest.approx(expected_softmax_b)
    assert timings["attention"].forward.flops == pytest.approx(expected_attn_score_flops + expected_softmax_f)
    assert timings["attention"].backward.flops == pytest.approx((2.0 * expected_attn_score_flops) + expected_softmax_b)
    assert timings["output_proj"].forward.flops == pytest.approx(expected_out_flops)
    assert timings["output_proj"].backward.flops == pytest.approx(2.0 * expected_out_flops)


@pytest.mark.parametrize(
    ("tp", "cp", "tp_sp"),
    [
        (2, 1, False),
        (2, 1, True),
        (1, 2, False),
        (2, 2, False),
    ],
)
def test_mla_training_parallel_configs_validate_and_run(tp, cp, tp_sp):
    hw_config = _build_hw_config(tp=tp, cp=cp, tp_sp=tp_sp)
    model = _build_mla_model()
    config.validate_configs(hw_config, model)
    tc = TimeCalculationLLM(hw_config, model, "LLM")

    batch_size = tc._effective_transformer_batch()
    timings, _ = tc.compute_all_gemm_and_node_times(
        batch_size,
        tc.vocab_size,
        tc.hidden_dim,
        tc.seq_len,
        tc.num_heads,
        tc.kv_heads,
        tc.intermediate_size,
        tc.hw_config.tech_config.core.num_bundles,
    )

    assert timings["qkv_proj"].forward.compute_time > 0.0
    assert timings["attention"].forward.compute_time > 0.0
    assert timings["output_proj"].forward.compute_time > 0.0


def test_mla_tp_training_comm_bytes_match_dense_parallel_rules():
    hw_config = _build_hw_config(tp=2)
    model = _build_mla_model()
    config.validate_configs(hw_config, model)
    tc = TimeCalculationLLM(hw_config, model, "LLM")

    batch_size = tc._effective_transformer_batch()
    timings, _ = tc.compute_all_gemm_and_node_times(
        batch_size,
        tc.vocab_size,
        tc.hidden_dim,
        tc.seq_len,
        tc.num_heads,
        tc.kv_heads,
        tc.intermediate_size,
        tc.hw_config.tech_config.core.num_bundles,
    )
    gemm_shapes = llm_util.process_gemm_shapes(
        tc,
        batch_size=batch_size,
        seq_len=tc.seq_len,
        d_model=tc.hidden_dim,
        num_heads=tc.num_heads,
        kv_heads=tc.kv_heads,
        intermediate_size=tc.intermediate_size,
        vocab_size=tc.vocab_size,
    )

    qkv_spec = tc._shard_gemm_descriptor(gemm_shapes["qkv_proj"], GemmType.QKV)
    out_spec = tc._shard_gemm_descriptor(gemm_shapes["output_proj"], GemmType.OUT_PROJ)
    expected_hidden_grad = qkv_spec.batch * qkv_spec.shard_m * qkv_spec.k * tc.precision.grad_communication
    expected_output_reduce = out_spec.batch * out_spec.shard_m * out_spec.n * tc.precision.activations

    assert timings["qkv_proj"].backward.comm_bytes == expected_hidden_grad
    assert timings["attention"].backward.comm_bytes == 0
    assert timings["output_proj"].forward.comm_bytes == expected_output_reduce


def test_mla_cp_training_comm_bytes_use_latent_cache_state():
    hw_config = _build_hw_config(cp=2)
    model = _build_mla_model()
    config.validate_configs(hw_config, model)
    tc = TimeCalculationLLM(hw_config, model, "LLM")

    batch_size = tc._effective_transformer_batch()
    timings, _ = tc.compute_all_gemm_and_node_times(
        batch_size,
        tc.vocab_size,
        tc.hidden_dim,
        tc.seq_len,
        tc.num_heads,
        tc.kv_heads,
        tc.intermediate_size,
        tc.hw_config.tech_config.core.num_bundles,
    )

    gemm_shapes = llm_util.process_gemm_shapes(
        tc,
        batch_size=batch_size,
        seq_len=tc.seq_len,
        d_model=tc.hidden_dim,
        num_heads=tc.num_heads,
        kv_heads=tc.kv_heads,
        intermediate_size=tc.intermediate_size,
        vocab_size=tc.vocab_size,
    )
    cache_state_bytes = _sharded_output_bytes(
        tc,
        gemm_shapes["D_proj_kv"],
        GemmType.QKV,
        precision_bytes=tc.precision.activations,
        visible_cp=True,
    ) + _sharded_output_bytes(
        tc,
        gemm_shapes["K_rope_proj"],
        GemmType.QKV,
        precision_bytes=tc.precision.activations,
        visible_cp=True,
    )
    cache_grad_bytes = _sharded_output_bytes(
        tc,
        gemm_shapes["D_proj_kv"],
        GemmType.QKV,
        precision_bytes=tc.precision.grad_communication,
        visible_cp=True,
    ) + _sharded_output_bytes(
        tc,
        gemm_shapes["K_rope_proj"],
        GemmType.QKV,
        precision_bytes=tc.precision.grad_communication,
        visible_cp=True,
    )

    assert timings["qkv_proj"].forward.comm_bytes == cache_state_bytes
    assert timings["qkv_proj"].backward.comm_bytes == 0
    assert timings["attention"].backward.comm_bytes == cache_grad_bytes
    assert timings["output_proj"].backward.comm_bytes == cache_state_bytes


def test_mla_hybrid_training_comm_bytes_match_one_tp_shard_of_latent_cache():
    hw_config = _build_hw_config(tp=2, cp=2)
    model = _build_mla_model()
    config.validate_configs(hw_config, model)
    tc = TimeCalculationLLM(hw_config, model, "LLM")

    batch_size = tc._effective_transformer_batch()
    timings, _ = tc.compute_all_gemm_and_node_times(
        batch_size,
        tc.vocab_size,
        tc.hidden_dim,
        tc.seq_len,
        tc.num_heads,
        tc.kv_heads,
        tc.intermediate_size,
        tc.hw_config.tech_config.core.num_bundles,
    )
    gemm_shapes = llm_util.process_gemm_shapes(
        tc,
        batch_size=batch_size,
        seq_len=tc.seq_len,
        d_model=tc.hidden_dim,
        num_heads=tc.num_heads,
        kv_heads=tc.kv_heads,
        intermediate_size=tc.intermediate_size,
        vocab_size=tc.vocab_size,
    )

    qkv_spec = tc._shard_gemm_descriptor(gemm_shapes["qkv_proj"], GemmType.QKV)
    out_spec = tc._shard_gemm_descriptor(gemm_shapes["output_proj"], GemmType.OUT_PROJ)
    expected_qkv_backward = qkv_spec.batch * qkv_spec.shard_m * qkv_spec.k * tc.precision.grad_communication
    expected_output_forward = out_spec.batch * out_spec.shard_m * out_spec.n * tc.precision.activations
    expected_cache_forward = _sharded_output_bytes(
        tc,
        gemm_shapes["D_proj_kv"],
        GemmType.QKV,
        precision_bytes=tc.precision.activations,
        visible_cp=True,
    ) + _sharded_output_bytes(
        tc,
        gemm_shapes["K_rope_proj"],
        GemmType.QKV,
        precision_bytes=tc.precision.activations,
        visible_cp=True,
    )
    expected_cache_backward = _sharded_output_bytes(
        tc,
        gemm_shapes["D_proj_kv"],
        GemmType.QKV,
        precision_bytes=tc.precision.grad_communication,
        visible_cp=True,
    ) + _sharded_output_bytes(
        tc,
        gemm_shapes["K_rope_proj"],
        GemmType.QKV,
        precision_bytes=tc.precision.grad_communication,
        visible_cp=True,
    )

    assert timings["qkv_proj"].forward.comm_bytes == expected_cache_forward
    assert timings["qkv_proj"].backward.comm_bytes == expected_qkv_backward
    assert timings["attention"].backward.comm_bytes == expected_cache_backward
    assert timings["output_proj"].forward.comm_bytes == expected_output_forward
    assert timings["output_proj"].backward.comm_bytes == expected_cache_forward


def test_mla_decode_composite_flops_match_runtime_components():
    hw_config = _build_hw_config()
    hw_config.inference_config = hw_config.sch_config.inference
    model = _build_mla_model(run_type="inference")
    config.validate_configs(hw_config, model)
    tc = TimeCalculationLLMInference(hw_config, model, "LLM")

    gemm_shapes = llm_util.process_decode_gemm_shapes(
        tc,
        batch_size=tc._effective_transformer_batch(),
        current_seq_len=8,
        d_model=tc.hidden_dim,
        num_heads=tc.num_heads,
        kv_heads=tc.kv_heads,
        intermediate_size=tc.intermediate_size,
        vocab_size=tc.vocab_size,
        model_type=tc.model_type,
    )
    timings, _ = tc._build_decode_transformer_results(
        batch_size=tc._effective_transformer_batch(),
        total_seq_len=8,
        use_moe_layer=False,
        gemm_shapes=gemm_shapes,
    )

    expected_qkv_flops = sum(
        _matmul_flops(gemm_shapes[key]) for key in ("D_proj_q", "D_proj_kv", "U_proj_q_rope", "K_rope_proj")
    )
    expected_attn_score_flops = sum(
        _matmul_flops(gemm_shapes[key])
        for key in ("attention_score_1", "attention_score_2", "attention_score_rope", "attention_ctx_latent")
    )
    expected_softmax_f = _softmax_forward_flops(gemm_shapes["attention_score"])
    expected_out_flops = _matmul_flops(gemm_shapes["O_proj_absorbed"])

    assert timings["qkv_proj"].forward.flops == pytest.approx(expected_qkv_flops)
    assert timings["attention_score"].forward.flops == pytest.approx(expected_attn_score_flops)
    assert timings["attention_scale_softmax"].forward.flops == pytest.approx(expected_softmax_f)
    assert timings["attention"].forward.flops == pytest.approx(expected_attn_score_flops + expected_softmax_f)
    assert timings["output_proj"].forward.flops == pytest.approx(expected_out_flops)


def test_mla_prefill_compute_path_runs_with_zero_backward_directions():
    hw_config = _build_hw_config()
    hw_config.inference_config = hw_config.sch_config.inference
    model = _build_mla_model(run_type="inference")
    config.validate_configs(hw_config, model)
    tc = TimeCalculationLLMInference(hw_config, model, "LLM")

    batch_size = tc._effective_transformer_batch()
    timings, _ = tc.compute_all_gemm_and_node_times(
        batch_size,
        tc.vocab_size,
        tc.hidden_dim,
        tc.seq_len - tc.model.decode_len,
        tc.num_heads,
        tc.kv_heads,
        tc.intermediate_size,
        tc.hw_config.tech_config.core.num_bundles,
    )

    assert timings["qkv_proj"].backward.compute_time == 0.0
    assert timings["attention"].backward.compute_time == 0.0
    assert timings["output_proj"].backward.compute_time == 0.0


def test_mla_training_flashattention_is_supported_and_removes_score_storage():
    hw_config = _build_hw_config()
    model = _build_mla_model(use_flashattention=True)
    config.validate_configs(hw_config, model)

    dense = llm_util.mla_activation_tensor_bytes(
        batch_size=4,
        seq_len=8,
        key_seq_len=8,
        hidden_dim=64,
        intermediate_size=128,
        num_heads=4,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_rope_head_dim=4,
        precision_bytes=2,
        model_type="llama",
        flash_attention=False,
        full_recomputation=False,
        tp=1,
        cp=1,
    )
    flash = llm_util.mla_activation_tensor_bytes(
        batch_size=4,
        seq_len=8,
        key_seq_len=8,
        hidden_dim=64,
        intermediate_size=128,
        num_heads=4,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_rope_head_dim=4,
        precision_bytes=2,
        model_type="llama",
        flash_attention=True,
        full_recomputation=False,
        tp=1,
        cp=1,
    )

    assert dense["attention_score_bytes"] > 0.0
    assert flash["attention_score_bytes"] == 0.0
    assert flash["training_bytes"] < dense["training_bytes"]


def test_mla_training_flashattention_backward_adds_selective_recompute_flops():
    hw_config = _build_hw_config()
    model = _build_mla_model(use_flashattention=True, full_recomputation=False)
    config.validate_configs(hw_config, model)
    tc = TimeCalculationLLM(hw_config, model, "LLM")

    batch_size = tc._effective_transformer_batch()
    gemm_shapes = llm_util.process_gemm_shapes(
        tc,
        batch_size=batch_size,
        seq_len=tc.seq_len,
        d_model=tc.hidden_dim,
        num_heads=tc.num_heads,
        kv_heads=tc.kv_heads,
        intermediate_size=tc.intermediate_size,
        vocab_size=tc.vocab_size,
    )
    timings, _ = tc.compute_all_gemm_and_node_times(
        batch_size,
        tc.vocab_size,
        tc.hidden_dim,
        tc.seq_len,
        tc.num_heads,
        tc.kv_heads,
        tc.intermediate_size,
        tc.hw_config.tech_config.core.num_bundles,
    )

    score_only_forward = sum(
        _matmul_flops(gemm_shapes[key])
        for key in ("attention_score_1", "attention_score_2", "attention_score_rope")
    )
    softmax_forward = _softmax_forward_flops(gemm_shapes["attention_score"])
    nonflash_attention_backward = (
        2.0
        * sum(
            _matmul_flops(gemm_shapes[key])
            for key in ("attention_score_1", "attention_score_2", "attention_score_rope", "attention_ctx_latent")
        )
        + _softmax_backward_flops(gemm_shapes["attention_score"])
    )

    assert timings["attention"].backward.flops == pytest.approx(
        nonflash_attention_backward + score_only_forward + softmax_forward
    )


def test_mla_moe_training_only_changes_attention_side():
    hw_config = _build_hw_config()
    model = _build_mla_model(use_moe=True)
    config.validate_configs(hw_config, model)
    tc = TimeCalculationLLM(hw_config, model, "LLM")

    batch_size = tc._effective_transformer_batch()
    num_sms = tc.hw_config.tech_config.core.num_bundles
    dense_timings, _ = tc.compute_all_gemm_and_node_times(
        batch_size,
        tc.vocab_size,
        tc.hidden_dim,
        tc.seq_len,
        tc.num_heads,
        tc.kv_heads,
        tc.intermediate_size,
        num_sms,
        use_moe_override=False,
    )
    moe_timings, _ = tc.compute_all_gemm_and_node_times(
        batch_size,
        tc.vocab_size,
        tc.hidden_dim,
        tc.seq_len,
        tc.num_heads,
        tc.kv_heads,
        tc.moe_intermediate_size,
        num_sms,
        use_moe_override=True,
    )

    for key in ("qkv_proj", "attention", "output_proj"):
        assert moe_timings[key].forward.flops == pytest.approx(dense_timings[key].forward.flops)
    assert "router" in moe_timings
    assert "moe_dispatch" in moe_timings
    assert "moe_combine" in moe_timings


def test_gqa_attention_params_respect_kv_heads_independent_of_model_type():
    hw_config = _build_hw_config()
    model = _build_gqa_model()
    config.validate_configs(hw_config, model)

    tc = TimeCalculationLLM(hw_config, model, "LLM")
    qkv_params, output_params = tc._attention_param_components(64)

    assert qkv_params == 6144
    assert output_params == 4096


def test_mla_kv_cache_bytes_use_latent_rank_plus_shared_rope():
    bytes_per_token = llm_util.attention_kv_cache_token_bytes(
        "mla",
        batch_size=4,
        kv_heads=4,
        head_dim=16,
        precision_bytes=2,
        kv_lora_rank=4,
        qk_rope_head_dim=4,
    )

    assert bytes_per_token == 64


def test_mla_inference_decode_sampling_runs():
    hw_config = _build_hw_config()
    hw_config.inference_config = hw_config.sch_config.inference
    model = _build_mla_model(run_type="inference")
    config.validate_configs(hw_config, model)

    tc = TimeCalculationLLMInference(hw_config, model, "LLM")
    decode_time, decode_energy, samples = tc.calc_decode_time()

    assert decode_time > 0.0
    assert decode_energy > 0.0
    assert len(samples) > 0


@pytest.mark.parametrize(
    ("tp", "cp", "tp_sp"),
    [
        (2, 1, False),
        (2, 1, True),
        (1, 2, False),
        (2, 2, False),
    ],
)
def test_mla_inference_parallel_configs_validate_and_run(tp, cp, tp_sp):
    hw_config = _build_hw_config(tp=tp, cp=cp, tp_sp=tp_sp)
    hw_config.inference_config = hw_config.sch_config.inference
    model = _build_mla_model(run_type="inference")
    config.validate_configs(hw_config, model)
    tc = TimeCalculationLLMInference(hw_config, model, "LLM")

    batch_size = tc._effective_transformer_batch()
    timings, _ = tc.compute_all_gemm_and_node_times(
        batch_size,
        tc.vocab_size,
        tc.hidden_dim,
        tc.seq_len - tc.model.decode_len,
        tc.num_heads,
        tc.kv_heads,
        tc.intermediate_size,
        tc.hw_config.tech_config.core.num_bundles,
    )
    decode_shapes = llm_util.process_decode_gemm_shapes(
        tc,
        batch_size=batch_size,
        current_seq_len=8,
        d_model=tc.hidden_dim,
        num_heads=tc.num_heads,
        kv_heads=tc.kv_heads,
        intermediate_size=tc.intermediate_size,
        vocab_size=tc.vocab_size,
        model_type=tc.model_type,
    )
    decode_timings, _ = tc._build_decode_transformer_results(
        batch_size=batch_size,
        total_seq_len=8,
        use_moe_layer=False,
        gemm_shapes=decode_shapes,
    )

    assert timings["qkv_proj"].forward.compute_time > 0.0
    assert decode_timings["qkv_proj"].forward.compute_time > 0.0


def test_mla_inference_decode_cp_broadcasts_query_state_not_hidden_state():
    hw_config = _build_hw_config(cp=2)
    hw_config.inference_config = hw_config.sch_config.inference
    model = _build_mla_model(run_type="inference")
    config.validate_configs(hw_config, model)
    tc = TimeCalculationLLMInference(hw_config, model, "LLM")

    batch_size = tc._effective_transformer_batch()
    decode_shapes = llm_util.process_decode_gemm_shapes(
        tc,
        batch_size=batch_size,
        current_seq_len=8,
        d_model=tc.hidden_dim,
        num_heads=tc.num_heads,
        kv_heads=tc.kv_heads,
        intermediate_size=tc.intermediate_size,
        vocab_size=tc.vocab_size,
        model_type=tc.model_type,
    )
    timings, _ = tc._build_decode_transformer_results(
        batch_size=batch_size,
        total_seq_len=8,
        use_moe_layer=False,
        gemm_shapes=decode_shapes,
    )

    expected_query_state_bytes = _sharded_output_bytes(
        tc,
        decode_shapes["D_proj_q"],
        GemmType.QKV,
        precision_bytes=tc.precision.activations,
    ) + _sharded_output_bytes(
        tc,
        decode_shapes["U_proj_q_rope"],
        GemmType.QKV,
        precision_bytes=tc.precision.activations,
    )
    assert timings["qkv_proj"].forward.comm_bytes == expected_query_state_bytes


def test_mla_inference_flashattention_prefill_is_supported_and_decode_still_runs():
    hw_config = _build_hw_config()
    hw_config.inference_config = hw_config.sch_config.inference
    model = _build_mla_model(run_type="inference", use_flashattention=True)
    config.validate_configs(hw_config, model)

    tc = TimeCalculationLLMInference(hw_config, model, "LLM")
    prefill_timings, _ = tc.compute_all_gemm_and_node_times(
        tc._effective_transformer_batch(),
        tc.vocab_size,
        tc.hidden_dim,
        tc.seq_len - tc.model.decode_len,
        tc.num_heads,
        tc.kv_heads,
        tc.intermediate_size,
        tc.hw_config.tech_config.core.num_bundles,
    )
    decode_time, decode_energy, samples = tc.calc_decode_time()

    assert prefill_timings["attention"].forward.compute_time > 0.0
    assert decode_time > 0.0
    assert decode_energy > 0.0
    assert len(samples) > 0


def test_mla_inference_memory_shards_kv_cache_with_tp_and_cp():
    hw_config = _build_hw_config(tp=2, cp=2)
    hw_config.inference_config = hw_config.sch_config.inference
    model = _build_mla_model(run_type="inference")
    config.validate_configs(hw_config, model)
    tc = TimeCalculationLLMInference(hw_config, model, "LLM")
    from memory_estimation import MemoryEstimator

    memory_data = MemoryEstimator(tc).build_memory_data(
        mode="inference",
        batch_size=tc._effective_transformer_batch(),
        seq_len=1,
        kv_cache_tokens=8,
    )

    per_token_bytes = llm_util.mla_kv_cache_token_bytes(
        batch_size=tc._effective_transformer_batch(),
        kv_lora_rank=tc.kv_lora_rank,
        qk_rope_head_dim=tc.qk_rope_head_dim,
        precision_bytes=tc.precision.kv_cache,
    )
    expected = (per_token_bytes * 4) / 2
    assert memory_data["kv_cache_bytes_per_layer"] == pytest.approx(expected)
