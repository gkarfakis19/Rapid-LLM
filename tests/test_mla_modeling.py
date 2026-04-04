from pathlib import Path

import pytest

import config
import llm_util
from inference_timing import TimeCalculationLLMInference
from memory_estimation import MemoryEstimator
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


def _build_gqa_model(*, run_type: str = "training") -> config.ModelConfig:
    model_param = {
        "mode": "LLM",
        "run_type": run_type,
        "tied_embeddings": False,
        "model_type": "llama",
        "global_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "seq_len": 8,
        "decode_len": 4 if run_type == "inference" else 0,
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
    inference_cfg = config.LLMInferenceConfig(sample_every=-1) if run_type == "inference" else None
    return config.ModelConfig(model_config=llm_config, inference_config=inference_cfg)


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


def _local_output_elements(tc, gemm, gemm_type):
    return tc._shard_gemm_descriptor(gemm, gemm_type).output_elements()


def test_mla_param_groups_match_megatron_stage_contract():
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
    per_rank = llm_util.mla_attention_params_per_rank(
        hidden_dim=64,
        num_heads=4,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=8,
        tp=2,
    )

    assert groups["projection_stage"] == 1856
    assert groups["attention_stage"] == 0
    assert groups["output_stage"] == 2048
    assert groups["total"] == 3904
    assert sizes["total"] == 3904
    assert per_rank["replicated"] == 1024
    assert per_rank["sharded"] == pytest.approx(1440.0)
    assert per_rank["total"] == pytest.approx(2464.0)


def test_mla_training_shapes_follow_full_megatron_runtime_path():
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

    assert gemm_shapes["D_proj_q"] == (32, 64, 8)
    assert gemm_shapes["D_proj_kv"] == (32, 64, 8)
    assert gemm_shapes["U_proj_q"] == (32, 8, 64)
    assert gemm_shapes["U_proj_kv"] == (32, 4, 80)
    assert gemm_shapes["attention_score"] == (16, 8, 16, 8)
    assert gemm_shapes["attention_output"] == (16, 8, 8, 8)
    assert gemm_shapes["output_proj"] == (32, 32, 64)
    assert "attention_score_1" not in gemm_shapes
    assert "O_proj_absorbed" not in gemm_shapes


def test_mla_inference_prefill_shapes_follow_full_megatron_runtime_path():
    hw_config = _build_hw_config(tp=2)
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

    assert gemm_shapes["D_proj_q"] == (32, 64, 8)
    assert gemm_shapes["D_proj_kv"] == (32, 64, 8)
    assert gemm_shapes["U_proj_q"] == (32, 8, 64)
    assert gemm_shapes["U_proj_kv"] == (32, 4, 80)
    assert gemm_shapes["attention_score"] == (16, 8, 16, 8)
    assert gemm_shapes["attention_output"] == (16, 8, 8, 8)
    assert gemm_shapes["output_proj"] == (32, 32, 64)


def test_mla_inference_decode_defaults_to_full_cache_path():
    hw_config = _build_hw_config(tp=2)
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

    assert gemm_shapes["D_proj_q"] == (4, 64, 8)
    assert gemm_shapes["D_proj_kv"] == (4, 64, 8)
    assert gemm_shapes["U_proj_q"] == (4, 8, 64)
    assert gemm_shapes["U_proj_kv"] == (4, 4, 80)
    assert gemm_shapes["attention_score"] == (16, 1, 16, 8)
    assert gemm_shapes["attention_output"] == (16, 1, 8, 8)
    assert gemm_shapes["output_proj"] == (4, 32, 64)
    assert "attention_score_1" not in gemm_shapes
    assert "O_proj_absorbed" not in gemm_shapes


def test_mla_local_projection_shapes_match_megatron_tp_and_cp_rules():
    hw_config = _build_hw_config(tp=2, cp=2)
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

    assert _local_output_elements(tc, gemm_shapes["D_proj_q"], GemmType.MLA_DOWN_PROJ) == 128
    assert _local_output_elements(tc, gemm_shapes["D_proj_kv"], GemmType.MLA_DOWN_PROJ) == 128
    assert _local_output_elements(tc, gemm_shapes["U_proj_q"], GemmType.QKV) == 512
    assert _local_output_elements(tc, gemm_shapes["U_proj_kv"], GemmType.QKV) == 640
    assert _local_output_elements(tc, gemm_shapes["attention_output"], GemmType.ATTENTION_OUTPUT) == 256


def test_mla_decode_cp_shapes_match_megatron_local_rules():
    hw_config = _build_hw_config(tp=2, cp=2)
    hw_config.inference_config = hw_config.sch_config.inference
    full_model = _build_mla_model(run_type="inference")
    config.validate_configs(hw_config, full_model)
    full_tc = TimeCalculationLLMInference(hw_config, full_model, "LLM")

    full_shapes = llm_util.process_decode_gemm_shapes(
        full_tc,
        batch_size=4,
        current_seq_len=8,
        d_model=64,
        num_heads=4,
        kv_heads=4,
        intermediate_size=128,
        vocab_size=512,
        model_type=full_tc.model_type,
    )

    assert _local_output_elements(full_tc, full_shapes["D_proj_q"], GemmType.MLA_DOWN_PROJ) == 16
    assert _local_output_elements(full_tc, full_shapes["D_proj_kv"], GemmType.MLA_DOWN_PROJ) == 16
    assert _local_output_elements(full_tc, full_shapes["U_proj_q"], GemmType.QKV) == 64
    assert _local_output_elements(full_tc, full_shapes["U_proj_kv"], GemmType.QKV) == 80
    assert _local_output_elements(full_tc, full_shapes["attention_output"], GemmType.ATTENTION_OUTPUT) == 64

def test_mla_activation_tensor_bytes_match_full_runtime_formula():
    activation = llm_util.mla_activation_tensor_bytes(
        batch_size=4,
        seq_len=8,
        key_seq_len=8,
        hidden_dim=64,
        intermediate_size=128,
        num_heads=4,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=8,
        precision_bytes=2,
        model_type="llama",
        flash_attention=False,
        full_recomputation=False,
        tp=1,
        cp=1,
    )

    assert activation["hidden_bytes"] == 4096.0
    assert activation["qkv_bytes"] == 10240.0
    assert activation["attention_ctx_bytes"] == 2048.0
    assert activation["attention_score_bytes"] == 2048.0
    assert activation["ffn_bytes"] == 16384.0
    assert activation["training_bytes"] == 47104.0
    assert activation["inference_peak_bytes"] == 16384.0


def test_mla_activation_tensor_bytes_track_tp_cp_and_flash_modes():
    full_flash = llm_util.mla_activation_tensor_bytes(
        batch_size=4,
        seq_len=4,
        key_seq_len=8,
        hidden_dim=64,
        intermediate_size=128,
        num_heads=4,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=8,
        precision_bytes=2,
        model_type="llama",
        flash_attention=True,
        full_recomputation=False,
        tp=2,
        cp=2,
    )

    assert full_flash["qkv_bytes"] == 2816.0
    assert full_flash["attention_ctx_bytes"] == 512.0
    assert full_flash["attention_score_bytes"] == 0.0
    assert full_flash["training_bytes"] == 15616.0
    assert full_flash["inference_peak_bytes"] == 4096.0


def test_mla_training_composite_flops_match_full_runtime_components():
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
        _matmul_flops(gemm_shapes[key]) for key in ("D_proj_q", "D_proj_kv", "U_proj_q", "U_proj_kv")
    )
    expected_score_flops = _matmul_flops(gemm_shapes["attention_score"])
    expected_output_flops = _matmul_flops(gemm_shapes["attention_output"])
    expected_softmax_f = _softmax_forward_flops(gemm_shapes["attention_score"])
    expected_softmax_b = _softmax_backward_flops(gemm_shapes["attention_score"])
    expected_out_proj_flops = _matmul_flops(gemm_shapes["output_proj"])

    assert timings["qkv_proj"].forward.flops == pytest.approx(expected_qkv_flops)
    assert timings["qkv_proj"].backward.flops == pytest.approx(2.0 * expected_qkv_flops)
    assert timings["attention_score"].forward.flops == pytest.approx(expected_score_flops)
    assert timings["attention_score"].backward.flops == pytest.approx(2.0 * expected_score_flops)
    assert timings["attention_output"].forward.flops == pytest.approx(expected_output_flops)
    assert timings["attention_output"].backward.flops == pytest.approx(2.0 * expected_output_flops)
    assert timings["attention_scale_softmax"].forward.flops == pytest.approx(expected_softmax_f)
    assert timings["attention_scale_softmax"].backward.flops == pytest.approx(expected_softmax_b)
    assert timings["attention"].forward.flops == pytest.approx(
        expected_score_flops + expected_output_flops + expected_softmax_f
    )
    assert timings["attention"].backward.flops == pytest.approx(
        (2.0 * expected_score_flops) + (2.0 * expected_output_flops) + expected_softmax_b
    )
    assert timings["output_proj"].forward.flops == pytest.approx(expected_out_proj_flops)
    assert timings["output_proj"].backward.flops == pytest.approx(2.0 * expected_out_proj_flops)


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


def test_mla_training_tp_and_cp_comm_bytes_follow_megatron_contract():
    hw_tp = _build_hw_config(tp=2)
    model = _build_mla_model()
    config.validate_configs(hw_tp, model)
    tc_tp = TimeCalculationLLM(hw_tp, model, "LLM")
    timings_tp, _ = tc_tp.compute_all_gemm_and_node_times(
        tc_tp._effective_transformer_batch(),
        tc_tp.vocab_size,
        tc_tp.hidden_dim,
        tc_tp.seq_len,
        tc_tp.num_heads,
        tc_tp.kv_heads,
        tc_tp.intermediate_size,
        tc_tp.hw_config.tech_config.core.num_bundles,
    )
    assert timings_tp["qkv_proj"].backward.comm_bytes == 1024
    assert timings_tp["attention"].backward.comm_bytes == 0
    assert timings_tp["output_proj"].forward.comm_bytes == 4096

    hw_cp = _build_hw_config(cp=2)
    config.validate_configs(hw_cp, model)
    tc_cp = TimeCalculationLLM(hw_cp, model, "LLM")
    timings_cp, _ = tc_cp.compute_all_gemm_and_node_times(
        tc_cp._effective_transformer_batch(),
        tc_cp.vocab_size,
        tc_cp.hidden_dim,
        tc_cp.seq_len,
        tc_cp.num_heads,
        tc_cp.kv_heads,
        tc_cp.intermediate_size,
        tc_cp.hw_config.tech_config.core.num_bundles,
    )
    assert timings_cp["qkv_proj"].forward.comm_bytes == 6144
    assert timings_cp["attention"].backward.comm_bytes == 6144
    assert timings_cp["output_proj"].backward.comm_bytes == 6144

    hw_hybrid = _build_hw_config(tp=2, cp=2)
    config.validate_configs(hw_hybrid, model)
    tc_hybrid = TimeCalculationLLM(hw_hybrid, model, "LLM")
    timings_hybrid, _ = tc_hybrid.compute_all_gemm_and_node_times(
        tc_hybrid._effective_transformer_batch(),
        tc_hybrid.vocab_size,
        tc_hybrid.hidden_dim,
        tc_hybrid.seq_len,
        tc_hybrid.num_heads,
        tc_hybrid.kv_heads,
        tc_hybrid.intermediate_size,
        tc_hybrid.hw_config.tech_config.core.num_bundles,
    )
    assert timings_hybrid["qkv_proj"].forward.comm_bytes == 3072
    assert timings_hybrid["qkv_proj"].backward.comm_bytes == 512
    assert timings_hybrid["attention"].backward.comm_bytes == 3072
    assert timings_hybrid["output_proj"].forward.comm_bytes == 2048
    assert timings_hybrid["output_proj"].backward.comm_bytes == 3072


def test_mla_inference_prefill_and_decode_comm_bytes_follow_full_cache_mode():
    hw_config = _build_hw_config(tp=2, cp=2)
    hw_config.inference_config = hw_config.sch_config.inference
    model = _build_mla_model(run_type="inference")
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
    decode_shapes = llm_util.process_decode_gemm_shapes(
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
    decode_timings, _ = tc._build_decode_transformer_results(
        batch_size=tc._effective_transformer_batch(),
        total_seq_len=8,
        use_moe_layer=False,
        gemm_shapes=decode_shapes,
    )

    assert prefill_timings["qkv_proj"].forward.comm_bytes == 1536
    assert prefill_timings["output_proj"].forward.comm_bytes == 1024
    assert decode_timings["qkv_proj"].forward.comm_bytes == 128
    assert decode_timings["output_proj"].forward.comm_bytes == 256

def test_mla_inference_decode_flash_flag_is_ignored_without_flashmla_support():
    hw_config = _build_hw_config(tp=2)
    hw_config.inference_config = hw_config.sch_config.inference

    def _decode_timings(*, use_flashattention: bool):
        model = _build_mla_model(
            run_type="inference",
            use_flashattention=use_flashattention,
        )
        config.validate_configs(hw_config, model)
        tc = TimeCalculationLLMInference(hw_config, model, "LLM")
        decode_shapes = llm_util.process_decode_gemm_shapes(
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
            gemm_shapes=decode_shapes,
        )
        return timings

    no_flash = _decode_timings(use_flashattention=False)
    with_flash = _decode_timings(use_flashattention=True)
    assert no_flash["attention"].forward.compute_time == pytest.approx(
        with_flash["attention"].forward.compute_time
    )
    assert no_flash["attention"].forward.flops == pytest.approx(
        with_flash["attention"].forward.flops
    )
    assert no_flash["attention"].forward.comm_bytes == with_flash["attention"].forward.comm_bytes
    assert no_flash["qkv_proj"].forward.comm_bytes == with_flash["qkv_proj"].forward.comm_bytes
    assert no_flash["output_proj"].forward.comm_bytes == with_flash["output_proj"].forward.comm_bytes

def test_mla_training_flashattention_uses_full_attention_contract():
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
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=8,
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
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=8,
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
    nonflash_attention_backward = (
        2.0 * _matmul_flops(gemm_shapes["attention_score"])
        + 2.0 * _matmul_flops(gemm_shapes["attention_output"])
        + _softmax_backward_flops(gemm_shapes["attention_score"])
    )
    assert timings["attention"].backward.flops == pytest.approx(
        nonflash_attention_backward
        + _matmul_flops(gemm_shapes["attention_score"])
        + _softmax_forward_flops(gemm_shapes["attention_score"])
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


def test_mla_kv_cache_bytes_and_memory_follow_full_cache_mode():
    full_bytes = llm_util.attention_kv_cache_token_bytes(
        "mla",
        batch_size=4,
        kv_heads=4,
        head_dim=16,
        precision_bytes=2,
        kv_lora_rank=4,
        num_heads=4,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=8,
    )

    assert full_bytes == 768

    hw_config = _build_hw_config(tp=2, cp=2)
    hw_config.inference_config = hw_config.sch_config.inference

    full_model = _build_mla_model(run_type="inference")
    config.validate_configs(hw_config, full_model)
    full_tc = TimeCalculationLLMInference(hw_config, full_model, "LLM")
    full_memory = MemoryEstimator(full_tc).build_memory_data(
        mode="inference",
        batch_size=full_tc._effective_transformer_batch(),
        seq_len=1,
        kv_cache_tokens=8,
    )
    assert full_memory["kv_cache_bytes_per_layer"] == pytest.approx(1536.0)


def test_mla_inference_decode_sampling_runs_for_full_mode():
    hw_config = _build_hw_config()
    hw_config.inference_config = hw_config.sch_config.inference
    model = _build_mla_model(run_type="inference")
    config.validate_configs(hw_config, model)
    tc = TimeCalculationLLMInference(hw_config, model, "LLM")
    decode_time, decode_energy, samples = tc.calc_decode_time()

    assert decode_time > 0.0
    assert decode_energy > 0.0
    assert len(samples) > 0


def test_mla_dense_vs_gqa_toy_training_and_inference_are_not_pathologically_reversed():
    train_hw = _build_hw_config()
    mla_train_model = _build_mla_model()
    gqa_train_model = _build_gqa_model()
    config.validate_configs(train_hw, mla_train_model)
    config.validate_configs(train_hw, gqa_train_model)
    mla_train = TimeCalculationLLM(train_hw, mla_train_model, "LLM").calc_time_llm()
    gqa_train = TimeCalculationLLM(train_hw, gqa_train_model, "LLM").calc_time_llm()

    inf_hw = _build_hw_config()
    inf_hw.inference_config = inf_hw.sch_config.inference
    mla_inf_model = _build_mla_model(run_type="inference")
    gqa_inf_model = _build_gqa_model(run_type="inference")
    config.validate_configs(inf_hw, mla_inf_model)
    config.validate_configs(inf_hw, gqa_inf_model)
    mla_inf = TimeCalculationLLMInference(inf_hw, mla_inf_model, "LLM").calc_total_inference_time()["total_inference_time"]
    gqa_inf = TimeCalculationLLMInference(inf_hw, gqa_inf_model, "LLM").calc_total_inference_time()["total_inference_time"]

    assert mla_train < gqa_train
    assert mla_inf < gqa_inf
