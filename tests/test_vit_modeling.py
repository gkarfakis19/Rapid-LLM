from pathlib import Path

import config
import llm_util
from inference_timing import TimeCalculationLLMInference
from memory_estimation import MemKind, MemoryEstimator
from train_timing import TimeCalculationLLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HW_BASE = PROJECT_ROOT / "validation_scripts" / "validation_configs" / "hardware-config" / "a100_80GB.yaml"
TIMM_VIT_INF = PROJECT_ROOT / "configs" / "model-config" / "vit_7b_patch16_dinov3_lvd1689m_inf.yaml"


def _build_hw_config(*, tp: int = 1, cp: int = 1) -> config.HWConfig:
    hw_config = config.parse_config(str(HW_BASE), config_type="hardware")
    hw_config.sch_config.tp = tp
    hw_config.sch_config.cp = cp
    hw_config.sch_config.pp = 1
    hw_config.sch_config.mb = 1
    hw_config.sch_config.tp_sp = False
    hw_config.sch_config.train.dp = 1
    hw_config.sch_config.train.ep = 1
    hw_config.sch_config.train.tp_ep = True
    hw_config.sch_config.inference.replica_count = 1
    hw_config.sch_config.inference.moe_dp = 1
    return hw_config


def _build_vit_model(
    *,
    model_type: str = "vit",
    run_type: str = "training",
    num_classes: int = 1000,
    use_flashattention: bool = False,
    seq_len=None,
) -> config.ModelConfig:
    model_param = {
        "mode": "ViT",
        "run_type": run_type,
        "tied_embeddings": True,
        "model_type": model_type,
        "disable_embedding_unembedding": False,
        "global_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "decode_len": 0,
        "hidden_dim": 64,
        "intermediate_size": 128,
        "vocab_size": num_classes,
        "vision": {
            "image_size": 32,
            "patch_size": 8,
        },
        "attention": {
            "attention_type": "mha",
            "num_heads": 4,
            "use_flashattention": use_flashattention,
            "attention_tile_size": 32,
        },
        "num_layers": 2,
        "moe": {
            "num_experts": 1,
            "top_k": 1,
            "moe_intermediate_size": 0,
            "n_shared_experts": 0,
            "moe_layer_freq": 1,
            "first_k_dense_replace": 0,
        },
    }
    if seq_len is not None:
        model_param["seq_len"] = seq_len
    llm_config = config.LLMConfig.from_dict(model_param)
    inference_cfg = config.LLMInferenceConfig(sample_every=-1) if run_type == "inference" else None
    return config.ModelConfig(model_config=llm_config, inference_config=inference_cfg)


def test_vit_training_support_and_family_default_geometry():
    vit = _build_vit_model(run_type="training").model_config
    dinov3 = _build_vit_model(model_type="vit_dinov3", run_type="training", num_classes=0).model_config

    assert vit.seq_len == 17
    assert vit.num_patches == 16
    assert vit.num_prefix_tokens == 1
    assert vit.patch_dim == 8 * 8 * 3
    assert vit.swiglu_mlp is False

    assert dinov3.seq_len == 21
    assert dinov3.num_prefix_tokens == 5
    assert dinov3.swiglu_mlp is True


def test_vit_patch_embed_and_head_params_are_nonzero():
    hw_config = _build_hw_config()
    model = _build_vit_model(run_type="training", num_classes=17)
    config.validate_configs(hw_config, model)
    tc = TimeCalculationLLM(hw_config, model, "VIT")

    _total, _max_layer, _layer, embedding_params, output_params = tc._param_stats_per_rank(
        tc.hidden_dim,
        tc.intermediate_size,
        tc.vocab_size,
    )

    expected_embedding = llm_util.vit_patch_embed_param_count(
        hidden_dim=tc.hidden_dim,
        patch_size=tc.patch_size,
        in_chans=tc.in_chans,
    )
    expected_head = llm_util.vit_head_param_count(
        hidden_dim=tc.hidden_dim,
        num_classes=tc.num_classes,
    )

    assert embedding_params == expected_embedding
    assert output_params == expected_head
    assert embedding_params > 0
    assert output_params > 0


def test_vit_training_entry_and_exit_stages_have_nonzero_cost():
    hw_config = _build_hw_config()
    model = _build_vit_model(run_type="training", num_classes=0)
    config.validate_configs(hw_config, model)
    tc = TimeCalculationLLM(hw_config, model, "VIT")

    transformer_timings, _ = tc.compute_all_gemm_and_node_times(
        batch_size=tc._effective_transformer_batch(),
        vocab_size=tc.vocab_size,
        hidden_dim=tc.hidden_dim,
        seq_len=tc.seq_len,
        num_heads=tc.num_heads,
        kv_heads=tc.kv_heads,
        intermediate_size=tc.intermediate_size,
        num_SMs=tc.hw_config.tech_config.core.num_bundles,
    )

    assert transformer_timings["embedding"].total_forward_time() > 0
    assert transformer_timings["embedding"].total_backward_time() > 0
    assert transformer_timings["linear_softmax"].total_forward_time() > 0
    assert transformer_timings["linear_softmax"].total_backward_time() > 0


def test_vit_inference_has_zero_kv_cache_and_nonzero_prefill_time():
    hw_config = _build_hw_config()
    hw_config.inference_config = hw_config.sch_config.inference
    model = _build_vit_model(run_type="inference", num_classes=0, use_flashattention=True)
    config.validate_configs(hw_config, model)
    tc = TimeCalculationLLMInference(hw_config, model, "VIT")

    result = tc.calc_total_inference_time()

    assert tc.get_kv_size_bytes() == 0
    assert result["prefill_time"] > 0
    assert result["decode_time"] == 0
    assert result["kv_cache_prefill_store_bytes"] == 0


def test_vit_memory_estimator_tracks_embedding_and_head_memory():
    hw_config = _build_hw_config()
    model = _build_vit_model(run_type="training", num_classes=10)
    config.validate_configs(hw_config, model)
    tc = TimeCalculationLLM(hw_config, model, "VIT")

    memory_data = MemoryEstimator(tc).build_memory_data(
        mode="training",
        batch_size=tc._effective_transformer_batch(),
        seq_len=tc.seq_len,
    )

    assert memory_data["persistent_bytes_by_kind"][MemKind.EMBEDDING] > 0
    assert memory_data["persistent_bytes_by_kind"][MemKind.SOFTMAX] > 0
    assert memory_data["activation_mem_per_layer"] > 0


def test_vit_context_parallelism_pooling_tracks_local_patch_tokens():
    hw_cp = _build_hw_config(cp=2)
    model = _build_vit_model(run_type="training", num_classes=0)
    config.validate_configs(hw_cp, model)
    tc_cp = TimeCalculationLLM(hw_cp, model, "VIT")

    assert tc_cp._vit_pool_token_count() == 8

    hw_single = _build_hw_config(cp=1)
    config.validate_configs(hw_single, model)
    tc_single = TimeCalculationLLM(hw_single, model, "VIT")
    cp_pool_time, _ = tc_cp._vit_pool_time(
        batch=tc_cp._effective_transformer_batch(),
        hidden_dim=tc_cp.hidden_dim,
        backward=False,
    )
    single_pool_time, _ = tc_single._vit_pool_time(
        batch=tc_single._effective_transformer_batch(),
        hidden_dim=tc_single.hidden_dim,
        backward=False,
    )
    assert cp_pool_time > single_pool_time


def test_vit_rejects_extra_vision_keys():
    model_param = {
        "mode": "ViT",
        "run_type": "training",
        "tied_embeddings": True,
        "model_type": "vit",
        "global_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "hidden_dim": 64,
        "intermediate_size": 128,
        "vocab_size": 1000,
        "vision": {
            "image_size": 32,
            "patch_size": 8,
            "rope_temperature": 100,
        },
        "attention": {
            "attention_type": "mha",
            "num_heads": 4,
            "use_flashattention": False,
        },
        "num_layers": 2,
        "moe": {
            "num_experts": 1,
            "top_k": 1,
            "moe_intermediate_size": 0,
            "n_shared_experts": 0,
            "moe_layer_freq": 1,
            "first_k_dense_replace": 0,
        },
    }

    try:
        config.LLMConfig.from_dict(model_param)
    except ValueError as exc:
        assert "only supports image_size and patch_size" in str(exc)
    else:
        raise AssertionError("Expected extra ViT vision keys to be rejected")


def test_timm_vit_7b_dinov3_config_parses_with_expected_geometry():
    model = config.parse_config(str(TIMM_VIT_INF), config_type="VIT")
    cfg = model.model_config

    assert cfg.model_type == "vit_dinov3"
    assert cfg.seq_len == 261
    assert cfg.num_patches == 256
    assert cfg.num_prefix_tokens == 5
    assert cfg.intermediate_size == 8192
    assert cfg.hidden_dim == 4096
    assert cfg.num_heads == 32
    assert cfg.num_classes == 0
    assert cfg.swiglu_mlp is True


def test_vit_seq_len_override_must_cover_derived_tokens():
    model = _build_vit_model(run_type="training", seq_len=18).model_config
    assert model.seq_len == 18

    try:
        _build_vit_model(model_type="vit_dinov3", run_type="training", seq_len=16)
    except ValueError as exc:
        assert "ViT-derived token count" in str(exc)
    else:
        raise AssertionError("Expected too-small ViT seq_len override to be rejected")
