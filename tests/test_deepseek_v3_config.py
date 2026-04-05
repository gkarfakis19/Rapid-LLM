from argparse import Namespace
import importlib.util
from pathlib import Path

import config
import llm_util
from inference_timing import TimeCalculationLLMInference
from train_timing import TimeCalculationLLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HW_BASE = PROJECT_ROOT / "validation_scripts" / "validation_configs" / "hardware-config" / "a100_80GB.yaml"
HF_TO_CONFIG_PATH = PROJECT_ROOT / "configs" / "model-config" / "hf_to_config.py"
DEEPSEEK_TRAIN = PROJECT_ROOT / "configs" / "model-config" / "DeepSeekV3.yaml"
DEEPSEEK_INF = PROJECT_ROOT / "configs" / "model-config" / "DeepSeekV3_inf.yaml"


def _load_hf_to_config_module():
    spec = importlib.util.spec_from_file_location("hf_to_config_module", HF_TO_CONFIG_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _build_hw_config():
    hw_config = config.parse_config(str(HW_BASE), config_type="hardware")
    hw_config.sch_config.tp = 1
    hw_config.sch_config.cp = 1
    hw_config.sch_config.pp = 1
    hw_config.sch_config.mb = 1
    hw_config.sch_config.tp_sp = False
    hw_config.sch_config.train.dp = 1
    hw_config.sch_config.train.ep = 1
    hw_config.sch_config.train.tp_ep = True
    hw_config.sch_config.inference.replica_count = 1
    hw_config.sch_config.inference.moe_dp = 1
    hw_config.execution_backend.model = "analytical"
    hw_config.execution_backend.astra = None
    for dim in hw_config.network_layout.dimensions:
        object.__setattr__(dim, "topology_type", "Ring")
    return hw_config


def _deepseek_v3_hf_config():
    return {
        "architectures": ["DeepseekV3ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "auto_map": {
            "AutoConfig": "configuration_deepseek.DeepseekV3Config",
            "AutoModel": "modeling_deepseek.DeepseekV3Model",
            "AutoModelForCausalLM": "modeling_deepseek.DeepseekV3ForCausalLM",
        },
        "bos_token_id": 0,
        "eos_token_id": 1,
        "ep_size": 1,
        "first_k_dense_replace": 3,
        "hidden_act": "silu",
        "hidden_size": 7168,
        "initializer_range": 0.02,
        "intermediate_size": 18432,
        "kv_lora_rank": 512,
        "max_position_embeddings": 163840,
        "model_type": "deepseek_v3",
        "moe_intermediate_size": 2048,
        "moe_layer_freq": 1,
        "n_group": 8,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "norm_topk_prob": True,
        "num_attention_heads": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 61,
        "num_key_value_heads": 128,
        "num_nextn_predict_layers": 1,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
        "rms_norm_eps": 1e-6,
        "rope_scaling": {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "yarn",
        },
        "rope_theta": 10000,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "tie_word_embeddings": False,
        "topk_group": 4,
        "topk_method": "noaux_tc",
        "torch_dtype": "bfloat16",
        "transformers_version": "4.33.1",
        "use_cache": True,
        "v_head_dim": 128,
        "vocab_size": 129280,
    }


def _build_toy_deepseek_model(*, run_type: str = "training") -> config.ModelConfig:
    model_param = {
        "mode": "LLM",
        "run_type": run_type,
        "tied_embeddings": False,
        "model_type": "deepseek_v3",
        "global_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "seq_len": 8,
        "decode_len": 4 if run_type == "inference" else 0,
        "hidden_dim": 64,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "num_nextn_predict_layers": 1,
        "rope_scaling": {"type": "yarn", "factor": 4},
        "rope_theta": 10000,
        "use_cache": True,
        "attention": {
            "attention_type": "mla",
            "num_heads": 4,
            "kv_heads": 4,
            "kv_lora_rank": 4,
            "q_lora_rank": 8,
            "qk_nope_head_dim": 12,
            "qk_rope_head_dim": 4,
            "v_head_dim": 8,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "use_flashattention": True,
            "attention_tile_size": 64,
        },
        "intermediate_size": 128,
        "vocab_size": 512,
        "num_layers": 4,
        "moe": {
            "num_experts": 8,
            "top_k": 2,
            "moe_intermediate_size": 64,
            "n_shared_experts": 1,
            "expert_imbalance_factor": 1.0,
            "moe_layer_freq": 1,
            "first_k_dense_replace": 1,
            "n_group": 2,
            "topk_group": 1,
            "topk_method": "noaux_tc",
            "routed_scaling_factor": 2.5,
            "scoring_func": "sigmoid",
            "norm_topk_prob": True,
        },
    }
    llm_config = config.LLMConfig.from_dict(model_param)
    inference_cfg = config.LLMInferenceConfig(sample_every=-1) if run_type == "inference" else None
    return config.ModelConfig(model_config=llm_config, inference_config=inference_cfg)


def test_hf_to_config_maps_deepseek_v3_to_mla_and_moe():
    hf_to_config = _load_hf_to_config_module()
    args = Namespace(
        global_batch_size=16,
        gradient_accumulation_steps=1,
        seq_len=None,
        decode_len=1024,
        run_type="training",
        use_flashattention=True,
        flash_tile_size=128,
    )
    yaml_cfg = hf_to_config._build_yaml_config(_deepseek_v3_hf_config(), args, "deepseek_v3")
    model = yaml_cfg["model_param"]

    assert model["model_type"] == "deepseek_v3"
    assert model["seq_len"] == 163840
    assert model["num_layers"] == 61
    assert model["attention"]["attention_type"] == "mla"
    assert model["attention"]["kv_lora_rank"] == 512
    assert model["attention"]["q_lora_rank"] == 1536
    assert model["attention"]["qk_nope_head_dim"] == 128
    assert model["attention"]["qk_rope_head_dim"] == 64
    assert model["attention"]["v_head_dim"] == 128
    assert model["attention"]["use_flashattention"] is True
    assert model["moe"]["num_experts"] == 256
    assert model["moe"]["top_k"] == 8
    assert model["moe"]["moe_intermediate_size"] == 2048
    assert model["moe"]["n_shared_experts"] == 1
    assert "topk_group" not in model["moe"]
    assert "scoring_func" not in model["moe"]
    assert "num_nextn_predict_layers" not in model
    assert "rope_scaling" not in model


def test_deepseek_v3_yaml_configs_parse_and_validate():
    hw_config = _build_hw_config()
    train_model = config.parse_config(str(DEEPSEEK_TRAIN), config_type="LLM")
    inf_model = config.parse_config(str(DEEPSEEK_INF), config_type="LLM")

    config.validate_configs(hw_config, train_model)
    config.validate_configs(hw_config, inf_model)

    train = train_model.model_config
    infer = inf_model.model_config
    assert train.model_type == "deepseek_v3"
    assert infer.model_type == "deepseek_v3"
    assert train.attention.attention_type == "mla"
    assert infer.attention.attention_type == "mla"
    assert train.num_experts == 256
    assert infer.num_experts == 256
    assert train.top_k == 8
    assert infer.top_k == 8
    assert train.seq_len == 163840
    assert infer.seq_len == 163840
    assert infer.decode_len == 1024


def test_deepseek_v3_model_family_uses_llama_style_mlp():
    model = _build_toy_deepseek_model().model_config

    assert llm_util.is_llama_style(model.model_type)
    assert llm_util.uses_gated_mlp(model.model_type) is True
    assert model.attention.attention_type == "mla"
    assert model.num_experts == 8
    assert model.n_shared_experts == 1


def test_deepseek_v3_toy_training_and_inference_run():
    hw_config = _build_hw_config()
    train_model = _build_toy_deepseek_model()
    inf_model = _build_toy_deepseek_model(run_type="inference")

    config.validate_configs(hw_config, train_model)
    config.validate_configs(hw_config, inf_model)

    train_time = TimeCalculationLLM(hw_config, train_model, "LLM").calc_time_llm()
    infer_time = TimeCalculationLLMInference(hw_config, inf_model, "LLM").calc_total_inference_time()[
        "total_inference_time"
    ]

    assert train_time > 0.0
    assert infer_time > 0.0
