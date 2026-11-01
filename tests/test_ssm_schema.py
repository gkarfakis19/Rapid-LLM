from pathlib import Path

import pytest
import yaml

from config import LLMConfig
from tools.convert_ssm_config import convert_model


@pytest.fixture
def model():
    return yaml.safe_load(Path("configs/model-config/mamba2_base_inf.yaml").read_text())


@pytest.mark.parametrize("key,target", [
    ("state_size", "ssm.d_state"), ("conv_kernel", "ssm.d_conv"),
    ("num_heads", "ssm.n_heads"), ("headdim", "ssm.d_head"),
    ("n_groups", "ssm.n_groups"), ("expand", "ssm.expand"),
    ("hybrid_schedule", "layer_plan"), ("linear", "ssm.variant"),
])
def test_old_top_level_keys_refused(model, key, target):
    model["model_param"][key] = 1
    with pytest.raises(ValueError, match=target):
        LLMConfig.from_dict(model["model_param"])


@pytest.mark.parametrize("key,target", [("state_size", "d_state"),
    ("conv_kernel", "d_conv"), ("num_heads", "n_heads"), ("headdim", "d_head")])
def test_old_nested_keys_refused(model, key, target):
    model["model_param"]["ssm"][key] = 1
    with pytest.raises(ValueError, match=target):
        LLMConfig.from_dict(model["model_param"])


def test_converter_intended_interval_zero(model):
    mp = model["model_param"]
    mp.pop("layer_plan")
    mp["model_type"] = "mamba2_ssm"
    mp["hybrid_schedule"] = {"full_attention_interval": 0}
    mp["linear"] = {"linear_type": "mamba2", "expand": 2, "state_size": 128,
                    "conv_kernel": 4, "num_heads": 24, "headdim": 64, "n_groups": 8}
    mp.pop("ssm")
    converted = convert_model(model)
    parsed = LLMConfig.from_dict(converted["model_param"])
    assert parsed.layer_mixers == (("ssm",),) * 24
    assert parsed.ssm.resolve_d_inner(768) == 1536
    assert parsed.ssm.chunk_size == 256


@pytest.mark.parametrize("name", ["mamba2_base_inf", "mamba2_hf_inf",
    "Granite-4.0-H-Small-32B_inf", "Jamba2-Mini-52B-A12B_inf"])
def test_converted_models_parse(name):
    data = yaml.safe_load(Path(f"configs/model-config/{name}.yaml").read_text())
    parsed = LLMConfig.from_dict(data["model_param"])
    assert parsed.run_type == "inference"
    assert parsed.layer_plan.count("ssm") > 0
