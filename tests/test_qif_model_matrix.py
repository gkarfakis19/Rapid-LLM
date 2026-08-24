"""QIF P1 model-matrix tests.

P1.1 the block-typed schema (layer plans, SSM / linear-attention / short-conv
parameter groups, attention window patterns, per-block FFN dims, depth-shared
weight groups), P1.2 the hf_to_config ingestion of those fields, P1.3 the
checked-in model YAMLs, P1.4 the clean rejection of block kinds nothing prices
yet, and P6.1 the raw-parameter consistency rows (D2 / ADJ-2).
"""

import argparse
import copy
import re
import importlib.util
from pathlib import Path

import pytest
import yaml

import config
import llm_util
from inference_timing import TimeCalculationLLMInference


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "configs" / "model-config"
HW_BASE = PROJECT_ROOT / "validation_scripts" / "validation_configs" / "hardware-config" / "a100_80GB.yaml"
HF_TO_CONFIG_PATH = MODEL_DIR / "hf_to_config.py"

#: Every YAML this plan adds, with the block kinds it is expected to declare.
MATRIX_YAMLS = {
    "granite_4_0_h_tiny": ("ssm",),
    "granite_4_0_h_1b": ("ssm",),
    "falcon_h1_7b": ("ssm",),
    "lfm2_2p6b": ("short_conv",),
    "qwen3_5_4b": ("linear_attn",),
    "minicpm3_4b": (),
    "falcon_mamba_7b": ("ssm",),
    "hunyuan_7b": (),
    "smollm3_3b": (),
    # Wave C carriers: the SUPPORTED sibling of each family whose only shipped
    # row was over the D2 cutoff, plus the first windowed-attention row.
    "falcon_h1_3b": ("ssm",),
    "hunyuan_4b": (),
    "gemma_3_4b": (),
}
#: The rows that must parse AND run on the existing (non-hybrid) paths.
RUNS_NOW_YAMLS = ("hunyuan_7b", "smollm3_3b", "hunyuan_4b")
#: D2 (ADJ-2) hard cutoff on RAW total parameters.
D2_PARAM_CUTOFF = 7_000_000_000


def _matrix_yaml_path(name):
    return MODEL_DIR / f"{name}_inf.yaml"


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _model_from_dict(model_param):
    model_dict = {"model_param": copy.deepcopy(model_param)}
    config.convert(model_dict)
    return config.LLMConfig.from_dict(model_dict["model_param"])


def _model_config_from_dict(model_param):
    """The parsed model wrapped as validate_model_config expects it."""
    return config.ModelConfig(
        model_config=_model_from_dict(model_param), inference_config=None
    )


def _load_hf_to_config_module():
    spec = importlib.util.spec_from_file_location("hf_to_config_module", HF_TO_CONFIG_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _import_args():
    return argparse.Namespace(
        run_type="inference",
        global_batch_size=1,
        gradient_accumulation_steps=1,
        seq_len=2048,
        decode_len=256,
        use_flashattention=False,
        flash_tile_size=None,
    )


def _build_hw_config():
    hw_config = config.parse_config(str(HW_BASE), config_type="hardware")
    hw_config.sch_config.tp = 1
    hw_config.sch_config.cp = 1
    hw_config.sch_config.pp = 1
    hw_config.sch_config.mb = 1
    hw_config.sch_config.tp_sp = False
    hw_config.sch_config.train.dp = 1
    hw_config.sch_config.train.ep = 1
    hw_config.sch_config.inference.replica_count = 1
    hw_config.sch_config.inference.moe_dp = 1
    hw_config.execution_backend.model = "analytical"
    hw_config.execution_backend.astra = None
    for dim in hw_config.network_layout.dimensions:
        object.__setattr__(dim, "topology_type", "Ring")
    return hw_config


#: A minimal dense GQA transformer to mutate in the schema tests.
BASE_MODEL_PARAM = {
    "mode": "LLM",
    "run_type": "inference",
    "tied_embeddings": True,
    "model_type": "llama",
    "global_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "seq_len": 256,
    "decode_len": 16,
    "hidden_dim": 512,
    "attention": {
        "attention_type": "gqa",
        "num_heads": 8,
        "kv_heads": 2,
        "use_flashattention": False,
    },
    "moe": {
        "num_experts": 1,
        "top_k": 1,
        "moe_intermediate_size": 1024,
        "n_shared_experts": 0,
        "moe_layer_freq": 1,
        "first_k_dense_replace": 8,
    },
    "intermediate_size": 1024,
    "vocab_size": 1000,
    "num_layers": 8,
}

SSM_BLOCK = {
    "variant": "mamba2",
    "d_state": 64,
    "n_groups": 1,
    "n_heads": 8,
    "d_head": 128,
    "expand": 2,
    "d_conv": 4,
    "chunk_size": 128,
}

LINEAR_ATTENTION_BLOCK = {
    "num_key_heads": 2,
    "key_head_dim": 64,
    "num_value_heads": 4,
    "value_head_dim": 64,
    "conv_kernel": 4,
}

SHORT_CONV_BLOCK = {"kernel_size": 3, "conv_dim": 512, "double_gated": True}


def _with(**overrides):
    model_param = copy.deepcopy(BASE_MODEL_PARAM)
    for key, value in overrides.items():
        if value is None:
            model_param.pop(key, None)
        else:
            model_param[key] = value
    return model_param


# ---------------------------------------------------------------------------
# P1.1 — layer plans
# ---------------------------------------------------------------------------


def test_pattern_tiles_the_depth():
    model = _model_from_dict(
        _with(layer_plan={"pattern": ["mamba", "mamba", "mamba", "attention"]}, ssm=SSM_BLOCK)
    )
    assert model.layer_plan.layer_types == ("ssm", "ssm", "ssm", "attention") * 2
    assert model.layer_plan.count("ssm") == 6
    assert model.layer_plan.count("attention") == 2
    assert model.block_kinds == ("attention", "ssm")


def test_explicit_layer_types_normalize_hf_spellings():
    layer_types = ["conv", "conv", "full_attention", "conv", "conv", "full_attention", "conv", "conv"]
    model = _model_from_dict(
        _with(layer_plan={"layer_types": layer_types}, short_conv=SHORT_CONV_BLOCK)
    )
    assert model.layer_plan.layer_types == (
        "short_conv", "short_conv", "attention", "short_conv",
        "short_conv", "attention", "short_conv", "short_conv",
    )
    assert model.hybrid_block_kinds == ("short_conv",)


def test_pattern_that_does_not_tile_is_rejected():
    with pytest.raises(ValueError, match="does not tile"):
        _model_from_dict(_with(layer_plan={"pattern": ["mamba", "mamba", "attention"]}, ssm=SSM_BLOCK))


def test_explicit_layer_types_length_must_match_num_layers():
    with pytest.raises(ValueError, match="layer_types"):
        _model_from_dict(_with(layer_plan={"layer_types": ["attention"] * 7}))


def test_layer_plan_requires_exactly_one_of_layer_types_or_pattern():
    with pytest.raises(ValueError, match="exactly one"):
        _model_from_dict(_with(layer_plan={}))
    with pytest.raises(ValueError, match="exactly one"):
        _model_from_dict(
            _with(layer_plan={"pattern": ["attention"], "layer_types": ["attention"] * 8})
        )


def test_unknown_block_kind_is_rejected():
    with pytest.raises(ValueError, match="block kind"):
        _model_from_dict(_with(layer_plan={"pattern": ["retention"]}))


def test_unknown_layer_plan_key_is_rejected():
    with pytest.raises(ValueError, match="layer_plan does not support"):
        _model_from_dict(_with(layer_plan={"pattern": ["attention"], "interval": 4}))


def test_dedicated_ffn_layer_needs_ffn_per_layer_false():
    with pytest.raises(ValueError, match="ffn_per_layer"):
        _model_from_dict(
            _with(layer_plan={"pattern": ["mamba", "ffn"]}, ssm=SSM_BLOCK)
        )
    model = _model_from_dict(
        _with(
            layer_plan={"pattern": ["mamba", "ffn"], "ffn_per_layer": False},
            ssm=SSM_BLOCK,
            attention=None,
        )
    )
    assert model.ffn_per_layer is False


def test_parallel_branch_repeats_the_template_on_every_layer():
    model = _model_from_dict(
        _with(
            layer_plan={"structure": "parallel_branch", "branches": ["attention", "mamba", "ffn"]},
            ssm=SSM_BLOCK,
        )
    )
    assert model.layer_plan.structure == "parallel_branch"
    assert model.layer_mixers == (("attention", "ssm", "ffn"),) * 8
    assert model.ffn_per_layer is False


def test_parallel_branch_rejects_layer_types_and_ffn_flag():
    with pytest.raises(ValueError, match="parallel_branch"):
        _model_from_dict(
            _with(layer_plan={"structure": "parallel_branch", "layer_types": ["attention"] * 8})
        )
    with pytest.raises(ValueError, match="ffn_per_layer"):
        _model_from_dict(
            _with(
                layer_plan={
                    "structure": "parallel_branch",
                    "branches": ["attention", "ffn"],
                    "ffn_per_layer": True,
                }
            )
        )


def test_branches_only_valid_for_parallel_branch():
    with pytest.raises(ValueError, match="branches"):
        _model_from_dict(_with(layer_plan={"pattern": ["attention"], "branches": ["attention"]}))


# ---------------------------------------------------------------------------
# P1.1 — block parameter groups and their cross-validation
# ---------------------------------------------------------------------------


def test_declared_block_kind_requires_its_parameter_group():
    with pytest.raises(ValueError, match="model_param.ssm must be specified"):
        _model_from_dict(_with(layer_plan={"pattern": ["mamba", "attention"]}))
    with pytest.raises(ValueError, match="model_param.linear_attention must be specified"):
        _model_from_dict(_with(layer_plan={"pattern": ["linear_attention", "attention"]}))
    with pytest.raises(ValueError, match="model_param.short_conv must be specified"):
        _model_from_dict(_with(layer_plan={"pattern": ["conv", "attention"]}))


def test_unreachable_block_group_is_rejected():
    with pytest.raises(ValueError, match="declares no 'ssm' block"):
        _model_from_dict(_with(layer_plan={"pattern": ["attention"]}, ssm=SSM_BLOCK))


def test_block_group_without_a_layer_plan_is_rejected():
    with pytest.raises(ValueError, match="requires model_param.layer_plan"):
        _model_from_dict(_with(ssm=SSM_BLOCK))


def test_attention_group_follows_the_layer_plan():
    model = _model_from_dict(
        _with(layer_plan={"pattern": ["mamba"], "ffn_per_layer": False}, ssm=SSM_BLOCK, attention=None)
    )
    assert model.attention is None
    with pytest.raises(ValueError, match="model_param.attention must be specified"):
        _model_from_dict(_with(layer_plan={"pattern": ["mamba", "attention"]}, ssm=SSM_BLOCK, attention=None))
    with pytest.raises(ValueError, match="declares no attention block"):
        _model_from_dict(
            _with(layer_plan={"pattern": ["mamba"], "ffn_per_layer": False}, ssm=SSM_BLOCK)
        )


def test_attention_still_required_without_a_layer_plan():
    with pytest.raises(ValueError, match="model_param.attention must be specified"):
        _model_from_dict(_with(attention=None))


def test_ssm_mamba2_requires_the_head_structure():
    block = {key: value for key, value in SSM_BLOCK.items() if key != "chunk_size"}
    with pytest.raises(ValueError, match="chunk_size"):
        _model_from_dict(_with(layer_plan={"pattern": ["mamba", "attention"]}, ssm=block))


def test_ssm_mamba1_requires_dt_rank_and_rejects_head_fields():
    with pytest.raises(ValueError, match="dt_rank"):
        _model_from_dict(
            _with(
                layer_plan={"pattern": ["mamba"], "ffn_per_layer": False},
                ssm={"variant": "mamba1", "d_state": 16, "d_conv": 4, "d_inner": 1024},
                attention=None,
            )
        )
    with pytest.raises(ValueError, match="no head structure"):
        _model_from_dict(
            _with(
                layer_plan={"pattern": ["mamba"], "ffn_per_layer": False},
                ssm={
                    "variant": "mamba1", "d_state": 16, "d_conv": 4,
                    "d_inner": 1024, "dt_rank": 64, "n_heads": 8,
                },
                attention=None,
            )
        )


def test_ssm_width_is_stated_once():
    with pytest.raises(ValueError, match="d_inner OR expand"):
        _model_from_dict(
            _with(layer_plan={"pattern": ["mamba", "attention"]}, ssm={**SSM_BLOCK, "d_inner": 1024})
        )
    with pytest.raises(ValueError, match="requires d_inner or expand"):
        block = {key: value for key, value in SSM_BLOCK.items() if key != "expand"}
        _model_from_dict(_with(layer_plan={"pattern": ["mamba", "attention"]}, ssm=block))


def test_ssm_head_width_must_equal_d_inner():
    with pytest.raises(ValueError, match="n_heads \\* d_head"):
        _model_from_dict(
            _with(layer_plan={"pattern": ["mamba", "attention"]}, ssm={**SSM_BLOCK, "n_heads": 7})
        )


def test_ssm_d_inner_wins_over_expand_when_stated():
    model = _model_from_dict(
        _with(
            layer_plan={"pattern": ["mamba", "attention"]},
            ssm={**{k: v for k, v in SSM_BLOCK.items() if k != "expand"}, "d_inner": 1024},
        )
    )
    assert model.ssm.resolve_d_inner(model.hidden_dim) == 1024


def test_linear_attention_key_heads_must_divide_value_heads():
    with pytest.raises(ValueError, match="must divide"):
        _model_from_dict(
            _with(
                layer_plan={"pattern": ["linear_attention"]},
                linear_attention={**LINEAR_ATTENTION_BLOCK, "num_key_heads": 3},
            )
        )


def test_linear_attention_dims_and_gate_defaults():
    model = _model_from_dict(
        _with(
            layer_plan={"pattern": ["linear_attention", "attention"]},
            linear_attention=LINEAR_ATTENTION_BLOCK,
        )
    )
    assert model.linear_attention.key_dim == 128
    assert model.linear_attention.value_dim == 256
    assert model.linear_attention.output_gate is True
    assert model.linear_attention.decay_gate is True


def test_short_conv_double_gating_flag():
    gated = _model_from_dict(
        _with(layer_plan={"pattern": ["conv", "attention"]}, short_conv=SHORT_CONV_BLOCK)
    )
    assert gated.short_conv.in_proj_streams == 3
    plain = _model_from_dict(
        _with(
            layer_plan={"pattern": ["conv", "attention"]},
            short_conv={**SHORT_CONV_BLOCK, "double_gated": False},
        )
    )
    assert plain.short_conv.in_proj_streams == 1


def test_attention_window_pattern_parses():
    attention = {
        **BASE_MODEL_PARAM["attention"],
        "window": {"window_size": 512, "local_global_interval": 6, "last_layer_global": True},
    }
    model = _model_from_dict(_with(attention=attention))
    assert model.attention.window.window_size == 512
    assert model.attention.window.local_global_interval == 6
    assert model.attention.window.last_layer_global is True


def test_attention_window_defaults_and_strict_keys():
    attention = {**BASE_MODEL_PARAM["attention"], "window": {"window_size": 128}}
    model = _model_from_dict(_with(attention=attention))
    assert model.attention.window.local_global_interval == 1
    assert model.attention.window.last_layer_global is False
    bad = {**BASE_MODEL_PARAM["attention"], "window": {"window_size": 128, "stride": 2}}
    with pytest.raises(ValueError, match="window does not support"):
        _model_from_dict(_with(attention=bad))


def test_attention_output_gate_flag():
    assert _model_from_dict(_with()).attention.output_gate is False
    gated = _model_from_dict(
        _with(attention={**BASE_MODEL_PARAM["attention"], "output_gate": True})
    )
    assert gated.attention.output_gate is True


def test_a_declared_head_dim_is_authoritative_and_is_honored():
    """A declared head_dim may decouple from hidden_dim/num_heads, on any plan.

    The old rule refused a declared head_dim that differed from
    hidden_dim // num_heads unless the model also declared a HYBRID layer plan.
    The model window D3 covers breaks that rule on plain transformers: Gemma 3
    4B is 2560 / 8 heads with head_dim 256 and Hunyuan-4B is 3072 / 32 heads
    with head_dim 128, and neither declares a layer plan of any kind — so the
    field was refusable on exactly the models it exists for.

    What the guard was standing in for is that the declared value must actually
    be USED, and that is what this asserts instead: q / k / v / o are sized from
    num_heads * head_dim everywhere, so a decoupled declaration changes the
    parameter census and the placed stage shapes rather than being ignored.
    """
    attention = {**BASE_MODEL_PARAM["attention"], "head_dim": 256}
    # 512 / 8 = 64, so 256 is a 4x decoupling on a plain attention-only model.
    model = _model_from_dict(_with(attention=attention))
    assert model.attention.head_dim == 256
    derived = _model_from_dict(_with())
    assert derived.attention.head_dim in (None, 64)

    # It is HONORED by the census: q_size and kv_size follow the declaration.
    _head_dim, q_size, kv_size = llm_util.attention_dim_sizes(
        512, 8, 2, head_dim=model.attention.head_dim
    )
    assert (q_size, kv_size) == (2048, 512)
    assert llm_util.attention_block_param_count(
        hidden_dim=512, attention=model.attention
    ) == 512 * 2048 + 2 * 512 * 512 + 2048 * 512

    # ... and it still decouples under a hybrid plan, which is where the
    # relaxation started (Qwen3.5, Falcon-H1).
    hybrid = _model_from_dict(
        _with(attention=attention, layer_plan={"pattern": ["linear_attention", "full_attention"]},
              linear_attention=LINEAR_ATTENTION_BLOCK)
    )
    assert hybrid.attention.head_dim == 256

    # A model that declares NO head_dim still has to be divisible, because that
    # is the only case where the value is derived.
    with pytest.raises(ValueError, match="divisible by attention.num_heads"):
        _model_from_dict(
            _with(hidden_dim=513, attention={**BASE_MODEL_PARAM["attention"]})
        )


def test_ffn_dims_per_block_kind():
    model = _model_from_dict(
        _with(
            layer_plan={"pattern": ["mamba", "attention"]},
            ssm=SSM_BLOCK,
            ffn_dims={"ssm": 2048, "shared_expert": 512},
        )
    )
    assert model.ffn_dim_for("ssm") == 2048
    assert model.ffn_dim_for("attention") == 1024  # falls back to intermediate_size
    with pytest.raises(ValueError, match="ffn_dims does not support"):
        _model_from_dict(_with(ffn_dims={"retention": 128}))


def test_shared_weight_groups_parse_and_validate():
    model = _model_from_dict(
        _with(shared_weight_groups=[{"name": "shared_attn", "layers": [1, 3, 5], "lora_rank": 64}])
    )
    group = model.shared_weight_groups[0]
    assert group.name == "shared_attn"
    assert group.layers == (1, 3, 5)
    assert group.lora_rank == 64

    with pytest.raises(ValueError, match="out of range"):
        _model_from_dict(_with(shared_weight_groups=[{"name": "g", "layers": [1, 99]}]))
    with pytest.raises(ValueError, match="at least two"):
        _model_from_dict(_with(shared_weight_groups=[{"name": "g", "layers": [1]}]))
    with pytest.raises(ValueError, match="at most one share group"):
        _model_from_dict(
            _with(
                shared_weight_groups=[
                    {"name": "a", "layers": [1, 2]},
                    {"name": "b", "layers": [2, 3]},
                ]
            )
        )


def test_vit_rejects_the_block_typed_schema():
    vit = {
        "mode": "ViT",
        "run_type": "inference",
        "tied_embeddings": False,
        "model_type": "vit",
        "global_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "decode_len": 0,
        "hidden_dim": 768,
        "attention": {"attention_type": "mha", "num_heads": 12, "use_flashattention": False},
        "moe": {"num_experts": 1, "top_k": 1, "moe_intermediate_size": 3072,
                "n_shared_experts": 0, "moe_layer_freq": 1, "first_k_dense_replace": 12},
        "intermediate_size": 3072,
        "vocab_size": 0,
        "num_layers": 12,
        "vision": {"image_size": 224, "patch_size": 16},
        "layer_plan": {"pattern": ["attention"]},
    }
    with pytest.raises(ValueError, match="uniform encoders"):
        _model_from_dict(vit)


def test_uniform_configs_are_untouched_by_the_new_schema():
    model = config.parse_config(str(MODEL_DIR / "llama2_7b_fws_inf.yaml"), "LLM").model_config
    assert model.layer_plan is None
    assert model.ssm is None
    assert model.block_kinds == ("attention",)
    assert model.hybrid_block_kinds == ()
    assert model.has_hybrid_blocks is False
    assert model.ffn_per_layer is True
    assert model.layer_mixers == (("attention",),) * model.num_layers


# ---------------------------------------------------------------------------
# P1.3 — the checked-in model YAMLs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(MATRIX_YAMLS))
def test_matrix_yaml_parses_with_expected_block_kinds(name):
    model = config.parse_config(str(_matrix_yaml_path(name)), "LLM").model_config
    assert model.run_type == "inference"
    assert model.hybrid_block_kinds == MATRIX_YAMLS[name]


@pytest.mark.parametrize("name", sorted(MATRIX_YAMLS))
def test_matrix_yaml_records_its_published_source(name):
    provenance = _load_yaml(_matrix_yaml_path(name))["provenance"]
    assert provenance["source"].startswith("https://huggingface.co/")
    # Every row records the DAY its dims were read off the published config.
    # Wave A verified its rows on 2026-08-23 and Wave C its carriers on
    # 2026-08-24; a row with no date, or a date before the matrix existed, is
    # a row nobody checked.
    assert re.fullmatch(r"20\d\d-\d\d-\d\d", str(provenance["verified"]))
    assert str(provenance["verified"]) >= "2026-08-23"
    assert provenance["d2_status"] in {"supported", "reference"}
    assert int(provenance["published_total_params"]) > 0


def test_headline_model_is_granite_4_0_h_tiny():
    # ADJ-1: Granite-4.0-H-Tiny is the P1 headline model.
    model = config.parse_config(str(_matrix_yaml_path("granite_4_0_h_tiny")), "LLM").model_config
    assert model.num_layers == 40
    assert model.hidden_dim == 1536
    assert model.layer_plan.count("attention") == 4
    assert model.layer_plan.count("ssm") == 36
    assert model.ssm.variant == "mamba2"
    assert model.ssm.d_state == 128
    assert model.ssm.chunk_size == 256
    assert model.num_experts == 64
    assert model.top_k == 6
    assert model.ffn_dim_for("shared_expert") == 1024


def test_falcon_h1_is_a_parallel_branch_model():
    model = config.parse_config(str(_matrix_yaml_path("falcon_h1_7b")), "LLM").model_config
    assert model.layer_plan.structure == "parallel_branch"
    assert model.layer_mixers[0] == ("attention", "ssm", "ffn")
    assert model.attention.head_dim == 128  # decoupled from 3072/12
    assert model.ssm.resolve_d_inner(model.hidden_dim) == 3072


def test_attention_only_properties_refuse_a_pure_recurrence_config_by_name():
    # A pure-recurrence stack legally has attention = None, so num_heads and
    # head_dim have no answer. They used to raise a bare
    # "'NoneType' object has no attribute 'num_heads'", which names neither the
    # config nor the reason.
    model = config.parse_config(str(_matrix_yaml_path("falcon_mamba_7b")), "LLM").model_config
    assert model.attention is None
    for name in ("num_heads", "head_dim"):
        with pytest.raises(config.NoAttentionBlockError, match="has no attention block"):
            getattr(model, name)
    # AttributeError is one of its bases ON PURPOSE: every getattr/hasattr probe
    # in the legacy timing path keeps swallowing it, so no priced number moves.
    assert getattr(model, "head_dim", None) is None
    assert not hasattr(model, "num_heads")
    # And a model that HAS attention is untouched.
    attentive = config.parse_config(str(_matrix_yaml_path("falcon_h1_7b")), "LLM").model_config
    assert attentive.num_heads > 0 and attentive.head_dim == 128


def test_falcon_mamba_is_pure_ssm():
    model = config.parse_config(str(_matrix_yaml_path("falcon_mamba_7b")), "LLM").model_config
    assert model.attention is None
    assert model.ssm.variant == "mamba1"
    assert model.ffn_per_layer is False
    assert set(model.block_kinds) == {"ssm"}


def test_minicpm3_is_the_mla_carrier():
    model = config.parse_config(str(_matrix_yaml_path("minicpm3_4b")), "LLM").model_config
    assert model.layer_plan is None
    assert model.attention.attention_type == "mla"
    assert model.kv_lora_rank == 256
    assert model.q_lora_rank == 768
    assert model.v_head_dim == 64


# ---------------------------------------------------------------------------
# P6.1 — raw parameter totals (D2 total, ADJ-2 raw)
# ---------------------------------------------------------------------------


#: The P6.1 gate: the census must land within this of the published total.
PARAM_TOLERANCE_DEFAULT = 0.03
#: And no row may widen it past this. The widened rows exist because a
#: published figure counts parts this repo does not model (a vision tower, an
#: MTP head) — a bounded, ARITHMETIC discrepancy. Past ~15% the check stops
#: distinguishing "the census is right and the card counts more" from "the
#: census is wrong", so a row needing more is a modeling bug, not a tolerance.
PARAM_TOLERANCE_CAP = 0.15


def _published_tolerance(name, provenance):
    """The row's P6.1 tolerance, capped, and justified above the default."""
    raw = provenance.get("published_total_params_tolerance")
    if raw is None:
        return PARAM_TOLERANCE_DEFAULT
    tolerance = float(raw)
    assert tolerance >= PARAM_TOLERANCE_DEFAULT, (
        f"{name}: published_total_params_tolerance = {tolerance} is TIGHTER than the "
        f"{PARAM_TOLERANCE_DEFAULT:.0%} default; drop the field instead of restating it"
    )
    assert tolerance <= PARAM_TOLERANCE_CAP, (
        f"{name}: published_total_params_tolerance = {tolerance:.0%} exceeds the "
        f"{PARAM_TOLERANCE_CAP:.0%} cap. A gap that wide is a census defect, not a "
        "tolerance"
    )
    reason = provenance.get("published_total_params_tolerance_reason")
    assert isinstance(reason, str) and len(reason.strip()) >= 40, (
        f"{name}: a tolerance above the {PARAM_TOLERANCE_DEFAULT:.0%} default is a "
        "DISCLOSED RELAXATION (D21), so provenance must carry a "
        "published_total_params_tolerance_reason saying what the published figure "
        f"counts that this YAML does not (got {reason!r})"
    )
    return tolerance


@pytest.mark.parametrize("name", sorted(MATRIX_YAMLS))
def test_param_total_matches_published_figure(name):
    path = _matrix_yaml_path(name)
    provenance = _load_yaml(path)["provenance"]
    model = config.parse_config(str(path), "LLM").model_config
    computed = llm_util.model_raw_param_count(model)
    published = int(provenance["published_total_params"])
    # The default gate is 3%; a row may widen it only by recording the
    # relaxation in its own provenance block (honesty rule, AUDIT_optima),
    # and only up to a cap.
    tolerance = _published_tolerance(name, provenance)
    relative_error = abs(computed - published) / published
    assert relative_error <= tolerance, (
        f"{name}: census {computed:,} vs published {published:,} "
        f"({relative_error:.2%} > {tolerance:.0%})"
    )


def test_the_published_total_tolerance_is_capped_and_justified():
    # The field itself is the thing under test: it used to be an uncapped,
    # unexplained float that any row could set to 1.0 and pass forever.
    good = {"published_total_params_tolerance": 0.06, "published_total_params_tolerance_reason": "x" * 40}
    assert _published_tolerance("row", good) == 0.06
    assert _published_tolerance("row", {}) == PARAM_TOLERANCE_DEFAULT
    with pytest.raises(AssertionError, match="exceeds the"):
        _published_tolerance("row", dict(good, published_total_params_tolerance=0.5))
    with pytest.raises(AssertionError, match="tolerance_reason"):
        _published_tolerance("row", {"published_total_params_tolerance": 0.06})
    with pytest.raises(AssertionError, match="tolerance_reason"):
        _published_tolerance("row", dict(good, published_total_params_tolerance_reason="too short"))
    with pytest.raises(AssertionError, match="TIGHTER"):
        _published_tolerance("row", dict(good, published_total_params_tolerance=0.01))


@pytest.mark.parametrize("name", sorted(MATRIX_YAMLS))
def test_every_widened_tolerance_row_is_disclosed(name):
    # Every shipped row, not just the two that widen today.
    _published_tolerance(name, _load_yaml(_matrix_yaml_path(name))["provenance"])


@pytest.mark.parametrize("name", sorted(MATRIX_YAMLS))
def test_param_total_agrees_with_the_d2_status(name):
    path = _matrix_yaml_path(name)
    provenance = _load_yaml(path)["provenance"]
    model = config.parse_config(str(path), "LLM").model_config
    computed = llm_util.model_raw_param_count(model)
    if provenance["d2_status"] == "supported":
        assert computed <= D2_PARAM_CUTOFF, f"{name}: {computed:,} raw params exceeds the D2 cutoff"
    else:
        assert computed > D2_PARAM_CUTOFF, (
            f"{name}: {computed:,} raw params is within the D2 cutoff, so it is not a reference row"
        )


def test_param_census_counts_a_depth_shared_group_once():
    plain = _model_from_dict(_with())
    shared = _model_from_dict(
        _with(shared_weight_groups=[{"name": "g", "layers": [1, 2, 3]}])
    )
    per_layer = llm_util.attention_block_param_count(
        hidden_dim=plain.hidden_dim, attention=plain.attention
    ) + llm_util.ffn_param_count(
        hidden_dim=plain.hidden_dim, intermediate_size=plain.intermediate_size, gated=True
    )
    expected = llm_util.model_raw_param_count(plain) - 2 * per_layer
    assert llm_util.model_raw_param_count(shared) == expected


def test_param_census_charges_the_shared_group_lora():
    shared = _model_from_dict(
        _with(shared_weight_groups=[{"name": "g", "layers": [1, 2], "lora_rank": 64}])
    )
    bare = _model_from_dict(_with(shared_weight_groups=[{"name": "g", "layers": [1, 2]}]))
    assert llm_util.model_raw_param_count(shared) - llm_util.model_raw_param_count(bare) == (
        2 * shared.hidden_dim * 64
    )


def test_param_census_rejects_vit_configs():
    model = config.parse_config(str(MODEL_DIR / "vit_base_inf.yaml"), "VIT").model_config
    with pytest.raises(ValueError, match="ViT"):
        llm_util.model_raw_param_count(model)


# ---------------------------------------------------------------------------
# P1.4 — clean rejection, and the rows that must still run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", sorted(key for key, kinds in MATRIX_YAMLS.items() if kinds)
)
def test_hybrid_layer_plans_are_rejected_on_the_gpu_path(name):
    hw_config = _build_hw_config()
    model = config.parse_config(str(_matrix_yaml_path(name)), "LLM")
    with pytest.raises(ValueError) as excinfo:
        config.validate_configs(hw_config, model)
    message = str(excinfo.value)
    assert "layer_plan" in message
    assert "P2" in message and "P4" in message
    for kind in MATRIX_YAMLS[name]:
        assert kind in message


def test_attention_only_layer_plan_is_not_rejected():
    # SmolLM3 declares a layer plan whose every entry is attention; a plan is
    # not a hybrid, so it must still validate.
    hw_config = _build_hw_config()
    model = config.parse_config(str(_matrix_yaml_path("smollm3_3b")), "LLM")
    assert model.model_config.layer_plan is not None
    config.validate_configs(hw_config, model)


#: Modeling inputs the schema accepts and no timing path prices (P1.4).
UNPRICED_INPUTS = {
    "attention.window": {
        "attention": {"window": {"window_size": 128, "local_global_interval": 4}},
    },
    "attention.output_gate": {"attention": {"output_gate": True}},
    "shared_weight_groups": {
        "shared_weight_groups": [{"name": "shared_attn", "layers": [1, 2]}]
    },
    "ffn_dims": {"ffn_dims": {"default": 999}},
}


@pytest.mark.parametrize("field", sorted(UNPRICED_INPUTS))
def test_unpriced_modeling_inputs_are_refused_by_name(field):
    # The other half of the P1.4 rule. A field that parses, validates and is
    # then ignored by every pricing path is the same wrong-number-rather-than-
    # missing-one failure the hybrid gate refuses, so it is refused too.
    hw_config = _build_hw_config()
    overrides = UNPRICED_INPUTS[field]
    model_param = dict(BASE_MODEL_PARAM)
    if "attention" in overrides:
        model_param["attention"] = {**BASE_MODEL_PARAM["attention"], **overrides["attention"]}
    for key, value in overrides.items():
        if key != "attention":
            model_param[key] = value
    model = _model_config_from_dict(model_param)
    with pytest.raises(ValueError) as excinfo:
        config.validate_model_config(hw_config, model)
    message = str(excinfo.value)
    assert field.split(".")[-1] in message
    assert "P2" in message and "P4" in message


def test_an_inert_window_is_not_refused():
    # local_global_interval = 1 makes every layer global, which is what the
    # existing law already prices. Refusing it would be a missing number.
    hw_config = _build_hw_config()
    model_param = dict(BASE_MODEL_PARAM)
    model_param["attention"] = {
        **BASE_MODEL_PARAM["attention"],
        "window": {"window_size": 128, "local_global_interval": 1},
    }
    config.validate_model_config(hw_config, _model_config_from_dict(model_param))


@pytest.mark.parametrize("name", RUNS_NOW_YAMLS)
def test_runs_now_rows_actually_run(name):
    hw_config = _build_hw_config()
    model = config.parse_config(str(_matrix_yaml_path(name)), "LLM")
    config.validate_configs(hw_config, model)
    total = TimeCalculationLLMInference(hw_config, model, "LLM").calc_total_inference_time()
    assert total["total_inference_time"] > 0.0


# ---------------------------------------------------------------------------
# P1.2 — hf_to_config ingestion
# ---------------------------------------------------------------------------

GRANITE_H_TINY_HF = {
    "model_type": "granitemoehybrid",
    "hidden_size": 1536,
    "num_hidden_layers": 40,
    "num_attention_heads": 12,
    "num_key_value_heads": 4,
    "intermediate_size": 512,
    "shared_intermediate_size": 1024,
    "num_local_experts": 64,
    "num_experts_per_tok": 6,
    "vocab_size": 100352,
    "tie_word_embeddings": True,
    "mamba_d_state": 128,
    "mamba_n_groups": 1,
    "mamba_n_heads": 48,
    "mamba_d_head": 64,
    "mamba_expand": 2,
    "mamba_d_conv": 4,
    "mamba_chunk_size": 256,
    "max_position_embeddings": 131072,
    "layer_types": (
        ["mamba"] * 5 + ["attention"] + ["mamba"] * 9 + ["attention"]
        + ["mamba"] * 9 + ["attention"] + ["mamba"] * 9 + ["attention"] + ["mamba"] * 4
    ),
}

#: Granite-4.0-H-1B — the DENSE twin of the headline model, and the only place
#: the importer's num_local_experts == 0 branch is exercised: no routed experts,
#: no shared-expert ffn_dims entry, first_k_dense_replace forced to num_layers.
#: Verbatim from the published config.json (2026-08-24), minus the fields the
#: importer does not read.
GRANITE_H_1B_HF = {
    "model_type": "granitemoehybrid",
    "hidden_size": 1536,
    "num_hidden_layers": 40,
    "num_attention_heads": 12,
    "num_key_value_heads": 4,
    "intermediate_size": 4096,
    "shared_intermediate_size": 4096,
    "num_local_experts": 0,
    "num_experts_per_tok": 0,
    "vocab_size": 100352,
    "tie_word_embeddings": True,
    "mamba_d_state": 128,
    "mamba_n_groups": 1,
    "mamba_n_heads": 48,
    "mamba_d_head": 64,
    "mamba_expand": 2,
    "mamba_d_conv": 4,
    "mamba_chunk_size": 256,
    "max_position_embeddings": 131072,
    "layer_types": (
        ["mamba"] * 5 + ["attention"] + ["mamba"] * 9 + ["attention"]
        + ["mamba"] * 9 + ["attention"] + ["mamba"] * 9 + ["attention"] + ["mamba"] * 4
    ),
}

QWEN3_5_4B_HF = {
    "model_type": "qwen3_5",
    "tie_word_embeddings": True,
    "text_config": {
        "hidden_size": 2560,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "attn_output_gate": True,
        "intermediate_size": 9216,
        "vocab_size": 248320,
        "tie_word_embeddings": True,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "max_position_embeddings": 262144,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 8,
    },
}

LFM2_2P6B_HF = {
    "model_type": "lfm2",
    "hidden_size": 2048,
    "num_hidden_layers": 30,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 10752,
    "vocab_size": 65536,
    "tie_embedding": True,
    "conv_L_cache": 3,
    "conv_dim": 2048,
    "max_position_embeddings": 128000,
    "layer_types": [
        "conv", "conv", "full_attention", "conv", "conv", "full_attention", "conv", "conv",
        "conv", "full_attention", "conv", "conv", "conv", "full_attention", "conv", "conv",
        "conv", "full_attention", "conv", "conv", "conv", "full_attention", "conv", "conv",
        "full_attention", "conv", "conv", "full_attention", "conv", "conv",
    ],
}

FALCON_H1_7B_HF = {
    "model_type": "falcon_h1",
    "hidden_size": 3072,
    "num_hidden_layers": 44,
    "num_attention_heads": 12,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "intermediate_size": 12288,
    "vocab_size": 130048,
    "tie_word_embeddings": False,
    "mamba_d_state": 256,
    "mamba_n_groups": 1,
    "mamba_n_heads": 24,
    "mamba_d_head": 128,
    "mamba_d_ssm": 3072,
    "mamba_expand": 2,
    "mamba_d_conv": 4,
    "mamba_chunk_size": 128,
    "max_position_embeddings": 262144,
}

FALCON_MAMBA_7B_HF = {
    "model_type": "falcon_mamba",
    "hidden_size": 4096,
    "num_hidden_layers": 64,
    "intermediate_size": 8192,
    "vocab_size": 65024,
    "tie_word_embeddings": False,
    "state_size": 16,
    "conv_kernel": 4,
    "time_step_rank": 256,
    "expand": 16,
    "max_position_embeddings": 2048,
}

MINICPM3_4B_HF = {
    "model_type": "minicpm3",
    "hidden_size": 2560,
    "num_hidden_layers": 62,
    "num_attention_heads": 40,
    "num_key_value_heads": 40,
    "intermediate_size": 6400,
    "vocab_size": 73448,
    "q_lora_rank": 768,
    "kv_lora_rank": 256,
    "qk_nope_head_dim": 64,
    "qk_rope_head_dim": 32,
    "max_position_embeddings": 32768,
}

SMOLLM3_3B_HF = {
    "model_type": "smollm3",
    "hidden_size": 2048,
    "num_hidden_layers": 36,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "intermediate_size": 11008,
    "vocab_size": 128256,
    "tie_word_embeddings": True,
    "max_position_embeddings": 65536,
    "layer_types": ["full_attention"] * 36,
}

HUNYUAN_7B_HF = {
    "model_type": "hunyuan_v1_dense",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 14336,
    "vocab_size": 128167,
    "tie_word_embeddings": True,
    "max_position_embeddings": 32768,
}

FALCON_H1_3B_HF = {
    "model_type": "falcon_h1",
    "hidden_size": 2560,
    "num_hidden_layers": 32,
    "num_attention_heads": 10,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "intermediate_size": 6144,
    "vocab_size": 65536,
    "tie_word_embeddings": False,
    "mamba_d_state": 256,
    "mamba_n_groups": 1,
    "mamba_n_heads": 32,
    "mamba_d_head": 128,
    "mamba_d_ssm": 4096,
    "mamba_expand": 2,
    "mamba_d_conv": 4,
    "mamba_chunk_size": 128,
    "max_position_embeddings": 131072,
}

HUNYUAN_4B_HF = {
    "model_type": "hunyuan_v1_dense",
    "hidden_size": 3072,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 8192,
    "vocab_size": 120818,
    "tie_word_embeddings": True,
    "max_position_embeddings": 262144,
}

#: Gemma 3's text_config, kept beside the fixtures it cannot join: the importer
#: REFUSES a sliding-window import family-agnostically (P1.2), so the checked-in
#: gemma_3_4b_inf.yaml is hand-authored and the refusal is what gets tested.
GEMMA_3_4B_HF = {
    "model_type": "gemma3_text",
    "hidden_size": 2560,
    "num_hidden_layers": 34,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "intermediate_size": 10240,
    "vocab_size": 262208,
    "tie_word_embeddings": True,
    "max_position_embeddings": 131072,
    "sliding_window": 1024,
    "sliding_window_pattern": 6,
}

HF_FIXTURES = {
    "granite_4_0_h_tiny": GRANITE_H_TINY_HF,
    "granite_4_0_h_1b": GRANITE_H_1B_HF,
    "falcon_h1_3b": FALCON_H1_3B_HF,
    "hunyuan_4b": HUNYUAN_4B_HF,
    "qwen3_5_4b": QWEN3_5_4B_HF,
    "lfm2_2p6b": LFM2_2P6B_HF,
    "falcon_h1_7b": FALCON_H1_7B_HF,
    "falcon_mamba_7b": FALCON_MAMBA_7B_HF,
    "minicpm3_4b": MINICPM3_4B_HF,
    "smollm3_3b": SMOLLM3_3B_HF,
    "hunyuan_7b": HUNYUAN_7B_HF,
}


def _import_hf_fixture(hf_config):
    module = _load_hf_to_config_module()
    model_type, alias = module._infer_model_type(hf_config["model_type"])
    yaml_config = module._build_yaml_config(hf_config, _import_args(), model_type, alias=alias)
    return _model_from_dict(yaml_config["model_param"])


@pytest.mark.parametrize("name", sorted(HF_FIXTURES))
def test_hf_ingestion_reproduces_the_checked_in_yaml(name):
    imported = _import_hf_fixture(HF_FIXTURES[name])
    checked_in = config.parse_config(str(_matrix_yaml_path(name)), "LLM").model_config
    assert imported.num_layers == checked_in.num_layers
    assert imported.hidden_dim == checked_in.hidden_dim
    assert imported.vocab_size == checked_in.vocab_size
    assert imported.tied_embeddings == checked_in.tied_embeddings
    assert imported.layer_mixers == checked_in.layer_mixers
    assert imported.ssm == checked_in.ssm
    assert imported.linear_attention == checked_in.linear_attention
    assert imported.short_conv == checked_in.short_conv
    assert imported.attention == checked_in.attention
    assert llm_util.model_raw_param_count(imported) == llm_util.model_raw_param_count(checked_in)


def test_hf_ingestion_of_the_dense_granite_leaves_no_moe_behind():
    # The dense branch: HF publishes num_local_experts: 0, so the import must
    # come out dense — one expert, every layer dense, and NO shared-expert
    # ffn_dims entry, which only belongs beside routed experts.
    imported = _import_hf_fixture(GRANITE_H_1B_HF)
    assert imported.num_experts == 1
    assert not imported.use_moe
    assert imported.moe.first_k_dense_replace == imported.num_layers
    assert "shared_expert" not in imported.ffn_dims
    assert imported.intermediate_size == 4096
    checked_in = config.parse_config(str(_matrix_yaml_path("granite_4_0_h_1b")), "LLM").model_config
    assert imported.moe.first_k_dense_replace == checked_in.moe.first_k_dense_replace
    assert imported.ffn_dims == checked_in.ffn_dims


def test_hf_ingestion_copies_layer_types_verbatim():
    module = _load_hf_to_config_module()
    model_type, alias = module._infer_model_type("granitemoehybrid")
    yaml_config = module._build_yaml_config(GRANITE_H_TINY_HF, _import_args(), model_type, alias=alias)
    assert yaml_config["model_param"]["layer_plan"]["layer_types"] == GRANITE_H_TINY_HF["layer_types"]
    assert yaml_config["model_param"]["ffn_dims"] == {"shared_expert": 1024}
    assert yaml_config["model_param"]["moe"]["n_shared_experts"] == 1


def test_hf_ingestion_rejects_a_layer_types_length_mismatch():
    module = _load_hf_to_config_module()
    broken = dict(GRANITE_H_TINY_HF, layer_types=["mamba"] * 39)
    with pytest.raises(SystemExit, match="layer_types"):
        module._build_yaml_config(broken, _import_args(), "llama", alias="granite_hybrid")


def test_hf_ingestion_omits_attention_for_a_pure_ssm_stack():
    module = _load_hf_to_config_module()
    yaml_config = module._build_yaml_config(
        FALCON_MAMBA_7B_HF, _import_args(), "llama", alias="falcon_mamba"
    )
    assert "attention" not in yaml_config["model_param"]
    assert yaml_config["model_param"]["layer_plan"]["ffn_per_layer"] is False
    assert yaml_config["model_param"]["ssm"]["variant"] == "mamba1"


def test_hf_ingestion_prefers_d_ssm_over_expand():
    module = _load_hf_to_config_module()
    block = module._build_ssm_block(FALCON_H1_7B_HF, {}, "falcon_h1")
    assert block["d_inner"] == 3072
    assert "expand" not in block


def test_hf_ingestion_reads_the_text_config_sub_block():
    module = _load_hf_to_config_module()
    yaml_config = module._build_yaml_config(
        QWEN3_5_4B_HF, _import_args(), "llama", alias="qwen3_5"
    )
    model_param = yaml_config["model_param"]
    assert model_param["hidden_dim"] == 2560
    assert model_param["attention"]["head_dim"] == 256
    assert model_param["attention"]["output_gate"] is True
    assert model_param["linear_attention"]["num_value_heads"] == 32


def test_hf_ingestion_fills_minicpm3_v_head_dim():
    module = _load_hf_to_config_module()
    yaml_config = module._build_yaml_config(
        MINICPM3_4B_HF, _import_args(), "deepseek_v3", alias="minicpm3"
    )
    attention = yaml_config["model_param"]["attention"]
    assert attention["attention_type"] == "mla"
    assert attention["v_head_dim"] == 64


def test_hf_ingestion_refuses_a_sliding_window_model_whatever_the_family():
    # The importer used to refuse Phi-3 and Qwen2 for sliding-window attention
    # while silently emitting a window pattern for every other family. One rule
    # now: nothing that RAPID prices against the full KV context is imported.
    module = _load_hf_to_config_module()
    hf_config = {
        "model_type": "smollm3",
        "hidden_size": 2048,
        "num_hidden_layers": 4,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "intermediate_size": 8192,
        "vocab_size": 1000,
        "sliding_window": 512,
        "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        "max_position_embeddings": 4096,
    }
    with pytest.raises(SystemExit) as exc:
        module._validate_supported_config_features(hf_config, "smollm3")
    assert "sliding-window attention is not modeled" in str(exc.value)
    # A model that publishes the field but switches the pattern off imports.
    off = dict(hf_config, use_sliding_window=False)
    module._validate_supported_config_features(off, "smollm3")


def test_the_gemma_3_row_is_hand_authored_because_the_importer_refuses_it():
    """gemma_3_4b_inf.yaml exists only because P1.2's rule holds against it.

    The importer refuses every sliding-window model family-agnostically, so the
    first WINDOWED row in the matrix cannot be imported and is hand-authored
    against the same published text_config. That is not a hole in P1.2: the
    importer's rule is about what the DEFAULT paths price, and the checked-in
    YAML carries the window precisely so the MAPPED path can price it.
    """
    module = _load_hf_to_config_module()
    with pytest.raises(SystemExit) as exc:
        module._validate_supported_config_features(GEMMA_3_4B_HF, "gemma3_text")
    assert "sliding-window attention is not modeled" in str(exc.value)

    # The hand-authored row carries exactly the published pattern.
    checked_in = config.parse_config(str(_matrix_yaml_path("gemma_3_4b")), "LLM").model_config
    assert checked_in.hidden_dim == GEMMA_3_4B_HF["hidden_size"]
    assert checked_in.num_layers == GEMMA_3_4B_HF["num_hidden_layers"]
    assert checked_in.vocab_size == GEMMA_3_4B_HF["vocab_size"]
    assert checked_in.attention.head_dim == GEMMA_3_4B_HF["head_dim"]
    assert checked_in.attention.window.window_size == GEMMA_3_4B_HF["sliding_window"]
    assert (
        checked_in.attention.window.local_global_interval
        == GEMMA_3_4B_HF["sliding_window_pattern"]
    )
