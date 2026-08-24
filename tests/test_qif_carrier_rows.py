"""QIF C1: the P1 family carriers, and the first windowed-attention row.

P1.3 shipped Falcon-H1 and Hunyuan as 7B-class REFERENCE rows and recorded an
assumption: a family whose only shipped row is over the D2 cutoff contributes
zero SUPPORTED models, so each gets its in-scope sibling. Those two siblings
land here, and so does Gemma-3-4B -- the sliding-window law has been priced end
to end since Wave B, but the window had only ever been INJECTED into another
model's config because no verifiable Gemma-class YAML existed.

Each row is here for a different verdict, and the verdict is the test:

* falcon_h1_3b  REFUSED on fws_cim. Its laws all exist; its LOWERING does not.
* hunyuan_4b    RUNS on the existing (non-mapped) path, end to end.
* gemma_3_4b    RUNS mapped, with both window laws live on the real pattern.

Dims for all three were verified 2026-08-24 against the published HuggingFace
config.json cited in each YAML's provenance block; the parameter censuses are
checked against the published checkpoint totals by
tests/test_qif_model_matrix.py.
"""

from __future__ import annotations

import collections
import copy
import math
import os

import pytest
import yaml

import config
import fws_eval
import fws_mapping
import llm_util
from inference_timing import TimeCalculationLLMInference
from program.fws_build import build_fws_program

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HW_DIR = os.path.join(PROJECT_ROOT, "configs", "hardware-config")
MODEL_DIR = os.path.join(PROJECT_ROOT, "configs", "model-config")

A100 = os.path.join(HW_DIR, "a100_80GB.yaml")
GRANITE_HW = os.path.join(HW_DIR, "fws_cim_granite_tiny.yaml")
# The 2560-row analog card. Its ANALOG POINT is not Qwen-specific -- `rows`
# matches hidden_dim, which Gemma-3-4B shares -- so the windowed row is read on
# a card that already ships rather than on a second invented design point
# (ADJ-4). WAVE C AUDIT: two knobs in that file ARE derived from Qwen, and the
# file's own comments say so: `layers_per_chip: 4` is Qwen's layer_plan period
# and `arrays_per_chip: 180` is Qwen's worst-chip arithmetic. Gemma fits both,
# but by arithmetic of its own, so
# test_the_shared_2560_card_still_fits_gemma_by_gemmas_own_arithmetic pins it:
# a later edit to Qwen's sizing that breaks this row FAILS instead of quietly
# changing it.
CARD_2560_HW = os.path.join(HW_DIR, "fws_cim_qwen3_5_4b.yaml")

FALCON_H1_3B = os.path.join(MODEL_DIR, "falcon_h1_3b_inf.yaml")
HUNYUAN_4B = os.path.join(MODEL_DIR, "hunyuan_4b_inf.yaml")
GEMMA_3_4B = os.path.join(MODEL_DIR, "gemma_3_4b_inf.yaml")


def _hw(path, mutate=None):
    with open(path) as handle:
        raw = copy.deepcopy(yaml.safe_load(handle))
    if mutate is not None:
        mutate(raw)
    config.convert(raw)
    return config.HWConfig.from_dict(raw)


def _gpu_hw():
    hw_config = config.parse_config(A100, config_type="hardware")
    sch = hw_config.sch_config
    sch.tp = sch.cp = sch.pp = sch.mb = 1
    sch.tp_sp = False
    sch.train.dp = sch.train.ep = 1
    sch.inference.replica_count = 1
    sch.inference.moe_dp = 1
    hw_config.execution_backend.model = "analytical"
    hw_config.execution_backend.astra = None
    for dim in hw_config.network_layout.dimensions:
        object.__setattr__(dim, "topology_type", "Ring")
    return hw_config


# ---------------------------------------------------------------------------
# Falcon-H1-3B: every law, no lowering
# ---------------------------------------------------------------------------


def test_falcon_h1_3b_is_the_supported_sibling_of_its_family():
    model = config.parse_config(FALCON_H1_3B, "LLM").model_config
    assert model.num_layers == 32
    assert model.hidden_dim == 2560
    assert model.layer_plan.structure == "parallel_branch"
    assert model.layer_mixers[0] == ("attention", "ssm", "ffn")
    assert model.attention.head_dim == 128  # decoupled from 2560/10 = 256
    assert model.ssm.resolve_d_inner(model.hidden_dim) == 4096  # d_ssm beats expand
    assert llm_util.model_raw_param_count(model) < 7_000_000_000  # D2: supported


def test_falcon_h1_3b_is_refused_on_fws_cim_and_the_refusal_names_the_branch():
    """The refusal is the deliverable: it must name the RIGHT reason.

    Every block kind Falcon-H1 uses is on the mapped list, so the block-kind
    gate would let it through, and it would then be lowered as a DEEPER
    SEQUENTIAL model -- the branches chained instead of summed. That is a wrong
    number, which is what P1.4's rule exists to stop.
    """
    model = config.parse_config(FALCON_H1_3B, "LLM")
    kinds = set(model.model_config.hybrid_block_kinds) | {"attention"}
    assert kinds <= set(config._FWS_CIM_MAPPED_BLOCK_KINDS), (
        "if a block kind were unpriced the refusal would be the block gate's, "
        "not the branch gate's, and this test would be proving nothing"
    )
    for hardware in (GRANITE_HW, CARD_2560_HW):
        with pytest.raises(ValueError) as error:
            config.validate_configs(_hw(hardware), model)
        message = str(error.value)
        assert "model_param.layer_plan.structure" in message
        assert "parallel_branch" in message
        # It says the LAWS are there and the LOWERING is not, so a reader knows
        # which half of the system to go and build.
        assert "lowering" in message.lower()

    # ... and dropping the mapping block does not make it run either.
    def unmapped(raw):
        raw.pop("mapping", None)

    with pytest.raises(ValueError):
        config.validate_configs(_hw(GRANITE_HW, unmapped), model)


def test_falcon_h1_3b_is_refused_on_the_gpu_path_too():
    with pytest.raises(ValueError) as error:
        config.validate_configs(_gpu_hw(), config.parse_config(FALCON_H1_3B, "LLM"))
    assert "layer_plan" in str(error.value)
    assert "ssm" in str(error.value)


# ---------------------------------------------------------------------------
# Hunyuan-4B: runs now
# ---------------------------------------------------------------------------


def test_hunyuan_4b_runs_end_to_end_on_the_existing_path():
    model = config.parse_config(HUNYUAN_4B, "LLM")
    hw_config = _gpu_hw()
    config.validate_configs(hw_config, model)
    total = TimeCalculationLLMInference(
        hw_config, model, "LLM"
    ).calc_total_inference_time()
    assert total["total_inference_time"] > 0.0


def test_hunyuan_4bs_decoupled_head_dim_is_honored_and_not_derived():
    """3072 / 32 heads = 96, and the checkpoint's head_dim is 128.

    The published checkpoint is 4,221,757,440 params (index total_size / 2).
    With head_dim 128 the census lands at 4,221,523,968 -- the 233,472
    difference is the RMSNorm and QK-norm vectors, which the census excludes by
    design. With the DERIVED 96 it would land at 3,938,408,448, 6.7% low, so
    the declared value is not cosmetic: it is the difference between matching
    the checkpoint and missing it.
    """
    model = config.parse_config(HUNYUAN_4B, "LLM").model_config
    assert model.attention.head_dim == 128
    assert model.hidden_dim // model.attention.num_heads == 96
    _head_dim, q_size, kv_size = llm_util.attention_dim_sizes(
        model.hidden_dim, model.attention.num_heads, model.attention.kv_heads,
        head_dim=model.attention.head_dim,
    )
    assert (q_size, kv_size) == (4096, 1024)
    assert llm_util.model_raw_param_count(model) == 4_221_523_968


# ---------------------------------------------------------------------------
# Gemma-3-4B: the first windowed row, on a real model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gemma():
    mapping = fws_mapping.build_mapping(
        _hw(CARD_2560_HW), config.parse_config(GEMMA_3_4B, "LLM")
    )
    return fws_eval.evaluate_fws(build_fws_program(mapping))


def test_gemma_3_4b_declares_the_published_window_pattern():
    model = config.parse_config(GEMMA_3_4B, "LLM").model_config
    assert model.num_layers == 34
    assert model.hidden_dim == 2560
    assert model.attention.head_dim == 256  # decoupled from 2560/8 = 320
    window = model.attention.window
    assert window.window_size == 1024
    assert window.local_global_interval == 6   # HF sliding_window_pattern
    assert window.last_layer_global is False


def test_gemma_3_4b_is_refused_off_the_mapped_path_by_name():
    """A local layer priced against the full KV context is a wrong number."""
    model = config.parse_config(GEMMA_3_4B, "LLM")
    with pytest.raises(ValueError) as error:
        config.validate_configs(_gpu_hw(), model)
    assert "model_param.attention.window" in str(error.value)

    def unmapped(raw):
        raw.pop("mapping", None)

    with pytest.raises(ValueError) as error:
        config.validate_configs(_hw(CARD_2560_HW, unmapped), model)
    assert "model_param.attention.window" in str(error.value)


def test_gemma_3_4b_runs_mapped_with_both_window_laws_live(gemma):
    """5 local : 1 global, priced layer by layer off the model's own pattern.

    This is what the Wave B audit fix could only assert against an injected
    window. Layer i is GLOBAL when (i + 1) % 6 == 0, so of 34 layers exactly 5
    are global (layers 5, 11, 17, 23, 29) and 29 are local.
    """
    prefill = [
        cost for cost in gemma.pricing.costs
        if cost.block == "attention_qk" and cost.phase == "prefill"
    ]
    assert len(prefill) == 34
    local = {c.layer for c in prefill if "sliding_window_prefill_timing" in c.basis}
    globals_ = {c.layer for c in prefill if "prefill_attention_timing" in c.basis
                and "sliding_window" not in c.basis}
    assert globals_ == {5, 11, 17, 23, 29}
    assert local == set(range(34)) - globals_
    assert len(local) == 29

    # A local layer is CHEAPER than a global one at this context, which is the
    # whole reason the window is modeled: 1024 of a 1792-token prefill.
    local_cost = next(c for c in prefill if c.layer == 0).duration_s
    global_cost = next(c for c in prefill if c.layer == 5).duration_s
    assert 0 < local_cost < global_cost

    # ... and the reader is told which rule produced the split.
    assert any(
        item.constraint == "sliding_window_layer_pattern" for item in gemma.disclosures
    )


def test_gemma_3_4b_decode_uses_the_window_too(gemma):
    decode = [
        cost for cost in gemma.pricing.costs
        if cost.block == "attention_qk" and cost.phase == "decode"
    ]
    assert decode
    windowed = [c for c in decode if "sliding_window_decode_timing" in c.basis]
    assert {c.layer for c in windowed} == set(range(34)) - {5, 11, 17, 23, 29}


def test_gemma_3_4b_places_no_gate_and_no_recurrence(gemma):
    """It is a plain windowed transformer: no output gate, no hybrid block."""
    assert gemma.mapping.model.layer_plan is None
    assert not [o for o in gemma.mapping.owners() if o.op == "attn_gate_proj"]
    assert not [c for c in gemma.pricing.costs if c.block in ("delta_rule", "ssm_scan")]


def test_the_shared_2560_card_still_fits_gemma_by_gemmas_own_arithmetic(gemma):
    """WAVE C AUDIT: the Gemma row rides a card sized for Qwen. Pin the fit.

    ``configs/hardware-config/fws_cim_qwen3_5_4b.yaml`` justifies
    ``layers_per_chip: 4`` by Qwen's layer_plan period and ``arrays_per_chip:
    180`` by Qwen's worst-chip arithmetic. Gemma has no layer plan and a
    different endpoint, so it fits by arithmetic of its own: 34 layers at 4 per
    chip is 9 analog chips, and the busiest of them (the last, which carries the
    262208-wide lm_head) holds 133 macros against the card's 180. Without this
    gate, an edit to Qwen's sizing could break or silently move the Gemma row.
    """
    counts = collections.Counter(
        slot.chip_id for slot in gemma.mapping.macros
        if slot.pool == "analog" and slot.tiles
    )
    card = gemma.mapping.hw.cim_config.chip
    assert card.layers_per_chip == 4
    assert len(counts) == math.ceil(34 / card.layers_per_chip) == 9
    assert max(counts.values()) == 133 <= card.arrays_per_chip
