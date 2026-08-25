"""QIF P1.5: Granite-4.0-H-Tiny runs end to end on the mapped fws_cim path.

ADJ-1 made Granite-4.0-H-Tiny the headline model. P1.4 refused it — nothing
priced a Mamba-2 recurrence — and P2.6, P3 and P4 then built the laws, the
placement and the pricing. This module is the proof that the three met: 36 SSD
layers and 4 attention layers, MoE on every one of them, priced op by op.

The narrowing is the risky part, so most of these tests are about what stays
REFUSED. A rejection seam that opens one notch too wide does not crash; it
prints a number that silently assumes a law nobody wrote.

The physics audits are here too, in the DESIGN2 §8 style: a law that exceeds a
declared peak is wrong even when nothing fails, and an analog op whose duration
does not match the M-law by hand is wrong even when the report is self-
consistent.
"""

from __future__ import annotations

import copy
import json
import math
import os

import pytest
import yaml

import cim_timing
import config
import fws_atlas_export
import fws_eval
import fws_mapping
from program.fws_build import build_fws_program

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HW_DIR = os.path.join(PROJECT_ROOT, "configs", "hardware-config")
MODEL_DIR = os.path.join(PROJECT_ROOT, "configs", "model-config")

GRANITE_HW = os.path.join(HW_DIR, "fws_cim_granite_tiny.yaml")
GRANITE_MODEL = os.path.join(MODEL_DIR, "granite_4_0_h_tiny_inf.yaml")
QWEN_MODEL = os.path.join(MODEL_DIR, "qwen3_5_4b_inf.yaml")
FWS_LLAMA7B = os.path.join(HW_DIR, "fws_cim_llama7b.yaml")
A100 = os.path.join(HW_DIR, "a100_80GB.yaml")


def _hw(path, mutate=None):
    with open(path) as handle:
        raw = copy.deepcopy(yaml.safe_load(handle))
    if mutate is not None:
        mutate(raw)
    config.convert(raw)
    return config.HWConfig.from_dict(raw)


@pytest.fixture(scope="module")
def granite():
    mapping = fws_mapping.build_mapping(
        _hw(GRANITE_HW), config.parse_config(GRANITE_MODEL, "LLM")
    )
    return fws_eval.evaluate_fws(build_fws_program(mapping))


# ---------------------------------------------------------------------------
# The narrowing: what opened, and everything that did not
# ---------------------------------------------------------------------------


def test_granite_validates_on_the_mapped_fws_cim_path():
    hw = _hw(GRANITE_HW)
    assert hw.mapping_config is not None  # the mapping block IS the narrowing
    config.validate_configs(hw, config.parse_config(GRANITE_MODEL, "LLM"))


def test_the_same_granite_config_is_still_refused_without_a_mapping_block():
    """The closed form has no stage for a recurrence and must keep saying so."""
    hw = _hw(GRANITE_HW, lambda raw: raw.pop("mapping"))
    with pytest.raises(ValueError) as excinfo:
        config.validate_configs(hw, config.parse_config(GRANITE_MODEL, "LLM"))
    message = str(excinfo.value)
    assert "ssm" in message
    assert "mapping:" in message  # the message names the way forward


def test_the_gpu_path_rejection_is_untouched():
    """P1.4's GPU refusal is not scoped to fws_cim and must not move."""
    hw = config.parse_config(A100, "hardware")
    with pytest.raises(ValueError) as excinfo:
        config.validate_configs(hw, config.parse_config(GRANITE_MODEL, "LLM"))
    assert "ssm" in str(excinfo.value)


@pytest.mark.parametrize(
    "field, override",
    [
        # attention.output_gate is NO LONGER on this list: C1 places W_g as an
        # analog tile and the gate multiply as pool work, so the mapped path
        # prices it. tests/test_qif_qwen_e2e.py holds it end to end, and it
        # stays refused on the GPU and unmapped paths there.
        (
            "shared_weight_groups",
            {"shared_weight_groups": [{"name": "depth_shared", "layers": [0, 1]}]},
        ),
        ("ffn_dims", {"ffn_dims": {"shared_expert": 1024, "default": 999}}),
    ],
)
def test_inputs_the_mapped_path_still_cannot_price_stay_refused(field, override):
    """The narrowing lifted a NAMED list. Everything else keeps its name.

    ``ffn_dims`` is the sharp one: ``shared_expert`` is priced now (it reaches
    the array census and the mapping's stage shapes), and any OTHER key still
    only moves llm_util's parameter count — so the refusal must survive on the
    key, not on the block.
    """
    with open(GRANITE_MODEL) as handle:
        raw = copy.deepcopy(yaml.safe_load(handle))
    model_param = raw["model_param"]
    for key, value in override.items():
        if key == "attention":
            model_param["attention"] = {**model_param["attention"], **value}
        else:
            model_param[key] = value
    # Parse through the normal file path so the schema validation is the real one.
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(raw, handle)
        path = handle.name
    try:
        model = config.parse_config(path, "LLM")
        with pytest.raises(ValueError) as excinfo:
            config.validate_configs(_hw(GRANITE_HW), model)
        message = str(excinfo.value)
        assert field.split(".")[-1] in message
    finally:
        os.unlink(path)


def test_a_block_kind_nobody_prices_would_still_be_refused_on_the_mapped_path():
    """The mapped narrowing is a NAMED list, not `hybrids are fine now`."""
    priced = set(config._FWS_CIM_MAPPED_BLOCK_KINDS)
    assert priced == {"attention", "ssm", "short_conv", "linear_attn"}
    # Every kind on the list must be one the P1 schema can express, or the
    # list is naming something that cannot arrive.
    assert priced <= set(config._BLOCK_KINDS)


def test_an_ssm_run_without_a_declared_vector_engine_derives_one_from_the_beat():
    """D31 (WAVE F): an undeclared engine inside a mapped run is DERIVED, not refused.

    REGIME-CHANGE REWRITE. The old claim was "a mapped Granite run whose card
    declares no vector_lanes refuses by name (ADJ-4: scan lanes have no
    default)". D31 retires vector_lanes as an input and makes the DEFAULT
    "derive it from the beat the analog stages set", so the new claim is: the
    run completes, the width is derived, and it is REPORTED with its provenance
    and the stage that bound it. ADJ-4 is not weakened — the refusal survives
    wherever there is no beat to derive from, which
    tests/test_qif_digital_ops.py::test_vector_lanes_have_no_default_and_refuse_by_name
    still pins on a bare device.
    """
    hw = _hw(GRANITE_HW, lambda raw: raw["cim"].pop("cards"))
    mapping = fws_mapping.build_mapping(hw, config.parse_config(GRANITE_MODEL, "LLM"))
    evaluation = fws_eval.evaluate_fws(build_fws_program(mapping))
    sizing = mapping.device.derived_engine
    assert sizing is not None
    assert sizing.vector_lanes >= 1
    assert mapping.device.vector_lanes == sizing.vector_lanes
    assert mapping.device.vector_lanes_provenance == cim_timing.PROVENANCE_DERIVED_COUNT
    # The width HOLDS the analog beat: the binding stage's own priced vector
    # time fits inside it, and one lane fewer would not (no margin, D28).
    binding = next(r for r in sizing.per_stage if r.stage == sizing.binding_stage)
    assert binding.time_s <= sizing.analog_beat_s
    assert binding.used_cycles <= binding.budget_cycles
    # ... and it is REPORTED, which is the other half of D31.
    row = next(
        d for d in evaluation.disclosures if d.constraint == "derived_engine_sizing"
    )
    assert str(sizing.vector_lanes) in row.value
    assert "D31" in row.reason
    # This card names no synthesis library, so nothing is composed for it (D32
    # composes from MEASURED blocks or not at all).
    assert sizing.composition is None


# ---------------------------------------------------------------------------
# The run: every block kind reaches a named law
# ---------------------------------------------------------------------------


def test_every_hybrid_block_kind_is_priced_by_a_named_law(granite):
    laws = {}
    for cost in granite.pricing.costs:
        laws.setdefault(cost.block, set()).add(cost.basis.split("(")[0].strip())
    assert "ssm_scan" in laws, sorted(laws)
    assert any("price_ssm_block" in basis for basis in laws["ssm_scan"])
    # WAVE F (D29/D30): a mapped run lowers DECODE only — the streams arrive
    # already prefilled — so attention is priced by the decode law, and there
    # is no lm_head block at all because D30 drops the endpoints entirely.
    assert any("decode_attention_timing" in b for b in laws["attention_qk"])
    assert "lm_head" not in laws
    for block in ("ssm_in_proj", "ssm_out_proj", "ffn1_routed", "ffn2_shared", "o_proj"):
        assert any("price_tiled_op" in b for b in laws[block]), block
    # No op may be priced by an unnamed law, ever.
    assert all(cost.basis for cost in granite.pricing.costs)


def test_the_layer_plan_reaches_the_placement_layer_by_layer(granite):
    """36 Mamba layers and 4 attention layers, in the pattern the YAML declares."""
    scan_layers = {
        cost.layer for cost in granite.pricing.costs if cost.block == "ssm_scan"
    }
    attention_layers = {
        cost.layer for cost in granite.pricing.costs if cost.block == "attention_qk"
    }
    assert len(scan_layers) == 36
    assert sorted(attention_layers) == [5, 15, 25, 35]
    assert not (scan_layers & attention_layers)


def test_the_short_conv_rides_the_pool_and_never_becomes_a_timed_op(granite):
    """ADJ-3: Mamba-2's d_conv = 4 is absorbed into the pool sizing, not timed."""
    assert granite.mapping.model.ssm.d_conv == 4
    assert not [cost for cost in granite.pricing.costs if cost.block == "short_conv"]
    # And the absorption is DISCLOSED rather than silent.
    assert any(
        item.constraint == "macro_pool_helper_ops" for item in granite.disclosures
    )


def test_the_shared_expert_is_wider_than_a_routed_one_everywhere(granite):
    """Granite's shared MLP is 1024 wide beside 64 routed experts at 512.

    llm_util's parameter census already read ``ffn_dims.shared_expert``; the
    array census and the mapping now read the SAME number, so one model does
    not carry two widths (D21).
    """
    params = granite.mapping.device.params
    assert params.moe_intermediate == 512
    assert params.shared_expert_intermediate == 1024
    census = granite.mapping.device.moe_layer_stage_arrays()
    assert census["ffn1_shared"] == 2  # ceil(2 * 1024 / 1536)
    assert census["ffn1_routed"] == 64 * 1  # ceil(2 * 512 / 1536)
    shapes = {
        (shape.op, shape.n)
        for shape in fws_mapping._moe_stages(granite.mapping.device)
        if shape.op == "ffn1_shared"
    }
    assert shapes == {("ffn1_shared", 2048)}


def test_the_shared_expert_width_defaults_to_the_routed_width(granite):
    """Degenerate identity: a model with no ffn_dims moves no array count."""
    device = granite.mapping.device
    params = device.params
    plain = cim_timing.CimModelParams(
        **{**params.__dict__, "shared_intermediate_size": 0}
    )
    assert plain.shared_expert_intermediate == plain.moe_intermediate


# ---------------------------------------------------------------------------
# Physics audits (DESIGN2 §8 style): peaks, and the M-law by hand
# ---------------------------------------------------------------------------


def test_no_op_exceeds_the_engine_peak_its_law_declares(granite):
    """A law that outruns its own silicon is wrong even when parity passes."""
    device = granite.mapping.device
    peak_ops_per_s = device.vector_lanes * device.vector_clock_hz
    checked = 0
    for cost in granite.pricing.costs:
        ops = float((cost.detail or {}).get("ops", 0.0))
        if ops <= 0 or cost.duration_s <= 0:
            continue
        checked += 1
        assert ops / cost.duration_s <= peak_ops_per_s * (1 + 1e-9), (
            cost.block,
            ops / cost.duration_s,
            peak_ops_per_s,
        )
    assert checked > 0, "no vector-engine op was audited"


def test_the_scan_engine_is_the_binding_resource_and_says_so(granite):
    """The SSD scan runs AT the declared peak: the engine, not the array, binds.

    WAVE F: the run is DECODE ONLY (D25/D29), so this is the recurrent scan of
    ONE token of ONE stream (M = 1). The engine still runs at its declared peak
    to within the pipeline drain it pays once, and it still dwarfs the analog
    projection that feeds it — by 25x here rather than the prefill's larger
    factor, because the analog side also fell to M = 1.
    """
    device = granite.mapping.device
    peak = device.vector_lanes * device.vector_clock_hz
    scan = next(
        cost for cost in granite.pricing.costs if cost.block == "ssm_scan"
    )
    utilisation = float(scan.detail["ops"]) / scan.duration_s / peak
    assert 0.98 < utilisation <= 1.0
    in_proj = next(
        cost for cost in granite.pricing.costs if cost.block == "ssm_in_proj"
    )
    assert scan.duration_s > 10 * in_proj.duration_s


def test_an_analog_op_matches_the_M_law_computed_by_hand(granite):
    """duration = M * slice_cycles / f_analog * active_column_sets.

    Recomputed here from the YAML's own numbers, not from CimDeviceModel, so a
    change inside the law has to move this line too (the pass-2A spot-check
    pattern).
    """
    device = granite.mapping.device
    slice_cycles = int(device.analog.slice_cycles)
    f_analog = float(device.analog.analog_clock_mhz) * 1e6
    # WAVE F: every op of a filled-pipeline run is a DECODE op at M = 1, so the
    # law is checked on the three blocks rather than on three phases.
    for block, phase in (("qkv", "decode"), ("ffn1_routed", "decode"), ("ssm_out_proj", "decode")):
        cost = next(
            c
            for c in granite.pricing.costs
            if c.block == block and c.phase == phase and c.kind == "weight_gemm"
        )
        tokens = float(cost.basis.split("m_tokens=")[1].split(",")[0])
        columns = float(cost.detail["active_column_sets"])
        assert cost.duration_s == pytest.approx(
            tokens * slice_cycles / f_analog * columns, rel=1e-12
        )


def test_a_routed_expert_sees_the_routed_token_count_by_hand(granite):
    """ceil(m * top_k * alpha / E) at the ONE token a stage holds per beat.

    WAVE F (D29): the local batch is 1, so the hot-expert count is
    ceil(1 * 6 / 64) = 1 — one token reaches an expert or it does not, and the
    law rounds up to the expert that gets it. The retired regime asked the same
    question at m = B x S = 7168 and got 672.
    """
    params = granite.mapping.device.params
    owner_tokens = 1.0
    expected = math.ceil(owner_tokens * params.top_k * params.expert_imbalance_factor
                         / params.num_experts)
    assert expected == 1
    assert granite.mapping.device.moe_tokens_hot(1) == 1
    cost = next(c for c in granite.pricing.costs if c.block == "ffn1_routed")
    assert "m_tokens=%d" % expected in cost.basis


# ---------------------------------------------------------------------------
# The artifacts: the report and the atlas
# ---------------------------------------------------------------------------


def test_the_report_is_one_document_with_every_component_labeled(granite):
    document = fws_eval.report_document(granite)
    assert document["schema"] == fws_eval.REPORT_SCHEMA
    energy = document["evaluation"]["energy"]
    assert energy["total_pj"] == pytest.approx(
        sum(component["energy_pj"] for component in energy["components"])
    )
    for component in energy["components"]:
        assert component["coverage"] in (
            fws_eval.COVERAGE_COVERED,
            fws_eval.COVERAGE_PARTIAL,
            fws_eval.COVERAGE_UNCOVERED,
        )
    keys = {metric["key"] for metric in document["evaluation"]["metrics"]}
    assert "sys.fws.tokens_per_s" in keys
    # WAVE F (D29): the metric set is the filled pipeline's. There is no
    # requests/s figure at all — a request occupies one stream for decode_len x
    # D beats and NONE completes inside the lowered window, so the only
    # request-scale number is the labeled extrapolation, which is what P4 3
    # allows and what D21 requires (one accounting per metric).
    assert keys == {
        "sys.fws.beat",
        "sys.fws.tokens_per_s",
        "sys.fws.per_stream_tokens_per_s",
        "sys.fws.resident_streams",
        "sys.fws.per_token_latency",
    }
    assert not any(key.startswith("sys.fws.requests_per_s") for key in keys)
    extrapolation = document["evaluation"]["extrapolation"]
    assert extrapolation["extrapolated"] is True
    assert "DECODE_LEN x D x BEAT" in extrapolation["basis"]


def test_the_declared_vector_engine_relaxations_ride_the_report(granite):
    """Every knob the card did NOT declare is named in the artifact (D21).

    Two separate conditions, asserted separately: the disclosure EXISTS under a
    ``vector_engine:`` constraint, and that entry's REASON names the undeclared
    knob. (An earlier version asserted `A or B` where both branches were the
    same joined string, so it passed without distinguishing the two shapes.)
    """
    engine = [
        item for item in granite.disclosures if item.constraint.startswith("vector_engine:")
    ]
    assert engine, "an inherited or undeclared engine knob must be disclosed"
    named = {
        knob: [item for item in engine if knob in item.constraint]
        for knob in ("vector_clock_ghz", "vector_pipeline_depth", "state_bytes_per_cycle")
    }
    for knob, entries in named.items():
        assert entries, knob
        assert all(knob in item.reason for item in entries), knob
        assert all(len(item.reason) > 80 for item in entries), knob


def test_the_atlas_export_covers_every_placed_macro(granite):
    document = fws_atlas_export.export_atlas(
        [granite.mapping],
        title="Granite-4.0-H-Tiny",
        duty_cycles=[granite.atlas_duty_cycles()],
        extra_metrics=[granite.atlas_metrics()],
    )
    assert document["schema"] == "fws_atlas/1"
    assert len(document["macros"]) >= len(granite.mapping.tiles)
    assert len(document["tiles"]) == len(granite.mapping.tiles)
    # D23: accuracy is never a factor, anywhere, including in a viz payload.
    assert "accuracy" not in json.dumps(document).lower()


# ---------------------------------------------------------------------------
# Qwen3.5-4B: the linear-attention row (ADJ-1's second P0)
# ---------------------------------------------------------------------------


def test_qwen3_5_4b_linear_attention_prices_on_the_mapped_path():
    """ADJ-1's second P0 no longer skips: it runs.

    P1.5 left it refused BY NAME on attention.output_gate — the delta law was
    wired but no stage placed the gate matrix. C1 places W_g as an ordinary
    analog tile, so this test now proves the delta law prices every linear
    layer of a real model instead of proving a refusal. The end-to-end proof of
    the GATE lives in tests/test_qif_qwen_e2e.py; what is asserted here is the
    thing this module has always asserted — the law, on its consumer.
    """
    hw = _hw(os.path.join(HW_DIR, "fws_cim_qwen3_5_4b.yaml"))
    model = config.parse_config(QWEN_MODEL, "LLM")
    config.validate_configs(hw, model)
    mapping = fws_mapping.build_mapping(hw, model)
    evaluation = fws_eval.evaluate_fws(build_fws_program(mapping))
    delta = [cost for cost in evaluation.pricing.costs if cost.block == "delta_rule"]
    assert delta
    assert all("price_linear_attention_block" in cost.basis for cost in delta)
    peak = mapping.device.vector_lanes * mapping.device.vector_clock_hz
    for cost in delta:
        ops = float((cost.detail or {}).get("ops", 0.0))
        if ops > 0 and cost.duration_s > 0:
            assert ops / cost.duration_s <= peak * (1 + 1e-9)


def test_the_granite_run_places_no_gate_because_granite_declares_none(granite):
    """The C1 narrowing is guarded by the model's own declaration.

    Granite-4.0-H-Tiny sets no attn_output_gate, so its placement, its report
    and its atlas must be exactly what they were before the gate existed.
    """
    assert not bool(granite.mapping.model.attention.output_gate)
    assert not [o for o in granite.mapping.owners() if o.op == "attn_gate_proj"]
    assert not [c for c in granite.pricing.costs if c.block == "attn_output_gate"]


def test_the_shipped_granite_atlas_validates_through_the_real_loader(tmp_path):
    """The emitted artifact, judged by atlas.html's own ATLAS.validate (P5.3).

    The document under docs/qif/atlas is a deliverable someone will open. A
    schema it does not satisfy is a broken page, not a failing test.
    """
    from tests.test_qif_atlas import _run_core

    document = os.path.join(
        PROJECT_ROOT, "docs", "qif", "atlas", "granite_4_0_h_tiny.json"
    )
    # A missing artifact FAILS (never skips). The repo's own precedent is
    # test_run_perf_gpu_results_unchanged_vs_baseline: a silent skip would let
    # deleting the file turn this gate green, which disarms it.
    assert os.path.exists(document), (
        "docs/qif/atlas/granite_4_0_h_tiny.json is the ADJ-1 headline artifact "
        "and is committed; regenerate it with the command in its provenance block"
    )
    out = _run_core(
        """
        const v = ATLAS.validate(doc);
        console.log(JSON.stringify({
          errors: v.errors.map(e => e.rule + ' ' + e.field),
          warnings: v.warnings.map(w => w.rule + ' ' + w.field),
        }));
        """,
        tmp_path,
        document=__import__("pathlib").Path(document),
    )
    verdict = json.loads(out)
    assert verdict["errors"] == []
    assert verdict["warnings"] == []


def test_the_shipped_granite_atlas_carries_p4s_duty_cycles():
    """--priced is what makes it the FULL artifact rather than a placement."""
    path = os.path.join(PROJECT_ROOT, "docs", "qif", "atlas", "granite_4_0_h_tiny.json")
    assert os.path.exists(path), (
        "docs/qif/atlas/granite_4_0_h_tiny.json is committed; a missing artifact "
        "must fail this gate, never skip it"
    )
    with open(path) as handle:
        document = json.load(handle)
    assert "--priced" in document["provenance"]["reference_run"]["command"]
    duty = [
        macro["duty_cycle"]
        for macro in document["macros"]
        if macro.get("duty_cycle") is not None
    ]
    assert duty, "the priced atlas carries no duty cycle"
    assert all(0.0 <= value <= 1.0 for value in duty)


def test_the_shipped_granite_report_is_what_a_fresh_run_produces(granite):
    """The committed artifact cannot rot: it is regenerated and compared.

    The report is a pure function of the two YAMLs — no clock, no path, no
    ordering by hash — so byte equality is the right bar. A law change that
    moves a Granite number now has to move this file too, in the same commit.
    """
    path = os.path.join(
        PROJECT_ROOT, "docs", "qif", "atlas", "granite_4_0_h_tiny_report.json"
    )
    assert os.path.exists(path), (
        "docs/qif/atlas/granite_4_0_h_tiny_report.json is committed; a missing "
        "artifact must fail this gate, never skip it"
    )
    shipped = open(path, encoding="utf-8").read()
    # BYTES, against the exact serialization inference_timing._write_fws_dag_report
    # uses (json.dump(..., indent=2), no trailing newline). Comparing the parsed
    # objects would prove value equality only.
    #
    # WAVE C AUDIT: what this gate proves is byte equality with
    # fws_eval.report_document, regenerated in-process. No run_perf subprocess
    # runs here, so it is not evidence about run_perf's serialization, key order
    # or indent -- the status files and the plan page now say so.
    assert shipped == json.dumps(fws_eval.report_document(granite), indent=2)


def test_the_shipped_granite_atlas_is_what_a_fresh_run_produces(tmp_path):
    """The priced atlas cannot rot either — regenerate and compare BYTES.

    tests/test_qif_mapping.py gives p3_llama7b_tp2.json exactly this gate. The
    Granite atlas is the ADJ-1 HEADLINE artifact and had none, so a law change
    could move a Granite number and leave the shipped picture stale.
    """
    import tools.fws_emit_atlas as emit
    import fws_atlas_export

    _mapping, _rows, document = emit.build(GRANITE_HW, GRANITE_MODEL, priced=True)
    regenerated = fws_atlas_export.write_atlas_json(
        document, os.path.join(str(tmp_path), "granite.json")
    )
    path = os.path.join(
        PROJECT_ROOT, "docs", "qif", "atlas", "granite_4_0_h_tiny.json"
    )
    assert regenerated == open(path, encoding="utf-8").read(), (
        "docs/qif/atlas/granite_4_0_h_tiny.json is stale; regenerate it with "
        "the command in its provenance block"
    )


def test_the_granite_artifacts_name_the_model_and_not_its_pricing_carrier():
    """D21's named owner has to be a name a reader can TRUST.

    granite_4_0_h_tiny_inf.yaml declares model_type: llama because the SwiGLU
    gated MLP is the arithmetic carrier. That is a pricing choice, not the
    model's identity, and every owner, label and title names the model.
    """
    atlas = json.load(
        open(
            os.path.join(
                PROJECT_ROOT, "docs", "qif", "atlas", "granite_4_0_h_tiny.json"
            ),
            encoding="utf-8",
        )
    )
    assert "granite" in atlas["title"].lower()
    system = atlas["systems"][0]
    assert "granite" in system["label"].lower()
    assert system["models"][0]["id"] == "granite_4_0_h_tiny"
    assert system["models"][0]["label"] == "granite_4_0_h_tiny"
    assert atlas["tiles"]
    assert all(tile["owner"]["model"] == "granite_4_0_h_tiny" for tile in atlas["tiles"])
    # The carrier is reconciled, not hidden.
    carrier = [
        item for item in atlas["relaxations"] if item["constraint"] == "model_type_carrier"
    ]
    assert carrier and "llama" in carrier[0]["value"]

    report = json.load(
        open(
            os.path.join(
                PROJECT_ROOT, "docs", "qif", "atlas", "granite_4_0_h_tiny_report.json"
            ),
            encoding="utf-8",
        )
    )
    assert report["model"]["model_id"] == "granite_4_0_h_tiny"
    assert report["model"]["model_type"] == "llama"


def test_the_atlas_and_the_report_size_the_pool_the_same_way(granite):
    """One quantity, one number, across two artifacts (D21).

    macros[].digital_pool and the report's P4.5 pool table are the same thing.
    Before this gate the atlas printed P2.5's per-unit derivation and the report
    printed it multiplied by the measured concurrency, so one macro carried two
    act_lanes figures under one field name in one run.
    """
    atlas = json.load(
        open(
            os.path.join(
                PROJECT_ROOT, "docs", "qif", "atlas", "granite_4_0_h_tiny.json"
            ),
            encoding="utf-8",
        )
    )
    sized = {
        int(entry.macro_id): entry.scaled.total_lanes for entry in granite.pools
    }
    assert sized, "the Granite run measures pool demand on some macro"
    checked = 0
    for macro in atlas["macros"]:
        pool = macro.get("digital_pool")
        if pool is None:
            continue
        macro_id = int(macro["id"].rsplit(".", 1)[1])
        if macro_id not in sized:
            continue
        assert pool["act_lanes"] == sized[macro_id], macro["id"]
        checked += 1
    assert checked == len(sized)
    # A pool area of zero is an ABSENT LAW, and the artifact says which.
    zero = [
        macro["digital_pool"]
        for macro in atlas["macros"]
        if macro.get("digital_pool") and macro["digital_pool"]["area_mm2"] == 0.0
    ]
    assert zero
    assert all("UNCOVERED" in entry["basis"] for entry in zero)


def test_the_granite_headline_numbers_are_the_ones_reported(granite):
    """Pins what the P1.5 status report states, so the two cannot drift apart.

    WAVE F (D29/D30). Granite is a FILLED PIPELINE now: 10 stages (one per
    analog chiplet), D = 10 resident streams, one token out per beat, local
    batch 1, and no lm_head (66 macros gone, 5610 -> 5544). There is no prefill
    figure any more — the streams arrive already prefilled (D25) — and the
    headline is 1 / beat, not tokens over a batched step.
    """
    document = fws_eval.report_document(granite)
    mapping = document["mapping"]
    assert mapping["analog_chips"] == 10
    assert mapping["macros_holding_tiles"] == 5544
    assert mapping["shared_digital_chiplets"] == 10
    assert mapping["serving_regime"] == "filled_pipeline"
    assert mapping["resident_streams"] == 10
    metrics = {metric["key"]: metric["value"] for metric in document["evaluation"]["metrics"]}
    assert "sys.fws.prefill_latency" not in metrics
    # WAVE F REWRITE (D31/D32, P7.8). OLD: beat 43.98 us / 22737 tokens/s on a
    # card that DECLARED 1024 vector lanes nobody measured. NEW: the card
    # declares no engine at all — the width is DERIVED from the beat the analog
    # stages set (220 lanes here, the smallest integer that holds it, no margin)
    # — and the beat is 66.21 us because the derived engine is by construction
    # only as fast as it must be. The old number was not more accurate; it was
    # a machine somebody guessed. D31 is exactly this trade: the width stops
    # being a choice and starts being a consequence, and the headline moves to
    # whatever the consequence is.
    assert metrics["sys.fws.beat"] == pytest.approx(6.620929e-05, rel=1e-3)
    assert metrics["sys.fws.tokens_per_s"] == pytest.approx(15103.6, rel=1e-3)
    assert metrics["sys.fws.per_stream_tokens_per_s"] == pytest.approx(1510.36, rel=1e-3)
    assert metrics["sys.fws.resident_streams"] == 10.0
    # The engine is REPORTED, which is the other half of D31, and its silicon is
    # COMPOSED from the measured 22nm synthesis library (D32).
    silicon = document["evaluation"]["digital_silicon"]
    assert silicon["vector_lanes"] == 220
    assert silicon["vector_lanes_provenance"] == "derived-count"
    assert silicon["library"]["technology"] == "22nm"
    assert silicon["digital_area_mm2_total"] == pytest.approx(52.5867, rel=1e-4)
    # The energy is the WINDOW's, and the window is D - 1 fill beats plus the
    # steady sample: a longer window is more beats of real work, not a
    # different machine. It is reported per run, never per token here.
    #
    # WAVE F (D32): it rose from 1.449e9 pJ because the shared digital chiplet
    # is no longer an UNCOVERED component. Its ops are now charged the composed
    # engine's measured power for their own duration, which is a term that
    # always existed and used to report zero.
    assert document["evaluation"]["energy"]["total_pj"] == pytest.approx(
        9.27615e09, rel=1e-3
    )
    digital = next(
        c
        for c in document["evaluation"]["energy"]["components"]
        if c["key"] == "shared_digital_chiplet"
    )
    assert digital["coverage"] == "partial" and digital["energy_pj"] > 0
    # THE GEORGE CONSTRAINT, on the headline model: every one of the 10 stages
    # holds all 10 streams' state and KV for its own 4 layers.
    residency = document["evaluation"]["state_residency"]
    assert residency["resident_streams"] == 10 and residency["stages"] == 10
    assert residency["max_stage_state_bytes"] == pytest.approx(60413952.0)
    assert residency["verdict"] == "fits"
