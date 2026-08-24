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
        (
            "attention.output_gate",
            {"attention": {"output_gate": True}},
        ),
        (
            "shared_weight_groups",
            {"shared_weight_groups": [{"name": "depth_shared", "layers": [0, 1]}]},
        ),
        ("ffn_dims", {"ffn_dims": {"shared_expert": 1024, "default": 999}}),
    ],
)
def test_inputs_the_mapped_path_still_cannot_price_stay_refused(field, override):
    """The narrowing lifted exactly two inputs. Everything else keeps its name.

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


def test_an_ssm_run_without_a_declared_vector_engine_refuses_by_name():
    """ADJ-4: scan lanes have no default. A missing engine is not a free one."""
    hw = _hw(GRANITE_HW, lambda raw: raw["cim"].pop("cards"))
    mapping = fws_mapping.build_mapping(hw, config.parse_config(GRANITE_MODEL, "LLM"))
    with pytest.raises(cim_timing.EngineCapabilityError) as excinfo:
        fws_eval.evaluate_fws(build_fws_program(mapping))
    assert "vector_lanes" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The run: every block kind reaches a named law
# ---------------------------------------------------------------------------


def test_every_hybrid_block_kind_is_priced_by_a_named_law(granite):
    laws = {}
    for cost in granite.pricing.costs:
        laws.setdefault(cost.block, set()).add(cost.basis.split("(")[0].strip())
    assert "ssm_scan" in laws, sorted(laws)
    assert any("price_ssm_block" in basis for basis in laws["ssm_scan"])
    assert any("prefill_attention_timing" in b for b in laws["attention_qk"])
    for block in ("ssm_in_proj", "ssm_out_proj", "ffn1_routed", "ffn2_shared", "lm_head"):
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
    """The SSD scan runs AT the declared peak: the engine, not the array, binds."""
    device = granite.mapping.device
    peak = device.vector_lanes * device.vector_clock_hz
    scan = next(
        cost
        for cost in granite.pricing.costs
        if cost.block == "ssm_scan" and cost.phase == "prefill"
    )
    utilisation = float(scan.detail["ops"]) / scan.duration_s / peak
    assert utilisation == pytest.approx(1.0, rel=1e-3)
    # ... and it dwarfs the analog projections that feed it, which is the
    # finding, not a bug (D5's "heavy digital co-compute", for a recurrence).
    in_proj = next(
        cost
        for cost in granite.pricing.costs
        if cost.block == "ssm_in_proj" and cost.phase == "prefill"
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
    for block, phase in (("qkv", "prefill"), ("ffn1_routed", "prefill"), ("qkv", "decode")):
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
    """ceil(B*S * top_k * alpha / E) — Granite: 7168 * 6 / 64 = 672."""
    params = granite.mapping.device.params
    owner_tokens = granite.serving.batch * granite.serving.prefill_len
    expected = math.ceil(owner_tokens * params.top_k * params.expert_imbalance_factor
                         / params.num_experts)
    assert expected == 672
    cost = next(
        c
        for c in granite.pricing.costs
        if c.block == "ffn1_routed" and c.phase == "prefill"
    )
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
    # The Granite point lowers 3 of 256 decode steps, so the requests/s figure
    # is a WINDOW rate and carries the name that says so (D21). An unqualified
    # "completed requests per second" here would be a second, contradicting
    # value for the quantity the extrapolation block prices.
    assert "sys.fws.requests_per_s_lowered_window" in keys
    assert "sys.fws.requests_per_s" not in keys
    assert document["evaluation"]["extrapolation"]["extrapolated"] is True


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
# Qwen3.5-4B: the linear-attention row, run only if it prices cleanly
# ---------------------------------------------------------------------------


def test_qwen3_5_4b_linear_attention_prices_or_refuses_by_name():
    """ADJ-1's second P0. It runs only if the delta law is wired end to end.

    If it does not price, the failure must NAME the missing law rather than
    produce a number — that is the whole P1.4 rule, and it does not stop
    applying because a model is on a plan page.
    """
    hw = _hw(GRANITE_HW)
    model = config.parse_config(QWEN_MODEL, "LLM")
    try:
        config.validate_configs(hw, model)
    except ValueError as error:
        # The refusal must name the model_param field that has no law. Today it
        # is attention.output_gate: Qwen3.5's gated attention output projection
        # is a weight matrix no stage places (the LINEAR-attention block's own
        # gate IS placed, which is why the delta law alone was not enough).
        message = str(error)
        assert "model_param." in message
        assert any(
            field in message
            for field in ("output_gate", "linear_attn", "layer_plan", "ffn_dims")
        ), message
        pytest.skip("Qwen3.5-4B is refused BY NAME (%s), which is the correct "
                    "answer until the named input has a law" % message.split(":")[1].strip()[:60])
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
    # objects would prove value equality only, and the status board claims byte
    # equality with run_perf's own output.
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
    """Pins what the P1.5 status report states, so the two cannot drift apart."""
    document = fws_eval.report_document(granite)
    mapping = document["mapping"]
    assert mapping["analog_chips"] == 10
    assert mapping["macros_holding_tiles"] == 5610
    assert mapping["shared_digital_chiplets"] == 10
    metrics = {metric["key"]: metric["value"] for metric in document["evaluation"]["metrics"]}
    # Prefill is scan-bound: 36 SSD layers on a 1024-lane engine.
    assert metrics["sys.fws.prefill_latency"] == pytest.approx(1.0795, rel=1e-3)
    assert metrics["sys.fws.tokens_per_s"] == pytest.approx(4485.9, rel=1e-3)
    assert document["evaluation"]["energy"]["total_pj"] == pytest.approx(
        1.9499e11, rel=1e-3
    )
