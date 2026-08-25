"""QIF P7.4/P7.5 — the FRONTIER, regime v2 (D29-D32).

What this file pins, and why each is a wrong NUMBER rather than a crash if it
rots:

  1. **The axes are the v2 axes.** Stage granularity (which under D29's default
     stage plan moves the chip split, the stage count, D, the beat and the
     per-stage state bill AT ONCE), mux/bank depth where the card admits it,
     and chip capacity. Digital provisioning is not among them: D31 derives the
     engine per point and D32 prices it, so ``vector_lanes`` is in
     :data:`RETIRED_AXES` and every candidate that sets it says so.
  2. **The silicon accounting has three terms and they sum.** Analog macro
     slots at the cell floor (Invariant W), shared digital chiplets composed
     from the measured library, and the per-macro digital pools composed from
     the same one. A total that quietly dropped the pools would under-report
     every point by a term that MOVES with the chip-capacity axis.
  3. **The ladder is a ladder.** It starts at the declared balanced
     initializer, steps ONE axis ONE rung, climbs toward throughput, trims idle
     silicon off every rung it accepts (D28), descends toward area, and keeps
     every point it looked at including the ones it declined. The walks here
     run on a SYNTHETIC evaluator with hand-computed numbers, so the walk's own
     arithmetic is checked without waiting for a model.
  4. **The knee is the bend, or nothing.** Max distance above the chord of the
     front, normalized over the front's own spans; a straight front and a front
     with no interior both refuse to name a point.
  5. **The state bill decides feasibility (D29).** A stage plan whose per-stage
     residency exceeds the declared budget is refused at the ``state`` stage
     with the stage, the layers and the bytes named — not with a word.
  6. **The checked-in curves regenerate.** Both frontier artifacts are
     re-walked end to end and compared, and both frontier configs are the
     shipped machine plus the sweep block and nothing else.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HW_DIR = PROJECT_ROOT / "configs" / "hardware-config"
MODEL_DIR = PROJECT_ROOT / "configs" / "model-config"
DSE_DIR = PROJECT_ROOT / "docs" / "qif" / "dse"

GRANITE_FRONTIER_HW = HW_DIR / "fws_cim_granite_tiny_frontier.yaml"
GRANITE_HW = HW_DIR / "fws_cim_granite_tiny.yaml"
GRANITE_MODEL = MODEL_DIR / "granite_4_0_h_tiny_inf.yaml"
QWEN_FRONTIER_HW = HW_DIR / "fws_cim_qwen3_5_4b_frontier.yaml"
QWEN_HW = HW_DIR / "fws_cim_qwen3_5_4b.yaml"
QWEN_MODEL = MODEL_DIR / "qwen3_5_4b_inf.yaml"
MOE_HW = HW_DIR / "fws_cim_moe.yaml"
MOE_MODEL = MODEL_DIR / "moe_small_fws_inf.yaml"

GRANITE_ART = DSE_DIR / "granite_4_0_h_tiny_frontier"
QWEN_ART = DSE_DIR / "qwen3_5_4b_frontier"


def _load_tool():
    """Import tools/fws_qif_dse.py in-process (tools/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "fws_qif_dse_frontier_under_test", str(PROJECT_ROOT / "tools" / "fws_qif_dse.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DSE = _load_tool()


# ---------------------------------------------------------------------------
# A SYNTHETIC evaluator: the ladder's own arithmetic, checked by hand
# ---------------------------------------------------------------------------


class FakeSpace:
    """A design space with a closed form, so the walk can be checked by hand.

    ``throughput`` depends only on axis ``a`` and has DIMINISHING RETURNS;
    ``area`` rises with both axes, so axis ``b`` is pure idle silicon and the
    trim phase has something real to remove. ``needs`` makes a high ``a``
    infeasible unless ``b`` is large enough — the coupling that a chip capacity
    has with a stage plan, and the reason the ladder has a repair rule.
    """

    THROUGHPUT = {1: 100.0, 2: 300.0, 3: 380.0, 4: 400.0}

    def __init__(self, needs=None):
        self.needs = needs or {}
        self.calls = []

    def __call__(self, point):
        a, b = int(point["layers_per_chip"]), int(point["arrays_per_chip"])
        self.calls.append((a, b))
        candidate = {
            "id": f"c{len(self.calls) - 1:03d}",
            "knobs": dict(point),
            "ok": True,
            "fail_stage": None,
            "fail_message": None,
            "stages": {},
            "notes": [],
            "metrics": {"tokens_per_s": self.THROUGHPUT[a]},
            "silicon": {"total_silicon_mm2": 10.0 * a + 5.0 * (b - 1)},
        }
        if b < self.needs.get(a, 0):
            candidate.update(
                ok=False,
                fail_stage="mapping",
                fail_message=f"a = {a} needs b >= {self.needs[a]}, and b = {b}.",
                metrics={},
                silicon={},
            )
        return candidate


def _fake_spec(**overrides):
    block = {
        "label": "synthetic",
        "search": "ladder",
        "initializer": {"layers_per_chip": 2, "arrays_per_chip": 3},
        "axes": {
            "layers_per_chip": [1, 2, 3, 4],
            "arrays_per_chip": [1, 2, 3],
        },
    }
    block.update(overrides)
    return DSE.SweepSpec.from_raw({DSE.DSE_BLOCK: block}, "synthetic")


def test_the_ladder_trims_the_initializer_before_it_walks_anywhere():
    """D28: idle silicon is a provisioning bug, so the start point is trimmed.

    The initializer is (a = 2, b = 3): 300 tokens/s at 10*2 + 5*2 = 30 mm2.
    Axis b buys nothing, so the trim walks b down to 1 in two steps and both
    walks start from (2, 1) at 300 tokens/s and 20 mm2. If the trim did not
    run first, the descend phase's first two steps would be undoing the
    declarer's over-provisioning and would be reported as trades.
    """
    space = FakeSpace()
    candidates, trail = DSE.ladder_candidates(_fake_spec(), space)
    trims = [row for row in trail if row["phase"] == "trim" and row.get("accepted")]
    assert [row["knobs"]["arrays_per_chip"] for row in trims[:2]] == [2, 1]
    by_knobs = {(c["knobs"]["layers_per_chip"], c["knobs"]["arrays_per_chip"]): c
                for c in candidates}
    assert by_knobs[(2, 1)]["silicon"]["total_silicon_mm2"] == 20.0
    assert by_knobs[(2, 1)]["metrics"]["tokens_per_s"] == 300.0


def test_the_ladder_climbs_on_throughput_and_descends_on_area():
    """The two outward walks, and the chain each one leaves behind.

    From the trimmed start (a = 2), the climb takes a = 3 (+80 tokens/s for
    +10 mm2) and then a = 4 (+20 for +10); the descend takes a = 1 (-10 mm2 for
    -200 tokens/s). Every accepted step moves exactly ONE axis by exactly ONE
    rung, which is what makes the walk a ladder rather than a search.
    """
    space = FakeSpace()
    candidates, trail = DSE.ladder_candidates(_fake_spec(), space)
    climbs = [row for row in trail if row["phase"] == "climb" and row.get("accepted")]
    assert [row["knobs"]["layers_per_chip"] for row in climbs] == [3, 4]
    assert all(row["axis"] == "layers_per_chip" for row in climbs)
    descends = [row for row in trail if row["phase"] == "descend" and row.get("accepted")]
    assert [row["knobs"]["layers_per_chip"] for row in descends] == [1]
    # Every accepted step is one rung on one axis.
    for row in trail:
        if row.get("accepted") and row.get("axis"):
            assert abs(row["direction"]) == 1


def test_the_ladder_evaluates_far_fewer_points_than_the_cross_product():
    """The whole reason the walk exists: it is directional, not exhaustive.

    The declared space is 4 x 3 = 12 points. The walk visits the ones it steps
    through and the neighbours of those, and NOTHING is estimated for the rest:
    a point the ladder never reached simply is not in the report, which is a
    different statement from a point that was sampled and interpolated.
    """
    space = FakeSpace()
    candidates, _ = DSE.ladder_candidates(_fake_spec(), space)
    assert _fake_spec().size == 12
    assert len(candidates) < 12
    assert len(space.calls) == len(candidates) == len(set(space.calls))


def test_every_point_the_walk_declined_is_still_in_the_report():
    """A priced point that was rejected is DATA about the frontier.

    (a = 1, b = 3) is evaluated as a neighbour of the initializer and declined
    by both the trim (throughput falls) and the climb (throughput falls); it is
    still a candidate row with its own numbers.
    """
    space = FakeSpace()
    candidates, trail = DSE.ladder_candidates(_fake_spec(), space)
    knobs = {(c["knobs"]["layers_per_chip"], c["knobs"]["arrays_per_chip"])
             for c in candidates}
    assert (1, 3) in knobs
    accepted = {row["accepted"] for row in trail if row.get("accepted")}
    declined = [c for c in candidates if c["id"] not in accepted]
    assert declined and all(c["metrics"] or not c["ok"] for c in declined)


def test_a_refused_step_is_repaired_on_the_declared_repair_axis():
    """Stage granularity and chip capacity are coupled, so a step gets repaired.

    Here a = 4 needs b >= 3. The climb's one-rung step to a = 4 at b = 1 is
    refused at the mapping stage; the ladder retries it at the SMALLEST
    declared b that admits it (b = 3) and takes that instead. Both the refusal
    and the repaired point are in the report, and the repaired point says which
    refusal it came from.
    """
    space = FakeSpace(needs={4: 3})
    candidates, trail = DSE.ladder_candidates(
        _fake_spec(repair_axis="arrays_per_chip"), space
    )
    refused = [c for c in candidates
               if c["knobs"] == {"layers_per_chip": 4, "arrays_per_chip": 1}]
    assert refused and refused[0]["fail_stage"] == "mapping"
    repaired = [c for c in candidates
                if c["knobs"] == {"layers_per_chip": 4, "arrays_per_chip": 3}]
    assert repaired and repaired[0]["ok"]
    assert repaired[0]["ladder"]["repaired_from"] == refused[0]["id"]
    assert "REPAIRED STEP" in "\n".join(repaired[0]["notes"])
    # Without the repair axis the same walk simply stops at a = 2.
    plain, _ = DSE.ladder_candidates(_fake_spec(), FakeSpace(needs={4: 3}))
    assert not [c for c in plain if c["ok"] and c["knobs"]["layers_per_chip"] == 4]


def test_a_ladder_whose_initializer_is_infeasible_stops_and_says_so():
    """A walk needs feasible ground to stand on; it does not guess at one."""
    space = FakeSpace(needs={2: 4})  # b maxes out at 3, so a = 2 never fits
    candidates, trail = DSE.ladder_candidates(_fake_spec(), space)
    assert len(candidates) == 1 and not candidates[0]["ok"]
    assert trail[-1]["accepted"] is None
    assert "INFEASIBLE at stage mapping" in trail[-1]["reason"]


# ---------------------------------------------------------------------------
# The knee
# ---------------------------------------------------------------------------


def _point(cid, area, throughput):
    return {
        "id": cid,
        "ok": True,
        "knobs": {},
        "metrics": {"tokens_per_s": float(throughput)},
        "silicon": {"total_silicon_mm2": float(area)},
    }


def test_the_knee_is_the_front_point_furthest_above_its_own_chord():
    """Hand arithmetic on a four-point front.

    areas 10/20/30/40, throughputs 100/300/380/400. Normalized over the front's
    own spans (30 mm2 x 300 tokens/s) the points sit at x = 0, 1/3, 2/3, 1 and
    y = 0, 2/3, 14/15, 1, so the distances above the chord y = x are
    0, 1/3, 4/15, 0. The knee is the second point at exactly 1/3.
    """
    front = [_point("c000", 10, 100), _point("c001", 20, 300),
             _point("c002", 30, 380), _point("c003", 40, 400)]
    knee = DSE.knee_of(front, [c["id"] for c in front])
    assert knee["point"] == "c001"
    assert knee["distance"] == pytest.approx(1.0 / 3.0)
    assert "c000" in knee["basis"] and "c003" in knee["basis"]


def test_a_straight_front_has_no_knee_and_says_so():
    """Silicon buying throughput at one rate everywhere is not a bend."""
    front = [_point("c000", 10, 100), _point("c001", 20, 200),
             _point("c002", 30, 300)]
    knee = DSE.knee_of(front, [c["id"] for c in front])
    assert knee["point"] is None
    assert "STRAIGHT" in knee["basis"]


def test_a_front_with_no_interior_has_no_knee():
    front = [_point("c000", 10, 100), _point("c001", 20, 300)]
    knee = DSE.knee_of(front, [c["id"] for c in front])
    assert knee["point"] is None
    assert "INTERIOR" in knee["basis"]


# ---------------------------------------------------------------------------
# The sweep block: what regime v2 admits and what it refuses
# ---------------------------------------------------------------------------


def test_stage_granularity_and_chip_capacity_are_axes_and_the_engine_is_not():
    """D31/ADJ-9: the scan/vector engine is not a design point any more.

    ADJ-9 REWRITE. OLD CLAIM: ``vector_lanes`` stays SPELLABLE and is named in
    a RETIRED register, and a candidate that sets it carries the retirement in
    its own notes. That is what Wave F built, and the labelled axis went on
    producing a checked-in artifact whose whole spread came from overriding the
    derivation. NEW CLAIM: it is REFUSED BY NAME at parse and it has no field
    to write, so no sweep can carry it at all — the frozen Wave-D artifact is
    frozen precisely because it can never be re-walked.
    """
    assert "layers_per_chip" in DSE.AXIS_TARGETS
    assert "layers_per_stage" in DSE.AXIS_TARGETS
    assert "arrays_per_chip" in DSE.AXIS_TARGETS
    assert "bank_depth" in DSE.AXIS_TARGETS
    assert set(DSE.REFUSED_AXES) == {"vector_lanes"}
    assert "vector_lanes" not in DSE.AXIS_TARGETS
    assert not hasattr(DSE, "RETIRED_AXES")
    reason = DSE.REFUSED_AXES["vector_lanes"]
    assert "D31/ADJ-9" in reason and "ANALOG m-pass" in reason
    with pytest.raises(DSE.QifDseUsageError, match="REFUSED"):
        DSE.SweepSpec.from_raw(
            {DSE.DSE_BLOCK: {"axes": {"vector_lanes": [512]}}}, "test"
        )
    # The frontier sweeps declare no refused axis, and every sweep's payload
    # carries the refusal register so a reader of the artifact sees it too.
    for path in (GRANITE_FRONTIER_HW, QWEN_FRONTIER_HW):
        axes = yaml.safe_load(path.read_text())[DSE.DSE_BLOCK]["axes"]
        assert not set(axes) & set(DSE.REFUSED_AXES), path.name


def test_the_ladder_refuses_a_missing_or_off_ladder_initializer():
    """A ladder starts where somebody put it, never where the tool guessed."""
    with pytest.raises(DSE.QifDseUsageError) as missing:
        _fake_spec(initializer=None)
    assert "initializer" in str(missing.value)
    with pytest.raises(DSE.QifDseUsageError) as partial:
        _fake_spec(initializer={"layers_per_chip": 2})
    assert "arrays_per_chip" in str(partial.value)
    with pytest.raises(DSE.QifDseUsageError) as off:
        _fake_spec(initializer={"layers_per_chip": 9, "arrays_per_chip": 1})
    assert "declared rungs" in str(off.value)
    with pytest.raises(DSE.QifDseUsageError) as pointless:
        _fake_spec(search="cross_product")
    assert "cross_product" in str(pointless.value)


def test_the_repair_axis_must_be_a_swept_ladder():
    with pytest.raises(DSE.QifDseUsageError) as exc:
        _fake_spec(repair_axis="bank_depth")
    assert "not a swept axis" in str(exc.value)


def test_the_state_stage_is_last_because_it_needs_the_priced_timeline():
    """``config < mapping < budget < lowering < pricing < memory < state``."""
    assert DSE.STAGES[-1] == "state"
    assert DSE.STAGES.index("state") > DSE.STAGES.index("pricing")


# ---------------------------------------------------------------------------
# The silicon accounting: three terms, and they sum
# ---------------------------------------------------------------------------


#: The MoE-small machine as a MAPPED, LIBRARY-PRICED fws_cim run.
#:
#: The shipped fws_cim_moe.yaml declares neither a `mapping:` block nor device
#: cards, so it lowers under the retired lockstep regime and synthesizes its
#: cards. Both are added HERE rather than in the shipped file: this file needs a
#: fast filled-pipeline point whose digital silicon is composed from the
#: measured library (D32), and the shipped fixture belongs to the bridge tests.
def _moe_raw(mutate=None):
    raw = copy.deepcopy(yaml.safe_load(MOE_HW.read_text()))
    raw["cim"]["cards"] = {
        "ctt": {"kind": "analog_macro", "device": "ctt"},
        "sa": {"kind": "digital_chiplet", "synthesis_library": "22nm"},
    }
    raw["mapping"] = {"parallelism": {"tp": 1}, "decode_window": 2}
    if mutate is not None:
        mutate(raw)
    return raw


#: MoE-small's shipped model config declares global_batch_size 4, which D29
#: refuses on a filled-pipeline run by name. The batch-1 twin is written to a
#: temp file rather than edited in place: the shipped one is the LOCKSTEP
#: fixture the bridge tests read.
def _moe_model_path(tmp_path):
    raw = copy.deepcopy(yaml.safe_load(MOE_MODEL.read_text()))
    raw["model_param"]["global_batch_size"] = 1
    # D30: embedding and lm_head are dropped entirely on the mapped path, and a
    # model that still declares them is refused by name.
    raw["model_param"]["disable_embedding_unembedding"] = True
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "moe_small_batch1.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def _moe_mapping(mutate=None):
    import config as config_module
    import fws_mapping

    raw = _moe_raw(mutate)
    converted = copy.deepcopy(raw)
    config_module.convert(converted)
    hw = config_module.HWConfig.from_dict(converted)
    model = config_module.parse_config(str(MOE_MODEL), "LLM")
    model.model_config.global_batch_size = 1
    model.model_config.disable_embedding_unembedding = True
    return hw, model, fws_mapping.build_mapping(hw, model, model_id="MoE-small")


def test_the_silicon_total_is_analog_plus_chiplets_plus_pools():
    """One accounting for the machine's silicon (D21/D32).

    The per-macro digital pool is composed from the SAME measured library as
    the shared chiplet, and it is priced per enumerated macro SLOT — the term
    that moves with the chip-capacity axis. A total that left it out would
    under-report every point on the frontier by a whole device class.
    """
    _hw, _model, mapping = _moe_mapping()
    chiplets = int(mapping.summary()["shared_digital_chiplets"])
    silicon = DSE._silicon(mapping, chiplets)
    slots = silicon["analog_macro_slots"]
    assert silicon["analog_macro_silicon_mm2"] == pytest.approx(
        slots * silicon["macro_footprint_mm2"]
    )
    assert silicon["shared_digital_silicon_mm2"] == pytest.approx(
        chiplets * silicon["shared_digital_area_mm2_per_chiplet"]
    )
    assert silicon["macro_pool_silicon_mm2"] == pytest.approx(
        slots * silicon["macro_pool_area_mm2_per_macro"]
    )
    assert silicon["digital_silicon_mm2"] == pytest.approx(
        silicon["shared_digital_silicon_mm2"] + silicon["macro_pool_silicon_mm2"]
    )
    assert silicon["total_silicon_mm2"] == pytest.approx(
        silicon["analog_macro_silicon_mm2"] + silicon["digital_silicon_mm2"]
    )
    assert "per-macro digital pools" in silicon["basis"]


def test_a_card_with_no_synthesis_library_reports_the_pool_as_uncovered():
    """An ABSENT law is named, never reported as a measured zero (D21)."""
    def drop_library(raw):
        raw["cim"]["cards"]["sa"].pop("synthesis_library", None)

    _hw, _model, mapping = _moe_mapping(drop_library)
    silicon = DSE._silicon(mapping, int(mapping.summary()["shared_digital_chiplets"]))
    assert silicon["macro_pool_area_provenance"] == "absent"
    assert silicon["macro_pool_silicon_mm2"] == 0.0
    assert any("per-macro digital pool" in term for term in silicon["uncovered_terms"])


# ---------------------------------------------------------------------------
# D29's feasibility check, on a real timeline
# ---------------------------------------------------------------------------


def _moe_sweep(tmp_path, block, mutate=None, **kwargs):
    raw = _moe_raw(mutate)
    raw[DSE.DSE_BLOCK] = block
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "moe_frontier.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return DSE.run_sweep(
        str(path),
        str(_moe_model_path(tmp_path)),
        model_id="MoE-small",
        output_dir=str(tmp_path / "out"),
        quiet=True,
        **kwargs,
    )


def test_every_point_reports_D_the_beat_and_the_state_bill(tmp_path):
    """D29's quantities are on EVERY row, because they define the point.

    D is the resident-stream count and it IS the stage count — derived from the
    stage plan, never configured — and the per-stage bill is D streams' state
    and KV for that stage's layers.
    """
    exit_code, payload = _moe_sweep(
        tmp_path,
        {
            "label": "MoE-small residency",
            "axes": {"layers_per_chip": [6, 4]},
        },
    )
    assert exit_code == 0
    for candidate in payload["candidates"]:
        assert candidate["ok"], candidate.get("fail_message")
        pipeline = candidate["pipeline"]
        assert pipeline["filled"] is True
        assert pipeline["beat_s"] > 0
        bill = candidate["state_bill"]
        assert bill["measured"] is True
        # D = the stage count, and the stage plan is one stage per chip here.
        assert bill["resident_streams"] == candidate["placement"]["analog_chips"]
        assert bill["stages"] == bill["resident_streams"]
        # tokens/s IS 1/beat (D29): one token exits per beat.
        assert candidate["metrics"]["tokens_per_s"] == pytest.approx(
            1.0 / pipeline["beat_s"], rel=1e-9
        )
        # The bill closes: the worst stage is the max over the per-stage rows,
        # and each row is its own recurrent state plus its own KV.
        rows = bill["per_stage"]
        assert bill["max_stage_state_bytes"] == pytest.approx(
            max(row["state_bytes"] for row in rows)
        )
        for row in rows:
            assert row["state_bytes"] == pytest.approx(
                row["recurrent_state_bytes"] + row["kv_bytes"]
            )
        assert bill["budget_verdict"] == "undeclared"
        # D28: every device class the mapping instantiated has a row.
        assert candidate["utilization"]
        assert candidate["binding_device_class"] in {
            row["device_class"] for row in candidate["utilization"]
        }


def test_a_stage_plan_over_the_declared_budget_is_refused_with_its_bytes(tmp_path):
    """INFEASIBLE BY MEMORY, named (D29).

    The budget is set to one byte under the finer plan's own measured
    worst-stage bill, so the refusal is a statement about THAT plan and not
    about a number pulled out of the air: the coarse plan still fits.
    """
    _code, measured = _moe_sweep(
        tmp_path / "measure",
        {"label": "measure", "axes": {"layers_per_chip": [6, 3]}},
    )
    bills = {
        candidate["knobs"]["layers_per_chip"]:
            candidate["state_bill"]["max_stage_state_bytes"]
        for candidate in measured["candidates"]
    }
    assert bills[3] > bills[6], "the finer stage plan must hold MORE state (D29)"
    budget = bills[3] - 1.0

    exit_code, payload = _moe_sweep(
        tmp_path / "budget",
        {
            "label": "budget",
            "max_stage_state_bytes": budget,
            "axes": {"layers_per_chip": [6, 3]},
        },
    )
    assert exit_code == 0
    by_knob = {c["knobs"]["layers_per_chip"]: c for c in payload["candidates"]}
    assert by_knob[6]["ok"] and by_knob[6]["state_bill"]["budget_verdict"] == "fits"
    refused = by_knob[3]
    assert not refused["ok"] and refused["fail_stage"] == "state"
    assert "INFEASIBLE BY MEMORY (D29)" in refused["fail_message"]
    assert refused["state_bill"]["budget_verdict"] == "VIOLATED"
    violating = refused["state_bill"]["budget_violating_stages"]
    assert violating and all(
        row["state_bytes"] > budget for row in violating
    )
    # The point was PRICED before it was refused, so its throughput is real.
    assert refused["metrics"]["tokens_per_s"] > by_knob[6]["metrics"]["tokens_per_s"]
    summary = payload["state_bill_summary"]
    assert [row["id"] for row in summary["infeasible_by_state"]] == [refused["id"]]
    assert summary["budget_bytes"] == budget
    report = (tmp_path / "budget" / "out" / "dse_report.md").read_text()
    assert "Infeasible by RESIDENCY" in report
    assert "The state bill (D29)" in report


def test_a_declared_budget_on_a_lockstep_run_refuses_rather_than_passes(tmp_path):
    """A check that COULD NOT BE MADE is not a pass (D21).

    The retired lockstep regime has no resident streams, so it measures no
    per-stage residency; a declared budget there is refused by name instead of
    being silently satisfied.
    """
    raw = _moe_raw()
    raw["mapping"]["regime"] = "lockstep"
    raw[DSE.DSE_BLOCK] = {
        "label": "lockstep",
        "max_stage_state_bytes": 1.0,
        "axes": {"layers_per_chip": [6]},
    }
    path = tmp_path / "lockstep.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    exit_code, payload = DSE.run_sweep(
        str(path),
        str(MOE_MODEL),
        model_id="MoE-small",
        output_dir=str(tmp_path / "out"),
        quiet=True,
    )
    assert exit_code == 3, payload["candidates"][0].get("fail_message")
    candidate = payload["candidates"][0]
    assert candidate["fail_stage"] == "state"
    assert "measures NO per-stage residency" in candidate["fail_message"]


# ---------------------------------------------------------------------------
# The checked-in curves
# ---------------------------------------------------------------------------


CURVES = (
    ("granite", GRANITE_FRONTIER_HW, GRANITE_HW, GRANITE_MODEL, GRANITE_ART,
     "Granite-4.0-H-Tiny"),
    ("qwen", QWEN_FRONTIER_HW, QWEN_HW, QWEN_MODEL, QWEN_ART, "Qwen3.5-4B"),
)


@pytest.mark.parametrize("name,frontier,shipped,model,art,model_id", CURVES)
def test_the_frontier_config_is_the_shipped_machine_plus_the_sweep_block(
    name, frontier, shipped, model, art, model_id
):
    """A fork would drift the first time the shipped point is retuned, and the
    regenerate gate would keep passing while the curve priced a machine that no
    longer exists."""
    swept = yaml.safe_load(frontier.read_text())
    assert DSE.DSE_BLOCK in swept
    assert {k: v for k, v in swept.items() if k != DSE.DSE_BLOCK} == yaml.safe_load(
        shipped.read_text()
    )
    block = swept[DSE.DSE_BLOCK]
    assert block["search"] == "ladder"
    assert block["repair_axis"] == "arrays_per_chip"
    assert set(block["axes"]) == {"layers_per_chip", "arrays_per_chip", "bank_depth"}
    # A declared residency budget is the config's OWN L2 tier and never a
    # number invented for the sweep. Qwen declares none, because its per-stage
    # bill is above that tier at every stage plan and a budget there would
    # refuse the whole sweep; the config says so in its own comment.
    raw = copy.deepcopy(swept)
    raw.pop(DSE.DSE_BLOCK)
    import config as config_module

    config_module.convert(raw)
    if "max_stage_state_bytes" in block:
        assert block["max_stage_state_bytes"] == raw["tech_param"]["SRAM-L2"]["size"]
    else:
        assert "DELIBERATELY NOT DECLARED" in frontier.read_text()


def _payload(art):
    return json.loads((art / "dse_report.json").read_text())


@pytest.mark.parametrize("name,frontier,shipped,model,art,model_id", CURVES)
def test_the_checked_in_curve_is_a_real_frontier_with_its_state_bill(
    name, frontier, shipped, model, art, model_id
):
    """What a P7.5 curve has to carry: a front, a knee, D, the bill, utilization."""
    payload = _payload(art)
    assert payload["sweep"]["search"] == "ladder"
    assert payload["num_valid"] >= 3
    assert len(payload["front_ids"]) >= 2
    front = [c for c in payload["candidates"] if c["id"] in payload["front_ids"]]
    # A front is monotone by construction: sorted by area, throughput rises.
    ordered = sorted(front, key=lambda c: c["silicon"]["total_silicon_mm2"])
    rates = [c["metrics"]["tokens_per_s"] for c in ordered]
    assert rates == sorted(rates)
    # A point that died before it was priced has no bill to report; one that
    # died AT the residency check has the fullest bill of all.
    priced = [
        c for c in payload["candidates"]
        if c["ok"] or c["fail_stage"] == "state"
    ]
    assert priced
    for candidate in priced:
        bill = candidate["state_bill"]
        assert bill["measured"] is True
        assert bill["resident_streams"] >= 1
        assert bill["max_stage_state_bytes"] > 0
        assert candidate["utilization"]
        assert candidate["derived_digital"]["vector_lanes_provenance"]
        # ADJ-9 (D31-v2): the engine is derived to the ANALOG FLOOR at every
        # point of the curve, not only on the shipped machine. A point whose
        # engine was sized against the beat would be a machine left
        # digital-bound by a minimal derivation, and the curve would be a
        # picture of that instead of of the frontier.
        derived = candidate["derived_digital"]
        assert derived["sizing_target"] == "analog_stage_time"
        assert derived["stages_sized"] >= 1
        assert derived["analog_bound_stages"] == derived["stages_sized"]
        assert 0.0 < derived["engine_duty"] <= 1.0
    # Invariant W (D27): the analog term is the cell floor, so every mm2 the
    # front spends over its cheapest point is DIGITAL or enumerated-but-unowned
    # analog slots — never a bigger model footprint.
    assert len({c["silicon"]["macro_pool_area_provenance"] for c in front}) == 1
    assert all(
        c["silicon"]["shared_digital_area_provenance"] == "composed-measured"
        for c in front
    )


@pytest.mark.parametrize("name,frontier,shipped,model,art,model_id", CURVES)
def test_the_checked_in_curve_names_its_knee(
    name, frontier, shipped, model, art, model_id
):
    """The knee is recomputed here from the front's two axes only."""
    payload = _payload(art)
    knee = payload["knee"]
    front = [c for c in payload["candidates"] if c["id"] in payload["front_ids"]]
    front.sort(key=lambda c: c["silicon"]["total_silicon_mm2"])
    if knee["point"] is None:
        assert "STRAIGHT" in knee["basis"] or "INTERIOR" in knee["basis"]
        return
    areas = [c["silicon"]["total_silicon_mm2"] for c in front]
    rates = [c["metrics"]["tokens_per_s"] for c in front]
    span_a, span_r = areas[-1] - areas[0], max(rates) - min(rates)
    distances = {
        c["id"]: (c["metrics"]["tokens_per_s"] - min(rates)) / span_r
        - (c["silicon"]["total_silicon_mm2"] - areas[0]) / span_a
        for c in front
    }
    assert knee["point"] == max(distances, key=distances.get)
    assert knee["distance"] == pytest.approx(distances[knee["point"]])


@pytest.mark.parametrize("name,frontier,shipped,model,art,model_id", CURVES)
def test_the_checked_in_curve_reports_its_infeasible_points_by_name(
    name, frontier, shipped, model, art, model_id
):
    """Infeasible is a verdict with bytes and a stage behind it, or it is a word."""
    payload = _payload(art)
    summary = payload["state_bill_summary"]
    assert summary["budget_bytes"] == (payload["sweep"]["max_stage_state_bytes"] or None)
    if summary["budget_bytes"] is None:
        assert not summary["infeasible_by_state"]
        assert "not declared" in summary["budget_basis"]
    for row in summary["infeasible_by_state"]:
        assert "INFEASIBLE BY MEMORY (D29)" in row["verdict"]
        assert row["violating_stages"]
        assert all(
            stage["state_bytes"] > summary["budget_bytes"]
            for stage in row["violating_stages"]
        )
    for row in summary["infeasible_by_memory"]:
        assert row["verdicts"]
        for verdict in row["verdicts"]:
            assert verdict["scope"] and verdict["tier"] and verdict["owner"]
    # Rendered from the CHECKED-IN payload rather than read off disk: the .md
    # is gitignored repo-wide (`*.md`), so the JSON is the artifact and the
    # renderer is what this asserts.
    text = DSE.render_markdown(payload)
    assert "## The knee" in text
    assert "## The state bill (D29)" in text
    assert "## The ladder (how the space was walked)" in text
    assert "## Per-device-class utilization (D28)" in text


@pytest.mark.parametrize("name,frontier,shipped,model,art,model_id", CURVES)
def test_the_checked_in_curve_verifies_its_own_emitted_config(
    name, frontier, shipped, model, art, model_id
):
    """--emit-config + --verify: the emitted machine reproduces the selection."""
    payload = _payload(art)
    verify = payload["verify"]
    assert verify["pass"] is True
    assert verify["checks"]
    assert all(check["ok"] for check in verify["checks"])
    names = {check["name"] for check in verify["checks"]}
    assert {"analog_chips", "tiles", "beat", "resident_streams"} <= names
    emitted = yaml.safe_load((art / "selected_config.yaml").read_text())
    assert DSE.DSE_BLOCK not in emitted
    selected = payload["selected"]
    assert emitted["mapping"]["layers_per_chip"] == selected["knobs"]["layers_per_chip"]
    assert emitted["cim"]["chip"]["arrays_per_chip"] == selected["knobs"]["arrays_per_chip"]
    assert emitted["cim"]["cards"]["ctt"]["bank_depth"] == selected["knobs"]["bank_depth"]


def _normalized(payload):
    """Everything except properties of the machine that ran the sweep."""
    out = copy.deepcopy(payload)
    out.pop("total_wall_s", None)
    out.pop("hardware_config", None)
    out.pop("model_config", None)
    out.pop("emitted_config", None)
    out.pop("output_dir", None)
    for candidate in out["candidates"]:
        candidate.pop("wall_s", None)
    if out.get("selected") is not None:
        out["selected"].pop("wall_s", None)
    if out.get("verify"):
        out["verify"].pop("emitted_config", None)
    return out


@pytest.mark.parametrize("name,frontier,shipped,model,art,model_id", CURVES)
def test_the_checked_in_curve_regenerates_identically(
    name, frontier, shipped, model, art, model_id, tmp_path
):
    """Regenerate-and-compare. THESE ARE THE SLOW ONES: the whole ladder is
    re-walked through the real mapped path and the verify round trip reruns
    run_perf. Nothing is sampled to make it faster — a gate that re-walks part
    of a ladder is a gate that lets the rest of it rot."""
    exit_code, payload = DSE.run_sweep(
        str(frontier),
        str(model),
        model_id=model_id,
        output_dir=str(tmp_path / "out"),
        emit_config=str(tmp_path / "selected.yaml"),
        verify=True,
        quiet=True,
    )
    assert exit_code == 0
    fresh = json.loads((tmp_path / "out" / "dse_report.json").read_text())
    assert _normalized(fresh) == _normalized(_payload(art))
    assert yaml.safe_load((tmp_path / "selected.yaml").read_text()) == yaml.safe_load(
        (art / "selected_config.yaml").read_text()
    )
