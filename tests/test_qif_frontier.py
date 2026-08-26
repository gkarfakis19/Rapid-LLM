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
        a, b = int(point["layers_per_chip"]), int(point["column_sets_per_tile"])
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
        "initializer": {"layers_per_chip": 2, "column_sets_per_tile": 3},
        "axes": {
            "layers_per_chip": [1, 2, 3, 4],
            "column_sets_per_tile": [1, 2, 3],
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
    assert [row["knobs"]["column_sets_per_tile"] for row in trims[:2]] == [2, 1]
    by_knobs = {(c["knobs"]["layers_per_chip"], c["knobs"]["column_sets_per_tile"]): c
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
    knobs = {(c["knobs"]["layers_per_chip"], c["knobs"]["column_sets_per_tile"])
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
        _fake_spec(repair_axis="column_sets_per_tile"), space
    )
    refused = [c for c in candidates
               if c["knobs"] == {"layers_per_chip": 4, "column_sets_per_tile": 1}]
    assert refused and refused[0]["fail_stage"] == "mapping"
    repaired = [c for c in candidates
                if c["knobs"] == {"layers_per_chip": 4, "column_sets_per_tile": 3}]
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

    ADJ-10 REWRITE of an ADJ-9 rewrite. OLD CLAIM (ADJ-9): ``vector_lanes`` is
    the ONE refused axis — the scan engine is derived and nothing else is.
    NEW CLAIM: ADJ-10 derives every digital engine whose width is a composition
    of MEASURED blocks, so the attention fabric's ``num_arrays`` and the
    softmax pipeline's ``softmax_lanes`` join it. All three are refused BY NAME
    at parse and none has a field to write, so no sweep can carry any of them.
    (The claim before both: the axes stayed SPELLABLE in a RETIRED register,
    and the labelled axis went on producing a checked-in artifact whose whole
    spread came from overriding the derivation.)

    What is NOT refused, and must not be: ``rows`` and ``cols``. They are not
    derived either — they are the geometry the systolic law was validated at,
    and a machine that wants a different one changes the card by hand.
    """
    assert "layers_per_chip" in DSE.AXIS_TARGETS
    assert "layers_per_stage" in DSE.AXIS_TARGETS
    assert "column_sets_per_tile" in DSE.AXIS_TARGETS
    assert "bank_depth" in DSE.AXIS_TARGETS
    # ADJ-13 adds the two SIZING knobs to the refused set: a chip is built to
    # its placement and the chiplets are packed to the beat, so neither is a
    # design point a sweep may carry.
    # ADJ-14 returns shared_chiplets to the swept axes: what stays refused is
    # the sizing that has no design choice behind it.
    assert set(DSE.REFUSED_AXES) == {
        "vector_lanes", "num_arrays", "softmax_lanes", "arrays_per_chip",
    }
    assert "shared_chiplets" in DSE.AXIS_TARGETS
    assert not set(DSE.REFUSED_AXES) & set(DSE.AXIS_TARGETS)
    assert not hasattr(DSE, "RETIRED_AXES")
    reason = DSE.REFUSED_AXES["vector_lanes"]
    assert "D31/ADJ-9" in reason and "ANALOG m-pass" in reason
    for axis in ("num_arrays", "softmax_lanes"):
        reason = DSE.REFUSED_AXES[axis]
        assert "ADJ-10" in reason and "DERIVED" in reason
        # The refusal names the OVERRIDE that stays legal, so a reader is told
        # what to do instead of being told only what not to do.
        assert "cim.cards.<card>.fabric_" in reason
    # rows/cols are neither an axis nor a refusal: they are geometry.
    assert "rows" not in DSE.REFUSED_AXES and "cols" not in DSE.REFUSED_AXES
    assert "rows" not in DSE.AXIS_TARGETS and "cols" not in DSE.AXIS_TARGETS
    for axis in DSE.REFUSED_AXES:
        with pytest.raises(DSE.QifDseUsageError, match="REFUSED"):
            DSE.SweepSpec.from_raw(
                {DSE.DSE_BLOCK: {"axes": {axis: [512]}}}, "test"
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
    assert "column_sets_per_tile" in str(partial.value)
    with pytest.raises(DSE.QifDseUsageError) as off:
        _fake_spec(initializer={"layers_per_chip": 9, "column_sets_per_tile": 1})
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
        # The store is SIZED to the bill, never checked against a cap.
        assert bill["budget_verdict"] == "sized"
        assert bill["budget_bytes"] is None
        assert bill["store_macros"] > 0
        assert bill["store_area_mm2"] > 0
        # D28: every device class the mapping instantiated has a row.
        assert candidate["utilization"]
        assert candidate["binding_device_class"] in {
            row["device_class"] for row in candidate["utilization"]
        }


def test_more_state_buys_more_sram_instead_of_refusing_the_plan(tmp_path):
    """A MACHINE IS NEVER REFUSED FOR NEEDING MEMORY.

    The finer stage plan holds more resident state (every stage holds all D
    streams, and a finer plan raises D). That used to refuse it. Now it SIZES
    the on-chip store to what it holds, out of integer copies of the measured
    SRAM macro, and charges the area — so the finer plan costs more silicon and
    stays buildable.
    """
    _code, measured = _moe_sweep(
        tmp_path / "measure",
        {"label": "measure", "axes": {"layers_per_chip": [6, 3]}},
    )
    by_knob = {c["knobs"]["layers_per_chip"]: c for c in measured["candidates"]}
    coarse, fine = by_knob[6], by_knob[3]
    bills = {k: c["state_bill"]["max_stage_state_bytes"] for k, c in by_knob.items()}
    assert bills[3] > bills[6], "the finer stage plan must hold MORE state (D29)"

    # Both are feasible, and neither carries a residency budget at all.
    assert coarse["ok"] and fine["ok"]
    for candidate in (coarse, fine):
        assert candidate["state_bill"]["budget_verdict"] == "sized"
        assert candidate["state_bill"]["budget_bytes"] is None

    # The store is sized from the bill, and the bigger bill buys more macros.
    assert fine["state_bill"]["store_macros"] > coarse["state_bill"]["store_macros"]
    for candidate in (coarse, fine):
        silicon = candidate["silicon"]
        store = candidate["state_bill"]
        assert silicon["state_sram_macros"] == store["store_macros"]
        assert silicon["state_sram_silicon_mm2"] == pytest.approx(store["store_area_mm2"])
        assert silicon["state_sram_silicon_mm2"] > 0
        # One accounting (D21): the total is its four named terms.
        assert silicon["total_silicon_mm2"] == pytest.approx(
            silicon["analog_macro_silicon_mm2"]
            + silicon["shared_digital_silicon_mm2"]
            + silicon["macro_pool_silicon_mm2"]
            + silicon["state_sram_silicon_mm2"]
        )
        # The store holds at least what the mapping needs, and the rounding is
        # whole macros rather than a margin (D28).
        assert store["store_macros"] * 33280 >= sum(
            row["state_bytes"] for row in store["per_stage"]
        )
    summary = measured["state_bill_summary"]
    assert summary["budget_bytes"] is None
    assert not summary["infeasible_by_state"]
    assert summary["store_area_mm2_max"] >= summary["store_area_mm2_min"] > 0


def test_a_residency_budget_is_refused_by_name(tmp_path):
    """The old per-stage residency cap is gone, and says so.

    Leaving it parseable would let a config quietly reintroduce a memory
    refusal; the area budget is the one feasibility statement now.
    """
    raw = _moe_raw()
    raw[DSE.DSE_BLOCK] = {
        "label": "capped",
        "max_stage_state_bytes": 1.0,
        "axes": {"layers_per_chip": [6]},
    }
    path = tmp_path / "capped.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(DSE.QifDseUsageError) as excinfo:
        DSE.run_sweep(
            str(path),
            str(MOE_MODEL),
            model_id="MoE-small",
            output_dir=str(tmp_path / "out"),
            quiet=True,
        )
    message = str(excinfo.value)
    assert "max_stage_state_bytes is REFUSED BY NAME" in message
    assert "max_silicon_mm2" in message


def test_the_area_budget_is_what_refuses_a_machine(tmp_path):
    """AREA is the feasibility statement, and it names the term that broke it."""
    _code, measured = _moe_sweep(
        tmp_path / "measure", {"label": "measure", "axes": {"layers_per_chip": [6]}}
    )
    total = measured["candidates"][0]["silicon"]["total_silicon_mm2"]
    exit_code, payload = _moe_sweep(
        tmp_path / "budget",
        {
            "label": "budget",
            "max_silicon_mm2": total * 0.5,
            "axes": {"layers_per_chip": [6]},
        },
    )
    refused = payload["candidates"][0]
    assert not refused["ok"]
    assert refused["fail_stage"] == "budget"
    assert "max_silicon_mm2" in refused["fail_message"]
    # It was PRICED before it was refused, so a reader can see what the design
    # that did not fit would have delivered, and the plot can still draw it.
    assert refused["metrics"]["tokens_per_s"] > 0
    assert refused["silicon"]["state_sram_silicon_mm2"] > 0


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
    # ADJ-13: the capacity axis is retired, and with it the repair that raised
    # it. A chip is built to its placement, so a placement can no longer be
    # refused for a capacity that was declared too small.
    assert "repair_axis" not in block
    assert "arrays_per_chip" not in block["axes"]
    # ADJ-14: the CHIPLET COUNT is an axis, because an engine is sized for the
    # stages its chiplet serves and the trade is real.
    assert set(block["axes"]) == {"layers_per_chip", "bank_depth", "shared_chiplets"}
    assert block["initializer"]["shared_chiplets"] > 0
    # The one feasibility budget is an AREA budget, and it is the same
    # appliance-level number for every model because it describes the box the
    # system ships in rather than the workload.
    raw = copy.deepcopy(swept)
    raw.pop(DSE.DSE_BLOCK)
    import config as config_module

    config_module.convert(raw)
    assert "max_stage_state_bytes" not in block, (
        "the per-stage residency cap is retired: state is sized and charged as area"
    )
    assert block["max_silicon_mm2"] > 0, "every sweep declares its area budget"


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
        # ADJ-10: the FABRIC derives at every point of the curve too, and what
        # binds after it is NAMED on every point. A curve that derived the scan
        # engine and left the fabric declared would be a picture of a machine
        # nobody would build, which is exactly what ADJ-9's curve was.
        assert derived["fabric_num_arrays_provenance"] == "derived-count"
        assert derived["fabric_num_arrays"] >= 2
        assert derived["fabric_softmax_lanes"] >= 1
        assert derived["fabric_stages_sized"] >= 1
        assert derived["binding_block"] and derived["binding_busy_s"] > 0
        assert derived["binding_analog_time_s"] > 0
        # Every one of these machines is still bound by the DECLARED array
        # geometry rather than by the analog floor, and the row says so instead
        # of the reader having to infer it from a throughput number.
        assert derived["binding_is_analog"] is False
        assert derived["binding_block"].startswith("attention_")
        assert derived["fabric_saturated_stages"] == derived["fabric_stages_sized"]
        named = [
            item for item in candidate["disclosures"]
            if item["constraint"] == "binding_term"
        ]
        assert len(named) == 1
        assert "THE NEXT REAL LIMIT" in named[0]["reason"]
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
    """A machine is never refused for needing memory, and the curve says so.

    ADJ-11 REWRITE. This used to check that a refusal carried its bytes and its
    stage. There are no residency refusals any more: the store is SIZED to each
    point's own bill and charged as silicon, so what the checked-in curve has to
    show is that nothing was refused for memory and that every point PAID for
    the store it needs.
    """
    payload = _payload(art)
    summary = payload["state_bill_summary"]
    assert summary["budget_bytes"] is None
    assert not summary["infeasible_by_state"]
    assert "SIZED" in summary["budget_basis"]
    assert "max_silicon_mm2" in summary["budget_basis"]
    for row in summary["infeasible_by_memory"]:
        assert row["verdicts"]
    sized = [
        c for c in payload["candidates"]
        if (c.get("state_bill") or {}).get("measured")
    ]
    assert sized
    for candidate in sized:
        store = candidate["state_bill"]
        silicon = candidate["silicon"]
        assert store["budget_verdict"] == "sized"
        assert silicon["state_sram_macros"] == store["store_macros"]
        assert silicon["state_sram_silicon_mm2"] > 0


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
    assert emitted["cim"]["cards"]["ctt"]["bank_depth"] == selected["knobs"]["bank_depth"]
    assert emitted["mapping"]["shared_chiplets"] == selected["knobs"]["shared_chiplets"]
    # ADJ-13: the emitted machine also carries the capacity the sweep DERIVED
    # for it. run_perf does not derive - deriving is what the sweep is for - so
    # without this the emitted config would describe a different machine and
    # the round trip above would not reproduce the selection.
    assert emitted["cim"]["chip"]["arrays_per_chip"] == selected["knobs_derived"]["arrays_per_chip"]


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
