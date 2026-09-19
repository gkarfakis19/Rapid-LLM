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
  7. **Every artifact of one run says which run made it (D33).** One run id --
     a content hash over the emitter's version, the sources that price a
     candidate and both inputs -- rides the report, the emitted config's header
     and every per-front-point atlas document, so a page built from two
     generations can be caught instead of averaged.
  8. **Every candidate carries what a reader binds a page with.** Where the
     beat goes (per stage: analog, engine, fabric), the per-stage state rows,
     the pipeline stage MAP (`pipeline.stages`, a list; the count is
     `pipeline.stage_count`), a derived-digital block even when the card
     composes nothing, and ONE authoritative binding field.
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
# The two sweeps this round migrated to the D36/D38 axis table (macro_z,
# macro_y, pp_util). The other shipped sweeps still declare the retired axes.
MAMBA2_FRONTIER_HW = HW_DIR / "fws_cim_mamba2_frontier.yaml"
VIT_LARGE_FRONTIER_HW = HW_DIR / "fws_cim_vit_large_frontier.yaml"
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

#: The evaluation layer, imported directly: D34's balance LAW lives there (one
#: accounting), and these tests check both the law's arithmetic and the sweep's
#: reporting of it.
import fws_eval as DSE_EVAL  # noqa: E402


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

    D36/D38 REWRITE of the SPELLING only. The synthetic space used to be
    written on ``layers_per_chip`` x ``column_sets_per_tile``; both are refused
    axes now, so it is written on the two that survive — ``macro_z`` and
    ``pp_util``. The closed form is UNCHANGED, because it never depended on
    what the axes meant: the knob value is mapped through its RUNG INDEX
    (``a`` = 1..4, ``b`` = 1..3, exactly as before) and every ordering,
    count and winner this file asserts is the same walk over the same shape.
    """

    THROUGHPUT = {1: 100.0, 2: 300.0, 3: 380.0, 4: 400.0}

    #: The declared rungs, in declaration order. ``a`` and ``b`` are the
    #: 1-based positions on them, which is all the closed form ever used.
    A_RUNGS = (8, 16, 32, 64)          # macro_z: legal Z values, 4 rungs
    B_RUNGS = (0.5, 0.75, 1.0)         # pp_util: D38's closed list, 3 rungs

    def __init__(self, needs=None):
        self.needs = needs or {}
        self.calls = []

    def __call__(self, point):
        a = self.A_RUNGS.index(int(point["macro_z"])) + 1
        b = self.B_RUNGS.index(float(point["pp_util"])) + 1
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
        "initializer": {"macro_z": 16, "pp_util": 1.0},
        "axes": {
            "macro_z": [8, 16, 32, 64],
            "pp_util": [0.5, 0.75, 1.0],
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
    assert [row["knobs"]["pp_util"] for row in trims[:2]] == [0.75, 0.5]
    by_knobs = {(c["knobs"]["macro_z"], c["knobs"]["pp_util"]): c
                for c in candidates}
    assert by_knobs[(16, 0.5)]["silicon"]["total_silicon_mm2"] == 20.0
    assert by_knobs[(16, 0.5)]["metrics"]["tokens_per_s"] == 300.0


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
    assert [row["knobs"]["macro_z"] for row in climbs] == [32, 64]
    assert all(row["axis"] == "macro_z" for row in climbs)
    descends = [row for row in trail if row["phase"] == "descend" and row.get("accepted")]
    assert [row["knobs"]["macro_z"] for row in descends] == [8]
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
    knobs = {(c["knobs"]["macro_z"], c["knobs"]["pp_util"])
             for c in candidates}
    assert (8, 1.0) in knobs
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
        _fake_spec(repair_axis="pp_util"), space
    )
    refused = [c for c in candidates
               if c["knobs"] == {"macro_z": 64, "pp_util": 0.5}]
    assert refused and refused[0]["fail_stage"] == "mapping"
    repaired = [c for c in candidates
                if c["knobs"] == {"macro_z": 64, "pp_util": 1.0}]
    assert repaired and repaired[0]["ok"]
    assert repaired[0]["ladder"]["repaired_from"] == refused[0]["id"]
    assert "REPAIRED STEP" in "\n".join(repaired[0]["notes"])
    # Without the repair axis the same walk simply stops at a = 2.
    plain, _ = DSE.ladder_candidates(_fake_spec(), FakeSpace(needs={4: 3}))
    assert not [c for c in plain if c["ok"] and c["knobs"]["macro_z"] == 64]


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


def test_the_axis_table_is_three_axes_and_every_retirement_is_refused_by_name():
    """D36 + D38 REWRITE. The sweepable list is now macro_z, macro_y, pp_util.

    OLD CLAIM (ADJ-10 over ADJ-9): stage granularity and chip capacity are the
    axes that matter and only the DERIVED digital widths are refused. NEW
    CLAIM: D38 retired the partition itself — "The partition is INTENT, not an
    axis. The fill fraction is the axis" — so ``layers_per_chip`` and
    ``layers_per_stage`` join the refusals, ``bank_depth`` (and its
    ``cim.allocation`` alias ``column_sets_per_tile``) is PARKED for ground-up
    reformulation, ``conv_placement`` retires because the conv's work is always
    on the shared digital chiplet, and D36 demotes ``shared_chiplets`` to a
    standing linearity check. What is left is the platform box (D35-REVISED)
    and D38's fill fraction.

    A RETIRED AXIS IS REFUSED, NEVER DROPPED. Every one of these was a real
    axis with a real trade until the decision that took it, so a config written
    against the old law must get a sentence naming the law that moved and the
    axis that replaced it — not a silent no-op and not an "unknown axis" the
    reader has to interpret. That is why the refusal register only ever grows.

    What is NOT refused, and must not be: ``rows`` and ``cols`` as a card
    GEOMETRY MODE. SA-GEOM (2026-08-28) made the array shape derivable, but
    only as an opt-in on the card (``cim.fabric.geometry: derived``), never as
    a rung a sweep walks, so the mode name stays out of BOTH tables.
    """
    # THE WHOLE TABLE. Three axes, and the count is the assertion: a fourth
    # would mean something got swept that no decision put back.
    assert set(DSE.AXIS_TARGETS) == {"macro_z", "macro_y", "pp_util"}
    assert DSE.AXIS_TARGETS["macro_z"] == "cim.cards.<analog card>.params.adcs"
    assert DSE.AXIS_TARGETS["macro_y"] == "cim.cards.<analog card>.params.cols"
    assert DSE.AXIS_TARGETS["pp_util"] == "mapping.pp_util"

    # THE RETIREMENTS, each quoting the decision that took it by number.
    retired = {
        "layers_per_chip": "D38",
        "layers_per_stage": "D38",
        "bank_depth": "D38",
        "column_sets_per_tile": "D38",
        "conv_placement": "D38",
        "shared_chiplets": "D36",
    }
    for axis, decision in retired.items():
        reason = DSE.REFUSED_AXES[axis]
        assert decision in reason, axis
        assert "2026-08-29" in reason, axis
    # The partition's refusals NAME the axis that replaced it, so a reader is
    # told what to sweep instead of only what not to sweep.
    for axis in ("layers_per_chip", "layers_per_stage"):
        assert "pp_util" in DSE.REFUSED_AXES[axis], axis
    # bank_depth is PARKED, not dominated, and its alias is parked WITH it —
    # refusing one and leaving the other live would park the knob in name only.
    assert "PARK" in DSE.REFUSED_AXES["bank_depth"].upper()
    assert "bank_depth" in DSE.REFUSED_AXES["column_sets_per_tile"]
    # D36 demoted shared_chiplets to a CHECK, and says the axis returns if a
    # future law puts the cliff back.
    assert "LINEARITY CHECK" in DSE.REFUSED_AXES["shared_chiplets"]
    assert "the axis returns" in DSE.REFUSED_AXES["shared_chiplets"]

    # The register from before D36/D38 is intact underneath: the derived
    # digital widths, the chip sizing, the analog clock and every macro
    # spelling D35-REVISED closed.
    assert set(DSE.REFUSED_AXES) == {
        "vector_lanes", "num_arrays", "softmax_lanes", "arrays_per_chip",
        "analog_clock_mhz",
    } | set(DSE.MACRO_GEOMETRY_AXES) | set(retired)
    assert "adc_capacity_mux" in DSE.REFUSED_AXES
    assert "George 2026-08-27" in DSE.REFUSED_AXES["analog_clock_mhz"]
    assert not set(DSE.REFUSED_AXES) & set(DSE.AXIS_TARGETS)
    assert not hasattr(DSE, "RETIRED_AXES")
    reason = DSE.REFUSED_AXES["vector_lanes"]
    assert "D31/ADJ-9" in reason and "ANALOG m-pass" in reason
    for axis in ("num_arrays", "softmax_lanes"):
        reason = DSE.REFUSED_AXES[axis]
        assert "ADJ-10" in reason and "DERIVED" in reason
        assert "cim.cards.<card>.fabric_" in reason
    assert "rows" in DSE.REFUSED_AXES and "cols" in DSE.REFUSED_AXES
    assert "rows" not in DSE.AXIS_TARGETS and "cols" not in DSE.AXIS_TARGETS
    for spelling in ("rows", "cols"):
        assert "macro_z" in DSE.REFUSED_AXES[spelling]
    assert "geometry" not in DSE.AXIS_TARGETS and "geometry" not in DSE.REFUSED_AXES
    with pytest.raises(DSE.QifDseUsageError, match="unknown axis"):
        DSE.SweepSpec.from_raw(
            {DSE.DSE_BLOCK: {"axes": {"geometry": ["derived"]}}}, "test"
        )
    for axis in DSE.REFUSED_AXES:
        with pytest.raises(DSE.QifDseUsageError, match="REFUSED"):
            DSE.SweepSpec.from_raw(
                {DSE.DSE_BLOCK: {"axes": {axis: [512]}}}, "test"
            )
    # THE MIGRATED SWEEPS declare no refused axis. Only the two configs this
    # round moved to the new table are walked: the rest of the shipped sweeps
    # still declare the retired axes and are REFUSED until they are migrated,
    # which is the visible debt the refusal register exists to create.
    for path in (MAMBA2_FRONTIER_HW, VIT_LARGE_FRONTIER_HW):
        axes = yaml.safe_load(path.read_text())[DSE.DSE_BLOCK]["axes"]
        assert not set(axes) & set(DSE.REFUSED_AXES), path.name
        assert set(axes) == set(DSE.AXIS_TARGETS), path.name


def test_the_ladder_refuses_a_missing_or_off_ladder_initializer():
    """A ladder starts where somebody put it, never where the tool guessed."""
    with pytest.raises(DSE.QifDseUsageError) as missing:
        _fake_spec(initializer=None)
    assert "initializer" in str(missing.value)
    with pytest.raises(DSE.QifDseUsageError) as partial:
        _fake_spec(initializer={"macro_z": 16})
    assert "pp_util" in str(partial.value)
    with pytest.raises(DSE.QifDseUsageError) as off:
        _fake_spec(initializer={"macro_z": 128, "pp_util": 0.5})
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
        # The peak-occupancy class is `busiest_device_class` now. It used to be
        # spelled `binding_device_class`, which made it collide with the
        # BEAT-SETTING field of that name inside derived_digital; the two
        # answer different questions and disagreed on every point of the
        # shipped Granite curve. See test_exactly_one_field_says_what_binds.
        assert candidate["busiest_device_class"] in {
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
    # THE FINER PLAN HOLDS MORE STATE IN TOTAL, and the same amount PER STAGE.
    # Per stage the two are equal by arithmetic: a stage holds
    # layers_per_stage x D streams, and a finer plan divides the layers while
    # multiplying D by the same factor, so the product is the model's layer
    # count either way. What grows is the number of stages holding it. This
    # used to read a per-stage difference that came from the measurement
    # WINDOW, which ADJ-16 removed from the sizing (M10).
    per_stage = {k: c["state_bill"]["max_stage_state_bytes"] for k, c in by_knob.items()}
    totals = {k: c["state_bill"]["total_state_bytes"] for k, c in by_knob.items()}
    assert per_stage[3] == per_stage[6], "a stage holds layers_per_stage x D either way"
    assert totals[3] > totals[6], "the finer stage plan must hold MORE state (D29)"
    bills = totals

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
    # ADJ-15 added `conv_placement`: where the short depthwise conv's WORK runs.
    # Its weights are never tiled onto analog arrays under either value, so what
    # the axis moves is the work - and the models disagree about which home is
    # cheaper (Granite picks shared_digital, Falcon-Mamba picks macro_pool),
    # which is exactly why it is swept and not assumed.
    assert set(block["axes"]) == {
        "layers_per_chip", "bank_depth", "shared_chiplets", "conv_placement"
    }
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
        # None of these machines is bound by the analog floor, and the row NAMES
        # what binds instead of leaving the reader to infer it from a throughput
        # number. ADJ-16 changed the answer on this curve: with the inter-chip
        # handoffs priced (they used to cost 0 s, because the producer resolved
        # to an op already on the destination chip) the BOUNDARY is what the
        # binding stage spends most of its beat on. Pinning "attention_" here is
        # what let the link hide - a test that names the expected bottleneck
        # cannot discover a new one.
        assert derived["binding_is_analog"] is False
        assert derived["binding_block"].startswith("attention_") or (
            derived["binding_block"] == "boundary"
        )
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
    # WHEN this copy was written is not what it was written from (D33). The
    # run_id itself STAYS in the comparison: it is a content hash over the
    # emitter, the sources that price a candidate and both inputs, so a fresh
    # walk of an unchanged tree reproduces it and a changed one does not - which
    # is exactly the mismatch this gate exists to catch.
    if out.get("run"):
        out["run"] = {
            key: value
            for key, value in out["run"].items()
            if key != "generated_at"
        }
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
        # This gate compares REPORT identity; the per-candidate map documents
        # have their own dedicated gate, and exporting one per valid candidate
        # here writes ~GBs into tmp and an hour into the suite for no coverage.
        atlas_points=False,
        quiet=True,
    )
    assert exit_code == 0
    fresh = json.loads((tmp_path / "out" / "dse_report.json").read_text())
    assert _normalized(fresh) == _normalized(_payload(art))
    assert yaml.safe_load((tmp_path / "selected.yaml").read_text()) == yaml.safe_load(
        (art / "selected_config.yaml").read_text()
    )


# ---------------------------------------------------------------------------
# D33 — the emitter side: one run identity, and enough on every candidate to
# bind a page from the point the reader clicked
# ---------------------------------------------------------------------------


VIT_FRONTIER_HW = HW_DIR / "fws_cim_vit_frontier.yaml"
VIT_MODEL = MODEL_DIR / "vit_huge_story_64_inf.yaml"

ATLAS_HTML = PROJECT_ROOT / "docs" / "qif" / "atlas" / "atlas.html"
ATLAS_CORE_START = "// ==ATLAS-CORE-START=="
ATLAS_CORE_END = "// ==ATLAS-CORE-END=="


def _validate_through_the_atlas_loader(document_path, tmp_path):
    """Run ``ATLAS.validate`` — the page's OWN schema check — over a document.

    The same loader that judges the shipped atlas documents judges these, so a
    per-front-point map cannot pass here and be refused by the page. The test
    SKIPS rather than passes when the harness is unavailable (no node, or the
    page has no core block to lift): a check that could not be made is not a
    check that passed (D21).
    """
    import shutil
    import subprocess
    import textwrap

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the atlas loader check needs it")
    if not ATLAS_HTML.exists():
        pytest.skip(f"{ATLAS_HTML} is absent; there is no loader to run")
    html = ATLAS_HTML.read_text(encoding="utf-8")
    if ATLAS_CORE_START not in html or ATLAS_CORE_END not in html:
        pytest.skip(
            "docs/qif/atlas/atlas.html carries no ==ATLAS-CORE== block to lift"
        )
    core = html[html.index(ATLAS_CORE_START) : html.index(ATLAS_CORE_END)]
    script = tmp_path / f"validate_{Path(document_path).stem}.js"
    script.write_text(
        "const fs = require('fs');\n"
        "const doc = JSON.parse(fs.readFileSync(%r, 'utf8'));\n" % str(document_path)
        + "const ATLAS = eval('(function(){' + "
        + json.dumps(core)
        + " + '; return ATLAS; })()');\n"
        + textwrap.dedent(
            """
            const v = ATLAS.validate(doc);
            console.log(JSON.stringify({
              errors: v.errors.map(e => e.rule + ' ' + e.field),
              warnings: v.warnings.map(w => w.rule + ' ' + w.field),
            }));
            """
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [node, str(script)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip())


def _vit_sweep(tmp_path, axes=None):
    """The shipped ViT frontier machine over a ONE-POINT space.

    The ViT card is the one shipped frontier card that names no synthesis
    library, so it is the machine that used to carry NO derived_digital block
    at all. Sweeping one point of it costs about a second; sweeping the shipped
    twenty costs twenty.
    """
    raw = copy.deepcopy(yaml.safe_load(VIT_FRONTIER_HW.read_text()))
    block = raw[DSE.DSE_BLOCK]
    block["search"] = "cross_product"
    block.pop("initializer", None)
    block["axes"] = axes or {"layers_per_chip": [32, 16]}
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "vit_frontier_1pt.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return DSE.run_sweep(
        str(path),
        str(VIT_MODEL),
        mode="VIT",
        model_id="ViT-Huge-story-64",
        output_dir=str(tmp_path / "out"),
        quiet=True,
    )


def test_one_run_id_rides_every_artifact_the_run_writes(tmp_path):
    """D33's identity. ONE id, on the report, the emitted config and every map.

    The atlas has no privileged point any more: a page is built from a report
    and the map of whatever point the reader clicked. If those came out of
    different generations the page averages two machines and says nothing about
    it. Every artifact therefore carries the id of the run that wrote it, and
    the id is a CONTENT HASH — same tool, same sources, same inputs, same id —
    so a regenerated artifact still matches its siblings.
    """
    exit_code, payload = _moe_sweep(
        tmp_path,
        {"label": "run identity", "axes": {"layers_per_chip": [6, 4]}},
        emit_config=str(tmp_path / "selected.yaml"),
    )
    assert exit_code == 0
    run_id = payload["run_id"]
    assert run_id and payload["run"]["run_id"] == run_id
    assert payload["run"]["tool"] == "fws_qif_dse"
    assert payload["run"]["tool_version"] == DSE.TOOL_VERSION
    assert payload["run"]["code_fingerprint"] == DSE.code_fingerprint()
    # An ISO-8601 UTC stamp, and it is OUTSIDE the hash: it says when this copy
    # was written, not what it was written from.
    assert payload["run"]["generated_at"].endswith("Z")
    assert "T" in payload["run"]["generated_at"]

    out = tmp_path / "out"
    on_disk = json.loads((out / "dse_report.json").read_text())
    assert on_disk["run_id"] == run_id

    # The emitted config carries it in its HEADER, where a human opening the
    # YAML sees it. yaml.safe_load drops comments, so this is read as text.
    header = (tmp_path / "selected.yaml").read_text()
    assert f"# run id:      {run_id}" in header
    assert DSE.TOOL_VERSION in header

    # Every per-front-point map carries it too.
    assert payload["atlas_points"]["emitted"]
    for cand_id in payload["atlas_points"]["emitted"]:
        document = json.loads((out / "points" / f"{cand_id}.json").read_text())
        assert document["run_id"] == run_id
        assert document["run"]["code_fingerprint"] == payload["run"]["code_fingerprint"]

    # SAME INPUTS, SAME TREE, SAME ID. A second run of the same sweep is the
    # same generation, so its artifacts are comparable with the first's.
    _again_code, again = _moe_sweep(
        tmp_path / "again",
        {"label": "run identity", "axes": {"layers_per_chip": [6, 4]}},
    )
    assert again["run_id"] == run_id


def test_the_run_id_moves_when_the_inputs_move(tmp_path):
    """An id that never moved would band nothing. A different machine is a
    different generation, and the id says so without anyone comparing 4 MB of
    JSON."""
    _code_a, one = _moe_sweep(
        tmp_path / "a", {"label": "a", "axes": {"layers_per_chip": [6]}}
    )
    _code_b, two = _moe_sweep(
        tmp_path / "b", {"label": "b", "axes": {"layers_per_chip": [4]}}
    )
    assert one["run_id"] != two["run_id"]


def test_every_candidate_says_where_its_beat_goes(tmp_path):
    """`binding_term` per candidate: the evaluator's own term plus the
    per-stage analog / engine / fabric roll-up.

    Without it a reader who clicked a point could see WHAT the point delivers
    and never WHY: the frontier's whole shape is which stage spends its beat on
    what, and that lived only on the selected point's disclosure string.
    """
    exit_code, payload = _moe_sweep(
        tmp_path,
        {"label": "beat terms", "axes": {"layers_per_chip": [6, 4]}},
    )
    assert exit_code == 0
    for candidate in payload["candidates"]:
        assert candidate["ok"], candidate.get("fail_message")
        block = candidate["binding_term"]
        assert block["measured"] is True, block.get("reason")
        assert isinstance(block["stage"], int)
        assert block["span_s"] > 0
        assert block["terms"] and block["terms"][0]["busy_s"] > 0
        assert {"device_class", "block", "busy_s"} <= set(block["terms"][0])
        # THE PER-STAGE ROLL-UP: one row per sized stage, three times on one
        # comparison. Both engines are sized against the stage's OWN analog
        # m-pass time (D31/ADJ-9, ADJ-10), so the three columns are one
        # question and not three.
        rows = block["per_stage"]
        assert rows, block["per_stage_basis"]
        stages = [row["stage"] for row in rows]
        assert stages == sorted(stages)
        for row in rows:
            assert {"stage", "analog_time_s", "engine_time_s", "fabric_time_s"} <= set(row)
            assert row["analog_time_s"] is not None
            assert row["analog_time_s"] >= 0.0
            engine = row["engine_time_s"] or 0.0
            fabric = row["fabric_time_s"] or 0.0
            assert row["digital_time_s"] == pytest.approx(engine + fabric)
            assert row["analog_bound"] == (row["analog_time_s"] >= row["digital_time_s"])
        # The roll-up is over the UNION of the two derivations' stages, so it
        # covers at least as many as either sized on its own — and never more
        # stages than the pipeline has. On this machine the scan engine sizes
        # nothing (no vector op runs) while the fabric sizes every stage, and
        # the rows carry `engine_time_s: null` for exactly that reason rather
        # than a zero.
        engine_sized = candidate["derived_digital"]["stages_sized"] or 0
        fabric_sized = candidate["derived_digital"]["fabric_stages_sized"] or 0
        assert len(rows) >= max(engine_sized, fabric_sized)
        assert len(rows) <= candidate["pipeline"]["stage_count"]
        assert {row["stage"] for row in rows} <= {
            row["stage"] for row in candidate["pipeline"]["stages"]
        }


def test_every_candidate_carries_the_per_stage_state_rows(tmp_path):
    """`state_bill.per_stage` on EVERY candidate, not only in the sweep summary.

    A stage's bill is what decides whether its plan is buildable and how much
    SRAM it buys, so the rows belong on the row that priced them.
    """
    exit_code, payload = _moe_sweep(
        tmp_path,
        {"label": "state rows", "axes": {"layers_per_chip": [6, 4]}},
    )
    assert exit_code == 0
    for candidate in payload["candidates"]:
        bill = candidate["state_bill"]
        # PRESENT EITHER WAY: an unmeasured bill has an empty list and a
        # reason, so no consumer has to tell a missing key from a missing
        # measurement.
        assert "per_stage" in bill
        assert bill["measured"] is True
        rows = bill["per_stage"]
        assert len(rows) == bill["stages"]
        for row in rows:
            assert {"stage", "state_bytes", "recurrent_state_bytes", "kv_bytes",
                    "verdict", "store_built", "store_macros", "chips"} <= set(row)
            assert row["state_bytes"] == pytest.approx(
                row["recurrent_state_bytes"] + row["kv_bytes"]
            )
            # "sized" is a claim a store was BUILT; store_built is that claim as
            # a boolean, so an absent composition law can never read as a
            # zero-sized store.
            assert row["store_built"] is (row["store_provenance"] == "composed-measured")
            assert row["store_built"] is True
            assert row["store_macros"] > 0
            assert row["chips"]
        assert sum(row["store_macros"] for row in rows) == bill["store_macros"]


def test_the_pipeline_map_is_a_list_and_the_count_is_a_count(tmp_path):
    """`pipeline.stages` is the stage MAP; `pipeline.stage_count` is the count.

    `stages` used to be an integer while every other `stages` in the document
    is a list, so a consumer that asked for the stage plan got a number and the
    pipeline map could not be drawn per point at all.
    """
    exit_code, payload = _moe_sweep(
        tmp_path,
        {"label": "stage map", "axes": {"layers_per_chip": [6, 4]}},
    )
    assert exit_code == 0
    for candidate in payload["candidates"]:
        pipeline = candidate["pipeline"]
        assert isinstance(pipeline["stage_count"], int)
        assert pipeline["stage_count"] == candidate["state_bill"]["stages"]
        rows = pipeline["stages"]
        assert isinstance(rows, list)
        assert len(rows) == pipeline["stage_count"]
        assert [row["stage"] for row in rows] == list(range(len(rows)))
        placed = 0
        for row in rows:
            assert {"stage", "chips", "layers", "layer_ids", "tiles", "blocks"} <= set(row)
            assert row["chips"]
            assert row["layers"] == len(row["layer_ids"])
            assert row["blocks"], "a stage with layers puts SOMETHING on analog silicon"
            assert row["tiles"] == sum(row["blocks"].values())
            placed += row["tiles"]
        # The stage plan is a PARTITION of the layers, and the block census is
        # over the PLACED TILES, so both close against the placement. A tile
        # whose owner sits on no stage is COUNTED, not dropped, so the census
        # can be checked instead of trusted.
        every_layer = [layer for row in rows for layer in row["layer_ids"]]
        assert len(every_layer) == len(set(every_layer))
        assert placed + pipeline["tiles_off_stage"] == candidate["placement"]["tiles"]
        assert pipeline["tiles_off_stage"] == 0, (
            "D30 drops embedding and lm_head on the mapped path, so every placed "
            "tile belongs to a backbone layer and every backbone layer is on a stage"
        )


def test_exactly_one_field_says_what_binds(tmp_path):
    """V10 / D21: two derivations shared the name `binding_device_class`.

    Top level took the class with the highest MAX OCCUPANCY (D28); the digital
    block took the class of the largest busy term on the BEAT-SETTING stage
    (ADJ-10). On the shipped Granite curve they read `link` and `shared_digital`
    on the same point. They measure genuinely different things, so they now
    have different names — and the one called `binding` is the one that answers
    "what sets the beat", with the SAME value in both of its homes.
    """
    exit_code, payload = _moe_sweep(
        tmp_path,
        {"label": "binding", "axes": {"layers_per_chip": [6, 4]}},
    )
    assert exit_code == 0
    for candidate in payload["candidates"]:
        binding = candidate["binding_device_class"]
        busiest = candidate["busiest_device_class"]
        assert binding
        assert busiest
        # ONE ACCOUNTING for `binding`: the top-level field and the digital
        # block's are the same number from the same place.
        assert binding == candidate["derived_digital"]["binding_device_class"]
        assert binding == candidate["binding_term"]["terms"][0]["device_class"]
        # The busiest class is D28's, and it is the max-occupancy row.
        peak = max(candidate["utilization"], key=lambda row: row["max_occupancy"])
        assert busiest == peak["device_class"]
        # Each says what it measures, and the two sentences are different.
        assert candidate["binding_device_class_basis"] == DSE.BINDING_CLASS_BASIS
        assert candidate["busiest_device_class_basis"] == DSE.BUSIEST_CLASS_BASIS
        assert DSE.BINDING_CLASS_BASIS != DSE.BUSIEST_CLASS_BASIS
        assert "beat" in DSE.BINDING_CLASS_BASIS
        assert "occupancy" in DSE.BUSIEST_CLASS_BASIS


def test_a_card_that_composes_nothing_still_gets_a_digital_row(tmp_path):
    """The ViT frontier card's condition, on the fast MoE machine.

    A card that names no synthesis library derives nothing, and the block used
    to be OMITTED — so the digital card of the atlas had nothing to fill and a
    reader could not tell an absent law from an unasked question. The block is
    present, `composed` is false, every derived width is None and the absences
    are named. What is still MEASURED is still there.
    """
    def drop_library(raw):
        raw["cim"]["cards"]["sa"].pop("synthesis_library", None)

    exit_code, payload = _moe_sweep(
        tmp_path,
        {"label": "no library", "axes": {"layers_per_chip": [6]}},
        mutate=drop_library,
    )
    assert exit_code == 0
    for candidate in payload["candidates"]:
        digital = candidate["derived_digital"]
        assert digital is not None
        assert digital["composed"] is False
        assert "synthesis_library" in digital["reason"]
        for key in ("vector_lanes", "fabric_num_arrays", "fabric_softmax_lanes",
                    "digital_area_mm2_total", "digital_power_W_total"):
            assert digital[key] is None, key
        assert digital["absent_terms"]
        # Measured, not composed: the beat-setting stage is read off the
        # timeline, so an uncomposed card still says what binds it.
        assert digital["binding_device_class"] == candidate["binding_device_class"]
        # And the silicon accounting names the same gap, in its own words.
        assert candidate["silicon"]["uncovered_terms"]


def test_a_vit_candidate_carries_a_derived_digital_block(tmp_path):
    """The real ViT machine, end to end. It is the shipped frontier card that
    names no synthesis library, and it is a PREFILL-ONLY workload lowered under
    the retired lockstep regime — so three things are absent at once and all
    three have to be named rather than printed as zeros."""
    exit_code, payload = _vit_sweep(tmp_path, axes={"layers_per_chip": [32]})
    assert exit_code == 0
    assert payload["headline_metric"] == "requests_per_s"
    for candidate in payload["candidates"]:
        assert candidate["ok"], candidate.get("fail_message")
        digital = candidate["derived_digital"]
        assert digital is not None, "a ViT candidate used to carry NO digital row"
        # The BLOCK is the contract; whether this particular card names a
        # synthesis library is that card's business and it has changed under
        # this test before. Either way the block is fillable and its absences
        # are named rather than missing.
        if digital["composed"]:
            assert digital["library_technology"]
        else:
            assert digital["absent_terms"]
            assert "synthesis_library" in digital["reason"]
        # A ViT lowers no vector op, so no scan engine is DERIVED whatever the
        # card composes - and the field says None rather than a made-up width.
        assert digital["vector_lanes"] is None
        # NO STAGE PLAN under lockstep, and the map says so instead of being
        # absent or full of zeros.
        assert candidate["pipeline"]["stages"] == []
        assert "no stage plan" in candidate["pipeline"]["stages_basis"].lower()
        # NO BEAT-SETTING TERM either: a prefill-only run lowered under
        # lockstep puts no op in the measurement slice (the last decode step).
        assert candidate["binding_term"]["measured"] is False
        assert "NOT MEASURED" in candidate["binding_term"]["reason"]
        assert candidate["binding_device_class"] is None
        # What IS measured is still measured.
        assert candidate["busiest_device_class"]
        assert candidate["state_bill"]["per_stage"] == []
        # D34's verdict is ABSENT, not "balanced": a lockstep run has no
        # partition to balance, and calling that balanced would be a pass a
        # check never made.
        balance = candidate["pipeline"]["stage_balance"]
        assert balance["measured"] is False
        assert "no stage plan" in balance["reason"].lower()
        assert "verdict" not in balance


def test_every_valid_candidate_has_a_map_that_loads_and_validates(tmp_path):
    """D33: the atlas has NO privileged point, so EVERY point is drawable.

    Not the front, not a sample of the front — every candidate that built a
    machine. The report is the one source of all results and every row in it
    points at its own full inventory, re-priced by nobody: the document is
    written from that candidate's own live evaluation, inside the evaluation.

    Each is judged by the page's OWN loader, so a map cannot pass here and be
    refused by the atlas.
    """
    # A sweep whose front is a STRICT SUBSET of its valid points: the coarser
    # stage plan dominates the finer one at equal macro (same area, more
    # throughput), so c002/c003 are valid, priced, and off the front.
    exit_code, payload = _vit_sweep(
        tmp_path,
        axes={"layers_per_chip": [32, 16], "macro_z": [512, 128]},
    )
    assert exit_code == 0
    out = tmp_path / "out"
    points = payload["atlas_points"]
    assert points["enabled"] is True
    assert points["failed"] == []
    valid = [c for c in payload["candidates"] if c["ok"]]
    assert len(valid) > len(payload["front_ids"]), (
        "this sweep has to contain OFF-FRONT valid points, or it cannot show that "
        "they are drawable too"
    )
    # EVERY valid candidate, not just the front.
    assert set(points["emitted"]) == {c["id"] for c in valid}
    assert points["valid"] == len(valid)
    assert points["candidates"] == len(payload["candidates"])
    off_front = [c for c in valid if c["id"] not in payload["front_ids"]]
    assert off_front, "the point of this test is the off-front points"
    for candidate in valid:
        relative = candidate["atlas_point"]
        assert relative == f"{DSE.ATLAS_POINTS_DIR}/{candidate['id']}.json"
        path = out / relative
        assert path.exists()
        document = json.loads(path.read_text())
        assert document["schema"] == "fws_atlas/1"
        assert document["run_id"] == payload["run_id"]
        # The document names WHICH row of the report it is the map of, and the
        # numbers on it are that point's own.
        assert document["dse_point"]["candidate_id"] == candidate["id"]
        assert document["dse_point"]["knobs"] == candidate["knobs"]
        assert document["dse_point"]["total_silicon_mm2"] == pytest.approx(
            candidate["silicon"]["total_silicon_mm2"]
        )
        # A per-point document is written BEFORE the sweep has a front, so it
        # must not claim membership of one. The report says that, once.
        assert "on_front" not in document["dse_point"]
        assert "front_membership" in document["dse_point"]
        # A PRICED document: P4's duty cycles and metrics rode the export.
        assert document["metrics"]
        report = _validate_through_the_atlas_loader(path, tmp_path)
        assert report["errors"] == [], (candidate["id"], report["errors"])
        # R23 IS EXPECTED AND IT IS TRUE. D35-REVISED puts X at 512 rows, so a
        # chip that used to hold a few hundred macros now holds thousands at a
        # small Z, and the page says out loud that such a chip is past its
        # phase-1 DRAW budget and will not be drawn. That is a statement about
        # the PAGE's rendering, not about the document's correctness — the
        # document still validates with no errors — so it is admitted here BY
        # NAME rather than asserted away or hidden behind a blanket allowance.
        assert {w.split(" ", 1)[0] for w in report["warnings"]} <= {"R23"}, (
            candidate["id"],
            report["warnings"],
        )
    assert "no privileged point" in points["basis"] or "no cap" in points["basis"]


def test_an_infeasible_candidate_carries_a_null_map_and_its_own_refusal(tmp_path):
    """An infeasible point never BUILT a machine, so there is nothing to draw.

    It gets `atlas_point: null` — present, explicitly null — beside the
    `fail_stage` and `fail_message` it already carried, so a reader who clicks
    it is shown the REFUSAL rather than an absence. A missing key would have
    been indistinguishable from an export that quietly failed.
    """
    exit_code, payload = _moe_sweep(
        tmp_path,
        {
            "label": "infeasible",
            "axes": {"layers_per_chip": [6]},
            "max_silicon_mm2": 1e-6,
        },
    )
    # Every point is refused by the area budget, so the sweep selects nothing.
    assert exit_code == 3
    refused = [c for c in payload["candidates"] if not c["ok"]]
    assert refused
    for candidate in payload["candidates"]:
        # PRESENT ON EVERY ROW, whatever happened to it.
        assert "atlas_point" in candidate
    for candidate in refused:
        assert candidate["atlas_point"] is None
        assert candidate["fail_stage"]
        assert candidate["fail_message"]
        assert not (tmp_path / "out" / "points" / f"{candidate['id']}.json").exists()
    assert set(payload["atlas_points"]["skipped_infeasible"]) == {
        c["id"] for c in refused
    }
    assert payload["atlas_points"]["emitted"] == [
        c["id"] for c in payload["candidates"] if c["ok"]
    ]


def test_no_atlas_points_writes_none_and_says_which_absence_it_is(tmp_path):
    """The opt-out stays, and it NAMES itself.

    Under `--no-atlas-points` every candidate carries `atlas_point: null` for a
    reason that has nothing to do with its machine, so the block says which
    absence a null is.
    """
    exit_code, payload = _moe_sweep(
        tmp_path,
        {"label": "opt out", "axes": {"layers_per_chip": [6, 4]}},
        atlas_points=False,
    )
    assert exit_code == 0
    assert not (tmp_path / "out" / "points").exists()
    for candidate in payload["candidates"]:
        assert candidate["atlas_point"] is None
    block = payload["atlas_points"]
    assert block["enabled"] is False
    assert block["emitted"] == []
    assert "--no-atlas-points" in block["basis"]


def test_a_map_is_exported_once_and_never_repriced(tmp_path):
    """The restructure, pinned: the export runs INSIDE the evaluation.

    A document used to cost a full re-map, re-lower, re-price and export, which
    is why the old code capped how many it would draw. It is now written from
    the evaluation that already exists, so the sweep prices each machine ONCE.
    `atlas_export_s` is that cost, recorded beside `wall_s` (which stays the
    time to EVALUATE the candidate, exactly as the report's column says).
    """
    exit_code, payload = _moe_sweep(
        tmp_path,
        {"label": "one pricing pass", "axes": {"layers_per_chip": [6, 4]}},
    )
    assert exit_code == 0
    # The re-run exporter is gone, not merely unused.
    assert not hasattr(DSE, "export_front_points")
    assert not hasattr(DSE, "atlas_point_ids")
    assert not hasattr(DSE, "MAX_ATLAS_POINTS")
    for candidate in payload["candidates"]:
        assert candidate["ok"]
        assert candidate["atlas_point"]
        # Both costs are reported, and they are different quantities.
        assert candidate["atlas_export_s"] >= 0.0
        assert candidate["wall_s"] > 0.0


# ---------------------------------------------------------------------------
# D34 — stage mismatch is a MAPPER FAILURE, and it is loud
# ---------------------------------------------------------------------------


def _lopsided(raw):
    """A deliberately uneven stage plan on the fast MoE machine.

    12 layers as 8 + 2 + 2 instead of 4 + 4 + 4. The mapping schema already
    admits a non-uniform ``layers_per_stage``, so this is a REAL machine the
    mapper can be asked to build — not a synthetic timeline — and it is the
    shape D34 is about: one stage carrying four times its neighbours' work,
    setting the beat while they idle.
    """
    raw["mapping"]["layers_per_stage"] = [8, 2, 2]


def test_a_lopsided_stage_plan_is_a_mismatch_and_says_so_out_loud(tmp_path, capsys):
    """D34. George: "one of the stages is much longer than others. this causes
    pipeline waste and sets the beat... The entire point of the mapper is to
    *try* and eliminate mismatch even if it comes at an area cost."

    So the verdict is not a number filed away in a JSON blob: it is a WARNING
    on the sweep's stdout, printed even under --quiet, because the thing it has
    to survive is a regeneration nobody was watching.
    """
    exit_code, payload = _moe_sweep(
        tmp_path,
        {"label": "lopsided", "axes": {"layers_per_chip": [6]}},
        mutate=_lopsided,
    )
    assert exit_code == 0
    candidate = payload["candidates"][0]
    balance = candidate["pipeline"]["stage_balance"]
    assert balance["measured"] is True
    assert balance["verdict"] == "MISMATCHED"
    # The measure is max over median of the per-beat stage occupancy - the same
    # quantity that sets the beat.
    assert balance["ratio"] == pytest.approx(
        balance["max_stage_busy_s"] / balance["median_stage_busy_s"]
    )
    assert balance["ratio"] > balance["threshold"]
    assert balance["threshold"] == DSE_EVAL.STAGE_BALANCE_MAX_RATIO
    # WHICH stages are long, and WHAT makes them long.
    assert balance["long_stages"]
    worst = balance["long_stages"][0]
    assert worst["stage"] == 0
    assert worst["dominant_block"]
    assert worst["dominant_device_class"]
    assert worst["ratio_to_median"] > balance["threshold"]
    assert worst["sets_the_beat"] is True
    assert worst["stage"] == balance["binding_stage"]
    # THE BOUND: what a balanced partition of this same work could reach. It is
    # the largest of three floors a repartition cannot get under, and the block
    # names which one binds.
    bound = balance["balanced_beat_bound_s"]
    assert bound > 0
    assert bound == pytest.approx(
        max(
            balance["unsplittable_layer_floor_s"],
            balance["distributable_floor_s"],
            balance["link_floor_s"],
        )
    )
    assert balance["balanced_beat_bound_binding"] in {
        "unsplittable_layer", "distributable_work_over_stages", "chip_link"
    }
    # The bound is a BOUND: the machine as built is slower than it.
    assert balance["max_stage_busy_s"] > bound
    assert balance["headroom_ratio"] == pytest.approx(balance["max_stage_busy_s"] / bound)

    # THE WARNING, on stdout, in plain language and with no decision codenames.
    printed = capsys.readouterr().out
    assert "STAGE MISMATCH" in printed, printed
    line = [row for row in printed.splitlines() if "STAGE MISMATCH" in row]
    assert len(line) == 1
    warning = line[0]
    assert candidate["id"] in warning
    assert f"stage {worst['stage']}" in warning
    assert str(worst["dominant_block"]) in warning
    assert str(worst["dominant_device_class"]) in warning
    assert "beat set here" in warning
    assert "balanced bound" in warning
    assert "tok/s vs measured" in warning
    for codename in ("D34", "ADJ-", "D29", "D21"):
        assert codename not in warning, "the warning is read by people, not by the docs"
    # The same line is available to any caller, not only to stdout.
    assert DSE.stage_mismatch_warning(candidate) == warning.split("WARNING: ", 1)[1]


def test_an_even_stage_plan_is_balanced_and_prints_no_warning(tmp_path, capsys):
    """The other half of the contract: a balanced machine says so and is quiet.

    A warning that fired on everything would be worth nothing.
    """
    exit_code, payload = _moe_sweep(
        tmp_path,
        {"label": "even", "axes": {"layers_per_chip": [4, 3]}},
    )
    assert exit_code == 0
    for candidate in payload["candidates"]:
        balance = candidate["pipeline"]["stage_balance"]
        assert balance["measured"] is True
        assert balance["verdict"] == "balanced"
        assert balance["ratio"] <= balance["threshold"]
        assert balance["long_stages"] == []
        assert DSE.stage_mismatch_warning(candidate) is None
    assert "STAGE MISMATCH" not in capsys.readouterr().out


def test_the_mismatch_verdict_reaches_the_markdown_report(tmp_path):
    """(c) a column in the candidates table and (d) the selected point's verdict
    in the header block — a reader of the .md must not have to open the JSON."""
    _code, _payload = _moe_sweep(
        tmp_path,
        {"label": "lopsided md", "axes": {"layers_per_chip": [6]}},
        mutate=_lopsided,
    )
    report = (tmp_path / "out" / "dse_report.md").read_text()
    assert "| stage balance |" in report
    assert "**MISMATCHED" in report
    assert "STAGE BALANCE: MISMATCHED" in report
    assert "balanced partition of the same work could reach" in report
    assert "MAPPER failure" in report

    _code, _payload = _moe_sweep(
        tmp_path / "even",
        {"label": "even md", "axes": {"layers_per_chip": [4]}},
    )
    even = (tmp_path / "even" / "out" / "dse_report.md").read_text()
    assert "stage balance: **balanced**" in even
    # The header block must not accuse a balanced machine, and its table cell
    # must read ok. (The word still appears once, in the column's own footnote
    # explaining what the flag means - that is the legend, not a verdict.)
    assert "STAGE BALANCE: MISMATCHED" not in even
    # `c000` keys a row in BOTH the candidates table and the utilization table,
    # so both are checked rather than one being picked by position.
    rows = [line for line in even.splitlines() if line.startswith("| c000 |")]
    assert len(rows) == 2
    assert all("MISMATCHED" not in row for row in rows)
    assert any("| ok " in row for row in rows)


def test_the_digital_spread_and_the_occupancy_spread_are_named_apart(tmp_path):
    """Two true numbers, two names (D21).

    A hybrid's attention stages run many times the median DIGITAL time while
    their whole-stage OCCUPANCY is a few times it, because every stage also
    walks its analog macros. The verdict is taken on the occupancy — that is
    what sets the beat — and the digital spread rides beside it so a reader
    holding the digital figure can see it is the same machine.
    """
    _code, payload = _moe_sweep(
        tmp_path,
        {"label": "two spreads", "axes": {"layers_per_chip": [6]}},
        mutate=_lopsided,
    )
    balance = payload["candidates"][0]["pipeline"]["stage_balance"]
    assert balance["digital_ratio"] == pytest.approx(
        balance["max_stage_digital_s"] / balance["median_stage_digital_s"]
    )
    # The verdict is the OCCUPANCY one, and the two are not the same number.
    assert balance["verdict"] == (
        "MISMATCHED" if balance["ratio"] > balance["threshold"] else "balanced"
    )
    assert balance["max_stage_digital_s"] <= balance["max_stage_busy_s"]
    assert "sets the beat" in balance["basis"]


def test_the_balanced_bound_is_the_largest_of_three_floors():
    """The bound's arithmetic, checked by hand on rows built here.

    A repartition moves LAYERS between stages. It cannot split a layer, it
    cannot make the total work smaller, and it cannot split a chip hop — so
    those three are the floors, and the bound is whichever is largest.
    """
    rows = [
        DSE_EVAL.OrderedDict(
            (
                ("stage", index),
                ("beats_fired", 1),
                ("compute_s_per_beat", busy),
                ("outgoing_link_s", link),
                ("outgoing_hops", 1 if link else 0),
                ("occupancy_s_per_beat", busy + link),
            )
        )
        for index, (busy, link) in enumerate([(9.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
    ]
    # One layer per stage, so the busiest single layer is 9.0 and no
    # repartition can beat it.
    layer_spans = {0: [(0.0, 9.0)], 1: [(0.0, 1.0)], 2: [(0.0, 2.0)]}
    layer_fires = {0: {0}, 1: {0}, 2: {0}}
    balance = DSE_EVAL._stage_balance(rows, {}, layer_spans, layer_fires, 0)
    assert balance["verdict"] == "MISMATCHED"
    assert balance["max_stage_busy_s"] == 9.0
    assert balance["median_stage_busy_s"] == 2.0
    assert balance["ratio"] == pytest.approx(4.5)
    assert balance["unsplittable_layer_floor_s"] == pytest.approx(9.0)
    assert balance["distributable_floor_s"] == pytest.approx(12.0 / 3)
    assert balance["link_floor_s"] == 0.0
    # The unsplittable layer dominates here, so no repartition helps at all.
    assert balance["balanced_beat_bound_s"] == pytest.approx(9.0)
    assert balance["balanced_beat_bound_binding"] == "unsplittable_layer"

    # SPLIT THAT LAYER INTO FOUR and the distributable floor takes over.
    layer_spans = {
        0: [(0.0, 2.25)], 1: [(0.0, 2.25)], 2: [(0.0, 2.25)], 3: [(0.0, 2.25)],
        4: [(0.0, 1.0)], 5: [(0.0, 2.0)],
    }
    layer_fires = {index: {0} for index in range(6)}
    balance = DSE_EVAL._stage_balance(rows, {}, layer_spans, layer_fires, 0)
    assert balance["balanced_beat_bound_s"] == pytest.approx(4.0)
    assert balance["balanced_beat_bound_binding"] == "distributable_work_over_stages"

    # A CHIP HOP IS ATOMIC: a big link floors the bound whatever the compute does.
    linked = [dict(row) for row in rows]
    linked[1]["outgoing_link_s"] = 6.0
    linked[1]["occupancy_s_per_beat"] = 7.0
    balance = DSE_EVAL._stage_balance(linked, {}, layer_spans, layer_fires, 0)
    assert balance["link_floor_s"] == pytest.approx(6.0)
    assert balance["balanced_beat_bound_s"] == pytest.approx(6.0)
    assert balance["balanced_beat_bound_binding"] == "chip_link"


def test_a_run_with_no_stages_says_so_instead_of_dividing_by_nothing():
    """An unmeasured verdict is named, never a zero ratio (D21)."""
    balance = DSE_EVAL._stage_balance([], {}, {}, {}, None)
    assert balance["measured"] is False
    assert "no partition to balance" in balance["reason"]
