"""QIF P7.8 — the measured synthesis library (D32) and derived engine sizing (D31).

Two decisions land here and they pull in opposite directions, which is why they
belong in one file.

D32 says digital AREA and POWER stop being declared placeholders: they become
compositions of blocks a synthesis run actually measured, checked in under
``configs/hardware-config/digital_components_<tech>.yaml`` with the OPTIMA
synthesis run named in their provenance headers. So the numbers get more real.

D31 says the scan/vector engine stops being a NUMBER SOMEBODY PICKED: its width
is derived from the beat the analog stages set, and reported. So one of the
inputs disappears.

Every composition below has a HAND-COMPUTED point written out in the comment, in
the P2.6 style, so a reader can redo the arithmetic without running anything.
The block values are transcribed here as literals ON PURPOSE: a test that read
the OPTIMA yaml would make this repo's suite depend on another project being
present on a shared mount, which is exactly what checking the library in was
meant to end.

Three refusals guard the seam:
  * a BLOCK the library does not name is refused by name (no zero-fill, ADJ-4);
  * a TECHNOLOGY that is not checked in is refused where the config is parsed;
  * a BEAT too short for the engine's own pipeline fill is refused instead of
    clamped (no lane count fixes a fill, and D28 forbids padding it).
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import yaml

import cim_timing
import config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HW_DIR = PROJECT_ROOT / "configs" / "hardware-config"
MODEL_DIR = PROJECT_ROOT / "configs" / "model-config"
LIB_22NM = HW_DIR / "digital_components_22nm.yaml"
LIB_12NM = HW_DIR / "digital_components_12nm.yaml"

FWS_T1 = HW_DIR / "fws_cim_optima_t1.yaml"
VIT_HUGE_64 = MODEL_DIR / "vit_huge_story_64_inf.yaml"

#: The 22nm blocks this file does arithmetic with, transcribed from
#: configs/hardware-config/digital_components_22nm.yaml, which was itself copied
#: value for value from OPTIMA's synthesis-generated library. area_um2, power_W.
BLOCKS_22NM = {
    "FP_COMP": (75.19, 3.913e-05),
    "FP_ADD": (1417.41, 0.00200231),
    "FP_MULT": (1296.845, 0.00214539),
    "BF16_EXP": (999.544, 0.0011236),
    "BF16_RECIP": (3472.065, 0.00886468),
    "GEMMINI_SYS_ARRAY": (767330.073, 0.92093),
    "TRANSPOSER": (10606.661, 0.0111271),
    "M_REG": (115.163, 0.00012228),
}


@pytest.fixture(scope="module")
def lib():
    return cim_timing.SynthesisLibrary.load("22nm")


def _hw_from_dict(raw):
    raw = copy.deepcopy(raw)
    config.convert(raw)
    return config.HWConfig.from_dict(raw)


def _device(**card):
    """T1 with a shared digital chiplet card built from the given knobs."""
    raw = yaml.safe_load(FWS_T1.read_text())
    entry = {"kind": "digital_chiplet"}
    entry.update(card)
    raw["cim"]["cards"] = {
        "ctt": {"kind": "analog_macro", "device": "ctt"},
        "sa": entry,
    }
    hw = _hw_from_dict(raw)
    model = config.parse_config(str(VIT_HUGE_64), "VIT").model_config
    return cim_timing.CimDeviceModel(hw, model)


# ---------------------------------------------------------------------------
# The library file: what it is, and what it refuses
# ---------------------------------------------------------------------------


def test_both_checked_in_libraries_carry_their_provenance():
    """A measured number without its measurement is an assertion (D21/D32)."""
    for path, tech in ((LIB_22NM, "22nm"), (LIB_12NM, "12nm")):
        raw = yaml.safe_load(path.read_text())
        assert raw["technology"] == tech
        assert "cim_ctt_big_optima" in raw["source"]
        assert f"reports_{tech}" in raw["source_reports"]
        assert raw["generated_by"] == "OPTIMA parse_rtl.py"
        for name, block in raw["blocks"].items():
            assert block["provenance"] == f"measured via synthesis, {tech}", name
            assert "area_um2" in block and "power_W" in block, name
        # The header says the source is provenance and not a dependency, and
        # nothing in the repo may make it one.
        head = " ".join(
            path.read_text().split("technology:")[0].replace("#", " ").lower().split()
        )
        assert "nothing in this repo fetches the source file at runtime" in head


def test_the_22nm_blocks_are_the_measured_values(lib):
    """Transcribed literals against the checked-in file: an edit has to be meant."""
    assert lib.technology == "22nm"
    for name, (area_um2, power_w) in BLOCKS_22NM.items():
        block = lib.block(name)
        assert block.area_um2 == area_um2, name
        assert block.power_w == power_w, name
        assert block.area_mm2 == area_um2 / 1e6, name
    # The systolic block carries its own geometry, which is what lets a
    # rows x cols fabric be COUNTED in blocks rather than assumed.
    gemmini = lib.block("GEMMINI_SYS_ARRAY")
    assert (gemmini.rows, gemmini.cols) == (32, 32)


def test_a_block_the_library_does_not_name_is_refused_by_name(lib):
    """D32/ADJ-4: no default block, no substitute, no zero-fill."""
    with pytest.raises(cim_timing.SynthesisLibraryError) as excinfo:
        lib.block("INT4_MAGIC_UNIT")
    message = str(excinfo.value)
    assert "INT4_MAGIC_UNIT" in message
    assert "22nm" in message
    # The refusal lists what DOES exist, so a caller is not left guessing.
    assert "FP_ADD" in message and "GEMMINI_SYS_ARRAY" in message
    assert "no default block" in message


def test_a_technology_that_is_not_checked_in_is_refused_where_it_is_declared():
    """The typo dies at config parse, not inside a composition."""
    with pytest.raises(ValueError) as excinfo:
        _device(synthesis_library="7nm")
    message = str(excinfo.value)
    assert "7nm" in message and "synthesis_library" in message
    assert "22nm" in message and "12nm" in message  # what IS checked in
    # And the card that names none simply has none — that is not an error.
    assert _device().has_synthesis_library() is False
    with pytest.raises(cim_timing.SynthesisLibraryError):
        _device().synthesis_library()


def test_optimas_overhead_pads_are_recorded_and_not_applied(lib):
    """D28: no safety margins. The pad that was dropped is a number, not a silence."""
    assert cim_timing.OPTIMA_COLLECTION_OVERHEADS["Softmax_Stage2"] == 0.2
    assert cim_timing.OPTIMA_COLLECTION_OVERHEADS["Mamba_S3_SSM_Scan"] == 0.1
    softmax = cim_timing.compose_softmax_engine(lib, 8)
    # The composed area is the bare sum: applying the 0.2 pad would make it 1.2x.
    bare = math.fsum(unit.count * unit.unit_area_mm2 for unit in softmax.units)
    assert softmax.area_mm2 == pytest.approx(bare, rel=1e-15)
    assert any("overhead 0.2" in note for note in softmax.disclosures)


# ---------------------------------------------------------------------------
# The compositions, each with its arithmetic written out
# ---------------------------------------------------------------------------


def test_the_vector_engine_composition_computed_by_hand(lib):
    """HAND CHECK — the scan/vector engine at the width Granite derives (220).

    One LANE holds three measured blocks, because the pricing law lets a lane
    retire a multiply OR an add on any cycle and the work laws hand it one
    ``ops`` total with no schedule:

        FP_MULT   1296.845 um2   0.00214539 W
        FP_ADD    1417.410 um2   0.00200231 W
        M_REG      115.163 um2   0.00012228 W
        ------------------------------------
        per lane  2829.418 um2   0.00426998 W

    At 220 lanes:

        area  = 220 * 2829.418 um2 = 622 471.96 um2 = 0.62247196 mm2
        power = 220 * 0.00426998 W = 0.9393956 W
    """
    per_lane_um2 = 1296.845 + 1417.410 + 115.163
    assert per_lane_um2 == pytest.approx(2829.418, abs=1e-9)
    per_lane_w = 0.00214539 + 0.00200231 + 0.00012228
    assert per_lane_w == pytest.approx(0.00426998, abs=1e-12)

    engine = cim_timing.compose_vector_engine(lib, 220)
    assert engine.area_mm2 == pytest.approx(220 * per_lane_um2 / 1e6, rel=1e-12)
    assert engine.area_mm2 == pytest.approx(0.62247196, rel=1e-9)
    assert engine.power_w == pytest.approx(220 * per_lane_w, rel=1e-12)
    assert engine.power_w == pytest.approx(0.9393956, rel=1e-9)
    assert engine.blocks == {"FP_MULT": 220, "FP_ADD": 220, "M_REG": 220}
    # Every row says where BOTH of its numbers came from (D21).
    for unit in engine.units:
        assert unit.unit_provenance == "measured via synthesis, 22nm"
        assert unit.count_provenance == cim_timing.PROVENANCE_DERIVED_COUNT
    # The stance is stated, not padded: a mix-split engine would be smaller and
    # is deliberately not modelled.
    assert any("mix-split" in note for note in engine.disclosures)


def test_the_softmax_composition_is_optimas_collection_term_for_term(lib):
    """HAND CHECK — width 4, 2 replicas, OPTIMA's create_softmax_collection.

    Per replica at width w: FP_COMP w-1, FP_ADD 2w-1, FP_MULT 2w, BF16_EXP w,
    BF16_RECIP 1. At w = 4 that is 3, 7, 8, 4, 1; times 2 replicas: 6, 14, 16,
    8, 2.

        area_um2 = 6*75.19 + 14*1417.41 + 16*1296.845 + 8*999.544 + 2*3472.065
                 = 451.14 + 19843.74 + 20749.52 + 7996.352 + 6944.13
                 = 55984.882 um2 = 0.055984882 mm2
    """
    engine = cim_timing.compose_softmax_engine(lib, 4, 2)
    assert engine.blocks == {
        "FP_COMP": 6,
        "FP_ADD": 14,
        "FP_MULT": 16,
        "BF16_EXP": 8,
        "BF16_RECIP": 2,
    }
    hand_um2 = (
        6 * 75.19 + 14 * 1417.41 + 16 * 1296.845 + 8 * 999.544 + 2 * 3472.065
    )
    assert hand_um2 == pytest.approx(55984.882, abs=1e-6)
    assert engine.area_mm2 == pytest.approx(hand_um2 / 1e6, rel=1e-12)
    # One reciprocal per row, shared by the whole width — not one per lane.
    assert engine.blocks["BF16_RECIP"] == 2


def test_the_softmax_pipeline_depth_agrees_with_the_card_default(lib):
    """OPTIMA's softmax collection declares pipeline_depth 20; so does our card.

    The two numbers have one origin, and a drift between them would silently
    price a different pipeline from the one whose area is composed.
    """
    raw = yaml.safe_load(FWS_T1.read_text())
    fabric = config.CIMFabricConfig.from_dict(raw["cim"]["fabric"])
    assert fabric.softmax_pipeline_depth == 20
    assert "Its pipeline depth is 20 in that collection" in (
        cim_timing.compose_softmax_engine.__doc__ or ""
    )


def test_the_sa_fabric_counts_whole_measured_blocks(lib):
    """HAND CHECK — a 32x64 fabric, 2 arrays, 1 replica.

        ceil(32/32) * ceil(64/32) = 1 * 2 = 2 blocks per array
        2 blocks * 2 arrays * 1 replica = 4 GEMMINI_SYS_ARRAY
        area  = 4 * 767330.073 um2 + 1 * 10606.661 um2 (one TRANSPOSER)
              = 3069320.292 + 10606.661 = 3079926.953 um2 = 3.079926953 mm2
        power = 4 * 0.92093 + 0.0111271 = 3.6948471 W
    """
    fabric = cim_timing.compose_sa_fabric(lib, 32, 64, 2, 1)
    assert fabric.blocks == {"GEMMINI_SYS_ARRAY": 4, "TRANSPOSER": 1}
    assert fabric.area_mm2 == pytest.approx(3.079926953, rel=1e-12)
    assert fabric.power_w == pytest.approx(3.6948471, rel=1e-12)
    # OPTIMA's unexplained 9 transposers per replica are NOT copied, and the
    # difference is named rather than absorbed.
    assert any("the 9 is unexplained" in note for note in fabric.disclosures)
    assert any("are NOT composed" in note for note in fabric.disclosures)


def test_a_fabric_that_is_not_a_whole_number_of_blocks_pays_for_the_remainder(lib):
    """Blocks are integers exactly as chips are (D21), and the idle part is named.

    OPTIMA multiplies the RATIOS and keeps the fraction (40/32 * 64/32 = 2.5
    arrays — half a synthesised block). A 40x64 fabric here costs
    ceil(40/32) * ceil(64/32) = 2 * 2 = 4 blocks, and the disclosure says the
    remainder rows are idle silicon this composition still pays for.
    """
    fabric = cim_timing.compose_sa_fabric(lib, 40, 64, 1, 1)
    assert fabric.blocks["GEMMINI_SYS_ARRAY"] == 4
    assert any("IDLE SILICON" in note for note in fabric.disclosures)
    # A geometry that IS a multiple carries no such note.
    exact = cim_timing.compose_sa_fabric(lib, 64, 64, 1, 1)
    assert exact.blocks["GEMMINI_SYS_ARRAY"] == 4
    assert not any("IDLE SILICON" in note for note in exact.disclosures)


def test_the_macro_pool_composition_closes_p7_2s_named_register_gap(lib):
    """P7.2 called the accumulator's holding register a named gap. M_REG closes it."""
    device = _device(synthesis_library="22nm", vector_lanes=64)
    sizing = device.digital_pool_sizing()
    pool = device.macro_pool_composition(sizing)
    assert pool.blocks["M_REG"] == sizing.lanes + sizing.conv_lanes
    assert pool.blocks.get("FP_ADD", 0) == sizing.adders
    hand = (
        sizing.adders * BLOCKS_22NM["FP_ADD"][0]
        + (sizing.lanes + sizing.conv_lanes) * BLOCKS_22NM["M_REG"][0]
    ) / 1e6
    assert pool.area_mm2 == pytest.approx(hand, rel=1e-12)
    assert any("named gap" in note for note in pool.disclosures)


def test_the_chiplet_area_is_composed_and_the_declared_placeholder_is_not_added():
    """D32 replaces the placeholder; D21 forbids counting it twice."""
    device = _device(synthesis_library="22nm", vector_lanes=100, area_mm2=999.0)
    parts = device.shared_digital_compositions()
    assert [p.name for p in parts] == [
        "SA attention fabric",
        "softmax lanes",
        "scan/vector engine",
    ]
    assert device.shared_digital_area_mm2() == pytest.approx(
        math.fsum(p.area_mm2 for p in parts), rel=1e-15
    )
    assert device.shared_digital_area_mm2() < 999.0
    assert device.shared_digital_power_w() == pytest.approx(
        math.fsum(p.power_w for p in parts), rel=1e-15
    )
    assert any(
        "not added to it" in note
        for note in device.shared_digital_area_disclosures()
    )
    # A card with no library keeps the declared number and has NO power at all.
    plain = _device(vector_lanes=100, area_mm2=999.0)
    assert plain.shared_digital_area_mm2() == 999.0
    with pytest.raises(cim_timing.SynthesisLibraryError):
        plain.shared_digital_power_w()


def test_the_composed_chiplet_names_what_it_does_not_cover():
    """A lower bound that calls itself a total is the dishonest version (D28)."""
    device = _device(synthesis_library="22nm", vector_lanes=100)
    notes = device.shared_digital_area_disclosures()
    assert any("LOWER BOUND" in note for note in notes)
    assert any("activation SRAM" in note for note in notes)
    assert any("nothing is read from the source project at runtime" in note for note in notes)


def test_a_run_with_no_scan_op_composes_no_scan_engine():
    """D31 derives the width from DEMAND; zero demand derives zero silicon."""
    device = _device(synthesis_library="22nm")   # no lanes declared, none derived
    assert device.resolved_vector_lanes() is None
    assert [p.name for p in device.shared_digital_compositions()] == [
        "SA attention fabric",
        "softmax lanes",
    ]
    assert any(
        "NO scan/vector engine is composed" in note
        for note in device.shared_digital_area_disclosures()
    )


# ---------------------------------------------------------------------------
# D31: the beat sets the width
# ---------------------------------------------------------------------------


def test_derive_vector_lanes_is_the_exact_inverse_of_the_pricing_law():
    """HAND CHECK — one call of 1 000 000 ops against the ANALOG FLOOR (ADJ-9).

    The budget handed in is the stage's own ANALOG m-pass time, 1.6 us — the
    number Granite's stages actually measure — at 0.95 GHz, depth 20:

        budget = floor(1.6e-6 * 0.95e9) = floor(1520.0) = 1520 cycles
        cost(lanes) = 20 + ceil(1e6 / lanes) - 1 = 19 + ceil(1e6 / lanes)

        at 667 lanes: ceil(1e6/667) = 1500 -> 1519 cycles  <= 1520  OK
        at 666 lanes: ceil(1e6/666) = 1502 -> 1521 cycles  >  1520  NO

    So 667 is the answer, and 667 is the SMALLEST answer for THAT budget, which
    is what "no margins" (D28) means written as an assertion.

    REWRITTEN BY ADJ-9. OLD CLAIM: the budget is the 43.98 us analog BEAT and
    the answer is 24 lanes. NEW CLAIM: the budget is the 1.6 us analog M-PASS
    of the stage itself and the answer is 667. Same law, same arithmetic, a
    28x tighter budget — which is the whole content of derive-to-the-floor: a
    smaller budget buys more lanes, and the analog floor is the smallest budget
    the machine can physically justify. The old number is kept below as the
    thing the new criterion replaced, so the two are comparable on one page.
    """
    ops = [1_000_000.0]
    analog_s, clock, depth = 1.6e-6, 0.95e9, 20
    assert math.floor(analog_s * clock) == 1520
    assert cim_timing.vector_cycles_at(ops, 667, depth) == 19 + math.ceil(1e6 / 667)
    assert cim_timing.vector_cycles_at(ops, 667, depth) == 1519
    assert cim_timing.vector_cycles_at(ops, 666, depth) == 1521

    lanes = cim_timing.derive_vector_lanes(ops, analog_s, clock, depth)
    assert lanes == 667
    assert cim_timing.vector_cycles_at(ops, lanes, depth) <= 1520
    assert cim_timing.vector_cycles_at(ops, lanes - 1, depth) > 1520

    # What the RETIRED criterion bought on the same call: the beat is the
    # slowest stage's time, so sizing against it leaves this stage's engine
    # 28x narrower than its own analog m-pass — and therefore its own binding
    # term. That is the machine ADJ-9 refuses.
    assert math.floor(43.98e-6 * clock) == 41781
    assert cim_timing.derive_vector_lanes(ops, 43.98e-6, clock, depth) == 24
    assert cim_timing.vector_cycles_at(ops, 24, depth) / clock > analog_s


def test_the_per_call_pipeline_fill_is_charged_and_not_amortized():
    """Four calls pay four fills; summing the ops first would lose 3 x 19 cycles."""
    calls = [250_000.0] * 4
    depth = 20
    assert cim_timing.vector_cycles_at(calls, 100, depth) == 4 * (
        depth + math.ceil(250_000 / 100) - 1
    )
    one_big = cim_timing.vector_cycles_at([1_000_000.0], 100, depth)
    assert cim_timing.vector_cycles_at(calls, 100, depth) - one_big == 3 * (depth - 1)


def test_a_beat_shorter_than_the_engines_own_fill_is_refused_by_name():
    """No lane count shortens a pipeline fill, so this is not a provisioning bug."""
    with pytest.raises(cim_timing.EngineSizingError) as excinfo:
        # 10 ns at 0.95 GHz is 9 cycles; one call needs at least depth = 20.
        cim_timing.derive_vector_lanes([1000.0], 10e-9, 0.95e9, 20)
    message = str(excinfo.value)
    assert "no vector-engine width holds" in message
    assert "never the FILL" in message
    assert "refuses to clamp" in message
    # A missing budget is refused too: D31 has nothing to derive from.
    with pytest.raises(cim_timing.EngineSizingError):
        cim_timing.derive_vector_lanes([1000.0], 0.0, 0.95e9, 20)


def test_every_stage_is_sized_to_its_own_analog_m_pass(): 
    """ADJ-9's FIXED POINT: digital per-stage time <= that stage's analog time.

    REWRITTEN BY ADJ-9. OLD CLAIM (test_one_card_carries_one_width...): the
    width is the smallest that holds the analog BEAT, and the binding stage is
    the one with the most work. NEW CLAIM: the width is the smallest for which
    EVERY stage fits its OWN analog m-pass, and the binding stage is the one
    whose (work / analog time) ratio is worst — which is not the same stage.

    HAND CHECK on stage 1 below: 1 000 000 ops in one call against a 1.6 us
    analog m-pass at 0.95 GHz, depth 20, is 667 lanes (see
    test_derive_vector_lanes_is_the_exact_inverse_of_the_pricing_law). Stage 0
    carries 8x the work but 16x the analog time, so it asks for fewer.
    """
    device = _device(synthesis_library="22nm")
    demand = [
        cim_timing.EngineDemand(stage=0, ops=(8_000_000.0,), analog_time_s=25.6e-6),
        cim_timing.EngineDemand(stage=1, ops=(1_000_000.0,), analog_time_s=1.6e-6),
    ]
    sizing = device.derive_engine_sizing(43.98e-6, demand)
    # The stage with the MOST work is not the binding one; the tightest ratio is.
    assert sizing.binding_stage == 1
    assert sizing.vector_lanes == 667
    # Both rows are priced at the PROVISIONED width, not at the width each
    # stage asked for, because the machine that gets built has one engine.
    assert {row.lanes for row in sizing.per_stage} == {sizing.vector_lanes}
    heavy, tight = sizing.per_stage
    # THE FIXED POINT: every stage is ANALOG-BOUND, by construction.
    assert sizing.analog_bound_stages == (0, 1)
    assert sizing.unreachable_stages == ()
    for row in sizing.per_stage:
        assert row.target_kind == cim_timing.TARGET_ANALOG_STAGE
        assert row.target_s == row.analog_time_s
        assert row.time_s <= row.analog_time_s
        assert row.analog_bound
    # D28: idle silicon stays visible. The heavy stage runs with far more
    # slack than the stage that paid for the width.
    assert heavy.slack_s > tight.slack_s >= 0
    # The width is the smallest that holds the BINDING stage's analog time,
    # hand-checkable both ways.
    assert cim_timing.vector_cycles_at(
        [1_000_000.0], sizing.vector_lanes, sizing.pipeline_depth
    ) <= tight.budget_cycles
    assert cim_timing.vector_cycles_at(
        [1_000_000.0], sizing.vector_lanes - 1, sizing.pipeline_depth
    ) > tight.budget_cycles
    # And it is COMPOSED, because this card names a library.
    assert sizing.composition is not None
    assert sizing.composition.blocks["FP_MULT"] == sizing.vector_lanes


def test_the_analog_floor_beats_the_retired_beat_criterion_on_the_same_demand():
    """Derive-UP, stated as a comparison rather than as a claim.

    The same stages sized against the analog BEAT (what D31-v1 did: pass no
    analog time and every stage falls back) buy a NARROWER engine whose own
    per-stage time then EXCEEDS the analog m-pass — the digital-bound machine
    ADJ-9 refuses.
    """
    device = _device(synthesis_library="22nm")
    ops = (1_000_000.0,)
    floor = device.derive_engine_sizing(
        43.98e-6, [cim_timing.EngineDemand(stage=0, ops=ops, analog_time_s=1.6e-6)]
    )
    beat = device.derive_engine_sizing(
        43.98e-6, [cim_timing.EngineDemand(stage=0, ops=ops)]
    )
    assert floor.vector_lanes > beat.vector_lanes
    assert floor.vector_lanes == 667 and beat.vector_lanes == 24
    assert floor.per_stage[0].analog_bound
    # The retired criterion's engine is 28x slower than the analog m-pass it
    # shares its stage with, which is exactly how a stage ends up digital-bound.
    assert beat.per_stage[0].time_s > 1.6e-6
    assert beat.per_stage[0].analog_bound is False


@pytest.mark.parametrize(
    "analog_time_s, kind",
    [
        (0.0, cim_timing.TARGET_NO_ANALOG_WORK),
        # 10 ns at 0.95 GHz is 9 cycles; one call needs at least depth = 20,
        # and no lane count shortens a fill.
        (10e-9, cim_timing.TARGET_ANALOG_BELOW_FILL),
    ],
)
def test_an_unreachable_analog_target_falls_back_and_is_named(analog_time_s, kind):
    """ADJ-9: where the analog side cannot bind, the row says WHY (D21, D28).

    Two stages cannot be analog-bound at any width: one with no analog work at
    all, and one whose analog m-pass is shorter than the engine's own pipeline
    fill. Neither is clamped and neither is padded — both fall back to the
    analog BEAT, which is the widest budget this derivation ever uses, and both
    are counted in unreachable_stages and named in the disclosures.
    """
    device = _device(synthesis_library="22nm")
    sizing = device.derive_engine_sizing(
        43.98e-6,
        [
            cim_timing.EngineDemand(
                stage=0, ops=(1_000_000.0,), analog_time_s=analog_time_s
            )
        ],
    )
    row = sizing.per_stage[0]
    assert row.target_kind == kind
    assert row.target_s == 43.98e-6
    assert sizing.unreachable_stages == (0,)
    assert sizing.analog_bound_stages == ()
    assert sizing.vector_lanes == 24          # the beat's answer, not the floor's
    assert any("UNREACHABLE" in note for note in sizing.disclosures)
    assert any(kind in note for note in sizing.disclosures)


def test_the_derived_width_installs_and_prices_the_ops_it_was_sized_for():
    """The derivation and the pricing are the same law, so they cannot drift."""
    device = _device(synthesis_library="22nm")
    demand = [
        cim_timing.EngineDemand(stage=0, ops=(5_000_000.0,), analog_time_s=8e-6)
    ]
    sizing = device.derive_engine_sizing(43.98e-6, demand)
    device.install_derived_engine(sizing)
    assert device.vector_lanes == sizing.vector_lanes
    assert device.vector_lanes_provenance == cim_timing.PROVENANCE_DERIVED_COUNT
    work = cim_timing.ssm_recurrent_scan_work(
        1.0, d_inner=4096, d_state=128, n_groups=1, n_heads=32
    )
    cost = device.price_vector_work(work)
    assert cost.lanes == sizing.vector_lanes
    assert any("is DERIVED (D31-v2, ADJ-9)" in note for note in cost.disclosures)
    # Clearing it restores the refusal: a device never silently keeps a width.
    device.install_derived_engine(None)
    with pytest.raises(cim_timing.EngineCapabilityError):
        _ = device.vector_lanes


def test_an_explicit_vector_lanes_is_an_override_that_rides_a_disclosure():
    """D31: a declared width still works, and it never passes as a derived one."""
    device = _device(synthesis_library="22nm", vector_lanes=4096)
    assert device.vector_lanes == 4096
    assert device.vector_lanes_provenance == cim_timing.PROVENANCE_DECLARED_COUNT
    notes = device.vector_engine_disclosures()
    assert any("an OVERRIDE" in note and "D31 retires" in note for note in notes)
    # With a derivation in hand the disclosure also prints what the beat WOULD
    # have bought, which is the number a reader of an override wants.
    sizing = device.derive_engine_sizing(
        43.98e-6,
        [cim_timing.EngineDemand(stage=0, ops=(1_000_000.0,), analog_time_s=1.6e-6)],
    )
    device.install_derived_engine(sizing)
    assert device.vector_lanes == 4096          # the override still wins
    assert sizing.vector_lanes == 667
    assert any(
        f"would have derived {sizing.vector_lanes} lane" in note
        for note in device.vector_engine_disclosures()
    )
    # ... and the composition labels the COUNT as declared, not derived.
    engine = device.vector_engine_composition()
    assert all(
        unit.count_provenance == cim_timing.PROVENANCE_DECLARED_COUNT
        for unit in engine.units
    )


def test_the_sizing_summary_carries_everything_it_was_derived_from():
    """A derived number with no inputs beside it is not a derivation (D21)."""
    device = _device(synthesis_library="22nm")
    sizing = device.derive_engine_sizing(
        43.98e-6,
        [
            cim_timing.EngineDemand(
                stage=0, ops=(3_000_000.0, 1_000_000.0), analog_time_s=1.6e-6
            ),
            cim_timing.EngineDemand(
                stage=1, ops=(1_000_000.0,), analog_time_s=1.6e-6
            ),
        ],
    )
    summary = sizing.summary()
    assert summary["analog_beat_s"] == 43.98e-6
    assert summary["vector_lanes"] == sizing.vector_lanes
    assert summary["lane_provenance"] == cim_timing.PROVENANCE_DERIVED_COUNT
    assert summary["binding_stage"] == 0
    assert [row["stage"] for row in summary["per_stage"]] == [0, 1]
    assert summary["per_stage"][0]["vector_calls"] == 2
    assert summary["per_stage"][0]["scalar_ops"] == 4_000_000.0
    assert summary["per_stage"][0]["scalar_ops_per_call"] == [3_000_000.0, 1_000_000.0]
    assert summary["composition"]["technology"] == "22nm"
    # ADJ-9's own fields: the target, what kind of target it is, and whether the
    # analog side ended up binding. A width with no target beside it is not a
    # derivation to the floor, it is just a width.
    assert summary["sizing_target"] == "analog_stage_time"
    assert summary["analog_bound_stages"] == [0, 1]
    assert summary["unreachable_stages"] == []
    assert summary["stages_sized"] == 2
    assert 0.0 < summary["engine_duty_at_target"] <= 1.0
    for row in summary["per_stage"]:
        assert row["analog_time_s"] == 1.6e-6
        assert row["target_s"] == 1.6e-6
        assert row["target_kind"] == cim_timing.TARGET_ANALOG_STAGE
        assert row["analog_bound"] is True
    # The criterion is stated where the number is, not only in a docstring.
    assert any("ANALOG-BOUND BY CONSTRUCTION" in note for note in sizing.disclosures)
    report = sizing.report()
    assert "sized to the ANALOG FLOOR" in report
    assert "analog-bound stages     2 of 2" in report
