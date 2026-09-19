"""SA-GEOM: the attention fabric's ARRAY SHAPE derives too — opt-in, decode-only.

ADJ-10 derives the fabric's COUNT and has always said, in its own disclosure,
that the SHAPE stays declared because inventing array geometry is refused
(ADJ-4). The SA geometry study (2026-08-28) removed the "invented" half of that
sentence for the DECODE case: it measured the best 2D geometry at iso-MAC on
six models and found R lands on ``next_pow2(decode M)`` every time, that the
short-R re-shape wins 5.7x (granite) to 20x (llama) in decode attention cycles
over 32 x 64, and that SPLIT QK/PV groups beat MERGED on all 72 decode rows.

WHAT THIS MODULE GATES, IN THE ORDER THE RISK RUNS:

1. **The default cannot move.** ``geometry`` is absent from every shipped card,
   absent parses to ``declared``, and a declared run reproduces the numbers it
   always produced. This is a HARD gate, not a preference: the rule is worth
   nothing if turning it off is not free.
2. **The refusals.** An unknown mode is refused BY NAME with both legal
   spellings. ``derived`` on a LOCKSTEP / prefill-regime machine is refused BY
   NAME with the reason — prefill and ViT INVERT the study's result, 32 x 64 is
   optimal there, and a decode rule applied to a prefill traversal would move a
   measured number onto the shapes it was measured to be wrong for.
3. **The rule itself**, as arithmetic, on the model matrix the study measured.
4. **The machine**, end to end: granite priced twice, declared and derived, and
   the direction of the win checked against the silicon it costs — with the
   three named disclosures riding the artifact.
"""

from __future__ import annotations

import copy
import math
import io
from pathlib import Path

import pytest
import yaml

import cim_timing
import config
import fws_eval
import fws_mapping
from program.fws_build import build_fws_program

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HW_DIR = PROJECT_ROOT / "configs" / "hardware-config"
MODEL_DIR = PROJECT_ROOT / "configs" / "model-config"

#: The SHIPPED granite frontier machine. `tests/test_qif_frontier.py` gates
#: that this file is the shipped card plus the sweep block and nothing else, so
#: stripping `mapping_dse` here yields exactly the shipped machine.
GRANITE_FRONTIER_HW = HW_DIR / "fws_cim_granite_tiny_frontier.yaml"
GRANITE_MODEL = MODEL_DIR / "granite_4_0_h_tiny_inf.yaml"
#: The ViT machine, which declares `mapping.regime: lockstep` by name because a
#: prefill-only model has nothing to stagger into a filled pipeline.
VIT_HW = HW_DIR / "fws_cim_vit_frontier.yaml"
VIT_MODEL = MODEL_DIR / "vit_huge_story_64_inf.yaml"

#: Every shipped fws_cim hardware config — the bit-identity gate's population.
#: Same construction as `tests/test_qif_analog_calc.py`'s.
SHIPPED_FWS_CIM_CONFIGS = sorted(
    path
    for path in HW_DIR.glob("fws_cim*.yaml")
    if isinstance(yaml.safe_load(io.open(path, encoding="utf-8")), dict)
)


def _raw(path, mutate=None):
    raw = copy.deepcopy(yaml.safe_load(io.open(path, encoding="utf-8")))
    raw.pop("mapping_dse", None)
    if mutate is not None:
        mutate(raw)
    return raw


def _hw(path, mutate=None):
    raw = _raw(path, mutate)
    config.convert(raw)
    return config.HWConfig.from_dict(raw)


def _document(hw_path, model_path, mutate=None, mode="LLM"):
    hw = _hw(hw_path, mutate)
    model = config.parse_config(str(model_path), mode)
    config.validate_configs(hw, model)
    mapping = fws_mapping.build_mapping(hw, model)
    return fws_eval.report_document(fws_eval.evaluate_fws(build_fws_program(mapping)))


def _derived(raw):
    raw["cim"]["fabric"]["geometry"] = "derived"


def _declared(raw):
    """Strip the geometry key: ABSENT is the declared path this file gates.

    Shipped cards have since ADOPTED ``geometry: derived`` (granite tiny/1b,
    qwen, the fullstack study), which is what the flag is for. The bit-identity
    gate is about what a card with NO key does, so this fixture removes it
    deliberately rather than relying on a config staying unchanged.
    """
    raw["cim"]["fabric"].pop("geometry", None)


@pytest.fixture(scope="module")
def granite_declared():
    """The shipped granite machine with the geometry key REMOVED -> declared."""
    return _document(GRANITE_FRONTIER_HW, GRANITE_MODEL, _declared)


@pytest.fixture(scope="module")
def granite_derived():
    """The same machine with ONE key added: cim.fabric.geometry: derived."""
    return _document(GRANITE_FRONTIER_HW, GRANITE_MODEL, _derived)


# ---------------------------------------------------------------------------
# 1. THE RULE, AS ARITHMETIC
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "m, rows",
    [
        (1, 1),    # llama:  decode M = 1
        (2, 2),    # gemma:  decode M = 2
        (3, 4),    # granite: decode M = 3
        (4, 4),    # qwen / moe: decode M = 4
        (8, 8),    # minicpm: decode M = 8
    ],
)
def test_R_is_next_pow2_of_the_decode_row_count_on_the_measured_matrix(m, rows):
    """The study's own result, model by model, as the function it became.

    These five rows are the SA geometry study's measured best-R at iso-MAC on
    six models; the rule is that table with the models' names taken off it.
    """
    assert cim_timing.sa_geometry_rows(m) == rows


@pytest.mark.parametrize(
    "m, rows",
    [
        (0, 1),      # no rows is not zero silicon: the shortest array is 1
        (-4, 1),
        (5, 8),
        (17, 32),
        (32, 32),
        (33, 32),    # CLAMPED: 32 is the measured block's own row count
        (4096, 32),
    ],
)
def test_R_is_clamped_to_the_measured_row_range(m, rows):
    """[1, 32]. Above 32 there is no measured block to stand on (ADJ-4)."""
    assert cim_timing.sa_geometry_rows(m) == rows
    assert cim_timing.SA_GEOMETRY_MIN_ROWS <= cim_timing.sa_geometry_rows(m)
    assert cim_timing.sa_geometry_rows(m) <= cim_timing.SA_GEOMETRY_MAX_ROWS


def test_the_column_candidates_are_the_powers_of_two_the_study_measured():
    """The ceiling was 1024 and is 2048 (George 2026-08-28).

    It moved because granite's QK argmin sat ON the old ceiling with its cycles
    still falling — a search that stopped rather than a law that turned. The
    at-ceiling disclosure is kept, because a ceiling that moves once can be hit
    again.
    """
    assert cim_timing.SA_GEOMETRY_COLS_CANDIDATES == (
        32, 64, 128, 256, 512, 1024, 2048,
    )


def test_sa_cycles_at_is_the_one_systolic_law_the_derivation_and_the_run_share():
    """One law, one site (M7): the closed form at an EXPLICIT geometry.

    ``ceil(M/R) * ceil(N/C) * (K + R + C - 2) - 1``, by hand.
    """
    assert cim_timing.sa_cycles_at(3, 512, 128, 4, 1024) == (
        1 * 1 * (128 + 4 + 1024 - 2) - 1
    )
    assert cim_timing.sa_cycles_at(64, 512, 128, 32, 64) == (
        2 * 8 * (128 + 32 + 64 - 2) - 1
    )


# ---------------------------------------------------------------------------
# 2. THE DEFAULT CANNOT MOVE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path", SHIPPED_FWS_CIM_CONFIGS, ids=lambda path: Path(path).stem
)
def test_every_shipped_card_declares_a_legal_geometry_and_absent_means_declared(path):
    """The hard gate: ABSENT means declared, and every declared value is legal.

    This used to assert that NO shipped card opted in, which was true on the
    day the flag landed and is not the property that matters. Cards have since
    adopted ``derived`` — that is what an opt-in flag is for. What must hold
    forever is the other half: a card that says NOTHING gets the shipped law,
    and a card that says something says one of exactly two words.
    """
    raw = _raw(path)
    fabric_raw = raw.get("cim", {}).get("fabric")
    if fabric_raw is None:
        pytest.skip(f"{Path(path).name} declares no cim.fabric block")
    declared_value = fabric_raw.get("geometry")
    if declared_value is not None:
        assert str(declared_value) in cim_timing.FABRIC_GEOMETRY_MODES, path
    hw = config.HWConfig.from_dict(config.convert(copy.deepcopy(raw)))
    expected = str(declared_value or cim_timing.FABRIC_GEOMETRY_DECLARED)
    assert hw.cim_config.fabric.geometry == expected
    assert hw.cim_config.cards.digital_card.fabric.geometry == expected
    # ABSENT IS DECLARED, checked on the card as it would parse with no key.
    stripped = copy.deepcopy(raw)
    stripped["cim"]["fabric"].pop("geometry", None)
    bare = config.HWConfig.from_dict(config.convert(stripped))
    assert bare.cim_config.fabric.geometry == cim_timing.FABRIC_GEOMETRY_DECLARED
    # The fill/drain convention is unchanged too: shipped cards declare no
    # penalty, so `3 * rows` is still what the law charges.
    assert hw.cim_config.fabric.fill_drain_declared is False
    assert hw.cim_config.fabric.fill_drain_penalty_cycles == 3 * int(
        fabric_raw["rows"]
    )


def test_a_declared_granite_reproduces_the_fabric_numbers_it_always_had(
    granite_declared,
):
    """BIT IDENTITY, on the shipped granite frontier machine.

    Same assertion pattern and same numbers as
    ``tests/test_qif_granite_e2e.py::test_the_derived_engine_and_fabric_are_reported``
    reads off ``evaluation.digital_silicon``: the fabric derives to 8 arrays on
    the DECLARED 32 x 64 geometry, its provenance says the COUNT is derived and
    the GEOMETRY is declared, and fold concurrency saturates on the same four
    attention stages. If the opt-in rule leaked into the default path, every
    one of these moves.
    """
    silicon = granite_declared["evaluation"]["digital_silicon"]
    fabric = silicon["derived_fabric_sizing"]
    assert silicon["fabric_num_arrays"] == 8
    assert silicon["fabric_num_arrays_provenance"] == "derived-count"
    assert fabric["num_arrays_declared"] == 2 and fabric["softmax_lanes_declared"] == 1
    assert fabric["array_rows"] == 32 and fabric["array_cols"] == 64
    assert fabric["array_geometry_provenance"] == "declared-count"
    assert fabric["saturated_stages"] == [2, 7, 12, 17]
    assert fabric["analog_bound_stages"] == []
    # The per-stage bill is the DECLARED law's, cycle for cycle: one QK run of
    # 6437 cycles, the 96-cycle (3 x 32) fill/drain, and a 20-cycle softmax.
    for row in fabric["per_stage"]:
        assert row["qk_cycles"] == 6437
        assert row["fill_drain_cycles"] == 96
        assert row["softmax_cycles"] == 20


def test_a_declared_run_writes_no_sa_geometry_key_anywhere(granite_declared):
    """An opt-in rule adds an opt-in KEY, never a null placeholder.

    The shipped atlas artifacts are compared BYTE FOR BYTE against a fresh run
    (``tests/test_qif_granite_e2e.py``), so a new key on the declared path is a
    broken gate even when every number in it is right.
    """
    silicon = granite_declared["evaluation"]["digital_silicon"]
    assert "sa_geometry" not in silicon["derived_fabric_sizing"]
    for key in (
        "fabric_geometry_mode",
        "fabric_array_rows",
        "fabric_array_cols_qk",
        "fabric_array_cols_pv",
        "fabric_array_geometry_provenance",
        "fabric_fill_drain_cycles",
    ):
        assert key not in silicon, key
    constraints = {item["constraint"] for item in granite_declared["disclosures"]}
    assert not {
        "sa_geometry_rule",
        "sa_geometry_dataflow_validity",
        "sa_geometry_bandwidth_assumption",
    } & constraints


# ---------------------------------------------------------------------------
# 3. THE REFUSALS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["derive", "auto", "32x64", "", "DERIVED-ISH", 4])
def test_an_unknown_geometry_mode_is_refused_by_name(value):
    """Strict-keyed on the VALUE: there is no third array shape law."""
    with pytest.raises(ValueError) as excinfo:
        _hw(GRANITE_FRONTIER_HW, lambda raw: raw["cim"]["fabric"].update(
            {"geometry": value}
        ))
    message = str(excinfo.value)
    assert "cim.fabric.geometry" in message
    assert "declared" in message and "derived" in message
    assert repr(value) in message


def test_the_two_legal_spellings_are_case_and_whitespace_tolerant():
    """A mode is a NAME, not a typo trap; the refusal is for real mistakes."""
    hw = _hw(
        GRANITE_FRONTIER_HW,
        lambda raw: raw["cim"]["fabric"].update({"geometry": "  DERIVED "}),
    )
    assert hw.cim_config.fabric.geometry == cim_timing.FABRIC_GEOMETRY_DERIVED


def test_derived_is_refused_by_name_on_a_lockstep_prefill_machine():
    """The rule is a DECODE rule, and prefill/ViT INVERT it.

    The ViT machine declares ``mapping.regime: lockstep`` by name because a
    prefill-only model has nothing to stagger into a filled pipeline. The study
    found 32 x 64 optimal on exactly those shapes, so asking for a short-R
    decode array here is refused with the reason rather than quietly served.
    """
    hw = _hw(VIT_HW, _derived)
    model = config.parse_config(str(VIT_MODEL), "VIT")
    with pytest.raises(ValueError) as excinfo:
        config.validate_configs(hw, model)
    message = str(excinfo.value)
    assert "cim.fabric.geometry" in message
    assert "lockstep" in message
    assert "DECODE rule" in message
    assert "next_pow2" in message
    # The refusal names the way forward, not only the wall.
    assert "filled_pipeline" in message


def test_derived_is_refused_on_a_run_that_declares_no_mapping_block():
    """No mapping block means lockstep AND means the derivation never runs.

    A key that changes nothing is worse than an error, because it reads as if
    it worked.
    """

    def mutate(raw):
        _derived(raw)
        raw.pop("mapping", None)

    hw = _hw(GRANITE_FRONTIER_HW, mutate)
    with pytest.raises(ValueError) as excinfo:
        config.validate_configs(hw, config.parse_config(str(GRANITE_MODEL), "LLM"))
    message = str(excinfo.value)
    assert "cim.fabric.geometry" in message
    assert "mapping:" in message


def test_the_lockstep_refusal_does_not_fire_on_a_declared_vit():
    """The ViT machine as SHIPPED still validates: the refusal is scoped."""
    hw = _hw(VIT_HW)
    config.validate_configs(hw, config.parse_config(str(VIT_MODEL), "VIT"))


# ---------------------------------------------------------------------------
# 4. THE MACHINE
# ---------------------------------------------------------------------------


def test_the_derived_granite_prices_decode_attention_on_an_R4_array(granite_derived):
    """R = next_pow2(3) = 4 on granite, and the artifact says where 3 came from."""
    silicon = granite_derived["evaluation"]["digital_silicon"]
    fabric = silicon["derived_fabric_sizing"]
    geometry = fabric["sa_geometry"]
    assert silicon["fabric_geometry_mode"] == "derived"
    assert silicon["fabric_array_rows"] == 4
    assert geometry["rows"] == 4
    assert geometry["decode_call_m"] == 3          # granite shared_heads = 12 // 4
    assert geometry["rows_declared"] == 32 and geometry["cols_declared"] == 64
    assert fabric["array_geometry_provenance"] == "derived-count"
    # C IS PER GROUP and the two groups DIFFER here, which is the half of the
    # rule a merged geometry cannot express: QK contracts over head_dim and
    # walks the context, PV contracts over the context and walks head_dim.
    assert geometry["cols_qk"] != geometry["cols_pv"]
    assert silicon["fabric_array_cols_qk"] == geometry["cols_qk"]
    assert silicon["fabric_array_cols_pv"] == geometry["cols_pv"]
    for group in ("qk_search_cycles", "pv_search_cycles"):
        searched = {int(cols) for cols, _ in geometry[group]}
        assert searched == set(cim_timing.SA_GEOMETRY_COLS_CANDIDATES)
    # The argmin is the argmin, and ties break to the SMALLEST C (D28).
    for group, chosen in (("qk_search_cycles", "cols_qk"), ("pv_search_cycles", "cols_pv")):
        best = min(geometry[group], key=lambda row: (row[1], row[0]))
        assert int(best[0]) == int(geometry[chosen])
    # The fill/drain convention is unchanged and follows the derived R: 3 x 4,
    # charged ONCE on the QK op, with no PV fill added.
    assert silicon["fabric_fill_drain_cycles"] == 12
    for row in fabric["per_stage"]:
        assert row["fill_drain_cycles"] == 12 * int(row["attention_calls"])


def test_the_stage_attention_time_drops_and_the_silicon_it_costs_is_reported(
    granite_declared, granite_derived
):
    """THE DIRECTION OF THE WIN, and what it was bought with.

    Measured on this machine: every attention stage's priced attention time
    falls ~17x (1.091e-05 s -> 6.41e-07 s), the array count grows from 8 to 192
    because CTX-SPLIT lets the widening continue past fold saturation, and the
    stage lands UNDER its own analog m-pass — the floor named is
    ``analog_pace`` on every stage, which is the one verdict of the four that
    is a success.

    THE ARRAY COUNT GROWING IS THE POINT, not a regression. Before the split
    the fabric saturated at ``2 * folds`` = 8 arrays with the stage still an
    order of magnitude over its analog pace, and more silicon bought NOTHING.
    The assertion is therefore the direction of the TIME and the naming of a
    REAL floor, with the silicon reported rather than bounded.
    """
    dec = granite_declared["evaluation"]["digital_silicon"]
    der = granite_derived["evaluation"]["digital_silicon"]
    dec_rows = {int(r["stage"]): r for r in dec["derived_fabric_sizing"]["per_stage"]}
    der_rows = {int(r["stage"]): r for r in der["derived_fabric_sizing"]["per_stage"]}
    assert set(dec_rows) == set(der_rows)
    for stage, declared_row in dec_rows.items():
        derived_row = der_rows[stage]
        assert derived_row["time_s"] < declared_row["time_s"] / 10.0, stage
        assert derived_row["qk_cycles"] < declared_row["qk_cycles"], stage
        assert derived_row["pv_cycles"] < declared_row["pv_cycles"], stage
    # CTX-SPLIT: the widening went PAST fold saturation, which is exactly what
    # the extension was for. 2 * folds is where the old law stopped.
    folds = 4  # granite kv_heads
    assert dec["fabric_num_arrays"] == 2 * folds
    assert der["fabric_num_arrays"] > 2 * folds
    assert der["fabric_context_split"] > 1
    # AND IT REACHED A REAL FLOOR, named, on every stage.
    floors = der["fabric_floors"]
    assert list(floors) == [cim_timing.FABRIC_FLOOR_ANALOG_PACE], floors
    assert sorted(floors[cim_timing.FABRIC_FLOOR_ANALOG_PACE]) == sorted(der_rows)
    for row in der_rows.values():
        assert row["floor_kind"] == cim_timing.FABRIC_FLOOR_ANALOG_PACE
        assert row["time_s"] <= row["analog_time_s"]
    # The silicon it cost is REPORTED, both directions. The fabric grew; the
    # softmax pipeline shrank because a shorter stage needs fewer copies, and
    # the report carries both rather than a net.
    assert der["derived_fabric_sizing"]["composition"]["area_mm2"] > (
        dec["derived_fabric_sizing"]["composition"]["area_mm2"]
    )
    assert isinstance(der["digital_area_mm2_total"], float)


def test_the_derived_fabric_area_is_linear_in_MAC_count_from_the_measured_block(
    granite_derived,
):
    """D32 with ONE substitution, and it is labelled where it happens.

    A declared geometry is INTEGER copies of the measured 32 x 32 block. A
    derived R x C array is not a tiling of that block at all, so its silicon is
    the block scaled by MAC count with the FRACTION KEPT — and the composition
    says so in its own basis rather than leaving a reader to divide.
    """
    silicon = granite_derived["evaluation"]["digital_silicon"]
    fabric = silicon["derived_fabric_sizing"]
    geometry = fabric["sa_geometry"]
    composition = fabric["composition"]
    assert "derived-linear-from-measured-block" in composition["basis"]
    library = cim_timing.SynthesisLibrary.load("22nm")
    block = library.block("GEMMINI_SYS_ARRAY")
    block_macs = int(block.rows) * int(block.cols)
    row = max(fabric["per_stage"], key=lambda item: item["time_s"])
    replicas = int(fabric["replicas"])
    expected_blocks = (
        int(geometry["rows"]) * int(geometry["cols_qk"]) / block_macs
    ) * int(row["qk_arrays"]) * replicas + (
        int(geometry["rows"]) * int(geometry["cols_pv"]) / block_macs
    ) * int(row["pv_arrays"]) * replicas
    # CTX-SPLIT: the add-reduce is REAL SILICON on this chiplet and joins the
    # fabric composition as INTEGER FP_ADD blocks (a lane is a whole adder,
    # unlike the arrays above which are priced fractionally by MAC count).
    combine_lanes = int(silicon["fabric_combine_lanes"])
    expected_area = (
        expected_blocks * block.area_mm2
        + replicas * library.area_mm2("TRANSPOSER")
        + combine_lanes * replicas * library.area_mm2("FP_ADD")
    )
    assert composition["area_mm2"] == pytest.approx(expected_area, rel=1e-12)
    if combine_lanes > 0:
        adders = next(
            unit for unit in composition["units"] if unit["block"] == "FP_ADD"
        )
        assert adders["count"] == combine_lanes * replicas
        assert float(adders["count"]).is_integer()
    # The TRANSPOSER count is unchanged: ONE per replica, exactly as the
    # declared-geometry composition does it (ADJ-4).
    transposer = next(
        unit for unit in composition["units"] if unit["block"] == "TRANSPOSER"
    )
    assert transposer["count"] == replicas


def test_the_three_named_disclosures_ride_the_derived_artifact(granite_derived):
    """(a) the rule and its basis, (b) the dataflow it is exact under, (c) the
    assumption it rests on — three constraints, because they fail separately.
    """
    disclosures = {item["constraint"]: item for item in granite_derived["disclosures"]}

    rule = disclosures["sa_geometry_rule"]
    assert "next_pow2(decode m)" in rule["reason"]
    assert "C per ENGINE GROUP" in rule["reason"] or "PER ENGINE GROUP" in rule["reason"]
    assert cim_timing.SA_GEOMETRY_STUDY in rule["reason"]
    assert "SA geometry study 2026-08-28" in rule["reason"]
    # The rule states its own scope, because that is what makes it opt-in.
    assert "PREFILL AND ViT INVERT" in rule["reason"]
    geometry = granite_derived["evaluation"]["digital_silicon"][
        "derived_fabric_sizing"
    ]["sa_geometry"]
    assert f"{geometry['rows']} x {geometry['cols_qk']} (QK group)" in rule["value"]
    assert f"{geometry['rows']} x {geometry['cols_pv']} (PV group)" in rule["value"]

    dataflow = disclosures["sa_geometry_dataflow_validity"]
    assert "OUTPUT-STATIONARY" in dataflow["reason"]
    assert "ScaleSim" in dataflow["reason"]
    assert "50 of 50" in dataflow["reason"]
    assert "58x" in dataflow["reason"]           # what ws/is costs, named

    bandwidth = disclosures["sa_geometry_bandwidth_assumption"]
    assert "ASSUMPTION" in bandwidth["reason"]
    assert "NO STALL MODEL IS INVENTED" in bandwidth["reason"]

    # The area substitution is named on the composition it changed, not only in
    # the rule's own paragraph.
    area = disclosures["derived_fabric_area"]
    assert "GEMMINI_SYS_ARRAY" in area["reason"]


def test_the_binding_term_verdict_stops_calling_a_derived_geometry_declared(
    granite_derived,
):
    """A verdict that names the DECLARED geometry as the residue would be a
    false sentence on a run whose geometry was derived. It says the other
    thing, and it says which levers are left.
    """
    named = next(
        item
        for item in granite_derived["disclosures"]
        if item["constraint"] == "binding_term"
    )
    binding = granite_derived["evaluation"]["digital_silicon"]["binding_term"]
    if binding["terms"][0]["block"].startswith("attention"):
        assert "THE NEXT REAL LIMIT IS THE DERIVED ARRAY GEOMETRY" in named["reason"]
        assert "THE RULE ALREADY TOOK THE GEOMETRY LEVER" in named["reason"]
    else:
        # The binding term is not attention on this machine; the geometry
        # verdict is not the one under test then, and the run still names what
        # binds it.
        assert named["reason"]


def test_an_argmin_at_the_search_ceiling_is_named_rather_than_widened_away(
    granite_derived,
):
    """D28: a limit named is a provisioning finding; a limit padded away is not.

    THIS IS A PROPERTY, NOT A GRANITE FACT. Granite's QK argmin used to land on
    the old 1024 ceiling; with the ceiling at 2048 AND the argmin now taken at
    each candidate's own best split rung it lands at 32 instead, because a
    narrow array fed by a deep split beats a wide array that swallows the whole
    context in one pass. So the gate is the implication: IF a group's answer is
    the last candidate, the artifact says so; if it is not, the artifact does
    not claim a ceiling it did not hit.
    """
    geometry = granite_derived["evaluation"]["digital_silicon"][
        "derived_fabric_sizing"
    ]["sa_geometry"]
    ceiling = cim_timing.SA_GEOMETRY_COLS_CANDIDATES[-1]
    at_ceiling = ceiling in (int(geometry["cols_qk"]), int(geometry["cols_pv"]))
    assert ("SEARCH CEILING" in geometry["basis"]) is at_ceiling, geometry["basis"]
    if at_ceiling:
        assert str(ceiling) in geometry["basis"]


def test_the_ceiling_note_fires_when_the_argmin_really_is_at_the_ceiling():
    """The other half of the implication, on a shape that DOES hit it.

    A call whose contraction dwarfs its context makes every column count cost
    the same pipeline term, so the argmin runs to the widest candidate and the
    basis has to say the range stopped it.
    """
    hw = _hw(GRANITE_FRONTIER_HW, _derived)
    model = config.parse_config(str(GRANITE_MODEL), "LLM")
    device = fws_mapping.build_mapping(hw, model).device
    call = cim_timing.AttentionCallDemand(
        m=3, k=8, n=cim_timing.SA_GEOMETRY_COLS_CANDIDATES[-1] * 4,
        folds=1, softmax_tokens=3, heads_chip=1, fill_drain_cycles=96,
    )
    geometry = device.derive_sa_geometry(
        (cim_timing.FabricDemand(stage=0, calls=(call,), analog_time_s=1e-9),)
    )
    if cim_timing.SA_GEOMETRY_COLS_CANDIDATES[-1] in (
        geometry.cols_qk, geometry.cols_pv
    ):
        assert "SEARCH CEILING" in geometry.basis
    else:
        assert "SEARCH CEILING" not in geometry.basis


def test_the_derivation_leaves_a_device_it_did_not_install_onto_untouched():
    """A derivation that is never INSTALLED must not move the device's shape.

    A mapping is reused across programs by callers and fixtures alike, so a
    geometry left behind by a probe would price the next run on the previous
    run's array (D21).
    """
    hw = _hw(GRANITE_FRONTIER_HW, _derived)
    model = config.parse_config(str(GRANITE_MODEL), "LLM")
    mapping = fws_mapping.build_mapping(hw, model)
    device = mapping.device
    assert device.derived_sa_geometry is None
    assert device.sa_geometry("qk") == (32, 64)
    assert device.sa_geometry("pv") == (32, 64)

    call = cim_timing.AttentionCallDemand(
        m=3, k=128, n=512, folds=3, softmax_tokens=3, heads_chip=4,
        fill_drain_cycles=96,
    )
    demand = (cim_timing.FabricDemand(stage=0, calls=(call,), analog_time_s=1e-7),)
    sizing = device.derive_fabric_sizing(1e-6, demand, compose=False)
    assert sizing.geometry is not None and sizing.geometry.rows == 4
    # NOT installed: the device is exactly as it was found.
    assert device.derived_sa_geometry is None
    assert device.sa_geometry("qk") == (32, 64)
    device.install_derived_fabric(sizing)
    assert device.derived_sa_geometry is sizing.geometry
    assert device.sa_geometry("qk") == (4, sizing.geometry.cols_qk)
    assert device.sa_geometry("pv") == (4, sizing.geometry.cols_pv)
    assert device.fabric_fill_drain_cycles == 12
    device.install_derived_fabric(None)
    assert device.derived_sa_geometry is None
    assert device.fabric_fill_drain_cycles == 96


def test_a_declared_card_ignores_the_group_argument_entirely():
    """``group`` is INERT under the shipped law: one card, one array shape.

    This is the bit-identity argument in one assertion — the parameter that
    makes the rule expressible cannot change a declared run's cycles.
    """
    hw = _hw(GRANITE_FRONTIER_HW, _declared)
    model = config.parse_config(str(GRANITE_MODEL), "LLM")
    device = fws_mapping.build_mapping(hw, model).device
    assert device.fabric_geometry_mode == cim_timing.FABRIC_GEOMETRY_DECLARED
    assert device.context_split_enabled is False
    assert device.derive_sa_geometry(()) is None
    for m, n, k in ((3, 512, 128), (64, 64, 4096), (1, 1, 1)):
        expected = cim_timing.sa_cycles_at(m, n, k, 32, 64)
        assert device.sa_cycles(m, n, k) == expected
        assert device.sa_cycles(m, n, k, group="qk") == expected
        assert device.sa_cycles(m, n, k, group="pv") == expected


# ---------------------------------------------------------------------------
# 5. CTX-SPLIT: the widening does not stop at fold saturation
# ---------------------------------------------------------------------------

GRANITE_1B_HW = HW_DIR / "fws_cim_granite_1b_frontier.yaml"
GRANITE_1B_MODEL = MODEL_DIR / "granite_4_0_h_1b_decode_inf.yaml"


def _unsplit(monkeypatch):
    """Run the law as it was BEFORE the context split existed.

    Flipping this one gate reproduces the pre-split behaviour exactly: the
    ladder collapses to one array per group, `sa_context_split` returns
    ``(folds, 1)``, the combine is zero, and `derive_sa_geometry` prices its
    candidates at the operating point it used before. It is the honest control
    for "what did the extension buy", and it is a gate rather than a fork
    precisely so the control is the same code.
    """
    monkeypatch.setattr(
        cim_timing.CimDeviceModel,
        "context_split_enabled",
        property(lambda self: False),
    )


@pytest.fixture(scope="module")
def granite_1b_split():
    return _document(GRANITE_1B_HW, GRANITE_1B_MODEL, mode="LLM")


# --- the split's own arithmetic, before any machine runs -------------------


@pytest.mark.parametrize(
    "folds, arrays, expected",
    [
        (4, 1, (4, 1)),      # one array carries every fold, no split
        (4, 2, (2, 1)),
        (4, 4, (1, 1)),      # fold saturation: one fold each, still no split
        (4, 8, (1, 2)),      # the first split rung
        (4, 12, (1, 2)),     # between rungs the LOWER split is returned
        (4, 16, (1, 4)),
        (4, 256, (1, 64)),
        (1, 8, (1, 8)),
    ],
)
def test_the_split_only_starts_where_fold_concurrency_ends(folds, arrays, expected):
    """Below fold saturation nothing changes; above it the surplus buys context.

    The row ``(4, 4) -> (1, 1)`` is the bit-identity boundary: at exactly fold
    granularity the split factor is still 1, so the expression the law
    evaluates is the one ADJ-10 shipped.
    """
    assert cim_timing.sa_context_split(folds, arrays) == expected


def test_the_ladder_is_fold_granular_then_doubles_and_stops_at_one_column():
    """Rungs only. An array count between two rungs is silicon that buys nothing."""
    assert cim_timing.sa_split_ladder(4, 16) == (1, 2, 3, 4, 8, 16, 32, 64)
    # The ladder stops when a chunk reaches ONE context column.
    assert cim_timing.sa_split_ladder(2, 1) == (1, 2)
    assert cim_timing.sa_split_ladder(1, 8) == (1, 2, 4, 8)


def test_an_unsplit_call_pays_no_combine_at_all():
    """S = 1 zeroes every field. This is the whole bit-identity argument."""
    combine = cim_timing.sa_split_combine(3, 128, 2048, 4, 1, 64)
    assert combine.split == 1
    assert combine.reduce_cycles == 0
    assert combine.adds == 0
    assert combine.gather_bytes == 0.0
    assert combine.cycles == 0


def test_the_combine_is_the_pipelined_tree_expression_this_module_already_uses():
    """``depth + ceil(results / lanes) - 1``, by hand, with depth = ceil(log2 S)."""
    combine = cim_timing.sa_split_combine(3, 128, 2048, 4, 8, 384)
    results = 4 * 3 * 128
    assert combine.results == results
    assert combine.adds == results * (8 - 1)
    assert combine.depth == 3                       # ceil(log2(8))
    assert combine.reduce_cycles == 3 + math.ceil(results / 384) - 1
    # The gather is the whole score row set, reported and priced at zero.
    assert combine.gather_bytes == 4 * 3 * 2048 * 2.0
    assert combine.gather_cycles == 0


def test_splitting_beats_not_splitting_until_the_combine_turns_the_sum():
    """SPLIT PRICING IDENTITY, both directions, on one call.

    A split call costs ``S`` chunks of ``n/S`` PLUS the combine. With enough
    reduce lanes the combine is a handful of cycles and the split WINS; starve
    the lanes and the same split LOSES to not splitting at all. Both directions
    are asserted, because a law that could only ever win would be a law that
    forgot to charge for something.
    """
    rows, cols = 4, 64
    # A SHAPE WHERE THE SPLIT IS MARGINAL: a short context (little to gain from
    # chunking) against a wide output block (a large reduce). This is the
    # regime where the combine can actually turn the sum, which is the half of
    # the law that a generous shape would hide.
    m, k, n, folds = 8, 1024, 64, 1
    split = 8

    def priced(lanes):
        chunk = -(-n // split)
        systolic = cim_timing.sa_cycles_at(m, k, chunk, rows, cols)
        combine = cim_timing.sa_split_combine(m, k, n, folds, split, lanes)
        return systolic + combine.cycles

    unsplit = cim_timing.sa_cycles_at(m, k, n * folds, rows, cols)
    # GENEROUS LANES: the reduce is a few cycles and the split WINS.
    assert priced(folds * m * k) < unsplit
    # STARVED LANES: one adder retires one partial per cycle, so the reduce
    # costs more than the systolic runs saved and the split LOSES.
    assert priced(1) > unsplit
    # And the turn is monotone in the lane count, which is what makes the
    # derivation's binary search over lanes meaningful.
    assert priced(1) > priced(16) > priced(folds * m * k)
    # On a LONG context the split wins even starved, because chunking saves
    # more than one adder can lose. Both regimes are the same expression.
    long_unsplit = cim_timing.sa_cycles_at(3, 128, 2048 * 4, 4, 32)
    long_split = cim_timing.sa_cycles_at(3, 128, 2048 // 32, 4, 32) + (
        cim_timing.sa_split_combine(3, 128, 2048, 4, 32, 1).cycles
    )
    assert long_split < long_unsplit


def test_the_declared_path_has_no_split_no_ladder_and_no_combine():
    """BIT IDENTITY, at the law level: the gate is the declared mode."""
    hw = _hw(GRANITE_FRONTIER_HW, _declared)
    model = config.parse_config(str(GRANITE_MODEL), "LLM")
    device = fws_mapping.build_mapping(hw, model).device
    assert device.context_split_enabled is False
    call = cim_timing.AttentionCallDemand(
        m=3, k=128, n=2048, folds=4, softmax_tokens=3, heads_chip=4,
        fill_drain_cycles=96,
    )
    for arrays in (2, 8, 64, 4096):
        a_qk, a_pv, s_qk, s_pv, _, _, combine = device.attention_split_at(call, arrays)
        assert (s_qk, s_pv) == (1, 1), arrays
        assert combine.cycles == 0
        assert a_qk + a_pv == arrays
    # Past fold saturation the declared law is FLAT — that is the wall the
    # extension exists to get past, asserted here so it stays the wall on a
    # card that did not ask for the extension.
    flat = {
        device._stage_attention_cycles((call,), arrays, 1)[0]
        for arrays in (8, 16, 256, 4096)
    }
    assert len(flat) == 1


# --- the machine: Granite-4.0-H-1B, the model the wall was measured on -----


def test_the_derivation_continues_past_fold_saturation_on_granite_1b(
    granite_1b_split,
):
    """The wall was ``2 * folds`` = 8 arrays. It is not the wall any more."""
    silicon = granite_1b_split["evaluation"]["digital_silicon"]
    fabric = silicon["derived_fabric_sizing"]
    assert silicon["fabric_geometry_mode"] == "derived"
    assert silicon["fabric_num_arrays"] > 8
    assert silicon["fabric_context_split"] > 1
    assert silicon["fabric_combine_lanes"] > 0
    for row in fabric["per_stage"]:
        assert row["num_arrays"] > 8
        assert max(int(row["qk_split"]), int(row["pv_split"])) > 1
        # The combine is charged, not assumed free.
        assert row["combine_cycles"] > 0
        assert row["combine_adds"] > 0
        assert row["gather_bytes"] > 0


def test_granite_1b_attention_drops_materially_and_a_real_floor_is_named(
    granite_1b_split, monkeypatch
):
    """DIRECTION plus a NAMED floor, against the same machine without the split.

    Measured: the attention stage falls from 4.368e-06 s to 5.811e-07 s (~7.5x)
    and lands under its own 6.4e-07 s analog m-pass, so the floor named is
    ``analog_pace``. The beat goes 6.767e-06 s -> 2.786e-06 s and tokens/s
    147786 -> 358962. The assertion is the direction and the naming; the
    numbers are in the report.
    """
    with pytest.MonkeyPatch.context() as patch:
        _unsplit(patch)
        before = _document(GRANITE_1B_HW, GRANITE_1B_MODEL, mode="LLM")

    def stage_times(doc):
        return {
            int(r["stage"]): float(r["time_s"])
            for r in doc["evaluation"]["digital_silicon"]["derived_fabric_sizing"][
                "per_stage"
            ]
        }

    was, now = stage_times(before), stage_times(granite_1b_split)
    assert set(was) == set(now)
    for stage, old in was.items():
        assert now[stage] < old / 2.0, (stage, old, now[stage])
    # A REAL FLOOR IS NAMED, and it is one of the four.
    floors = granite_1b_split["evaluation"]["digital_silicon"]["fabric_floors"]
    assert floors
    for name in floors:
        assert name in cim_timing.FABRIC_FLOORS
    # On this machine the split reaches the analog pace, which is the success
    # verdict — and every stage that names it really is under its analog time.
    for row in granite_1b_split["evaluation"]["digital_silicon"][
        "derived_fabric_sizing"
    ]["per_stage"]:
        assert row["floor_kind"] in cim_timing.FABRIC_FLOORS
        if row["floor_kind"] == cim_timing.FABRIC_FLOOR_ANALOG_PACE:
            assert row["time_s"] <= row["analog_time_s"]

    def beat(doc):
        return next(
            m["value"] for m in doc["evaluation"]["metrics"] if m["key"] == "sys.fws.beat"
        )

    assert beat(granite_1b_split) < beat(before)


def test_the_fabric_stops_being_what_binds_granite_1b(granite_1b_split):
    """What the extension was for: the attention fabric is no longer the beat.

    Before the split the binding term was ``shared_digital/attention_qk``. The
    run still NAMES whatever binds it — that is the ADJ-10 contract and it does
    not change — but on this machine the answer is no longer an engine ADJ-10
    can widen.
    """
    binding = granite_1b_split["evaluation"]["digital_silicon"]["binding_term"]
    assert binding["terms"], binding
    named = next(
        item
        for item in granite_1b_split["disclosures"]
        if item["constraint"] == "binding_term"
    )
    assert named["reason"]


def test_the_split_and_floor_disclosures_ride_the_artifact(granite_1b_split):
    """Two more named disclosures, and they say what they are for."""
    disclosures = {
        item["constraint"]: item for item in granite_1b_split["disclosures"]
    }
    split = disclosures["sa_context_split"]
    assert "SAME sa_cycles law" in split["reason"]
    assert "NO NEW VALIDATION" in split["reason"].upper()
    assert "add-reduced on FP_ADD lanes" in split["reason"]
    assert "D32" in split["reason"]
    assert "intra-chip convention" in split["reason"]

    floor = disclosures["sa_fabric_floor"]
    for name in cim_timing.FABRIC_FLOORS:
        assert name in floor["reason"]
    named = granite_1b_split["evaluation"]["digital_silicon"]["fabric_floors"]
    for name in named:
        assert name in floor["value"]

    # The QK gather is disclosed where it happens, on the priced op's own run.
    gather = disclosures["context_split_gather"]
    assert "concatenated" in gather["value"]
    assert "0 s" in gather["value"]
    assert "D17" in gather["reason"]


def test_the_operand_bandwidth_check_is_an_assumption_when_nothing_declares_one(
    granite_1b_split,
):
    """No invented numbers: an undeclared bandwidth is REPORTED, never bounded.

    The card declares no ``state_bytes_per_cycle``, so no rung is struck out,
    ``operand_bandwidth`` cannot be the named floor, and the derived K/V rate
    is put on the artifact so a reader can compare it against a real store.
    """
    fabric = granite_1b_split["evaluation"]["digital_silicon"][
        "derived_fabric_sizing"
    ]
    for row in fabric["per_stage"]:
        assert row["operand_bytes_per_cycle_declared"] == 0.0
        assert row["operand_bytes_per_cycle"] > 0.0
        assert row["floor_kind"] != cim_timing.FABRIC_FLOOR_OPERAND_BANDWIDTH
    note = " ".join(fabric["disclosures"])
    assert "DECLARES NO state_bytes_per_cycle" in note
    assert "ASSUMPTION AND NOT A BOUND" in note
    assert "Inventing a bandwidth to bound it with would be an invented number" in note


def test_a_declared_operand_bandwidth_strikes_rungs_out_and_names_the_floor():
    """The other side of the check: a card that DOES declare one is bounded by it.

    A deliberately tiny ``state_bytes_per_cycle`` makes every deep rung ask for
    more K/V per cycle than the card can deliver, so those rungs are struck out
    before they are compared and the floor is named ``operand_bandwidth``.
    """

    def mutate(raw):
        _derived(raw)
        raw["cim"]["cards"]["sa"]["state_bytes_per_cycle"] = 1.0

    hw = _hw(GRANITE_1B_HW, mutate)
    model = config.parse_config(str(GRANITE_1B_MODEL), "LLM")
    device = fws_mapping.build_mapping(hw, model).device
    assert float(device.digital_card.state_bytes_per_cycle) == 1.0
    call = cim_timing.AttentionCallDemand(
        m=3, k=128, n=2048, folds=4, softmax_tokens=3, heads_chip=4,
        fill_drain_cycles=96,
    )
    device._derived_geometry = device.derive_sa_geometry(
        (cim_timing.FabricDemand(stage=0, calls=(call,), analog_time_s=6.4e-7),)
    )
    entry = cim_timing.FabricDemand(stage=0, calls=(call,), analog_time_s=6.4e-7)
    found = device._stage_split_search(entry, budget=1, sat_width=1)
    assert found["rungs_struck"] > 0
    assert found["floor_kind"] == cim_timing.FABRIC_FLOOR_OPERAND_BANDWIDTH
    assert found["operand_bytes_per_cycle_declared"] == 1.0


def test_every_named_floor_has_a_reason_and_the_names_are_a_closed_set():
    """A controlled vocabulary a gate reads by name, with prose for each."""
    assert cim_timing.FABRIC_FLOORS == (
        "analog_pace",
        "operand_bandwidth",
        "combine_dominated",
        "saturated_geometry",
    )
    assert set(cim_timing.FABRIC_FLOOR_REASONS) == set(cim_timing.FABRIC_FLOORS)
    for name, reason in cim_timing.FABRIC_FLOOR_REASONS.items():
        assert reason and reason.strip()
    # The one admission of the four says so in its own words, so a reader can
    # tell it from the three that are physics.
    assert "admission" in (
        cim_timing.FABRIC_FLOOR_REASONS[cim_timing.FABRIC_FLOOR_SATURATED_GEOMETRY]
    )
    assert "success" in (
        cim_timing.FABRIC_FLOOR_REASONS[cim_timing.FABRIC_FLOOR_ANALOG_PACE]
    )


def test_the_priced_run_charges_the_same_combine_the_derivation_sized_for():
    """M7: the derivation inverts the pricing law, so the two must agree.

    A fabric sized against a bill the run does not pay is two accountings of
    one metric. The PV op's own cycles carry the reduce, and the timing record
    the derivation reads carries the identical number.
    """
    hw = _hw(GRANITE_1B_HW, _derived)
    model = config.parse_config(str(GRANITE_1B_MODEL), "LLM")
    device = fws_mapping.build_mapping(hw, model).device
    call = cim_timing.AttentionCallDemand(
        m=3, k=128, n=2048, folds=4, softmax_tokens=3, heads_chip=4,
        fill_drain_cycles=96,
    )
    device._derived_geometry = device.derive_sa_geometry(
        (cim_timing.FabricDemand(stage=0, calls=(call,), analog_time_s=6.4e-7),)
    )
    qk, pv, _sm, _a_qk, _a_pv, combine = device.attention_cycles_full(
        call, 256, 1, 1536
    )
    assert combine.split > 1
    assert combine.cycles > 0
    # The stage bill INCLUDES it, and excluding it would understate the stage.
    total = device._stage_attention_cycles((call,), 256, 1, 1536)[0]
    detail = device._stage_attention_detail((call,), 256, 1, 1536)
    assert int(detail["combine_cycles"]) == combine.cycles
    assert total == qk + pv + _sm + combine.cycles + device.call_fill_drain(call)


# ---------------------------------------------------------------------------
# 7. THE PLATFORM LAW LAYER MEETS SA-GEOM
#
# The stub round added a law layer to `cim_timing` — the platform's rows, the
# interim link prices, the analog chip's derived capacity, D36's group function
# and D38's frozen bank depth. SA-GEOM is the most number-sensitive derivation
# in the tree, so it is where the "additive, and additive means inert" claim is
# worth checking rather than asserting.
#
# THE CONTRACT THIS SECTION GATES:
#   * the derived GEOMETRY does not read one of those constants, proved by
#     breaking all of them and re-deriving;
#   * D36's group is read off the SAME layer plan the fabric derivation sees,
#     so a chiplet and an array can never be sized for two different models;
#   * every shipped fws_cim card's macro fits an analog chip, which is the one
#     precondition the capacity law has and the population it has to hold on.
# ---------------------------------------------------------------------------


def _poison_platform_constants(monkeypatch):
    """Set every platform-law constant to a value nothing could survive using."""
    monkeypatch.setattr(cim_timing, "PLATFORM_ROWS", 999_999)
    monkeypatch.setattr(cim_timing, "T_HOP_INTRA_CHIP", 1.0)
    monkeypatch.setattr(cim_timing, "T_HOP_CROSS_CHIP", 2.0)
    monkeypatch.setattr(cim_timing, "ANALOG_CHIP_AREA_MM2", 1.0)
    monkeypatch.setattr(cim_timing, "DIGITAL_CHIPLET_AREA_MIN_MM2", 1.0)
    monkeypatch.setattr(cim_timing, "DIGITAL_CHIPLET_AREA_MAX_MM2", 1.0)
    monkeypatch.setattr(cim_timing, "MAPPED_BANK_DEPTH", 7)


def test_the_derived_geometry_reads_no_platform_law_constant(monkeypatch):
    """SA-GEOM's answer is the study's arithmetic and nothing else.

    Break all seven constants and re-derive the same demand on the same
    machine: the array shape, the split and the cycles must be identical. If a
    future edit routes the geometry rule through the platform layer, this is
    the test that names it.

    THE MAPPINGS ARE BUILT BEFORE THE POISON, DELIBERATELY. `build_mapping`
    DOES read the platform layer on a mapped run — the stub packer's chip
    capacity is `analog_chip_capacity`, which is the point of that law — so
    poisoning first would refuse the machine and prove nothing about the
    geometry. What is gated here is the derivation, which must not read it.
    """
    model = config.parse_config(str(GRANITE_MODEL), "LLM")
    device = fws_mapping.build_mapping(_hw(GRANITE_FRONTIER_HW, _derived), model).device
    declared = fws_mapping.build_mapping(
        _hw(GRANITE_FRONTIER_HW, _declared), model
    ).device
    call = cim_timing.AttentionCallDemand(
        m=3, k=128, n=512, folds=3, softmax_tokens=3, heads_chip=4,
        fill_drain_cycles=96,
    )
    demand = (cim_timing.FabricDemand(stage=0, calls=(call,), analog_time_s=1e-7),)

    before = device.derive_fabric_sizing(1e-6, demand, compose=False)
    _poison_platform_constants(monkeypatch)
    after = device.derive_fabric_sizing(1e-6, demand, compose=False)

    assert after.summary() == before.summary()
    assert (after.geometry.rows, after.geometry.cols_qk, after.geometry.cols_pv) == (
        before.geometry.rows, before.geometry.cols_qk, before.geometry.cols_pv
    )
    # And the declared array is equally inert, on the same broken constants.
    assert declared.sa_cycles(3, 512, 128) == cim_timing.sa_cycles_at(
        3, 512, 128, 32, 64
    )


def test_the_group_is_read_off_the_same_layer_plan_the_fabric_derivation_sees():
    """One model, one plan (D21). Granite's motif is 10 layers, four times over.

    The chiplet (D36) and the array (SA-GEOM) are derived from the same mapped
    device. If the group ever came from somewhere other than the model the
    fabric is sized for, a machine could carry chiplets built for one stack and
    arrays built for another.
    """
    hw = _hw(GRANITE_FRONTIER_HW, _derived)
    model = config.parse_config(str(GRANITE_MODEL), "LLM")
    device = fws_mapping.build_mapping(hw, model).device

    plan = device.layer_group_plan()
    assert plan.motif == cim_timing.minimal_motif(device.model.layer_mixers)
    assert plan.layers == len(device.model.layer_mixers) == device.params.num_layers
    assert plan.period * plan.instances == plan.layers
    # Granite is a hybrid: the motif is NOT one layer, and it is not the stack.
    assert 1 < plan.period < plan.layers
    assert "attention" in {kind for kinds in plan.motif for kind in kinds}
    assert "ssm" in {kind for kinds in plan.motif for kind in kinds}
    # The stage sets are a partition of 0..layers-1, which is what the engine
    # sizer's ADJ-14 argument requires of them.
    flat = [stage for group in plan.stage_groups for stage in group]
    assert flat == list(range(plan.layers))


def test_the_shipped_granite_card_holds_the_frozen_bank_depth():
    """D38 parks bank_depth; the mapped card must be at the frozen 1, not refused."""
    hw = _hw(GRANITE_FRONTIER_HW, _derived)
    model = config.parse_config(str(GRANITE_MODEL), "LLM")
    device = fws_mapping.build_mapping(hw, model).device
    assert device.mapped_bank_depth() == cim_timing.MAPPED_BANK_DEPTH == 1


@pytest.mark.parametrize(
    "path", SHIPPED_FWS_CIM_CONFIGS, ids=lambda p: p.stem
)
def test_every_shipped_cards_macro_fits_an_analog_chip(path):
    """The capacity law's one precondition, over the whole shipped population.

    ``C = floor(400 mm2 / macro footprint)`` is only a slot count if the macro
    fits the chip at all. A shipped card whose macro does not is a machine that
    cannot be built, and it must fail HERE — by name, on the card — rather than
    downstream in the packer as an unplaceable macro.
    """
    raw = _raw(path)
    if raw.get("device_class") != "fws_cim":
        pytest.skip(f"{path.name} is not an fws_cim card")
    hw = _hw(path)
    analog = hw.cim_config.analog
    footprint = float(analog.area_mm2_per_array) / int(
        hw.cim_config.cards.analog_card.stack_3d_height
        if getattr(hw.cim_config, "cards", None) is not None
        else 1
    )
    slots = cim_timing.analog_chip_capacity(footprint, context=path.name)
    assert slots >= 1
    assert slots == math.floor(cim_timing.ANALOG_CHIP_AREA_MM2 / footprint)
    # Floor, not round: the leftover die is real and is never rounded into a slot.
    assert slots * footprint <= cim_timing.ANALOG_CHIP_AREA_MM2
    assert (slots + 1) * footprint > cim_timing.ANALOG_CHIP_AREA_MM2
