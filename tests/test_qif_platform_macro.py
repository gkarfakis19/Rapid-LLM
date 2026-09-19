"""D35-REVISED: the macro is X x Y : Z, mux is DERIVED, and the tiler pays.

Until D35 every fws_cim card set ``rows`` to the model's hidden dim. The
machine changed shape per workload — not a buildable chiplet platform, and it
quietly flattered every comparison, because a macro cut to fit the matrix never
pays a tiling remainder. D35-REVISED (2026-08-29) names the macro properly and
closes the sweepable list around it:

    X x Y : Z — X rows (height, the sheet's constrained dimension, FIXED per
    platform), Y total CTT columns (the storage tile), Z ADCs (the FLOP
    producers; muxed columns are storage). mux = Y/Z, always DERIVED.

WHAT THIS FILE PINS, in the order the change lands:

1. THE PLATFORM lives in ONE module (``cim_analog_calc``): X, the legal Z list,
   the legal Y list, and the ``X x Y : Z`` label. The config layer refuses an
   off-platform macro BY NAME against it, and the single exemption is
   ``cim.analog.nonstandard_macro``, which buys the exemption by STATING A
   REASON that rides a named disclosure into every artifact.
2. THE NAMING TRAP DIES. The retired schema spelled a macro ``cols_adc`` (which
   MEANT the ADC count) plus ``adc_mux``, and never wrote the total column
   count down at all. Both keys — and the retired ``adc_capacity_mux`` fold —
   are refused by name, pointing at what replaced them.
3. THE FOUR subset_v1 cards ARE the platform macro at the sweep's initializer
   point, and their area/energy are the SHEET's at that (X, Y, Z).
4. THE SWEEP walks the BOX: ``macro_z`` and ``macro_y`` are axes; the fold is
   refused as an axis under every spelling, because mux is derived and
   sweeping it would walk a diagonal through the legal region.
5. THE TILING PATH prices rows != hidden end to end. This is what X = 512
   stresses on every model at once: Granite-4.0-H-1B is hidden 1536 on 512-row
   macros, so every hidden-deep matrix is 3 row tiles with a real accumulation
   between them, and any remainder is in the D27 waste metric.
6. THE LAW FIX this exposed. The packer described a K stack as local-or-spread
   and only a WHOLLY local stack got an accumulator. A stack that spans macros
   still lands in RUNS, and those in-macro adds were counted
   (``fws_undescribed_local_k_stacks``) and disclosed but never priced. At
   rows == hidden the shape was rare. At X = 512 it is the common case:
   Mamba2-2.7B has ZERO wholly-local stacks and 2880 in-macro runs, every one
   of which was unpriced before this. The packer now emits one LOCAL SEGMENT
   per run and the DAG prices every one. A wholly local stack is one segment
   equal to itself, so nothing already priced moves.
7. THE LEGACY PATHS do not move: their geometry is untouched, they carry the
   override, and the granite_tiny placement is pinned here so a later edit
   cannot drift it silently.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest
import yaml

import cim_analog_calc
import cim_timing
import config
import fws_eval
import fws_mapping
from cim_timing import PackRequest, TileOwner
from program.fws_build import (
    LAW_ANALOG_GEMM,
    LAW_K_ACCUMULATION,
    annotations_of,
    build_fws_program,
)

REPO = Path(__file__).resolve().parents[1]
HW = REPO / "configs" / "hardware-config"
MODELS = REPO / "configs" / "model-config"

GRANITE_1B_HW = HW / "fws_cim_granite_1b_frontier.yaml"
GRANITE_1B_MODEL = MODELS / "granite_4_0_h_1b_decode_inf.yaml"
MAMBA2_HW = HW / "fws_cim_mamba2_frontier.yaml"
MAMBA2_MODEL = MODELS / "mamba2_2p7b_decode_inf.yaml"
GRANITE_TINY_HW = HW / "fws_cim_granite_tiny.yaml"
GRANITE_TINY_MODEL = MODELS / "granite_4_0_h_tiny_inf.yaml"
MOE_HW = HW / "fws_cim_moe.yaml"
MOE_MODEL = MODELS / "moe_small_fws_inf.yaml"

#: The four subset_v1 hardware cards. All four sit on the SAME platform macro:
#: that is the whole point of a platform.
SUBSET_V1 = (
    "fws_cim_mamba2_frontier.yaml",
    "fws_cim_vit_large_frontier.yaml",
    "fws_cim_vit_frontier.yaml",
    "fws_cim_granite_1b_frontier.yaml",
)

#: The initializer macro every subset card declares: 512 x 2048 : 512, mux 4.
SUBSET_MACRO = (512, 2048, 512)


def _raw(path, mutate=None):
    raw = copy.deepcopy(yaml.safe_load(Path(path).read_text()))
    if mutate is not None:
        mutate(raw)
    return raw


def _hw(path, mutate=None):
    raw = _raw(path, mutate)
    config.convert(raw)
    return config.HWConfig.from_dict(raw)


def _mapping(hw_path, model_path, mode="LLM", mutate=None):
    return fws_mapping.build_mapping(
        _hw(hw_path, mutate), config.parse_config(str(model_path), mode)
    )


# ---------------------------------------------------------------------------
# 1. The platform, in one module
# ---------------------------------------------------------------------------


def test_the_platform_is_one_box_in_one_module():
    calc = cim_analog_calc
    assert calc.PLATFORM_MACRO_ROWS == 512
    # Z: powers of two 8..X — never more converters than rows.
    assert calc.MACRO_Z_VALUES == (8, 16, 32, 64, 128, 256, 512)
    # Y: powers of two X..8X.
    assert calc.MACRO_Y_VALUES == (512, 1024, 2048, 4096)
    assert calc.MACRO_Y_CEILING_MULTIPLE == 8
    assert calc.macro_label(512, 2048, 512) == "512 x 2048 : 512"


def test_mux_is_derived_and_a_remainder_is_refused():
    assert cim_analog_calc.derived_mux(2048, 512) == 4
    assert cim_analog_calc.derived_mux(512, 512) == 1        # mux 1 is legal
    assert cim_analog_calc.derived_mux(4096, 8) == 512
    with pytest.raises(cim_analog_calc.AnalogCalcError, match="must DIVIDE"):
        cim_analog_calc.derived_mux(2048, 300, context="cim.analog")


def test_the_box_is_what_is_platform_macro_means():
    ok = cim_analog_calc.is_platform_macro
    assert ok(512, 2048, 512)
    assert ok(512, 512, 512)          # mux 1, the floor
    assert ok(512, 2048, 8)           # mux 256, the deepest fold the sheet has
    assert not ok(1024, 2048, 512)    # X is fixed
    assert not ok(512, 8192, 512)     # Y past the 8X ceiling
    assert not ok(512, 2048, 1024)    # Z past X
    assert not ok(512, 256, 512)      # Y < Z
    # THE ONE CORNER THE DECLARED BOX HAS AND THE SHEET DOES NOT: Y = 8X
    # against Z = 8 derives mux 512, past OPTIMA's ALLOWED_MUX_DEGREES. The box
    # is a rectangle and the measurements are not, and the difference is named
    # rather than extrapolated.
    assert not ok(512, 4096, 8)


def test_an_off_platform_macro_is_refused_and_names_the_offending_term():
    def widen(raw):
        raw["cim"]["analog"]["rows"] = 1024
        raw["cim"]["cards"]["ctt_sheet"]["params"]["rows"] = 1024

    with pytest.raises(ValueError) as excinfo:
        _hw(GRANITE_1B_HW, widen)
    message = str(excinfo.value)
    assert "D35-REVISED" in message
    assert "X = 1024 rows" in message and "fixes X = 512" in message
    # A refusal a reader cannot act on is a wall, not a law.
    assert "cim.analog.nonstandard_macro" in message


def test_a_y_over_the_ceiling_and_a_z_over_the_rows_are_both_refused():
    for field, value, needle in (
        ("cols", 8192, "past" if False else "not in"),
        ("adcs", 1024, "not in"),
    ):
        def mutate(raw, field=field, value=value):
            raw["cim"]["analog"][field] = value
            raw["cim"]["cards"]["ctt_sheet"]["params"][field] = value

        with pytest.raises(ValueError) as excinfo:
            _hw(GRANITE_1B_HW, mutate)
        assert needle in str(excinfo.value)


def test_a_card_that_departs_alone_is_refused_under_its_own_name():
    def widen_card(raw):
        raw["cim"]["cards"]["ctt_sheet"]["params"]["rows"] = 1024

    with pytest.raises(ValueError, match=r"cim\.cards\.ctt_sheet\.params"):
        _hw(GRANITE_1B_HW, widen_card)


def test_the_override_admits_the_macro_and_carries_the_reason():
    reason = "predates D35; kept for validator/OPTIMA parity"

    def widen(raw):
        raw["cim"]["analog"]["rows"] = 1024
        raw["cim"]["analog"]["nonstandard_macro"] = reason
        raw["cim"]["cards"]["ctt_sheet"]["params"]["rows"] = 1024

    hw = _hw(GRANITE_1B_HW, widen)
    assert hw.cim_config.analog.nonstandard_macro == reason
    # The reason reaches the CARD, which is what the artifact reads it from.
    assert hw.cim_config.analog_card.nonstandard_macro == reason


def test_the_override_is_a_reason_and_never_a_flag():
    for value in (True, "", "   ", 1):
        def mutate(raw, value=value):
            raw["cim"]["analog"]["nonstandard_macro"] = value

        with pytest.raises(ValueError, match="REASON"):
            _hw(GRANITE_1B_HW, mutate)


def test_the_override_on_a_platform_macro_is_refused_as_a_false_disclosure():
    def mutate(raw):
        raw["cim"]["analog"]["nonstandard_macro"] = "no reason at all"

    with pytest.raises(ValueError) as excinfo:
        _hw(GRANITE_1B_HW, mutate)
    assert "IS inside the platform box" in str(excinfo.value)


def test_the_override_rides_a_named_disclosure_into_the_artifact():
    # granite_tiny is a legacy card: 1536 x 1536 : 384, admitted by the override.
    mapping = _mapping(GRANITE_TINY_HW, GRANITE_TINY_MODEL)
    disclosures = fws_eval.nonstandard_macro_disclosures(mapping)
    assert len(disclosures) == 1
    item = disclosures[0]
    assert item.constraint == "nonstandard_macro"
    assert "1536 x 1536 : 384" in item.value
    assert "predates D35" in item.reason
    assert "NOT comparable" in item.reason


def test_a_platform_card_discloses_nothing_because_there_is_nothing_to_say():
    mapping = _mapping(GRANITE_1B_HW, GRANITE_1B_MODEL)
    assert fws_eval.nonstandard_macro_disclosures(mapping) == []


# ---------------------------------------------------------------------------
# 2. The naming trap dies: the retired spellings are refused BY NAME
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,value,points_at",
    [
        ("cols_adc", 512, "adcs"),
        ("adc_mux", 4, "DERIVED"),
        ("stored_cols", 2048, "cols"),
        ("adc_capacity_mux", 4, "DERIVED"),
    ],
)
def test_a_retired_macro_key_is_refused_and_names_its_replacement(key, value, points_at):
    def mutate(raw):
        raw["cim"]["analog"][key] = value

    with pytest.raises(ValueError) as excinfo:
        _hw(GRANITE_1B_HW, mutate)
    message = str(excinfo.value)
    assert key in message
    assert points_at in message
    assert "D35-REVISED" in message


def test_the_retired_keys_are_refused_on_a_card_too():
    def mutate(raw):
        raw["cim"]["cards"]["ctt_sheet"]["params"]["cols_adc"] = 512

    with pytest.raises(ValueError, match="RETIRED macro key"):
        _hw(GRANITE_1B_HW, mutate)


def test_the_old_names_survive_only_as_read_only_views_for_the_pricing_laws():
    # cim_timing counts MUX SLOTS per macro and STORED COLUMNS PER SET. Those
    # are Z and Y/Z under the new names, and the laws keep asking for them by
    # the old ones — as properties, which no config can write.
    analog = _hw(GRANITE_1B_HW).cim_config.analog
    assert (analog.rows, analog.cols, analog.adcs) == SUBSET_MACRO
    assert analog.mux == 4 and analog.adc_mux == 4
    assert analog.cols_adc == analog.adcs == 512
    assert analog.label == "512 x 2048 : 512"


# ---------------------------------------------------------------------------
# 3. The four subset_v1 cards ARE the platform macro, priced from the sheet
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", SUBSET_V1)
def test_a_subset_config_is_the_platform_macro_with_sheet_numbers(name):
    rows, cols, adcs = SUBSET_MACRO
    raw = _raw(HW / name)
    analog = raw["cim"]["analog"]
    card = raw["cim"]["cards"]["ctt_sheet"]
    derived = cim_analog_calc.derive_analog_card(rows, cols, adcs)
    # The CARD declares the macro and nothing else.
    assert (card["params"]["rows"], card["params"]["cols"], card["params"]["adcs"]) == (
        rows,
        cols,
        adcs,
    )
    assert "adc_capacity_mux" not in card
    # The FALLBACK block mirrors it EXACTLY — one file never has two answers.
    assert (analog["rows"], analog["cols"], analog["adcs"]) == (rows, cols, adcs)
    assert analog["energy_per_vec_pj"] == derived.energy_per_vec_pj
    assert analog["area_mm2_per_array"] == derived.area_mm2_per_array
    assert analog["slice_cycles"] == derived.slice_cycles
    assert analog["analog_clock_mhz"] == derived.analog_clock_mhz
    # No config in the set buys an exemption it does not need.
    assert "nonstandard_macro" not in analog


@pytest.mark.parametrize("name", SUBSET_V1)
def test_a_subset_config_parses_and_the_parsed_card_is_the_sheet(name):
    rows, cols, adcs = SUBSET_MACRO
    hw = _hw(HW / name)
    card = hw.cim_config.analog_card
    derived = cim_analog_calc.derive_analog_card(rows, cols, adcs)
    assert (card.params.rows, card.params.cols, card.params.adcs) == SUBSET_MACRO
    assert card.params.mux == derived.mux == 4
    assert card.params.energy_per_vec_pj == derived.energy_per_vec_pj
    assert card.params.area_mm2_per_array == derived.area_mm2_per_array
    assert card.calc_profile == "george_cima_mux1"
    assert card.derived_mux == 4
    assert card.nonstandard_macro is None


@pytest.mark.parametrize("name", SUBSET_V1)
def test_a_subset_sweep_declares_the_platform_box_and_not_the_fold(name):
    axes = (_raw(HW / name).get("mapping_dse") or {}).get("axes") or {}
    assert "adc_capacity_mux" not in axes
    assert set(axes["macro_z"]) <= set(cim_analog_calc.MACRO_Z_VALUES)
    assert set(axes["macro_y"]) <= set(cim_analog_calc.MACRO_Y_VALUES)
    # Both axes are really swept — a one-value axis would be a pinned knob
    # wearing an axis's name.
    assert len(axes["macro_z"]) >= 2 and len(axes["macro_y"]) >= 2


def test_the_sheet_anchor_still_reproduces_under_the_new_names():
    # The ViT card as shipped until D35 was 1280 x 1280 : 320 (mux 4). The
    # SEMANTICS did not move, only the names, and this is how that is checked.
    card = cim_analog_calc.derive_analog_card(1280, 1280, 320)
    assert card.mux == 4
    assert card.area_mm2_per_array == pytest.approx(1.403065, abs=1e-6)
    assert card.energy_per_vec_pj == pytest.approx(9534.9389, abs=1e-3)
    assert card.label == "1280 x 1280 : 320"


# ---------------------------------------------------------------------------
# 4. The sweep walks the BOX; the fold is refused as an axis
# ---------------------------------------------------------------------------


def _dse_module():
    spec = importlib.util.spec_from_file_location(
        "_d35_dse", REPO / "tools" / "fws_qif_dse.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_two_platform_axes_are_the_ones_that_exist():
    dse = _dse_module()
    assert "macro_z" in dse.AXIS_TARGETS and "macro_y" in dse.AXIS_TARGETS
    assert dse.AXIS_TARGETS["macro_z"].endswith(".params.adcs")
    assert dse.AXIS_TARGETS["macro_y"].endswith(".params.cols")
    assert "adc_capacity_mux" not in dse.AXIS_TARGETS


@pytest.mark.parametrize(
    "axis",
    ["rows", "cols", "cols_adc", "stored_cols", "adc_mux", "adc_capacity_mux", "mux"],
)
def test_the_macro_and_its_fold_are_refused_as_sweep_axes(axis):
    dse = _dse_module()
    assert axis in dse.REFUSED_AXES
    with pytest.raises(dse.QifDseUsageError) as excinfo:
        dse.SweepSpec.from_raw({dse.DSE_BLOCK: {"axes": {axis: [512, 1024]}}}, "d35")
    message = str(excinfo.value)
    assert axis in message
    assert "D35-REVISED" in message
    # The refusal names what IS on the sweep instead.
    assert "macro_z" in message and "macro_y" in message


def test_an_axis_value_off_the_platform_lists_is_refused_by_name():
    dse = _dse_module()
    for axis, bad in (("macro_z", 1024), ("macro_y", 8192), ("macro_y", 3000)):
        with pytest.raises(dse.QifDseUsageError) as excinfo:
            dse.SweepSpec.from_raw(
                {dse.DSE_BLOCK: {"axes": {axis: [bad]}}}, "d35"
            )
        assert axis in str(excinfo.value)
        assert "platform admits" in str(excinfo.value)


def test_a_corner_below_the_diagonal_is_not_a_macro_and_says_so():
    # Y >= Z always. With the shipped lists this corner cannot be DECLARED
    # (Y's floor IS Z's ceiling), so the law is exercised directly — it is the
    # rule that keeps a widened list honest, and a rule nothing checks rots.
    dse = _dse_module()
    spec = dse.SweepSpec.from_raw(
        {dse.DSE_BLOCK: {"axes": {"macro_z": [512], "macro_y": [512]}}}, "d35"
    )
    assert spec.illegal_reason({"macro_z": 512, "macro_y": 512}) is None  # mux 1
    reason = spec.illegal_reason({"macro_z": 512, "macro_y": 256})
    assert reason is not None and "Y >= Z always" in reason
    assert spec.illegal_reason({"macro_z": 512, "macro_y": 4096}) is None


def test_every_point_reports_the_macro_it_built_and_the_fold_that_followed():
    # mux is DERIVED, so the report is the ONE place a reader meets it.
    hw = _hw(GRANITE_1B_HW)
    analog = hw.cim_config.analog
    assert analog.label == "512 x 2048 : 512"
    assert analog.mux == 4
    card = hw.cim_config.analog_card
    assert card.calc_derivation["label"] == "512 x 2048 : 512"
    assert card.calc_derivation["mux"] == 4
    assert card.calc_derivation["cells"] == 512 * 2048


def test_the_vitality_referee_flags_an_axis_that_chose_nothing():
    dse = _dse_module()
    candidates = [
        {"id": "c0", "knobs": {"macro_z": 512, "bank_depth": 1}},
        {"id": "c1", "knobs": {"macro_z": 8, "bank_depth": 1}},
    ]
    declared = {"macro_z": [512, 8], "bank_depth": [1, 2]}
    rows = {
        r["axis"]: r
        for r in dse.axis_vitality(
            candidates, ["c0", "c1"], ("macro_z", "bank_depth"), declared=declared
        )
    }
    assert rows["macro_z"]["vital"] is True
    assert rows["macro_z"]["distinct"] == 2
    assert rows["bank_depth"]["vital"] is False
    assert "DEMOTION CANDIDATE" in rows["bank_depth"]["note"]
    # It is a SIGNAL, referred to the bundle — never a verdict from one model.
    assert "BUNDLE" in rows["bank_depth"]["note"]


def test_an_axis_the_sweep_pinned_is_not_a_demotion_candidate():
    # A sweep that HOLDS an axis at one value will always see one value on the
    # front. That is arithmetic about the run, not evidence about the axis, and
    # calling it a demotion signal would nominate whatever a smoke pinned.
    dse = _dse_module()
    candidates = [{"id": "c0", "knobs": {"layers_per_chip": 8}}]
    row = dse.axis_vitality(
        candidates, ["c0"], ("layers_per_chip",), declared={"layers_per_chip": [8]}
    )[0]
    assert row["vital"] is None
    assert "PINNED" in row["note"]
    assert "DEMOTION" not in row["note"]


def test_the_edge_crowding_referee_names_a_front_that_sits_on_a_wall():
    dse = _dse_module()
    candidates = [
        {"id": "c0", "knobs": {"macro_y": 4096, "macro_z": 512}},
        {"id": "c1", "knobs": {"macro_y": 4096, "macro_z": 256}},
    ]
    rows = dse.edge_crowding(candidates, ["c0", "c1"])
    ceilings = [r for r in rows if r["axis"] == "macro_y" and r["bound"] == "ceiling"]
    assert len(ceilings) == 1
    row = ceilings[0]
    assert row["value"] == 4096 and row["share_of_front"] == 1.0
    assert "ADJUDICATED" in row["note"]
    assert sorted(row["front_points"]) == ["c0", "c1"]


# ---------------------------------------------------------------------------
# 5. The tiling path at rows != hidden: Granite-4.0-H-1B on 512-row macros
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def granite_1b():
    return _mapping(GRANITE_1B_HW, GRANITE_1B_MODEL)


@pytest.fixture(scope="module")
def mamba2():
    return _mapping(MAMBA2_HW, MAMBA2_MODEL)


def test_a_hidden_deep_matrix_becomes_three_row_tiles(granite_1b):
    rows = int(granite_1b.device.analog.rows)
    assert rows == 512
    packing = next(iter(granite_1b.packings.values()))
    width = packing.column_sets_per_bank * int(granite_1b.device.analog.adcs)
    by_owner = {}
    for tile in packing.tiles:
        by_owner.setdefault(tile.owner, []).append(tile)
    checked = 0
    for owner, tiles in by_owner.items():
        k = max(tile.k_end for tile in tiles)
        n = max(tile.n_end for tile in tiles)
        row_blocks = len({(tile.k_start, tile.k_end) for tile in tiles})
        col_blocks = len({(tile.n_start, tile.n_end) for tile in tiles})
        # THE CEIL ARITHMETIC, not a stored count.
        assert row_blocks == -(-k // rows), owner.label
        assert col_blocks == -(-n // width), owner.label
        assert len(tiles) == row_blocks * col_blocks, owner.label
        if k == 1536:
            # hidden 1536 on 512 rows IS three row tiles. This is the shape the
            # platform macro creates and the one this file exists to pin.
            assert row_blocks == 3
            checked += 1
    assert checked, "no hidden-deep tensor found; the fixture no longer tests D35"


def test_the_k_stacks_are_real_and_the_dag_prices_every_in_macro_run(granite_1b):
    summary = granite_1b.packing_summary()
    assert summary["k_stacks"] > 0
    # The packer describes one segment per in-macro RUN, and there are more of
    # them than there are wholly local stacks: the extra ones are the runs
    # inside stacks that spill onto the next macro.
    assert summary["local_k_segments"] > summary["local_k_stacks"]

    program = build_fws_program(granite_1b)
    # NOTHING is left counted-but-unpriced.
    assert program.meta.misc["fws_undescribed_local_k_stacks"] == 0
    annotations = annotations_of(program)
    accumulations = [a for a in annotations if a.law == LAW_K_ACCUMULATION]
    assert accumulations
    for annotation in accumulations:
        assert annotation.k_stack is not None
        assert annotation.k_stack.local and annotation.k_stack.depth > 1
        # On the owning macro's own pool: the partials never leave it.
        assert annotation.macro_id == annotation.k_stack.sink_macro_id
    assert [a for a in annotations if a.law == LAW_ANALOG_GEMM]


def test_the_segment_law_is_what_prices_mamba2_at_all(mamba2):
    """THE MEASUREMENT THAT JUSTIFIES THE FIX.

    On 512-row macros Mamba2-2.7B has NO wholly-local K stack: every stack
    spans macros. Under the old local-or-spread law that meant not one
    accumulator op existed for it, while thousands of in-macro adds really
    happened. The segment law describes and prices exactly those.
    """
    summary = mamba2.packing_summary()
    assert summary["local_k_stacks"] == 0
    assert summary["spread_k_stacks"] > 0
    assert summary["local_k_segments"] > 0
    program = build_fws_program(mamba2)
    assert program.meta.misc["fws_undescribed_local_k_stacks"] == 0
    accumulations = [
        a for a in annotations_of(program) if a.law == LAW_K_ACCUMULATION
    ]
    assert accumulations, "every one of these was unpriced before the segment law"


def test_the_remainder_shows_up_in_the_reported_waste(granite_1b):
    summary = granite_1b.packing_summary()
    # ONE accounting: real + remainder + tail == committed, with no rounding.
    assert (
        summary["real_cells"] + summary["remainder_cells"] + summary["tail_cells"]
        == summary["committed_cells"]
    )
    assert summary["waste_pct"] == pytest.approx(
        100.0
        * (summary["remainder_cells"] + summary["tail_cells"])
        / summary["committed_cells"]
    )
    assert summary["waste_pct"] > 0.0


def test_the_dedicated_placement_still_leaves_its_in_macro_runs_unpriced():
    """A NAMED GAP, not a silence (D28).

    The ViT members place DEDICATED, so no packer runs and no segment
    descriptor exists — the DAG counts the in-macro runs it finds and P4 turns
    the count into a banner. X = 512 makes that count large (hidden 1280 is 3
    row tiles), so the gap is pinned here with a number rather than left to be
    discovered by whoever next reads a ViT report.
    """
    mapping = _mapping(
        HW / "fws_cim_vit_frontier.yaml",
        MODELS / "vit_huge_story_64_inf.yaml",
        mode="VIT",
    )
    assert mapping.packing == "dedicated"
    assert mapping.packings == {}
    program = build_fws_program(mapping)
    assert program.meta.misc["fws_undescribed_local_k_stacks"] > 0
    assert [a for a in annotations_of(program) if a.law == LAW_K_ACCUMULATION] == []


# ---------------------------------------------------------------------------
# 6. The law fix: an in-macro RUN of a spread stack is priced
# ---------------------------------------------------------------------------


def _fine_bank_device():
    """The MoE card at bank_depth 1 — the finest allocation the card admits.

    The device comes off a real MAPPING rather than being assembled by hand, so
    the packer under test is the one the run path uses.
    """
    hw = _hw(
        MOE_HW,
        lambda raw: raw["cim"].__setitem__(
            "cards", {"ctt": {"kind": "analog_macro", "device": "ctt", "bank_depth": 1}}
        ),
    )
    model = config.parse_config(str(MOE_MODEL), "LLM")
    return fws_mapping.build_mapping(hw, model).device


@pytest.fixture(scope="module")
def straddling_pack():
    """A depth-3 stack that straddles a macro boundary: blocks on (0, 0, 1).

    The pad tensor takes the first two banks, so the ffn2 stack starts at slot
    2 of macro 0 and its third block lands on macro 1. Two of its three blocks
    reduce INSIDE macro 0, which is the shape the old law counted and never
    priced.
    """
    device = _fine_bank_device()
    pad = TileOwner(model="m", layer=0, op="pad", expert=-1, shard=0)
    ffn2 = TileOwner(model="m", layer=0, op="ffn2", expert=-1, shard=0)
    return device.dense_pack(
        [PackRequest(owner=pad, k=1024, n=512), PackRequest(owner=ffn2, k=3072, n=256)],
        column_sets_per_tile=1,
    )


def test_the_straddling_stack_is_still_reported_as_spread(straddling_pack):
    # The WHOLE-STACK accounting does not move: it spans two macros and the
    # summary says so, exactly as before.
    stacks = [s for s in straddling_pack.k_stacks if s.depth > 1]
    assert len(stacks) == 1
    assert stacks[0].macro_ids == (0, 0, 1)
    assert not stacks[0].local
    assert len(straddling_pack.spread_k_stacks) == 1
    assert len(straddling_pack.local_k_stacks) == 0


def test_its_in_macro_run_is_described_as_a_segment(straddling_pack):
    segments = straddling_pack.local_k_segments
    assert len(segments) == 1
    segment = segments[0]
    assert segment.depth == 2                 # the two blocks on macro 0
    assert segment.macro_ids == (0, 0)
    assert segment.local and segment.sink_macro_id == 0
    assert segment.owner.op == "ffn2"
    assert straddling_pack.summary()["local_k_segments"] == 1


def test_the_segment_is_priced_by_the_same_p7_2_law(straddling_pack):
    device = _fine_bank_device()
    cost = device.price_accumulation(straddling_pack.local_k_segments[0], 1.0)
    # It is priced HERE — the row-block law prices only what crosses macros.
    assert cost.priced_here
    assert cost.depth == 2
    assert cost.time_s > 0.0


def test_a_wholly_local_stack_is_one_segment_equal_to_itself():
    # The identity that keeps every already-priced number where it was: when a
    # stack does not straddle, the segment IS the stack, so the DAG emits the
    # same one accumulator with the same depth, width and sink as before.
    device = _fine_bank_device()
    owner = TileOwner(model="m", layer=0, op="ffn2", expert=-1, shard=0)
    packing = device.dense_pack(
        [PackRequest(owner=owner, k=3072, n=256)], column_sets_per_tile=1
    )
    stacks = [s for s in packing.k_stacks if s.depth > 1]
    assert len(stacks) == 1 and stacks[0].local
    assert len(packing.local_k_segments) == 1
    assert packing.local_k_segments[0] == stacks[0]


def test_a_stack_of_depth_one_is_no_segment_at_all():
    device = _fine_bank_device()
    owner = TileOwner(model="m", layer=0, op="ffn1", expert=-1, shard=0)
    packing = device.dense_pack([PackRequest(owner=owner, k=1024, n=1024)])
    assert all(stack.depth == 1 for stack in packing.k_stacks)
    assert packing.local_k_segments == ()


def test_the_run_helper_never_merges_a_macro_a_stack_came_back_to():
    # Runs, not a set. A macro visited twice with another between is two
    # accumulations, and reporting one would under-price the walk.
    assert cim_timing._contiguous_runs((0, 0, 1, 0)) == ((0, 2), (1, 1), (0, 1))


# ---------------------------------------------------------------------------
# 7. The legacy paths do not move
# ---------------------------------------------------------------------------


LEGACY_GEOMETRY = {
    "fws_cim_granite_fullstack_study.yaml": (1536, 1536, 384),
    "fws_cim_granite_tiny.yaml": (1536, 1536, 384),
    "fws_cim_granite_tiny_dse.yaml": (1536, 1536, 384),
    "fws_cim_granite_tiny_dse_selected.yaml": (1536, 1536, 384),
    "fws_cim_granite_tiny_frontier.yaml": (1536, 1536, 384),
    "fws_cim_llama7b.yaml": (4096, 4096, 1024),
    "fws_cim_llama7b_kvdram.yaml": (4096, 4096, 1024),
    "fws_cim_llama7b_mapped.yaml": (4096, 4096, 1024),
    "fws_cim_mamba.yaml": (1536, 1536, 384),
    "fws_cim_mamba_adc_hedge.yaml": (1536, 1536, 384),
    "fws_cim_mamba_adc_round2.yaml": (1536, 1536, 384),
    "fws_cim_mamba_adc_study.yaml": (1536, 1536, 384),
    "fws_cim_mamba_frontier.yaml": (1536, 1536, 384),
    "fws_cim_moe.yaml": (1024, 1024, 256),
    "fws_cim_optima_t1.yaml": (1280, 1280, 320),
    "fws_cim_optima_t2.yaml": (1408, 1408, 704),
    "fws_cim_optima_t3.yaml": (1280, 1280, 80),
    "fws_cim_qwen3_5_4b.yaml": (2560, 2560, 640),
    "fws_cim_qwen3_5_4b_frontier.yaml": (2560, 2560, 640),
    "fws_cim_vit.yaml": (1280, 1280, 320),
}

LEGACY_REASON = "predates D35; kept for validator/OPTIMA parity"


@pytest.mark.parametrize("name", sorted(LEGACY_GEOMETRY))
def test_a_legacy_config_keeps_its_macro_and_says_why(name):
    rows, cols, adcs = LEGACY_GEOMETRY[name]
    raw = _raw(HW / name)
    analog = raw["cim"]["analog"]
    assert (analog["rows"], analog["cols"], analog["adcs"]) == (rows, cols, adcs)
    assert analog["nonstandard_macro"] == LEGACY_REASON
    hw = _hw(HW / name)
    assert (hw.cim_config.analog.rows, hw.cim_config.analog.cols) == (rows, cols)
    assert hw.cim_config.analog.adcs == adcs
    assert hw.cim_config.analog_card.nonstandard_macro == LEGACY_REASON


def test_the_override_moves_no_number_at_all():
    # The exemption is a DISCLOSURE, not a knob: parsing the same file with the
    # key removed (and the check bypassed) yields the identical analog macro.
    raw = _raw(GRANITE_TINY_HW)
    with_key = copy.deepcopy(raw)
    config.convert(with_key)
    card_a = config.HWConfig.from_dict(with_key).cim_config.analog

    without = copy.deepcopy(raw)
    without["cim"]["analog"].pop("nonstandard_macro")
    config.convert(without)
    analog_b = config.CIMAnalogConfig.from_dict(without["cim"]["analog"])
    for field in (
        "rows",
        "cols",
        "adcs",
        "mux",
        "slice_cycles",
        "analog_clock_mhz",
        "energy_per_vec_pj",
        "shots_per_output",
        "area_mm2_per_array",
    ):
        assert getattr(card_a, field) == getattr(analog_b, field), field


def test_the_granite_tiny_placement_is_bit_identical():
    # granite_tiny is the legacy gate that the OPTIMA-era tests and the DSE
    # allocation artifact both read. It has NO in-macro run that was left
    # unpriced, so the segment law leaves it exactly where it was.
    mapping = _mapping(GRANITE_TINY_HW, GRANITE_TINY_MODEL)
    summary = mapping.packing_summary()
    assert summary["k_stacks"] == 19160
    assert summary["local_k_stacks"] == 72
    assert summary["spread_k_stacks"] == 72
    # One segment per wholly local stack, and not one more.
    assert summary["local_k_segments"] == 72
    program = build_fws_program(mapping)
    assert program.meta.misc["fws_undescribed_local_k_stacks"] == 0
