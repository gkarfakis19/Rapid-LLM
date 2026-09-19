"""The ADC-macro sheet in this repo: the PORT, the CARD and the MUX AXIS.

What this file pins, and why each is a wrong NUMBER rather than a crash if it
rots:

  1. **The port is the source.** :mod:`cim_analog_calc` is a port of
     ``/app/nanocad/projects/cim_ctt_big_optima/perf_model/config/calc_analog.py``
     (md5 c750713450e5f47b78c1135bc2cf7c46, read 2026-08-28). The grid below
     was CAPTURED FROM THAT FILE — a snapshot copy of it, run under the repo
     interpreter — over rows x cols in {768x768, 1536x1536, 1280x320} x
     adc_muxing in {1..256}, in both the tied call OPTIMA's own loader makes
     (``mux_factor = adc_muxing``) and a two-mux call (``mux_factor`` pinned at
     4 while the capacity mux moves), which is what separates the vector mux
     from the capacity mux. Parity is asserted as EXACT float equality: the
     port is expression-for-expression, so a rounding difference is a drift.
  2. **The card derives IFF it names a profile.** With ``calc_profile`` the
     card's footprint, per-vector energy and vec_latency come from the sheet at
     (X rows, Y total CTT columns, Z ADCs); without it NOTHING changes,
     and every shipped fws_cim config still parses to exactly the numbers its
     YAML declares.
  3. **A contradiction is refused BY NAME.** A hand-declared
     ``energy_per_vec_pj`` / ``area_mm2_per_array`` / ``slice_cycles`` /
     ``analog_clock_mhz`` beside a profile, a capacity mux without a profile,
     an unknown profile, a mux outside 1..256 or one that does not divide the
     stored columns: each raises naming the field and quoting the reason.
  4. **The axis is real and it is a TRADE.** The smoke sweep prices the ViT
     machine at three capacity muxes: vec_latency grows with the mux, the ADC
     area term falls with it, and every point carries the derived card's
     provenance disclosure.

The anchor worth knowing: the ViT card AS SHIPPED UNTIL D35 (rows 1280,
cols_adc 320, adc_mux 4, area 1.403065 mm2, energy 9534.9389 pJ) IS this calc at
1280 x 320 with capacity mux 4 AND vector mux 4, to full printed precision. The
cards were generated at OPTIMA's tie of the two muxes, so a card that adds
``calc_profile`` and nothing else derives the point it already had. The parity
grid below still reads that geometry, because it tests the CALC and the calc did
not move; what moved is the CARD. D35-REVISED (2026-08-29) makes the macro
X x Y : Z with X FIXED per platform (X = 512) and mux = Y/Z DERIVED, so the
shipped ViT card is now this same calc at 512 x 2048 : 512 (capacity mux 4) —
see tests/test_qif_platform_macro.py, which owns that claim.
"""

from __future__ import annotations

import copy
import importlib.util
import io
import json
from pathlib import Path

import pytest
import yaml

import cim_analog_calc
import config as config_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HW_DIR = PROJECT_ROOT / "configs" / "hardware-config"
MODEL_DIR = PROJECT_ROOT / "configs" / "model-config"
VIT_FRONTIER_HW = HW_DIR / "fws_cim_vit_frontier.yaml"
VIT_MODEL = MODEL_DIR / "vit_huge_story_64_inf.yaml"

#: Every shipped fws_cim hardware config: the bit-identity gate's population.
SHIPPED_FWS_CIM_CONFIGS = sorted(
    path
    for path in HW_DIR.glob("fws_cim*.yaml")
    if isinstance(yaml.safe_load(io.open(path, encoding="utf-8")), dict)
)


def _load_dse_tool():
    """Import tools/fws_qif_dse.py in-process (tools/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "fws_qif_dse_under_analog_calc_test", str(PROJECT_ROOT / "tools" / "fws_qif_dse.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 1. THE PARITY GRID — captured from the SOURCE, not from the port
#
# Capture command (the snapshot copy lives in the job scratch; the source is
# read-only and was never written to):
#
#   cp /app/nanocad/projects/cim_ctt_big_optima/perf_model/config/calc_analog.py \
#      <scratch>/optima_calc_analog_snapshot.py
#   PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -c "build_calc_config over the grid"
#
# Columns: rows, cols, mux_factor (VECTOR), adc_muxing (CAPACITY), area_mm2,
# energy_per_vec_pJ, slice_cycles, vec_cycles.
# ---------------------------------------------------------------------------

PARITY_GRID = (
    (768, 768, 1, 1, 1.314362671104, 4912.64208470016, 2.0, 2.0),
    (768, 768, 4, 1, 1.314362671104, 19650.56833880064, 2.0, 8.0),
    (768, 768, 2, 2, 1.630984126464, 9953.28069599232, 2.0, 4.0),
    (768, 768, 4, 2, 1.630984126464, 19906.56139198464, 2.0, 8.0),
    (768, 768, 4, 4, 2.264227037184, 20418.54749835264, 2.0, 8.0),
    (768, 768, 4, 4, 2.264227037184, 20418.54749835264, 2.0, 8.0),
    (768, 768, 8, 8, 3.530712858624, 42885.03942217728, 2.0, 16.0),
    (768, 768, 4, 8, 3.530712858624, 21442.51971108864, 2.0, 8.0),
    (768, 768, 16, 16, 6.063684501504, 93961.85654624256, 2.0, 32.0),
    (768, 768, 4, 16, 6.063684501504, 23490.46413656064, 2.0, 8.0),
    (768, 768, 32, 32, 11.129627787264, 220690.82390003712, 2.0, 64.0),
    (768, 768, 4, 32, 11.129627787264, 27586.35298750464, 2.0, 8.0),
    (768, 768, 64, 64, 21.261514358784, 572450.0910302823, 2.0, 128.0),
    (768, 768, 4, 64, 21.261514358784, 35778.130689392645, 2.0, 8.0),
    (768, 768, 128, 128, 41.525287501823996, 1669173.9549813967, 2.0, 256.0),
    (768, 768, 4, 128, 41.525287501823996, 52161.686093168646, 2.0, 8.0),
    (768, 768, 256, 256, 82.052833787904, 5435443.001646122, 2.0, 512.0),
    (768, 768, 4, 256, 82.052833787904, 84928.79690072066, 2.0, 8.0),
    (1536, 1536, 1, 1, 3.139088252928, 10929.46753880064, 2.0, 2.0),
    (1536, 1536, 4, 1, 3.139088252928, 43717.87015520256, 2.0, 8.0),
    (1536, 1536, 2, 2, 4.282694074368, 22370.92118396928, 2.0, 4.0),
    (1536, 1536, 4, 2, 4.282694074368, 44741.84236793856, 2.0, 8.0),
    (1536, 1536, 4, 4, 6.569905717248, 46789.78679341056, 2.0, 8.0),
    (1536, 1536, 4, 4, 6.569905717248, 46789.78679341056, 2.0, 8.0),
    (1536, 1536, 8, 8, 11.144329003008, 101771.35128870912, 2.0, 16.0),
    (1536, 1536, 4, 8, 11.144329003008, 50885.67564435456, 2.0, 8.0),
    (1536, 1536, 16, 16, 20.293175574528, 236309.81338497024, 2.0, 32.0),
    (1536, 1536, 4, 16, 20.293175574528, 59077.45334624256, 2.0, 8.0),
    (1536, 1536, 32, 32, 38.590868717568, 603688.0700001485, 2.0, 64.0),
    (1536, 1536, 4, 32, 38.590868717568, 75461.00875001856, 2.0, 8.0),
    (1536, 1536, 64, 64, 75.186255003648, 1731649.912921129, 2.0, 128.0),
    (1536, 1536, 4, 64, 75.186255003648, 108228.11955757056, 2.0, 8.0),
    (1536, 1536, 128, 128, 148.377027575808, 5560394.917525588, 2.0, 256.0),
    (1536, 1536, 4, 128, 148.377027575808, 173762.3411726746, 2.0, 8.0),
    (1536, 1536, 256, 256, 294.758572720128, 19509170.201784488, 2.0, 512.0),
    (1536, 1536, 4, 256, 294.758572720128, 304830.7844028826, 2.0, 8.0),
    (1280, 320, 1, 1, 0.7946366361599999, 2250.405003264, 2.0, 2.0),
    (1280, 320, 4, 1, 0.7946366361599999, 9001.620013056, 2.0, 8.0),
    (1280, 320, 2, 2, 0.99744598016, 4589.696483328, 2.0, 4.0),
    (1280, 320, 4, 2, 0.99744598016, 9179.392966656, 2.0, 8.0),
    (1280, 320, 4, 4, 1.4030646681599999, 9534.938873856001, 2.0, 8.0),
    (1280, 320, 4, 4, 1.4030646681599999, 9534.938873856001, 2.0, 8.0),
    (1280, 320, 8, 8, 2.2143020441599996, 20492.061376512, 2.0, 16.0),
    (1280, 320, 4, 8, 2.2143020441599996, 10246.030688256, 2.0, 8.0),
    (1280, 320, 16, 16, 3.8367767961599992, 46672.857268224005, 2.0, 32.0),
    (1280, 320, 4, 16, 3.8367767961599992, 11668.214317056001, 2.0, 8.0),
    (1280, 320, 32, 32, 7.081726300159999, 116100.652597248, 2.0, 64.0),
    (1280, 320, 4, 32, 7.081726300159999, 14512.581574656, 2.0, 8.0),
    (1280, 320, 64, 64, 13.571625308159998, 323221.057437696, 2.0, 128.0),
    (1280, 320, 4, 64, 13.571625308159998, 20201.316089856, 2.0, 8.0),
    (1280, 320, 128, 128, 26.551423324159995, 1010521.1238481922, 2.0, 256.0),
    (1280, 320, 4, 128, 26.551423324159995, 31578.785120256005, 2.0, 8.0),
    (1280, 320, 256, 256, 52.51101935615999, 3477358.2835875847, 2.0, 512.0),
    (1280, 320, 4, 256, 52.51101935615999, 54333.72318105601, 2.0, 8.0),
)

CLI_STATS = (
    (768, 768, 4912.64208470016, 1.314362671104, 1.0, 1.0, 2.0, 2.0, 1.0),
    (1536, 1536, 10929.46753880064, 3.139088252928, 1.0, 1.0, 2.0, 2.0, 1.0),
    (1280, 320, 2250.405003264, 0.7946366361599999, 1.0, 1.0, 2.0, 2.0, 1.0),
)

#: The source CLI (``calc_analog.py --rows R --cols C``), which carries no mux
#: flag and therefore runs at the profile's own adc_muxing = 1. Printed to six
#: decimals by the CLI; pinned here at full precision AND at the CLI's own
#: formatting, so the port reproduces what a human reading that CLI would see.
CLI_PRINTED = (
    (768, 768, "4912.642085", "1.314363"),
    (1536, 1536, "10929.467539", "3.139088"),
    (1280, 320, "2250.405003", "0.794637"),
)


@pytest.mark.parametrize("row", PARITY_GRID)
def test_port_reproduces_optima_calc_exactly(row):
    """EXACT parity with the source over the whole rows x cols x mux grid."""
    rows, cols, mux_factor, adc_muxing, area_mm2, energy_pj, slice_cycles, vec_cycles = row
    cfg = cim_analog_calc.build_calc_config(
        {
            "calc_profile": "george_cima_mux1",
            "array_shapes": [(rows, cols)],
            "mux_factor": mux_factor,
            "adc_muxing": adc_muxing,
        }
    )
    array_spec = cfg["arrays"][0]
    assert array_spec["area_mm2"] == area_mm2
    assert array_spec["energy_per_vec_pJ"] == energy_pj
    assert float(cfg["slice_cycles"]) == slice_cycles
    assert float(cfg["mux_factor"]) * float(cfg["slice_cycles"]) == vec_cycles


@pytest.mark.parametrize("row", CLI_STATS)
def test_port_reproduces_the_source_cli(row):
    """``calculate_array_stats`` with no mux argument IS the source's CLI call."""
    (rows, cols, energy_pj, area_mm2, mux_factor, adc_muxing,
     slice_cycles, vec_cycles, shots) = row
    stats = cim_analog_calc.calculate_array_stats(rows, cols)
    assert stats["energy_per_vec_pJ"] == energy_pj
    assert stats["area_mm2"] == area_mm2
    assert stats["mux_factor"] == mux_factor
    assert stats["adc_muxing"] == adc_muxing
    assert stats["slice_cycles"] == slice_cycles
    assert stats["vec_cycles"] == vec_cycles
    assert stats["shots_per_output"] == shots


@pytest.mark.parametrize("rows,cols,energy_text,area_text", CLI_PRINTED)
def test_port_prints_what_the_source_cli_prints(rows, cols, energy_text, area_text):
    stats = cim_analog_calc.calculate_array_stats(rows, cols)
    assert f"{stats['energy_per_vec_pJ']:.6f}" == energy_text
    assert f"{stats['area_mm2']:.6f}" == area_text


def test_area_terms_sum_to_the_ported_area():
    """The reported term split IS the ported sum, not a second model of it."""
    for rows, cols in ((768, 768), (1536, 1536), (1280, 320)):
        for mux in (1, 4, 64):
            stats = cim_analog_calc.calculate_array_stats(
                rows, cols, mux_factor=mux, adc_muxing=mux
            )
            terms = stats["area_terms_mm2"]
            assert sum(terms.values()) == pytest.approx(stats["area_mm2"], rel=1e-12)
            # The ADC term is the ADC COLUMN count times the sheet's per-column
            # area, and the capacity mux is not in it. This is the fact the
            # whole axis rests on.
            expected_adc = (
                cim_analog_calc.OPTIMA_CALC_PROFILES["george_cima_mux1"]["adc_area_per_col"]
                * cols
                * (0.104 ** 2)
                / 1e6
            )
            assert terms["adc"] == pytest.approx(expected_adc, rel=1e-12)


def test_constants_match_the_source_profile():
    """The ported sheet constants, spot-checked where the sheet is loudest."""
    profile = cim_analog_calc.OPTIMA_CALC_PROFILES["george_cima_mux1"]
    assert profile["adc_area_per_col"] == 55473.0
    assert profile["adc_fom_w"] == 5.2
    assert profile["analog_clock_MHz"] == 100
    assert profile["analog_clock_MAX"] == 100
    assert profile["num_cycles"] == 2.0
    assert profile["gate_pitch"] == 0.104
    assert cim_analog_calc.ALLOWED_CAPACITY_MUX == tuple(range(1, 257))
    assert cim_analog_calc.CALC_PROFILE_NAMES == ("george_cima_mux1",)


# ---------------------------------------------------------------------------
# 2. THE DERIVATION — capacity re-split, and what it does to each term
# ---------------------------------------------------------------------------


def test_derivation_reproduces_the_shipped_vit_card():
    """The shipped hand card IS the sheet at its own fold, to full precision."""
    # D35-REVISED: the call is (X, Y, Z) and the fold mux = Y/Z is DERIVED.
    derived = cim_analog_calc.derive_analog_card(1280, 1280, 320)
    assert derived.adcs == 320
    assert derived.mux == 4
    assert derived.slice_cycles == 2
    assert derived.vec_cycles == 8
    assert derived.analog_clock_mhz == 100
    assert derived.area_mm2_per_array == 1.4030646681599999
    assert derived.energy_per_vec_pj == 9534.938873856001
    # The shipped YAML rounds both to 7 significant figures.
    assert round(derived.area_mm2_per_array, 6) == 1.403065
    assert round(derived.energy_per_vec_pj, 4) == 9534.9389


def test_capacity_mux_divides_the_adc_bill_and_multiplies_the_time():
    """The trade, term by term, at a FIXED declared capacity."""
    points = [
        cim_analog_calc.derive_analog_card(1280, 1280, 1280 // mux)
        for mux in (1, 4, 16, 64)
    ]
    for previous, current in zip(points, points[1:]):
        assert current.vec_cycles > previous.vec_cycles
        assert current.adc_area_mm2 < previous.adc_area_mm2
        assert current.energy_per_vec_pj > previous.energy_per_vec_pj
        assert current.area_mm2_per_array < previous.area_mm2_per_array
        # The CELL array is invariant: the capacity did not move, only the fold.
        assert current.area_terms_mm2["ctt_array"] == pytest.approx(
            previous.area_terms_mm2["ctt_array"], rel=1e-12
        )
        # The ADC bill is exactly divided by the mux step.
        step = current.mux / previous.mux
        assert current.adc_area_mm2 * step == pytest.approx(
            previous.adc_area_mm2, rel=1e-12
        )


def test_derivation_refuses_a_mux_it_has_no_point_for():
    with pytest.raises(cim_analog_calc.AnalogCalcError) as excinfo:
        cim_analog_calc.derive_analog_card(1280, 1280, 3)  # 1280 / 3 is not a fold
    assert "must DIVIDE" in str(excinfo.value)
    with pytest.raises(cim_analog_calc.AnalogCalcError) as excinfo:
        cim_analog_calc.derive_analog_card(1280 * 512, 1280 * 512, 1)
    assert "1..256" in str(excinfo.value)
    with pytest.raises(cim_analog_calc.AnalogCalcError) as excinfo:
        cim_analog_calc.profile_params("no_such_profile")
    assert "no_such_profile" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 3. THE CARD SCHEMA — derivation iff calc_profile, refusals by name
# ---------------------------------------------------------------------------


def _derived_card_dict(**overrides):
    card = {
        "kind": "analog_macro",
        "device": "ctt",
        "calc_profile": "george_cima_mux1",
        "params": {"rows": 1280, "cols": 1280, "adcs": 320, "shots_per_output": 2},
    }
    card.update(overrides)
    return card


def _card(card_dict, analog_default=None):
    return config_module.CIMAnalogCardConfig.from_dict(card_dict, "ctt", analog_default)


@pytest.mark.parametrize(
    "adcs,mux,vec_cycles,area,energy",
    [
        (1280, 1, 2, 2.4741394022399996, 8801.172013055999),
        (320, 4, 8, 1.4030646681599999, 9534.938873856001),
        (80, 16, 32, 1.13529598464, 12470.006317055999),
    ],
)
def test_card_derives_from_the_sheet(adcs, mux, vec_cycles, area, energy):
    """D35-REVISED: the card declares X x Y : Z and the FOLD falls out.

    Y is held at 1280 and Z walks, so the CAPACITY is invariant and only the
    converter count moves — which is the same three points this grid always
    checked, now named by what a config actually states.
    """
    card = _card(_derived_card_dict(params=dict(rows=1280, cols=1280, adcs=adcs,
                                                shots_per_output=2)))
    assert card.calc_profile == "george_cima_mux1"
    assert card.derived_mux == mux
    params = card.params
    assert params.rows == 1280
    assert params.adcs == adcs
    assert params.mux == mux
    assert params.cols == 1280            # the CAPACITY is invariant
    assert params.label == f"1280 x 1280 : {adcs}"
    assert params.slice_cycles == 2
    assert params.analog_clock_mhz == 100
    assert params.shots_per_output == 2   # energy only, and still the card's
    assert params.area_mm2_per_array == area
    assert params.energy_per_vec_pj == energy
    assert params.mux * params.slice_cycles == vec_cycles
    assert card.calc_derivation["mux"] == mux
    assert card.calc_derivation["adcs"] == adcs
    assert "george_cima_mux1" in card.calc_derivation["provenance"]
    assert cim_analog_calc.CALC_SOURCE_PATH in card.calc_derivation["provenance"]


def test_card_without_a_profile_derives_nothing():
    """The default path, unchanged: declared numbers, no derivation."""
    declared = {
        "rows": 1280,
        "cols": 1280,
        "adcs": 320,
        "slice_cycles": 2,
        "analog_clock_mhz": 100,
        "energy_per_vec_pj": 9534.9389,
        "shots_per_output": 2,
        "area_mm2_per_array": 1.403065,
    }
    card = _card({"kind": "analog_macro", "device": "ctt", "params": copy.deepcopy(declared)})
    assert card.calc_profile is None
    assert card.derived_mux == 0
    assert card.calc_derivation is None
    assert card.params.energy_per_vec_pj == 9534.9389
    assert card.params.area_mm2_per_array == 1.403065


@pytest.mark.parametrize(
    "overrides,needle",
    [
        ({"calc_profile": "not_a_profile"}, "names no profile this repo carries"),
        ({"calc_profile": 4}, "must be the NAME of a sheet-grounded calc profile"),
        ({"params": {"rows": 1280, "cols": 1280, "adcs": 3}}, "must DIVIDE"),
        ({"params": {"rows": 1280, "cols": 1280, "adcs": 0}}, "adcs must be >= 1"),
        ({"adc_capacity_mux": 4}, "RETIRED macro key"),
    ],
)
def test_card_refuses_bad_profiles_and_macros_by_name(overrides, needle):
    with pytest.raises(ValueError) as excinfo:
        _card(_derived_card_dict(**overrides))
    assert needle in str(excinfo.value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("energy_per_vec_pj", 9534.9389),
        ("area_mm2_per_array", 1.403065),
        ("slice_cycles", 2),
        ("analog_clock_mhz", 100),
    ],
)
def test_card_refuses_a_hand_declared_derived_quantity(field, value):
    card_dict = _derived_card_dict()
    card_dict["params"] = dict(card_dict["params"], **{field: value})
    with pytest.raises(ValueError) as excinfo:
        _card(card_dict)
    message = str(excinfo.value)
    assert field in message
    assert "CONTRADICTION" in message


def test_card_refuses_a_profile_on_another_device_family():
    """The sheet is a CTT sheet; another family's card may not borrow it."""
    with pytest.raises(ValueError) as excinfo:
        _card(_derived_card_dict(device="reram"))
    message = str(excinfo.value)
    assert "device = 'reram'" in message
    assert "'ctt' sheet" in message


def test_card_refuses_the_retired_fold_with_or_without_a_profile():
    """D35-REVISED retires the fold as a config field entirely.

    ``adc_capacity_mux`` used to be the derived card's one axis. mux is now
    Y/Z, so declaring it is declaring a derived quantity — refused by name on
    a hand-declared card and on a profile card alike, pointing at what states
    the macro instead.
    """
    hand = {
        "kind": "analog_macro",
        "device": "ctt",
        "adc_capacity_mux": 4,
        "params": {
            "rows": 1280,
            "cols": 1280,
            "adcs": 320,
            "slice_cycles": 2,
            "analog_clock_mhz": 100,
            "energy_per_vec_pj": 9534.9389,
            "area_mm2_per_array": 1.403065,
        },
    }
    with pytest.raises(ValueError) as excinfo:
        _card(hand)
    assert "adc_capacity_mux" in str(excinfo.value)
    assert "DERIVED" in str(excinfo.value)
    with pytest.raises(ValueError, match="RETIRED macro key"):
        _card(_derived_card_dict(adc_capacity_mux=4))


def test_derived_card_must_declare_its_own_capacity():
    """No inheriting cim.analog: that block declares what the profile derives."""
    analog_default = config_module.CIMAnalogConfig.from_dict(
        {
            "rows": 1280,
            "cols": 1280,
            "adcs": 320,
            "slice_cycles": 2,
            "analog_clock_mhz": 100,
            "energy_per_vec_pj": 9534.9389,
            "area_mm2_per_array": 1.403065,
        }
    )
    card_dict = _derived_card_dict()
    card_dict.pop("params")
    with pytest.raises(ValueError) as excinfo:
        _card(card_dict, analog_default)
    assert "must supply its own 'params' block" in str(excinfo.value)


def test_derived_card_params_are_strict_keyed():
    card_dict = _derived_card_dict()
    card_dict["params"] = dict(card_dict["params"], adc_muxing=4)
    with pytest.raises(ValueError) as excinfo:
        _card(card_dict)
    assert "adc_muxing" in str(excinfo.value)


def test_the_fold_is_derived_from_the_macro_the_card_states():
    """A card states X x Y : Z; the fold is Y/Z and nothing declares it."""
    card = _card(_derived_card_dict())
    assert card.derived_mux == 4
    assert card.params.adcs == 320
    assert card.params.cols == 1280
    assert card.params.mux == 4
    assert card.params.label == "1280 x 1280 : 320"
    assert round(card.params.area_mm2_per_array, 6) == 1.403065


# ---------------------------------------------------------------------------
# 4. BIT-IDENTITY — a config that names no profile moves NOTHING
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path", SHIPPED_FWS_CIM_CONFIGS, ids=lambda path: Path(path).stem
)
def test_shipped_configs_are_untouched(path):
    """Every shipped fws_cim card still parses to the numbers its YAML says.

    Two card regimes are both SHIPPED policy now: the *_frontier.yaml configs
    (and the frontier-derived study configs) carry sheet-derived cards
    (calc_profile: george_cima_mux1) so the capacity mux is a sweepable axis;
    every other config keeps its declared, hand-pinned card. A derived card
    must carry its derivation; a declared card must carry none.
    """
    raw = yaml.safe_load(io.open(path, encoding="utf-8"))
    raw.pop("mapping_dse", None)
    declared = raw.get("cim", {}).get("analog")
    hw = config_module.HWConfig.from_dict(config_module.convert(copy.deepcopy(raw)))
    cards = hw.cim_config.cards
    declared_profiles = {
        name: (spec or {}).get("calc_profile")
        for name, spec in (raw.get("cim", {}).get("cards") or {}).items()
    }
    for name, card in cards.analog.items():
        if declared_profiles.get(name):
            assert card.calc_profile == declared_profiles[name], (
                f"{path}: card {name} declares calc_profile "
                f"{declared_profiles[name]!r} but parsed to {card.calc_profile!r}"
            )
            assert card.derived_mux >= 1
            assert card.calc_derivation is not None, (
                f"{path}: card {name} declares a profile but carries no derivation"
            )
            return  # a derived card overrides cim.analog; nothing below applies
        assert card.calc_profile is None, f"{path}: card {name} derives, but declares no profile"
        assert card.derived_mux == 0
        assert card.calc_derivation is None
    if declared is None:
        return
    params = cards.analog_card.params
    assert params.rows == declared["rows"]
    assert params.cols == declared["cols"]
    assert params.adcs == declared["adcs"]
    assert params.slice_cycles == declared["slice_cycles"]
    assert params.analog_clock_mhz == declared["analog_clock_mhz"]
    assert params.energy_per_vec_pj == declared["energy_per_vec_pj"]
    assert params.area_mm2_per_array == declared["area_mm2_per_array"]


# ---------------------------------------------------------------------------
# 5. THE DSE AXIS
# ---------------------------------------------------------------------------


def test_axis_table_names_the_platform_box_and_refuses_the_fold():
    """D35-REVISED: Z and Y are the axes; mux is derived and refused as one."""
    tool = _load_dse_tool()
    assert tool.AXIS_TARGETS["macro_z"] == "cim.cards.<analog card>.params.adcs"
    assert tool.AXIS_TARGETS["macro_y"] == "cim.cards.<analog card>.params.cols"
    assert "adc_capacity_mux" not in tool.AXIS_TARGETS
    assert "adc_capacity_mux" in tool.REFUSED_AXES
    assert "analog_clock_mhz" in tool.REFUSED_AXES
    reason = tool.REFUSED_AXES["analog_clock_mhz"]
    assert "George 2026-08-27: mux only; the analog clock stays at the sheet's 100 MHz" in reason


def test_axis_is_refused_on_a_hand_pinned_card():
    """A hand-pinned card cannot be macro-swept honestly — and says exactly that."""
    tool = _load_dse_tool()
    raw = yaml.safe_load(io.open(VIT_FRONTIER_HW, encoding="utf-8"))
    raw = tool._base_without_sweep(raw)
    raw["cim"]["cards"] = {
        "ctt": {"kind": "analog_macro", "device": "ctt"}  # inherits cim.analog: hand-pinned
    }
    with pytest.raises(tool.QifDseUsageError) as excinfo:
        tool.apply_point(raw, {"macro_z": 128}, "ctt", "default", str(VIT_FRONTIER_HW))
    message = str(excinfo.value)
    assert "hand-pinned card cannot be macro-swept honestly" in message
    assert "calc_profile" in message


def test_the_fill_fraction_is_one_list_owned_by_the_mapper():
    """D38's pp_util is spelled on the config surface and MEANT in the mapper.

    The failure this refuses is a fraction that parses in a YAML, reaches a
    sweep as a legal rung and is then refused (or worse, silently accepted)
    by the module that turns it into resident streams. So there is exactly ONE
    list and ONE check: ``fws_mapping`` owns both, ``config.parse_pp_util``
    delegates, and the sweep tool validates its declared rungs through the same
    function — the same shape ``_parse_packing`` uses for the packing laws.
    """
    import config as config_module
    import fws_mapping

    tool = _load_dse_tool()
    # config restates NEITHER the list nor the check.
    assert not hasattr(config_module, "PP_UTIL_VALUES")
    assert fws_mapping.PP_UTIL_VALUES == (1.0, 0.75, 0.5)
    # The in-process default cannot drift from the mapper's full fill.
    assert config_module.MappingSystemConfig().pp_util == fws_mapping.PP_UTIL_FULL
    # An absent key IS the full pipeline, so no shipped config moves.
    assert config_module.MappingSystemConfig.from_dict({}, "mapping").pp_util == 1.0
    for fill in fws_mapping.PP_UTIL_VALUES:
        assert config_module.parse_pp_util(fill, "mapping.pp_util") == fill
        assert tool._axis_value("pp_util", fill, "axes.pp_util") == fill
    # A fraction nobody adjudicated is refused on BOTH surfaces, and the axis
    # refusal is the mapper's own sentence quoting D38.
    with pytest.raises(ValueError):
        config_module.parse_pp_util(0.6, "mapping.pp_util")
    with pytest.raises(tool.QifDseUsageError) as excinfo:
        tool._axis_value("pp_util", 0.6, "mapping_dse.axes.pp_util")
    assert "D38" in str(excinfo.value)
    # pp_util is the axis D38 put in the partition's place.
    assert tool.AXIS_TARGETS["pp_util"] == "mapping.pp_util"


def test_analog_clock_axis_is_refused_at_parse():
    tool = _load_dse_tool()
    raw = yaml.safe_load(io.open(VIT_FRONTIER_HW, encoding="utf-8"))
    raw["mapping_dse"]["axes"] = {"analog_clock_mhz": [100, 200]}
    with pytest.raises(tool.QifDseUsageError) as excinfo:
        tool.SweepSpec.from_raw(raw, str(VIT_FRONTIER_HW))
    assert "analog_clock_mhz" in str(excinfo.value)
    assert "100 MHz" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 6. THE SMOKE SWEEP — the axis priced end to end on the cheapest machine
# ---------------------------------------------------------------------------


def _smoke_sweep_config(tmp_path):
    """The shipped ViT frontier machine, with its card DERIVED instead of hand-declared."""
    raw = yaml.safe_load(io.open(VIT_FRONTIER_HW, encoding="utf-8"))
    analog = raw["cim"]["analog"]
    raw["cim"]["cards"] = {
        "ctt_sheet": {
            "kind": "analog_macro",
            "device": "ctt",
            "calc_profile": "george_cima_mux1",
            "params": {
                "rows": int(analog["rows"]),
                "cols": int(analog["cols"]),
                "adcs": int(analog["adcs"]),
                "shots_per_output": int(analog["shots_per_output"]),
            },
        }
    }
    # layers_per_chip 8, not the config's 32: on the D35-REVISED platform macro
    # (X = 512) a 32-layer chip overruns its activation SRAM, and a smoke that
    # only ever printed a memory refusal would test nothing. It is set on the
    # MACHINE, not as a one-value axis: D38 retired the partition as an axis
    # ("The partition is INTENT, not an axis"), so a sweep can no longer pin it
    # by declaring a single rung — which is the right shape anyway, because a
    # one-value axis was never a choice, only a way of writing down the machine.
    raw["cim"]["chip"]["layers_per_chip"] = 8
    raw["mapping"]["layers_per_chip"] = 8
    raw["mapping_dse"] = {
        "label": "D35-REVISED smoke: the Z axis of the platform box (ViT-Huge seq 64)",
        "objective": "throughput",
        "max_silicon_mm2": 20000,
        "search": "cross_product",
        # Y is HELD and Z walks, so the capacity is invariant across the three
        # points and the only thing moving is the converter count — which makes
        # the area/latency trade below a clean read of one axis.
        "axes": {"macro_z": [512, 128, 32]},
    }
    path = tmp_path / "fws_cim_vit_platform_box_smoke.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_smoke_sweep_prices_every_point_of_the_z_axis(tmp_path):
    tool = _load_dse_tool()
    hw_path = _smoke_sweep_config(tmp_path)
    exit_code, payload = tool.run_sweep(
        hw_path,
        VIT_MODEL,
        mode="VIT",
        output_dir=tmp_path / "artifacts",
        quiet=True,
    )
    assert exit_code == 0
    assert payload["num_valid"] == 3, payload
    report = json.loads((tmp_path / "artifacts" / "dse_report.json").read_text())
    by_z = {int(c["knobs"]["macro_z"]): c for c in report["candidates"]}
    assert sorted(by_z) == [32, 128, 512]

    cols = int(yaml.safe_load(io.open(hw_path, encoding="utf-8"))["cim"]["analog"]["cols"])
    footprints = []
    adc_terms = []
    latencies = []
    for adcs in (512, 128, 32):
        mux = cols // adcs
        candidate = by_z[adcs]
        assert candidate["ok"], candidate["fail_message"]
        # mux is DERIVED and REPORTED — it is never a knob.
        assert candidate["knobs_derived"]["macro_mux"] == mux
        assert candidate["knobs_derived"]["macro"].endswith(f": {adcs}")
        derived = [
            item
            for item in candidate["disclosures"]
            if item["constraint"] == "derived_analog_card"
        ]
        assert len(derived) == 1, candidate["disclosures"]
        note = derived[0]
        assert f"mux = Y/Z = {mux}" in note["value"]
        assert "george_cima_mux1" in note["value"]
        assert cim_analog_calc.CALC_SOURCE_PATH in note["reason"]

        card = cim_analog_calc.derive_analog_card(
            int(yaml.safe_load(io.open(hw_path, encoding="utf-8"))["cim"]["analog"]["rows"]),
            cols,
            adcs,
        )
        assert candidate["silicon"]["macro_footprint_mm2"] == card.area_mm2_per_array
        footprints.append(card.area_mm2_per_array)
        adc_terms.append(card.adc_area_mm2)
        latencies.append(card.vec_latency_s)

    # FEWER ADCs: less ADC area, less footprint, more time. One axis, three
    # numbers, all three moving the way the sheet says they must.
    assert latencies == sorted(latencies) and latencies[0] < latencies[-1]
    assert adc_terms == sorted(adc_terms, reverse=True) and adc_terms[0] > adc_terms[-1]
    assert footprints == sorted(footprints, reverse=True)
