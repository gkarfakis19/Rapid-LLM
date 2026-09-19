"""ADC-macro-sheet analog calc model: the CAPACITY mux as a design axis.

PROVENANCE (the same convention cim_timing.py uses for the SRAM macro and the
measured synthesis library: the number's origin is named in the file that
carries it, not in a commit message).

  SOURCE FILE   /app/nanocad/projects/cim_ctt_big_optima/perf_model/config/calc_analog.py
  SOURCE MD5    c750713450e5f47b78c1135bc2cf7c46
  READ ON       2026-08-28
  PROFILE       OPTIMA_CALC_PROFILES['george_cima_mux1'] — the OPTIMA.xlsx
                ADC-macro sheet constants (adc_area_per_col = 55473 um2/col,
                adc_fom_w = 5.2 fJ/conv-step, the WL/BL settle times and the
                CTT/mirror/amp currents), carried here VERBATIM.
  CONSUMER      OPTIMA's own config/loader.py: `materialize_analog_config`
                calls `build_calc_config` with mux_factor = adc_muxing =
                `pools.analog.mux_degree` and array shapes
                (hidden, hidden / mux_degree); `ALLOWED_MUX_DEGREES` is
                1..256 and the degree must DIVIDE the stored column count.
                Those two rules are reproduced here (:data:`ALLOWED_CAPACITY_MUX`,
                :func:`derive_analog_card`).

THE TWO MUXES (the distinction this module exists to make)
----------------------------------------------------------
``adc_muxing`` — CAPACITY mux: physical weight-column banks behind ONE ADC.
    It multiplies the CELL array (``ctt_array_area``), it multiplies the BL/SL
    driver pitch, and it multiplies the WL switching capacitance. It does NOT
    appear in the ADC term at all: ``adc_area = adc_area_per_col * cols``,
    where ``cols`` is the ADC COLUMN count. The ADC bill therefore falls only
    because a fixed stored-column capacity re-split at capacity mux M needs
    M times fewer ADCs — see :func:`derive_analog_card`, which is the only
    place that re-split happens.
``mux_factor`` — VECTOR mux: ADC slices needed for one full output vector.
    ``vec_cycles = mux_factor * slice_cycles``; ``energy_per_vec_pJ =
    energy_per_slice * mux_factor``. It touches NO area term.

They are INDEPENDENT INPUTS to :func:`build_calc_config` (the port keeps that,
and the parity grid exercises the split), and OPTIMA's own consumer TIES them:
one bank per ADC is one slice of the vector, so ``mux_factor = adc_muxing``.
:func:`derive_analog_card` ties them the same way, for the same reason, and
says so. Capacity mux > vector mux (OPTIMA's `bank_factor` > 1) is NOT exposed
here — it is a second knob no shipped point uses, and inventing its semantics
was refused.

WHAT WAS *NOT* PORTED (named omissions, so the gap is a fact and not a silence)
------------------------------------------------------------------------------
* ``LOOKUP_PROFILES`` and ``build_lookup_config`` — the hard-coded transformer
  and mamba area/energy tables. This module computes; it does not look up.
  (Recorded anchor: the transformer table's 1280x1280 row, 2.474 mm2 /
  8801.2 pJ, IS this calc at rows=cols=1280, adc_muxing=1.)
* ``OPTIMA_CALC_PROFILES['groot_cima_w5']`` — the 5-bit GR00T fabric profile
  (parallelism 4 x num_cycles 4 plus the serial-adder fields). No fws_cim card
  names it; the ``serial_adder_*`` GATES are ported, so adding the profile is
  data, not code.
* ``shots_per_output`` is read from the CARD, not from the profile: it is an
  energy-only multiplier this repo's analog law already applies
  (``cim_timing`` — E = M * energy_per_vec_pj * shots_per_output * arrays),
  so taking the profile's 1.0 would double-book it.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple


#: Where every constant below came from, quoted in the disclosure a derived
#: card rides into the artifacts.
CALC_SOURCE_PATH = (
    "/app/nanocad/projects/cim_ctt_big_optima/perf_model/config/calc_analog.py"
)
CALC_SOURCE_MD5 = "c750713450e5f47b78c1135bc2cf7c46"
CALC_SOURCE_READ_ON = "2026-08-28"


class AnalogCalcError(ValueError):
    """The calc model cannot answer, and no number is invented in its place."""


# ---------------------------------------------------------------------------
# THE SHEET CONSTANTS — verbatim from the source named above.
# ---------------------------------------------------------------------------

#: calc_analog.BASE_ANALOG_CONFIG, verbatim. `mux_factor` 4 / `slice_cycles` 2
#: here are the LOOKUP-mode defaults; calc mode overwrites both.
BASE_ANALOG_CONFIG: Dict[str, Any] = {
    "analog_clock_MHz": 100,
    "analog_clock_MAX": 100,
    "mux_factor": 4,
    "slice_cycles": 2,
    "shots_per_output": 2,  # e.g., pos/neg shots per output vector
    "capacity_mux_factor": 1.0,
    "area_model": None,
    "read_energy_model": None,
}

#: calc_analog.OPTIMA_CALC_PROFILES['george_cima_mux1'], verbatim.
OPTIMA_CALC_PROFILES: Dict[str, Dict[str, Any]] = {
    "george_cima_mux1": {
        "analog_clock_MHz": 100,
        "analog_clock_MAX": 100,
        "adc_muxing": 1,
        "adc_resolution_bits": 8,
        "parallelism": 8,
        "ctts_per_8bit": 1,
        "wl_settle_time": 1.0,
        "bl_settle_time": 2.5,
        "adc_compute_time": 4.0,
        "input_probability": 0.5,
        "num_cycles": 2.0,
        "active_cycles_per_ctt": 2.0,
        "effective_input_probability": 0.188375,
        "weight_probability": 0.11,
        "bl_vdd": 0.6,
        "wl_vdd": 0.4,
        "ff_energy": 2.9,
        "peak_ctt_current": 256.0,
        "mirror_current_ua": 288.0,
        "amp_current_ua": 80.0,
        "device_cap": 0.1,
        "cap_per_f": 0.025,
        "wl_length_per_mce": 5.0,
        "digital_pre_ff_per_row": 9.0,
        "digital_post_ff_per_col": 16.0,
        "adc_fom_w": 5.2,
        "gate_pitch": 0.104,
        "ff_area": 240.0,
        "ctt_y_dim": 5.0,
        "current_mirror_area_per_col": 40000.0,
        "adc_area_per_col": 55473.0,
        "wl_prg_driver_per_row_base": 594.0,
        "wl_inf_driver_per_row": 7888.0,
        "bl_driver_base": 40.0,
        "shots_per_output": 1.0,
        "default_array_shapes": [(768, 768)],
    },
}

#: The profile names a card may legally name. One entry today; an unlisted
#: name is refused BY NAME rather than defaulted to the only one there is.
CALC_PROFILE_NAMES: Tuple[str, ...] = tuple(sorted(OPTIMA_CALC_PROFILES))

#: OPTIMA config/loader.py: ``ALLOWED_MUX_DEGREES = tuple(range(1, 257))``.
ALLOWED_CAPACITY_MUX: Tuple[int, ...] = tuple(range(1, 257))

# ---------------------------------------------------------------------------
# D35-REVISED — THE MACRO IS X x Y : Z, AND MUX IS DERIVED.
#
# X rows (height: the analog sheet's CONSTRAINED dimension, fixed per
# platform), Y total CTT columns (the storage tile), Z ADCs (the FLOP
# producers). ``mux = Y / Z`` — muxed columns are storage, not throughput —
# and mux is a NOTATION, never an input and never a swept axis.
#
# WHY THE SWEEP WALKS (Z, Y) AND NOT MUX. Total system ADC throughput is
# cells / (X * mux): Y cancels, so mux is the one density label and makes the
# right plot LENS. But the physical CONSTRAINTS are native to Z (converter
# count, output interface, calibration) and to Y (wordline length, tile
# granularity), so the legal region is an axis-aligned box in (Z, Y) and the
# sweep walks that box. Sweeping mux instead would let an absurd corner hide
# inside a derived quantity.
#
# This is the ONE place the platform lives. config.py validates a card against
# it through :func:`check_platform_macro`; tools/fws_qif_dse.py sweeps
# :data:`MACRO_Z_VALUES` x :data:`MACRO_Y_VALUES` and refuses the retired mux
# axis BY NAME. Neither re-states a number.
# ---------------------------------------------------------------------------

#: X — macro ROWS, fixed per platform. George's device call (512 or 1024) is
#: pending; 512 is the standing value and this is the only place it is written.
PLATFORM_MACRO_ROWS: int = 512

#: Z — the ADC count. Powers of two from 8 up to X: never more converters than
#: there are rows to feed them.
MACRO_Z_VALUES: Tuple[int, ...] = tuple(
    1 << k for k in range(3, PLATFORM_MACRO_ROWS.bit_length()) if (1 << k) <= PLATFORM_MACRO_ROWS
)

#: Y — total CTT columns. Powers of two from X up to 8X. The 8X ceiling is
#: ADJUDICATED, not physical: an EDGE-CROWDING check reports by name when a
#: model's front crowds this wall, and raising it to 16X is then a one-line
#: change with data behind it.
MACRO_Y_CEILING_MULTIPLE: int = 8
MACRO_Y_VALUES: Tuple[int, ...] = tuple(
    PLATFORM_MACRO_ROWS << k
    for k in range(MACRO_Y_CEILING_MULTIPLE.bit_length())
    if (PLATFORM_MACRO_ROWS << k) <= PLATFORM_MACRO_ROWS * MACRO_Y_CEILING_MULTIPLE
)

#: The decision, quoted where it is enforced (D21: a refusal names its law).
PLATFORM_MACRO_DECISION: str = "D35-REVISED (GEORGE, 2026-08-29)"
PLATFORM_MACRO_QUOTE: str = (
    "The macro is X x Y : Z — X rows (height, the analog sheet's constrained "
    "dimension), Y total CTT columns (storage tile), Z ADCs (the FLOP producers; "
    "muxed columns are storage). mux = Y/Z, always derived notation. X is FIXED per "
    "platform. The PLATFORM AXES are Z in powers of two {8 .. X} — never more ADCs "
    "than rows — and Y in powers of two {X .. 8X}, with Y >= Z always (mux = 1 is "
    "legal). MUX IS NEVER SWEPT: it is derived, printed in every point's X x Y : Z "
    "label, and is the plot's LENS. Total system ADC throughput is cells/(X*mux) — Y "
    "cancels — so mux is the one density label; but the CONSTRAINTS are native to Z "
    "(converter count, output interface, calibration) and Y (wordline length, tile "
    "granularity), so the sweep walks the axis-aligned legal box and no absurd corner "
    "hides in a derived quantity."
)

#: The config key that buys an exemption, and it buys one only WITH A REASON.
NONSTANDARD_MACRO_KEY: str = "nonstandard_macro"
NONSTANDARD_MACRO_PATH: str = f"cim.analog.{NONSTANDARD_MACRO_KEY}"


def macro_label(rows: int, cols: int, adcs: int) -> str:
    """``X x Y : Z`` — the one way a macro is named in prose and in artifacts."""
    return f"{int(rows)} x {int(cols)} : {int(adcs)}"


def derived_mux(cols: int, adcs: int, *, context: str = "cim.analog") -> int:
    """``mux = Y / Z``, DERIVED. Never read from a config, never swept.

    A fold that leaves a remainder is not a macro: every muxed column has to
    sit behind some ADC, so Z must divide Y exactly. The refusal names both
    numbers rather than rounding one of them.
    """
    cols = int(cols)
    adcs = int(adcs)
    if adcs <= 0 or cols <= 0:
        raise AnalogCalcError(
            f"{context}: a macro needs positive Y (total CTT columns) and Z (ADCs); "
            f"got Y = {cols}, Z = {adcs}."
        )
    if cols % adcs:
        raise AnalogCalcError(
            f"{context}: Z = {adcs} ADCs must DIVIDE Y = {cols} total CTT columns "
            f"({macro_label(0, cols, adcs).split(' x ', 1)[1]}). mux = Y/Z is DERIVED "
            f"({PLATFORM_MACRO_DECISION}), so a fold with a remainder would either "
            "strand columns behind no converter or buy a fractional ADC, and neither "
            "is a macro."
        )
    return cols // adcs


def is_platform_macro(rows: int, cols: int, adcs: int) -> bool:
    """True when (X, Y, Z) is inside the platform's legal box (D35-REVISED).

    The box is the two declared lists, Y >= Z, AND a fold the SHEET can price:
    ``mux = Y/Z`` has to be inside OPTIMA's ``ALLOWED_MUX_DEGREES``. That last
    clause is not a fourth rule — it is the sheet saying where it has numbers,
    and at X = 512 it cuts exactly one corner off the declared box (Y = 8X
    against Z = 8, mux 512).
    """
    rows, cols, adcs = int(rows), int(cols), int(adcs)
    if adcs <= 0 or cols % adcs:
        return False
    return (
        rows == PLATFORM_MACRO_ROWS
        and adcs in MACRO_Z_VALUES
        and cols in MACRO_Y_VALUES
        and cols >= adcs
        and (cols // adcs) in ALLOWED_CAPACITY_MUX
    )


def platform_macro_refusal(rows: int, cols: int, adcs: int, *, context: str) -> str:
    """WHY this (X, Y, Z) is outside the box, named term by term."""
    reasons = []
    if int(rows) != PLATFORM_MACRO_ROWS:
        reasons.append(
            f"X = {int(rows)} rows, and the platform fixes X = {PLATFORM_MACRO_ROWS} "
            "(the analog sheet's constrained dimension; it is a device call, not a "
            "per-workload shape)"
        )
    if int(adcs) not in MACRO_Z_VALUES:
        reasons.append(
            f"Z = {int(adcs)} ADCs is not in {list(MACRO_Z_VALUES)} (powers of two from "
            f"8 to X = {PLATFORM_MACRO_ROWS}: never more converters than rows)"
        )
    if int(cols) not in MACRO_Y_VALUES:
        reasons.append(
            f"Y = {int(cols)} total CTT columns is not in {list(MACRO_Y_VALUES)} "
            f"(powers of two from X to {MACRO_Y_CEILING_MULTIPLE}X)"
        )
    if int(cols) < int(adcs):
        reasons.append(
            f"Y = {int(cols)} < Z = {int(adcs)}: mux = Y/Z would be below 1, which is "
            "an ADC behind no column at all (mux = 1 is the legal floor)"
        )
    if int(adcs) > 0 and int(cols) % int(adcs) == 0:
        _mux = int(cols) // int(adcs)
        if _mux not in ALLOWED_CAPACITY_MUX:
            reasons.append(
                f"the DERIVED mux = Y/Z = {int(cols)}/{int(adcs)} = {_mux} is outside "
                f"{ALLOWED_CAPACITY_MUX[0]}..{ALLOWED_CAPACITY_MUX[-1]}, the range the "
                "ADC-macro sheet has points for (OPTIMA config/loader.py "
                "ALLOWED_MUX_DEGREES); past it there is no measurement and none is "
                "extrapolated"
            )
    return (
        f"{context} declares the analog macro {macro_label(rows, cols, adcs)} "
        f"(X x Y : Z), which is outside the {PLATFORM_MACRO_DECISION} platform box: "
        + "; ".join(reasons)
        + f". {PLATFORM_MACRO_QUOTE} A weight matrix whose K or N does not match the "
        "macro TILES onto it — that path is priced (ceil row/col splits, the P7.2 "
        "K-accumulator) and its remainder is reported as packing waste (D27) — so a "
        "resized macro buys nothing the tiler does not already do honestly. If this "
        f"point genuinely predates or departs from the platform, declare "
        f"{NONSTANDARD_MACRO_PATH}: '<reason>' and the geometry rides that reason as a "
        "named disclosure into every artifact."
    )


def nonstandard_macro_disclosure(
    rows: int, cols: int, adcs: int, reason: str, *, context: str
) -> str:
    """The sentence a NONSTANDARD macro rides into every artifact (D21).

    An override is not a silence: the macro that was built, the platform box it
    departs from and the reason the config gave are all named, so a reader of
    the artifact never has to open the YAML to learn that this point is
    off-platform.
    """
    mux = int(cols) // int(adcs) if int(adcs) and not int(cols) % int(adcs) else None
    mux_text = f", mux = {mux}" if mux is not None else ""
    return (
        f"{context} declares a NONSTANDARD analog macro: "
        f"{macro_label(rows, cols, adcs)} (X x Y : Z{mux_text}), against the "
        f"{PLATFORM_MACRO_DECISION} platform box X = {PLATFORM_MACRO_ROWS}, "
        f"Z in {list(MACRO_Z_VALUES)}, Y in {list(MACRO_Y_VALUES)}. "
        f"The config buys this with {NONSTANDARD_MACRO_PATH}, whose declared reason is: "
        f"{reason!r}. Every number this point reports is priced on the macro it "
        "declares, not on a platform macro, so it is NOT comparable, cell for cell "
        "or mm2 for mm2, with a platform point."
    )


def check_platform_macro(
    rows: int,
    cols: int,
    adcs: int,
    *,
    context: str,
    override_reason: Optional[str] = None,
) -> Optional[str]:
    """Refuse an off-platform macro BY NAME, or return its disclosure.

    Returns ``None`` when (X, Y, Z) is inside the platform box (the common
    case, which discloses nothing because there is nothing to disclose), the
    disclosure sentence when an override reason was declared, and raises
    :class:`AnalogCalcError` otherwise. The refusal names the offending term,
    the box and the ONE key that buys an exemption — a refusal a reader cannot
    act on is a wall, not a law.
    """
    if is_platform_macro(rows, cols, adcs):
        if override_reason is not None:
            raise AnalogCalcError(
                f"{context} declares {NONSTANDARD_MACRO_PATH} = {override_reason!r} on a "
                f"macro that IS inside the platform box "
                f"({macro_label(rows, cols, adcs)}). The key exists to buy an exemption "
                f"from {PLATFORM_MACRO_DECISION} and disclose it; on a compliant card it "
                "would ride a false disclosure into every artifact, so it is refused "
                "rather than ignored."
            )
        return None
    if override_reason is None:
        raise AnalogCalcError(platform_macro_refusal(rows, cols, adcs, context=context))
    if not isinstance(override_reason, str) or not override_reason.strip():
        raise AnalogCalcError(
            f"{context}: {NONSTANDARD_MACRO_PATH} must be a non-empty REASON string "
            f"(got {override_reason!r}). The key is an override that DISCLOSES itself; "
            "a bare true would put an unexplained off-platform macro into the artifacts, "
            f"which is the silence {PLATFORM_MACRO_DECISION} exists to end."
        )
    return nonstandard_macro_disclosure(
        rows, cols, adcs, override_reason.strip(), context=context
    )


#: The area terms this module reports, in the SOURCE's summation order.
AREA_TERM_NAMES: Tuple[str, ...] = (
    "digital_pre",
    "digital_post",
    "mam_peripherals",
    "current_mirror",
    "ctt_array",
    "adc",
)


def profile_params(name: str) -> Dict[str, Any]:
    """A deep copy of one named calc profile, or a refusal naming it."""
    if name not in OPTIMA_CALC_PROFILES:
        raise AnalogCalcError(
            f"unknown calc_profile {name!r}: the sheet-grounded profiles carried in "
            f"cim_analog_calc are {list(CALC_PROFILE_NAMES)}. A profile is a set of "
            "MEASURED ADC-macro sheet constants; an unlisted name has no constants at "
            "all and is refused rather than silently resolved to the only one there is."
        )
    return deepcopy(OPTIMA_CALC_PROFILES[name])


# ---------------------------------------------------------------------------
# THE PORT — calc_analog._compute_optima_energy_area / build_calc_config /
# calculate_array_stats, expression for expression.
# ---------------------------------------------------------------------------


def _optima_energy_area_terms(
    params: Mapping[str, Any],
    rows: int,
    cols: int,
) -> Tuple[float, float, Dict[str, float]]:
    """Port of ``calc_analog._compute_optima_energy_area``, plus the area split.

    Returns ``(energy_per_slice_pJ, area_mm2, area_terms_mm2)``. The energy and
    the area are computed by the SOURCE's expressions in the SOURCE's order;
    the term dict re-scales the same six locals the source sums, so it is a
    report of that sum and not a second model of it.
    """
    adc_muxing = float(params["adc_muxing"])
    adc_resolution_bits = float(params["adc_resolution_bits"])
    parallelism = float(params["parallelism"])
    ctts_per_8bit = float(params["ctts_per_8bit"])

    active_rows = float(rows)
    active_cols = float(cols)

    bl_on_time = float(params["adc_compute_time"]) + float(params["bl_settle_time"])

    ctt_current_per_segment = (
        float(params["peak_ctt_current"])
        * float(params["weight_probability"])
        * float(params["effective_input_probability"])
        * active_rows
        * 2.0
        / 1000.0
    )
    total_bl_current = ctt_current_per_segment * (parallelism + 2.0)
    bl_energy_per_cycle = total_bl_current * float(params["bl_vdd"]) * bl_on_time
    bl_energy_per_col = bl_energy_per_cycle * float(params["num_cycles"])
    total_bl_energy = bl_energy_per_col * active_cols * 0.001

    mirror_amp_current = float(params["mirror_current_ua"]) + float(params["amp_current_ua"])
    mirror_energy_per_cycle = mirror_amp_current * float(params["bl_vdd"]) * bl_on_time
    mirror_energy_per_col = mirror_energy_per_cycle * float(params["num_cycles"])
    column_analog_energy = mirror_energy_per_col * active_cols / 1000.0

    wl_length_per_mce = float(params.get("wl_length_per_mce", params["ctt_y_dim"]))
    total_cap_per_mce = (
        float(params["device_cap"]) + wl_length_per_mce * float(params["cap_per_f"])
    ) * adc_muxing
    wl_switch_energy_per_cycle_per_col = (
        total_cap_per_mce
        * (float(params["wl_vdd"]) ** 2)
        * active_rows
        * ctts_per_8bit
        * parallelism
        * float(params["effective_input_probability"])
    )
    wl_energy_per_col = wl_switch_energy_per_cycle_per_col * float(params["num_cycles"])
    total_wl_energy = wl_energy_per_col * float(cols) / 1000.0

    pre_energy = (
        float(params["digital_pre_ff_per_row"])
        * active_rows
        * float(params["ff_energy"])
        * float(params["num_cycles"])
        / 1000.0
    )
    post_energy = (
        float(params["digital_post_ff_per_col"])
        * active_cols
        * float(params["ff_energy"])
        * float(params["num_cycles"])
        / 1000.0
    )

    adc_energy_per_eval = float(params["adc_fom_w"]) * (2.0**adc_resolution_bits)
    adc_energy_per_col = adc_energy_per_eval * float(params["num_cycles"])
    total_adc_energy = adc_energy_per_col * active_cols / 1000.0

    # Ported GATE (the GR00T W5 profile's shift-accumulate). Absent from
    # george_cima_mux1 -> 0.0, so the OPTIMA/W8 numbers are untouched.
    serial_adder_e_fJ = float(params.get("serial_adder_energy_fJ_per_add", 0.0))
    partial_adds = parallelism * float(params["num_cycles"])
    serial_adder_energy = serial_adder_e_fJ * partial_adds * active_cols / 1000.0

    # Energy total is per ADC slice; the vector mux is applied by the caller.
    energy_total = (
        total_adc_energy
        + post_energy
        + pre_energy
        + total_wl_energy
        + column_analog_energy
        + total_bl_energy
        + serial_adder_energy
    )

    compute_cell_area = float(params["ctt_y_dim"]) * ctts_per_8bit * parallelism
    ctt_array_per_col = compute_cell_area * float(rows) * adc_muxing
    ctt_array_area = ctt_array_per_col * float(cols)

    current_mirror_area = float(params["current_mirror_area_per_col"]) * float(cols)
    adc_area = float(params["adc_area_per_col"]) * float(cols)

    wl_prg_driver_per_row = float(params["wl_prg_driver_per_row_base"]) * parallelism
    wl_inf_driver_per_row = float(params["wl_inf_driver_per_row"])

    bl_driver_per_col = (
        float(params["bl_driver_base"])
        / float(params["gate_pitch"])
        / float(params["gate_pitch"])
        * adc_muxing
    )
    sl_driver_per_col = bl_driver_per_col
    mam_peripherals = (sl_driver_per_col + bl_driver_per_col) * float(cols) + float(
        rows
    ) * (wl_prg_driver_per_row + wl_inf_driver_per_row)

    digital_post_area = (
        float(params["digital_post_ff_per_col"])
        * float(params["ff_area"])
        / 0.5
    ) * float(cols)
    digital_pre_area = (
        float(params["digital_pre_ff_per_row"])
        * float(params["ff_area"])
        / 0.5
    ) * float(rows)

    area_gate_pitch = (
        digital_pre_area
        + digital_post_area
        + mam_peripherals
        + current_mirror_area
        + ctt_array_area
        + adc_area
    )
    area_mm2 = area_gate_pitch * (float(params["gate_pitch"]) ** 2) / 1_000_000.0

    serial_adder_area_um2 = float(params.get("serial_adder_area_um2_per_col", 0.0)) * float(cols)
    area_mm2 += serial_adder_area_um2 / 1_000_000.0

    scale = (float(params["gate_pitch"]) ** 2) / 1_000_000.0
    terms = {
        "digital_pre": digital_pre_area * scale,
        "digital_post": digital_post_area * scale,
        "mam_peripherals": mam_peripherals * scale,
        "current_mirror": current_mirror_area * scale,
        "ctt_array": ctt_array_area * scale,
        "adc": adc_area * scale,
        "serial_adder": serial_adder_area_um2 / 1_000_000.0,
    }
    return energy_total, area_mm2, terms


def _compute_optima_energy_area(
    params: Mapping[str, Any],
    rows: int,
    cols: int,
) -> Tuple[float, float]:
    """The SOURCE's signature: ``(energy_per_slice_pJ, area_mm2)``."""
    energy_total, area_mm2, _terms = _optima_energy_area_terms(params, rows, cols)
    return energy_total, area_mm2


def _coerce_array_shapes(shapes: Any, default_shapes: Any) -> List[Tuple[int, int]]:
    """Port of ``calc_analog._coerce_array_shapes``."""
    if shapes is None:
        shapes = default_shapes
    if shapes is None:
        raise AnalogCalcError("array_shapes must be provided for calc mode")

    normalized: List[Tuple[int, int]] = []
    for entry in shapes:
        if isinstance(entry, dict):
            rows = entry.get("rows")
            cols = entry.get("cols")
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            rows, cols = entry
        else:
            rows = entry
            cols = entry
        if rows is None or cols is None:
            raise AnalogCalcError(f"Invalid array shape entry: {entry}")
        normalized.append((int(rows), int(cols)))
    return normalized


def build_calc_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Port of ``calc_analog.build_calc_config`` (CALC MODE ONLY).

    ``mux_factor`` (vector) and ``adc_muxing`` (capacity) are independent
    inputs, exactly as in the source: ``mux_factor`` defaults to the profile's
    ``adc_muxing`` and ``adc_muxing`` then defaults to ``mux_factor``.
    """
    cfg = dict(cfg or {})
    profile_name = cfg.get("calc_profile", "george_cima_mux1")
    profile = profile_params(profile_name)
    shapes = _coerce_array_shapes(cfg.get("array_shapes"), profile.get("default_array_shapes"))

    for _k in ("parallelism", "num_cycles"):
        if cfg.get(_k) is not None:
            profile[_k] = float(cfg[_k])

    vec_mux_factor = cfg.get("mux_factor")
    adc_muxing = cfg.get("adc_muxing")
    if vec_mux_factor is None:
        vec_mux_factor = profile["adc_muxing"]
    if adc_muxing is None:
        adc_muxing = vec_mux_factor

    vec_mux_factor = float(vec_mux_factor)
    adc_muxing = float(adc_muxing)
    if vec_mux_factor <= 0 or adc_muxing <= 0:
        raise AnalogCalcError("mux_factor and adc_muxing must be positive values.")
    profile["adc_muxing"] = adc_muxing

    slice_cycles = cfg.get("slice_cycles")
    if slice_cycles is None:
        slice_cycles = float(profile["num_cycles"])

    shots_per_output = cfg.get("shots_per_output")
    if shots_per_output is None:
        shots_per_output = profile.get("shots_per_output", 1.0)

    arrays = []
    for rows, cols in shapes:
        energy_per_vec_pJ, area_mm2, terms = _optima_energy_area_terms(profile, rows, cols)
        energy_per_vec_pJ *= float(vec_mux_factor)
        arrays.append(
            {
                "rows": rows,
                "cols": cols,
                "area_mm2": area_mm2,
                "energy_per_vec_pJ": energy_per_vec_pJ,
                "area_terms_mm2": terms,
            }
        )

    config = deepcopy(BASE_ANALOG_CONFIG)
    config.update(
        {
            "analog_clock_MHz": profile["analog_clock_MHz"],
            "analog_clock_MAX": profile["analog_clock_MAX"],
            "mux_factor": vec_mux_factor,
            "adc_muxing": adc_muxing,
            "capacity_mux_factor": adc_muxing / vec_mux_factor if vec_mux_factor else 1.0,
            "slice_cycles": slice_cycles,
            "shots_per_output": shots_per_output,
            "arrays": arrays,
        }
    )
    return config


def calculate_array_stats(
    rows: int,
    cols: Optional[int] = None,
    *,
    profile: str = "george_cima_mux1",
    mux_factor: Optional[float] = None,
    adc_muxing: Optional[float] = None,
) -> Dict[str, Any]:
    """Port of ``calc_analog.calculate_array_stats`` (the CLI's entry point).

    ``mux_factor`` / ``adc_muxing`` are keyword-only ADDITIONS: omitted, the
    call is the source's call, expression for expression. They exist because
    the source's CLI carries no mux flag while `build_calc_config` (its own
    library entry point, and the one OPTIMA's loader uses) does.
    """
    if cols is None:
        cols = rows
    config = build_calc_config(
        {
            "calc_profile": profile,
            "array_shapes": [(rows, cols)],
            "mux_factor": mux_factor,
            "adc_muxing": adc_muxing,
        }
    )
    array_spec = config["arrays"][0]
    vec_cycles = float(config["mux_factor"]) * float(config["slice_cycles"])
    return {
        "rows": rows,
        "cols": cols,
        "energy_per_vec_pJ": array_spec["energy_per_vec_pJ"],
        "area_mm2": array_spec["area_mm2"],
        "area_terms_mm2": array_spec["area_terms_mm2"],
        "mux_factor": config["mux_factor"],
        "adc_muxing": config.get("adc_muxing"),
        "capacity_mux_factor": config.get("capacity_mux_factor", 1.0),
        "slice_cycles": config["slice_cycles"],
        "shots_per_output": config["shots_per_output"],
        "vec_cycles": vec_cycles,
    }


# ---------------------------------------------------------------------------
# THE CARD DERIVATION — the one place the capacity re-split happens.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DerivedAnalogCard:
    """One analog macro's numbers, DERIVED from the sheet at (X, Y, Z).

    ``X x Y : Z`` — ``rows`` x ``cols`` total CTT columns : ``adcs``
    converters, with ``mux = cols / adcs`` DERIVED (D35-REVISED). Everything
    here is computed; nothing is declared. ``provenance`` is the line the
    artifact carries (D21: a number states where it came from).
    """

    profile: str
    rows: int
    cols: int
    adcs: int
    mux: int
    slice_cycles: int
    analog_clock_mhz: float
    energy_per_vec_pj: float
    area_mm2_per_array: float
    adc_area_mm2: float
    area_terms_mm2: Mapping[str, float]
    vec_cycles: int

    @property
    def label(self) -> str:
        """``X x Y : Z`` — how this macro is named everywhere."""
        return macro_label(self.rows, self.cols, self.adcs)

    @property
    def cells(self) -> int:
        """Weights the macro stores: ``X * Y``."""
        return int(self.rows) * int(self.cols)

    @property
    def vec_latency_s(self) -> float:
        """``vec_cycles / f_analog`` — the same law cim_timing prices with."""
        return float(self.vec_cycles) / (float(self.analog_clock_mhz) * 1e6)

    @property
    def provenance(self) -> str:
        terms = ", ".join(
            f"{name} {self.area_terms_mm2[name]:.6g}" for name in AREA_TERM_NAMES
        )
        return (
            f"analog macro {self.label} (X x Y : Z) DERIVED from calc_profile "
            f"'{self.profile}' ({CALC_SOURCE_PATH}, md5 {CALC_SOURCE_MD5}, read "
            f"{CALC_SOURCE_READ_ON}): {self.rows} rows x {self.cols} total CTT columns "
            f"behind {self.adcs} ADC(s), so mux = Y/Z = {self.mux} is DERIVED and each "
            f"converter carries {self.mux} muxed column(s) of storage. "
            f"area_mm2_per_array = {self.area_mm2_per_array:.6g} mm2 "
            f"(mm2: {terms}), energy_per_vec_pj = {self.energy_per_vec_pj:.6g} and "
            f"vec_cycles = mux {self.mux} x slice_cycles {self.slice_cycles} = "
            f"{self.vec_cycles} at {self.analog_clock_mhz:g} MHz "
            f"({self.vec_latency_s:.6g} s per vector). The VECTOR mux equals the "
            "CAPACITY mux because one muxed column behind an ADC is one slice of the "
            "output vector, which is exactly how OPTIMA's own consumer "
            "(perf_model/config/loader.py materialize_analog_config) calls this model."
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "calc_profile": self.profile,
            "source": CALC_SOURCE_PATH,
            "source_md5": CALC_SOURCE_MD5,
            "label": self.label,
            "rows": int(self.rows),
            "cols": int(self.cols),
            "adcs": int(self.adcs),
            "mux": int(self.mux),
            "cells": int(self.cells),
            "slice_cycles": int(self.slice_cycles),
            "analog_clock_mhz": float(self.analog_clock_mhz),
            "energy_per_vec_pj": float(self.energy_per_vec_pj),
            "area_mm2_per_array": float(self.area_mm2_per_array),
            "adc_area_mm2": float(self.adc_area_mm2),
            "area_terms_mm2": {
                name: float(value) for name, value in self.area_terms_mm2.items()
            },
            "vec_cycles": int(self.vec_cycles),
            "vec_latency_s": float(self.vec_latency_s),
            "provenance": self.provenance,
        }


def derive_analog_card(
    rows: int,
    cols: int,
    adcs: int,
    *,
    profile: str = "george_cima_mux1",
    context: str = "cim.cards.<analog card>",
) -> DerivedAnalogCard:
    """The macro X x Y : Z, priced from the sheet.

    THE SHAPE. ``rows`` (X) x ``cols`` (Y) total CTT columns is the CAPACITY;
    ``adcs`` (Z) is how many converters that capacity sits behind. mux = Y/Z is
    DERIVED and is what the sheet is evaluated at: the calc is called with the
    ADC COLUMN count Z as its column argument and ``adc_muxing = mux``, which
    is exactly the call OPTIMA's own loader makes. The cell count X*Y is
    invariant under the fold; the ADC bill is not, and that is the whole trade.

    THE VECTOR MUX FOLLOWS THE CAPACITY MUX (``mux_factor = adc_muxing``): the
    mux muxed columns behind one ADC are digitized one after another, so a full
    output vector costs mux slices. That is what makes latency the price of the
    area this fold saves.
    """
    params = profile_params(profile)
    rows = int(rows)
    cols = int(cols)
    adcs = int(adcs)
    if rows <= 0:
        raise AnalogCalcError(
            f"{context}: derivation needs positive rows (got X = {rows})."
        )
    mux = derived_mux(cols, adcs, context=context)
    if mux not in ALLOWED_CAPACITY_MUX:
        raise AnalogCalcError(
            f"{context}: the DERIVED mux = Y/Z = {cols}/{adcs} = {mux} is outside "
            f"{ALLOWED_CAPACITY_MUX[0]}..{ALLOWED_CAPACITY_MUX[-1]}, the range OPTIMA's "
            "loader admits (config/loader.py ALLOWED_MUX_DEGREES = range(1, 257)). "
            "Outside it the sheet has no point and none is extrapolated."
        )
    config = build_calc_config(
        {
            "calc_profile": profile,
            "array_shapes": [(rows, adcs)],
            "mux_factor": mux,
            "adc_muxing": mux,
        }
    )
    array_spec = config["arrays"][0]
    slice_cycles = float(config["slice_cycles"])
    if slice_cycles != int(slice_cycles):
        raise AnalogCalcError(
            f"{context}: calc_profile '{profile}' declares num_cycles = {slice_cycles}, "
            "which is not a whole number of analog cycles. cim.analog.slice_cycles is an "
            "integer cycle count and a fractional one is refused rather than rounded."
        )
    return DerivedAnalogCard(
        profile=profile,
        rows=rows,
        cols=cols,
        adcs=adcs,
        mux=mux,
        slice_cycles=int(slice_cycles),
        analog_clock_mhz=float(params["analog_clock_MHz"]),
        energy_per_vec_pj=float(array_spec["energy_per_vec_pJ"]),
        area_mm2_per_array=float(array_spec["area_mm2"]),
        adc_area_mm2=float(array_spec["area_terms_mm2"]["adc"]),
        area_terms_mm2=dict(array_spec["area_terms_mm2"]),
        vec_cycles=int(mux * int(slice_cycles)),
    )
