# Copyright 2026 NanoCad lab, UCLA
# https://nanocad.ee.ucla.edu/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""QIF P3.7 — the MAPPED-PATH design-space sweep (D10's "swept by the DSE").

Every candidate is evaluated through the REAL mapped path and nothing else::

    fws_mapping.build_mapping -> program.fws_build.build_fws_program
                              -> fws_eval.evaluate_fws

so each candidate's headline is read off ITS OWN timeline (ADJ-6, A1). There
is no closed form anywhere in this tool: it never calls a period law, never
composes stage tables, and never reuses one candidate's numbers for another.

WHY A SECOND DSE TOOL (and not an extension of ``tools/fws_cim_dse.py``):
``tools/fws_cim_dse.py`` is the pass-2B CLOSED-FORM sweep. Its evaluator is a
composition of ``CimDeviceModel`` laws, its report schema names ``period_us``
and the fabric-ceiling / sustained throughput PAIR that ADJ-6 retired from the
DAG report, and the validator reruns its exact sweeps as the pass-2B rows that
hold the ADJ-8 bridge. Putting a timeline evaluator inside it would either
fork it into two evaluators under one report schema — two accountings under
one set of field names, which D21 forbids — or move a frozen artifact the
bridge depends on. So the mapped sweep is its own tool with its own schema,
and the closed-form tool is untouched.

The candidate space (``mapping_dse``, the block this tool owns)
---------------------------------------------------------------
EXPLICIT CANDIDATE LISTS ONLY. There is no search, no optimizer and no
sampling: the sweep is the full cross product of the declared axis lists, in
declaration order, and every point in it is evaluated (D19; D10's automatic
allocation phase stays future work).

The block is a SIBLING of ``mapping:`` in the hardware YAML, not a key inside
it, and this tool strips it before the config parser ever sees the dict.
``mapping:`` is strict-keyed by ``config.MappingSystemConfig`` (P3.1's own
refusal machinery), so a ``dse:`` key inside it would need a schema change in
``config.py`` — a shipped parser this deliverable does not own. A tool input
therefore gets a tool-owned block, strict-keyed HERE by the same rules, and a
config carrying it stays runnable through ``run_perf`` unchanged.

.. code-block:: yaml

    mapping_dse:
      label: "Granite-4.0-H-Tiny: vector lanes x banking"
      objective: throughput          # or min_silicon
      max_chips: 0                   # 0 = unbounded
      max_silicon_mm2: 0             # 0 = unbounded
      axes:
        vector_lanes: [512, 1024, 2048, 4096]   # shared digital card (D13)
        bank_depth: [1, 2]                      # analog card banking (A3/D10)
        # column_sets_per_tile: [1]             # cim.allocation (D10)
        # arrays_per_chip: [640, 768]           # the slot budget a split needs
        # layers_per_chip: [4, 5, "auto"]       # chip split (int | list | auto)
        # shared_chiplets: [1, 10]              # ADJ-5's declared count

Axis -> the config field it moves (one field each, never two):
  ``vector_lanes``          ``cim.cards.<digital card>.vector_lanes``
  ``bank_depth``            ``cim.cards.<analog card>.bank_depth``
  ``column_sets_per_tile``  ``cim.allocation.column_sets_per_tile``
  ``arrays_per_chip``       ``cim.chip.arrays_per_chip``
  ``layers_per_chip``       ``mapping.layers_per_chip``
  ``shared_chiplets``       ``mapping.shared_chiplets``

``bank_depth`` and ``column_sets_per_tile`` both set the allocation
granularity and ``cim.allocation`` WINS (``CimDeviceModel.column_sets_per_tile``
is the one law that decides). A candidate that declares both records the
EFFECTIVE granularity and a precedence note, so no report ever implies the
card's bank was the unit when the allocation block overrode it.

Conventions (the established DSE ones, kept)
--------------------------------------------
* Every infeasible candidate is RECORDED with a stage tag and a message —
  never dropped. Stage order (how far a candidate got):
  ``config < mapping < budget < lowering < pricing < memory``.
* Pareto front: headline throughput vs total silicon over the valid
  candidates (2-axis dominance).
* Lexicographic selection per ``--objective``, printed with the pick.
* ``dse_report.md`` + ``dse_report.json`` artifacts, always written once
  evaluation has started (an all-infeasible or unrankable sweep still
  writes both and selects nothing).
* ``--emit-config`` writes the winning RUNNABLE hardware YAML (the sweep
  block stripped); ``--verify`` reruns it through ``run_perf`` and
  cross-checks the headline metrics within 0.1 %.

Runtime is what it is. Every candidate's wall time is measured and printed,
the total is printed, and NOTHING is sampled, capped or skipped to save time.

Exit codes: 0 ok; 2 usage/config error, or a sweep whose valid candidates
publish different headline keys (recorded as ``headline_conflict``, both
reports still written); 3 no feasible candidate (the best violation is
printed); 4 ``--verify`` mismatch or verify-run failure.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

import config as config_module  # noqa: E402


#: The block this tool owns, and the only key it adds to a hardware YAML.
DSE_BLOCK = "mapping_dse"

#: Axis name -> a human sentence naming the ONE config field it moves.
AXIS_TARGETS = {
    "vector_lanes": "cim.cards.<digital card>.vector_lanes",
    "bank_depth": "cim.cards.<analog card>.bank_depth",
    "column_sets_per_tile": "cim.allocation.column_sets_per_tile",
    "arrays_per_chip": "cim.chip.arrays_per_chip",
    "layers_per_chip": "mapping.layers_per_chip",
    "shared_chiplets": "mapping.shared_chiplets",
}

#: Evaluation stages in order; the index is how far a candidate got.
STAGES = ("config", "mapping", "budget", "lowering", "pricing", "memory")

#: Selection rules, printed verbatim beside the pick.
SELECTION_RULES = {
    "throughput": (
        "lexicographic: max headline tokens/s (ADJ-6, off each candidate's own "
        "timeline), then min total silicon, then min analog chips, then min "
        "placed tiles, then first in declaration order"
    ),
    "min_silicon": (
        "lexicographic: min total silicon, then max headline tokens/s (ADJ-6), "
        "then min analog chips, then min placed tiles, then first in "
        "declaration order"
    ),
}

VERIFY_REL_TOL = 1e-3  # <= 0.1 %

#: Metrics the --verify round trip cross-checks, by key suffix.
VERIFY_METRIC_SUFFIXES = (
    "tokens_per_s",
    "prefill_latency",
    "decode_step_median",
    "request_latency",
    "lowered_window_latency",
    "requests_per_s",
    "requests_per_s_lowered_window",
)


class QifDseUsageError(ValueError):
    """Tool-level configuration error (exit code 2)."""


# ---------------------------------------------------------------------------
# The sweep block: parsed and strict-keyed HERE
# ---------------------------------------------------------------------------


def _require_mapping(context, value):
    if not isinstance(value, dict):
        raise QifDseUsageError(f"{context} must be a mapping (got {value!r}).")
    return value


def _axis_value(axis, raw, context):
    """One candidate value for one axis, validated by the axis's own shape."""
    if axis == "layers_per_chip":
        if isinstance(raw, str):
            if raw.strip().lower() != "auto":
                raise QifDseUsageError(
                    f"{context} must be an integer, a list of integers or 'auto' "
                    f"(got {raw!r}); it is spelled exactly like "
                    "mapping.layers_per_chip."
                )
            return "auto"
        if isinstance(raw, (list, tuple)):
            if not raw:
                raise QifDseUsageError(f"{context} list must not be empty.")
            return [int(item) for item in raw]
        return int(raw)
    value = int(raw)
    minimum = 0 if axis == "shared_chiplets" else 1
    if value < minimum:
        raise QifDseUsageError(f"{context} must be >= {minimum} (got {value}).")
    return value


class SweepSpec:
    """The parsed ``mapping_dse`` block: axes in declaration order."""

    def __init__(self, axes, objective, max_chips, max_silicon_mm2, label):
        self.axes = axes  # OrderedDict-like: {axis: [values]} in declaration order
        self.objective = objective
        self.max_chips = int(max_chips)
        self.max_silicon_mm2 = float(max_silicon_mm2)
        self.label = label

    @property
    def points(self):
        """The full cross product, in declaration order. Nothing is skipped."""
        names = list(self.axes)
        for combo in itertools.product(*(self.axes[name] for name in names)):
            yield dict(zip(names, combo))

    @property
    def size(self):
        total = 1
        for values in self.axes.values():
            total *= len(values)
        return total

    def echo(self):
        return {
            "block": DSE_BLOCK,
            "label": self.label,
            "objective": self.objective,
            "max_chips": self.max_chips,
            "max_silicon_mm2": self.max_silicon_mm2,
            "axes": {name: list(values) for name, values in self.axes.items()},
            "axis_targets": {name: AXIS_TARGETS[name] for name in self.axes},
            "enumeration": (
                "full cross product in declaration order; explicit candidate "
                "lists only, no search and no sampling (D19, D10)"
            ),
        }

    @classmethod
    def from_raw(cls, raw_hw, source):
        block = raw_hw.get(DSE_BLOCK)
        if block is None:
            raise QifDseUsageError(
                f"{source} declares no `{DSE_BLOCK}:` block, so the mapped sweep has "
                "no candidate space. Add one beside the `mapping:` block; every axis "
                "is an EXPLICIT candidate list "
                f"({', '.join(sorted(AXIS_TARGETS))})."
            )
        block = _require_mapping(DSE_BLOCK, block)
        known = ("axes", "objective", "max_chips", "max_silicon_mm2", "label")
        unknown = [key for key in block if key not in known]
        if unknown:
            raise QifDseUsageError(
                f"{DSE_BLOCK} has unknown key(s) {unknown}; known keys are "
                f"{list(known)}. A typo is refused rather than ignored, because an "
                "ignored knob silently sweeps something else."
            )
        axes_raw = _require_mapping(f"{DSE_BLOCK}.axes", block.get("axes"))
        unknown_axes = [name for name in axes_raw if name not in AXIS_TARGETS]
        if unknown_axes:
            raise QifDseUsageError(
                f"{DSE_BLOCK}.axes has unknown axis/axes {unknown_axes}; this tool "
                f"sweeps {sorted(AXIS_TARGETS)}. Each axis moves exactly one config "
                "field: " + "; ".join(f"{k} -> {v}" for k, v in AXIS_TARGETS.items())
            )
        axes = {}
        for name, values in axes_raw.items():
            context = f"{DSE_BLOCK}.axes.{name}"
            if not isinstance(values, (list, tuple)) or not values:
                raise QifDseUsageError(
                    f"{context} must be a NON-EMPTY list of candidate values (got "
                    f"{values!r}). A bare value is refused so the file always reads "
                    "as a list of design points."
                )
            parsed = [_axis_value(name, item, f"{context}[{index}]") for index, item in enumerate(values)]
            seen = []
            for item in parsed:
                key = json.dumps(item, sort_keys=True)
                if key in seen:
                    raise QifDseUsageError(
                        f"{context} repeats the candidate value {item!r}. A repeated "
                        "point would be evaluated twice and printed as two rows of "
                        "one design point."
                    )
                seen.append(key)
            axes[name] = parsed
        if not axes:
            raise QifDseUsageError(f"{DSE_BLOCK}.axes declares no axis; the sweep is empty.")
        objective = str(block.get("objective", "throughput")).strip().lower()
        if objective not in SELECTION_RULES:
            raise QifDseUsageError(
                f"{DSE_BLOCK}.objective must be one of {sorted(SELECTION_RULES)} "
                f"(got {objective!r})."
            )
        return cls(
            axes=axes,
            objective=objective,
            max_chips=int(block.get("max_chips", 0) or 0),
            max_silicon_mm2=float(block.get("max_silicon_mm2", 0.0) or 0.0),
            label=str(block.get("label", "") or ""),
        )


# ---------------------------------------------------------------------------
# Applying one candidate's knobs to the raw hardware dict
# ---------------------------------------------------------------------------


def _load_yaml(path):
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def _hw_from_raw(raw):
    """Mirror config.parse_config for an in-memory hardware dict."""
    converted = copy.deepcopy(raw)
    config_module.convert(converted)
    return config_module.HWConfig.from_dict(converted)


def _base_without_sweep(raw_hw):
    """The hardware dict as the config parser must see it: sweep block gone."""
    raw = copy.deepcopy(raw_hw)
    raw.pop(DSE_BLOCK, None)
    return raw


def _card_names(base_hw, source):
    """(analog card name, digital card name) as the parsed library names them."""
    cim = getattr(base_hw, "cim_config", None)
    if cim is None:
        raise QifDseUsageError(f"{source} has no `cim:` block; this is not an fws_cim config.")
    cards = cim.cards
    return cards.default_analog, cards.default_digital


def apply_point(raw_base, point, analog_card, digital_card, source):
    """One candidate's raw hardware dict: the base with this point's knobs set.

    Each axis writes exactly ONE field. A knob whose host block the config
    does not declare is refused BY NAME here rather than being invented,
    because inventing a card block would sweep a device the config never
    described.
    """
    raw = copy.deepcopy(raw_base)
    cim = raw.get("cim")
    if not isinstance(cim, dict):
        raise QifDseUsageError(f"{source} has no `cim:` block; this is not an fws_cim config.")

    def _card(kind, name):
        cards = cim.get("cards")
        if not isinstance(cards, dict) or name not in cards:
            raise QifDseUsageError(
                f"the sweep moves a {kind} CARD knob, but {source} declares no "
                f"`cim.cards.{name}` block to move it in. Declare the card (P2.1, D14) "
                "or drop the axis."
            )
        return cards[name]

    for axis, value in point.items():
        if axis == "vector_lanes":
            _card("digital", digital_card)["vector_lanes"] = int(value)
        elif axis == "bank_depth":
            _card("analog", analog_card)["bank_depth"] = int(value)
        elif axis == "column_sets_per_tile":
            cim.setdefault("allocation", {})["column_sets_per_tile"] = int(value)
        elif axis == "arrays_per_chip":
            chip = cim.get("chip")
            if not isinstance(chip, dict):
                raise QifDseUsageError(
                    f"the sweep moves cim.chip.arrays_per_chip, but {source} declares "
                    "no `cim.chip` block."
                )
            chip["arrays_per_chip"] = int(value)
        elif axis in ("layers_per_chip", "shared_chiplets"):
            mapping_block = raw.get("mapping")
            if not isinstance(mapping_block, dict):
                raise QifDseUsageError(
                    f"the sweep moves mapping.{axis}, but {source} declares no "
                    "`mapping:` block. A mapped sweep needs a mapped config: the "
                    "block is what makes P3 place and P4 price the run (ADJ-5/ADJ-6)."
                )
            mapping_block[axis] = value
        else:  # pragma: no cover - AXIS_TARGETS is the gate
            raise QifDseUsageError(f"unhandled axis {axis!r}")
    return raw


# ---------------------------------------------------------------------------
# Per-candidate evaluation — the REAL mapped path, every time
# ---------------------------------------------------------------------------


def _fail(candidate, stage, message):
    candidate["ok"] = False
    candidate["fail_stage"] = stage
    candidate["fail_message"] = str(message)
    candidate["stages"][stage] = {"ok": False, "message": str(message)}
    return candidate


def _metric_by_suffix(metrics, suffix):
    """A metric's value by key suffix; the system id prefixes every key."""
    for entry in metrics:
        if entry["key"].endswith("." + suffix):
            return entry["value"]
    return None


def _silicon(mapping, chiplets):
    """This machine's silicon, term by term, with coverage named.

    The analog term is the SAME quantity the P3.6 atlas prints as
    ``enumerated_macro_silicon``: the slots the machine HAS (unowned ones
    included) x ``CimDeviceModel.macro_footprint_mm2``. It is deliberately not
    ``CimDeviceModel.total_area_mm2``, which counts the arrays a MODEL needs —
    two names, two quantities (D21).
    """
    device = mapping.device
    summary = mapping.summary()
    footprint = float(device.macro_footprint_mm2())
    slots = int(summary["analog_macro_slots"])
    analog = slots * footprint
    per_chiplet = float(device.shared_digital_area_mm2())
    digital = per_chiplet * int(chiplets)
    uncovered = []
    if footprint <= 0:
        uncovered.append("analog macro footprint (cim.analog.area_mm2_per_array = 0)")
    if per_chiplet <= 0:
        uncovered.append("shared digital chiplet area (the card declares no area_mm2)")
    return {
        "analog_macro_slots": slots,
        "macro_footprint_mm2": footprint,
        "analog_macro_silicon_mm2": analog,
        "shared_digital_area_mm2_per_chiplet": per_chiplet,
        "shared_digital_silicon_mm2": digital,
        "total_silicon_mm2": analog + digital,
        "uncovered_terms": uncovered,
        "basis": (
            f"{slots} enumerated analog macro slots x macro_footprint_mm2 "
            f"{footprint:.6g} + {chiplets} shared digital chiplets x the card's "
            f"declared area_mm2 {per_chiplet:.6g}. The analog term is the P3.6 "
            "atlas's enumerated_macro_silicon (slots the machine HAS), NOT "
            "CimDeviceModel.total_area_mm2 (arrays a model NEEDS)."
            + (
                " UNCOVERED (absent law, not a measured zero): " + "; ".join(uncovered)
                if uncovered
                else ""
            )
        ),
    }


def _extend_disclosures(candidate, items):
    """Add disclosures to a candidate, one entry per constraint key.

    The evaluator's list and the mapping's own relaxations name several of
    the same constraints (the evaluator reads them off this mapping), so the
    two are merged here rather than concatenated: one constraint, one row.
    The FIRST entry wins, and a duplicate whose VALUE agrees is simply the
    same claim in other words. A duplicate whose value DISAGREES is a
    contradiction under one name, so it lands in the candidate's notes
    instead of being dropped silently (D21).
    """
    existing = {}
    for item in candidate["disclosures"]:
        existing.setdefault(item["constraint"], item)
    for item in items:
        seen = existing.get(item["constraint"])
        if seen is None:
            existing[item["constraint"]] = item
            candidate["disclosures"].append(item)
            continue
        if item["value"] != seen["value"]:
            candidate["notes"].append(
                f"constraint {item['constraint']!r} was disclosed TWICE with "
                f"different values on this candidate ({seen['value']!r} vs "
                f"{item['value']!r}); the first is the one reported."
            )


def evaluate_candidate(cand_id, raw_base, point, model_config, spec, analog_card,
                       digital_card, source, model_id=None):
    """One candidate, evaluated end to end on the REAL mapped path.

    Returns the candidate record. Evaluation stops at the FIRST stage that
    refuses, and the stage tag plus the refusal's own message are recorded —
    nothing is dropped and no message is rewritten.
    """
    import fws_eval
    import fws_mapping
    from program.fws_build import build_fws_program

    started = time.time()
    candidate = {
        "id": cand_id,
        "knobs": dict(point),
        "ok": True,
        "fail_stage": None,
        "fail_message": None,
        "stages": {},
        "metrics": {},
        "notes": [],
    }

    # Stage: config — the candidate's own YAML, parsed and validated exactly
    # as run_perf would parse it (card admissibility lives here: a bank_depth
    # that does not divide the card's mux is refused by the card, by name).
    # apply_point is NOT inside the try: a knob whose host block the config
    # never declared is a SWEEP-SPEC error (it is wrong for every candidate
    # identically), not one candidate's infeasibility, so it propagates as a
    # usage error instead of printing as N identical failed rows.
    raw = apply_point(raw_base, point, analog_card, digital_card, source)
    try:
        hw = _hw_from_raw(raw)
        config_module.validate_hw_config(hw)
        config_module.validate_model_config(hw, model_config)
    except (ValueError, KeyError, TypeError) as exc:
        candidate["wall_s"] = time.time() - started
        return _fail(candidate, "config", exc), None
    candidate["stages"]["config"] = {"ok": True}

    # Stage: mapping — P3 places the tiles. Every refusal here is a named
    # MappingError carrying its own stage word (residency / assembly / ...).
    try:
        mapping = fws_mapping.build_mapping(
            hw, model_config, model_id=model_id
        )
    except ValueError as exc:
        candidate["wall_s"] = time.time() - started
        return _fail(candidate, "mapping", exc), None
    candidate["stages"]["mapping"] = {"ok": True}

    summary = mapping.summary()
    chiplets = int(summary["shared_digital_chiplets"])
    silicon = _silicon(mapping, chiplets)
    candidate["placement"] = {
        "analog_chips": int(summary["analog_chips"]),
        "shared_digital_chiplets": chiplets,
        "shared_digital_suggestion": int(summary["shared_digital_suggestion"]),
        "analog_macro_slots": int(summary["analog_macro_slots"]),
        "macros_holding_tiles": int(summary["macros_holding_tiles"]),
        "unowned_macro_slots": int(summary["unowned_macro_slots"]),
        "unowned_columns": int(summary["unowned_columns"]),
        "tiles": int(summary["tiles"]),
        "owners": int(summary["owners"]),
        "devices": int(summary["devices"]),
        "decode_window": summary["decode_window"],
    }
    candidate["silicon"] = silicon
    effective_bank = int(mapping.device.column_sets_per_tile)
    candidate["effective_column_sets_per_tile"] = effective_bank
    if "bank_depth" in point and "column_sets_per_tile" in point:
        candidate["notes"].append(
            "both bank_depth and column_sets_per_tile are swept: cim.allocation "
            f"WINS, so the effective allocation granularity is {effective_bank} mux "
            "slot(s) (CimDeviceModel.column_sets_per_tile is the one law that decides)."
        )

    # Stage: budget — the DECLARED caps, checked against the placement.
    if spec.max_chips > 0 and candidate["placement"]["analog_chips"] > spec.max_chips:
        candidate["wall_s"] = time.time() - started
        return _fail(
            candidate,
            "budget",
            f"placement needs {candidate['placement']['analog_chips']} analog chips but "
            f"{DSE_BLOCK}.max_chips = {spec.max_chips}.",
        ), mapping
    if spec.max_silicon_mm2 > 0 and silicon["total_silicon_mm2"] > spec.max_silicon_mm2:
        candidate["wall_s"] = time.time() - started
        return _fail(
            candidate,
            "budget",
            f"placement needs {silicon['total_silicon_mm2']:.6g} mm2 of silicon but "
            f"{DSE_BLOCK}.max_silicon_mm2 = {spec.max_silicon_mm2:.6g}.",
        ), mapping
    candidate["stages"]["budget"] = {"ok": True}

    # Stage: lowering — P3.2 builds the placed, annotated, UNPRICED DAG.
    try:
        program = build_fws_program(mapping)
    except ValueError as exc:
        candidate["wall_s"] = time.time() - started
        return _fail(candidate, "lowering", exc), mapping
    candidate["stages"]["lowering"] = {"ok": True}

    # Stage: pricing — P4 prices that DAG and evaluates ONE timeline.
    try:
        evaluation = fws_eval.evaluate_fws(program)
    except ValueError as exc:
        candidate["wall_s"] = time.time() - started
        return _fail(candidate, "pricing", exc), mapping
    candidate["stages"]["pricing"] = {"ok": True}

    metrics = [dict(entry.as_dict()) for entry in evaluation.metrics
               if not isinstance(entry.value, (list, tuple))]
    candidate["metrics"] = {
        "ops_priced": len(evaluation.pricing.costs),
        "makespan_s": float(evaluation.makespan_s),
        "entries": metrics,
        "tokens_per_s": _metric_by_suffix(metrics, "tokens_per_s"),
        "prefill_latency_s": _metric_by_suffix(metrics, "prefill_latency"),
        "decode_step_median_s": _metric_by_suffix(metrics, "decode_step_median"),
        "request_latency_s": (
            _metric_by_suffix(metrics, "request_latency")
            if _metric_by_suffix(metrics, "request_latency") is not None
            else _metric_by_suffix(metrics, "lowered_window_latency")
        ),
        "requests_per_s": (
            _metric_by_suffix(metrics, "requests_per_s")
            if _metric_by_suffix(metrics, "requests_per_s") is not None
            else _metric_by_suffix(metrics, "requests_per_s_lowered_window")
        ),
        "total_energy_pj": float(evaluation.total_energy_pj),
        "energy_components": [
            {
                "component": item.key,
                "energy_pj": float(item.energy_pj),
                "coverage": item.coverage,
            }
            for item in evaluation.energy
        ],
        "extrapolation": dict(evaluation.extrapolation),
        "headline_basis": (
            "ADJ-6: tokens/s at the decode terminal, read off THIS candidate's own "
            "timeline (fws_eval.evaluate_fws over program.fws_build's placed DAG). "
            "No closed form and no period is involved anywhere in this row."
        ),
    }
    candidate["disclosures"] = []
    _extend_disclosures(
        candidate,
        (
            {"constraint": item.constraint, "value": item.value, "reason": item.reason}
            for item in evaluation.disclosures
        ),
    )

    # Stage: memory — P4.3's feasibility verdicts on the same timeline.
    violated = [v for v in evaluation.memory if v.status == "VIOLATED"]
    undeclared = [v for v in evaluation.memory if v.status == "undeclared"]
    candidate["memory"] = {
        "verdicts": len(evaluation.memory),
        "violated": len(violated),
        "undeclared": len(undeclared),
        "worst_ratio": max(
            (
                float(v.high_water_bytes) / float(v.capacity_bytes)
                for v in evaluation.memory
                if v.capacity_bytes
            ),
            default=None,
        ),
    }
    if violated:
        worst = max(violated, key=lambda v: float(v.high_water_bytes))
        candidate["wall_s"] = time.time() - started
        return _fail(
            candidate,
            "memory",
            f"{len(violated)} memory verdict(s) VIOLATED on this candidate's timeline; "
            f"worst is {worst.scope} / {worst.tier} at "
            f"{worst.high_water_bytes / 1024 ** 3:.3f} GiB against a declared "
            f"{worst.capacity_bytes / 1024 ** 3:.3f} GiB ({worst.owner}).",
        ), mapping
    candidate["stages"]["memory"] = {"ok": True}
    if undeclared:
        candidate["notes"].append(
            f"{len(undeclared)} memory verdict(s) are 'undeclared' (the capacity field "
            "is absent). That is NOT a pass — the check could not be made (D21)."
        )

    # P3.5's boundary census rides the row as bytes only: a RATE needs a time
    # base and this tool does not invent one for it.
    rows = mapping.boundary_table()
    candidate["boundaries"] = [
        {
            "boundary_id": row.boundary_id,
            "role": row.role,
            "bytes_per_unit": float(row.bytes_per_unit),
            "unit": row.unit,
            "required_time_s": float(row.required_time_s),
            "crosses_link": bool(row.crosses_link),
        }
        for row in rows
    ]
    # The mapping's own relaxations join the evaluator's list DEDUPLICATED by
    # constraint key: both lists carry the same constraints (the evaluator
    # reads several of them from this mapping), and printing one constraint
    # twice under one candidate reads as two findings.
    _extend_disclosures(
        candidate,
        (
            {"constraint": item.constraint, "value": item.value, "reason": item.reason}
            for item in mapping.relaxations(rows)
        ),
    )
    candidate["wall_s"] = time.time() - started
    return candidate, mapping


# ---------------------------------------------------------------------------
# Front and selection
# ---------------------------------------------------------------------------


#: The headline metric, in preference order. ADJ-6 makes tokens/s at the
#: decode terminal THE headline, and a decode sweep always ranks on it. A
#: prefill-only workload (a ViT) publishes no tokens/s at all, so the sweep
#: ranks on the request rate off the same timeline — one key per sweep,
#: NAMED in the report, never two keys mixed inside one ranking.
HEADLINE_KEYS = ("tokens_per_s", "requests_per_s")


def headline_key_of(candidate):
    """The name of the headline metric this candidate published, or None."""
    for key in HEADLINE_KEYS:
        if candidate["metrics"].get(key) is not None:
            return key
    return None


def _headline(candidate):
    key = headline_key_of(candidate)
    return float("-inf") if key is None else float(candidate["metrics"][key])


def headline_text(candidate):
    """`"6110.95 tokens/s"` — the value AND the name of the key it came from."""
    key = headline_key_of(candidate)
    if key is None:
        return "- (no headline metric on this timeline)"
    unit = "tokens/s" if key == "tokens_per_s" else "requests/s"
    return f"{_fmt(candidate['metrics'][key])} {unit}"


def _silicon_of(candidate):
    return float(candidate["silicon"]["total_silicon_mm2"])


def pareto_front_ids(valid):
    """Non-dominated ids under (max headline tokens/s, min total silicon)."""
    front = []
    for cand in valid:
        thr, area = _headline(cand), _silicon_of(cand)
        dominated = False
        for other in valid:
            if other is cand:
                continue
            othr, oarea = _headline(other), _silicon_of(other)
            if othr >= thr and oarea <= area and (othr > thr or oarea < area):
                dominated = True
                break
        if not dominated:
            front.append(cand["id"])
    return front


def front_shape(valid, front_ids, axes=()):
    """What the front's SHAPE says, read off the candidates themselves.

    A one-point front is a real answer, not a broken sweep: it means no
    swept axis trades silicon for speed. Saying which of the two it is
    keeps a reader from mistaking a flat axis for a missing one.

    Every clause of the returned note is DERIVED from the candidates in
    hand — the swept axis names, the two counts the silicon accounting
    multiplies, and the accounting's own uncovered terms. Nothing here
    asserts a mechanism the payload does not carry (D21): a note that
    named the cause of a flat front from a constant string would keep
    saying it on a sweep where the cause had moved.
    """
    if not valid:
        return "empty", "no valid candidate, so there is no front."
    areas = {round(_silicon_of(c), 9) for c in valid}
    if len(front_ids) > 1:
        # Two candidates that tie on BOTH axes are both non-dominated, which
        # is not a trade-off: it is a pair of knobs that moved neither number.
        # Calling that a "spread" would sell a tie as a design choice.
        corners = {
            (round(_headline(c), 12), round(_silicon_of(c), 12))
            for c in valid
            if c["id"] in front_ids
        }
        if len(corners) == 1:
            return "tied", (
                f"{len(front_ids)} candidates are INDISTINGUISHABLE on both axes: "
                "they differ in knobs that moved neither the headline nor the "
                "silicon, so nothing here is a trade and the lexicographic "
                "tie-break decides among them."
            )
        return "spread", (
            f"{len(front_ids)} non-dominated points over {len(corners)} distinct "
            "(throughput, silicon) corners: the sweep contains a real "
            "silicon-for-throughput trade."
        )
    if len(areas) == 1:
        swept = ", ".join(axes) if axes else "the swept axes"
        parts = [
            "every valid candidate carries the SAME total silicon "
            f"({_silicon_of(valid[0]):.6g} mm2), so the front collapses to the "
            "fastest one."
        ]
        slots = {c["silicon"].get("analog_macro_slots") for c in valid}
        chiplets = {c["placement"].get("shared_digital_chiplets") for c in valid}
        if len(slots) == 1 and len(chiplets) == 1 and None not in slots | chiplets:
            parts.append(
                "Both terms of the silicon accounting are CONSTANT over this sweep: "
                f"the ENUMERATED analog macro slot count is {slots.pop()} and the "
                f"shared digital chiplet count is {chiplets.pop()} on every "
                f"candidate, so no axis on this sweep ({swept}) moved a term of "
                "the accounting. The axes move TIME only."
            )
        else:
            parts.append(
                f"The axes on this sweep ({swept}) move the accounting's terms "
                "against each other and they sum to the same total, so none of "
                "them buys speed with area."
            )
        uncovered = []
        for cand in valid:
            for term in cand["silicon"].get("uncovered_terms", ()):
                if term not in uncovered:
                    uncovered.append(term)
        if uncovered:
            parts.append(
                "UNCOVERED in that accounting (absent law, not a measured zero): "
                + "; ".join(uncovered)
                + "."
            )
        return "flat_area", " ".join(parts)
    return "dominated_chain", (
        "one non-dominated point although the candidates differ in silicon: every "
        "other candidate is worse on BOTH axes, so no swept axis trades silicon "
        "for speed here."
    )


def _selection_key(candidate, objective, order):
    index = order[candidate["id"]]
    if objective == "min_silicon":
        return (
            _silicon_of(candidate),
            -_headline(candidate),
            candidate["placement"]["analog_chips"],
            candidate["placement"]["tiles"],
            index,
        )
    return (
        -_headline(candidate),
        _silicon_of(candidate),
        candidate["placement"]["analog_chips"],
        candidate["placement"]["tiles"],
        index,
    )


def best_violation(candidates):
    """The failure of the candidate that got FURTHEST (highest stage index)."""
    failed = [c for c in candidates if not c["ok"]]
    if not failed:
        return None
    order = {stage: index for index, stage in enumerate(STAGES)}
    best = max(failed, key=lambda c: order.get(c["fail_stage"], -1))
    return {
        "id": best["id"],
        "knobs": best["knobs"],
        "fail_stage": best["fail_stage"],
        "fail_message": best["fail_message"],
    }


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def _json_safe(obj):
    if isinstance(obj, dict):
        return {key: _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(value) for value in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    return obj


def _fmt(value, spec=".6g"):
    if value is None:
        return "-"
    if isinstance(value, float):
        return format(value, spec)
    return str(value)


def _knob_cell(value):
    if isinstance(value, list):
        return "[" + ",".join(str(item) for item in value) + "]"
    return str(value)


def render_markdown(payload):
    axes = list(payload["sweep"]["axes"])
    headline_key = payload.get("headline_metric") or "tokens_per_s"
    headline_unit = "tokens/s" if headline_key == "tokens_per_s" else "requests/s"
    lines = []
    lines.append("# QIF P3.7 — mapped-path DSE report")
    lines.append("")
    if payload["sweep"]["label"]:
        lines.append(f"**{payload['sweep']['label']}**")
        lines.append("")
    lines.append(f"- hardware config: `{payload['hardware_config']}`")
    lines.append(f"- model config: `{payload['model_config']}` (model id `{payload['model_id']}`)")
    lines.append(f"- objective: {payload['objective']}")
    lines.append(f"- selection rule: {payload['selection_rule']}")
    lines.append(f"- evaluation: {payload['evaluation_path']}")
    lines.append(
        f"- candidates: {payload['num_candidates']} "
        f"({payload['num_valid']} valid); every one evaluated, nothing sampled"
    )
    lines.append("")

    lines.append("## Selected design point")
    lines.append("")
    selected = payload["selected"]
    if selected is None and payload.get("headline_conflict"):
        lines.append(f"NONE — the sweep is UNRANKABLE: {payload['headline_conflict']}")
        lines.append("")
        lines.append(
            "Every candidate below was still evaluated on the real mapped path; "
            "only the ranking is refused."
        )
    elif selected is None:
        lines.append("NONE — every candidate is infeasible.")
        best = payload.get("best_violation")
        if best is not None:
            lines.append("")
            lines.append(
                f"Best violation (candidate {best['id']}, stage `{best['fail_stage']}`): "
                f"{best['fail_message']}"
            )
    else:
        knobs = ", ".join(f"{axis} = {_knob_cell(selected['knobs'][axis])}" for axis in axes)
        metrics = selected["metrics"]
        place = selected["placement"]
        silicon = selected["silicon"]
        lines.append(f"- candidate `{selected['id']}`: {knobs}")
        lines.append(
            f"- HEADLINE {_fmt(metrics[headline_key])} {headline_unit}"
            + (
                " at the decode terminal (ADJ-6)"
                if headline_key == "tokens_per_s"
                else " over the observed span (this workload lowers no decode step, "
                "so ADJ-6's tokens/s does not exist on its timeline)"
            )
            + ", off this candidate's own timeline"
        )
        lines.append(
            f"- prefill latency {_fmt(metrics['prefill_latency_s'])} s; median decode "
            f"step {_fmt(metrics['decode_step_median_s'])} s; "
            f"{_fmt(metrics['requests_per_s'])} requests/s"
        )
        lines.append(
            f"- {place['analog_chips']} analog chips + {place['shared_digital_chiplets']} "
            f"shared digital chiplets; {place['tiles']} tiles on "
            f"{place['macros_holding_tiles']} of {place['analog_macro_slots']} macro slots"
        )
        lines.append(
            f"- total silicon {_fmt(silicon['total_silicon_mm2'])} mm2 "
            f"({_fmt(silicon['analog_macro_silicon_mm2'])} analog + "
            f"{_fmt(silicon['shared_digital_silicon_mm2'])} shared digital)"
        )
        lines.append(f"- silicon basis: {silicon['basis']}")
        lines.append(f"- energy {_fmt(metrics['total_energy_pj'])} pJ over the timeline")
        lines.append(f"- wall time to evaluate this candidate: {_fmt(selected['wall_s'], '.3g')} s")
    lines.append("")

    lines.append("## Candidates")
    lines.append("")
    header = ["id"] + axes + [
        "ok", "fail stage", headline_unit, "silicon mm2", "chips", "tiles",
        "decode step s", "energy pJ", "wall s",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for cand in payload["candidates"]:
        metrics = cand.get("metrics") or {}
        place = cand.get("placement") or {}
        silicon = cand.get("silicon") or {}
        row = [cand["id"]] + [_knob_cell(cand["knobs"][axis]) for axis in axes] + [
            "yes" if cand["ok"] else "NO",
            cand["fail_stage"] or "-",
            _fmt(metrics.get(headline_key)),
            _fmt(silicon.get("total_silicon_mm2")),
            _fmt(place.get("analog_chips")),
            _fmt(place.get("tiles")),
            _fmt(metrics.get("decode_step_median_s")),
            _fmt(metrics.get("total_energy_pj"), ".4g"),
            _fmt(cand.get("wall_s"), ".3g"),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append(f"## Pareto front (headline {headline_unit} vs total silicon)")
    lines.append("")
    for cid in payload["front_ids"]:
        cand = next(c for c in payload["candidates"] if c["id"] == cid)
        lines.append(
            f"- `{cid}`: {_fmt(cand['metrics'][headline_key])} {headline_unit} at "
            f"{_fmt(cand['silicon']['total_silicon_mm2'])} mm2"
        )
    if not payload["front_ids"]:
        lines.append("(empty — no valid candidate)")
    lines.append("")
    lines.append(f"Front shape — **{payload['front_shape']}**: {payload['front_note']}")
    lines.append("")

    lines.append("## Infeasible candidates")
    lines.append("")
    failures = [c for c in payload["candidates"] if not c["ok"]]
    if failures:
        for cand in failures:
            lines.append(f"- `{cand['id']}` [{cand['fail_stage']}]: {cand['fail_message']}")
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("## Disclosures")
    lines.append("")
    # The silicon accounting's coverage rides EVERY report, whatever the
    # front's shape: a reader of the MD must be able to tell a 0 mm2 term
    # that no law covers from a measured zero (D21).
    coverage = payload.get("silicon_coverage") or {}
    if coverage:
        lines.append(f"- **silicon_accounting** — {coverage['basis']}")
        if coverage.get("uncovered_terms"):
            lines.append(
                "- **silicon_uncovered** — "
                + "; ".join(coverage["uncovered_terms"])
                + ": an ABSENT law, not a measured zero. Those terms contribute "
                "0 mm2 to the front's silicon axis and the ranking is optimistic "
                "by whatever they would have cost."
            )
    if payload["disclosures"]:
        for item in payload["disclosures"]:
            carried = item.get("candidates") or []
            where = ""
            if carried:
                where = (
                    " [all candidates]"
                    if len(carried) == payload["num_valid"]
                    else " [candidates " + ", ".join(carried) + "]"
                )
            lines.append(
                f"- **{item['constraint']}** — {item['value']}: {item['reason']}{where}"
            )
            for variant in item.get("varies_by_candidate", ()):
                lines.append(
                    f"  - candidate `{variant['id']}` carries a DIFFERENT value for "
                    f"this constraint — {variant['value']}: {variant['reason']}"
                )
    elif not coverage:
        lines.append("(none)")
    lines.append("")

    verify = payload.get("verify")
    if verify is not None:
        lines.append("## Verify round trip (`--verify`)")
        lines.append("")
        lines.append(f"- emitted config: `{verify['emitted_config']}`")
        lines.append(f"- run_perf report: `{verify.get('report_path') or '-'}`")
        lines.append(f"- verdict: {'PASS' if verify['pass'] else 'FAIL'} (tolerance 0.1 %)")
        for check in verify["checks"]:
            lines.append(
                f"  - {check['name']}: sweep {_fmt(check['dse'], '.9g')} vs run_perf "
                f"{_fmt(check['simulator'], '.9g')} (rel {check['rel_err']:.3e}) "
                f"{'ok' if check['ok'] else 'MISMATCH'}"
            )
        if verify.get("message"):
            lines.append(f"- {verify['message']}")
        lines.append("")

    lines.append("## Sweep block echo (`mapping_dse`)")
    lines.append("")
    lines.append("```yaml")
    lines.append(yaml.safe_dump(payload["sweep"], sort_keys=False).rstrip())
    lines.append("```")
    lines.append("")
    lines.append(
        f"Total wall time over {payload['num_candidates']} candidates: "
        f"{_fmt(payload['total_wall_s'], '.4g')} s. Wall times are properties of the "
        "machine that ran the sweep, not of the design points."
    )
    lines.append("")
    return "\n".join(lines)


EMITTED_HEADER = """\
# EMITTED BY tools/fws_qif_dse.py — the winning point of a mapped-path sweep
# (QIF P3.7). This is a COMPLETE RUNNABLE fws_cim hardware config: the base
# config of the sweep with the selected candidate's knobs written in and the
# `{block}` sweep block removed (a runnable config declares a machine, not a
# candidate space).
#
# sweep base:  {base}
# model:       {model}
# candidate:   {cand} ({knobs})
# headline:    {headline}, measured on this point's own timeline by fws_eval
#              (ADJ-6: tokens/s at the decode terminal when the run decodes).
"""


def emit_selected_config(raw_base, selected, out_path, *, base_path, model_path,
                         analog_card, digital_card, source):
    """Write the winning point as a complete, runnable hardware YAML."""
    raw = apply_point(raw_base, selected["knobs"], analog_card, digital_card, source)
    raw.pop(DSE_BLOCK, None)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = EMITTED_HEADER.format(
        block=DSE_BLOCK,
        base=base_path,
        model=model_path,
        cand=selected["id"],
        knobs=", ".join(f"{k} = {_knob_cell(v)}" for k, v in selected["knobs"].items()),
        headline=headline_text(selected),
    )
    out_path.write_text(header + yaml.safe_dump(raw, sort_keys=False))
    return out_path


def verify_selected(emitted_path, model_path, mode, output_dir, selected):
    """Rerun the emitted config through run_perf and cross-check the headline.

    The subprocess runs with cwd = the sweep's output dir, so run_perf's
    ``output/<MODE>`` tree lands inside the sweep artifacts and never touches
    the repo's own output directory.

    Both sides run the SAME evaluator, so a mismatch means the EMITTED CONFIG
    does not reproduce the point the sweep selected — which is exactly what
    this round trip exists to catch.
    """
    result = {
        "ran": True,
        "emitted_config": str(emitted_path),
        "report_path": None,
        "pass": False,
        "checks": [],
        "message": None,
    }
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "run_perf.py"),
            "--hardware_config",
            str(Path(emitted_path).resolve()),
            "--model_config",
            str(Path(model_path).resolve()),
        ],
        cwd=str(output_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        timeout=3600,
    )
    if proc.returncode != 0:
        result["message"] = (
            f"run_perf failed with exit code {proc.returncode}; last output:\n"
            + proc.stdout[-4000:]
        )
        return result
    report_path = Path(output_dir) / "output" / mode / "fws_qif_report.json"
    if not report_path.exists():
        result["message"] = (
            f"run_perf wrote no mapped-DAG report at {report_path}. A mapped run "
            "writes fws_qif_report.json; a config whose `mapping:` block went "
            "missing would write the legacy closed-form report instead."
        )
        return result
    # Recorded RELATIVE to the sweep's output dir: the absolute path is a
    # property of the machine that ran the sweep, and a checked-in artifact
    # that carried one would differ on every machine that regenerated it.
    result["report_path"] = str(report_path.relative_to(Path(output_dir)))
    report = json.loads(report_path.read_text())
    sim_metrics = report["evaluation"]["metrics"]

    checks = []
    for suffix in VERIFY_METRIC_SUFFIXES:
        mine = _metric_by_suffix(selected["metrics"]["entries"], suffix)
        theirs = _metric_by_suffix(sim_metrics, suffix)
        if mine is None and theirs is None:
            continue
        checks.append((suffix, mine, theirs))
    # Discrete placement quantities must be EXACT: they are counts, not times.
    for name, mine, theirs in (
        ("analog_chips", selected["placement"]["analog_chips"], report["mapping"]["analog_chips"]),
        ("tiles", selected["placement"]["tiles"], report["mapping"]["tiles"]),
        (
            "analog_macro_slots",
            selected["placement"]["analog_macro_slots"],
            report["mapping"]["analog_macro_slots"],
        ),
    ):
        result["checks"].append(
            {
                "name": name,
                "dse": mine,
                "simulator": theirs,
                "rel_err": 0.0 if mine == theirs else float("inf"),
                "ok": mine == theirs,
                "tolerance": "exact (a count)",
            }
        )

    for name, mine, theirs in checks:
        if mine is None or theirs is None:
            result["checks"].append(
                {
                    "name": name,
                    "dse": mine,
                    "simulator": theirs,
                    "rel_err": float("inf"),
                    "ok": False,
                    "tolerance": "<= 0.1 %",
                }
            )
            continue
        rel = (
            abs(float(mine) - float(theirs)) / abs(float(theirs))
            if float(theirs) != 0.0
            else (0.0 if float(mine) == 0.0 else float("inf"))
        )
        result["checks"].append(
            {
                "name": name,
                "dse": float(mine),
                "simulator": float(theirs),
                "rel_err": rel,
                "ok": rel <= VERIFY_REL_TOL,
                "tolerance": "<= 0.1 %",
            }
        )

    result["pass"] = all(check["ok"] for check in result["checks"])
    if not result["pass"]:
        result["message"] = (
            "the emitted config does not reproduce the selected point: "
            + "; ".join(
                f"{c['name']}: sweep {c['dse']!r} vs run_perf {c['simulator']!r} "
                f"(rel {c['rel_err']:.3e})"
                for c in result["checks"]
                if not c["ok"]
            )
        )
    return result


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


def run_sweep(hardware_config, model_config_path, *, mode="LLM", model_id=None,
              output_dir=None, objective=None, emit_config=None, verify=False,
              quiet=False):
    """Run the mapped sweep; return ``(exit_code, payload)``.

    Artifacts are ALWAYS written once the sweep has started evaluating: an
    all-infeasible sweep (its stage-tagged refusals ARE the result) and an
    unrankable one (candidates that publish different headline keys) both
    write both reports and select nothing. Only a refusal BEFORE any
    candidate is priced — a malformed sweep block, an unknown axis, a
    missing card — raises, and it raises with nothing evaluated to lose.
    """
    hw_path = Path(hardware_config)
    model_path = Path(model_config_path)
    raw_hw = _load_yaml(hw_path)
    if not isinstance(raw_hw, dict):
        raise QifDseUsageError(f"{hw_path} is not a YAML mapping.")
    spec = SweepSpec.from_raw(raw_hw, str(hw_path))
    if objective is not None:
        objective = str(objective).strip().lower()
        if objective not in SELECTION_RULES:
            raise QifDseUsageError(
                f"unknown --objective {objective!r}; use {' or '.join(sorted(SELECTION_RULES))}."
            )
        spec.objective = objective

    raw_base = _base_without_sweep(raw_hw)
    base_hw = _hw_from_raw(raw_base)
    if str(getattr(base_hw, "device_class", "gpu")).lower() != "fws_cim":
        raise QifDseUsageError(f"{hw_path} must set device_class: fws_cim.")
    if getattr(base_hw, "mapping_config", None) is None:
        raise QifDseUsageError(
            f"{hw_path} declares no `mapping:` block, so it is not a MAPPED config and "
            "this sweep would have nothing to sweep. P3.7 evaluates every candidate "
            "through fws_mapping -> fws_build -> fws_eval (ADJ-6 / A1); the pass-1/2 "
            "closed-form sweep is tools/fws_cim_dse.py."
        )
    analog_card, digital_card = _card_names(base_hw, str(hw_path))
    model_config = config_module.parse_config(str(model_path), mode)

    if output_dir is None:
        output_dir = REPO_ROOT / "output" / "fws_qif_dse" / f"{hw_path.stem}__{model_path.stem}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print(
            f"[QIF DSE] {spec.size} candidates over "
            + " x ".join(f"{axis}{spec.axes[axis]}" for axis in spec.axes)
            + "; every one is evaluated through the real mapped path "
            "(fws_mapping -> fws_build -> fws_eval). Nothing is sampled."
        )

    candidates = []
    started = time.time()
    for index, point in enumerate(spec.points):
        cand_id = f"c{index:03d}"
        candidate, _ = evaluate_candidate(
            cand_id,
            raw_base,
            point,
            model_config,
            spec,
            analog_card,
            digital_card,
            str(hw_path),
            model_id=model_id,
        )
        candidates.append(candidate)
        if not quiet:
            knobs = ", ".join(f"{k}={_knob_cell(v)}" for k, v in point.items())
            if candidate["ok"]:
                print(
                    f"[QIF DSE] {cand_id} ({knobs}): "
                    f"{headline_text(candidate)}, "
                    f"{_fmt(candidate['silicon']['total_silicon_mm2'])} mm2, "
                    f"{candidate['placement']['analog_chips']} chips "
                    f"[{candidate['wall_s']:.1f} s]"
                )
            else:
                print(
                    f"[QIF DSE] {cand_id} ({knobs}): INFEASIBLE at stage "
                    f"{candidate['fail_stage']} [{candidate['wall_s']:.1f} s]"
                )
    total_wall = time.time() - started

    valid = [c for c in candidates if c["ok"]]
    # One ranking key for the whole sweep, named in the report. Candidates of
    # one sweep share a workload, so they publish the same headline; if they
    # ever did not, ranking them against each other would be two accountings
    # in one ordering (D21) and the tool refuses instead of picking one.
    headline_keys = {headline_key_of(cand) for cand in valid}
    # A ranking conflict is RECORDED, never raised: every candidate above has
    # already been priced on the real mapped path, and throwing the run away
    # after that work would leave a long sweep with nothing on disk. The
    # payload names the conflict, selects nothing, still writes both reports
    # and the process exits 2 — the same shape the all-infeasible path uses.
    headline_conflict = None
    if len(headline_keys) > 1:
        headline_conflict = (
            "the valid candidates publish DIFFERENT headline metrics "
            f"({sorted(k or 'none' for k in headline_keys)}), so no single ranking "
            "covers them. One sweep is one workload. Nothing is selected; every "
            "candidate's own row is below."
        )
    headline_metric = (
        headline_keys.pop() if (headline_keys and headline_conflict is None) else None
    )
    if valid and headline_conflict is None and headline_metric is None:
        headline_conflict = (
            "no valid candidate publishes a headline metric "
            f"({' or '.join(HEADLINE_KEYS)}): this workload's timeline has nothing "
            "to rank on. ADJ-6's headline is tokens/s at the decode terminal; a run "
            "that lowers no decode step and no request span cannot be ranked."
        )
    rankable = [] if headline_conflict else valid
    order = {cand["id"]: index for index, cand in enumerate(candidates)}
    front_ids = pareto_front_ids(rankable)
    shape, note = front_shape(rankable, front_ids, tuple(spec.axes))
    if headline_conflict:
        shape, note = "unrankable", headline_conflict
    selected = (
        min(rankable, key=lambda c: _selection_key(c, spec.objective, order))
        if rankable
        else None
    )

    # One disclosure list for the sweep: the union over valid candidates,
    # deduplicated by constraint key (the same rule the priced atlas uses).
    # Each entry NAMES the candidates that carried it, and a candidate that
    # carries the same constraint with a DIFFERENT value is recorded beside
    # it rather than swallowed — a per-candidate quantity presented as a
    # sweep-level one would be a second accounting under one name (D21).
    disclosures = []
    by_constraint = {}
    for cand in valid:
        for item in cand.get("disclosures", ()):
            key = item["constraint"]
            entry = by_constraint.get(key)
            if entry is None:
                entry = dict(item)
                entry["candidates"] = [cand["id"]]
                by_constraint[key] = entry
                disclosures.append(entry)
                continue
            entry["candidates"].append(cand["id"])
            if item["value"] != entry["value"] or item["reason"] != entry["reason"]:
                entry.setdefault("varies_by_candidate", []).append(
                    {
                        "id": cand["id"],
                        "value": item["value"],
                        "reason": item["reason"],
                    }
                )

    # The silicon accounting's coverage, carried at sweep level so it reaches
    # the MD whatever the front's shape is.
    silicon_coverage = None
    if valid:
        source = selected or valid[0]
        uncovered = []
        for cand in valid:
            for term in cand["silicon"].get("uncovered_terms", ()):
                if term not in uncovered:
                    uncovered.append(term)
        silicon_coverage = {
            "basis_from": source["id"],
            "basis": source["silicon"]["basis"],
            "uncovered_terms": uncovered,
        }

    payload = {
        "tool": "fws_qif_dse",
        "schema": "fws_qif_dse/1",
        "hardware_config": str(hw_path),
        "model_config": str(model_path),
        "model_id": model_id or "",
        "mode": mode,
        "objective": spec.objective,
        "headline_metric": headline_metric,
        "headline_conflict": headline_conflict,
        "selection_rule": (
            SELECTION_RULES[spec.objective]
            if headline_metric in (None, "tokens_per_s")
            else SELECTION_RULES[spec.objective].replace(
                "headline tokens/s", "headline requests/s (this workload lowers no "
                "decode step, so ADJ-6's tokens/s does not exist on its timeline)"
            )
        ),
        "evaluation_path": (
            "fws_mapping.build_mapping -> program.fws_build.build_fws_program -> "
            "fws_eval.evaluate_fws, per candidate. No closed form, no period, no "
            "shared timeline (A1, ADJ-6)."
        ),
        "sweep": spec.echo(),
        "stages": list(STAGES),
        "num_candidates": len(candidates),
        "num_valid": len(valid),
        "candidates": candidates,
        "front_ids": front_ids,
        "front_shape": shape,
        "front_note": note,
        "selected_id": None if selected is None else selected["id"],
        "selected": selected,
        "best_violation": best_violation(candidates),
        "disclosures": disclosures,
        "silicon_coverage": silicon_coverage,
        "total_wall_s": total_wall,
        "verify": None,
    }

    exit_code = 0
    if headline_conflict:
        exit_code = 2
    elif selected is None:
        exit_code = 3
    else:
        if emit_config is None and verify:
            emit_config = output_dir / "selected_config.yaml"
        if emit_config is not None:
            emitted = emit_selected_config(
                raw_base,
                selected,
                emit_config,
                base_path=str(hw_path),
                model_path=str(model_path),
                analog_card=analog_card,
                digital_card=digital_card,
                source=str(hw_path),
            )
            payload["emitted_config"] = str(emitted)
            if verify:
                payload["verify"] = verify_selected(
                    emitted, model_path, mode, output_dir, selected
                )
                if not payload["verify"]["pass"]:
                    exit_code = 4

    (output_dir / "dse_report.json").write_text(
        json.dumps(_json_safe(payload), indent=2) + "\n"
    )
    (output_dir / "dse_report.md").write_text(render_markdown(payload))
    payload["output_dir"] = str(output_dir)
    return exit_code, payload


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="fws_qif_dse",
        description=(
            "QIF P3.7: sweep the declared mapped-path candidate space "
            f"(`{DSE_BLOCK}`), evaluating EVERY candidate through "
            "fws_mapping -> fws_build -> fws_eval."
        ),
    )
    parser.add_argument(
        "--hardware_config",
        required=True,
        help=f"Mapped fws_cim hardware YAML carrying a `{DSE_BLOCK}:` block.",
    )
    parser.add_argument("--model_config", required=True, help="Model YAML (LLM or VIT).")
    parser.add_argument("--mode", default="LLM", choices=("LLM", "VIT"))
    parser.add_argument(
        "--model_id",
        default=None,
        help=(
            "The model's own name for tile owners and labels. Defaults to "
            "model_param.model_type, which is a PRICING CARRIER and not always the "
            "model's identity."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Artifact directory (default output/fws_qif_dse/<hw-stem>__<model-stem>/).",
    )
    parser.add_argument(
        "--objective",
        choices=sorted(SELECTION_RULES),
        default=None,
        help=f"Override {DSE_BLOCK}.objective.",
    )
    parser.add_argument(
        "--emit-config",
        default=None,
        help="Write the winning point as a complete runnable hardware YAML.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Rerun the emitted config through run_perf and cross-check the headline "
            "metrics within 0.1%% (nonzero exit on mismatch)."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-candidate lines.")
    args = parser.parse_args(argv)

    try:
        exit_code, payload = run_sweep(
            args.hardware_config,
            args.model_config,
            mode=args.mode,
            model_id=args.model_id,
            output_dir=args.output_dir,
            objective=args.objective,
            emit_config=args.emit_config,
            verify=args.verify,
            quiet=args.quiet,
        )
    except ValueError as exc:
        print(f"[QIF DSE] error: {exc}")
        return 2

    print(
        f"[QIF DSE] {payload['num_valid']}/{payload['num_candidates']} candidates valid "
        f"in {payload['total_wall_s']:.1f} s; artifacts in {payload['output_dir']}"
    )
    selected = payload["selected"]
    if selected is None and payload.get("headline_conflict"):
        print(f"[QIF DSE] UNRANKABLE: {payload['headline_conflict']}")
        print(
            "[QIF DSE] both reports are written; every candidate row is in them."
        )
        return exit_code
    if selected is None:
        best = payload["best_violation"]
        if best is not None:
            print(
                f"[QIF DSE] no feasible candidate. Best violation (candidate "
                f"{best['id']}, stage {best['fail_stage']}): {best['fail_message']}"
            )
        else:
            print("[QIF DSE] no feasible candidate.")
        return exit_code
    knobs = ", ".join(f"{k}={_knob_cell(v)}" for k, v in selected["knobs"].items())
    print(
        f"[QIF DSE] selected {selected['id']} ({knobs}) by {payload['selection_rule']}: "
        f"{headline_text(selected)}, "
        f"{_fmt(selected['silicon']['total_silicon_mm2'])} mm2, "
        f"{selected['placement']['analog_chips']} chips"
    )
    print(
        f"[QIF DSE] Pareto front {payload['front_ids']} — {payload['front_shape']}: "
        f"{payload['front_note']}"
    )
    verify = payload.get("verify")
    if verify is not None:
        if verify["pass"]:
            print("[QIF DSE] verify: the emitted config reproduces the selection (<= 0.1%).")
        else:
            print(f"[QIF DSE] verify FAILED: {verify['message']}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
