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

THE FRONTIER, REGIME v2 (P7.4/P7.5, D29-D32)
---------------------------------------------
Under D29 the pipeline is ALWAYS FULL: decode is independent streams staggered
across the PP stages, one token exits per beat, the local batch is always 1 and
the resident-stream count D IS the stage count. That changes what a design
point is, so it changes what this tool sweeps:

* **stage granularity** (``layers_per_chip``, or ``layers_per_stage`` when the
  stage plan is decoupled from the chip split) is the axis that matters. It
  moves D, the beat, the per-stage state bill and the chip count AT ONCE, and
  it is the only axis on the sweep that trades throughput for anything.
* **mux/bank depth** (``bank_depth``) where the card admits it — a depth the
  card's ``adc_mux`` does not divide is refused BY THE CARD and recorded.
* **chip capacity** (``arrays_per_chip``) — the slot budget a chip offers. It
  is the AREA axis: Invariant W (D27) pins the analog macro count at the global
  cell floor, so the only way a point spends analog silicon is by enumerating
  slots no tensor lands on.

**Digital provisioning is NOT an axis.** D31/ADJ-9 retire the scan/vector engine
as a declared or swept knob: it is DERIVED per point — sized UP until every
stage's digital per-stage time fits that stage's own ANALOG m-pass time — and
D32 prices it (and the per-macro pools) from the measured synthesis library.
``vector_lanes`` therefore lives in :data:`REFUSED_AXES` and a sweep that
declares it is REFUSED BY NAME at parse.

Every point reports: total area (analog floor + composed shared chiplets +
composed per-macro pools), tokens/s = 1/beat, D, the per-stage state bytes with
its verdict, and per-device-class utilization (D28).

The candidate space (``mapping_dse``, the block this tool owns)
---------------------------------------------------------------
EXPLICIT CANDIDATE LISTS ONLY. There is no optimizer, no sampling and no random
seed anywhere. ``search`` picks how the declared lists are walked:

``cross_product`` (the default, Wave D's law)
    the full cross product in declaration order; every point is evaluated.
``ladder`` (P7.4)
    a DIRECTIONAL STEP-UP walk in the OPTIMA spirit: start at the declared
    balanced ``initializer``, step ONE axis ONE rung at a time, CLIMB toward
    throughput (best tokens/s per mm2), TRIM every accepted rung of idle
    silicon (D28), then DESCEND from the trimmed initializer trading
    throughput for area. Every neighbour the walk looks at is priced in full
    and kept in the report, including the ones it declines; the walk is
    deterministic and the trail is printed. A step the PLACEMENT refuses is
    retried once on the declared ``repair_axis`` at the smallest rung that
    admits it, because stage granularity and chip capacity are coupled — a
    chip has to hold the layers its stage owns — and both the refusal and the
    repaired point stay in the report.

    The front a ladder reports is the non-dominated set of the points it
    VISITED, which is a lower bound on the front of the whole declared space.
    That rides every ladder report as a disclosure by name; nothing unvisited
    is estimated or interpolated, it is simply absent.

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
      search: ladder                 # or cross_product (the default)
      repair_axis: arrays_per_chip   # ladder only: retry a refused step here
      max_stage_state_bytes: 0       # 0 = the D29 residency check does not run
      initializer:                   # required by `ladder`: one declared rung per axis
        layers_per_chip: 4
        arrays_per_chip: 640
      axes:
        layers_per_chip: [10, 5, 4, 2, 1]       # STAGE GRANULARITY -> D, beat, state
        arrays_per_chip: [160, 320, 560, 640]   # CHIP CAPACITY -> the area axis
        bank_depth: [1, 2]                      # analog card banking (A3/D10)
        # layers_per_stage: [4, 2]              # stages decoupled from chips (D29)
        # column_sets_per_tile: [1]             # cim.allocation (D10)
        # shared_chiplets: [1, 10]              # ADJ-5's declared count
        # vector_lanes: [...]                   # REFUSED BY NAME (D31/ADJ-9)

Axis -> the config field it moves (one field each, never two):
  ``bank_depth``            ``cim.cards.<analog card>.bank_depth``
  ``column_sets_per_tile``  ``cim.allocation.column_sets_per_tile``
  ``arrays_per_chip``       ``cim.chip.arrays_per_chip``
  ``layers_per_chip``       ``mapping.layers_per_chip``
  ``layers_per_stage``      ``mapping.layers_per_stage``
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
  ``config < mapping < budget < lowering < pricing < memory < state``.
  ``memory`` names the evaluator's own violated tier verdicts; ``state`` is
  D29's per-stage residency against the sweep's declared budget, and both are
  reported with the scope, the stage and the bytes rather than a word.
* Pareto front: headline throughput vs total silicon over the valid
  candidates (2-axis dominance), plus the KNEE — the front point furthest
  above the chord joining the front's own extremes.
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
#:
#: ``vector_lanes`` is NOT here: ADJ-9 refuses it by name (:data:`REFUSED_AXES`),
#: and an axis this table lists is an axis the tool offers to sweep.
AXIS_TARGETS = {
    "bank_depth": "cim.cards.<analog card>.bank_depth",
    "column_sets_per_tile": "cim.allocation.column_sets_per_tile",
    "arrays_per_chip": "cim.chip.arrays_per_chip",
    "layers_per_chip": "mapping.layers_per_chip",
    "layers_per_stage": "mapping.layers_per_stage",
    "shared_chiplets": "mapping.shared_chiplets",
}

#: Axes REFUSED BY NAME, with the decision that killed them (D31, ADJ-9).
#:
#: A refused axis is spellable only far enough to be REFUSED: the parser raises
#: with the axis named and the decision quoted, exactly as D26 refuses a dead
#: fold. It is not silently ignored (that would sweep something else) and it is
#: not merely labelled (Wave F did that, and the labelled axis went on
#: producing a checked-in artifact whose whole spread came from overriding the
#: derivation).
#:
#: ADJ-9 is the decision that closed it: the engine is now derived to the
#: ANALOG FLOOR, so a wider declared width is not a design point the frontier
#: may carry — it is a machine whose digital side was chosen instead of
#: derived. The Wave-D demo sweep that swept this axis was regenerated on
#: D31-legal axes when ADJ-9 landed.
REFUSED_AXES = {
    "vector_lanes": (
        "D31/ADJ-9 retire the scan/vector engine as a swept or declared knob: its "
        "width is DERIVED — sized UP until every stage's digital per-stage time fits "
        "that stage's own ANALOG m-pass time — and REPORTED "
        "(evaluation.digital_silicon.derived_engine_sizing). Sweeping it would sweep "
        "the answer. The frontier (P7.4) sweeps stage granularity, mux/bank depth and "
        "chip capacity, and derives the digital side at every point. An explicit "
        "cim.cards.<card>.vector_lanes on a single machine is still an OVERRIDE that "
        "rides its own disclosure (D31); what is refused here is making it an AXIS."
    ),
}

#: Evaluation stages in order; the index is how far a candidate got.
#:
#: ``state`` is D29's own feasibility check and it sits LAST because it needs
#: the priced timeline: every stage of a filled pipeline holds all D streams'
#: recurrent state and KV for its layers, and a stage plan whose per-stage bill
#: exceeds the declared budget is a machine that cannot be built.
STAGES = ("config", "mapping", "budget", "lowering", "pricing", "memory", "state")

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
    # P7.7 / D29, the filled pipeline's own metrics. A metric that neither run
    # publishes is simply absent from the round trip; these are checked exactly
    # like the others when the regime publishes them.
    "beat",
    "per_stream_tokens_per_s",
    "per_token_latency",
    "resident_streams",
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
    if axis == "layers_per_stage":
        # D29: an int is a uniform stage width, a list is the per-stage plan.
        # 'auto' does NOT exist here, because the default stage plan is not a
        # search: mapping.layers_per_stage absent means one stage per chip, and
        # a sweep that wanted that would sweep layers_per_chip instead.
        if isinstance(raw, (list, tuple)):
            if not raw:
                raise QifDseUsageError(f"{context} list must not be empty.")
            return [int(item) for item in raw]
        value = int(raw)
        if value < 1:
            raise QifDseUsageError(f"{context} must be >= 1 (got {value}).")
        return value
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


#: How the candidate space is walked.
#:
#: ``cross_product`` is Wave D's law and stays the default: the full cross
#: product in declaration order, nothing skipped. ``ladder`` is P7.4's
#: directional step-up (the OPTIMA spirit George named): start at a DECLARED
#: balanced initializer and step ONE axis ONE rung at a time, outward in both
#: directions, keeping the step that buys the most throughput per mm2 going up
#: and saves the most mm2 per token/s going down. It is not a search heuristic
#: with a random seed and it is not sampling: every point it visits is
#: evaluated in full and recorded, and the walk is deterministic.
SEARCH_MODES = ("cross_product", "ladder")


class SweepSpec:
    """The parsed ``mapping_dse`` block: axes in declaration order."""

    def __init__(self, axes, objective, max_chips, max_silicon_mm2, label,
                 search="cross_product", initializer=None, max_stage_state_bytes=0.0,
                 repair_axis=None):
        self.axes = axes  # OrderedDict-like: {axis: [values]} in declaration order
        self.objective = objective
        self.max_chips = int(max_chips)
        self.max_silicon_mm2 = float(max_silicon_mm2)
        self.max_stage_state_bytes = float(max_stage_state_bytes)
        self.label = label
        self.search = search
        self.initializer = initializer or {}
        self.repair_axis = repair_axis

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

    @property
    def initializer_index(self):
        """The initializer as an index vector over the declared axis lists."""
        return tuple(
            self.axes[name].index(self.initializer[name]) for name in self.axes
        )

    def echo(self):
        return {
            "block": DSE_BLOCK,
            "label": self.label,
            "objective": self.objective,
            "search": self.search,
            "initializer": dict(self.initializer),
            "repair_axis": self.repair_axis,
            "max_chips": self.max_chips,
            "max_silicon_mm2": self.max_silicon_mm2,
            "max_stage_state_bytes": self.max_stage_state_bytes,
            "axes": {name: list(values) for name, values in self.axes.items()},
            "axis_targets": {name: AXIS_TARGETS[name] for name in self.axes},
            "refused_axes": dict(REFUSED_AXES),
            "enumeration": (
                "full cross product in declaration order; explicit candidate "
                "lists only, no search and no sampling (D19, D10)"
                if self.search == "cross_product"
                else (
                    "DIRECTIONAL STEP-UP LADDER (P7.4): from the declared "
                    "initializer, one axis one rung at a time, outward in both "
                    "directions; every point visited is evaluated in full and "
                    "recorded, and the walk is deterministic. Never a blind cross "
                    "product."
                )
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
        known = (
            "axes",
            "objective",
            "max_chips",
            "max_silicon_mm2",
            "max_stage_state_bytes",
            "label",
            "search",
            "initializer",
            "repair_axis",
        )
        unknown = [key for key in block if key not in known]
        if unknown:
            raise QifDseUsageError(
                f"{DSE_BLOCK} has unknown key(s) {unknown}; known keys are "
                f"{list(known)}. A typo is refused rather than ignored, because an "
                "ignored knob silently sweeps something else."
            )
        axes_raw = _require_mapping(f"{DSE_BLOCK}.axes", block.get("axes"))
        refused = [name for name in axes_raw if name in REFUSED_AXES]
        if refused:
            raise QifDseUsageError(
                f"{DSE_BLOCK}.axes declares the REFUSED axis/axes {refused}. "
                + " ".join(REFUSED_AXES[name] for name in refused)
            )
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
        search = str(block.get("search", "cross_product") or "cross_product").strip().lower()
        if search not in SEARCH_MODES:
            raise QifDseUsageError(
                f"{DSE_BLOCK}.search must be one of {list(SEARCH_MODES)} (got "
                f"{search!r})."
            )
        initializer_raw = block.get("initializer")
        initializer = {}
        if initializer_raw is not None:
            initializer_raw = _require_mapping(f"{DSE_BLOCK}.initializer", initializer_raw)
            unknown_init = [name for name in initializer_raw if name not in axes]
            if unknown_init:
                raise QifDseUsageError(
                    f"{DSE_BLOCK}.initializer names {unknown_init}, which is not a "
                    f"swept axis; the initializer is one value per SWEPT axis "
                    f"({list(axes)})."
                )
            for name in axes:
                if name not in initializer_raw:
                    raise QifDseUsageError(
                        f"{DSE_BLOCK}.initializer is missing axis {name!r}. The "
                        "initializer is a COMPLETE design point — the balanced "
                        "machine the ladder steps outward from — so every swept axis "
                        "names its starting rung."
                    )
                value = _axis_value(
                    name, initializer_raw[name], f"{DSE_BLOCK}.initializer.{name}"
                )
                if value not in axes[name]:
                    raise QifDseUsageError(
                        f"{DSE_BLOCK}.initializer.{name} = {value!r} is not one of that "
                        f"axis's declared rungs {axes[name]}. The ladder steps ALONG "
                        "the declared list, so its starting point has to be on it."
                    )
                initializer[name] = value
        if search == "ladder" and not initializer:
            raise QifDseUsageError(
                f"{DSE_BLOCK}.search = 'ladder' needs an `initializer:` block naming "
                "one declared rung per swept axis. A ladder starts from a BALANCED "
                "machine somebody chose and steps outward; inventing the start point "
                "would make the walk — and therefore the front it reports — an "
                "accident of this tool's defaults."
            )
        if search == "cross_product" and initializer:
            raise QifDseUsageError(
                f"{DSE_BLOCK}.initializer is declared but search is 'cross_product', "
                "which visits every point regardless. An initializer that changes "
                "nothing would read as a knob."
            )
        repair_axis = block.get("repair_axis")
        if repair_axis is not None:
            repair_axis = str(repair_axis).strip()
            if repair_axis not in axes:
                raise QifDseUsageError(
                    f"{DSE_BLOCK}.repair_axis = {repair_axis!r} is not a swept axis "
                    f"({list(axes)}). The repair axis is the one a refused step is "
                    "retried on, so it has to be a ladder this sweep declares."
                )
            if search != "ladder":
                raise QifDseUsageError(
                    f"{DSE_BLOCK}.repair_axis only means anything to a ladder; a cross "
                    "product visits every point already."
                )
        return cls(
            axes=axes,
            objective=objective,
            max_chips=int(block.get("max_chips", 0) or 0),
            max_silicon_mm2=float(block.get("max_silicon_mm2", 0.0) or 0.0),
            max_stage_state_bytes=float(block.get("max_stage_state_bytes", 0.0) or 0.0),
            label=str(block.get("label", "") or ""),
            search=search,
            initializer=initializer,
            repair_axis=repair_axis,
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

    ``digital_card`` is unused since ADJ-9 refused ``vector_lanes``, which was
    the only axis that wrote a digital-card field. The parameter stays because
    the tool resolves the two card names together (:func:`_card_names`) and
    both call sites pass the pair; dropping it here would split them.
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
        if axis == "bank_depth":
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
        elif axis in ("layers_per_chip", "layers_per_stage", "shared_chiplets"):
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
    # D32 / P7.4: the PER-MACRO DIGITAL POOL is digital silicon too, and it is
    # composed from the same measured library as the chiplet. It used to be
    # missing from this total, which under-reported every point by one whole
    # device class and — worse for a frontier — by a term that MOVES with the
    # chip-capacity axis (the pool count is the enumerated macro SLOT count).
    # One accounting for the machine's silicon (D21): analog arrays + shared
    # digital chiplets + per-macro pools, each term named.
    pool_per_macro = 0.0
    pool_composed = False
    if device.has_synthesis_library():
        pool_composed = True
        pool_per_macro = float(
            device.macro_pool_composition(device.digital_pool_sizing()).area_mm2
        )
    pools = pool_per_macro * slots
    uncovered = []
    if footprint <= 0:
        uncovered.append("analog macro footprint (cim.analog.area_mm2_per_array = 0)")
    # D32: the digital term is COMPOSED from measured synthesis blocks the
    # moment the card names a library, and DECLARED (the old placeholder)
    # otherwise. The two are different accountings and the basis says which
    # one produced the number (D21).
    composed = bool(device.has_synthesis_library())
    if per_chiplet <= 0:
        uncovered.append(
            "shared digital chiplet area (the card declares no area_mm2 and names no "
            "synthesis_library)"
        )
    if not pool_composed:
        uncovered.append(
            "per-macro digital pool area (the card names no synthesis_library, so "
            "D32's block composition has nothing to compose)"
        )
    digital_basis = (
        f"the card's COMPOSED area {per_chiplet:.6g} mm2 (D32: measured "
        f"{device.synthesis_technology} synthesis blocks, unit counts x measured "
        "area, no overhead)"
        if composed
        else f"the card's declared area_mm2 {per_chiplet:.6g}"
    )
    return {
        "analog_macro_slots": slots,
        "macro_footprint_mm2": footprint,
        "analog_macro_silicon_mm2": analog,
        "shared_digital_area_mm2_per_chiplet": per_chiplet,
        "shared_digital_area_provenance": "composed-measured" if composed else "declared",
        "shared_digital_silicon_mm2": digital,
        "macro_pool_area_mm2_per_macro": pool_per_macro,
        "macro_pool_area_provenance": "composed-measured" if pool_composed else "absent",
        "macro_pool_silicon_mm2": pools,
        "digital_silicon_mm2": digital + pools,
        "total_silicon_mm2": analog + digital + pools,
        "uncovered_terms": uncovered,
        "basis": (
            f"{slots} enumerated analog macro slots x macro_footprint_mm2 "
            f"{footprint:.6g} + {chiplets} shared digital chiplets x "
            f"{digital_basis} + {slots} per-macro digital pools x "
            + (
                f"the COMPOSED pool area {pool_per_macro:.6g} mm2 (D32, same "
                "measured library)"
                if pool_composed
                else "0 mm2 (no library to compose it from)"
            )
            + ". The analog term is the P3.6 "
            "atlas's enumerated_macro_silicon (slots the machine HAS), NOT "
            "CimDeviceModel.total_area_mm2 (arrays a model NEEDS)."
            + (
                " UNCOVERED (absent law, not a measured zero): " + "; ".join(uncovered)
                if uncovered
                else ""
            )
        ),
    }


def _digital_block(evaluation):
    """What the DERIVED digital side came out as at this point (D31/D32).

    Read off ``fws_eval``'s own ``digital_silicon`` block so the sweep never
    recomposes a second version of it (D21). ``None`` when the card names no
    synthesis library — there is then nothing composed to report, and the
    silicon accounting already says so in its own uncovered terms.
    """
    import fws_eval

    block = fws_eval._digital_silicon_block(evaluation)
    if not block.get("composed"):
        return None
    sizing = block.get("derived_engine_sizing") or {}
    return {
        "library_technology": (block.get("library") or {}).get("technology"),
        "vector_lanes": block.get("vector_lanes"),
        "vector_lanes_provenance": block.get("vector_lanes_provenance"),
        "engine_duty": sizing.get("engine_duty_at_target"),
        "sizing_target": sizing.get("sizing_target"),
        "analog_bound_stages": len(sizing.get("analog_bound_stages") or ()),
        "stages_sized": sizing.get("stages_sized"),
        "analog_beat_s": sizing.get("analog_beat_s"),
        "shared_digital_chiplet_area_mm2": (
            block.get("shared_digital_chiplet") or {}
        ).get("area_mm2_per_chiplet"),
        "digital_area_mm2_total": block.get("digital_area_mm2_total"),
        "digital_power_W_total": block.get("digital_power_W_total"),
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
    from fws_mapping import REGIME_FILLED
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
    # PROVISIONAL. The shared-digital term is a D32 composition of measured
    # blocks, and one of those blocks — the scan/vector engine — is sized by
    # D31 from the beat, which only exists once the pricing stage below has run
    # a timeline. Reading it here would compose the chiplet WITHOUT its engine.
    # It is computed here only so the declared max_silicon_mm2 budget can fail a
    # candidate before the expensive lowering, and it is REPLACED by the
    # composition taken after pricing (search for "the silicon accounting is
    # finalised" below) so exactly one number leaves this function (D21).
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
    # D27 / INVARIANT W AT EVERY POINT. The analog term above prices the slots
    # the machine HAS; this is what landed in them. Without it the sweep could
    # (and did) claim the analog macro count is pinned at the cell floor while
    # reporting no census at all. Under the dedicated law there is no packing to
    # summarise and the block says so BY NAME rather than printing zeros.
    packing_summary = mapping.packing_summary()
    if packing_summary:
        candidate["packing"] = {
            "law": str(packing_summary["packing"]),
            "macros": int(packing_summary["macros"]),
            "dedicated_macros": int(packing_summary["dedicated_macros"]),
            "macros_saved": int(packing_summary["macros_saved"]),
            "cell_floor_macros": int(packing_summary["cell_floor_macros"]),
            "real_cells": float(packing_summary["real_cells"]),
            "committed_cells": float(packing_summary["committed_cells"]),
            "waste_pct": float(packing_summary["waste_pct"]),
            "basis": (
                "D27's waste, counted in CELLS: committed cells that hold no real "
                "weight, over the committed total. cell_floor_macros is the GLOBAL "
                "lower bound ceil(total model cells / cells per macro); a per-chip "
                "packer cannot reach it when the layers are split across chips, so "
                "macros >= cell_floor_macros is expected and the gap is the price of "
                "the chip split, not a packing failure."
            ),
        }
    else:
        candidate["packing"] = {
            "law": str(mapping.packing),
            "measured": False,
            "basis": (
                "this point is placed under the DEDICATED law, which runs no packer: "
                "there is no cell census to report and printing one would be an "
                "invented number (D21). Declare mapping.packing: dense for D27's "
                "census."
            ),
        }
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

    # THE SILICON ACCOUNTING IS FINALISED HERE (D21, D31, D32). evaluate_fws
    # has just derived this point's vector engine from its own beat and
    # installed it on the device, so the shared-digital chiplet now composes
    # WITH its scan engine — which is 0.6 to 1.3 mm2 per chiplet on the shipped
    # curves and moves from point to point, because the derived width does.
    # Composing before pricing dropped that term from every row while still
    # claiming provenance "composed-measured" and uncovered_terms [], and it
    # disagreed with fws_eval's own digital_silicon block for the same machine.
    # Recomputing here reads the same device the evaluator just sized, so the
    # sweep and the simulator publish ONE number.
    silicon = _silicon(mapping, chiplets)
    candidate["silicon"] = silicon
    if spec.max_silicon_mm2 > 0 and silicon["total_silicon_mm2"] > spec.max_silicon_mm2:
        candidate["wall_s"] = time.time() - started
        return _fail(
            candidate,
            "budget",
            f"with its DERIVED vector engine (D31) this point needs "
            f"{silicon['total_silicon_mm2']:.6g} mm2 of silicon but "
            f"{DSE_BLOCK}.max_silicon_mm2 = {spec.max_silicon_mm2:.6g}. The engine is "
            "sized from the beat, so this term does not exist until the point has been "
            "priced; the budget is checked again here rather than passing a candidate "
            "the pre-derivation area happened to admit.",
        ), mapping

    metrics = [dict(entry.as_dict()) for entry in evaluation.metrics
               if not isinstance(entry.value, (list, tuple))]
    candidate["metrics"] = {
        "ops_priced": len(evaluation.pricing.costs),
        "makespan_s": float(evaluation.makespan_s),
        "entries": metrics,
        "tokens_per_s": _metric_by_suffix(metrics, "tokens_per_s"),
        "prefill_latency_s": _metric_by_suffix(metrics, "prefill_latency"),
        # P7.7 / D29: under the FILLED PIPELINE there is no decode "step" and no
        # prefill in the run at all — the per-token period is the BEAT, and it
        # is the same quantity this column has always carried (the time between
        # one token and the next). A lockstep candidate keeps its step median;
        # a filled-pipeline candidate carries its beat, and the regime column
        # says which one a reader is looking at.
        "decode_step_median_s": (
            _metric_by_suffix(metrics, "decode_step_median")
            if _metric_by_suffix(metrics, "decode_step_median") is not None
            else _metric_by_suffix(metrics, "beat")
        ),
        "serving_regime": str(getattr(evaluation.serving, "regime", "")),
        "beat_s": _metric_by_suffix(metrics, "beat"),
        "resident_streams": _metric_by_suffix(metrics, "resident_streams"),
        "per_stream_tokens_per_s": _metric_by_suffix(metrics, "per_stream_tokens_per_s"),
        "per_token_latency_s": _metric_by_suffix(metrics, "per_token_latency"),
        "max_stage_state_bytes": (
            float(evaluation.state_residency.get("max_stage_state_bytes", 0.0))
            if evaluation.state_residency.get("measured")
            else None
        ),
        "state_residency_verdict": (
            str(evaluation.state_residency.get("verdict"))
            if evaluation.state_residency.get("measured")
            else None
        ),
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
    # D29's own headline block: the beat, the resident-stream count D that the
    # stage plan DERIVES, and the two rates that follow from them. A lockstep
    # candidate publishes none of it and says so rather than reporting zeros.
    residency = dict(evaluation.state_residency)
    measured_pipeline = dict(getattr(evaluation, "pipeline", {}) or {})
    candidate["pipeline"] = {
        "serving_regime": str(getattr(evaluation.serving, "regime", "")),
        "filled": str(getattr(evaluation.serving, "regime", "")) == REGIME_FILLED,
        "beat_s": _metric_by_suffix(metrics, "beat"),
        # The beat is a MEASUREMENT, so its convergence state belongs on the
        # row that carries it — a reader of the front's throughput endpoint
        # must not have to go to the sweep-level disclosure list to learn that
        # its exits were still ramping.
        "beat_converged": measured_pipeline.get("beat_converged"),
        "steady_beat_spread": measured_pipeline.get("steady_beat_spread"),
        "converged_tail_beats": measured_pipeline.get("converged_tail_beats"),
        "ramp_beats_dropped": measured_pipeline.get("ramp_beats_dropped"),
        "tokens_per_s": _metric_by_suffix(metrics, "tokens_per_s"),
        "resident_streams": _metric_by_suffix(metrics, "resident_streams"),
        "per_stream_tokens_per_s": _metric_by_suffix(metrics, "per_stream_tokens_per_s"),
        "per_token_latency_s": _metric_by_suffix(metrics, "per_token_latency"),
        "stages": residency.get("stages"),
        "basis": (
            "D29: the pipeline is ALWAYS FULL. tokens/s = 1/beat is the exit rate of "
            "ONE token per beat; D = the stage count, DERIVED from the stage plan and "
            "never configured; the per-stream rate is 1/(D x beat)."
        ),
    }
    # THE STATE BILL (D29's first-class reported quantity). Every stage holds
    # all D streams' recurrent state and KV for its layers, so this is what
    # decides whether a stage plan is buildable at all.
    if residency.get("measured"):
        candidate["state_bill"] = {
            "measured": True,
            "resident_streams": int(residency.get("resident_streams", 0) or 0),
            "stages": int(residency.get("stages", 0) or 0),
            "max_stage_state_bytes": float(residency.get("max_stage_state_bytes", 0.0)),
            # The SAME number under the name that says whose it is: under
            # `sweep` this key holds the DECLARED budget, and one document
            # must not spell two quantities the same way (D21).
            "measured_max_stage_state_bytes": float(
                residency.get("max_stage_state_bytes", 0.0)
            ),
            "onchip_tier_verdict": str(residency.get("verdict", "")),
            "activation_tier_verdict": str(residency.get("activation_tier_verdict", "")),
            "total_state_bytes": float(residency.get("total_state_bytes", 0.0)),
            "per_stream_model_state_bytes": float(
                residency.get("per_stream_model_state_bytes", 0.0)
            ),
            "declared_tier_capacity_bytes": float(residency.get("capacity_bytes", 0.0)),
            "verdict": str(residency.get("verdict", "")),
            "stages_violating": list(residency.get("stages_violating", ())),
            "representative_context": float(residency.get("representative_context", 0.0)),
            "per_stage": [
                {
                    "stage": int(row["stage"]),
                    "label": row["label"],
                    "layers": int(row["layers"]),
                    "recurrent_state_bytes": float(row["recurrent_state_bytes"]),
                    "kv_bytes": float(row["kv_bytes"]),
                    "state_bytes": float(row["state_bytes"]),
                    "verdict": str(row["verdict"]),
                }
                for row in residency.get("per_stage", ())
            ],
            "basis": str(residency.get("basis", "")),
        }
    else:
        candidate["state_bill"] = {
            "measured": False,
            "reason": str(residency.get("reason", "")),
        }
    # D28: per-device-class utilization at EVERY point, idle devices inside the
    # mean, so idle silicon on the frontier is always visible.
    candidate["utilization"] = [
        {
            "device_class": row.device_class,
            "devices": int(row.devices),
            "idle_devices": int(row.idle_devices),
            "idle_share": float(row.idle_share),
            "mean_occupancy": float(row.mean_occupancy),
            "max_occupancy": float(row.max_occupancy),
        }
        for row in evaluation.utilization
    ]
    binding = max(
        evaluation.utilization, key=lambda row: row.max_occupancy, default=None
    )
    candidate["binding_device_class"] = None if binding is None else binding.device_class
    # D31/D32: the digital side is DERIVED at this point, never swept. What it
    # derived to is a reported quantity of the point.
    digital = _digital_block(evaluation)
    if digital is not None:
        candidate["derived_digital"] = digital

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
        # NAMED verdicts, not a count: a point that dies of memory has to say
        # WHICH scope, WHICH tier and WHOSE it was, or "infeasible" is a word
        # with nothing behind it.
        "violated_verdicts": [
            {
                "scope": v.scope,
                "tier": v.tier,
                "owner": v.owner,
                "high_water_bytes": float(v.high_water_bytes),
                "capacity_bytes": float(v.capacity_bytes),
                "ratio": (
                    float(v.high_water_bytes) / float(v.capacity_bytes)
                    if v.capacity_bytes
                    else None
                ),
            }
            for v in violated
        ],
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

    # Stage: state — D29's per-stage residency against the sweep's DECLARED
    # per-stage budget. Every stage of a filled pipeline holds all D streams'
    # recurrent state and KV for its layers, so a stage plan that buys
    # throughput by shrinking the stages pays for it here. The budget is a
    # DECLARED sweep constraint, not a measured capacity and not a margin
    # (D28): absent, the check does not run and the bill is reported anyway.
    bill = candidate["state_bill"]
    if spec.max_stage_state_bytes > 0:
        if not bill.get("measured"):
            candidate["wall_s"] = time.time() - started
            return _fail(
                candidate,
                "state",
                f"{DSE_BLOCK}.max_stage_state_bytes is declared, but this candidate "
                "measures NO per-stage residency: " + str(bill.get("reason", ""))
                + " A budget cannot be checked against a quantity the run does not "
                "produce, and passing it silently would report an unmade check as a "
                "pass (D21).",
            ), mapping
        worst_stage = max(
            bill["per_stage"], key=lambda row: row["state_bytes"], default=None
        )
        if bill["max_stage_state_bytes"] > spec.max_stage_state_bytes:
            over = [
                row
                for row in bill["per_stage"]
                if row["state_bytes"] > spec.max_stage_state_bytes
            ]
            candidate["state_bill"]["budget_bytes"] = float(spec.max_stage_state_bytes)
            candidate["state_bill"]["budget_verdict"] = "VIOLATED"
            candidate["state_bill"]["budget_violating_stages"] = [
                {
                    "stage": row["stage"],
                    "label": row["label"],
                    "layers": row["layers"],
                    "state_bytes": row["state_bytes"],
                }
                for row in over
            ]
            candidate["wall_s"] = time.time() - started
            return _fail(
                candidate,
                "state",
                f"INFEASIBLE BY MEMORY (D29): {len(over)} of {bill['stages']} pipeline "
                f"stage(s) hold more resident state than {DSE_BLOCK}."
                f"max_stage_state_bytes = {spec.max_stage_state_bytes:.6g} B. The worst "
                f"is stage {worst_stage['stage']} ({worst_stage['label']}, "
                f"{worst_stage['layers']} layer(s)) at "
                f"{worst_stage['state_bytes']:.6g} B = "
                f"{worst_stage['recurrent_state_bytes']:.6g} B of recurrent state + "
                f"{worst_stage['kv_bytes']:.6g} B of KV for all "
                f"{bill['resident_streams']} resident streams.",
            ), mapping
        candidate["state_bill"]["budget_bytes"] = float(spec.max_stage_state_bytes)
        candidate["state_bill"]["budget_verdict"] = "fits"
        candidate["state_bill"]["budget_headroom_bytes"] = float(
            spec.max_stage_state_bytes - bill["max_stage_state_bytes"]
        )
    else:
        candidate["state_bill"]["budget_bytes"] = None
        candidate["state_bill"]["budget_verdict"] = "undeclared"
    candidate["stages"]["state"] = {"ok": True}
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


# ---------------------------------------------------------------------------
# The LADDER (P7.4): a directional step-up walk, never a blind cross product
# ---------------------------------------------------------------------------


#: Below this relative change two points are the SAME reading, not a step.
#: The evaluator is deterministic, so this only guards float noise in the
#: differences the ranking divides by — it never merges two design points.
LADDER_EPS = 1e-12


def _rel_change(new, old):
    base = abs(float(old))
    return (float(new) - float(old)) / base if base > 0 else float(new) - float(old)


class LadderWalk:
    """One directional walk over the declared rungs.

    The walk is defined by two things and nothing else: what counts as a step
    worth taking (``accept``) and, among the steps worth taking, which is the
    best buy (``rank``). Both are given by the phase, so the three phases below
    share one loop and one record of what they visited.

    Every neighbour the walk LOOKS at is evaluated in full and kept, including
    the ones it declines: a point that was priced and then rejected is data
    about the frontier, and dropping it would make the artifact a story about
    the path instead of a record of the space.
    """

    def __init__(self, spec, visit):
        self.spec = spec
        self.visit = visit
        self.names = list(spec.axes)
        self.values = {name: list(spec.axes[name]) for name in self.names}
        self.trail = []
        self.steps = 0

    def neighbours(self, index):
        """(index, axis, direction) for every ONE-RUNG move, both directions.

        Both directions, always: an axis's declared order is the DECLARER's
        (a bigger chip capacity is more silicon, a bigger layers_per_chip is
        FEWER stages), so a walk that only stepped one way would silently
        depend on which way somebody wrote the list.
        """
        out = []
        for position, name in enumerate(self.names):
            for direction in (-1, 1):
                rung = index[position] + direction
                if 0 <= rung < len(self.values[name]):
                    out.append(
                        (
                            index[:position] + (rung,) + index[position + 1:],
                            name,
                            direction,
                            position,
                        )
                    )
        return out

    def repair(self, index, refused, phase, origin_id, axis, direction):
        """A refused step retried on the REPAIR axis, smallest rung that admits it.

        Stage granularity and chip capacity are not independent: a chip has to
        hold the layers its stage owns, so `layers_per_chip` steps get refused
        by the placement the moment the declared capacity is too small. Without
        this, a one-axis-at-a-time walk could only ever move granularity in the
        direction the current capacity already allowed, and the front it drew
        would be an artefact of the initializer's chip size.

        The repair is DIRECTIONAL and minimal: only a step refused at the
        MAPPING stage is repaired, only along the declared repair axis, and only
        to the SMALLEST rung that admits the step. The refusal stays in the
        report beside the repaired point — the capacity that did not work is
        part of what the sweep found.
        """
        name = self.spec.repair_axis
        if name is None or name == axis or refused.get("fail_stage") != "mapping":
            return None
        position = self.names.index(name)
        for rung in range(index[position] + 1, len(self.values[name])):
            probe = index[:position] + (rung,) + index[position + 1:]
            other = self.visit(probe, phase, origin_id, axis, direction)
            if other["ok"]:
                other.setdefault("ladder", {})["repaired_from"] = refused["id"]
                other["ladder"]["repair_axis"] = name
                note = (
                    f"REPAIRED STEP: {axis} {direction:+d} was refused at the mapping "
                    f"stage on candidate {refused['id']} ({refused['fail_message']}); "
                    f"this point is the same step at the SMALLEST declared "
                    f"{name} rung that admits it ({self.values[name][rung]})."
                )
                # A cached point can be reached by the same repair more than
                # once (the trim and the descend both look at it); the note is
                # the same claim each time and is recorded once.
                if note not in other["notes"]:
                    other["notes"].append(note)
                return probe, other
        return None

    def step(self, index, candidate, phase, accept, rank):
        """ONE rung, or none. Returns (index, candidate, moved).

        Every neighbour is priced before the choice is made, so the step is a
        decision over measured points and never over an estimate of them.
        """
        origin_id = candidate["id"]
        options = []
        for neighbour, axis, direction, position in self.neighbours(index):
            other = self.visit(neighbour, phase, candidate["id"], axis, direction)
            if not other["ok"]:
                repaired = self.repair(
                    neighbour, other, phase, origin_id, axis, direction
                )
                if repaired is None:
                    continue
                neighbour, other = repaired
            verdict = accept(candidate, other)
            if verdict is None:
                continue
            options.append((rank(candidate, other), -position, neighbour, other,
                            axis, direction, verdict))
        if not options:
            self.trail.append(
                {
                    "phase": phase,
                    "from": origin_id,
                    "accepted": None,
                    "reason": (
                        "no one-rung neighbour of this point is a step this phase "
                        "would take; the walk stops here."
                    ),
                }
            )
            return index, candidate, False
        best = max(options, key=lambda item: (item[0], item[1]))
        index, candidate = best[2], best[3]
        self.steps += 1
        self.trail.append(
            {
                "phase": phase,
                "step": self.steps,
                "from": origin_id,
                "accepted": candidate["id"],
                "axis": best[4],
                "direction": best[5],
                "knobs": dict(candidate["knobs"]),
                "reason": best[6],
                "considered": len(options),
            }
        )
        return index, candidate, True

    def walk(self, index, candidate, phase, accept, rank):
        """Step until no neighbour is worth taking; return where it stopped.

        The step limit is the total number of declared rungs. Each phase moves
        monotonically in its own quantity - the climb in throughput, the trim
        and the descend in area - so the limit is a guard against a future
        phase, never the thing that ends these walks.
        """
        limit = sum(len(values) for values in self.values.values())
        for _ in range(limit):
            index, candidate, moved = self.step(index, candidate, phase, accept, rank)
            if not moved:
                return index, candidate
        self.trail.append(
            {
                "phase": phase,
                "from": candidate["id"],
                "accepted": None,
                "reason": (
                    f"the walk hit its step limit of {limit} (the total number of "
                    "declared rungs). A walk that long has cycled or the space is "
                    "degenerate; it is stopped and recorded rather than run forever."
                ),
            }
        )
        return index, candidate


def _climb_accept(current, other):
    """A step UP: strictly more tokens/s. Area may go either way."""
    gain = _headline(other) - _headline(current)
    if gain <= 0 or abs(_rel_change(_headline(other), _headline(current))) < LADDER_EPS:
        return None
    cost = _silicon_of(other) - _silicon_of(current)
    if cost <= 0:
        return (
            f"+{gain:.6g} tokens/s for {cost:.6g} mm2 — a free win: this step buys "
            "throughput and costs no silicon."
        )
    return f"+{gain:.6g} tokens/s for +{cost:.6g} mm2 ({gain / cost:.6g} per mm2)."


def _climb_rank(current, other):
    gain = _headline(other) - _headline(current)
    cost = _silicon_of(other) - _silicon_of(current)
    # Free (or negative-cost) throughput outranks every priced step; among
    # priced steps the best buy is the most tokens/s per mm2.
    return (1, gain) if cost <= 0 else (0, gain / cost)


def _trim_accept(current, other):
    """A TRIM step: strictly less silicon and NOT less throughput (D28).

    Idle silicon is a provisioning bug the DSE must expose, so every point the
    climb accepts is trimmed before the next climb step: the frontier should
    never carry a point that a smaller machine matches.
    """
    saved = _silicon_of(current) - _silicon_of(other)
    if saved <= 0:
        return None
    lost = _headline(current) - _headline(other)
    if lost > 0:
        return None
    return (
        f"-{saved:.6g} mm2 at no cost in throughput ({lost:.6g} tokens/s): "
        "idle silicon trimmed off the point (D28)."
    )


def _trim_rank(current, other):
    return (0, _silicon_of(current) - _silicon_of(other))


def _descend_accept(current, other):
    """A step DOWN: strictly less silicon, paid for in throughput."""
    saved = _silicon_of(current) - _silicon_of(other)
    if saved <= 0:
        return None
    lost = _headline(current) - _headline(other)
    if lost <= 0:
        return (
            f"-{saved:.6g} mm2 for {lost:.6g} tokens/s — a free win: this step saves "
            "silicon and costs no throughput."
        )
    return f"-{saved:.6g} mm2 for -{lost:.6g} tokens/s ({saved / lost:.6g} mm2 per token/s)."


def _descend_rank(current, other):
    saved = _silicon_of(current) - _silicon_of(other)
    lost = _headline(current) - _headline(other)
    return (1, saved) if lost <= 0 else (0, saved / lost)


def ladder_candidates(spec, evaluate, announce=None):
    """Walk the declared space as a LADDER; return (candidates, trail).

    Three phases, in this order:

    1. the DECLARED initializer, trimmed of idle silicon before anything else
       (otherwise the descend phase would spend its first steps undoing the
       declarer's over-provisioning and report it as a trade);
    2. CLIMB one rung at a time, each accepted rung TRIMMED before the next
       one is ranked;
    3. DESCEND from the trimmed initializer, trading throughput for area.

    Nothing is sampled and nothing is estimated: every point either walk looks
    at is placed by P3 and priced by P4 on its own timeline, and the ones the
    walk declines stay in the report with their numbers.
    """
    names = list(spec.axes)
    values = {name: list(spec.axes[name]) for name in names}
    visited = {}
    ordered = []

    def visit(index, phase, origin, axis, direction):
        if index in visited:
            return visited[index]
        point = {name: values[name][rung] for name, rung in zip(names, index)}
        candidate = evaluate(point)
        candidate["ladder"] = {
            "phase": phase,
            "from": origin,
            "axis": axis,
            "direction": direction,
            "rungs": {name: rung for name, rung in zip(names, index)},
        }
        visited[index] = candidate
        ordered.append(candidate)
        if announce is not None:
            announce(candidate)
        return candidate

    walk = LadderWalk(spec, visit)
    start = spec.initializer_index
    root = visit(start, "initializer", None, None, 0)
    walk.trail.append(
        {
            "phase": "initializer",
            "accepted": root["id"],
            "knobs": dict(root["knobs"]),
            "reason": "the declared balanced start point the ladder steps outward from.",
        }
    )
    if not root["ok"]:
        walk.trail.append(
            {
                "phase": "initializer",
                "from": root["id"],
                "accepted": None,
                "reason": (
                    "the initializer is INFEASIBLE at stage "
                    f"{root['fail_stage']}, so there is no point to step outward from. "
                    "The walk stops and the refusal is the whole result: "
                    f"{root['fail_message']}"
                ),
            }
        )
        return ordered, walk.trail

    # The initializer is trimmed FIRST, so both walks start from a machine with
    # no idle silicon on it — otherwise the descend phase would spend its first
    # steps undoing the declarer's over-provisioning and call it a trade.
    base_index, base = walk.walk(start, root, "trim", _trim_accept, _trim_rank)

    # ONE climb rung, then trim that rung of idle silicon, then climb again.
    # Trimming only at the top would leave every intermediate rung of the front
    # over-provisioned by whatever capacity the rung below happened to carry,
    # and would rank the next climb step against that inflated area (D28).
    index, candidate = base_index, base
    limit = sum(len(items) for items in values.values())
    for _ in range(limit):
        index, candidate, moved = walk.step(
            index, candidate, "climb", _climb_accept, _climb_rank
        )
        if not moved:
            break
        index, candidate = walk.walk(index, candidate, "trim", _trim_accept, _trim_rank)

    walk.walk(base_index, base, "descend", _descend_accept, _descend_rank)
    return ordered, walk.trail


# ---------------------------------------------------------------------------
# The knee
# ---------------------------------------------------------------------------


#: Below this, the front is a straight line and has no bend to name.
KNEE_FLAT_TOL = 1e-9


def knee_of(candidates, front_ids):
    """The KNEE of the front: max distance from the chord joining its extremes.

    Both axes are normalized to [0, 1] over the FRONT (not the whole sweep), so
    the measure is a shape of the front and not of the units. On the normalized
    picture the chord runs from the cheapest, slowest point to the dearest,
    fastest one, and the knee is the point furthest ABOVE it — the last point
    where silicon still buys throughput at better than the front's average rate.

    A front of fewer than three points has no interior, so it has no knee, and
    this says so rather than naming an endpoint as one.
    """
    front = [c for c in candidates if c["id"] in front_ids]
    front.sort(key=lambda c: (_silicon_of(c), -_headline(c)))
    if len(front) < 3:
        return {
            "point": None,
            "basis": (
                f"the front has {len(front)} point(s) and a knee needs an INTERIOR: "
                "with two or fewer non-dominated points there is nothing between the "
                "extremes to bend. No point is named."
            ),
            "distance": None,
            "points": [c["id"] for c in front],
        }
    areas = [_silicon_of(c) for c in front]
    rates = [_headline(c) for c in front]
    span_area = areas[-1] - areas[0]
    span_rate = max(rates) - min(rates)
    if span_area <= 0 or span_rate <= 0:
        return {
            "point": None,
            "basis": (
                "the front is degenerate on one axis (area span "
                f"{span_area:.6g} mm2, throughput span {span_rate:.6g}), so the chord "
                "is a point or a vertical line and no distance to it is meaningful."
            ),
            "distance": None,
            "points": [c["id"] for c in front],
        }
    best = None
    for cand in front:
        x = (_silicon_of(cand) - areas[0]) / span_area
        y = (_headline(cand) - min(rates)) / span_rate
        # Distance above the chord y = x, in units of the normalized picture.
        distance = y - x
        if best is None or distance > best[0]:
            best = (distance, cand)
    if best[0] <= KNEE_FLAT_TOL:
        return {
            "point": None,
            "distance": float(best[0]),
            "basis": (
                f"the front is STRAIGHT: no point lies more than {KNEE_FLAT_TOL:.0e} of "
                "the normalized picture above the chord joining its extremes, so "
                "silicon buys throughput at the same rate everywhere on it and there "
                "is no bend to name. Naming an endpoint as a knee would invent one."
            ),
            "points": [c["id"] for c in front],
        }
    return {
        "point": best[1]["id"],
        "distance": float(best[0]),
        "basis": (
            f"max distance above the chord of the front, normalized over the front's "
            f"own spans ({span_area:.6g} mm2 x {span_rate:.6g} tokens/s) between "
            f"`{front[0]['id']}` (cheapest) and `{front[-1]['id']}` (fastest). The "
            f"knee is `{best[1]['id']}` at {best[0]:.6g} of the normalized picture "
            "above that chord: past it, silicon buys throughput at worse than the "
            "front's average rate."
        ),
        "points": [c["id"] for c in front],
    }


def _state_bill_summary(candidates, spec):
    """The STATE BILL across the sweep (D29's first-class reported quantity).

    Per-stage residency is what decides whether a stage plan can be built, so
    it rides the sweep-level payload beside the front instead of only inside
    each row. Points refused for it are named here too, because "infeasible"
    without the bytes and the stage is not a verdict.
    """
    measured = [
        cand
        for cand in candidates
        if (cand.get("state_bill") or {}).get("measured")
    ]
    refused = [
        {
            "id": cand["id"],
            "knobs": dict(cand["knobs"]),
            "max_stage_state_bytes": (cand.get("state_bill") or {}).get(
                "max_stage_state_bytes"
            ),
            "resident_streams": (cand.get("state_bill") or {}).get("resident_streams"),
            "violating_stages": (cand.get("state_bill") or {}).get(
                "budget_violating_stages", []
            ),
            "verdict": cand["fail_message"],
        }
        for cand in candidates
        if cand.get("fail_stage") == "state"
    ]
    by_memory = [
        {
            "id": cand["id"],
            "knobs": dict(cand["knobs"]),
            "verdicts": (cand.get("memory") or {}).get("violated_verdicts", []),
            "verdict": cand["fail_message"],
        }
        for cand in candidates
        if cand.get("fail_stage") == "memory"
    ]
    return {
        "budget_bytes": spec.max_stage_state_bytes or None,
        "budget_basis": (
            f"{DSE_BLOCK}.max_stage_state_bytes — a DECLARED per-stage residency "
            "budget for this sweep, checked against the MEASURED per-stage bill. It "
            "is not a measured capacity and it is not a margin (D28); absent, the "
            "check does not run and the bill is still reported."
            if spec.max_stage_state_bytes > 0
            else f"{DSE_BLOCK}.max_stage_state_bytes is not declared, so no point is "
            "refused for residency; every point's measured bill is still reported."
        ),
        "points_measured": len(measured),
        "max_stage_state_bytes_min": min(
            (cand["state_bill"]["max_stage_state_bytes"] for cand in measured),
            default=None,
        ),
        "max_stage_state_bytes_max": max(
            (cand["state_bill"]["max_stage_state_bytes"] for cand in measured),
            default=None,
        ),
        "infeasible_by_state": refused,
        "infeasible_by_memory": by_memory,
        "basis": (
            "D29: every stage holds ALL D streams' recurrent state and KV for its "
            "layers, so the per-stage bill scales with the resident-stream count D "
            "and D is the stage count. The bytes are fws_eval's own measured "
            "residency (evaluation.state_residency), not a second accounting."
        ),
    }


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
        pipeline = selected.get("pipeline") or {}
        if pipeline.get("filled"):
            # D29: there is no prefill in the run and no decode "step" — the
            # per-token period IS the beat, and the per-stream rate is 1/(D x beat).
            lines.append(
                f"- beat {_fmt(pipeline['beat_s'])} s (the measured interval between "
                f"token exits); per-stream {_fmt(pipeline['per_stream_tokens_per_s'])} "
                f"tokens/s; per-token latency {_fmt(pipeline['per_token_latency_s'])} s"
            )
        else:
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
        bill = selected.get("state_bill") or {}
        if bill.get("measured"):
            lines.append(
                f"- D29 pipeline: D = {int(bill['resident_streams'])} resident streams "
                f"over {int(bill['stages'])} stage(s); worst stage holds "
                f"{bill['max_stage_state_bytes'] / 1024 ** 2:.5g} MiB of state + KV "
                f"for all of them (verdict {bill['verdict']}, budget "
                f"{bill.get('budget_verdict', 'undeclared')})"
            )
        digital = selected.get("derived_digital")
        if digital:
            lines.append(
                f"- derived digital (D31/D32): {digital['vector_lanes']} vector lanes "
                f"({digital['vector_lanes_provenance']}), "
                f"{_fmt(digital['digital_area_mm2_total'])} mm2 and "
                f"{_fmt(digital['digital_power_W_total'])} W composed from the measured "
                f"{digital['library_technology']} synthesis library"
            )
        lines.append(f"- energy {_fmt(metrics['total_energy_pj'])} pJ over the timeline")
        lines.append(f"- wall time to evaluate this candidate: {_fmt(selected['wall_s'], '.3g')} s")
    lines.append("")

    lines.append("## Candidates")
    lines.append("")
    header = ["id"] + axes + [
        "ok", "fail stage", headline_unit, "silicon mm2", "analog mm2", "digital mm2",
        "chips", "D", "beat s", "MiB/stage", "binding class", "energy pJ", "wall s",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for cand in payload["candidates"]:
        metrics = cand.get("metrics") or {}
        place = cand.get("placement") or {}
        silicon = cand.get("silicon") or {}
        bill = cand.get("state_bill") or {}
        row = [cand["id"]] + [_knob_cell(cand["knobs"][axis]) for axis in axes] + [
            "yes" if cand["ok"] else "NO",
            cand["fail_stage"] or "-",
            _fmt(metrics.get(headline_key)),
            _fmt(silicon.get("total_silicon_mm2")),
            _fmt(silicon.get("analog_macro_silicon_mm2")),
            _fmt(silicon.get("digital_silicon_mm2")),
            _fmt(place.get("analog_chips")),
            _fmt(bill.get("resident_streams")),
            _fmt(metrics.get("decode_step_median_s")),
            (
                _fmt(bill["max_stage_state_bytes"] / 1024 ** 2, ".5g")
                if bill.get("measured")
                else "-"
            ),
            str(cand.get("binding_device_class") or "-"),
            _fmt(metrics.get("total_energy_pj"), ".4g"),
            _fmt(cand.get("wall_s"), ".3g"),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append(
        "`D` is the resident-stream count, which under D29 IS the stage count and is "
        "DERIVED from the stage plan, never configured; `beat s` is the measured "
        "interval between token exits and the headline is 1/beat; `MiB/stage` is the "
        "worst stage's resident state and KV for all D streams."
    )
    lines.append("")

    lines.append(f"## Pareto front (headline {headline_unit} vs total silicon)")
    lines.append("")
    # Sorted by AREA, because that is the axis the front is read along and an
    # unsorted list of non-dominated points reads as an unordered set.
    front_rows = sorted(
        (c for c in payload["candidates"] if c["id"] in payload["front_ids"]),
        key=lambda c: c["silicon"]["total_silicon_mm2"],
    )
    for cand in front_rows:
        bill = cand.get("state_bill") or {}
        state = (
            f", D = {int(bill['resident_streams'])}, "
            f"{bill['max_stage_state_bytes'] / 1024 ** 2:.5g} MiB/stage "
            f"({bill.get('budget_verdict', 'undeclared')})"
            if bill.get("measured")
            else ""
        )
        lines.append(
            f"- `{cand['id']}`: {_fmt(cand['metrics'][headline_key])} {headline_unit} at "
            f"{_fmt(cand['silicon']['total_silicon_mm2'])} mm2{state}"
        )
    if not payload["front_ids"]:
        lines.append("(empty — no valid candidate)")
    lines.append("")
    lines.append(f"Front shape — **{payload['front_shape']}**: {payload['front_note']}")
    lines.append("")

    knee = payload.get("knee") or {}
    lines.append("## The knee")
    lines.append("")
    if knee.get("point"):
        cand = next(c for c in payload["candidates"] if c["id"] == knee["point"])
        knobs = ", ".join(f"{axis} = {_knob_cell(cand['knobs'][axis])}" for axis in axes)
        bill = cand.get("state_bill") or {}
        lines.append(
            f"**`{knee['point']}`** ({knobs}) — "
            f"{_fmt(cand['metrics'][headline_key])} {headline_unit} at "
            f"{_fmt(cand['silicon']['total_silicon_mm2'])} mm2"
            + (
                f", D = {int(bill['resident_streams'])}, "
                f"{bill['max_stage_state_bytes'] / 1024 ** 2:.5g} MiB of resident "
                "state on the worst stage"
                if bill.get("measured")
                else ""
            )
            + "."
        )
    else:
        lines.append("NONE.")
    lines.append("")
    lines.append(f"Basis: {knee.get('basis', '-')}")
    lines.append("")

    bill_summary = payload.get("state_bill_summary") or {}
    lines.append("## The state bill (D29)")
    lines.append("")
    lines.append(f"- {bill_summary.get('basis', '-')}")
    lines.append(f"- budget: {bill_summary.get('budget_basis', '-')}")
    low = bill_summary.get("max_stage_state_bytes_min")
    high = bill_summary.get("max_stage_state_bytes_max")
    if low is not None and high is not None:
        lines.append(
            f"- measured worst-stage residency over {bill_summary['points_measured']} "
            f"priced point(s): {low / 1024 ** 2:.5g} MiB to {high / 1024 ** 2:.5g} MiB."
        )
    for label, key in (
        ("Infeasible by RESIDENCY", "infeasible_by_state"),
        ("Infeasible by MEMORY", "infeasible_by_memory"),
    ):
        rows = bill_summary.get(key) or []
        lines.append("")
        lines.append(f"### {label}")
        lines.append("")
        if not rows:
            lines.append("(none on this sweep)")
            continue
        for row in rows:
            knobs = ", ".join(f"{k} = {_knob_cell(v)}" for k, v in row["knobs"].items())
            lines.append(f"- `{row['id']}` ({knobs}): {row['verdict']}")
            for verdict in row.get("verdicts", ()):
                lines.append(
                    f"  - {verdict['scope']} / {verdict['tier']} ({verdict['owner']}): "
                    f"{verdict['high_water_bytes'] / 1024 ** 3:.3f} GiB against "
                    f"{verdict['capacity_bytes'] / 1024 ** 3:.3f} GiB"
                )
            for stage in row.get("violating_stages", ()):
                lines.append(
                    f"  - stage {stage['stage']} ({stage['label']}, "
                    f"{stage['layers']} layer(s)): "
                    f"{stage['state_bytes'] / 1024 ** 2:.5g} MiB"
                )
    lines.append("")

    trail = payload.get("ladder_trail")
    if trail:
        lines.append("## The ladder (how the space was walked)")
        lines.append("")
        lines.append(
            "Every point below was placed by P3 and priced by P4 on its own timeline, "
            "including the neighbours the walk declined."
        )
        lines.append("")
        for entry in trail:
            head = f"- **{entry['phase']}**"
            if entry.get("step"):
                head += f" step {entry['step']}"
            if entry.get("from"):
                head += f" from `{entry['from']}`"
            if entry.get("accepted"):
                head += f" -> `{entry['accepted']}`"
                if entry.get("axis"):
                    head += f" ({entry['axis']} {entry['direction']:+d})"
            else:
                head += " -> stop"
            lines.append(f"{head}: {entry['reason']}")
        lines.append("")

    lines.append("## Per-device-class utilization (D28)")
    lines.append("")
    valid_rows = [c for c in payload["candidates"] if c["ok"]]
    classes = []
    for cand in valid_rows:
        for row in cand.get("utilization", ()):
            if row["device_class"] not in classes:
                classes.append(row["device_class"])
    if valid_rows and classes:
        util_header = ["id"] + [f"{name} mean" for name in classes] + ["binding"]
        lines.append("| " + " | ".join(util_header) + " |")
        lines.append("|" + "---|" * len(util_header))
        for cand in valid_rows:
            by_class = {row["device_class"]: row for row in cand.get("utilization", ())}
            row = [cand["id"]] + [
                (
                    f"{by_class[name]['mean_occupancy'] * 100:.3f}%"
                    if name in by_class
                    else "-"
                )
                for name in classes
            ] + [str(cand.get("binding_device_class") or "-")]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append(
            "Idle devices are INSIDE every mean (D28): a class average taken over the "
            "busy devices only would report idle silicon as well used."
        )
    else:
        lines.append("(no valid candidate)")
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
    # THE SILICON, cross-checked against the simulator's own composition (D21,
    # D32). The sweep used to compose the shared-digital chiplet BEFORE pricing,
    # so it silently dropped the D31-derived scan engine while the simulator's
    # digital_silicon block carried it — two accountings of one metric that no
    # gate could see, because nothing here compared an area. It does now.
    sim_digital = ((report.get("evaluation") or {}).get("digital_silicon") or {})
    sim_chiplet = (sim_digital.get("shared_digital_chiplet") or {}).get(
        "area_mm2_per_chiplet"
    )
    if sim_chiplet is not None:
        checks.append(
            (
                "shared_digital_area_mm2_per_chiplet",
                selected["silicon"]["shared_digital_area_mm2_per_chiplet"],
                sim_chiplet,
            )
        )
    sim_lanes = sim_digital.get("vector_lanes")
    if sim_lanes is not None and (selected.get("derived_digital") or {}).get(
        "vector_lanes"
    ) is not None:
        checks.append(
            (
                "vector_lanes",
                selected["derived_digital"]["vector_lanes"],
                sim_lanes,
            )
        )

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
        if spec.search == "ladder":
            print(
                f"[QIF DSE] LADDER over "
                + " x ".join(f"{axis}{spec.axes[axis]}" for axis in spec.axes)
                + f" from the declared initializer "
                + ", ".join(f"{k}={_knob_cell(v)}" for k, v in spec.initializer.items())
                + f" ({spec.size} points exist; the walk visits the ones it steps "
                "through and every one of those is evaluated in full through the "
                "real mapped path). Nothing is sampled or estimated."
            )
        else:
            print(
                f"[QIF DSE] {spec.size} candidates over "
                + " x ".join(f"{axis}{spec.axes[axis]}" for axis in spec.axes)
                + "; every one is evaluated through the real mapped path "
                "(fws_mapping -> fws_build -> fws_eval). Nothing is sampled."
            )

    candidates = []
    counter = [0]

    def _announce(candidate):
        if quiet:
            return
        knobs = ", ".join(f"{k}={_knob_cell(v)}" for k, v in candidate["knobs"].items())
        if candidate["ok"]:
            bill = candidate.get("state_bill") or {}
            state = (
                f", D={int(bill['resident_streams'])}, "
                f"{bill['max_stage_state_bytes'] / 1024 ** 2:.1f} MiB/stage"
                if bill.get("measured")
                else ""
            )
            print(
                f"[QIF DSE] {candidate['id']} ({knobs}): "
                f"{headline_text(candidate)}, "
                f"{_fmt(candidate['silicon']['total_silicon_mm2'])} mm2, "
                f"{candidate['placement']['analog_chips']} chips{state} "
                f"[{candidate['wall_s']:.1f} s]"
            )
        else:
            print(
                f"[QIF DSE] {candidate['id']} ({knobs}): INFEASIBLE at stage "
                f"{candidate['fail_stage']} [{candidate['wall_s']:.1f} s] "
                f"{candidate['fail_message']}"
            )

    def _evaluate(point):
        cand_id = f"c{counter[0]:03d}"
        counter[0] += 1
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
        return candidate

    started = time.time()
    ladder_trail = None
    if spec.search == "ladder":
        candidates, ladder_trail = ladder_candidates(spec, _evaluate, announce=_announce)
    else:
        for point in spec.points:
            candidate = _evaluate(point)
            candidates.append(candidate)
            _announce(candidate)
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

    # TOOL-LEVEL disclosures, first in the list because they are properties of
    # the SWEEP rather than of any candidate's timeline. A ladder is a walk, not
    # an enumeration: the front it reports is the non-dominated set of the
    # points it VISITED, which is a lower bound on the front of the declared
    # space. Saying so is the difference between a frontier and a claim.
    if spec.search == "ladder":
        disclosures.insert(
            0,
            {
                "constraint": "front_is_over_the_visited_points",
                "value": (
                    f"{len(candidates)} of {spec.size} declared points were visited"
                ),
                "reason": (
                    "the ladder is a DIRECTIONAL walk from the declared initializer: "
                    "it prices every point it steps to and every one-rung neighbour of "
                    "those, and it never touches the rest of the cross product. The "
                    "reported front is therefore the non-dominated set of the VISITED "
                    "points and is a lower bound on the front of the whole declared "
                    "space. Nothing unvisited is estimated, interpolated or sampled - "
                    "it is simply absent, and the ladder trail says which steps the "
                    "walk took and why it stopped."
                ),
                "candidates": [cand["id"] for cand in valid],
            },
        )
    if valid:
        disclosures.insert(
            0,
            {
                "constraint": "digital_provisioning_is_not_an_axis",
                "value": (
                    "refused axes: " + ", ".join(sorted(REFUSED_AXES))
                    + "; the engine is DERIVED at every point"
                ),
                "reason": "; ".join(
                    REFUSED_AXES[axis] for axis in sorted(REFUSED_AXES)
                ),
                "candidates": [cand["id"] for cand in valid],
            },
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
        "knee": knee_of(rankable, front_ids),
        "ladder_trail": ladder_trail,
        "state_bill_summary": _state_bill_summary(candidates, spec),
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
    knee = payload.get("knee") or {}
    print(f"[QIF DSE] knee: {knee.get('point') or 'NONE'} — {knee.get('basis', '-')}")
    bill = payload.get("state_bill_summary") or {}
    if bill.get("points_measured"):
        print(
            f"[QIF DSE] state bill (D29): worst-stage residency "
            f"{bill['max_stage_state_bytes_min'] / 1024 ** 2:.5g} to "
            f"{bill['max_stage_state_bytes_max'] / 1024 ** 2:.5g} MiB over "
            f"{bill['points_measured']} priced point(s); "
            f"{len(bill.get('infeasible_by_state') or ())} point(s) refused by "
            f"residency, {len(bill.get('infeasible_by_memory') or ())} by memory."
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
