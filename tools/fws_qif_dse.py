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

* **the platform box** (``macro_z``, ``macro_y``) on a card that DERIVES its
  numbers from the ADC-macro sheet (``cim.cards.<card>.calc_profile``). The
  macro is X x Y : Z and the fold ``mux = Y/Z`` is DERIVED (D35-REVISED), so
  the sweep walks the two CONSTRAINT-NATIVE coordinates and lets the fold fall
  out: Z is converter count / output interface / calibration, Y is wordline
  length / tile granularity. Holding Y and halving Z halves the ADC bill and
  doubles ``vec_cycles`` — and with it the beat and the per-vector energy.
  On a hand-declared card both axes are REFUSED BY NAME: a hand-pinned card
  cannot be macro-swept honestly, because its area and energy would stay at
  the macro a human wrote them for. ``adc_capacity_mux`` is refused outright
  (:data:`REFUSED_AXES`): sweeping a derived quantity walks a diagonal through
  the legal box and hides its corners. ``analog_clock_mhz`` is refused too:
  the sheet's 100 MHz is the operating point every analog constant was
  measured at.
* **the fill fraction** (``pp_util``, D38) — the axis that REPLACED the
  partition. Stage granularity no longer trades anything because it no longer
  varies: the machine is always maximally disaggregated (finest
  motif-compatible stages, local microbatch 1, everything time-matched by area
  sizing), so ``layers_per_chip`` and ``layers_per_stage`` are refused BY NAME
  and the stage plan DERIVES. What is still a choice is how full that pipeline
  is run — resident streams = ``max(1, round(pp_util * stages))`` — which
  trades throughput (linear in the fill) against state and KV area (linear in
  the resident streams). Usually dominated; for a state-limited large LLM it
  is the escape valve, and it is priced and disclosed like any other point.

**Chip capacity is NOT an axis.** ADJ-13 retired ``arrays_per_chip``: a chip
offers exactly the macro slots its layers need and that number is DERIVED from
the placement. Under D35-REVISED/D37 the packer fills chips in pipeline order
to a capacity the macro footprint sets, so the slot budget is arithmetic and
the single tail is reported rather than swept.

**Digital provisioning is NOT an axis.** D31/ADJ-9 retire the scan/vector engine
as a declared or swept knob: it is DERIVED per point — sized UP until every
stage's digital per-stage time fits that stage's own ANALOG m-pass time — and
D32 prices it (and the per-macro pools) from the measured synthesis library.
ADJ-10 extends that to EVERY digital engine whose width is a composition of
MEASURED blocks: the attention fabric's ``num_arrays`` (integer copies of the
measured GEMMINI 32x32 block) and the softmax pipeline's ``softmax_lanes`` are
derived against the same per-stage target and reported. All three live in
:data:`REFUSED_AXES` and a sweep that declares one is REFUSED BY NAME at parse.
What is left on the sweep is what ADJ-10 asks for: only knobs with a genuine
physical trade. ``rows`` and ``cols`` per array are neither swept nor derived —
they are the geometry the systolic closed form was validated at, and changing
them is a card change a human makes (ADJ-4).

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
      label: "Granite-4.0-H-Tiny: the platform box x the fill fraction"
      objective: throughput          # or min_silicon
      max_chips: 0                   # 0 = unbounded
      search: ladder                 # or cross_product (the default)
      repair_axis: macro_y           # ladder only: retry a refused step here
      max_silicon_mm2: 0             # 0 = unbounded; the ONLY feasibility budget
      initializer:                   # required by `ladder`: one declared rung per axis
        macro_z: 128
        macro_y: 2048
        pp_util: 1.0
      axes:
        macro_z: [64, 128]           # Z, the ADC count      (pow2 8..X)
        macro_y: [2048, 4096]        # Y, total CTT columns  (pow2 X..8X, Y >= Z)
        pp_util: [1.0, 0.5]          # THE FILL FRACTION (D38)
        # layers_per_chip: [...]     # REFUSED BY NAME (D38: the partition is intent)
        # layers_per_stage: [...]    # REFUSED BY NAME (D38: granularity derives)
        # bank_depth: [...]          # REFUSED BY NAME (D38: parked)
        # column_sets_per_tile: [...] # REFUSED BY NAME (D38: parked, bank_depth's alias)
        # conv_placement: [...]      # REFUSED BY NAME (D38: always shared digital)
        # shared_chiplets: [...]     # REFUSED BY NAME (D36: derived, linearity-checked)
        # arrays_per_chip: [...]     # REFUSED BY NAME (ADJ-13)
        # adc_capacity_mux: [...]    # REFUSED BY NAME (D35-REVISED: mux is derived)
        # vector_lanes: [...]        # REFUSED BY NAME (D31/ADJ-9)
        # num_arrays: [...]          # REFUSED BY NAME (ADJ-10)
        # softmax_lanes: [...]       # REFUSED BY NAME (ADJ-10)

Axis -> the config field it moves (one field each, never two):
  ``macro_z``               ``cim.cards.<analog card>.params.adcs``
  ``macro_y``               ``cim.cards.<analog card>.params.cols``
  ``pp_util``               ``mapping.pp_util``

That is the whole table. Every other spelling this tool has ever offered is in
:data:`REFUSED_AXES` with the decision that took it, because an axis that was
real until a decision retired it has to be spellable far enough to be REFUSED:
a config written against the old law gets a sentence naming the law that moved
and the axis that replaced it, never a silent no-op.

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

What every run stamps and every candidate carries (D33)
--------------------------------------------------------
D33 removes the atlas's privileged point: the FRONTIER is the entry and the
point a reader clicks binds the whole page. Two things follow, and both are
this tool's to emit.

**ONE RUN IDENTITY.** ``run_identity`` takes a content hash over the emitter's
version, the sources that price a candidate, both input configs, the mode, the
model id, the objective and the sweep block. It rides ``dse_report.json`` at
top level (``run_id`` plus the whole ``run`` block), the header comment of the
emitted config, and every per-front-point atlas document. A page assembled from
artifacts whose ids differ was assembled from two generations, and now it can
be told.

**ENOUGH ON EVERY CANDIDATE TO BIND A PAGE.** Not just the selected point:

* ``binding_term`` — the evaluator's own beat-setting stage and its terms by op
  block, plus a per-stage roll-up ``{stage, analog_time_s, engine_time_s,
  fabric_time_s}``: where the beat goes.
* ``state_bill.per_stage[]`` — the per-stage state rows with the store each
  stage buys, present (and empty, with a reason) even when nothing was measured.
* ``pipeline.stage_count`` (the count) and ``pipeline.stages[]`` (the compact
  stage MAP: chips, layers, block census over the placed tiles).
* ``derived_digital`` on EVERY candidate, including cards that name no
  synthesis library — the block is then present with ``composed: false`` and its
  absences named, rather than missing.
* ``binding_device_class`` (what SETS THE BEAT) and ``busiest_device_class``
  (D28's peak max-occupancy). One name used to carry both accountings and they
  disagreed on every point of the shipped curves.

**A MAP PER CANDIDATE.** Every candidate that BUILDS A MACHINE writes a full
``fws_atlas/1`` document to ``<output_dir>/points/<candidate_id>.json``, stamped
with the run id, and carries ``atlas_point`` in the report. Not the front — every
point, because the atlas has no privileged point and clicking any of them must
show everything. The document is written INSIDE
:func:`evaluate_candidate`, from the mapping and evaluation that are still in
scope, so a map costs an export and not a second pricing pass; there is
therefore no cap, no sampling and no truncation note. An INFEASIBLE candidate
built no machine, so it carries ``atlas_point: null`` beside its own
``fail_stage`` and ``fail_message`` — a refusal with a reason, not an absence.
``--no-atlas-points`` turns the whole thing off. These are heavy generated
artifacts and ``docs/qif/.gitignore`` ignores ``dse/*/points/``.

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
import datetime
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import time
from collections import OrderedDict
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
#: ``vector_lanes``, ``num_arrays`` and ``softmax_lanes`` are NOT here: ADJ-9
#: and ADJ-10 refuse all three by name (:data:`REFUSED_AXES`), and an axis this
#: table lists is an axis the tool offers to sweep.
AXIS_TARGETS = {
    # THE PLATFORM BOX (D35-REVISED, 2026-08-29). The macro is X x Y : Z and
    # the fold mux = Y/Z is DERIVED, so the sweep walks the two axes the
    # PHYSICAL CONSTRAINTS are native to and lets the fold fall out:
    #
    #   macro_z — the ADC count. Converter area, the output interface and the
    #     calibration burden are all counted in Z. Powers of two 8..X: never
    #     more converters than there are rows to feed them.
    #   macro_y — total CTT columns. Wordline length and tile granularity are
    #     counted in Y. Powers of two X..8X, and Y >= Z always (mux = 1 legal).
    #
    # Sweeping the FOLD instead would let an absurd corner hide inside a
    # derived quantity, which is why adc_capacity_mux is refused BY NAME
    # (:data:`REFUSED_AXES`). Both axes are legal only on a card that DERIVES
    # its numbers (cim.cards.<card>.calc_profile): on a hand-declared card they
    # would move the label and leave the declared area and energy at the old
    # macro's values.
    "macro_z": "cim.cards.<analog card>.params.adcs",
    "macro_y": "cim.cards.<analog card>.params.cols",
    # THE FILL FRACTION (D38, 2026-08-29). This is the axis that REPLACED the
    # partition. D38 retired layers_per_stage and layers_per_chip because the
    # machine is always maximally disaggregated — finest motif-compatible
    # stages, local microbatch 1, everything time-matched by area sizing — so
    # stage granularity DERIVES and is not a choice anybody makes. What IS a
    # choice is how FULL that pipeline is run: resident streams =
    # max(1, round(pp_util * stages)), throughput linear in the fill and the
    # per-stage state/KV bill linear in the resident streams. Usually
    # dominated; for a state-limited large LLM it is the escape valve. The
    # values are closed (config.PP_UTIL_VALUES) because a free fraction would
    # let a sweep dial throughput to any number and call it a design point.
    "pp_util": "mapping.pp_util",
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
#: The macro spellings D35-REVISED refuses, each pointing at what replaced it.
#: A sweep does not get to rename the thing it is trying to move: ``rows`` is
#: the platform-FIXED height X, ``cols_adc``/``adc_mux``/``adc_capacity_mux``
#: are the retired schema and the retired fold, and the two axes that exist are
#: ``macro_z`` and ``macro_y``. All of them are refused BY NAME and none is
#: left as an "unknown axis" the reader has to interpret.
MACRO_GEOMETRY_AXES = (
    "rows",
    "cols",
    "cols_adc",
    "stored_cols",
    "macro_rows",
    "macro_cols",
    "array_rows",
    "array_cols",
    "macro_size",
    "adc_mux",
    "adc_capacity_mux",
    "mux",
)

#: The one refusal every retired macro spelling raises (D35-REVISED).
MACRO_GEOMETRY_REFUSAL = (
    "D35-REVISED (GEORGE, 2026-08-29) closes the sweepable list around the macro. The "
    "macro is X x Y : Z — X rows (height, the analog sheet's constrained dimension), Y "
    "total CTT columns (the storage tile), Z ADCs (the FLOP producers; muxed columns "
    "are storage) — and mux = Y/Z is DERIVED NOTATION, never an input and never an "
    "axis. X is FIXED per platform, so it is not swept under any spelling, including "
    "the retired rows-equals-hidden convention: a machine that changes shape per "
    "workload is not a buildable chiplet platform, and it quietly flattered every "
    "comparison because a macro cut to fit the matrix never pays a tiling remainder. "
    "THE TWO AXES THAT EXIST ARE macro_z (Z, powers of two 8..X) and macro_y (Y, "
    "powers of two X..8X, with Y >= Z). They are the CONSTRAINT-NATIVE coordinates: "
    "converter count, output interface and calibration are counted in Z; wordline "
    "length and tile granularity are counted in Y. Total system ADC throughput is "
    "cells/(X*mux) — Y cancels — so mux is the right density LABEL and the right plot "
    "LENS, and it is printed on every point; but sweeping it would walk a diagonal "
    "through the legal box and let an absurd corner hide inside a derived quantity. "
    "Weights TILE onto the fixed macro (ceil row/col splits, the priced P7.2 "
    "K-accumulator) and the dimension-mismatch remainder is D27's reported waste "
    "metric, never hidden."
)

REFUSED_AXES = {
    "analog_clock_mhz": (
        "George 2026-08-27: mux only; the analog clock stays at the sheet's 100 MHz. "
        "The clock is not a free number on this device: it is a MEASURED operating "
        "point of the ADC-macro sheet this repo ports (cim_analog_calc, "
        "OPTIMA_CALC_PROFILES['george_cima_mux1'] analog_clock_MHz = analog_clock_MAX "
        "= 100), and the settle/compute times, currents and the ADC FoM that price a "
        "slice were all taken at it. OPTIMA's own mux sweep (perf_model/"
        "run_vla_sweep.py) sweeps pools.analog.mux_degree with the analog clock pinned "
        "for the same reason. Sweeping it here would move every analog number in the "
        "report with nothing behind the move, so it is refused BY NAME rather than "
        "offered. What IS on the sweep is adc_capacity_mux, which trades the same "
        "throughput against area with a sheet behind every point."
    ),
    "arrays_per_chip": (
        "ADJ-13 retires the chip's SLOT COUNT as a swept or declared knob. A chip "
        "offers exactly the macro slots the layers it holds need, and that number is "
        "DERIVED from the placement itself: the sweep places the layers once to learn "
        "the requirement and builds the chip to it. Sweeping it produced garbage "
        "points at both ends — a capacity under the requirement is refused by the "
        "assembly check and tells you nothing, and a capacity over it enumerates "
        "empty slots that are charged as silicon, so the 'frontier' bought area with "
        "no throughput and called it a design point. What remains free is the "
        "PARTITION (layers_per_chip / layers_per_stage), which is a real choice; the "
        "capacity that partition implies is arithmetic."
    ),
    "num_arrays": (
        "ADJ-10 retires the attention fabric's ARRAY COUNT as a swept or declared "
        "knob, for the same reason D31/ADJ-9 retired the scan engine's lane count: it "
        "is a count of MEASURED synthesis blocks, so it is DERIVED — sized UP in "
        "integer copies of the measured GEMMINI 32x32 block until every stage's "
        "attention time fits that stage's own ANALOG m-pass time — and REPORTED "
        "(evaluation.digital_silicon.derived_fabric_sizing). Sweeping it would sweep "
        "the answer. Only the COUNT is derived: rows x cols per array stay as measured "
        "and as declared, because inventing array geometry is refused (ADJ-4). An "
        "explicit cim.cards.<card>.fabric_num_arrays on a SINGLE machine is still an "
        "OVERRIDE that rides its own disclosure; what is refused here is making it an "
        "AXIS."
    ),
    "softmax_lanes": (
        "ADJ-10 retires the softmax pipeline's LANE COUNT as a swept or declared knob "
        "on the same grounds as num_arrays: the lane is a MEASURED per-lane census "
        "(OPTIMA's softmax collection, block for block), so the width is DERIVED "
        "against the same per-stage analog m-pass target and REPORTED. "
        "cim.cards.<card>.fabric_softmax_lanes pins ONE machine with a disclosure; as "
        "an AXIS it is refused."
    ),
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

# D35: every spelling of the macro's shape, refused by name with the decision
# quoted. Written as one loop over one message because there is ONE reason and
# restating it per spelling would be the same law written ten times.
REFUSED_AXES.update({name: MACRO_GEOMETRY_REFUSAL for name in MACRO_GEOMETRY_AXES})

#: THE AXES D36 AND D38 RETIRED (2026-08-29), each quoting the decision that
#: took it and naming what replaced it.
#:
#: These were REAL axes with real trades right up until the decision that
#: retired them, which is exactly why they are refused BY NAME rather than
#: dropped from the table: a sweep that still declares one was written against
#: the old law, and the sentence it gets back has to say which law moved and
#: what to sweep instead. Silently ignoring them would sweep something else;
#: leaving them live would let a config buy a number the machine no longer
#: offers.
REFUSED_AXES.update({
    "layers_per_stage": (
        "D38 (GEORGE, 2026-08-29) RETIRES layers_per_stage as a sweep axis: 'The "
        "machine is always maximally disaggregated: local microbatch 1, finest "
        "motif-compatible stages, everything time-matched by area sizing (the "
        "derive-to-floor doctrine is what makes 100% utilization possible). Stage "
        "granularity DERIVES.' The partition is INTENT, not a design point — one "
        "layer per stage is the finest the motif admits and it is what the mapper "
        "builds, so there is no partition input anywhere and nothing here to move. "
        "THE AXIS THAT REPLACED IT IS pp_util, the fill fraction: the pipeline's "
        "granularity is settled, how full it is run is not."
    ),
    "layers_per_chip": (
        "D38 (GEORGE, 2026-08-29) RETIRES the partition as a sweep axis, under this "
        "spelling as under layers_per_stage: 'The partition is INTENT, not an axis. "
        "The fill fraction is the axis.' Stage granularity DERIVES (finest "
        "motif-compatible stages, local microbatch 1) and the chip split follows "
        "from the packer filling analog chips in pipeline order, so neither number "
        "is a choice a sweep gets to make. SWEEP pp_util INSTEAD — resident streams "
        "= max(1, round(pp_util * stages)) — which is the trade the partition used "
        "to stand in for: throughput against per-stage state and KV area."
    ),
    "bank_depth": (
        "D38 (GEORGE, 2026-08-29) PARKS bank_depth: 'bank_depth is PARKED for "
        "ground-up reformulation (the mux-slot allocation question, next).' It is "
        "not merely dominated — it won depth 1 nearly everywhere on the old cards "
        "and D35-REVISED put it on probation — it is a knob whose QUESTION is being "
        "restated, so sweeping it now would price rungs against a law that is about "
        "to be replaced. The card freezes depth 1 until the reformulation lands. "
        "There is no replacement axis: what a parked knob asks for is a decision, "
        "not a substitute."
    ),
    "column_sets_per_tile": (
        "D38 (GEORGE, 2026-08-29) PARKS the mux-slot allocation question, and "
        "column_sets_per_tile is that question in its cim.allocation spelling — the "
        "same allocation granularity bank_depth sets, with cim.allocation winning "
        "the precedence. A sweep does not get to rename the thing it is trying to "
        "move: refusing bank_depth while leaving its alias live would park the knob "
        "in name only. Parked with bank_depth, and returning with it when the "
        "ground-up reformulation lands."
    ),
    "conv_placement": (
        "D38 (GEORGE, 2026-08-29) RETIRES conv_placement: 'the conv's work is ALWAYS "
        "on the shared digital chiplet. Not an axis.' The trade this axis priced — a "
        "per-macro pool on every enumerated slot against engine width and beat time "
        "on the shared chiplet — is settled by decision, not by the sweep, so both "
        "rungs no longer exist to walk. cim.allocation.conv_placement still parses "
        "for a single declared machine; what is refused here is making it an AXIS."
    ),
    "shared_chiplets": (
        "D36 (GEORGE, 2026-08-29) DEMOTES shared_chiplets 'from axis to a STANDING "
        "LINEARITY CHECK (price one point at 1x/2x/4x sharing; assert linearity "
        "within a few percent)'. The nonlinearity that once made it a real axis was "
        "fabric saturation, which the context-split law removed, so the rungs now "
        "buy a straight line and a straight line is not a frontier. Digital is a "
        "derived FUNCTION of the analog point (D36 stage 2): the chiplet is derived "
        "per group and the count follows from the packaging band, never from a "
        "sweep. 'If any future law reintroduces a cliff the check fails loudly and "
        "the axis returns.'"
    ),
})

#: Evaluation stages in order; the index is how far a candidate got.
#:
#: ``state`` sits LAST because it needs the priced timeline: every stage of a
#: filled pipeline holds all D streams' recurrent state and KV for its layers.
#: It REPORTS that bill and the store sized to hold it; it refuses nothing. A
#: design is refused by the AREA budget, never by needing memory.
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


#: THE EMITTER'S OWN VERSION (D33). Bumped whenever the SHAPE of what this
#: tool writes changes. It rides the run identity below so an artifact an older
#: emitter wrote is distinguishable from one this build wrote even when the
#: inputs are byte-identical.
TOOL_VERSION = "fws_qif_dse/1.1-d33-emitter"

#: The sources the run id FINGERPRINTS. D33 asks the atlas to band a page whose
#: artifacts came from different generations, and "generation" cannot mean the
#: inputs alone: an emitter change, an evaluator change or a device-law change
#: all move the numbers under identical YAML. These are the files that can move
#: them, so their content is inside the id.
RUN_ID_SOURCES = (
    "tools/fws_qif_dse.py",
    "fws_eval.py",
    "fws_mapping.py",
    "fws_atlas_export.py",
    "cim_timing.py",
)

#: Where the per-candidate documents land, relative to the sweep output dir.
#:
#: There is NO cap on how many are written, and the absence of one is the
#: point (D33). A cap existed while a document cost a second full pricing pass;
#: the export now runs inside the evaluation that already priced the machine,
#: so there is nothing left to ration. Rationing it was the architectural
#: mistake: it made "click any point" mean "click one of twenty".
ATLAS_POINTS_DIR = "points"


def _sha256_of_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_fingerprint():
    """A hash of the sources that decide this run's numbers (:data:`RUN_ID_SOURCES`).

    A missing source is recorded as ``absent`` rather than skipped: a
    fingerprint that silently ignored a file it could not read would call two
    different trees the same generation.
    """
    parts = []
    for relative in RUN_ID_SOURCES:
        path = REPO_ROOT / relative
        parts.append(
            "%s:%s" % (relative, _sha256_of_file(path) if path.exists() else "absent")
        )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def run_identity(hw_path, model_path, *, mode, model_id, objective, sweep_echo):
    """ONE identity for every artifact this run writes (D33).

    The id is a CONTENT HASH, not a uuid, and that is the whole point: two runs
    of the same tool over the same inputs on the same tree ARE the same
    generation and get the same id, so a regenerated artifact still matches its
    siblings. Anything that could move a number moves the id — the emitter's
    own version, the fingerprint of the sources that price a candidate, both
    input files' bytes, the mode, the model id, the objective and the sweep
    block itself.

    ``generated_at`` is a WALL-CLOCK stamp and is deliberately NOT inside the
    hash: it says when this copy was written, and folding it in would make
    every run a new generation and the comparison useless.
    """
    fingerprint = code_fingerprint()
    material = json.dumps(
        {
            "tool": "fws_qif_dse",
            "tool_version": TOOL_VERSION,
            "code_fingerprint": fingerprint,
            "hardware_config": Path(hw_path).read_text(),
            "model_config": Path(model_path).read_text(),
            "mode": str(mode),
            "model_id": str(model_id or ""),
            "objective": str(objective),
            "sweep": _json_safe(sweep_echo),
        },
        sort_keys=True,
        default=str,
    )
    return OrderedDict(
        (
            ("run_id", hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]),
            ("tool", "fws_qif_dse"),
            ("tool_version", TOOL_VERSION),
            ("code_fingerprint", fingerprint),
            (
                "generated_at",
                datetime.datetime.now(datetime.timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z"),
            ),
            (
                "basis",
                "the run id is a sha256 over this tool's version, the content of "
                + ", ".join(RUN_ID_SOURCES)
                + ", both input configs, the mode, the model id, the objective and "
                "the sweep block. EVERY artifact this run writes carries it - the "
                "report at top level, the emitted config in its header comment and "
                "each per-front-point atlas document - so a reader holding two of "
                "them can tell whether they describe one machine or two "
                "generations (D33). generated_at is outside the hash: it is when "
                "this copy was written, not what it was written from."
            ),
        )
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
    """One candidate value for one axis, validated by the axis's own shape.

    Only the axes :data:`AXIS_TARGETS` still offers reach here: a retired or
    refused spelling is stopped by name in :meth:`SweepSpec.from_raw` before
    any value of it is read, so this function never has to guess what a dead
    axis's values were supposed to mean.
    """
    if axis == "pp_util":
        # D38's fill fraction. A CLOSED list of floats, and the config surface
        # owns it (config.PP_UTIL_VALUES) so the sweep and the mapper cannot
        # disagree about which fractions exist. Refused here rather than at the
        # first candidate, because a sweep that only failed on its tenth point
        # would have printed nine rows of a machine it was never going to
        # finish.
        try:
            return config_module.parse_pp_util(raw, context)
        except ValueError as exc:
            raise QifDseUsageError(str(exc)) from exc
    # macro_z / macro_y: integer rungs, checked against the PLATFORM lists in
    # _check_platform_axis_values once the whole axis is parsed.
    value = int(raw)
    if value < 1:
        raise QifDseUsageError(f"{context} must be >= 1 (got {value}).")
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


def _check_platform_axis_values(axes):
    """Every macro_z / macro_y value is one the PLATFORM admits (D35-REVISED).

    The lists live in :mod:`cim_analog_calc` — one place, one law. A value off
    them is refused here rather than at the config layer, because a sweep that
    only failed on the tenth candidate would have printed nine rows first.
    """
    import cim_analog_calc as _calc

    for axis, legal, what in (
        ("macro_z", _calc.MACRO_Z_VALUES, "Z, the ADC count"),
        ("macro_y", _calc.MACRO_Y_VALUES, "Y, the total CTT columns"),
    ):
        for value in axes.get(axis, ()):
            if int(value) not in legal:
                raise QifDseUsageError(
                    f"{DSE_BLOCK}.axes.{axis} declares {value!r}, which is not a value "
                    f"the platform admits for {what}. Legal: {list(legal)} "
                    f"({_calc.PLATFORM_MACRO_DECISION}: Z is powers of two 8..X and Y is "
                    f"powers of two X..{_calc.MACRO_Y_CEILING_MULTIPLE}X, at "
                    f"X = {_calc.PLATFORM_MACRO_ROWS} rows). The 8X ceiling on Y is "
                    "ADJUDICATED, not physical: if a model's front crowds it the "
                    "edge-crowding check says so by name and raising it is a one-line "
                    "change with data behind it."
                )


class SweepSpec:
    """The parsed ``mapping_dse`` block: axes in declaration order."""

    def __init__(self, axes, objective, max_chips, max_silicon_mm2, label,
                 search="cross_product", initializer=None,
                 repair_axis=None):
        self.axes = axes  # OrderedDict-like: {axis: [values]} in declaration order
        self.objective = objective
        self.max_chips = int(max_chips)
        self.max_silicon_mm2 = float(max_silicon_mm2)
        self.label = label
        self.search = search
        self.initializer = initializer or {}
        self.repair_axis = repair_axis

    def illegal_reason(self, point):
        """Why this (Z, Y) corner is not a macro, or None (D35-REVISED).

        The platform box is axis-aligned, so almost every corner is legal —
        but not the ones below the diagonal. Y >= Z always: mux = Y/Z is the
        muxed columns behind ONE converter, and a Y under Z would be an ADC
        with less than one column to digitize. mux = 1 is the legal floor and
        it is a real design point, not an edge case.

        The refusal NAMES the corner rather than dropping it: a point the
        sweep skipped in silence is a hole a reader cannot see.
        """
        z = point.get("macro_z")
        y = point.get("macro_y")
        if z is None or y is None:
            return None
        z = int(z)
        y = int(y)
        if y < z:
            return (
                f"macro_y = {y} < macro_z = {z}: mux = Y/Z would be {y / z:g}, and a "
                "converter cannot digitize less than one column. Y >= Z always "
                "(D35-REVISED); mux = 1 is the legal floor and IS a design point."
            )
        if y % z:
            return (
                f"macro_z = {z} does not divide macro_y = {y}: mux = Y/Z is DERIVED "
                "(D35-REVISED), so a fold with a remainder would strand columns behind "
                "no converter. Both axes are powers of two, so this corner cannot "
                "arise from the declared platform lists."
            )
        import cim_analog_calc as _calc

        mux = y // z
        if mux not in _calc.ALLOWED_CAPACITY_MUX:
            return (
                f"macro_y = {y} against macro_z = {z} derives mux = {mux}, which is "
                f"outside {_calc.ALLOWED_CAPACITY_MUX[0]}.."
                f"{_calc.ALLOWED_CAPACITY_MUX[-1]} — the range the ADC-macro sheet has "
                "MEASUREMENTS for (OPTIMA config/loader.py ALLOWED_MUX_DEGREES). The "
                "declared box is a rectangle and the sheet's coverage is not, so this "
                "corner exists on paper and has no physics behind it. It is named and "
                "skipped rather than extrapolated — which is exactly the kind of absurd "
                "corner D35-REVISED walks the box to make visible."
            )
        return None

    @property
    def points(self):
        """The cross product, in declaration order, minus the illegal corners.

        The ONLY thing skipped is a corner :meth:`illegal_reason` names — the
        sub-diagonal of the (Z, Y) box, which is not a macro at all. Every
        skipped corner is reported by name in the sweep's own output
        (:attr:`illegal_points`), so the hole is a fact and not a silence.
        """
        for point, reason in self.enumerated_points:
            if reason is None:
                yield point

    @property
    def enumerated_points(self):
        """``(point, illegal_reason_or_None)`` over the whole declared box."""
        names = list(self.axes)
        for combo in itertools.product(*(self.axes[name] for name in names)):
            point = dict(zip(names, combo))
            yield point, self.illegal_reason(point)

    @property
    def illegal_points(self):
        """The corners the box declares and the platform refuses, with reasons."""
        return [
            {"knobs": point, "reason": reason}
            for point, reason in self.enumerated_points
            if reason is not None
        ]

    @property
    def size(self):
        """Points the sweep will actually price: the box minus its illegal corners."""
        return sum(1 for _ in self.points)

    @property
    def declared_size(self):
        """Every corner of the declared box, legal or not."""
        total = 1
        for values in self.axes.values():
            total *= len(values)
        return total

    @property
    def initializer_index(self):
        """The initializer as an index vector over the declared axis lists."""
        reason = self.illegal_reason(self.initializer)
        if reason is not None:
            raise QifDseUsageError(
                f"{DSE_BLOCK}.initializer is not a macro the platform admits: {reason} "
                "The ladder steps outward FROM this point, so an illegal start is not a "
                "step the walk could repair — it is refused here, by name."
            )
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
        if "max_stage_state_bytes" in block:
            raise QifDseUsageError(
                f"{DSE_BLOCK}.max_stage_state_bytes is REFUSED BY NAME. A machine is "
                "not infeasible for needing memory: the on-chip state store is SIZED "
                "to the state the mapping holds, composed from integer copies of the "
                "measured SRAM macro, and charged as silicon "
                "(silicon.state_sram_silicon_mm2). Holding more state costs AREA, and "
                "what refuses a design is the area budget "
                f"{DSE_BLOCK}.max_silicon_mm2 — a statement about the package, not "
                "about the memory. Delete this key and declare that budget instead."
            )
        known = (
            "axes",
            "objective",
            "max_chips",
            "max_silicon_mm2",
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
        _check_platform_axis_values(axes)
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


def _sync_analog_fallback(cim, params_block, card_block, source):
    """Keep `cim.analog` on the SAME macro as the card the sweep just moved.

    ONE FILE, ONE MACRO. The card is what the run prices, but `cim.analog` is
    what a reader sees when no card is named and what the emitted config
    carries; leaving it at the base point would put two macros in one file and
    the config layer refuses exactly that. The numbers come from ONE call to
    the same derivation the card uses — this is the sheet answering twice for
    one macro, not a second model of it.
    """
    import cim_analog_calc as _calc

    analog = cim.get("analog")
    if not isinstance(analog, dict):
        return
    rows = int(params_block.get("rows", analog.get("rows")))
    cols = int(params_block.get("cols", analog.get("cols")))
    adcs = int(params_block.get("adcs", analog.get("adcs")))
    profile = str(card_block.get("calc_profile") or "").strip()
    try:
        derived = _calc.derive_analog_card(rows, cols, adcs, profile=profile, context=source)
    except _calc.AnalogCalcError as exc:
        raise QifDseUsageError(str(exc)) from exc
    analog["rows"] = rows
    analog["cols"] = cols
    analog["adcs"] = adcs
    analog["slice_cycles"] = int(derived.slice_cycles)
    analog["analog_clock_mhz"] = float(derived.analog_clock_mhz)
    analog["energy_per_vec_pj"] = float(derived.energy_per_vec_pj)
    analog["area_mm2_per_array"] = float(derived.area_mm2_per_array)


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
        if axis in ("macro_z", "macro_y"):
            card_block = _card("analog", analog_card)
            if not str(card_block.get("calc_profile") or "").strip():
                raise QifDseUsageError(
                    f"the sweep moves cim.cards.{analog_card}.params."
                    f"{'adcs' if axis == 'macro_z' else 'cols'} ({axis}), but that card "
                    f"in {source} names no `calc_profile`, so it is HAND-PINNED: a "
                    "hand-pinned card cannot be macro-swept honestly. Its "
                    "area_mm2_per_array, energy_per_vec_pj and slice_cycles are numbers "
                    "a human wrote for ONE macro; moving Y or Z would relabel the macro "
                    "and leave all three at the old one's values, so every point past "
                    "the first would price silicon that does not exist. Give the card a "
                    "calc_profile (the numbers are then DERIVED from the ADC-macro sheet "
                    "at each (X, Y, Z)) or drop the axis."
                )
            params_block = card_block.get("params")
            if not isinstance(params_block, dict):
                raise QifDseUsageError(
                    f"the sweep moves cim.cards.{analog_card}.params, but that card in "
                    f"{source} declares no `params` block. A derived card states the "
                    "MACRO it is (rows X, cols Y, adcs Z) and the profile prices it."
                )
            params_block["adcs" if axis == "macro_z" else "cols"] = int(value)
            _sync_analog_fallback(cim, params_block, card_block, source)
        elif axis == "pp_util":
            mapping_block = raw.get("mapping")
            if not isinstance(mapping_block, dict):
                raise QifDseUsageError(
                    f"the sweep moves mapping.{axis}, but {source} declares no "
                    "`mapping:` block. A mapped sweep needs a mapped config: the "
                    "block is what makes P3 place and P4 price the run (ADJ-5/ADJ-6)."
                )
            mapping_block[axis] = float(value)
        else:  # pragma: no cover - AXIS_TARGETS is the gate
            raise QifDseUsageError(f"unhandled axis {axis!r}")
    return raw


# ---------------------------------------------------------------------------
# ADJ-13 — the machine sizes itself
# ---------------------------------------------------------------------------


def _slots_needed(mapping):
    """The macro slots the busiest chip of this placement actually needs.

    Counted as the highest slot INDEX a chip uses, plus one — not the number of
    macros holding tiles — because a chip must offer every slot up to the last
    one it fills, and a gap is still a slot the chip has to have.
    """
    per_chip = {}
    counts = {}
    for macro in mapping.macros:
        if not getattr(macro, "tiles", None):
            continue
        chip_id = getattr(macro, "chip_id", None)
        if chip_id is None:
            chip_id = getattr(macro, "chip", None)
        slot = getattr(macro, "slot", None)
        if slot is None:
            slot = counts.get(chip_id, 0)
            counts[chip_id] = slot + 1
        per_chip[chip_id] = max(per_chip.get(chip_id, 0), int(slot) + 1)
    return max(list(per_chip.values()) or [1])


def _derive_chip_capacity(raw, model_config, model_id, hw_from_raw):
    """Build the chip to the placement, instead of the placement to the chip.

    Places the layers once against a capacity that cannot bind, reads what the
    busiest chip needed, and writes that back as the chip's slot count. The
    machine then enumerates exactly the slots its weights occupy, so no design
    point is paying for empty silicon it never asked for.
    """
    import copy as _copy
    import fws_mapping

    probe_raw = _copy.deepcopy(raw)
    chip = probe_raw.setdefault("cim", {}).setdefault("chip", {})
    # The probe capacity is an UPPER BOUND, not a large number: every array the
    # whole model needs, which no single chip can exceed however the layers are
    # partitioned. A merely "big" capacity would make the placement enumerate
    # millions of empty macro slots before it could be asked what it needed.
    from cim_timing import CimDeviceModel

    probe_hw = hw_from_raw(probe_raw)
    inner = getattr(model_config, "model_config", model_config)
    bound = int(CimDeviceModel(probe_hw, inner).total_arrays())
    chip["arrays_per_chip"] = max(1, bound)
    probe_hw = hw_from_raw(probe_raw)
    probe = fws_mapping.build_mapping(probe_hw, model_config, model_id=model_id)
    needed = _slots_needed(probe)
    out = _copy.deepcopy(raw)
    out.setdefault("cim", {}).setdefault("chip", {})["arrays_per_chip"] = int(needed)
    # THE SLOT COUNT HAS TWO LEGAL HOMES and the config may declare either.
    # Writing only `cim.chip.arrays_per_chip` left a stale
    # `mapping.macros_per_chip` behind, and the very next validation refused the
    # candidate for a disagreement this function had just created - blaming the
    # user for the tool's own edit. If the config declares the mapping-side
    # field, it is derived too.
    mapping_block = out.get("mapping")
    if isinstance(mapping_block, dict) and mapping_block.get("macros_per_chip") is not None:
        mapping_block["macros_per_chip"] = int(needed)
    return out, int(needed)


def _digital_silicon_of(evaluation):
    """``fws_eval``'s own digital block for this evaluation, or ``{}``.

    Taken ONCE per candidate and handed to every consumer below. It used to be
    recomputed by each of them, which is three compositions of one quantity
    under one name — the shape D21 exists to stop — and it is not cheap.
    """
    import fws_eval as _fws_eval

    try:
        return _fws_eval._digital_silicon_block(evaluation) or {}
    except Exception:
        return {}


def _pipeline_stage_rows(mapping):
    """The compact per-stage MAP of the filled pipeline (D29's stage plan).

    One row per PP stage: which chips carry it, which layers sit on it, and
    what those layers put on analog silicon, counted by op block. It is the
    CHEAP summary that rides the report row itself, so a frontier plot can draw
    a point's pipeline without opening that point's full inventory
    (:func:`export_point_atlas` writes one of those per candidate too, and they
    are megabytes each).

    The block census is over PLACED TILES, so it is the machine's own inventory
    and not a re-derivation of the model: a stage's ``blocks`` are the op names
    of the tile owners whose layer this stage holds, counted.
    """
    stage_of_layer = {}
    for stage in mapping.stages:
        for layer in stage.layers:
            stage_of_layer[int(layer)] = int(stage.index)
    blocks_by_stage = {}
    owners_by_stage = {}
    unassigned = 0
    for tile in mapping.tiles:
        index = stage_of_layer.get(int(tile.owner.layer))
        if index is None:
            # A TILE ON NO STAGE. The stage plan is a partition of the BACKBONE
            # layers, so under D30 (embedding and lm_head dropped on the mapped
            # path) there should be none. It is counted rather than dropped:
            # the row census below has to close against the placement, and a
            # silently skipped tile would make it close by losing evidence.
            unassigned += 1
            continue
        blocks = blocks_by_stage.setdefault(index, {})
        key = str(tile.owner.op)
        blocks[key] = blocks.get(key, 0) + 1
        owners_by_stage.setdefault(index, set()).add(tile.owner.label)
    rows = []
    for stage in mapping.stages:
        index = int(stage.index)
        blocks = blocks_by_stage.get(index, {})
        rows.append(
            OrderedDict(
                (
                    ("stage", index),
                    ("label", str(stage.label)),
                    ("chips", [int(chip) for chip in stage.chips]),
                    ("layers", len(stage.layers)),
                    ("layer_ids", [int(layer) for layer in stage.layers]),
                    ("tiles", int(sum(blocks.values()))),
                    ("owners", len(owners_by_stage.get(index, ()))),
                    (
                        "blocks",
                        OrderedDict(
                            sorted(blocks.items(), key=lambda kv: (-kv[1], kv[0]))
                        ),
                    ),
                )
            )
        )
    return rows, unassigned


def _binding_term_block(evaluation, digital):
    """WHERE THE BEAT GOES on this candidate (ADJ-10), stage by stage.

    Two things under one key, because they answer one question:

    * the evaluator's own ``binding_term`` — the stage with the longest span on
      the PRICED slice and its terms by op block, mirrored here verbatim so the
      sweep publishes no second version of it (D21);
    * a per-stage roll-up of the two DERIVED sizings, one row per stage:
      ``{stage, analog_time_s, engine_time_s, fabric_time_s}``. The engine and
      the fabric are sized against the SAME per-stage analog m-pass time
      (D31/ADJ-9 and ADJ-10), so the three columns are one comparison and a
      reader can see which stages the analog floor really binds.

    A run whose card composes nothing derives nothing, so the per-stage roll-up
    is EMPTY and says why rather than printing zeros.
    """
    binding = dict(getattr(evaluation, "binding_term", None) or {})
    engine_rows = ((digital or {}).get("derived_engine_sizing") or {}).get(
        "per_stage"
    ) or []
    fabric_rows = ((digital or {}).get("derived_fabric_sizing") or {}).get(
        "per_stage"
    ) or []
    rows = {}

    def _row(stage):
        return rows.setdefault(
            int(stage),
            OrderedDict(
                (
                    ("stage", int(stage)),
                    ("analog_time_s", None),
                    ("analog_time_source", None),
                    ("engine_time_s", None),
                    ("fabric_time_s", None),
                    ("digital_time_s", None),
                    ("analog_bound", None),
                )
            ),
        )

    for source, entries, key in (
        ("derived_engine_sizing", engine_rows, "engine_time_s"),
        ("derived_fabric_sizing", fabric_rows, "fabric_time_s"),
    ):
        for entry in entries:
            row = _row(entry["stage"])
            row[key] = float(entry.get("time_s") or 0.0)
            analog = float(entry.get("analog_time_s") or 0.0)
            if row["analog_time_s"] is None:
                row["analog_time_s"] = analog
                row["analog_time_source"] = source
            elif abs(row["analog_time_s"] - analog) > 1e-15 * max(
                1.0, abs(analog)
            ):
                # TWO READINGS OF ONE QUANTITY. Both sizings measure the SAME
                # per-stage analog m-pass, so a disagreement is a finding, not
                # a rounding: it is carried under its own name instead of one
                # of the two silently winning (D21).
                row["analog_time_s_%s" % source] = analog
    for row in rows.values():
        engine = row["engine_time_s"] or 0.0
        fabric = row["fabric_time_s"] or 0.0
        row["digital_time_s"] = engine + fabric
        if row["analog_time_s"] is not None:
            row["analog_bound"] = bool(row["analog_time_s"] >= row["digital_time_s"])
    ordered = [rows[key] for key in sorted(rows)]
    block = OrderedDict(
        (
            ("measured", bool(binding)),
            ("stage", binding.get("stage")),
            ("span_s", binding.get("span_s")),
            ("analog_time_s", binding.get("analog_time_s")),
            ("analog_is_largest_term", binding.get("analog_is_largest_term")),
            ("terms", list(binding.get("terms") or ())),
            ("basis", binding.get("basis", "")),
            ("per_stage", ordered),
        )
    )
    block["per_stage_basis"] = (
        "one row per stage from evaluation.digital_silicon.derived_engine_sizing "
        "and .derived_fabric_sizing: the DERIVED scan/vector engine's per-stage "
        "time, the DERIVED attention fabric's per-stage time, and the stage's own "
        "analog m-pass time that both were sized against (D31/ADJ-9, ADJ-10). "
        "analog_bound is analog_time_s >= engine_time_s + fabric_time_s - the two "
        "digital engines share one chiplet queue, so the comparison is against "
        "their SUM."
        if ordered
        else "EMPTY, and not a measured zero: this candidate's card composes no "
        "synthesis library, so D31/ADJ-10 derived neither engine and there are no "
        "per-stage sizing rows to roll up. The evaluator's binding_term above is "
        "still measured - it is read off the priced timeline, not off a "
        "composition."
    )
    if not binding:
        block["reason"] = (
            "NOT MEASURED, and not a zero: fws_eval reads the binding term off the "
            "ops inside its MEASUREMENT SLICE (the last lowered beat under the "
            "filled pipeline, the last lowered DECODE STEP under the retired "
            "lockstep regime), and this run put no op in that slice - a "
            "prefill-only workload lowered under lockstep has no decode step at "
            "all. No stage therefore has a span to be the longest and nothing "
            "binds by measurement here. What is still measured is the per-class "
            "utilization: see busiest_device_class."
        )
    return block


def _derive_shared_chiplets(evaluation, stages_total, digital=None):
    """As few digital chiplets as the beat admits — a BALANCED pipeline.

    A shared chiplet carries the digital work of every stage it serves, one
    stage after another, inside one beat. So the question is a packing one:
    with each stage's measured digital busy time in hand, how few chiplets can
    hold all of them without any chiplet running past the beat. First-fit over
    the stages sorted heaviest-first answers it, and the answer is reported
    with the per-chiplet load that produced it rather than as a bare count.

    Returns (count, detail) — or (None, reason) when the timeline does not
    measure the per-stage digital work, in which case the caller keeps the
    declared count and says so.
    """
    # The per-stage digital sizing lives in the report's digital_silicon block,
    # which fws_eval composes from the evaluation rather than storing on it.
    # Asking the evaluation object directly quietly returned nothing, and the
    # derivation then never ran at all. The block is composed ONCE per candidate
    # by the caller (_digital_silicon_of) and handed in here.
    digital = _digital_silicon_of(evaluation) if digital is None else (digital or {})
    engine = (digital.get("derived_engine_sizing") or {}).get("per_stage") or []
    fabric = (digital.get("derived_fabric_sizing") or {}).get("per_stage") or []
    beat = float((getattr(evaluation, "pipeline", None) or {}).get("beat_s", 0.0) or 0.0)
    if beat <= 0:
        return None, "the run measures no beat, so nothing can be packed against it"
    load = {}
    for row in engine:
        load[int(row["stage"])] = load.get(int(row["stage"]), 0.0) + float(row.get("time_s", 0.0) or 0.0)
    for row in fabric:
        load[int(row["stage"])] = load.get(int(row["stage"]), 0.0) + float(row.get("time_s", 0.0) or 0.0)
    if not load:
        return None, "this run prices no shared-digital work, so it needs no chiplet"
    order = sorted(load.items(), key=lambda kv: -kv[1])
    bins = []          # each bin: [busy_s, [stage, ...]]
    for stage, busy in order:
        placed = False
        for slot in bins:
            if slot[0] + busy <= beat:
                slot[0] += busy
                slot[1].append(stage)
                placed = True
                break
        if not placed:
            bins.append([busy, [stage]])
    # The duty of the chiplets the machine ACTUALLY BUILDS: one per stage, so
    # each carries exactly its own stage's digital work.
    per_stage_duty = sorted((busy / beat) for busy in load.values())
    detail = {
        "beat_s": beat,
        "built_duty_max": per_stage_duty[-1] if per_stage_duty else None,
        "built_duty_mean": (sum(per_stage_duty) / len(per_stage_duty)) if per_stage_duty else None,
        "chiplets": len(bins),
        "stages_served": [sorted(slot[1]) for slot in bins],
        "busy_s": [slot[0] for slot in bins],
        "duty": [(slot[0] / beat) for slot in bins],
        "stages_with_digital_work": len(load),
        "stages_total": int(stages_total),
        "basis": (
            "first-fit over the stages sorted heaviest-first: a chiplet carries the "
            "measured digital busy time of every stage it serves, and may not run "
            "past one beat. The count is what falls out, and the per-chiplet duty "
            "below is how close each one runs."
        ),
    }
    return len(bins), detail


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


def _silicon(mapping, chiplets, state_bytes=None):
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
    # The ON-CHIP STATE STORE. A filled pipeline holds every stream's recurrent
    # state and KV while it decodes, and that store is silicon like any other.
    # It is SIZED to the state the mapping measured — never assumed, never a
    # constraint to fail against — so a plan that holds more state buys more
    # SRAM and pays for it here. `state_bytes` is None on the provisional call
    # (before pricing there is no bill), and the term is then absent rather
    # than guessed.
    # `state_bytes` is a LIST, one entry per pipeline stage: each stage gets its
    # own store because each sits on its own chip. A bare number is accepted as
    # one store, for the callers that hold only a total.
    state_store = None
    state_sram = 0.0
    state_macros = 0
    bills = None
    if state_bytes is not None:
        bills = ([float(b) for b in state_bytes]
                 if isinstance(state_bytes, (list, tuple)) else [float(state_bytes)])
    if bills is not None and device.has_synthesis_library():
        stores = [device.state_sram_composition(b) for b in bills if b > 0]
        state_store = stores[0] if stores else None
        state_sram = math.fsum(float(store.area_mm2) for store in stores)
        state_macros = int(sum(sum(store.blocks.values()) for store in stores))
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
    # THE STATE STORE, WHEN IT COULD NOT BE BUILT. `state_sram_provenance` said
    # "absent" while uncovered_terms stayed empty, so a total that omits the
    # store read as complete. A workload that holds no resident state at all
    # (a ViT encoder decodes nothing) is a measured zero and needs no row; a
    # store that could not be COMPOSED is an absent law and gets one.
    if bills is not None and math.fsum(bills) > 0 and state_store is None:
        uncovered.append(
            "on-chip state store area (this mapping holds "
            f"{math.fsum(bills):.6g} B of resident state, and the card names no "
            "synthesis library to compose the SRAM macro that would hold it)"
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
        "state_sram_macros": state_macros,
        "state_sram_bytes_required": (None if bills is None else math.fsum(bills)),
        "state_sram_silicon_mm2": state_sram,
        "state_sram_provenance": (
            "composed-measured" if state_store is not None
            else ("unpriced-before-lowering" if state_bytes is None else "absent")
        ),
        "digital_silicon_mm2": digital + pools + state_sram,
        "total_silicon_mm2": analog + digital + pools + state_sram,
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
            + (
                f" + an on-chip state store of {state_macros} measured SRAM macro(s) "
                f"= {state_sram:.6g} mm2, one store PER STAGE, sized to the "
                f"{math.fsum(bills):.6g} B of resident state this mapping holds"
                if state_store is not None
                else (
                    " + NO state-store term: this is the provisional area, taken "
                    "before the run is priced, and the state bill does not exist yet"
                    if state_bytes is None
                    else " + NO state-store term: the card names no synthesis library"
                )
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


#: What each of the two device-class fields on a candidate MEASURES. They used
#: to share the name ``binding_device_class`` at candidate top level and inside
#: ``derived_digital`` while answering different questions, and on the shipped
#: Granite curve they disagreed on every point (top level ``link``, the digital
#: block ``shared_digital``). Two accountings under one name is what D21
#: forbids, so each now has its own name and its own sentence.
BINDING_CLASS_BASIS = (
    "the device class of the LARGEST BUSY TERM on the beat-setting stage, read "
    "off this candidate's priced timeline (evaluation.binding_term.terms[0], "
    "ADJ-10). This is what SETS THE BEAT, and it is the one authoritative "
    "'binding' field on the candidate; derived_digital.binding_device_class is "
    "the same number from the same place."
)
BUSIEST_CLASS_BASIS = (
    "the device class with the highest MAX OCCUPANCY over the whole timeline "
    "(D28's per-class utilization). It answers 'which class is busiest', which "
    "is NOT 'which class sets the beat': on a weight-stationary machine "
    "hundreds of analog macros each fire once per beat, so max_occupancy can "
    "never select analog_macro however much of the beat the analog m-pass "
    "takes. This field used to be called binding_device_class and contradicted "
    "the beat-setting one on every point of the shipped Granite curve."
)


def _digital_block(evaluation, block=None):
    """What the DERIVED digital side came out as at this point (D31/D32).

    Read off ``fws_eval``'s own ``digital_silicon`` block so the sweep never
    recomposes a second version of it (D21).

    A card that names NO synthesis library still gets a block. It used to get
    none at all, so a ViT candidate — whose frontier card declares no library —
    carried no digital row of any kind and a reader could not tell an absent
    law from an unasked question. The block is now always present and names its
    absences the way ``fws_eval``'s own does: ``composed: false``, every derived
    width ``None``, and the terms that could not be composed listed by name.
    What is still MEASURED in that case is measured: the beat-setting stage
    comes off the timeline, not off a composition.
    """
    block = _digital_silicon_of(evaluation) if block is None else (block or {})
    binding = block.get("binding_term") or dict(
        getattr(evaluation, "binding_term", None) or {}
    )
    largest_any = (binding.get("terms") or [{}])[0]
    if not block.get("composed"):
        return {
            "composed": False,
            "reason": (
                "this candidate's analog card names no cim.cards.<card>."
                "synthesis_library, so D32 has no measured blocks to compose and "
                "D31/ADJ-10 derived nothing: there is no vector engine width, no "
                "attention-fabric array count and no softmax lane count to report, "
                "and no composed area or power. These are ABSENT LAWS, not "
                "measured zeros, and the silicon accounting names the same gap in "
                "silicon.uncovered_terms."
            ),
            "library_technology": None,
            "vector_lanes": None,
            "vector_lanes_provenance": None,
            "engine_duty": None,
            "sizing_target": None,
            "analog_bound_stages": None,
            "stages_sized": None,
            "analog_beat_s": None,
            "fabric_num_arrays": None,
            "fabric_num_arrays_provenance": None,
            "fabric_softmax_lanes": None,
            "fabric_time_over_target": None,
            "fabric_analog_bound_stages": None,
            "fabric_saturated_stages": None,
            "fabric_stages_sized": None,
            # MEASURED, not composed: the beat-setting stage and its terms are
            # read off the priced timeline, so an uncomposed card still says
            # what binds it.
            "binding_stage": binding.get("stage"),
            "binding_device_class": largest_any.get("device_class"),
            "binding_block": largest_any.get("block"),
            "binding_busy_s": largest_any.get("busy_s"),
            "binding_analog_time_s": binding.get("analog_time_s"),
            "binding_is_analog": binding.get("analog_is_largest_term"),
            "binding_device_class_basis": BINDING_CLASS_BASIS,
            "shared_digital_chiplet_area_mm2": block.get(
                "shared_digital_area_mm2_per_chiplet"
            ),
            "shared_digital_area_provenance": "declared",
            "digital_area_mm2_total": None,
            "digital_power_W_total": None,
            "absent_terms": [
                "derived vector engine (D31/ADJ-9)",
                "derived attention fabric and softmax pipeline (ADJ-10)",
                "composed digital area and power (D32)",
            ],
            "disclosures": list(block.get("disclosures") or ()),
        }
    sizing = block.get("derived_engine_sizing") or {}
    fabric = block.get("derived_fabric_sizing") or {}
    largest = largest_any
    return {
        "composed": True,
        "library_technology": (block.get("library") or {}).get("technology"),
        "vector_lanes": block.get("vector_lanes"),
        "vector_lanes_provenance": block.get("vector_lanes_provenance"),
        "engine_duty": sizing.get("engine_duty_at_target"),
        "sizing_target": sizing.get("sizing_target"),
        "analog_bound_stages": len(sizing.get("analog_bound_stages") or ()),
        "stages_sized": sizing.get("stages_sized"),
        "analog_beat_s": sizing.get("analog_beat_s"),
        # ADJ-10: the fabric derives at every point too, so what it derived TO
        # is a reported quantity of the point, and so is what binds afterwards.
        "fabric_num_arrays": block.get("fabric_num_arrays"),
        "fabric_num_arrays_provenance": block.get("fabric_num_arrays_provenance"),
        "fabric_softmax_lanes": block.get("fabric_softmax_lanes"),
        "fabric_time_over_target": fabric.get("attention_time_over_analog_target"),
        "fabric_analog_bound_stages": len(fabric.get("analog_bound_stages") or ()),
        "fabric_saturated_stages": len(fabric.get("saturated_stages") or ()),
        "fabric_stages_sized": fabric.get("stages_sized"),
        "binding_stage": binding.get("stage"),
        "binding_device_class": largest.get("device_class"),
        "binding_block": largest.get("block"),
        "binding_busy_s": largest.get("busy_s"),
        "binding_analog_time_s": binding.get("analog_time_s"),
        "binding_is_analog": binding.get("analog_is_largest_term"),
        # The SAME number the candidate carries at top level, from the SAME
        # place (evaluation.binding_term). One accounting, spelled once here
        # and mirrored there for readers who never open this block.
        "binding_device_class_basis": BINDING_CLASS_BASIS,
        "shared_digital_area_provenance": "composed-measured",
        "shared_digital_chiplet_area_mm2": (
            block.get("shared_digital_chiplet") or {}
        ).get("area_mm2_per_chiplet"),
        "digital_area_mm2_total": block.get("digital_area_mm2_total"),
        "digital_power_W_total": block.get("digital_power_W_total"),
        # D36's PACKAGING READING of the two areas just above: engines + this
        # chiplet's share of the state store in ONE area, walked against the
        # [50, 400] mm2 band. Read straight off fws_eval's own block so the
        # sweep never composes a second version of it (D21), and REPORTED, NOT
        # APPLIED — digital_area_mm2_total is unchanged by it.
        "chiplet_packaging": block.get("chiplet_packaging"),
    }


def stage_mismatch_warning(candidate):
    """One plain line naming a lopsided partition, or ``None`` (D34).

    George: the mapper's whole job is to eliminate stage mismatch even at an
    area cost, so a partition that is lopsided is a FAILURE and has to be
    impossible to miss in a regeneration log — not a field somebody has to go
    looking for. Plain language, no decision codenames: this is read by whoever
    is watching a sweep scroll past.
    """
    balance = (candidate.get("pipeline") or {}).get("stage_balance") or {}
    if not balance.get("measured") or balance.get("verdict") != "MISMATCHED":
        return None
    worst = (balance.get("long_stages") or [{}])[0]
    bound = float(balance.get("balanced_beat_bound_s") or 0.0)
    measured = float((candidate.get("metrics") or {}).get("tokens_per_s") or 0.0)
    balanced_rate = (1.0 / bound) if bound > 0 else None
    return (
        f"STAGE MISMATCH {candidate['id']}: stage {worst.get('stage')} busy "
        f"{float(worst.get('busy_s') or 0.0):.3g} s = "
        f"{float(worst.get('ratio_to_median') or 0.0):.1f}x median; dominant "
        f"{worst.get('dominant_block') or 'unknown'} on "
        f"{worst.get('dominant_device_class') or 'unknown'}; "
        + ("beat set here" if worst.get("sets_the_beat") else "beat set elsewhere")
        + "; balanced bound ~"
        + (f"{balanced_rate:.6g}" if balanced_rate else "unknown")
        + " tok/s vs measured "
        + (f"{measured:.6g}" if measured else "none")
        + f" ({len(balance.get('long_stages') or ())} of "
        f"{balance.get('stages')} stages run long)"
    )


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
                       digital_card, source, model_id=None, atlas=None):
    """One candidate, evaluated end to end on the REAL mapped path.

    Returns the candidate record. Evaluation stops at the FIRST stage that
    refuses, and the stage tag plus the refusal's own message are recorded —
    nothing is dropped and no message is rewritten.

    ``atlas`` is the export context (``{"output_dir", "run", "model_path",
    "mode"}``) or ``None``. When it is given, a candidate that gets all the way
    through writes its ``fws_atlas/1`` document HERE, from the mapping and
    evaluation that are still in scope — see :func:`export_point_atlas` for why
    the export lives inside the evaluation rather than in a pass after it.
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
        # D33: PRESENT ON EVERY ROW, null until a machine is built and drawn.
        # An infeasible candidate keeps the null and carries its own
        # fail_stage / fail_message beside it, so a reader who clicks it sees a
        # REFUSAL WITH A REASON rather than a missing file.
        "atlas_point": None,
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

    # ADJ-13: the chip is built to the placement. The layers are placed once
    # against a bound no chip can exceed, and the slot count is rewritten to
    # what the busiest chip actually needed. This runs AFTER the config stage
    # on purpose: a card or schema error belongs to `config`, and deriving
    # first would re-label it as a mapping refusal.
    try:
        raw, derived_slots = _derive_chip_capacity(
            raw, model_config, model_id, _hw_from_raw
        )
        candidate["knobs_derived"] = {"arrays_per_chip": derived_slots}
        hw = _hw_from_raw(raw)
        # D35-REVISED: the macro this point actually built, and the fold that
        # falls out of it. mux is DERIVED — it is never an axis and never a
        # config field — so this is the ONE place a reader meets it, under the
        # X x Y : Z label every artifact uses.
        _macro = hw.cim_config.analog_card.params
        candidate["knobs_derived"]["macro"] = _macro.label
        candidate["knobs_derived"]["macro_mux"] = int(_macro.mux)
        candidate["knobs_derived"]["macro_cells"] = int(_macro.rows) * int(_macro.cols)
    except ValueError as exc:
        candidate["wall_s"] = time.time() - started
        return _fail(candidate, "mapping", exc), None

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
            # THE STUB PACKER'S CHIP ARITHMETIC (D35-REVISED / the stub round).
            # `packing_summary` has carried these since the pipeline-order packer
            # landed; the sweep was dropping them on the floor, so a mapped point
            # reported a chip split whose CAPACITY and TAIL were invisible. Both
            # are read from the one packing summary, never recomputed here (D21).
            "chips_packed": int(packing_summary["chips_packed"]),
            "chip_capacity_slots": int(packing_summary["chip_capacity_slots"]),
            "tail_slots": int(packing_summary["tail_slots"]),
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
    # THE EFFECTIVE ALLOCATION GRANULARITY, still REPORTED though D38 parked
    # the knob: the number the point was actually priced at is a fact about the
    # machine, and it does not stop being one because nobody may sweep it any
    # more. There is no precedence note here now — bank_depth and
    # column_sets_per_tile are both refused as axes (D38), so no point can
    # declare the two at once and no report has to explain which won.
    effective_bank = int(mapping.device.column_sets_per_tile)
    candidate["effective_column_sets_per_tile"] = effective_bank

    # Stage: budget — the DECLARED caps, checked against the placement.
    if spec.max_chips > 0 and candidate["placement"]["analog_chips"] > spec.max_chips:
        candidate["wall_s"] = time.time() - started
        return _fail(
            candidate,
            "budget",
            f"placement needs {candidate['placement']['analog_chips']} analog chips but "
            f"{DSE_BLOCK}.max_chips = {spec.max_chips}.",
        ), mapping
    # THE AREA BUDGET IS NOT CHECKED HERE, deliberately. This area is
    # provisional: the digital engine is derived from the beat and the state
    # store is sized from the measured bill, so neither term exists until the
    # point has been priced. Refusing here would also throw away the one thing
    # a refused point is good for — what it WOULD have delivered — and a plot
    # of the design space would lose the points that did not fit rather than
    # showing them. The budget is applied after pricing instead.
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
    # The state store is sized from the bill the priced run just measured, so
    # the area below already carries it (see _silicon).
    # The store is sized PER STAGE and summed - not pooled and ceiled once -
    # because the stages sit on different chips and a macro cannot straddle two.
    # Pooling first under-charged by up to (stages - 1) macros and published a
    # different macro count from the evaluator's for the same bill (D21).
    residency = getattr(evaluation, "state_residency", None) or {}
    state_stage_bytes = [
        float(row.get("state_bytes", 0.0) or 0.0)
        for row in (residency.get("per_stage") or [])
    ]
    # ADJ-14: the chiplet count is a swept AXIS, and the packing below reports
    # what the machine would need if its stages shared perfectly. The count is
    # priced honestly at every point now, because the engine is sized for the
    # stages its chiplet serves: sharing widens the scan engine, and where the
    # attention fabric saturates instead, the beat grows and the point pays for
    # it in throughput. Both directions are visible on the frontier.
    # ONE composition of fws_eval's digital block per candidate, handed to
    # every consumer below (the chiplet packing, the derived-digital row and
    # the binding-term roll-up). Three independent calls composed it three
    # times under one name and cost three times the work.
    digital_silicon = _digital_silicon_of(evaluation)
    packed, chiplet_detail = _derive_shared_chiplets(
        evaluation, summary.get("pipeline_stages") or chiplets, digital=digital_silicon
    )
    if packed and isinstance(chiplet_detail, dict):
        chiplet_detail["applied"] = False
        chiplet_detail["basis"] = (
            "REPORTED, NOT APPLIED. First-fit over the stages sorted "
            "heaviest-first, bounded by the beat: this is how few chiplets the "
            "digital work would fit on IF a chiplet's engines were sized for "
            "every stage it serves. They are sized per stage today, so the "
            "machine keeps one chiplet per stage and this row is the headroom "
            "that sizing leaves on the table."
        )
        chiplet_detail["chiplets_if_shared"] = int(packed)
        chiplet_detail["chiplets_built"] = int(chiplets)
        candidate["derived_chiplets"] = chiplet_detail
        candidate["notes"].append(
            "the %d shared digital chiplets this machine builds (one per stage) "
            "could carry their work on %d if a chiplet's engines were sized for "
            "every stage it serves; they are sized per stage, so that is "
            "reported headroom and not a saving taken here."
            % (int(chiplets), int(packed))
        )
    else:
        candidate["derived_chiplets"] = {"chiplets_built": int(chiplets),
                                         "reason": chiplet_detail}
    silicon = _silicon(mapping, chiplets, state_bytes=state_stage_bytes)
    candidate["silicon"] = silicon
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
    stage_rows, tiles_off_stage = _pipeline_stage_rows(mapping)
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
        # THE STAGE COUNT, under a name that says it is a count. It was spelled
        # `stages` while every other `stages` in this document is a LIST, so a
        # consumer that wanted the stage plan got an integer. The plan itself is
        # `stages` below - one compact row per stage, enough to draw the
        # pipeline map of THIS candidate without a full atlas document.
        "stage_count": residency.get("stages"),
        "stages": stage_rows,
        # Tiles whose owner sits on no PP stage. Expected to be 0 under D30;
        # reported so the per-stage census can be checked against
        # placement.tiles instead of being trusted.
        "tiles_off_stage": tiles_off_stage,
        # D34: is this partition BALANCED? Mirrored from the evaluator, which
        # measures it off the same per-beat stage occupancy that sets the beat.
        # A lopsided partition is a mapper failure, so it rides every row and
        # the sweep WARNS on stdout for each one (see _stage_mismatch_warning).
        "stage_balance": measured_pipeline.get("stage_balance") or {
            "measured": False,
            "reason": (
                "this candidate is lowered under the "
                f"'{str(getattr(evaluation.serving, 'regime', '')) or 'unknown'}' "
                "regime, which has NO stage plan: one batch steps through the whole "
                "model rather than D streams rotating through stages, so there are "
                "no stage times to compare and no partition to call balanced or "
                "mismatched. Absent, not balanced."
            ),
        },
        "stages_basis": (
            "one compact row per PP stage of the stage plan (D29): the chips that "
            "carry it, the layers on it and what those layers put on analog "
            "silicon, counted by op block over the PLACED TILES. Enough to draw "
            "THIS candidate's pipeline map without a full atlas document; the full "
            "inventories are in report.atlas_points, one per candidate."
            if stage_rows
            else "EMPTY, and not a measured zero: this candidate is lowered under "
            f"the '{str(getattr(evaluation.serving, 'regime', '')) or 'unknown'}' "
            "regime, which has NO stage plan - one batch steps through the whole "
            "model rather than D streams rotating through stages - so there are no "
            "PP stages to draw. The placement is still in `placement` and the "
            "chips are still in the atlas document."
        ),
        "stage_plan_basis": str(summary.get("stage_plan_basis", "")),
        "basis": (
            "D29: the pipeline is ALWAYS FULL. tokens/s = 1/beat is the exit rate of "
            "ONE token per beat; D = the stage count, DERIVED from the stage plan and "
            "never configured; the per-stream rate is 1/(D x beat)."
        ),
    }
    # THE AREA BUDGET, applied here and nowhere else. Both terms it needs — the
    # derived digital engine and the state store sized to the measured bill —
    # exist only after pricing, and the metrics are stamped above, so a point
    # refused here still reports the throughput it would have delivered and
    # still draws on the design-space plot.
    if spec.max_silicon_mm2 > 0 and silicon["total_silicon_mm2"] > spec.max_silicon_mm2:
        candidate["wall_s"] = time.time() - started
        return _fail(
            candidate,
            "budget",
            f"this point needs {silicon['total_silicon_mm2']:.6g} mm2 of silicon but "
            f"{DSE_BLOCK}.max_silicon_mm2 = {spec.max_silicon_mm2:.6g}. Its terms: "
            f"{silicon['analog_macro_silicon_mm2']:.6g} mm2 of analog weight macros + "
            f"{silicon['shared_digital_silicon_mm2']:.6g} mm2 of shared digital "
            f"chiplets + {silicon['macro_pool_silicon_mm2']:.6g} mm2 of per-macro pools "
            f"+ {silicon['state_sram_silicon_mm2']:.6g} mm2 of on-chip state store "
            f"({silicon['state_sram_macros']} measured SRAM macros sized to the state "
            "this mapping holds). Area is the one thing that refuses a machine here: "
            "holding more state costs silicon, it never makes a design infeasible.",
        ), mapping

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
                    # WHICH CHIPS hold this stage. The state a stage holds sits
                    # on the chips the stage runs on, so a per-stage bill that
                    # did not name them could not be placed on the map.
                    "chips": [int(chip) for chip in (row.get("chips") or ())],
                    "recurrent_state_bytes": float(row["recurrent_state_bytes"]),
                    "kv_bytes": float(row["kv_bytes"]),
                    "state_bytes": float(row["state_bytes"]),
                    "per_stream_state_bytes": float(
                        row.get("per_stream_state_bytes", 0.0) or 0.0
                    ),
                    # WAS THE STORE ACTUALLY BUILT? "sized" is a claim that one
                    # was; a row whose card could compose no SRAM macro says
                    # "uncovered" and store_built is False. Reporting the macro
                    # count alone would let an absent law read as a zero-sized
                    # store (D21).
                    "store_macros": int(row.get("store_macros", 0) or 0),
                    "store_area_mm2": float(row.get("store_area_mm2", 0.0) or 0.0),
                    "store_provenance": str(row.get("store_provenance", "absent")),
                    "store_built": str(row.get("store_provenance", "")) == "composed-measured",
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
            # PRESENT AND EMPTY, so a consumer never has to tell a missing key
            # from a missing measurement. An unmeasured bill has no rows, and
            # the reason above says why.
            "per_stage": [],
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
    busiest = max(
        evaluation.utilization, key=lambda row: row.max_occupancy, default=None
    )
    # TWO QUESTIONS, TWO NAMES (D21). See BUSIEST_CLASS_BASIS / BINDING_CLASS_BASIS:
    # `busiest_device_class` is D28's peak-occupancy class and used to be called
    # `binding_device_class`, which made it contradict the beat-setting field of
    # the same name inside derived_digital on every point of the shipped curves.
    candidate["busiest_device_class"] = None if busiest is None else busiest.device_class
    candidate["busiest_device_class_basis"] = BUSIEST_CLASS_BASIS
    # WHERE THE BEAT GOES: the evaluator's own binding term, mirrored, plus the
    # per-stage analog / engine / fabric roll-up (ADJ-9, ADJ-10).
    candidate["binding_term"] = _binding_term_block(evaluation, digital_silicon)
    # THE ONE AUTHORITATIVE BINDING FIELD. Same accounting, same source and same
    # value as derived_digital.binding_device_class.
    candidate["binding_device_class"] = (
        (candidate["binding_term"].get("terms") or [{}])[0].get("device_class")
    )
    candidate["binding_device_class_basis"] = BINDING_CLASS_BASIS
    # D31/D32: the digital side is DERIVED at this point, never swept. What it
    # derived to is a reported quantity of the point. A card that composes
    # nothing still gets the block, with its absences named (see _digital_block).
    candidate["derived_digital"] = _digital_block(evaluation, digital_silicon)

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

    # Stage: state — the per-stage residency, REPORTED. It is not a gate any
    # more. Every stage of a filled pipeline holds all D streams' recurrent
    # state and KV for its layers; that store is sized from this bill and
    # charged as silicon in `silicon.state_sram_*`. A plan that holds more
    # state therefore costs more area, and what can refuse it is the AREA
    # BUDGET (mapping_dse.max_silicon_mm2), which is a statement about the
    # package. Nothing here refuses a design for needing memory.
    bill = candidate["state_bill"]
    # THE STORE COMES FROM THE EVALUATOR'S OWN PER-STAGE ROWS, not from the
    # silicon block. Copying the silicon figures here made the "one accounting"
    # gate below (and its twin in the validator) compare a value with its own
    # copy, so both passed unconditionally while the two sides genuinely
    # disagreed. Two independent computations of the same quantity is exactly
    # what D21 asks a gate to catch, and it can only catch it if they stay
    # independent.
    store_macros = int(sum(
        int(row.get("store_macros", 0) or 0)
        for row in (residency.get("per_stage") or [])
    ))
    store_area = float(math.fsum(
        float(row.get("store_area_mm2", 0.0) or 0.0)
        for row in (residency.get("per_stage") or [])
    ))
    candidate["state_bill"]["store_macros"] = store_macros
    candidate["state_bill"]["store_area_mm2"] = store_area
    candidate["state_bill"]["store_source"] = (
        "evaluation.state_residency.per_stage, summed; silicon.state_sram_* is the "
        "SAME quantity taken independently through _silicon, and the two are gated "
        "against each other rather than copied"
    )
    candidate["state_bill"]["budget_bytes"] = None
    candidate["state_bill"]["budget_verdict"] = "sized"
    if bill.get("measured") and store_macros:
        candidate["notes"].append(
            f"the on-chip state store is SIZED to this mapping: {store_macros} measured "
            f"SRAM macro(s) = {store_area:.6g} mm2 hold the "
            f"{bill['max_stage_state_bytes']:.6g} B worst stage and every other stage's "
            "share. Holding state costs area here; it never refuses a point."
        )
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
    # THE MAP OF THIS POINT, drawn where the machine exists (D33). `wall_s` is
    # stamped above and deliberately excludes this: it is the time to EVALUATE
    # the candidate, which is what the report's column says it is. The export's
    # own cost rides `atlas_export_s`.
    if atlas is not None:
        export_started = time.time()
        try:
            candidate["atlas_point"] = export_point_atlas(
                candidate,
                mapping,
                evaluation,
                output_dir=atlas["output_dir"],
                run=atlas["run"],
                source=source,
                model_path=atlas.get("model_path", ""),
                mode=atlas.get("mode", "LLM"),
                model_id=model_id,
            )
        except Exception as exc:  # noqa: BLE001 - an export must not kill a sweep
            # A machine that priced and then could not be DRAWN is a finding,
            # not a reason to throw away a long sweep. The row keeps its
            # numbers, atlas_point stays null and the failure is named.
            candidate["atlas_point_error"] = f"{type(exc).__name__}: {exc}"
            candidate["notes"].append(
                "this point was priced in full but its atlas document could not be "
                f"written: {candidate['atlas_point_error']}. The row's numbers stand; "
                "only its map is missing."
            )
        candidate["atlas_export_s"] = time.time() - export_started
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
        #: Corners the walk was offered and the PLATFORM refuses (D35-REVISED
        #: Y >= Z). Kept so a reader can see the wall the walk stopped at
        #: rather than infer it from a missing step.
        self.skipped = []

    def _point_of(self, index):
        return {name: self.values[name][rung] for name, rung in zip(self.names, index)}

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
                if not (0 <= rung < len(self.values[name])):
                    continue
                probe = index[:position] + (rung,) + index[position + 1:]
                # D35-REVISED: the sub-diagonal of the (Z, Y) box is not a
                # macro. The walk does not step there and does not pretend the
                # step was declined on merit — the corner is recorded as
                # REFUSED BY THE PLATFORM, with the reason.
                reason = self.spec.illegal_reason(self._point_of(probe))
                if reason is not None:
                    self.skipped.append(
                        {"knobs": self._point_of(probe), "axis": name, "reason": reason}
                    )
                    continue
                out.append((probe, name, direction, position))
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


def axis_vitality(candidates, front_ids, axes, declared=None):
    """PER AXIS: how many distinct values it contributed to the FRONT.

    THE REFEREE (D35-REVISED). "Every axis is refereed": an axis that lands the
    same value on every non-dominated point did not choose anything — the front
    would be the same front without it, and its rungs cost evaluation time and
    reader attention. One distinct value is the DEMOTION signal, and it is
    printed by name rather than left for somebody to notice.

    It is a SIGNAL, not a verdict: a bundle-wide run over several models is what
    actually retires an axis, and one model's front is one datum toward that.
    The note says so, so a single flagged run cannot be quoted as a decision.

    AN AXIS THAT WAS ONLY OFFERED ONE VALUE IS NOT A DEMOTION CANDIDATE. It is
    PINNED, which is a fact about this sweep and not about the axis, and the
    two are told apart by name — otherwise every smoke run would nominate
    whatever it happened to hold fixed.

    Counted over the FRONT and not over the valid set on purpose. An axis whose
    values are all priced but all dominated contributed nothing to the answer,
    which is exactly what this is asking.
    """
    front = [c for c in candidates if c["id"] in front_ids]
    rows = []
    for axis in axes:
        values = []
        for cand in front:
            value = (cand.get("knobs") or {}).get(axis)
            key = json.dumps(value, sort_keys=True)
            if key not in values:
                values.append(key)
        distinct = [json.loads(key) for key in values]
        row = {
            "axis": axis,
            "front_values": distinct,
            "distinct": len(distinct),
            "vital": len(distinct) >= 2,
        }
        if not front:
            row["note"] = (
                f"axis {axis!r} cannot be refereed: the front is empty, so no point "
                "carried any value of it. This is not a demotion signal."
            )
            row["vital"] = None
        elif declared is not None and len(list(declared.get(axis, ()))) < 2:
            row["vital"] = None
            row["note"] = (
                f"axis {axis!r} was PINNED by this sweep: it declares "
                f"{list(declared.get(axis, ()))}, so one value on the front is "
                "arithmetic, not a finding. It is not refereed here."
            )
        elif len(distinct) < 2:
            row["note"] = (
                f"DEMOTION CANDIDATE: axis {axis!r} contributed ONE distinct value "
                f"({distinct[0]!r}) to the whole non-dominated set, so on this model it "
                "chose nothing — the front is the front it would have been with the "
                "axis pinned. D35-REVISED refers this to the BUNDLE: an axis is retired "
                "on agreement across models, and this run is one datum toward that."
            )
        else:
            row["note"] = (
                f"axis {axis!r} is VITAL here: {len(distinct)} distinct value(s) "
                f"{distinct!r} sit on the non-dominated set, so the front really is "
                "two-dimensional in it."
            )
        rows.append(row)
    return rows


def edge_crowding(candidates, front_ids):
    """Front points sitting ON a platform BOUND, named (D35-REVISED).

    The Y ceiling (8X) and the Z bounds are ADJUDICATED numbers, not physics.
    A front that crowds a wall is the measurement that would justify moving it,
    and it has to be visible to be that: a front whose best points all sit at
    Y = 8X is telling you the box is too small, and a report that only printed
    the winner would hide it.
    """
    import cim_analog_calc as _calc

    front = [c for c in candidates if c["id"] in front_ids]
    bounds = (
        ("macro_y", "Y", max(_calc.MACRO_Y_VALUES), "ceiling", "8X"),
        ("macro_y", "Y", min(_calc.MACRO_Y_VALUES), "floor", "X"),
        ("macro_z", "Z", max(_calc.MACRO_Z_VALUES), "ceiling", "X"),
        ("macro_z", "Z", min(_calc.MACRO_Z_VALUES), "floor", "8"),
    )
    rows = []
    for axis, letter, value, which, spelled in bounds:
        at_bound = [
            cand["id"]
            for cand in front
            if (cand.get("knobs") or {}).get(axis) is not None
            and int((cand.get("knobs") or {})[axis]) == int(value)
        ]
        if not at_bound:
            continue
        share = len(at_bound) / len(front) if front else 0.0
        rows.append(
            {
                "axis": axis,
                "bound": which,
                "value": int(value),
                "spelled": spelled,
                "front_points": at_bound,
                "share_of_front": share,
                "note": (
                    f"{len(at_bound)} of {len(front)} front point(s) sit AT the {axis} "
                    f"{which} {letter} = {value} ({spelled}). That bound is ADJUDICATED, "
                    "not physical: the platform box is where the sweep was told to look, "
                    "and a front that crowds a wall is the data that would justify moving "
                    "it. Extending the list is a one-line change in "
                    "cim_analog_calc; it is NOT taken automatically, because the wall may "
                    "also be the honest answer."
                ),
            }
        )
    return rows


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
        "budget_bytes": None,
        "budget_basis": (
            "there is no residency budget. The on-chip state store is SIZED to each "
            "point's own measured per-stage bill, composed from integer copies of the "
            "measured SRAM macro and charged in silicon.state_sram_silicon_mm2, so "
            "holding state costs AREA and never refuses a point. The one feasibility "
            f"budget is {DSE_BLOCK}.max_silicon_mm2."
        ),
        "store_area_mm2_min": min(
            ((cand.get("state_bill") or {}).get("store_area_mm2") for cand in measured
             if (cand.get("state_bill") or {}).get("store_area_mm2") is not None),
            default=None,
        ),
        "store_area_mm2_max": max(
            ((cand.get("state_bill") or {}).get("store_area_mm2") for cand in measured
             if (cand.get("state_bill") or {}).get("store_area_mm2") is not None),
            default=None,
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


def balance_threshold_of(payload):
    """The balance threshold this report was judged against (D34).

    READ OFF THE EMITTED ROWS, never re-declared here. The law is
    ``fws_eval.STAGE_BALANCE_MAX_RATIO`` and it rides every measured candidate
    as ``stage_balance.threshold``; a second copy in this file would be a
    second accounting of one number and could drift from the one that actually
    produced the verdicts (D21).
    """
    for candidate in payload.get("candidates") or ():
        balance = (candidate.get("pipeline") or {}).get("stage_balance") or {}
        if balance.get("measured"):
            return balance.get("threshold")
    return None


def _balance_cell(candidate):
    """The candidates table's stage-balance cell (D34)."""
    balance = (candidate.get("pipeline") or {}).get("stage_balance") or {}
    if not balance.get("measured"):
        return "-"
    ratio = _fmt(balance.get("ratio"), ".3g")
    return (
        f"**MISMATCHED {ratio}x**"
        if balance.get("verdict") == "MISMATCHED"
        else f"ok {ratio}x"
    )


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
    run = payload.get("run") or {}
    if run:
        lines.append(
            f"- run id: `{run['run_id']}` "
            f"(emitter {run['tool_version']}, code `{run['code_fingerprint']}`, "
            f"generated {run['generated_at']}). Every artifact this run wrote "
            "carries it: this report, the emitted config's header and each "
            f"`{ATLAS_POINTS_DIR}/<id>.json`."
        )
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
        if digital and not digital.get("composed"):
            lines.append(
                "- derived digital (D31/D32/ADJ-10): NOT COMPOSED. "
                + str(digital.get("reason", ""))
            )
        if digital and digital.get("composed"):
            lines.append(
                f"- derived digital (D31/D32/ADJ-10): {digital['vector_lanes']} vector "
                f"lanes ({digital['vector_lanes_provenance']}), "
                f"{digital['fabric_num_arrays']} attention array(s) and "
                f"{digital['fabric_softmax_lanes']} softmax lane(s) "
                f"({digital['fabric_num_arrays_provenance']}), "
                f"{_fmt(digital['digital_area_mm2_total'])} mm2 and "
                f"{_fmt(digital['digital_power_W_total'])} W composed from the measured "
                f"{digital['library_technology']} synthesis library"
            )
        if digital and digital.get("binding_block"):
            lines.append(
                "- WHAT BINDS (ADJ-10, measured on the priced timeline): stage "
                f"{digital['binding_stage']} spends "
                f"{_fmt(digital['binding_busy_s'], '.4g')} s on "
                f"{digital['binding_device_class']}/{digital['binding_block']} against "
                f"{_fmt(digital['binding_analog_time_s'], '.4g')} s of analog m-pass"
                + (
                    " - the ANALOG FLOOR is the limit"
                    if digital["binding_is_analog"]
                    else " - the analog floor is NOT the limit; see the binding_term "
                    "disclosure for which real limit is"
                )
            )
            lines.append(
                "- BUSIEST DEVICE CLASS (D28, a different question): "
                f"{selected.get('busiest_device_class') or '-'} has the highest max "
                "occupancy over the timeline. It is not what sets the beat and it is "
                "no longer spelled `binding_device_class`."
            )
        balance = (selected.get("pipeline") or {}).get("stage_balance") or {}
        if balance.get("measured"):
            if balance.get("verdict") == "MISMATCHED":
                worst = (balance.get("long_stages") or [{}])[0]
                bound = float(balance.get("balanced_beat_bound_s") or 0.0)
                lines.append(
                    "- **STAGE BALANCE: MISMATCHED** - the busiest stage runs "
                    f"{_fmt(balance['ratio'], '.3g')}x the median "
                    f"({_fmt(balance['max_stage_busy_s'], '.4g')} s against "
                    f"{_fmt(balance['median_stage_busy_s'], '.4g')} s), and "
                    f"{len(balance.get('long_stages') or ())} of {balance['stages']} "
                    "stages run long. The worst is stage "
                    f"{worst.get('stage')}, dominated by "
                    f"{worst.get('dominant_block')} on "
                    f"{worst.get('dominant_device_class')}"
                    + ("; it sets the beat" if worst.get("sets_the_beat") else "")
                    + ". A balanced partition of the same work could reach a beat of "
                    f"{_fmt(bound, '.4g')} s"
                    + (
                        f" ({_fmt(1.0 / bound)} tokens/s)"
                        if bound > 0
                        else ""
                    )
                    + f", bounded by its {balance.get('balanced_beat_bound_binding')}. "
                    "This is a MAPPER failure, not a property of the workload: the "
                    "partitioner should spend area to even the stages out. Nothing is "
                    "refused for it here."
                )
            else:
                lines.append(
                    "- stage balance: **balanced** - the busiest stage runs "
                    f"{_fmt(balance['ratio'], '.3g')}x the median, inside the "
                    f"{_fmt(balance['threshold'])}x threshold."
                )
        lines.append(f"- energy {_fmt(metrics['total_energy_pj'])} pJ over the timeline")
        lines.append(f"- wall time to evaluate this candidate: {_fmt(selected['wall_s'], '.3g')} s")
    lines.append("")

    lines.append("## Candidates")
    lines.append("")
    header = ["id"] + axes + [
        "ok", "fail stage", headline_unit, "silicon mm2", "analog mm2", "digital mm2",
        "chips", "D", "beat s", "MiB/stage", "binding class", "stage balance",
        "energy pJ", "wall s",
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
            _balance_cell(cand),
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
    lines.append(
        "`stage balance` is D34's verdict: the busiest stage's per-beat occupancy "
        f"over the median, flagged **MISMATCHED** past "
        f"{_fmt(balance_threshold_of(payload))}x. "
        "A lopsided partition is a MAPPER FAILURE - it sets the beat while the other "
        "stages idle - so every mismatched candidate also prints a WARNING line on "
        "the sweep's stdout naming the stage, what makes it long, and the beat a "
        "balanced partition of the same work could reach. It refuses nothing; the "
        "balance-aware partitioner is the follow-on."
    )
    lines.append("")
    lines.append(
        "`binding class` is the device class of the LARGEST BUSY TERM on the "
        "beat-setting stage - what SETS THE BEAT (ADJ-10, "
        "`candidate.binding_device_class`, the same number as "
        "`derived_digital.binding_device_class`). It is NOT the busiest class: that "
        "is `candidate.busiest_device_class` (D28's peak max-occupancy) and it sits "
        "in the utilization table below. The two used to share this name and "
        "disagreed on every point."
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
            + (
                f" — map `{cand['atlas_point']}`"
                if cand.get("atlas_point")
                else " — NO MAP (see the point-maps note below)"
            )
        )
    if not payload["front_ids"]:
        lines.append("(empty — no valid candidate)")
    lines.append("")
    lines.append(f"Front shape — **{payload['front_shape']}**: {payload['front_note']}")
    lines.append("")
    points = payload.get("atlas_points")
    if points:
        if points["enabled"]:
            lines.append(
                f"Point maps: {len(points['emitted'])} of {points['valid']} valid "
                f"candidate(s) have an `fws_atlas/1` document under "
                f"`{points['dir']}/`, each written from that candidate's OWN live "
                f"evaluation and stamped with run id `{points['run_id']}`. Every "
                "point is drawable, not only the front: the atlas has no privileged "
                "point (D33). The "
                f"{len(points['skipped_infeasible'])} infeasible candidate(s) built "
                "no machine, so they carry `atlas_point: null` and their own "
                "`fail_stage` / `fail_message` instead."
            )
        else:
            lines.append(
                "Point maps: NONE. This sweep ran with `--no-atlas-points`, so every "
                "candidate carries `atlas_point: null` for that reason and not "
                "because a machine was missing."
            )
        for failure in points.get("failed") or ():
            lines.append("")
            lines.append(
                f"**`{failure['id']}` was priced in full but could NOT be drawn**: "
                f"{failure['message']}. Its numbers stand; only its map is missing."
            )
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
        util_header = (
            ["id"] + [f"{name} mean" for name in classes] + ["busiest", "binding"]
        )
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
            ] + [
                str(cand.get("busiest_device_class") or "-"),
                str(cand.get("binding_device_class") or "-"),
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append(
            "Idle devices are INSIDE every mean (D28): a class average taken over the "
            "busy devices only would report idle silicon as well used."
        )
        lines.append("")
        lines.append(
            "`busiest` is the class with the highest MAX OCCUPANCY on this timeline; "
            "`binding` is the class of the largest busy term on the beat-setting "
            "stage. They are different questions and they routinely give different "
            "answers - on a weight-stationary machine hundreds of analog macros each "
            "fire once per beat, so max occupancy can never select `analog_macro` "
            "however much of the beat the analog m-pass takes."
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
#
# run id:      {run_id}
# emitter:     {tool_version} (code {code_fingerprint})
# generated:   {generated_at}
#              D33: the SAME run id rides dse_report.json and every
#              points/<id>.json this run wrote. An artifact whose id differs
#              from its siblings' came out of a different generation and the
#              numbers on it were never compared with theirs.
"""


def emit_selected_config(raw_base, selected, out_path, *, base_path, model_path,
                         analog_card, digital_card, source, run=None):
    """Write the winning point as a complete, runnable hardware YAML."""
    raw = apply_point(raw_base, selected["knobs"], analog_card, digital_card, source)
    raw.pop(DSE_BLOCK, None)
    # ADJ-13: the emitted machine carries the sizing the sweep DERIVED for it.
    # run_perf does not derive — deriving is what the sweep is for — so a
    # config that shipped the base file's declared capacity would describe a
    # different machine from the one the sweep selected, and the round-trip
    # check below would catch it as a mismatch (it did).
    derived = selected.get("knobs_derived") or {}
    if derived.get("arrays_per_chip"):
        raw.setdefault("cim", {}).setdefault("chip", {})["arrays_per_chip"] = int(
            derived["arrays_per_chip"]
        )
        # Both homes, for the same reason apply_point writes both: an emitted
        # config that carries a stale mapping.macros_per_chip is not runnable,
        # whatever the header claims.
        emitted_mapping = raw.get("mapping")
        if isinstance(emitted_mapping, dict) and emitted_mapping.get("macros_per_chip") is not None:
            emitted_mapping["macros_per_chip"] = int(derived["arrays_per_chip"])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = EMITTED_HEADER.format(
        block=DSE_BLOCK,
        base=base_path,
        model=model_path,
        cand=selected["id"],
        knobs=", ".join(
            [f"{k} = {_knob_cell(v)}" for k, v in selected["knobs"].items()]
            + [f"{k} = {_knob_cell(v)} (DERIVED)"
               for k, v in sorted((selected.get("knobs_derived") or {}).items())]
        ),
        headline=headline_text(selected),
        run_id=(run or {}).get("run_id", "(none: this config was emitted outside a run)"),
        tool_version=(run or {}).get("tool_version", TOOL_VERSION),
        code_fingerprint=(run or {}).get("code_fingerprint", "unknown"),
        generated_at=(run or {}).get("generated_at", "unknown"),
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
# Per-candidate atlas documents (D33)
# ---------------------------------------------------------------------------


def export_point_atlas(candidate, mapping, evaluation, *, output_dir, run, source,
                       model_path, mode="LLM", model_id=None):
    """One ``fws_atlas/1`` document for THIS candidate, from ITS LIVE evaluation.

    D33: the atlas has no privileged point. Clicking ANY point has to show the
    whole machine, so EVERY candidate that built one gets a document — not the
    front, not a sample of the front. The report is the one source of all
    results and every row in it points at its own map.

    THE DOCUMENT IS BUILT WHERE THE MACHINE EXISTS. This runs inside
    :func:`evaluate_candidate`, at the moment ``mapping``, the lowered program
    and ``evaluation`` are all still in hand, so a map costs an export and
    nothing else. The earlier design re-walked ``apply_point ->
    _derive_chip_capacity -> build_mapping -> build_fws_program ->
    evaluate_fws`` per exported point, which priced every machine twice (8.07 s
    per document on Granite-4.0-H-1B) and is exactly why the old code capped
    how many points it was willing to draw. There is no cap now because there
    is nothing left to ration: the second pricing pass is gone.

    An INFEASIBLE candidate never built a machine, so it gets no document and
    keeps ``atlas_point: null`` beside its own ``fail_message`` — the absence is
    a refusal with a reason, not a missing file.

    Returns the report-relative path, or ``None`` with the failure recorded on
    the candidate.
    """
    import fws_atlas_export

    points_dir = Path(output_dir) / ATLAS_POINTS_DIR
    points_dir.mkdir(parents=True, exist_ok=True)
    knobs = ", ".join(f"{k}={_knob_cell(v)}" for k, v in candidate["knobs"].items())
    cand_id = candidate["id"]
    document = fws_atlas_export.export_atlas(
        [mapping],
        boundary_rows=[mapping.boundary_table()],
        title=f"{mapping.label} - point {cand_id}",
        subtitle=(
            f"QIF P3.7 sweep point {cand_id} ({knobs}); {headline_text(candidate)}, "
            f"{_fmt(candidate['silicon']['total_silicon_mm2'])} mm2. Placed by P3 "
            "and priced by P4 on THIS point's own timeline - no number here is "
            "carried over from another point of the sweep."
        ),
        reference_command=(
            "tools/fws_qif_dse.py --hardware_config "
            f"{source} --model_config {model_path} --mode {mode}"
            + (f" --model_id {model_id}" if model_id else "")
            + f"   (sweep point {cand_id}: {knobs})"
        ),
        duty_cycles=[evaluation.atlas_duty_cycles()],
        extra_metrics=[evaluation.atlas_metrics()],
        pool_sizing=[evaluation.atlas_pool_sizing()],
        extra_relaxations=[evaluation.atlas_relaxations()],
        service_links=[evaluation.atlas_service_links()],
    )
    # THE ARTIFACT SAYS WHICH RUN MADE IT (D33). Same id as the report and as
    # the emitted config; `dse_point` says which row of that report this
    # document is the map of. It does NOT claim front membership: the front is
    # a property of the whole sweep and does not exist yet when this is
    # written. `report.front_ids` is where that lives, once.
    document["run_id"] = run["run_id"]
    document["run"] = dict(run)
    document["dse_point"] = OrderedDict(
        (
            ("report", "dse_report.json"),
            ("candidate_id", cand_id),
            ("knobs", dict(candidate["knobs"])),
            ("knobs_derived", dict(candidate.get("knobs_derived") or {})),
            ("headline", headline_text(candidate)),
            ("total_silicon_mm2", candidate["silicon"]["total_silicon_mm2"]),
            (
                "front_membership",
                "not stated here: whether this point is on the Pareto front is a "
                "property of the SWEEP, and the sweep is not finished when this "
                "document is written. dse_report.json front_ids / knee say it, once.",
            ),
        )
    )
    fws_atlas_export.write_atlas_json(document, points_dir / f"{cand_id}.json")
    return f"{ATLAS_POINTS_DIR}/{cand_id}.json"


def atlas_points_block(candidates, run, enabled):
    """The report's roll-up of what got a map and what did not, by name."""
    emitted = [c["id"] for c in candidates if c.get("atlas_point")]
    infeasible = [c["id"] for c in candidates if not c["ok"]]
    failed = [
        {"id": c["id"], "message": c["atlas_point_error"]}
        for c in candidates
        if c.get("atlas_point_error")
    ]
    return OrderedDict(
        (
            ("schema", "fws_atlas/1"),
            ("dir", ATLAS_POINTS_DIR),
            ("run_id", run["run_id"]),
            ("enabled", bool(enabled)),
            ("candidates", len(candidates)),
            ("valid", sum(1 for c in candidates if c["ok"])),
            ("emitted", emitted),
            ("skipped_infeasible", infeasible),
            ("failed", failed),
            (
                "basis",
                "ONE fws_atlas/1 document per candidate that BUILT A MACHINE, "
                "written from that candidate's own live evaluation at the moment it "
                "existed - no second pricing pass, no sampling, no cap (D33: the "
                "atlas has no privileged point, so clicking any point must show "
                "everything). Every document carries this run's run_id. A candidate "
                "with atlas_point: null is an INFEASIBLE one: it never built a "
                "machine, so there is no inventory to draw, and its fail_stage and "
                "fail_message say what refused it."
                if enabled
                else "NO documents were written: this sweep ran with "
                "--no-atlas-points. Every candidate carries atlas_point: null for "
                "that reason and not because a machine was missing.",
            ),
        )
    )


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


def run_sweep(hardware_config, model_config_path, *, mode="LLM", model_id=None,
              output_dir=None, objective=None, emit_config=None, verify=False,
              quiet=False, atlas_points=True):
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

    # ONE IDENTITY FOR EVERYTHING THIS RUN WRITES (D33). Taken here, before the
    # first candidate is priced, so the report, the emitted config and every
    # per-front-point atlas document carry the same id even when the sweep runs
    # for hours.
    run = run_identity(
        hw_path,
        model_path,
        mode=mode,
        model_id=model_id,
        objective=spec.objective,
        sweep_echo=spec.echo(),
    )
    if not quiet:
        print(
            f"[QIF DSE] run {run['run_id']} ({run['tool_version']}, code "
            f"{run['code_fingerprint']}, {run['generated_at']}); every artifact "
            "this run writes carries it."
        )

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
        # D34's WARNING is printed even under --quiet. Quiet suppresses the
        # per-candidate progress rows, which are noise; a mapper failure is not
        # noise, and the whole point of the line is that it survives into a
        # regeneration log that nobody was watching live.
        warning = stage_mismatch_warning(candidate)
        if warning:
            print(f"[QIF DSE] WARNING: {warning}")
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
                + (
                    f" -> {candidate['atlas_point']} "
                    f"[{candidate.get('atlas_export_s', 0.0):.1f} s]"
                    if candidate.get("atlas_point")
                    else ""
                )
            )
        else:
            print(
                f"[QIF DSE] {candidate['id']} ({knobs}): INFEASIBLE at stage "
                f"{candidate['fail_stage']} [{candidate['wall_s']:.1f} s] "
                f"{candidate['fail_message']}"
            )

    # D33: the export context, or None under --no-atlas-points. It is handed to
    # every candidate so the map is written where the machine exists, and the
    # sweep never prices a point twice.
    atlas_context = (
        {
            "output_dir": output_dir,
            "run": run,
            "model_path": str(model_path),
            "mode": mode,
        }
        if atlas_points
        else None
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
            atlas=atlas_context,
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
        # D33: the run id at TOP LEVEL, and the whole identity block beside it.
        # Every artifact this run writes carries the same id, so a page built
        # from a report and a set of point documents can tell one generation
        # from two.
        "run_id": run["run_id"],
        "run": run,
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
        "axis_vitality": axis_vitality(
            rankable, front_ids, tuple(spec.axes), declared=dict(spec.axes)
        ),
        "edge_crowding": edge_crowding(rankable, front_ids),
        "illegal_points": spec.illegal_points,
        "ladder_trail": ladder_trail,
        "state_bill_summary": _state_bill_summary(candidates, spec),
        "selected_id": None if selected is None else selected["id"],
        "selected": selected,
        "best_violation": best_violation(candidates),
        "disclosures": disclosures,
        "silicon_coverage": silicon_coverage,
        "total_wall_s": total_wall,
        "atlas_points": None,
        "verify": None,
    }

    # THE MAPS ARE ALREADY ON DISK. Every candidate that built a machine wrote
    # its own document from its own live evaluation inside evaluate_candidate
    # (D33), so there is nothing to export here - only to roll up.
    payload["atlas_points"] = atlas_points_block(candidates, run, atlas_points)

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
                run=run,
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
    parser.add_argument(
        "--no-atlas-points",
        dest="atlas_points",
        action="store_false",
        help=(
            "Do NOT write the per-candidate fws_atlas/1 documents "
            f"(<output-dir>/{ATLAS_POINTS_DIR}/<id>.json). They are on by default: "
            "D33 makes the CLICKED POINT bind the atlas page and gives no point a "
            "privileged status, so every candidate that builds a machine gets one. "
            "They are written from each candidate's own live evaluation, so they "
            "cost an export rather than a second pricing pass - but they are large, "
            "and this turns them off."
        ),
    )
    parser.set_defaults(atlas_points=True)
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
            atlas_points=args.atlas_points,
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
    # D35-REVISED: every axis is refereed, and the referee speaks OUT LOUD.
    # A demotion candidate printed only into JSON is a finding nobody reads
    # during a regeneration, which is the same reason the stage-balance
    # verdict (D34) prints here.
    for row in payload.get("axis_vitality") or ():
        if row.get("vital") is False:
            print(f"[QIF DSE] AXIS VITALITY — {row['note']}")
    for row in payload.get("edge_crowding") or ():
        print(f"[QIF DSE] EDGE CROWDING — {row['note']}")
    illegal = payload.get("illegal_points") or ()
    if illegal:
        print(
            f"[QIF DSE] {len(illegal)} declared corner(s) are not macros and were not "
            f"priced: {illegal[0]['reason']}"
        )
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
