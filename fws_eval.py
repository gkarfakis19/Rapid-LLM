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

"""Pricing and evaluating a PLACED FWS-CIM DAG — QIF P4.

P3 places, P4 prices, and then P4 reads every published number off the ONE
timeline the pricing produced. That sentence is the whole design and it is
also the honesty defense (AUDIT finding 4: OPTIMA carried two TOPS accountings
that disagreed by 10x inside one artifact). Two accountings of one metric
cannot exist here because there is only one object to read.

The four seams, each owned elsewhere:

* **P2** (:mod:`cim_timing`) owns every law. This module calls
  ``price_tiled_op``, ``price_reduction``, ``attention_call_timing``,
  ``price_ssm_block``, ``price_linear_attention_block``, ``p2p_time_s`` and
  ``digital_pool_sizing``. It reimplements none of them.
* **P3** (:mod:`fws_mapping`, :mod:`program.fws_build`) owns the placement.
  Every op arrives with a :class:`~program.fws_build.FwsOpAnnotation` naming
  the law that prices it; :func:`price_program` dispatches on that string and
  never on an op name.
* **program.analytic_sim** owns the scheduler. P4 hands it a duration vector
  and a device-resource table (capacity + named owner) and gets one timeline
  back. Capacity 1 is the legacy exclusivity, bit-identically.
* **P5** (:mod:`fws_atlas_export`) consumes the duty cycles and the labeled
  metrics this module exports.

WHAT IS AND IS NOT A METRIC (P4 §3, ADJ-6)
==========================================
Every entry of :attr:`FwsEvaluation.metrics` is a PROJECTION of
:attr:`FwsEvaluation.timeline`. No metric is a period times a count — that
would reintroduce the steady-state assumption the DAG exists to remove — and
no metric is computed by a formula that runs beside the timeline.

The decode series is a WINDOW: ``ServingPoint.decode_steps`` steps are lowered
and priced. When the request decodes further than the window, the remaining
steps are an **extrapolation**, carried in its own block, labeled as an
extrapolation, and never promoted into ``metrics``. That is ADJ-6's "disclosed
extrapolation" read strictly: the disclosure is a field of the artifact, not a
footnote, and the extrapolated figure never becomes the headline.

The headline is ``tokens_per_s`` at the decode terminal; ``requests_per_s`` is
printed beside it (ADJ-6). The pass-1/2 fabric-ceiling / sustained pair does
not appear anywhere in this module: it retired with the closed form because
the DAG knows whether the pipeline is full.

ENERGY (P4 §6)
==============
One total, built from named components, each carrying its OWN coverage label.
No aggregate exists that is not the sum of the printed parts. A component with
no law says ``uncovered`` and contributes 0 — it never contributes an invented
number, and it never hides inside a blanket word.
"""

from __future__ import annotations

import math
import re

import cim_analog_calc as _analog_calc
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from cim_timing import (
    _delta_state_dims,
    SynthesisLibraryError,
    STATE_SRAM_BLOCK,
    sram_macro_bytes,
    LAW_UNVALIDATED,
    PROVENANCE_DERIVED_COUNT,
    AttentionCallDemand,
    CimDeviceModel,
    DerivedEngineSizing,
    DerivedFabricSizing,
    DigitalPoolSizing,
    EngineDemand,
    EngineSizingError,
    FabricDemand,
    ReductionOpDescriptor,
    SA_GEOMETRY_BANDWIDTH_DISCLOSURE,
    SA_GEOMETRY_DATAFLOW_DISCLOSURE,
    SA_GEOMETRY_RULE_DISCLOSURE,
    FABRIC_FLOORS as cim_timing_FABRIC_FLOORS,
    FABRIC_FLOOR_REASONS,
    derive_digital_chiplet_packaging,
    DIGITAL_CHIPLET_AREA_MIN_MM2,
    DIGITAL_CHIPLET_AREA_MAX_MM2,
)
from fws_mapping import (
    LINKS_PRICED,
    LINKS_ZEROED,
    MAPPING_STUB_KEY,
    MAPPING_STUB_REASON,
    MAPPING_STUB_VALUE,
    PP_UTIL_FULL,
    REGIME_FILLED,
    STATE_ELEMENT_BASIS,
    STATE_ELEMENT_BYTES,
    STATE_ELEMENT_DTYPE,
    FwsMapping,
    MappingError,
    Relaxation,
    hop_kind,
    link_price_disclosures,
    stage_handoff_time_s,
)
from program.analytic_sim import CoarseEvalResult, DeviceResource, evaluate_detailed
from program.fws_build import (
    LAW_ANALOG_GEMM,
    LAW_K_ACCUMULATION,
    LAW_PD_LINK,
    LAW_SLICE_REDUCTION,
    LAW_STAGE_HOP,
    FwsOpAnnotation,
    ServingPoint,
    annotations_of,
)
from program.ir import Program

__all__ = [
    "COVERAGE_COVERED",
    "COVERAGE_PARTIAL",
    "COVERAGE_UNCOVERED",
    "ClassUtilization",
    "EnergyComponent",
    "FwsEvaluation",
    "MemoryVerdict",
    "Metric",
    "OpCost",
    "PACKING_COMPARISON_SCHEMA",
    "PoolSizingReport",
    "PricingResult",
    "REPORT_SCHEMA",
    "device_resources",
    "evaluate_fws",
    "packing_comparison_document",
    "price_program",
    "render_report",
    "report_document",
]

#: The report schema id. One version field for the whole document (P4 §8).
REPORT_SCHEMA = "fws_qif_report/1"

#: Per-component energy coverage labels (ADJ-6). ``uncovered`` is a real
#: answer: the component exists, no law prices it, and the total says so.
COVERAGE_COVERED = "covered"
COVERAGE_PARTIAL = "partial"
COVERAGE_UNCOVERED = "uncovered"

#: Boundary-id prefix -> the parallelism axis whose link law prices it. The
#: mapping wrote the prefix; P4 never re-infers a role from a device pair.
_BOUNDARY_AXIS: Mapping[str, str] = {
    "tp": "tp",
    "ep": "ep",
    "act": "pp",
    "partial": "pp",
    "pd": "pp",
}

#: Blocks that run on the shared digital chiplet and the law that prices them.
_FABRIC_ATTENTION_BLOCKS = ("attention_qk", "attention_softmax", "attention_pv")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpCost:
    """One priced op: a duration, an energy, and the law that produced both.

    ``basis`` names the law CALL, not the law family, so a reader can find the
    number's source without reading this module.
    """

    uid: int
    kind: str
    block: str
    device_class: str
    device_id: int
    macro_id: int
    chip_id: int
    phase: str
    step: int
    layer: Optional[int]
    owner: str
    duration_s: float
    basis: str
    energy_pj: float = 0.0
    energy_component: str = ""
    coverage: str = COVERAGE_UNCOVERED
    bytes_moved: float = 0.0
    detail: Mapping[str, float] = field(default_factory=dict)
    note: str = ""


@dataclass(frozen=True)
class Metric:
    """One published number, its unit, and the timeline projection behind it.

    The shape is the atlas's ``metrics[]`` shape on purpose (P5): the same
    record is printed, exported and asserted on.
    """

    key: str
    label: str
    value: object
    unit: str
    basis: str

    def as_dict(self) -> "OrderedDict[str, object]":
        return OrderedDict(
            (
                ("key", self.key),
                ("label", self.label),
                ("value", self.value),
                ("unit", self.unit),
                ("basis", self.basis),
            )
        )


@dataclass(frozen=True)
class EnergyComponent:
    """One named energy term with its OWN coverage label (P4 §6, ADJ-6)."""

    key: str
    label: str
    energy_pj: float
    coverage: str
    basis: str


@dataclass(frozen=True)
class MemoryVerdict:
    """One feasibility check: a high-water mark against a declared capacity.

    ``status`` is ``fits`` | ``VIOLATED`` | ``undeclared``. ``undeclared`` is
    not a pass: the capacity field is absent, so the model says so instead of
    passing the check by default (D21).
    """

    scope: str                      # "chip 3" | "tp shard 0"
    tier: str                       # "activation_sram" | "kv"
    owner: str
    high_water_bytes: float
    capacity_bytes: float
    status: str
    basis: str
    contributors: Mapping[str, float] = field(default_factory=dict)
    disclosure: str = ""


@dataclass(frozen=True)
class PoolSizingReport:
    """The per-macro digital pool, sized FROM THE TIMELINE (P4.5, D12).

    ``peak_concurrent_demand`` is measured: the largest number of this macro's
    pool ops whose demand intervals — ``[ready, finish)``, where ``ready`` is
    the finish of the op's last dependency — overlap. Queued work is demand
    even when a capacity-1 pool serializes it, which is the whole point: the
    derived width is what would make the pool NON-BLOCKING (D12).

    Nothing is clamped. An absurd width prints as an absurd width.
    """

    macro_id: int
    owner: str
    peak_concurrent_demand: int
    pool_ops: int
    per_unit: DigitalPoolSizing
    scaled: DigitalPoolSizing
    report_line: str


@dataclass(frozen=True)
class PricingResult:
    """The priced program: one duration vector, one cost per op, disclosures."""

    durations: Tuple[float, ...]
    costs: Tuple[OpCost, ...]
    resources: Mapping[int, DeviceResource]
    disclosures: Tuple[Relaxation, ...]


# ---------------------------------------------------------------------------
# Device resources: capacity with named owners (P4.1, D21/D22)
# ---------------------------------------------------------------------------

#: Why every device class defaults to capacity 1. It is not an exclusivity
#: assumption — it is what the LAWS charge: P2's analog law charges the
#: macro's whole ADC/mux path, and the fabric laws charge the whole chiplet
#: including its replicas and its vector lanes. Giving such a device capacity
#: 2 would run the same silicon twice.
_CAPACITY_BASIS = (
    "capacity 1: the P2 law that prices an op on this device charges the WHOLE "
    "device (the macro's ADC/mux path; the chiplet's systolic replicas and vector "
    "lanes), so one op occupies one device. The number is a capacity, not an "
    "exclusivity assumption (D22) — raise it with capacity_overrides when a card "
    "declares independently schedulable units."
)


def device_resources(
    mapping: FwsMapping,
    *,
    capacity_overrides: Optional[Mapping[int, int]] = None,
) -> "OrderedDict[int, DeviceResource]":
    """Every device of the mapping as a capacity with a NAMED owner (D21).

    A macro-hosted device is owned by the tiles resident on its macro; a shared
    digital chiplet is owned by the chip it serves. No device is anonymous —
    "every macro has a named owner" is checked here, not hoped for.
    """
    overrides = {int(k): int(v) for k, v in (capacity_overrides or {}).items()}
    out: "OrderedDict[int, DeviceResource]" = OrderedDict()
    for device in mapping.devices:
        if device.macro_id >= 0:
            macro = mapping.macro(device.macro_id)
            owners = macro.owners
            if owners:
                owner = ", ".join(owner.label for owner in owners)
            else:
                owner = f"unowned macro slot {macro.macro_id} on chip {macro.chip_id}"
        else:
            owner = f"shared digital chiplet on chip {device.chip_id}"
        out[device.device_id] = DeviceResource(
            device_id=device.device_id,
            capacity=overrides.get(device.device_id, 1),
            owner=owner,
            device_class=device.device_class,
            basis=(
                _CAPACITY_BASIS
                if device.device_id not in overrides
                else f"capacity {overrides[device.device_id]} declared by the caller"
            ),
        )
    return out


# ---------------------------------------------------------------------------
# The pricing contract (P4 §1): one op, one device, one duration, one law
# ---------------------------------------------------------------------------


class _Pricer:
    """One pass over the annotated program, one duration vector out."""

    def __init__(self, mapping: FwsMapping, serving: ServingPoint) -> None:
        self.mapping = mapping
        self.serving = serving
        self.device: CimDeviceModel = mapping.device
        self.model = mapping.model
        self.params = self.device.params
        self.tp = max(1, int(mapping.degrees.get("tp", 1)))
        self.act_bytes = float(mapping.hw.sw_config.precision.activations)
        self.layout = mapping.hw.network_layout
        #: The LINK PRICING MODE the mapping was built under. The mapper spells
        #: it (``mapping.links``), this pricer applies it, and there is no
        #: third reader (D21).
        self.links = str(getattr(mapping, "links", LINKS_PRICED))
        self.disclosures: List[Relaxation] = []
        self._seen: set = set()
        self._link_cache: Dict[str, Tuple[float, float, float]] = {}
        self._attention_cache: Dict[Tuple[str, int, Optional[int]], object] = {}
        #: D32 engine power, composed from the measured library once per run.
        self._engine_power: Dict[str, Optional[float]] = {}
        mixers = tuple(getattr(self.model, "layer_mixers", ())) or (("attention",),)
        self._mixers = mixers
        if self.links == LINKS_ZEROED:
            self._disclose_links_zeroed()

    # -- disclosures ----------------------------------------------------

    def _disclose_links_zeroed(self) -> None:
        """The TEMPORARY zero-links relaxation, named on every artifact.

        It is raised in the constructor and not at the first hop, so a mapping
        that happens to have no inter-chip boundary still says which pricing
        mode produced its numbers. A reader who sees a throughput without this
        line is reading a machine whose links were charged.
        """
        self._disclose(
            "links_zeroed",
            "inter-chip handoffs priced at 0 s",
            "TEMPORARY relaxation (George, 2026-08-27): mapping.links = 'zeroed', so "
            "every INTER-CHIP boundary transfer is priced at 0 s instead of the D17 "
            "p2p law. Every link and boundary floor VACATES: a stage's per-beat "
            "occupancy is its own compute alone, the beat and the binding_term read "
            "COMPUTE ONLY, and no link can be the term that sets the pace. The BYTES "
            "are untouched - they are annotated and reported exactly as the priced run "
            "reports them, because bytes are a fact of the mapping and only the TIME is "
            "relaxed. This is not this machine's throughput: it is the throughput of "
            "the same machine with free links, and the difference is whatever the "
            "priced run charges for its handoffs. The D16 PD pair handoff rides the "
            "same law and is priced at 0 s with them. The ENERGY is NOT relaxed: every "
            "moved byte is still charged the card's pJ/bit and carries the coverage a "
            "priced run gives it. The boundary TABLE is not relaxed either - its "
            "required_time_s is what the DECLARED link would need for those bytes, "
            "which is the card's property and the one place a reader can still see "
            "what this relaxation is hiding.",
        )

    def _disclose_link_energy(self, axis: str) -> None:
        """0 pJ on a card that declares no pJ/bit — named in EITHER link mode.

        The energy law does not know about the link pricing mode: a card that
        declares ``energy_per_bit = 0`` reports zero link energy and says so by
        name whether the TIME was charged or relaxed. Raising it from one branch
        only would make a zeroed run print a 0 pJ link term with no named reason
        — zero-because-free read as zero-because-uncovered. Only the closing
        clause differs, because it states what the run did with the time, and
        the time is the one thing the mode changes.
        """
        timing = (
            "Link TIME is relaxed to 0 s in this run (mapping.links = 'zeroed', "
            "disclosed as links_zeroed)."
            if self.links == LINKS_ZEROED
            else "Link TIME is fully priced."
        )
        self._disclose(
            "link_energy",
            "0 pJ: no energy_per_bit declared",
            f"The {axis} network dimension declares energy_per_bit = 0, so link "
            f"traffic reports zero energy rather than an invented pJ/bit. {timing}",
        )

    def _disclose(self, constraint: str, value: str, reason: str) -> None:
        if constraint in self._seen:
            return
        self._seen.add(constraint)
        self.disclosures.append(Relaxation(constraint=constraint, value=value, reason=reason))

    # -- links ----------------------------------------------------------

    def _link(self, axis: str) -> Tuple[float, float, float]:
        """(bandwidth B/s, latency s, energy J/bit) of one parallelism axis."""
        if axis not in self._link_cache:
            bw, lat = self.layout.link_for_parallelism(axis)
            dim = self.layout.dimension_for_parallelism(axis)
            epb = float(getattr(dim, "energy_per_bit", 0.0) or 0.0) if dim is not None else 0.0
            self._link_cache[axis] = (float(bw), float(lat), epb)
        return self._link_cache[axis]

    @staticmethod
    def _axis_of(annotation: FwsOpAnnotation) -> str:
        prefix = annotation.boundary_id.split(".", 1)[0]
        return _BOUNDARY_AXIS.get(prefix, "pp")

    # -- the dispatch ---------------------------------------------------

    def price(self, annotation: FwsOpAnnotation) -> OpCost:
        if annotation.law == LAW_ANALOG_GEMM:
            return self._weight_gemm(annotation)
        if annotation.law == LAW_K_ACCUMULATION:
            return self._accumulation(annotation)
        if annotation.law in (LAW_SLICE_REDUCTION,) or annotation.kind == "reduction":
            return self._reduction(annotation)
        if annotation.kind == "pool":
            return self._pool_helper(annotation)
        if annotation.kind == "fabric":
            return self._fabric(annotation)
        if annotation.kind == "transfer":
            return self._transfer(annotation)
        raise MappingError(
            "execution",
            f"op {annotation.uid} carries law {annotation.law!r} and kind "
            f"{annotation.kind!r}, which P4's pricing table has no row for. A placed op "
            "with no priced law is a gap, not a zero.",
        )

    # -- rows of the pricing table --------------------------------------

    def _weight_gemm(self, a: FwsOpAnnotation) -> OpCost:
        cost = self.device.price_tiled_op(a.tokens, a.tiles)
        return self._cost(
            a,
            duration_s=cost.time_s,
            basis=(
                f"CimDeviceModel.price_tiled_op(m_tokens={a.tokens:g}, "
                f"{len(a.tiles)} tile(s)) — the active-column-set analog law (ADJ-4)"
            ),
            energy_pj=cost.energy_pj,
            energy_component="analog_arrays",
            coverage=COVERAGE_COVERED,
            detail=OrderedDict(
                (
                    ("active_column_sets", float(cost.active_column_sets)),
                    ("macros", float(cost.macros)),
                    ("total_active_column_sets", float(cost.total_active_column_sets)),
                )
            ),
        )

    def _accumulation(self, a: FwsOpAnnotation) -> OpCost:
        """One in-macro K accumulation, priced by P7.2's accumulator law.

        The DAG carries the PACKER's descriptor, so nothing about the stack is
        re-derived here: depth, width and sink macro are the packing's own
        numbers. The descriptor is a LOCAL SEGMENT — a run of K blocks that sit
        on one macro. A whole local stack is one such run. A stack that spans
        macros contributes one run per macro that holds more than one of its
        blocks, and only the BETWEEN-macro part of it reaches the row-block
        partial-sum law; a descriptor whose blocks are on different macros is
        therefore a contradiction here and is refused, not zeroed (D21).
        """
        if a.k_stack is None:
            raise MappingError(
                "execution",
                f"op {a.uid} is a placed K-stack accumulation carrying no descriptor. "
                "P4 prices this row FROM the packer's KStack; without it there is "
                "nothing to price, and a zero would be an invented number.",
            )
        pool = self.device.digital_pool_sizing()
        cost = self.device.price_accumulation(
            a.k_stack, a.tokens, act_bytes=self.act_bytes, pool=pool
        )
        if not cost.priced_here:
            raise MappingError(
                "execution",
                f"op {a.uid} accumulates a K stack that SPANS macros "
                f"({a.k_stack.macro_ids}). That shape is already priced end to end by "
                "the row-block partial-sum law, so an accumulation op for it would be "
                "a second accounting of one metric (D21).",
            )
        for note in cost.disclosures:
            if "pool_energy_per_add_pj" in note:
                # The same statement as the accumulator_energy banner below;
                # two banners for one gap would be two accountings (D21).
                continue
            self._disclose(
                "k_stack_accumulator_stall",
                "the pool cannot drain one pass before the next lands",
                note,
            )
        covered = float(self.device.card.pool_energy_per_add_pj) > 0
        if not covered:
            self._disclose(
                "accumulator_energy",
                "0 pJ: no card knob declared",
                "cim.cards.<card>.pool_energy_per_add_pj is 0, so the K-stack "
                "accumulators report zero energy rather than an invented per-add "
                "figure. The accumulator TIME is fully priced (P7.2).",
            )
        return self._cost(
            a,
            duration_s=cost.time_s,
            basis=(
                f"CimDeviceModel.price_accumulation over a depth-{cost.depth} K stack "
                f"{cost.width} columns wide on macro {a.k_stack.sink_macro_id}: "
                f"{cost.law}"
            ),
            energy_pj=cost.energy_pj,
            energy_component="digital_accumulation",
            coverage=COVERAGE_COVERED if covered else COVERAGE_UNCOVERED,
            detail=OrderedDict(
                (
                    ("depth", float(cost.depth)),
                    ("width", float(cost.width)),
                    ("partial_adds", float(cost.partial_adds)),
                    ("hidden_adds", float(cost.hidden_adds)),
                    ("drain_cycles", float(cost.drain_cycles)),
                    ("stall_s", float(cost.stall_s)),
                )
            ),
        )

    def _reduction(self, a: FwsOpAnnotation) -> OpCost:
        pool = self.device.digital_pool_sizing()
        if a.law == LAW_SLICE_REDUCTION:
            descriptors = self.device.reduction_descriptors(a.tiles)
            if not descriptors:
                raise MappingError(
                    "execution",
                    f"op {a.uid} is a placed bit-slice reduction but P2 emits no "
                    "reduction descriptor for its tiles. P3 placed an op the card does "
                    "not slice; that is a contradiction, not a zero-cost op.",
                )
            basis = (
                f"CimDeviceModel.price_reduction over {len(descriptors)} shift-add "
                f"descriptor(s) from reduction_descriptors (D11)"
            )
        else:
            descriptors = (self._row_block_descriptor(a),)
            basis = (
                "CimDeviceModel.price_reduction over one row-block partial-sum tree "
                f"({descriptors[0].n_slices} operands): the SAME adder-tree law the "
                "bit-slice reduction uses, on the same pool (D11/D12)"
            )
            self._disclose(
                "row_block_partial_sum_law",
                "priced with the shift-and-add tree law",
                "A row-block partial sum and a bit-slice composition are the same adder "
                "tree on the same per-macro pool, so they are priced by the same P2 law "
                "(price_reduction) with the operand count in place of the slice count. "
                "Declaring a second law for one shape would be a second accounting (D21).",
            )
        time_s = 0.0
        energy_pj = 0.0
        adds = 0.0
        transport = 0.0
        for descriptor in descriptors:
            cost = self.device.price_reduction(
                descriptor, a.tokens, act_bytes=self.act_bytes, pool=pool
            )
            time_s += cost.time_s
            energy_pj += cost.energy_pj
            adds += cost.adds
            transport += cost.transport_bytes
        covered = float(self.device.card.pool_energy_per_add_pj) > 0
        if not covered:
            self._disclose(
                "reduction_energy",
                "0 pJ: no card knob declared",
                "cim.cards.<card>.pool_energy_per_add_pj is 0, so the priced shift-add "
                "trees report zero energy rather than an invented per-add figure. The "
                "reduction TIME is fully priced; only its energy is uncovered (ADJ-4).",
            )
        return self._cost(
            a,
            duration_s=time_s,
            basis=basis,
            energy_pj=energy_pj,
            energy_component="digital_reduction",
            coverage=COVERAGE_COVERED if covered else COVERAGE_UNCOVERED,
            detail=OrderedDict(
                (("adds", adds), ("transport_bytes", transport), ("trees", float(len(descriptors))))
            ),
        )

    def _row_block_descriptor(self, a: FwsOpAnnotation) -> ReductionOpDescriptor:
        """The row-block partial-sum tree P3 placed, as P2's descriptor shape."""
        macros = sorted({tile.site.macro_id for tile in a.tiles})
        operands = max(2, len(macros))
        lanes = max(1, max(tile.logical_columns for tile in a.tiles)) if a.tiles else 1
        return ReductionOpDescriptor(
            kind="shift_add_tree",
            arrangement="row_blocks",
            owner=a.owner if a.owner is not None else a.tiles[0].owner,
            operand_tiles=tuple(a.tiles),
            output_lanes=int(lanes),
            n_slices=operands,
            adds_per_output=operands - 1,
            depth=(operands - 1).bit_length(),
            site_macro_id=int(a.macro_id),
            local=True,
            # The partials already crossed as EXPLICIT transfer ops in the DAG;
            # charging them here again would be the second accounting D21 bans.
            transport_partials=0,
        )

    def _pool_helper(self, a: FwsOpAnnotation) -> OpCost:
        self._disclose(
            "macro_pool_helper_ops",
            "absorbed: 0 s, 0 pJ",
            "D12 sizes the per-macro digital pool so it NEVER BLOCKS the pipeline, and "
            "no card declares a per-element cost for a norm, an activation, a residual "
            "add or a short depthwise conv. These ops therefore carry zero time and zero "
            "energy, exactly as the pass-1/2 report's 'helper lanes are absorbed into "
            "their stage' assumption does. Their CONCURRENCY is still measured and it "
            "drives the derived pool width (P4.5), so 'absorbed' is a sized claim here, "
            "not a hope.",
        )
        return self._cost(
            a,
            duration_s=0.0,
            basis="absorbed by the derived per-macro digital pool (D12); no per-element pool law is declared",
            energy_pj=0.0,
            energy_component="macro_pool_helpers",
            coverage=COVERAGE_PARTIAL,
        )

    def _fabric(self, a: FwsOpAnnotation) -> OpCost:
        if a.block in _FABRIC_ATTENTION_BLOCKS:
            return self._attention(a)
        if a.block == "short_conv":
            # The conv as a TIMED chiplet op (cim.allocation.conv_placement).
            # The pool placement never reaches here: there the conv is absorbed
            # into the pool sizing and emits a pool op, not a fabric one.
            cost = self.device.price_short_conv(a.tokens)
            basis = (
                "CimDeviceModel.price_short_conv (law short_conv): channels x (2k-1) "
                "scalar ops per token on the vector engine - the SAME per-result law "
                "the per-macro pool sizing uses when the card places the conv there "
                "instead, read as chiplet work rather than pool width (D21)."
            )
        elif a.block == "ssm_scan":
            cost = self.device.price_ssm_block(
                self.model.ssm,
                int(self.params.hidden_dim),
                a.tokens,
                act_bytes=self.act_bytes,
                chunked=(a.phase == "prefill"),
            )
            basis = f"CimDeviceModel.price_ssm_block (law {cost.law}, {cost.validated})"
        elif a.block == "delta_rule":
            # model_param.linear_attention.chunk_size is OPTIONAL and has no
            # default. Declared -> the chunked UT-transform form runs on
            # prefill (decode retires one token, so a chunk of Q is not a
            # thing it can do). Undeclared -> the recurrent form, and the
            # disclosure below says what that costs in BOTH directions.
            declared_q = getattr(self.model.linear_attention, "chunk_size", None)
            chunk_size = (
                int(declared_q) if (declared_q and a.phase == "prefill") else 1
            )
            cost = self.device.price_linear_attention_block(
                self.model.linear_attention,
                a.tokens,
                chunk_size=chunk_size,
                act_bytes=self.act_bytes,
            )
            if declared_q:
                self._disclose(
                    "delta_rule_chunk_size",
                    f"{int(declared_q)} (DECLARED), the chunked UT-transform form on "
                    "prefill; decode retires one token and stays recurrent",
                    "model_param.linear_attention.chunk_size is declared, so P4 prices "
                    "the chunked form P2.6's law carries. Both forms are "
                    "LAW_UNVALIDATED and neither is checked against a reference; what "
                    "the declaration buys is that the Q the number rests on is written "
                    "down instead of assumed.",
                )
            else:
                self._disclose(
                    "delta_rule_chunk_size",
                    "1 (the RECURRENT form), in prefill as well as decode",
                    "P2.6's delta-rule law has a chunked form, but this model declares "
                    "no chunk size — model_param.linear_attention.chunk_size is "
                    "optional and has no default — so P4 prices the token-by-token "
                    "recurrent form on both phases rather than invent a Q (ADJ-4, no "
                    "invented numbers). THE DIRECTION OF THE RESULTING ERROR IS NOT "
                    "ESTABLISHED, and this disclosure does not claim it is. The chunked "
                    "form's saving is state traffic, and this card prices state traffic "
                    "at ZERO: no cim.cards.<card>.state_bytes_per_cycle is declared, so "
                    "state_time_s is 0.0 and the whole duration is arithmetic (see the "
                    "vector_engine:state_bytes_per_cycle entry, which says recurrent-"
                    "state traffic is reported and bounds nothing). On retired OPS the "
                    "chunked form is CHEAPER only for small Q and DEARER above Q ~ 34 "
                    "at this model's dims — measured against the recurrent form: 0.92x "
                    "at Q=8, 0.99x at Q=32, 1.12x at Q=64, 1.38x at Q=128. Declaring "
                    "chunk_size is the fix, and the schema now has the field.",
                )
            basis = f"CimDeviceModel.price_linear_attention_block (law {cost.law}, {cost.validated})"
        else:
            raise MappingError(
                "execution",
                f"op {a.uid} runs block {a.block!r} on the shared digital chiplet and "
                "P4's pricing table has no law for it. An unpriced act x act op is a "
                "gap, not a zero.",
            )
        for note in cost.disclosures:
            self._disclose(f"vector_engine:{note[:48]}", "declared relaxation", note)
        if str(cost.validated) == LAW_UNVALIDATED:
            # D21: a law's provenance is part of its number. cim_timing labels
            # every priced digital op `validated` (optima_m3_reference,
            # optima_m3_subset or unvalidated); the label reached the op's
            # basis string and stopped there, so a reader of the report or the
            # atlas saw a duration with no way to know NO numeric reference
            # exists for the law that produced it.
            self._disclose(
                f"unvalidated_law:{cost.law}",
                f"{a.block}: law {cost.law} is UNVALIDATED",
                f"CimDeviceModel prices this op with {cost.law!r}, which carries "
                f"validated == {LAW_UNVALIDATED!r}: the work counts are derived from "
                "first principles and NO numeric reference exists to check them "
                "against anywhere (D21). Every number this op contributes to — the "
                "makespan, the headline throughput, this device's occupancy — inherits "
                "that status. The law is not a guess about the hardware card, which is "
                "declared; it is an unchecked count of the arithmetic the algorithm "
                "does.",
            )
        self._fabric_energy_disclosure("vector")
        energy_pj, coverage = self._fabric_energy("vector", cost.time_s)
        return self._cost(
            a,
            duration_s=cost.time_s,
            basis=basis,
            energy_pj=energy_pj,
            energy_component="shared_digital_chiplet",
            coverage=coverage,
            detail=OrderedDict(
                (
                    ("arith_cycles", float(cost.arith_cycles)),
                    ("state_time_s", float(cost.state_time_s)),
                    ("ops", float(cost.work.ops)),
                    ("state_bytes", float(cost.work.state_bytes)),
                )
            ),
        )

    def _engine_power_w(self, engine: str) -> Optional[float]:
        """Composed power (W) of one shared-chiplet engine, or None with no library.

        D32: the library reports a MEASURED power per block, so an engine's
        power is the same unit-count composition its area is. Two engines are
        distinguished because they are different silicon running different ops:
        ``vector`` is the scan engine, ``sa_fabric`` is the systolic array and
        ``softmax`` is the pipeline beside it. THESE ARE THREE SEPARATE ENGINES
        AND EACH OP PAYS ONLY ITS OWN (M9). They used to be folded into one
        "attention" power - fabric + softmax summed - and that one number was
        charged to qk, pv AND softmax, which the lowering runs one after the
        other. So the softmax engine was billed its full 17.03 W for the 261 us
        it sits idle during QK. That contradicted this module's own disclosure
        that "an idle engine costs 0 pJ here", and it inflated every energy
        conclusion by 1.74x (Granite) to 1.95x (Qwen).
        """
        if engine in self._engine_power:
            return self._engine_power[engine]
        value: Optional[float] = None
        if self.device.engine_probing:
            # D31 pass A composes nothing: there is no engine yet, and the
            # probe's costs are discarded anyway. Not cached, so pass B asks
            # the question again on the real engine.
            return None
        if self.device.has_synthesis_library():
            if engine == "vector":
                value = float(self.device.vector_engine_composition().power_w)
            elif engine == "sa_fabric":
                value = float(self.device.sa_fabric_composition().power_w)
            elif engine == "softmax":
                value = float(self.device.softmax_engine_composition().power_w)
            elif engine == "attention":
                # KEPT ONLY AS THE UNION, for a caller that really does mean
                # "both engines busy at once". No op on the lowered timeline
                # does; every attention op names the engine it runs on.
                value = float(
                    self.device.sa_fabric_composition().power_w
                    + self.device.softmax_engine_composition().power_w
                )
        self._engine_power[engine] = value
        return value

    def _fabric_energy(self, engine: str, duration_s: float) -> Tuple[float, str]:
        """(pJ, coverage) for one shared-chiplet op, from the measured library."""
        power = self._engine_power_w(engine)
        if power is None:
            return 0.0, COVERAGE_UNCOVERED
        return float(power) * float(duration_s) * 1e12, COVERAGE_PARTIAL

    def _fabric_energy_disclosure(self, engine: str = "vector") -> None:
        if self.device.engine_probing:
            # Pass A has no engine and its disclosures are discarded; pass B
            # makes this one against the engine that was actually derived.
            return
        if not self.device.has_synthesis_library():
            self._disclose(
                "shared_digital_energy",
                "0 pJ: no law",
                "No law prices a shared-digital-chiplet op's energy: the card names no "
                "cim.cards.<card>.synthesis_library, so D32's measured block powers are "
                "not available and the card schema refuses an energy_per_op_pj knob "
                "precisely because nothing would read it (P2.1). Fabric TIME is fully "
                "priced; fabric ENERGY is uncovered and the total says so instead of "
                "absorbing it into a blanket PARTIAL word (ADJ-6).",
            )
            return
        power = self._engine_power_w(engine)
        self._disclose(
            f"shared_digital_energy:{engine}",
            f"{engine} engine at {power:.6g} W (composed), charged for the op's own time",
            "D32: the shared digital chiplet's energy is now COMPOSED from the measured "
            "synthesis library — the engine's block census times each block's own "
            f"measured power gives {power:.6g} W, and an op is charged that power for "
            "exactly as long as it runs. The label is PARTIAL, not COVERED, and here is "
            "why: the library reports ONE average power per block and does not separate "
            "dynamic from static, so (a) a block's power is charged at its synthesis "
            "toggle rate rather than at this op's activity, and (b) the chiplet's "
            "LEAKAGE while it is idle is not charged at all — an idle engine costs 0 pJ "
            "here, which is a floor, not a measurement. Both directions are named rather "
            "than folded into one adjusted number (D21, one accounting per metric).",
        )

    def _attention(self, a: FwsOpAnnotation) -> OpCost:
        timing = self._attention_timing(a)
        f_fabric = self.device.f_fabric_hz
        # SA-GEOM: `3 * R` of the QK engine, charged ONCE on QK. Identical to
        # `fabric.fill_drain_penalty_cycles` on every declared-geometry card;
        # it follows the DERIVED R when the geometry rule installed one.
        fill_drain = int(self.device.fabric_fill_drain_cycles)
        # EACH OP NAMES THE ENGINE IT RUNS ON, and pays only that engine (M9)
        # -- and only the SHARE OF IT THAT IS ACTIVE. ADJ-10 partitions the
        # fabric's arrays between the two calls (`fold_group_split`), so QK runs
        # on `qk_arrays` of `fabric_num_arrays` and PV on `pv_arrays`. Charging
        # each the WHOLE fabric billed arrays that are provably running the
        # other call, and it contradicted this module's own disclosure that an
        # idle engine costs 0 pJ.
        total_arrays = max(1, int(self.device.fabric_num_arrays))
        share = 1.0
        if a.block == "attention_qk":
            cycles = timing.qk_cycles + fill_drain
            engine = "sa_fabric"
            share = min(1.0, max(0, int(timing.qk_arrays)) / total_arrays)
            what = f"QK^T systolic run ({timing.qk_cycles} cycles) + the array's fill/drain ({fill_drain})"
        elif a.block == "attention_pv":
            # CTX-SPLIT: the add-reduce that puts a context-split PV call back
            # together lands on THIS op's tail — the reduce cannot start before
            # the last partial arrives. It is 0 on every unsplit call, so no
            # declared-geometry number moves; where it is not 0 the DERIVATION
            # charged the same cycles, and the two must agree or the fabric was
            # sized against a bill the run does not pay (M7).
            cycles = timing.pv_cycles + int(timing.combine_cycles)
            engine = "sa_fabric"
            share = min(1.0, max(0, int(timing.pv_arrays)) / total_arrays)
            what = f"PV systolic run ({timing.pv_cycles} cycles)"
            if int(timing.combine_cycles) > 0:
                what += (
                    f" + the {timing.pv_split}-way context-split add-reduce "
                    f"({timing.combine_cycles} cycles on {timing.combine_lanes} "
                    "measured FP_ADD lane(s))"
                )
        else:
            cycles = timing.softmax_cycles
            engine = "softmax"
            what = f"softmax lanes ({timing.softmax_cycles} cycles)"
        self._disclose(
            "attention_op_folding",
            "three serial DAG ops, not one folded stage",
            "P2's attention law folds QK^T and PV into ONE concurrent systolic stage and "
            "runs the softmax pipeline beside it: stage = max(max(qk, pv) + fill_drain, "
            "softmax). P3 placed three dependent ops, so the DAG prices three ops and "
            "SERIALIZES them. The op-internal fill/drain penalty is charged ONCE, on the "
            "QK op, and it is never added to the DAG's own system-level ramp — two "
            "quantities, two names (P4 §1). On disagreement the DAG wins and the closed "
            "form is annotated (ADJ-8); this note is that annotation.",
        )
        self._fabric_energy_disclosure(engine)
        energy_pj, coverage = self._fabric_energy(engine, cycles / f_fabric)
        if share < 1.0:
            # SCALE THE ENERGY, NEVER THE COVERAGE TOKEN. `coverage` is a
            # controlled vocabulary a gate reads by name; the share belongs in
            # the op's own detail, beside the cycles it explains.
            energy_pj *= share
        detail = OrderedDict(
            (
                ("engine", engine),
                ("engine_array_share", float(share)),
                ("cycles", float(cycles)),
                ("heads_chip", float(timing.heads_chip)),
                ("context", float(self._context_of(a))),
            )
        )
        if a.block == "attention_qk":
            # ADJ-10 sizes the fabric from CALL DIMS, not from a duration: a
            # duration measured at the declared width cannot be re-priced at a
            # candidate one. They ride the QK op because that is the op the
            # lowering charges the array's fill/drain to, so ONE record per
            # attention call carries the whole call (D21).
            detail.update(
                (
                    ("call_m", float(timing.call_m)),
                    ("call_k", float(timing.call_k)),
                    ("call_n", float(timing.call_n)),
                    ("folds", float(timing.folds)),
                    ("softmax_tokens", float(timing.softmax_tokens)),
                    ("fill_drain_cycles", float(fill_drain)),
                )
            )
            if int(timing.qk_split) > 1:
                # CTX-SPLIT: the QK gather. The score chunks concatenate with
                # NO arithmetic, but the softmax normalises over the whole row
                # so they have to meet first. The bytes are reported and the
                # time is 0 s under the intra-chip convention (D17) — the same
                # convention every other on-chip movement in this module gets.
                detail.update(
                    (
                        ("context_split", float(timing.qk_split)),
                        ("gather_bytes", float(timing.gather_bytes)),
                        ("gather_cycles", 0.0),
                    )
                )
                self._disclose(
                    "context_split_gather",
                    f"{timing.gather_bytes:.6g} bytes concatenated, priced at 0 s",
                    "CTX-SPLIT: this QK call's context was dealt across "
                    f"{timing.qk_split} arrays, so the score row arrives as "
                    f"{timing.qk_split} chunks that must be CONCATENATED before the "
                    "softmax can normalise over it. The concatenation is not "
                    "arithmetic — it is movement — and this repo prices intra-chip "
                    "movement as a DAG dependency with a zero duration, because D17 "
                    "declares one network law and it is the p2p fabric BETWEEN chips "
                    "and no on-chip interconnect law is declared anywhere. The bytes "
                    "are on this op so the size of that optimism is visible; the DAG "
                    "time is optimistic by exactly this movement.",
                )
        return self._cost(
            a,
            duration_s=cycles / f_fabric,
            basis=(
                f"CimDeviceModel.{self._attention_law_name(a)}: {what} at "
                f"f_fabric = {f_fabric / 1e9:.4g} GHz"
            ),
            energy_pj=energy_pj,
            energy_component="shared_digital_chiplet",
            coverage=coverage,
            detail=detail,
        )

    def _attention_law_name(self, a: FwsOpAnnotation) -> str:
        attention = getattr(self.model, "attention", None)
        if attention is not None and str(getattr(attention, "attention_type", "")).lower() == "mla":
            return "mla_attention_timing_from_config"
        window = self._window_of(a.layer)
        if window:
            return (
                "sliding_window_prefill_timing"
                if a.phase == "prefill"
                else "sliding_window_decode_timing"
            )
        return (
            "prefill_attention_timing" if a.phase == "prefill" else "decode_attention_timing"
        )

    def _window_of(self, layer: Optional[int]) -> Optional[int]:
        """The window layer ``layer`` sees, or None when the layer is global.

        The pattern is P1's ``AttentionWindowConfig``, read here for the first
        time: ``local_global_interval = N`` means ONE global layer every N
        attention layers, so layer ``i`` is global when ``(i + 1) % N == 0``
        (Gemma 3's 6 -> five local then one global). ``N = 1`` makes every
        layer global and the window inert, which is what P1 says it means and
        what :meth:`CimDeviceModel.window_context` already assumes.
        """
        attention = getattr(self.model, "attention", None)
        window = getattr(attention, "window", None) if attention is not None else None
        if window is None:
            return None
        size = int(getattr(window, "window_size", 0) or 0)
        if size <= 0:
            return None
        interval = int(getattr(window, "local_global_interval", 1) or 1)
        if interval <= 1:
            return None
        index = 0 if layer is None else int(layer)
        if getattr(window, "last_layer_global", False) and index == int(self.params.num_layers) - 1:
            return None
        if (index + 1) % interval == 0:
            return None
        self._disclose(
            "sliding_window_layer_pattern",
            f"one global layer every {interval} layers",
            "model_param.attention.window declares window_size and "
            "local_global_interval; P4 reads the pattern as 'layer i is global when "
            "(i + 1) % local_global_interval == 0', plus last_layer_global. The window "
            "itself is applied by CimDeviceModel.window_context, which is the only "
            "place a context is shortened (P2.6 5).",
        )
        return size

    def _context_of(self, a: FwsOpAnnotation) -> int:
        if a.context is not None:
            # THE FILLED PIPELINE (D29): the resident streams sit at DIFFERENT
            # decode depths, so the context belongs to the STREAM and the
            # lowering stamps it on the op. A step index would be the wrong
            # question here — several streams share a beat and none of them
            # shares a context.
            return int(a.context)
        if a.phase == "prefill":
            return int(self.serving.prefill_len)
        # Decode step k attends to the prefill context plus the k tokens it has
        # already emitted plus the one it is emitting now — the same
        # prefill+1 .. final walk the closed-form decode section takes.
        return int(self.serving.prefill_len + int(a.step) + 1)

    def _attention_timing(self, a: FwsOpAnnotation):
        key = (a.phase, int(a.step), a.layer)
        cached = self._attention_cache.get(key)
        if cached is not None:
            return cached
        attention = getattr(self.model, "attention", None)
        window = self._window_of(a.layer)
        context = self._context_of(a)
        batch = int(self.serving.batch)
        if attention is not None and str(getattr(attention, "attention_type", "")).lower() == "mla":
            timing = self.device.mla_attention_timing_from_config(
                attention,
                context,
                batch_size=batch,
                tp=self.tp,
                seq_len=1 if a.phase == "decode" else int(self.serving.prefill_len),
            )
        elif a.phase == "prefill":
            if window:
                timing = self.device.sliding_window_prefill_timing(
                    seq_len=int(self.serving.prefill_len),
                    window=window,
                    tp=self.tp,
                    streams=batch,
                )
            else:
                # The GQA-AWARE score call (m = S * shared_heads, k = head_dim,
                # n = S), which is what llm_util's prefill descriptors and the
                # closed form both price. `attention_timing` is the MHA-only
                # view of the same law: it coincides when kv_heads == num_heads
                # and undercharges a GQA stack by exactly shared_heads, because
                # the folded K carries the KV groups but not the query heads
                # inside a group. Caught by the P6.3 bridge on fws_cim_moe
                # (16 heads over 4 KV groups: 4x).
                timing = self.device.prefill_attention_timing(
                    seq_len=int(self.serving.prefill_len), tp=self.tp, streams=batch
                )
        else:
            if window:
                timing = self.device.sliding_window_decode_timing(
                    context, window, batch_size=batch, tp=self.tp
                )
            else:
                timing = self.device.decode_attention_timing(
                    context, batch_size=batch, tp=self.tp
                )
        self._attention_cache[key] = timing
        return timing

    def _stage_hop(self, a: FwsOpAnnotation) -> OpCost:
        """A STAGE HANDOFF, priced at the law layer's flat hop latency.

        The law is ``cim_timing.stage_handoff_time_s``: 50 ns when the two
        stages sit on one analog chip, 500 ns when the handoff leaves the chip,
        0 s when it never leaves a macro. It is LATENCY ONLY - the cost does
        not depend on how many bytes the handoff carries - and that gap is
        disclosed by name rather than closed with a bandwidth this repo has
        not measured.

        ``mapping.links = 'zeroed'`` still overrides it to 0 s, exactly as it
        overrides the D17 p2p law: the relaxation vacates link TIME wherever
        link time is charged, and the bytes are reported under both.
        """
        src_chip = int(self.mapping.device_record(a.src_device).chip_id)
        dst_chip = int(self.mapping.device_record(a.dst_device).chip_id)
        kind = hop_kind(src_chip, dst_chip)
        axis = self._axis_of(a)
        _bw, _lat, epb = self._link(axis)
        if epb <= 0:
            self._disclose_link_energy(axis)
        # THE ENERGY IS NOT THE TIME. A moved byte costs the card's pJ/bit
        # whether the time is a hop latency, a p2p law or a relaxation, so the
        # energy law is applied here unchanged and only the DURATION comes from
        # the hop.
        energy_pj = float(a.bytes_moved) * 8.0 * epb * 1e12
        if self.links == LINKS_ZEROED:
            return self._cost(
                a,
                duration_s=0.0,
                basis=(
                    f"{a.bytes_moved:.6g} B stage handoff ({kind}) priced at 0 s: "
                    "mapping.links = 'zeroed', the TEMPORARY relaxation (disclosed as "
                    "links_zeroed). Bytes reported, time relaxed."
                ),
                energy_pj=energy_pj,
                energy_component="link_traffic",
                coverage=COVERAGE_COVERED if epb > 0 else COVERAGE_UNCOVERED,
                bytes_moved=a.bytes_moved,
            )
        self._disclose(
            "stage_handoff_hop_prices",
            "stage handoffs priced at a flat hop latency",
            " ".join(link_price_disclosures()),
        )
        duration = float(stage_handoff_time_s(src_chip, dst_chip))
        return self._cost(
            a,
            duration_s=duration,
            basis=(
                f"stage handoff c{src_chip} -> c{dst_chip} priced as a {kind} hop: "
                f"{duration * 1e9:.6g} ns flat (cim_timing.stage_handoff_time_s). "
                f"The {a.bytes_moved:.6g} B it carries are reported and do NOT enter "
                "the time: these interim prices are latency only."
            ),
            energy_pj=energy_pj,
            energy_component="link_traffic",
            coverage=COVERAGE_COVERED if epb > 0 else COVERAGE_UNCOVERED,
            bytes_moved=a.bytes_moved,
            detail=OrderedDict((("hop_kind", kind), ("hop_time_s", duration))),
        )

    def _transfer(self, a: FwsOpAnnotation) -> OpCost:
        if a.law == LAW_STAGE_HOP:
            return self._stage_hop(a)
        if not a.crosses_chip:
            self._disclose(
                "intra_chip_activation_movement",
                "a dependency, not priced bytes",
                "D17 declares ONE network law and it is the p2p fabric BETWEEN chips. "
                f"{a.bytes_moved:.6g} bytes moving inside a chip are carried as a DAG "
                "dependency with a zero duration: no on-chip interconnect law is "
                "declared anywhere, and inventing one would be a number no card supports. "
                "The DAG time is optimistic by exactly that unpriced movement.",
            )
            return self._cost(
                a,
                duration_s=0.0,
                basis="intra-chip movement: a dependency, not a priced link (D17, disclosed)",
                energy_pj=0.0,
                energy_component="link_traffic",
                coverage=COVERAGE_UNCOVERED,
                bytes_moved=a.bytes_moved,
            )
        axis = self._axis_of(a)
        if self.links == LINKS_ZEROED:
            # THE RELAXATION, APPLIED WHERE THE LAW IS. The bytes stay on the
            # cost and in the artifact; only the duration is vacated, and the
            # basis says which mode produced it. The link ENERGY is still the
            # bytes times the card's pJ/bit: moving bytes costs energy whether
            # or not the model charges the time for them. The ENERGY LAW is
            # therefore fully applied here and its coverage label is the priced
            # path's label, computed by the same rule: this relaxation moves the
            # TIME and nothing else, so an energy number the card supports may
            # not be relabeled `uncovered` beside it: a component that says
            # `uncovered` CONTRIBUTES 0 (the energy law at the top of this
            # module), and the pJ this branch reports is not 0.
            # The disclosure is already on this pricer (it is raised once, in
            # the constructor, so a machine with no inter-chip hop still says
            # which mode priced it); this branch only applies the mode.
            _bw, _lat, epb = self._link(axis)
            if epb <= 0:
                self._disclose_link_energy(axis)
            return self._cost(
                a,
                duration_s=0.0,
                basis=(
                    f"{a.bytes_moved:.6g} B across the {axis} link priced at 0 s: "
                    "mapping.links = 'zeroed', the TEMPORARY relaxation (disclosed as "
                    "links_zeroed). Bytes reported, time relaxed."
                ),
                energy_pj=float(a.bytes_moved) * 8.0 * epb * 1e12,
                energy_component="link_traffic",
                coverage=COVERAGE_COVERED if epb > 0 else COVERAGE_UNCOVERED,
                bytes_moved=a.bytes_moved,
            )
        bw, lat, epb = self._link(axis)
        if bw <= 0:
            raise MappingError(
                "package",
                f"boundary {a.boundary_id!r} rides the {axis} link and the network layout "
                "declares no bandwidth for it. A transfer with no link law is not a free "
                "transfer.",
            )
        time_s = CimDeviceModel.p2p_time_s(a.bytes_moved, bw, lat)
        energy_pj = float(a.bytes_moved) * 8.0 * epb * 1e12
        if epb <= 0:
            self._disclose_link_energy(axis)
        law = "D16 PD handoff" if a.law == LAW_PD_LINK else "D17 p2p"
        return self._cost(
            a,
            duration_s=time_s,
            basis=(
                f"CimDeviceModel.p2p_time_s({a.bytes_moved:.6g} B, {bw:.6g} B/s, "
                f"{lat:.6g} s) over the {axis} link ({law}); never contended"
            ),
            energy_pj=energy_pj,
            energy_component="link_traffic",
            coverage=COVERAGE_COVERED if epb > 0 else COVERAGE_UNCOVERED,
            bytes_moved=a.bytes_moved,
            detail=OrderedDict((("bandwidth_bytes_per_s", bw), ("latency_s", lat))),
        )

    # -- record ---------------------------------------------------------

    def _cost(
        self,
        a: FwsOpAnnotation,
        *,
        duration_s: float,
        basis: str,
        energy_pj: float,
        energy_component: str,
        coverage: str,
        bytes_moved: float = 0.0,
        detail: Optional[Mapping[str, float]] = None,
    ) -> OpCost:
        return OpCost(
            uid=a.uid,
            kind=a.kind,
            block=a.block,
            device_class=a.device_class,
            device_id=a.device_id if a.device_id >= 0 else a.dst_device,
            macro_id=a.macro_id,
            chip_id=a.chip_id,
            phase=a.phase,
            step=int(a.step),
            layer=a.layer,
            owner=a.owner.label if a.owner is not None else "",
            duration_s=float(duration_s),
            basis=basis,
            energy_pj=float(energy_pj),
            energy_component=energy_component,
            coverage=coverage,
            bytes_moved=float(bytes_moved),
            detail=dict(detail or {}),
            note=a.note,
        )


def price_program(
    program: Program,
    mapping: Optional[FwsMapping] = None,
    *,
    capacity_overrides: Optional[Mapping[int, int]] = None,
) -> PricingResult:
    """Price every op of a placed FWS DAG through the P2 laws (P4.1).

    The mapping and the serving point ride on the program (``meta.misc``), so
    a caller that has the program has everything; passing ``mapping`` is an
    override for a caller that built the program from a different object.
    """
    annotations = annotations_of(program)
    if not annotations:
        raise MappingError(
            "execution",
            "price_program needs an FWS program (program.fws_build.build_fws_program); "
            "this program carries no fws_annotations.",
        )
    mapping = mapping or program.meta.misc.get("fws_mapping")
    serving = program.meta.misc.get("fws_serving")
    if mapping is None or serving is None:
        raise MappingError(
            "execution",
            "price_program needs the mapping and the serving point the DAG was lowered "
            "from; they ride on program.meta.misc as 'fws_mapping' / 'fws_serving'.",
        )
    pricer = _Pricer(mapping, serving)
    costs = tuple(pricer.price(annotation) for annotation in annotations)
    durations = tuple(cost.duration_s for cost in costs)
    resources = device_resources(mapping, capacity_overrides=capacity_overrides)
    disclosures = list(pricer.disclosures)
    raised = sorted(
        device_id
        for device_id, resource in resources.items()
        if int(resource.capacity) > 1
    )
    if raised:
        # The executor's root rule releases a slot a root op never took
        # (``free = min(capacity, free + 1)``). At capacity 1 that IS the legacy
        # idempotent assignment and the goldens pin it; above 1 it lets up to one
        # extra op per root be in flight on that device, so the schedule is
        # OPTIMISTIC. It is a relaxation, so it is disclosed in the artifact and
        # not only in a source docstring (D21).
        disclosures.append(
            Relaxation(
                constraint="device_capacity_root_release",
                value=f"{len(raised)} device(s) above capacity 1: {raised[:8]}",
                reason=(
                    "program.analytic_sim releases a device slot when a ROOT op "
                    "completes, although a root never occupied one. At capacity 1 the "
                    "release is idempotent and this is exactly the legacy exclusivity; "
                    "above capacity 1 it can leave up to one extra op in flight per "
                    "root on that device, so the timeline is optimistic by that much. "
                    "No shipped run raises a capacity — this fires only when a caller "
                    "passes capacity_overrides."
                ),
            )
        )
    return PricingResult(
        durations=durations,
        costs=costs,
        resources=resources,
        disclosures=tuple(disclosures),
    )


# ---------------------------------------------------------------------------
# The timeline and its projections (P4.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceOccupancy:
    """Busy time of one device, with the owner the busy time belongs to."""

    device_id: int
    device_class: str
    chip_id: int
    macro_id: int
    owner: str
    busy_s: float
    ops: int
    occupancy: float


#: How closely two inter-exit intervals must agree before the beat they carry
#: is called SETTLED (P7.9 / D29). It is a convergence tolerance on a
#: measurement, not a margin (D28): nothing is padded by it and no number is
#: relaxed toward it.
BEAT_CONVERGENCE_TOL = 1e-3


def _converged_beat_tail(
    intervals: Sequence[float], tol: float = BEAT_CONVERGENCE_TOL
) -> Tuple[float, ...]:
    """The longest SUFFIX of ``intervals`` that agrees with its own median.

    The exits of a filling pipeline ramp: the first traversals leave early
    because nothing is queued behind them yet. D - 1 held-out beats is the
    right allowance only when every stage has the same service time, and an
    uneven stage plan takes longer to settle — so holding out a FIXED count
    leaves ramp in the sample and makes the reported beat depend on how many
    beats the window happened to lower (measured: 12% on Granite-4.0-H-Tiny
    between decode_window 2 and 3).

    This reads the beat off the settled TAIL instead: grow a suffix from the
    last interval backwards while it still agrees with its own median to
    ``tol``, and stop at the first interval that does not. A single interval
    is trivially its own tail, and the caller reports how many were dropped as
    ramp rather than smoothing them in.
    """
    values = [float(value) for value in intervals]
    if not values:
        return ()
    best = tuple(values[-1:])
    for start in range(len(values) - 2, -1, -1):
        window = values[start:]
        centre = _median(window)
        if centre <= 0:
            break
        if max(abs(value - centre) for value in window) / centre <= tol:
            best = tuple(window)
        else:
            break
    return best


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return 0.5 * (ordered[middle - 1] + ordered[middle])


@dataclass
class FwsEvaluation:
    """One priced DAG, one timeline, and every number projected from it.

    Every attribute below is derived from :attr:`timeline` and :attr:`pricing`
    and from nothing else. There is no second evaluator, no closed form and no
    period anywhere in this object.
    """

    mapping: FwsMapping
    program: Program
    serving: ServingPoint
    pricing: PricingResult
    timeline: CoarseEvalResult
    metrics: Tuple[Metric, ...]
    occupancy: Tuple[DeviceOccupancy, ...]
    duty_cycles: Mapping[int, float]
    #: Per-device-class utilization (P7.3, D28) — a REQUIRED output, printed
    #: whether it flatters the provisioning or not.
    utilization: Tuple[ClassUtilization, ...]
    #: The decode step's bank-pass accounting (P7 Fact 1), measured.
    bank_passes: Mapping[str, object]
    #: What co-residency in a bank costs on a decode step (P7 Fact 3), measured.
    bank_sharing: Mapping[str, object]
    energy: Tuple[EnergyComponent, ...]
    memory: Tuple[MemoryVerdict, ...]
    pools: Tuple[PoolSizingReport, ...]
    extrapolation: Mapping[str, object]
    disclosures: Tuple[Relaxation, ...]
    decode_step_times_s: Tuple[float, ...]
    #: The filled pipeline, MEASURED (P7.7, D29): the beat, the exits it was
    #: read from, the fill transient and the stage plan. Empty under the
    #: retired lockstep regime.
    pipeline: Mapping[str, object] = field(default_factory=OrderedDict)
    #: Per-stage resident state/KV for all D streams, with its verdict (D29).
    state_residency: Mapping[str, object] = field(default_factory=OrderedDict)
    #: ADJ-10's HEADLINE MEASUREMENT: which stage sets the beat of the machine
    #: that GETS BUILT — every derived width installed, every op timed — and
    #: what it spends the beat on, itemized by op block. Empty when the run has
    #: no lowered slice to read it from. The probe's own beat-setting evidence
    #: is a different quantity under a different name (it rides
    #: :attr:`DerivedEngineSizing.beat_setting_stage` and describes the machine
    #: the derivation ran AGAINST), and the two are never merged (D21).
    binding_term: Mapping[str, object] = field(default_factory=OrderedDict)

    # -- projections a consumer asks for by name ------------------------

    def metric(self, key: str) -> Metric:
        for entry in self.metrics:
            if entry.key == key:
                return entry
        raise KeyError(f"no metric named {key!r}; have {[m.key for m in self.metrics]}")

    @property
    def makespan_s(self) -> float:
        return float(self.timeline.total_time)

    @property
    def total_energy_pj(self) -> float:
        """The one total: the sum of the printed parts and nothing else."""
        return math.fsum(component.energy_pj for component in self.energy)

    def atlas_duty_cycles(self) -> Dict[int, float]:
        """``{macro_id: duty cycle}`` for the P5 atlas (P4 §3, D20)."""
        return dict(self.duty_cycles)

    def atlas_utilization(self) -> List["OrderedDict[str, object]"]:
        """Per-device-class utilization for the atlas (P7.3 item 3, D28).

        The atlas colors macros by duty cycle, which answers "which macro" and
        never "how much of each device CLASS is idle". This is that second
        question, in the same record shape the report prints, so an atlas and a
        report cannot disagree about it (D21).
        """
        return [row.as_dict() for row in self.utilization]

    def atlas_packing(self) -> "OrderedDict[str, object]":
        """The packing law and, under dense packing, its waste accounting."""
        return _packing_block(self.mapping)

    def atlas_service_links(self) -> List["OrderedDict[str, object]"]:
        """One ``svc`` link per (digital chiplet -> analog chip) it services.

        WHAT A SERVICE LINK IS. A shared digital chiplet does the act x act
        work — attention and the scan/state update (D13) — for the layers of
        the analog chips assigned to it, and that work costs REAL traffic in
        both directions: operands leave the analog chip (Q/K/V for attention,
        the projected inputs for a scan) and the result comes back to a macro
        or its pool. The atlas already drew the ``act`` boundary between
        consecutive analog chips; it drew nothing at all for this
        relationship, so the picture showed a pipeline with no wires to the
        silicon that does half its work.

        WHERE THE RELATIONSHIP COMES FROM. It is read off the LOWERED DAG, not
        recomputed: every shared-digital compute op names the chiplet it runs
        on and the LAYER it serves, and the placement names the analog chip
        that holds that layer. Nothing here re-implements the builder's
        chiplet-assignment rule, so the two cannot drift.

        WHERE THE BYTES COME FROM. They are MEASURED: every priced ``link`` op
        in the beat :func:`_measurement_slice` selects whose two endpoints are
        an analog chip and a shared digital chiplet is summed onto that pair,
        the two directions kept apart so the basis can print them. Each term is
        the lowering's own ``FwsOpAnnotation.bytes_moved`` — the number the
        P3.5 boundary table and the energy accounting also read (D21).

        WHAT IS UNPRICED, BY NAME. Only traffic this repo LOWERS as a link op
        can be counted. The attention hops are lowered (qkv out, the attention
        output back); the RECURRENT hops are not — ``_recurrent_block`` places
        the scan on the chiplet with no transfer op on either side — so a
        relationship that carries only scan work measures 0 B and the basis
        says which block that is. It is an absent op, not a measured zero, and
        it is named here rather than filled with an estimate (D21, D28).
        """
        select, scope = _measurement_slice(self.serving)
        mapping = self.mapping
        unit = str(scope.get("unit", "beat"))
        analog_chips = [chip for chip in mapping.chips if chip.pool == "analog"]
        rows: "OrderedDict[Tuple[int, int], Dict[str, object]]" = OrderedDict()

        def _row(chiplet_chip: int, analog_chip: int) -> Dict[str, object]:
            return rows.setdefault(
                (int(chiplet_chip), int(analog_chip)),
                {
                    "operands_out": 0.0,
                    "results_back": 0.0,
                    "link_blocks": OrderedDict(),
                    "fabric_blocks": OrderedDict(),
                    "ops": 0,
                },
            )

        for annotation, cost in zip(annotations_of(self.program), self.pricing.costs):
            if not select(annotation):
                continue
            if cost.device_class == "shared_digital":
                layer = annotation.layer
                if layer is None:
                    continue
                shard = mapping.device_record(int(annotation.device_id)).shard
                for chip in analog_chips:
                    if int(layer) not in tuple(chip.layers):
                        continue
                    if chip.shard.tp != shard.tp:
                        continue
                    row = _row(int(annotation.chip_id), int(chip.chip_id))
                    blocks = row["fabric_blocks"]
                    blocks[cost.block] = int(blocks.get(cost.block, 0)) + 1
                continue
            if cost.device_class != "link":
                continue
            src, dst = int(annotation.src_device), int(annotation.dst_device)
            if src < 0 or dst < 0:
                continue
            src_record = mapping.device_record(src)
            dst_record = mapping.device_record(dst)
            classes = (src_record.device_class, dst_record.device_class)
            if "shared_digital" not in classes or classes[0] == classes[1]:
                continue
            if src_record.device_class == "shared_digital":
                chiplet, analog, direction = src_record, dst_record, "results_back"
            else:
                chiplet, analog, direction = dst_record, src_record, "operands_out"
            row = _row(int(chiplet.chip_id), int(analog.chip_id))
            row[direction] = float(row[direction]) + float(annotation.bytes_moved)
            row["ops"] = int(row["ops"]) + 1
            link_blocks = row["link_blocks"]
            link_blocks[cost.block] = float(link_blocks.get(cost.block, 0.0)) + float(
                annotation.bytes_moved
            )

        system = mapping.system_id
        out: List["OrderedDict[str, object]"] = []
        for (chiplet_chip, analog_chip) in sorted(rows):
            row = rows[(chiplet_chip, analog_chip)]
            out_bytes = float(row["operands_out"])
            back_bytes = float(row["results_back"])
            itemized = (
                ", ".join(
                    f"{block} {value:.6g} B" for block, value in row["link_blocks"].items()
                )
                or "no lowered link op on this relationship"
            )
            fabric = ", ".join(
                f"{block} x {count}" for block, count in row["fabric_blocks"].items()
            ) or "none in this slice"
            unpriced = [
                block
                for block in row["fabric_blocks"]
                if block in _UNPRICED_SERVICE_BLOCKS
            ]
            out.append(
                OrderedDict(
                    (
                        ("id", f"{system}.link.svc.d{chiplet_chip}->c{analog_chip}"),
                        ("from", f"{system}.chip.{chiplet_chip:03d}"),
                        ("to", f"{system}.chip.{analog_chip:03d}"),
                        ("role", "svc"),
                        ("bytes", out_bytes + back_bytes),
                        ("per", unit),
                        (
                            "label",
                            "shared-digital service traffic (act x act operands out, "
                            "results back)",
                        ),
                        (
                            "basis",
                            f"MEASURED on {scope.get('scope', unit)}: chiplet "
                            f"{chiplet_chip} runs {fabric} for chip {analog_chip}'s "
                            f"layers. Bytes = operands out {out_bytes:.6g} B + results "
                            f"back {back_bytes:.6g} B = {out_bytes + back_bytes:.6g} B "
                            f"per {unit}, over {int(row['ops'])} lowered link op(s); "
                            f"itemized: {itemized}. Every term is the lowering's own "
                            "FwsOpAnnotation.bytes_moved, so this link and the P3.5 "
                            "boundary table are one accounting (D21)."
                            + (
                                " UNPRICED COMPONENT: "
                                + ", ".join(unpriced)
                                + " lowers no transfer op on either side (the "
                                "projection writes and the scan reads without a "
                                "lowered hop), so its operand and result traffic is NOT "
                                "in the number above. That is an absent op, not a "
                                "measured zero, and it is named rather than estimated "
                                "(D21, D28)."
                                if unpriced
                                else ""
                            ),
                        ),
                    )
                )
            )
        return out

    def utilization_of(self, device_class: str) -> ClassUtilization:
        for row in self.utilization:
            if row.device_class == device_class:
                return row
        raise KeyError(
            f"no utilization row for {device_class!r}; have "
            f"{[row.device_class for row in self.utilization]}"
        )

    def atlas_metrics(self) -> List["OrderedDict[str, object]"]:
        """The labeled metrics the atlas prints beside the placement figures.

        The per-device-class utilization rides here too (P7.3 item 3, D28), in
        the metric shape the atlas already renders, so an exported atlas can
        never show a placement whose idle silicon is invisible. The values are
        the ClassUtilization rows themselves — the report and the atlas print
        one number, not two derivations of it (D21).
        """
        system = self.mapping.system_id
        out = [
            entry.as_dict()
            for entry in self.metrics
            if not isinstance(entry.value, (list, tuple))
        ]
        for row in self.utilization:
            out.append(
                Metric(
                    key=f"{system}.utilization.{row.device_class}",
                    label=f"{row.device_class} utilization (mean over the class)",
                    value=float(row.mean_occupancy),
                    unit="fraction of the makespan",
                    basis=(
                        f"{row.devices_used} of {row.devices} devices ran ops "
                        f"({row.idle_devices} idle, and idle devices are INSIDE this "
                        f"mean); peak {row.max_occupancy:.6g}, median "
                        f"{row.median_occupancy:.6g}. {row.basis}"
                    ),
                ).as_dict()
            )
        return out

    def atlas_pool_sizing(self) -> Dict[int, "OrderedDict[str, object]"]:
        """``{macro_id: macros[].digital_pool}`` sized FROM THIS TIMELINE.

        The atlas field and the report's P4.5 table are the SAME quantity, so
        they must be the same number (D21). Without this the atlas would print
        P2's per-unit derivation under a field name the report uses for the
        measured, concurrency-scaled width — one name, two accountings.
        Macros with no measured pool demand keep the per-unit width, which is
        the peak-1 case of the same formula.
        """
        out: Dict[int, "OrderedDict[str, object]"] = {}
        for report in self.pools:
            sized = report.scaled
            out[int(report.macro_id)] = OrderedDict(
                (
                    ("shift_add_units", int(sized.adders)),
                    ("act_lanes", int(sized.total_lanes)),
                    ("area_mm2", float(sized.area_mm2)),
                    (
                        "basis",
                        f"sized from the timeline (P4.5): peak concurrent digital "
                        f"demand {report.peak_concurrent_demand} x P2.5's per-unit "
                        f"derivation. area_mm2 is "
                        + (
                            f"{sized.area_mm2:.6g} mm2"
                            if sized.area_mm2 > 0
                            else "0.0 = UNCOVERED: no card declares a pool area law, "
                            "so this is an absent law and not a measured zero"
                        ),
                    ),
                )
            )
        return out

    def atlas_relaxations(self) -> List["OrderedDict[str, object]"]:
        """The P4 disclosures the atlas must carry beside P3's placement ones.

        The decode-window truncation is the load-bearing one: an atlas reader
        seeing a windowed run's metrics without it cannot tell the window from
        the declared decode length.
        """
        return [
            OrderedDict(
                (
                    ("constraint", item.constraint),
                    ("value", item.value),
                    ("reason", item.reason),
                )
            )
            for item in self.disclosures
        ]


@dataclass(frozen=True)
class ClassUtilization:
    """How much of ONE device class actually worked (P7.3, D28).

    D28 makes low utilization a provisioning finding the report must SURFACE,
    so this row exists for every class the mapping instantiated, including the
    classes that did nothing: an idle class with no row would be invisible
    exactly when it matters most.

    The occupancies are taken over EVERY device of the class, idle devices
    included. A mean over the busy ones only would report a machine's silicon
    as well used by leaving the unused silicon out of the average, which is the
    number that hides the finding.
    """

    device_class: str
    devices: int
    devices_used: int
    idle_devices: int
    ops: int
    busy_s: float
    mean_occupancy: float
    median_occupancy: float
    max_occupancy: float
    min_occupancy: float
    basis: str

    @property
    def idle_share(self) -> float:
        """Devices of this class that ran NOTHING, as a share of the class."""
        return (self.idle_devices / self.devices) if self.devices else 0.0

    def as_dict(self) -> "OrderedDict[str, object]":
        return OrderedDict(
            (
                ("device_class", self.device_class),
                ("devices", int(self.devices)),
                ("devices_used", int(self.devices_used)),
                ("idle_devices", int(self.idle_devices)),
                ("idle_share", float(self.idle_share)),
                ("ops", int(self.ops)),
                ("busy_s", float(self.busy_s)),
                ("mean_occupancy", float(self.mean_occupancy)),
                ("median_occupancy", float(self.median_occupancy)),
                ("max_occupancy", float(self.max_occupancy)),
                ("min_occupancy", float(self.min_occupancy)),
                ("basis", self.basis),
            )
        )


_UTILIZATION_BASIS = (
    "per DEVICE CLASS, over every device of the class the mapping instantiated "
    "(idle devices included in the average, which is what makes idle silicon "
    "visible — D28). A device's occupancy is its busy union over the makespan, "
    "the same projection the per-device occupancy table prints, so the two are "
    "one accounting (D21)."
)

_LINK_UTILIZATION_BASIS = (
    "links are NOT devices: D17 fixes the fabric at fully-connected p2p with no "
    "congestion model, so a link occupies no resource on the timeline. This row "
    "is therefore a DEMAND ratio — the priced transfer time on one chip-to-chip "
    "link over the makespan — and a value above 1.0 would mean the link is "
    "oversubscribed rather than that it was busy that long."
)


def _build_utilization(
    mapping: FwsMapping,
    pricing: PricingResult,
    occupancy: Sequence[DeviceOccupancy],
    annotations: Sequence[FwsOpAnnotation],
    makespan: float,
) -> Tuple[ClassUtilization, ...]:
    """One row per device class, plus the link row (P7.3 item 3, D28)."""
    by_class: "OrderedDict[str, List[DeviceOccupancy]]" = OrderedDict()
    for device in mapping.devices:
        by_class.setdefault(device.device_class, [])
    rows_by_id = {row.device_id: row for row in occupancy}
    for device in mapping.devices:
        row = rows_by_id.get(device.device_id)
        if row is not None:
            by_class[device.device_class].append(row)
    out: List[ClassUtilization] = []
    for device_class, rows in by_class.items():
        values = [float(row.occupancy) for row in rows]
        used = sum(1 for row in rows if row.ops)
        out.append(
            ClassUtilization(
                device_class=device_class,
                devices=len(rows),
                devices_used=used,
                idle_devices=len(rows) - used,
                ops=sum(int(row.ops) for row in rows),
                busy_s=math.fsum(float(row.busy_s) for row in rows),
                mean_occupancy=(math.fsum(values) / len(values)) if values else 0.0,
                median_occupancy=_median(values),
                max_occupancy=max(values) if values else 0.0,
                min_occupancy=min(values) if values else 0.0,
                basis=_UTILIZATION_BASIS,
            )
        )
    # The link row. One "device" is one ORDERED chip pair, which is what the
    # p2p law prices; intra-chip movement is not a link (D17) and is excluded
    # here exactly as it is excluded from the link energy term.
    per_link: "OrderedDict[Tuple[int, int], List[float]]" = OrderedDict()
    for cost in pricing.costs:
        if cost.kind != "transfer":
            continue
        annotation = annotations[cost.uid]
        if not annotation.crosses_chip:
            continue
        src = mapping.device_record(annotation.src_device).chip_id
        dst = mapping.device_record(annotation.dst_device).chip_id
        per_link.setdefault((int(src), int(dst)), []).append(float(cost.duration_s))
    if per_link:
        shares = [
            (math.fsum(times) / makespan) if makespan > 0 else 0.0
            for times in per_link.values()
        ]
        out.append(
            ClassUtilization(
                device_class="link",
                devices=len(per_link),
                devices_used=sum(1 for times in per_link.values() if times),
                idle_devices=0,
                ops=sum(len(times) for times in per_link.values()),
                busy_s=math.fsum(math.fsum(times) for times in per_link.values()),
                mean_occupancy=math.fsum(shares) / len(shares),
                median_occupancy=_median(shares),
                max_occupancy=max(shares),
                min_occupancy=min(shares),
                basis=_LINK_UTILIZATION_BASIS,
            )
        )
    return tuple(out)


def _build_bank_sharing(
    mapping: FwsMapping,
    program: Program,
    pricing: PricingResult,
    timeline: CoarseEvalResult,
    annotations: Sequence[FwsOpAnnotation],
    serving: ServingPoint,
) -> "OrderedDict[str, object]":
    """What co-residency in a bank actually COSTS (P7 Fact 3, restated by D29).

    Dense packing puts several tensors — and several layers — in one macro, so
    the question a reader will ask is whether the strangers get in each other's
    way. This block answers it by MEASUREMENT, not by assertion: every op that
    started later than it was ready is charged to whatever else was running on
    its device at the time, split by whose weights the blocker holds.

      * ``same_owner_delay_s`` — one tensor's own blocks waiting for one
        another. That is the macro walking its banks, not co-residency.
      * ``cross_layer_delay_s`` — a resident of ANOTHER LAYER in the way. It
        splits in two, and D29 is why the split exists:

        - ``within_stage_cross_layer_delay_s`` — the other layer is on the SAME
          stage. A stage runs its layers in sequence for the one stream it
          holds, so this stays 0.0: within-stage sharing is serial-free.
        - ``cross_stage_delay_s`` — the other layer is on a DIFFERENT STAGE.
          Under D29 every stage fires EVERY BEAT, on a different stream, so two
          stages sharing a macro contend on every beat. This is the term the
          retired lockstep regime could not have: it reported 0 because the
          layers ran one after another. It is priced by the timeline and
          reported here.

      * ``cross_tensor_same_layer_delay_s`` — a resident of the SAME layer in
        the way. Two tensors of one layer CAN be concurrent (a routed expert
        and a shared expert, a router and a projection), and when the packer
        puts them in one macro they serialize.
    """
    if serving.decode_steps <= 0:
        return OrderedDict(
            (
                ("measured", False),
                (
                    "basis",
                    "no decode step is lowered, and P7 is DECODE ONLY (D25): "
                    "co-residency in prefill is a different question (concurrent "
                    "layers) and this block does not answer it.",
                ),
            )
        )
    select, scope = _measurement_slice(serving)
    step = int(scope.get("beat", scope.get("decode_step", 0)))
    stage_of_layer: Dict[int, int] = {}
    for stage in getattr(mapping, "stages", ()):  # empty under lockstep
        for layer in stage.layers:
            stage_of_layer[int(layer)] = int(stage.index)
    ready = _ready_times(program, timeline)
    by_device: "OrderedDict[int, List[int]]" = OrderedDict()
    for cost in pricing.costs:
        # ANALOG MACROS ONLY: this block is about BANK sharing, so the resource
        # in question is the macro's ADC/mux path and nothing else. A pool op
        # queueing behind a norm is a pool question (P4.5 sizes that) and
        # putting it in this sum would answer a different question under this
        # name.
        if cost.device_class != "analog_macro":
            continue
        if timeline.finish_times[cost.uid] < 0:
            continue
        by_device.setdefault(int(cost.device_id), []).append(int(cost.uid))
    same_owner = 0.0
    cross_layer = 0.0
    cross_stage = 0.0
    within_stage = 0.0
    cross_tensor = 0.0
    delayed = 0
    for uids in by_device.values():
        for uid in uids:
            annotation = annotations[uid]
            if not select(annotation):
                continue
            gap_start = float(ready[uid])
            gap_end = float(timeline.start_times[uid])
            if gap_end <= gap_start:
                continue
            delayed += 1
            for other in uids:
                if other == uid:
                    continue
                overlap = min(float(timeline.finish_times[other]), gap_end) - max(
                    float(timeline.start_times[other]), gap_start
                )
                if overlap <= 0:
                    continue
                blocker = annotations[other]
                if blocker.owner == annotation.owner:
                    same_owner += overlap
                elif blocker.layer != annotation.layer:
                    cross_layer += overlap
                    mine = stage_of_layer.get(int(annotation.layer or -1), -1)
                    theirs = stage_of_layer.get(int(blocker.layer or -1), -2)
                    if mine != theirs:
                        cross_stage += overlap
                    else:
                        within_stage += overlap
                else:
                    cross_tensor += overlap
    sharing_owners = 0
    sharing_layers = 0
    for macro in mapping.macros:
        if not macro.tiles:
            continue
        if len({tile.owner for tile in macro.tiles}) > 1:
            sharing_owners += 1
        if len({tile.owner.layer for tile in macro.tiles}) > 1:
            sharing_layers += 1
    sharing_stages = 0
    if stage_of_layer:
        for macro in mapping.macros:
            if not macro.tiles:
                continue
            if len({stage_of_layer.get(int(tile.owner.layer), -1) for tile in macro.tiles}) > 1:
                sharing_stages += 1
    block = OrderedDict(
        (
            ("measured", True),
            ("decode_step", step),
            ("macros_sharing_banks_across_tensors", sharing_owners),
            ("macros_sharing_banks_across_layers", sharing_layers),
            ("macros_sharing_banks_across_stages", sharing_stages),
            ("delayed_ops", delayed),
            ("same_owner_delay_s", same_owner),
            ("cross_layer_delay_s", cross_layer),
            ("cross_stage_delay_s", cross_stage),
            ("within_stage_cross_layer_delay_s", within_stage),
            ("cross_tensor_same_layer_delay_s", cross_tensor),
            (
                "basis",
                f"for every ANALOG op of {scope['scope']} that started later than its "
                "last dependency finished, the waiting interval is attributed to the ops "
                "that occupied the same macro during it, split by whose weights they "
                "are. WITHIN A STAGE the layers run in sequence for the one stream the "
                "stage holds, so within_stage_cross_layer_delay_s == 0 is the "
                "serial-free half of D29's statement, measured. ACROSS STAGES every "
                "stage fires on the same beat for a different stream, so "
                "cross_stage_delay_s is the contention D29 says a shared bank now pays "
                "every beat — it can only be non-zero when a macro actually holds two "
                "stages' weights (macros_sharing_banks_across_stages), which the default "
                "one-stage-per-chip plan never produces and a finer layers_per_stage "
                "does. A non-zero same-layer term is two concurrent tensors of ONE layer "
                "sharing a macro; the timeline already prices it and this line is where "
                "a reader sees it.",
            ),
        )
    )
    block.update(scope)
    return block


def _utilization_disclosures(
    mapping: FwsMapping,
    utilization: Sequence[ClassUtilization],
    bank_passes: Mapping[str, object],
    bank_sharing: Mapping[str, object],
    program: Program,
) -> List[Relaxation]:
    """The findings D28 forbids hiding, as banners rather than footnotes.

    The first one is UNCONDITIONAL. A "low utilization" banner that only fires
    below some threshold would make the threshold the finding; every run states
    what every class did, and a reader compares the numbers themselves.
    """
    out: List[Relaxation] = []
    if utilization:
        binding = max(utilization, key=lambda row: row.max_occupancy)
        summary = ", ".join(
            f"{row.device_class} {row.mean_occupancy * 100:.3f}% mean"
            f" / {row.max_occupancy * 100:.3f}% peak"
            f" ({row.idle_devices} of {row.devices} idle)"
            for row in utilization
        )
        out.append(
            Relaxation(
                constraint="device_class_utilization",
                value=summary,
                reason=(
                    f"D28: idle silicon is a provisioning finding, never a fact to "
                    f"accept, so every device class prints its utilization on every "
                    f"run. The BINDING class here is {binding.device_class} at "
                    f"{binding.max_occupancy * 100:.3f}% peak occupancy; a class far "
                    "below it is silicon this provisioning does not need, and the "
                    "system-sizing sweep is where that is fixed. No threshold decides "
                    "when this banner appears — it always appears."
                ),
            )
        )
    if bank_passes.get("measured") and float(bank_passes.get("column_sets_never_read", 0)) > 0:
        out.append(
            Relaxation(
                constraint="idle_weight_space_in_decode",
                value=(
                    f"{bank_passes['column_sets_never_read']:.0f} of "
                    f"{bank_passes['column_sets_occupied']:.0f} occupied column sets "
                    f"were not read in decode step {bank_passes['decode_step']}"
                ),
                reason=(
                    "Invariant W (D27) says every bank holds real weights and decode "
                    "reads each weight once per token, so a column set the step never "
                    "activated is either weight space the decode path does not use or "
                    "an op the DAG does not lower. It is reported here rather than "
                    "averaged away."
                ),
            )
        )
    sharing = bank_sharing or {}
    if sharing.get("measured") and float(
        sharing.get("cross_tensor_same_layer_delay_s", 0)
    ) > 0:
        out.append(
            Relaxation(
                constraint="same_layer_bank_sharing_serializes",
                value=(
                    f"{float(sharing['cross_tensor_same_layer_delay_s']) * 1e6:.3f} us "
                    f"in decode step {sharing['decode_step']}"
                ),
                reason=(
                    "Two tensors of the SAME layer can be concurrent (a routed expert "
                    "beside a shared expert, a router beside a projection), and when "
                    "the packer lands them in one macro they serialize on its ADC "
                    "path. The timeline prices that; this banner is where a reader "
                    "sees which kind of sharing cost time. The CROSS-LAYER terms are "
                    "the other half and D29 splits them: within a stage the layers run "
                    "in sequence for the one stream the stage holds and the measured "
                    f"delay is {float(sharing.get('within_stage_cross_layer_delay_s', 0.0)) * 1e6:.3f} "
                    "us; across stages every stage fires on the same beat for a "
                    "different stream and the measured delay is "
                    f"{float(sharing.get('cross_stage_delay_s', 0.0)) * 1e6:.3f} us."
                ),
            )
        )
    if sharing.get("measured") and float(sharing.get("cross_stage_delay_s", 0)) > 0:
        out.append(
            Relaxation(
                constraint="cross_stage_bank_sharing_contends_every_beat",
                value=(
                    f"{float(sharing['cross_stage_delay_s']) * 1e6:.3f} us on "
                    f"{sharing.get('scope', 'the measured slice')}, on "
                    f"{int(sharing.get('macros_sharing_banks_across_stages', 0))} macros "
                    "holding two stages' weights"
                ),
                reason=(
                    "D29's consequence, MEASURED: the pipeline is always full, so every "
                    "stage fires on every beat for a different stream. A macro that "
                    "holds weights of two STAGES is therefore wanted twice in the same "
                    "beat and the second op waits. Under the retired lockstep regime "
                    "this term was structurally 0 (the layers ran one after another), "
                    "which is why the packer may place a cross-stage sharing that used "
                    "to be free and is not free any more. Within-stage sharing stays "
                    "serial-free and its own term says so."
                ),
            )
        )
    undescribed = int(program.meta.misc.get("fws_undescribed_local_k_stacks", 0) or 0)
    if undescribed:
        out.append(
            Relaxation(
                constraint="unpriced_local_k_stack",
                value=f"{undescribed} local K stack(s) carry no accumulator op",
                reason=(
                    "A macro holds several K blocks of one output block, so their "
                    "partials must be summed, but the placement that produced them is "
                    f"{mapping.packing!r} and only the dense packer emits the KStack "
                    "descriptor P7.2's accumulator law prices. The adds are therefore "
                    "NOT in this timeline. Pack densely (mapping.packing: dense) to "
                    "price them; the gap is disclosed rather than silently absorbed."
                ),
            )
        )
    return out


def _packing_block(mapping: FwsMapping) -> "OrderedDict[str, object]":
    """The mapping's placement law and, under dense packing, its waste (D27).

    Under the dedicated law there is no packer and therefore no waste figure:
    the block says which law ran and stops. Printing a waste% for a placement
    no packer produced would be an invented number.
    """
    summary = mapping.packing_summary()
    if not summary:
        return OrderedDict(
            (
                ("packing", mapping.packing),
                ("waste_reported", False),
                (
                    "basis",
                    "the dedicated placement law: each weight matrix starts on a fresh "
                    "macro. No packer ran, so this run has no waste accounting — "
                    "Invariant W's waste% is the DENSE packer's metric (D27) and a "
                    "number here would be invented. Set mapping.packing: dense to get "
                    "it.",
                ),
            )
        )
    block = OrderedDict((("packing", mapping.packing), ("waste_reported", True)))
    block.update(summary)
    block["basis"] = (
        "cim_timing.dense_pack per chip (Invariant W, D27): real + remainder + tail == "
        "committed, against the GLOBAL cell floor ceil(real_cells / cells_per_macro). "
        "dedicated_macros is what the same tensors reach under the per-tensor "
        "placement, so macros_saved == 0 IS the degenerate identity."
    )
    return block


def _measurement_slice(serving: ServingPoint) -> Tuple[object, "OrderedDict[str, object]"]:
    """The slice of the timeline the per-step measurements are taken on.

    Under the FILLED PIPELINE the unit is a BEAT, not a step: in one beat every
    stage fires once, for D DIFFERENT streams, so a beat is exactly the window
    in which the whole machine walks all of its banks once. The last lowered
    beat is chosen because it is the one with every stage occupied — the fill
    is behind it. Under the retired lockstep regime the unit is the last
    lowered decode step, which is what it always was.
    """
    if str(getattr(serving, "regime", "")) == REGIME_FILLED:
        beat = int(serving.beats) - 1

        def _select(annotation: FwsOpAnnotation) -> bool:
            return int(annotation.beat) == beat

        return _select, OrderedDict(
            (
                ("unit", "beat"),
                ("beat", beat),
                ("resident_streams", int(serving.streams)),
                (
                    "scope",
                    f"beat {beat} of {int(serving.beats)} lowered — every stage "
                    f"occupied, one stream each (D29)",
                ),
            )
        )
    step = int(serving.decode_steps) - 1

    def _select_step(annotation: FwsOpAnnotation) -> bool:
        return annotation.phase == "decode" and int(annotation.step) == step

    return _select_step, OrderedDict(
        (
            ("unit", "decode_step"),
            ("decode_step", step),
            ("scope", f"lowered decode step {step} (the retired lockstep regime)"),
        )
    )


def _build_bank_passes(
    mapping: FwsMapping,
    pricing: PricingResult,
    annotations: Sequence[FwsOpAnnotation],
    serving: ServingPoint,
) -> "OrderedDict[str, object]":
    """Does one decode step read every stored bank exactly once? (P7 Fact 1)

    Invariant W says every bank holds REAL WEIGHTS. It does NOT say every bank
    is read every beat, and on a sparse model it is not: a MoE layer routes one
    token to `top_k` experts, so the other E - top_k hold their weights and sit
    the beat out. `column_sets_never_read` counts exactly those, and on a MoE
    model a large value there is the CORRECT reading, not a violation - Granite
    occupies 19,304 column sets and fires 3,064 of them.

    What Invariant W forbids is WASTE: space that holds nothing. What this block
    measures is ACTIVITY: space that fires. Residency is not activity (M6), and
    conflating them charged every idle expert a full analog pass - 84% of the
    analog energy term on arrays that never switched.

    On a DENSE model the two do coincide, and then this block's original
    reading holds: nothing read twice, no bank sitting out. It MEASURES that on
    the priced timeline instead of asserting it: it sums the active column sets
    the analog law charged each macro during one lowered decode step and compares
    them with the column sets the placement actually claimed on that macro.

    A macro that comes up short is idle weight space and the block says so.
    """
    if serving.decode_steps <= 0:
        return OrderedDict(
            (
                ("measured", False),
                (
                    "basis",
                    "the run lowers no decode step, so there is no step to walk. "
                    "P7 is DECODE ONLY (D25) and this block reports nothing rather "
                    "than measuring a prefill pass it does not describe.",
                ),
            )
        )
    select, scope = _measurement_slice(serving)
    step = int(scope.get("beat", scope.get("decode_step", 0)))
    charged: Dict[int, float] = {}
    for cost in pricing.costs:
        annotation = annotations[cost.uid]
        if annotation.law != LAW_ANALOG_GEMM:
            continue
        if not select(annotation):
            continue
        macros = {tile.site.macro_id for tile in annotation.tiles}
        if len(macros) != 1:
            raise MappingError(
                "execution",
                f"analog op {annotation.uid} holds tiles on {len(macros)} macros. The "
                "bank-pass accounting reads the per-op charge as ONE macro's column "
                "sets, which is what the builder emits; an op spanning macros would "
                "make that reading wrong rather than approximate.",
            )
        macro_id = int(next(iter(macros)))
        charged[macro_id] = charged.get(macro_id, 0.0) + float(
            cost.detail.get("active_column_sets", 0.0)
        )
    owned = {
        int(macro.macro_id): int(macro.claimed_column_sets)
        for macro in mapping.macros
        if macro.tiles
    }
    matching = sum(
        1 for macro_id, sets in owned.items() if charged.get(macro_id, 0.0) == sets
    )
    unread = math.fsum(
        max(0.0, sets - charged.get(macro_id, 0.0)) for macro_id, sets in owned.items()
    )
    reread = math.fsum(
        max(0.0, charged.get(macro_id, 0.0) - sets) for macro_id, sets in owned.items()
    )
    block = OrderedDict(
        (
            ("measured", True),
            ("decode_step", step),
            ("macros_holding_tiles", len(owned)),
            ("macros_walked_exactly_once", int(matching)),
            ("column_set_passes_charged", math.fsum(charged.values())),
            ("column_sets_occupied", float(sum(owned.values()))),
            ("column_sets_never_read", float(unread)),
            ("column_sets_read_more_than_once", float(reread)),
            (
                "basis",
                "sum of the active-column-set charge (ADJ-4) over every analog op of "
                f"{scope['scope']}, per macro, against the column sets the placement "
                "claimed on that macro. Equality per macro is Fact 1: each macro walks "
                f"its occupied banks exactly once per {scope['unit']}, whichever tensors "
                "co-reside there. Under the filled pipeline (D29) a beat is the unit "
                "because every stage fires once per beat, each for a different stream — "
                "so the whole machine reads every stored weight once per beat and one "
                "token leaves.",
            ),
        )
    )
    block.update(scope)
    return block


def _busy_intervals(
    program: Program, timeline: CoarseEvalResult, costs: Sequence[OpCost]
) -> "OrderedDict[int, List[Tuple[float, float]]]":
    """Per-device [start, finish) intervals. Transfers occupy no device (D17)."""
    out: "OrderedDict[int, List[Tuple[float, float]]]" = OrderedDict()
    for cost in costs:
        if cost.kind == "transfer":
            continue
        start = timeline.start_times[cost.uid]
        finish = timeline.finish_times[cost.uid]
        if start < 0 or finish < 0:
            continue
        out.setdefault(cost.device_id, []).append((float(start), float(finish)))
    return out


def _ready_times(program: Program, timeline: CoarseEvalResult) -> List[float]:
    """When each op's last dependency finished — a pure DAG property.

    Not the schedule's start time: an op can be READY and still be waiting for
    its device. The gap between the two is exactly the queueing P4.5 sizes the
    per-macro pool against.
    """
    finish = timeline.finish_times
    out: List[float] = [0.0] * len(program.ops)
    for op in program.ops:
        if op.deps:
            out[op.uid] = max(finish[int(dep)] for dep in op.deps)
    return out


def _peak_overlap(intervals: Sequence[Tuple[float, float]], *, closed: bool = False) -> int:
    """Largest number of intervals covering one instant.

    ``closed`` includes both endpoints. It is what a DEMAND count needs: a pool
    op that costs no time (D12 absorbs the helpers) has a zero-length interval,
    and a half-open reading would count zero simultaneous demands at the very
    instant several of them arrive. The demand is real whether or not the law
    charges it time — that is exactly what P4.5 measures.
    """
    events: List[Tuple[float, int]] = []
    for start, finish in intervals:
        if finish < start:
            continue
        if finish == start and not closed:
            continue
        events.append((start, 1))
        events.append((finish, -1))
    if not events:
        return 0
    # closed: opens before closes at a tie, so touching intervals coexist.
    events.sort(key=lambda item: (item[0], -item[1] if closed else item[1]))
    peak = 0
    current = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def _peak_bytes(intervals: Sequence[Tuple[float, float, float]]) -> float:
    """High-water mark of [start, finish, bytes) live ranges."""
    events: List[Tuple[float, int, float]] = []
    for start, finish, size in intervals:
        if size <= 0 or finish < start:
            continue
        events.append((start, 1, size))
        events.append((finish, 0, -size))
    if not events:
        return 0.0
    # Allocations first at a tie: two ops finishing and starting at one instant
    # coexist, which is the conservative and honest reading of a high-water mark.
    events.sort(key=lambda item: (item[0], -item[1]))
    peak = 0.0
    current = 0.0
    for _, _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def _phase_window(
    costs: Sequence[OpCost], timeline: CoarseEvalResult, phase: str, step: Optional[int] = None
) -> Tuple[float, float]:
    """(first issue, last finish) of one phase (or one decode step) — measured."""
    first = math.inf
    last = -math.inf
    for cost in costs:
        if cost.phase != phase or (step is not None and cost.step != step):
            continue
        start = timeline.start_times[cost.uid]
        finish = timeline.finish_times[cost.uid]
        if start < 0 or finish < 0:
            continue
        first = min(first, float(start))
        last = max(last, float(finish))
    if first is math.inf:
        return (0.0, 0.0)
    return (first, last)


_CHIP_HOP_RE = re.compile(r"^act\.c\d+->c\d+$")


def _is_chip_hop(boundary_id: str) -> bool:
    """Is this `act.*` op the stage's OUTGOING CHIP HOP, or work inside it?

    `act.c1->c2` crosses a chip boundary and is the handoff that ends a stage.
    `act.L5.qkv_to_fabric` moves operands to the chiplet WITHIN the stage and
    fires several times a beat - it is the stage's own work, not its handoff.
    Treating every `act.*` as the outgoing link put intra-stage traffic into a
    max() that then discarded it: 2.04 us per beat on Granite's stage 2 was
    charged as neither compute nor link.
    """
    return bool(_CHIP_HOP_RE.match(str(boundary_id or "")))


#: D34: the largest stage-occupancy-to-median ratio a partition may reach
#: before the mapper is judged to have FAILED at its own job.
#:
#: WHY 1.5 AND NOT 1.0. A stage plan partitions LAYERS, so a perfectly even
#: split only exists when the layer count divides the stage count and every
#: layer costs the same. Neither holds on a hybrid: Granite-4.0-H is 9:1
#: mamba:attention, so one stage in ten carries an attention block the others
#: do not, and the granularity of the remedy is one whole layer. 1.5 is half a
#: stage of slack on a two-layer stage - the coarsest plan where a one-layer
#: move is still meaningful - and it is a THRESHOLD FOR A WARNING, never a
#: gate: nothing is refused for crossing it, the report says it crossed.
#: Measured spread that forced the rule: 28.7x on Granite-4.0-H-1B at 2
#: layers/chip, where 4 attention stages set the beat and 16 stages idle ~85%
#: of it.
STAGE_BALANCE_MAX_RATIO = 1.5


def _stage_beat_floor(
    annotations: Sequence[FwsOpAnnotation],
    costs: Sequence[OpCost],
    timeline: CoarseEvalResult,
    beats: int,
) -> Tuple[float, "OrderedDict[str, object]", "OrderedDict[str, object]"]:
    """The slowest STAGE OCCUPANCY per beat, with each link charged to its source.

    A STAGE IS NOT DONE UNTIL IT HAS HANDED OFF. The filled pipeline gives every
    stage one beat to do its work and pass the activation on, and there is
    nowhere to park a sequence in between - the machine has no buffer for
    tokens in flight, only the D stream slots it is built for. So a stage's
    per-beat occupancy is its own compute PLUS its outgoing inter-chip link, and
    the beat cannot be shorter than the slowest such stage.

    This is a RATE, measured as busy time over the beats lowered, so it does not
    move with the measurement window. An earlier attempt used Little's law on
    the LAST traversal's latency, which grows with the window on an
    over-subscribed pipeline and made the beat depend on an analysis knob (M10).
    Occupancy per beat does not have that failure mode.
    """
    beats = max(1, int(beats))
    compute: Dict[int, List[Tuple[float, float]]] = {}
    #: boundary_id -> duration, per stage. A stage may send SEVERAL distinct
    #: hops in one beat and they are serial, so they SUM. Taking the max left
    #: 2.04 us per beat on Granite's stage 2 charged as neither compute nor
    #: link (M8).
    link_by_id: Dict[int, Dict[str, float]] = {}
    #: how many times each stage actually FIRED in the lowered window. Stage s
    #: is lowered for traversals 0 .. beats-s-1, so it fires `beats - s` times,
    #: NOT `beats`. Dividing every stage by the window made identical Mamba
    #: stages show a 1.95x spread in service time that is purely the divisor.
    fires: Dict[int, set] = {}
    #: D34's evidence, gathered in the SAME pass and off the SAME intervals as
    #: the occupancy above, so the balance verdict and the beat floor are one
    #: accounting (D21). Per (stage, device_class, block) — what makes a long
    #: stage long — and per LAYER, which is the smallest unit a stage plan can
    #: move: a layer sits on exactly one stage, so no repartition can produce a
    #: stage shorter than its largest layer.
    terms: Dict[int, Dict[Tuple[str, str], List[Tuple[float, float]]]] = {}
    layer_spans: Dict[int, List[Tuple[float, float]]] = {}
    layer_fires: Dict[int, set] = {}
    for cost in costs:
        annotation = annotations[cost.uid] if cost.uid < len(annotations) else None
        if annotation is None:
            continue
        stage = int(getattr(annotation, "stage", -1) or 0)
        boundary = str(getattr(annotation, "boundary_id", "") or "")
        duration = float(getattr(cost, "duration_s", 0.0) or 0.0)
        fires.setdefault(stage, set()).add(int(getattr(annotation, "step", 0) or 0))
        if _is_chip_hop(boundary):
            # THE STAGE'S OUTGOING CHIP HOP: one token crosses it once per beat,
            # so its duration IS its per-beat occupancy.
            link_by_id.setdefault(stage, {})[boundary] = duration
            continue
        # The START comes from the TIMELINE, not the cost: an OpCost carries a
        # duration, and reading a start off it silently gave every op start 0,
        # which collapsed the union to one op's length.
        start = float(timeline.start_times[cost.uid])
        finish = float(timeline.finish_times[cost.uid])
        if finish < start:
            finish = start + duration
        compute.setdefault(stage, []).append((start, finish))
        terms.setdefault(stage, {}).setdefault(
            (str(cost.device_class), str(cost.block)), []
        ).append((start, finish))
        layer = getattr(annotation, "layer", None)
        if layer is not None:
            layer_spans.setdefault(int(layer), []).append((start, finish))
            layer_fires.setdefault(int(layer), set()).add(
                int(getattr(annotation, "step", 0) or 0)
            )
    rows: List["OrderedDict[str, object]"] = []
    floor = 0.0
    binding = None
    for stage in sorted(set(list(compute) + list(link_by_id))):
        fired = max(1, len(fires.get(stage, ())))
        busy = _union_busy_s(compute.get(stage, ())) / float(fired)
        link = float(sum(link_by_id.get(stage, {}).values()))
        total = busy + link
        rows.append(OrderedDict((
            ("stage", stage),
            ("beats_fired", fired),
            ("compute_s_per_beat", busy),
            ("outgoing_link_s", link),
            ("outgoing_hops", len(link_by_id.get(stage, {}))),
            ("occupancy_s_per_beat", total),
        )))
        if total > floor:
            floor, binding = total, stage
    balance = _stage_balance(rows, terms, layer_spans, layer_fires, binding)
    return floor, OrderedDict((
        ("floor_s", floor),
        ("binding_stage", binding),
        ("per_stage", rows),
        ("basis", (
            "a stage holds its slot until the handoff completes, so its per-beat "
            "occupancy is its own compute plus its outgoing inter-chip link; the "
            "beat is the slowest of those. There is no buffer for a sequence "
            "between stages, so a link that does not fit inside a beat is not "
            "hidden by pipelining - it sets the pace."
        )),
    )), balance


def _stage_balance(
    rows: Sequence["OrderedDict[str, object]"],
    terms: Mapping[int, Mapping[Tuple[str, str], Sequence[Tuple[float, float]]]],
    layer_spans: Mapping[int, Sequence[Tuple[float, float]]],
    layer_fires: Mapping[int, Sequence[object]],
    binding_stage: Optional[int],
) -> "OrderedDict[str, object]":
    """D34: is this partition BALANCED, and if not, what would a balanced one reach?

    George, on the rule this implements: "one of the stages is much longer than
    others. this causes pipeline waste and sets the beat... The entire point of
    the mapper is to *try* and eliminate mismatch even if it comes at an area
    cost." So a lopsided partition is a MAPPER FAILURE that has to be said out
    loud, not a property of the workload that gets reported as a number and
    read past.

    The measure is the per-beat stage occupancy the beat floor already uses
    (compute union plus outgoing link), so this is the same accounting that
    sets the beat and not a second reading of it.

    ``balanced_beat_bound_s`` is what a PERFECTLY BALANCED partition of this
    same work could reach, and it is the largest of three floors this point
    cannot get under by repartitioning alone:

    * the **unsplittable layer floor** — a layer sits on exactly one stage, so
      no stage can be shorter than the busiest single layer;
    * the **distributable floor** — the total compute spread evenly over the
      stages that exist;
    * the **link floor** — a chip hop is atomic, so the busiest outgoing link
      is a floor whether or not the compute around it moves.

    It is a BOUND, not a promise: it says what is left on the table, and D34
    names the balance-aware partitioner as the follow-on that would collect it.
    Nothing here repartitions anything.
    """
    occupancies = [float(row["occupancy_s_per_beat"]) for row in rows]
    if not occupancies:
        return OrderedDict(
            (
                ("measured", False),
                ("reason", (
                    "this run lowers no pipeline stage, so there is no partition to "
                    "balance and no stage times to compare."
                )),
            )
        )
    largest = max(occupancies)
    median = float(_median(occupancies))
    ratio = (largest / median) if median > 0 else 0.0
    stage_count = len(rows)

    def _dominant(stage: int) -> Tuple[Optional[str], Optional[str], float]:
        by_term = terms.get(int(stage)) or {}
        best: Tuple[Optional[str], Optional[str], float] = (None, None, 0.0)
        for (device_class, block), intervals in by_term.items():
            busy = _union_busy_s(list(intervals))
            if busy > best[2]:
                best = (block, device_class, busy)
        return best

    # THE DIGITAL-ONLY SPREAD, beside the occupancy spread. They are different
    # quantities and both are true: a hybrid's attention stages can run tens of
    # times the median DIGITAL time while their whole-stage occupancy is only a
    # few times the median, because every stage also walks its analog macros.
    # The occupancy ratio is the one the verdict is taken on - it is what sets
    # the beat - and this one is here so a reader holding the digital figure
    # (28.7x on Granite-4.0-H-1B at 2 layers/chip) can see it is the same
    # machine and not a contradiction (D21).
    digital_busy: List[float] = []
    for row in rows:
        by_term = terms.get(int(row["stage"])) or {}
        intervals: List[Tuple[float, float]] = []
        for (device_class, _block), spans in by_term.items():
            if device_class != "analog_macro":
                intervals.extend(spans)
        fired = max(1, int(row["beats_fired"]))
        digital_busy.append(_union_busy_s(intervals) / float(fired))
    digital_max = max(digital_busy, default=0.0)
    digital_median = float(_median(digital_busy)) if digital_busy else 0.0
    digital_ratio = (digital_max / digital_median) if digital_median > 0 else None

    long_stages: List["OrderedDict[str, object]"] = []
    for row in rows:
        busy = float(row["occupancy_s_per_beat"])
        if median <= 0 or busy <= median * STAGE_BALANCE_MAX_RATIO:
            continue
        block, device_class, term_busy = _dominant(int(row["stage"]))
        long_stages.append(
            OrderedDict(
                (
                    ("stage", int(row["stage"])),
                    ("busy_s", busy),
                    ("ratio_to_median", busy / median),
                    ("dominant_block", block),
                    ("dominant_device_class", device_class),
                    ("dominant_busy_s", term_busy),
                    ("outgoing_link_s", float(row["outgoing_link_s"])),
                    ("sets_the_beat", int(row["stage"]) == (
                        -1 if binding_stage is None else int(binding_stage)
                    )),
                )
            )
        )
    long_stages.sort(key=lambda entry: -float(entry["busy_s"]))

    # THE THREE FLOORS A REPARTITION CANNOT GET UNDER.
    layer_floor = 0.0
    busiest_layer = None
    for layer, spans in layer_spans.items():
        fired = max(1, len(layer_fires.get(layer, ()) or ()))
        busy = _union_busy_s(list(spans)) / float(fired)
        if busy > layer_floor:
            layer_floor, busiest_layer = busy, int(layer)
    compute_total = math.fsum(float(row["compute_s_per_beat"]) for row in rows)
    distributable_floor = compute_total / float(max(1, stage_count))
    link_floor = max(
        (float(row["outgoing_link_s"]) for row in rows), default=0.0
    )
    bound = max(layer_floor, distributable_floor, link_floor)
    which = "unsplittable_layer"
    if distributable_floor >= layer_floor and distributable_floor >= link_floor:
        which = "distributable_work_over_stages"
    elif link_floor >= layer_floor and link_floor >= distributable_floor:
        which = "chip_link"
    balanced = ratio <= STAGE_BALANCE_MAX_RATIO
    return OrderedDict(
        (
            ("measured", True),
            ("verdict", "balanced" if balanced else "MISMATCHED"),
            ("max_stage_busy_s", largest),
            ("median_stage_busy_s", median),
            ("ratio", ratio),
            ("max_stage_digital_s", digital_max),
            ("median_stage_digital_s", digital_median),
            ("digital_ratio", digital_ratio),
            ("threshold", float(STAGE_BALANCE_MAX_RATIO)),
            ("stages", stage_count),
            ("binding_stage", None if binding_stage is None else int(binding_stage)),
            ("long_stages", long_stages),
            ("balanced_beat_bound_s", bound),
            ("balanced_beat_bound_binding", which),
            ("unsplittable_layer_floor_s", layer_floor),
            ("unsplittable_layer", busiest_layer),
            ("distributable_floor_s", distributable_floor),
            ("link_floor_s", link_floor),
            (
                "headroom_ratio",
                (largest / bound) if bound > 0 else None,
            ),
            (
                "basis",
                "D34. Balance is measured on the SAME per-beat stage occupancy that "
                "sets the beat (compute union plus outgoing link), so this is one "
                "accounting and not a second reading. `digital_ratio` is the same "
                "spread over the NON-ANALOG time only - a hybrid's attention stages "
                "run tens of times the median digital time while their whole-stage "
                "occupancy is a few times it, because every stage also walks its "
                "analog macros; the verdict is taken on the OCCUPANCY, which is what "
                "sets the beat. verdict is MISMATCHED when the "
                f"busiest stage exceeds {STAGE_BALANCE_MAX_RATIO}x the median - a "
                "WARNING, never a gate: nothing is refused for it. "
                "balanced_beat_bound_s is the beat a perfectly balanced partition of "
                "THIS work could reach, the largest of three floors a repartition "
                "cannot get under: the busiest single LAYER (a layer sits on exactly "
                "one stage), the total compute spread evenly over the stages that "
                "exist, and the busiest outgoing CHIP HOP (a hop is atomic). It is a "
                "bound on what is left on the table; nothing here repartitions "
                "anything, and D34 names the balance-aware partitioner as the "
                "follow-on that would collect it.",
            ),
        )
    )


def _build_pipeline_metrics(
    mapping: FwsMapping,
    serving: ServingPoint,
    costs: Sequence[OpCost],
    timeline: CoarseEvalResult,
    op_annotations: Sequence[FwsOpAnnotation] = (),
) -> Tuple[Tuple[Metric, ...], Tuple[float, ...], Dict[str, object], "OrderedDict[str, object]"]:
    """The FILLED PIPELINE's metrics (D29), every one read off the timeline.

    The beat is not assumed and it is not a max-stage-time formula: it is the
    MEASURED interval between consecutive token EXITS. Traversal j exits when
    its last stage finishes, so the exits are timeline finishes and the beat is
    their difference — which is where "throughput = 1/beat" comes from. The
    first D - 1 beats are the pipeline FILLING and the transient is reported
    beside the beat instead of being averaged into it.
    """
    system = mapping.system_id
    streams = max(1, int(serving.streams))
    stage_count = max(1, int(getattr(serving, "stages", 0) or len(mapping.stages)))
    fill = float(streams) / float(stage_count)
    beats_lowered = int(serving.beats)
    # A traversal COMPLETES inside the window when its LAST STAGE fits: it
    # enters at its own entry beat (one per beat at the full fill, stages/D
    # beats apart under an underfilled pipeline) and leaves `stages - 1` beats
    # later. The window is sized from the same law (`lowered_beats`), so this
    # is that law read back, not a second one.
    complete = [j for j in range(beats_lowered) if serving.completes(j)]
    exits: List[Tuple[int, float, float]] = []
    for traversal in complete:
        first, last = _phase_window(costs, timeline, "decode", traversal)
        if last <= 0:
            continue
        exits.append((traversal, float(first), float(last)))
    intervals = [
        float(exits[index][2] - exits[index - 1][2]) for index in range(1, len(exits))
    ]
    # THE FILL TRANSIENT, held out rather than averaged in. The first completed
    # traversal is the one that entered at beat 0: it walked the stages while
    # the pipeline was still filling, so nothing was ever waiting behind it and
    # its exit lands EARLY. Every later exit is spaced by the machine's own
    # steady rhythm. The transient interval is reported beside the beat instead
    # of being smoothed into it (D28: a transient is a fact, not noise).
    steady = intervals[1:] if len(intervals) > 1 else intervals
    # THE BEAT IS READ OFF THE SETTLED TAIL (P7.9). Holding out a fixed count
    # of fill intervals is exact only for a pipeline of equal-service stages;
    # an uneven stage plan is still ramping after D - 1 beats, and a median
    # taken over a sample that still contains ramp moves with the window size
    # (12% on Granite between decode_window 2 and 3). The tail is the longest
    # suffix of the steady sample that agrees with its own median, so the
    # reported beat is the machine's periodic rhythm and not an average of the
    # ramp and the rhythm. Every dropped interval is printed below.
    tail = list(_converged_beat_tail(steady))
    ramp = steady[: len(steady) - len(tail)]
    beat = float(_median(tail)) if tail else 0.0
    # LITTLE'S LAW IS A FLOOR ON THE BEAT, NOT A DIAGNOSTIC (M1/M4).
    #
    # A filled pipeline holds D streams. A token whose traversal takes L cannot
    # be replaced faster than L / D, because at 1/beat tokens per second there
    # would be (L / beat) tokens in flight and the machine has room for D. The
    # exit intervals alone cannot see this: they measure how fast tokens LEAVE
    # inside a bounded window, and a window that never reaches steady state
    # reports the ramp's spacing as the rhythm.
    #
    # Falcon-Mamba read 1,208,651 tokens/s off a 0.827 us exit spacing while its
    # own traversal took 59.33 us over 32 streams - 71.7 tokens in flight in a
    # machine built for 32. The cause is structural, not a short window: a token
    # crosses 32 compute stages AND the 31 inter-chip links between them, and
    # only the stages were counted as pipeline positions. Widening the window
    # from 3 to 16 beats does not converge it.
    #
    # The floor is applied and NAMED. When it binds, the analog side is not what
    # sets the pace and the artifact says so.
    # LITTLE'S LAW IS CHECKED AND REPORTED, NEVER SILENTLY IMPOSED.
    #
    # A filled pipeline holds D streams, so a token whose traversal takes L
    # cannot be replaced faster than L / D. The exit spacing can report a faster
    # rhythm than that, and when it does the machine is NOT in steady state -
    # tokens are entering faster than it drains them.
    #
    # An earlier revision of this block took the max and called it the beat.
    # That was wrong: the traversal latency itself GROWS with the measurement
    # window on an over-subscribed pipeline (Granite moved 1.53e-05 -> 2.51e-05
    # between decode_window 3 and 14), so the "fixed" beat simply inherited the
    # divergence and made a built number depend on an analysis knob (M10).
    #
    # So the violation is NAMED and the measured beat is left alone. A reader
    # who sees `beat_sustainable = False` must not quote the throughput.
    # OPTION 2, TAKEN: a stage's time includes its outgoing link. The beat can
    # never be shorter than the slowest stage's per-beat occupancy, because the
    # machine has no buffer to park a sequence in between stages. The exit
    # spacing can read faster than that while the pipeline is still filling, so
    # the floor is applied and the fact is disclosed.
    stage_floor, stage_occupancy, stage_balance = _stage_beat_floor(
        op_annotations, costs, timeline, beats_lowered
    )
    # A MEASUREMENT IS USED ONLY WHEN IT CONVERGED; otherwise the derived bound
    # stands. The exit spacing is one sample of the rhythm and it needs a window
    # long enough to settle - Granite's never settles (its last two intervals
    # disagree by 51% at window 3 and still by 62% at window 14), so reading a
    # beat off it reported the ramp. The stage-occupancy floor is a RATE, busy
    # time over beats lowered, and it is stable to 0.5% across those same
    # windows.
    #
    # When the exits DO converge and sit above the floor, the machine really is
    # stalling on dependencies rather than on resources, and the slower measured
    # rhythm is the honest one. So: converged exits win when they are slower,
    # and the floor stands whenever they are faster or unusable.
    # TWO PHYSICAL FLOORS, AND THE BEAT IS THE LARGER.
    #
    #   1. RESOURCE OCCUPANCY. No stage can be busy more than a whole beat, and
    #      a stage holds its slot until its outgoing link completes - there is
    #      no buffer to park a sequence in between stages.
    #   2. LITTLE'S LAW. A traversal takes L and the machine has D stream slots,
    #      so a token cannot be replaced faster than L / D.
    #
    # Either can bind. Falcon-Mamba is occupancy-bound (its inter-chip link is
    # longer than the analog pass). Granite and Qwen are Little-bound: their
    # resources idle 56% of the beat while the traversal is what limits them.
    # Applying only the first raised Granite to 81,203 tokens/s with 21 tokens
    # in flight in a machine built for 10.
    #
    # The measured exit spacing is used only when it CONVERGED, and then only if
    # it is slower than both floors - a converged, slower rhythm means the
    # machine really stalls on dependencies, and that is the honest reading.
    # Granite's exits never converge (51% disagreement at window 3, still 62% at
    # window 14), so reading a beat off them reported the ramp.
    little_floor = 0.0
    settled_latency_s = 0.0
    if streams > 0 and exits:
        # THE MEDIAN OF THE COMPLETED TRAVERSALS, not the last one. A single
        # traversal is one sample of a quantity that is still settling, and
        # reading the floor off it made the beat swing 2x with the measurement
        # window (Granite: 2.59e-05 at window 3 against 1.32e-05 at window 14).
        # The median over the settled half is stable and is still a measurement,
        # not a smoothing of ramp into rhythm: the fill transient sits in the
        # first half and is dropped whole.
        spans = sorted(float(last - first) for _t, first, last in exits)
        settled = spans[len(spans) // 2:] if len(spans) > 2 else spans
        settled_latency_s = float(_median(settled))
        little_floor = settled_latency_s / float(streams)
    exits_converged = bool(len(tail) >= 2)
    floor = max(float(stage_floor), float(little_floor))
    beat_binding = "exit_intervals"
    if not exits_converged or floor > beat > 0.0:
        if floor > 0.0:
            beat = floor
            beat_binding = (
                "stage_occupancy_including_outgoing_link"
                if stage_floor >= little_floor
                else "littles_law_traversal_over_streams"
            )
    tokens_in_flight = (float(little_floor) * float(streams) / beat) if beat > 0 else 0.0
    beat_sustainable = not (beat > 0.0 and little_floor > beat * (1.0 + 1e-9))
    metrics: List[Metric] = []
    pipeline: "OrderedDict[str, object]" = OrderedDict(
        (
            ("regime", str(serving.regime)),
            ("resident_streams", streams),
            ("stage_count", stage_count),
            # D38: the DECLARED fill and the one D realizes over the stages.
            ("pp_util", float(getattr(mapping, "pp_util", PP_UTIL_FULL))),
            ("pipeline_fill", fill),
            (
                "fill_basis",
                "D = round(pp_util x stages) resident streams over "
                f"{stage_count} stage(s), realized fill {fill:.6g}. The fill enters "
                "the machine ONCE, in the DAG's rotation: traversals are spaced "
                "stages/D beats apart, so D streams are in flight and D tokens leave "
                "every `stages` beats. Every number below is then MEASURED off that "
                "timeline - no throughput, occupancy or state figure is multiplied by "
                "the fill anywhere, because that would be a second accounting of one "
                "fact (D21).",
            ),
            ("stage_plan_basis", str(mapping.stage_basis)),
            ("stages", [stage.as_dict() for stage in mapping.stages]),
            ("beat_binding", beat_binding),
            ("stage_occupancy", stage_occupancy),
            # D34: is this partition balanced, and what would a balanced one
            # reach? Beside the occupancy it is measured from.
            ("stage_balance", stage_balance),
            ("beat_sustainable", bool(beat_sustainable)),
            ("beat_littles_law_floor_s", float(little_floor)),
            ("tokens_in_flight", float(tokens_in_flight)),
            ("beat_binding_note", (
                "the exit spacing measured on the timeline, and it satisfies "
                "Little's law against the resident stream count"
                if beat_sustainable else
                "THIS BEAT IS NOT SUSTAINABLE AND THE THROUGHPUT MUST NOT BE QUOTED. "
                "The exit spacing reports a faster rhythm than the machine can hold: at "
                "this beat there are `tokens_in_flight` tokens inside a pipeline built "
                "for `resident_streams`. A token crosses the compute stages AND the "
                "inter-chip links between them, and only the stages are counted as "
                "pipeline positions, so the link chain has nowhere to live. Either the "
                "links become stages (and D grows, with the state store that implies), "
                "or a stage's time must include its outgoing link. That is a modelling "
                "decision, and this model does not take it silently."
            )),
            ("beats_lowered", beats_lowered),
            ("fill_beats", stage_count - 1),
            ("steady_beats_measured", len(steady)),
            ("tokens_exited", len(exits)),
            ("exit_times_s", [row[2] for row in exits]),
            ("beat_intervals_s", list(intervals)),
            ("steady_beat_intervals_s", list(steady)),
            ("converged_tail_intervals_s", list(tail)),
            ("ramp_beat_intervals_s", list(ramp)),
            (
                "transient_beat_intervals_s",
                list(intervals[: len(intervals) - len(steady)]),
            ),
            ("beat_s", beat),
            (
                "beat_basis",
                "the MEDIAN of the SETTLED TAIL of the measured intervals between "
                f"consecutive token exits: {len(tail)} of the {len(intervals)} measured "
                "interval(s). One interval is held out as the traversal that entered at "
                "beat 0 and travelled through a filling pipeline (nothing queued behind "
                f"it, so its exit lands early), and a further {len(ramp)} interval(s) "
                "are dropped as RAMP because they do not agree with the tail's own "
                f"median to {BEAT_CONVERGENCE_TOL:.0e}. Both sets are printed above "
                "rather than smoothed in, and the tail rule is what makes the reported "
                "beat independent of how many beats the window lowered (P7.9). An exit "
                "is the finish of the last op of a completed traversal on the one "
                "timeline, so the beat is a difference of two timeline readings and "
                "never a max-stage-time formula (A1, P4 3).",
            ),
        )
    )
    if beat > 0:
        metrics.append(
            Metric(
                key=f"{system}.beat",
                label="pipeline beat (one token exits per beat)",
                value=beat,
                unit="s",
                basis=str(pipeline["beat_basis"]),
            )
        )
        metrics.append(
            Metric(
                key=f"{system}.tokens_per_s",
                label="steady throughput = 1 / beat (HEADLINE)",
                value=1.0 / beat,
                unit="tokens/s",
                basis=(
                    "D29's identity, on the measured beat: ONE token leaves the last "
                    "stage per beat and the system rate is 1/beat. Under an UNDERFILLED "
                    "pipeline (D38) the exit spacing is what carries the fill - "
                    f"{streams} stream(s) over {stage_count} stage(s) means D tokens "
                    "leave every `stages` beats, so the measured spacing is already the "
                    "full pipeline's beat divided by the fill and nothing is scaled "
                    f"here. The beat is the median of the SETTLED TAIL "
                    f"({len(tail)} interval(s)) of the {len(intervals)} measured "
                    "inter-exit intervals; the fill transient and any still-ramping "
                    "intervals are held out and printed beside it, and this headline is "
                    "only a settled measurement when pipeline.beat_converged is true — "
                    "the run says so by name when it is not. Nothing here is a period "
                    "times a count."
                ),
            )
        )
        metrics.append(
            Metric(
                key=f"{system}.per_stream_tokens_per_s",
                label="per-stream token rate",
                value=1.0 / (beat * streams),
                unit="tokens/s",
                basis=(
                    f"1 / (D x beat) with D = {streams} resident streams (D29). A stream "
                    "gets the machine once every D beats, which is the price of keeping "
                    "the pipeline full."
                ),
            )
        )
    metrics.append(
        Metric(
            key=f"{system}.resident_streams",
            label="resident streams D (= pipeline stages)",
            value=float(streams),
            unit="streams",
            basis=(
                f"round(pp_util x stages) = round({float(getattr(mapping, 'pp_util', PP_UTIL_FULL))} "
                f"x {stage_count}), on the stage plan DERIVED from "
                f"{mapping.stage_basis or 'the stage plan'}. D29 makes D a consequence "
                "of the stage plan and never a configured batch — the serving surface "
                "refuses a batch by name — and D38 makes the FILL the one axis that "
                "moves it."
            ),
        )
    )
    if exits:
        # ONE LATENCY, ONE NUMBER (D21). The SETTLED MEDIAN of the completed
        # traversals is the reading, and it is taken BEFORE the metric is
        # built: the metric used to publish the LAST traversal while the beat
        # floor was derived from the median, so the two disagreed and the
        # reported tokens-in-flight could exceed D by a few percent on a
        # machine the floor had already made legal. The last traversal is one
        # sample of a quantity that is still settling; on a 12-stage plan the
        # two readings part by 0.03%, on a 2-stage one they coincided and hid
        # the split.
        traversal, first, last = exits[-1]
        measured_latency = float(last - first)
        if settled_latency_s > 0.0:
            measured_latency = settled_latency_s
        metrics.append(
            Metric(
                key=f"{system}.per_token_latency",
                label="per-token latency (one stream's traversal of the pipeline)",
                value=measured_latency,
                unit="s",
                basis=(
                    "the MEDIAN of the settled half of the completed traversals' spans "
                    "(finish of a traversal's last op minus the issue of its first, "
                    f"both read off the one timeline; {len(exits)} traversal(s) "
                    "completed, the fill transient sits in the first half and is "
                    "dropped whole): the wall time one stream needs to walk every stage "
                    "and emit one token. It is the SAME number the Little's-law floor "
                    "is computed from, so the two can never disagree. D29's identity "
                    "says this is D x beat; the identity residual is reported in the "
                    "pipeline block rather than replacing the measurement."
                ),
            )
        )
        pipeline["per_token_latency_s"] = measured_latency
        pipeline["identity_d_times_beat_s"] = float(streams * beat)
        pipeline["identity_residual_s"] = float(measured_latency - streams * beat)
        pipeline["identity_residual_rel"] = (
            float((measured_latency - streams * beat) / measured_latency)
            if measured_latency > 0
            else 0.0
        )
        pipeline["identity_basis"] = (
            "D x beat is D29's IDENTITY for the per-token latency, and it holds exactly "
            "when every stage takes one beat. The metric is the MEASURED traversal "
            "instead, because a metric that is a period times a count is forbidden (P4 "
            "3) and because the two differ for a real reason: the SLOWEST stage sets the "
            "beat while a traversal pays the SUM of its stages, and a stage is released "
            "the moment it is done rather than held for a whole beat. An uneven stage "
            "plan therefore measures a traversal SHORTER than D x beat, and the residual "
            "printed here is exactly that skew — it is the stage plan's imbalance, "
            "reported, not an error term."
        )
    extrapolation: Dict[str, object] = OrderedDict(
        (
            ("lowered_decode_steps", len(intervals)),
            ("declared_decode_steps", int(serving.decode_len)),
            ("extrapolated", False),
        )
    )
    if beat > 0 and int(serving.decode_len) > 0:
        extrapolation.update(
            (
                ("extrapolated", True),
                ("remaining_steps", max(0, int(serving.decode_len) - 1)),
                ("median_step_s", beat),
                (
                    "extrapolated_request_latency_s",
                    float(serving.decode_len) * float(streams) * beat,
                ),
                (
                    "basis",
                    "DECODE_LEN x D x BEAT. A request occupies one stream and gets a "
                    "token every D beats, so its decode phase is decode_len x D x beat. "
                    "This is an EXTRAPOLATION — a period times a count — so it lives "
                    "here, labeled, and never enters metrics[] (P4 3, ADJ-6). Prefill is "
                    "not in it: D25/D29 make the mapped run decode-only and the streams "
                    "arrive already prefilled.",
                ),
            )
        )
    return tuple(metrics), tuple(intervals), extrapolation, pipeline


def _build_metrics(
    mapping: FwsMapping,
    serving: ServingPoint,
    costs: Sequence[OpCost],
    timeline: CoarseEvalResult,
) -> Tuple[Tuple[Metric, ...], Tuple[float, ...], Dict[str, object]]:
    """The five metrics plus the decode series — every one a projection (P4 §3)."""
    system = mapping.system_id
    release, makespan = _phase_window(costs, timeline, "prefill")
    has_prefill = any(cost.phase == "prefill" for cost in costs)
    if not has_prefill:
        release = 0.0
    prefill_terminal = makespan if has_prefill else 0.0
    decode_first, decode_terminal = _phase_window(costs, timeline, "decode")
    has_decode = any(cost.phase == "decode" for cost in costs)
    decode_start = prefill_terminal if has_prefill else decode_first

    metrics: List[Metric] = []
    if has_prefill:
        metrics.append(
            Metric(
                key=f"{system}.prefill_latency",
                label="prefill latency",
                value=float(prefill_terminal - release),
                unit="s",
                basis=(
                    "finish of the LAST prefill op minus the first prefill issue, both "
                    "read off the one timeline. It is measured, never assembled from "
                    "layers x block latency + endpoints (P4 §3)."
                ),
            )
        )

    step_times: List[float] = []
    if has_decode:
        previous = decode_start
        for step in range(int(serving.decode_steps)):
            _, terminal = _phase_window(costs, timeline, "decode", step)
            if terminal <= 0:
                continue
            step_times.append(float(terminal - previous))
            previous = terminal
    if step_times:
        metrics.extend(
            (
                Metric(
                    key=f"{system}.decode_step_first",
                    label="decode step time, first lowered step",
                    value=float(step_times[0]),
                    unit="s",
                    basis=(
                        "finish(step 0 terminal op) minus the decode window's start "
                        "(the prefill terminal). A difference of two timeline finishes, "
                        "never a period."
                    ),
                ),
                Metric(
                    key=f"{system}.decode_step_median",
                    label="decode step time, median of the lowered window",
                    value=float(_median(step_times)),
                    unit="s",
                    basis=(
                        f"median of the {len(step_times)} step differences "
                        "finish(step k) - finish(step k-1) on this timeline (ADJ-6: the "
                        "series is reported first / median / last)."
                    ),
                ),
                Metric(
                    key=f"{system}.decode_step_last",
                    label="decode step time, last lowered step",
                    value=float(step_times[-1]),
                    unit="s",
                    basis="finish(last step) - finish(previous step) on this timeline.",
                ),
            )
        )

    # A WINDOWED decode run lowers fewer steps than the serving point declares,
    # so the timeline's makespan is NOT a request's end-to-end latency and the
    # batch does NOT complete inside it. Naming those two metrics as if it did
    # would put a second, contradicting value for request latency in the same
    # artifact beside extrapolation.extrapolated_request_latency_s (D21). The
    # keys and labels therefore state the span they actually measure. At
    # decode_window >= decode_len nothing is truncated and both keep their
    # unqualified names.
    truncated = bool(step_times) and int(serving.decode_len) > len(step_times)
    lowered = len(step_times)
    declared = int(serving.decode_len)
    metrics.append(
        Metric(
            key=(
                f"{system}.lowered_window_latency"
                if truncated
                else f"{system}.request_latency"
            ),
            label=(
                "prefill + lowered decode window span (NOT a full request)"
                if truncated
                else "end-to-end request latency"
            ),
            value=float(makespan_of(timeline) - release),
            unit="s",
            basis=(
                (
                    "finish of the last op of the whole DAG minus the first issue — "
                    "measured once on the critical path, never assembled from "
                    f"per-stage terms (P4 §3). This timeline lowers {lowered} of the "
                    f"{declared} declared decode steps, so it is the span of the "
                    "LOWERED WINDOW and not a request's end-to-end latency; the only "
                    "request-scale figure is extrapolation.extrapolated_request_"
                    "latency_s, which is labeled an extrapolation and never a metric."
                )
                if truncated
                else (
                    "finish of the last op of the whole DAG minus the first issue — "
                    "measured once on the critical path, never assembled from "
                    "per-stage terms (P4 §3)."
                )
            ),
        )
    )

    tokens = float(serving.batch) * float(len(step_times))
    span = float(decode_terminal - decode_start)
    if step_times and span > 0:
        metrics.append(
            Metric(
                key=f"{system}.tokens_per_s",
                label="steady throughput at the decode terminal (HEADLINE)",
                value=tokens / span,
                unit="tokens/s",
                basis=(
                    f"{tokens:g} generated tokens (batch {serving.batch} x "
                    f"{len(step_times)} lowered decode steps) divided by the OBSERVED "
                    f"span of the same timeline ({span:.6g} s, decode terminal minus "
                    "decode start). ADJ-6 makes this the headline; it is a division of "
                    "two timeline readings, not 1 / period, and the fabric-ceiling / "
                    "sustained pair retired with the closed form."
                ),
            )
        )
    total_span = float(makespan_of(timeline) - release)
    if total_span > 0:
        metrics.append(
            Metric(
                key=(
                    f"{system}.requests_per_s_lowered_window"
                    if truncated
                    else f"{system}.requests_per_s"
                ),
                label=(
                    "batch over the lowered window (NOT a completed-request rate)"
                    if truncated
                    else "completed requests per second"
                ),
                value=float(serving.batch) / total_span,
                unit="requests/s",
                basis=(
                    (
                        f"{serving.batch} same-length requests (D15) divided by the "
                        "observed span of the same timeline. ZERO of them COMPLETE in "
                        f"that span: only {lowered} of {declared} decode steps are "
                        "lowered, so this is a window rate carrying the batch size, not "
                        "a completed-request rate. It is NOT recomputed from the "
                        "extrapolated latency — that would be a period times a count, "
                        "which P4 §3 forbids in metrics[]."
                    )
                    if truncated
                    else (
                        f"{serving.batch} same-length requests (D15) divided by the "
                        "observed span of the same timeline. Printed beside the "
                        "headline (ADJ-6)."
                    )
                ),
            )
        )

    extrapolation: Dict[str, object] = OrderedDict(
        (
            ("lowered_decode_steps", int(len(step_times))),
            ("declared_decode_steps", int(serving.decode_len)),
            ("extrapolated", False),
        )
    )
    if truncated:
        median_step = _median(step_times)
        remaining = int(serving.decode_len) - len(step_times)
        extrapolation.update(
            (
                ("extrapolated", True),
                ("remaining_steps", remaining),
                ("median_step_s", float(median_step)),
                ("extrapolated_request_latency_s", float(total_span + remaining * median_step)),
                (
                    "basis",
                    f"MEDIAN LOWERED STEP x {remaining} REMAINING STEPS. This is an "
                    "EXTRAPOLATION, not a metric: it is a period times a count, which "
                    "P4 §3 forbids for any published metric, so it lives here, labeled, "
                    "and never enters metrics[]. ADJ-6 allows the window plus a disclosed "
                    "extrapolation; this field IS the disclosure.",
                ),
            )
        )
    return tuple(metrics), tuple(step_times), extrapolation


def makespan_of(timeline: CoarseEvalResult) -> float:
    """The one span every latency metric is measured against."""
    return float(timeline.total_time)


def _union_length(intervals: Sequence[Tuple[float, float]]) -> float:
    """Time at least one op occupied the device — the honest busy time.

    At capacity 1 it is the plain sum; above it, overlapping ops do not make a
    device more than 100% busy, and an occupancy over 1.0 would be a number no
    reader could interpret.
    """
    if not intervals:
        return 0.0
    total = 0.0
    ordered = sorted(intervals)
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    total += current_end - current_start
    return float(total)


def _build_occupancy(
    mapping: FwsMapping,
    pricing: PricingResult,
    timeline: CoarseEvalResult,
    program: Program,
) -> Tuple[Tuple[DeviceOccupancy, ...], Dict[int, float]]:
    """Per-device occupancy and per-macro duty cycles, both owner-attributed."""
    makespan = makespan_of(timeline)
    intervals = _busy_intervals(program, timeline, pricing.costs)
    counts: Dict[int, int] = {}
    for cost in pricing.costs:
        if cost.kind == "transfer":
            continue
        counts[cost.device_id] = counts.get(cost.device_id, 0) + 1
    rows: List[DeviceOccupancy] = []
    duty: Dict[int, float] = {}
    for device in mapping.devices:
        busy = _union_length(intervals.get(device.device_id, ()))
        resource = pricing.resources[device.device_id]
        rows.append(
            DeviceOccupancy(
                device_id=device.device_id,
                device_class=device.device_class,
                chip_id=device.chip_id,
                macro_id=device.macro_id,
                owner=resource.owner,
                busy_s=busy,
                ops=counts.get(device.device_id, 0),
                occupancy=(busy / makespan) if makespan > 0 else 0.0,
            )
        )
        if device.device_class == "analog_macro" and device.macro_id >= 0:
            duty[int(device.macro_id)] = (busy / makespan) if makespan > 0 else 0.0
    return tuple(rows), duty


_COVERAGE_RANK = {COVERAGE_COVERED: 0, COVERAGE_PARTIAL: 1, COVERAGE_UNCOVERED: 2}

_ENERGY_LABELS: Mapping[str, str] = {
    "analog_arrays": "analog arrays (weight GEMM)",
    "link_traffic": "boundary and link traffic, PD handoff included",
    "digital_reduction": "priced shift-and-add reduction (D11)",
    "digital_accumulation": "in-macro K-stack accumulation (P7.2)",
    "macro_pool_helpers": "per-macro pool helpers (norm, activation, residual, short conv)",
    "shared_digital_chiplet": "shared digital chiplet (attention, scan, state update)",
    "kv_traffic": "KV cache traffic",
}

_ENERGY_ORDER = (
    "analog_arrays",
    "link_traffic",
    "digital_reduction",
    "digital_accumulation",
    "macro_pool_helpers",
    "shared_digital_chiplet",
    "kv_traffic",
)


def _build_energy(pricing: PricingResult) -> Tuple[EnergyComponent, ...]:
    """Itemized energy: one component, one coverage label, one basis (P4 §6)."""
    parts: Dict[str, List[float]] = {}
    coverage: Dict[str, set] = {}
    for cost in pricing.costs:
        key = cost.energy_component
        if not key:
            continue
        parts.setdefault(key, []).append(cost.energy_pj)
        coverage.setdefault(key, set()).add(cost.coverage)
    # fsum, not a running +=: the component IS the sum of its ops, and a
    # reader who re-adds the same ops must land on the same float.
    totals = {key: math.fsum(values) for key, values in parts.items()}
    bases = {
        "analog_arrays": (
            "sum of CimDeviceModel.price_tiled_op energies over every placed weight "
            "GEMM op: M x E_vec x shots x (claimed column sets / mux), the pass-1 law "
            "read per column set (ADJ-4)."
        ),
        "link_traffic": (
            "sum over every priced transfer of bytes x 8 x the network dimension's "
            "declared energy_per_bit. Intra-chip movement contributes nothing because "
            "no on-chip law is declared (disclosed)."
        ),
        "digital_reduction": (
            "sum of CimDeviceModel.price_reduction energies: adds x "
            "cim.cards.<card>.pool_energy_per_add_pj (0 when the card declares none)."
        ),
        "digital_accumulation": (
            "sum of CimDeviceModel.price_accumulation energies over the K stacks the "
            "packer put in ONE macro: (depth - 1) adds per output x "
            "cim.cards.<card>.pool_energy_per_add_pj (0 when the card declares none). "
            "A stack spread across macros contributes nothing here — the row-block "
            "partial-sum law prices that shape (D21)."
        ),
        "macro_pool_helpers": (
            "no law prices a norm, an activation, a residual add or a short depthwise "
            "conv on the per-macro pool; D12 absorbs them into the derived pool sizing, "
            "which P4.5 reports. The term is PARTIAL, not zero-because-free."
        ),
        "shared_digital_chiplet": (
            "no law prices a shared-digital-chiplet op's energy (the card schema refuses "
            "an energy_per_op_pj knob because nothing would read it). UNCOVERED."
        ),
        "kv_traffic": (
            "the placed DAG contains no KV read or write op — P3 places compute and "
            "transfers, and the KV story is a capacity/bandwidth story (P4 §5). There is "
            "nothing on this timeline to price, so the term is UNCOVERED rather than 0."
        ),
    }
    out: List[EnergyComponent] = []
    for key in _ENERGY_ORDER:
        if key not in totals and key != "kv_traffic":
            continue
        labels = coverage.get(key, {COVERAGE_UNCOVERED})
        # A component whose ops disagree is PARTIAL, not the worst of the two:
        # "some of this term is priced and some is not" is the true statement,
        # and it is the one the reader needs (ADJ-6).
        if labels == {COVERAGE_COVERED}:
            label = COVERAGE_COVERED
        elif COVERAGE_COVERED in labels or labels == {COVERAGE_PARTIAL}:
            label = COVERAGE_PARTIAL
        else:
            label = COVERAGE_UNCOVERED
        out.append(
            EnergyComponent(
                key=key,
                label=_ENERGY_LABELS[key],
                energy_pj=float(totals.get(key, 0.0)),
                coverage=label,
                basis=bases[key],
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# Memory feasibility (P4.3)
# ---------------------------------------------------------------------------

_ACTIVATION_BASIS = (
    "activation high-water: every op's OUTPUT is allocated at that op's finish and "
    "released at the finish of its last consumer, replayed over the SAME timeline the "
    "metrics project from (the program.memory_sim shape). Score blocks are NOT counted: "
    "the systolic array streams them straight into the softmax lanes, so only the "
    "attention output is resident. Weight bytes are not a memory question at all — "
    "weights live in the analog arrays (P2/P3 area)."
)


def _output_bytes(annotation: FwsOpAnnotation, mapping: FwsMapping, act_bytes: float) -> float:
    """Bytes one op's output occupies while it is live."""
    params = mapping.device.params
    tp = max(1, int(mapping.degrees.get("tp", 1)))
    tokens = float(annotation.tokens)
    if annotation.kind == "transfer":
        return float(annotation.bytes_moved)
    if annotation.tiles:
        width = max(tile.logical_columns for tile in annotation.tiles)
        return tokens * float(width) * act_bytes
    if annotation.block in _FABRIC_ATTENTION_BLOCKS:
        heads = max(1, int(params.num_heads) // tp)
        return tokens * heads * float(params.head_dim) * act_bytes
    if annotation.block == "ssm_scan":
        ssm = getattr(mapping.model, "ssm", None)
        width = (
            ssm.resolve_d_inner(int(params.hidden_dim))
            if ssm is not None
            else int(params.hidden_dim)
        )
        return tokens * float(width) / tp * act_bytes
    if annotation.block == "delta_rule":
        la = getattr(mapping.model, "linear_attention", None)
        width = la.value_dim if la is not None else int(params.hidden_dim)
        return tokens * float(width) / tp * act_bytes
    return tokens * float(params.hidden_dim) / tp * act_bytes


def _state_residency(mapping: FwsMapping, batch: int, act_bytes: float, block: str) -> float:
    """Resident recurrent state of one layer's mixer, per device."""
    params = mapping.device.params
    tp = max(1, int(mapping.degrees.get("tp", 1)))
    if block == "ssm_scan":
        ssm = getattr(mapping.model, "ssm", None)
        if ssm is None:
            return 0.0
        d_inner = ssm.resolve_d_inner(int(params.hidden_dim))
        # THE h-STATE AND THE CONV WINDOW. A selective-scan layer holds two
        # resident things per stream, not one: the recurrent state
        # (d_inner x d_state) and the short depthwise conv's ROLLING WINDOW of
        # the last (d_conv - 1) inputs over d_inner channels. The window was
        # missing, so a Mamba stage's bill - and the store sized from it - was
        # short by d_inner x (d_conv - 1) x act_bytes per stream.
        elems = float(d_inner) * float(ssm.d_state)
        window = max(0, int(getattr(ssm, "d_conv", 0) or 0) - 1)
        if window:
            elems += float(d_inner) * float(window)
        return float(batch) * elems / tp * act_bytes
    if block == "delta_rule":
        la = getattr(mapping.model, "linear_attention", None)
        if la is None:
            return 0.0
        # ONE LAW FOR ONE QUANTITY (M7). This used to multiply the three
        # declared dims straight through, which under-counts whenever value
        # heads outnumber key heads: each key head carries every value channel
        # it serves, so the state is key_head_dim x value_dim, not
        # num_key_heads x key_head_dim x value_head_dim. Qwen3.5 has 16 key
        # heads against 32 value heads, so the bill was 2.07x short - 686 mm2
        # of SRAM the machine has to build and was not being charged for. The
        # repo already had the right law and this site was not calling it.
        heads, d_k, d_v = _delta_state_dims(
            int(la.num_key_heads), int(la.key_head_dim),
            int(getattr(la, "num_value_heads", la.num_key_heads) or la.num_key_heads),
            int(la.value_head_dim),
        )
        elems = int(heads) * int(d_k) * int(d_v)
        # THE ROLLING CONV WINDOW rides with the mixer that declares it: a
        # k-tap depthwise conv must keep k-1 past activations per channel, and
        # that state is as resident as the fast weights are (ADJ-15).
        taps = int(getattr(la, "conv_kernel", 0) or 0)
        if taps > 1:
            channels = int(heads) * int(d_v)
            elems += channels * (taps - 1)
        return float(batch) * float(elems) / tp * act_bytes
    return 0.0


def _resident_contexts(serving: ServingPoint) -> List[int]:
    """The D resident streams' decode contexts at the LAST lowered beat (D29).

    The convention (disclosed on every filled-pipeline run): stream ``i``
    enters at ``prefill_len + i + 1`` and advances one token every D beats, so
    at beat ``B - 1`` it has completed ``floor((B - 1 - i) / D)`` further
    traversals. This is the same arithmetic the LOWERING stamps on each op, so
    the resident-memory figures and the priced attention costs are one
    accounting, not two.
    """
    streams = max(1, int(serving.streams))
    beats = max(1, int(serving.beats))
    out: List[int] = []
    for index in range(streams):
        turns = max(0, (beats - 1 - index) // streams)
        out.append(int(serving.prefill_len) + index + turns + 1)
    return out


def _served_contexts(serving: ServingPoint) -> List[int]:
    """The context each resident stream reaches over the WHOLE SERVED REQUEST.

    THE SILICON IS NOT SIZED BY THE MEASUREMENT WINDOW (M10). The store must
    hold every resident stream at its LARGEST context, and the largest context
    a stream reaches is the end of the request it was given -
    `prefill_len + decode_len` - not wherever the lowered beat count happened
    to stop. Sizing from `_resident_contexts` made `mapping.decode_window` -
    a pure analysis knob (ADJ-6) - buy SRAM macros: raising it added 7,542
    macros (399.7 mm2) on Qwen. A metric may be windowed; built silicon may
    not.

    `_resident_contexts` keeps its job of describing the priced ops, where the
    window reading is the honest one.
    """
    streams = max(1, int(serving.streams))
    full = int(serving.prefill_len) + int(getattr(serving, "decode_len", 0) or 0)
    window = _resident_contexts(serving)
    reached = max(window) if window else full
    return [max(full, reached)] * streams
    return out


def _build_memory(
    mapping: FwsMapping,
    serving: ServingPoint,
    program: Program,
    pricing: PricingResult,
    timeline: CoarseEvalResult,
) -> Tuple[Tuple[MemoryVerdict, ...], Tuple[Relaxation, ...]]:
    """Activation / state / KV high-water marks against the declared capacities."""
    annotations = annotations_of(program)
    act_bytes = float(mapping.hw.sw_config.precision.activations)
    kv_precision = float(mapping.hw.sw_config.precision.kv_cache)
    tp = max(1, int(mapping.degrees.get("tp", 1)))
    makespan = makespan_of(timeline)
    disclosures: List[Relaxation] = []

    consumers: List[List[int]] = [[] for _ in program.ops]
    for op in program.ops:
        for dep in op.deps:
            consumers[int(dep)].append(op.uid)

    live: Dict[int, List[Tuple[float, float, float]]] = {}
    for cost in pricing.costs:
        annotation = annotations[cost.uid]
        finish = timeline.finish_times[cost.uid]
        if finish < 0:
            continue
        size = _output_bytes(annotation, mapping, act_bytes)
        if size <= 0:
            continue
        successors = consumers[cost.uid]
        release = (
            max(timeline.finish_times[uid] for uid in successors) if successors else makespan
        )
        device_id = (
            annotation.dst_device if annotation.kind == "transfer" else annotation.device_id
        )
        if device_id < 0:
            continue
        live.setdefault(int(device_id), []).append((float(finish), float(release), size))

    per_chip_activation: Dict[int, float] = {}
    for device_id, intervals in live.items():
        chip_id = mapping.device_record(device_id).chip_id
        per_chip_activation[chip_id] = per_chip_activation.get(chip_id, 0.0) + _peak_bytes(
            intervals
        )

    # Recurrent state and KV are RESIDENT for the whole run, per layer, on the
    # chiplet that runs the layer's act x act op. Under the FILLED PIPELINE the
    # resident population is D STREAMS, not a batch (D29): every stage holds
    # ALL D streams' state and KV for its layers, and each of those streams is
    # at its OWN context, so the KV total is a sum over streams and never a
    # count times one context.
    filled = str(getattr(serving, "regime", "")) == REGIME_FILLED
    residents = int(serving.streams) if filled else int(serving.batch)
    # D38: KV AND RECURRENT STATE ARE FP8 on the mapped path. One byte per
    # element, from one named constant, and it does NOT follow the run's tensor
    # format: a card that declares bf16 activations still builds an FP8 state
    # store. The retired lockstep regime keeps the declared precisions, because
    # it is the pass-1/2 comparison and D38 did not move it.
    if filled:
        state_bytes_per_element = float(STATE_ELEMENT_BYTES)
        kv_bytes_per_element = float(STATE_ELEMENT_BYTES)
        disclosures.append(
            Relaxation(
                constraint="state_element_precision",
                value=(
                    f"{STATE_ELEMENT_DTYPE}: {STATE_ELEMENT_BYTES:g} B/element for KV "
                    "and recurrent state"
                ),
                reason=(
                    STATE_ELEMENT_BASIS
                    + " The declared precisions this REPLACES on the mapped path are "
                    f"sw_param.precision.activations = {act_bytes:g} B and "
                    f"sw_param.precision.kv_cache = {kv_precision:g} B; they keep "
                    "sizing activations and the retired lockstep regime."
                ),
            )
        )
    else:
        state_bytes_per_element = float(act_bytes)
        kv_bytes_per_element = float(kv_precision)
    # SIZING contexts, not measurement ones: the store is built for the whole
    # served request, never for wherever the lowered window stopped (M10).
    resident_contexts = _served_contexts(serving) if filled else []
    per_chip_state: Dict[int, float] = {}
    state_items: List[Tuple[int, int, float]] = []   # (chip, layer, bytes)
    seen_state: set = set()
    #: (chip, layer) -> {stream: the largest context that stream reached}. Under
    #: lockstep there is ONE pseudo-stream carrying the batch.
    kv_contexts: Dict[Tuple[int, int], Dict[int, int]] = {}
    for cost in pricing.costs:
        annotation = annotations[cost.uid]
        if annotation.kind != "fabric":
            continue
        chip_id = mapping.device_record(annotation.device_id).chip_id
        if annotation.block in ("ssm_scan", "delta_rule"):
            key = (chip_id, annotation.layer)
            if key in seen_state:
                continue
            seen_state.add(key)
            bytes_here = _state_residency(
                mapping, residents, state_bytes_per_element, annotation.block
            )
            per_chip_state[chip_id] = per_chip_state.get(chip_id, 0.0) + bytes_here
            state_items.append((chip_id, int(annotation.layer or 0), bytes_here))
        elif annotation.block == "attention_qk":
            key = (chip_id, int(annotation.layer if annotation.layer is not None else -1))
            context = (
                int(annotation.context)
                if annotation.context is not None
                else (
                    int(serving.prefill_len)
                    if annotation.phase == "prefill"
                    else int(serving.prefill_len + int(annotation.step) + 1)
                )
            )
            stream = int(annotation.stream) if annotation.stream >= 0 else 0
            per_stream = kv_contexts.setdefault(key, {})
            per_stream[stream] = max(per_stream.get(stream, 0), context)
            if filled:
                # D29: the stage holds ALL D streams' KV for this layer, whether
                # or not the WINDOW happened to lower every stream's pass on it
                # (a late stage sees fewer traversals inside a bounded window).
                # The contexts are the SERVED ones - each stream at the end of
                # the request it was given - so neither a truncated window nor
                # a generous one can move the silicon this store buys.
                for index, value in enumerate(resident_contexts):
                    per_stream[index] = max(per_stream.get(index, 0), int(value))

    kv_story = str(
        getattr(getattr(mapping.hw, "inference_config", None), "kvcache_type", "") or ""
    ).strip().lower()
    kv_enabled = kv_story in ("cim_sram", "cim_dram") and not mapping.device.params.is_vit_shaped
    per_chip_kv: Dict[int, float] = {}
    kv_items: List[Tuple[int, int, float]] = []      # (chip, layer, bytes)
    kv_context_census: List[int] = []
    if kv_enabled:
        for (chip_id, layer), per_stream in kv_contexts.items():
            if filled:
                # ONE accounting, summed over the streams that are actually
                # resident: stream i holds its own KV at its own context.
                contexts = [per_stream[key] for key in sorted(per_stream)]
                bytes_here = math.fsum(
                    mapping.device.kv_bytes_per_stream_layer(
                        ctx, kv_bytes_per_element, tp
                    )
                    for ctx in contexts
                )
            else:
                contexts = [max(per_stream.values())]
                bytes_here = float(residents) * mapping.device.kv_bytes_per_stream_layer(
                    contexts[0], kv_bytes_per_element, tp
                )
            kv_context_census.extend(contexts)
            per_chip_kv[chip_id] = per_chip_kv.get(chip_id, 0.0) + bytes_here
            kv_items.append((chip_id, int(layer), bytes_here))

    sram = mapping.hw.tech_config.DRAM
    sram_capacity = float(getattr(sram, "size", 0.0) or 0.0)
    kv_capacity = 0.0
    if kv_enabled:
        kv_capacity = mapping.device.kv_story_capacity(kv_story, sram_capacity)
    if sram_capacity > 0:
        disclosures.append(
            Relaxation(
                constraint="activation_tier_is_per_chip",
                value=f"{sram_capacity:.6g} B applied to EVERY chip",
                reason=(
                    "The hardware config declares ONE activation/SRAM tier size "
                    "(tech_config.DRAM.size); no per-chip capacity field exists anywhere "
                    "in the schema. The feasibility check therefore applies the declared "
                    "size to each chip independently. It is a relaxation and it is "
                    "disclosed rather than presented as a per-chip capacity the config "
                    "does not contain."
                ),
            )
        )
    if kv_enabled and kv_story == "cim_sram":
        disclosures.append(
            Relaxation(
                constraint="kv_shares_the_activation_tier",
                value="cim_sram: one tier, two tenants",
                reason=(
                    "inference.kvcache_type = cim_sram puts the KV cache in the same "
                    "activation SRAM tier the replay's activations draw on, so the verdict "
                    "is taken on the SUM. Under cim_dram the KV cache gets its own "
                    "declared tier and its own verdict."
                ),
            )
        )

    verdicts: List[MemoryVerdict] = []
    chips = sorted(
        set(per_chip_activation) | set(per_chip_state) | set(per_chip_kv)
    )
    for chip_id in chips:
        chip = mapping.chip(chip_id)
        owner = f"{chip.label} ({chip.role}, {chip.pool})"
        activation = per_chip_activation.get(chip_id, 0.0)
        state = per_chip_state.get(chip_id, 0.0)
        kv = per_chip_kv.get(chip_id, 0.0)
        shares = kv_enabled and kv_story == "cim_sram"
        contributors = OrderedDict(
            (("activations", activation), ("recurrent_state", state))
        )
        if shares:
            contributors["kv_cache"] = kv
        high_water = math.fsum(contributors.values())
        status = (
            "undeclared"
            if sram_capacity <= 0
            else ("fits" if high_water <= sram_capacity else "VIOLATED")
        )
        verdicts.append(
            MemoryVerdict(
                scope=f"chip {chip_id}",
                tier="activation_sram" + ("+kv" if shares else ""),
                owner=owner,
                high_water_bytes=high_water,
                capacity_bytes=sram_capacity,
                status=status,
                basis=_ACTIVATION_BASIS,
                contributors=contributors,
                disclosure=(
                    ""
                    if status != "VIOLATED"
                    else (
                        f"chip {chip_id} ({owner}) holds {high_water:.6g} B of "
                        f"activations, recurrent state and KV against a declared "
                        f"{sram_capacity:.6g} B tier. The overflow is REPORTED with its "
                        "owner and quantity; nothing is absorbed and no result is "
                        "relabeled feasible (D21)."
                    )
                ),
            )
        )
        if kv_enabled and not shares and kv > 0:
            kv_status = (
                "undeclared"
                if kv_capacity <= 0
                else ("fits" if kv <= kv_capacity else "VIOLATED")
            )
            verdicts.append(
                MemoryVerdict(
                    scope=f"chip {chip_id}",
                    tier="kv",
                    owner=owner,
                    high_water_bytes=kv,
                    capacity_bytes=kv_capacity,
                    status=kv_status,
                    basis=(
                        "KV bytes of the layers whose attention runs on this chip, at the "
                        "largest context the LOWERED decode window reached, from "
                        f"CimDeviceModel.kv_bytes_per_stream_layer x batch {serving.batch}. "
                        "The context is a timeline reading, so a windowed run reports a "
                        "windowed KV figure (ADJ-6)."
                    ),
                    contributors=OrderedDict((("kv_cache", kv),)),
                    disclosure=(
                        ""
                        if kv_status != "VIOLATED"
                        else (
                            f"chip {chip_id} holds {kv:.6g} KV bytes against a declared "
                            f"{kv_capacity:.6g} B {kv_story} tier."
                        )
                    ),
                )
            )
    # D29 makes the per-stage state a FEASIBILITY question, and the state it
    # asks about is RESIDENT and read every beat — so the tier that decides it
    # is the on-chip SRAM the config declares, not the activation/DRAM stub.
    # Both are reported, each under its own name (D21): `verdict` belongs to
    # the on-chip tier because that is the check that can fail, and
    # `activation_tier_verdict` keeps the DRAM reading beside it.
    onchip = mapping.hw.tech_config.SRAML2
    onchip_capacity = float(getattr(onchip, "size", 0.0) or 0.0)
    residency_disclosed = _build_state_residency(
        mapping,
        serving,
        state_items,
        kv_items,
        kv_context_census,
        residents,
        sram_capacity,
        kv_enabled,
        kv_story,
        onchip_capacity,
    )
    residency = residency_disclosed
    if residency.get("measured"):
        # The SAME bytes, at the scope D29 asks the question at. Named verdicts,
        # one row per stage, with the regrouping stated in the basis so a reader
        # never reads the stage total as a second, additional demand.
        for row in residency["per_stage"]:
            verdicts.append(
                MemoryVerdict(
                    scope=f"stage {int(row['stage'])}",
                    tier="stage_state_kv",
                    owner=str(row["label"]),
                    high_water_bytes=float(row["state_bytes"]),
                    # THE DECLARED ACTIVATION TIER, deliberately. These
                    # MemoryVerdict rows are P4's capacity machinery and are
                    # what the DSE's `memory` stage gates a candidate on, and
                    # that gate belongs to the size the config DECLARES for the
                    # activation/KV store. The ON-CHIP store is a different
                    # question with a different answer: ADJ-11 SIZES it to this
                    # very bill rather than checking it against anything, so it
                    # cannot appear here as a capacity a stage passes or fails.
                    capacity_bytes=float(row["activation_tier_capacity_bytes"]),
                    status=str(row["activation_tier_verdict"]),
                    basis=_RESIDENCY_BASIS
                    + " This row is taken against tech_param.DRAM.size, the declared "
                    "activation tier. The ON-CHIP store is not a check at all (ADJ-11): "
                    f"it is BUILT to this stage's bill - {int(row.get('store_macros', 0))} "
                    f"measured SRAM macro(s) holding {float(row['capacity_bytes']):.6g} B - "
                    "and its silicon is charged in evaluation.digital_silicon.",
                    contributors=OrderedDict(
                        (
                            ("recurrent_state", float(row["recurrent_state_bytes"])),
                            ("kv_cache", float(row["kv_bytes"])),
                        )
                    ),
                    disclosure=(
                        ""
                        if row["activation_tier_verdict"] != "VIOLATED"
                        else (
                            f"stage {int(row['stage'])} must hold "
                            f"{float(row['state_bytes']):.6g} B of state and KV for all "
                            f"{int(residency['resident_streams'])} resident streams "
                            f"against a declared {float(row['capacity_bytes']):.6g} B "
                            "tier. D29 makes this a FEASIBILITY question about the stage "
                            "plan itself: fewer layers per stage lowers D and the "
                            "per-stage demand together."
                        )
                    ),
                )
            )
        disclosures.append(
            Relaxation(
                constraint="stage_state_store_is_built_per_stage",
                value=(
                    "one store BUILT per stage, sized to that stage's own bill; the "
                    f"declared on-chip tier {onchip_capacity:.6g} B "
                    "(tech_param.SRAM-L2.size) gates nothing and the activation stub "
                    f"{sram_capacity:.6g} B (tech_param.DRAM.size) rides beside it"
                ),
                reason=(
                    "ADJ-11: the resident state is not checked against a capacity, it "
                    "is what the capacity is BUILT FROM. Each stage gets its own store "
                    "of whole measured SRAM macros, sized to that stage's measured bill, "
                    "and the silicon is charged in evaluation.digital_silicon - so "
                    "state_residency.verdict is the word 'sized' and there is no stage a "
                    "machine can fail here. What CAN refuse a machine is its area budget, "
                    "which is a statement about the package rather than about memory. "
                    "The declared SRAM-L2 size is still reported, under "
                    "declared_onchip_tier_bytes, because a reader comparing this machine "
                    "to a declared part will want it - but nothing is taken against it. "
                    "One store per STAGE, not one pooled store: the stages sit on "
                    "different chips and a macro cannot straddle two."
                ),
            )
        )
        disclosures.append(
            Relaxation(
                constraint="representative_context",
                value=(
                    f"median context {residency['representative_context']:.6g} over "
                    f"[{residency['context_min']:.6g}, {residency['context_max']:.6g}]"
                ),
                reason=(
                    "D29 leaves the resident streams at DIFFERENT decode depths. Every "
                    "priced attention and scan op uses ITS OWN stream's context — no op "
                    "is priced at the representative figure. The representative context "
                    "is the MEDIAN over the lowered window and exists so a reader can "
                    "name the depth the reported state figures belong to. The convention "
                    "is: stream i enters at prefill_len + i + 1 and advances one token "
                    "every D beats."
                ),
            )
        )
    return tuple(verdicts), tuple(disclosures), residency


#: The state/KV residency block of a filled-pipeline run (D29, P7.7 item 3).
_RESIDENCY_BASIS = (
    "D29: EVERY stage holds ALL D resident streams' state and KV for the layers it "
    "owns. The per-stage figure is the SAME per-(chip, layer) quantities the memory "
    "verdicts are taken on, regrouped by stage — not a second computation (D21): "
    "recurrent state is D x P2's per-stream state law, and KV is the SUM over the "
    "resident streams of P2's kv_bytes_per_stream_layer at each stream's OWN context, "
    "because the streams sit at different decode depths."
)


def _sized_store(mapping: FwsMapping, required_bytes: float) -> Dict[str, object]:
    """The on-chip store built to hold ``required_bytes``, in measured macros.

    A stage's resident state is not checked against a declared tier: the store
    is BUILT to it (ADJ-11).

    WHEN NOTHING CAN BE BUILT, THE ROW SAYS SO. A card that names no synthesis
    library, or one whose library has no SRAM macro, has no measured block to
    build from — that is an ABSENT LAW, and it is reported as one under
    ``provenance`` and ``uncovered``. It used to be swallowed by a bare
    ``except`` and returned as 0 macros / 0.0 mm2 with the verdict still
    "sized" and the prose still claiming the area was charged, so a store that
    could not be composed was byte-identical to a store that measured zero.
    That is the one thing the honesty rules forbid.
    """
    device = getattr(mapping, "device", None)
    if device is None or not device.has_synthesis_library():
        return {
            "bytes": float(required_bytes),
            "macros": 0,
            "area_mm2": 0.0,
            "provenance": "absent",
            "uncovered": (
                "the on-chip state store is NOT composed: this card names no synthesis "
                "library, so there is no measured SRAM macro to build it from. The bytes "
                "are the requirement; the silicon that would hold them is unpriced."
            ),
        }
    try:
        composition = device.state_sram_composition(float(required_bytes))
        macros = int(sum(composition.blocks.values()))
        per_macro = sram_macro_bytes(device.synthesis_library())
    except SynthesisLibraryError as exc:
        return {
            "bytes": float(required_bytes),
            "macros": 0,
            "area_mm2": 0.0,
            "provenance": "absent",
            "uncovered": (
                "the on-chip state store is NOT composed: " + str(exc).split(".")[0]
                + ". The bytes are the requirement; the silicon is unpriced."
            ),
        }
    return {
        "bytes": float(macros * per_macro),
        "macros": macros,
        "area_mm2": float(composition.area_mm2),
        "provenance": "composed-measured",
        "uncovered": None,
    }

def _stage_chiplet_groups(mapping: FwsMapping) -> Tuple[Tuple[int, ...], ...]:
    """Which pipeline stages share each shared digital chiplet.

    Taken from the mapping's own :meth:`chiplet_index_of_chip` — the SAME
    function the lowering places ops with — so an engine is always sized for
    the grouping the DAG actually builds. It used to deal the stages itself,
    contiguously, while the lowering dealt the chips round-robin; the two
    disagreed on every machine with fewer chiplets than stages, and the engine
    was widened for a chiplet nobody built.

    A chiplet that serves no stage still gets a row, because the machine builds
    it and a sizing that cannot see it cannot report it.
    """
    stages = list(getattr(mapping, "stages", ()) or ())
    if not stages:
        return ()
    engines = mapping.shared_digital_devices()
    if not engines:
        return tuple((int(stage.index),) for stage in stages)
    groups: List[List[int]] = [[] for _ in engines]
    for stage in stages:
        chips = list(getattr(stage, "chips", ()) or ())
        if not chips:
            continue
        # A stage's digital work rides the chiplet that serves its FIRST chip;
        # a stage spanning several chips is placed by the same rule op by op,
        # and the group list below is what the sizing sums over.
        groups[mapping.chiplet_index_of_chip(int(chips[0]))].append(int(stage.index))
    return tuple(tuple(group) for group in groups)

def _build_state_residency(
    mapping: FwsMapping,
    serving: ServingPoint,
    state_items: Sequence[Tuple[int, int, float]],
    kv_items: Sequence[Tuple[int, int, float]],
    contexts: Sequence[int],
    residents: int,
    capacity_bytes: float,
    kv_enabled: bool,
    kv_story: str,
    onchip_capacity_bytes: float = 0.0,
) -> "OrderedDict[str, object]":
    """Per-stage resident state/KV and its feasibility verdict (the D29 headline).

    The number rides the report's headline block because it is the constraint
    that decides whether a stage plan is buildable at all: D streams of KV and
    recurrent state, on every stage, for that stage's layers.
    """
    filled = str(getattr(serving, "regime", "")) == REGIME_FILLED
    if not filled or not mapping.stages:
        return OrderedDict(
            (
                ("measured", False),
                (
                    "reason",
                    "the retired lockstep regime has no resident streams: state and KV "
                    "belong to ONE batch stepping through the whole model, which the "
                    "per-chip memory verdicts already report (D29 supersedes it).",
                ),
            )
        )
    per_layer_state: Dict[int, float] = {}
    for _chip, layer, value in state_items:
        per_layer_state[int(layer)] = per_layer_state.get(int(layer), 0.0) + float(value)
    per_layer_kv: Dict[int, float] = {}
    for _chip, layer, value in kv_items:
        per_layer_kv[int(layer)] = per_layer_kv.get(int(layer), 0.0) + float(value)

    rows = []
    for stage in mapping.stages:
        state = math.fsum(per_layer_state.get(int(layer), 0.0) for layer in stage.layers)
        kv = math.fsum(per_layer_kv.get(int(layer), 0.0) for layer in stage.layers)
        total = state + kv
        activation_status = (
            "undeclared"
            if capacity_bytes <= 0
            else ("fits" if total <= capacity_bytes else "VIOLATED")
        )
        # THE ON-CHIP STORE IS SIZED, NOT CHECKED. A stage holds what it
        # holds; the SRAM that holds it is built to that size out of integer
        # measured macros and charged as silicon. So this row reports the
        # store the stage needs, and its verdict is "sized" — never a
        # violation. What can refuse a machine is its AREA budget.
        store = _sized_store(mapping, total)
        # "sized" is a claim that a store was BUILT. When none could be, the
        # row says "uncovered" instead, so a reader is never shown a composed
        # store and an absent law under the same word.
        status = "sized" if store.get("provenance") == "composed-measured" else "uncovered"
        rows.append(
            OrderedDict(
                (
                    ("stage", int(stage.index)),
                    ("label", stage.label),
                    ("layers", len(stage.layers)),
                    ("chips", list(stage.chips)),
                    ("recurrent_state_bytes", float(state)),
                    ("kv_bytes", float(kv)),
                    ("state_bytes", float(total)),
                    ("per_stream_state_bytes", float(total) / max(1, int(residents))),
                    ("capacity_bytes", float(store["bytes"])),
                    ("store_macros", int(store["macros"])),
                    ("store_area_mm2", float(store["area_mm2"])),
                    ("store_provenance", str(store.get("provenance", "absent"))),
                    ("store_uncovered", store.get("uncovered")),
                    ("declared_onchip_tier_bytes", float(onchip_capacity_bytes)),
                    ("verdict", status),
                    ("activation_tier_capacity_bytes", float(capacity_bytes)),
                    ("activation_tier_verdict", activation_status),
                )
            )
        )
    total_bytes = math.fsum(row["state_bytes"] for row in rows)
    ordered = sorted(int(value) for value in contexts)
    representative = (
        float(_median([float(value) for value in ordered])) if ordered else 0.0
    )
    store_macros = int(sum(int(row["store_macros"]) for row in rows))
    store_bytes = float(sum(float(row["capacity_bytes"]) for row in rows))
    store_area = float(math.fsum(float(row["store_area_mm2"]) for row in rows))
    violated = [row for row in rows if row["verdict"] == "VIOLATED"]
    activation_violated = [
        row for row in rows if row["activation_tier_verdict"] == "VIOLATED"
    ]
    return OrderedDict(
        (
            ("measured", True),
            ("resident_streams", int(residents)),
            ("stages", len(rows)),
            ("per_stage", rows),
            ("total_state_bytes", float(total_bytes)),
            ("per_stream_model_state_bytes", float(total_bytes) / max(1, int(residents))),
            ("max_stage_state_bytes", max((row["state_bytes"] for row in rows), default=0.0)),
            ("capacity_bytes", float(store_bytes)),
            ("store_macros", store_macros),
            ("store_bytes", float(store_bytes)),
            ("store_area_mm2", store_area),
            ("declared_onchip_tier_bytes", float(onchip_capacity_bytes)),
            (
                "tier",
                "an on-chip store SIZED to this mapping, in integer copies of the "
                "measured SRAM macro; its area is charged as silicon",
            ),
            ("stages_violating", [int(row["stage"]) for row in violated]),
            (
                "verdict",
                "sized"
                if all(row["store_provenance"] == "composed-measured" for row in rows)
                else "uncovered",
            ),
            (
                "store_uncovered",
                next((row["store_uncovered"] for row in rows if row["store_uncovered"]), None),
            ),
            ("activation_tier_capacity_bytes", float(capacity_bytes)),
            (
                "activation_tier",
                "tech_param.DRAM.size (the activation stub, orders of magnitude larger)",
            ),
            (
                "activation_tier_stages_violating",
                [int(row["stage"]) for row in activation_violated],
            ),
            (
                "activation_tier_verdict",
                "undeclared"
                if capacity_bytes <= 0
                else ("VIOLATED" if activation_violated else "fits"),
            ),
            ("representative_context", representative),
            ("context_min", float(ordered[0]) if ordered else 0.0),
            ("context_max", float(ordered[-1]) if ordered else 0.0),
            ("kv_story", kv_story if kv_enabled else "no KV in this run"),
            ("basis", _RESIDENCY_BASIS),
        )
    )


# ---------------------------------------------------------------------------
# The derived per-macro digital pool, sized FROM THE TIMELINE (P4.5, D12)
# ---------------------------------------------------------------------------


def _conv_pool_dims(mapping: FwsMapping) -> "OrderedDict[str, int]":
    """The short depthwise conv this model runs — the DEVICE's own answer.

    One definition (D21): :meth:`CimDeviceModel.conv_pool_dims` reads whichever
    of the three schemas declares it (`short_conv`, an `ssm` block's ``d_conv``,
    a gated linear-attention block's ``conv_kernel``), and the pool composition
    that prices the silicon reads the same function.
    """
    return mapping.device.conv_pool_dims()

def _build_pools(
    mapping: FwsMapping,
    program: Program,
    pricing: PricingResult,
    timeline: CoarseEvalResult,
) -> Tuple[PoolSizingReport, ...]:
    """Peak concurrent digital demand per macro -> the reported pool width."""
    ready = _ready_times(program, timeline)
    demand: Dict[int, List[Tuple[float, float]]] = {}
    for cost in pricing.costs:
        if cost.device_class != "macro_pool" or cost.macro_id < 0:
            continue
        finish = timeline.finish_times[cost.uid]
        if finish < 0:
            continue
        demand.setdefault(int(cost.macro_id), []).append((float(ready[cost.uid]), float(finish)))
    if not demand:
        return ()

    device = mapping.device
    # THE SHORT DEPTHWISE CONV, FROM EITHER PLACE IT IS DECLARED (ADJ-3). A
    # `short_conv:` block declares one; an `ssm:` block declares the SAME
    # operator as `d_conv` over `d_inner` channels, which is where Mamba-1 and
    # Mamba-2 keep it. Reading only the first meant a pure-SSM machine sized
    # ZERO conv lanes while the mapper had already dropped the conv weights
    # from the tile set "because they run on the per-macro pool" - so the work
    # was charged on neither side.
    conv_kwargs = _conv_pool_dims(mapping)
    per_unit = device.digital_pool_sizing(**conv_kwargs)

    out: List[PoolSizingReport] = []
    for macro_id in sorted(demand):
        intervals = demand[macro_id]
        peak = max(1, _peak_overlap(intervals, closed=True))
        macro = mapping.macro(macro_id)
        owners = macro.owners
        owner = (
            ", ".join(item.label for item in owners)
            if owners
            else f"unowned macro slot {macro_id}"
        )
        area = float(per_unit.area_mm2) * peak
        footprint = float(per_unit.macro_footprint_mm2)
        share = area / footprint if footprint > 0 else 0.0
        scaled = replace(
            per_unit,
            lanes=int(per_unit.lanes) * peak,
            adders=int(per_unit.adders) * peak,
            conv_lanes=int(per_unit.conv_lanes) * peak,
            conv_area_mm2=float(per_unit.conv_area_mm2) * peak,
            area_mm2=area,
            area_share=share,
            disclose=share >= 0.2,
            note=(
                f"per-macro digital pool, sized FROM THE TIMELINE (P4.5): macro "
                f"{macro_id} reached a peak concurrent digital demand of {peak} pool "
                f"op(s), so the non-blocking width is {peak} x P2's per-unit derivation. "
                f"Derived area {area:.6g} mm2 is {share * 100:.1f}% of the "
                f"{footprint:.6g} mm2 macro footprint it serves. P2 owns every per-unit "
                "number; P4 supplies only the measured concurrency, and nothing is "
                "clamped — an absurd width prints as an absurd width (D12)."
            ),
        )
        out.append(
            PoolSizingReport(
                macro_id=macro_id,
                owner=owner,
                peak_concurrent_demand=peak,
                pool_ops=len(intervals),
                per_unit=per_unit,
                scaled=scaled,
                report_line=device.report_digital_pool(scaled),
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# The evaluator (P4.1 - P4.5 in one call)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# D31 pass A: the beat the ANALOG stages set, and the engine derived from it
# ---------------------------------------------------------------------------


def _analog_beat_of(
    mapping: FwsMapping,
    serving: ServingPoint,
    pricing: PricingResult,
    timeline: CoarseEvalResult,
    op_annotations: Sequence[FwsOpAnnotation] = (),
) -> float:
    """The PROBE beat and its convergence, on a timeline whose vector ops cost nothing.

    Filled pipeline (D29): the beat is the median steady inter-exit interval,
    read by :func:`_build_pipeline_metrics` from this very timeline. Retired
    lockstep regime: the same quantity is the decode step, so that is what the
    engine is sized against there — one law, two regime spellings.
    """
    if str(getattr(serving, "regime", "")) == REGIME_FILLED:
        _, _, _, pipeline = _build_pipeline_metrics(
            mapping, serving, pricing.costs, timeline, op_annotations
        )
        beat = float(pipeline.get("beat_s", 0.0) or 0.0)
        settled = [float(v) for v in pipeline.get("converged_tail_intervals_s", ())]
        steady = [float(v) for v in pipeline.get("steady_beat_intervals_s", ())]
        spread = (
            max(abs(value - beat) for value in steady) / beat
            if steady and beat > 0
            else 0.0
        )
        return beat, OrderedDict(
            (
                ("regime", REGIME_FILLED),
                ("converged", len(settled) >= 2),
                ("converged_tail_beats", len(settled)),
                ("ramp_beats_dropped", len(steady) - len(settled)),
                ("steady_beat_spread", spread),
            )
        )
    _, step_times, _ = _build_metrics(mapping, serving, pricing.costs, timeline)
    beat = float(_median(step_times)) if step_times else 0.0
    return beat, OrderedDict(
        (
            ("regime", "lockstep"),
            ("converged", None),
            ("converged_tail_beats", len(step_times)),
            ("ramp_beats_dropped", 0),
            ("steady_beat_spread", None),
        )
    )


#: Shared-digital blocks whose analog<->chiplet traffic this repo LOWERS NO
#: transfer op for, so a ``svc`` link cannot count their bytes (D21). The
#: recurrent path places its scan on the chiplet directly: the projections
#: write their result and the scan reads it with no hop in between. Named here
#: once, so the disclosure and the byte total cannot disagree about which
#: component is missing.
_UNPRICED_SERVICE_BLOCKS = ("ssm_scan", "delta_rule", "rg_lru")


def _union_busy_s(intervals: Sequence[Tuple[float, float]]) -> float:
    """Wall time covered by at least one of ``intervals`` — measured, not summed.

    Analog macros within one stage fire in PARALLEL, so a sum over their ops
    would report a stage's analog time as hundreds of microseconds when the
    stage really spends a few. The union is the wall-clock time the stage's
    analog side is occupied, which is the quantity ADJ-9 compares the digital
    engine against.
    """
    ordered = sorted(intervals)
    if not ordered:
        return 0.0
    total = 0.0
    start, end = ordered[0]
    for begin, finish in ordered[1:]:
        if begin > end:
            total += end - start
            start, end = begin, finish
        else:
            end = max(end, finish)
    return total + (end - start)


def _engine_demand(
    serving: ServingPoint,
    annotations: Sequence[FwsOpAnnotation],
    costs: Sequence[OpCost],
    timeline: CoarseEvalResult,
) -> Tuple[EngineDemand, ...]:
    """Per-stage vector demand AND analog m-pass time in ONE beat (ADJ-9).

    The scope is :func:`_measurement_slice`'s — the last lowered beat under the
    filled pipeline, the last lowered decode step under lockstep — because that
    is the slice in which every stage fires exactly once. Sizing on the whole
    lowered window would size the engine for D beats' work in one beat.

    A vector op is one whose priced cost carries an ``ops`` detail, which is
    exactly the set :meth:`CimDeviceModel.price_vector_work` produces; the
    attention ops on the same chiplet run on the systolic array and the softmax
    lanes, which are different silicon with their own laws.

    THE ANALOG TARGET (D31-v2). ``analog_time_s`` is the UNION of the busy
    intervals of the stage's ``analog_macro`` ops in that slice, read off the
    probe timeline: the wall time during which at least one of the stage's
    macros is running an m-pass. Three properties make it the right target:

    * it is a MEASUREMENT off the same timeline the beat comes from, not a
      formula (A1, P4 §3);
    * it is INVARIANT to the width being derived — every analog duration is,
      and the union ignores the gaps the engine's own time would open between
      them — so ADJ-9 keeps D31's two-pass argument intact and needs no
      iteration;
    * it EXCLUDES the gaps, so it is the strictly smaller of the two candidate
      readings (union vs. first-issue-to-last-finish span) and therefore the
      tighter target. Sizing against the span would let the digital engine hide
      inside time the analog side is not actually working, which is the margin
      D28 forbids.
    """
    select, _ = _measurement_slice(serving)
    per_stage: "OrderedDict[int, List[float]]" = OrderedDict()
    analog: "OrderedDict[int, List[Tuple[float, float]]]" = OrderedDict()
    for annotation, cost in zip(annotations, costs):
        if not select(annotation):
            continue
        stage = int(annotation.stage)
        if "ops" in cost.detail:
            per_stage.setdefault(stage, []).append(float(cost.detail["ops"]))
        if cost.device_class == "analog_macro":
            start = float(timeline.start_times[cost.uid])
            finish = float(timeline.finish_times[cost.uid])
            if start >= 0 and finish >= start:
                analog.setdefault(stage, []).append((start, finish))
    return tuple(
        EngineDemand(
            stage=stage,
            ops=tuple(ops),
            analog_time_s=_union_busy_s(analog.get(stage, ())),
        )
        for stage, ops in per_stage.items()
    )


def _fabric_demand(
    serving: ServingPoint,
    annotations: Sequence[FwsOpAnnotation],
    costs: Sequence[OpCost],
    timeline: CoarseEvalResult,
) -> Tuple[FabricDemand, ...]:
    """Per-stage ATTENTION demand AND analog m-pass time in ONE beat (ADJ-10).

    The twin of :func:`_engine_demand`, over the same measurement slice and
    against the same target. An attention call is one ``attention_qk`` cost:
    the lowering places three ops per call and charges the array's fill/drain
    to the QK op, so the QK op is the one record that carries the whole call.
    What is collected is the CALL DIMS — ``m``, ``k``, ``n``, the fold count,
    the softmax token count — because ADJ-10 re-prices the call at candidate
    widths and a duration measured at the declared width cannot be re-priced.

    ``analog_time_s`` is read exactly as :func:`_engine_demand` reads it: the
    UNION of the busy intervals of the stage's ``analog_macro`` ops in the
    slice. One measurement, one target, two engines sized against it (D21).
    """
    select, _ = _measurement_slice(serving)
    calls: "OrderedDict[int, List[AttentionCallDemand]]" = OrderedDict()
    analog: "OrderedDict[int, List[Tuple[float, float]]]" = OrderedDict()
    for annotation, cost in zip(annotations, costs):
        if not select(annotation):
            continue
        stage = int(annotation.stage)
        if cost.block == "attention_qk" and "call_m" in cost.detail:
            calls.setdefault(stage, []).append(
                AttentionCallDemand(
                    m=int(cost.detail["call_m"]),
                    k=int(cost.detail["call_k"]),
                    n=int(cost.detail["call_n"]),
                    folds=int(cost.detail["folds"]),
                    softmax_tokens=int(cost.detail["softmax_tokens"]),
                    heads_chip=int(cost.detail["heads_chip"]),
                    fill_drain_cycles=int(cost.detail["fill_drain_cycles"]),
                )
            )
        if cost.device_class == "analog_macro":
            start = float(timeline.start_times[cost.uid])
            finish = float(timeline.finish_times[cost.uid])
            if start >= 0 and finish >= start:
                analog.setdefault(stage, []).append((start, finish))
    return tuple(
        FabricDemand(
            stage=stage,
            calls=tuple(rows),
            analog_time_s=_union_busy_s(analog.get(stage, ())),
        )
        for stage, rows in calls.items()
    )


#: The convergence state of the LAST probe beat a derivation was taken from
#: (D31). Written by :func:`_derive_engine` and read by
#: :func:`engine_sizing_disclosures` in the same call chain.
_PROBE_BEAT_CONVERGENCE: "OrderedDict[str, object]" = OrderedDict()


def _beat_setting_stage_terms(
    serving: ServingPoint,
    annotations: Sequence[FwsOpAnnotation],
    costs: Sequence[OpCost],
    timeline: CoarseEvalResult,
    *,
    slice_name: str = "PROBE",
) -> "OrderedDict[str, object]":
    """WHICH stage sets the probe beat, and WHAT it spends the beat on (ADJ-9).

    ADJ-9 asks for the analog side to be the binding term "wherever physically
    reachable". Reachability has two halves and this block measures the second
    one. The derivation can guarantee that the DERIVED engine fits inside a
    stage's analog m-pass time; it cannot guarantee that the analog m-pass is
    then the stage's longest term, because a stage also runs silicon D31 does
    not size — the attention systolic fabric and the softmax pipeline, whose
    geometry is DECLARED on the card. This function names the stage that takes
    longest in the probe and itemizes its terms by op block, so a machine that
    is still bound by a declared digital unit says which one and by how much,
    instead of leaving the reader to infer it from a throughput number.

    Every term is a UNION of busy intervals, the same reading
    :func:`_engine_demand` takes for the analog target, so the numbers on this
    row and the numbers the engine was sized against are one accounting (D21).
    """
    select, _ = _measurement_slice(serving)
    spans: Dict[int, List[Tuple[float, float]]] = {}
    terms: Dict[int, Dict[Tuple[str, str], List[Tuple[float, float]]]] = {}
    for annotation, cost in zip(annotations, costs):
        if not select(annotation):
            continue
        start = float(timeline.start_times[cost.uid])
        finish = float(timeline.finish_times[cost.uid])
        if start < 0 or finish < start:
            continue
        stage = int(annotation.stage)
        spans.setdefault(stage, []).append((start, finish))
        terms.setdefault(stage, {}).setdefault(
            (cost.device_class, cost.block), []
        ).append((start, finish))
    if not spans:
        return OrderedDict()
    stage = max(
        spans, key=lambda key: max(f for _, f in spans[key]) - min(s for s, _ in spans[key])
    )
    span_s = max(f for _, f in spans[stage]) - min(s for s, _ in spans[stage])
    rows = sorted(
        (
            (device_class, block, _union_busy_s(intervals))
            for (device_class, block), intervals in terms[stage].items()
        ),
        key=lambda row: -row[2],
    )
    analog_s = _union_busy_s(
        [
            interval
            for (device_class, _), intervals in terms[stage].items()
            if device_class == "analog_macro"
            for interval in intervals
        ]
    )
    return OrderedDict(
        (
            ("stage", stage),
            ("span_s", span_s),
            ("analog_time_s", analog_s),
            ("analog_is_largest_term", bool(rows and rows[0][0] == "analog_macro")),
            (
                "terms",
                [
                    OrderedDict(
                        (
                            ("device_class", device_class),
                            ("block", block),
                            ("busy_s", busy),
                        )
                    )
                    for device_class, block, busy in rows[:5]
                ],
            ),
            (
                "basis",
                "the stage with the longest first-issue-to-last-finish span in the "
                f"{slice_name} slice"
                + (
                    " (every vector op counted and untimed, and the fabric at its "
                    "DECLARED seed width — this is the evidence the derivation RAN "
                    "AGAINST, not the machine it produced; for the machine that gets "
                    "built see evaluation.digital_silicon.binding_term)"
                    if slice_name == "PROBE"
                    else " — the machine that gets BUILT, with every derived width "
                    "installed and every op timed"
                )
                + ", with each term the "
                "UNION of the busy intervals of its op block on that stage. A union, "
                "not a sum: devices of one class fire in parallel and a sum would "
                "report a stage's analog time as the total of hundreds of macros.",
            ),
        )
    )


def _derive_engine(
    program: Program,
    mapping: FwsMapping,
    serving: ServingPoint,
    capacity_overrides: Optional[Mapping[int, int]],
) -> Tuple[Optional[DerivedFabricSizing], Optional[DerivedEngineSizing]]:
    """Size the scan/vector engine from this run's own beat (D31), or do nothing.

    THE CIRCULARITY AND HOW IT IS BROKEN. The engine's width sets the time its
    ops take, which contributes to the beat. Pass A prices the run with every
    vector op COUNTED AND UNTIMED
    (:meth:`CimDeviceModel.begin_engine_probe`), so everything it measures is a
    property of the non-vector silicon — the per-stage ANALOG m-pass time that
    ADJ-9 sizes against, and the beat that serves as the fallback budget. The
    width is then derived, installed, and the caller prices the run again for
    real. TWO PASSES ARE ENOUGH because neither target depends on the sizing:
    an analog op's duration does not move with a lane count, and the union that
    measures the stage's analog time ignores the gaps between those ops, so
    pass B cannot change the numbers pass A derived from. No iteration is
    needed or attempted.

    WHAT PASS B THEN MEASURES IS LARGER, and that is the machine, not an error.
    The criterion is *digital per-stage time <= that stage's own ANALOG m-pass
    time* (ADJ-9; see :meth:`CimDeviceModel.derive_engine_sizing`), so the
    analog side is the binding term of every stage that has analog work. Under
    D29 a stage holds ONE stream at a time, so its analog passes and its scan
    run in SERIES and the beat pass B reports is their sum. What ADJ-9 buys is
    that the analog half dominates the sum — not that the sum equals it.

    ADJ-10 PUTS A SECOND ENGINE ON THE SAME PROBE. The attention fabric's
    array count and the softmax pipeline's lane count are derived from the same
    pass, against the same per-stage analog m-pass times, and installed before
    the run is priced. ONE probe still suffices, and for the same reason: both
    targets are properties of the ANALOG side, and an analog op's duration does
    not move with an array count any more than it moves with a lane count. The
    attention CALL DIMS the fabric derivation needs are likewise invariant —
    they are the model's shapes, not the fabric's.

    WHAT THE PROBE'S OWN BEAT THEN IS. It is measured with the fabric at its
    DECLARED seed width, so it is the beat of the machine BEFORE either
    derivation. It is used for exactly two things and neither depends on the
    widths: the fallback budget for a stage ADJ-9's criterion cannot reach, and
    the printed reference. What the machine that gets BUILT is bound by is
    measured on the PRICED timeline instead and reported under its own name
    (``evaluation.digital_silicon.binding_term``), because that is the question
    ADJ-10 asks and the probe cannot answer it.

    Returns ``(fabric_sizing, engine_sizing)``; either is None when there is
    nothing to derive — a card that PINS the fabric
    (``cim.cards.<card>.fabric_num_arrays``) or the scan engine
    (``vector_lanes``), a run with no attention call, or a run with no vector
    op in it at all.
    """
    device = mapping.device
    # Always CLEAR first. The sizing belongs to one program's beat and one
    # program's demand, and a device object outlives both (a mapping is reused
    # across programs by callers and fixtures alike). Keeping a previous run's
    # width would quietly price this run on another run's silicon — and the
    # override branch below is exactly the case that used to skip the clear,
    # so an override card reusing a device could publish another run's
    # derivation in evaluation.digital_silicon.
    device.install_derived_engine(None)
    device.install_derived_fabric(None)
    pins_fabric = device.digital_card.has_fabric_override
    pins_engine = device.digital_card.has_vector_engine
    if pins_fabric and pins_engine:
        return None, None
    annotations = annotations_of(program)
    device.begin_engine_probe()
    try:
        probe = price_program(program, mapping, capacity_overrides=capacity_overrides)
        saw_vector_op = bool(device.end_engine_probe())
    finally:
        # Idempotent: a probe that raised must not leave the device probing.
        device.end_engine_probe()
    timeline = evaluate_detailed(
        program,
        None,
        {},
        resources=probe.resources,
        durations=probe.durations,
        require_pipeline=False,
    )
    beat, convergence = _analog_beat_of(mapping, serving, probe, timeline, annotations)
    # ADJ-10 FIRST: the fabric is derived and installed before the scan engine,
    # so a report that prints both prints them in the order they were taken.
    # The ORDER MOVES NO NUMBER — both are sized against the per-stage analog
    # m-pass times, which are properties of the analog side and invariant to
    # either width — and it is fixed here only so the derivation is
    # deterministic and re-runnable.
    # ADJ-14: which stages share a chiplet. A machine with fewer shared digital
    # chiplets than pipeline stages runs several stages' digital work on one
    # chiplet, one after another inside a beat, and both derivations below have
    # to size for that. The grouping is CONTIGUOUS so a chiplet serves stages
    # that are neighbours in the pipeline, which keeps its operands local.
    stage_groups = _stage_chiplet_groups(mapping)
    fabric_sizing = None
    if not pins_fabric and beat > 0:
        fabric_demand = _fabric_demand(serving, annotations, probe.costs, timeline)
        if fabric_demand:
            fabric_sizing = device.derive_fabric_sizing(
                beat, fabric_demand, stage_groups=stage_groups
            )
            device.install_derived_fabric(fabric_sizing)
    engine_sizing = None
    demand = _engine_demand(serving, annotations, probe.costs, timeline)
    if saw_vector_op and not pins_engine and beat > 0 and demand:
        engine_sizing = device.derive_engine_sizing(
            beat, demand, stage_groups=stage_groups
        )
        # ADJ-9's reachability evidence rides the SIZING, not a module global: a
        # report is written long after the derivation ran, and a global would hand
        # one run's evidence to another run's report (D21).
        engine_sizing = replace(
            engine_sizing,
            beat_setting_stage=_beat_setting_stage_terms(
                serving, annotations, probe.costs, timeline
            ),
        )
        device.install_derived_engine(engine_sizing)
    # The PROBE's own convergence, carried out so the disclosure can say
    # whether the width was derived from a settled number. A shorter probe beat
    # means a tighter cycle budget and therefore MORE lanes, so a probe still
    # ramping would systematically over-size the engine — a reader must be able
    # to see that this run's was not.
    _PROBE_BEAT_CONVERGENCE.clear()
    _PROBE_BEAT_CONVERGENCE.update(convergence)
    return fabric_sizing, engine_sizing


def analog_card_disclosures(mapping: FwsMapping) -> List[Relaxation]:
    """The DERIVED analog card, stated in the artifact (not only in the code).

    A card that names a ``calc_profile`` did not have its footprint, per-vector
    energy or vec_latency written by a human: all three were computed from the
    ADC-macro sheet at the card's (X, Y, Z) — rows, total CTT columns, ADCs.
    That is exactly the kind of number D21 requires to name its origin, so the
    provenance rides the report the way a derived engine's sizing does.
    """
    device = getattr(mapping, "device", None)
    card = getattr(device, "card", None) if device is not None else None
    derivation = getattr(card, "calc_derivation", None) if card is not None else None
    if not derivation:
        return []
    # The SENTENCE is the law layer's (cim_timing owns what a card discloses);
    # this function only decides that it belongs in the artifact.
    reason = " ".join(device.analog_card_disclosures())
    terms = derivation.get("area_terms_mm2", {})
    return [
        Relaxation(
            constraint="derived_analog_card",
            value=(
                f"cim.cards.{card.name} = {derivation['label']} (X x Y : Z) DERIVED "
                f"from calc_profile '{derivation['calc_profile']}': "
                f"{int(derivation['rows'])} rows x {int(derivation['cols'])} total CTT "
                f"columns behind {int(derivation['adcs'])} ADC(s), so mux = Y/Z = "
                f"{int(derivation['mux'])}; "
                f"area_mm2_per_array = {float(derivation['area_mm2_per_array']):.6g} "
                f"(ADC term {float(derivation['adc_area_mm2']):.6g}, cell array "
                f"{float(terms.get('ctt_array', 0.0)):.6g}), energy_per_vec_pj = "
                f"{float(derivation['energy_per_vec_pj']):.6g}, vec_cycles = "
                f"{int(derivation['vec_cycles'])} at "
                f"{float(derivation['analog_clock_mhz']):g} MHz "
                f"({float(derivation['vec_latency_s']):.6g} s per vector)"
            ),
            reason=reason,
        )
    ]


def nonstandard_macro_disclosures(mapping: FwsMapping) -> List[Relaxation]:
    """The D35 platform-macro exemption, stated in the artifact (D35).

    D35 fixes the analog macro as a PLATFORM CONSTANT. A config may still build
    another geometry, but only through ``cim.analog.nonstandard_macro``, and
    only by saying why — and the reason it gave belongs in the report, not just
    in the YAML. Without it a reader could not tell an off-platform point from a
    platform one, which is exactly the comparison D35 exists to protect.
    """
    device = getattr(mapping, "device", None)
    if device is None:
        return []
    notes = device.nonstandard_macro_disclosures()
    if not notes:
        return []
    card = device.card
    return [
        Relaxation(
            constraint="nonstandard_macro",
            value=(
                f"cim.cards.{card.name} is OFF-PLATFORM: {device.analog.label} "
                f"(X x Y : Z, mux {int(device.analog.mux)}) against the "
                f"{_analog_calc.PLATFORM_MACRO_DECISION} box X = "
                f"{_analog_calc.PLATFORM_MACRO_ROWS}, Z in "
                f"{list(_analog_calc.MACRO_Z_VALUES)}, Y in "
                f"{list(_analog_calc.MACRO_Y_VALUES)}"
            ),
            reason=" ".join(notes),
        )
    ]


def fabric_sizing_disclosures(
    sizing: Optional[DerivedFabricSizing],
) -> List[Relaxation]:
    """The ADJ-10 fabric derivation, stated in the artifact (not only stdout)."""
    if sizing is None:
        return []
    rows = ", ".join(
        f"stage {row.stage}: qk {row.qk_cycles} + sm {row.softmax_cycles} + pv "
        f"{row.pv_cycles} (+{row.fill_drain_cycles} fill/drain) = {row.used_cycles}/"
        f"{row.budget_cycles} cycles, attention {row.time_s:.4g} s vs analog "
        f"{row.analog_time_s:.4g} s"
        for row in sizing.per_stage[:6]
    )
    binding = sizing.binding_row
    saturated = sizing.saturated_stages
    out = [
        Relaxation(
            constraint="derived_fabric_sizing",
            value=(
                f"cim.fabric.num_arrays = {int(sizing.num_arrays)} "
                f"({PROVENANCE_DERIVED_COUNT}, declared seed "
                f"{int(sizing.declared_num_arrays)}), softmax_lanes = "
                f"{int(sizing.softmax_lanes)} (declared seed "
                f"{int(sizing.declared_softmax_lanes)}), on the DECLARED "
                f"{int(sizing.rows)}x{int(sizing.cols)} array x "
                f"{int(sizing.replicas)} replica(s); binding stage "
                f"{int(sizing.binding_stage)}, attention time "
                f"{sizing.target_ratio * 100:.3f}% of that stage's own "
                f"{(binding.target_s if binding is not None else 0.0):.6g} s ANALOG "
                f"m-pass; {len(sizing.analog_bound_stages)} of {len(sizing.per_stage)} "
                "stage(s) analog-bound"
            ),
            reason=(
                "ADJ-10: EVERY digital engine whose width is a composition of MEASURED "
                "synthesis blocks derives to the analog floor, not just the scan engine. "
                "The attention fabric's ARRAY COUNT is derived here in integer copies of "
                "the measured GEMMINI 32x32 block and the softmax pipeline's LANE COUNT "
                "in copies of its measured per-lane census, as the smallest integers for "
                "which every stage's attention time — qk + fill/drain + softmax + pv, "
                "the three ops the lowering serializes — fits that stage's OWN measured "
                "analog m-pass time, with no margin (D28). The target is the same "
                "measurement ADJ-9 sizes the scan engine against, taken off the same "
                "probe pass. ONLY THE COUNT IS DERIVED: rows x cols per array stay as "
                "measured and as declared, because inventing array geometry is refused "
                "(ADJ-4). THE FOLD LAW: the arrays split into a QK group and a PV group "
                "and a call's folds (heads_per_replica x streams) are dealt across the "
                "group, so an array in a group of a carries ceil(folds / a) of them; at "
                "num_arrays = 2 the split is (1, 1) and the law is the pass-1 law bit "
                f"for bit. Per stage: {rows}."
                + (
                    " FOLD CONCURRENCY SATURATES on stage(s) "
                    + ", ".join(str(int(stage)) for stage in saturated)
                    + ": past 2 x folds arrays another copy of the measured block "
                    "carries no fold and buys no time, so those stages take the "
                    "SATURATION width and the residue — the DECLARED rows x cols "
                    "geometry's own floor, ceil(n / cols) column passes times the "
                    "k + rows + cols - 2 pipeline term — is NAMED rather than clamped "
                    "away. Splitting one fold's N across arrays would shrink it further "
                    "and is NOT modelled: it is a claim about the fabric's dataflow "
                    "that no recorded reference in this repo covers."
                    if saturated
                    else " Every stage's criterion was reachable."
                )
            ),
        )
    ]
    geometry = sizing.geometry
    if geometry is not None:
        # SA-GEOM: THREE NAMED DISCLOSURES, one per thing a reader has to know
        # before they can use a derived-geometry number — the rule, the
        # dataflow the law is exact under, and the assumption it rests on. They
        # are separate constraints rather than one paragraph because they fail
        # separately: a ws-dataflow machine invalidates the second and leaves
        # the first intact, and a stalling feed invalidates the third alone.
        out.append(
            Relaxation(
                constraint="sa_geometry_rule",
                value=(
                    f"array {int(geometry.rows)} x {int(geometry.cols_qk)} (QK group) "
                    f"and {int(geometry.rows)} x {int(geometry.cols_pv)} (PV group), "
                    f"DERIVED; declared {int(geometry.declared_rows)} x "
                    f"{int(geometry.declared_cols)}. R = next_pow2(decode m = "
                    f"{int(geometry.call_m)})"
                ),
                reason=(
                    SA_GEOMETRY_RULE_DISCLOSURE
                    + f" THIS RUN: {geometry.basis}."
                ),
            )
        )
        if int(getattr(sizing, "context_split", 1)) > 1 or sizing.floors:
            split_rows = ", ".join(
                f"stage {row.stage}: qk x{row.qk_split} / pv x{row.pv_split}, "
                f"reduce {row.combine_cycles} cycles on {row.combine_lanes} FP_ADD "
                f"lane(s), gather {row.gather_bytes:.4g} bytes at 0 s, "
                f"{row.operand_bytes_per_cycle:.4g} K/V bytes per cycle"
                for row in sizing.per_stage[:6]
            )
            out.append(
                Relaxation(
                    constraint="sa_context_split",
                    value=(
                        f"context dealt {int(sizing.context_split)}-ways at its deepest; "
                        f"{int(sizing.combine_lanes)} FP_ADD reduce lane(s) provisioned"
                    ),
                    reason=(
                        "CTX-SPLIT: ADJ-10's widening no longer stops at fold saturation. "
                        "A fold's QK call is dealt across S arrays over the CONTEXT — each "
                        "running THE SAME sa_cycles law on an n/S chunk — and PV is dealt "
                        "the same way over its own contraction, returning S partial output "
                        "blocks. THE PER-CHUNK PRICE NEEDS NO NEW VALIDATION: sa_cycles is "
                        "bit-exact against ScaleSim's output-stationary schedule at any "
                        "geometry and any n, so a chunk is a shape the validated law "
                        "already covers. THE COMBINE IS PRICED, NOT ASSUMED FREE: the QK "
                        "chunks concatenate (no arithmetic) but must meet before the "
                        "softmax normalises the row, and those bytes ride the intra-chip "
                        "convention — reported, 0 s, because D17 declares one network law "
                        "and it is the p2p fabric BETWEEN chips; the PV partials are "
                        "add-reduced on FP_ADD lanes composed from the measured 22nm "
                        "library (D32) and timed with depth + ceil(results / lanes) - 1, "
                        "the same pipelined-tree expression every other reduction here "
                        f"uses. Per stage: {split_rows}."
                    ),
                )
            )
            floors = sizing.floors
            out.append(
                Relaxation(
                    constraint="sa_fabric_floor",
                    value="; ".join(
                        f"{name}: stage(s) {stages}" for name, stages in floors.items()
                    )
                    or "no floor recorded",
                    reason=(
                        "CTX-SPLIT requires that whatever stops the widening be a REAL "
                        "limit and be NAMED. Every stage carries exactly one of "
                        + ", ".join(cim_timing_FABRIC_FLOORS)
                        + ", and the point of the list is that a reader can tell an "
                        "achievement from a wall from an admission. "
                        + " ".join(
                            f"[{name}] " + FABRIC_FLOOR_REASONS[name]
                            for name in floors
                        )
                    ),
                )
            )
        out.append(
            Relaxation(
                constraint="sa_geometry_dataflow_validity",
                value="sa_cycles is exact vs ScaleSim OUTPUT-STATIONARY only",
                reason=SA_GEOMETRY_DATAFLOW_DISCLOSURE,
            )
        )
        out.append(
            Relaxation(
                constraint="sa_geometry_bandwidth_assumption",
                value="stall-free operand bandwidth ASSUMED; no stall model",
                reason=SA_GEOMETRY_BANDWIDTH_DISCLOSURE,
            )
        )
    for composition, label in (
        (sizing.composition, "derived_fabric_area"),
        (sizing.softmax_composition, "derived_softmax_area"),
    ):
        if composition is None:
            continue
        out.append(
            Relaxation(
                constraint=label,
                value=(
                    f"{composition.area_mm2:.6g} mm2, {composition.power_w:.6g} W"
                ),
                reason=(
                    "D32: the derived engine's silicon is a COMPOSITION of measured "
                    "synthesis blocks — "
                    + ", ".join(
                        f"{name} x {count}"
                        for name, count in composition.blocks.items()
                    )
                    + f" — at {composition.technology}. "
                    + composition.basis
                    + "."
                ),
            )
        )
    return out


def binding_term_disclosures(
    binding_term: Mapping[str, object], mapping: FwsMapping
) -> List[Relaxation]:
    """WHAT BINDS THE MACHINE THAT GETS BUILT — ADJ-10's required answer.

    ADJ-10 says that whatever binds after every composable engine has been
    derived must be a REAL limit — unscalable declared geometry or a bandwidth
    term — and must be NAMED in the artifact. This is where it is named, with
    the arithmetic that produced it, measured on the PRICED timeline rather
    than on the probe.
    """
    terms = list(binding_term.get("terms") or ())
    if not terms:
        return []
    largest = terms[0]
    device_class = str(largest.get("device_class", "?"))
    block = str(largest.get("block", "?"))
    busy = float(largest.get("busy_s", 0.0))
    analog_s = float(binding_term.get("analog_time_s", 0.0))
    span_s = float(binding_term.get("span_s", 0.0))
    device = mapping.device
    if device_class == "analog_macro":
        verdict = (
            "THE ANALOG M-PASS IS THE LARGEST TERM of the beat-setting stage: this "
            "machine runs at the analog floor, which is what ADJ-9 and ADJ-10 were "
            "adjudicated to reach."
        )
    elif block in ("attention_qk", "attention_pv", "attention_softmax"):
        geometry = device.derived_sa_geometry
        rows = int(device.fabric_rows)
        cols = int(device.fabric_cols)
        if geometry is not None:
            # SA-GEOM: the array is no longer the declared one, so the verdict
            # below — which is entirely about the DECLARED geometry being the
            # residue — would be a false sentence. What binds here is the
            # DERIVED geometry, and the levers it names are different ones.
            verdict = (
                "THE NEXT REAL LIMIT IS THE DERIVED ARRAY GEOMETRY. The largest term is "
                f"the attention fabric's {block} on a fabric of "
                f"{device.fabric_num_arrays} DERIVED arrays whose SHAPE was also derived "
                f"(SA-GEOM): {geometry.rows} x {geometry.cols_qk} on the QK group and "
                f"{geometry.rows} x {geometry.cols_pv} on the PV group, against the "
                f"declared {geometry.declared_rows} x {geometry.declared_cols}. Every "
                "fold already runs on its own array and another copy carries none, so "
                f"what is left is ONE array's own walk — ceil(n / C) column passes, each "
                f"costing k + R + C - 2 cycles, plus the fill/drain (3 x R = "
                f"{int(device.fabric_fill_drain_cycles)}). THE RULE ALREADY TOOK THE "
                "GEOMETRY LEVER: C was searched over the powers of two in [32, 1024] "
                "and this is the argmin, so a wider array is not an untaken option here "
                "— it was priced and it was worse or equal. SPLITTING ONE FOLD'S N "
                "ACROSS ARRAYS is still NOT modelled and still needs a claim about K "
                "broadcast and column-tile merging that no recorded reference in this "
                "repo covers. And the walk above is exact only under an OUTPUT-"
                "STATIONARY dataflow with stall-free operand bandwidth; both conditions "
                "ride this run's sa_geometry_dataflow_validity and "
                "sa_geometry_bandwidth_assumption disclosures."
            )
        else:
            verdict = (
                "THE NEXT REAL LIMIT IS THE DECLARED ARRAY GEOMETRY. The largest term is "
                f"the attention fabric's {block}, and it is at the floor ADJ-10's derivation "
                f"walked it to: the fabric is {device.fabric_num_arrays} DERIVED arrays of "
                f"{rows} x {cols}, so every fold of the call already runs on its own array "
                "and another copy of the measured block would carry no fold. What is left "
                f"is ONE array's own walk — ceil(n / {cols}) column passes, each costing "
                f"k + {rows} + {cols} - 2 cycles, plus the fill/drain — and no COUNT of "
                "measured blocks shortens it. TWO LEVERS WOULD, AND BOTH ARE REFUSED HERE "
                "RATHER THAN TAKEN. (1) A WIDER ARRAY: cim.fabric.cols is what the "
                "ceil(n / cols) term divides by, and a wider array would also be a whole "
                "number of measured 32x32 blocks — but rows x cols is the geometry the "
                "systolic closed form was validated at, bit-exact against OPTIMA's recorded "
                "ScaleSim outputs, and the fill/drain surrogate is 3 x rows. Deriving a new "
                "geometry would move a VALIDATED law onto an unvalidated shape, which is a "
                "card change a human makes, not a derivation (ADJ-4, D21). (2) SPLITTING ONE "
                "FOLD'S N ACROSS ARRAYS, which needs a claim about how the K matrix is "
                "broadcast and how the column tiles are merged that no recorded reference in "
                "this repo covers. So it is REPORTED as the limit rather than derived away "
                "(D28: a limit named is a provisioning finding; a limit padded away is not)."
            )
    elif device_class == "link":
        verdict = (
            "THE NEXT REAL LIMIT IS A BANDWIDTH TERM: the largest term is a priced "
            f"boundary transfer on the {block} link, which is bytes over the declared "
            "p2p law (D17) and moves with the network card, not with any engine width."
        )
    elif device_class == "macro_pool":
        verdict = (
            "THE NEXT REAL LIMIT IS THE PER-MACRO POOL (D12), which is sized per unit "
            "from the macro's own result rate and is not one of ADJ-10's composable "
            "shared-chiplet engines."
        )
    else:
        verdict = (
            f"THE NEXT REAL LIMIT is {device_class}/{block}, which no ADJ-10 derivation "
            "sizes."
        )
    return [
        Relaxation(
            constraint="binding_term",
            value=(
                f"stage {int(binding_term.get('stage', -1))} spends {busy:.6g} s on "
                f"{device_class}/{block} against {analog_s:.6g} s of analog m-pass "
                f"(stage span {span_s:.6g} s)"
            ),
            reason=(
                "ADJ-10 requires that whatever binds AFTER every composable digital "
                "engine has been derived be a REAL limit and be named. This row is that "
                "name, measured on the PRICED timeline — the machine that gets built, "
                "with the derived fabric, the derived softmax width and the derived "
                "scan engine all installed — so it is a different quantity from the "
                "probe's beat-setting evidence, which describes the machine the "
                "derivation ran against. " + verdict + " Terms (union of busy intervals, "
                "priced slice): "
                + "; ".join(
                    f"{row['device_class']}/{row['block']} {row['busy_s']:.4g} s"
                    for row in terms
                )
                + "."
            ),
        )
    ]


def engine_sizing_disclosures(
    sizing: Optional[DerivedEngineSizing],
) -> List[Relaxation]:
    """The D31 derivation, stated in the artifact rather than only on stdout."""
    if sizing is None:
        return []
    rows = ", ".join(
        f"stage {row.stage}: {row.used_cycles}/{row.budget_cycles} cycles, digital "
        f"{row.time_s:.4g} s vs analog {row.analog_time_s:.4g} s"
        for row in sizing.per_stage[:6]
    )
    binding = sizing.binding_row
    unreachable = sizing.unreachable_stages
    out = [
        Relaxation(
            constraint="derived_engine_sizing",
            value=(
                f"vector_lanes = {int(sizing.vector_lanes)} ({PROVENANCE_DERIVED_COUNT}), "
                f"binding stage {int(sizing.binding_stage)}, engine duty "
                f"{sizing.utilization * 100:.3f}% of that stage's own "
                f"{(binding.target_s if binding is not None else 0.0):.6g} s ANALOG "
                f"m-pass; {len(sizing.analog_bound_stages)} of {len(sizing.per_stage)} "
                "stage(s) analog-bound"
            ),
            reason=(
                "D31-v2 (ADJ-9): the scan/vector engine is never a swept or declared "
                "free knob, and it is sized to the ANALOG FLOOR rather than to the "
                "beat. Its width is DERIVED here as the smallest integer lane count for "
                "which EVERY stage's digital per-stage time fits that stage's OWN "
                "measured analog m-pass time, with no margin (D28). Both targets come "
                "from a PROBE pass in which every vector op is counted and costs "
                "nothing, so nothing the engine is sized against depends on the engine. "
                "The derivation inverts CimDeviceModel.price_vector_work exactly, so "
                "the engine sized here is the engine the run is then priced on. THE "
                "CRITERION IS digital-per-stage <= the stage's ANALOG m-pass time: the "
                "analog side is the binding term by construction wherever it can be. It "
                "does not make the MEASURED beat equal the analog time and cannot — a "
                "D29 stage holds one stream at a time, so its analog passes and its "
                "scan run in series and the measured beat is their sum at any width"
                + (
                    f" (here {binding.time_s:.6g} s of digital against a "
                    f"{binding.analog_time_s:.6g} s analog m-pass on the binding stage)"
                    if binding is not None
                    else ""
                )
                + f". Per stage: {rows}."
                + (
                    " UNREACHABLE on stage(s) "
                    + ", ".join(str(int(stage)) for stage in unreachable)
                    + f", which fall back to the {sizing.analog_beat_s:.6g} s analog "
                    "beat: a stage with no analog work has no analog time to be bound "
                    "by, and a stage whose analog time is below the engine's own "
                    "pipeline fill cannot be held at any width."
                    if unreachable
                    else " Every stage's criterion was reachable."
                )
                + (
                    " Stage -1 is the RETIRED lockstep regime's spelling for 'no pipeline "
                    "stage': that regime has one decode STEP rather than D staggered "
                    "stages, so the engine is sized against the step and there is one row."
                    if sizing.binding_stage < 0
                    else ""
                )
            ),
        )
    ]
    setter = dict(sizing.beat_setting_stage or {})
    if setter:
        largest = (setter.get("terms") or [{}])[0]
        out.append(
            Relaxation(
                constraint="analog_floor_reachability",
                value=(
                    f"{len(sizing.analog_bound_stages)} of {len(sizing.per_stage)} "
                    "stage(s) analog-bound after the derivation; the beat-setting stage "
                    f"{int(setter.get('stage', -1))} spends "
                    f"{float(largest.get('busy_s', 0.0)):.6g} s on "
                    f"{largest.get('device_class', '?')}/{largest.get('block', '?')} "
                    f"against {float(setter.get('analog_time_s', 0.0)):.6g} s of analog "
                    "m-pass"
                ),
                reason=(
                    "ADJ-9 asks the ANALOG side to be the binding term wherever that is "
                    "physically reachable. The derivation guarantees the half it can: "
                    "the DERIVED scan/vector engine is sized so its per-stage time fits "
                    "inside that stage's own analog m-pass time. It cannot guarantee "
                    "the other half on its own, because a stage also runs silicon D31 "
                    "does not size. ADJ-10 then took the two that used to be DECLARED — "
                    "the attention fabric's array count and the softmax pipeline's lane "
                    "count — and derived them against this same per-stage target, so "
                    "what is left underived is the array GEOMETRY (rows x cols) and the "
                    "terms that are not engines. READ THIS ROW AS THE DERIVATION'S OWN "
                    "INPUT, NOT AS THE MACHINE: it is measured on the PROBE pass, where "
                    "the fabric still sits at its declared seed width. What binds the "
                    "machine that gets BUILT is measured on the priced timeline and "
                    "reported under its own name, "
                    "evaluation.digital_silicon.binding_term. On the probe: "
                    + (
                        "the analog m-pass IS the largest term of the beat-setting "
                        "stage."
                        if setter.get("analog_is_largest_term")
                        else "the largest term is NOT analog, so no scan lane count "
                        "could have moved the probe's beat. The terms are itemized in "
                        "evaluation.digital_silicon.derived_engine_sizing."
                        "beat_setting_stage."
                    )
                    + " Terms (union of busy intervals, probe slice): "
                    + "; ".join(
                        f"{row['device_class']}/{row['block']} {row['busy_s']:.4g} s"
                        for row in setter.get("terms", ())
                    )
                    + f". Stage span {float(setter.get('span_s', 0.0)):.6g} s."
                ),
            )
        )
    probe = dict(_PROBE_BEAT_CONVERGENCE)
    if probe.get("regime") == REGIME_FILLED:
        out.append(
            Relaxation(
                constraint="derived_engine_probe_beat",
                value=(
                    f"probe beat {sizing.analog_beat_s:.6g} s, "
                    + ("CONVERGED" if probe.get("converged") else "NOT CONVERGED")
                    + f" ({int(probe.get('converged_tail_beats', 0))} settled interval(s), "
                    f"{int(probe.get('ramp_beats_dropped', 0))} dropped as ramp, whole "
                    f"steady sample spread "
                    f"{float(probe.get('steady_beat_spread', 0.0)) * 100:.4f}%)"
                ),
                reason=(
                    "The probe pass is where BOTH of ADJ-9's inputs are measured: each "
                    "stage's analog m-pass time (the sizing target) and this beat (the "
                    "fallback budget for a stage the criterion cannot reach). The beat "
                    "is a measurement like any other, so this line is where a reader "
                    "sees whether it settled. Under D31-v2 the beat no longer sizes any "
                    "stage that has analog work, so a still-ramping beat can only move "
                    "a FALLBACK row; the analog stage times themselves are sums of "
                    "op durations and do not ramp. The beat is read off the same "
                    "settled tail the priced pass uses (P7.9), so it does not move with "
                    "the window."
                ),
            )
        )
    if sizing.composition is not None:
        out.append(
            Relaxation(
                constraint="derived_engine_area",
                value=(
                    f"{sizing.composition.area_mm2:.6g} mm2, "
                    f"{sizing.composition.power_w:.6g} W"
                ),
                reason=(
                    "D32: the derived engine's silicon is a COMPOSITION of measured "
                    "synthesis blocks — "
                    + ", ".join(
                        f"{name} x {count}"
                        for name, count in sizing.composition.blocks.items()
                    )
                    + f" — at {sizing.composition.technology}. "
                    + sizing.composition.basis
                    + "."
                ),
            )
        )
    return out


def evaluate_fws(
    program: Program,
    mapping: Optional[FwsMapping] = None,
    *,
    capacity_overrides: Optional[Mapping[int, int]] = None,
    pricing: Optional[PricingResult] = None,
) -> FwsEvaluation:
    """Price a placed DAG, replay it once, and project every number from it.

    This is the P4 contract in one function: a mapped DAG in, one timeline out,
    and every metric, occupancy, duty cycle, verdict, energy component and
    derived pool width read off that timeline (P4 §3).
    """
    mapping = mapping or program.meta.misc.get("fws_mapping")
    serving = program.meta.misc.get("fws_serving")
    if mapping is None or serving is None:
        raise MappingError(
            "execution",
            "evaluate_fws needs an FWS program built by program.fws_build.build_fws_program.",
        )
    # D31: the engine is DERIVED from the beat, so it must exist before the
    # run is priced. Pass A measures the beat the analog stages set; this
    # installs the width that holds it. A caller that passes its own `pricing`
    # has already priced the run and is not re-sized under it.
    engine_sizing = None
    fabric_sizing = None
    if pricing is None:
        fabric_sizing, engine_sizing = _derive_engine(
            program, mapping, serving, capacity_overrides
        )
    pricing = pricing or price_program(
        program, mapping, capacity_overrides=capacity_overrides
    )
    timeline = evaluate_detailed(
        program,
        None,
        {},
        resources=pricing.resources,
        durations=pricing.durations,
        require_pipeline=False,
    )
    pipeline: "OrderedDict[str, object]" = OrderedDict()
    if str(getattr(serving, "regime", "")) == REGIME_FILLED:
        metrics, step_times, extrapolation, pipeline = _build_pipeline_metrics(
            mapping, serving, pricing.costs, timeline, annotations_of(program)
        )
    else:
        metrics, step_times, extrapolation = _build_metrics(
            mapping, serving, pricing.costs, timeline
        )
    occupancy, duty = _build_occupancy(mapping, pricing, timeline, program)
    annotations = annotations_of(program)
    # ADJ-10's headline: WHAT BINDS THE MACHINE THAT GETS BUILT. Measured on
    # THIS timeline — every derived width installed, every op timed — which is
    # a different question from the probe evidence the derivations ran against,
    # and it gets a different name (D21).
    binding_term = _beat_setting_stage_terms(
        serving, annotations, pricing.costs, timeline, slice_name="PRICED"
    )
    utilization = _build_utilization(
        mapping, pricing, occupancy, annotations, makespan_of(timeline)
    )
    bank_passes = _build_bank_passes(mapping, pricing, annotations, serving)
    bank_sharing = _build_bank_sharing(
        mapping, program, pricing, timeline, annotations, serving
    )
    energy = _build_energy(pricing)
    memory, memory_disclosures, residency = _build_memory(
        mapping, serving, program, pricing, timeline
    )
    pools = _build_pools(mapping, program, pricing, timeline)
    disclosures = list(mapping.relaxations())
    # The mapping's op_durations relaxation is RETIRED by this call: the DAG it
    # describes is now priced. Keeping it would disclose a gap that no longer
    # exists, which is its own kind of dishonesty.
    disclosures = [item for item in disclosures if item.constraint != "op_durations"]
    disclosures.extend(pricing.disclosures)
    disclosures.extend(analog_card_disclosures(mapping))
    disclosures.extend(nonstandard_macro_disclosures(mapping))
    disclosures.extend(fabric_sizing_disclosures(fabric_sizing))
    disclosures.extend(engine_sizing_disclosures(engine_sizing))
    disclosures.extend(binding_term_disclosures(binding_term, mapping))
    disclosures.extend(memory_disclosures)
    disclosures.extend(
        _utilization_disclosures(
            mapping, utilization, bank_passes, bank_sharing, program
        )
    )
    if pipeline:
        disclosures.append(
            Relaxation(
                constraint="pipeline_fill_transient",
                value=(
                    f"{int(pipeline['fill_beats'])} fill beats before "
                    f"{int(pipeline['steady_beats_measured'])} measured steady beats"
                ),
                reason=(
                    "D29's pipeline is full only after every stage has a stream, which "
                    f"takes D - 1 = {int(pipeline['fill_beats'])} beats. The window "
                    f"lowers {int(pipeline['beats_lowered'])} beats: the fill is IN the "
                    "DAG (its contention is priced like any other) and the beat is read "
                    "from the exits of the completed traversals, whose intervals are "
                    "printed one by one in the pipeline block so a reader can see the "
                    "convergence rather than trust an average. Streams that enter too "
                    "late to finish inside the window are lowered as far as the window "
                    "reaches and their partial work is on the timeline, contending, "
                    "exactly as it would in the machine."
                ),
            )
        )
    if pipeline:
        # D29's beat is a MEASUREMENT, so whether it has CONVERGED inside the
        # lowered window is part of the number. The fill allowance is D - 1
        # beats, which is exactly right for a pipeline whose stages have equal
        # service times; an UNEVEN stage plan takes longer to settle, and a
        # reader must be told which of the two this run was rather than shown a
        # median that quietly averaged a ramp.
        steady = [float(value) for value in pipeline["steady_beat_intervals_s"]]
        settled = [float(value) for value in pipeline["converged_tail_intervals_s"]]
        ramp = [float(value) for value in pipeline["ramp_beat_intervals_s"]]
        beat_value = float(pipeline["beat_s"])
        spread = (
            max(abs(value - beat_value) for value in steady) / beat_value
            if steady and beat_value > 0
            else 0.0
        )
        tail = (
            max(abs(value - beat_value) for value in settled) / beat_value
            if len(settled) >= 2 and beat_value > 0
            else 0.0
        )
        pipeline["steady_beat_spread"] = spread
        pipeline["steady_beat_tail_agreement"] = tail
        # CONVERGED means the reported beat came off a tail of at least two
        # intervals that agree with each other — i.e. the machine was seen
        # repeating itself. A single steady interval cannot show that and says
        # so under its own name rather than being called converged.
        pipeline["converged_tail_beats"] = len(settled)
        pipeline["ramp_beats_dropped"] = len(ramp)
        pipeline["beat_converged"] = len(settled) >= 2
        if ramp:
            disclosures.append(
                Relaxation(
                    constraint="beat_read_from_the_converged_tail",
                    value=(
                        f"{len(ramp)} of {len(steady)} steady interval(s) dropped as "
                        f"RAMP; the beat is the median of the remaining {len(settled)}, "
                        f"which agree to {tail * 100:.4f}%"
                    ),
                    reason=(
                        "The exits were still SETTLING inside the lowered window: the "
                        "fill allowance of D - 1 beats is exact only when every stage "
                        "has the same service time, and this run's stage plan is uneven, "
                        "so a stream can wait on a slower stage and the exits take "
                        "longer than D - 1 beats to become periodic. The dropped "
                        "intervals are printed in the pipeline block as "
                        "ramp_beat_intervals_s and the whole steady sample spreads "
                        f"{spread * 100:.1f}% about the beat. Reading the beat off the "
                        "tail instead of the median of the whole sample is what makes "
                        "the headline independent of the window size; the alternative "
                        "measured 12% higher on Granite-4.0-H-Tiny at decode_window 2. "
                        "Nothing is padded and no interval is smoothed — the ramp is "
                        "named, not averaged."
                    ),
                )
            )
        if steady and not pipeline["beat_converged"]:
            disclosures.append(
                Relaxation(
                    constraint="pipeline_beat_NOT_converged",
                    value=(
                        f"{len(steady)} steady interval(s), no two of which agree to "
                        f"{BEAT_CONVERGENCE_TOL:.0e}"
                        if len(steady) >= 2
                        else "one steady interval: a single reading cannot show a period"
                    ),
                    reason=(
                        "The headline tokens/s = 1/beat is NOT a settled measurement on "
                        "this run. The beat is the last inter-exit interval, which is "
                        "the best reading the window contains, and it is published under "
                        "this flag rather than presented as converged. Lengthening "
                        "mapping.decode_window until pipeline.beat_converged is true is "
                        "the fix; it costs a proportionally larger DAG and is not done "
                        "silently (D28: a transient is a fact, not noise)."
                    ),
                )
            )
    if extrapolation.get("extrapolated"):
        disclosures.append(
            Relaxation(
                constraint="decode_series_window",
                value=(
                    f"{extrapolation['lowered_decode_steps']} of "
                    f"{extrapolation['declared_decode_steps']} steps lowered"
                ),
                reason=str(extrapolation.get("basis", "")),
            )
        )
    return FwsEvaluation(
        mapping=mapping,
        program=program,
        serving=serving,
        pricing=pricing,
        timeline=timeline,
        metrics=metrics,
        occupancy=occupancy,
        duty_cycles=duty,
        utilization=utilization,
        bank_passes=bank_passes,
        bank_sharing=bank_sharing,
        energy=energy,
        memory=memory,
        pools=pools,
        extrapolation=extrapolation,
        disclosures=tuple(disclosures),
        decode_step_times_s=step_times,
        pipeline=pipeline,
        state_residency=residency,
        binding_term=binding_term,
    )


# ---------------------------------------------------------------------------
# The reports (P4.6): one JSON, and a text section RENDERED FROM IT
# ---------------------------------------------------------------------------

#: The results txt this section is appended to still prints run_perf's own
#: prefill/decode totals. They are a DIFFERENT quantity with a different name,
#: and saying so is cheaper than a reader assuming otherwise (D21).
_RESULTS_FILE_NOTE = Relaxation(
    constraint="surrounding_results_file_totals",
    value="a different accounting, named",
    reason=(
        "run_perf's results file prints Prefill Time / Decode Time from the SHARED "
        "per-op analytic pipeline: the same P2 analog law prices the weight GEMMs "
        "there, but they are scheduled on the GPU-shaped program, not on the placed "
        "FWS DAG, and the decode total is an interpolation over sampled contexts. "
        "Those numbers are not this section's prefill latency or decode series and the "
        "two are not interchangeable. A1 makes the placed-DAG figures the authoritative "
        "ones for a mapped run."
    ),
)

_SINGLE_ACCOUNTING_NOTE = (
    "Every number in the evaluation block is a projection of ONE timeline "
    "(program.analytic_sim over the priced DAG). No metric is computed by a second "
    "formula, no metric is a period times a count, and the pass-1/2 closed-form "
    "section is NOT printed beside this one — two accountings of one metric never "
    "share an artifact (D21, A1, ADJ-6)."
)


def _pool_block(evaluation: FwsEvaluation) -> "OrderedDict[str, object]":
    """The derived pool sizing, grouped by measured concurrency (P4.5)."""
    groups: "OrderedDict[int, List[PoolSizingReport]]" = OrderedDict()
    for pool in evaluation.pools:
        groups.setdefault(pool.peak_concurrent_demand, []).append(pool)
    rows = []
    for peak in sorted(groups):
        members = groups[peak]
        example = members[0]
        rows.append(
            OrderedDict(
                (
                    ("peak_concurrent_demand", int(peak)),
                    ("macros", len(members)),
                    ("example_macro", int(example.macro_id)),
                    ("example_owner", example.owner),
                    ("lanes", int(example.scaled.lanes)),
                    ("adders", int(example.scaled.adders)),
                    ("conv_lanes", int(example.scaled.conv_lanes)),
                    ("area_mm2", float(example.scaled.area_mm2)),
                    ("macro_footprint_mm2", float(example.scaled.macro_footprint_mm2)),
                    ("area_share", float(example.scaled.area_share)),
                    ("disclose", bool(example.scaled.disclose)),
                    ("report", example.report_line),
                )
            )
        )
    return OrderedDict(
        (
            ("basis", (
                "peak concurrent digital demand per macro, measured on the timeline as "
                "the largest number of that macro's pool ops whose [ready, finish] "
                "demand intervals overlap; multiplied into P2's per-unit "
                "CimDeviceModel.digital_pool_sizing derivation. Nothing is clamped: an "
                "absurd derived width prints as an absurd derived width (D12)."
            )),
            ("macros_measured", len(evaluation.pools)),
            ("sizings", rows),
        )
    )


def _digital_silicon_block(evaluation: FwsEvaluation) -> "OrderedDict[str, object]":
    """D32: every digital mm2 and watt, composed from measured blocks.

    ONE ACCOUNTING (D21). The shared chiplet term is the composed engine census
    times the chiplet COUNT the mapping placed; the per-macro pool term is the
    composed pool census times the analog macro SLOT count the same summary
    reports. Neither is the declared `area_mm2` placeholder — when a card names
    a library the placeholder is not added on top, and when it does not, this
    block says so and carries no power figure at all.
    """
    device = evaluation.mapping.device
    summary = evaluation.mapping.summary()
    chiplets = int(summary.get("shared_digital_chiplets", 0) or 0)
    slots = int(summary.get("analog_macro_slots", 0) or 0)
    out: "OrderedDict[str, object]" = OrderedDict(
        (
            ("producer", "cim_timing engine compositions (QIF P7.8, D31/D32)"),
            ("library", None),
            ("shared_digital_chiplets", chiplets),
            ("analog_macro_slots", slots),
        )
    )
    if not device.has_synthesis_library():
        out["composed"] = False
        out["shared_digital_area_mm2_per_chiplet"] = float(
            device.shared_digital_area_mm2()
        )
        out["disclosures"] = list(device.shared_digital_area_disclosures())
        return out
    library = device.synthesis_library()
    out["library"] = OrderedDict(
        (
            ("technology", library.technology),
            ("path", library.path),
            ("source", library.source),
            ("source_reports", library.source_reports),
            ("blocks", len(library.blocks)),
            ("measurement", "measured via synthesis"),
        )
    )
    out["composed"] = True
    engines = [
        composition.summary() for composition in device.shared_digital_compositions()
    ]
    # THE POOL THE MACHINE BUILDS. The conv is deliberately NOT in it - see
    # CimDeviceModel.CONV_POOL_WIDTH_UNRESOLVED - so this is the shift-add
    # reduction and its registers, which do consume ADC results one for one.
    # The conv's absence from the SILICON is named in the disclosures below;
    # its WORK is priced wherever the card times it.
    pool_sizing = device.digital_pool_sizing()
    pool = device.macro_pool_composition(pool_sizing)
    chiplet_area = float(device.shared_digital_area_mm2())
    chiplet_power = float(device.shared_digital_power_w())
    out["shared_digital_chiplet"] = OrderedDict(
        (
            ("area_mm2_per_chiplet", chiplet_area),
            ("power_W_per_chiplet", chiplet_power),
            ("area_mm2_total", chiplet_area * chiplets),
            ("power_W_total", chiplet_power * chiplets),
            ("engines", engines),
        )
    )
    measured_peak = max(
        (int(report.peak_concurrent_demand) for report in evaluation.pools), default=0
    )
    out["per_macro_pool"] = OrderedDict(
        (
            ("area_mm2_per_macro", pool.area_mm2),
            ("power_W_per_macro", pool.power_w),
            ("area_mm2_total", pool.area_mm2 * slots),
            ("power_W_total", pool.power_w * slots),
            ("composition", pool.summary()),
            ("sizing", "D12 per-unit"),
            (
                "basis",
                "the SILICON term prices D12's PER-UNIT pool derivation on every one of "
                f"the {slots} enumerated macro slots. It is deliberately NOT the "
                "demand-scaled width: evaluation.digital_pool reports what the TIMELINE "
                "measured each macro needing (peak concurrent demand up to "
                f"{measured_peak} on this run), which is a different question — how wide "
                "a pool has to be so it never blocks — asked only of the macros that "
                "hold tiles. Two names, two quantities, and this one is the machine's "
                "provisioned silicon (D21).",
            ),
        )
    )
    # THE STATE STORE IS DIGITAL SILICON TOO (ADJ-11). It used to be missing
    # from this total while the sweep's own digital term carried it, so one
    # machine had two digital areas and the markdown report printed the smaller
    # one three lines below the line naming the store (D21). The store is sized
    # from the SAME measured per-stage bill the residency block reports, so this
    # is the same accounting read here, not a second one.
    store_area = 0.0
    store_power = 0.0
    store_macros = 0
    residency = getattr(evaluation, "state_residency", None) or {}
    for row in (residency.get("per_stage") or []):
        store_area += float(row.get("store_area_mm2", 0.0) or 0.0)
        store_macros += int(row.get("store_macros", 0) or 0)
    if store_macros and device.has_synthesis_library():
        try:
            block = device.synthesis_library().block(STATE_SRAM_BLOCK)
            store_power = float(block.power_w) * store_macros
        except Exception:
            store_power = 0.0
    out["state_store_area_mm2_total"] = store_area
    out["state_store_power_W_total"] = store_power
    out["state_store_macros"] = store_macros
    if device.conv_dims() and device.conv_placement == "macro_pool":
        out["conv_pool_unpriced"] = device.CONV_POOL_WIDTH_UNRESOLVED
    out["state_store_basis"] = (
        f"{store_macros} measured {STATE_SRAM_BLOCK} macro(s), summed over the per-stage "
        "stores in evaluation.state_residency. The power is the block's measured LEAKAGE "
        "only: the library measures read and write energy per word, this repo has no "
        "access law to apply them with, and the store's DYNAMIC energy is therefore "
        "UNPRICED and named here rather than reported as zero."
    )
    # D36 (GEORGE, 2026-08-29): THE CHIPLET IS ENGINES + STATE IN ONE AREA, and
    # the [50, 400] mm2 band is PACKAGING ARITHMETIC that follows from it.
    #
    # REPORTED, NOT APPLIED — the same standing as `derived_chiplets`. This block
    # says what the D36 packaging law makes of the areas the D32 accounting above
    # already composed; it does NOT re-price them, and `digital_area_mm2_total`
    # below is byte-for-byte the number it always was. There is ONE accounting of
    # every mm2 (D21): `engines_mm2` is `area_mm2_per_chiplet` read again, and
    # `state_mm2` is this chiplet's SHARE of the one `state_store_area_mm2_total`
    # reported three lines up — never a second store.
    #
    # The state is INSIDE the area on purpose: D36 says a plan that holds more
    # state buys a bigger chiplet, so the band is walked against engines + state
    # and not against the engines alone.
    packaging = None
    groups_available = 1
    group_plan = None
    try:
        group_plan = device.layer_group_plan()
        groups_available = max(1, int(group_plan.instances))
    except Exception:
        # A model with no layer plan the motif law can read is not a packaging
        # failure: it is one group, and the block says so rather than vanishing.
        group_plan = None
    state_share = (store_area / chiplets) if chiplets > 0 else store_area
    try:
        packaging = derive_digital_chiplet_packaging(
            chiplet_area,
            state_share,
            groups_available=groups_available,
            context="cim.cards.<digital chiplet> (D36 packaging, REPORTED)",
        )
    except Exception as exc:  # pragma: no cover - reported, never fatal
        out["chiplet_packaging"] = OrderedDict(
            (
                ("measured", False),
                ("reason", f"the D36 packaging law could not be read here: {exc}"),
            )
        )
    if packaging is not None:
        packed = OrderedDict(packaging.summary())
        packed["measured"] = True
        packed["applied"] = False
        packed["groups_available"] = groups_available
        packed["group_motif_period"] = (
            int(group_plan.period) if group_plan is not None else None
        )
        packed["state_share_basis"] = (
            f"{store_area:.6g} mm2 of state store over {chiplets} shared digital "
            f"chiplet(s) = {state_share:.6g} mm2 inside each. This is the SAME store "
            "state_store_area_mm2_total reports, divided, not a second one."
        )
        packed["band_arithmetic"] = (
            f"engines {chiplet_area:.6g} mm2 + state {state_share:.6g} mm2 = "
            f"{packaging.area_mm2:.6g} mm2 against the band "
            f"[{DIGITAL_CHIPLET_AREA_MIN_MM2:.6g}, {DIGITAL_CHIPLET_AREA_MAX_MM2:.6g}] "
            f"mm2 -> chips_per_group = {packaging.chips_per_group}, "
            f"groups_per_chiplet = {packaging.groups_per_chiplet}"
        )
        packed["disclosures"] = list(packaging.disclosures)
        packed["standing"] = (
            "REPORTED, NOT APPLIED: this is the D36 reading of the areas the D32 "
            "accounting above composed. It re-prices nothing, and "
            "digital_area_mm2_total is unchanged by it."
        )
        out["chiplet_packaging"] = packed
    out["digital_area_mm2_total"] = (
        chiplet_area * chiplets + pool.area_mm2 * slots + store_area
    )
    out["digital_power_W_total"] = (
        chiplet_power * chiplets + pool.power_w * slots + store_power
    )
    sizing = device.derived_engine
    # ADJ-9's beat-setting evidence rides derived_engine_sizing itself, so the
    # block carries no second copy of it (D21).
    out["derived_engine_sizing"] = sizing.summary() if sizing is not None else None
    # ADJ-10's twin: the attention fabric and the softmax pipeline, derived
    # against the SAME per-stage analog m-pass times. None = the card PINS them
    # (an override riding its own disclosure) or the run runs no attention.
    fabric = device.derived_fabric
    out["derived_fabric_sizing"] = fabric.summary() if fabric is not None else None
    out["fabric_num_arrays"] = int(device.fabric_num_arrays)
    out["fabric_num_arrays_provenance"] = device.fabric_provenance
    out["fabric_softmax_lanes"] = int(device.fabric_softmax_lanes)
    # SA-GEOM: the array SHAPE this run was priced on, beside the COUNT. It is
    # written ONLY when the shape was DERIVED. A declared-geometry run must
    # serialize BYTE for BYTE what it always did (the shipped atlas artifacts
    # are compared as bytes against a fresh run), so an opt-in rule adds an
    # opt-in key rather than a null one, and the declared shape already rides
    # derived_fabric_sizing.array_rows / array_cols where it always did.
    geometry = device.derived_sa_geometry
    if geometry is not None:
        out["fabric_geometry_mode"] = device.fabric_geometry_mode
        out["fabric_array_rows"] = int(device.fabric_rows)
        out["fabric_array_cols_qk"] = int(device.sa_geometry("qk")[1])
        out["fabric_array_cols_pv"] = int(device.sa_geometry("pv")[1])
        out["fabric_array_geometry_provenance"] = PROVENANCE_DERIVED_COUNT
        out["fabric_fill_drain_cycles"] = int(device.fabric_fill_drain_cycles)
        # CTX-SPLIT: how deep the context was dealt, what the reduce cost in
        # silicon, and WHICH REAL FLOOR stopped the widening on each stage.
        if fabric is not None:
            out["fabric_context_split"] = int(fabric.context_split)
            out["fabric_combine_lanes"] = int(fabric.combine_lanes)
            out["fabric_floors"] = dict(fabric.floors)
    # ADJ-10's headline: what binds the machine that GETS BUILT, measured on
    # the priced timeline. Not the probe's evidence and never merged with it.
    out["binding_term"] = dict(evaluation.binding_term or {}) or None
    # None = no vector op ran, so no engine was derived and none is invented (D31).
    out["vector_lanes"] = device.resolved_vector_lanes()
    out["vector_lanes_provenance"] = device.vector_lanes_provenance
    out["disclosures"] = list(device.shared_digital_area_disclosures())
    return out


def report_document(evaluation: FwsEvaluation) -> "OrderedDict[str, object]":
    """The shared report JSON: mapping block (P3), evaluation block (P4), disclosures.

    One version field, one producer per block, every metric names its accounting
    basis, and no derived figure appears without the inputs it came from (P4 §8).
    """
    mapping = evaluation.mapping
    serving = evaluation.serving
    params = mapping.device.params
    evaluated = OrderedDict(
        (
            ("producer", "fws_eval.evaluate_fws (QIF P4)"),
            ("single_accounting", _SINGLE_ACCOUNTING_NOTE),
            ("ops_priced", len(evaluation.pricing.costs)),
            ("makespan_s", evaluation.makespan_s),
            ("metrics", [metric.as_dict() for metric in evaluation.metrics]),
            (
                "decode_series",
                OrderedDict(
                    (
                        ("lowered_steps", len(evaluation.decode_step_times_s)),
                        ("step_times_s", list(evaluation.decode_step_times_s)),
                        (
                            "basis",
                            (
                                "finish(beat k terminal op) - finish(beat k-1 terminal "
                                "op) on the one timeline. Under D29 this is the SAME "
                                "series as pipeline.beat_intervals_s under the legacy "
                                "name (one accounting, two spellings, D21): there is no "
                                "decode STEP in a filled pipeline and no prefill op is "
                                "lowered at all (D25), so the pre-D29 gloss 'step -1 is "
                                "the prefill terminal' does not apply here. The HEADLINE "
                                "reads pipeline.beat_s, not this series."
                                if str(serving.regime) == REGIME_FILLED
                                else "finish(step k terminal op) - finish(step k-1 "
                                "terminal op) on the one timeline; step -1 is the "
                                "prefill terminal (P4 §3)."
                            ),
                        ),
                    )
                ),
            ),
            ("extrapolation", OrderedDict(evaluation.extrapolation)),
            # D29's two first-class quantities: the rotation the beat was read
            # from, and the state every stage must hold for all D streams.
            ("pipeline", OrderedDict(evaluation.pipeline)),
            ("state_residency", OrderedDict(evaluation.state_residency)),
            (
                "occupancy",
                [
                    OrderedDict(
                        (
                            ("device", row.device_id),
                            ("device_class", row.device_class),
                            ("chip", row.chip_id),
                            ("macro", row.macro_id),
                            ("owner", row.owner),
                            ("ops", row.ops),
                            ("busy_s", row.busy_s),
                            ("occupancy", row.occupancy),
                        )
                    )
                    for row in evaluation.occupancy
                    if row.ops
                ],
            ),
            (
                "occupancy_basis",
                "busy time / makespan per device, where busy time is the UNION of the "
                "op intervals the timeline scheduled on it, attributed to the device's "
                "named owner (D21). Transfers occupy no device (D17).",
            ),
            (
                "utilization",
                [row.as_dict() for row in evaluation.utilization],
            ),
            (
                "utilization_basis",
                "per-device-class utilization is a REQUIRED output of every run "
                "(P7.3, D28): low utilization is a provisioning finding the report "
                "surfaces, never a number it leaves out. Idle devices are inside the "
                "averages, and the classes with no ops still get a row.",
            ),
            ("bank_passes", OrderedDict(evaluation.bank_passes)),
            ("bank_sharing", OrderedDict(evaluation.bank_sharing)),
            ("packing", _packing_block(mapping)),
            (
                "duty_cycles",
                OrderedDict(
                    (str(macro_id), float(value))
                    for macro_id, value in sorted(evaluation.duty_cycles.items())
                ),
            ),
            (
                "duty_cycle_basis",
                "the same projection restricted to analog macro devices — the P5 atlas "
                "colorant. A macro's duty cycle is its analog device's busy union over "
                "the makespan; its pool is reported separately (digital_pool).",
            ),
            (
                "energy",
                OrderedDict(
                    (
                        (
                            "components",
                            [
                                OrderedDict(
                                    (
                                        ("key", component.key),
                                        ("label", component.label),
                                        ("energy_pj", component.energy_pj),
                                        ("coverage", component.coverage),
                                        ("basis", component.basis),
                                    )
                                )
                                for component in evaluation.energy
                            ],
                        ),
                        ("total_pj", evaluation.total_energy_pj),
                        (
                            "total_basis",
                            "the arithmetic sum of the components above and nothing else. "
                            "Per-component coverage labels replace the single PARTIAL word "
                            "the pass-1/2 report printed (ADJ-6).",
                        ),
                    )
                ),
            ),
            (
                "memory",
                [
                    OrderedDict(
                        (
                            ("scope", verdict.scope),
                            ("tier", verdict.tier),
                            ("owner", verdict.owner),
                            ("high_water_bytes", verdict.high_water_bytes),
                            ("capacity_bytes", verdict.capacity_bytes),
                            ("status", verdict.status),
                            ("contributors", OrderedDict(verdict.contributors)),
                            ("basis", verdict.basis),
                            ("disclosure", verdict.disclosure),
                        )
                    )
                    for verdict in evaluation.memory
                ],
            ),
            ("digital_pool", _pool_block(evaluation)),
            ("digital_silicon", _digital_silicon_block(evaluation)),
        )
    )
    return OrderedDict(
        (
            ("schema", REPORT_SCHEMA),
            (
                "note",
                "The DAG report. It REPLACES the pass-1/2 closed-form section for a "
                "mapped run; the closed form is legacy and is reachable from the "
                "validation script only (ADJ-6, A1).",
            ),
            (
                "model",
                OrderedDict(
                    (
                        ("producer", "P1 / config"),
                        # ``model_id`` is the model. ``model_type`` is the PRICING
                        # CARRIER it is priced through and is not always the same
                        # name; when they differ the mapping raises a
                        # ``model_type_carrier`` disclosure that reconciles them.
                        ("model_id", str(mapping.model_id)),
                        ("model_type", str(getattr(mapping.model, "model_type", ""))),
                        ("num_layers", int(params.num_layers)),
                        ("hidden_dim", int(params.hidden_dim)),
                        ("num_heads", int(params.num_heads)),
                        ("kv_heads", int(params.kv_heads)),
                        ("head_dim", int(params.head_dim)),
                    )
                ),
            ),
            (
                "serving",
                OrderedDict(
                    (
                        (
                            "producer",
                            "P1 / config (D29)"
                            if str(serving.regime) == REGIME_FILLED
                            else "P1 / config (D15, RETIRED regime)",
                        ),
                        ("regime", str(serving.regime)),
                        # D29: batch size does not exist on this surface. The
                        # field prints the local batch the regime PINS (1) so a
                        # reader of an old artifact and a reader of this one are
                        # never comparing two different machines under one word.
                        ("batch", int(serving.batch)),
                        ("resident_streams", int(serving.streams)),
                        # D38: the stages the machine has, the fill it was
                        # asked for and the fill the integer stream count
                        # realizes. A rate read without them is a rate for a
                        # machine the reader cannot identify.
                        ("stages", int(getattr(serving, "stages", 0))),
                        (
                            "pp_util",
                            float(getattr(evaluation.mapping, "pp_util", PP_UTIL_FULL)),
                        ),
                        ("pipeline_fill", float(getattr(serving, "fill", 0.0))),
                        ("beats_lowered", int(serving.beats)),
                        ("steady_beats", int(serving.steady_beats)),
                        ("prefill_len", int(serving.prefill_len)),
                        ("decode_len", int(serving.decode_len)),
                        ("decode_steps_lowered", int(serving.decode_steps)),
                    )
                ),
            ),
            (
                "mapping",
                OrderedDict(
                    [("producer", "P3 / fws_mapping.build_mapping")]
                    + list(mapping.summary().items())
                ),
            ),
            ("evaluation", evaluated),
            (
                "disclosures",
                merge_disclosures(tuple(evaluation.disclosures) + (_RESULTS_FILE_NOTE,)),
            ),
        )
    )


#: The packing-comparison artifact's schema id (P7.3 item 4).
PACKING_COMPARISON_SCHEMA = "fws_packing_comparison/1"


def _packing_point(evaluation: FwsEvaluation) -> "OrderedDict[str, object]":
    """One placement law, as the four numbers P7.3 asks for and their inputs."""
    mapping = evaluation.mapping
    summary = mapping.summary()
    steps = list(evaluation.decode_step_times_s)
    tokens = [
        metric for metric in evaluation.metrics if metric.key.endswith(".tokens_per_s")
    ]
    return OrderedDict(
        (
            ("packing", mapping.packing),
            ("analog_macro_slots", int(summary["analog_macro_slots"])),
            ("macros_holding_tiles", int(summary["macros_holding_tiles"])),
            ("unowned_macro_slots", int(summary["unowned_macro_slots"])),
            ("tiles", int(summary["tiles"])),
            ("packing_accounting", _packing_block(mapping)),
            ("makespan_s", float(evaluation.makespan_s)),
            # P7.7 / D29: under the FILLED PIPELINE the per-token period is the
            # BEAT and the series below is the beat series (the intervals
            # between token exits). The key keeps its name because it is the
            # same quantity — the time from one token to the next — and the
            # regime beside it says which machine produced it.
            ("serving_regime", str(getattr(evaluation.serving, "regime", ""))),
            ("resident_streams", int(getattr(evaluation.serving, "streams", 0))),
            ("stages", int(getattr(evaluation.serving, "stages", 0))),
            ("pp_util", float(getattr(mapping, "pp_util", PP_UTIL_FULL))),
            ("decode_step_times_s", [float(value) for value in steps]),
            ("median_decode_step_s", _median(steps)),
            ("tokens_per_s", float(tokens[0].value) if tokens else None),
            ("bank_passes", OrderedDict(evaluation.bank_passes)),
            ("bank_sharing", OrderedDict(evaluation.bank_sharing)),
            ("utilization", [row.as_dict() for row in evaluation.utilization]),
            (
                "accumulator_ops",
                sum(
                    1
                    for cost in evaluation.pricing.costs
                    if cost.energy_component == "digital_accumulation"
                ),
            ),
        )
    )


def packing_comparison_document(
    dedicated: FwsEvaluation,
    dense: FwsEvaluation,
    *,
    hardware_config: str,
    model_config: str,
) -> "OrderedDict[str, object]":
    """ONE model, ONE machine, two placement laws, side by side (P7.3 item 4).

    Both halves are complete evaluations of the same hardware and the same
    model — the ONLY difference is the packing law — so the deltas below are
    the law's, and nothing else's. The function refuses two evaluations that
    do not meet that condition by name, because a comparison of two different
    machines would answer a question nobody asked.

    Every field is read off the two evaluations. Nothing is recomputed and no
    clock, path or hash enters the document, which is what lets a test
    regenerate it and compare BYTES.
    """
    if dedicated.mapping.packing != "dedicated" or dense.mapping.packing != "dense":
        raise MappingError(
            "assembly",
            "packing_comparison_document takes the DEDICATED evaluation first and the "
            f"DENSE one second (got {dedicated.mapping.packing!r} and "
            f"{dense.mapping.packing!r}). The argument order is the comparison's "
            "meaning, so it is checked rather than assumed.",
        )
    left, right = dedicated.mapping, dense.mapping
    if left.model_id != right.model_id or left.phase != right.phase:
        raise MappingError(
            "assembly",
            f"the two halves run different workloads ({left.model_id}/{left.phase} vs "
            f"{right.model_id}/{right.phase}). Only the packing law may differ.",
        )
    if dedicated.serving != dense.serving:
        raise MappingError(
            "assembly",
            "the two halves serve different points. Only the packing law may differ, "
            "or the step times below compare two workloads — under D29 that includes "
            "the resident stream count D and the beat window, which are part of the "
            "serving point.",
        )
    left_point = _packing_point(dedicated)
    right_point = _packing_point(dense)
    left_step = float(left_point["median_decode_step_s"])
    right_step = float(right_point["median_decode_step_s"])
    accounting = right_point["packing_accounting"]
    delta = OrderedDict(
        (
            (
                "macros_saved",
                int(left_point["macros_holding_tiles"])
                - int(right_point["macros_holding_tiles"]),
            ),
            ("cell_floor_macros", int(accounting["cell_floor_macros"])),
            (
                "macros_above_floor",
                int(right_point["macros_holding_tiles"])
                - int(accounting["cell_floor_macros"]),
            ),
            ("waste_pct", float(accounting["waste_pct"])),
            ("median_decode_step_delta_s", right_step - left_step),
            (
                "median_decode_step_delta_pct",
                (100.0 * (right_step - left_step) / left_step) if left_step else 0.0,
            ),
            (
                "basis",
                "macros_saved is dense against the SAME tensors under the per-tensor "
                "placement, measured on both mappings rather than predicted. "
                "macros_above_floor is what dense packing still costs over the global "
                "cell floor: the floor is a whole-model quantity and this packing is "
                "per chip, so the difference is chip granularity plus the "
                "dimension-mismatch remainder, which waste_pct itemizes. The step "
                "delta is the two timelines' median lowered decode steps, and it is "
                "the accumulator/partial-transport trade (P7 Fact 1), not an analog "
                "speed-up: both laws charge the same column-set passes per step. "
                "Under the filled pipeline (D29) that step IS the beat and the passes "
                "are counted per beat, which is the same statement one level up: every "
                "stored weight is read once per beat and one token leaves.",
            ),
        )
    )
    return OrderedDict(
        (
            ("schema", PACKING_COMPARISON_SCHEMA),
            (
                "note",
                "DECODE ONLY (D25). Dense packing is Invariant W in the mapper (D27): "
                "one contiguous bank stream per chip, with cross-tensor and cross-layer "
                "bank sharing legal. Under the FILLED PIPELINE (D29) the cost of that "
                "sharing is no longer structurally zero: every stage fires on every "
                "beat for a different stream, so a macro holding two STAGES' weights "
                "contends every beat and the timeline prices it, while sharing WITHIN a "
                "stage stays serial-free. Both halves below are full placed-DAG "
                "evaluations, not estimates, and each carries its own bank_sharing "
                "measurement of exactly that split.",
            ),
            (
                "provenance",
                OrderedDict(
                    (
                        ("producer", "fws_eval.packing_comparison_document (QIF P7.3)"),
                        ("hardware_config", hardware_config),
                        ("model_config", model_config),
                        (
                            "regenerate",
                            ".venv/bin/python -m pytest -q tests/test_qif_folding_mapping.py "
                            "-k regenerate  (the test rebuilds this document from the two "
                            "configs above and compares BYTES; run it with "
                            "FWS_WRITE_PACKING_ARTIFACT=1 to rewrite the file)",
                        ),
                        ("invented_fields", []),
                    )
                ),
            ),
            (
                "model",
                OrderedDict(
                    (
                        ("model_id", str(left.model_id)),
                        ("phase", str(left.phase)),
                        ("num_layers", int(left.device.params.num_layers)),
                    )
                ),
            ),
            (
                "serving",
                OrderedDict(
                    (
                        ("batch", int(dedicated.serving.batch)),
                        ("prefill_len", int(dedicated.serving.prefill_len)),
                        ("decode_len", int(dedicated.serving.decode_len)),
                        ("decode_steps_lowered", int(dedicated.serving.decode_steps)),
                    )
                ),
            ),
            ("points", [left_point, right_point]),
            ("delta", delta),
            (
                "disclosures",
                merge_disclosures(
                    tuple(dedicated.disclosures) + tuple(dense.disclosures)
                ),
            ),
        )
    )


def merge_disclosures(items: Sequence[Relaxation]) -> List["OrderedDict[str, object]"]:
    """One entry per CONSTRAINT KEY, with every producer's reason kept.

    P3 and P4 can both relax the same constraint — ``intra_chip_activation_
    movement`` is declared generically by the mapping and again with a measured
    byte count by the pricer. Listing it twice makes a reader work out that two
    entries are one claim, which is the same failure as two accountings of one
    metric (D21). The entries are MERGED, never dropped: the first value wins
    (it is the one the artifact's other fields agree with) and every distinct
    reason survives, so nothing a producer said is lost.
    """
    order: List[str] = []
    values: Dict[str, object] = {}
    reasons: Dict[str, List[str]] = {}
    for item in items:
        key = str(item.constraint)
        if key not in values:
            order.append(key)
            values[key] = item.value
            reasons[key] = []
        if item.reason not in reasons[key]:
            reasons[key].append(str(item.reason))
    out: List["OrderedDict[str, object]"] = []
    for key in order:
        parts = reasons[key]
        reason = parts[0]
        for extra in parts[1:]:
            reason += (
                " ALSO DISCLOSED under this same constraint by a second producer: "
                + extra
            )
        out.append(
            OrderedDict(
                (("constraint", key), ("value", values[key]), ("reason", reason))
            )
        )
    return out


def render_report(document: Mapping[str, object]) -> List[str]:
    """The readable results section, rendered FROM the JSON document.

    Not a second computation and not a second accounting: every line below
    reads a field of ``document``. If a number is not in the JSON, it is not in
    the text (P4 §8).
    """
    evaluation = document["evaluation"]
    serving = document["serving"]
    mapping = document["mapping"]
    lines = [
        "",
        "==============================================",
        "FWS-CIM placed-DAG evaluation (QIF P4)",
        "==============================================",
        str(evaluation["single_accounting"]),
        "",
        (
            (
                f"Serving (D29):          FILLED PIPELINE, "
                f"D = {serving['resident_streams']} resident streams (= stages), local "
                f"batch {serving['batch']}, {serving['beats_lowered']} beats lowered "
                f"({serving['steady_beats']} steady); prefill is external"
            )
            if str(serving.get("regime")) == REGIME_FILLED
            else (
                f"Serving (D15, RETIRED): batch {serving['batch']}, prefill "
                f"{serving['prefill_len']} tokens, decode {serving['decode_len']} steps "
                f"({serving['decode_steps_lowered']} lowered)"
            )
        ),
        (
            f"Placement:              {mapping['analog_chips']} analog chiplets, "
            f"{mapping['shared_digital_chiplets']} shared digital, "
            f"{mapping['macros_holding_tiles']} owned macros, {mapping['tiles']} tiles"
        ),
        f"Ops priced:             {evaluation['ops_priced']}",
        f"Makespan:               {float(evaluation['makespan_s']) * 1e3:.6f} ms",
        "",
        "Metrics (every one a projection of the one timeline):",
    ]
    for metric in evaluation["metrics"]:
        value = metric["value"]
        rendered = f"{value:.6g}" if isinstance(value, float) else str(value)
        lines.append(f"  {metric['label']:<52} {rendered:>16} {metric['unit']}")
        lines.append(f"      basis: {metric['basis']}")
    residency = evaluation.get("state_residency") or {}
    if residency.get("measured"):
        # THE GEORGE CONSTRAINT, in the headline block: every stage holds all D
        # streams' state and KV for its layers, and the verdict is named.
        lines.extend(
            [
                "",
                (
                    f"Resident state (D29):   {float(residency['max_stage_state_bytes']) / 2 ** 20:.3f} MiB "
                    f"on the worst of {residency['stages']} stages, "
                    f"{float(residency['total_state_bytes']) / 2 ** 20:.3f} MiB total for "
                    f"{residency['resident_streams']} streams "
                    f"({float(residency['per_stream_model_state_bytes']) / 2 ** 20:.3f} MiB "
                    f"per stream) — verdict {residency['verdict']}"
                    + (
                        ""
                        if not residency["stages_violating"]
                        else f", VIOLATED on stage(s) {residency['stages_violating']}"
                    )
                ),
                (
                    f"  at a representative context of "
                    f"{residency['representative_context']:.6g} tokens (median of the "
                    f"window; the streams span {residency['context_min']:.6g} to "
                    f"{residency['context_max']:.6g} and each is priced at its own)"
                ),
                f"  basis: {residency['basis']}",
            ]
        )
    pipeline = evaluation.get("pipeline") or {}
    if pipeline:
        rendered = ", ".join(
            f"{value * 1e6:.3f}" for value in pipeline["beat_intervals_s"]
        )
        lines.extend(
            [
                "",
                (
                    f"Beat series (us, {pipeline['tokens_exited']} token exits over "
                    f"{pipeline['beats_lowered']} beats): {rendered}"
                ),
                (
                    f"  the first {len(pipeline['transient_beat_intervals_s'])} is the "
                    f"FILL TRANSIENT and is held out of the beat; "
                    f"{pipeline['fill_beats']} beats of fill precede the steady window"
                ),
                f"  basis: {pipeline['beat_basis']}",
            ]
        )
    series = evaluation["decode_series"]
    if series["lowered_steps"] and not pipeline:
        rendered = ", ".join(f"{value * 1e6:.3f}" for value in series["step_times_s"])
        lines.extend(
            [
                "",
                f"Decode step series (us, {series['lowered_steps']} lowered steps): {rendered}",
                f"  basis: {series['basis']}",
            ]
        )
    extrapolation = evaluation["extrapolation"]
    if extrapolation.get("extrapolated"):
        lines.extend(
            [
                "",
                (
                    (
                        "EXTRAPOLATION (not a metric): a REQUEST decodes "
                        f"{extrapolation['declared_decode_steps']} tokens; the window "
                        f"measured {extrapolation['lowered_decode_steps']} inter-exit "
                        f"intervals of the machine that serves it "
                        f"({pipeline['steady_beats_measured']} steady, the rest the fill "
                        "transient)."
                    )
                    if pipeline
                    else (
                        f"EXTRAPOLATION (not a metric): {extrapolation['remaining_steps']} "
                        f"of {extrapolation['declared_decode_steps']} decode steps were "
                        "NOT lowered."
                    )
                ),
                (
                    (
                        f"  beat {float(extrapolation['median_step_s']) * 1e6:.3f} us x D "
                        f"x {extrapolation['declared_decode_steps']} tokens -> "
                        "extrapolated request decode latency "
                        f"{float(extrapolation['extrapolated_request_latency_s']) * 1e3:.6f} ms"
                    )
                    if pipeline
                    else (
                        f"  median lowered step "
                        f"{float(extrapolation['median_step_s']) * 1e6:.3f} us "
                        f"-> extrapolated request latency "
                        f"{float(extrapolation['extrapolated_request_latency_s']) * 1e3:.6f} ms"
                    )
                ),
                f"  {extrapolation['basis']}",
            ]
        )
    energy = evaluation["energy"]
    lines.extend(["", "Energy, itemized (each component carries its own coverage label):"])
    for component in energy["components"]:
        lines.append(
            f"  {component['label']:<58} {component['energy_pj'] * 1e-6:>14.6f} uJ  "
            f"[{component['coverage'].upper()}]"
        )
    lines.append(f"  {'TOTAL (sum of the parts above)':<58} {float(energy['total_pj']) * 1e-6:>14.6f} uJ")
    lines.append(f"  basis: {energy['total_basis']}")

    lines.extend(["", "Memory feasibility (high-water replay vs declared capacities):"])
    for verdict in evaluation["memory"]:
        lines.append(
            f"  {verdict['scope']:<12} {verdict['tier']:<20} "
            f"{verdict['high_water_bytes'] / 1024 ** 3:>10.3f} GiB / "
            f"{verdict['capacity_bytes'] / 1024 ** 3:>10.3f} GiB  {verdict['status']}"
        )
        if verdict["disclosure"]:
            lines.append(f"      {verdict['disclosure']}")

    occupancy = evaluation["occupancy"]
    if occupancy:
        busiest = sorted(occupancy, key=lambda row: -row["occupancy"])[:5]
        lines.extend(["", "Busiest devices (occupancy = busy union / makespan, owner-attributed):"])
        for row in busiest:
            lines.append(
                f"  device {row['device']:<5} {row['device_class']:<15} "
                f"{row['occupancy'] * 100:>7.3f}%  {row['ops']:>5} ops  owner: {row['owner'][:64]}"
            )
        lines.append(f"  basis: {evaluation['occupancy_basis']}")

    utilization = evaluation.get("utilization") or []
    if utilization:
        lines.extend(
            [
                "",
                "Utilization by device class (D28 — idle silicon is a finding, not a footnote):",
                f"  {'class':<16}{'devices':>9}{'idle':>7}{'ops':>9}"
                f"{'mean':>10}{'median':>10}{'peak':>10}",
            ]
        )
        for row in utilization:
            lines.append(
                f"  {row['device_class']:<16}{row['devices']:>9}{row['idle_devices']:>7}"
                f"{row['ops']:>9}{row['mean_occupancy'] * 100:>9.3f}%"
                f"{row['median_occupancy'] * 100:>9.3f}%{row['max_occupancy'] * 100:>9.3f}%"
            )
        lines.append(f"  basis: {evaluation['utilization_basis']}")
        link = [row for row in utilization if row["device_class"] == "link"]
        if link:
            lines.append(f"  link row: {link[0]['basis']}")

    packing = evaluation.get("packing") or {}
    if packing:
        lines.extend(["", "Packing (Invariant W, D27):"])
        if packing.get("waste_reported"):
            lines.append(
                f"  {packing['packing']} ({packing['walk']} walk): {packing['macros']} macros "
                f"vs {packing['dedicated_macros']} dedicated ({packing['macros_saved']} saved), "
                f"global cell floor {packing['cell_floor_macros']}"
            )
            lines.append(
                f"  waste {packing['waste_pct']:.3f}% = remainder {packing['remainder_cells']} "
                f"+ tail {packing['tail_cells']} cells of {packing['committed_cells']} committed"
            )
            lines.append(
                f"  K stacks: {packing['local_k_stacks']} in-macro (accumulator priced), "
                f"{packing['spread_k_stacks']} spread (row-block law)"
            )
        else:
            lines.append(f"  {packing['packing']}: no packer ran, so no waste is reported")
        lines.append(f"  basis: {packing['basis']}")

    bank_passes = evaluation.get("bank_passes") or {}
    if bank_passes.get("measured"):
        lines.extend(
            [
                "",
                (
                    f"Bank passes in decode step {bank_passes['decode_step']} (P7 Fact 1 — "
                    "each macro walks its occupied banks once):"
                ),
                (
                    f"  {bank_passes['macros_walked_exactly_once']} of "
                    f"{bank_passes['macros_holding_tiles']} macros walked exactly once; "
                    f"{bank_passes['column_set_passes_charged']:.0f} column-set passes "
                    f"charged against {bank_passes['column_sets_occupied']:.0f} occupied"
                ),
                (
                    f"  never read {bank_passes['column_sets_never_read']:.0f}, "
                    f"read again {bank_passes['column_sets_read_more_than_once']:.0f}"
                ),
                f"  basis: {bank_passes['basis']}",
            ]
        )

    sharing = evaluation.get("bank_sharing") or {}
    if sharing.get("measured"):
        lines.extend(
            [
                "",
                (
                    "Bank sharing in decode step "
                    f"{sharing['decode_step']} (P7 Fact 3 — what co-residency costs):"
                ),
                (
                    f"  {sharing['macros_sharing_banks_across_tensors']} macros hold "
                    f"more than one tensor, {sharing['macros_sharing_banks_across_layers']}"
                    " hold more than one layer"
                ),
                (
                    f"  delay from ANOTHER LAYER's residents "
                    f"{sharing['cross_layer_delay_s'] * 1e6:.3f} us; from the SAME "
                    f"layer's other tensors "
                    f"{sharing['cross_tensor_same_layer_delay_s'] * 1e6:.3f} us; from "
                    f"the tensor's own blocks "
                    f"{sharing['same_owner_delay_s'] * 1e6:.3f} us"
                ),
                f"  basis: {sharing['basis']}",
            ]
        )

    pool = evaluation["digital_pool"]
    lines.extend(["", "Derived per-macro digital pool (D12, sized FROM the timeline):"])
    for row in pool["sizings"]:
        lines.append(
            f"  peak concurrent demand {row['peak_concurrent_demand']} on {row['macros']} macro(s): "
            f"{row['lanes']} lanes, {row['area_mm2']:.6g} mm2 "
            f"({row['area_share'] * 100:.1f}% of the macro footprint)"
        )
        lines.append(f"      {row['report']}")
    lines.append(f"  basis: {pool['basis']}")

    lines.extend(["", "Disclosures (D21 — these are banners, not footnotes):"])
    for item in document["disclosures"]:
        lines.append(f"  [{item['constraint']}] {item['value']}")
        lines.append(f"      {item['reason']}")
    lines.append("")
    return lines
