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

"""L4 — ``build()``: compose L1 + L2 + L3 into a :class:`Program` (INTERFACES §4).

**Deps are COMPUTED, never walked off a mutable graph.** There are exactly four
dependency classes (:class:`~program.work.DepClass`) and four rules, applied in
this order (INTERFACES §4.3):

===  =========================================================================
R1   data flow inside one ``ExpandedChain`` (``DepClass.DATA_FLOW``)
R2   cross-layer ``L -> L+1``: same device => dep, different device => one
     ``TransferOp`` (``DepClass.DATA_FLOW``)
R3   device serialization implied by the ``SchedulePolicy``
     (``DepClass.SCHEDULE``) — added ONLY when not already transitively
     implied (invariant **D1**)
R4   sync attach from each ``SyncRequirement``'s declared ``AttachMode``
     (``DepClass.SYNC``), resolved in a declared :class:`SyncOrder`
===  =========================================================================

What this module does NOT contain, and must never grow:

* **no proto graph** — there is one representation, the ops built here;
* **no clone cache** — a WorkItem is expanded once per device by
  :class:`~program.placement.BlockExpander`, and nothing is copied;
* **no adjacency walking** — no ``for child in node.children`` over a mutable
  object graph; every edge is added by a rule that names both endpoints;
* **no name dispatch** — not one ``"optimizer" in name`` / ``"bwd" in name`` /
  ``"attention" in name`` test. Every decision reads a typed field of a
  ``WorkItem``, a ``CommSpec`` or a policy object (rule §8.2). The ONE
  string comparison left is :class:`~program.policies.overlap.OverlapDecl`'s
  ``blocking_consumer`` against ``ComputeStep.entry_name`` — and that string is
  DATA on the policy, not a literal in this file.

**Program order** (INTERFACES §4.6) is a documented deterministic function

    key(op) = (slot_index, device_index, intra_index)

with ``intra_index = (phase, a, b)`` — see :meth:`_Builder._order` — and uids
are assigned by Kahn's algorithm over the complete dep graph with a min-heap on
that key. Keys are unique by construction, so the order is total, is
schedule-major (AstraSim node ids are scheduling priorities — CONTEXT.md), and
is a pure function of the inputs (**O1**).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from timing_model import CollectiveType

from program.groups import canonical_axis_label
from program.ir import (
    CollectiveOp,
    CommGroup,
    ComputeOp,
    GroupKey,
    Op,
    OpRole,
    Program,
    ProgramMeta,
    TransferOp,
)
from program.ir import Direction as IRDirection
from program.placement import (
    LEGACY_PLACEMENT,
    BlockExpander,
    CommStep,
    ComputeStep,
    ExpandedChain,
    Granularity,
    Placement,
    PlacementPolicy,
    StepKind,
)
from program.policies.gradaccum import GradAccumPolicy, grad_accum_policy_for
from program.policies.overlap import (
    AxisFractionOverlap,
    OverlapAnchor,
    OverlapDecl,
    OverlapPolicy,
)
from program.policies.recompute import RecomputePolicy
from program.policies.routing import MoERoutingPolicy, ep_sync_requirements
from program.policies.sharding import ShardingContext, ShardingPolicy
from program.schedule.policy import Schedule, SchedulePolicy
from program.types import CommKey, DeviceId
from program.work import (
    AttachMode,
    ByteSource,
    ByteSplit,
    DepClass,
    Direction,
    SyncAnchor,
    SyncKey,
    SyncPhase,
    SyncRequirement,
    WorkItem,
    WorkKind,
    WorkSet,
    enumerate_work,
)
from program.workload import FrozenWorkload

__all__ = [
    "BuildError",
    "SyncOrder",
    "build",
    "check_granularity_preconditions",
    "restrict_work_for",
]


class BuildError(ValueError):
    """A Program could not be composed from the given policies."""


#: The cross-layer activation transfer. THE only comm key ``build()`` names, and
#: it names it once: R2 is the rule that owns pipeline data movement
#: (``train_timing._build_comm_metadata`` registers it at ``:4429-4435``).
CROSS_LAYER_KEY: CommKey = "cross_layer"


# ---------------------------------------------------------------------------
# SyncOrder (INTERFACES §4.4)
# ---------------------------------------------------------------------------

#: FWD_ENTRY < FWD < GRAD < BWD_ENTRY < BWD (INTERFACES §4.4).
_PHASE_RANK: Mapping[SyncPhase, int] = {
    SyncPhase.FWD_ENTRY: 0,
    SyncPhase.FWD: 1,
    SyncPhase.GRAD: 2,
    SyncPhase.BWD_ENTRY: 3,
    SyncPhase.BWD: 4,
}


@dataclass(frozen=True)
class SyncOrder:
    """The total order in which :class:`SyncRequirement`s are resolved.

    A NAMED POLICY, not an accident of construction order. ``attach_parallel_edge``
    reads the *current* parents/children of a mutable object, so today the result
    depends on which lattice block ran first (the forward ZeRO-3 block at
    ``schedule.py:848-892`` runs before the backward GPipe wiring at ``:955-976``;
    the backward ZeRO-3 block at ``:978-1046`` runs after). That is not
    reproducible from a declaration set, so it is replaced by this declared order
    (INTERFACES §4.4).
    """

    name: str = "phase_major"

    def key(self, req: SyncRequirement) -> Tuple[Any, ...]:
        key = req.key
        return (
            _PHASE_RANK[key.phase],
            -1 if key.microbatch is None else int(key.microbatch),
            -1 if key.layer is None else int(key.layer),
            str(key.comm_key),
        )


# ---------------------------------------------------------------------------
# Proto nodes — the pre-uid form. One representation; these become ops 1:1.
# ---------------------------------------------------------------------------


#: The order key: ``(slot_index, device_index, phase, a, b)``. ``phase`` orders
#: the four construction rules within one (slot, device); ``a``/``b`` make each
#: rule's own emission deterministic. All five components are ints, so the tuple
#: is totally ordered and unique by construction (INTERFACES §4.6).
_OrderKey = Tuple[int, int, int, int, int]

_PHASE_CHAIN = 0  #: R1 steps (``a`` = chain position; ``b`` = split sub-index)
_PHASE_XFER = 1  #: R2 transfers (``a`` = R2 emission sequence)
_PHASE_SYNC = 2  #: R4 sync ops (``a`` = SyncOrder rank, ``b`` = instance index)


@dataclass
class _ProtoCompute:
    nid: int
    order: _OrderKey
    device: int
    name: str
    duration: float
    work: Optional[WorkItem]
    role: OpRole
    direction: IRDirection
    mem_kind: Any = None
    recompute: bool = False
    param_gather: bool = False
    micro_batch: Optional[int] = None
    layer: Optional[int] = None
    is_moe_layer: bool = False
    #: the ``BlockTemplate`` entry this compute came from (``None`` for a
    #: single-op expansion). A TYPED template field, which is what
    #: ``OverlapDecl.blocking_consumer`` is matched against — never the op name.
    entry_name: Optional[str] = None
    deps: List[int] = field(default_factory=list)
    succs: List[int] = field(default_factory=list)


@dataclass
class _ProtoCollective:
    nid: int
    order: _OrderKey
    device: int
    name: str
    coll: CollectiveType
    size_bytes: float
    participants: int
    axes: Tuple[str, ...]
    is_dp: bool
    group: Optional[GroupKey]
    work: Optional[WorkItem]
    comm_key: Optional[CommKey] = None
    deps: List[int] = field(default_factory=list)
    succs: List[int] = field(default_factory=list)


@dataclass
class _ProtoTransfer:
    nid: int
    order: _OrderKey
    device: int  #: the SRC device — a transfer is ordered where it is issued
    name: str
    src_device: int
    dst_device: int
    size_bytes: float
    comm_type: Optional[CollectiveType]
    producer: int
    consumers: List[int] = field(default_factory=list)
    moe_component: Optional[str] = None
    #: the declaring spec's ANALYTICAL timing surface (INTERFACES §4.7
    #: amendment 2026-07-28): participant count + interconnect axis key.
    participants: int = 0
    interconnect: Optional[str] = None
    deps: List[int] = field(default_factory=list)
    succs: List[int] = field(default_factory=list)


_ProtoOp = Union[_ProtoCompute, _ProtoCollective, _ProtoTransfer]


#: The WorkKinds a BLOCK build can express. BLOCK's device space is
#: ``layout.subset(("tp","cp","ep"))`` — it carries NO ``pp`` and no ``dp`` axis —
#: so pipeline work is not representable there at all: INTERFACES §3.1 defines
#: BLOCK as "one layer expanded over the (tp,cp,ep) sublayout only, no pipeline",
#: and the builder it replaces (``block_program.build_block_program``) takes ONE
#: ``BlockTemplate`` and ONE direction and nothing else.
_BLOCK_KINDS: Tuple[WorkKind, ...] = (WorkKind.LAYER, WorkKind.RECOMPUTE)

#: The work kinds whose BACKWARD direction produces a weight gradient, i.e. what
#: **R5** makes the optimizer wait for. Same set as
#: ``schedule.policy._GRADIENT_KINDS`` (which uses it to answer "which microbatch
#: finishes last"); named here so R5 does not reach into L3's private table.
_R5_GRADIENT_KINDS: Tuple[WorkKind, ...] = (
    WorkKind.LAYER,
    WorkKind.EMBEDDING,
    WorkKind.SOFTMAX,
)


def restrict_work_for(
    granularity: Granularity,
    work: WorkSet,
    *,
    directions: Optional[Sequence[Direction]] = None,
) -> WorkSet:
    """The work a granularity (and caller) can express (INTERFACES §3.1).

    COARSE and FINE express the whole workload. BLOCK expresses transformer-block
    work only — see :data:`_BLOCK_KINDS`. Stated as one named function so the
    coupling is visible in a dumped program
    (``meta.misc["work_restricted"]``) instead of being an accident of which
    dispatcher built the spec.

    ``directions`` is the AMENDMENT 2026-07-28 that makes §4.2's own consequence
    expressible: "``build()`` at BLOCK produces ONE program containing both
    directions ... a caller measuring a single direction builds a single
    direction's workload". AstraSim returns ONE makespan per bundle, so the
    hybrid/hierarchical retiming — which needs a forward time AND a backward time
    — genuinely needs two bundles. ``None`` means "every direction".
    """
    items = work.items
    if granularity is Granularity.BLOCK:
        items = tuple(item for item in items if item.kind in _BLOCK_KINDS)
    if directions is not None:
        allowed = tuple(directions)
        items = tuple(item for item in items if item.direction in allowed)
    if items is work.items:
        return work
    return WorkSet(items=items)


def check_granularity_preconditions(granularity: Granularity, fw: FrozenWorkload) -> None:
    """What a granularity REQUIRES of its workload (INTERFACES §3.1).

    BLOCK's device space is ``layout.subset(("tp","cp","ep"))``: it carries
    neither ``pp`` nor ``dp``, so a BLOCK workload must declare neither. This is
    the legacy ``dp_override=1`` semantics stated as a precondition instead of
    left implicit — ``block_program.build_block_program`` takes one template and
    emits with ``dp_count=1`` because a transformer-block run is a SINGLE-REPLICA
    measurement. Enforcing it here is what makes the data-parallel policy
    selection need no granularity special case at all: at ``dp == 1``
    ``sharding_policy_for`` answers ``NullSharding`` and ``GradAccumPolicy.emits``
    is unconditionally False, so neither the ZeRO lattice nor the EP grad sync
    can leak into a block measurement.
    """
    if granularity is not Granularity.BLOCK:
        return
    degrees = fw.spec.degrees
    if int(degrees.dp) != 1 or int(degrees.pp) != 1:
        raise BuildError(
            "Granularity.BLOCK requires a single-replica, single-stage workload "
            f"(got dp={degrees.dp}, pp={degrees.pp}). BLOCK's device space is the "
            "(tp,cp,ep) sublayout, which carries no dp and no pp axis; a block "
            "run is a single-replica measurement (legacy dp_override=1). Build "
            "the block spec with dp=pp=1, or use Granularity.FINE."
        )


#: WorkKind -> the semantic role of a single-op expansion. Keyed on the enum,
#: never on a name (this is what deletes ``pipeline_fine.py:417`` /``:426``).
_ROLE_BY_KIND: Mapping[WorkKind, OpRole] = {
    WorkKind.EMBEDDING: OpRole.EMBEDDING,
    WorkKind.SOFTMAX: OpRole.SOFTMAX,
    WorkKind.LAYER: OpRole.TRANSFORMER_LAYER,
    WorkKind.RECOMPUTE: OpRole.TRANSFORMER_LAYER,
    WorkKind.OPTIMIZER: OpRole.OPTIMIZER,
}


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------


def build(
    fw: FrozenWorkload,
    *,
    granularity: Granularity,
    sharding: ShardingPolicy,
    schedule_policy: SchedulePolicy,
    recompute: RecomputePolicy,
    routing: Optional[MoERoutingPolicy] = None,
    overlap: OverlapPolicy = AxisFractionOverlap(),
    grad_accum: Optional[GradAccumPolicy] = None,
    dp_count: Optional[int] = None,
    label: str = "",
    gmap_workdir: Optional[str] = None,
    placement_policy: PlacementPolicy = LEGACY_PLACEMENT,
    sync_order: SyncOrder = SyncOrder(),
    validate: bool = True,
    check_group_membership: bool = False,
    directions: Optional[Sequence[Direction]] = None,
) -> Program:
    """Compose L1 + L2 + L3 into a :class:`~program.ir.Program`.

    Mechanical: no policy decision of its own. Phases, in this exact order
    (INTERFACES §4.2):

    1. ``work      = enumerate_work(fw, recompute)``                    (L1)
    2. ``schedule  = schedule_policy.schedule(fw, work)``               (L3)
    3. ``placement = Placement(fw, granularity, schedule.layers)``      (L2)
    4. ``chains    = BlockExpander(...).expand(item) for item in work`` (L2)
    5. ``deps      = R1 + R2``                                          (§4.3)
    6. ``deps     += R3`` from ``schedule.implied_deps``                (§4.3)
    7. ``reqs      = sharding + routing requirements``                  (L1)
    8. ``deps     += R4``, materializing the sync ops                   (§4.4)
    9. realize every ``OverlapDecl``                                    (§4.5)
    10. program order (§4.6); assign uids; build Program; validate

    ``grad_accum`` defaults to ``grad_accum_policy_for(fw)`` rebound to the
    schedule (:meth:`GradAccumPolicy.for_schedule` — Class B 10h is data, not a
    hardcoded ``0``). ``dp_count`` defaults to ``run.retime_dp_count(degrees)``.

    ``gmap_workdir`` is RECORDED in ``meta.misc`` for the emission step; this
    function writes no files. The first-dimension SCOTCH collection is an
    emission-side concern (P5 wires it), and silently doing nothing with a path
    would be worse than saying so.

    ``check_group_membership`` enables invariant **V7** and DEFAULTS TO OFF.
    V7 says every member device of a communicator issues the group's
    collectives, which used to contradict BUG_LEDGER **A2**
    (``SyncSpread.CLUSTER_RANK_0``: exactly ONE instance of a stage-spanning
    collective, on cluster rank 0) whenever a non-dp requirement's group spanned
    more than one device — e.g. the ``ep`` grad sync at ``tp*cp*ep > 1``.

    **A2 is fixed** (final wave): ``policies.sharding._spread_for`` returns
    ``PER_CLUSTER_RANK`` for every dp requirement and row S12 declares it too,
    so no production requirement puts a grouped collective on one member of its
    group any more, and ``build(..., check_group_membership=True)`` was measured
    clean over 192 COARSE+FINE configurations (``dp/tp/cp/ep/pp`` in ``{1,2}`` x
    ``zero_stage`` in ``{0,2,3}`` x dense/MoE). The default stays OFF only
    because legacy-lowered / hand-built Programs are per-device clone programs
    for which V7 is false BY CONSTRUCTION (``program/ir.py`` module docstring);
    flipping it is a separate, now-unblocked decision.
    """
    builder = _Builder(
        fw=fw,
        granularity=granularity,
        sharding=sharding,
        schedule_policy=schedule_policy,
        recompute=recompute,
        routing=routing,
        overlap=overlap,
        grad_accum=grad_accum,
        dp_count=dp_count,
        label=label,
        gmap_workdir=gmap_workdir,
        placement_policy=placement_policy,
        sync_order=sync_order,
        directions=directions,
    )
    return builder.run(
        validate=validate, check_group_membership=check_group_membership
    )


class _Builder:
    """One build. Every phase is a method, called once, in §4.2's order."""

    def __init__(
        self,
        *,
        fw: FrozenWorkload,
        granularity: Granularity,
        sharding: ShardingPolicy,
        schedule_policy: SchedulePolicy,
        recompute: RecomputePolicy,
        routing: Optional[MoERoutingPolicy],
        overlap: OverlapPolicy,
        grad_accum: Optional[GradAccumPolicy],
        dp_count: Optional[int],
        label: str,
        gmap_workdir: Optional[str],
        placement_policy: PlacementPolicy,
        sync_order: SyncOrder,
        directions: Optional[Sequence[Direction]] = None,
    ) -> None:
        self._fw = fw
        self._granularity = granularity
        self._sharding = sharding
        self._schedule_policy = schedule_policy
        self._recompute = recompute
        self._routing = routing
        self._overlap = overlap
        self._grad_accum_in = grad_accum
        self._dp_count_in = dp_count
        self._label = label
        self._gmap_workdir = gmap_workdir
        self._placement_policy = placement_policy
        self._sync_order = sync_order
        self._directions = directions

        self._nodes: List[_ProtoOp] = []
        #: ``(u, v) -> the DepClasses that justify the edge``. An edge exists
        #: once; a second rule adding it contributes its class, not a duplicate.
        self._edges: Dict[Tuple[int, int], Set[DepClass]] = {}
        #: ``work -> chains``, and ``(work, device) -> the chain's node ids``.
        self._chains: Dict[WorkItem, Tuple[ExpandedChain, ...]] = {}
        self._chain_nodes: Dict[Tuple[WorkItem, int], List[int]] = {}
        self._chain_by_device: Dict[Tuple[WorkItem, int], ExpandedChain] = {}
        #: ``SyncKey -> the node ids the requirement materialized on``.
        self._sync_nodes: Dict[SyncKey, List[int]] = {}
        #: overlap work list: ``(collective nid, decl, producing compute nid)``.
        self._overlap_sites: List[Tuple[int, OverlapDecl, Optional[int]]] = []
        self._xfer_seq = 0
        self._cross_layer_bytes: Optional[float] = None
        self._dropped_requirements: List[str] = []

    # ==================================================================
    # phases
    # ==================================================================
    def run(self, *, validate: bool, check_group_membership: bool = False) -> Program:
        fw = self._fw
        # 1 / 2 / 3 -------------------------------------------------------
        check_granularity_preconditions(self._granularity, fw)
        full_work = enumerate_work(fw, self._recompute)
        self._work: WorkSet = restrict_work_for(
            self._granularity, full_work, directions=self._directions
        )
        self._work_restricted = len(self._work) != len(full_work)
        self._schedule: Schedule = self._schedule_policy.schedule(fw, self._work)
        self._schedule.check_permutation(self._work)
        self._placement = Placement(
            fw, self._granularity, self._schedule.layers, policy=self._placement_policy
        )
        self._devices: Tuple[DeviceId, ...] = self._placement.devices()
        self._device_pos: Dict[int, int] = {
            int(device): index for index, device in enumerate(self._devices)
        }
        self._grad_accum = (
            self._grad_accum_in
            if self._grad_accum_in is not None
            else grad_accum_policy_for(fw)
        ).for_schedule(self._schedule)

        # 4 / 5 -----------------------------------------------------------
        self._expand_all()
        self._apply_r2()
        # 6 ---------------------------------------------------------------
        self._apply_r3()
        # 7 / 8 -----------------------------------------------------------
        self._apply_r4(self._collect_requirements())
        # 8b ---------------------------------------------------------------
        self._apply_r5()
        # 9 ---------------------------------------------------------------
        self._realize_overlap()
        # 10 --------------------------------------------------------------
        return self._finish(
            validate=validate, check_group_membership=check_group_membership
        )

    # ------------------------------------------------------------------
    # node / edge primitives
    # ------------------------------------------------------------------
    def _order(
        self, work: Optional[WorkItem], device: int, phase: int, a: int, b: int = 0
    ) -> _OrderKey:
        """``(slot_index, device_index, phase, a, b)`` — INTERFACES §4.6.

        ``slot_index`` is ``Schedule.index_of`` of the OWNING WorkItem: a sync op
        inherits its ``place_on`` item's index, a ``TransferOp`` its producer's,
        a split op its source's. ``device_index`` is the position of the device
        in ``placement.devices()``.
        """
        slot = -1 if work is None else self._schedule.index_of(work)
        position = self._device_pos.get(int(device))
        if position is None:
            raise BuildError(
                f"device {device} is not one of placement.devices() {self._devices}"
            )
        return (slot, position, phase, a, b)

    def _add(self, node: _ProtoOp) -> int:
        self._nodes.append(node)
        return node.nid

    def _next_nid(self) -> int:
        return len(self._nodes)

    def _add_dep(self, dep: int, node: int, dep_class: DepClass) -> None:
        """Add ``dep -> node`` with ``dep_class``; idempotent per (edge, class).

        Idempotence is what reproduces ``attach_parallel_edge``'s ``not in``
        guards (``schedule.py:556,566``) without a membership scan over a
        children list.
        """
        if dep == node:
            raise BuildError(f"self-dependency on node {dep}")
        key = (dep, node)
        classes = self._edges.get(key)
        if classes is None:
            self._edges[key] = {dep_class}
            self._nodes[node].deps.append(dep)
            self._nodes[dep].succs.append(node)
        else:
            classes.add(dep_class)

    def _classes(self, dep: int, node: int) -> Set[DepClass]:
        """The classes carried by ``dep -> node``; ``{DATA_FLOW}`` if absent."""
        return set(self._edges.get((dep, node)) or {DepClass.DATA_FLOW})

    def _reparent_dep(self, dep: int, node: int, *, like: Tuple[int, int]) -> None:
        """Add ``dep -> node`` carrying the classes of the edge ``like``.

        Overlap realization (§4.5) MOVES edges; it does not invent them. Adding
        the moved edge as ``DATA_FLOW`` unconditionally erased the original
        class, which matters because ``PARALLEL_TO``'s ``via`` filter dispatches
        on it (§2.2) and because ``meta.misc["schedule_edges"]`` is derived from
        it.
        """
        for dep_class in self._classes(*like):
            self._add_dep(dep, node, dep_class)

    def _drop_dep(self, dep: int, node: int) -> None:
        """Remove ``dep -> node`` entirely. Used ONLY by overlap realization
        (§4.5), which re-parents rather than adds."""
        key = (dep, node)
        if key not in self._edges:
            return
        del self._edges[key]
        self._nodes[node].deps.remove(dep)
        self._nodes[dep].succs.remove(node)

    def _reaches(self, source: int, target: int) -> bool:
        """Transitive reachability ``source -> target`` in the graph so far.

        Backward BFS from ``target`` over ``deps`` (which carries every edge
        class, including the producer/consumer wiring of a ``TransferOp``), so
        R3 sees exactly the graph INTERFACES §4.3 describes: R1 + R2 + the R3
        edges already added. Early-exits on a hit; the "not implied" answer costs
        one ancestor-set walk, and R3 asks ``O(devices x slots)`` questions.
        """
        if source == target:
            return True
        stack = [target]
        seen: Set[int] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            for dep in self._nodes[current].deps:
                if dep == source:
                    return True
                if dep not in seen:
                    stack.append(dep)
        return False

    # ------------------------------------------------------------------
    # phase 4/5 — expansion (R1)
    # ------------------------------------------------------------------
    def _expand_all(self) -> None:
        """L2 expansion + **R1**: within one chain, ``steps[i]`` depends on the
        local indices ``steps[i].deps`` declare (the chain shape comes from the
        ``BlockTemplate`` and HONORS ``CommSpec.placement`` — INTERFACES §3.4)."""
        expander = BlockExpander(
            self._fw, self._placement, overlap=self._overlap, routing=self._routing
        )
        for slot in self._schedule.ordered_slots:
            work = slot.work
            chains = expander.expand(work)
            self._chains[work] = chains
            for chain in chains:
                device = int(chain.device)
                if (work, device) in self._chain_nodes:
                    raise BuildError(
                        f"{work!r} expanded twice onto device {device}; "
                        "Placement.devices_for must be injective on "
                        "(WorkItem, cluster_rank) (P5)"
                    )
                self._chain_by_device[(work, device)] = chain
                self._chain_nodes[(work, device)] = self._materialize_chain(work, chain)

    def _materialize_chain(self, work: WorkItem, chain: ExpandedChain) -> List[int]:
        device = int(chain.device)
        nids: List[int] = []
        for step in chain.steps:
            order = self._order(work, device, _PHASE_CHAIN, step.index)
            if isinstance(step, ComputeStep):
                nids.append(self._add_compute_step(work, chain, step, order))
            elif isinstance(step, CommStep):
                # ``nids`` is the IN-PROGRESS prefix of this chain: a comm step's
                # local dep indices always point BEHIND it, so the prefix is
                # enough and no second pass is needed.
                nids.append(self._add_comm_step(work, chain, step, order, nids))
            else:  # pragma: no cover - ChainStep is a closed union
                raise BuildError(f"Unsupported chain step {step!r}")
        # R1: local dep indices -> node ids.
        for step, nid in zip(chain.steps, nids):
            for local in step.deps:
                dep_nid = nids[local]
                self._add_dep(dep_nid, nid, DepClass.DATA_FLOW)
                # A chain step that depends on a block-template p2p is a
                # CONSUMER of it: the emitter has no ET node for the transfer
                # itself, it wires the SEND id (src side) or RECV id (dst side)
                # into the consumers. Without this the MoE cold-rank JOIN would
                # lose its ordering against its own residual send.
                dep_node = self._nodes[dep_nid]
                if isinstance(dep_node, _ProtoTransfer) and nid not in dep_node.consumers:
                    dep_node.consumers.append(nid)
        # The MoE hot/cold join: a step of a SIBLING chain that additionally
        # depends on this one (``block_program.py:366-373``). The referenced
        # chain is always already built — the hot device is ``min(members)`` and
        # devices are expanded ascending, which is invariant P4.
        for step, nid in zip(chain.steps, nids):
            if not isinstance(step, CommStep):
                continue
            for ref in step.extra_consumers:
                target = self._chain_step_nid(work, int(ref.device), ref.index)
                self._link_extra_consumer(nid, target)
        return nids

    def _chain_step_nid(self, work: WorkItem, device: int, index: int) -> int:
        nids = self._chain_nodes.get((work, device))
        if nids is None:
            raise BuildError(
                f"{work!r} has no expansion on device {device}; a StepRef must "
                "point at a sibling chain of the same expansion (P4: the hot "
                "rank is min(members) and expands first)"
            )
        return nids[index]

    def _link_extra_consumer(self, nid: int, target: int) -> None:
        node = self._nodes[nid]
        if isinstance(node, _ProtoTransfer):
            node.consumers.append(target)
        self._add_dep(nid, target, DepClass.DATA_FLOW)

    def _add_compute_step(
        self, work: WorkItem, chain: ExpandedChain, step: ComputeStep, order: _OrderKey
    ) -> int:
        role = (
            OpRole.JOIN
            if step.kind is StepKind.JOIN
            else (
                OpRole.GEMM
                if step.entry_name is not None
                else _ROLE_BY_KIND[work.kind]
            )
        )
        return self._add(
            _ProtoCompute(
                nid=self._next_nid(),
                order=order,
                device=int(chain.device),
                name=step.name,
                duration=float(step.duration),
                work=work,
                role=role,
                direction=(
                    IRDirection.FORWARD
                    if work.direction is Direction.FORWARD
                    else IRDirection.BACKWARD
                ),
                mem_kind=step.mem_kind,
                recompute=bool(step.recompute),
                param_gather=bool(step.param_gather),
                micro_batch=work.microbatch,
                layer=work.layer,
                is_moe_layer=bool(
                    work.layer is not None and self._fw.spec.is_moe_layer(work.layer)
                ),
                entry_name=step.entry_name,
            )
        )

    def _add_comm_step(
        self,
        work: WorkItem,
        chain: ExpandedChain,
        step: CommStep,
        order: _OrderKey,
        prefix: Sequence[int],
    ) -> int:
        spec = step.spec
        device = int(chain.device)
        if spec.kind is CollectiveType.PIPELINE:
            # A block-template p2p (the MoE ``residual_p2p``). ONE object, one
            # identity: its destination is the sibling step it feeds.
            dst = device
            for ref in step.extra_consumers:
                dst = int(ref.device)
                break
            producer = prefix[step.deps[0]] if step.deps else -1
            if producer < 0:
                raise BuildError(
                    f"block-template transfer {spec.key!r} has no producer step"
                )
            nid = self._add(
                _ProtoTransfer(
                    nid=self._next_nid(),
                    order=order,
                    device=device,
                    # Endpoints in the name: a block-template p2p key is
                    # instantiated once per (src, dst) pair, and the emitter
                    # derives the ET node name from this one. Diagnostics only —
                    # names are not part of canonical equivalence — but without
                    # it two distinct sends on one rank read identically.
                    name=f"{spec.key}_rank{device}_to_rank{dst}",
                    src_device=device,
                    dst_device=dst,
                    size_bytes=float(spec.size_bytes),
                    comm_type=spec.kind,
                    producer=producer,
                    moe_component=spec.moe_component,
                    participants=int(spec.participants),
                    interconnect=canonical_axis_label(spec.axes) if spec.axes else None,
                )
            )
            self._add_dep(producer, nid, DepClass.DATA_FLOW)
            return nid

        is_dp = tuple(spec.axes) == ("dp",)
        group = (
            None
            if is_dp
            else self._placement.communicators.group_for(spec.axes, DeviceId(device))
        )
        nid = self._add(
            _ProtoCollective(
                nid=self._next_nid(),
                order=order,
                device=device,
                name=spec.key,
                coll=spec.kind,
                size_bytes=float(spec.size_bytes),
                participants=int(spec.participants),
                axes=tuple(spec.axes),
                is_dp=is_dp,
                group=group,
                work=work,
                comm_key=spec.key,
            )
        )
        if step.overlap is not None:
            producer = None
            for local in step.deps:
                candidate = prefix[local]
                if isinstance(self._nodes[candidate], _ProtoCompute):
                    producer = candidate
            self._overlap_sites.append((nid, step.overlap, producer))
        return nid

    # ------------------------------------------------------------------
    # phase 5b — R2 (cross-layer data flow)
    # ------------------------------------------------------------------
    def _cross_layer_size(self) -> float:
        """``ByteSource("cross_layer", CEIL_DIV_CLUSTER).bytes_for(fw, instances)``
        with ``instances = placement.cluster_size()`` — 1 at COARSE (raw bytes,
        ``pipeline_coarse.py:222,238``), ``tp*cp*ep`` at FINE/BLOCK (divided,
        ``pipeline_fine.py:645``). B3's amendment is what lets ONE rule reproduce
        both."""
        if self._cross_layer_bytes is None:
            if CROSS_LAYER_KEY not in self._fw.spec.comm:
                raise BuildError(
                    f"Comm key {CROSS_LAYER_KEY!r} is required for a cross-device "
                    "data-flow edge but is not declared in WorkloadSpec.comm "
                    f"(declared: {sorted(self._fw.spec.comm)})"
                )
            self._cross_layer_bytes = ByteSource(
                key=CROSS_LAYER_KEY, split=ByteSplit.CEIL_DIV_CLUSTER
            ).bytes_for(self._fw, self._placement.cluster_size())
        return self._cross_layer_bytes

    def _cross_layer_spec(self):
        """The ``cross_layer`` :class:`~program.workload.CommSpec`.

        R2 needs three things off it: the byte count (``_cross_layer_size``) and
        — so the analytical evaluator can time the p2p from the OP rather than
        re-deriving it from a name — the analytical participant count and the
        interconnect axis key (INTERFACES §4.7 amendment 2026-07-28).
        """
        return self._fw.spec.comm.require(CROSS_LAYER_KEY)

    def _apply_r2(self) -> None:
        """**R2** — the microbatch's data-flow order, forward and backward.

        Forward: ``EMBEDDING -> LAYER 0 ... LAYER L-1 -> SOFTMAX``. Backward: the
        mirror image, where a layer's ENTRY is its ``RECOMPUTE`` chain when one
        exists (``_bwd_entry_node``, ``schedule.py:682-687``) — and the
        ``RECOMPUTE -> LAYER/BACKWARD`` link inside one layer is a PLAIN
        same-device dep, because the rematerialized activation never leaves the
        device that recomputed it (legacy wires it as a direct
        ``recompute_node.add_child(transformer_node_b)``, ``schedule.py:750``).

        **AMENDMENT 2026-07-29.** R2 also links the two ends of a microbatch:

        * ``SOFTMAX/FORWARD(b) -> SOFTMAX/BACKWARD(b)`` — the loss gradient
          needs the forward logits. Legacy never wired it and the new core did
          not either: the ordering was carried ONLY by R3, i.e. by whatever
          adjacency the ``SchedulePolicy`` happened to produce, and R3 is free
          to drop an edge it finds transitively implied (``:995``). Under GPipe
          it survives; under any schedule that interleaves forward and backward
          it is one adjacency away from being lost silently. It is a DATA_FLOW
          edge, so it is stated as one.
        * ``LAYER/FORWARD(b,l) -> RECOMPUTE(b,l)`` — the rematerialization
          replays the forward of the same layer and needs what that forward
          stashed. Same argument, same fix.

        Both are same-stage by construction (a microbatch's softmax lives on
        stage ``pp-1`` in both directions; a layer's recompute lives with the
        layer), so both are plain same-device deps and neither adds an op.
        """
        work = self._work
        shape = self._fw.spec.shape
        layers = int(shape.num_layers)
        for b in range(int(shape.micro_batches)):
            forward = [work.get(WorkKind.EMBEDDING, Direction.FORWARD, microbatch=b)]
            forward.extend(
                work.get(WorkKind.LAYER, Direction.FORWARD, microbatch=b, layer=layer)
                for layer in range(layers)
            )
            forward.append(work.get(WorkKind.SOFTMAX, Direction.FORWARD, microbatch=b))
            sequence = [item for item in forward if item is not None]
            for producer, consumer in zip(sequence, sequence[1:]):
                self._link(producer, consumer, transfer=True)

            if not self._fw.spec.run.include_backward:
                continue

            previous = work.get(WorkKind.SOFTMAX, Direction.BACKWARD, microbatch=b)
            softmax_f = work.get(WorkKind.SOFTMAX, Direction.FORWARD, microbatch=b)
            if softmax_f is not None and previous is not None:
                self._link(softmax_f, previous, transfer=False)
            for layer in reversed(range(layers)):
                backward = work.get(
                    WorkKind.LAYER, Direction.BACKWARD, microbatch=b, layer=layer
                )
                if backward is None:
                    continue
                remat = work.get(
                    WorkKind.RECOMPUTE, Direction.FORWARD, microbatch=b, layer=layer
                )
                entry = remat if remat is not None else backward
                if previous is not None:
                    self._link(previous, entry, transfer=True)
                if remat is not None:
                    layer_f = work.get(
                        WorkKind.LAYER, Direction.FORWARD, microbatch=b, layer=layer
                    )
                    if layer_f is not None:
                        self._link(layer_f, remat, transfer=False)
                    self._link(remat, backward, transfer=False)
                previous = backward
            embedding_b = work.get(
                WorkKind.EMBEDDING, Direction.BACKWARD, microbatch=b
            )
            if previous is not None and embedding_b is not None:
                self._link(previous, embedding_b, transfer=True)

    def _link(self, producer: WorkItem, consumer: WorkItem, *, transfer: bool) -> None:
        """Pair the producer's chains with the consumer's and link each pair.

        Pairing is by cluster rank (``pipeline_fine.py:659-694``: ``r -> r``).
        When one side is pinned to a single device (embedding, softmax — Class B
        10d) that one chain pairs with every chain of the other side, which is
        exactly what legacy does by giving one edge many children / many
        per-rank edges one child.
        """
        producers = self._chains[producer]
        consumers = self._chains[consumer]
        # THE BYTE DECISION IS PER STAGE, NOT PER DEVICE. ``cross_layer`` models
        # PIPELINE activation movement, so a link inside one stage carries no
        # payload however many devices the stage has: legacy decides it at the
        # coarse level with ``prev_node.hw_id == curr_node.hw_id`` (hw_id IS the
        # stage — ``schedule.py:628``, ``:639``, ``:648``, ``:755``, ``:763``,
        # ``:771``) and the FINE expansion inherits the zero-byte control edge.
        # Keying it on the DEVICE instead would invent a cross-tp-rank
        # ``cross_layer`` payload for every embedding -> layer 0 link.
        size = (
            0.0
            if self._placement.same_stage(producer, consumer)
            else self._cross_layer_size()
        )
        for prod, cons in _pair(producers, consumers, producer, consumer):
            src = int(prod.device)
            dst = int(cons.device)
            producer_nid = self._chain_nodes[(producer, src)][prod.exit]
            consumer_nid = self._chain_nodes[(consumer, dst)][cons.entry]
            if not transfer:
                if src != dst:
                    raise BuildError(
                        f"{producer!r} -> {consumer!r} is a same-device data-flow "
                        f"link but the chains are on {src} and {dst}"
                    )
                self._add_dep(producer_nid, consumer_nid, DepClass.DATA_FLOW)
                continue
            self._emit_cross_layer(producer, prod, producer_nid, cons, consumer_nid, size)

    def _emit_cross_layer(
        self,
        producer: WorkItem,
        prod: ExpandedChain,
        producer_nid: int,
        cons: ExpandedChain,
        consumer_nid: int,
        size: float,
    ) -> None:
        # A same-device link is still an OP, not a bare dep: Class B item 9 says
        # the presence of the zero-byte same-stage PIPELINE event is load-bearing
        # for the analytical evaluator's ready-scan, so it is emitted as a
        # same-device TransferOp exactly as today (INTERFACES §4.3 R2).
        src = int(prod.device)
        dst = int(cons.device)
        order = self._order(producer, src, _PHASE_XFER, self._xfer_seq)
        self._xfer_seq += 1
        spec = self._cross_layer_spec()
        nid = self._add(
            _ProtoTransfer(
                nid=self._next_nid(),
                order=order,
                device=src,
                name=CROSS_LAYER_KEY,
                src_device=src,
                dst_device=dst,
                size_bytes=size,
                comm_type=CollectiveType.PIPELINE,
                producer=producer_nid,
                consumers=[consumer_nid],
                participants=int(spec.participants),
                interconnect=canonical_axis_label(spec.axes) if spec.axes else None,
            )
        )
        self._add_dep(producer_nid, nid, DepClass.DATA_FLOW)
        # The compute-anchor double-dep (``pipeline_fine.py:672-687``): the SEND
        # must fire off the last COMPUTE, not off a trailing collective.
        anchor = self._nearest_compute(prod, producer_nid)
        if anchor is not None and anchor != producer_nid:
            self._add_dep(anchor, nid, DepClass.DATA_FLOW)
        self._add_dep(nid, consumer_nid, DepClass.DATA_FLOW)

    def _nearest_compute(self, chain: ExpandedChain, exit_nid: int) -> Optional[int]:
        if isinstance(self._nodes[exit_nid], _ProtoCompute):
            return None
        nids = self._chain_nodes[(chain.work, int(chain.device))]
        for index in range(chain.exit, -1, -1):
            if isinstance(self._nodes[nids[index]], _ProtoCompute):
                return nids[index]
        return None

    # ------------------------------------------------------------------
    # phase 6 — R3 (device serialization)
    # ------------------------------------------------------------------
    def _apply_r3(self) -> None:
        """**R3** — one edge per adjacent pair of a device's projection, added
        ONLY when the pair is not already transitively implied.

        Redundancy elimination is the point (**D1**): consecutive layers of one
        microbatch on one device are already chained by R2, so ``reaches`` is
        True and nothing is added — reproducing legacy, which only ever wires
        the MICROBATCH BOUNDARY. Blanket serialization would inflate the DAG
        with edges AstraSim then has to carry per rank.
        """
        for dep in self._schedule.implied_deps(self._placement.devices_for):
            device = int(dep.device)
            before = self._chain_nodes.get((dep.before, device))
            after = self._chain_nodes.get((dep.after, device))
            if before is None or after is None:  # pragma: no cover - projection
                raise BuildError(
                    f"ScheduleDep {dep.before!r} -> {dep.after!r} names device "
                    f"{device}, which is not where both are placed"
                )
            source = before[self._chain_by_device[(dep.before, device)].exit]
            target = after[self._chain_by_device[(dep.after, device)].entry]
            if self._reaches(source, target):
                continue
            self._add_dep(source, target, DepClass.SCHEDULE)

    # ------------------------------------------------------------------
    # phase 7 — requirements
    # ------------------------------------------------------------------
    def _collect_requirements(self) -> Tuple[SyncRequirement, ...]:
        ctx = ShardingContext(
            fw=self._fw,
            work=self._work,
            grad_accum=self._grad_accum,
            stages=self._schedule.layers,
            # A prefetch requirement names the work it prefetches FOR
            # (``ShardingContext.next_after``); that is a question about the
            # execution order, which L3 declares, so the order is handed over
            # rather than re-derived from a microbatch index.
            order=self._schedule.order(),
        )
        out: List[SyncRequirement] = list(self._sharding.workload_requirements(ctx))
        for item in self._work:
            out.extend(self._sharding.requirements(item, ctx))
            if self._routing is not None:
                out.extend(ep_sync_requirements(item, ctx))
        # K1/K3: the policy set is order-independent, so the RESOLUTION order is
        # declared here and nowhere else.
        out.sort(key=self._sync_order.key)
        return tuple(out)

    # ------------------------------------------------------------------
    # phase 8 — R4 (sync attach)
    # ------------------------------------------------------------------
    def _apply_r4(self, requirements: Sequence[SyncRequirement]) -> None:
        resolved: Set[SyncKey] = set()
        for rank, req in enumerate(requirements):
            if req.mode is AttachMode.AFTER:
                for anchor in req.anchors:
                    if isinstance(anchor, SyncKey) and anchor not in resolved:
                        # Makes the S3/S5/S10 chaining a CHECKABLE property
                        # instead of a construction-order coincidence (§4.4).
                        raise BuildError(
                            f"SyncRequirement {req.key!r} is AFTER({anchor!r}), which "
                            "is not strictly earlier in SyncOrder"
                        )
            devices = self._placement.devices_for_sync(req)
            groups = self._placement.communicators.groups_for(req, self._placement)
            if len(groups) != len(devices):  # pragma: no cover - defensive
                raise BuildError(
                    f"groups_for({req.key!r}) returned {len(groups)} entries for "
                    f"{len(devices)} instance devices"
                )
            size = req.bytes.bytes_for(self._fw, len(devices))
            instances: List[int] = []
            for instance, (device, group) in enumerate(zip(devices, groups)):
                attached = self._attach(req, rank, instance, int(device), group, size)
                if attached is not None:
                    instances.append(attached)
            if instances:
                self._sync_nodes[req.key] = instances
                resolved.add(req.key)
            else:
                # INTERFACES §2.3 note 1: a requirement whose resolved anchor
                # set is empty is DROPPED and materializes no op. Legacy consumes
                # a legacy op id for it and produces an unreachable object the
                # lowering never collects (divergence class C).
                self._dropped_requirements.append(f"{req.key!r} ({req.origin})")

    def _attach(
        self,
        req: SyncRequirement,
        rank: int,
        instance: int,
        device: int,
        group: Optional[GroupKey],
        size: float,
    ) -> Optional[int]:
        """Materialize ONE instance of ``req`` on ``device`` and attach it.

        Returns the node id, or ``None`` when the anchors resolved to nothing.
        """
        anchors = [self._resolve_anchor(anchor, device) for anchor in req.anchors]
        anchors = [pair for pair in anchors if pair[0] or pair[1]]
        if not anchors:
            return None

        nid = self._add(
            _ProtoCollective(
                nid=self._next_nid(),
                order=self._order(
                    req.place_on, device, _PHASE_SYNC, rank, instance
                ),
                device=device,
                name=self._sync_name(req, instance),
                coll=req.kind,
                size_bytes=float(size),
                participants=int(req.participants),
                axes=tuple(req.axes),
                is_dp=bool(req.is_dp),
                group=group,
                work=req.place_on,
                comm_key=req.comm_key,
            )
        )
        mode = req.mode
        for entries, exits in anchors:
            if mode is AttachMode.BEFORE:
                # deps(req) := deps(a); deps(a) += req. Degenerate when ``a`` is
                # a root, and then ``req`` becomes the root (S1).
                for entry in entries:
                    for dep in list(self._nodes[entry].deps):
                        self._add_dep(dep, nid, DepClass.SYNC)
                        dep_node = self._nodes[dep]
                        if (
                            isinstance(dep_node, _ProtoTransfer)
                            and int(dep_node.dst_device) == int(device)
                            and nid not in dep_node.consumers
                        ):
                            # a p2p edge is carried by ``consumers`` (the RECV id
                            # is wired there), not by a ctrl_dep
                            dep_node.consumers.append(nid)
                    self._add_dep(nid, entry, DepClass.SYNC)
            elif mode is AttachMode.AFTER:
                for exit_nid in exits:
                    self._add_dep(exit_nid, nid, DepClass.SYNC)
            elif mode is AttachMode.PARALLEL_TO:
                # deps(req) += deps(a); for s in succ(a, via): deps(s) += req.
                #
                # A MULTI-ANCHOR PARALLEL_TO (rows S6/S14, §2.3 note 2) whose
                # anchors are ordered by data flow could in principle close a
                # cycle: `req` would inherit anchor 1's successors AND anchor 2's
                # deps, and anchor 2's deps are downstream of anchor 1. It does
                # not happen for either production row — both use
                # VIA_NON_DATA_FLOW, whose successors are the NEXT microbatch's
                # entry, which is never upstream of this microbatch's anchors —
                # and if a future policy ever does, `_kahn` raises and names the
                # stuck ops rather than emitting an unschedulable Program.
                for entry in entries:
                    for dep in list(self._nodes[entry].deps):
                        self._add_dep(dep, nid, DepClass.SYNC)
                        dep_node = self._nodes[dep]
                        if (
                            isinstance(dep_node, _ProtoTransfer)
                            and int(dep_node.dst_device) == int(device)
                            and nid not in dep_node.consumers
                        ):
                            # a p2p edge is carried by ``consumers`` (the RECV id
                            # is wired there), not by a ctrl_dep
                            dep_node.consumers.append(nid)
                for exit_nid in exits:
                    for succ in list(self._nodes[exit_nid].succs):
                        if succ == nid:
                            continue
                        if self._edges[(exit_nid, succ)] & req.via:
                            self._add_dep(nid, succ, DepClass.SYNC)
            elif mode is AttachMode.OVERLAP_WITH:  # pragma: no cover - unused
                raise BuildError(
                    "AttachMode.OVERLAP_WITH on a SyncRequirement is realized by "
                    "the OverlapDecl (§4.5), not by an attach; no production "
                    "policy declares it"
                )
            else:  # pragma: no cover - AttachMode is closed
                from typing import assert_never

                assert_never(mode)

        # DECLARED consumers (amendment 2026-07-29, §2.2). Applied after the
        # mode so a requirement whose "who waits for me" answer is knowable at
        # declaration time does not have to be discovered through ``via`` — the
        # gap that makes rows S6/S14 ride an R3 edge. Idempotent: under GPipe
        # the ``via`` scan above already added exactly these edges.
        for consumer in req.consumers:
            consumer_entries, _consumer_exits = self._resolve_anchor(consumer, device)
            if not consumer_entries:
                raise BuildError(
                    f"SyncRequirement {req.key!r} ({req.origin}) declares consumer "
                    f"{consumer!r}, which has no chain reachable from device {device}"
                )
            for entry in consumer_entries:
                if entry == nid:  # pragma: no cover - defensive
                    continue
                self._add_dep(nid, entry, DepClass.SYNC)
        return nid

    def _sync_name(self, req: SyncRequirement, instance: int) -> str:
        """Diagnostic name. Names are NOT part of canonical equivalence
        (``equiv/canonical.py:20-27``) and nothing dispatches on them."""
        key = req.key
        parts = [str(key.comm_key), key.phase.name.lower()]
        if key.microbatch is not None:
            parts.append(f"b{key.microbatch}")
        if key.layer is not None:
            parts.append(f"l{key.layer}")
        if instance:
            parts.append(f"rank{instance}")
        return "_".join(parts)

    def _resolve_anchor(
        self, anchor: SyncAnchor, device: int
    ) -> Tuple[List[int], List[int]]:
        """``anchor -> (entry nodes, exit nodes)`` for one instance device.

        A chain's ``entry``/``exit`` are the only handles R4 may reference
        (INTERFACES §4.3 R1): ``deps(a)`` means the entry's deps and
        ``succ(a)`` the exit's successors, because a coarse node's parents and
        children become exactly those in a chain.

        Preference is the anchor's chain ON ``device``; when the anchor is pinned
        elsewhere (a ``PER_CLUSTER_RANK`` requirement hanging off a cluster-rank-0
        kind — rows S7/S13) every chain of the anchor is used. That region is new
        development with new goldens: legacy's ``_ensure_zero3_per_rank_edges``
        gives its per-rank gathers no parents at all in the ``hw_ids`` mode
        (``pipeline_fine.py:389-391``), i.e. it makes them graph roots.
        """
        if isinstance(anchor, SyncKey):
            nodes = self._sync_nodes.get(anchor)
            if not nodes:
                return ([], [])
            local = [nid for nid in nodes if self._nodes[nid].device == device]
            chosen = local if local else list(nodes)
            return (chosen, list(chosen))
        chains = self._chains.get(anchor)
        if not chains:
            return ([], [])
        local_chains = [chain for chain in chains if int(chain.device) == device]
        chosen_chains = local_chains if local_chains else list(chains)
        entries: List[int] = []
        exits: List[int] = []
        for chain in chosen_chains:
            nids = self._chain_nodes[(anchor, int(chain.device))]
            entries.append(nids[chain.entry])
            exits.append(nids[chain.exit])
        return (entries, exits)

    # ------------------------------------------------------------------
    # phase 8b — R5 (the optimizer's gradient dependency)
    # ------------------------------------------------------------------
    def _apply_r5(self) -> None:
        """**R5** (new, 2026-07-29) — ``OPTIMIZER(stage s)`` runs after EVERY
        gradient-producing backward item of stage ``s``, on each device it is
        placed on.

        Stated because nothing stated it. Legacy hand-picked ONE attach point
        per stage (``simulate_train_graph.py:1300-1318``: ``embedding_node_b[0]``
        on stage 0, ``_bwd_exit_node(0, min_layer(s))`` elsewhere) and the new
        core did not port it — the ordering came out of R3's per-device
        adjacency, which happens to end each stage's projection with exactly
        that item **because GPipe walks microbatches in reverse**. A schedule
        whose backward walks microbatches ASCENDING leaves 6 of 9 backward items
        un-ordered before the optimizer (measured), i.e. the optimizer applies
        two thirds of a gradient. That is a property of the model, not of the
        schedule, so it belongs in a rule.

        Redundancy-eliminated exactly like R3 (**D1**): the edge is materialized
        only when the ordering is not already implied, so under GPipe this rule
        adds NOTHING and the artifact is unchanged. It runs AFTER R4 so the
        sync lattice's own edges count as implications and so no ``via`` scan
        can see an R5 edge.

        **Not** in scope: "the optimizer runs after its stage's GRADIENT
        REDUCER" (audit item D11). That edge is absent in legacy and in the new
        core alike; it is a modeling question for the owner, not a regression.
        """
        if not self._fw.spec.run.include_backward:
            return
        optimizers = [
            item for item in self._work if item.kind is WorkKind.OPTIMIZER
        ]
        if not optimizers:
            return
        producers = [
            item
            for item in self._work
            if item.direction is Direction.BACKWARD
            and item.kind in _R5_GRADIENT_KINDS
        ]
        for optimizer in sorted(optimizers, key=WorkItem.sort_key):
            stage = int(self._schedule.stage_of(optimizer))
            for chain in self._chains.get(optimizer, ()):
                device = int(chain.device)
                target = self._chain_nodes[(optimizer, device)][chain.entry]
                for item in sorted(producers, key=WorkItem.sort_key):
                    if int(self._schedule.stage_of(item)) != stage:
                        continue
                    entry = self._chain_by_device.get((item, device))
                    if entry is None:
                        continue
                    source = self._chain_nodes[(item, device)][entry.exit]
                    if source == target or self._reaches(source, target):
                        continue
                    self._add_dep(source, target, DepClass.DATA_FLOW)

    # ------------------------------------------------------------------
    # phase 9 — overlap realization (INTERFACES §4.5)
    # ------------------------------------------------------------------
    def _realize_overlap(self) -> None:
        """Realize every :class:`OverlapDecl` a block-template ``CommStep``
        carries.

        Runs AFTER R4, which is where the legacy pipeline applies it (flatten ->
        overlap -> propagate -> lower) and therefore after the whole ZeRO
        lattice. Split ops get FRESH order keys; the ``head``/``_block``/``_ovlp``
        op-id-reuse quirks (``transforms.py:200,255,267``) do not survive,
        because program order is computed (§4.6) so duplicate ids cannot exist.

        ``retime`` interaction: a per-DP duration profile cannot be split
        (``transforms.py:468-478``), so overlap is realized here, before any
        retime write-back can apply. ``retime`` operates on a BUILT Program and
        is forbidden from running before ``build()`` returns.
        """
        # PRODUCER splits are grouped by the compute they split: one compute may
        # feed several overlapped collectives (a MoE parallel post-group), and
        # legacy splits the node once with all of them as ``tp_children``.
        producer_sites: Dict[int, List[int]] = {}
        producer_decl: Dict[int, OverlapDecl] = {}
        for nid, decl, producer in self._overlap_sites:
            if decl.anchor is OverlapAnchor.PRODUCER:
                if producer is None:
                    raise BuildError(
                        f"OverlapDecl(PRODUCER) on {self._nodes[nid].name!r} has no "
                        "producing ComputeStep in its chain"
                    )
                producer_sites.setdefault(producer, []).append(nid)
                producer_decl[producer] = decl
            else:
                self._split_collective(nid, decl)
        for producer in sorted(producer_sites):
            self._split_compute(producer, producer_sites[producer], producer_decl[producer])

    def _split_compute(
        self, compute: int, collectives: Sequence[int], decl: OverlapDecl
    ) -> None:
        """``OverlapAnchor.PRODUCER`` (verbatim ``_split_tp_node_fine``)."""
        node = self._nodes[compute]
        if not isinstance(node, _ProtoCompute):  # pragma: no cover - defensive
            raise BuildError("PRODUCER overlap must split a ComputeOp")
        duration = float(node.duration)
        if duration <= 0.0:
            return

        if decl.is_hoist:
            # f >= 1: hoist — the collective takes the compute's deps and the
            # compute takes the collective's consumers (``transforms.py:177-188``).
            for coll in collectives:
                for dep in list(self._nodes[compute].deps):
                    self._reparent_dep(dep, coll, like=(dep, compute))
                self._drop_dep(compute, coll)
                for succ in list(self._nodes[coll].succs):
                    self._reparent_dep(compute, succ, like=(coll, succ))
            return

        head_duration = duration * (1.0 - decl.fraction)
        tail_duration = duration * decl.fraction
        if head_duration <= 0.0 or tail_duration <= 0.0:
            return

        # The ORIGINAL node stays the tail (legacy mutates it in place), so every
        # successor keeps pointing at it and only the deps move.
        slot, device_index, _, a, _ = node.order
        head = self._add(
            _ProtoCompute(
                nid=self._next_nid(),
                order=(slot, device_index, _PHASE_CHAIN, a, -1),
                device=node.device,
                name=f"{node.name}_head",
                duration=head_duration,
                work=node.work,
                role=node.role,
                direction=node.direction,
                mem_kind=node.mem_kind,
                recompute=node.recompute,
                param_gather=node.param_gather,
                micro_batch=node.micro_batch,
                layer=node.layer,
                is_moe_layer=node.is_moe_layer,
                entry_name=node.entry_name,
            )
        )
        node.duration = tail_duration
        for dep in list(node.deps):
            for dep_class in set(self._edges[(dep, compute)]):
                self._add_dep(dep, head, dep_class)
            # A transfer whose CONSUMER was this compute now feeds the head:
            # the deps move, so the consumer wiring must move with them or the
            # emitter would wire the RECV id into the tail while the dep graph
            # says head (legacy did the same rewrite, transforms.py:_split_tp).
            dep_node = self._nodes[dep]
            if isinstance(dep_node, _ProtoTransfer):
                dep_node.consumers = [
                    head if consumer == compute else consumer
                    for consumer in dep_node.consumers
                ]
            self._drop_dep(dep, compute)
        self._add_dep(head, compute, DepClass.DATA_FLOW)
        for coll in collectives:
            self._reparent_dep(head, coll, like=(compute, coll))
            self._drop_dep(compute, coll)
            for succ in list(self._nodes[coll].succs):
                self._reparent_dep(compute, succ, like=(coll, succ))

    def _split_collective(self, coll: int, decl: OverlapDecl) -> None:
        """``OverlapAnchor.CONSUMER`` (verbatim ``_split_cp_edge_fine``)."""
        node = self._nodes[coll]
        if not isinstance(node, _ProtoCollective):  # pragma: no cover - defensive
            raise BuildError("CONSUMER overlap must split a CollectiveOp")
        blocking = [
            succ
            for succ in self._nodes[coll].succs
            if self._is_blocking_consumer(succ, decl)
        ]
        if not blocking:
            return
        total = float(node.size_bytes)
        block_bytes = math.ceil(total * (1.0 - decl.fraction))
        ovlp_bytes = max(0.0, total - block_bytes)
        preds = list(self._nodes[coll].deps)
        succs = list(self._nodes[coll].succs)

        if decl.is_hoist or total <= 0 or block_bytes <= 0:
            # The blocking consumer is re-parented onto the collective's
            # predecessors (``transforms.py:233-244``).
            for consumer in blocking:
                for pred in preds:
                    self._reparent_dep(pred, consumer, like=(pred, coll))
                self._drop_dep(coll, consumer)
            for succ in succs:
                if succ in blocking:
                    continue
                for consumer in blocking:
                    self._reparent_dep(consumer, succ, like=(coll, succ))
            return

        node.size_bytes = float(block_bytes)
        node.name = f"{node.name}_block"
        ovlp: Optional[int] = None
        if ovlp_bytes > 0:
            slot, device_index, _, a, _ = node.order
            ovlp = self._add(
                _ProtoCollective(
                    nid=self._next_nid(),
                    order=(slot, device_index, _PHASE_CHAIN, a, 1),
                    device=node.device,
                    name=f"{node.comm_key}_ovlp",
                    coll=node.coll,
                    size_bytes=float(ovlp_bytes),
                    participants=node.participants,
                    axes=node.axes,
                    is_dp=node.is_dp,
                    group=node.group,
                    work=node.work,
                    comm_key=node.comm_key,
                )
            )
            self._add_dep(coll, ovlp, DepClass.DATA_FLOW)
        for succ in succs:
            if succ in blocking:
                continue
            carried = self._classes(coll, succ)
            self._drop_dep(coll, succ)
            for consumer in blocking:
                for dep_class in carried:
                    self._add_dep(consumer, succ, dep_class)
            for dep_class in carried:
                self._add_dep(coll if ovlp is None else ovlp, succ, dep_class)

    def _is_blocking_consumer(self, nid: int, decl: OverlapDecl) -> bool:
        node = self._nodes[nid]
        if not isinstance(node, _ProtoCompute):
            return False
        target = decl.blocking_consumer
        if target is None:  # pragma: no cover - OverlapDecl rejects it
            return False
        # Matched against ``ComputeStep.entry_name`` — a TYPED BlockTemplate
        # field, not the op's display name (``transforms.py:224`` tests
        # ``"attention" in name.lower()``) — and the needle is DATA on the
        # policy, never a literal here.
        return target.lower() in str(node.entry_name or "").lower()

    # ------------------------------------------------------------------
    # phase 10 — program order, uids, Program
    # ------------------------------------------------------------------
    def _finish(self, *, validate: bool, check_group_membership: bool = False) -> Program:
        uid_of = self._kahn()
        ops: List[Op] = [None] * len(self._nodes)  # type: ignore[list-item]
        groups: Dict[GroupKey, CommGroup] = {}
        labels = _LabelInterner()
        for node in sorted(self._nodes, key=lambda n: uid_of[n.nid]):
            uid = uid_of[node.nid]
            deps = tuple(uid_of[dep] for dep in node.deps)
            if isinstance(node, _ProtoCompute):
                ops[uid] = ComputeOp(
                    uid=uid,
                    name=node.name,
                    device=node.device,
                    duration=(float(node.duration),),
                    deps=deps,
                    role=node.role,
                    direction=node.direction,
                    mem_kind=node.mem_kind,
                    recompute=node.recompute,
                    param_gather=node.param_gather,
                    micro_batch=node.micro_batch,
                    layer=node.layer,
                    is_moe_layer=node.is_moe_layer,
                    work=node.work,
                )
            elif isinstance(node, _ProtoCollective):
                label = None
                if node.group is not None:
                    label = labels.label_for(node.comm_key or node.name, node.group)
                    groups.setdefault(
                        node.group, CommGroup(key=node.group, label=label)
                    )
                ops[uid] = CollectiveOp(
                    uid=uid,
                    name=node.name,
                    device=node.device,
                    coll=node.coll,
                    size_bytes=node.size_bytes,
                    participants=node.participants,
                    interconnect=canonical_axis_label(node.axes) if node.axes else None,
                    is_dp=node.is_dp,
                    label=label,
                    group=node.group,
                    deps=deps,
                    comm_key=node.comm_key,
                    axes=node.axes,
                    work=node.work,
                )
            else:
                ops[uid] = TransferOp(
                    uid=uid,
                    name=node.name,
                    src_device=node.src_device,
                    dst_device=node.dst_device,
                    size_bytes=int(node.size_bytes),
                    comm_type=node.comm_type,
                    producer=uid_of[node.producer],
                    deps=deps,
                    consumers=tuple(uid_of[c] for c in node.consumers),
                    moe_component=node.moe_component,
                    participants=node.participants,
                    interconnect=node.interconnect,
                )

        devices = tuple(int(device) for device in self._devices)
        dp_count = (
            int(self._dp_count_in)
            if self._dp_count_in is not None
            else int(self._fw.spec.run.retime_dp_count(self._fw.spec.degrees))
        )
        meta = ProgramMeta(
            label=self._label,
            misc={
                "granularity": self._granularity.name.lower(),
                "program_order": "kahn(slot,device,intra)",
                "duration_revision": int(self._fw.durations.revision),
                "num_stages_initial": len(devices),
                "schedule_policy": self._schedule.policy,
                "sync_order": self._sync_order.name,
                "sharding_policy": self._sharding.name,
                "grad_accum_policy": self._grad_accum.name,
                "recompute_policy": self._recompute.name,
                "overlap_policy": self._overlap.name,
                "routing_policy": None if self._routing is None else self._routing.name,
                "placement_policy": self._placement_policy.name,
                "layer_assignment_contiguous": self._schedule.layers.contiguous_layers(),
                "work_restricted": self._work_restricted,
                "work_directions": (
                    None
                    if self._directions is None
                    else tuple(d.name for d in self._directions)
                ),
                # **D1's artifact.** DERIVED from the edge table rather than
                # recorded at R3 time: overlap realization (§4.5) re-parents
                # edges after R3 runs, so a recorded ``(source, target)`` nid
                # pair could name an edge that no longer exists (measured: 6/13
                # stale on ``dp1tp2cp1pp2mb2sp1``, 14/25 on
                # ``dp2tp2cp2pp2mb2sp1``). Reading the CLASS off the surviving
                # edge cannot go stale, and it is what makes D1 checkable on the
                # artifact instead of on a log line. Re-parenting preserves the
                # class (``_reparent_dep``), which is what makes this exact.
                "schedule_edges": tuple(
                    sorted(
                        (uid_of[source], uid_of[target])
                        for (source, target), classes in self._edges.items()
                        if DepClass.SCHEDULE in classes
                    )
                ),
                "dropped_requirements": tuple(self._dropped_requirements),
                "gmap_workdir": self._gmap_workdir,
            },
        )
        program = Program(
            layout=self._placement.layout,
            dp_count=max(1, dp_count),
            devices=devices,
            ops=ops,
            groups=groups,
            meta=meta,
        )
        if validate:
            from program.validate import validate_program

            # V6 is promoted to ALWAYS-ON here (INTERFACES §4.8): every legacy
            # production caller passes check_races=False, so CONTEXT constraint 2
            # is unenforced outside tests. V7 is opt-in — see build()'s docstring
            # for why A2 makes it non-fatal until P7.
            validate_program(
                program,
                check_races=True,
                check_group_membership=check_group_membership,
            )
        return program

    def _kahn(self) -> Dict[int, int]:
        """Uids by Kahn's algorithm with a min-heap keyed on :meth:`_order`.

        Total (keys are unique by construction), respects ``dep < uid`` (**V1**,
        **O2**), schedule-major, and a pure function of the inputs (**O1**).
        """
        indegree = {node.nid: len(node.deps) for node in self._nodes}
        heap: List[Tuple[_OrderKey, int]] = [
            (node.order, node.nid) for node in self._nodes if indegree[node.nid] == 0
        ]
        heapq.heapify(heap)
        uid_of: Dict[int, int] = {}
        while heap:
            _, nid = heapq.heappop(heap)
            uid_of[nid] = len(uid_of)
            for succ in self._nodes[nid].succs:
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    heapq.heappush(heap, (self._nodes[succ].order, succ))
        if len(uid_of) != len(self._nodes):
            stuck = [nid for nid in indegree if nid not in uid_of][:8]
            raise BuildError(
                f"dependency cycle: {len(self._nodes) - len(uid_of)} ops never "
                f"became ready (e.g. {[self._nodes[n].name for n in stuck]})"
            )
        return uid_of


def _pair(
    producers: Sequence[ExpandedChain],
    consumers: Sequence[ExpandedChain],
    producer_work: WorkItem,
    consumer_work: WorkItem,
) -> Tuple[Tuple[ExpandedChain, ExpandedChain], ...]:
    """Cluster-rank pairing of two expansions.

    Equal widths pair ``r -> r``; a width-1 side (a kind pinned to cluster rank
    0 — Class B 10d) pairs with every chain of the other side. Any other
    combination is a placement bug, not something to paper over.
    """
    if len(producers) == len(consumers):
        return tuple(zip(producers, consumers))
    if len(producers) == 1:
        return tuple((producers[0], cons) for cons in consumers)
    if len(consumers) == 1:
        return tuple((prod, consumers[0]) for prod in producers)
    raise BuildError(
        f"cannot pair {len(producers)} chains of {producer_work!r} with "
        f"{len(consumers)} chains of {consumer_work!r}"
    )


class _LabelInterner:
    """``(base name, group) -> label``, with a ``name_N`` suffix per distinct
    member set.

    Same shape as ``legacy_lowering._assign_collective_labels_with_members``
    (``:270-277``), assigned in uid order so it is a pure function of the
    program order. The label is what the emitter interns wire gids by
    (``et_emit.py:221-263``): one label must map to exactly one member set, and
    that is guaranteed here because the member set is part of the key.
    """

    def __init__(self) -> None:
        self._labels: Dict[Tuple[str, Tuple[int, ...]], str] = {}
        self._suffix: Dict[str, int] = {}

    def label_for(self, base: str, group: GroupKey) -> str:
        key = (str(base), tuple(group.members))
        label = self._labels.get(key)
        if label is None:
            count = self._suffix.get(str(base), 0)
            label = str(base) if count == 0 else f"{base}_{count}"
            self._suffix[str(base)] = count + 1
            self._labels[key] = label
        return label
