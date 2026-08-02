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

"""L2 — ``Placement`` + ``BlockExpander``: WHERE work runs and WHAT ops it
expands into (INTERFACES.md §3).

**Granularity is a parameter, not a builder.** The three legacy builders

* ``program/pipeline_coarse.py`` (325 LOC) — one op per event, on the stage,
* ``program/pipeline_fine.py``  (970 LOC) — per-cluster-rank block chains,
* ``program/block_program.py``  (588 LOC) — one layer over the (tp,cp,ep) grid,

differ in exactly two decisions: which device space they linearize into, and
how many chains one unit of work expands to. Both are captured by
:class:`Granularity` here, so the three files collapse into one class pair.

What this module deletes (INTERFACES §3.6):

* ``pipeline_fine._hw_id_for_rank`` (:268-286) and
  ``block_program._rank_id`` (:188-189) — the two surviving hand-rolled rank
  formulas. Every device id in this module comes from
  :meth:`program.layout.RankLayout.linearize`, via
  :func:`program.layout.cluster_coords`. There is no ``*`` or ``//`` on a
  device id anywhere in this file.
* ``pipeline_fine._offset_stage_device`` (:288-303) — the ZeRO-3 "``hw ±
  par_degree``" hack. A :class:`program.work.SyncRequirement` names the
  WorkItem it is placed on (``place_on``), so the device is *looked up*, never
  offset. The ±1 pp hop existed only because the legacy edge knew its host but
  not its target.
* ``pipeline_fine.propagate_local_hw_ids`` (:796-850) and ``local_hw_id`` as a
  concept — placement is DECLARED at L1 (``SyncRequirement.spread``) and
  RESOLVED here (:meth:`Placement.devices_for_sync`), never back-propagated
  from whichever parent happened to be visited first.
* the name dispatch in ``_FineExpander._clone`` (``"linear_softmax" in
  obj.name`` at :417, ``"optimizer" in obj.name`` at :427) — placement now
  keys off :class:`program.work.WorkKind` through a named, swappable
  :class:`PlacementPolicy`.

Divergence from the FLAT builder, deliberate and documented (INTERFACES §3.4):
:class:`BlockExpander` **honors ``CommSpec.placement``** (``"pre"``/``"post"``).
``pipeline_fine.py:577-589`` chains every comm key *after* its GEMM regardless
of the declared placement; ``block_program._split_comm_keys`` honors it. This
is a no-op for all 42 goldens (no production comm rule emits ``placement="pre"``
— ``train_timing`` ``COMMUNICATION_RULES`` / ``MOE_COMMUNICATION_RULES`` /
``_make_moe_comm_specs`` all emit ``"post"``; only
``tests/test_block_builder_diff.py:249`` uses ``"pre"``), and it is one of the
six blockers for flattened MoE.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from memory_estimation import mem_kind_from_op_name

from program.block import BlockTemplate, GemmEntry
from program.groups import CommunicatorFactory, GroupError, canonical_axis_label
from program.axes import (
    CLUSTER_AXES as _REGISTRY_CLUSTER_AXES,
    PIPELINE_AXES,
    REPLICA_AXES,
    activation_sharding_axes,
    axis_sizes_from,
)
from program.layout import CANONICAL_AXES, RankLayout, cluster_coords
from program.types import DP_AXIS, AxisName, CommKey, Coords, DeviceId, LayerId, StageId
from program.work import (
    Direction,
    SyncSpread,
    WorkItem,
    WorkKind,
    duration_for,
    mem_kind,
)
from program.workload import CommSpec, CommSpecTable, FrozenWorkload

if TYPE_CHECKING:  # pragma: no cover - typing only
    from program.policies.overlap import OverlapDecl, OverlapPolicy
    from program.policies.routing import MoERoutingPolicy
    from program.schedule.policy import LayerAssignment
    from program.work import SyncRequirement


class PlacementError(ValueError):
    """A WorkItem could not be placed, or a device space is inconsistent."""


# ---------------------------------------------------------------------------
# Granularity (INTERFACES §3.1)
# ---------------------------------------------------------------------------


class Granularity(Enum):
    """What a build makes of one :class:`~program.work.WorkItem`.

    ===========  ==================================  ==========================
    Granularity  device space                        WorkItem expands to
    ===========  ==================================  ==========================
    PIPELINE     ``layout.subset(("pp","dp"))``      1 ComputeStep on the stage
    FLAT         the full ``layout``                 cluster_size block chains
    BLOCK        ``layout.subset(("tp","cp","ep"))`` 1 chain per cluster rank
    ===========  ==================================  ==========================

    Named ``PIPELINE``/``FLAT`` until 2026-08-02. The names now say what the
    granularity IS rather than how much of it there is: ``PIPELINE`` is the
    pipeline-stage view the hierarchical and analytical modes evaluate, and
    ``FLAT`` is the every-GPU view the flattened mode emits.
    """

    PIPELINE = auto()
    FLAT = auto()
    BLOCK = auto()


#: Device axes each granularity linearizes into (``dp`` is stamped at
#: emission, never here — INTERFACES §3.3).
_GRANULARITY_AXES: Mapping[Granularity, Tuple[AxisName, ...]] = MappingProxyType(
    {
        # Derived from AxisRole (program.axes): PIPELINE devices ARE stages,
        # so it materializes the PIPELINE + REPLICA axes; BLOCK is one layer
        # over the cluster; FLAT is everything.
        Granularity.PIPELINE: PIPELINE_AXES + REPLICA_AXES,
        Granularity.FLAT: CANONICAL_AXES,
        Granularity.BLOCK: _REGISTRY_CLUSTER_AXES,
    }
)

#: Axes forming the transformer cluster (one pipeline stage).
CLUSTER_AXES: Tuple[AxisName, ...] = _REGISTRY_CLUSTER_AXES


# ---------------------------------------------------------------------------
# DeviceCoord
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class DeviceCoord:
    """A device expressed as axis coordinates.

    ``DeviceId`` is ``layout.linearize(coord.coords)``; this type exists so a
    caller can *say* "the same device but on pp+1" without touching strides.
    """

    coords: Coords

    def __post_init__(self) -> None:
        normalized = {str(axis): int(value) for axis, value in dict(self.coords).items()}
        object.__setattr__(self, "coords", MappingProxyType(normalized))

    def with_(self, **axes: int) -> "DeviceCoord":
        merged = dict(self.coords)
        merged.update({str(axis): int(value) for axis, value in axes.items()})
        return DeviceCoord(merged)

    def of(self, axis: AxisName) -> int:
        if axis not in self.coords:
            raise PlacementError(
                f"DeviceCoord has no coordinate for axis {axis!r} "
                f"(has {sorted(self.coords)})"
            )
        return int(self.coords[axis])

    def _key(self) -> Tuple[Tuple[str, int], ...]:
        return tuple(sorted(self.coords.items()))

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, DeviceCoord) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        inner = ",".join(f"{axis}={value}" for axis, value in self._key())
        return f"DeviceCoord({inner})"


# ---------------------------------------------------------------------------
# PlacementPolicy — the named, swappable home of BUG_LEDGER Class B 10d
# ---------------------------------------------------------------------------


class WorkSpread(Enum):
    """How many devices of a stage host one WorkItem's ops."""

    CLUSTER_RANK_0 = auto()  #: one device: cluster rank 0 of the stage
    PER_CLUSTER_RANK = auto()  #: one device per cluster rank


@dataclass(frozen=True)
class PlacementPolicy:
    """Which WorkKinds are pinned to cluster rank 0 and which are replicated.

    This is the NAMED, SWAPPABLE POLICY required by INTERFACES §0.2 for
    BUG_LEDGER Class B item 10d ("softmax pinned to cluster rank 0"). The
    legacy default is :data:`LEGACY_PLACEMENT`; a reviewer flips 10d by
    constructing ``PlacementPolicy(name="softmax_sharded", spread_by_kind={...
    WorkKind.SOFTMAX: WorkSpread.PER_CLUSTER_RANK ...})`` and passing it to
    :class:`Placement` — with no edit inside ``build.py``.
    """

    name: str
    spread_by_kind: Mapping[WorkKind, WorkSpread]

    def __post_init__(self) -> None:
        missing = [kind for kind in WorkKind if kind not in self.spread_by_kind]
        if missing:
            raise PlacementError(
                f"PlacementPolicy {self.name!r} does not declare a spread for "
                f"{[kind.name for kind in missing]}"
            )
        object.__setattr__(
            self, "spread_by_kind", MappingProxyType(dict(self.spread_by_kind))
        )

    def spread_for(self, kind: WorkKind) -> WorkSpread:
        return self.spread_by_kind[kind]


#: Today's placement, verbatim (``pipeline_fine._clone`` :409-457):
#:
#: * ``EMBEDDING``  -> the ``else`` branch, which kept the coarse ``hw_id``
#:   (stage 0) as the device id, i.e. cluster rank 0 of stage 0;
#: * ``SOFTMAX``    -> ``_hw_id_for_rank(obj.hw_id, 0)`` — Class B 10d;
#: * ``LAYER`` / ``RECOMPUTE`` -> ``_expand_transformer_node``, one chain per
#:   cluster rank;
#: * ``OPTIMIZER``  -> ``for tp_rank in range(par_degree)`` — one node per rank.
LEGACY_PLACEMENT = PlacementPolicy(
    name="legacy",
    spread_by_kind={
        WorkKind.EMBEDDING: WorkSpread.CLUSTER_RANK_0,
        WorkKind.SOFTMAX: WorkSpread.CLUSTER_RANK_0,
        WorkKind.LAYER: WorkSpread.PER_CLUSTER_RANK,
        WorkKind.RECOMPUTE: WorkSpread.PER_CLUSTER_RANK,
        WorkKind.OPTIMIZER: WorkSpread.PER_CLUSTER_RANK,
    },
)


# ---------------------------------------------------------------------------
# Layout helpers (all stride math delegated to RankLayout)
# ---------------------------------------------------------------------------


def canonical_layout(axis_sizes: Mapping[AxisName, int]) -> RankLayout:
    """The canonical row-major layout over :data:`CANONICAL_AXES`.

    Used when a workload carries no network layout at all (unit tests, the
    legacy ``layout is None`` path of ``pipeline_fine._hw_id_for_rank``, whose
    flat ``stage * par_degree + tp_rank`` formula is *exactly* this layout's
    linearization because pp's canonical stride is ``tp*cp*ep``).

    The strides come from :meth:`RankLayout.subset`, which is the one
    implementation of row-major stride assignment — this function performs no
    arithmetic of its own.
    """
    sizes = {axis: max(1, int(axis_sizes.get(axis, 1))) for axis in CANONICAL_AXES}
    seed = RankLayout(
        axis_order=CANONICAL_AXES,
        axis_sizes=sizes,
        axis_strides={},
    )
    return seed.subset(CANONICAL_AXES)


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


class Placement:
    """``WorkItem -> DeviceCoord(s)``. THE single placement authority.

    No consumer downstream may re-derive a stage from an ``hw_id`` or a device
    from an op name: everything that needs a device asks this object.
    """

    def __init__(
        self,
        fw: FrozenWorkload,
        granularity: Granularity,
        layers: "LayerAssignment",
        *,
        policy: PlacementPolicy = LEGACY_PLACEMENT,
    ) -> None:
        if not isinstance(granularity, Granularity):
            raise PlacementError(
                f"granularity must be a Granularity (got {type(granularity).__name__})"
            )
        degrees = fw.spec.degrees
        full = fw.spec.layout
        if full is None or not full.axis_order:
            full = canonical_layout(
                {
                    **axis_sizes_from(degrees),
                }
            )

        # Legacy ``_FineExpander._configure_rank_layout`` (:245-259): the layout
        # is authoritative for the axis sizes, the degrees are the fallback for
        # axes the layout does not carry, and the cluster product must agree.
        sizes = full.axis_sizes
        self._sizes: Dict[AxisName, int] = {}
        for axis in CANONICAL_AXES:
            self._sizes[axis] = (
                max(1, int(sizes[axis])) if axis in sizes else degrees.of(axis)
            )
        cluster = degrees.cluster_size()
        layout_cluster = 1
        for _axis in CLUSTER_AXES:
            layout_cluster *= max(1, int(self._sizes[_axis]))
        if layout_cluster != cluster:
            raise PlacementError(
                "Inconsistent tensor/context/expert parallel factors: layout says "
                + ", ".join(f"{a}={self._sizes[a]}" for a in CLUSTER_AXES) + " "
                f"(product {layout_cluster}) but the degrees say {cluster}"
            )

        self._fw = fw
        self._degrees = degrees
        self._granularity = granularity
        self._layers = layers
        self._policy = policy
        self._full_layout = full

        axes = _GRANULARITY_AXES[granularity]
        self._layout = full if granularity is Granularity.FLAT else full.subset(axes)
        #: the axes a device id ranges over (dp excluded — see module docstring)
        self._device_axes: Tuple[AxisName, ...] = tuple(
            axis for axis in self._layout.axis_order if axis != DP_AXIS
        )
        # THE COMMUNICATOR LAYOUT EXCLUDES dp. ``devices()`` never varies the dp
        # coordinate, so a group built over a dp-carrying layout produced members
        # that are not devices of this program at all (PIPELINE dp:(0,2) against
        # devices (0,1)). ``RankLayout`` always orders axes canonically with dp
        # LAST and assigns row-major strides, so dropping dp is stride-preserving:
        # every device id is unchanged. INTERFACES §3.3 amendment, 2026-07-26.
        self._group_layout = self._layout.subset(self._device_axes)
        #: PIPELINE devices ARE stages, so a "cluster" is one device there.
        self._cluster_size = 1 if granularity is Granularity.PIPELINE else cluster
        self._communicators = CommunicatorFactory(self._group_layout)

    # -- identity ---------------------------------------------------------
    @property
    def granularity(self) -> Granularity:
        return self._granularity

    @property
    def policy(self) -> PlacementPolicy:
        return self._policy

    @property
    def layers(self) -> "LayerAssignment":
        return self._layers

    @property
    def layout(self) -> RankLayout:
        """The layout DEVICE IDS LIVE IN for this granularity (already subset)."""
        return self._layout

    @property
    def full_layout(self) -> RankLayout:
        """The unsubsetted workload layout (diagnostics / emission descriptor)."""
        return self._full_layout

    @property
    def group_layout(self) -> RankLayout:
        """:attr:`layout` WITHOUT ``dp`` — the space communicator members live in.

        Identical device ids to :attr:`layout` (dp is always the outermost
        canonical axis, so removing it preserves every stride), but no
        communicator can span dp: a dp collective has no ``GroupKey``
        (INTERFACES §3.3).
        """
        return self._group_layout

    @property
    def communicators(self) -> CommunicatorFactory:
        """The :class:`~program.groups.CommunicatorFactory` over
        :attr:`group_layout`."""
        return self._communicators

    def cluster_size(self) -> int:
        return self._cluster_size

    def activation_shard_size(self) -> int:
        """How many ways a **per-token activation tensor** is sharded across one
        stage's cluster: ``tp*cp`` at FLAT/BLOCK, ``1`` at PIPELINE.

        This is deliberately NOT :meth:`cluster_size`. A cluster has
        ``tp*cp*ep`` devices, but ``ep`` is **not** a sharding axis for a
        token-indexed tensor: in training every EP rank owns a DISTINCT
        microbatch (``base_timing.py:493-495`` sets ``dp_dense = dp*ep``, so
        ``micro_batch`` already carries the ``/ep``; ``train_timing.py:417-431``
        and ``memory_estimation.py:499-510`` say so in words), and it therefore
        holds a full copy of *its own* residual stream — sequence-sharded by
        ``tp`` (under SP) and ``cp``, and by nothing else. Dividing a value that
        is already one owner's microbatch by ``ep`` a second time is the
        double-division that was BUG_LEDGER **D 10c (ep half)**.

        Axis sizes come from the layout, exactly as :meth:`cluster_size`'s
        cross-check does, so a mesh layout that re-factors the cluster is
        honored rather than second-guessed.
        """
        if self._granularity is Granularity.PIPELINE:
            return 1
        # ``tp * cp`` derived from ``AxisSpec.shards_activations`` rather than
        # written as a literal product — the 10c fact now lives on the axes.
        size = 1
        for _axis in activation_sharding_axes():
            size *= max(1, int(self._sizes[_axis]))
        return max(1, size)

    def num_stages(self) -> int:
        return int(self._degrees.pp)

    # -- device space -----------------------------------------------------
    def devices(self) -> Tuple[DeviceId, ...]:
        """All devices of this granularity, ascending (``linearize`` order)."""
        if not self._device_axes:
            return (DeviceId(0),)
        ranges = [range(self._layout.axis_sizes[axis]) for axis in self._device_axes]
        found = set()
        for combo in itertools.product(*ranges):
            found.add(self._layout.linearize(dict(zip(self._device_axes, combo))))
        return tuple(DeviceId(device) for device in sorted(found))

    def coords_of(self, device: DeviceId) -> DeviceCoord:
        return DeviceCoord(self._layout.coords_of(int(device)))

    def device_of(self, coord: DeviceCoord) -> DeviceId:
        return DeviceId(self._layout.linearize(coord.coords))

    def device_for(self, stage: StageId, cluster_rank: int = 0) -> DeviceId:
        """The device hosting cluster rank ``cluster_rank`` of ``stage``.

        The one and only device formula, and it is not a formula: it is
        :func:`program.layout.cluster_coords` (which decomposes a flat cluster
        rank into tp/cp/ep coordinates, tp fastest) followed by
        :meth:`RankLayout.linearize`.
        """
        rank = int(cluster_rank)
        if rank < 0 or rank >= self._cluster_size:
            raise PlacementError(
                f"cluster_rank {rank} is out of range for cluster_size "
                f"{self._cluster_size}"
            )
        stage_id = int(stage)
        if stage_id < 0 or stage_id >= self.num_stages():
            raise PlacementError(
                f"stage {stage_id} is out of range for pp={self.num_stages()}"
            )
        coords = cluster_coords(
            self._layout.axis_order,
            rank,
            stage_id,
            sizes=self._sizes,
        )
        return DeviceId(self._layout.linearize(coords))

    def stage_device(self, stage: StageId) -> DeviceId:
        """Cluster rank 0 of ``stage`` (at PIPELINE: the stage itself)."""
        return self.device_for(stage, 0)

    def cluster_devices(self, stage: StageId) -> Tuple[DeviceId, ...]:
        """Every device of ``stage``, in cluster-rank order."""
        return tuple(self.device_for(stage, rank) for rank in range(self._cluster_size))

    def cluster_rank_of(self, device: DeviceId) -> int:
        """Inverse of :meth:`device_for`'s cluster component."""
        coords = self._layout.coords_of(int(device))
        for rank in range(self._cluster_size):
            candidate = cluster_coords(
                self._layout.axis_order,
                rank,
                int(coords.get("pp", 0)),
                sizes=self._sizes,
            )
            if all(coords.get(axis, 0) == value for axis, value in candidate.items()):
                return rank
        raise PlacementError(f"Device {device} is not a cluster rank of this layout")

    # -- placement --------------------------------------------------------
    def stage_of(self, work: WorkItem) -> StageId:
        """Verbatim placement rules of ``schedule.py`` :583, :592, :603, :1054.

        Keyed on :class:`~program.work.WorkKind`, never on a name.
        """
        kind = work.kind
        if kind is WorkKind.EMBEDDING:
            return StageId(0)
        if kind is WorkKind.SOFTMAX:
            return StageId(self.num_stages() - 1)
        if kind is WorkKind.LAYER or kind is WorkKind.RECOMPUTE:
            if work.layer is None:
                raise PlacementError(f"{kind.name} WorkItem carries no layer: {work!r}")
            return StageId(int(self._layers.stage_of(work.layer)))
        if kind is WorkKind.OPTIMIZER:
            if work.stage is None:
                raise PlacementError(
                    "OPTIMIZER WorkItem carries no stage; its identity IS per-stage "
                    "(BUG_LEDGER 10b: one fused node per stage, priced over the "
                    "layers that stage owns)"
                )
            return StageId(int(work.stage))
        raise PlacementError(f"Unhandled WorkKind {kind!r}")

    def devices_for(self, work: WorkItem) -> Tuple[DeviceId, ...]:
        """The devices this WorkItem's chains are instantiated on.

        PIPELINE: always one device (``cluster_size == 1``). FLAT/BLOCK: one
        device per cluster rank, unless :attr:`policy` pins the kind to cluster
        rank 0 (Class B 10d for ``SOFTMAX``; embedding at device 0).
        """
        stage = self.stage_of(work)
        if self._policy.spread_for(work.kind) is WorkSpread.CLUSTER_RANK_0:
            return (self.device_for(stage, 0),)
        return self.cluster_devices(stage)

    def coords_for(self, work: WorkItem) -> Tuple[DeviceCoord, ...]:
        return tuple(self.coords_of(device) for device in self.devices_for(work))

    def devices_for_sync(self, req: "SyncRequirement") -> Tuple[DeviceId, ...]:
        """Resolve a :class:`~program.work.SyncRequirement`'s ``spread``.

        ``STAGE`` -> the stage device; ``CLUSTER_RANK_0`` -> the first device of
        ``place_on``; ``PER_CLUSTER_RANK`` -> **every device of ``place_on``'s
        stage**.

        This REPLACES ``local_hw_id`` + ``propagate_local_hw_ids`` entirely: the
        host is declared on the requirement, so nothing is back-propagated from
        a parent that happened to be visited first.

        AMENDMENT to INTERFACES §3.2 (dated 2026-07-26): ``PER_CLUSTER_RANK``
        resolves against ``cluster_devices(stage_of(place_on))``, NOT against
        ``devices_for(place_on)``. The two differ whenever :attr:`policy` pins
        ``place_on``'s kind to cluster rank 0 — which is exactly ZeRO-3 row S7,
        whose ``zero3_transformer_gather`` (``tp_shard=True``, therefore
        ``PER_CLUSTER_RANK``) is placed on ``EMBEDDING/FORWARD``, and row S13,
        placed on ``SOFTMAX/BACKWARD``. Delegating to ``devices_for`` collapsed
        the requirement to ONE device while legacy
        ``_ensure_zero3_per_rank_edges`` builds ``hw_ids`` for every
        ``par_degree`` rank (``pipeline_fine.py:471-480``). The SPREAD of a
        collective is a property of the collective, not of the placement of the
        work it hangs off.
        """
        spread = req.spread
        place_on = req.place_on
        if spread is SyncSpread.STAGE:
            return (self.stage_device(self.stage_of(place_on)),)
        if spread is SyncSpread.CLUSTER_RANK_0:
            return (self.devices_for(place_on)[0],)
        if spread is SyncSpread.PER_CLUSTER_RANK:
            devices = self.cluster_devices(self.stage_of(place_on))
            if len(devices) < self._cluster_size:
                raise PlacementError(
                    f"SyncSpread.PER_CLUSTER_RANK for {req.key!r} resolved to "
                    f"{len(devices)} device(s) but cluster_size is "
                    f"{self._cluster_size}; a per-cluster-rank collective must "
                    "exist on every cluster rank of its stage"
                )
            return devices
        raise PlacementError(f"Unhandled SyncSpread {spread!r}")

    def same_stage(self, a: WorkItem, b: WorkItem) -> bool:
        return self.stage_of(a) == self.stage_of(b)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"Placement(granularity={self._granularity.name}, "
            f"devices={len(self.devices())}, cluster_size={self._cluster_size}, "
            f"policy={self._policy.name!r})"
        )


# ---------------------------------------------------------------------------
# Expansion (INTERFACES §3.4)
# ---------------------------------------------------------------------------


class StepKind(Enum):
    COMPUTE = auto()
    COMM = auto()
    #: a zero-duration local rendezvous (the MoE hot/cold routing join,
    #: ``block_program._make_local_join_node`` :237-257)
    JOIN = auto()


@dataclass(frozen=True)
class StepRef:
    """A step of a sibling chain of the SAME expansion.

    ``index`` indexes ``ExpandedChain.steps`` of the chain on ``device``.
    Only valid within the tuple one :meth:`BlockExpander.expand` call returns.
    """

    device: DeviceId
    index: int


@dataclass(frozen=True)
class ComputeStep:
    """One compute op of an expanded chain."""

    index: int
    name: str
    duration: float
    deps: Tuple[int, ...] = ()  #: local step indices within the same chain
    kind: StepKind = StepKind.COMPUTE
    entry_name: Optional[str] = None  #: BlockTemplate entry (None at PIPELINE)
    mem_kind: Any = None
    param_gather: bool = False
    recompute: bool = False


@dataclass(frozen=True)
class CommStep:
    """One block-template collective of an expanded chain."""

    index: int
    comm_key: CommKey
    spec: CommSpec
    deps: Tuple[int, ...] = ()  #: local step indices within the same chain
    kind: StepKind = StepKind.COMM
    #: steps of SIBLING chains that additionally depend on this one. The MoE
    #: ``residual_p2p`` cold->hot edge is the only producer today
    #: (``block_program.py:366-373``: ``residual_edge.add_child(hot_join)``).
    extra_consumers: Tuple[StepRef, ...] = ()
    #: the L1 ``OverlapDecl`` this collective carries, if any. L2 only ATTACHES
    #: it; L4 realizes the split (INTERFACES §4.5).
    overlap: Optional["OverlapDecl"] = None

    @property
    def placement(self) -> str:
        return self.spec.placement


ChainStep = Union[ComputeStep, CommStep]


@dataclass(frozen=True)
class ExpandedChain:
    """One WorkItem, one device: the ordered steps and the identity of its ends."""

    work: WorkItem
    device: DeviceId
    cluster_rank: int
    steps: Tuple[ChainStep, ...]
    entry: int  #: index into ``steps``
    exit: int  #: index into ``steps``

    def __post_init__(self) -> None:
        if not self.steps:
            raise PlacementError(f"Expansion of {self.work!r} produced no steps")
        for position, step in enumerate(self.steps):
            if step.index != position:
                raise PlacementError(
                    f"ExpandedChain step {position} carries index {step.index}"
                )
            for dep in step.deps:
                if dep < 0 or dep >= len(self.steps):
                    raise PlacementError(
                        f"ExpandedChain step {position} depends on out-of-range {dep}"
                    )

    def compute_steps(self) -> Tuple[ComputeStep, ...]:
        return tuple(step for step in self.steps if isinstance(step, ComputeStep))

    def comm_steps(self) -> Tuple[CommStep, ...]:
        return tuple(step for step in self.steps if isinstance(step, CommStep))

    @property
    def entry_step(self) -> ChainStep:
        return self.steps[self.entry]

    @property
    def exit_step(self) -> ChainStep:
        return self.steps[self.exit]


# ---------------------------------------------------------------------------
# Comm-key grouping helpers, hoisted out of ``build_block_root``'s closure so
# the FLAT path can call them (INTERFACES §3.4 — this hoist is the single
# biggest blocker in ext_moe_flat.md and it is discharged by hoisting, not by
# a new branch).
# ---------------------------------------------------------------------------


def split_comm_keys(
    comm_keys: Sequence[CommKey],
    specs: Mapping[CommKey, CommSpec],
) -> Tuple[Tuple[CommKey, ...], Tuple[CommKey, ...]]:
    """Partition ``comm_keys`` into (pre, post) by ``CommSpec.placement``.

    Port of ``block_program._split_comm_keys`` (:148-169) with the missing-key
    silence removed: a key with no spec raises here rather than defaulting to
    ``"post"`` and failing later at edge creation.
    """
    pre: List[CommKey] = []
    post: List[CommKey] = []
    for key in comm_keys:
        spec = specs[key] if key in specs else None
        if spec is None:
            raise PlacementError(
                f"Block template references comm key {key!r} which is not registered"
            )
        if spec.placement == "pre":
            pre.append(key)
        elif spec.placement == "post":
            post.append(key)
        else:  # pragma: no cover - CommSpec.__post_init__ already rejects these
            raise PlacementError(
                f"Unsupported comm placement {spec.placement!r} for key {key!r}"
            )
    return tuple(pre), tuple(post)


def comm_parallel_groups(
    comm_keys: Sequence[CommKey],
    specs: Mapping[CommKey, CommSpec],
) -> Tuple[Tuple[CommKey, ...], ...]:
    """Group comm keys by ``CommSpec.parallel_group``, first-encounter order.

    Verbatim port of ``block_program._comm_parallel_groups`` (:171-186); keys
    without a parallel group each form their own singleton group.
    """
    grouped: List[List[CommKey]] = []
    index: Dict[str, int] = {}
    for key in comm_keys:
        spec = specs[key]
        group = spec.parallel_group
        if not group:
            grouped.append([key])
            continue
        if group not in index:
            index[group] = len(grouped)
            grouped.append([key])
        else:
            grouped[index[group]].append(key)
    return tuple(tuple(group) for group in grouped)


# ---------------------------------------------------------------------------
# BlockExpander
# ---------------------------------------------------------------------------

#: Diagnostic op-name stems per WorkKind x Direction. Names are NOT part of
#: canonical equivalence (``equiv/canonical.py:20-27``); nothing dispatches on
#: them. They exist so a dumped Program is readable.
_WORK_NAME: Mapping[Tuple[WorkKind, Direction], str] = MappingProxyType(
    {
        (WorkKind.EMBEDDING, Direction.FORWARD): "embedding",
        (WorkKind.EMBEDDING, Direction.BACKWARD): "embedding_b",
        (WorkKind.SOFTMAX, Direction.FORWARD): "linear_softmax",
        (WorkKind.SOFTMAX, Direction.BACKWARD): "linear_softmax_b",
        (WorkKind.LAYER, Direction.FORWARD): "layer",
        (WorkKind.LAYER, Direction.BACKWARD): "layer_b",
        (WorkKind.RECOMPUTE, Direction.FORWARD): "layer_recompute",
        (WorkKind.RECOMPUTE, Direction.BACKWARD): "layer_recompute_b",
        (WorkKind.OPTIMIZER, Direction.FORWARD): "optimizer",
        (WorkKind.OPTIMIZER, Direction.BACKWARD): "optimizer",
    }
)


def work_name(work: WorkItem) -> str:
    """A deterministic, human-readable stem for ``work``'s ops."""
    stem = _WORK_NAME[(work.kind, work.direction)]
    parts: List[str] = [stem]
    if work.stage is not None:
        parts.append(f"stage{work.stage}")
    if work.layer is not None:
        parts.append(f"l{work.layer}")
    if work.microbatch is not None:
        parts.append(f"mb{work.microbatch}")
    return "_".join(parts)


class BlockExpander:
    """``WorkItem -> ExpandedChain(s)``: the placed op structure of one unit of
    work, before any dependency between units exists.

    Replaces ``_FineExpander._expand_transformer_node``
    (``pipeline_fine.py:512-724``), ``build_block_root``'s per-rank loop
    (``block_program.py:388-493``) and ``pipeline_coarse``'s one-op-per-event
    projection with a single implementation parameterized by
    :class:`Granularity`.
    """

    def __init__(
        self,
        fw: FrozenWorkload,
        placement: Placement,
        overlap: Optional["OverlapPolicy"] = None,
        routing: Optional["MoERoutingPolicy"] = None,
    ) -> None:
        self._fw = fw
        self._placement = placement
        self._overlap = overlap
        self._routing = routing

    # -- comm resolution ---------------------------------------------------
    def _specs_for(self, work: WorkItem) -> CommSpecTable:
        """The comm table ``work``'s chain resolves its keys against.

        Block-template comm keys are named PER TEMPLATE, so the dense and MoE
        tables must be consulted separately: on the ``moe:ep2`` golden rows
        ``ep_dense_sync_layernorm1_backward`` is 337,641,472 bytes dense and
        67,141,632 MoE. Reading them from one flat union picked whichever
        template registered first and was a 5.03x byte error on both MoE rows
        (INTERFACES §1.6 / §3.4 amendment, 2026-07-26).
        """
        if work.layer is None:
            raise PlacementError(
                f"{work.kind.name} WorkItem has no layer, so it resolves no block "
                f"comm keys: {work!r}"
            )
        return self._fw.spec.block_comm(work.layer)

    # -- public ------------------------------------------------------------
    def expand(self, work: WorkItem) -> Tuple[ExpandedChain, ...]:
        """One chain per device of ``placement.devices_for(work)``."""
        if self._placement.granularity is Granularity.PIPELINE:
            return self._expand_single(work)
        if work.kind is WorkKind.LAYER or work.kind is WorkKind.RECOMPUTE:
            return self._expand_template(work)
        return self._expand_single(work)

    # -- single-step expansion (PIPELINE, and the non-layer FLAT/BLOCK kinds) --
    def _expand_single(self, work: WorkItem) -> Tuple[ExpandedChain, ...]:
        """One ComputeStep per device.

        At PIPELINE this is every WorkItem (``duration = durations[duration_key]``).
        At FLAT it is embedding / softmax / optimizer, which the legacy
        flattener also cloned as single nodes — ``pipeline_fine.py:409-457``.

        The :class:`LayerAssignment` is handed to ``duration_for`` because the
        fused per-stage OPTIMIZER node is priced from the layers ITS OWN stage
        owns (BUG_LEDGER 10b), and that split is remainder-first, hence uneven.
        """
        duration = float(duration_for(work, self._fw, self._placement.layers))
        kind_of_memory = mem_kind(work)
        stem = work_name(work)
        devices = self._placement.devices_for(work)
        chains: List[ExpandedChain] = []
        for cluster_rank, device in enumerate(devices):
            name = stem if len(devices) == 1 else f"{stem}_rank{cluster_rank}"
            step = ComputeStep(
                index=0,
                name=name,
                duration=duration,
                deps=(),
                mem_kind=kind_of_memory,
                recompute=(work.kind is WorkKind.RECOMPUTE),
            )
            chains.append(
                ExpandedChain(
                    work=work,
                    device=device,
                    cluster_rank=cluster_rank,
                    steps=(step,),
                    entry=0,
                    exit=0,
                )
            )
        return tuple(chains)

    # -- block-template expansion (FLAT / BLOCK) ----------------------------
    def _expand_template(self, work: WorkItem) -> Tuple[ExpandedChain, ...]:
        if work.layer is None:
            raise PlacementError(f"{work.kind.name} WorkItem carries no layer: {work!r}")
        template = self._fw.spec.block_template(work.layer)
        if not template.entries:
            raise PlacementError(
                f"BlockTemplate for layer {work.layer} has no GEMM entries"
            )
        direction = work.direction
        entries = (
            template.entries
            if direction is Direction.FORWARD
            else tuple(reversed(template.entries))
        )

        specs = self._specs_for(work)
        devices = self._placement.devices_for(work)
        #: (parallel_group, hot device) -> the hot rank's join step
        joins: Dict[Tuple[str, DeviceId], StepRef] = {}
        chains: List[ExpandedChain] = []
        for cluster_rank, device in enumerate(devices):
            steps = self._expand_chain(
                work, device, cluster_rank, entries, direction, joins, specs
            )
            chains.append(
                ExpandedChain(
                    work=work,
                    device=device,
                    cluster_rank=cluster_rank,
                    steps=steps,
                    entry=0,
                    exit=len(steps) - 1,
                )
            )
        return tuple(chains)

    def _expand_chain(
        self,
        work: WorkItem,
        device: DeviceId,
        cluster_rank: int,
        entries: Sequence[GemmEntry],
        direction: Direction,
        joins: Dict[Tuple[str, DeviceId], StepRef],
        specs: CommSpecTable,
    ) -> Tuple[ChainStep, ...]:
        steps: List[ChainStep] = []
        previous: Optional[int] = None
        direction_name = "forward" if direction is Direction.FORWARD else "backward"

        for position, entry in enumerate(entries):
            cfg = entry.direction(direction_name)
            if cfg.duration is None:
                raise PlacementError(
                    f"Missing duration for transformer entry {entry.name!r} in "
                    f"direction {direction_name!r}"
                )
            pre_keys, post_keys = split_comm_keys(cfg.comm_keys, specs)

            # placement="pre": the collective feeds the GEMM. The FLAT path
            # ignored this (pipeline_fine.py:577-589); here it is honored.
            for key in pre_keys:
                previous = self._append_comm(steps, key, previous, specs)

            compute_index = len(steps)
            steps.append(
                ComputeStep(
                    index=compute_index,
                    name=f"{entry.name}_{direction_name}_{work_name(work)}_rank{cluster_rank}",
                    duration=float(cfg.duration),
                    deps=() if previous is None else (previous,),
                    entry_name=entry.name,
                    mem_kind=mem_kind_from_op_name(entry.name),
                    param_gather=(position == 0),
                    recompute=(work.kind is WorkKind.RECOMPUTE),
                )
            )
            previous = compute_index

            for group in comm_parallel_groups(post_keys, specs):
                head = specs[group[0]]
                if head.parallel_group and head.moe_component:
                    previous = self._append_moe_group(
                        steps, group, previous, device, joins, specs
                    )
                else:
                    for key in group:
                        previous = self._append_comm(steps, key, previous, specs)

        if not steps:
            raise PlacementError("Transformer expansion produced no steps")
        return tuple(steps)

    def _append_comm(
        self,
        steps: List[ChainStep],
        key: CommKey,
        previous: Optional[int],
        specs: CommSpecTable,
    ) -> int:
        spec = specs.require(key)
        index = len(steps)
        steps.append(
            CommStep(
                index=index,
                comm_key=key,
                spec=spec,
                deps=() if previous is None else (previous,),
                overlap=self._declare_overlap(spec),
            )
        )
        return index

    def _declare_overlap(self, spec: CommSpec) -> Optional["OverlapDecl"]:
        if self._overlap is None:
            return None
        return self._overlap.declare(spec, self._fw)

    # -- the MoE hot/cold routing join --------------------------------------
    def _append_moe_group(
        self,
        steps: List[ChainStep],
        group: Sequence[CommKey],
        previous: Optional[int],
        device: DeviceId,
        joins: Dict[Tuple[str, DeviceId], StepRef],
        specs: CommSpecTable,
    ) -> int:
        """Port of ``block_program._attach_moe_parallel_post_group`` (:285-374).

        The two rank formulas it used are gone:

        * ``_moe_hot_rank`` (zero the routing axes, then re-linearize by hand)
          becomes ``min(CommunicatorFactory.members(routing_axes, device))`` —
          which is the same device because zeroing the spanned coordinates
          minimizes a row-major linearization. This makes P4 ("the hot rank of
          a routing group is ``min(members)``, so hot-before-cold is
          guaranteed") a construction, not the runtime assertion at :356-361.
        * ``_moe_parallel_token`` (a hand-built dict key per routing mode)
          becomes the hot device itself, which induces exactly the same
          partition of ranks.
        """
        base_key: Optional[CommKey] = None
        residual_key: Optional[CommKey] = None
        parallel_group: Optional[str] = None
        routing_mode: Optional[str] = None

        from program.policies.routing import MoEComponent, component_of, routing_for_mode

        for key in group:
            spec = specs.require(key)
            parallel_group = parallel_group or spec.parallel_group
            routing_mode = routing_mode or spec.moe_routing_mode
            # The component vocabulary is a CLOSED enum owned by the routing
            # policy (INTERFACES §2.6); this loop never compares a string.
            component = component_of(spec)
            if component is MoEComponent.BASE_ALL_TO_ALL:
                base_key = key
            elif component is MoEComponent.RESIDUAL_P2P:
                residual_key = key
            else:
                raise PlacementError(
                    f"Comm key {key!r} is in parallel group "
                    f"{spec.parallel_group!r} but declares no MoE component"
                )
        if not parallel_group or not routing_mode:
            raise PlacementError(
                f"MoE parallel comm group {tuple(group)} is missing required metadata "
                "(parallel_group / moe_routing_mode)"
            )
        if self._routing is None:
            raise PlacementError(
                f"Comm group {tuple(group)} declares moe_routing_mode "
                f"{routing_mode!r} but no MoERoutingPolicy was supplied to the "
                "BlockExpander"
            )
        # routing_for_mode is the ONE place a routing-mode string is interpreted.
        if routing_for_mode(routing_mode) != self._routing:
            raise PlacementError(
                f"Comm group {tuple(group)} declares moe_routing_mode "
                f"{routing_mode!r} but the supplied MoERoutingPolicy is "
                f"{self._routing.name!r}"
            )

        axes = tuple(self._routing.routing_axes())
        members = self._placement.communicators.members(axes, device)
        hot_device = DeviceId(self._routing.hot_device(members))
        token = (parallel_group, hot_device)

        base_index: Optional[int] = None
        if base_key is not None:
            base_index = self._append_comm(steps, base_key, previous, specs)

        residual_index: Optional[int] = None
        extra_consumers: Tuple[StepRef, ...] = ()
        if device != hot_device and residual_key is not None:
            hot_join = joins.get(token)
            if hot_join is None:
                raise PlacementError(
                    f"No hot-rank join for MoE parallel group {parallel_group!r} at "
                    f"device {hot_device} (routing axes {canonical_axis_label(axes)}); "
                    "hot ranks are min(members) and must expand first"
                )
            residual_index = len(steps)
            steps.append(
                CommStep(
                    index=residual_index,
                    comm_key=residual_key,
                    spec=specs.require(residual_key),
                    deps=() if previous is None else (previous,),
                    extra_consumers=(hot_join,),
                    overlap=self._declare_overlap(specs.require(residual_key)),
                )
            )

        join_deps: List[int] = []
        if base_index is not None:
            join_deps.append(base_index)
        if residual_index is not None:
            join_deps.append(residual_index)
        if not join_deps and previous is not None:
            join_deps.append(previous)

        join_index = len(steps)
        steps.append(
            ComputeStep(
                index=join_index,
                name=f"{parallel_group}_join_device{int(device)}",
                duration=0.0,
                deps=tuple(join_deps),
                kind=StepKind.JOIN,
            )
        )
        if device == hot_device:
            joins[token] = StepRef(device=device, index=join_index)
        return join_index


__all__ = [
    "BlockExpander",
    "ChainStep",
    "CommStep",
    "ComputeStep",
    "CLUSTER_AXES",
    "CommunicatorFactory",
    "DeviceCoord",
    "ExpandedChain",
    "Granularity",
    "GroupError",
    "LEGACY_PLACEMENT",
    "Placement",
    "PlacementError",
    "PlacementPolicy",
    "StepKind",
    "StepRef",
    "WorkSpread",
    "canonical_layout",
    "comm_parallel_groups",
    "split_comm_keys",
    "work_name",
]
