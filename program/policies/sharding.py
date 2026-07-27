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

"""L1 — sharding policies (INTERFACES.md §2.4).

Four small classes replace the 12-literal-key DP/ZeRO/EP cascade at
``schedule.py:778-1046`` (~270 LOC) together with ``attach_parallel_edge``
(:553-567) and its ``skip_non_comm_children`` / ``skip_comm_children`` boolean
escape hatches. Every emitted :class:`~program.work.SyncRequirement` carries a
TYPED :class:`~program.work.AttachMode`; no boolean ever crosses this boundary.

The row ids in the ``origin=`` fields below (``S1``..``S16``) are the rows of
the completeness table in INTERFACES §2.3, which enumerates *all 16*
create-and-attach operations of the legacy lattice over its 12 comm keys.
``tests/test_policies.py`` re-derives that table against
``schedule.build_pipeline_events`` and fails if any row moves.

**The byte/kind/axis seam.** ``train_timing`` owns the math: sizes, collective
kinds, analytical participant counts and ``ga_required_every_cycle`` come from
:class:`~program.workload.CommSpec` via
:meth:`~program.work.SyncRequirement.from_spec` and are never recomputed,
re-sniffed or name-derived here (this is what deletes the collective-name
sniffing at ``schedule.py:801-803``). The policy decides only EXISTENCE, AXIS
(by declaring which ``CommSpec`` it uses) and ATTACHMENT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Protocol, Sequence, Tuple

from program.policies.gradaccum import GradAccumPolicy
from program.types import CommKey, LayerId, StageId
from program.work import (
    AttachMode,
    Direction,
    SyncKey,
    SyncPhase,
    SyncRequirement,
    SyncSpread,
    VIA_ALL,
    VIA_DATA_FLOW,
    VIA_NON_DATA_FLOW,
    WorkItem,
    WorkKind,
    WorkSet,
    backward_entry,
    is_moe,
)
from program.workload import CommSpec, FrozenWorkload, WorkloadError

__all__ = [
    "StagePartition",
    "ContiguousStages",
    "ShardingContext",
    "ShardingPolicy",
    "NullSharding",
    "DDP",
    "ZeRO1",
    "ZeRO2",
    "ZeRO3",
    "SHARDING_POLICIES",
    "sharding_policy_for",
]


# ---------------------------------------------------------------------------
# Stage partition — the L2/L3 seam, with a transitional default
# ---------------------------------------------------------------------------


class StagePartition(Protocol):
    """The ONE thing L1 needs from L2's placement: do two WorkItems share a
    pipeline stage?

    AMENDMENT to INTERFACES §2.4 (dated 2026-07-26). This protocol used to also
    declare ``stage_of(work: WorkItem)``. That made it *unsatisfiable together
    with* ``LayerAssignment`` (INTERFACES §4.1), whose ``stage_of`` takes a
    ``LayerId`` — and ``Placement.__init__`` consumes a ``LayerAssignment``
    while ``ShardingContext`` consumes a ``StagePartition``, so a single object
    had to be both and no object could. ``same_stage`` is the only member L1
    ever called (``ShardingContext.same_stage``, ``ZeRO3._via``), so the
    protocol shrinks to it and the clash disappears:
    :class:`ContiguousStages` now satisfies both seams at once.

    ``program.placement.Placement`` (P3) satisfies this protocol as-is
    (INTERFACES §3.2 declares ``same_stage``), so P3 may pass a ``Placement``
    here instead.
    """

    def same_stage(self, a: WorkItem, b: WorkItem) -> bool: ...


@dataclass(frozen=True)
class ContiguousStages:
    """The legacy remainder-first layer split, satisfying BOTH stage seams.

    Port of ``schedule.legacy_layers_per_stage`` (:83-90) +
    ``_layer_to_stage`` (:93-100): ``base + 1`` layers for the first
    ``num_layers % pp`` stages.

    It implements:

    * INTERFACES §4.1 ``LayerAssignment`` — :meth:`stage_of` (a ``LayerId``),
      :meth:`layers_of`, :meth:`min_layer` — so it can be passed as
      ``Placement(fw, granularity, layers=...)``; and
    * :class:`StagePartition` — :meth:`same_stage` — so the SAME object can be
      passed as ``ShardingContext(stages=...)``.

    The WorkItem-keyed rule is :meth:`stage_of_work` (renamed from ``stage_of``
    on 2026-07-26 to free ``stage_of`` for the ``LayerAssignment`` signature).
    P4 replaces this class with ``program.schedule.policy.LayerAssignment``, which
    carries the same three ``LayerAssignment`` members.
    """

    stage_of_layer: Tuple[StageId, ...]
    num_stages: int

    @classmethod
    def legacy(cls, num_layers: int, pp: int) -> "ContiguousStages":
        stage_count = max(1, int(pp))
        base = num_layers // stage_count
        remainder = num_layers % stage_count
        mapping: List[StageId] = []
        for stage_idx in range(stage_count):
            count = base + (1 if stage_idx < remainder else 0)
            mapping.extend(StageId(stage_idx) for _ in range(count))
        return cls(stage_of_layer=tuple(mapping), num_stages=stage_count)

    @classmethod
    def contiguous(cls, num_layers: int, pp: int) -> "ContiguousStages":
        """:meth:`legacy` under the contract's name (INTERFACES §4.1
        ``LayerAssignment.contiguous``), so P4's swap is a rename of the type
        only."""
        return cls.legacy(num_layers, pp)

    # -- the LayerAssignment surface (INTERFACES §4.1) ---------------------
    def stage_of(self, layer: LayerId) -> StageId:
        """The stage hosting ``layer``. Keyed on a ``LayerId``, per §4.1."""
        if layer < 0 or layer >= len(self.stage_of_layer):
            raise WorkloadError(
                f"Layer index {layer} is out of bounds for "
                f"{len(self.stage_of_layer)} layers."
            )
        return self.stage_of_layer[layer]

    def layers_of(self, stage: StageId) -> Tuple[LayerId, ...]:
        return tuple(
            layer
            for layer, assigned in enumerate(self.stage_of_layer)
            if int(assigned) == int(stage)
        )

    def min_layer(self, stage: StageId) -> Optional[LayerId]:
        """The optimizer attach rule (§4.1); ``None`` for an empty stage."""
        layers = self.layers_of(stage)
        return layers[0] if layers else None

    # -- the StagePartition surface ----------------------------------------
    def stage_of_work(self, work: WorkItem) -> StageId:
        """Verbatim placement rules of schedule.py:583, :592, :603, :1054.

        Mirrors ``Placement.stage_of`` (which is WorkItem-keyed by §3.2); the
        name differs so this class can also expose ``LayerAssignment.stage_of``.
        """
        if work.kind is WorkKind.EMBEDDING:
            return StageId(0)
        if work.kind is WorkKind.SOFTMAX:
            return StageId(self.num_stages - 1)
        if work.kind in (WorkKind.LAYER, WorkKind.RECOMPUTE):
            return self.stage_of(work.layer)  # type: ignore[arg-type]
        if work.kind is WorkKind.OPTIMIZER:
            return StageId(int(work.stage))  # type: ignore[arg-type]
        raise WorkloadError(f"No stage rule for {work!r}")

    def same_stage(self, a: WorkItem, b: WorkItem) -> bool:
        return self.stage_of_work(a) == self.stage_of_work(b)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardingContext:
    """Everything a :class:`ShardingPolicy` may read. Deliberately small: a
    policy that needs more data needs the data ON a typed object, not a wider
    context."""

    fw: FrozenWorkload
    work: WorkSet
    grad_accum: GradAccumPolicy
    stages: StagePartition
    #: AMENDMENT 2026-07-29 — the SchedulePolicy's global work order.
    #:
    #: A PREFETCH requirement is "issue this gather early, before the work that
    #: consumes it" and the consuming work is *the next one of its kind in the
    #: execution order*. Rows S6/S14 used to leave that to ``via``, i.e. to
    #: whichever R3 edge the schedule produced — so under a backward pass that
    #: walks microbatches ascending one gather preceded nothing at all. The
    #: order is a declared L3 artifact (``Schedule.order``), so the policy reads
    #: it and NAMES the consumer instead. Empty means "no order supplied", and
    #: :meth:`next_after` then declares nothing (the ``via`` behavior).
    order: Tuple[WorkItem, ...] = ()

    def next_after(
        self, hosts: Sequence[WorkItem], kind: WorkKind, direction: Direction
    ) -> Optional[WorkItem]:
        """The first ``(kind, direction)`` item scheduled after every host.

        This is what "prefetch for the next one" means, resolved against the
        order that will actually run rather than against a microbatch-index
        assumption: under GPipe's descending backward it answers ``b - 1`` and
        under an ascending one ``b + 1``, with no branch here.
        """
        if not self.order or not hosts:
            return None
        index = {item: position for position, item in enumerate(self.order)}
        try:
            after = max(index[host] for host in hosts)
        except KeyError:  # pragma: no cover - a host outside the schedule
            return None
        for item in self.order[after + 1 :]:
            if item.kind is kind and item.direction is direction:
                return item
        return None

    def spec_for(self, key: CommKey) -> Optional[CommSpec]:
        """``None`` when the key is absent — an absent key means the upstream
        byte computation produced zero bytes, i.e. the feature is disabled
        (train_timing.py:4453-4507). This is the typed replacement for the
        legacy ``key in spec.comm_metadata`` guards."""
        return self.fw.spec.comm.get(key)

    def emits(self, key: CommKey, microbatch: Optional[int]) -> Optional[CommSpec]:
        """The spec iff this comm key must be emitted for ``microbatch``.

        Fuses the two legacy guards written at every one of the 16 sites:
        ``key in comm_metadata`` and ``should_emit_dp_comm(key, b)``.
        """
        spec = self.spec_for(key)
        if spec is None:
            return None
        return spec if self.grad_accum.emits(spec, microbatch) else None

    # -- placement helpers (delegated; never re-derived) -------------------
    def same_stage(self, a: WorkItem, b: WorkItem) -> bool:
        return self.stages.same_stage(a, b)

    def layer_item(self, direction: Direction, microbatch: int, layer: LayerId) -> WorkItem:
        return self.work.require(
            WorkKind.LAYER, direction, microbatch=microbatch, layer=layer
        )

    def backward_entry(self, microbatch: int, layer: LayerId) -> WorkItem:
        """``_bwd_entry_node`` (schedule.py:682-687): the RECOMPUTE
        rematerialization when materialized, else the backward layer."""
        return backward_entry(
            self.layer_item(Direction.BACKWARD, microbatch, layer), self.work
        )


def _spread_for(spec: CommSpec) -> SyncSpread:
    """Every data-parallel collective exists on EVERY cluster rank of its stage.

    **BUG_LEDGER A2, FIXED (final wave).** This function used to return
    ``PER_CLUSTER_RANK`` only when ``CommSpec.tp_shard`` was set — which
    ``train_timing.py:4495`` sets on ``zero3_transformer_gather`` alone — so
    every other dp collective fell back to ``CLUSTER_RANK_0`` and reproduced the
    legacy ``rank_tails[0]``-only attach (``pipeline_fine.py:616-629``): ONE of
    the ``tp*cp*ep`` cluster ranks bore the whole gradient all-reduce /
    reduce-scatter and the other ``par_degree - 1`` ranks emitted nothing at all.

    A dp collective reduces the gradient shard THAT RANK OWNS over that rank's
    dp replicas. A rank cannot reduce a peer's shard, and the ``par_degree``
    dp-axis communicators are DISJOINT (``et_emit`` interns one wire group per
    owning device: ``gid = device_index + 1``, members
    ``{dp_idx * ns_initial + device}``), so no other rank's collective can stand
    in for the missing one. One instance per (stage, cluster rank) is the only
    consistent reading, and it is what :class:`~program.work.SyncSpread`
    already spells:
    :meth:`program.placement.Placement.devices_for_sync` resolves
    ``PER_CLUSTER_RANK`` against ``cluster_devices(stage_of(place_on))`` (the
    B4 fix, so a requirement hanging off a cluster-rank-0 kind still spans the
    stage) and
    :meth:`program.groups.CommunicatorFactory.groups_for` returns ``None`` per
    instance device for a dp requirement (the B2 fix, so the builder stamps
    ``group=None, is_dp=True`` and dp membership stays an emission-time stamp
    over pre-dp device ids).

    Nothing moves where ``cluster_size == 1``: COARSE placement reports a
    cluster of one (the stage IS the device) and ``tp*cp*ep == 1`` workloads
    have one cluster rank, so ``PER_CLUSTER_RANK`` and ``CLUSTER_RANK_0``
    resolve to the same single device.

    ``CommSpec.tp_shard`` survives as train_timing's declaration that a key's
    BYTES are a per-rank quantity; it is no longer a placement input, because
    the spread of a collective is a property of the collective's communicator,
    not of how its byte count was computed (BUG_LEDGER Class B item 1).
    """
    return SyncSpread.PER_CLUSTER_RANK


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


class ShardingPolicy(Protocol):
    name: str

    def requirements(
        self, work: WorkItem, ctx: ShardingContext
    ) -> Sequence[SyncRequirement]:
        """All sync a single WorkItem induces. Called once per WorkItem.
        MUST be pure and MUST NOT depend on call order (K3)."""

    def workload_requirements(
        self, ctx: ShardingContext
    ) -> Sequence[SyncRequirement]:
        """Sync that belongs to the workload rather than to one WorkItem —
        exactly the two entry gathers, rows S1 and S13. Called once."""


@dataclass(frozen=True)
class NullSharding(ShardingPolicy):
    """``dp <= 1``: no data-parallel collective exists at all.

    Today this is expressed twice — ``if spec.dp > 1`` at schedule.py:797 and
    the ``dp <= 1`` early return inside ``should_emit_dp_comm`` (:257).
    """

    name: str = "none"

    def requirements(self, work: WorkItem, ctx: ShardingContext) -> Tuple[SyncRequirement, ...]:
        return ()

    def workload_requirements(self, ctx: ShardingContext) -> Tuple[SyncRequirement, ...]:
        return ()


#: WorkKind -> the gradient reducer comm key (schedule.py:807, :824, :896).
_REDUCER_KEYS: Mapping[WorkKind, Tuple[CommKey, CommKey]] = {
    #                       dense              moe
    WorkKind.EMBEDDING: ("embedding", "embedding"),
    WorkKind.SOFTMAX: ("softmax", "softmax"),
    WorkKind.LAYER: ("transformer_dense", "transformer_moe"),
}


@dataclass(frozen=True)
class DDP(ShardingPolicy):
    """Plain data parallelism: one gradient reducer per backward work item.

    Rows **S2** (embedding), **S4** (layer), **S9** (softmax). The collective
    KIND is read from the table — ``ALL_REDUCE`` at ``zero_stage < 2``,
    ``REDUCE_SCATTER`` at ``>= 2`` (train_timing.py:4392) — never inferred from
    which keys happen to exist (schedule.py:801-803).
    """

    name: str = "ddp"

    # -- the reducer ------------------------------------------------------
    def reducer_key(self, work: WorkItem, ctx: ShardingContext) -> Optional[CommKey]:
        if work.direction is not Direction.BACKWARD:
            return None
        entry = _REDUCER_KEYS.get(work.kind)
        if entry is None:
            return None
        dense_key, moe_key = entry
        return moe_key if is_moe(work, ctx.fw) else dense_key

    def requirements(self, work: WorkItem, ctx: ShardingContext) -> Tuple[SyncRequirement, ...]:
        key = self.reducer_key(work, ctx)
        if key is None:
            return ()
        spec = ctx.emits(key, work.microbatch)
        if spec is None:
            return ()
        reducer = SyncRequirement.from_spec(
            spec,
            key=SyncKey(key, SyncPhase.GRAD, work.microbatch, work.layer),
            place_on=work,
            mode=AttachMode.AFTER,
            anchors=(work,),
            spread=_spread_for(spec),
            # THE reducer — see SyncRequirement.is_reducer. Everything
            # `after_reducer` adds shares this phase and is NOT one.
            is_reducer=True,
            origin=_REDUCER_ORIGIN[work.kind],
        )
        return (reducer,) + tuple(self.after_reducer(reducer, work, ctx))

    def after_reducer(
        self, reducer: SyncRequirement, work: WorkItem, ctx: ShardingContext
    ) -> Tuple[SyncRequirement, ...]:
        """Hook: what chains onto a gradient reducer. Empty for DDP/ZeRO-1."""
        return ()

    def workload_requirements(self, ctx: ShardingContext) -> Tuple[SyncRequirement, ...]:
        return ()


_REDUCER_ORIGIN: Mapping[WorkKind, str] = {
    WorkKind.EMBEDDING: "S2",
    WorkKind.LAYER: "S4",
    WorkKind.SOFTMAX: "S9",
}


@dataclass(frozen=True)
class ZeRO1(DDP):
    """Structurally identical to :class:`DDP`.

    ZeRO-1 shards optimizer state only, which changes MEMORY, not the
    collective set (train_timing.py:5128-5132). It is a separate class so the
    selection is explicit and so a future optimizer-state model has a home.
    """

    name: str = "zero1"


#: WorkKind -> the ZeRO-2 post-reduce parameter gather (schedule.py:812, :840, :906).
_ZERO2_GATHER_KEYS: Mapping[WorkKind, CommKey] = {
    WorkKind.EMBEDDING: "zero2_embedding_gather",
    WorkKind.LAYER: "zero2_transformer_gather",
    WorkKind.SOFTMAX: "zero2_softmax_gather",
}

_ZERO2_ORIGIN: Mapping[WorkKind, str] = {
    WorkKind.EMBEDDING: "S3",
    WorkKind.LAYER: "S5",
    WorkKind.SOFTMAX: "S10",
}


@dataclass(frozen=True)
class ZeRO2(DDP):
    """:class:`DDP`'s reducers PLUS one ``AFTER(reducer)`` parameter all-gather
    per reducer — rows **S3**, **S5**, **S10**.

    The ``AFTER(SyncKey)`` chaining is what makes the legacy
    ``reducer.add_child(gather)`` a CHECKABLE property (INTERFACES §4.4: the
    referenced requirement must be strictly earlier in ``SyncOrder``) instead of
    a construction-order coincidence.
    """

    name: str = "zero2"

    def after_reducer(
        self, reducer: SyncRequirement, work: WorkItem, ctx: ShardingContext
    ) -> Tuple[SyncRequirement, ...]:
        key = _ZERO2_GATHER_KEYS.get(work.kind)
        if key is None:
            return ()
        spec = ctx.emits(key, work.microbatch)
        if spec is None:
            return ()
        return (
            SyncRequirement.from_spec(
                spec,
                key=SyncKey(key, SyncPhase.GRAD, work.microbatch, work.layer),
                place_on=work,
                mode=AttachMode.AFTER,
                anchors=(reducer.key,),
                spread=_spread_for(spec),
                # This gather broadcasts the parameters the optimizer just
                # wrote, so it cannot precede or overlap the update.
                after_update=True,
                origin=_ZERO2_ORIGIN[work.kind],
            ),
        )


ZERO3_EMBEDDING_KEY: CommKey = "zero3_embedding_gather"
ZERO3_TRANSFORMER_KEY: CommKey = "zero3_transformer_gather"
ZERO3_SOFTMAX_KEY: CommKey = "zero3_softmax_gather"


@dataclass(frozen=True)
class ZeRO3(ZeRO2):
    """:class:`ZeRO2` PLUS the per-layer parameter gathers.

    Rows **S1**, **S6**, **S7**, **S8** (forward), **S13**, **S14**, **S15**,
    **S16** (backward).

    ``prefetch_depth`` IS the ``+/-1`` at schedule.py:873
    (``transformer_nodes[b][layer_idx-1]``) and :1018
    (``_bwd_exit_node(b, layer_idx+1)``). Named here so the ext_zero_policy
    exercise (and an FSDP-style prefetch policy) is a constructor argument, not
    a new lattice.
    """

    name: str = "zero3"
    prefetch_depth: int = 1

    def __post_init__(self) -> None:
        if int(self.prefetch_depth) < 1:
            raise WorkloadError(
                f"ZeRO3.prefetch_depth must be >= 1 (got {self.prefetch_depth})"
            )

    # -- via selection is a RULE, not a per-site literal -------------------
    def _via(self, host: WorkItem, target: WorkItem, ctx: ShardingContext):
        """schedule.py:885-892, as one branch with one docstring.

        Same device: the gather rides ALL of the host's successor edges.
        Cross device: dependencies only apply to the ``cross_layer`` comm
        (``skip_non_comm_children=True``) — cross-layer edges are *always*
        ``CollectiveType.PIPELINE``, so that boolean IS ``VIA_DATA_FLOW``.
        """
        return VIA_ALL if ctx.same_stage(host, target) else VIA_DATA_FLOW

    def _cross_device_forward_hosts(
        self, microbatch: int, ctx: ShardingContext
    ) -> Tuple[WorkItem, ...]:
        """The forward stage boundaries of one microbatch: row S6's anchors."""
        hosts: List[WorkItem] = []
        for layer in range(1, ctx.fw.spec.shape.num_layers):
            host = ctx.layer_item(Direction.FORWARD, microbatch, layer - 1)
            target = ctx.layer_item(Direction.FORWARD, microbatch, layer)
            if not ctx.same_stage(host, target):
                hosts.append(host)
        return tuple(hosts)

    def _backward_host(self, microbatch: int, layer: LayerId, ctx: ShardingContext) -> WorkItem:
        num_layers = ctx.fw.spec.shape.num_layers
        ahead = layer + int(self.prefetch_depth)
        if ahead > num_layers - 1:
            return ctx.work.require(
                WorkKind.SOFTMAX, Direction.BACKWARD, microbatch=microbatch
            )
        return ctx.layer_item(Direction.BACKWARD, microbatch, ahead)

    def _cross_device_backward_hosts(
        self, microbatch: int, ctx: ShardingContext
    ) -> Tuple[WorkItem, ...]:
        """The backward stage boundaries of one microbatch: row S14's anchors."""
        hosts: List[WorkItem] = []
        for layer in reversed(range(ctx.fw.spec.shape.num_layers)):
            host = self._backward_host(microbatch, layer, ctx)
            target = ctx.backward_entry(microbatch, layer)
            if not ctx.same_stage(host, target):
                hosts.append(host)
        return tuple(hosts)

    # -- workload-level entry gathers (S1, S13) ---------------------------
    def workload_requirements(self, ctx: ShardingContext) -> Tuple[SyncRequirement, ...]:
        out: List[SyncRequirement] = []

        # S1 — the forward entry gather becomes the program root.
        spec = ctx.emits(ZERO3_EMBEDDING_KEY, 0)
        if spec is not None:
            entry = ctx.work.require(WorkKind.EMBEDDING, Direction.FORWARD, microbatch=0)
            out.append(
                SyncRequirement.from_spec(
                    spec,
                    key=SyncKey(ZERO3_EMBEDDING_KEY, SyncPhase.FWD_ENTRY, 0),
                    place_on=entry,
                    mode=AttachMode.BEFORE,
                    anchors=(entry,),
                    spread=_spread_for(spec),
                    origin="S1",
                )
            )

        # S13 — the backward entry gather, hung off the LAST forward softmax.
        spec = ctx.emits(ZERO3_SOFTMAX_KEY, 0)
        if spec is not None and ctx.fw.spec.run.include_backward:
            last_mb = ctx.fw.spec.shape.micro_batches - 1
            anchor = ctx.work.require(
                WorkKind.SOFTMAX, Direction.FORWARD, microbatch=last_mb
            )
            out.append(
                SyncRequirement.from_spec(
                    spec,
                    key=SyncKey(ZERO3_SOFTMAX_KEY, SyncPhase.BWD_ENTRY, 0),
                    place_on=ctx.work.require(
                        WorkKind.SOFTMAX, Direction.BACKWARD, microbatch=0
                    ),
                    mode=AttachMode.PARALLEL_TO,
                    anchors=(anchor,),
                    via=VIA_ALL,
                    spread=_spread_for(spec),
                    origin="S13",
                )
            )
        return tuple(out)

    # -- per-WorkItem -----------------------------------------------------
    def requirements(self, work: WorkItem, ctx: ShardingContext) -> Tuple[SyncRequirement, ...]:
        out: List[SyncRequirement] = list(super().requirements(work, ctx))
        handler = _ZERO3_HANDLERS.get((work.kind, work.direction))
        if handler is not None:
            out.extend(handler(self, work, ctx))
        return tuple(out)

    # -- forward ----------------------------------------------------------
    def _forward_embedding(
        self, work: WorkItem, ctx: ShardingContext
    ) -> Tuple[SyncRequirement, ...]:
        """Rows S7 (the layer-0 parameter gather) and S6 (next microbatch's
        embedding gather, riding the forward stage boundaries)."""
        transformer = ctx.emits(ZERO3_TRANSFORMER_KEY, work.microbatch)
        if transformer is None:
            return ()
        out: List[SyncRequirement] = []
        b = work.microbatch

        # S7 — layers shallower than the prefetch depth prefetch off the
        # embedding, not off a previous layer.
        for layer in range(min(int(self.prefetch_depth), ctx.fw.spec.shape.num_layers)):
            target = ctx.layer_item(Direction.FORWARD, b, layer)
            out.append(
                SyncRequirement.from_spec(
                    transformer,
                    key=SyncKey(ZERO3_TRANSFORMER_KEY, SyncPhase.FWD, b, layer),
                    place_on=work,
                    mode=AttachMode.PARALLEL_TO,
                    anchors=(work,),
                    via=self._via(work, target, ctx),
                    spread=_spread_for(transformer),
                    origin="S7",
                )
            )

        # S6 — created once per microbatch, attached only at cross-device
        # boundaries. With no boundary the legacy lattice creates an object it
        # never attaches (INTERFACES §2.3 note 1, Class C); here it simply does
        # not exist.
        if b + 1 < ctx.fw.spec.shape.micro_batches:
            embedding = ctx.emits(ZERO3_EMBEDDING_KEY, b + 1)
            hosts = self._cross_device_forward_hosts(b, ctx)
            if embedding is not None and hosts:
                next_embedding = ctx.next_after(
                    hosts, WorkKind.EMBEDDING, Direction.FORWARD
                )
                out.append(
                    SyncRequirement.from_spec(
                        embedding,
                        key=SyncKey(ZERO3_EMBEDDING_KEY, SyncPhase.FWD, b + 1),
                        place_on=work,
                        mode=AttachMode.PARALLEL_TO,
                        anchors=hosts,
                        via=VIA_NON_DATA_FLOW,
                        # WHO WAITS FOR IT, declared: the NEXT forward embedding
                        # after the boundary this gather is issued at. Under
                        # GPipe that is microbatch b+1's, i.e. exactly what
                        # ``via`` finds (stage 0's next projected item), so
                        # nothing moves; under any other schedule the dependency
                        # is still enforced instead of being lost with the R3
                        # edge it used to ride.
                        consumers=() if next_embedding is None else (next_embedding,),
                        spread=_spread_for(embedding),
                        origin="S6",
                    )
                )
        return tuple(out)

    def _forward_layer(
        self, work: WorkItem, ctx: ShardingContext
    ) -> Tuple[SyncRequirement, ...]:
        """Row S8: layer ``l``'s parameter gather prefetched at layer
        ``l - prefetch_depth``."""
        layer = int(work.layer)
        depth = int(self.prefetch_depth)
        if layer < depth:
            return ()  # handled by S7, off the embedding
        spec = ctx.emits(ZERO3_TRANSFORMER_KEY, work.microbatch)
        if spec is None:
            return ()
        host = ctx.layer_item(Direction.FORWARD, work.microbatch, layer - depth)
        return (
            SyncRequirement.from_spec(
                spec,
                key=SyncKey(ZERO3_TRANSFORMER_KEY, SyncPhase.FWD, work.microbatch, layer),
                place_on=work,
                mode=AttachMode.PARALLEL_TO,
                anchors=(host,),
                via=self._via(host, work, ctx),
                spread=_spread_for(spec),
                origin="S8",
            ),
        )

    def _forward_softmax(
        self, work: WorkItem, ctx: ShardingContext
    ) -> Tuple[SyncRequirement, ...]:
        """Row S11: the softmax parameter gather, prefetched at the last layer.

        Legacy has no cross-device branch here (schedule.py:920-926), so ``via``
        is unconditionally ``VIA_ALL`` — stated, not inferred.
        """
        spec = ctx.emits(ZERO3_SOFTMAX_KEY, work.microbatch)
        if spec is None:
            return ()
        host = ctx.layer_item(
            Direction.FORWARD, work.microbatch, ctx.fw.spec.shape.num_layers - 1
        )
        return (
            SyncRequirement.from_spec(
                spec,
                key=SyncKey(ZERO3_SOFTMAX_KEY, SyncPhase.FWD, work.microbatch),
                place_on=host,
                mode=AttachMode.PARALLEL_TO,
                anchors=(host,),
                via=VIA_ALL,
                spread=_spread_for(spec),
                origin="S11",
            ),
        )

    # -- backward ---------------------------------------------------------
    def _backward_softmax(
        self, work: WorkItem, ctx: ShardingContext
    ) -> Tuple[SyncRequirement, ...]:
        """Row S14: microbatch ``b``'s softmax gather, riding the backward stage
        boundaries. Like S6, an unattachable instance simply does not exist."""
        b = int(work.microbatch)
        if b == 0:
            return ()
        spec = ctx.emits(ZERO3_SOFTMAX_KEY, b)
        if spec is None:
            return ()
        hosts = self._cross_device_backward_hosts(b, ctx)
        if not hosts:
            return ()
        # WHO WAITS FOR IT, declared. The gather is issued at microbatch ``b``'s
        # backward stage boundary and prefetches for the NEXT backward softmax in
        # the schedule's own order — ``b - 1`` under GPipe's descending backward,
        # which is exactly the op ``via`` finds today (stage ``pp-1``'s next
        # projected item), so declaring it moves nothing while making it survive
        # a different ``SchedulePolicy``. Resolving it by microbatch ARITHMETIC
        # instead would close a cycle under an ascending backward (measured).
        consumer = ctx.next_after(hosts, WorkKind.SOFTMAX, Direction.BACKWARD)
        return (
            SyncRequirement.from_spec(
                spec,
                key=SyncKey(ZERO3_SOFTMAX_KEY, SyncPhase.BWD, b),
                place_on=ctx.work.require(
                    WorkKind.SOFTMAX, Direction.FORWARD, microbatch=b
                ),
                mode=AttachMode.PARALLEL_TO,
                anchors=hosts,
                via=VIA_NON_DATA_FLOW,
                consumers=() if consumer is None else (consumer,),
                spread=_spread_for(spec),
                origin="S14",
            ),
        )

    def _backward_layer(
        self, work: WorkItem, ctx: ShardingContext
    ) -> Tuple[SyncRequirement, ...]:
        """Row S15: layer ``l``'s parameter gather prefetched at layer
        ``l + prefetch_depth`` (or at the backward softmax past the end)."""
        spec = ctx.emits(ZERO3_TRANSFORMER_KEY, work.microbatch)
        if spec is None:
            return ()
        layer = int(work.layer)
        host = self._backward_host(work.microbatch, layer, ctx)
        target = ctx.backward_entry(work.microbatch, layer)
        return (
            SyncRequirement.from_spec(
                spec,
                key=SyncKey(ZERO3_TRANSFORMER_KEY, SyncPhase.BWD, work.microbatch, layer),
                place_on=target,
                mode=AttachMode.PARALLEL_TO,
                anchors=(host,),
                via=self._via(host, target, ctx),
                spread=_spread_for(spec),
                origin="S15",
            ),
        )

    def _backward_embedding(
        self, work: WorkItem, ctx: ShardingContext
    ) -> Tuple[SyncRequirement, ...]:
        """Row S16: the embedding parameter gather, prefetched at layer 0's
        backward. Legacy has no cross-device branch (schedule.py:1040-1046)."""
        spec = ctx.emits(ZERO3_EMBEDDING_KEY, work.microbatch)
        if spec is None:
            return ()
        host = ctx.layer_item(Direction.BACKWARD, work.microbatch, 0)
        return (
            SyncRequirement.from_spec(
                spec,
                key=SyncKey(ZERO3_EMBEDDING_KEY, SyncPhase.BWD, work.microbatch),
                place_on=host,
                mode=AttachMode.PARALLEL_TO,
                anchors=(host,),
                via=VIA_ALL,
                spread=_spread_for(spec),
                origin="S16",
            ),
        )


#: (kind, direction) -> the ZeRO-3 handler. A table, so adding a prefetch site
#: is a row, not a branch inside an expansion loop.
_ZERO3_HANDLERS: Mapping[Tuple[WorkKind, Direction], Any] = {
    (WorkKind.EMBEDDING, Direction.FORWARD): ZeRO3._forward_embedding,
    (WorkKind.LAYER, Direction.FORWARD): ZeRO3._forward_layer,
    (WorkKind.SOFTMAX, Direction.FORWARD): ZeRO3._forward_softmax,
    (WorkKind.SOFTMAX, Direction.BACKWARD): ZeRO3._backward_softmax,
    (WorkKind.LAYER, Direction.BACKWARD): ZeRO3._backward_layer,
    (WorkKind.EMBEDDING, Direction.BACKWARD): ZeRO3._backward_embedding,
}


#: zero_stage -> policy. ``>= 3`` clamps to ZeRO-3, matching every legacy
#: ``zero_stage >= 3`` test.
SHARDING_POLICIES: Mapping[int, Any] = {0: DDP, 1: ZeRO1, 2: ZeRO2, 3: ZeRO3}


def sharding_policy_for(run: Any, degrees: Any) -> ShardingPolicy:
    """``{0: DDP, 1: ZeRO1, 2: ZeRO2, 3+: ZeRO3}[run.zero_stage]``.

    ``dp <= 1`` selects :class:`NullSharding`, which emits nothing — today that
    is ``if spec.dp > 1`` at schedule.py:797 *plus* the ``dp <= 1`` early return
    in ``should_emit_dp_comm`` (:257).

    ``not include_backward`` also selects :class:`NullSharding`: the ENTIRE
    lattice lives after ``build_pipeline_events``' inference early return
    (schedule.py:674-680), so a dp>1 inference run emits no gradient sync. One
    documented line replaces that control-flow accident.
    """
    if int(degrees.dp) <= 1 or not run.include_backward:
        return NullSharding()
    stage = int(run.zero_stage)
    if stage < 0:
        raise WorkloadError(f"zero_stage must be >= 0 (got {stage})")
    return SHARDING_POLICIES[min(stage, 3)]()
