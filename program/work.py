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

"""L1 — ``WorkItem`` / ``WorkSet`` / ``SyncRequirement`` (INTERFACES.md §2.1-2.2).

A :class:`WorkItem` says WHAT work exists: no order, no device, no duration.
Derived facts (``is_moe``, block template, duration key, mem kind) are
*functions* of ``(WorkItem, FrozenWorkload)``, never fields — which is what
kills ``FINE_EXPANDER_COPY_ATTRS`` / ``OVERLAP_NODE_COPY_ATTRS``
(pipeline_fine.py:172-203) and, with them, bug **A3**: a split op still refers
to the same ``WorkItem``, so no attribute can be lost in a copy.

A :class:`SyncRequirement` says WHAT collective exists, WHERE it is placed and
WHEN it attaches, using a CLOSED attach vocabulary (:class:`AttachMode`) that
replaces ``attach_parallel_edge``'s two boolean escape hatches
(schedule.py:553-567). The completeness proof for that vocabulary — all 16
create-and-attach operations of the legacy lattice — is INTERFACES §2.3 and is
re-checked by ``tests/test_policies.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from functools import total_ordering
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    FrozenSet,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from timing_model import CollectiveType

from program.types import DP_AXIS, AxisName, CommKey, LayerId, MicroBatch, StageId
from program.workload import CommSpec, FrozenWorkload, ceil_div

if TYPE_CHECKING:  # pragma: no cover - typing only; L3 must not be imported at L1
    from program.schedule.policy import LayerAssignment

try:  # MemKind lives in memory_estimation; keep this module importable without it.
    from memory_estimation import MemKind
except Exception:  # pragma: no cover - defensive
    MemKind = None  # type: ignore[assignment]


__all__ = [
    "WorkError",
    "SyncError",
    "WorkKind",
    "Direction",
    "WorkItem",
    "WorkRef",
    "WorkSet",
    "enumerate_work",
    "duration_key",
    "duration_for",
    "optimizer_duration",
    "stage_layers",
    "mem_kind",
    "is_moe",
    "block_prefix",
    "AttachMode",
    "attach_mode_arity",
    "DepClass",
    "VIA_DATA_FLOW",
    "VIA_NON_DATA_FLOW",
    "VIA_ALL",
    "SyncSpread",
    "ByteSplit",
    "ByteSource",
    "SyncPhase",
    "SyncKey",
    "SyncAnchor",
    "SyncRequirement",
]


class WorkError(ValueError):
    """A WorkSet lookup failed or a WorkSet is inconsistent."""


class SyncError(ValueError):
    """A SyncRequirement is malformed or unresolvable."""


# ---------------------------------------------------------------------------
# WorkItem
# ---------------------------------------------------------------------------


class WorkKind(Enum):
    EMBEDDING = auto()
    LAYER = auto()
    RECOMPUTE = auto()  #: a forward-direction rematerialization inside the backward pass
    SOFTMAX = auto()
    OPTIMIZER = auto()


class Direction(Enum):
    FORWARD = auto()
    BACKWARD = auto()


_UNSET = -1  #: sort-key stand-in for an absent Optional field


@total_ordering
@dataclass(frozen=True)
class WorkItem:
    """WHAT work exists. No order, no device, no duration.

    Frozen and totally ordered, so a WorkItem IS its own reference
    (``WorkRef = WorkItem``) and can key dicts / sort deterministically.

    Ordering uses :meth:`sort_key` — field order with ``None`` mapped to ``-1``
    — rather than ``dataclass(order=True)``, which cannot compare ``Enum``
    fields or ``None`` against ``int``. It is the tie-break of last resort and
    must never be load-bearing.
    """

    kind: WorkKind
    direction: Direction
    microbatch: Optional[MicroBatch] = None  #: None only for OPTIMIZER
    layer: Optional[LayerId] = None  #: None for EMBEDDING/SOFTMAX/OPTIMIZER
    #: SET ONLY for OPTIMIZER, whose identity is per-stage; None otherwise
    #: (L2 derives every other placement).
    stage: Optional[StageId] = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, WorkKind):
            raise WorkError("WorkItem.kind must be a WorkKind")
        if not isinstance(self.direction, Direction):
            raise WorkError("WorkItem.direction must be a Direction")
        if self.kind is WorkKind.OPTIMIZER:
            if self.stage is None:
                raise WorkError("WorkItem(OPTIMIZER) requires a stage")
            if self.microbatch is not None or self.layer is not None:
                raise WorkError("WorkItem(OPTIMIZER) carries neither microbatch nor layer")
        else:
            if self.stage is not None:
                raise WorkError(
                    f"WorkItem({self.kind.name}) must not carry a stage "
                    "(placement is derived at L2)"
                )
            if self.microbatch is None:
                raise WorkError(f"WorkItem({self.kind.name}) requires a microbatch")
        if self.kind in (WorkKind.LAYER, WorkKind.RECOMPUTE) and self.layer is None:
            raise WorkError(f"WorkItem({self.kind.name}) requires a layer")
        if self.kind in (WorkKind.EMBEDDING, WorkKind.SOFTMAX) and self.layer is not None:
            raise WorkError(f"WorkItem({self.kind.name}) must not carry a layer")
        if self.kind is WorkKind.RECOMPUTE and self.direction is not Direction.FORWARD:
            raise WorkError("WorkItem(RECOMPUTE) is a forward-direction rematerialization")

    def sort_key(self) -> Tuple[int, int, int, int, int]:
        return (
            self.kind.value,
            self.direction.value,
            _UNSET if self.microbatch is None else int(self.microbatch),
            _UNSET if self.layer is None else int(self.layer),
            _UNSET if self.stage is None else int(self.stage),
        )

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, WorkItem):
            return NotImplemented
        return self.sort_key() < other.sort_key()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        parts = [self.kind.name, self.direction.name[:3]]
        if self.microbatch is not None:
            parts.append(f"b{self.microbatch}")
        if self.layer is not None:
            parts.append(f"l{self.layer}")
        if self.stage is not None:
            parts.append(f"s{self.stage}")
        return "W(" + ",".join(parts) + ")"


WorkRef = WorkItem


@dataclass(frozen=True)
class WorkSet:
    """The complete set of work for one workload, ordered by
    :meth:`WorkItem.sort_key`."""

    items: Tuple[WorkItem, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.items, key=WorkItem.sort_key))
        if len(set(ordered)) != len(ordered):
            raise WorkError("WorkSet contains duplicate WorkItems")
        object.__setattr__(self, "items", ordered)
        object.__setattr__(self, "_index", frozenset(ordered))

    def __iter__(self) -> Iterator[WorkItem]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __contains__(self, item: Any) -> bool:
        return item in self.__dict__["_index"]

    def get(
        self,
        kind: WorkKind,
        direction: Direction,
        *,
        microbatch: Optional[MicroBatch] = None,
        layer: Optional[LayerId] = None,
        stage: Optional[StageId] = None,
    ) -> Optional[WorkItem]:
        candidate = WorkItem(
            kind=kind, direction=direction, microbatch=microbatch, layer=layer, stage=stage
        )
        return candidate if candidate in self else None

    def require(
        self,
        kind: WorkKind,
        direction: Direction,
        *,
        microbatch: Optional[MicroBatch] = None,
        layer: Optional[LayerId] = None,
        stage: Optional[StageId] = None,
    ) -> WorkItem:
        """K1: raises :class:`WorkError`, never ``KeyError``."""
        found = self.get(
            kind, direction, microbatch=microbatch, layer=layer, stage=stage
        )
        if found is None:
            raise WorkError(
                f"No WorkItem {kind.name}/{direction.name} "
                f"(microbatch={microbatch}, layer={layer}, stage={stage}) in this WorkSet"
            )
        return found

    def layers(self, direction: Direction, microbatch: MicroBatch) -> Tuple[WorkItem, ...]:
        return tuple(
            item
            for item in self.items
            if item.kind is WorkKind.LAYER
            and item.direction is direction
            and item.microbatch == microbatch
        )

    def of_kind(self, kind: WorkKind) -> Tuple[WorkItem, ...]:
        return tuple(item for item in self.items if item.kind is kind)


# ---------------------------------------------------------------------------
# Derived facts — functions of (WorkItem, FrozenWorkload), never fields
# ---------------------------------------------------------------------------


def is_moe(item: WorkItem, fw: FrozenWorkload) -> bool:
    """A3 becomes unrepresentable: the MoE-ness of a split op is looked up from
    its ``WorkItem``, not copied through an attribute tuple."""
    if item.layer is None:
        return False
    return fw.spec.is_moe_layer(item.layer)


#: Primary duration key per (kind, direction). RECOMPUTE deliberately reads the
#: FORWARD keys (schedule.py:721).
_DURATION_KEYS: Dict[Tuple[WorkKind, Direction], Tuple[str, str]] = {
    (WorkKind.EMBEDDING, Direction.FORWARD): ("embedding_f", "embedding_f"),
    (WorkKind.EMBEDDING, Direction.BACKWARD): ("embedding_b", "embedding_b"),
    (WorkKind.SOFTMAX, Direction.FORWARD): ("linear_softmax_f", "linear_softmax_f"),
    (WorkKind.SOFTMAX, Direction.BACKWARD): ("linear_softmax_b", "linear_softmax_b"),
    (WorkKind.LAYER, Direction.FORWARD): ("transformer_f_dense", "transformer_f_moe"),
    (WorkKind.LAYER, Direction.BACKWARD): ("transformer_b_dense", "transformer_b_moe"),
    (WorkKind.RECOMPUTE, Direction.FORWARD): ("transformer_f_dense", "transformer_f_moe"),
    (WorkKind.OPTIMIZER, Direction.BACKWARD): ("optimizer", "optimizer_moe"),
}

#: Legacy fallback chain: the dense/moe split keys fall back to the undifferentiated
#: ``transformer_f``/``transformer_b`` (schedule.py:539-542). One place, not four.
#: ``optimizer_moe`` falls back to ``optimizer`` the same way, so a producer that
#: never registered the MoE apply-grad price (dense-only runs, and every legacy
#: ``comp_times`` dict) prices every layer dense instead of raising.
_DURATION_FALLBACK: Dict[str, str] = {
    "transformer_f_dense": "transformer_f",
    "transformer_f_moe": "transformer_f",
    "transformer_b_dense": "transformer_b",
    "transformer_b_moe": "transformer_b",
    "optimizer_moe": "optimizer",
}


def duration_key(item: WorkItem, fw: FrozenWorkload) -> str:
    """The PER-LAYER duration key ``item`` reads.

    For an OPTIMIZER item this is the price of ONE layer's apply-grad; the node
    itself covers every layer of its stage (see :func:`optimizer_duration`).
    An OPTIMIZER item carries a ``stage``, not a ``layer``, so ``is_moe`` is
    False for it and this returns the dense key — the MoE key is selected
    per owned layer inside :func:`optimizer_duration`.
    """
    entry = _DURATION_KEYS.get((item.kind, item.direction))
    if entry is None:
        raise WorkError(f"No duration key for {item!r}")
    dense_key, moe_key = entry
    return moe_key if is_moe(item, fw) else dense_key


def _duration_by_key(key: str, fw: FrozenWorkload) -> float:
    fallback_key = _DURATION_FALLBACK.get(key)
    fallback = 0.0 if fallback_key is None else fw.durations.get_or(fallback_key, 0.0)
    return fw.durations.get_or(key, fallback)


def stage_layers(
    item: WorkItem,
    fw: FrozenWorkload,
    layers: Optional["LayerAssignment"] = None,
) -> Tuple[LayerId, ...]:
    """The transformer layers whose parameters ``item``'s pipeline stage owns.

    The layer split is REMAINDER-FIRST and therefore uneven: at ``L=10, pp=4``
    stage 0 owns 3 layers and stage 3 owns 2. Any per-stage quantity derived
    from "how many layers do I own" is consequently per-stage and cannot be a
    single global scalar (``ceil(L/pp)`` over-counts the tail stages,
    ``floor`` under-counts the head ones).

    ``layers`` is the AUTHORITATIVE :class:`LayerAssignment` and should be
    passed whenever the caller has one (``BlockExpander`` does) — it is the
    same map ``Placement.stage_of`` uses, so a future non-contiguous
    assignment stays consistent. Without it the legacy remainder-first split
    is reconstructed from ``(num_layers, pp)``.
    """
    if item.stage is None:
        raise WorkError(f"{item!r} carries no stage, so it owns no layers")
    if layers is None:
        # Lazy: program.policies.sharding imports this module, so a top-level
        # import would close the cycle. Same pattern as workload._declare_axes.
        from program.policies.sharding import ContiguousStages

        layers = ContiguousStages.legacy(fw.shape.num_layers, fw.degrees.pp)
    return tuple(LayerId(int(layer)) for layer in layers.layers_of(StageId(int(item.stage))))


def optimizer_duration(
    item: WorkItem,
    fw: FrozenWorkload,
    layers: Optional["LayerAssignment"] = None,
) -> float:
    """The fused per-stage optimizer node's duration (BUG_LEDGER 10b).

    The GRAPH structure is deliberately unchanged: **one** optimizer node per
    pipeline stage, which is how real frameworks fuse the weight update. What
    is fixed here is the DURATION. ``durations['optimizer']`` is the apply-grad
    cost of ONE layer's parameters (``train_timing.get_data_parallel_reduction_llm``
    sums QKV + output + FFN1 + FFN2 for a single layer), so charging a stage
    that owns ``ceil(L/pp)`` layers one layer's worth under-counted the weight
    update by that factor.

    Each owned layer is priced with ITS OWN kind: a stage whose layers are a
    mix of dense and MoE (``moe_layer_mask`` / ``first_k_dense_replace``) pays
    ``durations['optimizer']`` for the dense ones and
    ``durations['optimizer_moe']`` for the MoE ones.

    The stage's NON-layer parameters are added on top (BUG_LEDGER 12): the
    input embedding on stage 0 and the vocab projection on stage ``pp-1``.
    Those are ``durations['optimizer_embedding']`` / ``['optimizer_softmax']``,
    already per-rank and already ZeRO-sharded by the producer, and they are
    NOT per-layer — each is added exactly once, on the one stage that owns it.
    A producer that does not register them (inference, or any pre-item-12
    ``comp_times``) contributes 0.0 and this reduces to the 10b behavior.

    Why stage 0 / ``pp-1`` rather than a placement query: the embedding and the
    softmax are pipeline ENDPOINTS by construction — ``Placement.stage_of``
    returns exactly these for ``WorkKind.EMBEDDING`` / ``SOFTMAX`` — and this
    function is called from inside placement, so asking it back would close a
    cycle. The single-stage case (``pp == 1``) correctly gets both.
    """
    per_layer_dense = _duration_by_key("optimizer", fw)
    per_layer_moe = _duration_by_key("optimizer_moe", fw)
    total = 0.0
    for layer in stage_layers(item, fw, layers):
        total += per_layer_moe if fw.is_moe_layer(layer) else per_layer_dense
    stage = int(item.stage)
    if stage == 0:
        total += _duration_by_key("optimizer_embedding", fw)
    if stage == max(0, int(fw.degrees.pp) - 1):
        total += _duration_by_key("optimizer_softmax", fw)
    return total


def duration_for(
    item: WorkItem,
    fw: FrozenWorkload,
    layers: Optional["LayerAssignment"] = None,
) -> float:
    """The duration this WorkItem contributes, with the ONE documented legacy
    fallback applied explicitly (see :data:`_DURATION_FALLBACK`).

    Every kind but OPTIMIZER is one unit of work priced by one key. OPTIMIZER
    is one node per STAGE covering every layer that stage owns, so it is priced
    by :func:`optimizer_duration`."""
    if item.kind is WorkKind.OPTIMIZER:
        return optimizer_duration(item, fw, layers)
    return _duration_by_key(duration_key(item, fw), fw)


_MEM_KIND_NAMES: Dict[WorkKind, str] = {
    WorkKind.EMBEDDING: "EMBEDDING",
    WorkKind.SOFTMAX: "SOFTMAX",
    WorkKind.LAYER: "TRANSFORMER",
    WorkKind.RECOMPUTE: "TRANSFORMER",
    WorkKind.OPTIMIZER: "OPTIMIZER",
}


def mem_kind(item: WorkItem) -> Optional[Any]:
    """The memory census bucket of this WorkItem (``memory_estimation.MemKind``)."""
    if MemKind is None:  # pragma: no cover - defensive
        return None
    name = _MEM_KIND_NAMES.get(item.kind)
    if name is None:
        return None
    return MemKind[name]


def block_prefix(fw: FrozenWorkload) -> str:
    """Diagnostic block-name prefix (schedule.py:280-282). Names are NOT part of
    canonical equivalence (equiv/canonical.py:20-27); this exists for viz only
    and is never dispatched on."""
    return "vit_block" if fw.spec.shape.model_type.startswith("vit") else "transformer_layer"


# ---------------------------------------------------------------------------
# enumerate_work
# ---------------------------------------------------------------------------


def enumerate_work(fw: FrozenWorkload, recompute: Any) -> WorkSet:
    """The complete work enumeration (INTERFACES §2.1).

    Replaces the node-creation half of ``build_pipeline_events``
    (schedule.py:579-776). Pure, order-free::

        for b in range(mb):
          EMBEDDING/FORWARD(b);  LAYER/FORWARD(b,l) for l;  SOFTMAX/FORWARD(b)
          if include_backward:
            SOFTMAX/BACKWARD(b); LAYER/BACKWARD(b,l) for l; EMBEDDING/BACKWARD(b)
            RECOMPUTE/FORWARD(b,l) for each l where recompute.materializes(...)
        if include_backward and include_optimizer and durations['optimizer'] > 0:
          OPTIMIZER/BACKWARD(stage=s) for s in range(pp)   # one FUSED node per stage
    """
    shape = fw.spec.shape
    run = fw.spec.run
    items: list[WorkItem] = []

    for b in range(shape.micro_batches):
        items.append(WorkItem(WorkKind.EMBEDDING, Direction.FORWARD, microbatch=b))
        for layer in range(shape.num_layers):
            items.append(WorkItem(WorkKind.LAYER, Direction.FORWARD, microbatch=b, layer=layer))
        items.append(WorkItem(WorkKind.SOFTMAX, Direction.FORWARD, microbatch=b))

        if not run.include_backward:
            continue

        items.append(WorkItem(WorkKind.SOFTMAX, Direction.BACKWARD, microbatch=b))
        for layer in range(shape.num_layers):
            items.append(WorkItem(WorkKind.LAYER, Direction.BACKWARD, microbatch=b, layer=layer))
            if recompute.materializes(layer, b, fw):
                items.append(
                    WorkItem(WorkKind.RECOMPUTE, Direction.FORWARD, microbatch=b, layer=layer)
                )
        items.append(WorkItem(WorkKind.EMBEDDING, Direction.BACKWARD, microbatch=b))

    if (
        run.include_backward
        and run.include_optimizer
        and fw.durations.get_or("optimizer", 0.0) > 0.0
    ):
        # BUG_LEDGER 10b (owner decision, 2026-07-27): the optimizer stays ONE
        # FUSED node per pipeline STAGE — that is the intended structure and
        # matches how real frameworks fuse the weight update. What was wrong was
        # the DURATION: train_timing supplies ONE layer's apply-grad price, and
        # a stage owns ceil(L/pp) of them. `optimizer_duration` now sums the
        # stage's own layers, each at its own dense/MoE price. The gate below
        # still reads the per-layer key, since a zero per-layer price means no
        # apply-grad work exists at all.
        for stage in range(fw.spec.degrees.pp):
            items.append(WorkItem(WorkKind.OPTIMIZER, Direction.BACKWARD, stage=StageId(stage)))

    return WorkSet(items=tuple(items))


def backward_entry(item: WorkItem, work: WorkSet) -> WorkItem:
    """The item a backward layer's predecessors attach to: the RECOMPUTE
    rematerialization when one exists, else the backward layer itself.
    Port of ``_bwd_entry_node`` (schedule.py:682-687)."""
    if item.kind is not WorkKind.LAYER or item.direction is not Direction.BACKWARD:
        return item
    candidate = work.get(
        WorkKind.RECOMPUTE, Direction.FORWARD, microbatch=item.microbatch, layer=item.layer
    )
    return candidate if candidate is not None else item


# ---------------------------------------------------------------------------
# The typed attach vocabulary
# ---------------------------------------------------------------------------


class AttachMode(Enum):
    """CLOSED vocabulary. Replaces ``attach_parallel_edge``'s two boolean
    escape hatches (schedule.py:553-567). Completeness proof: INTERFACES §2.3.

    ``BEFORE(a)``       deps(req) := deps(a); deps(a) += req
    ``AFTER(a)``        deps(req) := {a}
    ``PARALLEL_TO(a)``  deps(req) += deps(a); for s in succ(a, via): deps(s) += req
    ``OVERLAP_WITH(a)`` defined by the OverlapDecl; realized at L4 by splitting
                        ``a`` and re-parenting, not by adding a dep.
    """

    BEFORE = auto()
    AFTER = auto()
    PARALLEL_TO = auto()
    OVERLAP_WITH = auto()


def attach_mode_arity(mode: AttachMode) -> Optional[int]:
    """Number of anchors a mode admits (``None`` = any number >= 1).

    K2: the dispatch over :class:`AttachMode` is exhaustive; this function is
    the canonical exhaustiveness witness and ends in ``assert_never``.
    """
    if mode is AttachMode.BEFORE:
        return 1
    if mode is AttachMode.AFTER:
        return 1
    if mode is AttachMode.OVERLAP_WITH:
        return 1
    if mode is AttachMode.PARALLEL_TO:
        return None
    from typing import assert_never  # py3.11

    assert_never(mode)


class DepClass(Enum):
    """The four dependency classes of INTERFACES §4.3. Also the ``via``
    vocabulary of PARALLEL_TO: exactly the successor-edge classes the
    requirement inherits."""

    DATA_FLOW = auto()  #: tensor producer -> consumer (same-device dep OR TransferOp)
    SCHEDULE = auto()  #: device serialization implied by the SchedulePolicy
    SYNC = auto()  #: another SyncRequirement's attach


VIA_DATA_FLOW: FrozenSet[DepClass] = frozenset({DepClass.DATA_FLOW})
VIA_NON_DATA_FLOW: FrozenSet[DepClass] = frozenset({DepClass.SCHEDULE, DepClass.SYNC})
VIA_ALL: FrozenSet[DepClass] = frozenset(DepClass)


class SyncSpread(Enum):
    """How many instances of the requirement exist inside one pipeline stage.

    NAMED POLICY for BUG_LEDGER **A2** (DP collective attached to
    ``rank_tails[0]`` only, pipeline_fine.py:616-629) and **B/10d**.

    **A2 is FIXED**: every requirement the ``ShardingPolicy`` emits now declares
    ``PER_CLUSTER_RANK`` (``policies.sharding._spread_for``). ``CLUSTER_RANK_0``
    remains in the vocabulary because it is still the honest declaration for a
    collective whose *work* the model pins to one cluster rank — the EP grad
    sync (row S12, ``policies/routing.py``) and any future collective whose byte
    count is a whole-cluster quantity rather than a per-rank one.
    """

    STAGE = auto()  #: one instance; the stage IS the device (COARSE/BLOCK)
    CLUSTER_RANK_0 = auto()  #: one instance, on cluster rank 0 of the stage [legacy default]
    PER_CLUSTER_RANK = auto()  #: one instance per cluster rank [A2's fix; every dp collective]


class ByteSplit(Enum):
    """How a comm key's byte count is divided across the instances of a
    requirement.

    ``WHOLE`` is BUG_LEDGER **Class B item 1** (ZeRO-3 per-rank gather bytes
    "un-divided"): not a bug for tp/cp — each cluster rank owns its own TP shard
    and gathers from the dp group, so total fine traffic ``cp*ep*P_layer`` is
    correct. ``CEIL_DIV_CLUSTER`` is BUG_LEDGER **D 10c** (cross-layer activation
    bytes split across the stage's cluster, pipeline_fine.py:645).

    ⚠ The argument ``CEIL_DIV_CLUSTER`` divides by is a **SHARD count, not a
    device count**. Its one caller (``build.py::_cross_layer_size``) passes
    ``Placement.activation_shard_size() == tp*cp``, NOT ``cluster_size() ==
    tp*cp*ep``: EP ranks own distinct microbatches, so the ``/ep`` was a second
    application of a division the raw value already carried (D 10c ep half,
    fixed 2026-07-27). Reusing a placement device count as a byte-sharding
    degree is exactly how that defect entered.
    """

    WHOLE = auto()
    CEIL_DIV_CLUSTER = auto()


@dataclass(frozen=True)
class ByteSource:
    """Where a requirement's bytes come from. The VALUE is owned by
    ``train_timing`` (``CommSpec.size_bytes``); only the SPLIT is policy.

    ``key`` addresses the **pipeline-level** table ``WorkloadSpec.comm``, which
    is where every key a ``SyncRequirement`` or an R2 transfer can name lives
    (the dp reducers, the ZeRO gathers, ``cross_layer``, the EP grad syncs — all
    from ``train_timing._build_comm_metadata``). Block-template collectives are
    not requirements: they are ``CommStep``s carrying their own per-template
    ``CommSpec`` (INTERFACES §1.6 / §3.4).
    """

    key: CommKey
    split: ByteSplit = ByteSplit.WHOLE

    def bytes_for(self, fw: FrozenWorkload, instances: int) -> float:
        """``instances`` is how many ways the VALUE is split — for a collective
        whose bytes are a whole-cluster quantity that is the number of copies the
        caller materializes (``len(devices)``), and for ``cross_layer`` it is
        ``Placement.activation_shard_size()`` (INTERFACES §4.3 R2, §4.4). The two
        coincide on every axis that shards tokens and differ on ``ep``, which
        does not (see :class:`ByteSplit`).

        AMENDMENT to INTERFACES §2.2 (dated 2026-07-26): ``CEIL_DIV_CLUSTER``
        divides by ``instances``, NOT by ``FrozenWorkload.cluster_size()``. The
        two differ exactly where it matters: ``Placement`` reports
        ``cluster_size() == 1`` at COARSE (the stage IS the device) while
        ``fw.spec.cluster_size()`` is always ``tp*cp*ep``. Legacy COARSE uses
        the RAW value (pipeline_coarse.py:222,238) and legacy FINE divides
        (pipeline_fine.py:645), so ignoring ``instances`` returned 2048.0 where
        COARSE wants 4096.0 on every coarse/hybrid/hierarchical row with
        ``cluster_size > 1``.
        """
        count = int(instances)
        if count < 1:
            raise SyncError(
                f"ByteSource({self.key!r}).bytes_for needs instances >= 1 (got {instances})"
            )
        spec = fw.spec.comm.require(self.key)
        if self.split is ByteSplit.WHOLE:
            return float(spec.size_bytes)
        if self.split is ByteSplit.CEIL_DIV_CLUSTER:
            return ceil_div(spec.size_bytes, count)
        from typing import assert_never  # py3.11

        assert_never(self.split)


class SyncPhase(Enum):
    """Where in the schedule a requirement belongs. Also the primary
    ``SyncOrder`` rank at L4 (INTERFACES §4.4)."""

    FWD_ENTRY = auto()
    FWD = auto()
    BWD_ENTRY = auto()
    BWD = auto()
    GRAD = auto()


@dataclass(frozen=True)
class SyncKey:
    """Identity of one requirement instance. Unique by construction."""

    comm_key: CommKey
    phase: SyncPhase
    microbatch: Optional[MicroBatch] = None
    layer: Optional[LayerId] = None

    def sort_key(self) -> Tuple[int, int, int, str]:
        return (
            self.phase.value,
            _UNSET if self.microbatch is None else int(self.microbatch),
            _UNSET if self.layer is None else int(self.layer),
            self.comm_key,
        )

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, SyncKey):
            return NotImplemented
        return self.sort_key() < other.sort_key()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        parts = [self.comm_key, self.phase.name]
        if self.microbatch is not None:
            parts.append(f"b{self.microbatch}")
        if self.layer is not None:
            parts.append(f"l{self.layer}")
        return "S(" + ",".join(parts) + ")"


SyncAnchor = Union[WorkItem, SyncKey]


@dataclass(frozen=True)
class SyncRequirement:
    """One collective, declared: what / where / when / how."""

    key: SyncKey

    # -- what --------------------------------------------------------------
    bytes: ByteSource
    kind: CollectiveType  #: from CommSpecTable[bytes.key].kind — NEVER sniffed
    axes: Tuple[AxisName, ...]  #: from CommSpecTable[bytes.key].axes
    participants: int  #: ANALYTICAL count (workload.CommSpec docstring)

    # -- where -------------------------------------------------------------
    place_on: WorkItem  #: the work whose device hosts this collective
    spread: SyncSpread = SyncSpread.CLUSTER_RANK_0

    # -- when --------------------------------------------------------------
    mode: AttachMode = AttachMode.AFTER
    anchors: Tuple[SyncAnchor, ...] = ()
    via: FrozenSet[DepClass] = VIA_ALL  #: PARALLEL_TO only
    #: AMENDMENT 2026-07-29 — work whose ENTRY must depend on this requirement,
    #: DECLARED rather than discovered through ``via``.
    #:
    #: ``PARALLEL_TO(a, via)`` answers "who waits for me?" with
    #: ``succ(a, via)`` — the successors the anchor happens to have when R4
    #: runs. For rows **S6/S14** ``via`` is ``VIA_NON_DATA_FLOW``, i.e. the
    #: answer is literally *an R3 edge*, whose identity is chosen by the
    #: ``SchedulePolicy``. Under GPipe it is the right op; under a schedule
    #: whose backward walks microbatches ascending, one gather ends up
    #: preceding NOTHING and another attaches to the wrong microbatch
    #: (measured). The consumer is known at DECLARATION time, so it is
    #: declared. Under GPipe it names exactly the op ``via`` already finds, so
    #: the attach is idempotent and the artifact does not move.
    consumers: Tuple[WorkItem, ...] = ()

    # -- how ---------------------------------------------------------------
    overlap: Optional[Any] = None  #: OverlapDecl (program.policies.overlap)

    #: True for the GRADIENT REDUCER itself — the collective that produces the
    #: reduced gradient the optimizer applies. Declared by the policy, because
    #: only the policy knows: :meth:`ShardingPolicy.after_reducer` hangs further
    #: collectives on the reducer and they land in the SAME ``SyncPhase.GRAD``,
    #: so the phase cannot distinguish them.
    #:
    #: **R5b depends on this.** It first filtered by ``key.phase is GRAD``, which
    #: swept up ZeRO-2's post-reduce PARAMETER ALL-GATHER and ordered
    #: ``reduce-scatter -> all-gather -> update`` — the physical inverse of
    #: ``reduce-scatter -> update -> all-gather``. A phase is a position in the
    #: schedule (``work.SyncPhase``), never a role; asking it a role question is
    #: how the wrong edge got in.
    is_reducer: bool = False

    #: True for a collective that broadcasts the UPDATED parameters, i.e. one
    #: that must run AFTER the optimizer. ZeRO-2/3's post-reduce parameter
    #: all-gather is the only such row today (row S3/S5/S10).
    #:
    #: The real ZeRO-2 sequence is ``reduce-scatter -> update my 1/dp shard ->
    #: all-gather``. Declaring the gather ``AFTER(reducer)`` alone leaves it a
    #: SIBLING of the optimizer, and AstraSim's one-COMP/one-COMM-slot model
    #: then overlaps them — cheaper than physics allows. R5b wires
    #: ``optimizer -> gather`` from this flag.
    after_update: bool = False

    #: Diagnostics only: the INTERFACES §2.3 row this requirement reproduces.
    origin: str = ""

    def __post_init__(self) -> None:
        if not self.anchors:
            raise SyncError(f"SyncRequirement {self.key!r} has no anchors")
        arity = attach_mode_arity(self.mode)
        if arity is not None and len(self.anchors) != arity:
            raise SyncError(
                f"SyncRequirement {self.key!r} mode {self.mode.name} admits exactly "
                f"{arity} anchor(s), got {len(self.anchors)}"
            )
        if self.mode is not AttachMode.PARALLEL_TO and self.via != VIA_ALL:
            raise SyncError(
                f"SyncRequirement {self.key!r} sets via={sorted(c.name for c in self.via)} "
                f"but mode {self.mode.name} ignores it (silent-misuse guard)"
            )
        if self.kind is CollectiveType.PIPELINE:
            raise SyncError(
                f"SyncRequirement {self.key!r} has kind PIPELINE; a p2p transfer is a "
                "data-flow TransferOp (R2), not a SyncRequirement"
            )
        if DP_AXIS in tuple(self.axes) and tuple(self.axes) != (DP_AXIS,):
            # dp is stamped at emission over PRE-dp device ids (ir.py:93-104), so
            # it can never be one axis of a composite communicator built from a
            # device layout. Every production dp spec declares
            # interconnect_type="dp" alone; a mixed tuple is a declaration bug.
            raise SyncError(
                f"SyncRequirement {self.key!r} declares axes {tuple(self.axes)}: "
                "'dp' is replicated at emission and cannot be combined with a "
                "device-layout axis"
            )
        if self.mode is AttachMode.AFTER:
            pass  # anchors may be a WorkItem OR another SyncKey (S3/S5/S10 chaining)
        elif any(isinstance(anchor, SyncKey) for anchor in self.anchors):
            raise SyncError(
                f"SyncRequirement {self.key!r} mode {self.mode.name} may only anchor on "
                "WorkItems; only AFTER may chain onto another SyncKey"
            )
        if any(not isinstance(item, WorkItem) for item in self.consumers):
            raise SyncError(
                f"SyncRequirement {self.key!r} declares a non-WorkItem consumer: "
                f"{list(self.consumers)!r}"
            )
        object.__setattr__(self, "anchors", tuple(self.anchors))
        object.__setattr__(self, "consumers", tuple(self.consumers))
        object.__setattr__(self, "axes", tuple(self.axes))
        object.__setattr__(self, "participants", int(self.participants))

    @classmethod
    def from_spec(
        cls,
        spec: CommSpec,
        *,
        key: SyncKey,
        place_on: WorkItem,
        mode: AttachMode,
        anchors: Sequence[SyncAnchor],
        spread: SyncSpread = SyncSpread.CLUSTER_RANK_0,
        via: FrozenSet[DepClass] = VIA_ALL,
        consumers: Sequence[WorkItem] = (),
        split: ByteSplit = ByteSplit.WHOLE,
        overlap: Optional[Any] = None,
        is_reducer: bool = False,
        after_update: bool = False,
        origin: str = "",
    ) -> "SyncRequirement":
        """Build a requirement from its :class:`CommSpec`.

        **K5**: ``kind``, ``axes`` and ``participants`` are copied from the
        table, never derived from a name, a byte count or a participant count.
        This is what deletes the collective-name sniffing at schedule.py:801-803
        and the participant-count inference at legacy_lowering.py:243-248.
        """
        if key.comm_key != spec.key:
            raise SyncError(
                f"SyncKey.comm_key {key.comm_key!r} disagrees with CommSpec.key {spec.key!r}"
            )
        return cls(
            key=key,
            bytes=ByteSource(key=spec.key, split=split),
            kind=spec.kind,
            axes=spec.axes,
            participants=spec.participants,
            place_on=place_on,
            spread=spread,
            mode=mode,
            anchors=tuple(anchors),
            via=via,
            consumers=tuple(consumers),
            overlap=overlap,
            is_reducer=is_reducer,
            after_update=after_update,
            origin=origin,
        )

    @property
    def comm_key(self) -> CommKey:
        return self.bytes.key

    @property
    def is_dp(self) -> bool:
        """``True`` iff this collective's communicator is the dp replica group.

        A dp collective has **no** ``GroupKey``: its members are not devices of
        the program's device space at all. dp replication is stamped at emission
        over pre-dp device ids (``ir.py:93-104``,
        ``CollectiveOp.is_dp`` / ``label is None``), so the builder must emit
        ``group=None, is_dp=True`` — never a communicator built from the device
        layout (INTERFACES §3.3 amendment, 2026-07-26).
        """
        return tuple(self.axes) == (DP_AXIS,)

    def size_bytes(self, fw: FrozenWorkload, instances: int = 1) -> float:
        return self.bytes.bytes_for(fw, instances)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"Sync({self.key!r},{self.mode.name},on={self.place_on!r},"
            f"anchors={list(self.anchors)},spread={self.spread.name})"
        )
