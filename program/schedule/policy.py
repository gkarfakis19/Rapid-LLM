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

"""L3 — ``SchedulePolicy`` / ``Schedule`` / ``LayerAssignment`` (INTERFACES §4.1).

A :class:`SchedulePolicy` answers two questions and nothing else:

1. **which stage hosts which layer** (:class:`LayerAssignment`), and
2. **in what order does each device execute its work** (:class:`Schedule`).

Everything else a pipelining pattern implies is *derived*: the cross-microbatch
edges legacy hand-wires at ``schedule.py:655-672`` (forward) and ``:955-976``
(backward) are not modeled here, they fall out of :meth:`Schedule.implied_deps`
(INTERFACES §4.3 R3).

**Why the order must be materialized as DEPENDENCIES.** AstraSim's one-slot rule
constrains *concurrency*, not *order*: a rank runs one compute at a time but is
free to pick any ready node, so an intended order that is not in the DAG is not
an order at all. Legacy therefore materializes the cross-microbatch dependency
explicitly — ``git show 85894c6:simulate_train_graph.py:836-856``,

    # add dependency edges
    ...
    if transformer_nodes[b][l].hw_id == 0:      # if first pipeline stage
        layer_exit_nodes[b][l].add_child(embedding_node[b+1])
        # dependency: finish stage before next batch embedding starts

— and so must every :class:`Schedule`. :meth:`Schedule.implied_deps` is that
declaration; ``build()`` materializes each one that is not already transitively
implied (INTERFACES §4.3 R3, invariant **D1**).

**The interleaving seam (student project, NOT implemented here).**
:class:`LayerAssignment` is an explicit ``layer -> stage`` MAP, not the monotone
``layers_per_stage`` count tuple of ``schedule.py:163``. Nothing in L2/L3/L4
requires it to be monotone or contiguous:

* :meth:`Placement.stage_of` reads ``layers.stage_of(layer)`` per layer;
* R2 (cross-layer data flow) compares the *devices* of consecutive layers, so a
  layer whose stage revisits an earlier device is an ordinary same-device link;
* R3 projects :attr:`Schedule.slots` onto each device through
  ``Placement.devices_for``, so a device that appears twice in the layer order
  simply has more work in its projection.

A 1F1B / interleaved-virtual-stage policy is therefore a new
:class:`SchedulePolicy` implementation and NOTHING else: a non-contiguous
``LayerAssignment`` plus a different ``slots`` permutation. The one policy that
GPipe's shape leaks into is :class:`~program.policies.gradaccum.GradAccumPolicy`'s
"``b == 0`` is the last microbatch" constant (BUG_LEDGER Class B 10h), and that
is already a data field rebound through
:meth:`GradAccumPolicy.for_schedule` -> :meth:`Schedule.last_microbatch_of`,
which is computed here from the slot order rather than hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from program.policies.sharding import ContiguousStages
from program.types import DeviceId, MicroBatch, StageId
from program.work import Direction, WorkItem, WorkKind, WorkSet
from program.workload import FrozenWorkload

__all__ = [
    "ScheduleError",
    "LayerAssignment",
    "ScheduleSlot",
    "ScheduleDep",
    "Schedule",
    "SchedulePolicy",
]


class ScheduleError(ValueError):
    """A Schedule is not a total order over the WorkSet, or a lookup failed."""


# ---------------------------------------------------------------------------
# LayerAssignment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerAssignment(ContiguousStages):
    """Explicit ``layer -> stage`` map (INTERFACES §4.1).

    GENERALIZES ``layers_per_stage: Tuple[int, ...]`` (``schedule.py:163``) from
    monotone COUNTS to an arbitrary map: ``stage_of_layer[l]`` may be any stage
    id, in any order, and a stage may own a non-contiguous layer set. That is
    the whole prerequisite for interleaved virtual stages (ext_1f1b.md item 1),
    and it is why this type is a map and not a count tuple.

    Members (all inherited unchanged — see the class body of
    :class:`~program.policies.sharding.ContiguousStages`):

    * ``stage_of(layer)`` / ``layers_of(stage)`` / ``min_layer(stage)`` — the
      §4.1 ``LayerAssignment`` surface, keyed on a ``LayerId``;
    * ``stage_of_work(work)`` / ``same_stage(a, b)`` — the L1
      :class:`~program.policies.sharding.StagePartition` surface, keyed on a
      ``WorkItem``.

    One object therefore satisfies BOTH seams (``Placement(layers=...)`` and
    ``ShardingContext(stages=...)``) — the B5 amendment of 2026-07-26.

    **Transitional note.** ``ContiguousStages`` is the same class under its
    pre-L3 name: INTERFACES §2.4 says "P4 replaces it with
    ``program.schedule.policy.LayerAssignment``, same members", so this subclass IS
    that rename, with zero new behavior and zero duplicated logic. P5 collapses
    the two by moving the body here and deleting the L1 name (which exists only
    because L2 landed before L3).
    """

    @classmethod
    def explicit(
        cls, stage_of_layer: Sequence[int], num_stages: Optional[int] = None
    ) -> "LayerAssignment":
        """An ARBITRARY map — the interleaving constructor.

        ``contiguous`` (the default, legacy remainder-first split) is a special
        case of this. ``num_stages`` defaults to ``max(stage) + 1``; pass it
        explicitly when a trailing stage owns no layer.
        """
        mapping = tuple(StageId(int(stage)) for stage in stage_of_layer)
        if any(int(stage) < 0 for stage in mapping):
            raise ScheduleError(
                f"LayerAssignment.explicit got a negative stage in {tuple(mapping)}"
            )
        stages = int(num_stages) if num_stages is not None else (
            max(int(stage) for stage in mapping) + 1 if mapping else 1
        )
        if mapping and max(int(stage) for stage in mapping) >= stages:
            raise ScheduleError(
                f"LayerAssignment.explicit: stage {max(int(s) for s in mapping)} "
                f"is out of range for num_stages={stages}"
            )
        return cls(stage_of_layer=mapping, num_stages=stages)

    def contiguous_layers(self) -> bool:
        """``True`` iff every stage owns one contiguous, ascending layer run.

        Diagnostics only: no rule in L2/L3/L4 requires it. ``build()`` stamps
        the answer into ``Program.meta.misc["layer_assignment_contiguous"]`` so
        an interleaved schedule is visible in a dumped program.
        """
        for stage in range(self.num_stages):
            layers = self.layers_of(StageId(stage))
            if not layers:
                continue
            if tuple(range(layers[0], layers[-1] + 1)) != layers:
                return False
        return True


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScheduleSlot:
    """One unit of work at one position of the global schedule order."""

    index: int  #: GLOBAL, dense, 0..N-1 — the schedule step
    stage: StageId
    work: WorkItem


@dataclass(frozen=True)
class ScheduleDep:
    """A dependency the pipelining pattern implies on ONE device.

    ``before`` must complete before ``after`` may start *on ``device``*. This is
    the declaration the legacy builder hand-wired (``schedule.py:655-672`` /
    ``:955-976``); ``build()`` materializes it as a ``DepClass.SCHEDULE`` edge
    unless the base graph already implies it transitively (INTERFACES §4.3 R3).
    """

    before: WorkItem
    after: WorkItem
    device: DeviceId
    reason: str = "device_serialization"


#: Kinds whose backward work carries a microbatch's gradients — the input to
#: :meth:`Schedule.last_microbatch_of` (Class B 10h).
_GRADIENT_KINDS = (WorkKind.LAYER, WorkKind.EMBEDDING, WorkKind.SOFTMAX)


@dataclass(frozen=True)
class Schedule:
    """A total order over ALL work, plus the layer->stage map it was built from.

    ``slots`` is the GLOBAL order; a device's order is the projection of it onto
    the work placed on that device (:meth:`device_projection`). Keeping one
    global order rather than per-device lists is what makes
    ``Program`` order schedule-major (INTERFACES §4.6) and what lets R3 be a
    projection instead of a second schedule representation.
    """

    policy: str
    layers: LayerAssignment
    slots: Tuple[ScheduleSlot, ...]

    def __post_init__(self) -> None:
        slots = tuple(self.slots)
        object.__setattr__(self, "slots", slots)
        indices = [int(slot.index) for slot in slots]
        # S1: dense 0..N-1 and unique.
        if sorted(indices) != list(range(len(slots))):
            raise ScheduleError(
                f"Schedule.slots indices must be dense 0..{len(slots) - 1} and "
                f"unique (got {sorted(indices)})"
            )
        by_work: Dict[WorkItem, int] = {}
        stage_by_work: Dict[WorkItem, StageId] = {}
        for slot in slots:
            if slot.work in by_work:
                raise ScheduleError(
                    f"Schedule contains {slot.work!r} twice (S2: every WorkItem "
                    "appears exactly once)"
                )
            by_work[slot.work] = int(slot.index)
            stage_by_work[slot.work] = StageId(int(slot.stage))
        object.__setattr__(self, "_by_work", by_work)
        object.__setattr__(self, "_stage_by_work", stage_by_work)
        object.__setattr__(
            self, "_ordered", tuple(sorted(slots, key=lambda s: int(s.index)))
        )

    # -- lookups ----------------------------------------------------------
    @property
    def ordered_slots(self) -> Tuple[ScheduleSlot, ...]:
        """``slots`` sorted by :attr:`ScheduleSlot.index`."""
        return self.__dict__["_ordered"]

    def order(self) -> Tuple[WorkItem, ...]:
        """The global work order."""
        return tuple(slot.work for slot in self.ordered_slots)

    def index_of(self, work: WorkItem) -> int:
        found = self.__dict__["_by_work"].get(work)
        if found is None:
            raise ScheduleError(f"{work!r} is not scheduled")
        return found

    def stage_of(self, work: WorkItem) -> StageId:
        found = self.__dict__["_stage_by_work"].get(work)
        if found is None:
            raise ScheduleError(f"{work!r} is not scheduled")
        return found

    def order_for(self, stage: StageId) -> Tuple[WorkItem, ...]:
        """The stage's execution order = its slots, by ``index``."""
        return tuple(
            slot.work
            for slot in self.ordered_slots
            if int(slot.stage) == int(stage)
        )

    def __len__(self) -> int:
        return len(self.slots)

    # -- Class B 10h: which microbatch "last_mb" means under THIS schedule --
    def last_microbatch_of(self) -> MicroBatch:
        """The microbatch whose gradients complete LAST under this order.

        Computed, not assumed: the highest-indexed backward gradient-bearing
        slot's microbatch. Under GPipe the backward pass walks microbatches in
        reverse, so the answer is ``0`` — which is exactly the legacy constant
        (``schedule.py:266-268``), now derived. A 1F1B policy gets the right
        answer with no edit to :class:`~program.policies.gradaccum.GradAccumPolicy`
        (INTERFACES §4.1).
        """
        for slot in reversed(self.ordered_slots):
            work = slot.work
            if (
                work.direction is Direction.BACKWARD
                and work.kind in _GRADIENT_KINDS
                and work.microbatch is not None
            ):
                return MicroBatch(int(work.microbatch))
        return MicroBatch(0)

    # -- R3's input: the per-device projection and the deps it implies ------
    def device_projection(
        self, devices_for: Callable[[WorkItem], Sequence[DeviceId]]
    ) -> Mapping[DeviceId, Tuple[WorkItem, ...]]:
        """``device -> the work on it, in schedule order``.

        ``devices_for`` is INJECTED (``Placement.devices_for``) rather than
        imported: L3 declares an order over work and must not know what a device
        is. At FINE a stage is ``cluster_size`` devices and a pinned kind
        (softmax, Class B 10d) lives on only one of them, so the projection is
        genuinely per DEVICE and not per stage — which is exactly why R3 is
        stated per device (INTERFACES §4.3).
        """
        projection: Dict[DeviceId, List[WorkItem]] = {}
        for slot in self.ordered_slots:
            for device in devices_for(slot.work):
                projection.setdefault(DeviceId(int(device)), []).append(slot.work)
        return {device: tuple(items) for device, items in projection.items()}

    def implied_deps(
        self, devices_for: Callable[[WorkItem], Sequence[DeviceId]]
    ) -> Tuple[ScheduleDep, ...]:
        """The dependencies this pipelining pattern implies, per device.

        Adjacent pairs of each device's projection. Returned in
        ``(after-slot, device)`` order so a consumer that adds them one at a
        time sees a monotone slot sequence (R3 relies on that for its
        reachability check).

        The legacy special cases are NOT re-coded — they fall out:

        * ``schedule.py:663-664`` "if the stage is 0, the successor is
          ``embedding_node[b+1]``": stage 0's projection is
          ``[..., LAYER/FWD(b, last), EMBEDDING/FWD(b+1), ...]``, so the
          adjacent pair IS that edge;
        * ``:670-672`` ``softmax_node[b] -> layer_entry_nodes[b+1][first]``:
          stage ``pp-1``'s projection is
          ``[..., SOFTMAX/FWD(b), LAYER/FWD(b+1, ·), ...]``;
        * ``:969-970`` / ``:976`` — the mirror images on the backward
          projection, where microbatches descend.
        """
        deps: List[ScheduleDep] = []
        projection = self.device_projection(devices_for)
        for device in sorted(projection, key=int):
            items = projection[device]
            for before, after in zip(items, items[1:]):
                deps.append(ScheduleDep(before=before, after=after, device=device))
        deps.sort(
            key=lambda dep: (
                self.index_of(dep.after),
                int(dep.device),
                self.index_of(dep.before),
            )
        )
        return tuple(deps)

    # -- self-check (S1-S3) ------------------------------------------------
    def check_permutation(self, work: WorkSet) -> None:
        """**S3**: the schedule is a permutation of ``work.items``."""
        scheduled = set(self.__dict__["_by_work"])
        declared = set(work.items)
        missing = declared - scheduled
        extra = scheduled - declared
        if missing or extra:
            raise ScheduleError(
                f"Schedule({self.policy!r}) is not a permutation of the WorkSet: "
                f"{len(missing)} unscheduled {sorted(missing, key=WorkItem.sort_key)[:4]}, "
                f"{len(extra)} unknown {sorted(extra, key=WorkItem.sort_key)[:4]}"
            )


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------


class SchedulePolicy(Protocol):
    name: str

    def layer_assignment(self, fw: FrozenWorkload) -> LayerAssignment:
        """The ``layer -> stage`` map this policy runs. May be non-contiguous."""

    def schedule(self, fw: FrozenWorkload, work: WorkSet) -> Schedule:
        """Total order over ALL work, globally indexed. MUST be a permutation of
        ``work.items`` (S3) with dense unique indices (S1)."""
