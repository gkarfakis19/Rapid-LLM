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

"""L3 — ``GPipeSchedule``: today's order, declared (INTERFACES §4.1).

The legacy builder never *states* an order; it emits work in a creation sequence
and then hand-wires the cross-microbatch edges that make that sequence real
(``schedule.py:655-672`` forward, ``:955-976`` backward). This class states the
order; ``build()``'s R3 derives the edges.

The order, and where each part comes from:

===============  ==========================================================
phase            order
===============  ==========================================================
forward          microbatch-ASCENDING (``schedule.py:580`` ``for b in range(B)``),
                 then data-flow position within the microbatch:
                 ``EMBEDDING/F, LAYER/F(0..L-1), SOFTMAX/F``
backward         microbatch-DESCENDING (``:693`` ``for b in reversed(range(B))``),
                 then reverse data-flow position:
                 ``SOFTMAX/B, (RECOMPUTE/F(l), LAYER/B(l)) for l in L-1..0,
                 EMBEDDING/B``
optimizer        after all backward work, stage-ASCENDING (``:1054``)
===============  ==========================================================

Projected onto any device this reproduces the legacy per-device order exactly:

* forward on stage ``s``: microbatch-major, because legacy's explicit
  ``exit(b) -> entry(b+1)`` edge at each stage boundary is precisely the
  adjacent pair of that projection;
* backward on stage ``s``: microbatch-DESCENDING, because legacy's
  ``exit(b, l) -> entry(b-1, first_layer_of_stage)`` edge is the adjacent pair;
* the optimizer of stage ``s`` lands after that stage's last backward item,
  which is ``EMBEDDING/BWD(0)`` on stage 0 and ``LAYER/BWD(0, min_layer(s))``
  elsewhere — verbatim the legacy attach points at ``:1057`` / ``:1061``.

**RECOMPUTE placement.** Legacy creates the rematerialization node inside the
same ``for l in reversed(range(L))`` iteration as the backward layer and wires
``recompute_node.add_child(transformer_node_b)`` (``:717-750``), so the
recompute immediately precedes its own backward layer. That is what
``_bwd_entry_node`` (``:682-687``) means when it returns the recompute node as
the layer's ENTRY, and it is reproduced here by emitting the pair adjacently.

This class is the DEFAULT. ``program/sched/onefonebee.py`` (student project)
implements the same protocol and needs no other edit — see
:mod:`program.schedule.policy` for the seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from program.schedule.policy import (
    LayerAssignment,
    Schedule,
    ScheduleError,
    ScheduleSlot,
)
from program.types import StageId
from program.work import Direction, WorkItem, WorkKind, WorkSet
from program.workload import FrozenWorkload

__all__ = ["GPipeSchedule"]


@dataclass(frozen=True)
class GPipeSchedule:
    """THE DEFAULT schedule policy."""

    name: str = "gpipe"

    # -- layer -> stage ----------------------------------------------------
    def layer_assignment(self, fw: FrozenWorkload) -> LayerAssignment:
        """The legacy remainder-first contiguous split
        (``schedule.legacy_layers_per_stage``: ``base + 1`` layers for the first
        ``num_layers % pp`` stages)."""
        return LayerAssignment.contiguous(
            int(fw.spec.shape.num_layers), int(fw.spec.degrees.pp)
        )

    # -- the order ---------------------------------------------------------
    def schedule(self, fw: FrozenWorkload, work: WorkSet) -> Schedule:
        layers = self.layer_assignment(fw)
        shape = fw.spec.shape
        order: List[WorkItem] = []

        def take(item: Optional[WorkItem]) -> None:
            if item is not None:
                order.append(item)

        # -- forward: microbatch-ascending, data-flow order ----------------
        for b in range(int(shape.micro_batches)):
            take(work.get(WorkKind.EMBEDDING, Direction.FORWARD, microbatch=b))
            for layer in range(int(shape.num_layers)):
                take(
                    work.get(
                        WorkKind.LAYER, Direction.FORWARD, microbatch=b, layer=layer
                    )
                )
            take(work.get(WorkKind.SOFTMAX, Direction.FORWARD, microbatch=b))

        # -- backward: microbatch-DESCENDING, reverse data-flow order ------
        for b in reversed(range(int(shape.micro_batches))):
            take(work.get(WorkKind.SOFTMAX, Direction.BACKWARD, microbatch=b))
            for layer in reversed(range(int(shape.num_layers))):
                # The rematerialization is the backward layer's ENTRY
                # (``_bwd_entry_node``), so it comes immediately before it.
                take(
                    work.get(
                        WorkKind.RECOMPUTE,
                        Direction.FORWARD,
                        microbatch=b,
                        layer=layer,
                    )
                )
                take(
                    work.get(
                        WorkKind.LAYER, Direction.BACKWARD, microbatch=b, layer=layer
                    )
                )
            take(work.get(WorkKind.EMBEDDING, Direction.BACKWARD, microbatch=b))

        # -- optimizer tail: stage-ascending -------------------------------
        for stage in range(int(fw.spec.degrees.pp)):
            take(
                work.get(
                    WorkKind.OPTIMIZER, Direction.BACKWARD, stage=StageId(stage)
                )
            )

        if len(order) != len(work.items):
            # A WorkSet member this policy did not place is a contract breach,
            # not something to silently drop (S3 is checked below as well, but
            # this message names the count difference first).
            raise ScheduleError(
                f"GPipeSchedule placed {len(order)} of {len(work.items)} WorkItems; "
                "every WorkItem must appear exactly once (S3)"
            )

        slots = tuple(
            ScheduleSlot(index=index, stage=layers.stage_of_work(item), work=item)
            for index, item in enumerate(order)
        )
        schedule = Schedule(policy=self.name, layers=layers, slots=slots)
        schedule.check_permutation(work)
        return schedule
