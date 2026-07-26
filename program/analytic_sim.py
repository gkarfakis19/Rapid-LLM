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

"""Analytical evaluation of COARSE Programs (M5).

Verbatim-semantics port of the deleted legacy pair

* ``Graph.convert_comm_sizes_to_times`` (simulate_train_graph.py, formerly
  lines 316-361) — :func:`convert_comm_sizes_to_times`, and
* ``Graph.simulate`` (formerly lines 998-1110) — :func:`evaluate`,

onto the COARSE builder's output (:mod:`program.pipeline_coarse`). The
engine is the legacy list scheduler exactly:

* event heap keyed ``(finish_time, insertion_counter)``;
* one outstanding compute per device (``GPU_list`` boolean exclusivity);
  comm events (collectives, pipeline transfers, same-stage zero-byte
  control edges) NEVER occupy a device slot;
* a FIFO ready-list scanned in append order after every completion;
  successors iterated in legacy children-list order;
* same-stage zero-byte transfers are enqueued as real heap events — they
  consume insertion counters and therefore shift analytical tie order
  (DESIGN.md §2 amendment 2);
* the root is pushed at t=0 WITHOUT occupying its device (legacy quirk) and
  a comm-event root keeps duration 0 (the legacy conversion only ever
  converted *children*, so the ZeRO-3 entry-edge root was never timed);
* compute durations read ``ComputeOp.duration[0]`` — exactly the legacy
  ``Node.duration`` property over a per-DP profile tuple (DESIGN.md §2
  amendment 4), which is how hybrid retiming (:mod:`program.retime`)
  reaches the evaluator.

The comm conversion pass reproduces the legacy call args and call ORDER of
``NetworkModel.collective``: a recursive first-encounter DFS from the root
converting each *child* with ``comm_size_bytes > 0`` before descending into
it — including the duplicate conversions of multi-parent edges (an already
-visited child is re-converted, not re-descended). Sizes/participants/
interconnect labels are read from the schedule events (the RAW legacy
``Edge`` attribute surface — dp reducer sizes are floats and must stay
floats, see ``CommMeta``), and results are written onto ``event.duration``
in place, exactly like the legacy pass (the memory dispatcher replays this
on its own coarse events — ``build_fine_program_for_memory``).

The event loop runs over ``meta.misc["coarse_events"]`` /
``["coarse_proto_root"]`` because the legacy FIFO discipline depends on the
children-list adjacency order, which uid order cannot represent (see
:mod:`program.pipeline_coarse`; the :mod:`program.memory_sim` precedent).
Scheduling state lives in evaluator-local structures — the events are only
mutated by the conversion pass (durations), as legacy was.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Any, Dict, List, Mapping, Set, Tuple

from timing_model import CollectiveType

from program.ir import ComputeOp, Program
from program.schedule import CommEvent, ComputeEvent


def convert_comm_sizes_to_times(roots, network_model, interconnect_params):
    """Convert comm sizes to durations on a coarse event graph, in place.

    Free-function port of ``Graph.convert_comm_sizes_to_times`` (the legacy
    body verbatim, ``self`` dropped); duck-typed over the legacy ``Edge``
    attribute surface so it accepts :class:`~program.schedule.CommEvent`
    graphs (the coarse evaluator and the memory dispatcher's events hook).
    """

    def traverse_and_convert(node, visited=None):
        if visited is None:
            visited = set()
        if id(node) in visited:
            return
        visited.add(id(node))

        # Process children (edges and nodes)
        for child in node.children:
            # If it's an edge with communication size, convert to time
            if hasattr(child, "comm_size_bytes") and child.comm_size_bytes > 0:
                # Get the appropriate bandwidth/latency for this interconnect type
                interconnect_type = child.comm_interconnect_type
                if interconnect_type and interconnect_type in interconnect_params:
                    ib, ll = interconnect_params[interconnect_type]
                else:
                    raise ValueError(f"Invalid interconnect type: {interconnect_type}")
                if not isinstance(child.comm_type, CollectiveType):
                    raise TypeError(
                        f"Comm edge {getattr(child, 'name', '<unnamed>')} missing "
                        "CollectiveType comm_type"
                    )

                child.duration = network_model.collective(
                    kind=child.comm_type,
                    size_bytes=child.comm_size_bytes,
                    participants=child.participants,
                    ib=ib,
                    ll=ll,
                    local_bytes=0.0,
                    local_ops=0.0,
                    debug_label=f"{child.name}_conversion",
                )

            # Recursively process this child
            traverse_and_convert(child, visited)

    traverse_and_convert(roots)
    return roots


@dataclass
class CoarseEvalResult:
    """Detailed evaluation output (parity/differential surface).

    ``finish_times[uid]`` is the finish time of op ``uid`` (-1 when the op
    never ran — the legacy ``Node.finish_time`` reset value).
    """

    total_time: float
    finish_times: List[float]


def _coarse_events(program: Program) -> Tuple[Any, Tuple[Any, ...]]:
    if not isinstance(program, Program):
        raise TypeError(f"evaluate expects a Program (got {type(program).__name__})")
    if program.meta.misc.get("granularity") != "coarse":
        raise RuntimeError(
            "Analytical evaluation requires a COARSE program "
            "(program.pipeline_coarse.build_coarse_program)."
        )
    events = program.meta.misc.get("coarse_events")
    root = program.meta.misc.get("coarse_proto_root")
    if events is None or root is None:
        raise RuntimeError(
            "COARSE program does not carry its schedule events "
            "(meta.misc['coarse_events'/'coarse_proto_root'])."
        )
    return root, tuple(events)


def evaluate_detailed(
    program: Program,
    network_model: Any,
    interconnect_params: Mapping[str, Tuple[float, float]],
) -> CoarseEvalResult:
    """Convert comm sizes and replay the legacy list scheduler; see module
    docstring for the reproduced discipline."""
    root, events = _coarse_events(program)
    uid_of: Dict[int, int] = {id(event): uid for uid, event in enumerate(events)}

    convert_comm_sizes_to_times(root, network_model, interconnect_params)

    def _duration(event: Any) -> float:
        if isinstance(event, ComputeEvent):
            op = program.ops[uid_of[id(event)]]
            assert isinstance(op, ComputeOp)
            # Legacy Node.duration property: profile tuples read index 0.
            return op.duration[0]
        return event.duration

    # ---- device discovery (legacy: max Node hw_id over the reachable
    # graph, floored at Graph.pp; comm events carry no ``hw_id``) ----------
    base_devices = max(1, int(program.meta.misc.get("coarse_pp", 0) or 0))
    max_hw_id = -1
    visited_nodes: Set[int] = set()
    stack: List[Any] = [root]
    while stack:
        node = stack.pop()
        node_id = id(node)
        if node_id in visited_nodes:
            continue
        visited_nodes.add(node_id)
        hw_id = getattr(node, "hw_id", None)
        if hw_id is not None and int(hw_id) >= 0:
            max_hw_id = max(max_hw_id, int(hw_id))
        stack.extend(node.children)
    if max_hw_id >= 0:
        base_devices = max(base_devices, max_hw_id + 1)

    # ---- the legacy event loop ------------------------------------------
    time: float = 0
    counter = 0
    event_queue: List[Tuple[float, int, Any]] = []
    ready_list: List[Any] = []
    done: Set[int] = set()
    scheduled: Set[int] = set()
    finish: Dict[int, float] = {}

    GPU_list = [True for _ in range(base_devices)]

    ready_list.append(root)
    scheduled.add(id(root))
    heappush(event_queue, (_duration(root), counter, root))
    ready_list.remove(root)
    counter = counter + 1

    while len(event_queue) > 0:
        time, _, event = heappop(event_queue)
        done.add(id(event))
        scheduled.discard(id(event))
        finish[id(event)] = time

        for child in event.children:
            is_ready = True
            for parent in child.parents:
                if id(parent) not in done:
                    is_ready = False
            if (
                is_ready
                and (child not in ready_list)
                and (id(child) not in done)
                and (id(child) not in scheduled)
            ):
                ready_list.append(child)

        if isinstance(event, ComputeEvent):
            GPU_list[int(event.hw_id)] = True

        for event in ready_list[:]:
            if isinstance(event, ComputeEvent):
                if GPU_list[int(event.hw_id)] == True:  # noqa: E712 - legacy kept
                    new_time = time + _duration(event)
                    heappush(event_queue, (new_time, counter, event))
                    scheduled.add(id(event))
                    counter = counter + 1
                    GPU_list[int(event.hw_id)] = False
                    ready_list.remove(event)
            elif isinstance(event, CommEvent):
                new_time = time + _duration(event)
                heappush(event_queue, (new_time, counter, event))
                scheduled.add(id(event))
                counter = counter + 1
                ready_list.remove(event)

    finish_times = [finish.get(id(event), -1) for event in events]
    return CoarseEvalResult(total_time=time, finish_times=finish_times)


def evaluate(
    program: Program,
    network_model: Any,
    interconnect_params: Mapping[str, Tuple[float, float]],
) -> float:
    """Total analytical time of a COARSE program (legacy ``Graph.simulate``
    return value: the finish time of the last completed event)."""
    return evaluate_detailed(program, network_model, interconnect_params).total_time
