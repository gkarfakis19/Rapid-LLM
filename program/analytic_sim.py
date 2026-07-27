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

"""Analytical evaluation of COARSE :class:`~program.ir.Program`\\ s (P6).

Reads the **Program** — there is one representation. The proto event graph the
M5 evaluator walked (``meta.misc["coarse_events"]`` /
``["coarse_proto_root"]``) is gone, and with it the only reason the coarse
builder existed.

Two free functions:

* :func:`comm_durations` — the byte -> time conversion (the port of
  ``Graph.convert_comm_sizes_to_times``). It returns a per-uid duration vector
  instead of mutating a graph in place, so the analytical evaluator, the memory
  replay and the renderer all read the SAME numbers without a shared mutable
  object.
* :func:`evaluate_detailed` / :func:`evaluate` — the legacy list scheduler
  (``Graph.simulate``) over ``Program.ops``.

THE IR-LEVEL TIE DISCIPLINE (P6's normative deliverable)
=======================================================
The legacy event loop's outcome depended on **children-list adjacency order**:
the order ``add_child`` happened to be called in, i.e. an artifact of the
construction sequence. Nothing about it was declared, so it could not survive
the cutover. It is replaced by ONE rule:

    **PROGRAM ORDER (ascending uid) IS THE TIE DISCIPLINE.**

Concretely, and this is the whole contract:

1. the successors of a finished op are visited in ascending uid;
2. the ops that start ready (``deps == ()``) are seeded in ascending uid;
3. the ready list stays a FIFO, scanned front to back, exactly as the legacy
   loop scanned it.

Program order is ``kahn(slot, device, intra)`` (INTERFACES §4.6) — the same
total order AstraSim consumes as node-id priority (``CONTEXT.md``) — so the
replay is a pure function of the Program and agrees with the priority the
emitted bundle declares. Measured: on all 10 fully-analytical golden specs plus
every hybrid spec, uid order and construction order give **bit-identical**
totals, and both are bit-identical to the deleted proto-graph evaluator.

Preserved legacy semantics (each one measured, not assumed):

* one outstanding compute per device (``GPU_list`` boolean exclusivity); comm
  ops NEVER occupy a device slot;
* the event heap is keyed ``(finish_time, insertion_counter)``;
* same-device zero-byte ``TransferOp``\\ s are real heap events — they consume
  insertion counters and drive the ready scan (Class B item 9);
* the roots are pushed at t=0 **without** occupying their device;
* **a comm op that is a program ROOT carries no time.** The legacy conversion
  pass only ever converted a node's *children*, so the ZeRO-3 forward-entry
  gather — which IS the coarse root (``schedule.py:790``) — was never timed.
  Dropping the quirk moves ``train:analytical:dp2tp1cp1pp2mb2sp0:zero3`` by
  **+5.76%**, so it is preserved and named here (:data:`_ROOT_COMM_IS_UNTIMED`)
  rather than rediscovered later;
* compute durations read ``ComputeOp.duration[0]`` — the legacy
  ``Node.duration`` property over a per-DP profile tuple, which is how hybrid
  retiming (:mod:`program.retime`) reaches the evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Any, Collection, Dict, List, Mapping, Optional, Set, Tuple

from program.ir import CollectiveOp, ComputeOp, Program, TransferOp

__all__ = [
    "COMM_DURATIONS_KEY",
    "CoarseEvalResult",
    "comm_durations",
    "evaluate",
    "evaluate_detailed",
]

#: Preserved legacy quirk: the conversion pass converted *children* only, so a
#: comm op with no predecessor keeps duration 0. See the module docstring.
_ROOT_COMM_IS_UNTIMED = True

#: ``meta.misc`` key holding a per-uid comm-duration vector. Written by
#: :func:`evaluate_detailed` (and by the dispatcher for the memory replay), so
#: the renderer and :mod:`program.memory_sim` read the same numbers the
#: evaluator used instead of a second representation being mutated in place.
COMM_DURATIONS_KEY = "comm_durations"


def comm_durations(
    program: Program,
    network_model: Any,
    interconnect_params: Mapping[str, Tuple[float, float]],
    *,
    collective_keys: Optional[Collection[str]] = None,
) -> Tuple[float, ...]:
    """Per-uid comm durations (0.0 for compute ops and untimed comm ops).

    Port of ``Graph.convert_comm_sizes_to_times``: the same
    ``NetworkModel.collective`` call with the same arguments, for every comm op
    whose ``size_bytes > 0``. The conversion is a pure function of
    ``(kind, size, participants, ib, ll)``, so the legacy recursion order (and
    its duplicate conversions of multi-parent edges) is unobservable and is not
    reproduced.

    ``collective_keys`` restricts the conversion to ``CollectiveOp``s whose
    ``comm_key`` is in that set — the memory replay's rule, which passes the
    PIPELINE-LEVEL comm keys (``WorkloadSpec.comm``). Block-template comm keys
    are deliberately left untimed there; see :mod:`program.memory_sim`.
    """
    if not isinstance(program, Program):
        raise TypeError(
            f"comm_durations expects a Program (got {type(program).__name__})"
        )
    out: List[float] = [0.0] * len(program.ops)
    for op in program.ops:
        if isinstance(op, ComputeOp):
            continue
        if collective_keys is not None and (
            not isinstance(op, CollectiveOp) or op.comm_key not in collective_keys
        ):
            continue
        if _ROOT_COMM_IS_UNTIMED and not op.deps:
            continue
        size = float(op.size_bytes)
        if size <= 0:
            continue
        kind = op.coll if isinstance(op, CollectiveOp) else op.comm_type
        if kind is None:
            raise ValueError(f"Comm op {op.name!r} (uid {op.uid}) has no CollectiveType")
        axis = op.interconnect
        if not axis or axis not in interconnect_params:
            raise ValueError(f"Invalid interconnect type: {axis}")
        ib, ll = interconnect_params[axis]
        out[op.uid] = float(
            network_model.collective(
                kind=kind,
                size_bytes=size,
                participants=op.participants,
                ib=ib,
                ll=ll,
                local_bytes=0.0,
                local_ops=0.0,
                debug_label=f"{op.name}_conversion",
            )
        )
    return tuple(out)


@dataclass
class CoarseEvalResult:
    """Detailed evaluation output (parity/differential surface).

    ``finish_times[uid]`` is the finish time of op ``uid`` (-1 when the op never
    ran — the legacy ``Node.finish_time`` reset value).
    """

    total_time: float
    finish_times: List[float]


def _device_count(program: Program) -> int:
    """GPU-slot count: the legacy rule (``max(Graph.pp, max hw_id + 1)``)."""
    count = max(1, len(program.devices))
    for op in program.ops:
        if isinstance(op, ComputeOp):
            count = max(count, int(op.device) + 1)
    return count


def evaluate_detailed(
    program: Program,
    network_model: Any,
    interconnect_params: Mapping[str, Tuple[float, float]],
) -> CoarseEvalResult:
    """Convert comm sizes and replay the legacy list scheduler over the ops."""
    if not isinstance(program, Program):
        raise TypeError(f"evaluate expects a Program (got {type(program).__name__})")
    if program.meta.misc.get("granularity") != "coarse":
        raise RuntimeError(
            "Analytical evaluation requires a COARSE program "
            "(build(..., granularity=Granularity.COARSE))."
        )

    ops = program.ops
    durations = list(comm_durations(program, network_model, interconnect_params))
    for op in ops:
        if isinstance(op, ComputeOp):
            # Legacy Node.duration property: profile tuples read index 0.
            durations[op.uid] = float(op.duration[0])
    #: the renderer reads these back (the legacy pass mutated the events).
    program.meta.misc[COMM_DURATIONS_KEY] = tuple(
        0.0 if isinstance(op, ComputeOp) else durations[op.uid] for op in ops
    )

    # THE TIE DISCIPLINE: successors in ascending uid (module docstring).
    succs: List[List[int]] = [[] for _ in ops]
    for op in ops:
        for dep in op.deps:
            succs[dep].append(op.uid)
    for entry in succs:
        entry.sort()

    time: float = 0
    counter = 0
    heap: List[Tuple[float, int, int]] = []
    ready: List[int] = []
    done: Set[int] = set()
    scheduled: Set[int] = set()
    finish: Dict[int, float] = {}
    gpu_free = [True] * _device_count(program)

    # Roots at t=0, in program order, WITHOUT occupying their device.
    for op in ops:
        if not op.deps:
            heappush(heap, (durations[op.uid], counter, op.uid))
            scheduled.add(op.uid)
            counter += 1

    while heap:
        time, _, uid = heappop(heap)
        done.add(uid)
        scheduled.discard(uid)
        finish[uid] = time

        for child in succs[uid]:
            if child in done or child in scheduled or child in ready:
                continue
            if all(dep in done for dep in ops[child].deps):
                ready.append(child)

        if isinstance(ops[uid], ComputeOp):
            gpu_free[int(ops[uid].device)] = True

        for candidate in ready[:]:
            op = ops[candidate]
            if isinstance(op, ComputeOp):
                if gpu_free[int(op.device)]:
                    heappush(heap, (time + durations[candidate], counter, candidate))
                    scheduled.add(candidate)
                    counter += 1
                    gpu_free[int(op.device)] = False
                    ready.remove(candidate)
            else:
                heappush(heap, (time + durations[candidate], counter, candidate))
                scheduled.add(candidate)
                counter += 1
                ready.remove(candidate)

    return CoarseEvalResult(
        total_time=time,
        finish_times=[finish.get(op.uid, -1) for op in ops],
    )


def evaluate(
    program: Program,
    network_model: Any,
    interconnect_params: Mapping[str, Tuple[float, float]],
) -> float:
    """Total analytical time of a COARSE program (legacy ``Graph.simulate``
    return value: the finish time of the last completed event)."""
    return evaluate_detailed(program, network_model, interconnect_params).total_time
