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

"""Peak-memory replay over a FLAT :class:`~program.ir.Program` (P6).

Reads the **Program**. The FLAT proto graph (``meta.misc["fine_proto_root"]``,
``FineNode``/``FineEdge``) is gone; the census and the scheduling state that
used to live on it are ``ComputeOp`` fields (``mem_kind``, ``layer``,
``is_moe_layer``, ``direction``, ``param_gather``) and evaluator locals.

The tie discipline is the one :mod:`program.analytic_sim` documents:
**ascending uid** for successors and for the initially-ready set, FIFO ready
list. Program order is ``kahn(slot, device, intra)`` (INTERFACES §4.6).

Preserved legacy memory hooks (``Graph.simulate_memory``), unchanged:

* static per-device init from dense/MoE transformer-layer counts per device
  (census over op metadata — ``mem_kind``/``layer``/``is_moe_layer``/direction,
  never names), with the legacy zero-layer fallback (``dense=1`` per device);
* persistent activation alloc on forward compute completion, released on the
  matching backward completion; transient alloc+release at completion;
  inference mode allocates transients only;
* ZeRO-3 ``param_gather`` ephemeral bytes allocated at compute ISSUE and
  released at completion;
* the ``_MemorySnapshot`` per-device accounting, peak tracking and the opt-in
  per-device event logs (``RAPID_DEBUG_MEMORY`` / ``RAPID_MEMORY_EVENT_LOGS``).

**Comm durations.** Legacy replayed the analytical byte->time conversion on the
*coarse* events before flattening (``build_fine_program(events_hook=...)``), so
in the flattened memory graph exactly the cloned PIPELINE-LEVEL COLLECTIVE edges
(the dp / ZeRO reducers and gathers, and the EP grad sync) carried a nonzero
duration:

* the per-rank cross-stage pipeline transfers were created FRESH with
  ``duration=0`` (``pipeline_fine.py:661``) — the coarse conversion never
  reached them;
* the same-stage control edges carry zero bytes, so the conversion skipped them
  anyway;
* the block-template comm edges were created AFTER the hook ran
  (``pipeline_fine.py:753``), also at ``duration=0``.

Embedding/softmax pipeline edges are always same-stage
(``stage_for_layer(0) == 0``, and softmax sits on ``pp-1`` with the last layer),
so "the pipeline-level collectives only" is not an approximation of the old
behavior — it *is* the old behavior, for every representable configuration. The
rule is expressed against the L0 namespace split (INTERFACES §1.6 amendment B1):
a collective is timed iff its ``comm_key`` is in ``WorkloadSpec.comm``.

The vector is therefore supplied by the dispatcher, which is the only thing
that knows whether the analytical conversion happened at all, and is carried on
the artifact as ``meta.misc[analytic_sim.COMM_DURATIONS_KEY]``; absent means
every comm op is untimed (the flattened/hierarchical modes, which never ran the
conversion).
"""

from __future__ import annotations

import os
from heapq import heappop, heappush
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from memory_estimation import MemKind, NON_TRANSFORMER_KINDS, TRANSFORMER_OP_KINDS
from program import _env_flag
from program.analytic_sim import COMM_DURATIONS_KEY
from program.ir import ComputeOp, Direction, Program

debug = False
BYTES_PER_GIB = 1024 ** 3


def require_flat_program(program: Program) -> Program:
    """Return ``program`` when it is a FLAT Program, else raise."""
    if not isinstance(program, Program):
        raise TypeError(
            f"simulate_memory expects a Program (got {type(program).__name__})"
        )
    if program.meta.misc.get("granularity") != "flat":
        raise RuntimeError(
            "Memory simulation requires a FLAT program. "
            "Use LLMExecutionDispatcher.build_flat_program_for_memory()."
        )
    return program


class _MemorySnapshot:
    """Per-device memory ledger — verbatim ``Graph.simulate_memory`` port."""

    def __init__(self, num_devices: int, output_dir: Optional[str], file_basename: str) -> None:
        self.static: List[float] = [0.0 for _ in range(num_devices)]
        self.current: List[float] = [0.0 for _ in range(num_devices)]
        self.peak: List[float] = [0.0 for _ in range(num_devices)]
        self._log_files: List[Optional[Any]] = [None for _ in range(num_devices)]
        self._log_paths: List[Optional[str]] = [None for _ in range(num_devices)]
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for gpu_idx in range(num_devices):
                log_path = os.path.join(output_dir, f"{file_basename}_gpu_{gpu_idx}_memory_log.txt")
                self._log_paths[gpu_idx] = log_path
                log_file = open(log_path, "w", encoding="utf-8", buffering=1024 * 1024)
                log_file.write(
                    "timestamp_s | action               | delta_gib   | static_gib  | current_gib | peak_gib    | details\n"
                )
                self._log_files[gpu_idx] = log_file

    def add_static(self, gpu_id: int, size_bytes: float, timestamp: float = 0.0) -> None:
        if size_bytes <= 0:
            return
        self.static[gpu_id] += size_bytes
        self.current[gpu_id] += size_bytes
        self._update_peak(gpu_id)
        self._record_change(
            gpu_id,
            "static_allocate",
            size_bytes,
            timestamp,
            details="static_mem",
        )

    def allocate_activation(
        self,
        gpu_id: int,
        op: Any,
        timestamp: float,
        size_bytes: float,
        action: str = "allocate_activation",
    ) -> None:
        if size_bytes <= 0 or gpu_id < 0 or gpu_id >= len(self.current):
            return
        self.current[gpu_id] += size_bytes
        details = _details(op)
        self._update_peak(gpu_id)
        self._record_change(gpu_id, action, size_bytes, timestamp, details=details)

    def release_activation(
        self,
        gpu_id: int,
        op: Any,
        timestamp: float,
        size_bytes: float,
        action: str = "release_activation",
    ) -> None:
        if size_bytes <= 0 or gpu_id < 0 or gpu_id >= len(self.current):
            return
        self.current[gpu_id] = max(0.0, self.current[gpu_id] - size_bytes)
        details = _details(op)
        self._record_change(gpu_id, action, -size_bytes, timestamp, details=details)

    def allocate_ephemeral(self, gpu_id: int, op: Any, timestamp: float, size_bytes: float) -> None:
        self.allocate_activation(
            gpu_id,
            op,
            timestamp,
            size_bytes,
            action="allocate_ephemeral",
        )

    def release_ephemeral(self, gpu_id: int, op: Any, timestamp: float, size_bytes: float) -> None:
        self.release_activation(
            gpu_id,
            op,
            timestamp,
            size_bytes,
            action="release_ephemeral",
        )

    def summary(self) -> List[Dict[str, float]]:
        summary: List[Dict[str, float]] = []
        for idx in range(len(self.current)):
            static_gib = self.static[idx] / BYTES_PER_GIB
            current_gib = self.current[idx] / BYTES_PER_GIB
            peak_gib = self.peak[idx] / BYTES_PER_GIB
            summary.append(
                {
                    "gpu_id": idx,
                    "static_gib": static_gib,
                    "current_gib": current_gib,
                    "peak_gib": peak_gib,
                }
            )
        return summary

    def _update_peak(self, gpu_id: int) -> None:
        if self.current[gpu_id] > self.peak[gpu_id]:
            self.peak[gpu_id] = self.current[gpu_id]

    def _record_change(
        self,
        gpu_id: int,
        action: str,
        delta_bytes: float,
        timestamp: float,
        *,
        details: str = "",
    ) -> None:
        if not self._log_files or gpu_id < 0 or gpu_id >= len(self._log_files):
            return
        log_file = self._log_files[gpu_id]
        if log_file is None:
            return

        delta_gib = delta_bytes / BYTES_PER_GIB
        static_gib = self.static[gpu_id] / BYTES_PER_GIB
        current_gib = self.current[gpu_id] / BYTES_PER_GIB
        peak_gib = self.peak[gpu_id] / BYTES_PER_GIB
        line = "{:.6f} | {:<20} | {:>+.6f} | {:>+.6f} | {:>+.6f} | {:>+.6f}".format(
            timestamp,
            action,
            delta_gib,
            static_gib,
            current_gib,
            peak_gib,
        )
        if details:
            line = f"{line} | {details}"
        log_file.write(line + "\n")

    def close(self) -> None:
        if not self._log_files:
            return
        for handle in self._log_files:
            if handle:
                handle.close()


def _details(op: Any) -> str:
    return "node={} op_id={} fwd={}".format(
        getattr(op, "name", "unknown"),
        getattr(op, "uid", "N/A"),
        _is_forward(op),
    )


def _is_forward(op: Any) -> bool:
    """The legacy ``Node.fwd`` flag, from the typed direction."""
    return getattr(op, "direction", Direction.FORWARD) is Direction.FORWARD


def simulate_memory(
    program: Program,
    memory_data: Dict[str, Any],
    mode: str = "training",
    output_folder: str = "output/LLM/",
    filename: str = "memory_graph",
) -> Tuple[float, float]:
    """Replay the FLAT program and return ``(finish_time, peak_gib)``."""
    require_flat_program(program)
    ops = program.ops

    persistent_by_kind = memory_data.get("persistent_bytes_by_kind", {}) or {}
    transient_by_kind = memory_data.get("transient_bytes_by_kind", {}) or {}
    persistent_by_kind_dense = memory_data.get("persistent_bytes_by_kind_dense", persistent_by_kind) or {}
    persistent_by_kind_moe = memory_data.get("persistent_bytes_by_kind_moe", persistent_by_kind_dense) or {}
    transient_by_kind_dense = memory_data.get("transient_bytes_by_kind_dense", transient_by_kind) or {}
    transient_by_kind_moe = memory_data.get("transient_bytes_by_kind_moe", transient_by_kind_dense) or {}
    valid_kinds = set(persistent_by_kind) | set(transient_by_kind)
    param_gather_bytes = float(memory_data.get("param_gather_bytes", 0.0) or 0.0)

    def _is_transformer_block(op: Any) -> bool:
        """True when the op is a transformer compute block (metadata only:
        ``mem_kind`` membership, never op names)."""
        if not isinstance(op, ComputeOp):
            return False
        mem_kind = op.mem_kind
        if not isinstance(mem_kind, MemKind):
            return False
        if mem_kind in NON_TRANSFORMER_KINDS:
            return False
        return mem_kind in TRANSFORMER_OP_KINDS or mem_kind in valid_kinds

    def _op_mem_kind(op: Any) -> Optional[MemKind]:
        if not isinstance(op, ComputeOp):
            return None
        return op.mem_kind if isinstance(op.mem_kind, MemKind) else None

    def _bytes_for_kind(kind: Optional[MemKind], mapping: Dict[MemKind, float]) -> float:
        if kind not in mapping:
            return 0.0
        return float(mapping[kind] or 0.0)

    def _persistent_bytes_for_op(op: Any) -> float:
        mem_kind = _op_mem_kind(op)
        mapping = (
            persistent_by_kind_moe
            if getattr(op, "is_moe_layer", False) and _is_transformer_block(op)
            else persistent_by_kind_dense
        )
        return _bytes_for_kind(mem_kind, mapping)

    def _transient_bytes_for_op(op: Any) -> float:
        mem_kind = _op_mem_kind(op)
        mapping = (
            transient_by_kind_moe
            if getattr(op, "is_moe_layer", False) and _is_transformer_block(op)
            else transient_by_kind_dense
        )
        return _bytes_for_kind(mem_kind, mapping)

    # ---- census: dense/MoE forward-layer counts per device ----------------
    base_devices = max(1, len(program.devices))
    dense_layers_by_device: Dict[int, Set[Any]] = {}
    moe_layers_by_device: Dict[int, Set[Any]] = {}
    for op in ops:
        if not isinstance(op, ComputeOp):
            continue
        device = int(op.device)
        if device < 0:
            continue
        base_devices = max(base_devices, device + 1)
        if op.layer is None or not _is_transformer_block(op) or not _is_forward(op):
            continue
        if op.is_moe_layer:
            moe_layers_by_device.setdefault(device, set()).add(op.layer)
        else:
            dense_layers_by_device.setdefault(device, set()).add(op.layer)

    transformer_dense_layers_per_device = [
        len(dense_layers_by_device.get(idx, ())) for idx in range(base_devices)
    ]
    transformer_moe_layers_per_device = [
        len(moe_layers_by_device.get(idx, ())) for idx in range(base_devices)
    ]

    write_memory_logs = _env_flag("RAPID_DEBUG_MEMORY") or _env_flag("RAPID_MEMORY_EVENT_LOGS")
    memory_output_dir: Optional[str] = None
    if output_folder and write_memory_logs:
        memory_output_dir = os.path.join(output_folder, "memory-summary")
    static_mem_per_layer = memory_data.get("static_mem_per_layer", 0)
    static_mem_per_layer_dense = memory_data.get("static_mem_per_layer_dense", static_mem_per_layer)
    static_mem_per_layer_moe = memory_data.get("static_mem_per_layer_moe", static_mem_per_layer_dense)
    weight_mem_per_layer = memory_data.get("weight_mem_per_layer", 0)
    weight_mem_per_layer_dense = memory_data.get("weight_mem_per_layer_dense", weight_mem_per_layer)
    weight_mem_per_layer_moe = memory_data.get("weight_mem_per_layer_moe", weight_mem_per_layer_dense)
    kv_cache_bytes_per_layer = memory_data.get("kv_cache_bytes_per_layer", 0.0)
    extra_static_bytes_per_device = memory_data.get("extra_static_bytes_per_device", {}) or {}
    total_tracked_layers = sum(transformer_dense_layers_per_device) + sum(transformer_moe_layers_per_device)
    if total_tracked_layers == 0:
        # Legacy zero-layer fallback: one dense layer per device.
        transformer_dense_layers_per_device = [1 for _ in range(base_devices)]
        transformer_moe_layers_per_device = [0 for _ in range(base_devices)]

    gpu_free = [True for _ in range(base_devices)]
    memory_snapshot = _MemorySnapshot(base_devices, memory_output_dir, filename)

    for idx in range(base_devices):
        if mode == "inference":
            dense_count = (
                transformer_dense_layers_per_device[idx]
                if idx < len(transformer_dense_layers_per_device)
                else 0
            )
            moe_count = (
                transformer_moe_layers_per_device[idx]
                if idx < len(transformer_moe_layers_per_device)
                else 0
            )
            layer_count = dense_count + moe_count
            if layer_count == 0:
                dense_count = 1
                layer_count = 1
            static_bytes = (weight_mem_per_layer_dense * dense_count) + (
                weight_mem_per_layer_moe * moe_count
            )
            if kv_cache_bytes_per_layer:
                static_bytes += kv_cache_bytes_per_layer * layer_count
        elif mode == "training":
            dense_count = (
                transformer_dense_layers_per_device[idx]
                if idx < len(transformer_dense_layers_per_device)
                else 0
            )
            moe_count = (
                transformer_moe_layers_per_device[idx]
                if idx < len(transformer_moe_layers_per_device)
                else 0
            )
            layer_count = dense_count + moe_count
            if layer_count == 0:
                dense_count = 1
                layer_count = 1
            static_bytes = (static_mem_per_layer_dense * dense_count) + (
                static_mem_per_layer_moe * moe_count
            )
        else:
            raise ValueError(f"Invalid mode '{mode}' for memory simulation")
        static_bytes += extra_static_bytes_per_device.get(idx, 0.0)
        memory_snapshot.add_static(idx, static_bytes, timestamp=0.0)

    # ---- durations + the tie discipline ----------------------------------
    stored: Optional[Sequence[float]] = program.meta.misc.get(COMM_DURATIONS_KEY)
    durations: List[float] = [0.0] * len(ops)
    for op in ops:
        if isinstance(op, ComputeOp):
            durations[op.uid] = float(op.duration[0])
        elif stored is not None:
            durations[op.uid] = float(stored[op.uid])

    succs: List[List[int]] = [[] for _ in ops]
    for op in ops:
        for dep in op.deps:
            succs[dep].append(op.uid)
    for entry in succs:
        entry.sort()

    # ---- per-op precomputation (perf pass, 2026-08-02) --------------------
    # The replay used to re-derive "is this a transformer compute / which
    # MemKind / how many bytes / is it forward" through closure chains on
    # EVERY event, scan `ready` (a list) for membership per child, and copy
    # the whole ready list per event. All of it is a pure function of the op,
    # so it is tabulated once here; readiness is a dependency COUNTDOWN, which
    # fires exactly when the old `all(dep in done)` scan did and appends
    # children in the same sorted-successor order, so the FIFO order — and the
    # peak — are bit-identical. GPT 175B FLAT (~1.2M ops): 8.5 s -> ~3 s.
    is_compute_arr: List[bool] = [isinstance(op, ComputeOp) for op in ops]
    persistent_arr: List[float] = [0.0] * len(ops)
    transient_arr: List[float] = [0.0] * len(ops)
    forward_arr: List[bool] = [False] * len(ops)
    gather_arr: List[bool] = [False] * len(ops)
    device_arr: List[int] = [0] * len(ops)
    for op in ops:
        if isinstance(op, ComputeOp):
            uid = int(op.uid)
            persistent_arr[uid] = _persistent_bytes_for_op(op)
            transient_arr[uid] = _transient_bytes_for_op(op)
            forward_arr[uid] = _is_forward(op)
            gather_arr[uid] = bool(op.param_gather)
            device_arr[uid] = int(op.device)
    pending: List[int] = [len(op.deps) for op in ops]

    time: float = 0
    counter = 0
    heap: List[Tuple[float, int, int]] = []
    ready: List[int] = []

    for op in ops:
        if not op.deps:
            heappush(heap, (durations[op.uid], counter, op.uid))
            counter += 1
            if debug:
                print("{} enqueued at time 0".format(op.name))

    training = mode == "training"
    inference = mode == "inference"
    while heap:
        time, _, uid = heappop(heap)
        if debug:
            print("Event {} finished at time {}".format(ops[uid].name, time))

        for child in succs[uid]:
            pending[child] -= 1
            if pending[child] == 0:
                ready.append(child)
                if debug:
                    print("child {}  ready at time {} ".format(ops[child].name, time))

        if is_compute_arr[uid]:
            event = ops[uid]
            device = device_arr[uid]
            gpu_free[device] = True
            forward = forward_arr[uid]
            persistent_bytes = persistent_arr[uid]
            transient_bytes = transient_arr[uid]
            if training:
                if forward and persistent_bytes:
                    memory_snapshot.allocate_activation(device, event, time, persistent_bytes)
                if forward and transient_bytes:
                    memory_snapshot.allocate_activation(device, event, time, transient_bytes)
                    memory_snapshot.release_activation(device, event, time, transient_bytes)
                if not forward and persistent_bytes:
                    memory_snapshot.release_activation(device, event, time, persistent_bytes)
            elif inference:
                if forward and transient_bytes:
                    memory_snapshot.allocate_activation(device, event, time, transient_bytes)
                    memory_snapshot.release_activation(device, event, time, transient_bytes)
            if training and param_gather_bytes is not None and gather_arr[uid]:
                memory_snapshot.release_ephemeral(device, event, time, param_gather_bytes)

        # FIFO ready scan (legacy order). Rebuild-in-place: same scan order,
        # same decisions, without the per-event list copy + O(n) removes.
        if ready:
            still_waiting: List[int] = []
            for candidate in ready:
                if is_compute_arr[candidate]:
                    device = device_arr[candidate]
                    if gpu_free[device]:
                        if training and param_gather_bytes and gather_arr[candidate]:
                            memory_snapshot.allocate_ephemeral(
                                device, ops[candidate], time, param_gather_bytes
                            )
                        heappush(heap, (time + durations[candidate], counter, candidate))
                        if debug:
                            print(
                                "{}.{} enqueued at time {} at device {}".format(
                                    ops[candidate].name, candidate, time, device
                                )
                            )
                        counter = counter + 1
                        gpu_free[device] = False
                    else:
                        still_waiting.append(candidate)
                else:
                    heappush(heap, (time + durations[candidate], counter, candidate))
                    if debug:
                        print(
                            "{}.{} enqueued at time {}".format(
                                ops[candidate].name, candidate, time
                            )
                        )
                    counter = counter + 1
            ready = still_waiting

    summary = memory_snapshot.summary()
    memory_snapshot.close()
    peak_mem = max(entry["peak_gib"] for entry in summary) if summary else 0.0
    return time, peak_mem
