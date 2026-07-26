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

"""Peak-memory replay over a FINE :class:`~program.ir.Program` (M3b).

Verbatim-semantics port of the deleted ``Graph.simulate_memory``
(simulate_train_graph.py, formerly lines 1477-1877) onto the FINE builder's
output. The engine is the legacy list scheduler exactly — heap keyed
``(finish_time, insertion_counter)``, per-device boolean compute
exclusivity, comm events slot-free, FIFO ready-list scanned after every
completion, children appended in graph children-list order — with the
legacy memory hooks:

* static per-device init from dense/moe transformer-layer counts per device
  (census over op metadata: ``mem_kind``/``layer_index``/``is_moe_layer``/
  ``fwd`` — never node names), with the legacy zero-layer fallback
  (``dense=1`` per device);
* persistent activation alloc on forward compute completion, released on
  the matching backward completion; transient alloc+release at completion;
  inference mode allocates transients only;
* ZeRO-3 ``param_gather`` ephemeral bytes allocated at compute ISSUE and
  released at completion;
* the ``_MemorySnapshot`` per-device accounting, peak tracking, and the
  opt-in per-device event logs (``RAPID_DEBUG_MEMORY`` /
  ``RAPID_MEMORY_EVENT_LOGS``) byte-for-byte.

The replay runs over the fine builder's proto graph
(``program.meta.misc["fine_proto_root"]``, :class:`~program.pipeline_fine.
FineNode`/``FineEdge``) rather than the uid-ordered op list: the event
loop's FIFO discipline depends on the children-list adjacency order, which
uid order (the per-stage Kahn emission order) does not preserve. The proto
root is the fine builder's own product — no legacy graph or flattener is
involved. M8 resolution: the proto graph is NOT collapsed into the op
list — this adjacency-order dependence is why it stays, by design.
"""

from __future__ import annotations

import os
from heapq import heappop, heappush
from typing import Any, Dict, List, Optional, Set, Tuple

from memory_estimation import MemKind, NON_TRANSFORMER_KINDS, TRANSFORMER_OP_KINDS
from program.ir import Program
from program.pipeline_fine import FineEdge, FineNode

debug = False
BYTES_PER_GIB = 1024 ** 3


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized not in {"", "0", "false", "no"}


def fine_proto_root(program: Program) -> Any:
    """Return the FINE proto root the memory replay consumes, or raise."""
    if not isinstance(program, Program):
        raise TypeError(f"simulate_memory expects a Program (got {type(program).__name__})")
    if program.meta.misc.get("granularity") != "fine":
        raise RuntimeError(
            "Memory simulation requires a FINE program. "
            "Use LLMExecutionDispatcher.build_fine_program_for_memory()."
        )
    root = program.meta.misc.get("fine_proto_root")
    if root is None:
        raise RuntimeError(
            "FINE program does not carry its proto root "
            "(meta.misc['fine_proto_root']); it was not built by build_fine_program()."
        )
    return root


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
        node: Any,
        timestamp: float,
        size_bytes: float,
        action: str = "allocate_activation",
    ) -> None:
        if size_bytes <= 0 or gpu_id < 0 or gpu_id >= len(self.current):
            return
        self.current[gpu_id] += size_bytes
        details = "node={} op_id={} fwd={}".format(
            getattr(node, "name", "unknown"),
            getattr(node, "op_id", "N/A"),
            getattr(node, "fwd", "N/A"),
        )
        self._update_peak(gpu_id)
        self._record_change(gpu_id, action, size_bytes, timestamp, details=details)

    def release_activation(
        self,
        gpu_id: int,
        node: Any,
        timestamp: float,
        size_bytes: float,
        action: str = "release_activation",
    ) -> None:
        if size_bytes <= 0 or gpu_id < 0 or gpu_id >= len(self.current):
            return
        self.current[gpu_id] = max(0.0, self.current[gpu_id] - size_bytes)
        details = "node={} op_id={} fwd={}".format(
            getattr(node, "name", "unknown"),
            getattr(node, "op_id", "N/A"),
            getattr(node, "fwd", "N/A"),
        )
        self._record_change(gpu_id, action, -size_bytes, timestamp, details=details)

    def allocate_ephemeral(self, gpu_id: int, node: Any, timestamp: float, size_bytes: float) -> None:
        self.allocate_activation(
            gpu_id,
            node,
            timestamp,
            size_bytes,
            action="allocate_ephemeral",
        )

    def release_ephemeral(self, gpu_id: int, node: Any, timestamp: float, size_bytes: float) -> None:
        self.release_activation(
            gpu_id,
            node,
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


def _reset_execution_state(root: Any) -> None:
    """Clear scheduling state so the proto graph can be (re-)simulated.

    Port of ``Graph._reset_execution_state``; FineNode/FineEdge do not
    pre-carry the scheduling attributes, so they are stamped unconditionally.
    """
    if root is None:
        return
    visited: Set[int] = set()
    stack: List[Any] = list(root) if isinstance(root, (list, tuple, set)) else [root]
    while stack:
        obj = stack.pop()
        obj_id = id(obj)
        if obj_id in visited:
            continue
        visited.add(obj_id)
        obj.done = False
        obj.scheduled = False
        obj.finish_time = -1
        stack.extend(getattr(obj, "children", []))


def simulate_memory(
    program: Program,
    memory_data: Dict[str, Any],
    mode: str = "training",
    output_folder: str = "output/LLM/",
    filename: str = "memory_graph",
) -> Tuple[float, float]:
    """Replay the FINE program and return ``(finish_time, peak_gib)``.

    The full per-device summary (the legacy ``memory_monitor_summary``) is
    stored at ``program.meta.misc["memory_summary"]``.
    """
    root = fine_proto_root(program)
    pipeline_pp = int(program.meta.misc.get("fine_pp", 0) or 0)

    time = 0
    counter = 0
    event_queue: List[Tuple[float, int, Any]] = []
    ready_list: List[Any] = []

    _reset_execution_state(root)
    ready_list.append(root)
    root.scheduled = True

    persistent_by_kind = memory_data.get("persistent_bytes_by_kind", {}) or {}
    transient_by_kind = memory_data.get("transient_bytes_by_kind", {}) or {}
    persistent_by_kind_dense = memory_data.get("persistent_bytes_by_kind_dense", persistent_by_kind) or {}
    persistent_by_kind_moe = memory_data.get("persistent_bytes_by_kind_moe", persistent_by_kind_dense) or {}
    transient_by_kind_dense = memory_data.get("transient_bytes_by_kind_dense", transient_by_kind) or {}
    transient_by_kind_moe = memory_data.get("transient_bytes_by_kind_moe", transient_by_kind_dense) or {}
    valid_kinds = set(persistent_by_kind) | set(transient_by_kind)
    param_gather_bytes = float(memory_data.get("param_gather_bytes", 0.0) or 0.0)

    def _is_transformer_block(node: Any) -> bool:
        """True when the node represents a transformer compute block
        (metadata-only: ``mem_kind`` membership, never node names)."""
        if not isinstance(node, FineNode):
            return False
        mem_kind = getattr(node, "mem_kind", None)
        if not isinstance(mem_kind, MemKind):
            return False
        if mem_kind in NON_TRANSFORMER_KINDS:
            return False
        return mem_kind in TRANSFORMER_OP_KINDS or mem_kind in valid_kinds

    def _node_mem_kind(node: Any) -> Optional[MemKind]:
        if not isinstance(node, FineNode):
            return None
        mem_kind = getattr(node, "mem_kind", None)
        return mem_kind if isinstance(mem_kind, MemKind) else None

    def _bytes_for_kind(kind: Optional[MemKind], mapping: Dict[MemKind, float]) -> float:
        if kind not in mapping:
            return 0.0
        return float(mapping[kind] or 0.0)

    def _persistent_bytes_for_node(node: Any) -> float:
        mem_kind = _node_mem_kind(node)
        mapping = (
            persistent_by_kind_moe
            if getattr(node, "is_moe_layer", False) and _is_transformer_block(node)
            else persistent_by_kind_dense
        )
        return _bytes_for_kind(mem_kind, mapping)

    def _transient_bytes_for_node(node: Any) -> float:
        mem_kind = _node_mem_kind(node)
        mapping = (
            transient_by_kind_moe
            if getattr(node, "is_moe_layer", False) and _is_transformer_block(node)
            else transient_by_kind_dense
        )
        return _bytes_for_kind(mem_kind, mapping)

    def _collect_graph_layout(root_obj: Any) -> Tuple[int, List[int], List[int]]:
        """Census: dense/moe forward-layer counts per device (op metadata)."""
        base_devices_local = max(1, pipeline_pp if pipeline_pp else 1)
        max_hw_id_local = -1
        dense_layers_by_device: Dict[int, Set[Any]] = {}
        moe_layers_by_device: Dict[int, Set[Any]] = {}
        visited_local: Set[int] = set()
        stack_local = list(root_obj if isinstance(root_obj, (list, tuple)) else [root_obj])

        while stack_local:
            current = stack_local.pop()
            current_id = id(current)
            if current_id in visited_local:
                continue
            visited_local.add(current_id)

            children = getattr(current, "children", None)
            if isinstance(children, (list, tuple)):
                stack_local.extend(children)
            elif children is not None:
                stack_local.append(children)

            if not isinstance(current, FineNode):
                continue

            hw_id = getattr(current, "hw_id", None)
            try:
                hw_val = int(hw_id)
            except (TypeError, ValueError):
                hw_val = None
            if hw_val is None or hw_val < 0:
                continue

            max_hw_id_local = max(max_hw_id_local, hw_val)
            layer_idx = getattr(current, "layer_index", None)
            if (
                layer_idx is None
                or not _is_transformer_block(current)
                or not getattr(current, "fwd", True)
            ):
                continue
            if getattr(current, "is_moe_layer", False):
                moe_layers_by_device.setdefault(hw_val, set()).add(layer_idx)
            else:
                dense_layers_by_device.setdefault(hw_val, set()).add(layer_idx)

        if max_hw_id_local >= 0:
            base_devices_local = max(base_devices_local, max_hw_id_local + 1)

        dense_counts = [len(dense_layers_by_device.get(idx, ())) for idx in range(base_devices_local)]
        moe_counts = [len(moe_layers_by_device.get(idx, ())) for idx in range(base_devices_local)]
        return base_devices_local, dense_counts, moe_counts

    write_memory_logs = _env_flag("RAPID_DEBUG_MEMORY") or _env_flag("RAPID_MEMORY_EVENT_LOGS")
    base_devices, transformer_dense_layers_per_device, transformer_moe_layers_per_device = (
        _collect_graph_layout(root)
    )

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

    GPU_list = [True for _ in range(base_devices)]
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

    heappush(event_queue, (root.duration, counter, root))
    if debug:
        print("{} enqueued at time 0".format(root.name))
    ready_list.remove(root)
    counter = counter + 1

    while len(event_queue) > 0:
        time, _, event = heappop(event_queue)
        event.done = True
        event.scheduled = False
        event.finish_time = time
        if debug:
            print("Event {} finished at time {}".format(event.name, time))

        for child in event.children:
            is_ready = True
            for parent in child.parents:
                if parent.done == False:  # noqa: E712 - legacy comparison kept
                    is_ready = False
            if is_ready and (child not in ready_list) and (not child.done) and (not child.scheduled):
                ready_list.append(child)
                if debug:
                    print("child {}  ready at time {} ".format(child.name, time))

        if isinstance(event, FineNode):
            GPU_list[int(event.hw_id)] = True
            persistent_bytes = _persistent_bytes_for_node(event)
            transient_bytes = _transient_bytes_for_node(event)
            if mode == "training":
                if event.fwd and persistent_bytes:
                    memory_snapshot.allocate_activation(int(event.hw_id), event, time, persistent_bytes)
                if event.fwd and transient_bytes:
                    memory_snapshot.allocate_activation(int(event.hw_id), event, time, transient_bytes)
                    memory_snapshot.release_activation(int(event.hw_id), event, time, transient_bytes)
                if not event.fwd and persistent_bytes:
                    memory_snapshot.release_activation(int(event.hw_id), event, time, persistent_bytes)
            elif mode == "inference":
                if event.fwd and transient_bytes:
                    memory_snapshot.allocate_activation(int(event.hw_id), event, time, transient_bytes)
                    memory_snapshot.release_activation(int(event.hw_id), event, time, transient_bytes)
            if mode == "training" and param_gather_bytes is not None and getattr(event, "param_gather", False):
                memory_snapshot.release_ephemeral(
                    int(event.hw_id),
                    event,
                    time,
                    param_gather_bytes,
                )

        # FIFO ready scan (legacy order). The proto graph carries no
        # Data_batch events (the schedule enumeration drops them exactly as
        # the legacy constructor removed them before returning the root).
        for event in ready_list[:]:
            if isinstance(event, FineNode):
                if GPU_list[int(event.hw_id)] == True:  # noqa: E712 - legacy kept
                    new_time = time + event.duration
                    if mode == "training" and param_gather_bytes and getattr(event, "param_gather", False):
                        memory_snapshot.allocate_ephemeral(
                            int(event.hw_id),
                            event,
                            time,
                            param_gather_bytes,
                        )
                    heappush(event_queue, (new_time, counter, event))
                    event.scheduled = True
                    if debug:
                        print(
                            "{}.{} enqueued at time {} at device {}".format(
                                event.name, event.op_id, time, event.hw_id
                            )
                        )
                    counter = counter + 1
                    GPU_list[int(event.hw_id)] = False
                    ready_list.remove(event)
            elif isinstance(event, FineEdge):
                new_time = time + event.duration
                heappush(event_queue, (new_time, counter, event))
                event.scheduled = True
                if debug:
                    print("{}.{} enqueued at time {}".format(event.name, event.op_id, time))
                counter = counter + 1
                ready_list.remove(event)

    summary = memory_snapshot.summary()
    memory_snapshot.close()
    program.meta.misc["memory_summary"] = summary
    peak_mem = max(entry["peak_gib"] for entry in summary) if summary else 0.0
    return time, peak_mem
