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

"""Event graph -> Program emission-ordering pass (PERMANENT — kept at M8).

``lower_to_program`` turns a proto event DAG — the coarse schedule events
(:mod:`program.schedule`) for the hierarchical pipeline emission
(:func:`program.pipeline_coarse.lower_coarse_for_emission`) — into a
validated, emission-ordered :class:`~program.ir.Program`. It is deliberately
load-bearing after the legacy retirement: the pinned Chakra ET emission
order (per-stage Kahn toposort keyed on ``op_id``, Step-11 transfer replay,
collective label assignment) is defined over children-list adjacency order,
which the uid-ordered coarse op list cannot represent, so the ordering pass
over the events IS the single source of that order. The name records its
lineage: it is an exact reproduction of converter Steps 1-7 of the deleted
``astrasim_lib.executor.convert_rapid_llm_graph_to_chakra_et`` (executor.py
747-1334), originally applied to the (retired) legacy
``Node``/``Edge``/``Data_batch`` graphs, whose duck-typed attribute surface
the schedule events preserve:

* Step 1: DFS object collection order (preorder, children in list order);
* Step 2: compute-node filter (``hw_id >= 0``, not ``flatten_placeholder``),
  sorted stage ids, dp-major rank arithmetic;
* Step 3: axis-layout recovery (``_extract_axis_layout`` semantics) and gmap
  traffic collection when ``root._optimize_2dmap`` is present;
* Step 4: per-edge stage attribution (``local_hw_id`` -> parent ``hw_id`` ->
  child ``hw_id``), with the legacy error paths verbatim;
* Step 5: BOTH dependency walkers (``analyze_for_compute`` /
  ``analyze_for_collective``) with the ``(id(obj), via_collective)`` visited
  discipline and ``pipeline_edge_map`` recovery quirks (compute targets
  ``setdefault(..., None)``; collective targets assign the *parent object*);
* Step 6/7: collective label assignment ((base_name, primary member set) ->
  label; gid interning deferred to the emitter), stage-task assembly
  including the collective-only stage discovery with pre-extension
  ``num_stages`` rank arithmetic (executor.py:1251-1262), and the per-stage
  Kahn toposort keyed ``(op_id, tie-counter)`` via heapq;
* SCOTCH stage remap (``_remap_stages_for_mapping`` ordering) when gmap
  produced a mapping, with labels re-derived from the remapped ranks exactly
  as legacy does.

Program uids are assigned so that uid order equals the legacy EMISSION
order: concatenation of per-stage toposorts in (final, possibly remapped)
``stage_ids`` order for main ops, then TransferOps in legacy Step-11
creation order (executor.py:1697-1750 — stage order = ``stage_order`` dict
insertion order, i.e. the PRE-remap stage order; per-stage task order;
parents/local edges iterated sorted by ``op_id``; dp handled at emission;
send/recv dedup caches keyed ``(parent, dst_stage, edge_obj)``). Same-stage
PIPELINE edges additionally become same-device zero-byte TransferOps (after
all cross-stage transfers; the emitter elides them — DESIGN §2.2).
"""

from __future__ import annotations

import heapq
import itertools
import os
import tempfile
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from timing_model import CollectiveType

from astrasim_lib import gmap
from program.ir import (
    CollectiveOp,
    CommGroup,
    ComputeOp,
    GroupKey,
    Program,
    ProgramMeta,
    TransferOp,
)
from program.layout import RankLayout
from program.validate import validate_program


class LoweringError(RuntimeError):
    """Lowering hit a graph shape the legacy converter would crash on."""


# ---------------------------------------------------------------------------
# Verbatim helper ports (executor.py)
# ---------------------------------------------------------------------------


def _extract_axis_layout(
    rank_layout: Optional[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
    """Port of ``executor._extract_axis_layout`` (executor.py:120-147)."""
    if not isinstance(rank_layout, dict):
        return [], {}, {}
    axis_order = list(rank_layout.get("axis_order", []))
    raw_sizes = rank_layout.get("axis_sizes", {})
    if isinstance(axis_order, tuple):
        axis_order = list(axis_order)
    axis_sizes: Dict[str, int] = {}
    if isinstance(raw_sizes, dict):
        for key, value in raw_sizes.items():
            try:
                axis_sizes[str(key)] = max(1, int(value))
            except Exception:
                axis_sizes[str(key)] = 1
    raw_strides = rank_layout.get("axis_strides", {})
    axis_strides: Dict[str, int] = {}
    if isinstance(raw_strides, dict):
        for key, value in raw_strides.items():
            try:
                axis_strides[str(key)] = int(value)
            except Exception:
                axis_strides[str(key)] = 0
    else:
        span = 1
        for axis in axis_order:
            axis_strides[axis] = span
            span *= axis_sizes.get(axis, 1)
    return axis_order, axis_sizes, axis_strides


def _compute_stage_axis_coords(
    stage_ids: Sequence[int],
    axis_order: Sequence[str],
    axis_sizes: Mapping[str, int],
) -> Dict[int, Dict[str, int]]:
    """Port of ``executor._compute_stage_axis_coords`` (executor.py:215-234)."""
    coords: Dict[int, Dict[str, int]] = {}
    if not axis_order:
        return coords
    for stage in stage_ids:
        remaining = int(stage)
        stage_coords: Dict[str, int] = {}
        for axis in axis_order:
            size = max(1, int(axis_sizes.get(axis, 1)))
            if size > 1:
                stage_coords[axis] = remaining % size
                remaining //= size
            else:
                stage_coords[axis] = 0
        coords[stage] = stage_coords
    return coords


def _build_axis_groups(
    axis_order: Sequence[str],
    axis_sizes: Mapping[str, int],
    stage_axis_coords: Mapping[int, Dict[str, int]],
    stage_to_ranks: Mapping[int, List[int]],
) -> Dict[str, Dict[Tuple[int, Tuple[Tuple[str, int], ...]], List[int]]]:
    """Port of ``executor._build_axis_groups`` (executor.py:237-263)."""
    axis_groups: Dict[str, Dict[Tuple[int, Tuple[Tuple[str, int], ...]], List[int]]] = {}
    if not axis_order:
        return axis_groups
    for axis in axis_order:
        size = max(1, int(axis_sizes.get(axis, 1)))
        if size <= 1:
            continue
        groups: Dict[Tuple[int, Tuple[Tuple[str, int], ...]], List[int]] = defaultdict(list)
        for stage, coords in stage_axis_coords.items():
            key_base = tuple((ax, coords.get(ax, 0)) for ax in axis_order if ax != axis)
            ranks = stage_to_ranks.get(stage, [])
            for dp_idx, rank in enumerate(ranks):
                groups[(dp_idx, key_base)].append(rank)
        axis_groups[axis] = {
            key: sorted(set(values)) for key, values in groups.items() if values
        }
    return axis_groups


def _check_collective_type(comm_type: Any) -> CollectiveType:
    """Validation half of ``executor.get_collective_type`` (executor.py:439-453);
    the pb-enum mapping lives in the emitter."""
    if comm_type is None:
        raise ValueError("Collective comm_type is required")
    if not isinstance(comm_type, CollectiveType):
        raise TypeError(f"comm_type must be CollectiveType (got {type(comm_type).__name__})")
    if comm_type == CollectiveType.PIPELINE:
        raise ValueError("Pipeline comm_type should not be mapped to a collective enum")
    return comm_type


def _assign_collective_labels_with_members(
    tp_collective_groups: Mapping[str, List[Any]],
    collective_info: Mapping[Any, Dict[str, Any]],
    axis_order: Sequence[str],
    axis_sizes: Mapping[str, int],
    stage_axis_coords: Mapping[int, Dict[str, int]],
    axis_groups: Mapping[str, Dict[Tuple[int, Tuple[Tuple[str, int], ...]], List[int]]],
    stage_to_ranks: Mapping[int, List[int]],
    dp_count: int,
) -> Tuple[Dict[Any, str], Dict[Any, Tuple[str, Tuple[int, ...]]]]:
    """Port of ``executor._assign_collective_labels`` (executor.py:266-347),
    extended to also return each edge's ``(axis, dp0 member ranks)`` so the
    lowering can register the GroupKey the emitter re-derives tokens from.

    The label key is ``(base_name, primary member set)``; the ``name_N``
    suffix counter increments in group-encounter (insertion) x op_id order.
    """

    tp_collective_labels: Dict[Any, str] = {}
    edge_primary: Dict[Any, Tuple[str, Tuple[int, ...]]] = {}
    group_key_to_label: Dict[Tuple[str, Tuple[int, ...]], str] = {}
    label_suffix_counter: Dict[str, int] = defaultdict(int)

    def _composite_members(stage: int, dp_idx: int, axes: Sequence[str]) -> List[int]:
        coords = stage_axis_coords.get(stage, {})
        key_base = tuple((ax, coords.get(ax, 0)) for ax in axis_order if ax not in axes)
        members: List[int] = []
        for stage_id, stage_coords in stage_axis_coords.items():
            stage_key = tuple((ax, stage_coords.get(ax, 0)) for ax in axis_order if ax not in axes)
            if stage_key != key_base:
                continue
            ranks = stage_to_ranks.get(stage_id, [])
            if dp_idx < len(ranks):
                members.append(ranks[dp_idx])
        return sorted(set(members))

    for base_name, edges in tp_collective_groups.items():
        if not edges:
            continue
        for edge in sorted(edges, key=lambda e: getattr(e, "op_id", 0)):
            info = collective_info[edge]
            axis = str(info.get("interconnect_type", "tp")).lower()
            stage = info["stage"]
            coords = stage_axis_coords.get(stage, {})
            key_base = tuple((ax, coords.get(ax, 0)) for ax in axis_order if ax != axis)
            participants = int(info.get("participants", 0) or 0)

            composite_axes: Optional[Tuple[str, ...]] = None
            if axis == "ep":
                tp_size = max(1, int(axis_sizes.get("tp", 1)))
                ep_size = max(1, int(axis_sizes.get("ep", 1)))
                if tp_size > 1 and participants == tp_size * ep_size:
                    composite_axes = ("tp", "ep")

            members_per_dp: List[Tuple[int, Tuple[int, ...]]] = []
            for dp_idx in range(dp_count):
                members = None
                if composite_axes is not None:
                    members = _composite_members(stage, dp_idx, composite_axes)
                else:
                    axis_map = axis_groups.get(axis)
                    if axis_map is not None:
                        members = axis_map.get((dp_idx, key_base))
                if not members:
                    ranks = stage_to_ranks.get(stage, [])
                    if dp_idx < len(ranks):
                        members = [ranks[dp_idx]]
                if members:
                    members_tuple = tuple(sorted(members))
                    members_per_dp.append((dp_idx, members_tuple))

            if not members_per_dp:
                continue

            primary_members = members_per_dp[0][1]
            label_key = (base_name, primary_members)
            label = group_key_to_label.get(label_key)
            if label is None:
                suffix = label_suffix_counter[base_name]
                label = base_name if suffix == 0 else f"{base_name}_{suffix}"
                label_suffix_counter[base_name] += 1
                group_key_to_label[label_key] = label

            tp_collective_labels[edge] = label
            edge_primary[edge] = (axis, primary_members)

    return tp_collective_labels, edge_primary


# ---------------------------------------------------------------------------
# Lowering
# ---------------------------------------------------------------------------


def lower_to_program(
    graph_root: Any,
    dp_size: int,
    layout_descriptor: Optional[Dict[str, Any]] = None,
    *,
    gmap_workdir: Optional[str] = None,
    optimize_2dmap: Optional[Dict[str, Any]] = None,
) -> Program:
    """Lower an unchanged legacy graph to a :class:`Program`.

    ``layout_descriptor`` is the legacy ``root._astrasim_rank_layout`` dict
    (falls back to reading the attribute when omitted). ``gmap_workdir``
    receives the SCOTCH artifacts (``first_dim_comm.*``) when the root
    carries ``_optimize_2dmap``; a temp dir is used when not given.
    ``optimize_2dmap`` overrides the root-attribute read (M6: the coarse
    schedule-event roots are slotted, so the hierarchical path passes the
    config explicitly instead of via a ``_optimize_2dmap`` attribute).
    """

    # --- Step 1: snapshot every reachable object in DFS preorder ---------
    all_objects = _collect_objects(graph_root)

    compute_nodes = [
        obj
        for obj in all_objects
        if getattr(obj, "hw_id", None) is not None
        and obj.hw_id >= 0
        and not getattr(obj, "flatten_placeholder", False)
    ]
    if not compute_nodes:
        raise ValueError("RAPID-LLM graph did not expose any executable compute nodes (hw_id >= 0).")

    # --- Step 2: stages + dp-major rank arithmetic -----------------------
    stage_ids: List[int] = sorted({node.hw_id for node in compute_nodes})
    dp_count = max(int(dp_size) if dp_size else 1, 1)
    stage_index: Dict[int, int] = {stage: idx for idx, stage in enumerate(stage_ids)}
    stage_to_ranks: Dict[int, List[int]] = {stage: [] for stage in stage_ids}
    num_stages = len(stage_ids)  # PRE-extension count (Step 7 quirk / DESIGN §2.5)
    for dp_idx in range(dp_count):
        for stage in stage_ids:
            stage_to_ranks[stage].append(dp_idx * num_stages + stage_index[stage])
    initial_stages = set(stage_ids)

    # --- Step 3: axis layout + gmap collection begin ---------------------
    if layout_descriptor is None:
        layout_descriptor = getattr(graph_root, "_astrasim_rank_layout", None)
    axis_order, axis_sizes, axis_strides = _extract_axis_layout(layout_descriptor)
    stage_axis_coords = _compute_stage_axis_coords(stage_ids, axis_order, axis_sizes)
    axis_groups = _build_axis_groups(axis_order, axis_sizes, stage_axis_coords, stage_to_ranks)
    optimize_cfg = optimize_2dmap
    if optimize_cfg is None:
        optimize_cfg = getattr(graph_root, "_optimize_2dmap", None) if hasattr(graph_root, "_optimize_2dmap") else None
    gmap_tmpdir: Optional[str] = None
    if optimize_cfg and gmap_workdir is None:
        gmap_tmpdir = tempfile.mkdtemp(prefix="rapid_lowering_gmap_")
        gmap_workdir = gmap_tmpdir
    if gmap_workdir is not None:
        os.makedirs(gmap_workdir, exist_ok=True)
    collector = gmap.begin_collection(optimize_cfg, axis_sizes, stage_axis_coords, gmap_workdir or ".")

    # --- Step 4: attribute each non-pipeline comm edge to a stage --------
    edge_stage: Dict[Any, int] = {}
    for obj in all_objects:
        comm = getattr(obj, "comm_type", None)
        if comm and comm != CollectiveType.PIPELINE:
            stage = None
            local_hw = getattr(obj, "local_hw_id", None)
            if local_hw is not None:
                stage = local_hw
            else:
                for candidate in getattr(obj, "parents", []):
                    if getattr(candidate, "hw_id", None) is not None and candidate.hw_id >= 0:
                        stage = candidate.hw_id
                        break
                if stage is None:
                    for candidate in getattr(obj, "children", []):
                        if getattr(candidate, "hw_id", None) is not None and candidate.hw_id >= 0:
                            stage = candidate.hw_id
                            break

            children = getattr(obj, "children", [])
            parents = getattr(obj, "parents", [])
            if len(children) == 0 and len(parents) == 0:
                raise ValueError(f"Object {obj.name} has no children or parents")

            if stage is None:
                for child in children:
                    print(f"Child: {child.name}")
                for parent in parents:
                    print(f"Parent: {parent.name}")
                raise ValueError(f"Stage not found for object {obj.name}")
            edge_stage[obj] = stage

    pipeline_edge_map: Dict[Tuple[Any, Any], Any] = {}

    def _resolve_stage(obj: Any) -> Optional[int]:
        hw = getattr(obj, "hw_id", None)
        if hw is None:
            hw = getattr(obj, "local_hw_id", None)
        if hw is None:
            return None
        return int(hw)

    def _pipeline_edge_remote_child_stages(edge_obj: Any, *, local_stage: int) -> List[int]:
        stages: List[int] = []
        for child in getattr(edge_obj, "children", []):
            child_stage = _resolve_stage(child)
            if child_stage is None or child_stage == local_stage:
                continue
            if child_stage not in stages:
                stages.append(child_stage)
        return stages

    def _pipeline_edge_has_remote_child(edge_obj: Any, *, local_stage: int) -> bool:
        return bool(_pipeline_edge_remote_child_stages(edge_obj, local_stage=local_stage))

    # --- Step 5a/5b: the two dependency walkers ---------------------------
    # ``same_device_edges``: (pipeline_edge, same_stage_src) pairs recorded
    # alongside the stage_deps/collective_deps adds — the legacy converter
    # collapses these into plain deps; we additionally lower them to
    # same-device zero-byte TransferOps for the (M5) analytical evaluator.

    def analyze_for_compute(node: Any) -> Tuple[Set[Any], Set[Any], Set[Any], Set[Any], Set[Tuple[Any, Any]]]:
        stage = node.hw_id
        stage_deps: Set[Any] = set()
        pipeline_deps: Set[Any] = set()
        collective_deps: Set[Any] = set()
        local_pipeline_deps: Set[Any] = set()
        same_device_edges: Set[Tuple[Any, Any]] = set()
        visited: Set[Tuple[int, bool]] = set()
        stack: List[Tuple[Any, bool]] = [(parent, False) for parent in getattr(node, "parents", [])]

        while stack:
            cur, via_collective = stack.pop()
            key = (id(cur), via_collective)
            if key in visited:
                continue
            visited.add(key)

            hw = getattr(cur, "hw_id", None)
            if hw is not None and hw >= 0:
                if hw == stage:
                    if not via_collective:
                        stage_deps.add(cur)
                else:
                    pipeline_deps.add(cur)
                    pipeline_edge_map.setdefault((cur, node), None)
                continue

            comm = getattr(cur, "comm_type", None)
            if comm:
                if comm == CollectiveType.PIPELINE:
                    srcs = [p for p in cur.parents if getattr(p, "hw_id", None) is not None and p.hw_id >= 0]
                    coll_srcs = [p for p in cur.parents if getattr(p, "local_hw_id", None) is not None and p.local_hw_id >= 0]
                    if srcs:
                        for src in srcs:
                            if src.hw_id == stage:
                                if not via_collective:
                                    if _pipeline_edge_has_remote_child(cur, local_stage=stage):
                                        local_pipeline_deps.add(cur)
                                    stage_deps.add(src)
                                    same_device_edges.add((cur, src))
                            else:
                                pipeline_deps.add(src)
                                pipeline_edge_map[(src, node)] = cur
                    if coll_srcs:
                        for src in coll_srcs:
                            if src.local_hw_id == stage:
                                if not via_collective:
                                    if _pipeline_edge_has_remote_child(cur, local_stage=stage):
                                        local_pipeline_deps.add(cur)
                                    collective_deps.add(src)
                                    same_device_edges.add((cur, src))
                            # else: cross-pipeline dp collective deps unsupported; ignored
                    continue
                else:
                    collective_deps.add(cur)
                    for parent in getattr(cur, "parents", []):
                        stack.append((parent, True))
                    continue

            for parent in getattr(cur, "parents", []):
                stack.append((parent, via_collective))

        stage_deps.discard(node)
        return stage_deps, pipeline_deps, collective_deps, local_pipeline_deps, same_device_edges

    def analyze_for_collective(edge: Any) -> Tuple[Set[Any], Set[Any], Set[Any], Set[Any], Set[Tuple[Any, Any]]]:
        stage = edge_stage[edge]
        stage_deps: Set[Any] = set()
        pipeline_deps: Set[Any] = set()
        collective_deps: Set[Any] = set()
        local_pipeline_deps: Set[Any] = set()
        same_device_edges: Set[Tuple[Any, Any]] = set()
        visited: Set[Tuple[int, bool]] = set()
        stack: List[Tuple[Any, bool]] = [(parent, False) for parent in getattr(edge, "parents", [])]

        while stack:
            cur, via_collective = stack.pop()
            key = (id(cur), via_collective)
            if key in visited:
                continue
            visited.add(key)

            hw = getattr(cur, "hw_id", None)
            if hw is None:
                hw = getattr(cur, "local_hw_id", None)
            if hw is not None and hw >= 0:
                if hw == stage:
                    if not via_collective:
                        stage_deps.add(cur)
                else:
                    pipeline_deps.add(cur)
                    # Legacy quirk: the *parent object itself* is recorded as
                    # the pipeline "edge" for collective targets.
                    pipeline_edge_map[(cur, edge)] = cur
                continue

            comm = getattr(cur, "comm_type", None)
            if comm == CollectiveType.PIPELINE:
                srcs = [p for p in cur.parents if getattr(p, "hw_id", None) is not None and p.hw_id >= 0]
                coll_srcs = [p for p in cur.parents if getattr(p, "local_hw_id", None) is not None and getattr(p, "local_hw_id") >= 0]
                if srcs:
                    for src in srcs:
                        if src.hw_id == stage:
                            if not via_collective:
                                if _pipeline_edge_has_remote_child(cur, local_stage=stage):
                                    local_pipeline_deps.add(cur)
                                stage_deps.add(src)
                                same_device_edges.add((cur, src))
                        else:
                            pipeline_deps.add(src)
                            pipeline_edge_map[(src, edge)] = cur
                if coll_srcs:
                    for src in coll_srcs:
                        if src.local_hw_id == stage:
                            if not via_collective:
                                if _pipeline_edge_has_remote_child(cur, local_stage=stage):
                                    local_pipeline_deps.add(cur)
                                collective_deps.add(src)
                                same_device_edges.add((cur, src))
                continue
            elif comm:
                collective_deps.add(cur)
                for parent in getattr(cur, "parents", []):
                    stack.append((parent, True))
                continue

            for parent in getattr(cur, "parents", []):
                stack.append((parent, via_collective))

        return stage_deps, pipeline_deps, collective_deps, local_pipeline_deps, same_device_edges

    compute_info: Dict[Any, Dict[str, Any]] = {}
    for node in compute_nodes:
        stage_deps, pipeline_deps, collective_deps, local_pipeline_deps, same_dev = analyze_for_compute(node)
        compute_info[node] = {
            "stage": node.hw_id,
            "stage_deps": stage_deps,
            "pipeline_deps": pipeline_deps,
            "collective_deps": collective_deps,
            "local_pipeline_deps": local_pipeline_deps,
            "same_device_edges": same_dev,
            "name": node.name,
        }

    collective_info: Dict[Any, Dict[str, Any]] = {}
    for edge, stage in edge_stage.items():
        stage_deps, pipeline_deps, collective_deps, local_pipeline_deps, same_dev = analyze_for_collective(edge)
        entry = {
            "stage": stage,
            "stage_deps": stage_deps,
            "pipeline_deps": pipeline_deps,
            "collective_deps": collective_deps,
            "local_pipeline_deps": local_pipeline_deps,
            "same_device_edges": same_dev,
            "size": int(getattr(edge, "comm_size_bytes", 0)),
            "comm_type": _check_collective_type(edge.comm_type),
            "name": edge.name,
            "interconnect_type": getattr(edge, "comm_interconnect_type", None),
            "participants": int(getattr(edge, "participants", 0) or 0),
        }
        collective_info[edge] = entry
        if collector:
            axis_name = entry["interconnect_type"]
            if axis_name:
                axis_norm = str(axis_name).strip().lower()
                if axis_norm in collector.subset_axes:
                    should_double = entry["comm_type"] in (
                        CollectiveType.ALL_REDUCE,
                        CollectiveType.ALL_TO_ALL,
                    )
                    collector.record_collective(
                        axis=axis_norm,
                        stage_id=stage,
                        size_bytes=entry["size"],
                        participant_count=entry["participants"],
                        double_weight=should_double,
                    )

    if collector:
        _record_pipeline_edges_for_first_dim(collector, all_objects)

    # --- Step 6: group collectives by name for label assignment ----------
    tp_collective_groups: Dict[str, List[Any]] = defaultdict(list)
    for edge, info in collective_info.items():
        interconnect_type = info.get("interconnect_type")
        if interconnect_type and interconnect_type not in {"dp", "pp", "pipeline"}:
            tp_collective_groups[info["name"]].append(edge)

    # --- Step 7: per-stage task assembly + toposort ----------------------
    stage_tasks: Dict[int, Set[Any]] = {stage: set() for stage in stage_ids}
    for node in compute_nodes:
        stage_tasks[node.hw_id].add(node)
    for edge, info in collective_info.items():
        stage = info["stage"]
        if stage not in stage_tasks:
            stage_tasks[stage] = set()
            # Collective-only stage discovery: ranks use the PRE-extension
            # num_stages (executor.py:1251-1262 — reproduced exactly).
            stage_idx = stage_index.setdefault(stage, len(stage_index))
            stage_to_ranks[stage] = [dp_idx * num_stages + stage_idx for dp_idx in range(dp_count)]
            stage_ids.append(stage)
        stage_tasks[stage].add(edge)

    stage_adj: Dict[int, Dict[Any, Set[Any]]] = {stage: defaultdict(set) for stage in stage_tasks}
    stage_indegree: Dict[int, Dict[Any, int]] = {stage: defaultdict(int) for stage in stage_tasks}

    for stage, tasks in stage_tasks.items():
        for task in tasks:
            stage_indegree[stage].setdefault(task, 0)

    for node, info in compute_info.items():
        stage = info["stage"]
        for dep in info["stage_deps"]:
            if dep in stage_tasks.get(stage, set()):
                stage_adj[stage][dep].add(node)
                stage_indegree[stage][node] += 1
        for edge in info["collective_deps"]:
            stage_edge = collective_info.get(edge, {}).get("stage")
            if stage_edge == stage:
                stage_adj[stage][edge].add(node)
                stage_indegree[stage][node] += 1

    for edge, info in collective_info.items():
        stage = info["stage"]
        for dep in info["stage_deps"]:
            if dep in stage_tasks.get(stage, set()):
                stage_adj[stage][dep].add(edge)
                stage_indegree[stage][edge] += 1
        for dep in info.get("collective_deps", set()):
            dep_stage = collective_info.get(dep, {}).get("stage")
            if dep_stage == stage:
                stage_adj[stage][dep].add(edge)
                stage_indegree[stage][edge] += 1

    # Kahn keyed (op_id, tie-counter) via heapq — the 85894c6-fixed
    # discipline. Tasks/neighbors are iterated in RAW set order like legacy:
    # op_ids are NOT unique (the tp-overlap head split reuses its source's
    # op_id), and the tie-counter sequence — hence the order of equal-op_id
    # tasks — depends on set iteration order. Our walkers insert the same
    # objects in the same sequence as legacy's, so raw iteration reproduces
    # the legacy tie-break exactly in-process.
    def _stage_task_key(task: Any) -> int:
        return int(getattr(task, "op_id", 0) or 0)

    stage_order: Dict[int, List[Any]] = {}
    for stage, tasks in stage_tasks.items():
        indeg = stage_indegree[stage]
        heap: List[Tuple[int, int, Any]] = []
        seq = itertools.count()
        for task in tasks:
            if indeg.get(task, 0) == 0:
                heapq.heappush(heap, (_stage_task_key(task), next(seq), task))
        order: List[Any] = []
        while heap:
            _, _, task = heapq.heappop(heap)
            order.append(task)
            for neighbor in stage_adj[stage].get(task, set()):
                indeg[neighbor] -= 1
                if indeg[neighbor] == 0:
                    heapq.heappush(heap, (_stage_task_key(neighbor), next(seq), neighbor))
        if len(order) != len(tasks):
            raise RuntimeError(f"Cycle detected in stage {stage} while scheduling")
        stage_order[stage] = order

    step11_stage_order = list(stage_order.keys())  # PRE-remap iteration order

    # --- SCOTCH remap (ordering of _remap_stages_for_mapping) -------------
    mapping_result = gmap.finalize_collection(collector)
    if mapping_result:
        permutation = mapping_result.permutation
        if len(permutation) != collector.vertex_count:
            raise ValueError("Permutation length does not match first-dimension vertex count.")

        def _sort_key(stage: int) -> Tuple[int, ...]:
            coords = collector.stage_axis_coords.get(stage, {})
            higher = tuple(int(coords.get(axis, 0)) for axis in collector.replication_axes)
            local_idx = collector.local_index_for_stage(stage)
            target_idx = permutation[local_idx]
            return higher + (target_idx,)

        ordered_stages = sorted(stage_ids, key=_sort_key)
        new_stage_index = {stage: idx for idx, stage in enumerate(ordered_stages)}
        new_stage_to_ranks: Dict[int, List[int]] = {}
        for stage in stage_ids:
            new_idx = new_stage_index[stage]
            new_stage_to_ranks[stage] = [dp_idx * num_stages + new_idx for dp_idx in range(dp_count)]
        stage_ids = ordered_stages
        stage_index = new_stage_index
        stage_to_ranks = new_stage_to_ranks
        # Rebuild communicator groupings from the remapped ranks (legacy
        # re-runs _build_axis_groups + _assign_collective_labels here).
        axis_groups = _build_axis_groups(axis_order, axis_sizes, stage_axis_coords, stage_to_ranks)
    if gmap_tmpdir is not None:
        import shutil

        shutil.rmtree(gmap_tmpdir, ignore_errors=True)

    # --- Labels + GroupKeys (final, post-remap ranks) ---------------------
    tp_collective_labels, edge_primary = _assign_collective_labels_with_members(
        tp_collective_groups,
        collective_info,
        axis_order,
        axis_sizes,
        stage_axis_coords,
        axis_groups,
        stage_to_ranks,
        dp_count,
    )

    rank_to_stage_dp0: Dict[int, int] = {}
    for stage, ranks in stage_to_ranks.items():
        if ranks:
            rank_to_stage_dp0[ranks[0]] = stage

    # --- uid assignment: main ops in Step-10 emission order ---------------
    ops: List[Any] = []
    task_uid: Dict[Any, int] = {}
    for stage in stage_ids:
        order = stage_order.get(stage)
        if not order:
            continue
        for task in order:
            task_uid[task] = len(ops)
            ops.append(task)  # placeholder; replaced below

    groups: Dict[GroupKey, CommGroup] = {}

    def _group_key_for(edge: Any) -> GroupKey:
        axis, primary_members = edge_primary[edge]
        try:
            member_stages = tuple(sorted(rank_to_stage_dp0[r] for r in primary_members))
        except KeyError as exc:
            raise LoweringError(
                f"Collective '{collective_info[edge]['name']}' group member rank {exc} "
                "does not map back to a stage at dp index 0"
            ) from exc
        return GroupKey(axis=axis, members=member_stages)

    def _dep_uid(dep: Any, owner_info: Dict[str, Any]) -> int:
        uid = task_uid.get(dep)
        if uid is None:
            raise LoweringError(
                f"Dependency {dep!r} of task '{owner_info['name']}' is not a lowered "
                "task (the legacy converter would KeyError resolving its ET id here)"
            )
        return uid

    for task, uid in task_uid.items():
        if task in collective_info:
            info = collective_info[task]
            dep_uids: List[int] = []
            for dep in sorted(info["stage_deps"], key=lambda d: task_uid.get(d, len(ops))):
                dep_uids.append(_dep_uid(dep, info))
            for dep in sorted(info.get("collective_deps", set()), key=lambda d: task_uid.get(d, len(ops))):
                dep_uids.append(_dep_uid(dep, info))
            unique_deps = tuple(dict.fromkeys(sorted(dep_uids)))
            label = tp_collective_labels.get(task)
            group_key: Optional[GroupKey] = None
            if label is not None:
                group_key = _group_key_for(task)
                if group_key not in groups:
                    groups[group_key] = CommGroup(key=group_key, label=label)
            ops[uid] = CollectiveOp(
                uid=uid,
                name=info["name"],
                device=int(info["stage"]),
                coll=info["comm_type"],
                size_bytes=int(info["size"]),
                participants=int(info["participants"]),
                interconnect=info.get("interconnect_type"),
                is_dp=label is None,
                label=label,
                group=group_key,
                deps=unique_deps,
                legacy_op_id=getattr(task, "op_id", None),
            )
        else:
            info = compute_info[task]
            stage = info["stage"]
            dep_uids = []
            for dep in sorted(info["stage_deps"], key=lambda d: task_uid.get(d, len(ops))):
                dep_uids.append(_dep_uid(dep, info))
            for edge in sorted(info["collective_deps"], key=lambda d: task_uid.get(d, len(ops))):
                stage_edge = collective_info.get(edge, {}).get("stage")
                if stage_edge is not None and stage_edge == stage:
                    dep_uids.append(_dep_uid(edge, info))
            unique_deps = tuple(dict.fromkeys(sorted(dep_uids)))
            profile = getattr(task, "duration_profile", None)
            if profile:
                if len(profile) != dp_count:
                    raise ValueError(
                        f"Duration profile for node '{getattr(task, 'name', '<unnamed>')}' "
                        f"has length {len(profile)} but dp_count={dp_count}."
                    )
                duration = tuple(float(v) for v in profile)
            else:
                duration = (float(getattr(task, "duration", 0.0) or 0.0),)
            ops[uid] = ComputeOp(
                uid=uid,
                name=info["name"],
                device=int(stage),
                duration=duration,
                deps=unique_deps,
                legacy_op_id=getattr(task, "op_id", None),
            )

    # --- TransferOps: legacy Step-11 creation order (executor.py:1697-1750)
    _replay_step11(
        ops=ops,
        task_uid=task_uid,
        stage_order=stage_order,
        step11_stage_order=step11_stage_order,
        compute_info=compute_info,
        collective_info=collective_info,
        pipeline_edge_map=pipeline_edge_map,
        remote_child_stages=_pipeline_edge_remote_child_stages,
    )

    # --- Program assembly -------------------------------------------------
    layout = RankLayout(
        axis_order=tuple(axis_order),
        axis_sizes=dict(axis_sizes),
        axis_strides=dict(axis_strides),
    )
    meta = ProgramMeta(
        label="legacy_lowering",
        optimize_2dmap=dict(optimize_cfg) if isinstance(optimize_cfg, dict) else optimize_cfg,
        misc={
            "num_stages_initial": num_stages,
            "compute_devices": tuple(s for s in stage_ids if s in initial_stages),
        },
    )
    program = Program(
        layout=layout,
        dp_count=dp_count,
        devices=tuple(int(s) for s in stage_ids),
        ops=ops,
        groups=groups,
        meta=meta,
    )
    validate_program(program, check_races=False)
    return program


def _collect_objects(root: Any) -> List[Any]:
    """Step 1: iterative preorder DFS identical to the legacy recursion
    (visited-at-entry; children explored depth-first in list order)."""
    visited: Set[int] = set()
    ordered: List[Any] = []
    roots = list(root) if isinstance(root, (list, tuple)) else [root]
    stack: List[Any] = list(reversed(roots))
    while stack:
        obj = stack.pop()
        if id(obj) in visited:
            continue
        visited.add(id(obj))
        ordered.append(obj)
        children = getattr(obj, "children", [])
        for child in reversed(children):
            stack.append(child)
    return ordered


def _record_pipeline_edges_for_first_dim(collector_obj: "gmap.GMapCollector", all_objects: Sequence[Any]) -> None:
    """Port of the closure in executor.py:893-920."""
    if not collector_obj.include_pipeline:
        return
    for obj in all_objects:
        if getattr(obj, "comm_type", None) != CollectiveType.PIPELINE:
            continue
        parents = getattr(obj, "parents", [])
        src_stage = None
        for parent in parents:
            if hasattr(parent, "hw_id") and parent.hw_id is not None and int(parent.hw_id) >= 0:
                src_stage = int(parent.hw_id)
                break
        if src_stage is None:
            raise ValueError("Unable to resolve source stage for pipeline edge during gmap collection.")
        children = getattr(obj, "children", [])
        dst_stage = None
        for child in children:
            if hasattr(child, "hw_id") and child.hw_id is not None and int(child.hw_id) >= 0:
                dst_stage = int(child.hw_id)
                break
        if dst_stage is None:
            raise ValueError("Unable to resolve destination stage for pipeline edge during gmap collection.")
        if src_stage == dst_stage:
            continue
        if not hasattr(obj, "comm_size_bytes"):
            raise ValueError("Pipeline edge is missing comm_size_bytes for gmap collection.")
        size_bytes = int(getattr(obj, "comm_size_bytes"))
        collector_obj.record_pipeline(src_stage=src_stage, dst_stage=dst_stage, size_bytes=size_bytes)


def _replay_step11(
    *,
    ops: List[Any],
    task_uid: Dict[Any, int],
    stage_order: Dict[int, List[Any]],
    step11_stage_order: List[int],
    compute_info: Dict[Any, Dict[str, Any]],
    collective_info: Dict[Any, Dict[str, Any]],
    pipeline_edge_map: Dict[Tuple[Any, Any], Any],
    remote_child_stages: Any,
) -> None:
    """Create TransferOps in legacy Step-11 order (executor.py:1697-1750).

    Iteration: ``stage_order`` in dict-insertion order (the PRE-remap stage
    order — Step 11 does NOT honor the SCOTCH remap ordering, unlike Step
    10), per-stage task order, parents sorted by ``op_id``, then local
    pipeline edges sorted by ``op_id``. The dp loop is dropped: legacy
    creates one send/recv pair per dp on disjoint ranks with identical
    per-dp ordering, so a single TransferOp stamped per dp at emission
    reproduces every rank's node order. Dedup caches are keyed
    ``(parent, dst_stage, edge_obj)`` — the legacy caches minus dp.

    ``send_seq``/``recv_seq`` capture the global node-creation positions:
    ``_append_pipeline_send`` creates the SEND at a transfer's first
    reference of either kind; ``ensure_pipeline`` creates the RECV at its
    first receive-side reference — which can be *later* than the sends of
    other transfers (a local send materializes before its remote consumer
    requests the recv), so recv positions are recorded independently.
    """

    def _op_id_key(obj: Any) -> int:
        return int(getattr(obj, "op_id", 0) or 0)

    send_map: Dict[Tuple[Any, int, Any], TransferOp] = {}
    recv_done: Set[Tuple[Any, int, Any]] = set()
    event_seq = itertools.count()

    def _task_stage(target: Any) -> int:
        return collective_info[target]["stage"] if target in collective_info else compute_info[target]["stage"]

    def _parent_stage(parent: Any) -> int:
        parent_stage = getattr(parent, "hw_id", None)
        if parent_stage is None:
            parent_stage = getattr(parent, "local_hw_id", None)
        if parent_stage is None:
            raise LoweringError(f"Stage not found for parent {parent}")
        return int(parent_stage)

    def _append_send(parent: Any, edge_obj: Any, *, dst_stage: int) -> TransferOp:
        key = (parent, int(dst_stage), edge_obj)
        cached = send_map.get(key)
        if cached is not None:
            return cached
        src_stage = _parent_stage(parent)
        producer_uid = task_uid.get(parent)
        if producer_uid is None:
            raise LoweringError(f"Parent {parent} not found in collective_info or compute_info")
        size = int(getattr(edge_obj, "comm_size_bytes", 0))
        comm_type = getattr(edge_obj, "comm_type", None)
        legacy_tag = getattr(edge_obj, "op_id", None) if edge_obj is not None else None
        transfer = TransferOp(
            uid=len(ops),
            name=getattr(edge_obj, "name", "pipeline") if edge_obj is not None else "pipeline",
            src_device=src_stage,
            dst_device=int(dst_stage),
            size_bytes=size,
            comm_type=comm_type,
            producer=producer_uid,
            deps=(producer_uid,),
            consumers=(),
            send_seq=next(event_seq),
            recv_seq=None,
            legacy_tag=int(legacy_tag) if legacy_tag is not None else None,
        )
        ops.append(transfer)
        send_map[key] = transfer
        return transfer

    def _add_consumer(transfer: TransferOp, consumer_uid: int) -> None:
        if consumer_uid not in transfer.consumers:
            transfer.consumers = transfer.consumers + (consumer_uid,)

    def _add_post_dep(consumer_uid: int, dep_uid: int) -> None:
        op = ops[consumer_uid]
        if dep_uid not in op.post_deps:
            op.post_deps = op.post_deps + (dep_uid,)

    def _ensure_pipeline(parent: Any, target: Any) -> None:
        parent_stage = _parent_stage(parent)
        target_stage = _task_stage(target)
        target_uid = task_uid[target]
        if parent_stage == target_stage:
            # Legacy dead-in-practice branch: ensure_pipeline returns the
            # parent's own ET id and Step 11 appends it as a plain ctrl dep.
            _add_post_dep(target_uid, task_uid[parent])
            return
        edge_obj = pipeline_edge_map.get((parent, target))
        key = (parent, int(target_stage), edge_obj)
        if key in recv_done:
            _add_consumer(send_map[key], target_uid)
            return
        transfer = _append_send(parent, edge_obj, dst_stage=target_stage)
        transfer.recv_seq = next(event_seq)
        recv_done.add(key)
        _add_consumer(transfer, target_uid)

    def _pipeline_parent_for_same_stage_edge(edge_obj: Any, *, local_stage: int) -> Any:
        for parent in getattr(edge_obj, "parents", []):
            hw = getattr(parent, "hw_id", None)
            if hw is None:
                hw = getattr(parent, "local_hw_id", None)
            if hw is not None and int(hw) == local_stage:
                return parent
        raise LoweringError(
            f"Unable to resolve same-stage pipeline source for edge '{getattr(edge_obj, 'name', edge_obj)}' "
            f"and stage {local_stage}."
        )

    def _ensure_local_pipeline_sends(edge_obj: Any, target: Any) -> None:
        target_stage = _task_stage(target)
        remote_stages = remote_child_stages(edge_obj, local_stage=target_stage)
        if not remote_stages:
            return
        parent = _pipeline_parent_for_same_stage_edge(edge_obj, local_stage=target_stage)
        target_uid = task_uid[target]
        for remote_stage in remote_stages:
            transfer = _append_send(parent, edge_obj, dst_stage=remote_stage)
            _add_consumer(transfer, target_uid)

    for stage in step11_stage_order:
        for task in stage_order.get(stage, []):
            info = collective_info[task] if task in collective_info else compute_info[task]
            if not info["pipeline_deps"] and not info.get("local_pipeline_deps"):
                continue
            # Primary key = op_id (legacy). op_ids repeat (the tp-overlap
            # head split reuses its source's op_id), where legacy's stable
            # sort fell back to SET ITERATION ORDER — i.e. object addresses,
            # not reproducible across graph instances (M3a differential
            # finding). The tie is broken here by the task's emission
            # position (task_uid) instead: deterministic, identical for the
            # legacy-flatten and fine-builder paths, and invisible to every
            # golden gate (the tied transfers are same-payload control
            # sends; canonical forms and the manifest sort ignore the swap,
            # and wall seconds never depended on the legacy coin flip —
            # the gates pass today across processes with varying id order).
            for parent in sorted(
                info["pipeline_deps"],
                key=lambda p: (_op_id_key(p), task_uid.get(p, len(ops))),
            ):
                if parent not in collective_info and parent not in compute_info:
                    continue  # orphaned parent, ignored (legacy TODO comment)
                _ensure_pipeline(parent, task)
            for edge_obj in sorted(info.get("local_pipeline_deps", set()), key=_op_id_key):
                _ensure_local_pipeline_sends(edge_obj, task)

    # Same-device zero-cost transfers (DESIGN §2.2): appended after all
    # cross-stage transfers (position among transfers is a free choice: the
    # emitter elides them; the M5 evaluator enqueues them). One TransferOp
    # per (edge, same-stage source), consumers accumulated in task order.
    same_device_map: Dict[Tuple[Any, Any], TransferOp] = {}
    for stage in step11_stage_order:
        for task in stage_order.get(stage, []):
            info = collective_info[task] if task in collective_info else compute_info[task]
            for edge_obj, src in sorted(
                info.get("same_device_edges", set()),
                key=lambda pair: (_op_id_key(pair[0]), _op_id_key(pair[1]), task_uid.get(pair[1], len(ops))),
            ):
                src_uid = task_uid.get(src)
                if src_uid is None:
                    continue  # source not lowered; the plain dep never existed either
                key = (edge_obj, src)
                transfer = same_device_map.get(key)
                if transfer is None:
                    device = int(_task_stage(task))
                    transfer = TransferOp(
                        uid=len(ops),
                        name=getattr(edge_obj, "name", "pipeline"),
                        src_device=device,
                        dst_device=device,
                        size_bytes=int(getattr(edge_obj, "comm_size_bytes", 0)),
                        comm_type=getattr(edge_obj, "comm_type", None),
                        producer=src_uid,
                        deps=(src_uid,),
                        consumers=(),
                        send_seq=None,
                        recv_seq=None,
                        legacy_tag=getattr(edge_obj, "op_id", None),
                    )
                    ops.append(transfer)
                    same_device_map[key] = transfer
                _add_consumer(transfer, task_uid[task])
