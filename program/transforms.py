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

"""TP/TP-SP/CP overlap as Program -> Program passes (M3a).

Ports of the legacy graph rewrites ``_split_tp_node`` / ``_split_cp_edge``
(llm_execution.py 97-292), applied to the FINE :class:`~program.ir.Program`
AFTER construction instead of to the flattened legacy graph BEFORE lowering.

ORDERING (the op-id-reuse quirk, and how the legacy emission order is
reproduced). Legacy rewrites the graph and *then* lowers it; the per-stage
emission order is a Kahn toposort keyed by ``op_id``, and both split
products REUSE the source's op_id (`_split_tp_node` head, `_split_cp_edge`
block/ovlp). M1 discovered emission order depends on that reuse. The
resulting order is fully determined, because equal-op_id ops are always
chain-connected (head -> tail, block -> ovlp) and the Kahn heap pops the
minimal ready key:

* tp split: ``head`` pops exactly where the unsplit node would have (same
  key, same in-edges), and ``tail`` — whose only dependency is ``head`` and
  whose (reused) key is <= every other ready key — pops IMMEDIATELY after.
  Program equivalent: insert ``head`` at the node's uid slot, ``tail``
  right after. ``legacy_op_id`` is copied onto the head (the reuse).
* cp split: ``block`` pops at the edge's slot; ``ovlp`` (reused key, only
  dependency ``block``) pops immediately after, BEFORE the attention node
  (whose op_id is larger). Program equivalent: mutate the edge into
  ``block`` in place and insert ``ovlp`` right after it.

Because Program uid order is the per-stage concatenation, these in-place
insertions reproduce the legacy post-rewrite lowering order without
re-running the toposort.

LIMITATIONS DISCOVERED BY THE M3a DIFFERENTIAL (documented per the M3a
mandate) — two ways the legacy in-graph rewrite reaches state a
post-lowering pass no longer has:

1. ORDER. ``_detach_edge`` + ``_connect_edge`` REORDER children lists
   (remove + append), so a same-stage zero-byte PIPELINE edge whose
   per-rank consumer heads get tp-split ends up with its children in
   split-processing (DFS) order instead of rank order — and
   ``_ensure_local_pipeline_sends`` creates its control SENDs in exactly
   that child order. Send/recv creation sequence numbers are frozen at
   lowering; the pass cannot replay the DFS.
2. MEMBERSHIP. A cross-device successor that reaches an unsplit compute
   BOTH directly and through its trailing tp collective discovers ONE
   producer pre-split but TWO post-split (the head via the collective's
   parents, the tail via the direct link), i.e. the legacy rewrite CREATES
   an extra zero-byte control transfer that does not exist in the
   pre-overlap lowering.

The FINE builder therefore applies :func:`apply_overlap_to_fine_root` — a
verbatim port of the legacy rewrite over the builder's proto elements —
BEFORE lowering, which reproduces both by construction. The Program ->
Program passes below remain the forward-looking API: they are exact for
programs whose split computes feed no cross-device consumers (e.g. single-
stage single-microbatch programs — unit-covered by the differential test
file), and they are the basis for the post-M9 ``program`` id policy where
the legacy creation-order coincidences retire.

Dep rewires mirror the legacy walker outcomes on the rewritten graph:

* tp (0 < f < 1): head takes the node's deps, transfer-consumer slots and
  post_deps; the node (tail, duration ``f*d``) depends only on head; the tp
  collectives move their node-dep to head; every successor of a moved tp
  collective gains a dep on tail (legacy ``_connect_edge(tail, succ)``).
* tp (f >= 1): the tp collectives are hoisted before the node (deps := the
  node's deps; consumers of the node's transfers extended with them) and
  their successors gain a dep on the node.
* cp (0 < f < 1): attention keeps its dep on block; non-attention
  successors trade the edge dep for attention (+ ovlp when present).
* cp (f >= 1 or no bytes): attention trades the edge dep for the edge's
  deps; non-attention successors keep the edge dep and gain attention.

After any cp change the collective labels are re-derived (the split renames
ops to ``*_block`` / ``*_ovlp``; legacy assigned labels post-rewrite) with
the exact ``_assign_collective_labels`` rules — label assignment is a pure
function of (base name, op_id order, member sets), so re-deriving it from
the Program matches the legacy in-graph assignment.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from timing_model import CollectiveType

from program.ir import CollectiveOp, CommGroup, ComputeOp, GroupKey, Program, TransferOp
from program.validate import validate_program


def _mode_label(mode: Any) -> str:
    for attr in ("value", "name"):
        if hasattr(mode, attr):
            return str(getattr(mode, attr)).lower()
    return str(mode).lower()


# ---------------------------------------------------------------------------
# Proto-level overlap (applied by the FINE builder BEFORE lowering)
# ---------------------------------------------------------------------------
#
# Verbatim ports of llm_execution._split_tp_node / _split_cp_edge /
# _apply_tp_overlap_transforms / _apply_cp_overlap_transforms over the fine
# builder's FineNode/FineEdge elements (see the module docstring for why the
# in-graph form is load-bearing: children-list reordering drives Step-11
# control-send creation order).


def _detach_edge(parent: Any, child: Any) -> None:
    children = getattr(parent, "children", None)
    if isinstance(children, list):
        try:
            children.remove(child)
        except ValueError:
            pass
    parents = getattr(child, "parents", None)
    if isinstance(parents, list):
        try:
            parents.remove(parent)
        except ValueError:
            pass


def _connect_edge(parent: Any, child: Any) -> None:
    if child in getattr(parent, "children", []):
        return
    parent.add_child(child)


def _copy_fine_node_metadata(source: Any, target: Any) -> None:
    for attr in (
        "micro_batch_index",
        "layer_index",
        "direction",
        "stage_id",
        "tp_rank",
        "cp_rank",
        "mem_kind",
        "recompute",
        "param_gather",
    ):
        if hasattr(source, attr):
            setattr(target, attr, getattr(source, attr))


def _copy_fine_edge_metadata(source: Any, target: Any) -> None:
    for attr in (
        "local_hw_id",
        "stage_id",
        "micro_batch_index",
        "layer_index",
        "direction",
        "tp_rank",
        "cp_rank",
    ):
        if hasattr(source, attr):
            setattr(target, attr, getattr(source, attr))


def _split_tp_node_fine(node: Any, tp_children: List[Any], overlap: float, node_cls: Any) -> None:
    if overlap <= 0.0 or not tp_children:
        return

    duration = float(getattr(node, "duration", 0.0) or 0.0)
    if duration <= 0.0:
        return

    if overlap >= 1.0:
        parents = list(getattr(node, "parents", []))
        tp_succs: List[Any] = []
        for tp_edge in tp_children:
            _detach_edge(node, tp_edge)
            for parent in parents:
                _connect_edge(parent, tp_edge)
            for succ in list(getattr(tp_edge, "children", []) or []):
                tp_succs.append(succ)
        for succ in tp_succs:
            _connect_edge(node, succ)
        return

    head_duration = duration * (1.0 - overlap)
    tail_duration = duration * overlap
    if head_duration <= 0.0 or tail_duration <= 0.0:
        return

    tail = node
    tail.duration = tail_duration
    head = node_cls(
        name=f"{tail.name}_head",
        # THE op-id-reuse quirk: the head shares the source's op_id.
        op_id=getattr(tail, "op_id", 0),
        hw_id=tail.hw_id,
        duration=head_duration,
        fwd=tail.fwd,
    )
    _copy_fine_node_metadata(tail, head)

    parents = list(getattr(tail, "parents", []))
    for parent in parents:
        _detach_edge(parent, tail)
        _connect_edge(parent, head)

    _connect_edge(head, tail)

    for tp_edge in tp_children:
        _detach_edge(tail, tp_edge)
        _connect_edge(head, tp_edge)
        for succ in list(getattr(tp_edge, "children", []) or []):
            _connect_edge(tail, succ)


def _split_cp_edge_fine(edge: Any, overlap: float, node_cls: Any, edge_cls: Any) -> None:
    attention_children = [
        child for child in getattr(edge, "children", [])
        if isinstance(child, node_cls) and "attention" in str(getattr(child, "name", "")).lower()
    ]
    if not attention_children:
        return

    total_bytes = int(getattr(edge, "comm_size_bytes", 0) or 0)
    preds = list(getattr(edge, "parents", []))
    succs = list(getattr(edge, "children", []))

    if overlap >= 1.0 or total_bytes <= 0:
        for attention in attention_children:
            _detach_edge(edge, attention)
            for pred in preds:
                _connect_edge(pred, attention)
        for succ in succs:
            if succ in attention_children:
                continue
            for attention in attention_children:
                _connect_edge(attention, succ)
            _connect_edge(edge, succ)
        return

    block_bytes = int(math.ceil(total_bytes * (1.0 - overlap)))
    ovlp_bytes = max(0, total_bytes - block_bytes)
    if block_bytes <= 0:
        _split_cp_edge_fine(edge, 1.0, node_cls, edge_cls)
        return

    block_edge = edge_cls(
        name=f"{edge.name}_block",
        # THE op-id-reuse quirk: block reuses the source edge's op_id.
        op_id=getattr(edge, "op_id", 0),
        duration=0,
        is_dp=edge.is_dp,
        comm_size_bytes=block_bytes,
        comm_type=edge.comm_type,
        participants=edge.participants,
        comm_interconnect_type=edge.comm_interconnect_type,
    )
    ovlp_edge = None
    if ovlp_bytes > 0:
        ovlp_edge = edge_cls(
            name=f"{edge.name}_ovlp",
            op_id=getattr(edge, "op_id", 0),
            duration=0,
            is_dp=edge.is_dp,
            comm_size_bytes=ovlp_bytes,
            comm_type=edge.comm_type,
            participants=edge.participants,
            comm_interconnect_type=edge.comm_interconnect_type,
        )
        _copy_fine_edge_metadata(edge, ovlp_edge)
    _copy_fine_edge_metadata(edge, block_edge)

    for pred in preds:
        _detach_edge(pred, edge)
        _connect_edge(pred, block_edge)

    for attention in attention_children:
        _detach_edge(edge, attention)
        _connect_edge(block_edge, attention)

    if ovlp_edge is not None:
        _connect_edge(block_edge, ovlp_edge)

    for succ in succs:
        if succ in attention_children:
            continue
        _detach_edge(edge, succ)
        for attention in attention_children:
            _connect_edge(attention, succ)
        if ovlp_edge is not None:
            _connect_edge(ovlp_edge, succ)
        else:
            _connect_edge(block_edge, succ)


def apply_overlap_to_fine_root(
    root: Any,
    parallelism_mode: Any,
    tp_overlap: float,
    tp_sp_overlap: float,
    cp_overlap: float,
) -> Any:
    """Apply TP/TP-SP/CP overlap rewrites to a proto graph in place — the
    (only remaining) implementation of the legacy
    ``llm_execution.apply_overlap_transforms`` rewrite, run BEFORE lowering
    (same point in the legacy pipeline: flatten -> overlap -> propagate ->
    lower). Operates on the ``FineNode``/``FineEdge`` proto elements shared
    by the FINE builder (``program.pipeline_fine``) and the BLOCK builder
    (``program.block_program``, M4)."""
    from program.pipeline_fine import FineEdge, FineNode  # lazy

    node_cls = FineNode
    edge_cls = FineEdge

    if root is None:
        return None

    mode = _mode_label(parallelism_mode)

    # -- tp / tp_sp ------------------------------------------------------
    if mode == "tensor_sequence":
        tp_value = tp_sp_overlap
    elif mode in {"tensor", "tensor_context_hybrid"}:
        tp_value = tp_overlap
    else:
        tp_value = 0.0
    if tp_value > 0.0:
        nodes_to_process: List[Any] = []
        visited = set()
        stack: List[Any] = list(root) if isinstance(root, (list, tuple)) else [root]
        while stack:
            obj = stack.pop()
            obj_id = id(obj)
            if obj_id in visited:
                continue
            visited.add(obj_id)
            if isinstance(obj, node_cls):
                nodes_to_process.append(obj)
            for child in getattr(obj, "children", []):
                stack.append(child)

        for node in nodes_to_process:
            tp_children = [
                child for child in getattr(node, "children", [])
                if isinstance(child, edge_cls)
                and getattr(child, "comm_interconnect_type", None) == "tp"
            ]
            if not tp_children:
                continue
            _split_tp_node_fine(node, tp_children, tp_value, node_cls)

    # -- cp ---------------------------------------------------------------
    if mode in {"context", "tensor_context_hybrid"} and cp_overlap > 0.0:
        cp_edges: List[Any] = []
        visited = set()
        stack = list(root) if isinstance(root, (list, tuple)) else [root]
        while stack:
            obj = stack.pop()
            obj_id = id(obj)
            if obj_id in visited:
                continue
            visited.add(obj_id)
            if isinstance(obj, edge_cls) and getattr(obj, "comm_interconnect_type", None) == "cp":
                cp_edges.append(obj)
            for child in getattr(obj, "children", []):
                stack.append(child)

        for edge in cp_edges:
            _split_cp_edge_fine(edge, cp_overlap, node_cls, edge_cls)

    return root


# ---------------------------------------------------------------------------
# Object-graph scaffolding (uids -> object refs -> uids)
# ---------------------------------------------------------------------------


def _objectify(program: Program) -> None:
    ops = program.ops
    for op in ops:
        if isinstance(op, TransferOp):
            op._producer_obj = ops[op.producer]
            op._consumer_objs = [ops[c] for c in op.consumers]
        else:
            op._dep_objs = [ops[d] for d in op.deps]
            op._post_dep_objs = [ops[d] for d in op.post_deps]


def _successor_map(program: Program) -> Dict[int, List[Any]]:
    """id(op) -> ops listing it in ``_dep_objs`` (main ops only)."""
    succs: Dict[int, List[Any]] = {}
    for op in program.ops:
        if isinstance(op, TransferOp):
            continue
        for dep in op._dep_objs:
            succs.setdefault(id(dep), []).append(op)
    return succs


def _renumber(program: Program, ordered_ops: List[Any]) -> None:
    for idx, op in enumerate(ordered_ops):
        op.uid = idx
    for op in ordered_ops:
        if isinstance(op, TransferOp):
            op.producer = op._producer_obj.uid
            op.deps = (op.producer,)
            op.consumers = tuple(dict.fromkeys(c.uid for c in op._consumer_objs))
            del op._producer_obj
            del op._consumer_objs
        else:
            # legacy_lowering dep convention: sorted unique uids.
            op.deps = tuple(dict.fromkeys(sorted(d.uid for d in op._dep_objs)))
            op.post_deps = tuple(dict.fromkeys(d.uid for d in op._post_dep_objs))
            del op._dep_objs
            del op._post_dep_objs
    program.ops = ordered_ops


# ---------------------------------------------------------------------------
# TP / TP-SP overlap
# ---------------------------------------------------------------------------


def apply_tp_overlap(
    program: Program,
    parallelism_mode: Any,
    tp_overlap: float,
    tp_sp_overlap: float,
) -> Program:
    """Port of ``_apply_tp_overlap_transforms`` + ``_split_tp_node``."""
    mode = _mode_label(parallelism_mode)
    if mode == "tensor_sequence":
        overlap = tp_sp_overlap
    elif mode in {"tensor", "tensor_context_hybrid"}:
        overlap = tp_overlap
    else:
        return program
    if overlap <= 0.0:
        return program

    _objectify(program)
    succs = _successor_map(program)
    transfers = [op for op in program.ops if isinstance(op, TransferOp)]

    inserts: Dict[int, ComputeOp] = {}  # id(tail) -> head
    changed = False

    computes = [op for op in program.ops if isinstance(op, ComputeOp)]
    for node in computes:
        tp_colls = [
            op
            for op in succs.get(id(node), [])
            if isinstance(op, CollectiveOp) and op.interconnect == "tp"
        ]
        if not tp_colls:
            continue

        duration = float(node.duration[0] or 0.0) if node.duration else 0.0
        if duration <= 0.0:
            continue

        if overlap >= 1.0:
            # Hoist: collectives take the node's deps and its transfer
            # consumer slots; their successors gain a dep on the node.
            node_deps = list(node._dep_objs)
            for coll in tp_colls:
                coll._dep_objs = [d for d in coll._dep_objs if d is not node] + list(node_deps)
            for transfer in transfers:
                if node in transfer._consumer_objs:
                    for coll in tp_colls:
                        if coll not in transfer._consumer_objs:
                            transfer._consumer_objs.append(coll)
            for coll in tp_colls:
                for succ in succs.get(id(coll), []):
                    if node not in succ._dep_objs:
                        succ._dep_objs.append(node)
            changed = True
            continue

        head_duration = duration * (1.0 - overlap)
        tail_duration = duration * overlap
        if head_duration <= 0.0 or tail_duration <= 0.0:
            continue

        tail = node
        head = ComputeOp(
            uid=-1,
            name=f"{tail.name}_head",
            device=tail.device,
            duration=(head_duration,),
            role=tail.role,
            direction=tail.direction,
            mem_kind=tail.mem_kind,
            recompute=tail.recompute,
            param_gather=tail.param_gather,
            micro_batch=tail.micro_batch,
            layer=tail.layer,
            is_moe_layer=tail.is_moe_layer,
            # THE op-id-reuse quirk: the head shares the source's op_id.
            legacy_op_id=tail.legacy_op_id,
        )
        head._dep_objs = list(tail._dep_objs)
        head._post_dep_objs = list(tail._post_dep_objs)
        tail._post_dep_objs = []
        tail.duration = (tail_duration,)
        tail._dep_objs = [head]

        for coll in tp_colls:
            coll._dep_objs = [head if d is tail else d for d in coll._dep_objs]
        for coll in tp_colls:
            for succ in succs.get(id(coll), []):
                if succ is not tail and tail not in succ._dep_objs:
                    succ._dep_objs.append(tail)
        for transfer in transfers:
            if tail in transfer._consumer_objs:
                transfer._consumer_objs = [
                    head if c is tail else c for c in transfer._consumer_objs
                ]
        inserts[id(tail)] = head
        changed = True

    if not changed:
        # Undo scaffolding without structural change.
        _renumber(program, list(program.ops))
        return program

    ordered: List[Any] = []
    for op in program.ops:
        head = inserts.get(id(op))
        if head is not None:
            ordered.append(head)  # head immediately before its tail
        ordered.append(op)
    _renumber(program, ordered)
    validate_program(program, check_races=False)
    return program


# ---------------------------------------------------------------------------
# CP overlap
# ---------------------------------------------------------------------------


def apply_cp_overlap(
    program: Program,
    parallelism_mode: Any,
    cp_overlap: float,
) -> Program:
    """Port of ``_apply_cp_overlap_transforms`` + ``_split_cp_edge``."""
    mode = _mode_label(parallelism_mode)
    if mode not in {"context", "tensor_context_hybrid"}:
        return program
    overlap = cp_overlap
    if overlap <= 0.0:
        return program

    _objectify(program)
    succs = _successor_map(program)

    inserts_after: Dict[int, CollectiveOp] = {}  # id(block) -> ovlp
    changed = False

    cp_edges = [
        op for op in program.ops if isinstance(op, CollectiveOp) and op.interconnect == "cp"
    ]
    for edge in cp_edges:
        succ_list = list(succs.get(id(edge), []))
        attention_children = [
            op
            for op in succ_list
            if isinstance(op, ComputeOp) and "attention" in str(op.name).lower()
        ]
        if not attention_children:
            continue

        total_bytes = int(edge.size_bytes or 0)
        preds = list(edge._dep_objs)

        block_bytes = int(math.ceil(total_bytes * (1.0 - overlap))) if total_bytes > 0 else 0
        degenerate = overlap >= 1.0 or total_bytes <= 0 or block_bytes <= 0

        if degenerate:
            # Attention consumes the edge's deps directly; non-attention
            # successors keep the edge dep and gain the attention nodes.
            for attention in attention_children:
                attn_deps = [d for d in attention._dep_objs if d is not edge]
                for pred in preds:
                    if pred not in attn_deps:
                        attn_deps.append(pred)
                attention._dep_objs = attn_deps
            for succ in succ_list:
                if succ in attention_children:
                    continue
                for attention in attention_children:
                    if attention not in succ._dep_objs:
                        succ._dep_objs.append(attention)
            changed = True
            continue

        ovlp_bytes = max(0, total_bytes - block_bytes)
        original_name = edge.name
        edge.name = f"{original_name}_block"
        edge.size_bytes = block_bytes

        ovlp_edge: Optional[CollectiveOp] = None
        if ovlp_bytes > 0:
            ovlp_edge = CollectiveOp(
                uid=-1,
                name=f"{original_name}_ovlp",
                device=edge.device,
                coll=edge.coll,
                size_bytes=ovlp_bytes,
                participants=edge.participants,
                interconnect=edge.interconnect,
                is_dp=edge.is_dp,
                label=None,  # re-derived by the relabel pass below
                group=None,
                # THE op-id-reuse quirk: ovlp shares the source's op_id.
                legacy_op_id=edge.legacy_op_id,
            )
            ovlp_edge._dep_objs = [edge]
            ovlp_edge._post_dep_objs = []
            inserts_after[id(edge)] = ovlp_edge

        for succ in succ_list:
            if succ in attention_children:
                continue
            new_deps = [d for d in succ._dep_objs if d is not edge]
            for attention in attention_children:
                if attention not in new_deps:
                    new_deps.append(attention)
            if ovlp_edge is not None:
                if ovlp_edge not in new_deps:
                    new_deps.append(ovlp_edge)
            else:
                if edge not in new_deps:
                    new_deps.append(edge)
            succ._dep_objs = new_deps
        changed = True

    if not changed:
        _renumber(program, list(program.ops))
        return program

    ordered: List[Any] = []
    for op in program.ops:
        ordered.append(op)
        ovlp = inserts_after.get(id(op))
        if ovlp is not None:
            ordered.append(ovlp)  # ovlp immediately after its block
    _renumber(program, ordered)
    _relabel(program)
    validate_program(program, check_races=False)
    return program


def apply_overlap_transforms(
    program: Program,
    parallelism_mode: Any,
    tp_overlap: float,
    tp_sp_overlap: float,
    cp_overlap: float,
) -> Program:
    """Program-level counterpart of ``llm_execution.apply_overlap_transforms``."""
    if program is None:
        return None
    program = apply_tp_overlap(program, parallelism_mode, tp_overlap, tp_sp_overlap)
    program = apply_cp_overlap(program, parallelism_mode, cp_overlap)
    return program


# ---------------------------------------------------------------------------
# Label re-derivation (after cp renames/creations)
# ---------------------------------------------------------------------------


class _EdgeShim:
    """Minimal object surface for ``_assign_collective_labels_with_members``
    (needs only ``op_id`` for the per-name sort)."""

    __slots__ = ("op", "op_id")

    def __init__(self, op: CollectiveOp) -> None:
        self.op = op
        self.op_id = op.legacy_op_id if op.legacy_op_id is not None else op.uid


def _relabel(program: Program) -> None:
    """Re-derive labels/groups for the labeled collective family with the
    legacy ``_assign_collective_labels`` rules. Label assignment depends
    only on (base name, op_id order within the name, member sets) — never
    on graph discovery order — so recomputing it from the Program matches
    the legacy in-graph assignment (suffix counters are per-name and the
    emitter re-sorts labels for gid interning anyway)."""
    from program.legacy_lowering import (
        _assign_collective_labels_with_members,
        _build_axis_groups,
        _compute_stage_axis_coords,
        _extract_axis_layout,
    )

    descriptor = program.layout.descriptor() if program.layout.axis_order else None
    axis_order, axis_sizes, _ = _extract_axis_layout(descriptor)
    devices = list(program.devices)
    dp_count = max(int(program.dp_count), 1)
    stage_axis_coords = _compute_stage_axis_coords(devices, axis_order, axis_sizes)
    stage_to_ranks = {
        device: [program.rank_for(device, dp_idx) for dp_idx in range(dp_count)]
        for device in devices
    }
    axis_groups = _build_axis_groups(axis_order, axis_sizes, stage_axis_coords, stage_to_ranks)

    rank_to_stage_dp0 = {
        ranks[0]: stage for stage, ranks in stage_to_ranks.items() if ranks
    }

    labeled_family = [
        op
        for op in program.ops
        if isinstance(op, CollectiveOp)
        and op.interconnect
        and op.interconnect not in {"dp", "pp", "pipeline"}
    ]
    shims = [_EdgeShim(op) for op in labeled_family]
    tp_collective_groups: Dict[str, List[_EdgeShim]] = {}
    collective_info: Dict[_EdgeShim, Dict[str, Any]] = {}
    for shim in shims:
        op = shim.op
        tp_collective_groups.setdefault(op.name, []).append(shim)
        collective_info[shim] = {
            "stage": op.device,
            "interconnect_type": op.interconnect,
            "participants": int(op.participants or 0),
            "name": op.name,
        }

    labels, edge_primary = _assign_collective_labels_with_members(
        tp_collective_groups,
        collective_info,
        axis_order,
        axis_sizes,
        stage_axis_coords,
        axis_groups,
        stage_to_ranks,
        dp_count,
    )

    groups: Dict[GroupKey, CommGroup] = {}
    for shim in shims:
        op = shim.op
        label = labels.get(shim)
        op.label = label
        if label is None:
            op.group = None
            op.is_dp = True
            continue
        op.is_dp = False
        axis, primary_members = edge_primary[shim]
        member_stages = tuple(sorted(rank_to_stage_dp0[r] for r in primary_members))
        key = GroupKey(axis=axis, members=member_stages)
        op.group = key
        if key not in groups:
            groups[key] = CommGroup(key=key, label=label)
    program.groups = groups
