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

LIMITATIONS (1-2 discovered by the M3a differential, documented per the
M3a mandate — two ways the legacy in-graph rewrite reaches state a
post-lowering pass no longer has; 3 is a guard on the Program-level pass):

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

3. PER-DP PROFILES. :func:`apply_tp_overlap` splits SCALAR durations only:
   it raises ``ValueError`` on any split candidate whose ``duration``
   carries a per-DP profile (``len > 1``, written by ``program.retime``) —
   the head/tail 1-tuple split would silently discard the dp1..dpN entries,
   violating the DESIGN §2.4 "loud, never silently stale" discipline.

The FINE builder therefore applies :func:`apply_overlap_to_fine_root` — a
verbatim port of the legacy rewrite over the builder's proto elements —
BEFORE lowering, which reproduces both by construction; it is the only
overlap implementation production uses. Of the Program -> Program passes,
only :func:`apply_tp_overlap` (with its ``_objectify`` / ``_successor_map``
/ ``_renumber`` scaffolding) is retained, as the seed of the post-M9
``program`` id-policy API where the legacy creation-order coincidences
retire: it is exact for programs whose split computes feed no cross-device
consumers (e.g. single-stage single-microbatch programs) and is unit-
covered by ``tests/test_fine_builder_diff.py``. The Program-level cp chain
(``apply_cp_overlap`` / ``apply_overlap_transforms`` / the ``_relabel``
label re-derivation) had zero callers and no test coverage and was deleted;
resurrect it from git history when M9 actually needs it.

Dep rewires mirror the legacy walker outcomes on the rewritten graph:

* tp (0 < f < 1): head takes the node's deps, transfer-consumer slots and
  post_deps; the node (tail, duration ``f*d``) depends only on head; the tp
  collectives move their node-dep to head; every successor of a moved tp
  collective gains a dep on tail (legacy ``_connect_edge(tail, succ)``).
* tp (f >= 1): the tp collectives are hoisted before the node (deps := the
  node's deps; consumers of the node's transfers extended with them) and
  their successors gain a dep on the node.
* cp (proto-level ``_split_cp_edge_fine`` only): 0 < f < 1 — attention
  keeps its dep on block; non-attention successors trade the edge dep for
  attention (+ ovlp when present); f >= 1 or no bytes — attention trades
  the edge dep for the edge's deps; non-attention successors keep the edge
  dep and gain attention. Collective labels are assigned AFTER the rewrite
  by the shared lowering pass, exactly like legacy assigned labels
  post-rewrite.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from program.ir import CollectiveOp, ComputeOp, Program, TransferOp
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
    # Shared attribute tuple defined next to FineNode/FineEdge. PINNED
    # QUIRK: OVERLAP_NODE_COPY_ATTRS deliberately omits ``is_moe_layer``
    # (verbatim legacy ``_copy_node_metadata`` port) — see the constant's
    # comment in program/pipeline_fine.py before changing it.
    from program.pipeline_fine import OVERLAP_NODE_COPY_ATTRS  # lazy (cycle)

    for attr in OVERLAP_NODE_COPY_ATTRS:
        if hasattr(source, attr):
            setattr(target, attr, getattr(source, attr))


def _copy_fine_edge_metadata(source: Any, target: Any) -> None:
    from program.pipeline_fine import OVERLAP_EDGE_COPY_ATTRS  # lazy (cycle)

    for attr in OVERLAP_EDGE_COPY_ATTRS:
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
    """Port of ``_apply_tp_overlap_transforms`` + ``_split_tp_node``.

    Scalar durations only: raises ``ValueError`` on a split candidate with
    a per-DP duration profile (module docstring, LIMITATIONS item 3).
    """
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

        if node.duration and len(node.duration) > 1:
            # LIMITATIONS item 3: the head/tail (and hoist) rewrites read
            # duration[0] and write 1-tuples, which would silently discard
            # a retimed op's dp1..dpN profile entries. Fail loudly instead
            # (DESIGN §2.4). No production caller passes retimed programs.
            raise ValueError(
                f"apply_tp_overlap cannot split op {node.uid} ('{node.name}'): "
                f"it carries a per-DP duration profile (len {len(node.duration)}); "
                "splitting would silently collapse the profile to duration[0]. "
                "Apply overlap before retiming, or split the profile explicitly."
            )

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
