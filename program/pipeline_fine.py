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

"""``build_fine_program`` — the direct FINE-program builder (M3a).

Replaces ``PipelineGraphFlattener`` for the flattened *execution* path: the
fine per-rank structure is enumerated directly from a
:class:`~program.schedule.ScheduleSpec` + :class:`~program.block.
BlockTemplate` — no legacy ``simulate_train_graph`` Graph is constructed and
no legacy graph is cloned. The enumeration walks the schedule's event DAG
(:func:`program.schedule.build_pipeline_events`) with EXACTLY the legacy
flattener's traversal (a first-encounter DFS with per-event expansion), so
the op-id counter — which keys the per-stage emission toposort — advances in
the identical sequence. Reproduced rules, each pinned by the golden gates
(DESIGN.md §6):

* per-(tp*cp*ep)-rank GEMM chains from the BlockTemplate, comm collectives
  chained serially in ``comm_keys`` order (placement pre/post is a
  ``construct_transformer_graph`` concept the flattener never used);
* per-rank cross-stage pipeline transfers with ``ceil(total / par_degree)``
  bytes, anchored on BOTH the chain tail and the nearest compute ancestor
  (the compute-anchor double-dep);
* DP collectives attached to ``rank_tails[0]`` ONLY (asymmetric, legacy);
* ZeRO-3 per-rank gathers with UN-DIVIDED ``int(base_bytes)`` and the
  cross-device placement expressed as an explicit ±1 pp-stage coordinate
  hop (the legacy ``hw ± par_degree`` offset hack — identical because the
  canonical layout's pp stride equals ``tp*cp*ep``);
* softmax pinned to rank 0 of its (last) stage; embedding kept at device 0;
  optimizer expanded per rank with 1-to-1 tail wiring; EP sync collectives
  (dormant: flattened MoE is rejected upstream);
* same-stage zero-byte PIPELINE control edges (lowered to same-device
  TransferOps that the ET emitter elides — DESIGN §2.2).

M3a scope note: the builder owns the fine STRUCTURE; the projection of that
structure into Program uid order (per-stage Kahn keyed by op_id, Step-11
transfer replay, collective label assignment, gmap traffic collection +
SCOTCH stage remap) is shared with the quarantined
:func:`program.legacy_lowering.lower_to_program`, invoked here over the
builder's own typed elements. Both the legacy path (flatten -> lower) and
this builder therefore agree on ordering semantics by sharing code; M3b/M8
collapse the shared pass into the builder when the legacy graph path dies.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from memory_estimation import mem_kind_from_op_name
from timing_model import CollectiveType

from program.block import BlockTemplate, CommMeta
from program.ir import Program
from program.layout import RankLayout, cluster_coords
from program.schedule import (
    CommEvent,
    ComputeEvent,
    ScheduleSpec,
    build_pipeline_events,
)


# ---------------------------------------------------------------------------
# Fine proto elements (the builder's own typed stand-ins for the flattened
# legacy Node/Edge objects; attribute surface matches what the shared
# lowering pass reads).
# ---------------------------------------------------------------------------


class FineNode:
    """A fine compute element (flattened legacy ``Node`` stand-in)."""

    def __init__(
        self,
        name: str,
        op_id: int,
        hw_id: int,
        duration: float,
        fwd: bool = True,
        mem_kind: Any = None,
        recompute: bool = False,
        param_gather: bool = False,
    ) -> None:
        self.name = name
        self.op_id = op_id
        self.hw_id = hw_id
        self.duration = duration
        self.fwd = fwd
        self.mem_kind = mem_kind
        self.recompute = bool(recompute)
        self.param_gather = bool(param_gather)
        self.children: List[Any] = []
        self.parents: List[Any] = []

    def add_child(self, obj: Any) -> None:
        self.children.append(obj)
        obj.parents.append(self)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"FineNode({self.name},op={self.op_id},hw={self.hw_id})"


class FineEdge:
    """A fine comm element (flattened legacy ``Edge`` stand-in)."""

    def __init__(
        self,
        name: str,
        op_id: int,
        duration: float = 0,
        is_dp: bool = False,
        comm_size_bytes: Any = 0,
        comm_type: Optional[CollectiveType] = None,
        participants: int = 1,
        comm_interconnect_type: Optional[str] = None,
    ) -> None:
        self.name = name
        self.op_id = op_id
        self.duration = duration
        self.is_dp = is_dp
        self.comm_size_bytes = comm_size_bytes
        if comm_type is not None and not isinstance(comm_type, CollectiveType):
            raise TypeError(
                f"FineEdge.comm_type must be a CollectiveType or None "
                f"(got {type(comm_type).__name__})"
            )
        self.comm_type = comm_type
        self.participants = participants
        self.comm_interconnect_type = comm_interconnect_type
        self.children: List[Any] = []
        self.parents: List[Any] = []

    def add_child(self, obj: Any) -> None:
        self.children.append(obj)
        obj.parents.append(self)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"FineEdge({self.name},op={self.op_id})"


# ---------------------------------------------------------------------------
# The expansion (PipelineGraphFlattener port over schedule events)
# ---------------------------------------------------------------------------


class _FineExpander:
    """Port of ``PipelineGraphFlattener`` (llm_execution.py 339-994) with
    the coarse legacy graph replaced by schedule events and the clone
    outputs replaced by :class:`FineNode`/:class:`FineEdge`. The traversal,
    branch structure, op-id counter discipline, and attach orders are kept
    verbatim so the resulting op ids and children-list orders match the
    legacy flattener's output element-for-element."""

    def __init__(
        self,
        spec: ScheduleSpec,
        block_templates: Mapping[str, BlockTemplate],
        layout: Optional[RankLayout],
    ) -> None:
        template = block_templates.get("dense")
        if template is None or not template.entries:
            raise ValueError("Transformer GEMM template is missing")
        self._template = template
        self._template_moe = block_templates.get("moe")
        if self._template_moe is not None and not self._template_moe.entries:
            raise ValueError("MoE transformer GEMM template is missing")

        self.spec = spec
        par_degree = int(spec.tp) * int(spec.cp) * int(spec.ep)
        self._par_degree = max(1, par_degree)
        self._zero_stage = int(spec.zero_stage)
        self._rank_layout_obj: Optional[RankLayout] = None
        self._tp_size = int(spec.tp)
        self._cp_size = int(spec.cp)
        self._ep_size = int(spec.ep)
        self._pp_size = int(spec.pp)
        if layout is not None and layout.axis_order:
            self._configure_rank_layout(layout)

        self._clone_cache: Dict[int, Any] = {}
        self._op_id_counter: int = 0

    def _configure_rank_layout(self, layout: RankLayout) -> None:
        """Port of ``PipelineGraphFlattener._configure_rank_layout``."""
        axis_sizes = layout.axis_sizes
        self._rank_layout_obj = layout
        self._tp_size = max(1, axis_sizes.get("tp", self._tp_size))
        self._cp_size = max(1, axis_sizes.get("cp", self._cp_size))
        self._ep_size = max(1, axis_sizes.get("ep", self._ep_size))
        self._pp_size = max(1, axis_sizes.get("pp", self._pp_size))
        if self._par_degree != self._tp_size * self._cp_size * self._ep_size:
            raise ValueError(
                f"Inconsistent tensor/context/expert parallel factors: tp={self._tp_size}, "
                f"cp={self._cp_size}, ep={self._ep_size}, "
                f"product does not equal par_degree={self._par_degree}"
            )

    # -- shared helpers ---------------------------------------------------
    def _next_op_id(self) -> int:
        self._op_id_counter += 1
        return self._op_id_counter

    def _hw_id_for_rank(self, stage_id: Any, tp_rank: int) -> int:
        """Port of ``PipelineGraphFlattener._hw_id_for_rank``."""
        stage_int = int(stage_id) if stage_id is not None else 0
        tp_rank_int = int(tp_rank)
        if tp_rank_int < 0 or tp_rank_int >= self._par_degree:
            raise ValueError(
                f"tp_rank {tp_rank_int} is out of range for par_degree {self._par_degree}"
            )
        layout = self._rank_layout_obj
        if layout is None:
            return stage_int * self._par_degree + tp_rank_int
        coords = cluster_coords(
            layout.axis_order,
            tp_rank_int,
            stage_int,
            tp_size=self._tp_size,
            cp_size=self._cp_size,
            ep_size=self._ep_size,
            pp_size=self._pp_size,
        )
        return layout.linearize(coords)

    def _offset_stage_device(self, base_device: int, stage_delta: int) -> int:
        """ZeRO-3 cross-device gather placement as an explicit pp-stage hop.

        Legacy computed ``base_device ± par_degree`` (the "HACK!!!!" in
        ``_ensure_zero3_per_rank_edges``); with the canonical layout the pp
        stride equals ``tp*cp*ep`` so a ±1 pp-coordinate hop is the same
        device. Without a layout the legacy flat arithmetic is used.
        """
        layout = self._rank_layout_obj
        if layout is None:
            return int(base_device) + stage_delta * self._par_degree
        coords = layout.coords_of(int(base_device))
        coords["pp"] = coords.get("pp", 0) + stage_delta
        return layout.linearize(coords)

    def _should_shard_zero3_transformer(self, edge: Any) -> bool:
        """Port of ``PipelineGraphFlattener._should_shard_zero3_transformer``."""
        if self._par_degree <= 1 or self._zero_stage < 3:
            return False
        if isinstance(edge, CommEvent):
            if getattr(edge, "tp_shard", False):
                return True
        return False

    # -- ZeRO-3 per-rank gathers ------------------------------------------
    def _ensure_zero3_per_rank_edges(
        self,
        edge: CommEvent,
        cross_device: bool = False,
        rank_heads: Optional[List[Any]] = None,
        rank_tails: Optional[List[Any]] = None,
        hw_ids: Optional[List[int]] = None,
    ) -> List[FineEdge]:
        """Port of ``PipelineGraphFlattener._ensure_zero3_per_rank_edges``.

        Per-rank bytes stay the UN-DIVIDED ``int(base_bytes)`` (legacy
        modeling quirk, DESIGN §6). Anchors are ``rank_tails`` unless the
        source event carries a "backward" direction (coarse ZeRO-3 gather
        events never do — the legacy edges lacked the attribute — so the
        rank_tails anchor is what actually runs; the branch is kept for
        fidelity)."""

        transformer_mode: Any = ""
        per_rank_edges: List[FineEdge] = []
        if rank_tails and rank_heads:
            wire_anchors = rank_tails
            direction = getattr(edge, "direction", None)
            if direction and str(direction).lower() == "backward":
                wire_anchors = rank_heads
            iterable: Sequence[Any] = wire_anchors
            transformer_mode = True
        elif hw_ids:
            iterable = hw_ids
            transformer_mode = False

        if transformer_mode == "":
            raise Exception(
                "Invalid _ensure_zero3_per_rank_edges call. At least one of "
                "(rank_tails,rank_heads) or (hw_ids) must be provided."
            )

        # Legacy offset hack "not 'bwd' in edge.name" — typed here: the
        # schedule stamps zero3_offset_hint = "fwd"/"bwd" on the gathers.
        hint = getattr(edge, "zero3_offset_hint", None)
        if hint is None:
            hint = "bwd" if "bwd" in edge.name else "fwd"
        stage_delta = 1 if hint == "fwd" else -1

        for r, item in enumerate(iterable):
            base_bytes = getattr(edge, "comm_size_bytes", 0)
            per_rank_bytes = int(base_bytes)
            gather_edge = FineEdge(
                name=f"{edge.name}_rank{r}",
                op_id=self._next_op_id(),
                duration=0,
                is_dp=True,
                comm_size_bytes=per_rank_bytes,
                comm_type=getattr(edge, "comm_type", None),
                participants=getattr(edge, "participants", 0),
                comm_interconnect_type=getattr(edge, "comm_interconnect_type", None),
            )
            gather_edge.tp_rank = r
            gather_edge.stage_id = getattr(edge, "stage_id", None)
            gather_edge.micro_batch_index = getattr(edge, "micro_batch_index", None)
            gather_edge.layer_index = getattr(edge, "layer_index", None)
            gather_edge.direction = getattr(edge, "direction", None)
            gather_edge.tp_shard = True
            if transformer_mode:
                if getattr(item, "hw_id", None) is not None:
                    base_device = item.hw_id
                else:
                    base_device = getattr(item, "local_hw_id", None)
                if cross_device:
                    gather_edge.local_hw_id = self._offset_stage_device(base_device, stage_delta)
                else:
                    gather_edge.local_hw_id = base_device
                # has the same children as item
                for child in getattr(item, "children", []):
                    gather_edge.add_child(child)
            else:
                gather_edge.local_hw_id = item
                # cannot attach children here: they do not exist yet.

            per_rank_edges.append(gather_edge)

        return per_rank_edges

    # -- the traversal ----------------------------------------------------
    def build(self, root: Any) -> Any:
        if root is None:
            raise ValueError("Pipeline root is required for flattening")
        return self._clone(root)

    def _clone(self, obj: Any) -> Any:
        if obj is None:
            return None

        obj_id = id(obj)
        if obj_id in self._clone_cache:
            return self._clone_cache[obj_id]

        if isinstance(obj, ComputeEvent):
            if obj.role in ("layer", "recompute"):
                expanded = self._expand_transformer_node(obj)
                self._clone_cache[obj_id] = expanded
                return expanded

            if "linear_softmax" in obj.name:
                # Softmax is pinned to rank 0 of its (last) stage.
                cloned = FineNode(
                    obj.name,
                    self._next_op_id(),
                    self._hw_id_for_rank(obj.hw_id, 0),
                    obj.duration,
                    fwd=obj.fwd,
                )
            elif "optimizer" in obj.name:
                # Optimizer nodes are expanded per cluster rank.
                cloned_nodes = []
                for tp_rank in range(self._par_degree):
                    hw_id = self._hw_id_for_rank(obj.hw_id, tp_rank)
                    cloned_node = FineNode(
                        name=f"{obj.name}_rank{tp_rank}",
                        op_id=self._next_op_id(),
                        hw_id=hw_id,
                        duration=obj.duration,
                        fwd=obj.fwd,
                    )
                    cloned_nodes.append(cloned_node)

                cloned_tuple = tuple(cloned_nodes)
                self._clone_cache[obj_id] = cloned_tuple

                for cloned_node in cloned_nodes:
                    self._copy_metadata(obj, cloned_node)

                for child in getattr(obj, "children", []):
                    child_clone = self._clone(child)
                    if child_clone is not None:
                        self._attach(cloned_tuple, child_clone)

                return cloned_tuple
            else:
                cloned = FineNode(
                    obj.name,
                    self._next_op_id(),
                    obj.hw_id,
                    obj.duration,
                    fwd=obj.fwd,
                )

            # ZeRO-3 sibling scan (legacy: parents' other children that are
            # sharded gathers become per-rank edges keyed on this node).
            zero3_attachments: List[CommEvent] = []
            for parent in getattr(obj, "parents", []):
                for sibling in getattr(parent, "children", []):
                    if sibling is obj:
                        continue
                    if self._should_shard_zero3_transformer(sibling):
                        zero3_attachments.append(sibling)

            hw_ids = []
            for tp_rank in range(self._par_degree):
                hw_ids.append(self._hw_id_for_rank(obj.hw_id, tp_rank))

            per_rank_edges: List[FineEdge] = []
            for zero3_attachment in zero3_attachments:
                per_rank_edges = self._ensure_zero3_per_rank_edges(
                    zero3_attachment, rank_heads=None, rank_tails=None, hw_ids=hw_ids
                )
                self._clone_cache[id(zero3_attachment)] = per_rank_edges

            self._clone_cache[obj_id] = cloned
            self._copy_metadata(obj, cloned)
            for child in getattr(obj, "children", []):
                child_clone = self._clone(child)
                if child_clone is not None:
                    if per_rank_edges:
                        for per_rank_edge in per_rank_edges:
                            self._attach(per_rank_edge, child_clone)
                    self._attach(cloned, child_clone)
            return cloned

        if isinstance(obj, CommEvent):
            cloned_edge = FineEdge(
                obj.name,
                self._next_op_id(),
                obj.duration,
                is_dp=getattr(obj, "is_dp", False),
                comm_size_bytes=getattr(obj, "comm_size_bytes", 0),
                comm_type=getattr(obj, "comm_type", None),
                participants=getattr(obj, "participants", 1),
                comm_interconnect_type=getattr(obj, "comm_interconnect_type", None),
            )
            cloned_edge.local_hw_id = getattr(obj, "local_hw_id", None)
            self._clone_cache[obj_id] = cloned_edge
            self._copy_metadata(obj, cloned_edge)
            for child in getattr(obj, "children", []):
                child_clone = self._clone(child)
                if child_clone is not None:
                    self._attach(cloned_edge, child_clone)
            return cloned_edge

        raise TypeError(f"Unsupported schedule event type: {type(obj)!r}")

    # -- transformer expansion --------------------------------------------
    def _expand_transformer_node(self, node: ComputeEvent) -> Tuple[Any, ...]:
        node_id = id(node)
        if node_id in self._clone_cache:
            cached_entry = self._clone_cache[node_id]
            if isinstance(cached_entry, (list, tuple)):
                return tuple(cached_entry)

        stage_id = getattr(node, "stage_id", node.hw_id)
        micro_batch = getattr(node, "micro_batch_index", None)
        layer_index = getattr(node, "layer_index", None)
        direction = getattr(node, "direction", "forward" if node.fwd else "backward")
        is_moe_layer = bool(getattr(node, "is_moe_layer", False))
        template = (
            self._template_moe if is_moe_layer and self._template_moe else self._template
        )
        gemm_entries = template.entries

        rank_heads: List[Any] = []
        rank_tails: List[Any] = []

        for tp_rank in range(self._par_degree):
            previous: Optional[Any] = None
            head: Optional[Any] = None
            hw_id = self._hw_id_for_rank(stage_id, tp_rank)

            gemm_iterable: Sequence[Any] = gemm_entries
            if direction == "backward":
                gemm_iterable = list(reversed(gemm_entries))

            for gemm_idx, entry in enumerate(gemm_iterable):
                entry_name = entry.name
                cfg = entry.direction(direction)
                duration = cfg.duration
                if duration is None:
                    raise ValueError(
                        f"Missing duration for transformer entry '{entry_name}' in direction '{direction}'"
                    )

                gemm_node = FineNode(
                    name=self._format_gemm_name(entry_name, direction, micro_batch, layer_index, tp_rank),
                    op_id=self._next_op_id(),
                    hw_id=hw_id,
                    duration=duration,
                    fwd=(direction == "forward"),
                    mem_kind=mem_kind_from_op_name(entry_name),
                )
                gemm_node.stage_id = stage_id
                gemm_node.tp_rank = tp_rank
                gemm_node.micro_batch_index = micro_batch
                gemm_node.layer_index = layer_index
                gemm_node.direction = direction
                gemm_node.recompute = bool(getattr(node, "recompute", False))
                gemm_node.param_gather = (gemm_idx == 0)
                gemm_node.is_moe_layer = is_moe_layer

                if previous is not None:
                    previous.add_child(gemm_node)
                previous = gemm_node
                if head is None:
                    head = gemm_node

                for comm_key in cfg.comm_keys:
                    comm_edge = self._create_transformer_comm_edge(
                        comm_key,
                        hw_id,
                        stage_id,
                        micro_batch,
                        layer_index,
                        direction,
                        tp_rank,
                        template_override=template,
                    )
                    previous.add_child(comm_edge)
                    previous = comm_edge

            if head is None:
                raise ValueError("Transformer expansion produced no GEMM nodes")

            rank_heads.append(head)
            rank_tails.append(previous or head)

        dp_children: List[Any] = []
        other_children: List[Any] = []
        zero3_attachments: List[CommEvent] = []

        for child in getattr(node, "children", []):
            comm_type = getattr(child, "comm_interconnect_type", None)
            if comm_type == "dp":
                dp_children.append(child)
            else:
                other_children.append(child)

        for parent in getattr(node, "parents", []):
            for sibling in getattr(parent, "children", []):
                if sibling is node:
                    continue
                if self._should_shard_zero3_transformer(sibling):
                    zero3_attachments.append(sibling)

        # Keep the main trunk pointing to the per-rank compute tails.
        downstream_parents: List[Any] = list(rank_tails)

        # DP collectives are side branches from rank_tails[0] ONLY (legacy
        # asymmetric attach — DESIGN §6 pinned quirk).
        for child in dp_children:
            if self._should_shard_zero3_transformer(child):
                per_rank_edges = self._ensure_zero3_per_rank_edges(
                    child, rank_heads=rank_heads, rank_tails=rank_tails, hw_ids=None
                )

            child_clone = self._clone(child)
            if child_clone is None:
                continue
            self._attach(rank_tails[0], child_clone)

        # Non-DP children: cross-stage PIPELINE edges become per-rank
        # transfers; everything else stays on the trunk.
        for child in other_children:
            is_pipeline_edge = False
            if isinstance(child, CommEvent):
                comm_type = getattr(child, "comm_type", None)
                if comm_type == CollectiveType.PIPELINE:
                    is_pipeline_edge = True
            if is_pipeline_edge:
                # Per-rank byte size: ceil split across the cluster ranks.
                try:
                    total_bytes = int(getattr(child, "comm_size_bytes", 0))
                except Exception:
                    total_bytes = 0
                per_rank_bytes = int(math.ceil(float(total_bytes) / float(max(1, self._par_degree))))

                # Clone the original targets of this pipeline edge first
                # (the legacy DFS descends into the destination subtree
                # before materializing the per-rank edges).
                target_clones: List[Any] = []
                for tgt in getattr(child, "children", []):
                    tgt_clone = self._clone(tgt)
                    if tgt_clone is None:
                        continue
                    target_clones.append(tgt_clone)
                if not target_clones:
                    continue

                for r, tail in enumerate(rank_tails):
                    edge_obj = FineEdge(
                        name=f"{getattr(child, 'name', '')}_rank{r}",
                        op_id=self._next_op_id(),
                        duration=0,
                        is_dp=False,
                        comm_size_bytes=per_rank_bytes,
                        comm_type=CollectiveType.PIPELINE,
                        participants=2,
                        comm_interconnect_type="pp",
                    )
                    edge_obj.is_cross_layer = True
                    tail.add_child(edge_obj)
                    # Anchor on the nearest compute ancestor as well (the
                    # compute-anchor double-dep: the emitted SEND fires off
                    # the last compute, not the trailing collectives).
                    compute_anchor = None
                    cur: Any = tail
                    visited_ids = set()
                    while cur is not None and id(cur) not in visited_ids:
                        visited_ids.add(id(cur))
                        if isinstance(cur, FineNode):
                            compute_anchor = cur
                            break
                        parents = getattr(cur, "parents", [])
                        cur = parents[-1] if parents else None
                    if compute_anchor is not None and compute_anchor is not edge_obj:
                        if compute_anchor != tail:
                            compute_anchor.add_child(edge_obj)

                    for tgt_clone in target_clones:
                        if isinstance(tgt_clone, (list, tuple)):
                            idx = r % len(tgt_clone)
                            edge_obj.add_child(tgt_clone[idx])
                        else:
                            edge_obj.add_child(tgt_clone)

                continue

            child_clone = self._clone(child)
            if child_clone is None:
                continue

            # Optimizer nodes: 1-to-1 tail wiring.
            if (
                isinstance(child, ComputeEvent)
                and "optimizer" in child.name
                and isinstance(child_clone, (list, tuple))
            ):
                if len(child_clone) == len(rank_tails):
                    for r in range(len(rank_tails)):
                        rank_tails[r].add_child(child_clone[r])
                    continue

            self._attach(downstream_parents, child_clone)

        for zero3_edge in zero3_attachments:
            is_cross_device = zero3_edge.local_hw_id != stage_id
            per_rank_edges = self._ensure_zero3_per_rank_edges(
                zero3_edge, cross_device=is_cross_device, rank_heads=rank_heads,
                rank_tails=rank_tails, hw_ids=None,
            )
            self._clone_cache[id(zero3_edge)] = per_rank_edges

        heads_tuple = tuple(rank_heads)
        self._clone_cache[node_id] = heads_tuple
        return heads_tuple

    def _create_transformer_comm_edge(
        self,
        comm_key: str,
        hw_id: int,
        stage_id: int,
        micro_batch: Optional[int],
        layer_index: Optional[int],
        direction: str,
        tp_rank: int,
        *,
        template_override: Optional[BlockTemplate] = None,
    ) -> FineEdge:
        """Port of ``_create_transformer_comm_edge`` + ``create_comm_edge``
        (the ``local_comp_time`` zeroing means no local node exists)."""
        template = template_override or self._template
        meta: CommMeta = template.comm_metadata[comm_key]
        is_dp_edge = meta.interconnect == "dp"

        comm_edge = FineEdge(
            name=comm_key,
            op_id=self._next_op_id(),
            duration=0,
            is_dp=is_dp_edge,
            comm_size_bytes=meta.size_bytes,
            comm_type=meta.kind,
            participants=meta.participants,
            comm_interconnect_type=meta.interconnect,
        )
        comm_edge.local_hw_id = int(hw_id)
        if meta.tp_shard:
            comm_edge.tp_shard = True
        comm_edge.stage_id = stage_id
        comm_edge.micro_batch_index = micro_batch
        comm_edge.layer_index = layer_index
        comm_edge.direction = direction
        comm_edge.tp_rank = tp_rank
        return comm_edge

    def _copy_metadata(self, source: Any, target: Any) -> None:
        for attr in (
            "micro_batch_index",
            "layer_index",
            "direction",
            "stage_id",
            "tp_rank",
            "mem_kind",
            "recompute",
            "is_moe_layer",
        ):
            if hasattr(source, attr):
                setattr(target, attr, getattr(source, attr))

    def _attach(self, parent: Any, child: Any) -> None:
        if parent is None or child is None:
            return
        if isinstance(parent, (list, tuple)):
            for item in parent:
                self._attach(item, child)
            return
        if isinstance(child, (list, tuple)):
            for item in child:
                self._attach(parent, item)
            return
        parent.add_child(child)

    def _format_gemm_name(
        self,
        base_name: str,
        direction: str,
        micro_batch: Optional[int],
        layer_index: Optional[int],
        tp_rank: int,
    ) -> str:
        return f"{base_name}_{direction}_mb{micro_batch}_l{layer_index}_rank{tp_rank}"

    # -- local hw-id propagation ------------------------------------------
    def propagate_local_hw_ids(self, roots: Any) -> None:
        """Port of ``PipelineGraphFlattener._propagate_local_hw_ids``:
        rewrite comm elements' ``local_hw_id`` from the first placed parent
        (this is what maps coarse-stage dp/ZeRO local ids onto fine device
        ids); PIPELINE transfers and per-rank ZeRO-3 gathers keep their
        explicit placement."""
        stack: List[Any] = []
        if isinstance(roots, (list, tuple)):
            stack.extend(list(roots))
        elif roots is not None:
            stack.append(roots)
        visited: set = set()
        while stack:
            obj = stack.pop()
            obj_id = id(obj)
            if obj_id in visited:
                continue
            visited.add(obj_id)

            children = getattr(obj, "children", [])
            if isinstance(children, (list, tuple)):
                stack.extend(children)
            elif children is not None:
                stack.append(children)

            parents = getattr(obj, "parents", None)
            if parents:
                if isinstance(parents, (list, tuple)):
                    stack.extend(parents)
                else:
                    stack.append(parents)

            if isinstance(obj, FineEdge):
                if getattr(obj, "comm_type", None) == CollectiveType.PIPELINE:
                    continue
                if getattr(obj, "tp_shard", False) and getattr(obj, "tp_rank", None) is not None:
                    # Preserve per-rank ZeRO-3 gather placement.
                    continue
                new_hw = None
                for parent in getattr(obj, "parents", []) or []:
                    parent_hw = getattr(parent, "hw_id", None)
                    if parent_hw is None:
                        parent_hw = getattr(parent, "local_hw_id", None)
                    if parent_hw is not None and parent_hw >= 0:
                        new_hw = parent_hw
                        break
                if new_hw is None:
                    stage_id = getattr(obj, "stage_id", None)
                    if stage_id is not None:
                        try:
                            new_hw = self._hw_id_for_rank(stage_id, 0)
                        except Exception:
                            new_hw = None
                if new_hw is not None and new_hw >= 0:
                    obj.local_hw_id = new_hw


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_fine_root(
    schedule_spec: ScheduleSpec,
    block_templates: Mapping[str, BlockTemplate],
    layout: Optional[RankLayout],
    *,
    parallelism_mode: Any = None,
    tp_overlap: float = 0.0,
    tp_sp_overlap: float = 0.0,
    cp_overlap: float = 0.0,
    events_hook: Optional[Callable[[Any], None]] = None,
) -> Any:
    """Enumerate the schedule and return the fine proto-graph root (exposed
    separately for the differential tests). The overlap rewrite happens at
    the legacy pipeline position: build -> overlap -> propagate.

    ``events_hook`` (memory path, M3b) is invoked with the coarse events
    root between enumeration and expansion — the exact point where the
    legacy pipeline mutated the coarse graph before flattening. The memory
    dispatcher uses it to replay ``convert_comm_sizes_to_times`` for the
    analytical/hybrid modes, whose coarse comm edges carried converted
    durations into the legacy memory flatten.
    """
    from program.transforms import apply_overlap_to_fine_root

    events = build_pipeline_events(schedule_spec)
    if events_hook is not None:
        events_hook(events.root)
    expander = _FineExpander(schedule_spec, block_templates, layout)
    fine_root = expander.build(events.root)
    if fine_root is None:
        raise RuntimeError("Fine enumeration produced an empty graph")
    if parallelism_mode is not None:
        fine_root = apply_overlap_to_fine_root(
            fine_root,
            parallelism_mode,
            tp_overlap,
            tp_sp_overlap,
            cp_overlap,
        )
    expander.propagate_local_hw_ids(fine_root)
    return fine_root


def build_fine_program(
    schedule_spec: ScheduleSpec,
    block_templates: Mapping[str, BlockTemplate],
    layout: Optional[RankLayout],
    *,
    no_data_parallel: bool = False,
    dp_count: Optional[int] = None,
    optimize_2dmap: Optional[Dict[str, Any]] = None,
    gmap_workdir: Optional[str] = None,
    parallelism_mode: Any = None,
    tp_overlap: float = 0.0,
    tp_sp_overlap: float = 0.0,
    cp_overlap: float = 0.0,
    events_hook: Optional[Callable[[Any], None]] = None,
) -> Program:
    """Build the flattened FINE :class:`Program` directly from the schedule.

    ``dp_count`` is the emission replication degree (defaults to
    ``schedule_spec.dp``; the dispatcher passes 1 for inference).
    ``optimize_2dmap`` is the first-dimension SCOTCH config (was the legacy
    root's ``_optimize_2dmap`` attribute); ``gmap_workdir`` receives the
    ``first_dim_comm.*`` artifacts when it is set.

    ``parallelism_mode`` + the overlap fractions apply the TP/TP-SP/CP
    overlap rewrite at the proto level, exactly where the legacy pipeline
    applied it (flatten -> overlap -> propagate -> lower). See
    ``program/transforms.py`` for why the rewrite must run before lowering
    (the M3a-discovered Step-11 children-order dependence).
    """
    from program.legacy_lowering import lower_to_program  # shared ordering pass

    fine_root = build_fine_root(
        schedule_spec,
        block_templates,
        layout,
        parallelism_mode=parallelism_mode,
        tp_overlap=tp_overlap,
        tp_sp_overlap=tp_sp_overlap,
        cp_overlap=cp_overlap,
        events_hook=events_hook,
    )

    layout_descriptor = layout.descriptor() if layout is not None and layout.axis_order else None
    if layout_descriptor is not None:
        fine_root._astrasim_rank_layout = layout_descriptor
    if optimize_2dmap:
        fine_root._optimize_2dmap = dict(optimize_2dmap)

    effective_dp = int(dp_count) if dp_count is not None else int(schedule_spec.dp)
    program = lower_to_program(
        fine_root,
        effective_dp,
        layout_descriptor,
        gmap_workdir=gmap_workdir,
    )
    program.meta.label = "fine_no_dp" if no_data_parallel else "fine"
    if optimize_2dmap:
        program.meta.optimize_2dmap = dict(optimize_2dmap)
    # M3b: the memory replay (program/memory_sim.py) consumes the builder's
    # own proto graph — the event-loop replica needs the children-list
    # adjacency order, which the uid-ordered op list does not preserve (uid
    # order is the per-stage Kahn emission order). The proto root is a fine
    # builder product (no legacy Graph involved); collapsing it into the op
    # list is M8 work. ``granularity`` is the typed replacement of the legacy
    # ``_is_non_flattened`` name-sniffing guard in MemoryEstimator.
    program.meta.misc["granularity"] = "fine"
    program.meta.misc["fine_proto_root"] = fine_root
    program.meta.misc["fine_pp"] = int(schedule_spec.pp)
    return program
