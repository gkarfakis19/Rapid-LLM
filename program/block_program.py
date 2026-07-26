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

"""``build_block_program`` — BLOCK-granularity transformer programs (M4).

Replaces ``simulate_train_graph.Graph.construct_transformer_graph`` for the
hybrid / hierarchical transformer AstraSim runs: one transformer layer's
per-(tp, cp, ep)-rank chains, one direction, over the transformer sublayout
(``hw_id = tp + cp*tp_degree + ep*tp_degree*cp_degree``), emitted with
``dp_count = 1`` (the legacy ``dp_override=1`` semantics).

Like M3a's :mod:`program.pipeline_fine`, this builder owns only the block
STRUCTURE: it enumerates the exact legacy ``Node``/``Edge`` graph over the
fine proto elements (:class:`~program.pipeline_fine.FineNode` /
:class:`~program.pipeline_fine.FineEdge`) in the exact legacy creation order
— the ``ep -> cp -> tp`` triple loop, per-rank pre-comm / GEMM / post-comm
sequence, and the MoE hot/cold join + residual wiring — and hands the proto
graph to the shared :func:`program.legacy_lowering.lower_to_program` pass,
so the ``legacy`` id-policy emission is byte-identical to lowering the
legacy graph (verified by the M4 differential across every hybrid /
hierarchical golden spec before the legacy constructor was deleted).

Reproduced legacy rules (each pinned by the hier/hybrid golden bundles):

* the ``Data_batch`` root fans out to every rank's chain head; it is not a
  compute node, so lowering turns the fan-out into "ops with no deps";
* per-rank serial chains: ``pre`` comm keys chained before the GEMM node,
  ``post`` keys after, in ``comm_keys`` order (placement is read from the
  comm metadata, default "post"); ``param_gather`` is set on each rank's
  first-processed GEMM (which for backward is the LAST template entry —
  the loop enumerates the reversed entry list);
* parallel comm groups (same ``parallel_group``) are grouped in
  first-encounter order; non-MoE groups still chain serially (legacy);
* MoE parallel post groups: base all-to-all collective per rank, a
  zero-duration local join compute per rank, and cold->hot residual
  PIPELINE edges (which lower to cross-device TransferOps). The hot rank is
  ``(tp_idx, cp_idx, ep=0)`` for routing mode "ep" and ``(0, cp_idx, 0)``
  for "tp_ep"; hot joins must be constructed before their cold ranks (the
  ``ep`` outer loop guarantees it — the legacy error is preserved);
* op_ids start at 0 and advance once per created element — they key the
  per-stage emission toposort and the ET node name suffixes.

Overlap transforms are applied to the proto graph BEFORE lowering via
:func:`program.transforms.apply_overlap_to_fine_root` — the same point in
the flow where ``train_timing._prepare_execution_graphs`` applied them to
the legacy transformer roots (and for the same reason the FINE builder
applies them proto-level: the rewrite's children-list reordering is
load-bearing for lowering order — see ``program/transforms.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from memory_estimation import mem_kind_from_op_name

from program.block import BlockTemplate, CommMeta
from program.ir import Program
from program.layout import RankLayout
from program.pipeline_fine import FineEdge, FineNode


class BlockRoot:
    """Stand-in for the legacy ``Data_batch("transformer_root", 0, 0)``:
    a non-compute fan-out object (no ``hw_id``, no ``comm_type``) that the
    lowering walks through, leaving the chain heads dependency-free."""

    def __init__(self, name: str = "transformer_root") -> None:
        self.name = name
        self.children: List[Any] = []
        self.parents: List[Any] = []

    def add_child(self, obj: Any) -> None:
        self.children.append(obj)
        obj.parents.append(self)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"BlockRoot({self.name})"


@dataclass
class TransformerBlockSpec:
    """The transformer-block bundle ``train_timing._prepare_execution_graphs``
    hands to the dispatcher (replacing the legacy transformer ``Graph`` +
    fwd/bwd root six-tuple).

    ``tp``/``cp``/``ep`` are the cluster degrees the legacy transformer
    ``Graph`` was constructed with (``ep`` is the *graph* ep: ``time_calc.ep``
    when MoE is active, else 1). ``include_backward`` mirrors the legacy
    "backward root was built" condition (False for inference prefill/decode).
    """

    dense: BlockTemplate
    moe: Optional[BlockTemplate]
    tp: int
    cp: int
    ep: int
    include_backward: bool


# ---------------------------------------------------------------------------
# Proto-graph enumeration (the construct_transformer_graph port)
# ---------------------------------------------------------------------------


def build_block_root(
    template: BlockTemplate,
    direction: str = "both",
    *,
    tp: int = 1,
    cp: int = 1,
    ep: int = 1,
) -> BlockRoot:
    """Enumerate one transformer block's proto graph in the exact legacy
    ``construct_transformer_graph`` creation order (op_ids from 0)."""

    gemm_entries = template.entries
    if not gemm_entries:
        raise ValueError("Transformer GEMM times not provided")

    comm_metadata: Mapping[str, CommMeta] = template.comm_metadata

    tp_degree = max(1, int(tp))
    cp_degree = max(1, int(cp))
    ep_degree = max(1, int(ep))

    root = BlockRoot("transformer_root")
    op_id = 0

    def _comm_meta(comm_key: str) -> CommMeta:
        meta = comm_metadata.get(comm_key)
        if meta is None:
            raise KeyError(f"Missing transformer comm metadata for key '{comm_key}'")
        return meta

    def _split_comm_keys(comm_keys: Optional[List[str]]) -> Tuple[List[str], List[str]]:
        pre_keys: List[str] = []
        post_keys: List[str] = []
        if not comm_keys:
            return pre_keys, post_keys
        for key in comm_keys:
            # Legacy read: comm_metadata.get(key, {}).get("placement", "post")
            # (a missing entry defaults to "post" here and only errors when
            # the edge is actually created).
            meta = comm_metadata.get(key)
            placement = meta.placement if meta is not None else "post"
            if placement == "pre":
                pre_keys.append(key)
            elif placement == "post":
                post_keys.append(key)
            else:
                raise ValueError(
                    f"Unsupported comm placement '{placement}' for key '{key}' "
                    "(expected 'pre' or 'post')"
                )
        return pre_keys, post_keys

    def _comm_parallel_groups(comm_keys: List[str]) -> List[List[str]]:
        grouped: List[List[str]] = []
        parallel_index: Dict[str, int] = {}
        for comm_key in comm_keys:
            meta = _comm_meta(comm_key)
            parallel_group = meta.parallel_group
            if not parallel_group:
                grouped.append([comm_key])
                continue
            existing_idx = parallel_index.get(parallel_group)
            if existing_idx is None:
                parallel_index[parallel_group] = len(grouped)
                grouped.append([comm_key])
            else:
                grouped[existing_idx].append(comm_key)
        return grouped

    def _rank_id(tp_idx: int, cp_idx: int, ep_idx: int) -> int:
        return tp_idx + cp_idx * tp_degree + ep_idx * tp_degree * cp_degree

    def _create_rank_comm_edge(
        *,
        comm_key: str,
        rank: int,
        tp_idx: int,
        cp_idx: int,
        ep_idx: int,
        name: Optional[str] = None,
    ) -> FineEdge:
        nonlocal op_id
        meta = _comm_meta(comm_key)
        comm_edge = FineEdge(
            name=name or comm_key,
            op_id=op_id,
            duration=0,  # converted downstream, exactly like legacy create_comm_edge
            is_dp=False,
            comm_size_bytes=meta.size_bytes,
            comm_type=meta.kind,
            participants=meta.participants,
            comm_interconnect_type=meta.interconnect,
        )
        comm_edge.local_hw_id = int(rank)
        # Legacy create_comm_edge: local_comp_time is force-zeroed (the
        # "match astrasim" TODO), so no local compute node is ever created.
        if meta.tp_shard:
            comm_edge.tp_shard = True
        comm_edge.tp_rank = tp_idx
        comm_edge.cp_rank = cp_idx
        comm_edge.ep_rank = ep_idx
        op_id += 1
        return comm_edge

    def _moe_hot_rank(tp_idx: int, cp_idx: int, routing_mode: str) -> int:
        if routing_mode == "ep":
            return _rank_id(tp_idx, cp_idx, 0)
        if routing_mode == "tp_ep":
            return _rank_id(0, cp_idx, 0)
        raise ValueError(f"Unsupported MoE routing mode '{routing_mode}'")

    def _moe_parallel_token(tp_idx: int, cp_idx: int, routing_mode: str) -> Tuple[Any, ...]:
        if routing_mode == "ep":
            return (routing_mode, cp_idx, tp_idx)
        if routing_mode == "tp_ep":
            return (routing_mode, cp_idx)
        raise ValueError(f"Unsupported MoE routing mode '{routing_mode}'")

    moe_parallel_joins: Dict[Tuple[str, Tuple[Any, ...]], FineNode] = {}

    def _make_local_join_node(
        *,
        name: str,
        rank: int,
        tp_idx: int,
        cp_idx: int,
        ep_idx: int,
        direction_name: str,
    ) -> FineNode:
        nonlocal op_id
        join = FineNode(
            name=name,
            op_id=op_id,
            hw_id=rank,
            duration=0.0,
            fwd=(direction_name == "forward"),
            mem_kind=None,
        )
        join.tp_rank = tp_idx
        join.cp_rank = cp_idx
        join.ep_rank = ep_idx
        op_id += 1
        return join

    def _attach_serial_comm_chain(
        previous_obj: Any,
        comm_keys: List[str],
        *,
        rank: int,
        tp_idx: int,
        cp_idx: int,
        ep_idx: int,
    ) -> Any:
        current = previous_obj
        for comm_key in comm_keys:
            comm_edge = _create_rank_comm_edge(
                comm_key=comm_key,
                rank=rank,
                tp_idx=tp_idx,
                cp_idx=cp_idx,
                ep_idx=ep_idx,
            )
            current.add_child(comm_edge)
            current = comm_edge
        return current

    def _attach_moe_parallel_post_group(
        previous_obj: Any,
        comm_keys: List[str],
        *,
        rank: int,
        tp_idx: int,
        cp_idx: int,
        ep_idx: int,
        direction_name: str,
    ) -> Any:
        base_key: Optional[str] = None
        residual_key: Optional[str] = None
        parallel_group: Optional[str] = None
        routing_mode: Optional[str] = None

        for comm_key in comm_keys:
            meta = _comm_meta(comm_key)
            component = meta.moe_component
            parallel_group = parallel_group or meta.parallel_group
            routing_mode = routing_mode or meta.moe_routing_mode
            if component == "base_all_to_all":
                base_key = comm_key
            elif component == "residual_p2p":
                residual_key = comm_key
            else:
                raise ValueError(
                    f"MoE parallel comm group contains unsupported component '{component}' "
                    f"for key '{comm_key}'"
                )

        if not parallel_group or not routing_mode:
            raise ValueError(
                f"MoE parallel comm group {comm_keys} is missing required metadata "
                "(parallel_group / moe_routing_mode)."
            )

        base_edge: Optional[FineEdge] = None
        if base_key is not None:
            base_edge = _create_rank_comm_edge(
                comm_key=base_key,
                rank=rank,
                tp_idx=tp_idx,
                cp_idx=cp_idx,
                ep_idx=ep_idx,
            )
            previous_obj.add_child(base_edge)

        hot_rank = _moe_hot_rank(tp_idx, cp_idx, routing_mode)
        group_token = (parallel_group, _moe_parallel_token(tp_idx, cp_idx, routing_mode))

        join = _make_local_join_node(
            name=f"{parallel_group}_join_rank{rank}",
            rank=rank,
            tp_idx=tp_idx,
            cp_idx=cp_idx,
            ep_idx=ep_idx,
            direction_name=direction_name,
        )
        if base_edge is not None:
            base_edge.add_child(join)
        else:
            previous_obj.add_child(join)

        if rank == hot_rank:
            moe_parallel_joins[group_token] = join
            return join

        if residual_key is None:
            return join

        hot_join = moe_parallel_joins.get(group_token)
        if hot_join is None:
            raise ValueError(
                f"Missing hot-rank MoE join node for parallel group '{parallel_group}'. "
                "The simulator assumes the first rank in the routing group is hot "
                "and must be constructed before colder ranks."
            )

        residual_edge = _create_rank_comm_edge(
            comm_key=residual_key,
            rank=rank,
            tp_idx=tp_idx,
            cp_idx=cp_idx,
            ep_idx=ep_idx,
            name=f"{residual_key}_rank{rank}_to_hot{hot_rank}",
        )
        previous_obj.add_child(residual_edge)
        residual_edge.add_child(join)
        residual_edge.add_child(hot_join)
        return join

    for ep_idx in range(ep_degree):
        for cp_idx in range(cp_degree):
            for tp_idx in range(tp_degree):
                rank = _rank_id(tp_idx, cp_idx, ep_idx)
                previous: Any = root

                if direction in {"forward", "both"}:
                    for idx, entry in enumerate(gemm_entries):
                        entry_name = entry.name
                        forward_cfg = entry.forward
                        fwd_duration = forward_cfg.duration
                        if fwd_duration is None:
                            raise ValueError("Transformer GEMM entry missing forward duration")

                        comm_keys = list(forward_cfg.comm_keys)
                        pre_keys, post_keys = _split_comm_keys(comm_keys)

                        previous = _attach_serial_comm_chain(
                            previous,
                            pre_keys,
                            rank=rank,
                            tp_idx=tp_idx,
                            cp_idx=cp_idx,
                            ep_idx=ep_idx,
                        )

                        node = FineNode(
                            name=f"{entry_name}_fwd_rank{rank}",
                            op_id=op_id,
                            hw_id=rank,
                            duration=fwd_duration,
                            fwd=True,
                            mem_kind=mem_kind_from_op_name(entry_name),
                            param_gather=(idx == 0),
                        )
                        node.tp_rank = tp_idx
                        node.cp_rank = cp_idx
                        node.ep_rank = ep_idx
                        op_id += 1
                        previous.add_child(node)
                        previous = node
                        for key_group in _comm_parallel_groups(post_keys):
                            metadata = _comm_meta(key_group[0])
                            if metadata.parallel_group and metadata.moe_component:
                                previous = _attach_moe_parallel_post_group(
                                    previous,
                                    key_group,
                                    rank=rank,
                                    tp_idx=tp_idx,
                                    cp_idx=cp_idx,
                                    ep_idx=ep_idx,
                                    direction_name="forward",
                                )
                            else:
                                previous = _attach_serial_comm_chain(
                                    previous,
                                    key_group,
                                    rank=rank,
                                    tp_idx=tp_idx,
                                    cp_idx=cp_idx,
                                    ep_idx=ep_idx,
                                )

                if direction in {"backward", "both"}:
                    for idx, entry in enumerate(reversed(gemm_entries)):
                        entry_name = entry.name
                        backward_cfg = entry.backward
                        bwd_duration = backward_cfg.duration
                        if bwd_duration is None:
                            raise ValueError("Transformer GEMM entry missing backward duration")

                        comm_keys = list(backward_cfg.comm_keys)
                        pre_keys, post_keys = _split_comm_keys(comm_keys)

                        previous = _attach_serial_comm_chain(
                            previous,
                            pre_keys,
                            rank=rank,
                            tp_idx=tp_idx,
                            cp_idx=cp_idx,
                            ep_idx=ep_idx,
                        )

                        node = FineNode(
                            name=f"{entry_name}_bwd_rank{rank}",
                            op_id=op_id,
                            hw_id=rank,
                            duration=bwd_duration,
                            fwd=False,
                            mem_kind=mem_kind_from_op_name(entry_name),
                            param_gather=(idx == 0),
                        )
                        node.tp_rank = tp_idx
                        node.cp_rank = cp_idx
                        node.ep_rank = ep_idx
                        op_id += 1
                        previous.add_child(node)
                        previous = node

                        for key_group in _comm_parallel_groups(post_keys):
                            metadata = _comm_meta(key_group[0])
                            if metadata.parallel_group and metadata.moe_component:
                                previous = _attach_moe_parallel_post_group(
                                    previous,
                                    key_group,
                                    rank=rank,
                                    tp_idx=tp_idx,
                                    cp_idx=cp_idx,
                                    ep_idx=ep_idx,
                                    direction_name="backward",
                                )
                            else:
                                previous = _attach_serial_comm_chain(
                                    previous,
                                    key_group,
                                    rank=rank,
                                    tp_idx=tp_idx,
                                    cp_idx=cp_idx,
                                    ep_idx=ep_idx,
                                )

    return root


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _normalize_layout(
    layout_tcep: Union[RankLayout, Mapping[str, Any], None],
) -> Optional[Dict[str, Any]]:
    """Accept a transformer-sublayout ``RankLayout`` or a legacy descriptor
    dict (the dispatcher's ``_transformer_rank_layout``); return the legacy
    descriptor or ``None`` when no axes are present."""
    if layout_tcep is None:
        return None
    if isinstance(layout_tcep, RankLayout):
        if not layout_tcep.axis_order:
            return None
        return layout_tcep.descriptor()
    descriptor = dict(layout_tcep)
    if not descriptor.get("axis_order"):
        return None
    return descriptor


def build_block_program(
    block_template: BlockTemplate,
    direction: str,
    layout_tcep: Union[RankLayout, Mapping[str, Any], None] = None,
    *,
    tp: Optional[int] = None,
    cp: Optional[int] = None,
    ep: Optional[int] = None,
    parallelism_mode: Any = None,
    tp_overlap: float = 0.0,
    tp_sp_overlap: float = 0.0,
    cp_overlap: float = 0.0,
    label: Optional[str] = None,
) -> Program:
    """Build one direction's BLOCK :class:`Program` for a transformer layer.

    ``layout_tcep`` is the transformer sublayout (axes ``tp``/``cp``/``ep``),
    as a :class:`RankLayout` or the legacy descriptor dict; it becomes the
    Program's layout exactly as the legacy path attached it to the root as
    ``_astrasim_rank_layout``. ``tp``/``cp``/``ep`` are the cluster degrees
    the legacy transformer ``Graph`` carried (defaulting to the layout's
    axis sizes); note MoE runs pass the *graph* ep here — see
    :class:`TransformerBlockSpec`.

    The program is emitted with ``dp_count=1`` (legacy ``dp_override=1``:
    transformer runs are single-replica measurements; the per-dp fault
    variants change only the AstraSim network configs, never the bundle).
    """
    # Lazy: legacy_lowering imports astrasim_lib.gmap (heavy / cyclic with
    # the astrasim_lib package init), same pattern as pipeline_fine.
    from program.legacy_lowering import lower_to_program
    from program.transforms import apply_overlap_to_fine_root

    descriptor = _normalize_layout(layout_tcep)
    axis_sizes = (descriptor or {}).get("axis_sizes", {})
    tp_degree = int(tp) if tp is not None else int(axis_sizes.get("tp", 1) or 1)
    cp_degree = int(cp) if cp is not None else int(axis_sizes.get("cp", 1) or 1)
    ep_degree = int(ep) if ep is not None else int(axis_sizes.get("ep", 1) or 1)

    root = build_block_root(
        block_template,
        direction,
        tp=tp_degree,
        cp=cp_degree,
        ep=ep_degree,
    )

    # Same overlap rewrite, same position in the flow as the legacy path
    # (train_timing applied it to the fresh transformer roots, i.e. before
    # any lowering).
    if parallelism_mode is not None:
        root = apply_overlap_to_fine_root(
            root,
            parallelism_mode,
            tp_overlap,
            tp_sp_overlap,
            cp_overlap,
        )

    if descriptor is not None:
        root._astrasim_rank_layout = descriptor

    program = lower_to_program(root, 1, descriptor)
    program.meta.label = label if label is not None else f"block_{direction}"
    program.meta.misc["granularity"] = "block"
    return program
