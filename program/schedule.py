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

"""``ScheduleSpec`` + the GPipe event enumeration (M3a, DESIGN.md §2/§5 M3).

``ScheduleSpec`` distills everything the FINE pipeline builder needs from the
legacy pipeline ``Graph`` inputs (``comp_times`` / ``comm_metadata`` /
``misc_metadata`` / parallelism degrees) into a frozen, dependency-light
dataclass (no imports from ``train_timing`` or ``simulate_train_graph``).
It carries the two legacy scheduling policies as methods:

* :meth:`ScheduleSpec.should_emit_dp_comm` — verbatim port of
  ``Graph._should_emit_dp_comm`` (dp>1 gate, grad-accum nonfinal cycles emit
  only ``ga_required_every_cycle`` comms, ``last_mb`` mode emits at ``b==0``
  because backward flips microbatch order, ZeRO-3 forces every-mb);
* :func:`legacy_layers_per_stage` — the legacy remainder-first layer split
  (``base + 1`` for the first ``num_layers % pp`` stages).

``build_pipeline_events`` enumerates the SAME schedule
``Graph.construct_fwd_bwd_graph`` walks — fwd per (mb, layer) with GPipe
cross-microbatch stage dependencies, bwd reversed with recompute nodes when
enabled, embedding/softmax placement, the DP/ZeRO-2/3 collective lattice
gated by ``should_emit_dp_comm`` (incl. ``attach_parallel_edge`` with its
``skip_non_comm_children`` / ``skip_comm_children`` rules), EP sync, and the
per-stage optimizer tail — as a DAG of lightweight typed events
(:class:`ComputeEvent` / :class:`CommEvent`). Event ``children`` lists
reproduce the legacy ``add_child`` call order exactly: the FINE builder's
expansion traversal (and therefore its op-id assignment and the emission
order derived from it) depends on that order.

The event classes deliberately use the legacy attribute names (``hw_id``,
``local_hw_id``, ``micro_batch_index``, ...) so the flattening port in
:mod:`program.pipeline_fine` reads them exactly as the legacy code did.

M6 additions, for the hierarchical pipeline emission (which lowers the
coarse events through :func:`program.legacy_lowering.lower_to_program`):

* every event carries ``op_id``, stamped by ``build_pipeline_events`` in
  CREATION order — the exact sequence ``construct_fwd_bwd_graph`` assigned
  its ``op_id`` counter in (one increment per Node/Edge creation;
  Data_batch tokens never consumed an op_id). The lowering's per-stage Kahn
  toposort and Step-11 transfer replay key on it;
* :class:`ComputeEvent` ports the legacy ``Node`` duration normalization
  verbatim: writing a tuple stores a per-DP profile, ``duration`` reads
  index 0, and ``duration_profile`` exposes the tuple (or ``None``) — this
  is how the :func:`program.retime.apply_block_timings` events mirror
  reaches the lowering's per-DP ``ComputeOp.duration``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from timing_model import CollectiveType

from program.block import CommMeta, comm_metadata_from_legacy

try:  # MemKind lives in memory_estimation; keep schedule importable without it.
    from memory_estimation import MemKind
except Exception:  # pragma: no cover - defensive
    MemKind = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Legacy layer split helpers
# ---------------------------------------------------------------------------


def legacy_layers_per_stage(num_layers: int, pp: int) -> Tuple[int, ...]:
    """Port of ``Graph._compute_layers_per_stage``: remainder-first split."""
    stage_count = max(1, int(pp) if pp else 1)
    if num_layers <= 0:
        return tuple(0 for _ in range(stage_count))
    base = num_layers // stage_count
    remainder = num_layers % stage_count
    return tuple(base + (1 if stage_idx < remainder else 0) for stage_idx in range(stage_count))


def _layer_to_stage(layers_per_stage: Tuple[int, ...]) -> Tuple[int, ...]:
    """Port of ``Graph._build_layer_to_stage``."""
    mapping: List[int] = []
    for stage_idx, count in enumerate(layers_per_stage):
        if count <= 0:
            continue
        mapping.extend([stage_idx] * count)
    return tuple(mapping)


# ---------------------------------------------------------------------------
# ScheduleSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScheduleSpec:
    """The pipeline schedule, distilled from the legacy Graph inputs."""

    mb: int
    num_layers: int
    pp: int
    dp: int
    tp: int
    cp: int
    ep: int
    layers_per_stage: Tuple[int, ...]
    moe_layer_mask: Tuple[bool, ...]
    zero_stage: int
    dp_microbatch_mode: str
    grad_accum_cycle: str
    include_backward: bool
    include_optimizer: bool
    full_recomputation: bool
    pipeline_style_recompute: bool
    flattened_mode: bool
    model_type: str
    comp_times: Mapping[str, float]
    comm_metadata: Mapping[str, CommMeta]

    # -- constructors -----------------------------------------------------
    @classmethod
    def from_pipeline_graph(
        cls,
        pipeline_graph: Any,
        *,
        include_backward: bool,
        include_optimizer: bool = True,
    ) -> "ScheduleSpec":
        """Distill from a legacy pipeline ``Graph`` (duck-typed attr reads
        only; no simulate_train_graph import)."""
        misc = getattr(pipeline_graph, "misc_metadata", None) or {}
        comp_times_raw = getattr(pipeline_graph, "comp_times", None) or {}
        comp_times = {
            key: float(value)
            for key, value in comp_times_raw.items()
            if isinstance(value, (int, float))
        }
        num_layers = int(misc.get("num_layer", getattr(pipeline_graph, "num_layer", 0)) or 0)
        pp = int(getattr(pipeline_graph, "pp", 1) or 1)
        return cls(
            mb=int(misc.get("num_batch", getattr(pipeline_graph, "num_batch", 0)) or 0),
            num_layers=num_layers,
            pp=pp,
            dp=int(getattr(pipeline_graph, "dp", 1) or 1),
            tp=int(getattr(pipeline_graph, "tp", 1) or 1),
            cp=int(getattr(pipeline_graph, "cp", 1) or 1),
            ep=int(getattr(pipeline_graph, "ep", 1) or 1),
            layers_per_stage=legacy_layers_per_stage(num_layers, pp),
            moe_layer_mask=tuple(bool(v) for v in (misc.get("moe_layer_mask", []) or [])),
            zero_stage=int(misc.get("dp_zero_stage", 0) or 0),
            dp_microbatch_mode=str(misc.get("dp_microbatch_mode", "every_mb")).lower(),
            grad_accum_cycle=str(misc.get("grad_accum_cycle", "final") or "final").lower(),
            include_backward=bool(include_backward),
            include_optimizer=bool(include_optimizer),
            full_recomputation=bool(misc.get("full_recomputation", False)),
            pipeline_style_recompute=bool(misc.get("pipeline_style_recompute", False)),
            flattened_mode=bool(misc.get("flattened_mode", False)),
            model_type=str(misc.get("model_type", "") or "").lower(),
            comp_times=comp_times,
            comm_metadata=comm_metadata_from_legacy(
                getattr(pipeline_graph, "comm_metadata", None) or {}
            ),
        )

    # -- legacy policy ports ----------------------------------------------
    def time(self, key: str, default: float = 0.0) -> float:
        """Port of ``Graph._time``."""
        value = self.comp_times.get(key)
        return float(value) if value is not None else default

    def stage_for_layer(self, layer_idx: int) -> int:
        """Port of ``Graph._stage_for_layer``."""
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(
                f"Layer index {layer_idx} is out of bounds for {self.num_layers} layers."
            )
        mapping = _layer_to_stage(self.layers_per_stage)
        if mapping:
            return mapping[layer_idx]
        return 0

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Port of ``Graph._is_moe_layer``."""
        if not self.moe_layer_mask:
            return False
        if layer_idx < 0 or layer_idx >= len(self.moe_layer_mask):
            return False
        return bool(self.moe_layer_mask[layer_idx])

    def is_nonfinal_grad_accum_cycle(self) -> bool:
        return self.grad_accum_cycle == "nonfinal"

    def _dp_comm_required_every_cycle(self, comm_key: str) -> bool:
        meta = self.comm_metadata.get(comm_key)
        return bool(meta.ga_required_every_cycle) if meta is not None else False

    def should_emit_dp_comm(self, comm_key: str, micro_batch_idx: int) -> bool:
        """Verbatim port of ``Graph._should_emit_dp_comm``."""
        if self.dp <= 1:
            return False
        required_every_cycle = self._dp_comm_required_every_cycle(comm_key)
        if self.is_nonfinal_grad_accum_cycle():
            return required_every_cycle
        if required_every_cycle:
            return True
        if self.dp_microbatch_mode != "last_mb" or self.zero_stage >= 3:
            return True
        # Backward pass processes micro-batches in reverse, so b==0
        # corresponds to "last_mb" (legacy comment preserved).
        return int(micro_batch_idx) == 0

    @property
    def recompute_enabled(self) -> bool:
        """Port of the ``recompute_enabled`` predicate in
        ``construct_fwd_bwd_graph``."""
        return (
            self.include_backward
            and self.full_recomputation
            and (self.flattened_mode or self.pipeline_style_recompute)
        )

    @property
    def block_prefix(self) -> str:
        return "vit_block" if str(self.model_type).lower().startswith("vit") else "transformer_layer"

    def par_degree(self) -> int:
        return max(1, int(self.tp) * int(self.cp) * int(self.ep))

    def stage_min_layer(self) -> Dict[int, int]:
        """Lowest layer index per stage (the optimizer attach rule)."""
        stage_min: Dict[int, int] = {}
        for l in range(self.num_layers):
            stage = self.stage_for_layer(l)
            if stage not in stage_min:
                stage_min[stage] = l
            else:
                stage_min[stage] = min(stage_min[stage], l)
        return stage_min


# ---------------------------------------------------------------------------
# Schedule events (lightweight typed Node/Edge stand-ins)
# ---------------------------------------------------------------------------


class ComputeEvent:
    """A coarse compute event (legacy ``Node`` stand-in). Uses the legacy
    attribute names so the flattening port reads them verbatim. Duration
    storage ports the legacy ``Node`` property pair: scalar or per-DP tuple
    in ``_duration_data``, ``duration`` reads index 0, ``duration_profile``
    exposes the tuple (M6 — the lowering reads both, exactly like it read
    legacy Nodes)."""

    __slots__ = (
        "name",
        "hw_id",
        "op_id",
        "_duration_data",
        "fwd",
        "mem_kind",
        "recompute",
        "role",
        "micro_batch_index",
        "layer_index",
        "direction",
        "stage_id",
        "is_moe_layer",
        "children",
        "parents",
    )

    def __init__(
        self,
        name: str,
        hw_id: int,
        duration: float,
        *,
        fwd: bool = True,
        mem_kind: Any = None,
        recompute: bool = False,
        role: str = "generic",
    ) -> None:
        self.name = name
        self.hw_id = int(hw_id)
        self.op_id: Optional[int] = None
        self.duration = duration
        self.fwd = fwd
        self.mem_kind = mem_kind
        self.recompute = bool(recompute)
        self.role = role
        self.micro_batch_index: Optional[int] = None
        self.layer_index: Optional[int] = None
        self.direction: Optional[str] = None
        self.stage_id: Optional[int] = None
        self.is_moe_layer: bool = False
        self.children: List[Any] = []
        self.parents: List[Any] = []

    def add_child(self, obj: Any) -> None:
        self.children.append(obj)
        obj.parents.append(self)

    @staticmethod
    def _normalize_duration(value: Any) -> Any:
        """Verbatim legacy ``Node._normalize_duration``."""
        if isinstance(value, (list, tuple)):
            entries = tuple(float(v) for v in value)
            if not entries:
                raise ValueError("Duration tuple must contain at least one entry.")
            return entries
        return float(value)

    @property
    def duration(self) -> float:
        data = self._duration_data
        if isinstance(data, tuple):
            return data[0]
        return data

    @duration.setter
    def duration(self, value: Any) -> None:
        self._duration_data = self._normalize_duration(value)

    @property
    def duration_profile(self) -> Optional[Tuple[float, ...]]:
        data = self._duration_data
        if isinstance(data, tuple):
            return data
        return None

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"ComputeEvent({self.name},hw={self.hw_id})"


class CommEvent:
    """A coarse comm event (legacy ``Edge`` stand-in). ``comm_type`` is
    ``CollectiveType.PIPELINE`` with ``comm_size_bytes == 0`` for the
    same-stage zero-byte control edges."""

    __slots__ = (
        "name",
        "op_id",
        "duration",
        "is_dp",
        "comm_size_bytes",
        "comm_type",
        "participants",
        "comm_interconnect_type",
        "local_hw_id",
        "tp_shard",
        "comm_key",
        "zero3_offset_hint",
        "micro_batch_index",
        "layer_index",
        "direction",
        "stage_id",
        "children",
        "parents",
    )

    def __init__(
        self,
        name: str,
        *,
        comm_size_bytes: Any = 0,
        comm_type: Optional[CollectiveType] = None,
        participants: int = 1,
        comm_interconnect_type: Optional[str] = None,
        is_dp: bool = False,
        local_hw_id: Optional[int] = None,
        tp_shard: bool = False,
        comm_key: Optional[str] = None,
    ) -> None:
        self.name = name
        self.op_id: Optional[int] = None
        self.duration = 0
        self.is_dp = bool(is_dp)
        self.comm_size_bytes = comm_size_bytes
        if comm_type is not None and not isinstance(comm_type, CollectiveType):
            raise TypeError(
                f"CommEvent.comm_type must be a CollectiveType or None "
                f"(got {type(comm_type).__name__})"
            )
        self.comm_type = comm_type
        self.participants = participants
        self.comm_interconnect_type = comm_interconnect_type
        self.local_hw_id = local_hw_id
        self.tp_shard = bool(tp_shard)
        self.comm_key = comm_key
        #: "fwd"/"bwd" for ZeRO-3 transformer gathers — the typed stand-in
        #: for the legacy ``"bwd" in edge.name`` offset hack.
        self.zero3_offset_hint: Optional[str] = None
        self.micro_batch_index: Optional[int] = None
        self.layer_index: Optional[int] = None
        self.direction: Optional[str] = None
        self.stage_id: Optional[int] = None
        self.children: List[Any] = []
        self.parents: List[Any] = []

    def add_child(self, obj: Any) -> None:
        self.children.append(obj)
        obj.parents.append(self)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"CommEvent({self.name})"


@dataclass
class PipelineEvents:
    """The enumerated schedule: root plus handles used by tests/debugging."""

    root: Any
    embedding: List[ComputeEvent] = field(default_factory=list)
    softmax: List[ComputeEvent] = field(default_factory=list)
    layers: List[List[ComputeEvent]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# build_pipeline_events — port of Graph.construct_fwd_bwd_graph
# ---------------------------------------------------------------------------


def build_pipeline_events(spec: ScheduleSpec) -> PipelineEvents:
    """Enumerate the coarse schedule as an event DAG.

    Structural port of ``Graph.construct_fwd_bwd_graph`` (simulate_train_
    graph.py 371-1004) with ``Node``/``Edge`` replaced by events. Every
    ``add_child`` call is made in the same order as legacy so children-list
    order — which drives the expansion DFS — is identical. Data_batch nodes
    are omitted: legacy removed them from the graph before returning (and
    they never consumed an op_id — legacy tracked them by ``batch_id``).

    ``op_id`` stamping (M6): every event creation stamps the next counter
    value, reproducing the legacy ``op_id`` sequence exactly (legacy used
    the counter at creation and incremented right after, one per Node/Edge)
    — the hierarchical lowering's Kahn toposort keys and Step-11 transfer
    tags depend on these values. The FINE builder still assigns its own op
    ids during expansion, like the legacy flattener did.
    """

    B = spec.mb
    L = spec.num_layers

    _op_ids = itertools.count()

    def _stamp(event: Any) -> Any:
        """Assign the next legacy op_id (call at each creation site, in
        legacy creation order)."""
        event.op_id = next(_op_ids)
        return event

    def _comm_event(
        name: str,
        comm_key: str,
        *,
        is_dp: bool = False,
        local_hw_id: Optional[int] = None,
    ) -> CommEvent:
        """Port of ``Graph.create_comm_edge`` (KeyError on missing key; the
        ``local_comp_time`` zeroing means no local node is ever created)."""
        meta = spec.comm_metadata[comm_key]
        event = CommEvent(
            name,
            comm_size_bytes=meta.size_bytes,
            comm_type=meta.kind,
            participants=meta.participants,
            comm_interconnect_type=meta.interconnect,
            is_dp=is_dp,
            local_hw_id=int(local_hw_id) if local_hw_id is not None else None,
            tp_shard=meta.tp_shard,
            comm_key=comm_key,
        )
        return _stamp(event)

    embedding_node: List[ComputeEvent] = []
    softmax_node: List[ComputeEvent] = []
    transformer_nodes: List[List[ComputeEvent]] = [[] for _ in range(B)]
    layer_entry_nodes: List[List[List[ComputeEvent]]] = [[] for _ in range(B)]
    layer_exit_nodes: List[List[ComputeEvent]] = [[] for _ in range(B)]

    linear_softmax_f_time = spec.time("linear_softmax_f")
    linear_softmax_b_time = spec.time("linear_softmax_b")
    transformer_f_time = spec.time("transformer_f")
    transformer_b_time = spec.time("transformer_b")
    transformer_f_dense = spec.time("transformer_f_dense", transformer_f_time)
    transformer_f_moe = spec.time("transformer_f_moe", transformer_f_time)
    transformer_b_dense = spec.time("transformer_b_dense", transformer_b_time)
    transformer_b_moe = spec.time("transformer_b_moe", transformer_b_time)
    embedding_f_time = spec.time("embedding_f")
    embedding_b_time = spec.time("embedding_b")
    recompute_enabled = spec.recompute_enabled
    block_prefix = spec.block_prefix

    mem_embedding = getattr(MemKind, "EMBEDDING", None)
    mem_softmax = getattr(MemKind, "SOFTMAX", None)
    mem_transformer = getattr(MemKind, "TRANSFORMER", None)
    mem_optimizer = getattr(MemKind, "OPTIMIZER", None)

    def attach_parallel_edge(target, gather_edge, skip_non_comm_children=None, skip_comm_children=None):
        parents = list(getattr(target, "parents", []))
        for parent in parents:
            if hasattr(parent, "children") and gather_edge not in parent.children:
                parent.add_child(gather_edge)
        children = list(getattr(target, "children", []))
        for child in children:
            if skip_non_comm_children:
                if getattr(child, "comm_type", None) != CollectiveType.PIPELINE:
                    continue
            if skip_comm_children:
                if getattr(child, "comm_type", None) == CollectiveType.PIPELINE:
                    continue
            if gather_edge not in getattr(child, "parents", []):
                gather_edge.add_child(child)

    if spec.include_backward:
        embedding_node_b: List[Optional[ComputeEvent]] = [None for _ in range(B)]
        softmax_node_b: List[Optional[ComputeEvent]] = [None for _ in range(B)]
        transformer_nodes_b: List[List[Optional[ComputeEvent]]] = [
            [None for _ in range(L)] for _ in range(B)
        ]
        recompute_nodes_b: Optional[List[List[Optional[ComputeEvent]]]] = None
        if recompute_enabled:
            recompute_nodes_b = [[None for _ in range(L)] for _ in range(B)]

    # ---- forward chains per micro-batch ---------------------------------
    for b in range(B):
        linear_softmax = _stamp(ComputeEvent(
            f"linear_softmax{b}",
            spec.pp - 1,
            linear_softmax_f_time,
            mem_kind=mem_softmax,
            role="softmax",
        ))
        softmax_node.append(linear_softmax)
        emb = _stamp(ComputeEvent(
            f"embedding{b}",
            0,
            embedding_f_time,
            mem_kind=mem_embedding,
            role="embedding",
        ))
        embedding_node.append(emb)

        transformer_nodes[b] = []
        layer_entry_nodes[b] = []
        layer_exit_nodes[b] = []

        for l in range(L):
            hw_id = spec.stage_for_layer(l)
            is_moe_layer = spec.is_moe_layer(l)
            transformer_duration = transformer_f_moe if is_moe_layer else transformer_f_dense
            transformer_node = _stamp(ComputeEvent(
                f"{block_prefix}{l}",
                hw_id,
                transformer_duration,
                mem_kind=mem_transformer,
                role="layer",
            ))
            transformer_node.micro_batch_index = b
            transformer_node.layer_index = l
            transformer_node.direction = "forward"
            transformer_node.stage_id = hw_id
            transformer_node.is_moe_layer = is_moe_layer
            transformer_nodes[b].append(transformer_node)
            layer_entry_nodes[b].append([transformer_node])
            layer_exit_nodes[b].append(transformer_node)

        for l in range(1, L):
            prev_node = transformer_nodes[b][l - 1]
            curr_node = transformer_nodes[b][l]
            prev_exit = layer_exit_nodes[b][l - 1]
            curr_entries = layer_entry_nodes[b][l]

            if prev_node.hw_id == curr_node.hw_id:
                edge = _stamp(CommEvent("cross_layer", comm_type=CollectiveType.PIPELINE))
            else:
                edge = _comm_event("cross_layer", "cross_layer")

            prev_exit.add_child(edge)
            for entry in curr_entries:
                edge.add_child(entry)

        first_entries = layer_entry_nodes[b][0]
        primary_entry = first_entries[0]
        if primary_entry.hw_id == embedding_node[b].hw_id:
            edge = _stamp(CommEvent("Emb_node0", comm_type=CollectiveType.PIPELINE))
        else:
            edge = _comm_event("cross_layer", "cross_layer")
        embedding_node[b].add_child(edge)
        for entry in first_entries:
            edge.add_child(entry)

        last_exit = layer_exit_nodes[b][-1]
        if last_exit.hw_id == softmax_node[b].hw_id:
            node_softmax = _stamp(CommEvent("node_Softmax", comm_type=CollectiveType.PIPELINE))
        else:
            node_softmax = _comm_event("cross_layer", "cross_layer")
        last_exit.add_child(node_softmax)
        node_softmax.add_child(softmax_node[b])

    # ---- forward cross-microbatch GPipe dependencies --------------------
    for b in range(B - 1):
        gpu_index = 0
        first_transformer_layer = [0]
        for l in range(L - 1):
            if transformer_nodes[b][l].hw_id != transformer_nodes[b][l + 1].hw_id:
                first_transformer_layer.append(l + 1)
                gpu_index += 1
                if transformer_nodes[b][l].hw_id == 0:
                    layer_exit_nodes[b][l].add_child(embedding_node[b + 1])
                else:
                    next_entries = layer_entry_nodes[b + 1][first_transformer_layer[gpu_index - 1]]
                    for entry in next_entries:
                        layer_exit_nodes[b][l].add_child(entry)

        next_entries = layer_entry_nodes[b + 1][first_transformer_layer[-1]]
        for entry in next_entries:
            softmax_node[b].add_child(entry)

    if not spec.include_backward:
        return PipelineEvents(
            root=embedding_node[0],
            embedding=embedding_node,
            softmax=softmax_node,
            layers=transformer_nodes,
        )

    def _bwd_entry_node(mb_idx: int, layer_idx: int) -> Any:
        if recompute_enabled and recompute_nodes_b is not None:
            candidate = recompute_nodes_b[mb_idx][layer_idx]
            if candidate is not None:
                return candidate
        return transformer_nodes_b[mb_idx][layer_idx]

    def _bwd_exit_node(mb_idx: int, layer_idx: int) -> Any:
        return transformer_nodes_b[mb_idx][layer_idx]

    # ---- backward chains per micro-batch (reversed) ---------------------
    for b in reversed(range(B)):
        emb_b = _stamp(ComputeEvent(
            "embedding_b",
            0,
            embedding_b_time,
            fwd=False,
            mem_kind=mem_embedding,
            role="embedding_b",
        ))
        embedding_node_b[b] = emb_b
        linear_softmax_b = _stamp(ComputeEvent(
            "linear_softmax_b",
            spec.pp - 1,
            linear_softmax_b_time,
            fwd=False,
            mem_kind=mem_softmax,
            role="softmax_b",
        ))
        softmax_node_b[b] = linear_softmax_b
        softmax_node[b].add_child(linear_softmax_b)

        for l in reversed(range(L)):
            hw_id = spec.stage_for_layer(l)
            recompute_node = None
            if recompute_enabled and recompute_nodes_b is not None:
                recompute_node = _stamp(ComputeEvent(
                    f"{block_prefix}{l}_recompute",
                    hw_id,
                    transformer_f_moe if spec.is_moe_layer(l) else transformer_f_dense,
                    fwd=True,
                    mem_kind=mem_transformer,
                    recompute=True,
                    role="recompute",
                ))
                recompute_node.micro_batch_index = b
                recompute_node.layer_index = l
                recompute_node.direction = "forward"
                recompute_node.stage_id = hw_id
                recompute_node.is_moe_layer = spec.is_moe_layer(l)
                recompute_nodes_b[b][l] = recompute_node

            is_moe_layer = spec.is_moe_layer(l)
            transformer_node_b = _stamp(ComputeEvent(
                f"{block_prefix}{l}_b",
                hw_id,
                transformer_b_moe if is_moe_layer else transformer_b_dense,
                fwd=False,
                mem_kind=mem_transformer,
                role="layer",
            ))
            transformer_node_b.micro_batch_index = b
            transformer_node_b.layer_index = l
            transformer_node_b.direction = "backward"
            transformer_node_b.stage_id = hw_id
            transformer_node_b.is_moe_layer = is_moe_layer
            transformer_nodes_b[b][l] = transformer_node_b
            if recompute_node is not None:
                recompute_node.add_child(transformer_node_b)

        for l in reversed(range(1, L)):
            curr_node = _bwd_exit_node(b, l)
            next_ffn2 = _bwd_entry_node(b, l - 1)
            if curr_node.hw_id == next_ffn2.hw_id:
                edge = _stamp(CommEvent("cross_layer", comm_type=CollectiveType.PIPELINE))
            else:
                edge = _comm_event("cross_layer", "cross_layer")
            curr_node.add_child(edge)
            edge.add_child(next_ffn2)

        qkv_0_b = _bwd_exit_node(b, 0)
        if qkv_0_b.hw_id == emb_b.hw_id:
            edge = _stamp(CommEvent("Emb_node0", comm_type=CollectiveType.PIPELINE))
        else:
            edge = _comm_event("cross_layer", "cross_layer")
        qkv_0_b.add_child(edge)
        edge.add_child(emb_b)

        prev_layer_norm2 = _bwd_entry_node(b, L - 1)
        if prev_layer_norm2.hw_id == softmax_node_b[b].hw_id:
            layernorm_softmax = _stamp(CommEvent("layernorm2_Softmax", comm_type=CollectiveType.PIPELINE))
        else:
            layernorm_softmax = _comm_event("cross_layer", "cross_layer")
        softmax_node_b[b].add_child(layernorm_softmax)
        layernorm_softmax.add_child(prev_layer_norm2)

    # ---- ZeRO-3 forward entry gather ------------------------------------
    zero3_embedding_key = "zero3_embedding_gather"
    if (
        zero3_embedding_key in spec.comm_metadata
        and spec.should_emit_dp_comm(zero3_embedding_key, 0)
    ):
        zero3_entry_embedding_edge = _comm_event(
            f"{zero3_embedding_key}_b0_fwd_entry",
            zero3_embedding_key,
            is_dp=True,
            local_hw_id=embedding_node[0].hw_id,
        )
        root_forward_entry: Any = zero3_entry_embedding_edge
        zero3_entry_embedding_edge.add_child(embedding_node[0])
    else:
        root_forward_entry = embedding_node[0]

    # ---- DP / ZeRO collective lattice per micro-batch -------------------
    for b in range(B):
        if spec.dp > 1:
            zero2_embedding_key = "zero2_embedding_gather"
            zero2_transformer_key = "zero2_transformer_gather"
            zero2_softmax_key = "zero2_softmax_gather"
            postfix = "_all_reduce"
            if zero2_embedding_key in spec.comm_metadata or zero3_embedding_key in spec.comm_metadata:
                postfix = "_reduce_scatter"

            embedding_edge = None
            if spec.should_emit_dp_comm("embedding", b):
                embedding_edge = _comm_event("embedding", "embedding", is_dp=True)
                if (
                    zero2_embedding_key in spec.comm_metadata
                    and spec.should_emit_dp_comm(zero2_embedding_key, b)
                ):
                    gather_edge = _comm_event(
                        zero2_embedding_key,
                        zero2_embedding_key,
                        is_dp=True,
                        local_hw_id=embedding_node[b].hw_id,
                    )
                    embedding_edge.add_child(gather_edge)

            zero3_transformer_key = "zero3_transformer_gather"
            zero3_softmax_key = "zero3_softmax_gather"
            transformer_reducers: List[Optional[CommEvent]] = []
            for layer_idx in range(L):
                comm_key = "transformer_moe" if spec.is_moe_layer(layer_idx) else "transformer_dense"
                if not spec.should_emit_dp_comm(comm_key, b):
                    transformer_reducers.append(None)
                    continue
                reducer = _comm_event(
                    f"transformer_b{b}_layer{layer_idx}" + postfix,
                    comm_key,
                    is_dp=True,
                    local_hw_id=_bwd_exit_node(b, layer_idx).hw_id,
                )
                transformer_reducers.append(reducer)

                if (
                    zero2_transformer_key in spec.comm_metadata
                    and spec.should_emit_dp_comm(zero2_transformer_key, b)
                ):
                    gather_edge = _comm_event(
                        zero2_transformer_key,
                        zero2_transformer_key,
                        is_dp=True,
                        local_hw_id=_bwd_exit_node(b, layer_idx).hw_id,
                    )
                    reducer.add_child(gather_edge)

            if (
                zero3_transformer_key in spec.comm_metadata
                and spec.should_emit_dp_comm(zero3_transformer_key, b)
            ):
                zero3_embedding_gather_edge = None
                if (
                    b < B - 1
                    and spec.should_emit_dp_comm(zero3_embedding_key, b + 1)
                ):
                    zero3_embedding_gather_edge = _comm_event(
                        f"{zero3_embedding_key}_b{b+1}_fwd",
                        zero3_embedding_key,
                        is_dp=True,
                        local_hw_id=embedding_node[b].hw_id,
                    )

                gather_edge = _comm_event(
                    f"{zero3_transformer_key}_b{b}_layer0_fwd",
                    zero3_transformer_key,
                    is_dp=True,
                    local_hw_id=embedding_node[b].hw_id,
                )
                gather_edge.zero3_offset_hint = "fwd"
                attach_parallel_edge(embedding_node[b], gather_edge)
                for layer_idx in range(1, L):
                    host = transformer_nodes[b][layer_idx - 1]
                    target = transformer_nodes[b][layer_idx]
                    gather_edge = _comm_event(
                        f"{zero3_transformer_key}_b{b}_layer{layer_idx}_fwd",
                        zero3_transformer_key,
                        is_dp=True,
                        local_hw_id=target.hw_id,
                    )
                    gather_edge.zero3_offset_hint = "fwd"
                    if host.hw_id == target.hw_id:
                        attach_parallel_edge(host, gather_edge)
                    else:
                        # Cross-device: dependencies only apply to the
                        # "cross_layer" comm; the next batch's embedding
                        # gather rides the non-comm children (legacy rule).
                        attach_parallel_edge(host, gather_edge, skip_non_comm_children=True)
                        if zero3_embedding_gather_edge:
                            attach_parallel_edge(
                                host, zero3_embedding_gather_edge, skip_comm_children=True
                            )

            softmax_edge = None
            if spec.should_emit_dp_comm("softmax", b):
                softmax_edge = _comm_event(
                    "softmax" + postfix,
                    "softmax",
                    is_dp=True,
                    local_hw_id=softmax_node[b].hw_id,
                )
                if (
                    zero2_softmax_key in spec.comm_metadata
                    and spec.should_emit_dp_comm(zero2_softmax_key, b)
                ):
                    gather_edge = _comm_event(
                        zero2_softmax_key,
                        zero2_softmax_key,
                        is_dp=True,
                        local_hw_id=softmax_node[b].hw_id,
                    )
                    softmax_edge.add_child(gather_edge)

            if (
                zero3_softmax_key in spec.comm_metadata
                and L > 0
                and spec.should_emit_dp_comm(zero3_softmax_key, b)
            ):
                host = transformer_nodes[b][L - 1]
                gather_edge = _comm_event(
                    f"{zero3_softmax_key}_b{b}_fwd",
                    zero3_softmax_key,
                    is_dp=True,
                    local_hw_id=host.hw_id,
                )
                attach_parallel_edge(host, gather_edge)

            if softmax_edge is not None:
                softmax_node_b[b].add_child(softmax_edge)
            if embedding_edge is not None:
                embedding_node_b[b].add_child(embedding_edge)
            for layer_idx, reducer in enumerate(transformer_reducers):
                if reducer is not None:
                    _bwd_exit_node(b, layer_idx).add_child(reducer)

        if spec.ep > 1 and spec.dp > 1 and not spec.is_nonfinal_grad_accum_cycle():
            apply_ep_all_mbs = spec.dp_microbatch_mode != "last_mb" or spec.zero_stage >= 3
            if apply_ep_all_mbs or b == 0:
                for layer_idx in range(L):
                    comm_key = (
                        "transformer_moe_ep_sync"
                        if spec.is_moe_layer(layer_idx)
                        else "transformer_dense_ep_sync"
                    )
                    if comm_key not in spec.comm_metadata:
                        continue
                    ep_edge = _comm_event(
                        f"{comm_key}_b{b}_layer{layer_idx}",
                        comm_key,
                        is_dp=False,
                        local_hw_id=_bwd_exit_node(b, layer_idx).hw_id,
                    )
                    _bwd_exit_node(b, layer_idx).add_child(ep_edge)

    # ---- backward cross-microbatch GPipe dependencies -------------------
    last_transformer_layer = [-1] * spec.pp
    first_transformer_layer_arr = [-1] * spec.pp
    gpu_index = spec.pp - 1
    for l in range(L - 1, 0, -1):
        if _bwd_exit_node(0, l).hw_id != _bwd_exit_node(0, l - 1).hw_id:
            first_transformer_layer_arr[gpu_index - 1] = l - 1
            last_transformer_layer[gpu_index] = l
            gpu_index -= 1

    for b in range(B - 1, 0, -1):
        gpu_index = spec.pp - 1
        for l in range(L - 1, 0, -1):
            if _bwd_exit_node(b, l).hw_id != _bwd_exit_node(b, l - 1).hw_id:
                if _bwd_exit_node(b, l).hw_id == spec.pp - 1:
                    _bwd_exit_node(b, l).add_child(softmax_node_b[b - 1])
                else:
                    _bwd_exit_node(b, l).add_child(
                        _bwd_entry_node(b - 1, first_transformer_layer_arr[gpu_index])
                    )
                gpu_index -= 1
        embedding_node_b[b].add_child(_bwd_entry_node(b - 1, first_transformer_layer_arr[0]))

    # ---- ZeRO-3 backward gathers ----------------------------------------
    zero3_embedding_key = "zero3_embedding_gather"
    zero3_transformer_key = "zero3_transformer_gather"
    zero3_softmax_key = "zero3_softmax_gather"

    zero3_softmax_bwd_entry = None
    if (
        zero3_softmax_key in spec.comm_metadata
        and L > 0
        and spec.should_emit_dp_comm(zero3_softmax_key, 0)
    ):
        zero3_softmax_bwd_entry = _comm_event(
            f"{zero3_softmax_key}_b0_bwd_entry",
            zero3_softmax_key,
            is_dp=True,
            local_hw_id=softmax_node_b[0].hw_id,
        )
        attach_parallel_edge(softmax_node[-1], zero3_softmax_bwd_entry)
    for b in range(B):
        zero3_softmax_bwd_edge = None
        if b > 0:
            if (
                zero3_softmax_key in spec.comm_metadata
                and spec.should_emit_dp_comm(zero3_softmax_key, b)
            ):
                host = softmax_node[b]
                gather_edge = _comm_event(
                    f"{zero3_softmax_key}_b{b}_bwd",
                    zero3_softmax_key,
                    is_dp=True,
                    local_hw_id=host.hw_id,
                )
                zero3_softmax_bwd_edge = gather_edge

        if (
            zero3_transformer_key in spec.comm_metadata
            and L > 0
            and spec.should_emit_dp_comm(zero3_transformer_key, b)
        ):
            for layer_idx in reversed(range(L)):
                host = softmax_node_b[b] if layer_idx == L - 1 else _bwd_exit_node(b, layer_idx + 1)
                target = _bwd_entry_node(b, layer_idx)
                gather_edge = _comm_event(
                    f"{zero3_transformer_key}_b{b}_layer{layer_idx}_bwd",
                    zero3_transformer_key,
                    is_dp=True,
                    local_hw_id=target.hw_id,
                )
                gather_edge.zero3_offset_hint = "bwd"
                if host.hw_id == target.hw_id:
                    attach_parallel_edge(host, gather_edge)
                else:
                    attach_parallel_edge(host, gather_edge, skip_non_comm_children=True)
                    if zero3_softmax_bwd_edge:
                        attach_parallel_edge(host, zero3_softmax_bwd_edge, skip_comm_children=True)

        if (
            zero3_embedding_key in spec.comm_metadata
            and L > 0
            and spec.should_emit_dp_comm(zero3_embedding_key, b)
        ):
            host = _bwd_exit_node(b, 0)
            gather_edge = _comm_event(
                f"{zero3_embedding_key}_b{b}_bwd",
                zero3_embedding_key,
                is_dp=True,
                local_hw_id=host.hw_id,
            )
            attach_parallel_edge(host, gather_edge)

    # ---- optimizer tail per stage ---------------------------------------
    if spec.include_backward and spec.include_optimizer:
        optimizer_time = spec.time("optimizer")
        if optimizer_time > 0:
            stage_min_layer = spec.stage_min_layer()
            # bwd flips mbs, so we attach to the first one, not the last.
            for stage in range(spec.pp):
                last_node = None
                if stage == 0:
                    last_node = embedding_node_b[0]
                else:
                    min_l = stage_min_layer.get(stage)
                    if min_l is not None:
                        last_node = _bwd_exit_node(0, min_l)

                if last_node:
                    opt_node = _stamp(ComputeEvent(
                        f"optimizer_stage{stage}",
                        stage,
                        optimizer_time,
                        fwd=False,
                        mem_kind=mem_optimizer,
                        role="optimizer",
                    ))
                    last_node.add_child(opt_node)

    return PipelineEvents(
        root=root_forward_entry,
        embedding=embedding_node,
        softmax=softmax_node,
        layers=transformer_nodes,
    )
