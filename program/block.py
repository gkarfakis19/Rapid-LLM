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

"""``BlockTemplate`` / ``CommMeta`` — the typed transformer block template
(M3a, DESIGN.md §2 / design_A §1).

A :class:`BlockTemplate` is the typed port of the two inputs
``PipelineGraphFlattener`` consumed from a transformer ``Graph``:

* ``transformer_graph.transformer_cfg["gemms"]`` — the per-GEMM entry list
  (name + per-direction duration + comm_keys), and
* ``transformer_graph.comm_metadata`` — the CommSpec registration output of
  ``train_timing._prepare_execution_graphs`` (size / CollectiveType /
  participants / interconnect / placement / MoE grouping / tp_shard).

The flattener never read the transformer graph's *node structure* (verified
in docs/rewrite/panel — it only consumed the template + metadata), so these
two mappings are the complete block description. Dense and MoE variants are
two templates, and ONE expander (:class:`program.placement.BlockExpander`)
realizes both at every granularity: the MoE hot/cold joins + residual
transfers the deleted ``construct_transformer_graph`` used to build are now
produced by the FLAT (flattened) build as well as the BLOCK one, which is
what let the flattened path stop rejecting MoE (``ext_moe_flat.md`` P8).

``CommMeta`` is shared with :mod:`program.schedule` for the pipeline graph's
comm metadata (``cross_layer`` / dp reducers / ZeRO gathers / EP sync).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from timing_model import CollectiveType


@dataclass(frozen=True)
class CommMeta:
    """Typed port of one ``comm_metadata`` entry.

    ``size_bytes`` is kept as the RAW legacy value (may be a float for the
    dp grad reducers: ``get_data_parallel_reduction_sizes`` returns floats);
    the ``int()`` truncation happens exactly where legacy applies it (the
    lowering's ``int(edge.comm_size_bytes)``).
    """

    name: str
    size_bytes: Any
    kind: CollectiveType
    participants: int
    interconnect: Optional[str]
    local_comp_time: float = 0.0
    tp_shard: bool = False
    #: dp-comm grad-accumulation flag (``ga_required_every_cycle``).
    ga_required_every_cycle: bool = False
    #: transformer-template extras ("pre"/"post" placement, MoE grouping).
    placement: str = "post"
    parallel_group: Optional[str] = None
    moe_component: Optional[str] = None
    moe_routing_mode: Optional[str] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy(cls, name: str, data: Mapping[str, Any]) -> "CommMeta":
        kind = data.get("type")
        if kind is not None and not isinstance(kind, CollectiveType):
            raise TypeError(
                f"comm_metadata[{name!r}]['type'] must be CollectiveType "
                f"(got {type(kind).__name__})"
            )
        known = {
            "size",
            "type",
            "participants",
            "interconnect_type",
            "local_comp_time",
            "tp_shard",
            "ga_required_every_cycle",
            "placement",
            "parallel_group",
            "moe_component",
            "moe_routing_mode",
        }
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            name=name,
            size_bytes=data.get("size", 0),
            kind=kind,
            participants=int(data.get("participants", 1) or 0),
            interconnect=data.get("interconnect_type"),
            local_comp_time=float(data.get("local_comp_time", 0) or 0.0),
            tp_shard=bool(data.get("tp_shard", False)),
            ga_required_every_cycle=bool(data.get("ga_required_every_cycle", False)),
            placement=str(data.get("placement", "post")),
            parallel_group=data.get("parallel_group"),
            moe_component=data.get("moe_component"),
            moe_routing_mode=data.get("moe_routing_mode"),
            extra=extra,
        )


def comm_metadata_from_legacy(raw: Mapping[str, Mapping[str, Any]]) -> Dict[str, CommMeta]:
    """Convert a legacy ``comm_metadata`` dict (insertion order preserved)."""
    return {name: CommMeta.from_legacy(name, data) for name, data in (raw or {}).items()}


@dataclass(frozen=True)
class GemmDirection:
    """One direction of a GEMM template entry.

    ``duration`` may be ``None`` when the legacy entry carried no duration —
    the flattened expansion raises the legacy error message in that case.
    ``comm_keys`` keeps the legacy list order: the fine expansion chains them
    serially after the GEMM node (it ignores pre/post placement — that is a
    BLOCK-builder concept, see :mod:`program.block_program`).
    """

    duration: Optional[float]
    comm_keys: Tuple[str, ...] = ()


@dataclass(frozen=True)
class GemmEntry:
    name: str
    forward: GemmDirection
    backward: GemmDirection

    def direction(self, direction_name: str) -> GemmDirection:
        if direction_name == "forward":
            return self.forward
        if direction_name == "backward":
            return self.backward
        raise ValueError(f"Unknown direction {direction_name!r}")


@dataclass(frozen=True)
class BlockTemplate:
    """A transformer block as the fine builder consumes it."""

    entries: Tuple[GemmEntry, ...]
    comm_metadata: Mapping[str, CommMeta]

    @classmethod
    def from_gemm_entries(
        cls,
        gemm_entries: Any,
        comm_metadata: Mapping[str, Mapping[str, Any]],
    ) -> "BlockTemplate":
        entries = []
        for idx, entry in enumerate(gemm_entries):
            name = entry.get("name", f"g{idx}")

            def _direction(cfg: Mapping[str, Any]) -> GemmDirection:
                duration = cfg.get("duration")
                return GemmDirection(
                    duration=float(duration) if duration is not None else None,
                    comm_keys=tuple(cfg.get("comm_keys", []) or []),
                )

            entries.append(
                GemmEntry(
                    name=name,
                    forward=_direction(entry.get("forward", {}) or {}),
                    backward=_direction(entry.get("backward", {}) or {}),
                )
            )
        return cls(
            entries=tuple(entries),
            comm_metadata=comm_metadata_from_legacy(comm_metadata),
        )
