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

"""L1 — MoE routing policy (INTERFACES.md §2.6).

Absorbs ``block_program._moe_hot_rank`` (:223-228) and
``_moe_parallel_token`` (:230-235) — two byte-identical two-value if-chains —
as **data**, so a third routing mode is one :data:`ROUTING_MODES` row, not two
new ``if`` branches.

It also owns:

* the declared communicator axes of every MoE collective — THIS is the source
  of the composite ``("tp","ep")`` group that ``legacy_lowering.py:243-248``
  infers from ``participants == tp_size * ep_size``;
* the MoE component vocabulary (``base_all_to_all`` / ``residual_p2p``,
  block_program.py:305-313);
* the EP gradient-sync requirement (row **S12** of INTERFACES §2.3), whose
  three-way split today — attach point ``train_timing.py:4762-4768``, gate
  ``schedule.py:936-953``, bytes ``train_timing.py:4599-4620`` — collapses to
  "bytes stay upstream, gate + attach live here".

Equivalence to the deleted code, checked term by term:

===========================================  =========================================
legacy                                       :class:`AxisRouting`
===========================================  =========================================
``_moe_hot_rank("ep")   = rank(tp, cp, 0)``  ``hot_coords`` zeroes ``ep``
``_moe_hot_rank("tp_ep")= rank(0, cp, 0)``   ``hot_coords`` zeroes ``tp``,``ep``
``_moe_parallel_token("ep")    = (m,cp,tp)`` ``("ep", tp, cp)`` — same partition of
                                             ranks; the tuple is a DICT KEY, so
                                             component order is inert
``_moe_parallel_token("tp_ep") = (m,cp)``    ``("tp_ep", cp)``
===========================================  =========================================
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple

from program.types import AxisName, Coords

__all__ = [
    "RoutingError",
    "MoEComponent",
    "MoERoutingPolicy",
    "AxisRouting",
    "EP_ROUTING",
    "TP_EP_ROUTING",
    "ROUTING_MODES",
    "routing_for_mode",
    "routing_policy_for",
]


class RoutingError(ValueError):
    """An unsupported or inconsistent MoE routing declaration."""


class MoEComponent(Enum):
    """The MoE post-group component vocabulary. A third component is a new
    member plus one row in :data:`MOE_COMPONENTS` — no new ``if`` branch in an
    expansion loop (block_program.py:305-313 is what this replaces)."""

    BASE_ALL_TO_ALL = "base_all_to_all"
    RESIDUAL_P2P = "residual_p2p"


MOE_COMPONENTS: Mapping[str, MoEComponent] = {c.value: c for c in MoEComponent}


def component_of(spec: Any) -> Optional[MoEComponent]:
    """The declared component of a ``CommSpec`` (``None`` when not a MoE
    parallel-group member). Raises on an unknown component name — the
    ``ValueError`` at block_program.py:310, moved to the declaration."""
    name = spec.moe_component
    if name is None:
        return None
    if name not in MOE_COMPONENTS:
        raise RoutingError(
            f"MoE parallel comm group contains unsupported component {name!r} "
            f"for key {spec.key!r} (known: {sorted(MOE_COMPONENTS)})"
        )
    return MOE_COMPONENTS[name]


class MoERoutingPolicy(Protocol):
    name: str

    def routing_axes(self) -> Tuple[AxisName, ...]:
        """The axes the routing collective spans."""

    def hot_coords(self, coords: Coords) -> Coords:
        """The coordinates of the hot rank of ``coords``' routing group."""

    def join_token(self, coords: Coords) -> Tuple[Any, ...]:
        """A hashable identity for the routing group ``coords`` belongs to."""


@dataclass(frozen=True)
class AxisRouting(MoERoutingPolicy):
    """ONE implementation covering both existing modes and any future one.

    ``hot_coords`` = coords with every routing axis zeroed.
    ``join_token`` = the coords on the axes NOT routed over, in canonical order.
    """

    name: str
    axes: Tuple[AxisName, ...]
    cluster_axes: Tuple[AxisName, ...] = ("tp", "cp", "ep")

    def __post_init__(self) -> None:
        if not self.axes:
            raise RoutingError(f"AxisRouting {self.name!r} declares no routing axes")
        unknown = [a for a in self.axes if a not in self.cluster_axes]
        if unknown:
            raise RoutingError(
                f"AxisRouting {self.name!r} routes over {unknown} which are not "
                f"cluster axes {list(self.cluster_axes)}"
            )

    def routing_axes(self) -> Tuple[AxisName, ...]:
        return tuple(self.axes)

    def hot_coords(self, coords: Coords) -> Dict[AxisName, int]:
        return {**dict(coords), **{axis: 0 for axis in self.axes}}

    def is_hot(self, coords: Coords) -> bool:
        return all(int(dict(coords).get(axis, 0)) == 0 for axis in self.axes)

    def join_token(self, coords: Coords) -> Tuple[Any, ...]:
        base = dict(coords)
        rest = tuple(a for a in self.cluster_axes if a not in self.axes)
        return (self.name,) + tuple(int(base.get(axis, 0)) for axis in rest)

    def hot_device(self, members: Sequence[int]) -> int:
        """P4: the hot rank of a routing group is ``min(members)`` under
        ``RankLayout.linearize`` — hot-before-cold construction is guaranteed by
        construction, not asserted at runtime (block_program.py:356-361)."""
        if not members:
            raise RoutingError(f"AxisRouting {self.name!r}: empty routing group")
        return min(int(m) for m in members)


#: training — experts sharded across ``ep`` only
EP_ROUTING = AxisRouting(name="ep", axes=("ep",))
#: inference — experts sharded across ``tp`` x ``ep`` (the composite communicator)
TP_EP_ROUTING = AxisRouting(name="tp_ep", axes=("tp", "ep"))

#: keyed by ``CommSpec.moe_routing_mode`` (train_timing._moe_comm_routing_mode)
ROUTING_MODES: Mapping[str, AxisRouting] = {
    EP_ROUTING.name: EP_ROUTING,
    TP_EP_ROUTING.name: TP_EP_ROUTING,
}


def routing_for_mode(mode: str) -> AxisRouting:
    """Table lookup; raises :class:`RoutingError` naming every known mode.
    This is the ONLY place a routing-mode string is interpreted."""
    if mode not in ROUTING_MODES:
        raise RoutingError(
            f"Unsupported MoE routing mode {mode!r} (known: {sorted(ROUTING_MODES)})"
        )
    return ROUTING_MODES[mode]


def routing_policy_for(fw: Any) -> Optional[AxisRouting]:
    """Select the routing policy declared by the workload's comm table.

    The mode is DATA on the MoE comm specs (``CommSpec.moe_routing_mode``); a
    workload with no MoE collectives has no routing policy. When ``ep > 1``
    without any MoE collective (dense layers of a mixed model) the EP grad-sync
    still needs a home, so the default :data:`EP_ROUTING` is used.
    """
    declared = {
        spec.moe_routing_mode
        for spec in fw.spec.comm.values()
        if spec.moe_routing_mode is not None
    }
    if len(declared) > 1:
        raise RoutingError(
            f"Workload declares conflicting MoE routing modes {sorted(declared)}"
        )
    if declared:
        return routing_for_mode(next(iter(declared)))
    if fw.spec.degrees.ep > 1:
        return EP_ROUTING
    return None


# ---------------------------------------------------------------------------
# S12 — the EP gradient sync
# ---------------------------------------------------------------------------

#: (is_moe_layer) -> comm key. schedule.py:940-944, as a table.
EP_SYNC_KEYS: Mapping[bool, str] = {
    False: "transformer_dense_ep_sync",
    True: "transformer_moe_ep_sync",
}


def ep_sync_requirements(work: Any, ctx: Any) -> Tuple[Any, ...]:
    """Row **S12**: ``AFTER(LAYER/BACKWARD(b, l))`` on the EP grad-sync key.

    The gate that legacy writes twice — ``schedule.py:936-938``
    (``apply_ep_all_mbs = dp_microbatch_mode != "last_mb" or zero_stage >= 3``)
    is a re-derivation of ``Graph._should_emit_dp_comm`` (schedule.py:255-268) —
    is here exactly ONE call to :meth:`GradAccumPolicy.emits` (audit A6).
    """
    from program.work import (
        AttachMode,
        Direction,
        SyncKey,
        SyncPhase,
        SyncRequirement,
        SyncSpread,
        WorkKind,
        is_moe,
    )

    if work.kind is not WorkKind.LAYER or work.direction is not Direction.BACKWARD:
        return ()
    # ``spec.ep > 1`` (schedule.py:936) — the GRAPH ep. The keys are only
    # registered upstream when ep > 1, but the degree is the declared gate.
    if ctx.fw.spec.degrees.ep <= 1:
        return ()
    key = EP_SYNC_KEYS[bool(is_moe(work, ctx.fw))]
    spec = ctx.spec_for(key)
    if spec is None or not ctx.grad_accum.emits(spec, work.microbatch):
        return ()
    return (
        SyncRequirement.from_spec(
            spec,
            key=SyncKey(key, SyncPhase.GRAD, work.microbatch, work.layer),
            place_on=work,
            mode=AttachMode.AFTER,
            anchors=(work,),
            spread=SyncSpread.CLUSTER_RANK_0,
            origin="S12",
        ),
    )
