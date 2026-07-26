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

"""``build_coarse_program`` — the COARSE pipeline Program builder (M5).

The COARSE program is the typed replacement of the legacy analytical/hybrid
pipeline graph (``Graph.construct_fwd_bwd_graph`` output): one COMPUTE op
per (micro-batch, layer, direction) on its stage device, embedding /
softmax / optimizer / recompute ops, cross-stage pipeline ``TransferOp``s
with byte sizes, same-stage zero-byte ``TransferOp``s preserved AS EVENTS
(DESIGN.md §2 amendment 2: they are heap events in the analytical evaluator
and shift tie order), the DP/ZeRO-2/3 collective lattice and EP sync as
``CollectiveOp``s, and the GPipe cross-microbatch dependencies as plain
deps.

It is built from the SAME :func:`program.schedule.build_pipeline_events`
stream the FINE builder expands — the one-builder-family-over-one-schedule
end state: the events ARE the coarse structure, so the coarse ops are a 1:1
typed projection of them (no expansion pass).

M5 scope note (the :mod:`program.memory_sim` precedent): the analytical
evaluator (:mod:`program.analytic_sim`) replays ``Graph.simulate``'s event
loop, whose FIFO/tie discipline depends on the legacy *children-list
adjacency order* — an order that op-uid order cannot represent (legacy
attaches GPipe/ZeRO deps to already-created nodes, so a node's children are
not uid-sorted). The builder therefore keeps the schedule events alive in
``meta.misc``:

* ``meta.misc["coarse_events"]`` — tuple of events aligned with op uids
  (``coarse_events[uid]`` is the event op ``uid`` was built from);
* ``meta.misc["coarse_proto_root"]`` — the events root (evaluation +
  ``RAPID_VISUALIZE_GRAPHS`` rendering);
* ``meta.misc["coarse_pp"]`` — the legacy ``Graph.pp`` (device-count rule).

The events carry the legacy-exact comm surface (RAW ``comm_size_bytes``
floats, participants, interconnect labels — exactly the legacy ``Edge``
attributes); the ops are the typed metadata + retiming surface
(:mod:`program.retime` writes per-DP duration tuples onto layer COMPUTE ops
selected by ``role``/``layer``/``micro_batch``/``direction``/
``is_moe_layer``/``device``). Collapsing the events into the op list proper
is M8 work (DESIGN.md §5).

Validation note: the coarse Program is constructed with validation OFF and
uids in first-encounter DFS order (the ``convert_comm_sizes_to_times``
traversal order). The legacy coarse graph legally violates V1's
``dep < uid`` rule (late-attached GPipe/ZeRO parents) and V5's dp/label
coupling (EP-sync collectives are ``is_dp=False`` with no label); the
coarse OP LIST is consumed only by :mod:`program.analytic_sim` /
:mod:`program.retime`. The M6 hierarchical emission does not emit it
either: :func:`lower_coarse_for_emission` lowers the (op_id-stamped,
retimed) EVENTS through :func:`program.legacy_lowering.lower_to_program`,
which produces its own validated emission-ordered Program.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from timing_model import CollectiveType

from program.ir import (
    CollectiveOp,
    ComputeOp,
    Direction,
    Op,
    OpRole,
    Program,
    ProgramMeta,
    TransferOp,
)
from program.layout import RankLayout
from program.schedule import (
    CommEvent,
    ComputeEvent,
    ScheduleSpec,
    build_pipeline_events,
)

#: legacy ``ComputeEvent.role`` -> (OpRole, Direction-from-fwd override)
_ROLE_MAP: Dict[str, OpRole] = {
    "layer": OpRole.TRANSFORMER_LAYER,
    "recompute": OpRole.TRANSFORMER_LAYER,
    "embedding": OpRole.EMBEDDING,
    "embedding_b": OpRole.EMBEDDING,
    "softmax": OpRole.SOFTMAX,
    "softmax_b": OpRole.SOFTMAX,
    "optimizer": OpRole.OPTIMIZER,
}


def _enumerate_events(root: Any) -> List[Any]:
    """All events reachable from ``root`` in first-encounter DFS preorder.

    This is exactly the encounter order of the legacy
    ``convert_comm_sizes_to_times`` recursion (children in list order,
    descend before the next sibling), so uid order is the conversion-pass
    order.
    """
    order: List[Any] = []
    seen: set = set()
    stack: List[Any] = [root]
    while stack:
        event = stack.pop()
        event_id = id(event)
        if event_id in seen:
            continue
        seen.add(event_id)
        order.append(event)
        stack.extend(reversed(event.children))
    return order


def _placement(event: Any) -> Optional[int]:
    """Stage placement of an event (legacy ``hw_id`` / ``local_hw_id``)."""
    hw_id = getattr(event, "hw_id", None)
    if hw_id is not None:
        return int(hw_id)
    local = getattr(event, "local_hw_id", None)
    return int(local) if local is not None else None


def build_coarse_program(
    schedule_spec: ScheduleSpec,
    layout: Optional[RankLayout],
    *,
    dp_count: Optional[int] = None,
    label: str = "coarse",
) -> Program:
    """Build the COARSE pipeline :class:`~program.ir.Program`.

    ``layout`` is the ("pp","dp") sublayout when available (informational —
    the M6 hierarchical emission passes the sublayout descriptor to
    :func:`lower_coarse_for_emission` explicitly). ``dp_count`` is the
    duration-profile replication degree AND the hierarchical emission dp
    (the dispatcher passes 1 for inference — the legacy ``dp_override=1``
    rule).
    """
    events = build_pipeline_events(schedule_spec)
    order = _enumerate_events(events.root)
    uid_of: Dict[int, int] = {id(event): uid for uid, event in enumerate(order)}

    def _dep_uids(event: Any) -> Tuple[int, ...]:
        deps: List[int] = []
        for parent in event.parents:
            parent_uid = uid_of.get(id(parent))
            if parent_uid is None:
                raise ValueError(
                    f"Coarse event '{event.name}' has a parent "
                    f"('{parent.name}') unreachable from the schedule root; "
                    "the legacy simulator would deadlock on it"
                )
            deps.append(parent_uid)
        return tuple(deps)

    ops: List[Op] = []
    max_compute_stage = -1
    for uid, event in enumerate(order):
        if isinstance(event, ComputeEvent):
            role = _ROLE_MAP.get(event.role, OpRole.GENERIC)
            max_compute_stage = max(max_compute_stage, int(event.hw_id))
            ops.append(
                ComputeOp(
                    uid=uid,
                    name=event.name,
                    device=int(event.hw_id),
                    duration=(float(event.duration),),
                    deps=_dep_uids(event),
                    role=role,
                    direction=Direction.FORWARD if event.fwd else Direction.BACKWARD,
                    mem_kind=event.mem_kind,
                    recompute=bool(event.recompute),
                    micro_batch=event.micro_batch_index,
                    layer=event.layer_index,
                    is_moe_layer=bool(event.is_moe_layer),
                )
            )
            continue

        if not isinstance(event, CommEvent):
            raise TypeError(f"Unsupported schedule event type: {type(event)!r}")

        if event.comm_type is None:
            raise ValueError(
                f"Coarse comm event '{event.name}' has no CollectiveType"
            )

        if event.comm_type == CollectiveType.PIPELINE:
            # Pipeline p2p (cross-stage with bytes, or the same-stage
            # zero-byte control edge — kept as an op/event, DESIGN §2.2).
            deps = _dep_uids(event)
            if not deps:
                raise ValueError(
                    f"Coarse pipeline edge '{event.name}' has no producer"
                )
            src_device = _placement(event.parents[0])
            if src_device is None:
                src_device = -1
            dst_event = event.children[0] if event.children else None
            dst_device = _placement(dst_event) if dst_event is not None else None
            if dst_device is None:
                dst_device = src_device
            ops.append(
                TransferOp(
                    uid=uid,
                    name=event.name,
                    src_device=src_device,
                    dst_device=dst_device,
                    size_bytes=int(event.comm_size_bytes),
                    comm_type=event.comm_type,
                    producer=deps[0],
                    deps=deps,
                )
            )
            continue

        placement = _placement(event)
        ops.append(
            CollectiveOp(
                uid=uid,
                name=event.name,
                # -1 = unplaced (legacy dp edges without local_hw_id).
                device=placement if placement is not None else -1,
                coll=event.comm_type,
                size_bytes=int(event.comm_size_bytes),
                participants=int(event.participants),
                interconnect=event.comm_interconnect_type,
                is_dp=bool(event.is_dp),
                deps=_dep_uids(event),
            )
        )

    base_devices = max(1, int(schedule_spec.pp) if schedule_spec.pp else 1)
    if max_compute_stage >= 0:
        base_devices = max(base_devices, max_compute_stage + 1)

    effective_dp = int(dp_count) if dp_count is not None else int(schedule_spec.dp)
    meta = ProgramMeta(
        label=label,
        misc={
            "granularity": "coarse",
            "coarse_events": tuple(order),
            "coarse_proto_root": events.root,
            "coarse_pp": int(schedule_spec.pp),
            "num_stages_initial": base_devices,
        },
    )
    return Program(
        layout=layout if layout is not None else RankLayout((), {}, {}),
        dp_count=max(1, effective_dp),
        devices=tuple(range(base_devices)),
        ops=ops,
        groups={},
        meta=meta,
    )


def lower_coarse_for_emission(
    coarse_program: Program,
    *,
    layout_descriptor: Optional[Dict[str, Any]] = None,
    optimize_2dmap: Optional[Dict[str, Any]] = None,
    gmap_workdir: Optional[str] = None,
) -> Program:
    """Lower the (retimed) coarse schedule events for AstraSim emission (M6).

    The COARSE program's own op list is uid-ordered in the legacy
    *conversion* DFS order and is consumed by :mod:`program.analytic_sim`;
    Chakra ET emission needs the legacy EMISSION order (per-stage Kahn
    toposort keyed by ``op_id``, Step-11 transfer replay, collective label
    assignment). Rather than duplicating that ordering logic, the coarse
    proto root — the schedule events, ``op_id``-stamped by
    :func:`program.schedule.build_pipeline_events` in the exact legacy
    creation sequence — is lowered through the SAME
    :func:`program.legacy_lowering.lower_to_program` pass the fine builder
    shares (M3a pattern). The events graph is ``add_child``-order
    isomorphic to the legacy ``construct_fwd_bwd_graph`` output, so the
    resulting Program (and hence the emitted bundle) is element-for-element
    what the legacy hierarchical pipeline path produced.

    ``layout_descriptor`` is the ("pp","dp") pipeline sublayout descriptor
    (was the legacy root's ``_astrasim_rank_layout``); ``optimize_2dmap``
    the first-dimension SCOTCH config (was ``_optimize_2dmap`` — passed
    explicitly because the event classes are slotted). Retimed per-DP layer
    durations are read from the events' ``duration_profile`` (written by
    the :func:`program.retime.apply_block_timings` mirror), exactly like
    the legacy ``Node.duration`` tuples were.
    """
    from program.legacy_lowering import lower_to_program

    if not isinstance(coarse_program, Program):
        raise TypeError(
            f"lower_coarse_for_emission expects a Program (got {type(coarse_program).__name__})"
        )
    if coarse_program.meta.misc.get("granularity") != "coarse":
        raise RuntimeError("lower_coarse_for_emission requires a COARSE program")
    root = coarse_program.meta.misc.get("coarse_proto_root")
    if root is None:
        raise RuntimeError(
            "COARSE program does not carry its schedule events root "
            "(meta.misc['coarse_proto_root'])"
        )

    program = lower_to_program(
        root,
        coarse_program.dp_count,
        layout_descriptor,
        gmap_workdir=gmap_workdir,
        optimize_2dmap=optimize_2dmap,
    )
    program.meta.label = coarse_program.meta.label
    return program
