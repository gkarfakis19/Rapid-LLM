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

"""Chakra ET emission over a :class:`program.ir.Program` (DESIGN.md §3).

``emit_chakra(program, output_dir)`` runs three phases — all AstraSim contract
rules live here, once, with contract citations. **There is one id policy:
program order** (INTERFACES §4.6, ``kahn(slot, device, intra)``); the migration's
``id_policy`` parameter and its ``NotImplementedError`` are gone with the legacy
lowering that motivated them.

* **Phase A** — main ops in uid order, stamped per dp index with the
  dp-major rank formula ``rank = dp_idx * num_stages_initial + stage_idx``
  (pre-extension stage count — executor.py Step 7 quirk, DESIGN §2.5).
  ``dp_count <= 1`` skips ``is_dp`` collectives (legacy Step 10);
  singleton wire groups emit a zero-duration ``*_noop`` COMP (contract:
  1-member native ring collectives never terminate); durations are
  ``int(round(sec * 1e6))`` clamped >= 0; wire gids intern from 1000 in
  sorted-label, sorted-token order (verbatim ``_assign_collective_labels``
  + ``_TP_MEMBERS_TO_GID`` semantics — DESIGN §2.6); dp collectives get the
  stage group ``str(stage_idx + 1)`` when ``dp_count > 1``.
* **Phase B** — p2p materialization: one TransferOp => one tag (its uid) =>
  SEND on the src rank + RECV on the dst rank, created in **program order**
  (transfer uid, send before recv); control transfers
  (``TransferOp.is_control``) emit 1 byte with ``*_send_control`` /
  ``*_recv_control`` names; RECV ids are wired into dst-side consumers and
  SEND ids into src-side consumers (legacy ``ensure_pipeline`` /
  ``ensure_local_pipeline_sends`` dep appends, membership-checked).
  Same-device transfers are elided (the consumer already deps on the
  producer — DESIGN §2.2).
* **Phase C** — stable control-first renumber, a verbatim port of
  ``_RankTrace._renumber_control_priority`` (node ids are AstraSim
  scheduling priorities; tiny control sends must not starve). The partition
  reads the CARRIED ``TransferOp.is_control`` decision (recorded on the trace
  when the node is created), not the node name it wrote — re-parsing a string
  the emitter itself produced was audit item 7.

ALWAYS-ON POSTCONDITION (DESIGN §3): after Phase C, for every wire
communicator group, every member rank's sequence of that group's COLL
``(comm_type, comm_size)`` payloads in id order must be identical —
AstraSim matches collectives by per-rank issue order, and divergence is a
silent deadlock. Violations raise :class:`EmissionError` before AstraSim
ever runs.

Outputs: per-rank ``llm_graph.<rank>.et`` files, ``manifest.json`` (exact
``_manifest_op_key`` sort — the AstraSim cache key), and
``comm_groups.json`` (dp stage groups when dp > 1, plus interned wire
groups — reproducing ``_write_comm_groups_json``).
"""

from __future__ import annotations

import itertools
import json
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from timing_model import CollectiveType

from astrasim_lib.et_utils import (
    chakra_encode,
    new_comm_node,
    new_comp_node,
    new_recv_node,
    new_send_node,
    pb,
    write_et_node,
)
from program.ir import CollectiveOp, ComputeOp, Program, TransferOp
from program.validate import validate_program

_WIRE_GROUP_BASE_ID = 1000  # legacy _TP_GROUP_BASE_ID


class EmissionError(RuntimeError):
    """Emission hit an inconsistency (or the group-order postcondition)."""


class CrossRankDepWarning(UserWarning):
    """A dependency edge crossed ranks with no transfer carrying it.

    A Chakra ``ctrl_dep`` is a node id in the same trace, so such an edge cannot
    be emitted. It is dropped and reported — never silently — see
    ``emit_chakra``'s ``_resolved_deps``.
    """


@dataclass
class EmittedBundle:
    et_prefix: str
    rank_ids: List[int]
    manifest_path: str
    comm_groups: Dict[str, List[int]] = field(default_factory=dict)
    comm_groups_path: Optional[str] = None


def _collective_enum(coll: CollectiveType) -> int:
    """pb-enum half of legacy ``get_collective_type`` (executor.py:439-453)."""
    if coll is None:
        raise ValueError("Collective comm_type is required")
    if not isinstance(coll, CollectiveType):
        raise TypeError(f"comm_type must be CollectiveType (got {type(coll).__name__})")
    if coll == CollectiveType.PIPELINE:
        raise ValueError("Pipeline comm_type should not be mapped to a collective enum")
    mapping = {
        CollectiveType.ALL_REDUCE: pb.ALL_REDUCE,
        CollectiveType.ALL_GATHER: pb.ALL_GATHER,
        CollectiveType.REDUCE_SCATTER: pb.REDUCE_SCATTER,
        CollectiveType.ALL_TO_ALL: pb.ALL_TO_ALL,
    }
    return mapping[coll]


class _Trace:
    """Per-rank node list (the emission-side ``_RankTrace``)."""

    def __init__(self, rank: int, path: str) -> None:
        self.rank = rank
        self.path = path
        self.nodes: List["pb.Node"] = []
        #: node indices created for a control transfer (``TransferOp.is_control``).
        #: Phase C partitions on THIS, not on the node name (audit item 7).
        self.control_ids: Set[int] = set()

    @property
    def next_id(self) -> int:
        return len(self.nodes)

    def append_node(self, node: "pb.Node", *, control: bool = False) -> None:
        if control:
            self.control_ids.add(len(self.nodes))
        self.nodes.append(node)

    def renumber_control_priority(self) -> None:
        """Verbatim port of ``_RankTrace._renumber_control_priority``
        (executor.py:368-429): stable-partition ``*_send_control`` /
        ``*_recv_control`` nodes to the lowest ids, remap all ctrl_deps."""
        if not self.nodes:
            return

        control_nodes: List["pb.Node"] = []
        regular_nodes: List["pb.Node"] = []
        for index, node in enumerate(self.nodes):
            if index in self.control_ids:
                control_nodes.append(node)
            else:
                regular_nodes.append(node)

        if not control_nodes:
            return

        id_map: Dict[int, int] = {}
        new_order: List["pb.Node"] = []
        for node in control_nodes + regular_nodes:
            new_id = len(new_order)
            id_map[int(node.id)] = new_id
            node.id = new_id
            new_order.append(node)

        for node in new_order:
            remapped = [id_map.get(int(dep), int(dep)) for dep in node.ctrl_deps]
            node.ctrl_deps[:] = remapped

        self.nodes = new_order
        self.control_ids = set(range(len(control_nodes)))

    def write(self) -> None:
        with open(self.path, "wb") as fh:
            chakra_encode(fh, pb.GlobalMetadata(version="0.0.4"))
            for node in self.nodes:
                write_et_node(fh, node)


def emit_chakra(program: Program, output_dir: str) -> EmittedBundle:
    """Emit one bundle. THE id policy is program order — there is no other."""
    validate_program(program, check_races=False)
    os.makedirs(output_dir, exist_ok=True)

    ops = program.ops
    dp_count = max(int(program.dp_count), 1)
    devices = program.devices
    device_index = {device: idx for idx, device in enumerate(devices)}
    ns_initial = program.num_stages_initial()
    compute_devices = program.compute_devices()

    def rank_for(device: int, dp_idx: int) -> int:
        idx = device_index.get(device)
        if idx is None:
            raise EmissionError(f"Device {device} is not in Program.devices")
        return dp_idx * ns_initial + idx

    # --- traces: one per (compute device, dp) — legacy Step 2 -------------
    traces: Dict[int, _Trace] = {}
    for dp_idx in range(dp_count):
        for device in compute_devices:
            rank = rank_for(device, dp_idx)
            traces[rank] = _Trace(rank, os.path.join(output_dir, f"llm_graph.{rank}.et"))

    def _trace_for(rank: int, context: str) -> _Trace:
        trace = traces.get(rank)
        if trace is None:
            raise EmissionError(
                f"{context}: rank {rank} has no trace (device outside the compute "
                "stage set — the legacy converter would KeyError here)"
            )
        return trace

    # --- wire gid interning (sorted-label, sorted-token; base 1000) -------
    label_tokens: Dict[str, set] = defaultdict(set)
    for op in ops:
        if isinstance(op, CollectiveOp) and op.label is not None and op.group is not None:
            for dp_idx in range(dp_count):
                members_tuple = tuple(sorted(rank_for(d, dp_idx) for d in op.group.members))
                label_tokens[op.label].add((op.group.axis, dp_idx, members_tuple))

    label_dp_gid: Dict[Tuple[str, int], str] = {}
    members_to_gid: Dict[Tuple[int, ...], str] = {}
    gid_members: Dict[str, List[int]] = {}
    # gid -> the parallelism axes / labels its collectives were built for.
    # Communicator *ids* are an emitter choice, so this sidecar is what lets a
    # consumer attribute bytes to an interconnect axis without re-inferring it
    # from participant counts.
    gid_axes: Dict[str, set] = defaultdict(set)
    gid_labels: Dict[str, set] = defaultdict(set)
    gid_counter = itertools.count(start=_WIRE_GROUP_BASE_ID)
    for label in sorted(label_tokens.keys()):
        for axis, dp_idx, members_tuple in sorted(label_tokens[label]):
            gid = members_to_gid.get(members_tuple)
            if gid is None:
                gid = str(next(gid_counter))
                members_to_gid[members_tuple] = gid
                if gid not in gid_members:
                    gid_members[gid] = list(members_tuple)
            existing = label_dp_gid.get((label, dp_idx))
            if existing is not None and existing != gid:
                # One label mapping to two different member sets at the same
                # dp index would silently last-win here, stamping the losing
                # ops with a pg_name of a communicator their rank does not
                # belong to — the silent-AstraSim-deadlock class the group-
                # order postcondition exists to make loud. Fail at interning.
                raise EmissionError(
                    f"collective label '{label}' maps to two different "
                    f"communicator member sets at dp index {dp_idx}: gid "
                    f"{existing} (members {gid_members.get(existing)}) vs gid "
                    f"{gid} (members {gid_members.get(gid)}). One label must "
                    "identify exactly one communicator per dp index."
                )
            label_dp_gid[(label, dp_idx)] = gid
            gid_axes[gid].add(str(axis))
            gid_labels[gid].add(str(label))

    # --- Phase A: main ops in uid order ------------------------------------
    et_ids: Dict[Tuple[int, int], int] = {}
    # gid -> rank -> [(node, comm_type_enum, comm_size)] for the postcondition
    group_records: Dict[str, Dict[int, List[Tuple[Any, int, int]]]] = defaultdict(lambda: defaultdict(list))

    #: (consumer uid, dep uid) pairs dropped because they cross ranks — see
    #: ``_resolved_deps``. Reported once per bundle, never silently.
    cross_rank_drops: List[Tuple[int, int]] = []

    def _resolved_deps(op: Any, dp_idx: int) -> List[int]:
        """Phase-A ctrl_deps of a main op.

        **A Chakra ``ctrl_dep`` is a node id in the SAME trace.** That is an ET
        fact, not an IR fact, so this is the one place that knows it, and it
        forces three cases:

        * a **cross-device TransferOp** dep is skipped: the transfer has no ET
          node of its own, it materializes as a SEND/RECV pair in Phase B, and
          the pair's ids are wired into the consumers there;
        * a **same-device TransferOp** dep is ELIDED (never emitted, DESIGN §2.2)
          and must be REPLACED by a plain dep on what it depends on — its
          producer and, when present, the compute anchor. Dropping the edge
          instead lets the ranks of one stage drift out of collective-issue
          lockstep, which AstraSim reports as ``Hardware Resource ... has
          unreleased nodes``: a silent deadlock (measured; this is why the
          recursion exists);
        * a dep on an op on **another rank** with no transfer carrying it is not
          expressible. Legacy materialized such an edge as a zero-byte control
          transfer inside the lowering; ``build()`` keeps it as a plain SYNC dep
          (a ZeRO-3 gather inheriting a cross-stage anchor's deps — rows S6/S14),
          which the analytical evaluator honors and the wire cannot. It is
          dropped HERE and counted, because writing a foreign rank's node id
          would mis-order the trace or close a cycle (measured on
          ``train:*:dp2tp1cp1pp2mb2sp0:zero3``).
        """
        deps: List[int] = []
        own_rank = rank_for(op.device, dp_idx)

        def _resolve(uid: int) -> None:
            dep_op = ops[uid]
            if isinstance(dep_op, TransferOp):
                if dep_op.src_device != dep_op.dst_device:
                    return  # Phase B wires the SEND/RECV ids
                if rank_for(dep_op.src_device, dp_idx) != own_rank:
                    cross_rank_drops.append((op.uid, uid))
                    return
                for inner in dep_op.deps:
                    _resolve(int(inner))
                return
            dep_rank = rank_for(dep_op.device, dp_idx)
            if dep_rank != own_rank:
                cross_rank_drops.append((op.uid, uid))
                return
            key = (uid, dep_rank)
            if key not in et_ids:
                raise EmissionError(
                    f"op {op.uid} ('{op.name}') depends on op {uid} "
                    f"('{dep_op.name}') which has no ET node on rank {dep_rank} "
                    "(skipped or not yet emitted)"
                )
            deps.append(et_ids[key])

        for dep in op.deps:
            _resolve(int(dep))
        unique: List[int] = []
        for dep in deps:
            if dep not in unique:
                unique.append(dep)
        return unique

    for op in ops:
        if isinstance(op, TransferOp):
            continue
        if isinstance(op, ComputeOp):
            for dp_idx in range(dp_count):
                rank = rank_for(op.device, dp_idx)
                trace = _trace_for(rank, f"compute op {op.uid}")
                unique_deps = _resolved_deps(op, dp_idx)
                duration_sec = float(op.duration[dp_idx] if len(op.duration) > 1 else op.duration[0])
                duration_micros = int(round(duration_sec * 1e6)) if duration_sec else 0
                node_id = trace.next_id
                comp_node = new_comp_node(node_id, f"{op.name}_{op.uid}", max(duration_micros, 0))
                comp_node.ctrl_deps.extend(unique_deps)
                trace.append_node(comp_node)
                et_ids[(op.uid, rank)] = node_id
            continue

        # CollectiveOp
        if dp_count <= 1 and op.is_dp:
            # Legacy Step-10 skip of dp stage collectives at dp <= 1.
            continue
        for dp_idx in range(dp_count):
            rank = rank_for(op.device, dp_idx)
            trace = _trace_for(rank, f"collective op {op.uid}")
            unique_deps = _resolved_deps(op, dp_idx)
            node_id = trace.next_id
            if op.label is not None:
                comm_name = op.label
                gid = label_dp_gid.get((op.label, dp_idx))
                members = gid_members.get(gid) if gid else None
                if members is not None and len(members) <= 1:
                    # Contract: single-member native ring collectives deadlock;
                    # emit a zero-duration COMP no-op instead (legacy rule).
                    noop_node = new_comp_node(node_id, f"{comm_name}_noop", 0)
                    noop_node.ctrl_deps.extend(unique_deps)
                    trace.append_node(noop_node)
                    et_ids[(op.uid, rank)] = node_id
                    continue
                comm_node = new_comm_node(node_id, comm_name, _collective_enum(op.coll), op.size_bytes)
                if gid:
                    comm_node.attr.append(pb.AttributeProto(name="pg_name", string_val=str(gid)))
                    group_records[str(gid)][rank].append(
                        (comm_node, _collective_enum(op.coll), int(op.size_bytes))
                    )
            else:
                comm_name = f"{op.name}_{op.uid}_dp{dp_idx}"
                comm_node = new_comm_node(node_id, comm_name, _collective_enum(op.coll), op.size_bytes)
                if dp_count > 1:
                    gid = str(device_index[op.device] + 1)
                    comm_node.attr.append(pb.AttributeProto(name="pg_name", string_val=gid))
                    group_records[gid][rank].append(
                        (comm_node, _collective_enum(op.coll), int(op.size_bytes))
                    )
            comm_node.ctrl_deps.extend(unique_deps)
            trace.append_node(comm_node)
            et_ids[(op.uid, rank)] = node_id

    # --- Phase B: p2p materialization --------------------------------------
    transfers = [op for op in ops if isinstance(op, TransferOp)]
    # PROGRAM ORDER is the only ordering policy: transfers in uid order, SEND
    # before its RECV. (The legacy Step-11 ``send_seq``/``recv_seq`` creation
    # positions are deleted with the lowering that produced them.)
    events: List[Tuple[str, TransferOp]] = []
    for transfer in transfers:
        if transfer.src_device == transfer.dst_device:
            continue  # same-device transfers are elided (DESIGN §2.2)
        events.append(("send", transfer))
        events.append(("recv", transfer))

    send_ids: Dict[Tuple[int, int], int] = {}
    recv_ids: Dict[Tuple[int, int], int] = {}
    for kind, transfer in events:
        # The control decision is CARRIED on the op (``TransferOp.is_control``).
        is_control = transfer.is_control
        size = 1 if is_control else transfer.size_bytes  # >= 1 byte for AstraSim
        for dp_idx in range(dp_count):
            src_rank = rank_for(transfer.src_device, dp_idx)
            dst_rank = rank_for(transfer.dst_device, dp_idx)
            if kind == "send":
                trace = _trace_for(src_rank, f"transfer {transfer.uid} send")
                name = (
                    f"{transfer.name}_send_control"
                    if is_control
                    else f"{transfer.name}_send_dp{dp_idx}"
                )
                node_id = trace.next_id
                node = new_send_node(node_id, name, size, dst_rank, transfer.uid)
                producer_key = (transfer.producer, src_rank)
                if producer_key not in et_ids:
                    raise EmissionError(
                        f"transfer {transfer.uid} ('{transfer.name}'): producer op "
                        f"{transfer.producer} has no ET node on rank {src_rank}"
                    )
                node.ctrl_deps.append(et_ids[producer_key])
                trace.append_node(node, control=is_control)
                send_ids[(transfer.uid, dp_idx)] = node_id
            else:
                trace = _trace_for(dst_rank, f"transfer {transfer.uid} recv")
                name = (
                    f"{transfer.name}_recv_control"
                    if is_control
                    else f"{transfer.name}_recv_dp{dp_idx}"
                )
                node_id = trace.next_id
                node = new_recv_node(node_id, name, size, src_rank, transfer.uid)
                trace.append_node(node, control=is_control)
                recv_ids[(transfer.uid, dp_idx)] = node_id

    # Consumer wiring: RECV id into dst-side consumers, SEND id into src-side
    # consumers (legacy Step-11 dep appends; membership-checked, order-free).
    for transfer in transfers:
        if transfer.src_device == transfer.dst_device:
            continue
        for consumer_uid in transfer.consumers:
            consumer = ops[consumer_uid]
            on_dst = consumer.device == transfer.dst_device
            for dp_idx in range(dp_count):
                rank = rank_for(consumer.device, dp_idx)
                consumer_key = (consumer_uid, rank)
                if consumer_key not in et_ids:
                    raise EmissionError(
                        f"transfer {transfer.uid} ('{transfer.name}'): consumer op "
                        f"{consumer_uid} has no ET node on rank {rank}"
                    )
                node = traces[rank].nodes[et_ids[consumer_key]]
                wire_ids = recv_ids if on_dst else send_ids
                wire_id = wire_ids.get((transfer.uid, dp_idx))
                if wire_id is None:
                    raise EmissionError(
                        f"transfer {transfer.uid} ('{transfer.name}'): no "
                        f"{'RECV' if on_dst else 'SEND'} node exists for consumer "
                        f"{consumer_uid} at dp {dp_idx}"
                    )
                if wire_id not in node.ctrl_deps:
                    node.ctrl_deps.append(wire_id)

    if cross_rank_drops:
        pairs = sorted(set(cross_rank_drops))
        sample = ", ".join(
            f"{ops[c].name!r}(uid {c}, dev {ops[c].device}) <- {ops[d].name!r}(uid {d})"
            for c, d in pairs[:4]
        )
        warnings.warn(
            CrossRankDepWarning(
                f"{len(pairs)} dependency edge(s) cross ranks with no transfer "
                "carrying them and are NOT expressible as Chakra ctrl_deps; they "
                "are dropped from the emitted bundle. The ordering they encode is "
                "carried by the shared per-stage collectives (a collective cannot "
                f"complete until every member issues it). Examples: {sample}"
            ),
            stacklevel=2,
        )

    # --- Phase C: stable control-first renumber ----------------------------
    for trace in traces.values():
        trace.renumber_control_priority()

    # --- ALWAYS-ON POSTCONDITION: per-group collective sequences -----------
    _check_group_order_postcondition(group_records, gid_members, dp_count, ns_initial)

    # --- write ETs ----------------------------------------------------------
    for trace in traces.values():
        trace.write()

    et_prefix = os.path.join(output_dir, "llm_graph")
    rank_ids = sorted(traces.keys())

    manifest_path = _write_manifest(output_dir, traces, rank_ids)
    comm_groups, comm_groups_path = _write_comm_groups(output_dir, dp_count, rank_ids, gid_members)
    _write_comm_axes(output_dir, comm_groups, gid_axes, gid_labels)

    return EmittedBundle(
        et_prefix=et_prefix,
        rank_ids=rank_ids,
        manifest_path=manifest_path,
        comm_groups=comm_groups,
        comm_groups_path=comm_groups_path,
    )


def _check_group_order_postcondition(
    group_records: Dict[str, Dict[int, List[Tuple[Any, int, int]]]],
    gid_members: Dict[str, List[int]],
    dp_count: int,
    ns_initial: int,
) -> None:
    """AstraSim matches collectives within a communicator by PER-RANK ISSUE
    ORDER; if two members issue a group's collectives in different relative
    orders the simulation deadlocks silently (CONTEXT.md, verified). Assert
    (a) every rank a group's collectives were recorded on is a member of
    that group, and (b) every member rank's post-renumber (comm_type, size)
    sequence for each wire group is identical."""

    for gid, per_rank in group_records.items():
        members = gid_members.get(gid)
        if members is None:
            # dp stage group "stage_idx + 1": members across dp of one stage.
            stage_idx = int(gid) - 1
            members = [dp_idx * ns_initial + stage_idx for dp_idx in range(dp_count)]
        non_members = set(per_rank.keys()) - set(members)
        if non_members:
            # A collective stamped with a pg_name whose member list does not
            # include its own rank would never be compared by the sequence
            # check below — and deadlocks AstraSim silently at runtime.
            raise EmissionError(
                f"group-order postcondition violated for communicator group "
                f"{gid} (members {members}): collectives were recorded on "
                f"non-member ranks {sorted(non_members)}. Ops are stamped "
                "with a pg_name of a communicator their rank does not "
                "belong to, which deadlocks AstraSim silently."
            )
        sequences: Dict[int, List[Tuple[int, int]]] = {}
        for rank in members:
            records = per_rank.get(rank, [])
            records_sorted = sorted(records, key=lambda item: int(item[0].id))
            sequences[rank] = [(ctype, size) for _node, ctype, size in records_sorted]
        reference_rank = members[0]
        reference = sequences[reference_rank]
        for rank in members[1:]:
            if sequences[rank] != reference:
                detail = "; ".join(
                    f"rank {r}: {sequences[r]}" for r in members
                )
                raise EmissionError(
                    f"group-order postcondition violated for communicator group "
                    f"{gid} (members {members}): member ranks would issue this "
                    f"group's collectives in different orders, which deadlocks "
                    f"AstraSim silently. Sequences: {detail}"
                )


def _write_manifest(output_dir: str, traces: Dict[int, _Trace], rank_ids: List[int]) -> str:
    """Port of converter Step 12 (executor.py:1784-1866): per-rank op lists
    sorted by ``_manifest_op_key`` (the sort key fully determines each entry,
    so sorting post-renumber nodes is byte-identical to legacy's
    pre-renumber sort), dumped with ``sort_keys=True`` and compact
    separators — the AstraSim cache key."""

    def _manifest_op_key(op: List) -> tuple:
        try:
            kind = op[0]
        except Exception:
            kind = None
        if kind == "COMP":
            return (0, int(op[1]) if len(op) > 1 else 0, 0)
        if kind == "COMM":
            size_val = int(op[2]) if len(op) > 2 else 0
            ctype_val = int(op[1]) if len(op) > 1 else -1
            return (1, size_val, ctype_val)
        if kind == "SEND":
            return (2, int(op[1]) if len(op) > 1 else 0, 0)
        if kind == "RECV":
            return (3, int(op[1]) if len(op) > 1 else 0, 0)
        return (9, 0, 0)

    manifest_ranks: Dict[str, List[List]] = {}
    for rank, trace in sorted(traces.items()):
        rank_ops: List[List] = []
        for node in trace.nodes:
            t = int(node.type)
            if t == pb.COMP_NODE:
                rank_ops.append(["COMP", int(node.duration_micros or 0)])
            elif t == pb.COMM_COLL_NODE:
                ctype = None
                csize = None
                for attr in node.attr:
                    if attr.name == "comm_type":
                        which = attr.WhichOneof("value")
                        ctype = int(getattr(attr, which)) if which else None
                    elif attr.name == "comm_size":
                        which = attr.WhichOneof("value")
                        csize = int(getattr(attr, which)) if which else None
                rank_ops.append(["COMM", int(ctype or -1), int(csize or 0), None])
            elif t == pb.COMM_SEND_NODE:
                csize = None
                for attr in node.attr:
                    if attr.name == "comm_size":
                        which = attr.WhichOneof("value")
                        csize = int(getattr(attr, which)) if which else None
                rank_ops.append(["SEND", int(csize or 0)])
            elif t == pb.COMM_RECV_NODE:
                csize = None
                for attr in node.attr:
                    if attr.name == "comm_size":
                        which = attr.WhichOneof("value")
                        csize = int(getattr(attr, which)) if which else None
                rank_ops.append(["RECV", int(csize or 0)])
        manifest_ranks[str(int(rank))] = sorted(rank_ops, key=_manifest_op_key)

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as mf:
        json.dump(
            {
                "version": "df-astra-manifest/1",
                "npus": len(rank_ids),
                "ranks": manifest_ranks,
            },
            mf,
            sort_keys=True,
            separators=(",", ":"),
        )
    return manifest_path


def _write_comm_groups(
    output_dir: str,
    dp_count: int,
    rank_ids: List[int],
    gid_members: Dict[str, List[int]],
) -> Tuple[Dict[str, List[int]], Optional[str]]:
    """Port of ``_write_comm_groups_json`` (executor.py:671-714): dp stage
    groups (ids ``str(stage_idx + 1)`` — AstraSim requires ids > 0) only
    when dp > 1, then the interned wire groups (present even at dp == 1)."""
    dp = int(dp_count)
    if not rank_ids:
        return {}, None
    rank_ids = sorted(rank_ids)
    total_ranks = len(rank_ids)

    groups: Dict[str, List[int]] = {}
    if dp > 1:
        if total_ranks % dp != 0:
            raise ValueError(f"Cannot partition {total_ranks} ranks into {dp} stages evenly")
        num_stages = total_ranks // dp
        for stage_idx in range(num_stages):
            groups[str(stage_idx + 1)] = [
                dp_idx * num_stages + stage_idx for dp_idx in range(dp)
            ]

    for gid, members in gid_members.items():
        groups[str(gid)] = list(sorted(members))

    if not groups:
        return {}, None

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "comm_groups.json")
    with open(path, "w") as fh:
        json.dump(groups, fh, indent=2)
    return groups, path


def _write_comm_axes(
    output_dir: str,
    comm_groups: Dict[str, List[int]],
    gid_axes: Dict[str, Any],
    gid_labels: Dict[str, Any],
) -> Optional[str]:
    """Write the ``comm_axes.json`` SIDECAR: gid -> parallelism axes/labels.

    AstraSim never reads this file (``comm_groups.json`` stays exactly the
    ``{gid: [ranks]}`` map the binary is given). It exists so that consumers —
    the T1 structural gate's per-axis byte histogram, the viz/report layer —
    can attribute communication to an axis that construction already knew,
    instead of re-inferring it from participant counts
    (``legacy_lowering.py:243-248``, the inference the restructure deletes).

    Wire-group axes come from each collective's ``GroupKey.axis``; the dp
    stage groups (numeric ids below the wire base) are the dp communicators.
    """
    if not comm_groups:
        return None
    entries: Dict[str, Dict[str, List[str]]] = {}
    for gid in sorted(comm_groups, key=lambda g: (len(g), g)):
        axes = sorted(str(a) for a in gid_axes.get(gid, ()))
        labels = sorted(str(la) for la in gid_labels.get(gid, ()))
        if not axes:
            # A group with no recorded GroupKey is a dp stage group: its
            # members are the dp replicas of one stage (_write_comm_groups).
            axes = ["dp"]
        entries[str(gid)] = {"axes": axes, "labels": labels}
    path = os.path.join(output_dir, "comm_axes.json")
    with open(path, "w") as fh:
        json.dump(
            {"version": "df-astra-comm-axes/1", "groups": entries},
            fh,
            indent=2,
            sort_keys=True,
        )
    return path
