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
  producer — DESIGN §2.2). A SEND carries the union of ``transfer.producer``
  and every other dep of the transfer that resolves on the SEND's own rank —
  in particular the COMPUTE ANCHOR ``build()`` records beside the producer.
* **Phase B2** — cross-rank sync materialization: a dep whose two ends live on
  different ranks with no transfer carrying it cannot be a ``ctrl_dep`` (that
  is a node id in the SAME trace), so it goes ON THE WIRE as a 1-byte control
  SEND/RECV pair, tagged from ``_SYNC_TAG_BASE``. This is what the legacy
  lowering did (a zero-byte ``cross_layer_rank0`` ``TransferOp``), and it is
  what makes the analytical evaluator and AstraSim honor the SAME DAG. Nothing
  is dropped: an edge that cannot be carried raises :class:`EmissionError`.
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

#: p2p tag base for the SYNTHESIZED cross-rank sync pairs (§4.3 R4 amendment
#: 2026-07-29). A normal transfer's tag IS its ``TransferOp.uid``, so the base
#: has to sit above any uid a Program can reach; 10**6 is four orders of
#: magnitude above the largest program in the matrix and keeps the tags
#: readable in an AstraSim log.
_SYNC_TAG_BASE = 1_000_000


class EmissionError(RuntimeError):
    """Emission hit an inconsistency (or the group-order postcondition)."""


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

    layout = program.layout
    _dp_sibling_cache: Dict[Tuple[int, Tuple[str, ...]], Tuple[int, ...]] = {}

    def _intern_dp_group(siblings: Tuple[int, ...]) -> str:
        """Intern the communicator of a composite dp collective (BUG_LEDGER 19).

        Its members are the dp replicas of every sibling device — one group
        spanning dp, unlike a wire group, which lives inside a single dp
        replica. Interning is BY MEMBER SET, through the same
        ``members_to_gid``/``gid_members`` tables the wire groups use, so an id
        is never shared by two different member sets and
        ``_write_comm_groups`` publishes it without further help.
        """
        members = tuple(
            sorted(
                rank_for(device, dp_idx)
                for device in siblings
                for dp_idx in range(dp_count)
            )
        )
        gid = members_to_gid.get(members)
        if gid is None:
            gid = str(next(gid_counter))
            members_to_gid[members] = gid
            gid_members[gid] = list(members)
        return gid

    _dependents: Optional[set] = None

    def _has_dependents(uid: int) -> bool:
        """Does any op depend on ``uid``? Computed once, lazily.

        A dp collective that nothing waits on and that has no communicator can
        be dropped (the legacy Step-10 skip). One that IS waited on must still
        produce an ET node, or the dependent's edge dangles.
        """
        nonlocal _dependents
        if _dependents is None:
            found = set()
            for other in ops:
                for dep in getattr(other, "deps", ()):
                    found.add(int(dep))
                producer = getattr(other, "producer", None)
                if producer is not None:
                    found.add(int(producer))
                for consumer in getattr(other, "consumers", ()) or ():
                    found.add(int(consumer))
            _dependents = found
        return int(uid) in _dependents

    def _dp_group_devices(op: "CollectiveOp") -> Tuple[int, ...]:
        """The in-device-space members of a dp collective's communicator.

        ``('dp',)`` -> just the owning device: the group is its dp replicas, and
        that is the whole legacy behavior. ``('dp','cp')`` (BUG_LEDGER 19) -> the
        owning device AND its cp siblings, so the emitted communicator is the
        ``dp x cp`` group Megatron reduces gradients over.

        Membership is resolved from the DECLARED axes against the rank layout —
        never from ``op.participants`` (**K5**). If the layout cannot answer
        (a legacy program with an empty ``axis_order``), the sibling set
        degenerates to the device itself, which is the pre-19 behavior.
        """
        companions = tuple(a for a in (op.axes or ()) if a != "dp")
        key = (int(op.device), companions)
        cached = _dp_sibling_cache.get(key)
        if cached is not None:
            return cached
        # A companion axis that this granularity does not MATERIALIZE as devices
        # is not an error. PIPELINE devices are stages
        # (``_GRANULARITY_AXES[PIPELINE] == ("pp","dp")``), so ``cp`` has no device
        # extent there and the dp x cp reducer collapses to one collective whose
        # ``participants`` (dp*cp) still carries the group size to the analytical
        # evaluator — the same treatment PIPELINE already gives tp and ep.
        live = tuple(a for a in companions if a in layout.axis_order)
        if not live or not layout.axis_order:
            result = (int(op.device),)
        else:
            own = layout.coords_of(int(op.device))
            companions = live
            fixed = {a: v for a, v in own.items() if a not in companions}
            result = tuple(
                sorted(
                    d
                    for d in devices
                    if all(layout.coords_of(int(d)).get(a) == v for a, v in fixed.items())
                )
            )
            if int(op.device) not in result:  # pragma: no cover - defensive
                raise EmissionError(
                    f"Collective {op.name!r}: owning device {op.device} is not in its "
                    f"own communicator {result}"
                )
        _dp_sibling_cache[key] = result
        return result

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

    #: ``(consumer uid, dep uid)`` pairs whose ordering crosses ranks. A Chakra
    #: ``ctrl_dep`` cannot express them, so **Phase B2** materializes each one as
    #: a 1-byte control SEND/RECV pair — exactly what the legacy lowering did
    #: (``pipeline_fine`` created a zero-byte ``cross_layer_rank0`` TransferOp).
    #: Nothing is ever dropped: what cannot be carried raises.
    cross_rank_syncs: List[Tuple[int, int]] = []

    def _resolve_deps(
        dep_uids: Any, own_rank: int, dp_idx: int, *, owner: str, owner_uid: int
    ) -> List[int]:
        """``dep_uids`` -> ctrl_deps on ``own_rank``, per the ET's own rules.

        **A Chakra ``ctrl_dep`` is a node id in the SAME trace.** That is an ET
        fact, not an IR fact, so this is the one place that knows it, and it
        forces three cases:

        * a **cross-device TransferOp** dep is skipped HERE: the transfer has no
          ET node of its own, it materializes as a SEND/RECV pair in Phase B, and
          the pair's ids are wired into the consumers there;
        * a **same-device TransferOp** dep is ELIDED (never emitted, DESIGN §2.2)
          and must be REPLACED by a plain dep on what it depends on — its
          producer and, when present, the compute anchor. Dropping the edge
          instead lets the ranks of one stage drift out of collective-issue
          lockstep, which AstraSim reports as ``Hardware Resource ... has
          unreleased nodes``: a silent deadlock (measured; this is why the
          recursion exists);
        * a dep on an op on **another rank** with no transfer carrying it is not
          expressible as a ctrl_dep — writing a foreign rank's node id would
          mis-order the trace or close a cycle. It is recorded in
          ``cross_rank_syncs`` and Phase B puts it ON THE WIRE as a 1-byte
          control pair. Legacy did exactly that (a zero-byte ``cross_layer_rank0``
          ``TransferOp`` per cross-stage ZeRO-3 gather anchor — rows S8/S15), so
          the analytical evaluator and AstraSim see the SAME DAG.

        ``owner``/``owner_uid`` name the op whose deps these are, for errors.
        """
        deps: List[int] = []

        def _resolve(uid: int) -> None:
            dep_op = ops[uid]
            if isinstance(dep_op, TransferOp):
                if dep_op.src_device != dep_op.dst_device:
                    return  # Phase B wires the SEND/RECV ids
                if rank_for(dep_op.src_device, dp_idx) != own_rank:
                    cross_rank_syncs.append((owner_uid, uid))
                    return
                for inner in dep_op.deps:
                    _resolve(int(inner))
                return
            dep_rank = rank_for(dep_op.device, dp_idx)
            if dep_rank != own_rank:
                cross_rank_syncs.append((owner_uid, uid))
                return
            key = (uid, dep_rank)
            if key not in et_ids:
                raise EmissionError(
                    f"op {owner_uid} ('{owner}') depends on op {uid} "
                    f"('{dep_op.name}') which has no ET node on rank {dep_rank} "
                    "(skipped or not yet emitted)"
                )
            deps.append(et_ids[key])

        for dep in dep_uids:
            _resolve(int(dep))
        unique: List[int] = []
        for dep in deps:
            if dep not in unique:
                unique.append(dep)
        return unique

    def _resolved_deps(op: Any, dp_idx: int) -> List[int]:
        """Phase-A ctrl_deps of a main op."""
        return _resolve_deps(
            op.deps,
            rank_for(op.device, dp_idx),
            dp_idx,
            owner=op.name,
            owner_uid=op.uid,
        )

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
        singleton_dp = op.is_dp and len(_dp_group_devices(op)) * dp_count <= 1
        if singleton_dp and not _has_dependents(op.uid):
            # Legacy Step-10 skip, stated precisely: a dp collective with a
            # one-member communicator contributes nothing, and if it is a SINK
            # dropping it also loses no edge. Its own ``deps`` are irrelevant —
            # deleting a sink cannot orphan anything. Only a collective some
            # other op WAITS ON has to survive as a node (below).
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
            elif singleton_dp:
                # A one-member communicator cannot be a real collective — a
                # single-member native ring deadlocks, which is why the labeled
                # path above emits a no-op for the same shape. It is emitted
                # rather than skipped because ops DEPEND on it: at PIPELINE the cp
                # axis has no device extent, so a dp x cp reducer at ``dp == 1``
                # collapses to one device here while still being a real 2-rank
                # collective in the model. Skipping it outright left R5b's
                # ``optimizer -> reducer`` edge pointing at a node with no ET
                # id (BUG_LEDGER 19 residual). The analytical evaluator still
                # prices it from ``participants``; only the ET node is vacuous.
                noop = new_comp_node(node_id, f"{op.name}_{op.uid}_dp{dp_idx}_noop", 0)
                noop.ctrl_deps.extend(unique_deps)
                trace.append_node(noop)
                et_ids[(op.uid, rank)] = node_id
                continue
            else:
                comm_name = f"{op.name}_{op.uid}_dp{dp_idx}"
                comm_node = new_comm_node(node_id, comm_name, _collective_enum(op.coll), op.size_bytes)
                siblings = _dp_group_devices(op)
                if len(siblings) * dp_count > 1:
                    if len(siblings) == 1:
                        # Pure dp: the legacy stage group, id ``stage_idx + 1``.
                        # Untouched, so every cp == 1 program is byte-identical.
                        gid = str(device_index[op.device] + 1)
                    else:
                        # BUG_LEDGER 19 — a dp x cp communicator spans the dp
                        # replicas of EVERY cp sibling, so it is one group over
                        # all of them, not one group per device.
                        #
                        # It gets an INTERNED id from the wire-group space rather
                        # than ``device_index + 1``: at cp > 1 the ZeRO gathers
                        # are still pure-dp and keep the legacy per-device ids,
                        # so reusing ``min(sibling) + 1`` here would hand two
                        # DIFFERENT member sets the same id. Interning by member
                        # set is what makes that unrepresentable.
                        gid = _intern_dp_group(siblings)
                    comm_node.attr.append(pb.AttributeProto(name="pg_name", string_val=gid))
                    group_records[gid][rank].append(
                        (comm_node, _collective_enum(op.coll), int(op.size_bytes))
                    )
                    gid_axes[gid].update(op.axes or ("dp",))
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
                # A TransferOp's NON-producer deps are ordering constraints too,
                # and they land on the SEND's OWN rank — so this is a plain
                # omission if they are skipped, not an inexpressibility.
                # ``_emit_cross_layer`` (build.py) deliberately records a second
                # DATA_FLOW dep, the COMPUTE ANCHOR: "the SEND must fire off the
                # last COMPUTE, not off a trailing collective"
                # (``pipeline_fine.py:672-687``). At ``tp>1`` with a hoisted
                # tp/sp collective the producer IS that trailing collective and
                # the anchor is its SIBLING, so emitting only ``producer`` left
                # the inter-stage SEND with NO dependency on the compute that
                # produced the activation it carries — the layer's last COMPUTE
                # became a dependency SINK. Measured: 62-95% of the P5 T2
                # movement on every ``tp>1`` flattened spec.
                for extra in _resolve_deps(
                    transfer.deps,
                    src_rank,
                    dp_idx,
                    owner=transfer.name,
                    owner_uid=transfer.uid,
                ):
                    if extra not in node.ctrl_deps:
                        node.ctrl_deps.append(extra)
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

    # --- Phase B2: cross-rank sync deps go ON THE WIRE ----------------------
    #
    # A dep that crosses ranks with no transfer carrying it is NOT droppable:
    # measured on ``train:flattened:dp2tp1cp1pp2mb2sp0:zero3``, the 4 dropped
    # edges were the only thing ordering the ZeRO-3 parameter gathers, dropping
    # them made all 4 gathers graph ROOTS that issue at t=0, and the bundle's
    # wall time moved +1.366%. No collective spans the two pipeline stages, so
    # the "carried by the shared per-stage collectives" story cannot hold: the
    # gather's communicator is stage 1's dp pair and its dep is on stage 0.
    #
    # Legacy put them on the wire, so we put them on the wire: one 1-byte
    # control SEND on the dep's rank + RECV on the consumer's rank, per dp
    # replica. Deterministic in sorted (consumer, dep) order; the tags come from
    # ``_SYNC_TAG_BASE`` so they cannot collide with a TransferOp uid.
    for index, (consumer_uid, dep_uid) in enumerate(sorted(set(cross_rank_syncs))):
        consumer = ops[consumer_uid]
        dep_op = ops[dep_uid]
        src_device = (
            dep_op.src_device if isinstance(dep_op, TransferOp) else dep_op.device
        )
        # The CONSUMER may itself be a TransferOp: a cross-stage ZeRO-3 gather is
        # spliced in front of the anchor's DATA_FLOW successor, and at a stage
        # boundary that successor IS the ``cross_layer`` p2p on the OTHER stage
        # (rows S8/S15, ``ZeRO3._via`` -> ``VIA_DATA_FLOW``). A transfer starts
        # at its SEND, so that is the node the wire delays.
        consumer_is_transfer = isinstance(consumer, TransferOp)
        if consumer_is_transfer:
            if consumer.src_device == consumer.dst_device:
                raise EmissionError(
                    f"cross-rank sync into transfer {consumer_uid} "
                    f"('{consumer.name}') is impossible: the transfer is "
                    "same-device and therefore elided, so it has no ET node"
                )
            consumer_device = consumer.src_device
        else:
            consumer_device = consumer.device
        tag = _SYNC_TAG_BASE + index
        for dp_idx in range(dp_count):
            src_rank = rank_for(src_device, dp_idx)
            dst_rank = rank_for(consumer_device, dp_idx)
            if src_rank == dst_rank:  # pragma: no cover - defensive
                raise EmissionError(
                    f"cross-rank sync {consumer_uid} <- {dep_uid} resolved to the "
                    f"same rank {src_rank}; it should have been a ctrl_dep"
                )
            before = len(cross_rank_syncs)
            anchors = _resolve_deps(
                (dep_uid,), src_rank, dp_idx, owner=dep_op.name, owner_uid=dep_uid
            )
            if len(cross_rank_syncs) != before or not anchors:
                # A nested cross-rank carrier (the dep's own deps also cross a
                # rank) would need a chain of wires. No production Program has
                # one; make it LOUD rather than half-emitted.
                raise EmissionError(
                    f"cross-rank sync {consumer_uid} ('{consumer.name}') <- "
                    f"{dep_uid} ('{dep_op.name}') cannot be carried: resolving the "
                    f"dep on its own rank {src_rank} yielded {anchors!r} and "
                    f"{len(cross_rank_syncs) - before} further cross-rank dep(s). "
                    "Anchor the requirement on the consumer's device instead "
                    "(INTERFACES §4.4)."
                )
            send_trace = _trace_for(src_rank, f"cross-rank sync {consumer_uid} send")
            send_id = send_trace.next_id
            send_node = new_send_node(
                send_id,
                f"cross_rank_sync_{consumer_uid}_{dep_uid}_send_control",
                1,
                dst_rank,
                tag,
            )
            send_node.ctrl_deps.extend(anchors)
            send_trace.append_node(send_node, control=True)

            recv_trace = _trace_for(dst_rank, f"cross-rank sync {consumer_uid} recv")
            recv_id = recv_trace.next_id
            recv_node = new_recv_node(
                recv_id,
                f"cross_rank_sync_{consumer_uid}_{dep_uid}_recv_control",
                1,
                src_rank,
                tag,
            )
            recv_trace.append_node(recv_node, control=True)
            if consumer_is_transfer:
                target_id = send_ids.get((consumer_uid, dp_idx))
            else:
                target_id = et_ids.get((consumer_uid, dst_rank))
            if target_id is None:
                raise EmissionError(
                    f"cross-rank sync {consumer_uid} ('{consumer.name}') has no ET "
                    f"node on rank {dst_rank}"
                )
            consumer_node = recv_trace.nodes[target_id]
            if recv_id not in consumer_node.ctrl_deps:
                consumer_node.ctrl_deps.append(recv_id)

    # --- ALWAYS-ON POSTCONDITION: no op silently loses its successors ------
    _check_no_lost_successors(ops, et_ids, traces, dp_count)

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


def _check_no_lost_successors(
    ops: Any,
    et_ids: Dict[Tuple[int, int], int],
    traces: Dict[int, "_Trace"],
    dp_count: int,
) -> None:
    """ALWAYS-ON: an op with successors in the IR keeps them in the ET.

    The P5 cutover shipped an emitter that wired a cross-device SEND to
    ``transfer.producer`` alone, so the transfer's OTHER dep — the compute
    anchor — vanished and the layer's last COMPUTE became a dependency SINK on
    every ``tp > 1`` spec. Nothing caught it: the op multisets, the byte
    histograms, ``dlsim`` and the group-order postcondition are all indifferent
    to a lost edge, and only the AstraSim wall clock moved (by up to 5.9%).

    This is the check that would have caught it, stated where the loss happens.
    ``validate.py``'s **V6** asks the same question of the IR and only about
    collectives; this asks it of the EMITTED trace and about every op.

    A genuine sink (a gradient reducer, the optimizer, a SINK collective) has no
    successors in the IR either, so it is not flagged: the invariant is
    *preservation*, not "everything has a successor".
    """
    # Which ops MUST still have a successor in the trace? Everything some
    # EMITTED op (or some wire) depends on. Deps of an op the emitter legally
    # skips do not count — a dp collective at ``dp_count <= 1`` (legacy Step 10)
    # is gone entirely, so its producer legitimately ends up a sink — and a
    # same-device transfer is elided, so its deps propagate to whatever depends
    # on the transfer.
    has_ir_successor: Set[int] = set()
    pending: List[int] = []

    def _mark(dep_uids: Any) -> None:
        for raw in dep_uids:
            dep = int(raw)
            if dep in has_ir_successor:
                continue
            has_ir_successor.add(dep)
            dep_op = ops[dep]
            if (
                isinstance(dep_op, TransferOp)
                and dep_op.src_device == dep_op.dst_device
            ):
                pending.append(dep)

    for uid, _rank in et_ids:
        _mark(ops[uid].deps)
    for op in ops:
        if isinstance(op, TransferOp) and op.src_device != op.dst_device:
            _mark(op.deps)
    while pending:
        _mark(ops[pending.pop()].deps)

    wired: Dict[int, Set[int]] = {
        rank: {int(dep) for node in trace.nodes for dep in node.ctrl_deps}
        for rank, trace in traces.items()
    }
    lost: List[str] = []
    for (uid, rank), node_id in sorted(et_ids.items()):
        if uid not in has_ir_successor:
            continue
        if node_id in wired.get(rank, ()):
            continue
        op = ops[uid]
        lost.append(f"{op.name!r} (uid {uid}, rank {rank}, node {node_id})")
    if lost:
        raise EmissionError(
            f"{len(lost)} op(s) have successors in the Program but NONE in the "
            "emitted trace — an ordering constraint was dropped at emission, "
            "which no other gate can see (it moves only the AstraSim wall "
            f"clock). Examples: {', '.join(lost[:6])}"
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
