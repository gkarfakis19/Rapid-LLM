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

"""The Program IR — typed placed operations (DESIGN.md §2, design C §1.2).

A :class:`Program` is a list of placed operations (``ComputeOp`` /
``CollectiveOp`` / ``TransferOp``) whose list index *is* the operation uid.
Uids are dense (``ops[uid].uid == uid``) and uid order is THE global total
order: ``kahn(slot, device, intra)`` as assigned by :func:`program.build.build`
(INTERFACES §4.6). It is the emitter's id policy, the analytical/memory
replays' tie discipline, and AstraSim's node-id priority — one order, declared
once. **No op carries ordering metadata**: ``legacy_op_id``, ``post_deps``,
``send_seq``, ``recv_seq`` and ``legacy_tag`` are deleted along with the legacy
lowering that wrote them (P5).

Amendments from DESIGN.md §2 relative to the panel document:

* same-placement TransferOps are legal (``src_device == dst_device``); the
  ET emitter elides them into plain deps while the analytical evaluator
  (:mod:`program.analytic_sim`) enqueues them (DESIGN §2.2 — extended: real
  flattened graphs carry same-stage ``cross_layer`` edges with nonzero
  sizes, preserved here);
* per-DP durations are first class: ``ComputeOp.duration`` is a tuple of
  length 1 or ``Program.dp_count`` (DESIGN §2.4, legacy ``duration_profile``);
* ``Program.devices`` includes devices seen only via collectives, and
  ``meta.misc["num_stages_initial"]`` preserves the converter's
  *pre-extension* stage count for the dp-major rank arithmetic
  (DESIGN §2.5, executor.py Step 7 quirk);
* wire group-id allocation is label-sorted and lives in the emitter
  (DESIGN §2.6); the IR only carries :class:`GroupKey`/:class:`CommGroup`
  plus the per-op ``label``.

Deviation from design C §1.2, forced by the legacy converter's shape and
documented here once:

* Programs are *per-device clone* programs: a labeled collective op is emitted
  only on its own device, and the communicator is formed by the isomorphic ops
  on the other member devices sharing the same label. ``CollectiveOp.label``
  therefore lives on the op (two distinct labels may share one ``GroupKey``;
  the emitter deduplicates member sets when interning wire gids).
* ``TransferOp`` carries its consumer wiring explicitly (``consumers``): the
  emitter wires the RECV id into dst-side consumers and the SEND id into
  src-side ones, and one transfer is one identity (one tag).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from timing_model import CollectiveType

from program.layout import RankLayout

OpUid = int
DeviceId = int


class OpRole(Enum):
    """Semantic role vocabulary (design A §1); ``GENERIC`` for lowered ops."""

    GENERIC = auto()
    TRANSFORMER_LAYER = auto()
    EMBEDDING = auto()
    SOFTMAX = auto()
    OPTIMIZER = auto()
    # GEMM/JOIN are not stamped by any current builder (lowered ops stay
    # GENERIC; coarse ops use the five roles above) — reserved API surface
    # for the post-M9 typed block programs (DESIGN §5 M9).
    GEMM = auto()
    JOIN = auto()


class Direction(Enum):
    FORWARD = auto()
    BACKWARD = auto()


@dataclass(frozen=True)
class GroupKey:
    """Pre-DP communicator identity: an axis plus its member *devices*.

    ``members`` are sorted device ids (stages). DP replication is stamped at
    emission: the wire members for dp index ``d`` are
    ``sorted(rank_for(device, d) for device in members)``.
    """

    axis: str
    members: Tuple[DeviceId, ...]


@dataclass
class CommGroup:
    """Registered communicator group; ``label`` is diagnostics-only here (the
    authoritative label for gid interning is per-op, see module docstring)."""

    key: GroupKey
    label: str


@dataclass
class ComputeOp:
    uid: OpUid
    name: str
    device: DeviceId
    duration: Tuple[float, ...]  # length 1 or dp_count (V4)
    deps: Tuple[OpUid, ...] = ()
    role: OpRole = OpRole.GENERIC
    direction: Direction = Direction.FORWARD
    mem_kind: Optional[Any] = None
    recompute: bool = False
    param_gather: bool = False
    micro_batch: Optional[int] = None
    layer: Optional[int] = None
    is_moe_layer: bool = False
    #: INTERFACES §4.7 — the stable SEMANTIC identity (a ``program.work.WorkItem``)
    #: that ``retime`` / ``memory_sim`` / fault projection look ops up by,
    #: replacing positional mirroring and name-prefix walks.
    work: Optional[Any] = None


@dataclass
class CollectiveOp:
    uid: OpUid
    name: str
    device: DeviceId  # owning device (legacy "edge stage")
    coll: CollectiveType  # never PIPELINE
    #: INTERFACES §4.7: FLOAT. dp reducer sizes are floats and must stay floats
    #: (``analytic_sim.py:49-51``); the ``int()`` truncation belongs to the
    #: emitter, where AstraSim needs it. Legacy-lowered programs still store an
    #: int here (``int()`` applied at construction), which is the truncation this
    #: field type stops mandating.
    size_bytes: float
    participants: int = 0
    interconnect: Optional[str] = None
    #: legacy dp-collective flag: skipped entirely at emission when
    #: ``dp_count <= 1`` (executor.py Step 10). True iff ``label is None``.
    is_dp: bool = False
    #: label from ``_assign_collective_labels`` ((base_name, primary member
    #: set) -> label); None for dp/pp-interconnect collectives.
    label: Optional[str] = None
    group: Optional[GroupKey] = None
    deps: Tuple[OpUid, ...] = ()
    #: the ``CommSpecTable`` key that DECLARED this collective. Consumers use it
    #: to ask which table a collective came from — in particular the memory
    #: replay, which times the PIPELINE-LEVEL collectives (``WorkloadSpec.comm``)
    #: and leaves the block-template ones untimed (INTERFACES §1.6 amendment B1
    #: is what makes those two namespaces distinct in the first place).
    comm_key: Optional[str] = None
    #: INTERFACES §4.7 — the DECLARED communicator axes (``CommSpec.axes``), for
    #: diagnostics and V2/V7. Never inferred from a participant count.
    axes: Tuple[str, ...] = ()
    #: INTERFACES §4.7 — stable semantic identity; see ``ComputeOp.work``.
    work: Optional[Any] = None


@dataclass
class TransferOp:
    """One logical p2p transfer: both endpoints, one identity (= one tag).

    ``size_bytes == 0`` or a non-PIPELINE ``comm_type`` marks a *control*
    transfer (:attr:`is_control`): emitted as a 1-byte
    ``*_send_control``/``*_recv_control`` pair, and Phase C of the emitter
    partitions on the CARRIED decision, never on the name it wrote. A
    same-device transfer (``src_device == dst_device``) is never emitted — the
    consumer already depends on the producer directly — but exists because the
    presence of the zero-byte same-stage PIPELINE event is load-bearing for the
    analytical evaluator's ready scan (Class B item 9, DESIGN §2.2).
    """

    uid: OpUid
    name: str
    src_device: DeviceId
    dst_device: DeviceId
    size_bytes: int
    comm_type: Optional[CollectiveType]
    producer: OpUid
    deps: Tuple[OpUid, ...] = ()
    #: main ops whose ET nodes receive this transfer's RECV id (consumer on
    #: dst_device) or SEND id (consumer on src_device) as a ctrl dep.
    consumers: Tuple[OpUid, ...] = ()
    #: the declaring ``CommSpec.moe_component`` when this p2p came from a MoE
    #: routing group. Carried so **V8** can reject a same-device
    #: ``residual_p2p`` with bytes instead of shrugging at it
    #: (``ext_moe_flat.md`` blocker 4).
    moe_component: Optional[str] = None
    #: AMENDMENT 2026-07-28 (P5/P6) — the declaring ``CommSpec``'s ANALYTICAL
    #: timing surface, carried so the analytical evaluator can time a p2p from
    #: the op instead of re-deriving it from the op's name. ``participants`` is
    #: the analytical participant count (2 for ``cross_layer``) and
    #: ``interconnect`` the axis key into ``WorkloadSpec.interconnect``. These
    #: are the two ``CommEvent`` fields the deleted proto graph carried that no
    #: other IR field expresses; see :mod:`program.analytic_sim`.
    participants: int = 0
    interconnect: Optional[str] = None

    @property
    def is_control(self) -> bool:
        return self.size_bytes == 0 or self.comm_type != CollectiveType.PIPELINE


Op = Union[ComputeOp, CollectiveOp, TransferOp]


@dataclass
class ProgramMeta:
    label: str = ""
    misc: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Program:
    #: layout the device ids live in (may have an empty axis_order when the
    #: legacy graph carried no ``_astrasim_rank_layout``).
    layout: RankLayout
    dp_count: int
    #: devices in EMISSION order (legacy final ``stage_ids``: sorted compute
    #: stages, then collective-only stages in discovery order, permuted by
    #: the SCOTCH remap when active). ``devices.index(d)`` is the legacy
    #: ``stage_index[d]``.
    devices: Tuple[DeviceId, ...]
    ops: List[Op]
    groups: Dict[GroupKey, CommGroup]
    meta: ProgramMeta

    # -- derived helpers -------------------------------------------------
    def num_stages_initial(self) -> int:
        """Pre-extension stage count used by the dp-major rank formula."""
        return int(self.meta.misc.get("num_stages_initial", len(self.devices)))

    def compute_devices(self) -> Tuple[DeviceId, ...]:
        """Devices that own ET traces (legacy Step-2 stages), emission order."""
        stored = self.meta.misc.get("compute_devices")
        if stored is None:
            return self.devices
        return tuple(stored)

    def device_index(self) -> Dict[DeviceId, int]:
        return {device: idx for idx, device in enumerate(self.devices)}

    def rank_for(self, device: DeviceId, dp_idx: int) -> int:
        """dp-major rank: ``dp_idx * num_stages_initial + stage_index``.

        Uses the *pre-extension* stage count on purpose (executor.py Step 7:
        collective-only stages extend ``stage_to_ranks`` with ranks computed
        from the original ``num_stages``) — DESIGN §2.5.
        """
        return dp_idx * self.num_stages_initial() + self.device_index()[device]

    def validate(self) -> None:
        from program.validate import validate_program

        validate_program(self)


class ProgramBuilder:
    """Append-only builder: ``add_*`` returns the op's uid; uids are the
    creation sequence, which therefore *is* the program's total order.

    ``program.build.build()`` is the production construction path and builds
    ``Program`` directly (it needs Kahn-ordered uids, which an append-only
    builder cannot express); this class remains the small hand-construction API
    used by the IR/emitter unit tests."""

    def __init__(
        self,
        layout: Optional[RankLayout] = None,
        dp_count: int = 1,
        meta: Optional[ProgramMeta] = None,
    ) -> None:
        self.layout = layout if layout is not None else RankLayout((), {}, {})
        self.dp_count = max(int(dp_count), 1)
        self.meta = meta if meta is not None else ProgramMeta()
        self._ops: List[Op] = []
        self._groups: Dict[GroupKey, CommGroup] = {}

    # -- ops -------------------------------------------------------------
    def _next_uid(self) -> OpUid:
        return len(self._ops)

    def add_compute(
        self,
        name: str,
        device: DeviceId,
        duration: Union[float, Sequence[float]],
        deps: Sequence[OpUid] = (),
        **fields: Any,
    ) -> OpUid:
        if isinstance(duration, (int, float)):
            duration_tuple = (float(duration),)
        else:
            duration_tuple = tuple(float(v) for v in duration)
        op = ComputeOp(
            uid=self._next_uid(),
            name=name,
            device=int(device),
            duration=duration_tuple,
            deps=tuple(deps),
            **fields,
        )
        self._ops.append(op)
        return op.uid

    def add_collective(
        self,
        name: str,
        device: DeviceId,
        coll: CollectiveType,
        size_bytes: int,
        deps: Sequence[OpUid] = (),
        **fields: Any,
    ) -> OpUid:
        op = CollectiveOp(
            uid=self._next_uid(),
            name=name,
            device=int(device),
            coll=coll,
            size_bytes=int(size_bytes),
            deps=tuple(deps),
            **fields,
        )
        self._ops.append(op)
        return op.uid

    def add_transfer(
        self,
        name: str,
        src_device: DeviceId,
        dst_device: DeviceId,
        size_bytes: int,
        producer: OpUid,
        comm_type: Optional[CollectiveType] = CollectiveType.PIPELINE,
        consumers: Sequence[OpUid] = (),
        **fields: Any,
    ) -> OpUid:
        op = TransferOp(
            uid=self._next_uid(),
            name=name,
            src_device=int(src_device),
            dst_device=int(dst_device),
            size_bytes=int(size_bytes),
            comm_type=comm_type,
            producer=int(producer),
            deps=(int(producer),),
            consumers=tuple(consumers),
            **fields,
        )
        self._ops.append(op)
        return op.uid

    def op(self, uid: OpUid) -> Op:
        return self._ops[uid]

    # -- groups ----------------------------------------------------------
    def group(self, axis: str, members: Sequence[DeviceId], label: str) -> GroupKey:
        key = GroupKey(axis=str(axis), members=tuple(sorted(int(m) for m in members)))
        if key not in self._groups:
            self._groups[key] = CommGroup(key=key, label=label)
        return key

    # -- finish ----------------------------------------------------------
    def finish(
        self,
        devices: Optional[Sequence[DeviceId]] = None,
        *,
        num_stages_initial: Optional[int] = None,
        compute_devices: Optional[Sequence[DeviceId]] = None,
        validate: bool = True,
    ) -> Program:
        if devices is None:
            seen: List[DeviceId] = []
            for op in self._ops:
                if isinstance(op, TransferOp):
                    candidates = (op.src_device, op.dst_device)
                else:
                    candidates = (op.device,)
                for device in candidates:
                    if device not in seen:
                        seen.append(device)
            for key in self._groups:
                for device in key.members:
                    if device not in seen:
                        seen.append(device)
            devices = sorted(seen)
        devices_tuple = tuple(int(d) for d in devices)
        misc = dict(self.meta.misc)
        misc["num_stages_initial"] = (
            int(num_stages_initial) if num_stages_initial is not None else len(
                tuple(compute_devices) if compute_devices is not None else devices_tuple
            )
        )
        if compute_devices is not None:
            misc["compute_devices"] = tuple(int(d) for d in compute_devices)
        meta = ProgramMeta(label=self.meta.label, misc=misc)
        program = Program(
            layout=self.layout,
            dp_count=self.dp_count,
            devices=devices_tuple,
            ops=list(self._ops),
            groups=dict(self._groups),
            meta=meta,
        )
        if validate:
            program.validate()
        return program
