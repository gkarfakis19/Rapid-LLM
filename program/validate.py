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

"""Program invariants V1-V6 (design C §1.2, DESIGN.md §2 amendments).

``validate_program`` raises :class:`ProgramInvariantError` on the fatal
invariants V1-V5 and emits a non-fatal :class:`GroupRaceWarning` for V6.

V6 scope note: every LEGACY production construction path calls
``validate_program(..., check_races=False)`` (legacy_lowering, et_emit,
transforms) — the O(paths) race scan is a test-time/builder-time
diagnostic only (``ProgramBuilder.finish`` and unit tests enable it). The
85894c6 deadlock class it describes is enforced in production wire-level
by the always-on emission postcondition in :mod:`program.et_emit`.
:func:`program.build.build` promotes it to ALWAYS-ON (INTERFACES §4.8), which
exposes a known false-positive class: a gradient reducer is a SINK by
construction (it hangs off a backward exit and nothing follows it), so any two
reducers of one group on one device trip V6 even though the program's order is
ONE global order projected onto every device. A caller building in bulk should
silence :class:`GroupRaceWarning`.

V3 is relaxed per DESIGN §2.2: a ``TransferOp`` with ``src_device ==
dst_device`` is legal (legacy same-stage PIPELINE edges; the ET emitter
elides them into plain deps, the analytical evaluator will enqueue them).
Observed-reality amendment to the DESIGN wording: flattened graphs carry
same-stage ``cross_layer`` edges with NONZERO sizes (per-rank clones of
cross-GPU edges keep their byte counts), so same-device transfers may carry
any size — the size is preserved for the M5 evaluator and never emitted.

V5 note (deviation from the design C wording, see program/ir.py module
docstring): legacy-lowered programs are *per-device clone* programs, so the
strong "every member device sees the group's collectives" property is a
wire-level property of gid interning, enforced by the always-on emission
postcondition in :mod:`program.et_emit`. Here V5 asserts the structural
consistency the emitter relies on: labeled <=> grouped <=> not dp-flagged,
and every grouped op's device is one of its group's members.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from program.ir import CollectiveOp, ComputeOp, GroupKey, Program, TransferOp
from timing_model import CollectiveType


class ProgramInvariantError(ValueError):
    """A Program violated one of the fatal invariants V1-V5."""


class GroupRaceWarning(UserWarning):
    """V6: two same-group collectives on one device have no ordering path."""


def _fail(invariant: str, message: str) -> None:
    raise ProgramInvariantError(f"{invariant}: {message}")


def validate_program(
    program: Program,
    *,
    check_races: bool = True,
    check_group_membership: bool = False,
) -> None:
    """Check V1-V5 + V8 (fatal), V6 (warning) and V7 (fatal, opt-in).

    ``check_group_membership`` enables **V7** — "every grouped ``CollectiveOp``
    is instantiated on EVERY member device of its group" (INTERFACES §4.8).
    It is opt-in because legacy-lowered programs are *per-device clone* programs:
    a labeled collective is emitted only on its own device and the communicator
    is formed by the isomorphic clones on the other members (see
    ``program/ir.py``'s module docstring), so the strong form is false there BY
    CONSTRUCTION. ``program.build`` materializes one op per member device and
    therefore passes it — which is what makes ``ext_moe_flat.md`` blocker 5 a
    build-time error instead of a silent AstraSim deadlock.

    **V8** is always on and costs nothing on legacy programs: it fires only when
    a ``TransferOp`` carries a ``moe_component``, which no legacy path sets.
    """

    ops = program.ops
    devices: Set[int] = set(program.devices)
    if len(devices) != len(program.devices):
        _fail("V1", "Program.devices contains duplicate device ids")
    dp_count = max(int(program.dp_count), 1)

    # ---------------- V1: dense uids, topological deps ----------------
    for idx, op in enumerate(ops):
        if op.uid != idx:
            _fail("V1", f"ops[{idx}].uid == {op.uid}; uids must be dense list indices")
        for dep in op.deps:
            if not (0 <= int(dep) < op.uid):
                _fail(
                    "V1",
                    f"op {op.uid} ('{op.name}') has dep {dep}; every dep uid must "
                    f"be a valid uid < {op.uid}",
                )
        if isinstance(op, TransferOp):
            if not (0 <= op.producer < op.uid):
                _fail(
                    "V1",
                    f"TransferOp {op.uid} ('{op.name}') producer {op.producer} must "
                    f"be a valid uid < {op.uid}",
                )
            for consumer in op.consumers:
                if not (0 <= int(consumer) < len(ops)):
                    _fail("V1", f"TransferOp {op.uid} consumer {consumer} is not a valid uid")
                if isinstance(ops[int(consumer)], TransferOp):
                    _fail("V1", f"TransferOp {op.uid} consumer {consumer} is itself a TransferOp")

    # ---------------- V2: groups registered, device in members ----------------
    for op in ops:
        if isinstance(op, CollectiveOp):
            if op.coll == CollectiveType.PIPELINE:
                _fail("V2", f"CollectiveOp {op.uid} ('{op.name}') has PIPELINE comm type")
            if op.group is not None:
                registered = program.groups.get(op.group)
                if registered is None:
                    _fail(
                        "V2",
                        f"CollectiveOp {op.uid} ('{op.name}') references unregistered "
                        f"group {op.group}",
                    )
                if op.device not in op.group.members:
                    _fail(
                        "V2",
                        f"CollectiveOp {op.uid} ('{op.name}') on device {op.device} is "
                        f"not a member of its group {op.group.members}",
                    )
                if not set(op.group.members) <= devices:
                    _fail(
                        "V2",
                        f"group {op.group} of op {op.uid} has members outside "
                        f"Program.devices {sorted(devices)}",
                    )
    for key in program.groups:
        if tuple(sorted(key.members)) != key.members:
            _fail("V2", f"GroupKey members must be sorted: {key}")

    # ---------------- V3: transfer endpoint discipline (relaxed) ----------------
    for op in ops:
        if not isinstance(op, TransferOp):
            continue
        if op.src_device not in devices or op.dst_device not in devices:
            _fail(
                "V3",
                f"TransferOp {op.uid} ('{op.name}') endpoints "
                f"({op.src_device} -> {op.dst_device}) must be in Program.devices",
            )
        producer = ops[op.producer]
        if isinstance(producer, TransferOp):
            _fail("V3", f"TransferOp {op.uid} producer {op.producer} is a TransferOp")
        if producer.device != op.src_device:
            _fail(
                "V3",
                f"TransferOp {op.uid} ('{op.name}') producer {op.producer} lives on "
                f"device {producer.device}, not src_device {op.src_device}",
            )
        for consumer_uid in op.consumers:
            consumer = ops[int(consumer_uid)]
            if consumer.device not in (op.src_device, op.dst_device):
                _fail(
                    "V3",
                    f"TransferOp {op.uid} consumer {consumer_uid} on device "
                    f"{consumer.device} is on neither endpoint "
                    f"({op.src_device} -> {op.dst_device})",
                )

    # ---------------- V4: per-DP duration profile length ----------------
    for op in ops:
        if isinstance(op, ComputeOp):
            if len(op.duration) not in (1, dp_count):
                _fail(
                    "V4",
                    f"ComputeOp {op.uid} ('{op.name}') duration has length "
                    f"{len(op.duration)} but dp_count={dp_count} (must be 1 or dp_count)",
                )

    # ---------------- V5: emitter dispatch consistency ----------------
    for op in ops:
        if not isinstance(op, CollectiveOp):
            continue
        if (op.label is None) != bool(op.is_dp):
            _fail(
                "V5",
                f"CollectiveOp {op.uid} ('{op.name}'): is_dp={op.is_dp} but "
                f"label={op.label!r} (is_dp must hold exactly when unlabeled)",
            )
        if op.group is not None and op.label is None:
            _fail(
                "V5",
                f"CollectiveOp {op.uid} ('{op.name}') has a group but no label; "
                "wire gid interning is label-driven",
            )
        if op.label is not None and op.group is None:
            _fail(
                "V5",
                f"CollectiveOp {op.uid} ('{op.name}') has label {op.label!r} but "
                "no group",
            )

    # ---------------- V7: a grouped collective spans its group ----------
    if check_group_membership:
        _check_group_instantiation(program)

    # ---------------- V8: same-device MoE p2p with bytes ----------------
    for op in ops:
        if not isinstance(op, TransferOp):
            continue
        if (
            op.src_device == op.dst_device
            and int(op.size_bytes) > 0
            and getattr(op, "moe_component", None)
        ):
            _fail(
                "V8",
                f"TransferOp {op.uid} ('{op.name}') is a same-device p2p carrying "
                f"{op.size_bytes} bytes for MoE component "
                f"{op.moe_component!r}; same-device transfers are elided at "
                "emission, so this silently DROPS the payload "
                "(ext_moe_flat.md blocker 4)",
            )

    # ---------------- V6: group-race warning (non-fatal) ----------------
    if check_races:
        _warn_group_races(program)


def _check_group_instantiation(program: Program) -> None:
    """V7: every grouped ``CollectiveOp`` exists on every member of its group.

    AstraSim matches a communicator's collectives by per-rank ISSUE ORDER, so a
    group whose member device never issues the collective deadlocks that group
    (``et_emit`` catches it at emission; this catches it at build time). The
    check is per (group, label, size, coll): a requirement instantiated on a
    subset of its group's members is the failure this names.
    """
    seen: Dict[Tuple[GroupKey, str, int, object], Set[int]] = defaultdict(set)
    for op in program.ops:
        if isinstance(op, CollectiveOp) and op.group is not None:
            key = (op.group, str(op.label), int(op.size_bytes), op.coll)
            seen[key].add(int(op.device))
    for (group, label, size, coll), devices in seen.items():
        missing = sorted(set(group.members) - devices)
        if missing:
            _fail(
                "V7",
                f"collective '{label}' ({coll.name}, {size} bytes) of group "
                f"{group.axis}{list(group.members)} is instantiated on "
                f"{sorted(devices)} but NOT on {missing}; every member device of "
                "a communicator must issue the group's collectives or that "
                "group deadlocks",
            )


def _warn_group_races(program: Program) -> None:
    """Warn for consecutive same-group collectives on one device with no
    dependency path between them (the 85894c6 bug class: their relative
    order is a toposort accident that clone devices may resolve differently).

    NARROWED (P5, INTERFACES §4.8): a pair is only reported when the EARLIER
    collective has a successor. V6 is always-on in ``build()`` now, and a
    gradient reducer is a graph SINK by construction (it hangs off a backward
    exit and nothing follows it), so every pair of reducers of one group on one
    device tripped the old form. For a program whose order is ONE global order
    projected onto every device (CONTEXT constraint 2) a sink pair cannot race:
    nothing observes which of the two issued first. A pair where the earlier op
    HAS a successor is the real race and is still reported.
    """

    ops = program.ops
    # Reverse adjacency: uid -> dep uids (deps, plus the transfer wiring:
    # consumer <- transfer <- producer).
    consumer_transfers: Dict[int, List[TransferOp]] = defaultdict(list)
    for op in ops:
        if isinstance(op, TransferOp):
            for consumer in op.consumers:
                consumer_transfers[int(consumer)].append(op)

    def preds(uid: int) -> List[int]:
        op = ops[uid]
        if isinstance(op, TransferOp):
            return [op.producer]
        result = [int(d) for d in op.deps]
        result.extend(t.producer for t in consumer_transfers.get(uid, ()))
        return result

    def has_path(src: int, dst: int) -> bool:
        """Is there a dependency path src -> dst (src < dst)?"""
        stack = [dst]
        seen: Set[int] = set()
        while stack:
            cur = stack.pop()
            if cur == src:
                return True
            if cur in seen or cur < src:
                continue
            seen.add(cur)
            stack.extend(preds(cur))
        return False

    per_device_group: Dict[Tuple[GroupKey, int], List[int]] = defaultdict(list)
    for op in ops:
        if isinstance(op, CollectiveOp) and op.group is not None:
            per_device_group[(op.group, op.device)].append(op.uid)

    succ_count: Dict[int, int] = defaultdict(int)
    for op in ops:
        for dep in op.deps:  # a TransferOp's deps include its producer
            succ_count[int(dep)] += 1

    for (group, device), uids in per_device_group.items():
        for earlier, later in zip(uids, uids[1:]):
            if not succ_count[earlier]:
                continue  # a SINK collective cannot race (see the docstring)
            if not has_path(earlier, later):
                warnings.warn(
                    GroupRaceWarning(
                        f"V6: collectives {earlier} ('{ops[earlier].name}') and "
                        f"{later} ('{ops[later].name}') of group {group} on device "
                        f"{device} have no dependency path between them; their "
                        "relative issue order is a scheduling accident that member "
                        "devices may resolve differently (AstraSim matches "
                        "collectives by per-rank issue order)"
                    ),
                    stacklevel=3,
                )
