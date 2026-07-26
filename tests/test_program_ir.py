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

"""Unit tests for the M1 Program IR stack (ir / validate / et_emit /
legacy_lowering quirks). Pure and fast: no subprocesses, no AstraSim binary —
emitted ETs are decoded with ``astrasim_lib.et_utils`` and compared in
memory.

Run: ./.venv/bin/python -m pytest tests/test_program_ir.py -q
"""

from __future__ import annotations

import json
import warnings

import pytest

from astrasim_lib.et_utils import (
    chakra_decode,
    chakra_open,
    new_comm_node,
    new_comp_node,
    new_recv_node,
    new_send_node,
    pb,
)
from program.et_emit import EmissionError, emit_chakra
from program.ir import (
    CollectiveOp,
    CommGroup,
    ComputeOp,
    GroupKey,
    Program,
    ProgramBuilder,
    ProgramMeta,
    TransferOp,
)
from program.layout import RankLayout
from program.validate import (
    GroupRaceWarning,
    ProgramInvariantError,
    validate_program,
)
from timing_model import CollectiveType

AR = CollectiveType.ALL_REDUCE
PIPE = CollectiveType.PIPELINE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_layout() -> RankLayout:
    return RankLayout(axis_order=(), axis_sizes={}, axis_strides={})


def _program(ops, devices, dp_count=1, groups=None, num_stages_initial=None):
    return Program(
        layout=_empty_layout(),
        dp_count=dp_count,
        devices=tuple(devices),
        ops=list(ops),
        groups=dict(groups or {}),
        meta=ProgramMeta(
            misc={
                "num_stages_initial": (
                    num_stages_initial if num_stages_initial is not None else len(devices)
                )
            }
        ),
    )


def _load_nodes(path):
    fh = chakra_open(path)
    try:
        meta = pb.GlobalMetadata()
        assert chakra_decode(fh, meta)
        nodes = []
        while True:
            node = pb.Node()
            if not chakra_decode(fh, node):
                break
            nodes.append(node)
        return nodes
    finally:
        fh.close()


def _attrs(node):
    out = {}
    for attr in node.attr:
        which = attr.WhichOneof("value")
        if which is None:
            continue
        raw = getattr(attr, which)
        out[attr.name] = list(raw.values) if hasattr(raw, "values") else raw
    return out


def _node_view(node):
    """Full comparison view of one ET node (ids, name, payload, deps)."""
    return (
        int(node.id),
        node.name,
        int(node.type),
        int(node.duration_micros or 0),
        tuple(sorted(_attrs(node).items())),
        tuple(sorted(int(d) for d in node.ctrl_deps)),
    )


# ---------------------------------------------------------------------------
# Builder + validator invariants
# ---------------------------------------------------------------------------


def test_builder_uids_are_creation_order():
    b = ProgramBuilder(dp_count=1)
    u0 = b.add_compute("a", 0, 1e-6)
    u1 = b.add_compute("b", 1, 1e-6, deps=[u0])
    u2 = b.add_transfer("t", 0, 1, 8, producer=u0, consumers=[u1])
    prog = b.finish()
    assert [op.uid for op in prog.ops] == [0, 1, 2]
    assert (u0, u1, u2) == (0, 1, 2)
    assert prog.devices == (0, 1)


def test_v1_rejects_forward_dep():
    ops = [
        ComputeOp(uid=0, name="a", device=0, duration=(1e-6,), deps=(1,)),
        ComputeOp(uid=1, name="b", device=0, duration=(1e-6,)),
    ]
    with pytest.raises(ProgramInvariantError, match="V1"):
        validate_program(_program(ops, [0]), check_races=False)


def test_v1_rejects_non_dense_uids():
    ops = [ComputeOp(uid=1, name="a", device=0, duration=(1e-6,))]
    with pytest.raises(ProgramInvariantError, match="V1"):
        validate_program(_program(ops, [0]), check_races=False)


def test_v2_rejects_unregistered_group_and_nonmember_device():
    gk = GroupKey(axis="tp", members=(0, 1))
    ops = [
        CollectiveOp(uid=0, name="c", device=0, coll=AR, size_bytes=4, label="g", group=gk),
    ]
    with pytest.raises(ProgramInvariantError, match="V2"):
        validate_program(_program(ops, [0, 1]), check_races=False)
    # registered, but device not a member
    gk2 = GroupKey(axis="tp", members=(1,))
    ops2 = [
        CollectiveOp(uid=0, name="c", device=0, coll=AR, size_bytes=4, label="g", group=gk2),
    ]
    groups = {gk2: CommGroup(key=gk2, label="g")}
    with pytest.raises(ProgramInvariantError, match="V2"):
        validate_program(_program(ops2, [0, 1], groups=groups), check_races=False)


def test_v3_same_device_transfers_are_legal_any_size():
    # DESIGN §2.2 legalizes same-device transfers; observed reality (flattened
    # graphs' per-rank cross_layer clones) carries nonzero sizes too. Both are
    # valid IR; the emitter elides them either way.
    ops = [
        ComputeOp(uid=0, name="a", device=0, duration=(1e-6,)),
        TransferOp(
            uid=1, name="t", src_device=0, dst_device=0, size_bytes=8,
            comm_type=PIPE, producer=0, deps=(0,),
        ),
    ]
    validate_program(_program(ops, [0]), check_races=False)
    ops[1].size_bytes = 0
    validate_program(_program(ops, [0]), check_races=False)
    # ...but endpoints must be devices of the program.
    ops[1].dst_device = 7
    with pytest.raises(ProgramInvariantError, match="V3"):
        validate_program(_program(ops, [0]), check_races=False)


def test_v3_producer_must_live_on_src_device():
    ops = [
        ComputeOp(uid=0, name="a", device=1, duration=(1e-6,)),
        TransferOp(
            uid=1, name="t", src_device=0, dst_device=1, size_bytes=8,
            comm_type=PIPE, producer=0, deps=(0,),
        ),
    ]
    with pytest.raises(ProgramInvariantError, match="V3"):
        validate_program(_program(ops, [0, 1]), check_races=False)


def test_v4_duration_profile_length():
    ops = [ComputeOp(uid=0, name="a", device=0, duration=(1e-6, 2e-6))]
    with pytest.raises(ProgramInvariantError, match="V4"):
        validate_program(_program(ops, [0], dp_count=3), check_races=False)
    validate_program(_program(ops, [0], dp_count=2), check_races=False)
    ops1 = [ComputeOp(uid=0, name="a", device=0, duration=(1e-6,))]
    validate_program(_program(ops1, [0], dp_count=3), check_races=False)


def test_v5_label_group_isdp_consistency():
    gk = GroupKey(axis="tp", members=(0,))
    groups = {gk: CommGroup(key=gk, label="g")}
    bad = [
        CollectiveOp(uid=0, name="c", device=0, coll=AR, size_bytes=4,
                     label="g", group=gk, is_dp=True),
    ]
    with pytest.raises(ProgramInvariantError, match="V5"):
        validate_program(_program(bad, [0], groups=groups), check_races=False)
    bad2 = [
        CollectiveOp(uid=0, name="c", device=0, coll=AR, size_bytes=4,
                     label="g", group=None),
    ]
    with pytest.raises(ProgramInvariantError, match="V5"):
        validate_program(_program(bad2, [0]), check_races=False)


def test_v6_group_race_warning_fires_and_is_silenced_by_a_path():
    gk = GroupKey(axis="tp", members=(0,))
    groups = {gk: CommGroup(key=gk, label="g")}
    racy = [
        CollectiveOp(uid=0, name="c0", device=0, coll=AR, size_bytes=4, label="g", group=gk),
        CollectiveOp(uid=1, name="c1", device=0, coll=AR, size_bytes=4, label="g", group=gk),
    ]
    with pytest.warns(GroupRaceWarning):
        validate_program(_program(racy, [0], groups=groups), check_races=True)

    ordered = [
        CollectiveOp(uid=0, name="c0", device=0, coll=AR, size_bytes=4, label="g", group=gk),
        CollectiveOp(uid=1, name="c1", device=0, coll=AR, size_bytes=4, label="g",
                     group=gk, deps=(0,)),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error", GroupRaceWarning)
        validate_program(_program(ordered, [0], groups=groups), check_races=True)


# ---------------------------------------------------------------------------
# Emitter: contract rules on synthetic programs
# ---------------------------------------------------------------------------


def test_singleton_group_emits_zero_duration_noop(tmp_path):
    b = ProgramBuilder(dp_count=1)
    u0 = b.add_compute("comp", 0, 3e-6)
    gk = b.group("tp", [0], "lonely")
    b.add_collective("coll", 0, AR, 128, deps=[u0], label="lonely", group=gk, participants=1)
    bundle = emit_chakra(b.finish(), str(tmp_path))
    nodes = _load_nodes(f"{bundle.et_prefix}.0.et")
    assert len(nodes) == 2
    noop = nodes[1]
    assert int(noop.type) == pb.COMP_NODE
    assert noop.name == "lonely_noop"
    assert int(noop.duration_micros or 0) == 0
    assert list(noop.ctrl_deps) == [0]
    assert bundle.comm_groups == {"1000": [0]}


def test_dp_skip_and_stage_groups():
    b1 = ProgramBuilder(dp_count=1)
    u0 = b1.add_compute("comp", 0, 1e-6)
    b1.add_collective("grad_sync", 0, AR, 64, deps=[u0], is_dp=True)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        bundle = emit_chakra(b1.finish(), tmp)
        nodes = _load_nodes(f"{bundle.et_prefix}.0.et")
        assert len(nodes) == 1  # dp collective skipped entirely at dp_count <= 1
        assert int(nodes[0].type) == pb.COMP_NODE
        assert bundle.comm_groups == {}
        assert bundle.comm_groups_path is None

    b2 = ProgramBuilder(dp_count=2)
    u0 = b2.add_compute("comp", 0, [1e-6, 2e-6])
    b2.add_collective("grad_sync", 0, AR, 64, deps=[u0], is_dp=True)
    with tempfile.TemporaryDirectory() as tmp:
        bundle = emit_chakra(b2.finish(), tmp)
        assert bundle.rank_ids == [0, 1]
        for dp_idx, expected_us in ((0, 1), (1, 2)):
            nodes = _load_nodes(f"{bundle.et_prefix}.{dp_idx}.et")
            assert len(nodes) == 2
            assert int(nodes[0].duration_micros) == expected_us  # per-DP profile indexing
            attrs = _attrs(nodes[1])
            assert attrs["pg_name"] == "1"  # dp stage group str(stage_idx + 1)
        assert bundle.comm_groups == {"1": [0, 1]}


def test_control_transfer_classification_and_renumber(tmp_path):
    b = ProgramBuilder(dp_count=1)
    a = b.add_compute("a", 0, 5e-6)
    c = b.add_compute("b", 1, 5e-6)
    t = b.add_transfer("ctrl", 0, 1, 0, producer=a, comm_type=PIPE, consumers=[c])
    bundle = emit_chakra(b.finish(), str(tmp_path))

    r0 = _load_nodes(f"{bundle.et_prefix}.0.et")
    # Phase C moved the control send to the lowest id.
    assert int(r0[0].type) == pb.COMM_SEND_NODE
    assert r0[0].name.endswith("_send_control")
    a0 = _attrs(r0[0])
    assert a0["comm_size"] == 1  # zero-byte control emitted as 1 byte
    assert a0["comm_tag"] == t  # tag == TransferOp uid
    assert list(r0[0].ctrl_deps) == [1]  # producer comp, renumbered to id 1
    assert int(r0[1].type) == pb.COMP_NODE

    r1 = _load_nodes(f"{bundle.et_prefix}.1.et")
    assert int(r1[0].type) == pb.COMM_RECV_NODE
    assert r1[0].name.endswith("_recv_control")
    assert _attrs(r1[0])["comm_tag"] == t
    assert int(r1[1].type) == pb.COMP_NODE
    assert list(r1[1].ctrl_deps) == [0]  # consumer waits on the recv


def test_data_transfer_keeps_natural_position(tmp_path):
    b = ProgramBuilder(dp_count=1)
    a = b.add_compute("a", 0, 5e-6)
    c = b.add_compute("b", 1, 5e-6)
    b.add_transfer("act", 0, 1, 4096, producer=a, comm_type=PIPE, consumers=[c])
    bundle = emit_chakra(b.finish(), str(tmp_path))
    r0 = _load_nodes(f"{bundle.et_prefix}.0.et")
    assert [int(n.type) for n in r0] == [pb.COMP_NODE, pb.COMM_SEND_NODE]
    assert r0[1].name == "act_send_dp0"
    assert _attrs(r0[1])["comm_size"] == 4096


def test_same_device_transfer_is_elided(tmp_path):
    b = ProgramBuilder(dp_count=1)
    a = b.add_compute("a", 0, 5e-6)
    c = b.add_compute("b", 0, 5e-6, deps=[a])
    b.add_transfer("local", 0, 0, 0, producer=a, comm_type=PIPE, consumers=[c])
    bundle = emit_chakra(b.finish(), str(tmp_path))
    nodes = _load_nodes(f"{bundle.et_prefix}.0.et")
    assert [int(n.type) for n in nodes] == [pb.COMP_NODE, pb.COMP_NODE]
    assert list(nodes[1].ctrl_deps) == [0]


def test_postcondition_trips_on_misordered_group(tmp_path):
    b = ProgramBuilder(dp_count=1)
    gk = b.group("tp", [0, 1], "g")
    x0 = b.add_collective("m1", 0, AR, 100, label="g", group=gk)
    b.add_collective("m2", 0, AR, 200, deps=[x0], label="g", group=gk)
    x1 = b.add_collective("m2c", 1, AR, 200, label="g", group=gk)
    b.add_collective("m1c", 1, AR, 100, deps=[x1], label="g", group=gk)
    prog = b.finish()
    with pytest.raises(EmissionError, match="group-order postcondition"):
        emit_chakra(prog, str(tmp_path))


def test_postcondition_passes_on_consistent_group(tmp_path):
    b = ProgramBuilder(dp_count=1)
    gk = b.group("tp", [0, 1], "g")
    x0 = b.add_collective("m1", 0, AR, 100, label="g", group=gk)
    b.add_collective("m2", 0, AR, 200, deps=[x0], label="g", group=gk)
    x1 = b.add_collective("m1c", 1, AR, 100, label="g", group=gk)
    b.add_collective("m2c", 1, AR, 200, deps=[x1], label="g", group=gk)
    emit_chakra(b.finish(), str(tmp_path))  # must not raise


def test_interning_rejects_label_with_divergent_member_sets(tmp_path):
    # One label mapping to two DIFFERENT member sets at the same dp index:
    # before the guard, the (label, dp) -> gid interning silently last-won
    # and the losing ops were stamped with a pg_name of a communicator
    # their rank does not belong to — which the member-iterating
    # postcondition never compared (silent AstraSim deadlock). V5 does not
    # cross-check label/member-set consistency across ops, so the emitter
    # must catch it.
    b = ProgramBuilder(dp_count=1)
    gk_a = b.group("tp", [1, 2], "ar")
    gk_b = b.group("tp", [0, 3], "ar")
    b.add_collective("ar_coll", 1, AR, 128, label="ar", group=gk_a, participants=2)
    b.add_collective("ar_coll", 0, AR, 128, label="ar", group=gk_b, participants=2)
    prog = b.finish()
    with pytest.raises(EmissionError, match="maps to two different"):
        emit_chakra(prog, str(tmp_path))


def test_postcondition_rejects_records_on_nonmember_rank():
    # Coverage-hole guard: a collective recorded under a gid on a rank that
    # is NOT in gid_members[gid] must trip the postcondition (before the
    # fix, only member ranks were compared, so the record was invisible).
    from program.et_emit import _check_group_order_postcondition

    node_a = new_comm_node(0, "ar", pb.ALL_REDUCE, 512)
    node_b = new_comm_node(0, "ar", pb.ALL_REDUCE, 512)
    group_records = {"1000": {0: [(node_a, 0, 512)], 5: [(node_b, 0, 512)]}}
    gid_members = {"1000": [0, 1]}
    with pytest.raises(EmissionError, match="non-member ranks"):
        _check_group_order_postcondition(group_records, gid_members, 1, 2)


# ---------------------------------------------------------------------------
# EmittedBundle equality against a hand-built expected ET
# ---------------------------------------------------------------------------


def test_emitted_bundle_matches_hand_built_et(tmp_path):
    b = ProgramBuilder(dp_count=1)
    e0 = b.add_compute("e0", 0, 10e-6)
    gk = b.group("tp", [0, 1], "ar")
    b.add_collective("ar_coll", 0, AR, 512, deps=[e0], label="ar", group=gk, participants=2)
    e1 = b.add_compute("e1", 1, 20e-6)
    b.add_collective("ar_coll", 1, AR, 512, deps=[e1], label="ar", group=gk, participants=2)
    t = b.add_transfer("xfer", 0, 1, 256, producer=e0, comm_type=PIPE, consumers=[e1])
    bundle = emit_chakra(b.finish(), str(tmp_path))

    assert bundle.rank_ids == [0, 1]
    assert bundle.comm_groups == {"1000": [0, 1]}

    # Hand-built expectation, node by node, via the same et_utils helpers.
    exp_r0 = []
    n = new_comp_node(0, "e0_0", 10)
    exp_r0.append(n)
    n = new_comm_node(1, "ar", pb.ALL_REDUCE, 512)
    n.attr.append(pb.AttributeProto(name="pg_name", string_val="1000"))
    n.ctrl_deps.append(0)
    exp_r0.append(n)
    n = new_send_node(2, "xfer_send_dp0", 256, 1, t)
    n.ctrl_deps.append(0)
    exp_r0.append(n)

    exp_r1 = []
    n = new_comp_node(0, "e1_2", 20)
    n.ctrl_deps.append(2)  # recv id wired into the consumer
    exp_r1.append(n)
    n = new_comm_node(1, "ar", pb.ALL_REDUCE, 512)
    n.attr.append(pb.AttributeProto(name="pg_name", string_val="1000"))
    n.ctrl_deps.append(0)
    exp_r1.append(n)
    n = new_recv_node(2, "xfer_recv_dp0", 256, 0, t)
    exp_r1.append(n)

    for rank, expected in ((0, exp_r0), (1, exp_r1)):
        actual = _load_nodes(f"{bundle.et_prefix}.{rank}.et")
        assert [_node_view(n) for n in actual] == [_node_view(n) for n in expected], (
            f"rank {rank} ET differs from hand-built expectation"
        )

    with open(bundle.manifest_path) as fh:
        manifest = json.load(fh)
    assert manifest["npus"] == 2
    # Legacy manifest quirk (reproduced byte-for-byte): the extractor does
    # ``int(ctype or -1)`` and pb.ALL_REDUCE == 0, so ALL_REDUCE records as -1.
    assert manifest["ranks"]["0"] == [["COMP", 10], ["COMM", -1, 512, None], ["SEND", 256]]
    assert manifest["ranks"]["1"] == [["COMP", 20], ["COMM", -1, 512, None], ["RECV", 256]]


# ---------------------------------------------------------------------------
# Legacy lowering quirks (synthetic legacy graphs)
# ---------------------------------------------------------------------------


def test_lowering_collective_only_stage_uses_pre_extension_rank_arithmetic():
    from program.legacy_lowering import lower_to_program
    from program.schedule import CommEvent, ComputeEvent

    # M8: synthetic graphs use the schedule events (the legacy Node/Edge
    # classes are retired; the events expose the same duck-typed surface).
    a = ComputeEvent("A", 0, 1.0)
    a.op_id = 0
    edge = CommEvent(
        "dp_sync",
        comm_size_bytes=64, comm_type=AR, participants=2, comm_interconnect_type="dp",
        local_hw_id=3,  # a stage no compute node lives on
    )
    edge.op_id = 2
    b_node = ComputeEvent("B", 0, 2.0)
    b_node.op_id = 1
    a.add_child(edge)
    edge.add_child(b_node)

    prog = lower_to_program(a, dp_size=2, layout_descriptor=None)
    # Collective-only stage discovered via the edge extends the device set...
    assert prog.devices == (0, 3)
    # ...but the rank formula keeps the PRE-extension stage count
    # (executor.py:1251-1262): num_stages == 1, extension index == 1.
    assert prog.num_stages_initial() == 1
    assert prog.compute_devices() == (0,)
    assert prog.rank_for(0, 0) == 0 and prog.rank_for(0, 1) == 1
    assert prog.rank_for(3, 0) == 1 and prog.rank_for(3, 1) == 2  # collides — legacy quirk

    kinds = [type(op).__name__ for op in prog.ops]
    assert kinds[:3] == ["ComputeOp", "ComputeOp", "CollectiveOp"]
    coll = prog.ops[2]
    assert coll.is_dp and coll.label is None and coll.device == 3
    # The edge's compute parent at another stage became a control transfer
    # whose pseudo-edge is the parent object itself (walker quirk).
    transfer = prog.ops[3]
    assert isinstance(transfer, TransferOp)
    assert (transfer.src_device, transfer.dst_device) == (0, 3)
    assert transfer.size_bytes == 0 and transfer.is_control
    assert transfer.name == "A"


def test_lowering_same_stage_pipeline_edge_becomes_same_device_transfer(tmp_path):
    from program.legacy_lowering import lower_to_program
    from program.schedule import CommEvent, ComputeEvent

    a = ComputeEvent("A", 0, 1.0)
    a.op_id = 0
    xl = CommEvent("cross_layer", comm_type=PIPE)
    xl.op_id = 1
    b_node = ComputeEvent("B", 0, 2.0)
    b_node.op_id = 2
    a.add_child(xl)
    xl.add_child(b_node)

    prog = lower_to_program(a, dp_size=1, layout_descriptor=None)
    assert [type(op).__name__ for op in prog.ops] == ["ComputeOp", "ComputeOp", "TransferOp"]
    b_op = prog.ops[1]
    assert b_op.deps == (0,)  # plain dep, exactly like the legacy converter
    transfer = prog.ops[2]
    assert transfer.src_device == transfer.dst_device == 0
    assert transfer.size_bytes == 0
    assert transfer.consumers == (1,)
    assert transfer.send_seq is None and transfer.recv_seq is None

    bundle = emit_chakra(prog, str(tmp_path))
    nodes = _load_nodes(f"{bundle.et_prefix}.0.et")
    assert [int(n.type) for n in nodes] == [pb.COMP_NODE, pb.COMP_NODE]
    assert list(nodes[1].ctrl_deps) == [0]
