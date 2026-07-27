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
    """V6 warns on an unordered same-group pair whose EARLIER member has a
    successor — the narrowing P5 applied (INTERFACES §4.8).

    V6 became always-on in ``build()``, and a gradient reducer is a graph SINK by
    construction: nothing follows it. Two sinks cannot race, because nothing
    observes which issued first for a program whose order is ONE global order
    projected onto every device (CONTEXT constraint 2). A pair whose earlier
    member feeds something IS the race, and is still reported.
    """
    gk = GroupKey(axis="tp", members=(0,))
    groups = {gk: CommGroup(key=gk, label="g")}
    racy = [
        CollectiveOp(uid=0, name="c0", device=0, coll=AR, size_bytes=4, label="g", group=gk),
        CollectiveOp(uid=1, name="c1", device=0, coll=AR, size_bytes=4, label="g", group=gk),
        ComputeOp(uid=2, name="consumer", device=0, duration=(1.0,), deps=(0,)),
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

    # ... and the narrowed-away class: two SINK reducers of one group.
    sinks = [
        CollectiveOp(uid=0, name="c0", device=0, coll=AR, size_bytes=4, label="g", group=gk),
        CollectiveOp(uid=1, name="c1", device=0, coll=AR, size_bytes=4, label="g", group=gk),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error", GroupRaceWarning)
        validate_program(_program(sinks, [0], groups=groups), check_races=True)


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
# Same-device transfers and the dp-major rank formula
# ---------------------------------------------------------------------------
#
# P5: the two tests that drove synthetic graphs through
# ``legacy_lowering.lower_to_program`` are gone with that pass. What they were
# really pinning survives here, stated against the IR the emitter consumes.


def test_same_device_transfer_edge_survives_elision(tmp_path):
    """A same-device ``TransferOp`` emits NO wire op, and its dependency must
    reappear as a plain ctrl_dep on the consumer.

    Elide the OP, keep the EDGE. Dropping the edge instead lets the ranks of one
    stage drift out of collective-issue lockstep, which AstraSim reports as
    ``Hardware Resource ... has unreleased nodes`` — a silent deadlock.
    """
    b = ProgramBuilder(dp_count=1)
    a_uid = b.add_compute("A", 0, 1.0)
    t_uid = b.add_transfer("cross_layer", 0, 0, 0, a_uid, comm_type=PIPE)
    b.add_compute("B", 0, 2.0, deps=(t_uid,))
    prog = b.finish(validate=False)

    transfer = prog.ops[t_uid]
    assert transfer.src_device == transfer.dst_device == 0
    assert transfer.size_bytes == 0

    bundle = emit_chakra(prog, str(tmp_path))
    nodes = _load_nodes(f"{bundle.et_prefix}.0.et")
    assert [int(n.type) for n in nodes] == [pb.COMP_NODE, pb.COMP_NODE]
    assert list(nodes[1].ctrl_deps) == [0]


def test_cross_device_send_carries_every_same_rank_dep_of_the_transfer(tmp_path):
    """A ``TransferOp``'s NON-producer deps land on the SEND, not on the floor.

    ``build()._emit_cross_layer`` deliberately records a second DATA_FLOW dep on
    a pipeline transfer — the COMPUTE ANCHOR: "the SEND must fire off the last
    COMPUTE, not off a trailing collective" (``pipeline_fine.py:672-687``).
    Phase B used to emit ``transfer.producer`` and nothing else, so with a
    fully-hoisted tp/sp collective (the anchor is then the producer's SIBLING,
    not its ancestor) the inter-stage SEND had NO dependency on the compute that
    produced the activation it carries, and that compute became a dependency
    SINK. This is the shape that costs 62-95% of the P5 T2 movement on every
    ``tp > 1`` flattened spec.
    """
    b = ProgramBuilder(dp_count=1)
    root = b.add_compute("root", 0, 1e-6)
    # anchor and coll are SIBLINGS, exactly as a hoisted overlap leaves them
    anchor = b.add_compute("MLP_forward", 0, 5e-6, deps=(root,))
    gk = b.group("tp", [0, 1], "mlp_tp")
    coll = b.add_collective(
        "mlp_rs", 0, AR, 4096, deps=[root], label="mlp_tp", group=gk, participants=2
    )
    peer_root = b.add_compute("peer_root", 1, 1e-6)
    b.add_collective(
        "mlp_rs", 1, AR, 4096, deps=[peer_root], label="mlp_tp", group=gk, participants=2
    )
    consumer = b.add_compute("next_stage", 1, 2e-6)
    t = b.add_transfer(
        "cross_layer", 0, 1, 2097152, producer=coll, comm_type=PIPE, consumers=[consumer]
    )
    prog = b.finish(validate=False)
    # what build() records: producer + compute anchor
    prog.ops[t].deps = (coll, anchor)

    bundle = emit_chakra(prog, str(tmp_path))
    nodes = _load_nodes(f"{bundle.et_prefix}.0.et")
    by_id = {int(n.id): n for n in nodes}
    send = [n for n in nodes if int(n.type) == pb.COMM_SEND_NODE]
    assert len(send) == 1
    dep_names = {by_id[int(d)].name for d in send[0].ctrl_deps}
    assert any("mlp_rs" in name or "mlp_tp" in name for name in dep_names), dep_names
    assert any("MLP_forward" in name for name in dep_names), (
        f"the compute anchor is missing from the SEND's ctrl_deps: {dep_names}"
    )
    # ... and therefore the compute is not a dependency sink
    anchor_node = next(n for n in nodes if "MLP_forward" in n.name)
    successors = [n for n in nodes if int(anchor_node.id) in set(n.ctrl_deps)]
    assert successors, "the layer's last COMPUTE must not be a dependency sink"


def test_cross_rank_dep_is_materialized_as_a_control_pair(tmp_path):
    """A dep that crosses ranks with no transfer carrying it goes ON THE WIRE.

    It used to be DROPPED with a ``CrossRankDepWarning`` justified by "the
    ordering is carried by the shared per-stage collectives". On
    ``train:flattened:dp2tp1cp1pp2mb2sp0:zero3`` that was false — no collective
    spans the two pipeline stages — and the four dropped edges were the only
    thing ordering the ZeRO-3 parameter gathers, which became graph ROOTS. The
    emitter is TOTAL now: every IR edge is a ctrl_dep, a wire, or an
    ``EmissionError``.
    """
    b = ProgramBuilder(dp_count=1)
    a = b.add_compute("producer_on_0", 0, 4e-6)
    consumer = b.add_compute("consumer_on_1", 1, 4e-6, deps=(a,))
    prog = b.finish(validate=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning class may survive
        bundle = emit_chakra(prog, str(tmp_path))

    r0 = _load_nodes(f"{bundle.et_prefix}.0.et")
    r1 = _load_nodes(f"{bundle.et_prefix}.1.et")
    sends = [n for n in r0 if int(n.type) == pb.COMM_SEND_NODE]
    recvs = [n for n in r1 if int(n.type) == pb.COMM_RECV_NODE]
    assert len(sends) == len(recvs) == 1
    assert _attrs(sends[0])["comm_size"] == 1
    assert _attrs(sends[0])["comm_dst"] == 1
    assert _attrs(recvs[0])["comm_src"] == 0
    assert _attrs(sends[0])["comm_tag"] == _attrs(recvs[0])["comm_tag"]
    assert _attrs(sends[0])["comm_tag"] >= 1_000_000, "sync tags may not collide with a uid"
    # the SEND fires off the dep; the RECV gates the consumer
    by_id0 = {int(n.id): n for n in r0}
    assert "producer_on_0" in by_id0[int(sends[0].ctrl_deps[0])].name
    consumer_node = next(n for n in r1 if "consumer_on_1" in n.name)
    assert int(recvs[0].id) in set(consumer_node.ctrl_deps)
    assert consumer_node.ctrl_deps, "the consumer must not become a graph root"
    assert not hasattr(__import__("program.et_emit", fromlist=["x"]), "CrossRankDepWarning")


def test_emission_postcondition_catches_a_lost_successor():
    """The gate that would have caught the P5 anchor drop.

    A trace in which the ``MLP_forward`` compute has a successor in the Program
    but none in the emitted DAG is exactly the shape the anchor drop produced,
    and it is invisible to every other gate: same op multiset, same bytes, same
    collectives, ``dlsim`` still completes. Only the AstraSim wall clock moved.
    """
    from program.et_emit import _check_no_lost_successors

    b = ProgramBuilder(dp_count=1)
    anchor = b.add_compute("MLP_forward", 0, 5e-6)
    consumer = b.add_compute("consumer", 0, 1e-6, deps=(anchor,))
    prog = b.finish(validate=False)
    et_ids = {(anchor, 0): 0, (consumer, 0): 1}

    class _T:
        def __init__(self, nodes):
            self.nodes = nodes

    healthy = _T([new_comp_node(0, "MLP_forward", 5), new_comp_node(1, "consumer", 1)])
    healthy.nodes[1].ctrl_deps.append(0)
    _check_no_lost_successors(prog.ops, et_ids, {0: healthy}, 1)

    lost = _T([new_comp_node(0, "MLP_forward", 5), new_comp_node(1, "consumer", 1)])
    with pytest.raises(EmissionError, match="successors in the Program but NONE"):
        _check_no_lost_successors(prog.ops, et_ids, {0: lost}, 1)


def test_uncarriable_cross_rank_dep_is_an_error_not_a_shrug(tmp_path):
    """The class cannot silently return: an edge that cannot be put on the wire
    raises instead of being dropped."""
    b = ProgramBuilder(dp_count=1)
    a = b.add_compute("a", 0, 1e-6)
    host = b.add_compute("host", 1, 1e-6)
    same_device = b.add_transfer("cross_layer", 1, 1, 0, producer=host, comm_type=PIPE)
    b.add_compute("consumer", 2, 1e-6, deps=(same_device,))
    prog = b.finish(validate=False)
    # A NESTED cross-rank carrier: the consumer (rank 2) depends on an elided
    # same-device transfer on rank 1 whose own deps reach rank 0. One wire
    # cannot express that, so it is loud.
    prog.ops[same_device].deps = (host, a)
    with pytest.raises(EmissionError, match="cannot be carried"):
        emit_chakra(prog, str(tmp_path))


def test_rank_formula_uses_the_declared_stage_count():
    """``rank = dp_idx * num_stages_initial + stage_index`` (DESIGN §2.5).

    ``num_stages_initial`` is DECLARED on the program, which is what keeps
    BUG_LEDGER **A4** expressible: a device set larger than the declared stage
    count makes two (device, dp) pairs share a rank, and P7 flips the declared
    value rather than editing arithmetic.
    """
    b = ProgramBuilder(dp_count=2)
    b.add_compute("A", 0, 1.0)
    prog = b.finish(devices=(0, 3), num_stages_initial=1, compute_devices=(0,))
    assert prog.num_stages_initial() == 1
    assert prog.compute_devices() == (0,)
    assert prog.rank_for(0, 0) == 0 and prog.rank_for(0, 1) == 1
    # the collision A4 describes, asserted so a fix cannot land unnoticed
    assert prog.rank_for(3, 0) == 1 and prog.rank_for(3, 1) == 2
