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

"""Fast, pure unit tests for the ``equiv/`` package.

No subprocesses and no AstraSim binary: tiny synthetic Chakra ET bundles are
written into ``tmp_path`` with ``astrasim_lib.et_utils`` and fed directly to
``equiv.canonical`` and ``equiv.dlsim``.

Run: ./.venv/bin/python -m pytest tests/test_equiv_unit.py -q
"""

from __future__ import annotations

import json

import pytest

from astrasim_lib.et_utils import (
    chakra_encode,
    new_comm_node,
    new_comp_node,
    new_recv_node,
    new_send_node,
    pb,
)
from equiv.canonical import canonicalize_bundle, canonicalize_rank, diff_bundles
from equiv.dlsim import BundleSim

PREFIX = "llm_graph"


# ---------------------------------------------------------------------------
# Bundle-building helpers
# ---------------------------------------------------------------------------


def _with_deps(node, deps):
    node.ctrl_deps.extend(int(d) for d in deps)
    return node


def _with_pg(node, pg_id):
    node.attr.append(pb.AttributeProto(name="pg_name", string_val=str(pg_id)))
    return node


def _write_rank(bundle_dir, rank, nodes, prefix=PREFIX):
    path = bundle_dir / f"{prefix}.{rank}.et"
    with open(path, "wb") as fh:
        chakra_encode(fh, pb.GlobalMetadata(version="0.0.4"))
        for node in nodes:
            chakra_encode(fh, node)
    return path


def _write_bundle(bundle_dir, rank_nodes, comm_groups=None):
    """Write ``{rank: [pb.Node, ...]}`` plus comm_groups.json under bundle_dir."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    for rank, nodes in rank_nodes.items():
        _write_rank(bundle_dir, rank, nodes)
    (bundle_dir / "comm_groups.json").write_text(json.dumps(comm_groups or {}))
    return bundle_dir


def _make_reference(
    bundle_dir,
    *,
    pg="0",
    names0=("r0_comp", "r0_coll", "r0_send"),
    names1=("r1_comp", "r1_coll", "r1_recv"),
    ids0=(0, 1, 2),
    ids1=(0, 1, 2),
    order0=(0, 1, 2),
    order1=(0, 1, 2),
    comp_dur=5,
    coll_size=1024,
    p2p_tag=7,
):
    """Two-rank comp -> coll -> send/recv bundle.

    ``ids*`` permute node ids (ctrl_deps are remapped accordingly), ``order*``
    permute the on-disk record order, ``names*`` rename nodes: all of these
    must be canonical no-ops.
    """
    c0, g0, s0 = ids0
    rank0 = [
        new_comp_node(c0, names0[0], comp_dur),
        _with_deps(_with_pg(new_comm_node(g0, names0[1], pb.ALL_REDUCE, coll_size), pg), [c0]),
        _with_deps(new_send_node(s0, names0[2], 64, dst_rank=1, tag=p2p_tag), [g0]),
    ]
    c1, g1, r1 = ids1
    rank1 = [
        new_comp_node(c1, names1[0], comp_dur),
        _with_deps(_with_pg(new_comm_node(g1, names1[1], pb.ALL_REDUCE, coll_size), pg), [c1]),
        _with_deps(new_recv_node(r1, names1[2], 64, src_rank=0, tag=p2p_tag), [g1]),
    ]
    rank0 = [rank0[k] for k in order0]
    rank1 = [rank1[k] for k in order1]
    return _write_bundle(bundle_dir, {0: rank0, 1: rank1}, {str(pg): [0, 1]})


def _comp_chain(durs, deps_of):
    """COMP nodes id 0..n-1 with durations ``durs`` and parents ``deps_of[i]``."""
    return [
        _with_deps(new_comp_node(i, f"comp{i}", dur), deps_of.get(i, []))
        for i, dur in enumerate(durs)
    ]


# ---------------------------------------------------------------------------
# 1. canonical: hash invariances and sensitivities
# ---------------------------------------------------------------------------


class TestCanonical:
    def test_identical_bundles_identical_hashes(self, tmp_path):
        a = canonicalize_bundle(str(_make_reference(tmp_path / "a")))
        b = canonicalize_bundle(str(_make_reference(tmp_path / "b")))
        assert a.bundle_hash == b.bundle_hash
        for rank in (0, 1):
            assert a.ranks[rank].dag_hash == b.ranks[rank].dag_hash
            assert a.ranks[rank].ops_hash == b.ranks[rank].ops_hash
        assert a.summary() == b.summary()

    def test_rename_and_id_permutation_invariant(self, tmp_path):
        ref = canonicalize_bundle(str(_make_reference(tmp_path / "ref")))
        perm = canonicalize_bundle(
            str(
                _make_reference(
                    tmp_path / "perm",
                    names0=("alpha", "beta", "gamma"),
                    names1=("delta", "epsilon", "zeta"),
                    ids0=(10, 3, 7),  # ctrl_deps remapped inside the builder
                    ids1=(5, 2, 9),
                    order0=(2, 0, 1),  # record order permuted on disk too
                    order1=(1, 2, 0),
                )
            )
        )
        assert perm.bundle_hash == ref.bundle_hash
        for rank in (0, 1):
            assert perm.ranks[rank].dag_hash == ref.ranks[rank].dag_hash
            assert perm.ranks[rank].ops_hash == ref.ranks[rank].ops_hash

    def test_duration_change_changes_hashes(self, tmp_path):
        ref = canonicalize_bundle(str(_make_reference(tmp_path / "ref")))
        mut = canonicalize_bundle(str(_make_reference(tmp_path / "mut", comp_dur=6)))
        assert mut.ranks[0].ops_hash != ref.ranks[0].ops_hash
        assert mut.ranks[0].dag_hash != ref.ranks[0].dag_hash
        assert mut.bundle_hash != ref.bundle_hash

    def test_comm_size_change_changes_hashes(self, tmp_path):
        ref = canonicalize_bundle(str(_make_reference(tmp_path / "ref")))
        mut = canonicalize_bundle(str(_make_reference(tmp_path / "mut", coll_size=2048)))
        assert mut.ranks[0].ops_hash != ref.ranks[0].ops_hash
        assert mut.ranks[0].dag_hash != ref.ranks[0].dag_hash
        assert mut.bundle_hash != ref.bundle_hash

    def test_dependency_edge_change_changes_dag_hash_only(self, tmp_path):
        # Same op multiset (COMP 5/9/13), different wiring of the third node.
        chain = _write_bundle(
            tmp_path / "chain", {0: _comp_chain([5, 9, 13], {2: [0]})}
        )
        rewired = _write_bundle(
            tmp_path / "rewired", {0: _comp_chain([5, 9, 13], {2: [1]})}
        )
        a = canonicalize_bundle(str(chain))
        b = canonicalize_bundle(str(rewired))
        assert a.ranks[0].ops_hash == b.ranks[0].ops_hash
        assert a.ranks[0].dag_hash != b.ranks[0].dag_hash
        assert a.bundle_hash != b.bundle_hash

    def test_pg_id_renumbering_same_members_same_hashes(self, tmp_path):
        a = canonicalize_bundle(str(_make_reference(tmp_path / "a", pg="0")))
        b = canonicalize_bundle(str(_make_reference(tmp_path / "b", pg="42")))
        assert a.bundle_hash == b.bundle_hash
        for rank in (0, 1):
            assert a.ranks[rank].dag_hash == b.ranks[rank].dag_hash
            assert a.ranks[rank].ops_hash == b.ranks[rank].ops_hash

    def test_missing_pg_in_comm_groups_raises(self, tmp_path):
        coll = _with_pg(new_comm_node(0, "coll", pb.ALL_REDUCE, 512), "7")
        bundle = _write_bundle(tmp_path / "bad", {0: [coll]}, comm_groups={"other": [0]})
        with pytest.raises(ValueError, match="no such group"):
            canonicalize_bundle(str(bundle))

    def test_cycle_in_ctrl_deps_raises(self, tmp_path):
        cyclic = _write_bundle(
            tmp_path / "cyclic", {0: _comp_chain([1, 2], {0: [1], 1: [0]})}
        )
        et_path = str(tmp_path / "cyclic" / f"{PREFIX}.0.et")
        with pytest.raises(ValueError, match="cycle detected"):
            canonicalize_rank(0, et_path, {}, [0])


# ---------------------------------------------------------------------------
# 2. diff_bundles
# ---------------------------------------------------------------------------


class TestDiffBundles:
    def test_identical_bundles_no_problems(self, tmp_path):
        golden = canonicalize_bundle(str(_make_reference(tmp_path / "a"))).summary()
        cand = canonicalize_bundle(str(_make_reference(tmp_path / "b"))).summary()
        assert diff_bundles(golden, cand) == []

    def test_rank_count_mismatch(self, tmp_path):
        golden = canonicalize_bundle(
            str(_write_bundle(tmp_path / "g", {0: _comp_chain([5], {}), 1: _comp_chain([5], {})}))
        ).summary()
        cand = canonicalize_bundle(
            str(_write_bundle(tmp_path / "c", {0: _comp_chain([5], {})}))
        ).summary()
        problems = diff_bundles(golden, cand, label="unit")
        assert len(problems) == 1
        assert "rank count differs" in problems[0]
        assert "[unit]" in problems[0]

    def test_op_multiset_mismatch_reports_counts(self, tmp_path):
        golden = canonicalize_bundle(
            str(_write_bundle(tmp_path / "g", {0: _comp_chain([5, 5], {})}))
        ).summary()
        cand = canonicalize_bundle(
            str(_write_bundle(tmp_path / "c", {0: _comp_chain([5, 5, 5], {})}))
        ).summary()
        problems = diff_bundles(golden, cand)
        assert len(problems) == 1
        assert "op multiset differs" in problems[0]
        assert "'COMP': 2" in problems[0]  # golden counts
        assert "'COMP': 3" in problems[0]  # candidate counts

    def test_dag_only_mismatch(self, tmp_path):
        golden = canonicalize_bundle(
            str(_write_bundle(tmp_path / "g", {0: _comp_chain([1, 2, 3], {1: [0], 2: [1]})}))
        ).summary()
        cand = canonicalize_bundle(
            str(_write_bundle(tmp_path / "c", {0: _comp_chain([1, 2, 3], {1: [0], 2: [0]})}))
        ).summary()
        problems = diff_bundles(golden, cand)
        assert len(problems) == 1
        assert "same ops but dependency DAG differs" in problems[0]


# ---------------------------------------------------------------------------
# 3. dlsim: completion, deadlocks, single-comm-slot semantics
# ---------------------------------------------------------------------------


class TestDlsim:
    def test_valid_bundle_completes(self, tmp_path):
        _make_reference(tmp_path / "ok")
        res = BundleSim(str(tmp_path / "ok")).simulate()
        assert res.completed
        assert res.done_counts == res.total_counts == {0: 3, 1: 3}
        assert res.blocked_report == []
        assert res.cycle == []

    def test_group_collective_order_mismatch_deadlocks(self, tmp_path):
        # rank0 issues g1 then g2; rank1 issues g2 then g1. Each rank's comm
        # slot is stuck on a collective the peer never issues.
        rank0 = [
            _with_pg(new_comm_node(0, "a", pb.ALL_REDUCE, 256), "g1"),
            _with_deps(_with_pg(new_comm_node(1, "b", pb.ALL_REDUCE, 256), "g2"), [0]),
        ]
        rank1 = [
            _with_pg(new_comm_node(0, "c", pb.ALL_REDUCE, 256), "g2"),
            _with_deps(_with_pg(new_comm_node(1, "d", pb.ALL_REDUCE, 256), "g1"), [0]),
        ]
        bundle = _write_bundle(
            tmp_path / "order", {0: rank0, 1: rank1}, {"g1": [0, 1], "g2": [0, 1]}
        )
        res = BundleSim(str(bundle)).simulate()
        assert not res.completed
        assert res.done_counts == {0: 0, 1: 0}
        # explain() output must be non-empty and name the stuck ranks
        assert res.blocked_report
        assert any("not-done" in line for line in res.blocked_report)

    def test_crossed_recv_send_deps_yield_wait_for_cycle(self, tmp_path):
        # Each rank's SEND depends on its own RECV, whose matching SEND is the
        # peer's blocked SEND: a genuine 4-node wait-for cycle.
        rank0 = [
            new_recv_node(0, "recv_a", 64, src_rank=1, tag=1),
            _with_deps(new_send_node(1, "send_b", 64, dst_rank=1, tag=2), [0]),
        ]
        rank1 = [
            new_recv_node(0, "recv_b", 64, src_rank=0, tag=2),
            _with_deps(new_send_node(1, "send_a", 64, dst_rank=0, tag=1), [0]),
        ]
        bundle = _write_bundle(tmp_path / "crossed", {0: rank0, 1: rank1})
        res = BundleSim(str(bundle)).simulate()
        assert not res.completed
        assert res.cycle, "explain() should surface the wait-for cycle"
        joined = "\n".join(res.cycle)
        assert "RECV" in joined and "SEND" in joined

    def test_send_recv_tag_mismatch_deadlocks_no_matching_send(self, tmp_path):
        rank0 = [new_send_node(0, "send", 64, dst_rank=1, tag=1)]
        rank1 = [new_recv_node(0, "recv", 64, src_rank=0, tag=2)]
        bundle = _write_bundle(tmp_path / "tags", {0: rank0, 1: rank1})
        res = BundleSim(str(bundle)).simulate()
        assert not res.completed
        # the buffered send itself completes; the recv can never match
        assert res.done_counts == {0: 1, 1: 0}
        assert "NO MATCHING SEND" in "\n".join(res.blocked_report)

    def test_ready_send_blocked_behind_stuck_coll_single_slot(self, tmp_path):
        # SEND and COLL share ONE in-flight comm slot: rank0's collective
        # occupies it forever (rank1's matching collective waits on a recv fed
        # by rank0's send), so the ready send can never issue -> deadlock.
        rank0 = [
            _with_pg(new_comm_node(0, "coll0", pb.ALL_REDUCE, 256), "g"),
            new_send_node(1, "send", 64, dst_rank=1, tag=3),  # ready, no deps
        ]
        rank1 = [
            new_recv_node(0, "recv", 64, src_rank=0, tag=3),
            _with_deps(_with_pg(new_comm_node(1, "coll1", pb.ALL_REDUCE, 256), "g"), [0]),
        ]
        bundle = _write_bundle(tmp_path / "slot", {0: rank0, 1: rank1}, {"g": [0, 1]})
        res = BundleSim(str(bundle)).simulate()
        assert not res.completed
        assert res.done_counts == {0: 0, 1: 0}
        assert res.blocked_report

    def test_pending_recv_does_not_block_comm_slot(self, tmp_path):
        # An outstanding (not yet matched) RECV must not occupy the comm slot:
        # the collective behind it issues and completes, unblocking the send
        # that eventually satisfies the recv.
        rank0 = [
            new_recv_node(0, "recv", 64, src_rank=1, tag=3),
            _with_pg(new_comm_node(1, "coll0", pb.ALL_REDUCE, 256), "g"),
        ]
        rank1 = [
            _with_pg(new_comm_node(0, "coll1", pb.ALL_REDUCE, 256), "g"),
            _with_deps(new_send_node(1, "send", 64, dst_rank=0, tag=3), [0]),
        ]
        bundle = _write_bundle(tmp_path / "recvslot", {0: rank0, 1: rank1}, {"g": [0, 1]})
        res = BundleSim(str(bundle)).simulate()
        assert res.completed
        assert res.done_counts == res.total_counts == {0: 2, 1: 2}

    def test_unmatched_send_is_buffered_and_frees_slot(self, tmp_path):
        # Documented AstraSim contract (dlsim docstring + docs/rewrite/
        # CONTEXT.md): SEND completes unconditionally once issued -- delivery
        # is buffered. A COLL queued behind a send with NO matching recv
        # anywhere therefore still issues, and the bundle completes.
        rank0 = [
            new_send_node(0, "send_unmatched", 64, dst_rank=1, tag=9),
            _with_deps(_with_pg(new_comm_node(1, "coll0", pb.ALL_REDUCE, 256), "g"), [0]),
        ]
        rank1 = [_with_pg(new_comm_node(0, "coll1", pb.ALL_REDUCE, 256), "g")]
        bundle = _write_bundle(tmp_path / "buffered", {0: rank0, 1: rank1}, {"g": [0, 1]})
        res = BundleSim(str(bundle)).simulate()
        assert res.completed
        assert res.done_counts == res.total_counts == {0: 2, 1: 1}
