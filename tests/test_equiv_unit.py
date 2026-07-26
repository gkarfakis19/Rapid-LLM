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
        assert any("op multiset differs" in p for p in problems)
        head = next(p for p in problems if "op multiset differs" in p)
        assert "'COMP': 2" in head  # golden counts
        assert "'COMP': 3" in head  # candidate counts
        # T1 quantities localize the change instead of only hashing it.
        assert any("compute_micros: golden=10 candidate=15" in p for p in problems)
        assert any("compute_micros_total: golden=10 candidate=15" in p for p in problems)

    def test_dag_only_mismatch(self, tmp_path):
        golden = canonicalize_bundle(
            str(_write_bundle(tmp_path / "g", {0: _comp_chain([1, 2, 3], {1: [0], 2: [1]})}))
        ).summary()
        cand = canonicalize_bundle(
            str(_write_bundle(tmp_path / "c", {0: _comp_chain([1, 2, 3], {1: [0], 2: [0]})}))
        ).summary()
        problems = diff_bundles(golden, cand)
        assert any("same ops but dependency DAG differs" in p for p in problems)
        # Same op multiset, shorter chain: the critical path is what moved.
        assert any("critical_path_nodes: golden=3 candidate=2" in p for p in problems)
        assert any("critical_path_weight: golden=6 candidate=4" in p for p in problems)

    def test_diff_fields_are_stable_ledger_keys(self, tmp_path):
        from equiv.canonical import diff_bundles_detailed

        golden = canonicalize_bundle(
            str(_write_bundle(tmp_path / "g", {0: _comp_chain([1, 2, 3], {1: [0], 2: [1]})}))
        ).summary()
        cand = canonicalize_bundle(
            str(_write_bundle(tmp_path / "c", {0: _comp_chain([1, 2, 9], {1: [0], 2: [1]})}))
        ).summary()
        fields = {field for field, _msg, _old, _new in diff_bundles_detailed(golden, cand)}
        assert "rank/0/ops_hash" in fields
        assert "rank/0/compute_micros" in fields
        assert "compute_micros_total" in fields


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


# ---------------------------------------------------------------------------
# 4. T1 extras: memory summaries and the axis sidecar
# ---------------------------------------------------------------------------


class TestMemorySummaries:
    def _write(self, run_dir, mode_label, lines):
        target = run_dir / "output" / mode_label / "memory-summary"
        target.mkdir(parents=True, exist_ok=True)
        (target / "memory_capacity_comparison.txt").write_text("\n".join(lines) + "\n")

    def test_training_summary_parsed_into_floats(self, tmp_path):
        from equiv.runner import _parse_memory_summaries

        self._write(
            tmp_path,
            "LLM",
            [
                "Simulation mode: training",
                "Hardware memory capacity (per gpu): 80.00 GiB",
                "Simulated peak memory usage(per gpu): 1.51 GiB",
                "Remaining memory headroom: 78.49 GiB",
            ],
        )
        parsed = _parse_memory_summaries(tmp_path)
        assert set(parsed) == {"LLM"}
        fields = parsed["LLM"]["fields"]
        assert fields["hardware_memory_capacity_per_gpu"] == 80.0
        assert fields["simulated_peak_memory_usage_per_gpu"] == 1.51
        assert fields["remaining_memory_headroom"] == 78.49
        assert fields["simulation_mode"] == "training"
        assert parsed["LLM"]["warnings"] == []

    def test_capacity_violation_is_recorded_as_a_warning(self, tmp_path):
        from equiv.runner import _parse_memory_summaries

        self._write(
            tmp_path,
            "LLM",
            [
                "Simulation mode: training",
                "Hardware memory capacity (per gpu): 80.00 GiB",
                "Simulated peak memory usage(per gpu): 91.00 GiB",
                "[WARN] Peak memory exceeds capacity by 11.00 GiB",
            ],
        )
        parsed = _parse_memory_summaries(tmp_path)
        assert parsed["LLM"]["warnings"] == [
            "[WARN] Peak memory exceeds capacity by 11.00 GiB"
        ]

    def test_axis_histogram_absent_without_the_sidecar(self, tmp_path):
        bundle = _write_bundle(
            tmp_path / "noaxes",
            {0: [_with_pg(new_comm_node(0, "c", pb.ALL_REDUCE, 512), "1")],
             1: [_with_pg(new_comm_node(0, "c", pb.ALL_REDUCE, 512), "1")]},
            {"1": [0, 1]},
        )
        summary = canonicalize_bundle(str(bundle)).summary()
        assert "bytes_by_axis" not in summary
        assert summary["bytes_by_kind"] == {"COLL": 1024}
        assert summary["collectives_by_group"]["0,1"]["bytes"] == 1024

    def test_axis_histogram_uses_the_sidecar_keyed_by_member_set(self, tmp_path):
        bundle = _write_bundle(
            tmp_path / "axes",
            {0: [_with_pg(new_comm_node(0, "c", pb.ALL_REDUCE, 512), "1000")],
             1: [_with_pg(new_comm_node(0, "c", pb.ALL_REDUCE, 512), "1000")]},
            {"1000": [0, 1]},
        )
        (bundle / "comm_axes.json").write_text(
            json.dumps({"version": "df-astra-comm-axes/1",
                        "groups": {"1000": {"axes": ["tp"], "labels": ["x"]}}})
        )
        summary = canonicalize_bundle(str(bundle)).summary()
        assert summary["bytes_by_axis"] == {"tp": 1024}


# ---------------------------------------------------------------------------
# 5. T3 property: p2p tag pairing on ONE bundle
# ---------------------------------------------------------------------------


class TestP2PPairing:
    def test_paired_send_recv_is_clean(self, tmp_path):
        bundle = _write_bundle(
            tmp_path / "ok",
            {
                0: [new_send_node(0, "s", 64, dst_rank=1, tag=7)],
                1: [new_recv_node(0, "r", 64, src_rank=0, tag=7)],
            },
        )
        from program.shadow import p2p_pairing_problems

        assert p2p_pairing_problems(str(bundle)) == []

    def test_two_sends_sharing_one_identity_are_reported(self, tmp_path):
        bundle = _write_bundle(
            tmp_path / "dup",
            {
                0: [
                    new_send_node(0, "s0", 64, dst_rank=1, tag=7),
                    new_send_node(1, "s1", 64, dst_rank=1, tag=7),
                ],
                1: [new_recv_node(0, "r", 64, src_rank=0, tag=7)],
            },
        )
        from program.shadow import p2p_pairing_problems

        problems = p2p_pairing_problems(str(bundle), label="unit")
        assert len(problems) == 1
        assert "used by 2 SEND nodes" in problems[0]
        assert "[unit]" in problems[0]

    def test_recv_without_a_send_is_reported(self, tmp_path):
        bundle = _write_bundle(
            tmp_path / "orphan",
            {
                0: [new_send_node(0, "s", 64, dst_rank=1, tag=7)],
                1: [new_recv_node(0, "r", 64, src_rank=0, tag=8)],
            },
        )
        from program.shadow import p2p_pairing_problems

        problems = p2p_pairing_problems(str(bundle))
        assert len(problems) == 1
        assert "no matching SEND" in problems[0]


# ---------------------------------------------------------------------------
# 6. T4 bug ledger
# ---------------------------------------------------------------------------


class TestBugLedger:
    def _write(self, tmp_path, entries):
        path = tmp_path / "bug_ledger.json"
        path.write_text(json.dumps({"version": "df-bug-ledger/1", "entries": entries}))
        return path

    def test_shipped_ledger_parses(self):
        from equiv.ledger import default_ledger_path, load_ledger

        assert default_ledger_path().exists()
        load_ledger()  # must not raise

    def test_missing_file_means_no_exceptions(self, tmp_path):
        from equiv.ledger import load_ledger

        assert load_ledger(tmp_path / "absent.json") == []

    def test_entry_absorbs_the_declared_difference(self, tmp_path):
        from equiv.ledger import Mismatch, apply_ledger, load_ledger

        path = self._write(
            tmp_path,
            [
                {
                    "spec": "s1",
                    "level": "structural",
                    "field": "bundles/flat/compute_micros_total",
                    "old": 100,
                    "new": 90,
                    "commit": "A3",
                    "justification": "double-counted MoE layer removed",
                }
            ],
        )
        entries = load_ledger(path)
        mismatch = Mismatch("structural", "bundles/flat/compute_micros_total", "m", 100, 90)
        failures, recorded, matched = apply_ledger("s1", [mismatch], entries)
        assert failures == []
        assert len(recorded) == 1 and "A3" in recorded[0]
        assert matched == entries

    def test_wrong_new_value_is_a_louder_failure(self, tmp_path):
        from equiv.ledger import Mismatch, apply_ledger, load_ledger

        entries = load_ledger(
            self._write(
                tmp_path,
                [
                    {
                        "spec": "s1",
                        "level": "timing",
                        "field": "total_time",
                        "old": 1.0,
                        "new": 0.9,
                        "commit": "A2",
                        "justification": "dp collectives now on every rank",
                    }
                ],
            )
        )
        mismatch = Mismatch("timing", "total_time", "total_time moved", 1.0, 0.5)
        failures, recorded, _matched = apply_ledger("s1", [mismatch], entries)
        assert recorded == []
        assert len(failures) == 1
        assert "LEDGER MISMATCH" in failures[0].message

    def test_entry_expires_when_old_stops_matching(self, tmp_path):
        from equiv.ledger import Mismatch, apply_ledger, expired_entries, load_ledger

        entries = load_ledger(
            self._write(
                tmp_path,
                [
                    {
                        "spec": "s1",
                        "level": "timing",
                        "field": "total_time",
                        "old": 1.0,
                        "new": 0.9,
                        "commit": "A2",
                        "justification": "recaptured",
                    }
                ],
            )
        )
        # goldens were recaptured: the golden value is now 0.9, so nothing
        # matches 'old' any more and the entry is dead weight.
        failures, recorded, matched = apply_ledger("s1", [], entries)
        assert failures == [] and recorded == []
        assert expired_entries(entries, matched) == entries

    def test_malformed_entries_are_rejected(self, tmp_path):
        from equiv.ledger import LedgerError, load_ledger

        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"version": "df-bug-ledger/1",
                                   "entries": [{"spec": "s", "level": "nope"}]}))
        with pytest.raises(LedgerError):
            load_ledger(bad)
