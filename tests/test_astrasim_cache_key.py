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

"""The AstraSim result-cache key must cover the emitted DAG (BUG_LEDGER A1).

The old graph-mode key was ``sha256(manifest ‖ system ‖ network ‖ remote_mem
‖ comm_groups)``. The manifest is a per-rank *sorted multiset* of ops with no
dependencies, no order, no p2p peer and no tag, so two workloads that differ
only in dependency structure, node-id priority order or p2p pairing hashed
EQUAL and the second one silently returned the first one's wall time — the
exact class of change every restructure phase makes.

These tests pin the fix: the ET bytes are hashed, name-bound to content, and
mixed into the key; and the *workload signature* (the old formula) survives
separately as the stable per-bundle run identity recorded in
``astra_runs.json``.

Run: ./.venv/bin/python -m pytest tests/test_astrasim_cache_key.py -q
"""

from __future__ import annotations

import json

from astrasim_lib.et_utils import chakra_encode, new_comp_node, new_send_node, pb
from astrasim_lib.integration import (
    _hash_file_bundle,
    _hash_sig,
    _hash_workload_files,
    _record_astrasim_run,
    _workload_et_paths,
)

PREFIX = "llm_graph"


def _write_rank(bundle_dir, rank, nodes):
    path = bundle_dir / f"{PREFIX}.{rank}.et"
    with open(path, "wb") as fh:
        chakra_encode(fh, pb.GlobalMetadata(version="0.0.4"))
        for node in nodes:
            chakra_encode(fh, node)
    return path


def _chain(durations, deps):
    nodes = []
    for idx, micros in enumerate(durations):
        node = new_comp_node(idx, f"op{idx}", micros)
        node.ctrl_deps.extend(deps.get(idx, []))
        nodes.append(node)
    return nodes


def _manifest(bundle_dir, ranks):
    """The manifest as the emitter writes it: per-rank SORTED op multisets."""
    path = bundle_dir / "manifest.json"
    path.write_text(
        json.dumps(
            {"version": "df-astra-manifest/1", "npus": len(ranks), "ranks": ranks},
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return path


def _cache_key(workload_sig: str, et_sig: str) -> str:
    """The key formula under test (integration.run_cache_astrasim)."""
    return _hash_sig(f"df-astra-cache/2|{workload_sig}|{et_sig}")


def _bundle(tmp_path, name, deps):
    """Two ranks with the SAME op multiset; ``deps`` shapes rank 0's DAG."""
    bundle = tmp_path / name
    bundle.mkdir(parents=True, exist_ok=True)
    _write_rank(bundle, 0, _chain([10, 20, 30], deps))
    _write_rank(bundle, 1, _chain([10, 20, 30], deps))
    manifest = _manifest(
        bundle, {"0": [["COMP", 10], ["COMP", 20], ["COMP", 30]],
                 "1": [["COMP", 10], ["COMP", 20], ["COMP", 30]]}
    )
    return bundle, manifest


class TestDagSensitivity:
    def test_manifest_alone_cannot_see_a_dependency_change(self, tmp_path):
        """The bug: identical multiset, different DAG, identical old key."""
        _chain_bundle, chain_manifest = _bundle(tmp_path, "chain", {1: [0], 2: [1]})
        _fan_bundle, fan_manifest = _bundle(tmp_path, "fan", {1: [0], 2: [0]})
        assert chain_manifest.read_bytes() == fan_manifest.read_bytes()
        assert _hash_file_bundle([str(chain_manifest)]) == _hash_file_bundle(
            [str(fan_manifest)]
        )

    def test_cache_key_separates_the_two_dags(self, tmp_path):
        chain_dir, chain_manifest = _bundle(tmp_path, "chain", {1: [0], 2: [1]})
        fan_dir, fan_manifest = _bundle(tmp_path, "fan", {1: [0], 2: [0]})
        chain_key = _cache_key(
            _hash_file_bundle([str(chain_manifest)]),
            _hash_workload_files(_workload_et_paths(str(chain_dir / PREFIX))),
        )
        fan_key = _cache_key(
            _hash_file_bundle([str(fan_manifest)]),
            _hash_workload_files(_workload_et_paths(str(fan_dir / PREFIX))),
        )
        assert chain_key != fan_key

    def test_identical_bundles_still_hit(self, tmp_path):
        a_dir, a_manifest = _bundle(tmp_path, "a", {1: [0], 2: [1]})
        b_dir, b_manifest = _bundle(tmp_path, "b", {1: [0], 2: [1]})
        assert _cache_key(
            _hash_file_bundle([str(a_manifest)]),
            _hash_workload_files(_workload_et_paths(str(a_dir / PREFIX))),
        ) == _cache_key(
            _hash_file_bundle([str(b_manifest)]),
            _hash_workload_files(_workload_et_paths(str(b_dir / PREFIX))),
        )

    def test_p2p_tag_change_moves_the_key(self, tmp_path):
        """Tags are invisible to the manifest but decide who pairs with whom."""
        keys = []
        for name, tag in (("tag7", 7), ("tag8", 8)):
            bundle = tmp_path / name
            bundle.mkdir(parents=True, exist_ok=True)
            _write_rank(bundle, 0, [new_send_node(0, "s", 64, dst_rank=1, tag=tag)])
            _write_rank(bundle, 1, [new_comp_node(0, "c", 10)])
            manifest = _manifest(bundle, {"0": [["SEND", 64]], "1": [["COMP", 10]]})
            keys.append(
                _cache_key(
                    _hash_file_bundle([str(manifest)]),
                    _hash_workload_files(_workload_et_paths(str(bundle / PREFIX))),
                )
            )
        assert keys[0] != keys[1]

    def test_swapping_two_ranks_traces_moves_the_key(self, tmp_path):
        """Name-bound hashing: rank 0 and rank 1 exchanging traces is a change."""
        straight = tmp_path / "straight"
        straight.mkdir()
        _write_rank(straight, 0, _chain([10], {}))
        _write_rank(straight, 1, _chain([20], {}))
        swapped = tmp_path / "swapped"
        swapped.mkdir()
        _write_rank(swapped, 0, _chain([20], {}))
        _write_rank(swapped, 1, _chain([10], {}))
        assert _hash_workload_files(
            _workload_et_paths(str(straight / PREFIX))
        ) != _hash_workload_files(_workload_et_paths(str(swapped / PREFIX)))


class TestWorkloadSignatureIdentity:
    def test_signature_ignores_the_dag(self, tmp_path):
        """The run identity stays stable when only emission order changes.

        That is what keeps ``astra_runs.json`` (and therefore the golden
        records' per-run wall seconds) addressable across the restructure.
        """
        _a, a_manifest = _bundle(tmp_path, "a", {1: [0], 2: [1]})
        _b, b_manifest = _bundle(tmp_path, "b", {1: [0], 2: [0]})
        assert _hash_file_bundle([str(a_manifest)]) == _hash_file_bundle([str(b_manifest)])

    def test_record_accumulates_one_entry_per_signature(self, tmp_path):
        record = tmp_path / "astra_runs.json"
        _record_astrasim_run(str(record), "sigA", {"per_node_sec": [1.0, 2.0], "max_sec": 2.0})
        _record_astrasim_run(str(record), "sigB", {"per_node_sec": [3.0], "max_sec": 3.0})
        _record_astrasim_run(str(record), "sigA", {"per_node_sec": [1.0, 2.0], "max_sec": 2.0})
        data = json.loads(record.read_text())
        assert set(data) == {"sigA", "sigB"}
        assert data["sigA"]["max_sec"] == 2.0

    def test_record_path_none_is_a_no_op(self, tmp_path):
        _record_astrasim_run(None, "sig", {"per_node_sec": [1.0], "max_sec": 1.0})
        assert list(tmp_path.iterdir()) == []


class TestWorkloadEtDiscovery:
    def test_only_matching_prefix_and_rank_files_are_collected(self, tmp_path):
        bundle = tmp_path / "b"
        bundle.mkdir()
        _write_rank(bundle, 0, _chain([1], {}))
        _write_rank(bundle, 10, _chain([1], {}))
        (bundle / "manifest.json").write_text("{}")
        (bundle / "llm_graph.notarank.et").write_text("x")
        (bundle / "other.0.et").write_text("x")
        found = [p.rsplit("/", 1)[-1] for p in _workload_et_paths(str(bundle / PREFIX))]
        assert sorted(found) == ["llm_graph.0.et", "llm_graph.10.et"]

    def test_missing_directory_is_empty(self, tmp_path):
        assert _workload_et_paths(str(tmp_path / "absent" / PREFIX)) == []
        assert _workload_et_paths(None) == []
