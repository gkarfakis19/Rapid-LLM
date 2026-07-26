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

"""Golden-equivalence gates for the AstraSim execution rewrite.

Every spec in ``equiv.configs.MATRIX`` that has a recorded golden is re-run
against the current code (with the AstraSim result cache FORCED OFF) and
compared tier by tier:

* ``test_structural`` — **T1**: id-independent structure. Per-rank op
  multisets and dependency-DAG hashes, plus the quantities that make a
  failure diagnosable: compute microseconds, byte histograms by op kind and
  by interconnect axis, collectives grouped by resolved member set, transfer
  counts, payload-weighted critical-path length, ``manifest.json`` content,
  ``comm_groups.json`` member sets, and the per-GPU memory peaks.
* ``test_timing`` — **T2**: exact AstraSim per-rank wall seconds and the
  end-to-end reported times.
* ``test_contract`` — **T3**: properties, not comparisons. Every bundle must
  run to completion under the AstraSim scheduling contract and its p2p tags
  must pair bijectively. These fail on ANY violation, golden or not.
* ``test_reemission_deterministic`` — **T3**: a second, independent run must
  re-emit canonically identical bundles.
* ``test_bug_ledger_entries_are_live`` — **T4**: recorded exceptions expire
  when their ``old`` value stops matching and must then be pruned.

Regenerate goldens deliberately with ``python -m equiv.capture`` and commit
the diff with justification; prefer declaring the change in
``tests/golden_equiv/bug_ledger.json`` first.

Run: pytest tests/test_equiv_golden.py -q          (whole matrix)
     pytest tests/test_equiv_golden.py -q -k flat  (subset)
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import pytest

from equiv.capture import GOLDEN_DIR, sanitize
from equiv.configs import MATRIX
from equiv.ledger import LedgerEntry, Mismatch, expired_entries, load_ledger
from equiv.runner import compare_all, compare_determinism, compare_observation, run_specs

pytestmark = pytest.mark.equiv_golden

_Comparison = Tuple[List[Mismatch], List[str], List[LedgerEntry]]


def _specs_with_goldens():
    if not GOLDEN_DIR.is_dir():
        return []
    return [
        spec
        for spec in MATRIX
        if (GOLDEN_DIR / f"{sanitize(spec.spec_id)}.json").exists()
    ]


_SPECS = _specs_with_goldens()
_SPEC_IDS = [s.spec_id for s in _SPECS]


def _golden(spec_id: str) -> Dict:
    return json.loads((GOLDEN_DIR / f"{sanitize(spec_id)}.json").read_text())


@pytest.fixture(scope="session")
def observations(tmp_path_factory):
    if not _SPECS:
        pytest.skip("No goldens recorded; run `python -m equiv.capture` first")
    tmp_root = tmp_path_factory.mktemp("equiv_runs")
    return run_specs(_SPECS, tmp_root=str(tmp_root))


@pytest.fixture(scope="session")
def repeat_observations(tmp_path_factory):
    """A second, independent run of the whole matrix (T3 determinism)."""
    if not _SPECS:
        pytest.skip("No goldens recorded; run `python -m equiv.capture` first")
    tmp_root = tmp_path_factory.mktemp("equiv_runs_repeat")
    return run_specs(_SPECS, tmp_root=str(tmp_root))


@pytest.fixture(scope="session")
def comparisons(observations) -> Tuple[Dict[str, _Comparison], List[LedgerEntry]]:
    entries = load_ledger()
    results: Dict[str, _Comparison] = {}
    for spec_id in _SPEC_IDS:
        results[spec_id] = compare_all(_golden(spec_id), observations[spec_id], ledger=entries)
    recorded = [line for _f, rec, _m in results.values() for line in rec]
    if recorded:
        print(f"\n[equiv] T4 ledger absorbed {len(recorded)} difference(s):")
        for line in recorded:
            print("  " + line)
    return results, entries


def _failures(comparisons, spec_id: str, level: str) -> List[str]:
    results, _entries = comparisons
    failures, _recorded, _matched = results[spec_id]
    return [str(problem) for problem in failures if problem.level == level]


@pytest.mark.parametrize("spec_id", _SPEC_IDS)
def test_structural(spec_id, comparisons, observations):
    """T1: id-independent structure (canonical bundles + memory peaks)."""
    problems = _failures(comparisons, spec_id, "structural")
    assert not problems, (
        f"{spec_id}: structure diverged from golden "
        f"(run dir: {observations[spec_id].run_dir}):\n" + "\n".join(problems)
    )


@pytest.mark.parametrize("spec_id", _SPEC_IDS)
def test_timing(spec_id, comparisons, observations):
    """T2: AstraSim per-rank wall seconds and end-to-end reported times."""
    problems = _failures(comparisons, spec_id, "timing")
    assert not problems, (
        f"{spec_id}: predictions diverged from golden "
        f"(run dir: {observations[spec_id].run_dir}):\n" + "\n".join(problems)
    )


@pytest.mark.parametrize("spec_id", _SPEC_IDS)
def test_contract(spec_id, comparisons, observations):
    """T3: contracts every emitted bundle must satisfy, golden or not."""
    problems = _failures(comparisons, spec_id, "contract")
    assert not problems, (
        f"{spec_id}: AstraSim workload contract violated "
        f"(run dir: {observations[spec_id].run_dir}):\n" + "\n".join(problems)
    )


@pytest.mark.parametrize("spec_id", _SPEC_IDS)
def test_reemission_deterministic(spec_id, observations, repeat_observations):
    """T3: two independent runs re-emit canonically identical bundles."""
    first = observations[spec_id]
    second = repeat_observations[spec_id]
    if not first.success or not second.success:
        pytest.skip(f"{spec_id}: pinned as a failing run; determinism is not observable")
    problems = [str(p) for p in compare_determinism(first, second)]
    assert not problems, (
        f"{spec_id}: emission is not deterministic across runs "
        f"(run dirs: {first.run_dir} vs {second.run_dir}):\n" + "\n".join(problems)
    )


def test_bug_ledger_entries_are_live(comparisons):
    """T4: an entry whose ``old`` stopped matching is dead and must be pruned."""
    results, entries = comparisons
    if not entries:
        return
    matched = [entry for _f, _r, entries_matched in results.values() for entry in entries_matched]
    dead = expired_entries(entries, matched)
    assert not dead, (
        "bug_ledger.json has entries that no longer match any difference "
        "(the goldens moved past them — prune them):\n"
        + "\n".join("  " + entry.describe() for entry in dead)
    )


@pytest.mark.parametrize("spec_id", _SPEC_IDS)
def test_matches_golden(spec_id, observations):
    """Whole-observation gate: the union of T1-T4 in one assertion."""
    problems = compare_observation(_golden(spec_id), observations[spec_id])
    assert not problems, (
        f"{spec_id}: behavior diverged from golden "
        f"(run dir: {observations[spec_id].run_dir}):\n" + "\n".join(problems)
    )
