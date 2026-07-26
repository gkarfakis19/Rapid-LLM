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
against the current code and compared at four levels: per-rank op multisets,
dependency-DAG hashes, exact AstraSim per-rank wall seconds, and end-to-end
reported times. Regenerate goldens deliberately with ``python -m
equiv.capture`` and commit the diff with justification.

Run: pytest tests/test_equiv_golden.py -q          (whole matrix, ~2-4 min)
     pytest tests/test_equiv_golden.py -q -k flat  (subset)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from equiv.capture import GOLDEN_DIR, sanitize
from equiv.configs import MATRIX, MATRIX_BY_ID
from equiv.runner import compare_observation, run_specs

pytestmark = pytest.mark.equiv_golden


def _specs_with_goldens():
    if not GOLDEN_DIR.is_dir():
        return []
    return [
        spec
        for spec in MATRIX
        if (GOLDEN_DIR / f"{sanitize(spec.spec_id)}.json").exists()
    ]


_SPECS = _specs_with_goldens()


@pytest.fixture(scope="session")
def observations(tmp_path_factory):
    if not _SPECS:
        pytest.skip("No goldens recorded; run `python -m equiv.capture` first")
    tmp_root = tmp_path_factory.mktemp("equiv_runs")
    return run_specs(_SPECS, tmp_root=str(tmp_root))


@pytest.mark.parametrize("spec_id", [s.spec_id for s in _SPECS])
def test_matches_golden(spec_id, observations):
    golden = json.loads(
        (GOLDEN_DIR / f"{sanitize(spec_id)}.json").read_text()
    )
    obs = observations[spec_id]
    problems = compare_observation(golden, obs)
    assert not problems, (
        f"{spec_id}: behavior diverged from golden "
        f"(run dir: {obs.run_dir}):\n" + "\n".join(problems)
    )
