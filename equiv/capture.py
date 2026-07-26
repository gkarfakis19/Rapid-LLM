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

"""Capture golden behavior records for the equivalence matrix.

Usage:
    python -m equiv.capture                       # full matrix
    python -m equiv.capture --specs id1 id2 ...   # subset
    python -m equiv.capture --list                # show matrix ids

Goldens land in tests/golden_equiv/<sanitized-spec-id>.json plus an
index.json recording the capture git revision and per-spec status.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .configs import MATRIX, MATRIX_BY_ID
from .runner import REPO_ROOT, run_specs

GOLDEN_DIR = REPO_ROOT / "tests" / "golden_equiv"


def sanitize(spec_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", spec_id)


def git_rev() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", nargs="*", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--tmp-root", default=None)
    args = parser.parse_args(argv)

    if args.list:
        for spec in MATRIX:
            print(spec.spec_id)
        return 0

    if args.specs:
        missing = [s for s in args.specs if s not in MATRIX_BY_ID]
        if missing:
            parser.error(f"Unknown spec ids: {missing}")
        specs = [MATRIX_BY_ID[s] for s in args.specs]
    else:
        specs = list(MATRIX)

    tmp_root = args.tmp_root or tempfile.mkdtemp(prefix="equiv_capture_")
    rev = git_rev()
    print(f"[equiv] Capturing {len(specs)} specs at {rev} (tmp: {tmp_root})")
    observations = run_specs(specs, tmp_root=tmp_root, max_workers=args.max_workers)

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    index: Dict[str, Dict[str, str]] = {}
    if (GOLDEN_DIR / "index.json").exists():
        index = json.loads((GOLDEN_DIR / "index.json").read_text())
    ok = errors = 0
    for spec in specs:
        obs = observations[spec.spec_id]
        golden = obs.to_golden(rev)
        path = GOLDEN_DIR / f"{sanitize(spec.spec_id)}.json"
        path.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")
        index[spec.spec_id] = {"file": path.name, "status": golden["status"], "git_rev": rev}
        if golden["status"] == "ok":
            ok += 1
            deadlocked = [k for k, v in golden["dlsim_ok"].items() if not v]
            unpaired = [k for k, v in golden["p2p_ok"].items() if not v]
            extra = f" DLSIM-FAIL:{deadlocked}" if deadlocked else ""
            extra += f" P2P-FAIL:{unpaired}" if unpaired else ""
            print(f"[equiv]  ok    {spec.spec_id} total={golden['total_time']}{extra}")
        else:
            errors += 1
            print(f"[equiv]  ERROR {spec.spec_id}")
    (GOLDEN_DIR / "index.json").write_text(
        json.dumps(index, indent=1, sort_keys=True) + "\n"
    )
    print(f"[equiv] Done: {ok} ok, {errors} error (pinned), goldens in {GOLDEN_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
