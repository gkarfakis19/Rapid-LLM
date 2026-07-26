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

"""Prove that a golden recapture is a STRICT SUPERSET of the previous one.

A gate rebuild is only trustworthy if recapturing the goldens adds pins
without moving any. This walks both golden trees leaf by leaf and classifies
every path:

  ADDED    present only in the new record  -> allowed (that is the point)
  REMOVED  present only in the old record  -> FAILURE, a pin was dropped
  CHANGED  present in both, different value -> FAILURE, a pin moved

``git_rev`` is provenance, not behavior (``equiv.capture`` stamps the HEAD it
ran at), so it is reported separately and never counted as a change.

Usage:
    python -m tools.check_golden_superset OLD_DIR [NEW_DIR]
    python tools/check_golden_superset.py /tmp/golden_baseline tests/golden_equiv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NEW = REPO_ROOT / "tests" / "golden_equiv"

#: leaf paths excluded from the identity check (capture provenance only)
PROVENANCE_LEAVES = ("git_rev",)


def flatten(value: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten nested JSON into ``dotted.path -> leaf`` pairs."""
    out: Dict[str, Any] = {}
    if isinstance(value, dict):
        if not value:
            out[prefix or "<root>"] = {}
            return out
        for key in value:
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(value[key], child))
        return out
    if isinstance(value, list):
        if not value:
            out[prefix or "<root>"] = []
            return out
        for idx, item in enumerate(value):
            out.update(flatten(item, f"{prefix}[{idx}]"))
        return out
    out[prefix or "<root>"] = value
    return out


def _is_provenance(path: str) -> bool:
    tail = path.split(".")[-1].split("[")[0]
    return tail in PROVENANCE_LEAVES


def compare_records(
    old: Dict[str, Any], new: Dict[str, Any]
) -> Tuple[List[str], List[Tuple[str, Any, Any]], List[str], List[Tuple[str, Any, Any]]]:
    """Return (added, changed, removed, provenance_changes) leaf paths."""
    old_flat = flatten(old)
    new_flat = flatten(new)
    added = sorted(set(new_flat) - set(old_flat))
    removed = sorted(set(old_flat) - set(new_flat))
    changed: List[Tuple[str, Any, Any]] = []
    provenance: List[Tuple[str, Any, Any]] = []
    for path in sorted(set(old_flat) & set(new_flat)):
        if old_flat[path] == new_flat[path]:
            continue
        record = (path, old_flat[path], new_flat[path])
        (provenance if _is_provenance(path) else changed).append(record)
    return added, changed, removed, provenance


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def check(old_dir: Path, new_dir: Path, *, verbose: bool = False,
          max_examples: int = 8) -> int:
    old_files = {p.name for p in old_dir.glob("*.json")}
    new_files = {p.name for p in new_dir.glob("*.json")}
    dropped = sorted(old_files - new_files)
    fresh = sorted(new_files - old_files)

    total_added = 0
    total_changed: List[str] = []
    total_removed: List[str] = []
    provenance_specs: List[str] = []
    added_keys: Dict[str, int] = {}

    print(f"[superset] old={old_dir}")
    print(f"[superset] new={new_dir}")
    print(f"[superset] {len(old_files)} old files, {len(new_files)} new files")
    if fresh:
        print(f"[superset] NEW FILES (allowed): {fresh}")
    if dropped:
        print(f"[superset] MISSING FILES (failure): {dropped}")

    for name in sorted(old_files & new_files):
        added, changed, removed, provenance = compare_records(
            _load(old_dir / name), _load(new_dir / name)
        )
        total_added += len(added)
        for path in added:
            # collapse indices/rank ids so the summary stays readable
            shape = ".".join(
                part.split("[")[0] for part in path.split(".")
            )
            added_keys[shape] = added_keys.get(shape, 0) + 1
        if provenance:
            provenance_specs.append(name)
        for path, old_value, new_value in changed:
            total_changed.append(f"{name}: {path}: {old_value!r} -> {new_value!r}")
        for path in removed:
            total_removed.append(f"{name}: {path}")
        if verbose and (added or changed or removed):
            print(f"  {name}: +{len(added)} added, {len(changed)} changed, {len(removed)} removed")

    print()
    print(f"[superset] leaves ADDED   : {total_added}")
    print(f"[superset] leaves CHANGED : {len(total_changed)}")
    print(f"[superset] leaves REMOVED : {len(total_removed)}")
    if provenance_specs:
        print(
            f"[superset] provenance-only changes (git_rev) in {len(provenance_specs)} file(s) "
            "- not counted"
        )
    if added_keys:
        print("[superset] added key shapes:")
        for shape, count in sorted(added_keys.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"    {count:6d}  {shape}")
    for line in total_changed[:max_examples]:
        print(f"    CHANGED {line}")
    if len(total_changed) > max_examples:
        print(f"    ... and {len(total_changed) - max_examples} more")
    for line in total_removed[:max_examples]:
        print(f"    REMOVED {line}")
    if len(total_removed) > max_examples:
        print(f"    ... and {len(total_removed) - max_examples} more")

    ok = not total_changed and not total_removed and not dropped
    print()
    print(
        "[superset] RESULT: STRICT SUPERSET (every pre-existing pinned value "
        "byte-identical, only new keys added)"
        if ok
        else "[superset] RESULT: NOT A SUPERSET"
    )
    return 0 if ok else 1


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("old_dir", help="directory holding the previous goldens")
    parser.add_argument("new_dir", nargs="?", default=str(DEFAULT_NEW),
                        help="directory holding the recaptured goldens")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    return check(Path(args.old_dir), Path(args.new_dir), verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
