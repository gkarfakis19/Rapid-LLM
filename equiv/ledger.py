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

"""T4 bug ledger: recorded, justified exceptions to the golden records.

Before this tier the only way to land a deliberate behavior change was to
regenerate every golden wholesale (``python -m equiv.capture``), which drops
the *evidence* on the floor: the diff shows numbers moving with no statement
of which bug moved them or why the new value is right.

The ledger inverts that. A change is declared FIRST, per
``(spec, level, field)``, with the old value, the new value, the commit that
makes it and a justification. While an entry is live the harness reports the
mismatch as a RECORDED EXCEPTION instead of a failure, so the rest of the
matrix keeps gating. Once the goldens are recaptured the old value stops
matching, the entry becomes dead weight, and
:func:`expired_entries` says so — entries are meant to be pruned in the
recapture commit.

File: ``tests/golden_equiv/bug_ledger.json``::

    {
      "version": "df-bug-ledger/1",
      "entries": [
        {
          "spec": "train:flattened:dp2tp2cp2pp2mb2sp1",
          "level": "structural",
          "field": "bundles/flat/rank/3/compute_micros",
          "old": 1234,
          "new": 1200,
          "commit": "A3",
          "justification": "is_moe_layer restored on the tp-overlap head ...",
          "dlsim_evidence": "optional: bundle still completes"
        }
      ]
    }

``new`` may be omitted to accept any new value (use sparingly); every other
key is required. ``level`` is one of ``structural`` / ``timing`` /
``contract`` — contract entries exist so a KNOWN-broken property can be
carried explicitly rather than by weakening the check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

LEDGER_VERSION = "df-bug-ledger/1"
LEVELS = ("structural", "timing", "contract")

_REQUIRED = ("spec", "level", "field", "old", "commit", "justification")


@dataclass(frozen=True)
class Mismatch:
    """One difference between a golden record and a candidate run."""

    level: str
    field: str
    message: str
    old: Any = None
    new: Any = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.level}] {self.message}"


@dataclass(frozen=True)
class LedgerEntry:
    spec: str
    level: str
    field: str
    old: Any
    commit: str
    justification: str
    new: Any = None
    has_new: bool = False
    dlsim_evidence: str = ""

    def describe(self) -> str:
        target = repr(self.new) if self.has_new else "<any>"
        return (
            f"{self.spec} [{self.level}] {self.field}: {self.old!r} -> "
            f"{target} ({self.commit}: {self.justification})"
        )


class LedgerError(ValueError):
    """The ledger file itself is malformed."""


def default_ledger_path() -> Path:
    return Path(__file__).resolve().parents[1] / "tests" / "golden_equiv" / "bug_ledger.json"


def load_ledger(path: Optional[Path] = None) -> List[LedgerEntry]:
    """Parse the ledger; a missing file means "no recorded exceptions"."""
    path = Path(path) if path is not None else default_ledger_path()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise LedgerError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(raw, dict):
        raise LedgerError(f"{path}: top level must be an object")
    version = raw.get("version")
    if version != LEDGER_VERSION:
        raise LedgerError(f"{path}: version {version!r} != {LEDGER_VERSION!r}")
    entries_raw = raw.get("entries", [])
    if not isinstance(entries_raw, list):
        raise LedgerError(f"{path}: 'entries' must be a list")
    entries: List[LedgerEntry] = []
    seen: set = set()
    for idx, item in enumerate(entries_raw):
        if not isinstance(item, dict):
            raise LedgerError(f"{path}: entry {idx} is not an object")
        missing = [key for key in _REQUIRED if key not in item]
        if missing:
            raise LedgerError(f"{path}: entry {idx} is missing {missing}")
        if item["level"] not in LEVELS:
            raise LedgerError(
                f"{path}: entry {idx} has level {item['level']!r}, expected one of {LEVELS}"
            )
        key = (str(item["spec"]), str(item["level"]), str(item["field"]))
        if key in seen:
            raise LedgerError(f"{path}: duplicate entry for {key}")
        seen.add(key)
        entries.append(
            LedgerEntry(
                spec=str(item["spec"]),
                level=str(item["level"]),
                field=str(item["field"]),
                old=item["old"],
                commit=str(item["commit"]),
                justification=str(item["justification"]),
                new=item.get("new"),
                has_new="new" in item,
                dlsim_evidence=str(item.get("dlsim_evidence", "")),
            )
        )
    return entries


def values_match(a: Any, b: Any, *, rel_tol: float = 1e-9) -> bool:
    """Equality with a relative tolerance for floats (JSON round-trips)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b:
            return True
        denom = max(abs(a), abs(b), 1e-30)
        return abs(a - b) / denom <= rel_tol
    return a == b


def apply_ledger(
    spec_id: str,
    mismatches: Sequence[Mismatch],
    entries: Iterable[LedgerEntry],
) -> Tuple[List[Mismatch], List[str], List[LedgerEntry]]:
    """Split ``mismatches`` into failures and recorded exceptions.

    Returns ``(failures, recorded_descriptions, matched_entries)``. A ledger
    entry absorbs a mismatch only when the spec, level and field agree AND
    the golden value still equals the entry's ``old`` (that is the expiry
    rule) AND — when the entry pins ``new`` — the candidate produced exactly
    that new value. An entry that pins a different ``new`` turns the mismatch
    into a LOUDER failure: the change happened, but not the declared one.
    """
    index: Dict[Tuple[str, str], LedgerEntry] = {
        (entry.level, entry.field): entry for entry in entries if entry.spec == spec_id
    }
    failures: List[Mismatch] = []
    recorded: List[str] = []
    matched: List[LedgerEntry] = []
    for mismatch in mismatches:
        entry = index.get((mismatch.level, mismatch.field))
        if entry is None or not values_match(entry.old, mismatch.old):
            failures.append(mismatch)
            continue
        if entry.has_new and not values_match(entry.new, mismatch.new):
            failures.append(
                Mismatch(
                    level=mismatch.level,
                    field=mismatch.field,
                    message=(
                        f"{mismatch.message} [LEDGER MISMATCH: {entry.commit} declared "
                        f"new={entry.new!r}, run produced {mismatch.new!r}]"
                    ),
                    old=mismatch.old,
                    new=mismatch.new,
                )
            )
            continue
        matched.append(entry)
        recorded.append(f"RECORDED {entry.describe()} | observed: {mismatch.message}")
    return failures, recorded, matched


def expired_entries(
    entries: Iterable[LedgerEntry], matched: Iterable[LedgerEntry]
) -> List[LedgerEntry]:
    """Ledger entries that no longer match anything and must be pruned."""
    live = {(e.spec, e.level, e.field) for e in matched}
    return [e for e in entries if (e.spec, e.level, e.field) not in live]
