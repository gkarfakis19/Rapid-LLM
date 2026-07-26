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

"""Run equivalence specs and collect observable behavior.

Each spec runs as a ``run_perf.py`` subprocess with an isolated cache and
persisted AstraSim artifacts. The collected ``RunObservation`` is the full
set of behaviors the harness pins, organized in the restructure gate tiers:

**T1 structural** (exact, always; all ID-INDEPENDENT)
    canonical forms of every emitted Chakra ET bundle — op multisets,
    dependency-DAG Merkle hashes, per-rank compute-microsecond totals, byte
    histograms by op kind and by interconnect axis, collectives grouped by
    resolved member set, transfer counts, payload-weighted critical-path
    length, ``manifest.json`` content, parsed ``comm_groups.json`` member
    sets (see ``equiv.canonical``) — plus the per-GPU memory peaks parsed
    from ``output/*/memory-summary/memory_capacity_comparison.txt``.

**T2 predictions**
    exact AstraSim per-rank wall seconds for every bundle (from the
    ``astra_runs.json`` result log written next to the ETs) and the
    end-to-end reported time parsed from the results file.

**T3 contracts** (boolean, always-on, golden-independent)
    causal completability of every bundle (``equiv.dlsim``) and the p2p
    tag-pairing bijection (``program.shadow.p2p_pairing_problems``). These
    fail on ANY candidate violation — a bundle that deadlocks is a bug
    whether or not the recorded golden happened to deadlock too.

**T4 ledger**
    declared, justified exceptions (``equiv.ledger``).

``run_specs`` FORCES ``RAPID_ASTRA_CACHE_MODE=NO_CACHE``. Any cache mode that
can return a stored result makes the harness gate the cache instead of the
code; combined with BUG_LEDGER A1 (the key omitted the ET bytes) that was the
single most dangerous hazard in the whole rewrite.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from program.shadow import p2p_pairing_problems
from validation_scripts.validation_helpers import ValidationSpec, run_validation_suite

from .canonical import canonicalize_bundle, diff_bundles_detailed
from .configs import EquivSpec
from .dlsim import BundleSim
from .ledger import LedgerEntry, Mismatch, apply_ledger, load_ledger

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_CONFIG = (
    REPO_ROOT / "validation_scripts" / "validation_configs" / "model-config" / "Llama2-7B.yaml"
)
BASE_HW_CONFIG = (
    REPO_ROOT / "validation_scripts" / "validation_configs" / "hardware-config" / "a100_80GB.yaml"
)

#: the only honest cache mode for a validation run (see module docstring)
CACHE_MODE = "NO_CACHE"

_TOTAL_TIME_RE = re.compile(r"^Total Time:\s*([0-9.eE+-]+)", re.MULTILINE)
_INFER_TIME_RE = re.compile(r"^Inference Time for batch:\s*([0-9.eE+-]+)", re.MULTILINE)
_PREFILL_RE = re.compile(r"^Prefill Time:\s*([0-9.eE+-]+)", re.MULTILINE)
_DECODE_RE = re.compile(r"^Decode Time:\s*([0-9.eE+-]+)", re.MULTILINE)

_GIB_RE = re.compile(r"^(-?[0-9]+(?:\.[0-9]+)?)\s*GiB$")


@dataclass
class RunObservation:
    spec_id: str
    success: bool
    returncode: Optional[int]
    error_tail: str = ""
    total_time: Optional[float] = None
    phase_times: Dict[str, float] = field(default_factory=dict)
    bundles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    astra_times: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    dlsim_ok: Dict[str, bool] = field(default_factory=dict)
    memory: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    p2p_ok: Dict[str, bool] = field(default_factory=dict)
    p2p_problems: Dict[str, List[str]] = field(default_factory=dict)
    run_dir: Optional[str] = None

    def to_golden(self, git_rev: str) -> Dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "git_rev": git_rev,
            "status": "ok" if self.success else "error",
            "error_tail": "" if self.success else self.error_tail,
            "total_time": self.total_time,
            "phase_times": self.phase_times,
            "bundles": self.bundles,
            "astra_times": self.astra_times,
            "dlsim_ok": self.dlsim_ok,
            "memory": self.memory,
            "p2p_ok": self.p2p_ok,
        }


def _find_bundle_dirs(output_root: Path) -> Dict[str, Path]:
    """Locate persisted ET bundle directories, keyed by a stable label."""
    bundles: Dict[str, Path] = {}
    if not output_root.is_dir():
        return bundles
    for mode_dir in sorted(output_root.iterdir()):
        if not mode_dir.is_dir():
            continue
        for name, label in (("astra_flat", "flat"), ("astra_hier", "hier")):
            base = mode_dir / name
            if not base.is_dir():
                continue
            if any(base.glob("llm_graph.*.et")):
                bundles[label] = base
            for sub in sorted(base.iterdir()):
                if sub.is_dir() and any(sub.glob("llm_graph.*.et")):
                    bundles[f"{label}/{sub.name}"] = sub
    return bundles


def _read_astra_times(bundle_dir: Path) -> Optional[Dict[str, Any]]:
    """Read exact per-rank AstraSim seconds from the bundle's run log.

    ``astra_runs.json`` is written by ``astrasim_lib.integration`` for every
    invocation, in every cache mode, keyed by the WORKLOAD SIGNATURE (manifest
    + system + network + remote memory + comm groups). Multi-run flows (grad
    accumulation's no-DP + final dispatchers, inference prefill + decode
    samples) share one artifact dir, so entries accumulate; that choreography
    is part of the pinned surface: record every entry, keyed and ordered by
    the signature, so all runs' wall seconds are gated.

    The signature deliberately excludes the ET digest that the *cache* key now
    includes (A1): it identifies which simulation was asked for, which is the
    stable run identity across emission-order changes.
    """
    runs_path = bundle_dir / "astra_runs.json"
    if not runs_path.exists():
        return None
    try:
        with open(runs_path) as fh:
            cache = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    entries = {
        key: value
        for key, value in cache.items()
        if isinstance(value, dict) and "per_node_sec" in value
    }
    if not entries:
        return None
    if len(entries) == 1:
        entry = next(iter(entries.values()))
        return {
            "per_rank_sec": list(entry.get("per_node_sec", [])),
            "max_sec": entry.get("max_sec"),
        }
    return {
        "runs": [
            {
                "sig": key,
                "per_rank_sec": list(value.get("per_node_sec", [])),
                "max_sec": value.get("max_sec"),
            }
            for key, value in sorted(entries.items())
        ]
    }


def _parse_times(run_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"total_time": None, "phase_times": {}}
    for results in run_dir.glob("output/*/*_results.txt"):
        text = results.read_text()
        m = _TOTAL_TIME_RE.search(text)
        if m:
            out["total_time"] = float(m.group(1))
        m = _INFER_TIME_RE.search(text)
        if m:
            out["total_time"] = float(m.group(1))
        m = _PREFILL_RE.search(text)
        if m:
            out["phase_times"]["prefill"] = float(m.group(1))
        m = _DECODE_RE.search(text)
        if m:
            out["phase_times"]["decode"] = float(m.group(1))
    return out


def _memory_field_key(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
    return slug or "field"


def _parse_memory_summaries(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Parse every ``memory-summary/memory_capacity_comparison.txt``.

    Memory peaks are ID-INDEPENDENT and were ungated entirely before the T1
    rebuild, even though the memory replay reads the same Program the emitter
    does; a misattributed layer (BUG_LEDGER A3) moves the reported peak
    without moving a single wall-clock number. Values are recorded as parsed
    GiB floats (the file formats them to two decimals, so equality is exact)
    plus the raw ``[WARN]`` lines, which is where a capacity violation shows.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(run_dir.glob("output/*/memory-summary/memory_capacity_comparison.txt")):
        label = path.parent.parent.name
        fields: Dict[str, Any] = {}
        warnings: List[str] = []
        try:
            text = path.read_text()
        except OSError as exc:  # noqa: BLE001 - recorded, not fatal
            out[label] = {"error": str(exc)}
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("[WARN]"):
                warnings.append(line)
                continue
            key, sep, value = line.partition(":")
            if not sep:
                warnings.append(line)
                continue
            value = value.strip()
            match = _GIB_RE.match(value)
            fields[_memory_field_key(key)] = float(match.group(1)) if match else value
        out[label] = {"fields": dict(sorted(fields.items())), "warnings": warnings}
    return out


def observe_run(spec_id: str, result: Any) -> RunObservation:
    """Build a RunObservation from a ValidationResult (kept tmp dir)."""
    obs = RunObservation(
        spec_id=spec_id,
        success=bool(result.success),
        returncode=result.returncode,
    )
    if not result.success:
        tail = (result.raw_output or "").strip().splitlines()
        obs.error_tail = "\n".join(tail[-15:])
    run_dir: Optional[Path] = None
    if result.model_config_used:
        run_dir = Path(result.model_config_used).parent
    if run_dir is None or not run_dir.is_dir():
        return obs
    obs.run_dir = str(run_dir)

    times = _parse_times(run_dir)
    obs.total_time = times["total_time"]
    obs.phase_times = times["phase_times"]
    obs.memory = _parse_memory_summaries(run_dir)

    for label, bundle_dir in _find_bundle_dirs(run_dir / "output").items():
        try:
            obs.bundles[label] = canonicalize_bundle(str(bundle_dir)).summary()
        except Exception as exc:  # noqa: BLE001 - recorded, not fatal
            obs.bundles[label] = {"error": str(exc)}
        times_entry = _read_astra_times(bundle_dir)
        if times_entry is not None:
            obs.astra_times[label] = times_entry
        try:
            obs.dlsim_ok[label] = BundleSim(str(bundle_dir)).simulate().completed
        except Exception as exc:  # noqa: BLE001
            obs.dlsim_ok[label] = False
            obs.bundles.setdefault(label, {})
            obs.bundles[label]["dlsim_error"] = str(exc)
        try:
            problems = p2p_pairing_problems(str(bundle_dir), label=label)
        except Exception as exc:  # noqa: BLE001
            problems = [f"[{label}] p2p pairing check raised: {exc}"]
        obs.p2p_problems[label] = problems
        obs.p2p_ok[label] = not problems
    return obs


def run_specs(
    specs: Sequence[EquivSpec],
    *,
    tmp_root: str,
    max_workers: Optional[int] = None,
) -> Dict[str, RunObservation]:
    os.makedirs(tmp_root, exist_ok=True)
    vspecs = [
        ValidationSpec(
            label=s.spec_id,
            model_overrides=s.model_overrides(),
            hardware_overrides=s.hardware_overrides(),
        )
        for s in specs
    ]
    env_overrides = {
        "RAPID_PERSIST_ASTRASIM_ARTIFACTS": "1",
        # Explicit, not inherited: a harness that can read a cached AstraSim
        # result gates the cache, not the code (module docstring; A1).
        "RAPID_ASTRA_CACHE_MODE": CACHE_MODE,
    }
    # validation_helpers decides tmp cleanup from the *parent* process env.
    prev_keep = os.environ.get("RAPID_VALIDATION_KEEP_TMP")
    prev_mode = os.environ.get("RAPID_ASTRA_CACHE_MODE")
    os.environ["RAPID_VALIDATION_KEEP_TMP"] = "1"
    os.environ["RAPID_ASTRA_CACHE_MODE"] = CACHE_MODE
    try:
        results = run_validation_suite(
            vspecs,
            base_model_config_path=str(BASE_MODEL_CONFIG),
            base_hardware_config_path=str(BASE_HW_CONFIG),
            result_parser=lambda _out, _spec: {},
            tmp_root=tmp_root,
            max_workers=max_workers,
            env_overrides=env_overrides,
            cache_mode=CACHE_MODE,
        )
    finally:
        for name, prev in (
            ("RAPID_VALIDATION_KEEP_TMP", prev_keep),
            ("RAPID_ASTRA_CACHE_MODE", prev_mode),
        ):
            if prev is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prev
    observations: Dict[str, RunObservation] = {}
    for spec, result in zip(specs, results):
        observations[spec.spec_id] = observe_run(spec.spec_id, result)
    return observations


# ---------------------------------------------------------------------------
# Comparison: one tier per function
# ---------------------------------------------------------------------------


def _close(a: Optional[float], b: Optional[float], rel_tol: float) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if a == b:
        return True
    denom = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / denom <= rel_tol


def compare_status(golden: Dict[str, Any], obs: RunObservation) -> List[Mismatch]:
    """Did the run get as far as the golden did? (gate for every other tier)"""
    golden_status = golden.get("status", "ok")
    obs_status = "ok" if obs.success else "error"
    if golden_status != obs_status:
        return [
            Mismatch(
                level="contract",
                field="status",
                message=(
                    f"status: golden={golden_status} candidate={obs_status} "
                    f"(candidate tail: {obs.error_tail[-400:]})"
                ),
                old=golden_status,
                new=obs_status,
            )
        ]
    return []


def compare_structural(golden: Dict[str, Any], obs: RunObservation) -> List[Mismatch]:
    """T1: id-independent structure — canonical bundles and memory peaks."""
    problems: List[Mismatch] = []
    golden_bundles = golden.get("bundles") or {}
    if set(golden_bundles) != set(obs.bundles):
        problems.append(
            Mismatch(
                level="structural",
                field="bundle_set",
                message=(
                    f"bundle set: golden={sorted(golden_bundles)} "
                    f"candidate={sorted(obs.bundles)}"
                ),
                old=sorted(golden_bundles),
                new=sorted(obs.bundles),
            )
        )
    for label in sorted(set(golden_bundles) & set(obs.bundles)):
        for bfield, message, old, new in diff_bundles_detailed(
            golden_bundles[label], obs.bundles[label], label=label
        ):
            problems.append(
                Mismatch(
                    level="structural",
                    field=f"bundles/{label}/{bfield}",
                    message=message,
                    old=old,
                    new=new,
                )
            )

    golden_memory = golden.get("memory") or {}
    if set(golden_memory) != set(obs.memory):
        problems.append(
            Mismatch(
                level="structural",
                field="memory_set",
                message=(
                    f"memory summary set: golden={sorted(golden_memory)} "
                    f"candidate={sorted(obs.memory)}"
                ),
                old=sorted(golden_memory),
                new=sorted(obs.memory),
            )
        )
    for label in sorted(set(golden_memory) & set(obs.memory)):
        gold_entry = golden_memory[label] or {}
        cand_entry = obs.memory[label] or {}
        gold_fields = gold_entry.get("fields") or {}
        cand_fields = cand_entry.get("fields") or {}
        for key in sorted(set(gold_fields) | set(cand_fields)):
            gval, cval = gold_fields.get(key), cand_fields.get(key)
            if gval == cval:
                continue
            problems.append(
                Mismatch(
                    level="structural",
                    field=f"memory/{label}/{key}",
                    message=f"memory[{label}] {key}: golden={gval} candidate={cval}",
                    old=gval,
                    new=cval,
                )
            )
        if (gold_entry.get("warnings") or []) != (cand_entry.get("warnings") or []):
            problems.append(
                Mismatch(
                    level="structural",
                    field=f"memory/{label}/warnings",
                    message=(
                        f"memory[{label}] warnings: golden={gold_entry.get('warnings')} "
                        f"candidate={cand_entry.get('warnings')}"
                    ),
                    old=gold_entry.get("warnings"),
                    new=cand_entry.get("warnings"),
                )
            )
    return problems


def compare_timing(
    golden: Dict[str, Any], obs: RunObservation, *, rel_tol: float = 1e-9
) -> List[Mismatch]:
    """T2: end-to-end reported times and exact AstraSim per-rank wall seconds."""
    problems: List[Mismatch] = []

    if not _close(golden.get("total_time"), obs.total_time, rel_tol):
        problems.append(
            Mismatch(
                level="timing",
                field="total_time",
                message=f"total_time: golden={golden.get('total_time')} candidate={obs.total_time}",
                old=golden.get("total_time"),
                new=obs.total_time,
            )
        )
    for phase, value in sorted((golden.get("phase_times") or {}).items()):
        candidate = obs.phase_times.get(phase)
        if not _close(value, candidate, rel_tol):
            problems.append(
                Mismatch(
                    level="timing",
                    field=f"phase/{phase}",
                    message=f"phase {phase}: golden={value} candidate={candidate}",
                    old=value,
                    new=candidate,
                )
            )

    def _diff_rank_seconds(
        label: str, field_prefix: str, gold_entry: Dict[str, Any], cand_entry: Dict[str, Any]
    ) -> None:
        gold_ranks = gold_entry.get("per_rank_sec") or []
        cand_ranks = cand_entry.get("per_rank_sec") or []
        if len(gold_ranks) != len(cand_ranks):
            problems.append(
                Mismatch(
                    level="timing",
                    field=f"{field_prefix}/rank_count",
                    message=(
                        f"astra[{label}]: rank count golden={len(gold_ranks)} "
                        f"candidate={len(cand_ranks)}"
                    ),
                    old=len(gold_ranks),
                    new=len(cand_ranks),
                )
            )
            return
        for idx, (a, b) in enumerate(zip(gold_ranks, cand_ranks)):
            if not _close(a, b, rel_tol):
                problems.append(
                    Mismatch(
                        level="timing",
                        field=f"{field_prefix}/rank{idx}",
                        message=f"astra[{label}] rank {idx}: golden={a} candidate={b}",
                        old=a,
                        new=b,
                    )
                )

    golden_astra = golden.get("astra_times") or {}
    if set(golden_astra) != set(obs.astra_times):
        problems.append(
            Mismatch(
                level="timing",
                field="astra_set",
                message=(
                    f"astra bundle set: golden={sorted(golden_astra)} "
                    f"candidate={sorted(obs.astra_times)}"
                ),
                old=sorted(golden_astra),
                new=sorted(obs.astra_times),
            )
        )
    for label in sorted(set(golden_astra) & set(obs.astra_times)):
        gold_entry, cand_entry = golden_astra[label], obs.astra_times[label]
        gold_runs = gold_entry.get("runs")
        cand_runs = cand_entry.get("runs")
        if (gold_runs is None) != (cand_runs is None):
            problems.append(
                Mismatch(
                    level="timing",
                    field=f"astra/{label}/multiplicity",
                    message=(
                        f"astra[{label}]: run multiplicity differs (golden "
                        f"{'multi' if gold_runs else 'single'}, candidate "
                        f"{'multi' if cand_runs else 'single'})"
                    ),
                    old="multi" if gold_runs else "single",
                    new="multi" if cand_runs else "single",
                )
            )
            continue
        if gold_runs is None:
            _diff_rank_seconds(label, f"astra/{label}", gold_entry, cand_entry)
            continue
        if len(gold_runs) != len(cand_runs):
            problems.append(
                Mismatch(
                    level="timing",
                    field=f"astra/{label}/run_count",
                    message=(
                        f"astra[{label}]: run count golden={len(gold_runs)} "
                        f"candidate={len(cand_runs)}"
                    ),
                    old=len(gold_runs),
                    new=len(cand_runs),
                )
            )
            continue
        for run_idx, (grun, crun) in enumerate(zip(gold_runs, cand_runs)):
            if grun.get("sig") != crun.get("sig"):
                problems.append(
                    Mismatch(
                        level="timing",
                        field=f"astra/{label}/run{run_idx}/sig",
                        message=f"astra[{label}] run {run_idx}: signature differs",
                        old=grun.get("sig"),
                        new=crun.get("sig"),
                    )
                )
                break
            _diff_rank_seconds(
                f"{label}/run{run_idx}", f"astra/{label}/run{run_idx}", grun, crun
            )
    return problems


def compare_contract(golden: Dict[str, Any], obs: RunObservation) -> List[Mismatch]:
    """T3: properties every emitted bundle must satisfy, golden or not.

    Deliberately asymmetric with T1/T2: these do not ask "did this change",
    they ask "is this valid". A candidate bundle that deadlocks is a failure
    even if the recorded golden deadlocked too — pinning a deadlock as
    expected behavior is how a silent AstraSim hang becomes permanent.
    """
    problems: List[Mismatch] = []
    for label in sorted(obs.dlsim_ok):
        if not obs.dlsim_ok[label]:
            golden_ok = (golden.get("dlsim_ok") or {}).get(label)
            note = "" if golden_ok else " (golden did not complete either)"
            problems.append(
                Mismatch(
                    level="contract",
                    field=f"dlsim/{label}",
                    message=(
                        f"dlsim[{label}]: bundle does not run to completion under the "
                        f"AstraSim scheduling contract{note}"
                    ),
                    old=golden_ok,
                    new=False,
                )
            )
    for label in sorted(golden.get("dlsim_ok") or {}):
        if label not in obs.dlsim_ok:
            problems.append(
                Mismatch(
                    level="contract",
                    field=f"dlsim/{label}",
                    message=f"dlsim[{label}]: bundle present in golden, missing from candidate",
                    old=(golden.get("dlsim_ok") or {}).get(label),
                    new=None,
                )
            )
    for label in sorted(obs.p2p_problems):
        detail = obs.p2p_problems[label]
        if detail:
            problems.append(
                Mismatch(
                    level="contract",
                    field=f"p2p/{label}",
                    message=(
                        f"p2p tag pairing[{label}]: " + "; ".join(detail[:5])
                    ),
                    old=True,
                    new=False,
                )
            )
    return problems


def compare_determinism(
    first: RunObservation, second: RunObservation, *, rel_tol: float = 1e-9
) -> List[Mismatch]:
    """T3: two independent runs must re-emit the same CANONICAL bundles.

    Compared canonically rather than with ``filecmp``: determinism is a claim
    about the modeling content and the dependency structure, not about
    protobuf byte layout. Because the two runs are separate processes this
    also covers the PYTHONHASHSEED-dependent set/dict iteration order that
    the in-process builder sweeps (``RAPID_*_DIFF=1``) cannot see.
    """
    problems: List[Mismatch] = []
    if set(first.bundles) != set(second.bundles):
        problems.append(
            Mismatch(
                level="contract",
                field="reemit/bundle_set",
                message=(
                    f"re-emission bundle set differs: run1={sorted(first.bundles)} "
                    f"run2={sorted(second.bundles)}"
                ),
                old=sorted(first.bundles),
                new=sorted(second.bundles),
            )
        )
    for label in sorted(set(first.bundles) & set(second.bundles)):
        for bfield, message, old, new in diff_bundles_detailed(
            first.bundles[label], second.bundles[label], label=label
        ):
            problems.append(
                Mismatch(
                    level="contract",
                    field=f"reemit/{label}/{bfield}",
                    message=f"re-emission is not deterministic: {message}",
                    old=old,
                    new=new,
                )
            )
    if not _close(first.total_time, second.total_time, rel_tol):
        problems.append(
            Mismatch(
                level="contract",
                field="reemit/total_time",
                message=(
                    f"re-run reported a different total time: run1={first.total_time} "
                    f"run2={second.total_time}"
                ),
                old=first.total_time,
                new=second.total_time,
            )
        )
    return problems


def compare_all(
    golden: Dict[str, Any],
    obs: RunObservation,
    *,
    rel_tol: float = 1e-9,
    ledger: Optional[Sequence[LedgerEntry]] = None,
) -> Tuple[List[Mismatch], List[str], List[LedgerEntry]]:
    """Run every tier and apply the T4 ledger.

    Returns ``(failures, recorded_exceptions, matched_ledger_entries)``.
    """
    entries = list(load_ledger() if ledger is None else ledger)
    status = compare_status(golden, obs)
    if status:
        return apply_ledger(obs.spec_id, status, entries)
    if golden.get("status", "ok") == "error":
        # Both runs fail: pinned as known-broken, nothing else is observable.
        return [], [], []
    mismatches: List[Mismatch] = []
    mismatches.extend(compare_structural(golden, obs))
    mismatches.extend(compare_timing(golden, obs, rel_tol=rel_tol))
    mismatches.extend(compare_contract(golden, obs))
    return apply_ledger(obs.spec_id, mismatches, entries)


def compare_observation(
    golden: Dict[str, Any],
    obs: RunObservation,
    *,
    rel_tol: float = 1e-9,
    ledger: Optional[Sequence[LedgerEntry]] = None,
) -> List[str]:
    """Return a list of mismatch messages between a golden record and a run.

    Ledger-absorbed differences are reported as ``RECORDED ...`` lines by
    :func:`compare_all` and are NOT returned here — they are exceptions, not
    failures.
    """
    failures, _recorded, _matched = compare_all(golden, obs, rel_tol=rel_tol, ledger=ledger)
    return [str(problem) for problem in failures]
