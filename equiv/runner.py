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
set of behaviors the harness pins:

- end-to-end reported time (parsed from the results file, full precision),
- canonical forms of every emitted Chakra ET bundle (see ``equiv.canonical``),
- exact AstraSim per-rank wall seconds for every bundle (from the per-bundle
  ``cache.json`` written next to the ETs),
- causal completability of every bundle (``equiv.dlsim``).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from validation_scripts.validation_helpers import ValidationSpec, run_validation_suite

from .canonical import canonicalize_bundle
from .configs import EquivSpec
from .dlsim import BundleSim

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_CONFIG = (
    REPO_ROOT / "validation_scripts" / "validation_configs" / "model-config" / "Llama2-7B.yaml"
)
BASE_HW_CONFIG = (
    REPO_ROOT / "validation_scripts" / "validation_configs" / "hardware-config" / "a100_80GB.yaml"
)

_TOTAL_TIME_RE = re.compile(r"^Total Time:\s*([0-9.eE+-]+)", re.MULTILINE)
_INFER_TIME_RE = re.compile(r"^Inference Time for batch:\s*([0-9.eE+-]+)", re.MULTILINE)
_PREFILL_RE = re.compile(r"^Prefill Time:\s*([0-9.eE+-]+)", re.MULTILINE)
_DECODE_RE = re.compile(r"^Decode Time:\s*([0-9.eE+-]+)", re.MULTILINE)


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
    """Read exact per-rank AstraSim seconds from the bundle's cache.json.

    Multi-run flows (grad accumulation's no-DP + final dispatchers, inference
    prefill + decode samples) share one artifact dir, and the legacy cleanup
    deletes ETs but not cache.json, so entries accumulate. That choreography
    is part of the pinned surface: record every entry, keyed and ordered by
    the cache's own signature hash, so all runs' wall seconds are gated.
    """
    cache_path = bundle_dir / "cache.json"
    if not cache_path.exists():
        return None
    try:
        with open(cache_path) as fh:
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
    }
    # validation_helpers decides tmp cleanup from the *parent* process env.
    prev_keep = os.environ.get("RAPID_VALIDATION_KEEP_TMP")
    os.environ["RAPID_VALIDATION_KEEP_TMP"] = "1"
    try:
        results = run_validation_suite(
            vspecs,
            base_model_config_path=str(BASE_MODEL_CONFIG),
            base_hardware_config_path=str(BASE_HW_CONFIG),
            result_parser=lambda _out, _spec: {},
            tmp_root=tmp_root,
            max_workers=max_workers,
            env_overrides=env_overrides,
            cache_mode="CACHE_READWRITE",
        )
    finally:
        if prev_keep is None:
            os.environ.pop("RAPID_VALIDATION_KEEP_TMP", None)
        else:
            os.environ["RAPID_VALIDATION_KEEP_TMP"] = prev_keep
    observations: Dict[str, RunObservation] = {}
    for spec, result in zip(specs, results):
        observations[spec.spec_id] = observe_run(spec.spec_id, result)
    return observations


def compare_observation(
    golden: Dict[str, Any], obs: RunObservation, *, rel_tol: float = 1e-9
) -> List[str]:
    """Return a list of mismatches between a golden record and a fresh run."""
    from .canonical import diff_bundles

    problems: List[str] = []
    golden_status = golden.get("status", "ok")
    obs_status = "ok" if obs.success else "error"
    if golden_status != obs_status:
        problems.append(
            f"status: golden={golden_status} candidate={obs_status} "
            f"(candidate tail: {obs.error_tail[-400:]})"
        )
        return problems
    if golden_status == "error":
        return problems  # both fail; pinned as known-broken

    def close(a: Optional[float], b: Optional[float]) -> bool:
        if a is None or b is None:
            return a is None and b is None
        if a == b:
            return True
        denom = max(abs(a), abs(b), 1e-30)
        return abs(a - b) / denom <= rel_tol

    if not close(golden.get("total_time"), obs.total_time):
        problems.append(
            f"total_time: golden={golden.get('total_time')} candidate={obs.total_time}"
        )
    for phase, value in (golden.get("phase_times") or {}).items():
        if not close(value, obs.phase_times.get(phase)):
            problems.append(
                f"phase {phase}: golden={value} candidate={obs.phase_times.get(phase)}"
            )

    golden_bundles = golden.get("bundles") or {}
    if set(golden_bundles) != set(obs.bundles):
        problems.append(
            f"bundle set: golden={sorted(golden_bundles)} candidate={sorted(obs.bundles)}"
        )
    for label in sorted(set(golden_bundles) & set(obs.bundles)):
        problems.extend(diff_bundles(golden_bundles[label], obs.bundles[label], label=label))

    def _diff_rank_seconds(label: str, gold_entry: Dict[str, Any], cand_entry: Dict[str, Any]) -> None:
        gold_ranks = gold_entry.get("per_rank_sec") or []
        cand_ranks = cand_entry.get("per_rank_sec") or []
        if len(gold_ranks) != len(cand_ranks):
            problems.append(
                f"astra[{label}]: rank count golden={len(gold_ranks)} candidate={len(cand_ranks)}"
            )
            return
        for idx, (a, b) in enumerate(zip(gold_ranks, cand_ranks)):
            if not close(a, b):
                problems.append(f"astra[{label}] rank {idx}: golden={a} candidate={b}")
                return

    golden_astra = golden.get("astra_times") or {}
    for label in sorted(set(golden_astra) & set(obs.astra_times)):
        gold_entry, cand_entry = golden_astra[label], obs.astra_times[label]
        gold_runs = gold_entry.get("runs")
        cand_runs = cand_entry.get("runs")
        if (gold_runs is None) != (cand_runs is None):
            problems.append(
                f"astra[{label}]: run multiplicity differs "
                f"(golden {'multi' if gold_runs else 'single'}, candidate {'multi' if cand_runs else 'single'})"
            )
            continue
        if gold_runs is None:
            _diff_rank_seconds(label, gold_entry, cand_entry)
            continue
        if len(gold_runs) != len(cand_runs):
            problems.append(
                f"astra[{label}]: run count golden={len(gold_runs)} candidate={len(cand_runs)}"
            )
            continue
        for run_idx, (grun, crun) in enumerate(zip(gold_runs, cand_runs)):
            if grun.get("sig") != crun.get("sig"):
                problems.append(
                    f"astra[{label}] run {run_idx}: signature differs"
                )
                break
            _diff_rank_seconds(f"{label}/run{run_idx}", grun, crun)
    for label, ok in (golden.get("dlsim_ok") or {}).items():
        if ok and not obs.dlsim_ok.get(label, False):
            problems.append(f"dlsim[{label}]: golden completed but candidate deadlocks")
    return problems
