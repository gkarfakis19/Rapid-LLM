#!/usr/bin/env python3
"""
Fit H100 hardware util parameters to MosaicML training validation data (num_gpus=8).

Primary target: minimize tail-focused objective on abs percent error.
Default objective:
  score = p90_abs + tail_lambda * max(0, max_abs - target_abs) + fail_lambda * fail_rate

Tunables:
  - sw_param.kernel_launch_overhead
  - tech_param.DRAM.util
  - network.dimensions[id=dim0].topology.util
Optional 4th tunable:
  - tech_param.core.util  (enabled via --fit-core-util)

Search:
  - Stage A: coarse global random search (kernel sampled in log-space)
  - Stage B: coordinate refinement from top Stage-A candidates
  - Stage C: local random polish
  - Full re-check + full-data refinement
  - Adaptive bound expansion if best candidate is boundary-constrained
  - Feasibility guardrail if expanded search stalls above target
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_BASE_HW = (
    PROJECT_ROOT
    / "validation_scripts"
    / "validation_configs"
    / "hardware-config"
    / "H100_SXM5_80GB copy.yaml"
)
DEFAULT_INPUT_CSV = PROJECT_ROOT / "validation_scripts" / "mosiacml_data" / "h100_80gb_bf16.csv"
DEFAULT_HW_OUTPUT_DIR = PROJECT_ROOT / "validation_scripts" / "validation_configs" / "hardware-config"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "output" / "validation" / "train" / "mosaic_fit"

# Hard parameter limits used for adaptive bound expansion clamping.
KERNEL_HARD_MIN = 1.0e-7
KERNEL_HARD_MAX = 2.0e-4
UTIL_HARD_MIN = 0.05
UTIL_HARD_MAX = 2.0


@dataclass(frozen=True)
class Params:
    kernel_launch_overhead: float
    dram_util: float
    dim0_util: float
    core_util: float


@dataclass(frozen=True)
class Bounds:
    kernel_min: float
    kernel_max: float
    dram_util_min: float
    dram_util_max: float
    dim0_util_min: float
    dim0_util_max: float
    core_util_min: float
    core_util_max: float


@dataclass(frozen=True)
class ObjectiveConfig:
    target_abs: float
    tail_lambda: float
    fail_lambda: float
    no_data_penalty: float


@dataclass
class Metrics:
    score: float
    mean_abs: float
    median_abs: float
    p90_abs: float
    p95_abs: float
    max_abs: float
    tail_excess: float
    fail_count: int
    fail_rate: float
    total_count: int
    valid_count: int


@dataclass
class CandidateResult:
    params: Params
    stage: str
    dataset: str
    metrics: Metrics
    output_csv: str


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} is not a mapping.")
    return data


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _extract_base_params(base_hw: Dict[str, Any]) -> Params:
    sw = base_hw.get("sw_param") or {}
    tech = base_hw.get("tech_param") or {}
    dram = tech.get("DRAM") or {}
    core = tech.get("core") or {}
    net = base_hw.get("network") or {}
    dims = net.get("dimensions") or []

    kernel = float(sw.get("kernel_launch_overhead", 9e-6))
    dram_util = float(dram.get("util", 1.0))
    core_util = float(core.get("util", 1.0))

    dim0_util = 0.7
    if isinstance(dims, list):
        for dim in dims:
            if not isinstance(dim, dict):
                continue
            if str(dim.get("id", "")).strip() == "dim0":
                topo = dim.get("topology") or {}
                dim0_util = float(topo.get("util", dim0_util))
                break

    return Params(
        kernel_launch_overhead=kernel,
        dram_util=dram_util,
        dim0_util=dim0_util,
        core_util=core_util,
    )


def _apply_params_to_hw(base_hw: Dict[str, Any], params: Params) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_hw)

    sw = cfg.setdefault("sw_param", {})
    if not isinstance(sw, dict):
        sw = {}
        cfg["sw_param"] = sw
    sw["kernel_launch_overhead"] = float(params.kernel_launch_overhead)

    tech = cfg.setdefault("tech_param", {})
    if not isinstance(tech, dict):
        tech = {}
        cfg["tech_param"] = tech

    core = tech.setdefault("core", {})
    if not isinstance(core, dict):
        core = {}
        tech["core"] = core
    core["util"] = float(params.core_util)

    dram = tech.setdefault("DRAM", {})
    if not isinstance(dram, dict):
        dram = {}
        tech["DRAM"] = dram
    dram["util"] = float(params.dram_util)

    net = cfg.setdefault("network", {})
    if not isinstance(net, dict):
        net = {}
        cfg["network"] = net
    dims = net.setdefault("dimensions", [])
    if not isinstance(dims, list):
        dims = []
        net["dimensions"] = dims

    found = False
    for dim in dims:
        if not isinstance(dim, dict):
            continue
        if str(dim.get("id", "")).strip() == "dim0":
            topo = dim.setdefault("topology", {})
            if not isinstance(topo, dict):
                topo = {}
                dim["topology"] = topo
            topo["util"] = float(params.dim0_util)
            found = True
            break

    if not found:
        dims.insert(
            0,
            {
                "id": "dim0",
                "topology": {"util": float(params.dim0_util)},
            },
        )

    return cfg


def _filter_num_gpus_csv(input_csv: Path, output_csv: Path, num_gpus: int) -> int:
    df = pd.read_csv(input_csv)
    if "num_gpus" not in df.columns:
        raise ValueError(f"Input CSV missing required column 'num_gpus': {input_csv}")
    filtered = df[pd.to_numeric(df["num_gpus"], errors="coerce") == int(num_gpus)].copy()
    if filtered.empty:
        raise ValueError(f"No rows with num_gpus={num_gpus} found in {input_csv}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_csv, index=False)
    return int(len(filtered))


def _build_proxy_subset(full_csv: Path, proxy_csv: Path, max_rows: int) -> int:
    """Deterministic subset with broad model/seq coverage."""
    df = pd.read_csv(full_csv)
    required = {"model_size", "seq_len", "inferred_total_latency_s"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {full_csv}: {sorted(missing)}")

    if len(df) <= max_rows:
        df.to_csv(proxy_csv, index=False)
        return int(len(df))

    picks: List[pd.DataFrame] = []
    grouped = df.groupby(df["model_size"].astype(str).str.strip().str.lower(), sort=True)

    for _, group in grouped:
        g = group.copy()
        g["seq_len_num"] = pd.to_numeric(g["seq_len"], errors="coerce")
        g = g.dropna(subset=["seq_len_num"])
        if g.empty:
            continue
        g = g.sort_values(["seq_len_num", "inferred_total_latency_s"], ascending=[True, True])
        picks.append(g.head(1))
        if len(g) > 1:
            picks.append(g.tail(1))

    chosen = pd.concat(picks, ignore_index=False).drop_duplicates() if picks else pd.DataFrame(columns=df.columns)
    remaining = df.drop(index=chosen.index, errors="ignore")

    if len(chosen) < max_rows and not remaining.empty:
        remaining = remaining.copy()
        remaining["seq_len_num"] = pd.to_numeric(remaining["seq_len"], errors="coerce")
        remaining = remaining.dropna(subset=["seq_len_num"]).sort_values(
            ["seq_len_num", "inferred_total_latency_s"], ascending=[True, True]
        )
        need = max_rows - len(chosen)
        if need > 0 and len(remaining) > 0:
            if need >= len(remaining):
                extra = remaining
            else:
                idx = np.linspace(0, len(remaining) - 1, num=need, dtype=int)
                extra = remaining.iloc[idx]
            chosen = pd.concat([chosen, extra], ignore_index=False).drop_duplicates()

    chosen = chosen.head(max_rows).copy()
    if "seq_len_num" in chosen.columns:
        chosen = chosen.drop(columns=["seq_len_num"])
    if chosen.empty:
        raise ValueError("Proxy subset selection produced zero rows.")

    chosen.to_csv(proxy_csv, index=False)
    return int(len(chosen))


def _compute_metrics(rows: Sequence[Dict[str, Any]], objective: ObjectiveConfig) -> Metrics:
    errs: List[float] = []
    fail_count = 0
    total_count = 0

    for row in rows:
        total_count += 1
        success = bool(row.get("success", False))
        raw_err = row.get("abs_pct_error")
        try:
            err = float(raw_err)
        except (TypeError, ValueError):
            err = float("nan")

        if (not success) or math.isnan(err) or (not math.isfinite(err)):
            fail_count += 1
            continue
        errs.append(err)

    valid_count = len(errs)
    fail_rate = float(fail_count) / float(total_count) if total_count > 0 else 1.0

    if valid_count == 0:
        return Metrics(
            score=float(objective.no_data_penalty),
            mean_abs=float("nan"),
            median_abs=float("nan"),
            p90_abs=float("nan"),
            p95_abs=float("nan"),
            max_abs=float("nan"),
            tail_excess=float("nan"),
            fail_count=int(fail_count),
            fail_rate=float(fail_rate),
            total_count=int(total_count),
            valid_count=0,
        )

    arr = np.array(errs, dtype=float)
    mean_abs = float(np.mean(arr))
    median_abs = float(np.median(arr))
    p90_abs = float(np.percentile(arr, 90))
    p95_abs = float(np.percentile(arr, 95))
    max_abs = float(np.max(arr))
    tail_excess = max(0.0, max_abs - float(objective.target_abs))

    score = (
        p90_abs
        + float(objective.tail_lambda) * tail_excess
        + float(objective.fail_lambda) * fail_rate
    )

    return Metrics(
        score=float(score),
        mean_abs=mean_abs,
        median_abs=median_abs,
        p90_abs=p90_abs,
        p95_abs=p95_abs,
        max_abs=max_abs,
        tail_excess=float(tail_excess),
        fail_count=int(fail_count),
        fail_rate=float(fail_rate),
        total_count=int(total_count),
        valid_count=int(valid_count),
    )


class Evaluator:
    def __init__(
        self,
        *,
        base_hw: Dict[str, Any],
        full_csv: Path,
        proxy_csv: Path,
        run_artifact_dir: Path,
        validation_workers: int,
        quiet_validation: bool,
        keep_run_csv: bool,
        bounds: Bounds,
        objective: ObjectiveConfig,
        fit_core_util: bool,
        fixed_core_util: float,
        use_flashattention: bool,
        attention_tile_size: int | None,
    ) -> None:
        self.base_hw = copy.deepcopy(base_hw)
        self.full_csv = full_csv
        self.proxy_csv = proxy_csv
        self.run_artifact_dir = run_artifact_dir
        self.validation_workers = max(1, int(validation_workers))
        self.quiet_validation = bool(quiet_validation)
        self.keep_run_csv = bool(keep_run_csv)
        self.bounds = bounds
        self.objective = objective
        self.fit_core_util = bool(fit_core_util)
        self.fixed_core_util = float(fixed_core_util)
        self.use_flashattention = bool(use_flashattention)
        self.attention_tile_size = (
            int(attention_tile_size) if attention_tile_size is not None else None
        )

        self._cache: Dict[Tuple[str, Tuple[float, float, float, float]], CandidateResult] = {}
        self.run_artifact_dir.mkdir(parents=True, exist_ok=True)

    def set_bounds(self, bounds: Bounds) -> None:
        self.bounds = bounds

    def clamp_params(self, params: Params) -> Params:
        b = self.bounds
        core_val = float(params.core_util)
        if not self.fit_core_util:
            core_val = self.fixed_core_util
        return Params(
            kernel_launch_overhead=_clamp(float(params.kernel_launch_overhead), float(b.kernel_min), float(b.kernel_max)),
            dram_util=_clamp(float(params.dram_util), float(b.dram_util_min), float(b.dram_util_max)),
            dim0_util=_clamp(float(params.dim0_util), float(b.dim0_util_min), float(b.dim0_util_max)),
            core_util=_clamp(float(core_val), float(b.core_util_min), float(b.core_util_max)),
        )

    def params_key(self, params: Params, digits: int = 12) -> Tuple[float, float, float, float]:
        p = self.clamp_params(params)
        return (
            round(float(p.kernel_launch_overhead), digits),
            round(float(p.dram_util), digits),
            round(float(p.dim0_util), digits),
            round(float(p.core_util), digits),
        )

    def evaluate(self, params: Params, *, dataset: str, stage: str) -> CandidateResult:
        params = self.clamp_params(params)
        dkey = str(dataset).strip().lower()
        if dkey not in {"proxy", "full"}:
            raise ValueError(f"Unknown dataset key: {dataset}")

        key = (dkey, self.params_key(params))
        cached = self._cache.get(key)
        if cached is not None:
            return CandidateResult(
                params=cached.params,
                stage=stage,
                dataset=dkey,
                metrics=cached.metrics,
                output_csv=cached.output_csv,
            )

        csv_path = self.proxy_csv if dkey == "proxy" else self.full_csv
        run_id = uuid.uuid4().hex[:10]
        cfg_path = self.run_artifact_dir / f"hw_{dkey}_{run_id}.yaml"
        out_csv_path = self.run_artifact_dir / f"rows_{dkey}_{run_id}.csv"

        hw_cfg = _apply_params_to_hw(self.base_hw, params)
        _write_yaml(cfg_path, hw_cfg)

        env_prev_workers = os.environ.get("RAPID_VALIDATION_WORKERS")
        env_prev_quiet = os.environ.get("RAPID_VALIDATION_QUIET")

        os.environ["RAPID_VALIDATION_WORKERS"] = str(self.validation_workers)
        if self.quiet_validation:
            os.environ["RAPID_VALIDATION_QUIET"] = "1"

        try:
            from validation_scripts import mosiacml_train

            rows = mosiacml_train.run(
                input_csv=str(csv_path),
                hardware_config=str(cfg_path),
                output_csv=str(out_csv_path),
                enable_plot=False,
                show_progress=False,
                emit_logs=False,
                use_flashattention=self.use_flashattention,
                attention_tile_size=self.attention_tile_size,
            )
        finally:
            if env_prev_workers is None:
                os.environ.pop("RAPID_VALIDATION_WORKERS", None)
            else:
                os.environ["RAPID_VALIDATION_WORKERS"] = env_prev_workers
            if env_prev_quiet is None:
                os.environ.pop("RAPID_VALIDATION_QUIET", None)
            else:
                os.environ["RAPID_VALIDATION_QUIET"] = env_prev_quiet

        metrics = _compute_metrics(rows, self.objective)
        result = CandidateResult(
            params=params,
            stage=stage,
            dataset=dkey,
            metrics=metrics,
            output_csv=str(out_csv_path),
        )
        self._cache[key] = result

        if not self.keep_run_csv:
            try:
                cfg_path.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                out_csv_path.unlink(missing_ok=True)
            except Exception:
                pass

        return result


def _rank_key(result: CandidateResult) -> Tuple[float, float, float, float, float, float, float, float]:
    m = result.metrics
    mean_abs = m.mean_abs if math.isfinite(m.mean_abs) else float("inf")
    max_abs = m.max_abs if math.isfinite(m.max_abs) else float("inf")
    return (
        float(m.score),
        float(mean_abs),
        float(max_abs),
        float(m.fail_rate),
        float(result.params.kernel_launch_overhead),
        float(result.params.dram_util),
        float(result.params.dim0_util),
        float(result.params.core_util),
    )


def _sort_results(results: Iterable[CandidateResult]) -> List[CandidateResult]:
    return sorted(results, key=_rank_key)


def _dedup_params(params_list: Iterable[Params], evaluator: Evaluator) -> List[Params]:
    out: List[Params] = []
    seen = set()
    for params in params_list:
        key = evaluator.params_key(params)
        if key in seen:
            continue
        seen.add(key)
        out.append(evaluator.clamp_params(params))
    return out


def _format_params(params: Params) -> str:
    return (
        f"kernel={params.kernel_launch_overhead:.3e}, "
        f"dram_util={params.dram_util:.5f}, "
        f"dim0_util={params.dim0_util:.5f}, "
        f"core_util={params.core_util:.5f}"
    )


def _candidate_stats_line(res: CandidateResult) -> str:
    m = res.metrics
    return (
        f"score={m.score:.3f}, p90={m.p90_abs:.2f}%, max={m.max_abs:.2f}%, "
        f"tail_excess={m.tail_excess:.2f}, mean={m.mean_abs:.2f}%, fail_rate={m.fail_rate:.3f}, "
        f"{_format_params(res.params)}"
    )


def _sample_params_random(rng: np.random.Generator, evaluator: Evaluator) -> Params:
    b = evaluator.bounds

    # Kernel sampled in log-space for better multiplicative coverage.
    log_k_min = math.log(float(b.kernel_min))
    log_k_max = math.log(float(b.kernel_max))
    kernel = float(math.exp(rng.uniform(log_k_min, log_k_max)))

    dram_util = float(rng.uniform(float(b.dram_util_min), float(b.dram_util_max)))
    dim0_util = float(rng.uniform(float(b.dim0_util_min), float(b.dim0_util_max)))

    if evaluator.fit_core_util:
        core_util = float(rng.uniform(float(b.core_util_min), float(b.core_util_max)))
    else:
        core_util = float(evaluator.fixed_core_util)

    return evaluator.clamp_params(
        Params(
            kernel_launch_overhead=kernel,
            dram_util=dram_util,
            dim0_util=dim0_util,
            core_util=core_util,
        )
    )


def _coordinate_refine(
    evaluator: Evaluator,
    start: Params,
    *,
    stage_label: str,
    dataset: str,
    step_fracs: Sequence[float],
) -> CandidateResult:
    current = evaluator.evaluate(start, dataset=dataset, stage=stage_label)
    b = evaluator.bounds

    ranges = {
        "kernel": float(b.kernel_max - b.kernel_min),
        "dram": float(b.dram_util_max - b.dram_util_min),
        "dim0": float(b.dim0_util_max - b.dim0_util_min),
        "core": float(b.core_util_max - b.core_util_min),
    }

    axes = ["kernel", "dram", "dim0"]
    if evaluator.fit_core_util:
        axes.append("core")

    for frac in step_fracs:
        frac = float(frac)
        if frac <= 0:
            continue

        for axis in axes:
            delta = frac * ranges[axis]
            base = current.params
            probes = [base]

            if axis == "kernel":
                probes.append(Params(base.kernel_launch_overhead + delta, base.dram_util, base.dim0_util, base.core_util))
                probes.append(Params(base.kernel_launch_overhead - delta, base.dram_util, base.dim0_util, base.core_util))
            elif axis == "dram":
                probes.append(Params(base.kernel_launch_overhead, base.dram_util + delta, base.dim0_util, base.core_util))
                probes.append(Params(base.kernel_launch_overhead, base.dram_util - delta, base.dim0_util, base.core_util))
            elif axis == "dim0":
                probes.append(Params(base.kernel_launch_overhead, base.dram_util, base.dim0_util + delta, base.core_util))
                probes.append(Params(base.kernel_launch_overhead, base.dram_util, base.dim0_util - delta, base.core_util))
            else:
                probes.append(Params(base.kernel_launch_overhead, base.dram_util, base.dim0_util, base.core_util + delta))
                probes.append(Params(base.kernel_launch_overhead, base.dram_util, base.dim0_util, base.core_util - delta))

            probe_results = [
                evaluator.evaluate(p, dataset=dataset, stage=stage_label)
                for p in _dedup_params(probes, evaluator)
            ]
            current = _sort_results(probe_results)[0]

    return current


def _random_polish(
    evaluator: Evaluator,
    start: Params,
    *,
    stage_label: str,
    dataset: str,
    rng: np.random.Generator,
    trials: int,
    noise_frac: float,
) -> CandidateResult:
    current = evaluator.evaluate(start, dataset=dataset, stage=stage_label)
    b = evaluator.bounds

    # Kernel perturbation in log-space.
    log_span = max(1e-12, math.log(float(b.kernel_max)) - math.log(float(b.kernel_min)))
    log_sigma = float(noise_frac) * log_span

    dram_sigma = float(noise_frac) * float(b.dram_util_max - b.dram_util_min)
    dim0_sigma = float(noise_frac) * float(b.dim0_util_max - b.dim0_util_min)
    core_sigma = float(noise_frac) * float(b.core_util_max - b.core_util_min)

    for _ in range(max(0, int(trials))):
        k = float(math.exp(math.log(current.params.kernel_launch_overhead) + rng.normal(0.0, log_sigma)))
        d = float(current.params.dram_util + rng.normal(0.0, dram_sigma))
        n = float(current.params.dim0_util + rng.normal(0.0, dim0_sigma))
        c = float(current.params.core_util + rng.normal(0.0, core_sigma)) if evaluator.fit_core_util else float(evaluator.fixed_core_util)

        cand = evaluator.clamp_params(
            Params(
                kernel_launch_overhead=k,
                dram_util=d,
                dim0_util=n,
                core_util=c,
            )
        )
        res = evaluator.evaluate(cand, dataset=dataset, stage=stage_label)
        if _rank_key(res) < _rank_key(current):
            current = res

    return current


def _touch_bounds(params: Params, evaluator: Evaluator, touch_frac: float) -> Dict[str, Tuple[bool, bool]]:
    p = evaluator.clamp_params(params)
    b = evaluator.bounds

    def near(value: float, low: float, high: float) -> Tuple[bool, bool]:
        tol = max(1e-12, float(touch_frac) * max(1e-12, high - low))
        return (value <= low + tol, value >= high - tol)

    touches = {
        "kernel": near(float(p.kernel_launch_overhead), float(b.kernel_min), float(b.kernel_max)),
        "dram": near(float(p.dram_util), float(b.dram_util_min), float(b.dram_util_max)),
        "dim0": near(float(p.dim0_util), float(b.dim0_util_min), float(b.dim0_util_max)),
        "core": near(float(p.core_util), float(b.core_util_min), float(b.core_util_max)),
    }
    if not evaluator.fit_core_util:
        touches["core"] = (False, False)
    return touches


def _expand_bounds(
    bounds: Bounds,
    touches: Dict[str, Tuple[bool, bool]],
    expand_factor: float,
    fit_core_util: bool,
) -> Tuple[Bounds, Dict[str, Dict[str, float]], bool]:
    b = bounds
    details: Dict[str, Dict[str, float]] = {}
    changed = False

    def expand_pair(
        name: str,
        low: float,
        high: float,
        hard_low: float,
        hard_high: float,
    ) -> Tuple[float, float]:
        nonlocal changed
        hit_min, hit_max = touches.get(name, (False, False))
        span = max(1e-12, high - low)
        new_low = low
        new_high = high
        if hit_min:
            new_low = max(hard_low, low - float(expand_factor) * span)
        if hit_max:
            new_high = min(hard_high, high + float(expand_factor) * span)
        if (new_low != low) or (new_high != high):
            changed = True
        details[name] = {
            "old_min": low,
            "old_max": high,
            "new_min": new_low,
            "new_max": new_high,
            "hit_min": float(hit_min),
            "hit_max": float(hit_max),
        }
        return new_low, new_high

    kmin, kmax = expand_pair("kernel", b.kernel_min, b.kernel_max, KERNEL_HARD_MIN, KERNEL_HARD_MAX)
    dmin, dmax = expand_pair("dram", b.dram_util_min, b.dram_util_max, UTIL_HARD_MIN, UTIL_HARD_MAX)
    nmin, nmax = expand_pair("dim0", b.dim0_util_min, b.dim0_util_max, UTIL_HARD_MIN, UTIL_HARD_MAX)

    if fit_core_util:
        cmin, cmax = expand_pair("core", b.core_util_min, b.core_util_max, UTIL_HARD_MIN, UTIL_HARD_MAX)
    else:
        cmin, cmax = b.core_util_min, b.core_util_max
        details["core"] = {
            "old_min": cmin,
            "old_max": cmax,
            "new_min": cmin,
            "new_max": cmax,
            "hit_min": 0.0,
            "hit_max": 0.0,
        }

    new_bounds = Bounds(
        kernel_min=float(kmin),
        kernel_max=float(kmax),
        dram_util_min=float(dmin),
        dram_util_max=float(dmax),
        dim0_util_min=float(nmin),
        dim0_util_max=float(nmax),
        core_util_min=float(cmin),
        core_util_max=float(cmax),
    )
    return new_bounds, details, changed


def _write_leaderboard(path: Path, rows: Sequence[CandidateResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "dataset",
                "stage",
                "score",
                "mean_abs",
                "median_abs",
                "p90_abs",
                "p95_abs",
                "max_abs",
                "tail_excess",
                "fail_count",
                "fail_rate",
                "total_count",
                "valid_count",
                "kernel_launch_overhead",
                "dram_util",
                "dim0_util",
                "core_util",
                "output_csv",
            ],
        )
        writer.writeheader()
        for idx, res in enumerate(rows, start=1):
            m = res.metrics
            writer.writerow(
                {
                    "rank": idx,
                    "dataset": res.dataset,
                    "stage": res.stage,
                    "score": f"{m.score:.10f}",
                    "mean_abs": f"{m.mean_abs:.10f}",
                    "median_abs": f"{m.median_abs:.10f}",
                    "p90_abs": f"{m.p90_abs:.10f}",
                    "p95_abs": f"{m.p95_abs:.10f}",
                    "max_abs": f"{m.max_abs:.10f}",
                    "tail_excess": f"{m.tail_excess:.10f}",
                    "fail_count": m.fail_count,
                    "fail_rate": f"{m.fail_rate:.10f}",
                    "total_count": m.total_count,
                    "valid_count": m.valid_count,
                    "kernel_launch_overhead": f"{res.params.kernel_launch_overhead:.12e}",
                    "dram_util": f"{res.params.dram_util:.12f}",
                    "dim0_util": f"{res.params.dim0_util:.12f}",
                    "core_util": f"{res.params.core_util:.12f}",
                    "output_csv": res.output_csv,
                }
            )


def _parse_float_list(text: str) -> List[float]:
    vals: List[float] = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    if not vals:
        raise ValueError(f"Expected comma-separated float list, got: {text!r}")
    return vals


def _emit_fitted_configs(
    *,
    base_hw: Dict[str, Any],
    ranked_full: Sequence[CandidateResult],
    output_dir: Path,
) -> List[Path]:
    top = list(ranked_full[:4])
    if not top:
        return []

    names = [
        "H100_SXM5_80GB.mosaic_train.numgpus8.best.yaml",
        "H100_SXM5_80GB.mosaic_train.numgpus8.rank2.yaml",
        "H100_SXM5_80GB.mosaic_train.numgpus8.rank3.yaml",
        "H100_SXM5_80GB.mosaic_train.numgpus8.rank4.yaml",
    ]

    out: List[Path] = []
    for idx, res in enumerate(top):
        path = output_dir / names[idx]
        cfg = _apply_params_to_hw(base_hw, res.params)
        _write_yaml(path, cfg)
        out.append(path)
    return out


def _build_bounds(args: argparse.Namespace) -> Bounds:
    return Bounds(
        kernel_min=float(args.kernel_min),
        kernel_max=float(args.kernel_max),
        dram_util_min=float(args.dram_util_min),
        dram_util_max=float(args.dram_util_max),
        dim0_util_min=float(args.dim0_util_min),
        dim0_util_max=float(args.dim0_util_max),
        core_util_min=float(args.core_util_min),
        core_util_max=float(args.core_util_max),
    )


def _run_search_round(
    *,
    evaluator: Evaluator,
    args: argparse.Namespace,
    rng: np.random.Generator,
    round_idx: int,
    seed_params: Sequence[Params],
) -> Tuple[List[CandidateResult], List[CandidateResult]]:
    prefix = f"r{round_idx + 1}"

    stage_a_samples = int(args.stage_a_samples)
    full_recheck_topk = int(args.full_recheck_topk)
    if round_idx > 0:
        stage_a_samples = max(16, stage_a_samples // 2)
        full_recheck_topk = max(12, full_recheck_topk // 2)

    # Stage A
    stage_a_params: List[Params] = list(seed_params)
    for _ in range(max(0, stage_a_samples)):
        stage_a_params.append(_sample_params_random(rng, evaluator))
    stage_a_params = _dedup_params(stage_a_params, evaluator)

    print(f"[{prefix}] Stage A: {len(stage_a_params)} proxy candidates")
    stage_a_results = [
        evaluator.evaluate(p, dataset="proxy", stage=f"{prefix}_stage_a") for p in stage_a_params
    ]
    stage_a_ranked = _sort_results(stage_a_results)
    print(f"[{prefix}] Stage A best: {_candidate_stats_line(stage_a_ranked[0])}")

    # Stage B
    starts_b = [res.params for res in stage_a_ranked[: max(1, int(args.stage_b_topk))]]
    print(f"[{prefix}] Stage B: refine {len(starts_b)} starts, step_fracs={args.stage_b_step_fracs}")
    stage_b_results: List[CandidateResult] = []
    for idx, start in enumerate(starts_b, start=1):
        stage_b_results.append(
            _coordinate_refine(
                evaluator,
                start,
                stage_label=f"{prefix}_stage_b_start{idx}",
                dataset="proxy",
                step_fracs=args.stage_b_step_fracs,
            )
        )
    stage_b_ranked = _sort_results(stage_b_results)
    print(f"[{prefix}] Stage B best: {_candidate_stats_line(stage_b_ranked[0])}")

    # Stage C
    starts_c = [res.params for res in stage_b_ranked[: max(1, int(args.stage_c_starts))]]
    print(
        f"[{prefix}] Stage C: polish {len(starts_c)} starts, "
        f"trials/start={int(args.stage_c_trials)}, noise_frac={float(args.stage_c_noise_frac):.4f}"
    )
    stage_c_results: List[CandidateResult] = []
    for idx, start in enumerate(starts_c, start=1):
        stage_c_results.append(
            _random_polish(
                evaluator,
                start,
                stage_label=f"{prefix}_stage_c_start{idx}",
                dataset="proxy",
                rng=rng,
                trials=int(args.stage_c_trials),
                noise_frac=float(args.stage_c_noise_frac),
            )
        )
    stage_c_ranked = _sort_results(stage_c_results)
    print(f"[{prefix}] Stage C best: {_candidate_stats_line(stage_c_ranked[0])}")

    proxy_all = _sort_results(stage_a_results + stage_b_results + stage_c_results)
    proxy_params = _dedup_params([r.params for r in proxy_all], evaluator)

    # Full re-check
    finalists = proxy_params[: max(1, full_recheck_topk)]
    print(f"[{prefix}] Full re-check: {len(finalists)} candidates")
    full_results = [
        evaluator.evaluate(p, dataset="full", stage=f"{prefix}_full_recheck") for p in finalists
    ]

    # Full refinement
    full_ranked = _sort_results(full_results)
    starts_full_ref = [res.params for res in full_ranked[: max(0, int(args.full_refine_topk))]]
    if starts_full_ref and args.full_refine_step_fracs:
        print(
            f"[{prefix}] Full refine: {len(starts_full_ref)} starts, "
            f"step_fracs={args.full_refine_step_fracs}"
        )
        for idx, start in enumerate(starts_full_ref, start=1):
            full_results.append(
                _coordinate_refine(
                    evaluator,
                    start,
                    stage_label=f"{prefix}_full_refine_start{idx}",
                    dataset="full",
                    step_fracs=args.full_refine_step_fracs,
                )
            )

    # Unique + sorted outputs.
    proxy_unique: List[CandidateResult] = []
    seen_proxy = set()
    for res in _sort_results(proxy_all):
        key = evaluator.params_key(res.params)
        if key in seen_proxy:
            continue
        seen_proxy.add(key)
        proxy_unique.append(res)

    full_unique: List[CandidateResult] = []
    seen_full = set()
    for res in _sort_results(full_results):
        key = evaluator.params_key(res.params)
        if key in seen_full:
            continue
        seen_full.add(key)
        full_unique.append(res)

    print(f"[{prefix}] Round best full: {_candidate_stats_line(full_unique[0])}")
    return proxy_unique, full_unique


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fit H100 util parameters to Mosaic training benchmark (num_gpus=8)."
    )

    p.add_argument("--hardware-config", default=str(DEFAULT_BASE_HW))
    p.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    p.add_argument("--num-gpus", type=int, default=8)
    p.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    p.add_argument("--output-dir", default=str(DEFAULT_HW_OUTPUT_DIR))
    p.add_argument("--seed", type=int, default=20260320)

    # Objective.
    p.add_argument("--target-abs-error", type=float, default=15.0)
    p.add_argument("--tail-lambda", type=float, default=2.0)
    p.add_argument("--fail-lambda", type=float, default=200.0)
    p.add_argument("--no-data-penalty", type=float, default=1e6)

    # Bounds.
    p.add_argument("--kernel-min", type=float, default=5.0e-6)
    p.add_argument("--kernel-max", type=float, default=3.0e-5)
    p.add_argument("--dram-util-min", type=float, default=0.45)
    p.add_argument("--dram-util-max", type=float, default=1.00)
    p.add_argument("--dim0-util-min", type=float, default=0.45)
    p.add_argument("--dim0-util-max", type=float, default=1.10)
    p.add_argument("--core-util-min", type=float, default=0.45)
    p.add_argument("--core-util-max", type=float, default=1.00)

    # Optional 4th parameter.
    p.add_argument("--fit-core-util", action="store_true", help="Enable tuning tech_param.core.util")

    # Search budget (increased defaults).
    p.add_argument("--proxy-max-rows", type=int, default=14)
    p.add_argument("--stage-a-samples", type=int, default=120)
    p.add_argument("--stage-b-topk", type=int, default=8)
    p.add_argument("--stage-b-step-fracs", type=_parse_float_list, default=[0.15, 0.06])
    p.add_argument("--stage-c-starts", type=int, default=4)
    p.add_argument("--stage-c-trials", type=int, default=10)
    p.add_argument("--stage-c-noise-frac", type=float, default=0.03)
    p.add_argument("--full-recheck-topk", type=int, default=40)
    p.add_argument("--full-refine-topk", type=int, default=3)
    p.add_argument("--full-refine-step-fracs", type=_parse_float_list, default=[0.03])

    # Adaptive bound expansion.
    p.add_argument("--max-bound-expansions", type=int, default=2)
    p.add_argument("--expand-factor", type=float, default=0.35)
    p.add_argument("--bound-touch-frac", type=float, default=0.015)

    # Feasibility guardrail.
    p.add_argument("--guardrail-stall-delta", type=float, default=0.5)

    p.add_argument("--validation-workers", type=int, default=4)
    p.add_argument("--quiet-validation", action="store_true")
    p.add_argument("--keep-run-csv", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    p.add_argument(
        "--disable-flashattention",
        action="store_true",
        help="Disable flash attention in Mosaic validation during fitting.",
    )
    p.add_argument(
        "--attention-tile-size",
        type=int,
        default=None,
        help=(
            "Attention tile size override for flash attention during fitting. "
            "Default: keep model-config tile size."
        ),
    )

    return p


def run_fit(args: argparse.Namespace) -> int:
    seed = int(args.seed)
    rng = np.random.default_rng(seed)

    hw_path = Path(args.hardware_config)
    input_csv = Path(args.input_csv)
    if not hw_path.exists():
        raise FileNotFoundError(f"Hardware config not found: {hw_path}")
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    base_hw = _load_yaml(hw_path)
    base_params_raw = _extract_base_params(base_hw)

    artifact_dir = Path(args.artifact_dir)
    output_dir = Path(args.output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_tag = _now_tag()
    run_dir = artifact_dir / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    filtered_csv = run_dir / f"h100_80gb_bf16_numgpus{int(args.num_gpus)}.csv"
    full_rows = _filter_num_gpus_csv(input_csv, filtered_csv, int(args.num_gpus))

    proxy_csv = run_dir / f"h100_80gb_bf16_numgpus{int(args.num_gpus)}_proxy.csv"
    proxy_rows = _build_proxy_subset(filtered_csv, proxy_csv, int(args.proxy_max_rows))

    objective = ObjectiveConfig(
        target_abs=float(args.target_abs_error),
        tail_lambda=float(args.tail_lambda),
        fail_lambda=float(args.fail_lambda),
        no_data_penalty=float(args.no_data_penalty),
    )

    bounds = _build_bounds(args)
    evaluator = Evaluator(
        base_hw=base_hw,
        full_csv=filtered_csv,
        proxy_csv=proxy_csv,
        run_artifact_dir=run_dir / "candidate_runs",
        validation_workers=int(args.validation_workers),
        quiet_validation=bool(args.quiet_validation),
        keep_run_csv=bool(args.keep_run_csv),
        bounds=bounds,
        objective=objective,
        fit_core_util=bool(args.fit_core_util),
        fixed_core_util=float(base_params_raw.core_util),
        use_flashattention=(not bool(args.disable_flashattention)),
        attention_tile_size=(None if args.attention_tile_size is None else int(args.attention_tile_size)),
    )

    base_params = evaluator.clamp_params(base_params_raw)

    print("=== Mosaic train fitter ===")
    print(f"seed: {seed}")
    print(f"base hw: {hw_path}")
    print(f"input csv: {input_csv}")
    print(f"filtered rows (num_gpus={int(args.num_gpus)}): {full_rows}")
    print(f"proxy rows: {proxy_rows}")
    print(
        f"objective: score = p90 + {objective.tail_lambda} * max(0, max - {objective.target_abs}) "
        f"+ {objective.fail_lambda} * fail_rate"
    )
    print(f"fit_core_util: {bool(args.fit_core_util)}")
    print(f"flashattention: {not bool(args.disable_flashattention)}")
    print(f"attention_tile_size: {args.attention_tile_size}")
    print(f"base params: {_format_params(base_params)}")
    print(f"initial bounds: {asdict(bounds)}")

    if args.dry_run:
        print("dry-run enabled; exiting")
        return 0

    baseline_proxy = evaluator.evaluate(base_params, dataset="proxy", stage="baseline")
    baseline_full = evaluator.evaluate(base_params, dataset="full", stage="baseline")
    print("baseline proxy:", _candidate_stats_line(baseline_proxy))
    print("baseline full :", _candidate_stats_line(baseline_full))

    all_proxy: List[CandidateResult] = [baseline_proxy]
    all_full: List[CandidateResult] = [baseline_full]

    best_global = baseline_full
    seed_pool: List[Params] = [base_params]

    bound_history: List[Dict[str, Any]] = [asdict(bounds)]
    expansion_events: List[Dict[str, Any]] = []
    round_history: List[Dict[str, Any]] = []

    feasibility_guardrail_triggered = False
    feasibility_guardrail_reason = ""

    prev_round_best_p90 = baseline_full.metrics.p90_abs

    rounds_total = max(0, int(args.max_bound_expansions)) + 1
    for round_idx in range(rounds_total):
        print(f"\n=== Search round {round_idx + 1}/{rounds_total} ===")
        evaluator.set_bounds(bounds)

        proxy_round, full_round = _run_search_round(
            evaluator=evaluator,
            args=args,
            rng=rng,
            round_idx=round_idx,
            seed_params=seed_pool,
        )

        all_proxy.extend(proxy_round)
        all_full.extend(full_round)

        round_best = full_round[0]
        if _rank_key(round_best) < _rank_key(best_global):
            best_global = round_best

        round_history.append(
            {
                "round": round_idx + 1,
                "best_full": {
                    "params": asdict(round_best.params),
                    "metrics": asdict(round_best.metrics),
                    "stage": round_best.stage,
                },
                "bounds": asdict(bounds),
            }
        )

        # Feasibility guardrail: if expanded search does not improve meaningfully above target.
        if round_idx > 0 and math.isfinite(round_best.metrics.p90_abs):
            improve = prev_round_best_p90 - round_best.metrics.p90_abs
            if (improve < float(args.guardrail_stall_delta)) and (round_best.metrics.p90_abs > objective.target_abs):
                feasibility_guardrail_triggered = True
                feasibility_guardrail_reason = (
                    f"Expanded-bound search stalled (delta_p90={improve:.3f} < {float(args.guardrail_stall_delta):.3f}) "
                    f"while p90={round_best.metrics.p90_abs:.3f}% > target={objective.target_abs:.3f}%"
                )
                print("[guardrail]", feasibility_guardrail_reason)
                break
        prev_round_best_p90 = min(prev_round_best_p90, round_best.metrics.p90_abs)

        if round_idx >= int(args.max_bound_expansions):
            break

        touches = _touch_bounds(round_best.params, evaluator, float(args.bound_touch_frac))
        any_touch = any(lo or hi for lo, hi in touches.values())
        if not any_touch:
            print("No boundary touch on round best; stopping bound expansion.")
            break

        new_bounds, details, changed = _expand_bounds(
            bounds,
            touches,
            float(args.expand_factor),
            bool(args.fit_core_util),
        )
        expansion_events.append(
            {
                "round": round_idx + 1,
                "touches": {k: {"min": bool(v[0]), "max": bool(v[1])} for k, v in touches.items()},
                "details": details,
                "changed": bool(changed),
            }
        )

        if not changed:
            print("Boundary touched but bounds could not expand further (hard limits reached).")
            break

        bounds = new_bounds
        bound_history.append(asdict(bounds))
        print(f"Expanded bounds for next round: {asdict(bounds)}")

        # Carry strong seeds forward.
        seed_pool = [r.params for r in full_round[: max(4, int(args.full_refine_topk) + 1)]]

    # Unique sorted leaderboards across all rounds.
    proxy_unique: List[CandidateResult] = []
    seen_proxy = set()
    for res in _sort_results(all_proxy):
        key = evaluator.params_key(res.params)
        if key in seen_proxy:
            continue
        seen_proxy.add(key)
        proxy_unique.append(res)

    full_unique: List[CandidateResult] = []
    seen_full = set()
    for res in _sort_results(all_full):
        key = evaluator.params_key(res.params)
        if key in seen_full:
            continue
        seen_full.add(key)
        full_unique.append(res)

    best_full = full_unique[0]
    print("\nBest full:", _candidate_stats_line(best_full))
    target_pass = bool(math.isfinite(best_full.metrics.p90_abs) and best_full.metrics.p90_abs <= objective.target_abs)
    print(
        "Target check: p90 <= {:.2f}% -> {}".format(
            objective.target_abs,
            "PASS" if target_pass else "FAIL",
        )
    )

    emitted_paths = _emit_fitted_configs(base_hw=base_hw, ranked_full=full_unique, output_dir=output_dir)

    full_leaderboard = run_dir / "leaderboard_full.csv"
    proxy_leaderboard = run_dir / "leaderboard_proxy.csv"
    _write_leaderboard(full_leaderboard, full_unique)
    _write_leaderboard(proxy_leaderboard, proxy_unique)

    summary = {
        "created_at": _now_tag(),
        "seed": seed,
        "fit_core_util": bool(args.fit_core_util),
        "target_pass": target_pass,
        "objective": asdict(objective),
        "input": {
            "base_hardware_config": str(hw_path.resolve()),
            "input_csv": str(input_csv.resolve()),
            "filtered_csv": str(filtered_csv.resolve()),
            "proxy_csv": str(proxy_csv.resolve()),
            "num_gpus": int(args.num_gpus),
            "full_rows": int(full_rows),
            "proxy_rows": int(proxy_rows),
        },
        "search_args": {
            "stage_a_samples": int(args.stage_a_samples),
            "stage_b_topk": int(args.stage_b_topk),
            "stage_b_step_fracs": [float(x) for x in args.stage_b_step_fracs],
            "stage_c_starts": int(args.stage_c_starts),
            "stage_c_trials": int(args.stage_c_trials),
            "stage_c_noise_frac": float(args.stage_c_noise_frac),
            "full_recheck_topk": int(args.full_recheck_topk),
            "full_refine_topk": int(args.full_refine_topk),
            "full_refine_step_fracs": [float(x) for x in args.full_refine_step_fracs],
            "max_bound_expansions": int(args.max_bound_expansions),
            "expand_factor": float(args.expand_factor),
            "bound_touch_frac": float(args.bound_touch_frac),
            "guardrail_stall_delta": float(args.guardrail_stall_delta),
            "validation_workers": int(args.validation_workers),
            "use_flashattention": (not bool(args.disable_flashattention)),
            "attention_tile_size": (
                None if args.attention_tile_size is None else int(args.attention_tile_size)
            ),
        },
        "baseline_proxy": {
            "params": asdict(baseline_proxy.params),
            "metrics": asdict(baseline_proxy.metrics),
        },
        "baseline_full": {
            "params": asdict(baseline_full.params),
            "metrics": asdict(baseline_full.metrics),
        },
        "best_full": {
            "params": asdict(best_full.params),
            "metrics": asdict(best_full.metrics),
            "stage": best_full.stage,
        },
        "top_full": [
            {
                "rank": i + 1,
                "stage": r.stage,
                "params": asdict(r.params),
                "metrics": asdict(r.metrics),
            }
            for i, r in enumerate(full_unique[:4])
        ],
        "bounds_history": bound_history,
        "expansion_events": expansion_events,
        "round_history": round_history,
        "guardrail": {
            "triggered": feasibility_guardrail_triggered,
            "reason": feasibility_guardrail_reason,
        },
    }

    summary_json = run_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nArtifacts:")
    print(f"  run dir: {run_dir}")
    print(f"  full leaderboard: {full_leaderboard}")
    print(f"  proxy leaderboard: {proxy_leaderboard}")
    print(f"  summary: {summary_json}")
    print("  emitted fitted configs:")
    for path in emitted_paths:
        print(f"    - {path}")

    return 0


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    return run_fit(args)


if __name__ == "__main__":
    raise SystemExit(main())
