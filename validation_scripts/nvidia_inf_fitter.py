#!/usr/bin/env python3
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

"""
Fit inference hardware configs to IMEC + NVIDIA validation data.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from validation_scripts import nvidia_inf  # noqa: E402
from validation_scripts.validation_helpers import (  # noqa: E402
    ValidationSpec,
    run_validation_suite,
    parse_inference_time,
)


SPSA_ITERS = 35
SPSA_DIRECTIONS = 3
SPSA_DECAY_C = True
SPSA_C_DECAY_POWER = 0.04


ADAM_MULT = 0.8
ADAM_ALPHA = ADAM_MULT * 0.25
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

LOSS_FAIL_LAMBDA = 100.0
LOSS_FAIL_VALUE = 1e6

CORE_UTIL_MIN_DEFAULT = 0.5
CORE_UTIL_MAX_DEFAULT = 1.0
DRAM_UTIL_MIN_DEFAULT = 0.5
DRAM_UTIL_MAX_DEFAULT = 1.0
NET_UTIL_MIN_DEFAULT = 0.5
NET_UTIL_MAX_DEFAULT = 1.2
OVERLAP_MIN = 0.0
OVERLAP_MAX = 1.0
KERNEL_LAUNCH_MIN = 6.5e-6
KERNEL_LAUNCH_MAX = 8.5e-6
KERNEL_STAGE1_COARSE_POINTS = 5
KERNEL_STAGE1_FINE_POINTS = 5


INIT_DEFAULTS = {
    "u_flops": 0.94,
    "u_mem": 0.92,
    "u_net_tp": 0.77,
    "tp_sp_overlap": 0.13,
    "kernel_launch_overhead": 7e-6,
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return math.log(p / (1.0 - p))


def _bounded_sigmoid(s: float, low: float, high: float) -> float:
    return low + (high - low) * _sigmoid(s)


def _s0_from_val(val: float, low: float, high: float, *, cushion_frac: float = 0.01) -> float:
    if high <= low:
        raise ValueError(f"Invalid bounds (low={low}, high={high})")
    cushion = cushion_frac * (high - low)
    lo = low + cushion
    hi = high - cushion
    if lo >= hi:
        mid = (low + high) / 2.0
        return _logit((mid - low) / (high - low))
    val = min(max(val, lo), hi)
    return _logit((val - low) / (high - low))


@dataclass(frozen=True)
class ParamSpec:
    name: str
    low: float
    high: float
    s0: float
    alpha: float
    c0: float

    def to_real(self, s: float) -> float:
        return _bounded_sigmoid(s, self.low, self.high)


def _is_list_of_dicts(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    return all(isinstance(item, dict) for item in value)


def _merge_list_of_dicts(orig: List[Dict[str, Any]], overrides: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    orig_index = {d["id"]: d for d in orig if "id" in d}
    orig_order = [d["id"] for d in orig if "id" in d]
    for d in overrides:
        dim_id = d.get("id")
        if dim_id is None:
            continue
        if dim_id in orig_index:
            orig_index[dim_id] = _deep_update(copy.deepcopy(orig_index[dim_id]), d)
        else:
            orig_index[dim_id] = copy.deepcopy(d)
    result: List[Dict[str, Any]] = []
    for dim_id in orig_order:
        result.append(orig_index[dim_id])
    for d in overrides:
        dim_id = d.get("id")
        if dim_id is None or dim_id in orig_order:
            continue
        result.append(orig_index[dim_id])
    return result


def _deep_update(target: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict):
            existing = target.get(key)
            if not isinstance(existing, dict):
                existing = {}
            target[key] = _deep_update(existing, value)
        elif isinstance(value, list):
            existing_list = target.get(key)
            if (
                isinstance(existing_list, list)
                and _is_list_of_dicts(existing_list)
                and _is_list_of_dicts(value)
            ):
                target[key] = _merge_list_of_dicts(existing_list, value)
            else:
                target[key] = copy.deepcopy(value)
        else:
            target[key] = value
    return target


def _merge_overrides(
    base: Optional[Dict[str, Any]],
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    if base is None:
        return copy.deepcopy(extra)
    merged = copy.deepcopy(base)
    return _deep_update(merged, extra)


def _params_to_overrides(
    params: Dict[str, float],
    *,
    fixed_kernel_launch_overhead: Optional[float] = None,
    core_clock_hz: Optional[float] = None,
) -> Dict[str, Any]:
    kernel_launch = (
        float(fixed_kernel_launch_overhead)
        if fixed_kernel_launch_overhead is not None
        else float(params.get("kernel_launch_overhead", INIT_DEFAULTS["kernel_launch_overhead"]))
    )
    core_overrides: Dict[str, float] = {"util": float(params["u_flops"])}
    if core_clock_hz is not None:
        core_overrides["operating_frequency"] = float(core_clock_hz)
    return {
        "tech_param": {
            "core": core_overrides,
            "DRAM": {"util": float(params["u_mem"])},
        },
        "sw_param": {"kernel_launch_overhead": kernel_launch},
        "network": {
            "overlap": {"tp_sp_overlap": float(params["tp_sp_overlap"])},
            "dimensions": [
                {"id": "dim0", "topology": {"util": float(params["u_net_tp"])}}
            ],
        },
    }


def _apply_params_to_hw_config(
    base_hw: Dict[str, Any],
    params: Dict[str, float],
    *,
    fixed_kernel_launch_overhead: Optional[float] = None,
    core_clock_hz: Optional[float] = None,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_hw)
    kernel_launch = (
        float(fixed_kernel_launch_overhead)
        if fixed_kernel_launch_overhead is not None
        else float(params.get("kernel_launch_overhead", INIT_DEFAULTS["kernel_launch_overhead"]))
    )

    tech = cfg.get("tech_param")
    if tech is None:
        tech = cfg.setdefault("tech_config", {})
    core = tech.setdefault("core", {})
    core["util"] = float(params["u_flops"])
    if core_clock_hz is not None:
        core["operating_frequency"] = float(core_clock_hz)

    dram = tech.get("DRAM")
    if dram is None:
        dram = tech.setdefault("DRAM", {})
    dram["util"] = float(params["u_mem"])

    sw = cfg.setdefault("sw_param", {})
    sw["kernel_launch_overhead"] = kernel_launch

    net = cfg.setdefault("network", {})
    overlap = net.setdefault("overlap", {})
    overlap["tp_sp_overlap"] = float(params["tp_sp_overlap"])
    dims = net.setdefault("dimensions", [])
    if dims:
        dim0 = dims[0].setdefault("topology", {})
        dim0["util"] = float(params["u_net_tp"])
    else:
        dims.append(
            {
                "id": "dim0",
                "topology": {"util": float(params["u_net_tp"])},
            }
        )

    return cfg


def _extract_initial_params(base_hw: Dict[str, Any]) -> Dict[str, float]:
    return {
        "u_flops": float(INIT_DEFAULTS["u_flops"]),
        "u_mem": float(INIT_DEFAULTS["u_mem"]),
        "u_net_tp": float(INIT_DEFAULTS["u_net_tp"]),
        "tp_sp_overlap": float(INIT_DEFAULTS["tp_sp_overlap"]),
        "kernel_launch_overhead": float(INIT_DEFAULTS["kernel_launch_overhead"]),
    }


def _build_param_specs(
    init: Dict[str, float],
    *,
    core_util_min: float,
    core_util_max: float,
    dram_util_min: float,
    dram_util_max: float,
    net_util_min: float,
    net_util_max: float,
) -> Sequence[ParamSpec]:
    return (
        ParamSpec(
            "u_flops",
            core_util_min,
            core_util_max,
            _s0_from_val(init["u_flops"], core_util_min, core_util_max, cushion_frac=0.02),
            ADAM_ALPHA,
            0.05,
        ),
        ParamSpec(
            "u_mem",
            dram_util_min,
            dram_util_max,
            _s0_from_val(init["u_mem"], dram_util_min, dram_util_max, cushion_frac=0.02),
            ADAM_ALPHA,
            0.05,
        ),
        ParamSpec(
            "u_net_tp",
            net_util_min,
            net_util_max,
            _s0_from_val(init["u_net_tp"], net_util_min, net_util_max, cushion_frac=0.02),
            ADAM_ALPHA,
            0.05,
        ),
        ParamSpec(
            "tp_sp_overlap",
            OVERLAP_MIN,
            OVERLAP_MAX,
            _s0_from_val(init["tp_sp_overlap"], OVERLAP_MIN, OVERLAP_MAX, cushion_frac=0.02),
            ADAM_ALPHA,
            0.05,
        ),
        # ParamSpec(
        #     "kernel_launch_overhead",
        #     KERNEL_LAUNCH_MIN,
        #     KERNEL_LAUNCH_MAX,
        #     _s0_from_val(
        #         init["kernel_launch_overhead"],
        #         KERNEL_LAUNCH_MIN,
        #         KERNEL_LAUNCH_MAX,
        #         cushion_frac=0.01,
        #     ),
        #     ADAM_ALPHA,
        #     0.25,
        # ),
    )


def _params_from_s(s: np.ndarray, specs: Sequence[ParamSpec]) -> Dict[str, float]:
    return {spec.name: float(spec.to_real(float(s[idx]))) for idx, spec in enumerate(specs)}


def _apply_overrides_to_specs(
    specs: Sequence[ValidationSpec],
    overrides: Dict[str, Any],
) -> List[ValidationSpec]:
    updated: List[ValidationSpec] = []
    for spec in specs:
        merged = _merge_overrides(spec.hardware_overrides, overrides)
        updated.append(replace(spec, hardware_overrides=merged))
    return updated


def _compute_suite_stats(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    errors: List[float] = []
    failures = 0
    for row in rows:
        err = row.get("pct_error")
        if row.get("success") is False or err is None:
            failures += 1
            continue
        try:
            err_val = float(err)
        except (TypeError, ValueError):
            failures += 1
            continue
        if math.isnan(err_val):
            failures += 1
            continue
        errors.append(abs(err_val))
    total = len(errors) + failures
    mean_abs = (sum(errors) / len(errors)) if errors else float("nan")
    fail_rate = float(failures) / float(total) if total > 0 else 1.0
    return {
        "mean_abs": float(mean_abs),
        "fail_rate": float(fail_rate),
        "count": float(len(errors)),
        "fail": float(failures),
    }


def _safe_mean_abs(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else float(LOSS_FAIL_VALUE)


def _compute_eval_metrics(
    imec_rows: Sequence[Dict[str, object]],
    nvidia_rows: Sequence[Dict[str, object]],
) -> Dict[str, float]:
    imec_stats = _compute_suite_stats(imec_rows)
    nvidia_stats = _compute_suite_stats(nvidia_rows)
    imec_mean = _safe_mean_abs(float(imec_stats["mean_abs"]))
    nvidia_mean = _safe_mean_abs(float(nvidia_stats["mean_abs"]))
    imec_fail_rate = float(imec_stats["fail_rate"])
    nvidia_fail_rate = float(nvidia_stats["fail_rate"])
    gap_abs = abs(imec_mean - nvidia_mean)
    balanced_mean_abs = 0.5 * (imec_mean + nvidia_mean)
    balanced_fail_rate = 0.5 * (imec_fail_rate + nvidia_fail_rate)
    stage2_loss = balanced_mean_abs + LOSS_FAIL_LAMBDA * balanced_fail_rate
    return {
        "imec_mean_abs": imec_mean,
        "nvidia_mean_abs": nvidia_mean,
        "imec_fail_rate": imec_fail_rate,
        "nvidia_fail_rate": nvidia_fail_rate,
        "imec_count": float(imec_stats["count"]),
        "nvidia_count": float(nvidia_stats["count"]),
        "imec_fail": float(imec_stats["fail"]),
        "nvidia_fail": float(nvidia_stats["fail"]),
        "gap_abs": float(gap_abs),
        "balanced_mean_abs": float(balanced_mean_abs),
        "balanced_fail_rate": float(balanced_fail_rate),
        "stage2_loss": float(stage2_loss),
        # Keep aliases for existing logs/callers.
        "mean_abs": float(balanced_mean_abs),
        "fail_rate": float(balanced_fail_rate),
    }


def _stage1_rank_key(metrics: Dict[str, float], kernel_launch_overhead: float) -> Tuple[float, float, float, float]:
    return (
        float(metrics["gap_abs"]),
        float(metrics["balanced_mean_abs"]),
        float(metrics["imec_fail_rate"] + metrics["nvidia_fail_rate"]),
        float(kernel_launch_overhead),
    )


def _kernel_candidates(low: float, high: float, points: int) -> List[float]:
    if points <= 1 or low >= high:
        return [float(low)]
    return [float(x) for x in np.linspace(float(low), float(high), int(points))]


def _evaluate_params(
    specs: Sequence[ValidationSpec],
    imec_count: int,
    imec_lookup: Dict[Tuple[str, int], float],
    nvidia_lookup: Dict[Tuple[str, int, int, int, int], float],
    base_model_path: str,
    base_hw_path: str,
    overrides: Dict[str, Any],
    *,
    max_workers: Optional[int],
    cache_mode: str,
) -> Tuple[float, Dict[str, float]]:
    return _evaluate_param_sets(
        specs,
        imec_count,
        imec_lookup,
        nvidia_lookup,
        base_model_path,
        base_hw_path,
        [overrides],
        max_workers=max_workers,
        cache_mode=cache_mode,
    )[0]


def _evaluate_param_sets(
    specs: Sequence[ValidationSpec],
    imec_count: int,
    imec_lookup: Dict[Tuple[str, int], float],
    nvidia_lookup: Dict[Tuple[str, int, int, int, int], float],
    base_model_path: str,
    base_hw_path: str,
    overrides_list: Sequence[Dict[str, Any]],
    *,
    max_workers: Optional[int],
    cache_mode: str,
) -> List[Tuple[float, Dict[str, float]]]:
    if not overrides_list:
        return []
    combined_specs: List[ValidationSpec] = []
    spec_count = len(specs)
    for overrides in overrides_list:
        combined_specs.extend(_apply_overrides_to_specs(specs, overrides))
    results = run_validation_suite(
        combined_specs,
        base_model_config_path=base_model_path,
        base_hardware_config_path=base_hw_path,
        result_parser=parse_inference_time,
        run_perf_path=nvidia_inf.RUN_PERF,
        max_workers=max_workers,
        cache_mode=cache_mode,
        show_progress=False,
    )
    outputs: List[Tuple[float, Dict[str, float]]] = []
    offset = 0
    for _ in overrides_list:
        subset = results[offset:offset + spec_count]
        imec_results = subset[:imec_count]
        nvidia_results = subset[imec_count:]
        imec_rows: List[Dict[str, object]] = []
        nvidia_rows: List[Dict[str, object]] = []
        if imec_results:
            imec_rows = nvidia_inf.compute_pct_errors(imec_results, imec_lookup)
        if nvidia_results:
            nvidia_rows = nvidia_inf.compute_nvidia_pct_errors(nvidia_results, nvidia_lookup)
        metrics = _compute_eval_metrics(imec_rows, nvidia_rows)
        outputs.append((float(metrics["stage2_loss"]), metrics))
        offset += spec_count
    return outputs


def _search_kernel_launch_overhead(
    specs: Sequence[ValidationSpec],
    imec_count: int,
    imec_lookup: Dict[Tuple[str, int], float],
    nvidia_lookup: Dict[Tuple[str, int, int, int, int], float],
    base_model_path: str,
    base_hw_path: str,
    base_params: Dict[str, float],
    *,
    max_workers: Optional[int],
    cache_mode: str,
    fixed_kernel_launch_overhead: Optional[float] = None,
    core_clock_hz: Optional[float] = None,
) -> Tuple[float, Dict[str, float]]:
    if fixed_kernel_launch_overhead is not None:
        fixed = float(fixed_kernel_launch_overhead)
        params = dict(base_params)
        params["kernel_launch_overhead"] = fixed
        loss, metrics = _evaluate_params(
            specs,
            imec_count,
            imec_lookup,
            nvidia_lookup,
            base_model_path,
            base_hw_path,
            _params_to_overrides(
                params,
                fixed_kernel_launch_overhead=fixed,
                core_clock_hz=core_clock_hz,
            ),
            max_workers=max_workers,
            cache_mode=cache_mode,
        )
        _ = loss
        return fixed, metrics

    def _eval_kernel_grid(candidates: Sequence[float]) -> Tuple[float, Dict[str, float]]:
        overrides_list: List[Dict[str, Any]] = []
        kernels: List[float] = []
        for kernel in candidates:
            params = dict(base_params)
            params["kernel_launch_overhead"] = float(kernel)
            overrides_list.append(
                _params_to_overrides(
                    params,
                    fixed_kernel_launch_overhead=None,
                    core_clock_hz=core_clock_hz,
                )
            )
            kernels.append(float(kernel))

        losses_stats = _evaluate_param_sets(
            specs,
            imec_count,
            imec_lookup,
            nvidia_lookup,
            base_model_path,
            base_hw_path,
            overrides_list,
            max_workers=max_workers,
            cache_mode=cache_mode,
        )
        if not losses_stats:
            raise RuntimeError("Kernel stage-1 search did not produce any evaluations.")

        best_idx = min(
            range(len(losses_stats)),
            key=lambda idx: _stage1_rank_key(losses_stats[idx][1], kernels[idx]),
        )
        return kernels[best_idx], losses_stats[best_idx][1]

    coarse = _kernel_candidates(KERNEL_LAUNCH_MIN, KERNEL_LAUNCH_MAX, KERNEL_STAGE1_COARSE_POINTS)
    coarse_best_kernel, coarse_best_metrics = _eval_kernel_grid(coarse)

    if len(coarse) >= 2:
        coarse_step = abs(float(coarse[1]) - float(coarse[0]))
    else:
        coarse_step = float(KERNEL_LAUNCH_MAX - KERNEL_LAUNCH_MIN) / 2.0
    fine_low = max(float(KERNEL_LAUNCH_MIN), float(coarse_best_kernel) - coarse_step)
    fine_high = min(float(KERNEL_LAUNCH_MAX), float(coarse_best_kernel) + coarse_step)
    if fine_low >= fine_high:
        return float(coarse_best_kernel), coarse_best_metrics

    fine = _kernel_candidates(fine_low, fine_high, KERNEL_STAGE1_FINE_POINTS)
    fine_best_kernel, fine_best_metrics = _eval_kernel_grid(fine)
    return float(fine_best_kernel), fine_best_metrics


def _run_spsa(
    specs: Sequence[ValidationSpec],
    imec_count: int,
    imec_lookup: Dict[Tuple[str, int], float],
    nvidia_lookup: Dict[Tuple[str, int, int, int, int], float],
    base_model_path: str,
    base_hw_path: str,
    base_hw: Dict[str, Any],
    *,
    iters: int,
    seed: int,
    max_workers: Optional[int],
    cache_mode: str,
    core_util_min: float,
    core_util_max: float,
    dram_util_min: float,
    dram_util_max: float,
    net_util_min: float,
    net_util_max: float,
    fixed_kernel_launch_overhead: Optional[float],
    core_clock_hz: Optional[float],
) -> Dict[str, float]:
    init = _extract_initial_params(base_hw)
    param_specs = _build_param_specs(
        init,
        core_util_min=core_util_min,
        core_util_max=core_util_max,
        dram_util_min=dram_util_min,
        dram_util_max=dram_util_max,
        net_util_min=net_util_min,
        net_util_max=net_util_max,
    )
    s = np.array([spec.s0 for spec in param_specs], dtype=float)
    m = np.zeros_like(s)
    v = np.zeros_like(s)
    rng = random.Random(seed)

    stage1_base_params = _params_from_s(s, param_specs)
    stage1_base_params["kernel_launch_overhead"] = float(init["kernel_launch_overhead"])
    if fixed_kernel_launch_overhead is None:
        print(
            "stage1 kernel search: range=[{:.2e}, {:.2e}] coarse_points={} fine_points={}".format(
                float(KERNEL_LAUNCH_MIN),
                float(KERNEL_LAUNCH_MAX),
                int(KERNEL_STAGE1_COARSE_POINTS),
                int(KERNEL_STAGE1_FINE_POINTS),
            )
        )
    else:
        print(f"stage1 kernel fixed: kernel_launch_overhead={float(fixed_kernel_launch_overhead):.2e}")
    best_kernel_launch_overhead, stage1_metrics = _search_kernel_launch_overhead(
        specs,
        imec_count,
        imec_lookup,
        nvidia_lookup,
        base_model_path,
        base_hw_path,
        stage1_base_params,
        max_workers=max_workers,
        cache_mode=cache_mode,
        fixed_kernel_launch_overhead=fixed_kernel_launch_overhead,
        core_clock_hz=core_clock_hz,
    )
    print(
        "stage1 winner: kernel_launch_overhead={:.2e} | gap={:.2f}% | imec_mae={:.2f}% | "
        "nvidia_mae={:.2f}% | fail_sum={:.2f}".format(
            float(best_kernel_launch_overhead),
            float(stage1_metrics.get("gap_abs", float("nan"))),
            float(stage1_metrics.get("imec_mean_abs", float("nan"))),
            float(stage1_metrics.get("nvidia_mean_abs", float("nan"))),
            float(stage1_metrics.get("imec_fail_rate", float("nan")))
            + float(stage1_metrics.get("nvidia_fail_rate", float("nan"))),
        )
    )

    best_loss = float("inf")
    best_s = s.copy()

    init_params = _params_from_s(s, param_specs)
    init_params["kernel_launch_overhead"] = float(best_kernel_launch_overhead)
    print(
        "init params: u_flops={:.3f}, u_mem={:.3f}, u_net_tp={:.3f}, tp_sp_overlap={:.3f}, "
        "kernel_launch_overhead={:.2e}".format(
            init_params["u_flops"],
            init_params["u_mem"],
            init_params["u_net_tp"],
            init_params["tp_sp_overlap"],
            init_params["kernel_launch_overhead"],
        )
    )

    for k in range(max(1, int(iters))):
        c_vec = np.array([spec.c0 for spec in param_specs], dtype=float)
        if SPSA_DECAY_C:
            c_vec = c_vec / ((k + 1) ** SPSA_C_DECAY_POWER)
        grad = np.zeros_like(s)
        iter_best_loss = None
        iter_best_stats = None

        deltas = []
        s_variants = []
        for _ in range(max(1, int(SPSA_DIRECTIONS))):
            delta = np.array([1.0 if rng.random() > 0.5 else -1.0 for _ in range(len(s))], dtype=float)
            deltas.append(delta)
            s_variants.append(s + c_vec * delta)
            s_variants.append(s - c_vec * delta)

        overrides_list = [
            _params_to_overrides(
                {
                    **_params_from_s(variant, param_specs),
                    "kernel_launch_overhead": float(best_kernel_launch_overhead),
                },
                fixed_kernel_launch_overhead=fixed_kernel_launch_overhead,
                core_clock_hz=core_clock_hz,
            )
            for variant in s_variants
        ]
        losses_stats = _evaluate_param_sets(
            specs,
            imec_count,
            imec_lookup,
            nvidia_lookup,
            base_model_path,
            base_hw_path,
            overrides_list,
            max_workers=max_workers,
            cache_mode=cache_mode,
        )

        for dir_idx, delta in enumerate(deltas):
            loss_plus, stats_plus = losses_stats[2 * dir_idx]
            loss_minus, stats_minus = losses_stats[2 * dir_idx + 1]

            grad += (loss_plus - loss_minus) / (2.0 * c_vec) * delta

            for loss_val, stats_val, s_val in (
                (loss_plus, stats_plus, s_variants[2 * dir_idx]),
                (loss_minus, stats_minus, s_variants[2 * dir_idx + 1]),
            ):
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_s = s_val.copy()
                if iter_best_loss is None or loss_val < iter_best_loss:
                    iter_best_loss = loss_val
                    iter_best_stats = stats_val

        grad /= max(1, int(SPSA_DIRECTIONS))
        m = ADAM_BETA1 * m + (1.0 - ADAM_BETA1) * grad
        v = ADAM_BETA2 * v + (1.0 - ADAM_BETA2) * (grad ** 2)
        m_hat = m / (1.0 - ADAM_BETA1 ** (k + 1))
        v_hat = v / (1.0 - ADAM_BETA2 ** (k + 1))
        alpha_vec = np.array([spec.alpha for spec in param_specs], dtype=float)
        s = s - alpha_vec * m_hat / (np.sqrt(v_hat) + ADAM_EPS)

        stats_ref = iter_best_stats or {}
        balanced_mean_abs = stats_ref.get("balanced_mean_abs", float("nan"))
        imec_mae = stats_ref.get("imec_mean_abs", float("nan"))
        nvidia_mae = stats_ref.get("nvidia_mean_abs", float("nan"))
        gap_abs = stats_ref.get("gap_abs", float("nan"))
        fail_rate = stats_ref.get("balanced_fail_rate", float("nan"))
        print(
            "iter {:02d} | loss={:.3f} | best={:.3f} | balanced_mae={:.2f}% | imec_mae={:.2f}% | "
            "nvidia_mae={:.2f}% | gap={:.2f}% | fail_rate={:.2f}".format(
                k + 1,
                float(iter_best_loss) if iter_best_loss is not None else float("nan"),
                float(best_loss),
                float(balanced_mean_abs) if math.isfinite(float(balanced_mean_abs)) else float("nan"),
                float(imec_mae) if math.isfinite(float(imec_mae)) else float("nan"),
                float(nvidia_mae) if math.isfinite(float(nvidia_mae)) else float("nan"),
                float(gap_abs) if math.isfinite(float(gap_abs)) else float("nan"),
                float(fail_rate) if math.isfinite(float(fail_rate)) else float("nan"),
            )
        )
        current_params = _params_from_s(s, param_specs)
        current_params["kernel_launch_overhead"] = float(best_kernel_launch_overhead)
        print(
            "  params: u_flops={:.3f}, u_mem={:.3f}, u_net_tp={:.3f}, tp_sp_overlap={:.3f}, "
            "kernel_launch_overhead={:.2e}".format(
                current_params["u_flops"],
                current_params["u_mem"],
                current_params["u_net_tp"],
                current_params["tp_sp_overlap"],
                current_params["kernel_launch_overhead"],
            )
        )

    final_params = _params_from_s(best_s, param_specs)
    final_params["kernel_launch_overhead"] = float(best_kernel_launch_overhead)
    return final_params


def _load_base_hw(path: str) -> Dict[str, Any]:
    with open(path, "r") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Hardware config {path} did not contain a YAML dict.")
    return data


def _build_specs(
    device: str,
    *,
    network_ignored: bool,
) -> Tuple[
    List[ValidationSpec],
    int,
    Dict[Tuple[str, int], float],
    Dict[Tuple[str, int, int, int, int], float],
    str,
    str,
]:
    imec_specs: List[ValidationSpec] = []
    nvidia_specs: List[ValidationSpec] = []
    imec_lookup: Dict[Tuple[str, int], float] = {}
    nvidia_lookup: Dict[Tuple[str, int, int, int, int], float] = {}
    base_model_path: Optional[str] = None
    base_hw_path = os.path.join(nvidia_inf.HARDWARE_CONFIG_PATH, nvidia_inf.HW_CONFIGS[device])

    try:
        specs, lookup, model_path, hw_path = nvidia_inf.build_specs_for_device(
            device, network_ignored=network_ignored
        )
        imec_specs = specs
        imec_lookup = lookup
        base_model_path = base_model_path or model_path
        base_hw_path = base_hw_path or hw_path
    except FileNotFoundError:
        pass

    try:
        specs, lookup, model_path, hw_path = nvidia_inf.build_nvidia_specs_for_device(
            device, network_ignored=network_ignored
        )
        nvidia_specs = specs
        nvidia_lookup = lookup
        base_model_path = base_model_path or model_path
        base_hw_path = base_hw_path or hw_path
    except FileNotFoundError:
        pass

    if not imec_specs and not nvidia_specs:
        raise ValueError(f"No validation specs found for device {device}.")
    if base_model_path is None or base_hw_path is None:
        raise ValueError(f"Missing base paths for device {device}.")

    combined = []
    for spec in imec_specs + nvidia_specs:
        combined.append(replace(spec, hardware_config_path=base_hw_path))
    return combined, len(imec_specs), imec_lookup, nvidia_lookup, base_model_path, base_hw_path


def fit_device(
    device: str,
    *,
    iters: int,
    seed: int,
    max_workers: Optional[int],
    cache_mode: str,
    output_dir: Optional[str],
    core_util_min: float,
    core_util_max: float,
    dram_util_min: float,
    dram_util_max: float,
    net_util_min: float,
    net_util_max: float,
    fixed_kernel_launch_overhead: Optional[float],
    core_clock_hz: Optional[float],
) -> Optional[Path]:
    specs, imec_count, imec_lookup, nvidia_lookup, base_model_path, base_hw_path = _build_specs(
        device, network_ignored=False
    )
    base_hw = _load_base_hw(base_hw_path)
    params = _run_spsa(
        specs,
        imec_count,
        imec_lookup,
        nvidia_lookup,
        base_model_path,
        base_hw_path,
        base_hw,
        iters=iters,
        seed=seed,
        max_workers=max_workers,
        cache_mode=cache_mode,
        core_util_min=core_util_min,
        core_util_max=core_util_max,
        dram_util_min=dram_util_min,
        dram_util_max=dram_util_max,
        net_util_min=net_util_min,
        net_util_max=net_util_max,
        fixed_kernel_launch_overhead=fixed_kernel_launch_overhead,
        core_clock_hz=core_clock_hz,
    )
    fitted = _apply_params_to_hw_config(
        base_hw,
        params,
        fixed_kernel_launch_overhead=fixed_kernel_launch_overhead,
        core_clock_hz=core_clock_hz,
    )
    fitted_name = os.path.basename(nvidia_inf.HW_CONFIGS[device]).replace(".yaml", nvidia_inf.FITTED_HW_SUFFIX)
    if output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fitted_name = fitted_name.replace(".yaml", f".{timestamp}.yaml")
        out_path = Path(output_dir) / fitted_name
    else:
        out_path = Path(nvidia_inf.HARDWARE_CONFIG_PATH) / fitted_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        yaml.safe_dump(fitted, handle, sort_keys=False)
    print(f"Saved fitted config for {device}: {out_path}")
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fit inference hardware configs to IMEC + NVIDIA data.")
    parser.add_argument(
        "--device",
        default="all",
        choices=["all", "A100", "H100"],
        help="Device to fit (default: all).",
    )
    parser.add_argument("--iters", type=int, default=SPSA_ITERS, help="Number of SPSA iterations.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--max-workers", type=int, default=105, help="Max validation workers.")
    parser.add_argument("--cache-mode", type=str, default="NO_CACHE", help="Astra cache mode.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write fitted config(s). Defaults to hardware-config path when unset.",
    )
    parser.add_argument(
        "--fixed-kernel-launch-overhead",
        type=float,
        default=None,
        help="Fix kernel_launch_overhead (seconds) instead of searching over a range.",
    )
    parser.add_argument(
        "--core-clock-hz",
        type=float,
        default=None,
        help="If set, override tech_param.core.operating_frequency (Hz) during fitting and in output config.",
    )
    parser.add_argument("--core-util-min", type=float, default=CORE_UTIL_MIN_DEFAULT)
    parser.add_argument("--core-util-max", type=float, default=CORE_UTIL_MAX_DEFAULT)
    parser.add_argument("--dram-util-min", type=float, default=DRAM_UTIL_MIN_DEFAULT)
    parser.add_argument("--dram-util-max", type=float, default=DRAM_UTIL_MAX_DEFAULT)
    parser.add_argument("--net-util-min", type=float, default=NET_UTIL_MIN_DEFAULT)
    parser.add_argument("--net-util-max", type=float, default=NET_UTIL_MAX_DEFAULT)
    args = parser.parse_args(argv)
    os.environ.setdefault("RAPID_VALIDATION_QUIET", "1")

    bounds = [
        ("core util", float(args.core_util_min), float(args.core_util_max)),
        ("dram util", float(args.dram_util_min), float(args.dram_util_max)),
        ("network dim0 util", float(args.net_util_min), float(args.net_util_max)),
    ]
    for label, low, high in bounds:
        if high < low:
            parser.error(f"Invalid {label} bounds: min={low} must be <= max={high}")

    devices = ["A100", "H100"] if args.device == "all" else [args.device]
    for device in devices:
        try:
            fit_device(
                device,
                iters=int(args.iters),
                seed=int(args.seed),
                max_workers=args.max_workers,
                cache_mode=str(args.cache_mode),
                output_dir=args.output_dir,
                core_util_min=float(args.core_util_min),
                core_util_max=float(args.core_util_max),
                dram_util_min=float(args.dram_util_min),
                dram_util_max=float(args.dram_util_max),
                net_util_min=float(args.net_util_min),
                net_util_max=float(args.net_util_max),
                fixed_kernel_launch_overhead=(
                    float(args.fixed_kernel_launch_overhead)
                    if args.fixed_kernel_launch_overhead is not None
                    else None
                ),
                core_clock_hz=float(args.core_clock_hz) if args.core_clock_hz is not None else None,
            )
        except FileNotFoundError as exc:
            print(f"[warn] Missing data for {device}: {exc}")
        except ValueError as exc:
            print(f"[warn] {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
