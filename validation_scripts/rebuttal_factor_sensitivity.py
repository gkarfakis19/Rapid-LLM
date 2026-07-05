#!/usr/bin/env python3
"""Bounded Table III factor sensitivity experiment for rebuttal.

This script follows validation_scripts/rebuttal_factor_sensitivity/GOAL.md.
It re-runs RAPID-LLM on four existing validation configurations while
perturbing one frozen Table III factor at a time.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
OUT_ROOT = SCRIPT_DIR / "rebuttal_factor_sensitivity"
CASE_ROOT = OUT_ROOT / "case_configs"
RESULTS_CSV = OUT_ROOT / "factor_sensitivity_results.csv"
SUMMARY_MD = OUT_ROOT / "factor_sensitivity_summary.md"
COMMANDS_TXT = OUT_ROOT / "commands.txt"
DERATES_YAML = SCRIPT_DIR / "validation_configs" / "harness_derates.yaml"
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
RUN_PERF = PROJECT_ROOT / "run_perf.py"

TRAIN_RE = re.compile(r"Training time for batch:\s*([0-9]+(?:\.[0-9]+)?)s")
INF_RE = re.compile(r"LLM inference time:\s*([0-9]+(?:\.[0-9]+)?)s")

VARIANTS = [
    ("baseline", "", 1.0),
    ("compute_m10", "compute", 0.9),
    ("compute_p10", "compute", 1.1),
    ("memory_m10", "memory", 0.9),
    ("memory_p10", "memory", 1.1),
    ("communication_m10", "communication", 0.9),
    ("communication_p10", "communication", 1.1),
]

CSV_COLUMNS = [
    "case_id",
    "variant",
    "status",
    "regime",
    "source_csv",
    "model",
    "training_or_inference",
    "gpu_type",
    "num_gpus",
    "parallelism",
    "baseline_compute_factor",
    "baseline_memory_factor",
    "baseline_communication_factor",
    "launch_overhead_s",
    "perturbed_factor",
    "perturbation_multiplier",
    "compute_factor",
    "memory_factor",
    "communication_factor",
    "predicted_runtime_s",
    "baseline_runtime_s",
    "signed_runtime_change_percent",
    "absolute_runtime_change_percent",
    "elasticity",
    "hardware_config",
    "model_config",
    "command",
    "error",
]


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    regime: str
    source_csv: Path
    model: str
    training_or_inference: str
    gpu_type: str
    device_type: str
    num_gpus: int
    parallelism: str
    model_config: dict[str, Any]
    hardware_config: dict[str, Any]


@dataclass(frozen=True)
class PlannedRun:
    order: int
    case: CaseSpec
    variant: str
    perturbed_factor: str
    perturbation_multiplier: float
    factors: dict[str, float]
    run_dir: Path
    hardware_path: Path
    model_path: Path
    command: str


@dataclass(frozen=True)
class RunResult:
    order: int
    status: str
    predicted_runtime_s: float | None
    error: str


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _write_yaml(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False)


def _is_list_of_dicts(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, Mapping) for item in value)


def _merge_list_by_id(original: list[Any], overrides: list[Any]) -> list[Any]:
    original_by_id = {
        item.get("id"): copy.deepcopy(dict(item))
        for item in original
        if isinstance(item, Mapping) and "id" in item
    }
    original_order = [
        item.get("id") for item in original if isinstance(item, Mapping) and "id" in item
    ]
    merged: list[Any] = [
        copy.deepcopy(item)
        for item in original
        if not isinstance(item, Mapping) or "id" not in item
    ]

    for item in overrides:
        if not isinstance(item, Mapping) or "id" not in item:
            continue
        item_id = item["id"]
        base = original_by_id.get(item_id, {})
        original_by_id[item_id] = _deep_update_merge_lists(copy.deepcopy(base), item)

    for item_id in original_order:
        merged.append(original_by_id[item_id])
    for item in overrides:
        if not isinstance(item, Mapping) or "id" not in item:
            continue
        item_id = item["id"]
        if item_id not in original_order:
            merged.append(original_by_id[item_id])
    return merged


def _deep_update_merge_lists(target: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            target[key] = _deep_update_merge_lists(dict(target[key]), value)
        elif _is_list_of_dicts(value) and _is_list_of_dicts(target.get(key)):
            target[key] = _merge_list_by_id(list(target[key]), list(value))
        else:
            target[key] = copy.deepcopy(value)
    return target


def _deep_update_replace_lists(target: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            target[key] = _deep_update_replace_lists(dict(target[key]), value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def _merge_configs(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any],
    *,
    merge_lists_by_id: bool,
) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    if merge_lists_by_id:
        return _deep_update_merge_lists(merged, overrides)
    return _deep_update_replace_lists(merged, overrides)


def _read_first_matching_row(
    path: Path,
    predicate: Callable[[Mapping[str, str]], bool],
    *,
    encoding: str = "utf-8",
) -> dict[str, str]:
    with path.open(newline="", encoding=encoding) as handle:
        for row in csv.DictReader(handle):
            if predicate(row):
                return dict(row)
    raise ValueError(f"No matching row in {path}")


def _load_derates() -> dict[str, Any]:
    data = _load_yaml(DERATES_YAML)
    shared = data["shared"]
    devices = data["device_types"]
    return {
        "launch_overhead_s": float(shared["kernel_launch_overhead_s"]),
        "A100_SXM4": {
            "compute": float(devices["A100_SXM4"]["compute_util"]),
            "memory": float(devices["A100_SXM4"]["dram_util"]),
            "communication": float(devices["A100_SXM4"]["network_util"]),
        },
        "A100_PCIe": {
            "compute": float(devices["A100_PCIe"]["compute_util"]),
            "memory": float(devices["A100_PCIe"]["dram_util"]),
            "communication": float(devices["A100_PCIe"]["network_util"]),
        },
    }


def _apply_table_factors(
    hardware_config: dict[str, Any],
    *,
    factors: Mapping[str, float],
    launch_overhead_s: float,
) -> dict[str, Any]:
    hw = copy.deepcopy(hardware_config)
    hw.setdefault("sw_param", {})["kernel_launch_overhead"] = float(launch_overhead_s)
    hw.setdefault("tech_param", {}).setdefault("core", {})["util"] = float(
        factors["compute"]
    )
    hw.setdefault("tech_param", {}).setdefault("DRAM", {})["util"] = float(
        factors["memory"]
    )
    for dim in hw.get("network", {}).get("dimensions", []):
        if isinstance(dim, Mapping):
            dim.setdefault("topology", {})["util"] = float(factors["communication"])
    return hw


def _build_nvidia_train_case() -> CaseSpec:
    from validation_scripts import nvidia_train_validation as ntv

    source_csv = SCRIPT_DIR / "train_validation_data" / "nvidia_train_validation_cases.csv"
    row = _read_first_matching_row(
        source_csv,
        lambda r: (
            r.get("device") == "A100_korthi"
            and r.get("model") == "GPT 1T"
            and r.get("batch") == "512"
            and r.get("mb") == "512"
            and r.get("dp") == "1"
            and r.get("tp") == "8"
            and r.get("pp") == "64"
            and r.get("cp") == "1"
            and str(r.get("tp_sp")).lower() == "false"
            and r.get("recomputation") == "full"
        ),
    )
    base_model_path = ntv._resolve_model_config(row["model"])  # type: ignore[attr-defined]
    base_hw_path = SCRIPT_DIR / "validation_configs" / "hardware-config" / "a100_80GB_train_validation.yaml"
    model_overrides, hw_overrides = ntv._build_overrides(row)  # type: ignore[attr-defined]
    model_cfg = _merge_configs(_load_yaml(base_model_path), model_overrides, merge_lists_by_id=True)
    hw_cfg = _merge_configs(_load_yaml(base_hw_path), hw_overrides, merge_lists_by_id=True)
    return CaseSpec(
        case_id="large_train_gpt1t_korthi_full",
        regime="large-scale compute-heavy training",
        source_csv=source_csv,
        model="GPT 1T",
        training_or_inference="training",
        gpu_type="A100 SXM4 80GB",
        device_type="A100_SXM4",
        num_gpus=1 * 8 * 64 * 1,
        parallelism="DP=1, TP=8, PP=64, CP=1, TP_SP=False",
        model_config=model_cfg,
        hardware_config=hw_cfg,
    )


def _build_inference_case(case_id: str, model: str, regime: str) -> CaseSpec:
    from validation_scripts import nvidia_inf

    source_csv = SCRIPT_DIR / "imec_data" / "A100_inf.csv"
    _read_first_matching_row(
        source_csv,
        lambda r: r.get("device") == "A100" and r.get("model") == model and r.get("TP") == "8",
        encoding="utf-8-sig",
    )
    spec, hw_path, model_path = nvidia_inf._build_spec(  # type: ignore[attr-defined]
        "A100",
        model,
        8,
        0,
        network_ignored=False,
        fit_model=False,
    )
    model_cfg = _merge_configs(
        _load_yaml(Path(model_path)),
        spec.model_overrides or {},
        merge_lists_by_id=True,
    )
    hw_cfg = _merge_configs(
        _load_yaml(Path(hw_path)),
        spec.hardware_overrides or {},
        merge_lists_by_id=True,
    )
    return CaseSpec(
        case_id=case_id,
        regime=regime,
        source_csv=source_csv,
        model=model,
        training_or_inference="inference",
        gpu_type="A100 SXM4 80GB",
        device_type="A100_SXM4",
        num_gpus=8,
        parallelism="DP=1, TP=8, PP=1, CP=1, TP_SP=True",
        model_config=model_cfg,
        hardware_config=hw_cfg,
    )


def _build_uci_case() -> CaseSpec:
    from validation_scripts import uci_train_validation as uci

    source_csv = SCRIPT_DIR / "train_validation_data" / "uci_train.csv"
    row = _read_first_matching_row(
        source_csv,
        lambda r: (
            r.get("variant") == "DDP"
            and r.get("TP") == "1"
            and r.get("PP") == "1"
            and r.get("CP") == "1"
            and r.get("DP") == "4"
        ),
    )
    tp = int(row["TP"])
    pp = int(row["PP"])
    cp = int(row["CP"])
    dp = int(row["DP"])
    mb = uci.GLOBAL_BATCH_SIZE // (uci.MICRO_BATCH_SIZE * dp)
    eff_mb = mb if pp > 1 else 1
    network_override, _ = uci._build_network_override(  # type: ignore[attr-defined]
        tp,
        pp,
        cp,
        dp,
        "full_astrasim_hierarchical",
    )
    model_overrides = {
        "model_param": {
            "seq_len": 4096,
            "run_type": "training",
        }
    }
    hw_overrides: dict[str, Any] = {
        "parallelism": {
            "tp": tp,
            "tp_sp": False,
            "cp": cp,
            "pp": pp,
            "mb": eff_mb,
            "train": {"dp": dp, "ep": 1, "tp_ep": True},
            "inference": {"replica_count": 1, "moe_dp": 1},
        },
        "sw_param": {
            "dp_zero_stage": 0,
        },
    }
    hw_overrides.update(network_override)
    model_cfg = _merge_configs(
        _load_yaml(uci.DEFAULT_MODEL_CONFIG),
        model_overrides,
        merge_lists_by_id=False,
    )
    hw_cfg = _merge_configs(
        _load_yaml(uci.DEFAULT_HW_CONFIG),
        hw_overrides,
        merge_lists_by_id=False,
    )
    return CaseSpec(
        case_id="pcie_train_uci_ddp_4gpu",
        regime="PCIe/DDP 4-GPU training",
        source_csv=source_csv,
        model="Llama 2-7B",
        training_or_inference="training",
        gpu_type="A100 PCIe 80GB",
        device_type="A100_PCIe",
        num_gpus=4,
        parallelism="DP=4, TP=1, PP=1, CP=1, TP_SP=False",
        model_config=model_cfg,
        hardware_config=hw_cfg,
    )


def _build_cases() -> list[CaseSpec]:
    return [
        _build_nvidia_train_case(),
        _build_inference_case(
            "decode_inf_llama2_70b_tp8",
            "Llama 2-70B",
            "large-model decode-heavy inference",
        ),
        _build_inference_case(
            "comm_inf_llama2_7b_tp8",
            "Llama 2-7B",
            "high-TP small-model inference",
        ),
        _build_uci_case(),
    ]


def _variant_factors(
    baseline: Mapping[str, float],
    perturbed_factor: str,
    multiplier: float,
) -> dict[str, float]:
    factors = {
        "compute": float(baseline["compute"]),
        "memory": float(baseline["memory"]),
        "communication": float(baseline["communication"]),
    }
    if perturbed_factor:
        factors[perturbed_factor] = factors[perturbed_factor] * float(multiplier)
    return factors


def _format_command(run_dir: Path, hardware_path: Path, model_path: Path) -> str:
    parts = [
        "cd",
        str(run_dir),
        "&&",
        str(PYTHON),
        str(RUN_PERF),
        "--hardware_config",
        str(hardware_path),
        "--model_config",
        str(model_path),
    ]
    return " ".join(shlex.quote(part) for part in parts)


def _plan_runs(cases: Iterable[CaseSpec], derates: Mapping[str, Any]) -> list[PlannedRun]:
    planned: list[PlannedRun] = []
    order = 0
    launch_overhead_s = float(derates["launch_overhead_s"])
    case_list = list(cases)
    for variant, perturbed_factor, multiplier in VARIANTS:
        for case in case_list:
            baseline = derates[case.device_type]
            factors = _variant_factors(baseline, perturbed_factor, multiplier)
            run_dir = (CASE_ROOT / case.case_id / variant).resolve()
            hardware_path = run_dir / "hardware.yaml"
            model_path = run_dir / "model.yaml"
            hw_cfg = _apply_table_factors(
                case.hardware_config,
                factors=factors,
                launch_overhead_s=launch_overhead_s,
            )
            _write_yaml(model_path, case.model_config)
            _write_yaml(hardware_path, hw_cfg)
            planned.append(
                PlannedRun(
                    order=order,
                    case=case,
                    variant=variant,
                    perturbed_factor=perturbed_factor,
                    perturbation_multiplier=float(multiplier),
                    factors=factors,
                    run_dir=run_dir,
                    hardware_path=hardware_path,
                    model_path=model_path,
                    command=_format_command(run_dir, hardware_path, model_path),
                )
            )
            order += 1
    return planned


def _parse_runtime(output: str, training_or_inference: str) -> float | None:
    regex = TRAIN_RE if training_or_inference == "training" else INF_RE
    match = regex.search(output or "")
    if not match:
        return None
    return float(match.group(1))


def _execute_run(run: PlannedRun) -> RunResult:
    run.run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run.run_dir / "run.log"
    env = os.environ.copy()
    env["ASTRA_CACHE_DIR"] = str(run.run_dir / "astra_cache")
    env["RAPID_ASTRA_CACHE_MODE"] = "NO_CACHE"
    env["DEEPFLOW_ASTRA_CACHE_MODE"] = "no_cache"
    cmd = [
        str(PYTHON),
        str(RUN_PERF),
        "--hardware_config",
        str(run.hardware_path),
        "--model_config",
        str(run.model_path),
    ]
    with log_path.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.run(
            cmd,
            cwd=run.run_dir,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    output = log_path.read_text(encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        return RunResult(
            order=run.order,
            status="failed",
            predicted_runtime_s=None,
            error=f"return code {proc.returncode}; see {_rel(log_path)}",
        )
    runtime = _parse_runtime(output, run.case.training_or_inference)
    if runtime is None:
        return RunResult(
            order=run.order,
            status="failed",
            predicted_runtime_s=None,
            error=f"failed to parse runtime; see {_rel(log_path)}",
        )
    return RunResult(
        order=run.order,
        status="success",
        predicted_runtime_s=runtime,
        error="",
    )


def _run_all(planned: list[PlannedRun], workers: int) -> dict[int, RunResult]:
    results: dict[int, RunResult] = {}
    worker_count = max(1, min(int(workers), len(planned)))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(_execute_run, run): run for run in planned}
        for future in as_completed(futures):
            run = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - protects the report.
                result = RunResult(
                    order=run.order,
                    status="failed",
                    predicted_runtime_s=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            results[result.order] = result
            print(f"[{result.status}] {run.case.case_id} {run.variant}")
    return results


def _num(value: float | None) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.6f}"


def _pct(value: float | None) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.2f}"


def _rows_from_results(
    planned: list[PlannedRun],
    results: Mapping[int, RunResult],
    derates: Mapping[str, Any],
) -> list[dict[str, str]]:
    baseline_by_case: dict[str, float | None] = {}
    for run in planned:
        if run.variant == "baseline":
            result = results[run.order]
            baseline_by_case[run.case.case_id] = result.predicted_runtime_s

    rows: list[dict[str, str]] = []
    launch_overhead_s = float(derates["launch_overhead_s"])
    for run in planned:
        result = results[run.order]
        baseline = derates[run.case.device_type]
        baseline_runtime = baseline_by_case.get(run.case.case_id)
        runtime = result.predicted_runtime_s
        signed_change: float | None = None
        abs_change: float | None = None
        elasticity: float | None = None
        if (
            runtime is not None
            and baseline_runtime is not None
            and baseline_runtime != 0
            and run.variant != "baseline"
        ):
            signed_change = (runtime - baseline_runtime) / baseline_runtime * 100.0
            abs_change = abs(signed_change)
            elasticity = abs_change / 10.0
        rows.append(
            {
                "case_id": run.case.case_id,
                "variant": run.variant,
                "status": result.status,
                "regime": run.case.regime,
                "source_csv": _rel(run.case.source_csv),
                "model": run.case.model,
                "training_or_inference": run.case.training_or_inference,
                "gpu_type": run.case.gpu_type,
                "num_gpus": str(run.case.num_gpus),
                "parallelism": run.case.parallelism,
                "baseline_compute_factor": _num(float(baseline["compute"])),
                "baseline_memory_factor": _num(float(baseline["memory"])),
                "baseline_communication_factor": _num(float(baseline["communication"])),
                "launch_overhead_s": f"{launch_overhead_s:.8g}",
                "perturbed_factor": run.perturbed_factor,
                "perturbation_multiplier": _num(run.perturbation_multiplier),
                "compute_factor": _num(run.factors["compute"]),
                "memory_factor": _num(run.factors["memory"]),
                "communication_factor": _num(run.factors["communication"]),
                "predicted_runtime_s": _num(runtime),
                "baseline_runtime_s": _num(baseline_runtime),
                "signed_runtime_change_percent": _pct(signed_change),
                "absolute_runtime_change_percent": _pct(abs_change),
                "elasticity": _num(elasticity),
                "hardware_config": _rel(run.hardware_path),
                "model_config": _rel(run.model_path),
                "command": run.command,
                "error": result.error,
            }
        )
    return rows


def _write_csv(rows: list[dict[str, str]]) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_commands(planned: list[PlannedRun]) -> None:
    COMMANDS_TXT.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{run.case.case_id},{run.variant}: {run.command}" for run in planned]
    COMMANDS_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_runtime(value: str) -> str:
    return f"{float(value):.2f}s" if value else "FAILED"


def _fmt_change(value: str) -> str:
    if not value:
        return "FAILED"
    number = float(value)
    return f"{number:+.2f}%"


def _range_by_factor(rows: list[dict[str, str]]) -> dict[str, tuple[float, float] | None]:
    ranges: dict[str, tuple[float, float] | None] = {}
    for factor in ("compute", "memory", "communication"):
        values = [
            float(row["absolute_runtime_change_percent"])
            for row in rows
            if row["status"] == "success"
            and row["perturbed_factor"] == factor
            and row["absolute_runtime_change_percent"]
        ]
        ranges[factor] = (min(values), max(values)) if values else None
    return ranges


def _range_text(value: tuple[float, float] | None) -> str:
    if value is None:
        return "no successful perturbations"
    return f"{value[0]:.2f}-{value[1]:.2f}%"


def _find_row(rows: list[dict[str, str]], case_id: str, variant: str) -> dict[str, str] | None:
    for row in rows:
        if row["case_id"] == case_id and row["variant"] == variant:
            return row
    return None


def _verification(rows: list[dict[str, str]], planned: list[PlannedRun]) -> dict[str, bool]:
    line_count_ok = len(rows) + 1 == 29
    per_case_ok = True
    baseline_ok = True
    for case_id in sorted({row["case_id"] for row in rows}):
        case_rows = [row for row in rows if row["case_id"] == case_id]
        per_case_ok = per_case_ok and len(case_rows) == 7
        baseline_ok = baseline_ok and sum(row["variant"] == "baseline" for row in case_rows) == 1
    no_fitted = all(
        ".fitted.yaml" not in row["hardware_config"] and ".fitted.yaml" not in row["command"]
        for row in rows
    )
    launch_ok = True
    for run in planned:
        hw = _load_yaml(run.hardware_path)
        launch_ok = launch_ok and float(hw["sw_param"]["kernel_launch_overhead"]) == 6e-6
    case_ids = {run.case.case_id for run in planned}
    md_text = SUMMARY_MD.read_text(encoding="utf-8") if SUMMARY_MD.exists() else ""
    md_case_ids = all(case_id in md_text for case_id in case_ids)
    md_failures = "## Failed Runs" in md_text
    return {
        "csv_has_29_lines_including_header": line_count_ok,
        "each_case_has_7_rows": per_case_ok,
        "each_case_has_one_baseline": baseline_ok,
        "no_fitted_yaml_referenced": no_fitted,
        "launch_overhead_fixed_6e-6": launch_ok,
        "markdown_includes_all_case_ids": md_case_ids,
        "markdown_reports_failures": md_failures,
    }


def _write_summary(rows: list[dict[str, str]], derates: Mapping[str, Any]) -> None:
    case_ids = [
        "large_train_gpt1t_korthi_full",
        "decode_inf_llama2_70b_tp8",
        "comm_inf_llama2_7b_tp8",
        "pcie_train_uci_ddp_4gpu",
    ]
    ranges = _range_by_factor(rows)
    failures = [row for row in rows if row["status"] != "success"]
    sentence = (
        "To clarify that the calibrated constants are not fragile per-workload knobs, "
        "we performed a bounded one-at-a-time sensitivity check on four already-submitted "
        "validation configurations spanning large-scale training, decode-heavy inference, "
        "high-TP inference, and PCIe/DDP training. Perturbing each Table III factor by "
        f"+/-10% changed predicted end-to-end runtime by {_range_text(ranges['compute'])} "
        f"for compute, {_range_text(ranges['memory'])} for memory, and "
        f"{_range_text(ranges['communication'])} for communication, with all four cases reported."
    )

    lines = [
        "# Bounded Table III Factor Sensitivity",
        "",
        "This is a model-output sensitivity characterization of RAPID-LLM using already-submitted validation configurations. It does not add new measured ground truth, new benchmarks, new workloads, or recalibration.",
        "",
        "## Fixed Selection Rule",
        "",
        "The case set was fixed before inspecting perturbation results and all four cases are reported:",
        "",
    ]
    for case_id in case_ids:
        row = _find_row(rows, case_id, "baseline")
        if row is None:
            lines.append(f"- `{case_id}`")
        else:
            lines.append(
                f"- `{case_id}`: {row['regime']}; {row['model']}; {row['gpu_type']}; "
                f"{row['num_gpus']} GPUs; {row['parallelism']}"
            )
    lines += [
        "",
        "## Frozen Table III Factors",
        "",
        f"- A100 SXM4: compute `{derates['A100_SXM4']['compute']:.2f}`, memory `{derates['A100_SXM4']['memory']:.2f}`, communication `{derates['A100_SXM4']['communication']:.2f}`",
        f"- A100 PCIe: compute `{derates['A100_PCIe']['compute']:.2f}`, memory `{derates['A100_PCIe']['memory']:.2f}`, communication `{derates['A100_PCIe']['communication']:.2f}`",
        f"- Launch overhead fixed at `{float(derates['launch_overhead_s']):.8g}` seconds",
        "- No `.fitted.yaml` hardware config was used.",
        "",
        "## Per-Case Sensitivity",
        "",
        "| case_id | baseline runtime | compute -10% / +10% | memory -10% / +10% | communication -10% / +10% | max abs change |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case_id in case_ids:
        baseline = _find_row(rows, case_id, "baseline")
        values = {
            variant: _find_row(rows, case_id, variant)
            for variant in (
                "compute_m10",
                "compute_p10",
                "memory_m10",
                "memory_p10",
                "communication_m10",
                "communication_p10",
            )
        }
        abs_values = [
            float(row["absolute_runtime_change_percent"])
            for row in rows
            if row["case_id"] == case_id and row["absolute_runtime_change_percent"]
        ]
        max_abs = f"{max(abs_values):.2f}%" if abs_values else "FAILED"
        lines.append(
            "| `{}` | {} | {} / {} | {} / {} | {} / {} | {} |".format(
                case_id,
                _fmt_runtime(baseline["predicted_runtime_s"] if baseline else ""),
                _fmt_change(values["compute_m10"]["signed_runtime_change_percent"] if values["compute_m10"] else ""),
                _fmt_change(values["compute_p10"]["signed_runtime_change_percent"] if values["compute_p10"] else ""),
                _fmt_change(values["memory_m10"]["signed_runtime_change_percent"] if values["memory_m10"] else ""),
                _fmt_change(values["memory_p10"]["signed_runtime_change_percent"] if values["memory_p10"] else ""),
                _fmt_change(values["communication_m10"]["signed_runtime_change_percent"] if values["communication_m10"] else ""),
                _fmt_change(values["communication_p10"]["signed_runtime_change_percent"] if values["communication_p10"] else ""),
                max_abs,
            )
        )

    lines += [
        "",
        "## Observed Ranges",
        "",
        f"- Compute factor +/-10%: {_range_text(ranges['compute'])} absolute runtime change across successful perturbations.",
        f"- Memory factor +/-10%: {_range_text(ranges['memory'])} absolute runtime change across successful perturbations.",
        f"- Communication factor +/-10%: {_range_text(ranges['communication'])} absolute runtime change across successful perturbations.",
        "",
        "## Failed Runs",
        "",
    ]
    if failures:
        lines += ["| case_id | variant | error |", "|---|---|---|"]
        for row in failures:
            lines.append(f"| `{row['case_id']}` | `{row['variant']}` | {row['error']} |")
    else:
        lines.append("None.")

    lines += [
        "",
        "## Rebuttal-Ready Sentence",
        "",
        sentence,
        "",
        "## Output Files",
        "",
        f"- CSV: `{_rel(RESULTS_CSV)}`",
        f"- Commands: `{_rel(COMMANDS_TXT)}`",
        f"- Generated configs: `{_rel(CASE_ROOT)}`",
        "",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    verification = _verification(rows, [run for run in _LAST_PLANNED_RUNS])
    with SUMMARY_MD.open("a", encoding="utf-8") as handle:
        handle.write("## Verification\n\n")
        for name, ok in verification.items():
            handle.write(f"- `{name}`: {'PASS' if ok else 'FAIL'}\n")


_LAST_PLANNED_RUNS: list[PlannedRun] = []


def _print_plan(planned: list[PlannedRun]) -> None:
    print(f"Planned runs: {len(planned)}")
    for run in planned:
        print(f"{run.order:02d}: {run.case.case_id} {run.variant}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4, help="Parallel RAPID-LLM runs.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Materialize configs and commands, but do not execute RAPID-LLM.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not PYTHON.exists():
        raise FileNotFoundError(f"Expected repo Python at {PYTHON}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    mpl_dir = OUT_ROOT / ".matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    derates = _load_derates()
    cases = _build_cases()
    planned = _plan_runs(cases, derates)
    global _LAST_PLANNED_RUNS
    _LAST_PLANNED_RUNS = planned
    _write_commands(planned)
    _print_plan(planned)

    if args.dry_run:
        print(f"Wrote commands: {_rel(COMMANDS_TXT)}")
        return 0

    results = _run_all(planned, args.workers)
    rows = _rows_from_results(planned, results, derates)
    _write_csv(rows)
    _write_summary(rows, derates)
    print(f"Wrote CSV: {_rel(RESULTS_CSV)}")
    print(f"Wrote summary: {_rel(SUMMARY_MD)}")
    print(f"Wrote commands: {_rel(COMMANDS_TXT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
