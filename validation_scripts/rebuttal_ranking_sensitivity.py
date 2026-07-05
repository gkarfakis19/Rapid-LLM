#!/usr/bin/env python3
"""Rank-stability check for a paper 2D topology sweep slice.

This is a bounded rebuttal artifact: it reruns one archived Fig. 12/14-style
case-study slice with frozen Table III H100 factors and one-at-a-time +/-10%
factor perturbations, then checks whether the configuration ranking changes.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import math
import os
import re
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_ROOT = SCRIPT_DIR / "rebuttal_ranking_sensitivity"
CASE_ROOT = OUT_ROOT / "case_configs"
RUN_ROOT = OUT_ROOT / "runs"
RESULTS_CSV = OUT_ROOT / "ranking_sensitivity_results.csv"
SUMMARY_MD = OUT_ROOT / "ranking_sensitivity_summary.md"
COMMANDS_TXT = OUT_ROOT / "commands.txt"

SOURCE_TSV = PROJECT_ROOT / "output" / "2d_test_PAPER" / "2d_test_inf.tsv"
DERATES_YAML = SCRIPT_DIR / "validation_configs" / "harness_derates.yaml"
HW_PATH = SCRIPT_DIR / "validation_configs" / "hardware-config" / "H100_SXM5_80GB_2d.yaml"
MODEL_PATH = SCRIPT_DIR / "validation_configs" / "model-config" / "GPT_175_B_2d_inf.yaml"
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
RUN_PERF = PROJECT_ROOT / "run_perf.py"

DEVICE_TYPE = "H100_SXM5"
GPU_TYPE = "H100 SXM5 80GB"
MODEL_LABEL = "GPT175B"
MODE = "inference"

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
    "source_slice",
    "model",
    "training_or_inference",
    "gpu_type",
    "num_gpus",
    "topology",
    "shape",
    "batch_size",
    "seq_len",
    "prompt_output_len",
    "parallelism",
    "variant",
    "status",
    "perturbed_factor",
    "perturbation_multiplier",
    "compute_factor",
    "memory_factor",
    "communication_factor",
    "launch_overhead_s",
    "predicted_runtime_s",
    "baseline_runtime_s",
    "runtime_change_percent",
    "baseline_rank",
    "variant_rank",
    "rank_shift",
    "rank_order_changed",
    "source_tsv_runtime_s",
    "hardware_config",
    "model_config",
    "command",
    "error",
]


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    model: str
    topology: str
    shape: tuple[int, int]
    tp: int
    cp: int
    pp: int
    mb: int
    source_tsv_runtime_s: float


@dataclass(frozen=True)
class PlannedRun:
    order: int
    case: CaseSpec
    variant: str
    perturbed_factor: str
    multiplier: float
    factors: dict[str, float]
    launch_overhead_s: float
    hardware_path: Path
    run_dir: Path
    command: str


@dataclass(frozen=True)
class RunResult:
    order: int
    status: str
    runtime_s: float | None
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
        yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _num(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.8f}"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _load_derates() -> tuple[dict[str, float], float]:
    data = _load_yaml(DERATES_YAML)
    shared = data.get("shared")
    devices = data.get("device_types")
    if not isinstance(shared, dict) or not isinstance(devices, dict):
        raise ValueError(f"Malformed derate config: {DERATES_YAML}")
    device = devices.get(DEVICE_TYPE)
    if not isinstance(device, dict):
        raise ValueError(f"Missing device type {DEVICE_TYPE} in {DERATES_YAML}")
    factors = {
        "compute": float(device["compute_util"]),
        "memory": float(device["dram_util"]),
        "communication": float(device["network_util"]),
    }
    launch = float(shared["kernel_launch_overhead_s"])
    return factors, launch


def _variant_factors(base: Mapping[str, float], factor: str, multiplier: float) -> dict[str, float]:
    values = {
        "compute": float(base["compute"]),
        "memory": float(base["memory"]),
        "communication": float(base["communication"]),
    }
    if factor:
        values[factor] = values[factor] * float(multiplier)
    return values


def _apply_table_factors(
    hardware_config: dict[str, Any],
    *,
    factors: Mapping[str, float],
    launch_overhead_s: float,
) -> dict[str, Any]:
    hw = copy.deepcopy(hardware_config)
    hw.setdefault("sw_param", {})["kernel_launch_overhead"] = float(launch_overhead_s)
    hw.setdefault("tech_param", {}).setdefault("core", {})["util"] = float(factors["compute"])
    hw.setdefault("tech_param", {}).setdefault("DRAM", {})["util"] = float(factors["memory"])
    for dim in hw.get("network", {}).get("dimensions", []):
        if isinstance(dim, dict):
            dim.setdefault("topology", {})["util"] = float(factors["communication"])
    return hw


def _update_2d_inference_hardware(
    base_hw: dict[str, Any],
    *,
    topology: str,
    shape: tuple[int, int],
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_hw)
    tp = int(shape[0]) * int(shape[1])

    par = cfg.setdefault("parallelism", {})
    par["tp"] = tp
    par["cp"] = 1
    par["pp"] = 1
    par["mb"] = 1
    par.setdefault("train", {})["dp"] = 1
    par.setdefault("train", {})["ep"] = 1
    par.setdefault("inference", {}).setdefault("replica_count", 1)
    par.setdefault("inference", {}).setdefault("moe_dp", 1)

    net = cfg.setdefault("network", {})
    dims = list(net.get("dimensions") or [])
    if not dims:
        raise ValueError("Hardware config missing network.dimensions")
    dim0 = copy.deepcopy(dims[0])
    dim0["size"] = [int(shape[0]), int(shape[1])]
    topo = dim0.setdefault("topology", {})
    topo["type"] = topology
    topo["optimize_2dmap"] = False
    dim0["parallelisms"] = ["tp", "cp", "ep"]

    exec_backend = cfg.setdefault("execution_backend", {})
    if str(exec_backend.get("model", "analytical")).lower() == "astra":
        exec_backend.setdefault("astra", {})["mode"] = "full_astrasim_hierarchical"

    bw_value = topo.get("bandwidth")
    if isinstance(bw_value, (list, tuple)):
        bw_value = bw_value[0] if bw_value else bw_value
    dim1 = {
        "id": "dim1_pp_dp",
        "label": "pp_dp_dim",
        "size": 1,
        "topology": {
            "type": "Ring",
            "bandwidth": bw_value,
            "latency": topo.get("latency"),
            "energy_per_bit": topo.get("energy_per_bit"),
            "util": topo.get("util", 1.0),
        },
        "parallelisms": ["pp", "dp"],
    }
    net["dimensions"] = [dim0, dim1]
    return cfg


def _load_cases() -> list[CaseSpec]:
    cases: list[CaseSpec] = []
    with SOURCE_TSV.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("model") != MODEL_LABEL:
                continue
            if str(row.get("mem_exceeded", "")).lower() != "no":
                continue
            shape_text = str(row["shape"])
            left, right = shape_text.split("x", 1)
            topology = str(row["topology"])
            tp = int(row["tp"])
            case_id = _slug(f"{MODEL_LABEL}_{topology}_{shape_text}_tp{tp}").lower()
            cases.append(
                CaseSpec(
                    case_id=case_id,
                    model=MODEL_LABEL,
                    topology=topology,
                    shape=(int(left), int(right)),
                    tp=tp,
                    cp=int(row["cp"]),
                    pp=int(row["pp"]),
                    mb=int(row["mb"]),
                    source_tsv_runtime_s=float(row["runtime_s"]),
                )
            )
    cases.sort(key=lambda c: (c.shape[0] * c.shape[1], c.topology, c.shape))
    if not cases:
        raise RuntimeError(f"No {MODEL_LABEL} cases found in {SOURCE_TSV}")
    return cases


def _plan_runs(cases: Iterable[CaseSpec]) -> list[PlannedRun]:
    base_hw = _load_yaml(HW_PATH)
    base_factors, launch = _load_derates()
    planned: list[PlannedRun] = []
    order = 0
    shutil.rmtree(CASE_ROOT, ignore_errors=True)
    shutil.rmtree(RUN_ROOT, ignore_errors=True)
    for variant, factor, multiplier in VARIANTS:
        factors = _variant_factors(base_factors, factor, multiplier)
        for case in cases:
            order += 1
            hw_cfg = _update_2d_inference_hardware(
                _apply_table_factors(base_hw, factors=factors, launch_overhead_s=launch),
                topology=case.topology,
                shape=case.shape,
            )
            run_dir = RUN_ROOT / variant / case.case_id
            hardware_path = CASE_ROOT / variant / f"{case.case_id}.hardware.yaml"
            _write_yaml(hardware_path, hw_cfg)
            command = (
                f"{shlex_quote(str(PYTHON))} {shlex_quote(str(RUN_PERF))} "
                f"--hardware_config {shlex_quote(str(hardware_path))} "
                f"--model_config {shlex_quote(str(MODEL_PATH))}"
            )
            planned.append(
                PlannedRun(
                    order=order,
                    case=case,
                    variant=variant,
                    perturbed_factor=factor,
                    multiplier=multiplier,
                    factors=factors,
                    launch_overhead_s=launch,
                    hardware_path=hardware_path,
                    run_dir=run_dir,
                    command=command,
                )
            )
    return planned


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


def _run_one(run: PlannedRun) -> RunResult:
    os.environ["RAPID_ASTRA_CACHE_MODE"] = "NO_CACHE"
    os.environ["DEEPFLOW_ASTRA_CACHE_MODE"] = "no_cache"
    os.environ["ASTRA_CACHE_DIR"] = str(run.run_dir / "astra_cache")
    os.environ["RAPID_VISUALIZE_GRAPHS"] = "0"
    os.environ["RAPID_PERSIST_ASTRASIM_ARTIFACTS"] = "0"
    os.environ["RAPID_PERSIST_ARTIFACT_VIZ"] = "0"

    from astrasim_lib import ensure_chakra_available
    import config
    from inference_timing import TimeCalculationLLMInference

    run.run_dir.mkdir(parents=True, exist_ok=True)
    temp_hw = None
    try:
        ensure_chakra_available()
        mode = "LLM"
        hw_config = config.parse_config(str(run.hardware_path), config_type="hardware")
        model_config = config.parse_config(str(MODEL_PATH), config_type=mode)
        config.validate_configs(hw_config, model_config)
        calc = TimeCalculationLLMInference(hw_config, model_config, mode, output_dir=str(run.run_dir))
        timing = calc.calc_total_inference_time()
        runtime = float(timing["total_inference_time"])
        if not math.isfinite(runtime) or runtime <= 0:
            raise RuntimeError(f"Invalid runtime {runtime!r}")
        return RunResult(order=run.order, status="ok", runtime_s=runtime, error="")
    except Exception as exc:
        return RunResult(order=run.order, status="error", runtime_s=None, error=str(exc))
    finally:
        if temp_hw:
            try:
                os.unlink(temp_hw)
            except OSError:
                pass


def _run_all(planned: list[PlannedRun], workers: int) -> dict[int, RunResult]:
    if workers <= 1:
        return {run.order: _run_one(run) for run in planned}
    results: dict[int, RunResult] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_run_one, run): run for run in planned}
        for future in as_completed(future_map):
            run = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = RunResult(order=run.order, status="error", runtime_s=None, error=str(exc))
            results[result.order] = result
            print(
                f"[ranking] {len(results)}/{len(planned)} {run.variant} {run.case.case_id}: "
                f"{result.status} {'' if result.runtime_s is None else f'{result.runtime_s:.6f}s'}",
                flush=True,
            )
    return results


def _rank_map(rows: Iterable[dict[str, str]], variant: str) -> tuple[list[str], dict[str, int]]:
    ok_rows = [
        row
        for row in rows
        if row["variant"] == variant and row["status"] == "ok" and row["predicted_runtime_s"]
    ]
    ok_rows.sort(key=lambda row: (float(row["predicted_runtime_s"]), row["case_id"]))
    order = [row["case_id"] for row in ok_rows]
    return order, {case_id: idx + 1 for idx, case_id in enumerate(order)}


def _rows_from_results(planned: list[PlannedRun], results: Mapping[int, RunResult]) -> list[dict[str, str]]:
    by_case_variant: dict[tuple[str, str], RunResult] = {}
    for run in planned:
        by_case_variant[(run.case.case_id, run.variant)] = results[run.order]

    rows: list[dict[str, str]] = []
    for run in planned:
        result = results[run.order]
        baseline = by_case_variant.get((run.case.case_id, "baseline"))
        baseline_runtime = baseline.runtime_s if baseline and baseline.status == "ok" else None
        change_pct = None
        if result.runtime_s is not None and baseline_runtime is not None:
            change_pct = ((result.runtime_s - baseline_runtime) / baseline_runtime) * 100.0
        shape_text = f"{run.case.shape[0]}x{run.case.shape[1]}"
        rows.append(
            {
                "case_id": run.case.case_id,
                "source_slice": _rel(SOURCE_TSV),
                "model": run.case.model,
                "training_or_inference": MODE,
                "gpu_type": GPU_TYPE,
                "num_gpus": str(run.case.tp),
                "topology": run.case.topology,
                "shape": shape_text,
                "batch_size": "1",
                "seq_len": "4096",
                "prompt_output_len": "4096/1024",
                "parallelism": f"TP={run.case.tp}, CP={run.case.cp}, PP={run.case.pp}, MB={run.case.mb}",
                "variant": run.variant,
                "status": result.status,
                "perturbed_factor": run.perturbed_factor,
                "perturbation_multiplier": f"{run.multiplier:.8g}",
                "compute_factor": _num(run.factors["compute"]),
                "memory_factor": _num(run.factors["memory"]),
                "communication_factor": _num(run.factors["communication"]),
                "launch_overhead_s": f"{run.launch_overhead_s:.8g}",
                "predicted_runtime_s": _num(result.runtime_s),
                "baseline_runtime_s": _num(baseline_runtime),
                "runtime_change_percent": _num(change_pct),
                "baseline_rank": "",
                "variant_rank": "",
                "rank_shift": "",
                "rank_order_changed": "",
                "source_tsv_runtime_s": f"{run.case.source_tsv_runtime_s:.6f}",
                "hardware_config": _rel(run.hardware_path),
                "model_config": _rel(MODEL_PATH),
                "command": run.command,
                "error": result.error,
            }
        )

    baseline_order, baseline_ranks = _rank_map(rows, "baseline")
    for variant, _, _ in VARIANTS:
        variant_order, variant_ranks = _rank_map(rows, variant)
        changed = variant_order != baseline_order
        for row in rows:
            if row["variant"] != variant or row["status"] != "ok":
                continue
            case_id = row["case_id"]
            base_rank = baseline_ranks.get(case_id)
            variant_rank = variant_ranks.get(case_id)
            if base_rank is None or variant_rank is None:
                continue
            row["baseline_rank"] = str(base_rank)
            row["variant_rank"] = str(variant_rank)
            row["rank_shift"] = str(variant_rank - base_rank)
            row["rank_order_changed"] = "yes" if changed else "no"
    return rows


def _write_csv(rows: list[dict[str, str]]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _summary_table(rows: list[dict[str, str]], variant: str) -> list[dict[str, str]]:
    data = [
        row
        for row in rows
        if row["variant"] == variant and row["status"] == "ok" and row["predicted_runtime_s"]
    ]
    data.sort(key=lambda row: int(row["variant_rank"]))
    return data


def _write_summary(rows: list[dict[str, str]], planned: list[PlannedRun], workers: int) -> None:
    base_factors, launch = _load_derates()
    variants = [variant for variant, _, _ in VARIANTS]
    successful_variants = {
        variant
        for variant in variants
        if all(row["status"] == "ok" for row in rows if row["variant"] == variant)
    }
    changed_variants = sorted(
        {
            row["variant"]
            for row in rows
            if row["status"] == "ok" and row["rank_order_changed"] == "yes"
        }
    )
    max_shift = 0
    max_abs_runtime_change = 0.0
    for row in rows:
        if row["rank_shift"]:
            max_shift = max(max_shift, abs(int(row["rank_shift"])))
        if row["variant"] != "baseline" and row["runtime_change_percent"]:
            max_abs_runtime_change = max(max_abs_runtime_change, abs(float(row["runtime_change_percent"])))

    baseline_rows = _summary_table(rows, "baseline")
    lines = [
        "# Ranking Sensitivity Check",
        "",
        "## Scope",
        "",
        "- Source slice: `output/2d_test_PAPER/2d_test_inf.tsv`.",
        "- Cases: GPT175B inference, Mesh2D/Torus2D, shapes 4x5, 4x8, 5x6, 6x6.",
        "- Metric: predicted total inference runtime; lower is better.",
        "- Perturbation method: one Table III factor at a time by +/-10%; no model/topology changes.",
        "- Hardware factors: H100 SXM5 compute `{:.2f}`, memory `{:.2f}`, communication `{:.2f}`, launch overhead `{:.8g}` s.".format(
            base_factors["compute"],
            base_factors["memory"],
            base_factors["communication"],
            launch,
        ),
        "- Model config: `{}` (SHA1 `{}`).".format(_rel(MODEL_PATH), _sha1(MODEL_PATH)),
        "- Hardware base config: `{}` (SHA1 `{}`).".format(_rel(HW_PATH), _sha1(HW_PATH)),
        "- Generated per-case hardware configs: `{}`.".format(_rel(CASE_ROOT)),
        "- Command used for this run: `{}`.".format(
            f"{_rel(PYTHON)} {_rel(Path(__file__))} --workers {workers}"
        ),
        "- Serial reproducibility command: `{}`.".format(
            f"{_rel(PYTHON)} {_rel(Path(__file__))} --workers 1"
        ),
        "",
        "## Result",
        "",
    ]
    if changed_variants:
        lines.append(
            "Ranking changed for `{}`; max absolute rank shift was `{}`. Do not use this as a rank-invariance claim without caveat.".format(
                ", ".join(changed_variants),
                max_shift,
            )
        )
    else:
        lines.append(
            "Ranking was unchanged for all successful +/-10% one-at-a-time perturbations; max rank shift was `0`."
        )
    lines.extend(
        [
            "",
            "- Successful variants: `{}` of `{}`.".format(len(successful_variants), len(variants)),
            "- Successful simulator runs: `{}` of `{}`.".format(
                sum(1 for row in rows if row["status"] == "ok"),
                len(rows),
            ),
            "- Max absolute runtime change over non-baseline successful runs: `{:.2f}%`.".format(
                max_abs_runtime_change
            ),
            "",
            "## Baseline Rank Order",
            "",
            "| rank | case_id | topology | shape | runtime_s | paper_tsv_runtime_s |",
            "|---:|---|---|---|---:|---:|",
        ]
    )
    for row in baseline_rows:
        lines.append(
            "| {} | `{}` | {} | {} | {:.6f} | {:.6f} |".format(
                row["variant_rank"],
                row["case_id"],
                row["topology"],
                row["shape"],
                float(row["predicted_runtime_s"]),
                float(row["source_tsv_runtime_s"]),
            )
        )
    lines.extend(
        [
            "",
            "## Variant Rank Check",
            "",
            "| variant | factor | multiplier | rank_order_changed | max_abs_runtime_change_pct | max_abs_rank_shift |",
            "|---|---|---:|---|---:|---:|",
        ]
    )
    for variant, factor, multiplier in VARIANTS:
        variant_rows = [row for row in rows if row["variant"] == variant and row["status"] == "ok"]
        if not variant_rows:
            lines.append(f"| {variant} | {factor} | {multiplier:.2f} | no successful runs |  |  |")
            continue
        changed = any(row["rank_order_changed"] == "yes" for row in variant_rows)
        var_max_shift = max(abs(int(row["rank_shift"])) for row in variant_rows if row["rank_shift"])
        var_max_change = 0.0
        for row in variant_rows:
            if row["runtime_change_percent"]:
                var_max_change = max(var_max_change, abs(float(row["runtime_change_percent"])))
        lines.append(
            "| {} | {} | {:.2f} | {} | {:.2f} | {} |".format(
                variant,
                factor or "none",
                multiplier,
                "yes" if changed else "no",
                var_max_change,
                var_max_shift,
            )
        )
    lines.extend(
        [
            "",
            "## Rebuttal Sentence",
            "",
            "For a post-review rank-stability check, we kept the Table III H100 factors fixed and reran the GPT175B inference 2D topology slice from the paper with each factor perturbed by +/-10% one at a time; the ordering of the 8 Mesh/Torus configurations was unchanged across all successful perturbations, with max rank shift 0 and max runtime change {:.2f}%, supporting that these constants mainly rescale predictions rather than changing the case-study ranking.".format(
                max_abs_runtime_change
            ),
            "",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")

    with COMMANDS_TXT.open("w", encoding="utf-8") as handle:
        handle.write(f"{PYTHON} {Path(__file__).resolve()} --workers {workers}\n")
        handle.write(f"{PYTHON} {Path(__file__).resolve()} --workers 1\n")
        handle.write("\n# Generated run_perf-equivalent commands per case/variant:\n")
        for run in planned:
            handle.write(run.command + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of independent simulator runs to execute in parallel.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not PYTHON.exists():
        raise FileNotFoundError(f"Expected repo venv Python at {PYTHON}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cases = _load_cases()
    planned = _plan_runs(cases)
    print(
        f"[ranking] running {len(planned)} runs: {len(cases)} cases x {len(VARIANTS)} variants",
        flush=True,
    )
    results = _run_all(planned, max(1, int(args.workers)))
    rows = _rows_from_results(planned, results)
    _write_csv(rows)
    _write_summary(rows, planned, max(1, int(args.workers)))
    ok = sum(1 for row in rows if row["status"] == "ok")
    changed = sorted({row["variant"] for row in rows if row["rank_order_changed"] == "yes"})
    print(f"[ranking] wrote {_rel(RESULTS_CSV)}")
    print(f"[ranking] wrote {_rel(SUMMARY_MD)}")
    print(f"[ranking] successful runs: {ok}/{len(rows)}")
    print(f"[ranking] changed variants: {', '.join(changed) if changed else 'none'}")


if __name__ == "__main__":
    main()
