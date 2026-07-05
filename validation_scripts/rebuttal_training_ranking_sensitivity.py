#!/usr/bin/env python3
"""Training rank-stability check for a near-tie 96-GPU paper sweep slice.

This rebuttal artifact reruns the archived Fig. 12/fault-sweep-style 96-GPU
training slice with fixed Table III A100 SXM4 factors, then perturbs only the
compute factor by +/-10%. It is intentionally bounded: no recalibration, no
new measured ground truth, and all configurations from the archived slice are
included.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_ROOT = SCRIPT_DIR / "rebuttal_training_ranking_sensitivity"
CASE_ROOT = OUT_ROOT / "case_configs"
RESULTS_CSV = OUT_ROOT / "training_ranking_sensitivity_results.csv"
SUMMARY_MD = OUT_ROOT / "training_ranking_sensitivity_summary.md"
COMMANDS_TXT = OUT_ROOT / "commands.txt"

SOURCE_TSV = PROJECT_ROOT / "outputs" / "fault_PAPER" / "fault_sweep_PAPER.tsv"
BASE_HW_PATH = SCRIPT_DIR / "validation_configs" / "hardware-config" / "a100_80GB_fault.yaml"
MODEL_PATH = PROJECT_ROOT / "configs" / "model-config" / "Llama3.1-70B.yaml"
DERATES_YAML = SCRIPT_DIR / "validation_configs" / "harness_derates.yaml"
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
RUN_PERF = PROJECT_ROOT / "run_perf.py"

DEVICE_TYPE = "A100_SXM4"
GPU_TYPE = "A100 SXM4 80GB"
MODEL_LABEL = "Llama3.1-70B"
MODE = "training"

TRAIN_RE = re.compile(r"Training time for batch:\s*([0-9]+(?:\.[0-9]+)?)s")

VARIANTS = [
    ("baseline", "", 1.0),
    ("compute_m10", "compute", 0.9),
    ("compute_p10", "compute", 1.1),
]

CSV_COLUMNS = [
    "case_id",
    "source_slice",
    "model",
    "training_or_inference",
    "gpu_type",
    "num_gpus",
    "global_batch_size",
    "seq_len",
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
    "source_tsv_runtime_s",
    "runtime_change_percent",
    "elasticity_abs",
    "baseline_rank",
    "variant_rank",
    "rank_shift",
    "rank_order_changed",
    "hardware_config",
    "model_config",
    "command",
    "error",
]

NEAR_TIE_A = "tp12-cp2-dp4-pp1"
NEAR_TIE_B = "tp4-cp6-dp1-pp4"


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    tp: int
    cp: int
    ep: int
    dp: int
    pp: int
    mb: int
    source_tsv_runtime_s: float

    @property
    def num_gpus(self) -> int:
        return self.tp * self.cp * self.ep * self.dp * self.pp

    @property
    def parallelism_label(self) -> str:
        return f"TP={self.tp}, CP={self.cp}, EP={self.ep}, DP={self.dp}, PP={self.pp}, MB={self.mb}, TP_SP=True"


@dataclass(frozen=True)
class PlannedRun:
    order: int
    case: CaseSpec
    variant: str
    perturbed_factor: str
    multiplier: float
    factors: dict[str, float]
    launch_overhead_s: float
    run_dir: Path
    hardware_path: Path
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
        yaml.safe_dump(dict(data), handle, default_flow_style=False, sort_keys=False)


def _num(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.8f}"


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
    return factors, float(shared["kernel_launch_overhead_s"])


def _variant_factors(base: Mapping[str, float], factor: str, multiplier: float) -> dict[str, float]:
    values = {
        "compute": float(base["compute"]),
        "memory": float(base["memory"]),
        "communication": float(base["communication"]),
    }
    if factor:
        values[factor] = values[factor] * float(multiplier)
    return values


def _choose_2d_shape(total: int) -> tuple[int, int]:
    root = int(math.isqrt(max(1, total)))
    for factor in range(root, 0, -1):
        if total % factor == 0:
            return factor, total // factor
    return 1, total


def _apply_torus2d_rect_shape(hw: dict[str, Any]) -> None:
    network = hw.get("network")
    if not isinstance(network, dict):
        return
    dims = network.get("dimensions")
    if not isinstance(dims, list):
        return
    parallelism = hw.get("parallelism")
    if not isinstance(parallelism, dict):
        return
    tp = int(parallelism.get("tp", 1) or 1)
    cp = int(parallelism.get("cp", 1) or 1)
    train = parallelism.get("train") if isinstance(parallelism.get("train"), dict) else {}
    ep = int(train.get("ep", 1) or 1)
    total = tp * cp * ep
    for dim in dims:
        if not isinstance(dim, dict):
            continue
        topology = dim.get("topology")
        topo_type = ""
        if isinstance(topology, dict):
            topo_type = str(topology.get("type", "")).strip().lower().replace("_", "").replace("-", "")
        if topo_type == "torus2d":
            rows, cols = _choose_2d_shape(total)
            dim["size"] = [int(rows), int(cols)]


def _apply_case_and_factors(
    base_hw: Mapping[str, Any],
    case: CaseSpec,
    factors: Mapping[str, float],
    launch_overhead_s: float,
) -> dict[str, Any]:
    hw = copy.deepcopy(dict(base_hw))
    hw.setdefault("sw_param", {})["kernel_launch_overhead"] = float(launch_overhead_s)
    hw.setdefault("tech_param", {}).setdefault("core", {})["util"] = float(factors["compute"])
    hw.setdefault("tech_param", {}).setdefault("DRAM", {})["util"] = float(factors["memory"])

    parallelism = hw.setdefault("parallelism", {})
    parallelism["tp"] = int(case.tp)
    parallelism["cp"] = int(case.cp)
    parallelism["pp"] = int(case.pp)
    parallelism["mb"] = int(case.mb)
    parallelism["tp_sp"] = True
    train = parallelism.setdefault("train", {})
    train["dp"] = int(case.dp)
    train["ep"] = int(case.ep)
    train["tp_ep"] = True
    inference = parallelism.setdefault("inference", {})
    inference["replica_count"] = 1
    inference["moe_dp"] = 1

    for dim in hw.get("network", {}).get("dimensions", []):
        if isinstance(dim, dict):
            dim.setdefault("topology", {})["util"] = float(factors["communication"])
    _apply_torus2d_rect_shape(hw)
    return hw


def _read_cases() -> list[CaseSpec]:
    cases: list[CaseSpec] = []
    with SOURCE_TSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            tp = int(row["tp"])
            cp = int(row["cp"])
            ep = int(row["ep"])
            dp = int(row["dp"])
            pp = int(row["pp"])
            mb = 1 if pp == 1 else 2 * pp
            label = row.get("label") or f"tp{tp}-cp{cp}-dp{dp}-pp{pp}"
            cases.append(
                CaseSpec(
                    case_id=label,
                    tp=tp,
                    cp=cp,
                    ep=ep,
                    dp=dp,
                    pp=pp,
                    mb=mb,
                    source_tsv_runtime_s=float(row["baseline_runtime"]),
                )
            )
    if not cases:
        raise ValueError(f"No cases loaded from {SOURCE_TSV}")
    bad_gpu_counts = {case.case_id: case.num_gpus for case in cases if case.num_gpus != 96}
    if bad_gpu_counts:
        raise ValueError(f"Expected all source cases to use 96 GPUs, got {bad_gpu_counts}")
    return cases


def _format_command(run_dir: Path, hardware_path: Path) -> str:
    return (
        f"cd {shlex.quote(str(run_dir))} && "
        f"{shlex.quote(str(PYTHON))} {shlex.quote(str(RUN_PERF))} "
        f"--hardware_config {shlex.quote(str(hardware_path))} "
        f"--model_config {shlex.quote(str(MODEL_PATH))}"
    )


def _plan_runs(cases: Iterable[CaseSpec]) -> tuple[list[PlannedRun], dict[str, float], float]:
    CASE_ROOT.mkdir(parents=True, exist_ok=True)
    base_hw = _load_yaml(BASE_HW_PATH)
    base_factors, launch = _load_derates()
    planned: list[PlannedRun] = []
    order = 0
    for variant, factor, multiplier in VARIANTS:
        factors = _variant_factors(base_factors, factor, multiplier)
        for case in cases:
            run_dir = (CASE_ROOT / case.case_id / variant).resolve()
            hardware_path = run_dir / "hardware.yaml"
            hw_cfg = _apply_case_and_factors(base_hw, case, factors, launch)
            _write_yaml(hardware_path, hw_cfg)
            planned.append(
                PlannedRun(
                    order=order,
                    case=case,
                    variant=variant,
                    perturbed_factor=factor,
                    multiplier=float(multiplier),
                    factors=factors,
                    launch_overhead_s=launch,
                    run_dir=run_dir,
                    hardware_path=hardware_path,
                    command=_format_command(run_dir, hardware_path),
                )
            )
            order += 1
    return planned, base_factors, launch


def _execute_run(run: PlannedRun) -> RunResult:
    run.run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run.run_dir / "run.log"
    env = os.environ.copy()
    env["ASTRA_CACHE_DIR"] = str(run.run_dir / "astra_cache")
    env["RAPID_ASTRA_CACHE_MODE"] = "NO_CACHE"
    env["DEEPFLOW_ASTRA_CACHE_MODE"] = "no_cache"
    env["RAPID_VISUALIZE_GRAPHS"] = "0"
    env["RAPID_PERSIST_ASTRASIM_ARTIFACTS"] = "0"
    env["RAPID_PERSIST_ARTIFACT_VIZ"] = "0"
    cmd = [
        str(PYTHON),
        str(RUN_PERF),
        "--hardware_config",
        str(run.hardware_path),
        "--model_config",
        str(MODEL_PATH),
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
        return RunResult(run.order, "failed", None, f"return code {proc.returncode}; see {_rel(log_path)}")
    match = TRAIN_RE.search(output)
    if not match:
        return RunResult(run.order, "failed", None, f"could not parse training runtime; see {_rel(log_path)}")
    runtime = float(match.group(1))
    if not math.isfinite(runtime) or runtime <= 0:
        return RunResult(run.order, "failed", None, f"invalid runtime {runtime!r}; see {_rel(log_path)}")
    return RunResult(run.order, "ok", runtime, "")


def _run_all(planned: list[PlannedRun], workers: int) -> dict[int, RunResult]:
    if workers <= 1:
        return {run.order: _execute_run(run) for run in planned}
    results: dict[int, RunResult] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_execute_run, run): run for run in planned}
        for future in as_completed(future_map):
            run = future_map[future]
            try:
                results[run.order] = future.result()
            except Exception as exc:  # pragma: no cover - defensive
                results[run.order] = RunResult(run.order, "failed", None, str(exc))
    return results


def _rank_map(values: Mapping[str, float]) -> dict[str, int]:
    ordered = sorted(values, key=lambda case_id: (values[case_id], case_id))
    return {case_id: idx + 1 for idx, case_id in enumerate(ordered)}


def _rank_order(values: Mapping[str, float]) -> list[str]:
    return sorted(values, key=lambda case_id: (values[case_id], case_id))


def _rows_from_results(planned: list[PlannedRun], results: Mapping[int, RunResult]) -> list[dict[str, str]]:
    by_variant: dict[str, dict[str, float]] = {variant: {} for variant, _, _ in VARIANTS}
    for run in planned:
        result = results[run.order]
        if result.status == "ok" and result.runtime_s is not None:
            by_variant[run.variant][run.case.case_id] = result.runtime_s

    baseline_values = by_variant["baseline"]
    baseline_ranks = _rank_map(baseline_values) if len(baseline_values) == len(by_variant["baseline"]) else {}
    baseline_order = _rank_order(baseline_values) if baseline_values else []
    variant_ranks = {
        variant: _rank_map(values) if values else {}
        for variant, values in by_variant.items()
    }
    variant_orders = {variant: _rank_order(values) for variant, values in by_variant.items()}

    rows: list[dict[str, str]] = []
    for run in planned:
        result = results[run.order]
        baseline_runtime = baseline_values.get(run.case.case_id)
        runtime_change_pct = None
        elasticity_abs = None
        if result.status == "ok" and result.runtime_s is not None and baseline_runtime:
            runtime_change_pct = 100.0 * (result.runtime_s - baseline_runtime) / baseline_runtime
            if run.perturbed_factor:
                elasticity_abs = abs(runtime_change_pct) / abs((run.multiplier - 1.0) * 100.0)
        rank_changed = variant_orders.get(run.variant, []) != baseline_order
        baseline_rank = baseline_ranks.get(run.case.case_id)
        variant_rank = variant_ranks.get(run.variant, {}).get(run.case.case_id)
        rank_shift = None
        if baseline_rank is not None and variant_rank is not None:
            rank_shift = variant_rank - baseline_rank
        rows.append(
            {
                "case_id": run.case.case_id,
                "source_slice": _rel(SOURCE_TSV),
                "model": MODEL_LABEL,
                "training_or_inference": MODE,
                "gpu_type": GPU_TYPE,
                "num_gpus": str(run.case.num_gpus),
                "global_batch_size": "16",
                "seq_len": "65536",
                "parallelism": run.case.parallelism_label,
                "variant": run.variant,
                "status": result.status,
                "perturbed_factor": run.perturbed_factor,
                "perturbation_multiplier": f"{run.multiplier:.8g}",
                "compute_factor": f"{run.factors['compute']:.8f}",
                "memory_factor": f"{run.factors['memory']:.8f}",
                "communication_factor": f"{run.factors['communication']:.8f}",
                "launch_overhead_s": f"{run.launch_overhead_s:.8g}",
                "predicted_runtime_s": _num(result.runtime_s),
                "baseline_runtime_s": _num(baseline_runtime),
                "source_tsv_runtime_s": _num(run.case.source_tsv_runtime_s),
                "runtime_change_percent": _num(runtime_change_pct),
                "elasticity_abs": _num(elasticity_abs),
                "baseline_rank": "" if baseline_rank is None else str(baseline_rank),
                "variant_rank": "" if variant_rank is None else str(variant_rank),
                "rank_shift": "" if rank_shift is None else str(rank_shift),
                "rank_order_changed": "yes" if rank_changed else "no",
                "hardware_config": _rel(run.hardware_path),
                "model_config": _rel(MODEL_PATH),
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


def _runtime_by_variant(rows: Iterable[Mapping[str, str]]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        runtime = float(row["predicted_runtime_s"])
        values.setdefault(row["variant"], {})[row["case_id"]] = runtime
    return values


def _gap_percent(a: float, b: float) -> float:
    return 100.0 * (b - a) / a


def _write_summary(rows: list[dict[str, str]], base_factors: Mapping[str, float], launch: float) -> None:
    values = _runtime_by_variant(rows)
    baseline = values.get("baseline", {})
    baseline_order = _rank_order(baseline) if baseline else []

    ok_rows = [row for row in rows if row["status"] == "ok"]
    failed_rows = [row for row in rows if row["status"] != "ok"]
    sensitivity_rows = [
        row for row in ok_rows if row["variant"] != "baseline" and row["runtime_change_percent"]
    ]
    max_abs_change = max((abs(float(row["runtime_change_percent"])) for row in sensitivity_rows), default=float("nan"))
    min_elasticity = min((float(row["elasticity_abs"]) for row in sensitivity_rows), default=float("nan"))
    max_elasticity = max((float(row["elasticity_abs"]) for row in sensitivity_rows), default=float("nan"))

    lines: list[str] = []
    lines.append("# Training Ranking Sensitivity: 96-GPU Near-Tie Slice")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Source slice: `{_rel(SOURCE_TSV)}`.")
    lines.append("- Included all 13 archived 96-GPU training configurations from that slice.")
    lines.append(f"- Model config: `{_rel(MODEL_PATH)}` (`global_batch_size=16`, `seq_len=65536`).")
    lines.append(f"- Hardware base config: `{_rel(BASE_HW_PATH)}`.")
    lines.append(
        "- Frozen Table III factors used for baseline: "
        f"compute `{base_factors['compute']:.2f}`, memory `{base_factors['memory']:.2f}`, "
        f"communication `{base_factors['communication']:.2f}`, launch overhead `{launch:.8g}` seconds."
    )
    lines.append("- Only the compute factor was perturbed, one-at-a-time, by -10% and +10%.")
    lines.append("- No derate recalibration, topology change, model change, or source-row exclusion was applied.")
    lines.append("")

    lines.append("## Result")
    lines.append("")
    lines.append(f"- Runs completed: `{len(ok_rows)}/{len(rows)}`.")
    if failed_rows:
        lines.append(f"- Failed runs: `{len(failed_rows)}`; see CSV `error` column.")
    if math.isfinite(max_abs_change):
        lines.append(f"- Max absolute runtime movement under compute +/-10%: `{max_abs_change:.4f}%`.")
        lines.append(f"- Runtime elasticity magnitude range: `{min_elasticity:.4f}` to `{max_elasticity:.4f}`.")

    for variant in ("compute_m10", "compute_p10"):
        order = _rank_order(values.get(variant, {})) if values.get(variant) else []
        lines.append(
            f"- `{variant}` rank order changed vs baseline: "
            f"`{'yes' if order != baseline_order else 'no'}`."
        )

    rank_shift_rows = [
        row
        for row in rows
        if row["status"] == "ok" and row["variant"] != "baseline" and row["rank_shift"] not in ("", "0")
    ]
    if rank_shift_rows:
        lines.append("- Rank-shift details:")
        for row in rank_shift_rows:
            lines.append(
                f"  - `{row['variant']}`: `{row['case_id']}` moved from rank "
                f"`{row['baseline_rank']}` to `{row['variant_rank']}` "
                f"(shift `{row['rank_shift']}`)."
            )
    else:
        lines.append("- No non-baseline variant changed any per-configuration rank.")

    if baseline_order:
        lines.append(f"- Baseline best config: `{baseline_order[0]}` at `{baseline[baseline_order[0]]:.6f}s`.")
        for variant in ("compute_m10", "compute_p10"):
            if values.get(variant):
                order = _rank_order(values[variant])
                lines.append(f"- `{variant}` best config: `{order[0]}` at `{values[variant][order[0]]:.6f}s`.")

    if len(baseline_order) >= 3:
        second = baseline_order[1]
        third = baseline_order[2]
        gap = _gap_percent(baseline[second], baseline[third])
        lines.append(
            f"- Closest top-ranked Table III baseline pair: rank 2 `{second}` vs rank 3 `{third}` "
            f"with `{gap:.4f}%` gap."
        )

    lines.append("")
    lines.append("## Near-Tie Pair")
    lines.append("")
    source_rows = {row["case_id"]: float(row["source_tsv_runtime_s"]) for row in rows if row["variant"] == "baseline"}
    if NEAR_TIE_A in source_rows and NEAR_TIE_B in source_rows:
        source_gap = _gap_percent(source_rows[NEAR_TIE_A], source_rows[NEAR_TIE_B])
        lines.append(
            f"- Archived source gap `{NEAR_TIE_A}` -> `{NEAR_TIE_B}`: "
            f"`{source_gap:.4f}%` (`{source_rows[NEAR_TIE_A]:.6f}s` vs `{source_rows[NEAR_TIE_B]:.6f}s`)."
        )
    for variant in ("baseline", "compute_m10", "compute_p10"):
        vals = values.get(variant, {})
        if NEAR_TIE_A in vals and NEAR_TIE_B in vals:
            first = NEAR_TIE_A if vals[NEAR_TIE_A] <= vals[NEAR_TIE_B] else NEAR_TIE_B
            second = NEAR_TIE_B if first == NEAR_TIE_A else NEAR_TIE_A
            gap = _gap_percent(vals[first], vals[second])
            lines.append(
                f"- `{variant}`: `{first}` faster than `{second}` by `{gap:.4f}%` "
                f"(`{vals[first]:.6f}s` vs `{vals[second]:.6f}s`)."
            )

    lines.append("")
    lines.append("## Baseline Ranking")
    lines.append("")
    for idx, case_id in enumerate(baseline_order, start=1):
        lines.append(f"{idx}. `{case_id}`: `{baseline[case_id]:.6f}s`")

    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- CSV: `{_rel(RESULTS_CSV)}`")
    lines.append(f"- Commands: `{_rel(COMMANDS_TXT)}`")
    lines.append(f"- Case configs and logs: `{_rel(CASE_ROOT)}/`")

    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_commands(planned: Iterable[PlannedRun], workers: int) -> None:
    lines = [
        f"{shlex.quote(str(PYTHON))} {shlex.quote(str(Path(__file__).resolve()))} --workers {workers}",
        "",
    ]
    lines.extend(run.command for run in planned)
    COMMANDS_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    cases = _read_cases()
    planned, base_factors, launch = _plan_runs(cases)
    _write_commands(planned, args.workers)
    print(
        f"Running {len(planned)} training simulations: "
        f"{len(cases)} cases x {len(VARIANTS)} variants with workers={args.workers}"
    )
    results = _run_all(planned, max(1, args.workers))
    rows = _rows_from_results(planned, results)
    _write_csv(rows)
    _write_summary(rows, base_factors, launch)

    failed = [result for result in results.values() if result.status != "ok"]
    print(f"Wrote {_rel(RESULTS_CSV)}")
    print(f"Wrote {_rel(SUMMARY_MD)}")
    if failed:
        print(f"Failed runs: {len(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
