#!/usr/bin/env python3
"""A100 training sanity check for possible held-out UCI FSDP rows.

The submitted paper text/validation CSV focuses on DDP rows for the local
4xA100 training figure. This script runs the successful FSDP rows present in
validation_scripts/imec_data/uci_train.csv as a separate, caveated check using
the frozen Table III A100 PCIe factors.
"""

from __future__ import annotations

import csv
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_ROOT = SCRIPT_DIR / "validation_configs"
DERATE_CONFIG = CONFIG_ROOT / "harness_derates.yaml"
BASE_HW = CONFIG_ROOT / "hardware-config" / "a100_80GB_train_validation.yaml"
BASE_MODEL = CONFIG_ROOT / "model-config" / "Llama2-7B.yaml"
INPUT_CSV = SCRIPT_DIR / "imec_data" / "uci_train.csv"
RUN_PERF = PROJECT_ROOT / "run_perf.py"

OUT_ROOT = SCRIPT_DIR / "rebuttal_train_sanity"
CASE_ROOT = OUT_ROOT / "case_configs"
RESULTS_CSV = OUT_ROOT / "a100_fsdp_train_results.csv"
SUMMARY_TXT = OUT_ROOT / "summary.txt"

GLOBAL_BATCH_SIZE = 128
MICRO_BATCH_SIZE = 1
TRAIN_TIME_RE = re.compile(r"Training time for batch:\s*([0-9]+(?:\.[0-9]+)?)s")


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


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _deep_update(target: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _a100_pcie_derates() -> dict[str, float]:
    config = _load_yaml(DERATE_CONFIG)
    shared = config["shared"]
    device = config["device_types"]["A100_PCIe"]
    return {
        "kernel_launch_overhead_s": float(shared["kernel_launch_overhead_s"]),
        "dram_util": float(device["dram_util"]),
        "network_util": float(device["network_util"]),
        "compute_util": float(device["compute_util"]),
    }


def _apply_derates(hw: dict[str, Any], factors: dict[str, float]) -> None:
    hw.setdefault("sw_param", {})["kernel_launch_overhead"] = factors[
        "kernel_launch_overhead_s"
    ]
    hw.setdefault("tech_param", {}).setdefault("core", {})["util"] = factors[
        "compute_util"
    ]
    hw.setdefault("tech_param", {}).setdefault("DRAM", {})["util"] = factors["dram_util"]
    for dim in hw.get("network", {}).get("dimensions", []):
        dim.setdefault("topology", {})["util"] = factors["network_util"]


def _dimension(
    idx: int,
    label: str,
    axes: list[str],
    bandwidth_gb: float,
    latency_s: float,
) -> dict[str, Any]:
    return {
        "id": f"dim{idx}",
        "label": label,
        "size": "auto" if axes else 1,
        "topology": {
            "type": "Ring",
            "bandwidth": f"{bandwidth_gb:g} GB",
            "latency": latency_s,
            "energy_per_bit": 8e-12,
            "util": 1.0,
            "optimize_2dmap": False,
        },
        "collective_override": {},
        "parallelisms": axes,
    }


def _network_override(tp: int, pp: int, cp: int, dp: int) -> dict[str, Any]:
    del tp, pp, cp, dp
    # Matches validation_scripts/uci_train_validation.py: TP/CP on PCIe,
    # PP/DP on NVLink for the local 4xA100 machine mapping.
    return {
        "network": {
            "dimensions": [
                _dimension(0, "tp_cp_ring", ["tp", "cp"], 25, 5e-6),
                _dimension(1, "pp_dp_ring", ["pp", "dp"], 300, 1e-6),
            ]
        },
        "execution_backend": {
            "model": "astra",
            "astra": {"mode": "full_astrasim_hierarchical"},
        },
    }


def _load_cases() -> list[dict[str, Any]]:
    df = pd.read_csv(INPUT_CSV)
    df = df[(df["variant"].str.upper() == "FSDP") & (df["Status"].str.upper() == "SUCCESS")]
    cases: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        cases.append(
            {
                "variant": str(row["variant"]).upper(),
                "tp": int(row["TP"]),
                "pp": int(row["PP"]),
                "cp": int(row["CP"]),
                "dp": int(row["DP"]),
                "actual": float(row["Avg_Step_Time_s"]),
            }
        )
    return cases


def _build_configs(case: dict[str, Any], case_dir: Path, factors: dict[str, float]) -> tuple[Path, Path]:
    hw = _load_yaml(BASE_HW)
    model = _load_yaml(BASE_MODEL)

    mb = GLOBAL_BATCH_SIZE // (MICRO_BATCH_SIZE * int(case["dp"]))
    effective_mb = mb if int(case["pp"]) > 1 else 1

    _deep_update(
        model,
        {
            "model_param": {
                "seq_len": 4096,
                "run_type": "training",
                "global_batch_size": GLOBAL_BATCH_SIZE,
            }
        },
    )
    _deep_update(
        hw,
        {
            "parallelism": {
                "tp": int(case["tp"]),
                "tp_sp": False,
                "cp": int(case["cp"]),
                "pp": int(case["pp"]),
                "mb": int(effective_mb),
                "train": {"dp": int(case["dp"]), "ep": 1, "tp_ep": True},
                "inference": {"replica_count": 1, "moe_dp": 1},
            },
            "sw_param": {"dp_zero_stage": 3},
        },
    )
    _deep_update(hw, _network_override(case["tp"], case["pp"], case["cp"], case["dp"]))
    _apply_derates(hw, factors)

    hw_path = case_dir / "hardware.yaml"
    model_path = case_dir / "model.yaml"
    _write_yaml(hw_path, hw)
    _write_yaml(model_path, model)
    return hw_path, model_path


def _run_case(case: dict[str, Any], factors: dict[str, float]) -> dict[str, Any]:
    case_id = f"uci_fsdp_tp{case['tp']}_pp{case['pp']}_cp{case['cp']}_dp{case['dp']}"
    case_dir = CASE_ROOT / case_id
    hw_path, model_path = _build_configs(case, case_dir, factors)
    log_path = case_dir / "run.log"

    command = [
        sys.executable,
        str(RUN_PERF),
        "--hardware_config",
        str(hw_path),
        "--model_config",
        str(model_path),
    ]
    result = subprocess.run(
        command,
        cwd=case_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(result.stdout or "", encoding="utf-8")
    match = TRAIN_TIME_RE.search(result.stdout or "")
    predicted = float(match.group(1)) if match else math.nan
    actual = float(case["actual"])
    ape = abs(predicted - actual) / actual * 100.0 if predicted == predicted else math.nan

    return {
        "case_id": case_id,
        "status": "success" if result.returncode == 0 and predicted == predicted else "excluded",
        "model": "Llama2-7B",
        "training_or_inference": "training",
        "gpu_type": "A100 PCIe 80GB",
        "num_gpus": int(case["tp"]) * int(case["pp"]) * int(case["cp"]) * int(case["dp"]),
        "batch_size": GLOBAL_BATCH_SIZE,
        "sequence_length": 4096,
        "parallelism": f"FSDP/ZeRO-3 TP={case['tp']}, PP={case['pp']}, CP={case['cp']}, DP={case['dp']}",
        "measured_step_time_s": f"{actual:.6f}",
        "predicted_step_time_s": f"{predicted:.6f}" if predicted == predicted else "",
        "absolute_percentage_error": f"{ape:.2f}" if ape == ape else "",
        "source_csv": _rel(INPUT_CSV),
        "hardware_config": _rel(hw_path),
        "model_config": _rel(model_path),
        "command": " ".join([_rel(Path(x)) if Path(x).is_absolute() and Path(x).exists() else x for x in command]),
        "notes": (
            "Candidate held-out local A100 FSDP row; modeled as ZeRO-3. "
            "Use only if confirming submitted Fig. 8 used DDP rows only."
        ),
    }


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    factors = _a100_pcie_derates()
    rows = [_run_case(case, factors) for case in _load_cases()]

    fieldnames = list(rows[0].keys()) if rows else []
    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    errors = [
        float(row["absolute_percentage_error"])
        for row in rows
        if row["status"] == "success" and row["absolute_percentage_error"]
    ]
    lines = [
        "A100 local UCI FSDP training sanity check",
        "",
        f"Frozen factors: A100_PCIe compute={factors['compute_util']}, memory={factors['dram_util']}, communication={factors['network_util']}, launch={factors['kernel_launch_overhead_s']}s.",
        "No fitted hardware configs were used; base config values were overwritten with Table III factors.",
        f"CSV: {_rel(RESULTS_CSV)}",
    ]
    if errors:
        lines.extend(
            [
                f"N: {len(errors)}",
                f"MAPE: {statistics.fmean(errors):.2f}%",
                f"Median absolute percentage error: {statistics.median(errors):.2f}%",
                f"Max absolute percentage error: {max(errors):.2f}%",
            ]
        )
    else:
        lines.append("No successful cases.")
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if errors:
        print(f"Wrote {RESULTS_CSV} with N={len(errors)}, MAPE={statistics.fmean(errors):.2f}%, max={max(errors):.2f}%.")
    else:
        print(f"Wrote {RESULTS_CSV}; no successful cases.")
    print(f"Summary: {SUMMARY_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
