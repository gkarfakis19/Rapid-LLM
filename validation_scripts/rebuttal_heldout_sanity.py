#!/usr/bin/env python3
"""Held-out RAPID-LLM sanity check for rebuttal.

This script runs a small set of NVIDIA NIM Llama 3.3-70B inference cases that
are not part of the submitted Figs. 6-9 validation plots. It intentionally uses
the frozen Table III factors from validation_configs/harness_derates.yaml and
does not use any per-dataset fitted hardware YAMLs.
"""

from __future__ import annotations

import csv
import copy
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
VALIDATION_CONFIG_ROOT = SCRIPT_DIR / "validation_configs"
HW_CONFIG_ROOT = VALIDATION_CONFIG_ROOT / "hardware-config"
MODEL_CONFIG_ROOT = VALIDATION_CONFIG_ROOT / "model-config"
NVIDIA_DATA_ROOT = SCRIPT_DIR / "nvidia_data"
DERATE_CONFIG = VALIDATION_CONFIG_ROOT / "harness_derates.yaml"
RUN_PERF = PROJECT_ROOT / "run_perf.py"

OUT_ROOT = SCRIPT_DIR / "rebuttal_heldout_sanity"
CASE_CONFIG_ROOT = OUT_ROOT / "case_configs"
RESULTS_CSV = OUT_ROOT / "heldout_results.csv"
SUMMARY_TXT = OUT_ROOT / "summary.txt"
COMMANDS_TXT = OUT_ROOT / "commands.txt"


DATASETS: dict[str, dict[str, Any]] = {
    "A100_SXM4": {
        "source_csv": NVIDIA_DATA_ROOT / "8xA100_bf16_Llama3_3-70B.csv",
        "base_hw": HW_CONFIG_ROOT / "a100_80GB_inf.yaml",
        "derate_key": "A100_SXM4",
        "gpu_type": "A100 SXM4 80GB",
        "num_gpus": 8,
        "tp": 8,
        "source_precision": "bf16",
        "notes": (
            "NVIDIA source CSV labels the system as 8xA100 BF16 but does not "
            "state PCIe vs SXM; modeled with the repo's unfitted A100 inference "
            "base config and A100_SXM4/NVLink-class Table III factors."
        ),
    },
    "H100_SXM5": {
        "source_csv": NVIDIA_DATA_ROOT / "4xH100_fp16_Llama3_3-70B.csv",
        "base_hw": HW_CONFIG_ROOT / "H100_SXM5_80GB.yaml",
        "derate_key": "H100_SXM5",
        "gpu_type": "H100 SXM5 80GB",
        "num_gpus": 4,
        "tp": 4,
        "source_precision": "fp16",
        "notes": (
            "NVIDIA source CSV labels the system as 4xH100 FP16; local H100 "
            "config uses bf16-equivalent tensor width/peak timing."
        ),
    },
}


MODEL_NAME = "Llama 3.3-70B"
MODEL_CONFIG = MODEL_CONFIG_ROOT / "Llama3.1-70B_inf.yaml"


CASES: list[dict[str, Any]] = [
    {
        "case_id": "nim_a100sxm4_llama33_70b_p200_o200_b1",
        "dataset": "A100_SXM4",
        "input_tokens": 200,
        "output_tokens": 200,
        "concurrency": 1,
    },
    {
        "case_id": "nim_a100sxm4_llama33_70b_p500_o2000_b5",
        "dataset": "A100_SXM4",
        "input_tokens": 500,
        "output_tokens": 2000,
        "concurrency": 5,
    },
    {
        "case_id": "nim_a100sxm4_llama33_70b_p20000_o2000_b25",
        "dataset": "A100_SXM4",
        "input_tokens": 20000,
        "output_tokens": 2000,
        "concurrency": 25,
    },
    {
        "case_id": "nim_h100sxm5_llama33_70b_p200_o200_b25",
        "dataset": "H100_SXM5",
        "input_tokens": 200,
        "output_tokens": 200,
        "concurrency": 25,
    },
    {
        "case_id": "nim_h100sxm5_llama33_70b_p500_o2000_b5",
        "dataset": "H100_SXM5",
        "input_tokens": 500,
        "output_tokens": 2000,
        "concurrency": 5,
    },
    {
        "case_id": "nim_h100sxm5_llama33_70b_p2500_o300_b1",
        "dataset": "H100_SXM5",
        "input_tokens": 2500,
        "output_tokens": 300,
        "concurrency": 1,
    },
]


CSV_COLUMNS = [
    "case_id",
    "status",
    "model",
    "training_or_inference",
    "gpu_type",
    "num_gpus",
    "batch_size_or_concurrency",
    "input_tokens",
    "output_tokens",
    "sequence_length_configured",
    "parallelism",
    "measured_metric",
    "measured_total_latency_s",
    "predicted_metric",
    "predicted_total_latency_s",
    "absolute_percentage_error",
    "source_csv",
    "hardware_config",
    "model_config",
    "command",
    "notes",
]


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


def _load_derates() -> dict[str, Any]:
    derate_config = _load_yaml(DERATE_CONFIG)
    shared = derate_config.get("shared", {})
    device_types = derate_config.get("device_types", {})
    if not isinstance(shared, dict) or not isinstance(device_types, dict):
        raise ValueError(f"Bad derate config schema in {DERATE_CONFIG}")
    return derate_config


def _derates_for(derate_config: dict[str, Any], key: str) -> dict[str, float]:
    shared = derate_config["shared"]
    device = derate_config["device_types"][key]
    return {
        "kernel_launch_overhead_s": float(shared["kernel_launch_overhead_s"]),
        "dram_util": float(device["dram_util"]),
        "network_util": float(device["network_util"]),
        "compute_util": float(device["compute_util"]),
    }


def _apply_table_factors(hw_config: dict[str, Any], factors: dict[str, float]) -> None:
    hw_config.setdefault("sw_param", {})["kernel_launch_overhead"] = factors[
        "kernel_launch_overhead_s"
    ]
    hw_config.setdefault("tech_param", {}).setdefault("core", {})["util"] = factors[
        "compute_util"
    ]
    hw_config.setdefault("tech_param", {}).setdefault("DRAM", {})["util"] = factors[
        "dram_util"
    ]
    for dimension in hw_config.get("network", {}).get("dimensions", []):
        topology = dimension.setdefault("topology", {})
        topology["util"] = factors["network_util"]


def _configure_parallelism(hw_config: dict[str, Any], tp: int) -> None:
    parallelism = hw_config.setdefault("parallelism", {})
    parallelism["auto"] = False
    parallelism["tp"] = tp
    parallelism["tp_sp"] = True
    parallelism["cp"] = 1
    parallelism["pp"] = 1
    parallelism["mb"] = 1

    train = parallelism.setdefault("train", {})
    train["dp"] = 1
    train["ep"] = 1
    train["tp_ep"] = True

    inference = parallelism.setdefault("inference", {})
    inference["replica_count"] = 1
    inference["moe_dp"] = 1


def _configure_model(
    model_config: dict[str, Any],
    input_tokens: int,
    output_tokens: int,
    concurrency: int,
) -> None:
    model_param = model_config.setdefault("model_param", {})
    model_param["run_type"] = "inference"
    model_param["global_batch_size"] = concurrency
    model_param["seq_len"] = input_tokens + output_tokens
    model_param["decode_len"] = output_tokens


def _load_nvidia_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _find_measurement(dataset: dict[str, Any], case: dict[str, Any]) -> dict[str, float] | None:
    source_csv = dataset["source_csv"]
    for row in _load_nvidia_rows(source_csv):
        input_tokens = int(row["Input Tokens"])
        output_tokens = int(row["Output Tokens"])
        concurrency = int(row["Concurrency"])
        if (
            input_tokens == case["input_tokens"]
            and output_tokens == case["output_tokens"]
            and concurrency == case["concurrency"]
        ):
            ttft_ms = float(row["TTFT (ms)"])
            itl_ms = float(row["ITL (ms)"])
            throughput_tps = float(row["Throughput (Tokens/s)"])
            total_latency_s = (ttft_ms + itl_ms * max(output_tokens - 1, 0)) / 1000.0
            return {
                "ttft_ms": ttft_ms,
                "itl_ms": itl_ms,
                "throughput_tps": throughput_tps,
                "total_latency_s": total_latency_s,
            }
    return None


def _parse_prediction(stdout: str) -> float | None:
    match = re.search(r"LLM inference time:\s*([0-9.eE+-]+)s", stdout)
    if not match:
        return None
    return float(match.group(1))


def _command_display(command: list[str]) -> str:
    rendered: list[str] = []
    for item in command:
        path = Path(item)
        if path.is_absolute() and path.exists():
            rendered.append(_rel(path))
        else:
            rendered.append(item)
    return " ".join(rendered)


def _run_case(
    case: dict[str, Any],
    dataset: dict[str, Any],
    derates: dict[str, float],
) -> dict[str, Any]:
    measurement = _find_measurement(dataset, case)
    case_dir = CASE_CONFIG_ROOT / case["case_id"]
    hw_path = case_dir / "hardware.yaml"
    model_path = case_dir / "model.yaml"
    log_path = case_dir / "run.log"

    base_notes = [
        "Held-out NVIDIA NIM data; not one of submitted Figs. 6-9 validation datasets.",
        "Measured total latency is computed as TTFT + ITL * (output_tokens - 1).",
        "Llama 3.3-70B is represented by the repo's Llama3.1-70B inference config.",
        "Concurrency is modeled as global_batch_size; no serving scheduler constants were tuned.",
        dataset["notes"],
    ]

    if measurement is None:
        return {
            "case_id": case["case_id"],
            "status": "excluded",
            "model": MODEL_NAME,
            "training_or_inference": "inference",
            "gpu_type": dataset["gpu_type"],
            "num_gpus": dataset["num_gpus"],
            "batch_size_or_concurrency": case["concurrency"],
            "input_tokens": case["input_tokens"],
            "output_tokens": case["output_tokens"],
            "sequence_length_configured": case["input_tokens"] + case["output_tokens"],
            "parallelism": f"TP={dataset['tp']}, PP=1, DP=1, CP=1, TP_SP=True",
            "measured_metric": "total_latency_s",
            "measured_total_latency_s": "",
            "predicted_metric": "total_latency_s",
            "predicted_total_latency_s": "",
            "absolute_percentage_error": "",
            "source_csv": _rel(dataset["source_csv"]),
            "hardware_config": "",
            "model_config": "",
            "command": "",
            "notes": "; ".join(base_notes + ["No exact measurement row found."]),
        }

    hw_config = copy.deepcopy(_load_yaml(dataset["base_hw"]))
    model_config = copy.deepcopy(_load_yaml(MODEL_CONFIG))
    _apply_table_factors(hw_config, derates)
    _configure_parallelism(hw_config, dataset["tp"])
    _configure_model(
        model_config,
        case["input_tokens"],
        case["output_tokens"],
        case["concurrency"],
    )
    _write_yaml(hw_path, hw_config)
    _write_yaml(model_path, model_config)

    command = [
        sys.executable,
        str(RUN_PERF),
        "--hardware_config",
        str(hw_path),
        "--model_config",
        str(model_path),
    ]
    command_text = _command_display(command)
    process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    case_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "COMMAND: "
        + command_text
        + "\n\nSTDOUT:\n"
        + process.stdout
        + "\n\nSTDERR:\n"
        + process.stderr,
        encoding="utf-8",
    )

    predicted = _parse_prediction(process.stdout)
    status = "success" if process.returncode == 0 and predicted is not None else "excluded"
    notes = list(base_notes)
    if process.returncode != 0:
        notes.append(f"run_perf exited with code {process.returncode}; see {_rel(log_path)}.")
    if predicted is None:
        notes.append(f"Could not parse LLM inference time; see {_rel(log_path)}.")

    measured = measurement["total_latency_s"]
    ape = abs(predicted - measured) / measured * 100.0 if predicted is not None else ""

    return {
        "case_id": case["case_id"],
        "status": status,
        "model": MODEL_NAME,
        "training_or_inference": "inference",
        "gpu_type": dataset["gpu_type"],
        "num_gpus": dataset["num_gpus"],
        "batch_size_or_concurrency": case["concurrency"],
        "input_tokens": case["input_tokens"],
        "output_tokens": case["output_tokens"],
        "sequence_length_configured": case["input_tokens"] + case["output_tokens"],
        "parallelism": f"TP={dataset['tp']}, PP=1, DP=1, CP=1, TP_SP=True",
        "measured_metric": "total_latency_s",
        "measured_total_latency_s": f"{measured:.6f}",
        "predicted_metric": "total_latency_s",
        "predicted_total_latency_s": f"{predicted:.6f}" if predicted is not None else "",
        "absolute_percentage_error": f"{ape:.2f}" if isinstance(ape, float) else "",
        "source_csv": _rel(dataset["source_csv"]),
        "hardware_config": _rel(hw_path),
        "model_config": _rel(model_path),
        "command": command_text,
        "notes": "; ".join(notes),
    }


def _write_results(rows: list[dict[str, Any]]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [
        float(row["absolute_percentage_error"])
        for row in rows
        if row["status"] == "success" and row["absolute_percentage_error"] != ""
    ]
    if not errors:
        return {"n": 0, "mape": None, "median": None, "max": None}
    return {
        "n": len(errors),
        "mape": statistics.fmean(errors),
        "median": statistics.median(errors),
        "max": max(errors),
    }


def _write_summary(rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
    source = ""
    source_path = NVIDIA_DATA_ROOT / "source.txt"
    if source_path.exists():
        source = source_path.read_text(encoding="utf-8").strip()

    derate_text = (
        "Frozen Table III factors used from "
        f"{_rel(DERATE_CONFIG)}: A100 PCIe 80GB compute=0.60, memory=0.70, "
        "communication=0.85; A100 SXM4 80GB compute=0.90, memory=0.70, "
        "communication=0.80; H100 SXM5 80GB compute=0.56, memory=0.80, "
        "communication=0.85; launch overhead=6 us."
    )

    lines = [
        "RAPID-LLM held-out rebuttal sanity check",
        "",
        derate_text,
        "No .fitted.yaml hardware configs were used. No Table III factors were changed.",
        f"Source data: {source}" if source else "Source data: validation_scripts/nvidia_data",
        f"Python used: {_rel(Path(sys.executable))}",
        "",
    ]

    if aggregate["n"]:
        lines.extend(
            [
                "Aggregate over successful included cases:",
                f"  N: {aggregate['n']}",
                f"  MAPE: {aggregate['mape']:.2f}%",
                f"  Median absolute percentage error: {aggregate['median']:.2f}%",
                f"  Max absolute percentage error: {aggregate['max']:.2f}%",
                "",
                "Rebuttal-ready sentence:",
                (
                    "As a post-review sanity check, we kept all Table III factors fixed "
                    f"and applied RAPID-LLM to {aggregate['n']} held-out inference "
                    "configurations spanning Llama 3.3-70B on A100 SXM4/H100 SXM5, "
                    "4-8 GPUs, prompt lengths 200-20000, output lengths 200-2000, "
                    "and concurrency 1-25; the resulting aggregate error was "
                    f"{aggregate['mape']:.2f}% (max {aggregate['max']:.2f}%), "
                    "consistent with the factors acting as hardware/platform-class "
                    "constants rather than per-workload fitting knobs."
                ),
            ]
        )
    else:
        lines.append("No successful included cases. See CSV/logs for exclusions.")

    excluded = [row for row in rows if row["status"] != "success"]
    if excluded:
        lines.extend(["", "Excluded/failed cases:"])
        for row in excluded:
            lines.append(f"  {row['case_id']}: {row['notes']}")

    lines.extend(["", f"CSV: {_rel(RESULTS_CSV)}", f"Commands: {_rel(COMMANDS_TXT)}"])
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_commands(rows: list[dict[str, Any]]) -> None:
    command_lines = [
        "# Exact run_perf command lines used for successful parsed runs.",
        "# Case configs are materialized under validation_scripts/rebuttal_heldout_sanity/case_configs.",
        "",
    ]
    for row in rows:
        if row["command"]:
            command_lines.append(f"{row['case_id']}: {row['command']}")
    COMMANDS_TXT.write_text("\n".join(command_lines) + "\n", encoding="utf-8")


def main() -> int:
    derate_config = _load_derates()
    rows: list[dict[str, Any]] = []
    for case in CASES:
        dataset = DATASETS[case["dataset"]]
        derates = _derates_for(derate_config, dataset["derate_key"])
        rows.append(_run_case(case, dataset, derates))

    _write_results(rows)
    _write_commands(rows)
    aggregate = _aggregate(rows)
    _write_summary(rows, aggregate)

    if aggregate["n"]:
        print(
            f"Wrote {RESULTS_CSV} with N={aggregate['n']}, "
            f"MAPE={aggregate['mape']:.2f}%, max={aggregate['max']:.2f}%."
        )
    else:
        print(f"Wrote {RESULTS_CSV}; no successful parsed runs.")
    print(f"Summary: {SUMMARY_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
