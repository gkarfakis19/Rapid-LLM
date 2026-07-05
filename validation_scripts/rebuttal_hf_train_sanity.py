#!/usr/bin/env python3
"""HF/Nanotron training sanity check with frozen Table III factors.

This evaluates a small structurally selected subset of HF benchmark training
rows without using the separately calibrated/tuned HF hardware YAML. Two modes
are emitted:

* fixed_network: use the base H100 topology and only apply Table III factors.
* hf_network: mimic the HF validation harness's row bandwidth mapping, while
  still using frozen Table III realization factors and no tuned util values.
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
BASE_HW = PROJECT_ROOT / "configs" / "hardware-config" / "H100_SXM5_80GB.yaml"
INPUT_CSV = SCRIPT_DIR / "huggingface_data" / "bench_final2_mod_add.csv"
RUN_PERF = PROJECT_ROOT / "run_perf.py"

OUT_ROOT = SCRIPT_DIR / "rebuttal_hf_train_sanity"
CASE_ROOT = OUT_ROOT / "case_configs"
RESULTS_CSV = OUT_ROOT / "hf_train_results.csv"
SUMMARY_TXT = OUT_ROOT / "summary.txt"

TRAIN_TIME_RE = re.compile(r"Training time for batch:\s*([0-9]+(?:\.[0-9]+)?)s")
GCN = 8


SELECTED_JOB_IDS = [
    14098532,  # 1.34B, TP+PP, 8 GPUs, ZeRO-0
    14098534,  # 1.34B, TP-heavy PP, 8 GPUs, ZeRO-0
    14098568,  # 8.86B, DP+TP+PP, 8 GPUs, ZeRO-1
    14098628,  # 8.86B, DP+TP+PP, 16 GPUs, ZeRO-1
    14097163,  # 80B, DP+TP+PP, 64 GPUs, ZeRO-1
    14280962,  # 80B GQA, DP+TP+PP, 64 GPUs, ZeRO-0
]


CSV_COLUMNS = [
    "case_id",
    "mode",
    "status",
    "model",
    "training_or_inference",
    "gpu_type",
    "num_gpus",
    "global_batch_size",
    "micro_batch_size",
    "gradient_accumulation_steps_source",
    "sequence_length",
    "parallelism",
    "zero_stage",
    "attention_type",
    "measured_metric",
    "measured_tok_s_gpu",
    "predicted_metric",
    "predicted_tok_s_gpu",
    "predicted_step_time_s",
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


def _deep_update(target: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _derates() -> dict[str, float]:
    config = _load_yaml(DERATE_CONFIG)
    shared = config["shared"]
    device = config["device_types"]["H100_SXM5"]
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
    # Remove HF-specific tuned overheads when present.
    hw.setdefault("sw_param", {}).pop("grad_acc_overhead", None)
    hw.setdefault("tech_param", {}).setdefault("core", {})["util"] = factors[
        "compute_util"
    ]
    hw.setdefault("tech_param", {}).setdefault("DRAM", {})["util"] = factors["dram_util"]
    for dim in hw.get("network", {}).get("dimensions", []):
        dim.setdefault("topology", {})["util"] = factors["network_util"]


def _positive_min(values: list[float]) -> float | None:
    positive = [float(v) for v in values if v is not None and float(v) > 0]
    return min(positive) if positive else None


def _hf_network_update(hw: dict[str, Any], row: pd.Series) -> None:
    dims = hw.get("network", {}).get("dimensions", [])
    if len(dims) < 3:
        return

    tp = int(row["tp"])
    pp = int(row["pp"])
    dp = int(row["dp"])
    nodes = int(float(row["nodes"]))
    tp_use_inter = tp > GCN
    pp_use_inter = (tp * pp) > GCN
    dp_use_inter = nodes > 1 and dp > 1

    rs_inter = float(row["ReduceScatter (GB/s)"])
    ag_inter = float(row["AllGather (GB/s)"])
    ar_inter = float(row["AllReduce (GB/s)"])
    rs_intra = float(row["RS Intra-node (GB/s)"])
    ag_intra = float(row["AG Intra-node (GB/s)"])
    ar_intra = float(row["AR Intra-node (GB/s)"])

    tp_bw = _positive_min([rs_inter, ag_inter] if tp_use_inter else [rs_intra, ag_intra])
    pp_bw = _positive_min([rs_inter, ag_inter] if pp_use_inter else [rs_intra, ag_intra])
    dp_bw = _positive_min([ar_inter] if dp_use_inter else [ar_intra])

    dims[0]["parallelisms"] = ["tp", "cp"]
    if tp_bw:
        dims[0].setdefault("topology", {})["bandwidth"] = f"{int(round(tp_bw))} GB"
    dims[1]["parallelisms"] = ["pp"]
    if pp_bw:
        dims[1].setdefault("topology", {})["bandwidth"] = f"{int(round(pp_bw))} GB"
    dims[2]["parallelisms"] = ["dp"]
    if dp_bw:
        dims[2].setdefault("topology", {})["bandwidth"] = f"{int(round(dp_bw))} GB"


def _rapid_mb_and_gas(row: pd.Series) -> tuple[int, int]:
    batch_accum = max(1, int(float(row["batch_accum"])))
    if int(row["pp"]) <= 1:
        return 1, batch_accum
    return batch_accum, 1


def _model_config(row: pd.Series) -> dict[str, Any]:
    num_heads = int(row["num_heads"])
    kv_heads = int(float(row["num_kv_heads"]))
    attention_type = "gqa" if kv_heads < num_heads else "mha"
    _, gas = _rapid_mb_and_gas(row)
    intermediate_size = int(float(row["interm_dim"]))
    num_layers = int(row["num_layers"])
    return {
        "model_param": {
            "mode": "LLM",
            "run_type": "training",
            "model_type": "llama",
            "tied_embeddings": True,
            "global_batch_size": int(row["gbs"]),
            "gradient_accumulation_steps": gas,
            "seq_len": int(row["seq_len"]),
            "hidden_dim": int(row["hidden_size"]),
            "num_layers": num_layers,
            "intermediate_size": intermediate_size,
            "vocab_size": int(row["vocab_size"]),
            "moe": {
                "num_experts": 1,
                "top_k": 1,
                "moe_intermediate_size": intermediate_size,
                "n_shared_experts": 0,
                "moe_layer_freq": 1,
                "first_k_dense_replace": num_layers,
            },
            "attention": {
                "attention_type": attention_type,
                "num_heads": num_heads,
                "kv_heads": kv_heads if attention_type == "gqa" else None,
                "use_flashattention": True,
                "attention_tile_size": 128,
            },
        }
    }


def _load_rows() -> list[pd.Series]:
    df = pd.read_csv(INPUT_CSV)
    selected = df[df["job_id"].isin(SELECTED_JOB_IDS)].copy()
    missing = sorted(set(SELECTED_JOB_IDS) - set(selected["job_id"].astype(int)))
    if missing:
        raise ValueError(f"Missing selected HF job IDs: {missing}")
    selected["num_gpus"] = selected["dp"] * selected["pp"] * selected["tp"]
    selected["calib_subset"] = (
        (selected["pp"] == 1)
        & (selected["tp"].between(2, 16))
        & (selected["dp"].between(8, 64))
    )
    bad = selected[
        (selected["status"] != "Success")
        | (selected["after_pp_fix"] != True)  # noqa: E712
        | (selected["calib_subset"] == True)  # noqa: E712
    ]
    if not bad.empty:
        raise ValueError(f"Selected rows are not clean held-out Success rows: {bad['job_id'].tolist()}")
    by_id = {int(row["job_id"]): row for _, row in selected.iterrows()}
    return [by_id[job_id] for job_id in SELECTED_JOB_IDS]


def _case_id(row: pd.Series) -> str:
    return (
        f"hf_job{int(row['job_id'])}_h{int(row['hidden_size'])}_L{int(row['num_layers'])}"
        f"_dp{int(row['dp'])}_tp{int(row['tp'])}_pp{int(row['pp'])}_z{int(row['zero_stage'])}"
    )


def _build_configs(row: pd.Series, mode: str, case_dir: Path, factors: dict[str, float]) -> tuple[Path, Path]:
    hw = _load_yaml(BASE_HW)
    rapid_mb, _ = _rapid_mb_and_gas(row)
    _deep_update(
        hw,
        {
            "parallelism": {
                "auto": False,
                "tp": int(row["tp"]),
                "tp_sp": True,
                "cp": 1,
                "pp": int(row["pp"]),
                "mb": int(rapid_mb),
                "train": {"dp": int(row["dp"]), "ep": 1, "tp_ep": True},
                "inference": {"replica_count": 1, "moe_dp": 1},
            },
            "sw_param": {"dp_zero_stage": int(row["zero_stage"])},
        },
    )
    if mode == "hf_network":
        _hf_network_update(hw, row)
    elif mode != "fixed_network":
        raise ValueError(f"Unknown mode {mode}")
    _apply_derates(hw, factors)

    model = _model_config(row)
    hw_path = case_dir / "hardware.yaml"
    model_path = case_dir / "model.yaml"
    _write_yaml(hw_path, hw)
    _write_yaml(model_path, model)
    return hw_path, model_path


def _run(row: pd.Series, mode: str, factors: dict[str, float]) -> dict[str, Any]:
    case_id = _case_id(row)
    case_dir = CASE_ROOT / mode / case_id
    hw_path, model_path = _build_configs(row, mode, case_dir, factors)
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
    predicted_time = float(match.group(1)) if match else math.nan

    num_gpus = int(row["dp"]) * int(row["tp"]) * int(row["pp"])
    total_tokens = int(row["gbs"]) * int(row["seq_len"])
    predicted_tok_s_gpu = (
        total_tokens / (predicted_time * num_gpus) if predicted_time == predicted_time and predicted_time > 0 else math.nan
    )
    measured_tok_s_gpu = float(row["tok/s/gpu"])
    ape = (
        abs(predicted_tok_s_gpu - measured_tok_s_gpu) / measured_tok_s_gpu * 100.0
        if predicted_tok_s_gpu == predicted_tok_s_gpu and measured_tok_s_gpu > 0
        else math.nan
    )
    attention_type = "gqa" if int(float(row["num_kv_heads"])) < int(row["num_heads"]) else "mha"
    status = "success" if result.returncode == 0 and predicted_tok_s_gpu == predicted_tok_s_gpu else "excluded"
    notes = [
        "HF/Nanotron Success row outside hf_meta_sweep calibration subset (not pp=1,tp=2-16,dp=8-64).",
        "No H100 calibrated/tuned hardware YAML used.",
        "Throughput error computed directly on tok/s/gpu.",
    ]
    if mode == "hf_network":
        notes.append("HF row collective bandwidths used to map network dimensions, matching HF harness style.")
    else:
        notes.append("Strict fixed-network mode: base H100 topology preserved.")
    if result.returncode != 0:
        notes.append(f"run_perf return code {result.returncode}; see {_rel(log_path)}.")

    return {
        "case_id": case_id,
        "mode": mode,
        "status": status,
        "model": str(row["name"]).split("_dp", 1)[0],
        "training_or_inference": "training",
        "gpu_type": "H100 SXM5 80GB",
        "num_gpus": num_gpus,
        "global_batch_size": int(row["gbs"]),
        "micro_batch_size": int(row["mbs"]),
        "gradient_accumulation_steps_source": int(float(row["batch_accum"])),
        "sequence_length": int(row["seq_len"]),
        "parallelism": f"DP={int(row['dp'])}, TP={int(row['tp'])}, PP={int(row['pp'])}, CP=1",
        "zero_stage": int(row["zero_stage"]),
        "attention_type": attention_type,
        "measured_metric": "tok/s/gpu",
        "measured_tok_s_gpu": f"{measured_tok_s_gpu:.6f}",
        "predicted_metric": "tok/s/gpu",
        "predicted_tok_s_gpu": f"{predicted_tok_s_gpu:.6f}" if predicted_tok_s_gpu == predicted_tok_s_gpu else "",
        "predicted_step_time_s": f"{predicted_time:.6f}" if predicted_time == predicted_time else "",
        "absolute_percentage_error": f"{ape:.2f}" if ape == ape else "",
        "source_csv": _rel(INPUT_CSV),
        "hardware_config": _rel(hw_path),
        "model_config": _rel(model_path),
        "command": " ".join([_rel(Path(x)) if Path(x).is_absolute() and Path(x).exists() else x for x in command]),
        "notes": "; ".join(notes),
    }


def _write_summary(rows: list[dict[str, Any]], factors: dict[str, float]) -> None:
    lines = [
        "HF/Nanotron H100 training sanity check",
        "",
        f"Frozen factors: H100_SXM5 compute={factors['compute_util']}, memory={factors['dram_util']}, communication={factors['network_util']}, launch={factors['kernel_launch_overhead_s']}s.",
        f"Base hardware: {_rel(BASE_HW)}",
        "No configs/hardware-config/H100_SXM5_80GB_calibrated_tuned.yaml was used.",
        "Rows were selected structurally before running: Success, after_pp_fix=True, outside hf_meta_sweep pp=1/tp=2-16/dp=8-64 calibration subset.",
        f"CSV: {_rel(RESULTS_CSV)}",
        "",
    ]
    for mode in ("fixed_network", "hf_network"):
        errors = [
            float(row["absolute_percentage_error"])
            for row in rows
            if row["mode"] == mode and row["status"] == "success" and row["absolute_percentage_error"]
        ]
        lines.append(f"{mode}:")
        if errors:
            lines.append(f"  N: {len(errors)}")
            lines.append(f"  MAPE: {statistics.fmean(errors):.2f}%")
            lines.append(f"  Median absolute percentage error: {statistics.median(errors):.2f}%")
            lines.append(f"  Max absolute percentage error: {max(errors):.2f}%")
        else:
            lines.append("  No successful cases.")
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    factors = _derates()
    rows: list[dict[str, Any]] = []
    for mode in ("fixed_network", "hf_network"):
        for row in _load_rows():
            rows.append(_run(row, mode, factors))

    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    _write_summary(rows, factors)

    for mode in ("fixed_network", "hf_network"):
        errors = [
            float(row["absolute_percentage_error"])
            for row in rows
            if row["mode"] == mode and row["status"] == "success" and row["absolute_percentage_error"]
        ]
        if errors:
            print(f"{mode}: N={len(errors)}, MAPE={statistics.fmean(errors):.2f}%, max={max(errors):.2f}%")
        else:
            print(f"{mode}: no successful cases")
    print(f"Wrote {RESULTS_CSV}")
    print(f"Summary: {SUMMARY_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
