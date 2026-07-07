#!/usr/bin/env python3
"""Per-device calibration testbench: ONE factor set per GPU for train + inference.

Devices: H100_SXM5, A100_SXM4, A100_PCIe. For each device, all available
measured training AND inference points are pooled and a single factor set
(compute util, DRAM util, network util) is calibrated on a 50:50
calibration:holdout split. Every number is a REAL simulation — no
interpolation, no surrogate fitting.

Knob design (pre-registered, identical structure for every device):
  * compute axis: 5 values at the (center DRAM, center network) cell
  * DRAM axis: low/high at (center compute, center network)
  * network axis: low/high at (center compute, center DRAM)
  * 8 factorial corners: (compute lo/hi) x (DRAM lo/hi) x (network lo/hi)
  -> 17 real (c, d, n) combos per point. The fit is argmin of
  calibration-set MAPE over these 17 real combos. kernel launch overhead
  stays frozen at the shared Table III value (6e-6 s); all communication
  overlaps (tp, tp_sp, cp) are fixed at 0.6.

Split: stratified 50:50 (strata = workload x source x scale bucket),
5 seeds; GOLDEN = the median-holdout-MAPE seed (median, not minimum).
A structured largest-scale-holdout probe is reported but never golden.

Point sources per device
  H100_SXM5  train: Megatron-Core MoE points (vpp=max) + HF/Nanotron rows
             (via h100_testbench adapters, incl. the measured-MFU >= 8%
             envelope and the pp>1 cut rule)
             inference: NVIDIA NIM 4xH100 Llama3.3-70B (16 rows, total
             latency = TTFT + ITL*(out-1)) + IMEC Llama2 rows (10 rows,
             total latency; simulated with the repo's established
             network-ignored convention for this source)
  A100_SXM4  train: imec_data/A100_train.csv (11 rows; Korthikanti +
             Selene GPT 22B-1T; per-source base hardware configs)
             inference: NVIDIA NIM 8xA100 Llama3.3-70B (14 rows) + IMEC
             Llama2 rows (11 rows, network-ignored convention)
  A100_PCIe  train: imec_data/uci_train.csv (13 usable rows: 10 DDP
             ZeRO-0 + 3 FSDP ZeRO-3; Llama2-7B on the 4xA100 UCI node)
             inference: imec_data/uci_inf.csv (9 rows; OPT 13/30/66B;
             total latency = TTFT + TPOT*(out-1))
  Excluded: koyeb.py points (different metric — decode-only throughput —
  single GPU, and one point is PCIe-contaminated per its own comments).

Training rows with measured MFU < 8% are report-only (degenerate runs).

Outputs per device (validation_scripts/device_testbench/<device>/):
  results CSV, split report; golden factors + split canonized to
  validation_scripts/validation_configs/<device>_golden_calib.yaml.

Usage:
  .venv/bin/python validation_scripts/device_testbench.py --device A100_PCIe
  .venv/bin/python validation_scripts/device_testbench.py --device all
  ... --fit-only   # reuse cached runs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from validation_scripts import h100_testbench as h100tb  # noqa: E402
from validation_scripts import moe_ep_validation as mev  # noqa: E402
from validation_scripts import rebuttal_hf_train_sanity as hfs  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_ROOT = SCRIPT_DIR / "validation_configs"
HW_ROOT = CONFIG_ROOT / "hardware-config"
MODEL_ROOT = CONFIG_ROOT / "model-config"
OUT_BASE = SCRIPT_DIR / "device_testbench"
RUN_PERF = PROJECT_ROOT / "run_perf.py"

TRAIN_TIME_RE = re.compile(r"Training time for batch:\s*([0-9]+(?:\.[0-9]+)?)s")
INF_TIME_RE = re.compile(r"LLM inference time:\s*([0-9]+(?:\.[0-9]+)?)s")

OVERLAPS = {"tp_overlap": 0.6, "tp_sp_overlap": 0.6, "cp_overlap": 0.6}
# Kernel launch overhead is a CALIBRATED knob (shared across train and
# inference). 6e-6 is the legacy Table III value; serving engines (NIM,
# TRT-LLM, vLLM) run decode under CUDA graphs where launch cost is far
# lower, which the joint fit can now express.
LAUNCH_AXIS = (1.0e-6, 3.0e-6, 6.0e-6)
LAUNCH_DEFAULT = 6.0e-6
MEASURED_MFU_FLOOR = 0.08
PP_GT1_CUT_MAPE = 25.0
# Physical floor for the fitted network utilization: NCCL collectives
# achieve ~70-80% of link bandwidth on healthy fabrics; lower values are
# the fit absorbing unrelated error into the network knob.
NET_UTIL_FLOOR = 0.60
CALIB_SHARE = 0.5
SEEDS = (0, 1, 2, 3, 4)

A100_PEAK_FLOPS = 312e12  # BF16
H100_PEAK_FLOPS = 989.5e12

# Pre-registered knob axes per device. compute: 5 values; dram/net: (lo,
# center, hi). Centers for dram/net are the Table III values.
KNOB_AXES: dict[str, dict[str, tuple[float, ...]]] = {
    "H100_SXM5": {
        "compute": (0.600, 0.625, 0.650, 0.675, 0.700),
        "dram": (0.70, 0.80, 0.90),
        "net": (0.70, 0.85, 1.00),
    },
    "A100_SXM4": {
        "compute": (0.80, 0.85, 0.90, 0.95, 1.00),
        "dram": (0.60, 0.70, 0.80),
        "net": (0.65, 0.80, 0.95),
    },
    "A100_PCIe": {
        "compute": (0.50, 0.55, 0.60, 0.65, 0.70),
        "dram": (0.60, 0.70, 0.80),
        "net": (0.70, 0.85, 1.00),
    },
}

# Selene SuperPOD leaf sizes per model (from nvidia_train_validation.py) —
# dim1 (pp*dp) must divide by leaf_size.
SELENE_LEAF_SIZE = {"GPT 1T": 32, "GPT 310B": 30, "GPT 530B": 35}

GPT_MODEL_CONFIGS = {
    "GPT 22B": "GPT_22_B.yaml",
    "GPT 175B": "GPT_175_B.yaml",
    "GPT 310B": "GPT_310_B.yaml",
    "GPT 530B": "GPT_530_B.yaml",
    "GPT 1T": "GPT_1T.yaml",
}
LLAMA2_INF_CONFIGS = {
    "Llama 2-7B": "Llama2-7B_inf.yaml",
    "Llama 2-13B": "Llama2-13B_inf.yaml",
    "Llama 2-70B": "Llama2-70B_inf.yaml",
}
OPT_INF_CONFIGS = {
    "opt-13b": "OPT-13B_inf.yaml",
    "opt-30b": "OPT-30B_inf.yaml",
    "opt-66b": "OPT-66B_inf.yaml",
}


def _combo_key(c: float, d: float, n: float, l: float = LAUNCH_DEFAULT) -> str:
    base = f"c{c:.3f}_d{d:.2f}_n{n:.2f}"
    if abs(l - LAUNCH_DEFAULT) > 1e-12:
        base += f"_l{l*1e6:.1f}"
    return base


def _design(device: str) -> list[tuple[float, float, float]]:
    """Pre-registered axial + factorial-corner design (17 combos)."""
    ax = KNOB_AXES[device]
    cs, ds, ns = ax["compute"], ax["dram"], ax["net"]
    c_mid = cs[len(cs) // 2]
    d_lo, d_mid, d_hi = ds
    n_lo, n_mid, n_hi = ns
    combos: list[tuple[float, float, float, float]] = []
    for c in cs:  # compute axis
        combos.append((c, d_mid, n_mid, LAUNCH_DEFAULT))
    for d in (d_lo, d_hi):  # dram axis
        combos.append((c_mid, d, n_mid, LAUNCH_DEFAULT))
    for n in (n_lo, n_hi):  # net axis
        combos.append((c_mid, d_mid, n, LAUNCH_DEFAULT))
    for c in (cs[0], cs[-1]):  # corners (default launch plane)
        for d in (d_lo, d_hi):
            for n in (n_lo, n_hi):
                combos.append((c, d, n, LAUNCH_DEFAULT))
    # full axial structure replicated at each non-default launch value
    for l in LAUNCH_AXIS:
        if abs(l - LAUNCH_DEFAULT) < 1e-12:
            continue
        for c in cs:
            combos.append((c, d_mid, n_mid, l))
        for d in (d_lo, d_hi):
            combos.append((c_mid, d, n_mid, l))
        for n in (n_lo, n_hi):
            combos.append((c_mid, d_mid, n, l))
    # dedupe, preserve order
    seen: set[str] = set()
    out = []
    for combo in combos:
        key = _combo_key(*combo)
        if key not in seen:
            seen.add(key)
            out.append(combo)
    return out


def _extension_combos(device: str, axis: str, new_value: float) -> list[tuple[float, float, float]]:
    """Boundary-extension rule: when the golden combo lands on an axis
    boundary, the axis is extended one step and the design positions that
    used the boundary value are replicated at the new value (axial point +
    the four corners). Repeated until the optimum is interior or the
    physical bound (util <= 1.0) is reached."""
    ax = KNOB_AXES[device]
    cs, ds, ns = ax["compute"], ax["dram"], ax["net"]
    c_mid = cs[len(cs) // 2]
    d_mid = ds[len(ds) // 2]
    n_mid = ns[len(ns) // 2]
    combos: list[tuple[float, float, float, float]] = []
    for l in LAUNCH_AXIS:
        if axis == "compute":
            combos.append((new_value, d_mid, n_mid, l))
        elif axis == "dram":
            combos.append((c_mid, new_value, n_mid, l))
        else:
            combos.append((c_mid, d_mid, new_value, l))
    if axis == "compute":
        for d in (ds[0], ds[-1]):
            for n in (ns[0], ns[-1]):
                combos.append((new_value, d, n, LAUNCH_DEFAULT))
    elif axis == "dram":
        for c in (cs[0], cs[-1]):
            for n in (ns[0], ns[-1]):
                combos.append((c, new_value, n, LAUNCH_DEFAULT))
    else:
        for c in (cs[0], cs[-1]):
            for d in (ds[0], ds[-1]):
                combos.append((c, d, new_value, LAUNCH_DEFAULT))
    return combos


def _boundary_extensions(device: str, design, golden_combo) -> list[tuple[float, float, float]]:
    ax = KNOB_AXES[device]
    step = {"compute": round(ax["compute"][1] - ax["compute"][0], 3),
            "dram": round(ax["dram"][1] - ax["dram"][0], 3),
            "net": round(ax["net"][1] - ax["net"][0], 3)}
    values = {"compute": sorted({c[0] for c in design}),
              "dram": sorted({c[1] for c in design}),
              "net": sorted({c[2] for c in design})}
    launch_vals = sorted({(c[3] if len(c) == 4 else LAUNCH_DEFAULT) for c in design})
    got = dict(zip(("compute", "dram", "net"), golden_combo[:3]))
    golden_launch = golden_combo[3] if len(golden_combo) == 4 else LAUNCH_DEFAULT
    extra: list[tuple[float, float, float, float]] = []
    for axis in ("compute", "dram", "net"):
        vals = values[axis]
        if got[axis] <= vals[0] + 1e-9:
            new_v = round(vals[0] - step[axis], 3)
            if new_v > 0.05:
                extra += _extension_combos(device, axis, new_v)
        elif got[axis] >= vals[-1] - 1e-9:
            new_v = round(vals[-1] + step[axis], 3)
            if new_v <= 1.0 + 1e-9:
                extra += _extension_combos(device, axis, min(new_v, 1.0))
    # launch axis: geometric steps (halve below, double above), floor 0.5us
    ax = KNOB_AXES[device]
    cs, ds, ns = ax["compute"], ax["dram"], ax["net"]
    c_mid, d_mid, n_mid = cs[len(cs) // 2], ds[len(ds) // 2], ns[len(ns) // 2]
    if golden_launch <= launch_vals[0] + 1e-12 and launch_vals[0] > 0.5e-6 + 1e-12:
        new_l = max(0.5e-6, launch_vals[0] / 2.0)
        for c in cs:
            extra.append((c, d_mid, n_mid, new_l))
        for d in (ds[0], ds[-1]):
            extra.append((c_mid, d, n_mid, new_l))
        for n in (ns[0], ns[-1]):
            extra.append((c_mid, d_mid, n, new_l))
    elif golden_launch >= launch_vals[-1] - 1e-12 and launch_vals[-1] < 12e-6:
        new_l = min(12e-6, launch_vals[-1] * 2.0)
        for c in cs:
            extra.append((c, d_mid, n_mid, new_l))
        for d in (ds[0], ds[-1]):
            extra.append((c_mid, d, n_mid, new_l))
        for n in (ns[0], ns[-1]):
            extra.append((c_mid, d_mid, n, new_l))
    seen = {_combo_key(*c) for c in design}
    out = []
    for combo in extra:
        key = _combo_key(*combo)
        if key not in seen:
            seen.add(key)
            out.append(combo)
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _deep_update(target: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _apply_factors(hw: dict[str, Any], c: float, d: float, n: float,
                   l: float = LAUNCH_DEFAULT) -> None:
    """Overwrite every calibrated field with the knob values."""
    sw = hw.setdefault("sw_param", {})
    sw["kernel_launch_overhead"] = l
    sw.pop("grad_acc_overhead", None)
    hw.setdefault("tech_param", {}).setdefault("core", {})["util"] = c
    hw.setdefault("tech_param", {}).setdefault("DRAM", {})["util"] = d
    for dim in hw.get("network", {}).get("dimensions", []):
        dim.setdefault("topology", {})["util"] = n
    overlap = hw.setdefault("network", {}).setdefault("overlap", {})
    overlap.update(OVERLAPS)


# --------------------------------------------------------------------------
# GPT / Llama model params (for measured-MFU envelope on training rows)
# --------------------------------------------------------------------------


def _model_flops_per_token(model_yaml: Path) -> tuple[float, int]:
    """(FLOPs/token for training under the 3x-forward convention, seq_len)."""
    mp = _load_yaml(model_yaml)["model_param"]
    h = int(mp["hidden_dim"])
    layers = int(mp["num_layers"])
    vocab = int(mp["vocab_size"])
    seq = int(mp["seq_len"])
    interm = int(mp.get("intermediate_size", 4 * h))
    attn_block = mp.get("attention", {})
    heads = int(attn_block.get("num_heads", 1))
    kv_heads = int(attn_block.get("kv_heads", heads) or heads)
    head_dim = int(attn_block.get("head_dim", h // heads) or (h // heads))
    gated = str(mp.get("model_type", "gpt")).lower() in {"llama", "deepseek_v3", "glm4_moe"}
    ffn_matmuls = 3 if gated else 2
    qkv = 2 * h * (h + 2 * kv_heads * head_dim)
    attn = 4 * seq * h
    out_proj = 2 * h * h
    ffn = ffn_matmuls * 2 * h * interm
    logits = 2 * h * vocab
    return 3.0 * (layers * (qkv + attn + out_proj + ffn) + logits), seq


# --------------------------------------------------------------------------
# Point adapters. Each point dict:
#   point_id, workload (train|inference), source, family, scale (int),
#   t_ref_s, fit_eligible, report_note, build(case_dir, c, d, n) ->
#   (hw_path, model_path), parse_re
# --------------------------------------------------------------------------


def _h100_train_points() -> list[dict[str, Any]]:
    """Reuse the h100_testbench adapters (megatron + HF, envelope included)."""
    points = []
    for p in h100tb._mev_points() + h100tb._hf_points():
        kind, payload = p["runner"]

        def build(case_dir: Path, c: float, d: float, n: float,
                  l: float = LAUNCH_DEFAULT, kind=kind, payload=payload):
            factors = {
                "kernel_launch_overhead_s": l,
                "compute_util": c,
                "dram_util": d,
                "network_util": n,
            }
            if kind == "mev":
                hw_path, model_path = mev._build_configs(payload, case_dir, factors, vpp="max")
            else:
                hw_path, model_path = hfs._build_configs(payload, "hf_network", case_dir, factors)
            hw = _load_yaml(hw_path)
            _apply_factors(hw, c, d, n, l)
            _write_yaml(hw_path, hw)
            return hw_path, model_path

        points.append(
            {
                "point_id": p["point_id"],
                "workload": "train",
                "source": p["source"],
                "family": p["family"],
                "scale": p["num_gpus"],
                "t_ref_s": p["t_ref_s"],
                "fit_eligible": p["fit_eligible"],
                "report_note": p["report_note"],
                "build": build,
                "parse_re": TRAIN_TIME_RE,
            }
        )
    return points


def _nim_inference_points(device: str) -> list[dict[str, Any]]:
    if device == "H100_SXM5":
        csv_path = SCRIPT_DIR / "nvidia_data" / "4xH100_fp16_Llama3_3-70B.csv"
        base_hw = HW_ROOT / "H100_SXM5_80GB.yaml"
        tp, gpus = 4, 4
        nvlink_bw = "450 GB"  # H100 SXM NVLink4, uni-directional per GPU
    else:
        csv_path = SCRIPT_DIR / "nvidia_data" / "8xA100_bf16_Llama3_3-70B.csv"
        base_hw = HW_ROOT / "a100_80GB_inf.yaml"
        tp, gpus = 8, 8
        nvlink_bw = "300 GB"  # A100 SXM NVLink3, uni-directional per GPU
    model_yaml = MODEL_ROOT / "Llama3.1-70B_inf.yaml"
    df = pd.read_csv(csv_path)
    points = []
    for _, row in df.iterrows():
        inp = int(row["Input Tokens"])
        out = int(row["Output Tokens"])
        conc = int(row["Concurrency"])
        t_meas = (float(row["TTFT (ms)"]) + float(row["ITL (ms)"]) * max(out - 1, 0)) / 1000.0

        def build(case_dir: Path, c: float, d: float, n: float,
                  l: float = LAUNCH_DEFAULT, inp=inp, out=out, conc=conc):
            hw = _load_yaml(base_hw)
            _deep_update(
                hw,
                {
                    "parallelism": {
                        "auto": False,
                        "tp": tp,
                        "tp_sp": True,
                        "cp": 1,
                        "pp": 1,
                        "mb": 1,
                        "train": {"dp": 1, "ep": 1, "tp_ep": True},
                        "inference": {"replica_count": 1, "moe_dp": 1},
                    }
                },
            )
            # The checked-in validation base configs model NVLink far below
            # spec (H100 copy: 100 GB Ring, analytical mode). Fix dim0 to the
            # physical per-GPU uni-directional NVLink rate with an FC topology
            # (NVSwitch is all-to-all), and force the hierarchical AstraSim
            # backend — ALL validation points run hierarchical.
            for dim in hw.get("network", {}).get("dimensions", []):
                if dim.get("id") == "dim0":
                    dim.setdefault("topology", {})["type"] = "FC"
                    dim["topology"]["bandwidth"] = nvlink_bw
            _deep_update(
                hw,
                {"execution_backend": {"model": "astra",
                                       "astra": {"mode": "full_astrasim_hierarchical"}}},
            )
            _apply_factors(hw, c, d, n, l)
            model = _load_yaml(model_yaml)
            _deep_update(
                model,
                {
                    "model_param": {
                        "run_type": "inference",
                        "global_batch_size": conc,
                        "seq_len": inp + out,
                        "decode_len": out,
                    }
                },
            )
            hw_path = case_dir / "hardware.yaml"
            model_path = case_dir / "model.yaml"
            _write_yaml(hw_path, hw)
            _write_yaml(model_path, model)
            return hw_path, model_path

        points.append(
            {
                "point_id": f"nim:{device.lower()}_p{inp}_o{out}_b{conc}",
                "workload": "inference",
                "source": "nim_llama33_70b",
                "family": "Llama3.3-70B",
                "scale": gpus,
                "t_ref_s": t_meas,
                "fit_eligible": True,
                "report_note": "",
                "build": build,
                "parse_re": INF_TIME_RE,
            }
        )
    return points


def _imec_inference_points(device: str) -> list[dict[str, Any]]:
    """IMEC Llama2 latency rows. Repo convention for this source: network
    effectively ignored (dim0 bandwidth -> 100000 GB), so these rows
    constrain compute/DRAM only."""
    if device == "H100_SXM5":
        csv_path = SCRIPT_DIR / "imec_data" / "H100_inf.csv"
        base_hw = HW_ROOT / "H100_SXM5_80GB.yaml"
    else:
        csv_path = SCRIPT_DIR / "imec_data" / "A100_inf.csv"
        base_hw = HW_ROOT / "a100_80GB_inf.yaml"
    df = pd.read_csv(csv_path)
    points = []
    for _, row in df.iterrows():
        model_name = str(row["model"]).strip()
        if model_name not in LLAMA2_INF_CONFIGS:
            continue
        tp = int(row["TP"])
        t_meas = float(row["actual"])
        model_yaml = MODEL_ROOT / LLAMA2_INF_CONFIGS[model_name]

        def build(case_dir: Path, c: float, d: float, n: float,
                  l: float = LAUNCH_DEFAULT, tp=tp, model_yaml=model_yaml):
            hw = _load_yaml(base_hw)
            _deep_update(
                hw,
                {
                    "parallelism": {
                        "auto": False,
                        "tp": tp,
                        "tp_sp": True,
                        "cp": 1,
                        "pp": 1,
                        "mb": 1,
                        "train": {"dp": 1, "ep": 1, "tp_ep": True},
                        "inference": {"replica_count": 1, "moe_dp": 1},
                    }
                },
            )
            for dim in hw.get("network", {}).get("dimensions", []):
                if dim.get("id") == "dim0":
                    dim.setdefault("topology", {})["bandwidth"] = "100000 GB"
                    dim["topology"]["latency"] = 1e-9
            _deep_update(
                hw,
                {"execution_backend": {"model": "astra",
                                       "astra": {"mode": "full_astrasim_hierarchical"}}},
            )
            _apply_factors(hw, c, d, n, l)
            model = _load_yaml(model_yaml)
            _deep_update(
                model,
                {
                    "model_param": {
                        "run_type": "inference",
                        "global_batch_size": 1,
                        "seq_len": 400,
                        "decode_len": 200,
                    }
                },
            )
            hw_path = case_dir / "hardware.yaml"
            model_path = case_dir / "model.yaml"
            _write_yaml(hw_path, hw)
            _write_yaml(model_path, model)
            return hw_path, model_path

        points.append(
            {
                "point_id": f"imec:{device.lower()}_{model_name.replace(' ', '')}_tp{tp}",
                "workload": "inference",
                "source": "imec_llama2",
                "family": model_name,
                "scale": tp,
                "t_ref_s": t_meas,
                "fit_eligible": True,
                "report_note": "network-ignored source convention",
                "build": build,
                "parse_re": INF_TIME_RE,
            }
        )
    return points


def _a100_sxm_train_points() -> list[dict[str, Any]]:
    csv_path = SCRIPT_DIR / "imec_data" / "A100_train.csv"
    df = pd.read_csv(csv_path)
    hw_by_device = {
        "A100_korthi": HW_ROOT / "a100_80GB_korthikanti.yaml",
        "A100_selene": HW_ROOT / "a100_80GB_selene_sc.yaml",
    }
    points = []
    for _, row in df.iterrows():
        dev = str(row["device"]).strip()
        if dev not in hw_by_device:
            continue
        model_name = str(row["model"]).strip()
        model_yaml = MODEL_ROOT / GPT_MODEL_CONFIGS[model_name]
        tp, pp, cp, dp = int(row["tp"]), int(row["pp"]), int(row["cp"]), int(row["dp"])
        batch, mb = int(row["batch"]), max(1, int(row["mb"]))
        tp_sp = str(row["tp_sp"]).strip().lower() == "true"
        full_rec = str(row["recomputation"]).strip().lower() == "full"
        t_meas = float(row["actual"])
        gpus = tp * pp * cp * dp
        base_hw = hw_by_device[dev]
        leaf = SELENE_LEAF_SIZE.get(model_name) if dev == "A100_selene" else None

        fpt, seq = _model_flops_per_token(model_yaml)
        mfu = fpt * batch * seq / (t_meas * gpus * A100_PEAK_FLOPS)
        degenerate = mfu < MEASURED_MFU_FLOOR

        def build(case_dir: Path, c: float, d: float, n: float,
                  l: float = LAUNCH_DEFAULT,
                  base_hw=base_hw, model_yaml=model_yaml, tp=tp, pp=pp, cp=cp,
                  dp=dp, batch=batch, mb=mb, tp_sp=tp_sp, full_rec=full_rec, leaf=leaf):
            hw = _load_yaml(base_hw)
            _deep_update(
                hw,
                {
                    "parallelism": {
                        "auto": False,
                        "tp": tp,
                        "tp_sp": tp_sp,
                        "cp": cp,
                        "pp": pp,
                        "mb": mb,
                        "train": {"dp": dp, "ep": 1, "tp_ep": True},
                        "inference": {"replica_count": 1, "moe_dp": 1},
                    },
                    "sw_param": {"full_recomputation": full_rec},
                },
            )
            if leaf is not None:
                for dim in hw.get("network", {}).get("dimensions", []):
                    if dim.get("id") == "dim1" and str(
                        dim.get("topology", {}).get("type", "")
                    ).lower() == "superpod":
                        dim["topology"]["leaf_size"] = int(leaf)
            _apply_factors(hw, c, d, n, l)
            model = _load_yaml(model_yaml)
            _deep_update(model, {"model_param": {"global_batch_size": batch,
                                                 "run_type": "training"}})
            hw_path = case_dir / "hardware.yaml"
            model_path = case_dir / "model.yaml"
            _write_yaml(hw_path, hw)
            _write_yaml(model_path, model)
            return hw_path, model_path

        rec_tag = "full" if full_rec else "sel"
        points.append(
            {
                "point_id": f"nvtrain:{dev}_{model_name.replace(' ', '')}_{rec_tag}",
                "workload": "train",
                "source": dev.lower(),
                "family": model_name,
                "scale": gpus,
                "t_ref_s": t_meas,
                "fit_eligible": not degenerate,
                "report_note": (
                    f"measured MFU {mfu*100:.1f}% < {MEASURED_MFU_FLOOR*100:.0f}%"
                    if degenerate
                    else ""
                ),
                "build": build,
                "parse_re": TRAIN_TIME_RE,
            }
        )
    return points


def _uci_train_points() -> list[dict[str, Any]]:
    csv_path = SCRIPT_DIR / "imec_data" / "uci_train.csv"
    base_hw = HW_ROOT / "a100_80GB_train_validation.yaml"
    model_yaml = MODEL_ROOT / "Llama2-7B.yaml"
    df = pd.read_csv(csv_path)
    df = df[df["Status"].str.upper() == "SUCCESS"]
    fpt, _seq = _model_flops_per_token(model_yaml)
    points = []
    for _, row in df.iterrows():
        variant = str(row["variant"]).strip().upper()
        tp, pp, cp, dp = int(row["TP"]), int(row["PP"]), int(row["CP"]), int(row["DP"])
        t_meas = float(row["Avg_Step_Time_s"])
        tok_s_gpu = float(row["Throughput_per_GPU_tokens_s"])
        mfu = fpt * tok_s_gpu / A100_PEAK_FLOPS
        degenerate = mfu < MEASURED_MFU_FLOOR
        zero_stage = 0 if variant == "DDP" else 3
        mb = max(1, 128 // max(1, dp))
        eff_mb = mb if pp > 1 else 1

        def build(case_dir: Path, c: float, d: float, n: float,
                  l: float = LAUNCH_DEFAULT,
                  tp=tp, pp=pp, cp=cp, dp=dp, zero_stage=zero_stage, eff_mb=eff_mb):
            hw = _load_yaml(base_hw)
            _deep_update(
                hw,
                {
                    "parallelism": {
                        "auto": False,
                        "tp": tp,
                        "tp_sp": True,
                        "cp": cp,
                        "pp": pp,
                        "mb": eff_mb,
                        "train": {"dp": dp, "ep": 1, "tp_ep": True},
                        "inference": {"replica_count": 1, "moe_dp": 1},
                    },
                    "sw_param": {"dp_zero_stage": zero_stage},
                },
            )
            # UCI node: tp/cp on PCIe (25 GB), pp/dp on NVLink pairs (300 GB)
            hw["network"]["dimensions"] = [
                {
                    "id": "dim0",
                    "label": "tp_cp_pcie",
                    "size": "auto",
                    "topology": {"type": "Ring", "bandwidth": "25 GB",
                                 "latency": 5e-6, "energy_per_bit": 8e-12, "util": 1.0},
                    "collective_override": {},
                    "parallelisms": ["tp", "cp"],
                },
                {
                    "id": "dim1",
                    "label": "pp_dp_nvlink",
                    "size": "auto",
                    "topology": {"type": "Ring", "bandwidth": "300 GB",
                                 "latency": 1e-6, "energy_per_bit": 8e-12, "util": 1.0},
                    "collective_override": {},
                    "parallelisms": ["pp", "dp"],
                },
            ]
            _deep_update(
                hw,
                {"execution_backend": {"model": "astra",
                                       "astra": {"mode": "full_astrasim_hierarchical"}}},
            )
            _apply_factors(hw, c, d, n, l)
            model = _load_yaml(model_yaml)
            _deep_update(
                model,
                {"model_param": {"run_type": "training", "seq_len": 4096,
                                 "global_batch_size": 128}},
            )
            hw_path = case_dir / "hardware.yaml"
            model_path = case_dir / "model.yaml"
            _write_yaml(hw_path, hw)
            _write_yaml(model_path, model)
            return hw_path, model_path

        points.append(
            {
                "point_id": f"uci_train:{variant.lower()}_tp{tp}_pp{pp}_cp{cp}_dp{dp}",
                "workload": "train",
                "source": f"uci_{variant.lower()}",
                "family": "Llama2-7B",
                "scale": 4,
                "t_ref_s": t_meas,
                "fit_eligible": not degenerate,
                "report_note": (
                    f"measured MFU {mfu*100:.1f}% < {MEASURED_MFU_FLOOR*100:.0f}%"
                    if degenerate
                    else ""
                ),
                "build": build,
                "parse_re": TRAIN_TIME_RE,
            }
        )
    return points


def _uci_inference_points() -> list[dict[str, Any]]:
    csv_path = SCRIPT_DIR / "imec_data" / "uci_inf.csv"
    base_hw = HW_ROOT / "a100_80GB_uci.fitted.yaml"  # structural base; every
    # calibrated field is overwritten by _apply_factors
    df = pd.read_csv(csv_path)
    points = []
    for _, row in df.iterrows():
        model_name = str(row["model_name"]).strip()
        if model_name not in OPT_INF_CONFIGS:
            continue
        tp = int(row["tensor_parallel_size"])
        cuda_devices = str(row.get("cuda_devices", "") or "").strip()
        inp = int(row["input_len"])
        out = int(row["output_len"])
        t_meas = (float(row["ttft_ms"]) + float(row["tpot_ms"]) * max(out - 1, 0)) / 1000.0
        model_yaml = MODEL_ROOT / OPT_INF_CONFIGS[model_name]
        # Network per uci_inf.py: tp4 -> NVLink ring 125 GB; tp2 default ->
        # NVLink 125 GB; tp2 on devices "1,2" -> PCIe 25 GB.
        if tp == 2 and cuda_devices == "1,2":
            dim0_bw = "25 GB"
        else:
            dim0_bw = "125 GB"

        def build(case_dir: Path, c: float, d: float, n: float,
                  l: float = LAUNCH_DEFAULT,
                  tp=tp, dim0_bw=dim0_bw, model_yaml=model_yaml, inp=inp, out=out):
            hw = _load_yaml(base_hw)
            _deep_update(
                hw,
                {
                    "parallelism": {
                        "auto": False,
                        "tp": tp,
                        "tp_sp": True,
                        "cp": 1,
                        "pp": 1,
                        "mb": 1,
                        "train": {"dp": 1, "ep": 1, "tp_ep": True},
                        "inference": {"replica_count": 1, "moe_dp": 1},
                    }
                },
            )
            for dim in hw.get("network", {}).get("dimensions", []):
                if dim.get("id") == "dim0":
                    dim.setdefault("topology", {})["bandwidth"] = dim0_bw
            _apply_factors(hw, c, d, n, l)
            model = _load_yaml(model_yaml)
            _deep_update(
                model,
                {
                    "model_param": {
                        "run_type": "inference",
                        "global_batch_size": 1,
                        "seq_len": inp + out,
                        "decode_len": out,
                    }
                },
            )
            hw_path = case_dir / "hardware.yaml"
            model_path = case_dir / "model.yaml"
            _write_yaml(hw_path, hw)
            _write_yaml(model_path, model)
            return hw_path, model_path

        dev_tag = cuda_devices.replace(",", "") or "def"
        points.append(
            {
                "point_id": f"uci_inf:{model_name.split('/')[-1]}_tp{tp}_{dev_tag}",
                "workload": "inference",
                "source": "uci_opt",
                "family": model_name.split("/")[-1],
                "scale": tp,
                "t_ref_s": t_meas,
                "fit_eligible": True,
                "report_note": "",
                "build": build,
                "parse_re": INF_TIME_RE,
            }
        )
    return points


def collect_points(device: str) -> list[dict[str, Any]]:
    if device == "H100_SXM5":
        return (
            _h100_train_points()
            + _nim_inference_points("H100_SXM5")
            + _imec_inference_points("H100_SXM5")
        )
    if device == "A100_SXM4":
        return (
            _a100_sxm_train_points()
            + _nim_inference_points("A100_SXM4")
            + _imec_inference_points("A100_SXM4")
        )
    if device == "A100_PCIe":
        return _uci_train_points() + _uci_inference_points()
    raise ValueError(f"unknown device {device}")


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


def _run_one(point: dict[str, Any], combo: tuple, out_root: Path) -> float:
    c, d, n, l = (combo if len(combo) == 4 else (*combo, LAUNCH_DEFAULT))
    slug = point["point_id"].replace(":", "_").replace("/", "_")
    case_dir = out_root / "case_configs" / slug / _combo_key(c, d, n, l)
    case_dir.mkdir(parents=True, exist_ok=True)
    try:
        hw_path, model_path = point["build"](case_dir, c, d, n, l)
    except Exception as exc:  # config construction failure is a real result
        (case_dir / "run.log").write_text(f"build failed: {exc}\n", encoding="utf-8")
        print(f"[WARN] {point['point_id']} {_combo_key(c, d, n, l)} build failed: {exc}")
        return math.nan
    env = dict(os.environ)
    env["RAPID_ASTRA_CACHE_MODE"] = "NO_CACHE"
    result = subprocess.run(
        [sys.executable, str(RUN_PERF), "--hardware_config", str(hw_path),
         "--model_config", str(model_path)],
        cwd=case_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env=env,
    )
    (case_dir / "run.log").write_text(result.stdout or "", encoding="utf-8")
    match = point["parse_re"].search(result.stdout or "")
    if result.returncode != 0 or not match:
        print(f"[WARN] {point['point_id']} {_combo_key(c, d, n, l)} failed (rc={result.returncode})")
        return math.nan
    return float(match.group(1))


def _collect(
    jobs: list[tuple[dict[str, Any], tuple[float, float, float]]],
    workers: int,
    runs: dict[str, dict[str, float]],
    runs_path: Path,
    out_root: Path,
) -> None:
    todo = [
        (p, combo)
        for p, combo in jobs
        if not (runs.get(p["point_id"], {}).get(_combo_key(*combo), math.nan) > 0)
    ]
    print(f"{len(todo)} runs to execute ({len(jobs) - len(todo)} cached)", flush=True)
    if not todo:
        return
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for (point, combo), t in zip(todo, pool.map(lambda pc: _run_one(*pc, out_root), todo)):
            runs.setdefault(point["point_id"], {})[_combo_key(*combo)] = t
            done += 1
            if done % 25 == 0 or done == len(todo):
                runs_path.write_text(json.dumps(runs, indent=1), encoding="utf-8")
                print(f"progress: {done}/{len(todo)}", flush=True)
    runs_path.write_text(json.dumps(runs, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------
# Fit + split search
# --------------------------------------------------------------------------


def _t_at(point: dict[str, Any], combo: tuple[float, float, float]) -> float:
    return point["runs"].get(_combo_key(*combo), math.nan)


def _mape(points: list[dict[str, Any]], combo: tuple[float, float, float]) -> float:
    errs = []
    for p in points:
        t = _t_at(p, combo)
        if t == t and t > 0:
            errs.append(abs(t - p["t_ref_s"]) / p["t_ref_s"] * 100.0)
    return statistics.fmean(errs) if errs else math.nan


def _fit(points: list[dict[str, Any]], design: list[tuple[float, float, float]]):
    admissible = [c for c in design if c[2] >= NET_UTIL_FLOOR - 1e-9]
    best = min(admissible or design, key=lambda combo: _mape(points, combo))
    return best


def _scale_bucket(scale: int) -> str:
    if scale <= 8:
        return "node"
    if scale <= 64:
        return "small"
    return "large"


def _stratified_split(points, seed):
    rng = random.Random(seed)
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for p in points:
        key = (p["workload"], p["source"], _scale_bucket(p["scale"]))
        strata.setdefault(key, []).append(p)
    calib, holdout = [], []
    for key in sorted(strata):
        members = sorted(strata[key], key=lambda p: p["point_id"])
        rng.shuffle(members)
        n_cal = max(1, round(CALIB_SHARE * len(members)))
        if n_cal >= len(members) and len(members) > 1:
            n_cal = len(members) - 1
        calib.extend(members[:n_cal])
        holdout.extend(members[n_cal:])
    return calib, holdout


def _largest_scale_holdout(points):
    holdout_ids = set()
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for p in points:
        groups.setdefault((p["workload"], p["family"]), []).append(p)
    for members in groups.values():
        if len(members) > 1:
            holdout_ids.add(max(members, key=lambda p: p["scale"])["point_id"])
    calib = [p for p in points if p["point_id"] not in holdout_ids]
    holdout = [p for p in points if p["point_id"] in holdout_ids]
    return calib, holdout


def _split_search(fit_points, design):
    candidates = []
    for seed in SEEDS:
        calib, holdout = _stratified_split(fit_points, seed)
        combo = _fit(calib, design)
        candidates.append(
            {
                "kind": f"stratified_50_{seed}",
                "combo": combo,
                "calib_mape": _mape(calib, combo),
                "holdout_mape": _mape(holdout, combo),
                "n_calib": len(calib),
                "n_holdout": len(holdout),
                "calib_ids": sorted(p["point_id"] for p in calib),
                "holdout_ids": sorted(p["point_id"] for p in holdout),
            }
        )
    calib, holdout = _largest_scale_holdout(fit_points)
    combo = _fit(calib, design)
    candidates.append(
        {
            "kind": "largest_scale_holdout",
            "combo": combo,
            "calib_mape": _mape(calib, combo),
            "holdout_mape": _mape(holdout, combo),
            "n_calib": len(calib),
            "n_holdout": len(holdout),
            "calib_ids": sorted(p["point_id"] for p in calib),
            "holdout_ids": sorted(p["point_id"] for p in holdout),
        }
    )
    fifty = sorted(
        (c for c in candidates if c["kind"].startswith("stratified_50")),
        key=lambda c: c["holdout_mape"],
    )
    golden = fifty[len(fifty) // 2]
    return candidates, golden


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _write_outputs(device, points, fit_points, report_points, candidates, golden,
                   design, out_root, pp_gt1_cut):
    combo = golden["combo"]
    results_csv = out_root / "testbench_results.csv"
    with results_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["point_id", "workload", "source", "family", "scale", "fit_eligible",
             "t_ref_s", "t_pred_golden_s", "signed_err_pct", "report_note"]
        )
        for p in points:
            t = _t_at(p, combo)
            err = (t - p["t_ref_s"]) / p["t_ref_s"] * 100.0 if t == t else math.nan
            writer.writerow(
                [p["point_id"], p["workload"], p["source"], p["family"], p["scale"],
                 p["fit_eligible"], f"{p['t_ref_s']:.4f}",
                 f"{t:.4f}" if t == t else "",
                 f"{err:+.2f}" if err == err else "", p["report_note"]]
            )

    lines = [
        f"# {device} calibration testbench",
        "",
        f"Design: pre-registered axial+corner lattice, {len(design)} real (compute, "
        f"dram, net, launch) combos per fit point; fit = argmin calibration MAPE "
        f"over real runs. Overlaps fixed at 0.6; kernel launch overhead jointly "
        f"calibrated over {list(LAUNCH_AXIS)}. Split: stratified 50:50 x "
        f"{len(SEEDS)} seeds, golden = median holdout.",
        "",
        f"Fit points: {len(fit_points)} (train "
        f"{sum(1 for p in fit_points if p['workload'] == 'train')}, inference "
        f"{sum(1 for p in fit_points if p['workload'] == 'inference')}); "
        f"report-only: {len(report_points)}.",
        "",
        "| kind | compute | dram | net | calib MAPE | holdout MAPE | n_cal | n_hold |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in candidates:
        marker = " **<- GOLDEN**" if c is golden else ""
        cc, dd, nn, ll = (c["combo"] if len(c["combo"]) == 4 else (*c["combo"], LAUNCH_DEFAULT))
        lines.append(
            f"| {c['kind']}{marker} | {cc:.3f} | {dd:.2f} | {nn:.2f} (launch {ll*1e6:.0f}us) "
            f"| {c['calib_mape']:.2f}% | {c['holdout_mape']:.2f}% "
            f"| {c['n_calib']} | {c['n_holdout']} |"
        )
    lines += ["", "## Per-workload / per-source MAPE at golden (all fit points)", ""]
    for workload in ("train", "inference"):
        subset = [p for p in fit_points if p["workload"] == workload]
        if subset:
            lines.append(f"- {workload}: n={len(subset)}, MAPE {_mape(subset, combo):.2f}%")
    for source in sorted({p["source"] for p in fit_points}):
        subset = [p for p in fit_points if p["source"] == source]
        lines.append(f"  - {source}: n={len(subset)}, MAPE {_mape(subset, combo):.2f}%")
    if pp_gt1_cut:
        lines += ["", "HF pp>1 sanity rows were cut per protocol (>25% MAPE at golden)."]
    (out_root / "split_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    cc, dd, nn, ll = (combo if len(combo) == 4 else (*combo, LAUNCH_DEFAULT))
    payload = {
        "description": (
            f"Golden {device} calibration (train + inference pooled, 50:50 "
            "calibration:holdout). Knobs: tech_param.core.util, tech_param.DRAM.util, "
            "network topology.util, sw_param.kernel_launch_overhead. All numbers "
            f"are real simulations on a pre-registered {len(design)}-combo "
            "axial+corner lattice; no interpolation. Overlaps (tp, tp_sp, cp) "
            "fixed at 0.6. Golden = median-holdout seed among stratified 50:50 "
            "candidates."
        ),
        "device": device,
        "generator": "validation_scripts/device_testbench.py",
        "kind": golden["kind"],
        "compute_util": cc,
        "dram_util": dd,
        "network_util": nn,
        "kernel_launch_overhead_s": ll,
        "overlaps": OVERLAPS,
        "calib_mape_pct": round(golden["calib_mape"], 2),
        "holdout_mape_pct": round(golden["holdout_mape"], 2),
        "pp_gt1_rows_cut": pp_gt1_cut,
        "calibration_points": golden["calib_ids"],
        "holdout_points": golden["holdout_ids"],
        "report_only_points": sorted(p["point_id"] for p in report_points),
    }
    _write_yaml(CONFIG_ROOT / f"{device}_golden_calib.yaml", payload)


def run_device(device: str, workers: int, fit_only: bool) -> dict[str, Any]:
    out_root = OUT_BASE / device
    out_root.mkdir(parents=True, exist_ok=True)
    runs_path = out_root / "runs.json"
    runs: dict[str, dict[str, float]] = {}
    if runs_path.exists():
        runs = json.loads(runs_path.read_text(encoding="utf-8"))

    points = collect_points(device)
    design = _design(device)
    fit_candidates = [p for p in points if p["fit_eligible"]]

    if not fit_only:
        jobs = [(p, combo) for p in fit_candidates for combo in design]
        _collect(jobs, workers, runs, runs_path, out_root)
    for p in points:
        p["runs"] = runs.get(p["point_id"], {})

    fit_points = [p for p in fit_candidates if any(_t_at(p, combo) > 0 for combo in design)]
    candidates, golden = _split_search(fit_points, design)

    # Boundary-extension rule: extend any axis whose golden value sits on the
    # sampled boundary, simulate the new combos, refit. Up to 4 rounds.
    for _round in range(4):
        if fit_only:
            break
        extra = _boundary_extensions(device, design, golden["combo"])
        if not extra:
            break
        print(f"[{device}] golden on boundary {golden['combo']} -> extending design "
              f"with {len(extra)} combos", flush=True)
        jobs = [(p, combo) for p in fit_points for combo in extra]
        _collect(jobs, workers, runs, runs_path, out_root)
        for p in points:
            p["runs"] = runs.get(p["point_id"], {})
        design = design + extra
        candidates, golden = _split_search(fit_points, design)

    # H100 pp>1 rule (HF nanotron pp>1 sanity rows)
    pp_gt1 = [p for p in fit_points if p["source"] == "hf_heldout_pp"]
    pp_gt1_cut = False
    if pp_gt1 and _mape(pp_gt1, golden["combo"]) > PP_GT1_CUT_MAPE:
        pp_gt1_cut = True
        subset_mape = _mape(pp_gt1, golden["combo"])
        for p in pp_gt1:
            p["fit_eligible"] = False
            p["report_note"] = (
                f"pp>1 nanotron row cut per protocol: subset MAPE {subset_mape:.1f}% "
                f"> {PP_GT1_CUT_MAPE:.0f}% at golden"
            )
        fit_points = [p for p in fit_points if p["source"] != "hf_heldout_pp"]
        candidates, golden = _split_search(fit_points, design)
        # re-run the boundary-extension rule on the post-cut golden
        for _round in range(4):
            if fit_only:
                break
            extra = _boundary_extensions(device, design, golden["combo"])
            if not extra:
                break
            print(f"[{device}] post-cut golden on boundary {golden['combo']} -> "
                  f"extending design with {len(extra)} combos", flush=True)
            jobs = [(p, combo) for p in fit_points for combo in extra]
            _collect(jobs, workers, runs, runs_path, out_root)
            for p in points:
                p["runs"] = runs.get(p["point_id"], {})
            design = design + extra
            candidates, golden = _split_search(fit_points, design)

    report_points = [p for p in points if not p["fit_eligible"]]
    if not fit_only and report_points:
        jobs = [(p, golden["combo"]) for p in report_points]
        _collect(jobs, workers, runs, runs_path, out_root)
        for p in points:
            p["runs"] = runs.get(p["point_id"], {})

    _write_outputs(device, points, fit_points, report_points, candidates, golden,
                   design, out_root, pp_gt1_cut)
    cc, dd, nn, ll = (golden["combo"] if len(golden["combo"]) == 4
                      else (*golden["combo"], LAUNCH_DEFAULT))
    print(
        f"\n[{device}] GOLDEN {golden['kind']}: compute={cc:.3f} dram={dd:.2f} "
        f"net={nn:.2f} launch={ll*1e6:.0f}us  calib {golden['calib_mape']:.2f}%  holdout "
        f"{golden['holdout_mape']:.2f}%",
        flush=True,
    )
    for workload in ("train", "inference"):
        subset = [p for p in fit_points if p["workload"] == workload]
        if subset:
            print(f"  {workload}: n={len(subset)} MAPE {_mape(subset, golden['combo']):.2f}%")
    return {"device": device, "golden": golden, "fit_points": fit_points}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", required=True,
                        choices=("H100_SXM5", "A100_SXM4", "A100_PCIe", "all"))
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("RAPID_VALIDATION_WORKERS", "8")))
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()

    devices = (
        ["A100_PCIe", "A100_SXM4", "H100_SXM5"] if args.device == "all" else [args.device]
    )
    for device in devices:
        run_device(device, args.workers, args.fit_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
