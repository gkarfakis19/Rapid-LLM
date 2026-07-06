#!/usr/bin/env python3
"""MoE/EP training validation vs published Megatron-Core measurements.

Validates RAPID-LLM's predicted training time-per-iteration against the
"MCore" (non-folding) strong-scaling rows of NVIDIA's MoE Parallel Folding
paper (arXiv:2504.14960, Tables 3/4), measured on Eos (DGX H100 SuperPOD):

* Mixtral 8x22B  (coarse-grained MoE): TP2 PP8 EP4, 128-1024 GPUs
* Qwen2-57B-A14B (fine-grained MoE + shared expert): TP2 PP4 EP4, 64-1024 GPUs

All points use GBS 1024, seq 4096, MBS 1, BF16, balanced routing
(capacity factor 1 token-drop training -> expert_imbalance_factor 1.0).

Reference times are recovered from the paper's reported MFU via the Megatron
FLOPs-per-token convention (3x forward; full attention, SwiGLU = 3 matmuls,
router, shared expert, logits) against the 989.5 TFLOP/s BF16 peak.

Frozen Table III H100_SXM5 realization factors are applied (compute 0.56,
memory 0.80, communication 0.85, 6 us launch overhead). Nothing is tuned on
these points.

Topology: dim0 = intra-node NVLink FC 450 GB/s over [tp, cp, ep] (8 GPUs =
one DGX node); dim1 = inter-node InfiniBand over [pp, dp] assuming one NDR400
HCA (50 GB/s) per GPU, rail-optimized. Jobs spanning >= 2 SuperPOD scalable
units (>= 64 nodes) use the two-level SuperPOD model (leaf 50 GB/s x 8
switches, spine 100 GB/s); smaller jobs fit inside one SU and only ever
traverse the leaf tier, so they use its single-level equivalent: a flat
switch at 8 x 50 = 400 GB/s per node. The spine tier is over-provisioned, so
the two regimes are numerically continuous.

Out of scope (frontend cannot express): MCore w/ Folding (ETP != TP), FSDP
(ZeRO-3 + MoE), CP + MoE, FP8.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_ROOT = SCRIPT_DIR / "validation_configs"
DERATE_CONFIG = CONFIG_ROOT / "harness_derates.yaml"
MODEL_CONFIG_DIR = CONFIG_ROOT / "model-config"
BASE_HW = PROJECT_ROOT / "configs" / "hardware-config" / "H100_SXM5_80GB_superpod.yaml"
RUN_PERF = PROJECT_ROOT / "run_perf.py"

OUT_ROOT = SCRIPT_DIR / "moe_ep_validation"
CASE_ROOT = OUT_ROOT / "case_configs"
RESULTS_CSV = OUT_ROOT / "moe_validation_results.csv"
SUMMARY_TXT = OUT_ROOT / "summary.txt"
SCATTER_PNG = OUT_ROOT / "moe_validation_scatter.png"
MFU_PNG = OUT_ROOT / "moe_validation_mfu_trend.png"

TRAIN_TIME_RE = re.compile(r"Training time for batch:\s*([0-9]+(?:\.[0-9]+)?)s")
MEM_WARN_RE = re.compile(r"memory capacity|out of memory|OOM", re.IGNORECASE)

# Eos / DGX H100: BF16 peak per GPU (the paper's MFU denominator).
PEAK_FLOPS_PER_GPU = 989.5e12
GPUS_PER_NODE = 8

# SuperPOD scalable-unit geometry (H100 reference architecture).
SUPERPOD_LEAF_SIZE = 32
SUPERPOD_LEAF_SWITCHES = 8
SUPERPOD_SPINE_SWITCHES = 4
IB_BW_PER_GPU_GB = 50  # NDR400, one HCA per GPU
NVLINK_BW_GB = 450  # 4th gen NVLink, uni-directional per GPU


MODELS: dict[str, dict[str, Any]] = {
    "mixtral8x22b": {
        "display": "Mixtral 8x22B",
        "model_config": MODEL_CONFIG_DIR / "Mixtral-8x22B.yaml",
        "hidden_dim": 6144,
        "num_layers": 56,
        "kv_heads": 8,
        "head_dim": 128,
        "num_experts": 8,
        "top_k": 2,
        "moe_intermediate_size": 16384,
        "n_shared_experts": 0,
        "vocab_size": 32000,
        "active_params": 39e9,
    },
    "qwen2_57b_a14b": {
        "display": "Qwen2-57B-A14B",
        "model_config": MODEL_CONFIG_DIR / "Qwen2-57B-A14B.yaml",
        "hidden_dim": 3584,
        "num_layers": 28,
        "kv_heads": 4,
        "head_dim": 128,
        "num_experts": 64,
        "top_k": 8,
        "moe_intermediate_size": 2560,
        "n_shared_experts": 8,
        "vocab_size": 151936,
        "active_params": 14e9,
    },
    # Fine-grained Mixtral variant: each expert split 8-ways, still top-8 of
    # 64 — so the routed compute is HALF the base model's (8*2048 vs 2*16384
    # expert width). Active params: 39e9 - 33.8e9 (base routed) + 16.9e9.
    "mixtral8x22b_g8t8": {
        "display": "Mixtral 8x22B G8T8",
        "model_config": MODEL_CONFIG_DIR / "Mixtral-8x22B-G8T8.yaml",
        "hidden_dim": 6144,
        "num_layers": 56,
        "kv_heads": 8,
        "head_dim": 128,
        "num_experts": 64,
        "top_k": 8,
        "moe_intermediate_size": 2048,
        "n_shared_experts": 0,
        "vocab_size": 32000,
        "active_params": 22.1e9,
    },
    # Upcycled Llama3-70B x 8 experts. top_k=2 is an ASSUMPTION (not stated in
    # the paper) — flag wherever these points are reported.
    "llama3_8x70b": {
        "display": "Llama3-8x70B",
        "model_config": MODEL_CONFIG_DIR / "Llama3-8x70B.yaml",
        "hidden_dim": 8192,
        "num_layers": 80,
        "kv_heads": 8,
        "head_dim": 128,
        "num_experts": 8,
        "top_k": 2,
        "moe_intermediate_size": 28672,
        "n_shared_experts": 0,
        "vocab_size": 128256,
        "active_params": 130e9,
    },
}

SEQ_LEN = 4096
DEFAULT_GBS = 1024

# Core set: Table 4 "MCore" strong-scaling rows (GBS 1024, seq 4096, MBS 1).
# Optional set A: the two Table 3 "MCore" single points at GBS 256 (same
# parallelism; tests GBS sensitivity).
# Parallelism held fixed per model across scale; Megatron-DP = GPUs/(TP*PP).
#
# Stretch set (layout "ep_dim"): rows with tp * ep > 8, where the EP
# all-to-all spans NVLink nodes. These use a three-dimension network layout —
# dim0=[tp,cp] NVLink, dim1=[ep] on its own IB switch, dim2=[pp,dp] IB — which
# places ALL EP dispatch/combine traffic on the slow (inter-node) domain: a
# conservative bound, near-exact when most EP peers are in other nodes.
# The Table 3 "TP+EP+DP" rows additionally exercise ZeRO-1 without PP
# (gas = GBS / Megatron-DP so MBS stays 1). Llama3-8x70B's top_k=2 is an
# assumption (not stated in the paper).
CASES: list[dict[str, Any]] = [
    {"model": "mixtral8x22b", "gpus": 128, "tp": 2, "pp": 8, "ep": 4, "mfu_ref": 0.494},
    {"model": "mixtral8x22b", "gpus": 256, "tp": 2, "pp": 8, "ep": 4, "mfu_ref": 0.480},
    {"model": "mixtral8x22b", "gpus": 512, "tp": 2, "pp": 8, "ep": 4, "mfu_ref": 0.455},
    {"model": "mixtral8x22b", "gpus": 1024, "tp": 2, "pp": 8, "ep": 4, "mfu_ref": 0.423},
    {"model": "qwen2_57b_a14b", "gpus": 64, "tp": 2, "pp": 4, "ep": 4, "mfu_ref": 0.362},
    {"model": "qwen2_57b_a14b", "gpus": 128, "tp": 2, "pp": 4, "ep": 4, "mfu_ref": 0.360},
    {"model": "qwen2_57b_a14b", "gpus": 256, "tp": 2, "pp": 4, "ep": 4, "mfu_ref": 0.348},
    {"model": "qwen2_57b_a14b", "gpus": 512, "tp": 2, "pp": 4, "ep": 4, "mfu_ref": 0.325},
    {"model": "qwen2_57b_a14b", "gpus": 1024, "tp": 2, "pp": 4, "ep": 4, "mfu_ref": 0.298},
    # Optional set A (Table 3, GBS 256)
    {"model": "mixtral8x22b", "gpus": 128, "tp": 2, "pp": 8, "ep": 4, "gbs": 256, "mfu_ref": 0.463},
    {"model": "qwen2_57b_a14b", "gpus": 64, "tp": 2, "pp": 4, "ep": 4, "gbs": 256, "mfu_ref": 0.353},
    # Stretch: Table 3 "TP+EP+DP" rows (ZeRO-1, no PP, GBS 256, MBS 1 via gas)
    {"model": "mixtral8x22b", "gpus": 128, "tp": 4, "pp": 1, "ep": 8, "gbs": 256,
     "gas": 8, "layout": "ep_dim", "mfu_ref": 0.366},
    {"model": "qwen2_57b_a14b", "gpus": 64, "tp": 4, "pp": 1, "ep": 4, "gbs": 256,
     "gas": 16, "layout": "ep_dim", "mfu_ref": 0.231},
    # Stretch: Table 4 Mixtral-8x22B-G8T8 (TP2 PP8 EP8)
    {"model": "mixtral8x22b_g8t8", "gpus": 128, "tp": 2, "pp": 8, "ep": 8, "layout": "ep_dim", "mfu_ref": 0.198},
    {"model": "mixtral8x22b_g8t8", "gpus": 256, "tp": 2, "pp": 8, "ep": 8, "layout": "ep_dim", "mfu_ref": 0.184},
    {"model": "mixtral8x22b_g8t8", "gpus": 512, "tp": 2, "pp": 8, "ep": 8, "layout": "ep_dim", "mfu_ref": 0.163},
    {"model": "mixtral8x22b_g8t8", "gpus": 1024, "tp": 2, "pp": 8, "ep": 8, "layout": "ep_dim", "mfu_ref": 0.134},
    # Stretch: Table 4 Llama3-8x70B (TP8 PP8 EP4; top_k=2 assumed)
    {"model": "llama3_8x70b", "gpus": 256, "tp": 8, "pp": 8, "ep": 4, "layout": "ep_dim", "mfu_ref": 0.401},
    {"model": "llama3_8x70b", "gpus": 512, "tp": 8, "pp": 8, "ep": 4, "layout": "ep_dim", "mfu_ref": 0.395},
    {"model": "llama3_8x70b", "gpus": 1024, "tp": 8, "pp": 8, "ep": 4, "layout": "ep_dim", "mfu_ref": 0.391},
]

CSV_COLUMNS = [
    "case_id",
    "status",
    "model",
    "num_gpus",
    "global_batch_size",
    "seq_len",
    "tp",
    "pp",
    "ep",
    "cp",
    "dp_megatron",
    "dp_rapid",
    "mb",
    "gas",
    "layout",
    "vpp",
    "dim1_regime",
    "mfu_ref",
    "t_ref_s",
    "t_pred_s",
    "mfu_pred",
    "signed_error_pct",
    "abs_error_pct",
    "hardware_config",
    "model_config",
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
    hw.setdefault("sw_param", {}).pop("grad_acc_overhead", None)
    hw.setdefault("tech_param", {}).setdefault("core", {})["util"] = factors[
        "compute_util"
    ]
    hw.setdefault("tech_param", {}).setdefault("DRAM", {})["util"] = factors["dram_util"]
    for dim in hw.get("network", {}).get("dimensions", []):
        dim.setdefault("topology", {})["util"] = factors["network_util"]


def _case_gbs(case: dict[str, Any]) -> int:
    return int(case.get("gbs", DEFAULT_GBS))


def _flops_per_token(model: dict[str, Any], seq_len: int) -> float:
    """Megatron MFU convention: 3x forward, full (non-causal-halved) attention."""
    h = model["hidden_dim"]
    kv_dim = model["kv_heads"] * model["head_dim"]
    num_experts = model["num_experts"]
    qkv = 2 * h * (h + 2 * kv_dim)
    attn = 4 * seq_len * h
    out_proj = 2 * h * h
    router = 2 * h * num_experts if num_experts > 1 else 0
    routed = model["top_k"] * 3 * 2 * h * model["moe_intermediate_size"]
    shared = model["n_shared_experts"] * 3 * 2 * h * model["moe_intermediate_size"]
    logits = 2 * h * model["vocab_size"]
    fwd = model["num_layers"] * (qkv + attn + out_proj + router + routed + shared) + logits
    return 3.0 * fwd


def _reference_time_s(model: dict[str, Any], gpus: int, mfu: float, gbs: int) -> float:
    flops_per_iter = _flops_per_token(model, SEQ_LEN) * gbs * SEQ_LEN
    return flops_per_iter / (gpus * PEAK_FLOPS_PER_GPU * mfu)


def _predicted_mfu(model: dict[str, Any], gpus: int, t_pred_s: float, gbs: int) -> float:
    flops_per_iter = _flops_per_token(model, SEQ_LEN) * gbs * SEQ_LEN
    return flops_per_iter / (gpus * PEAK_FLOPS_PER_GPU * t_pred_s)


def _dim1_regime(nodes: int) -> str:
    if nodes % SUPERPOD_LEAF_SIZE == 0 and nodes // SUPERPOD_LEAF_SIZE > 1:
        return "superpod"
    return "leaf_switch"


def _ib_switch_topology(gpus_per_box: int) -> dict[str, Any]:
    """Flat IB switch endpoint: one NDR400 HCA (50 GB/s) per member GPU."""
    return {
        "type": "Switch",
        "bandwidth": f"{gpus_per_box * IB_BW_PER_GPU_GB} GB",
        "latency": 5e-6,
        "energy_per_bit": 8e-12,
        "util": 1.0,
    }


def _network_dimensions_ep_dim(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Three-dim layout for tp*ep > 8: EP crosses NVLink nodes.

    dim0 = [tp, cp] on NVLink; dim1 = [ep] on its own IB switch (ALL EP
    dispatch/combine traffic bound to the slow domain — conservative);
    dim2 = [pp, dp] on IB. Per-box IB bandwidth aggregates the member GPUs'
    HCAs. halvingDoubling does not implement all-to-all, so the ep dimension
    pins all_to_all to 'direct'.
    """
    tp_cp = case["tp"]  # cp is always 1 in this sweep
    dim0 = {
        "id": "dim0",
        "label": "nvlink_fc",
        "size": "auto",
        "topology": {
            "type": "FC",
            "bandwidth": f"{NVLINK_BW_GB} GB",
            "latency": 5e-6,
            "energy_per_bit": 8e-12,
            "util": 1.0,
            "optimize_2dmap": False,
        },
        "collective_override": {},
        "parallelisms": ["tp", "cp"],
    }
    dim1 = {
        "id": "dim1",
        "label": "infiniband_ep",
        "size": "auto",
        "topology": _ib_switch_topology(tp_cp),
        "collective_override": {"all_to_all": "direct"},
        "parallelisms": ["ep"],
    }
    dim2 = {
        "id": "dim2",
        "label": "infiniband_ppdp",
        "size": "auto",
        "topology": _ib_switch_topology(tp_cp * case["ep"]),
        "collective_override": {},
        "parallelisms": ["pp", "dp"],
    }
    return [dim0, dim1, dim2]


def _network_dimensions(nodes: int) -> list[dict[str, Any]]:
    """dim0 = NVLink node, dim1 = IB fabric over [pp, dp] (one box = one node)."""
    dim0 = {
        "id": "dim0",
        "label": "nvlink_fc",
        "size": "auto",
        "topology": {
            "type": "FC",
            "bandwidth": f"{NVLINK_BW_GB} GB",
            "latency": 5e-6,
            "energy_per_bit": 8e-12,
            "util": 1.0,
            "optimize_2dmap": False,
        },
        "collective_override": {},
        "parallelisms": ["tp", "cp", "ep"],
    }
    if _dim1_regime(nodes) == "superpod":
        dim1_topology: dict[str, Any] = {
            "type": "SuperPOD",
            "superpod_variant": "h100",
            "leaf_size": SUPERPOD_LEAF_SIZE,
            "leaf_switches_per_su": SUPERPOD_LEAF_SWITCHES,
            "spine_switches_per_su": SUPERPOD_SPINE_SWITCHES,
            "bandwidth": [f"{IB_BW_PER_GPU_GB} GB", f"{2 * IB_BW_PER_GPU_GB} GB"],
            "latency": 5e-6,
            "energy_per_bit": 8e-12,
            "util": 1.0,
        }
    else:
        # Job fits inside one scalable unit: only the leaf tier is traversed.
        # Equivalent single-level switch at leaf_switches * 50 GB/s per node.
        dim1_topology = {
            "type": "Switch",
            "bandwidth": f"{SUPERPOD_LEAF_SWITCHES * IB_BW_PER_GPU_GB} GB",
            "latency": 5e-6,
            "energy_per_bit": 8e-12,
            "util": 1.0,
        }
    dim1 = {
        "id": "dim1",
        "label": "infiniband",
        "size": "auto",
        "topology": dim1_topology,
        "collective_override": {},
        "parallelisms": ["pp", "dp"],
    }
    dim2 = {
        "id": "dim2",
        "label": "unused",
        "size": "auto",
        "topology": {
            "type": "Ring",
            "bandwidth": "25 GB",
            "latency": 5e-6,
            "energy_per_bit": 8e-12,
            "util": 1.0,
        },
        "collective_override": {},
        "parallelisms": [],
    }
    return [dim0, dim1, dim2]


def _case_gas(case: dict[str, Any]) -> int:
    return int(case.get("gas", 1))


def _case_layout(case: dict[str, Any]) -> str:
    return str(case.get("layout", "node"))


def _case_id(case: dict[str, Any]) -> str:
    base = f"{case['model']}_g{case['gpus']}"
    if _case_gbs(case) != DEFAULT_GBS:
        base += f"_gbs{_case_gbs(case)}"
    if _case_layout(case) == "ep_dim" and case["pp"] == 1:
        base += "_tpepdp"
    return base


def _case_parallelism(case: dict[str, Any]) -> dict[str, int]:
    gpus = case["gpus"]
    tp, pp, ep = case["tp"], case["pp"], case["ep"]
    gas = _case_gas(case)
    dp_rapid, rem = divmod(gpus, tp * pp * ep)
    if rem:
        raise ValueError(f"{_case_id(case)}: GPUs {gpus} not divisible by tp*pp*ep")
    dp_megatron = dp_rapid * ep
    per_step_batch, rem = divmod(_case_gbs(case), gas)
    if rem:
        raise ValueError(f"{_case_id(case)}: GBS not divisible by gas {gas}")
    mini_batch, rem = divmod(per_step_batch, dp_megatron)
    if rem:
        raise ValueError(f"{_case_id(case)}: per-step batch not divisible by Megatron-DP {dp_megatron}")
    # MBS = 1: with pp > 1 the microbatch count equals the per-replica batch;
    # with pp = 1 RAPID ignores mb (micro_batch = mini_batch), so gas must be
    # chosen to make mini_batch = 1.
    mb = mini_batch if pp > 1 else 1
    return {
        "dp_rapid": dp_rapid,
        "dp_megatron": dp_megatron,
        "mb": mb,
        "gas": gas,
        "nodes": gpus // GPUS_PER_NODE,
    }


def _build_configs(
    case: dict[str, Any],
    case_dir: Path,
    factors: dict[str, float],
    *,
    vpp: str = "off",
) -> tuple[Path, Path]:
    derived = _case_parallelism(case)
    hw = _load_yaml(BASE_HW)
    if _case_layout(case) == "ep_dim":
        hw["network"]["dimensions"] = _network_dimensions_ep_dim(case)
    else:
        hw["network"]["dimensions"] = _network_dimensions(derived["nodes"])
    # Interleaved-1F1B (VPP) policy: "max" = one layer per virtual stage
    # (Megatron-Core's usual benchmark setting), "off" = GPipe schedule.
    pipeline_interleave = 1
    if vpp == "max" and case["pp"] > 1:
        layers_per_stage, rem = divmod(MODELS[case["model"]]["num_layers"], case["pp"])
        if rem:
            raise ValueError(f"{_case_id(case)}: num_layers not divisible by pp")
        pipeline_interleave = layers_per_stage
    _deep_update(
        hw,
        {
            "parallelism": {
                "auto": False,
                "tp": case["tp"],
                "tp_sp": True,  # Megatron enables SP with TP; required for TP+EP
                "cp": 1,
                "pp": case["pp"],
                "mb": derived["mb"],
                "train": {"dp": derived["dp_rapid"], "ep": case["ep"], "tp_ep": True},
                "inference": {"replica_count": 1, "moe_dp": 1},
            },
            "sw_param": {
                "dp_zero_stage": 1,  # Megatron distributed optimizer
                # Reported MFU levels (up to 49.4%) are inconsistent with full
                # activation recompute (would imply ~66% effective utilization);
                # Megatron-Core recipes use selective/no recompute.
                "full_recomputation": False,
                "pipeline_interleave": pipeline_interleave,
            },
        },
    )
    _apply_derates(hw, factors)

    model = _load_yaml(MODELS[case["model"]]["model_config"])
    _deep_update(
        model,
        {
            "model_param": {
                "global_batch_size": _case_gbs(case),
                "gradient_accumulation_steps": _case_gas(case),
            }
        },
    )

    hw_path = case_dir / "hardware.yaml"
    model_path = case_dir / "model.yaml"
    _write_yaml(hw_path, hw)
    _write_yaml(model_path, model)
    return hw_path, model_path


def _run_case(
    case: dict[str, Any], factors: dict[str, float], *, vpp: str = "off"
) -> dict[str, Any]:
    case_id = _case_id(case)
    model = MODELS[case["model"]]
    derived = _case_parallelism(case)
    case_dir = CASE_ROOT / case_id
    hw_path, model_path = _build_configs(case, case_dir, factors, vpp=vpp)
    log_path = case_dir / "run.log"
    command = [
        sys.executable,
        str(RUN_PERF),
        "--hardware_config",
        str(hw_path),
        "--model_config",
        str(model_path),
    ]
    env = dict(os.environ)
    env["RAPID_ASTRA_CACHE_MODE"] = "NO_CACHE"
    result = subprocess.run(
        command,
        cwd=case_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env=env,
    )
    output = result.stdout or ""
    log_path.write_text(output, encoding="utf-8")

    gbs = _case_gbs(case)
    match = TRAIN_TIME_RE.search(output)
    t_pred = float(match.group(1)) if match else math.nan
    t_ref = _reference_time_s(model, case["gpus"], case["mfu_ref"], gbs)
    mfu_pred = _predicted_mfu(model, case["gpus"], t_pred, gbs) if t_pred > 0 else math.nan
    signed_err = (t_pred - t_ref) / t_ref * 100.0 if t_pred == t_pred else math.nan

    notes = []
    if MEM_WARN_RE.search(output):
        notes.append("memory warning in run.log")
    if result.returncode != 0:
        notes.append(f"run_perf return code {result.returncode}; see {_rel(log_path)}")
    status = "success" if result.returncode == 0 and t_pred == t_pred else "failed"

    row = {
        "case_id": case_id,
        "status": status,
        "model": model["display"],
        "num_gpus": case["gpus"],
        "global_batch_size": gbs,
        "seq_len": SEQ_LEN,
        "tp": case["tp"],
        "pp": case["pp"],
        "ep": case["ep"],
        "cp": 1,
        "dp_megatron": derived["dp_megatron"],
        "dp_rapid": derived["dp_rapid"],
        "mb": derived["mb"],
        "gas": derived["gas"],
        "layout": _case_layout(case),
        "vpp": vpp,
        "dim1_regime": (
            "ep_dim" if _case_layout(case) == "ep_dim" else _dim1_regime(derived["nodes"])
        ),
        "mfu_ref": f"{case['mfu_ref']:.4f}",
        "t_ref_s": f"{t_ref:.4f}",
        "t_pred_s": f"{t_pred:.4f}" if t_pred == t_pred else "",
        "mfu_pred": f"{mfu_pred:.4f}" if mfu_pred == mfu_pred else "",
        "signed_error_pct": f"{signed_err:+.2f}" if signed_err == signed_err else "",
        "abs_error_pct": f"{abs(signed_err):.2f}" if signed_err == signed_err else "",
        "hardware_config": _rel(hw_path),
        "model_config": _rel(model_path),
        "notes": "; ".join(notes),
    }
    print(
        f"[{case_id}] status={status} t_ref={t_ref:.2f}s t_pred={row['t_pred_s'] or 'n/a'}s "
        f"err={row['signed_error_pct'] or 'n/a'}%"
    )
    return row


def _plot(rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in rows if r["status"] == "success"]
    if not ok:
        return

    markers = {
        "Mixtral 8x22B": "o",
        "Qwen2-57B-A14B": "s",
        "Mixtral 8x22B G8T8": "D",
        "Llama3-8x70B": "^",
    }
    gpu_counts = sorted({int(r["num_gpus"]) for r in ok})
    cmap = plt.get_cmap("viridis")
    colors = {g: cmap(i / max(1, len(gpu_counts) - 1)) for i, g in enumerate(gpu_counts)}

    # Predicted-vs-actual scatter with y=x and +/-20% band.
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    lo = 0.5 * min(float(r["t_ref_s"]) for r in ok)
    hi = 2.0 * max(float(r["t_ref_s"]) for r in ok)
    xs = [lo, hi]
    ax.plot(xs, xs, "k-", linewidth=0.8, label="y = x")
    ax.fill_between(xs, [0.8 * x for x in xs], [1.2 * x for x in xs], color="grey", alpha=0.2, label="±20%")
    for row in ok:
        ax.scatter(
            float(row["t_ref_s"]),
            float(row["t_pred_s"]),
            marker=markers.get(row["model"], "x"),
            color=colors[int(row["num_gpus"])],
            edgecolors="black",
            linewidths=0.4,
            s=45,
            zorder=3,
        )
    for model, marker in markers.items():
        if any(r["model"] == model for r in ok):
            ax.scatter([], [], marker=marker, color="grey", edgecolors="black", linewidths=0.4, label=model)
    for gpus in gpu_counts:
        ax.scatter([], [], marker="o", color=colors[gpus], label=f"{gpus} GPUs")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Measured time per iteration (s)")
    ax.set_ylabel("Predicted time per iteration (s)")
    ax.set_title("MoE/EP validation vs Megatron-Core (arXiv:2504.14960)")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(SCATTER_PNG, dpi=200)
    plt.close(fig)

    # MFU-vs-scale trend (the strong-scaling slope is the point of the network
    # model). Only the GBS-1024 strong-scaling rows form the trend lines.
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    for model, marker in markers.items():
        model_rows = sorted(
            (
                r
                for r in ok
                if r["model"] == model and int(r["global_batch_size"]) == DEFAULT_GBS
            ),
            key=lambda r: int(r["num_gpus"]),
        )
        if not model_rows:
            continue
        gpus = [int(r["num_gpus"]) for r in model_rows]
        ax.plot(
            gpus,
            [100 * float(r["mfu_ref"]) for r in model_rows],
            marker=marker,
            linestyle="-",
            label=f"{model} measured",
        )
        ax.plot(
            gpus,
            [100 * float(r["mfu_pred"]) for r in model_rows],
            marker=marker,
            linestyle="--",
            label=f"{model} predicted",
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(gpu_counts)
    ax.set_xticklabels([str(g) for g in gpu_counts])
    ax.set_xlabel("GPUs")
    ax.set_ylabel("MFU (%)")
    ax.set_title("Strong-scaling MFU trend")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(MFU_PNG, dpi=200)
    plt.close(fig)


def _write_summary(rows: list[dict[str, Any]], factors: dict[str, float]) -> None:
    ok = [r for r in rows if r["status"] == "success" and r["abs_error_pct"]]
    errors = [float(r["abs_error_pct"]) for r in ok]

    lines = [
        "MoE/EP training validation vs Megatron-Core (arXiv:2504.14960, Tables 3/4 'MCore' rows)",
        "",
        f"Frozen factors: H100_SXM5 compute={factors['compute_util']}, memory={factors['dram_util']}, "
        f"communication={factors['network_util']}, launch={factors['kernel_launch_overhead_s']}s. Not tuned.",
        f"Base hardware: {_rel(BASE_HW)}",
        "Assumptions: BF16; MBS=1; balanced routing (capacity factor 1 token drop -> "
        "expert_imbalance_factor 1.0); one NDR400 HCA (50 GB/s) per GPU, rail-optimized; "
        "selective (not full) activation recomputation; NVLink 450 GB/s uni per GPU; "
        "Mixtral vocab 32000 per HF config.json (source doc said 32768; ~0.01% FLOPs impact).",
        "Jobs spanning >= 2 SuperPOD SUs use the two-level SuperPOD fabric model; jobs inside "
        "one SU use the equivalent single-level leaf switch (8 x 50 GB/s per node).",
        "Excluded as unsupported mappings: MCore w/ Folding (ETP != TP), FSDP, CP+MoE, FP8.",
        f"CSV: {_rel(RESULTS_CSV)}",
        "",
    ]

    if errors:
        lines.append(f"N (successful points): {len(errors)}")
        lines.append(f"MAPE: {statistics.fmean(errors):.2f}%")
        lines.append(f"Median absolute percentage error: {statistics.median(errors):.2f}%")
        lines.append(f"Max absolute percentage error: {max(errors):.2f}%")
        within = sum(1 for e in errors if e <= 20.0)
        lines.append(f"Points within ±20%: {within}/{len(errors)}")
    else:
        lines.append("No successful points.")
    lines.append("")

    for model in sorted({r["model"] for r in rows}):
        model_rows = sorted(
            (r for r in rows if r["model"] == model),
            key=lambda r: (int(r["global_batch_size"]), int(r["num_gpus"])),
        )
        lines.append(f"{model}:")
        for r in model_rows:
            lines.append(
                f"  {r['num_gpus']:>5} GPUs GBS {r['global_batch_size']:>4}  "
                f"MFU_ref={float(r['mfu_ref'])*100:5.1f}%  "
                f"t_ref={r['t_ref_s']}s  t_pred={r['t_pred_s'] or 'n/a'}s  "
                f"MFU_pred={(float(r['mfu_pred'])*100):.1f}%  err={r['signed_error_pct'] or 'n/a'}%"
                if r["mfu_pred"]
                else f"  {r['num_gpus']:>5} GPUs GBS {r['global_batch_size']:>4}  "
                f"MFU_ref={float(r['mfu_ref'])*100:5.1f}%  "
                f"t_ref={r['t_ref_s']}s  t_pred=n/a  ({r['status']})"
            )
        # Trend bar: only the GBS-1024 strong-scaling rows form the sweep.
        preds = [
            float(r["mfu_pred"])
            for r in model_rows
            if r["mfu_pred"] and int(r["global_batch_size"]) == DEFAULT_GBS
        ]
        if len(preds) >= 2:
            monotone = all(a > b for a, b in zip(preds, preds[1:]))
            lines.append(
                f"  predicted MFU monotonically decreasing with scale: {'YES' if monotone else 'NO'}"
            )
        lines.append("")

    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--filter",
        default="",
        help="only run cases whose case_id contains this substring (e.g. mixtral8x22b_g128)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("RAPID_VALIDATION_WORKERS", "4")),
        help="concurrent run_perf subprocesses",
    )
    parser.add_argument(
        "--vpp",
        choices=("off", "max"),
        default="off",
        help="pipeline schedule: off = GPipe (graph-native), max = interleaved 1F1B "
        "with one layer per virtual stage (Megatron-Core benchmark convention)",
    )
    args = parser.parse_args()

    cases = [c for c in CASES if args.filter in _case_id(c)]
    if not cases:
        print(f"No cases match filter {args.filter!r}")
        return 1

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    factors = _derates()

    # Sanity-check the reference FLOPs convention against 6 * active params.
    for key, model in MODELS.items():
        fpt = _flops_per_token(model, SEQ_LEN)
        ratio = fpt / (6 * model["active_params"])
        print(f"{key}: FLOPs/token={fpt:.4e} ({ratio:.3f}x of 6*active_params)")
        if not 0.85 <= ratio <= 1.15:
            raise ValueError(f"{key}: FLOPs/token deviates >15% from 6*active_params")

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        rows = list(pool.map(lambda c: _run_case(c, factors, vpp=args.vpp), cases))
    rows.sort(key=lambda r: (r["model"], int(r["num_gpus"])))

    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    _write_summary(rows, factors)
    _plot(rows)

    errors = [
        float(r["abs_error_pct"]) for r in rows if r["status"] == "success" and r["abs_error_pct"]
    ]
    if errors:
        print(
            f"N={len(errors)}, MAPE={statistics.fmean(errors):.2f}%, "
            f"max={max(errors):.2f}%, within ±20%: {sum(1 for e in errors if e <= 20)}/{len(errors)}"
        )
    print(f"Wrote {RESULTS_CSV}")
    print(f"Summary: {SUMMARY_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
