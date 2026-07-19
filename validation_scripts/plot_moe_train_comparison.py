#!/usr/bin/env python3
"""MoE training validation figure: published Megatron-Core MFU vs Rapid-LLM.

The comparison is in the MFU basis the source publishes (arXiv:2504.14960,
MCore non-folding rows; MFU quoted verbatim below). Rapid-LLM's simulated
iteration times come from device_testbench/H100_SXM5_opmodel/runs.json at the
canonical factor key c0.850_d0.80_n0.80 (H100 residual compute 0.85, global
merged mem/net utilization 0.80, 6us launch) and are converted to MFU via

    MFU_pred = F_tok * GBS * seq / (N_gpus * 989.5e12 * t_pred)

with seq_len 4096 and exact analytical FLOPs/token (3x forward):
Mixtral 8x22B 2.5067e11, Qwen2-57B-A14B 8.7158e10.

The calibration/holdout split is read from the canonized golden file
validation_configs/H100_SXM5_golden_calib_backend.yaml (holdout_points), so
every reported number (full MAPE, held-out-half MAPE, worst case) is
recomputed from committed artifacts on each run.
"""
import json
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE = SCRIPT_DIR / "device_testbench" / "H100_SXM5_opmodel" / "runs.json"
GOLDEN = SCRIPT_DIR / "validation_configs" / "H100_SXM5_golden_calib_backend.yaml"
FACTOR_KEY = "c0.850_d0.80_n0.80"
OUT = SCRIPT_DIR.parent / "output" / "validation" / "paper" / "moe_train_comparison.png"

SEQ_LEN = 4096
PEAK_FLOPS = 989.5e12
FLOPS_PER_TOKEN = {"mixtral": 2.5067e11, "qwen2": 8.7158e10}

# (cache case key, display label, published MFU, N_gpus, GBS)
CASES = [
    ("megatron:mixtral8x22b_g128", "Mixtral 8x22B $\\times$128\ntp2-pp8-ep4-dp2", 0.494, 128, 1024),
    ("megatron:mixtral8x22b_g256", "Mixtral 8x22B $\\times$256\ntp2-pp8-ep4-dp4", 0.480, 256, 1024),
    ("megatron:mixtral8x22b_g512", "Mixtral 8x22B $\\times$512\ntp2-pp8-ep4-dp8", 0.455, 512, 1024),
    ("megatron:mixtral8x22b_g128_gbs256", "Mixtral 8x22B $\\times$128\ntp2-pp8-ep4-dp2 (gbs 256)", 0.463, 128, 256),
    ("megatron:mixtral8x22b_g1024", "Mixtral 8x22B $\\times$1024\ntp2-pp8-ep4-dp16", 0.423, 1024, 1024),
    ("megatron:qwen2_57b_a14b_g64", "Qwen2-57B-A14B $\\times$64\ntp2-pp4-ep4-dp2", 0.362, 64, 1024),
    ("megatron:qwen2_57b_a14b_g128", "Qwen2-57B-A14B $\\times$128\ntp2-pp4-ep4-dp4", 0.360, 128, 1024),
    ("megatron:qwen2_57b_a14b_g256", "Qwen2-57B-A14B $\\times$256\ntp2-pp4-ep4-dp8", 0.348, 256, 1024),
    ("megatron:qwen2_57b_a14b_g64_gbs256", "Qwen2-57B-A14B $\\times$64\ntp2-pp4-ep4-dp2 (gbs 256)", 0.353, 64, 256),
    ("megatron:qwen2_57b_a14b_g512", "Qwen2-57B-A14B $\\times$512\ntp2-pp4-ep4-dp16", 0.325, 512, 1024),
    ("megatron:qwen2_57b_a14b_g1024", "Qwen2-57B-A14B $\\times$1024\ntp2-pp4-ep4-dp32", 0.298, 1024, 1024),
]


def main() -> None:
    cache = json.loads(CACHE.read_text())
    golden = yaml.safe_load(GOLDEN.read_text())
    holdout_ids = {p for p in golden["holdout_points"] if p.startswith("megatron:")}

    labels, mfu_ref, mfu_pred, errs, holdout_errs = [], [], [], [], []
    for key, label, ref, n_gpus, gbs in CASES:
        t_pred = float(cache[key][FACTOR_KEY])
        f_tok = FLOPS_PER_TOKEN["mixtral" if "mixtral" in key else "qwen2"]
        pred = f_tok * gbs * SEQ_LEN / (n_gpus * PEAK_FLOPS * t_pred)
        err = abs(pred - ref) / ref * 100
        labels.append(label)
        mfu_ref.append(ref * 100)
        mfu_pred.append(pred * 100)
        errs.append(err)
        if key in holdout_ids:
            holdout_errs.append(err)
    print(f"MFU MAPE {sum(errs)/len(errs):.2f}%  worst {max(errs):.1f}%  ({len(errs)} rows)")
    print(f"held-out half ({len(holdout_errs)} rows, per {GOLDEN.name}): "
          f"{sum(holdout_errs)/len(holdout_errs):.2f}%")

    x = list(range(len(labels)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.bar([i - width / 2 for i in x], mfu_ref, width=width,
           label="Actual (Megatron-Core, pub.)", color="#4c566a")
    ax.bar([i + width / 2 for i in x], mfu_pred, width=width,
           label="Rapid-LLM", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("MFU (%)", fontsize=14)
    ax.set_title("MoE training MFU comparison on H100 systems", fontsize=15)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
