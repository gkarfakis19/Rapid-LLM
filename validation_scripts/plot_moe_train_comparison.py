#!/usr/bin/env python3
"""MoE training validation figure: Actual (Megatron-Core, published) vs Rapid-LLM.

Rapid-LLM values: canonical extended-roofline runs from
device_testbench/H100_SXM5_opmodel/runs.json at factor key c0.850_d0.80_n0.80
(H100 residual compute 0.85, global merged mem/net utilization 0.80, 6us launch).

Reference times derive from the published MFU of the MCore (non-folding) rows of
arXiv:2504.14960 (Tables 1/3/4) via
    t_ref = FLOPs_per_token * GBS * seq / (N_gpus * 989.5e12 * MFU)
with seq_len 4096 and exact FLOPs/token constants (3x forward):
Mixtral 8x22B 2.5067e11, Qwen2-57B-A14B 8.7158e10.
Published MFU: Mixtral 49.4/48.0/45.5/42.3% at 128/256/512/1024 GPUs (GBS 1024)
and 46.3% at 128 GPUs (GBS 256); Qwen2 36.2/36.0/34.8/32.5/29.8% at
64/128/256/512/1024 GPUs (GBS 1024) and 35.3% at 64 GPUs (GBS 256).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE = SCRIPT_DIR / "device_testbench" / "H100_SXM5_opmodel" / "runs.json"
FACTOR_KEY = "c0.850_d0.80_n0.80"
OUT = SCRIPT_DIR.parent / "output" / "validation" / "paper" / "moe_train_comparison.png"

# (cache case key, display label, t_ref_s)
CASES = [
    ("megatron:mixtral8x22b_g128", "Mixtral 8x22B $\\times$128\ntp2-pp8-ep4-dp2", 16.804),
    ("megatron:mixtral8x22b_g256", "Mixtral 8x22B $\\times$256\ntp2-pp8-ep4-dp4", 8.647),
    ("megatron:mixtral8x22b_g512", "Mixtral 8x22B $\\times$512\ntp2-pp8-ep4-dp8", 4.561),
    ("megatron:mixtral8x22b_g128_gbs256", "Mixtral 8x22B $\\times$128\ntp2-pp8-ep4-dp2 (gbs 256)", 4.482),
    ("megatron:mixtral8x22b_g1024", "Mixtral 8x22B $\\times$1024\ntp2-pp8-ep4-dp16", 2.453),
    ("megatron:qwen2_57b_a14b_g64", "Qwen2-57B-A14B $\\times$64\ntp2-pp4-ep4-dp2", 15.947),
    ("megatron:qwen2_57b_a14b_g128", "Qwen2-57B-A14B $\\times$128\ntp2-pp4-ep4-dp4", 8.018),
    ("megatron:qwen2_57b_a14b_g256", "Qwen2-57B-A14B $\\times$256\ntp2-pp4-ep4-dp8", 4.147),
    ("megatron:qwen2_57b_a14b_g64_gbs256", "Qwen2-57B-A14B $\\times$64\ntp2-pp4-ep4-dp2 (gbs 256)", 4.088),
    ("megatron:qwen2_57b_a14b_g512", "Qwen2-57B-A14B $\\times$512\ntp2-pp4-ep4-dp16", 2.220),
    ("megatron:qwen2_57b_a14b_g1024", "Qwen2-57B-A14B $\\times$1024\ntp2-pp4-ep4-dp32", 1.211),
]


def main() -> None:
    cache = json.loads(CACHE.read_text())
    labels, refs, preds = [], [], []
    for key, label, t_ref in CASES:
        pred = cache[key][FACTOR_KEY]
        labels.append(label)
        refs.append(t_ref)
        preds.append(float(pred))
    errs = [abs(p - a) / a * 100 for a, p in zip(refs, preds)]
    print(f"MAPE {sum(errs)/len(errs):.2f}%  worst {max(errs):.1f}%  ({len(errs)} rows)")

    x = list(range(len(labels)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.bar([i - width / 2 for i in x], refs, width=width,
           label="Actual (Megatron-Core, pub.)", color="#4c566a")
    ax.bar([i + width / 2 for i in x], preds, width=width,
           label="Rapid-LLM", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("Training time (s)", fontsize=14)
    ax.set_title("MoE training runtime comparison on H100 systems", fontsize=15)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
