#!/usr/bin/env python3
"""Plot the dense-training validation figure: Actual vs Rapid-LLM vs Calculon.

Rapid-LLM values come from the canonical u=0.80 extended-roofline runs
(device_testbench/A100_SXM4_opmodel cache, key c1.000_d0.80_n0.80 — the same
values as the paper's 8.5% MAPE figure). Calculon values were produced by
running github.com/calculon-ai/calculon (commit 2024-02-22) with its own
shipped seqsel validation execution configs for the Korthikanti cases and
dp/batch-scaled variants of those configs for the SC'21 Selene cases
(see train_validation_data/calculon_data.csv).
"""
from pathlib import Path
import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
DATA = SCRIPT_DIR / "train_validation_data"

COLOR_MAP = {"actual": "#4c566a", "rapid_llm": "#1f77b4", "calculon": "#ff7f0e"}
DISPLAY_LABELS = {"actual": "Actual", "rapid_llm": "Rapid-LLM", "calculon": "Calculon"}

# label -> (actual_s, rapid_u80_s, calculon_s)
ROWS = [
    ("GPT 1T tp8-cp1-pp64-dp1-recompute-full", 87.9, 98.70, 90.08),
    ("GPT 1T tp8-cp1-pp64-dp1-recompute-selective", 69.1, 78.82, 66.04),
    ("GPT 1T tp8-cp1-pp64-dp6-recompute-full", 100.7, 106.82, 90.26),
    ("GPT 310B tp8-cp1-pp16-dp15-recompute-full", 34.1, 31.26, 33.61),
    ("GPT 530B tp8-cp1-pp35-dp9-recompute-full", 51.2, 52.11, 50.06),
    ("GPT 175B tp8-cp1-pp8-dp1-recompute-full", 16.9, 15.90, 18.03),
    ("GPT 175B tp8-cp1-pp8-dp1-recompute-selective", 12.9, 11.44, 13.64),
]


def main() -> None:
    labels = [r[0] for r in ROWS]
    series = {
        "actual": [r[1] for r in ROWS],
        "rapid_llm": [r[2] for r in ROWS],
        "calculon": [r[3] for r in ROWS],
    }
    for tool in ("rapid_llm", "calculon"):
        errs = [abs(p - a) / a * 100 for a, p in zip(series["actual"], series[tool])]
        print(f"{tool} MAPE: {sum(errs)/len(errs):.2f}%  worst {max(errs):.1f}%")

    tool_list = ["actual", "rapid_llm", "calculon"]
    width = 0.2
    x = list(range(len(labels)))
    fig_w = max(8.0, 0.8 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    for idx, name in enumerate(tool_list):
        offsets = [pos + (idx - (len(tool_list) - 1) / 2) * width for pos in x]
        ax.bar(offsets, series[name], width=width, label=DISPLAY_LABELS[name], color=COLOR_MAP[name])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Training time (s)")
    ax.set_title("Runtime comparison on large-scale system")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    out = DATA / "nvidia_train_validation_calculon.png"
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
