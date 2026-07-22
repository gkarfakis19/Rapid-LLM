#!/usr/bin/env python3
"""Dense-training validation figure: Actual vs Rapid-LLM vs llm-analysis vs STAGE.

Rapid-LLM values: canonical u=0.80 extended-roofline runs
(device_testbench/A100_SXM4_opmodel cache, key c1.000_d0.80_n0.80).
llm-analysis values: STOCK github.com/cli99/llm-analysis @ d841e40 (unmodified),
a100-sxm-80gb, w16a16e16, flops_efficiency=0.5 (its README-documented
large-scale-training value), all other efficiencies at tool defaults
(see train_validation_data/llm_analysis_configs/), with a POST-EMISSION
arithmetic correction removing its unconditional per-layer weight-allgather
floor (a ZeRO-3-style unsharding term inapplicable to these unsharded runs):
  t_corr = t_iter - 3 * n_acc * max(0, t_fwd_sharded_dp_comm - t_fwd_compute)
computed purely from the tool's own emitted summary components
(t_fwd_compute = attn + mlp + layernorm + tp_comm). dp=1 rows are unchanged;
uncorrected (stock-emitted) numbers reach 78.5% MAPE (worst +158%).
STAGE values: tool_seconds from train_validation_data/STAGE_data.csv (the
original STAGE runs; 4/7 cases — 530B ran out of time/memory budget, and
STAGE's recomputation flag is dead so selective rows are unsupported).
"""
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
DATA = SCRIPT_DIR / "train_validation_data"

COLOR_MAP = {"actual": "#4c566a", "rapid_llm": "#1f77b4", "llm_analysis": "#2ca02c", "stage": "#d62728"}
DISPLAY_LABELS = {"actual": "Actual", "rapid_llm": "RAPID-LLM", "llm_analysis": "llm-analysis", "stage": "STAGE+AS"}

# label -> (actual_s, rapid_u80_s, llm_analysis_s, stage_s or None)
ROWS = [
    ("GPT 1T tp8-cp1-pp64-dp1-recompute-full", 87.9, 98.70, 114.03, 18.24),
    ("GPT 1T tp8-cp1-pp64-dp1-recompute-selective", 69.1, 78.82, 87.93, None),
    ("GPT 1T tp8-cp1-pp64-dp6-recompute-full", 100.7, 106.82, 114.03, 18.46),
    ("GPT 310B tp8-cp1-pp16-dp15-recompute-full", 34.1, 31.26, 40.01, 6.16),
    ("GPT 530B tp8-cp1-pp35-dp9-recompute-full", 51.2, 52.11, 60.34, None),
    ("GPT 175B tp8-cp1-pp8-dp1-recompute-full", 16.9, 15.90, 20.42, 8.13),
    ("GPT 175B tp8-cp1-pp8-dp1-recompute-selective", 12.9, 11.44, 17.24, None),
]


def main() -> None:
    labels = [r[0] for r in ROWS]
    series = {
        "actual": [r[1] for r in ROWS],
        "rapid_llm": [r[2] for r in ROWS],
        "llm_analysis": [r[3] for r in ROWS],
        "stage": [r[4] for r in ROWS],
    }
    for tool in ("rapid_llm", "llm_analysis", "stage"):
        errs = [abs(p - a) / a * 100 for a, p in zip(series["actual"], series[tool]) if p is not None]
        print(f"{tool} MAPE: {sum(errs)/len(errs):.2f}%  worst {max(errs):.1f}%  ({len(errs)} rows)")

    tool_list = ["actual", "rapid_llm", "llm_analysis", "stage"]
    width = 0.19
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(max(8.0, 0.8 * len(labels)), 5))
    for idx, name in enumerate(tool_list):
        offsets = [pos + (idx - (len(tool_list) - 1) / 2) * width for pos in x]
        pts = [(o, v) for o, v in zip(offsets, series[name]) if v is not None]
        ax.bar([p[0] for p in pts], [p[1] for p in pts], width=width,
               label=DISPLAY_LABELS[name], color=COLOR_MAP[name])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Training time (s)")
    ax.set_title("GPT Dense Training Runtime Comparison (A100 Systems)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    out = DATA / "nvidia_train_validation_llmanalysis.png"
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
