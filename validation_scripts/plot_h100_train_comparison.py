#!/usr/bin/env python3
"""H100 dense-training validation figure: nanotron benchmark runs vs Rapid-LLM.

Reference data: Hugging Face nanotron large-scale training benchmark sweep
(huggingface_data/bench_final2_mfu2.csv; the same rows that participate in the
H100 golden calibration/holdout, point ids hf:job*). Rapid-LLM values come
from device_testbench/H100_SXM5_opmodel/runs.json at the canonical factor key
c0.850_d0.80_n0.80.

Exclusion (disclosed in the paper): job 14113401 (8.86B, dp64-tp4, 256 GPUs)
is excluded as a degenerate reference run by the benchmark's own telemetry —
it achieved 11.2% MFU with 26 GB/s effective all-reduce bandwidth, far below
every other run in the family (16.4-38.4% MFU); the same pathology class the
H100 testbench's MFU<8% report-only cut targets. All other cached rows are
included; stats are recomputed from committed artifacts on every run.
"""
import json
import re
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE = SCRIPT_DIR / "device_testbench" / "H100_SXM5_opmodel" / "runs.json"
BENCH = SCRIPT_DIR / "huggingface_data" / "bench_final2_mfu2.csv"
FACTOR_KEY = "c0.850_d0.80_n0.80"
OUT = SCRIPT_DIR.parent / "output" / "validation" / "paper" / "h100_train_comparison.png"
EXCLUDED = {14113401}  # degenerate reference run (11.2% MFU, 26 GB/s allreduce)

COLOR_MAP = {"actual": "#4c566a", "rapid_llm": "#1f77b4"}


def main() -> None:
    import sys
    sys.path.insert(0, str(SCRIPT_DIR.parent))
    sys.path.insert(0, str(SCRIPT_DIR))
    from validation_scripts import device_testbench as dtb

    cache = json.loads(CACHE.read_text())
    bench = pd.read_csv(BENCH).set_index("job_id")

    rows = []
    for p in dtb.collect_points("H100_SXM5"):
        pid = p["point_id"]
        if not pid.startswith("hf:"):
            continue
        pred = cache.get(pid, {}).get(FACTOR_KEY)
        if pred is None:
            continue
        job_id = int(pid.split("job")[1])
        name = bench.loc[job_id, "name"]
        m = re.match(r"([0-9.]+)G_dp(\d+)_tp(\d+)_pp\d+_acc(\d+)_mbs(\d+)", name)
        size, dp, tp, acc, mbs = m.groups()
        label = f"{size}B dp{dp}-tp{tp}\nacc{acc}-mbs{mbs} ({p['scale']} GPUs)"
        excluded = job_id in EXCLUDED
        err = abs(pred - p["t_ref_s"]) / p["t_ref_s"] * 100
        rows.append((float(size), p["scale"], label, p["t_ref_s"], float(pred), err, excluded, job_id))

    rows.sort(key=lambda r: (r[0], r[1], r[7]))
    inc = [r for r in rows if not r[6]]
    exc = [r for r in rows if r[6]]
    errs = [r[5] for r in inc]
    print(f"H100 dense train (nanotron): {len(inc)} rows  MAPE {sum(errs)/len(errs):.2f}%  worst {max(errs):.1f}%")
    for _, _, lbl, a, pr, e, _, jid in exc:
        print(f"excluded job{jid}: actual {a:.2f}s pred {pr:.2f}s ({e:.1f}%) — degenerate reference run")

    labels = [r[2] for r in inc]
    x = list(range(len(inc)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar([i - width / 2 for i in x], [r[3] for r in inc], width=width,
           label="Actual (nanotron, pub.)", color=COLOR_MAP["actual"])
    ax.bar([i + width / 2 for i in x], [r[4] for r in inc], width=width,
           label="RAPID-LLM", color=COLOR_MAP["rapid_llm"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("Training time per batch (s)", fontsize=14)
    ax.set_title("Dense Training Runtime Comparison (H100 Nanotron Sweep)", fontsize=15)
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
