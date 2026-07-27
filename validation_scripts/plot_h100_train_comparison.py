#!/usr/bin/env python3
"""H100 dense-training validation figure: nanotron benchmark runs vs RAPID-LLM.

Reference data: Hugging Face nanotron large-scale training benchmark sweep
(huggingface_data/bench_final2_mfu2.csv). Selection is the FULL pp==1
population of successful post-pp-fix runs -- 68 rows, no tp or dp window
(see validation_scripts/h100_testbench.py::_hf_rows). RAPID-LLM values come
from device_testbench/H100_SXM5_opmodel/runs.json at the canonical frozen
factor key c0.850_d0.80_n0.80 (residual compute 0.85, merged memory/network
0.80, 6us kernel-launch overhead). No factor is fitted here.

The only exclusion is the disclosed applicability envelope
h100_testbench.MEASURED_MFU_FLOOR (20% measured MFU): RAPID-LLM is a
roofline-based analytical model, not cycle-accurate, so configurations the
benchmark itself records as running far below achievable efficiency are
outside the regime it targets. The floor is a property of the reference run,
never of the prediction. It lands in a 4.2 pp gap in the sweep's efficiency
distribution (highest below-floor run 16.4% MFU, lowest above-floor run
20.6%), so any cut in (16.4%, 20.6%] selects the same 32 rows.

Layout is a single-panel predicted-vs-actual parity scatter matched to the
MPT figure (mosaicml_train.py::_plot_parity_combined_seq_len) so the two
training-validation figures read as a pair: muted-red dashed y=x, marker
SHAPE = model size, marker COLOR = GPU count, two boxed legends. The axes
are LOG-LOG (shared, identical limits on both, so y=x stays at 45 degrees):
step time spans 0.53 s to 36 s and 28 of the 32 retained runs sit below 6 s,
which a linear axis crushes into the lower-left corner. Ticks are the 1-3
decade ladder with plain (non-scientific) labels so values read directly in
seconds. Stats are recomputed from the committed artifacts on every run.
"""
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import (  # noqa: E402
    FixedFormatter,
    FixedLocator,
    NullLocator,
)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

from validation_scripts.plot_style import (  # noqa: E402
    IEEE_AXIS_TITLE_SIZE_PT,
    IEEE_DPI,
    IEEE_FONT_SIZE_PT,
    IEEE_HALF_COLUMN_WIDTH_IN,
    IEEE_TITLE_SIZE_PT,
    ieee_rc_params,
)

CACHE = SCRIPT_DIR / "device_testbench" / "H100_SXM5_opmodel" / "runs.json"
BENCH = SCRIPT_DIR / "huggingface_data" / "bench_final2_mfu2.csv"
FACTOR_KEY = "c0.850_d0.80_n0.80"
OUT = SCRIPT_DIR.parent / "output" / "validation" / "paper" / "h100_train_comparison.png"

# Same marker vocabulary and ordering as the MPT parity figure.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
# GPU count is an ordinal magnitude, so its colors are a single-hue ramp
# (light -> dark) rather than a categorical cycle: monotone in lightness,
# adjacent dL >= 0.06, light end clears the surface, and unlike tab10 it
# stays separable under protanopia.
GPU_RAMP = ["#6aa9d6", "#3585bf", "#19629f", "#0a3a66"]
PARITY_LINE = "#b04a4a"  # muted red dashed y=x, as in the MPT figure
# Log-axis padding, in decades, applied to both ends of the shared range.
# 0.16 dec (a factor of 1.45) is the smallest pad that keeps the 0.3 s tick
# inside the axis for the current population while leaving the extreme runs
# (0.43 s predicted, 36.2 s actual) comfortably off the frame edge.
LOG_PAD_DECADES = 0.16
# 1-3 ladder: decade ticks plus their 3x midpoints, labelled plainly.
TICK_LADDER = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]


def collect() -> pd.DataFrame:
    """Every pp==1 nanotron row with a cached prediction at the frozen key."""
    from validation_scripts import device_testbench as dtb
    from validation_scripts import h100_testbench as tb
    from validation_scripts import moe_ep_validation as mev

    cache = json.loads(CACHE.read_text())
    # NOTE: bench_final2_mfu2.csv contains duplicated job_id records, so the
    # selected row itself (not a job_id lookup) is the authority for the
    # parallelism actually simulated.
    hf_rows = {int(r["job_id"]): r for s, r in tb._hf_rows() if s == "hf_pp1"}

    recs, missing = [], []
    for p in dtb.collect_points("H100_SXM5"):
        if p["source"] != "hf_pp1":
            continue
        job_id = int(p["point_id"].split("job")[1])
        pred = cache.get(p["point_id"], {}).get(FACTOR_KEY)
        if not (pred and pred > 0):
            missing.append(job_id)
            continue
        row = hf_rows[job_id]
        mfu = (
            tb._hf_flops_per_token(row)
            * float(row["tok/s/gpu"])
            / mev.PEAK_FLOPS_PER_GPU
        )
        recs.append(
            dict(
                job=job_id,
                size=float(re.match(r"([0-9.]+)G_", str(row["name"])).group(1)),
                dp=int(row["dp"]), tp=int(row["tp"]), gpus=p["scale"], mfu=mfu,
                actual=p["t_ref_s"], predicted=float(pred),
                err=(float(pred) - p["t_ref_s"]) / p["t_ref_s"] * 100.0,
                retained=bool(p["fit_eligible"]),
            )
        )
    if missing:
        print(f"[WARN] no cached prediction at {FACTOR_KEY} for jobs {missing}")
    df = pd.DataFrame(recs)
    df["abs_err"] = df["err"].abs()
    return df.sort_values(["size", "gpus", "job"]).reset_index(drop=True)


def main() -> None:
    from validation_scripts import h100_testbench as tb

    df = collect()
    keep = df[df.retained].copy()
    drop = df[~df.retained]
    floor_pct = tb.MEASURED_MFU_FLOOR * 100

    print(
        f"H100 dense train (nanotron, pp==1, measured MFU >= {floor_pct:.0f}%): "
        f"{len(keep)} rows  MAPE {keep.abs_err.mean():.2f}%  "
        f"worst {keep.abs_err.max():.2f}%  "
        f"mean signed {keep.err.mean():+.2f}%  median {keep.abs_err.median():.2f}%"
    )
    print(
        f"  full pp==1 population: {len(df)} rows  MAPE {df.abs_err.mean():.2f}%; "
        f"{len(drop)} rows below the floor are report-only"
    )
    if len(drop):
        print(
            f"  floor gap: highest below-floor run {drop.mfu.max()*100:.2f}% MFU, "
            f"lowest retained run {keep.mfu.min()*100:.2f}% MFU"
        )

    sizes = sorted(keep["size"].unique())
    gpu_counts = sorted(keep["gpus"].unique())
    size_to_marker = {s: MARKERS[i % len(MARKERS)] for i, s in enumerate(sizes)}
    gpu_to_color = {g: GPU_RAMP[i % len(GPU_RAMP)] for i, g in enumerate(gpu_counts)}

    with plt.rc_context(ieee_rc_params()):
        fig, ax = plt.subplots(figsize=(IEEE_HALF_COLUMN_WIDTH_IN * 1.45, 3.2))
        for size in sizes:
            for gpus in gpu_counts:
                sub = keep[(keep["size"] == size) & (keep["gpus"] == gpus)]
                if sub.empty:
                    continue
                ax.scatter(
                    sub["actual"], sub["predicted"],
                    marker=size_to_marker[size], s=26, alpha=0.9,
                    c=[gpu_to_color[gpus]], edgecolors="none", linewidths=0.0,
                )

        lower = float(min(keep["actual"].min(), keep["predicted"].min()))
        upper = float(max(keep["actual"].max(), keep["predicted"].max()))
        # Pad symmetrically in log space so the shared limits (and hence the
        # 45-degree y=x line) stay identical on both axes.
        lower_lim = 10.0 ** (math.log10(lower) - LOG_PAD_DECADES)
        upper_lim = 10.0 ** (math.log10(upper) + LOG_PAD_DECADES)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot([lower_lim, upper_lim], [lower_lim, upper_lim],
                linestyle="--", color=PARITY_LINE, linewidth=1.1)
        ax.set_xlim(lower_lim, upper_lim)
        ax.set_ylim(lower_lim, upper_lim)

        ticks = [t for t in TICK_LADDER if lower_lim <= t <= upper_lim]
        labels = [f"{t:g}" for t in ticks]
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator(ticks))
            axis.set_major_formatter(FixedFormatter(labels))
            # No minor ticks: keeps the gridline density identical to the
            # linear version instead of drawing 8 faint lines per decade.
            axis.set_minor_locator(NullLocator())
        ax.set_title("Dense Training Runtime Comparison (H100 Nanotron Sweep)",
                     fontsize=IEEE_TITLE_SIZE_PT)
        ax.set_xlabel("Actual (s)", fontsize=IEEE_AXIS_TITLE_SIZE_PT)
        ax.set_ylabel("Predicted (s)", fontsize=IEEE_AXIS_TITLE_SIZE_PT)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

        size_handles = [
            Line2D([0], [0], marker=size_to_marker[s], linestyle="None",
                   markerfacecolor="white", markeredgecolor="black",
                   markeredgewidth=0.9, markersize=6.5, label=f"{s:g}B")
            for s in sizes
        ]
        gpu_handles = [
            Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor=gpu_to_color[g], markeredgecolor="none",
                   markersize=6.5, label=str(g))
            for g in gpu_counts
        ]
        size_legend = ax.legend(
            handles=size_handles, loc="upper left", title="Model Size",
            fontsize=IEEE_FONT_SIZE_PT, title_fontsize=IEEE_FONT_SIZE_PT,
            framealpha=0.9,
        )
        ax.add_artist(size_legend)
        ax.legend(
            handles=gpu_handles, loc="lower right", title="GPUs",
            fontsize=IEEE_FONT_SIZE_PT, title_fontsize=IEEE_FONT_SIZE_PT,
            framealpha=0.9,
        )
        fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=IEEE_DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
