#!/usr/bin/env python3
"""
Create a merged parallelism sweep plot from two TSV reports.

Side-by-side layout with a shared Y axis and a single color-encoding note.
"""

import argparse
import math
from pathlib import Path
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tools.parallelism_sweep as ps


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    idx = 1
    while True:
        candidate = path.with_name(f"{stem}_v{idx}{suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def _prepare_entries(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not results:
        return []
    plot_entries = results
    if ps.MEM_AWARE_FILTER:
        plot_entries = [item for item in results if not item.get("memory_exceeded", False)]
        if not plot_entries:
            return []
    plot_entries = ps._filter_by_runtime_ratio(plot_entries, ps.PLOT_RUNTIME_RATIO_CUTOFF)
    return plot_entries


def _build_frame(plot_entries: List[Dict[str, object]], metric_key: str) -> pd.DataFrame:
    finite_values = [
        float(item[metric_key])
        for item in plot_entries
        if math.isfinite(float(item[metric_key]))
    ]
    fallback_metric = max(finite_values) * 1.05 if finite_values else 1.0
    rows = []
    for i, item in enumerate(plot_entries):
        p = item["parallelism"]
        snap = ps._parallelism_snapshot(p)
        ng = int(item["num_gpus"])
        metric_val = float(item[metric_key])
        if not math.isfinite(metric_val):
            metric_val = fallback_metric
        rows.append({
            "row_id": i,
            "num_gpus": ng,
            "gpu_exp": ps._gpu_exp(ng),
            "tp": snap["tp"],
            "cp": snap["cp"],
            "dp": snap["dp"],
            "pp": snap["pp"],
            "memory_exceeded": bool(item.get("memory_exceeded", False)),
            metric_key: metric_val,
        })
    df = pd.DataFrame(rows)
    order = sorted(df["gpu_exp"].unique())
    df["gpu_exp_cat"] = pd.Categorical(df["gpu_exp"], categories=order, ordered=True)
    return df


def _plot_on_axis(
    ax: plt.Axes,
    plot_entries: List[Dict[str, object]],
    *,
    metric_key: str,
    xtick_size: float,
    ytick_size: float,
) -> tuple[list[float], list[float]]:
    if not plot_entries:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return [], []

    df = _build_frame(plot_entries, metric_key)
    order = sorted(df["gpu_exp"].unique())

    tps = df["tp"].to_numpy(dtype=float)
    cps = df["cp"].to_numpy(dtype=float)
    dps = df["dp"].to_numpy(dtype=float)
    pps = df["pp"].to_numpy(dtype=float)

    r_raw = np.log2(tps + cps)
    g_raw = np.log2(pps)
    b_raw = np.log2(dps)

    def _norm(x, gamma=0.85):
        x = x.astype(float)
        xmin, xmax = float(np.min(x)), float(np.max(x))
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
            y = np.zeros_like(x)
        else:
            y = (x - xmin) / (xmax - xmin)
        return np.power(y, gamma) if gamma and gamma != 1.0 else y

    r = _norm(r_raw)
    g = _norm(g_raw)
    b = _norm(b_raw)
    a = np.full_like(r, 0.9)

    palette = {}
    for i, rid in enumerate(df["row_id"]):
        if bool(df.iloc[i]["memory_exceeded"]):
            palette[rid] = (0.6, 0.6, 0.6, 0.7)
        else:
            palette[rid] = (float(r[i]), float(g[i]), float(b[i]), float(a[i]))

    sns.swarmplot(
        data=df,
        x="gpu_exp_cat",
        y=metric_key,
        hue="row_id",
        palette=palette,
        size=ps.PLOT_POINT_SIZE,
        linewidth=ps.PLOT_POINT_EDGE,
        edgecolor="black",
        dodge=False,
        legend=False,
        ax=ax,
    )

    spread_factor = 1.6
    max_span = 0.45
    total_points = 0
    for coll in ax.collections:
        offsets = getattr(coll, "get_offsets", None)
        if not callable(offsets):
            continue
        pts = offsets()
        if pts is None or not len(pts):
            continue
        total_points += len(pts)
        x_vals = pts[:, 0]
        centers = np.round(x_vals)
        scaled = centers + (x_vals - centers) * spread_factor
        pts[:, 0] = centers + np.clip(scaled - centers, -max_span, max_span)
        coll.set_offsets(pts)

    if total_points < len(df):
        ax.clear()
        rng = np.random.RandomState(ps.PLOT_JITTER_SEED)
        x_base = df["gpu_exp_cat"].cat.codes.to_numpy(dtype=float)
        jitter_width = min(0.48, ps.PLOT_JITTER_WIDTH * 2.5)
        jitter = rng.uniform(-jitter_width, jitter_width, size=len(df))
        xs = x_base + jitter
        ys = df[metric_key].to_numpy(dtype=float)
        colors = [palette[rid] for rid in df["row_id"]]
        ax.scatter(
            xs,
            ys,
            s=ps.PLOT_POINT_SIZE ** 2,
            c=colors,
            edgecolor="black",
            linewidth=ps.PLOT_POINT_EDGE,
        )

    ax.tick_params(axis="x", labelsize=xtick_size)
    ax.tick_params(axis="y", labelsize=ytick_size)
    ax.set_xlabel("")
    ax.grid(alpha=0.3, axis="y")

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        [
            str(int(round(2 ** e))) if float(e).is_integer() else f"2^{e:.2f}"
            for e in order
        ]
    )

    best_by_gpu: Dict[int, float] = {}
    for item in plot_entries:
        if item.get("memory_exceeded"):
            continue
        runtime_val = float(item.get("runtime", float("nan")))
        if not math.isfinite(runtime_val):
            continue
        ng = int(item.get("num_gpus", 0))
        best = best_by_gpu.get(ng)
        if best is None or runtime_val < best:
            best_by_gpu[ng] = runtime_val
    if best_by_gpu:
        xs: list[float] = []
        ys: list[float] = []
        for ng, runtime_val in best_by_gpu.items():
            gpu_exp = ps._gpu_exp(ng)
            if gpu_exp in order:
                xs.append(float(order.index(gpu_exp)))
                ys.append(float(runtime_val))
        if xs and ys:
            ax.scatter(
                xs,
                ys,
                s=100,
                marker="*",
                c="white",
                edgecolor="black",
                zorder=6,
                label="Best runtime",
            )
            ax.legend(loc="best")
            return xs, ys
    return [], []


def _plot_regression_line(
    ax: plt.Axes,
    xs: list[float],
    ys: list[float],
) -> None:
    if len(xs) < 2:
        return
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    order = np.argsort(x_arr)
    x_sorted = x_arr[order]
    y_sorted = y_arr[order]

    degree = min(3, len(x_sorted) - 1)
    coeffs = np.polyfit(x_sorted, y_sorted, deg=degree)
    poly = np.poly1d(coeffs)

    x_min = float(np.min(x_sorted)) - 0.6
    x_max = float(np.max(x_sorted)) + 0.6
    x_fit = np.linspace(x_min, x_max, 240)
    y_fit = poly(x_fit)

    line_kwargs = dict(color="#8ec9ff", alpha=0.5, linewidth=3.0, zorder=4)
    ax.plot(x_fit, y_fit, **line_kwargs)
def main() -> None:
    parser = argparse.ArgumentParser(description="Merged parallelism sweep plot.")
    parser.add_argument("tsv_1", help="Path to the 1x SuperPOD TSV report.")
    parser.add_argument("tsv_2", help="Path to the 2x SuperPOD TSV report.")
    parser.add_argument("--out", required=True, help="Output PNG path.")
    args = parser.parse_args()

    results_1 = ps.load_results_from_report(args.tsv_1)
    results_2 = ps.load_results_from_report(args.tsv_2)

    entries_1 = _prepare_entries(results_1)
    entries_2 = _prepare_entries(results_2)

    metric_key = "runtime" if ps.PLOT_METRIC.lower() == "runtime" else "performance"

    def _scale_font_size(value: object, factor: float) -> float:
        size = FontProperties(size=value).get_size_in_points()
        return size * factor

    base_label_size = _scale_font_size(plt.rcParams["axes.labelsize"], 1.0)
    ylabel_size = base_label_size * 1.25
    xlabel_size = base_label_size * 1.5
    xtick_size = _scale_font_size(plt.rcParams["xtick.labelsize"], 1.25)
    ytick_size = _scale_font_size(plt.rcParams["ytick.labelsize"], 1.25)
    title_base = plt.rcParams.get("figure.titlesize", plt.rcParams["axes.titlesize"])
    title_size = _scale_font_size(title_base, 1.5)
    color_legend_size = 9 * 1.5

    fig, axes = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(16, 6),
    )

    star_x_1, star_y_1 = _plot_on_axis(
        axes[0],
        entries_1,
        metric_key=metric_key,
        xtick_size=xtick_size,
        ytick_size=ytick_size,
    )
    star_x_2, star_y_2 = _plot_on_axis(
        axes[1],
        entries_2,
        metric_key=metric_key,
        xtick_size=xtick_size,
        ytick_size=ytick_size,
    )

    # _plot_regression_line(axes[0], star_x_1, star_y_1)
    # _plot_regression_line(axes[1], star_x_2, star_y_2)

    if metric_key == "runtime":
        axes[0].set_ylabel("Runtime (s)", fontsize=ylabel_size)
        min_metric = min(
            float(item.get("runtime", float("inf")))
            for item in (entries_1 + entries_2)
            if math.isfinite(float(item.get("runtime", float("inf"))))
        )
        if min_metric > 0:
            for ax in axes:
                ax.set_yscale("log")
        else:
            print("Warning: non-positive runtime values; using linear scale.")
    else:
        axes[0].set_ylabel("Performance (1 / s)", fontsize=ylabel_size)

    fig.supxlabel("Number of GPUs", fontsize=xlabel_size)
    fig.suptitle(
        "Parallelism options vs runtime (Left: baseline fabric | Right: 1/8x fabric bandwidth)",
        fontsize=title_size,
    )

    fig.text(
        0.985,
        0.02,
        "Color: R=log2(tp+cp), G=log2(pp), B=log2(dp)",
        ha="right",
        va="bottom",
        fontsize=color_legend_size,
        alpha=0.8,
    )

    # fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    fig.tight_layout()

    ax0_pos = axes[0].get_position()
    ax1_pos = axes[1].get_position()
    mid_x = (ax0_pos.x1 + ax1_pos.x0) / 2
    y0 = min(ax0_pos.y0, ax1_pos.y0)
    y1 = max(ax0_pos.y1, ax1_pos.y1)
    fig.add_artist(Line2D([mid_x, mid_x], [y0, y1], transform=fig.transFigure, color="#999999", linewidth=1.0, alpha=0.6))

    out_path = _unique_path(Path(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved merged plot to {out_path}")


if __name__ == "__main__":
    main()
