#!/usr/bin/env python3
"""
Compile per-run combined aten::mm energy breakdowns into a single plot.

This script scans run directories under:
  Rapid-LLM/energy/imec/Llama2-7B/prompt_*_predict_*

For each run, it reads all per-phase comparison CSVs produced by
`Rapid-LLM/energy/mem_accesses.py`:
  <run>/mm_access_compare/*_mm_access_compare.csv

Then it aggregates the DRAM/L2/SHMEM energy (uJ) for each method.

Finally it produces:
  - a single plot with runs on the x-axis and 3 stacked bars per run
  - an optional CSV dump of the aggregated energies

Uses matplotlib for plotting.
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


RAPID_LLM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = RAPID_LLM_ROOT / "energy/imec/Llama2-7B"

RUN_DIR_RE = re.compile(r"^prompt_(\d+)_predict_(\d+)$")

# Method keys must match the column names in *_mm_access_compare.csv.
NCU_KEY = "NCU"
GEMMCALC_KEY = "GEMMCalc"
DEEPFLOW_KEY = "DeepFlow"

# Display names are independent of the CSV keys. You can change these freely
# (or use --display-name) without breaking the data lookup.
GEMMCALC_LABEL = "Rapid-LLM"
DEEPCALC_LABEL = "DeepFlow"
NCU_LABEL = "NCU"

METHOD_ORDER = [NCU_KEY, GEMMCALC_KEY, DEEPFLOW_KEY]
DEFAULT_DISPLAY_NAMES = {
    NCU_KEY: NCU_LABEL,
    GEMMCALC_KEY: GEMMCALC_LABEL,
    DEEPFLOW_KEY: DEEPCALC_LABEL,
}

DRAM_PJ_PER_SECTOR_32B = 2090.0
L2_PJ_PER_SECTOR_32B = 368.0
SHMEM_PJ_PER_32B = 82.1


def _parse_float(s: str) -> float:
    if s is None:
        return 0.0
    t = str(s).strip()
    if not t or t.lower() in ("n/a", "na", "nan", "none"):
        return 0.0
    t = t.replace(",", "")
    try:
        return float(t)
    except Exception:
        return 0.0


def _energy_uJ(dram_sectors_32b: float, l2_sectors_32b: float, shmem_bytes: float) -> Dict[str, float]:
    dram_uJ = (float(dram_sectors_32b) * DRAM_PJ_PER_SECTOR_32B) / 1e6
    l2_uJ = (float(l2_sectors_32b) * L2_PJ_PER_SECTOR_32B) / 1e6
    shmem_uJ = ((float(shmem_bytes) / 32.0) * SHMEM_PJ_PER_32B) / 1e6
    total_uJ = dram_uJ + l2_uJ + shmem_uJ
    return {"dram_uJ": dram_uJ, "l2_uJ": l2_uJ, "shmem_uJ": shmem_uJ, "total_uJ": total_uJ}


def _iter_run_dirs(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return []
    runs = []
    for p in root.iterdir():
        m = RUN_DIR_RE.match(p.name) if p.is_dir() else None
        if not m:
            continue
        runs.append((int(m.group(1)), int(m.group(2)), p))
    for _prompt, _predict, p in sorted(runs, key=lambda t: (t[0], t[1], t[2].name)):
        yield p


def _read_mm_access_compare_csv(path: Path) -> Dict[str, Dict[str, float]]:
    """Return {method_label: {metric_name: value}}."""
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    header_idx = None
    for i, row in enumerate(rows):
        if row and row[0].strip() == "metric":
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Missing 'metric' header in %s" % path)

    header = rows[header_idx]
    if len(header) < 2:
        raise ValueError("Malformed metric header in %s" % path)
    methods = [m.strip() for m in header[1:]]
    out: Dict[str, Dict[str, float]] = {m: {} for m in methods}

    for row in rows[header_idx + 1 :]:
        if not row or not row[0].strip():
            continue
        metric = row[0].strip()
        for mi, m in enumerate(methods):
            if 1 + mi < len(row):
                out[m][metric] = _parse_float(row[1 + mi])
            else:
                out[m][metric] = 0.0
    return out


def _aggregate_run_energy(run_dir: Path) -> Tuple[Dict[str, Dict[str, float]], List[Path]]:
    """Aggregate energies across all phases for a run."""
    mm_dir = run_dir / "mm_access_compare"
    if not mm_dir.is_dir():
        return {}, []

    per_phase = sorted([p for p in mm_dir.glob("*_mm_access_compare.csv") if p.is_file()])
    if not per_phase:
        return {}, []

    # Accumulate access counts first, then convert to energy once to avoid rounding drift.
    sums: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for csv_path in per_phase:
        data = _read_mm_access_compare_csv(csv_path)
        for method, metrics in data.items():
            sums[method]["dram_sectors"] += metrics.get("dram_load_sectors_32B", 0.0) + metrics.get(
                "dram_store_sectors_32B", 0.0
            )
            sums[method]["l2_sectors"] += metrics.get("l2_load_sectors_32B", 0.0) + metrics.get(
                "l2_store_sectors_32B", 0.0
            )
            sums[method]["shmem_bytes"] += metrics.get("shmem_load_bytes", 0.0) + metrics.get(
                "shmem_store_bytes", 0.0
            )

    energies: Dict[str, Dict[str, float]] = {}
    for method, acc in sums.items():
        energies[method] = _energy_uJ(
            dram_sectors_32b=acc["dram_sectors"],
            l2_sectors_32b=acc["l2_sectors"],
            shmem_bytes=acc["shmem_bytes"],
        )
    return energies, per_phase


def _write_plot_all_runs(
    out_path: Path,
    title: str,
    by_run: List[Tuple[str, Dict[str, Dict[str, float]]]],
    display_names: Dict[str, str],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required to generate plots but could not be imported (%s).\n"
            "Install it for your interpreter, e.g.\n"
            "  python3 -m pip install matplotlib"
            % str(e)
        )

    methods = METHOD_ORDER
    comps = [
        ("DRAM", "dram_uJ", "#E45756"),
        ("L2", "l2_uJ", "#72B7B2"),
        ("SHMEM", "shmem_uJ", "#54A24B"),
    ]

    run_names = [r for r, _e in by_run]
    n = len(run_names)
    x = list(range(n))
    bar_w = 0.22
    offsets = [(-bar_w), 0.0, (bar_w)]

    # Compute per-method component arrays
    data = {m: {key: [0.0] * n for _c, key, _col in comps} for m in methods}
    for i, (_run, energies) in enumerate(by_run):
        for m in methods:
            e = energies.get(m, {})
            for _c, key, _col in comps:
                data[m][key][i] = float(e.get(key, 0.0))

    fig_w = max(12.0, min(2.4 * n, 28.0))
    fig_h = 6.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Method styling: edge + hatch
    method_style = {
        NCU_KEY: {"edgecolor": "#F58518", "hatch": "//"},
        GEMMCALC_KEY: {"edgecolor": "#4C78A8", "hatch": "xx"},
        DEEPFLOW_KEY: {"edgecolor": "#9ECAE9", "hatch": "oo"},
    }

    for m_idx, m in enumerate(methods):
        pos = [xi + offsets[m_idx] for xi in x]
        bottom = [0.0] * n
        style = method_style.get(m, {"edgecolor": "#333333", "hatch": None})
        for _cname, key, color in comps:
            vals = data[m][key]
            ax.bar(
                pos,
                vals,
                width=bar_w,
                bottom=bottom,
                color=color,
                edgecolor=style["edgecolor"],
                linewidth=1.5,
                hatch=style["hatch"],
                label=None,
            )
            bottom = [b + v for b, v in zip(bottom, vals)]

    ax.set_title(title)
    ax.set_ylabel("Energy (uJ)")
    ax.set_xticks(x)
    ax.set_xticklabels(run_names, rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    # Legends
    comp_handles = [Patch(facecolor=col, edgecolor="none", label=name) for name, _k, col in comps]
    method_handles = []
    for m in methods:
        style = method_style.get(m, {"edgecolor": "#333333", "hatch": None})
        method_handles.append(
            Patch(
                facecolor="white",
                edgecolor=style["edgecolor"],
                hatch=style["hatch"],
                label=display_names.get(m, m),
                linewidth=1.5,
            )
        )
    leg1 = ax.legend(handles=comp_handles, title="Component", loc="upper left", frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=method_handles, title="Method", ncol=3, loc="upper center", frameon=True)

    fig.text(
        0.01,
        0.01,
        "Energy model: DRAM={:.0f} pJ/sector, L2={:.0f} pJ/sector, SHMEM={:.1f} pJ/32B".format(
            DRAM_PJ_PER_SECTOR_32B, L2_PJ_PER_SECTOR_32B, SHMEM_PJ_PER_32B
        ),
        fontsize=9,
        color="#444",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=str(DEFAULT_ROOT),
        help="Root directory containing prompt_*_predict_* runs (default: %s)." % str(DEFAULT_ROOT),
    )
    parser.add_argument(
        "--out-svg",
        type=str,
        default=None,
        help="(Deprecated) Alias for --out-plot. Use a .svg suffix to write SVG output.",
    )
    parser.add_argument(
        "--out-plot",
        type=str,
        default=None,
        help="Path to write the combined plot (default: <root>/combined_mm_energy_all_runs.svg).",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Optional path to write aggregated energy CSV.",
    )
    parser.add_argument(
        "--display-name",
        action="append",
        default=[],
        help=(
            "Override method display name for plotting only. Repeatable. Format: <method_key>=<display>. "
            "method_key must match CSV columns: %s."
            % ", ".join(METHOD_ORDER)
        ),
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        print("Root not found or not a dir: %s" % root, file=sys.stderr)
        return 2

    rows: List[Tuple[str, Dict[str, Dict[str, float]]]] = []
    missing = 0
    for run_dir in _iter_run_dirs(root):
        energies, _inputs = _aggregate_run_energy(run_dir)
        if not energies:
            missing += 1
            continue
        rows.append((run_dir.name, energies))

    if not rows:
        print("No runs with mm_access_compare/*_mm_access_compare.csv found under: %s" % root, file=sys.stderr)
        return 2

    if args.out_plot:
        out_plot = Path(args.out_plot)
    elif args.out_svg:
        out_plot = Path(args.out_svg)
    else:
        out_plot = root / "combined_mm_energy_all_runs.png"

    display_names = dict(DEFAULT_DISPLAY_NAMES)
    for spec in args.display_name:
        if "=" not in spec:
            print("Ignoring malformed --display-name (expected key=value): %r" % spec, file=sys.stderr)
            continue
        key, value = spec.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key not in METHOD_ORDER:
            print(
                "Ignoring --display-name for unknown method key %r (known: %s)"
                % (key, ", ".join(METHOD_ORDER)),
                file=sys.stderr,
            )
            continue
        display_names[key] = value if value else key

    _write_plot_all_runs(
        out_path=out_plot,
        title="aten::mm energy breakdown across runs (combined phases)",
        by_run=rows,
        display_names=display_names,
    )
    print("Wrote %s" % out_plot, file=sys.stderr)

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "method", "dram_uJ", "l2_uJ", "shmem_uJ", "total_uJ"])
            for run, energies in rows:
                for method in METHOD_ORDER:
                    e = energies.get(method, {"dram_uJ": 0.0, "l2_uJ": 0.0, "shmem_uJ": 0.0, "total_uJ": 0.0})
                    w.writerow([run, method, e["dram_uJ"], e["l2_uJ"], e["shmem_uJ"], e["total_uJ"]])
        print("Wrote %s" % out_csv, file=sys.stderr)

    if missing:
        print("Skipped %d run(s) missing mm_access_compare outputs." % missing, file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
