#!/usr/bin/env python3
"""
Bucketize Nsight Compute (NCU) kernel metrics by CPU op name.

Given:
  1) An NCU-exported CSV (with columns like "Kernel Name" and metrics such as
     "dram__sectors_read.sum"), and
  2) A profiler-derived summary CSV mapping kernel names -> cpu_op_name
     (e.g. prefill_summary.csv with columns: cpu_op_name, kernel_names, ...),

this script attempts to match each profiled kernel name in the NCU CSV to a
cpu_op_name bucket, sums metrics per bucket, and reports totals.

Kernel names in the summary CSV often do not exactly match NCU kernel names.
We apply a small amount of sanitization/normalization and generate multiple
matching keys (strict + relaxed, with op markers when possible).
"""

import argparse
import csv
import io
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


RAPID_LLM_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RUN_DIR = RAPID_LLM_ROOT / "energy/imec/Llama2-7B/prompt_32_predict_4"

DEFAULT_METRICS = {
    "dram__sectors_read.sum": "DRAM Load Accesses",
    "dram__sectors_write.sum": "DRAM Store Accesses",
    "lts__t_sectors_op_read.sum": "L2 Load Accesses",
    "lts__t_sectors_op_write.sum": "L2 Store Accesses",
    "lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_miss.sum": "L2 Read Misses to DRAM",
    "lts__t_sectors_srcunit_tex_aperture_device_op_write_lookup_miss.sum": "L2 Write Misses to DRAM",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum": "L2 Load Hits",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum": "L2 Load Misses",
    "lts__t_sectors_srcunit_tex_op_write_lookup_hit.sum": "L2 Store Hits",
    "lts__t_sectors_srcunit_tex_op_write_lookup_miss.sum": "L2 Store Misses",
    "sm__sass_data_bytes_mem_shared_op_ld.sum": "SHMEM LD Bytes",
    "sm__sass_data_bytes_mem_shared_op_ldsm.sum": "SHMEM LDSM Bytes",
    "sm__sass_data_bytes_mem_shared_op_st.sum": "SHMEM ST Bytes",
    "sm__sass_data_bytes_mem_shared_op_ldgsts.sum": "SHMEM LDGSTS Bytes",
    # "sm__sass_data_bytes_mem_shared_op_ldgsts_cache_access.sum": "SHMEM LDGSTS Cache Access Bytes",
    # "sm__sass_data_bytes_mem_shared_op_ldgsts_cache_bypass.sum": "SHMEM LDGSTS Cache Bypass Bytes",
}

_DECODE_SUMMARY_RE = re.compile(r"^decode_(\d+)_summary\.csv$")

_DRAM_READ_METRIC = "dram__sectors_read.sum"
_DRAM_WRITE_METRIC = "dram__sectors_write.sum"

_L2_READ_METRIC = "lts__t_sectors_op_read.sum"
_L2_WRITE_METRIC = "lts__t_sectors_op_write.sum"

_SHMEM_LD_BYTES_METRIC = "sm__sass_data_bytes_mem_shared_op_ld.sum"
_SHMEM_LDSM_BYTES_METRIC = "sm__sass_data_bytes_mem_shared_op_ldsm.sum"
_SHMEM_ST_BYTES_METRIC = "sm__sass_data_bytes_mem_shared_op_st.sum"


def _strip_top_level_params(s):
    """Strip the first top-level '(...)' parameter list (but not parentheses within templates)."""
    depth = 0
    for i, ch in enumerate(s):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif ch == "(" and depth == 0:
            return s[:i]
    return s


def _normalize_kernel_name(s):
    """Normalize a kernel name to improve match rate across sources."""
    s = (s or "").strip()
    if not s:
        return ""

    # Normalize some common source differences:
    # - prefill_summary often uses at::native::; NCU often uses at::
    # - prefill_summary uses "(anonymous namespace)" while NCU may use "<unnamed>" or similar
    s = re.sub(r"^void\s+", "", s)
    s = s.replace("at::native::", "at::")
    # Avoid literal angle brackets here: later parsing uses '<' to detect templates.
    s = s.replace("(anonymous namespace)", "unnamed")
    s = s.replace("(anonymous_namespace)", "unnamed")
    s = s.replace("<unnamed>", "unnamed")

    # Normalize NCU-style lambdas: ::[lambda() (instance 3)] -> ::{lambda()#3}
    s = re.sub(r"\[lambda\(\)\s*\(instance\s*(\d+)\)\]", r"{lambda()#\1}", s)

    # Remove all whitespace for stability.
    s = re.sub(r"\s+", "", s)

    # Drop top-level params.
    s = _strip_top_level_params(s)
    return s


def _extract_base_symbol(s_norm):
    """Return the leading symbol before the first template '<...>' (if present)."""
    return s_norm.split("<", 1)[0]


def _extract_op_marker(s_norm):
    """Extract an 'op marker' token that helps disambiguate generic kernel wrappers."""
    patterns = [
        # Often uniquely identifies the CPU op for *_out kernels.
        r"([A-Za-z0-9_]+_cuda_out)",
        # Common, highly specific.
        r"([A-Za-z0-9_]+_kernel_cuda)",
        # Binary internal functors (e.g. MulFunctor)
        r"binary_internal::([A-Za-z0-9_]+Functor)",
        # CUDAFunctor_* variants (often used for add/copy/etc.)
        r"(CUDAFunctorOnSelf_[A-Za-z0-9_]+)",
        r"(CUDAFunctor_[A-Za-z0-9_]+)",
        r"(CUDAFunctor[A-Za-z0-9_]+)",
        # Reductions (e.g. MeanOps)
        r"(MeanOps|SumOps|MaxOps|MinOps)",
        # Cat kernels often embed this helper
        r"(CatArrayBatchedCopy)",
        # Some ops show up as *_kernel (without _cuda suffix) in templates.
        r"(silu_kernel)",
        r"(pow_tensor_scalar_kernel_impl)",
        # Generic fallback (avoid returning gpu_kernel_impl_nocast; it's too generic).
        r"([A-Za-z0-9_]+_kernel_impl[A-Za-z0-9_]*)",
    ]
    for pat in patterns:
        m = re.search(pat, s_norm, flags=re.IGNORECASE)
        if m:
            marker = m.group(1)
            # gpu_kernel_impl_nocast matches the generic pattern but is not an op identifier.
            if marker.lower().startswith("gpu_kernel_impl"):
                continue
            return marker
    return None


def _kernel_keys(name):
    """Generate match keys from most- to least-specific."""
    n = _normalize_kernel_name(name).lower()
    if not n:
        return []

    base = _extract_base_symbol(n)
    marker = _extract_op_marker(n)

    out = [n]
    if marker:
        out.append("%s|%s" % (base, marker.lower()))
    out.append(base)
    return out


def _parse_number(value):
    """Parse NCU numeric cells (often include thousands separators)."""
    if value is None:
        return 0.0
    s = str(value).strip().strip('"')
    if not s or s.lower() in ("n/a", "na", "nan", "none"):
        return 0.0
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return 0.0


def _read_prefill_summary_kernel_to_cpu_op(prefill_summary_csv):
    """Build a mapping key -> Counter(cpu_op_name) from prefill_summary.csv."""
    key_to_ops = defaultdict(Counter)
    with prefill_summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Empty CSV or missing header: %s" % prefill_summary_csv)
        if "cpu_op_name" not in reader.fieldnames or "kernel_names" not in reader.fieldnames:
            raise ValueError(
                "Expected columns cpu_op_name and kernel_names in %s; got: %s"
                % (prefill_summary_csv, reader.fieldnames)
            )
        for row in reader:
            cpu_op = (row.get("cpu_op_name") or "").strip()
            kernel_names = row.get("kernel_names") or ""
            # kernel_names is usually 1 name, but occasionally contains multiple separated by ';'
            for k in [p.strip() for p in kernel_names.split(";") if p.strip()]:
                for key in _kernel_keys(k):
                    key_to_ops[key][cpu_op] += 1
    return key_to_ops


def _find_ncu_header_start(lines):
    for i, line in enumerate(lines):
        # Typical NCU CSV header begins with "ID","Process ID",...,"Kernel Name",...
        if line.startswith('"ID"') and '"Kernel Name"' in line:
            return i
    return None


def _read_ncu_rows(ncu_csv):
    """Return a csv.DictReader over NCU rows, skipping leading non-CSV noise lines."""
    text = ncu_csv.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = _find_ncu_header_start(text)
    if start is None:
        raise ValueError("Could not find NCU CSV header row in %s" % ncu_csv)
    return csv.DictReader(io.StringIO("\n".join(text[start:])))


def _choose_cpu_op_for_kernel(kernel_name, key_to_ops):
    """Return (cpu_op_name, match_key, ambiguous:bool) or (None, None, False)."""
    best_ambiguous = None  # (cpu_op, key)
    for key in _kernel_keys(kernel_name):
        ops = key_to_ops.get(key)
        if not ops:
            continue
        if len(ops) == 1:
            return next(iter(ops.keys())), key, False
        # Ambiguous key: keep best guess but try other keys (which may disambiguate).
        if best_ambiguous is None:
            best_ambiguous = (ops.most_common(1)[0][0], key)
    if best_ambiguous is not None:
        return best_ambiguous[0], best_ambiguous[1], True
    return None, None, False


def _gemm_like_cpu_op_fallback(kernel_name, cpu_ops_in_summary):
    """Heuristic fallback for GEMM-like kernels absent from prefill_summary mapping."""
    s = (kernel_name or "").lower()
    if not s:
        return None
    if not any(tok in s for tok in ("gemm", "sgemm", "hgemm", "xmma_gemm", "cublas", "cutlass")):
        return None
    if "stridedbatched" in s or "batched" in s:
        if "aten::bmm" in cpu_ops_in_summary:
            return "aten::bmm"
    if "aten::mm" in cpu_ops_in_summary:
        return "aten::mm"
    return None


def _format_metric_column(metric_name, metric_labels):
    label = metric_labels.get(metric_name)
    if label:
        return "%s" % (label)
    return metric_name


def _read_bucketized_dram_counts(bucketized_csv, metric_labels):
    """Return dict cpu_op -> (dram_read, dram_write) from a bucketized CSV."""
    return _read_bucketized_counts(
        bucketized_csv=bucketized_csv,
        metric_labels=metric_labels,
        metric_names=[_DRAM_READ_METRIC, _DRAM_WRITE_METRIC],
    )

def _read_bucketized_counts(bucketized_csv, metric_labels, metric_names):
    """Return dict cpu_op -> tuple(values...) for the requested metric_names."""
    missing_from_labels = [m for m in metric_names if m not in metric_labels]
    if missing_from_labels:
        raise ValueError(
            "DEFAULT_METRICS missing required metrics: %s" % ", ".join(missing_from_labels)
        )

    cols = [_format_metric_column(m, metric_labels) for m in metric_names]
    with bucketized_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Empty bucketized CSV: %s" % bucketized_csv)
        if "cpu_op_name" not in reader.fieldnames:
            raise ValueError("Bucketized CSV missing cpu_op_name column: %s" % bucketized_csv)
        for c in cols:
            if c not in reader.fieldnames:
                raise ValueError(
                    "Bucketized CSV missing required column (%s) for %s" % (c, bucketized_csv)
                )

        out = {}
        for row in reader:
            cpu_op = (row.get("cpu_op_name") or "").strip()
            if not cpu_op:
                continue
            values = tuple(_parse_number(row.get(c)) for c in cols)
            out[cpu_op] = values
        return out


def _plot_pareto_stacked(counts_by_cpu_op, series_labels, series_colors, y_label, title, out_path):
    """Pareto chart: stacked bars + cumulative percent line."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required to generate plots but could not be imported (%s)." % str(e)
        )

    series_count = len(series_labels)
    if series_count != len(series_colors):
        raise ValueError("series_labels and series_colors must have same length.")
    if series_count <= 0:
        raise ValueError("No series provided to plot.")

    items = []
    for cpu_op, values in counts_by_cpu_op.items():
        if values is None:
            continue
        vals = list(values)
        if len(vals) != series_count:
            raise ValueError(
                "Mismatched series count for %s: expected %d values, got %d"
                % (cpu_op, series_count, len(vals))
            )
        total = float(sum(float(v) for v in vals))
        items.append((cpu_op, [float(v) for v in vals], total))
    items.sort(key=lambda x: x[2], reverse=True)

    labels = [x[0] for x in items]
    series = list(zip(*[x[1] for x in items])) if items else [[] for _ in range(series_count)]
    totals = [x[2] for x in items]
    total_sum = sum(totals) or 1.0

    cum = []
    run = 0.0
    for t in totals:
        run += t
        cum.append(100.0 * run / total_sum)

    fig_w = max(12.0, min(0.6 * len(labels), 30.0))
    fig_h = 7.0
    fig, ax1 = plt.subplots(figsize=(fig_w, fig_h))

    x = list(range(len(labels)))
    bottom = [0.0 for _ in x]
    for idx in range(series_count):
        values = list(series[idx]) if series else [0.0 for _ in x]
        ax1.bar(
            x,
            values,
            bottom=bottom,
            label=series_labels[idx],
            color=series_colors[idx],
        )
        bottom = [b + v for b, v in zip(bottom, values)]
    ax1.set_ylabel(y_label)
    ax1.set_title(title)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax2 = ax1.twinx()
    ax2.plot(x, cum, color="#54A24B", marker="o", linewidth=2.0, label="Cumulative %")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Cumulative %")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def _plot_pareto_stacked_dram(counts_by_cpu_op, title, out_path):
    return _plot_pareto_stacked(
        counts_by_cpu_op=counts_by_cpu_op,
        series_labels=["DRAM Load", "DRAM Store"],
        series_colors=["#4C78A8", "#F58518"],
        y_label="DRAM sectors",
        title=title,
        out_path=out_path,
    )


def _plot_pareto_stacked_l2(counts_by_cpu_op, title, out_path):
    return _plot_pareto_stacked(
        counts_by_cpu_op=counts_by_cpu_op,
        series_labels=["L2 Load", "L2 Store"],
        series_colors=["#4C78A8", "#F58518"],
        y_label="L2 sectors",
        title=title,
        out_path=out_path,
    )


def _plot_pareto_stacked_shmem(counts_by_cpu_op, title, out_path):
    return _plot_pareto_stacked(
        counts_by_cpu_op=counts_by_cpu_op,
        series_labels=["SHMEM LD", "SHMEM LDSM", "SHMEM ST"],
        series_colors=["#4C78A8", "#9ECae9", "#F58518"],
        y_label="Shared memory bytes",
        title=title,
        out_path=out_path,
    )


def _bucketize_ncu_csv_by_cpu_op(ncu_csv, kernel_summary_csv, metric_labels, no_heuristics=False):
    key_to_ops = _read_prefill_summary_kernel_to_cpu_op(kernel_summary_csv)
    cpu_ops_in_summary = set()
    for counter in key_to_ops.values():
        cpu_ops_in_summary.update(counter.keys())

    reader = _read_ncu_rows(ncu_csv)
    if not reader.fieldnames:
        raise ValueError("NCU CSV appears empty: %s" % ncu_csv)

    # NCU exports sometimes include both "Kernel Name" and "launch__kernel_name".
    kernel_name_col = None
    for candidate in ("Kernel Name", "launch__kernel_name"):
        if candidate in reader.fieldnames:
            kernel_name_col = candidate
            break
    if not kernel_name_col:
        raise ValueError(
            'NCU CSV missing kernel name column ("Kernel Name" or "launch__kernel_name"): %s'
            % ncu_csv
        )

    metrics = list(metric_labels.keys())
    if not metrics:
        raise ValueError("DEFAULT_METRICS is empty; nothing to aggregate.")

    missing = [m for m in metrics if m not in reader.fieldnames]
    if missing:
        lines = ["Missing metric columns in NCU CSV: %s" % ncu_csv]
        for m in missing:
            lines.append("  %s" % _format_metric_column(m, metric_labels))
        raise ValueError("\n".join(lines))

    bucket_totals_by_metric = {m: Counter() for m in metrics}
    match_method_counts = Counter()
    unmatched_kernels = set()

    for row in reader:
        kernel_name = row.get(kernel_name_col) or ""

        cpu_op, match_key, ambiguous = _choose_cpu_op_for_kernel(kernel_name, key_to_ops)
        if cpu_op is None and not no_heuristics:
            cpu_op = _gemm_like_cpu_op_fallback(kernel_name, cpu_ops_in_summary)
            if cpu_op is not None:
                match_method_counts["heuristic_gemm_like"] += 1

        if cpu_op is None:
            cpu_op = "__unmatched__"
            unmatched_kernels.add(kernel_name)
            match_method_counts["unmatched"] += 1
        else:
            if match_key is not None:
                if ambiguous:
                    match_method_counts["matched_ambiguous_key"] += 1
                else:
                    match_method_counts["matched"] += 1
            else:
                match_method_counts["matched"] += 1

        for m in metrics:
            bucket_totals_by_metric[m][cpu_op] += _parse_number(row.get(m))

    cpu_ops = set()
    for c in bucket_totals_by_metric.values():
        cpu_ops.update(c.keys())

    sort_metric = metrics[0]
    cpu_ops_sorted = sorted(
        cpu_ops,
        key=lambda op: bucket_totals_by_metric.get(sort_metric, Counter()).get(op, 0.0),
        reverse=True,
    )

    metric_totals = []
    for m in metrics:
        label = metric_labels.get(m, m)
        total = float(sum(bucket_totals_by_metric[m].values()))
        if abs(total - int(total)) < 1e-9:
            metric_totals.append("%s=%d" % (label, int(total)))
        else:
            metric_totals.append("%s=%.6f" % (label, total))

    return {
        "metrics": metrics,
        "cpu_ops_sorted": cpu_ops_sorted,
        "bucket_totals_by_metric": bucket_totals_by_metric,
        "metric_totals": metric_totals,
        "match_method_counts": match_method_counts,
        "unmatched_kernels": unmatched_kernels,
    }


def _find_step_ncu_csv(ncu_dir, step_name):
    if not ncu_dir.is_dir():
        return None

    # Prefer exact suffix matches (stable across different prefixes).
    exact = [p for p in ncu_dir.glob("*.csv") if p.name.endswith("_%s.csv" % step_name)]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return sorted(exact, key=lambda p: len(p.name))[0]

    # Fallback: any file containing the step name.
    candidates = list(ncu_dir.glob("*%s*.csv" % step_name))
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: (len(p.name), p.name))[0]


def _discover_steps(run_dir):
    steps = []
    prefill_summary = run_dir / "prefill_summary.csv"
    if prefill_summary.exists():
        steps.append(("prefill", prefill_summary))

    decode = []
    for p in run_dir.glob("decode_*_summary.csv"):
        m = _DECODE_SUMMARY_RE.match(p.name)
        if not m:
            continue
        decode.append((int(m.group(1)), p))
    for idx, p in sorted(decode):
        steps.append(("decode_%d" % idx, p))
    return steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=str,
        default=str(DEFAULT_RUN_DIR),
        help="Run directory containing *_summary.csv and an ncu/ subdir (default: %s)." % str(DEFAULT_RUN_DIR),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to write per-step bucketized CSVs (default: <run-dir>/bucketized_ncu).",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=None,
        help="Directory to write pareto plots (default: <out-dir>/plots).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="Disable generating pareto plots from bucketized CSVs.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="(Deprecated) Kept for backward compatibility; ignored in --run-dir mode.",
    )
    parser.add_argument(
        "--print-unmatched",
        action="store_true",
        default=False,
        help="Print unmatched kernel names to stderr.",
    )
    parser.add_argument(
        "--no-heuristics",
        action="store_true",
        default=False,
        help="Disable heuristic fallbacks (e.g., mapping GEMM-like kernels to aten::mm).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print("Run dir not found: %s" % run_dir, file=sys.stderr)
        return 2
    if not run_dir.is_dir():
        print("Run dir is not a directory: %s" % run_dir, file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "bucketized_ncu")
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(args.plots_dir) if args.plots_dir else (out_dir / "plots")

    ncu_dir = run_dir / "ncu"
    steps = _discover_steps(run_dir)
    if not steps:
        print("No steps found in %s (expected prefill_summary.csv / decode_*_summary.csv)." % run_dir, file=sys.stderr)
        return 2
    if not ncu_dir.is_dir():
        print("Missing ncu/ directory under run dir: %s" % ncu_dir, file=sys.stderr)
        return 2

    metric_labels = DEFAULT_METRICS
    metrics = list(metric_labels.keys())
    if not metrics:
        print("DEFAULT_METRICS is empty; nothing to aggregate.", file=sys.stderr)
        return 2

    failures = 0
    for step_name, summary_csv in steps:
        ncu_csv = _find_step_ncu_csv(ncu_dir, step_name)
        if not ncu_csv:
            failures += 1
            print("Step %s: missing NCU CSV under %s" % (step_name, ncu_dir), file=sys.stderr)
            continue

        out_csv = out_dir / ("%s_bucketized_ncu.csv" % step_name)
        try:
            result = _bucketize_ncu_csv_by_cpu_op(
                ncu_csv=ncu_csv,
                kernel_summary_csv=summary_csv,
                metric_labels=metric_labels,
                no_heuristics=args.no_heuristics,
            )
        except Exception as e:
            failures += 1
            print("Step %s: FAILED (%s)" % (step_name, str(e)), file=sys.stderr)
            continue

        with out_csv.open("w", encoding="utf-8", newline="") as out_f:
            w = csv.writer(out_f)
            header = ["cpu_op_name"]
            for m in result["metrics"]:
                header.append(_format_metric_column(m, metric_labels))
            w.writerow(header)
            for cpu_op in result["cpu_ops_sorted"]:
                row_out = [cpu_op]
                for m in result["metrics"]:
                    v = result["bucket_totals_by_metric"][m].get(cpu_op, 0.0)
                    if abs(v - int(v)) < 1e-9:
                        row_out.append(str(int(v)))
                    else:
                        row_out.append("%.6f" % v)
                w.writerow(row_out)

        print(
            "Step %s: wrote %s | Buckets=%d | Totals: %s"
            % (
                step_name,
                out_csv,
                len(result["cpu_ops_sorted"]),
                ", ".join(result["metric_totals"]) or "none",
            ),
            file=sys.stderr,
        )
        print(
            "Step %s: match counts: %s"
            % (
                step_name,
                ", ".join(
                    "%s=%d" % (k, v) for k, v in result["match_method_counts"].most_common()
                )
                or "none",
            ),
            file=sys.stderr,
        )
        if result["unmatched_kernels"]:
            print(
                "Step %s: unmatched unique kernels: %d"
                % (step_name, len(result["unmatched_kernels"])),
                file=sys.stderr,
            )
            if args.print_unmatched:
                for k in sorted(result["unmatched_kernels"]):
                    if k.strip():
                        print("  %s" % k, file=sys.stderr)

    plot_failures = 0
    if not args.no_plots:
        # Plot from the bucketized_ncu CSVs on disk.
        combined_dram = {}
        combined_l2 = {}
        combined_shmem = {}
        bucket_csvs = sorted(out_dir.glob("*_bucketized_ncu.csv"))
        for bucket_csv in bucket_csvs:
            step_name = bucket_csv.name
            if step_name.endswith("_bucketized_ncu.csv"):
                step_name = step_name[: -len("_bucketized_ncu.csv")]
            # DRAM
            try:
                dram = _read_bucketized_counts(
                    bucketized_csv=bucket_csv,
                    metric_labels=metric_labels,
                    metric_names=[_DRAM_READ_METRIC, _DRAM_WRITE_METRIC],
                )
                for cpu_op, (r, w) in dram.items():
                    prev = combined_dram.get(cpu_op)
                    if prev is None:
                        combined_dram[cpu_op] = (r, w)
                    else:
                        combined_dram[cpu_op] = (prev[0] + r, prev[1] + w)
                out_png = plots_dir / ("%s_dram_pareto.png" % step_name)
                _plot_pareto_stacked_dram(
                    counts_by_cpu_op=dram,
                    title="%s DRAM Access Pareto" % step_name,
                    out_path=out_png,
                )
                print("Plots %s: wrote %s" % (step_name, out_png), file=sys.stderr)
            except Exception as e:
                plot_failures += 1
                print("Plots %s (DRAM): FAILED (%s)" % (step_name, str(e)), file=sys.stderr)

            # L2
            try:
                l2 = _read_bucketized_counts(
                    bucketized_csv=bucket_csv,
                    metric_labels=metric_labels,
                    metric_names=[_L2_READ_METRIC, _L2_WRITE_METRIC],
                )
                for cpu_op, (r, w) in l2.items():
                    prev = combined_l2.get(cpu_op)
                    if prev is None:
                        combined_l2[cpu_op] = (r, w)
                    else:
                        combined_l2[cpu_op] = (prev[0] + r, prev[1] + w)
                out_png = plots_dir / ("%s_l2_pareto.png" % step_name)
                _plot_pareto_stacked_l2(
                    counts_by_cpu_op=l2,
                    title="%s L2 Access Pareto" % step_name,
                    out_path=out_png,
                )
                print("Plots %s: wrote %s" % (step_name, out_png), file=sys.stderr)
            except Exception as e:
                plot_failures += 1
                print("Plots %s (L2): FAILED (%s)" % (step_name, str(e)), file=sys.stderr)

            # SHMEM
            try:
                shmem = _read_bucketized_counts(
                    bucketized_csv=bucket_csv,
                    metric_labels=metric_labels,
                    metric_names=[
                        _SHMEM_LD_BYTES_METRIC,
                        _SHMEM_LDSM_BYTES_METRIC,
                        _SHMEM_ST_BYTES_METRIC,
                    ],
                )
                for cpu_op, (ld, ldsm, st) in shmem.items():
                    prev = combined_shmem.get(cpu_op)
                    if prev is None:
                        combined_shmem[cpu_op] = (ld, ldsm, st)
                    else:
                        combined_shmem[cpu_op] = (
                            prev[0] + ld,
                            prev[1] + ldsm,
                            prev[2] + st,
                        )
                out_png = plots_dir / ("%s_shmem_pareto.png" % step_name)
                _plot_pareto_stacked_shmem(
                    counts_by_cpu_op=shmem,
                    title="%s SHMEM Access Pareto" % step_name,
                    out_path=out_png,
                )
                print("Plots %s: wrote %s" % (step_name, out_png), file=sys.stderr)
            except Exception as e:
                plot_failures += 1
                print("Plots %s (SHMEM): FAILED (%s)" % (step_name, str(e)), file=sys.stderr)

        if combined_dram and bucket_csvs:
            try:
                out_png = plots_dir / "combined_dram_pareto.png"
                _plot_pareto_stacked_dram(
                    counts_by_cpu_op=combined_dram,
                    title="Combined DRAM Access Pareto",
                    out_path=out_png,
                )
                print("Plots combined: wrote %s" % out_png, file=sys.stderr)
            except Exception as e:
                plot_failures += 1
                print("Plots combined: FAILED (%s)" % str(e), file=sys.stderr)
        if combined_l2 and bucket_csvs:
            try:
                out_png = plots_dir / "combined_l2_pareto.png"
                _plot_pareto_stacked_l2(
                    counts_by_cpu_op=combined_l2,
                    title="Combined L2 Access Pareto",
                    out_path=out_png,
                )
                print("Plots combined: wrote %s" % out_png, file=sys.stderr)
            except Exception as e:
                plot_failures += 1
                print("Plots combined: FAILED (%s)" % str(e), file=sys.stderr)
        if combined_shmem and bucket_csvs:
            try:
                out_png = plots_dir / "combined_shmem_pareto.png"
                _plot_pareto_stacked_shmem(
                    counts_by_cpu_op=combined_shmem,
                    title="Combined SHMEM Access Pareto",
                    out_path=out_png,
                )
                print("Plots combined: wrote %s" % out_png, file=sys.stderr)
            except Exception as e:
                plot_failures += 1
                print("Plots combined: FAILED (%s)" % str(e), file=sys.stderr)

    if failures:
        print("Completed with %d failure(s). Output dir: %s" % (failures, out_dir), file=sys.stderr)
        return 1

    if plot_failures:
        print("Completed with %d plot failure(s). Output dir: %s" % (plot_failures, out_dir), file=sys.stderr)
        return 1

    print("Done. Output dir: %s" % out_dir, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
