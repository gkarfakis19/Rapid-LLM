#!/usr/bin/env python3
"""
Generate an inference speedup comparison across selected H100 SXM5 hardware cases
for DeepSeek-V3 (MLA) and GLM 4.7 358B (GQA).

The script sweeps GBS via model-config variants and aggregates the best non-OOM
result per hardware case / GPU count / model family across those GBS values.
"""

import argparse
import csv
import math
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import yaml

import parallelism_sweep as ps


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
OUTPUT_PATH = TOOLS_DIR / "parallelism_speedup_per_gpu_combined_inference_compare.png"
OUTPUT_PARENT_DIR = TOOLS_DIR / "parallelism_results"
SWEEP_SCRIPT = TOOLS_DIR / "parallelism_sweep.py"
DERATE_CONFIG_PATH = (
    REPO_ROOT / "validation_scripts" / "validation_configs" / "harness_derates.yaml"
)
DERATE_DEVICE_TYPE = "H100_SXM5"

GPU_COUNTS = [32, 48, 64, 80]
GBS_VALUES = [1, 2, 4, 8]
EP_VALUES = [1, 2, 4]

# Inference study keeps the five requested hardware points:
# Base, compute uplift, memory capacity uplift, HBM bandwidth uplift,
# HBM-throttled bandwidth, and end-to-end network uplift.
CASE_CONFIGS = [
    (
        "Base",
        REPO_ROOT
        / "validation_scripts"
        / "validation_configs"
        / "hardware-config"
        / "H100_SXM5_80GB_base.yaml",
    ),
    (
        "Case A",
        REPO_ROOT
        / "validation_scripts"
        / "validation_configs"
        / "hardware-config"
        / "H100_SXM5_80GB_case-A.yaml",
    ),
    (
        "Case B",
        REPO_ROOT
        / "validation_scripts"
        / "validation_configs"
        / "hardware-config"
        / "H100_SXM5_80GB_case-B.yaml",
    ),
    (
        "Case C",
        REPO_ROOT
        / "validation_scripts"
        / "validation_configs"
        / "hardware-config"
        / "H100_SXM5_80GB_case-C.yaml",
    ),
    (
        "Case D",
        REPO_ROOT
        / "validation_scripts"
        / "validation_configs"
        / "hardware-config"
        / "H100_SXM5_80GB_case-D.yaml",
    ),
    (
        "Case E",
        REPO_ROOT
        / "validation_scripts"
        / "validation_configs"
        / "hardware-config"
        / "H100_SXM5_80GB_case-E.yaml",
    ),
]

MODEL_FAMILIES = [
    {
        "label": "DeepSeekV3",
        "attention_label": "DeepSeekV3 / MLA",
        "base_model_config": (
            REPO_ROOT
            / "validation_scripts"
            / "validation_configs"
            / "model-config"
            / "DeepSeekV3_inf_16k.yaml"
        ),
    },
    {
        "label": "GLM4.7-358B",
        "attention_label": "GLM4.7-358B / GQA",
        "base_model_config": (
            REPO_ROOT
            / "validation_scripts"
            / "validation_configs"
            / "model-config"
            / "GLM4.7_358B_inf_16k.yaml"
        ),
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the H100 SXM5 inference speedup comparison for DeepSeek-V3 and GLM 4.7."
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip sweep execution and only regenerate the figure from existing TSV reports.",
    )
    parser.add_argument(
        "--gpu-counts",
        type=str,
        default="",
        help="Comma-separated GPU counts to include in the plot and sweep bounds.",
    )
    parser.add_argument(
        "--gbs-values",
        type=str,
        default=",".join(str(v) for v in GBS_VALUES),
        help="Comma-separated global batch sizes to sweep as model variants.",
    )
    parser.add_argument(
        "--ep-values",
        type=str,
        default=",".join(str(v) for v in EP_VALUES),
        help="Comma-separated EP values to include in the sweep.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="tokens_per_s",
        choices=("runtime", "decode", "prefill", "requests_per_s", "tokens_per_s"),
        help="Metric to compare in the speedup plot.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Optional directory where sweep TSVs/PNGs and the combined plot will be stored.",
    )
    return parser.parse_args()


def _sanitize_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))


def _parse_int_list(arg: str, default_values):
    if not str(arg or "").strip():
        return list(default_values)
    values = []
    for raw in str(arg).split(","):
        text = raw.strip()
        if text:
            values.append(int(text))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return sorted(set(values))


def _python_executable() -> str:
    preferred = [
        REPO_ROOT / "venv" / "bin" / "python",
        REPO_ROOT / ".venv" / "bin" / "python",
    ]
    for repo_venv_python in preferred:
        if repo_venv_python.exists():
            return str(repo_venv_python)
    return sys.executable


def _metric_title(metric: str) -> str:
    return {
        "runtime": "Total Inference Time",
        "decode": "Decode Time",
        "prefill": "Prefill Time",
        "requests_per_s": "Request Throughput",
        "tokens_per_s": "Token Throughput",
    }[metric]


def _metric_is_lower_better(metric: str) -> bool:
    return metric in {"runtime", "decode", "prefill"}


def _load_model_metadata(model_config_path: Path) -> dict:
    with open(model_config_path, "r") as handle:
        raw = yaml.safe_load(handle) or {}
    model_param = raw.get("model_param", {}) or {}
    gbs = int(model_param.get("global_batch_size", 1) or 1)
    seq_len = int(model_param.get("seq_len", 0) or 0)
    return {
        "global_batch_size": gbs,
        "seq_len": seq_len,
        "total_tokens_per_batch": gbs * seq_len,
    }


def _metric_value_from_row(row: dict, metric: str, model_meta: dict):
    runtime_s = float(row.get("runtime_s", "nan"))
    if metric == "runtime":
        return runtime_s
    if metric == "decode":
        return float(row.get("decode_time_s", "nan"))
    if metric == "prefill":
        return float(row.get("prefill_time_s", "nan"))
    if not math.isfinite(runtime_s) or runtime_s <= 0:
        return float("nan")
    if metric == "requests_per_s":
        return float(model_meta["global_batch_size"]) / runtime_s
    if metric == "tokens_per_s":
        return float(model_meta["total_tokens_per_batch"]) / runtime_s
    raise KeyError(f"Unsupported metric: {metric}")


def _stage_output_tag(model_label: str, gbs: int, gpu_count: int, metric: str) -> str:
    metric_suffix = "" if metric == "runtime" else f"_{metric}"
    return _sanitize_tag(f"{model_label}_gbs{gbs}_{gpu_count}gpus{metric_suffix}")


def _report_path_for_hw_config(hw_config_path: Path, output_root: Path, output_tag: str) -> Path:
    return output_root / f"parallelism_sweep_{output_tag}_{hw_config_path.stem}.tsv"


def _output_plot_path(output_root: Path, gpu_counts, metric: str) -> Path:
    gpu_part = "-".join(str(count) for count in gpu_counts)
    metric_suffix = "" if metric == "runtime" else f"_{metric}"
    return output_root / f"{OUTPUT_PATH.stem}_{gpu_part}gpus{metric_suffix}{OUTPUT_PATH.suffix}"


def _family_plot_path(output_root: Path, family_label: str, gpu_counts, metric: str) -> Path:
    gpu_part = "-".join(str(count) for count in gpu_counts)
    metric_suffix = "" if metric == "runtime" else f"_{metric}"
    tag = _sanitize_tag(family_label)
    return output_root / f"parallelism_speedup_per_gpu_{tag}_{gpu_part}gpus{metric_suffix}.png"


def _variant_dir(output_root: Path) -> Path:
    path = output_root / "model_variants"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _materialize_model_variant(base_model_config: Path, output_root: Path, gbs: int) -> Path:
    with open(base_model_config, "r") as handle:
        raw = yaml.safe_load(handle) or {}
    model_param = raw.setdefault("model_param", {})
    model_param["global_batch_size"] = int(gbs)
    variant_path = _variant_dir(output_root) / f"{base_model_config.stem}_gbs{int(gbs)}.yaml"
    text = yaml.safe_dump(raw, default_flow_style=False, sort_keys=False)
    if variant_path.exists():
        try:
            if variant_path.read_text() == text:
                return variant_path
        except OSError:
            pass
    with open(variant_path, "w") as handle:
        handle.write(text)
    return variant_path


def _load_best_no_oom(path: Path, gpu_counts, metric: str, model_meta: dict):
    best = {}
    if not path.exists():
        raise FileNotFoundError(f"Missing sweep report: {path}")
    lower_is_better = _metric_is_lower_better(metric)
    with open(path, "r") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            try:
                num_gpus = int(row["num_gpus"])
            except (KeyError, ValueError):
                continue
            if num_gpus not in gpu_counts:
                continue
            if str(row.get("memory_exceeded", "")).strip().lower() == "true":
                continue
            try:
                value = _metric_value_from_row(row, metric, model_meta)
            except (KeyError, ValueError, TypeError):
                continue
            if not math.isfinite(value):
                continue
            prev = best.get(num_gpus)
            if prev is None or (value < prev["metric_value"] if lower_is_better else value > prev["metric_value"]):
                best[num_gpus] = {
                    "metric_value": value,
                    "runtime_s": float(row.get("runtime_s", "nan")),
                    "prefill_time_s": float(row.get("prefill_time_s", "nan")),
                    "decode_time_s": float(row.get("decode_time_s", "nan")),
                    "parallelism": row.get("parallelism", ""),
                    "report_path": str(path),
                }
    return best


def _run_sweep_for_variant(
    model_config_path: Path,
    gpu_count: int,
    output_tag: str,
    output_root: Path,
    ep_values: str,
):
    hardware_configs = ",".join(str(path) for _, path in CASE_CONFIGS)
    labels = ",".join(label for label, _ in CASE_CONFIGS)
    cmd = [
        _python_executable(),
        str(SWEEP_SCRIPT),
        "--hardware-configs",
        hardware_configs,
        "--hardware-labels",
        labels,
        "--model-config",
        str(model_config_path),
        "--output-tag",
        output_tag,
        "--output-root",
        str(output_root),
        "--gpu-count-min",
        str(gpu_count),
        "--gpu-count-max",
        str(gpu_count),
        "--ep-values",
        str(ep_values),
        "--derate-config",
        str(DERATE_CONFIG_PATH),
        "--derate-device-type",
        DERATE_DEVICE_TYPE,
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _write_family_summary(output_root: Path, family_label: str, rows) -> Path:
    path = output_root / f"inference_best_summary_{_sanitize_tag(family_label)}.tsv"
    with open(path, "w") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "model_family",
                "gpu_count",
                "case",
                "gbs",
                "metric_value",
                "runtime_s",
                "prefill_time_s",
                "decode_time_s",
                "parallelism",
                "report_path",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["model_family"],
                    row["gpu_count"],
                    row["case"],
                    row["gbs"],
                    row["metric_value"],
                    row["runtime_s"],
                    row["prefill_time_s"],
                    row["decode_time_s"],
                    row["parallelism"],
                    row["report_path"],
                ]
            )
    print(f"Wrote summary to {path}")
    return path


def _aggregate_family_results(
    family: dict,
    load_gpu_counts,
    run_gpu_counts,
    gbs_values,
    metric: str,
    output_root: Path,
    plot_only: bool,
    ep_values: str,
):
    case_best = {label: {} for label, _ in CASE_CONFIGS}
    summary_rows = []

    run_gpu_count_set = set(run_gpu_counts)
    load_gpu_count_set = set(load_gpu_counts)

    for gpu_count in sorted(load_gpu_count_set):
        for gbs in gbs_values:
            variant_model_config = _materialize_model_variant(
                family["base_model_config"],
                output_root,
                gbs,
            )
            model_meta = _load_model_metadata(variant_model_config)
            output_tag = _stage_output_tag(family["label"], gbs, gpu_count, metric)
            if not plot_only and gpu_count in run_gpu_count_set:
                _run_sweep_for_variant(
                    variant_model_config,
                    gpu_count,
                    output_tag,
                    output_root,
                    ep_values,
                )
            for case_label, hw_path in CASE_CONFIGS:
                report_path = _report_path_for_hw_config(hw_path, output_root, output_tag)
                best_by_gpu = _load_best_no_oom(report_path, [gpu_count], metric, model_meta)
                best_entry = best_by_gpu.get(gpu_count)
                if best_entry is None:
                    continue
                prev = case_best[case_label].get(gpu_count)
                lower_is_better = _metric_is_lower_better(metric)
                if prev is None or (
                    best_entry["metric_value"] < prev["metric_value"]
                    if lower_is_better
                    else best_entry["metric_value"] > prev["metric_value"]
                ):
                    stored = dict(best_entry)
                    stored["gbs"] = gbs
                    case_best[case_label][gpu_count] = stored

    for case_label, _ in CASE_CONFIGS:
        for gpu_count in sorted(case_best[case_label]):
            item = case_best[case_label][gpu_count]
            summary_rows.append(
                {
                    "model_family": family["label"],
                    "gpu_count": gpu_count,
                    "case": case_label,
                    "gbs": item["gbs"],
                    "metric_value": item["metric_value"],
                    "runtime_s": item["runtime_s"],
                    "prefill_time_s": item["prefill_time_s"],
                    "decode_time_s": item["decode_time_s"],
                    "parallelism": item["parallelism"],
                    "report_path": item["report_path"],
                }
            )

    _write_family_summary(output_root, family["label"], summary_rows)
    return {
        "family_label": family["label"],
        "attention_label": family["attention_label"],
        "case_best": case_best,
    }


def _plot_family_speedups(ax, family_run, gpu_counts, case_palette, metric: str):
    case_best = family_run["case_best"]
    base_best = case_best["Base"]
    plotted_cases = [label for label, _ in CASE_CONFIGS if label != "Base"]
    group_width = 0.84
    bar_width = group_width / max(1, len(plotted_cases))
    lower_is_better = _metric_is_lower_better(metric)

    for case_index, case_label in enumerate(plotted_cases):
        xs = []
        ys = []
        for gpu_index, gpu_count in enumerate(gpu_counts):
            base_runtime = base_best.get(gpu_count, {}).get("metric_value")
            case_runtime = case_best.get(case_label, {}).get(gpu_count, {}).get("metric_value")
            if base_runtime is None or case_runtime is None or case_runtime <= 0:
                continue
            speedup = (base_runtime / case_runtime) if lower_is_better else (case_runtime / base_runtime)
            if not math.isfinite(speedup):
                continue
            x_pos = gpu_index - (group_width / 2) + bar_width * case_index + bar_width / 2
            xs.append(x_pos)
            ys.append(speedup)
        if xs:
            ax.bar(xs, ys, width=bar_width, color=case_palette[case_label], label=case_label)

    ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(gpu_counts)), [str(g) for g in gpu_counts])
    ax.set_xlabel("GPU count")
    ax.set_ylim(0.8, None)
    ax.set_title(family_run["attention_label"])


def _render_plot(output_root: Path, gpu_counts, metric: str, run_data):
    metric_title = _metric_title(metric)
    plotted_cases = [label for label, _ in CASE_CONFIGS if label != "Base"]
    palette = sns.color_palette("deep", n_colors=max(3, len(plotted_cases)))
    case_palette = {
        label: palette[index % len(palette)]
        for index, label in enumerate(plotted_cases)
    }

    fig, axes = plt.subplots(
        1,
        len(run_data),
        figsize=(9 * len(run_data), 4.5),
        sharey=True,
        squeeze=False,
    )
    flat_axes = axes[0]
    for ax, family_run in zip(flat_axes, run_data):
        _plot_family_speedups(ax, family_run, gpu_counts, case_palette, metric)
    flat_axes[0].set_ylabel(f"Speedup / Base {metric_title}")
    handles = [
        Patch(facecolor=case_palette[label], edgecolor="black", linewidth=0.4, label=label)
        for label in plotted_cases
    ]
    fig.suptitle(
        f"Inference DeepSeekV3 vs GLM4.7-358B: {metric_title} Speedup on H100 SXM5",
    )
    fig.legend(
        handles,
        [label for label in plotted_cases],
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        ncol=len(plotted_cases),
        frameon=True,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    output_path = _output_plot_path(output_root, gpu_counts, metric)
    fig.savefig(str(output_path), dpi=200)
    plt.close(fig)
    print(f"Saved plot to {output_path}")
    return output_path


def _render_family_plot(output_root: Path, gpu_counts, metric: str, family_run):
    metric_title = _metric_title(metric)
    plotted_cases = [label for label, _ in CASE_CONFIGS if label != "Base"]
    palette = sns.color_palette("deep", n_colors=max(3, len(plotted_cases)))
    case_palette = {
        label: palette[index % len(palette)]
        for index, label in enumerate(plotted_cases)
    }

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    _plot_family_speedups(ax, family_run, gpu_counts, case_palette, metric)
    ax.set_ylabel(f"Speedup / Base {metric_title}")
    handles = [
        Patch(facecolor=case_palette[label], edgecolor="black", linewidth=0.4, label=label)
        for label in plotted_cases
    ]
    fig.suptitle(f"{family_run['attention_label']}: {metric_title} Speedup on H100 SXM5")
    fig.legend(
        handles,
        [label for label in plotted_cases],
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        ncol=len(plotted_cases),
        frameon=True,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    output_path = _family_plot_path(output_root, family_run["family_label"], gpu_counts, metric)
    fig.savefig(str(output_path), dpi=200)
    plt.close(fig)
    print(f"Saved family plot to {output_path}")
    return output_path


def _cleanup_plot_artifacts(output_root: Path, keep_paths) -> None:
    keep_set = {str(Path(path).resolve()) for path in keep_paths}
    for png_path in output_root.rglob("*.png"):
        resolved = str(png_path.resolve())
        if resolved in keep_set:
            continue
        png_path.unlink()
    for path in sorted(output_root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir() and path.name != "model_variants":
            try:
                path.rmdir()
            except OSError:
                pass


def _default_output_root(gpu_counts, metric: str) -> Path:
    gpu_part = "-".join(str(count) for count in gpu_counts)
    metric_suffix = "" if metric == "runtime" else f"_{metric}"
    return OUTPUT_PARENT_DIR / f"inference_compare_deepseek_glm_{gpu_part}gpus{metric_suffix}"


def main():
    args = parse_args()
    gpu_counts = _parse_int_list(args.gpu_counts, GPU_COUNTS)
    gbs_values = _parse_int_list(args.gbs_values, GBS_VALUES)
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if str(args.output_root or "").strip()
        else _default_output_root(gpu_counts, args.metric)
    )
    output_root.mkdir(parents=True, exist_ok=True)

    completed_gpu_counts = []
    total_buckets = len(gpu_counts)
    for bucket_index, gpu_count in enumerate(gpu_counts, start=1):
        print(f"\n=== GPU bucket {bucket_index}/{total_buckets}: {gpu_count} GPUs ===")
        completed_gpu_counts.append(gpu_count)
        run_data = [
            _aggregate_family_results(
                family,
                completed_gpu_counts,
                [] if args.plot_only else [gpu_count],
                gbs_values,
                args.metric,
                output_root,
                args.plot_only,
                args.ep_values,
            )
            for family in MODEL_FAMILIES
        ]
        keep_paths = [_render_plot(output_root, completed_gpu_counts, args.metric, run_data)]
        for family_run in run_data:
            keep_paths.append(
                _render_family_plot(output_root, completed_gpu_counts, args.metric, family_run)
            )
        _cleanup_plot_artifacts(output_root, keep_paths)


if __name__ == "__main__":
    main()
