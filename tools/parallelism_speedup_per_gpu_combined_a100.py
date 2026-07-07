#!/usr/bin/env python3
"""
Generate the speedup-per-GPU combined plot for the A100 SXM4 case configs.

By default this script runs the parallelism sweep across
A100_SXM4_80GB_{base,case-*}.yaml with harness derates for A100_SXM4,
then renders the combined speedup plot from generated TSV reports.
"""

import argparse
import csv
import math
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
OUTPUT_PATH = TOOLS_DIR / "parallelism_speedup_per_gpu_combined.png"
OUTPUT_PARENT_DIR = TOOLS_DIR / "parallelism_results"
SWEEP_SCRIPT = TOOLS_DIR / "parallelism_sweep.py"
DERATE_CONFIG_PATH = (
    REPO_ROOT / "validation_scripts" / "validation_configs" / "harness_derates.yaml"
)
DERATE_DEVICE_TYPE = "A100_SXM4"

GPU_COUNTS = [128, 256, 512, 1024, 2048] #, 256, 512, 1024, 2048
DEFAULT_MODEL_CONFIG = (
    REPO_ROOT
    / "validation_scripts"
    / "validation_configs"
    / "model-config"
    / "Llama3.1-70B_2d_train.yaml"
)

CASE_CONFIGS = [
    (
        "Base",
        REPO_ROOT
        / "validation_scripts"
        / "validation_configs"
        / "hardware-config"
        / "A100_SXM4_80GB_base.yaml",
    ),
    (
        "Case A",
        REPO_ROOT
        / "validation_scripts"
        / "validation_configs"
        / "hardware-config"
        / "A100_SXM4_80GB_case-A.yaml",
    ),
    (
        "Case B",
        REPO_ROOT
        / "validation_scripts"
        / "validation_configs"
        / "hardware-config"
        / "A100_SXM4_80GB_case-B.yaml",
    ),
    (
        "Case C",
        REPO_ROOT
        / "validation_scripts"
        / "validation_configs"
        / "hardware-config"
        / "A100_SXM4_80GB_case-C.yaml",
    ),
    (
        "Case D",
        REPO_ROOT
        / "validation_scripts"
        / "validation_configs"
        / "hardware-config"
        / "A100_SXM4_80GB_case-D.yaml",
    ),
    (
        "Case E",
        REPO_ROOT
        / "validation_scripts"
        / "validation_configs"
        / "hardware-config"
        / "A100_SXM4_80GB_case-E.yaml",
    ),
    (
        "Case F",
        REPO_ROOT
        / "validation_scripts"
        / "validation_configs"
        / "hardware-config"
        / "A100_SXM4_80GB_case-F.yaml",
    ),
    (
        "Case G",
        REPO_ROOT
        / "validation_scripts"
        / "validation_configs"
        / "hardware-config"
        / "A100_SXM4_80GB_case-G.yaml",
    ),
]

OMITTED_PLOT_CASES = set()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the combined A100 SXM4 speedup plot from sweep reports."
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip sweep execution and only regenerate the figure from existing TSV reports.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=str(DEFAULT_MODEL_CONFIG),
        help="Path to model config YAML passed through to tools/parallelism_sweep.py.",
    )
    parser.add_argument(
        "--model-configs",
        type=str,
        default="",
        help=(
            "Comma-separated list of model config YAMLs. When provided, the script runs each "
            "config and renders them as subplots in one figure."
        ),
    )
    parser.add_argument(
        "--gpu-counts",
        type=str,
        default="",
        help="Comma-separated GPU counts to include in the plot and sweep bounds.",
    )
    parser.add_argument(
        "--shared-base-model-config",
        type=str,
        default="",
        help=(
            "Optional model config YAML whose Base-case runtimes are used as the denominator for "
            "all plotted models. When set with multiple model configs, the figure is rendered as "
            "a single combined bar chart."
        ),
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="runtime",
        choices=("runtime", "decode", "prefill"),
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


def _output_tag_for_run(model_config_path: Path, gpu_counts, metric: str = "runtime") -> str:
    gpu_part = "-".join(str(count) for count in gpu_counts)
    metric_suffix = "" if metric == "runtime" else f"_{metric}"
    return _sanitize_tag(f"{model_config_path.stem}_{gpu_part}gpus{metric_suffix}")


def _report_path_for_hw_config(hw_config_path: Path, output_root: Path, output_tag: str = "") -> Path:
    tag = hw_config_path.stem.replace(os.sep, "_")
    if output_tag:
        tag = f"{output_tag}_{tag}"
    return output_root / f"parallelism_sweep_{tag}.tsv"


def _output_plot_path(output_root: Path, output_tag: str = "") -> Path:
    if not output_tag:
        return output_root / OUTPUT_PATH.name
    return output_root / f"{OUTPUT_PATH.stem}_{output_tag}{OUTPUT_PATH.suffix}"


def _python_executable() -> str:
    preferred = [
        REPO_ROOT / "venv" / "bin" / "python",
        REPO_ROOT / ".venv" / "bin" / "python",
    ]
    for repo_venv_python in preferred:
        if repo_venv_python.exists():
            return str(repo_venv_python)
    return sys.executable


def _parse_gpu_counts(arg: str):
    if not arg:
        return list(GPU_COUNTS)
    gpu_counts = []
    for raw in arg.split(","):
        text = raw.strip()
        if not text:
            continue
        gpu_counts.append(int(text))
    if not gpu_counts:
        raise ValueError("--gpu-counts must include at least one integer.")
    return sorted(set(gpu_counts))


def _model_title_fragment(model_config_path: Path) -> str:
    stem = model_config_path.stem.replace("_", "-")
    if stem.endswith("-2d-train"):
        stem = stem[: -len("-2d-train")]
    elif stem.endswith("-train"):
        stem = stem[: -len("-train")]
    return stem


@lru_cache(maxsize=None)
def _load_model_metadata(model_config_path: Path) -> dict:
    try:
        with open(model_config_path, "r") as handle:
            raw = yaml.safe_load(handle) or {}
    except OSError:
        raw = {}
    model_param = raw.get("model_param", {}) if isinstance(raw, dict) else {}
    attention = model_param.get("attention", {}) if isinstance(model_param, dict) else {}
    return {
        "run_type": str(model_param.get("run_type", "")).strip().lower(),
        "model_type": str(model_param.get("model_type", "")).strip().lower(),
        "attention_type": str(attention.get("attention_type", "")).strip().upper(),
    }


def _run_type_title(run_type: str) -> str:
    return {"training": "Training", "inference": "Inference"}.get(run_type, run_type.title() or "Run")


def _model_type_title(model_type: str) -> str:
    return model_type.upper() if model_type else "Model"


def _figure_context_title(model_config_paths, shared_base_model_config: Path | None = None) -> str:
    reference_path = shared_base_model_config or model_config_paths[0]
    metadata = _load_model_metadata(reference_path)
    model_name = _model_title_fragment(reference_path)
    run_type = _run_type_title(metadata.get("run_type", ""))
    model_type = _model_type_title(metadata.get("model_type", ""))
    return f"{run_type} {model_name} ({model_type})"


def _parse_model_configs(args) -> list[Path]:
    if args.model_configs.strip():
        raw_paths = [item.strip() for item in args.model_configs.split(",") if item.strip()]
        if not raw_paths:
            raise ValueError("--model-configs must include at least one path.")
        return [Path(path).expanduser().resolve() for path in raw_paths]
    return [Path(args.model_config).expanduser().resolve()]


def _parse_shared_base_model_config(arg: str) -> Path | None:
    if not arg.strip():
        return None
    return Path(arg).expanduser().resolve()


def _metric_column(metric: str) -> str:
    return {
        "runtime": "runtime_s",
        "decode": "decode_time_s",
        "prefill": "prefill_time_s",
    }[metric]


def _metric_title(metric: str, run_type: str = "") -> str:
    return {
        "runtime": "Total Inference Time" if run_type == "inference" else "Runtime",
        "decode": "Decode Time",
        "prefill": "Prefill Time",
    }[metric]


def _attention_label(model_config_path: Path) -> str:
    try:
        with open(model_config_path, "r") as handle:
            for line in handle:
                if "attention_type" not in line:
                    continue
                _, _, value = line.partition(":")
                label = value.split("#", 1)[0].strip().strip("\"'").upper()
                if label:
                    return label
    except OSError:
        pass
    stem = model_config_path.stem.upper()
    if "MLA" in stem:
        return "MLA"
    if "GQA" in stem:
        return "GQA"
    return _model_title_fragment(model_config_path)


def run_sweep_for_cases(model_config_path: Path, gpu_counts, output_tag: str, output_root: Path) -> None:
    python_bin = _python_executable()
    hardware_configs = ",".join(str(path) for _, path in CASE_CONFIGS)
    labels = ",".join(label for label, _ in CASE_CONFIGS)
    cmd = [
        python_bin,
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
        str(min(gpu_counts)),
        "--gpu-count-max",
        str(max(gpu_counts)),
        "--derate-config",
        str(DERATE_CONFIG_PATH),
        "--derate-device-type",
        DERATE_DEVICE_TYPE,
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def load_best_no_oom(path, gpu_counts, metric_column: str):
    best = {}
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing sweep report: {path}")
    with open(path, "r") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            try:
                num_gpus = int(row["num_gpus"])
            except (KeyError, ValueError):
                continue
            if num_gpus not in gpu_counts:
                continue
            memory_exceeded = str(row.get("memory_exceeded", "")).strip().lower() == "true"
            if memory_exceeded:
                continue
            try:
                runtime = float(row[metric_column])
            except (KeyError, ValueError):
                continue
            prev = best.get(num_gpus)
            if prev is None or runtime < prev:
                best[num_gpus] = runtime
    return best


def _load_case_best_for_model(
    model_config_path: Path,
    gpu_counts,
    plot_only: bool,
    metric: str,
    output_root: Path,
):
    output_tag = _output_tag_for_run(model_config_path, gpu_counts, metric)
    if not plot_only:
        run_sweep_for_cases(model_config_path, gpu_counts, output_tag, output_root)

    cases = [
        (label, _report_path_for_hw_config(hw_path, output_root, output_tag))
        for label, hw_path in CASE_CONFIGS
    ]
    case_best = {}
    for label, path in cases:
        case_best[label] = load_best_no_oom(path, gpu_counts, _metric_column(metric))
    return {
        "model_config_path": model_config_path,
        "output_tag": output_tag,
        "cases": cases,
        "case_best": case_best,
        "attention_label": _attention_label(model_config_path),
    }


def _plot_model_speedups(ax, run_data, gpu_counts, palette):
    case_best = run_data["case_best"]
    base_label = CASE_CONFIGS[0][0]
    base_best = case_best.get(base_label, {})
    plotted_cases = [
        (label, path)
        for (label, path) in run_data["cases"]
        if label != base_label and label not in OMITTED_PLOT_CASES
    ]

    group_width = 0.85
    bar_width = group_width / max(1, len(plotted_cases))

    for li, (label, _) in enumerate(plotted_cases):
        xs = []
        ys = []
        for idx, gpu_count in enumerate(gpu_counts):
            base_runtime = base_best.get(gpu_count)
            runtime = case_best.get(label, {}).get(gpu_count)
            if base_runtime is None or runtime is None or runtime <= 0:
                continue
            speedup = base_runtime / runtime
            if not math.isfinite(speedup):
                continue
            x_pos = idx - (group_width / 2) + bar_width * li + bar_width / 2
            xs.append(x_pos)
            ys.append(speedup)
        if xs:
            ax.bar(xs, ys, width=bar_width, label=label, color=palette[li % len(palette)])

    ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(gpu_counts)), [str(g) for g in gpu_counts])
    ax.set_xlabel("GPU count")
    ax.set_ylim(0.8, None)
    ax.set_title(run_data["attention_label"])


def _plot_shared_base_speedups(ax, run_data, gpu_counts, shared_base_best, case_palette, metric_title: str):
    plotted_cases = [label for label, _ in CASE_CONFIGS if label != CASE_CONFIGS[0][0]]
    hatches = ["", "//", "xx", "\\\\"]
    total_series = len(run_data) * len(plotted_cases)
    group_width = 0.9
    bar_width = group_width / max(1, total_series)

    for model_idx, item in enumerate(run_data):
        hatch = hatches[model_idx % len(hatches)]
        for case_idx, case_label in enumerate(plotted_cases):
            xs = []
            ys = []
            series_idx = model_idx * len(plotted_cases) + case_idx
            for gpu_idx, gpu_count in enumerate(gpu_counts):
                base_runtime = shared_base_best.get(gpu_count)
                runtime = item["case_best"].get(case_label, {}).get(gpu_count)
                if base_runtime is None or runtime is None or runtime <= 0:
                    continue
                speedup = base_runtime / runtime
                if not math.isfinite(speedup):
                    continue
                x_pos = gpu_idx - (group_width / 2) + bar_width * series_idx + bar_width / 2
                xs.append(x_pos)
                ys.append(speedup)
            if xs:
                ax.bar(
                    xs,
                    ys,
                    width=bar_width,
                    color=case_palette[case_label],
                    edgecolor="black",
                    linewidth=0.4,
                    hatch=hatch,
                )

    ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(gpu_counts)), [str(g) for g in gpu_counts])
    ax.set_xlabel("GPU count")
    ax.set_ylabel(f"Speedup / GQA Base {metric_title}")
    ax.set_ylim(0.8, None)
    ax.set_title(f"GQA and MLA Speedup vs GQA Base {metric_title}")

    case_handles = [
        Patch(facecolor=case_palette[label], edgecolor="black", linewidth=0.4, label=label)
        for label in plotted_cases
    ]
    attention_handles = [
        Patch(facecolor="white", edgecolor="black", linewidth=0.6, hatch=hatches[idx % len(hatches)], label=item["attention_label"])
        for idx, item in enumerate(run_data)
    ]
    case_legend = ax.legend(
        handles=case_handles,
        loc="upper left",
        bbox_to_anchor=(-0.02, 1.26),
        ncol=4,
        frameon=True,
        fontsize=9,
        title="Hardware case",
    )
    ax.add_artist(case_legend)
    ax.legend(
        handles=attention_handles,
        loc="upper right",
        bbox_to_anchor=(1.02, 1.26),
        ncol=max(1, len(attention_handles)),
        frameon=True,
        fontsize=9,
        title="Attention",
    )


def _comparison_output_path(
    output_root: Path,
    model_config_paths,
    gpu_counts,
    shared_base_model_config: Path | None = None,
    metric: str = "runtime",
) -> Path:
    if len(model_config_paths) == 1:
        return _output_plot_path(output_root, _output_tag_for_run(model_config_paths[0], gpu_counts, metric))
    model_part = "__".join(_sanitize_tag(path.stem) for path in model_config_paths)
    gpu_part = "-".join(str(count) for count in gpu_counts)
    shared_suffix = ""
    if shared_base_model_config is not None:
        shared_suffix = f"_sharedbase_{_sanitize_tag(shared_base_model_config.stem)}"
    metric_suffix = "" if metric == "runtime" else f"_{metric}"
    return output_root / (
        f"{OUTPUT_PATH.stem}_{model_part}_{gpu_part}gpus{shared_suffix}{metric_suffix}{OUTPUT_PATH.suffix}"
    )


def _default_output_root(
    model_config_paths,
    gpu_counts,
    shared_base_model_config: Path | None,
    metric: str,
) -> Path:
    comparison_name = _comparison_output_path(
        Path("."),
        model_config_paths,
        gpu_counts,
        shared_base_model_config,
        metric,
    ).stem
    return OUTPUT_PARENT_DIR / comparison_name


def main():
    args = parse_args()
    gpu_counts = _parse_gpu_counts(args.gpu_counts)
    model_config_paths = _parse_model_configs(args)
    shared_base_model_config = _parse_shared_base_model_config(args.shared_base_model_config)
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if str(args.output_root or "").strip()
        else _default_output_root(model_config_paths, gpu_counts, shared_base_model_config, args.metric)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = _comparison_output_path(
        output_root,
        model_config_paths,
        gpu_counts,
        shared_base_model_config,
        args.metric,
    )
    reference_metadata = _load_model_metadata(shared_base_model_config or model_config_paths[0])
    metric_title = _metric_title(args.metric, reference_metadata.get("run_type", ""))
    context_title = _figure_context_title(model_config_paths, shared_base_model_config)
    run_data = [
        _load_case_best_for_model(
            model_config_path,
            gpu_counts,
            args.plot_only,
            args.metric,
            output_root,
        )
        for model_config_path in model_config_paths
    ]

    palette = sns.color_palette(
        "deep",
        n_colors=max(
            3,
            len([label for label, _ in CASE_CONFIGS if label != CASE_CONFIGS[0][0]]),
        ),
    )
    if shared_base_model_config is not None and len(run_data) > 1:
        shared_item = next(
            (item for item in run_data if item["model_config_path"] == shared_base_model_config),
            None,
        )
        if shared_item is None:
            raise ValueError(
                "--shared-base-model-config must match one of the provided --model-configs entries."
            )
        shared_base_best = shared_item["case_best"].get(CASE_CONFIGS[0][0], {})
        case_palette = {
            label: palette[idx % len(palette)]
            for idx, (label, _) in enumerate(CASE_CONFIGS[1:])
        }
        fig, ax = plt.subplots(1, 1, figsize=(13, 5))
        _plot_shared_base_speedups(
            ax,
            run_data,
            gpu_counts,
            shared_base_best,
            case_palette,
            metric_title,
        )
        fig.suptitle(
            f"{context_title}: {metric_title} Speedup on A100 SXM4 Using GQA Base"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.9))
    else:
        fig, axes = plt.subplots(
            1,
            len(run_data),
            figsize=(9 * len(run_data), 4),
            sharey=True,
            squeeze=False,
        )
        flat_axes = axes[0]
        for ax, item in zip(flat_axes, run_data):
            _plot_model_speedups(ax, item, gpu_counts, palette)
        flat_axes[0].set_ylabel(f"Speedup / Base {metric_title}")

        handles, labels = flat_axes[0].get_legend_handles_labels()
        fig.suptitle(
            f"{context_title}: {metric_title} Speedup on A100 SXM4 by Attention Type"
        )
        fig.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            ncol=2,
            frameon=True,
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(str(output_path), dpi=200)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
