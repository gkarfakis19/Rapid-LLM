#!/usr/bin/env python3
"""
Generate the speedup-per-GPU combined plot for H100 SXM5 synthetic case configs.

By default this script runs the parallelism sweep across
H100_SXM5_80GB_base.yaml plus generated Case A-G variants,
then renders the combined speedup plot from generated TSV reports.
"""

import argparse
import ast
import csv
import math
import os
import re
import subprocess
import sys
import time
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import yaml

import parallelism_sweep as sweep

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
OUTPUT_PATH = TOOLS_DIR / "parallelism_speedup_per_gpu_combined_h100.png"
OUTPUT_PARENT_DIR = TOOLS_DIR / "parallelism_results"
SWEEP_SCRIPT = TOOLS_DIR / "parallelism_sweep.py"
H100_BASE_CONFIG = (
    REPO_ROOT
    / "validation_scripts"
    / "validation_configs"
    / "hardware-config"
    / "H100_SXM5_80GB_base.yaml"
)
GENERATED_CASE_DIRNAME = "generated_hw_configs_h100"

GPU_COUNTS = [128, 256, 512, 1024, 2048] #, 256, 512, 1024, 2048
INFERENCE_GPU_COUNTS = [4, 8, 16, 32, 64]
INFERENCE_GPU_COUNT_MIN = 4
INFERENCE_GPU_COUNT_MAX = 64
INFERENCE_GLOBAL_BATCH_SIZES = [1, 2, 4]
DEFAULT_MODEL_CONFIG = (
    REPO_ROOT
    / "validation_scripts"
    / "validation_configs"
    / "model-config"
    / "Llama3.1-70B_2d_train.yaml"
)

CASE_LABELS = [
    "Base",
    "Case A",
    "Case B",
    "Case C",
    "Case D",
    "Case E",
]
BASE_CASE_LABEL = CASE_LABELS[0]
CASE_FILENAME_BY_LABEL = {
    "Base": "base.yaml",
    "Case A": "case-A.yaml",
    "Case B": "case-B.yaml",
    "Case C": "case-C.yaml",
    "Case D": "case-D.yaml",
    "Case E": "case-E.yaml",
}

OMITTED_PLOT_CASES = set()

PLOT_FONT_SCALE = 2.0
PLOT_BASE_FONT_SIZE = 10
PLOT_FONT_SIZE = int(PLOT_BASE_FONT_SIZE * PLOT_FONT_SCALE)
PLOT_TITLE_FONT_SIZE = int(16 * PLOT_FONT_SCALE)
PLOT_YLABEL_FONT_SIZE = int(PLOT_FONT_SIZE * 1.35)
PLOT_LEGEND_FONT_SIZE = int(PLOT_YLABEL_FONT_SIZE * 0.9)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the combined H100 SXM5 speedup plot from sweep reports."
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
        choices=("runtime", "decode", "prefill", "decode_throughput"),
        help="Metric to compare in the speedup plot.",
    )
    parser.add_argument(
        "--inference-decode-throughput",
        action="store_true",
        help=(
            "When enabled, inference model configs compare speedup using decode throughput "
            "(aggregate tok/s = decode_len * global_batch_size * replica_count / decode_time_s) "
            "regardless of --metric."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Optional directory where sweep TSVs/PNGs and the combined plot will be stored.",
    )
    parser.add_argument(
        "--inference-gbs",
        type=str,
        default="1,2,4",
        help="Comma-separated inference global batch sizes to sweep (used when run_type is inference).",
    )
    parser.add_argument(
        "--case-d-use-case-c-best",
        action="store_true",
        help=(
            "Do not sweep Case D. Instead, for each GPU count, evaluate Case D only at the "
            "best parallelism selected from Case C for the active metric."
        ),
    )
    return parser.parse_args()


def _apply_plot_font_style() -> None:
    plt.rcParams.update(
        {
            "font.size": PLOT_FONT_SIZE,
            "axes.titlesize": PLOT_FONT_SIZE,
            "axes.labelsize": PLOT_FONT_SIZE,
            "xtick.labelsize": PLOT_FONT_SIZE,
            "ytick.labelsize": PLOT_FONT_SIZE,
            "legend.fontsize": PLOT_LEGEND_FONT_SIZE,
            "legend.title_fontsize": PLOT_LEGEND_FONT_SIZE,
            "figure.titlesize": PLOT_TITLE_FONT_SIZE,
        }
    )


def _sanitize_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))


def _short_model_tag(model_config_path: Path) -> str:
    stem = model_config_path.stem
    run_type = _load_model_metadata(model_config_path).get("run_type", "")
    run_tag = "inf" if run_type == "inference" else "train" if run_type == "training" else "run"
    size_match = re.search(r"(\d+B)", stem, flags=re.IGNORECASE)
    if size_match:
        size = size_match.group(1).upper()
        suffixes = []
        lower_stem = stem.lower()
        if "smallctx" in lower_stem:
            suffixes.append("smallctx")
        if "halfseqlen" in lower_stem:
            suffixes.append("halfseqlen")
        suffix = f"_{'_'.join(suffixes)}" if suffixes else ""
        return _sanitize_tag(f"{size}_{run_tag}{suffix}")
    return _sanitize_tag(stem)


def _metric_short(metric: str) -> str:
    return {
        "runtime": "runtime",
        "decode": "decode",
        "prefill": "prefill",
        "decode_throughput": "decode_tp",
    }[metric]


def _output_tag_for_run(model_config_path: Path, gpu_counts, metric: str = "runtime") -> str:
    gpu_part = "-".join(str(count) for count in gpu_counts)
    return _sanitize_tag(f"{_short_model_tag(model_config_path)}_{_metric_short(metric)}_{gpu_part}gpus")


def _report_path_for_hw_config(
    hw_config_path: Path, output_root: Path, output_tag: str = "", case_label: str = ""
) -> Path:
    tag = hw_config_path.stem.replace(os.sep, "_")
    if case_label.strip():
        tag = _sanitize_tag(case_label.replace("Case ", "case-").replace("Base", "base"))
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


def _parse_int_csv(arg: str, flag_name: str) -> list[int]:
    values = []
    for raw in str(arg or "").split(","):
        text = raw.strip()
        if not text:
            continue
        values.append(int(text))
    if not values:
        raise ValueError(f"{flag_name} must include at least one integer.")
    return sorted(set(values))


def _model_title_fragment(model_config_path: Path) -> str:
    stem = model_config_path.stem.replace("_", "-")
    if stem.endswith("-2d-train"):
        stem = stem[: -len("-2d-train")]
    elif stem.endswith("-2d-inf-halfseqlen"):
        stem = stem[: -len("-2d-inf-halfseqlen")]
    elif stem.endswith("-2d-inf-seqlen2x"):
        stem = stem[: -len("-2d-inf-seqlen2x")]
    elif stem.endswith("-2d-inf-smallctx"):
        stem = stem[: -len("-2d-inf-smallctx")]
    elif stem.endswith("-2d-inf"):
        stem = stem[: -len("-2d-inf")]
    elif stem.endswith("-inf-halfseqlen"):
        stem = stem[: -len("-inf-halfseqlen")]
    elif stem.endswith("-inf-seqlen2x"):
        stem = stem[: -len("-inf-seqlen2x")]
    elif stem.endswith("-inf-smallctx"):
        stem = stem[: -len("-inf-smallctx")]
    elif stem.endswith("-inf"):
        stem = stem[: -len("-inf")]
    elif stem.endswith("-train"):
        stem = stem[: -len("-train")]
    stem = re.sub(r"^Llama3\.1-(\d+B)$", r"Llama3-\1", stem, flags=re.IGNORECASE)
    return stem


def _is_inference_model(model_config_path: Path) -> bool:
    return _load_model_metadata(model_config_path).get("run_type", "") == "inference"


@lru_cache(maxsize=None)
def _load_model_metadata(model_config_path: Path) -> dict:
    try:
        with open(model_config_path, "r") as handle:
            raw = yaml.safe_load(handle) or {}
    except OSError:
        raw = {}
    model_param = raw.get("model_param", {}) if isinstance(raw, dict) else {}
    attention = model_param.get("attention", {}) if isinstance(model_param, dict) else {}
    decode_len = model_param.get("decode_len", None) if isinstance(model_param, dict) else None
    gbs = model_param.get("global_batch_size", None) if isinstance(model_param, dict) else None
    try:
        decode_len = int(decode_len) if decode_len is not None else None
    except (TypeError, ValueError):
        decode_len = None
    try:
        gbs = int(gbs) if gbs is not None else None
    except (TypeError, ValueError):
        gbs = None
    return {
        "run_type": str(model_param.get("run_type", "")).strip().lower(),
        "model_type": str(model_param.get("model_type", "")).strip().lower(),
        "attention_type": str(attention.get("attention_type", "")).strip().upper(),
        "decode_len": decode_len,
        "global_batch_size": gbs,
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


def _speedup_figure_title(reference_model_path: Path, metric_title: str) -> str:
    metadata = _load_model_metadata(reference_model_path)
    run_type = metadata.get("run_type", "")
    model_name = _model_title_fragment(reference_model_path)
    if run_type == "inference":
        workload = "Inference"
    elif run_type == "training":
        workload = "Training"
    else:
        workload = "Run"
    return f"{model_name} {workload} Time Speedup on H100 SXM5 Varying GPU Count"


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
        "decode_throughput": "decode_time_s",
    }[metric]


def _metric_title(metric: str, run_type: str = "") -> str:
    return {
        "runtime": "Total Inference Time" if run_type == "inference" else "Runtime",
        "decode": "Decode Time",
        "prefill": "Prefill Time",
        "decode_throughput": "Aggregate Decode Throughput (tok/s)",
    }[metric]


def _metric_higher_is_better(metric: str) -> bool:
    return metric == "decode_throughput"


def _effective_metric_for_model(
    model_config_path: Path, requested_metric: str, inference_decode_throughput: bool
) -> str:
    if inference_decode_throughput and _load_model_metadata(model_config_path).get("run_type") == "inference":
        return "decode_throughput"
    return requested_metric


def _inference_replica_count_from_row(row: dict) -> int:
    raw_parallelism = row.get("parallelism")
    if raw_parallelism is None:
        return 1
    try:
        parallelism = ast.literal_eval(str(raw_parallelism))
    except (SyntaxError, ValueError):
        return 1
    if not isinstance(parallelism, dict):
        return 1
    inference_block = parallelism.get("inference", {})
    if not isinstance(inference_block, dict):
        return 1
    try:
        replica_count = int(inference_block.get("replica_count", 1))
    except (TypeError, ValueError):
        return 1
    return max(1, replica_count)


def _metric_value_from_row(row: dict, metric: str, model_metadata: dict | None = None) -> float:
    metric_column = _metric_column(metric)
    raw_value = row.get(metric_column)
    if raw_value is None:
        raise KeyError(metric_column)
    value = float(raw_value)
    if metric == "decode_throughput":
        if value <= 0:
            return float("nan")
        if isinstance(model_metadata, dict) and model_metadata.get("run_type") == "inference":
            decode_len = int(model_metadata.get("decode_len") or 0)
            global_batch_size = int(model_metadata.get("global_batch_size") or 1)
            replica_count = _inference_replica_count_from_row(row)
            if decode_len > 0 and global_batch_size > 0:
                return (float(decode_len) * float(global_batch_size) * float(replica_count)) / value
        return 1.0 / value
    return value


def _compute_speedup(base_value: float, case_value: float, higher_is_better: bool) -> float:
    if higher_is_better:
        return case_value / base_value
    return base_value / case_value


def _merge_best_metric_dicts(metric_dicts, metric: str):
    merged = {}
    higher_is_better = _metric_higher_is_better(metric)
    for values in metric_dicts:
        for gpu_count, value in values.items():
            prev = merged.get(gpu_count)
            if prev is None:
                merged[gpu_count] = value
            elif higher_is_better and value > prev:
                merged[gpu_count] = value
            elif (not higher_is_better) and value < prev:
                merged[gpu_count] = value
    return merged


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


def _scale_quantity(value, factor: float):
    if isinstance(value, (int, float)):
        return float(value) * factor
    text = str(value).strip()
    tokens = text.split()
    if not tokens:
        raise ValueError(f"Cannot scale empty quantity value: {value!r}")
    base = float(tokens[0])
    unit = " ".join(tokens[1:])
    scaled = base * factor
    if not unit:
        return float(scaled)
    if abs(scaled - round(scaled)) < 1e-12:
        number_text = str(int(round(scaled)))
    else:
        number_text = f"{scaled:.12g}"
    return f"{number_text} {unit}".rstrip()


def _scale_dimension_bandwidth(hw_dict: dict, dim_index: int, factor: float) -> None:
    dimensions = hw_dict.get("network", {}).get("dimensions", [])
    if dim_index >= len(dimensions):
        raise ValueError(f"Missing network.dimensions[{dim_index}] in generated case config")
    topology = dimensions[dim_index].get("topology", {})
    bandwidth = topology.get("bandwidth")
    if bandwidth is None:
        raise ValueError(f"Missing topology.bandwidth for network.dimensions[{dim_index}]")
    if isinstance(bandwidth, list):
        topology["bandwidth"] = [_scale_quantity(item, factor) for item in bandwidth]
    else:
        topology["bandwidth"] = _scale_quantity(bandwidth, factor)


def _load_base_hardware_config() -> dict:
    with open(H100_BASE_CONFIG, "r") as handle:
        return yaml.safe_load(handle) or {}


def _force_zero_network_overlap(hw_dict: dict) -> None:
    network = hw_dict.setdefault("network", {})
    if not isinstance(network, dict):
        hw_dict["network"] = {"overlap": {"tp_overlap": 0, "tp_sp_overlap": 0, "cp_overlap": 0}}
        return
    overlap = network.setdefault("overlap", {})
    if not isinstance(overlap, dict):
        network["overlap"] = {}
        overlap = network["overlap"]
    overlap["tp_overlap"] = 0
    overlap["tp_sp_overlap"] = 0
    overlap["cp_overlap"] = 0


def _build_generated_case_dicts(base_hw: dict) -> dict[str, dict]:
    cases = {}
    cases["Base"] = deepcopy(base_hw)
    _force_zero_network_overlap(cases["Base"])

    case_a = deepcopy(base_hw)
    _force_zero_network_overlap(case_a)
    case_a["tech_param"]["core"]["util"] = float(case_a["tech_param"]["core"]["util"]) * 1.2
    cases["Case A"] = case_a

    case_b = deepcopy(base_hw)
    _force_zero_network_overlap(case_b)
    case_b["tech_param"]["DRAM"]["size"] = _scale_quantity(case_b["tech_param"]["DRAM"]["size"], 2.0)
    cases["Case B"] = case_b

    case_c = deepcopy(base_hw)
    _force_zero_network_overlap(case_c)
    case_c["tech_param"]["DRAM"]["size"] = _scale_quantity(case_c["tech_param"]["DRAM"]["size"], 2.0)
    case_c["tech_param"]["DRAM"]["bandwidth"] = _scale_quantity(
        case_c["tech_param"]["DRAM"]["bandwidth"], 5.33
    )
    cases["Case C"] = case_c

    case_d = deepcopy(case_c)
    case_d["tech_param"]["DRAM"]["bandwidth"] = _scale_quantity(
        case_d["tech_param"]["DRAM"]["bandwidth"], 0.73
    )
    cases["Case D"] = case_d

    case_e = deepcopy(case_c)
    _scale_dimension_bandwidth(case_e, dim_index=0, factor=2.0)
    _scale_dimension_bandwidth(case_e, dim_index=1, factor=2.0)
    cases["Case E"] = case_e

    return cases


def _materialize_case_configs(output_root: Path) -> list[tuple[str, Path]]:
    output_dir = output_root / GENERATED_CASE_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    base_hw = _load_base_hardware_config()
    case_dicts = _build_generated_case_dicts(base_hw)

    case_configs = []
    for label in CASE_LABELS:
        path = output_dir / CASE_FILENAME_BY_LABEL[label]
        with open(path, "w") as handle:
            yaml.safe_dump(case_dicts[label], handle, sort_keys=False)
        case_configs.append((label, path))
    return case_configs


def _materialize_model_configs_for_gbs(
    model_config_path: Path, output_root: Path, gbs_values
) -> list[tuple[int, Path]]:
    output_dir = output_root / "generated_model_configs_h100"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(model_config_path, "r") as handle:
        base_model = yaml.safe_load(handle) or {}
    if not isinstance(base_model, dict):
        raise ValueError(f"Model config must be a YAML mapping: {model_config_path}")
    model_param = base_model.get("model_param")
    if not isinstance(model_param, dict):
        raise ValueError(f"model_param section missing in model config: {model_config_path}")

    variants = []
    for gbs in gbs_values:
        variant = deepcopy(base_model)
        variant_model_param = variant.setdefault("model_param", {})
        variant_model_param["global_batch_size"] = int(gbs)
        out_path = output_dir / f"{model_config_path.stem}_gbs{int(gbs)}.yaml"
        with open(out_path, "w") as handle:
            yaml.safe_dump(variant, handle, sort_keys=False)
        variants.append((int(gbs), out_path))
    return variants


def _parse_parallelism_text(raw_parallelism: str) -> dict | None:
    try:
        parsed = ast.literal_eval(str(raw_parallelism))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _best_parallelism_by_gpu_from_report(
    report_path: Path,
    gpu_counts,
    metric: str,
    model_metadata: dict | None = None,
) -> dict[int, dict]:
    best_by_gpu: dict[int, tuple[float, dict]] = {}
    higher_is_better = _metric_higher_is_better(metric)
    if not report_path.exists():
        return {}
    with open(report_path, "r") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            try:
                num_gpus = int(row.get("num_gpus", -1))
            except (TypeError, ValueError):
                continue
            if num_gpus not in gpu_counts:
                continue
            if str(row.get("memory_exceeded", "")).strip().lower() == "true":
                continue
            parallelism = _parse_parallelism_text(row.get("parallelism", ""))
            if not isinstance(parallelism, dict):
                continue
            try:
                metric_value = _metric_value_from_row(row, metric, model_metadata=model_metadata)
            except (KeyError, ValueError):
                continue
            if not math.isfinite(metric_value):
                continue
            prev = best_by_gpu.get(num_gpus)
            if prev is None:
                best_by_gpu[num_gpus] = (metric_value, parallelism)
                continue
            prev_metric_value = prev[0]
            if higher_is_better and metric_value > prev_metric_value:
                best_by_gpu[num_gpus] = (metric_value, parallelism)
            elif (not higher_is_better) and metric_value < prev_metric_value:
                best_by_gpu[num_gpus] = (metric_value, parallelism)
    return {gpu: value[1] for gpu, value in best_by_gpu.items()}


def _materialize_case_d_report_from_case_c_best(
    case_c_report_path: Path,
    case_d_hw_path: Path,
    model_config_path: Path,
    case_d_report_path: Path,
    gpu_counts,
    metric: str,
) -> None:
    model_metadata = _load_model_metadata(model_config_path)
    selected = _best_parallelism_by_gpu_from_report(
        case_c_report_path,
        gpu_counts,
        metric,
        model_metadata=model_metadata,
    )
    mode = sweep.determine_model_mode(str(model_config_path))
    model_config_obj = sweep.config.parse_config(str(model_config_path), config_type=mode)
    hw_dict = sweep.read_yaml(str(case_d_hw_path))
    results = []
    for gpu_count in sorted(selected.keys()):
        parallelism = selected[gpu_count]
        metrics = sweep.evaluate_parallelism(hw_dict, model_config_obj, mode, parallelism)
        results.append(
            {
                "parallelism": parallelism,
                "num_gpus": int(metrics.get("num_gpus", gpu_count)),
                "runtime": metrics["runtime"],
                "prefill_time": metrics.get("prefill_time", float("nan")),
                "decode_time": metrics.get("decode_time", float("nan")),
                "performance": metrics["performance"],
                "total_flops": metrics["total_flops"],
                "achieved_flops": metrics["achieved_flops"],
                "peak_flops": metrics["peak_flops"],
                "mfu": metrics["mfu"],
                "memory_exceeded": metrics["memory_exceeded"],
                "memory_violation_gb": metrics["memory_violation_gb"],
            }
        )
    sweep.write_report(results, str(case_d_report_path))
    print(
        f"[progress] case-d-from-case-c wrote {len(results)} row(s) to {case_d_report_path}",
        flush=True,
    )


def run_sweep_for_cases(
    model_config_path: Path,
    gpu_counts,
    output_tag: str,
    output_root: Path,
    case_configs: list[tuple[str, Path]],
    inference_gbs_values: list[int],
    effective_metric: str,
    case_d_use_case_c_best: bool = False,
) -> list[str]:
    is_inference = _is_inference_model(model_config_path)
    if is_inference:
        gbs_model_configs = _materialize_model_configs_for_gbs(
            model_config_path, output_root, inference_gbs_values
        )
        gpu_count_min = int(min(gpu_counts))
        gpu_count_max = int(max(gpu_counts))
        run_specs = [
            (f"{output_tag}_gbs{gbs}", cfg_path, gpu_count_min, gpu_count_max)
            for gbs, cfg_path in gbs_model_configs
        ]
    else:
        run_specs = [(output_tag, model_config_path, int(min(gpu_counts)), int(max(gpu_counts)))]

    python_bin = _python_executable()
    case_configs_to_sweep = list(case_configs)
    if case_d_use_case_c_best:
        case_configs_to_sweep = [item for item in case_configs if item[0] != "Case D"]
    hardware_configs = ",".join(str(path) for _, path in case_configs_to_sweep)
    labels = ",".join(label for label, _ in case_configs_to_sweep)
    report_tags = []
    total_runs = len(run_specs)
    print(
        f"[progress] sweep runs={total_runs} model={model_config_path.name} "
        f"cases={len(case_configs)}",
        flush=True,
    )
    for run_idx, (run_output_tag, run_model_config, gpu_count_min, gpu_count_max) in enumerate(run_specs, start=1):
        run_start = time.time()
        print(
            f"[progress] run {run_idx}/{total_runs} start tag={run_output_tag} "
            f"model={Path(run_model_config).name} gpu_range={gpu_count_min}-{gpu_count_max}",
            flush=True,
        )
        cmd = [
            python_bin,
            str(SWEEP_SCRIPT),
            "--hardware-configs",
            hardware_configs,
            "--hardware-labels",
            labels,
            "--model-config",
            str(run_model_config),
            "--output-tag",
            run_output_tag,
            "--output-root",
            str(output_root),
            "--gpu-count-min",
            str(gpu_count_min),
            "--gpu-count-max",
            str(gpu_count_max),
        ]
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
        if case_d_use_case_c_best:
            case_lookup = {label: path for label, path in case_configs}
            case_c_hw = case_lookup.get("Case C")
            case_d_hw = case_lookup.get("Case D")
            if case_c_hw is not None and case_d_hw is not None:
                case_c_report = _report_path_for_hw_config(
                    case_c_hw, output_root, run_output_tag, case_label="Case C"
                )
                case_d_report = _report_path_for_hw_config(
                    case_d_hw, output_root, run_output_tag, case_label="Case D"
                )
                _materialize_case_d_report_from_case_c_best(
                    case_c_report,
                    case_d_hw,
                    Path(run_model_config),
                    case_d_report,
                    gpu_counts,
                    effective_metric,
                )
        elapsed = time.time() - run_start
        print(
            f"[progress] run {run_idx}/{total_runs} done tag={run_output_tag} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        report_tags.append(run_output_tag)
    return report_tags


def load_best_no_oom(path, gpu_counts, metric: str, model_metadata: dict | None = None):
    best = {}
    higher_is_better = _metric_higher_is_better(metric)
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
                metric_value = _metric_value_from_row(row, metric, model_metadata=model_metadata)
            except (KeyError, ValueError):
                continue
            if not math.isfinite(metric_value):
                continue
            prev = best.get(num_gpus)
            if prev is None:
                best[num_gpus] = metric_value
            elif higher_is_better and metric_value > prev:
                best[num_gpus] = metric_value
            elif (not higher_is_better) and metric_value < prev:
                best[num_gpus] = metric_value
    return best


def _load_case_best_for_model(
    model_config_path: Path,
    gpu_counts,
    plot_only: bool,
    requested_metric: str,
    inference_decode_throughput: bool,
    output_root: Path,
    inference_gbs_values: list[int],
    case_d_use_case_c_best: bool = False,
):
    effective_metric = _effective_metric_for_model(
        model_config_path, requested_metric, inference_decode_throughput
    )
    output_tag = _output_tag_for_run(model_config_path, gpu_counts, effective_metric)
    case_configs = _materialize_case_configs(output_root)
    is_inference = _is_inference_model(model_config_path)
    report_specs: list[tuple[str, Path]]
    if is_inference:
        gbs_model_configs = _materialize_model_configs_for_gbs(
            model_config_path, output_root, inference_gbs_values
        )
    else:
        gbs_model_configs = []
    if not plot_only:
        report_tags = run_sweep_for_cases(
            model_config_path,
            gpu_counts,
            output_tag,
            output_root,
            case_configs,
            inference_gbs_values,
            effective_metric,
            case_d_use_case_c_best=case_d_use_case_c_best,
        )
        if is_inference:
            report_specs = [
                (f"{output_tag}_gbs{gbs}", cfg_path)
                for (gbs, cfg_path) in gbs_model_configs
            ]
        else:
            report_specs = [(report_tags[0], model_config_path)]
    elif is_inference:
        report_specs = [
            (f"{output_tag}_gbs{gbs}", cfg_path)
            for (gbs, cfg_path) in gbs_model_configs
        ]
    else:
        report_specs = [(output_tag, model_config_path)]

    cases = [
        (label, _report_path_for_hw_config(hw_path, output_root, output_tag, case_label=label))
        for label, hw_path in case_configs
    ]
    case_best = {}
    for label, hw_path in case_configs:
        metric_dicts = []
        for tag, run_model_config in report_specs:
            report_path = _report_path_for_hw_config(hw_path, output_root, tag, case_label=label)
            run_metadata = _load_model_metadata(run_model_config)
            metric_dicts.append(
                load_best_no_oom(
                    report_path,
                    gpu_counts,
                    effective_metric,
                    model_metadata=run_metadata,
                )
            )
        case_best[label] = _merge_best_metric_dicts(metric_dicts, effective_metric)
    return {
        "model_config_path": model_config_path,
        "output_tag": output_tag,
        "cases": cases,
        "case_best": case_best,
        "metric": effective_metric,
        "higher_is_better": _metric_higher_is_better(effective_metric),
        "attention_label": _attention_label(model_config_path),
    }


def _plot_model_speedups(ax, run_data, gpu_counts, palette):
    case_best = run_data["case_best"]
    base_label = BASE_CASE_LABEL
    base_best = case_best.get(base_label, {})
    plotted_cases = [
        (label, path)
        for (label, path) in run_data["cases"]
        if label != base_label and label not in OMITTED_PLOT_CASES
    ]
    higher_is_better = bool(run_data.get("higher_is_better", False))

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
            speedup = _compute_speedup(base_runtime, runtime, higher_is_better)
            if not math.isfinite(speedup):
                continue
            x_pos = idx - (group_width / 2) + bar_width * li + bar_width / 2
            xs.append(x_pos)
            ys.append(speedup)
        if xs:
            ax.bar(xs, ys, width=bar_width, label=label, color=palette[li % len(palette)])

    ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(gpu_counts)), [str(g) for g in gpu_counts])
    ax.set_xlabel("GPU count", fontsize=PLOT_YLABEL_FONT_SIZE)
    _, y_max = ax.get_ylim()
    ax.set_ylim(0.8, max(1.2, y_max * 1.30))
    ax.set_title("")


def _plot_shared_base_speedups(ax, run_data, gpu_counts, shared_base_best, case_palette, metric_title: str):
    plotted_cases = [label for label in CASE_LABELS if label != BASE_CASE_LABEL]
    hatches = ["", "//", "xx", "\\\\"]
    total_series = len(run_data) * len(plotted_cases)
    group_width = 0.9
    bar_width = group_width / max(1, total_series)

    for model_idx, item in enumerate(run_data):
        higher_is_better = bool(item.get("higher_is_better", False))
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
                speedup = _compute_speedup(base_runtime, runtime, higher_is_better)
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
    ax.set_xlabel("GPU count", fontsize=PLOT_YLABEL_FONT_SIZE)
    ax.set_ylabel("Speedup / Base Runtime", fontsize=PLOT_YLABEL_FONT_SIZE)
    _, y_max = ax.get_ylim()
    ax.set_ylim(0.8, max(1.2, y_max * 1.30))
    ax.set_title("")

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
        bbox_to_anchor=(0.01, 0.90),
        ncol=1,
        frameon=True,
        fontsize=PLOT_LEGEND_FONT_SIZE,
        title="Hardware case",
        title_fontsize=PLOT_LEGEND_FONT_SIZE,
    )
    ax.add_artist(case_legend)
    ax.legend(
        handles=attention_handles,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.90),
        ncol=1,
        frameon=True,
        fontsize=PLOT_LEGEND_FONT_SIZE,
        title="Attention",
        title_fontsize=PLOT_LEGEND_FONT_SIZE,
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
    _apply_plot_font_style()
    inference_gbs_values = _parse_int_csv(args.inference_gbs, "--inference-gbs")
    model_config_paths = _parse_model_configs(args)
    shared_base_model_config = _parse_shared_base_model_config(args.shared_base_model_config)
    run_type_paths = list(model_config_paths)
    if shared_base_model_config is not None:
        run_type_paths.append(shared_base_model_config)
    all_inference = bool(run_type_paths) and all(_is_inference_model(path) for path in run_type_paths)
    if str(args.gpu_counts or "").strip():
        gpu_counts = _parse_gpu_counts(args.gpu_counts)
    else:
        if all_inference:
            gpu_counts = list(INFERENCE_GPU_COUNTS)
        else:
            gpu_counts = list(GPU_COUNTS)
    if all_inference:
        invalid_gpu_counts = [count for count in gpu_counts if count < INFERENCE_GPU_COUNT_MIN or count > INFERENCE_GPU_COUNT_MAX]
        if invalid_gpu_counts:
            raise ValueError(
                "Inference sweeps require --gpu-counts within [4,64]. "
                f"Invalid values: {invalid_gpu_counts}"
            )

    reference_model_path = shared_base_model_config or model_config_paths[0]
    reference_metric = _effective_metric_for_model(
        reference_model_path,
        args.metric,
        args.inference_decode_throughput,
    )
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if str(args.output_root or "").strip()
        else _default_output_root(model_config_paths, gpu_counts, shared_base_model_config, reference_metric)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = _comparison_output_path(
        output_root,
        model_config_paths,
        gpu_counts,
        shared_base_model_config,
        reference_metric,
    )
    reference_metadata = _load_model_metadata(reference_model_path)
    metric_title = _metric_title(reference_metric, reference_metadata.get("run_type", ""))
    context_title = _figure_context_title(model_config_paths, shared_base_model_config)
    figure_title = _speedup_figure_title(reference_model_path, metric_title)
    run_data = []
    total_models = len(model_config_paths)
    for model_idx, model_config_path in enumerate(model_config_paths, start=1):
        model_start = time.time()
        print(
            f"[progress] model {model_idx}/{total_models} start config={model_config_path}",
            flush=True,
        )
        item = _load_case_best_for_model(
            model_config_path,
            gpu_counts,
            args.plot_only,
            args.metric,
            args.inference_decode_throughput,
            output_root,
            inference_gbs_values,
            case_d_use_case_c_best=args.case_d_use_case_c_best,
        )
        run_data.append(item)
        print(
            f"[progress] model {model_idx}/{total_models} done "
            f"config={model_config_path.name} elapsed={time.time() - model_start:.1f}s",
            flush=True,
        )

    palette = sns.color_palette(
        "deep",
        n_colors=max(
            3,
            len([label for label in CASE_LABELS if label != BASE_CASE_LABEL]),
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
        for item in run_data:
            if item["metric"] != shared_item["metric"]:
                raise ValueError(
                    "All model configs must use the same effective metric for shared-base comparison. "
                    "Adjust --metric/--inference-decode-throughput inputs."
                )
        shared_base_best = shared_item["case_best"].get(BASE_CASE_LABEL, {})
        case_palette = {
            label: palette[idx % len(palette)]
            for idx, label in enumerate(CASE_LABELS[1:])
        }
        fig, ax = plt.subplots(1, 1, figsize=(28, 12))
        _plot_shared_base_speedups(
            ax,
            run_data,
            gpu_counts,
            shared_base_best,
            case_palette,
            metric_title,
        )
        fig.suptitle(figure_title, y=0.87, fontsize=PLOT_TITLE_FONT_SIZE)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
    else:
        fig, axes = plt.subplots(
            1,
            len(run_data),
            figsize=(18 * len(run_data), 10),
            sharey=True,
            squeeze=False,
        )
        flat_axes = axes[0]
        for ax, item in zip(flat_axes, run_data):
            _plot_model_speedups(ax, item, gpu_counts, palette)
        flat_axes[0].set_ylabel("Speedup / Base Runtime", fontsize=PLOT_YLABEL_FONT_SIZE)

        handles, labels = flat_axes[0].get_legend_handles_labels()
        fig.suptitle(figure_title, y=0.87, fontsize=PLOT_TITLE_FONT_SIZE)
        flat_axes[0].legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.84),
            ncol=max(1, len(labels)),
            frameon=True,
            fontsize=PLOT_LEGEND_FONT_SIZE,
            title_fontsize=PLOT_LEGEND_FONT_SIZE,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(str(output_path), dpi=200)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
