#!/usr/bin/env python3
"""
Canonical SuperPOD sweep harness for the A100 SXM4 bandwidth case study.

This harness orchestrates two parallelism sweeps on the same base hardware:
  - bw1x: use the base dim1/dim2 bandwidth
  - bw2x: double dim1/dim2 bandwidth

It then renders a combined two-panel beeswarm plot using the resulting shard
reports. The canonical outputs live under tools/pod_sweep with simpler artifact
names, while compatibility wrappers can still request the legacy layout.
"""

from __future__ import annotations

import argparse
import copy
import math
import re
import shlex
import shutil
import sys
import tempfile
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
for import_root in (str(REPO_ROOT), str(TOOLS_DIR)):
    if import_root in sys.path:
        sys.path.remove(import_root)
    sys.path.insert(0, import_root)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.lines import Line2D

import parallelism_sweep as sweep

try:
    from validation_scripts.plot_style import (
        IEEE_AXIS_TITLE_SIZE_PT,
        IEEE_DPI,
        IEEE_HALF_COLUMN_WIDTH_IN,
        IEEE_TITLE_SIZE_PT,
        ieee_rc_params,
    )
except ImportError:
    IEEE_HALF_COLUMN_WIDTH_IN = 3.5
    IEEE_AXIS_TITLE_SIZE_PT = 10
    IEEE_TITLE_SIZE_PT = 10
    IEEE_DPI = 200

    def ieee_rc_params():
        return {}


CANONICAL_OUTPUT_ROOT = TOOLS_DIR / "pod_sweep"
LEGACY_OUTPUT_ROOT = TOOLS_DIR / "superpod_sweep"
LEGACY_CHECKED_IN_PLOT = LEGACY_OUTPUT_ROOT / "parallelism_sweep_superpod_bw_combined_tp_cp_le16.png"

BASE_HW_CONFIG = (
    REPO_ROOT
    / "validation_scripts"
    / "validation_configs"
    / "hardware-config"
    / "A100_SXM4_80GB_base.yaml"
)
BASE_MODEL_CONFIG = (
    REPO_ROOT
    / "validation_scripts"
    / "validation_configs"
    / "model-config"
    / "Llama3.1-405B.yaml"
)
DERATE_CONFIG = (
    REPO_ROOT
    / "validation_scripts"
    / "validation_configs"
    / "harness_derates.yaml"
)
DERATE_DEVICE_TYPE = "A100_SXM4"

EXACT_GPU_COUNTS = [512, 1024, 2048, 4096]
SEQ_LEN = 33072
GLOBAL_BATCH_SIZE = 256
COLOR_GAMMA = 0.85
BEST_LINE_COLOR = "#1a1a1a"
CANONICAL_PLOT_NAME = "pod_sweep.png"
CANONICAL_SHARED_CACHE_NAME = "shared_runtime_cache.csv"
CANONICAL_MODE_CHOICES = ("plot", "shards", "all")
ARTIFACT_LAYOUT_CHOICES = ("canonical", "legacy")
AUTO_WORKER_LIMITS_BY_GPU_COUNT = {
    4096: 1,
}


@dataclass(frozen=True)
class OutputLayout:
    artifact_layout: str
    output_root: Path
    plot_path: Path
    shard_dir: Path
    cache_dir: Path
    shared_runtime_cache_path: Optional[Path]


def _append_derate_args(
    argv: List[str],
    derate_config_path: Optional[Path],
    derate_device_type: Optional[str],
) -> None:
    if derate_config_path is None:
        return
    if not str(derate_device_type or "").strip():
        raise ValueError("--derate-device-type must be provided when --derate-config is set.")
    argv.extend(
        [
            "--derate-config",
            str(derate_config_path),
            "--derate-device-type",
            str(derate_device_type),
        ]
    )


def _format_derate_status(
    derate_config_path: Optional[Path],
    derate_device_type: Optional[str],
) -> str:
    if derate_config_path is None:
        return "none"
    return f"{derate_config_path} ({derate_device_type})"


def _parse_gpu_counts(raw: str, default_values: Sequence[int]) -> List[int]:
    if not str(raw or "").strip():
        return list(default_values)
    values: List[int] = []
    for token in str(raw).split(","):
        text = token.strip()
        if not text:
            continue
        values.append(int(text))
    if not values:
        raise ValueError("--gpu-counts must include at least one integer.")
    return sorted(set(values))


def _resolve_output_layout(args: argparse.Namespace) -> OutputLayout:
    artifact_layout = str(args.artifact_layout or "canonical").strip().lower()
    if artifact_layout not in ARTIFACT_LAYOUT_CHOICES:
        raise ValueError(
            f"--artifact-layout must be one of {', '.join(ARTIFACT_LAYOUT_CHOICES)}."
        )

    if str(args.output_root or "").strip():
        output_root = Path(args.output_root).expanduser().resolve()
    elif artifact_layout == "legacy":
        output_root = LEGACY_OUTPUT_ROOT.resolve()
    else:
        output_root = CANONICAL_OUTPUT_ROOT.resolve()

    if str(args.plot_path or "").strip():
        plot_path = Path(args.plot_path).expanduser().resolve()
    elif artifact_layout == "legacy":
        plot_path = output_root / "parallelism_sweep_superpod_bw_combined.png"
    else:
        plot_path = output_root / CANONICAL_PLOT_NAME

    if str(args.shard_dir or "").strip():
        shard_dir = Path(args.shard_dir).expanduser().resolve()
    elif artifact_layout == "legacy":
        shard_dir = output_root
    else:
        shard_dir = output_root / "shards"

    if str(args.cache_dir or "").strip():
        cache_dir = Path(args.cache_dir).expanduser().resolve()
    elif artifact_layout == "legacy":
        cache_dir = output_root
    else:
        cache_dir = output_root / "cache"

    shared_runtime_cache_path: Optional[Path]
    if str(args.shared_runtime_cache_path or "").strip():
        shared_runtime_cache_path = Path(args.shared_runtime_cache_path).expanduser().resolve()
    elif artifact_layout == "legacy":
        shared_runtime_cache_path = None
    else:
        shared_runtime_cache_path = cache_dir / CANONICAL_SHARED_CACHE_NAME

    return OutputLayout(
        artifact_layout=artifact_layout,
        output_root=output_root,
        plot_path=plot_path,
        shard_dir=shard_dir,
        cache_dir=cache_dir,
        shared_runtime_cache_path=shared_runtime_cache_path,
    )


def _report_path(layout: OutputLayout, variant: str, gpu_count: int) -> Path:
    if layout.artifact_layout == "legacy":
        return layout.shard_dir / f"parallelism_sweep_superpod_{variant}_{gpu_count}g.tsv"
    return layout.shard_dir / f"{variant}_{gpu_count}g.tsv"


def _runtime_cache_path(layout: OutputLayout, variant: str, gpu_count: int) -> Path:
    if layout.shared_runtime_cache_path is not None:
        return layout.shared_runtime_cache_path
    return layout.cache_dir / f"parallelism_sweep_superpod_{variant}_{gpu_count}g_cache.csv"


def _error_log_path(
    layout: OutputLayout,
    variant: str,
    gpu_count: int,
    error_log_dir: Optional[Path],
) -> Optional[Path]:
    if error_log_dir is None:
        return None
    if layout.artifact_layout == "legacy":
        name = f"parallelism_sweep_superpod_{variant}_{gpu_count}g_errors.jsonl"
    else:
        name = f"{variant}_{gpu_count}g_errors.jsonl"
    return error_log_dir / name


def _legacy_report_path(variant: str, gpu_count: int, legacy_root: Path) -> Path:
    return legacy_root / f"parallelism_sweep_superpod_{variant}_{gpu_count}g.tsv"


def _legacy_aggregate_report_path(variant: str) -> Path:
    return TOOLS_DIR / f"parallelism_sweep_superpod_{variant}.tsv"


def _read_yaml(path: Path) -> MutableMapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, MutableMapping):
        raise ValueError(f"Expected YAML mapping at root: {path}")
    return data


def _write_yaml(path: Path, data: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)


def _scale_numeric_with_optional_unit(value: object, factor: float) -> object:
    if isinstance(value, (int, float)):
        return float(value) * factor
    if isinstance(value, list):
        return [_scale_numeric_with_optional_unit(v, factor) for v in value]
    if isinstance(value, str):
        match = re.match(
            r"^\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*([A-Za-z][A-Za-z0-9/_-]*)?\s*$",
            value,
        )
        if not match:
            raise ValueError(f"Unsupported bandwidth format: {value!r}")
        number = float(match.group(1)) * factor
        unit = match.group(2)
        number_text = f"{number:g}"
        return f"{number_text} {unit}".strip() if unit else number_text
    raise ValueError(f"Unsupported bandwidth type: {type(value).__name__}")


def _set_scaled_bandwidth(hw_dict: MutableMapping[str, object], dim_id: str, factor: float) -> None:
    network = hw_dict.get("network")
    if not isinstance(network, MutableMapping):
        raise ValueError("hardware config missing mapping: network")
    dims = network.get("dimensions")
    if not isinstance(dims, list):
        raise ValueError("hardware config missing list: network.dimensions")

    for dim in dims:
        if not isinstance(dim, MutableMapping):
            continue
        if str(dim.get("id", "")).strip() != dim_id:
            continue
        topology = dim.get("topology")
        if not isinstance(topology, MutableMapping):
            raise ValueError(f"network.dimensions[{dim_id}].topology is not a mapping")
        if "bandwidth" not in topology:
            raise ValueError(f"network.dimensions[{dim_id}].topology missing bandwidth")
        topology["bandwidth"] = _scale_numeric_with_optional_unit(topology["bandwidth"], factor)
        return
    raise ValueError(f"Could not find network dimension id={dim_id!r}")


def _build_temp_model(path: Path) -> MutableMapping[str, object]:
    model = _read_yaml(path)
    model_param = model.setdefault("model_param", {})
    if not isinstance(model_param, MutableMapping):
        raise ValueError("model config has non-mapping model_param")
    model_param["global_batch_size"] = int(GLOBAL_BATCH_SIZE)
    model_param["seq_len"] = int(SEQ_LEN)
    return model


def _parallelism_snapshot(parallelism: Mapping[str, object]) -> Dict[str, int]:
    train = parallelism.get("train")
    if not isinstance(train, Mapping):
        train = {}
    return {
        "tp": int(parallelism.get("tp", 1) or 1),
        "cp": int(parallelism.get("cp", 1) or 1),
        "pp": int(parallelism.get("pp", 1) or 1),
        "dp": int(train.get("dp", 1) or 1),
    }


def _rows_from_results(
    results: Sequence[Mapping[str, object]],
    variant: str,
    gpu_counts: Sequence[int],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    finite_runtime = [
        float(item.get("runtime", float("nan")))
        for item in results
        if math.isfinite(float(item.get("runtime", float("nan"))))
    ]
    fallback_runtime = max(finite_runtime) * 1.05 if finite_runtime else 1.0

    for idx, item in enumerate(results):
        num_gpus = int(item.get("num_gpus", 0) or 0)
        if num_gpus not in gpu_counts:
            continue
        parallelism = item.get("parallelism")
        if not isinstance(parallelism, Mapping):
            continue
        snap = _parallelism_snapshot(parallelism)
        runtime = float(item.get("runtime", float("nan")))
        if not math.isfinite(runtime):
            runtime = fallback_runtime
        rows.append(
            {
                "variant": variant,
                "row_id": f"{variant}_{idx}",
                "num_gpus": num_gpus,
                "gpu_exp": math.log2(float(num_gpus)),
                "tp": snap["tp"],
                "cp": snap["cp"],
                "dp": snap["dp"],
                "pp": snap["pp"],
                "runtime": runtime,
                "memory_exceeded": bool(item.get("memory_exceeded", False)),
            }
        )
    return rows


def _filter_results_by_tp_plus_cp(
    results: Sequence[Mapping[str, object]],
    max_tp_plus_cp: int,
) -> List[Mapping[str, object]]:
    if max_tp_plus_cp <= 0:
        return list(results)

    filtered: List[Mapping[str, object]] = []
    for item in results:
        parallelism = item.get("parallelism")
        if not isinstance(parallelism, Mapping):
            filtered.append(item)
            continue
        snap = _parallelism_snapshot(parallelism)
        if (snap["tp"] + snap["cp"]) <= int(max_tp_plus_cp):
            filtered.append(item)
    return filtered


def _normalize(values: np.ndarray, gamma: float = COLOR_GAMMA) -> np.ndarray:
    if values.size == 0:
        return values
    xmin = float(np.min(values))
    xmax = float(np.max(values))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        base = np.zeros_like(values, dtype=float)
    else:
        base = (values - xmin) / (xmax - xmin)
    return np.power(base, gamma)


def _assign_colors(df_all: pd.DataFrame) -> pd.DataFrame:
    df = df_all.copy()
    df["rgba"] = pd.Series([(0.6, 0.6, 0.6, 0.7)] * len(df), index=df.index, dtype=object)

    valid = ~df["memory_exceeded"].astype(bool)
    if valid.any():
        r_raw = np.log2(df.loc[valid, "tp"].to_numpy(dtype=float) + df.loc[valid, "cp"].to_numpy(dtype=float))
        g_raw = np.log2(df.loc[valid, "pp"].to_numpy(dtype=float))
        b_raw = np.log2(df.loc[valid, "dp"].to_numpy(dtype=float))
        r = _normalize(r_raw)
        g = _normalize(g_raw)
        b = _normalize(b_raw)
        rgba_vals = [(float(r[i]), float(g[i]), float(b[i]), 0.9) for i in range(len(r))]
        df.loc[valid, "rgba"] = pd.Series(rgba_vals, index=df.index[valid], dtype=object)
    return df


def _draw_panel(ax: plt.Axes, df: pd.DataFrame, subtitle: str, order: List[float]) -> None:
    ax.set_title(subtitle)
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([str(int(2**e)) for e in order])
        ax.grid(alpha=0.3, axis="y")
        return

    dfx = df.copy()
    dfx["gpu_exp_cat"] = pd.Categorical(dfx["gpu_exp"], categories=order, ordered=True)

    palette = {rid: rgba for rid, rgba in zip(dfx["row_id"], dfx["rgba"])}
    sns.swarmplot(
        data=dfx,
        x="gpu_exp_cat",
        y="runtime",
        hue="row_id",
        palette=palette,
        size=max(1.0, float(sweep.PLOT_POINT_SIZE)),
        linewidth=float(sweep.PLOT_POINT_EDGE),
        edgecolor="black",
        dodge=False,
        legend=False,
        ax=ax,
    )

    has_points = False
    for coll in ax.collections:
        offsets = getattr(coll, "get_offsets", lambda: None)()
        if offsets is not None and len(offsets):
            has_points = True
            break
    if not has_points:
        rng = np.random.RandomState(int(sweep.PLOT_JITTER_SEED))
        x_base = dfx["gpu_exp_cat"].cat.codes.to_numpy(dtype=float)
        jitter = rng.uniform(-0.28, 0.28, size=len(dfx))
        ax.scatter(
            x_base + jitter,
            dfx["runtime"].to_numpy(dtype=float),
            s=float(sweep.PLOT_POINT_SIZE) ** 2,
            c=list(dfx["rgba"]),
            edgecolor="black",
            linewidth=float(sweep.PLOT_POINT_EDGE),
        )

    best_by_gpu: Dict[int, float] = {}
    for _, row in dfx.iterrows():
        if bool(row["memory_exceeded"]):
            continue
        runtime = float(row["runtime"])
        if not math.isfinite(runtime):
            continue
        ng = int(row["num_gpus"])
        cur = best_by_gpu.get(ng)
        if cur is None or runtime < cur:
            best_by_gpu[ng] = runtime
    if best_by_gpu:
        points: List[Tuple[int, float]] = []
        for ng, runtime in best_by_gpu.items():
            exp = math.log2(float(ng))
            if exp in order:
                points.append((order.index(exp), runtime))
        if points:
            points.sort(key=lambda item: item[0])
            xs = [item[0] for item in points]
            ys = [item[1] for item in points]
            ax.plot(xs, ys, color=BEST_LINE_COLOR, linewidth=1.2, zorder=5)
            ax.scatter(xs, ys, s=100, marker="*", c="white", edgecolor="black", zorder=6)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([str(int(2**e)) for e in order])
    ax.grid(alpha=0.3, axis="y")
    ax.set_xlabel("")
    ax.set_ylabel("")


@contextmanager
def _suppress_sweep_plots() -> Iterable[None]:
    names = (
        "plot_results",
        "plot_best_runtimes",
        "plot_best_runtimes_per_gpu",
        "plot_best_runtimes_per_gpu_combined",
        "plot_speedup_per_gpu_combined",
    )
    saved = {name: getattr(sweep, name) for name in names if hasattr(sweep, name)}

    def _noop(*_args, **_kwargs):
        return None

    try:
        for name in saved:
            setattr(sweep, name, _noop)
        yield
    finally:
        for name, value in saved.items():
            setattr(sweep, name, value)


def _append_superpod_error_log(
    *,
    error_log_path: Optional[Path],
    variant: str,
    gpu_count: int,
    hw_path: Path,
    model_path: Path,
    report_path: Path,
    runtime_cache_path: Path,
    exc: BaseException,
) -> None:
    if error_log_path is None:
        return
    sweep._append_error_log(
        {
            "status": "error",
            "source": "superpod_sweep",
            "variant": variant,
            "num_gpus": int(gpu_count),
            "hardware_config": str(hw_path.resolve()),
            "model_config": str(model_path.resolve()),
            "report_path": str(report_path.resolve()),
            "runtime_cache_path": str(runtime_cache_path.resolve()),
            "error": f"{exc.__class__.__name__}: {exc}",
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        },
        str(error_log_path),
    )


def _run_sweep_variant(
    variant: str,
    hw_path: Path,
    model_path: Path,
    report_path: Path,
    runtime_cache_path: Path,
    error_log_path: Optional[Path],
    gpu_count: int,
    max_tp_plus_cp: int,
    workers: Optional[int],
    label: str,
    derate_config_path: Optional[Path],
    derate_device_type: Optional[str],
) -> None:
    temp_report_path = report_path.with_name(f".{report_path.name}.tmp")
    overrides = {
        "REPORT_OUTPUT_PATH": str(temp_report_path),
        "RUNTIME_CACHE_PATH": str(runtime_cache_path),
        "PLOT_TITLE": "Parallelism options vs runtime",
        "MEM_AWARE_FILTER": False,
    }
    if error_log_path is not None:
        overrides["ERROR_LOG_PATH"] = str(error_log_path)
    if workers is not None and workers > 0:
        overrides["MAX_WORKERS"] = int(workers)

    old_values: Dict[str, object] = {}
    old_argv = list(sys.argv)
    try:
        if temp_report_path.exists():
            temp_report_path.unlink()
        for key, value in overrides.items():
            old_values[key] = getattr(sweep, key)
            setattr(sweep, key, value)
        sys.argv = [
            "parallelism_sweep.py",
            "--hardware-configs",
            str(hw_path),
            "--hardware-labels",
            label,
            "--model-config",
            str(model_path),
            "--gpu-count-min",
            str(int(gpu_count)),
            "--gpu-count-max",
            str(int(gpu_count)),
        ]
        _append_derate_args(sys.argv, derate_config_path, derate_device_type)
        try:
            with _suppress_sweep_plots():
                sweep.main()
            results = sweep.load_results_from_report(str(temp_report_path))
            if max_tp_plus_cp > 0:
                filtered = _filter_results_by_tp_plus_cp(results, max_tp_plus_cp)
                removed = len(results) - len(filtered)
                if removed > 0:
                    if not filtered:
                        raise RuntimeError(
                            f"{temp_report_path} has no rows after applying tp + cp <= {max_tp_plus_cp}."
                        )
                    results = filtered
                    print(
                        "[superpod_sweep] filtered "
                        f"{removed} row(s) with tp + cp > {max_tp_plus_cp} from {report_path}"
                    )
            sweep.write_report(results, str(report_path))
            print(f"[superpod_sweep] staged shard report to {report_path}")
        except BaseException as exc:
            _append_superpod_error_log(
                error_log_path=error_log_path,
                variant=variant,
                gpu_count=gpu_count,
                hw_path=hw_path,
                model_path=model_path,
                report_path=report_path,
                runtime_cache_path=runtime_cache_path,
                exc=exc,
            )
            raise
    finally:
        temp_report_path.unlink(missing_ok=True)
        sys.argv = old_argv
        for key, value in old_values.items():
            setattr(sweep, key, value)


def _results_from_aggregate_report(path: Path, gpu_count: int) -> List[Dict[str, object]]:
    results = sweep.load_results_from_report(str(path))
    return [
        item
        for item in results
        if int(item.get("num_gpus", 0) or 0) == int(gpu_count)
    ]


def _load_results_for_variant_gpu(
    layout: OutputLayout,
    variant: str,
    gpu_count: int,
    legacy_shard_root: Path,
) -> Tuple[List[Dict[str, object]], Optional[Path], str]:
    preferred = _report_path(layout, variant, gpu_count)
    if preferred.exists():
        return sweep.load_results_from_report(str(preferred)), preferred, "canonical"

    if layout.artifact_layout == "canonical":
        return [], None, "missing"

    legacy_shard = _legacy_report_path(variant, gpu_count, legacy_shard_root)
    if legacy_shard != preferred and legacy_shard.exists():
        return sweep.load_results_from_report(str(legacy_shard)), legacy_shard, "legacy-shard"

    legacy_aggregate = _legacy_aggregate_report_path(variant)
    if legacy_aggregate.exists():
        filtered = _results_from_aggregate_report(legacy_aggregate, gpu_count)
        if filtered:
            return filtered, legacy_aggregate, "legacy-aggregate"

    return [], None, "missing"


def _load_variant_results(
    layout: OutputLayout,
    variant: str,
    gpu_counts: Sequence[int],
    legacy_shard_root: Path,
) -> Tuple[List[Dict[str, object]], Dict[int, Tuple[Path, str]]]:
    results: List[Dict[str, object]] = []
    sources: Dict[int, Tuple[Path, str]] = {}
    missing: List[int] = []
    for gpu_count in gpu_counts:
        loaded, source_path, source_kind = _load_results_for_variant_gpu(
            layout,
            variant,
            gpu_count,
            legacy_shard_root,
        )
        if not loaded or source_path is None:
            missing.append(int(gpu_count))
            continue
        results.extend(loaded)
        sources[int(gpu_count)] = (source_path, source_kind)
    if missing:
        missing_text = ", ".join(str(item) for item in missing)
        raise RuntimeError(
            f"Missing {variant} shard reports for GPU counts: {missing_text}. "
            f"Expected them under {layout.shard_dir}. "
            "Run with '--mode shards' or '--mode all' to regenerate them."
        )
    return results, sources


def _print_run_header(title: str) -> None:
    print(f"\n[superpod_sweep] {title}")


def _print_command(label: str, argv: Sequence[str]) -> None:
    print(f"[superpod_sweep] {label}: {shlex.join([str(item) for item in argv])}")


def _effective_workers(requested_workers: Optional[int], gpu_count: int) -> Optional[int]:
    if requested_workers is not None:
        return int(requested_workers)
    return AUTO_WORKER_LIMITS_BY_GPU_COUNT.get(int(gpu_count))


def _generate_shards(
    layout: OutputLayout,
    gpu_counts: Sequence[int],
    max_tp_plus_cp: int,
    workers: Optional[int],
    keep_temp: bool,
    error_log_dir: Optional[Path],
    derate_config_path: Optional[Path],
    derate_device_type: Optional[str],
) -> Tuple[List[Path], List[Path], List[Path]]:
    report_paths: List[Path] = []
    cache_paths: List[Path] = []
    error_log_paths: List[Path] = []
    layout.output_root.mkdir(parents=True, exist_ok=True)
    layout.shard_dir.mkdir(parents=True, exist_ok=True)
    layout.cache_dir.mkdir(parents=True, exist_ok=True)
    if error_log_dir is not None:
        error_log_dir.mkdir(parents=True, exist_ok=True)
    if layout.shared_runtime_cache_path is not None:
        layout.shared_runtime_cache_path.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(tempfile.mkdtemp(prefix="superpod_sweep_", dir="/tmp"))
    try:
        model_override = _build_temp_model(BASE_MODEL_CONFIG)
        model_path = temp_dir / "Llama3.1-405B_gbs256_seq33072.yaml"
        _write_yaml(model_path, model_override)

        hw_base = _read_yaml(BASE_HW_CONFIG)

        hw_1x = copy.deepcopy(hw_base)
        _set_scaled_bandwidth(hw_1x, "dim1", 1.0)
        _set_scaled_bandwidth(hw_1x, "dim2", 1.0)
        hw_1x_path = temp_dir / "A100_SXM4_80GB_base_dim12_bw1x.yaml"
        _write_yaml(hw_1x_path, hw_1x)

        hw_2x = copy.deepcopy(hw_base)
        _set_scaled_bandwidth(hw_2x, "dim1", 2.0)
        _set_scaled_bandwidth(hw_2x, "dim2", 2.0)
        hw_2x_path = temp_dir / "A100_SXM4_80GB_base_dim12_bw2x.yaml"
        _write_yaml(hw_2x_path, hw_2x)

        print(
            "[superpod_sweep] base configs: "
            f"hardware={BASE_HW_CONFIG} model={BASE_MODEL_CONFIG} "
            f"derates={_format_derate_status(derate_config_path, derate_device_type)}"
        )
        print(
            "[superpod_sweep] comparison overrides: "
            "bw1x=dim1/dim2 x1, bw2x=dim1/dim2 x2, "
            f"global_batch_size={GLOBAL_BATCH_SIZE}, seq_len={SEQ_LEN}"
        )

        for gpu_count in gpu_counts:
            for variant, label, hw_path in (
                ("bw1x", "BW1X", hw_1x_path),
                ("bw2x", "BW2X", hw_2x_path),
            ):
                effective_workers = _effective_workers(workers, int(gpu_count))
                report_path = _report_path(layout, variant, int(gpu_count))
                runtime_cache_path = _runtime_cache_path(layout, variant, int(gpu_count))
                error_log_path = _error_log_path(layout, variant, int(gpu_count), error_log_dir)
                report_path.parent.mkdir(parents=True, exist_ok=True)
                runtime_cache_path.parent.mkdir(parents=True, exist_ok=True)
                if error_log_path is not None:
                    error_log_path.parent.mkdir(parents=True, exist_ok=True)
                cache_paths.append(runtime_cache_path)
                report_paths.append(report_path)
                if error_log_path is not None:
                    error_log_paths.append(error_log_path)
                if report_path.exists():
                    print(
                        "[superpod_sweep] skip existing shard: "
                        f"{variant} {int(gpu_count)}g -> {report_path}"
                    )
                    continue
                sweep_argv = [
                    "parallelism_sweep.py",
                    "--hardware-configs",
                    str(hw_path),
                    "--hardware-labels",
                    label,
                    "--model-config",
                    str(model_path),
                    "--gpu-count-min",
                    str(int(gpu_count)),
                    "--gpu-count-max",
                    str(int(gpu_count)),
                ]
                if effective_workers is not None and effective_workers > 0:
                    print(
                        "[superpod_sweep] worker limit: "
                        f"{variant} {int(gpu_count)}g uses {effective_workers} worker(s)"
                    )
                _append_derate_args(sweep_argv, derate_config_path, derate_device_type)
                _print_command(
                    f"run {variant} {int(gpu_count)}g",
                    sweep_argv,
                )
                _run_sweep_variant(
                    variant=variant,
                    hw_path=hw_path,
                    model_path=model_path,
                    report_path=report_path,
                    runtime_cache_path=runtime_cache_path,
                    error_log_path=error_log_path,
                    gpu_count=int(gpu_count),
                    max_tp_plus_cp=max_tp_plus_cp,
                    workers=effective_workers,
                    label=label,
                    derate_config_path=derate_config_path,
                    derate_device_type=derate_device_type,
                )
    finally:
        if keep_temp:
            print(f"[superpod_sweep] temporary configs kept at {temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)

    deduped_cache_paths = sorted(set(cache_paths))
    deduped_report_paths = sorted(set(report_paths))
    deduped_error_log_paths = sorted(set(error_log_paths))
    return deduped_report_paths, deduped_cache_paths, deduped_error_log_paths


def _render_combined_plot(
    plot_path: Path,
    gpu_counts: Sequence[int],
    results_1x: Sequence[Mapping[str, object]],
    results_2x: Sequence[Mapping[str, object]],
    max_tp_plus_cp: int,
) -> None:
    rows_1x = _rows_from_results(results_1x, "bw1x", gpu_counts)
    rows_2x = _rows_from_results(results_2x, "bw2x", gpu_counts)
    df_all = pd.DataFrame(rows_1x + rows_2x)
    if df_all.empty:
        raise RuntimeError("No results found in reports; cannot build combined plot.")
    if max_tp_plus_cp > 0:
        df_all = df_all[(df_all["tp"] + df_all["cp"]) <= max_tp_plus_cp].copy()
        if df_all.empty:
            raise RuntimeError(
                f"No results remain after applying tp + cp <= {max_tp_plus_cp} filter."
            )
    df_all = _assign_colors(df_all)

    order = [math.log2(float(g)) for g in gpu_counts]
    df_1x = df_all[df_all["variant"] == "bw1x"].copy()
    df_2x = df_all[df_all["variant"] == "bw2x"].copy()

    with plt.rc_context(ieee_rc_params()):
        fig_w = IEEE_HALF_COLUMN_WIDTH_IN * 2.05
        fig_h = IEEE_HALF_COLUMN_WIDTH_IN * 0.95
        fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), sharey=True)
        _draw_panel(axes[0], df_1x, "1x SuperPOD BW", order)
        _draw_panel(axes[1], df_2x, "2x SuperPOD BW", order)
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")

        fig.suptitle("Parallelism options vs runtime", fontsize=IEEE_TITLE_SIZE_PT, y=0.99)
        fig.supxlabel("Number of GPUs", y=0.04, fontsize=IEEE_AXIS_TITLE_SIZE_PT)
        fig.supylabel("Runtime (s)", fontsize=IEEE_AXIS_TITLE_SIZE_PT)
        fig.text(
            0.985,
            0.015,
            "Color: R=log2(tp+cp), G=log2(pp), B=log2(dp)",
            ha="right",
            va="bottom",
            fontsize=7,
            alpha=0.85,
        )

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=6,
                markerfacecolor=(0.2, 0.4, 0.8, 0.9),
                markeredgecolor="black",
                label="Valid config (parallelism color)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=6,
                markerfacecolor=(0.6, 0.6, 0.6, 0.7),
                markeredgecolor="black",
                label="OOM (rejected)",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="-",
                linewidth=1.2,
                color=BEST_LINE_COLOR,
                markersize=9,
                markerfacecolor="white",
                markeredgecolor="black",
                label="Best per GPU count",
            ),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 0.92),
            frameon=True,
        )

        fig.subplots_adjust(left=0.08, right=0.995, top=0.77, bottom=0.18, wspace=0.08)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(plot_path), dpi=IEEE_DPI, bbox_inches="tight")
        plt.close(fig)
    print(f"[superpod_sweep] saved combined plot to {plot_path}")


def _print_artifact_summary(
    *,
    plot_path: Path,
    report_paths: Sequence[Path],
    cache_paths: Sequence[Path],
    error_log_paths: Sequence[Path],
    report_sources: Mapping[str, Dict[int, Tuple[Path, str]]],
) -> None:
    print("\n[superpod_sweep] artifacts")
    print(f"[superpod_sweep] plot: {plot_path}")
    if report_paths:
        print("[superpod_sweep] shard reports:")
        for path in report_paths:
            print(f"[superpod_sweep]   - {path}")
    if cache_paths:
        print("[superpod_sweep] runtime cache:")
        for path in cache_paths:
            print(f"[superpod_sweep]   - {path}")
    if error_log_paths:
        print("[superpod_sweep] error logs:")
        for path in error_log_paths:
            suffix = "" if path.exists() else " (not created)"
            print(f"[superpod_sweep]   - {path}{suffix}")
    if report_sources:
        print("[superpod_sweep] plot inputs:")
        for variant in sorted(report_sources):
            for gpu_count in sorted(report_sources[variant]):
                path, source_kind = report_sources[variant][gpu_count]
                print(
                    "[superpod_sweep]   - "
                    f"{variant} {gpu_count}g: {path} ({source_kind})"
                )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonical SuperPOD bandwidth case-study harness."
    )
    parser.add_argument(
        "mode_positional",
        nargs="?",
        choices=CANONICAL_MODE_CHOICES,
        help="Optional shorthand for --mode.",
    )
    parser.add_argument(
        "--mode",
        choices=CANONICAL_MODE_CHOICES,
        default="",
        help="Select whether to render the plot, regenerate shards, or do both.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Compatibility alias for '--mode plot'.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional worker override passed through to tools/parallelism_sweep.py.",
    )
    parser.add_argument(
        "--output",
        "--plot-path",
        dest="plot_path",
        type=str,
        default="",
        help="Output path for the combined plot.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Root directory for canonical artifacts.",
    )
    parser.add_argument(
        "--shard-dir",
        type=str,
        default="",
        help="Optional directory override for shard TSVs.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="",
        help="Optional directory override for runtime-cache artifacts.",
    )
    parser.add_argument(
        "--gpu-counts",
        type=str,
        default=",".join(str(item) for item in EXACT_GPU_COUNTS),
        help="Comma-separated GPU counts to evaluate or render.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary model/hardware YAML files for debugging.",
    )
    parser.add_argument(
        "--max-tp-plus-cp",
        type=int,
        default=16,
        help="Keep only shard rows and plot points with tp + cp <= this value. Use 0 to disable.",
    )
    parser.add_argument(
        "--derate-config",
        type=str,
        default="",
        help=(
            "Optional path to derate YAML. When unset, pod_sweep uses the raw hardware config "
            f"without derates. Example: {DERATE_CONFIG}"
        ),
    )
    parser.add_argument(
        "--derate-device-type",
        type=str,
        default="",
        help=(
            "Optional device type entry in --derate-config to apply. Must be provided together "
            f"with --derate-config. Example: {DERATE_DEVICE_TYPE}"
        ),
    )
    parser.add_argument(
        "--shared-runtime-cache-path",
        type=str,
        default="",
        help="Optional shared runtime cache CSV for tools/parallelism_sweep.py cache reuse.",
    )
    parser.add_argument(
        "--error-log-dir",
        type=str,
        default="",
        help="Optional directory for per-shard JSONL error logs. When unset, errors are only printed.",
    )
    parser.add_argument(
        "--artifact-layout",
        choices=ARTIFACT_LAYOUT_CHOICES,
        default="canonical",
        help="Artifact naming/layout to use. Canonical is the new simplified layout.",
    )
    parser.add_argument(
        "--legacy-shard-root",
        type=str,
        default=str(LEGACY_OUTPUT_ROOT),
        help="Legacy shard directory used as a read-only fallback only for legacy artifact layout.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _selected_mode(args: argparse.Namespace) -> str:
    mode = str(args.mode or "").strip().lower()
    positional = str(args.mode_positional or "").strip().lower()
    if mode and positional and mode != positional:
        raise ValueError("Positional mode and --mode must match when both are provided.")
    if args.plot_only:
        if mode and mode != "plot":
            raise ValueError("--plot-only cannot be combined with --mode values other than 'plot'.")
        mode = "plot"
    if positional:
        mode = positional
    return mode or "plot"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    mode = _selected_mode(args)
    layout = _resolve_output_layout(args)
    gpu_counts = _parse_gpu_counts(args.gpu_counts, EXACT_GPU_COUNTS)
    derate_config_raw = str(args.derate_config or "").strip()
    derate_device_type_raw = str(args.derate_device_type or "").strip()
    if bool(derate_config_raw) != bool(derate_device_type_raw):
        raise ValueError("Both --derate-config and --derate-device-type must be provided together.")
    derate_config_path = (
        Path(derate_config_raw).expanduser().resolve() if derate_config_raw else None
    )
    derate_device_type = derate_device_type_raw or None
    legacy_shard_root = Path(args.legacy_shard_root).expanduser().resolve()
    error_log_dir = (
        Path(args.error_log_dir).expanduser().resolve()
        if str(args.error_log_dir or "").strip()
        else None
    )

    report_paths: List[Path] = []
    cache_paths: List[Path] = []
    error_log_paths: List[Path] = []
    report_sources: Dict[str, Dict[int, Tuple[Path, str]]] = {}

    _print_run_header(f"mode={mode}")
    print(
        "[superpod_sweep] canonical harness: "
        f"{Path(__file__).resolve()}"
    )
    print(
        "[superpod_sweep] configs: "
        f"hardware={BASE_HW_CONFIG} model={BASE_MODEL_CONFIG} "
        f"derates={_format_derate_status(derate_config_path, derate_device_type)}"
    )
    print(
        "[superpod_sweep] overrides: "
        "compare bw1x vs bw2x by scaling dim1/dim2 bandwidth, "
        f"global_batch_size={GLOBAL_BATCH_SIZE}, seq_len={SEQ_LEN}, "
        f"max_tp_plus_cp={int(args.max_tp_plus_cp)}"
    )
    print(f"[superpod_sweep] artifact layout: {layout.artifact_layout}")
    print(f"[superpod_sweep] output root: {layout.output_root}")
    if error_log_dir is not None:
        print(f"[superpod_sweep] error log dir: {error_log_dir}")

    if mode in ("shards", "all"):
        report_paths, cache_paths, error_log_paths = _generate_shards(
            layout=layout,
            gpu_counts=gpu_counts,
            max_tp_plus_cp=int(args.max_tp_plus_cp),
            workers=args.workers,
            keep_temp=bool(args.keep_temp),
            error_log_dir=error_log_dir,
            derate_config_path=derate_config_path,
            derate_device_type=derate_device_type,
        )

    if mode in ("plot", "all"):
        results_1x, sources_1x = _load_variant_results(layout, "bw1x", gpu_counts, legacy_shard_root)
        results_2x, sources_2x = _load_variant_results(layout, "bw2x", gpu_counts, legacy_shard_root)
        report_sources["bw1x"] = sources_1x
        report_sources["bw2x"] = sources_2x
        if not report_paths:
            report_paths = [
                sources_1x[int(gpu_count)][0] for gpu_count in gpu_counts
            ] + [
                sources_2x[int(gpu_count)][0] for gpu_count in gpu_counts
            ]
        _render_combined_plot(
            plot_path=layout.plot_path,
            gpu_counts=gpu_counts,
            results_1x=results_1x,
            results_2x=results_2x,
            max_tp_plus_cp=int(args.max_tp_plus_cp),
        )

    _print_artifact_summary(
        plot_path=layout.plot_path,
        report_paths=sorted(set(report_paths)),
        cache_paths=sorted(set(cache_paths)),
        error_log_paths=sorted(set(error_log_paths)),
        report_sources=report_sources,
    )
    if layout.artifact_layout == "canonical" and layout.plot_path == CANONICAL_OUTPUT_ROOT / CANONICAL_PLOT_NAME:
        print(
            "[superpod_sweep] reference legacy plot: "
            f"{LEGACY_CHECKED_IN_PLOT}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
