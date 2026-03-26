#!/usr/bin/env python3

"""
MosaicML training validation harness.

Runs RAPID-LLM training-time estimation against rows in:
  validation_scripts/mosiacml_data/h100_80gb_bf16.csv
and compares predicted training time to inferred_total_latency_s.
"""

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

try:
    from .validation_helpers import ValidationSpec, parse_training_time, run_validation_suite
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from validation_scripts.validation_helpers import (  # type: ignore
        ValidationSpec,
        parse_training_time,
        run_validation_suite,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_PERF = PROJECT_ROOT / "run_perf.py"

DEFAULT_INPUT_CSV = PROJECT_ROOT / "validation_scripts" / "mosiacml_data" / "h100_80gb_bf16.csv"
DEFAULT_MODEL_CONFIG = (
    PROJECT_ROOT / "validation_scripts" / "validation_configs" / "model-config" / "MPT-7b.yaml"
)
DEFAULT_MODEL_CONFIG_DIR = (
    PROJECT_ROOT / "validation_scripts" / "validation_configs" / "model-config"
)
DEFAULT_HW_CONFIG = (
    PROJECT_ROOT / "validation_scripts" / "validation_configs" / "hardware-config" / "H100_SXM5_80GB.mosaic_train.numgpus8.best.yaml"
)
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "output" / "validation" / "train" / "mosiacml_h100_bf16_all.csv"
DEFAULT_PLOT_OUTPUT = PROJECT_ROOT / "output" / "validation" / "train" / "mosiacml_h100_bf16_all.png"

MODEL_CONFIG_BY_SIZE = {
    "760m": "MPT-760m.yaml",
    "1b": "MPT-1b.yaml",
    "3b": "MPT-3b.yaml",
    "7b": "MPT-7b.yaml",
    "13b": "MPT-13b.yaml",
    "30b": "MPT-30b.yaml",
    "70b": "MPT-70b.yaml",
}

FIXED_TP = 1
FIXED_PP = 1
FIXED_CP = 1
FIXED_TP_SP = False
FIXED_MB = 1

REQUIRED_COLUMNS = (
    "model_size",
    "seq_len",
    "num_gpus",
    "micro_batch_size",
    "gradient_accumulation_steps",
    "global_batch_size",
    "activation_checkpointing",
    "inferred_total_latency_s",
)


@dataclass
class MosaicCase:
    case_index: int
    model_size: str
    seq_len: int
    num_gpus: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    global_batch_size: int
    activation_checkpointing: bool
    inferred_total_latency_s: float
    dp: int

    @property
    def label(self) -> str:
        return (
            f"{self.model_size} seq={self.seq_len} gpus={self.num_gpus} "
            f"mb={self.micro_batch_size} ga={self.gradient_accumulation_steps} "
            f"gbs={self.global_batch_size} ckpt={self.activation_checkpointing}"
        )


def _parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _normalize_model_size(value: str) -> str:
    return str(value).strip().lower()


def _is_all_models(value: str) -> bool:
    return _normalize_model_size(value) in {"", "all", "*"}


def _load_cases(csv_path: Path, model_size: str) -> List[MosaicCase]:
    df = pd.read_csv(csv_path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    # Normalize numeric fields.
    for col in (
        "seq_len",
        "num_gpus",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "global_batch_size",
        "inferred_total_latency_s",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    model_filter = _normalize_model_size(model_size)
    if not _is_all_models(model_filter):
        df = df[df["model_size"].astype(str).str.strip().str.lower() == model_filter].copy()
        if df.empty:
            raise ValueError(f"No rows found for model_size={model_size!r} in {csv_path}.")
    elif df.empty:
        raise ValueError(f"No rows found in {csv_path}.")

    cases: List[MosaicCase] = []
    for case_index, (_, row) in enumerate(df.iterrows()):
        if pd.isna(row["inferred_total_latency_s"]):
            raise ValueError(
                f"Row {case_index} has missing/non-numeric inferred_total_latency_s for model_size={model_size!r}."
            )

        seq_len = int(row["seq_len"])
        num_gpus = int(row["num_gpus"])
        micro_batch_size = int(row["micro_batch_size"])
        grad_accum = int(row["gradient_accumulation_steps"])
        global_batch_size = int(row["global_batch_size"])
        activation_checkpointing = _parse_bool(row["activation_checkpointing"])
        inferred_total_latency_s = float(row["inferred_total_latency_s"])

        if num_gpus % FIXED_TP != 0:
            raise ValueError(
                f"Row {case_index}: num_gpus={num_gpus} is not divisible by tp={FIXED_TP}."
            )
        dp = num_gpus // FIXED_TP
        if dp <= 0:
            raise ValueError(f"Row {case_index}: computed dp must be positive (got {dp}).")

        cases.append(
            MosaicCase(
                case_index=case_index,
                model_size=_normalize_model_size(str(row["model_size"])),
                seq_len=seq_len,
                num_gpus=num_gpus,
                micro_batch_size=micro_batch_size,
                gradient_accumulation_steps=grad_accum,
                global_batch_size=global_batch_size,
                activation_checkpointing=activation_checkpointing,
                inferred_total_latency_s=inferred_total_latency_s,
                dp=dp,
            )
        )
    return cases


def _resolve_model_config_for_case(
    *,
    case: MosaicCase,
    model_size_filter: str,
    explicit_model_config: Optional[str],
    model_config_dir: Path,
) -> str:
    if explicit_model_config:
        if _is_all_models(model_size_filter):
            raise ValueError(
                "--model-config cannot be combined with --model-size=all. "
                "Use --model-config-dir for multi-model validation."
            )
        return str(Path(explicit_model_config))

    filename = MODEL_CONFIG_BY_SIZE.get(_normalize_model_size(case.model_size))
    if not filename:
        raise ValueError(
            f"Unsupported model_size={case.model_size!r}. "
            f"Known sizes: {sorted(MODEL_CONFIG_BY_SIZE.keys())}"
        )
    resolved = model_config_dir / filename
    if not resolved.exists():
        raise FileNotFoundError(
            f"Model config for model_size={case.model_size!r} was not found: {resolved}"
        )
    return str(resolved)


def _build_spec(
    case: MosaicCase,
    idx: int,
    model_config_path: str,
    hardware_config_path: str,
    *,
    use_flashattention: bool,
    attention_tile_size: Optional[int],
) -> ValidationSpec:
    effective_grad_accum = int(case.gradient_accumulation_steps) # * int(case.micro_batch_size)
    model_overrides = {
        "model_param": {
            "run_type": "training",
            "seq_len": int(case.seq_len),
            "global_batch_size": int(case.global_batch_size),
            "gradient_accumulation_steps": effective_grad_accum,
            "attention": {
                "use_flashattention": bool(use_flashattention),
            },
        }
    }
    if use_flashattention and attention_tile_size is not None:
        model_overrides["model_param"]["attention"]["attention_tile_size"] = int(attention_tile_size)

    hardware_overrides = {
        "parallelism": {
            "tp": int(FIXED_TP),
            "tp_sp": bool(FIXED_TP_SP),
            "cp": int(FIXED_CP),
            "pp": int(FIXED_PP),
            "mb": int(FIXED_MB),
            "train": {"dp": int(case.dp), "ep": 1, "tp_ep": True},
            "inference": {"replica_count": 1, "moe_dp": 1},
        },
        "sw_param": {
            "full_recomputation": bool(case.activation_checkpointing),
            "dp_zero_stage": 3,
        },
    }

    return ValidationSpec(
        label=case.label,
        model_overrides=model_overrides,
        hardware_overrides=hardware_overrides,
        model_config_path=model_config_path,
        hardware_config_path=hardware_config_path,
        metadata={
            "case_index": int(case.case_index),
            "model_size": case.model_size,
            "seq_len": int(case.seq_len),
            "num_gpus": int(case.num_gpus),
            "tp": int(FIXED_TP),
            "dp": int(case.dp),
            "pp": int(FIXED_PP),
            "cp": int(FIXED_CP),
            "micro_batch_size": int(case.micro_batch_size),
            "gradient_accumulation_steps": int(case.gradient_accumulation_steps),
            "effective_gradient_accumulation_steps": effective_grad_accum,
            "global_batch_size": int(case.global_batch_size),
            "activation_checkpointing": bool(case.activation_checkpointing),
        },
        order=idx,
    )


def build_specs(
    cases: Sequence[MosaicCase],
    model_size_filter: str,
    explicit_model_config: Optional[str],
    model_config_dir: Path,
    hardware_config_path: str,
    *,
    use_flashattention: bool,
    attention_tile_size: Optional[int],
) -> Tuple[List[ValidationSpec], Dict[int, float]]:
    specs: List[ValidationSpec] = []
    actual_lookup: Dict[int, float] = {}
    for idx, case in enumerate(cases):
        model_config_path = _resolve_model_config_for_case(
            case=case,
            model_size_filter=model_size_filter,
            explicit_model_config=explicit_model_config,
            model_config_dir=model_config_dir,
        )
        specs.append(
            _build_spec(
                case,
                idx,
                model_config_path,
                hardware_config_path,
                use_flashattention=use_flashattention,
                attention_tile_size=attention_tile_size,
            )
        )
        actual_lookup[int(case.case_index)] = float(case.inferred_total_latency_s)
    return specs, actual_lookup


def compute_rows(results, actual_lookup: Dict[int, float]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for res in results:
        meta = res.spec.metadata or {}
        case_index = int(meta.get("case_index", -1))
        pred = float(res.metrics.get("training_time_s", float("nan"))) if res.success else float("nan")
        actual = float(actual_lookup.get(case_index, float("nan")))

        if math.isnan(pred) or math.isnan(actual) or actual == 0:
            signed_pct_error = float("nan")
            abs_pct_error = float("nan")
        else:
            signed_pct_error = (pred - actual) / actual * 100.0
            abs_pct_error = abs(signed_pct_error)

        rows.append(
            {
                "case_index": case_index,
                "model_size": meta.get("model_size"),
                "seq_len": int(meta.get("seq_len", 0)),
                "num_gpus": int(meta.get("num_gpus", 0)),
                "tp": int(meta.get("tp", FIXED_TP)),
                "dp": int(meta.get("dp", 0)),
                "pp": int(meta.get("pp", FIXED_PP)),
                "cp": int(meta.get("cp", FIXED_CP)),
                "micro_batch_size": int(meta.get("micro_batch_size", 0)),
                "gradient_accumulation_steps": int(meta.get("gradient_accumulation_steps", 0)),
                "effective_gradient_accumulation_steps": int(
                    meta.get("effective_gradient_accumulation_steps", 0)
                ),
                "global_batch_size": int(meta.get("global_batch_size", 0)),
                "activation_checkpointing": bool(meta.get("activation_checkpointing", False)),
                "pred_training_time_s": pred,
                "actual_inferred_total_latency_s": actual,
                "signed_pct_error": signed_pct_error,
                "abs_pct_error": abs_pct_error,
                "success": bool(res.success),
                "error": res.error,
                "raw_output": res.raw_output,
            }
        )
    return rows


def _plot_error_facet_heatmap(rows: List[Dict[str, object]], title: str, output: Path) -> Optional[Path]:
    if not rows:
        return None

    df = pd.DataFrame(rows).copy()
    if df.empty:
        return None
    for col in ("seq_len", "num_gpus", "signed_pct_error"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    preferred_order = ["760m", "1b", "3b", "7b", "13b", "30b", "70b"]
    present_models = sorted({str(v).strip().lower() for v in df["model_size"].dropna().tolist()})
    ordered_models = [m for m in preferred_order if m in present_models]
    ordered_models.extend([m for m in present_models if m not in ordered_models])
    if not ordered_models:
        return None

    seq_values = sorted({int(v) for v in df["seq_len"].dropna().tolist()})
    gpu_values = sorted({int(v) for v in df["num_gpus"].dropna().tolist()})
    if not seq_values or not gpu_values:
        return None

    finite_signed = df["signed_pct_error"].replace([np.inf, -np.inf], np.nan).dropna()
    if finite_signed.empty:
        vbound = 1.0
    else:
        min_signed = float(finite_signed.min())
        max_signed = float(finite_signed.max())
        vbound = max(abs(min_signed), abs(max_signed))
    if vbound <= 0:
        vbound = 1.0

    n_models = len(ordered_models)
    ncols = min(3, n_models)
    nrows = int(math.ceil(n_models / float(ncols)))

    fig_w = max(8.0, 4.8 * ncols)
    fig_h = max(4.5, 3.9 * nrows)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(title, fontsize=13)

    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad(color="#efefef")
    image = None

    for idx, model in enumerate(ordered_models):
        ax = axes[idx // ncols][idx % ncols]
        model_df = df[df["model_size"].astype(str).str.lower() == model]

        pivot_signed = (
            model_df.pivot_table(
                index="num_gpus",
                columns="seq_len",
                values="signed_pct_error",
                aggfunc="mean",
            )
            .reindex(index=gpu_values, columns=seq_values)
        )

        heat = pivot_signed.to_numpy(dtype=float)
        heat_masked = np.ma.masked_invalid(heat)
        image = ax.imshow(
            heat_masked,
            cmap=cmap,
            vmin=-vbound,
            vmax=vbound,
            aspect="auto",
            origin="lower",
        )

        ax.set_title("Model {}".format(model), fontsize=10)
        ax.set_xlabel("seq_len")
        ax.set_ylabel("num_gpus")
        ax.set_xticks(range(len(seq_values)))
        ax.set_xticklabels([str(v) for v in seq_values], rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(gpu_values)))
        ax.set_yticklabels([str(v) for v in gpu_values], fontsize=8)
        ax.set_xticks(np.arange(-0.5, len(seq_values), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(gpu_values), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)

        signed_vals = pivot_signed.to_numpy(dtype=float)
        for y in range(signed_vals.shape[0]):
            for x in range(signed_vals.shape[1]):
                signed = signed_vals[y, x]
                if math.isnan(signed):
                    continue
                text_color = "white" if abs(signed) > (0.45 * vbound) else "black"
                ax.text(
                    x,
                    y,
                    "{:+.1f}%".format(float(signed)),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=text_color,
                )

    for idx in range(n_models, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
        cbar.set_label("Signed Percent Error")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def _default_parity_plot_path(plot_output: Path) -> Path:
    return plot_output.with_name("{}_parity{}".format(plot_output.stem, plot_output.suffix))


def _prepare_parity_df(rows: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows).copy()
    if df.empty:
        return df
    for col in (
        "actual_inferred_total_latency_s",
        "pred_training_time_s",
        "num_gpus",
        "seq_len",
        "global_batch_size",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "model_size" in df.columns:
        df["model_size"] = df["model_size"].astype(str).str.strip().str.lower()
    if "success" in df.columns:
        df = df[df["success"] == True].copy()  # noqa: E712
    df = df[
        df["actual_inferred_total_latency_s"].notna()
        & df["pred_training_time_s"].notna()
        & df["seq_len"].notna()
        & df["global_batch_size"].notna()
        & np.isfinite(df["actual_inferred_total_latency_s"])
        & np.isfinite(df["pred_training_time_s"])
        & np.isfinite(df["seq_len"])
        & np.isfinite(df["global_batch_size"])
        & (df["actual_inferred_total_latency_s"] > 0)
        & (df["pred_training_time_s"] > 0)
        & (df["seq_len"] > 0)
        & (df["global_batch_size"] > 0)
    ].copy()
    total_tokens = df["seq_len"] * df["global_batch_size"]
    df["actual_throughput_tps"] = total_tokens / df["actual_inferred_total_latency_s"]
    df["pred_throughput_tps"] = total_tokens / df["pred_training_time_s"]
    df = df[
        np.isfinite(df["actual_throughput_tps"])
        & np.isfinite(df["pred_throughput_tps"])
        & (df["actual_throughput_tps"] > 0)
        & (df["pred_throughput_tps"] > 0)
    ].copy()
    return df


def _ordered_model_sizes(df: pd.DataFrame) -> List[str]:
    preferred_order = ["760m", "1b", "3b", "7b", "13b", "30b", "70b"]
    present_models = sorted({str(v).strip().lower() for v in df["model_size"].dropna().tolist()})
    ordered_models = [m for m in preferred_order if m in present_models]
    ordered_models.extend([m for m in present_models if m not in ordered_models])
    return ordered_models


def _draw_parity_subplot(
    ax,
    subset_df: pd.DataFrame,
    *,
    subtitle: str,
    color_key: str,
    legend_title: str,
    color_cmap_name: str,
    model_to_marker: Dict[str, str],
) -> bool:
    if subset_df.empty:
        ax.set_title(subtitle)
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return False

    color_values = sorted({int(v) for v in subset_df[color_key].dropna().tolist()})
    if not color_values:
        ax.set_title(subtitle)
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return False

    color_cmap = plt.get_cmap(color_cmap_name, max(1, len(color_values)))
    color_to_value = {value: color_cmap(idx) for idx, value in enumerate(color_values)}

    plotted_any = False
    for model, marker in model_to_marker.items():
        model_df = subset_df[subset_df["model_size"] == model]
        if model_df.empty:
            continue
        for color_value in color_values:
            group = model_df[model_df[color_key] == color_value]
            if group.empty:
                continue
            plotted_any = True
            ax.scatter(
                group["actual_throughput_tps"],
                group["pred_throughput_tps"],
                marker=marker,
                color=color_to_value[color_value],
                alpha=0.9,
                s=60,
                edgecolors="black",
                linewidths=0.35,
            )

    if not plotted_any:
        ax.set_title(subtitle)
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return False

    x_values = subset_df["actual_throughput_tps"].to_numpy(dtype=float)
    y_values = subset_df["pred_throughput_tps"].to_numpy(dtype=float)
    lower = min(float(np.nanmin(x_values)), float(np.nanmin(y_values)))
    upper = max(float(np.nanmax(x_values)), float(np.nanmax(y_values)))
    if not np.isfinite(lower) or not np.isfinite(upper):
        return False
    if upper <= lower:
        upper = lower + 1.0

    lower_lim = lower / 1.2
    upper_lim = upper * 1.2
    ax.plot(
        [lower_lim, upper_lim],
        [lower_lim, upper_lim],
        linestyle="--",
        color="gray",
        linewidth=1.1,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower_lim, upper_lim)
    ax.set_ylim(lower_lim, upper_lim)
    ax.set_xlabel("Actual (tokens/sec)")
    ax.set_ylabel("Predicted (tokens/sec)")
    ax.set_title(subtitle)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color_to_value[value],
            markeredgecolor="black",
            markersize=7,
            label=str(value),
        )
        for value in color_values
    ]
    ax.legend(
        handles=color_handles,
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
        title=legend_title,
        title_fontsize=9,
    )
    return True


def _plot_parity_subsets(
    rows: List[Dict[str, object]],
    title: str,
    parity_plot_output: Path,
) -> Optional[Path]:
    del title  # Title is fixed per user request for this benchmark view.
    df = _prepare_parity_df(rows)
    if df.empty:
        return None

    ordered_models = _ordered_model_sizes(df)
    if not ordered_models:
        return None

    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
    model_to_marker = {model: markers[idx % len(markers)] for idx, model in enumerate(ordered_models)}

    parity_width = 10.0
    parity_height = parity_width * (6.7 / 15.5)
    fig, axes = plt.subplots(1, 2, figsize=(parity_width, parity_height))
    ax_left, ax_right = axes
    fig.suptitle("MPT Training Throughput (H100 80GB BF16)", fontsize=14)

    left_df = df[df["num_gpus"] == 8].copy()
    right_df = df[df["seq_len"] == 2048].copy()

    left_ok = _draw_parity_subplot(
        ax_left,
        left_df,
        subtitle="8 GPUs",
        color_key="seq_len",
        legend_title="SeqLen",
        color_cmap_name="viridis",
        model_to_marker=model_to_marker,
    )
    right_ok = _draw_parity_subplot(
        ax_right,
        right_df,
        subtitle="SeqLen = 2048",
        color_key="num_gpus",
        legend_title="# GPUs",
        color_cmap_name="plasma",
        model_to_marker=model_to_marker,
    )
    if not left_ok and not right_ok:
        plt.close(fig)
        return None

    model_handles = [
        Line2D(
            [0],
            [0],
            marker=model_to_marker[model],
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label="MPT-{}".format(str(model).upper()),
        )
        for model in ordered_models
    ]
    fig.legend(
        handles=model_handles,
        loc="upper center",
        ncol=min(7, max(1, len(model_handles))),
        bbox_to_anchor=(0.5, 0.93),
        fontsize=8,
        framealpha=0.9,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    parity_plot_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(parity_plot_output, dpi=200)
    plt.close(fig)
    return parity_plot_output


def _write_rows_csv(rows: List[Dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fieldnames = [
            "case_index",
            "model_size",
            "seq_len",
            "num_gpus",
            "tp",
            "dp",
            "pp",
            "cp",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "effective_gradient_accumulation_steps",
            "global_batch_size",
            "activation_checkpointing",
            "pred_training_time_s",
            "actual_inferred_total_latency_s",
            "signed_pct_error",
            "abs_pct_error",
            "success",
            "error",
            "raw_output",
        ]
    else:
        fieldnames = list(rows[0].keys())

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(
    *,
    input_csv: Optional[str] = None,
    model_config: Optional[str] = None,
    model_config_dir: Optional[str] = None,
    hardware_config: Optional[str] = None,
    model_size: str = "all",
    output_csv: Optional[str] = None,
    plot_output: Optional[str] = None,
    parity_plot_output: Optional[str] = None,
    enable_plot: bool = True,
    show_progress: bool = False,
    emit_logs: bool = True,
    use_flashattention: bool = True,
    attention_tile_size: Optional[int] = None,
) -> List[Dict[str, object]]:
    input_path = Path(input_csv) if input_csv else DEFAULT_INPUT_CSV
    model_cfg = str(model_config) if model_config else None
    model_cfg_dir = Path(model_config_dir) if model_config_dir else DEFAULT_MODEL_CONFIG_DIR
    hw_cfg = str(hardware_config) if hardware_config else str(DEFAULT_HW_CONFIG)
    output_path = Path(output_csv) if output_csv else DEFAULT_OUTPUT_CSV
    plot_path = Path(plot_output) if plot_output else DEFAULT_PLOT_OUTPUT
    parity_plot_path = Path(parity_plot_output) if parity_plot_output else _default_parity_plot_path(plot_path)

    if use_flashattention and attention_tile_size is not None and int(attention_tile_size) <= 0:
        raise ValueError(
            f"attention_tile_size must be positive when flash attention is enabled (got {attention_tile_size})"
        )

    cases = _load_cases(input_path, model_size=model_size)
    if emit_logs:
        print("Loaded {} case(s) for model_size={} from {}".format(len(cases), model_size, input_path))

    specs, actual_lookup = build_specs(
        cases,
        model_size_filter=model_size,
        explicit_model_config=model_cfg,
        model_config_dir=model_cfg_dir,
        hardware_config_path=hw_cfg,
        use_flashattention=bool(use_flashattention),
        attention_tile_size=None if attention_tile_size is None else int(attention_tile_size),
    )
    base_model_cfg = specs[0].model_config_path if specs and specs[0].model_config_path else str(DEFAULT_MODEL_CONFIG)

    validation_results = run_validation_suite(
        specs,
        base_model_config_path=str(base_model_cfg),
        base_hardware_config_path=hw_cfg,
        result_parser=parse_training_time,
        run_perf_path=str(RUN_PERF),
        show_progress=show_progress,
    )

    rows = compute_rows(validation_results, actual_lookup)
    _write_rows_csv(rows, output_path)
    if emit_logs:
        print("Wrote results CSV: {}".format(output_path))

    if emit_logs:
        for row in rows:
            block = [
                (
                    "\n=== Result (case={case}, model={model}, seq={seq}, gpus={gpus}, "
                    "tp={tp}, dp={dp}, mb={mb}, ga={ga}, gbs={gbs}, ckpt={ckpt}) ==="
                ).format(
                    case=row["case_index"],
                    model=row["model_size"],
                    seq=row["seq_len"],
                    gpus=row["num_gpus"],
                    tp=row["tp"],
                    dp=row["dp"],
                    mb=FIXED_MB,
                    ga=row["gradient_accumulation_steps"],
                    gbs=row["global_batch_size"],
                    ckpt=row["activation_checkpointing"],
                )
            ]
            if row["success"] and not math.isnan(float(row["abs_pct_error"])):
                block.append("  RAPID-LLM train time: {:.6f}s".format(float(row["pred_training_time_s"])))
                block.append(
                    "  Actual (inferred_total_latency_s): {:.6f}s".format(
                        float(row["actual_inferred_total_latency_s"])
                    )
                )
                block.append("  Signed Error: {:+.2f}%".format(float(row["signed_pct_error"])))
                block.append("  Abs Error: {:.2f}%".format(float(row["abs_pct_error"])))
            else:
                block.append("  RAPID-LLM run failed. {}".format((row.get("error") or "").strip()))
            print("\n".join(block))

    if enable_plot:
        model_size_label = "All MPT sizes" if _is_all_models(model_size) else str(model_size)
        heatmap_created = _plot_error_facet_heatmap(
            rows,
            "MosaicML H100 BF16 {}: RAPID-LLM vs inferred_total_latency_s".format(model_size_label),
            plot_path,
        )
        parity_created = _plot_parity_subsets(
            rows,
            "MosaicML H100 BF16 {}: parity (predicted vs actual)".format(model_size_label),
            parity_plot_path,
        )
        if emit_logs and heatmap_created:
            print("Wrote heatmap plot: {}".format(heatmap_created))
        if emit_logs and parity_created:
            print("Wrote parity plot: {}".format(parity_created))

    pct_errors = [float(r["abs_pct_error"]) for r in rows if not math.isnan(float(r["abs_pct_error"]))]
    avg_abs_error = sum(pct_errors) / len(pct_errors) if pct_errors else float("nan")
    if emit_logs:
        print("Average absolute percent error for model_size {}: {:.2f}%".format(model_size, avg_abs_error))

    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate RAPID-LLM training time against MosaicML inferred_total_latency_s."
    )
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_INPUT_CSV),
        help="Input CSV path (default: {}).".format(DEFAULT_INPUT_CSV),
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help=(
            "Explicit model config path. Intended for single-model validation; "
            "cannot be used with --model-size=all."
        ),
    )
    parser.add_argument(
        "--model-config-dir",
        default=str(DEFAULT_MODEL_CONFIG_DIR),
        help="Directory containing per-size model configs (default: {}).".format(DEFAULT_MODEL_CONFIG_DIR),
    )
    parser.add_argument(
        "--hardware-config",
        default=str(DEFAULT_HW_CONFIG),
        help="Hardware config path (default: {}).".format(DEFAULT_HW_CONFIG),
    )
    parser.add_argument(
        "--model-size",
        default="all",
        help="Model size filter from CSV (default: all).",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_CSV),
        help="Output CSV path (default: {}).".format(DEFAULT_OUTPUT_CSV),
    )
    parser.add_argument(
        "--plot-output",
        default=str(DEFAULT_PLOT_OUTPUT),
        help="Output plot path (default: {}).".format(DEFAULT_PLOT_OUTPUT),
    )
    parser.add_argument(
        "--parity-plot-output",
        default=None,
        help=(
            "Output parity scatter path. "
            "Default: <plot-output stem>_parity<suffix> in the same directory."
        ),
    )
    parser.add_argument("--no-plot", dest="enable_plot", action="store_false", help="Disable plot generation.")
    parser.add_argument("--show-progress", action="store_true", help="Show per-run progress.")
    parser.add_argument(
        "--disable-flashattention",
        action="store_true",
        help="Disable flash attention override in generated model overrides.",
    )
    parser.add_argument(
        "--attention-tile-size",
        type=int,
        default=None,
        help=(
            "Attention tile size override used when flash attention is enabled. "
            "Default: leave unchanged from model config."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        input_csv=args.input_csv,
        model_config=args.model_config,
        model_config_dir=args.model_config_dir,
        hardware_config=args.hardware_config,
        model_size=args.model_size,
        output_csv=args.output_csv,
        plot_output=args.plot_output,
        parity_plot_output=args.parity_plot_output,
        enable_plot=args.enable_plot,
        show_progress=args.show_progress,
        use_flashattention=(not bool(args.disable_flashattention)),
        attention_tile_size=args.attention_tile_size,
    )
