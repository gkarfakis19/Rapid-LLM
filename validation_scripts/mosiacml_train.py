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
from matplotlib.ticker import ScalarFormatter

try:
    from .validation_helpers import ValidationSpec, parse_training_time, run_validation_suite
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from validation_scripts.validation_helpers import (  # type: ignore
        ValidationSpec,
        parse_training_time,
        run_validation_suite,
    )
try:
    from .plot_style import (
        IEEE_AXIS_TITLE_SIZE_PT,
        IEEE_DPI,
        IEEE_FONT_SIZE_PT,
        IEEE_HALF_COLUMN_WIDTH_IN,
        IEEE_TITLE_SIZE_PT,
        ieee_rc_params,
    )
except ImportError:
    from plot_style import (  # type: ignore
        IEEE_AXIS_TITLE_SIZE_PT,
        IEEE_DPI,
        IEEE_FONT_SIZE_PT,
        IEEE_HALF_COLUMN_WIDTH_IN,
        IEEE_TITLE_SIZE_PT,
        ieee_rc_params,
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
    PROJECT_ROOT / "validation_scripts" / "validation_configs" / "hardware-config" / "H100_SXM5_80GB.yaml"
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

MAX_ACTUAL_TPS_FOR_PLOTS = 1e6


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


def _normalize_activation_checkpointing_true_mode(value: object) -> str:
    mode = str(value).strip().lower()
    if mode not in {"full", "selective"}:
        raise ValueError(
            "activation_checkpointing_true_mode must be one of: full, selective "
            f"(got {value!r})"
        )
    return mode


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


def _mosaic_hardware_overrides() -> Dict[str, object]:
    return {
        "network": {
            "dimensions": [
                {
                    "id": "dim0",
                    "topology": {
                        "type": "FullyConnected",
                        "bandwidth": "50 GB",
                        "latency": "5e-6",
                        "energy_per_bit": "8e-12",
                        "util": 0.8453452918110231,
                        "optimize_2dmap": False,
                    },
                    "parallelisms": ["dp"],
                },
                {
                    "id": "dim1",
                    "parallelisms": [],
                },
            ]
        }
    }


def _build_spec(
    case: MosaicCase,
    idx: int,
    model_config_path: str,
    hardware_config_path: str,
    *,
    use_flashattention: bool,
    attention_tile_size: Optional[int],
    activation_checkpointing_true_mode: str,
) -> ValidationSpec:
    effective_grad_accum = int(case.gradient_accumulation_steps) # * int(case.micro_batch_size)
    activation_checkpointing_mode = (
        str(activation_checkpointing_true_mode).strip().lower()
        if bool(case.activation_checkpointing)
        else "none"
    )
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
            "activation_checkpointing": activation_checkpointing_mode,
            "full_recomputation": activation_checkpointing_mode == "full",
            "dp_zero_stage": 3,
        },
    }
    hardware_overrides.update(_mosaic_hardware_overrides())

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
            "activation_checkpointing_mode": activation_checkpointing_mode,
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
    activation_checkpointing_true_mode: str = "selective",
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
                activation_checkpointing_true_mode=activation_checkpointing_true_mode,
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
                "activation_checkpointing_mode": str(
                    meta.get("activation_checkpointing_mode", "none")
                ),
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
    rows = _filter_rows_by_actual_throughput(rows)
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
    ncols = 1
    nrows = int(math.ceil(n_models / float(ncols)))

    fig_h = max(3.0, 2.6 * nrows + 0.6)
    with plt.rc_context(ieee_rc_params()):
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(IEEE_HALF_COLUMN_WIDTH_IN, fig_h),
            squeeze=False,
            constrained_layout=True,
        )
        fig.suptitle(title, fontsize=IEEE_TITLE_SIZE_PT)

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

            ax.set_title("Model {}".format(model), fontsize=IEEE_TITLE_SIZE_PT)
            ax.set_xlabel("seq_len", fontsize=IEEE_AXIS_TITLE_SIZE_PT)
            ax.set_ylabel("num_gpus", fontsize=IEEE_AXIS_TITLE_SIZE_PT)
            ax.set_xticks(range(len(seq_values)))
            ax.set_xticklabels([str(v) for v in seq_values], rotation=35, ha="right", fontsize=IEEE_FONT_SIZE_PT)
            ax.set_yticks(range(len(gpu_values)))
            ax.set_yticklabels([str(v) for v in gpu_values], fontsize=IEEE_FONT_SIZE_PT)
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
                        fontsize=IEEE_FONT_SIZE_PT,
                        color=text_color,
                    )

        for idx in range(n_models, nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")

        if image is not None:
            cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
            cbar.set_label("Signed Percent Error", fontsize=IEEE_AXIS_TITLE_SIZE_PT)
            cbar.ax.tick_params(labelsize=IEEE_FONT_SIZE_PT)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=IEEE_DPI)
    plt.close(fig)
    return output


def _default_parity_plot_path(plot_output: Path) -> Path:
    return plot_output.with_name("{}_parity{}".format(plot_output.stem, plot_output.suffix))


def _default_parity_combined_plot_path(parity_plot_output: Path) -> Path:
    return parity_plot_output.with_name("{}_combined{}".format(parity_plot_output.stem, parity_plot_output.suffix))


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _actual_tps_from_row(row: Dict[str, object]) -> float:
    actual_s = _safe_float(row.get("actual_inferred_total_latency_s"))
    seq_len = _safe_float(row.get("seq_len"))
    global_batch_size = _safe_float(row.get("global_batch_size"))
    if (
        not np.isfinite(actual_s)
        or not np.isfinite(seq_len)
        or not np.isfinite(global_batch_size)
        or actual_s <= 0
        or seq_len <= 0
        or global_batch_size <= 0
    ):
        return float("nan")
    return (seq_len * global_batch_size) / actual_s


def _filter_rows_by_actual_throughput(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    filtered: List[Dict[str, object]] = []
    for row in rows:
        actual_tps = _actual_tps_from_row(row)
        if np.isfinite(actual_tps) and actual_tps > MAX_ACTUAL_TPS_FOR_PLOTS:
            continue
        filtered.append(row)
    return filtered


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


def _draw_parity_plot(
    ax,
    df: pd.DataFrame,
    *,
    model_to_marker: Dict[str, str],
) -> bool:
    if df.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return False

    plotted_any = False
    for model, marker in model_to_marker.items():
        model_df = df[df["model_size"] == model]
        if model_df.empty:
            continue
        plotted_any = True
        ax.scatter(
            model_df["actual_throughput_tps"],
            model_df["pred_throughput_tps"],
            marker=marker,
            s=30,
            alpha=0.92,
            c="#1f77b4",
            edgecolors="none",
            linewidths=0.0,
        )

    if not plotted_any:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return False

    x_values = df["actual_throughput_tps"].to_numpy(dtype=float)
    y_values = df["pred_throughput_tps"].to_numpy(dtype=float)
    lower = min(float(np.nanmin(x_values)), float(np.nanmin(y_values)))
    upper = max(float(np.nanmax(x_values)), float(np.nanmax(y_values)))
    if not np.isfinite(lower) or not np.isfinite(upper):
        return False
    if upper <= lower:
        upper = lower + 1.0

    pad = max((upper - lower) * 0.08, upper * 0.03)
    lower_lim = max(0.0, lower - pad)
    upper_lim = upper + pad
    diag_handle = Line2D(
        [lower_lim, upper_lim],
        [lower_lim, upper_lim],
        linestyle="--",
        color="#b04a4a",
        linewidth=1.1,
    )
    ax.add_line(diag_handle)
    ax.set_xlim(lower_lim, upper_lim)
    ax.set_ylim(lower_lim, upper_lim)
    ax.set_xlabel("")
    ax.set_ylabel("")
    sci_x = ScalarFormatter(useMathText=True)
    sci_x.set_scientific(True)
    sci_x.set_powerlimits((0, 0))
    sci_y = ScalarFormatter(useMathText=True)
    sci_y.set_scientific(True)
    sci_y.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(sci_x)
    ax.yaxis.set_major_formatter(sci_y)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
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

    seq_lens = sorted({int(v) for v in df["seq_len"].dropna().tolist()})
    if not seq_lens:
        return None
    subplot_seq_lens = seq_lens[:8]

    parity_width = 7.16
    parity_height = 4.8
    with plt.rc_context(ieee_rc_params()):
        fig, axes = plt.subplots(
            2,
            4,
            figsize=(parity_width, parity_height),
            squeeze=False,
        )
        fig.suptitle(
            "MPT Training Throughput (H100 80GB BF16)",
            fontsize=IEEE_TITLE_SIZE_PT,
            y=0.955,
        )

        axes_flat = axes.ravel()
        for idx, ax in enumerate(axes_flat):
            if idx >= len(subplot_seq_lens):
                ax.axis("off")
                continue
            seq_len = subplot_seq_lens[idx]
            seq_df = df[df["seq_len"] == seq_len]
            ok = _draw_parity_plot(
                ax,
                seq_df,
                model_to_marker=model_to_marker,
            )
            if not ok:
                ax.axis("off")
                continue

            ax.set_title(f"seq_len={seq_len}", fontsize=IEEE_AXIS_TITLE_SIZE_PT)
            row_idx = idx // 4
            col_idx = idx % 4
            if row_idx != 1:
                ax.set_xlabel("")
            if col_idx != 0:
                ax.set_ylabel("")

        model_handles = [
            Line2D(
                [0],
                [0],
                marker=model_to_marker[model],
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.9,
                markersize=6.5,
                label=str(model).upper(),
            )
            for model in model_to_marker
        ]
        fig.legend(
            handles=model_handles,
            loc="upper center",
            ncol=max(1, len(model_handles)),
            bbox_to_anchor=(0.5, 0.915),
            fontsize=IEEE_FONT_SIZE_PT,
            framealpha=0.9,
        )
    fig.tight_layout(rect=(0.02, 0.09, 0.98, 0.91))
    parity_plot_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(parity_plot_output, dpi=IEEE_DPI, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return parity_plot_output


def _plot_parity_combined_seq_len(
    rows: List[Dict[str, object]],
    title: str,
    parity_plot_output: Path,
) -> Optional[Path]:
    del title  # fixed title per request
    filtered_rows = _filter_rows_by_actual_throughput(rows)
    df = _prepare_parity_df(filtered_rows)
    if df.empty:
        return None

    ordered_models = _ordered_model_sizes(df)
    if not ordered_models:
        return None

    seq_lens = sorted({int(v) for v in df["seq_len"].dropna().tolist()})
    if not seq_lens:
        return None

    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
    model_to_marker = {model: markers[idx % len(markers)] for idx, model in enumerate(ordered_models)}
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, len(seq_lens))))
    seq_to_color = {seq_len: colors[idx] for idx, seq_len in enumerate(seq_lens)}
    with plt.rc_context(ieee_rc_params()):
        fig, ax = plt.subplots(
            figsize=(IEEE_HALF_COLUMN_WIDTH_IN * 1.2, 3.0),
        )
        for model in ordered_models:
            for seq_len in seq_lens:
                sub = df[(df["model_size"] == model) & (df["seq_len"] == seq_len)]
                if sub.empty:
                    continue
                ax.scatter(
                    sub["actual_throughput_tps"],
                    sub["pred_throughput_tps"],
                    marker=model_to_marker[model],
                    s=26,
                    alpha=0.9,
                    c=[seq_to_color[seq_len]],
                    edgecolors="none",
                    linewidths=0.0,
                )

        x_values = df["actual_throughput_tps"].to_numpy(dtype=float)
        y_values = df["pred_throughput_tps"].to_numpy(dtype=float)
        lower = min(float(np.nanmin(x_values)), float(np.nanmin(y_values)))
        upper = max(float(np.nanmax(x_values)), float(np.nanmax(y_values)))
        if not np.isfinite(lower) or not np.isfinite(upper):
            plt.close(fig)
            return None
        if upper <= lower:
            upper = lower + 1.0

        pad = max((upper - lower) * 0.08, upper * 0.03)
        lower_lim = max(0.0, lower - pad)
        upper_lim = upper + pad
        ax.plot(
            [lower_lim, upper_lim],
            [lower_lim, upper_lim],
            linestyle="--",
            color="#b04a4a",
            linewidth=1.1,
        )
        ax.set_xlim(lower_lim, upper_lim)
        ax.set_ylim(lower_lim, upper_lim)
        ax.set_title("MPT Training Throughput (tokens/s) w/ H100 BF16", fontsize=IEEE_TITLE_SIZE_PT)
        ax.set_xlabel("Actual", fontsize=IEEE_AXIS_TITLE_SIZE_PT)
        ax.set_ylabel("Predicted", fontsize=IEEE_AXIS_TITLE_SIZE_PT)
        sci_x = ScalarFormatter(useMathText=True)
        sci_x.set_scientific(True)
        sci_x.set_powerlimits((0, 0))
        sci_y = ScalarFormatter(useMathText=True)
        sci_y.set_scientific(True)
        sci_y.set_powerlimits((0, 0))
        ax.xaxis.set_major_formatter(sci_x)
        ax.yaxis.set_major_formatter(sci_y)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

        def _model_param_label(model_size: str) -> str:
            text = str(model_size).strip().lower()
            if text.endswith("m") and len(text) > 1:
                return f"{text[:-1]}M"
            if text.endswith("b") and len(text) > 1:
                return f"{text[:-1]}B"
            return text.upper()

        model_handles = [
            Line2D(
                [0],
                [0],
                marker=model_to_marker[model],
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.9,
                markersize=6.5,
                label=_model_param_label(model),
            )
            for model in ordered_models
        ]
        seq_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=seq_to_color[seq_len],
                markeredgecolor="none",
                markersize=6.5,
                label=str(seq_len),
            )
            for seq_len in seq_lens
        ]

        model_legend = ax.legend(
            handles=model_handles,
            loc="upper left",
            title="MPT Size",
            fontsize=IEEE_FONT_SIZE_PT,
            title_fontsize=IEEE_FONT_SIZE_PT,
            framealpha=0.9,
        )
        ax.add_artist(model_legend)
        ax.legend(
            handles=seq_handles,
            loc="lower right",
            title="SeqLen",
            fontsize=IEEE_FONT_SIZE_PT,
            title_fontsize=IEEE_FONT_SIZE_PT,
            framealpha=0.9,
        )
        fig.tight_layout()

    parity_plot_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(parity_plot_output, dpi=IEEE_DPI, bbox_inches="tight", pad_inches=0.04)
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
    activation_checkpointing_true_mode: str = "selective",
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
    ckpt_true_mode = _normalize_activation_checkpointing_true_mode(activation_checkpointing_true_mode)

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
        activation_checkpointing_true_mode=ckpt_true_mode,
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
                    "tp={tp}, dp={dp}, mb={mb}, ga={ga}, gbs={gbs}, ckpt={ckpt}, ckpt_mode={ckpt_mode}) ==="
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
                    ckpt_mode=row.get("activation_checkpointing_mode", "none"),
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
        parity_combined_path = _default_parity_combined_plot_path(parity_plot_path)
        parity_combined_created = _plot_parity_combined_seq_len(
            rows,
            "MosaicML H100 BF16 {}: parity (all seq_len)".format(model_size_label),
            parity_combined_path,
        )
        if emit_logs and heatmap_created:
            print("Wrote heatmap plot: {}".format(heatmap_created))
        if emit_logs and parity_created:
            print("Wrote parity plot: {}".format(parity_created))
        if emit_logs and parity_combined_created:
            print("Wrote parity combined plot: {}".format(parity_combined_created))

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
    parser.add_argument(
        "--activation-checkpointing-true-mode",
        choices=("full", "selective"),
        default="selective",
        help=(
            "Interpretation of CSV activation_checkpointing=True. "
            "'full' maps to full recomputation; 'selective' maps to selective checkpointing."
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
        activation_checkpointing_true_mode=args.activation_checkpointing_true_mode,
    )
