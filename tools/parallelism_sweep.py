#!/usr/bin/env python3
# Copyright 2026 NanoCad lab, UCLA
# https://nanocad.ee.ucla.edu/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Parallelism sweep utility for RAPID-LLM LLM configurations.

Update the global configuration section below to point at the desired hardware
and model config files and to tailor the parallelism search space. The tool
enumerates every combination, filters those whose total GPU count falls inside
the configured bounds, evaluates runtime with RAPID-LLM, and plots a scatter
chart of accuracy (by default, 1 / runtime) versus GPU count with horizontal
jitter to avoid overlap.
"""

from __future__ import print_function

import argparse
import ast
import copy
import csv
import itertools
import json
import math
import os
import random
import traceback
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Iterable, List, Tuple, Optional
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from train_timing import TimeCalculationLLM
from inference_timing import TimeCalculationLLMInference
from llm_util import process_gemm_shapes

import seaborn as sns
import numpy as np
import pandas as pd
import math
import numpy as np
from matplotlib.patches import Polygon
plt.rcParams.update({"font.size": 13})

# -----------------------------------------------------------------------------
# Global configuration knobs (no CLI)
# -----------------------------------------------------------------------------

# Paths to the baseline configuration files
HARDWARE_CONFIG_PATH = "validation_scripts/validation_configs/hardware-config/A100_SXM4_80GB_case-A.yaml"
HARDWARE_CONFIG_DIR = "validation_scripts/validation_configs/hardware-config"
HW_CONFIGS = [  # Optional: list of hardware configs to sweep; if empty, uses HARDWARE_CONFIG_PATH.
    "A100_SXM4_80GB_case-A.yaml",
    "A100_SXM4_80GB_case-B.yaml",
]
HW_LABELS = [  # Optional: labels aligned with HW_CONFIGS.
    "Case A",
    "Case B",
]
MODEL_CONFIG_PATH = "validation_scripts/validation_configs/model-config/Llama3.1-70B_2d_train.yaml"

# Parallelism values to sweep (dense grid). Edit to suit your search space.
# Values map to the training parallelism block (parallelism.train.dp, etc.).
PARALLELISM_SWEEP = {
    "tp": [2**i for i in range(0, 7)],
    "cp": [2**i for i in range(0, 7)],
    "dp": [2**i for i in range(0, 11)],
    "pp": [2**i for i in range(0, 6)],
}

# Optional knobs that still live inside the parallelism section but do not
# affect GPU counts. Leave empty if you do not want to explore them.
OTHER_PARALLELISM_OPTIONS = {
    "tp_sp": [True],
}

# GPU count filter: only evaluate combinations whose TP*CP*DP*PP fall inside
# this inclusive range.
GPU_COUNT_MIN = 128
GPU_COUNT_MAX = 2048
TP_CP_PRODUCT_MIN = 1  # Optional: set to int to filter tp*cp below this threshold.
TP_CP_PRODUCT_MAX = 512  # Optional: set to int to filter tp*cp above this threshold.

# When True, discard configurations whose tp*cp product is not a square power of two.
ENFORCE_SQUARE_TP_CP = False

# Select the y-axis metric for the scatter plot and report. Accepted values:
#   "runtime"     -> plot raw runtime in seconds (lower is better)
#   "performance" -> plot 1 / runtime (higher is better)
PLOT_METRIC = "runtime"

# Random seed for reproducible jitter in the scatter plot.
PLOT_JITTER_SEED = 1234
# Maximum absolute horizontal jitter (in GPU units).
PLOT_JITTER_WIDTH = 0.175
# Swarm/point styling for better visibility.
PLOT_POINT_SIZE = 5.0
PLOT_POINT_EDGE = 0.2
# Drop points slower than this multiple of the per-GPU best runtime (plot only).
PLOT_RUNTIME_RATIO_CUTOFF = 10.0

# Default output artefacts
PLOT_OUTPUT_PATH = "tools/parallelism_sweep.png"
PLOT_MFU_OUTPUT_PATH = "tools/parallelism_sweep_mfu.png"
REPORT_OUTPUT_PATH = "tools/parallelism_sweep.tsv"
BEST_RUNTIME_PLOT_PATH = "tools/parallelism_best_runtimes.png"
BEST_RUNTIME_PER_GPU_DIR = "tools/parallelism_best_runtimes_per_gpu"
BEST_RUNTIME_PER_GPU_COMBINED_PATH = "tools/parallelism_best_runtimes_per_gpu_combined.png"
BEST_SPEEDUP_PER_GPU_COMBINED_PATH = "tools/parallelism_speedup_per_gpu_combined.png"
RUNTIME_CACHE_PATH = "tools/parallelism_sweep_cache.csv"
ERROR_LOG_PATH = ""
PLOT_TITLE = "Parallelism options vs runtime"

# AstraSim cache handling within RAPID-LLM (mirrors run_perf default options).
ASTRA_CACHE_MODE = "NO_CACHE"  # Options: NO_CACHE, CACHE_READONLY, CACHE_READWRITE

# Plotting behaviour toggles
MEM_AWARE_FILTER = False  # When True, skip memory-violating configurations in plots.
EVALUATE_MEMORY_EXCEEDED = True  # When True, still compute runtime even if memory limits are exceeded (may crash).

# Maximum number of parallel worker processes (set <= available CPUs - 1). Set to 1 to disable multiprocessing.
MAX_WORKERS = 127
# When True, use a thread pool instead of a process pool (more stable, avoids worker crashes at the cost of GIL contention).
USE_THREADPOOL = False


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def add_rgb_ternary_legend(ax,
                           corner_labels=("R = log2(tp+cp)",
                                          "G = log2(pp)",
                                          "B = log2(dp)"),
                           gamma=0.85,
                           inset_xywh=(0.72, 0.58, 0.24, 0.24),  # Axes fraction: x, y, w, h
                           n=120):
    """
    Draw a tiny triangular legend that shows how the RGB mix works.
    - Uses barycentric blending inside an equilateral triangle.
    - 'gamma' should match your color gamma used in the scatter.
    - inset_xywh is (x0,y0,w,h) in axes-relative coords.
    """
    # Inset axis
    iax = ax.inset_axes(inset_xywh, transform=ax.transAxes)
    iax.set_aspect("equal")
    iax.set_axis_off()

    # Equilateral triangle vertices (R, G, B corners)
    A = np.array([0.0, 0.0])                        # R corner
    B = np.array([1.0, 0.0])                        # G corner
    C = np.array([0.5, math.sqrt(3)/2.0])           # B corner

    # Sample interior with barycentric coords r,g,b (r+g+b=1)
    xs, ys, cols = [], [], []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            r = i / n
            g = j / n
            b = 1.0 - r - g
            p = r * A + g * B + b * C
            # apply same gamma tweak used in your plot colors
            rr = r ** gamma
            gg = g ** gamma
            bb = b ** gamma
            xs.append(p[0]); ys.append(p[1]); cols.append((rr, gg, bb))

    # iax.scatter(xs, ys, s=2, c=cols, edgecolors="none")
    iax.add_patch(Polygon([A, B, C], facecolor="none", edgecolor="black", linewidth=1))

    # Corner labels
    iax.text(A[0]-0.06, A[1]-0.04, corner_labels[0], ha="right", va="top", fontsize=8)
    iax.text(B[0]+0.06, B[1]-0.04, corner_labels[1], ha="left",  va="top", fontsize=8)
    iax.text(C[0],      C[1]+0.06, corner_labels[2], ha="center", va="bottom", fontsize=8)

    # Optional tick marks (25/50/75%) along edges (comment out if you want it cleaner)
    for t in (0.25, 0.5, 0.75):
        # Edge AB (vary r vs g, b=0)
        p = (1-t) * A + t * B
        iax.plot([p[0]], [p[1]], marker="|", color="black", ms=6)
        # Edge BC (vary g vs b, r=0)
        p = (1-t) * B + t * C
        iax.plot([p[0]], [p[1]], marker="_", color="black", ms=6)
        # Edge CA (vary b vs r, g=0)
        p = (1-t) * C + t * A
        iax.plot([p[0]], [p[1]], marker="|", color="black", ms=6)

    # Tiny caption (how channels were normalized)
    iax.text(0.5, -0.18, "Each channel min–max normalized\nthen gamma-adjusted",
             ha="center", va="top", fontsize=7, transform=iax.transAxes)


def read_yaml(path):
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def _parse_csv_list(arg: str) -> List[str]:
    if not arg:
        return []
    paths = []
    for part in arg.split(","):
        part = part.strip()
        if part:
            paths.append(part)
    return paths


def _resolve_repo_relative_path(path_like: str) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    repo_root = Path(__file__).resolve().parents[1]
    repo_candidate = repo_root / p
    if repo_candidate.exists():
        return repo_candidate
    return p.resolve()


def _normalize_device_type_key(value: str) -> str:
    text = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text


def _to_positive_float(value: object, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Expected numeric value for '{field_name}', got {value!r}.")
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"Expected '{field_name}' > 0, got {value!r}.")
    return parsed


def _load_device_derates(path_like: str, device_type: str) -> Dict[str, float]:
    config_path = _resolve_repo_relative_path(path_like)
    raw = read_yaml(str(config_path))
    if not isinstance(raw, dict):
        raise ValueError(f"Derate config must be a YAML mapping at root: {config_path}")

    shared_raw = raw.get("shared")
    if not isinstance(shared_raw, dict):
        raise ValueError(f"Derate config missing 'shared' mapping: {config_path}")
    kernel_launch_overhead_s = _to_positive_float(
        shared_raw.get("kernel_launch_overhead_s"),
        "shared.kernel_launch_overhead_s",
    )

    device_types_raw = raw.get("device_types")
    if not isinstance(device_types_raw, dict):
        raise ValueError(f"Derate config missing 'device_types' mapping: {config_path}")

    target_key = _normalize_device_type_key(device_type)
    selected = None
    for raw_key, entry in device_types_raw.items():
        if _normalize_device_type_key(str(raw_key)) == target_key:
            selected = entry
            break
    if not isinstance(selected, dict):
        raise ValueError(
            f"Derate config has no device_types entry for '{device_type}' in {config_path}"
        )

    return {
        "kernel_launch_overhead_s": kernel_launch_overhead_s,
        "dram_util": _to_positive_float(selected.get("dram_util"), f"device_types.{device_type}.dram_util"),
        "network_util": _to_positive_float(
            selected.get("network_util"), f"device_types.{device_type}.network_util"
        ),
        "compute_util": _to_positive_float(
            selected.get("compute_util"), f"device_types.{device_type}.compute_util"
        ),
    }


def _apply_device_derates(hw_dict: Dict[str, object], derates: Dict[str, float]) -> None:
    sw_param = hw_dict.setdefault("sw_param", {})
    if not isinstance(sw_param, dict):
        raise ValueError("Expected sw_param to be a mapping in hardware config.")
    sw_param["kernel_launch_overhead"] = float(derates["kernel_launch_overhead_s"])

    tech_param = hw_dict.setdefault("tech_param", {})
    if not isinstance(tech_param, dict):
        raise ValueError("Expected tech_param to be a mapping in hardware config.")
    dram_cfg = tech_param.setdefault("DRAM", {})
    if not isinstance(dram_cfg, dict):
        raise ValueError("Expected tech_param.DRAM to be a mapping in hardware config.")
    core_cfg = tech_param.setdefault("core", {})
    if not isinstance(core_cfg, dict):
        raise ValueError("Expected tech_param.core to be a mapping in hardware config.")
    dram_cfg["util"] = float(derates["dram_util"])
    core_cfg["util"] = float(derates["compute_util"])

    network = hw_dict.get("network")
    if not isinstance(network, dict):
        raise ValueError("Expected network to be a mapping in hardware config.")
    dimensions = network.get("dimensions")
    if not isinstance(dimensions, list):
        raise ValueError("Expected network.dimensions to be a list in hardware config.")

    dim0_found = False
    for dim in dimensions:
        if not isinstance(dim, dict):
            continue
        if str(dim.get("id", "")).strip() != "dim0":
            continue
        topology = dim.setdefault("topology", {})
        if not isinstance(topology, dict):
            raise ValueError("Expected network.dimensions[id=dim0].topology to be a mapping.")
        topology["util"] = float(derates["network_util"])
        dim0_found = True
        break
    if not dim0_found:
        raise ValueError("Expected a network dimension with id='dim0' in hardware config.")


def _tagged_output_path(base_path: str, tag: str) -> str:
    """Insert a tag before the file extension in base_path."""
    p = Path(base_path)
    tag = tag.replace(os.sep, "_")
    new_name = f"{p.stem}_{tag}{p.suffix}"
    return str(p.with_name(new_name))


def _root_output_path(base_path: str, output_root: str) -> str:
    if not output_root:
        return base_path
    return str(Path(output_root) / Path(base_path).name)


def _combine_tags(*parts: str) -> str:
    cleaned = []
    for part in parts:
        text = str(part or "").strip()
        if text:
            cleaned.append(text.replace(os.sep, "_"))
    return "_".join(cleaned)


def _resolve_hw_paths(config_list: List[str]) -> List[str]:
    resolved: List[str] = []
    for cfg in config_list:
        p = Path(cfg)
        if not p.is_absolute() and not p.exists():
            p = Path(HARDWARE_CONFIG_DIR) / p
        resolved.append(str(p))
    return resolved


def _hardware_label(index: int, hw_path: str) -> str:
    try:
        if 0 <= index < len(HW_LABELS):
            return str(HW_LABELS[index])
    except Exception:
        pass
    return Path(hw_path).stem


def _model_config_with_overrides(base_model_path: str, hw_config_path: str) -> Tuple[str, str]:
    """
    Return a path to a model config that includes hardware-specific overrides.
    Case A scales attention tile size with the larger effective L2 only when
    FlashAttention is enabled in the base model config.
    """
    hw_name = Path(hw_config_path).name
    cache_id = str(Path(base_model_path).resolve())
    if hw_name != "A100_SXM4_80GB_case-A.yaml":
        return base_model_path, cache_id

    def _parse_size_to_bytes(value) -> int:
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value).strip()
        number, unit = text.split(None, 1)
        scale = {
            "B": 1,
            "KB": 1024,
            "MB": 1024 ** 2,
            "GB": 1024 ** 3,
            "TB": 1024 ** 4,
        }
        return int(float(number) * scale[unit.upper()])

    model_dict = read_yaml(base_model_path)
    model_param = model_dict.setdefault("model_param", {})
    attention = model_param.setdefault("attention", {})
    if not bool(attention.get("use_flashattention", False)):
        return base_model_path, cache_id
    base_hw_path = (
        Path(__file__).resolve().parents[1]
        / "validation_scripts"
        / "validation_configs"
        / "hardware-config"
        / "A100_SXM4_80GB_base.yaml"
    )
    base_hw = read_yaml(str(base_hw_path))
    case_hw = read_yaml(hw_config_path)
    base_l2 = _parse_size_to_bytes(base_hw["tech_param"]["SRAM-L2"]["size"])
    case_l2 = _parse_size_to_bytes(case_hw["tech_param"]["SRAM-L2"]["size"])
    base_tile = int(attention.get("attention_tile_size", 128) or 128)
    scaled_tile = max(1, int(base_tile * (float(case_l2) / float(base_l2))))
    attention["attention_tile_size"] = scaled_tile

    tmp_handle = tempfile.NamedTemporaryFile("w", suffix="_model_override.yaml", delete=False)
    try:
        yaml.safe_dump(model_dict, tmp_handle, default_flow_style=False, sort_keys=False)
        tmp_handle.flush()
        cache_id = f"{cache_id}|caseA_flashattention_tile={scaled_tile}"
        return tmp_handle.name, cache_id
    finally:
        try:
            tmp_handle.close()
        except Exception:
            pass


def determine_model_mode(model_config_path):
    model_dict = read_yaml(model_config_path)
    model_param = model_dict.get("model_param") or {}
    mode = model_param.get("mode")
    if not mode:
        raise ValueError("model_param.mode must be defined in {}".format(model_config_path))
    return mode


def _cache_key(hw_path: str, model_id: str, settings: Dict[str, object]) -> Tuple[str, str, str]:
    hw_id = str(Path(hw_path).resolve())
    settings_json = json.dumps(settings, sort_keys=True)
    return (hw_id, model_id, settings_json)


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _append_error_log(record: Dict[str, object], path: str = "") -> None:
    active_path = str(path or ERROR_LOG_PATH or "").strip()
    if not active_path:
        return
    log_path = Path(active_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_safe(record)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _load_runtime_cache(path: str) -> Dict[Tuple[str, str, str], Dict[str, object]]:
    cache: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    p = Path(path)
    if not p.exists():
        return cache
    try:
        with p.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    settings = json.loads(row.get("parallelism", "{}"))
                except Exception:
                    continue
                hw_id = row.get("hardware_config")
                model_id = row.get("model_config")
                if not hw_id or not model_id:
                    continue
                key = _cache_key(hw_id, model_id, settings)
                entry = {
                    "hardware_config": hw_id,
                    "model_config": model_id,
                    "parallelism": settings,
                    "num_gpus": int(row.get("num_gpus", 0)),
                    "runtime": float(row.get("runtime", "nan")),
                    "prefill_time": float(row.get("prefill_time", "nan")),
                    "decode_time": float(row.get("decode_time", "nan")),
                    "performance": float(row.get("performance", "nan")),
                    "total_flops": float(row.get("total_flops", "nan")),
                    "achieved_flops": float(row.get("achieved_flops", "nan")),
                    "peak_flops": float(row.get("peak_flops", "nan")),
                    "mfu": float(row.get("mfu", "nan")),
                    "memory_exceeded": str(row.get("memory_exceeded", "")).strip().lower() == "true",
                    "memory_violation_gb": float(row.get("memory_violation_gb", "0.0") or 0.0),
                }
                cache[key] = entry
    except Exception as exc:
        print(f"Warning: failed to load runtime cache from {path}: {exc}", file=sys.stderr)
    return cache


def _save_runtime_cache(cache: Dict[Tuple[str, str, str], Dict[str, object]], path: str) -> None:
    if not cache:
        return
    fieldnames = [
        "hardware_config",
        "model_config",
        "parallelism",
        "num_gpus",
        "runtime",
        "prefill_time",
        "decode_time",
        "performance",
        "total_flops",
        "achieved_flops",
        "peak_flops",
        "mfu",
        "memory_exceeded",
        "memory_violation_gb",
    ]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with p.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for entry in cache.values():
                row = dict(entry)
                row["parallelism"] = json.dumps(entry.get("parallelism", {}), sort_keys=True)
                writer.writerow(row)
    except Exception as exc:
        print(f"Warning: failed to save runtime cache to {path}: {exc}", file=sys.stderr)


def _cache_entry_has_inference_breakdown(entry: Dict[str, object]) -> bool:
    for key in ("prefill_time", "decode_time"):
        value = entry.get(key)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return False
        if math.isnan(parsed):
            return False
    return True


def cartesian_product(option_map):
    """Yield dictionaries for every combination inside option_map."""
    if not option_map:
        yield {}
        return
    keys = sorted(option_map.keys())
    value_lists = [option_map[key] for key in keys]
    for values in itertools.product(*value_lists):
        yield dict(zip(keys, values))


def build_parallelism_settings(flat_settings: Dict[str, object]) -> Dict[str, object]:
    parallelism: Dict[str, object] = {}
    for key in ("tp", "cp", "pp", "mb", "tp_sp"):
        if key in flat_settings:
            parallelism[key] = flat_settings[key]
    train_block = {
        "dp": int(flat_settings.get("dp", 1) or 1),
        "ep": int(flat_settings.get("ep", 1) or 1),
        "tp_ep": bool(flat_settings.get("tp_ep", True)),
    }
    parallelism["train"] = train_block
    parallelism["inference"] = {
        "replica_count": int(flat_settings.get("replica_count", 1) or 1),
        "moe_dp": int(flat_settings.get("moe_dp", 1) or 1),
    }
    return parallelism


def _parallelism_snapshot(parallelism: Dict[str, object]) -> Dict[str, int]:
    train_block = parallelism.get("train")
    if not isinstance(train_block, dict) or "dp" not in train_block:
        raise ValueError("parallelism.train.dp must be set for sweep entries.")
    return {
        "tp": int(parallelism.get("tp", 1) or 1),
        "cp": int(parallelism.get("cp", 1) or 1),
        "dp": int(train_block["dp"]),
        "pp": int(parallelism.get("pp", 1) or 1),
    }


def total_gpu_count(parallel_cfg):
    values = _parallelism_snapshot(parallel_cfg)
    total = 1
    for axis in ("tp", "cp", "dp", "pp"):
        total *= max(1, int(values.get(axis, 1)))
    return total


def tp_cp_product_is_power_of_two_square(tp_value, cp_value):
    """Return True when tp*cp is a square whose root is also a power of two."""
    if tp_value == 1 and cp_value == 1:
        return False # special case
    try:
        tp_int = int(tp_value)
        cp_int = int(cp_value)
    except (TypeError, ValueError):
        return False
    if tp_int <= 0 or cp_int <= 0:
        return False
    product = tp_int * cp_int
    root = math.isqrt(product)
    if root * root != product:
        return False
    return (root & (root - 1)) == 0


def make_temp_hw_config(base_hw_dict, parallel_settings, hw_mutator=None):
    """Return (parsed HW config, YAML string) for the given override."""
    updated = copy.deepcopy(base_hw_dict)
    parallel_block = updated.setdefault("parallelism", {})
    for key in ("tp", "cp", "pp", "mb", "tp_sp"):
        if key in parallel_settings:
            parallel_block[key] = parallel_settings[key]

    train_settings = parallel_settings.get("train")
    if not isinstance(train_settings, dict):
        raise ValueError("parallelism.train is required for sweep entries")
    train_block = parallel_block.setdefault("train", {})
    train_block["dp"] = int(train_settings["dp"])
    train_block["ep"] = int(train_settings.get("ep", 1))
    train_block["tp_ep"] = bool(train_settings.get("tp_ep", True))

    inference_settings = parallel_settings.get("inference")
    if not isinstance(inference_settings, dict):
        raise ValueError("parallelism.inference is required for sweep entries")
    inference_block = parallel_block.setdefault("inference", {})
    inference_block["replica_count"] = int(inference_settings["replica_count"])
    inference_block["moe_dp"] = int(inference_settings["moe_dp"])

    if hw_mutator:
        hw_mutator(updated)
    try:
        debug_yaml = yaml.safe_dump(updated, default_flow_style=False)
    except Exception:
        debug_yaml = None

    tmp_file = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    try:
        yaml.safe_dump(updated, tmp_file, default_flow_style=False, sort_keys=False)
        tmp_file.flush()
        tmp_file.close()
        hw_config = config.parse_config(tmp_file.name, config_type="hardware")
        return hw_config, debug_yaml
    finally:
        try:
            tmp_file.close()
        except Exception:
            pass
        try:
            os.unlink(tmp_file.name)
        except Exception:
            pass

def _rgb_from_parallelism(entries, gamma: float = 0.85):
    """
    Build per-point RGBA colors using:
      R = log2(tp+cp), G = log2(pp), B = log2(dp)
    Each channel is min-max normalized over the dataset, then gamma-adjusted.
    """
    snapshots = [_parallelism_snapshot(e["parallelism"]) for e in entries]
    tps = np.array([max(1, int(snap["tp"])) for snap in snapshots], dtype=float)
    cps = np.array([max(1, int(snap["cp"])) for snap in snapshots], dtype=float)
    dps = np.array([max(1, int(snap["dp"])) for snap in snapshots], dtype=float)
    pps = np.array([max(1, int(snap["pp"])) for snap in snapshots], dtype=float)

    r_raw = np.log2(tps + cps)
    g_raw = np.log2(pps)
    b_raw = np.log2(dps)

    def _norm(x):
        xmin, xmax = float(np.min(x)), float(np.max(x))
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
            return np.zeros_like(x)
        y = (x - xmin) / (xmax - xmin)
        if gamma and gamma != 1.0:
            y = np.power(y, gamma)
        return y

    r = _norm(r_raw)
    g = _norm(g_raw)
    b = _norm(b_raw)

    # RGBA with slight transparency to reduce overdraw
    return np.stack([r, g, b, np.full_like(r, 0.9)], axis=1)



def gpu_peak_flops(hw_config) -> float:
    """Return the theoretical peak FLOPs/s for a single GPU."""
    core = hw_config.tech_config.core
    bundles = core.num_bundles or 1
    per_cycle_flops = float(core.nominal_flop_rate_per_mcu)
    mcu_per_bundle = float(core.num_mcu_per_bundle)
    frequency = float(core.operating_frequency or core.nominal_frequency or 0.0)
    util = float(getattr(core, "util", 1.0) or 1.0)
    peak = per_cycle_flops * mcu_per_bundle * float(bundles)
    if frequency > 0:
        peak *= frequency
    return peak * util if peak > 0 else 0.0


def compute_total_flops(calculator: TimeCalculationLLM) -> float:
    """Estimate total FLOPs executed for one global batch (forward + backward)."""
    global_batch = getattr(calculator, "batch_size", None)
    if global_batch is None or global_batch <= 0:
        global_batch = calculator._effective_transformer_batch()
    if not global_batch or global_batch <= 0:
        return float("nan")

    vocab_size = calculator.vocab_size
    hidden_dim = calculator.hidden_dim
    seq_len = calculator.seq_len
    num_heads = calculator.num_heads
    kv_heads = calculator.kv_heads
    intermediate_size = calculator.intermediate_size

    gemm_shapes = process_gemm_shapes(
        calculator,
        global_batch,
        seq_len,
        hidden_dim,
        num_heads,
        kv_heads,
        intermediate_size,
        vocab_size,
    )

    def _gemm_flops(shape) -> float:
        if shape is None:
            return 0.0
        try:
            dims = list(shape)
        except TypeError:
            return 0.0
        if len(dims) == 4:
            b, m, k, n = dims
            return 2.0 * float(b) * float(m) * float(k) * float(n)
        if len(dims) == 3:
            m, k, n = dims
            return 2.0 * float(m) * float(k) * float(n)
        return 0.0

    def _forward_backward(shape) -> float:
        forward = _gemm_flops(shape)
        if forward <= 0.0:
            return 0.0
        backward = 2.0 * forward
        return forward + backward

    per_layer_flops = 0.0
    per_layer_flops += _forward_backward(gemm_shapes.get("qkv_proj"))
    per_layer_flops += _forward_backward(gemm_shapes.get("attention_score"))
    per_layer_flops += _forward_backward(gemm_shapes.get("attention_output"))
    per_layer_flops += _forward_backward(gemm_shapes.get("output_proj"))
    per_layer_flops += _forward_backward(gemm_shapes.get("ffn1"))
    per_layer_flops += _forward_backward(gemm_shapes.get("ffn2"))

    total_flops = per_layer_flops * float(calculator.num_layers)

    total_flops += _forward_backward(gemm_shapes.get("linear"))

    # Embedding forward/backward is comparatively small; keep placeholder for future refinement.

    return float(total_flops)


def evaluate_parallelism(hw_dict, model_config_obj, mode, parallel_settings, hw_mutator=None):
    hw_config, debug_yaml = make_temp_hw_config(hw_dict, parallel_settings, hw_mutator=hw_mutator)
    # if debug_yaml:
    #     print("=== DEBUG HW CONFIG ===")
    #     print(debug_yaml)
    #     try:
    #         with open("debug.yaml", "w") as debug_handle:
    #             debug_handle.write(debug_yaml)
    #     except Exception as exc:
            # print(f"Warning: failed to write debug.yaml: {exc}", file=sys.stderr)
    temp_dir = tempfile.mkdtemp(prefix="parallelism_sweep_")
    try:
        run_type = str(getattr(getattr(model_config_obj, "model_config", None), "run_type", "training")).lower()
        if run_type == "inference":
            calculator = TimeCalculationLLMInference(hw_config, model_config_obj, mode, output_dir=temp_dir)
            with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
                inference_timing = calculator.calc_total_inference_time()
                runtime = float(inference_timing["total_inference_time"])
                prefill_time = float(inference_timing.get("prefill_time", float("nan")))
                decode_time = float(inference_timing.get("decode_time", float("nan")))
            mem_exceeded = bool(getattr(calculator, "memory_capacity_exceeded", False))
            mem_violation = float(getattr(calculator, "memory_capacity_violation_gb", 0.0) or 0.0)
            if mem_exceeded and not EVALUATE_MEMORY_EXCEEDED:
                return {
                    "runtime": float("nan"),
                    "prefill_time": prefill_time,
                    "decode_time": decode_time,
                    "performance": float("nan"),
                    "total_flops": float("nan"),
                    "peak_flops": gpu_peak_flops(hw_config),
                    "mfu": float("nan"),
                    "achieved_flops": float("nan"),
                    "memory_exceeded": True,
                    "memory_violation_gb": mem_violation,
                    "hw_yaml": debug_yaml,
                }
            performance = (1.0 / runtime) if runtime > 0.0 else float("nan")
            return {
                "runtime": runtime,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "performance": performance,
                "total_flops": float("nan"),
                "peak_flops": gpu_peak_flops(hw_config),
                "mfu": float("nan"),
                "achieved_flops": float("nan"),
                "memory_exceeded": mem_exceeded,
                "memory_violation_gb": mem_violation,
                "hw_yaml": debug_yaml,
            }

        calculator = TimeCalculationLLM(hw_config, model_config_obj, mode, output_dir=temp_dir)
        mem_exceeded = bool(getattr(calculator, "memory_capacity_exceeded", False))
        mem_violation = float(getattr(calculator, "memory_capacity_violation_gb", 0.0) or 0.0)
        if mem_exceeded and not EVALUATE_MEMORY_EXCEEDED:
            total_flops = compute_total_flops(calculator)
            peak_flops = gpu_peak_flops(hw_config)
            num_gpus = total_gpu_count(parallel_settings)
            return {
                "runtime": float("nan"),
                "prefill_time": float("nan"),
                "decode_time": float("nan"),
                "performance": float("nan"),
                "total_flops": total_flops,
                "peak_flops": peak_flops,
                "mfu": float("nan"),
                "achieved_flops": float("nan"),
                "memory_exceeded": True,
                "memory_violation_gb": mem_violation,
                "hw_yaml": debug_yaml,
            }
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            runtime = calculator.calc_time_llm()
            total_flops = compute_total_flops(calculator)
        performance = (1.0 / runtime) if runtime and runtime > 0.0 else float("nan")
        peak_flops = gpu_peak_flops(hw_config)
        num_gpus = total_gpu_count(parallel_settings)
        achieved_flops = (total_flops / runtime) if runtime > 0 else float("nan")
        denom = peak_flops * num_gpus if peak_flops and num_gpus else float("nan")
        mfu = (achieved_flops / denom) if denom and denom > 0 else float("nan")
        return {
            "runtime": runtime,
            "prefill_time": float("nan"),
            "decode_time": float("nan"),
            "performance": performance,
            "total_flops": total_flops,
            "peak_flops": peak_flops,
            "mfu": mfu,
            "achieved_flops": achieved_flops,
            "memory_exceeded": getattr(calculator, "memory_capacity_exceeded", False),
            "memory_violation_gb": getattr(calculator, "memory_capacity_violation_gb", 0.0),
            "hw_yaml": debug_yaml,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def set_astrasim_cache_mode(mode_str):
    mapping = {
        "NO_CACHE": "NO_CACHE",
        "CACHE_READONLY": "CACHE_READONLY",
        "CACHE_READWRITE": "CACHE_READWRITE",
    }
    env_value = mapping.get(str(mode_str).strip().upper(), "NO_CACHE")
    os.environ["RAPID_ASTRA_CACHE_MODE"] = env_value


def write_report(results, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    header = [
        "num_gpus",
        "runtime_s",
        "prefill_time_s",
        "decode_time_s",
        "performance_1_over_s",
        "total_flops",
        "achieved_flops_per_s",
        "peak_flops_per_gpu",
        "mfu",
        "memory_exceeded",
        "memory_violation_gb",
        "parallelism",
    ]
    with open(path, "w") as handle:
        handle.write("\t".join(header) + "\n")
        for entry in results:
            row = [
                str(entry["num_gpus"]),
                "{:.6f}".format(entry["runtime"]),
                "{:.6f}".format(entry.get("prefill_time", float("nan"))),
                "{:.6f}".format(entry.get("decode_time", float("nan"))),
                (
                    "{:.6f}".format(entry["performance"])
                    if entry["performance"] == entry["performance"]
                    else "nan"
                ),
                "{:.6e}".format(entry["total_flops"]),
                "{:.6e}".format(entry["achieved_flops"]) if entry["achieved_flops"] == entry["achieved_flops"] else "nan",
                "{:.6e}".format(entry["peak_flops"]),
                "{:.6f}".format(entry["mfu"]) if entry["mfu"] == entry["mfu"] else "nan",
                str(entry["memory_exceeded"]),
                "{:.6f}".format(entry["memory_violation_gb"]),
                repr(entry["parallelism"]),
            ]
            handle.write("\t".join(row) + "\n")


def _parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_results_from_report(path: str) -> List[Dict[str, object]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No report found at {path}")

    results: List[Dict[str, object]] = []
    with open(path, "r") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            return results
        for row in reader:
            try:
                parallelism = ast.literal_eval(row["parallelism"])
            except (SyntaxError, ValueError) as exc:
                print(
                    f"Warning: failed to parse parallelism '{row.get('parallelism', '')}': {exc}",
                    file=sys.stderr,
                )
                continue
            entry = {
                "num_gpus": int(row["num_gpus"]),
                "runtime": _parse_float(row.get("runtime_s")),
                "prefill_time": _parse_float(row.get("prefill_time_s")),
                "decode_time": _parse_float(row.get("decode_time_s")),
                "performance": _parse_float(row.get("performance_1_over_s")),
                "total_flops": _parse_float(row.get("total_flops")),
                "achieved_flops": _parse_float(row.get("achieved_flops_per_s")),
                "peak_flops": _parse_float(row.get("peak_flops_per_gpu")),
                "mfu": _parse_float(row.get("mfu")),
                "memory_exceeded": str(row.get("memory_exceeded", "")).strip().lower() == "true",
                "memory_violation_gb": _parse_float(row.get("memory_violation_gb")),
                "parallelism": parallelism,
            }
            results.append(entry)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="RAPID-LLM parallelism sweep utility.")
    parser.add_argument(
        "--plot_only",
        "--plot-only",
        dest="plot_only",
        action="store_true",
        help="Skip evaluation and only regenerate plots from the existing TSV report.",
    )
    parser.add_argument(
        "--enforce-square-tp-cp",
        action="store_true",
        help="Require tp*cp to be a square power of two when enumerating configurations.",
    )
    parser.add_argument(
        "--hardware-configs",
        type=str,
        default="",
        help="Comma-separated hardware config paths to evaluate (overrides HARDWARE_CONFIG_PATH/HW_CONFIGS).",
    )
    parser.add_argument(
        "--hardware-labels",
        type=str,
        default="",
        help="Comma-separated labels aligned with --hardware-configs (for combined plots).",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="",
        help="Path to model config YAML (overrides MODEL_CONFIG_PATH).",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default="",
        help="Suffix added to generated report/plot filenames to keep runs separate.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Optional directory to place generated TSV/PNG/cache artifacts for this run.",
    )
    parser.add_argument(
        "--gpu-count-min",
        type=int,
        default=None,
        help="Minimum total GPU count to evaluate (overrides GPU_COUNT_MIN).",
    )
    parser.add_argument(
        "--gpu-count-max",
        type=int,
        default=None,
        help="Maximum total GPU count to evaluate (overrides GPU_COUNT_MAX).",
    )
    parser.add_argument(
        "--derate-config",
        type=str,
        default="",
        help="Path to derate YAML (same schema as validation_scripts/validation_configs/harness_derates.yaml).",
    )
    parser.add_argument(
        "--derate-device-type",
        type=str,
        default="",
        help="Device type entry to load from --derate-config (e.g., A100_SXM4).",
    )
    return parser.parse_args()


def jitter_positions(gpu_counts, jitter_width):
    jittered = []
    for count in gpu_counts:
        offset = random.uniform(1 - jitter_width, 1 + jitter_width)
        jittered.append(max(count * offset, 1e-3))
    return jittered


# def plot_results_legacy(results, output_path):
#     if not results:
#         print("No successful configurations to plot.", file=sys.stderr)
#         return

#     random.seed(PLOT_JITTER_SEED)
#     xs = jitter_positions([item["num_gpus"] for item in results], PLOT_JITTER_WIDTH)
#     metric_key = "runtime" if PLOT_METRIC.lower() == "runtime" else "performance"
#     ys = [item[metric_key] for item in results]

#     plt.figure(figsize=(10, 6))
#     plt.scatter(xs, ys, s=60, alpha=0.7, edgecolors="none")

#     best = min(results, key=lambda item: item["runtime"])
#     best_x = best["num_gpus"]
#     best_y = best[metric_key]
#     plt.scatter([best_x], [best_y], s=180, marker="*", c="red", label="Best runtime")

#     plt.xlabel("Number of GPUs")
#     if metric_key == "runtime":
#         plt.ylabel("Runtime (s)")
#     else:
#         plt.ylabel("Performance (1 / s)")
#     plt.xscale("log")
#     if metric_key == "runtime":
#         plt.yscale("log")
#     plt.title("Parallelism sweep")
#     plt.grid(alpha=0.3)
#     xticks = sorted(set(item["num_gpus"] for item in results))
#     plt.xticks(xticks, [str(int(x)) for x in xticks])
#     plt.legend(loc="best")
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=200)
#     plt.close()
#     print("Saved scatter plot to {}".format(output_path))

def _gpu_exp(n: int) -> float:
    """Return log2(num_gpus). Assumes powers of two; will return floats otherwise."""
    return math.log2(float(n))


def _filter_by_runtime_ratio(
    entries: List[Dict[str, object]],
    ratio: float,
) -> List[Dict[str, object]]:
    if ratio <= 0:
        return list(entries)
    best_by_gpu: Dict[int, float] = {}
    for item in entries:
        runtime_val = float(item.get("runtime", float("nan")))
        if not math.isfinite(runtime_val) or runtime_val <= 0:
            continue
        ng = int(item.get("num_gpus", 0))
        best = best_by_gpu.get(ng)
        if best is None or runtime_val < best:
            best_by_gpu[ng] = runtime_val
    if not best_by_gpu:
        return list(entries)
    filtered: List[Dict[str, object]] = []
    for item in entries:
        ng = int(item.get("num_gpus", 0))
        best = best_by_gpu.get(ng)
        runtime_val = float(item.get("runtime", float("nan")))
        if best is None:
            filtered.append(item)
            continue
        if math.isfinite(runtime_val) and runtime_val <= ratio * best:
            filtered.append(item)
    return filtered


def plot_results(results, output_path):
    if not results:
        print("No successful configurations to plot.", file=sys.stderr)
        return

    plot_entries = results
    if MEM_AWARE_FILTER:
        plot_entries = [item for item in results if not item.get("memory_exceeded", False)]
        if not plot_entries:
            print("Warning: all configurations violate memory limits; skipping plot.", file=sys.stderr)
            return
    plot_entries = _filter_by_runtime_ratio(plot_entries, PLOT_RUNTIME_RATIO_CUTOFF)
    if not plot_entries:
        print("Warning: no configurations remain after runtime ratio filtering; skipping plot.", file=sys.stderr)
        return

    metric_key = "runtime" if PLOT_METRIC.lower() == "runtime" else "performance"

    finite_values = [
        float(item[metric_key])
        for item in plot_entries
        if math.isfinite(float(item[metric_key]))
    ]
    fallback_metric = max(finite_values) * 1.05 if finite_values else 1.0

    # Build tidy frame (include GPU exponent)
    rows = []
    for i, item in enumerate(plot_entries):
        p = item["parallelism"]
        snap = _parallelism_snapshot(p)
        ng = int(item["num_gpus"])
        metric_val = float(item[metric_key])
        if not math.isfinite(metric_val):
            metric_val = fallback_metric
        rows.append({
            "row_id": i,
            "num_gpus": ng,
            "gpu_exp": _gpu_exp(ng),
            "tp": snap["tp"],
            "cp": snap["cp"],
            "dp": snap["dp"],
            "pp": snap["pp"],
            "memory_exceeded": bool(item.get("memory_exceeded", False)),
            metric_key: metric_val,
        })
    df = pd.DataFrame(rows)

    # Order categories by exponent (gives even spacing like log2)
    order = sorted(df["gpu_exp"].unique())
    df["gpu_exp_cat"] = pd.Categorical(df["gpu_exp"], categories=order, ordered=True)

    # --- Per-point RGB colors: R=log2(tp+cp), G=log2(pp), B=log2(dp) ---
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

    # Palette: unique color per row via hue="row_id"
    palette = {}
    for i, rid in enumerate(df["row_id"]):
        if bool(df.iloc[i]["memory_exceeded"]):
            palette[rid] = (0.6, 0.6, 0.6, 0.7)
        else:
            palette[rid] = (float(r[i]), float(g[i]), float(b[i]), float(a[i]))

    plt.figure(figsize=(10, 6))
    ax = sns.swarmplot(
        data=df,
        x="gpu_exp_cat",
        y=metric_key,
        hue="row_id",
        palette=palette,   # per-point RGBA
        size=PLOT_POINT_SIZE,
        linewidth=PLOT_POINT_EDGE,
        edgecolor="black",
        dodge=False,
        legend=False
    )
    has_points = False
    for coll in ax.collections:
        get_offsets = getattr(coll, "get_offsets", None)
        offsets = get_offsets() if callable(get_offsets) else None
        if offsets is not None and len(offsets):
            has_points = True
            break
    if not has_points:
        ax.clear()
        rng = np.random.RandomState(PLOT_JITTER_SEED)
        x_base = df["gpu_exp_cat"].cat.codes.to_numpy(dtype=float)
        jitter = rng.uniform(-0.28, 0.28, size=len(df))
        xs = x_base + jitter
        ys = df[metric_key].to_numpy(dtype=float)
        colors = [palette[rid] for rid in df["row_id"]]
        ax.scatter(
            xs,
            ys,
            s=PLOT_POINT_SIZE ** 2,
            c=colors,
            edgecolor="black",
            linewidth=PLOT_POINT_EDGE,
        )
    # add_rgb_ternary_legend(
    #     ax,
    #     corner_labels=("R = log2(tp+cp)", "G = log2(pp)", "B = log2(dp)"),
    #     gamma=0.85,                      # match your _rgb_from_parallelism gamma
    #     inset_xywh=(0.70, 0.50, 0.26, 0.26)  # tweak position/size to taste
    # )


    # Best runtime star — place at the matching exponent category index
    best_candidates = [
        it for it in plot_entries
        if not it.get("memory_exceeded") and math.isfinite(it.get("runtime", float("nan")))
    ]
    # if best_candidates:
    #     best = min(best_candidates, key=lambda it: it["runtime"])
    #     best_exp = _gpu_exp(int(best["num_gpus"]))
    #     best_idx = order.index(best_exp)
    #     best_y = float(best[metric_key])
    #     ax.scatter([best_idx], [best_y], s=180, marker="*", c="red", zorder=5, label="Best runtime")

    # Labels & scales
    ax.set_xlabel("Number of GPUs")
    if metric_key == "runtime":
        ax.set_ylabel("Runtime (s)")
        min_metric = float(df[metric_key].min())
        if min_metric > 0:
            ax.set_yscale("log")
        else:
            print("Warning: non-positive runtime values; using linear scale.", file=sys.stderr)
    else:
        ax.set_ylabel("Performance (1 / s)")

    # Tick labels show the REAL GPU count (2**exp) so axis *looks* log2
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([str(int(round(2 ** e))) if float(e).is_integer() else f"2^{e:.2f}"
                        for e in order])

    ax.set_title(PLOT_TITLE)
    ax.grid(alpha=0.3, axis="y")

    # Reminder of color encoding
    ax.text(0.99, 0.01,
            "Color: R=log2(tp+cp), G=log2(pp), B=log2(dp)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, alpha=0.8)

    # Per-GPU-count best runtime markers (exclude memory violations)
    best_by_gpu: Dict[int, Tuple[float, float]] = {}
    for item in plot_entries:
        if item.get("memory_exceeded"):
            continue
        runtime_val = float(item.get("runtime", float("nan")))
        if not math.isfinite(runtime_val):
            continue
        ng = int(item.get("num_gpus", 0))
        gpu_exp = _gpu_exp(ng)
        best = best_by_gpu.get(ng)
        if best is None or runtime_val < best[1]:
            best_by_gpu[ng] = (gpu_exp, runtime_val)
    if best_by_gpu:
        xs = []
        ys = []
        for ng, (gpu_exp, runtime_val) in best_by_gpu.items():
            if gpu_exp in order:
                xs.append(order.index(gpu_exp))
                ys.append(runtime_val)
        if xs and ys:
            ax.scatter(xs, ys, s=100, marker="*", c="white", edgecolor="black", zorder=6, label="Best runtime")
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved swarm plot to {output_path}")


def plot_best_runtimes(best_entries: List[Dict[str, object]], output_path: str, label_order: List[str]) -> None:
    if not best_entries:
        print("No best runtimes to plot.", file=sys.stderr)
        return
    labels: List[str] = []
    runtimes: List[float] = []
    ordered_labels = [lbl for lbl in label_order if any(str(entry["label"]) == str(lbl) for entry in best_entries)]
    for lbl in ordered_labels:
        match = next((entry for entry in best_entries if str(entry["label"]) == str(lbl)), None)
        if match is None:
            continue
        labels.append(str(lbl))
        runtimes.append(float(match["runtime"]))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, runtimes, color="#1f78b4")
    plt.ylabel("Best runtime (s)")
    plt.title("Best runtime per hardware config")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    for bar, value in zip(bars, runtimes):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved best-runtime bar chart to {output_path}")


def plot_best_runtimes_per_gpu(best_by_gpu: Dict[int, List[Dict[str, object]]], output_dir: str, label_order: List[str]) -> None:
    if not best_by_gpu:
        print("No per-GPU best runtimes to plot.", file=sys.stderr)
        return
    os.makedirs(output_dir, exist_ok=True)
    for gpu_count, entries in sorted(best_by_gpu.items()):
        if not entries:
            continue
        labels: List[str] = []
        runtimes: List[float] = []
        ordered_labels = [lbl for lbl in label_order if any(str(ent["label"]) == str(lbl) for ent in entries)]
        for lbl in ordered_labels:
            match = next((ent for ent in entries if str(ent["label"]) == str(lbl)), None)
            if match is None:
                continue
            labels.append(str(lbl))
            runtimes.append(float(match["runtime"]))

        plt.figure(figsize=(8, 5))
        bars = plt.bar(labels, runtimes, color="#1f78b4")
        plt.ylabel("Best runtime (s)")
        plt.title(f"Best runtime per hardware config (GPU count = {gpu_count})")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        for bar, value in zip(bars, runtimes):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=8)

        path = os.path.join(output_dir, f"best_runtime_{gpu_count}gpus.png")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"Saved per-GPU best-runtime bar chart to {path}")


def plot_best_runtimes_per_gpu_combined(best_by_gpu: Dict[int, List[Dict[str, object]]], output_path: str, label_order: List[str]) -> None:
    """
    Combined view: x-axis GPU count categories, each showing bars for the best runtime
    achieved by every hardware config at that GPU count.
    """
    if not best_by_gpu:
        print("No per-GPU best runtimes to plot.", file=sys.stderr)
        return

    gpu_counts = sorted(best_by_gpu.keys())
    labels = [lbl for lbl in label_order if any(str(entry["label"]) == str(lbl) for entries in best_by_gpu.values() for entry in entries)]
    if not labels:
        print("No hardware labels found for combined per-GPU plot.", file=sys.stderr)
        return

    group_width = 0.7
    bar_width = group_width / max(1, len(labels))

    plt.figure(figsize=(8, 5))
    palette = sns.color_palette("tab10", n_colors=max(len(labels), 3))

    for li, label in enumerate(labels):
        xs = []
        ys = []
        for idx, gpu_count in enumerate(gpu_counts):
            entries = best_by_gpu.get(gpu_count, [])
            runtime = next((float(ent["runtime"]) for ent in entries if ent.get("label") == label), float("nan"))
            if not math.isfinite(runtime):
                continue
            x_pos = idx - (group_width / 2) + bar_width * li + bar_width / 2
            xs.append(x_pos)
            ys.append(runtime)
        if xs:
            plt.bar(xs, ys, width=bar_width, label=label, color=palette[li % len(palette)])

    plt.xticks(range(len(gpu_counts)), [str(g) for g in gpu_counts])
    plt.xlabel("GPU count")
    plt.ylabel("Best runtime (s)")
    plt.title("Best runtime per GPU count across hardware configs")
    plt.legend(title="Hardware config", loc="best")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved combined per-GPU best-runtime chart to {output_path}")


def plot_speedup_per_gpu_combined(
    best_by_gpu: Dict[int, List[Dict[str, object]]],
    output_path: str,
    label_order: List[str],
    base_label: str = "Base",
    omit_gpu_counts: List[int] = None,
) -> None:
    """
    Combined view: bars show speedup vs base hardware (runtime_base / runtime_other) per GPU count.
    """
    if omit_gpu_counts is None:
        omit_gpu_counts = []
    if not best_by_gpu:
        print("No per-GPU best runtimes to plot for speedup.", file=sys.stderr)
        return

    gpu_counts = [g for g in sorted(best_by_gpu.keys()) if g not in omit_gpu_counts]
    labels = [lbl for lbl in label_order if any(str(entry["label"]) == str(lbl) for entries in best_by_gpu.values() for entry in entries)]
    if not labels:
        print("No hardware labels found for speedup plot.", file=sys.stderr)
        return

    group_width = 0.85
    bar_width = group_width / max(1, len(labels))

    plt.figure(figsize=(10, 5))
    palette = sns.color_palette("tab10", n_colors=max(len(labels), 3))

    for li, label in enumerate(labels):
        xs = []
        ys = []
        runtimes_for_labels = []
        for idx, gpu_count in enumerate(gpu_counts):
            entries = best_by_gpu.get(gpu_count, [])
            base_entry = next((ent for ent in entries if str(ent.get("label")) == str(base_label)), None)
            if base_entry is None or not math.isfinite(float(base_entry.get("runtime", float("nan")))):
                continue
            base_runtime = float(base_entry["runtime"])
            runtime = next((float(ent["runtime"]) for ent in entries if ent.get("label") == label), float("nan"))
            if not math.isfinite(runtime):
                continue
            speedup = base_runtime / runtime if runtime > 0 else float("nan")
            if not math.isfinite(speedup):
                continue
            x_pos = idx - (group_width / 2) + bar_width * li + bar_width / 2
            xs.append(x_pos)
            ys.append(speedup)
            runtimes_for_labels.append(runtime)
        if xs:
            bars = plt.bar(xs, ys, width=bar_width, label=label, color=palette[li % len(palette)])
            for bar, runtime_val in zip(bars, runtimes_for_labels):
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    f"{int(round(runtime_val))}s",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.xticks(range(len(gpu_counts)), [str(g) for g in gpu_counts])
    plt.xlabel("GPU count")
    plt.ylabel("Speedup vs Base")
    plt.title("Optimal training speedup per GPU count")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved combined per-GPU speedup chart to {output_path}")

# def plot_results_categorical(results, output_path):
#     if not results:
#         print("No successful configurations to plot.", file=sys.stderr)
#         return

#     metric_key = "runtime" if PLOT_METRIC.lower() == "runtime" else "performance"

#     # Build a tidy DataFrame for seaborn (categorical x keeps each GPU count together)
#     df_rows = []
#     for item in results:
#         p = item["parallelism"]
#         df_rows.append({
#             "num_gpus": int(item["num_gpus"]),
#             metric_key: float(item[metric_key]),
#             "tp": int(p.get("tp", 1)),
#             "cp": int(p.get("cp", 1)),
#             "dp": int(p.get("dp", 1)),
#             "pp": int(p.get("pp", 1)),
#         })
#     df = pd.DataFrame(df_rows)

#     # Order GPU categories left->right by numeric value
#     order = sorted(df["num_gpus"].unique())
#     df["num_gpus_cat"] = pd.Categorical(df["num_gpus"], categories=order, ordered=True)

#     # --- Beeswarm placement ---
#     plt.figure(figsize=(10, 6))
#     ax = sns.swarmplot(
#         data=df,
#         x="num_gpus_cat",
#         y=metric_key,
#         size=5,
#         color="k",         # temporary (we'll recolor with our RGBs after layout)
#         linewidth=0,
#         alpha=0.0          # invisible; we only want the positions it computes
#     )

#     # Grab the laid-out positions, remove seaborn's collection, and redraw with our colors
#     if not ax.collections:
#         print("Warning: swarmplot produced no collections.", file=sys.stderr)
#         return
#     coll = ax.collections[0]
#     offsets = coll.get_offsets()        # Nx2 array (x_index, y_value) in data coords (x is categorical index)
#     coll.remove()

#     # Compute per-point RGBA colors from parallelism
#     colors = _rgb_from_parallelism(results, gamma=0.85)

#     # Plot the recolored points at the computed positions
#     ax.scatter(
#         offsets[:, 0], offsets[:, 1],
#         s=60,
#         c=colors,
#         edgecolors="none",
#         zorder=3
#     )

#     # Global best marker (use category index for x)
#     best = min(results, key=lambda item: item["runtime"])
#     best_x_idx = order.index(int(best["num_gpus"]))  # categorical index
#     best_y = float(best[metric_key])
#     ax.scatter([best_x_idx], [best_y], s=180, marker="*", c="red", zorder=5, label="Best runtime")

#     # Axes/labels
#     ax.set_xlabel("Number of GPUs")
#     if metric_key == "runtime":
#         ax.set_ylabel("Runtime (s)")
#         ax.set_yscale("log")
#     else:
#         ax.set_ylabel("Performance (1 / s)")

#     # Pretty ticks for categorical axis
#     ax.set_xticks(range(len(order)))
#     ax.set_xticklabels([str(x) for x in order])

#     ax.set_title("Parallelism sweep (beeswarm with log-RGB coloring)")
#     ax.grid(alpha=0.3, axis="y")
#     ax.legend(loc="best")

#     # Small caption to recall color encoding
#     ax.text(0.99, 0.01,
#             "Color channels: R=log2(tp+cp), G=log2(pp), B=log2(dp)",
#             transform=ax.transAxes, ha="right", va="bottom", fontsize=9, alpha=0.8)

#     plt.tight_layout()
#     plt.savefig(output_path, dpi=200)
#     plt.close()
#     print(f"Saved swarm plot to {output_path}")


# def plot_mfu(results, output_path):
#     if not results:
#         return
#     random.seed(PLOT_JITTER_SEED)
#     xs = jitter_positions([item["num_gpus"] for item in results], PLOT_JITTER_WIDTH)
#     ys = [item["mfu"] for item in results]
#     plt.figure(figsize=(10, 6))
#     plt.scatter(xs, ys, s=60, alpha=0.7, edgecolors="none")
#     valid = [item for item in results if item["mfu"] == item["mfu"]]
#     if valid:
#         best = max(valid, key=lambda item: item["mfu"])
#         plt.scatter([best["num_gpus"]], [best["mfu"]], s=180, marker="*", c="green", label="Max MFU")
#     plt.xlabel("Number of GPUs")
#     plt.ylabel("MFU")
#     plt.xscale("log")
#     plt.ylim(0.0, 1.05)
#     plt.title("Parallelism sweep - MFU")
#     plt.grid(alpha=0.3)
#     xticks = sorted(set(item["num_gpus"] for item in results))
#     plt.xticks(xticks, [str(int(x)) for x in xticks])
#     if valid:
#         plt.legend(loc="best")
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=200)
#     plt.close()
#     print("Saved MFU scatter plot to {}".format(output_path))


_GLOBAL_MODEL_CONFIG = None
_GLOBAL_MODE = None
_GLOBAL_HW_DICT = None


def _worker_init(hw_dict, model_config_path, mode):
    global _GLOBAL_MODEL_CONFIG, _GLOBAL_MODE, _GLOBAL_HW_DICT
    _GLOBAL_HW_DICT = hw_dict
    _GLOBAL_MODE = mode
    _GLOBAL_MODEL_CONFIG = config.parse_config(model_config_path, config_type=mode)


def _worker_task(parallel_items: Tuple[Tuple[str, object], ...]):
    flat_settings = {k: v for k, v in parallel_items}
    parallel_settings = build_parallelism_settings(flat_settings)
    try:
        metrics = evaluate_parallelism(
            _GLOBAL_HW_DICT,
            _GLOBAL_MODEL_CONFIG,
            _GLOBAL_MODE,
            parallel_settings,
        )
        return {
            "status": "ok",
            "parallelism": parallel_settings,
            "metrics": metrics,
        }
    except BaseException as exc:
        return {
            "status": "error",
            "parallelism": parallel_settings,
            "error": f"{exc.__class__.__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _run_task_in_isolated_process(
    parallel_items: Tuple[Tuple[str, object], ...],
    hw_dict,
    model_config_path: str,
    mode: str,
):
    with ProcessPoolExecutor(
        max_workers=1,
        initializer=_worker_init,
        initargs=(hw_dict, model_config_path, mode),
    ) as executor:
        future = executor.submit(_worker_task, parallel_items)
        return future.result()


def _build_tasks(
    gpu_choices: Iterable[Dict[str, int]],
    other_choices: Iterable[Dict[str, object]],
) -> List[Tuple[Tuple[str, object], ...]]:
    tasks: List[Tuple[Tuple[str, object], ...]] = []
    for gpu_choice in gpu_choices:
        for other_choice in other_choices:
            settings: Dict[str, object] = {}
            settings.update(gpu_choice)
            settings.update(other_choice)
            settings["mb"] = settings.get("pp", 1)
            tasks.append(tuple(sorted(settings.items())))
    return tasks


def main():
    args = parse_args()
    set_astrasim_cache_mode(ASTRA_CACHE_MODE)

    hw_config_cli = _parse_csv_list(args.hardware_configs)
    hw_config_paths = _resolve_hw_paths(hw_config_cli or HW_CONFIGS or [HARDWARE_CONFIG_PATH])
    label_override = _parse_csv_list(args.hardware_labels)
    if label_override and len(label_override) != len(hw_config_paths):
        raise ValueError(
            f"--hardware-labels count ({len(label_override)}) must match hardware configs count "
            f"({len(hw_config_paths)})."
        )

    derate_config_path = str(args.derate_config or "").strip()
    derate_device_type = str(args.derate_device_type or "").strip()
    if bool(derate_config_path) != bool(derate_device_type):
        raise ValueError("Both --derate-config and --derate-device-type must be provided together.")
    active_derates: Dict[str, float] = {}
    derate_cache_tag = ""
    if derate_config_path and derate_device_type:
        active_derates = _load_device_derates(derate_config_path, derate_device_type)
        resolved_derate_path = _resolve_repo_relative_path(derate_config_path)
        device_type_key = _normalize_device_type_key(derate_device_type)
        derate_cache_tag = (
            f"|derate:{resolved_derate_path.resolve()}:{device_type_key}"
            f":klo={active_derates['kernel_launch_overhead_s']:.12g}"
            f":dram={active_derates['dram_util']:.12g}"
            f":net={active_derates['network_util']:.12g}"
            f":core={active_derates['compute_util']:.12g}"
        )
        print(
            "Applying derates from {} for {}: kernel_launch_overhead={}, dram.util={}, "
            "network.dim0.util={}, core.util={}".format(
                resolved_derate_path,
                device_type_key,
                active_derates["kernel_launch_overhead_s"],
                active_derates["dram_util"],
                active_derates["network_util"],
                active_derates["compute_util"],
            )
        )

    enforce_square = ENFORCE_SQUARE_TP_CP or args.enforce_square_tp_cp
    active_model_config_path = str(args.model_config or "").strip() or MODEL_CONFIG_PATH
    active_gpu_count_min = GPU_COUNT_MIN if args.gpu_count_min is None else int(args.gpu_count_min)
    active_gpu_count_max = GPU_COUNT_MAX if args.gpu_count_max is None else int(args.gpu_count_max)
    if active_gpu_count_min > active_gpu_count_max:
        raise ValueError(
            f"--gpu-count-min ({active_gpu_count_min}) must be <= --gpu-count-max ({active_gpu_count_max})."
        )
    output_tag = str(args.output_tag or "").strip()
    output_root = str(args.output_root or "").strip()
    rooted_plot_output_path = _root_output_path(PLOT_OUTPUT_PATH, output_root)
    rooted_report_output_path = _root_output_path(REPORT_OUTPUT_PATH, output_root)
    rooted_best_runtime_plot_path = _root_output_path(BEST_RUNTIME_PLOT_PATH, output_root)
    rooted_best_runtime_per_gpu_dir = _root_output_path(BEST_RUNTIME_PER_GPU_DIR, output_root)
    rooted_best_runtime_per_gpu_combined_path = _root_output_path(
        BEST_RUNTIME_PER_GPU_COMBINED_PATH,
        output_root,
    )
    rooted_best_speedup_per_gpu_combined_path = _root_output_path(
        BEST_SPEEDUP_PER_GPU_COMBINED_PATH,
        output_root,
    )
    rooted_runtime_cache_path = _root_output_path(RUNTIME_CACHE_PATH, output_root)
    if output_root:
        Path(output_root).mkdir(parents=True, exist_ok=True)
    active_plot_output_path = (
        _tagged_output_path(rooted_plot_output_path, output_tag)
        if output_tag
        else rooted_plot_output_path
    )
    active_report_output_path = (
        _tagged_output_path(rooted_report_output_path, output_tag)
        if output_tag
        else rooted_report_output_path
    )
    active_best_runtime_plot_path = (
        _tagged_output_path(rooted_best_runtime_plot_path, output_tag)
        if output_tag
        else rooted_best_runtime_plot_path
    )
    active_best_runtime_per_gpu_dir = (
        _tagged_output_path(rooted_best_runtime_per_gpu_dir, output_tag)
        if output_tag
        else rooted_best_runtime_per_gpu_dir
    )
    active_best_runtime_per_gpu_combined_path = (
        _tagged_output_path(rooted_best_runtime_per_gpu_combined_path, output_tag)
        if output_tag
        else rooted_best_runtime_per_gpu_combined_path
    )
    active_best_speedup_per_gpu_combined_path = (
        _tagged_output_path(rooted_best_speedup_per_gpu_combined_path, output_tag)
        if output_tag
        else rooted_best_speedup_per_gpu_combined_path
    )
    active_error_log_path = str(ERROR_LOG_PATH or "").strip()
    if active_error_log_path and not args.plot_only:
        error_log = Path(active_error_log_path)
        error_log.parent.mkdir(parents=True, exist_ok=True)
        if error_log.exists():
            error_log.unlink()
    mode = determine_model_mode(active_model_config_path)
    runtime_cache = _load_runtime_cache(rooted_runtime_cache_path)
    cache_dirty = False

    results: List[Dict[str, object]] = []
    label_order = label_override or [_hardware_label(i, path) for i, path in enumerate(hw_config_paths)]

    if args.plot_only:
        for hw_path in hw_config_paths:
            tag = Path(hw_path).stem if len(hw_config_paths) > 1 else None
            report_tag = _combine_tags(output_tag, tag)
            report_path = (
                _tagged_output_path(rooted_report_output_path, report_tag)
                if report_tag
                else rooted_report_output_path
            )
            try:
                results = load_results_from_report(report_path)
            except FileNotFoundError as exc:
                print(str(exc))
                continue

            if not results:
                print(f"No entries found in {report_path}; nothing to plot.")
                continue

            best = min(results, key=lambda item: item["runtime"])
            print(f"[{hw_path}] Loaded {len(results)} configuration(s) from {report_path}")
            print("\nBest configuration (lowest runtime):")
            print("  Parallelism: {}".format(best["parallelism"]))
            print("  GPUs: {}".format(best["num_gpus"]))
            print("  Runtime: {:.4f} s".format(best["runtime"]))
            print("  Performance (1/s): {:.4f}".format(best["performance"]))
            print("  Total FLOPs: {:.3e}".format(best["total_flops"]))
            print("  MFU: {:.3f}".format(best["mfu"]))
            if best["memory_exceeded"]:
                print("  Memory capacity exceeded by {:.3f} GB".format(best["memory_violation_gb"]))
            plot_tag = _combine_tags(output_tag, tag)
            plot_path = _tagged_output_path(PLOT_OUTPUT_PATH, plot_tag) if plot_tag else PLOT_OUTPUT_PATH
            plot_results(results, plot_path)
        return

    gpu_axes = list(PARALLELISM_SWEEP.keys())
    other_axes = list(OTHER_PARALLELISM_OPTIONS.keys())
    gpu_combos = list(cartesian_product(PARALLELISM_SWEEP))
    other_combos = list(cartesian_product(OTHER_PARALLELISM_OPTIONS))
    task_items = _build_tasks(gpu_combos, other_combos)
    print("Enumerating {} parallelism combinations: {}".format(len(task_items), ", ".join(gpu_axes)))

    filtered_tasks: List[Tuple[Tuple[str, object], ...]] = []
    skipped_out_of_range = 0
    skipped_square_constraint = 0
    for items in task_items:
        settings = build_parallelism_settings(dict(items))
        num_gpus = total_gpu_count(settings)
        if enforce_square and not tp_cp_product_is_power_of_two_square(settings.get("tp"), settings.get("cp")):
            skipped_square_constraint += 1
            continue
        if TP_CP_PRODUCT_MIN is not None or TP_CP_PRODUCT_MAX is not None:
            tp_cp_prod = int(settings.get("tp", 1)) * int(settings.get("cp", 1))
            if TP_CP_PRODUCT_MIN is not None and tp_cp_prod < TP_CP_PRODUCT_MIN:
                skipped_out_of_range += 1
                continue
            if TP_CP_PRODUCT_MAX is not None and tp_cp_prod > TP_CP_PRODUCT_MAX:
                skipped_out_of_range += 1
                continue
        if active_gpu_count_min <= num_gpus <= active_gpu_count_max:
            filtered_tasks.append(items)
        else:
            skipped_out_of_range += 1

    if not filtered_tasks:
        print("No configurations within GPU count bounds.")
        return
    if enforce_square and skipped_square_constraint:
        print(f"Skipped {skipped_square_constraint} configuration(s) due to tp*cp square constraint.")

    available_cpus = max(1, os.cpu_count() or 1)
    if MAX_WORKERS is None or MAX_WORKERS <= 0:
        worker_limit = max(1, available_cpus - 1)
    else:
        worker_limit = min(MAX_WORKERS, max(1, available_cpus - 1))
    worker_count = max(1, worker_limit)
    print(f"Using {worker_count} worker process(es) (out of {available_cpus} CPUs).")

    best_runtime_entries: List[Dict[str, object]] = []
    best_runtime_by_gpu_count: Dict[int, List[Dict[str, object]]] = {}

    for hw_index, hw_path in enumerate(hw_config_paths):
        tag = Path(hw_path).stem if len(hw_config_paths) > 1 else None
        report_tag = _combine_tags(output_tag, tag)
        plot_tag = _combine_tags(output_tag, tag)
        report_path = _tagged_output_path(active_report_output_path, tag) if tag else active_report_output_path
        plot_path = _tagged_output_path(active_plot_output_path, tag) if tag else active_plot_output_path
        hw_label = label_order[hw_index] if hw_index < len(label_order) else Path(hw_path).stem
        print(f"\n=== Evaluating hardware config: {hw_path} ===")

        results = []
        results_from_report = False
        skipped_errors = 0
        memory_violations = 0
        error_messages: List[str] = []
        evaluated = 0
        model_config_id = ""

        def _record_error(
            *,
            status: str,
            error: str,
            source: str,
            parallelism: Optional[Dict[str, object]] = None,
            num_gpus: Optional[int] = None,
            traceback_text: str = "",
        ) -> None:
            record: Dict[str, object] = {
                "status": status,
                "source": source,
                "error": error,
                "hardware_config": str(Path(hw_path).resolve()),
                "hardware_label": hw_label,
            }
            if model_config_id:
                record["model_config"] = model_config_id
            if parallelism is not None:
                record["parallelism"] = dict(parallelism)
            if num_gpus is not None:
                record["num_gpus"] = int(num_gpus)
            if traceback_text:
                record["traceback"] = traceback_text
            _append_error_log(record, active_error_log_path)

        def _consume_worker_result(result: Dict[str, object]) -> None:
            nonlocal skipped_errors, memory_violations, evaluated, cache_dirty
            settings = result.get("parallelism", {})
            num_gpus = total_gpu_count(settings)
            if result.get("status") != "ok":
                skipped_errors += 1
                msg = result.get("error") or "unknown error"
                tb = result.get("traceback")
                if tb:
                    error_messages.append(f"{settings}: {msg}\n{tb}")
                else:
                    error_messages.append(f"{settings}: {msg}")
                _record_error(
                    status="error",
                    source="worker_result",
                    error=str(msg),
                    parallelism=dict(settings),
                    num_gpus=num_gpus,
                    traceback_text=str(tb or ""),
                )
                return
            metrics = result.get("metrics", {})
            entry = {
                "parallelism": settings,
                "num_gpus": num_gpus,
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
            key = _cache_key(hw_path, model_config_id, settings)
            runtime_cache[key] = {
                "hardware_config": str(Path(hw_path).resolve()),
                "model_config": model_config_id,
                "parallelism": dict(settings),
                "num_gpus": num_gpus,
                "runtime": entry["runtime"],
                "prefill_time": entry["prefill_time"],
                "decode_time": entry["decode_time"],
                "performance": entry["performance"],
                "total_flops": entry["total_flops"],
                "achieved_flops": entry["achieved_flops"],
                "peak_flops": entry["peak_flops"],
                "mfu": entry["mfu"],
                "memory_exceeded": entry["memory_exceeded"],
                "memory_violation_gb": entry["memory_violation_gb"],
            }
            cache_dirty = True

            if metrics.get("memory_exceeded"):
                memory_violations += 1
            evaluated += 1
            results.append(entry)

        if os.path.exists(report_path):
            try:
                loaded = load_results_from_report(report_path)
                if loaded:
                    results = loaded
                    results_from_report = True
                    print(f"Found existing TSV report at {report_path}; using cached results.")
                else:
                    print(f"Existing TSV report at {report_path} is empty; running sweep.")
            except Exception as exc:
                print(f"Warning: failed to load existing TSV report {report_path}: {exc}; running sweep.")

        if not results_from_report:
            base_hw_dict = read_yaml(hw_path)
            model_config_path, model_config_id = _model_config_with_overrides(active_model_config_path, hw_path)
            if active_derates:
                _apply_device_derates(base_hw_dict, active_derates)
                model_config_id = f"{model_config_id}{derate_cache_tag}"
            model_config_obj = config.parse_config(model_config_path, config_type=mode)
            active_run_type = str(
                getattr(getattr(model_config_obj, "model_config", None), "run_type", "training")
            ).lower()

            try:
                tasks_to_eval: List[Tuple[Tuple[str, object], ...]] = []
                for items in filtered_tasks:
                    settings = build_parallelism_settings(dict(items))
                    num_gpus = total_gpu_count(settings)
                    key = _cache_key(hw_path, model_config_id, settings)
                    cached_entry = runtime_cache.get(key)
                    if cached_entry is None or (
                        active_run_type == "inference"
                        and not _cache_entry_has_inference_breakdown(cached_entry)
                    ):
                        tasks_to_eval.append(items)
                        continue
                    entry = {
                        "parallelism": settings,
                        "num_gpus": num_gpus,
                        "runtime": float(cached_entry.get("runtime", float("nan"))),
                        "prefill_time": float(cached_entry.get("prefill_time", float("nan"))),
                        "decode_time": float(cached_entry.get("decode_time", float("nan"))),
                        "performance": float(cached_entry.get("performance", float("nan"))),
                        "total_flops": float(cached_entry.get("total_flops", float("nan"))),
                        "achieved_flops": float(cached_entry.get("achieved_flops", float("nan"))),
                        "peak_flops": float(cached_entry.get("peak_flops", float("nan"))),
                        "mfu": float(cached_entry.get("mfu", float("nan"))),
                        "memory_exceeded": bool(cached_entry.get("memory_exceeded", False)),
                        "memory_violation_gb": float(cached_entry.get("memory_violation_gb", 0.0) or 0.0),
                    }
                    results.append(entry)

                if worker_count > 1 and len(tasks_to_eval) > 1:
                    futures = {}
                    processed_futures = set()
                    pending_items: List[Tuple[Tuple[str, object], ...]] = []
                    pool_broken = False

                    def _retry_isolated(items: Tuple[Tuple[str, object], ...]) -> None:
                        nonlocal skipped_errors
                        retry_settings = build_parallelism_settings(dict(items))
                        retry_num_gpus = total_gpu_count(retry_settings)
                        try:
                            result = _run_task_in_isolated_process(items, base_hw_dict, model_config_path, mode)
                        except BrokenProcessPool as exc2:
                            skipped_errors += 1
                            error_messages.append(f"{dict(items)}: isolated worker died ({exc2})")
                            _record_error(
                                status="error",
                                source="isolated_worker",
                                error=str(exc2),
                                parallelism=retry_settings,
                                num_gpus=retry_num_gpus,
                            )
                            return
                        except BaseException as exc2:
                            skipped_errors += 1
                            tb = traceback.format_exc()
                            error_messages.append(
                                f"{dict(items)}: isolated retry crashed ({exc2.__class__.__name__}: {exc2})\n"
                                f"{tb}"
                            )
                            _record_error(
                                status="error",
                                source="isolated_retry",
                                error=f"{exc2.__class__.__name__}: {exc2}",
                                parallelism=retry_settings,
                                num_gpus=retry_num_gpus,
                                traceback_text=tb,
                            )
                            return
                        _consume_worker_result(result)

                    try:
                        if USE_THREADPOOL:
                            Executor = ThreadPoolExecutor
                            executor_kwargs = {"max_workers": worker_count}
                            # Threads share process memory; seed globals once.
                            _worker_init(base_hw_dict, model_config_path, mode)
                        else:
                            Executor = ProcessPoolExecutor
                            executor_kwargs = {
                                "max_workers": worker_count,
                                "initializer": _worker_init,
                                "initargs": (base_hw_dict, model_config_path, mode),
                            }

                        with Executor(**executor_kwargs) as executor:
                            futures = {executor.submit(_worker_task, items): items for items in tasks_to_eval}
                            with tqdm(total=len(tasks_to_eval), desc="Evaluating", unit="config") as progress:
                                for future in as_completed(futures):
                                    items = futures[future]
                                    progress.update(1)
                                    try:
                                        result = future.result()
                                    except BrokenProcessPool as exc:
                                        pool_broken = True
                                        skipped_errors += 1
                                        error_messages.append(f"{dict(items)}: worker process died ({exc})")
                                        broken_settings = build_parallelism_settings(dict(items))
                                        _record_error(
                                            status="error",
                                            source="process_pool",
                                            error=str(exc),
                                            parallelism=broken_settings,
                                            num_gpus=total_gpu_count(broken_settings),
                                        )
                                        processed_futures.add(future)
                                        _retry_isolated(items)
                                        break
                                    except Exception as exc:
                                        skipped_errors += 1
                                        error_messages.append(f"{dict(items)}: {exc}")
                                        future_settings = build_parallelism_settings(dict(items))
                                        _record_error(
                                            status="error",
                                            source="future_result",
                                            error=f"{exc.__class__.__name__}: {exc}",
                                            parallelism=future_settings,
                                            num_gpus=total_gpu_count(future_settings),
                                        )
                                        processed_futures.add(future)
                                        _retry_isolated(items)
                                        continue
                                    processed_futures.add(future)
                                    _consume_worker_result(result)
                    except BrokenProcessPool as exc:
                        pool_broken = True
                        skipped_errors += 1
                        error_messages.append(f"worker pool terminated early: {exc}")
                        _record_error(
                            status="error",
                            source="process_pool",
                            error=str(exc),
                        )

                    if pool_broken:
                        if not futures:
                            pending_items = list(tasks_to_eval)
                        else:
                            for future, items in futures.items():
                                if future in processed_futures:
                                    continue
                                if future.done():
                                    try:
                                        result = future.result()
                                    except Exception as exc:
                                        pending_settings = build_parallelism_settings(dict(items))
                                        # A broken pool can leave completed futures unreadable. Retry the
                                        # configuration in a fresh isolated worker instead of dropping it.
                                        _record_error(
                                            status="retry",
                                            source="pending_future",
                                            error=f"{exc.__class__.__name__}: {exc}",
                                            parallelism=pending_settings,
                                            num_gpus=total_gpu_count(pending_settings),
                                        )
                                        _retry_isolated(items)
                                        continue
                                    _consume_worker_result(result)
                                else:
                                    pending_items.append(items)

                        if pending_items:
                            print(
                                "Worker pool broke; retrying remaining configs in isolated workers.",
                                file=sys.stderr,
                            )
                            for items in pending_items:
                                try:
                                    result = _run_task_in_isolated_process(
                                        items, base_hw_dict, model_config_path, mode
                                    )
                                except BrokenProcessPool as exc:
                                    skipped_errors += 1
                                    error_messages.append(f"{dict(items)}: worker process died ({exc})")
                                    retry_settings = build_parallelism_settings(dict(items))
                                    _record_error(
                                        status="error",
                                        source="isolated_worker",
                                        error=str(exc),
                                        parallelism=retry_settings,
                                        num_gpus=total_gpu_count(retry_settings),
                                    )
                                    continue
                                except Exception as exc:
                                    skipped_errors += 1
                                    error_messages.append(f"{dict(items)}: {exc}")
                                    retry_settings = build_parallelism_settings(dict(items))
                                    _record_error(
                                        status="error",
                                        source="isolated_retry",
                                        error=f"{exc.__class__.__name__}: {exc}",
                                        parallelism=retry_settings,
                                        num_gpus=total_gpu_count(retry_settings),
                                    )
                                    continue
                                _consume_worker_result(result)
                elif tasks_to_eval:
                    with tqdm(total=len(tasks_to_eval), desc="Evaluating", unit="config") as progress:
                        for items in tasks_to_eval:
                            settings = build_parallelism_settings(dict(items))
                            num_gpus = total_gpu_count(settings)
                            progress.update(1)
                            try:
                                metrics = evaluate_parallelism(base_hw_dict, model_config_obj, mode, settings)
                            except Exception as exc:
                                skipped_errors += 1
                                error_messages.append(f"{settings}: {exc}")
                                _record_error(
                                    status="error",
                                    source="serial_eval",
                                    error=f"{exc.__class__.__name__}: {exc}",
                                    parallelism=settings,
                                    num_gpus=num_gpus,
                                )
                                continue

                            entry = {
                                "parallelism": settings,
                                "num_gpus": num_gpus,
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
                            key = _cache_key(hw_path, model_config_id, settings)
                            runtime_cache[key] = {
                                "hardware_config": str(Path(hw_path).resolve()),
                                "model_config": model_config_id,
                                "parallelism": dict(settings),
                                "num_gpus": num_gpus,
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
                            cache_dirty = True

                            if metrics.get("memory_exceeded"):
                                memory_violations += 1
                                print(
                                    f"Memory capacity exceeded for configuration {settings} "
                                    f"({metrics.get('memory_violation_gb', 0.0):.3f} GB).",
                                    file=sys.stderr,
                                )

                            evaluated += 1
                            results.append(entry)
            finally:
                if model_config_path != active_model_config_path and os.path.exists(model_config_path):
                    try:
                        os.unlink(model_config_path)
                    except Exception:
                        pass

        total_skipped = skipped_out_of_range + skipped_errors
        if not results:
            print(
                "No valid configurations evaluated ({} skipped: {} out-of-range, {} errors; memory violations: {}).".format(
                    total_skipped, skipped_out_of_range, skipped_errors, memory_violations
                )
            )
            if error_messages:
                print("Encountered errors for configurations:")
                for msg in error_messages:
                    print(f"  {msg}")
            continue

        if results_from_report:
            print(f"Loaded {len(results)} configuration(s) from cached TSV; skipping sweep.")
        else:
            print(
                f"Evaluated {evaluated} configuration(s); skipped {total_skipped} "
                f"(out_of_range={skipped_out_of_range}, errors={skipped_errors}); "
                f"memory violations={memory_violations}."
            )
            if error_messages:
                print("Some configurations failed:")
                for msg in error_messages:
                    print(f"  {msg}")

        best_candidates = [
            item for item in results
            if not item.get("memory_exceeded") and math.isfinite(item.get("runtime", float("nan")))
        ]
        if not best_candidates:
            print(
                "No valid (non-memory-violating) configurations for best-runtime selection; "
                "skipping best/runtime summary for this hardware.",
                file=sys.stderr,
            )
            continue

        # Track best runtime per GPU count for this hardware (exclude memory violations).
        per_gpu_best: Dict[int, float] = {}
        for item in best_candidates:
            num_gpus = int(item.get("num_gpus", 0))
            runtime_val = float(item.get("runtime", float("nan")))
            if not math.isfinite(runtime_val):
                continue
            current = per_gpu_best.get(num_gpus)
            if current is None or runtime_val < current:
                per_gpu_best[num_gpus] = runtime_val

        best = min(best_candidates, key=lambda item: item["runtime"])
        print("\nBest configuration (lowest runtime):")
        print("  Parallelism: {}".format(best["parallelism"]))
        print("  GPUs: {}".format(best["num_gpus"]))
        print("  Runtime: {:.4f} s".format(best["runtime"]))
        print("  Performance (1/s): {:.4f}".format(best["performance"]))
        print("  Total FLOPs: {:.3e}".format(best["total_flops"]))
        print("  MFU: {:.3f}".format(best["mfu"]))
        if best["memory_exceeded"]:
            print("  Memory capacity exceeded by {:.3f} GB".format(best["memory_violation_gb"]))
        best_runtime_entries.append({"label": hw_label, "runtime": best["runtime"]})
        for num_gpus, runtime_val in per_gpu_best.items():
            best_runtime_by_gpu_count.setdefault(num_gpus, []).append(
                {"label": hw_label, "runtime": runtime_val}
            )

        if not results_from_report:
            try:
                write_report(results, report_path)
                print("Wrote detailed report to {}".format(report_path))
            except Exception as exc:
                print("Warning: failed to write report: {}".format(exc), file=sys.stderr)

        if results:
            plot_results(results, plot_path)
    # plot_mfu(results, PLOT_MFU_OUTPUT_PATH)

    if best_runtime_entries:
        plot_best_runtimes(best_runtime_entries, active_best_runtime_plot_path, label_order)
    if best_runtime_by_gpu_count:
        plot_best_runtimes_per_gpu(
            best_runtime_by_gpu_count, active_best_runtime_per_gpu_dir, label_order
        )
        plot_best_runtimes_per_gpu_combined(
            best_runtime_by_gpu_count, active_best_runtime_per_gpu_combined_path, label_order
        )
        plot_speedup_per_gpu_combined(
            best_runtime_by_gpu_count,
            active_best_speedup_per_gpu_combined_path,
            label_order,
            base_label=label_order[0] if label_order else "Base",
            omit_gpu_counts=[64],
        )
    if cache_dirty:
        _save_runtime_cache(runtime_cache, rooted_runtime_cache_path)

if __name__ == "__main__":
    main()
