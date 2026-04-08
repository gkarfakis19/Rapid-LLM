#!/usr/bin/env python3
"""
Unified validation harness for inference/training suites with a shared worker pool.

This script composes specs from:
  - nvidia_inf.py (A100/H100, IMEC + NVIDIA rows)
  - nvidia_train_validation.py (A100)
  - uci_train_validation.py (A100)
  - mosiacml_train.py (H100)

All selected suites are executed in one run_validation_suite(...) call so they share
the same ProcessPoolExecutor worker pool.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import yaml

try:
    from . import mosiacml_train as mosaic_train
    from . import nvidia_inf
    from . import nvidia_train_validation as nvidia_train
    from . import uci_train_validation
    from .validation_helpers import (
        ValidationResult,
        ValidationSpec,
        parse_inference_time,
        parse_training_time,
        run_validation_suite,
    )
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from validation_scripts import mosiacml_train as mosaic_train  # type: ignore
    from validation_scripts import nvidia_inf  # type: ignore
    from validation_scripts import nvidia_train_validation as nvidia_train  # type: ignore
    from validation_scripts import uci_train_validation  # type: ignore
    from validation_scripts.validation_helpers import (  # type: ignore
        ValidationResult,
        ValidationSpec,
        parse_inference_time,
        parse_training_time,
        run_validation_suite,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "validation" / "harness"
DEFAULT_SUITES = ("nvidia_inf", "nvidia_train", "uci_train", "mosaic_train")
DEFAULT_DERATE_CONFIG = PROJECT_ROOT / "validation_scripts" / "validation_configs" / "harness_derates.yaml"

NVIDIA_TRAIN_DEFAULT_HW = (
    PROJECT_ROOT
    / "validation_scripts"
    / "validation_configs"
    / "hardware-config"
    / "A100_SXM4_80GB_train_validation.yaml"
)
UCI_TRAIN_DEFAULT_HW = (
    PROJECT_ROOT
    / "validation_scripts"
    / "validation_configs"
    / "hardware-config"
    / "A100_PCIe_80GB_train_validation.yaml"
)
NVIDIA_TRAIN_DEFAULT_INPUT = (
    PROJECT_ROOT / "validation_scripts" / "train_validation_data" / "nvidia_train_validation_cases.csv"
)
NVIDIA_TRAIN_DEFAULT_STAGE = PROJECT_ROOT / "validation_scripts" / "train_validation_data" / "STAGE_data.csv"

MOSAIC_DEFAULT_HW = (
    PROJECT_ROOT / "validation_scripts" / "validation_configs" / "hardware-config" / "H100_SXM5_80GB.mosaic_train.yaml"
)


@dataclass
class SuiteBundle:
    name: str
    specs: List[ValidationSpec]
    finalize: Callable[[List[ValidationResult], Path, bool], Dict[str, Any]]


@dataclass(frozen=True)
class DerateConfig:
    path: Path
    shared: Dict[str, Any]
    device_type_derates: Dict[str, Dict[str, float]]


DEVICE_TYPE_A100_PCIE = "A100_PCIE"
DEVICE_TYPE_A100_SXM4 = "A100_SXM4"
DEVICE_TYPE_H100_SXM5 = "H100_SXM5"

# Maps suite/device pairs to the corresponding hardware class derates.
SUITE_DEVICE_TYPE_BY_DEVICE: Dict[str, Dict[str, str]] = {
    "nvidia_inf": {
        "A100": DEVICE_TYPE_A100_SXM4,
        "H100": DEVICE_TYPE_H100_SXM5,
    },
    "nvidia_train": {
        "A100": DEVICE_TYPE_A100_SXM4,
    },
    "uci_train": {
        "A100": DEVICE_TYPE_A100_PCIE,
    },
    "mosaic_train": {
        "H100": DEVICE_TYPE_H100_SXM5,
    },
}


def _new_run_id() -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return f"{timestamp}_{os.getpid()}"


def _resolve_repo_relative_path(path_like: Path | str) -> Path:
    """Resolve relative paths against repo root first, then cwd."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    repo_candidate = PROJECT_ROOT / path
    if repo_candidate.exists():
        return repo_candidate
    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    # Fall back to repo-relative to keep behavior deterministic.
    return repo_candidate


def _assert_uci_suite_module_binding() -> None:
    module_file = getattr(uci_train_validation, "__file__", None)
    if not module_file:
        raise RuntimeError(
            "Failed to resolve UCI suite module file; expected uci_train_validation.py."
        )
    module_path = Path(str(module_file)).resolve()
    if module_path.name != "uci_train_validation.py":
        raise RuntimeError(
            "UCI suite is miswired: expected validation_scripts/uci_train_validation.py "
            f"but resolved {module_path}."
        )


def _parse_suite_list(raw: str) -> List[str]:
    suites: List[str] = []
    for part in str(raw).split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name not in DEFAULT_SUITES:
            raise ValueError(
                f"Unknown suite '{name}'. Expected one of: {', '.join(DEFAULT_SUITES)}."
            )
        if name not in suites:
            suites.append(name)
    if not suites:
        return list(DEFAULT_SUITES)
    return suites


def _parse_inference_devices(raw: str) -> List[str]:
    tokens = [d.strip().upper() for d in str(raw).split(",") if d.strip()]
    if not tokens:
        return ["A100", "H100"]
    unknown = [dev for dev in tokens if dev not in {"A100", "H100"}]
    if unknown:
        raise ValueError(
            f"Unsupported inference device(s): {', '.join(unknown)}. Expected A100 and/or H100."
        )
    ordered: List[str] = []
    seen = set()
    for dev in tokens:
        if dev not in seen:
            ordered.append(dev)
            seen.add(dev)
    return ordered or ["A100", "H100"]


def _order_suites_for_submission(suites: Sequence[str]) -> List[str]:
    indexed = list(enumerate(suites))
    indexed.sort(key=lambda item: (0 if item[1] == "nvidia_train" else 1, item[0]))
    return [name for _, name in indexed]


def _to_positive_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Expected numeric value for '{field_name}', got {value!r}.")
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"Expected '{field_name}' > 0, got {value!r}.")
    return parsed


def _to_unit_interval_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Expected numeric value for '{field_name}', got {value!r}.")
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"Expected '{field_name}' to be in [0.0, 1.0], got {value!r}.")
    return parsed


def _normalize_device_type_key(value: Any) -> str:
    text = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text


def _device_type_for_suite_device(suite: str, device: str) -> str:
    suite_key = str(suite).strip().lower()
    device_key = str(device).strip().upper()
    suite_map = SUITE_DEVICE_TYPE_BY_DEVICE.get(suite_key)
    if suite_map is None:
        raise ValueError(f"No device-type mapping for suite {suite!r}.")
    device_type = suite_map.get(device_key)
    if device_type is None:
        raise ValueError(
            f"No device-type mapping for suite={suite!r}, device={device_key!r}."
        )
    return device_type


def _required_suite_devices(
    selected_suites: Sequence[str],
    *,
    inference_devices: Sequence[str],
) -> Dict[str, List[str]]:
    required: Dict[str, List[str]] = {}
    for suite in selected_suites:
        if suite == "nvidia_inf":
            required[suite] = [d.upper() for d in inference_devices]
        elif suite in {"nvidia_train", "uci_train"}:
            required[suite] = ["A100"]
        elif suite == "mosaic_train":
            required[suite] = ["H100"]
    return required


def _load_derate_config(
    path_like: Path | str,
    *,
    selected_suites: Sequence[str],
    inference_devices: Sequence[str],
) -> DerateConfig:
    path = _resolve_repo_relative_path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"Derate config YAML not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Derate config must be a YAML mapping at root: {path}")

    shared_raw = raw.get("shared")
    if not isinstance(shared_raw, Mapping):
        raise ValueError("Derate config requires a 'shared' mapping.")
    shared_network_raw = shared_raw.get("network")
    if not isinstance(shared_network_raw, Mapping):
        raise ValueError("Derate config requires a 'shared.network' mapping.")
    shared_overlap_raw = shared_network_raw.get("overlap")
    if not isinstance(shared_overlap_raw, Mapping):
        raise ValueError("Derate config requires a 'shared.network.overlap' mapping.")
    shared_overlap = {
        "tp_overlap": _to_unit_interval_float(
            shared_overlap_raw.get("tp_overlap"),
            "shared.network.overlap.tp_overlap",
        ),
        "tp_sp_overlap": _to_unit_interval_float(
            shared_overlap_raw.get("tp_sp_overlap"),
            "shared.network.overlap.tp_sp_overlap",
        ),
        "cp_overlap": _to_unit_interval_float(
            shared_overlap_raw.get("cp_overlap"),
            "shared.network.overlap.cp_overlap",
        ),
    }
    shared = {
        "kernel_launch_overhead_s": _to_positive_float(
            shared_raw.get("kernel_launch_overhead_s"),
            "shared.kernel_launch_overhead_s",
        ),
        "network": {
            "overlap": shared_overlap,
        },
    }

    device_types_raw = raw.get("device_types")
    if not isinstance(device_types_raw, Mapping):
        raise ValueError("Derate config requires a 'device_types' mapping.")
    device_type_derates: Dict[str, Dict[str, float]] = {}
    for raw_name, derate_raw in device_types_raw.items():
        device_type = _normalize_device_type_key(raw_name)
        if not device_type:
            raise ValueError(f"Invalid empty device type in config: {raw_name!r}")
        if device_type in device_type_derates:
            raise ValueError(f"Duplicate derate device type entry: {device_type!r}")
        if not isinstance(derate_raw, Mapping):
            raise ValueError(f"Derate config mapping required for device_types.{raw_name}")
        device_type_derates[device_type] = {
            "dram_util": _to_positive_float(
                derate_raw.get("dram_util"),
                f"device_types.{raw_name}.dram_util",
            ),
            "network_util": _to_positive_float(
                derate_raw.get("network_util"),
                f"device_types.{raw_name}.network_util",
            ),
            "compute_util": _to_positive_float(
                derate_raw.get("compute_util"),
                f"device_types.{raw_name}.compute_util",
            ),
        }

    required = _required_suite_devices(
        selected_suites,
        inference_devices=inference_devices,
    )
    for suite, devices in required.items():
        for device in devices:
            device_type = _device_type_for_suite_device(suite, device)
            if device_type not in device_type_derates:
                raise ValueError(
                    "Derate config missing device type entry for "
                    f"suite={suite}, device={device}: device_types.{device_type}"
                )

    return DerateConfig(
        path=path,
        shared=shared,
        device_type_derates=device_type_derates,
    )


def _suite_compute_util_from_device_type_derates(
    *,
    selected_suites: Sequence[str],
    inference_devices: Sequence[str],
    derate_config: DerateConfig,
) -> Dict[str, Dict[str, float]]:
    required = _required_suite_devices(
        selected_suites,
        inference_devices=inference_devices,
    )
    suite_compute_util: Dict[str, Dict[str, float]] = {}
    for suite, devices in required.items():
        suite_compute_util[suite] = {}
        for device in devices:
            device_type = _device_type_for_suite_device(suite, device)
            suite_compute_util[suite][device] = float(
                derate_config.device_type_derates[device_type]["compute_util"]
            )
    return suite_compute_util


def _suite_device_type_map(
    *,
    selected_suites: Sequence[str],
    inference_devices: Sequence[str],
) -> Dict[str, Dict[str, str]]:
    required = _required_suite_devices(
        selected_suites,
        inference_devices=inference_devices,
    )
    out: Dict[str, Dict[str, str]] = {}
    for suite, devices in required.items():
        out[suite] = {}
        for device in devices:
            out[suite][device] = _device_type_for_suite_device(suite, device)
    return out


def _is_list_of_dicts(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    return all(isinstance(item, Mapping) for item in value)


def _deep_update(target: Dict[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, Mapping):
            existing = target.get(key)
            if not isinstance(existing, dict):
                existing = {}
            target[key] = _deep_update(copy.deepcopy(existing), value)
        elif _is_list_of_dicts(value) and _is_list_of_dicts(target.get(key)):
            target[key] = _merge_list_of_dicts_by_id(
                copy.deepcopy(target.get(key, [])), copy.deepcopy(value)
            )
        else:
            target[key] = copy.deepcopy(value)
    return target


def _merge_list_of_dicts_by_id(
    base_list: List[Mapping[str, Any]],
    override_list: List[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    base_index: Dict[Any, Dict[str, Any]] = {}
    base_order: List[Any] = []
    passthrough: List[Dict[str, Any]] = []

    for entry in base_list:
        if not isinstance(entry, Mapping):
            continue
        as_dict = dict(entry)
        entry_id = as_dict.get("id")
        if entry_id is None:
            passthrough.append(copy.deepcopy(as_dict))
            continue
        base_index[entry_id] = copy.deepcopy(as_dict)
        base_order.append(entry_id)

    for entry in override_list:
        if not isinstance(entry, Mapping):
            continue
        as_dict = dict(entry)
        entry_id = as_dict.get("id")
        if entry_id is None:
            passthrough.append(copy.deepcopy(as_dict))
            continue
        if entry_id in base_index:
            merged = _deep_update(copy.deepcopy(base_index[entry_id]), as_dict)
            base_index[entry_id] = merged
        else:
            base_index[entry_id] = copy.deepcopy(as_dict)
            base_order.append(entry_id)

    merged_list: List[Dict[str, Any]] = list(passthrough)
    for entry_id in base_order:
        merged_list.append(base_index[entry_id])
    return merged_list


def _mean_abs(values: Iterable[float]) -> float:
    finite = [abs(float(v)) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def _max_abs(values: Iterable[float]) -> float:
    finite = [abs(float(v)) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return float("nan")
    return max(finite)


def _common_hw_override(
    *,
    suite_name: str,
    device: str,
    derate_config: DerateConfig,
) -> Dict[str, Any]:
    device_upper = str(device).upper()
    device_type = _device_type_for_suite_device(suite_name, device_upper)
    device_derates = derate_config.device_type_derates.get(device_type)
    if device_derates is None:
        raise ValueError(
            f"Missing derate entry for device_type={device_type!r} "
            f"(suite={suite_name!r}, device={device_upper!r}) in {derate_config.path}."
        )
    common: Dict[str, Any] = {
        "sw_param": {
            "kernel_launch_overhead": float(derate_config.shared["kernel_launch_overhead_s"]),
        },
        "tech_param": {
            "DRAM": {
                "util": float(device_derates["dram_util"]),
            }
        },
        "network": {
            "dimensions": [
                {
                    "id": "dim0",
                    "topology": {
                        "util": float(device_derates["network_util"]),
                    },
                }
            ],
            "overlap": copy.deepcopy(derate_config.shared["network"]["overlap"]),
        },
    }
    common.setdefault("tech_param", {}).setdefault("core", {})
    common["tech_param"]["core"]["util"] = float(device_derates["compute_util"])
    return common


def _apply_common_override(
    spec: ValidationSpec,
    *,
    suite_name: str,
    derate_config: DerateConfig,
) -> ValidationSpec:
    metadata = dict(spec.metadata or {})
    metadata["suite"] = suite_name
    device = str(metadata.get("device", "A100")).upper()
    common = _common_hw_override(
        suite_name=suite_name,
        device=device,
        derate_config=derate_config,
    )

    hw_overrides = copy.deepcopy(spec.hardware_overrides or {})
    merged_hw = _deep_update(hw_overrides, common)
    return replace(spec, hardware_overrides=merged_hw, metadata=metadata)


def _parse_runtime_metrics(output: str, spec: ValidationSpec) -> Dict[str, Any]:
    metric_kind = str((spec.metadata or {}).get("metric_kind", "auto")).strip().lower()
    if metric_kind == "inference":
        value = parse_inference_time(output, spec)["inference_time_s"]
        return {"inference_time_s": float(value), "training_time_s": float(value)}
    if metric_kind == "training":
        value = parse_training_time(output, spec)["training_time_s"]
        return {"training_time_s": float(value), "inference_time_s": float(value)}

    # Fallback for mixed/unknown specs.
    try:
        value = parse_inference_time(output, spec)["inference_time_s"]
        return {"inference_time_s": float(value), "training_time_s": float(value)}
    except Exception:
        value = parse_training_time(output, spec)["training_time_s"]
        return {"training_time_s": float(value), "inference_time_s": float(value)}


def _augment_specs(
    specs: Sequence[ValidationSpec],
    *,
    suite: str,
    dataset: Optional[str] = None,
    metric_kind: str,
    device: Optional[str] = None,
) -> List[ValidationSpec]:
    out: List[ValidationSpec] = []
    for idx, spec in enumerate(specs):
        metadata = dict(spec.metadata or {})
        metadata["suite"] = suite
        metadata["dataset"] = dataset
        metadata["metric_kind"] = metric_kind
        if device is not None:
            metadata["device"] = str(device).upper()
        metadata["local_index"] = int(idx)
        out.append(replace(spec, metadata=metadata))
    return out


def _build_nvidia_inf_bundle(args: argparse.Namespace) -> SuiteBundle:
    devices = _parse_inference_devices(args.inference_devices)

    all_specs: List[ValidationSpec] = []
    imec_lookup: Dict[str, Dict[Tuple[str, int], float]] = {}
    nvidia_lookup: Dict[str, Dict[Tuple[str, int, int, int, int], float]] = {}

    for device in devices:
        specs_imec, actual_imec, _, _ = nvidia_inf.build_specs_for_device(
            device,
            network_ignored=False,
            models=None,
            fit_model=True,
        )
        specs_nv, actual_nv, _, _ = nvidia_inf.build_nvidia_specs_for_device(
            device,
            network_ignored=False,
            models=None,
        )
        imec_lookup[device] = actual_imec
        nvidia_lookup[device] = actual_nv

        all_specs.extend(
            _augment_specs(
                specs_imec,
                suite="nvidia_inf",
                dataset="imec",
                metric_kind="inference",
                device=device,
            )
        )
        all_specs.extend(
            _augment_specs(
                specs_nv,
                suite="nvidia_inf",
                dataset="nvidia",
                metric_kind="inference",
                device=device,
            )
        )

    def finalize(results: List[ValidationResult], suite_dir: Path, enable_plot: bool) -> Dict[str, Any]:
        suite_dir.mkdir(parents=True, exist_ok=True)
        csv_paths: List[str] = []
        plot_paths: List[str] = []
        all_pct_errors: List[float] = []
        total_cases = 0
        total_success = 0
        per_device_summaries: List[Dict[str, Any]] = []
        imec_plot_inputs: Dict[str, pd.DataFrame] = {}
        nvidia_plot_inputs: Dict[str, pd.DataFrame] = {}

        imec_all_rows: List[Dict[str, object]] = []
        nvidia_all_rows: List[Dict[str, object]] = []

        for device in devices:
            vidur_rows: Optional[List[Dict[str, object]]] = None
            vidur_nvidia_rows: Optional[List[Dict[str, object]]] = None
            genz_rows: Optional[List[Dict[str, object]]] = None
            genz_nvidia_rows: Optional[List[Dict[str, object]]] = None
            if device == "A100":
                vidur_rows = nvidia_inf._load_vidur_imec_errors(nvidia_inf.VIDUR_IMEC)
                vidur_nvidia_rows = nvidia_inf._load_vidur_nvidia_errors(
                    nvidia_inf.VIDUR_NVIDIA,
                    device=device,
                )
                genz_rows = nvidia_inf._load_genz_imec_errors(nvidia_inf.GENZ_IMEC)
                genz_nvidia_rows = nvidia_inf._load_genz_nvidia_errors(
                    nvidia_inf.GENZ_NVIDIA,
                    device=device,
                )

            device_imec_results = [
                res
                for res in results
                if str((res.spec.metadata or {}).get("device", "")).upper() == device
                and str((res.spec.metadata or {}).get("dataset", "")).lower() == "imec"
            ]
            device_nv_results = [
                res
                for res in results
                if str((res.spec.metadata or {}).get("device", "")).upper() == device
                and str((res.spec.metadata or {}).get("dataset", "")).lower() == "nvidia"
            ]

            imec_rows = nvidia_inf.compute_pct_errors(device_imec_results, imec_lookup[device])
            nv_rows = nvidia_inf.compute_nvidia_pct_errors(device_nv_results, nvidia_lookup[device])
            imec_all_rows.extend(imec_rows)
            nvidia_all_rows.extend(nv_rows)

            total_cases += len(imec_rows) + len(nv_rows)
            total_success += sum(1 for row in imec_rows if bool(row.get("success")))
            total_success += sum(1 for row in nv_rows if bool(row.get("success")))
            all_pct_errors.extend(
                float(row["pct_error"])
                for row in imec_rows
                if row.get("pct_error") is not None and math.isfinite(float(row["pct_error"]))
            )
            all_pct_errors.extend(
                float(row["pct_error"])
                for row in nv_rows
                if row.get("pct_error") is not None and math.isfinite(float(row["pct_error"]))
            )

            imec_df = nvidia_inf._load_device_data(device).copy()
            for row in imec_rows:
                imec_df.loc[
                    (imec_df["device"] == row["device"])
                    & (imec_df["model"] == row["model"])
                    & (imec_df["TP"] == row["tp"]),
                    "seconds",
                ] = row["inference_time_s"]
                imec_df.loc[
                    (imec_df["device"] == row["device"])
                    & (imec_df["model"] == row["model"])
                    & (imec_df["TP"] == row["tp"]),
                    "pct_error",
                ] = row["pct_error"]

            imec_csv = suite_dir / f"imec_{device.lower()}_validation.csv"
            imec_df.to_csv(imec_csv, index=False)
            csv_paths.append(str(imec_csv))
            imec_plot_inputs[device] = imec_df.copy()

            nv_df = nvidia_inf._load_nvidia_device_data(device).copy()
            for row in nv_rows:
                nv_df.loc[
                    (nv_df["device"] == row["device"])
                    & (nv_df["model"] == row["model"])
                    & (nv_df["TP"] == row["tp"])
                    & (nv_df["input_tokens"] == row["input_tokens"])
                    & (nv_df["output_tokens"] == row["output_tokens"])
                    & (nv_df["concurrency"] == row["concurrency"]),
                    "seconds",
                ] = row["inference_time_s"]
                nv_df.loc[
                    (nv_df["device"] == row["device"])
                    & (nv_df["model"] == row["model"])
                    & (nv_df["TP"] == row["tp"])
                    & (nv_df["input_tokens"] == row["input_tokens"])
                    & (nv_df["output_tokens"] == row["output_tokens"])
                    & (nv_df["concurrency"] == row["concurrency"]),
                    "pct_error",
                ] = row["pct_error"]
            nv_csv = suite_dir / f"nvidia_{device.lower()}_validation.csv"
            nv_df.to_csv(nv_csv, index=False)
            csv_paths.append(str(nv_csv))
            nvidia_plot_inputs[device] = nv_df.copy()

            device_pct_errors: List[float] = [
                float(row["pct_error"])
                for row in imec_rows
                if row.get("pct_error") is not None and math.isfinite(float(row["pct_error"]))
            ]
            device_pct_errors.extend(
                float(row["pct_error"])
                for row in nv_rows
                if row.get("pct_error") is not None and math.isfinite(float(row["pct_error"]))
            )
            device_cases_total = len(imec_rows) + len(nv_rows)
            device_cases_success = sum(1 for row in imec_rows if bool(row.get("success")))
            device_cases_success += sum(1 for row in nv_rows if bool(row.get("success")))
            per_device_summaries.append(
                {
                    "suite": "nvidia_inf",
                    "device": device,
                    "cases_total": device_cases_total,
                    "cases_success": device_cases_success,
                    "avg_abs_pct_error": _mean_abs(device_pct_errors),
                    "max_abs_pct_error": _max_abs(device_pct_errors),
                }
            )

            if enable_plot:
                nvidia_tp = int(
                    nvidia_inf.NVIDIA_DATASETS.get(device, {}).get(
                        "tp",
                        int(pd.to_numeric(nv_df.get("TP", pd.Series(dtype=float)), errors="coerce").dropna().iloc[0])
                        if "TP" in nv_df.columns and not nv_df.empty
                        else 8,
                    )
                )
                imec_plot = nvidia_inf.plot_device(
                    imec_df,
                    device,
                    suite_dir,
                    vidur_rows=vidur_rows,
                    genz_rows=genz_rows,
                )
                nv_plot = nvidia_inf.plot_nvidia_device(
                    nv_df,
                    device,
                    suite_dir,
                    vidur_rows=vidur_nvidia_rows,
                    genz_rows=genz_nvidia_rows,
                )
                parity_plot = nvidia_inf.plot_validation_parity_combined(
                    imec_df,
                    nv_df,
                    device,
                    suite_dir,
                    nvidia_tp=nvidia_tp,
                )
                if imec_plot is not None:
                    plot_paths.append(str(imec_plot))
                if nv_plot is not None:
                    plot_paths.append(str(nv_plot))
                if parity_plot is not None:
                    plot_paths.append(str(parity_plot))

                combined_error = nvidia_inf._plot_combined_a100_error_bars(
                    imec_all_rows,
                    nvidia_all_rows,
                    suite_dir,
                    device=device,
                    vidur_rows=vidur_rows,
                    vidur_nvidia_rows=vidur_nvidia_rows,
                    genz_nvidia_rows=genz_nvidia_rows,
                    genz_rows=genz_rows,
                )
                combined_ratio = nvidia_inf._plot_combined_ratio_grids(
                    imec_all_rows,
                    nvidia_all_rows,
                    suite_dir,
                    device=device,
                    vidur_rows=vidur_rows,
                    vidur_nvidia_rows=vidur_nvidia_rows,
                    genz_nvidia_rows=genz_nvidia_rows,
                    genz_rows=genz_rows,
                )
                if combined_error is not None:
                    plot_paths.append(str(combined_error))
                if combined_ratio is not None:
                    plot_paths.append(str(combined_ratio))

        if enable_plot:
            device_order = [dev for dev in devices if dev in imec_plot_inputs and dev in nvidia_plot_inputs]
            if device_order:
                stacked_parity = nvidia_inf.plot_validation_parity_combined_devices(
                    imec_plot_inputs,
                    nvidia_plot_inputs,
                    suite_dir,
                    device_order=device_order,
                )
                if stacked_parity is not None:
                    plot_paths.append(str(stacked_parity))

        imec_rows_csv = suite_dir / "imec_rows.csv"
        nvidia_rows_csv = suite_dir / "nvidia_rows.csv"
        pd.DataFrame(imec_all_rows).to_csv(imec_rows_csv, index=False)
        pd.DataFrame(nvidia_all_rows).to_csv(nvidia_rows_csv, index=False)
        csv_paths.extend([str(imec_rows_csv), str(nvidia_rows_csv)])

        return {
            "suite": "nvidia_inf",
            "cases_total": total_cases,
            "cases_success": total_success,
            "avg_abs_pct_error": _mean_abs(all_pct_errors),
            "max_abs_pct_error": _max_abs(all_pct_errors),
            "per_device": per_device_summaries,
            "csv_paths": csv_paths,
            "plot_paths": plot_paths,
        }

    return SuiteBundle(name="nvidia_inf", specs=all_specs, finalize=finalize)


def _build_nvidia_train_bundle(args: argparse.Namespace) -> SuiteBundle:
    input_csv = _resolve_repo_relative_path(args.nvidia_train_input_csv)
    hardware_config = _resolve_repo_relative_path(args.nvidia_train_hardware_config)
    stage_csv = _resolve_repo_relative_path(args.nvidia_train_stage_csv)
    title = str(args.nvidia_train_title)

    if not input_csv.exists():
        raise FileNotFoundError(f"NVIDIA training input CSV not found: {input_csv}")
    if not hardware_config.exists():
        raise FileNotFoundError(f"NVIDIA training hardware config not found: {hardware_config}")

    with input_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in NVIDIA training CSV: {input_csv}")

    specs: List[ValidationSpec] = []
    for idx, row in enumerate(rows):
        model_path = nvidia_train._resolve_model_config(row.get("model", ""))
        hw_path = _resolve_repo_relative_path(nvidia_train._resolve_hw_config(row, hardware_config))
        model_overrides, hw_overrides = nvidia_train._build_overrides(row)
        metadata = {
            "suite": "nvidia_train",
            "device": "A100",
            "dataset": "training",
            "metric_kind": "training",
            "case_index": int(idx),
            "row": dict(row),
            "hardware_config_name": hw_path.name,
        }
        specs.append(
            ValidationSpec(
                label=nvidia_train._short_label(row),
                model_overrides=model_overrides,
                hardware_overrides=hw_overrides,
                model_config_path=str(model_path),
                hardware_config_path=str(hw_path),
                metadata=metadata,
                order=idx,
            )
        )

    def finalize(results: List[ValidationResult], suite_dir: Path, enable_plot: bool) -> Dict[str, Any]:
        suite_dir.mkdir(parents=True, exist_ok=True)
        out_rows: List[Tuple[int, Dict[str, str]]] = []

        for res in results:
            meta = res.spec.metadata or {}
            case_index = int(meta.get("case_index", 0))
            src_row = dict(meta.get("row", {}))

            predicted = (
                float(res.metrics.get("training_time_s", float("nan")))
                if res.success
                else None
            )
            if predicted is not None and not math.isfinite(predicted):
                predicted = None
            actual = nvidia_train._to_float(src_row.get("actual_time_s"))
            signed_pct, abs_pct, abs_error = nvidia_train._compute_errors(actual, predicted)

            out_row = dict(src_row)
            out_row["hardware_config"] = str(meta.get("hardware_config_name", ""))
            out_row[nvidia_train.RAPID_TIME_COLUMN] = "" if predicted is None else str(predicted)
            out_row["signed_pct_error"] = signed_pct
            out_row["abs_pct_error"] = abs_pct
            out_row["abs_error_s"] = abs_error
            out_row["output_root"] = ""
            out_row["error"] = "" if res.success else (res.error or "")
            out_rows.append((case_index, out_row))

        ordered_rows = [row for _, row in sorted(out_rows, key=lambda item: item[0])]

        fieldnames = list(rows[0].keys())
        for field in (
            "hardware_config",
            nvidia_train.RAPID_TIME_COLUMN,
            "signed_pct_error",
            "abs_pct_error",
            "abs_error_s",
            "output_root",
            "error",
        ):
            if field not in fieldnames:
                fieldnames.append(field)

        output_csv = suite_dir / "nvidia_train_validation_result.csv"
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ordered_rows)

        plot_paths: List[str] = []
        if enable_plot:
            stage_rows: List[Dict[str, str]] = []
            if stage_csv.exists():
                with stage_csv.open(newline="", encoding="utf-8") as handle:
                    stage_rows = list(csv.DictReader(handle))
            labels, series = nvidia_train._build_series(ordered_rows, stage_rows)
            if labels:
                plot_output = suite_dir / "nvidia_train_validation_compare.png"
                out_plot = nvidia_train._plot_results(labels, series, plot_output, title)
                plot_paths.append(str(out_plot))

        abs_errors = [
            float(row.get("abs_pct_error", "nan"))
            for row in ordered_rows
            if row.get("abs_pct_error") not in ("", None)
        ]
        success_count = sum(1 for row in ordered_rows if str(row.get("error", "")).strip() == "")
        return {
            "suite": "nvidia_train",
            "cases_total": len(ordered_rows),
            "cases_success": success_count,
            "avg_abs_pct_error": _mean_abs(abs_errors),
            "max_abs_pct_error": _max_abs(abs_errors),
            "csv_paths": [str(output_csv)],
            "plot_paths": plot_paths,
        }

    return SuiteBundle(name="nvidia_train", specs=specs, finalize=finalize)


def _build_uci_train_bundle(args: argparse.Namespace) -> SuiteBundle:
    input_csv = _resolve_repo_relative_path(args.uci_input_csv)
    include_csv = _resolve_repo_relative_path(args.uci_include_csv) if args.uci_include_csv else None
    exclude_csv = _resolve_repo_relative_path(args.uci_exclude_csv) if args.uci_exclude_csv else None
    model_config = _resolve_repo_relative_path(args.uci_model_config)
    hardware_config = _resolve_repo_relative_path(args.uci_hardware_config)
    stage_csv = _resolve_repo_relative_path(args.uci_stage_csv)
    compare_title = "Runtime comparison on 4-GPU system"

    if not input_csv.exists():
        raise FileNotFoundError(f"UCI input CSV not found: {input_csv}")
    if not model_config.exists():
        raise FileNotFoundError(f"UCI model config not found: {model_config}")
    if not hardware_config.exists():
        raise FileNotFoundError(f"UCI hardware config not found: {hardware_config}")

    include_keys = uci_train_validation._load_case_keys(include_csv) if include_csv else None
    exclude_keys = uci_train_validation._load_case_keys(exclude_csv) if exclude_csv else set()
    cases = uci_train_validation._load_cases(
        input_csv,
        tuple(args.uci_variants),
        include_keys=include_keys,
        exclude_keys=exclude_keys,
    )
    if not cases:
        raise ValueError("No UCI validation cases after filtering.")

    astra_modes = (
        list(args.uci_astra_modes)
        if args.uci_astra_modes
        else list(uci_train_validation.DEFAULT_ASTRA_MODES)
    )
    normalized_modes: List[str] = []
    seen_modes = set()
    for mode in astra_modes:
        normalized = uci_train_validation._normalize_astra_mode(mode)
        if normalized not in seen_modes:
            normalized_modes.append(normalized)
            seen_modes.add(normalized)
    if not normalized_modes:
        normalized_modes = list(uci_train_validation.DEFAULT_ASTRA_MODES)

    specs: List[ValidationSpec] = []
    case_order_keys: List[Tuple[str, str, str, str, str]] = [
        uci_train_validation._case_order_key(case) for case in cases
    ]
    for mode in normalized_modes:
        for idx, case in enumerate(cases):
            label = f"{case.variant.upper()} TP={case.tp} PP={case.pp} CP={case.cp} DP={case.dp}"
            local_step_batch = (
                uci_train_validation.GLOBAL_BATCH_SIZE
                // (uci_train_validation.MICRO_BATCH_SIZE * case.dp)
            )
            grad_acc_steps = 1 if int(case.pp) > 1 else int(local_step_batch)
            eff_mb = int(local_step_batch) if int(case.pp) > 1 else 1
            network_override, mapping_desc = uci_train_validation._build_network_override(
                case.tp,
                case.pp,
                case.cp,
                case.dp,
                mode,
            )
            model_overrides = {
                "model_param": {
                    "global_batch_size": uci_train_validation.GLOBAL_BATCH_SIZE,
                    "gradient_accumulation_steps": int(grad_acc_steps),
                    "micro_batch_size": uci_train_validation.MICRO_BATCH_SIZE,
                    "seq_len": 4096,
                    "run_type": "training",
                }
            }
            hw_overrides: Dict[str, object] = {
                "parallelism": {
                    "tp": int(case.tp),
                    "tp_sp": False,
                    "cp": int(case.cp),
                    "pp": int(case.pp),
                    "mb": int(eff_mb),
                    "train": {"dp": int(case.dp), "ep": 1, "tp_ep": True},
                    "inference": {"replica_count": 1, "moe_dp": 1},
                },
                "sw_param": {
                    "dp_zero_stage": 1 if case.variant.upper() == "DDP" else 3,
                    "activation_checkpointing": "selective",
                },
            }
            hw_overrides.update(network_override)

            metadata = {
                "suite": "uci_train",
                "device": "A100",
                "dataset": "training",
                "metric_kind": "training",
                "case_index": int(idx),
                "variant": case.variant,
                "tp": int(case.tp),
                "pp": int(case.pp),
                "cp": int(case.cp),
                "dp": int(case.dp),
                "actual_training_time_s": float(case.actual),
                "network_mapping": mapping_desc,
                "astra": uci_train_validation._normalize_astra_mode(mode),
            }
            specs.append(
                ValidationSpec(
                    label=label,
                    model_overrides=model_overrides,
                    hardware_overrides=hw_overrides,
                    model_config_path=str(model_config),
                    hardware_config_path=str(hardware_config),
                    metadata=metadata,
                    order=len(specs),
                )
            )

    def finalize(results: List[ValidationResult], suite_dir: Path, enable_plot: bool) -> Dict[str, Any]:
        suite_dir.mkdir(parents=True, exist_ok=True)
        rows: List[Dict[str, object]] = []
        for res in results:
            meta = res.spec.metadata or {}
            actual_time = float(meta.get("actual_training_time_s", float("nan")))
            training_time = (
                float(res.metrics.get("training_time_s", float("nan")))
                if res.success
                else float("nan")
            )
            if math.isnan(training_time) or math.isnan(actual_time) or actual_time == 0:
                signed_pct_error = float("nan")
                pct_error = float("nan")
                success = False
            else:
                signed_pct_error = (training_time - actual_time) / actual_time * 100.0
                pct_error = signed_pct_error
                success = True

            rows.append(
                {
                    "variant": meta.get("variant"),
                    "tp": meta.get("tp"),
                    "pp": meta.get("pp"),
                    "cp": meta.get("cp"),
                    "dp": meta.get("dp"),
                    "training_time_s": training_time,
                    "actual_training_time_s": actual_time,
                    "signed_pct_error": signed_pct_error,
                    "pct_error": pct_error,
                    "network_mapping": meta.get("network_mapping"),
                    "success": success,
                    "error": "" if res.success else (res.error or ""),
                    "raw_output": res.raw_output,
                    "astra": meta.get("astra"),
                }
            )

        output_csv = suite_dir / "uci_train_validation_result.csv"
        fieldnames: List[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    fieldnames.append(str(key))
                    seen.add(key)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        plot_paths: List[str] = []
        if enable_plot and stage_csv.exists():
            stage_rows = uci_train_validation._read_csv(stage_csv)
            labels, series = uci_train_validation._build_compare_series(
                rows,
                stage_rows,
                key_order=case_order_keys,
            )
            if labels:
                plot_output = suite_dir / "uci_train_validation_compare.png"
                out_plot = uci_train_validation._plot_compare_results(
                    list(reversed(labels)),
                    {name: list(reversed(values)) for name, values in series.items()},
                    plot_output,
                    compare_title,
                )
                plot_paths.append(str(out_plot))

        pct_values = [
            float(row["pct_error"])
            for row in rows
            if row.get("pct_error") is not None and math.isfinite(float(row["pct_error"]))
        ]
        success_count = sum(1 for row in rows if bool(row.get("success")))
        return {
            "suite": "uci_train",
            "cases_total": len(rows),
            "cases_success": success_count,
            "avg_abs_pct_error": _mean_abs(pct_values),
            "max_abs_pct_error": _max_abs(pct_values),
            "csv_paths": [str(output_csv)],
            "plot_paths": plot_paths,
        }

    return SuiteBundle(name="uci_train", specs=specs, finalize=finalize)


def _build_mosaic_train_bundle(args: argparse.Namespace) -> SuiteBundle:
    input_csv = _resolve_repo_relative_path(args.mosaic_input_csv)
    model_size = str(args.mosaic_model_size)
    model_config = (
        str(_resolve_repo_relative_path(args.mosaic_model_config))
        if args.mosaic_model_config
        else None
    )
    model_config_dir = _resolve_repo_relative_path(args.mosaic_model_config_dir)
    hardware_config = str(_resolve_repo_relative_path(args.mosaic_hardware_config))
    use_flashattention = not bool(args.mosaic_disable_flashattention)
    tile_size = int(args.mosaic_attention_tile_size) if args.mosaic_attention_tile_size is not None else None

    if not input_csv.exists():
        raise FileNotFoundError(f"Mosaic input CSV not found: {input_csv}")
    if not model_config_dir.exists():
        raise FileNotFoundError(f"Mosaic model config dir not found: {model_config_dir}")
    if not Path(hardware_config).exists():
        raise FileNotFoundError(f"Mosaic hardware config not found: {hardware_config}")

    cases = mosaic_train._load_cases(input_csv, model_size=model_size)
    specs, actual_lookup = mosaic_train.build_specs(
        cases,
        model_size_filter=model_size,
        explicit_model_config=model_config,
        model_config_dir=model_config_dir,
        hardware_config_path=hardware_config,
        use_flashattention=use_flashattention,
        attention_tile_size=tile_size,
    )
    specs = _augment_specs(
        specs,
        suite="mosaic_train",
        dataset="training",
        metric_kind="training",
        device="H100",
    )

    def finalize(results: List[ValidationResult], suite_dir: Path, enable_plot: bool) -> Dict[str, Any]:
        suite_dir.mkdir(parents=True, exist_ok=True)
        rows = mosaic_train.compute_rows(results, actual_lookup)

        output_csv = suite_dir / "mosiacml_h100_bf16_all.csv"
        mosaic_train._write_rows_csv(rows, output_csv)

        plot_paths: List[str] = []
        if enable_plot:
            model_size_label = "All MPT sizes" if mosaic_train._is_all_models(model_size) else str(model_size)
            heatmap_path = suite_dir / "mosiacml_h100_bf16_all.png"
            parity_path = mosaic_train._default_parity_plot_path(heatmap_path)
            heatmap = mosaic_train._plot_error_facet_heatmap(
                rows,
                "MosaicML H100 BF16 {}: RAPID-LLM vs inferred_total_latency_s".format(model_size_label),
                heatmap_path,
            )
            parity = mosaic_train._plot_parity_subsets(
                rows,
                "MosaicML H100 BF16 {}: parity (predicted vs actual)".format(model_size_label),
                parity_path,
            )
            if heatmap is not None:
                plot_paths.append(str(heatmap))
            if parity is not None:
                plot_paths.append(str(parity))

        pct_values = [
            float(row["abs_pct_error"])
            for row in rows
            if row.get("abs_pct_error") is not None and math.isfinite(float(row["abs_pct_error"]))
        ]
        success_count = sum(1 for row in rows if bool(row.get("success")))
        return {
            "suite": "mosaic_train",
            "cases_total": len(rows),
            "cases_success": success_count,
            "avg_abs_pct_error": _mean_abs(pct_values),
            "max_abs_pct_error": _max_abs(pct_values),
            "csv_paths": [str(output_csv)],
            "plot_paths": plot_paths,
        }

    return SuiteBundle(name="mosaic_train", specs=specs, finalize=finalize)


def _write_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified validation harness: run selected inference/training suites in one shared "
            "worker pool with YAML-defined shared and device-type derate overrides."
        )
    )
    parser.add_argument(
        "--suites",
        type=str,
        default=",".join(DEFAULT_SUITES),
        help=f"Comma-separated suites (default: {','.join(DEFAULT_SUITES)}).",
    )
    parser.add_argument("--workers", type=int, default=None, help="Shared worker count for all selected suites.")
    parser.add_argument("--cache-mode", type=str, default="NO_CACHE", help="Astra cache mode.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output root for harness runs.")
    parser.add_argument("--no-plot", dest="enable_plot", action="store_false", help="Disable plot generation.")
    parser.add_argument(
        "--no-progress",
        dest="show_progress",
        action="store_false",
        help="Disable progress bar for the shared validation run.",
    )
    parser.set_defaults(show_progress=True)

    parser.add_argument(
        "--derate-config",
        type=Path,
        default=DEFAULT_DERATE_CONFIG,
        help=f"YAML file with shared and device-type util derates (default: {DEFAULT_DERATE_CONFIG}).",
    )

    parser.add_argument(
        "--inference-devices",
        type=str,
        default="A100,H100",
        help="Comma-separated devices for nvidia_inf suite (default: A100,H100).",
    )

    parser.add_argument(
        "--nvidia-train-input-csv",
        type=Path,
        default=NVIDIA_TRAIN_DEFAULT_INPUT,
        help=f"NVIDIA train cases CSV (default: {NVIDIA_TRAIN_DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--nvidia-train-hardware-config",
        type=Path,
        default=NVIDIA_TRAIN_DEFAULT_HW,
        help=f"NVIDIA train hardware config (default: {NVIDIA_TRAIN_DEFAULT_HW}).",
    )
    parser.add_argument(
        "--nvidia-train-stage-csv",
        type=Path,
        default=NVIDIA_TRAIN_DEFAULT_STAGE,
        help=f"NVIDIA train STAGE CSV (default: {NVIDIA_TRAIN_DEFAULT_STAGE}).",
    )
    parser.add_argument(
        "--nvidia-train-title",
        type=str,
        default="Runtime comparison on large-scale system",
        help="NVIDIA train plot title.",
    )

    parser.add_argument(
        "--uci-hardware-config",
        type=Path,
        default=UCI_TRAIN_DEFAULT_HW,
        help=f"Hardware config used for the uci_train suite (default: {UCI_TRAIN_DEFAULT_HW}).",
    )
    parser.add_argument(
        "--uci-model-config",
        type=Path,
        default=uci_train_validation.DEFAULT_MODEL_CONFIG,
        help="Model config used for the uci_train suite (from uci_train_validation.py).",
    )
    parser.add_argument(
        "--uci-input-csv",
        type=Path,
        default=uci_train_validation.DEFAULT_INPUT_CSV,
        help="Input CSV for the uci_train suite (uci_train_validation.py format).",
    )
    parser.add_argument(
        "--uci-include-csv",
        type=Path,
        default=uci_train_validation.DEFAULT_INCLUDE_CSV,
        help="Optional include CSV for uci_train case filtering.",
    )
    parser.add_argument("--uci-exclude-csv", type=Path, default=None)
    parser.add_argument(
        "--uci-stage-csv",
        type=Path,
        default=uci_train_validation.TRAIN_VALIDATION_DATA / "uci_train_stg_mlsynth.csv",
        help="STAGE/MLSynth comparison CSV for uci_train_validation plotting.",
    )
    parser.add_argument(
        "--uci-variants",
        nargs="+",
        default=["ddp"],
        choices=["ddp", "fsdp"],
        help="UCI variants to include (uci_train_validation.py semantics).",
    )
    parser.add_argument(
        "--uci-astra-modes",
        nargs="+",
        default=None,
        help="Optional Astra modes for uci_train_validation.py behavior.",
    )

    parser.add_argument("--mosaic-input-csv", type=Path, default=mosaic_train.DEFAULT_INPUT_CSV)
    parser.add_argument("--mosaic-model-size", type=str, default="all")
    parser.add_argument("--mosaic-model-config", type=Path, default=None)
    parser.add_argument("--mosaic-model-config-dir", type=Path, default=mosaic_train.DEFAULT_MODEL_CONFIG_DIR)
    parser.add_argument("--mosaic-hardware-config", type=Path, default=MOSAIC_DEFAULT_HW)
    parser.add_argument(
        "--mosaic-disable-flashattention",
        action="store_true",
        help="Disable flash attention override for mosaic suite specs.",
    )
    parser.add_argument("--mosaic-attention-tile-size", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    suites = _parse_suite_list(args.suites)
    if "uci_train" in suites:
        _assert_uci_suite_module_binding()
    ordered_suites = _order_suites_for_submission(suites)
    inference_devices = (
        _parse_inference_devices(args.inference_devices)
        if "nvidia_inf" in ordered_suites
        else ["A100", "H100"]
    )
    derate_config = _load_derate_config(
        args.derate_config,
        selected_suites=ordered_suites,
        inference_devices=inference_devices,
    )
    run_root = args.output_root / _new_run_id()
    run_root.mkdir(parents=True, exist_ok=True)

    suite_builders: Dict[str, Callable[[argparse.Namespace], SuiteBundle]] = {
        "nvidia_inf": _build_nvidia_inf_bundle,
        "nvidia_train": _build_nvidia_train_bundle,
        "uci_train": _build_uci_train_bundle,
        "mosaic_train": _build_mosaic_train_bundle,
    }
    bundles: List[SuiteBundle] = [suite_builders[name](args) for name in ordered_suites]

    all_specs: List[ValidationSpec] = []
    segments: List[Tuple[SuiteBundle, int, int]] = []
    for bundle in bundles:
        start = len(all_specs)
        for spec in bundle.specs:
            all_specs.append(
                _apply_common_override(
                    spec,
                    suite_name=bundle.name,
                    derate_config=derate_config,
                )
            )
        end = len(all_specs)
        segments.append((bundle, start, end))

    if not all_specs:
        raise ValueError("No validation specs generated. Check --suites selection and inputs.")

    base_model = next((spec.model_config_path for spec in all_specs if spec.model_config_path), None)
    base_hw = next((spec.hardware_config_path for spec in all_specs if spec.hardware_config_path), None)
    if not base_model or not base_hw:
        raise ValueError("Unable to resolve base model/hardware config paths from generated specs.")

    print(f"[harness] Running {len(all_specs)} spec(s) across suites: {', '.join(ordered_suites)}")
    print(f"[harness] Output root: {run_root}")
    print(f"[harness] Derate config: {derate_config.path}")

    results = run_validation_suite(
        all_specs,
        base_model_config_path=str(base_model),
        base_hardware_config_path=str(base_hw),
        result_parser=_parse_runtime_metrics,
        run_perf_path=str(PROJECT_ROOT / "run_perf.py"),
        max_workers=args.workers,
        cache_mode=str(args.cache_mode),
        show_progress=bool(args.show_progress),
        progress_group_key="suite",
    )

    suite_summaries: List[Dict[str, Any]] = []
    for bundle, start, end in segments:
        suite_results = results[start:end]
        suite_dir = run_root / bundle.name
        summary = bundle.finalize(suite_results, suite_dir, bool(args.enable_plot))
        summary["output_dir"] = str(suite_dir)
        suite_summaries.append(summary)

    summary_csv_rows: List[Dict[str, Any]] = []
    for row in suite_summaries:
        summary_csv_rows.append(
            {
                "suite": row.get("suite"),
                "cases_total": row.get("cases_total"),
                "cases_success": row.get("cases_success"),
                "avg_abs_pct_error": row.get("avg_abs_pct_error"),
                "max_abs_pct_error": row.get("max_abs_pct_error"),
                "output_dir": row.get("output_dir"),
            }
        )
        per_device = row.get("per_device")
        if isinstance(per_device, list):
            for device_row in per_device:
                if not isinstance(device_row, Mapping):
                    continue
                device = str(device_row.get("device", "")).upper()
                summary_csv_rows.append(
                    {
                        "suite": f"{row.get('suite')}[{device}]" if device else row.get("suite"),
                        "parent_suite": row.get("suite"),
                        "device": device,
                        "cases_total": device_row.get("cases_total"),
                        "cases_success": device_row.get("cases_success"),
                        "avg_abs_pct_error": device_row.get("avg_abs_pct_error"),
                        "max_abs_pct_error": device_row.get("max_abs_pct_error"),
                        "output_dir": row.get("output_dir"),
                    }
                )
    summary_csv = run_root / "summary.csv"
    _write_summary_csv(summary_csv_rows, summary_csv)

    summary_json = run_root / "summary.json"
    suite_device_types = _suite_device_type_map(
        selected_suites=ordered_suites,
        inference_devices=inference_devices,
    )
    suite_compute_util = _suite_compute_util_from_device_type_derates(
        selected_suites=ordered_suites,
        inference_devices=inference_devices,
        derate_config=derate_config,
    )
    summary_payload = {
        "run_root": str(run_root),
        "suites": suite_summaries,
        "config": {
            "derate_config_path": str(derate_config.path),
            "shared_params": dict(derate_config.shared),
            "device_type_derates": copy.deepcopy(derate_config.device_type_derates),
            "suite_device_types": suite_device_types,
            "suite_compute_util": suite_compute_util,
            "suite_submission_order": list(ordered_suites),
            "workers": args.workers,
            "cache_mode": args.cache_mode,
            "show_progress": bool(args.show_progress),
            "enable_plot": bool(args.enable_plot),
        },
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("[harness] Suite summary:")
    for row in summary_csv_rows:
        avg_abs = row.get("avg_abs_pct_error")
        max_abs = row.get("max_abs_pct_error")
        avg_abs_text = "nan" if avg_abs is None or not math.isfinite(float(avg_abs)) else f"{float(avg_abs):.2f}%"
        max_abs_text = "nan" if max_abs is None or not math.isfinite(float(max_abs)) else f"{float(max_abs):.2f}%"
        print(
            f"  - {row.get('suite')}: avg_abs_pct_error={avg_abs_text}, "
            f"max_abs_pct_error={max_abs_text}, "
            f"success={row.get('cases_success')}/{row.get('cases_total')}"
        )
    print(f"[harness] Wrote summary CSV: {summary_csv}")
    print(f"[harness] Wrote summary JSON: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
