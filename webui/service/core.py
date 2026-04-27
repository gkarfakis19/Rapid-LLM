from __future__ import annotations

import copy
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import psutil
import yaml

import config as rapid_config


ROOT = Path(__file__).resolve().parents[2]
WEBUI_ROOT = ROOT / "webui"
WORKSPACE_ROOT = Path(os.environ.get("RAPID_WEBUI_WORKSPACE_ROOT", WEBUI_ROOT / "workspace")).expanduser().resolve()
RUNS_ROOT = WORKSPACE_ROOT / "runs"
SWEEPS_ROOT = WORKSPACE_ROOT / "sweeps"
CONFIGS_ROOT = WORKSPACE_ROOT / "configs"
LOCKS_ROOT = WORKSPACE_ROOT / "locks"
LOGS_ROOT = WORKSPACE_ROOT / "logs"
DB_ROOT = WORKSPACE_ROOT / "db"
ARTIFACTS_ROOT = WORKSPACE_ROOT / "artifacts"
PYTHON_BIN = Path(os.environ.get("RAPID_WEBUI_PYTHON_BIN", ROOT / ".venv" / "bin" / "python")).expanduser().resolve()
ACTIVE_JOB_LOCK = LOCKS_ROOT / "active_job.lock"
SCHEMA_VERSION_PATH = WORKSPACE_ROOT / "schema_version.json"
WORKER_MODULE = "webui.service.worker_runner"
DEFAULT_MODEL_CONFIG_SOURCES = [
    ROOT / "validation_scripts" / "validation_configs" / "model-config" / "Llama2-7B.yaml",
    ROOT / "validation_scripts" / "validation_configs" / "model-config" / "Llama2-7B_inf.yaml",
    ROOT / "validation_scripts" / "validation_configs" / "model-config" / "Llama3.1-70B_2d_train.yaml",
    ROOT / "validation_scripts" / "validation_configs" / "model-config" / "Llama3.1-70B_2d_inf.yaml",
    ROOT / "validation_scripts" / "validation_configs" / "model-config" / "Llama3.1-405B_2d_train.yaml",
    ROOT / "validation_scripts" / "validation_configs" / "model-config" / "Llama3.1-405B_2d_inf.yaml",
    ROOT / "validation_scripts" / "validation_configs" / "model-config" / "DeepSeekV3.yaml",
    ROOT / "validation_scripts" / "validation_configs" / "model-config" / "DeepSeekV3_inf_16k.yaml",
    ROOT / "validation_scripts" / "validation_configs" / "model-config" / "GLM4.7_358B_inf_16k.yaml",
]
DEFAULT_HARDWARE_CONFIG_SOURCES = [
    ROOT / "validation_scripts" / "validation_configs" / "hardware-config" / "H100_SXM5_80GB.yaml",
    ROOT / "validation_scripts" / "validation_configs" / "hardware-config" / "H100_SXM5_80GB_2d.yaml",
    ROOT / "validation_scripts" / "validation_configs" / "hardware-config" / "H100_SXM5_80GB_base.yaml",
    ROOT / "validation_scripts" / "validation_configs" / "hardware-config" / "A100_SXM4_80GB_base.yaml",
    ROOT / "configs" / "hardware-config" / "a100_80GB.yaml",
    ROOT / "configs" / "hardware-config" / "H100_SXM5_80GB_superpod.yaml",
]


FIELD_OPTIONS: List[Dict[str, str]] = [
    {"value": "model.seq_len", "label": "Sequence Length"},
    {"value": "model.decode_len", "label": "Decode Length"},
    {"value": "model.global_batch_size", "label": "Batch Size"},
    {"value": "model.gradient_accumulation_steps", "label": "Grad Accumulation"},
    {"value": "hardware.total_gpus", "label": "Total GPUs"},
    {"value": "hardware.hbm_gb", "label": "HBM Capacity"},
    {"value": "hardware.compute_derate", "label": "Compute Derate"},
    {"value": "hardware.memory_derate", "label": "Memory Derate"},
    {"value": "hardware.network_derate", "label": "Network Derate"},
    {"value": "hardware.gpu_clock_ghz", "label": "GPU Clock (GHz)"},
    {"value": "hardware.memory_bw_gbs", "label": "Memory BW (GB/s)"},
]

FIELD_TYPES: Dict[str, Dict[str, Any]] = {
    "model_config": {"kind": "config", "config_kind": "models", "label": "Model Config"},
    "hardware_config": {"kind": "config", "config_kind": "hardware", "label": "Hardware Config"},
    "model.seq_len": {"kind": "int"},
    "model.decode_len": {"kind": "int"},
    "model.global_batch_size": {"kind": "int"},
    "model.gradient_accumulation_steps": {"kind": "int"},
    "hardware.total_gpus": {"kind": "int"},
    "hardware.hbm_gb": {"kind": "float"},
    "hardware.compute_derate": {"kind": "float"},
    "hardware.memory_derate": {"kind": "float"},
    "hardware.network_derate": {"kind": "float"},
    "hardware.gpu_clock_ghz": {"kind": "float"},
    "hardware.memory_bw_gbs": {"kind": "float"},
}

METRIC_LABELS = {
    "training_time_s": "Time / Batch",
    "approx_mfu": "Approx. MFU",
    "prefill_time_s": "Prefill Time",
    "decode_throughput_tok_s": "Decode Throughput",
}
SUPPORTED_MODEL_TYPES = {"gpt", "llama", "deepseek_v3", "glm4_moe", "vit", "vit_dinov3"}

OPTIMIZER_PRESETS = {
    "Fast": {
        "training": {
            "tp": [4, 8, 16],
            "cp": [1, 2, 4],
            "dp": [1, 2, 4, 8, 16],
            "pp": [1, 2, 4, 8, 16],
            "ep": [1, 2, 4, 8, 16],
            "tp_cp_min": 1,
            "tp_cp_max": 512,
        },
        "inference": {
            "tp": [8, 16],
            "cp": [1, 2, 3, 4, 5],
            "pp": [1, 2, 3],
            "ep": [1, 2, 4],
            "tp_cp_min": 1,
            "tp_cp_max": 512,
        },
    },
    "Exhaustive": {
        "training": {
            "tp": [1, 2, 4, 8, 16, 32],
            "cp": [1, 2, 4, 8, 16],
            "dp": [1, 2, 4, 8, 16, 32],
            "pp": [1, 2, 4, 8, 16, 32],
            "ep": [1, 2, 4, 8, 16, 32],
            "tp_cp_min": None,
            "tp_cp_max": None,
        },
        "inference": {
            "tp": [1, 2, 4, 8, 16, 32],
            "cp": [1, 2, 3, 4, 5, 6, 7, 8],
            "pp": [1, 2, 3, 4, 5, 6, 7, 8],
            "ep": [1, 2, 4, 8, 16, 32],
            "tp_cp_min": None,
            "tp_cp_max": None,
        },
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_case_config_name(path: Path) -> bool:
    return bool(re.search(r"case[-_][a-z0-9]+$", path.stem, re.IGNORECASE))


def _seed_editable_configs() -> None:
    seed_groups = {
        "models": DEFAULT_MODEL_CONFIG_SOURCES,
        "hardware": DEFAULT_HARDWARE_CONFIG_SOURCES,
    }
    for kind, source_paths in seed_groups.items():
        target_dir = CONFIGS_ROOT / kind
        target_dir.mkdir(parents=True, exist_ok=True)
        for source in source_paths:
            if not source.exists() or _is_case_config_name(source):
                continue
            target = target_dir / source.name
            if not target.exists():
                shutil.copyfile(source, target)


def ensure_workspace() -> None:
    for path in [
        WORKSPACE_ROOT,
        RUNS_ROOT,
        SWEEPS_ROOT,
        CONFIGS_ROOT / "models",
        CONFIGS_ROOT / "hardware",
        LOCKS_ROOT,
        LOGS_ROOT,
        DB_ROOT,
        ARTIFACTS_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    _seed_editable_configs()
    if not SCHEMA_VERSION_PATH.exists():
        SCHEMA_VERSION_PATH.write_text(json.dumps({"schema_version": 1}, indent=2))


def _yaml_load(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def _yaml_dump(data: Dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False)


def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _json_load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _path_artifact_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return "config"
    if suffix == ".json":
        return "json"
    if suffix == ".log":
        return "log"
    if suffix in {".csv", ".tsv"}:
        return "table"
    if suffix in {".png", ".jpg", ".jpeg", ".svg", ".html"}:
        return "visual"
    return "file"


def prettify_name(stem: str) -> str:
    return stem.replace("_", " ")


def _config_filename_from_user_text(raw_name: str) -> str:
    text = Path(str(raw_name or "").strip()).name
    if not text:
        raise ValueError("Enter a config filename.")
    if text.endswith((".yaml", ".yml")):
        stem = Path(text).stem
        suffix = Path(text).suffix
    else:
        stem = text
        suffix = ".yaml"
    stem = re.sub(r"\s+", "_", stem.strip())
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", stem)
    stem = stem.strip("._-")
    if not stem:
        raise ValueError("Config filename must contain letters or numbers.")
    filename = f"{stem}{suffix}"
    if _is_case_config_name(Path(filename)):
        raise ValueError("Case-generated config names are reserved.")
    return filename


def config_file_path(kind: str, preset_id: str) -> Path:
    if kind not in {"models", "hardware"}:
        raise ValueError(f"Unknown config kind: {kind}")
    filename = Path(str(preset_id)).name
    if not filename.endswith((".yaml", ".yml")):
        raise ValueError(f"Config id must be a YAML filename: {preset_id}")
    return CONFIGS_ROOT / kind / filename


def list_presets(kind: str) -> List[Dict[str, Any]]:
    ensure_workspace()
    directory = CONFIGS_ROOT / kind
    records: List[Dict[str, Any]] = []
    for path in sorted(directory.glob("*.yaml")):
        if _is_case_config_name(path):
            continue
        data = _yaml_load(path)
        run_type = str(data.get("model_param", {}).get("run_type", "")).lower() if kind == "models" else ""
        records.append({"id": path.name, "label": prettify_name(path.stem), "path": str(path), "run_type": run_type})
    return records


def load_preset(kind: str, preset_id: str) -> Dict[str, Any]:
    ensure_workspace()
    path = config_file_path(kind, preset_id)
    if not path.exists():
        raise FileNotFoundError(f"Preset not found: {path}")
    return _yaml_load(path)


def create_config_copy(kind: str, source_id: str, new_name: str) -> str:
    ensure_workspace()
    source = config_file_path(kind, source_id)
    if not source.exists():
        raise FileNotFoundError(f"Preset not found: {source}")
    filename = _config_filename_from_user_text(new_name)
    target = config_file_path(kind, filename)
    if target.exists():
        raise FileExistsError(f"Config already exists: {filename}")
    shutil.copyfile(source, target)
    return filename


def rename_config_file(kind: str, preset_id: str, new_name: str) -> str:
    ensure_workspace()
    source = config_file_path(kind, preset_id)
    if not source.exists():
        raise FileNotFoundError(f"Preset not found: {source}")
    filename = _config_filename_from_user_text(new_name)
    target = config_file_path(kind, filename)
    if target.exists() and target != source:
        raise FileExistsError(f"Config already exists: {filename}")
    if target != source:
        source.rename(target)
    return filename


def parse_size_to_gb(raw: Any) -> float:
    if isinstance(raw, (int, float)):
        return float(raw) / (1024**3)
    if raw is None:
        return 0.0
    text = str(raw).strip()
    match = re.match(r"^\s*([+-]?(?:[0-9]*\.?[0-9]+)(?:[eE][+-]?[0-9]+)?)\s*([A-Za-z]+)?\s*$", text)
    if not match:
        raise ValueError(f"Could not parse size value '{raw}'")
    value = float(match.group(1))
    unit = (match.group(2) or "B").upper()
    scale = {"B": 1.0 / (1024**3), "KB": 1.0 / (1024**2), "MB": 1.0 / 1024.0, "GB": 1.0, "TB": 1024.0}
    if unit not in scale:
        raise ValueError(f"Unsupported size unit '{unit}'")
    return value * scale[unit]


def format_gb(value_gb: float) -> str:
    return f"{int(round(value_gb))} GB" if abs(value_gb - round(value_gb)) < 1e-9 else f"{value_gb:.2f} GB"


def _format_bandwidth_field(raw: Any) -> str:
    if isinstance(raw, (list, tuple)):
        return ", ".join(str(item) for item in raw)
    return "" if raw is None else str(raw)


def _coerce_network_bandwidth(raw: Any) -> Any:
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if len(parts) > 1:
            return parts
    return raw


def parse_bandwidth_to_gbs(raw: Any) -> float:
    if isinstance(raw, (int, float)):
        return float(raw) / 1e9
    if raw is None:
        return 0.0
    text = str(raw).strip()
    match = re.match(r"^\s*([+-]?(?:[0-9]*\.?[0-9]+)(?:[eE][+-]?[0-9]+)?)\s*([A-Za-z/]+)?\s*$", text)
    if not match:
        raise ValueError(f"Could not parse bandwidth value '{raw}'")
    value = float(match.group(1))
    unit = (match.group(2) or "B/S").upper()
    aliases = {"GB": 1.0, "GB/S": 1.0, "TB": 1000.0, "TB/S": 1000.0, "MB": 1.0 / 1000.0, "MB/S": 1.0 / 1000.0}
    return value * aliases[unit] if unit in aliases else value / 1e9


def get_model_run_type(model_dict: Dict[str, Any]) -> str:
    return str(model_dict.get("model_param", {}).get("run_type", "training")).lower()


def get_model_mode(model_dict: Dict[str, Any]) -> str:
    return str(model_dict.get("model_param", {}).get("mode", "LLM")).upper()


def get_model_type(model_dict: Dict[str, Any]) -> str:
    raw = str(model_dict.get("model_param", {}).get("model_type", "gpt")).strip().lower()
    return "glm4_moe" if raw in {"glm", "glm4"} else raw


def get_total_gpu_count(hw_dict: Dict[str, Any], run_type: str) -> int:
    parallelism = hw_dict.get("parallelism", {})
    tp = int(parallelism.get("tp", 1) or 1)
    cp = int(parallelism.get("cp", 1) or 1)
    pp = int(parallelism.get("pp", 1) or 1)
    train_block = parallelism.get("train", {}) or {}
    ep = int(train_block.get("ep", 1) or 1)
    if run_type == "inference":
        replica_count = int((parallelism.get("inference", {}) or {}).get("replica_count", 1) or 1)
        return max(1, tp * cp * pp * ep * replica_count)
    dp = int(train_block.get("dp", 1) or 1)
    return max(1, tp * cp * pp * ep * dp)


def get_default_metric_for_run_type(run_type: str) -> str:
    return "prefill_time_s" if run_type == "inference" else "training_time_s"


def get_metric_options(run_type: str) -> List[Dict[str, str]]:
    if run_type == "inference":
        return [
            {"value": "prefill_time_s", "label": METRIC_LABELS["prefill_time_s"]},
            {"value": "decode_throughput_tok_s", "label": METRIC_LABELS["decode_throughput_tok_s"]},
        ]
    return [
        {"value": "training_time_s", "label": METRIC_LABELS["training_time_s"]},
        {"value": "approx_mfu", "label": METRIC_LABELS["approx_mfu"]},
    ]


def dimension_label(field_key: str) -> str:
    for option in FIELD_OPTIONS:
        if option["value"] == field_key:
            return option["label"]
    if field_key in FIELD_TYPES and FIELD_TYPES[field_key].get("label"):
        return str(FIELD_TYPES[field_key]["label"])
    return field_key


def _initial_network_derate(hw_dict: Dict[str, Any]) -> float:
    dimensions = hw_dict.get("network", {}).get("dimensions", []) or []
    utils = []
    for dim in dimensions:
        topology = dim.get("topology", {}) or {}
        if topology.get("util") is not None:
            utils.append(float(topology["util"]))
    return sum(utils) / len(utils) if utils else 1.0


def build_form_defaults(model_preset_id: str, hardware_preset_id: str) -> Dict[str, Any]:
    model = load_preset("models", model_preset_id)
    hardware = load_preset("hardware", hardware_preset_id)
    run_type = get_model_run_type(model)
    parallelism = hardware.get("parallelism", {})
    train_block = parallelism.get("train", {}) or {}
    inference_block = parallelism.get("inference", {}) or {}
    dimensions = []
    for idx, dim in enumerate(hardware.get("network", {}).get("dimensions", []) or []):
        topology = dim.get("topology", {}) or {}
        dimensions.append(
            {
                "id": dim.get("id") or f"dim{idx}",
                "label": f"Dimension {idx}",
                "topology_type": str(topology.get("type", "Ring")),
                "bandwidth": _format_bandwidth_field(topology.get("bandwidth", "")),
                "util": float(topology.get("util", 1.0) or 1.0),
            }
        )
    return {
        "run_type": run_type,
        "model_yaml": _yaml_dump(model),
        "hardware_yaml": _yaml_dump(hardware),
        "simple": {
            "seq_len": int(model.get("model_param", {}).get("seq_len", 0) or 0),
            "decode_len": int(model.get("model_param", {}).get("decode_len", 0) or 0),
            "batch_size": int(model.get("model_param", {}).get("global_batch_size", 1) or 1),
            "grad_accum": int(model.get("model_param", {}).get("gradient_accumulation_steps", 1) or 1),
            "total_gpus": get_total_gpu_count(hardware, run_type),
            "tp": int(parallelism.get("tp", 1) or 1),
            "cp": int(parallelism.get("cp", 1) or 1),
            "pp": int(parallelism.get("pp", 1) or 1),
            "dp": int(train_block.get("dp", 1) or 1),
            "ep": int(train_block.get("ep", 1) or 1),
            "replica_count": int(inference_block.get("replica_count", 1) or 1),
            "hbm_gb": parse_size_to_gb(hardware.get("tech_param", {}).get("DRAM", {}).get("size", "0 GB")),
            "compute_derate": float(hardware.get("tech_param", {}).get("core", {}).get("util", 1.0) or 1.0),
            "memory_derate": float(hardware.get("tech_param", {}).get("DRAM", {}).get("util", 1.0) or 1.0),
            "network_derate": float(_initial_network_derate(hardware)),
            "gpu_clock_ghz": float(hardware.get("tech_param", {}).get("core", {}).get("operating_frequency", 0.0) or 0.0) / 1e9,
            "memory_bw_gbs": parse_bandwidth_to_gbs(hardware.get("tech_param", {}).get("DRAM", {}).get("bandwidth", 0.0)),
            "use_astrasim": str(hardware.get("execution_backend", {}).get("model", "analytical")).lower() == "astra",
        },
        "advanced": {
            "model_mode": get_model_mode(model),
            "model_type": get_model_type(model),
            "full_recomputation": bool(hardware.get("sw_param", {}).get("full_recomputation", False)),
            "dp_zero_stage": int(hardware.get("sw_param", {}).get("dp_zero_stage", 0) or 0),
            "tensor_format": str(hardware.get("sw_param", {}).get("precision", {}).get("tensor_format", "bf16")),
            "execution_backend": str(hardware.get("execution_backend", {}).get("model", "analytical")),
            "execution_mode": str(hardware.get("execution_backend", {}).get("astra", {}).get("mode", "full_astrasim_hierarchical")),
            "tied_embeddings": bool(model.get("model_param", {}).get("tied_embeddings", False)),
            "hidden_dim": int(model.get("model_param", {}).get("hidden_dim", 0) or 0),
            "intermediate_size": int(model.get("model_param", {}).get("intermediate_size", 0) or 0),
            "num_layers": int(model.get("model_param", {}).get("num_layers", 0) or 0),
            "vocab_size": int(model.get("model_param", {}).get("vocab_size", 0) or 0),
            "attention_type": str(model.get("model_param", {}).get("attention", {}).get("attention_type", "mha")),
            "num_heads": int(model.get("model_param", {}).get("attention", {}).get("num_heads", 0) or 0),
            "use_flashattention": bool(model.get("model_param", {}).get("attention", {}).get("use_flashattention", False)),
            "attention_tile_size": int(model.get("model_param", {}).get("attention", {}).get("attention_tile_size", 128) or 128),
            "num_experts": int(model.get("model_param", {}).get("moe", {}).get("num_experts", 1) or 1),
            "top_k": int(model.get("model_param", {}).get("moe", {}).get("top_k", 1) or 1),
            "moe_intermediate_size": int(model.get("model_param", {}).get("moe", {}).get("moe_intermediate_size", 0) or 0),
            "expert_imbalance_factor": float(model.get("model_param", {}).get("moe", {}).get("expert_imbalance_factor", 1.0) or 1.0),
        },
        "network_dimensions": dimensions,
    }


def _set_path(mapping: Dict[str, Any], path: Iterable[str], value: Any) -> None:
    cursor = mapping
    parts = list(path)
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _apply_model_overrides(model: Dict[str, Any], payload: Dict[str, Any]) -> None:
    simple = payload.get("simple", {}) or {}
    advanced = payload.get("advanced", {}) or {}
    run_type = str(simple.get("run_type") or get_model_run_type(model) or "training").lower()
    model_param = model.setdefault("model_param", {})
    attention = model_param.setdefault("attention", {})
    moe = model_param.setdefault("moe", {})
    if advanced.get("model_mode"):
        model_param["mode"] = str(advanced["model_mode"]).strip().upper()
    if advanced.get("model_type"):
        model_type = str(advanced["model_type"]).strip().lower()
        if model_type in {"glm", "glm4"}:
            model_type = "glm4_moe"
        if model_type in SUPPORTED_MODEL_TYPES:
            model_param["model_type"] = model_type
    model_param["run_type"] = run_type
    model_param["seq_len"] = _safe_int(simple.get("seq_len"), _safe_int(model_param.get("seq_len"), 0))
    if run_type == "inference":
        model_param["decode_len"] = _safe_int(simple.get("decode_len"), _safe_int(model_param.get("decode_len"), 0))
    else:
        model_param.pop("decode_len", None)
    model_param["global_batch_size"] = _safe_int(simple.get("batch_size"), _safe_int(model_param.get("global_batch_size"), 1))
    model_param["gradient_accumulation_steps"] = _safe_int(simple.get("grad_accum"), _safe_int(model_param.get("gradient_accumulation_steps"), 1))
    model_param["tied_embeddings"] = bool(advanced.get("tied_embeddings", model_param.get("tied_embeddings", False)))
    if _safe_int(advanced.get("hidden_dim"), 0) > 0:
        model_param["hidden_dim"] = _safe_int(advanced.get("hidden_dim"), _safe_int(model_param.get("hidden_dim"), 0))
    if _safe_int(advanced.get("intermediate_size"), 0) > 0:
        model_param["intermediate_size"] = _safe_int(advanced.get("intermediate_size"), _safe_int(model_param.get("intermediate_size"), 0))
    if _safe_int(advanced.get("num_layers"), 0) > 0:
        model_param["num_layers"] = _safe_int(advanced.get("num_layers"), _safe_int(model_param.get("num_layers"), 0))
    if _safe_int(advanced.get("vocab_size"), 0) > 0:
        model_param["vocab_size"] = _safe_int(advanced.get("vocab_size"), _safe_int(model_param.get("vocab_size"), 0))
    if advanced.get("attention_type"):
        attention["attention_type"] = advanced["attention_type"]
    if _safe_int(advanced.get("num_heads"), 0) > 0:
        attention["num_heads"] = _safe_int(advanced.get("num_heads"), _safe_int(attention.get("num_heads"), 0))
    attention["use_flashattention"] = bool(advanced.get("use_flashattention", attention.get("use_flashattention", False)))
    attention["attention_tile_size"] = _safe_int(advanced.get("attention_tile_size"), _safe_int(attention.get("attention_tile_size"), 128))
    if _safe_int(advanced.get("num_experts"), 0) > 0:
        moe["num_experts"] = _safe_int(advanced.get("num_experts"), _safe_int(moe.get("num_experts"), 1))
    if _safe_int(advanced.get("top_k"), 0) > 0:
        moe["top_k"] = _safe_int(advanced.get("top_k"), _safe_int(moe.get("top_k"), 1))
    if _safe_int(advanced.get("moe_intermediate_size"), 0) > 0:
        moe["moe_intermediate_size"] = _safe_int(advanced.get("moe_intermediate_size"), _safe_int(moe.get("moe_intermediate_size"), 0))
    moe["expert_imbalance_factor"] = _safe_float(advanced.get("expert_imbalance_factor"), _safe_float(moe.get("expert_imbalance_factor"), 1.0))


def _apply_hardware_overrides(hardware: Dict[str, Any], payload: Dict[str, Any]) -> None:
    simple = payload.get("simple", {}) or {}
    advanced = payload.get("advanced", {}) or {}
    sw_param = hardware.setdefault("sw_param", {})
    precision = sw_param.setdefault("precision", {})
    sw_param["full_recomputation"] = bool(advanced.get("full_recomputation", sw_param.get("full_recomputation", False)))
    sw_param["dp_zero_stage"] = _safe_int(advanced.get("dp_zero_stage"), _safe_int(sw_param.get("dp_zero_stage"), 0))
    if advanced.get("tensor_format"):
        precision["tensor_format"] = advanced["tensor_format"]
    tech = hardware.setdefault("tech_param", {})
    core = tech.setdefault("core", {})
    dram = tech.setdefault("DRAM", {})
    core["util"] = _safe_float(simple.get("compute_derate"), _safe_float(core.get("util"), 1.0))
    dram["util"] = _safe_float(simple.get("memory_derate"), _safe_float(dram.get("util"), 1.0))
    core["operating_frequency"] = _safe_float(simple.get("gpu_clock_ghz"), 0.0) * 1e9
    dram["size"] = format_gb(_safe_float(simple.get("hbm_gb"), parse_size_to_gb(dram.get("size", "0 GB"))))
    dram["bandwidth"] = _safe_float(simple.get("memory_bw_gbs"), parse_bandwidth_to_gbs(dram.get("bandwidth", 0.0))) * 1e9

    parallelism = hardware.setdefault("parallelism", {})
    parallelism["tp"] = _safe_int(simple.get("tp"), _safe_int(parallelism.get("tp"), 1))
    parallelism["cp"] = _safe_int(simple.get("cp"), _safe_int(parallelism.get("cp"), 1))
    parallelism["pp"] = _safe_int(simple.get("pp"), _safe_int(parallelism.get("pp"), 1))
    train_block = parallelism.setdefault("train", {})
    inference_block = parallelism.setdefault("inference", {})
    train_block["dp"] = _safe_int(simple.get("dp"), _safe_int(train_block.get("dp"), 1))
    train_block["ep"] = _safe_int(simple.get("ep"), _safe_int(train_block.get("ep"), 1))
    inference_block["replica_count"] = _safe_int(simple.get("replica_count"), _safe_int(inference_block.get("replica_count"), 1))
    execution_backend = hardware.setdefault("execution_backend", {})
    execution_backend["model"] = "astra" if bool(simple.get("use_astrasim", False)) else "analytical"
    astra = execution_backend.setdefault("astra", {})
    astra["mode"] = "full_astrasim_hierarchical"

    network = hardware.setdefault("network", {})
    dimensions = network.setdefault("dimensions", [])
    network_derate = _safe_float(simple.get("network_derate"), _initial_network_derate(hardware))
    for idx, row in enumerate(payload.get("network_dimensions", []) or []):
        if idx >= len(dimensions):
            continue
        topology = dimensions[idx].setdefault("topology", {})
        if row.get("topology_type"):
            topology["type"] = row["topology_type"]
        if row.get("bandwidth") not in (None, ""):
            topology["bandwidth"] = _coerce_network_bandwidth(row["bandwidth"])
        topology["util"] = _safe_float(row.get("util"), network_derate)
    for idx in range(len(payload.get("network_dimensions", []) or []), len(dimensions)):
        topology = dimensions[idx].setdefault("topology", {})
        topology["util"] = network_derate


def _apply_form_overrides(model: Dict[str, Any], hardware: Dict[str, Any], payload: Dict[str, Any]) -> None:
    _apply_model_overrides(model, payload)
    _apply_hardware_overrides(hardware, payload)


def _validate_pair(model_dict: Dict[str, Any], hardware_dict: Dict[str, Any]) -> None:
    tmp_dir = WORKSPACE_ROOT / ".validation_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    model_path = tmp_dir / f"model-{uuid.uuid4().hex}.yaml"
    hw_path = tmp_dir / f"hw-{uuid.uuid4().hex}.yaml"
    try:
        model_path.write_text(_yaml_dump(model_dict))
        hw_path.write_text(_yaml_dump(hardware_dict))
        mode = get_model_mode(model_dict)
        hw_cfg = rapid_config.parse_config(str(hw_path), config_type="hardware")
        model_cfg = rapid_config.parse_config(str(model_path), config_type=mode)
        rapid_config.validate_configs(hw_cfg, model_cfg)
    finally:
        model_path.unlink(missing_ok=True)
        hw_path.unlink(missing_ok=True)


def build_editable_configs_from_payload(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[str]]:
    model_preset_id = payload.get("model_preset_id")
    hardware_preset_id = payload.get("hardware_preset_id")
    if not model_preset_id or not hardware_preset_id:
        return None, None, ["Model and hardware presets are required."]
    try:
        model = load_preset("models", model_preset_id)
        hardware = load_preset("hardware", hardware_preset_id)
    except (FileNotFoundError, ValueError) as exc:
        return None, None, [str(exc)]
    _apply_form_overrides(model, hardware, payload)
    return model, hardware, []


def render_editable_config_texts(payload: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    model, hardware, errors = build_editable_configs_from_payload(payload)
    if model is None or hardware is None:
        return "", "", errors
    return _yaml_dump(model), _yaml_dump(hardware), errors


def save_config_edits_from_payload(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[str]]:
    model, hardware, errors = build_editable_configs_from_payload(payload)
    if model is None or hardware is None:
        return model, hardware, errors
    try:
        model_path = config_file_path("models", payload["model_preset_id"])
        hardware_path = config_file_path("hardware", payload["hardware_preset_id"])
        model_path.write_text(_yaml_dump(model))
        hardware_path.write_text(_yaml_dump(hardware))
    except (OSError, ValueError) as exc:
        return None, None, [str(exc)]
    return model, hardware, []


def save_config_edits_for_selection(payload: Dict[str, Any], model_ids: List[str] | None, hardware_ids: List[str] | None) -> List[str]:
    primary_model_id = payload.get("model_preset_id")
    primary_hardware_id = payload.get("hardware_preset_id")
    selected_models = list(dict.fromkeys([item for item in (model_ids or [primary_model_id]) if item]))
    selected_hardware = list(dict.fromkeys([item for item in (hardware_ids or [primary_hardware_id]) if item]))
    if not selected_models or not selected_hardware:
        return ["Select at least one model YAML and one hardware YAML."]
    errors: List[str] = []
    for model_id in selected_models:
        edit_payload = {**payload, "model_preset_id": model_id, "hardware_preset_id": primary_hardware_id}
        model, _, model_errors = build_editable_configs_from_payload(edit_payload)
        if model is None or model_errors:
            errors.extend(model_errors)
            continue
        try:
            config_file_path("models", model_id).write_text(_yaml_dump(model))
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    for hardware_id in selected_hardware:
        edit_payload = {**payload, "model_preset_id": primary_model_id, "hardware_preset_id": hardware_id}
        _, hardware, hardware_errors = build_editable_configs_from_payload(edit_payload)
        if hardware is None or hardware_errors:
            errors.extend(hardware_errors)
            continue
        try:
            config_file_path("hardware", hardware_id).write_text(_yaml_dump(hardware))
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    return errors


def build_configs_from_payload(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    model, hardware, build_errors = build_editable_configs_from_payload(payload)
    if model is None or hardware is None:
        return None, None, build_errors
    try:
        _validate_pair(model, hardware)
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    return model, hardware, errors


def _parse_list_values(raw_text: str, field_key: str) -> List[Any]:
    field_type = FIELD_TYPES[field_key]
    parts = [part.strip() for part in str(raw_text or "").split(",") if part.strip()]
    if field_type["kind"] == "int":
        return [int(part) for part in parts]
    if field_type["kind"] == "float":
        return [float(part) for part in parts]
    return parts


def _parse_dimension_values(dim: Dict[str, Any]) -> List[Any]:
    field_key = dim.get("field_key")
    if not field_key:
        return []
    field_type = FIELD_TYPES.get(field_key, {"kind": "str"})
    mode = dim.get("mode") or "values"
    if field_type.get("kind") == "config":
        return list(dim.get("config_values") or [])
    if mode in {"list", "values", "points"}:
        return _parse_list_values(dim.get("list_text", ""), field_key)
    if mode in {"step", "range"}:
        start = float(dim.get("start"))
        end = float(dim.get("end"))
        step = float(dim.get("step"))
        if step <= 0:
            raise ValueError(f"{dimension_label(field_key)} range step must be greater than 0.")
        values, current = [], start
        while current <= end + 1e-12:
            values.append(current)
            current += step
        return [int(round(item)) for item in values] if field_type["kind"] == "int" else values
    if mode == "linspace":
        start = float(dim.get("start"))
        end = float(dim.get("end"))
        points = max(1, int(dim.get("points") or 1))
        values = [start] if points == 1 else [start + idx * (end - start) / (points - 1) for idx in range(points)]
        return [int(round(item)) for item in values] if field_type["kind"] == "int" else values
    return []


def _scale_parallelism_to_total_gpus(hardware: Dict[str, Any], run_type: str, target_total_gpus: int) -> None:
    parallelism = hardware.setdefault("parallelism", {})
    tp = _safe_int(parallelism.get("tp"), 1)
    cp = _safe_int(parallelism.get("cp"), 1)
    pp = _safe_int(parallelism.get("pp"), 1)
    train_block = parallelism.setdefault("train", {})
    ep = _safe_int(train_block.get("ep"), 1)
    fixed_product = max(1, tp * cp * pp * ep)
    if target_total_gpus % fixed_product != 0:
        raise ValueError(
            f"Total GPUs={target_total_gpus} is not divisible by fixed TP*CP*PP*EP={fixed_product}. "
            "Enable Optimize parallelism or change the fixed parallelism axes."
        )
    scaled_axis = max(1, target_total_gpus // fixed_product)
    if run_type == "inference":
        parallelism.setdefault("inference", {})["replica_count"] = scaled_axis
    else:
        train_block["dp"] = scaled_axis


def _apply_scalar_dimension(model: Dict[str, Any], hardware: Dict[str, Any], field_key: str, value: Any, case_meta: Dict[str, Any], optimize_parallelism: bool = False) -> None:
    mapping = {
        "model.seq_len": (model, ("model_param", "seq_len"), int(value)),
        "model.decode_len": (model, ("model_param", "decode_len"), int(value)),
        "model.global_batch_size": (model, ("model_param", "global_batch_size"), int(value)),
        "model.gradient_accumulation_steps": (model, ("model_param", "gradient_accumulation_steps"), int(value)),
        "hardware.hbm_gb": (hardware, ("tech_param", "DRAM", "size"), format_gb(float(value))),
        "hardware.compute_derate": (hardware, ("tech_param", "core", "util"), float(value)),
        "hardware.memory_derate": (hardware, ("tech_param", "DRAM", "util"), float(value)),
        "hardware.gpu_clock_ghz": (hardware, ("tech_param", "core", "operating_frequency"), float(value) * 1e9),
        "hardware.memory_bw_gbs": (hardware, ("tech_param", "DRAM", "bandwidth"), float(value) * 1e9),
    }
    if field_key == "hardware.total_gpus":
        target_total_gpus = int(value)
        case_meta["target_total_gpus"] = target_total_gpus
        if not optimize_parallelism:
            _scale_parallelism_to_total_gpus(hardware, get_model_run_type(model), target_total_gpus)
        return
    if field_key == "hardware.network_derate":
        for dim in hardware.get("network", {}).get("dimensions", []) or []:
            dim.setdefault("topology", {})["util"] = float(value)
        return
    target = mapping.get(field_key)
    if target:
        _set_path(target[0], target[1], target[2])


def build_case_label(values: Dict[str, Any]) -> str:
    if not values:
        return "Base Case"
    return " | ".join(f"{dimension_label(key)}={value}" for key, value in values.items())


def default_worker_count() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, 8))


def generate_parallelism_candidates(hardware_dict: Dict[str, Any], run_type: str, target_total_gpus: int, preset_name: str) -> List[Dict[str, Any]]:
    preset = OPTIMIZER_PRESETS.get(preset_name, OPTIMIZER_PRESETS["Fast"])[run_type]
    candidates = []
    for tp, cp, pp, ep in itertools.product(preset["tp"], preset["cp"], preset["pp"], preset["ep"]):
        tp_cp = tp * cp
        if preset["tp_cp_min"] is not None and tp_cp < preset["tp_cp_min"]:
            continue
        if preset["tp_cp_max"] is not None and tp_cp > preset["tp_cp_max"]:
            continue
        if run_type == "training":
            for dp in preset["dp"]:
                total = tp * cp * pp * ep * dp
                if total == target_total_gpus:
                    candidates.append({"tp": tp, "cp": cp, "pp": pp, "dp": dp, "ep": ep, "replica_count": 1})
        else:
            total = tp * cp * pp * ep
            if total == target_total_gpus:
                candidates.append({"tp": tp, "cp": cp, "pp": pp, "dp": 1, "ep": ep, "replica_count": 1})
    return candidates


def apply_parallelism_candidate(hardware_dict: Dict[str, Any], candidate: Dict[str, int]) -> Dict[str, Any]:
    updated = copy.deepcopy(hardware_dict)
    parallelism = updated.setdefault("parallelism", {})
    parallelism["tp"] = candidate["tp"]
    parallelism["cp"] = candidate["cp"]
    parallelism["pp"] = candidate["pp"]
    parallelism.setdefault("train", {})["dp"] = candidate["dp"]
    parallelism.setdefault("train", {})["ep"] = candidate["ep"]
    parallelism.setdefault("inference", {})["replica_count"] = candidate.get("replica_count", 1)
    return updated


def build_launch_preview(payload: Dict[str, Any]) -> Dict[str, Any]:
    model, hardware, base_errors = build_configs_from_payload(payload)
    if model is None or hardware is None:
        return {"ok": False, "errors": base_errors, "warnings": [], "top_level_cases": []}
    run_mode = str(payload.get("run_mode") or "sweep").lower()
    single_run_mode = run_mode == "single"
    optimize = False if single_run_mode else bool(payload.get("optimize_parallelism"))
    dimensions = []
    if not single_run_mode:
        for dim in payload.get("dimensions", []) or []:
            field_key = dim.get("field_key")
            if not field_key:
                continue
            if field_key not in FIELD_TYPES:
                base_errors.append(f"Unsupported sweep field: {field_key}")
                continue
            try:
                values = _parse_dimension_values(dim)
            except Exception as exc:  # noqa: BLE001
                base_errors.append(f"{dimension_label(field_key)} sweep values are invalid: {exc}")
                continue
            if values:
                dimensions.append({**dim, "values": values})
    run_type = get_model_run_type(model)
    errors = list(base_errors)
    warnings: List[str] = []
    if single_run_mode and (payload.get("dimensions") or payload.get("optimize_parallelism")):
        warnings.append("Single launch mode ignores sweep dimensions and parallelism optimization.")
    if any(dim.get("field_key") == "model_config" for dim in dimensions):
        run_types = []
        for dim in dimensions:
            if dim.get("field_key") == "model_config":
                for preset_name in dim["values"]:
                    run_types.append(get_model_run_type(load_preset("models", preset_name)))
        if len(set(run_types)) > 1:
            errors.append("Mixed training and inference model sweeps are not supported in one run.")
            return {"ok": False, "errors": errors, "warnings": warnings, "top_level_cases": []}
        if run_types:
            run_type = run_types[0]

    base_model = copy.deepcopy(model)
    base_hw = copy.deepcopy(hardware)
    base_target_gpus = payload.get("simple", {}).get("total_gpus") or get_total_gpu_count(base_hw, run_type)
    dimension_values = [dim["values"] for dim in dimensions]
    top_level_cases, invalid_cases = [], []
    combos = itertools.product(*dimension_values) if dimension_values else [()]
    for combo_index, combo in enumerate(combos):
        case_model, case_hw = copy.deepcopy(base_model), copy.deepcopy(base_hw)
        case_meta: Dict[str, Any] = {"target_total_gpus": int(base_target_gpus), "dimension_values": {}, "index": combo_index}
        try:
            for dim, value in zip(dimensions, combo):
                field_key = dim["field_key"]
                case_meta["dimension_values"][field_key] = value
                if field_key == "model_config":
                    case_model = load_preset("models", value)
                    if value == payload.get("model_preset_id"):
                        _apply_model_overrides(case_model, payload)
                elif field_key == "hardware_config":
                    case_hw = load_preset("hardware", value)
                    if value == payload.get("hardware_preset_id"):
                        _apply_hardware_overrides(case_hw, payload)
                else:
                    _apply_scalar_dimension(case_model, case_hw, field_key, value, case_meta, optimize)
            _validate_pair(case_model, case_hw)
            top_level_cases.append({
                "case_id": f"case-{combo_index + 1:04d}",
                "label": build_case_label(case_meta["dimension_values"]),
                "model": case_model,
                "hardware": case_hw,
                "run_type": get_model_run_type(case_model),
                "target_total_gpus": int(case_meta["target_total_gpus"]),
                "dimension_values": case_meta["dimension_values"],
            })
        except Exception as exc:  # noqa: BLE001
            invalid_cases.append({"label": build_case_label(case_meta["dimension_values"]), "error": str(exc)})
    if invalid_cases:
        warnings.append(f"Pruned {len(invalid_cases)} invalid case(s) before launch.")
    metric = payload.get("metric") or get_default_metric_for_run_type(run_type)
    timeout_s = _safe_int(payload.get("timeout_seconds"), 180)
    workers = _safe_int(payload.get("worker_count"), default_worker_count())
    total_invocations, candidate_breakdown = 0, []
    if optimize:
        for case in top_level_cases:
            candidates = generate_parallelism_candidates(case["hardware"], case["run_type"], case["target_total_gpus"], payload.get("optimizer_preset") or "Fast")
            candidate_breakdown.append({"case_id": case["case_id"], "count": len(candidates)})
            total_invocations += len(candidates)
        if total_invocations == 0 and top_level_cases:
            warnings.append("Parallelism optimization produced zero feasible candidates across all cases.")
    else:
        total_invocations = len(top_level_cases)
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings + (["This launch is large. Expect long wall-clock time and heavy local resource usage."] if total_invocations > 256 else []),
        "run_type": run_type,
        "metric": metric,
        "optimizer_enabled": optimize,
        "optimizer_preset": payload.get("optimizer_preset") or "Fast",
        "worker_count": workers,
        "timeout_seconds": timeout_s,
        "top_level_case_count": len(top_level_cases),
        "candidate_breakdown": candidate_breakdown,
        "total_invocations": total_invocations,
        "worst_case_wall_clock_s": math.ceil(total_invocations / max(1, workers)) * timeout_s if total_invocations else 0,
        "invalid_cases": invalid_cases[:20],
        "top_level_cases": top_level_cases,
    }


def get_telemetry() -> Dict[str, Any]:
    vm = psutil.virtual_memory()
    return {"available_ram_gb": round(vm.available / (1024**3), 1), "used_percent": round(vm.percent, 1), "cpu_percent": round(psutil.cpu_percent(interval=None), 1)}


def _detail_root(job_kind: str) -> Path:
    return SWEEPS_ROOT if job_kind == "sweep" else RUNS_ROOT


def _job_summary_from_dir(path: Path, job_kind: str) -> Dict[str, Any]:
    status = _json_load(path / "status.json") if (path / "status.json").exists() else {}
    summary = _json_load(path / "summary.json") if (path / "summary.json").exists() else {}
    request = _json_load(path / "request.json") if (path / "request.json").exists() else {}
    return {
        "id": path.name,
        "kind": job_kind,
        "status": status.get("status", "unknown"),
        "created_at": status.get("created_at") or request.get("created_at"),
        "updated_at": status.get("updated_at") or status.get("created_at"),
        "title": request.get("title") or summary.get("title") or path.name,
        "metric": summary.get("best_metric_label") or summary.get("primary_metric_label"),
        "metric_value": summary.get("best_metric_value") or summary.get("primary_metric_value"),
        "summary": summary,
        "request": request,
    }


def list_history(limit: int = 50) -> List[Dict[str, Any]]:
    ensure_workspace()
    entries = []
    for root, kind in ((RUNS_ROOT, "run"), (SWEEPS_ROOT, "sweep")):
        for path in root.iterdir():
            if path.is_dir():
                entries.append(_job_summary_from_dir(path, kind))
    entries.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    return entries[:limit]


def load_job_detail(job_kind: str, job_id: str) -> Dict[str, Any]:
    path = _detail_root(job_kind) / job_id
    detail = _job_summary_from_dir(path, job_kind)
    detail["status_record"] = _json_load(path / "status.json") if (path / "status.json").exists() else {}
    detail["request_record"] = _json_load(path / "request.json") if (path / "request.json").exists() else {}
    detail["summary_record"] = _json_load(path / "summary.json") if (path / "summary.json").exists() else {}
    if job_kind == "sweep":
        cases_dir = path / "cases"
        detail["cases"] = [_json_load(case_path) for case_path in sorted(cases_dir.glob("*.json"))] if cases_dir.exists() else []
    else:
        detail["result"] = _json_load(path / "result.json") if (path / "result.json").exists() else {}
    return detail


def build_job_title(payload: Dict[str, Any], preview: Dict[str, Any]) -> str:
    model_label = prettify_name(Path(payload.get("model_preset_id", "model")).stem)
    hw_label = prettify_name(Path(payload.get("hardware_preset_id", "hardware")).stem)
    return f"{model_label} on {hw_label} Sweep" if preview.get("top_level_case_count", 0) > 1 or preview.get("optimizer_enabled") else f"{model_label} on {hw_label}"


def is_metric_better(metric: str, candidate: float, incumbent: float) -> bool:
    return float(candidate) > float(incumbent) if metric in {"decode_throughput_tok_s", "approx_mfu"} else float(candidate) < float(incumbent)


def metric_sort_key(metric: str, value: Optional[float]) -> Tuple[int, float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return (1, float("inf"))
    return (0, -float(value)) if metric in {"decode_throughput_tok_s", "approx_mfu"} else (0, float(value))


def pick_best_result(results: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    return sorted(results, key=lambda item: metric_sort_key(metric, item.get("metrics", {}).get(metric)))[0]


def pick_fallback_result(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    for item in results:
        if item.get("status") in {"timed_out", "failed"}:
            return item
    return results[0]


@dataclass
class ActiveProcess:
    process: subprocess.Popen
    log_path: Path
    err_path: Path


class RunManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cancel_event = threading.Event()
        self._active_job: Optional[Dict[str, Any]] = None
        self._thread: Optional[threading.Thread] = None
        self._processes: Dict[str, ActiveProcess] = {}

    def active_job(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._active_job)

    def _existing_active_lock(self) -> Optional[Dict[str, Any]]:
        if not ACTIVE_JOB_LOCK.exists():
            return None
        try:
            record = json.loads(ACTIVE_JOB_LOCK.read_text())
        except (OSError, json.JSONDecodeError):
            ACTIVE_JOB_LOCK.unlink(missing_ok=True)
            return None
        pid = _safe_int(record.get("pid"), -1)
        if pid == os.getpid() and not self._active_job:
            ACTIVE_JOB_LOCK.unlink(missing_ok=True)
            return None
        if pid > 0 and psutil.pid_exists(pid):
            return record
        ACTIVE_JOB_LOCK.unlink(missing_ok=True)
        return None

    def _write_expanded_cases(self, job_root: Path, preview: Dict[str, Any]) -> None:
        case_lines = []
        for case in preview.get("top_level_cases", []) or []:
            case_lines.append(
                json.dumps(
                    {
                        "case_id": case["case_id"],
                        "label": case["label"],
                        "run_type": case["run_type"],
                        "target_total_gpus": case.get("target_total_gpus"),
                        "dimension_values": case.get("dimension_values", {}),
                    },
                    sort_keys=True,
                )
            )
        if case_lines:
            (job_root / "expanded_cases.jsonl").write_text("\n".join(case_lines) + "\n")

    def _write_artifacts_manifest(self, job_root: Path) -> None:
        entries = []
        for path in sorted(job_root.rglob("*")):
            if not path.is_file() or path.name == "artifacts.json":
                continue
            rel_path = path.relative_to(job_root).as_posix()
            entries.append(
                {
                    "artifact_id": rel_path.replace("/", "__"),
                    "path": rel_path,
                    "type": _path_artifact_type(path),
                    "label": path.name,
                    "job_id": job_root.name,
                }
            )
        _json_dump(job_root / "artifacts.json", {"job_id": job_root.name, "artifacts": entries})

    def start_job(self, payload: Dict[str, Any], preview: Dict[str, Any]) -> Tuple[bool, str]:
        with self._lock:
            if self._active_job and self._active_job.get("status") in {"queued", "running", "cancel_requested"}:
                return False, "Another job is already running."
            existing_lock = self._existing_active_lock()
            if existing_lock:
                return False, f"Another job appears active from process {existing_lock.get('pid')}."
            self._cancel_event.clear()
            job_kind = "sweep" if preview.get("top_level_case_count", 0) > 1 or preview.get("optimizer_enabled") else "run"
            job_id = f"{job_kind}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
            job_root = _detail_root(job_kind) / job_id
            job_root.mkdir(parents=True, exist_ok=True)
            request_record = {
                "id": job_id,
                "kind": job_kind,
                "title": build_job_title(payload, preview),
                "created_at": utc_now(),
                "payload": payload,
                "preview": {key: value for key, value in preview.items() if key != "top_level_cases"},
            }
            status_record = {"status": "queued", "created_at": request_record["created_at"], "updated_at": request_record["created_at"], "progress_completed": 0, "progress_total": preview.get("total_invocations", 0)}
            _json_dump(job_root / "request.json", request_record)
            _json_dump(job_root / "status.json", status_record)
            (job_root / "artifacts").mkdir(parents=True, exist_ok=True)
            self._write_expanded_cases(job_root, preview)
            if job_kind == "sweep":
                _json_dump(job_root / "launch_preview.json", request_record["preview"])
                (job_root / "cases").mkdir(parents=True, exist_ok=True)
            ACTIVE_JOB_LOCK.write_text(json.dumps({"job_id": job_id, "kind": job_kind, "pid": os.getpid(), "created_at": utc_now()}))
            self._active_job = {"id": job_id, "kind": job_kind, "root": str(job_root), "status": "queued", "title": request_record["title"], "progress_completed": 0, "progress_total": preview.get("total_invocations", 0)}
            self._thread = threading.Thread(target=self._run_job_thread, args=(job_root, payload, preview), daemon=True)
            self._thread.start()
            return True, job_id

    def cancel(self) -> Tuple[bool, str]:
        with self._lock:
            if not self._active_job:
                return False, "No active job."
            self._cancel_event.set()
            self._active_job["status"] = "cancel_requested"
            self._write_status(Path(self._active_job["root"]), status="cancel_requested")
            processes = list(self._processes.values())
        for active in processes:
            try:
                if active.process.poll() is None:
                    active.process.terminate()
            except Exception:
                pass
        return True, "Cancellation requested."

    def _run_job_thread(self, job_root: Path, payload: Dict[str, Any], preview: Dict[str, Any]) -> None:
        self._write_status(job_root, status="running")
        with self._lock:
            if self._active_job:
                self._active_job["status"] = "running"
        try:
            if preview.get("optimizer_enabled"):
                summary = self._run_optimized_sweep(job_root, payload, preview)
            elif preview.get("top_level_case_count", 0) > 1:
                summary = self._run_plain_sweep(job_root, preview)
            else:
                summary = self._run_single(job_root, preview)
            final_status = "cancelled" if self._cancel_event.is_set() else summary.get("status", "completed")
            self._write_status(job_root, status=final_status)
            _json_dump(job_root / "summary.json", summary)
            self._write_artifacts_manifest(job_root)
        except Exception as exc:
            _json_dump(job_root / "summary.json", {"error": str(exc)})
            self._write_status(job_root, status="failed", error=str(exc))
            self._write_artifacts_manifest(job_root)
        finally:
            ACTIVE_JOB_LOCK.unlink(missing_ok=True)
            with self._lock:
                self._processes.clear()
                self._active_job = None

    def _run_single(self, job_root: Path, preview: Dict[str, Any]) -> Dict[str, Any]:
        case = preview["top_level_cases"][0]
        (job_root / "model_resolved.yaml").write_text(_yaml_dump(case["model"]))
        (job_root / "hardware_resolved.yaml").write_text(_yaml_dump(case["hardware"]))
        result = self._execute_worker_case(job_root, "case-0001", case["model"], case["hardware"], 0, dimension_values=case.get("dimension_values", {}))
        _json_dump(job_root / "result.json", result)
        _json_dump(job_root / "metrics.json", result.get("metrics", {}))
        return {
            "title": "Single Launch",
            "status": result.get("status"),
            "primary_metric_label": result.get("primary_metric_label"),
            "primary_metric_value": result.get("primary_metric_value"),
            "run_type": case["run_type"],
            "error": result.get("error"),
        }

    def _run_plain_sweep(self, job_root: Path, preview: Dict[str, Any]) -> Dict[str, Any]:
        return self._execute_case_set(job_root, [{"top_case": case, "candidate": None} for case in preview["top_level_cases"]], preview["metric"], False, 0, preview["worker_count"])

    def _run_optimized_sweep(self, job_root: Path, payload: Dict[str, Any], preview: Dict[str, Any]) -> Dict[str, Any]:
        case_plans = []
        for case in preview["top_level_cases"]:
            candidates = generate_parallelism_candidates(case["hardware"], case["run_type"], case["target_total_gpus"], payload.get("optimizer_preset") or "Fast")
            for idx, candidate in enumerate(candidates):
                case_plans.append({"top_case": case, "candidate": candidate, "candidate_index": idx})
        return self._execute_case_set(job_root, case_plans, preview["metric"], True, preview["timeout_seconds"], preview["worker_count"])

    def _execute_case_set(self, job_root: Path, case_plans: List[Dict[str, Any]], metric: str, optimize: bool, timeout_seconds: int, workers: int) -> Dict[str, Any]:
        results_by_top_case: Dict[str, List[Dict[str, Any]]] = {}
        progress_total, completed = len(case_plans), 0
        self._write_status(job_root, progress_total=progress_total, progress_completed=0)

        def task(plan: Dict[str, Any]) -> Dict[str, Any]:
            if self._cancel_event.is_set():
                return {"status": "cancelled", "top_case_id": plan["top_case"]["case_id"]}
            top_case = plan["top_case"]
            hardware = apply_parallelism_candidate(top_case["hardware"], plan["candidate"]) if plan.get("candidate") else top_case["hardware"]
            case_id = top_case["case_id"] if not optimize else f"{top_case['case_id']}-cand-{plan['candidate_index']:03d}"
            return self._execute_worker_case(job_root, case_id, top_case["model"], hardware, timeout_seconds, top_case_id=top_case["case_id"], candidate=plan.get("candidate"), case_label=top_case["label"], dimension_values=top_case.get("dimension_values", {}))

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            future_map = {executor.submit(task, plan): plan for plan in case_plans}
            for future in as_completed(future_map):
                plan = future_map[future]
                completed += 1
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"status": "failed", "error": str(exc), "top_case_id": plan["top_case"]["case_id"], "case_id": plan["top_case"]["case_id"], "label": plan["top_case"]["label"]}
                results_by_top_case.setdefault(result["top_case_id"], []).append(result)
                self._write_status(job_root, progress_total=progress_total, progress_completed=completed)
                if self._cancel_event.is_set():
                    break

        case_summaries, best_metric_value = [], None
        best_metric_label = METRIC_LABELS.get(metric, metric)
        for top_case_id, result_list in results_by_top_case.items():
            if optimize:
                successful = [item for item in result_list if item.get("status") == "completed" and item.get("metrics")]
                chosen = pick_best_result(successful, metric) if successful else pick_fallback_result(result_list)
                case_record = {"case_id": top_case_id, "label": chosen.get("label"), "status": chosen.get("status"), "dimension_values": chosen.get("dimension_values", {}), "candidate_count": len(result_list), "chosen_candidate": chosen.get("candidate"), "metrics": chosen.get("metrics", {}), "warnings": chosen.get("warnings", []), "error": chosen.get("error")}
            else:
                chosen = result_list[0]
                case_record = {"case_id": top_case_id, "label": chosen.get("label"), "status": chosen.get("status"), "dimension_values": chosen.get("dimension_values", {}), "metrics": chosen.get("metrics", {}), "warnings": chosen.get("warnings", []), "error": chosen.get("error")}
            _json_dump(job_root / "cases" / f"{top_case_id}.json", case_record)
            case_summaries.append(case_record)
            metric_value = case_record.get("metrics", {}).get(metric)
            if metric_value is not None and (best_metric_value is None or is_metric_better(metric, metric_value, best_metric_value)):
                best_metric_value = metric_value
        completed_case_count = sum(1 for item in case_summaries if item.get("status") == "completed")
        if completed_case_count == len(case_summaries):
            overall_status = "completed"
        elif completed_case_count == 0:
            overall_status = "failed"
        else:
            overall_status = "partial"
        return {"title": "Sweep Results", "status": overall_status, "best_metric_label": best_metric_label, "best_metric_value": best_metric_value, "case_count": len(case_summaries), "completed_case_count": completed_case_count}

    def _execute_worker_case(self, job_root: Path, case_id: str, model_dict: Dict[str, Any], hardware_dict: Dict[str, Any], timeout_seconds: int, *, top_case_id: Optional[str] = None, candidate: Optional[Dict[str, Any]] = None, case_label: Optional[str] = None, dimension_values: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        case_root = job_root / "artifacts" / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        model_path, hardware_path = case_root / "model.yaml", case_root / "hardware.yaml"
        result_path, stdout_path, stderr_path = case_root / "result.json", case_root / "stdout.log", case_root / "stderr.log"
        model_path.write_text(_yaml_dump(model_dict))
        hardware_path.write_text(_yaml_dump(hardware_dict))
        cmd = [str(PYTHON_BIN), "-m", WORKER_MODULE, "--model-config", str(model_path), "--hardware-config", str(hardware_path), "--result-json", str(result_path), "--output-dir", str(case_root)]
        with stdout_path.open("w") as stdout_handle, stderr_path.open("w") as stderr_handle:
            process = subprocess.Popen(cmd, cwd=str(ROOT), stdout=stdout_handle, stderr=stderr_handle, env={**os.environ, "PYTHONUNBUFFERED": "1"})  # noqa: S603
            with self._lock:
                self._processes[case_id] = ActiveProcess(process=process, log_path=stdout_path, err_path=stderr_path)
            try:
                process.wait(timeout=float(timeout_seconds) if timeout_seconds and timeout_seconds > 0 else None)
            except subprocess.TimeoutExpired:
                process.kill()
                return {"case_id": case_id, "top_case_id": top_case_id or case_id, "label": case_label or case_id, "status": "timed_out", "error": f"Timed out after {timeout_seconds}s", "candidate": candidate, "metrics": {}, "warnings": [], "dimension_values": dimension_values or {}}
            finally:
                with self._lock:
                    self._processes.pop(case_id, None)
        worker_result = _json_load(result_path) if result_path.exists() else {"success": False, "error": "Worker did not produce a result file.", "metrics": {}}
        return {
            "case_id": case_id,
            "top_case_id": top_case_id or case_id,
            "label": case_label or case_id,
            "status": "completed" if worker_result.get("success") else "failed",
            "candidate": candidate,
            "metrics": worker_result.get("metrics", {}),
            "warnings": worker_result.get("warnings", []),
            "error": worker_result.get("error"),
            "dimension_values": dimension_values or worker_result.get("dimension_values", {}),
            "primary_metric_label": worker_result.get("primary_metric_label"),
            "primary_metric_value": worker_result.get("primary_metric_value"),
        }

    def _write_status(self, job_root: Path, *, status: Optional[str] = None, error: Optional[str] = None, progress_total: Optional[int] = None, progress_completed: Optional[int] = None) -> None:
        path = job_root / "status.json"
        record = _json_load(path) if path.exists() else {}
        if status is not None:
            record["status"] = status
        if error is not None:
            record["error"] = error
        if progress_total is not None:
            record["progress_total"] = progress_total
        if progress_completed is not None:
            record["progress_completed"] = progress_completed
        record["updated_at"] = utc_now()
        if "created_at" not in record:
            record["created_at"] = record["updated_at"]
        _json_dump(path, record)
        with self._lock:
            if self._active_job and str(job_root) == self._active_job.get("root"):
                self._active_job.update({"status": record.get("status"), "progress_total": record.get("progress_total"), "progress_completed": record.get("progress_completed"), "updated_at": record.get("updated_at")})


RUN_MANAGER = RunManager()
