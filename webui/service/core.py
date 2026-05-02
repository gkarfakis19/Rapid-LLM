from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import threading
import uuid
from bisect import insort
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote, unquote, urlparse

import psutil
import yaml

import config as rapid_config


ROOT = Path(__file__).resolve().parents[2]
WEBUI_ROOT = ROOT / "webui"
CONFIG_SEEDS_ROOT = WEBUI_ROOT / "config_seeds"
WORKSPACE_ROOT = Path(os.environ.get("RAPID_WEBUI_WORKSPACE_ROOT", WEBUI_ROOT / "workspace")).expanduser().resolve()
RUNS_ROOT = WORKSPACE_ROOT / "runs"
SWEEPS_ROOT = WORKSPACE_ROOT / "sweeps"
CONFIGS_ROOT = WORKSPACE_ROOT / "configs"
LOCKS_ROOT = WORKSPACE_ROOT / "locks"
LOGS_ROOT = WORKSPACE_ROOT / "logs"
DB_ROOT = WORKSPACE_ROOT / "db"
ARTIFACTS_ROOT = WORKSPACE_ROOT / "artifacts"
PYTHON_BIN = Path(os.environ.get("RAPID_WEBUI_PYTHON_BIN", ROOT / ".venv" / "bin" / "python")).expanduser()
ACTIVE_JOB_LOCK = LOCKS_ROOT / "active_job.lock"
SCHEMA_VERSION_PATH = WORKSPACE_ROOT / "schema_version.json"
WORKER_MODULE = "webui.service.worker_runner"
SWEEP_CASES_JSONL = "cases.jsonl"
LAST_UI_STATE_FILENAME = "last_ui_state.json"
LAST_UI_STATE_LIMIT = 5
LAST_UI_STATE_MAX_TEXT = 2048
HF_TO_CONFIG_PATH = ROOT / "configs" / "model-config" / "hf_to_config.py"
YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
YAML_DUMPER = getattr(yaml, "CSafeDumper", yaml.SafeDumper)
_YAML_CACHE: Dict[Path, Tuple[int, int, Dict[str, Any]]] = {}
DEFAULT_MODEL_CONFIG_SOURCES = sorted((CONFIG_SEEDS_ROOT / "models").glob("*.yaml"))
DEFAULT_HARDWARE_CONFIG_SOURCES = sorted((CONFIG_SEEDS_ROOT / "hardware").glob("*.yaml"))
LEGACY_WEBUI_HARDWARE_CONFIG_NAMES = {
    "A100_SXM4_80GB_base.yaml",
    "H100_SXM5_80GB_2d.yaml",
    "H100_SXM5_80GB_base.yaml",
    "H100_SXM5_80GB_superpod.yaml",
    "a100_80GB.yaml",
}


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
NETWORK_SWEEP_FIELD_OPTIONS: List[Dict[str, str]] = [
    option
    for idx in range(3)
    for option in (
        {"value": f"hardware.network.dim{idx}.bandwidth_gbs", "label": f"Dimension {idx} Bandwidth (GB/s)"},
        {"value": f"hardware.network.dim{idx}.latency_s", "label": f"Dimension {idx} Latency (s)"},
    )
]
for option in NETWORK_SWEEP_FIELD_OPTIONS:
    FIELD_TYPES[option["value"]] = {"kind": "float", "label": f"Network {option['label']}"}
NETWORK_SWEEP_FIELD_RE = re.compile(r"^hardware\.network\.dim([0-2])\.(bandwidth_gbs|latency_s)$")

METRIC_LABELS = {
    "training_time_s": "Time / Batch",
    "approx_mfu": "Approx. MFU",
    "prefill_time_s": "Prefill Time",
    "decode_throughput_tok_s": "Throughput (TPOT)",
    "ttft_s": "TTFT",
    "total_inference_time_s": "Time / Batch",
}
INNER_HIERARCHICAL_AXES = ["tp", "cp", "ep"]
PP_TOPOLOGY_DIMENSIONS = {"dim1_shared", "dim1_dim2", "dim1", "dim2", "dim2_shared"}
PAPER_DERATE_DEFAULTS = {
    "A100_PCIe_80GB.yaml": {"compute": 0.60, "memory": 0.70, "communication": 0.85},
    "A100_SXM4_80GB.yaml": {"compute": 0.90, "memory": 0.70, "communication": 0.80},
    "H100_SXM5_80GB.yaml": {"compute": 0.56, "memory": 0.80, "communication": 0.85},
}
SUPPORTED_MODEL_TYPES = {"gpt", "llama", "deepseek_v3", "glm4_moe", "vit", "vit_dinov3"}
VIT_MODEL_TYPES = {"vit", "vit_dinov3"}
SUPPORTED_HF_IMPORT_MODEL_TYPES = {
    "deepseek_v3",
    "deepseekv3",
    "gpt2",
    "gpt_bigcode",
    "gpt_j",
    "gpt_neox",
    "gptj",
    "glm4",
    "glm4_moe",
    "llama",
    "mpt",
    "opt",
    "phi3",
    "qwen2",
}
HF_REPO_ID_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
HF_REVISION_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
PRECISION_OVERRIDE_FIELDS = [
    "kv_cache",
    "parameters",
    "gradients",
    "grad_communication",
    "optimizer_states",
    "stats",
    "master_parameters",
]

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
        if kind == "hardware":
            for legacy_name in LEGACY_WEBUI_HARDWARE_CONFIG_NAMES:
                (target_dir / legacy_name).unlink(missing_ok=True)
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
        WORKSPACE_ROOT / "scratch",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    _seed_editable_configs()
    if not SCHEMA_VERSION_PATH.exists():
        SCHEMA_VERSION_PATH.write_text(json.dumps({"schema_version": 1}, indent=2))


def _yaml_cache_key(path: Path) -> tuple[Path, int, int]:
    stat = path.stat()
    return path.resolve(), stat.st_mtime_ns, stat.st_size


def _yaml_load(path: Path) -> Dict[str, Any]:
    resolved_path, mtime_ns, size = _yaml_cache_key(path)
    cached = _YAML_CACHE.get(resolved_path)
    if cached and cached[0] == mtime_ns and cached[1] == size:
        return copy.deepcopy(cached[2])
    data = yaml.load(path.read_text(), Loader=YAML_LOADER) or {}
    _YAML_CACHE[resolved_path] = (mtime_ns, size, data)
    return copy.deepcopy(data)


def _yaml_dump(data: Dict[str, Any]) -> str:
    return yaml.dump(data, Dumper=YAML_DUMPER, sort_keys=False)


def _yaml_write_if_changed(path: Path, data: Dict[str, Any]) -> None:
    text = _yaml_dump(data)
    if path.exists() and path.read_text() == text:
        return
    path.write_text(text)
    _YAML_CACHE.pop(path.resolve(), None)


def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _json_load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _last_ui_state_path() -> Path:
    return WORKSPACE_ROOT / "scratch" / LAST_UI_STATE_FILENAME


def _trim_text(value: Any, limit: int = LAST_UI_STATE_MAX_TEXT) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text[:limit]


def _trim_string_list(values: Any, limit: int = 24) -> List[str]:
    if not isinstance(values, list):
        return []
    trimmed: List[str] = []
    for value in values[:limit]:
        text = _trim_text(value, 256)
        if text:
            trimmed.append(text)
    return trimmed


def _trim_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _sanitize_dimension_controls(rows: Any) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    if not isinstance(rows, list):
        rows = []
    for row in rows[:3]:
        row = row if isinstance(row, dict) else {}
        mode = row.get("mode") if row.get("mode") in {"values", "range"} else "values"
        sanitized.append(
            {
                "field": _trim_text(row.get("field"), 128),
                "network_field": _trim_text(row.get("network_field"), 128),
                "mode": mode,
                "list_text": _trim_text(row.get("list_text"), LAST_UI_STATE_MAX_TEXT) or "",
                "config_values": _trim_string_list(row.get("config_values"), 24),
                "start": _trim_number(row.get("start")),
                "end": _trim_number(row.get("end")),
                "step_or_points": _trim_number(row.get("step_or_points")),
            }
        )
    while len(sanitized) < 3:
        sanitized.append({"field": None, "network_field": None, "mode": "values", "list_text": "", "config_values": [], "start": None, "end": None, "step_or_points": None})
    return sanitized


def sanitize_last_ui_state(state: Dict[str, Any] | None) -> Dict[str, Any]:
    state = state if isinstance(state, dict) else {}
    return {
        "model_run_configs": _trim_string_list(state.get("model_run_configs"), 24),
        "hardware_run_configs": _trim_string_list(state.get("hardware_run_configs"), 24),
        "model_preset": _trim_text(state.get("model_preset"), 256),
        "hardware_preset": _trim_text(state.get("hardware_preset"), 256),
        "active_config_tab": _trim_text(state.get("active_config_tab"), 320),
        "run_mode": state.get("run_mode") if state.get("run_mode") in {"sweep", "single"} else "sweep",
        "optimize_parallelism": bool(state.get("optimize_parallelism")),
        "optimizer_preset": _trim_text(state.get("optimizer_preset"), 64) or "Fast",
        "sweep_rows": _sanitize_dimension_controls(state.get("sweep_rows")),
        "metric": _trim_text(state.get("metric"), 128),
        "x_axis": _trim_text(state.get("x_axis"), 128),
        "series_axis": _trim_text(state.get("series_axis"), 128),
        "worker_count": _trim_number(state.get("worker_count")),
        "timeout_seconds": _trim_number(state.get("timeout_seconds")),
    }


def load_last_ui_state() -> Dict[str, Any]:
    path = _last_ui_state_path()
    if not path.exists():
        return {}
    try:
        record = _json_load(path)
    except (OSError, json.JSONDecodeError):
        return {}
    current = record.get("current") if isinstance(record, dict) else None
    return sanitize_last_ui_state(current) if isinstance(current, dict) else {}


def save_last_ui_state(state: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = sanitize_last_ui_state(state)
    path = _last_ui_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    recent: List[Dict[str, Any]] = []
    if path.exists():
        try:
            existing = _json_load(path)
            recent = list(existing.get("recent") or [])
        except (OSError, json.JSONDecodeError):
            recent = []
    entry = {"saved_at": utc_now(), "state": sanitized}
    deduped = [item for item in recent if item.get("state") != sanitized]
    record = {"schema_version": 1, "current": sanitized, "recent": [entry, *deduped][:LAST_UI_STATE_LIMIT]}
    _json_dump(path, record)
    return record


def clear_last_ui_state() -> None:
    _last_ui_state_path().unlink(missing_ok=True)


def _write_sweep_cases_jsonl(job_root: Path, case_records: List[Dict[str, Any]]) -> Path:
    path = job_root / SWEEP_CASES_JSONL
    tmp_path = path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w") as handle:
        for index, record in enumerate(case_records, start=1):
            indexed_record = dict(record)
            indexed_record.setdefault("case_index", index)
            handle.write(json.dumps(indexed_record, sort_keys=True) + "\n")
    tmp_path.replace(path)
    return path


def _iter_sweep_cases_from_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            text = line.strip()
            if text:
                yield json.loads(text)


def _iter_sweep_case_records(job_root: Path) -> Tuple[Iterable[Dict[str, Any]], str]:
    jsonl_path = job_root / SWEEP_CASES_JSONL
    if jsonl_path.exists():
        return _iter_sweep_cases_from_jsonl(jsonl_path), "jsonl"
    cases_dir = job_root / "cases"
    if cases_dir.exists():
        return (_json_load(case_path) for case_path in sorted(cases_dir.glob("*.json"))), "legacy-json"
    return iter(()), "missing"


def _path_artifact_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return "config"
    if suffix == ".json":
        return "json"
    if suffix == ".log":
        return "log"
    if suffix in {".csv", ".tsv", ".jsonl"}:
        return "table"
    if suffix in {".png", ".jpg", ".jpeg", ".svg", ".html"}:
        return "visual"
    return "file"


def prettify_name(stem: str) -> str:
    return stem.replace("_", " ")


def config_label(config_id: Any) -> str:
    if not config_id:
        return ""
    return prettify_name(Path(str(config_id)).stem)


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


def _validate_hf_repo_id(model_id: str) -> str:
    model_id = str(model_id or "").strip().strip("/")
    parts = model_id.split("/")
    if len(parts) not in {1, 2} or any(not part for part in parts):
        raise ValueError("Use a Hugging Face model id like org/model or model.")
    for part in parts:
        if not HF_REPO_ID_SEGMENT_RE.fullmatch(part) or "--" in part or ".." in part:
            raise ValueError("Hugging Face model ids may contain only letters, numbers, '.', '_', and '-' in one or two path segments.")
    return "/".join(parts)


def _validate_hf_revision(revision: str | None) -> str:
    revision = str(revision or "main").strip().strip("/")
    if not revision:
        return "main"
    parts = revision.split("/")
    if any(not part for part in parts) or any(part in {".", ".."} for part in parts):
        raise ValueError("Hugging Face revisions may not contain empty, '.', or '..' path segments.")
    for part in parts:
        if not HF_REVISION_SEGMENT_RE.fullmatch(part) or "--" in part or ".." in part:
            raise ValueError("Hugging Face revisions may contain only letters, numbers, '/', '.', '_', and '-'.")
    return "/".join(parts)


def _extract_hf_revision_parts(parts: List[str], marker_index: int) -> List[str]:
    revision_parts = parts[marker_index + 1 :]
    if revision_parts and revision_parts[-1] == "config.json":
        revision_parts = revision_parts[:-1]
    if not revision_parts:
        return ["main"]
    return revision_parts


def parse_huggingface_model_reference(raw_reference: str) -> Tuple[str, str]:
    text = str(raw_reference or "").strip()
    if not text:
        raise ValueError("Enter a Hugging Face model URL or model id.")
    if "://" not in text:
        if "@" in text:
            model_id, revision = text.rsplit("@", 1)
            return _validate_hf_repo_id(model_id), _validate_hf_revision(revision)
        return _validate_hf_repo_id(text), "main"

    parsed = urlparse(text)
    if parsed.scheme.lower() != "https":
        raise ValueError("Use an https:// Hugging Face model URL.")
    if parsed.username or parsed.password:
        raise ValueError("Hugging Face URLs may not include embedded credentials.")
    if parsed.query or parsed.fragment:
        raise ValueError("Hugging Face URLs may not include query strings or fragments.")
    host = parsed.netloc.lower()
    if host not in {"huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co"}:
        raise ValueError("Use a huggingface.co or hf.co model URL.")
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if parts and parts[0] == "models":
        parts = parts[1:]
    if not parts:
        raise ValueError("Hugging Face URL does not contain a model id.")

    revision = "main"
    for marker in ("resolve", "blob", "tree"):
        if marker in parts:
            marker_index = parts.index(marker)
            model_parts = parts[:marker_index]
            revision = "/".join(_extract_hf_revision_parts(parts, marker_index))
            break
    else:
        if len(parts) > 2:
            raise ValueError("Use the model page URL or a /blob/<revision>/config.json URL.")
        model_parts = parts
    if not model_parts:
        raise ValueError("Hugging Face URL does not contain a model id.")
    return _validate_hf_repo_id("/".join(model_parts)), _validate_hf_revision(revision)


def _hf_language_config(hf_config: Dict[str, Any]) -> Dict[str, Any]:
    lang_cfg = hf_config.get("language_config", {})
    return lang_cfg if isinstance(lang_cfg, dict) else {}


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_truthy_flag(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def validate_huggingface_import_features(hf_config: Dict[str, Any], alias: str | None) -> None:
    lang_cfg = _hf_language_config(hf_config)
    if alias == "phi3":
        sliding_candidates = [hf_config.get("sliding_window"), lang_cfg.get("sliding_window")]
        if any(_is_positive_int(value) for value in sliding_candidates):
            raise ValueError(
                "Unsupported Phi-3 config: sliding-window attention is not modeled by RAPID-LLM, "
                "so this Hugging Face model cannot be imported safely."
            )
    if alias == "qwen2":
        sliding_flags = [hf_config.get("use_sliding_window"), lang_cfg.get("use_sliding_window")]
        if any(_is_truthy_flag(value) for value in sliding_flags):
            raise ValueError(
                "Unsupported Qwen2 config: active sliding-window attention is not modeled by RAPID-LLM, "
                "so this Hugging Face model cannot be imported safely."
            )


def validate_huggingface_import_config(hf_config: Dict[str, Any], converter: Any | None = None) -> Tuple[str, str | None]:
    converter = converter or _load_hf_to_config_module()
    raw_model_type = str(hf_config.get("model_type") or "").strip().lower()
    normalized_model_type = raw_model_type.replace("-", "_")
    if normalized_model_type not in SUPPORTED_HF_IMPORT_MODEL_TYPES:
        supported = ", ".join(sorted(SUPPORTED_HF_IMPORT_MODEL_TYPES))
        raise ValueError(f"Unsupported Hugging Face model_type '{raw_model_type or 'missing'}'. Supported importer model types: {supported}.")
    inferred_model_type, alias = converter._infer_model_type(raw_model_type)
    if inferred_model_type not in {"gpt", "llama", "deepseek_v3", "glm4_moe"}:
        raise ValueError(f"Imported Hugging Face model_type '{raw_model_type}' maps to unsupported RAPID-LLM model_type '{inferred_model_type}'.")
    validate_huggingface_import_features(hf_config, alias)
    return inferred_model_type, alias


def build_huggingface_config_url(model_id: str, revision: str = "main") -> str:
    safe_model_id = "/".join(quote(part, safe="") for part in _validate_hf_repo_id(model_id).split("/"))
    safe_revision = "/".join(quote(part, safe="") for part in _validate_hf_revision(revision).split("/"))
    return f"https://huggingface.co/{safe_model_id}/resolve/{safe_revision}/config.json"


def _load_hf_to_config_module():
    if not HF_TO_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Hugging Face converter not found: {HF_TO_CONFIG_PATH}")
    spec = importlib.util.spec_from_file_location("rapid_llm_hf_to_config", HF_TO_CONFIG_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Hugging Face converter: {HF_TO_CONFIG_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_model_config_from_huggingface(raw_reference: str, new_name: str | None = None) -> Dict[str, Any]:
    ensure_workspace()
    model_id, revision = parse_huggingface_model_reference(raw_reference)
    filename = _config_filename_from_user_text(new_name or f"{Path(model_id).name}_hf")
    target = config_file_path("models", filename)
    if target.exists():
        raise FileExistsError(f"Config already exists: {filename}")
    try:
        converter = _load_hf_to_config_module()
        hf_config = converter._fetch_hf_config(model_id, revision=revision)
        inferred_model_type, alias = validate_huggingface_import_config(hf_config, converter)
        args = SimpleNamespace(
            global_batch_size=1,
            gradient_accumulation_steps=1,
            seq_len=None,
            decode_len=0,
            run_type="training",
            use_flashattention=None,
            flash_tile_size=None,
        )
        yaml_config = converter._build_yaml_config(hf_config, args, inferred_model_type)
    except SystemExit as exc:
        message = str(exc) or "Hugging Face config conversion failed."
        raise ValueError(message) from exc
    yaml_config.setdefault("metadata", {})["huggingface_source"] = {
        "model_id": model_id,
        "revision": revision,
        "reference": raw_reference,
    }
    _yaml_write_if_changed(target, yaml_config)
    return {
        "id": filename,
        "model_id": model_id,
        "revision": revision,
        "model_type": inferred_model_type,
        "alias": alias,
        "path": str(target),
    }


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


def is_vit_model(model_dict: Dict[str, Any]) -> bool:
    return get_model_type(model_dict) in VIT_MODEL_TYPES


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
    return "decode_throughput_tok_s" if str(run_type).lower() == "inference" else "training_time_s"


def get_metric_options(run_type: str) -> List[Dict[str, str]]:
    if str(run_type).lower() == "inference":
        return [
            {"value": "decode_throughput_tok_s", "label": METRIC_LABELS["decode_throughput_tok_s"]},
            {"value": "ttft_s", "label": METRIC_LABELS["ttft_s"]},
            {"value": "total_inference_time_s", "label": METRIC_LABELS["total_inference_time_s"]},
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


def infer_pp_topology_dimension(hw_dict: Dict[str, Any]) -> str:
    dimensions = hw_dict.get("network", {}).get("dimensions", []) or []
    dim1_axes = {str(axis).strip().lower() for axis in (dimensions[1].get("parallelisms", []) if len(dimensions) > 1 else []) or []}
    dim2_axes = {str(axis).strip().lower() for axis in (dimensions[2].get("parallelisms", []) if len(dimensions) > 2 else []) or []}
    if {"pp", "dp"} <= dim1_axes:
        return "dim1_shared"
    if "pp" in dim1_axes and "dp" in dim2_axes:
        return "dim1_dim2"
    if {"pp", "dp"} <= dim2_axes:
        return "dim1_dim2"
    for idx, dim in enumerate(dimensions):
        axes = {str(axis).strip().lower() for axis in dim.get("parallelisms", []) or []}
        if "pp" in axes:
            return "dim1_dim2" if idx >= 2 else "dim1_shared"
    return "dim1_shared"


def _normalize_pp_topology_dimension(value: Any) -> str:
    text = str(value or "dim1_shared").strip().lower()
    if text == "dim1":
        return "dim1_dim2"
    if text in {"dim2", "dim2_shared"}:
        return "dim1_dim2"
    return text if text in PP_TOPOLOGY_DIMENSIONS else "dim1_shared"


def _apply_parallelism_topology_mapping(hardware: Dict[str, Any], pp_dimension: Any) -> None:
    dimensions = hardware.setdefault("network", {}).setdefault("dimensions", [])
    if not dimensions:
        return

    dimensions[0]["parallelisms"] = list(INNER_HIERARCHICAL_AXES)
    for idx in range(1, len(dimensions)):
        dimensions[idx]["parallelisms"] = []
    if len(dimensions) == 1:
        return

    pp_dimension = _normalize_pp_topology_dimension(pp_dimension)
    dim1_idx = 1
    dim2_idx = min(2, len(dimensions) - 1)
    if pp_dimension == "dim1_shared":
        dimensions[dim1_idx]["parallelisms"] = ["pp", "dp"]
    else:
        dimensions[dim1_idx]["parallelisms"] = ["pp"]
        dimensions[dim2_idx]["parallelisms"] = ["dp"]


def paper_derate_defaults_for_hardware(hardware_id: str | None, hardware: Dict[str, Any] | None = None) -> Dict[str, float] | None:
    config_name = Path(str(hardware_id or "")).name
    if config_name in PAPER_DERATE_DEFAULTS:
        return copy.deepcopy(PAPER_DERATE_DEFAULTS[config_name])

    if hardware is None and config_name:
        try:
            hardware = load_preset("hardware", config_name)
        except Exception:
            hardware = None

    metadata = (hardware or {}).get("metadata", {}) or {}
    signature = " ".join(
        str(value)
        for value in [
            config_name,
            metadata.get("display_name", ""),
            metadata.get("source_config", ""),
        ]
    ).lower()
    if "a100" in signature and "pcie" in signature:
        return copy.deepcopy(PAPER_DERATE_DEFAULTS["A100_PCIe_80GB.yaml"])
    if "a100" in signature and "sxm" in signature:
        return copy.deepcopy(PAPER_DERATE_DEFAULTS["A100_SXM4_80GB.yaml"])
    if "h100" in signature and "sxm" in signature:
        return copy.deepcopy(PAPER_DERATE_DEFAULTS["H100_SXM5_80GB.yaml"])
    return None


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
        topology_type = str(topology.get("type", "Ring"))
        if topology_type.strip().lower() == "superpod":
            topology_type = "Ring"
        dimensions.append(
            {
                "id": dim.get("id") or f"dim{idx}",
                "label": f"Dimension {idx}",
                "topology_type": topology_type,
                "bandwidth": _format_bandwidth_field(topology.get("bandwidth", "")),
                "latency": float(topology.get("latency", 0.0) or 0.0),
                "util": float(topology.get("util", 1.0) or 1.0),
                "parallelisms": [str(axis).upper() for axis in dim.get("parallelisms", []) or []],
            }
        )
    precision = hardware.get("sw_param", {}).get("precision", {}) or {}
    return {
        "run_type": run_type,
        "model_yaml": _yaml_dump(model),
        "hardware_yaml": _yaml_dump(hardware),
        "simple": {
            "seq_len": int(model.get("model_param", {}).get("seq_len", 0) or 0),
            "decode_len": int(model.get("model_param", {}).get("decode_len", 0) or 0),
            "batch_size": int(model.get("model_param", {}).get("global_batch_size", 1) or 1),
            "grad_accum": 1 if run_type == "inference" else int(model.get("model_param", {}).get("gradient_accumulation_steps", 1) or 1),
            "total_gpus": get_total_gpu_count(hardware, run_type),
            "tp": int(parallelism.get("tp", 1) or 1),
            "cp": int(parallelism.get("cp", 1) or 1),
            "pp": int(parallelism.get("pp", 1) or 1),
            "dp": int(train_block.get("dp", 1) or 1),
            "ep": int(train_block.get("ep", 1) or 1),
            "replica_count": int(inference_block.get("replica_count", 1) or 1) if run_type == "inference" else 1,
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
            "tensor_format": str(precision.get("tensor_format", "bf16")),
            "precision_kv_cache": str(precision.get("kv_cache", "as_tensor_format")),
            "precision_parameters": str(precision.get("parameters", "as_tensor_format")),
            "precision_gradients": str(precision.get("gradients", "fp32")),
            "precision_grad_communication": str(precision.get("grad_communication", "as_tensor_format")),
            "precision_optimizer_states": str(precision.get("optimizer_states", "fp32")),
            "precision_stats": str(precision.get("stats", "fp32")),
            "precision_master_parameters": "0" if str(precision.get("master_parameters", 0.0)).strip().lower() in {"0", "0.0"} else str(precision.get("master_parameters", 0.0)),
            "pp_network_dimension": infer_pp_topology_dimension(hardware),
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
    model_param["gradient_accumulation_steps"] = 1 if run_type == "inference" else _safe_int(simple.get("grad_accum"), _safe_int(model_param.get("gradient_accumulation_steps"), 1))
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
    run_type = str(simple.get("run_type") or "training").lower()
    sw_param = hardware.setdefault("sw_param", {})
    precision = sw_param.setdefault("precision", {})
    sw_param["full_recomputation"] = bool(advanced.get("full_recomputation", sw_param.get("full_recomputation", False)))
    sw_param["dp_zero_stage"] = _safe_int(advanced.get("dp_zero_stage"), _safe_int(sw_param.get("dp_zero_stage"), 0))
    if advanced.get("tensor_format"):
        precision["tensor_format"] = advanced["tensor_format"]
    for field in PRECISION_OVERRIDE_FIELDS:
        advanced_key = f"precision_{field}"
        if advanced.get(advanced_key) not in (None, ""):
            raw_value = advanced[advanced_key]
            if field == "master_parameters" and str(raw_value).strip().lower() in {"0", "0.0", "disabled"}:
                precision[field] = 0.0
            else:
                precision[field] = raw_value
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
    inference_block["replica_count"] = _safe_int(simple.get("replica_count"), _safe_int(inference_block.get("replica_count"), 1)) if run_type == "inference" else 1
    execution_backend = hardware.setdefault("execution_backend", {})
    execution_backend["model"] = "astra" if bool(simple.get("use_astrasim", False)) else "analytical"
    astra = execution_backend.setdefault("astra", {})
    astra["mode"] = "full_astrasim_hierarchical"

    network = hardware.setdefault("network", {})
    dimensions = network.setdefault("dimensions", [])
    network_derate = _safe_float(simple.get("network_derate"), _initial_network_derate(hardware))
    uses_astra_only_topology = False
    for idx, row in enumerate(payload.get("network_dimensions", []) or []):
        if idx >= len(dimensions):
            continue
        topology = dimensions[idx].setdefault("topology", {})
        if row.get("topology_type"):
            topology["type"] = row["topology_type"]
            topology_type = str(row["topology_type"]).strip().lower()
            if topology_type != "ring":
                uses_astra_only_topology = True
            if topology_type == "superpod":
                topology["superpod_variant"] = "h100"
                topology["leaf_size"] = 1
        if row.get("bandwidth") not in (None, ""):
            bandwidth = _coerce_network_bandwidth(row["bandwidth"])
            if str(topology.get("type", "")).strip().lower() == "superpod" and not isinstance(bandwidth, (list, tuple)):
                bandwidth = [bandwidth, bandwidth]
            topology["bandwidth"] = bandwidth
        if row.get("latency") not in (None, ""):
            topology["latency"] = _safe_float(row.get("latency"), _safe_float(topology.get("latency"), 0.0))
        topology["util"] = _safe_float(row.get("util"), network_derate)
    for idx in range(len(payload.get("network_dimensions", []) or []), len(dimensions)):
        topology = dimensions[idx].setdefault("topology", {})
        topology["util"] = network_derate
        if str(topology.get("type", "")).strip().lower() != "ring":
            uses_astra_only_topology = True
    _apply_parallelism_topology_mapping(hardware, advanced.get("pp_network_dimension"))
    if uses_astra_only_topology:
        execution_backend["model"] = "astra"
        astra["mode"] = "full_astrasim_hierarchical"


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
        _yaml_write_if_changed(model_path, model)
        _yaml_write_if_changed(hardware_path, hardware)
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
            _yaml_write_if_changed(config_file_path("models", model_id), model)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    for hardware_id in selected_hardware:
        edit_payload = {**payload, "model_preset_id": primary_model_id, "hardware_preset_id": hardware_id}
        _, hardware, hardware_errors = build_editable_configs_from_payload(edit_payload)
        if hardware is None or hardware_errors:
            errors.extend(hardware_errors)
            continue
        try:
            _yaml_write_if_changed(config_file_path("hardware", hardware_id), hardware)
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
    network_match = NETWORK_SWEEP_FIELD_RE.match(field_key)
    if network_match:
        dim_index = int(network_match.group(1))
        network_field = network_match.group(2)
        dimensions = hardware.setdefault("network", {}).setdefault("dimensions", [])
        if dim_index >= len(dimensions):
            raise ValueError(f"Network Dimension {dim_index} is not present in the selected hardware config.")
        topology = dimensions[dim_index].setdefault("topology", {})
        if network_field == "bandwidth_gbs":
            bandwidth_gbs = float(value)
            if bandwidth_gbs <= 0:
                raise ValueError(f"{dimension_label(field_key)} must be greater than 0.")
            topology["bandwidth"] = format_gb(bandwidth_gbs)
        elif network_field == "latency_s":
            latency_s = float(value)
            if latency_s < 0:
                raise ValueError(f"{dimension_label(field_key)} cannot be negative.")
            topology["latency"] = latency_s
        return
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


def _format_case_dimension(key: str, value: Any) -> str:
    if key in {"model_config", "hardware_config"}:
        return config_label(value)
    return f"{dimension_label(key)} {value}"


def build_case_label(values: Dict[str, Any], base_model_id: Any = None, base_hardware_id: Any = None) -> str:
    if not values:
        return "Base Case"
    has_config_dimension = "model_config" in values or "hardware_config" in values
    pieces: List[str] = []
    if has_config_dimension:
        model_value = values.get("model_config") or base_model_id
        hardware_value = values.get("hardware_config") or base_hardware_id
        if model_value and hardware_value:
            pieces.append(f"{config_label(model_value)} on {config_label(hardware_value)}")
        elif model_value:
            pieces.append(config_label(model_value))
        elif hardware_value:
            pieces.append(config_label(hardware_value))
    pieces.extend(_format_case_dimension(key, value) for key, value in values.items() if key not in {"model_config", "hardware_config"})
    return " | ".join(item for item in pieces if item) or "Base Case"


def _has_dimension(dimensions: List[Dict[str, Any]], field_key: str) -> bool:
    return any(dim.get("field_key") == field_key for dim in dimensions)


def _selected_models_for_dimensions(base_model: Dict[str, Any], dimensions: List[Dict[str, Any]], payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    model_dimensions = [dim for dim in dimensions if dim.get("field_key") == "model_config"]
    if not model_dimensions:
        return [base_model]
    selected: List[Dict[str, Any]] = []
    for dim in model_dimensions:
        for preset_name in dim.get("values", []) or []:
            model = load_preset("models", preset_name)
            if preset_name == payload.get("model_preset_id"):
                _apply_model_overrides(model, payload)
            selected.append(model)
    return selected or [base_model]


def default_worker_count() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, 8))


def _candidate_axis_values(configured_values: Iterable[int], target_total_gpus: int) -> List[int]:
    values = {1}
    for raw_value in configured_values:
        value = _safe_int(raw_value, 0)
        if value >= 1 and value <= target_total_gpus and target_total_gpus % value == 0:
            values.add(value)
    return sorted(values)


def _fallback_parallelism_candidate(run_type: str, target_total_gpus: int, replica_count: int = 1) -> Optional[Dict[str, int]]:
    if target_total_gpus < 1:
        return None
    if run_type == "training":
        return {"tp": 1, "cp": 1, "pp": 1, "dp": target_total_gpus, "ep": 1, "replica_count": 1}
    replica_count = max(1, replica_count)
    if target_total_gpus % replica_count != 0:
        return None
    return {"tp": target_total_gpus // replica_count, "cp": 1, "pp": 1, "dp": 1, "ep": 1, "replica_count": replica_count}


def generate_parallelism_candidates(hardware_dict: Dict[str, Any], run_type: str, target_total_gpus: int, preset_name: str) -> List[Dict[str, Any]]:
    preset = OPTIMIZER_PRESETS.get(preset_name, OPTIMIZER_PRESETS["Fast"])[run_type]
    candidates = []
    target_total_gpus = _safe_int(target_total_gpus, 0)
    tp_values = _candidate_axis_values(preset["tp"], target_total_gpus)
    cp_values = _candidate_axis_values(preset["cp"], target_total_gpus)
    pp_values = _candidate_axis_values(preset["pp"], target_total_gpus)
    ep_values = _candidate_axis_values(preset["ep"], target_total_gpus)
    for tp, cp, pp, ep in itertools.product(tp_values, cp_values, pp_values, ep_values):
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
            replica_count = _safe_int((hardware_dict.get("parallelism", {}).get("inference", {}) or {}).get("replica_count"), 1)
            total = tp * cp * pp * ep * replica_count
            if total == target_total_gpus:
                candidates.append({"tp": tp, "cp": cp, "pp": pp, "dp": 1, "ep": ep, "replica_count": replica_count})
    if not candidates:
        fallback = _fallback_parallelism_candidate(run_type, target_total_gpus, _safe_int((hardware_dict.get("parallelism", {}).get("inference", {}) or {}).get("replica_count"), 1))
        if fallback:
            candidates.append(fallback)
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
    if get_model_mode(model) == "GEMM":
        errors.append("GEMM mode is not supported in the Web UI yet. Choose LLM or ViT model families.")
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
    if _has_dimension(dimensions, "model.seq_len") and any(is_vit_model(item) for item in _selected_models_for_dimensions(model, dimensions, payload)):
        errors.append("ViT model families derive sequence length from vision.image_size and vision.patch_size. Sequence-length sweeps for ViT are not supported in the Web UI right now.")
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings, "top_level_cases": []}

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
                "label": build_case_label(case_meta["dimension_values"], payload.get("model_preset_id"), payload.get("hardware_preset_id")),
                "model": case_model,
                "hardware": case_hw,
                "run_type": get_model_run_type(case_model),
                "target_total_gpus": int(case_meta["target_total_gpus"]),
                "dimension_values": case_meta["dimension_values"],
            })
        except Exception as exc:  # noqa: BLE001
            invalid_cases.append({"label": build_case_label(case_meta["dimension_values"], payload.get("model_preset_id"), payload.get("hardware_preset_id")), "error": str(exc)})
    if invalid_cases:
        warnings.append(f"Pruned {len(invalid_cases)} invalid case(s) before launch.")
    if invalid_cases and not top_level_cases:
        errors.append("All sweep cases were invalid. Check Total GPUs, fixed parallelism axes, or enable Optimize parallelism.")
        return {"ok": False, "errors": errors, "warnings": warnings, "top_level_case_count": 0, "candidate_breakdown": [], "top_level_cases": [], "invalid_cases": invalid_cases[:20], "total_invocations": 0}
    valid_metrics = {option["value"] for option in get_metric_options(run_type)}
    metric = payload.get("metric") or get_default_metric_for_run_type(run_type)
    if metric not in valid_metrics:
        metric = get_default_metric_for_run_type(run_type)
    timeout_s = max(0, _safe_int(payload.get("timeout_seconds"), 180))
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
        "worst_case_wall_clock_s": None if timeout_s == 0 and total_invocations else math.ceil(total_invocations / max(1, workers)) * timeout_s if total_invocations else 0,
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
    _assign_history_title_indices(entries)
    entries.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    return entries[:limit]


def _assign_history_title_indices(entries: List[Dict[str, Any]]) -> None:
    by_title: Dict[str, List[Dict[str, Any]]] = {}
    for item in entries:
        by_title.setdefault(str(item.get("title") or ""), []).append(item)
    for group in by_title.values():
        group.sort(key=lambda item: (item.get("created_at") or "", item.get("id") or ""))
        duplicate_count = len(group)
        for index, item in enumerate(group, start=1):
            item["title_index"] = index
            item["title_duplicate_count"] = duplicate_count


def _case_sort_tuple(case: Dict[str, Any], metric: str, ordinal: int) -> Tuple[Tuple[int, float], str, int]:
    metric_value = (case.get("metrics") or {}).get(metric)
    return metric_sort_key(metric, metric_value), str(case.get("case_id") or ""), ordinal


def _load_sweep_cases(job_root: Path, metric: str, case_limit: int | None = None, display_mode: str | None = "top") -> Tuple[List[Dict[str, Any]], int, int, str, str]:
    mode = "full" if display_mode == "full" else "top"
    records, source = _iter_sweep_case_records(job_root)
    if mode == "full" or case_limit is None or case_limit < 0:
        cases = list(records)
        return cases, len(cases), len(cases), mode, source

    limit = max(0, int(case_limit))
    selected: List[Tuple[Tuple[Tuple[int, float], str, int], Dict[str, Any]]] = []
    total = 0
    for ordinal, case in enumerate(records):
        total += 1
        if limit == 0:
            continue
        insort(selected, (_case_sort_tuple(case, metric, ordinal), case))
        if len(selected) > limit:
            selected.pop()
    cases = [case for _, case in selected]
    return cases, total, len(cases), mode, source


def load_job_detail(job_kind: str, job_id: str, case_limit: int | None = None, display_mode: str | None = "top") -> Dict[str, Any]:
    path = _detail_root(job_kind) / job_id
    detail = _job_summary_from_dir(path, job_kind)
    detail["status_record"] = _json_load(path / "status.json") if (path / "status.json").exists() else {}
    detail["request_record"] = _json_load(path / "request.json") if (path / "request.json").exists() else {}
    detail["summary_record"] = _json_load(path / "summary.json") if (path / "summary.json").exists() else {}
    if job_kind == "sweep":
        payload = detail["request_record"].get("payload") or {}
        metric = payload.get("metric") or detail.get("metric") or "training_time_s"
        cases, total_cases, loaded_cases, mode, source = _load_sweep_cases(path, metric, case_limit, display_mode)
        detail["cases"] = cases
        detail["_case_count_total"] = total_cases
        detail["_case_count_loaded"] = loaded_cases
        detail["_case_limit"] = case_limit
        detail["_case_display_mode"] = mode
        detail["_case_sort_metric"] = metric
        detail["_case_source"] = source
    else:
        detail["result"] = _json_load(path / "result.json") if (path / "result.json").exists() else {}
    return detail


def _slugify_artifact_name(raw: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(raw or "").strip().lower()).strip("-")
    return slug or "plot"


def save_plot_png(job_kind: str, job_id: str, title: str, png_bytes: bytes) -> str:
    job_root = _detail_root(job_kind) / job_id
    if not job_root.exists():
        raise FileNotFoundError(f"Unknown job: {job_id}")
    if not png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Plot export did not produce a PNG image.")
    plot_dir = job_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify_artifact_name(title or job_id)
    index = 1
    while True:
        path = plot_dir / f"{slug}-plot-{index:03d}.png"
        if not path.exists():
            path.write_bytes(png_bytes)
            return str(path)
        index += 1


def save_table_export(job_kind: str, job_id: str, title: str, fmt: str, content: str) -> str:
    job_root = _detail_root(job_kind) / job_id
    if not job_root.exists():
        raise FileNotFoundError(f"Unknown job: {job_id}")
    normalized_fmt = str(fmt).lower()
    if normalized_fmt not in {"csv", "json"}:
        raise ValueError("Table exports support CSV or JSON.")
    export_dir = job_root / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify_artifact_name(title or job_id)
    index = 1
    while True:
        path = export_dir / f"{slug}-table-{index:03d}.{normalized_fmt}"
        if not path.exists():
            path.write_text(content)
            return str(path)
        index += 1


def _preview_config_values(preview: Dict[str, Any], field_key: str, fallback: Any) -> List[Any]:
    values: List[Any] = []
    for case in preview.get("top_level_cases", []) or []:
        value = (case.get("dimension_values") or {}).get(field_key) or fallback
        if value and value not in values:
            values.append(value)
    if not values and fallback:
        values.append(fallback)
    return values


def build_job_title(payload: Dict[str, Any], preview: Dict[str, Any]) -> str:
    model_values = _preview_config_values(preview, "model_config", payload.get("model_preset_id", "model"))
    hardware_values = _preview_config_values(preview, "hardware_config", payload.get("hardware_preset_id", "hardware"))
    run_type = str(preview.get("run_type") or (payload.get("simple") or {}).get("run_type") or "").strip().lower()
    sweep_label = "Inference Sweep" if run_type == "inference" else "Training Sweep" if run_type == "training" else "Sweep"
    if len(model_values) > 1 and len(hardware_values) > 1:
        base = f"{len(model_values)} models x {len(hardware_values)} hardware targets"
    elif len(model_values) > 1:
        base = f"{len(model_values)} models on {config_label(hardware_values[0])}"
    elif len(hardware_values) > 1:
        base = f"{config_label(model_values[0])} on {len(hardware_values)} hardware targets"
    else:
        base = f"{config_label(model_values[0])} on {config_label(hardware_values[0])}"
    return f"{base} {sweep_label}" if preview.get("top_level_case_count", 0) > 1 or preview.get("optimizer_enabled") else base


def is_metric_better(metric: str, candidate: float, incumbent: float) -> bool:
    return float(candidate) > float(incumbent) if metric in {"decode_throughput_tok_s", "approx_mfu"} else float(candidate) < float(incumbent)


def metric_sort_key(metric: str, value: Optional[float]) -> Tuple[int, float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return (1, float("inf"))
    return (0, -float(value)) if metric in {"decode_throughput_tok_s", "approx_mfu"} else (0, float(value))


def pick_best_result(results: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    return sorted(results, key=lambda item: metric_sort_key(metric, item.get("metrics", {}).get(metric)))[0]


def result_memory_violation_gb(result: Dict[str, Any]) -> float:
    metrics = result.get("metrics") or {}
    try:
        return max(0.0, float(metrics.get("memory_violation_gb") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def result_memory_exceeded(result: Dict[str, Any]) -> bool:
    metrics = result.get("metrics") or {}
    violation_gb = result_memory_violation_gb(result)
    raw_exceeded = metrics.get("memory_exceeded")
    if isinstance(raw_exceeded, str):
        exceeded = raw_exceeded.strip().lower() in {"1", "true", "yes"}
    else:
        exceeded = bool(raw_exceeded)
    return exceeded or violation_gb > 0


def pick_best_optimized_result(results: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    memory_fitting = [item for item in results if not result_memory_exceeded(item)]
    if memory_fitting:
        return pick_best_result(memory_fitting, metric)
    chosen = sorted(
        results,
        key=lambda item: (
            result_memory_violation_gb(item),
            metric_sort_key(metric, item.get("metrics", {}).get(metric)),
        ),
    )[0]
    chosen = copy.deepcopy(chosen)
    warnings = list(chosen.get("warnings") or [])
    message = "Every tested parallelism candidate exceeded memory capacity."
    if message not in warnings:
        warnings.append(message)
    chosen["warnings"] = warnings
    return chosen


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


def worker_subprocess_env() -> Dict[str, str]:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    existing_pythonpath = env.get("PYTHONPATH", "")
    root_text = str(ROOT)
    entries = [root_text]
    if existing_pythonpath:
        entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


class RunManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cancel_event = threading.Event()
        self._active_job: Optional[Dict[str, Any]] = None
        self._last_finished_job: Optional[Dict[str, Any]] = None
        self._thread: Optional[threading.Thread] = None
        self._processes: Dict[str, ActiveProcess] = {}

    def active_job(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._active_job)

    def last_finished_job(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._last_finished_job)

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
            self._last_finished_job = None
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
            ACTIVE_JOB_LOCK.write_text(json.dumps({"job_id": job_id, "kind": job_kind, "pid": os.getpid(), "created_at": utc_now()}))
            self._active_job = {"id": job_id, "kind": job_kind, "root": str(job_root), "status": "queued", "title": request_record["title"], "created_at": request_record["created_at"], "updated_at": request_record["created_at"], "progress_completed": 0, "progress_total": preview.get("total_invocations", 0)}
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
                if self._active_job:
                    finished_job = copy.deepcopy(self._active_job)
                    total = _safe_int(finished_job.get("progress_total"), 0)
                    if total > 0:
                        finished_job["progress_completed"] = total
                    self._last_finished_job = finished_job
                self._active_job = None

    def _run_single(self, job_root: Path, preview: Dict[str, Any]) -> Dict[str, Any]:
        case = preview["top_level_cases"][0]
        (job_root / "model_resolved.yaml").write_text(_yaml_dump(case["model"]))
        (job_root / "hardware_resolved.yaml").write_text(_yaml_dump(case["hardware"]))
        result = self._execute_worker_case(job_root, "case-0001", case["model"], case["hardware"], 0, dimension_values=case.get("dimension_values", {}))
        selected_metric = preview.get("metric")
        if selected_metric in (result.get("metrics") or {}):
            result["primary_metric_label"] = METRIC_LABELS.get(selected_metric, selected_metric)
            result["primary_metric_value"] = result["metrics"][selected_metric]
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
                chosen = pick_best_optimized_result(successful, metric) if successful else pick_fallback_result(result_list)
                case_record = {"case_id": top_case_id, "label": chosen.get("label"), "status": chosen.get("status"), "dimension_values": chosen.get("dimension_values", {}), "candidate_count": len(result_list), "chosen_candidate": chosen.get("candidate"), "metrics": chosen.get("metrics", {}), "warnings": chosen.get("warnings", []), "error": chosen.get("error")}
            else:
                chosen = result_list[0]
                case_record = {"case_id": top_case_id, "label": chosen.get("label"), "status": chosen.get("status"), "dimension_values": chosen.get("dimension_values", {}), "metrics": chosen.get("metrics", {}), "warnings": chosen.get("warnings", []), "error": chosen.get("error")}
            case_summaries.append(case_record)
            metric_value = case_record.get("metrics", {}).get(metric)
            if metric_value is not None and (best_metric_value is None or is_metric_better(metric, metric_value, best_metric_value)):
                best_metric_value = metric_value
        case_summaries.sort(key=lambda item: str(item.get("case_id") or ""))
        _write_sweep_cases_jsonl(job_root, case_summaries)
        completed_case_count = sum(1 for item in case_summaries if item.get("status") == "completed")
        if completed_case_count == len(case_summaries):
            overall_status = "completed"
        elif completed_case_count == 0:
            overall_status = "failed"
        else:
            overall_status = "partial"
        return {"title": "Sweep Results", "status": overall_status, "best_metric_label": best_metric_label, "best_metric_value": best_metric_value, "case_count": len(case_summaries), "completed_case_count": completed_case_count}

    def _execute_worker_case(self, job_root: Path, case_id: str, model_dict: Dict[str, Any], hardware_dict: Dict[str, Any], timeout_seconds: int, *, top_case_id: Optional[str] = None, candidate: Optional[Dict[str, Any]] = None, case_label: Optional[str] = None, dimension_values: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        case_root = (job_root / "artifacts" / case_id).resolve()
        case_root.mkdir(parents=True, exist_ok=True)
        (case_root / "astra_cache").mkdir(exist_ok=True)
        model_path, hardware_path = case_root / "model.yaml", case_root / "hardware.yaml"
        result_path, stdout_path, stderr_path = case_root / "result.json", case_root / "stdout.log", case_root / "stderr.log"
        model_path.write_text(_yaml_dump(model_dict))
        hardware_path.write_text(_yaml_dump(hardware_dict))
        cmd = [str(PYTHON_BIN), "-m", WORKER_MODULE, "--model-config", str(model_path), "--hardware-config", str(hardware_path), "--result-json", str(result_path), "--output-dir", str(case_root)]
        with stdout_path.open("w") as stdout_handle, stderr_path.open("w") as stderr_handle:
            process = subprocess.Popen(cmd, cwd=str(case_root), stdout=stdout_handle, stderr=stderr_handle, env=worker_subprocess_env())  # noqa: S603
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
                self._active_job.update({"status": record.get("status"), "created_at": record.get("created_at"), "progress_total": record.get("progress_total"), "progress_completed": record.get("progress_completed"), "updated_at": record.get("updated_at")})


RUN_MANAGER = RunManager()
