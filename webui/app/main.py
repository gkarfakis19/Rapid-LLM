from __future__ import annotations

import base64
import binascii
import csv
import json
import os
import re
from datetime import datetime, timedelta
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, List

import dash
import dash_mantine_components as dmc
import matplotlib
import plotly.express as px
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, callback, dash_table, dcc, html, no_update
from dash_iconify import DashIconify
from flask import Response, request

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from webui.service.core import (
    FIELD_OPTIONS,
    FIELD_TYPES,
    METRIC_LABELS,
    NETWORK_SWEEP_FIELD_OPTIONS,
    NETWORK_SWEEP_TARGETS,
    RUN_MANAGER,
    build_form_defaults,
    build_case_label,
    build_job_title,
    build_launch_preview,
    clear_last_ui_state,
    create_config_copy,
    create_model_config_from_huggingface,
    default_worker_count,
    config_label,
    dimension_label,
    ensure_workspace,
    get_default_metric_for_run_type,
    get_metric_options,
    get_telemetry,
    list_history,
    list_presets,
    load_last_ui_state,
    load_job_detail,
    paper_derate_defaults_for_hardware,
    render_editable_config_texts,
    rename_config_file,
    save_last_ui_state,
    save_plot_png,
    save_table_export,
    save_config_edits_from_payload,
)


def humanize_seconds(value: float) -> str:
    seconds = int(round(value))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def format_worst_case_wall_clock(value: Any) -> str:
    return "N/A" if value is None else humanize_seconds(float(value))


FIELD_LABEL_MAP = {item["value"]: item["label"] for item in FIELD_OPTIONS}
FIELD_LABEL_MAP.update({key: str(meta["label"]) for key, meta in FIELD_TYPES.items() if meta.get("label")})
NETWORK_SWEEP_GROUP_VALUE = "__network__"
NETWORK_SWEEP_FIELD_KEYS = {item["value"] for item in NETWORK_SWEEP_FIELD_OPTIONS}
DEFAULT_NETWORK_SWEEP_FIELD = NETWORK_SWEEP_FIELD_OPTIONS[0]["value"]
DEFAULT_NETWORK_SWEEP_TARGETS = [DEFAULT_NETWORK_SWEEP_FIELD]
NETWORK_SWEEP_TARGET_BY_VALUE = {target["value"]: target for target in NETWORK_SWEEP_TARGETS}
NETWORK_SWEEP_TARGET_BY_SLUG = {target["slug"]: target for target in NETWORK_SWEEP_TARGETS}
SWEEP_FIELD_OPTIONS = FIELD_OPTIONS + [{"value": NETWORK_SWEEP_GROUP_VALUE, "label": "Network"}]
NETWORK_TOPOLOGY_OPTIONS = [
    {"value": "Ring", "label": "Ring"},
    {"value": "FC", "label": "Fully Connected"},
    {"value": "Mesh2D", "label": "Mesh2D"},
    {"value": "Torus2D", "label": "Torus2D"},
]
METRIC_KEY_BY_LABEL: Dict[str, str] = {}
for metric_key, metric_label in METRIC_LABELS.items():
    METRIC_KEY_BY_LABEL.setdefault(metric_label, metric_key)
METRIC_KEY_BY_LABEL.update(
    {
        "Decode Throughput": "decode_throughput_tok_s",
        "Time To First Token": "ttft_s",
        "Total Inference Time": "total_inference_time_s",
        "Prefill Time": "prefill_time_s",
        "Training Time": "training_time_s",
    }
)
DISPLAY_LABELS = {
    "achieved_flops": "Achieved System FLOPS",
    "achieved_flops_per_gpu": "Achieved FLOPS / GPU",
    "case": "Case",
    "decode_time_s": "Decode Time",
    "memory_exceeded": "Memory Exceeded",
    "num_gpus": "GPUs",
    "parallelism": "Parallelism",
    "total_flops": "Total FLOPs",
    "approx_mfu": "Approx. MFU",
    "total_inference_time_s": "Time / Batch",
    "ttft_s": "Time To First Token",
}
APP_CREDIT_TEXT = "By George Karfakis, georgekarfakis@ucla.edu"
APP_LOGO_ASSET = "nanocad-logo.png"
HF_SAMPLE_MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-7B"
DEFAULT_OPTIMIZE_PARALLELISM = True
DETAIL_RENDER_CASE_LIMIT = 2_000
DETAIL_TABLE_ROW_LIMIT = 1_000
DETAIL_PLOT_POINT_LIMIT = 1_500
DETAIL_HIDDEN_METRIC_KEYS = {"peak_flops_per_gpu", "peak_system_flops"}
DETAIL_TABLE_COLUMN_ORDER = [
    "case",
    "case_id",
    "label",
    "memory_exceeded",
    "training_time_s",
    "prefill_time_s",
    "decode_time_s",
    "total_inference_time_s",
    "ttft_s",
    "num_gpus",
    "approx_mfu",
    "total_flops",
    "achieved_flops",
    "achieved_flops_per_gpu",
    "status",
    "model_config",
    "hardware_config",
    "model.seq_len",
    "model.decode_len",
    "parallelism",
]
EARLY_TERMINATION_MILD_THRESHOLD = 35.0
EARLY_TERMINATION_BIG_THRESHOLD = 70.0
AUTH_ADMIN_USERNAME = "admin"
AUTH_ADMIN_PASSWORD = "!@#$57005!@#$"
AUTH_GUEST_USERNAME = "guest"
AUTH_GUEST_PASSWORD_RE = re.compile(r"^\$extern_V[A-Z0-9]{4}\$")
FLOP_COUNT_KEYS = {"total_flops"}
FLOP_RATE_KEYS = {"achieved_flops", "achieved_flops_per_gpu", "peak_flops_per_gpu", "peak_system_flops"}
TOKEN_RATE_KEYS = {"decode_throughput_tok_s"}
TIME_KEYS = {"training_time_s", "prefill_time_s", "decode_time_s", "total_inference_time_s", "ttft_s"}
RANGE_PREVIEW_LIMIT = 12
MODEL_MODE_OPTIONS = [
    {"value": "LLM", "label": "LLM"},
    {"value": "VIT", "label": "ViT"},
]
MODEL_MODE_HELP = {
    "LLM": "Transformer language-model path. Uses token sequence length, attention, MLP/MoE, training or inference timing, memory estimates, parallelism, and communication modeling.",
    "VIT": "Vision Transformer execution path. This should be paired with a ViT model_type and a model_param.vision block.",
}
MODEL_ARCH_TYPE_OPTIONS = [
    {"value": "gpt", "label": "gpt"},
    {"value": "llama", "label": "llama"},
    {"value": "deepseek_v3", "label": "deepseek_v3"},
    {"value": "glm4_moe", "label": "glm4_moe"},
    {"value": "vit", "label": "vit"},
    {"value": "vit_dinov3", "label": "vit_dinov3"},
]
MODEL_ARCH_TYPE_HELP = {
    "gpt": "Dense GPT-style Transformer. Uses standard dense MLP projections and GELU-style behavior unless other fields override it.",
    "llama": "Llama-style Transformer. Uses gated/SwiGLU MLP accounting and standard dense attention or configured GQA/MLA settings.",
    "deepseek_v3": "DeepSeek-V3 family. Treated as Llama-style for MLP accounting and used with DeepSeek MLA/MoE configuration fields.",
    "glm4_moe": "GLM-4 MoE family. Uses GLM-style handling and requires model_param.attention.head_dim in the model YAML.",
    "vit": "Vision Transformer family. Requires model_param.vision and models patch embedding, pooling/head work, and ViT memory behavior.",
    "vit_dinov3": "DINOv3 ViT family. Uses ViT handling with DINOv3 defaults, including five prefix tokens and SwiGLU MLP accounting.",
}
TENSOR_FORMAT_OPTIONS = [
    {"value": "mxfp4", "label": "MXFP4 (4.25 bits)"},
    {"value": "int4", "label": "INT4 (4 bits)"},
    {"value": "fp8", "label": "FP8 (8 bits)"},
    {"value": "fp16", "label": "FP16 (16 bits)"},
    {"value": "bf16", "label": "BF16 (16 bits)"},
    {"value": "fp32", "label": "FP32 (32 bits)"},
]
PRECISION_FORMAT_OPTIONS = [{"value": "as_tensor_format", "label": "Match tensor format"}] + TENSOR_FORMAT_OPTIONS
MASTER_PRECISION_OPTIONS = [{"value": "0", "label": "Disabled (0 bytes)"}] + PRECISION_FORMAT_OPTIONS
ZERO_STAGE_OPTIONS = [
    {"value": "0", "label": "0 (DDP)"},
    {"value": "1", "label": "1 (optimizer shard)"},
    {"value": "2", "label": "2 (optimizer+grad shard)"},
    {"value": "3", "label": "3 (FSDP/full shard)"},
]
PP_TOPOLOGY_OPTIONS = [
    {"value": "dim1_shared", "label": "Dimension 1 | Dimension 2: PP+DP | None"},
    {"value": "dim1_dim2", "label": "Dimension 1 | Dimension 2: PP | DP"},
]
HELP_TEXT = {
    "app_title": "RAPID-LLM launch workspace for editing YAML configs, previewing run size, launching jobs, and reviewing saved results.",
    "app_flow": "Basic flow: configure in Launch, start the run, open Run log, then click Details. Hover controls, metrics, and table columns for explanations.",
    "telemetry_ram": "Available host memory reported by psutil.",
    "telemetry_cpu": "Current host CPU utilization reported by psutil.",
    "telemetry_job": "Current job state.",
    "models_to_run": "Select model YAML files to include as run cases.",
    "hardware_to_run": "Select hardware YAML files to include as run cases.",
    "editor_tabs": "Choose the YAML file to edit.",
    "model_preset": "Active model YAML loaded for editing.",
    "hardware_preset": "Active hardware YAML loaded for editing.",
    "config_file_name": "Filename for a new or renamed YAML under webui/workspace/configs.",
    "new_config": "Create a new editable YAML by copying the active model or hardware config.",
    "rename_config": "Rename the active editable YAML file and keep it selected.",
    "hf_model_url": "Hugging Face model page URL or model id. The importer accepts only strict repo ids and supported LLM model_type families.",
    "hf_config_name": "Filename for the imported model YAML under webui/workspace/configs/models.",
    "hf_import": "Create a model YAML from Hugging Face config.json when the model family is explicitly supported by RAPID-LLM.",
    "worker_count": "Number of worker processes used inside a sweep. More workers consume more CPU and memory.",
    "timeout": "Maximum wall-clock time allowed for each simulator invocation. Set 0 to disable the timeout.",
    "run_type": "Choose training or inference; this controls which fields and result metrics are active.",
    "model_mode": "Execution family written to model_param.mode. LLM uses token sequences; ViT uses image and patch dimensions.",
    "model_type": "Select the architecture family written to model_param.model_type. This controls modeling details such as MLP style, ViT handling, and GLM/DeepSeek special cases.",
    "seq_len": "Number of tokens processed in the input context.",
    "decode_len": "Number of generated tokens to model for inference runs.",
    "batch_size": "Global batch size across all participating devices.",
    "grad_accum": "Number of microbatch accumulation steps before optimizer update. Inference always uses 1 and disables this field.",
    "optimize_parallelism": "Search TP, CP, PP, DP, and EP combinations for each Total GPUs target. If Total GPUs is swept, each sweep value becomes its own search target.",
    "optimizer_preset": "Controls how many parallelism candidates the search evaluates for each target GPU count.",
    "total_gpus": "Target device count. If Total GPUs is swept, this field is replaced by the sweep values and no longer affects the launch plan.",
    "tp": "Tensor parallel shards each layer's matrix work across devices.",
    "cp": "Context parallel shards long sequences across devices.",
    "pp": "Pipeline parallel splits model layers into pipeline stages.",
    "dp": "Data parallel replicates the model across training batches.",
    "ep": "Expert parallel distributes MoE experts across devices.",
    "replica_count": "Inference replica count used to scale throughput. Training locks this to 1.",
    "hbm_gb": "Usable high-bandwidth memory capacity per GPU.",
    "gpu_clock": "GPU core clock used for peak compute estimates.",
    "memory_bw": "Memory bandwidth used by the analytical memory model.",
    "use_astrasim": "Run the simulator through AstraSim using hierarchical mode. When off, use the analytical backend.",
    "compute_derate": "Multiplier for sustained compute efficiency.",
    "memory_derate": "Multiplier for sustained memory bandwidth efficiency.",
    "network_derate": "Default network utilization multiplier applied to dimensions without an explicit value.",
    "reset_paper_derates": "Restore compute, memory, and communication derates from the calibrated paper defaults for this hardware target.",
    "parallelism_topology": "Hierarchical AstraSim fixes Dimension 0 to TP, CP, and EP. Choose whether PP shares the active outer dimension with DP or gets a separate middle dimension.",
    "pp_topology_dimension": "Select how PP and DP map to Dimensions 1 and 2. The PP+DP option disables Dimension 2 and leaves its YAML parallelisms list empty.",
    "network_topology": "Collective topology model for this network dimension. SuperPOD is not exposed because support is not reliable enough yet.",
    "network_bandwidth": "Per-link or dimension bandwidth, such as 100 GB.",
    "network_latency": "Per-hop latency for this network dimension in seconds.",
    "network_util": "Utilization multiplier for this network dimension.",
    "full_recomp": "Enable full activation recomputation to trade compute for memory.",
    "zero_stage": "ZeRO sharding stage for optimizer, gradient, and parameter state.",
    "tensor_format": "Default tensor precision used for activations and compute. Other precision fields can either match this value or override it.",
    "precision_kv_cache": "Precision for inference KV cache storage. Match tensor format keeps it tied to the Tensor format field.",
    "precision_parameters": "Precision for model parameter storage.",
    "precision_gradients": "Precision for accumulated gradients.",
    "precision_grad_communication": "Precision used for gradient communication payloads.",
    "precision_optimizer_states": "Precision for optimizer state tensors.",
    "precision_stats": "Precision for normalization and statistics tensors.",
    "precision_master_parameters": "Precision for an optional master parameter copy. Disabled means no master copy is counted.",
    "tied_embeddings": "Share token embedding and output projection weights for architectures that allow it.",
    "hidden_dim": "Transformer hidden dimension.",
    "intermediate_size": "Dense MLP intermediate dimension.",
    "num_layers": "Number of transformer layers.",
    "vocab_size": "Vocabulary size used by embeddings and output projection.",
    "attention_type": "Attention implementation family.",
    "num_heads": "Number of attention heads.",
    "use_flash": "Enable FlashAttention for training or prefill paths that model it.",
    "attention_tile": "Tile size used by FlashAttention estimates.",
    "num_experts": "Number of MoE experts.",
    "top_k": "Number of experts selected per token.",
    "moe_intermediate_size": "MoE expert MLP intermediate dimension.",
    "expert_imbalance": "Worst-case expert load multiplier for imbalanced routing.",
    "metric": "Metric used for ranking sweep cases and picking the best result. Inference runs offer TPOT throughput, TTFT, and time per batch.",
    "x_axis": "Sweep field used as the plot x-axis in Details. Defaults to the first active sweep dimension.",
    "series_axis": "Optional sweep field used to color plot series. Defaults to the last active sweep dimension when there is more than one.",
    "sweep_dimensions": "Sweep workload size or hardware scaling. Network opens per-dimension bandwidth and latency choices without exposing every network field in this list. Use Total GPUs for GPU scaling; raw TP/CP/PP/DP/EP are edited once or found by Optimize parallelism, not swept independently.",
    "network_sweep_field": "Choose one or more network dimensions and parameters for this sweep dimension.",
    "network_sweep_apply": "Set values writes the sweep value into selected same-unit fields. Scale baseline multiplies each selected field's current value by the sweep value.",
    "reset_last_state": "Clear remembered selections and return sweep/config controls to defaults.",
    "run_launch": "Start the launch plan shown on the right.",
    "progress_run_log": "Open the run log entry for the finished job.",
    "cancel_job": "Request cancellation for the active job.",
    "load_details": "Open this saved job's Details.",
}
METRIC_HELP = {
    "training_time_s": "Modeled time for one training batch.",
    "approx_mfu": "Approximate MFU = achieved system FLOPS divided by configured theoretical system FLOPS.",
    "prefill_time_s": "Modeled time to process the input prompt before token generation.",
    "decode_time_s": "Modeled time spent generating decode tokens.",
    "total_inference_time_s": "Prefill time plus modeled decode time.",
    "ttft_s": "Time to first token, including prefill.",
    "decode_throughput_tok_s": "Midpoint decode token throughput multiplied by batch size and replica count.",
    "num_gpus": "Total devices implied by TP, CP, PP, DP/replicas, and EP.",
    "total_flops": "Estimated total training work for the modeled batch. This is a count, not a rate.",
    "achieved_flops": "System-wide estimated FLOP/s: total modeled FLOPs divided by simulated runtime.",
    "achieved_flops_per_gpu": "System achieved FLOP/s divided by the number of modeled GPUs.",
    "memory_exceeded": "No means estimated peak memory fits. A GB value is the selected result's amount over device memory; under parallelism optimization it appears only when every tested candidate exceeded memory, and the least-over-capacity candidate is shown.",
}
DETAIL_COLUMN_HELP = {
    "case": "Case identifier plus the model config used for this row.",
    "case_id": "Stable case identifier generated by the launcher.",
    "label": "Human-readable case label derived from config and sweep choices.",
    "model.seq_len": "Input sequence length represented for this case.",
    "model.decode_len": "Generated token count represented for this inference case.",
    "status": "Completed, failed, timed out, partial, or cancelled.",
    "metric": "Metric name emitted by the worker.",
    "value": "Formatted metric value.",
    "model_config": "Model YAML file used for this case.",
    "hardware_config": "Hardware YAML file used for this case.",
    "parallelism": "Compact parallelism assignment for this row, such as TP/CP/PP/DP/EP or replica count.",
}


def pretty_label(key: str) -> str:
    if key in DISPLAY_LABELS:
        return DISPLAY_LABELS[key]
    if key in METRIC_LABELS:
        return METRIC_LABELS[key]
    if key in FIELD_LABEL_MAP:
        return FIELD_LABEL_MAP[key]
    label = dimension_label(key)
    if label != key:
        return label
    cleaned = key.replace(".", " ").replace("_", " ")
    return cleaned[:1].upper() + cleaned[1:]


def help_for_key(key: str) -> str:
    if key in METRIC_HELP:
        return METRIC_HELP[key]
    if key in DETAIL_COLUMN_HELP:
        return DETAIL_COLUMN_HELP[key]
    if key in FIELD_LABEL_MAP:
        return f"Sweep value for {FIELD_LABEL_MAP[key]}."
    return f"Value for {pretty_label(key)}."


def with_tip(component: Any, text: str) -> dmc.Tooltip:
    return dmc.Tooltip(
        label=text,
        multiline=True,
        w=280,
        position="top-start",
        openDelay=700,
        closeDelay=100,
        withArrow=True,
        children=html.Div(component),
    )


def mode_badge(value: str) -> dmc.Tooltip:
    label = next((item["label"] for item in MODEL_MODE_OPTIONS if item["value"] == value), value)
    return with_tip(dmc.Badge(label, radius="xl", color="teal" if value == "VIT" else "blue" if value == "LLM" else "orange", variant="light"), MODEL_MODE_HELP[value])


def model_type_badge(value: str) -> dmc.Tooltip:
    return with_tip(dmc.Badge(value, radius="xl", color="blue" if value in {"gpt", "llama", "deepseek_v3", "glm4_moe"} else "teal", variant="light"), MODEL_ARCH_TYPE_HELP[value])


def flow_help() -> dmc.Group:
    return dmc.Group(
        className="flow-help",
        gap=6,
        children=[
            DashIconify(icon="solar:rocket-2-bold", width=16),
            dmc.Text("Launch", size="xs", fw=700),
            DashIconify(icon="solar:arrow-right-linear", width=14),
            DashIconify(icon="solar:clock-circle-bold", width=16),
            dmc.Text("Run log", size="xs", fw=700),
            DashIconify(icon="solar:arrow-right-linear", width=14),
            DashIconify(icon="solar:chart-2-bold", width=16),
            dmc.Text("Details", size="xs", fw=700),
            dmc.Text("Hover any control for details.", size="xs", fw=700, ml="xs", className="flow-hover-copy"),
        ],
    )


def metric_key_from_label(label: str | None) -> str | None:
    if not label:
        return None
    if label in METRIC_LABELS:
        return label
    return METRIC_KEY_BY_LABEL.get(label, label)


def _format_scaled(value: float, units: List[str]) -> str:
    magnitude = abs(float(value))
    unit_index = 0
    while magnitude >= 1000 and unit_index < len(units) - 1:
        magnitude /= 1000
        unit_index += 1
    scaled = float(value) / (1000 ** unit_index)
    if abs(scaled) >= 100:
        text = f"{scaled:,.0f}"
    elif abs(scaled) >= 10:
        text = f"{scaled:,.1f}"
    else:
        text = f"{scaled:,.2f}"
    return f"{text} {units[unit_index]}"


def _format_time_seconds(value: float) -> str:
    if value < 1e-3:
        return f"{value * 1e6:.2f} us"
    if value < 1:
        return f"{value * 1e3:.2f} ms"
    if value < 60:
        return f"{value:.3f} s"
    return humanize_seconds(value)


def _format_gb(value: float) -> str:
    return _format_scaled(value, ["GB", "TB", "PB", "EB"])


def format_metric_value(value: Any, metric_key: str | None = None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        numeric = float(value)
        key = metric_key or ""
        if key in FLOP_COUNT_KEYS:
            return _format_scaled(numeric, ["FLOP", "KFLOP", "MFLOP", "GFLOP", "TFLOP", "PFLOP", "EFLOP"])
        if key in FLOP_RATE_KEYS or (key.endswith("_flops") and key not in FLOP_COUNT_KEYS):
            return _format_scaled(numeric, ["FLOPS", "KFLOPS", "MFLOPS", "GFLOPS", "TFLOPS", "PFLOPS", "EFLOPS"])
        if key in TOKEN_RATE_KEYS:
            return _format_scaled(numeric, ["tok/s", "Ktok/s", "Mtok/s", "Btok/s"])
        if key == "timeout_seconds" and numeric <= 0:
            return "Disabled"
        if key in TIME_KEYS or key.endswith("_time_s") or key.endswith("_seconds") or key.endswith("_s"):
            return _format_time_seconds(numeric)
        if key.endswith("_gb") or key == "hbm_gb":
            return _format_gb(numeric)
        if key == "approx_mfu":
            return f"{numeric * 100:.2f}%"
        if isinstance(value, int):
            return _format_scaled(numeric, ["", "K", "M", "B", "T"]).strip() if abs(value) >= 10000 else f"{value:,}"
        if abs(numeric) >= 10000:
            return _format_scaled(numeric, ["", "K", "M", "B", "T"]).strip()
        if abs(numeric) >= 1000:
            return f"{numeric:,.0f}"
        if abs(numeric) >= 100:
            return f"{numeric:,.1f}"
        if abs(numeric) >= 1:
            return f"{numeric:,.3f}"
        return f"{numeric:.4f}"
    return str(value)


def format_sweep_preview_value(value: float, field_key: str | None) -> str:
    kind = FIELD_TYPES.get(field_key or "", {}).get("kind")
    if kind == "int":
        return f"{int(round(value)):,}"
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    if abs(value) >= 100:
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def build_range_preview(field_key: str | None, mode: str | None, start: Any, end: Any, step: Any, limit: int = RANGE_PREVIEW_LIMIT) -> str:
    if not field_key or mode != "range":
        return ""
    try:
        start_value = float(start)
        end_value = float(end)
        step_value = float(step)
    except (TypeError, ValueError):
        return "Preview: enter start, end, and step size."
    if step_value <= 0:
        return "Preview: step size must be greater than 0."
    if end_value < start_value:
        return "Preview: end must be greater than or equal to start."
    count = int((end_value - start_value) // step_value) + 1
    if count <= 0:
        return "Preview: no values will be generated."
    shown_count = min(count, max(1, limit))
    values = [start_value + idx * step_value for idx in range(shown_count)]
    text_values = [format_sweep_preview_value(value, field_key) for value in values]
    suffix = ", ..." if count > shown_count else ""
    plural = "value" if count == 1 else "values"
    return f"Preview: {', '.join(text_values)}{suffix} ({count:,} {plural})"


def format_primary_metric(result: Dict[str, Any]) -> str:
    metrics = result.get("metrics", {}) or {}
    for key, value in metrics.items():
        if value == result.get("primary_metric_value"):
            return format_metric_value(value, key)
    return format_metric_value(result.get("primary_metric_value"))


def compact_timestamp(raw: str | None) -> str:
    if not raw:
        return ""
    return raw.replace("T", " ").replace("+00:00", " UTC")


def format_finished_job_badge(job: Dict[str, Any]) -> str:
    status = str(job.get("status") or "completed").replace("_", " ").title()
    raw_time = job.get("updated_at") or job.get("created_at")
    if not raw_time:
        return status
    try:
        parsed = datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone()
        return f"{status}, {parsed:%H:%M}"
    except ValueError:
        return status


def to_local_datetime(value: datetime) -> datetime:
    return value.astimezone() if value.tzinfo is not None else value


def config_display_name(config_id: Any) -> str:
    if not config_id:
        return "base config"
    return config_label(config_id)


def memory_exceeded_display(metrics: Dict[str, Any]) -> str:
    try:
        violation_gb = float(metrics.get("memory_violation_gb") or 0.0)
    except (TypeError, ValueError):
        violation_gb = 0.0
    raw_exceeded = metrics.get("memory_exceeded")
    exceeded = raw_exceeded.strip().lower() in {"1", "true", "yes"} if isinstance(raw_exceeded, str) else bool(raw_exceeded)
    exceeded = exceeded or violation_gb > 0
    if not exceeded:
        return "No"
    if violation_gb > 0:
        return format_metric_value(violation_gb, "memory_violation_gb")
    return ">0 GB"


def metrics_with_derived_flops(metrics: Dict[str, Any]) -> Dict[str, Any]:
    derived = dict(metrics or {})
    num_gpus = derived.get("num_gpus")
    try:
        gpu_count = float(num_gpus)
    except (TypeError, ValueError):
        gpu_count = 0.0
    if gpu_count > 0:
        if derived.get("peak_system_flops") is None and derived.get("peak_flops_per_gpu") is not None:
            derived["peak_system_flops"] = float(derived["peak_flops_per_gpu"]) * gpu_count
        if derived.get("achieved_flops_per_gpu") is None and derived.get("achieved_flops") is not None:
            derived["achieved_flops_per_gpu"] = float(derived["achieved_flops"]) / gpu_count
    if derived.get("approx_mfu") is None and derived.get("achieved_flops") is not None and derived.get("peak_system_flops") not in (None, 0):
        derived["approx_mfu"] = float(derived["achieved_flops"]) / float(derived["peak_system_flops"])
    return derived


def detail_metric_rows(metrics: Dict[str, Any]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    inserted_memory = False
    for key, value in metrics_with_derived_flops(metrics).items():
        if key == "memory_violation_gb" or key in DETAIL_HIDDEN_METRIC_KEYS:
            continue
        if key == "memory_exceeded":
            rows.append({"metric": pretty_label(key), "value": memory_exceeded_display(metrics), "metric_key": key})
            inserted_memory = True
            continue
        rows.append({"metric": pretty_label(key), "value": format_metric_value(value, key), "metric_key": key})
    if not inserted_memory and ("memory_violation_gb" in metrics or "memory_exceeded" in metrics):
        rows.append({"metric": pretty_label("memory_exceeded"), "value": memory_exceeded_display(metrics), "metric_key": "memory_exceeded"})
    return rows


def parallelism_summary_from_payload(payload: Dict[str, Any] | None, candidate: Dict[str, Any] | None = None) -> str:
    if candidate:
        pieces = [
            f"TP {candidate.get('tp', 1)}",
            f"CP {candidate.get('cp', 1)}",
            f"PP {candidate.get('pp', 1)}",
            f"DP {candidate.get('dp', 1)}",
            f"EP {candidate.get('ep', 1)}",
        ]
        if candidate.get("replica_count", 1) != 1:
            pieces.append(f"Replicas {candidate.get('replica_count')}")
        return " / ".join(pieces)
    simple = (payload or {}).get("simple", {}) or {}
    pieces = [
        f"TP {simple.get('tp', 1)}",
        f"CP {simple.get('cp', 1)}",
        f"PP {simple.get('pp', 1)}",
        f"DP {simple.get('dp', 1)}",
        f"EP {simple.get('ep', 1)}",
    ]
    if str(simple.get("run_type", "")).lower() == "inference":
        pieces.append(f"Replicas {simple.get('replica_count', 1)}")
    return " / ".join(pieces)


def detail_payload(detail: Dict[str, Any] | None) -> Dict[str, Any]:
    return (detail or {}).get("request_record", {}).get("payload") or (detail or {}).get("request", {}).get("payload") or {}


def detail_model_config(detail: Dict[str, Any], dimension_values: Dict[str, Any] | None = None) -> str:
    payload = detail_payload(detail)
    values = dimension_values or {}
    return str(values.get("model_config") or payload.get("model_preset_id") or "model")


def detail_hardware_config(detail: Dict[str, Any], dimension_values: Dict[str, Any] | None = None) -> str:
    payload = detail_payload(detail)
    values = dimension_values or {}
    return str(values.get("hardware_config") or payload.get("hardware_preset_id") or "hardware")


def detail_run_type(detail: Dict[str, Any]) -> str:
    payload = detail_payload(detail)
    preview = (detail.get("request_record", {}).get("preview") or detail.get("request", {}).get("preview") or {})
    raw = preview.get("run_type") or (payload.get("simple", {}) or {}).get("run_type") or detail.get("run_type") or "training"
    return str(raw).lower()


def detail_sequence_length(detail: Dict[str, Any], dimension_values: Dict[str, Any] | None = None) -> int:
    values = dimension_values or {}
    if values.get("model.seq_len") not in (None, ""):
        try:
            return int(values["model.seq_len"])
        except (TypeError, ValueError):
            return 0
    payload = detail_payload(detail)
    try:
        return int((payload.get("simple", {}) or {}).get("seq_len") or 0)
    except (TypeError, ValueError):
        return 0


def detail_decode_length(detail: Dict[str, Any], dimension_values: Dict[str, Any] | None = None) -> int:
    if detail_run_type(detail) == "training":
        return 0
    values = dimension_values or {}
    if values.get("model.decode_len") not in (None, ""):
        try:
            return int(values["model.decode_len"])
        except (TypeError, ValueError):
            return 0
    payload = detail_payload(detail)
    try:
        return int((payload.get("simple", {}) or {}).get("decode_len") or 0)
    except (TypeError, ValueError):
        return 0


dash._dash_renderer._set_react_version("18.2.0")
ensure_workspace()
MODEL_PRESETS = list_presets("models")
HW_PRESETS = list_presets("hardware")
CPU_CORES = os.cpu_count() or 1
MODEL_LABELS = {item["id"]: item["label"] for item in MODEL_PRESETS}
HW_LABELS = {item["id"]: item["label"] for item in HW_PRESETS}


def preset_records(kind: str) -> List[Dict[str, Any]]:
    return list_presets(kind)


def model_option_label(record: Dict[str, Any]) -> str:
    label = str(record.get("label") or record.get("id") or "")
    run_type = str(record.get("run_type") or "").strip().lower()
    if run_type in {"training", "inference"}:
        return f"{label} ({run_type})"
    return label


def preset_options(kind: str) -> List[Dict[str, str]]:
    if kind == "models":
        return [{"value": item["id"], "label": model_option_label(item)} for item in preset_records(kind)]
    return [{"value": item["id"], "label": item["label"]} for item in preset_records(kind)]


def preset_labels(kind: str) -> Dict[str, str]:
    return {item["id"]: item["label"] for item in preset_records(kind)}


def _preferred_preset_id(records: List[Dict[str, Any]], preferred_id: str) -> str:
    for item in records:
        if item["id"] == preferred_id:
            return item["id"]
    return records[0]["id"]


DEFAULT_MODEL_ID = _preferred_preset_id(MODEL_PRESETS, "Llama2-7B.yaml")
DEFAULT_HW_ID = _preferred_preset_id(HW_PRESETS, "H100_SXM5_80GB.yaml")
DEFAULTS = build_form_defaults(DEFAULT_MODEL_ID, DEFAULT_HW_ID)
DEFAULT_METRIC = get_default_metric_for_run_type(DEFAULTS["run_type"])

app = Dash(__name__, title="RAPID-LLM Workbench", assets_folder=str(Path(__file__).parent / "assets"), suppress_callback_exceptions=True)
server = app.server


def password_matches_required_pattern(password: str) -> bool:
    return password == AUTH_ADMIN_PASSWORD or bool(AUTH_GUEST_PASSWORD_RE.fullmatch(password or ""))


def basic_auth_credentials_from_header(header_value: str | None) -> tuple[str, str] | None:
    if not header_value or not header_value.startswith("Basic "):
        return None
    try:
        decoded = base64.b64decode(header_value.removeprefix("Basic ").strip(), validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return None
    if ":" not in decoded:
        return None
    username, password = decoded.split(":", 1)
    return username, password


def basic_auth_password_from_header(header_value: str | None) -> str | None:
    credentials = basic_auth_credentials_from_header(header_value)
    return credentials[1] if credentials else None


def basic_auth_credentials_are_valid(username: str, password: str) -> bool:
    if username == AUTH_ADMIN_USERNAME:
        return password == AUTH_ADMIN_PASSWORD
    if username == AUTH_GUEST_USERNAME:
        return bool(AUTH_GUEST_PASSWORD_RE.fullmatch(password or ""))
    return False


@server.before_request
def require_basic_auth() -> Response | None:
    credentials = basic_auth_credentials_from_header(request.headers.get("Authorization"))
    if credentials and basic_auth_credentials_are_valid(*credentials):
        return None
    return Response(
        "Authentication required for RAPID-LLM Workbench.",
        401,
        {"WWW-Authenticate": 'Basic realm="RAPID-LLM Workbench", charset="UTF-8"'},
    )


def stat_card(label: str, value: Any, icon: str, color: str) -> dmc.Paper:
    return dmc.Paper(
        radius="lg",
        p="md",
        withBorder=True,
        className="stat-card",
        children=dmc.Group(
            justify="space-between",
            children=[
                dmc.Stack(gap=2, children=[dmc.Text(label, size="sm", c="dimmed"), dmc.Text(str(value), fw=800, size="xl")]),
                dmc.ThemeIcon(size=44, radius="xl", color=color, variant="light", children=DashIconify(icon=icon, width=24)),
            ],
        ),
    )


def _default_payload() -> Dict[str, Any]:
    return {
        "model_preset_id": DEFAULT_MODEL_ID,
        "hardware_preset_id": DEFAULT_HW_ID,
        "run_mode": "sweep",
        "optimize_parallelism": DEFAULT_OPTIMIZE_PARALLELISM,
        "optimizer_preset": "Fast",
        "use_raw_yaml": False,
        "model_yaml_text": "",
        "hardware_yaml_text": "",
        "simple": DEFAULTS["simple"] | {"run_type": DEFAULTS["run_type"]},
        "advanced": DEFAULTS["advanced"],
        "network_dimensions": DEFAULTS["network_dimensions"],
        "dimensions": [],
        "metric": DEFAULT_METRIC,
        "x_axis": None,
        "series_axis": None,
        "worker_count": default_worker_count(),
        "timeout_seconds": 180,
    }


def default_sweep_rows() -> List[Dict[str, Any]]:
    return [{"field": None, "network_targets": list(DEFAULT_NETWORK_SWEEP_TARGETS), "network_apply": "set", "mode": "values", "list_text": "", "config_values": [], "start": None, "end": None, "step_or_points": None} for _ in range(3)]


def is_network_sweep_field(field_key: str | None) -> bool:
    return field_key in NETWORK_SWEEP_FIELD_KEYS


def is_network_bundle_field(field_key: str | None) -> bool:
    return bool(re.match(r"^hardware\.network\.(set|scale)\.", str(field_key or "")))


def network_targets_from_bundle_field(field_key: str | None) -> List[str]:
    match = re.match(r"^hardware\.network\.(set|scale)\.([A-Za-z0-9_.-]+)$", str(field_key or ""))
    if not match:
        return []
    return [NETWORK_SWEEP_TARGET_BY_SLUG[slug]["value"] for slug in match.group(2).split(".") if slug in NETWORK_SWEEP_TARGET_BY_SLUG]


def display_sweep_field_value(field_key: str | None) -> str | None:
    return NETWORK_SWEEP_GROUP_VALUE if is_network_sweep_field(field_key) or is_network_bundle_field(field_key) else field_key


def selected_network_sweep_targets(row: Dict[str, Any] | None) -> List[str]:
    row = row or {}
    raw_targets = row.get("network_targets")
    targets = [str(item) for item in (raw_targets or []) if str(item) in NETWORK_SWEEP_FIELD_KEYS]
    if targets:
        return targets
    bundle_targets = network_targets_from_bundle_field(row.get("field"))
    if bundle_targets:
        return bundle_targets
    candidate = row.get("network_field") or row.get("field")
    return [str(candidate)] if is_network_sweep_field(candidate) else list(DEFAULT_NETWORK_SWEEP_TARGETS)


def selected_network_apply_mode(row: Dict[str, Any] | None) -> str:
    row = row or {}
    if row.get("network_apply") not in {"set", "scale"} and is_network_bundle_field(row.get("field")):
        return "scale" if ".scale." in str(row.get("field")) else "set"
    return str(row.get("network_apply")) if row.get("network_apply") in {"set", "scale"} else "set"


def _network_target_slug(field_key: str) -> str:
    return str(NETWORK_SWEEP_TARGET_BY_VALUE[field_key]["slug"])


def network_sweep_dimension_key(targets: List[str] | None, apply_mode: str | None) -> str | None:
    selected = [target for target in (targets or []) if target in NETWORK_SWEEP_FIELD_KEYS]
    if not selected:
        return None
    mode = "scale" if apply_mode == "scale" else "set"
    if mode == "set" and len(selected) == 1:
        return selected[0]
    slugs = ".".join(_network_target_slug(target) for target in selected)
    return f"hardware.network.{mode}.{slugs}"


def resolve_sweep_field(field_key: str | None, network_targets: List[str] | str | None = None, network_apply: str | None = None) -> str | None:
    if field_key == NETWORK_SWEEP_GROUP_VALUE:
        if network_targets is None:
            network_targets = list(DEFAULT_NETWORK_SWEEP_TARGETS)
        if isinstance(network_targets, str):
            network_targets = [network_targets]
        return network_sweep_dimension_key(network_targets, network_apply)
    return field_key


def _valid_preset_ids(kind: str) -> set[str]:
    return {item["id"] for item in preset_records(kind)}


def _valid_preset_id(kind: str, candidate: Any, fallback: str) -> str:
    text = str(candidate) if candidate else ""
    return text if text in _valid_preset_ids(kind) else fallback


def _valid_config_list(kind: str, values: Any, fallback: str) -> List[str]:
    valid = _valid_preset_ids(kind)
    selected = [str(item) for item in (values or []) if str(item) in valid]
    if fallback not in selected:
        selected.insert(0, fallback)
    return selected or [fallback]


def initial_ui_state(*, ignore_saved: bool = False) -> Dict[str, Any]:
    saved = {} if ignore_saved else load_last_ui_state()
    model_preset = _valid_preset_id("models", saved.get("model_preset"), DEFAULT_MODEL_ID)
    hardware_preset = _valid_preset_id("hardware", saved.get("hardware_preset"), DEFAULT_HW_ID)
    model_run_configs = _valid_config_list("models", saved.get("model_run_configs"), model_preset)
    hardware_run_configs = _valid_config_list("hardware", saved.get("hardware_run_configs"), hardware_preset)
    defaults = build_form_defaults(model_preset, hardware_preset)
    active_tab = saved.get("active_config_tab")
    kind, config_id = parse_config_tab(active_tab)
    if kind == "models" and config_id not in model_run_configs:
        active_tab = config_tab_value("models", model_preset)
    elif kind == "hardware" and config_id not in hardware_run_configs:
        active_tab = config_tab_value("hardware", hardware_preset)
    elif kind not in {"models", "hardware"}:
        active_tab = config_tab_value("models", model_preset)
    return {
        "model_preset": model_preset,
        "hardware_preset": hardware_preset,
        "model_run_configs": model_run_configs,
        "hardware_run_configs": hardware_run_configs,
        "active_config_tab": active_tab,
        "run_mode": saved.get("run_mode") or "sweep",
        "optimize_parallelism": saved.get("optimize_parallelism") if "optimize_parallelism" in saved else DEFAULT_OPTIMIZE_PARALLELISM,
        "optimizer_preset": saved.get("optimizer_preset") or "Fast",
        "sweep_rows": saved.get("sweep_rows") or default_sweep_rows(),
        "metric": saved.get("metric") or get_default_metric_for_run_type(defaults["run_type"]),
        "x_axis": saved.get("x_axis"),
        "series_axis": saved.get("series_axis"),
        "worker_count": int(saved.get("worker_count") or default_worker_count()),
        "timeout_seconds": int(saved.get("timeout_seconds")) if saved.get("timeout_seconds") is not None else 180,
        "defaults": defaults,
    }


def state_dimensions(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = state.get("sweep_rows") or default_sweep_rows()
    return _dimensions_from_inputs(
        [
            {
                "field": row.get("field"),
                "network_field": row.get("network_field"),
                "network_targets": selected_network_sweep_targets(row),
                "network_apply": selected_network_apply_mode(row),
                "mode": row.get("mode"),
                "list_text": row.get("list_text"),
                "config_values": row.get("config_values"),
                "start": row.get("start"),
                "end": row.get("end"),
                "step_or_points": row.get("step_or_points"),
            }
            for row in rows
        ]
    )


def initial_payload_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    defaults = state["defaults"]
    dimensions = config_dimensions_from_selection(state["model_run_configs"], state["hardware_run_configs"], state["model_preset"], state["hardware_preset"])
    dimensions.extend(state_dimensions(state))
    return collect_payload(
        state["model_preset"],
        state["hardware_preset"],
        state["run_mode"],
        bool(state["optimize_parallelism"]),
        state["optimizer_preset"],
        defaults["simple"] | {"run_type": defaults["run_type"]},
        defaults["advanced"],
        defaults["network_dimensions"],
        dimensions,
        state["metric"],
        state["x_axis"],
        state["series_axis"],
        state["worker_count"],
        state["timeout_seconds"],
    )


def selected_config_values(values: List[str] | None, fallback: str) -> List[str]:
    selected = [item for item in (values or []) if item]
    return selected or [fallback]


def selected_active_value(values: List[str] | None, fallback: str, current: str | None) -> str:
    selected = selected_config_values(values, fallback)
    return current if current in selected else selected[0]


def config_tab_value(kind: str, config_id: str) -> str:
    return f"{kind}::{config_id}"


def parse_config_tab(value: str | None) -> tuple[str | None, str | None]:
    if not value or "::" not in value:
        return None, None
    kind, config_id = value.split("::", 1)
    if kind not in {"models", "hardware"} or not config_id:
        return None, None
    return kind, config_id


def active_config_tab_value(
    model_ids: List[str] | None,
    hardware_ids: List[str] | None,
    fallback_model: str,
    fallback_hardware: str,
    current_tab: str | None,
    current_model: str | None,
    current_hardware: str | None,
) -> str:
    selected_models = selected_config_values(model_ids, fallback_model)
    selected_hardware = selected_config_values(hardware_ids, fallback_hardware)
    current_kind, current_id = parse_config_tab(current_tab)
    if current_kind == "models" and current_id in selected_models:
        return config_tab_value("models", current_id)
    if current_kind == "hardware" and current_id in selected_hardware:
        return config_tab_value("hardware", current_id)
    if current_model in selected_models:
        return config_tab_value("models", current_model or selected_models[0])
    if current_hardware in selected_hardware:
        return config_tab_value("hardware", current_hardware or selected_hardware[0])
    return config_tab_value("models", selected_models[0]) if selected_models else config_tab_value("hardware", selected_hardware[0])


def active_config_tab_for_selection_change(
    model_ids: List[str] | None,
    hardware_ids: List[str] | None,
    fallback_model: str,
    fallback_hardware: str,
    current_tab: str | None,
    current_model: str | None,
    current_hardware: str | None,
    triggered_id: str | None,
) -> str:
    selected_models = selected_config_values(model_ids, fallback_model)
    selected_hardware = selected_config_values(hardware_ids, fallback_hardware)
    if triggered_id == "model-run-configs":
        return config_tab_value("models", selected_models[-1])
    if triggered_id == "hardware-run-configs":
        return config_tab_value("hardware", selected_hardware[-1])
    return active_config_tab_value(model_ids, hardware_ids, fallback_model, fallback_hardware, current_tab, current_model, current_hardware)


def config_workbook_tabs_children(model_ids: List[str] | None, hardware_ids: List[str] | None, model_labels: Dict[str, str], hardware_labels: Dict[str, str]) -> List[Any]:
    tabs = []
    for kind, values, labels, icon, badge in [
        ("models", selected_config_values(model_ids, DEFAULT_MODEL_ID), model_labels, "solar:document-text-bold", "Model"),
        ("hardware", selected_config_values(hardware_ids, DEFAULT_HW_ID), hardware_labels, "solar:cpu-bold", "Hardware"),
    ]:
        for value in values:
            label = labels.get(value, Path(value).stem)
            tabs.append(
                dmc.TabsTab(
                    html.Div(
                        className="config-workbook-tab-content",
                        children=[
                            html.Span(badge, className="config-workbook-tab-kind"),
                            html.Span(label, className="config-workbook-tab-title"),
                        ],
                    ),
                    value=config_tab_value(kind, value),
                    leftSection=DashIconify(icon=icon, width=15),
                    className="config-workbook-tab",
                    attributes={"title": f"{badge}: {label}"},
                )
            )
    return [dmc.TabsList(className="config-workbook-tab-list", children=tabs)]


def config_dimensions_from_selection(model_ids: List[str] | None, hardware_ids: List[str] | None, primary_model_id: str, primary_hardware_id: str) -> List[Dict[str, Any]]:
    dimensions: List[Dict[str, Any]] = []
    selected_models = selected_config_values(model_ids, primary_model_id)
    selected_hardware = selected_config_values(hardware_ids, primary_hardware_id)
    if len(selected_models) > 1:
        dimensions.append({"field_key": "model_config", "mode": "values", "config_values": selected_models})
    if len(selected_hardware) > 1:
        dimensions.append({"field_key": "hardware_config", "mode": "values", "config_values": selected_hardware})
    return dimensions


def render_error_summary(errors: List[str]) -> dmc.Stack:
    return dmc.Stack(
        gap="sm",
        children=[
            dmc.Alert(
                item,
                title="Launch plan error",
                color="red",
                radius="lg",
                className="launch-error-alert",
                icon=DashIconify(icon="solar:danger-triangle-bold", width=22),
            )
            for item in errors
        ],
    )


def hf_import_status_alert(message: str, *, ok: bool) -> dmc.Alert:
    return dmc.Alert(
        message,
        title="Hugging Face import" if ok else "Hugging Face import blocked",
        color="green" if ok else "red",
        radius="lg",
        mt="xs",
        icon=DashIconify(icon="solar:check-circle-bold" if ok else "solar:danger-triangle-bold", width=20),
    )


def clamp_percent(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(100.0, numeric))


def branded_progress_bar(progress: Any, count_label: str, status_label: str, *, show_run_log_button: bool = False) -> html.Div:
    percent = clamp_percent(progress)
    percent_text = f"{percent:.0f}%"
    meta_children = [
        html.Span(count_label, className="rapid-progress-meta-pill"),
        html.Span(status_label, className="rapid-progress-meta-pill rapid-progress-meta-status"),
    ]
    if show_run_log_button:
        meta_children.append(
            html.Div(
                className="rapid-progress-complete-action",
                children=with_tip(
                    dmc.Button(
                        "Run Log",
                        id="progress-run-log-button",
                        className="rapid-progress-run-log-button",
                        radius="xl",
                        leftSection=DashIconify(icon="solar:clock-circle-bold", width=18),
                    ),
                    HELP_TEXT["progress_run_log"],
                ),
            )
        )
    return html.Div(
        className="rapid-progress-shell",
        children=[
            html.Div(
                className="rapid-progress-track",
                role="progressbar",
                style={"--rapid-progress": f"{percent:.2f}%"},
                **{
                    "aria-valuemin": "0",
                    "aria-valuemax": "100",
                    "aria-valuenow": f"{percent:.2f}",
                    "aria-label": f"RAPID-LLM job progress {percent_text}",
                },
                children=[
                    html.Div(className="rapid-progress-fill"),
                    html.Div("RAPID-LLM", className="rapid-progress-core", **{"data-label": "RAPID-LLM"}),
                    html.Div(percent_text, className="rapid-progress-percent"),
                ],
            ),
            html.Div(
                className="rapid-progress-meta",
                children=meta_children,
            ),
        ],
    )


def detail_loading_placeholder(selected_detail: Dict[str, Any] | None) -> dmc.Paper:
    label = "details" if not selected_detail else f"{selected_detail.get('kind', 'job')} details"
    return dmc.Paper(
        radius="xl",
        p="xl",
        withBorder=True,
        children=dmc.Group(
            gap="md",
            align="center",
            children=[
                dmc.Loader(size="md", color="blue"),
                dmc.Stack(
                    gap=2,
                    children=[
                        dmc.Text(f"Loading {label}...", fw=800),
                        dmc.Text("Preparing large sweep tables and plots.", size="sm", c="dimmed"),
                    ],
                ),
            ],
        ),
    )


def render_preview_summary(preview: Dict[str, Any], metric: str) -> dmc.Stack:
    if not preview.get("ok"):
        return render_error_summary(preview.get("errors", []))
    telemetry = get_telemetry()
    return dmc.Stack(
        gap="sm",
        children=[
            dmc.Alert(
                "Worst-case wall clock assumes every simulator invocation hits its own timeout. Expected runtime is often 10-70% of worst-case for small and medium launches, but it can rapidly approach worst-case on larger runs, especially beyond 256 GPUs. If final runtime lands close to worst-case, increase the timeout for higher result fidelity.",
                color="blue",
                radius="lg",
                className="preview-runtime-note",
            ),
            dmc.SimpleGrid(
                cols={"base": 1, "sm": 2, "lg": 4},
                spacing="sm",
                children=[
                    stat_card("Top-level cases", format_metric_value(preview["top_level_case_count"], "case_count"), "solar:box-bold", "teal"),
                    stat_card("Simulator invocations", format_metric_value(preview["total_invocations"], "invocation_count"), "solar:play-bold", "blue"),
                    stat_card("Worst-case wall clock", format_worst_case_wall_clock(preview["worst_case_wall_clock_s"]), "solar:clock-circle-bold", "orange"),
                    stat_card("Available RAM", format_metric_value(telemetry["available_ram_gb"], "available_ram_gb"), "solar:memory-bold", "grape"),
                ],
            ),
            dmc.Group(
                gap="sm",
                children=[
                    dmc.Badge(f"Metric: {METRIC_LABELS.get(metric, metric)}", radius="xl", color="teal", variant="light"),
                    dmc.Badge(f"Workers: {format_metric_value(preview['worker_count'], 'worker_count')}", radius="xl", color="blue", variant="light"),
                    dmc.Badge(f"Timeout: {format_metric_value(preview['timeout_seconds'], 'timeout_seconds')}", radius="xl", color="cyan", variant="light"),
                ],
            ),
            dmc.Stack(children=[dmc.Alert(w, color="red", radius="lg") for w in preview.get("warnings", [])]) if preview.get("warnings") else html.Div(),
        ],
    )


def launch_button_label(preview: Dict[str, Any] | None) -> str:
    count = int((preview or {}).get("total_invocations") or 0)
    noun = "run" if count == 1 else "runs"
    return f"Launch {count} {noun}"


PREVIEW_LOAD_ONLY_TRIGGER_IDS = {"config-editor-tabs", "model-preset", "hardware-preset"}


def preview_rebuild_is_load_only(triggered_prop_ids: Dict[str, Any] | None) -> bool:
    triggered_ids = {str(prop_id).split(".", 1)[0] for prop_id in (triggered_prop_ids or {})}
    return bool(triggered_ids) and triggered_ids <= PREVIEW_LOAD_ONLY_TRIGGER_IDS


def progress_count_label(job: Dict[str, Any], *, terminal: bool = False) -> str:
    total = int(job.get("progress_total") or 0)
    completed = int(job.get("progress_completed") or 0)
    if terminal:
        if total > 0:
            return f"{total} / {total}"
        return "Done"
    return f"{completed} / {total}"


def _parse_job_timestamp(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        return to_local_datetime(datetime.fromisoformat(str(raw).replace("Z", "+00:00")))
    except ValueError:
        return None


def job_eta_readout(job: Dict[str, Any], now: datetime | None = None) -> str:
    total = int(job.get("progress_total") or 0)
    completed = int(job.get("progress_completed") or 0)
    if total <= 0:
        return "ETA: unavailable"
    if completed >= total:
        return "ETA: complete"
    if completed <= 0:
        return "ETA: calculating"
    started_at = _parse_job_timestamp(job.get("created_at") or job.get("updated_at"))
    if not started_at:
        return "ETA: calculating"
    if now is None:
        now = datetime.now().astimezone() if started_at.tzinfo else datetime.now()
    elif started_at.tzinfo and now.tzinfo is None:
        now = now.replace(tzinfo=started_at.tzinfo)
    elif now.tzinfo and started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=now.tzinfo)
    elapsed_s = max(0.0, (now - started_at).total_seconds())
    if elapsed_s <= 0:
        return "ETA: calculating"
    remaining_s = (elapsed_s / max(1, completed)) * max(0, total - completed)
    finish_at = now + timedelta(seconds=remaining_s)
    return f"ETA: ~{humanize_seconds(remaining_s)} remaining ({finish_at:%H:%M})"


def history_title_component(item: Dict[str, Any]) -> dmc.Group:
    children: List[Any] = [dmc.Text(item["title"], fw=700, size="lg")]
    if int(item.get("title_duplicate_count") or 0) > 1:
        children.append(html.Span(f"#{int(item.get('title_index') or 1)}", className="history-title-index"))
    return dmc.Group(gap=8, align="baseline", children=children)


def network_sweep_target_grid(index: int, selected_targets: List[str]) -> dmc.CheckboxGroup:
    return dmc.CheckboxGroup(
        id=f"dim-{index}-network-targets",
        value=selected_targets,
        children=dmc.SimpleGrid(
            cols={"base": 1, "sm": 3},
            spacing="xs",
            children=[
                html.Div(
                    className="network-sweep-dim-group",
                    children=[
                        dmc.Text(f"Dimension {dim}", fw=800, size="xs", c="dimmed"),
                        dmc.Checkbox(
                            value=f"hardware.network.dim{dim}.bandwidth_gbs",
                            label="Bandwidth",
                            size="xs",
                            className="network-sweep-check",
                        ),
                        dmc.Checkbox(
                            value=f"hardware.network.dim{dim}.latency_s",
                            label="Latency",
                            size="xs",
                            className="network-sweep-check",
                        ),
                    ],
                )
                for dim in range(3)
            ],
        ),
    )


def dim_card(index: int, row: Dict[str, Any] | None = None) -> dmc.Paper:
    row = row or default_sweep_rows()[0]
    field_value = display_sweep_field_value(row.get("field"))
    network_targets = selected_network_sweep_targets(row)
    network_apply = selected_network_apply_mode(row)
    return dmc.Paper(
        radius="xl",
        p="md",
        withBorder=True,
        className="dimension-card",
        children=dmc.Stack(
            gap="sm",
            children=[
                dmc.Group(justify="space-between", children=[dmc.Text(f"Dimension {index}", fw=700, size="sm"), dmc.Badge("Optional", variant="light", color="gray")]),
                with_tip(dmc.Select(id=f"dim-{index}-field", label="Field", placeholder="Select a sweep field", value=field_value, data=SWEEP_FIELD_OPTIONS, clearable=True), "Select a workload or hardware-scaling field to vary. Use Network for per-dimension bandwidth and latency; individual parallelism axes are not sweep fields."),
                html.Div(
                    id=f"dim-{index}-network-wrap",
                    className="sweep-submenu",
                    style={"display": "none"},
                    children=dmc.Stack(
                        gap="xs",
                        children=[
                            dmc.Group(
                                justify="space-between",
                                align="center",
                                gap="xs",
                                children=[
                                    dmc.Text("Network settings", fw=800, size="sm"),
                                    with_tip(
                                        dmc.SegmentedControl(
                                            id=f"dim-{index}-network-apply",
                                            size="xs",
                                            value=network_apply,
                                            data=[{"label": "Set values", "value": "set"}, {"label": "Scale baseline", "value": "scale"}],
                                        ),
                                        HELP_TEXT["network_sweep_apply"],
                                    ),
                                ],
                            ),
                            with_tip(network_sweep_target_grid(index, network_targets), HELP_TEXT["network_sweep_field"]),
                            dmc.Text("Select one target for a direct sweep, multiple same-unit targets to set together, or Scale baseline to multiply selected bandwidth/latency baselines by each sweep value.", size="xs", c="dimmed", className="network-sweep-hint"),
                        ],
                    ),
                ),
                html.Div(
                    id=f"dim-{index}-mode-wrap",
                    style={"display": "none"},
                    children=with_tip(dmc.SegmentedControl(id=f"dim-{index}-mode", fullWidth=True, data=[{"label": "Values", "value": "values"}, {"label": "Range", "value": "range"}], value=row.get("mode") or "values"), "Values uses a comma-separated list; Range uses start, end, and step size."),
                ),
                html.Div(id=f"dim-{index}-values-wrap", style={"display": "none"}, children=with_tip(dmc.TextInput(id=f"dim-{index}-list", label="Values", value=row.get("list_text") or "", placeholder="Example: 8192, 16384"), "Comma-separated values to run for the selected sweep field.")),
                html.Div(id=f"dim-{index}-configs-wrap", style={"display": "none"}, children=with_tip(dmc.MultiSelect(id=f"dim-{index}-configs", label="Preset values", value=row.get("config_values") or [], data=[], placeholder="Pick config files"), "Configuration file values for this sweep dimension.")),
                html.Div(
                    id=f"dim-{index}-range-wrap",
                    style={"display": "none"},
                    children=dmc.Stack(
                        gap="xs",
                        children=[
                            dmc.SimpleGrid(
                                cols={"base": 1, "sm": 3},
                                spacing="sm",
                                children=[
                                    with_tip(dmc.NumberInput(id=f"dim-{index}-start", label="Start", value=row.get("start"), allowDecimal=True), "First value in the numeric sweep range."),
                                    with_tip(dmc.NumberInput(id=f"dim-{index}-end", label="End", value=row.get("end"), allowDecimal=True), "Last value included in the numeric sweep range."),
                                    with_tip(dmc.NumberInput(id=f"dim-{index}-step_or_points", label="Step size", value=row.get("step_or_points"), allowDecimal=True), "Increment between adjacent values in the numeric sweep range."),
                                ],
                            ),
                            html.Div(id=f"dim-{index}-range-preview", className="range-preview", children="Preview: enter start, end, and step size."),
                        ],
                    ),
                ),
            ],
        ),
    )


def topology_axis_badges(axes: List[str] | None) -> dmc.Group:
    items = [str(axis).upper() for axis in (axes or []) if str(axis).strip()]
    if not items:
        items = ["none"]
    def _badge_color(axis: str) -> str:
        return "gray" if axis.lower() in {"none", "disabled"} else "blue"
    return dmc.Group(
        gap=4,
        children=[
            dmc.Badge(axis, size="xs", radius="sm", color=_badge_color(axis), variant="light")
            for axis in items
        ],
    )


def parallelism_topology_preview(pp_dimension: str | None) -> dmc.Paper:
    mode = str(pp_dimension or "dim1_shared")
    if mode == "dim1":
        mode = "dim1_dim2"
    elif mode in {"dim2", "dim2_shared"}:
        mode = "dim1_dim2"
    if mode == "dim1_dim2":
        dim1_axes, dim1_note = ["PP"], "Pipeline axis"
        dim2_axes, dim2_note = ["DP"], "Outermost data-parallel axis"
    else:
        dim1_axes, dim1_note = ["PP", "DP"], "Pipeline and data share Dimension 1"
        dim2_axes, dim2_note = ["disabled"], "Dimension 2 off"
    rows = [
        ("Dimension 0", ["TP", "CP", "EP"], "Fixed inner transformer/expert axis (intra-DGX box/waferscale)"),
        ("Dimension 1", dim1_axes, dim1_note),
        ("Dimension 2", dim2_axes, dim2_note),
    ]
    return dmc.Paper(
        radius="md",
        p="sm",
        withBorder=True,
        className="topology-mapping-preview",
        children=dmc.SimpleGrid(
            cols={"base": 1, "sm": 3},
            spacing="xs",
            children=[
                html.Div(
                    className="topology-mapping-cell",
                    children=[
                        dmc.Text(label, fw=800, size="sm"),
                        topology_axis_badges(axes),
                        dmc.Text(note, size="xs", c="dimmed"),
                    ],
                )
                for label, axes, note in rows
            ],
        ),
    )


def disabled_network_dimension_indices(pp_dimension: str | None) -> set[int]:
    mode = str(pp_dimension or "dim1_shared")
    if mode == "dim1":
        mode = "dim1_dim2"
    elif mode in {"dim2", "dim2_shared"}:
        mode = "dim1_dim2"
    if mode == "dim1_shared":
        return {2}
    return set()


def network_editor(defaults: List[Dict[str, Any]], pp_dimension: str | None = None) -> List[dmc.Paper]:
    rows: List[dmc.Paper] = []
    disabled_indices = disabled_network_dimension_indices(pp_dimension)
    for idx, row in enumerate(defaults):
        disabled = idx in disabled_indices
        rows.append(
            dmc.Paper(
                radius="lg",
                p="sm",
                withBorder=True,
                className="network-row",
                children=dmc.Stack(
                    gap="xs",
                    children=[
                        dmc.Group(
                            justify="space-between",
                            gap="xs",
                            children=[
                                dmc.Text(row["label"], fw=700, size="sm"),
                                topology_axis_badges(row.get("parallelisms")),
                            ],
                        ),
                        dmc.SimpleGrid(
                            cols={"base": 1, "sm": 4},
                            spacing="sm",
                            children=[
                                with_tip(dmc.Select(id={"type": "net-topology", "index": idx}, label="Topology", value=row["topology_type"], data=NETWORK_TOPOLOGY_OPTIONS, disabled=disabled), HELP_TEXT["network_topology"]),
                                with_tip(dmc.TextInput(id={"type": "net-bandwidth", "index": idx}, label="Bandwidth", value=str(row["bandwidth"]), disabled=disabled), HELP_TEXT["network_bandwidth"]),
                                with_tip(dmc.NumberInput(id={"type": "net-latency", "index": idx}, label="Latency (s)", min=0, step=0.000001, decimalScale=9, value=float(row.get("latency", 0.0) or 0.0), disabled=disabled), HELP_TEXT["network_latency"]),
                                with_tip(dmc.NumberInput(id={"type": "net-util", "index": idx}, label="Utilization", min=0, max=1, step=0.01, decimalScale=3, value=float(row["util"]), disabled=disabled), HELP_TEXT["network_util"]),
                            ],
                        ),
                    ],
                ),
            )
        )
    return rows


def build_header() -> html.Div:
    return html.Div(
        className="app-header-shell",
        children=dmc.Container(
            fluid=True,
            px="lg",
            children=dmc.Group(
                className="topbar",
                justify="space-between",
                h="100%",
                children=[
                    dmc.Group(
                        className="topbar-brand",
                        gap="md",
                        children=[
                            dmc.Stack(
                                gap=2,
                                children=[
                                    with_tip(
                                        html.Div(
                                            className="topbar-title-block",
                                            children=[
                                                dmc.Title("RAPID-LLM Workbench", order=2, c="#ffffff", className="topbar-title"),
                                                dmc.Text("v0.9, last updated 4/27/2026", className="topbar-version"),
                                            ],
                                        ),
                                        HELP_TEXT["app_title"],
                                    ),
                                    with_tip(flow_help(), HELP_TEXT["app_flow"]),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="topbar-logo-slot",
                        children=html.Div(
                            className="nanocad-logo-frame",
                            children=html.Img(src=app.get_asset_url(APP_LOGO_ASSET), className="nanocad-logo", alt="NanoCAD"),
                        ),
                    ),
                    dmc.Group(
                        className="telemetry-pills",
                        gap="sm",
                        children=[
                            with_tip(dmc.Badge("RAM --", id="telemetry-ram", size="lg", radius="xl", color="teal", variant="light", className="telemetry-badge"), HELP_TEXT["telemetry_ram"]),
                            with_tip(dmc.Badge("CPU --", id="telemetry-cpu", size="lg", radius="xl", color="cyan", variant="light", className="telemetry-badge"), HELP_TEXT["telemetry_cpu"]),
                            with_tip(dmc.Badge("Idle", id="telemetry-job", size="lg", radius="xl", color="blue", variant="light", className="telemetry-badge"), HELP_TEXT["telemetry_job"]),
                        ],
                    ),
                ],
            ),
        ),
    )


def create_layout() -> dmc.MantineProvider:
    state = initial_ui_state()
    metric_options = get_metric_options(state["defaults"]["run_type"])
    return dmc.MantineProvider(
        theme={"primaryColor": "blue", "fontFamily": "Arial, Calibri, Segoe UI, sans-serif", "defaultRadius": "sm"},
        children=html.Div(
            className="app-shell",
            children=[
                build_header(),
                dcc.Interval(id="telemetry-poller", interval=5000, n_intervals=0),
                dcc.Interval(id="job-poller", interval=1500, n_intervals=0),
                dcc.Store(id="preview-store"),
                dcc.Store(id="selected-detail-store"),
                dcc.Store(id="history-refresh-store"),
                dcc.Download(id="plot-download"),
                dcc.Download(id="table-download"),
                html.Div(
                    id="detail-overlay",
                    className="detail-overlay",
                    style={"display": "none"},
                    children=[
                        html.Button(id="detail-backdrop", className="detail-backdrop", **{"aria-label": "Close details"}),
                        html.Div(
                            className="detail-dialog",
                            role="dialog",
                            **{"aria-modal": "true"},
                            children=[
                                html.Div(
                                    className="detail-dialog-header",
                                    children=[
                                        html.Div(
                                            className="detail-dialog-title-row",
                                            children=[
                                                html.Div(id="detail-modal-title"),
                                                dmc.ActionIcon(
                                                    DashIconify(icon="solar:close-circle-bold", width=24),
                                                    id="detail-close-button",
                                                    className="detail-close-button",
                                                    variant="light",
                                                    radius="xl",
                                                    size="xl",
                                                    **{"aria-label": "Close details"},
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            id="detail-plot-toolbar",
                                            className="detail-plot-toolbar",
                                            style={"display": "none"},
                                            children=dmc.Stack(
                                                gap=6,
                                                children=[
                                                    dmc.Group(
                                                        gap="sm",
                                                        align="center",
                                                        children=[
                                                            dmc.SegmentedControl(
                                                                id="detail-plot-type",
                                                                value="line",
                                                                data=[
                                                                    {"label": "Line Plot", "value": "line"},
                                                                    {"label": "Scatter", "value": "scatter"},
                                                                    {"label": "Bar chart", "value": "bar"},
                                                                ],
                                                            ),
                                                            with_tip(dmc.Button("Download plot", id="save-plot-button", variant="light", leftSection=DashIconify(icon="solar:download-bold", width=18)), "Save the current plot under this job's plots folder and download it as a PNG."),
                                                        ],
                                                    ),
                                                    dmc.Group(
                                                        gap="sm",
                                                        align="center",
                                                        children=[
                                                            with_tip(
                                                                dmc.SegmentedControl(
                                                                    id="detail-display-mode",
                                                                    value="top",
                                                                    data=[
                                                                        {"label": "Top results", "value": "top"},
                                                                        {"label": "All results (slow)", "value": "full"},
                                                                    ],
                                                                ),
                                                                "Top results loads the best capped rows by the selected metric. All results renders every stored case and can be slow for large sweeps.",
                                                            ),
                                                            dmc.Text("Download full table", size="xs", fw=800),
                                                            with_tip(dmc.Button("CSV", id="export-table-csv-button", variant="light", size="xs"), "Save every stored detail row to CSV and download it, regardless of the current display limit."),
                                                            with_tip(dmc.Button("JSON", id="export-table-json-button", variant="light", size="xs"), "Save every stored detail row to JSON and download it, regardless of the current display limit."),
                                                        ],
                                                    ),
                                                    dmc.Text(id="plot-save-status", size="xs", c="dimmed"),
                                                    dmc.Text(id="table-export-status", size="xs", c="dimmed"),
                                                ],
                                            ),
                                        ),
                                    ],
                                ),
                                html.Div(id="details-panel", className="detail-dialog-body"),
                            ],
                        ),
                    ],
                ),
                dmc.Container(
                    fluid=True,
                    px="lg",
                    py="lg",
                    className="app-main",
                    children=[
                        dmc.Tabs(
                            id="workspace-tabs",
                            value="builder",
                            children=[
                                dmc.TabsList(
                                    grow=True,
                                    mb="md",
                                    className="workspace-tabs-list",
                                    children=[
                                        dmc.TabsTab("1 Launch", value="builder", leftSection=DashIconify(icon="solar:rocket-2-bold")),
                                        dmc.TabsTab("2 Run log", value="history", leftSection=DashIconify(icon="solar:clock-circle-bold")),
                                    ],
                                ),
                                dmc.TabsPanel(value="builder", children=builder_panel(metric_options, state)),
                                dmc.TabsPanel(value="history", children=html.Div(id="history-panel", children=render_history_panel())),
                            ],
                        )
                    ],
                ),
                html.Footer(APP_CREDIT_TEXT, className="app-credit"),
            ],
        ),
    )


def builder_panel(metric_options: List[Dict[str, str]], state: Dict[str, Any] | None = None) -> dmc.Grid:
    state = state or initial_ui_state(ignore_saved=True)
    return dmc.Grid(
        className="builder-grid",
        gutter="lg",
        children=[
            dmc.GridCol(span={"base": 12, "xl": 5}, children=left_column(metric_options, state)),
            dmc.GridCol(span={"base": 12, "xl": 7}, children=right_column(state)),
        ],
    )


def left_column(metric_options: List[Dict[str, str]], state: Dict[str, Any]) -> dmc.Stack:
    hero = dmc.Paper(
        className="hero-card",
        radius="xl",
        p="lg",
        withBorder=True,
        children=dmc.Stack(
            gap="sm",
            children=[
                dmc.Group(
                    justify="space-between",
                    children=[
                        dmc.Badge("Launch Builder", radius="xl", color="teal", variant="light"),
                        dmc.Text("Local execution", size="sm", c="dimmed"),
                    ],
                ),
                html.H2("Launch runs, review log, inspect details", className="hero-heading"),
                html.P(
                    "Pick YAML files, adjust run options, launch the job, then open Details from the run log.",
                    className="hero-copy",
                ),
            ],
        ),
    )
    return dmc.Stack(
        className="builder-left-scroll",
        gap="lg",
        children=[
            hero,
            run_setup_card(state),
            config_options_card(state),
            html.Div(id="sweep-dimensions-section", children=dimensions_card(state)),
            launch_controls_card(metric_options, state),
        ],
    )


def right_column(state: Dict[str, Any]) -> dmc.Stack:
    initial_preview = build_launch_preview(initial_payload_from_state(state))
    preview_card = dmc.Paper(
        radius="xl",
        p="lg",
        withBorder=True,
        className="preview-card",
        children=dmc.Stack(
            gap="md",
            children=[
                dmc.Title("Launch Plan", order=3),
                html.Div(id="preview-summary", children=render_preview_summary(initial_preview, DEFAULT_METRIC)),
            ],
        ),
    )
    status_card = dmc.Paper(
        radius="xl",
        p="lg",
        withBorder=True,
        children=dmc.Stack(
            gap="md",
            children=[
                dmc.Group(
                    justify="space-between",
                    children=[
                        dmc.Title("Live Status", order=3),
                        dmc.Badge("Current job", color="blue", variant="light"),
                    ],
                ),
                                html.Div(id="active-job-panel", children=render_active_job_panel()),
            ],
        ),
    )
    context_card = dmc.Paper(
        radius="xl",
        p="lg",
        withBorder=True,
        children=dmc.Stack(
            gap="md",
            children=[
                dmc.Group(
                    justify="space-between",
                    children=[dmc.Title("Workflow", order=3)],
                ),
                dmc.SimpleGrid(
                    cols={"base": 1, "sm": 2},
                    children=[
                        dmc.Alert(
                            color="teal",
                            radius="lg",
                            title="Choose cases",
                            children="Pick model and hardware YAML files above to define the launch cases.",
                        ),
                        dmc.Alert(
                            color="violet",
                            radius="lg",
                            title="Read results",
                            children="Open Details from the run log to compare timings, memory fit, FLOPS, and selected parallelism.",
                        ),
                    ],
                ),
            ],
        ),
    )
    return dmc.Stack(
        className="right-rail",
        gap="lg",
        children=[
            preview_card,
            status_card,
            context_card,
        ],
    )


def run_setup_card(state: Dict[str, Any]) -> dmc.Paper:
    return dmc.Paper(
        radius="xl",
        p="lg",
        withBorder=True,
        children=dmc.Stack(
            gap="md",
            children=[
                dmc.Group(justify="space-between", children=[dmc.Title("Launch Setup", order=3), dmc.Badge("workspace/configs", color="teal", variant="light")]),
                with_tip(dmc.MultiSelect(id="model-run-configs", label="Models to run", value=state["model_run_configs"], data=preset_options("models"), clearable=False), HELP_TEXT["models_to_run"]),
                with_tip(dmc.MultiSelect(id="hardware-run-configs", label="Hardware to run", value=state["hardware_run_configs"], data=preset_options("hardware"), clearable=False), HELP_TEXT["hardware_to_run"]),
                html.Div(
                    style={"display": "none"},
                    children=[
                        dmc.Select(id="model-preset", value=state["model_preset"], data=preset_options("models")),
                        dmc.Select(id="hardware-preset", value=state["hardware_preset"], data=preset_options("hardware")),
                    ],
                ),
                with_tip(dmc.SegmentedControl(id="run-mode", fullWidth=True, value=state["run_mode"], data=[{"label": "Sweep", "value": "sweep"}, {"label": "Single Launch", "value": "single"}]), "Choose whether to launch one edited config or expand sweep dimensions into multiple cases."),
                dmc.SimpleGrid(
                    cols={"base": 1, "sm": 2},
                    spacing="sm",
                    children=[
                        with_tip(dmc.NumberInput(id="worker-count", label="Workers", min=1, max=CPU_CORES, value=state["worker_count"]), HELP_TEXT["worker_count"]),
                        with_tip(dmc.NumberInput(id="timeout-seconds", label="Timeout / candidate (s)", min=0, value=state["timeout_seconds"]), HELP_TEXT["timeout"]),
                    ],
                ),
                dmc.Text(f"CPU cores detected: {CPU_CORES}. Worker default: {default_worker_count()}.", size="xs", c="dimmed"),
                dmc.Text(id="config-sync-status", size="xs", c="dimmed"),
            ],
        ),
    )

def parallelism_axis_input(component_id: str, label: str, value: int, help_text: str) -> html.Div:
    return html.Div(
        id=f"{component_id}-wrap",
        className="parallelism-axis-field",
        children=with_tip(dmc.NumberInput(id=component_id, label=label, min=1, value=value), help_text),
    )


def option_section(title: str, children: Any, subtitle: str | None = None) -> html.Div:
    header_children: List[Any] = [dmc.Title(title, order=4)]
    if subtitle:
        header_children.append(dmc.Text(subtitle, size="sm", c="dimmed"))
    return html.Div(
        className="option-section",
        children=dmc.Stack(
            gap="sm",
            children=[
                dmc.Stack(gap=2, children=header_children),
                children,
            ],
        ),
    )


def yaml_mirror_section(textarea_id: str, label: str, value: str) -> html.Details:
    return html.Details(
        className="option-section yaml-mirror-section",
        open=False,
        children=[
            html.Summary(
                className="yaml-mirror-summary",
                children=[
                    dmc.Title("YAML mirror", order=4),
                    dmc.Text("Full-file edits belong in the YAML file; option fields above keep this mirror in sync.", size="sm", c="dimmed"),
                ],
            ),
            dmc.Textarea(id=textarea_id, label=label, autosize=True, minRows=14, value=value, readOnly=True),
        ],
    )


def config_file_actions() -> html.Div:
    return html.Div(
        className="config-file-actions",
        children=dmc.Stack(
            gap="sm",
            children=[
                dmc.Group(
                    justify="space-between",
                    align="center",
                    children=[
                        dmc.Text("Active file actions", size="sm", fw=800),
                        dmc.Text(id="config-action-target", size="xs", c="dimmed"),
                    ],
                ),
                dmc.SimpleGrid(
                    cols={"base": 1, "sm": 2},
                    spacing="sm",
                    children=[
                        with_tip(dmc.TextInput(id="config-file-name", label="Config name", placeholder="my_experiment_config"), HELP_TEXT["config_file_name"]),
                        dmc.Group(
                            align="end",
                            gap="xs",
                            children=[
                                with_tip(dmc.Button("New copy", id="create-config-button", variant="light", leftSection=DashIconify(icon="solar:add-circle-bold")), HELP_TEXT["new_config"]),
                                with_tip(dmc.Button("Rename", id="rename-config-button", variant="light", leftSection=DashIconify(icon="solar:pen-bold")), HELP_TEXT["rename_config"]),
                            ],
                        ),
                    ],
                ),
                dmc.Text(id="config-action-status", size="xs", c="dimmed"),
                dmc.Divider(label="Import model from Hugging Face", labelPosition="center"),
                dmc.SimpleGrid(
                    cols={"base": 1, "sm": 2},
                    spacing="sm",
                    children=[
                        with_tip(dmc.TextInput(id="hf-model-url", label="Hugging Face URL or model ID", value=HF_SAMPLE_MODEL_URL), HELP_TEXT["hf_model_url"]),
                        with_tip(dmc.TextInput(id="hf-config-name", label="Save as", placeholder="qwen2_5_7b_from_hf"), HELP_TEXT["hf_config_name"]),
                    ],
                ),
                dmc.Group(
                    justify="space-between",
                    align="center",
                    children=[
                        with_tip(dmc.Button("Create model config", id="import-hf-config-button", variant="light", leftSection=DashIconify(icon="solar:download-bold", width=18)), HELP_TEXT["hf_import"]),
                        dmc.Text("Supported importer families: GPT/OPT/MPT, Llama/Qwen2/Phi3 without active sliding-window attention, DeepSeek-V3, and GLM-4 MoE. Use model options after import for workload fields.", size="xs", c="dimmed", className="hf-import-note"),
                    ],
                ),
                html.Div(id="hf-import-status"),
            ],
        ),
    )


def model_options_pane(state: Dict[str, Any]) -> html.Div:
    defaults = state["defaults"]
    return html.Div(
        id="model-options-pane",
        children=dmc.Stack(
            gap="md",
            children=[
                option_section(
                    "Basic Options",
                    dmc.Stack(
                        gap="sm",
                        children=[
                            with_tip(dmc.Select(id="simple-run-type", label="Run type", value=defaults["run_type"], data=[{"value": "training", "label": "Training"}, {"value": "inference", "label": "Inference"}]), HELP_TEXT["run_type"]),
                            dmc.SimpleGrid(
                                cols={"base": 1, "sm": 2},
                                spacing="sm",
                                children=[
                                    with_tip(dmc.NumberInput(id="simple-seq-len", label="Sequence length", min=1, value=defaults["simple"]["seq_len"]), HELP_TEXT["seq_len"]),
                                    with_tip(dmc.NumberInput(id="simple-decode-len", label="Decode length", min=0, value=defaults["simple"]["decode_len"]), HELP_TEXT["decode_len"]),
                                    with_tip(dmc.NumberInput(id="simple-batch-size", label="Batch size", min=1, value=defaults["simple"]["batch_size"]), HELP_TEXT["batch_size"]),
                                    with_tip(dmc.NumberInput(id="simple-grad-accum", label="Grad accumulation", min=1, value=defaults["simple"]["grad_accum"]), HELP_TEXT["grad_accum"]),
                                ],
                            ),
                        ],
                    ),
                    "Workload fields saved into the active model YAML.",
                ),
                option_section(
                    "Advanced Options",
                    dmc.Stack(
                        gap="sm",
                        children=[
                            dmc.SimpleGrid(
                                cols={"base": 1, "sm": 2},
                                spacing="sm",
                                children=[
                                    with_tip(dmc.Select(id="adv-model-type", label="Model type", value=defaults["advanced"]["model_type"], data=MODEL_ARCH_TYPE_OPTIONS), HELP_TEXT["model_type"]),
                                    with_tip(dmc.Select(id="adv-model-mode", label="Execution family", value=defaults["advanced"]["model_mode"], data=MODEL_MODE_OPTIONS), HELP_TEXT["model_mode"]),
                                    with_tip(dmc.Switch(id="adv-tied-embeddings", checked=defaults["advanced"]["tied_embeddings"], label="Tied embeddings"), HELP_TEXT["tied_embeddings"]),
                                ],
                            ),
                            dmc.Group(gap="xs", children=[dmc.Text("Model type guide", size="xs", c="dimmed"), *[model_type_badge(item["value"]) for item in MODEL_ARCH_TYPE_OPTIONS]]),
                            dmc.Group(gap="xs", children=[dmc.Text("Execution family guide", size="xs", c="dimmed"), mode_badge("LLM"), mode_badge("VIT")]),
                            dmc.SimpleGrid(
                                cols={"base": 1, "sm": 2},
                                spacing="sm",
                                children=[
                                    with_tip(dmc.NumberInput(id="adv-hidden-dim", label="Hidden dim", min=1, value=defaults["advanced"]["hidden_dim"]), HELP_TEXT["hidden_dim"]),
                                    with_tip(dmc.NumberInput(id="adv-intermediate-size", label="Intermediate size", min=1, value=defaults["advanced"]["intermediate_size"]), HELP_TEXT["intermediate_size"]),
                                    with_tip(dmc.NumberInput(id="adv-num-layers", label="Layers", min=1, value=defaults["advanced"]["num_layers"]), HELP_TEXT["num_layers"]),
                                    with_tip(dmc.NumberInput(id="adv-vocab-size", label="Vocab size", min=1, value=defaults["advanced"]["vocab_size"]), HELP_TEXT["vocab_size"]),
                                    with_tip(dmc.Select(id="adv-attention-type", label="Attention type", value=defaults["advanced"]["attention_type"], data=[{"value": "mha", "label": "MHA"}, {"value": "gqa", "label": "GQA"}, {"value": "mla", "label": "MLA"}]), HELP_TEXT["attention_type"]),
                                    with_tip(dmc.NumberInput(id="adv-num-heads", label="Attention heads", min=1, value=defaults["advanced"]["num_heads"]), HELP_TEXT["num_heads"]),
                                    with_tip(dmc.Switch(id="adv-use-flash", checked=defaults["advanced"]["use_flashattention"], label="Use FlashAttention"), HELP_TEXT["use_flash"]),
                                    with_tip(dmc.NumberInput(id="adv-attn-tile", label="Attention tile", min=1, value=defaults["advanced"]["attention_tile_size"]), HELP_TEXT["attention_tile"]),
                                    with_tip(dmc.NumberInput(id="adv-num-experts", label="Experts", min=1, value=defaults["advanced"]["num_experts"]), HELP_TEXT["num_experts"]),
                                    with_tip(dmc.NumberInput(id="adv-top-k", label="MoE top-k", min=1, value=defaults["advanced"]["top_k"]), HELP_TEXT["top_k"]),
                                    with_tip(dmc.NumberInput(id="adv-moe-intermediate-size", label="MoE intermediate size", min=1, value=defaults["advanced"]["moe_intermediate_size"]), HELP_TEXT["moe_intermediate_size"]),
                                    with_tip(dmc.NumberInput(id="adv-imbalance", label="Expert imbalance", min=0.1, step=0.1, value=defaults["advanced"]["expert_imbalance_factor"]), HELP_TEXT["expert_imbalance"]),
                                ],
                            ),
                        ],
                    ),
                    "Architecture fields saved into the active model YAML.",
                ),
                yaml_mirror_section("model-yaml", "Model YAML", defaults["model_yaml"]),
            ],
        ),
    )


def hardware_options_pane(state: Dict[str, Any]) -> html.Div:
    defaults = state["defaults"]
    return html.Div(
        id="hardware-options-pane",
        children=dmc.Stack(
            gap="md",
            children=[
                option_section(
                    "Basic Options",
                    dmc.Stack(
                        gap="sm",
                        children=[
                            dmc.Divider(label="Parallelism", labelPosition="center"),
                            dmc.SimpleGrid(
                                cols={"base": 1, "sm": 2},
                                spacing="sm",
                                children=[
                                    dmc.Stack(
                                        gap=4,
                                        children=[
                                            with_tip(dmc.Switch(id="optimize-switch", size="md", checked=bool(state["optimize_parallelism"]), label="Optimize parallelism"), HELP_TEXT["optimize_parallelism"]),
                                            dmc.Text("WARNING: This may increase runtime dramatically.", c="red", size="xs", fw=800),
                                        ],
                                    ),
                                    with_tip(dmc.Select(id="optimizer-preset", label="Parallelism search", value=state["optimizer_preset"], data=[{"value": "Fast", "label": "Fast candidate set"}, {"value": "Exhaustive", "label": "Full candidate set"}]), HELP_TEXT["optimizer_preset"]),
                                ],
                            ),
                            dmc.SimpleGrid(
                                className="parallelism-grid",
                                cols={"base": 1, "sm": 2},
                                spacing="sm",
                                children=[
                                    html.Div(id="simple-total-gpus-wrap", className="parallelism-axis-field", children=with_tip(dmc.NumberInput(id="simple-total-gpus", label="Total GPUs", min=1, value=defaults["simple"]["total_gpus"]), HELP_TEXT["total_gpus"])),
                                    parallelism_axis_input("simple-tp", "TP", defaults["simple"]["tp"], HELP_TEXT["tp"]),
                                    parallelism_axis_input("simple-cp", "CP", defaults["simple"]["cp"], HELP_TEXT["cp"]),
                                    parallelism_axis_input("simple-pp", "PP", defaults["simple"]["pp"], HELP_TEXT["pp"]),
                                    parallelism_axis_input("simple-dp", "DP", defaults["simple"]["dp"], HELP_TEXT["dp"]),
                                    parallelism_axis_input("simple-ep", "EP", defaults["simple"]["ep"], HELP_TEXT["ep"]),
                                    parallelism_axis_input("simple-replica-count", "Replica count", defaults["simple"]["replica_count"], HELP_TEXT["replica_count"]),
                                ],
                            ),
                            dmc.Divider(label="Derates", labelPosition="center"),
                            dmc.Group(
                                justify="flex-end",
                                children=[
                                    with_tip(
                                        dmc.Button(
                                            "Reset to paper defaults",
                                            id="reset-paper-derates-button",
                                            variant="light",
                                            size="xs",
                                            leftSection=DashIconify(icon="solar:restart-bold", width=16),
                                        ),
                                        HELP_TEXT["reset_paper_derates"],
                                    )
                                ],
                            ),
                            dmc.SimpleGrid(
                                cols={"base": 1, "sm": 3},
                                spacing="sm",
                                children=[
                                    with_tip(dmc.NumberInput(id="simple-compute-derate", label="Compute derate", min=0.0, max=1.0, step=0.01, decimalScale=3, value=defaults["simple"]["compute_derate"]), HELP_TEXT["compute_derate"]),
                                    with_tip(dmc.NumberInput(id="simple-memory-derate", label="Memory derate", min=0.0, max=1.0, step=0.01, decimalScale=3, value=defaults["simple"]["memory_derate"]), HELP_TEXT["memory_derate"]),
                                    with_tip(dmc.NumberInput(id="simple-network-derate", label="Network derate", min=0.0, max=1.0, step=0.01, decimalScale=3, value=defaults["simple"]["network_derate"]), HELP_TEXT["network_derate"]),
                                ],
                            ),
                            dmc.Divider(label="Hardware limits", labelPosition="center"),
                            dmc.SimpleGrid(
                                cols={"base": 1, "sm": 3},
                                spacing="sm",
                                children=[
                                    with_tip(dmc.NumberInput(id="simple-hbm-gb", label="HBM capacity (GB)", min=1, value=defaults["simple"]["hbm_gb"]), HELP_TEXT["hbm_gb"]),
                                    with_tip(dmc.NumberInput(id="simple-gpu-clock", label="GPU clock (GHz)", min=0.1, step=0.01, decimalScale=3, value=defaults["simple"]["gpu_clock_ghz"]), HELP_TEXT["gpu_clock"]),
                                    with_tip(dmc.NumberInput(id="simple-memory-bw", label="Memory BW (GB/s)", min=1, step=1, decimalScale=2, value=defaults["simple"]["memory_bw_gbs"]), HELP_TEXT["memory_bw"]),
                                    with_tip(dmc.Switch(id="simple-use-astrasim", checked=defaults["simple"]["use_astrasim"], label="Use AstraSim"), HELP_TEXT["use_astrasim"]),
                                ],
                            ),
                            dmc.Divider(label="Network dimensions", labelPosition="center"),
                            dmc.Alert(
                                "Hierarchical AstraSim uses Dimension 0 for TP/CP/EP. DP is always written to the outer network dimension.",
                                color="blue",
                                radius="md",
                                className="topology-mapping-note",
                            ),
                            dmc.SimpleGrid(
                                cols={"base": 1, "sm": 2},
                                spacing="sm",
                                children=[
                                    with_tip(dmc.Select(id="parallelism-topology-mode", label="PP placement", value=defaults["advanced"]["pp_network_dimension"], data=PP_TOPOLOGY_OPTIONS, clearable=False), HELP_TEXT["pp_topology_dimension"]),
                                    with_tip(dmc.Text("Choose the Dimension 1 | Dimension 2 mapping; then edit each active dimension's topology below.", size="sm", c="dimmed", className="topology-mapping-copy"), HELP_TEXT["parallelism_topology"]),
                                ],
                            ),
                            html.Div(id="parallelism-topology-preview", children=parallelism_topology_preview(defaults["advanced"]["pp_network_dimension"])),
                            html.Div(id="network-dimensions-editor", children=network_editor(defaults["network_dimensions"], defaults["advanced"]["pp_network_dimension"])),
                        ],
                    ),
                    "Hardware scaling, bandwidth, derate, and backend fields saved into the active hardware YAML.",
                ),
                option_section(
                    "Advanced Options",
                    dmc.SimpleGrid(
                        cols={"base": 1, "sm": 2},
                        spacing="sm",
                        children=[
                            with_tip(dmc.Switch(id="adv-full-recomp", checked=defaults["advanced"]["full_recomputation"], label="Full recomputation"), HELP_TEXT["full_recomp"]),
                            with_tip(dmc.Select(id="adv-dp-zero", label="ZeRO stage", value=str(defaults["advanced"]["dp_zero_stage"]), data=ZERO_STAGE_OPTIONS, clearable=False), HELP_TEXT["zero_stage"]),
                            with_tip(dmc.Select(id="adv-tensor-format", label="Tensor format", value=defaults["advanced"]["tensor_format"], data=TENSOR_FORMAT_OPTIONS), HELP_TEXT["tensor_format"]),
                            with_tip(dmc.Select(id="adv-precision-kv-cache", label="KV cache precision", value=defaults["advanced"]["precision_kv_cache"], data=PRECISION_FORMAT_OPTIONS), HELP_TEXT["precision_kv_cache"]),
                            with_tip(dmc.Select(id="adv-precision-parameters", label="Parameter precision", value=defaults["advanced"]["precision_parameters"], data=PRECISION_FORMAT_OPTIONS), HELP_TEXT["precision_parameters"]),
                            with_tip(dmc.Select(id="adv-precision-gradients", label="Gradient precision", value=defaults["advanced"]["precision_gradients"], data=PRECISION_FORMAT_OPTIONS), HELP_TEXT["precision_gradients"]),
                            with_tip(dmc.Select(id="adv-precision-grad-communication", label="Gradient comm precision", value=defaults["advanced"]["precision_grad_communication"], data=PRECISION_FORMAT_OPTIONS), HELP_TEXT["precision_grad_communication"]),
                            with_tip(dmc.Select(id="adv-precision-optimizer-states", label="Optimizer state precision", value=defaults["advanced"]["precision_optimizer_states"], data=PRECISION_FORMAT_OPTIONS), HELP_TEXT["precision_optimizer_states"]),
                            with_tip(dmc.Select(id="adv-precision-stats", label="Stats precision", value=defaults["advanced"]["precision_stats"], data=PRECISION_FORMAT_OPTIONS), HELP_TEXT["precision_stats"]),
                            with_tip(dmc.Select(id="adv-precision-master-parameters", label="Master parameter copy", value=str(defaults["advanced"]["precision_master_parameters"]), data=MASTER_PRECISION_OPTIONS), HELP_TEXT["precision_master_parameters"]),
                        ],
                    ),
                    "Software and precision fields saved into the active hardware YAML.",
                ),
                yaml_mirror_section("hardware-yaml", "Hardware YAML", defaults["hardware_yaml"]),
            ],
        ),
    )


def config_options_card(state: Dict[str, Any]) -> dmc.Paper:
    return dmc.Paper(
        radius="xl",
        p="lg",
        withBorder=True,
        className="config-options-card",
        children=dmc.Stack(
            gap="md",
            children=[
                dmc.Group(
                    justify="space-between",
                    align="center",
                    children=[
                        with_tip(dmc.Title("Config Options", order=3), HELP_TEXT["editor_tabs"]),
                    ],
                ),
                with_tip(
                    dmc.Text("Choose a YAML file, then edit the options below.", size="sm", c="dimmed"),
                    HELP_TEXT["editor_tabs"],
                ),
                with_tip(
                    dmc.Tabs(
                        id="config-editor-tabs",
                        value=state["active_config_tab"],
                        className="config-workbook-tabs",
                        children=config_workbook_tabs_children(state["model_run_configs"], state["hardware_run_configs"], MODEL_LABELS, HW_LABELS),
                    ),
                    HELP_TEXT["editor_tabs"],
                ),
                config_file_actions(),
                model_options_pane(state),
                hardware_options_pane(state),
            ],
        ),
    )


def dimensions_card(state: Dict[str, Any]) -> dmc.Paper:
    rows = state.get("sweep_rows") or default_sweep_rows()
    return dmc.Paper(
        radius="xl",
        p="lg",
        withBorder=True,
        children=dmc.Stack(
            gap="md",
            children=[
                dmc.Group(
                    justify="space-between",
                    align="center",
                    children=[
                        dmc.Title("Sweep Dimensions", order=3),
                        with_tip(dmc.Button("Reset selections", id="reset-last-state-button", variant="light", size="xs", leftSection=DashIconify(icon="solar:restart-bold", width=16)), HELP_TEXT["reset_last_state"]),
                    ],
                ),
                with_tip(
                    dmc.Text("Sweep workload size and hardware scaling. Select multiple model or hardware files in Launch Setup for config comparisons.", size="sm", c="dimmed"),
                    HELP_TEXT["sweep_dimensions"],
                ),
                dmc.Text(id="last-state-status", size="xs", c="dimmed"),
                dim_card(1, rows[0]),
                dim_card(2, rows[1]),
                dim_card(3, rows[2]),
                dmc.SimpleGrid(
                    cols={"base": 1, "sm": 2},
                    spacing="sm",
                    children=[
                        with_tip(dmc.Select(id="x-axis-select", label="X-axis", value=state.get("x_axis"), data=[]), HELP_TEXT["x_axis"]),
                        with_tip(dmc.Select(id="series-select", label="Color grouping (optional)", value=state.get("series_axis"), data=[], clearable=True), HELP_TEXT["series_axis"]),
                    ],
                ),
            ],
        ),
    )


def launch_controls_card(metric_options: List[Dict[str, str]], state: Dict[str, Any]) -> dmc.Paper:
    return dmc.Paper(
        radius="xl",
        p="lg",
        withBorder=True,
        children=dmc.Stack(
            gap="md",
            children=[
                dmc.Title("Launch", order=3),
                with_tip(dmc.Select(id="metric-select", label="Metric", value=state["metric"], data=metric_options), HELP_TEXT["metric"]),
                dmc.Group(
                    justify="flex-end",
                    children=[
                        with_tip(dmc.Button("Launch 1 run", id="run-button", leftSection=DashIconify(icon="solar:rocket-bold")), HELP_TEXT["run_launch"]),
                        with_tip(dmc.Button("Cancel Active Job", id="cancel-button", color="red", variant="light"), HELP_TEXT["cancel_job"]),
                    ],
                ),
            ],
        ),
    )


app.layout = create_layout


def collect_payload(model_preset_id: str, hardware_preset_id: str, run_mode: str, optimize_parallelism: bool, optimizer_preset: str, simple_values: Dict[str, Any], advanced_values: Dict[str, Any], network_rows: List[Dict[str, Any]], dimensions: List[Dict[str, Any]], metric: str, x_axis: str | None, series_axis: str | None, worker_count: int, timeout_seconds: int) -> Dict[str, Any]:
    return {"model_preset_id": model_preset_id, "hardware_preset_id": hardware_preset_id, "run_mode": run_mode, "optimize_parallelism": optimize_parallelism, "optimizer_preset": optimizer_preset, "use_raw_yaml": False, "model_yaml_text": "", "hardware_yaml_text": "", "simple": simple_values, "advanced": advanced_values, "network_dimensions": network_rows, "dimensions": dimensions, "metric": metric, "x_axis": x_axis, "series_axis": series_axis, "worker_count": worker_count, "timeout_seconds": timeout_seconds}


def collect_form_payload(
    model_preset: str,
    hardware_preset: str,
    run_mode: str,
    optimize_parallelism: bool,
    optimizer_preset: str,
    simple_run_type: str,
    simple_seq_len: int,
    simple_decode_len: int,
    simple_batch_size: int,
    simple_grad_accum: int,
    simple_total_gpus: int,
    simple_tp: int,
    simple_cp: int,
    simple_pp: int,
    simple_dp: int,
    simple_ep: int,
    simple_replica_count: int,
    simple_hbm_gb: float,
    simple_compute_derate: float,
    simple_memory_derate: float,
    simple_network_derate: float,
    simple_gpu_clock: float,
    simple_memory_bw: float,
    simple_use_astrasim: bool,
    adv_model_type: str,
    adv_model_mode: str,
    adv_full_recomp: bool,
    adv_dp_zero: int | str,
    adv_tensor_format: str,
    adv_precision_kv_cache: str,
    adv_precision_parameters: str,
    adv_precision_gradients: str,
    adv_precision_grad_communication: str,
    adv_precision_optimizer_states: str,
    adv_precision_stats: str,
    adv_precision_master_parameters: str,
    adv_tied_embeddings: bool,
    adv_hidden_dim: int,
    adv_intermediate_size: int,
    adv_num_layers: int,
    adv_vocab_size: int,
    adv_attention_type: str,
    adv_num_heads: int,
    adv_use_flash: bool,
    adv_attn_tile: int,
    adv_num_experts: int,
    adv_top_k: int,
    adv_moe_intermediate_size: int,
    adv_imbalance: float,
    net_topologies: List[str],
    net_bandwidths: List[str],
    net_latencies: List[float],
    net_utils: List[float],
    parallelism_topology_mode: str,
    dimensions: List[Dict[str, Any]],
    metric: str,
    x_axis: str | None,
    series_axis: str | None,
    worker_count: int,
    timeout_seconds: int,
) -> Dict[str, Any]:
    return collect_payload(
        model_preset,
        hardware_preset,
        run_mode,
        optimize_parallelism,
        optimizer_preset,
        {"run_type": simple_run_type, "seq_len": simple_seq_len, "decode_len": simple_decode_len, "batch_size": simple_batch_size, "grad_accum": 1 if simple_run_type == "inference" else simple_grad_accum, "total_gpus": simple_total_gpus, "tp": simple_tp, "cp": simple_cp, "pp": simple_pp, "dp": simple_dp, "ep": simple_ep, "replica_count": simple_replica_count if simple_run_type == "inference" else 1, "hbm_gb": simple_hbm_gb, "compute_derate": simple_compute_derate, "memory_derate": simple_memory_derate, "network_derate": simple_network_derate, "gpu_clock_ghz": simple_gpu_clock, "memory_bw_gbs": simple_memory_bw, "use_astrasim": bool(simple_use_astrasim)},
        {"model_type": adv_model_type, "model_mode": adv_model_mode, "full_recomputation": adv_full_recomp, "dp_zero_stage": int(adv_dp_zero or 0), "tensor_format": adv_tensor_format, "precision_kv_cache": adv_precision_kv_cache, "precision_parameters": adv_precision_parameters, "precision_gradients": adv_precision_gradients, "precision_grad_communication": adv_precision_grad_communication, "precision_optimizer_states": adv_precision_optimizer_states, "precision_stats": adv_precision_stats, "precision_master_parameters": adv_precision_master_parameters, "pp_network_dimension": parallelism_topology_mode, "tied_embeddings": adv_tied_embeddings, "hidden_dim": adv_hidden_dim, "intermediate_size": adv_intermediate_size, "num_layers": adv_num_layers, "vocab_size": adv_vocab_size, "attention_type": adv_attention_type, "num_heads": adv_num_heads, "use_flashattention": adv_use_flash, "attention_tile_size": adv_attn_tile, "num_experts": adv_num_experts, "top_k": adv_top_k, "moe_intermediate_size": adv_moe_intermediate_size, "expert_imbalance_factor": adv_imbalance},
        _network_rows_from_callback(net_topologies, net_bandwidths, net_latencies, net_utils),
        dimensions,
        metric,
        x_axis,
        series_axis,
        worker_count,
        timeout_seconds,
    )


@callback(
    Output("config-editor-tabs", "children"),
    Output("config-editor-tabs", "value"),
    Input("model-run-configs", "value"),
    Input("hardware-run-configs", "value"),
    State("config-editor-tabs", "value"),
    State("model-preset", "value"),
    State("hardware-preset", "value"),
)
def refresh_editor_tab_sets(model_ids: List[str] | None, hardware_ids: List[str] | None, current_tab: str | None, current_model: str | None, current_hardware: str | None):
    active_tab = active_config_tab_for_selection_change(
        model_ids,
        hardware_ids,
        DEFAULT_MODEL_ID,
        DEFAULT_HW_ID,
        current_tab,
        current_model,
        current_hardware,
        dash.ctx.triggered_id,
    )
    return (
        config_workbook_tabs_children(model_ids, hardware_ids, preset_labels("models"), preset_labels("hardware")),
        active_tab,
    )


@callback(
    Output("model-preset", "value"),
    Output("hardware-preset", "value"),
    Input("config-editor-tabs", "value"),
    State("model-preset", "value"),
    State("hardware-preset", "value"),
)
def sync_primary_config_selection(active_tab: str | None, current_model: str | None, current_hardware: str | None):
    kind, config_id = parse_config_tab(active_tab)
    if kind == "models":
        return config_id or DEFAULT_MODEL_ID, current_hardware or DEFAULT_HW_ID
    if kind == "hardware":
        return current_model or DEFAULT_MODEL_ID, config_id or DEFAULT_HW_ID
    return current_model or DEFAULT_MODEL_ID, current_hardware or DEFAULT_HW_ID


@callback(
    Output("model-options-pane", "style"),
    Output("hardware-options-pane", "style"),
    Output("config-action-target", "children"),
    Input("config-editor-tabs", "value"),
)
def show_active_config_options(active_tab: str | None):
    kind, config_id = parse_config_tab(active_tab)
    if kind == "hardware":
        return {"display": "none"}, {}, f"Editing hardware: {Path(config_id or '').stem}"
    return {}, {"display": "none"}, f"Editing model: {Path(config_id or '').stem}"


def replace_selected_config(values: List[str] | None, old_id: str, new_id: str) -> List[str]:
    selected = list(values or [])
    if old_id in selected:
        return [new_id if item == old_id else item for item in selected]
    return selected + [new_id]


@callback(
    Output("config-action-status", "children"),
    Output("model-run-configs", "data"),
    Output("hardware-run-configs", "data"),
    Output("model-preset", "data"),
    Output("hardware-preset", "data"),
    Output("model-run-configs", "value"),
    Output("hardware-run-configs", "value"),
    Output("config-editor-tabs", "value", allow_duplicate=True),
    Input("create-config-button", "n_clicks"),
    Input("rename-config-button", "n_clicks"),
    State("config-file-name", "value"),
    State("config-editor-tabs", "value"),
    State("model-preset", "value"),
    State("hardware-preset", "value"),
    State("model-run-configs", "value"),
    State("hardware-run-configs", "value"),
    prevent_initial_call=True,
)
def handle_config_file_action(
    create_clicks: int | None,
    rename_clicks: int | None,
    new_name: str | None,
    active_tab: str | None,
    model_preset: str,
    hardware_preset: str,
    model_values: List[str] | None,
    hardware_values: List[str] | None,
):
    del create_clicks, rename_clicks
    kind, active_id = parse_config_tab(active_tab)
    if kind not in {"models", "hardware"}:
        return "Choose a model or hardware tab before using file actions.", preset_options("models"), preset_options("hardware"), preset_options("models"), preset_options("hardware"), model_values, hardware_values, no_update
    source_id = active_id or (model_preset if kind == "models" else hardware_preset)
    try:
        if dash.ctx.triggered_id == "create-config-button":
            new_id = create_config_copy(kind, source_id, new_name or "")
            message = f"Created {new_id}."
            if kind == "models":
                model_values = replace_selected_config(model_values, source_id, new_id)
            else:
                hardware_values = replace_selected_config(hardware_values, source_id, new_id)
        elif dash.ctx.triggered_id == "rename-config-button":
            new_id = rename_config_file(kind, source_id, new_name or "")
            message = f"Renamed {source_id} to {new_id}."
            if kind == "models":
                model_values = replace_selected_config(model_values, source_id, new_id)
            else:
                hardware_values = replace_selected_config(hardware_values, source_id, new_id)
        else:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
    except Exception as exc:  # noqa: BLE001
        return f"Config action failed: {exc}", preset_options("models"), preset_options("hardware"), preset_options("models"), preset_options("hardware"), model_values, hardware_values, no_update
    return message, preset_options("models"), preset_options("hardware"), preset_options("models"), preset_options("hardware"), model_values, hardware_values, config_tab_value(kind, new_id)


@callback(
    Output("hf-import-status", "children"),
    Output("model-run-configs", "data", allow_duplicate=True),
    Output("model-preset", "data", allow_duplicate=True),
    Output("model-run-configs", "value", allow_duplicate=True),
    Output("model-preset", "value", allow_duplicate=True),
    Output("config-editor-tabs", "value", allow_duplicate=True),
    Input("import-hf-config-button", "n_clicks"),
    State("hf-model-url", "value"),
    State("hf-config-name", "value"),
    State("model-run-configs", "value"),
    prevent_initial_call=True,
)
def import_huggingface_model_config(n_clicks: int | None, hf_reference: str | None, config_name: str | None, model_values: List[str] | None):
    if not n_clicks:
        return no_update, no_update, no_update, no_update, no_update, no_update
    try:
        created = create_model_config_from_huggingface(hf_reference or "", config_name or "")
    except Exception as exc:  # noqa: BLE001
        return hf_import_status_alert(str(exc), ok=False), preset_options("models"), preset_options("models"), model_values, no_update, no_update
    model_ids = list(model_values or [])
    if created["id"] not in model_ids:
        model_ids.append(created["id"])
    alias_note = f" Mapped source family through {created['alias']}." if created.get("alias") else ""
    message = (
        f"Created {created['id']} from {created['model_id']}@{created['revision']} with model_type={created['model_type']}."
        f"{alias_note} Adjust not automatically determinable workload fields in Basic Options and Advanced Options before launch."
    )
    model_options = preset_options("models")
    return hf_import_status_alert(message, ok=True), model_options, model_options, model_ids, created["id"], config_tab_value("models", created["id"])


@callback(
    Output("model-yaml", "value"),
    Output("hardware-yaml", "value"),
    Output("simple-run-type", "value"),
    Output("simple-seq-len", "value"),
    Output("simple-decode-len", "value"),
    Output("simple-batch-size", "value"),
    Output("simple-grad-accum", "value"),
    Output("simple-total-gpus", "value"),
    Output("simple-tp", "value"),
    Output("simple-cp", "value"),
    Output("simple-pp", "value"),
    Output("simple-dp", "value"),
    Output("simple-ep", "value"),
    Output("simple-replica-count", "value"),
    Output("simple-hbm-gb", "value"),
    Output("simple-compute-derate", "value"),
    Output("simple-memory-derate", "value"),
    Output("simple-network-derate", "value"),
    Output("simple-gpu-clock", "value"),
    Output("simple-memory-bw", "value"),
    Output("simple-use-astrasim", "checked"),
    Output("adv-model-type", "value"),
    Output("adv-model-mode", "value"),
    Output("adv-full-recomp", "checked"),
    Output("adv-dp-zero", "value"),
    Output("adv-tensor-format", "value"),
    Output("adv-precision-kv-cache", "value"),
    Output("adv-precision-parameters", "value"),
    Output("adv-precision-gradients", "value"),
    Output("adv-precision-grad-communication", "value"),
    Output("adv-precision-optimizer-states", "value"),
    Output("adv-precision-stats", "value"),
    Output("adv-precision-master-parameters", "value"),
    Output("adv-tied-embeddings", "checked"),
    Output("adv-hidden-dim", "value"),
    Output("adv-intermediate-size", "value"),
    Output("adv-num-layers", "value"),
    Output("adv-vocab-size", "value"),
    Output("adv-attention-type", "value"),
    Output("adv-num-heads", "value"),
    Output("adv-use-flash", "checked"),
    Output("adv-attn-tile", "value"),
    Output("adv-num-experts", "value"),
    Output("adv-top-k", "value"),
    Output("adv-moe-intermediate-size", "value"),
    Output("adv-imbalance", "value"),
    Output("parallelism-topology-mode", "value"),
    Output("parallelism-topology-preview", "children"),
    Output("network-dimensions-editor", "children"),
    Output("metric-select", "data"),
    Output("metric-select", "value"),
    Input("model-preset", "value"),
    Input("hardware-preset", "value"),
    prevent_initial_call=True,
)
def refresh_defaults(model_preset_id: str, hardware_preset_id: str):
    defaults = build_form_defaults(model_preset_id, hardware_preset_id)
    metric_options = get_metric_options(defaults["run_type"])
    return (
        defaults["model_yaml"],
        defaults["hardware_yaml"],
        defaults["run_type"],
        defaults["simple"]["seq_len"],
        defaults["simple"]["decode_len"],
        defaults["simple"]["batch_size"],
        defaults["simple"]["grad_accum"],
        defaults["simple"]["total_gpus"],
        defaults["simple"]["tp"],
        defaults["simple"]["cp"],
        defaults["simple"]["pp"],
        defaults["simple"]["dp"],
        defaults["simple"]["ep"],
        defaults["simple"]["replica_count"],
        defaults["simple"]["hbm_gb"],
        defaults["simple"]["compute_derate"],
        defaults["simple"]["memory_derate"],
        defaults["simple"]["network_derate"],
        defaults["simple"]["gpu_clock_ghz"],
        defaults["simple"]["memory_bw_gbs"],
        defaults["simple"]["use_astrasim"],
        defaults["advanced"]["model_type"],
        defaults["advanced"]["model_mode"],
        defaults["advanced"]["full_recomputation"],
        str(defaults["advanced"]["dp_zero_stage"]),
        defaults["advanced"]["tensor_format"],
        defaults["advanced"]["precision_kv_cache"],
        defaults["advanced"]["precision_parameters"],
        defaults["advanced"]["precision_gradients"],
        defaults["advanced"]["precision_grad_communication"],
        defaults["advanced"]["precision_optimizer_states"],
        defaults["advanced"]["precision_stats"],
        defaults["advanced"]["precision_master_parameters"],
        defaults["advanced"]["tied_embeddings"],
        defaults["advanced"]["hidden_dim"],
        defaults["advanced"]["intermediate_size"],
        defaults["advanced"]["num_layers"],
        defaults["advanced"]["vocab_size"],
        defaults["advanced"]["attention_type"],
        defaults["advanced"]["num_heads"],
        defaults["advanced"]["use_flashattention"],
        defaults["advanced"]["attention_tile_size"],
        defaults["advanced"]["num_experts"],
        defaults["advanced"]["top_k"],
        defaults["advanced"]["moe_intermediate_size"],
        defaults["advanced"]["expert_imbalance_factor"],
        defaults["advanced"]["pp_network_dimension"],
        parallelism_topology_preview(defaults["advanced"]["pp_network_dimension"]),
        network_editor(defaults["network_dimensions"], defaults["advanced"]["pp_network_dimension"]),
        metric_options,
        get_default_metric_for_run_type(defaults["run_type"]),
    )


@callback(
    Output("simple-compute-derate", "value", allow_duplicate=True),
    Output("simple-memory-derate", "value", allow_duplicate=True),
    Output("simple-network-derate", "value", allow_duplicate=True),
    Output({"type": "net-util", "index": ALL}, "value", allow_duplicate=True),
    Input("reset-paper-derates-button", "n_clicks"),
    State("hardware-preset", "value"),
    State({"type": "net-util", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def reset_paper_derates(n_clicks: int | None, hardware_preset_id: str | None, net_util_ids: List[Dict[str, Any]]):
    if not n_clicks:
        return no_update, no_update, no_update, no_update
    defaults = paper_derate_defaults_for_hardware(hardware_preset_id)
    if defaults is None:
        return no_update, no_update, no_update, no_update
    communication = defaults["communication"]
    return defaults["compute"], defaults["memory"], communication, [communication for _ in net_util_ids]


@callback(
    Output("parallelism-topology-preview", "children", allow_duplicate=True),
    Input("parallelism-topology-mode", "value"),
    prevent_initial_call=True,
)
def refresh_parallelism_topology_preview(pp_dimension: str | None):
    return parallelism_topology_preview(pp_dimension)


@callback(
    Output({"type": "net-topology", "index": ALL}, "disabled"),
    Output({"type": "net-bandwidth", "index": ALL}, "disabled"),
    Output({"type": "net-latency", "index": ALL}, "disabled"),
    Output({"type": "net-util", "index": ALL}, "disabled"),
    Input("parallelism-topology-mode", "value"),
    State({"type": "net-topology", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def toggle_network_dimension_controls(pp_dimension: str | None, topology_ids: List[Dict[str, Any]]):
    mode = str(pp_dimension or "dim1_shared")
    if mode == "dim1":
        mode = "dim1_dim2"
    elif mode in {"dim2", "dim2_shared"}:
        mode = "dim1_dim2"
    disabled_indices = disabled_network_dimension_indices(mode)
    disabled = [item.get("index") in disabled_indices for item in topology_ids]
    return disabled, disabled, disabled, disabled


@callback(
    Output("metric-select", "data", allow_duplicate=True),
    Output("metric-select", "value", allow_duplicate=True),
    Input("simple-run-type", "value"),
    prevent_initial_call=True,
)
def refresh_metric_options_for_run_type(run_type: str):
    return get_metric_options(run_type), get_default_metric_for_run_type(run_type)


@callback(
    Output("simple-decode-len", "disabled"),
    Output("simple-replica-count", "disabled"),
    Output("simple-grad-accum", "disabled"),
    Output("simple-total-gpus", "disabled"),
    Output("simple-tp", "disabled"),
    Output("simple-cp", "disabled"),
    Output("simple-pp", "disabled"),
    Output("simple-dp", "disabled"),
    Output("simple-ep", "disabled"),
    Output("optimizer-preset", "disabled"),
    Output("simple-tp-wrap", "className"),
    Output("simple-total-gpus-wrap", "className"),
    Output("simple-cp-wrap", "className"),
    Output("simple-pp-wrap", "className"),
    Output("simple-dp-wrap", "className"),
    Output("simple-ep-wrap", "className"),
    Output("simple-replica-count-wrap", "className"),
    Input("simple-run-type", "value"),
    Input("optimize-switch", "checked"),
    Input("run-mode", "value"),
    Input("dim-1-field", "value"),
    Input("dim-2-field", "value"),
    Input("dim-3-field", "value"),
)
def toggle_inputs(run_type: str, optimize_parallelism: bool, run_mode: str, dim1_field: str | None, dim2_field: str | None, dim3_field: str | None):
    inference = run_type == "inference"
    manual_disabled = bool(optimize_parallelism)
    total_gpus_swept = run_mode == "sweep" and "hardware.total_gpus" in {dim1_field, dim2_field, dim3_field}
    axis_class = "parallelism-axis-field is-auto" if manual_disabled else "parallelism-axis-field"
    total_gpus_class = "parallelism-axis-field is-swept" if total_gpus_swept else "parallelism-axis-field"
    replica_class = "parallelism-axis-field"
    return (
        not inference,
        not inference,
        inference,
        total_gpus_swept or not optimize_parallelism,
        manual_disabled,
        manual_disabled,
        manual_disabled,
        manual_disabled or inference,
        manual_disabled,
        not optimize_parallelism,
        axis_class,
        total_gpus_class,
        axis_class,
        axis_class,
        axis_class,
        axis_class,
        replica_class,
    )


@callback(
    Output("simple-grad-accum", "value", allow_duplicate=True),
    Input("simple-run-type", "value"),
    State("simple-grad-accum", "value"),
    prevent_initial_call=True,
)
def enforce_inference_grad_accum(run_type: str, current_value: int | None):
    return 1 if run_type == "inference" else current_value


@callback(
    Output("simple-replica-count", "value", allow_duplicate=True),
    Input("simple-run-type", "value"),
    State("simple-replica-count", "value"),
    prevent_initial_call=True,
)
def enforce_training_replica_count(run_type: str, current_value: int | None):
    if run_type == "inference":
        return no_update
    return 1


@callback(Output("sweep-dimensions-section", "style"), Input("run-mode", "value"))
def toggle_sweep_dimensions_section(run_mode: str):
    return {} if run_mode == "sweep" else {"display": "none"}


@callback(
    Output("x-axis-select", "data"),
    Output("x-axis-select", "value"),
    Output("series-select", "data"),
    Output("series-select", "value"),
    Output("dim-1-configs", "data"),
    Output("dim-2-configs", "data"),
    Output("dim-3-configs", "data"),
    Input("dim-1-field", "value"),
    Input("dim-1-network-targets", "value"),
    Input("dim-1-network-apply", "value"),
    Input("dim-2-field", "value"),
    Input("dim-2-network-targets", "value"),
    Input("dim-2-network-apply", "value"),
    Input("dim-3-field", "value"),
    Input("dim-3-network-targets", "value"),
    Input("dim-3-network-apply", "value"),
    State("x-axis-select", "value"),
    State("series-select", "value"),
)
def refresh_dimension_options(
    field1: str | None,
    network_targets1: List[str] | None,
    network_apply1: str | None,
    field2: str | None,
    network_targets2: List[str] | None,
    network_apply2: str | None,
    field3: str | None,
    network_targets3: List[str] | None,
    network_apply3: str | None,
    current_x_axis: str | None,
    current_series_axis: str | None,
):
    row_fields = [
        resolve_sweep_field(field1, network_targets1, network_apply1),
        resolve_sweep_field(field2, network_targets2, network_apply2),
        resolve_sweep_field(field3, network_targets3, network_apply3),
    ]
    active_fields = [field for field in row_fields if field]
    options = [{"value": field, "label": dimension_label(field)} for field in active_fields]
    x_value = current_x_axis if current_x_axis in active_fields else (active_fields[0] if active_fields else None)
    series_default = active_fields[-1] if len(active_fields) > 1 else None
    series_value = current_series_axis if current_series_axis in active_fields else series_default
    if series_value == x_value and len(active_fields) > 1:
        series_value = active_fields[-1] if active_fields[-1] != x_value else active_fields[0]
    config_data = {"model_config": preset_options("models"), "hardware_config": preset_options("hardware")}
    rows = []
    for field in row_fields:
        rows.append(config_data.get(field, []))
    return options, x_value, options, series_value, rows[0], rows[1], rows[2]


def sweep_control_visibility(field_key: str | None, mode: str | None, network_targets: List[str] | str | None = None, network_apply: str | None = None) -> Dict[str, Any]:
    hidden = {"display": "none"}
    shown = {}
    if not field_key:
        return {"network_style": hidden, "mode_style": hidden, "values_style": hidden, "configs_style": hidden, "range_style": hidden, "mode": "values"}
    resolved_field = resolve_sweep_field(field_key, network_targets, network_apply)
    network_style = shown if field_key == NETWORK_SWEEP_GROUP_VALUE else hidden
    kind = FIELD_TYPES.get(resolved_field, {}).get("kind")
    if kind == "config":
        return {"network_style": network_style, "mode_style": hidden, "values_style": hidden, "configs_style": shown, "range_style": hidden, "mode": "values"}
    active_mode = "range" if mode == "range" else "values"
    return {
        "network_style": network_style,
        "mode_style": shown,
        "values_style": shown if active_mode == "values" else hidden,
        "configs_style": hidden,
        "range_style": shown if active_mode == "range" else hidden,
        "mode": active_mode,
    }


@callback(
    Output("dim-1-network-wrap", "style"),
    Output("dim-1-mode-wrap", "style"),
    Output("dim-1-values-wrap", "style"),
    Output("dim-1-configs-wrap", "style"),
    Output("dim-1-range-wrap", "style"),
    Output("dim-2-network-wrap", "style"),
    Output("dim-2-mode-wrap", "style"),
    Output("dim-2-values-wrap", "style"),
    Output("dim-2-configs-wrap", "style"),
    Output("dim-2-range-wrap", "style"),
    Output("dim-3-network-wrap", "style"),
    Output("dim-3-mode-wrap", "style"),
    Output("dim-3-values-wrap", "style"),
    Output("dim-3-configs-wrap", "style"),
    Output("dim-3-range-wrap", "style"),
    Input("dim-1-field", "value"),
    Input("dim-1-network-targets", "value"),
    Input("dim-1-network-apply", "value"),
    Input("dim-1-mode", "value"),
    Input("dim-2-field", "value"),
    Input("dim-2-network-targets", "value"),
    Input("dim-2-network-apply", "value"),
    Input("dim-2-mode", "value"),
    Input("dim-3-field", "value"),
    Input("dim-3-network-targets", "value"),
    Input("dim-3-network-apply", "value"),
    Input("dim-3-mode", "value"),
)
def refresh_sweep_controls(
    field1: str | None,
    network_targets1: List[str] | None,
    network_apply1: str | None,
    mode1: str | None,
    field2: str | None,
    network_targets2: List[str] | None,
    network_apply2: str | None,
    mode2: str | None,
    field3: str | None,
    network_targets3: List[str] | None,
    network_apply3: str | None,
    mode3: str | None,
):
    outputs = []
    for field, targets, apply_mode, mode in [(field1, network_targets1, network_apply1, mode1), (field2, network_targets2, network_apply2, mode2), (field3, network_targets3, network_apply3, mode3)]:
        visibility = sweep_control_visibility(field, mode, targets, apply_mode)
        outputs.extend([visibility["network_style"], visibility["mode_style"], visibility["values_style"], visibility["configs_style"], visibility["range_style"]])
    return tuple(outputs)


@callback(
    Output("dim-1-range-preview", "children"),
    Output("dim-2-range-preview", "children"),
    Output("dim-3-range-preview", "children"),
    Input("dim-1-field", "value"),
    Input("dim-1-network-targets", "value"),
    Input("dim-1-network-apply", "value"),
    Input("dim-1-mode", "value"),
    Input("dim-1-start", "value"),
    Input("dim-1-end", "value"),
    Input("dim-1-step_or_points", "value"),
    Input("dim-2-field", "value"),
    Input("dim-2-network-targets", "value"),
    Input("dim-2-network-apply", "value"),
    Input("dim-2-mode", "value"),
    Input("dim-2-start", "value"),
    Input("dim-2-end", "value"),
    Input("dim-2-step_or_points", "value"),
    Input("dim-3-field", "value"),
    Input("dim-3-network-targets", "value"),
    Input("dim-3-network-apply", "value"),
    Input("dim-3-mode", "value"),
    Input("dim-3-start", "value"),
    Input("dim-3-end", "value"),
    Input("dim-3-step_or_points", "value"),
)
def refresh_range_previews(
    field1: str | None,
    network_targets1: List[str] | None,
    network_apply1: str | None,
    mode1: str | None,
    start1: float | None,
    end1: float | None,
    step1: float | None,
    field2: str | None,
    network_targets2: List[str] | None,
    network_apply2: str | None,
    mode2: str | None,
    start2: float | None,
    end2: float | None,
    step2: float | None,
    field3: str | None,
    network_targets3: List[str] | None,
    network_apply3: str | None,
    mode3: str | None,
    start3: float | None,
    end3: float | None,
    step3: float | None,
):
    return (
        build_range_preview(resolve_sweep_field(field1, network_targets1, network_apply1), mode1, start1, end1, step1),
        build_range_preview(resolve_sweep_field(field2, network_targets2, network_apply2), mode2, start2, end2, step2),
        build_range_preview(resolve_sweep_field(field3, network_targets3, network_apply3), mode3, start3, end3, step3),
    )


def reset_last_state_values() -> tuple[Any, ...]:
    state = initial_ui_state(ignore_saved=True)
    rows = state["sweep_rows"]
    row_values: List[Any] = []
    for row in rows:
        row_values.extend([display_sweep_field_value(row.get("field")), selected_network_sweep_targets(row), selected_network_apply_mode(row), row.get("mode"), row.get("list_text"), row.get("config_values"), row.get("start"), row.get("end"), row.get("step_or_points")])
    return (
        "Restored default selections.",
        state["model_run_configs"],
        state["hardware_run_configs"],
        state["model_preset"],
        state["hardware_preset"],
        state["active_config_tab"],
        state["run_mode"],
        state["optimize_parallelism"],
        state["optimizer_preset"],
        *row_values,
        state["x_axis"],
        state["series_axis"],
        state["metric"],
        state["worker_count"],
        state["timeout_seconds"],
    )


@callback(
    Output("last-state-status", "children"),
    Output("model-run-configs", "value", allow_duplicate=True),
    Output("hardware-run-configs", "value", allow_duplicate=True),
    Output("model-preset", "value", allow_duplicate=True),
    Output("hardware-preset", "value", allow_duplicate=True),
    Output("config-editor-tabs", "value", allow_duplicate=True),
    Output("run-mode", "value"),
    Output("optimize-switch", "checked"),
    Output("optimizer-preset", "value"),
    Output("dim-1-field", "value"),
    Output("dim-1-network-targets", "value"),
    Output("dim-1-network-apply", "value"),
    Output("dim-1-mode", "value"),
    Output("dim-1-list", "value"),
    Output("dim-1-configs", "value"),
    Output("dim-1-start", "value"),
    Output("dim-1-end", "value"),
    Output("dim-1-step_or_points", "value"),
    Output("dim-2-field", "value"),
    Output("dim-2-network-targets", "value"),
    Output("dim-2-network-apply", "value"),
    Output("dim-2-mode", "value"),
    Output("dim-2-list", "value"),
    Output("dim-2-configs", "value"),
    Output("dim-2-start", "value"),
    Output("dim-2-end", "value"),
    Output("dim-2-step_or_points", "value"),
    Output("dim-3-field", "value"),
    Output("dim-3-network-targets", "value"),
    Output("dim-3-network-apply", "value"),
    Output("dim-3-mode", "value"),
    Output("dim-3-list", "value"),
    Output("dim-3-configs", "value"),
    Output("dim-3-start", "value"),
    Output("dim-3-end", "value"),
    Output("dim-3-step_or_points", "value"),
    Output("x-axis-select", "value", allow_duplicate=True),
    Output("series-select", "value", allow_duplicate=True),
    Output("metric-select", "value", allow_duplicate=True),
    Output("worker-count", "value"),
    Output("timeout-seconds", "value"),
    Input("reset-last-state-button", "n_clicks"),
    prevent_initial_call=True,
)
def reset_saved_last_state(n_clicks: int | None):
    if not n_clicks:
        return tuple(no_update for _ in range(41))
    clear_last_ui_state()
    return reset_last_state_values()


def _network_rows_from_callback(topologies: List[str], bandwidths: List[str], latencies: List[float], utils: List[float]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, topology in enumerate(topologies or []):
        rows.append(
            {
                "topology_type": topology,
                "bandwidth": (bandwidths or [""])[idx] if idx < len(bandwidths or []) else "",
                "latency": (latencies or [0.0])[idx] if idx < len(latencies or []) else 0.0,
                "util": (utils or [1.0])[idx] if idx < len(utils or []) else 1.0,
            }
        )
    return rows


def _dimensions_from_inputs(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dimensions = []
    for row in raw_rows:
        network_targets = row.get("network_targets") if "network_targets" in row else row.get("network_field")
        field_key = resolve_sweep_field(row.get("field"), network_targets, row.get("network_apply"))
        if field_key:
            mode = row["mode"] or "values"
            dimension = {"field_key": field_key, "mode": mode, "list_text": row["list_text"], "config_values": row["config_values"], "start": row["start"], "end": row["end"], "points": None, "step": row["step_or_points"] if mode == "range" else None}
            if row.get("field") == NETWORK_SWEEP_GROUP_VALUE:
                dimension["network_targets"] = selected_network_sweep_targets(row)
                dimension["network_apply"] = selected_network_apply_mode(row)
            dimensions.append(dimension)
    return dimensions


@callback(
    Output("model-yaml", "value", allow_duplicate=True),
    Output("hardware-yaml", "value", allow_duplicate=True),
    Output("config-sync-status", "children"),
    Input("simple-run-type", "value"),
    Input("simple-seq-len", "value"),
    Input("simple-decode-len", "value"),
    Input("simple-batch-size", "value"),
    Input("simple-grad-accum", "value"),
    Input("simple-total-gpus", "value"),
    Input("simple-tp", "value"),
    Input("simple-cp", "value"),
    Input("simple-pp", "value"),
    Input("simple-dp", "value"),
    Input("simple-ep", "value"),
    Input("simple-replica-count", "value"),
    Input("simple-hbm-gb", "value"),
    Input("simple-compute-derate", "value"),
    Input("simple-memory-derate", "value"),
    Input("simple-network-derate", "value"),
    Input("simple-gpu-clock", "value"),
    Input("simple-memory-bw", "value"),
    Input("simple-use-astrasim", "checked"),
    Input("adv-model-type", "value"),
    Input("adv-model-mode", "value"),
    Input("adv-full-recomp", "checked"),
    Input("adv-dp-zero", "value"),
    Input("adv-tensor-format", "value"),
    Input("adv-precision-kv-cache", "value"),
    Input("adv-precision-parameters", "value"),
    Input("adv-precision-gradients", "value"),
    Input("adv-precision-grad-communication", "value"),
    Input("adv-precision-optimizer-states", "value"),
    Input("adv-precision-stats", "value"),
    Input("adv-precision-master-parameters", "value"),
    Input("adv-tied-embeddings", "checked"),
    Input("adv-hidden-dim", "value"),
    Input("adv-intermediate-size", "value"),
    Input("adv-num-layers", "value"),
    Input("adv-vocab-size", "value"),
    Input("adv-attention-type", "value"),
    Input("adv-num-heads", "value"),
    Input("adv-use-flash", "checked"),
    Input("adv-attn-tile", "value"),
    Input("adv-num-experts", "value"),
    Input("adv-top-k", "value"),
    Input("adv-moe-intermediate-size", "value"),
    Input("adv-imbalance", "value"),
    Input({"type": "net-topology", "index": ALL}, "value"),
    Input({"type": "net-bandwidth", "index": ALL}, "value"),
    Input({"type": "net-latency", "index": ALL}, "value"),
    Input({"type": "net-util", "index": ALL}, "value"),
    Input("parallelism-topology-mode", "value"),
    State("model-preset", "value"),
    State("hardware-preset", "value"),
    State("model-run-configs", "value"),
    State("hardware-run-configs", "value"),
    State("run-mode", "value"),
    State("optimize-switch", "checked"),
    State("optimizer-preset", "value"),
    State("metric-select", "value"),
    State("x-axis-select", "value"),
    State("series-select", "value"),
    State("worker-count", "value"),
    State("timeout-seconds", "value"),
    prevent_initial_call=True,
)
def sync_config_files(
    simple_run_type: str,
    simple_seq_len: int,
    simple_decode_len: int,
    simple_batch_size: int,
    simple_grad_accum: int,
    simple_total_gpus: int,
    simple_tp: int,
    simple_cp: int,
    simple_pp: int,
    simple_dp: int,
    simple_ep: int,
    simple_replica_count: int,
    simple_hbm_gb: float,
    simple_compute_derate: float,
    simple_memory_derate: float,
    simple_network_derate: float,
    simple_gpu_clock: float,
    simple_memory_bw: float,
    simple_use_astrasim: bool,
    adv_model_type: str,
    adv_model_mode: str,
    adv_full_recomp: bool,
    adv_dp_zero: int | str,
    adv_tensor_format: str,
    adv_precision_kv_cache: str,
    adv_precision_parameters: str,
    adv_precision_gradients: str,
    adv_precision_grad_communication: str,
    adv_precision_optimizer_states: str,
    adv_precision_stats: str,
    adv_precision_master_parameters: str,
    adv_tied_embeddings: bool,
    adv_hidden_dim: int,
    adv_intermediate_size: int,
    adv_num_layers: int,
    adv_vocab_size: int,
    adv_attention_type: str,
    adv_num_heads: int,
    adv_use_flash: bool,
    adv_attn_tile: int,
    adv_num_experts: int,
    adv_top_k: int,
    adv_moe_intermediate_size: int,
    adv_imbalance: float,
    net_topologies: List[str],
    net_bandwidths: List[str],
    net_latencies: List[float],
    net_utils: List[float],
    parallelism_topology_mode: str,
    model_preset: str,
    hardware_preset: str,
    model_run_configs: List[str] | None,
    hardware_run_configs: List[str] | None,
    run_mode: str,
    optimize_parallelism: bool,
    optimizer_preset: str,
    metric: str,
    x_axis: str | None,
    series_axis: str | None,
    worker_count: int,
    timeout_seconds: int,
):
    payload = collect_form_payload(
        model_preset,
        hardware_preset,
        run_mode,
        optimize_parallelism,
        optimizer_preset,
        simple_run_type,
        simple_seq_len,
        simple_decode_len,
        simple_batch_size,
        simple_grad_accum,
        simple_total_gpus,
        simple_tp,
        simple_cp,
        simple_pp,
        simple_dp,
        simple_ep,
        simple_replica_count,
        simple_hbm_gb,
        simple_compute_derate,
        simple_memory_derate,
        simple_network_derate,
        simple_gpu_clock,
        simple_memory_bw,
        simple_use_astrasim,
        adv_model_type,
        adv_model_mode,
        adv_full_recomp,
        adv_dp_zero,
        adv_tensor_format,
        adv_precision_kv_cache,
        adv_precision_parameters,
        adv_precision_gradients,
        adv_precision_grad_communication,
        adv_precision_optimizer_states,
        adv_precision_stats,
        adv_precision_master_parameters,
        adv_tied_embeddings,
        adv_hidden_dim,
        adv_intermediate_size,
        adv_num_layers,
        adv_vocab_size,
        adv_attention_type,
        adv_num_heads,
        adv_use_flash,
        adv_attn_tile,
        adv_num_experts,
        adv_top_k,
        adv_moe_intermediate_size,
        adv_imbalance,
        net_topologies,
        net_bandwidths,
        net_latencies,
        net_utils,
        parallelism_topology_mode,
        [],
        metric or get_default_metric_for_run_type(simple_run_type),
        x_axis,
        series_axis,
        worker_count or default_worker_count(),
        timeout_seconds if timeout_seconds is not None else 180,
    )
    save_errors: List[str] = []
    _, _, save_errors = save_config_edits_from_payload(payload)
    if save_errors:
        return no_update, no_update, f"Config sync failed: {save_errors[0]}"
    model_yaml, hardware_yaml, render_errors = render_editable_config_texts(payload)
    if render_errors:
        return no_update, no_update, f"Config sync failed: {render_errors[0]}"
    return model_yaml, hardware_yaml, ""


@callback(
    Output("preview-store", "data"),
    Output("preview-summary", "children"),
    Output("run-button", "children"),
    Input("model-preset", "value"),
    Input("hardware-preset", "value"),
    Input("model-run-configs", "value"),
    Input("hardware-run-configs", "value"),
    Input("run-mode", "value"),
    Input("optimize-switch", "checked"),
    Input("optimizer-preset", "value"),
    Input("simple-run-type", "value"),
    Input("simple-seq-len", "value"),
    Input("simple-decode-len", "value"),
    Input("simple-batch-size", "value"),
    Input("simple-grad-accum", "value"),
    Input("simple-total-gpus", "value"),
    Input("simple-tp", "value"),
    Input("simple-cp", "value"),
    Input("simple-pp", "value"),
    Input("simple-dp", "value"),
    Input("simple-ep", "value"),
    Input("simple-replica-count", "value"),
    Input("simple-hbm-gb", "value"),
    Input("simple-compute-derate", "value"),
    Input("simple-memory-derate", "value"),
    Input("simple-network-derate", "value"),
    Input("simple-gpu-clock", "value"),
    Input("simple-memory-bw", "value"),
    Input("simple-use-astrasim", "checked"),
    Input("adv-model-type", "value"),
    Input("adv-model-mode", "value"),
    Input("adv-full-recomp", "checked"),
    Input("adv-dp-zero", "value"),
    Input("adv-tensor-format", "value"),
    Input("adv-precision-kv-cache", "value"),
    Input("adv-precision-parameters", "value"),
    Input("adv-precision-gradients", "value"),
    Input("adv-precision-grad-communication", "value"),
    Input("adv-precision-optimizer-states", "value"),
    Input("adv-precision-stats", "value"),
    Input("adv-precision-master-parameters", "value"),
    Input("adv-tied-embeddings", "checked"),
    Input("adv-hidden-dim", "value"),
    Input("adv-intermediate-size", "value"),
    Input("adv-num-layers", "value"),
    Input("adv-vocab-size", "value"),
    Input("adv-attention-type", "value"),
    Input("adv-num-heads", "value"),
    Input("adv-use-flash", "checked"),
    Input("adv-attn-tile", "value"),
    Input("adv-num-experts", "value"),
    Input("adv-top-k", "value"),
    Input("adv-moe-intermediate-size", "value"),
    Input("adv-imbalance", "value"),
    Input({"type": "net-topology", "index": ALL}, "value"),
    Input({"type": "net-bandwidth", "index": ALL}, "value"),
    Input({"type": "net-latency", "index": ALL}, "value"),
    Input({"type": "net-util", "index": ALL}, "value"),
    Input("parallelism-topology-mode", "value"),
    Input("dim-1-field", "value"),
    Input("dim-1-network-targets", "value"),
    Input("dim-1-network-apply", "value"),
    Input("dim-1-mode", "value"),
    Input("dim-1-list", "value"),
    Input("dim-1-configs", "value"),
    Input("dim-1-start", "value"),
    Input("dim-1-end", "value"),
    Input("dim-1-step_or_points", "value"),
    Input("dim-2-field", "value"),
    Input("dim-2-network-targets", "value"),
    Input("dim-2-network-apply", "value"),
    Input("dim-2-mode", "value"),
    Input("dim-2-list", "value"),
    Input("dim-2-configs", "value"),
    Input("dim-2-start", "value"),
    Input("dim-2-end", "value"),
    Input("dim-2-step_or_points", "value"),
    Input("dim-3-field", "value"),
    Input("dim-3-network-targets", "value"),
    Input("dim-3-network-apply", "value"),
    Input("dim-3-mode", "value"),
    Input("dim-3-list", "value"),
    Input("dim-3-configs", "value"),
    Input("dim-3-start", "value"),
    Input("dim-3-end", "value"),
    Input("dim-3-step_or_points", "value"),
    Input("metric-select", "value"),
    Input("x-axis-select", "value"),
    Input("series-select", "value"),
    Input("worker-count", "value"),
    Input("timeout-seconds", "value"),
    Input("config-editor-tabs", "value"),
)
def build_preview(
    model_preset: str,
    hardware_preset: str,
    model_run_configs: List[str] | None,
    hardware_run_configs: List[str] | None,
    run_mode: str,
    optimize_parallelism: bool,
    optimizer_preset: str,
    simple_run_type: str,
    simple_seq_len: int,
    simple_decode_len: int,
    simple_batch_size: int,
    simple_grad_accum: int,
    simple_total_gpus: int,
    simple_tp: int,
    simple_cp: int,
    simple_pp: int,
    simple_dp: int,
    simple_ep: int,
    simple_replica_count: int,
    simple_hbm_gb: float,
    simple_compute_derate: float,
    simple_memory_derate: float,
    simple_network_derate: float,
    simple_gpu_clock: float,
    simple_memory_bw: float,
    simple_use_astrasim: bool,
    adv_model_type: str,
    adv_model_mode: str,
    adv_full_recomp: bool,
    adv_dp_zero: int | str,
    adv_tensor_format: str,
    adv_precision_kv_cache: str,
    adv_precision_parameters: str,
    adv_precision_gradients: str,
    adv_precision_grad_communication: str,
    adv_precision_optimizer_states: str,
    adv_precision_stats: str,
    adv_precision_master_parameters: str,
    adv_tied_embeddings: bool,
    adv_hidden_dim: int,
    adv_intermediate_size: int,
    adv_num_layers: int,
    adv_vocab_size: int,
    adv_attention_type: str,
    adv_num_heads: int,
    adv_use_flash: bool,
    adv_attn_tile: int,
    adv_num_experts: int,
    adv_top_k: int,
    adv_moe_intermediate_size: int,
    adv_imbalance: float,
    net_topologies: List[str],
    net_bandwidths: List[str],
    net_latencies: List[float],
    net_utils: List[float],
    parallelism_topology_mode: str,
    dim1_field: str,
    dim1_network_targets: List[str],
    dim1_network_apply: str,
    dim1_mode: str,
    dim1_list: str,
    dim1_configs: List[str],
    dim1_start: float,
    dim1_end: float,
    dim1_step_or_points: float,
    dim2_field: str,
    dim2_network_targets: List[str],
    dim2_network_apply: str,
    dim2_mode: str,
    dim2_list: str,
    dim2_configs: List[str],
    dim2_start: float,
    dim2_end: float,
    dim2_step_or_points: float,
    dim3_field: str,
    dim3_network_targets: List[str],
    dim3_network_apply: str,
    dim3_mode: str,
    dim3_list: str,
    dim3_configs: List[str],
    dim3_start: float,
    dim3_end: float,
    dim3_step_or_points: float,
    metric: str,
    x_axis: str | None,
    series_axis: str | None,
    worker_count: int,
    timeout_seconds: int,
    active_config_tab: str | None,
):
    if preview_rebuild_is_load_only(dash.ctx.triggered_prop_ids):
        return no_update, no_update, no_update
    sweep_rows = [
        {"field": dim1_field, "network_targets": dim1_network_targets, "network_apply": dim1_network_apply, "mode": dim1_mode, "list_text": dim1_list, "config_values": dim1_configs, "start": dim1_start, "end": dim1_end, "step_or_points": dim1_step_or_points},
        {"field": dim2_field, "network_targets": dim2_network_targets, "network_apply": dim2_network_apply, "mode": dim2_mode, "list_text": dim2_list, "config_values": dim2_configs, "start": dim2_start, "end": dim2_end, "step_or_points": dim2_step_or_points},
        {"field": dim3_field, "network_targets": dim3_network_targets, "network_apply": dim3_network_apply, "mode": dim3_mode, "list_text": dim3_list, "config_values": dim3_configs, "start": dim3_start, "end": dim3_end, "step_or_points": dim3_step_or_points},
    ]
    dimensions = config_dimensions_from_selection(model_run_configs, hardware_run_configs, model_preset, hardware_preset)
    dimensions.extend(_dimensions_from_inputs(sweep_rows))
    payload = collect_form_payload(
        model_preset,
        hardware_preset,
        run_mode,
        optimize_parallelism,
        optimizer_preset,
        simple_run_type,
        simple_seq_len,
        simple_decode_len,
        simple_batch_size,
        simple_grad_accum,
        simple_total_gpus,
        simple_tp,
        simple_cp,
        simple_pp,
        simple_dp,
        simple_ep,
        simple_replica_count,
        simple_hbm_gb,
        simple_compute_derate,
        simple_memory_derate,
        simple_network_derate,
        simple_gpu_clock,
        simple_memory_bw,
        simple_use_astrasim,
        adv_model_type,
        adv_model_mode,
        adv_full_recomp,
        adv_dp_zero,
        adv_tensor_format,
        adv_precision_kv_cache,
        adv_precision_parameters,
        adv_precision_gradients,
        adv_precision_grad_communication,
        adv_precision_optimizer_states,
        adv_precision_stats,
        adv_precision_master_parameters,
        adv_tied_embeddings,
        adv_hidden_dim,
        adv_intermediate_size,
        adv_num_layers,
        adv_vocab_size,
        adv_attention_type,
        adv_num_heads,
        adv_use_flash,
        adv_attn_tile,
        adv_num_experts,
        adv_top_k,
        adv_moe_intermediate_size,
        adv_imbalance,
        net_topologies,
        net_bandwidths,
        net_latencies,
        net_utils,
        parallelism_topology_mode,
        dimensions,
        metric,
        x_axis,
        series_axis,
        worker_count,
        timeout_seconds,
    )
    save_last_ui_state(
        {
            "model_run_configs": model_run_configs,
            "hardware_run_configs": hardware_run_configs,
            "model_preset": model_preset,
            "hardware_preset": hardware_preset,
            "active_config_tab": active_config_tab,
            "run_mode": run_mode,
            "optimize_parallelism": optimize_parallelism,
            "optimizer_preset": optimizer_preset,
            "sweep_rows": sweep_rows,
            "metric": metric,
            "x_axis": x_axis,
            "series_axis": series_axis,
            "worker_count": worker_count,
            "timeout_seconds": timeout_seconds,
        }
    )
    preview = build_launch_preview(payload)
    if not preview.get("ok"):
        return {"payload": payload, "preview": preview}, render_error_summary(preview.get("errors", [])), launch_button_label(preview)
    return {"payload": payload, "preview": preview}, render_preview_summary(preview, metric), launch_button_label(preview)


@callback(Output("preview-summary", "children", allow_duplicate=True), Input("run-button", "n_clicks"), State("preview-store", "data"), prevent_initial_call=True)
def launch_job(_: int, preview_store: Dict[str, Any] | None):
    if not preview_store:
        return dmc.Alert("The launch plan is still loading. Try again in a moment.", color="red", radius="lg")
    preview, payload = preview_store["preview"], preview_store["payload"]
    if not preview.get("ok"):
        return dmc.Alert("Cannot run until preview errors are fixed.", color="red", radius="lg")
    ok, message = RUN_MANAGER.start_job(payload, preview)
    if ok:
        return no_update
    return dmc.Alert(f"Did not launch: {message}", color="red", radius="lg")


@callback(Output("preview-summary", "children", allow_duplicate=True), Input("cancel-button", "n_clicks"), prevent_initial_call=True)
def cancel_active_job(_: int):
    ok, message = RUN_MANAGER.cancel()
    return dmc.Alert(message, color="blue" if ok else "red", radius="lg")


@callback(Output("workspace-tabs", "value"), Input("progress-run-log-button", "n_clicks"), prevent_initial_call=True)
def open_run_log_from_completed_progress(n_clicks: int | None):
    if not n_clicks:
        return no_update
    return "history"


@callback(Output("selected-detail-store", "data"), Input({"type": "open-detail", "job_kind": ALL, "job_id": ALL}, "n_clicks"), prevent_initial_call=True)
def open_detail(_: List[int]):
    trigger = dash.ctx.triggered_id
    n_clicks = dash.ctx.triggered[0].get("value") if dash.ctx.triggered else None
    if not isinstance(trigger, dict) or not n_clicks:
        return no_update
    return {"kind": trigger["job_kind"], "id": trigger["job_id"], "nonce": n_clicks}


@callback(
    Output("selected-detail-store", "data", allow_duplicate=True),
    Input("detail-close-button", "n_clicks"),
    Input("detail-backdrop", "n_clicks"),
    prevent_initial_call=True,
)
def close_detail_modal(_: int | None, __: int | None):
    if not dash.ctx.triggered_id:
        return no_update
    return None


@callback(
    Output("detail-overlay", "style"),
    Output("detail-plot-toolbar", "style"),
    Output("detail-modal-title", "children"),
    Output("details-panel", "children", allow_duplicate=True),
    Output("detail-display-mode", "value"),
    Input("selected-detail-store", "data"),
    prevent_initial_call=True,
)
def open_detail_modal_shell(selected_detail: Dict[str, Any] | None):
    if not selected_detail:
        return {"display": "none"}, {"display": "none"}, "", render_detail(None), "top"
    plot_toolbar_style = {"display": "block"} if selected_detail.get("kind") == "sweep" else {"display": "none"}
    return {"display": "flex"}, plot_toolbar_style, html.Div("Details", className="detail-modal-heading"), detail_loading_placeholder(selected_detail), "top"


@callback(
    Output("details-panel", "children", allow_duplicate=True),
    Input("selected-detail-store", "data"),
    Input("detail-plot-type", "value"),
    Input("detail-display-mode", "value"),
    prevent_initial_call=True,
)
def render_detail_modal_content(selected_detail: Dict[str, Any] | None, plot_type: str | None, display_mode: str | None):
    if not selected_detail:
        return render_detail(None, plot_type)
    case_limit = None if display_mode == "full" else DETAIL_RENDER_CASE_LIMIT
    detail = load_job_detail(selected_detail["kind"], selected_detail["id"], case_limit=case_limit, display_mode=display_mode or "top")
    return render_detail(detail, plot_type or "line")


@callback(
    Output("plot-save-status", "children"),
    Output("plot-download", "data"),
    Input("save-plot-button", "n_clicks"),
    State("selected-detail-store", "data"),
    State("detail-plot-type", "value"),
    prevent_initial_call=True,
)
def save_current_detail_plot(n_clicks: int | None, selected_detail: Dict[str, Any] | None, plot_type: str | None):
    if not n_clicks or not selected_detail:
        return no_update, no_update
    detail = load_job_detail(selected_detail["kind"], selected_detail["id"])
    if detail.get("kind") != "sweep":
        return "This job does not have a sweep plot to save.", no_update
    rows = sweep_rows_from_detail(detail)
    if not rows:
        return "No case rows are available to plot.", no_update
    x_axis, y_axis, series_axis = detail_axes(detail, rows)
    for row in rows:
        if y_axis not in row:
            row[y_axis] = 0
        if row.get("status") != "completed" and y_axis:
            row[y_axis] = 0
    png_bytes = detail_plot_png_bytes(rows, x_axis, y_axis, series_axis, plot_type or "line", detail.get("title") or selected_detail["id"])
    path = save_plot_png(selected_detail["kind"], selected_detail["id"], detail.get("title") or selected_detail["id"], png_bytes)
    return f"Downloaded and saved plot: {path}", download_payload_for_file(path, png_bytes, "image/png")


@callback(
    Output("table-export-status", "children"),
    Output("table-download", "data"),
    Input("export-table-csv-button", "n_clicks"),
    Input("export-table-json-button", "n_clicks"),
    State("selected-detail-store", "data"),
    prevent_initial_call=True,
)
def export_current_detail_table(csv_clicks: int | None, json_clicks: int | None, selected_detail: Dict[str, Any] | None):
    if not selected_detail or selected_detail.get("kind") != "sweep":
        return "Only sweep details have an exportable table.", no_update
    trigger = dash.ctx.triggered_id
    if trigger not in {"export-table-csv-button", "export-table-json-button"}:
        return no_update, no_update
    if trigger == "export-table-csv-button" and not csv_clicks:
        return no_update, no_update
    if trigger == "export-table-json-button" and not json_clicks:
        return no_update, no_update
    fmt = "csv" if trigger == "export-table-csv-button" else "json"
    detail = load_job_detail(selected_detail["kind"], selected_detail["id"], display_mode="full")
    content, row_count = detail_table_export_payload(detail, fmt)
    path = save_table_export(selected_detail["kind"], selected_detail["id"], detail.get("title") or selected_detail["id"], fmt, content)
    mime_type = "text/csv" if fmt == "csv" else "application/json"
    return f"Downloaded {row_count:,} rows and saved: {path}", download_payload_for_file(path, content, mime_type)


def render_active_job_panel(active: Dict[str, Any] | None = None, finished: Dict[str, Any] | None = None) -> Any:
    if active:
        progress = ((active.get("progress_completed") or 0) / max(1, active.get("progress_total") or 1)) * 100
        return dmc.Stack(
            gap="md",
            children=[
                dmc.Text(active["title"], fw=700, size="lg"),
                branded_progress_bar(
                    progress,
                    progress_count_label(active),
                    str(active["status"]).upper(),
                ),
                dmc.Text(job_eta_readout(active), size="sm", fw=800, className="live-status-eta"),
            ],
        )
    if finished:
        return dmc.Stack(
            gap="md",
            children=[
                dmc.Text(finished["title"], fw=700, size="lg"),
                branded_progress_bar(
                    100,
                    progress_count_label(finished, terminal=True),
                    str(finished["status"]).upper(),
                    show_run_log_button=True,
                ),
            ],
        )
    return dmc.Alert("No active job. Adjust settings and launch when ready.", color="green", radius="lg")


def render_history_panel() -> dmc.Stack:
    history_items = list_history()
    history_children = []
    for item in history_items:
        summary = item.get("summary") or {}
        selected_metric = (item.get("request") or {}).get("payload", {}).get("metric")
        summary_badges = [
            dmc.Badge(item["kind"].upper(), radius="xl", variant="light", color="blue"),
            dmc.Badge(str(item["status"]).upper(), radius="xl", variant="light", color="gray"),
        ]
        if item["kind"] == "run" and summary.get("primary_metric_value") is not None:
            primary_metric_key = metric_key_from_label(summary.get("primary_metric_label"))
            summary_badges.append(
                dmc.Badge(
                    f"{pretty_label(summary.get('primary_metric_label') or '')}: {format_metric_value(summary.get('primary_metric_value'), primary_metric_key)}",
                    radius="xl",
                    variant="light",
                    color="teal",
                )
            )
        if item["kind"] == "sweep":
            if summary.get("case_count") is not None:
                summary_badges.append(dmc.Badge(f"{format_metric_value(summary['case_count'], 'case_count')} cases", radius="xl", variant="light", color="violet"))
            if summary.get("best_metric_value") is not None:
                best_metric_key = selected_metric or metric_key_from_label(summary.get("best_metric_label"))
                summary_badges.append(
                    dmc.Badge(
                        f"Best {pretty_label(summary.get('best_metric_label') or '')}: {format_metric_value(summary.get('best_metric_value'), best_metric_key)}",
                        radius="xl",
                        variant="light",
                        color="teal",
                    )
                )
        history_children.append(
            dmc.Paper(
                radius="xl",
                p="md",
                withBorder=True,
                mb="sm",
                className="history-card",
                children=dmc.Group(
                    justify="space-between",
                    align="center",
                    children=[
                        dmc.Stack(
                            gap="xs",
                            children=[
                                history_title_component(item),
                                dmc.Group(gap="xs", children=summary_badges),
                                dmc.Text(compact_timestamp(item.get("created_at")), size="xs", c="dimmed"),
                            ],
                        ),
                        with_tip(dmc.Button("Details", id={"type": "open-detail", "job_kind": item["kind"], "job_id": item["id"]}, variant="light", radius="xl", leftSection=DashIconify(icon="solar:arrow-right-up-bold")), HELP_TEXT["load_details"]),
                    ],
                ),
            )
        )
    if not history_children:
        history_children = [dmc.Alert("No runs yet. Use the Launch tab to create one.", color="gray", radius="lg")]
    return dmc.Stack(children=history_children)


def job_status_signature(active: Dict[str, Any] | None, finished: Dict[str, Any] | None) -> str:
    source = active or finished or {}
    return json.dumps(
        {
            "active": bool(active),
            "id": source.get("id"),
            "status": source.get("status"),
            "progress_completed": source.get("progress_completed"),
            "progress_total": source.get("progress_total"),
            "updated_at": source.get("updated_at"),
        },
        sort_keys=True,
    )


@callback(
    Output("telemetry-ram", "children"),
    Output("telemetry-cpu", "children"),
    Input("telemetry-poller", "n_intervals"),
)
def refresh_telemetry(_: int):
    telemetry = get_telemetry()
    return f"RAM {format_metric_value(telemetry['available_ram_gb'], 'available_ram_gb')} free", f"CPU {telemetry['cpu_percent']}%"


@callback(
    Output("telemetry-job", "children"),
    Output("active-job-panel", "children"),
    Output("history-refresh-store", "data"),
    Input("job-poller", "n_intervals"),
    State("history-refresh-store", "data"),
)
def refresh_job_status(_: int, current_signature_record: Dict[str, Any] | None):
    active = RUN_MANAGER.active_job()
    if active:
        active_panel = render_active_job_panel(active=active)
        job_badge = f"Active: {active['status']}"
        finished = None
    else:
        finished = RUN_MANAGER.last_finished_job()
        if finished:
            active_panel = render_active_job_panel(finished=finished)
            job_badge = format_finished_job_badge(finished)
        else:
            active_panel = render_active_job_panel()
            job_badge = "Idle"
    signature = job_status_signature(active, finished)
    if isinstance(current_signature_record, dict) and current_signature_record.get("signature") == signature:
        return job_badge, no_update, no_update
    return job_badge, active_panel, {"signature": signature, "updated_at": datetime.now().isoformat()}


@callback(Output("history-panel", "children"), Input("history-refresh-store", "data"))
def refresh_history(_: Dict[str, Any] | None):
    return render_history_panel()


DATA_TABLE_STYLE_CELL = {
    "padding": "10px",
    "fontFamily": "Arial, Calibri, Segoe UI, sans-serif",
    "backgroundColor": "#ffffff",
    "color": "#17212b",
    "border": "1px solid #d7e4ee",
    "minWidth": "120px",
    "maxWidth": "300px",
    "whiteSpace": "normal",
}
DATA_TABLE_STYLE_HEADER = {"backgroundColor": "#eef7fc", "fontWeight": 800, "color": "#0055A6", "border": "1px solid #bfd2df"}
DATA_TABLE_STYLE_FILTER = {"backgroundColor": "#ffffff", "color": "#17212b", "border": "1px solid #bfd2df", "fontWeight": 500}
MEMORY_EXCEEDED_CELL_STYLE = {"color": "#c1121f", "fontWeight": 900}
RUN_MEMORY_STYLE_DATA_CONDITIONAL = [
    {
        "if": {"filter_query": '{metric} = "Memory Exceeded" && {value} != "No"', "column_id": "value"},
        **MEMORY_EXCEEDED_CELL_STYLE,
    }
]
SWEEP_MEMORY_STYLE_DATA_CONDITIONAL = [
    {
        "if": {"filter_query": '{memory_exceeded} != "No"', "column_id": "memory_exceeded"},
        **MEMORY_EXCEEDED_CELL_STYLE,
    }
]


def sweep_rows_from_detail(detail: Dict[str, Any]) -> List[Dict[str, Any]]:
    payload = detail_payload(detail)
    rows = []
    for case in detail.get("cases", []) or []:
        dimension_values = case.get("dimension_values", {}) or {}
        metrics = metrics_with_derived_flops(case.get("metrics", {}) or {})
        model_config = detail_model_config(detail, dimension_values)
        hardware_config = detail_hardware_config(detail, dimension_values)
        generated_label = build_case_label(dimension_values, model_config, hardware_config)
        raw_label = case.get("label")
        case_label = generated_label if dimension_values and raw_label and "=" in str(raw_label) else (raw_label or generated_label)
        row = {
            "case": f"{case['case_id']} - {case_label}",
            "case_id": case["case_id"],
            "label": case_label,
            "status": case.get("status", "unknown"),
            "model_config": model_config,
            "hardware_config": hardware_config,
            "parallelism": parallelism_summary_from_payload(payload, case.get("chosen_candidate")),
        }
        row.update({key: value for key, value in dimension_values.items() if key not in {"model_config", "hardware_config"}})
        row["model.seq_len"] = detail_sequence_length(detail, dimension_values)
        if detail_run_type(detail) == "inference":
            row["model.decode_len"] = detail_decode_length(detail, dimension_values)
        row.update({key: value for key, value in metrics.items() if key != "memory_violation_gb" and key not in DETAIL_HIDDEN_METRIC_KEYS})
        if "memory_exceeded" in metrics or "memory_violation_gb" in metrics:
            row["memory_exceeded"] = memory_exceeded_display(metrics)
        rows.append(row)
    return rows


def detail_axes(detail: Dict[str, Any], rows: List[Dict[str, Any]]) -> tuple[str, str, str | None]:
    payload = detail_payload(detail)
    cases = detail.get("cases", []) or []
    preferred_x = payload.get("x_axis")
    preferred_metric = payload.get("metric") or "training_time_s"
    dimension_keys = list((cases[0].get("dimension_values") or {}).keys()) if cases else []
    if preferred_x in rows[0]:
        x_axis = preferred_x
    elif len(dimension_keys) == 1 and dimension_keys[0] in rows[0]:
        x_axis = dimension_keys[0]
    else:
        x_axis = "case_id"
    if preferred_metric in DETAIL_HIDDEN_METRIC_KEYS:
        preferred_metric = None
    y_axis = preferred_metric if preferred_metric and preferred_metric in rows[0] else ("training_time_s" if "training_time_s" in rows[0] else "prefill_time_s")
    return x_axis, y_axis, payload.get("series_axis")


def sample_rows_evenly(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    if limit == 1:
        return [rows[0]]
    last = len(rows) - 1
    indexes = sorted({round(index * last / (limit - 1)) for index in range(limit)})
    return [rows[index] for index in indexes]


def detail_case_counts(detail: Dict[str, Any], loaded_rows: int) -> tuple[int, int]:
    loaded_cases = int(detail.get("_case_count_loaded") or loaded_rows)
    total_cases = int(detail.get("_case_count_total") or len(detail.get("cases", []) or []) or loaded_cases)
    return total_cases, loaded_cases


def detail_termination_stats(detail: Dict[str, Any], rows: List[Dict[str, Any]] | None = None) -> tuple[int, int, float]:
    summary = detail.get("summary_record") or detail.get("summary") or {}
    if detail.get("kind") == "sweep":
        total = summary.get("case_count")
        completed = summary.get("completed_case_count")
        if total is not None and completed is not None:
            try:
                total_i = max(0, int(total))
                completed_i = max(0, int(completed))
                early_i = max(0, total_i - completed_i)
                pct = (early_i / total_i * 100.0) if total_i else 0.0
                return total_i, early_i, pct
            except (TypeError, ValueError):
                pass
        source_rows = rows if rows is not None else sweep_rows_from_detail(detail)
        total_i = len(source_rows)
        early_i = sum(1 for row in source_rows if str(row.get("status", "")).lower() != "completed")
        pct = (early_i / total_i * 100.0) if total_i else 0.0
        return total_i, early_i, pct
    result = detail.get("result", {}) or {}
    status = str(result.get("status", "unknown")).lower()
    total_i = 1 if status else 0
    early_i = 1 if status and status != "completed" else 0
    pct = (early_i / total_i * 100.0) if total_i else 0.0
    return total_i, early_i, pct


def termination_summary_component(detail: Dict[str, Any], rows: List[Dict[str, Any]] | None = None):
    total, early, percent = detail_termination_stats(detail, rows)
    base_text = f"Early termination rate: {percent:.1f}% ({early:,}/{total:,} runs)."
    if percent > EARLY_TERMINATION_BIG_THRESHOLD:
        return dmc.Alert(
            f"{base_text} Results are unreliable because most runs terminated before producing usable metrics. Increase the timeout and rerun for higher result fidelity.",
            color="red",
            radius="lg",
            title="Results unreliable",
            style={"fontStyle": "italic"},
        )
    if percent > EARLY_TERMINATION_MILD_THRESHOLD:
        return dmc.Alert(
            f"{base_text} Many runs terminated early; consider increasing the timeout for higher result fidelity.",
            color="red",
            radius="lg",
            title="Early termination warning",
            style={"fontStyle": "italic"},
        )
    return dmc.Text(base_text, size="sm", c="dimmed")


def row_memory_exceeded(row: Dict[str, Any]) -> bool:
    raw = row.get("memory_exceeded")
    if isinstance(raw, str):
        return raw.strip().lower() not in {"", "no", "false", "0", "none"}
    return bool(raw)


def _customdata_memory_flags(trace: Any) -> List[str]:
    customdata = getattr(trace, "customdata", None)
    if customdata is None:
        return []
    flags: List[str] = []
    for item in customdata:
        if isinstance(item, (str, bytes)):
            flags.append(str(item))
            continue
        try:
            if len(item):
                flags.append(str(item[0]))
                continue
        except (TypeError, IndexError):
            pass
        flags.append(str(item))
    return flags


def _apply_plotly_oom_styling(figure: Any, plot_kind: str, has_oom: bool) -> None:
    for trace in figure.data:
        flags = _customdata_memory_flags(trace)
        if not flags:
            continue
        if plot_kind == "bar":
            trace.marker.pattern.shape = ["x" if flag == "OOM" else "" for flag in flags]
            trace.marker.pattern.fgcolor = "#c1121f"
            trace.marker.line.color = ["#c1121f" if flag == "OOM" else "rgba(0,0,0,0)" for flag in flags]
            trace.marker.line.width = [1.2 if flag == "OOM" else 0 for flag in flags]
        else:
            trace.marker.symbol = ["x" if flag == "OOM" else "circle" for flag in flags]
            trace.marker.size = [11 if flag == "OOM" else 7 for flag in flags]
            trace.marker.line.width = [2 if flag == "OOM" else 0 for flag in flags]
            trace.marker.line.color = ["#c1121f" if flag == "OOM" else "rgba(0,0,0,0)" for flag in flags]
    if not has_oom:
        return
    if plot_kind == "bar":
        figure.add_trace(
            go.Bar(
                x=[None],
                y=[None],
                name="OOM",
                marker={"color": "#ffffff", "line": {"color": "#c1121f", "width": 1.2}, "pattern": {"shape": "x", "fgcolor": "#c1121f"}},
                showlegend=True,
                hoverinfo="skip",
            )
        )
    else:
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="OOM",
                marker={"symbol": "x", "color": "#c1121f", "size": 11, "line": {"width": 2, "color": "#c1121f"}},
                showlegend=True,
                hoverinfo="skip",
            )
        )


def detail_plot_figure(rows: List[Dict[str, Any]], x_axis: str, y_axis: str, series_axis: str | None, plot_type: str | None):
    plot_rows = [dict(row) for row in rows]
    for row in plot_rows:
        if row.get("status") != "completed":
            row[y_axis] = 0
        row["_memory_fit"] = "OOM" if row_memory_exceeded(row) else "Fits memory"
    has_oom = any(row["_memory_fit"] == "OOM" for row in plot_rows)
    color_axis = series_axis if series_axis and series_axis in plot_rows[0] and series_axis != "status" else None
    if color_axis is None:
        for candidate in ["model_config", "hardware_config"]:
            values = {row.get(candidate) for row in plot_rows}
            if len(values) > 1:
                color_axis = candidate
                break
    hover_fields = [key for key in ["status", "memory_exceeded", "model_config", "hardware_config", "parallelism"] if key in plot_rows[0]]
    plot_kind = plot_type or "line"
    common = {
        "data_frame": plot_rows,
        "x": x_axis,
        "y": y_axis,
        "hover_name": "case",
        "hover_data": hover_fields,
        "custom_data": ["_memory_fit"],
        "color": color_axis,
        "labels": {"_memory_fit": "Memory fit"},
        "template": "plotly_white",
        "color_discrete_sequence": ["#0055A6", "#00A5E5", "#3284BF", "#7DD3F7", "#4F8F2F"],
    }
    if plot_kind == "bar":
        figure = px.bar(**common)
    elif plot_kind == "scatter":
        figure = px.scatter(**common)
    else:
        figure = px.line(**common, markers=True)
    _apply_plotly_oom_styling(figure, plot_kind, has_oom)
    figure.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font_family="Arial, Calibri, Segoe UI, sans-serif",
        font_color="#17212b",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title=pretty_label(x_axis),
        yaxis_title=pretty_label(y_axis),
        legend_title_text=pretty_label(color_axis) if color_axis else "",
    )
    figure.update_xaxes(gridcolor="#e3edf4", zerolinecolor="#bfd2df")
    figure.update_yaxes(gridcolor="#e3edf4", zerolinecolor="#bfd2df")
    return figure


PLOT_EXPORT_COLORS = ["#0055A6", "#00A5E5", "#3284BF", "#7DD3F7", "#4F8F2F", "#6BAED6"]


def _plot_color_axis(plot_rows: List[Dict[str, Any]], series_axis: str | None) -> str | None:
    if series_axis and series_axis in plot_rows[0] and series_axis != "status":
        return series_axis
    for candidate in ["model_config", "hardware_config"]:
        values = {row.get(candidate) for row in plot_rows}
        if len(values) > 1:
            return candidate
    return None


def _plot_number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _plot_x_positions(plot_rows: List[Dict[str, Any]], x_axis: str) -> tuple[List[float], List[str] | None]:
    raw_values = [row.get(x_axis) for row in plot_rows]
    numeric_values: List[float] = []
    for value in raw_values:
        if isinstance(value, bool):
            break
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            break
    else:
        return numeric_values, None
    labels = [str(value) for value in raw_values]
    unique_labels = list(dict.fromkeys(labels))
    positions = {label: float(idx) for idx, label in enumerate(unique_labels)}
    return [positions[label] for label in labels], unique_labels


def _format_plot_axis_tick(value: float, axis_key: str) -> str:
    if axis_key in FIELD_TYPES:
        return format_sweep_preview_value(value, axis_key)
    if axis_key not in (FLOP_COUNT_KEYS | FLOP_RATE_KEYS | TOKEN_RATE_KEYS | TIME_KEYS) and not axis_key.endswith(("_flops", "_time_s", "_seconds", "_s")):
        if abs(value - round(value)) < 1e-9:
            return f"{int(round(value)):,}"
    return format_metric_value(value, axis_key)


def detail_plot_png_bytes(rows: List[Dict[str, Any]], x_axis: str, y_axis: str, series_axis: str | None, plot_type: str | None, title: str) -> bytes:
    plot_rows = [dict(row) for row in rows]
    for row in plot_rows:
        if row.get("status") != "completed":
            row[y_axis] = 0
    color_axis = _plot_color_axis(plot_rows, series_axis)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in plot_rows:
        group_label = str(row.get(color_axis)) if color_axis else "Result"
        grouped.setdefault(group_label, []).append(row)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    plot_kind = plot_type or "line"
    all_x_positions, category_labels = _plot_x_positions(plot_rows, x_axis)
    x_lookup = {id(row): x for row, x in zip(plot_rows, all_x_positions)}

    if plot_kind == "bar":
        group_count = max(1, len(grouped))
        bar_width = min(0.8 / group_count, 0.34)
        for group_index, (group_label, group_rows) in enumerate(grouped.items()):
            offset = (group_index - (group_count - 1) / 2) * bar_width
            xs = [x_lookup[id(row)] + offset for row in group_rows]
            ys = [_plot_number(row.get(y_axis)) for row in group_rows]
            bars = ax.bar(xs, ys, width=bar_width * 0.9, label=group_label if color_axis else None, color=PLOT_EXPORT_COLORS[group_index % len(PLOT_EXPORT_COLORS)])
            for bar, row in zip(bars, group_rows):
                if row_memory_exceeded(row):
                    bar.set_hatch("///")
                    bar.set_edgecolor("#c1121f")
                    bar.set_linewidth(1.3)
    else:
        for group_index, (group_label, group_rows) in enumerate(grouped.items()):
            xs = [x_lookup[id(row)] for row in group_rows]
            ys = [_plot_number(row.get(y_axis)) for row in group_rows]
            color = PLOT_EXPORT_COLORS[group_index % len(PLOT_EXPORT_COLORS)]
            label = group_label if color_axis else None
            if plot_kind == "scatter":
                fit_points = [(x, y) for x, y, row in zip(xs, ys, group_rows) if not row_memory_exceeded(row)]
                oom_points = [(x, y) for x, y, row in zip(xs, ys, group_rows) if row_memory_exceeded(row)]
                if fit_points:
                    ax.scatter([item[0] for item in fit_points], [item[1] for item in fit_points], s=48, label=label, color=color, edgecolors="#ffffff", linewidths=0.8, zorder=3)
                    label = None
                if oom_points:
                    ax.scatter([item[0] for item in oom_points], [item[1] for item in oom_points], s=78, label=label, color="#c1121f", marker="x", linewidths=2.0, zorder=4)
            else:
                ax.plot(xs, ys, linewidth=2.2, label=label, color=color, zorder=2)
                fit_points = [(x, y) for x, y, row in zip(xs, ys, group_rows) if not row_memory_exceeded(row)]
                oom_points = [(x, y) for x, y, row in zip(xs, ys, group_rows) if row_memory_exceeded(row)]
                if fit_points:
                    ax.scatter([item[0] for item in fit_points], [item[1] for item in fit_points], s=36, color=color, edgecolors="#ffffff", linewidths=0.8, zorder=3)
                if oom_points:
                    ax.scatter([item[0] for item in oom_points], [item[1] for item in oom_points], s=78, color="#c1121f", marker="x", linewidths=2.0, zorder=4)

    ax.set_title(title, loc="left", fontsize=15, fontweight=800, color="#17212b", pad=16)
    ax.set_xlabel(pretty_label(x_axis), fontsize=11, color="#17212b")
    ax.set_ylabel(pretty_label(y_axis), fontsize=11, color="#17212b")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: format_metric_value(value, y_axis)))
    if category_labels is not None:
        ax.set_xticks(range(len(category_labels)))
        ax.set_xticklabels(category_labels, rotation=25, ha="right")
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _format_plot_axis_tick(value, x_axis)))
    ax.grid(True, axis="y", color="#dce8f1", linewidth=1.0)
    ax.grid(True, axis="x", color="#edf4f8", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#bfd2df")
    ax.spines["bottom"].set_color("#bfd2df")
    ax.tick_params(colors="#17212b", labelsize=9)
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    if any(row_memory_exceeded(row) for row in plot_rows):
        if plot_kind == "bar":
            legend_handles.append(Patch(facecolor="#ffffff", edgecolor="#c1121f", hatch="///", label="OOM"))
        else:
            legend_handles.append(Line2D([0], [0], marker="x", color="#c1121f", linestyle="None", markersize=8, markeredgewidth=2, label="OOM"))
        legend_labels.append("OOM")
    if legend_handles:
        title_text = pretty_label(color_axis) if color_axis and len(grouped) > 1 else None
        ax.legend(legend_handles, legend_labels, title=title_text, frameon=False, loc="best", fontsize=9, title_fontsize=9)
    fig.tight_layout()
    output = BytesIO()
    fig.savefig(output, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return output.getvalue()


def detail_table_keys(rows: List[Dict[str, Any]], *, include_case_id: bool = False) -> List[str]:
    if not rows:
        return []
    ordered = [key for key in DETAIL_TABLE_COLUMN_ORDER if key in rows[0] and (include_case_id or key != "case_id")]
    ordered_set = set(ordered)
    return ordered + [key for key in rows[0].keys() if key not in ordered_set and (include_case_id or key != "case_id")]


def detail_display_rows(rows: List[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
    display_rows = []
    passthrough_keys = {"case", "case_id", "label", "status", "model_config", "hardware_config", "parallelism", "memory_exceeded"}
    for row in rows:
        display_row = {}
        for key in keys:
            value = row.get(key)
            display_row[key] = value if key in passthrough_keys else format_metric_value(value, key)
        display_rows.append(display_row)
    return display_rows


def _export_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def detail_table_export_payload(detail: Dict[str, Any], fmt: str) -> tuple[str, int]:
    rows = sweep_rows_from_detail(detail)
    keys = detail_table_keys(rows, include_case_id=True)
    export_rows = [{key: _export_cell(row.get(key)) for key in keys} for row in rows]
    if fmt == "json":
        return json.dumps(export_rows, indent=2), len(export_rows)
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=keys)
    writer.writeheader()
    writer.writerows(export_rows)
    return output.getvalue(), len(export_rows)


def download_payload_for_file(path: str | Path, content: str | bytes, mime_type: str) -> Dict[str, Any]:
    filename = Path(path).name
    if isinstance(content, bytes):
        return {"content": base64.b64encode(content).decode("ascii"), "filename": filename, "type": mime_type, "base64": True}
    return {"content": content, "filename": filename, "type": mime_type}


def render_detail(detail: Dict[str, Any] | None, plot_type: str | None = "line"):
    if not detail:
        return dmc.Alert("Choose a run from Run log to inspect details.", color="gray", radius="lg")
    payload = detail_payload(detail)
    optimizer_enabled = bool((detail.get("request_record", {}).get("preview") or detail.get("request", {}).get("preview") or {}).get("optimizer_enabled"))
    if detail["kind"] == "run":
        result = detail.get("result", {})
        if result.get("status") == "failed":
            return dmc.Stack(
                gap="lg",
                children=[
                    dmc.Title(detail["title"], order=2),
                    termination_summary_component(detail),
                    dmc.Alert(result.get("error", "The run failed."), color="red", radius="lg", title="Run failed"),
                ],
            )
        metrics = result.get("metrics", {})
        cards = [
            stat_card("Primary", format_primary_metric(result), "solar:star-bold", "teal"),
            stat_card("Model", config_display_name(payload.get("model_preset_id")), "solar:document-text-bold", "blue"),
            stat_card("Hardware", config_display_name(payload.get("hardware_preset_id")), "solar:cpu-bold", "orange"),
            stat_card("GPUs", format_metric_value(metrics.get("num_gpus", "n/a"), "num_gpus"), "solar:server-bold", "blue"),
        ]
        if "training_time_s" in metrics:
            cards.append(stat_card("Time / Batch", format_metric_value(metrics["training_time_s"], "training_time_s"), "solar:clock-circle-bold", "orange"))
        if "prefill_time_s" in metrics:
            cards.append(stat_card("Prefill", format_metric_value(metrics["prefill_time_s"], "prefill_time_s"), "solar:hourglass-bold", "orange"))
        cards.append(stat_card("Parallelism", parallelism_summary_from_payload(payload), "solar:widget-bold", "teal"))
        metric_rows = detail_metric_rows(metrics)
        table = dash_table.DataTable(
            data=[{"metric": row["metric"], "value": row["value"]} for row in metric_rows],
            columns=[{"name": "Metric", "id": "metric"}, {"name": "Value", "id": "value"}],
            tooltip_header={"metric": DETAIL_COLUMN_HELP["metric"], "value": DETAIL_COLUMN_HELP["value"]},
            tooltip_data=[
                {
                    "metric": {"value": help_for_key(row["metric_key"]), "type": "markdown"},
                    "value": {"value": help_for_key(row["metric_key"]), "type": "markdown"},
                }
                for row in metric_rows
            ],
            tooltip_duration=None,
            style_table={"overflowX": "auto"},
            style_cell=DATA_TABLE_STYLE_CELL,
            style_header=DATA_TABLE_STYLE_HEADER,
            style_filter=DATA_TABLE_STYLE_FILTER,
            style_data_conditional=RUN_MEMORY_STYLE_DATA_CONDITIONAL,
        )
        return dmc.Stack(
            gap="lg",
            children=[
                dmc.Group(
                    justify="space-between",
                    children=[
                        dmc.Title(detail["title"], order=2),
                        dmc.Badge("Parallelism optimized" if optimizer_enabled else "Fixed parallelism", color="blue" if optimizer_enabled else "gray", variant="light", radius="xl"),
                    ],
                ),
                termination_summary_component(detail),
                dmc.SimpleGrid(cols={"base": 1, "sm": 2, "lg": 4}, children=cards),
                dmc.Paper(radius="xl", p="lg", withBorder=True, children=table),
            ],
        )
    cases = detail.get("cases", [])
    if not cases:
        return dmc.Alert("This sweep has no completed case records yet.", color="gray", radius="lg")
    rows = sweep_rows_from_detail(detail)
    x_axis, y_axis, series_axis = detail_axes(detail, rows)
    for row in rows:
        if y_axis not in row:
            row[y_axis] = 0
        if row.get("status") != "completed" and y_axis:
            row[y_axis] = 0
    display_mode = detail.get("_case_display_mode") or "full"
    if display_mode == "full":
        plot_rows = rows
        table_rows = rows
    else:
        plot_rows = sample_rows_evenly(rows, DETAIL_PLOT_POINT_LIMIT)
        table_rows = rows[:DETAIL_TABLE_ROW_LIMIT]
    figure = detail_plot_figure(plot_rows, x_axis, y_axis, series_axis, plot_type)
    ordered_keys = detail_table_keys(rows)
    display_rows = detail_display_rows(table_rows, ordered_keys)
    table = dash_table.DataTable(
        data=[{key: row.get(key) for key in ordered_keys} for row in display_rows],
        columns=[{"name": pretty_label(key), "id": key} for key in ordered_keys],
        tooltip_header={key: help_for_key(key) for key in ordered_keys},
        tooltip_duration=None,
        page_size=12,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell=DATA_TABLE_STYLE_CELL,
        style_header=DATA_TABLE_STYLE_HEADER,
        style_filter=DATA_TABLE_STYLE_FILTER,
        style_data_conditional=SWEEP_MEMORY_STYLE_DATA_CONDITIONAL if "memory_exceeded" in ordered_keys else [],
    )
    total_cases, loaded_cases = detail_case_counts(detail, len(rows))
    detail_note_parts = [f"{total_cases:,} case rows total."]
    if display_mode != "full" and loaded_cases < total_cases:
        metric_label = pretty_label(detail.get("_case_sort_metric") or payload.get("metric") or y_axis)
        detail_note_parts.append(f"Showing top {loaded_cases:,} loaded rows by {metric_label} for a faster view.")
    if len(table_rows) < len(rows):
        detail_note_parts.append(f"Table shows first {len(table_rows):,} loaded rows.")
    if len(plot_rows) < len(rows):
        detail_note_parts.append(f"Plot samples {len(plot_rows):,} loaded rows.")
    if optimizer_enabled:
        detail_note_parts.append("Candidate trials from parallelism search are summarized, not listed individually.")
    detail_note = " ".join(detail_note_parts)
    return dmc.Stack(
        gap="lg",
        children=[
            dmc.Group(
                justify="space-between",
                align="center",
                children=[
                    dmc.Stack(gap=2, children=[dmc.Title(detail["title"], order=2), dmc.Text(detail_note, size="sm", c="dimmed")]),
                    dmc.Group(
                        gap="xs",
                        className="detail-status-badges",
                        children=[
                            dmc.Badge("Parallelism optimized" if optimizer_enabled else "Fixed parallelism", color="blue" if optimizer_enabled else "gray", variant="light", radius="xl"),
                        ],
                    ),
                ],
            ),
            termination_summary_component(detail, rows),
            dmc.Title("Graph", order=3),
            dcc.Graph(figure=figure, config={"displayModeBar": False}, style={"height": "420px"}),
            dmc.Paper(radius="xl", p="lg", withBorder=True, children=table),
        ],
    )


def main() -> None:
    host = os.environ.get("RAPID_WEBUI_HOST", "127.0.0.1")
    port = int(os.environ.get("RAPID_WEBUI_PORT", "8050"))
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
