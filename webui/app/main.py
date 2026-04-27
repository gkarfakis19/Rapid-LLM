from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import dash
import dash_mantine_components as dmc
import plotly.express as px
from dash import ALL, Dash, Input, Output, State, callback, dash_table, dcc, html, no_update
from dash_iconify import DashIconify

from webui.service.core import (
    FIELD_OPTIONS,
    FIELD_TYPES,
    METRIC_LABELS,
    RUN_MANAGER,
    build_form_defaults,
    build_job_title,
    build_launch_preview,
    create_config_copy,
    default_worker_count,
    dimension_label,
    ensure_workspace,
    get_default_metric_for_run_type,
    get_metric_options,
    get_telemetry,
    list_history,
    list_presets,
    load_job_detail,
    render_editable_config_texts,
    rename_config_file,
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


FIELD_LABEL_MAP = {item["value"]: item["label"] for item in FIELD_OPTIONS}
METRIC_KEY_BY_LABEL = {label: key for key, label in METRIC_LABELS.items()}
DISPLAY_LABELS = {
    "achieved_flops": "Achieved System FLOPS",
    "achieved_flops_per_gpu": "Achieved FLOPS / GPU",
    "case": "Case",
    "decode_time_s": "Decode Time",
    "memory_exceeded": "Memory Exceeded",
    "num_gpus": "GPUs",
    "parallelism": "Parallelism",
    "peak_flops_per_gpu": "Peak FLOPS / GPU",
    "peak_system_flops": "Peak System FLOPS",
    "total_flops": "Total FLOPs",
    "total_inference_time_s": "Total Inference Time",
    "ttft_s": "Time To First Token",
}
FLOP_COUNT_KEYS = {"total_flops"}
FLOP_RATE_KEYS = {"achieved_flops", "achieved_flops_per_gpu", "peak_flops_per_gpu", "peak_system_flops"}
TOKEN_RATE_KEYS = {"decode_throughput_tok_s"}
TIME_KEYS = {"training_time_s", "prefill_time_s", "decode_time_s", "total_inference_time_s", "ttft_s"}
RANGE_PREVIEW_LIMIT = 12
MODEL_MODE_OPTIONS = [
    {"value": "LLM", "label": "LLM"},
    {"value": "VIT", "label": "ViT"},
    {"value": "GEMM", "label": "GEMM"},
]
MODEL_MODE_HELP = {
    "LLM": "Transformer language-model path. Uses token sequence length, attention, MLP/MoE, training or inference timing, memory estimates, parallelism, and communication modeling.",
    "VIT": "Vision Transformer execution path. This should be paired with a ViT model_type and a model_param.vision block.",
    "GEMM": "Standalone matrix multiply sanity path. Uses M, K, N and GEMM sharding fields instead of transformer layer fields; many LLM-specific controls do not apply.",
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
    {"value": "mxfp4", "label": "mxfp4 (4.25)"},
    {"value": "int4", "label": "int4 (4)"},
    {"value": "fp8", "label": "fp8 (8)"},
    {"value": "fp16", "label": "fp16 (16)"},
    {"value": "bf16", "label": "bf16 (16)"},
    {"value": "fp32", "label": "fp32 (32)"},
]
HELP_TEXT = {
    "app_title": "RAPID-LLM local workbench for editing supported YAML fields, previewing launch size, running jobs, and loading saved details.",
    "app_flow": "Basic flow: configure in Launch, start the run, open Run log, then click Details. Hover controls, metrics, and table columns for explanations.",
    "telemetry_ram": "Currently available host memory reported by psutil.",
    "telemetry_cpu": "Current host CPU utilization reported by psutil.",
    "telemetry_job": "Top-level Web UI job state.",
    "models_to_run": "Select model YAML files to include as run cases. Use the model file tabs below to choose which selected model is loaded into the editor.",
    "hardware_to_run": "Select hardware YAML files to include as run cases. Use the hardware file tabs below to choose which selected hardware file is loaded into the editor.",
    "editor_tabs": "The highlighted model and hardware tabs are the YAML files currently loaded into the editor. Switch tabs before changing fields to edit a different selected file.",
    "model_preset": "Selected model YAML currently loaded into Basic and Advanced model controls.",
    "hardware_preset": "Selected hardware YAML currently loaded into Basic and Advanced hardware controls.",
    "config_file_name": "Filename to create or rename. The UI writes YAML files under webui/workspace/configs.",
    "new_config": "Create a new editable YAML by copying the active model or hardware config.",
    "rename_config": "Rename the active editable YAML file and keep it selected.",
    "worker_count": "Number of local worker processes used inside a sweep. More workers consume more CPU and memory.",
    "timeout": "Maximum wall-clock time allowed for each simulator invocation.",
    "run_type": "Choose training or inference; this controls which fields and result metrics are active.",
    "model_mode": "Select the execution family written to model_param.mode. This chooses the broad simulator path: LLM, VIT, or GEMM.",
    "model_type": "Select the architecture family written to model_param.model_type. This controls modeling details such as MLP style, ViT handling, and GLM/DeepSeek special cases.",
    "seq_len": "Number of tokens processed in the input context.",
    "decode_len": "Number of generated tokens to model for inference runs.",
    "batch_size": "Global batch size across all participating devices.",
    "grad_accum": "Number of microbatch accumulation steps before optimizer update.",
    "optimize_parallelism": "Search TP, CP, PP, DP, and EP combinations for each Total GPUs target.",
    "optimizer_preset": "Controls how many parallelism candidates the search evaluates.",
    "total_gpus": "Target device count. With optimization on, this is the searched GPU count; with optimization off, it scales DP for training or replicas for inference when divisible by fixed TP*CP*PP*EP.",
    "tp": "Tensor parallel shards each layer's matrix work across devices.",
    "cp": "Context parallel shards long sequences across devices.",
    "pp": "Pipeline parallel splits model layers into pipeline stages.",
    "dp": "Data parallel replicates the model across training batches.",
    "ep": "Expert parallel distributes MoE experts across devices.",
    "replica_count": "Inference replica count used to scale throughput.",
    "hbm_gb": "Usable high-bandwidth memory capacity per GPU.",
    "gpu_clock": "GPU core clock used for peak compute estimates.",
    "memory_bw": "Memory bandwidth used by the analytical memory model.",
    "use_astrasim": "Run the simulator through AstraSim using hierarchical mode. When off, use the analytical backend.",
    "compute_derate": "Multiplier for sustained compute efficiency.",
    "memory_derate": "Multiplier for sustained memory bandwidth efficiency.",
    "network_derate": "Default network utilization multiplier applied to dimensions without an explicit value.",
    "network_topology": "Collective topology model for this network dimension.",
    "network_bandwidth": "Per-link or dimension bandwidth, such as 100 GB.",
    "network_util": "Utilization multiplier for this network dimension.",
    "full_recomp": "Enable full activation recomputation to trade compute for memory.",
    "zero_stage": "ZeRO sharding stage for optimizer, gradient, and parameter state.",
    "tensor_format": "Numeric tensor format used for model compute.",
    "tied_embeddings": "Share token embedding and output projection weights when supported.",
    "hidden_dim": "Transformer hidden dimension.",
    "intermediate_size": "Dense MLP intermediate dimension.",
    "num_layers": "Number of transformer layers.",
    "vocab_size": "Vocabulary size used by embeddings and output projection.",
    "attention_type": "Attention implementation family.",
    "num_heads": "Number of attention heads.",
    "use_flash": "Enable FlashAttention for supported training or prefill paths.",
    "attention_tile": "Tile size used by FlashAttention estimates.",
    "num_experts": "Number of MoE experts.",
    "top_k": "Number of experts selected per token.",
    "moe_intermediate_size": "MoE expert MLP intermediate dimension.",
    "expert_imbalance": "Worst-case expert load multiplier for imbalanced routing.",
    "metric": "Metric used for ranking sweep cases and picking the best result.",
    "x_axis": "Sweep field used as the plot x-axis in Details.",
    "series_axis": "Optional sweep field used to group plotted cases.",
    "sweep_dimensions": "Sweep workload size or hardware scaling. Use Total GPUs for GPU scaling; raw TP/CP/PP/DP/EP are edited once or found by Optimize parallelism, not swept independently.",
    "run_launch": "Start the live launch plan currently shown on the right.",
    "cancel_job": "Request cancellation for the currently active top-level job.",
    "load_details": "Open this saved job in the Details screen.",
}
METRIC_HELP = {
    "training_time_s": "Modeled time for one training batch.",
    "approx_mfu": "Achieved system FLOPS divided by modeled system peak FLOPS. Values above 100% indicate an optimistic timing path or a FLOP accounting mismatch.",
    "prefill_time_s": "Modeled time to process the input prompt before token generation.",
    "decode_time_s": "Modeled time spent generating decode tokens.",
    "total_inference_time_s": "Prefill time plus modeled decode time.",
    "ttft_s": "Time to first token, including prefill.",
    "decode_throughput_tok_s": "Midpoint decode token throughput multiplied by batch size and replica count.",
    "num_gpus": "Total devices implied by TP, CP, PP, DP/replicas, and EP.",
    "total_flops": "Estimated total training work for the modeled batch. This is a count, not a rate.",
    "achieved_flops": "System-wide estimated FLOP/s: total modeled FLOPs divided by simulated runtime.",
    "achieved_flops_per_gpu": "System achieved FLOP/s divided by the number of modeled GPUs.",
    "peak_flops_per_gpu": "Per-GPU theoretical tensor-core peak from the selected hardware YAML and compute derate.",
    "peak_system_flops": "Per-GPU theoretical peak multiplied by modeled GPU count.",
    "memory_exceeded": "No means estimated peak memory fits; otherwise this is the estimated amount over device memory.",
}
DETAIL_COLUMN_HELP = {
    "case": "Case identifier plus the model config used for this row.",
    "case_id": "Stable case identifier generated by the launcher.",
    "label": "Human-readable case label derived from config and sweep choices.",
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
    return with_tip(dmc.Badge(label, radius="xl", color="teal" if value == "LLM" else "blue" if value == "VIT" else "orange", variant="light"), MODEL_MODE_HELP[value])


def model_type_badge(value: str) -> dmc.Tooltip:
    return with_tip(dmc.Badge(value, radius="xl", color="blue" if value in {"gpt", "llama", "deepseek_v3", "glm4_moe"} else "teal", variant="light"), MODEL_ARCH_TYPE_HELP[value])


def flow_help() -> dmc.Group:
    return dmc.Group(
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
            dmc.Text("Hover any control for details.", size="xs", c="dimmed", ml="xs"),
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


def config_display_name(config_id: Any) -> str:
    if not config_id:
        return "base config"
    return Path(str(config_id)).stem.replace("_", " ")


def memory_exceeded_display(metrics: Dict[str, Any]) -> str:
    violation_gb = float(metrics.get("memory_violation_gb") or 0.0)
    exceeded = bool(metrics.get("memory_exceeded")) or violation_gb > 0
    if not exceeded:
        return "No"
    if violation_gb > 0:
        return format_metric_value(violation_gb, "memory_violation_gb")
    return ">0 GB"


def detail_metric_rows(metrics: Dict[str, Any]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    inserted_memory = False
    for key, value in metrics.items():
        if key == "memory_violation_gb":
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


dash._dash_renderer._set_react_version("18.2.0")
ensure_workspace()
MODEL_PRESETS = list_presets("models")
HW_PRESETS = list_presets("hardware")
CPU_CORES = os.cpu_count() or 1
MODEL_LABELS = {item["id"]: item["label"] for item in MODEL_PRESETS}
HW_LABELS = {item["id"]: item["label"] for item in HW_PRESETS}


def preset_records(kind: str) -> List[Dict[str, Any]]:
    return list_presets(kind)


def preset_options(kind: str) -> List[Dict[str, str]]:
    return [{"value": item["id"], "label": item["label"]} for item in preset_records(kind)]


def preset_labels(kind: str) -> Dict[str, str]:
    return {item["id"]: item["label"] for item in preset_records(kind)}


def _preferred_preset_id(records: List[Dict[str, Any]], preferred_id: str) -> str:
    for item in records:
        if item["id"] == preferred_id:
            return item["id"]
    return records[0]["id"]


DEFAULT_MODEL_ID = _preferred_preset_id(MODEL_PRESETS, "Llama2-7B.yaml")
DEFAULT_HW_ID = _preferred_preset_id(HW_PRESETS, "H100_SXM5_80GB_base.yaml")
DEFAULTS = build_form_defaults(DEFAULT_MODEL_ID, DEFAULT_HW_ID)
DEFAULT_METRIC = get_default_metric_for_run_type(DEFAULTS["run_type"])

app = Dash(__name__, title="RAPID-LLM Workbench", assets_folder=str(Path(__file__).parent / "assets"), suppress_callback_exceptions=True)
server = app.server


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
        "optimize_parallelism": False,
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


def selected_config_values(values: List[str] | None, fallback: str) -> List[str]:
    selected = [item for item in (values or []) if item]
    return selected or [fallback]


def selected_active_value(values: List[str] | None, fallback: str, current: str | None) -> str:
    selected = selected_config_values(values, fallback)
    return current if current in selected else selected[0]


def editor_tabs_children(selected_ids: List[str] | None, fallback: str, labels: Dict[str, str], icon: str) -> List[Any]:
    values = selected_config_values(selected_ids, fallback)
    return [
        dmc.TabsList(
            className="config-editor-tab-list",
            children=[
                dmc.TabsTab(
                    labels.get(value, Path(value).stem),
                    value=value,
                    leftSection=DashIconify(icon=icon, width=15),
                    className="config-editor-tab",
                    attributes={"title": value},
                )
                for value in values
            ],
        )
    ]


def config_dimensions_from_selection(model_ids: List[str] | None, hardware_ids: List[str] | None, primary_model_id: str, primary_hardware_id: str) -> List[Dict[str, Any]]:
    dimensions: List[Dict[str, Any]] = []
    selected_models = selected_config_values(model_ids, primary_model_id)
    selected_hardware = selected_config_values(hardware_ids, primary_hardware_id)
    if len(selected_models) > 1:
        dimensions.append({"field_key": "model_config", "mode": "values", "config_values": selected_models})
    if len(selected_hardware) > 1:
        dimensions.append({"field_key": "hardware_config", "mode": "values", "config_values": selected_hardware})
    return dimensions


def render_preview_summary(preview: Dict[str, Any], metric: str) -> dmc.Stack:
    if not preview.get("ok"):
        return dmc.Stack(children=[dmc.Alert(item, color="red", radius="lg") for item in preview.get("errors", [])])
    telemetry = get_telemetry()
    return dmc.Stack(
        gap="md",
        children=[
            dmc.SimpleGrid(
                cols={"base": 1, "sm": 2, "lg": 4},
                spacing="sm",
                children=[
                    stat_card("Top-level cases", format_metric_value(preview["top_level_case_count"], "case_count"), "solar:box-bold", "teal"),
                    stat_card("Simulator invocations", format_metric_value(preview["total_invocations"], "invocation_count"), "solar:play-bold", "blue"),
                    stat_card("Worst-case wall clock", humanize_seconds(preview["worst_case_wall_clock_s"]), "solar:clock-circle-bold", "orange"),
                    stat_card("Available RAM", format_metric_value(telemetry["available_ram_gb"], "available_ram_gb"), "solar:memory-bold", "grape"),
                ],
            ),
            dmc.Group(
                gap="sm",
                children=[
                    dmc.Badge(f"Metric: {METRIC_LABELS.get(metric, metric)}", radius="xl", color="teal", variant="light"),
                    dmc.Badge(f"Workers: {format_metric_value(preview['worker_count'], 'worker_count')}", radius="xl", color="blue", variant="light"),
                    dmc.Badge(f"Timeout: {format_metric_value(preview['timeout_seconds'], 'timeout_seconds')}", radius="xl", color="orange", variant="light"),
                ],
            ),
            dmc.Text("Worst-case wall clock assumes every invocation consumes the full timeout. Real runs are usually faster.", size="sm", c="dimmed"),
            dmc.Stack(children=[dmc.Alert(w, color="yellow", radius="lg") for w in preview.get("warnings", [])]) if preview.get("warnings") else html.Div(),
        ],
    )


def launch_button_label(preview: Dict[str, Any] | None) -> str:
    count = int((preview or {}).get("total_invocations") or 0)
    noun = "run" if count == 1 else "runs"
    return f"Launch {count} {noun}"


def dim_card(index: int) -> dmc.Paper:
    return dmc.Paper(
        radius="xl",
        p="md",
        withBorder=True,
        className="dimension-card",
        children=dmc.Stack(
            gap="sm",
            children=[
                dmc.Group(justify="space-between", children=[dmc.Text(f"Dimension {index}", fw=700, size="sm"), dmc.Badge("Optional", variant="light", color="gray")]),
                with_tip(dmc.Select(id=f"dim-{index}-field", label="Field", placeholder="Select a sweep field", data=FIELD_OPTIONS, clearable=True), "Select a workload or hardware-scaling field to vary. Use Total GPUs for GPU scaling; individual parallelism axes are not sweep fields."),
                html.Div(
                    id=f"dim-{index}-mode-wrap",
                    style={"display": "none"},
                    children=with_tip(dmc.SegmentedControl(id=f"dim-{index}-mode", fullWidth=True, data=[{"label": "Values", "value": "values"}, {"label": "Range", "value": "range"}], value="values"), "Values uses a comma-separated list; Range uses start, end, and step size."),
                ),
                html.Div(id=f"dim-{index}-values-wrap", style={"display": "none"}, children=with_tip(dmc.TextInput(id=f"dim-{index}-list", label="Values", placeholder="Example: 8192, 16384"), "Comma-separated values to run for the selected sweep field.")),
                html.Div(id=f"dim-{index}-configs-wrap", style={"display": "none"}, children=with_tip(dmc.MultiSelect(id=f"dim-{index}-configs", label="Preset values", data=[], placeholder="Pick config files"), "Configuration file values for this sweep dimension.")),
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
                                    with_tip(dmc.NumberInput(id=f"dim-{index}-start", label="Start", allowDecimal=True), "First value in the numeric sweep range."),
                                    with_tip(dmc.NumberInput(id=f"dim-{index}-end", label="End", allowDecimal=True), "Last value included in the numeric sweep range."),
                                    with_tip(dmc.NumberInput(id=f"dim-{index}-step_or_points", label="Step size", allowDecimal=True), "Increment between adjacent values in the numeric sweep range."),
                                ],
                            ),
                            html.Div(id=f"dim-{index}-range-preview", className="range-preview", children="Preview: enter start, end, and step size."),
                        ],
                    ),
                ),
            ],
        ),
    )


def network_editor(defaults: List[Dict[str, Any]]) -> List[dmc.Paper]:
    rows: List[dmc.Paper] = []
    for idx, row in enumerate(defaults):
        rows.append(
            dmc.Paper(
                radius="lg",
                p="sm",
                withBorder=True,
                className="network-row",
                children=dmc.Stack(
                    gap="xs",
                    children=[
                        dmc.Text(row["label"], fw=700, size="sm"),
                        dmc.SimpleGrid(
                            cols={"base": 1, "sm": 3},
                            spacing="sm",
                            children=[
                                with_tip(dmc.Select(id={"type": "net-topology", "index": idx}, label="Topology", value=row["topology_type"], data=[{"value": "Ring", "label": "Ring"}, {"value": "FC", "label": "Fully Connected"}, {"value": "SuperPOD", "label": "SuperPOD"}, {"value": "Torus2D", "label": "Torus2D"}]), HELP_TEXT["network_topology"]),
                                with_tip(dmc.TextInput(id={"type": "net-bandwidth", "index": idx}, label="Bandwidth", value=str(row["bandwidth"])), HELP_TEXT["network_bandwidth"]),
                                with_tip(dmc.NumberInput(id={"type": "net-util", "index": idx}, label="Utilization", min=0, max=1, step=0.01, decimalScale=3, value=float(row["util"])), HELP_TEXT["network_util"]),
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
                        gap="md",
                        children=[
                            with_tip(html.Div(className="brand-orb"), HELP_TEXT["app_title"]),
                            dmc.Stack(
                                gap=2,
                                children=[
                                    with_tip(dmc.Title("RAPID-LLM Workbench", order=2, c="#0055A6"), HELP_TEXT["app_title"]),
                                    with_tip(flow_help(), HELP_TEXT["app_flow"]),
                                ],
                            ),
                        ],
                    ),
                    dmc.Group(
                        gap="sm",
                        children=[
                            with_tip(dmc.Badge(id="telemetry-ram", size="lg", radius="xl", color="teal", variant="light"), HELP_TEXT["telemetry_ram"]),
                            with_tip(dmc.Badge(id="telemetry-cpu", size="lg", radius="xl", color="orange", variant="light"), HELP_TEXT["telemetry_cpu"]),
                            with_tip(dmc.Badge(id="telemetry-job", size="lg", radius="xl", color="blue", variant="light"), HELP_TEXT["telemetry_job"]),
                        ],
                    ),
                ],
            ),
        ),
    )


def create_layout() -> dmc.MantineProvider:
    metric_options = get_metric_options(DEFAULTS["run_type"])
    return dmc.MantineProvider(
        theme={"primaryColor": "blue", "fontFamily": "Arial, Calibri, Segoe UI, sans-serif", "defaultRadius": "sm"},
        children=html.Div(
            className="app-shell",
            children=[
                build_header(),
                dcc.Interval(id="poller", interval=1500, n_intervals=0),
                dcc.Store(id="preview-store"),
                dcc.Store(id="selected-detail-store"),
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
                                    children=dmc.Group(
                                        justify="space-between",
                                        align="center",
                                        children=[
                                            html.Div(id="detail-modal-title"),
                                            dmc.Group(
                                                gap="sm",
                                                children=[
                                                    html.Div(
                                                        id="detail-plot-toolbar",
                                                        style={"display": "none"},
                                                        children=dmc.SegmentedControl(
                                                            id="detail-plot-type",
                                                            value="line",
                                                            data=[
                                                                {"label": "Normal plot", "value": "line"},
                                                                {"label": "Scatter", "value": "scatter"},
                                                                {"label": "Bar chart", "value": "bar"},
                                                            ],
                                                        ),
                                                    ),
                                                    dmc.ActionIcon(
                                                        DashIconify(icon="solar:close-circle-bold", width=22),
                                                        id="detail-close-button",
                                                        variant="subtle",
                                                        radius="xl",
                                                        size="lg",
                                                        **{"aria-label": "Close details"},
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
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
                                dmc.TabsPanel(value="builder", children=builder_panel(metric_options)),
                                dmc.TabsPanel(value="history", children=html.Div(id="history-panel")),
                            ],
                        )
                    ],
                ),
            ],
        ),
    )


def builder_panel(metric_options: List[Dict[str, str]]) -> dmc.Grid:
    return dmc.Grid(
        className="builder-grid",
        gutter="lg",
        children=[
            dmc.GridCol(span={"base": 12, "xl": 5}, children=left_column(metric_options)),
            dmc.GridCol(span={"base": 12, "xl": 7}, children=right_column()),
        ],
    )


def left_column(metric_options: List[Dict[str, str]]) -> dmc.Stack:
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
                    "Pick editable YAML files, adjust supported options, launch the job, then open Details from the run log.",
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
            run_setup_card(),
            basic_options_card(),
            html.Div(id="sweep-dimensions-section", children=dimensions_card()),
            launch_controls_card(metric_options),
            advanced_options_card(),
        ],
    )


def right_column() -> dmc.Stack:
    initial_preview = build_launch_preview(_default_payload())
    preview_card = dmc.Paper(
        radius="xl",
        p="lg",
        withBorder=True,
        className="preview-card",
        children=dmc.Stack(
            gap="md",
            children=[
                dmc.Group(
                    justify="space-between",
                    children=[
                        dmc.Title("Launch Plan", order=3),
                        dmc.Badge("Updates as settings change", color="blue", variant="light"),
                    ],
                ),
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
                html.Div(id="active-job-panel"),
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
                            children=[dmc.Title("Workflow", order=3), dmc.Text("Launch -> Run log -> Details", size="sm", c="dimmed")],
                ),
                dmc.SimpleGrid(
                    cols={"base": 1, "sm": 2},
                    children=[
                        dmc.Alert(
                            color="teal",
                            radius="lg",
                            title="Choose cases",
                            children="Select one hardware target with any number of models, or compare hardware targets for the selected model set.",
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


def run_setup_card() -> dmc.Paper:
    return dmc.Paper(
        radius="xl",
        p="lg",
        withBorder=True,
        children=dmc.Stack(
            gap="md",
            children=[
                dmc.Group(justify="space-between", children=[dmc.Title("Launch Setup", order=3), dmc.Badge("workspace/configs", color="teal", variant="light")]),
                with_tip(dmc.MultiSelect(id="model-run-configs", label="Models to run", value=[DEFAULT_MODEL_ID], data=preset_options("models"), clearable=False), HELP_TEXT["models_to_run"]),
                with_tip(dmc.MultiSelect(id="hardware-run-configs", label="Hardware to run", value=[DEFAULT_HW_ID], data=preset_options("hardware"), clearable=False), HELP_TEXT["hardware_to_run"]),
                editor_switchboard(),
                html.Div(
                    style={"display": "none"},
                    children=[
                        dmc.Select(id="model-preset", value=DEFAULT_MODEL_ID, data=preset_options("models")),
                        dmc.Select(id="hardware-preset", value=DEFAULT_HW_ID, data=preset_options("hardware")),
                    ],
                ),
                with_tip(dmc.SegmentedControl(id="run-mode", fullWidth=True, value="sweep", data=[{"label": "Sweep", "value": "sweep"}, {"label": "Single Launch", "value": "single"}]), "Choose whether to launch one edited config or expand sweep dimensions into multiple cases."),
                dmc.SimpleGrid(
                    cols={"base": 1, "sm": 2},
                    spacing="sm",
                    children=[
                        with_tip(dmc.NumberInput(id="worker-count", label="Workers", min=1, max=CPU_CORES, value=default_worker_count()), HELP_TEXT["worker_count"]),
                        with_tip(dmc.NumberInput(id="timeout-seconds", label="Timeout / candidate (s)", min=1, value=180), HELP_TEXT["timeout"]),
                    ],
                ),
                dmc.Text(f"CPU cores detected: {CPU_CORES}. Worker default: {default_worker_count()}.", size="xs", c="dimmed"),
                dmc.Text(id="config-sync-status", size="xs", c="dimmed"),
            ],
        ),
    )


def editor_switchboard() -> html.Div:
    return html.Div(
        className="config-editor-switchboard",
        children=[
            dmc.Group(
                justify="space-between",
                align="center",
                children=[
                    with_tip(dmc.Text("Selected file editor", fw=800), HELP_TEXT["editor_tabs"]),
                    dmc.Badge("active YAML tabs", radius="xl", color="teal", variant="light"),
                ],
            ),
            with_tip(
                dmc.Text("The highlighted model and hardware tabs are loaded below. Switch tabs to edit another selected YAML file.", size="sm", c="dimmed"),
                HELP_TEXT["editor_tabs"],
            ),
            dmc.Stack(
                gap="xs",
                children=[
                    dmc.Text("Model files", size="xs", fw=800, tt="uppercase", c="dimmed"),
                    with_tip(
                        dmc.Tabs(
                            id="model-editor-tabs",
                            value=DEFAULT_MODEL_ID,
                            className="config-editor-tabs",
                            children=editor_tabs_children([DEFAULT_MODEL_ID], DEFAULT_MODEL_ID, MODEL_LABELS, "solar:document-text-bold"),
                        ),
                        HELP_TEXT["model_preset"],
                    ),
                    dmc.Text("Hardware files", size="xs", fw=800, tt="uppercase", c="dimmed"),
                    with_tip(
                        dmc.Tabs(
                            id="hardware-editor-tabs",
                            value=DEFAULT_HW_ID,
                            className="config-editor-tabs",
                            children=editor_tabs_children([DEFAULT_HW_ID], DEFAULT_HW_ID, HW_LABELS, "solar:cpu-bold"),
                        ),
                        HELP_TEXT["hardware_preset"],
                    ),
                ],
            ),
            dmc.Divider(label="File actions", labelPosition="center"),
            dmc.SimpleGrid(
                cols={"base": 1, "sm": 3},
                spacing="sm",
                children=[
                    with_tip(
                        dmc.SegmentedControl(
                            id="config-action-kind",
                            value="models",
                            data=[{"label": "Model", "value": "models"}, {"label": "Hardware", "value": "hardware"}],
                            fullWidth=True,
                        ),
                        "Choose whether the action targets the active model tab or active hardware tab.",
                    ),
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
        ],
    )


def parallelism_axis_input(component_id: str, label: str, value: int, help_text: str) -> html.Div:
    return html.Div(
        id=f"{component_id}-wrap",
        className="parallelism-axis-field",
        children=with_tip(dmc.NumberInput(id=component_id, label=label, min=1, value=value), help_text),
    )


def basic_options_card() -> dmc.Paper:
    return dmc.Paper(
        radius="xl",
        p="lg",
        withBorder=True,
        children=dmc.Stack(
            gap="lg",
            children=[
                dmc.Title("Basic Options", order=3),
                dmc.Accordion(
                    multiple=True,
                    value=["model", "hardware"],
                    children=[
                        dmc.AccordionItem(
                            value="model",
                            children=[
                                dmc.AccordionControl("Model"),
                                dmc.AccordionPanel(
                                    dmc.Stack(
                                        gap="sm",
                                        children=[
                                            with_tip(dmc.Select(id="simple-run-type", label="Run type", value=DEFAULTS["run_type"], data=[{"value": "training", "label": "Training"}, {"value": "inference", "label": "Inference"}]), HELP_TEXT["run_type"]),
                                            dmc.SimpleGrid(
                                                cols={"base": 1, "sm": 2},
                                                spacing="sm",
                                                children=[
                                                    with_tip(dmc.NumberInput(id="simple-seq-len", label="Sequence length", min=1, value=DEFAULTS["simple"]["seq_len"]), HELP_TEXT["seq_len"]),
                                                    with_tip(dmc.NumberInput(id="simple-decode-len", label="Decode length", min=0, value=DEFAULTS["simple"]["decode_len"]), HELP_TEXT["decode_len"]),
                                                    with_tip(dmc.NumberInput(id="simple-batch-size", label="Batch size", min=1, value=DEFAULTS["simple"]["batch_size"]), HELP_TEXT["batch_size"]),
                                                    with_tip(dmc.NumberInput(id="simple-grad-accum", label="Grad accumulation", min=1, value=DEFAULTS["simple"]["grad_accum"]), HELP_TEXT["grad_accum"]),
                                                ],
                                            ),
                                        ],
                                    )
                                ),
                            ],
                        ),
                        dmc.AccordionItem(
                            value="hardware",
                            children=[
                                dmc.AccordionControl("Hardware"),
                                dmc.AccordionPanel(
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
                                                            with_tip(dmc.Switch(id="optimize-switch", size="md", checked=False, label="Optimize parallelism"), HELP_TEXT["optimize_parallelism"]),
                                                            dmc.Text("WARNING: This may increase runtime dramatically.", c="red", size="xs", fw=800),
                                                        ],
                                                    ),
                                                    with_tip(dmc.Select(id="optimizer-preset", label="Parallelism search", value="Fast", data=[{"value": "Fast", "label": "Fast candidate set"}, {"value": "Exhaustive", "label": "Full candidate set"}]), HELP_TEXT["optimizer_preset"]),
                                                ],
                                            ),
                                            dmc.SimpleGrid(
                                                className="parallelism-grid",
                                                cols={"base": 1, "sm": 2},
                                                spacing="sm",
                                                children=[
                                                    with_tip(dmc.NumberInput(id="simple-total-gpus", label="Total GPUs", min=1, value=DEFAULTS["simple"]["total_gpus"]), HELP_TEXT["total_gpus"]),
                                                    parallelism_axis_input("simple-tp", "TP", DEFAULTS["simple"]["tp"], HELP_TEXT["tp"]),
                                                    parallelism_axis_input("simple-cp", "CP", DEFAULTS["simple"]["cp"], HELP_TEXT["cp"]),
                                                    parallelism_axis_input("simple-pp", "PP", DEFAULTS["simple"]["pp"], HELP_TEXT["pp"]),
                                                    parallelism_axis_input("simple-dp", "DP", DEFAULTS["simple"]["dp"], HELP_TEXT["dp"]),
                                                    parallelism_axis_input("simple-ep", "EP", DEFAULTS["simple"]["ep"], HELP_TEXT["ep"]),
                                                    parallelism_axis_input("simple-replica-count", "Replica count", DEFAULTS["simple"]["replica_count"], HELP_TEXT["replica_count"]),
                                                ],
                                            ),
                                            dmc.Divider(label="Derates", labelPosition="center"),
                                            dmc.SimpleGrid(
                                                cols={"base": 1, "sm": 3},
                                                spacing="sm",
                                                children=[
                                                    with_tip(dmc.NumberInput(id="simple-compute-derate", label="Compute derate", min=0.0, max=1.0, step=0.01, decimalScale=3, value=DEFAULTS["simple"]["compute_derate"]), HELP_TEXT["compute_derate"]),
                                                    with_tip(dmc.NumberInput(id="simple-memory-derate", label="Memory derate", min=0.0, max=1.0, step=0.01, decimalScale=3, value=DEFAULTS["simple"]["memory_derate"]), HELP_TEXT["memory_derate"]),
                                                    with_tip(dmc.NumberInput(id="simple-network-derate", label="Network derate", min=0.0, max=1.0, step=0.01, decimalScale=3, value=DEFAULTS["simple"]["network_derate"]), HELP_TEXT["network_derate"]),
                                                ],
                                            ),
                                            dmc.Divider(label="Hardware limits", labelPosition="center"),
                                            dmc.SimpleGrid(
                                                cols={"base": 1, "sm": 3},
                                                spacing="sm",
                                                children=[
                                                    with_tip(dmc.NumberInput(id="simple-hbm-gb", label="HBM capacity (GB)", min=1, value=DEFAULTS["simple"]["hbm_gb"]), HELP_TEXT["hbm_gb"]),
                                                    with_tip(dmc.NumberInput(id="simple-gpu-clock", label="GPU clock (GHz)", min=0.1, step=0.01, decimalScale=3, value=DEFAULTS["simple"]["gpu_clock_ghz"]), HELP_TEXT["gpu_clock"]),
                                                    with_tip(dmc.NumberInput(id="simple-memory-bw", label="Memory BW (GB/s)", min=1, step=1, decimalScale=2, value=DEFAULTS["simple"]["memory_bw_gbs"]), HELP_TEXT["memory_bw"]),
                                                    with_tip(dmc.Switch(id="simple-use-astrasim", checked=DEFAULTS["simple"]["use_astrasim"], label="Use AstraSim"), HELP_TEXT["use_astrasim"]),
                                                ],
                                            ),
                                            dmc.Divider(label="Network dimensions", labelPosition="center"),
                                            html.Div(id="network-dimensions-editor", children=network_editor(DEFAULTS["network_dimensions"])),
                                        ],
                                    )
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    )


def dimensions_card() -> dmc.Paper:
    return dmc.Paper(
        radius="xl",
        p="lg",
        withBorder=True,
        children=dmc.Stack(
            gap="md",
            children=[
                dmc.Title("Sweep Dimensions", order=3),
                with_tip(
                    dmc.Text("Sweep workload size and hardware scaling. Select multiple model or hardware files in Launch Setup for config comparisons.", size="sm", c="dimmed"),
                    HELP_TEXT["sweep_dimensions"],
                ),
                dim_card(1),
                dim_card(2),
                dim_card(3),
                dmc.SimpleGrid(
                    cols={"base": 1, "sm": 2},
                    spacing="sm",
                    children=[
                        with_tip(dmc.Select(id="x-axis-select", label="X-axis", data=[]), HELP_TEXT["x_axis"]),
                        with_tip(dmc.Select(id="series-select", label="Grouping", data=[], clearable=True), HELP_TEXT["series_axis"]),
                    ],
                ),
            ],
        ),
    )


def launch_controls_card(metric_options: List[Dict[str, str]]) -> dmc.Paper:
    return dmc.Paper(
        radius="xl",
        p="lg",
        withBorder=True,
        children=dmc.Stack(
            gap="md",
            children=[
                dmc.Title("Launch", order=3),
                with_tip(dmc.Select(id="metric-select", label="Metric", value=get_default_metric_for_run_type(DEFAULTS["run_type"]), data=metric_options), HELP_TEXT["metric"]),
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


def advanced_options_card() -> dmc.Paper:
    return dmc.Paper(
        radius="xl",
        p="lg",
        withBorder=True,
        children=dmc.Stack(
            gap="md",
            children=[
                dmc.Title("Advanced Options", order=3),
                dmc.Accordion(
                    multiple=True,
                    value=["model", "hardware"],
                    children=[
                        dmc.AccordionItem(
                            value="model",
                            children=[
                                dmc.AccordionControl("Model"),
                                dmc.AccordionPanel(
                                    dmc.Stack(
                                        gap="sm",
                                        children=[
                                            dmc.SimpleGrid(
                                                cols={"base": 1, "sm": 2},
                                                spacing="sm",
                                                children=[
                                                    with_tip(dmc.Select(id="adv-model-type", label="Model type", value=DEFAULTS["advanced"]["model_type"], data=MODEL_ARCH_TYPE_OPTIONS), HELP_TEXT["model_type"]),
                                                    with_tip(dmc.Select(id="adv-model-mode", label="Execution family", value=DEFAULTS["advanced"]["model_mode"], data=MODEL_MODE_OPTIONS), HELP_TEXT["model_mode"]),
                                                    with_tip(dmc.Switch(id="adv-tied-embeddings", checked=DEFAULTS["advanced"]["tied_embeddings"], label="Tied embeddings"), HELP_TEXT["tied_embeddings"]),
                                                ],
                                            ),
                                            dmc.Group(gap="xs", children=[dmc.Text("Model type guide", size="xs", c="dimmed"), *[model_type_badge(item["value"]) for item in MODEL_ARCH_TYPE_OPTIONS]]),
                                            dmc.Group(gap="xs", children=[dmc.Text("Execution family guide", size="xs", c="dimmed"), mode_badge("LLM"), mode_badge("VIT"), mode_badge("GEMM")]),
                                            dmc.SimpleGrid(
                                                cols={"base": 1, "sm": 2},
                                                spacing="sm",
                                                children=[
                                                    with_tip(dmc.NumberInput(id="adv-hidden-dim", label="Hidden dim", min=1, value=DEFAULTS["advanced"]["hidden_dim"]), HELP_TEXT["hidden_dim"]),
                                                    with_tip(dmc.NumberInput(id="adv-intermediate-size", label="Intermediate size", min=1, value=DEFAULTS["advanced"]["intermediate_size"]), HELP_TEXT["intermediate_size"]),
                                                    with_tip(dmc.NumberInput(id="adv-num-layers", label="Layers", min=1, value=DEFAULTS["advanced"]["num_layers"]), HELP_TEXT["num_layers"]),
                                                    with_tip(dmc.NumberInput(id="adv-vocab-size", label="Vocab size", min=1, value=DEFAULTS["advanced"]["vocab_size"]), HELP_TEXT["vocab_size"]),
                                                    with_tip(dmc.Select(id="adv-attention-type", label="Attention type", value=DEFAULTS["advanced"]["attention_type"], data=[{"value": "mha", "label": "MHA"}, {"value": "gqa", "label": "GQA"}, {"value": "mla", "label": "MLA"}]), HELP_TEXT["attention_type"]),
                                                    with_tip(dmc.NumberInput(id="adv-num-heads", label="Attention heads", min=1, value=DEFAULTS["advanced"]["num_heads"]), HELP_TEXT["num_heads"]),
                                                    with_tip(dmc.Switch(id="adv-use-flash", checked=DEFAULTS["advanced"]["use_flashattention"], label="Use FlashAttention"), HELP_TEXT["use_flash"]),
                                                    with_tip(dmc.NumberInput(id="adv-attn-tile", label="Attention tile", min=1, value=DEFAULTS["advanced"]["attention_tile_size"]), HELP_TEXT["attention_tile"]),
                                                    with_tip(dmc.NumberInput(id="adv-num-experts", label="Experts", min=1, value=DEFAULTS["advanced"]["num_experts"]), HELP_TEXT["num_experts"]),
                                                    with_tip(dmc.NumberInput(id="adv-top-k", label="MoE top-k", min=1, value=DEFAULTS["advanced"]["top_k"]), HELP_TEXT["top_k"]),
                                                    with_tip(dmc.NumberInput(id="adv-moe-intermediate-size", label="MoE intermediate size", min=1, value=DEFAULTS["advanced"]["moe_intermediate_size"]), HELP_TEXT["moe_intermediate_size"]),
                                                    with_tip(dmc.NumberInput(id="adv-imbalance", label="Expert imbalance", min=0.1, step=0.1, value=DEFAULTS["advanced"]["expert_imbalance_factor"]), HELP_TEXT["expert_imbalance"]),
                                                ],
                                            ),
                                        ],
                                    )
                                ),
                            ],
                        ),
                        dmc.AccordionItem(
                            value="hardware",
                            children=[
                                dmc.AccordionControl("Hardware"),
                                dmc.AccordionPanel(
                                    dmc.SimpleGrid(
                                        cols={"base": 1, "sm": 2},
                                        spacing="sm",
                                        children=[
                                            with_tip(dmc.Switch(id="adv-full-recomp", checked=DEFAULTS["advanced"]["full_recomputation"], label="Full recomputation"), HELP_TEXT["full_recomp"]),
                                            with_tip(dmc.NumberInput(id="adv-dp-zero", label="ZeRO stage", min=0, max=3, value=DEFAULTS["advanced"]["dp_zero_stage"]), HELP_TEXT["zero_stage"]),
                                            with_tip(dmc.Select(id="adv-tensor-format", label="Tensor format", value=DEFAULTS["advanced"]["tensor_format"], data=TENSOR_FORMAT_OPTIONS), HELP_TEXT["tensor_format"]),
                                        ],
                                    )
                                ),
                            ],
                        ),
                        dmc.AccordionItem(
                            value="yaml",
                            children=[
                                dmc.AccordionControl("YAML mirrors"),
                                dmc.AccordionPanel(
                                    dmc.Stack(
                                        gap="sm",
                                        children=[
                                            dmc.Text("These mirrors show the selected YAML files after supported UI edits. Edit unsupported YAML fields directly in webui/workspace/configs.", size="sm", c="dimmed"),
                                            dmc.SimpleGrid(cols={"base": 1, "xl": 2}, spacing="sm", children=[dmc.Textarea(id="model-yaml", label="Model YAML", autosize=True, minRows=16, value=DEFAULTS["model_yaml"], readOnly=True), dmc.Textarea(id="hardware-yaml", label="Hardware YAML", autosize=True, minRows=16, value=DEFAULTS["hardware_yaml"], readOnly=True)]),
                                        ],
                                    )
                                ),
                            ],
                        ),
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
    adv_dp_zero: int,
    adv_tensor_format: str,
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
    net_utils: List[float],
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
        {"run_type": simple_run_type, "seq_len": simple_seq_len, "decode_len": simple_decode_len, "batch_size": simple_batch_size, "grad_accum": simple_grad_accum, "total_gpus": simple_total_gpus, "tp": simple_tp, "cp": simple_cp, "pp": simple_pp, "dp": simple_dp, "ep": simple_ep, "replica_count": simple_replica_count, "hbm_gb": simple_hbm_gb, "compute_derate": simple_compute_derate, "memory_derate": simple_memory_derate, "network_derate": simple_network_derate, "gpu_clock_ghz": simple_gpu_clock, "memory_bw_gbs": simple_memory_bw, "use_astrasim": bool(simple_use_astrasim)},
        {"model_type": adv_model_type, "model_mode": adv_model_mode, "full_recomputation": adv_full_recomp, "dp_zero_stage": adv_dp_zero, "tensor_format": adv_tensor_format, "tied_embeddings": adv_tied_embeddings, "hidden_dim": adv_hidden_dim, "intermediate_size": adv_intermediate_size, "num_layers": adv_num_layers, "vocab_size": adv_vocab_size, "attention_type": adv_attention_type, "num_heads": adv_num_heads, "use_flashattention": adv_use_flash, "attention_tile_size": adv_attn_tile, "num_experts": adv_num_experts, "top_k": adv_top_k, "moe_intermediate_size": adv_moe_intermediate_size, "expert_imbalance_factor": adv_imbalance},
        _network_rows_from_callback(net_topologies, net_bandwidths, net_utils),
        dimensions,
        metric,
        x_axis,
        series_axis,
        worker_count,
        timeout_seconds,
    )


@callback(
    Output("model-editor-tabs", "children"),
    Output("model-editor-tabs", "value"),
    Output("hardware-editor-tabs", "children"),
    Output("hardware-editor-tabs", "value"),
    Input("model-run-configs", "value"),
    Input("hardware-run-configs", "value"),
    State("model-editor-tabs", "value"),
    State("hardware-editor-tabs", "value"),
)
def refresh_editor_tab_sets(model_ids: List[str] | None, hardware_ids: List[str] | None, current_model: str | None, current_hardware: str | None):
    active_model = selected_active_value(model_ids, DEFAULT_MODEL_ID, current_model)
    active_hardware = selected_active_value(hardware_ids, DEFAULT_HW_ID, current_hardware)
    return (
        editor_tabs_children(model_ids, DEFAULT_MODEL_ID, preset_labels("models"), "solar:document-text-bold"),
        active_model,
        editor_tabs_children(hardware_ids, DEFAULT_HW_ID, preset_labels("hardware"), "solar:cpu-bold"),
        active_hardware,
    )


@callback(
    Output("model-preset", "value"),
    Output("hardware-preset", "value"),
    Input("model-editor-tabs", "value"),
    Input("hardware-editor-tabs", "value"),
)
def sync_primary_config_selection(model_id: str | None, hardware_id: str | None):
    return model_id or DEFAULT_MODEL_ID, hardware_id or DEFAULT_HW_ID


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
    Input("create-config-button", "n_clicks"),
    Input("rename-config-button", "n_clicks"),
    State("config-action-kind", "value"),
    State("config-file-name", "value"),
    State("model-preset", "value"),
    State("hardware-preset", "value"),
    State("model-run-configs", "value"),
    State("hardware-run-configs", "value"),
    prevent_initial_call=True,
)
def handle_config_file_action(
    create_clicks: int | None,
    rename_clicks: int | None,
    kind: str,
    new_name: str | None,
    model_preset: str,
    hardware_preset: str,
    model_values: List[str] | None,
    hardware_values: List[str] | None,
):
    del create_clicks, rename_clicks
    source_id = model_preset if kind == "models" else hardware_preset
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
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update
    except Exception as exc:  # noqa: BLE001
        return f"Config action failed: {exc}", preset_options("models"), preset_options("hardware"), preset_options("models"), preset_options("hardware"), model_values, hardware_values
    return message, preset_options("models"), preset_options("hardware"), preset_options("models"), preset_options("hardware"), model_values, hardware_values


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
    Output("network-dimensions-editor", "children"),
    Output("metric-select", "data"),
    Output("metric-select", "value"),
    Input("model-preset", "value"),
    Input("hardware-preset", "value"),
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
        defaults["advanced"]["dp_zero_stage"],
        defaults["advanced"]["tensor_format"],
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
        network_editor(defaults["network_dimensions"]),
        metric_options,
        get_default_metric_for_run_type(defaults["run_type"]),
    )


@callback(
    Output("simple-decode-len", "disabled"),
    Output("simple-replica-count", "disabled"),
    Output("simple-total-gpus", "disabled"),
    Output("simple-tp", "disabled"),
    Output("simple-cp", "disabled"),
    Output("simple-pp", "disabled"),
    Output("simple-dp", "disabled"),
    Output("simple-ep", "disabled"),
    Output("optimizer-preset", "disabled"),
    Output("simple-tp-wrap", "className"),
    Output("simple-cp-wrap", "className"),
    Output("simple-pp-wrap", "className"),
    Output("simple-dp-wrap", "className"),
    Output("simple-ep-wrap", "className"),
    Output("simple-replica-count-wrap", "className"),
    Input("simple-run-type", "value"),
    Input("optimize-switch", "checked"),
)
def toggle_inputs(run_type: str, optimize_parallelism: bool):
    inference = run_type == "inference"
    manual_disabled = bool(optimize_parallelism)
    axis_class = "parallelism-axis-field is-auto" if manual_disabled else "parallelism-axis-field"
    return (
        not inference,
        not inference,
        False if optimize_parallelism else True,
        manual_disabled,
        manual_disabled,
        manual_disabled,
        manual_disabled or inference,
        manual_disabled,
        not optimize_parallelism,
        axis_class,
        axis_class,
        axis_class,
        axis_class,
        axis_class,
        axis_class,
    )


@callback(Output("sweep-dimensions-section", "style"), Input("run-mode", "value"))
def toggle_sweep_dimensions_section(run_mode: str):
    return {} if run_mode == "sweep" else {"display": "none"}


@callback(
    Output("x-axis-select", "data"),
    Output("series-select", "data"),
    Output("dim-1-configs", "data"),
    Output("dim-2-configs", "data"),
    Output("dim-3-configs", "data"),
    Input("dim-1-field", "value"),
    Input("dim-2-field", "value"),
    Input("dim-3-field", "value"),
)
def refresh_dimension_options(field1: str | None, field2: str | None, field3: str | None):
    active_fields = [field for field in [field1, field2, field3] if field]
    options = [{"value": field, "label": dimension_label(field)} for field in active_fields]
    config_data = {"model_config": preset_options("models"), "hardware_config": preset_options("hardware")}
    rows = []
    for field in [field1, field2, field3]:
        rows.append(config_data.get(field, []))
    return options, options, rows[0], rows[1], rows[2]


def sweep_control_visibility(field_key: str | None, mode: str | None) -> Dict[str, Any]:
    hidden = {"display": "none"}
    shown = {}
    if not field_key:
        return {"mode_style": hidden, "values_style": hidden, "configs_style": hidden, "range_style": hidden, "mode": "values"}
    kind = FIELD_TYPES.get(field_key, {}).get("kind")
    if kind == "config":
        return {"mode_style": hidden, "values_style": hidden, "configs_style": shown, "range_style": hidden, "mode": "values"}
    active_mode = "range" if mode == "range" else "values"
    return {
        "mode_style": shown,
        "values_style": shown if active_mode == "values" else hidden,
        "configs_style": hidden,
        "range_style": shown if active_mode == "range" else hidden,
        "mode": active_mode,
    }


@callback(
    Output("dim-1-mode-wrap", "style"),
    Output("dim-1-values-wrap", "style"),
    Output("dim-1-configs-wrap", "style"),
    Output("dim-1-range-wrap", "style"),
    Output("dim-2-mode-wrap", "style"),
    Output("dim-2-values-wrap", "style"),
    Output("dim-2-configs-wrap", "style"),
    Output("dim-2-range-wrap", "style"),
    Output("dim-3-mode-wrap", "style"),
    Output("dim-3-values-wrap", "style"),
    Output("dim-3-configs-wrap", "style"),
    Output("dim-3-range-wrap", "style"),
    Input("dim-1-field", "value"),
    Input("dim-1-mode", "value"),
    Input("dim-2-field", "value"),
    Input("dim-2-mode", "value"),
    Input("dim-3-field", "value"),
    Input("dim-3-mode", "value"),
)
def refresh_sweep_controls(field1: str | None, mode1: str | None, field2: str | None, mode2: str | None, field3: str | None, mode3: str | None):
    outputs = []
    for field, mode in [(field1, mode1), (field2, mode2), (field3, mode3)]:
        visibility = sweep_control_visibility(field, mode)
        outputs.extend([visibility["mode_style"], visibility["values_style"], visibility["configs_style"], visibility["range_style"]])
    return tuple(outputs)


@callback(
    Output("dim-1-range-preview", "children"),
    Output("dim-2-range-preview", "children"),
    Output("dim-3-range-preview", "children"),
    Input("dim-1-field", "value"),
    Input("dim-1-mode", "value"),
    Input("dim-1-start", "value"),
    Input("dim-1-end", "value"),
    Input("dim-1-step_or_points", "value"),
    Input("dim-2-field", "value"),
    Input("dim-2-mode", "value"),
    Input("dim-2-start", "value"),
    Input("dim-2-end", "value"),
    Input("dim-2-step_or_points", "value"),
    Input("dim-3-field", "value"),
    Input("dim-3-mode", "value"),
    Input("dim-3-start", "value"),
    Input("dim-3-end", "value"),
    Input("dim-3-step_or_points", "value"),
)
def refresh_range_previews(
    field1: str | None,
    mode1: str | None,
    start1: float | None,
    end1: float | None,
    step1: float | None,
    field2: str | None,
    mode2: str | None,
    start2: float | None,
    end2: float | None,
    step2: float | None,
    field3: str | None,
    mode3: str | None,
    start3: float | None,
    end3: float | None,
    step3: float | None,
):
    return (
        build_range_preview(field1, mode1, start1, end1, step1),
        build_range_preview(field2, mode2, start2, end2, step2),
        build_range_preview(field3, mode3, start3, end3, step3),
    )


def _network_rows_from_callback(topologies: List[str], bandwidths: List[str], utils: List[float]) -> List[Dict[str, Any]]:
    return [{"topology_type": topology, "bandwidth": bandwidth, "util": util} for topology, bandwidth, util in zip(topologies, bandwidths, utils)]


def _dimensions_from_inputs(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dimensions = []
    for row in raw_rows:
        if row["field"]:
            mode = row["mode"] or "values"
            dimensions.append({"field_key": row["field"], "mode": mode, "list_text": row["list_text"], "config_values": row["config_values"], "start": row["start"], "end": row["end"], "points": None, "step": row["step_or_points"] if mode == "range" else None})
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
    Input({"type": "net-util", "index": ALL}, "value"),
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
    adv_dp_zero: int,
    adv_tensor_format: str,
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
    net_utils: List[float],
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
        net_utils,
        [],
        metric or get_default_metric_for_run_type(simple_run_type),
        x_axis,
        series_axis,
        worker_count or default_worker_count(),
        timeout_seconds or 180,
    )
    save_errors: List[str] = []
    _, _, save_errors = save_config_edits_from_payload(payload)
    if save_errors:
        return no_update, no_update, f"Config sync failed: {save_errors[0]}"
    model_yaml, hardware_yaml, render_errors = render_editable_config_texts(payload)
    if render_errors:
        return no_update, no_update, f"Config sync failed: {render_errors[0]}"
    return model_yaml, hardware_yaml, "Saved UI edits to the active model and hardware YAML files."


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
    Input({"type": "net-util", "index": ALL}, "value"),
    Input("dim-1-field", "value"),
    Input("dim-1-mode", "value"),
    Input("dim-1-list", "value"),
    Input("dim-1-configs", "value"),
    Input("dim-1-start", "value"),
    Input("dim-1-end", "value"),
    Input("dim-1-step_or_points", "value"),
    Input("dim-2-field", "value"),
    Input("dim-2-mode", "value"),
    Input("dim-2-list", "value"),
    Input("dim-2-configs", "value"),
    Input("dim-2-start", "value"),
    Input("dim-2-end", "value"),
    Input("dim-2-step_or_points", "value"),
    Input("dim-3-field", "value"),
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
    adv_dp_zero: int,
    adv_tensor_format: str,
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
    net_utils: List[float],
    dim1_field: str,
    dim1_mode: str,
    dim1_list: str,
    dim1_configs: List[str],
    dim1_start: float,
    dim1_end: float,
    dim1_step_or_points: float,
    dim2_field: str,
    dim2_mode: str,
    dim2_list: str,
    dim2_configs: List[str],
    dim2_start: float,
    dim2_end: float,
    dim2_step_or_points: float,
    dim3_field: str,
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
):
    dimensions = config_dimensions_from_selection(model_run_configs, hardware_run_configs, model_preset, hardware_preset)
    dimensions.extend(_dimensions_from_inputs([{"field": dim1_field, "mode": dim1_mode, "list_text": dim1_list, "config_values": dim1_configs, "start": dim1_start, "end": dim1_end, "step_or_points": dim1_step_or_points}, {"field": dim2_field, "mode": dim2_mode, "list_text": dim2_list, "config_values": dim2_configs, "start": dim2_start, "end": dim2_end, "step_or_points": dim2_step_or_points}, {"field": dim3_field, "mode": dim3_mode, "list_text": dim3_list, "config_values": dim3_configs, "start": dim3_start, "end": dim3_end, "step_or_points": dim3_step_or_points}]))
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
        net_utils,
        dimensions,
        metric,
        x_axis,
        series_axis,
        worker_count,
        timeout_seconds,
    )
    preview = build_launch_preview(payload)
    if not preview.get("ok"):
        return {"payload": payload, "preview": preview}, dmc.Stack(children=[dmc.Alert(item, color="red", radius="lg") for item in preview.get("errors", [])]), launch_button_label(preview)
    return {"payload": payload, "preview": preview}, render_preview_summary(preview, metric), launch_button_label(preview)


@callback(Output("preview-summary", "children", allow_duplicate=True), Input("run-button", "n_clicks"), State("preview-store", "data"), prevent_initial_call=True)
def launch_job(_: int, preview_store: Dict[str, Any] | None):
    if not preview_store:
        return dmc.Alert("The launch plan is still loading. Try again in a moment.", color="red", radius="lg")
    preview, payload = preview_store["preview"], preview_store["payload"]
    if not preview.get("ok"):
        return dmc.Alert("Cannot run until preview errors are fixed.", color="red", radius="lg")
    ok, message = RUN_MANAGER.start_job(payload, preview)
    return dmc.Alert(f"{'Launched' if ok else 'Did not launch'}: {message}", color="green" if ok else "red", radius="lg")


@callback(Output("preview-summary", "children", allow_duplicate=True), Input("cancel-button", "n_clicks"), prevent_initial_call=True)
def cancel_active_job(_: int):
    ok, message = RUN_MANAGER.cancel()
    return dmc.Alert(message, color="yellow" if ok else "red", radius="lg")


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
    Input("selected-detail-store", "data"),
    Input("detail-plot-type", "value"),
    prevent_initial_call=True,
)
def render_detail_modal(selected_detail: Dict[str, Any] | None, plot_type: str | None):
    if not selected_detail:
        return {"display": "none"}, {"display": "none"}, "", render_detail(None, plot_type)
    detail = load_job_detail(selected_detail["kind"], selected_detail["id"])
    plot_toolbar_style = {"display": "block"} if detail.get("kind") == "sweep" else {"display": "none"}
    return {"display": "flex"}, plot_toolbar_style, dmc.Text("Details", fw=800), render_detail(detail, plot_type or "line")


@callback(
    Output("telemetry-ram", "children"),
    Output("telemetry-cpu", "children"),
    Output("telemetry-job", "children"),
    Output("active-job-panel", "children"),
    Output("history-panel", "children"),
    Output("details-panel", "children"),
    Input("poller", "n_intervals"),
    Input("selected-detail-store", "data"),
)
def refresh_status(_: int, selected_detail: Dict[str, Any] | None):
    del selected_detail
    telemetry = get_telemetry()
    active = RUN_MANAGER.active_job()
    if active:
        progress = ((active.get("progress_completed") or 0) / max(1, active.get("progress_total") or 1)) * 100
        active_panel = dmc.Stack(gap="md", children=[dmc.Text(active["title"], fw=700, size="lg"), dmc.Progress(value=progress, size="xl", radius="xl"), dmc.Group(gap="sm", children=[dmc.Badge(f"{active.get('progress_completed', 0)} / {active.get('progress_total', 0)}", radius="xl"), dmc.Badge(active["status"], color="blue", radius="xl", variant="light")])])
        job_badge = f"Active: {active['status']}"
    else:
        active_panel = dmc.Alert("No active job. Adjust settings and launch when ready.", color="green", radius="lg")
        job_badge = "Idle"
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
                                dmc.Text(item["title"], fw=700, size="lg"),
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
    return f"RAM {format_metric_value(telemetry['available_ram_gb'], 'available_ram_gb')} free", f"CPU {telemetry['cpu_percent']}%", job_badge, active_panel, dmc.Stack(children=history_children), no_update


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


def detail_plot_figure(rows: List[Dict[str, Any]], x_axis: str, y_axis: str, series_axis: str | None, plot_type: str | None):
    plot_rows = [dict(row) for row in rows]
    for row in plot_rows:
        if row.get("status") != "completed":
            row[y_axis] = 0
    color_axis = series_axis if series_axis and series_axis in plot_rows[0] and series_axis != "status" else None
    if color_axis is None:
        for candidate in ["model_config", "hardware_config"]:
            values = {row.get(candidate) for row in plot_rows}
            if len(values) > 1:
                color_axis = candidate
                break
    hover_fields = [key for key in ["status", "model_config", "hardware_config", "parallelism"] if key in plot_rows[0]]
    plot_kind = plot_type or "line"
    common = {
        "data_frame": plot_rows,
        "x": x_axis,
        "y": y_axis,
        "hover_name": "case",
        "hover_data": hover_fields,
        "color": color_axis,
        "template": "plotly_white",
        "color_discrete_sequence": ["#0055A6", "#00A5E5", "#3284BF", "#FFC000", "#4F8F2F"],
    }
    if plot_kind == "bar":
        figure = px.bar(**common)
    elif plot_kind == "scatter":
        figure = px.scatter(**common)
    else:
        figure = px.line(**common, markers=True)
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
                dmc.SimpleGrid(cols={"base": 1, "sm": 2, "lg": 4}, children=cards),
                dmc.Paper(radius="xl", p="lg", withBorder=True, children=table),
            ],
        )
    cases = detail.get("cases", [])
    if not cases:
        return dmc.Alert("This sweep has no completed case records yet.", color="gray", radius="lg")
    rows = []
    for case in cases:
        dimension_values = case.get("dimension_values", {}) or {}
        metrics = dict(case.get("metrics", {}) or {})
        model_config = detail_model_config(detail, dimension_values)
        hardware_config = detail_hardware_config(detail, dimension_values)
        row = {
            "case": f"{case['case_id']} - {config_display_name(model_config)}",
            "case_id": case["case_id"],
            "label": case.get("label") or case["case_id"],
            "status": case.get("status", "unknown"),
            "model_config": model_config,
            "hardware_config": hardware_config,
            "parallelism": parallelism_summary_from_payload(payload, case.get("chosen_candidate")),
        }
        row.update({key: value for key, value in dimension_values.items() if key not in {"model_config", "hardware_config"}})
        row.update({key: value for key, value in metrics.items() if key != "memory_violation_gb"})
        if "memory_exceeded" in metrics or "memory_violation_gb" in metrics:
            row["memory_exceeded"] = memory_exceeded_display(metrics)
        rows.append(row)
    preferred_x = payload.get("x_axis")
    preferred_metric = payload.get("metric") or "training_time_s"
    dimension_keys = list((cases[0].get("dimension_values") or {}).keys())
    if preferred_x in rows[0]:
        x_axis = preferred_x
    elif len(dimension_keys) == 1 and dimension_keys[0] in rows[0]:
        x_axis = dimension_keys[0]
    else:
        x_axis = "case_id"
    y_axis = preferred_metric if preferred_metric in rows[0] else ("training_time_s" if "training_time_s" in rows[0] else "prefill_time_s")
    for row in rows:
        if y_axis not in row:
            row[y_axis] = 0
        if row.get("status") != "completed" and y_axis:
            row[y_axis] = 0
    series_axis = payload.get("series_axis")
    figure = detail_plot_figure(rows, x_axis, y_axis, series_axis, plot_type)
    display_rows = []
    for row in rows:
        display_row = {}
        for key, value in row.items():
            if key in {"case", "case_id", "label", "status", "model_config", "hardware_config", "parallelism", "memory_exceeded"}:
                display_row[key] = value
            else:
                display_row[key] = format_metric_value(value, key)
        display_rows.append(display_row)
    column_order = [
        "case",
        "status",
        "model_config",
        "hardware_config",
        "parallelism",
        "label",
    ]
    ordered_keys = [key for key in column_order if key in rows[0]] + [key for key in rows[0].keys() if key not in column_order and key != "case_id"]
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
    )
    detail_note = (
        f"{len(rows):,} displayed case rows. Candidate trials from parallelism search are summarized, not listed individually."
        if optimizer_enabled
        else f"{len(rows):,} displayed case rows."
    )
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
                        children=[
                            dmc.Badge("Parallelism optimized" if optimizer_enabled else "Fixed parallelism", color="blue" if optimizer_enabled else "gray", variant="light", radius="xl"),
                            dmc.Badge(parallelism_summary_from_payload(payload), color="teal", variant="light", radius="xl"),
                        ],
                    ),
                ],
            ),
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
