from __future__ import annotations

from dash import dash_table, dcc

from webui.service.core import FIELD_OPTIONS
from webui.app.main import (
    MODEL_ARCH_TYPE_HELP,
    MODEL_ARCH_TYPE_OPTIONS,
    TENSOR_FORMAT_OPTIONS,
    build_range_preview,
    config_dimensions_from_selection,
    editor_tabs_children,
    format_metric_value,
    help_for_key,
    metric_key_from_label,
    render_detail,
    selected_active_value,
    sweep_control_visibility,
)


def test_flop_count_values_use_engineering_units():
    assert format_metric_value(1.23e15, "total_flops") == "1.23 PFLOP"


def test_flop_rate_values_use_engineering_units():
    assert format_metric_value(9.89e14, "achieved_flops") == "989 TFLOPS"
    assert format_metric_value(3.35e12, "peak_flops_per_gpu") == "3.35 TFLOPS"
    assert format_metric_value(1.12e15, "peak_system_flops") == "1.12 PFLOPS"


def test_token_rate_and_time_values_are_human_readable():
    assert format_metric_value(1_234_567, "decode_throughput_tok_s") == "1.23 Mtok/s"
    assert format_metric_value(0.00123, "training_time_s") == "1.23 ms"
    assert format_metric_value(180, "timeout_seconds") == "3m 0s"


def test_other_large_quantities_are_compacted():
    assert format_metric_value(123_456_789, "model.vocab_size") == "123 M"
    assert format_metric_value(123_456.0, "available_ram_gb") == "123 TB"


def test_metric_labels_resolve_to_formatter_keys():
    assert metric_key_from_label("Time / Batch") == "training_time_s"
    assert metric_key_from_label("Decode Throughput") == "decode_throughput_tok_s"


def test_supported_model_types_have_explanations():
    values = [item["value"] for item in MODEL_ARCH_TYPE_OPTIONS]

    assert values == ["gpt", "llama", "deepseek_v3", "glm4_moe", "vit", "vit_dinov3"]
    assert set(values) == set(MODEL_ARCH_TYPE_HELP)
    assert all(len(MODEL_ARCH_TYPE_HELP[value]) > 40 for value in values)


def test_tensor_format_options_show_precision_widths():
    assert TENSOR_FORMAT_OPTIONS == [
        {"value": "mxfp4", "label": "mxfp4 (4.25)"},
        {"value": "int4", "label": "int4 (4)"},
        {"value": "fp8", "label": "fp8 (8)"},
        {"value": "fp16", "label": "fp16 (16)"},
        {"value": "bf16", "label": "bf16 (16)"},
        {"value": "fp32", "label": "fp32 (32)"},
    ]


def test_sweep_controls_mutate_by_field_kind():
    config_state = sweep_control_visibility("model_config", "values")
    assert config_state["configs_style"] == {}
    assert config_state["values_style"] == {"display": "none"}
    assert config_state["range_style"] == {"display": "none"}

    values_state = sweep_control_visibility("model.global_batch_size", "values")
    assert values_state["values_style"] == {}
    assert values_state["configs_style"] == {"display": "none"}
    assert values_state["range_style"] == {"display": "none"}

    range_state = sweep_control_visibility("model.global_batch_size", "range")
    assert range_state["range_style"] == {}
    assert range_state["values_style"] == {"display": "none"}


def test_sweep_field_options_exclude_raw_parallelism_axes():
    values = {item["value"] for item in FIELD_OPTIONS}

    assert "hardware.total_gpus" in values
    assert "hardware.parallelism.tp" not in values
    assert "hardware.parallelism.cp" not in values
    assert "hardware.parallelism.pp" not in values
    assert "hardware.parallelism.dp" not in values
    assert "hardware.parallelism.ep" not in values


def test_range_preview_lists_steps_and_caps_long_ranges():
    assert build_range_preview("model.global_batch_size", "range", 32, 96, 32) == "Preview: 32, 64, 96 (3 values)"

    long_preview = build_range_preview("model.seq_len", "range", 1, 40, 1, limit=6)

    assert long_preview == "Preview: 1, 2, 3, 4, 5, 6, ... (40 values)"


def test_range_preview_explains_invalid_inputs():
    assert build_range_preview("model.seq_len", "range", None, 40, 1) == "Preview: enter start, end, and step size."
    assert build_range_preview("model.seq_len", "range", 1, 40, 0) == "Preview: step size must be greater than 0."
    assert build_range_preview("model.seq_len", "range", 40, 1, 1) == "Preview: end must be greater than or equal to start."


def test_run_setup_selection_creates_hidden_config_dimensions():
    dimensions = config_dimensions_from_selection(
        ["Llama2-7B.yaml", "Llama3.1-70B_2d_train.yaml"],
        ["A100_SXM4_80GB_base.yaml"],
        "Llama2-7B.yaml",
        "A100_SXM4_80GB_base.yaml",
    )

    assert dimensions == [
        {
            "field_key": "model_config",
            "mode": "values",
            "config_values": ["Llama2-7B.yaml", "Llama3.1-70B_2d_train.yaml"],
        }
    ]


def test_editor_tabs_track_selected_files():
    children = editor_tabs_children(
        ["Llama2-7B.yaml", "Llama3.1-70B_2d_train.yaml"],
        "Llama2-7B.yaml",
        {"Llama2-7B.yaml": "Llama2-7B", "Llama3.1-70B_2d_train.yaml": "Llama3 train"},
        "solar:document-text-bold",
    )

    tabs = children[0].children
    assert len(tabs) == 2
    assert [tab.value for tab in tabs] == ["Llama2-7B.yaml", "Llama3.1-70B_2d_train.yaml"]
    assert selected_active_value(["Llama2-7B.yaml", "Llama3.1-70B_2d_train.yaml"], "Llama2-7B.yaml", "Llama3.1-70B_2d_train.yaml") == "Llama3.1-70B_2d_train.yaml"
    assert selected_active_value(["Llama2-7B.yaml"], "Llama2-7B.yaml", "Llama3.1-70B_2d_train.yaml") == "Llama2-7B.yaml"


def _collect_datatables(component):
    if isinstance(component, dash_table.DataTable):
        return [component]
    children = getattr(component, "children", None)
    if children is None:
        return []
    if not isinstance(children, list):
        children = [children]
    tables = []
    for child in children:
        tables.extend(_collect_datatables(child))
    return tables


def _collect_graphs(component):
    if isinstance(component, dcc.Graph):
        return [component]
    children = getattr(component, "children", None)
    if children is None:
        return []
    if not isinstance(children, list):
        children = [children]
    graphs = []
    for child in children:
        graphs.extend(_collect_graphs(child))
    return graphs


def test_detail_tables_explain_metric_columns():
    detail = {
        "kind": "run",
        "title": "Tiny run",
        "result": {
            "status": "completed",
            "metrics": {
                "training_time_s": 1.25,
                "achieved_flops": 9.89e14,
                "achieved_flops_per_gpu": 1.23e14,
                "peak_system_flops": 1.12e15,
            },
            "primary_metric_value": 1.25,
        },
    }

    tables = _collect_datatables(render_detail(detail))

    assert tables
    assert tables[0].tooltip_header["metric"] == "Metric name emitted by the worker."
    assert "system-wide" in help_for_key("achieved_flops").lower()
    assert any("Peak System FLOPS" == row["metric"] for row in tables[0].data)


def test_detail_memory_fields_are_merged():
    detail = {
        "kind": "run",
        "title": "Memory run",
        "request_record": {"payload": {"model_preset_id": "Llama2-7B.yaml", "hardware_preset_id": "A100.yaml", "simple": {"tp": 1, "cp": 1, "pp": 1, "dp": 1, "ep": 1}}},
        "result": {
            "status": "completed",
            "metrics": {
                "training_time_s": 1.0,
                "memory_exceeded": True,
                "memory_violation_gb": 12.5,
            },
            "primary_metric_value": 1.0,
        },
    }

    table = _collect_datatables(render_detail(detail))[0]

    labels = [row["metric"] for row in table.data]
    assert "Memory Exceeded" in labels
    assert "Memory Violation" not in labels
    assert next(row["value"] for row in table.data if row["metric"] == "Memory Exceeded") == "12.5 GB"


def test_sweep_detail_uses_model_config_case_labels_and_status_not_legend():
    detail = {
        "kind": "sweep",
        "title": "Sweep",
        "request_record": {
            "payload": {
                "model_preset_id": "Llama2-7B.yaml",
                "hardware_preset_id": "H100.yaml",
                "metric": "training_time_s",
                "simple": {"tp": 1, "cp": 1, "pp": 1, "dp": 8, "ep": 1},
            },
            "preview": {"optimizer_enabled": True},
        },
        "cases": [
            {
                "case_id": "case-0001",
                "label": "Model Config=Llama2-7B.yaml",
                "status": "completed",
                "dimension_values": {"model_config": "Llama2-7B.yaml"},
                "chosen_candidate": {"tp": 2, "cp": 1, "pp": 1, "dp": 4, "ep": 1},
                "metrics": {"training_time_s": 2.0, "memory_exceeded": False, "memory_violation_gb": 0.0},
            },
            {
                "case_id": "case-0002",
                "label": "Model Config=Llama3.yaml",
                "status": "failed",
                "dimension_values": {"model_config": "Llama3.yaml"},
                "chosen_candidate": {"tp": 4, "cp": 1, "pp": 1, "dp": 2, "ep": 1},
                "metrics": {},
            },
        ],
    }

    rendered = render_detail(detail)
    table = _collect_datatables(rendered)[0]
    graph = _collect_graphs(rendered)[0]

    assert table.data[0]["case"] == "case-0001 - Llama2-7B"
    assert table.data[0]["parallelism"] == "TP 2 / CP 1 / PP 1 / DP 4 / EP 1"
    assert table.data[1]["training_time_s"] == "0.00 us"
    assert all(trace.name not in {"completed", "failed", "timed_out"} for trace in graph.figure.data)
