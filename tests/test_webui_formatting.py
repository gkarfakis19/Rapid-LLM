from __future__ import annotations

import json
import os
import time
from datetime import datetime
from time import perf_counter
from pathlib import Path

from dash import no_update
from dash import dash_table, dcc, html

import webui.app.main as webui_main
from webui.service.core import FIELD_OPTIONS
from webui.app.main import (
    APP_CREDIT_TEXT,
    APP_LOGO_ASSET,
    DATA_TABLE_STYLE_FILTER,
    DEFAULT_OPTIMIZE_PARALLELISM,
    DETAIL_HIDDEN_METRIC_KEYS,
    DETAIL_PLOT_POINT_LIMIT,
    DETAIL_RENDER_CASE_LIMIT,
    DETAIL_TABLE_ROW_LIMIT,
    HELP_TEXT,
    HF_SAMPLE_MODEL_URL,
    MODEL_ARCH_TYPE_HELP,
    MODEL_ARCH_TYPE_OPTIONS,
    MODEL_MODE_OPTIONS,
    PP_TOPOLOGY_OPTIONS,
    PRECISION_FORMAT_OPTIONS,
    TENSOR_FORMAT_OPTIONS,
    ZERO_STAGE_OPTIONS,
    _default_payload,
    active_config_tab_for_selection_change,
    active_config_tab_value,
    branded_progress_bar,
    build_range_preview,
    clamp_percent,
    config_dimensions_from_selection,
    config_tab_value,
    config_workbook_tabs_children,
    create_layout,
    detail_axes,
    detail_plot_png_bytes,
    detail_table_export_payload,
    detail_loading_placeholder,
    enforce_training_replica_count,
    format_finished_job_badge,
    format_metric_value,
    format_worst_case_wall_clock,
    help_for_key,
    history_title_component,
    job_eta_readout,
    launch_job,
    metric_key_from_label,
    mode_badge,
    open_run_log_from_completed_progress,
    parse_config_tab,
    preview_rebuild_is_load_only,
    progress_count_label,
    render_detail,
    render_preview_summary,
    refresh_status,
    reset_paper_derates,
    reset_last_state_values,
    refresh_metric_options_for_run_type,
    refresh_dimension_options,
    selected_active_value,
    sweep_rows_from_detail,
    sweep_control_visibility,
    toggle_inputs,
)


def test_flop_count_values_use_engineering_units():
    assert format_metric_value(1.23e15, "total_flops") == "1.23 PFLOP"


def test_flop_rate_values_use_engineering_units():
    assert format_metric_value(9.89e14, "achieved_flops") == "989 TFLOPS"
    assert format_metric_value(1.23e14, "achieved_flops_per_gpu") == "123 TFLOPS"


def test_token_rate_and_time_values_are_human_readable():
    assert format_metric_value(1_234_567, "decode_throughput_tok_s") == "1.23 Mtok/s"
    assert format_metric_value(0.00123, "training_time_s") == "1.23 ms"
    assert format_metric_value(180, "timeout_seconds") == "3m 0s"
    assert format_metric_value(0, "timeout_seconds") == "Disabled"
    assert format_worst_case_wall_clock(None) == "N/A"
    assert format_worst_case_wall_clock(180) == "3m 0s"


def test_progress_percent_is_clamped():
    assert clamp_percent(-12) == 0.0
    assert clamp_percent("57.5") == 57.5
    assert clamp_percent(180) == 100.0
    assert clamp_percent("not-a-number") == 0.0


def test_other_large_quantities_are_compacted():
    assert format_metric_value(123_456_789, "model.vocab_size") == "123 M"
    assert format_metric_value(123_456.0, "available_ram_gb") == "123 TB"


def test_metric_labels_resolve_to_formatter_keys():
    assert metric_key_from_label("Time / Batch") == "training_time_s"
    assert metric_key_from_label("Decode Throughput") == "decode_throughput_tok_s"
    assert metric_key_from_label("Throughput (TPOT)") == "decode_throughput_tok_s"
    assert metric_key_from_label("TTFT") == "ttft_s"


def test_finished_job_badge_uses_completed_time():
    assert format_finished_job_badge({"status": "completed", "updated_at": "2026-04-27T12:25:53"}) == "Completed, 12:25"


def test_launch_success_leaves_launch_plan_unchanged(monkeypatch):
    class FakeRunManager:
        def start_job(self, payload, preview):
            assert payload == {"payload": True}
            assert preview == {"ok": True}
            return True, "sweep-20260427-122553-b67458"

    monkeypatch.setattr(webui_main, "RUN_MANAGER", FakeRunManager())

    assert launch_job(1, {"preview": {"ok": True}, "payload": {"payload": True}}) is no_update


def test_launch_failure_still_reports_launch_plan_error(monkeypatch):
    class FakeRunManager:
        def start_job(self, payload, preview):
            return False, "Another job is already running."

    monkeypatch.setattr(webui_main, "RUN_MANAGER", FakeRunManager())

    alert = launch_job(1, {"preview": {"ok": True}, "payload": {}})

    assert any("Did not launch: Another job is already running." in text for text in _collect_text(alert))


def test_metric_options_follow_run_type_selection():
    inference_data, inference_value = refresh_metric_options_for_run_type("inference")
    training_data, training_value = refresh_metric_options_for_run_type("training")

    assert inference_value == "decode_throughput_tok_s"
    assert inference_data == [
        {"value": "decode_throughput_tok_s", "label": "Throughput (TPOT)"},
        {"value": "ttft_s", "label": "TTFT"},
        {"value": "total_inference_time_s", "label": "Time / Batch"},
    ]
    assert training_value == "training_time_s"
    assert training_data == [
        {"value": "training_time_s", "label": "Time / Batch"},
        {"value": "approx_mfu", "label": "Approx. MFU"},
    ]


def test_supported_model_types_have_explanations():
    values = [item["value"] for item in MODEL_ARCH_TYPE_OPTIONS]

    assert values == ["gpt", "llama", "deepseek_v3", "glm4_moe", "vit", "vit_dinov3"]
    assert set(values) == set(MODEL_ARCH_TYPE_HELP)
    assert all(len(MODEL_ARCH_TYPE_HELP[value]) > 40 for value in values)


def test_gemm_mode_is_not_exposed_in_model_mode_picker():
    values = [item["value"] for item in MODEL_MODE_OPTIONS]

    assert values == ["LLM", "VIT"]


def test_execution_family_guide_uses_vit_green_and_llm_blue():
    llm_badge = mode_badge("LLM").children.children
    vit_badge = mode_badge("VIT").children.children

    assert llm_badge.color == "blue"
    assert vit_badge.color == "teal"


def test_tensor_format_options_show_precision_widths():
    assert TENSOR_FORMAT_OPTIONS == [
        {"value": "mxfp4", "label": "MXFP4 (4.25 bits)"},
        {"value": "int4", "label": "INT4 (4 bits)"},
        {"value": "fp8", "label": "FP8 (8 bits)"},
        {"value": "fp16", "label": "FP16 (16 bits)"},
        {"value": "bf16", "label": "BF16 (16 bits)"},
        {"value": "fp32", "label": "FP32 (32 bits)"},
    ]


def test_sub_precision_options_can_match_tensor_format():
    assert PRECISION_FORMAT_OPTIONS[0] == {"value": "as_tensor_format", "label": "Match tensor format"}
    assert {"value": "fp32", "label": "FP32 (32 bits)"} in PRECISION_FORMAT_OPTIONS


def test_zero_stage_options_are_dropdown_labels():
    assert ZERO_STAGE_OPTIONS == [
        {"value": "0", "label": "0 (DDP)"},
        {"value": "1", "label": "1 (optimizer shard)"},
        {"value": "2", "label": "2 (optimizer+grad shard)"},
        {"value": "3", "label": "3 (FSDP/full shard)"},
    ]


def test_plot_grouping_defaults_to_last_active_sweep_dimension():
    x_data, x_value, series_data, series_value, *_ = refresh_dimension_options(
        "model.global_batch_size",
        "hardware.total_gpus",
        None,
        None,
        None,
    )

    assert [item["value"] for item in x_data] == ["model.global_batch_size", "hardware.total_gpus"]
    assert x_value == "model.global_batch_size"
    assert [item["value"] for item in series_data] == ["model.global_batch_size", "hardware.total_gpus"]
    assert series_value == "hardware.total_gpus"


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
        ["A100_SXM4_80GB.yaml"],
        "Llama2-7B.yaml",
        "A100_SXM4_80GB.yaml",
    )

    assert dimensions == [
        {
            "field_key": "model_config",
            "mode": "values",
            "config_values": ["Llama2-7B.yaml", "Llama3.1-70B_2d_train.yaml"],
        }
    ]


def test_config_workbook_tabs_track_selected_files():
    children = config_workbook_tabs_children(
        ["Llama2-7B.yaml", "DeepSeekV3_inf_16k.yaml"],
        ["H100_SXM5_80GB.yaml"],
        {"Llama2-7B.yaml": "Llama2-7B", "DeepSeekV3_inf_16k.yaml": "DeepSeekV3 inference 16k"},
        {"H100_SXM5_80GB.yaml": "H100 SXM5 80GB"},
    )

    tabs = children[0].children
    assert len(tabs) == 3
    assert [tab.value for tab in tabs] == [
        "models::Llama2-7B.yaml",
        "models::DeepSeekV3_inf_16k.yaml",
        "hardware::H100_SXM5_80GB.yaml",
    ]
    assert tabs[1].className == "config-workbook-tab"
    assert tabs[1].children.children[1].children == "DeepSeekV3 inference 16k"
    assert parse_config_tab("hardware::H100_SXM5_80GB.yaml") == ("hardware", "H100_SXM5_80GB.yaml")
    assert config_tab_value("models", "DeepSeekV3_inference_16k.yaml") == "models::DeepSeekV3_inference_16k.yaml"
    assert active_config_tab_value(
        ["Llama2-7B.yaml", "DeepSeekV3_inf_16k.yaml"],
        ["H100_SXM5_80GB.yaml"],
        "Llama2-7B.yaml",
        "H100_SXM5_80GB.yaml",
        "hardware::H100_SXM5_80GB.yaml",
        "Llama2-7B.yaml",
        "H100_SXM5_80GB.yaml",
    ) == "hardware::H100_SXM5_80GB.yaml"
    assert selected_active_value(["Llama2-7B.yaml", "Llama3.1-70B_2d_train.yaml"], "Llama2-7B.yaml", "Llama3.1-70B_2d_train.yaml") == "Llama3.1-70B_2d_train.yaml"
    assert selected_active_value(["Llama2-7B.yaml"], "Llama2-7B.yaml", "Llama3.1-70B_2d_train.yaml") == "Llama2-7B.yaml"


def test_launch_setup_selection_change_activates_matching_editor_tab():
    active_model = active_config_tab_for_selection_change(
        ["Llama2-7B.yaml", "DeepSeekV3_inf_16k.yaml"],
        ["H100_SXM5_80GB.yaml"],
        "Llama2-7B.yaml",
        "H100_SXM5_80GB.yaml",
        config_tab_value("hardware", "H100_SXM5_80GB.yaml"),
        "Llama2-7B.yaml",
        "H100_SXM5_80GB.yaml",
        "model-run-configs",
    )
    active_hardware = active_config_tab_for_selection_change(
        ["Llama2-7B.yaml", "DeepSeekV3_inf_16k.yaml"],
        ["H100_SXM5_80GB.yaml", "A100_SXM4_80GB.yaml"],
        "Llama2-7B.yaml",
        "H100_SXM5_80GB.yaml",
        config_tab_value("models", "DeepSeekV3_inf_16k.yaml"),
        "DeepSeekV3_inf_16k.yaml",
        "H100_SXM5_80GB.yaml",
        "hardware-run-configs",
    )
    children = config_workbook_tabs_children(
        ["Llama2-7B.yaml", "DeepSeekV3_inf_16k.yaml"],
        ["H100_SXM5_80GB.yaml", "A100_SXM4_80GB.yaml"],
        {"Llama2-7B.yaml": "Llama2-7B", "DeepSeekV3_inf_16k.yaml": "DeepSeekV3 inference 16k"},
        {"H100_SXM5_80GB.yaml": "H100", "A100_SXM4_80GB.yaml": "A100"},
    )
    tab_values = [tab.value for tab in children[0].children]

    assert active_model == config_tab_value("models", "DeepSeekV3_inf_16k.yaml")
    assert active_hardware == config_tab_value("hardware", "A100_SXM4_80GB.yaml")
    assert config_tab_value("hardware", "H100_SXM5_80GB.yaml") in tab_values
    assert config_tab_value("hardware", "A100_SXM4_80GB.yaml") in tab_values


def test_preview_rebuild_skips_config_load_only_triggers():
    assert preview_rebuild_is_load_only({"config-editor-tabs.value": "models::DeepSeekV3_inf_16k.yaml"}) is True
    assert preview_rebuild_is_load_only({"model-preset.value": "DeepSeekV3_inf_16k.yaml", "hardware-preset.value": "H100_SXM5_80GB.yaml"}) is True
    assert preview_rebuild_is_load_only({"model-preset.value": "DeepSeekV3_inf_16k.yaml", "simple-batch-size.value": 64}) is False
    assert preview_rebuild_is_load_only({}) is False


def test_layout_restores_last_saved_sweep_and_config_state(monkeypatch):
    saved_state = {
        "model_run_configs": ["Llama2-7B.yaml", "DeepSeekV3_inf_16k.yaml"],
        "hardware_run_configs": ["A100_SXM4_80GB.yaml"],
        "model_preset": "DeepSeekV3_inf_16k.yaml",
        "hardware_preset": "A100_SXM4_80GB.yaml",
        "active_config_tab": config_tab_value("hardware", "A100_SXM4_80GB.yaml"),
        "run_mode": "single",
        "optimize_parallelism": False,
        "optimizer_preset": "Exhaustive",
        "sweep_rows": [
            {"field": "model.global_batch_size", "mode": "range", "list_text": "", "config_values": [], "start": 64, "end": 256, "step_or_points": 64},
            {"field": "hardware.total_gpus", "mode": "values", "list_text": "8, 16", "config_values": [], "start": None, "end": None, "step_or_points": None},
            {"field": None, "mode": "values", "list_text": "", "config_values": [], "start": None, "end": None, "step_or_points": None},
        ],
        "metric": "training_time_s",
        "x_axis": "model.global_batch_size",
        "series_axis": "hardware.total_gpus",
        "worker_count": 3,
        "timeout_seconds": 77,
    }
    monkeypatch.setattr(webui_main, "load_last_ui_state", lambda: saved_state)

    layout = create_layout()

    assert _collect_by_id(layout, "model-run-configs")[0].value == ["Llama2-7B.yaml", "DeepSeekV3_inf_16k.yaml"]
    assert _collect_by_id(layout, "hardware-run-configs")[0].value == ["A100_SXM4_80GB.yaml"]
    assert _collect_by_id(layout, "model-preset")[0].value == "DeepSeekV3_inf_16k.yaml"
    assert _collect_by_id(layout, "hardware-preset")[0].value == "A100_SXM4_80GB.yaml"
    assert _collect_by_id(layout, "config-editor-tabs")[0].value == config_tab_value("hardware", "A100_SXM4_80GB.yaml")
    assert _collect_by_id(layout, "run-mode")[0].value == "single"
    assert _collect_by_id(layout, "optimize-switch")[0].checked is False
    assert _collect_by_id(layout, "optimizer-preset")[0].value == "Exhaustive"
    assert _collect_by_id(layout, "dim-1-field")[0].value == "model.global_batch_size"
    assert _collect_by_id(layout, "dim-1-mode")[0].value == "range"
    assert _collect_by_id(layout, "dim-1-start")[0].value == 64
    assert _collect_by_id(layout, "dim-2-list")[0].value == "8, 16"
    assert _collect_by_id(layout, "x-axis-select")[0].value == "model.global_batch_size"
    assert _collect_by_id(layout, "series-select")[0].value == "hardware.total_gpus"
    assert _collect_by_id(layout, "worker-count")[0].value == 3
    assert _collect_by_id(layout, "timeout-seconds")[0].value == 77
    assert _collect_by_id(layout, "timeout-seconds")[0].min == 0
    assert _collect_by_id(layout, "reset-last-state-button")


def test_model_dropdown_options_show_training_or_inference(monkeypatch):
    def fake_records(kind):
        if kind == "models":
            return [
                {"id": "Llama2-7B.yaml", "label": "Llama2-7B", "run_type": "training"},
                {"id": "Llama2-7B_inf.yaml", "label": "Llama2-7B inf", "run_type": "inference"},
            ]
        return [{"id": "H100.yaml", "label": "H100"}]

    monkeypatch.setattr(webui_main, "preset_records", fake_records)

    assert webui_main.preset_options("models") == [
        {"value": "Llama2-7B.yaml", "label": "Llama2-7B (training)"},
        {"value": "Llama2-7B_inf.yaml", "label": "Llama2-7B inf (inference)"},
    ]
    assert webui_main.preset_options("hardware") == [{"value": "H100.yaml", "label": "H100"}]


def test_launch_plan_renders_zero_timeout_as_unbounded(monkeypatch):
    monkeypatch.setattr(webui_main, "get_telemetry", lambda: {"available_ram_gb": 32.0, "used_percent": 50.0, "cpu_percent": 10.0})
    preview = {
        "ok": True,
        "top_level_case_count": 2,
        "total_invocations": 6,
        "worst_case_wall_clock_s": None,
        "worker_count": 2,
        "timeout_seconds": 0,
        "warnings": [],
    }

    texts = _collect_text(render_preview_summary(preview, "training_time_s"))

    assert "N/A" in texts
    assert "Timeout: Disabled" in texts


def test_reset_last_state_values_returns_default_scratchpad_controls():
    values = reset_last_state_values()

    assert values[0] == "Restored default selections."
    assert values[1] == ["Llama2-7B.yaml"]
    assert values[2] == ["H100_SXM5_80GB.yaml"]
    assert values[5] == config_tab_value("models", "Llama2-7B.yaml")
    assert values[6] == "sweep"
    assert values[7] is DEFAULT_OPTIMIZE_PARALLELISM
    assert values[9:30] == tuple([None, "values", "", [], None, None, None] * 3)


def test_optimize_parallelism_does_not_auto_replica_count_for_inference():
    outputs = toggle_inputs("inference", True, "sweep", None, None, None)

    assert outputs[1] is False  # replica count remains editable for inference
    assert outputs[10] == "parallelism-axis-field is-auto"
    assert outputs[16] == "parallelism-axis-field"


def test_training_replica_count_is_disabled_and_forced_to_one():
    outputs = toggle_inputs("training", False, "sweep", None, None, None)

    assert outputs[1] is True
    assert enforce_training_replica_count("training", 8) == 1
    assert enforce_training_replica_count("inference", 8) is no_update


def test_optimize_parallelism_defaults_on_in_layout_and_payload():
    layout = create_layout()
    optimize_switch = _collect_by_id(layout, "optimize-switch")[0]
    payload = _default_payload()

    assert DEFAULT_OPTIMIZE_PARALLELISM is True
    assert optimize_switch.checked is True
    assert payload["optimize_parallelism"] is True


def test_detail_plot_png_export_is_valid_image_bytes():
    rows = [
        {"case": "case-0001", "case_id": "case-0001", "status": "completed", "model.global_batch_size": 64, "training_time_s": 1.2, "model_config": "Llama2-7B.yaml", "hardware_config": "H100.yaml", "parallelism": "TP 4"},
        {"case": "case-0002", "case_id": "case-0002", "status": "completed", "model.global_batch_size": 128, "training_time_s": 0.8, "model_config": "Llama2-7B.yaml", "hardware_config": "H100.yaml", "parallelism": "TP 8"},
    ]

    png_bytes = detail_plot_png_bytes(rows, "model.global_batch_size", "training_time_s", None, "line", "Llama2-7B on H100 Sweep")

    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png_bytes) > 20_000


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


def _collect_images(component):
    if isinstance(component, html.Img):
        return [component]
    children = getattr(component, "children", None)
    if children is None:
        return []
    if not isinstance(children, list):
        children = [children]
    images = []
    for child in children:
        images.extend(_collect_images(child))
    return images


def _collect_details(component):
    if isinstance(component, html.Details):
        return [component]
    children = getattr(component, "children", None)
    if children is None:
        return []
    if not isinstance(children, list):
        children = [children]
    details = []
    for child in children:
        details.extend(_collect_details(child))
    return details


def _collect_class_names(component):
    names = []
    class_name = getattr(component, "className", None)
    if class_name:
        names.append(class_name)
    children = getattr(component, "children", None)
    if children is None:
        return names
    if not isinstance(children, list):
        children = [children]
    for child in children:
        names.extend(_collect_class_names(child))
    return names


def _collect_text(component):
    if isinstance(component, str):
        return [component]
    children = getattr(component, "children", None)
    if children is None:
        return []
    if not isinstance(children, list):
        children = [children]
    texts = []
    for child in children:
        texts.extend(_collect_text(child))
    return texts


def _collect_by_id(component, component_id):
    if getattr(component, "id", None) == component_id:
        return [component]
    children = getattr(component, "children", None)
    if children is None:
        return []
    if not isinstance(children, list):
        children = [children]
    matches = []
    for child in children:
        matches.extend(_collect_by_id(child, component_id))
    return matches


def test_layout_includes_discreet_page_credit():
    texts = _collect_text(create_layout())

    assert APP_CREDIT_TEXT in texts


def test_layout_includes_nanocad_logo_asset_and_no_update_badge():
    layout = create_layout()
    images = _collect_images(layout)
    texts = _collect_text(layout)
    class_names = _collect_class_names(layout)

    assert Path("webui/app/assets").joinpath(APP_LOGO_ASSET).is_file()
    assert any(APP_LOGO_ASSET in str(image.src) and image.alt == "NanoCAD" for image in images)
    assert "nanocad-logo-frame" in class_names
    assert "topbar-logo-slot" in class_names
    assert "brand-orb" not in class_names
    assert "Updates as settings change" not in texts
    assert any("rapidly approach worst-case" in text and "beyond 256 GPUs" in text for text in texts)


def test_header_telemetry_pills_have_initial_display_text():
    texts = _collect_text(create_layout())

    assert "RAM --" in texts
    assert "CPU --" in texts
    assert "Idle" in texts


def test_yaml_mirror_sections_are_collapsed_by_default():
    layout = create_layout()
    details = [item for item in _collect_details(layout) if item.className == "option-section yaml-mirror-section"]
    texts = _collect_text(layout)

    assert len(details) == 2
    assert all(item.open is False for item in details)
    assert texts.count("YAML mirror") == 2


def test_layout_includes_huggingface_model_import_controls():
    layout = create_layout()
    texts = _collect_text(layout)
    hf_url_fields = _collect_by_id(layout, "hf-model-url")
    hf_name_fields = _collect_by_id(layout, "hf-config-name")

    assert hf_url_fields
    assert hf_url_fields[0].value == HF_SAMPLE_MODEL_URL
    assert hf_url_fields[0].label == "Hugging Face URL or model ID"
    assert hf_name_fields
    assert hf_name_fields[0].label == "Save as"
    assert "Create model config" in texts
    assert any("Not automatically determinable:" in text and "decode length" in text for text in texts)


def test_branded_progress_bar_fills_rapid_llm_text():
    progress = branded_progress_bar(62.5, "5 / 8", "RUNNING")
    track = progress.children[0]
    core = track.children[1]
    props = track.to_plotly_json()["props"]
    track_classes = _collect_class_names(track)
    texts = _collect_text(progress)

    assert progress.className == "rapid-progress-shell"
    assert track.className == "rapid-progress-track"
    assert props["role"] == "progressbar"
    assert props["style"]["--rapid-progress"] == "62.50%"
    assert props["aria-valuenow"] == "62.50"
    assert texts.count("RAPID-LLM") == 1
    assert "62%" in texts
    assert "5 / 8" in texts
    assert "RUNNING" in texts
    assert "rapid-progress-fill" in track_classes
    assert "rapid-progress-core" in track_classes
    assert core.to_plotly_json()["props"]["data-label"] == "RAPID-LLM"


def test_completed_progress_bar_shows_run_log_action():
    progress = branded_progress_bar(100, "8 / 8", "COMPLETED", show_run_log_button=True)
    button = _collect_by_id(progress, "progress-run-log-button")[0]
    class_names = _collect_class_names(progress)
    texts = _collect_text(progress)

    assert "100%" in texts
    assert "Run Log" in texts
    assert button.children == "Run Log"
    assert "rapid-progress-complete-action" in class_names
    assert button.className == "rapid-progress-run-log-button"


def test_history_title_component_shows_muted_duplicate_index():
    title = history_title_component({"title": "Llama2-7B on H100", "title_index": 2, "title_duplicate_count": 3})

    assert "Llama2-7B on H100" in _collect_text(title)
    assert "#2" in _collect_text(title)
    assert "history-title-index" in _collect_class_names(title)


def test_progress_run_log_button_switches_to_history_tab():
    assert open_run_log_from_completed_progress(1) == "history"
    assert open_run_log_from_completed_progress(None) is no_update


def test_terminal_progress_count_stays_complete():
    assert progress_count_label({"progress_completed": 0, "progress_total": 1}, terminal=True) == "1 / 1"
    assert progress_count_label({"progress_completed": 3, "progress_total": 8}) == "3 / 8"


def test_job_eta_readout_estimates_remaining_time():
    assert job_eta_readout(
        {"progress_completed": 5, "progress_total": 10, "created_at": "2026-04-27T12:00:00"},
        now=datetime(2026, 4, 27, 12, 10, 0),
    ) == "ETA: ~10m 0s remaining (12:20)"
    assert job_eta_readout({"progress_completed": 0, "progress_total": 10, "created_at": "2026-04-27T12:00:00"}, now=datetime(2026, 4, 27, 12, 10, 0)) == "ETA: calculating"
    assert job_eta_readout({"progress_completed": 10, "progress_total": 10, "created_at": "2026-04-27T12:00:00"}, now=datetime(2026, 4, 27, 12, 10, 0)) == "ETA: complete"


def test_job_eta_readout_uses_local_time_for_utc_job_timestamps(monkeypatch):
    previous_tz = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    if hasattr(time, "tzset"):
        time.tzset()
    try:
        assert job_eta_readout(
            {"progress_completed": 5, "progress_total": 10, "created_at": "2026-04-27T12:00:00+00:00"},
            now=datetime(2026, 4, 27, 5, 10, 0),
        ) == "ETA: ~10m 0s remaining (05:20)"
    finally:
        if previous_tz is None:
            monkeypatch.delenv("TZ", raising=False)
        else:
            monkeypatch.setenv("TZ", previous_tz)
        if hasattr(time, "tzset"):
            time.tzset()


def test_live_status_active_panel_includes_eta(monkeypatch):
    class FakeRunManager:
        def active_job(self):
            return {
                "title": "Active sweep",
                "status": "running",
                "progress_completed": 5,
                "progress_total": 10,
                "created_at": "2026-04-27T12:00:00",
            }

        def last_finished_job(self):
            return None

    monkeypatch.setattr(webui_main, "RUN_MANAGER", FakeRunManager())
    monkeypatch.setattr(webui_main, "get_telemetry", lambda: {"available_ram_gb": 116, "cpu_percent": 1.2})
    monkeypatch.setattr(webui_main, "list_history", lambda: [])
    monkeypatch.setattr(webui_main, "job_eta_readout", lambda job: "ETA: ~10m 0s remaining (12:20)")

    _, _, _, active_panel, _, _ = refresh_status(0, None)

    texts = _collect_text(active_panel)
    assert "ETA: ~10m 0s remaining (12:20)" in texts
    assert "live-status-eta" in _collect_class_names(active_panel)


def test_workspace_tabs_are_callback_addressable():
    tabs = _collect_by_id(create_layout(), "workspace-tabs")

    assert tabs
    assert tabs[0].value == "builder"


def test_hardware_layout_explains_parallelism_topology_mapping():
    layout = create_layout()
    texts = _collect_text(layout)
    selector = _collect_by_id(layout, "parallelism-topology-mode")[0]
    preview_text = _collect_text(webui_main.parallelism_topology_preview("dim1_shared"))
    split_preview_text = _collect_text(webui_main.parallelism_topology_preview("dim1_dim2"))
    topology_select = _collect_by_id(layout, {"type": "net-topology", "index": 0})[0]
    reset_button = _collect_by_id(layout, "reset-paper-derates-button")

    assert selector.data == PP_TOPOLOGY_OPTIONS
    assert selector.value in {"dim1_shared", "dim1_dim2"}
    assert "Hierarchical AstraSim uses Dimension 0 for TP/CP/EP. DP is always written to the outer network dimension." in texts
    assert "Reset to paper defaults" in texts
    assert HELP_TEXT["parallelism_topology"].startswith("Hierarchical AstraSim fixes Dimension 0")
    assert webui_main.disabled_network_dimension_indices("dim1_shared") == {2}
    assert webui_main.disabled_network_dimension_indices("dim1_dim2") == set()
    assert webui_main.disabled_network_dimension_indices("dim2_shared") == set()
    assert "Dimension 0" in preview_text
    assert "Fixed inner transformer/expert axis (intra-DGX box/waferscale)" in preview_text
    assert "Dimension 2" in preview_text
    assert "Dimension 2 off" in preview_text
    assert "Dimension 1 off" not in split_preview_text
    assert {"value": "Mesh2D", "label": "Mesh2D"} in topology_select.data
    assert all(option["value"] != "SuperPOD" for option in topology_select.data)
    assert "SuperPOD is not exposed because support is not reliable enough yet." in HELP_TEXT["network_topology"]
    assert reset_button


def test_reset_paper_derates_uses_calibrated_table_defaults():
    compute, memory, network, net_utils = reset_paper_derates(
        1,
        "A100_PCIe_80GB.yaml",
        [{"index": 0}, {"index": 1}, {"index": 2}],
    )

    assert (compute, memory, network) == (0.60, 0.70, 0.85)
    assert net_utils == [0.85, 0.85, 0.85]


def test_visual_theme_removes_yellow_accents():
    css = Path("webui/app/assets/styles.css").read_text().lower()

    assert "#ffc000" not in css
    assert "#fff4cc" not in css
    assert "ucla-gold" not in css
    assert "nanocad-blue" in css


def test_header_theme_uses_readable_dark_topbar():
    css = Path("webui/app/assets/styles.css").read_text()
    texts = _collect_text(create_layout())

    assert ".brand-orb" not in css
    assert "background: var(--nanocad-blue);" in css
    assert "linear-gradient(96deg, #003b73 0%, #0055a6 54%, #007bb8 100%)" not in css
    assert ".app-header-shell::before {\n  content: \"\";\n  display: none;\n}" in css
    assert ".nanocad-logo-frame" in css
    assert "background: transparent;\n  box-shadow: none;" in css
    assert ".topbar-title" in css
    assert ".topbar-title-block" in css
    assert ".topbar-version" in css
    assert ".app-header-shell .topbar-title" in css
    assert "-webkit-text-fill-color: #ffffff !important;" in css
    assert "font-weight: 900 !important;" in css
    assert "font-style: italic;" in css
    assert "font-size: 13px !important;" in css
    assert ".flow-hover-copy" in css
    assert "color: #dff5fd !important;" in css
    assert ".telemetry-pills" in css
    assert ".telemetry-pills .telemetry-badge" in css
    assert "margin-left: auto;" in css
    assert "min-width: 76px;" in css
    assert "RAPID-LLM Workbench" in texts
    assert "v0.9, last updated 4/27/2026" in texts


def test_progress_bar_css_uses_one_smooth_logo_fill_layer():
    css = Path("webui/app/assets/styles.css").read_text()

    assert ".rapid-progress-track" in css
    assert "--rapid-progress-gradient:" in css
    assert "--rapid-logo-fill-gradient:" in css
    assert "#5feaff" in css
    assert ".rapid-progress-core" in css
    assert ".rapid-progress-core::after" in css
    assert ".rapid-progress-core::before" not in css
    assert ".rapid-progress-core-fill" not in css
    assert ".rapid-progress-word" not in css
    assert ".rapid-progress-complete-action" in css
    assert ".rapid-progress-run-log-button" in css
    assert ".history-title-index" in css
    assert "inset: 0;" in css
    assert "clip-path: inset(0 calc(100% - var(--rapid-progress)) 0 0);" in css
    assert "width 900ms cubic-bezier(0.16, 1, 0.3, 1)" in css
    assert "clip-path 900ms cubic-bezier(0.16, 1, 0.3, 1)" in css
    assert "background-clip: text;" in css
    assert "background: var(--rapid-logo-fill-gradient);" in css
    assert "-webkit-text-stroke: 1px rgba(232, 251, 255, 0.35);" in css
    assert "animation: rapid-progress-stripes 2600ms linear infinite;" in css
    assert "@keyframes rapid-progress-stripes" in css
    assert "@keyframes rapid-progress-shine" not in css
    assert ".live-status-eta" in css


def test_details_shell_has_immediate_loading_message():
    texts = _collect_text(detail_loading_placeholder({"kind": "sweep", "id": "fake"}))

    assert any("Loading sweep details" in text for text in texts)
    assert any("Preparing large sweep tables and plots." in text for text in texts)


def test_user_facing_copy_avoids_meta_references():
    visible_text = "\n".join(_collect_text(create_layout()))
    help_text = "\n".join(HELP_TEXT.values())
    loading_text = "\n".join(_collect_text(detail_loading_placeholder({"kind": "sweep", "id": "fake"})))
    combined = "\n".join([visible_text, help_text, loading_text])

    banned = [
        "".join(parts)
        for parts in [
            ("Model", " tabs show"),
            ("hardware", " tabs show"),
            ("supported", " fields"),
            ("supported", " options"),
            ("unsupported", " YAML"),
            ("Collapsed", " debug view"),
            ("debug", " view"),
            ("UI", " writes"),
            ("Saved", " UI edits"),
            ("Web", " UI"),
            ("screen", " opens"),
            ("prepared", " in the background"),
            ("responsive", " Details rendering"),
            ("currently", " shown"),
            ("active", " top-level"),
            ("local", " workbench"),
            ("Saved", " active model and hardware YAML files"),
        ]
    ]
    for phrase in banned:
        assert phrase not in combined


def test_config_options_css_does_not_clip_stacked_layout():
    css = Path("webui/app/assets/styles.css").read_text()
    stacked_breakpoint = css[css.index("@media (max-width: 75em)") :]
    short_viewport_breakpoint = css[css.index("@media (max-height: 900px)") :]

    assert ".config-options-card {\n  overflow: visible;\n}" in css
    assert ".builder-grid {\n    height: auto;\n    overflow: visible;" in stacked_breakpoint
    assert ".builder-left-scroll {\n    max-height: none;\n    overflow: visible;" in stacked_breakpoint
    assert ".right-rail {\n    position: static;\n    max-height: none;\n    overflow: visible;" in short_viewport_breakpoint


def _large_sweep_detail(case_count: int):
    cases = []
    for index in range(case_count):
        cases.append(
            {
                "case_id": f"case-{index:05d}",
                "label": f"Batch {index}",
                "status": "completed" if index % 19 else "failed",
                "dimension_values": {
                    "model.global_batch_size": index + 1,
                    "hardware.total_gpus": 8 + (index % 8),
                    "model_config": "Llama2-7B.yaml" if index % 2 else "DeepSeekV3_inf_16k.yaml",
                },
                "chosen_candidate": {"tp": 2, "cp": 1, "pp": 1, "dp": 4, "ep": 1},
                "metrics": {
                    "training_time_s": 0.1 + index * 0.001,
                    "num_gpus": 8 + (index % 8),
                    "approx_mfu": 0.42,
                    "total_flops": 1e15 + index,
                    "achieved_flops": 5e14 + index,
                    "achieved_flops_per_gpu": 6e13,
                    "peak_flops_per_gpu": 1e15,
                    "peak_system_flops": 8e15,
                    "memory_exceeded": False,
                    "memory_violation_gb": 0.0,
                },
            }
        )
    return {
        "kind": "sweep",
        "title": f"Large sweep {case_count}",
        "request_record": {
            "payload": {
                "model_preset_id": "Llama2-7B.yaml",
                "hardware_preset_id": "H100.yaml",
                "metric": "training_time_s",
                "x_axis": "model.global_batch_size",
                "series_axis": "hardware.total_gpus",
                "simple": {"run_type": "training", "decode_len": 0, "tp": 1, "cp": 1, "pp": 1, "dp": 8, "ep": 1},
            },
            "preview": {"optimizer_enabled": False, "run_type": "training"},
        },
        "_case_count_total": case_count,
        "_case_count_loaded": case_count,
        "_case_display_mode": "top",
        "_case_sort_metric": "training_time_s",
        "cases": cases,
    }


def test_large_sweep_detail_caps_initial_table_and_plot_rendering():
    detail = _large_sweep_detail(5_000)

    started = perf_counter()
    rendered = render_detail(detail)
    elapsed = perf_counter() - started
    table = _collect_datatables(rendered)[0]
    graph = _collect_graphs(rendered)[0]
    texts = _collect_text(rendered)
    plotted_points = sum(len(trace.x) for trace in graph.figure.data)

    assert len(table.data) == DETAIL_TABLE_ROW_LIMIT
    assert plotted_points <= DETAIL_PLOT_POINT_LIMIT
    assert any("5,000 case rows total" in text for text in texts)
    assert any(f"Table shows first {DETAIL_TABLE_ROW_LIMIT:,}" in text for text in texts)
    assert any(f"Plot samples {DETAIL_PLOT_POINT_LIMIT:,}" in text for text in texts)
    assert elapsed < 3.0


def test_loaded_case_limit_is_reported_in_detail_note():
    detail = _large_sweep_detail(DETAIL_RENDER_CASE_LIMIT)
    detail["_case_count_total"] = 9_000
    detail["_case_count_loaded"] = DETAIL_RENDER_CASE_LIMIT

    texts = _collect_text(render_detail(detail))

    assert any("9,000 case rows total" in text for text in texts)
    assert any(f"Showing top {DETAIL_RENDER_CASE_LIMIT:,} loaded rows by Time / Batch for a faster view." in text for text in texts)


def test_full_load_detail_renders_all_loaded_rows_without_slow_comment():
    detail = _large_sweep_detail(25)
    detail["_case_display_mode"] = "full"

    rendered = render_detail(detail)
    table = _collect_datatables(rendered)[0]
    graph = _collect_graphs(rendered)[0]
    texts = _collect_text(rendered)
    plotted_points = sum(len(trace.x) for trace in graph.figure.data)

    assert len(table.data) == 25
    assert plotted_points == 25
    assert not any("Full load is active; every stored case is loaded and rendering can be slow." in text for text in texts)
    assert any("25 case rows total." in text for text in texts)


def test_detail_display_mode_uses_all_results_label():
    display_mode = _collect_by_id(create_layout(), "detail-display-mode")[0]

    assert {"label": "All results (slow)", "value": "full"} in display_mode.data
    assert {"label": "Full load (slow)", "value": "full"} not in display_mode.data


def test_detail_table_export_payload_uses_uncapped_table_rows():
    detail = _large_sweep_detail(3)

    csv_text, csv_rows = detail_table_export_payload(detail, "csv")
    json_text, json_rows = detail_table_export_payload(detail, "json")

    assert csv_rows == 3
    assert json_rows == 3
    assert "case_id" in csv_text.splitlines()[0]
    assert len(json.loads(json_text)) == 3


def test_detail_termination_rate_uses_summary_counts_and_warns_when_unreliable():
    detail = _large_sweep_detail(10)
    detail["summary_record"] = {"case_count": 100, "completed_case_count": 20}

    texts = _collect_text(render_detail(detail))

    assert any("Early termination rate: 80.0% (80/100 runs)." in text for text in texts)
    assert any("Results are unreliable" in text for text in texts)
    assert any("Increase the timeout" in text for text in texts)


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
                "peak_flops_per_gpu": 2.24e14,
                "peak_system_flops": 1.12e15,
            },
            "primary_metric_value": 1.25,
        },
    }

    tables = _collect_datatables(render_detail(detail))
    texts = _collect_text(render_detail(detail))

    assert tables
    assert any("Early termination rate: 0.0% (0/1 runs)." in text for text in texts)
    assert tables[0].tooltip_header["metric"] == "Metric name emitted by the worker."
    assert "system-wide" in help_for_key("achieved_flops").lower()
    assert help_for_key("approx_mfu") == "Approximate MFU = achieved system FLOPS divided by configured theoretical system FLOPS."
    assert not any("Peak" in row["metric"] for row in tables[0].data)


def test_detail_uses_peak_flops_internally_without_reporting_peak_metrics():
    detail = {
        "kind": "run",
        "title": "Old run",
        "result": {
            "status": "completed",
            "metrics": {
                "training_time_s": 1.0,
                "achieved_flops": 1.0e15,
                "peak_flops_per_gpu": 1.0e15,
                "num_gpus": 8,
            },
            "primary_metric_value": 1.0,
        },
    }

    table = _collect_datatables(render_detail(detail))[0]

    labels = [row["metric"] for row in table.data]
    assert not any("Peak" in label for label in labels)
    assert any(row["metric"] == "Approx. MFU" and row["value"] == "12.50%" for row in table.data)
    assert any(row["metric"] == "Achieved FLOPS / GPU" and row["value"] == "125 TFLOPS" for row in table.data)


def test_sweep_detail_never_reports_peak_flops_metrics():
    detail = _large_sweep_detail(3)

    rows = sweep_rows_from_detail(detail)
    table = _collect_datatables(render_detail(detail))[0]

    assert all(hidden not in row for hidden in DETAIL_HIDDEN_METRIC_KEYS for row in rows)
    assert all(column["id"] not in DETAIL_HIDDEN_METRIC_KEYS for column in table.columns)


def test_sweep_detail_ignores_hidden_peak_metric_axis():
    detail = _large_sweep_detail(3)
    detail["request_record"]["payload"]["metric"] = "peak_system_flops"

    rows = sweep_rows_from_detail(detail)
    _, y_axis, _ = detail_axes(detail, rows)
    graph = _collect_graphs(render_detail(detail))[0]

    assert y_axis == "training_time_s"
    assert graph.figure.layout.yaxis.title.text == "Time / Batch"


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
    assert table.style_data_conditional[0]["if"]["filter_query"] == '{metric} = "Memory Exceeded" && {value} != "No"'
    assert table.style_data_conditional[0]["color"] == "#c1121f"
    assert table.style_data_conditional[0]["fontWeight"] == 900
    assert "every tested candidate exceeded memory" in help_for_key("memory_exceeded")
    assert "least-over-capacity candidate" in help_for_key("memory_exceeded")


def test_sweep_detail_uses_model_config_case_labels_and_status_not_legend():
    detail = {
        "kind": "sweep",
        "title": "Sweep",
        "request_record": {
            "payload": {
                "model_preset_id": "Llama2-7B.yaml",
                "hardware_preset_id": "H100.yaml",
                "metric": "training_time_s",
                "simple": {"run_type": "training", "decode_len": 512, "tp": 1, "cp": 1, "pp": 1, "dp": 8, "ep": 1},
            },
            "preview": {"optimizer_enabled": True, "run_type": "training"},
        },
        "cases": [
            {
                "case_id": "case-0001",
                "label": "Model Config=Llama2-7B.yaml",
                "status": "completed",
                "dimension_values": {"model_config": "Llama2-7B.yaml"},
                "chosen_candidate": {"tp": 2, "cp": 1, "pp": 1, "dp": 4, "ep": 1},
                "metrics": {"training_time_s": 2.0, "num_gpus": 8, "approx_mfu": 0.42, "total_flops": 1e15, "achieved_flops": 5e14, "memory_exceeded": False, "memory_violation_gb": 0.0},
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
    texts = _collect_text(rendered)

    assert table.data[0]["case"] == "case-0001 - Llama2-7B on H100"
    assert table.data[1]["case"] == "case-0002 - Llama3 on H100"
    assert any("Early termination rate: 50.0% (1/2 runs)." in text for text in texts)
    assert any("Many runs terminated early" in text for text in texts)
    assert table.data[0]["parallelism"] == "TP 2 / CP 1 / PP 1 / DP 4 / EP 1"
    assert table.data[0]["model.decode_len"] == "0"
    assert table.tooltip_header["model.decode_len"].startswith("Generated token count")
    assert all("model.decode_len" not in str(trace.hovertemplate) for trace in graph.figure.data)
    assert table.style_filter == DATA_TABLE_STYLE_FILTER
    assert table.style_data_conditional[0]["if"]["column_id"] == "memory_exceeded"
    assert table.style_data_conditional[0]["color"] == "#c1121f"
    assert table.data[1]["training_time_s"] == "0.00 us"
    assert [column["id"] for column in table.columns[:8]] == [
        "case",
        "label",
        "memory_exceeded",
        "training_time_s",
        "num_gpus",
        "approx_mfu",
        "total_flops",
        "achieved_flops",
    ]
    assert all(trace.name not in {"completed", "failed", "timed_out"} for trace in graph.figure.data)


def test_detail_rows_include_decode_length_for_inference_and_training():
    inference_detail = {
        "request_record": {
            "payload": {"simple": {"run_type": "inference", "decode_len": 64}},
            "preview": {"run_type": "inference"},
        },
        "cases": [
            {
                "case_id": "case-0001",
                "label": "decode 128",
                "status": "completed",
                "dimension_values": {"model.decode_len": 128},
                "metrics": {"prefill_time_s": 1.0},
            },
            {
                "case_id": "case-0002",
                "label": "base decode",
                "status": "completed",
                "dimension_values": {},
                "metrics": {"prefill_time_s": 1.0},
            },
        ],
    }
    training_detail = {
        "request_record": {
            "payload": {"simple": {"run_type": "training", "decode_len": 512}},
            "preview": {"run_type": "training"},
        },
        "cases": [{"case_id": "case-0001", "label": "train", "status": "completed", "dimension_values": {}, "metrics": {"training_time_s": 1.0}}],
    }

    inference_rows = sweep_rows_from_detail(inference_detail)
    training_rows = sweep_rows_from_detail(training_detail)

    assert [row["model.decode_len"] for row in inference_rows] == [128, 64]
    assert training_rows[0]["model.decode_len"] == 0
