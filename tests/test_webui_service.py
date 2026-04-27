from __future__ import annotations

import json
from pathlib import Path

import config as rapid_config
from webui.service import core


def _isolate_workspace(monkeypatch, tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(core, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(core, "RUNS_ROOT", workspace / "runs")
    monkeypatch.setattr(core, "SWEEPS_ROOT", workspace / "sweeps")
    monkeypatch.setattr(core, "CONFIGS_ROOT", workspace / "configs")
    monkeypatch.setattr(core, "LOCKS_ROOT", workspace / "locks")
    monkeypatch.setattr(core, "LOGS_ROOT", workspace / "logs")
    monkeypatch.setattr(core, "DB_ROOT", workspace / "db")
    monkeypatch.setattr(core, "ARTIFACTS_ROOT", workspace / "artifacts")
    monkeypatch.setattr(core, "ACTIVE_JOB_LOCK", workspace / "locks" / "active_job.lock")
    monkeypatch.setattr(core, "SCHEMA_VERSION_PATH", workspace / "schema_version.json")
    core.ensure_workspace()
    return workspace


def _payload() -> dict:
    model_id = "Llama2-7B.yaml"
    hardware_id = "H100_SXM5_80GB_base.yaml"
    defaults = core.build_form_defaults(model_id, hardware_id)
    return {
        "model_preset_id": model_id,
        "hardware_preset_id": hardware_id,
        "run_mode": "sweep",
        "optimize_parallelism": False,
        "optimizer_preset": "Fast",
        "use_raw_yaml": False,
        "model_yaml_text": defaults["model_yaml"],
        "hardware_yaml_text": defaults["hardware_yaml"],
        "simple": defaults["simple"] | {"run_type": defaults["run_type"]},
        "advanced": defaults["advanced"],
        "network_dimensions": defaults["network_dimensions"],
        "dimensions": [],
        "metric": core.get_default_metric_for_run_type(defaults["run_type"]),
        "x_axis": None,
        "series_axis": None,
        "worker_count": 1,
        "timeout_seconds": 10,
    }


def test_default_launch_preview_is_valid(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)

    preview = core.build_launch_preview(_payload())

    assert preview["ok"] is True
    assert preview["top_level_case_count"] == 1
    assert preview["total_invocations"] == 1
    assert preview["top_level_cases"][0]["case_id"] == "case-0001"


def test_worker_python_path_preserves_virtualenv_symlink():
    assert ".venv" in str(core.PYTHON_BIN)


def test_workspace_configs_are_seeded_as_editable_defaults(monkeypatch, tmp_path):
    workspace = _isolate_workspace(monkeypatch, tmp_path)

    model_records = core.list_presets("models")
    hardware_records = core.list_presets("hardware")

    assert any(item["id"] == "Llama2-7B.yaml" for item in model_records)
    assert any(item["id"] == "H100_SXM5_80GB_base.yaml" for item in hardware_records)
    assert all("case-" not in item["id"].lower() for item in model_records + hardware_records)
    assert all(Path(item["path"]).is_relative_to(workspace / "configs") for item in model_records + hardware_records)


def test_network_dimensions_use_generic_labels(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)

    defaults = core.build_form_defaults("Llama2-7B.yaml", "H100_SXM5_80GB_base.yaml")

    assert [item["label"] for item in defaults["network_dimensions"]] == [
        f"Dimension {idx}" for idx in range(len(defaults["network_dimensions"]))
    ]


def test_form_edits_are_saved_to_editable_yaml_files(monkeypatch, tmp_path):
    workspace = _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["simple"]["seq_len"] = 4096
    payload["simple"]["batch_size"] = 64
    payload["simple"]["tp"] = 4
    payload["simple"]["compute_derate"] = 0.72
    payload["simple"]["hbm_gb"] = 96
    payload["network_dimensions"][0]["util"] = 0.81

    model, hardware, errors = core.save_config_edits_from_payload(payload)

    assert errors == []
    assert model is not None
    assert hardware is not None
    model_yaml = core._yaml_load(workspace / "configs" / "models" / "Llama2-7B.yaml")
    hardware_yaml = core._yaml_load(workspace / "configs" / "hardware" / "H100_SXM5_80GB_base.yaml")
    assert model_yaml["model_param"]["seq_len"] == 4096
    assert model_yaml["model_param"]["global_batch_size"] == 64
    assert hardware_yaml["parallelism"]["tp"] == 4
    assert hardware_yaml["tech_param"]["core"]["util"] == 0.72
    assert hardware_yaml["tech_param"]["DRAM"]["size"] == "96 GB"
    assert hardware_yaml["network"]["dimensions"][0]["topology"]["util"] == 0.81


def test_config_files_can_be_created_and_renamed(monkeypatch, tmp_path):
    workspace = _isolate_workspace(monkeypatch, tmp_path)

    created = core.create_config_copy("models", "Llama2-7B.yaml", "my llama copy")
    renamed = core.rename_config_file("models", created, "renamed_llama")

    assert created == "my_llama_copy.yaml"
    assert renamed == "renamed_llama.yaml"
    assert not (workspace / "configs" / "models" / created).exists()
    assert (workspace / "configs" / "models" / renamed).exists()
    assert any(item["id"] == renamed for item in core.list_presets("models"))


def test_config_file_actions_reject_case_names_and_duplicates(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)

    try:
        core.create_config_copy("models", "Llama2-7B.yaml", "case-a")
    except ValueError as exc:
        assert "reserved" in str(exc)
    else:
        raise AssertionError("case-style config name should be rejected")

    try:
        core.rename_config_file("hardware", "H100_SXM5_80GB_base.yaml", "A100_SXM4_80GB_base.yaml")
    except FileExistsError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("renaming over an existing config should fail")


def test_use_astrasim_simple_option_forces_hierarchical_backend(monkeypatch, tmp_path):
    workspace = _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["simple"]["use_astrasim"] = True
    payload["advanced"]["execution_backend"] = "analytical"
    payload["advanced"]["execution_mode"] = "full_astrasim_flattened"

    _, hardware, errors = core.save_config_edits_from_payload(payload)

    assert errors == []
    assert hardware is not None
    hardware_yaml = core._yaml_load(workspace / "configs" / "hardware" / "H100_SXM5_80GB_base.yaml")
    assert hardware_yaml["execution_backend"]["model"] == "astra"
    assert hardware_yaml["execution_backend"]["astra"]["mode"] == "full_astrasim_hierarchical"

    payload["simple"]["use_astrasim"] = False
    _, hardware, errors = core.save_config_edits_from_payload(payload)

    assert errors == []
    assert hardware is not None
    hardware_yaml = core._yaml_load(workspace / "configs" / "hardware" / "H100_SXM5_80GB_base.yaml")
    assert hardware_yaml["execution_backend"]["model"] == "analytical"
    assert hardware_yaml["execution_backend"]["astra"]["mode"] == "full_astrasim_hierarchical"


def test_model_type_advanced_option_is_saved(monkeypatch, tmp_path):
    workspace = _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["advanced"]["model_type"] = "deepseek_v3"

    model, _, errors = core.save_config_edits_from_payload(payload)

    assert errors == []
    assert model is not None
    model_yaml = core._yaml_load(workspace / "configs" / "models" / "Llama2-7B.yaml")
    assert model_yaml["model_param"]["model_type"] == "deepseek_v3"


def test_low_precision_strings_use_expected_byte_widths():
    assert rapid_config._coerce_precision_value("mxfp4") == 4.25 / 8.0
    assert rapid_config._coerce_precision_value("int4") == 0.5


def test_tensor_format_advanced_option_saves_and_validates_low_precision(monkeypatch, tmp_path):
    workspace = _isolate_workspace(monkeypatch, tmp_path)

    for tensor_format in ("mxfp4", "int4"):
        payload = _payload()
        payload["advanced"]["tensor_format"] = tensor_format

        _, hardware, errors = core.save_config_edits_from_payload(payload)
        preview = core.build_launch_preview(payload)

        assert errors == []
        assert hardware is not None
        assert preview["ok"] is True
        hardware_yaml = core._yaml_load(workspace / "configs" / "hardware" / "H100_SXM5_80GB_base.yaml")
        assert hardware_yaml["sw_param"]["precision"]["tensor_format"] == tensor_format
        assert preview["top_level_cases"][0]["hardware"]["sw_param"]["precision"]["tensor_format"] == tensor_format


def test_all_supported_precision_overrides_are_saved(monkeypatch, tmp_path):
    workspace = _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["advanced"] |= {
        "tensor_format": "fp8",
        "precision_kv_cache": "mxfp4",
        "precision_parameters": "int4",
        "precision_gradients": "fp16",
        "precision_grad_communication": "as_tensor_format",
        "precision_optimizer_states": "fp32",
        "precision_stats": "bf16",
        "precision_master_parameters": "0",
    }

    _, hardware, errors = core.save_config_edits_from_payload(payload)
    preview = core.build_launch_preview(payload)

    assert errors == []
    assert hardware is not None
    assert preview["ok"] is True
    precision = core._yaml_load(workspace / "configs" / "hardware" / "H100_SXM5_80GB_base.yaml")["sw_param"]["precision"]
    assert precision["tensor_format"] == "fp8"
    assert precision["kv_cache"] == "mxfp4"
    assert precision["parameters"] == "int4"
    assert precision["gradients"] == "fp16"
    assert precision["grad_communication"] == "as_tensor_format"
    assert precision["optimizer_states"] == "fp32"
    assert precision["stats"] == "bf16"
    assert precision["master_parameters"] == 0.0


def test_inference_forces_grad_accumulation_to_one(monkeypatch, tmp_path):
    workspace = _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["simple"]["run_type"] = "inference"
    payload["simple"]["grad_accum"] = 8

    model, _, errors = core.save_config_edits_from_payload(payload)

    assert errors == []
    assert model is not None
    model_yaml = core._yaml_load(workspace / "configs" / "models" / "Llama2-7B.yaml")
    assert model_yaml["model_param"]["run_type"] == "inference"
    assert model_yaml["model_param"]["gradient_accumulation_steps"] == 1


def test_form_edits_are_saved_only_to_active_editor_configs(monkeypatch, tmp_path):
    workspace = _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["simple"]["seq_len"] = 2048
    payload["simple"]["hbm_gb"] = 72
    payload["simple"]["compute_derate"] = 0.61

    _, _, errors = core.save_config_edits_from_payload(payload)

    assert errors == []
    active_model = core._yaml_load(workspace / "configs" / "models" / "Llama2-7B.yaml")
    inactive_model = core._yaml_load(workspace / "configs" / "models" / "Llama3.1-70B_2d_train.yaml")
    active_hardware = core._yaml_load(workspace / "configs" / "hardware" / "H100_SXM5_80GB_base.yaml")
    inactive_hardware = core._yaml_load(workspace / "configs" / "hardware" / "A100_SXM4_80GB_base.yaml")
    assert active_model["model_param"]["seq_len"] == 2048
    assert inactive_model["model_param"]["seq_len"] == 32768
    assert active_hardware["tech_param"]["DRAM"]["size"] == "72 GB"
    assert active_hardware["tech_param"]["core"]["util"] == 0.61
    assert inactive_hardware["tech_param"]["DRAM"]["size"] == "80 GB"
    assert inactive_hardware["tech_param"]["core"]["util"] == 1.0


def test_config_comparison_dimensions_expand_from_run_setup(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["dimensions"] = [
        {
            "field_key": "model_config",
            "mode": "values",
            "config_values": ["Llama2-7B.yaml", "Llama3.1-70B_2d_train.yaml"],
        },
        {
            "field_key": "hardware_config",
            "mode": "values",
            "config_values": ["H100_SXM5_80GB_base.yaml", "H100_SXM5_80GB.yaml"],
        },
    ]

    preview = core.build_launch_preview(payload)

    assert preview["ok"] is True
    assert preview["top_level_case_count"] == 4
    labels = [case["label"] for case in preview["top_level_cases"]]
    assert any("Llama3.1-70B_2d_train.yaml" in label for label in labels)
    assert any("H100_SXM5_80GB.yaml" in label for label in labels)


def test_active_editor_overrides_do_not_smear_across_selected_model_cases(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["simple"]["seq_len"] = 4096
    payload["dimensions"] = [
        {
            "field_key": "model_config",
            "mode": "values",
            "config_values": ["Llama2-7B.yaml", "Llama3.1-70B_2d_train.yaml"],
        }
    ]

    preview = core.build_launch_preview(payload)

    assert preview["ok"] is True
    seq_by_model = {
        case["dimension_values"]["model_config"]: case["model"]["model_param"]["seq_len"]
        for case in preview["top_level_cases"]
    }
    assert seq_by_model["Llama2-7B.yaml"] == 4096
    assert seq_by_model["Llama3.1-70B_2d_train.yaml"] == 32768


def test_total_gpu_sweep_scales_data_parallelism_when_not_optimizing(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["simple"]["tp"] = 8
    payload["simple"]["cp"] = 1
    payload["simple"]["pp"] = 1
    payload["simple"]["ep"] = 1
    payload["optimize_parallelism"] = False
    payload["dimensions"] = [
        {
            "field_key": "hardware.total_gpus",
            "mode": "list",
            "list_text": "8, 16",
        }
    ]

    preview = core.build_launch_preview(payload)

    assert preview["ok"] is True
    assert [
        case["hardware"]["parallelism"]["train"]["dp"]
        for case in preview["top_level_cases"]
    ] == [1, 2]


def test_optimized_total_gpu_sweep_uses_sweep_values_not_simple_total(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["simple"]["total_gpus"] = 999
    payload["optimize_parallelism"] = True
    payload["dimensions"] = [
        {
            "field_key": "hardware.total_gpus",
            "mode": "list",
            "list_text": "8, 16",
        }
    ]

    preview = core.build_launch_preview(payload)

    assert preview["ok"] is True
    assert preview["optimizer_enabled"] is True
    assert [case["target_total_gpus"] for case in preview["top_level_cases"]] == [8, 16]
    assert all(item["count"] > 0 for item in preview["candidate_breakdown"])


def test_raw_parallelism_axis_sweeps_are_rejected(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["dimensions"] = [{"field_key": "hardware.parallelism.tp", "mode": "list", "list_text": "1, 2"}]

    preview = core.build_launch_preview(payload)

    assert preview["ok"] is False
    assert "Unsupported sweep field: hardware.parallelism.tp" in preview["errors"]


def test_vit_model_type_rejects_sequence_length_sweeps(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["advanced"]["model_type"] = "vit"
    payload["advanced"]["model_mode"] = "VIT"
    payload["dimensions"] = [
        {
            "field_key": "model.seq_len",
            "mode": "list",
            "list_text": "1024, 2048",
        }
    ]

    preview = core.build_launch_preview(payload)

    assert preview["ok"] is False
    assert any("ViT model families derive sequence length" in error for error in preview["errors"])


def test_numeric_range_sweep_expands_without_preset_values(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["dimensions"] = [
        {
            "field_key": "model.global_batch_size",
            "mode": "range",
            "start": 32,
            "end": 96,
            "step": 32,
            "list_text": "",
            "config_values": ["unused.yaml"],
        }
    ]

    preview = core.build_launch_preview(payload)

    assert preview["ok"] is True
    assert preview["top_level_case_count"] == 3
    assert [case["dimension_values"]["model.global_batch_size"] for case in preview["top_level_cases"]] == [32, 64, 96]


def test_worst_case_wall_clock_is_divided_by_workers(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["worker_count"] = 2
    payload["timeout_seconds"] = 10
    payload["dimensions"] = [
        {
            "field_key": "model.global_batch_size",
            "mode": "range",
            "start": 32,
            "end": 96,
            "step": 32,
        }
    ]

    preview = core.build_launch_preview(payload)

    assert preview["ok"] is True
    assert preview["total_invocations"] == 3
    assert preview["worst_case_wall_clock_s"] == 20


def test_single_run_mode_ignores_sweep_dimensions_and_optimizer(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["run_mode"] = "single"
    payload["optimize_parallelism"] = True
    payload["dimensions"] = [
        {
            "field_key": "model.seq_len",
            "mode": "list",
            "list_text": "1024, 2048",
        }
    ]

    preview = core.build_launch_preview(payload)

    assert preview["ok"] is True
    assert preview["optimizer_enabled"] is False
    assert preview["top_level_case_count"] == 1
    assert preview["total_invocations"] == 1
    assert any("Single launch mode ignores" in warning for warning in preview["warnings"])


def test_scalar_sweep_expands_shared_dimensions(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    seq_len = int(payload["simple"]["seq_len"])
    payload["dimensions"] = [
        {
            "field_key": "model.seq_len",
            "mode": "list",
            "list_text": f"{seq_len}, {seq_len * 2}",
        }
    ]

    preview = core.build_launch_preview(payload)

    assert preview["ok"] is True
    assert preview["top_level_case_count"] == 2
    assert [case["dimension_values"]["model.seq_len"] for case in preview["top_level_cases"]] == [
        seq_len,
        seq_len * 2,
    ]


def test_stale_active_lock_is_removed(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    core.ACTIVE_JOB_LOCK.write_text(json.dumps({"job_id": "old", "pid": 99999999}))
    manager = core.RunManager()

    assert manager._existing_active_lock() is None
    assert not core.ACTIVE_JOB_LOCK.exists()


def test_saved_plot_paths_are_standardized_and_incremented(monkeypatch, tmp_path):
    workspace = _isolate_workspace(monkeypatch, tmp_path)
    job_root = core.SWEEPS_ROOT / "sweep-test"
    job_root.mkdir(parents=True)

    first = Path(core.save_plot_html("sweep", "sweep-test", "Llama2-7B on H100 Sweep", "<html>one</html>"))
    second = Path(core.save_plot_html("sweep", "sweep-test", "Llama2-7B on H100 Sweep", "<html>two</html>"))

    assert first == workspace / "sweeps" / "sweep-test" / "plots" / "llama2-7b-on-h100-sweep-plot-001.html"
    assert second == workspace / "sweeps" / "sweep-test" / "plots" / "llama2-7b-on-h100-sweep-plot-002.html"
    assert first.read_text() == "<html>one</html>"
    assert second.read_text() == "<html>two</html>"


def test_single_run_writes_snapshots_and_artifact_manifest(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    preview = core.build_launch_preview(_payload())
    job_root = core.RUNS_ROOT / "run-test"
    job_root.mkdir(parents=True)
    manager = core.RunManager()

    def fake_execute(job_root, case_id, model_dict, hardware_dict, timeout_seconds, **kwargs):
        case_root = job_root / "artifacts" / case_id
        case_root.mkdir(parents=True)
        (case_root / "stdout.log").write_text("ok\n")
        return {
            "case_id": case_id,
            "top_case_id": case_id,
            "label": case_id,
            "status": "completed",
            "metrics": {"training_time_s": 1.25, "num_gpus": 8},
            "warnings": [],
            "error": None,
            "primary_metric_label": "Time / Batch",
            "primary_metric_value": 1.25,
        }

    monkeypatch.setattr(manager, "_execute_worker_case", fake_execute)
    summary = manager._run_single(job_root, preview)
    core._json_dump(job_root / "summary.json", summary)
    manager._write_artifacts_manifest(job_root)

    manifest = json.loads((job_root / "artifacts.json").read_text())
    artifact_paths = {item["path"] for item in manifest["artifacts"]}
    assert "model_resolved.yaml" in artifact_paths
    assert "hardware_resolved.yaml" in artifact_paths
    assert "metrics.json" in artifact_paths
    assert "artifacts/case-0001/stdout.log" in artifact_paths
