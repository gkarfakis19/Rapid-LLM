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
    hardware_id = "H100_SXM5_80GB.yaml"
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


def test_inference_metric_options_default_to_tpot(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["simple"]["run_type"] = "inference"
    payload["metric"] = None

    preview = core.build_launch_preview(payload)

    assert core.get_default_metric_for_run_type("inference") == "decode_throughput_tok_s"
    assert core.get_metric_options("inference") == [
        {"value": "decode_throughput_tok_s", "label": "Throughput (TPOT)"},
        {"value": "ttft_s", "label": "TTFT"},
        {"value": "total_inference_time_s", "label": "Time / Batch"},
    ]
    assert preview["ok"] is True
    assert preview["metric"] == "decode_throughput_tok_s"


def test_metric_options_are_run_type_scoped(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    inference_payload = _payload()
    inference_payload["simple"]["run_type"] = "inference"
    inference_payload["metric"] = "approx_mfu"
    training_payload = _payload()
    training_payload["metric"] = "ttft_s"

    inference_preview = core.build_launch_preview(inference_payload)
    training_preview = core.build_launch_preview(training_payload)

    assert {item["value"] for item in core.get_metric_options("training")} == {"training_time_s", "approx_mfu"}
    assert "ttft_s" not in {item["value"] for item in core.get_metric_options("training")}
    assert inference_preview["metric"] == "decode_throughput_tok_s"
    assert training_preview["metric"] == "training_time_s"


def test_parallelism_optimizer_prefers_memory_fitting_result():
    memory_exceeded_fast = {
        "status": "completed",
        "metrics": {"training_time_s": 1.0, "memory_exceeded": True, "memory_violation_gb": 8.0},
        "candidate": {"tp": 8},
    }
    memory_fitting_slow = {
        "status": "completed",
        "metrics": {"training_time_s": 2.0, "memory_exceeded": False, "memory_violation_gb": 0.0},
        "candidate": {"tp": 4},
    }

    chosen = core.pick_best_optimized_result([memory_exceeded_fast, memory_fitting_slow], "training_time_s")

    assert chosen is memory_fitting_slow


def test_parallelism_optimizer_reports_when_every_candidate_exceeds_memory_and_picks_least_violation():
    candidates = [
        {
            "status": "completed",
            "metrics": {"training_time_s": 2.0, "memory_exceeded": True, "memory_violation_gb": 10.0},
            "candidate": {"tp": 2},
        },
        {
            "status": "completed",
            "metrics": {"training_time_s": 1.0, "memory_exceeded": True, "memory_violation_gb": 20.0},
            "candidate": {"tp": 4},
            "warnings": ["existing"],
        },
    ]

    chosen = core.pick_best_optimized_result(candidates, "training_time_s")

    assert chosen["candidate"] == {"tp": 2}
    assert chosen["metrics"]["memory_violation_gb"] == 10.0
    assert chosen["warnings"] == ["Every tested parallelism candidate exceeded memory capacity."]
    assert candidates[1]["warnings"] == ["existing"]


def test_worker_python_path_preserves_virtualenv_symlink():
    assert ".venv" in str(core.PYTHON_BIN)


def test_yaml_load_cache_returns_isolated_copies_and_refreshes_on_file_change(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("value: 1\n")
    core._YAML_CACHE.clear()

    first = core._yaml_load(path)
    first["value"] = 99

    assert core._yaml_load(path)["value"] == 1

    path.write_text("value: 22\nextra: true\n")

    assert core._yaml_load(path) == {"value": 22, "extra": True}


def test_workspace_configs_are_seeded_as_editable_defaults(monkeypatch, tmp_path):
    workspace = _isolate_workspace(monkeypatch, tmp_path)

    model_records = core.list_presets("models")
    hardware_records = core.list_presets("hardware")

    assert any(item["id"] == "Llama2-7B.yaml" for item in model_records)
    assert any(item["id"] == "H100_SXM5_80GB.yaml" for item in hardware_records)
    assert {item["id"] for item in hardware_records} == {
        "A100_PCIe_80GB.yaml",
        "A100_SXM4_80GB.yaml",
        "H100_SXM5_80GB.yaml",
    }
    assert all("case-" not in item["id"].lower() for item in model_records + hardware_records)
    assert all(Path(item["path"]).is_relative_to(workspace / "configs") for item in model_records + hardware_records)
    assert len(model_records) >= 20


def test_curated_hardware_defaults_use_calibrated_derates(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    expected = {
        "A100_PCIe_80GB.yaml": (0.60, 0.70, 0.85),
        "A100_SXM4_80GB.yaml": (0.90, 0.70, 0.80),
        "H100_SXM5_80GB.yaml": (0.56, 0.80, 0.85),
    }

    for hardware_id, (compute, memory, communication) in expected.items():
        hardware = core.load_preset("hardware", hardware_id)
        network_utils = [dim["topology"]["util"] for dim in hardware["network"]["dimensions"]]
        paper_defaults = core.paper_derate_defaults_for_hardware(hardware_id)
        assert hardware["sw_param"]["kernel_launch_overhead"] == "6e-6"
        assert hardware["tech_param"]["core"]["util"] == compute
        assert hardware["tech_param"]["DRAM"]["util"] == memory
        assert network_utils == [communication, communication, communication]
        assert paper_defaults == {"compute": compute, "memory": memory, "communication": communication}


def test_last_ui_state_scratchpad_is_limited_and_clearable(monkeypatch, tmp_path):
    workspace = _isolate_workspace(monkeypatch, tmp_path)

    for index in range(core.LAST_UI_STATE_LIMIT + 2):
        core.save_last_ui_state(
            {
                "model_run_configs": ["Llama2-7B.yaml"],
                "hardware_run_configs": ["H100_SXM5_80GB.yaml"],
                "model_preset": "Llama2-7B.yaml",
                "hardware_preset": "H100_SXM5_80GB.yaml",
                "active_config_tab": "models::Llama2-7B.yaml",
                "run_mode": "sweep",
                "optimize_parallelism": bool(index % 2),
                "optimizer_preset": "Fast",
                "sweep_rows": [
                    {"field": "model.global_batch_size", "mode": "range", "list_text": "1,2,3", "config_values": [], "start": index, "end": index + 4, "step_or_points": 2}
                ],
                "metric": "training_time_s",
                "x_axis": "model.global_batch_size",
                "series_axis": None,
                "worker_count": index + 1,
                "timeout_seconds": 180,
            }
        )

    scratchpad = json.loads((workspace / "scratch" / core.LAST_UI_STATE_FILENAME).read_text())
    loaded = core.load_last_ui_state()

    assert len(scratchpad["recent"]) == core.LAST_UI_STATE_LIMIT
    assert loaded["worker_count"] == core.LAST_UI_STATE_LIMIT + 2
    assert loaded["sweep_rows"][0]["start"] == core.LAST_UI_STATE_LIMIT + 1
    assert len(loaded["sweep_rows"]) == 3

    core.clear_last_ui_state()

    assert core.load_last_ui_state() == {}
    assert not (workspace / "scratch" / core.LAST_UI_STATE_FILENAME).exists()


def test_network_dimensions_use_generic_labels(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)

    defaults = core.build_form_defaults("Llama2-7B.yaml", "H100_SXM5_80GB.yaml")

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
    hardware_yaml = core._yaml_load(workspace / "configs" / "hardware" / "H100_SXM5_80GB.yaml")
    assert model_yaml["model_param"]["seq_len"] == 4096
    assert model_yaml["model_param"]["global_batch_size"] == 64
    assert hardware_yaml["parallelism"]["tp"] == 4
    assert hardware_yaml["tech_param"]["core"]["util"] == 0.72
    assert hardware_yaml["tech_param"]["DRAM"]["size"] == "96 GB"
    assert hardware_yaml["network"]["dimensions"][0]["topology"]["util"] == 0.81


def test_parallelism_topology_mapping_is_saved_to_hardware_yaml(monkeypatch, tmp_path):
    workspace = _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()

    assert payload["advanced"]["pp_network_dimension"] == "dim1_shared"

    _, hardware, errors = core.save_config_edits_from_payload(payload)

    assert errors == []
    assert hardware is not None
    hardware_yaml = core._yaml_load(workspace / "configs" / "hardware" / "H100_SXM5_80GB.yaml")
    dimensions = hardware_yaml["network"]["dimensions"]
    assert dimensions[0]["parallelisms"] == ["tp", "cp", "ep"]
    assert dimensions[1]["parallelisms"] == ["pp", "dp"]
    assert dimensions[2]["parallelisms"] == []

    payload["advanced"]["pp_network_dimension"] = "dim1_dim2"
    _, hardware, errors = core.save_config_edits_from_payload(payload)

    assert errors == []
    assert hardware is not None
    hardware_yaml = core._yaml_load(workspace / "configs" / "hardware" / "H100_SXM5_80GB.yaml")
    dimensions = hardware_yaml["network"]["dimensions"]
    assert dimensions[0]["parallelisms"] == ["tp", "cp", "ep"]
    assert dimensions[1]["parallelisms"] == ["pp"]
    assert dimensions[2]["parallelisms"] == ["dp"]

    payload["advanced"]["pp_network_dimension"] = "dim2_shared"
    _, hardware, errors = core.save_config_edits_from_payload(payload)

    assert errors == []
    assert hardware is not None
    hardware_yaml = core._yaml_load(workspace / "configs" / "hardware" / "H100_SXM5_80GB.yaml")
    dimensions = hardware_yaml["network"]["dimensions"]
    assert dimensions[0]["parallelisms"] == ["tp", "cp", "ep"]
    assert dimensions[1]["parallelisms"] == ["pp"]
    assert dimensions[2]["parallelisms"] == ["dp"]


def test_superpod_workspace_topology_loads_as_supported_webui_choice(monkeypatch, tmp_path):
    workspace = _isolate_workspace(monkeypatch, tmp_path)
    hardware_path = workspace / "configs" / "hardware" / "H100_SXM5_80GB.yaml"
    hardware_yaml = core._yaml_load(hardware_path)
    hardware_yaml["network"]["dimensions"][1]["topology"]["type"] = "SuperPOD"
    hardware_path.write_text(core._yaml_dump(hardware_yaml))

    defaults = core.build_form_defaults("Llama2-7B.yaml", "H100_SXM5_80GB.yaml")

    assert defaults["network_dimensions"][1]["topology_type"] == "Ring"


def test_stale_superpod_payload_forces_h100_variant(monkeypatch, tmp_path):
    workspace = _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["network_dimensions"][1]["topology_type"] = "SuperPOD"

    _, hardware, errors = core.save_config_edits_from_payload(payload)

    assert errors == []
    assert hardware is not None
    hardware_yaml = core._yaml_load(workspace / "configs" / "hardware" / "H100_SXM5_80GB.yaml")
    topology = hardware_yaml["network"]["dimensions"][1]["topology"]
    assert topology["type"] == "SuperPOD"
    assert topology["superpod_variant"] == "h100"
    assert topology["leaf_size"] == 1


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
        core.rename_config_file("hardware", "H100_SXM5_80GB.yaml", "A100_SXM4_80GB.yaml")
    except FileExistsError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("renaming over an existing config should fail")


def test_huggingface_model_references_are_parsed():
    assert core.parse_huggingface_model_reference("https://huggingface.co/Qwen/Qwen2.5-7B") == ("Qwen/Qwen2.5-7B", "main")
    assert core.parse_huggingface_model_reference("https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/config.json") == ("Qwen/Qwen2.5-7B", "main")
    assert core.parse_huggingface_model_reference("https://huggingface.co/Qwen/Qwen2.5-7B/blob/refs/pr/1/config.json") == ("Qwen/Qwen2.5-7B", "refs/pr/1")
    assert core.parse_huggingface_model_reference("Qwen/Qwen2.5-7B@refs/pr/1") == ("Qwen/Qwen2.5-7B", "refs/pr/1")


def test_huggingface_model_references_reject_unsafe_inputs():
    for reference in [
        "http://huggingface.co/Qwen/Qwen2.5-7B",
        "https://huggingface.co.evil.test/Qwen/Qwen2.5-7B",
        "https://huggingface.co/Qwen/Qwen2.5-7B?download=1",
        "Qwen/Qwen2.5-7B?x=1",
        "Qwen/../../secret",
        "Qwen/Qwen2.5-7B@refs/../main",
    ]:
        try:
            core.parse_huggingface_model_reference(reference)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe Hugging Face reference was accepted: {reference}")


def test_huggingface_import_rejects_unsupported_model_type(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)

    class FakeConverter:
        @staticmethod
        def _fetch_hf_config(model_id, revision="main"):
            del model_id, revision
            return {"model_type": "bert"}

        @staticmethod
        def _infer_model_type(model_type):
            raise AssertionError(f"unsupported model_type should be rejected before inference: {model_type}")

    monkeypatch.setattr(core, "_load_hf_to_config_module", lambda: FakeConverter)

    try:
        core.create_model_config_from_huggingface("bert-base-uncased", "bert")
    except ValueError as exc:
        assert "Unsupported Hugging Face model_type" in str(exc)
    else:
        raise AssertionError("unsupported Hugging Face model_type should fail")


def test_huggingface_import_rejects_unsupported_active_attention_features(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)

    class FakeConverter:
        @staticmethod
        def _fetch_hf_config(model_id, revision="main"):
            del model_id, revision
            return {"model_type": "qwen2", "use_sliding_window": True}

        @staticmethod
        def _infer_model_type(model_type):
            assert model_type == "qwen2"
            return "llama", "qwen2"

        @staticmethod
        def _build_yaml_config(cfg, args, model_type):
            raise AssertionError("unsupported active sliding-window attention should be rejected before YAML generation")

    monkeypatch.setattr(core, "_load_hf_to_config_module", lambda: FakeConverter)

    try:
        core.create_model_config_from_huggingface("Qwen/Qwen2.5-7B", "qwen")
    except ValueError as exc:
        assert "sliding-window attention is not modeled" in str(exc)
    else:
        raise AssertionError("unsupported active Hugging Face attention feature should fail")


def test_huggingface_import_creates_editable_model_config(monkeypatch, tmp_path):
    workspace = _isolate_workspace(monkeypatch, tmp_path)

    class FakeConverter:
        @staticmethod
        def _fetch_hf_config(model_id, revision="main"):
            assert model_id == "Qwen/Qwen2.5-7B"
            assert revision == "main"
            return {"model_type": "qwen2"}

        @staticmethod
        def _infer_model_type(model_type):
            assert model_type == "qwen2"
            return "llama", "qwen2"

        @staticmethod
        def _build_yaml_config(cfg, args, model_type):
            assert cfg == {"model_type": "qwen2"}
            assert args.run_type == "training"
            assert args.global_batch_size == 1
            assert args.decode_len == 0
            assert args.seq_len is None
            assert model_type == "llama"
            return {
                "model_param": {
                    "mode": "LLM",
                    "run_type": args.run_type,
                    "model_type": model_type,
                    "global_batch_size": args.global_batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "seq_len": 32768,
                    "decode_len": args.decode_len,
                    "hidden_dim": 3584,
                    "attention": {"attention_type": "gqa", "num_heads": 28, "kv_heads": 4},
                    "intermediate_size": 18944,
                    "moe": {"num_experts": 1, "top_k": 1, "moe_intermediate_size": 18944},
                    "vocab_size": 152064,
                    "num_layers": 28,
                }
            }

    monkeypatch.setattr(core, "_load_hf_to_config_module", lambda: FakeConverter)

    created = core.create_model_config_from_huggingface("https://huggingface.co/Qwen/Qwen2.5-7B", "qwen imported")
    saved = core._yaml_load(workspace / "configs" / "models" / created["id"])

    assert created["id"] == "qwen_imported.yaml"
    assert created["model_type"] == "llama"
    assert created["alias"] == "qwen2"
    assert saved["model_param"]["model_type"] == "llama"
    assert saved["model_param"]["run_type"] == "training"
    assert saved["metadata"]["huggingface_source"]["model_id"] == "Qwen/Qwen2.5-7B"
    assert any(item["id"] == "qwen_imported.yaml" for item in core.list_presets("models"))


def test_use_astrasim_simple_option_forces_hierarchical_backend(monkeypatch, tmp_path):
    workspace = _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["simple"]["use_astrasim"] = True
    payload["advanced"]["execution_backend"] = "analytical"
    payload["advanced"]["execution_mode"] = "full_astrasim_flattened"

    _, hardware, errors = core.save_config_edits_from_payload(payload)

    assert errors == []
    assert hardware is not None
    hardware_yaml = core._yaml_load(workspace / "configs" / "hardware" / "H100_SXM5_80GB.yaml")
    assert hardware_yaml["execution_backend"]["model"] == "astra"
    assert hardware_yaml["execution_backend"]["astra"]["mode"] == "full_astrasim_hierarchical"

    payload["simple"]["use_astrasim"] = False
    _, hardware, errors = core.save_config_edits_from_payload(payload)

    assert errors == []
    assert hardware is not None
    hardware_yaml = core._yaml_load(workspace / "configs" / "hardware" / "H100_SXM5_80GB.yaml")
    assert hardware_yaml["execution_backend"]["model"] == "analytical"
    assert hardware_yaml["execution_backend"]["astra"]["mode"] == "full_astrasim_hierarchical"


def test_zero_stage_dropdown_string_saves_as_integer(monkeypatch, tmp_path):
    workspace = _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["advanced"]["dp_zero_stage"] = "3"

    _, hardware, errors = core.save_config_edits_from_payload(payload)

    assert errors == []
    assert hardware is not None
    hardware_yaml = core._yaml_load(workspace / "configs" / "hardware" / "H100_SXM5_80GB.yaml")
    assert hardware_yaml["sw_param"]["dp_zero_stage"] == 3


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
        hardware_yaml = core._yaml_load(workspace / "configs" / "hardware" / "H100_SXM5_80GB.yaml")
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
    precision = core._yaml_load(workspace / "configs" / "hardware" / "H100_SXM5_80GB.yaml")["sw_param"]["precision"]
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


def test_training_form_defaults_show_replica_count_one(monkeypatch, tmp_path):
    workspace = _isolate_workspace(monkeypatch, tmp_path)
    hardware_path = workspace / "configs" / "hardware" / "H100_SXM5_80GB.yaml"
    hardware_yaml = core._yaml_load(hardware_path)
    hardware_yaml.setdefault("parallelism", {}).setdefault("inference", {})["replica_count"] = 8
    hardware_path.write_text(core._yaml_dump(hardware_yaml), encoding="utf-8")

    defaults = core.build_form_defaults("Llama2-7B.yaml", "H100_SXM5_80GB.yaml")

    assert defaults["run_type"] == "training"
    assert defaults["simple"]["replica_count"] == 1


def test_training_forces_replica_count_to_one(monkeypatch, tmp_path):
    workspace = _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["simple"]["run_type"] = "training"
    payload["simple"]["replica_count"] = 8

    _, hardware, errors = core.save_config_edits_from_payload(payload)
    preview = core.build_launch_preview(payload)

    assert errors == []
    assert hardware is not None
    hardware_yaml = core._yaml_load(workspace / "configs" / "hardware" / "H100_SXM5_80GB.yaml")
    assert hardware_yaml["parallelism"]["inference"]["replica_count"] == 1
    assert preview["ok"] is True
    assert preview["top_level_cases"][0]["hardware"]["parallelism"]["inference"]["replica_count"] == 1


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
    active_hardware = core._yaml_load(workspace / "configs" / "hardware" / "H100_SXM5_80GB.yaml")
    inactive_hardware = core._yaml_load(workspace / "configs" / "hardware" / "A100_SXM4_80GB.yaml")
    assert active_model["model_param"]["seq_len"] == 2048
    assert inactive_model["model_param"]["seq_len"] == 32768
    assert active_hardware["tech_param"]["DRAM"]["size"] == "72 GB"
    assert active_hardware["tech_param"]["core"]["util"] == 0.61
    assert inactive_hardware["tech_param"]["DRAM"]["size"] == "80 GB"
    assert inactive_hardware["tech_param"]["core"]["util"] == 0.9


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
            "config_values": ["H100_SXM5_80GB.yaml", "A100_SXM4_80GB.yaml"],
        },
    ]

    preview = core.build_launch_preview(payload)

    assert preview["ok"] is True
    assert preview["top_level_case_count"] == 4
    labels = [case["label"] for case in preview["top_level_cases"]]
    assert "Llama2-7B on H100 SXM5 80GB" in labels
    assert "Llama3.1-70B 2d train on A100 SXM4 80GB" in labels
    assert all("Model Config=" not in label and "Hardware Config=" not in label for label in labels)

    assert core.build_job_title(payload, preview) == "2 models x 2 hardware targets Training Sweep"


def test_config_comparison_job_titles_summarize_selected_side(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["dimensions"] = [
        {
            "field_key": "hardware_config",
            "mode": "values",
            "config_values": ["H100_SXM5_80GB.yaml", "A100_SXM4_80GB.yaml"],
        }
    ]

    preview = core.build_launch_preview(payload)

    assert preview["ok"] is True
    assert core.build_job_title(payload, preview) == "Llama2-7B on 2 hardware targets Training Sweep"
    assert [case["label"] for case in preview["top_level_cases"]] == [
        "Llama2-7B on H100 SXM5 80GB",
        "Llama2-7B on A100 SXM4 80GB",
    ]


def test_repeated_history_titles_get_stable_duplicate_indices(monkeypatch, tmp_path):
    workspace = _isolate_workspace(monkeypatch, tmp_path)
    for index, job_id in enumerate(["run-old", "run-new", "run-other"], start=1):
        job_root = workspace / "runs" / job_id
        job_root.mkdir(parents=True)
        title = "Llama2-7B on H100" if job_id != "run-other" else "Different title"
        core._json_dump(job_root / "request.json", {"title": title, "created_at": f"2026-04-26T00:0{index}:00+00:00"})
        core._json_dump(job_root / "status.json", {"status": "completed", "created_at": f"2026-04-26T00:0{index}:00+00:00", "updated_at": f"2026-04-26T00:0{index}:30+00:00"})

    history = core.list_history()
    indexed = {item["id"]: item for item in history}

    assert indexed["run-old"]["title_index"] == 1
    assert indexed["run-new"]["title_index"] == 2
    assert indexed["run-old"]["title_duplicate_count"] == 2
    assert indexed["run-new"]["title_duplicate_count"] == 2
    assert indexed["run-other"]["title_duplicate_count"] == 1


def test_inference_sweep_titles_include_run_type(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["simple"]["run_type"] = "inference"
    payload["dimensions"] = [
        {
            "field_key": "hardware.hbm_gb",
            "mode": "list",
            "list_text": "80, 320",
        }
    ]

    preview = core.build_launch_preview(payload)

    assert preview["ok"] is True
    assert core.build_job_title(payload, preview) == "Llama2-7B on H100 SXM5 80GB Inference Sweep"


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


def test_optimized_single_value_total_gpu_sweep_has_nonzero_invocations(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["optimize_parallelism"] = True
    payload["dimensions"] = [
        {
            "field_key": "hardware.total_gpus",
            "mode": "list",
            "list_text": "1",
        }
    ]

    preview = core.build_launch_preview(payload)

    assert preview["ok"] is True
    assert preview["top_level_case_count"] == 1
    assert preview["top_level_cases"][0]["target_total_gpus"] == 1
    assert preview["candidate_breakdown"] == [{"case_id": "case-0001", "count": 1}]
    assert preview["total_invocations"] == 1
    case = preview["top_level_cases"][0]
    assert core.generate_parallelism_candidates(case["hardware"], case["run_type"], 1, "Fast") == [
        {"tp": 1, "cp": 1, "pp": 1, "dp": 1, "ep": 1, "replica_count": 1}
    ]


def test_optimized_total_gpu_sweep_invocations_scale_per_top_level_case(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["optimize_parallelism"] = True
    payload["dimensions"] = [
        {
            "field_key": "hardware.total_gpus",
            "mode": "list",
            "list_text": "1, 8",
        },
        {
            "field_key": "model.global_batch_size",
            "mode": "list",
            "list_text": "32, 64",
        },
    ]

    preview = core.build_launch_preview(payload)

    assert preview["ok"] is True
    assert preview["top_level_case_count"] == 4
    counts_by_target = {}
    for case, breakdown in zip(preview["top_level_cases"], preview["candidate_breakdown"]):
        assert breakdown["count"] > 0
        counts_by_target.setdefault(case["target_total_gpus"], breakdown["count"])
        assert counts_by_target[case["target_total_gpus"]] == breakdown["count"]
    assert preview["total_invocations"] == sum(item["count"] for item in preview["candidate_breakdown"])
    assert counts_by_target[8] > counts_by_target[1]


def test_optimized_hbm_capacity_sweep_searches_each_case_hardware(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["optimize_parallelism"] = True
    payload["dimensions"] = [
        {
            "field_key": "hardware.hbm_gb",
            "mode": "list",
            "list_text": "40, 80",
        }
    ]
    seen_hbm_sizes = []

    def fake_generate_parallelism_candidates(hardware, run_type, target_total_gpus, preset_name):
        seen_hbm_sizes.append(hardware["tech_param"]["DRAM"]["size"])
        return [{"tp": 1, "cp": 1, "pp": 1, "dp": target_total_gpus, "ep": 1, "replica_count": 1}]

    monkeypatch.setattr(core, "generate_parallelism_candidates", fake_generate_parallelism_candidates)

    preview = core.build_launch_preview(payload)

    assert preview["ok"] is True
    assert [case["dimension_values"]["hardware.hbm_gb"] for case in preview["top_level_cases"]] == [40, 80]
    assert seen_hbm_sizes == ["40 GB", "80 GB"]
    assert preview["candidate_breakdown"] == [
        {"case_id": "case-0001", "count": 1},
        {"case_id": "case-0002", "count": 1},
    ]
    assert preview["total_invocations"] == 2


def test_all_invalid_total_gpu_sweep_is_error_not_silent_zero(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["optimize_parallelism"] = False
    payload["dimensions"] = [
        {
            "field_key": "hardware.total_gpus",
            "mode": "list",
            "list_text": "1",
        }
    ]

    preview = core.build_launch_preview(payload)

    assert preview["ok"] is False
    assert preview["total_invocations"] == 0
    assert any("All sweep cases were invalid" in error for error in preview["errors"])
    assert preview["invalid_cases"]


def test_inference_parallelism_optimizer_preserves_replica_count(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["simple"]["run_type"] = "inference"
    payload["simple"]["total_gpus"] = 16
    payload["simple"]["replica_count"] = 2
    payload["optimize_parallelism"] = True

    preview = core.build_launch_preview(payload)

    assert preview["ok"] is True
    assert preview["optimizer_enabled"] is True
    case = preview["top_level_cases"][0]
    candidates = core.generate_parallelism_candidates(case["hardware"], "inference", case["target_total_gpus"], "Fast")
    assert candidates
    assert {candidate["replica_count"] for candidate in candidates} == {2}
    assert all(candidate["tp"] * candidate["cp"] * candidate["pp"] * candidate["ep"] * candidate["replica_count"] == 16 for candidate in candidates)
    optimized = core.apply_parallelism_candidate(case["hardware"], candidates[0])
    assert optimized["parallelism"]["inference"]["replica_count"] == 2


def test_raw_parallelism_axis_sweeps_are_rejected(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["dimensions"] = [{"field_key": "hardware.parallelism.tp", "mode": "list", "list_text": "1, 2"}]

    preview = core.build_launch_preview(payload)

    assert preview["ok"] is False
    assert "Unsupported sweep field: hardware.parallelism.tp" in preview["errors"]


def test_network_bandwidth_and_latency_sweeps_apply_per_dimension(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["dimensions"] = [
        {"field_key": "hardware.network.dim0.bandwidth_gbs", "mode": "list", "list_text": "100, 200"},
        {"field_key": "hardware.network.dim1.latency_s", "mode": "list", "list_text": "0.000001, 0.000002"},
    ]

    preview = core.build_launch_preview(payload)

    assert preview["ok"] is True
    assert preview["top_level_case_count"] == 4
    first_dimensions = preview["top_level_cases"][0]["hardware"]["network"]["dimensions"]
    last_dimensions = preview["top_level_cases"][-1]["hardware"]["network"]["dimensions"]
    assert first_dimensions[0]["topology"]["bandwidth"] == "100 GB"
    assert first_dimensions[1]["topology"]["latency"] == 0.000001
    assert last_dimensions[0]["topology"]["bandwidth"] == "200 GB"
    assert last_dimensions[1]["topology"]["latency"] == 0.000002
    assert preview["top_level_cases"][0]["dimension_values"]["hardware.network.dim0.bandwidth_gbs"] == 100.0
    assert preview["top_level_cases"][0]["dimension_values"]["hardware.network.dim1.latency_s"] == 0.000001


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


def test_zero_timeout_disables_worst_case_wall_clock(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    payload = _payload()
    payload["worker_count"] = 2
    payload["timeout_seconds"] = 0
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
    assert preview["timeout_seconds"] == 0
    assert preview["total_invocations"] == 3
    assert preview["worst_case_wall_clock_s"] is None


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
    png_bytes = b"\x89PNG\r\n\x1a\nfake-png-data"

    first = Path(core.save_plot_png("sweep", "sweep-test", "Llama2-7B on H100 Sweep", png_bytes))
    second = Path(core.save_plot_png("sweep", "sweep-test", "Llama2-7B on H100 Sweep", png_bytes))

    assert first == workspace / "sweeps" / "sweep-test" / "plots" / "llama2-7b-on-h100-sweep-plot-001.png"
    assert second == workspace / "sweeps" / "sweep-test" / "plots" / "llama2-7b-on-h100-sweep-plot-002.png"
    assert first.read_bytes() == png_bytes
    assert second.read_bytes() == png_bytes


def test_saved_table_exports_are_standardized_and_incremented(monkeypatch, tmp_path):
    workspace = _isolate_workspace(monkeypatch, tmp_path)
    job_root = core.SWEEPS_ROOT / "sweep-test"
    job_root.mkdir(parents=True)

    first = Path(core.save_table_export("sweep", "sweep-test", "Llama2-7B on H100 Sweep", "csv", "case,value\ncase-1,1\n"))
    second = Path(core.save_table_export("sweep", "sweep-test", "Llama2-7B on H100 Sweep", "json", "[{\"case\":\"case-1\"}]\n"))

    assert first == workspace / "sweeps" / "sweep-test" / "exports" / "llama2-7b-on-h100-sweep-table-001.csv"
    assert second == workspace / "sweeps" / "sweep-test" / "exports" / "llama2-7b-on-h100-sweep-table-001.json"
    assert first.read_text().startswith("case,value")
    assert second.read_text().startswith("[")


def test_load_job_detail_reads_ranked_jsonl_top_results_and_full_load(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    job_root = core.SWEEPS_ROOT / "large-sweep"
    job_root.mkdir(parents=True)
    (job_root / "request.json").write_text(json.dumps({"title": "Large Sweep", "payload": {"metric": "training_time_s"}}))
    records = [
        {"case_id": f"case-{index:05d}", "status": "completed", "dimension_values": {}, "metrics": {"training_time_s": 50 - index}}
        for index in range(50)
    ]
    core._write_sweep_cases_jsonl(job_root, records)

    top_detail = core.load_job_detail("sweep", "large-sweep", case_limit=12)
    full_detail = core.load_job_detail("sweep", "large-sweep", display_mode="full")

    assert top_detail["_case_count_total"] == 50
    assert top_detail["_case_count_loaded"] == 12
    assert top_detail["_case_limit"] == 12
    assert top_detail["_case_display_mode"] == "top"
    assert top_detail["_case_source"] == "jsonl"
    assert len(top_detail["cases"]) == 12
    assert top_detail["cases"][0]["case_id"] == "case-00049"
    assert top_detail["cases"][-1]["case_id"] == "case-00038"
    assert full_detail["_case_count_loaded"] == 50
    assert len(full_detail["cases"]) == 50
    assert full_detail["cases"][0]["case_id"] == "case-00000"


def test_load_job_detail_can_limit_legacy_sweep_case_reads(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    job_root = core.SWEEPS_ROOT / "large-sweep"
    cases_root = job_root / "cases"
    cases_root.mkdir(parents=True)
    (job_root / "request.json").write_text(json.dumps({"title": "Large Sweep", "payload": {"metric": "training_time_s"}}))
    for index in range(50):
        (cases_root / f"case-{index:05d}.json").write_text(
            json.dumps({"case_id": f"case-{index:05d}", "status": "completed", "dimension_values": {}, "metrics": {"training_time_s": index}})
        )

    detail = core.load_job_detail("sweep", "large-sweep", case_limit=12)

    assert detail["_case_count_total"] == 50
    assert detail["_case_count_loaded"] == 12
    assert detail["_case_limit"] == 12
    assert detail["_case_source"] == "legacy-json"
    assert len(detail["cases"]) == 12
    assert detail["cases"][-1]["case_id"] == "case-00011"


def test_sweep_execution_writes_one_cases_jsonl_file(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    job_root = core.SWEEPS_ROOT / "jsonl-sweep"
    job_root.mkdir(parents=True)
    (job_root / "status.json").write_text(json.dumps({"status": "running"}))
    manager = core.RunManager()
    case_plans = []
    for index in range(3):
        case_plans.append(
            {
                "top_case": {
                    "case_id": f"case-{index + 1:04d}",
                    "label": f"Case {index + 1}",
                    "model": {},
                    "hardware": {},
                    "dimension_values": {"model.global_batch_size": index + 1},
                },
                "candidate": None,
            }
        )

    def fake_execute(job_root, case_id, model_dict, hardware_dict, timeout_seconds, **kwargs):
        del job_root, model_dict, hardware_dict, timeout_seconds
        return {
            "case_id": case_id,
            "top_case_id": kwargs.get("top_case_id") or case_id,
            "label": kwargs.get("case_label") or case_id,
            "status": "completed",
            "metrics": {"training_time_s": float(case_id.rsplit("-", 1)[-1])},
            "warnings": [],
            "error": None,
            "dimension_values": kwargs.get("dimension_values") or {},
        }

    monkeypatch.setattr(manager, "_execute_worker_case", fake_execute)
    summary = manager._execute_case_set(job_root, case_plans, "training_time_s", False, 0, 1)

    lines = (job_root / "cases.jsonl").read_text().splitlines()
    assert summary["case_count"] == 3
    assert len(lines) == 3
    assert not (job_root / "cases").exists()
    assert json.loads(lines[0])["case_index"] == 1
    assert json.loads(lines[0])["case_id"] == "case-0001"


def test_worker_cases_run_from_case_directory_with_repo_pythonpath(monkeypatch, tmp_path):
    _isolate_workspace(monkeypatch, tmp_path)
    manager = core.RunManager()
    job_root = tmp_path / "job"
    captured = {}

    class FakeProcess:
        def wait(self, timeout=None):
            del timeout
            result_arg = captured["cmd"][captured["cmd"].index("--result-json") + 1]
            Path(result_arg).write_text(json.dumps({"success": True, "metrics": {"training_time_s": 1.0}, "warnings": []}))

        def poll(self):
            return 0

    def fake_popen(cmd, cwd, stdout, stderr, env):
        del stdout, stderr
        captured.update({"cmd": cmd, "cwd": cwd, "env": env})
        return FakeProcess()

    monkeypatch.setattr(core.subprocess, "Popen", fake_popen)

    result = manager._execute_worker_case(
        job_root,
        "case-0001",
        {"model_param": {"mode": "LLM", "run_type": "training"}},
        {"parallelism": {"tp": 1, "cp": 1, "pp": 1, "ep": 1, "dp": 1}},
        0,
    )

    expected_case_root = (job_root / "artifacts" / "case-0001").resolve()
    assert result["status"] == "completed"
    assert Path(captured["cwd"]) == expected_case_root
    assert (expected_case_root / "astra_cache").is_dir()
    assert Path(captured["cmd"][captured["cmd"].index("--model-config") + 1]).is_absolute()
    assert Path(captured["cmd"][captured["cmd"].index("--hardware-config") + 1]).is_absolute()
    assert Path(captured["cmd"][captured["cmd"].index("--result-json") + 1]).is_absolute()
    assert str(core.ROOT) in captured["env"]["PYTHONPATH"].split(core.os.pathsep)


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
