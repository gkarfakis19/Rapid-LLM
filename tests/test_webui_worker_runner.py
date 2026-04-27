from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from webui.service import core
from webui.service.worker_runner import gpu_peak_flops


def test_worker_runner_executes_tiny_training_case(tmp_path):
    model = core.load_preset("models", "Llama2-7B.yaml")
    hardware = core.load_preset("hardware", "H100_SXM5_80GB_base.yaml")

    model_param = model["model_param"]
    model_param.update(
        {
            "run_type": "training",
            "global_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "seq_len": 16,
            "hidden_dim": 128,
            "intermediate_size": 512,
            "vocab_size": 1024,
            "num_layers": 1,
        }
    )
    model_param["attention"].update(
        {
            "attention_type": "mha",
            "num_heads": 4,
            "kv_heads": None,
            "use_flashattention": False,
            "attention_tile_size": 16,
        }
    )
    model_param["moe"].update(
        {
            "num_experts": 1,
            "top_k": 1,
            "moe_intermediate_size": 512,
            "n_shared_experts": 0,
            "moe_layer_freq": 1,
            "first_k_dense_replace": 1,
        }
    )

    parallelism = hardware["parallelism"]
    parallelism.update({"tp": 1, "cp": 1, "pp": 1, "mb": 1})
    parallelism["train"].update({"dp": 1, "ep": 1, "tp_ep": False})
    hardware["execution_backend"]["model"] = "analytical"
    dimensions = hardware.get("network", {}).get("dimensions", []) or []
    if dimensions:
        dimensions[0]["size"] = 1
        dimensions[0]["parallelisms"] = ["tp", "cp", "ep", "pp", "dp"]
        hardware["network"]["dimensions"] = [dimensions[0]]

    model_path = tmp_path / "model.yaml"
    hardware_path = tmp_path / "hardware.yaml"
    result_path = tmp_path / "result.json"
    output_dir = tmp_path / "worker-output"
    model_path.write_text(core._yaml_dump(model))
    hardware_path.write_text(core._yaml_dump(hardware))

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "webui.service.worker_runner",
            "--model-config",
            str(model_path),
            "--hardware-config",
            str(hardware_path),
            "--result-json",
            str(result_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=core.ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(result_path.read_text())
    assert result["success"] is True
    assert result["run_type"] == "training"
    assert result["metrics"]["training_time_s"] > 0
    assert result["metrics"]["num_gpus"] == 1
    assert result["metrics"]["achieved_flops"] == pytest.approx(result["metrics"]["achieved_flops_per_gpu"])
    assert result["metrics"]["peak_system_flops"] == pytest.approx(result["metrics"]["peak_flops_per_gpu"])


def test_a100_peak_flops_is_per_gpu_tensor_peak_not_system_pflops():
    core.ensure_workspace()
    hardware = core.load_preset("hardware", "A100_SXM4_80GB_base.yaml")
    core_fields = hardware["tech_param"]["core"]
    hw_cfg = SimpleNamespace(
        tech_config=SimpleNamespace(
            core=SimpleNamespace(
                num_bundles=core_fields["num_bundles"],
                nominal_flop_rate_per_mcu=core_fields["nominal_flop_rate_per_mcu"],
                num_mcu_per_bundle=core_fields["num_mcu_per_bundle"],
                operating_frequency=core_fields["operating_frequency"],
                nominal_frequency=None,
                util=core_fields["util"],
            )
        )
    )

    peak = gpu_peak_flops(hw_cfg)

    assert 2.0e14 < peak < 4.0e14


def test_h100_peak_flops_and_system_peak_are_consistent_for_eight_gpus():
    core.ensure_workspace()
    hardware = core.load_preset("hardware", "H100_SXM5_80GB_base.yaml")
    core_fields = hardware["tech_param"]["core"]
    hw_cfg = SimpleNamespace(
        tech_config=SimpleNamespace(
            core=SimpleNamespace(
                num_bundles=core_fields["num_bundles"],
                nominal_flop_rate_per_mcu=core_fields["nominal_flop_rate_per_mcu"],
                num_mcu_per_bundle=core_fields["num_mcu_per_bundle"],
                operating_frequency=core_fields["operating_frequency"],
                nominal_frequency=None,
                util=core_fields["util"],
            )
        )
    )

    peak_per_gpu = gpu_peak_flops(hw_cfg)
    peak_system = peak_per_gpu * 8

    assert peak_per_gpu == pytest.approx(1.07053056e15)
    assert peak_system == pytest.approx(8.56424448e15)
