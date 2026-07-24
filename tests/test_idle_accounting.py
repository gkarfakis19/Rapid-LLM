# Copyright 2026 NanoCad lab, UCLA
# https://nanocad.ee.ucla.edu/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for kernel-level idle-time accounting (thermal_stco output contract).

Covers:
  (a) training run_perf run emits all idle lines and satisfies the thermal_stco
      consumer regexes,
  (b) inference run_perf run emits all idle lines (incl. the fallback pair),
  (c) zero behavior change: Total Time / Inference Time for batch are
      bit-identical to the pre-instrumentation baseline,
  (d) record_idle_from_gemm edge cases (zero throughput, non-finite inputs,
      bucket routing, reset).
"""

import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- Consumer regexes -------------------------------------------------------
# Copied VERBATIM from thermal_stco's
# dedeepyo_integration/thermal_analysis_george.py::_parse_rapid_result_files
# (do NOT import thermal_stco here; the literals are the contract).
NUMBER = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
RE_THERMAL_IDLE = rf"GPU_time_frac_idle_thermal:\s*{NUMBER}"
RE_IDLE = rf"GPU_time_frac_idle:\s*{NUMBER}"
RE_TOTAL_TIME = rf"Total Time:\s*{NUMBER}"
RE_INFERENCE_TIME = rf"Inference Time for batch:\s*{NUMBER}s"
RE_TOTAL_INFERENCE_TIME = rf"Total Inference Time:\s*{NUMBER}s"
RE_PREFILL_TIME = rf"Prefill Time:\s*{NUMBER}s"
RE_DECODE_TIME = rf"Decode Time:\s*{NUMBER}s"
RE_PREFILL_IDLE = rf"Prefill Idle Time:\s*{NUMBER}s"
RE_DECODE_IDLE = rf"Decode Idle Time:\s*{NUMBER}s"

# --- Zero-behavior-change baselines -----------------------------------------
# Captured on branch `heterogeneous` (commit 82d0df7) BEFORE the idle
# instrumentation was added; the idle recording is pure observation, so these
# formatted values must stay bit-identical.
BASELINE_TRAIN_TOTAL_TIME_LINE = "Total Time: 0.21964824"
BASELINE_INFERENCE_TIME_LINE = "Inference Time for batch: 22.01s"

TRAIN_HW = REPO_ROOT / "configs" / "hardware-config" / "a100_80GB_legacy_thermal_port.yaml"
TRAIN_MODEL = REPO_ROOT / "configs" / "model-config" / "Llama2-7B_train_2048_thermal.yaml"
INF_HW = REPO_ROOT / "configs" / "hardware-config" / "a100_80GB_no_parallelism.yaml"
INF_MODEL = REPO_ROOT / "configs" / "model-config" / "Llama2-7B_inf.yaml"


def _run_perf(tmp_dir: Path, hw_config: Path, model_config: Path) -> Path:
    """Run run_perf.py as a subprocess from a scratch cwd; return output dir."""
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "run_perf.py"),
            "--hardware_config",
            str(hw_config),
            "--model_config",
            str(model_config),
        ],
        cwd=str(tmp_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=560,
    )
    assert result.returncode == 0, (
        f"run_perf failed (rc={result.returncode})\n"
        f"stdout tail:\n{result.stdout[-2000:]}\nstderr tail:\n{result.stderr[-2000:]}"
    )
    return tmp_dir / "output" / "LLM"


@pytest.fixture(scope="module")
def training_results_text(tmp_path_factory) -> str:
    tmp_dir = tmp_path_factory.mktemp("idle_train")
    out_dir = _run_perf(tmp_dir, TRAIN_HW, TRAIN_MODEL)
    return (out_dir / "LLM_training_results.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def inference_results_text(tmp_path_factory) -> str:
    tmp_dir = tmp_path_factory.mktemp("idle_inf")
    out_dir = _run_perf(tmp_dir, INF_HW, INF_MODEL)
    return (out_dir / "LLM_inference_results.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# (a) training output contract
# ---------------------------------------------------------------------------

def test_training_results_have_all_idle_lines(training_results_text):
    text = training_results_text
    assert re.findall(RE_THERMAL_IDLE, text), "GPU_time_frac_idle_thermal line missing"
    assert re.findall(RE_IDLE, text), "GPU_time_frac_idle line missing"
    assert re.findall(RE_TOTAL_TIME, text), "Total Time line missing"
    assert re.findall(rf"Idle Time Layer:\s*{NUMBER}s", text), "Idle Time Layer line missing"
    assert re.findall(rf"Idle Time Global:\s*{NUMBER}s", text), "Idle Time Global line missing"


def test_training_thermal_idle_fraction_in_range(training_results_text):
    text = training_results_text
    thermal_idle = float(re.findall(RE_THERMAL_IDLE, text)[-1])
    plain_idle = float(re.findall(RE_IDLE, text)[-1])
    total_time = float(re.findall(RE_TOTAL_TIME, text)[-1])
    layer_idle = float(re.findall(rf"Idle Time Layer:\s*{NUMBER}s", text)[-1])
    global_idle = float(re.findall(rf"Idle Time Global:\s*{NUMBER}s", text)[-1])
    assert 0.0 < thermal_idle < 1.0
    assert 0.0 <= plain_idle < 1.0
    assert total_time > 0.0
    assert layer_idle >= 0.0
    assert global_idle >= 0.0
    # Thermal fraction reconstruction: (layer*num_layers + global) / total.
    num_layers = 32  # Llama2-7B_train_2048_thermal.yaml
    reconstructed = (layer_idle * num_layers + global_idle) / total_time
    assert thermal_idle == pytest.approx(reconstructed, rel=1e-4)


def test_training_total_time_bit_identical_to_baseline(training_results_text):
    # (c) zero behavior change.
    assert BASELINE_TRAIN_TOTAL_TIME_LINE in training_results_text


# ---------------------------------------------------------------------------
# (b) inference output contract
# ---------------------------------------------------------------------------

def test_inference_results_have_all_idle_lines(inference_results_text):
    text = inference_results_text
    for pattern, label in [
        (RE_IDLE, "GPU_time_frac_idle"),
        (RE_THERMAL_IDLE, "GPU_time_frac_idle_thermal"),
        (RE_INFERENCE_TIME, "Inference Time for batch"),
        (RE_PREFILL_TIME, "Prefill Time"),
        (RE_DECODE_TIME, "Decode Time"),
        (RE_PREFILL_IDLE, "Prefill Idle Time"),
        (RE_DECODE_IDLE, "Decode Idle Time"),
        (rf"Prefill Idle Layer Time:\s*{NUMBER}s", "Prefill Idle Layer Time"),
        (rf"Prefill Idle Global Time:\s*{NUMBER}s", "Prefill Idle Global Time"),
        (rf"Decode Idle Layer Time:\s*{NUMBER}s", "Decode Idle Layer Time"),
        (rf"Decode Idle Global Time:\s*{NUMBER}s", "Decode Idle Global Time"),
    ]:
        assert re.findall(pattern, text), f"{label} line missing"


def test_inference_never_emits_total_inference_time_line(inference_results_text):
    # The consumer's runtime pick is last-match over an ordered regex list; a
    # "Total Inference Time:" line would silently win over "Inference Time for
    # batch" and change its runtime source.
    assert not re.findall(RE_TOTAL_INFERENCE_TIME, inference_results_text)


def test_inference_idle_values_and_fallback_reconstruction(inference_results_text):
    text = inference_results_text
    thermal_idle = float(re.findall(RE_THERMAL_IDLE, text)[-1])
    prefill_time = float(re.findall(RE_PREFILL_TIME, text)[-1])
    decode_time = float(re.findall(RE_DECODE_TIME, text)[-1])
    prefill_idle = float(re.findall(RE_PREFILL_IDLE, text)[-1])
    decode_idle = float(re.findall(RE_DECODE_IDLE, text)[-1])
    assert 0.0 < thermal_idle < 1.0
    assert prefill_idle >= 0.0
    assert decode_idle >= 0.0
    # Consumer fallback path (no thermal line): reconstructs
    # idle = ((prefill_idle * num_layers) + decode_idle) / (prefill + decode).
    num_layers = 32  # Llama2-7B_inf.yaml
    total_time = prefill_time + decode_time
    assert total_time > 0.0
    fallback_idle = ((prefill_idle * num_layers) + decode_idle) / total_time
    assert math.isfinite(fallback_idle)
    assert 0.0 <= fallback_idle <= 1.0


def test_inference_time_bit_identical_to_baseline(inference_results_text):
    # (c) zero behavior change.
    assert BASELINE_INFERENCE_TIME_LINE in inference_results_text


# ---------------------------------------------------------------------------
# (d) record_idle_from_gemm unit tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def time_calc(tmp_path):
    import config
    from train_timing import TimeCalculationLLM

    hw = config.parse_config(str(TRAIN_HW), "hardware")
    model = config.parse_config(str(TRAIN_MODEL), "LLM")
    config.validate_configs(hw, model)
    return TimeCalculationLLM(hw, model, "LLM", output_dir=str(tmp_path / "out"))


def test_record_idle_basic_and_bucket_routing(time_calc):
    tc = time_calc
    tc.reset_idle_accounting()
    tc.th = 1e12  # deterministic throughput for the test
    # observed 1.0 s, ideal 0.5 s -> idle 0.5 s into the layer bucket
    tc.record_idle_from_gemm(1.0, 0.5e12)
    breakdown = tc.get_idle_breakdown_seconds()
    assert breakdown["layer"] == pytest.approx(0.5)
    assert breakdown["global"] == 0.0
    assert breakdown["total"] == pytest.approx(0.5)
    # global bucket routing
    tc.record_idle_from_gemm(1.0, 0.25e12, bucket="global")
    breakdown = tc.get_idle_breakdown_seconds()
    assert breakdown["global"] == pytest.approx(0.75)
    assert breakdown["layer"] == pytest.approx(0.5)
    assert breakdown["total"] == pytest.approx(1.25)
    assert tc.get_idle_time_seconds() == pytest.approx(1.25)
    assert tc._idle_samples == 2


def test_record_idle_clamps_negative_to_zero(time_calc):
    tc = time_calc
    tc.reset_idle_accounting()
    tc.th = 1e12
    # observed < ideal -> idle clamps to 0 (still counted as a sample)
    tc.record_idle_from_gemm(0.1, 1e12)
    assert tc.get_idle_time_seconds() == 0.0
    assert tc._idle_samples == 1


def test_record_idle_zero_throughput(time_calc):
    tc = time_calc
    tc.reset_idle_accounting()
    tc.th = 0.0
    # th <= 0 -> ideal is 0 -> the full observed time counts as idle
    tc.record_idle_from_gemm(0.25, 1e15)
    assert tc.get_idle_time_seconds() == pytest.approx(0.25)


def test_record_idle_ignores_non_finite_and_bad_inputs(time_calc):
    tc = time_calc
    tc.reset_idle_accounting()
    tc.th = 1e12
    tc.record_idle_from_gemm(float("nan"), 1.0)
    tc.record_idle_from_gemm(float("inf"), 1.0)
    tc.record_idle_from_gemm(1.0, float("nan"))
    tc.record_idle_from_gemm(1.0, float("inf"))
    tc.record_idle_from_gemm("not-a-number", 1.0)
    tc.record_idle_from_gemm(1.0, object())
    assert tc.get_idle_time_seconds() == 0.0
    assert tc._idle_samples == 0


def test_record_idle_scale_kwarg(time_calc):
    tc = time_calc
    tc.reset_idle_accounting()
    tc.th = 1e12
    # scale multiplies observed and flops alike
    tc.record_idle_from_gemm(1.0, 0.5e12, scale=2.0)
    assert tc.get_idle_time_seconds() == pytest.approx(1.0)


def test_reset_idle_accounting(time_calc):
    tc = time_calc
    tc.th = 1e12
    tc.record_idle_from_gemm(1.0, 0.0)
    tc.record_idle_from_gemm(1.0, 0.0, bucket="global")
    assert tc.get_idle_time_seconds() > 0.0
    tc.reset_idle_accounting()
    breakdown = tc.get_idle_breakdown_seconds()
    assert breakdown == {"layer": 0.0, "global": 0.0, "total": 0.0}
    assert tc._idle_samples == 0


def test_get_idle_fraction_edge_cases(time_calc):
    tc = time_calc
    tc.reset_idle_accounting()
    tc.th = 1e12
    assert tc.get_idle_fraction(0.0) == 0.0
    assert tc.get_idle_fraction(-1.0) == 0.0
    tc.record_idle_from_gemm(1.0, 0.0)  # 1 s idle
    assert tc.get_idle_fraction(2.0) == pytest.approx(0.5)
