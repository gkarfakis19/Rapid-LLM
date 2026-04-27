from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict

import yaml

import config
import llm_util
from inference_timing import TimeCalculationLLMInference
from train_timing import TimeCalculationLLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one RAPID-LLM case and emit structured JSON.")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--hardware-config", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_mode(model_config_path: str) -> str:
    data = yaml.safe_load(Path(model_config_path).read_text()) or {}
    return str(data.get("model_param", {}).get("mode", "LLM")).upper()


def gpu_peak_flops(hw_config) -> float:
    core = hw_config.tech_config.core
    bundles = core.num_bundles or 1
    per_cycle_flops = float(core.nominal_flop_rate_per_mcu)
    mcu_per_bundle = float(core.num_mcu_per_bundle)
    frequency = float(core.operating_frequency or core.nominal_frequency or 0.0)
    util = float(getattr(core, "util", 1.0) or 1.0)
    peak = per_cycle_flops * mcu_per_bundle * float(bundles)
    if frequency > 0:
        peak *= frequency
    return peak * util if peak > 0 else 0.0


def compute_total_flops(calculator: TimeCalculationLLM) -> float:
    global_batch = getattr(calculator, "batch_size", None)
    if global_batch is None or global_batch <= 0:
        global_batch = calculator._effective_transformer_batch()
    if not global_batch or global_batch <= 0:
        return float("nan")

    gemm_shapes = llm_util.process_gemm_shapes(
        calculator,
        global_batch,
        calculator.seq_len,
        calculator.hidden_dim,
        calculator.num_heads,
        calculator.kv_heads,
        calculator.intermediate_size,
        calculator.vocab_size,
    )

    def gemm_flops(shape) -> float:
        if shape is None:
            return 0.0
        try:
            dims = list(shape)
        except TypeError:
            return 0.0
        if len(dims) == 4:
            b, m, k, n = dims
            return 2.0 * float(b) * float(m) * float(k) * float(n)
        if len(dims) == 3:
            m, k, n = dims
            return 2.0 * float(m) * float(k) * float(n)
        return 0.0

    def forward_backward(shape) -> float:
        forward = gemm_flops(shape)
        return 3.0 * forward if forward > 0.0 else 0.0

    per_layer_flops = 0.0
    per_layer_flops += forward_backward(gemm_shapes.get("qkv_proj"))
    per_layer_flops += forward_backward(gemm_shapes.get("attention_score"))
    per_layer_flops += forward_backward(gemm_shapes.get("attention_output"))
    per_layer_flops += forward_backward(gemm_shapes.get("output_proj"))
    per_layer_flops += forward_backward(gemm_shapes.get("ffn1"))
    per_layer_flops += forward_backward(gemm_shapes.get("ffn2"))

    total_flops = per_layer_flops * float(calculator.num_layers)
    total_flops += forward_backward(gemm_shapes.get("linear"))
    return float(total_flops)


def training_num_gpus(hw_cfg) -> int:
    p = hw_cfg.sch_config
    return int(p.tp) * int(p.cp) * int(p.pp) * int(p.train.dp) * int(p.train.ep)


def inference_num_gpus(hw_cfg) -> int:
    p = hw_cfg.sch_config
    return int(p.tp) * int(p.cp) * int(p.pp) * int(p.inference.replica_count) * int(p.train.ep)


def run_training(hw_cfg, model_cfg, mode: str, output_dir: str) -> Dict[str, Any]:
    calc = TimeCalculationLLM(hw_cfg, model_cfg, mode, output_dir=output_dir)
    runtime = float(calc.calc_time_llm())
    total_flops = compute_total_flops(calc)
    peak_flops = gpu_peak_flops(hw_cfg)
    num_gpus = training_num_gpus(hw_cfg)
    achieved_flops = (total_flops / runtime) if runtime > 0 else float("nan")
    achieved_flops_per_gpu = (achieved_flops / num_gpus) if num_gpus > 0 else float("nan")
    peak_system_flops = peak_flops * num_gpus if peak_flops and num_gpus else float("nan")
    denom = peak_system_flops if peak_system_flops and peak_system_flops > 0 else float("nan")
    mfu = (achieved_flops / denom) if denom and denom > 0 else float("nan")
    warning = calc.memory_capacity_warning()
    return {
        "success": True,
        "run_type": "training",
        "metrics": {
            "training_time_s": runtime,
            "approx_mfu": None if math.isnan(mfu) else float(mfu),
            "num_gpus": num_gpus,
            "total_flops": None if math.isnan(total_flops) else float(total_flops),
            "achieved_flops": None if math.isnan(achieved_flops) else float(achieved_flops),
            "achieved_flops_per_gpu": None if math.isnan(achieved_flops_per_gpu) else float(achieved_flops_per_gpu),
            "peak_flops_per_gpu": None if math.isnan(peak_flops) else float(peak_flops),
            "peak_system_flops": None if math.isnan(peak_system_flops) else float(peak_system_flops),
            "memory_exceeded": bool(getattr(calc, "memory_capacity_exceeded", False)),
            "memory_violation_gb": float(getattr(calc, "memory_capacity_violation_gb", 0.0) or 0.0),
        },
        "warnings": [warning] if warning else [],
        "primary_metric_label": "Time / Batch",
        "primary_metric_value": runtime,
    }


def run_inference(hw_cfg, model_cfg, mode: str, output_dir: str) -> Dict[str, Any]:
    calc = TimeCalculationLLMInference(hw_cfg, model_cfg, mode, output_dir=output_dir)
    timing = calc.calc_total_inference_time()
    decode_rates = timing.get("decode_tokens_per_s") or {}
    batch_size = getattr(calc, "batch_size", 1)
    replica_count = max(1, getattr(calc, "replica_count", 1))
    midpoint_rate = float(decode_rates.get("midpoint", 0.0) or 0.0)
    decode_throughput = midpoint_rate * batch_size * replica_count
    warning = calc.memory_capacity_warning()
    return {
        "success": True,
        "run_type": "inference",
        "metrics": {
            "prefill_time_s": float(timing["prefill_time"]),
            "decode_time_s": float(timing["decode_time"]),
            "total_inference_time_s": float(timing["total_inference_time"]),
            "ttft_s": float(timing["time_to_first_token"]),
            "decode_throughput_tok_s": float(decode_throughput),
            "num_gpus": inference_num_gpus(hw_cfg),
            "memory_exceeded": bool(getattr(calc, "memory_capacity_exceeded", False)),
            "memory_violation_gb": float(getattr(calc, "memory_capacity_violation_gb", 0.0) or 0.0),
        },
        "warnings": [warning] if warning else [],
        "primary_metric_label": "Prefill Time",
        "primary_metric_value": float(timing["prefill_time"]),
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        mode = read_mode(args.model_config)
        hw_cfg = config.parse_config(args.hardware_config, config_type="hardware")
        model_cfg = config.parse_config(args.model_config, config_type=mode)
        config.validate_configs(hw_cfg, model_cfg)
        run_type = str(model_cfg.model_config.run_type).lower()
        result = run_inference(hw_cfg, model_cfg, mode, str(output_dir)) if run_type == "inference" else run_training(hw_cfg, model_cfg, mode, str(output_dir))
    except Exception as exc:  # noqa: BLE001
        result = {"success": False, "error": str(exc), "metrics": {}, "warnings": []}
    Path(args.result_json).write_text(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
