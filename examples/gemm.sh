#!/bin/bash

# Runs performance model in GEMM mode
# Setup model parameter in configs/model-config/GEMM.yaml
# Setup hardware parameter in configs/hardware-config/a100_80GB.yaml

HW_CONFIG="configs/hardware-config/a100_80GB.yaml"
MODEL_CONFIG="configs/model-config/GEMM.yaml"

if [[ -n "${RAPID_UV_RUN:-}" ]]; then
    run_cmd=(uv run python run_perf.py)
else
    run_cmd=(python run_perf.py)
fi

"${run_cmd[@]}" --hardware_config "${HW_CONFIG}" --model_config "${MODEL_CONFIG}"
