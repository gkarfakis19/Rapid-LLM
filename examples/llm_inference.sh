#!/bin/bash

# Runs performance model in LLM mode
# Setup model parameters in configs/model-config/Llama2-7B_inf.yaml
# Setup hardware parameters (incl. parallelism/network) in configs/hardware-config/a100_80GB_no_parallelism.yaml

HW_CONFIG="configs/hardware-config/a100_80GB_no_parallelism.yaml"
MODEL_CONFIG="configs/model-config/Llama2-7B_inf.yaml"

if [[ -n "${RAPID_UV_RUN:-}" ]]; then
    run_cmd=(uv run python run_perf.py)
else
    run_cmd=(python run_perf.py)
fi

"${run_cmd[@]}" --hardware_config "${HW_CONFIG}" --model_config "${MODEL_CONFIG}"
