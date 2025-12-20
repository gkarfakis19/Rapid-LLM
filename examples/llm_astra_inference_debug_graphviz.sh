#!/bin/bash

# Runs performance model in LLM mode
# Setup model parameter in configs/model-config/Llama2-7B.yaml
# Setup hardware parameters (incl. parallelism/network) in configs/hardware-config/a100_80GB.yaml

HW_CONFIG="configs/hardware-config/a100_80GB.yaml"
MODEL_CONFIG="configs/model-config/LLM_inf.yaml"

if [[ -n "${RAPID_UV_RUN:-}" ]]; then
    run_cmd=(uv run python run_perf.py)
else
    run_cmd=(python run_perf.py)
fi

RAPID_PERSIST_ASTRASIM_ARTIFACTS=1 \
RAPID_VISUALIZE_GRAPHS=1 \
RAPID_PERSIST_ARTIFACT_VIZ=1 \
"${run_cmd[@]}" --hardware_config "${HW_CONFIG}" --model_config "${MODEL_CONFIG}"
