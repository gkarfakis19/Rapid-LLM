#!/bin/bash

# Runs performance model in LLM mode
# Setup model parameter in configs/model-config/Llama2-7B.yaml
# Setup hardware parameters (incl. parallelism/network) in configs/hardware-config/a100_80GB.yaml

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

HW_CONFIG="${REPO_ROOT}/configs/hardware-config/H100_SXM5_80GB.yaml"
MODEL_CONFIG="${REPO_ROOT}/configs/model-config/Llama2-7B.yaml"

if [[ -n "${RAPID_UV_RUN:-}" ]]; then
    run_cmd=(uv run run_perf.py)
else
    run_cmd=(python3 run_perf.py)
fi

RAPID_PERSIST_ASTRASIM_ARTIFACTS=1 \
RAPID_VISUALIZE_GRAPHS=1 \
RAPID_PERSIST_ARTIFACT_VIZ=1 \
"${run_cmd[@]}" --hardware_config "${HW_CONFIG}" --model_config "${MODEL_CONFIG}"
