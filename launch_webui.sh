#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/rapid-llm-matplotlib}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/rapid-llm-uv-cache}"
export RAPID_WEBUI_HOST="${RAPID_WEBUI_HOST:-127.0.0.1}"
export RAPID_WEBUI_PORT="${RAPID_WEBUI_PORT:-8050}"

mkdir -p "$MPLCONFIGDIR"

exec uv run --frozen --no-sync python -m webui.app.main
