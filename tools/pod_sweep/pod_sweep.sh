#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -x "$REPO_ROOT/venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/venv/bin/python"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
  echo "Could not find repo-local Python at venv/bin/python or .venv/bin/python" >&2
  exit 1
fi

DEFAULT_ERROR_LOG_DIR="$REPO_ROOT/tools/pod_sweep/error_logs"
HAS_ERROR_LOG_DIR=0
for arg in "$@"; do
  if [[ "$arg" == "--error-log-dir" || "$arg" == --error-log-dir=* ]]; then
    HAS_ERROR_LOG_DIR=1
    break
  fi
done

if [[ "$HAS_ERROR_LOG_DIR" -eq 1 ]]; then
  exec "$PYTHON_BIN" "$REPO_ROOT/tools/pod_sweep/superpod_sweep.py" "$@"
fi

exec "$PYTHON_BIN" \
  "$REPO_ROOT/tools/pod_sweep/superpod_sweep.py" \
  --error-log-dir "$DEFAULT_ERROR_LOG_DIR" \
  "$@"
