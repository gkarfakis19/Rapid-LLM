# RAPID-LLM Web UI

The Web UI is a local Dash workbench for building RAPID-LLM runs, editing supported YAML fields, launching sweeps, and inspecting saved results.

## Layout

- `webui/app/`: Dash frontend, styling, and callbacks.
- `webui/service/`: frontend-neutral launch, config, history, and worker orchestration.
- `webui/workspace/configs/`: editable model and hardware YAML files seeded from curated repo configs.
- `webui/workspace/runs/` and `webui/workspace/sweeps/`: local history, resolved configs, metrics, logs, and artifacts.

## Launch

```bash
./.venv/bin/python -m webui.app.main
```

Optional port override:

```bash
RAPID_WEBUI_PORT=8051 ./.venv/bin/python -m webui.app.main
```

## Tests

Service, formatting, and worker checks:

```bash
./.venv/bin/python -m pytest tests/test_webui_formatting.py tests/test_webui_service.py tests/test_webui_worker_runner.py -q
```

Browser and aesthetic check:

```bash
RAPID_WEBUI_SCREENSHOT_DIR=webui/workspace/ui_checks ./.venv/bin/python -m pytest tests/test_webui_browser.py -q -s
```

The browser check launches the Dash app, captures desktop and mobile screenshots, verifies the main workflow panels, checks dropdown contrast, checks hover text, and fails on horizontal overflow.

## Current Behavior

- Run Setup selects one or more model YAML files and one or more hardware YAML files.
- Selecting multiple models or hardware targets creates comparison cases automatically.
- The first selected model and hardware load the editor defaults.
- Supported Basic Options and Advanced Options are written back to the editable YAML files under `webui/workspace/configs/`.
- Advanced Options are split into Model and Hardware sections. Model type writes `model_param.mode` and includes hover explanations for LLM, ViT, and GEMM.
- YAML mirrors are display-only; unsupported fields are edited directly in the workspace YAML files.
- Sweep Dimensions vary numeric fields with either comma-separated values or numeric ranges, and are hidden for Single Run mode.
- Use AstraSim is a Basic Options toggle. Enabled runs force AstraSim hierarchical mode; disabled runs use the analytical backend.
- Parallelism search evaluates generated TP/CP/PP/DP/EP candidates against the requested GPU count.
- One top-level job runs at a time; sweeps can use multiple local workers internally.
- The details view reports formatted quantities such as GFLOPS, TFLOPS, PFLOPS, GB, TB, and human-readable durations.

## Completeness Snapshot

Implemented:
- local launch and browser-tested layout
- editable workspace config seeding from validation and hardware config folders
- model and hardware comparison runs from Run Setup
- supported YAML edits persisted through Basic and Advanced Options
- launch previews with invocation count and worst-case wall-clock estimate
- single run, sweep run, cancellation, history, load, details, and artifact manifests
- worker-level metrics for achieved system FLOPS, achieved FLOPS per GPU, peak FLOPS per GPU, peak system FLOPS, MFU, memory status, and inference timing
- hover explanations for core controls and Details columns

Missing or weak:
- no browser editor for unsupported YAML fields
- no named config versioning or undo stack after a UI edit writes YAML
- no queue for multiple top-level jobs
- no rich compare dashboard beyond the current plot and Details table
- no conditional per-model sweep values inside one launch
- FLOPS are model-derived estimates divided by modeled runtime, not hardware-counter measurements
