# Validation Harness

This document describes how to run `validation_scripts/validation_harness.py`
and the `validation_scripts/validation.sh` wrapper.

## What It Runs

The harness can run these suites in one shared worker pool:

- `nvidia_inf` (A100/H100 inference validation)
- `nvidia_train` (A100 training validation)
- `uci_train` (A100 training validation)
- `mosaic_train` (H100 training validation)

`nvidia_train` is submitted first by design, and progress is shown per suite.

## Prerequisites

Run from repo root:

```bash
cd /app/nanocad/projects/personal/yaoe888/Rapid-LLM
```

Use the repo virtualenv Python (not system Python):

```bash
./.venv/bin/python --version
```

## Running `validation.sh`

Recommended way to run the wrapper:

```bash
source .venv/bin/activate
cd validation_scripts
bash validation.sh
```

The checked-in wrapper currently does not restrict suites. The commented line in
`validation.sh` is only a note, so the wrapper runs the harness defaults:

- `nvidia_inf`
- `nvidia_train`
- `uci_train`
- `mosaic_train`

## Derate Configuration (YAML)

Harness derates come from `validation_scripts/validation_configs/harness_derates.yaml`
Shared values are applied to all suites.

## Common Commands

Run all suites:

```bash
./.venv/bin/python validation_scripts/validation_harness.py
```

Run all suites with explicit workers:

```bash
./.venv/bin/python validation_scripts/validation_harness.py --workers 64
```

Run only selected suites:

```bash
./.venv/bin/python validation_scripts/validation_harness.py \
  --suites nvidia_train,uci_train,mosaic_train
```

Run inference suite only for H100:

```bash
./.venv/bin/python validation_scripts/validation_harness.py \
  --suites nvidia_inf \
  --inference-devices H100
```

Disable plots and progress:

```bash
./.venv/bin/python validation_scripts/validation_harness.py --no-plot --no-progress
```

## Output Layout

By default, outputs are written under:

`output/validation/harness/<run_id>/`

Each run directory contains:

- Per-suite subdirectories (`nvidia_inf/`, `nvidia_train/`, `uci_train/`, `mosaic_train/`)
- `summary.csv`
- `summary.json` (includes derate config path, shared derates, suite compute utils, and suite submission order)

## Per-Suite Artifacts

The harness writes the following files under
`output/validation/harness/<run_id>/<suite>/` when that suite is enabled.

### `nvidia_inf`

Always written:

- `imec_a100_validation.csv`
- `imec_h100_validation.csv`
- `nvidia_a100_validation.csv`
- `nvidia_h100_validation.csv`
- `imec_rows.csv`
- `nvidia_rows.csv`

Written when plots are enabled:

- `inf_ratio_grid_combined_a100.png`
- `inf_ratio_grid_combined_h100.png`

### `nvidia_train`

- `nvidia_train_validation_result.csv`
- `nvidia_train_validation_compare.png` when plots are enabled

### `uci_train`

- `uci_train_validation_result.csv`
- `uci_train_validation_compare.png` when plots are enabled

### `mosaic_train`

- `mosiacml_h100_bf16_all.csv`
- `mosiacml_h100_bf16_all.png` when plots are enabled
- `mosiacml_h100_bf16_all_parity.png` when plots are enabled
- `mosiacml_h100_bf16_all_parity_combined.png` when plots are enabled

## Suite Configs, Inputs, and Comparison Data

All suites share the derate YAML:

`validation_scripts/validation_configs/harness_derates.yaml`

### `nvidia_inf`

Harness defaults:

- Devices: `A100,H100`
- Hardware configs:
  - `validation_scripts/validation_configs/hardware-config/A100_SXM4_80GB_inf.yaml`
  - `validation_scripts/validation_configs/hardware-config/H100_SXM5_80GB.yaml`
- IMEC model configs:
  - `validation_scripts/validation_configs/model-config/Llama2-7B_inf.yaml`
  - `validation_scripts/validation_configs/model-config/Llama2-13B_inf.yaml`
  - `validation_scripts/validation_configs/model-config/Llama2-70B_inf.yaml`
- NVIDIA dataset model config:
  - `validation_scripts/validation_configs/model-config/Llama3.1-70B_inf.yaml`

Reference and comparison data:

- IMEC reference CSVs:
  - `validation_scripts/imec_data/A100_inf.csv`
  - `validation_scripts/imec_data/H100_inf.csv`
- NVIDIA reference CSVs:
  - `validation_scripts/nvidia_data/8xA100_bf16_Llama3_3-70B.csv`
  - `validation_scripts/nvidia_data/4xH100_fp16_Llama3_3-70B.csv`
- Optional A100 comparison CSVs:
  - `validation_scripts/imec_data/A100_inf_llmcompass.csv`
  - `validation_scripts/imec_data/A100_inf_vidur.csv`
  - `validation_scripts/imec_data/A100_inf_genz.csv`
  - `validation_scripts/nvidia_data/8xA100_bf16_Llama3_3-70B_llmcompass.csv`
  - `validation_scripts/nvidia_data/8xA100_bf16_Llama3_3-70B_vidur.csv`
  - `validation_scripts/nvidia_data/8xA100_bf16_Llama3_3-70B_genz.csv`

Per-case overrides applied by the suite:

- IMEC cases force `global_batch_size=1`, `seq_len=400`, `decode_len=200`
- NVIDIA cases derive:
  - `global_batch_size` from the dataset `concurrency`
  - `seq_len` from `input_tokens + output_tokens`
  - `decode_len` from `output_tokens`
- Both paths set `tp` from the dataset and keep `cp=1`, `pp=1`, `mb=1`

### `nvidia_train`

Harness defaults:

- Input cases CSV:
  - `validation_scripts/train_validation_data/nvidia_train_validation_cases.csv`
- Base hardware config:
  - `validation_scripts/validation_configs/hardware-config/A100_SXM4_80GB_train_validation.yaml`
- Alternate hardware config used for rows with `model=GPT 175B` or `dim1_topology=switch`:
  - `validation_scripts/validation_configs/hardware-config/A100_SXM4_80GB_train_validation_switch.yaml`
- Model config directory:
  - `validation_scripts/validation_configs/model-config/`
- Model mapping:
  - `Llama 2-7B -> Llama2-7B_inf.yaml`
  - `GPT 22B -> GPT_22_B.yaml`
  - `GPT 175B -> GPT_175_B.yaml`
  - `GPT 310B -> GPT_310_B.yaml`
  - `GPT 530B -> GPT_530_B.yaml`
  - `GPT 1T -> GPT_1T.yaml`

Reference and comparison data:

- Reference cases live in:
  - `validation_scripts/train_validation_data/nvidia_train_validation_cases.csv`
- STAGE comparison data for the plot lives in:
  - `validation_scripts/train_validation_data/STAGE_data.csv`

### `uci_train`

Harness defaults:

- Hardware config:
  - `validation_scripts/validation_configs/hardware-config/A100_PCIe_80GB_train_validation.yaml`
- Model config:
  - `validation_scripts/validation_configs/model-config/Llama2-7B.yaml`
- Input CSV:
  - `validation_scripts/train_validation_data/uci_train.csv`
- Include filter CSV:
  - `validation_scripts/train_validation_data/uci_train.csv`
- Exclude filter CSV:
  - none by default
- Variants:
  - `ddp`
- Astra modes:
  - `full_astrasim_hierarchical`

Reference and comparison data:

- Reference cases live in:
  - `validation_scripts/train_validation_data/uci_train.csv`
- STAGE and MLSynth comparison data for the plot lives in:
  - `validation_scripts/train_validation_data/uci_train_stg_mlsynth.csv`

### `mosaic_train`

Harness defaults:

- Input CSV:
  - `validation_scripts/mosiacml_data/h100_80gb_bf16.csv`
- Hardware config used by the harness:
  - `validation_scripts/validation_configs/hardware-config/H100_SXM5_80GB.yaml`
- Model config directory:
  - `validation_scripts/validation_configs/model-config/`
- Model sizes are resolved from the input CSV via:
  - `MPT-760m.yaml`
  - `MPT-1b.yaml`
  - `MPT-3b.yaml`
  - `MPT-7b.yaml`
  - `MPT-13b.yaml`
  - `MPT-30b.yaml`
  - `MPT-70b.yaml`

Reference and comparison data:

- Reference cases live in:
  - `validation_scripts/mosiacml_data/h100_80gb_bf16.csv`
- The suite compares predictions against the CSV column:
  - `inferred_total_latency_s`

The MosaicML suite applies additional in-code network overrides so the
effective topology matches the old Mosaic-specific hardware config:

- `network.dimensions[id=dim0]`: `FullyConnected`, `50 GB`, `parallelisms: [dp]`
- `network.dimensions[id=dim1]`: `parallelisms: []`
