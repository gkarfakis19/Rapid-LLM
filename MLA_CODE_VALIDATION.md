# MLA Code Validation

## Scope

This is the lightweight external validation pass for RAPID-LLM's MLA implementation after the
PT#1 Megatron-alignment work.

The goal is deliberately narrow:

- read the reference Megatron/DeepSpeed code directly
- derive the expected MLA contract from that code
- compare RAPID-LLM's toy-case formulas against that contract
- run a small practical smoke matrix in analytical and hierarchical mode to ensure the modeled MLA
  paths execute cleanly

This validation does **not** add tracing hooks to RAPID-LLM and it does **not** claim cycle-level
parity with Megatron kernels. It is a code-contract and toy-runtime validation for the supported
subset: training, prefill, and default full-cache decode.

## Pinned references

- Megatron-LM: `10e7b74597b47a57449599bf2780bd9e82ef79ff`
- DeepSpeed: `37e232faf9c810e0f84b467b9c70fed82b792d0a`

Local checkouts used for this validation:

- `tmp/reference_repos/Megatron-LM`
- `tmp/reference_repos/DeepSpeed`

## Lightweight method

The validator intentionally avoids heavy instrumentation.

1. Assert a small set of MLA-defining facts from the pinned reference trees.
2. Derive expected local tensor-shape and cache formulas from those facts.
3. Compare RAPID-LLM against those expectations on a compact toy matrix:
   - training: `tp=1`, `tp=2`, `cp=2`, `tp=2 cp=2`
   - inference prefill: `tp=1`, `tp=2`, `cp=2`, `tp=2 cp=2`
   - inference decode: `tp=1`, `tp=2`, `cp=2`, `tp=2 cp=2`
4. Run a small practical smoke matrix through RAPID-LLM itself in:
   - analytical mode
   - full AstraSim hierarchical mode
   - dense and one MoE sanity case
   - default full-cache decode

Primary artifacts:

- validator: [validation_scripts/mla_code_validation.py](/app/nanocad/projects/ispass_deepflow/deepflow_astra_dev/Rapid-LLM/validation_scripts/mla_code_validation.py)
- validator output: [tmp/mla_code_validation/results.json](/app/nanocad/projects/ispass_deepflow/deepflow_astra_dev/Rapid-LLM/tmp/mla_code_validation/results.json)
- practical smoke output: [tmp/mla_pt1_smoke/results.json](/app/nanocad/projects/ispass_deepflow/deepflow_astra_dev/Rapid-LLM/tmp/mla_pt1_smoke/results.json)

Latest validator summary:

- validator counts come from the canonical machine-readable output in
  [tmp/mla_code_validation/results.json](/app/nanocad/projects/ispass_deepflow/deepflow_astra_dev/Rapid-LLM/tmp/mla_code_validation/results.json)

## Reference code facts used

### Megatron-LM

The validator asserts these pinned Megatron facts:

1. Default MLA GPT layer spec uses local `linear()` down projections for `q_down` and `kv_down`.
2. Default training and prefill use the unabsorbed MLA path:
   `q_down/kv_down -> q_up/kv_up -> full per-head attention -> row-parallel output`.

### DeepSpeed

DeepSpeed is used only as a TP partitioning cross-check. The pinned rules confirm:

1. low-rank down projections are skipped by AutoTP
2. latent up projections are column-sharded
3. output projection is row-sharded

This matches the Megatron-side contract that RAPID-LLM now models.

## What now matches

### 1. TP partitioning of MLA projections

RAPID-LLM now matches the Megatron/DeepSpeed partitioning contract:

- `q_down` and `kv_down` are not TP-sharded
- `q_up` and `kv_up` are TP column-parallel
- output projection is TP row-parallel

This is validated directly by local output-element checks on the toy matrix.

### 2. Training and prefill MLA runtime contract

RAPID-LLM now models Megatron default training/prefill MLA as:

1. low-rank down projections
2. up projection to full per-head `Q/K/V`
3. full attention score/output
4. row-parallel output projection

The old absorbed-latent training/prefill path is no longer the default model.

### 3. Inference decode mode split

RAPID-LLM follows Megatron's supported default decode contract:

- default decode uses full per-head KV cache

The default decode path is validated by checking the emitted runtime components:

- default decode exposes `U_proj_q` and `U_proj_kv`

### 4. KV-cache bytes

RAPID-LLM matches the Megatron default full-cache mode exactly:

- full-cache mode: per-head `K/V`, TP-sharded over heads

The toy validator checks per-rank cache bytes directly.

### 5. Decode CP local-shape behavior

The remaining PT#1 validator mismatches were resolved by aligning the validator with Megatron's
actual decode behavior under THD context parallelism:

- decode projection work is partitioned over local active tokens per CP rank
- after the CP gather into attention, local attention outputs are produced for the full decode batch

RAPID-LLM's current decode CP behavior is consistent with that contract.

## Practical smoke results

The practical smoke matrix completed successfully in both analytical and hierarchical mode.

Covered toy cases:

- training analytical dense: `tp=2 cp=2 flash=true`
- training analytical MoE: `tp=1 cp=1`
- training hierarchical dense: `tp=2 cp=2 flash=true`
- training hierarchical MoE: `tp=1 cp=1`
- inference analytical full-cache: `tp=2 cp=2`
- inference hierarchical full-cache: `tp=2 cp=2`

Key sanity outcomes from [tmp/mla_pt1_smoke/results.json](/app/nanocad/projects/ispass_deepflow/deepflow_astra_dev/Rapid-LLM/tmp/mla_pt1_smoke/results.json):

- all cases executed without MLA-specific runtime failures
- full-cache inference KV-cache sizing matches the expected Megatron-style per-head storage contract
- hierarchical runs complete for both dense and MoE sanity cases

The MoE practical cases still print the repo's existing MoE energy warning. That is unrelated to the
MLA contract work and remains a known simulator limitation.

## Bottom line

Within the intended scope of this lightweight validation, RAPID-LLM's current MLA model is aligned
with the pinned Megatron/DeepSpeed code contract for training, prefill, and default full-cache
decode.

More precisely, it is aligned with the pinned Megatron/DeepSpeed **code contract** for:

- parameter grouping
- TP partitioning
- training/prefill full-MLA execution
- default decode mode split
- KV-cache sizing
- CP-aware local shape behavior

What this validation still does **not** prove:

- exact kernel-level parity with Megatron execution traces
- exact peak activation-memory parity with a live Megatron run
- exact end-to-end latency parity with Megatron on real hardware

Those would require a heavier trace- or measurement-based validation pass.
