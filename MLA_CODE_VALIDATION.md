# MLA Code Validation

## Scope

This is the lightweight external validation pass for RAPID-LLM's current MLA implementation.

The goal is not to build a trace framework. The goal is to compare RAPID-LLM's modeled MLA contract
against the reference code directly, using:

- pinned Megatron-LM source as the primary semantic reference
- pinned DeepSpeed source as a secondary TP-partitioning reference
- a compact toy matrix inside RAPID-LLM to check whether the local formulas implied by the reference
  code match RAPID-LLM's current formulas

## Pinned references

- Megatron-LM: `10e7b74597b47a57449599bf2780bd9e82ef79ff`
- DeepSpeed: `37e232faf9c810e0f84b467b9c70fed82b792d0a`

Local checkouts used for this validation:

- `tmp/reference_repos/Megatron-LM`
- `tmp/reference_repos/DeepSpeed`

## Lightweight method

The validation intentionally avoids heavy instrumentation.

1. Read the reference code and assert the key MLA code facts that determine semantics.
2. Derive the expected local formulas from those code facts.
3. Compare those expected formulas against RAPID-LLM on a tiny toy matrix:
   - training: `tp=1`, `tp=2`, `cp=2`, `tp=2 cp=2`
   - inference decode: `tp=1`, `tp=2`, `cp=2`, `tp=2 cp=2`
4. Record passes and mismatches.

Latest run summary:

- `19` checks passed
- `34` checks mismatched

The validator script is:

- [validation_scripts/mla_code_validation.py](/app/nanocad/projects/ispass_deepflow/deepflow_astra_dev/Rapid-LLM/validation_scripts/mla_code_validation.py)

Its latest machine-readable output is:

- [results.json](/app/nanocad/projects/ispass_deepflow/deepflow_astra_dev/Rapid-LLM/tmp/mla_code_validation/results.json)

## Reference code facts used

### Megatron-LM

The validation script asserts these code facts from the pinned Megatron tree:

1. Default MLA GPT layer spec uses `backend.linear()` for `linear_q_down_proj` and
   `linear_kv_down_proj`, while the up projections are TP-parallel and the output projection is
   row-parallel.

2. `cache_mla_latents` defaults to `False`.

3. When latent caching is enabled, the dynamic inference cache stores
   `kv_lora_rank + qk_pos_emb_head_dim`.

4. Default training/prefill MLA uses the unabsorbed path:
   `linear_kv_up_proj(kv_compressed)` reconstructs full per-head `K` and `V` before core attention.

5. Absorption is decode-only and gated by `cache_mla_latents`.

### DeepSpeed

The pinned DeepSpeed tree does not expose a full native MLA implementation comparable to Megatron.
What it does provide is useful TP partitioning evidence for DeepSeek-V2 style MLA:

1. `q_a_proj` and `kv_a_proj_with_mqa` are `SKIP`, which means the low-rank down projections are not
   TP-sharded by the AutoTP rules.

2. `q_b_proj` and `kv_b_proj` are `COLUMN`, which matches TP-sharded latent up projections.

3. `o_proj` is `ROW`, which matches a TP row-parallel output projection.

So DeepSpeed is a secondary corroboration of partitioning, not a full semantic MLA reference here.

## What passed

### Parameter count algebra

RAPID-LLM's MLA parameter total matches the Megatron-equivalent algebraic total for the toy config.

This is the good part of the current implementation. The high-level parameter decomposition is
compatible with Megatron if we map:

- RAPID `qk_nope_head_dim` -> Megatron `qk_head_dim`
- RAPID `qk_rope_head_dim` -> Megatron `qk_pos_emb_head_dim`
- RAPID split `D_proj_kv + K_rope_proj` -> Megatron fused `kv_down_proj`
- RAPID split `u_proj_k_nope + u_proj_v` -> Megatron fused `kv_up_proj`

## What did not pass

### 1. RAPID TP-shards the low-rank down projections, but Megatron does not by default

This is the first major correctness problem.

Megatron default MLA keeps the low-rank down projections local and unsharded across TP:

- `q_down_proj`: full `q_lora_rank`
- `kv_down_proj`: full `kv_lora_rank + qk_pos_emb_head_dim`

RAPID currently routes `D_proj_q`, `D_proj_kv`, and `K_rope_proj` through the generic `GemmType.QKV`
TP-sharding rules, which split the output dimension across TP. That means RAPID reduces the local
low-rank activation sizes by `1 / tp`.

This disagrees with both:

- Megatron's default MLA module spec
- DeepSpeed's DeepSeek-V2 AutoTP rules

Implication:

- RAPID underestimates projection-stage compute and activation footprint when `tp > 1`.

Concrete toy example from the validator:

- `train_tp2`:
  - RAPID local `q_down` output elements: `128`
  - Megatron-expected local `q_down` output elements: `256`
- `infer_decode_tp2`:
  - RAPID local `q_down` output elements: `16`
  - Megatron-expected local `q_down` output elements: `32`

### 2. RAPID models absorbed latent attention for training and prefill, but Megatron default does not

This is the largest structural mismatch.

RAPID's current MLA runtime path for training and prefill uses:

- latent `attention_score_1`
- latent `attention_score_2`
- separate rope score term
- latent attention context
- absorbed output projection

Megatron default training/prefill does not do that.

Megatron default path is:

1. low-rank down projections
2. up projection back to full per-head `Q`, `K`, and `V`
3. core attention on full `Q/K/V`
4. row-parallel output projection

Megatron only enters an absorbed-style path in decode-only inference when `cache_mla_latents=True`.

Implication:

- RAPID's current MLA training model is not Megatron-style MLA.
- RAPID's current MLA prefill model is also not Megatron-style MLA.

### 3. RAPID KV-cache bytes only coincide with Megatron latent-cache mode at `tp=1`, and otherwise do not match either Megatron cache mode

RAPID currently uses latent-cache bytes unconditionally for MLA inference and also divides them by TP.

That means:

- at `tp=1`, RAPID coincides with Megatron latent-cache mode
- at `tp>1`, RAPID matches neither Megatron mode

Megatron has two distinct cache behaviors:

- Megatron default: full per-head `K/V` cache, TP-sharded by heads
- Megatron latent-cache mode: cache `kv_lora_rank + qk_pos_emb_head_dim`, but that latent cache is
  not TP-sharded in the same way because it comes from the unsharded down projection path

So RAPID's current MLA cache formula is only accidentally aligned in the single-rank case and is
otherwise caught in the middle:

- smaller than Megatron default full-cache mode
- smaller than Megatron latent-cache mode when `tp > 1`

Concrete toy example from the validator:

- `infer_prefill_tp2`:
  - RAPID per-rank cache bytes per token: `32`
  - Megatron default full-cache bytes per token: `384`
  - Megatron latent-cache bytes per token: `64`

Implication:

- RAPID inference memory is optimistic relative to Megatron's default MLA inference path.
- RAPID TP scaling of MLA cache memory is not aligned with the reference code.

### 4. RAPID decode semantics do not match Megatron's mode split

Megatron decode has a clear mode split:

- default decode uses regular cached `K/V`
- decode with `cache_mla_latents=True` uses the latent-cache absorption path
- MLA decode uses a dedicated FlashMLA kernel path

RAPID currently models absorbed latent decode semantics directly and does not represent this Megatron
mode split.

Implication:

- RAPID decode timing and cache assumptions are not directly comparable to Megatron default MLA.

## Practical conclusion

The current RAPID-LLM MLA implementation is not validated against Megatron in the strong sense.

What is validated:

- the parameter algebra is broadly compatible

What is not validated and is currently mismatched:

- TP sharding of low-rank down projections
- training and prefill compute contract
- inference KV-cache semantics
- decode mode split and FlashMLA behavior

DeepSpeed does not rescue this result. Its pinned code only corroborates the TP partitioning
assumption that low-rank down projections are not sharded, which reinforces the Megatron mismatch.

## Bottom line

This lightweight external validation says the current RAPID-LLM MLA model is still structurally
different from Megatron/DeepSpeed-style MLA.

The main issue is not a small formula bug. The main issue is that RAPID models an absorbed latent MLA
contract across training/prefill/decode, while Megatron's current code uses:

- unabsorbed MLA for training and prefill
- optional latent-cache absorption only for decode

That means any claim that the current RAPID MLA model is "Megatron-correct" would be inaccurate.
