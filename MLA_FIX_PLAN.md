# MLA Fix Plan

## Goal

Make RAPID-LLM's MLA model support the same practical feature envelope as the dense attention path for the currently supported backends:

- TP
- TP+SP
- CP
- TP+CP hybrid
- FlashAttention in training and inference prefill
- better MLA activation-memory and KV-cache sharding

The fix keeps the composite MLA runtime path introduced earlier, but removes the old single-rank limitation and aligns timing, communication, and memory with the simulator's existing parallelism rules.

## Canonical Modeling Assumptions

### Runtime path

MLA timing continues to use the absorbed latent path:

- `D_proj_q`
- `D_proj_kv`
- `U_proj_q_rope`
- `K_rope_proj`
- `attention_score_1`
- `attention_score_2`
- `attention_score_rope`
- `attention_ctx_latent`
- `O_proj_absorbed`

Parameter bookkeeping remains on the original trainable matrices. Runtime compute remains on the absorbed latent path.

### Distributed sharding contract

The implementation follows RAPID-LLM's existing parallel GEMM contract rather than introducing a separate MLA-only distribution model.

- TP and TP+SP reuse the simulator's tensor-parallel sharding rules.
- CP and TP+CP reuse the simulator's context-parallel sharding rules.
- MLA stage communication is attached at the same high-level boundaries as dense attention:
  - `qkv_proj` forward for CP gather/broadcast
  - `qkv_proj` backward for TP hidden-gradient reduction
  - `attention` backward for CP reduce-scatter
  - `output_proj` forward for TP output reduction
  - `output_proj` backward for CP all-gather

This is deliberate. It keeps MLA compatible with the existing graph builder and AstraSim wiring.

### FlashAttention

MLA FlashAttention is modeled as a memory-improved variant of the same latent attention path:

- forward FLOPs stay equal to the non-flash MLA attention FLOPs
- attention-score materialization is removed from activation memory
- backward includes selective recompute of the score-only latent path plus softmax when full recomputation is disabled
- decode still ignores FlashAttention, matching the dense path

## Concrete Changes

### 1. Remove MLA feature gates

Delete the config-time rejection of:

- `attention_type='mla'` with `tp > 1`
- `attention_type='mla'` with `cp > 1`
- `attention_type='mla'` with `use_flashattention=true`

### 2. Generalize MLA composite timing to distributed modes

Replace the old single-rank MLA composite helpers with stage-aware helpers that:

- shard each latent MLA component according to the current parallelism mode
- accumulate local FLOPs and local memory accesses for timing
- preserve global FLOP accounting for reporting
- attach stage-level collectives using MLA-specific byte formulas

Stage mapping:

- `qkv_proj`
  - compute: `D_proj_q`, `D_proj_kv`, `U_proj_q_rope`, `K_rope_proj`
  - forward comm: CP gather/broadcast of MLA state
  - backward comm: TP hidden-gradient reduction

- `attention`
  - compute: `attention_score_1`, `attention_score_2`, `attention_score_rope`, `attention_ctx_latent`
  - backward comm: CP reduce-scatter of gathered MLA cache-state gradients

- `output_proj`
  - compute: `O_proj_absorbed`
  - forward comm: TP output reduction
  - backward comm: CP all-gather of MLA cache state

### 3. Add MLA-specific communication formulas

Use MLA state sizes instead of dense K/V sizes:

- query state bytes:
  - local `D_proj_q` output
  - local `U_proj_q_rope` output

- cache state bytes:
  - local `D_proj_kv` output
  - local `K_rope_proj` output

For CP training/prefill the visible cache state is the CP-visible version of those tensors.
For CP decode the forward comm uses only the current query state, not the full hidden state.

### 4. Add MLA FlashAttention to the composite path

Keep the same latent attention FLOPs, but replace non-flash attention materialization with a flash-style memory envelope:

- no materialized attention-score storage
- flash forward uses query-state bytes + visible cache-state bytes + latent-context bytes
- flash backward uses a larger memory envelope and selective recompute of:
  - `attention_score_1`
  - `attention_score_2`
  - `attention_score_rope`
  - softmax forward

### 5. Fix MLA activation memory

Upgrade `llm_util.mla_activation_tensor_bytes(...)` so it models:

- TP sharding of latent/query/cache projection tensors
- CP sharding of query tokens
- full visible key length for attention-score storage in training/prefill
- FlashAttention removal of attention-score storage
- inference peak as the max of the local latent tensors, local attention-score workspace, and MLP workspace

Also thread explicit `key_seq_len` through the transformer memory helper so decode activation memory can see the KV-cache length rather than assuming `key_seq_len == query_seq_len`.

### 6. Fix MLA inference KV-cache sharding

For MLA inference memory:

- shard cache bytes by TP
- shard stored cache tokens by CP

This keeps MLA cache memory aligned with the newly supported distributed MLA modes.

## Validation Matrix

### Unit tests

`tests/test_mla_modeling.py` must cover:

- exact MLA parameter groups
- runtime latent shapes for training, prefill, and decode
- activation-memory formulas for:
  - baseline MLA
  - TP+CP sharded MLA
  - FlashAttention MLA
- stage FLOP accounting
- TP / TP+SP / CP / TP+CP config validation and execution
- stage communication bytes for TP, CP, and hybrid training
- FlashAttention selective recompute FLOPs
- inference decode CP query broadcast behavior
- inference KV-cache sharding with TP+CP

### Practical toy sweeps

Run toy analytical and hierarchical sweeps for both training and inference on:

- TP
- TP+SP
- CP
- TP+CP hybrid
- hybrid + FlashAttention
- hybrid + FlashAttention + MoE

Compare MLA against GQA on matched toy configs and confirm:

- the runs complete
- the new distributed MLA paths do not crash
- MLA remains faster than GQA in these toy cases

## Success Criteria

- MLA no longer rejects TP, CP, TP+SP, TP+CP, or FlashAttention.
- Training, prefill, and decode all run on the composite MLA path under the supported distributed modes.
- MLA communication bytes follow MLA latent-state sizes instead of dense K/V sizes.
- FlashAttention removes MLA attention-score storage and adds the expected selective-recompute backward FLOPs.
- MLA activation-memory formulas reflect TP, CP, and visible key length.
- MLA KV-cache memory shrinks correctly under TP and CP.
- Analytical and hierarchical toy sweeps succeed across dense and MoE cases.

## Out Of Scope

This plan still does not attempt to solve:

- flattened-mode MLA validation
- an exact congestion-aware communication model
- a full activation-lifetime simulator
- a separate MLA-specific distributed kernel implementation outside RAPID-LLM's current sharding contract
