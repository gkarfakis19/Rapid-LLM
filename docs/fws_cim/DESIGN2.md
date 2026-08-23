# FWS-CIM pass 2 — LLM prefill, MoE experts-per-chip, decode + KV stories, DSE

Status: locked for implementation (2026-08-23). Extends pass 1 (`DESIGN.md`, all of
whose laws, gates, and validation targets REMAIN BINDING — the 72/72 validator and
both GPU bit-identity gates are standing regression). Facts below come from a
5-agent recon over the current working tree and the OPTIMA reference; line numbers
cite the CURRENT uncommitted tree.

## 0. Scope

Pass 2A (workloads): dense-LLM prefill, MoE prefill (experts-per-chip), decode
with two KV stories (`cim_sram`, `cim_dram`). Pass 2B (mapping): auto placement
and a DSE tool rederiving the OPTIMA compiler's concepts.

Stays rejected, with clear messages: training (physics, permanent);
`use_flashattention: true` (fabric prices attention; flash tile calls would hit
the sa law per tile — recon-verified hazard); astra backend; cp>1; **MLA**
(new): `_mla_component_forward_stats` discards get_gemm_time's time and keeps
only mem_accesses (train_timing.py:3298-3305), so the CIM intercept's zero
mem_access tuple feeds an aggregate roofline and can hit the roofline
sys-exit trap — MLA needs its own seam, out of pass 2. Reject
`attention_type: mla` for fws_cim naming that reason.

## 1. Law changes and additions (all in `cim_timing.py`; single source of truth)

### 1.1 Attention law generalizes to CALL DIMS (refactor, must stay bit-green)

Today `price_gemm` prices attention from `tc.seq_len` (cim_timing.py:349-357).
Replace with call dims: for a call (m, k, n), fold heads into K exactly as now:
`sa_cycles(M=m, N=n, K_fold=k*heads_per_replica)` with
`heads_chip = ceil(kv_heads/tp)` (tp>=2) else kv_heads,
`heads_per_replica = ceil(heads_chip/replicas)`, total = max(QK, PV) +
fill_drain, N_mult unchanged (B*kv_heads or B*kv_heads/tp — recon re-verified
train_timing.py:1529, :1875). For MHA prefill (m=S, k=head_dim, n=S) this is
numerically IDENTICAL to the current law — T1/T2/T3 cycles must not move.
It additionally makes GQA prefill (m=S*shared_heads) and decode
(score: m=shared_heads, k=head_dim, n=context; output: m=shared_heads,
k=context, n=head_dim — llm_util.py:1103-1116) priced by the same folded
formula with no special cases. Document the GQA/decode folding semantics in the
module docstring (OPTIMA refuses GQA; this is our stated generalization).

### 1.2 Op-name normalization (decode interception fix)

Decode ops are named `decode_qkv_proj_f`, `decode_attention_score_f`,
`decode_attention_output_f`, `decode_output_projection_f`, `decode_ffn1_f`,
`decode_ffn2_f` (inference_timing.py:175,215,218,278,304,307). Today ALL of
them fall to the unknown-op analog fallback — decode attention would be
context-independent. Fix in `_classify_op`: strip a leading `decode_` prefix
and map `qkv_proj` → weight before prefix matching. Add these names to the
prefix tables: `router` (weight; arrays ceil(H/rows)*ceil(E/(cols*mux))),
`moe` (moe_dispatch/moe_combine are pure-comm OperationTimings that never
reach get_gemm_time — no pricing entry needed, but never warn on them either).
`ffn1_f_hot`/`_cold`/`_shared` already match the `ffn` prefix.

### 1.3 MoE laws (experts-per-chip)

New CimModelParams fields (duck-typed from the model config): `num_experts`,
`top_k`, `moe_intermediate_size`, `n_shared_experts`, `moe_layer_mask` (or the
(first_k_dense_replace, moe_layer_freq) pair), `expert_imbalance_factor`
(exists in model config; one-hot contract per MOE_ONE_HOT_EXPERT_MODEL.md:
hot expert factor alpha, beta=(E-alpha)/(E-1)), `vocab_size`.

Arrays for a MoE layer = attention arrays (3 QKV + 1 O; K/N law) + routed
experts E * [arrays_ffn1(K=H, N=(2 if gated else 1)*I_moe) +
arrays_ffn2(K=I_moe, N=H)] + shared experts n_shared * same + router arrays.
Gated-MLP note: the fused descriptor already arrives with N=2*I
(llm_util.py:617-621) — do NOT double again; derive gatedness from the model's
`uses_gated_mlp` source (train_timing.py:1088-1095), not model_type=='vit_dinov3'.

FWS routed-FFN stage time (all expert arrays coexist and fire in parallel —
the whole point of the user's experts-per-chip insight):
`T = tokens_hot * vec_latency` with
`tokens_hot = ceil(tokens_owner * top_k * alpha / E)` and
`tokens_owner = B * S` (cp==1 enforced). Shared experts run on their own
arrays concurrently over ALL owner tokens: `T_shared = tokens_owner *
vec_latency`. MoE FFN stage time = max(routed, shared). Dispatch/combine are
boundary transfers (1.5).

PER-OP PRICING (sequential figure) deliberately does NOT cancel the caller's
expert serialization: `_batched_gemm_forward_compute` prices one per-expert
shape and multiplies by expert count (train_timing.py:2356-2362). price_gemm
prices the per-expert call with the plain analog M-law; the outer multiplier
then models serialized experts — correct for the labeled sequential contrast,
wrong for FWS steady state. The SPATIAL REPORT is authoritative and computes
the parallel-expert law above directly from CimDeviceModel. Extend the
sequential disclosure to say experts are serialized in that figure.

### 1.4 Decode laws + KV stories

Decode weight ops: analog law, M = B (streams). Decode attention: the 1.1
generalized law at per-step context. Decode softmax: existing lanes law with
S*heads -> B-scaled elements at the step's context — softmax lanes law input
becomes `tokens_q * heads_chip` where tokens_q = m of the score call.

KV stories (config `inference.kvcache_type`; for fws_cim the value MUST be
`cim_sram` or `cim_dram` — `hbm_only` errors, "the device has no HBM"; update
the three pass-1 template YAMLs to `cim_sram`; the validator reads none of
them as inputs, so 72/72 is unaffected):

- `cim_sram`: KV lives in the fabric activation SRAM (the DRAM-stub tier,
  which pass 1 already labels the activation buffer). Timing: decode S2 =
  max(sa_time, softmax_time, kv_read_bytes / sram_bw) where kv_read_bytes =
  2 * context * head_dim * kv_heads_chip * kv_precision_bytes * B (GQA form;
  use the per-device SHARDED form — memory_estimation.py:602-613 — never the
  unsharded traffic helper llm_util.py:496, which overstates by tp).
  sram_bw = the stub DRAM tier bandwidth. Capacity: the existing memory pass
  already lands KV in static bytes vs the stub size (kv survives the pass-1
  weight exclusion) — extend the report label to say KV is included.
- `cim_dram`: new optional config block `cim.kv_dram { capacity_bytes,
  bandwidth_bytes_per_s, energy_per_bit_pj (default 0) }` (plain numbers only —
  the cim block rejects unit strings by design). Timing: same max() with the
  kv_dram bandwidth. Memory: zero `kv_cache_bytes_per_layer` in
  build_memory_data for this mode (mirror the device_class pattern at
  memory_estimation.py:155) so the SRAM check stays activations-only, and add
  a side check: num_layers * kv_bytes_per_layer(final context) vs
  kv_dram.capacity — reported, WARN on overflow (matches the existing
  warn-never-raise capacity precedent; do NOT hard-error).
- Report adds a KV section: kv_bytes_per_stream at final context, max_streams
  = floor(kv_capacity / per-stream bytes), max_context at configured B, story
  name, and the fits/warn flag. Energy: add kv_read/write bytes *
  energy_per_bit for cim_dram to the PARTIAL energy (still labeled partial).

Decode FWS report section: computed by DIRECT LAW EVALUATION at three contexts
(prefill+1, midpoint, final) — never plumbed through the per-step temporary
instances (they are discarded; recon-verified). Report per-context: stage
table, period, bottleneck, single-stream step latency (sum of stages +
boundaries), and aggregate tok/s = B / period_decode(final) alongside the
existing integrated decode totals (which now flow through CIM-priced ops via
1.2). State the staircase caveat: per-step times are stepwise in context, so
the existing trapezoid integration (simulate_inference_graph.py:242-307) is an
approximation between samples; recommend sample_every small enough for the
paper runs rather than changing the integrator.

### 1.5 Endpoints and boundary transfers generalize

Endpoints become a model-shaped list, not hard-named ViT stages:
ViT → patch_embed (M=seq) + vit_head (M=B); LLM → lm_head (linear_softmax:
arrays ceil(H/rows)*ceil(vocab/(cols*mux)), stage M = B*S prefill / B decode,
placed on the last chip). The LLM embedding lookup is NOT a GEMM (it is a
memory roofline, train_timing.py:2570-2588) — leave its pricing on the stub,
list it in the report as a note, and recommend `disable_embedding_unembedding`
for pure-transformer studies (it works at inference; recon-verified
train_timing.py:4355-4358, 4467-4470). MoE dispatch/combine: per-MoE-layer
boundary entries priced from the ep link (`self.links["ep"]`) with bytes =
tokens_owner * top_k * hidden * act_bytes each way (matches the balanced-A2A
sizing train_timing.py:4209-4212); checked against the period like pass-1
boundaries (warn when bound). Expert spreading: `cim.chip.moe_expert_parallel`
(int, default 1) spreads each MoE layer's routed experts over k chips —
divides per-chip expert arrays by k and dispatch/combine time by k (parallel
links), multiplies chip count; report expert chips as their own pool with
occupancy.

## 2. Config schema additions

```yaml
cim:
  analog: { ... unchanged ... }
  fabric: { ... unchanged ... }
  chip:
    arrays_per_chip: 400
    layers_per_chip: 32        # int | list | "auto" (NEW: derive from arrays_per_chip)
    moe_expert_parallel: 1     # NEW
  kv_dram:                     # NEW, optional; required iff kvcache_type == cim_dram
    capacity_bytes: 8589934592
    bandwidth_bytes_per_s: 100000000000
    energy_per_bit_pj: 0.0
  dse:                         # NEW, optional; consumed only by the DSE tool
    mux_candidates: [1, 2, 4, 8, 16]
    variants:                  # per-mux array points (rederive OPTIMA's materialization as data)
      - { adc_mux: 4, cols_adc: 320, energy_per_vec_pj: 9534.9389, area_mm2_per_array: 1.403065 }
      - ...
    tp_candidates: auto        # auto = divisors of num_heads; or explicit list
    max_chips: 0               # 0 = unbounded
inference:
  kvcache_type: cim_sram       # fws_cim: cim_sram | cim_dram (hbm_only errors)
```

`layers_per_chip: "auto"`: chips and the per-chip layer split are DERIVED
greedily from arrays_per_chip (capacity-first packing, layer order preserved,
endpoint arrays on first/last chips) — mapping becomes an output. Requires
arrays_per_chip > 0; error otherwise. Gate relaxations in
`_validate_fws_cim_model` (config.py:2636-2668): admit non-ViT LLMConfig at
inference (dense + MoE, mha/gqa only — reject mla per §0); admit decode_len>0;
keep training/flash rejections. Every new numeric field goes through
_parse_cim_float/_coerce_int (unit-string trap).

## 3. Integration points

- `cim_timing.py`: 1.1-1.5. CimModelParams gains the MoE/vocab fields;
  `from_model` reads them duck-typed with safe defaults (0/None ⇒ dense).
  Two model-param variants exist per MoE run (dense + moe intermediate — the
  SimpleNamespace ctx swap, train_timing.py:3900-3924): CimDeviceModel must
  take intermediate sizes per LAYER CLASS, not freeze one (recon risk).
- `inference_timing.py`: report generalization — per-layer-class stage tables
  (dense/MoE), endpoint list, KV section, decode section (direct law eval),
  dispatch/combine boundaries, disclosures extended (attention twice-per-layer
  AND serialized-experts). The report writer must handle `output/LLM` as well
  as `output/VIT` (run-mode dirs; tests/validators parameterize the path).
- `memory_estimation.py`: cim_dram KV zeroing + side check (§1.4).
- `config.py`: §2 schema + gate relaxations + kvcache_type validation.
- `base_timing.py`: no structural change expected (mode LLM already admitted);
  keep the GPU path byte-identical.
- Landmines carried forward: sys-exit roofline trap; O=0 templates; stash-on-
  self does not survive decode temp instances; run-order/output-dir clobbering
  (validator captures T3 before GPU gates — keep that ordering); *.md
  gitignore; convert() unit strings; N_mult only mirrors the MHA/GQA callers.

## 4. Validation matrix (2A)

Regression (unchanged, hard): pass-1 pytest suite green; 72/72 validator; both
GPU gates bit-identical; T1/T2/T3 cycles/times EXACT after the 1.1 refactor.

New law tests (exact, hand-computed in the test file):
- 1.1 equivalence: old-law vs call-dim law identical on T1/T2/T3 shapes.
- Decode attention cycles at (sh=1, d=128, ctx∈{129, 512, 4096}, kv_heads=32,
  tp=1, R=32, C=64): assert the closed form and monotone context growth.
- Decode classification: every decode_* name maps to the intended law; a
  decode run's attention time GROWS with context (kill the pass-1 silent
  fallback bug class forever).
- MoE: arrays/layer for a hand case (E=8, top_k=2, I_moe, shared=1, gated);
  routed stage time law incl. alpha; E=1,top_k=1,shared=0 reduces to the dense
  FFN stage time exactly (identity test); moe_expert_parallel divides arrays
  and dispatch time as specified.
- KV: per-stream bytes match memory_estimation's sharded law (import or
  mirror); max_streams/max_context arithmetic; cim_dram bandwidth binds S2
  when configured low (construct a case where kv term > sa term).
- lm_head arrays law (H=4096, V=32000, rows=4096, cols_adc*mux=4096 → 8).

End-to-end smokes (subprocess run_perf, assert exit 0 + report fields):
- Dense LLM prefill+decode: Llama2-7B_inf variant (flash false, cim_sram) on a
  new `fws_cim_llama7b.yaml` (rows=4096 arrays; kv_dram variant too).
- MoE: a GLM/DeepSeekMOE-derived non-MLA config (flash false, decode>0 ok) —
  expert arrays in capacity census, dispatch boundaries in report.
- ViT T1 unchanged output (bit-compare the pass-1 fws_cim_report.json fields).
- gpu_native variant still runs for LLM decode (native tile prices skinny ops).

## 5. Pass 2B — DSE tool (`tools/fws_cim_dse.py`)

Rederive the OPTIMA compiler concepts (recon-extracted) on CimDeviceModel:
- Closed-form evaluation only — never run_perf per candidate (sub-second
  sweeps; OPTIMA's own evaluator is sub-second for 10-20 candidates).
- Knobs: mux (from cim.dse.variants — each variant is a complete array point),
  tp (divisor candidates or pinned), moe_expert_parallel (list), chips DERIVED
  (auto placement §2), digital fabric fixed (pass 2B does not resize the
  fabric). Enumerate the small product exhaustively.
- Constraints per candidate: array capacity/placement feasibility, boundary
  and dispatch bandwidth vs period (reuse the report laws), KV capacity at the
  model's context, optional max_chips. Record EVERY infeasible candidate with
  a stage tag + message (fail-fast, self-explaining; OPTIMA concept).
- Metrics: period, fps or tok/s, single-item latency, chips, total array area,
  partial energy/inference, arrays utilization. Pareto front: throughput vs
  area (valid candidates only, 2-axis dominance). Final pick: lexicographic
  (max throughput, then min chips, min area, min tp, min mux) — state it.
- Artifacts: markdown report (selected mapping, candidate table, front,
  failures, config echo) + JSON; optional `--emit-config out.yaml` writes a
  concrete fws_cim hardware YAML for the selected point; `--verify` runs
  run_perf once on the emitted config and cross-checks period/fps against the
  DSE's closed-form numbers (must match ≤0.1% — the DSE and the simulator
  share CimDeviceModel, so this guards drift).
- Validation: T1 pinned — a DSE over variants containing exactly the T1 array
  point with tp=[1] must select a candidate whose period/fps/arrays equal the
  T1 targets; a variants list where the T1 point dominates must keep it on the
  front; an infeasible-only sweep must exit nonzero with the best violation.

## 6. File plan (2A then 2B)

2A: cim_timing.py (laws), config.py (schema/gates), inference_timing.py
(report), memory_estimation.py (KV), tests/test_fws_cim.py (append),
configs/hardware-config/fws_cim_llama7b.yaml (+ _kvdram variant, + moe
variant), configs/model-config LLM/MoE variants (flash false), pass-1 YAMLs
kvcache_type update, validation_scripts/validate_fws_cim_vs_optima.py (append
2A checks; keep 72 intact), docs/fws_cim/README.md update.
2B: tools/fws_cim_dse.py, cim dse schema in config.py, tests + validator
appendix, README update.

## 7. Rejected alternatives

- Cancelling the expert-serialization multiplier inside price_gemm: fragile
  coupling to _batched_gemm_forward_compute internals; the spatial report owns
  FWS-correct MoE timing instead.
- Plumbing per-step FWS data through decode temp instances: discarded objects;
  direct law evaluation replaces it.
- MLA support via the GEMM seam: impossible (time discarded upstream); own
  seam later.
- A 5th memory level for the KV tier: touches MemoryHierarchy everywhere;
  cim.kv_dram block instead.
- Changing the decode trapezoid integrator for staircase laws: documented
  caveat + sample_every guidance instead.

## 8. Erratum — post-audit physics fix pass (2026-08-23)

The whole-feature audit found the laws below physically impossible or
internally inconsistent as locked. `cim_timing.py` (the single source of
truth) and `docs/fws_cim/README.md` carry the corrected laws; the original
sections above are preserved unedited as the historical contract.

1. **Attention batch fold (supersedes the B-free reading of 1.1/1.4).**
   The SA law now takes `streams` (the B batch streams a wavefront
   carries), folded into the contraction exactly like heads:
   `run = sa(m, n, k*h_rep*streams)` (PV mirrored). LLM prefill and
   decode price `streams = B`; ViT and every recorded validation point
   stay at `streams = 1` (bit-identical — T1/T2/T3 unmoved). Rationale:
   S1/S3/S4/S5 price tokens = B*S while a B-free S2 exceeded the declared
   fabric's arithmetic peak by ~B x; softmax/kv_read in the SAME stage
   were already B-scaled. Prefill softmax tokens_q is B-scaled the same
   way (`streams * m` default).
2. **Sustained decode rate caps (supersedes the uncapped 1.4 formula).**
   `decode_sustained_throughput` = `min(resident * B / step_latency,
   B / period, kv_bw / kv_bytes_per_token)`; new limit tag
   `kv_bandwidth`. A pipeline with period P cannot beat B/P (the old
   fabric-limited branch overshot by up to ceil(x)/x), and all resident
   wavefronts read the ONE declared KV tier concurrently (the old figure
   implied W x the tier bandwidth).
3. **tp resource accounting (completes 1.1's tp sharding).** tp >= 2 is
   a system of tp shard devices: the DSE's chips/arrays/area metrics and
   max_chips constraint multiply per-shard figures by tp (census stays
   unsharded per shard — stated conservative upper bound); the report
   discloses `tp_shards` / `system_chips` / `system_area_mm2`. tp is no
   longer a free throughput knob.
4. **MoE dispatch/combine energy (completes 1.3/1.5).** Dispatch/combine
   are boundary transfers; their bytes now join the PARTIAL interconnect
   energy at the ep link's energy_per_bit (bytes are link-count-invariant
   under moe_expert_parallel).
5. **Decode-only runs rejected.** `decode_len >= seq_len` (prefill_len
   <= 0) silently skipped the whole FWS report while exiting 0; config
   validation now rejects it with the DSE's message.
6. Message hygiene: the "in pass 1" qualifiers left in the astra/cp
   rejection messages are dropped (both remain rejected); a CIM KV story
   on a non-fws_cim device now warns that it is ignored.
