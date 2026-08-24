# FWS-CIM device class

`device_class: fws_cim` models a fixed-weight-stationary compute-in-memory
accelerator inside Rapid-LLM. The model is device-generic: you supply array
geometry, energy, and area per array. The CTT-based OPTIMA model is only the
numeric validation reference.

## What the device class models

- **Analog weight arrays.** Every weight GEMM (patch embed, QKV, O-proj,
  FFN1/FFN2, MoE router and experts, classifier or LM head) is programmed
  into analog arrays once. All arrays that hold one weight matrix fire in
  parallel, so weight-GEMM time depends only on the token count, never on
  K or N.
- **Digital attention fabric.** Activation-by-activation GEMMs (QK^T, PV)
  run on a systolic-array sidecar, modeled with a closed-form cycle count
  verified bit-exact against recorded ScaleSim outputs. The same folded law
  prices MHA/GQA prefill and per-step decode at the arriving call dims.
- **Spatial pipeline.** Layers map to chips (`layers_per_chip`). A dense
  layer runs as five stages (QKV, attention, O-proj, FFN1, FFN2); a MoE
  layer runs as six (QKV, attention, O-proj, router, FFN1, FFN2). The
  pipeline period is the slowest stage over all layer classes plus
  endpoints; throughput is `1 / period`.
- **Autoregressive decode.** Decode weight stages run at `M = B` streams;
  decode attention grows with the step's context; decode stage 2 also
  covers the KV read (`cim_sram` or `cim_dram` story).
- **MoE experts-per-chip.** All routed expert arrays of a layer coexist
  and fire in parallel; dispatch/combine are boundary transfers on the
  `ep` link.
- **Chip-to-chip transfers.** Boundary activations move over the existing
  analytical p2p network (`bytes = tokens * hidden * act_bytes`).

All laws live in one module, `cim_timing.py` (`CimDeviceModel`). The per-op
GEMM path (`base_timing.get_gemm_time`) and the spatial report
(`inference_timing.calc_time`) both call it; no law is duplicated.

## Laws in brief

- Analog vector latency: `vec_latency = adc_mux * slice_cycles / f_analog`.
- Analog weight-GEMM time: `T = M_tokens * vec_latency` (K/N-independent).
- Arrays per K×N matrix: `ceil(K / rows) * ceil(N / (cols_adc * adc_mux))`.
- Systolic array: `cycles(M, N, K) = ceil(M/R) * ceil(N/C) * (K + R + C - 2) - 1`.
- Attention folds heads into K (dual-buffered, fully pipelined arrays) and
  prices the arriving call dims `(m, k, n)`: QK^T is `M=m, N=n,
  K=k*heads_per_replica*streams`, PV mirrors it. `streams` is the number
  of batch streams the wavefront carries: the B independent attention
  problems fold into the contraction exactly like heads (an LLM wavefront
  carries B streams, so S2 prices the same B streams whose tokens
  S1/S3/S4/S5 price; ViT keeps the one-image wavefront, streams = 1). MHA
  prefill at streams=1 is `m=S, k=head_dim, n=S` (the pass-1 law,
  bit-identical); GQA prefill folds shared heads into `m = S *
  shared_heads`; decode prices the score call `m=shared_heads, k=head_dim,
  n=context, streams=B` per step. QK and PV run concurrently:
  `total = max(QK, PV) + fill_drain_penalty`, default penalty `3 * rows`.
- Decode stage 2 is `max(T_sa, T_softmax, kv_read_bytes / kv_bw)`; decode
  weight stages run the analog law at `M = B`. The lm_head endpoint is
  `arrays = ceil(H/rows) * ceil(vocab/(cols_adc*adc_mux))` at `M = B*S`
  prefill / `M = B` decode.
- MoE routed-FFN stage: `T = ceil(tokens_owner * top_k * alpha / E) *
  vec_latency` (hot expert, one-hot imbalance factor alpha); shared experts
  run concurrently over all owner tokens; the stage is `max(routed,
  shared)`. At `E=1, top_k=1, shared=0` this reduces exactly to the dense
  FFN law.
- Softmax lanes: `cycles = pipeline_depth + ceil(tokens_q * heads_chip /
  lanes) - 1` (prefill: `tokens_q = streams * S` — the pass-1 form `S` at
  the recorded B=1 points; decode: `tokens_q = B * shared_heads`).
  Attention stage time is `max(T_sa, T_softmax)`.
- Energy per stage per layer: `E = M_tokens * energy_per_vec_pj *
  shots_per_output * arrays`. `shots_per_output` affects energy only.
- Macro resource model (QIF P2): an op pays
  `M_tokens * active_column_sets * slice_cycles / f_analog`. A column set is
  one mux slot (`cols_adc` stored columns, one ADC pass) and is the smallest
  allocatable unit; the charge is the worst macro's active-set count because
  the macros holding an op's tiles convert concurrently. One owner filling
  every mux slot pays `mux * slice_cycles` — `M * vec_latency`, the pass-1
  charge, bit-identically. Bit slicing: `n_s = ceil(weight_bits /
  bits_per_cell)` (1 unless a card opts in), stored-column demand `x n_s`,
  and a shift-add tree of `(n_s - 1)` adds per output at depth
  `ceil(log2(n_s))`, pipelined as `cycles = depth + ceil(results/lanes) - 1`.
  The per-macro digital pool carries `lanes = ceil(cols_adc * f_analog /
  slice_cycles / f_pool)` lanes and `lanes * (n_s - 1)` adders — derived and
  REPORTED (every `device_class: fws_cim` run prints a `[FWS-CIM] per-macro
  digital pool (derived, D12)` line), never a constraint.
- Area: `arrays * area_mm2_per_array` — analog only, the OPTIMA parity
  accounting. A shared digital chiplet card's declared `area_mm2` is a
  separate figure (`system_area_mm2`), never smeared into that total.
- Report metrics: `period = max(stage times)`, `fps = 1/period`,
  `block_latency = sum(S1..S5)`, `end_to_end_latency = num_layers *
  block_latency + endpoint stages + boundary transfers`.

## Assumptions (pass 2A)

- **Transformer inference only.** ViT-class models plus
  dense and MoE LLMs (`attention_type: mha`/`gqa`) with autoregressive
  decode. Training, flash attention, MLA, `cp > 1`, `pp > 1`,
  decode-only runs (`decode_len >= seq_len` — the spatial report prices
  the prefill wavefront, so `prefill_len` must be > 0), and the
  AstraSim backend are rejected at config validation with a message that
  names the offending setting. Training is permanently out: fixed weights
  admit no writes.
- **KV stories.** For `fws_cim`, `inference.kvcache_type` must be
  `cim_sram` (KV shares the activation SRAM — the DRAM-stub tier) or
  `cim_dram` (a dedicated `cim.kv_dram` tier); `hbm_only` errors — the
  device has no HBM. Decode stage 2 is
  `max(T_sa, T_softmax, kv_read_bytes / kv_bw)`; the report adds a KV
  section (bytes/stream, max streams, max context, fits flag — capacity
  overflow WARNS, never fails) and a decode section evaluated directly
  from the laws at three contexts (first, midpoint, final).
- **Decode throughput: fabric ceiling vs sustained.** The aggregate
  `B / period` figure is a FABRIC CEILING: it assumes the spatial
  pipeline is fully occupied, which takes `wavefronts_full =
  ceil(step_latency / period)` wavefronts (each one batch of B streams)
  in flight. The KV capacity at the same context holds only
  `wavefronts_kv = floor(kv_max_streams / B)` wavefronts, so the report
  also prints the sustained figure
  `min(min(wavefronts_full, wavefronts_kv) * B / step_latency,
  B / period, kv_bw / kv_bytes_per_token)` with its limiting factor
  (`fabric`, `kv_capacity`, `kv_bandwidth`, or `infeasible` when the KV
  capacity holds no full wavefront). The `B / period` cap holds because
  the bottleneck stage completes at most one wavefront of B per period;
  the bandwidth cap holds because every resident wavefront's kv-reading
  S2 stage draws on the ONE declared KV tier concurrently
  (`kv_bytes_per_token` = the per-device KV bytes one generated token
  reads across all layers). JSON: `sustained_tokens_per_s`,
  `wavefronts_full`, `wavefronts_kv`, `decode_throughput_limit` next to
  the unchanged `aggregate_tokens_per_s_final`.
- **Decode staircase caveat.** Per-step decode times are stepwise in
  context (systolic tile bins), so the integrated decode totals — a
  trapezoid over sampled steps — are an approximation between samples.
  For paper runs set `inference_param.sample_every` small enough to
  resolve the staircase (`1` prices every step exactly; `-1` prices only
  the first and last steps, which is the smoke-config setting). The decode
  section's direct-law numbers at the three contexts are exact either way.
- **MoE experts-per-chip.** All routed expert arrays coexist and fire in
  parallel; the routed stage is `ceil(tokens_owner * top_k * alpha / E) *
  vec_latency`, shared experts run concurrently over all owner tokens, and
  dispatch/combine are per-MoE-layer boundary transfers on the `ep` link.
  `cim.chip.moe_expert_parallel: k` spreads each layer's routed experts
  over k dedicated chips (reported as their own pool). The per-op
  sequential figure keeps the caller's serialized-expert multiplier and
  the report discloses it.
- **Placement.** `cim.chip.layers_per_chip` takes an int, a per-chip list,
  or `auto` (greedy capacity-first packing from `arrays_per_chip`; the
  derived split is reported).
- **Helper absorption.** LN / GELU / adder lanes are sized so they never
  bound a stage (the OPTIMA contract). Analog stage time is the analog law;
  helpers are absorbed. The report states this.
- **Partial energy.** The energy figure covers analog arrays plus boundary
  interconnect — chip-boundary pp transfers AND MoE dispatch/combine
  bytes on the ep link (link-count-invariant under `moe_expert_parallel`)
  — plus `cim.kv_dram` KV traffic when that story is
  configured. No fabric, SRAM, or helper energy; the report labels it
  PARTIAL.
- **Macro resource model is law-only (QIF P2).** Cards, tiles, active-
  column-set pricing, bit slicing with its reduction descriptors, and the
  per-macro pool sizing are pure functions plus config plus tests. Nothing
  is wired into the closed-form spatial report, which stays legacy-frozen:
  the shipped configs produce byte-identical reports with or without an
  explicit (inert) `cim.cards` / `cim.allocation` block, and a test gate
  proves it.
- **Folded-K attention.** Per-head fill/drain is not priced; a single
  configurable penalty (`fill_drain_penalty_cycles`, default `3*rows`)
  covers it.
- **Mapping is given or derived, never searched by the simulator.**
  `layers_per_chip` comes from the config (int, list, or `auto`); capacity
  (`arrays_per_chip`) is validated, and overflow is a hard error. The
  design-space search lives in a separate tool, `tools/fws_cim_dse.py`
  (see "DSE tool" below); a run_perf run never sweeps anything.
- **pp must be 1.** Chip placement is a post-hoc report, not pipeline
  parallelism. Weights live in arrays, so weight bytes are excluded from the
  memory tables; the DRAM capacity check becomes an activation-feasibility
  check.
- The sequential per-op total in the results file is kept for contrast only.
  Two disclosures ride with it (in the results text and the report's
  `sequential_latency_note`): under `fabric.model: sa` it counts the one
  folded QK+PV fabric run twice per layer (both attention ops return the
  same folded run time), and on MoE models it prices the experts serialized
  (the caller multiplies one per-expert call by the expert count). The FWS
  spatial pipeline section is the authoritative number for both.

## How to run

From the repo root, with the project virtualenv:

```bash
# T1: ViT-Huge-story, seq 64, adc_mux 4
.venv/bin/python run_perf.py \
  --hardware_config configs/hardware-config/fws_cim_optima_t1.yaml \
  --model_config configs/model-config/vit_huge_story_64_inf.yaml

# T2: ViT-g, seq 64, adc_mux 2
.venv/bin/python run_perf.py \
  --hardware_config configs/hardware-config/fws_cim_optima_t2.yaml \
  --model_config configs/model-config/vit_g_64_inf.yaml

# T3: ViT-Huge-story, seq 196, adc_mux 16
.venv/bin/python run_perf.py \
  --hardware_config configs/hardware-config/fws_cim_optima_t3.yaml \
  --model_config configs/model-config/vit_huge_story_196_inf.yaml

# Dense LLM prefill + decode: Llama2-7B-class, B=4, prefill 1792, decode 256,
# cim_sram KV story (swap the hardware config for fws_cim_llama7b_kvdram.yaml
# to move the KV cache to a dedicated cim.kv_dram tier)
.venv/bin/python run_perf.py \
  --hardware_config configs/hardware-config/fws_cim_llama7b.yaml \
  --model_config configs/model-config/llama2_7b_fws_inf.yaml

# MoE prefill + decode: GQA, E=16, top_k=2, 1 shared expert, experts-per-chip
.venv/bin/python run_perf.py \
  --hardware_config configs/hardware-config/fws_cim_moe.yaml \
  --model_config configs/model-config/moe_small_fws_inf.yaml
```

Each run writes:

- `output/<MODE>/LLM_inference_results.txt` (`VIT` or `LLM` per the model
  config) — the usual results plus a readable `FWS-CIM spatial pipeline`
  section.
- `output/<MODE>/fws_cim_report.json` — machine-readable: period, fps,
  block latency, end-to-end latency, per-layer-class stage tables (dense
  and MoE) with QK/PV cycles, endpoint list (ViT: patch_embed/vit_head;
  LLM: lm_head, with the embedding lookup as a note), array counts, area,
  per-chip occupancy, boundary and MoE dispatch/combine times, KV section,
  decode section, partial energy.

Validation against the recorded OPTIMA targets (cycles and array counts
exact; times, areas, and fps within 0.1 %), plus the two GPU regression
gates, an A100 comparison, the pass-2A rows (dense-LLM, cim_dram, and
MoE smokes plus law spot-checks the script recomputes independently from
the closed forms), and the pass-2B DSE rows (a T1-pinned sweep selection
with its Pareto front, a Llama2-7B sweep with the `--emit-config`/
`--verify` round trip, and the MoE `moe_expert_parallel` scaling checks;
these DSE runs write only under `output/fws_cim_dse/validation_*`):

```bash
.venv/bin/python validation_scripts/validate_fws_cim_vs_optima.py
```

Exit code 0 means every check passed. Unit tests:
`.venv/bin/python -m pytest -q tests/test_fws_cim.py`.

## DSE tool (pass 2B)

`tools/fws_cim_dse.py` sweeps the candidate space declared in `cim.dse`
with CLOSED-FORM evaluation on `CimDeviceModel` only — it never runs
run_perf per candidate. Knobs: array variants (`cim.dse.variants`, each a
complete analog point after `cim.analog` inheritance), tp
(`tp_candidates`; `auto` = divisors of `num_heads`, MoE-filtered by the
routing-group divisibility gate), and `moe_expert_parallel`
(`cim.dse.moe_expert_parallel`, MoE models only). tp >= 2 evaluates a
SYSTEM of tp shard devices: the timing laws shard kv heads and KV bytes
per device, and the chips / arrays / area metrics (plus the `max_chips`
constraint) multiply the per-shard figures by tp — the array census does
not shard weight matrices, so each shard is counted at the full
per-device figure, a stated conservative upper bound. tp therefore costs
real silicon in the selection instead of acting as a free throughput
knob.
`cim.dse.mux_candidates` never expands the sweep: when set it is a
cross-check only — every variant's `adc_mux` must appear in it (a typo
guard), or the tool exits with a config error. Chips are DERIVED per
candidate via the `layers_per_chip: auto` greedy placement; the digital
fabric is fixed. Constraints per candidate (every infeasible candidate is
recorded with a stage tag and message): placement/capacity, optional
`max_chips`, boundary and MoE dispatch bandwidth vs the pipeline period,
and story-aware KV capacity. Metrics mirror the spatial report: period,
throughput (fps for prefill/ViT workloads; on decode workloads the
fabric-ceiling tok/s at the final decode context plus the sustained
figure `sustained_tokens_per_s` with `wavefronts_full` / `wavefronts_kv`
/ `decode_throughput_limit`), single-item latency, chips (backbone +
expert pool), the derived per-chip layer split, one-way MoE dispatch time
(`dispatch_time_us` in the JSON metrics; `moe_expert_parallel: k` divides
it by k), total array area, arrays utilization, PARTIAL energy per
inference. The Pareto front is throughput vs total array area; the final
pick is lexicographic per `--objective` (`throughput`, the default: max
throughput, then min chips, min area, min tp, min adc_mux; `min_chips`:
min chips first, same tie-breakers). On decode workloads both the front's
throughput axis and the selection rank on `sustained_tokens_per_s` — the
honest headline; the fabric ceiling stays as info. An all-infeasible
sweep exits nonzero and surfaces the best violation.

```bash
.venv/bin/python tools/fws_cim_dse.py \
  --hardware_config configs/hardware-config/my_fws_cim_with_dse.yaml \
  --model_config configs/model-config/llama2_7b_fws_inf.yaml \
  --emit-config out/selected.yaml --verify
```

Artifacts land in `--output-dir` (default
`output/fws_cim_dse/<hw-stem>__<model-stem>/`): `dse_report.md` (selected
mapping, candidate table with ok flags, front, failures, config echo) and
`dse_report.json` (machine-readable). `--emit-config` writes a complete
runnable fws_cim hardware YAML for the selected point (chosen variant's
analog fields, the DERIVED `layers_per_chip` list, tp, and
`moe_expert_parallel`). `--verify` then runs run_perf once on the emitted
config (with the run's `output/` tree placed inside the DSE output dir)
and cross-checks period and fps (prefill) or the fabric-ceiling tok/s
plus the sustained tok/s (decode) against the closed form — they must
match within 0.1% (the DSE and the simulator share `CimDeviceModel`, so
this guards drift); a mismatch exits nonzero.

### Worked example (dense Llama2-7B decode; real output)

The shipped templates carry no `cim.dse` block (a schema test pins that),
so copy one and append the candidate space. This is the exact sweep the
validator reruns as its pass-2B llama7b rows: the shipped Llama analog
point plus a deliberately infeasible tiny array.

```yaml
# copy of configs/hardware-config/fws_cim_llama7b.yaml, plus under cim:
  dse:
    variants:
      - { adc_mux: 4, cols_adc: 1024,          # the shipped Llama point;
          energy_per_vec_pj: 97637.774,        # rows/slice_cycles/clock
          area_mm2_per_array: 14.367386 }      # inherit cim.analog
      - { adc_mux: 1, cols_adc: 8, rows: 64,   # too small to place a layer
          energy_per_vec_pj: 1.0, area_mm2_per_array: 0.001 }
    tp_candidates: [1]
```

```bash
.venv/bin/python tools/fws_cim_dse.py \
  --hardware_config my_llama_dse.yaml \
  --model_config configs/model-config/llama2_7b_fws_inf.yaml \
  --output-dir output/fws_cim_dse/validation_llama7b \
  --emit-config output/fws_cim_dse/validation_llama7b/selected_config.yaml \
  --verify
```

Printed output from this run:

```
[FWS-CIM DSE] 1/2 candidates valid; artifacts in output/fws_cim_dse/validation_llama7b
[FWS-CIM DSE] selected c000 (lexicographic: max sustained throughput (sustained_tokens_per_s; the fabric-ceiling tok/s stays as info), then min chips, then min total array area, then min tp, then min adc_mux): period 27197.5 us, throughput 7205.28 tok/s, chips 4, area 6091.77 mm2
[FWS-CIM DSE] sustained 449.187 tok/s at KV capacity (wavefronts: full 33, kv 2; limited by kv_capacity); the throughput figure above is the fabric ceiling.
[FWS-CIM DSE] verify: run_perf matches the closed form (<= 0.1%).
```

What the artifacts recorded (`dse_report.md` / `dse_report.json`):

- Selected mapping `c000`: the shipped array point at tp=1; decode
  workload; period 27197.5 us (bottleneck `S2_attention`), throughput
  7205.28 tok/s (the fabric ceiling B=4 / decode period 555.148 us at
  the final context; filling the pipeline would take 33 wavefronts =
  132 streams), sustained 449.19 tok/s (the KV tier holds 8 streams =
  2 wavefronts of B=4, so `kv_capacity` limits — this is the decode
  selection key), single-item latency 5.29214e+06 us, 424 arrays, total
  array area 6091.77 mm2, utilization 0.8833, PARTIAL energy/inference
  5.94897e+11 pJ.
- Auto placement derived `layers_per_chip [9, 9, 9, 5]` — 4 chips.
  Greedy capacity-first packs 9 Llama layers per chip (9 x 13 = 117
  arrays <= 120), one more than the shipped manual `layers_per_chip: 8`;
  the last chip carries 5 layers plus the 8 lm_head arrays.
- The tiny variant is recorded, not dropped:
  `c001 [placement]: cim.chip.layers_per_chip: 'auto' cannot place
  layer 0 (dense): it needs 395264 analog arrays on one chip but
  cim.chip.arrays_per_chip = 120. ...`
- The Pareto front is `[c000]`, and `--verify` reran the emitted config
  through run_perf: period, the fabric-ceiling tok/s, and the sustained
  tok/s matched the closed form with rel_err 0.0 (the run's `output/LLM`
  tree lands inside the DSE output dir, so repo outputs stay untouched).

Note when comparing to OPTIMA outputs directly: OPTIMA's recorded block
latency sums six stages (S1..S5 plus a trailing peripherals stage that costs
one analog stage time). The report's `block_latency_us` is `sum(S1..S5)` per
the design contract; add one analog stage time to reproduce OPTIMA's figure.

## Mapped-path DSE (QIF P3.7)

`tools/fws_qif_dse.py` is the SECOND DSE and the one D10 means by
"allocation is swept by the DSE". It shares the conventions above and
nothing else: every candidate is evaluated through the REAL mapped path —
`fws_mapping.build_mapping` -> `program.fws_build.build_fws_program` ->
`fws_eval.evaluate_fws` — so each candidate's headline is `tokens/s` at
the decode terminal read off ITS OWN timeline (ADJ-6, A1). There is no
closed form in it. The pass-2B tool above stays exactly as it is: its
evaluator is `CimDeviceModel` laws, its schema names `period_us` and the
ceiling/sustained pair ADJ-6 retired, and the validator reruns its sweeps
to hold the ADJ-8 bridge — one evaluator per tool, one accounting per
metric.

The candidate space is a `mapping_dse:` block, a SIBLING of `mapping:`
that the tool owns, strict-keys itself, and strips before the config
parser sees the file (so a config carrying it still runs through
`run_perf` as the single point its `cim:`/`mapping:` blocks declare).
EXPLICIT candidate lists only: the sweep is the full cross product in
declaration order, every point is evaluated, and nothing is sampled or
capped. Six axes, each moving exactly one field:

| axis | field it moves |
|---|---|
| `vector_lanes` | `cim.cards.<digital card>.vector_lanes` (D13) |
| `bank_depth` | `cim.cards.<analog card>.bank_depth` (A3/D10 banking) |
| `column_sets_per_tile` | `cim.allocation.column_sets_per_tile` (D10) |
| `arrays_per_chip` | `cim.chip.arrays_per_chip` |
| `layers_per_chip` | `mapping.layers_per_chip` (int, list or `auto`) |
| `shared_chiplets` | `mapping.shared_chiplets` (ADJ-5) |

`bank_depth` and `column_sets_per_tile` both set one law and
`cim.allocation` WINS; a candidate declaring both records the EFFECTIVE
granularity and the precedence. A card knob whose `cim.cards` block the
config never declared is refused by name rather than invented.

Stages, in order, each recorded with the refusal's own message:
`config < mapping < budget < lowering < pricing < memory`. `budget` is
`max_chips` / `max_silicon_mm2`; `memory` is P4.3's measured high-water
mark against the declared tier. The Pareto front is headline tokens/s vs
total silicon (enumerated analog macro slots x `macro_footprint_mm2` plus
the declared shared-chiplet area, with any UNCOVERED term named; that
footprint law divides by the card's `stack_3d_height`, so on a 3D card the
axis ranks package footprint rather than silicon area), and the
report states the front's SHAPE, read off the data: a one-point front on
flat area means no swept axis trades silicon for speed.

```bash
.venv/bin/python tools/fws_qif_dse.py \
  --hardware_config configs/hardware-config/fws_cim_granite_tiny_dse.yaml \
  --model_config configs/model-config/granite_4_0_h_tiny_inf.yaml \
  --model_id Granite-4.0-H-Tiny \
  --output-dir docs/qif/dse/granite_lanes_banks \
  --emit-config configs/hardware-config/fws_cim_granite_tiny_dse_selected.yaml \
  --verify
```

### The checked-in demo sweep (real output)

Granite-4.0-H-Tiny over `vector_lanes {512, 1024, 2048, 4096}` x
`bank_depth {1, 2}` — 8 candidates, 8 valid, ~100 s of wall clock, every
candidate placed and priced on its own timeline. Artifacts:
`docs/qif/dse/granite_lanes_banks/dse_report.{json,md}`; the winner is
`configs/hardware-config/fws_cim_granite_tiny_dse_selected.yaml`.
`tests/test_qif_dse_allocation.py` regenerates the sweep and compares the
artifact; the validator reads it for the selection rows and runs a live
Llama sweep for the `--verify` round trip.

| vector_lanes | bank_depth | tokens/s | silicon mm2 | tiles |
|---|---|---|---|---|
| 512 | 1 | 3432.12 | 12930.6 | 19566 |
| 512 | 2 | 3414.40 | 12930.6 | 11103 |
| 1024 | 1 | 4579.18 | 12930.6 | 19566 |
| 1024 | 2 | 4547.69 | 12930.6 | 11103 |
| 2048 | 1 | 5497.92 | 12930.6 | 19566 |
| 2048 | 2 | 5452.59 | 12930.6 | 11103 |
| 4096 | 1 | **6110.95** | 12930.6 | 19566 |
| 4096 | 2 | 6055.00 | 12930.6 | 11103 |

What the sweep found:

- The headline scales with the declared scan-engine width, SUB-linearly:
  8x the lanes buys 1.79x the tokens/s, because 36 Mamba-2 layers put the
  Mamba-2 scan on the critical path but the analog GEMMs, the pool ops and
  the chip-boundary transfers do not move with `vector_lanes`. The decode
  steps the headline is read at are priced by the per-token recurrence
  `ssm_recurrent_scan[mamba2]` (`validated = optima_m3_reference`); the
  chunked `ssd_chunked_scan` form (UNVALIDATED) prices prefill only, since
  `fws_eval` chunks on `phase == "prefill"` and the scan law falls back to
  the recurrence at `tokens <= 1`. Both forms run on the same vector
  engine. The declared 1024-lane point is the shipped config's, and every
  scan number in it scales with a width no measured silicon has supplied.
- Finer banking is faster and cheaper in energy (bank 1 vs bank 2:
  +0.5 to +0.9 % tokens/s, -8.6 % energy at every lane count). The
  mechanism is ADJ-4's active-column-set pricing, read off the report: the
  WASTE does not move (`unowned_columns` is 2,671,424 and
  `macros_holding_tiles` is 5610 in all eight rows), and the whole energy
  delta sits in the `analog_arrays` component (1.6418e11 pJ at bank 1 vs
  1.7998e11 pJ at bank 2, with `link_traffic` bit-identical) — a finer
  bank cuts the ACTIVE column sets an evaluation lights up, not the stored
  columns it strands. A finer bank therefore makes MORE tiles (19566 vs
  11103), each narrower. It changes what is held INSIDE a macro, never how
  many macros exist: the slot count, the chip count and the silicon are
  identical across all eight rows.
- The front is therefore ONE point (`flat_area`). That is the finding,
  not a broken sweep, and the report DERIVES it rather than asserting it:
  both terms of the silicon accounting are constant over the sweep (6400
  ENUMERATED analog macro slots and 10 shared digital chiplets on every
  candidate), because neither swept axis moves `arrays_per_chip`, the chip
  count or the chiplet count. The shared digital chiplet's area is an
  UNCOVERED term (the card declares no `area_mm2`; ADJ-4 forbids inventing
  one), which the report names. A front spreads only once a swept knob has
  a declared area law or a chip split trades silicon for capacity.
- `--verify` re-ran the winner through `run_perf` from the emitted YAML:
  tokens/s, prefill latency, the median decode step, the window latency,
  the request rate and the chip / tile / slot counts all matched at
  rel_err 0.0.

## Config reference

Hardware YAML (see `configs/hardware-config/fws_cim_optima_t1.yaml` for a
complete template). Use plain numbers inside `cim:` — never unit strings
such as `"100 MB"`; the global config preprocessor rewrites those into byte
counts before the CIM parser sees them.

```yaml
device_class: fws_cim        # absent -> gpu (existing behavior, bit-identical)
cim:
  analog:
    rows: 1280               # array rows (K per array tile)
    cols_adc: 320            # ADC columns per array
    adc_mux: 4               # column banks per ADC; stored cols = cols_adc*adc_mux
    slice_cycles: 2          # analog cycles per ADC evaluation
    analog_clock_mhz: 100
    energy_per_vec_pj: 9534.9389   # per array per input vector (includes mux)
    shots_per_output: 2      # energy only, never time
    area_mm2_per_array: 1.403065   # 0 disables area reporting
  fabric:
    model: sa                # sa | gpu_native
    rows: 32
    cols: 64
    num_arrays: 2            # QK and PV run concurrently on these
    replicas: 1
    clock_ghz: 0.95
    fill_drain_penalty_cycles: 96  # default 3*rows when absent
    softmax_lanes: 1
    softmax_pipeline_depth: 20
  chip:
    arrays_per_chip: 400     # capacity check; 0 = unchecked
    layers_per_chip: 32      # int | list ([16, 16]) | "auto" (derive from arrays_per_chip)
    moe_expert_parallel: 1   # spread each MoE layer's routed experts over k chips
  kv_dram:                   # optional; required iff inference.kvcache_type: cim_dram
    capacity_bytes: 8589934592           # 8 GiB (plain numbers, no unit strings)
    bandwidth_bytes_per_s: 100000000000  # 100 GB/s
    energy_per_bit_pj: 2.0               # feeds the PARTIAL energy; 0 disables
  cards:                     # optional (QIF P2.1); absent -> a card is synthesized
                             # from cim.analog + cim.fabric with every knob inert
    ctt:                     # the analog macro card
      kind: analog_macro     # analog_macro | digital_chiplet
      device: ctt            # ctt | reram | mram (reram/mram are NAMED EMPTY
                             # SLOTS: schema only, no shipped numbers, so a card
                             # on them must supply its own complete params block)
      # params: {...}        # a full cim.analog parameter set; omitted -> inherit
      bits_per_cell: 2       # slicing is present IFF bits_per_cell < weight_bits
      weight_bits: 8         # declare both or neither (absent -> no slicing)
      slicing: column_sets   # column_sets | chained_macros
      bank_depth: 1          # mux slots per allocatable bank; must divide adc_mux
                             # (absent -> the whole macro = dedicated per matrix)
      stack_3d_height: 1     # divides the package FOOTPRINT only, never silicon
      validity:              # admitted points; empty -> the card declares no menu
        - { bits_per_cell: 2, weight_bits: 8, mux: 4 }
      pool_clock_ghz: 0.0    # per-macro digital pool; 0 -> inherit the fabric clock
      pool_energy_per_add_pj: 0.0   # 0 reports zero (no invented numbers)
      pool_area_mm2_per_adder: 0.0  # 0 reports zero
      bank_switch_cycles: 0  # analog cycles lost between active column sets.
                             # 0 is a DISCLOSED relaxation, not an omission:
                             # no shipped device declares a switch cost, and
                             # OPTIMA priced switching at zero silently
                             # (AUDIT finding 3). A card that knows its number
                             # declares it and every op pays (active - 1) x it.
    shared_digital:          # the shared digital chiplet card (SA + softmax laws)
      kind: digital_chiplet
      # params: {...}        # a full cim.fabric parameter set; omitted -> inherit
      area_mm2: 0.0          # chiplet silicon; reported in system_area_mm2 and
                             # never folded into the analog (parity) area total.
                             # There is no energy knob: no law prices a fabric
                             # op's energy yet, so declaring one is refused.
    # default_analog: ctt    # which card the laws use (default: the first one)
    # default_digital: shared_digital
  allocation:                # optional (QIF P2.2); ABSENT = dedicated per matrix
    column_sets_per_tile: 4  # allocation granularity in mux slots (must divide it)
    assignments:             # explicit tile placement, capacity-checked while
                             # the hardware YAML is parsed: an out-of-range or
                             # doubly-claimed column set is a hard error on the
                             # run path, not only under the device model
      - { model: vit, layer: 0, op: qkv, expert: -1, shard: 0,
          slice_index: 0, macro: 0, column_sets: [0, 1] }
  dse:                       # optional; consumed only by tools/fws_cim_dse.py
    mux_candidates: [1, 2, 4, 8, 16]  # optional cross-check: every variant's adc_mux
                                      # must appear here; the sweep enumerates
                                      # variants, never this list
    variants:                # per-mux array points; omitted fields inherit cim.analog
      - { adc_mux: 4, cols_adc: 320, energy_per_vec_pj: 9534.9389, area_mm2_per_array: 1.403065 }
    tp_candidates: auto      # auto = divisors of num_heads; or an explicit list
    moe_expert_parallel: [1] # expert-spreading candidates (MoE models only)
    max_chips: 0             # 0 = unbounded
inference:
  kvcache_type: cim_sram     # fws_cim: cim_sram | cim_dram (hbm_only errors — no HBM)

mapping:                     # optional TOP-LEVEL block (QIF P3.1, ADJ-5).
                             # ABSENT = the derived dedicated mapping, which
                             # reproduces cim.chip.layers_per_chip exactly.
                             # Requires device_class: fws_cim.
  chips: 8                   # total ANALOG chips; CHECKED against the chips the
                             # placement enumerates (a mismatch is a hard error,
                             # and a chip count is never a quotient — D21)
  macros_per_chip: 120       # macro slots per analog chip; must agree with
                             # cim.chip.arrays_per_chip when that is declared
  shared_chiplets: 2         # shared digital chiplets (D13). A CONFIG INPUT; the
                             # derived suggestion (one per analog chip) is always
                             # reported next to the declared value
  parallelism:               # tp and ep must AGREE with parallelism.tp and
    tp: 2                    # cim.chip.moe_expert_parallel (one degree, one home).
    ep: 1                    # pp is the exception: it is an INDEPENDENT annotation
    pp: 1                    # over chips (ADJ-5), so mapping pp > 1 is legal while
                             # the hardware parallelism.pp stays 1
  layers_per_chip: 8         # int | list | "auto"; absent -> the cim.chip value
  membership:                # per-analog-chip axis indices, in chip-id order;
    pp: [0, 0, 1, 1, 0, 0, 1, 1]   # an axis you do not list is derived
  decode_window: 2           # decode steps the DAG lowers (ADJ-6); the truncation
                             # is disclosed, never silently extrapolated
  pd:                        # optional PD disaggregation (D16): TWO inventories,
    prefill: {shared_chiplets: 4}  # two mappings, one handoff priced as bytes.
    decode: {shared_chiplets: 1}   # Setting both halves equal reproduces the
                             # unified machine. Both halves are required.
```

Shipped pass-2A templates: `fws_cim_llama7b.yaml` (dense LLM, `cim_sram`),
`fws_cim_llama7b_kvdram.yaml` (same device, `cim_dram` KV tier), and
`fws_cim_moe.yaml` (MoE experts-per-chip), paired with the model configs
`llama2_7b_fws_inf.yaml` and `moe_small_fws_inf.yaml`
(`use_flashattention: false`, small batch/decode for smoke speed).

Notes:

- `fabric.model: gpu_native` sends the attention act-GEMMs through the
  native tile/roofline machinery instead of the SA law. The stub
  `tech_param` then describes the sidecar, so calibrate it to the fabric
  (peak flops ≈ `num_arrays * rows * cols * 2 * clock`), not to a GPU.
- fws_cim YAMLs still need structurally complete `tech_param`,
  `memory_hierarchy`, `parallelism`, and `network` blocks (ring dims,
  `kernel_launch_overhead: 0`). Copy them from a shipped template.
- Model YAMLs may set `model_param.vision.num_prefix_tokens` (int >= 0) to
  override the default prefix-token count (vit: 1, vit_dinov3: 5). The
  validation configs use 0 to hit the reference sequence lengths exactly.
- `layers_per_chip: "auto"` requires `arrays_per_chip > 0` and packs layers
  greedily (capacity-first, layer order preserved, endpoints on the first
  and last chips). The derived split is reported
  (`derived_layers_per_chip` in the JSON; `auto -> derived [...]` in the
  text).
- `inference.kvcache_type` accepts only `hbm_only`, `cim_sram`, or
  `cim_dram` — enforced at parse time for every hardware config. For
  `fws_cim`, `hbm_only` errors (the device has no HBM) and `cim_dram`
  requires the `cim.kv_dram` block. On a non-`fws_cim` device a CIM KV
  story is inert: validation warns that it is ignored and the run behaves
  as `hbm_only`.
- fws_cim model configs must keep `decode_len < seq_len`
  (`prefill_len > 0`): the spatial report prices the prefill wavefront,
  so a decode-only run is rejected at validation (matching the DSE).
- The `mapping:` block is a CLAIM the tool checks, not an input that wins
  over the machine it describes: `chips`, `macros_per_chip`, `parallelism`
  and `membership` are each validated against the placement
  `fws_mapping.build_mapping` enumerates, and every refusal names the
  offending setting with a stage tag (`residency`, `assembly`, `package`,
  `execution`, `annotation`).
- The mapping PLACES; it prices nothing. Every op in the DAG it emits
  carries a zero duration and the annotation P4's pricing table needs
  (per-op device class, tiles, bytes, owner, shard group, law name). A
  duration written by P3 would be a second accounting of a metric P4 owns.

## Mapping and the system atlas (QIF P3)

`fws_mapping.build_mapping(hw, model)` returns the mapping object: tiles on
macro column sets, macros on integer chips, ops on typed devices (analog
macro, per-macro digital pool, shared digital chiplet, link), and tp/ep/pp
annotations constructed from (axis, coords) by
`program.groups.CommunicatorFactory`. `program.fws_build.build_fws_program`
lowers it into the rewrite's program IR — a placed, annotated, UNPRICED DAG.
`FwsMapping.boundary_table()` reports per-boundary bytes and, when the caller
supplies a period and names its basis, the rate against the declared link
bandwidth; a violated boundary is REPORTED, not priced (D17), and the report
says so.

```bash
# Write an fws_atlas/1 document and open it with docs/qif/atlas/atlas.html
python3 tools/fws_emit_atlas.py \
  --hardware_config configs/hardware-config/fws_cim_llama7b.yaml \
  --model_config configs/model-config/llama2_7b_fws_inf.yaml \
  --tp 2 --shared_chiplets 2 --out docs/qif/atlas/p3_llama7b_tp2.json
```

## Future seams

- **MLA.** The MLA inference path discards per-op GEMM times upstream, so
  it needs its own pricing seam; `attention_type: mla` is rejected.
- **DSE extensions.** `tools/fws_cim_dse.py` sweeps mux/tp/expert
  spreading today. Analog clock optimization, partial banking, fabric
  resizing, and multi-tenant pools are later phases.
- **Energy completeness.** Fabric, SRAM, and helper energy need a component
  library story before the PARTIAL label can go.
