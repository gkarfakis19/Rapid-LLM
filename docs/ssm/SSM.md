# Shared SSM schema and GPU inference

The sole SSM/hybrid schema is `model_param.layer_plan` plus typed block
parameters. `model_type` selects FFN arithmetic, not the mixer schedule.
`run_type: inference` in the model YAML is the rule for both GPU and CIM;
neither path silently converts training to inference. GPU SSM training is
out of scope. Existing CIM laws and pricing are retained.

## Migration from archive/ssm-gpu-old (fdbb0ea)

Run `./.venv/bin/python tools/convert_ssm_config.py old.yaml new.yaml`.
Runtime parsing refuses retired keys by name. It does not alias them.

| Archive key | Shared key | Meaning |
|---|---|---|
| state_size | ssm.d_state | recurrent state width per channel |
| conv_kernel | ssm.d_conv | depthwise causal filter length |
| num_heads | ssm.n_heads | mixer heads, not attention heads |
| headdim or head_dim | ssm.d_head | mixer channels per head |
| expand | ssm.expand | mixer width = expand * hidden_dim |
| n_groups | ssm.n_groups | B/C and RMS groups |
| linear.linear_type: mamba2 | ssm.variant: mamba2 | head-structured SSD |
| hybrid_schedule | layer_plan.layer_types | explicit per-layer kinds |
| chunk_size | ssm.chunk_size | SSD prefill chunk length |

The first six dimensions may formerly appear at model level, in `ssm`, or
in `linear`. The converter rejects conflicting duplicates. The archive has
no chunk-size timing input; migration declares 256 explicitly. This is a
NEW prefill assumption, not a recovered old parameter.
Explicit old schedules are 1-based; the converter writes a complete ordered
list. Interval 0 means all SSM. The archive's `int(interval or 1)` bug is not
preserved. Mamba2 and Granite/Jamba use the llama/SwiGLU FFN carrier: the
archive treated every non-GPT model as SwiGLU. Mixer-only models set `ffn_per_layer: false`;
their required positive intermediate_size is unused.
The archive's pure-Mamba `decode_len = seq_len` default is retained when the
old YAML omitted decode_len. The HF interval-0 golden still matches exactly:
despite its broken schedule property, the old runtime selected its primary
SSM block type, as recorded in that reference's runtime snapshot.

Conv history contains `(d_conv - 1)` cached inputs per channel, including
the Mamba2 X/B/C channels. The archive memory helper used `d_conv`, one
extra slot. CIM already uses `d_conv - 1`; its values remain unchanged.

The archive only records RMS subgroup **counts**, not rank IDs. For TP=T,
groups=G, let g=gcd(G,T), local groups=G/g, subgroup size s=T/g.
The shared GPU convention assigns contiguous TP coordinates
`[floor(t/s)*s, ..., floor(t/s)*s+s-1]`, fixing PP/EP/CP coordinates.
Statistics are all-reduced only when s>1, with B*tokens*(G/g)*stats_bytes
per rank. This preserves the archive's counts and gcd law. Rank membership
is now explicit in the program IR. NEEDS GEORGE: approve this explicit
membership convention; the archive did not establish rank IDs.

## Implementation and evidence

`ssm_gpu.py` ports the archive's Mamba2 inference arithmetic, fusion choices,
kernel overheads and grouping. Projections use the existing GPU GEMM sharding
and kernel backend. This does not select an alternate GEMM model for SSM.
`inference_timing.py:_layer_plan_timings` produces each reachable template;
`train_timing.py:_prepare_execution_graphs` supplies them to `WorkloadSpec`.
`ModelShape.block_kinds` selects the template per layer, extending the existing
dense/MoE selection. `program.build.build` remains the only GPU Program builder.
`llm_execution.py` measures each kind with BLOCK Programs and retimes its
PIPELINE layers. FLAT expansion and memory use the same per-layer templates.
There is no old graph builder or parallel SSM execution graph.

For local mixer width W=d_inner/TP, local groups G'=G/gcd(G,TP), local heads
H'=n_heads/TP, state width N=d_state and K=d_conv, the decode laws are:

| Kernel | Work / bytes before roofline |
|---|---|
| in_proj | column-sharded GEMM, hidden -> 2*d_inner + 2*G*N + n_heads |
| depthwise conv | 2*K*B*local_conv_width ops; B*local_conv_width*(K+1)*activation_bytes |
| scan | G'*(7*(W/G')*N + (H'/G')*N - W/G')*B ops |
| local RMS statistics | 2*W*B ops; B*W*stats_bytes (included in scan roofline) |
| recurrent state traffic | 2*B*W*N*activation_bytes per decode token |
| gate | B*W ops; 3*B*W*activation_bytes |
| out_proj | row-sharded GEMM, d_inner -> hidden |

Conv width is `W + 2*ceil(G*N/TP)` (X/B/C). Projection/conv and gate/out
remain grouped as in the archive; FFN1, activation and FFN2 remain separate.
Changing those operation boundaries changes AstraSim integer-microsecond
rounding even when analytical sums agree. LN/residual/FFN use existing helpers.
TP output projection and FFN2 use all-reduce, or reduce-scatter with sequence
parallelism; LN uses the existing all-gather rule. RMS uses its declared TP
subgroup and is not widened to all TP ranks by the Program builder.

`ssm_gpu.build_memory_data` provides per-layer weights, recurrent and conv
state, and transient activation maps. `program.memory_sim` sums the resident
weights/cache of the exact layers each device owns under PP and uses that
layer's activation map. The data-parallel inference setting remains a replica
multiplier; it does not shard a replica's state. Attention KV memory and traffic
are charged only to attention layers. The standalone memory estimator and
the inference memory pass use the same templates and agree exactly in tests.

Prefill is **NEW, UNVALIDATED against old goldens**. The GPU scan roofline
reuses `cim_timing.ssd_chunked_scan_work` with the declared chunk size. Each
sequence is chunked independently (not B*sequence tokens concatenated into
one sequence), including a fully charged tail chunk, dense intra-chunk scores
and mask, chunk-state construction, state passing and inter-chunk output.
State bytes are read/written once per chunk. Chunk size 1 and single tokens
use recurrence. Conv/projection/gate/RMS/FFN costs scale with tokens.
The CLI labels the new prefill status. This is an analytical inference law,
not a GPU kernel implementation or a hardware benchmark.
The phase is explicit: a one-token prefill uses the one-token prefill
attention workload, never decode attention at the final context length.

### Shared model workloads

The four archive YAMLs were converted into `configs/model-config/` under their
original filenames. Their checked-in workloads explicitly use the converter's
`--shared-inference-workload` adaptation: batch 1, endpoints dropped,
FlashAttention disabled, and one token reserved for prefill when the archive
example was decode-only. Dimensions, layer counts, mixer order, experts and
SSM groups are preserved. This is required by CIM D29/D30 and the existing
prefill-wavefront rule. A GPU/CIM run must also choose hardware with adequate
capacity and a supported mapping; a 32B/52B model does not fit every CIM card.
Conversion **without** that option preserves the original workload, as the
23-case reproduction does. No hardware-specific SSM schema is introduced.

`mamba2_2p7b_decode_inf.yaml` and `granite_4_0_h_1b_decode_inf.yaml` were each
loaded unchanged for GPU decode and mapped CIM evaluation (full-size slow
probes, 2 passed in 39.52 s). Fast probes run a single shared YAML through GPU
prefill/decode and the CIM mapper/Program/evaluator for pure and mixed stacks.
CIM's authoritative evaluator is `fws_mapping -> program.fws_build -> fws_eval`.
No CIM law or saved QIF numerical artifact was edited.

### Hybrid composition

`tests/test_ssm_program.py` checks:

- All attention: the explicit plan and the existing transformer path have
  exactly equal templates, duration tables, FLAT op records and total time.
- All SSM: explicit and repeated-pattern plans have exactly equal templates,
  durations, FLAT op records and total time, using the same pure SSM path.
- Mixed SSM/attention at TP/PP 1/1, 2/1 and 2/2: every layer's template,
  collective spec and memory match its single-kind counterpart. The serial
  PP=1 schedule total is checked in layer order, without a floating-point
  tolerance; PP=2 also exercises placement and per-device memory replay.
- MoE: the existing expert/router/dispatch/combine operations are reused on
  SSM layers, with their own SSM mixer retained. The resulting FLAT Program
  runs through timing and memory replay.
- Granite-4.0-H-Small: 36 SSM and 4 attention layers; Jamba2-Mini: 28 SSM and
  4 attention layers. Per-layer emitted template kinds follow their migrated
  schedules; these are composition checks, not comparisons with the invalid
  archive hybrid fallback timings.
- RMS subgroup members and bytes, FFN output collectives, per-device resident
  state for TP=2/PP=2, per-sequence SSD prefill work and memory-only agreement.

Qwen3.5 has a `linear_attn` plan, not a Mamba2 plan. GPU execution is explicitly
refused because the archive supplies no gated-DeltaNet GPU law. Its existing
CIM path and evidence remain intact; it is not reported as a validated GPU
hybrid. See the gaps below.

### Archive reproduction

Run from this worktree with the documented LD_LIBRARY_PATH:

```sh
export LD_LIBRARY_PATH=/u1/ee/karfakis/gcc-10.2.0/lib64:/app/nanocad/projects/personal/gkarfakis/anaconda3/lib
./.venv/bin/python tools/compare_ssm_goldens.py
```

The script checks the recorded input YAML hashes, reads only completed pure
Mamba2 references, disables the AstraSim result cache, and writes logs and
full-precision comparison JSON under `tmp/ssm-comparison/`. It compares every
op's compute/communication/FLOPs/memory fields, standalone kernel times,
network calls and IR collective bytes/participants, recurrent state and
weights. Conv history is checked against the explicit K-1 convention.
Per-device static and peak bytes are captured from the actual memory replay.
The 19 completed archive hybrid fallback cases are never treated as SSM
references. A missing reference or unexpected numeric movement fails.

One difference is pinned in `tests/ssm_gpu_pins.json`, without a tolerance.
The wide-batch A100 RMS case uses main's `extended_timing` GEMM backend; the
archive predates it. Four op durations round up by 3+1+3+3=10 us per layer,
adding 24*32*10 us = 0.00768 s. Disabling only the extended backend restores
the old kernel times **and the exact old total** in the reproduction's causal
ablation. Batch-4 skinny kernels use the backend's native fallback.

The checked-in `archive_comparison.json` and table below retain the full
precision values; MATCH means exact Python-float equality, not rounded display.
The receipt includes the reference/config hashes and the executed code-file
hashes. Runtime scratch and Matplotlib cache are forced under the selected
output directory, which must be inside this worktree's `tmp/`.

| Case | Archive seconds | Program seconds | Result |
|---|---:|---:|---|
| mamba_a100_tp1_pp1_dp1_ep1_analytical | 0.126775061858006 | 0.126775061858006 | MATCH |
| mamba_a100_tp1_pp1_dp1_ep1_hierarchical | 0.125952064 | 0.125952064 | MATCH |
| mamba_a100_tp1_pp2_dp1_ep1_analytical | 0.1269442171314435 | 0.1269442171314435 | MATCH |
| mamba_a100_tp1_pp2_dp1_ep1_hierarchical | 0.126121216 | 0.126121216 | MATCH |
| mamba_a100_tp2_pp1_dp1_ep2_analytical | 0.17970288730431272 | 0.17970288730431272 | MATCH |
| mamba_a100_tp2_pp1_dp1_ep2_hierarchical | 0.137472064 | 0.137472064 | MATCH |
| mamba_a100_tp2_pp2_dp1_ep2_analytical | 0.17987204257775025 | 0.17987204257775025 | MATCH |
| mamba_a100_tp2_pp2_dp1_ep2_hierarchical | 0.137641216 | 0.137641216 | MATCH |
| mamba_a100_tp2_pp2_dp2_ep2_hierarchical | 0.137641216 | 0.137641216 | MATCH |
| mamba_a100_tp4_pp1_dp1_ep4_analytical | 0.23638247190246603 | 0.23638247190246603 | MATCH |
| mamba_a100_tp4_pp1_dp1_ep4_analytical_rms | 0.266080361385973 | 0.266080361385973 | MATCH |
| mamba_a100_tp4_pp1_dp1_ep4_analytical_rms_subgroup | 0.2507566301451211 | 0.2507566301451211 | MATCH |
| mamba_a100_tp4_pp1_dp1_ep4_hierarchical | 0.165888064 | 0.165888064 | MATCH |
| mamba_a100_tp4_pp2_dp1_ep4_analytical | 0.23655162717590356 | 0.23655162717590356 | MATCH |
| mamba_a100_tp4_pp2_dp1_ep4_hierarchical | 0.166057216 | 0.166057216 | MATCH |
| mamba_a100_tp4_pp2_dp1_ep4_hierarchical_rms_wide_fullgroup | 0.205509024 | 0.213189024 | EXPLAINED |
| mamba_h100_tp1_pp1_dp1_ep1_hierarchical | 0.122112064 | 0.122112064 | MATCH |
| mamba_h100_tp1_pp2_dp1_ep1_hierarchical | 0.12227936 | 0.12227936 | MATCH |
| mamba_h100_tp2_pp1_dp1_ep2_hierarchical | 0.13670406400000001 | 0.13670406400000001 | MATCH |
| mamba_h100_tp2_pp2_dp1_ep2_hierarchical | 0.13687136 | 0.13687136 | MATCH |
| mamba_h100_tp4_pp1_dp1_ep4_hierarchical | 0.133632064 | 0.133632064 | MATCH |
| mamba_h100_tp4_pp2_dp1_ep4_hierarchical | 0.13379936 | 0.13379936 | MATCH |
| mamba_hf_a100_tp1_pp1_dp1_ep1_hierarchical | 0.264192064 | 0.264192064 | MATCH |

### Gates

| Gate | Result | Command wall seconds |
|---|---|---:|
| Existing rewrite golden gate | 221 passed; 46 tracked golden files byte-identical to 1257cbe2 | 41.21 |
| Full default test tree, including webui suites | 2737 passed, 295 skipped, 8 xfailed; zero failures | 685.97 |
| CIM vs OPTIMA validator | 544/544; exit 0 | 300.62 |
| Archive Mamba2 comparison | 23/23 accepted: 22 exact, 1 explained and exactly pinned | 11.05 |

The full default baseline was 2701 passed / 292 skipped / 8 xfailed. This
change adds 36 fast cases and 3 slow cases; the fast SSM suites report 36
passed / 3 skipped in 1.82 s. The 23-case matrix and two full-size GPU/CIM
probes are behind the existing slow marker. Full-size dual-device probes
were also run separately: 2 passed in 39.52 s.

All gates used the repo interpreter and required LD_LIBRARY_PATH, CPU only,
with scratch under this worktree's tmp/. Pytest runs used separate temporary
roots; a preliminary parallel golden attempt that failed before producing
one run artifact was superseded by the completed passing run above.
`gates.json` records the exact commands, counts, wall times and byte hashes.
No saved golden or numerical CIM/QIF artifact was regenerated.


### Known gaps and owner decisions

- GPU laws currently cover sequential attention/Mamba2 plans and their dense
  or MoE FFNs. Mamba1, linear_attn (including Qwen3.5), short_conv, standalone
  FFN/MoE mixer kinds and parallel_branch plans are explicitly refused. GPU
  training remains out of scope. CP inference keeps the existing refusal.
- SSM-only MoE plans, per-kind FFN width overrides, depth-shared weights and
  hybrid fault retiming remain refused; no new result silently ignores them.
- GPU SSM energy is UNMODELED. Reported zeros are explicitly labelled
  placeholders; they are not an energy prediction. Prefill lacks an old
  reference and lacks hardware validation.
- The archive's tiny hierarchical RMS-stat payload abort and its subgroup
  mapping refusal are not successful references. The new IR expresses subgroup
  membership, but those previously refused/failed configurations are not part
  of the 23-case acceptance claim. No claim of a repaired tiny-payload AstraSim
  transport is made.
- NEEDS GEORGE: approve the explicit contiguous RMS TP membership described
  above; only subgroup counts were observable in the archive.
- NEEDS GEORGE: CIM's frozen SSM history census uses d_inner channels, while
  the GPU port preserves the archive's X/B/C channel census. Both now use K-1
  history depth; including B/C in CIM would change protected numerical results
  and is intentionally deferred.
- NEEDS GEORGE: the archive's optional SSM FFN prices its second GEMM with the
  doubled SwiGLU projection width, and its weight census includes that width
  plus replicated grouped B/C columns when TP splits a group. Those archive
  approximations are retained for exact decode compatibility. Ordinary
  transformer FFNs retain main's existing laws. Real mixer-only Mamba2 stacks
  (`ffn_per_layer: false`) do not invoke that optional FFN.
  The residency census's grouped B/C columns can exceed the timing GEMM's
  column shard; this is inherited conservative residency accounting, not a
  new physical mapping claim.
- NEEDS GEORGE: extending GPU pricing to Mamba1/DeltaNet/short-conv/parallel
  branches requires additional GPU assumptions and validation. This port
  conservatively refuses them rather than presenting CIM scalar work counts
  as established GPU timing laws.
