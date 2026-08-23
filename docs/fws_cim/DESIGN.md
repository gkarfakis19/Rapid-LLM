# FWS-CIM device class — pass-1 design contract

Status: locked for implementation (2026-08-22). Scope decisions came from the user;
laws and seams come from a 5-agent recon over this repo, `origin/main`,
`origin/heterogeneous`, and the OPTIMA reference model at
`/app/nanocad/projects/cim_ctt_big_optima/perf_model` (read-only reference — do NOT
import code from it; rederive).

## 1. Scope (fixed by user decisions — do not re-litigate)

- New `device_class: fws_cim` in Rapid-LLM, clean-slate and device-generic (not
  CTT-specific). OPTIMA is the numeric validation reference only.
- **ViT-class inference only** in pass 1. `fws_cim` + non-ViT model, or
  `run_type: training`, or `decode_len > 0`, or `use_flashattention: true`, or
  `execution_backend.model: astra` → clean config-validation error with an
  actionable message. (Weights admit no writes: training on FWS CIM is
  permanently out, not deferred.)
- Analytical backend only; chip-to-chip transfers are simple p2p over the
  existing network `dimensions` (ring dims keep the analytical validator happy).
- Mapping: `layers_per_chip` given in config; capacity validated. Real DSE later.
- MoE/KV-cache/decode: out of pass 1; leave the seams undisturbed
  (`inference.kvcache_type` stays untouched).

## 2. Device model (single source of truth: new module `cim_timing.py`)

One class, `CimDeviceModel`, holds every law below. Consumers: (a) the
per-op GEMM branch in `base_timing.get_gemm_time`, (b) the FWS spatial report in
`inference_timing.calc_time`. No law may be duplicated at a consumer.

### 2.1 Analog weight-GEMM law

- `vec_cycles = adc_mux * slice_cycles`; `vec_latency = vec_cycles / f_analog`.
- Weight-GEMM time is **independent of K and N** (all mapped arrays fire in
  parallel): `T = M_tokens * vec_latency`. M is the token count of the call as
  received (works for tp-sharded dims too; sharding K/N never changes T).
- Array count for a K×N weight matrix:
  `arrays(K, N) = ceil(K / rows) * ceil(N / (cols_adc * adc_mux))`.
  This reproduces OPTIMA per-layer counts: QKV (K=H, N=3H) → 3, O-proj → 1,
  FFN1 (K=H, N=I) → ceil(I/H), FFN2 (K=I, N=H) → ceil(I/H); ×2 for the FFN1 pair
  when the model uses SwiGLU/gated MLP.
- `shots_per_output` affects **energy only, never time**.
- Energy per stage per layer: `E = M_tokens * energy_per_vec_pj * shots_per_output
  * arrays`. Area: `arrays_total * area_mm2_per_array`.

### 2.2 Digital fabric (attention sidecar)

Two modes, config `cim.fabric.model`:

- `sa` (default): closed-form systolic model, verified bit-exact against
  OPTIMA's recorded ScaleSim outputs (os dataflow, CALC bandwidth,
  generous SRAM):
  `cycles(M, N, K; R, C) = ceil(M/R) * ceil(N/C) * (K + R + C - 2) - 1`.
  Attention uses OPTIMA's **folded-K** convention (heads folded into K to model
  dual-buffered fully-pipelined arrays; do NOT price per-head fill/drain):
  - per chip, `heads_chip = ceil(kv_heads / tp)` when sharded, else `kv_heads`;
    `heads_per_replica = ceil(heads_chip / replicas)`.
  - QK^T: M=S, N=S, K = head_dim * heads_per_replica.
  - PV:   M=S, N=head_dim, K = S * heads_per_replica.
  - The QK and PV runs occupy the fabric's arrays **concurrently**:
    `total_cycles = max(QK, PV) + fill_drain_penalty` with
    `fill_drain_penalty = 3 * R` (hand-picked surrogate; keep as config with
    default 3*R). `T = total_cycles / f_fabric`.
  - Softmax lanes: `cycles = pipeline_depth + ceil(S * heads_chip / lanes) - 1`
    with defaults pipeline_depth=20, lanes = softmax_lanes * replicas.
    Stage-2 time in the report: `T_S2 = max(T_sa, T_softmax)`.
- `gpu_native`: do **not** intercept the attention act×act GEMMs; they fall
  through to the native tile/roofline machinery priced against the stub GPU
  tech_param (§3.3), which then must describe the sidecar. This is the
  "run existing RAPID infra a new way" mode; implement as: classification says
  ACT_GEMM → return None from the CIM branch → native path continues.
  CAUTION (recon fact): with `gpu_native`, `extended_timing` must still be
  skipped (§4.1) or it will silently reprice big act-GEMMs against A100-latency
  constants.

### 2.3 Digital helper absorption (LN / GELU / adder)

OPTIMA sizes helper lanes so they NEVER bound a stage (it raises if one would).
Pass 1 adopts the same contract as an assumption, not a computation: analog
stage time = the analog law; helpers are absorbed. State this in the report
output and in docs. Native per-op pointwise pricing (roofline at DRAM level
against the stub hierarchy) is left untouched — it only feeds the sequential
(non-FWS) total, not the spatial report.

## 3. Config schema

### 3.1 New top-level keys (hardware YAML)

```yaml
device_class: fws_cim        # default when absent: gpu (existing behavior, bit-identical)
cim:
  analog:
    rows: 1280               # array rows (K per array tile)
    cols_adc: 320            # ADC columns per array
    adc_mux: 4               # weight-column banks per ADC; stored cols = cols_adc*adc_mux
    slice_cycles: 2          # analog cycles per ADC evaluation
    analog_clock_mhz: 100
    energy_per_vec_pj: 9534.9389   # per array per input vector (already includes mux)
    shots_per_output: 2      # energy only
    area_mm2_per_array: 1.403065   # optional, 0 disables area reporting
  fabric:
    model: sa                # sa | gpu_native
    rows: 32
    cols: 64
    num_arrays: 2            # QK and PV run concurrently on these
    replicas: 1              # attention_array_replicas
    clock_ghz: 0.95
    fill_drain_penalty_cycles: 96   # default 3*rows if absent
    softmax_lanes: 1
    softmax_pipeline_depth: 20
  chip:
    arrays_per_chip: 400     # capacity; 0 = unchecked
    layers_per_chip: 32      # int (uniform) or explicit list [16, 16]
```

Parsing: a `CIMConfig` dataclass beside `InferenceHWConfig`
(config.py:2142–2151 is the pattern), fields on `HWConfig` (config.py:2156–2165),
wired in `HWConfig.from_dict` (config.py:2168–2231). Unknown top-level YAML keys
are silently ignored today, so this is inert for all existing configs.
TRAP (recon): the global `convert()` preprocessor (config.py:2287) rewrites any
`"<int> <Unit>"` string anywhere in the YAML into byte counts — use plain
numbers inside `cim:`, never unit strings.

Validation (`validate_hw_config`, config.py:2383–2396): `device_class: fws_cim`
requires the `cim:` block (and vice versa warn), plus the §1 rejection list.
Model-side rejections that need the model config (ViT-only check) go where
ViT constraints already live (config.py:1812–1819 pattern) or in run_perf
after both configs parse — implementer's choice, but the error must name the
offending setting.

### 3.2 Model config addition

Optional `model_param.vision.num_prefix_tokens` override (int ≥ 0). Default:
current behavior (`_vit_default_num_prefix_tokens`: vit→1, vit_dinov3→5,
config.py:47–48, consumed at 1648–1649, 1727, 1736). Needed so validation
configs can hit OPTIMA's exact seq (64 = 128px/16 patches + 0 prefix;
196 = 224px/16 + 0 prefix).

### 3.3 Stub tech_param (still required)

Keep `tech_param` + `memory_hierarchy` structurally intact in fws_cim YAMLs
(relaxing `TechConfig.from_dict` config.py:290–296 and `MemoryHierarchy`
hw_component.py:284–330 touches far more code than a stub YAML). Calibrate the
stub to the FABRIC, not to a GPU: core peak flops ≈ num_arrays·R·C·2·clock;
DRAM level = the chip's activation SRAM (size/bandwidth); sane L2/L1/L0.
It prices: vector ops (LN/softmax/GELU/residual — all roofline'd at
`mem_level=num_levels-1`), analytical collectives (which roofline at level 0 —
recon: level-0 bandwidth is load-bearing for NETWORK time; keep it high), and
everything under `fabric.model: gpu_native`. Set `sw_param.kernel_launch_overhead: 0`
(deterministic fabric; avoids self.O double-count at get_gemm_time:780).
Keep `core.util: 1`.

## 4. Integration points (all verified file:line)

### 4.1 `base_timing.TimeCalculation.__init__` (389–560)

- Guard `extended_timing.create` (429–430): skip for fws_cim, set
  `self._extended_gemm_backend = None` (getattr guard at 773 makes this safe).
  It TypeErrors on missing GPU fields anyway, and must never price CIM runs.
- Build `self.cim_model = CimDeviceModel(hw_config, model...)` for fws_cim; None
  otherwise. Keep Core/MemoryHierarchy/DRAM built from the stub (self.th,
  self.mem_layer 4 levels, self.num_levels, self.DRAM must stay well-formed).
  Do NOT return early: lines 412–417 mutate hw_config (active_parallelism,
  network_layout) that downstream consumers need.

### 4.2 `base_timing.get_gemm_time` (680)

Early branch BEFORE the tile enumeration at 691 (mandatory: the tile loop, not
the backend seam, is what requires the 4-level GPU machinery):

```
if self.cim_model is not None:
    t = self.cim_model.price_gemm(name, dim1, dim2, dim3, self)  # None → fall through (gpu_native attention)
    if t is not None:
        return t (+ self.O unless disable_overhead), inner_code 0,
               stub tile dims ((1,1,1),)*3, mem_access (0,0,0,0)
```

Role classification by op-name (names verified from the ViT trace):
- ACT_GEMM: `attention_score_*`, `attention_output_*` → SA law (or None in
  gpu_native mode).
- WEIGHT_GEMM: `vit_patch_embed_*`, `qkv_projection_*`, `output_projection_*`,
  `ffn_*`, `ffn2_*`, `vit_head_*`, `embedding*`, `linear*` → analog law.
- Unknown names: analog law + one-time warning log.
- Backward twins (`*_b`) ARE priced during inference and then discarded
  (train_timing.py:3985–4450, include_transformer_backward=False); price them
  with the same law on whatever dims arrive; never crash, never log per-call.

Attention outer-multiplier compensation (critical, from the trace): callers
price ONE head-shape then multiply outside by `N_mult`:
TENSOR_SEQUENCE (tp≥2): `N_mult = B*kv_heads/tp` (train_timing.py:1917–1925);
SINGLE (tp=1): `N_mult = B*kv_heads` (1533–1535). The SA law computes the
folded per-chip TOTAL `T_chip`; `price_gemm` must return `T_chip / N_mult` so
the caller's multiplication restores `T_chip`. Derive `N_mult` from
`self.batch_size`, model kv_heads, `self.tp`, and the active parallelism mode
exactly as those lines do; unit-test at tp=1 and tp=2.

Flash-attention path (flashattn_enable=True) never reaches the branch — but
fws_cim configs reject use_flashattention anyway.

### 4.3 FWS spatial report — `inference_timing.calc_time`

Attach post-hoc, right after `compute_all_gemm_and_node_times` returns
(inference_timing.py:743–767), guarded on device_class. Compute from
`CimDeviceModel` stage laws directly (never reconstruct from op sums):

- Per-layer stages: S1 QKV, S2 attention (max(SA, softmax)), S3 O-proj,
  S4 FFN1, S5 FFN2 — S1/S3/S4/S5 = analog law at the run's seq_len.
- Endpoint stages, listed separately: patch-embed (analog, M=seq) on chip 0,
  vit head (analog, M=B) on the last chip.
- `pipeline_period = max(all stage times)`; `fps = 1/period`;
  `block_latency = sum(S1..S5)`; `end_to_end_latency = num_layers*block_latency
  + endpoint stages + boundary transfers`.
- Placement: `layers_per_chip` → chip index per layer; boundary bytes
  `= B*seq*hidden*act_bytes`; boundary time via
  `self.network_model._analytical_point_to_point(bytes, *self.links["pp"])`
  (base_timing.py:381–386, 443–449). Do NOT also add the model's own
  cross_layer term (recon risk: it is 0 at pp==1; assert pp==1 for fws_cim).
- Warn (not fail) if any boundary transfer time > period (bandwidth-bound).
- Capacity: arrays needed per chip (its layers × per-layer counts + endpoint
  arrays) vs `arrays_per_chip` → hard error on overflow when arrays_per_chip>0.
- Area: total arrays × area_mm2_per_array (skip if 0). Energy (labeled
  PARTIAL — analog + interconnect only): §2.1 energy summed over stages/layers
  + boundary bytes × dim energy_per_bit. No fabric/SRAM energy in pass 1.
- Output: a readable `FWS-CIM spatial pipeline` section appended to
  `output/VIT/LLM_inference_results.txt` AND a machine-readable
  `output/VIT/fws_cim_report.json` (period_us, fps, block_latency_us,
  end_to_end_latency_us, per-stage table, qk/sv cycles, arrays_per_layer,
  arrays_total, area_mm2, per-chip occupancy, boundary times, partial energy).
- The existing sequential total (sum of ops) remains untouched and is labeled
  as sequential-execution latency in the new section for contrast.

### 4.4 Memory estimation

For fws_cim, weights live in arrays, not DRAM: exclude weight bytes from the
inference memory tables. Seam: `MemoryEstimator.build_memory_data` is the single
producer (memory_estimation.py:81+; instantiated at inference_timing.py:732 and
llm_util.py:891). Smallest honest change wins (flag on the estimator or a
device_class check inside build_memory_data). The DRAM stub `.size` then
represents the activation buffer, so the existing capacity report becomes an
activation-feasibility check — say so in the report label.

### 4.5 Known landmines (from recon — honor every one)

- `roofline` failure calls `sys.exit(0)` (base_timing.py:646–657): never feed it
  a zero last-level multi-level access list from CIM code paths (our branch
  bypasses multi-level rooflines for GEMMs; vector ops keep nonzero bytes).
- Two DRAM objects exist (self.DRAM at 438–440 vs mem_layer[3]); keep the stub
  tech_param.DRAM and memory_hierarchy l3 consistent.
- Analytical mode hard-requires ring topologies (config.py:2383–2396) — CIM
  YAML network dims use type Ring.
- `validate_hw_config` must cross-check device_class vs cim-block presence
  (unknown-key silence otherwise turns a typo into a silent GPU run).
- parallelism.train / parallelism.inference / network.overlap blocks are
  REQUIRED by the parser even for a single-chip CIM config — keep them in the
  YAML template.
- ViT head collectives are baked into linear_softmax compute (train_timing.py:
  2597–2607) — fine for pass 1 (tp=1 validation), note in docs.
- `num_classes := vocab_size` for ViT (config.py:1631–1633): the 768×1000 head
  GEMM is real; it maps to arrays (counted as endpoint arrays).

## 5. Validation targets (must match before done)

Reference outputs recorded from OPTIMA (`output/ctt_800mm2_vit/summary.tsv`,
`output/ctt_800mm2_vit_seq196_mux16/baseline_2d_huge_seq196_mux16.json`).
Tolerances: cycle counts and array counts EXACT; times/areas/fps ≤ 0.1 %
relative. All at tp=1, B=1, single chip (layers_per_chip = num_layers),
analog 100 MHz, slice_cycles 2, replicas 1, post-LN semantics (absorbed).

| | T1 (A_2D) | T2 (B_3D) | T3 (seq196) |
|---|---|---|---|
| model | H1280 I5120 L32 h16 | H1408 I6144 L40 h16 | H1280 I5120 L32 h16 |
| seq (prefix 0) | 64 | 64 | 196 |
| adc_mux (rows, cols_adc) | 4 (1280,320) | 2 (1408,704) | 16 (1280,80) |
| E_vec pJ / area mm² | 9534.9389 / 1.403065 | 10138.5868 / 2.014071 | 12470.0063 / 1.135296 |
| fabric clock GHz | 0.95 | 1.0 | 0.95 |
| QK / PV cycles | 2747 / 4471 | 3003 / 4471 | 38471 / 45219 |
| period µs (bottleneck) | 5.12 (S1) | 4.567 (S2) | 62.72 (S1) |
| block latency µs | 30.407368 | 17.367 | 361.3 |
| fps | 195312.5 | 218962.12 | 15943.8776 |
| arrays/layer | 12 | 14 | 12 |
| CTT area mm² | 538.7768 | 1127.8796 | 435.9537 |

Closed-form cross-checks: QK T1 = 2·(1280+94)−1; PV T1 = 2·2·(1024+94)−1;
QK T3 = 7·4·(1280+94)−1; PV T3 = 7·2·(3136+94)−1. S2 time = (max+96)/clock.

Regression gates (bit-identical, no tolerance):
- `a100_80GB.yaml` + `vit_base_inf.yaml` → prefill matches saved baseline
  (0.003 s; scratchpad copy `baseline_a100_vit_results.txt`).
- `a100_80GB.yaml` + `Llama2-7B_inf.yaml` → matches
  `baseline_a100_llama2_7b_results.txt` (prefill 0.592 s, decode 15.869 s).
- Existing test suite status no worse than before the change.

## 6. File plan

- `cim_timing.py` — CimDeviceModel (laws §2), price_gemm, stage/report math.
- `config.py` — CIMConfig, device_class, validation, vision.num_prefix_tokens.
- `base_timing.py` — §4.1 init branch, §4.2 get_gemm_time branch.
- `inference_timing.py` — §4.3 report, §4.4 memory hook.
- `configs/hardware-config/fws_cim_optima_t1.yaml` (+ t2/t3 variants).
- `configs/model-config/vit_huge_story_64_inf.yaml`, `vit_g_64_inf.yaml`,
  `vit_huge_story_196_inf.yaml` (targets' models; prefix 0).
- `validation_scripts/validate_fws_cim_vs_optima.py` — runs T1–T3 + the two
  regression gates, prints a pass/fail table, exit code reflects result; also
  prints an A100-vs-FWS-CIM comparison table for the same ViT workload (free
  cross-architecture demo).
- `tests/test_fws_cim.py` — law unit tests (recorded cycles/counts), config
  parse/reject tests, N_mult compensation at tp=1/tp=2, gpu-default inertness.
- `docs/fws_cim/README.md` — user-facing: model description, assumptions
  (helper absorption, partial energy, folded-K), usage, future seams (KV
  stories, MoE experts-per-chip, DSE).

## 7. Explicitly rejected alternatives (do not resurrect in pass 1)

- Chips-as-pp-stages: pp forces micro-batch reshaping of GEMM dims
  (train_timing.py:4547–4552) and homogeneous per-class templates — wrong tool;
  post-hoc report instead. pp must be 1 for fws_cim (assert).
- Porting OPTIMA's calc_analog physics or component library: CTT-specific;
  users supply E_vec/area per array directly.
- optimize_analog_clock, multi-tenant pools, partial banking, Pareto DSE:
  later phases.
- Energy beyond analog+interconnect: later (needs a component library story).
