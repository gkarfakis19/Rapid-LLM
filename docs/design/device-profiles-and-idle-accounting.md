# Design: Per-Device Throttle Profiles + Idle Accounting

Status: approved for implementation on branch `heterogeneous` (2026-07-24).
Grounding: six deep-read reports under
`/tmp/claude-7923/-app-nanocad-projects-george-thermal-26/4a4aa370-aa51-4a9d-b94d-129f1d91fc75/scratchpad/wf1-reports/`
(file names are shuffled vs content; each file self-identifies in its first line).
Prior art: `gkarfakis19/DeepFlow` branch `heterogeneous` commits `eca30b9`/`1b9e274` (Dec 2025 scalar derate — superseded by this design).

## Goals

1. **Per-device hardware throttle profiles**: each physical device (GPU) in a multi-device
   run can have its own effective frequency / HBM bandwidth / HBM latency / cache
   bandwidths. No scalar duration multipliers — per-op durations are **re-priced through the
   full roofline/tiling model** under each profile's parameters (measured: flat factors are
   wrong by up to ~30% between BW-bound and compute-bound ops at 0.7×freq/0.8×BW).
2. **Idle accounting**, two levels:
   - **Kernel-level (roofline-stall) idle** — the vendored thermal_stco metric, restored
     byte-format-compatible and extended to all GEMM paths.
   - **Schedule-level per-device idle** — NEW: per-rank busy vs makespan from the flattened
     AstraSim run; this is what a waferscale thermal loop consumes per device.
3. Mainline quality: config validation, mode gating with clear errors, human-readable
   summary logging, docs, sample configs, tests, and zero behavior change when the feature
   is off.

## Non-goals (v1, documented)

- Per-device **network/link** throttling (the `faulty_links` machinery already derates links).
- MoE + profiles (flattened mode rejects MoE: `llm_execution.py:1630`), ViT + profiles.
- Analytical/hybrid/hierarchical per-device profiles — **hard error**, mirroring the
  faulty-links gate (`llm_execution.py:1291-1292`). Rationale: those modes structurally
  cannot represent per-device compute (see executor-dispatcher report §6).
- SCOTCH `optimize_2dmap` + profiles — hard error v1 (stage→rank permutation happens
  post-build; profile keying would silently target wrong devices).
- Fixing `calc_waves_per_sm` hardcoded A100 constants (`hw_component.py:72-73`) — pre-existing
  model wart, identical across profiles; touching it changes all baseline numbers.

## Feature 1 — Device profiles

### Config surface

New CLI flag on `run_perf.py`: `--device_profiles <yaml>`. Schema (new module
`device_profiles.py`, loader + frozen dataclasses, validation errors as `DeviceProfileError`):

```yaml
# configs/device-profiles/example.yaml
profiles:
  hot:
    frequency_scale: 0.80          # × tech_param.core.operating_frequency
    hbm_bandwidth_scale: 0.90      # × tech_param.DRAM.bandwidth
    hbm_latency_scale: 1.00        # × tech_param.DRAM.latency
    l2_bandwidth_scale: 1.00       # × tech_param.SRAM-L2.bandwidth
    l1_bandwidth_scale: 1.00       # × tech_param.SRAM-L1.bandwidth
    register_bandwidth_scale: 1.00 # × tech_param.SRAM-R.bandwidth
  nominal: {}                      # all scales default 1.0
devices:                           # keyed by flattened hw_id (the dp=0 slice device id)
  0: hot
  1: nominal
  default: nominal                 # optional; else every hw_id must be listed
dp_devices:                        # OPTIONAL, only when dp>1: per-(hw_id, dp_idx) override
  "0,1": hot                       # device hw_id=0 in dp replica 1
```

- All scales must be finite and > 0. `frequency_scale` multiplies `operating_frequency`
  (never `nominal_frequency` — the voltage-derivation path must not activate; see
  config-plumbing report risks).
- Profile resolution order: `dp_devices["h,d"]` → `devices[h]` → `devices.default` → error.
- Validation at load: every device referenced resolves; after graph build: the resolved
  hw_id set must equal the flattened stage set exactly (mirror old patch's check).
- Mode gate: `FULL_ASTRASIM_FLATTENED` only; raise `ValueError` with a message pointing to
  the execution_backend keys otherwise. Gate in `TimeCalculationLLM.__init__`-adjacent
  validation AND in `_run_full_astrasim_flattened` (defense in depth, faulty-links style).
- Log a summary table at dispatch (faulty-links `_log_fault_summary` style): device → profile
  → scales, plus count of distinct profiles.

### Plumbing

`run_perf.py --device_profiles` → `run_LLM(..., device_profiles_path)` →
`TimeCalculationLLM.__init__(..., device_profiles_path=None)`:

- Parse + validate; store the parsed object BOTH as `self.device_profiles` and as an
  attribute on the hw_config object (`hw_config.device_profiles`, following the existing
  `active_parallelism` setattr precedent at `base_timing.py:388-393`). The hw_config
  attachment is what makes **inference decode work by construction**: decode's per-sample
  fresh `TimeCalculationLLMInference` instances receive `self.hw_config` unchanged
  (`simulate_inference_graph.py` `_execute_decode_step`) and pick profiles up automatically —
  closing the exact gap that killed the Dec-2025 inference derate (see inference report §2).
- `TimeCalculationLLMInference` inherits; both prefill and decode dispatchers receive the
  profiles via the time_calc they already hold.
- Dispatcher ctor sites that must pass through: `train_timing.py:5420/5439/5489`,
  `inference_timing.py:836/906`.

### Per-profile re-pricing (the core)

New helper (suggested: `HeterogeneousTimingBank` in `device_profiles.py` or a small new
module), driven from `TimeCalculationLLM` after baseline pricing:

For each **distinct** profile k (dedupe by scale tuple):
1. `hw_k = copy.deepcopy(base_hw_config)` (ctor mutates hw_config in place — deepcopy is
   mandatory; verified zero-drift in the train-timing report §2.3). Multiply the six scaled
   fields. Ensure `operating_frequency` is explicitly set (extended backend crashes on None).
2. `tc_k = TimeCalculationLLM(hw_k, model_config, mode, output_dir=<baseline_output>/profile_<k>/)`
   (own output dir: ctor mkdirs and writes memory-summary files).
3. `timings_k, breakdown_k = tc_k.compute_all_gemm_and_node_times(...)` with the same
   arguments the baseline used. Cost ≈ 2.35 s per profile (measured, thermal-scale config);
   `tile.py` lru_caches are hw-scale-independent and shared, so later profiles are cheaper.
4. Keep `tc_k`'s kernel-idle counters (per-instance) — they are the per-profile kernel idle.

Result: per profile, per op, per direction: re-priced `compute_time` (+ flops, mem accesses),
i.e. `ratio_k[op][dir] = duration_k / duration_baseline` (guard: baseline 0 → ratio 1).

**Known modeling choice (document in code + README):** each profile re-runs tile/kernel
selection under its own parameters (the model's best kernel at that operating point). Real
DVFS does not re-tile mid-run; both are approximations within the model's fidelity, and this
choice keeps the uniform-profile equivalence exact (identical code path to a scaled config).

### Injection into the flattened graph

- In `PipelineGraphFlattener._expand_transformer_node` (and the stage-level emission for
  embedding/linear_softmax/optimizer nodes), stash the op identity on each emitted Node as a
  plain attribute, e.g. `node.op_key = ("qkv_proj", "fwd")` / `("linear_softmax", "fwd")` —
  metadata only, no behavior change.
- **Injection point: inside `_run_full_astrasim_flattened`, AFTER `apply_overlap_transforms`
  (llm_execution.py:1655-1661), BEFORE `run_astra_simulation_only_onepath`** (the overlap
  transform `_split_tp_node` collapses tuple durations — llm_execution.py:104/127 — so tuples
  must be written post-transform).
- Application is **multiplicative with per-op re-priced ratios** (transform-safe: if a
  transform split a node's duration, the ratio still applies exactly):
  - dp == 1: `node.duration = node.duration * ratio_{profile(hw_id)}[op_key]` (scalar).
  - dp > 1: `node.duration = tuple(base * ratio_{profile(hw_id, d)}[op_key] for d in range(dp))`
    — `_compute_duration_seconds(task, dp_idx)` (executor.py:802-813) consumes the tuple with
    **zero executor changes**.
- Nodes without `op_key` (comm placeholders, no-ops, ZeRO-3 gathers with 0 duration) are left
  untouched. Optimizer nodes get the profile of their hw_id with the closest matching op
  ratio class (use `transformer` aggregate ratio; document).
- Validate: resolved profile hw_id set == flattened stage set; grad-accum's second dispatcher
  gets the same treatment automatically (same code path).

### AstraSim runnability (part of this feature)

- Add env override `RAPID_ASTRASIM_BINARY` in `_astrasim_binary_path()`
  (`astrasim_lib/integration.py:273-284`, 3 lines) so a checkout without a built submodule can
  point at an external binary. Chakra protobufs already honor PYTHONPATH
  (`astrasim_lib/bootstrap.py`).
- On this machine: binary =
  `/app/nanocad/projects/ispass_deepflow/deepflow_astra_dev/Rapid-LLM/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware`,
  needs `LD_LIBRARY_PATH=/u1/ee/karfakis/gcc-10.2.0/lib64` (GLIBCXX_3.4.26), chakra dirs on
  PYTHONPATH from the same tree. Known baseline: flattened tp2/cp2 func tests fail with this
  binary (skew vs pinned submodule) — treat as out of scope; use tp1/pp-based flattened cases
  for equivalence tests.

## Feature 2 — Idle accounting

### (a) Kernel-level (restore + extend)

- Re-add to `base_timing.py` (`TimeCalculation`): `_idle_time_sum_s`, `_idle_time_layer_s`,
  `_idle_time_global_s`, `_idle_samples`; `reset_idle_accounting()`,
  `record_idle_from_gemm(observed_time_s, flop, *, scale=1.0, bucket="layer")`
  (`idle = max(0, observed − flop/self.th)`), `get_idle_time_seconds()`,
  `get_idle_breakdown_seconds()`, `get_idle_fraction(total)` — exact vendored semantics
  (isolated patches: `scratchpad/wf1/idle-patch/*.patch`, base = mirror commit 339ef6a).
- Restore the three vendored call sites in `single_gpu_gemm_forward/backward`
  (target `train_timing.py:1415-1444`; bucket = global iff `LINEAR_SOFTMAX`).
- **Extend recording to the paths the vendored patch missed** (required for multi-GPU
  fidelity; each already computes observed time + flops locally):
  `_tensor_parallelism_gemm_*` (1761/1853), `_context_parallelism_gemm_*` (1982/2059),
  `_tensor_context_hybrid_gemm_*` (1446/1577), flash-attention kernels (1161/1230), and the
  MLA helpers (verify they route through instrumented primitives; if not, instrument
  directly). Idle from per-layer ops → `layer` bucket; once-per-step ops → `global`.
  The "ideal" throughput is the instance's own `self.th` — per-profile instances therefore
  measure stall against their **throttled** peak (the thermally meaningful reference).
- Inference: restore prefill snapshot + decode integration exactly as vendored
  (`DecodeSample` idle fields, `_execute_decode_step` reads fresh-instance counters,
  `_integrate_decode_samples` trapezoids them, 6-tuple returns, `calc_total_inference_time`
  dict keys) — **with one fix**: snapshot prefill idle immediately after the prefill
  dispatcher run, before the decode-shaped-graph-for-memory build (vendored contamination
  quirk, inference report §3).
- Training: restore `_run_llm_training` reset → calc → fraction lines.

### (b) Schedule-level per-device (new)

- `integration.py:326-331`: key per-rank wall times by the captured `sys[<rank>]` index
  (stop relying on print order).
- In the ET emission loop (executor.py:1560-1688): accumulate per-rank compute-busy seconds
  from the **pre-rounding float** durations (sidesteps µs quantization). Prune the synthetic
  second rank (mirror `llm_execution.py:1712-1720`).
- `run_astra_simulation_only_onepath`: return per-rank busy alongside per-rank wall times;
  `_run_full_astrasim_flattened` stores `time_calc.flattened_astrasim_per_rank_busy`.
- Per-device schedule idle fraction = `1 − busy_r / makespan`. Comm time on a rank is neither
  busy-compute nor idle in this metric (not decomposable from ET; documented).

### Output contract

Byte-format-compatible with `thermal_analysis_george._parse_rapid_result_files`
(exact strings in the idle-extraction report §4):

- `LLM_training_results.txt`: `Total Time: {:.8f}`, `GPU_time_frac_idle: {:.8f}`,
  `GPU_time_frac_idle_thermal: {:.8f}`, `Idle Time Layer: {:.8f}s`, `Idle Time Global: {:.8f}s`.
  Thermal numerator = `layer_idle*num_layers + global_idle` over total time, computed from the
  **baseline** instance counters (unchanged semantics when profiles are off).
- `LLM_inference_results.txt`: `GPU_time_frac_idle`, `GPU_time_frac_idle_thermal`,
  `Prefill Time`/`Decode Time` at `.8f` + `s`, `Prefill/Decode Idle [Layer|Global] Time: {:.8f}s`.
  (Do NOT emit a `Total Inference Time:` line — the consumer's last-match regex would switch
  its runtime source.)
- NEW `output/<MODE>/device_metrics.json` (always in flattened mode; also when profiles
  active), schema_version 1:

```json
{ "schema_version": 1, "execution_mode": "full_astrasim_flattened",
  "dp_count": 1, "num_devices": 4, "makespan_s": 1.23,
  "profiles": {"hot": {"frequency_scale": 0.8, "...": 1.0}},
  "devices": [
    {"rank": 0, "hw_id": 0, "dp_idx": 0, "profile": "hot",
     "compute_busy_s": 1.10, "wall_time_s": 1.23, "sched_idle_frac": 0.106,
     "layers_hosted": 8, "hosts_lm_head": false,
     "kernel_idle_layer_s": 0.004, "kernel_idle_global_s": 0.0,
     "kernel_idle_frac_thermal": 0.026}] }
```
  Per-device kernel idle = that device's profile-instance counters scaled by
  `layers_hosted(device)` (= layers on its pp stage) + global bucket iff it hosts the
  vocab-projection stage. Also print `Device Metrics JSON: <path>` in the results txt
  (informational; no consumer regex matches it).

## Behavior-preservation invariants (tests enforce)

1. No `--device_profiles` and no idle consumers → **identical timing results** to the branch
   baseline (idle recording is pure observation; assert e.g. koyeb accuracy test values and
   the analytical func-test set are unchanged: 60/60 in ~40 s).
2. Profiles all-1.0 == no profiles: identical total time and per-rank wall times (flattened,
   a tp1 or pp2 case from the passing set).
3. Uniform profile f == globally-scaled hw config: exactly equal totals.
4. Hetero sanity: one slowed device gates the makespan; other devices' `sched_idle_frac`
   rises accordingly; `device_metrics.json` schema validates.
5. Thermal-port parity: analytic run of `a100_80GB_legacy_thermal_port.yaml` +
   `Llama2-7B_train_2048_thermal.yaml` (configs to be added under `configs/` from the vendored
   snapshot) emits all consumer-required lines; `_parse_rapid_result_files` accepts them
   (import the parser in a test or replicate its regexes).
6. Full-suite: no new failures beyond the known 30 flattened tp2/cp2 binary-skew failures.

## AMENDMENTS (v2) — binding resolutions from the 3-critic review

Critique reports: `/tmp/claude-7923/-app-nanocad-projects-george-thermal-26/4a4aa370-aa51-4a9d-b94d-129f1d91fc75/scratchpad/critique/critic{0,1,2}.md`.
All verdicts approve-with-fixes. The following OVERRIDE the corresponding sections above.

**A1. Config surface (overrides "Config surface").** Primary surface = a top-level
`device_profiles:` block **inside the hardware YAML**, parsed natively in `config.py` onto
`HWConfig` (frozen dataclasses, `DeviceProfileError`-style validation in config.py, exactly
like `network.faulty_links`). Same schema content as before (`profiles:`, `devices:`,
`dp_devices:`). Rationale: faulty-links precedent; webui/test-harness reachability
(worker_runner + validation_helpers invoke with hw+model configs only); consumer
hardware_hash captures profiles automatically. Keep `--device_profiles <yaml>` as an
optional CLI **override** (replaces the hw-YAML block when given). The `hw_config.device_profiles`
setattr workaround is GONE — it's a real config field. Decode inheritance holds identically
(fresh decode instances receive the same hw_config).

**A2. op_key must survive `_split_tp_node`.** Add `"op_key"` to the `_copy_node_metadata`
attribute tuple (llm_execution.py:45-58). At injection time, ASSERT every compute node with
duration > 0 carries `op_key` OR is on an explicit whitelist (comm placeholders, ZeRO-3
gathers, zero-duration no-ops) — raise otherwise. Add a binary-free graph-level unit test:
build a flattened tp2 + tp_overlap>0 graph, apply transforms, inject, assert every
nonzero-duration compute node was scaled (this path is untestable end-to-end due to the
binary-skew failures, so the unit test is mandatory).

**A3. Stage-level nodes are additive, not multiplicative.** Per-GEMM transformer nodes carry
pure compute_time (multiplicative per-op ratio is exact — verified, incl. the +self.O term).
But embedding/linear_softmax stage nodes carry `compute_time + comm_time`
(train_timing.py:4354-4357), and comm must NOT scale. For those: `new = base + (compute_k −
compute_base)` using the bank's compute/comm split. Optimizer nodes: re-price via
`tc_k.get_data_parallel_reduction_llm(...)` per profile (cheap; HBM-BW-bound — the
transformer aggregate ratio is wrong for them) and use the true optimizer duration delta.

**A4. Decode re-pricing goes through decode's own pricing path.** The bank is generalized:
"re-price the op set of THIS instance under profile k via the SAME pricing entry point this
instance used." For training/prefill: `compute_all_gemm_and_node_times`. For each decode
sample step: the fresh temp_time_calc re-prices via `_build_decode_transformer_results` /
`prepare_decode_graphs` pricing with that step's gemm_shapes/total_seq_len, under K
deepcopied scaled configs with `device_profiles` STRIPPED (see A6). Cost: K× per sampled
step — document; `sample_every` controls it. Decode-only op_keys (e.g.
`attention_scale_softmax` decode variants) must resolve in the per-step bank. Add a
flattened-inference uniform-profile==scaled-config equivalence test.

**A5. Grad accumulation contract.** Both flattened runs' per-rank data are kept (do not let
the final run overwrite the no_dp run's): per-device
`compute_busy_s = busy_no_dp × (GA−1) + busy_final`, `makespan_s = no_dp_total × (GA−1) +
final_total` (= Total Time, pre-interleave-scale; see A7), and `device_metrics.json` also
carries a `runs: [{name: "no_dp", weight: GA−1, ...}, {name: "final", weight: 1, ...}]`
sub-array with per-run per-rank busy/wall. The results-txt idle lines use pricing-time
(baseline-instance) counters and are GA-independent — unchanged semantics.

**A6. No recursive banks.** When building profile variant configs: `hw_k.device_profiles =
None` (explicit step). Profile-variant instances (tc_k) must never construct banks or
dispatch.

**A7. sched_idle_frac semantics.** `sched_idle_frac = clamp(1 − busy_r/makespan, 0, 1)`.
Document: collective/SEND/RECV time lands in the idle complement (comm is not separable from
ET); a thermal consumer must not treat it as pure idle-power time. `makespan_s` = RAW
AstraSim total (`flattened_astrasim_total`, pre `_pipeline_interleave_scale`); record
`pipeline_interleave_scale` in the JSON so consumers can reconcile with Total Time.
Per-rank busy attaches to `time_calc_obj` inside `run_astra_simulation_only_onepath` (do NOT
change its return shape — 4 call sites).

**A8. lm-head / global-bucket attribution.** The flattened linear_softmax node lands on
exactly ONE device: rank 0 of its stage (`llm_execution.py:529-537`). `hosts_lm_head` and
the global kernel-idle bucket attach to that single hw_id only.

**A9. Idle instrumentation details.** New record sites in TP/CP/hybrid/flash paths must pass
per-rank SHARDED flops matching the per-rank observed time (total flops would clamp idle to
0). MLA: `_mla_composite_direction` prices via roofline directly — record
`compute_time − total_local_flops/self.th` there; the three attention paths (MLA/non-flash/
flash) are mutually exclusive, no double count. Embedding + pointwise ops remain
uninstrumented in v1 — document in the JSON docs (stage-0 devices under-report kernel idle).

**A10. layers_hosted** comes from the actual `_stage_for_layer` partition
(simulate_train_graph.py:777-790; front-loaded remainder), never `num_layers//pp`.

**A11. Gates & misc.** Second mode gate lives in `LLMExecutionDispatcher.__init__` (mirror
`_initialize_fault_mappings`), not in the flattened runner (which only runs in flattened
mode); the runner keeps the hw_id-set validation. Profiles are read via `time_calc` — zero
dispatcher-signature changes (all 5 ctor sites are keyword-only and already pass time_calc).
Validate `dp_devices` ⇒ training ∧ dp>1 at TimeCalculation init; error `--device_profiles` /
hw-YAML profiles with GEMM/VIT modes in run_perf. Invariant 2/3 equality asserted at ≤1 µs
per rank (exact only for overlap=0 configs). Inference results file must contain ALL of:
`GPU_time_frac_idle`, `GPU_time_frac_idle_thermal`, `Prefill Time`, `Decode Time`,
`Prefill Idle Time`, `Decode Idle Time`, `Prefill Idle Layer Time`, `Prefill Idle Global
Time`, `Decode Idle Layer Time`, `Decode Idle Global Time` (plain pair is a consumer
fallback requirement). device_metrics.json gains per-device axis coordinates
(tp/cp/ep/pp/dp indices decoded from the rank layout) — the waferscale grid needs physical
placement. Document: txt idle fractions carry baseline semantics when profiles are ON
(profile-aware consumers must use the JSON); repointing thermal_stco at TARGET gives FORMAT
parity, not numeric parity (timing model evolved; fingerprint auto-invalidates); the memory
peak sim sees injected dp0-slice durations (verify it uses durations for ordering only —
note in code). Add negative-path tests (malformed YAML, unresolved hw_id, every mode gate,
optimize_2dmap, dp_devices misuse, GEMM/VIT) and an astra-cache distinctness test (two
profiles, one cache dir, different results).

## Deliverables checklist

- `device_profiles.py` (+ `configs/device-profiles/{uniform_baseline,hot_example}.yaml`)
- base_timing/train_timing/inference_timing/simulate_inference_graph idle restoration+extension
- run_perf CLI + output lines + device_metrics.json writer
- flattener `op_key` metadata + post-transform injector + validation + summary logging
- executor per-rank busy accumulator + stdout rank-keyed parsing + `RAPID_ASTRASIM_BINARY`
- thermal-port configs copied from the vendored snapshot (hardware + 4 model YAMLs +
  `LLM_thermal_legacy_compat.yaml`)
- tests (new file(s) under `tests/`), README + AGENTS.md sections

## IMPLEMENTATION NOTES (Phase B, 2026-07-24)

Feature 1 + Feature 2(b) + device_metrics.json landed on `heterogeneous`
(Phase A — idle accounting — landed earlier as f690836..33bbf14). Map:

- **Schema (A1)**: `config.py` `parse_device_profiles` → frozen
  `DeviceProfilesConfig`/`DeviceProfileSpec` on `HWConfig.device_profiles`;
  `DeviceProfileError(ValueError)`. CLI `--device_profiles` loads the same
  schema via `device_profiles.load_device_profiles_yaml` and replaces the block.
- **Bank**: `device_profiles.build_timing_bank(owner)` reads the owner's
  `_device_profile_pricing_context` recorded at pricing time
  (train `_build_training_graphs_and_memory_data`, inference `calc_time`
  prefill, `prepare_decode_graphs` per decode step) and re-prices per distinct
  scale tuple via `type(owner)(deepcopy(hw), ...)` with `device_profiles=None`
  (A6). Identity profiles alias the baseline pricing bit-exactly. Optimizer
  re-priced via `get_data_parallel_reduction_llm` (A3); per-profile kernel idle
  = the variant instance's own counters (throttled `self.th` reference).
- **Injection (A2/A3/A4)**: `PipelineGraphFlattener` stamps
  `op_key=(op, direction)` on every compute node (`op_key` added to
  `_copy_node_metadata` so `_split_tp_node` heads keep it);
  `LLMExecutionDispatcher._build_and_inject_flattened_root` injects after
  `apply_overlap_transforms`: multiplicative per-op compute ratios for GEMM
  nodes, additive compute-deltas for embedding/linear_softmax/optimizer,
  per-dp tuples for dp>1. Assertion: every duration>0 compute node carries
  op_key (zero-duration nodes are the only keyless whitelist). Faulty-links
  style summary table logged under category `profiles`.
- **Gates (A11)**: primary in `TimeCalculationLLM.__init__` (mode, ViT,
  dp_devices ⇒ training ∧ dp>1); defense-in-depth + optimize_2dmap + MoE in
  `LLMExecutionDispatcher._initialize_device_profiles`; GEMM/VIT in run_perf.
- **Schedule idle (A5/A7)**: executor accumulates pre-rounding float busy per
  rank; onepath attaches `last_astrasim_per_rank_busy/_rank_meta` to
  time_calc (return shape unchanged); the flattened runner stores per-run
  records under `flattened_run_no_dp` / `flattened_run_final`. Wall times are
  keyed by the captured `sys[<rank>]` index.
- **device_metrics.json**: `device_profiles.write_device_metrics_json`,
  called from run_perf for every flattened run. GA-combined makespan/busy with
  a `runs` sub-array (A5); inference = prefill + trapezoid-integrated decode
  (per-sample records + per-profile idle integrated with the same weights as
  decode time, `simulate_inference_graph._decode_sample_weights`).
  `makespan_s` raw + `pipeline_interleave_scale` recorded (A7); coordinates
  via `layout_utils.decode_axis_coordinates` (A11); `layers_hosted` from the
  graph's front-loaded `layers_per_stage` (A10); `hosts_lm_head` on the single
  hw_id where the flattened linear_softmax landed (A8; true for that hw_id in
  every dp replica — each replica executes its own vocab projection).

Deviations / notes:

- The hetero-sanity invariant is asserted on per-device DELTAS vs the
  profile-free baseline (slowed device's sched_idle_frac drops, fast devices'
  rise, makespan grows). Comparing absolute idle fractions between devices
  within one run is confounded by baseline pipeline asymmetry (the last stage
  hosts the lm head + optimizer), so "slowed < fast" does not hold in general.
- `convert_rapid_llm_graph_to_chakra_et` returns a 4th element (rank stats);
  its two test consumers in `test_moe_expert_imbalance.py` were adapted.
- `get_remote_memory_path` also follows `RAPID_ASTRASIM_BINARY` (the bundled
  remote-memory JSON lives in the same astra-sim tree as the binary) — without
  it the env override could generate ETs but never run.
- Binary-free tp2+overlap graph tests stub `NetworkModel._astra_collective`
  (tp>1 comm pricing otherwise requires the AstraSim binary even to build).
- JSON kernel-idle numerators keep pricing-time semantics (once per layer
  type; GA/microbatch-independent), matching the txt lines; documented in the
  JSON `notes` field.
