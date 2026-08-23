# Recon: OPTIMA's Mamba2, pools, and VLM machinery

Read-only recon of `/app/nanocad/projects/cim_ctt_big_optima/perf_model/`,
2026-08-23, feeding the plan-of-plans iteration and the WS1/WS3 detail plans.

## 1. The Mamba2 model (`modeling/stages_mamba.py`, 426 lines)

Five stages per layer, per token (M = 1 vector through the pipe):

| Stage | Kind | What it prices |
|---|---|---|
| M1_Projection | analog | fused W_xBC (D -> d_inner + 2*groups*state) + dt-proj (D -> heads, own array shape `cols = heads/mux`; mux must divide heads); RMSNorm absorbed (raises if it would bind) |
| M2_ConvDecay | digital | max(depthwise conv k=4, softplus/add for delta, exp/mul for decay); conv-state SRAM holds k-1 taps |
| M3_Scan | digital | the SSM state update: `s3_scan_collection.get_execution_cycles(d_ssm, N, groups, heads, microcycles_per_token=100)`; state SRAM d_ssm x N sized by bandwidth; optional "trick" mode with its own T-SRAM |
| M4_GatePrep | analog | W_z (D -> d_ssm); SiLU + GroupNorm + mul absorbed (raise-if-bind) |
| M5_OutResidual | analog | W_o (d_ssm -> D); adders absorbed |

Phase law (`pipeline.py: model_mamba2`): **prefill and decode share the same
per-token stage times** — the scan is recurrent in both phases. Sequence time
= `max(stage) * (prefill_tokens + decode_tokens)`; tok/s = 1/max(stage); FPS
= sequences/s. An `opt_prefill` variant (stage-4 skipped on the last-layer
fraction) exists but is hardcoded False. Analog vec time is **quantized to
the digital clock**: `((ceil(vec_us*1000/cycle_ns)+1) * cycle_ns)` — our
fws_cim analog law does not do this; WS1 parity must replicate it for Mamba
stages or the gate will miss by a cycle.

Mamba TP exists (`_apply_mamba_tp_bytes`): shards d_ssm, groups, heads by
`tp.degree`, adjusts per-stage byte flows; reports show "TP groups".

## 2. OPTIMA already has a two-pool chiplet model

`cfg.pools.analog` / `cfg.pools.digital`: per-pool `num_chips`,
`chip_area_mm2`, `overhead_fraction`, plus a `stage_pool_map` assigning each
stage to a pool. The compiler sizes both pools per candidate (e.g. Mamba
Giant at mux=1 needs 9 digital chips — the scan dominates; at mux=16, 1).
Digital engines are a component library (`digital_hw_components_{22nm,12nm}
.yaml`, scaled to 5nm via `_digital_tech_scaling`); `interconnect_cfg.yaml`
carries link bandwidth caps and pJ/bit. **WS3 is therefore a port-and-improve
of an existing pool model, not an invention** — what OPTIMA lacks is
contention (pools are sized so stages never queue) and any notion of
schedule; that is what Tier S adds.

Other knobs already in OPTIMA the plan should absorb: `optimize_analog_clock`
(DSE axis we deferred), `stack_3d` + `stack_3d_height` per scenario, and a
`bank_factor` (see §3). Reports carry a per-stage "mismatch" ratio
(max_stage/stage) — a useful balance diagnostic worth keeping.

## 3. The VLM is modeled as MULTI-TENANT, not as a chain

`compiler_vlm_vit_mamba_*.yaml` declares:

```yaml
multi_tenant:
  models: [vit_huge.yaml, mamba_giant.yaml]
  weights: [11, 1]        # 11 ViT frames per 1 Mamba sequence
  bank_factor: 2          # array banking for co-residency
```

Throughput is combined as weighted raw throughputs into a
"model-sequence throughput (SPS)". There is **no dataflow coupling** — no
frames-feed-tokens latency, no handoff. Model type `MultiTenant`.

**Consequence for the plan:** WS2 (VLM) and WS8 (multi-tenancy) are closer
than the plan assumed. Decision needed from George: reproduce OPTIMA's
weighted co-residency as the WS2 baseline (cheap, matches Table 1) and treat
the true latency-coupled chain as the upgrade — or go straight to the chain.
The plan's current WS2 wording implies the chain; Table 1 parity needs the
weighted form.

## 4. Table 1 provenance — answered (plan §7 item 6)

`run_compiler_table.py`: DEFAULT_SCENARIOS = vit_large/huge/g 5nm,
mamba_base/giant 5nm, vlm 2d3d8, vlm max_perf; SUMMARY_MUX_BY_RUN pins the
mux per row (e.g. mamba_base 24, vlm_2d3d8 8, vlm_max_perf 2). Recorded
outputs in `output/`: per-scenario `.md` reports (summary, full mux sweep
table, candidate table, Pareto png) and TSV summaries. The proposal's Table 1
rows map onto these.

## 5. WS1 validation targets — answered (plan §7 item 5)

Concrete recordings to gate against:

- `output/compiler_mamba_giant_22nm.md` (+ `_3d_h4`, `_3d_h8`): mamba_giant,
  mux sweep {1,2,4,8,16} with chips/area/FPS/period per point; selected
  mux=16: period 657.920 us, 1.520 KFPS, analog 2 chips / digital 1,
  bottleneck M1_Projection, TOPS/W 38.480.
- Mamba model configs: `configs/models/mamba_base.yaml` (D=768, 24L, 24H,
  N=128, groups=8, expand=2, k=4, prefill 1792, decode 256, A8) and
  `mamba_giant.yaml`; also `mamba2_2p8b_zephyr_cobra.yaml`.
- `configs/analog_mamba_lookup.yaml`: the Mamba-shaped analog array library
  (incl. the dt-projection array variant).
- Tests: `tests/test_calc_analog_optima.py`, `tests/test_ctt_800mm2_vit.py`.
- VLM recordings (WS2/WS3 targets): `compiler_vlm_vit_mamba_22nm.md`,
  `_5nm.md`, `_5nm_2d3d8_*`, `_5nm_max_perf` — the last two are Table 1's
  VLM rows (5.858 KSPS max-perf figure reproduced in the recorded report).

## 6. Digital-chiplet scope — mostly answered (plan §7 item 7)

In OPTIMA the digital side is a **fixed-function engine library**: per-stage
collections (depthwise conv, softplus/add, exp/mul, scan, SiLU, GroupNorm,
adders, RMSNorm, plus the transformer-side attention/softmax) with cycle
models and area/power from the components YAML, instantiated into the
digital pool. No programmable-kernel cost model anywhere. Recommendation:
WS1/WS3 adopt the fixed-function library form (port the component YAMLs);
programmability stays out unless George says otherwise.

## 7. Corrections this recon forces on the plan of plans

1. WS3 reframed: port OPTIMA's pool model; the new content is contention +
   ownership, not the pool concept itself.
2. WS2 needs the George decision in §3 (weighted co-residency vs true chain
   — or baseline + upgrade).
3. WS1 gate must include the analog-time digital-clock quantization quirk
   and Mamba TP byte sharding.
4. WS6 gains two OPTIMA-proven axes early: `optimize_analog_clock`,
   per-scenario 3D stack height; and should keep the mismatch diagnostic.
5. §7 open questions 5-7 are now answered or narrowed; item 4 (MX timing)
   and items 1-3 remain.
