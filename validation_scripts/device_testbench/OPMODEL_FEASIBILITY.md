# Feasibility: eyao600/op-model dynamic-util model for RAPID-LLM factors

Analyzed 2026-07-06. Repo: https://github.com/eyao600/op-model.git (main @ c152e01, 6 commits).
Working copy + experiment artifacts: scratchpad `op-model/` and `opmodel_study/`.

## What it actually is

**Not a fork of the Rapid-LLM tree.** A standalone package (`src/opmodel/`): per-operator
prediction `LocalOp + HardwareSpec -> OpProfile` (latency, traffic, energy), deliberately
dependency-free and graph-free. Three registered models: `base`, `roofline`, and
**`extended_roofline`** — the "dynamic util" contribution (2.7k lines): CTA tiling, wave
quantization with tail efficiency, first-touch L2 reuse, per-level active utilization
(compute/SMEM/L2/DRAM), compute–memory overlap, bottleneck classification, and a
calibration module fitted against the EnergAIzer per-op artifact (A10 + A100-40GB-PCIe
measurements).

## Code state (it is indeed very raw)

- `numpy` used but not declared → CLI broken out of the box (`pip install numpy` fixes).
- 4 of 25 tests fail (half-landed `kernel_launch_overhead_s` rename; CLI tests fail via
  the numpy issue). 21 pass.
- Example yaml contains a typo (`beta_zero: truLooke`) silently absorbed by defaults.
- Only A10 / A100-40GB-PCIe hardware configs ship; no H100, no A100-80GB (we wrote both
  for this study from RAPID's spec values — 15 minutes each).
- Their own artifact validation is weak in aggregate: batched-GEMM time MAPE 72–75%,
  median predicted/measured ≈ 0.1 (microbenchmark sweeps dominated by tiny ops where
  measured time is launch/sync overhead); softmax/layernorm worse.

## The experiments (small workloads, all runnable in minutes)

**1. Our workload shapes through both models** (per-rank local GEMMs from the golden
testbench points; opmodel `extended_roofline` vs RAPID `get_gemm_time` at core.util = 1.0):

| shape | opmodel eff | RAPID tile eff @util=1 | RAPID eff × golden c\* | opmodel bottleneck |
|---|---|---|---|---|
| H100 nanotron h2048 ffn1 | 0.381 | 1.013 | 0.608 | smem_bandwidth_limited |
| H100 Mixtral expert ffn1 | 0.439 | 1.016 | 0.609 | l2_bandwidth_limited |
| H100 Qwen2 expert ffn1 (B=16) | 0.429 | 1.015 | 0.609 | l2_bandwidth_limited |
| H100 NIM 70B decode (M=25) | 0.030 | 0.091 | 0.055 | dram_bandwidth_limited |
| H100 NIM 70B prefill ffn1 | 0.424 | 1.016 | 0.609 | smem_bandwidth_limited |
| A100 GPT-530B qkv (tp8) | 0.605 | 1.000 | 1.000 | compute_roof_limited |
| A100 GPT-1T ffn1 (tp8) | 0.607 | 1.000 | 1.000 | compute_roof_limited |
| A100 UCI Llama2-7B ffn1 | 0.583 | 1.000 | 1.000 | compute_roof_limited |
| A100 NIM 70B decode (M=25) | 0.031 | 0.060 | 0.060 | dram_bandwidth_limited |

**2. M-sweep on H100 (K=2048, N=4096)**: opmodel eff rises smoothly 0.056 (M=64) → 0.406
(M=32k); RAPID tile model stays ≈ 1.0 across the whole range.

## Key findings

1. **RAPID's tile model has NO intrinsic large-GEMM inefficiency** (≈101% of tensor peak
   at util=1 for every LLM-layer shape). Our flat per-device compute factors therefore
   absorb *all* real kernel physics — which is exactly why they came out implausibly
   spread (H100 0.60, A100 1.00) and why A100=1.00 "works": it is a system-level
   compensation, not a kernel-efficiency estimate.
2. **The dynamic model reproduces our device gap from first principles.** opmodel says
   H100 LLM GEMMs run at 0.38–0.44 of peak (tensor peak outruns SMEM/L2 at A100-style
   kernel parameters) while A100 runs 0.58–0.61 (compute-roof-limited): an A100:H100
   efficiency ratio of ~1.43 vs our fitted flat-factor ratio 1.67 — ~85% of the mystery
   gap explained by shape/architecture physics, zero fitting on our data.
3. **It expresses the within-device workload spread too** (the M-sweep curve) — the thing
   that forced our H100 factor down to 0.60 to average nanotron-small against Megatron-big.
4. **Memory-bound regime is classified correctly** (decode → dram_bandwidth_limited), but
   opmodel and RAPID disagree 2–3× there (0.03 vs 0.06–0.09 of peak); measured NIM decode
   sits near the weights-read roofline, i.e., both are pessimistic, RAPID less so.
5. **H100 absolute numbers are systematically pessimistic**: the model's kernel-parameter
   defaults are A100-generation (mma 16×8×16, no wgmma/TMA modeling). The knobs to fix
   this exist (`mma_m/n/k`, `pipeline_stages`, tile sizes are op attrs), but nobody has
   H100-realistic defaults or an H100 artifact to calibrate against.

## Verdict

**Feasible and worth pursuing — as a mechanism, not as a drop-in.** The credibility
upgrade is real: "per-op physically-derived efficiency + one global scale" is a much
stronger paper story than three per-device fudge factors (one of which is 1.00). But the
fork today is a first-pass analytical model with weak absolute calibration, no H100
support, and hygiene issues. Its *relative* (shape- and device-wise) signal is already
good; its *absolute* scale is not.

## Recommended path (increasing effort)

1. **Zero-integration credibility cite (~0 days)**: use the study above in the paper's
   discussion — the dynamic model independently explains ~85% of the per-device factor
   spread from first principles, supporting the interpretation of our factors as
   shape-mix aggregates. (Results in `opmodel_study/{comparison.json,opmodel_results.json}`.)
2. **Adapter experiment (~1–2 days)**: `gemm_backend: opmodel` flag in RAPID routing
   `get_gemm_time` through `extended_roofline` (LocalOp maps 1:1 onto RAPID's GEMM
   descriptors; opmodel is import-clean). Then re-run OUR golden device testbench with
   compute-util replaced by (opmodel eff × one global scale) — the testbench we already
   built is exactly the evaluation harness, cache-compatible, ~one overnight of sims.
   Success criterion: one global scalar replaces three per-device compute factors at
   comparable holdout MAPE.
3. **H100 kernel parameters (research risk, days–weeks)**: wgmma-era defaults or an
   H100 per-op benchmark artifact for their calibration module. Without this, H100
   absolutes stay ~30% pessimistic and step 2 will still need a per-arch scale.
4. Upstream hygiene PRs to eyao600 (numpy dep, test fixes, yaml typo) if we adopt it.

## Integration pass results (2026-07-06, committed as db5ce88)

Adapter landed: `opmodel_adapter.py` + a kernel-time-only hook in
`base_timing.get_gemm_time`, env-gated (`RAPID_GEMM_BACKEND=extended_roofline`,
`RAPID_OPMODEL_PATH=<checkout>`), default-off and bit-identical when disabled.
The opmodel HardwareSpec is derived from the RAPID hardware config (clocks
audited: A100 configs use 1.41 GHz = boost = official 312 TF peak, so there is
NO hidden base-clock headroom; core.util stays a residual global scale).

A/B at the A100_SXM4 golden factors (native -> extended_roofline):

| point | native | opmodel |
|---|---|---|
| nvtrain GPT-22B selective (worst A100 case) | -41.8% | **-8.2%** |
| nvtrain GPT-22B full | -28.6% | **+6.4%** |
| imec Llama2-7B tp8 | -20.8% | **-5.6%** |
| nvtrain GPT-530B full (control, was good) | -2.2% | +51.1% |
| nim p5000_o500_b25 (decode-heavy) | -19.6% | +86.6% |

Verdict sharpened: the dynamic-util mechanism fixes exactly the small-GEMM
efficiency collapse that a flat factor cannot express — but opmodel's absolute
ceiling breaks the already-good regimes. Root cause located: the extended
roofline's pipeline model serializes the SMEM prologue per k-group
(`sm_stage_cycles = smem + math + (groups_k-1)*max(...)`) instead of hiding it
behind the multi-stage cp.async pipeline, capping compute-bound GEMMs at ~0.6
of configured peak (kernel-parameter attrs cannot fix this; larger tiles make
it worse). Separately, the streaming-GEMV/decode path over-costs memory-bound
ops 2-3x. Both are upstream fixes in extended_roofline.py — until then the
backend must stay experimental and the H100 port is premature.

## Final result: compute-derate elimination study (2026-07-06, commit 2ba3de6)

After fixing the SMEM-prologue serialization in extended_roofline (op-model
commit 3025422: steady-state stage = max(smem, math) under multi-stage
pipelines; was smem + math, capping compute-roof-limited GEMMs at ~0.6 peak)
and gating skinny GEMV ops (min(M,N) < 128) to the native path (RAPID commit
42d4b87), the residual-compute sweep on the canonized golden splits gives:

| device | native golden | backend result | holdout MAPE |
|---|---|---|---|
| A100_SXM4 | compute = 1.00 | **compute = 1.00 (derate eliminated)** | 12.04% -> **11.59%** |
| H100_SXM5 | compute = 0.60 | **residual 0.85** (or 0.90 @ net 1.00) | 16.83% -> **16.43%** |

The unexplained 0.60-vs-1.00 per-device spread collapses to 0.85-vs-1.00;
the H100 residual is attributable to unmodeled thermal/clock effects and the
model's A100-era mma parameters vs real wgmma kernels. The Megatron MoE
subset improves 9.0% -> 5.9%. Unchanged structural residuals: H100 IMEC
inference (~39%, decode-dominated rows on the conflicting NeMo-era serving
stack) and Korthikanti GPT-22B (~15%, tp8 comm/overhead, not GEMM).

Reproduce: RAPID_GEMM_BACKEND=extended_roofline RAPID_OPMODEL_PATH=
../op-model .venv/bin/python validation_scripts/opmodel_residual_study.py
(op-model checkout with the prologue fix lives at ../op-model).

## What was NOT verified

- opmodel accuracy against *measured per-op H100/A100-80GB kernels* (no artifact data for
  these devices exists in the fork; their A10/A100-40GB artifact shows large aggregate
  errors dominated by tiny-op overhead).
- Non-GEMM ops (attention/softmax/layernorm fall back to the simple roofline; their
  artifact accuracy there is poor).
- Energy modeling (present, first-pass coefficients; out of scope for our factors).
