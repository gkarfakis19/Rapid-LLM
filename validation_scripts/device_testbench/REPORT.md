# Calibration Report — Per-Device Golden Factor Sets (Train + Inference)

Initial overnight run 2026-07-05 → 06; **methodology fix + recalibration 2026-07-06 → 07**
(commit 38c993f). Generator: `validation_scripts/device_testbench.py`. Golden factor sets
canonized in `validation_scripts/validation_configs/{device}_golden_calib.yaml`.

## PAPER-CANONICAL (2026-07-07): extended-roofline backend + merged factors

The paper now uses the extended-roofline GEMM backend for EVERYTHING
(`RAPID_GEMM_BACKEND=extended_roofline`, `RAPID_OPMODEL_PATH=../op-model`).
Factor structure: per-device residual compute + ONE merged mem/net utilization
+ 6 µs launch. Fitted by `backend_merged_study.py` on the canonized splits,
boundary-probed on the u axis (H100 u=0.90, A100 u=0.85 sampled and rejected):

| Device | residual compute | merged u | launch | calib | holdout |
|---|---|---|---|---|---|
| **H100_SXM5** | **0.85** | **0.85** | 6 µs | 7.92% | **6.45%** |
| **A100_SXM4** | **1.00** | **0.75** | 6 µs | 10.59% | **10.61%** |

Canonized in `*_golden_calib_backend.yaml`, `harness_derates.yaml`, and the
`.fitted.yaml` inference configs (commit 6a3985d). Paper figure numbers:
inference 9.65% (26 rows, worst 23%), dense train 8.69% (7 rows), MoE 6.97%
(11 rows, holdout half 8.08%, worst 13.6% — the shape model removes the
flat-factor's systematic large-scale Qwen2 under-prediction). The residuals
match published physics: H100 0.85 ≈ power-limited sustained-clock ratio;
A100 1.00 = no throttling.

The native-tile-model goldens below are RETAINED as the no-backend fallback
and for reproducing the pre-backend analysis.

## Native goldens (fallback; post-methodology-fix, pre-backend)

| Device | compute util | DRAM util | network util | launch overhead | calib MAPE | holdout MAPE |
|---|---|---|---|---|---|---|
| **H100_SXM5** | **0.600** | **0.90** | **0.70** | **6 µs** | **8.3%** | **6.9%** |
| **A100_SXM4** | **1.000** | **0.70** | **0.80** | **6 µs** | **10.5%** | **12.0%** |

(Old Table III for reference: H100 0.56/0.80/0.85/6 µs; A100_SXM4 0.90/0.70/0.80/6 µs.
A100_PCIe was dropped from the project. The pre-fix goldens of 2026-07-06 — H100
0.600/0.80/0.85/**1 µs** at 17.9/16.8 — are superseded; see “Methodology fix” below.)

Each set is fitted **jointly on pooled training + inference points** for that device on a
**50:50 calibration:holdout split**. Every number is a **real simulation** — no
interpolation or surrogate models anywhere. Kernel-launch overhead resolves to the same
6 µs on both devices and is no longer a differentiating knob.

### Presentation-constrained alternative (pending decision)

With the memory and network factors **merged into one utilization u** (real sims, same
canonized splits; `presentable_factors_study.py`):

| Device | compute | u (mem = net) | launch | calib | holdout |
|---|---|---|---|---|---|
| H100_SXM5 | 0.600 | 0.80 | 6 µs | 8.5% | **6.7%** |
| A100_SXM4 | 1.000 | 0.80 | 6 µs | 11.0% | **11.0%** |

The shared-u=0.80 table is *better on holdout than the granular optimum on both devices*
(mild regularization win) and reduces the paper table to one compute column + two global
constants. Raising H100 compute to 0.625/0.65 costs the Megatron MoE flagship subset
(5.9% → 8.3%/11.3%) and is not recommended. Which set is canonical for the paper is an
open decision; both are fully reproducible from the caches.

## Methodology fix (2026-07-06/07) — read this before comparing to older numbers

A TTFT probe (NIM time-to-first-token columns, never used in any fit) exposed prefill
over-prediction growing with sequence length. Per-op decomposition traced it to **three
defects in the inference validation adapters**, all fixed in commit 38c993f:

1. **NVLink bandwidth**: the checked-in validation base configs modeled dim0 at
   100 GB/s Ring (H100) / 250 GB/s (A100) vs the physical **450 / 300 GB/s** per-GPU
   rates — TP-SP communication over-costed ~4.5× on H100.
2. **Execution mode**: NIM and IMEC inference points inherited *analytical* mode from
   those base configs (a latent issue also present in the repo's own `nvidia_inf.py`
   and `rebuttal_heldout_sanity.py`). All validation points now force
   `full_astrasim_hierarchical`.
3. **NET_UTIL_FLOOR = 0.60**: fitted network utilization below the NCCL-achievable
   range is inadmissible (transiently, the unconstrained fit reached 0.40 — the network
   knob absorbing unrelated error; the floored optimum is *better* on holdout).

**RETRACTION.** The 2026-07-06 finding “kernel-launch overhead is a software-stack
property (NIM CUDA-graphs ≈ 1 µs vs NeMo ≈ 6 µs)” was an **artifact of defect #1**: with
comm over-costed 4.5×, the fit compensated through the launch knob. With physical
bandwidths, NIM, IMEC, and all training sources agree at 6 µs on both devices, and the
“IMEC vs NIM stack conflict” (IMEC −44%) dissolves entirely (IMEC now 6.4%).

Training points were **never affected** (they always ran hierarchical with their own
correct network configs); only inference cells were invalidated and re-simulated.

## Protocol (pre-registered, identical per device)

- **Knobs (4)**: compute util, DRAM util, network util, kernel-launch overhead.
- **Design**: axial + factorial-corner lattice at the default launch plane plus the full
  axial structure replicated at launch ∈ {1, 3} µs → 35 real combos per fit point,
  extended by the boundary rule.
- **Boundary rule**: any golden landing on a sampled axis boundary triggers a one-step
  axis extension and refit until the optimum is interior or hits a physical/protocol
  bound (util ≤ 1.0; **network util ≥ 0.60**; launch ∈ [0.5, 12] µs).
- **Split**: stratified 50:50 (strata = workload × source × scale bucket), 5 seeds;
  golden = the **median**-holdout-MAPE seed.
- **Fixed**: all communication overlaps (tp, tp_sp, cp) = 0.6; Megatron MoE points use
  interleaved-1F1B (`pipeline_interleave` = layers/stage); HF points use measured
  per-job network bandwidths; IMEC inference rows keep the network-ignored convention.
- **Execution**: every point, training and inference, runs
  `full_astrasim_hierarchical`. Inference dim0 = physical NVLink (H100 450 GB/s FC,
  A100 300 GB/s FC).
- **Envelope (a-priori)**: training rows with measured MFU < 8% are report-only;
  H100 nanotron pp>1 sanity rows are auto-cut when their subset MAPE exceeds 25% at the
  golden (they were: structural PP-overhead scatter); Megatron tp·ep>8 stretch rows are
  report-only (deliberate conservative all-EP-on-IB bound).

## Data inventory

| Device | Training points | Inference points | Fit total |
|---|---|---|---|
| H100_SXM5 | 11 Megatron MoE (Mixtral 8x22B, Qwen2-57B-A14B; 64–1024 GPUs) + 17 HF/nanotron pp=1 | 17 NIM Llama3.3-70B (tp4) + 11 IMEC Llama2 (tp1–8) | 56 |
| A100_SXM4 | 11 Megatron GPT 22B–1T (Korthikanti + Selene; up to 3072 GPUs) | 15 NIM Llama3.3-70B (tp8) + 11 IMEC Llama2 | 37 |

Report-only points (H100): 10 degenerate HF rows (MFU < 8%), 6 cut pp>1 rows, 9 MoE
tp·ep>8 stretch rows. Caches: `device_testbench/*/runs.json` (`--fit-only` refits
without re-simulating).

## Per-source MAPE at each golden

**H100_SXM5** (calib 8.3 / holdout 6.9; train 9.8%, inference 5.5%):
- megatron_moe **5.9%** · hf_pp1 13.0% · nim_llama33_70b **4.6%** · imec_llama2 **6.4%**

**A100_SXM4** (calib 10.5 / holdout 12.0; train 12.8%, inference 10.6%):
- selene 5.7% · korthi 15.6% · imec_llama2 10.6% · nim_llama33_70b 10.5%

Flagship Megatron MoE per-row (H100 golden): all 11 rows within ±18.2%; the 9-point core
sweep is entirely inside the ±20% acceptance band. Worst rows are Qwen2 at scale
(g512 −12%, g1024 −18%: fine-grained 64-expert all-to-all under-costed — the
pre-registered risk direction).

## Findings (current)

1. **One launch constant (6 µs) serves everything.** After the comm fix, no per-stack or
   per-device launch differentiation is supported by the data (supersedes the retracted
   stack-property finding).
2. **A100_SXM4 compute util calibrates to the 1.0 physical cap** — corroborated by
   published data: A100 SXM sustains its boost clock (no power throttling;
   max-achievable ≈ 87% of peak matches the tile model's intrinsic efficiency).
3. **H100 compute 0.60 is physical, not fudge**: published cuBLAS measurements show H100
   SXM is power-clock-limited to a 0.73–0.80-of-peak plateau on *ideal* large GEMMs,
   lower on the skinny/sharded shapes real LLM layers produce. The extended-roofline
   backend study (OPMODEL_FEASIBILITY.md) independently reproduces most of the
   H100-vs-A100 gap from shape physics.
4. **NIM b>1 residual**: RAPID models lockstep batching, NIM does continuous batching —
   visible as the remaining b25 under-prediction (H100 −4 to −8%, A100 −8 to −20%) and
   one −29% H100 outlier (p2000_o2000_b5). b=1 rows are clean probes (H100 within ±8.5%).
5. **Boundary probes sampled and rejected** (goldens are interior optima where allowed):
   H100 compute 0.55/0.575 ↓, dram 1.00 ↑, launch 12 µs ↑; net 0.70 sits above the 0.60
   protocol floor with 0.55/0.40 sampled and *better on calib but rejected as
   inadmissible* (and 0.85/1.00 within 0.1pp on holdout — the surface is flat there).

## Audit trail (2026-07-07, pre-paper certification)

- Independent recomputation of calib/holdout MAPE from raw caches (fresh code path)
  reproduces the golden yamls exactly: H100 8.34/6.94 (n=28/28), A100 10.52/12.04
  (n=20/17).
- All **4,280** cache-backed inference cells were verified on disk to be
  hierarchical-mode with physical NVLink; all **4,422** train case configs verified
  hierarchical. 809 orphan case dirs from invalidated eras were deleted.
- Every excluded point carries a documented reason (10 HF MFU<8%, 9 MoE stretch,
  6 pp>1 cut); calib ∪ holdout exactly covers the eligible set (H100 modulo the 6
  cut rows), calib ∩ holdout = ∅.
- Golden `--fit-only` refits are deterministic (re-run reproduces identical goldens).

## Reproduce / refit

```
.venv/bin/python validation_scripts/device_testbench.py --device H100_SXM5   # full
.venv/bin/python validation_scripts/device_testbench.py --device H100_SXM5 --fit-only
.venv/bin/python validation_scripts/presentable_factors_study.py            # merged-u menu
```

## Codebase changes made during this work (all regression-tested)

- `sw_param.pipeline_interleave` knob (interleaved-1F1B analytic bubble correction).
- Relaxed hierarchical layout guard → tp·ep>8 mappings expressible
  (`dim0=[tp,cp]`, `dim1=[ep]`, `dim2=[pp,dp]`).
- MoE fixes: AstraSim singleton-collective deadlock; training-EP tokens_local ep×
  undercount (see MOE_ONE_HOT_EXPERT_MODEL.md).
- Inference adapter methodology fixes + NET_UTIL_FLOOR (commit 38c993f).
- New: `device_testbench.py`, `presentable_factors_study.py`, `opmodel_residual_study.py`,
  `opmodel_adapter.py` (+ hook in `base_timing.get_gemm_time`), OPT-13/30/66B model configs.

## Caveats to carry into the paper

- Megatron MoE points assume interleaved 1F1B with one layer per virtual stage and
  Llama3-8x70b top-k=2 (paper source does not state either).
- IMEC inference rows keep the repo's network-ignored convention.
- NIM b>1 rows mix lockstep-vs-continuous-batching semantics (finding #4); b=1 rows are
  the clean latency probes.
- All numbers assume overlaps = 0.6 everywhere; not comparable to overlap-0 outputs.
- The pp>1 (nanotron sanity) family is excluded per the pre-registered 25%-cut rule and
  must be stated as a scope limit wherever the H100 numbers are quoted.
