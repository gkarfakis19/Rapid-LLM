{
  "summary": "Audit every published number in the QIF model: arithmetic, units, and whether the component census matches the law that spends it",
  "agentCount": 11,
  "logs": [
    "census-vs-law: 30 numbers recomputed, 4 findings",
    "scale-counts: 41 numbers recomputed, 5 findings",
    "units: 54 numbers recomputed, 6 findings",
    "arithmetic: 76 numbers recomputed, 8 findings",
    "roundtrip: 45 numbers recomputed, 9 findings",
    "survivors: 25 findings over 246 numbers (7 refuted)"
  ],
  "result": {
    "report": "# NUMBERS AUDIT — QIF FWS-CIM PERFORMANCE MODEL

Scope: every published number of the mapped (ADJ-10) decode path, the closed-form bridge, and the DSE artifacts. Repo root `/app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM`. D23 applies: no accuracy findings.

---

## 1. THE HEADLINE

Five auditors recomputed **246 published numbers**. **188 reproduced exactly** (76.4%). **58 failed** a recomputation. After adversarial re-derivation, **25 findings survived**; four of them are the same defect seen from four angles, so the report below covers **20 distinct defects**.

| Verdict | Count |
|---|---:|
| CORRECT (reproduced to the last printed digit) | 188 |
| CENSUS_MISMATCH (composition and timing law describe different machines) | 19 |
| DISAGREES_WITH_TWIN (two paths, two answers for one quantity) | 14 |
| WRONG_UNIT | 9 |
| WRONG_SCALE | 8 |
| WRONG_ARITHMETIC | 8 |

The 246 checks cover fewer than 246 distinct quantities. Lanes overlapped deliberately, so 31 quantities were recomputed by two or more auditors. Every overlap agreed.

### The three that change a conclusion

**1. Every energy conclusion is 1.74x–1.95x too high.** `fws_eval.py:735` folds two engines into one power number and charges it to three serialised ops. Granite total energy falls from 1.657466e10 pJ to 9.529562e9 pJ. Qwen falls from 4.467580e10 pJ to 2.293397e10 pJ. The over-charge is 42.5% (Granite) and 48.7% (Qwen) of the published bill. Any pJ/token claim, any energy-efficiency comparison, and any energy axis of the frontier moves.

**2. The Falcon-Mamba headline of 1,240,209 tokens/s is not reachable by the machine that is built.** Little's law on the model's own two metrics: (1/8.063157894737724e-07) × 5.9441426038242136e-05 = **73.72 tokens in flight**, against **32 resident streams** and a state store sized for exactly 32. The ceiling is 32 / 5.9441426e-05 = **538,345.09 tokens/s**. The published rate is 2.3037x above it. Granite (0.8712) and Qwen (0.8859) are legal; only Falcon-Mamba is not.

**3. The root cause of (2) is an arithmetic defect in the lowering.** `program/fws_build.py:639` keeps only `deps[-1]`, which is the stage-hold edge, not the activation. The producer then lives on the destination chip, so the link law returns \"intra-chip\". **72 of 81** Granite chip handoffs and **589 of 620** Falcon-Mamba handoffs are priced at 0.0 s. The traversal's real data edge is dropped from the graph, not merely un-priced.

### Two more that block trust in the artifacts

* **The closed-form ↔ DAG bridge gate is red.** `fabric_total_cycles` is an SA-fabric-only figure compared against a whole-stage fold. T1 4567 vs 65555; T3 45315 vs 614675; MoE 536735 vs 14745619 (13.6x–27.5x).
* **No shipped DSE artifact reproduces on the current tree.** Re-evaluating each artifact's own `selected_config.yaml`: ViT 789.84 → 290.77 requests/s (2.7164x), Qwen total silicon +14.68%, Granite +2.82%, Falcon +1.22%. Two of the four curves have no freshness gate at all.

### What the audit could not reach

* The 508-check OPTIMA validator was not run end to end.
* Full DSE sweeps were not re-run. Artifacts were re-evaluated from their own emitted `selected_config.yaml`.
* One arithmetic-lane pair — \"energy per completed token\" 1.743744e9 pJ and \"implied average power\" 113.824 W — is internally self-consistent but its denominator could not be reproduced. Those two columns are withheld.
* The Falcon-Mamba `decode_window = 34` robustness re-run was not repeated. It is not load-bearing.
* Accuracy is out of scope (D23).

---

## 2. THE FINDINGS, RANKED BY HOW MUCH EACH MOVES A PUBLISHED NUMBER

### F1 — Attention energy charges the whole chiplet to three ops that use disjoint silicon
`fws_eval.py:735` (power), `fws_eval.py:807` (application). Severity: critical. Converged: 2 auditors.

**Value now.** Granite `shared_digital_chiplet` = 1.6372116858e10 pJ, `total_energy_pj` = 1.6574659371e10 pJ. Qwen = 4.4285724364e10 pJ, total 4.4675803697e10 pJ.

**Correct value.** Granite chiplet 9.327019017e9 pJ, total **9.529561530e9 pJ**. Qwen chiplet 2.2543892629e10 pJ, total **2.2933971962e10 pJ**.

**Arithmetic.** `_engine_power_w(\"attention\")` returns `sa_fabric_composition().power_w + softmax_engine_composition().power_w` = 14.7460071 + 17.03145724 = **31.77746434 W**. `_attention` then calls `_fabric_energy(\"attention\", cycles / f_fabric)` for all three blocks. The DAG serialises them (`program/fws_build.py:1075-1080`; `_stage_attention_cycles` sums qk + pv + softmax + fill_drain).

Granite measured busy time by block: attention_qk 2.6132000e-04 s, attention_pv 1.5121474e-04 s, attention_softmax 1.2905263e-06 s, ssm_scan 1.1592000e-04 s.
Charged: 2.6132e-04 × 31.77746434 × 1e12 = 8.30408698e9 pJ, reproducing the reported figure to nine figures.
Engine-matched: 14.7460071 × (2.6132e-04 + 1.5121474e-04) + 17.03145724 × 1.2905263e-06 + 27.79329982 × 1.1592e-04, all × 1e12 = 6.105219702e9 + 2.1979544e7 + 3.221799315e9 = **9.327019017e9 pJ**.
Over-charge = 7.045098e9 pJ = 43.03% of the chiplet term, **42.51% of the whole published energy**. Ratio charged/matched = 1.755343. Total ratio = **1.7392888**.
Qwen: ratio 1.964422 on the chiplet term, **1.9480186** on the total, 48.67% of the published bill.
Falcon-Mamba is the control: no attention op, ratio 1.000000.

The defect contradicts the model's own disclosure at `fws_eval.py:783`: \"an idle engine costs 0 pJ here, which is a floor\". The softmax engine is provably idle during QK and is charged 17.03145724 W for 2.6132e-04 s.

**Fix.** Give `_engine_power_w` three cases — `sa_fabric`, `softmax`, `vector` — and select by block in `_attention`. Both compositions already exist. `derived_fabric_sizing.per_stage` already stores `qk_arrays`, `pv_arrays` and `softmax_width`, so the finer within-engine split (SIZING_METHODOLOGY V8, 2.03x) is available at the same site.

**OPTIMA cost.** None. OPTIMA prices no shared-chiplet energy. The measured per-block powers stay verbatim.

---

### F2 — Falcon-Mamba headline throughput violates Little's law by 2.30x
`fws_eval.py:2310`. Severity: critical.

**Value now.** 1,240,208.8772844595 tokens/s, beat 8.063157894737724e-07 s, resident_streams 32, per_token_latency 5.9441426038242136e-05 s.

**Correct value.** The invariant `beat >= per_token_latency / D`. On the currently published latency that ceiling is **538,345.09 tokens/s**, beat 1.8575446e-06 s. The correct beat is not known until F3 is fixed, because the latency itself moves.

**Arithmetic.** L = λW = 73.71978 tokens in flight. D = 32. The store is built for 32: 32 stages × 599 `SRAM_MACRO_256x1040` = 19,168 macros = 1016.026 mm2, sized to 2 layers × 311,296 B × 32 streams = 19,922,944 B per stage.
Root cause sum: 32 × 8.063158e-07 = 2.5802105e-05 s of stage occupancy (this equals the run's own `identity_d_times_beat_s`), plus 31 × 1.07629395e-06 = 3.33651e-05 s of link latency, total 5.916722e-05 s against the measured 5.944143e-05 s — a 0.46% remainder. The 31 links account for the whole gap, and that latency never enters the beat.
The run contradicts itself by name: `pipeline.identity_residual_rel` = **+0.5659**, while `identity_basis` states \"An uneven stage plan therefore measures a traversal SHORTER than D × beat\". Granite prints −0.1478 and Qwen −0.1289, both matching the text. Only Falcon's sign is wrong.

**Fix.** Fix F3 first, so the stream's second token really chains through the stages. Then refuse or correct any beat that violates `beat >= per_token_latency / D`.

**OPTIMA cost.** None. OPTIMA has no filled-pipeline regime.

---

### F3 — `_chip_boundary` keeps only the last dep, so steady-state handoffs cost 0 s
`program/fws_build.py:639`. Severity: critical.

**Value now.** Granite: 72 of 81 boundaries at 0.0 s. Falcon-Mamba: 589 of 620 at 0.0 s. Qwen: same pattern.

**Correct value.** Granite 9 × 1.02861023e-06 s per traversal = 9.257492e-06 s each. Falcon-Mamba 31 × 1.07629395e-06 s per traversal = 3.33651e-05 s each.

**Arithmetic.** `_rotation` (`fws_build.py:487`) passes `list(deps) + list(held)`. `deps` is the previous stage's tail of the same traversal. `held` is the stage-holds-one-stream edge from traversal j−1. `_chip_boundary` takes `producer = deps[-1]`, which is `held[-1]`, and `transfer()` reads `src_device` from that annotation (`fws_build.py:365`). `held[-1]` lives on the destination chip, so `src_chip == dst_chip` and the law returns \"intra-chip movement: a dependency, not a priced link (D17, disclosed)\" at 0.0 s.

Verified: Granite uid 4289 `decode1.boundary.c0->c1` deps = (776,) = `decode0.L7.residual_ffn`, annotated chip 1 — the *destination*. Bytes moved 3072.0, identical to uid 392, which costs 1.02861023e-06 s. Falcon uid 13891 `decode1.boundary.c0->c1` deps = (840,) on chip 1, 8192 B, 0.0 s. Byte count is identical in both, so this is not a size effect.

The dropped edge is real, not inferred. Consumer map over `prog.ops`: uid 391 `decode0.L3.residual_ffn` has consumers [392 (priced boundary), 3897]. uid 4288 `decode1.L3.residual_ffn` has consumers [7794] only — the boundary is absent. Traversal 1's stage-0 output feeds nothing in traversal 1.

Internal contradiction, which clears the \"intended, not a bug\" default: the `_rotation` docstring at `fws_build.py:456-463` says the hold edge is \"One more edge\", additional to the traversal chain described at 448-453. The code substitutes it instead. Second contradiction: two ops with the same name, the same `boundary_id`, the same byte count and the same law get prices of 1.02861e-06 s and 0.0 s.

**Fix.** Pass all of `deps` to the transfer, choosing `producer` as the last dep whose annotation chip_id equals `src_chip`, and attach the rest as control deps. Add the stage-hold edge, never substitute it.

**OPTIMA cost.** None. The ViT lockstep path never appends a `held` list, and the priced-boundary total on both mapped runs is exactly one traversal's worth, so no OPTIMA figure reads it.

---

### F4 — `total_energy_pj` is a window total published as a per-candidate figure of merit
`fws_eval.py:1246`, printed at `tools/fws_qif_dse.py:1359 / 2436 / 2469`. Severity: critical.

**Value now.** Granite 1.65747e10 pJ, printed as \"energy 1.657e+10 pJ over the timeline\" and as the `energy pJ` column of the Candidates table. The same machine at `mapping.decode_window` = 1 / 3 / 6 / 10 gives **1.30870e10 / 1.65747e10 / 2.18142e10 / 2.88083e10** pJ — a **2.2013x** range on a byte-identical beat.

**Correct value.** **1.744995e9 pJ per beat = pJ per token** under D29, measured on the settled tail the beat metric already selects.

**Arithmetic.** The window is `beats = streams + decode_window + 1` (`fws_build.py:198`), and both terms move.
`layers_per_chip` 4/8/20/40 → D 10/5/2/1, beats 14/9/6/5, E 1.65747e10 / 1.21999e10 / 9.57449e9 / 8.70697e9 = **1.9036x**.
Layer-firings per window: sum over t of min(D, beats−t) × layers_per_stage. D=10: 10+10+10+10+10+9+8+7+6+5+4+3+2+1 = 95, × 4 layers = **380 firings for 5 delivered tokens** = 76 per token. D=1: 5 × 40 = 200 = **40 per token**. The model has 40 layers, so 40 is the honest count. Predicted ratio 380/200 = 1.9000 against measured 1.9036.
Per-beat energy is the same on both machines: 1.74511e9 vs 1.74139e9, ratio 1.0021.
Settled tail from the run's own `converged_tail_intervals_s` (beats 11, 12, 13): 1.74506e9, 1.74533e9, 1.74459e9 → mean **1,744,994,598.30 pJ**. The published window total is 9.4984x that.

**Note on M10.** `docs/qif/SIZING_METHODOLOGY.md:118` expressly permits an analysis knob to size a *reported metric*. The finding stands on N2 alone: one column, one unit label, different windows per row, and no `beats_lowered` beside it.

**Blast radius.** Three sites only. The field feeds no Pareto objective, no knee/trim rank and no gate. No selected design and no refusal moves. What moves is every published energy and the cross-row energy ranking.

**Fix.** Publish `energy_per_beat_pj` over the settled tail, per component so the itemisation stays addable. Keep the window sum only as `window_energy_pj` with `beats_lowered` beside it.

**OPTIMA cost.** None.

---

### F5 — The bridge attention-fold identity compares an SA-fabric figure with a whole-stage fold
`validation_scripts/validate_fws_cim_vs_optima.py:1256`, `fws_bridge.py:725-732`, `cim_timing.py:3391`. Severity: critical.

**Value now.** T1 4567 vs 65555; T3 45315 vs 614675; Llama7B 25,837,599 vs 411,041,811; MoE 536,735 vs 14,745,619. Ratios 14.3541 / 13.5645 / 15.9087 / 27.4728.

**Correct value.** One stage cycle count on both sides.

**Arithmetic.** Of 35 bridge rows on T3, exactly one fails, and the stage itself agrees on both sides: `softmax cycles` 614675 == 614675 and `S2_attention time_us` 647.0263157894736 == 647.0263157894736. The closed form publishes `fabric_total_cycles = max(qk, pv) + fill_drain` (`cim_timing.py:3391`, no softmax term): 45219 + 96 = 45315. The DAG folds `max(qk_op, pv + fill_drain, softmax)` (`fws_bridge.py:538`) = 614675. Llama7B DECODE passes (477957 == 477957) because its softmax does not dominate.

**Stale text.** `cim_timing.py:3273-3305` (SOFTMAX_UNIT_UNRESOLVED) still says \"The law below therefore stays on the reference's unit\", while the law 16 lines below at `cim_timing.py:3321` prices **elements**: `passes = rows * ceil(keys/width)`.

**Fix.** Publish `stage_cycles = max(qk, pv + fill_drain, softmax)` and point the bridge at it, renaming `fabric_total_cycles` to `sa_fabric_cycles`. Delete the stale SOFTMAX_UNIT_UNRESOLVED text.

**OPTIMA cost.** Already paid. The element law is installed. The file itself states the cost: 20 of the validator's 508 checks — every recorded T1/T2/T3 period, fps and bottleneck, and all five bridge attention-fold identities. The fix above costs nothing further; it only makes the two internal paths agree.

---

### F6 — No shipped DSE artifact reproduces on the current tree
`docs/qif/dse/*`. Severity: major.

| Artifact | Shipped | Current tree | Move |
|---|---:|---:|---:|
| ViT `requests_per_s` | 789.8438141017363 | 290.77299801595257 | 2.7164x |
| Granite `total_silicon_mm2` | 10783.884513030001 | 11087.640301559999 | +2.82% |
| Qwen `total_silicon_mm2` | 8124.339678480001 | 9317.42819712 | +14.68% |
| Falcon `total_silicon_mm2` | 8719.972783776 | 8826.652730016 | +1.22% |

**Arithmetic.** Each artifact's own `selected_config.yaml` was re-evaluated. ViT prefill latency moves 0.0012660730921052632 → 0.003439108881578947 s, and 789.8438141017363/290.77299801595257 = 2.716358876137472 equals the latency ratio to 15 digits, so it is one time-scale change on one machine (ops_priced 994 on both sides). Granite per-macro pool moves 0.002418423 → 0.032184033 mm2/macro (13.31x); state store 12933 → 13752 macros. Qwen state store 69112 → 89296 macros; pool 0.003915542 → 0.052107482 (13.31x). Mtimes: the newest artifact is falcon at 2026-08-26 19:24:20; `cim_timing.py` 21:15:22, `fws_eval.py` 20:57:51, `program/fws_build.py` 20:55:44 are all later.

**Gate coverage.** `tests/test_qif_frontier.py:869` does re-walk the ladder and assert identity, but its CURVES tuple (line 651) holds granite and qwen only. ViT and falcon appear in no test and in no validator row (`validate_fws_cim_vs_optima.py:904-907`). The granite/qwen gate is currently red.

**Fix.** Regenerate all four curves. Add ViT and falcon to CURVES. Widen `verify_selected` (`tools/fws_qif_dse.py:2766`) to cover `total_silicon_mm2`, `state_sram_macros` and `macro_pool_silicon_mm2` — none is currently checked, and `total_silicon_mm2` is the number `max_silicon_mm2` refuses machines on.

**OPTIMA cost.** None.

---

### F7 — `derived_chiplets.built_duty_*` averages over stages, not over built chiplets — exactly 4.00x under
`tools/fws_qif_dse.py:830-831`. Severity: major.

**Value now.** Qwen selected point: `built_duty_mean` 0.2257015348261403, `built_duty_max` 0.8896393382396948.
**Correct value.** mean **0.9028061393045612**, max **0.9055585214480644**.

**Arithmetic.** beat = 2.1225632773951334e-05 s. 32 stages carry load: 24 at 1.1263157894736842e-07 s and 8 at 1.8769e-05 … 1.888316e-05 s. The code's own line divides each stage's busy by the beat and averages over 32 values. `fws_eval._stage_chiplet_groups(mapping)` returns 8 contiguous groups of 4 — the 8 chiplets the machine builds. Summing over those groups gives busy 1.9107e-05 … 1.9221e-05 s, duties 0.9002 … 0.9056, mean 0.9028061393045612. 0.9028061393045612 / 0.2257015348261403 = **4.000000** = 32 stages / 8 chiplets. The busiest chiplet carries 1.888316e-05 + 3 × 1.1263158e-07 = 1.9221053e-05 s, so `built_duty_max` is the wrong maximum too.

The wrong assumption is written down: `tools/fws_qif_dse.py:824-825` comments \"one per stage\", and the basis at 1292-1298 repeats it, while the same candidate reports `chiplets_built = 8`. Granite hides the defect because stages = chiplets = 10.

**Fix.** Compute duty over `_stage_chiplet_groups`, the same grouping ADJ-14 sizes the engines against (`fws_eval.py:4014`). Keep the first-fit packing as a separate \"how few could it be\" row under a different key.

**OPTIMA cost.** None.

---

### F8 — The report's headline silicon line decomposes a four-term total into two terms
`tools/fws_qif_dse.py:2396-2400`. Severity: major. Converged: 4 auditors.

**Value now (generated).** Granite `- total silicon 11087.6 mm2 (9778.8 analog + 424.124 shared digital)`. Qwen frontier `4552.488246624` printed against `9317.42819712`. Shipped Granite artifact line 15: `10783.9 mm2 (9778.8 analog + 307.846 shared digital)`.

**Arithmetic.** `total_silicon_mm2 = analog + digital + pools + state_sram` (`dse:976`); the line prints `analog_macro_silicon_mm2` and `shared_digital_silicon_mm2`.

| Machine | Printed pair | Stated total | Unnamed | Share |
|---|---:|---:|---:|---:|
| Granite (card) | 10202.925707 | 11087.640302 | 884.714595 | 7.98% |
| Falcon-Mamba | 7695.278672 | 8826.652730 | 1131.374058 | 12.82% |
| Qwen (card) | 3589.483848 | 4803.441293 | 1213.957445 | 25.27% |
| Qwen (frontier, current tree) | 4552.488247 | 9317.428197 | 4764.939950 | **51.14%** |

The missing terms are the per-macro pools and the state store. The same file gets it right twice: the budget-refusal message at `dse:1413-1419` names all four, and the Candidates table at `dse:2457-2459` closes exactly (9778.801824 + 1308.83847756 = 11087.64030156). One artifact, two decompositions, one total, only one closes — a D21 failure.

**Mitigation.** The next line printed (`dse:2401`, `silicon basis`) does name all four terms in prose.

**Fix.** Print `digital_silicon_mm2`, or expand to all four named terms.

**OPTIMA cost.** None.

---

### F9 — Attention fabric and softmax engine are charged to every chiplet, including the six that can never run attention (Granite)
`fws_eval.py:4753`, `tools/fws_qif_dse.py:884`. Severity: major.

**Value now.** `shared_digital_chiplet.area_mm2_total` 424.123882560 mm2, `power_W_total` 595.707641600 W; `digital_area_mm2_total` 1308.838477560 mm2; `digital_power_W_total` 812.096671520 W.
**Correct value.** 280.149643596 mm2 / 405.042855560 W; digital 1164.864238596 mm2 / 621.431885480 W; `total_silicon_mm2` **10943.666062596**.

**Arithmetic.** Per chiplet: SA fabric 16 × GEMMINI 0.767330073 + 1 × TRANSPOSER 0.010606661 = 12.287887829 mm2 / 14.7460071 W; softmax at w=1800 = 11.707818665 mm2 / 17.03145724 W; scan engine 6509 × 0.002829418 = 18.416681762 mm2 / 27.79329982 W. Sum 42.412388256 mm2 / 59.57076416 W, × 10 = 424.12388256 / 595.7076416.
Attention lands on 4 chiplets only. DAG census by chip_id: chip 11 {ssm_scan 39, attention_qk 13, attention_softmax 13, attention_pv 13}; 13, 16, 18 likewise; chips 10, 12, 14, 15, 17, 19 carry ssm_scan only. `derived_fabric.summary()` independently reports `stages_sized = 4`, stages 1/3/6/8.
Demand-matched: 4 × 23.995706494 + 10 × 18.416681762 = **280.149643596 mm2**. Excess **143.974238964 mm2** = 6 × 23.995706494 exactly = 33.95% of the chiplet term, 11.00% of digital area, 1.30% of total silicon. Power excess 6 × 31.77746434 = **190.66478604 W** = 23.48% of digital power.
Throughput is unaffected: binding term is stage 8, span 1.5323843241526032e-05 s, and no attention op sits on the six chiplets.

**Counter-argument weighed and rejected.** A homogeneous SKU is a legitimate design. But the repo applies the opposite rule one line below (`cim_timing.py:4541-4543` gates the scan engine on measured demand), no disclosure names the six fabric-less chiplets, and `SIZING_METHODOLOGY.md:437` (M3) classes this pattern as MAJOR. This is not a duplicate of V6a: Granite already has chiplets = stages = 10, so V6a's clamp does not fire.

**Fix.** Compose per chiplet against the demand of the stages it serves, using `_stage_chiplet_groups` (`fws_eval.py:3340`). If a single homogeneous SKU is intended, disclose its cost.

**OPTIMA cost.** None.

---

### F10 — A saturated attention stage takes the saturation softmax width: 27.6% of the chiplet buys 0.30% of the stage
`cim_timing.py:4797`. Severity: major.

**Value now.** Granite and Qwen derive `softmax_lanes = 1800`. Composed: 11.707818665 mm2 / 17.03145724 W per chiplet, × 10 = 117.078 mm2 / 170.31 W. Softmax is 31 cycles of the binding stage's 10,351.
**Correct value at width 900.** 5.85489907 mm2 / 8.51914024 W per chiplet; softmax 43 cycles; beat 1.5319633e-05 → 1.5332264e-05 s; tokens/s 65275.716 → 65221.939 (**−0.0824%**).

**Arithmetic.** `derive_fabric_sizing` runs the minimality search only on the non-saturated branch. All four Granite attention stages saturate (`target_ratio` 8.5123), so `width = saturation_softmax_width = n = 1800`. Law: `passes = rows × ceil(keys/width)`; rows = 3 tokens × 4 heads = 12; at 1800 passes = 12, cycles = 20 + 12 − 1 = 31; at 900 passes = 24, cycles = 43. Other stage terms are unchanged (qk 6437 + fill 96 + pv 3783/3787). 12 cycles / 0.95e9 = 1.2631579e-08 s.
Census at width 900 from `compose_softmax_engine`: FP_COMP 899, FP_ADD 1799, FP_MULT 1800, BF16_EXP 900, BF16_RECIP 1 → 5.85489907 mm2 / 8.51914024 W. Saving over 10 chiplets: **58.529 mm2 / 85.123 W**.

**Correction to the submitted fix.** The auditor's rule (\"smallest width matching the sat_width cycle count\") is a no-op: cycles(1800) = 31 and cycles(1799) = 43, so 1800 is already minimal for that target. The real change is the *target*, not the search.

**Fix.** On the saturated branch, pick the smallest width at which `softmax_cycles` no longer exceeds the stage's own qk + pv + fill term. On Granite that admits 900 (43 against 10,316) and every narrower width down to 10,316 cycles. No tolerance knob is needed, so D28 holds.

**OPTIMA cost.** None on shipped numbers. The OPTIMA T1/T2/T3 ViT card declares `softmax_lanes = 1` and skips the derivation.

---

### F11 — The SA fabric census pays for `a_qk + a_pv` arrays, but the DAG runs QK and PV strictly in series
`cim_timing.py:2180` (census), `cim_timing.py:2696` (partition), `program/fws_build.py:1075-1080` (serialisation). Severity: major.

**Value now.** 16 GEMMINI_SYS_ARRAY per chiplet = 12.287887829 mm2 / 14.7460071 W (derived num_arrays 8, replicas 1, 2 blocks per 32×64 array).
**Correct value.** 8 GEMMINI_SYS_ARRAY per chiplet = **6.149247245 mm2 / 7.3785671 W**, at bit-identical cycle counts.

**Arithmetic.** `fold_group_split` returns a disjoint partition, measured live as `a_qk = 4, a_pv = 4` on all 8 Qwen stages. Sweeping `_stage_attention_cycles` over num_arrays 2..16 on the live device: 2→61679, 3→46831, 4→32455, 5→32455, 6→25031, 7→25031, 8→17843, 9..16→17843.
Reading the same law with **one shared pool** of `folds` arrays serving each run in turn: 1→(32421, 29127); 2→(17573, 14751); **4→(10149, 7563)**; 8→(10149, 7563). Four shared arrays reproduce the shipped eight-array answer exactly: 10149 + 7563 + 96 fill + 35 softmax = 17843, the shipped `used_cycles`. Granite likewise: 8 arrays → (6437, 3783); 4 shared arrays → (6437, 3783).
No weight-reload penalty applies: `sa_cycles = ceil(m/32) × ceil(n/64) × (k+94) − 1` is output-stationary over the m×n tile, so neither operand is resident across the pair. Even charging fill/drain twice gives 17939 against 17843 (+0.54%) at half the silicon.
Unreachable silicon: 8 × 0.767330073 = 6.138640584 mm2 and 8 × 0.92093 = 7.36744 W per chiplet. Qwen (4 chiplets) 24.554562 mm2 / 29.46976 W. Granite (10 chiplets) **61.386406 mm2 / 73.6744 W**.
The serialisation is the evaluator's own rule: `fws_eval.py:281-287` states the chiplet has capacity 1, so the QK op occupies all 8 arrays while being priced on 4.
`SIZING_METHODOLOGY.md:263` already records `fold_group_split` as \"minimises qk + pv (serial ops) | OK\". Minimising a serial sum over a *partition* is the wrong optimisation: the sum is minimised by giving each run everything.

**What is dropped from the original claim.** The +29.1% throughput claim (24724.5 → 31920.5 tokens/s) is not carried. It needs a column split of one fold's n across arrays, which `fws_eval.py:4189-4196` refuses by name for want of a broadcast/merge reference. That is a stated design refusal under D28.

**Fix.** In the ADJ-10 derived path only, price each run on the whole pool: `qk = sa_cycles(m, n, k*ceil(folds/num_arrays))`, `pv = sa_cycles(m, k, n*ceil(folds/num_arrays))`. Then `saturation_arrays` becomes `folds`, not `2*folds`.

**OPTIMA cost.** None if confined to the derived path. It must **not** be applied inside `attention_call_timing` globally: at the closed form's declared `num_arrays = 2` with folds 4, a shared pool would read qk 17573 / pv 14751 instead of 32421 / 29127, which moves T1/T2/T3.

---

### F12 — `total_silicon_mm2` sums a package footprint with three silicon areas
`tools/fws_qif_dse.py:976`, `cim_timing.py:5768-5774`. Severity: major. Latent today.

**Value now.** `analog = slots × macro_footprint_mm2`, and `macro_footprint_mm2 = area_mm2_per_array / stack_3d_height`. Its own docstring reads: \"Stack height divides the FOOTPRINT only — never silicon area, never energy.\"
**Correct value.** 11087.64 mm2 of silicon at every stack height.

**Arithmetic.** With `stack_3d_height` = 1 / 2 / 4 the footprint is 2.0204136 / 1.0102068 / 0.5051034 while `area_mm2_per_array` stays 2.0204136. The analog term is the only term that moves: 9778.801824 / 4889.400912 / 2444.700456. Totals against a fixed digital 1308.83847756: 11087.640302 / **6198.239390** / 3753.538934, i.e. **44.098% and 66.147% under** the true silicon.
Zero today: `grep -rn stack_3d_height configs/hardware-config/` returns nothing, so every shipped card sits at 1. The knob is live at `config.py:3433, 3556-3557`.
The same substitution reaches `fws_atlas_export.py:479`, where the metric is named `enumerated_macro_silicon`, labelled \"silicon\", unit mm2.
This field is the gate at `dse:1408` (`max_silicon_mm2`), the frontier's area axis, and the Pareto/knee/trim objective (`dse:1658, 1949, 1977, 2104`).

**Refuted sub-claim.** The \"4x at h=4\" claim for the pool `area_share` at `fws_eval.py:3574` is vacuous: the numerator `per_unit.area_mm2` is 0.0 on every shipped card, so the ratio is 0 at all h. The denominator is still a footprint where a silicon area belongs, so the fix is right, but the ratio is latent, not 4x wrong.

**Fix.** Keep two named terms — `analog_macro_silicon_mm2 = slots × area_mm2_per_array` and `analog_macro_footprint_mm2 = slots × macro_footprint_mm2` — and sum only the silicon one into the total and the gate. Rename the atlas metric. Take the pool `area_share` denominator from `area_mm2_per_array`.

**OPTIMA cost.** None.

---

### F13 — `cim.chip.arrays_per_chip` carries two units, and the DSE's \"bound that cannot bind\" is 7.35% below the requirement
`tools/fws_qif_dse.py:758`, `cim_timing.py:5691`. Severity: major.

**Value now.** `bound = int(CimDeviceModel(probe_hw, inner).total_arrays())` = 544 arrays for Qwen, documented as an upper bound \"which no single chip can exceed\". The requirement at `layers_per_chip = 32` is **584 macro slots**.
**Correct value.** State a slot bound in slots. Running the probe with `arrays_per_chip = 0` lets `build_mapping` derive `macro_slots = max(1, peak_usage)` (`fws_mapping.py:1955-1956`).

**Arithmetic.** Driving the DSE's own axis and probe on the shipped Qwen frontier card: lpc=32 raises `MappingError: chip 0 needs 584 macro slots but the chip declares 544`; lpc=16/8/4/2/1 derive 292/146/73/37/19. 584/544 = **1.0735**. The shipped axis is `layers_per_chip: [32, 16, 8, 4, 2, 1]` (`fws_cim_qwen3_5_4b_frontier.yaml:312`), so the whole lpc=32 rung dies. `evaluate_point` catches it at `dse:1135-1143` and records a silent mapping refusal, so every `bank_depth × shared_chiplets × conv_placement` combination on the cheapest-silicon rung is lost.
The same field, other unit, on Granite: declared 484; `chip_array_usage()` = 540 arrays/chip; `validate_capacity()` raises \"chip 0 needs 540 analog arrays but cim.chip.arrays_per_chip = 484\". Read as slots, 484 is exactly right: `analog_macro_slots` 4840 = 10 × 484, `macros_holding_tiles` 4828, unowned 12. One field, simultaneously legal as slots and illegal as arrays, on a shipped card. 540 vs 484 = 11.57%.
Note: the mapped path never calls `validate_capacity`. Its only non-test caller is `tools/fws_cim_dse.py:243`, the old closed-form DSE.

**Fix.** Probe with `arrays_per_chip = 0` and delete the `total_arrays()` bound. Split the two meanings: keep `cim.chip.arrays_per_chip` as the array capacity, and use the existing `mapping.macros_per_chip` for slots.

**OPTIMA cost.** None.

---

### F14 — Two fields of one shipped candidate both name the digital silicon and disagree by up to 8.2x
`tools/fws_qif_dse.py:1055`. Severity: minor (fixed in the tree, live in the artifacts).

**Value now.** Shipped qwen c007: `silicon.digital_silicon_mm2` 4712.08559848 vs `derived_digital.digital_area_mm2_total` 5118.073128464001, three lines apart in one report. Shipped falcon c022: 174.80876499200002 vs 1608.3503712000002 (**+820.1%**).

**Arithmetic.** The delta per analog slot is constant: qwen c007 405.9875299840005 / 608 = 0.6677426480000008; qwen c021 389.9617064320001 / 584 = 0.6677426480000002; falcon c013 1439.9700439 / 3584 = 0.4017773560; falcon c022 1433.541606208 / 3568 = 0.40177735600000003. Every `conv_placement = shared_digital` row is 0 to 2e-13. One path priced a conv-free pool, the other a conv-inclusive one.
Current tree: both paths agree exactly — granite 1308.83847756, qwen 5905.1741171199965, falcon 1585.4903876159997.

**Fix.** Regenerate the curves. Keep the two computations independent, but assert equality to 1e-9 in `evaluate_candidate`, as `validate_fws_cim_vs_optima.py:2655-2688` already does for the per-chiplet area.

**OPTIMA cost.** None.

---

### F15 — A machine with zero attention ops still composes an attention fabric and a softmax engine
`cim_timing.py:4540`. Severity: minor.

**Value now.** Falcon-Mamba: 113.529082456 mm2 / 170.38125781 W per chiplet, × 4 = 454.116329824 mm2 / 681.52503124 W.
**Correct value.** 110.440672794 mm2 / 166.67012934 W per chiplet, × 4 = **441.762691176 mm2 / 666.68051736 W**; `total_silicon_mm2` 8814.299091368.

**Arithmetic.** `out = [self.sa_fabric_composition(), self.softmax_engine_composition()]` runs unconditionally; the scan engine is appended only when `resolved_vector_lanes()` is not None. The model already owns the zero-demand rule and applies it to one engine of three.
Falcon seed: replicas 1, num_arrays 2, ceil(32/32)×ceil(64/32) = 2 blocks per array × 2 = 4 → 4 × 0.767330073 + 0.010606661 = 3.079926953 mm2 / 3.6948471 W. Softmax at width 1: FP_ADD 1, FP_MULT 2, BF16_EXP 1, BF16_RECIP 1 = 0.008482709 mm2 / 0.01628137 W.
Billed for nothing: 4 × 3.088409662 = **12.353638648 mm2** (2.72% of the chiplet term, 0.78% of digital) and 4 × 3.71112847 = **14.84451388 W** (2.18%).
DAG census: chips 32/33/34/35 carry {ssm_scan 520/392/264/136} and no attention op. `derived_fabric` is None; `fabric_provenance` = \"declared-count\". The card comment states \"THE PURE-RECURRENCE CASE\".
`SIZING_METHODOLOGY.md:445` records the same defect as V6b at 12.35 mm2 / 14.84 W.

**Fix.** Gate fabric and softmax on attention demand, as the scan engine is already gated one line below.

**OPTIMA cost.** None.

---

### F16 — `mapping.decode_window`, an analysis knob, moves built silicon
`fws_eval.py:3757`. Severity: minor. M10 is binary.

**Value now.** Granite at window 2/3/4/5/6/8: `fabric_softmax_lanes` 1802 / 1800 / 1801 / 1802 / 1801 / 1801; `shared_digital_area_mm2_per_chiplet` 42.425394744 / 42.412388256 / 42.4188915 / 42.425394744 / 42.4188915 / 42.4188915.
**Correct value.** One built width at every window.

**Arithmetic.** 42.425394744 − 42.412388256 = 0.013006488 mm2 per chiplet × 10 = **0.13006488 mm2** of built silicon and 0.1891626 W. That is 0.00117% of the machine. Vector lanes (6509), fabric arrays (8) and the state store (13752 macros / 728.94387528 mm2) are all window-independent, so ADJ-11's store fix holds.
`SIZING_METHODOLOGY.md:118` states M10 verbatim: the decode window \"never size[s] built silicon\".
The measured beat also moves: 1.5318580e-05 (w=2) … 1.5330159e-05 (w=8) s, a spread of 1.1578947e-08 s = 7.558e-4 relative, against a `beat_basis` that asserts \"the reported beat [is] independent of how many beats the window lowered (P7.9)\".

**Fix.** Derive the fabric against per-beat demand, not against the probe slice of length `streams + decode_window + 1`. Qualify the `beat_basis` sentence.

**OPTIMA cost.** None.

---

### F17 — `state_bill.declared_tier_capacity_bytes` holds the sized store, not the declared tier — 6.82x apart
`tools/fws_qif_dse.py:1445`. Severity: minor.

**Value now.** Granite 457,666,560 B under that name.
**Correct value.** The declared on-chip tier is 67,108,864 B (64 MiB), published separately as `state_residency.declared_onchip_tier_bytes`.

**Arithmetic.** The field reads `residency[\"capacity_bytes\"]`, which `fws_eval.py:3456-3474` publishes twice, as `capacity_bytes` and `store_bytes` — the store the mapping builds. 13752 × 33280 = 457,666,560 exactly. 457666560 / 67108864 = **6.81976318359375**.
The shipped granite artifact carries 430,643,200 = 12940 × 33280 under the same name while `store_macros` = 12933 and area = 12933 × 0.05300639 = 685.53164187 mm2. 12940 macros would be 685.9026866 — 0.37104473 mm2 unpaid.
The same dict also carries `max_stage_state_bytes` and `measured_max_stage_state_bytes`, both 60,413,952.0, under a comment (`dse:1434-1437`) that invokes D21.

**Fix.** Rename to `store_capacity_bytes` and add the real `declared_onchip_tier_bytes`.

**OPTIMA cost.** None.

---

### F18 — The `shared_digital_chiplet` energy basis says \"UNCOVERED\" while the term carries 98.8% of the bill
`fws_eval.py:2754-2757`. Severity: minor.

**Value now.** Granite `shared_digital_chiplet` = 1.637212e10 pJ, coverage \"partial\", basis \"no law prices a shared-digital-chiplet op's energy … UNCOVERED.\"
**Correct value.** The term is fully derived: 1.1592e-04 s × 27.79329982 W × 1e12 = 3.221799e9 pJ, the reported `ssm_scan` energy to seven figures; 2.6132e-04 × 31.77746434 × 1e12 = 8.304087e9 pJ, the reported `attention_qk` energy.

**Arithmetic.** The string is a hard-coded literal, unlike `_fabric_energy_disclosure` (`fws_eval.py:750-783`), which *is* conditional on `has_synthesis_library()`. The label (\"partial\") and the sentence (\"UNCOVERED\") contradict each other in one record. A reader who trusts the basis discards 1.637212e10 of 1.657466e10 pJ = **98.78%** (Qwen: 99.13%).

**Fix.** Make the basis conditional on `has_synthesis_library()`, keeping both named gaps (toggle-rate power; idle leakage not charged).

**OPTIMA cost.** None.

---

### F19 — `bank_passes.basis` asserts an equality its own fields refute
`fws_eval.py:2098`. Severity: minor.

**Value now.** The basis ends \"… so the whole machine reads every stored weight once per beat and one token leaves\".
**Measured on the same run.** `macros_walked_exactly_once` 748 of 4828; `column_set_passes_charged` 3064.0 of `column_sets_occupied` 19304.0; `column_sets_never_read` 16240.0; `column_sets_read_more_than_once` 0.0.

**Arithmetic.** 748/4828 = 15.49%. 3064/19304 = 15.87%, a 6.3003x shortfall. 16240/19304 = **84.13%** of occupied column sets never fire. The measured fields are all correct and internally consistent (3064 + 16240 = 19304). Only the trailing clause of the prose is false. The docstring at `fws_eval.py:2012-2020` already states the correct reading and quotes the same figures, but does not reach the artifact.

**Fix.** Rewrite the trailing clause: Invariant W is about residency, not activity. Keep every field.

**OPTIMA cost.** None.

---

### F20 — `mapping.summary()[\"macro_slots\"]` sums analog weight-macro slots with digital engine slots
`fws_mapping.py:995`. Severity: minor.

**Value now.** Granite `macro_slots = 4860`, published in every report (`fws_eval.py:5045-5051` dumps the whole summary).
**Correct value.** `analog_macro_slots = 4840` and, if wanted, `digital_engine_slots = 20` as its own row.

**Arithmetic.** Chips 0..9 pool \"analog\", 484 slots each = 4840. Chips 10..19 pool \"digital\", 2 slots each = 20 (`engines_per_chiplet = max(1, num_arrays 2 × replicas 1)`). An analog slot is one CTT macro at 2.0204136 mm2 holding weights. A digital slot, by the code's own sentence at `fws_mapping.py:2071-2072`, \"hold[s] no tile, carr[ies] no device … and price[s] no time\".
No priced or plotted consumer reads this field. Every consumer reads `analog_macro_slots` or `chip.macro_slots`. No number moves.

**Refuted sub-claim.** The digital slot count is not \"stale\". `fws_mapping.py:2066-2075` explains that the derived fabric width does not exist at `build_mapping` time, and `DerivedFabricSizing` reconciles the two by name. Do not re-source it from the derived sizing.

**Fix.** Drop the mixed sum.

**OPTIMA cost.** None.

---

### F21 — Card comments state arithmetic the run they configure refutes
`configs/hardware-config/fws_cim_granite_tiny.yaml:59`, `fws_cim_mamba.yaml:33-39, 40, 211`. Severity: minor. No published number moves.

**Value now.** Granite: `arrays_per_chip: 484  # worst chip: 4 layers * 139 = 556 … 640 is deliberately held at the pre-D30 value so the 84 unowned slots are the frontier's own area axis`. The card does not declare 640. Falcon-Mamba carries the same comment byte for byte beside `arrays_per_chip: 112`, on a 64-layer, 2-layer-per-chip card; its mapping block also says \"dense saves 716 macros\".

**Arithmetic.** Measured: 10 chips × 484 slots = 4840; `macros_holding_tiles` 4828; **unowned = 12**, not 84. Split: 6 × 484 + 4 × 481 = 4828. `dedicated_macros` 5544 = 36 × 139 + 4 × 135; `macros_saved` 716 — 556/chip is the dedicated-law figure the dense packer never spends. The 12 unowned slots cost 12 × (2.0204136 + 0.032184033) = **24.631 mm2** = 0.222% of the total; the comment's 84 would be 172.418 mm2. Falcon `atlas_packing()`: macros 3584, dedicated 3712, **saved 128**, and 3584/32 = 112, so the value is right and only its justification is imported.

**Fix.** Re-derive each comment from the placement, or delete the arithmetic and point at `packing_summary`.

---

### Flagged, not promoted to a finding

**Mamba-2 conv state is 1536 B/layer/stream short.** `fws_eval.py:2847` computes `(d_inner × d_state + d_inner × (d_conv − 1)) × 2` = (3072×128 + 3072×3) × 2 = 804,864 B. In Mamba-2 the conv window spans `d_inner + 2 × n_groups × d_state` = 3328 channels, giving (393,216 + 9,984) × 2 = **806,400 B**. Granite consequence: stage 0 32,256,000 B → 970 macros (from 968); stage 1 66,135,040 B → 1988 (from 1986); total 13,772 macros (from 13,752), **+1.060 mm2** of state store. Two auditors raised it independently. It is left unpromoted because it turns on a model-architecture reading, and D23 puts accuracy out of scope. It is a *census* question, not an accuracy question, so it deserves an owner decision.

---

## 3. THE CENSUS-VS-LAW TABLE

Read: what the composition instantiates → what one pass therefore finishes → what the timing law spends → agree?

| Unit | Composition instantiates (block × count) | One pass therefore does | Timing law | Agree? |
|---|---|---|---|---|
| **SA attention fabric** (`cim_timing.py:2180`) | GEMMINI_SYS_ARRAY × `2 × num_arrays × replicas` = **16** at derived num_arrays 8; TRANSPOSER × 1 | 16 arrays available; `fold_group_split` gives QK only 4 and PV only 4 | `sa_cycles = ceil(m/32)·ceil(n/64)·(k+94) − 1` on a **disjoint** partition; the DAG runs QK then PV serially, and `fws_eval.py:281` gives the chiplet capacity 1 | **NO — F11.** 8 blocks are idle at all times. 4 shared arrays reproduce the shipped cycles exactly |
| **Softmax engine** (`cim_timing.py:2129`) at w=1800 | FP_COMP 1799, FP_ADD 3599, FP_MULT 3600, BF16_EXP 1800, **BF16_RECIP 1** | **One row per pass** — one reciprocal retires per cycle | `passes = rows × ceil(keys/width)`; `cycles = depth + passes − 1`. Granite rows 12, width 1800 → 12 passes → 31 cycles | **YES.** The element law now matches the count-1 reciprocal. The earlier row-width defect is fixed |
| **Softmax width choice** (`cim_timing.py:4797`) | 1800 lanes = 11.707819 mm2 / 17.031457 W = 27.6% of the chiplet | 12 cycles removed from a 10,351-cycle stage (0.30%) | Saturated branch takes `saturation_softmax_width = n` with no search | **NO — F10.** The census is sized to a target the stage cannot use |
| **Vector / scan engine** (`cim_timing.py:2073`) | N × {FP_MULT, FP_ADD, M_REG}; N = 6509 (Granite), 19080 (Qwen), 39033 (Falcon) | N scalar ops retire per cycle | `cycles = depth + ceil(work/N) − 1`. Granite 20 + ceil(1,972,224/6509) − 1 = 322; Qwen 20 + ceil(3,682,304/19080) − 1 = 212 | **YES.** `derive_vector_lanes` is the same law inverted; the round trip closes (636 cycles = 6.6947e-7 s) |
| **Per-macro digital pool** (`cim_timing.py:6608`) | FP_ADD `lanes + lanes·(n_slices−1)`, M_REG `lanes + conv_lanes`. Granite 21 + 21; Qwen 34 + 34 | `lanes` ADC results retire per cycle. Qwen: 2560 cols/80 ns = 3.2e10/s → ceil(3.2e10/0.95e9) = 34 | `ceil(results / lanes)` | **YES** on the census. **NO between twins**: the timeline block prices the pool at 0.0 mm2 (knob-based, undeclared) while the silicon census charges 0.032184033 mm2 × 4840 = 155.771 mm2. Printed `area_share` 0.0% against a true 1.593% |
| **Analog CIM macro** | `cols_adc × adc_mux` stored columns; `slice_cycles`; f_analog 1e8 | 384 × 1e8 / 2 = 1.92e10 results/s (Granite); 2560/80 ns = 3.2e10/s (Qwen) | `time = m × (column_sets × slice_cycles) / f_analog`; energy `M × E_vec × shots × total_sets / mux` | **YES.** Time uses `active_column_sets` (max, concurrent); energy uses `total_sets` (sum). Parallel time, summed energy — the correct pairing |
| **State store** (`cim_timing.py:2326`) | SRAM_MACRO_256x1040, 1040 × 256 / 8 = 33,280 B, 0.05300639 mm2, 3.241e-5 W | `ceil(state_bytes / 33280)` macros per stage, summed over stages, never pooled and ceiled once | Sized from `per_stream_state_bytes × D` per stage | **YES** on Granite/Qwen/Falcon. One census question open: Mamba-2 conv width (see \"Flagged\") |
| **Shared-digital chiplet** (`cim_timing.py:4540`) | fabric + softmax **unconditionally**, vector engine **gated on demand** | 10 chiplets each carry the full attention engine | The DAG places attention on 4 of 10 (Granite) and 0 of 4 (Falcon) | **NO — F9, F15.** Granite 143.974 mm2 / 190.665 W unreachable; Falcon 12.354 mm2 / 14.845 W for an operator never emitted |
| **Chiplet energy law** (`fws_eval.py:735`) | One power number = fabric + softmax = 31.77746434 W | Charged to qk, pv and softmax alike | The three ops are serial by the DAG's own disclosure | **NO — F1.** 42.5–48.7% of the published energy |
| **Chip-to-chip link** (`program/fws_build.py:639`) | p2p latency 1e-6 s + bytes / bandwidth | 3072 B at 100 GiB/s → 1.0286102294921875e-06 s | The law is right; `producer = deps[-1]` makes `src == dst` on 72 of 81 instances | **NO — F3.** The law is correct; the census of which edges reach it is not |
| **Chiplet count vs stage count** (`fws_mapping.py:993`) | Granite 10 chiplets, 10 stages; Qwen 8 chiplets, 32 stages | ADJ-14 groups 4 stages per chiplet on Qwen | `built_duty_*` divides by **stage** count | **NO — F7.** Exactly 4.00x under on Qwen |

---

## 4. BETTER HARDWARE FROM THE SAME BLOCKS

Every proposal uses only blocks the checked-in 22 nm library names. No block is invented. Baseline: **Granite, 65,275.716 tokens/s, 11,087.640 mm2 = 5.8873 tokens/s per mm2.**

| # | Proposal | Census change | Area | Power | Throughput | tokens/s per mm2 |
|---|---|---|---:|---:|---:|---:|
| **P1** | **Two chiplet SKUs (Granite).** SKU-A ×4 (stages 1/3/6/8) = full chiplet 42.412388256 mm2. SKU-B ×6 = scan only, 6509 × {FP_MULT, FP_ADD, M_REG} = 18.416681762 mm2 | Remove fabric + softmax from 6 chiplets | **−143.974 mm2** (424.124 → 280.150) | −190.665 W | **0.000%** | 5.9647 (**+1.316%**) |
| **P2** | **Shared SA array pool.** GEMMINI_SYS_ARRAY 16 → **8** per chiplet, TRANSPOSER 1 unchanged | 4 arrays serve QK, then the same 4 serve PV | **−61.386 mm2** (12.287888 → 6.149247 per chiplet) | −73.674 W | **0.000%** — verified bit-identical (Granite qk 6437 / pv 3783; Qwen qk 10149 / pv 7563) | 5.9200 (**+0.556%**) |
| **P3** | **Halve the softmax width.** FP_COMP 899, FP_ADD 1799, FP_MULT 1800, BF16_EXP 900, BF16_RECIP 1 | Softmax 31 → 43 cycles per stage | **−58.529 mm2** (11.707819 → 5.854899 per chiplet) | −85.123 W | **−0.0824%** (65,275.72 → 65,221.94) | 5.9136 (**+0.446%**) |
| **P4** | **Scan-only chiplet (Falcon-Mamba).** Drop the 4-array seed fabric and the width-1 softmax | 39,033 × {FP_MULT, FP_ADD, M_REG} only | **−12.354 mm2** (113.529082 → 110.440673 per chiplet) | −14.845 W | **0.000%** | +0.140% of that machine's total silicon |
| **P1+P2** | Two SKUs, each with a shared array pool | SKU-A ×4 at 36.273748 mm2, SKU-B ×6 at 18.416682 mm2 | **−168.529 mm2** | −264.339 W | 0.000% | 5.9781 (**+1.543%**) |
| **P1+P2+P3** | All three, together | SKU-A ×4 at 30.420828 mm2, SKU-B ×6 at 18.416682 mm2 = 232.183 mm2 | **−191.940 mm2** (−45.3% of the chiplet term) | −349.462 W | −0.0824% | **5.9860 (+1.678%)** |

**Ranking by throughput per mm2 gained: P1 > P2 > P3 > P4.** P1, P2 and P4 are free — they remove silicon that executes no op on the measured timeline. P3 is the only proposal that costs throughput, at 710 mm2 of saving per 1% of tokens/s lost.

**On the reciprocal.** The N3 briefing's shape — instantiate several BF16_RECIPs and finish several rows per pass — does **not** pay on this workload. `_softmax_cycles_at` finishes one row per pass because of the per-row FP_COMP max tree and FP_ADD sum tree. Granite has 12 rows against 24 passes at width 900, so extra reciprocals (0.003472065 mm2 each) remove no pass. Here the **width**, not the reciprocal, is over-bought. The census-vs-law defect that the briefing describes is real in general; on this workload it points the other way.

**Not proposed.** The column split — partitioning one fold's `n` across arrays — is refused by name at `fws_eval.py:4189-4196` for want of a broadcast/merge reference. That is a stated design decision under D28, not an arithmetic defect, and no proposal here depends on overturning it.

---

## 5. WHAT IS RIGHT

188 of 246 numbers reproduced exactly. The following are safe to build on.

**Analog timing and energy.** `time = m × (column_sets × slice_cycles) / f_analog`: ViT 64 × 8 / 1e8 = 5.12e-06 s, exact. Per-op energy `m × E_vec × shots × total_sets/mux`: ViT 64 × 9534.9389 × 2 × 1 = 1,220,472.18 pJ, exact; Granite per-column-set 6865.16 pJ, exact. Granite `analog_arrays` total 199,830,961.08 pJ closes against a hand-summed block census. The parallel-time / summed-energy pairing is correct.

**Systolic and softmax cycle laws.** ViT QK 2747, PV 4471, softmax 65555; Granite QK 6437 (+96 fill), PV 3783, softmax 31; Qwen QK 10149, PV 7563, softmax 35. Every one recomputed by hand. `fill_drain_penalty_cycles = 3 × rows = 96`, charged once per call.

**The vector-lane derivation round trip.** Granite: 4 calls × (20 + ceil(1,972,224/6509) − 1) = 1288 cycles ≤ floor(1.36e-06 × 0.95e9) = 1291, and 6508 lanes gives 1292 > 1291, so 6509 is minimal. The derived width reproduces the timeline's busy time to 6.6947e-7 s.

**Invariant W and the packing chain.** Granite real cells 6,784,057,344 rebuilt layer by layer; `cell_floor_macros` ceil(6,784,057,344/2,359,296) = 2876; committed 4828 × 2,359,296 = 11,390,681,088; real + remainder + tail closes exactly; `waste_pct` 40.44203949185308. ViT: 386 placed macros against a cell floor of exactly 386. Falcon: 6,727,663,616 cells, 2852 floor, 20.4365079% waste.

**The state-store sizing chain, end to end, on all three models.** Bytes → `ceil(bytes/33280)` per stage → macros → mm2 → W. Granite 968 × 6 + 1986 × 4 = 13,752 macros = 728.94387528 mm2. Qwen 1032 × 24 + 8066 × 8 = 89,296 = 4733.25860144 mm2. Falcon 599 × 32 = 19,168 = 1016.02648352 mm2. KV bytes `2 × kv_heads/tp × head_dim × ctx × act_bytes` and the Qwen V2a delta-rule state (1,073,152 B) both check out. The store is window-independent.

**Every block composition.** SA fabric 12.287887829 mm2, softmax 11.707818665 mm2 at w=1800, vector 0.002829418 mm2 per lane triple, per-macro pool 0.032184033 mm2 — each recomposed from the checked-in 22 nm library, each exact to nine decimal places.

**D21 round trips on the current tree.** `tools/fws_qif_dse._silicon.digital_silicon_mm2` == `fws_eval.digital_area_mm2_total`: Granite delta 0.0, Qwen delta 0.0, Falcon delta 4.5e-13 (float association only). The four-term silicon identity closes on every point. The Candidates table's three-column decomposition closes.

**The beat and the rate identity.** `tokens_per_s = 1/beat` and `per_stream = 1/(D × beat)` hold on all three machines. The beat is the median of the settled tail, and it is window-independent on Falcon (windows 3 and 34 give the identical beat and all 35 inter-exit intervals equal). `beats = streams + decode_window + 1` and the exit/interval indexing check out on all three. Little's law is satisfied on Granite (0.8712) and Qwen (0.8859).

**The ViT closed-form path, end to end.** Hand-summed critical chain 3.439108881579e-03 s against the model's value at 1.3e-16 relative error, then 1/that = 290.7729980159525 requests/s. 32 layers × a 1.073097e-04 s per-layer chain, plus patch embed and head.

**Utilization denominators and disclosure.** `mean_occupancy` for analog macros (5.939504313906364e-04) and shared digital (0.26158948562011963) both reconstruct from busy/devices/makespan, with idle devices inside the mean as the basis claims. The link row uses a different, and named, convention.

**Unit chains.** `bytes × 8 × J/bit × 1e12` = pJ. `floor(budget_s × clock_hz)` = cycles, floored in the safe direction. `size(B)/period(s)` = B/s. The GiB memory verdict is a self-consistent binary reading throughout.

---

## 6. ORDERED FIX LIST

Cheapest correct first. **[S]** moves a shipped number. **[V]** moves a validator or test check.

| # | Fix | File:line | Effort | Moves |
|---:|---|---|---|---|
| 1 | Print all four silicon terms in the headline line | `tools/fws_qif_dse.py:2397-2399` | 1 line | **[S]** report prose only; closes a 7.98–51.14% gap |
| 2 | Make the shared-digital energy basis conditional on `has_synthesis_library()` | `fws_eval.py:2754` | 1 branch | **[S]** basis text |
| 3 | Rewrite the `bank_passes` trailing clause to the docstring's reading | `fws_eval.py:2098` | prose | **[S]** basis text |
| 4 | Rename `declared_tier_capacity_bytes` → `store_capacity_bytes`; add the real declared tier | `tools/fws_qif_dse.py:1445` | 2 lines | **[S]** field name |
| 5 | Drop the mixed `macro_slots` sum | `fws_mapping.py:995` | 1 line | **[S]** one published field |
| 6 | Re-derive the card comments from the placement | `fws_cim_granite_tiny.yaml:59`, `fws_cim_mamba.yaml:33-39, 211` | comments | none |
| 7 | **Split attention energy by block**: `sa_fabric` for qk/pv, `softmax` for softmax | `fws_eval.py:735, 807` | 3 cases | **[S] LARGE** — Granite total energy ÷1.7393, Qwen ÷1.9480 |
| 8 | Publish `energy_per_beat_pj` over the settled tail, per component; rename the window sum | `fws_eval.py:1246`, `tools/fws_qif_dse.py:1359/2436/2469` | new field | **[S] LARGE** — the energy column becomes 1.744995e9 pJ/token and stops moving with a knob |
| 9 | Compute `built_duty_*` over `_stage_chiplet_groups` and divide by built chiplets | `tools/fws_qif_dse.py:830-831, 1292` | 5 lines | **[S]** Qwen duty ×4.000 |
| 10 | Split `analog_macro_silicon_mm2` from `analog_macro_footprint_mm2`; sum silicon into the total and the gate; rename the atlas metric | `tools/fws_qif_dse.py:880-882, 1408`, `fws_atlas_export.py:479`, `fws_eval.py:3574` | ~10 lines | **[S]** latent — no shipped card sets `stack_3d_height`. Prevents a 44.10% under-count |
| 11 | Probe with `arrays_per_chip = 0`; split the field's two units | `tools/fws_qif_dse.py:758`, `cim_timing.py:5691` | ~10 lines | **[S]** restores the Qwen `layers_per_chip = 32` frontier rung |
| 12 | Gate fabric and softmax on attention demand, as the scan engine already is | `cim_timing.py:4540` | 1 branch | **[S]** Falcon −12.354 mm2 / −14.845 W |
| 13 | Compose per chiplet against its own stage group (two SKUs) | `fws_eval.py:4747-4754, 3340`; `tools/fws_qif_dse.py:884` | ~20 lines | **[S]** Granite −143.974 mm2 / −190.665 W; total silicon 11087.640 → 10943.666 |
| 14 | Choose the saturated softmax width against a reachable target | `cim_timing.py:4797` | ~10 lines | **[S]** −58.529 mm2 / −85.123 W; −0.0824% tokens/s |
| 15 | Price QK and PV on the whole array pool in the derived path only | `cim_timing.py:2696, 2781, 3382` | ~15 lines | **[S]** −61.386 mm2 / −73.674 W on Granite. **Must not touch the closed form** — at num_arrays 2 it moves T1/T2/T3 |
| 16 | Publish `stage_cycles = max(qk, pv+fill, softmax)`; rename `fabric_total_cycles`; delete the stale SOFTMAX_UNIT_UNRESOLVED text | `cim_timing.py:3273-3305, 3391`, `inference_timing.py:1252`, `fws_bridge.py:725` | ~20 lines | **[V]** turns the red bridge gate green. The OPTIMA divergence is already paid by the element law |
| 17 | **Give the boundary transfer the real data edge**: pick `producer` from the source chip, attach the hold edge in addition | `program/fws_build.py:639` | ~10 lines | **[S] LARGE** — 72/81 Granite and 589/620 Falcon links become priced; the beat, the latency and the Falcon headline all move |
| 18 | Enforce `beat >= per_token_latency / D`; refuse or correct a violation | `fws_eval.py:2310` | ~10 lines | **[S] LARGE** — depends on 17. Falcon 1,240,209 tokens/s is capped at 538,345 on the current latency |
| 19 | Derive the fabric against per-beat demand, not the probe slice; qualify `beat_basis` | `fws_eval.py:3757` | ~10 lines | **[S]** 0.130 mm2 (0.00117%). M10 is binary |
| 20 | Add a 1e-9 equality gate between `silicon.digital_silicon_mm2` and `derived_digital.digital_area_mm2_total` | `tools/fws_qif_dse.py` `evaluate_candidate` | 3 lines | **[V]** new gate |
| 21 | Widen `verify_selected` to cover `total_silicon_mm2`, `state_sram_macros`, `macro_pool_silicon_mm2` | `tools/fws_qif_dse.py:2766` | ~10 lines | **[V]** the budget-gate quantity is currently unchecked |
| 22 | Add ViT and falcon to `CURVES`; regenerate all four artifacts **last**, after 1–21 | `tests/test_qif_frontier.py:651`, `docs/qif/dse/*` | regeneration | **[S] + [V]** — ViT 789.84 → 290.77 requests/s; Qwen silicon +14.68% |
| 23 | Owner decision on the Mamba-2 conv-state width (3072 vs 3328 channels) | `fws_eval.py:2847` | 1 line | **[S]** Granite +20 SRAM macros, +1.060 mm2 |

**Only fixes 15 and 16 touch OPTIMA agreement.** Fix 16's cost is already spent by the installed element law: 20 of 508 checks — every recorded T1/T2/T3 period, fps and bottleneck, and all five bridge attention-fold identities. Fix 15 costs nothing if it is confined to the derived path, and moves T1/T2/T3 if it is not. Every other fix on this list leaves the OPTIMA reference untouched.",
    "checked": 246,
    "findings": 25,
    "refuted": 7
  },
  "workflowProgress": [
    {
      "type": "workflow_phase",
      "index": 1,
      "title": "Trace"
    },
    {
      "type": "workflow_phase",
      "index": 2,
      "title": "Verify"
    },
    {
      "type": "workflow_phase",
      "index": 3,
      "title": "Report"
    },
    {
      "type": "workflow_agent",
      "index": 1,
      "label": "numbers:census-vs-law",
      "phaseIndex": 1,
      "phaseTitle": "Trace",
      "agentId": "a9a6902c7b4b46c53",
      "model": "claude-opus-5[1m]",
      "state": "done",
      "startedAt": 1787803812660,
      "queuedAt": 1787803810397,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "You are auditing the NUMBERS of a research performance model for a chiplet
system: analog compute-in-memory (CIM) weight-stationary macros plus shared
digital chiplets, running transformer / state-space / hybrid inference in DECODE.

REPO ROOT: /app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM
Work ONLY inside that directory. Shared NFS mount: never run an unbounded
recurs…",
      "lastProgressAt": 1787804721209,
      "tokens": 144162,
      "toolCalls": 60,
      "durationMs": 908548,
      "resultPreview": "{\"findings\":[{\"title\":\"SA fabric census pays for a_qk + a_pv arrays, but the DAG runs QK and PV strictly in series — half the composed GEMMINI blocks can never be busy\",\"file\":\"cim_timing.py\",\"line\":2180,\"standard\":\"N3\",\"value_now\":\"16 GEMMINI_SYS_ARRAY per chiplet = 12.287888 mm2 / 14.7460 W (num_arrays derived = 8, replicas 1, 2 blocks per 32x64 array)\",\"value_correct\":\"8 GEMMINI_SYS_ARRAY per c…"
    },
    {
      "type": "workflow_agent",
      "index": 2,
      "label": "numbers:units",
      "phaseIndex": 1,
      "phaseTitle": "Trace",
      "agentId": "af32bf4f05a5d5a22",
      "model": "claude-opus-5[1m]",
      "state": "done",
      "startedAt": 1787803813248,
      "queuedAt": 1787803810398,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "You are auditing the NUMBERS of a research performance model for a chiplet
system: analog compute-in-memory (CIM) weight-stationary macros plus shared
digital chiplets, running transformer / state-space / hybrid inference in DECODE.

REPO ROOT: /app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM
Work ONLY inside that directory. Shared NFS mount: never run an unbounded
recurs…",
      "lastProgressAt": 1787805207140,
      "tokens": 196579,
      "toolCalls": 111,
      "durationMs": 1393087,
      "resultPreview": "{\"findings\":[{\"title\":\"total_energy_pj is a per-WINDOW total whose window length is set by an analysis knob and by the swept axis, and it is published as a per-candidate figure of merit beside a rate\",\"file\":\"tools/fws_qif_dse.py\",\"line\":1359,\"standard\":\"N2\",\"value_now\":\"Granite, shipped card: 1.65747e10 pJ, printed as `- energy 1.657e+10 pJ over the timeline` (dse:2436) and as the `energy pJ` col…"
    },
    {
      "type": "workflow_agent",
      "index": 3,
      "label": "numbers:scale-counts",
      "phaseIndex": 1,
      "phaseTitle": "Trace",
      "agentId": "a2e2a0f78b345edb0",
      "model": "claude-opus-5[1m]",
      "state": "done",
      "startedAt": 1787803812786,
      "queuedAt": 1787803810398,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "You are auditing the NUMBERS of a research performance model for a chiplet
system: analog compute-in-memory (CIM) weight-stationary macros plus shared
digital chiplets, running transformer / state-space / hybrid inference in DECODE.

REPO ROOT: /app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM
Work ONLY inside that directory. Shared NFS mount: never run an unbounded
recurs…",
      "lastProgressAt": 1787804903663,
      "tokens": 179765,
      "toolCalls": 88,
      "durationMs": 1089611,
      "resultPreview": "{\"findings\":[{\"title\":\"The attention fabric + softmax engine are charged to EVERY shared digital chiplet, including the 6 of 10 that can never receive an attention op (Granite)\",\"file\":\"fws_eval.py\",\"line\":4753,\"standard\":\"N4\",\"value_now\":\"shared_digital_chiplet.area_mm2_total = 424.123882560 mm2, power_W_total = 595.707641600 W, digital_area_mm2_total = 1308.838477560 mm2, digital_power_W_total =…"
    },
    {
      "type": "workflow_agent",
      "index": 4,
      "label": "numbers:roundtrip",
      "phaseIndex": 1,
      "phaseTitle": "Trace",
      "agentId": "af97b9fe377753ec2",
      "model": "claude-opus-5[1m]",
      "state": "done",
      "startedAt": 1787803812936,
      "queuedAt": 1787803810398,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "You are auditing the NUMBERS of a research performance model for a chiplet
system: analog compute-in-memory (CIM) weight-stationary macros plus shared
digital chiplets, running transformer / state-space / hybrid inference in DECODE.

REPO ROOT: /app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM
Work ONLY inside that directory. Shared NFS mount: never run an unbounded
recurs…",
      "lastProgressAt": 1787809022272,
      "tokens": 274072,
      "toolCalls": 148,
      "durationMs": 5208219,
      "resultPreview": "{\"findings\":[{\"title\":\"The closed-form ↔ DAG bridge gate fails on all five points: `fabric_total_cycles` omits the softmax the DAG's fold includes (13.6x–27.5x)\",\"file\":\"validation_scripts/validate_fws_cim_vs_optima.py\",\"line\":1256,\"standard\":\"N5\",\"value_now\":\"bridge attention fold identity: T1 4567 vs 65555, T2 4567 vs 65555, T3 45315 vs 614675, Llama7B 25837599 vs 411041811, MoE 536735 vs 147456…"
    },
    {
      "type": "workflow_agent",
      "index": 5,
      "label": "numbers:arithmetic",
      "phaseIndex": 1,
      "phaseTitle": "Trace",
      "agentId": "a3a4ff86fb23d073a",
      "model": "claude-opus-5[1m]",
      "state": "done",
      "startedAt": 1787803813184,
      "queuedAt": 1787803810398,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "You are auditing the NUMBERS of a research performance model for a chiplet
system: analog compute-in-memory (CIM) weight-stationary macros plus shared
digital chiplets, running transformer / state-space / hybrid inference in DECODE.

REPO ROOT: /app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM
Work ONLY inside that directory. Shared NFS mount: never run an unbounded
recurs…",
      "lastProgressAt": 1787805958711,
      "tokens": 289576,
      "toolCalls": 115,
      "durationMs": 2144658,
      "resultPreview": "{\"findings\":[{\"title\":\"_chip_boundary keeps only the LAST dep, so every steady-state chip handoff is priced 0 s and the traversal's own data edge is dropped\",\"file\":\"program/fws_build.py\",\"line\":639,\"standard\":\"N1\",\"value_now\":\"Granite: 72 of 81 chip-boundary handoffs cost 0.0 s. Falcon-Mamba: 589 of 620 cost 0.0 s. Qwen: same pattern. Only traversal 0's boundaries carry a link cost.\",\"value_corre…"
    },
    {
      "type": "workflow_agent",
      "index": 6,
      "label": "recheck:census-vs-law",
      "phaseIndex": 2,
      "phaseTitle": "Verify",
      "agentId": "a8ad0559ad4ca4bdf",
      "model": "claude-opus-5[1m]",
      "state": "done",
      "startedAt": 1787804727612,
      "queuedAt": 1787804722576,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "You are auditing the NUMBERS of a research performance model for a chiplet
system: analog compute-in-memory (CIM) weight-stationary macros plus shared
digital chiplets, running transformer / state-space / hybrid inference in DECODE.

REPO ROOT: /app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM
Work ONLY inside that directory. Shared NFS mount: never run an unbounded
recurs…",
      "lastProgressAt": 1787805432829,
      "tokens": 115719,
      "toolCalls": 51,
      "durationMs": 705216,
      "resultPreview": "{\"findings\":[{\"title\":\"SA fabric census pays for a_qk + a_pv arrays, but the DAG runs QK and PV strictly in series — half the composed GEMMINI blocks can never be busy\",\"file\":\"cim_timing.py\",\"line\":2180,\"standard\":\"N3\",\"value_now\":\"16 GEMMINI_SYS_ARRAY per chiplet = 12.287888 mm2 / 14.7460 W (num_arrays derived = 8, replicas 1, 2 blocks per 32x64 array)\",\"value_correct\":\"8 GEMMINI_SYS_ARRAY per c…"
    },
    {
      "type": "workflow_agent",
      "index": 7,
      "label": "recheck:scale-counts",
      "phaseIndex": 2,
      "phaseTitle": "Verify",
      "agentId": "ae2081575727c90ef",
      "model": "claude-opus-5[1m]",
      "state": "done",
      "startedAt": 1787804908677,
      "queuedAt": 1787804906190,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "You are auditing the NUMBERS of a research performance model for a chiplet
system: analog compute-in-memory (CIM) weight-stationary macros plus shared
digital chiplets, running transformer / state-space / hybrid inference in DECODE.

REPO ROOT: /app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM
Work ONLY inside that directory. Shared NFS mount: never run an unbounded
recurs…",
      "lastProgressAt": 1787805659610,
      "tokens": 102498,
      "toolCalls": 43,
      "durationMs": 750933,
      "resultPreview": "{\"findings\":[{\"title\":\"Every attention op is charged the FULL attention-engine power (systolic fabric + softmax pipeline), so the softmax lanes are billed while QK/PV run and the fabric is billed while softmax runs\",\"file\":\"fws_eval.py\",\"line\":735,\"standard\":\"N4\",\"value_now\":\"Granite: shared_digital_chiplet energy = 1.63721169e+10 pJ, total_pj = 1.65746594e+10. Qwen3.5-4B: shared_digital_chiplet e…"
    },
    {
      "type": "workflow_agent",
      "index": 8,
      "label": "recheck:units",
      "phaseIndex": 2,
      "phaseTitle": "Verify",
      "agentId": "ae2060908837db116",
      "model": "claude-opus-5[1m]",
      "state": "done",
      "startedAt": 1787805210192,
      "queuedAt": 1787805208685,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "You are auditing the NUMBERS of a research performance model for a chiplet
system: analog compute-in-memory (CIM) weight-stationary macros plus shared
digital chiplets, running transformer / state-space / hybrid inference in DECODE.

REPO ROOT: /app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM
Work ONLY inside that directory. Shared NFS mount: never run an unbounded
recurs…",
      "lastProgressAt": 1787806086788,
      "tokens": 113237,
      "toolCalls": 63,
      "durationMs": 876595,
      "resultPreview": "{\"findings\":[{\"title\":\"total_energy_pj is a per-WINDOW total whose window length is set by an analysis knob and by the swept axis, and it is published as a per-candidate figure of merit beside a rate\",\"file\":\"tools/fws_qif_dse.py\",\"line\":1359,\"standard\":\"N2\",\"value_now\":\"Granite, shipped card: 1.65747e10 pJ, printed as `- energy 1.657e+10 pJ over the timeline` (dse:2436) and as the `energy pJ` col…"
    },
    {
      "type": "workflow_agent",
      "index": 9,
      "label": "recheck:arithmetic",
      "phaseIndex": 2,
      "phaseTitle": "Verify",
      "agentId": "a85d9f3117e806b3b",
      "model": "claude-opus-5[1m]",
      "state": "done",
      "startedAt": 1787805962651,
      "queuedAt": 1787805960874,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "You are auditing the NUMBERS of a research performance model for a chiplet
system: analog compute-in-memory (CIM) weight-stationary macros plus shared
digital chiplets, running transformer / state-space / hybrid inference in DECODE.

REPO ROOT: /app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM
Work ONLY inside that directory. Shared NFS mount: never run an unbounded
recurs…",
      "lastProgressAt": 1787806976970,
      "tokens": 125820,
      "toolCalls": 72,
      "durationMs": 1014318,
      "resultPreview": "{\"findings\":[{\"title\":\"_chip_boundary keeps only the LAST dep, so every steady-state chip handoff is priced 0 s and the traversal's own data edge is dropped\",\"file\":\"program/fws_build.py\",\"line\":639,\"standard\":\"N1\",\"value_now\":\"Granite: 72 of 81 chip-boundary handoffs cost 0.0 s. Falcon-Mamba: 589 of 620 cost 0.0 s. Qwen: same pattern. Only traversal 0's boundaries carry a link cost.\",\"value_corre…"
    },
    {
      "type": "workflow_agent",
      "index": 10,
      "label": "recheck:roundtrip",
      "phaseIndex": 2,
      "phaseTitle": "Verify",
      "agentId": "aedddfdc2e525d059",
      "model": "claude-opus-5[1m]",
      "state": "done",
      "startedAt": 1787809029186,
      "queuedAt": 1787809024541,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "You are auditing the NUMBERS of a research performance model for a chiplet
system: analog compute-in-memory (CIM) weight-stationary macros plus shared
digital chiplets, running transformer / state-space / hybrid inference in DECODE.

REPO ROOT: /app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM
Work ONLY inside that directory. Shared NFS mount: never run an unbounded
recurs…",
      "lastProgressAt": 1787809835893,
      "tokens": 120057,
      "toolCalls": 69,
      "durationMs": 806707,
      "resultPreview": "{\"findings\":[{\"title\":\"The closed-form ↔ DAG bridge gate is red: `fabric_total_cycles` omits the softmax the DAG's fold includes (13.6x–27.5x)\",\"file\":\"validation_scripts/validate_fws_cim_vs_optima.py\",\"line\":1256,\"standard\":\"N5\",\"value_now\":\"bridge attention fold identity, closed form vs DAG: T1 4567 vs 65555, T3 45315 vs 614675, MoE 536735 vs 14745619\",\"value_correct\":\"the two paths must return …"
    },
    {
      "type": "workflow_agent",
      "index": 11,
      "label": "numbers-report",
      "phaseIndex": 3,
      "phaseTitle": "Report",
      "agentId": "acb128d961704462f",
      "model": "claude-opus-5[1m]",
      "state": "done",
      "startedAt": 1787809842318,
      "queuedAt": 1787809837789,
      "attempt": 1,
      "lastToolName": "Bash",
      "lastToolSummary": "python3 - <<'EOF' # energy g_tot=16574659371.20638; g_chip=…",
      "promptPreview": "You are auditing the NUMBERS of a research performance model for a chiplet
system: analog compute-in-memory (CIM) weight-stationary macros plus shared
digital chiplets, running transformer / state-space / hybrid inference in DECODE.

REPO ROOT: /app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM
Work ONLY inside that directory. Shared NFS mount: never run an unbounded
recurs…",
      "lastProgressAt": 1787810312940,
      "tokens": 131211,
      "toolCalls": 2,
      "durationMs": 470621,
      "resultPreview": "# NUMBERS AUDIT — QIF FWS-CIM PERFORMANCE MODEL

Scope: every published number of the mapped (ADJ-10) decode path, the closed-form bridge, and the DSE artifacts. Repo root `/app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM`. D23 applies: no accuracy findings.

---

## 1. THE HEADLINE

Five auditors recomputed **246 published numbers**. **188 reproduced exactly** (76.4%). *…"
    }
  ],
  "totalTokens": 1792696,
  "totalToolCalls": 822
}