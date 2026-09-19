# FWS checkpoint and mapping runs — 2026-09-18

The existing August–September working tree was preserved in local commit
`0176fa6` on `fws_cim_dev` (98 files). No production mapping or evaluator code
was changed for these diagnostics. Nothing was pushed.

The current toy is runnable and checks its schedules. It does **not** establish
a useful real-model frontier: it omits state/digital area, uses a restricted
scheduling policy, and reports a waste measure that excludes unassigned tail
slots. We found a concrete schedule excluded by that policy which improves
throughput at unchanged weight placement and unchanged required concurrency.
That is a reason to test the schedule restriction, not proof that FWS is a
dead end or that the relaxed schedule is a complete hardware implementation.

## Fresh runs

All Python runs used the repository virtualenv through
`/tmp/rapid-fws-audit-20260918/venv/bin/python`. Tests disabled bytecode and
pytest cache writes; scratch outputs are separate from historical reports.

| Run | Result |
|---|---|
| Existing six-point toy | All six schedules independently checked and exported. |
| Four targeted suites: slot mapper, bundle runner, SA geometry, link relaxation | 179 passed, 8 failed in 188 s. The slot mapper's 34 tests all passed. |
| Existing ViT-Large smoke grid | 4/8 valid in 3.5 s. The other four request pp_util=0.5 under lockstep, which the code refuses. |
| Mamba2-2.7B, one existing-grid point | Restricted the grid to Z=128, Y=2048, pp_util=1.0 without changing model dimensions. Did not finish within a 240-second timeout (exit 124); no result reported. Confirmed the process stopped. |
| Toy sensitivity checks | 72 combinations of geometry, chip capacity and inter-bundle delay passed the existing document checker. |
| Diagnostic flexible-phase schedules | Preserved physical weights/macros/slots; independently checked eight overlapping tokens, unique macro occupancy, exact slot coverage, and chain/seam dependencies. |

ViT-L's non-dominated points were **236.193 requests/s at 232.294 mm²** and
**438.521 requests/s at 254.335 mm²** (Y=2048, Z=64 and 128 respectively).
These come from the existing lockstep/stub path, **not the new slot mapper**,
and are execution checks, not validated design claims.

## Toy results

L is end-to-end token latency. D is the number of resident sequences needed
by the periodic schedule. Per-user rate accounts for integer D and admission
phase, rather than using 1/L without rounding.

| Point | Mux | Chips | L (ns) | D | Aggregate tokens/s | Per-user tokens/s | Analog area (mm²) |
|---|---:|---:|---:|---:|---:|---:|---:|
| p04 | 4 | 2 | 1,020 | 13 | 12,500,000 | 961,538 | 15.392 |
| p00 | 8 | 7 | 4,460 | 28 | 6,250,000 | 223,214 | 13.427 |
| p01 | 16 | 3 | 2,900 | 10 | 3,125,000 | 312,500 | 8.533 |
| p02 | 32 | 2 | 2,460 | 4 | 1,562,500 | 390,625 | 9.394 |
| p03 | 64 | 1 | 1,180 | 1 | 781,250 | 781,250 | 8.402 |
| p05 | 128 | 2 | 6,000 | 3 | 390,625 | 130,208 | 8.751 |

These are analog-only debug machines, with six macros per chip rather than
the production die-capacity rule. Their chip counts and seam penalties must
not be extrapolated to real models.

## Specific limitations exposed

1. **Zero reported waste is not full physical utilization.** p03 reports 0%
   waste, but uses 252 of 384 slot-layers: **34.375% of column capacity is
   unused**, and mean ADC duty is 65.625%. The code counts ragged columns and
   stranded slots as waste but excludes the free slots at the final tail.
   Across the six points, all-unused capacity is 12.5–34.375%. This distinction
   matters when interpreting a density claim; it is not hidden data, since
   the free-slot and utilization fields exist.

2. **Slot address = firing phase is an extra restriction.** The mapper and
   checker require physical slot j to fire only at phase j of a mux-tick
   period. In p03 there are just 42 occupied slots per macro, despite mux=64.
   The diagnostic keeps all weights in the exact same physical slots but
   permits a different periodic slot-selection table. With **period 59 ticks**
   it retains **L=59 ticks and D=1**, while throughput rises from **781,250 to
   847,458 tokens/s (+8.47%)**. A 42-tick period reaches **1,190,476 tokens/s
   (+52.38%)**, but L grows to 110 ticks and D to 3; per-user rate then falls
   to 396,825 tokens/s. Both schedules obey the toy's one-firing-per-macro,
   shared-input, weight-coverage and dependency constraints. They deliberately
   do not obey D39's slot=phase rule and are not passed off as existing-mapper
   output. An arbitrary periodic slot-selection table is the hardware
   capability assumed by the probe; its control cost is not modeled.

   This is **not** a counterexample to the service bound for a fully occupied
   dense macro. It demonstrates that nominal storage mux is not forced to be
   the achieved initiation interval for this partially occupied placement.

3. **The important state trade is absent from the plotted area.** p03's faster
   42-tick schedule needs three sequences instead of one. Whether that is a
   win depends on per-sequence state size and SRAM cost, which the toy omits.
   At fixed geometry/capacity, increasing the simplified inter-bundle gap from
   2 to 32 ticks changes p03's D from **1 to 7**, while its plotted throughput
   and analog area stay identical. These delay values are sensitivity inputs,
   not claimed digital timings. A throughput-versus-analog-area frontier cannot
   assess that change.

4. **The tiny chip boundary is a major experimental variable.** At mux 32,
   combining two six-macro chips into one twelve-macro chip (same twelve macros,
   same modeled analog area) reduces L from **123 to 39 ticks** and D from
   **4 to 2**. This includes scheduling changes as well as removal of a seam.
   Packaging and IO costs are absent. The toy's seam story is therefore highly
   dependent on its artificial capacity limit.

5. **The real-model code is not cleanly integrated.** Of the eight test
   failures, four use retired sweep axes; one expects old stage indices; one
   expects attention to reach analog pace but receives `saturated_geometry`;
   two link-relaxation assertions disagree with observed behavior (intra-chip
   hops are also zeroed, and timeline memory high-water marks change). These
   are observations, not a claim that all failures are stale tests. Full logs
   are retained. Group-level sizing and D39 real-model state integration remain
   unfinished, as recorded at the checkpoint.

## Reproduction and evidence

`tools/fws_mapping_checkpoint_probe.py --repo <rewrite-root> --out <json>`
runs the capacity/delay sweep and the flexible-phase experiment using the repo
virtualenv. The JSON includes explicit counterexample event schedules.

Evidence is under `docs/qif/diagnostics/20260918/`: diagnostics JSON/text,
targeted pytest log, toy report, ViT report/log, and the single-point Mamba
input. The Mamba log was empty at timeout because output was buffered.

The baseline toy command was `tools/fws_toy_slotmap.py --out <scratch>
--manifest <scratch-manifest>`. ViT used `tools/fws_qif_dse.py` with the existing
`fws_cim_vit_large_frontier.yaml`, `vit_large_story_64_inf.yaml`, and `--mode VIT`.
Mamba used the retained single-point YAML with `mamba2_2p7b_decode_inf.yaml`,
`--mode LLM --no-atlas-points`, under `timeout 240s`.

The next discriminating experiment is on a **fixed physical placement**:
compare permitted schedules by throughput, actual latency, and **total area
including the state they require**. The current six-point toy cannot decide
whether the architecture is worthwhile without that cost.
