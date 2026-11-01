# Consolidated-main CIM/QIF test repair

Baseline: `3cdce5c`, with the original failure logs supplied for this repair.
The old-checkout log has exactly 80 failing cases in the 13 files below.
The new-checkout log has 112 failures; resolving external artifacts exposes
the same historical checks without restoring generated results to git.

The policy authority is `DECISIONS.md`, especially D35-REVISED, D36, D37,
D38, the 2026-08-29 link amendment, and ADJ-16. Original checkpoint
`0176fa6` (rewritten `6ccaf2d`) explicitly describes historical reports as
provisional, the real-model mapper as stub-v0, and two subset configs as
retaining retired axes. `28009de` (rewritten `3cdce5c`) says production
mapper behavior is unchanged by its scheduling diagnostics.

## Accounting for all 80 pre-existing failures

Counts are failing **cases**, including parametrization, rather than functions.

| Test file | Stale tests repaired | NEEDS GEORGE | Diagnosis |
|---|---:|---:|---|
| `test_fws_cim.py` | 3 | 0 | The link amendment changed intra-chip latency from 50 ns to 5 ns. |
| `test_qif_atlas.py` | 0 | 1 | Historical PD atlas byte comparison needs a reference decision. |
| `test_qif_dse_allocation.py` | 23 | 0 | D36/D38 retired the old axes; use legal Y/Z/fill axes, named refusals, the committed batch-one Llama twin, and explicit small-capacity controls. The frozen demo is history, not a copy of a current point. |
| `test_qif_folding_mapping.py` | 2 | 1 | D37 saves two further boundary remainders; D38/ADJ-16 enlarge the lowered observation window. Pin current op counts, lowered-step deltas and both headline rates as regression anchors. Historical comparison JSON needs a reference decision. |
| `test_qif_folding.py` | 2 | 0 | Dense packing across stage boundaries changes macro count from 4828 to 4826 and the endpoint comparison by two boundary remainders. |
| `test_qif_frontier.py` | 25 | 2 | Current axes, derived per-layer stages, fill-dependent state, explicit partition refusal, source-specific binding terms, and historical reports with no verification claim. Two full regeneration comparisons need reference decisions. |
| `test_qif_granite_e2e.py` | 5 | 1 | D37/D38 change placement, stages, demand and softmax sizing. Read all stages for the pool comparison and export the current pool view from the same evaluation. Historical report comparison needs a reference decision. |
| `test_qif_links_relaxation.py` | 4 | 0 | All flat stage-boundary hops are relaxed; traffic bytes stay fixed but concurrent memory high water need not. DSE uses the current fill axis. |
| `test_qif_mapping.py` | 0 | 1 | Historical P3 atlas byte comparison needs a reference decision. |
| `test_qif_pipeline.py` | 2 | 1 | MoE residency is not activity (M6); inactive experts remain unread. Hop expectation is 5 ns. The slow-ADC contention probe still returns zero without a documented explanation. |
| `test_qif_qwen_e2e.py` | 1 | 2 | D37 no longer guarantees the gate and QKV share a chip; weight ownership and dimensions remain checked. Historical report and atlas comparisons need reference decisions. |
| `test_qif_run_bundle.py` | 2 | 0 | The checkpoint explicitly preserves the ViT-Huge and Granite-1B bundle members with retired axes; assert their named refusals. |
| `test_qif_sa_geometry.py` | 2 | 0 | D38 changes attention stage indices and smaller per-layer targets can reach geometry saturation before analog pace. |
| **Total** | **71** | **9** | **80 cases; no numerical production-law bug identified.** |

## Strict expected failures

Every new expected failure is `xfail(strict=True)` with a `NEEDS GEORGE:`
reason. The contention case remains in the default suite:

- `test_qif_pipeline.py::test_cross_stage_bank_sharing_costs_real_time_when_the_macro_collides`

The reference question is whether to retain an archived source-bound golden
or establish a new reference for the deliberately redesigned implementation.
No historical artifact is overwritten or promoted to a current performance reference.
These nine cases are also `slow` and therefore skipped by default:

- `test_qif_atlas.py::test_the_pd_document_is_what_a_fresh_run_produces`
- `test_qif_mapping.py::test_the_checked_in_export_is_the_one_the_tool_emits_today`
- `test_qif_granite_e2e.py::test_the_shipped_granite_report_is_what_a_fresh_run_produces`
- `test_qif_granite_e2e.py::test_the_shipped_granite_atlas_is_what_a_fresh_run_produces`
- `test_qif_qwen_e2e.py::test_the_shipped_qwen_report_is_what_a_fresh_run_produces`
- `test_qif_qwen_e2e.py::test_the_shipped_qwen_atlas_is_what_a_fresh_run_produces`
- `test_qif_folding_mapping.py::test_the_granite_packing_artifact_is_what_a_fresh_run_produces`
- `test_qif_frontier.py::test_the_checked_in_curve_regenerates_identically[granite-frontier0-shipped0-model0-art0-Granite-4.0-H-Tiny]`
- `test_qif_frontier.py::test_the_checked_in_curve_regenerates_identically[qwen-frontier1-shipped1-model1-art1-Qwen3.5-4B]`

Eight of these nine slow cases belong to the 80-case baseline. The Granite
atlas case was deselected there because it had already run for over an hour.
The three other whole-Granite packing-report readers are slow too: their
shared setup measured 95.45 seconds.

The full-suite pass also identified the existing Koyeb validation sweep as
longer than 60 seconds. It is marked `slow`; its error threshold and all
model inputs are unchanged. This additional lane change is outside the
80-case CIM/QIF failure accounting above. The completed timing report also
measured 86.46 s in the shared fixture for the 234-case functional matrix.
All its consumers are marked `slow`, since skipping one row alone would
move the entire setup cost to the next row. The matrix's inputs, assertions
and two existing non-strict expected-failure markers are unchanged; the
221-test golden gate remains always on.

The seven pre-existing default xfails remain in `test_IMEC_inf.py`, all with
reason `Pending model refinements`: `test_avg_abs_error_by_model` for
`H100-model-Llama 2-13B`, `H100-model-Llama 2-70B`, and
`H100-model-Llama 2-7B`; and `test_avg_abs_error_by_tp` for
`H100-TP-1`, `H100-TP-2`, `H100-TP-4`, and `H100-TP-8`.

## Missing inputs and validation scope

The never-committed six toy point JSONs are rebuilt by the committed
`tools/fws_toy_slotmap.py` in pytest scratch, with fixed authored/generated
timestamps. Generation takes less than one second; all six pass both the
independent slot checker and the atlas loader. They are additional input
repairs, not members of the 80-case old-checkout failure list.
Two fixed-time runs also reproduce all six point files, the report and
the manifest byte for byte.

The retired `granite_capacity_banks_retired/dse_report.md` was never
committed. Its old sweep axes are now refused, but the committed
`fws_qif_dse.render_markdown` can rebuild the document directly from its
archived JSON without running that sweep. The reader renders it twice,
checks byte equality, and writes only to pytest scratch. Both classes of
never-committed input are therefore rebuilt, with no missing-input skip.

The standalone OPTIMA validator uses the same external-artifact resolver.
Its schema checks read Z from `adcs` and derive mux as `cols/adcs`, preserving
the original OPTIMA reference numbers. Its live mapped-DSE gate now sweeps
fill on the legacy lockstep vehicle: full fill round-trips, underfill carries
a named mapping refusal. Historical frontier rows explicitly check that
provisional reports claim no unrun verification; the separate live round-trip
gate still runs. The audit repair additionally selects and round-trips a
CURRENT frontier for each model using the legal fill axis; this does not
rebaseline either historical curve. No frozen OPTIMA targets are changed.

The only production edit corrects the link provenance **text** from 50 ns to
5 ns, matching the existing numeric constant and documented amendment.
All other changes are tests, validation code, test documentation or ignore
rules. Golden recordings and A100 baseline files remain unchanged.

## Stage 1 verification (before the audit repair)

- Golden gate: 221 passed before the production text correction and 221
  passed afterward (39.48 s for the final dedicated run). SHA-256 comparison
  finds all 46 golden files unchanged; `git diff` against `3cdce5c` also
  confirms no golden or A100-baseline file changed.
- Standalone validator: 524/524 checks pass, exit 0, 126.44 s wall time.
  Its A100 result-file comparisons remain exact byte comparisons.
- Targeted runs: 129 physical/e2e tests passed, 107 SA-geometry tests
  passed, and 115 DSE/frontier/bundle/helper tests passed. Three slow
  whole-Granite packing checks also passed (97.19 s).
- Missing results root (superseded by the audit): the artifact consumer skips with the full missing
  path and `RAPID_QIF_RESULTS`; the local fixture check still passes.
- Slow-lane probes: default gives one pass/one skip; `-m slow` gives one
  pass/one deselection; `RAPID_RUN_SLOW=1` gives two passes.
- `git check-ignore` verifies generated output patterns, the fixture
  exception, and zero newly ignored tracked files (including a comparison
  with the pre-change ignored set).

- Final default suite: **2695 passed, 292 skipped, 8 xfailed, 0 failures**,
  exit 0. Pytest reports 653.04 s; process wall time is **659.97 s**
  (10 min 59.97 s). Of the skips, 247 are opt-in slow cases (234 functional
  matrix rows plus 13 other slow cases), 44 are the existing environment-gated
  determinism sweep, and one is the sandbox's blocked local socket. There
  are no missing-input skips with the provided results root.
- Before the functional matrix moved lanes, a full run also completed with
  zero failures: 2926 passed, 59 skipped, 8 xfailed, 2 xpassed. Its 234
  functional rows were 232 passes and the two existing XPASS cases. The
  retired Markdown reader was subsequently rebuilt and passed its focused
  check; the final full run above includes it.

## Audit repairs

Retiring an input must preserve the property its test guarded. No production
code, pricing law, model parameter or historical result is changed in this round.

1. D34 mismatch warnings are positively exercised with synthetic spans on
   legal D38 stage IDs. The real balance law, quiet-mode stdout warning,
   Markdown headline, candidate cell and legend all run. Separate assertions
   pin occupancy spread (5x) and digital spread (9x). Explicit-partition
   refusals and the balanced negative control remain additional coverage.
2. The run-ID test holds its label fixed and changes `pp_util` from 1 to 0.5;
   it also checks that the machine's resident-stream count changes.
3. Granite's headline test pins beat 26.3210577032 us, 37992.3941992 tokens/s,
   digital area 30306.92152689 mm2 and total energy 807802607914.4937 pJ.
   It restores digital-component energy/partial coverage, engine/fabric
   provenance and binding terms, and all-stream residency/store checks.
   These are regression anchors at `3cdce5c`, **not validation targets**:
   D37/D38 changed packing and stage count; ADJ-16 derives the window.
4. The packing comparison uses pinned anchors, not a new NEEDS GEORGE:
   dense accumulator ops 4432; dedicated/dense lowered-step medians
   13.3004221889/4.01578947368 us; delta -9.28463271521 us (-69.8070526134%).
   Both headline rates are independently pinned (75244.8206412 and
   18575.6707049 tokens/s). A smaller lowered-step median does not establish
   a faster headline: the latter includes pipeline floors. These are current
   regression anchors with the same D37/D38/ADJ-16 qualification.
5. Allocation checks again require distinct headlines and timelines, retain
   both feasible and budget-refused candidates in one legal fill sweep, and
   continue to a feasible point after a config/card refusal. The last test
   injects one bad card after applying a legal fill input, exercising the
   actual parser's refusal rather than a fake candidate result. The frontier
   state summary again checks its budget basis, measured count and min/max
   area and state-byte ranges.
6. Required artifacts now FAIL by default. Only the explicit environment
   opt-out `RAPID_QIF_ALLOW_MISSING=1` permits skips; `RAPID_QIF_RESULTS` and
   the missing path remain in the message. Unit checks cover absent, false,
   non-1 and explicit-1 opt-out values. TESTING.md documents this contract.
7. The validator now runs `--emit-config --verify` on CURRENT two-point fill
   frontiers for Granite and Qwen, requires selected IDs on the live Pareto
   front and checks throughput and placement round trips. The full validator
   measures 145.34 s for Granite and 21.43 s for Qwen, each below five minutes.
   These are required live checks, with fresh scratch directories to prevent
   stale outputs satisfying presence assertions. There is no remaining
   KNOWN GAP for the historical checks' lost round-trip property; only the
   original source-bound reference comparisons remain unresolved.
8. Both frontier regeneration xfails now name the actual first failure:
   D38 refuses retired axes before comparison with historical results.
9. The 234 functional cases and Koyeb stay slow. TESTING.md now requires
   `-m slow` (or `RAPID_RUN_SLOW=1`) before merging. The entire 247-case slow
   lane was run for this audit, including the historical regeneration
   checks; the completed results below distinguish expected failures and XPASS.

No additional NEEDS GEORGE case has been introduced. The Stage 1 list above
remains the full list; the two frontier reasons are now more precise.

### Exact artifact equality and failure diagnostics

The first full slow run was interrupted after approximately 69 minutes while
the Granite atlas case was active. A controlled run of the **unchanged** case
with only assertion rewriting disabled (`--assert=plain`) regenerated and
compared the atlas in **57.62 s** wall time (50.25 s pytest), producing the
expected historical-reference failure. This identifies pytest's large-string
failure diagnostics as the runtime problem rather than atlas generation.

Seven report/atlas comparisons now compute their original exact string
equality into `same_bytes`, then assert that Boolean. Every character and
whitespace difference is still rejected; no parsed-value or hash comparison
replaces equality. Only the failure rendering changes. The final full slow
gate uses normal assertion rewriting and executes every regeneration check;
the two retired-axis frontier cases still stop at their documented input
refusals. The Granite atlas case completes in **51.87 s**, with the
same historical reference mismatch. The interrupted attempt is retained in
`tmp/qif-audit/slow-interrupted.log` and is not counted as a completed gate.
Default, golden and validator gates were also rerun after this diagnostic fix.

## First audit gate results (before the re-audit below)

- Golden gate: **221 passed**, exit 0 (39.35 s pytest; 41.52 s process wall).
  All **46 golden SHA-256 hashes are unchanged** from the pre-repair manifest.
- Full default suite: **2701 passed, 292 skipped, 8 xfailed, 0 failures**,
  exit 0. Pytest time 678.61 s; process wall time **681.32 s (11m 21.32s)**.
  The skips are 247 slow, 44 environment-gated, and one blocked local socket;
  none skips missing QIF evidence. The eight default xfails are listed above.
- Standalone validator: **544/544 checks pass**, exit 0; process wall time
  **291.43 s**. This includes 20 additional checks for the two CURRENT
  frontier selections and their emitted-config round trips.
- Complete slow lane: **236 passed, 9 xfailed, 2 existing xpassed,
  0 failures**, exit 0; **439.87 s (7m 19.87s)** process wall time.
  Details, including all 234 functional cases and Koyeb, follow below.
- Generated-output ignore probes match, `fixture_*.json` stays exempt, and
  no tracked generated-output path is ignored. No production pricing changes
  or golden/A100-baseline edits were made in this audit.

Final logs: `tmp/qif-audit/golden-final.log`, `default-final.log`/`.xml`,
`validator-final.log`, and `slow.log`/`.xml`; each gate has a zero `.exit` file.

## Completed slow lane after the diagnostic fix

`pytest -m slow` with normal assertion rewriting completes with **236 passed,
9 xfailed, 2 xpassed, 0 failures**, exit 0; 2754 cases are deselected. Pytest
time is 436.98 s; process wall time is **439.87 s (7m 19.87s)**.

All 234 functional cases execute (232 pass plus the two existing non-strict
XPASS markers), and Koyeb passes at 5.98% average absolute error against its
unchanged 6.5% threshold. The three positive Granite packing checks pass.
All nine historical regeneration checks execute and remain strict xfails;
their full IDs and reasons are listed above and in `slow.log`/`slow.xml`.
The two existing XPASS cases are hierarchical GLM4-MoE training with EP,
for MHA and GQA, both DP2/EP2/TP2/CP1/PP2/MB2 with sequence parallel enabled.

## Re-audit: required evidence and restored independent assertions

The five remaining coverage findings in `tmp/qif-reaudit/AUDIT.txt` are
addressed by tests/docs-only commits. No production or pricing file changes
in this round, no reference files are rewritten, and no new NEEDS GEORGE
cases are added.

1. All historical regeneration xfails now specify their known exception:
   `AssertionError` for the seven byte comparisons, `QifDseUsageError` for
   the two historical frontiers' retired-axis refusals. Required frontier
   references resolve before the refused sweep. Missing evidence raises
   `pytest.fail`, which cannot satisfy either exception expectation. The
   six external atlas names remain in collection even with an empty root;
   discovery can still add documents, but cannot silently drop these cases.
   Missing tracked packing evidence also fails outside its comparison xfail.
2. Both archived curves pin every priced candidate's binding block, analog
   binding flag and saturation count: **56 Granite and 61 Qwen candidates**.
   The anchors explicitly retain Granite's analog-bound `c037` and the
   scan/delta/PV alternatives, rather than prescribing attention everywhere.
3. The explicit 1 MiB memory control pins **eight** violations, the eight
   chip scopes, tier, owner, capacity and high water. The summary count must
   equal the detailed row count, and each detailed ratio is recalculated.
4. The MoE bank-pass test counts occupied macros and their claimed column
   sets directly from the mapping (**1708** resident sets). Active reads and
   unread expert residency remain distinct under M6.
5. Legal full/half-fill runs check the balanced Markdown headline, absence
   of the mismatched headline, and an `ok` stage-balance cell on every
   candidate row, as well as the existing JSON/stdout negative control.

Both additional suggestions are implemented: the objective and full
objective-specific selection-rule text are pinned, and the Granite fabric
ratio has a separate hand calculation. At the current probe's 1829 keys,
32x64 arrays and four folds, QK/PV/softmax/fill total **10398 cycles**.
At 0.95 GHz over the D38 per-layer 300 ns analog target this gives
**36.48421052631579**, independently of the reported sizing ratios. These
are regression anchors at `3cdce5c`, not physical validation targets.

### Mutation evidence

`tmp/qif-reaudit/test_repaired_mutations.py` applies each corruption without
editing production files. All **nine cases FAIL**, exit 1 (2.66 s pytest;
8.04 s process wall in `repaired-mutations.log`):

- Flip the analog-binding flag, insert `NONEXISTENT_BLOCK`, and invert the
  saturation count separately on each archived curve: **6 failures**.
- Replace the balanced Markdown headline with `STAGE BALANCE: MISMATCHED`:
  **1 failure** at the required balanced headline.
- Change the memory count from 8 to 999: **1 failure** at the exact count.
- Add 100 to both occupied and unread column counters: **1 failure** against
  the independent mapping census (1808 versus 1708).

With an empty `RAPID_QIF_RESULTS`, all **49 required-artifact consumers**
reject missing evidence, including all **eight external regeneration
xfails**: **48 failures and one fixture-setup error**, exit 1, with **zero
XFAILs** (62.71 s process wall). The fixture error is the missing JSON used
to rebuild retired Markdown, and names the same required path/env var. Missing-evidence
behavior is therefore never reported as an expected historical mismatch.
With `RAPID_QIF_ALLOW_MISSING=1`, the same **49 consumers SKIP**, exit 0
(61.75 s process wall). A separate missing tracked-packing probe also gives
an ordinary failure under the original regeneration marker (2.92 s wall).

The existing non-strict markers in `test_func_test.py` and `test_IMEC_inf.py`
are unchanged. The two functional XPASS cases remain:

- `test_func_test[training:moe:glm4_moe:mha:hierarchical:dp2:ep2:tp2:cp1:pp2:mb2:tp_spTrue]`
- `test_func_test[training:moe:glm4_moe:gqa:hierarchical:dp2:ep2:tp2:cp1:pp2:mb2:tp_spTrue]`

### Re-audit gate results

All four gates were rerun after the test changes, with the supplied sibling
results root present. Each exits **0**:

| Gate | Result | Process wall time |
|---|---|---:|
| Golden | **221 passed**; **46/46 golden SHA-256 hashes unchanged** | **40.80 s** |
| Full default suite | **2701 passed, 292 skipped, 8 xfailed, 0 failures** | **674.49 s** |
| Standalone validator | **544/544 checks pass** | **291.81 s** |
| Full slow lane | **236 passed, 9 strict xfailed, 2 existing xpassed, 0 failures** | **433.58 s** |

The 292 default skips remain 247 slow cases, 44 environment-gated cases and
one blocked local socket. No default skip is missing QIF evidence. All nine
historical regeneration checks execute in the slow lane; the two existing
functional XPASS markers are listed above. The existing NEEDS GEORGE list
and 80-case diagnosis table are unchanged.

The scratch observer records every required-artifact consumer in both full
lanes. Their union is exactly the **49 cases** tested with the empty root
and opt-out: no consuming case is omitted or reclassified as an XFAIL.
Generated-output ignore checks also pass with zero tracked output files
ignored. Only tests and docs changed in this re-audit; no production,
pricing, reference, golden or existing non-strict marker changes.

Gate evidence: `tmp/qif-reaudit/repair-golden.log`,
`repair-default.log`/`.xml`, `repair-validator.log`, and
`repair-slow.log`/`.xml`, with matching zero `.exit` files. Missing-evidence
proof is in `repair-missing-fail.log`/`.xml`,
`repair-missing-skip.log`/`.xml`, and `repair-missing-packing.log`;
the deliberate corruption failures are in `repaired-mutations.log`/`.xml`.
