# Plan: abstraction-level restructure of `program/` (post-M8)

## Context

The M0–M8 migration replaced the legacy graph machinery with the typed `program/` core, gated on
**byte-identical output** at every step. That was right for safety and structurally conservative:
holding emission order fixed forced the legacy *structure* to survive in new clothes. An adversarial
audit measured it (normalized line identity vs. the deleted originals, docstrings/renames stripped):

| new code | legacy original | identity |
|---|---|---|
| `schedule.build_pipeline_events` (:478-1079) | `Graph.construct_fwd_bwd_graph` | **0.74** |
| `pipeline_fine._FineExpander` (:304-851) | `PipelineGraphFlattener` | **0.72** |
| `block_program.build_block_root` (:119-497) | `Graph.construct_transformer_graph` | **0.89** |

Four extension exercises (flattened MoE, a new ZeRO policy, native 1F1B, a new parallelism axis) all
came back at **ratio ≈ 0.9–1.0 : 1** against the legacy code — i.e. the new core is not yet cheaper
to extend. What the migration *did* genuinely buy: `RankLayout` (3 copies → 1), `et_emit.py` (one
emitter, contract in one place, always-on group-order postcondition, p2p = one object so tag
divergence is unrepresentable), `pipeline_coarse.py`, and the 42-spec safety net + `dlsim`.

Concretely still wrong, with citations:

- **Two representations.** Builders make a proto graph (`ComputeEvent` is documented at
  `schedule.py:304` as a "legacy `Node` stand-in … uses the legacy attribute names so the flattening
  port reads them verbatim"); `legacy_lowering.py` (1,095 LOC, permanent) linearizes it. The emitter
  reads the IR; `analytic_sim`, `memory_sim`, `retime`, `viz` read the **proto graph**.
- **Flattening never died**: `_clone`, `_clone_cache`, `_expand_transformer_node`, the ZeRO-3 sibling
  scan (`pipeline_fine.py:461-469`, `:599-613`).
- **Name dispatch survives where a typed field already exists**: `pipeline_fine.py:417` `"linear_softmax" in obj.name`,
  `:426`/`:705` `"optimizer" in …`, `transforms.py:224` `"attention" in name.lower()` — while
  `ComputeEvent.role` is stamped and unused for those branches.
- **The `"bwd" in name` hack is still the fallback** (`pipeline_fine.py:353-356`), and both dead
  branches around it can never fire (`CommEvent.direction` is never assigned).
- **The ZeRO lattice is a 12-literal-key cascade** (`schedule.py:798-1046`) with
  `attach_parallel_edge(skip_non_comm_children, skip_comm_children)` copied byte-identical; the
  collective name is even derived by sniffing which keys exist (`:801-803`) though `CommMeta.kind`
  already is the type.
- **Communicator membership is still *inferred*** from participant counts, with a hand-written axis
  special case (`legacy_lowering.py:243-248`: `if axis == "ep" and participants == tp*ep → ("tp","ep")`).
- **Policy duplicated**: `should_emit_dp_comm` re-derived at `schedule.py:937`; EP-sync split across
  `train_timing.py:4762-4768` (attach point) + `schedule.py:936-953` (gate) + `:4599-4620` (bytes).
- Ops carry legacy ordering as data: `legacy_op_id`, `send_seq`/`recv_seq`, `post_deps`, `label`.

**Goal**: a real level-of-abstraction reorganization — sharding/scheduling/placement become
first-class policies, one representation, deps computed not walked — so the core can express *more*
than before. Predictions stay static except for (a) one measured, reviewed node-ID rebaseline and
(b) explicitly confirmed bugs, each fixed in its own commit with its own delta table.

**Owner decisions**: one-time reviewed rebaseline; bugs in a separate phase, one commit each;
flattened MoE in scope, 1F1B reserved as a student project, a second sharding policy only if
practice-relevant; blast radius = `program/` + `llm_execution.py` + the comm-spec seam in
`train_timing.py` (timing/byte **math** untouched).

---

## Target architecture — five levels, one representation

```
L0  workload.py         WorkloadSpec: typed model/parallelism facts + duration & byte oracles
                        (replaces the ScheduleInputs dict-carrier and the duck-typed
                        ScheduleSpec.from_pipeline_graph, schedule.py:186 — which today
                        silently accepts a wrong object and yields mb=0/num_layers=0)

L1  work.py             WorkItem(kind, microbatch, layer, direction) — what work exists, no order
    policies/sharding.py   ShardingPolicy: .requirements(work, ctx) -> [SyncRequirement]
                           DDP | ZeRO-1 | ZeRO-2 | ZeRO-3  (+ FSDP-prefetch in P8)
    policies/recompute.py  which layers materialize a recompute WorkItem
    policies/routing.py    MoE expert routing + EP sync (absorbs the two _moe_* if-chains)
    policies/overlap.py    overlap DECLARED on a requirement (fraction + anchor),
                           replacing the two competing rewrite implementations in transforms.py

    SyncRequirement carries: collective kind, axis, byte source, participants, and a typed
    ATTACH MODE — a closed vocabulary (before/after/overlap_with/parallel_to(target, edge_class))
    replacing the skip_non_comm_children / skip_comm_children boolean escape hatches.

L2  placement.py        Placement: WorkItem -> DeviceCoord(s). GRANULARITY IS A PARAMETER:
    block.py            COARSE = one op on the stage; FINE = expand via BlockTemplate;
                        BLOCK = one layer over the (tp,cp,ep) sublayout.
                        Replaces pipeline_coarse.py + pipeline_fine.py + block_program.py
                        Communicator groups are CONSTRUCTED here from (axis, fixed coords)
                        via RankLayout — never re-inferred from participant counts.

L3  schedule/policy.py  SchedulePolicy -> per-device ordered task list + cross-stage deps
    schedule/gpipe.py   GPipe (default; reproduces today's order)
    schedule/onefonebee.py  [student project] 1F1B / interleaved virtual stages

L4  ir.py               Program: ops with explicit deps/groups/transfers. No ordering metadata.
    build.py            compose(L1,L2,L3) -> Program (mechanical, ~600 LOC)

L5  emit/chakra.py      ET emission (id policy = deterministic program order)
    sim/analytic.py     list-scheduler evaluator — reads the IR
    sim/memory.py       memory replay — reads the IR
    report/             viz, gmap feed, fault adapter
```

**Dependency rules** (this is what replaces graph walking — deps are *computed*):
1. data-flow inside a block chain, from `BlockTemplate` (honoring `CommMeta.placement` pre/post,
   which the fine path currently ignores);
2. cross-layer L→L+1: same device ⇒ dep, different device ⇒ `TransferOp`;
3. **schedule deps**: device serialization implied by `SchedulePolicy` — GPipe's cross-microbatch
   edges fall out here instead of being hand-wired;
4. sync deps: from each `SyncRequirement`'s declared attach mode.

**Program order** = deterministic function of (schedule step, device, intra-step index). Documented,
stable; no `legacy_op_id`/`send_seq`/`recv_seq`/`post_deps`.

Expected: ~7,850 → ~4,000–4,500 LOC; the ZeRO lattice becomes four small policy classes.

---

## Gate redesign — do this FIRST

Findings that shape it: `equiv/canonical.py` levels 1–2 are **already ID-invariant**
(`tests/test_equiv_unit.py:138` proves it); **10 of 42 specs are fully analytical** (no ETs — stay
bit-exact for free); `manifest.json` is ID-independent *and* is the AstraSim cache key
(`integration.py:456-465`); **memory peaks are ungated today**; and `test_func_test.py` /
`test_IMEC_*` / `test_koyeb_*` / `validation_scripts/*` assert error thresholds against **real
measured hardware** — those are the only tests with physical meaning and are the truest
"predictions must not move" contract.

| Tier | Content | Tolerance | Reuses |
|---|---|---|---|
| **T1 structural** | canonical op multisets + DAG Merkle hashes; `manifest.json`; parsed `comm_groups.json` member sets; **new**: per-rank compute-µs totals, byte histograms by kind/axis, collectives per group, critical-path length; **new**: per-GPU memory peaks | exact, always | `equiv/canonical.py`, `program/shadow.py::_manifest_diff_detail`, `program/validate.py` |
| **T2 predictions** | AstraSim per-rank + `max_sec`; end-to-end totals | analytical specs **exact (1e-9)**; AstraSim-backed within `TOLERANCE_ASTRA` (measured in P1) | `equiv/runner.py::_read_astra_times` |
| **T3 contracts** | dlsim completability (strengthen: fail on candidate deadlock regardless of golden status); group-order postcondition; tag-pairing bijection as a standalone property; V1–V6; deterministic re-emission compared **canonically**, not `filecmp` | boolean | `equiv/dlsim.py`, `program/et_emit.py`, `program/shadow.py:212-244` |
| **T4 bug ledger** | `tests/golden_equiv/bug_ledger.json`: per-(spec, level, field) `{old,new,commit,justification,dlsim_evidence}`; entries expire when `old` stops matching | explicit | replaces wholesale `equiv.capture` regen |
| **T5 physical** | `test_func_test.py`, `test_IMEC_*`, `test_koyeb_*`, `validation_scripts` error thresholds vs measured hardware | existing thresholds | unchanged |

Split `equiv/runner.py::compare_observation` (87 LOC) into
`compare_structural / compare_timing / compare_contract / apply_ledger`.

> **⚠ Stale-cache trap — the single most dangerous hazard.** For graph runs the AstraSim cache key
> hashes the *sorted* manifest + configs, **not** the `.et` files (`integration.py:456-465`). A
> reorg that changes emission order but preserves the op multiset gets a **cache hit on the old
> result** — goldens pass while the schedule silently changed. Every validation run in every phase
> must force `RAPID_ASTRA_CACHE_MODE=NO_CACHE`; note `equiv/runner.py:234` currently uses
> `CACHE_READWRITE` and must be overridden. Also stale after any change:
> `tools/parallelism_sweep_cache.csv` (keyed on inputs only — will not self-invalidate).

---

## Phase status

Each lands independently with T1/T3/T5 green. Only P5 and P7 may move T2 numbers.

| phase | status | landed in | note |
|---|---|---|---|
| **P0** Gate rebuild | ✅ **COMPLETE** | `11fe448` | T1 extras + memory peaks in `RunObservation`; T3 always-on; `compare_observation` split into `compare_structural`/`compare_timing`/`compare_contract`/`compare_determinism`; `NO_CACHE` forced in the harness. Recaptured as a proven strict superset. |
| **P1** Measure & inventory | ✅ **COMPLETE** | `d4f92ed` | Audit findings + the signed-off bug ledger + the approved interface spec (`INTERFACES.md`). |
| **P2** L1 policies extracted | ✅ **COMPLETE** | `11fe448`, `38ac682` | `program/work.py` + `program/policies/{sharding,recompute,routing,overlap,gradaccum}.py`; the typed attach-mode vocabulary replaced the `skip_*_children` booleans; the duplicated `should_emit_dp_comm` and the three-way-split EP-sync policy are unified. |
| **P3** L2 placement + granularity | ✅ **COMPLETE** | `11fe448`, `38ac682` | One `placement.py` + `block.py`; `Granularity` is a parameter; communicator members **constructed** from `(axis, coords)` via `groups.py::CommunicatorFactory`, killing participant-count inference. |
| **P4** L3 schedule policy | ✅ **COMPLETE** | `f06ef6f` | `program/schedule/{policy,gpipe}.py`; cross-microbatch edges are `Schedule.implied_deps`; `layers_per_stage` counts generalized to an explicit `LayerAssignment` map. |
| **P5** L4 direct IR — **the rebaseline** | ✅ **COMPLETE, rebaseline APPROVED** | `f06ef6f`, `5191c9f`, `70938f8` | `build()` is the only Program constructor. `legacy_lowering.py`, `schedule.py`, `pipeline_fine.py`, `pipeline_coarse.py`, `block_program.py`, `transforms.py`, the proto graphs, the clone cache and all ordering metadata are **deleted**. Delta table: `REBASELINE.md`. Owner accepted the residual T2 movement (max \|Δ\| 0.215% on `total_time`, −12.554% on one rank sample); goldens recaptured; **T4 ledger 449 → 0 entries**. |
| **P6** L5 consumers onto the IR | ✅ **COMPLETE** | `f06ef6f` | `analytic_sim`, `memory_sim`, `retime`, `viz` all read the `Program`. The IR-level tie discipline is normative and was **measured to be free**: program order (ascending uid) gives bit-identical totals to the deleted proto-graph evaluator on all 10 analytical + all hybrid specs, and it is the order AstraSim already consumes as node-id priority. |
| **P7** Bug fixes, one commit each | 🔶 **PARTIAL** | `11fe448` (A1), `f06ef6f` (A3) | **A1** (AstraSim cache key omitted all DAG structure) fixed first, as planned — it masked exactly the changes the restructure makes. **A3** (`is_moe_layer` dropped on the tp-overlap head split → memory double-count) fixed with the cutover. **A4** and **A5** are recorded as **dormant / unreachable today** (no reachable config found) and are not fixed. **A2** and **A6** are the ones with real modelling content and large predicted movement, and each needs its own delta table + owner approval — `BUG_LEDGER.md` is the authority on their current state. |
| **P8** Capabilities | 🔶 **IN PROGRESS** | — | Flattened MoE: the audit's blocker was that the memory path does **not** validate MoE comm topology (`memory_sim` reads no comm attributes; template edges are created at `duration=0`), so it is real work at ~6 sites. The FINE MoE program builds for the memory replay but its group-order postcondition did not hold at the cutover (`REBASELINE.md` §6). A second sharding policy (FSDP prefetch) is deferred to a student project (#5 below). Check `equiv/configs.py` for the current matrix. |
| **P9** Cleanup + docs + student scaffolding | 🔶 **IN PROGRESS** | — | **Docs done:** `CONTEXT.md` rewritten to the current architecture (the M8 description with proto graphs and `legacy_lowering` is gone, and the historical part is marked as such); `TESTING.md` updated for the T1–T5 tier split, the ledger↔`equiv.capture` workflow, and the fact that the four `RAPID_*_DIFF` sweeps are replaced by one `RAPID_BUILD_DIFF` sweep; `REBASELINE.md` corrected (mesh2d attribution, cause (2) everywhere, the per-rank column, four arithmetic nits); `STUDENT_PROJECTS.md` written. **Code cleanup done:** `RunPolicy` owns `interleave_scale` (Class B item 8) and the `include_backward`/`include_optimizer` derivations; `WorkloadSpec.from_timing` is the single producer seam, and the five construction flows (`train_timing`, `inference_timing`, `llm_util.estimate_inference_memory`, `simulate_inference_graph`, the validation drivers) now all go through `_prepare_execution_graphs` → `LLMExecutionDispatcher(tc, workload)` with no per-caller re-derivation. **Still open:** `tests/test_program_ir.py`'s literal-id assertions were removed with the lowering pass they tested rather than rewritten against `canonicalize_bundle`. |

**Net effect on the extension exercises** — the reason the restructure was done, re-run as
executable experiments rather than paper plans (`f06ef6f`):

| exercise | before | after |
|---|---:|---:|
| flattened MoE | 1.0 | **0.03** |
| ZeRO-3 prefetch depth | 0.9 | **0.3** |
| native 1F1B | 1.0 | **0.25** (151 LOC in one new file, validated end-to-end) |
| a new parallelism axis | — | **0.43** (weakest; no `AxisSpec` registry was built) |

Briefs built on these numbers: `STUDENT_PROJECTS.md`.

---

## Phases (as planned)

**P0 — Gate rebuild (no behavior change).** Add the T1 extras (compute-µs, byte histograms,
per-group collectives, critical path, manifest, **memory peaks**) to `RunObservation`; T3
always-on; split `compare_observation`; force `NO_CACHE` in the harness. Recapture once: must be a
strict superset — every pre-existing pinned value byte-identical, only new keys added.

**P1 — Measure & inventory (no behavior change).**
(a) Implement a `permute:<seed>` id policy (~40 LOC beside `renumber_control_priority`) + a driver
reusing `equiv.runner.run_specs`; for the ~26 AstraSim-backed specs verify canonical identity +
dlsim completion, then record the Δ distribution → publish `TOLERANCE_ASTRA`.
(b) Finalize the **bug ledger** from the quirk inventory (candidates below). **Owner signs off on
the A-list before any fix is written.**

**P2 — L1 policies extracted.** `WorkItem`, `ShardingPolicy` (DDP/ZeRO-1/2/3), recompute, routing,
overlap-as-declaration, and the typed **attach-mode vocabulary**. Existing builders consume them, so
the lattice is *derived* rather than hardcoded. Lift attachment decisions out of `train_timing.py`'s
name-keyed `comm_metadata` (bytes stay upstream); unify the two `should_emit_dp_comm` copies and the
three-way-split EP-sync policy. *Gate: T1 exact, T2 exact.*

**P3 — L2 placement + granularity unification.** One `placement.py` + `BlockTemplate`; COARSE/FINE/
BLOCK become a parameter. Groups constructed from `RankLayout`, killing the participant-count
inference. Honor `CommMeta.placement` (pre/post) in the fine path. Deletes the three-builder split.
*Gate: T1 exact, T2 exact.*

**P4 — L3 schedule policy.** GPipe extracted; cross-microbatch edges become schedule deps.
Generalize `layers_per_stage` from monotone counts to an explicit layer→stage map (prerequisite for
interleaving). *Gate: T1 exact, T2 exact.*

**P5 — L4 direct IR construction. THE REBASELINE.** `build.py` composes L1+L2+L3 with computed deps.
Delete `legacy_lowering.py`, the proto graphs, `_clone_cache`, name dispatch, ordering metadata.
Note the coupling: Kahn ordering, Step-11 replay, label assignment, gmap collection and the SCOTCH
remap are **one atomic cutover**, no partial step.
*Gate: T1 **exact** (proves semantics unchanged) + T3 + T5 within thresholds; T2 produces a
**reviewed per-spec delta table** → one-time recapture on owner approval.*

**P6 — L5 consumers onto the IR.** `analytic_sim`, `memory_sim`, `viz` read the Program. Requires
defining the IR-level tie discipline that today comes from children-list adjacency order.
*Gate: T1 exact incl. memory peaks; analytical totals bit-exact.*

**P7 — Bug fixes, one commit each**, each with its own delta table + ledger entry. Candidates to
confirm/reject in P1 (evidence-first, none pre-approved):
rank collisions from pre-extension `num_stages_initial`; un-divided ZeRO-3 gather bytes
(`pipeline_fine.py:360`); DP collective attached only to `rank_tails[0]` (`:629`); manifest recording
`ALL_REDUCE` as `-1` (cache-key collision); `is_moe_layer` dropped on tp-overlap head split (memory
misattribution); Step-11 iterating pre-SCOTCH-remap order; `local_comp_time` zeroing; interleave
scale applied to the DP grad-sync tail.

**P8 — Capabilities.** Flattened MoE — note the audit's finding that the memory path does **not**
validate MoE comm topology (`memory_sim` ignores comm attributes entirely, and template edges are
created with `duration=0`), so this is real work at ~6 sites, not a flag flip. Second sharding
policy (FSDP forward/backward prefetch — what PyTorch FSDP actually does) only if it earns its keep.

**P9 — Cleanup + docs + student-project scaffolding.** Collapse the five duplicated
dispatcher-construction flows (`train_timing`, `inference_timing`, `llm_util.estimate_inference_memory`
— which is invisible to `calc_time`-level refactors and mutates `tc.pipeline_graph` as a side effect
— `simulate_inference_graph`, `huggingface_bench_validation`). Relocate `_pipeline_interleave_scale`
and the thrice-duplicated `include_backward`/`include_optimizer` derivations into a `RunPolicy`.
Rewrite `tests/test_program_ir.py:304,411` against `canonicalize_bundle` instead of literal ids.

---

## Student projects the restructure enables

Each is scoped so the abstraction boundary is the deliverable's edge: the student implements one
policy or consumer behind a stable interface, and the existing gates prove they broke nothing.

> **Expanded into startable briefs — with the exact interface, the acceptance gates, a difficulty
> estimate and a concrete first step for each — in
> [`STUDENT_PROJECTS.md`](STUDENT_PROJECTS.md).** The table below is the index.

| # | Project | Size | Interface exercised |
|---|---|---|---|
| 1 | **Native interleaved-1F1B / virtual stages** — replace the closed-form bubble multiplier with a real schedule | L | `SchedulePolicy` (L3) |
| 2 | **Fast surrogate simulator** — give `equiv/dlsim.py` a clock + comm cost model (`analytic_sim.convert_comm_sizes_to_times` already free-function-shaped); validate makespan vs AstraSim across 42 specs, report accuracy/speedup | M–L | consumer over the IR |
| 3 | **Auto-parallelism search** — search (tp,cp,ep,pp,dp,zero,recompute,overlap) using #2 as the inner loop | L (capstone) | L1+L2+L3 policy space |
| 4 | **Overlap as first-class scheduling** — replace fractional heuristics with explicit async comm + wait ops; validate against measured overlap efficiency | M | `OverlapPolicy` (L1) |
| 5 | **FSDP-style sharding with prefetch depth** — implement + validate against real FSDP traces | M | `ShardingPolicy` (L1) |
| 6 | **Congestion/topology-aware analytical collectives** — go beyond ring formulas; compare to AstraSim's congestion-aware backend | M | `analytic_sim` (L5) |
| 7 | **Memory-aware recompute policy search** — minimize time subject to a memory cap, using `memory_sim` + `analytic_sim` | M | `RecomputePolicy` (L1) |
| 8 | **MoE routing/imbalance from real traces** — extend the existing one-hot imbalance model | M | `policies/routing.py` (L1) |
| 9 | **New hardware/model validation** — add a device or model family and close the loop against measured data | S–M | T5 harness |

---

## Verification

- Golden gate every phase: `env RAPID_ASTRA_CACHE_MODE=NO_CACHE LD_LIBRARY_PATH=/u1/ee/karfakis/gcc-10.2.0/lib64:/app/nanocad/projects/personal/gkarfakis/anaconda3/lib ./.venv/bin/python -m pytest tests/test_equiv_golden.py -q` → 42/42.
- Full suite (webui browser/remote/service/worker excluded) matches baseline + intentional additions.
- ~~`RAPID_FINE_DIFF/BLOCK_DIFF/COARSE_DIFF/HIER_DIFF=1` sweeps until the builders they compare are
  deleted, then replaced by the T3 canonical determinism check.~~ **Done as planned**: the four
  sweeps were deleted with their builders in `f06ef6f`. Their determinism property is now held by
  the always-on `test_reemission_deterministic` (two separate processes, compared canonically) and
  by one env-gated whole-matrix sweep, `RAPID_BUILD_DIFF=1` on `tests/test_build.py`
  (every spec × {COARSE, FINE}: **O1** field-for-field on the Program, then canonically on the
  bundle, then dlsim). See `docs/rewrite/TESTING.md` §Tier 3.
- **T5 physical suites within existing error thresholds at every phase** — the real evidence that
  predictions did not move.
- P1 sensitivity report and P5 delta table archived under `docs/rewrite/`; `docs/rewrite/DESIGN.md`
  and `TESTING.md` updated at P9.
