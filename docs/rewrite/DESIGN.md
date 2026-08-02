# DESIGN: the `program/` core — locked architecture for the AstraSim execution rewrite

Status: **locked** (synthesized 2026-07-26 from a three-design panel + two judges +
an adversarial gap review; full panel documents in `docs/rewrite/panel/`).
Read `docs/rewrite/CONTEXT.md` first — its facts and constraints are assumed here.

## 1. Verdict

The end state is **Design A (typed placed-operation IR)**: one `Program` of placed
ops constructed directly from an explicit schedule + block templates; every
consumer (ET emission, analytical evaluation, memory replay, retiming, gmap,
faults, viz) is a pure function over it; flattening ceases to exist as a graph
rewrite.

Two grafts are mandatory, per the judges' convergent recommendation:

- **From Design B:** the separation of *program order* from *ET id policy*, with
  a byte-compatible `legacy` id policy used for every existing mode during the
  migration, plus an **always-on emission postcondition** asserting that every
  communicator member's per-group collective sequence is identical (turns the
  historical silent deadlock into a loud pre-AstraSim failure, forever).
  Deviation from B: the `legacy` policy is *temporary*. After retirement, one
  final, deliberate commit switches the default to the pure `program` policy and
  regenerates goldens once, with dlsim traces attached — so the end state does
  not carry two id policies (the maintainer judge's condition), and the
  migration never bets on order coincidence (the risk judge's condition).
- **From Design C:** the migration spine. The emitter cuts over FIRST, fed by a
  quarantined `legacy_lowering` pass over *unchanged* legacy graphs, verified in
  shadow mode (canonical hashes + full per-rank id sequences + `.grf` bytes) on
  all 42 golden specs before any builder changes. Builder replacements then land
  one seam at a time, each with structural differentials against the legacy
  constructor *before* AstraSim is involved.

## 2. Core abstractions

As specified in `panel/design_A_ir_first.md` §1 (`program/ir.py`,
`program/layout.py`, `program/schedule.py`), with these amendments from the
gap review:

1. **`OpKey` gains a `device` component** (critic gap 13): fine programs repeat
   `(role, direction, mb, layer, gemm)` per (tp,cp,ep) rank; identity must not
   lean on a `seq` disambiguator. Key = semantic tuple × placement coordinate.
2. **Same-placement zero-byte P2P ops are legal** (gap 9): the legacy pipeline
   graph's same-stage zero-size PIPELINE edges are heap events in
   `Graph.simulate` and shift analytical tie order; the IR must represent them
   (`P2PDesc` with `src == dst`, `size_bytes == 0`) and the evaluator must
   enqueue them. The ET emitter elides them into plain deps (legacy behavior).
3. **Control classification is `size_bytes == 0 OR non-PIPELINE kind`** (gaps
   3/14): the builders model every legacy edge-less or non-PIPELINE cross-stage
   transfer explicitly, and the emitter's control predicate matches legacy
   exactly.
4. **Per-DP durations are first-class everywhere** (gap 4): `duration_by_dp`
   (len == dp_count) is read by the ET emitter (`COMP` per dp index) *and* by
   the analytical evaluator (index 0 outside dp-stamped emission — exactly
   legacy `Node.duration` property semantics). A missing-profile-on-retimed-op
   is a validation error, not a silent stale duration.
5. **Collective-only device discovery** (gap 12): `Program.devices` includes
   devices seen only via collectives, and the emitter computes
   `rank = dp * num_stages_initial + idx` with the *pre-extension* stage count,
   reproducing converter Step 7's quirk (C's R8) — unit-tested even though the
   current matrix leaves it dormant.
6. **Group-id allocation is label-sorted** (gaps 2/15): wire gids intern from
   1000 in sorted-label, sorted-token order (verbatim
   `_assign_collective_labels` + `_TP_MEMBERS_TO_GID` semantics), not
   first-instantiation order.

## 3. Emission

`program/et_emit.py` per `panel/design_B_schedule_first.md` §3.5, adapted to the
Program IR:

- **Phase A** main ops (`COMP`/`COLL`), **Phase B** p2p materialization (one
  `P2P` op ⇒ one tag ⇒ SEND/RECV pair; the tag-divergence bug class is
  unrepresentable), **Phase C** control-first stable renumber (verbatim
  `_RankTrace._renumber_control_priority` port).
- **Id policies**: `legacy` = per-stage Kahn over intra-rank deps keyed by
  `prio` (the builder creation index, constructed to reproduce legacy op_id
  sequences), Phase-B send/recv creation in legacy Step-11 iteration order
  (data p2ps therefore land at the END of the trace — critic gap 7), then
  Phase C. `program` = pure projection of program order with the two-band
  (control/rest) layout. `legacy` is the only policy used while goldens are
  live; deleted after the final switch.
- **Always-on postcondition**: per instantiated communicator group, every
  member rank's emitted collective sequence is asserted identical; failure
  raises before AstraSim runs. dlsim completability remains the harness gate.
- Singleton groups → zero-duration `COMP` no-op; `dp_count<=1` skip of dp
  collectives; `int(round(sec*1e6))`; manifest sorted by `_manifest_op_key`;
  1-rank duplicate-and-prune stays in the runner shell. All quirks live here,
  once, with contract citations.

## 4. Consumers

Per `panel/design_A_ir_first.md` §3 with amendments:

- `analytic_sim` replays `Graph.simulate` discipline exactly (heap
  `(finish, counter)`, FIFO ready-scan, per-device compute exclusivity,
  comm/p2p slot-free, successor iteration in creation order, same-placement
  zero-byte events enqueued).
- `memory_sim` ports `simulate_memory` onto FLAT programs. **The FLAT builder
  must support MoE block templates from the moment memory migrates** (critic
  gap 5): hybrid/hier MoE goldens flatten MoE for memory today. Flattened MoE
  *execution* stays rejected until post-migration.
- `retime` writes `duration_by_dp` via `OpKey` lookups; fault-variant labels
  (`fault{i}_dp{d}_stage{s}_{fwd,bwd}`) and artifact directory names are a
  **pinned compatibility surface** (gap 10): golden bundle labels are these
  directory names, and the runner's cache.json choreography (multi-run
  accumulation in shared dirs — `ga2`/inference pins) must be preserved by the
  dispatcher port (gap 3-blocker; see §6 note on the harness).
- `dispatch` keeps: mode surface, grad-accum dual-program flow, inference
  `dp_override=1`, interleave closed-form, rank-count validation, artifact
  dirs, and the **time_calc result-attribute surface**
  (`transformer_astrasim_per_rank_forward/...`, `pipeline_astrasim_*`,
  `flattened_astrasim_*` — gap 17, enumerated in the port's tests).
- Inference decode: `DecodeGraph` and `llm_util.estimate_inference_memory` are
  **explicit consumers with their own migration step** (gap 1-blocker) that
  lands BEFORE any deletion of `Graph`/`simulate_train_graph.py`.
- gmap: traffic collection in the legacy DFS-discovery-compatible order during
  the parity phase, `.grf` byte-diff gated (gap 11); SCOTCH permutation applied
  as a pure device relabeling before emission. Faults: `FaultSpace` untouched,
  adapters as in Design A §3.6.

## 5. Migration stages

Every stage lands with all 42 golden gates green; shadow/differential tests are
stage-internal and stricter than the gates. No golden regen before M9.

| Stage | Content | Deletes |
|---|---|---|
| M0 | `RankLayout` unification; delete dead code (`extract_forward_graph`/`extract_backward_graph` — zero call sites) | 3 layout copies, ~310 LOC dead extractors |
| M1 | `ir.py`, `layout.py`, `validate.py`, `legacy_lowering.py`, `et_emit.py` in **shadow mode**: lower legacy graphs → emit → diff (canonical + id sequences + `.grf`) on all 42 specs, no cutover | — |
| M2 | Emitter cutover: `convert_rapid_llm_graph_to_chakra_et` = lower → optimize → emit | converter body (~1,150 LOC); walkers quarantined into `legacy_lowering` |
| M3 | FLAT builder (`schedule.py`, `block.py`, `pipeline.py` FLAT, `transforms.py` overlap) replaces flattener, producing Programs natively; differential = `lower(legacy_flatten(root))` vs `build_fine()`; then memory replay → `memory_sim` (MoE templates required here) | `PipelineGraphFlattener`, `_propagate_local_hw_ids`, legacy overlap transforms, `build_flattened_root_for_memory`, `simulate_memory` graph plumbing |
| M4 | BLOCK programs power hybrid/hier transformer runs + `retime` | `construct_transformer_graph` |
| M5 | PIPELINE programs + `analytic_sim` for analytical/hybrid (incl. ga2 no-dp twin) | `Graph.simulate`, `convert_comm_sizes_to_times`, `_apply_transformer_time`/`_assign_transformer_durations` |
| M6 | Hierarchical pipeline emission over ("pp","dp"); gmap/faults adapters complete | remaining converter shell, `_remap_stages_for_mapping` |
| M7 | Inference/decode port: `DecodeGraph` de-inherited, `estimate_inference_memory`, `prepare_decode_graphs` | decode graph plumbing |
| M8 | Retirement: `simulate_train_graph.py`, `LLMExecutionDispatcher` graph fields, `legacy_lowering.py`, shims | ~5,600 LOC total legacy |
| M9 | Deliberate switch: id policy `legacy`→`program`, single justified golden regen with dlsim evidence; then (optional, new goldens) native 1F1B generator, flattened MoE | `legacy` id policy |

## 6. Equivalence risk policy

The union of the three panel risk registers applies (A §5, B §5, C §5 — read
them). Rules of engagement:

- **Match** is the default for every gate-visible surface; regen is a per-spec,
  separately-committed event with a written justification and (for wall-second
  shifts) a dlsim trace demonstrating both schedules valid. Target through M8:
  **zero regens**.
- Legacy modeling quirks (rank-tails[0] DP attach, ±par_degree ZeRO-3 offsets
  as stage coordinates, un-divided ZeRO-3 bytes, ceil pipeline splits,
  `local_comp_time` zeroing, `b==0` = last microbatch) are reproduced as
  single named, commented rules. Fixing any of them is post-M9 work.
- Harness note: the runner's recording of multi-entry `cache.json` as an error
  string is itself a (weak but deterministic) pin of the artifact choreography.
  Improving the harness to record all entries is allowed **only** as a
  standalone commit with a same-commit recapture whose totals/bundles are
  byte-identical, before M2.

## 7. Explicitly out of scope

Everything in Design A §6 / B §6 / C §6 (they agree): the AstraSim side of the
wall, timing/sizing math, `MemoryEstimator.build_memory_data`, GPipe+interleave
closed form for existing configs, the `equiv/` harness itself, MoE-flattened
enablement during migration, and the rank-layout semantics (implementation
unifies; meaning frozen).
