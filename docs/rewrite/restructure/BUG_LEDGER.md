# Bug ledger — evidence-backed quirk classification (restructure P1)

Produced by an evidence-first audit against `git show 85894c6:*` (the pre-rewrite originals) and the
live code at `03ace0e`. Classes: **A** confirmed bug (fixing SHOULD change output) · **B** deliberate
modeling choice (preserve) · **C** semantically inert (ordering/naming; may perturb AstraSim wall
clock via node-id priority) · **D** unclear, needs an experiment or an owner call.

> **Scope note added 2026-07-29 (P5 verification).** The reframe below is about the ZeRO-3 **per-rank
> gather** machinery at `par_degree > 1`, and it scopes items 1, 4, 5(10a) and 10f. It does **not**
> cover **A6**: the S8/S15 cross-stage prefetch anchor fires at exactly `tp=cp=1, pp=2`, which every
> zero2/zero3 spec is, and both zero3 goldens pin it.

> **Reframe that removes four constraints:** the ZeRO-3 per-rank gather machinery is **dead in the
> golden matrix**. `equiv/configs.py:163-174` builds zero2/zero3 specs only at `dp=2, tp=1, cp=1`, so
> `par_degree == 1` and `_should_shard_zero3_transformer` (`program/pipeline_fine.py:307`) always
> returns False. Items 1, 4, 5(10a) and 10f are **unvalidated code, not pinned behavior** — no golden
> constrains them. Treat that region as new development with new goldens.

## A-list (confirmed bugs, ranked by impact)

> **A1 is fixed (restructure W1 / P0).** `cache_key = sha256("df-astra-cache/2|" ‖ workload_sig ‖ et_sig)`
> where `workload_sig` is the old formula (manifest + system + network + remote memory + comm groups)
> and `et_sig` is a name-bound digest of every emitted `llm_graph.<rank>.et`
> (`astrasim_lib/integration.py::_hash_workload_files`). Two workloads with equal op multisets but
> different dependencies, node-id priority order, p2p tags or rank assignment no longer collide —
> pinned by `tests/test_astrasim_cache_key.py`. `workload_sig` survives as the stable per-bundle RUN
> IDENTITY in the new `astra_runs.json` result log (written in every cache mode), which is what let
> the golden recapture stay a strict superset. Two enabling fixes shipped with it: `run_perf.py` now
> HONORS an inherited `RAPID_ASTRA_CACHE_MODE` (its map keys used spaces, so the declared `NO_CACHE`
> silently resolved to `CACHE_READWRITE` and *overrode* every caller — the golden gate's
> `RAPID_ASTRA_CACHE_MODE=NO_CACHE` never reached the subprocess), and `equiv/runner.py` now forces
> `NO_CACHE` explicitly instead of `CACHE_READWRITE`. No modeling output changed.

| id | Bug | Site | Correct behavior · expected delta |
|---|---|---|---|
| **A1** ✅ FIXED (W1) | **AstraSim result-cache key omits all DAG structure.** `cache_key = sha256(manifest ‖ system ‖ network ‖ remote_mem ‖ comm_groups)`; the `.et` files are never hashed, and the manifest is a *sorted multiset* with no deps, no order, no p2p peer, no tag. | `astrasim_lib/integration.py:456-466`, `_hash_file_bundle:140-153`; manifest `program/et_emit.py:560-606` | Two workloads with equal op multisets but different dependency structure / priority order / p2p pairing **collide** and a stale wall time is returned silently. Hash the emitted ET bytes (or add deps+peers+order to the manifest). **No model change; makes staleness loud.** ⚠ This masks exactly the kind of change the restructure makes — fix FIRST. |
| **A2** | **DP collectives attached to `rank_tails[0]` only.** One cluster rank bears the whole grad all-reduce/reduce-scatter; the other `par_degree-1` ranks emit nothing. | `program/pipeline_fine.py:616-629`, placement at `:829-849` | Correct: one dp collective per (stage, cluster-rank) on `par_degree` disjoint dp-axis communicators. Fixing **increases** dp traffic by `par_degree`×. Error is in **contention, not the serial critical path** (max-over-ranks may hide it) — measure per-link busy time, not just totals. |
| **A3** | **tp-overlap head drops `is_moe_layer` → memory DOUBLE-COUNT** (not misclassification): tail files layer L under moe, head files the same L under dense, and the census sums both. | `program/pipeline_fine.py:186-196` (`OVERLAP_NODE_COPY_ATTRS`), census `program/memory_sim.py:352-368`, `:412`/`:432` | Add `is_moe_layer` to the overlap copy list. Reported peak/static memory **decreases** by one dense layer's worth per split MoE layer per device; can flip capacity warnings. **Timing unaffected.** Path is live (`llm_execution.py:776-787` passes MoE templates *and* tp_overlap). |
| **A4** | **Collective-only stage rank collision.** `rank = dp_idx*num_stages_initial + stage_idx` with a pre-extension count: `rank(first_ext, dp=0) == rank(devices[0], dp=1)`. | `program/legacy_lowering.py:327`, `:604-613`; `program/ir.py:235-242` | Asserted verbatim today in `tests/test_program_ir.py:490-493`. At dp≥2 the collective is silently appended to another replica's trace. Correct: use the post-extension device count, or refuse collectives on non-compute devices. **Dormant — no reachable config found** (all ZeRO-3 ±pp hops are one-stage between adjacent host/target pairs, and `program/layout.py:113-114` bounds-checks). |
| **A5** | **ZeRO-3 dead store with live side effects.** `per_rank_edges = self._ensure_zero3_per_rank_edges(...)` is assigned and never read (unlike the sibling at `:717-721`), yet it consumes `par_degree` op ids (shifting emission priority) and appends the discarded edges to real nodes' `.parents`. | `program/pipeline_fine.py:620-629` | Benign only *by accident* (the lowering's `_collect_objects` is children-only, so the orphans are never discovered). Also a clone-cache hazard: a gather reachable as both `dp_child` and `zero3_attachment` gets whichever expansion runs last. Delete or use. **Unreachable today.** |
| **A6** ⚠ NEW (P5 verification, 2026-07-29) | **The ZeRO-3 prefetch anchor is chosen by LAYER ARITHMETIC and lands on another device.** Rows **S8** (`_forward_layer`) and **S15** (`_backward_layer`) host layer `L`'s parameter gather on `layer_item(L - prefetch_depth)`. At a stage boundary that layer is on the OTHER pipeline stage, so a gather correctly PLACED on the device that consumes the parameters inherits the *foreign* stage's predecessor set (`build.py` `PARALLEL_TO`: `deps(req) += deps(anchor)`, and `_resolve_anchor` falls back to every chain when the anchor has none on the requirement's device, `:1200-1201`). The policy already KNOWS: `ZeRO3._via` returns `VIA_DATA_FLOW` instead of `VIA_ALL` for exactly these requirements. It narrows the SUCCESSOR side and leaves the DEP side to inherit blind. | `program/policies/sharding.py` `_forward_layer` (S8) `:594-618`, `_backward_layer` (S15) `:677-699`, `_via` `:449-457`; `program/build.py` `_resolve_anchor` `:1200-1201` | A layer-`L` parameter all-gather is a **dp-axis** collective over `stage(L)`'s dp replicas; no rank of another stage participates and no other stage produces its input, so the dependency on a stage-0 op is not a data dependency in any sense. It is the *prefetch-depth* heuristic ("issue the gather one layer early"), which is a **device-local** notion; at a stage boundary "one layer earlier" resolves to a layer on another device, where it means nothing. Real FSDP/ZeRO-3 + PP has no cross-stage signal: each stage prefetches against its own schedule and issues its first layer's gather at its own entry. **Correct fix:** re-anchor on the gather's own device (a `_dep_anchor` rule beside `_via`). **Expected delta:** moves `train:analytical:dp2tp1cp1pp2mb2sp0:zero3` (measured ≈ **-4.3%** with a crude local anchor) and needs `analytic_sim._ROOT_COMM_IS_UNTIMED` settled first, because a stage's first gather legitimately becomes a program ROOT and would otherwise be free. **NOT fixed in P5**: it is a deliberate modeling change with its own golden movement, not a cutover artifact. P5 makes the two evaluators agree on the CURRENT anchor by putting the cross-rank edges on the wire (INTERFACES amendment 2026-07-29). |

## B — preserve (deliberate modeling choices)

| id | Item | Why it stays |
|---|---|---|
| 1 | ZeRO-3 per-rank gather bytes "un-divided" | Not a bug for tp/cp: each cluster rank owns its own TP shard (`params_per_layer_per_rank` is already `/tp`, `train_timing.py:783-787`) and gathers from the **dp** group. Total fine traffic `= cp·ep·P_layer` is correct. `int()` is a no-op. |
| 5 | `local_comp_time` zeroing | The assignment was already dead in legacy; the same quantity is modeled once per stage as the `optimizer` node. Attaching it per dp edge would double-count by `layers_per_stage`. |
| 8 | `_pipeline_interleave_scale` multiplies the whole total | Owner-scoped closed form (DESIGN §6). Quantified bias, both making totals **too low**: the additive DP tail is scaled by `s<1`, and interleaving's `v`× p2p hop count is ignored. Worst at small `mb`, large `pp`, large `v`, large tail share (e.g. `pp=8,mb=8,v=4` → `s=0.65`; a 20% tail is under-counted by ~7% of total). Under ZeRO-3 + large dp the tail is **not** small, so the docstring's "small comm share" bound does not hold there. |
| 9 | Same-stage zero-byte PIPELINE events | The *counter* consumption is inert (monotonic ⇒ relative order preserved), but their **presence** is load-bearing: they pop at the current timestamp and drive the ready-scan that admits successors. |
| 10d | Softmax pinned to cluster rank 0 | Critical-path time is right (TP ranks would do this concurrently); occupancy is wrong. Same class as A2 — revisit together. |
| 10g | Singleton-group no-op; 1-byte controls | Required AstraSim-side workarounds, correctly one-homed in `et_emit`. |
| 10h | `b == 0` means "last microbatch" | Backward walks microbatches in reverse. Semantics correct; comment confusing. |

## C — inert (ordering/naming only)

`3` manifest records ALL_REDUCE as `-1` (collision unreachable — `_collective_enum` raises on `None`;
`-1` only sorts it first) · `7` Step-11 pre-SCOTCH-remap iteration (**endpoints are correct**; the
remap never changes stage identity, only `send_seq`/`recv_seq` → node-id priority) · `10e` non-unique
op ids from overlap head splits · `10f` `_offset_stage_device` raises out-of-range where legacy
produced a phantom device (equivalent in range; docstring overclaims equivalence).

## D — needs an experiment or an owner call

| id | Question | Experiment |
|---|---|---|
| 10b | **Optimizer apply-grad modeled once per pipeline *stage*, not per layer.** `comp_times["optimizer"] = get_data_parallel_reduction_llm(...)` is one layer's params (`train_timing.py:2949-2971`), emitted once per stage — under-counts weight-update compute by ~`ceil(L/pp)`×. | Owner call on the intended semantics. Mechanical check: `pp=1,L=32` vs `pp=8,L=32` — if the optimizer contribution is identical, it is per-stage. If per-layer, totals **increase**, most at small `pp`. Lives in `train_timing.py` (out of restructure scope). |
| 10c | **Cross-layer activation bytes split by `par_degree`.** Correct under TP+SP and CP; under **plain TP** the boundary activation is *replicated*, so real aggregate is `tp`×; under EP it is not expert-sharded, so `/ep` under-counts. | A/B a `pp=2, tp=4, tp_sp=0` spec with and without the `/tp` division against the existing nanotron sweep harness (used for commit `3d0e0f2`). If replication is real, pp traffic is under-modeled by `tp`× and totals **increase**. |
| 1(ep) | ZeRO-3 gather over the `ep` axis: `_param_stats_per_rank` never divides by `ep`, so each of `ep` ranks gathers un-sharded expert params (over-count `ep`×). | Only settleable once flattened MoE is enabled (`llm_execution.py:697`). Unreachable, not wrong, today. |
| A2 | Whether rank-0-only DP attach matters in practice at studied scales. | Compare AstraSim **dp-dimension link utilization** (per-link busy time) between the current and `par_degree`-way variants, not end-to-end totals. |
| **11** ⚠ NEW (P5 verification, 2026-07-29) | **Nothing orders the optimizer after its stage's GRADIENT REDUCER.** Measured on the whole matrix: 0 of 9 (COARSE) / 0 of 18 (FINE) `DepClass` paths run from a grad collective to the `OPTIMIZER` op, at HEAD **and at HEAD~1**. The weight update is therefore modeled as concurrent with the all-reduce that produces the gradients it applies. | Preserved legacy behavior, so it is not a regression and P5 does not touch it: **R5** (INTERFACES §4.3 amendment 2026-07-29) makes the optimizer wait for every backward COMPUTE item of its stage and deliberately stops there. Owner call: if the optimizer must also wait for the reducer, add the reducer's `SyncKey` to R5's source set. **Expected delta:** totals **increase** wherever the dp reduction is not already hidden — largest at large `dp` and small `pp`, i.e. exactly the ZeRO rows. |
