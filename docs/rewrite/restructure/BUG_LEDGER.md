# Bug ledger — evidence-backed quirk classification (restructure P1)

Produced by an evidence-first audit against `git show 85894c6:*` (the pre-rewrite originals) and the
live code at `03ace0e`. Classes: **A** confirmed bug (fixing SHOULD change output) · **B** deliberate
modeling choice (preserve) · **C** semantically inert (ordering/naming; may perturb AstraSim wall
clock via node-id priority) · **D** unclear, needs an experiment or an owner call.

> **Reframe that removes four constraints:** the ZeRO-3 per-rank gather machinery is **dead in the
> golden matrix**. `equiv/configs.py:163-174` builds zero2/zero3 specs only at `dp=2, tp=1, cp=1`, so
> `par_degree == 1` and `_should_shard_zero3_transformer` (`program/pipeline_fine.py:307`) always
> returns False. Items 1, 4, 5(10a) and 10f are **unvalidated code, not pinned behavior** — no golden
> constrains them. Treat that region as new development with new goldens.

## A-list (confirmed bugs, ranked by impact)

| id | Bug | Site | Correct behavior · expected delta |
|---|---|---|---|
| **A1** | **AstraSim result-cache key omits all DAG structure.** `cache_key = sha256(manifest ‖ system ‖ network ‖ remote_mem ‖ comm_groups)`; the `.et` files are never hashed, and the manifest is a *sorted multiset* with no deps, no order, no p2p peer, no tag. | `astrasim_lib/integration.py:456-466`, `_hash_file_bundle:140-153`; manifest `program/et_emit.py:560-606` | Two workloads with equal op multisets but different dependency structure / priority order / p2p pairing **collide** and a stale wall time is returned silently. Hash the emitted ET bytes (or add deps+peers+order to the manifest). **No model change; makes staleness loud.** ⚠ This masks exactly the kind of change the restructure makes — fix FIRST. |
| **A2** | **DP collectives attached to `rank_tails[0]` only.** One cluster rank bears the whole grad all-reduce/reduce-scatter; the other `par_degree-1` ranks emit nothing. | `program/pipeline_fine.py:616-629`, placement at `:829-849` | Correct: one dp collective per (stage, cluster-rank) on `par_degree` disjoint dp-axis communicators. Fixing **increases** dp traffic by `par_degree`×. Error is in **contention, not the serial critical path** (max-over-ranks may hide it) — measure per-link busy time, not just totals. |
| **A3** | **tp-overlap head drops `is_moe_layer` → memory DOUBLE-COUNT** (not misclassification): tail files layer L under moe, head files the same L under dense, and the census sums both. | `program/pipeline_fine.py:186-196` (`OVERLAP_NODE_COPY_ATTRS`), census `program/memory_sim.py:352-368`, `:412`/`:432` | Add `is_moe_layer` to the overlap copy list. Reported peak/static memory **decreases** by one dense layer's worth per split MoE layer per device; can flip capacity warnings. **Timing unaffected.** Path is live (`llm_execution.py:776-787` passes MoE templates *and* tp_overlap). |
| **A4** | **Collective-only stage rank collision.** `rank = dp_idx*num_stages_initial + stage_idx` with a pre-extension count: `rank(first_ext, dp=0) == rank(devices[0], dp=1)`. | `program/legacy_lowering.py:327`, `:604-613`; `program/ir.py:235-242` | Asserted verbatim today in `tests/test_program_ir.py:490-493`. At dp≥2 the collective is silently appended to another replica's trace. Correct: use the post-extension device count, or refuse collectives on non-compute devices. **Dormant — no reachable config found** (all ZeRO-3 ±pp hops are one-stage between adjacent host/target pairs, and `program/layout.py:113-114` bounds-checks). |
| **A5** | **ZeRO-3 dead store with live side effects.** `per_rank_edges = self._ensure_zero3_per_rank_edges(...)` is assigned and never read (unlike the sibling at `:717-721`), yet it consumes `par_degree` op ids (shifting emission priority) and appends the discarded edges to real nodes' `.parents`. | `program/pipeline_fine.py:620-629` | Benign only *by accident* (the lowering's `_collect_objects` is children-only, so the orphans are never discovered). Also a clone-cache hazard: a gather reachable as both `dp_child` and `zero3_attachment` gets whichever expansion runs last. Delete or use. **Unreachable today.** |

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
