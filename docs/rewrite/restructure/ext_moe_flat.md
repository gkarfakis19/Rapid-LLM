# Verdict: for this feature the rewrite bought a *diagnostic*, not a *generalization*. Ratio ≈ 1.0 : 1.

## Q1: "Is the fine builder's MoE template path already complete because memory replay uses it?"

**No. That inference is unsound, and the code says so in its own docstring.**

`program/block.py:30-35` is a written confession:

> "Dense and MoE variants are two templates; the flattened expansion treats both as **serial per-rank chains, exactly like the legacy flattener**, while the BLOCK builder (`program.block_program`, M4) *additionally* realizes the MoE hot/cold joins + residual transfers … (the flattened path still rejects MoE upstream)."

The memory replay cannot possibly validate the MoE comm topology, because it never looks at it. `program/memory_sim.py:517-523` is the entire `FineEdge` handling in the replay: enqueue at `time + event.duration`, pop, done. It reads no `comm_type`, no `participants`, no `local_hw_id`, no group, no `moe_component`. Worse: transformer-template edges are created with `duration=0` unconditionally (`pipeline_fine.py:742`, the `local_comp_time`-zeroing port), and the `events_hook` conversion only touches *coarse* events (`pipeline_fine.py:889-895`), so MoE comm edges contribute **literally zero** to the memory result. What the memory path actually exercises of the MoE template is: the entry list, `mem_kind_from_op_name`, durations, and the `is_moe_layer` flag. That's it.

And `pipeline_fine.py:516-595` is a **line-for-line transplant** of `git show 85894c6:llm_execution.py:690-800` — same `is_moe_layer` template select, same `for comm_key in cfg.comm_keys: previous.add_child(edge); previous = edge`. Diff them: only the type names changed.

## Q2: What actually blocks it — six sites, all concrete

**1. The guard.** `llm_execution.py:596-597`. Two lines. Also the dead branch at `:620-621` (`# pragma: no cover - rejected above`).

**2. `placement` is ignored.** `pipeline_fine.py:577-589` chains every comm key *after* the GEMM. `placement: "pre"|"post"` is a **generic, config-driven** field (`train_timing.py:4824-4869` reads it per comm rule from the op-model), and `CommMeta.placement` exists (`block.py:69`). Only `block_program.py:149-171` (`_split_comm_keys`) honors it — as a **nested closure**, unreachable from the fine builder.

**3. `parallel_group` / `moe_component` / `moe_routing_mode` are ignored.** They are typed on `CommMeta` (`block.py:70-72`), populated at `train_timing.py:596-618`, and consumed *only* by `block_program.py:285-372`. The fine builder never reads them. So MoE flattened would emit `base_all_to_all` **and** `residual_p2p` as two serial same-rank edges on every rank — no hot rank, no cold rank, no join.

**4. The residual p2p is silently deleted — no error, no warning.** `residual_p2p` is `CollectiveType.PIPELINE` (`train_timing.py:611`). In the fine chain its parent and child are both the same rank's GEMMs. Trace it: `legacy_lowering.py:354` skips PIPELINE in stage attribution → `legacy_lowering.py:441-451` sees `src.hw_id == stage` → `same_device_edges` → `legacy_lowering.py:1062-1090` makes a same-device `TransferOp` → `et_emit.py:351-352` `continue  # same-device transfers are elided`. The bytes are gone. Nothing catches it, because `program/validate.py:31-38` was *deliberately widened* to legalize exactly this shape: "same-device transfers may carry **any size** — the size is preserved for the M5 evaluator and **never emitted**." The rewrite's own safety net was relaxed into blindness for this failure mode.

**5. The EP-sync collective is structurally impossible in the fine builder as written.** `program/schedule.py:934-950` emits one `transformer_moe_ep_sync` `CommEvent` per (mb, layer) with `interconnect_type: 'ep'` (`train_timing.py:4445-4451`), fired when `spec.ep > 1 and spec.dp > 1`. In `_expand_transformer_node` it is not `"dp"` (`pipeline_fine.py:602`) and not `PIPELINE` (`:641`), so it falls to `pipeline_fine.py:713` `self._attach(downstream_parents, child_clone)` — **one** `FineEdge`, attached to all rank tails, placed on **one** device by `propagate_local_hw_ids`. Its communicator is the whole ep axis (`legacy_lowering.py:155-180` + `:230-260`). One member issues an all-reduce; the others issue nothing.

  Note the contrast that exposes the non-generality: the identical single-instance attach is *correct* for the dp reducers (`pipeline_fine.py:634-638`, "`rank_tails[0]` ONLY — legacy asymmetric quirk"), because a dp group's members are the *same device at different dp indices* and emission stamps the op per-dp. For an `ep` group the members are *different devices at the same dp index*, and nothing replicates it. The builder has one attach policy and two group geometries. Fixing it needs a **third** copy of "explode a coarse comm event into `par_degree` per-rank clones" — the first two being `pipeline_fine.py:640-698` (PIPELINE) and `pipeline_fine.py:317-395` (`_ensure_zero3_per_rank_edges`). None of the three is a shared mechanism.

**6. `is_moe_layer` is dropped by the overlap transform.** `OVERLAP_NODE_COPY_ATTRS` (`pipeline_fine.py:181-191`) omits it, documented at `pipeline_fine.py:161-166` and `transforms.py:151` as "Bug-compatible by design." Dormant-ish today; live the moment flattened MoE + `tp_overlap` runs.

## The concrete diff plan (current code)

| Site | Change | ~LOC | New branch? |
|---|---|---|---|
| `llm_execution.py:596-597,620-621` | delete guard | −3 | — |
| `block_program.py:149-186` → `block.py` | hoist `_split_comm_keys`, `_comm_parallel_groups` out of the `build_block_root` closure (they capture `comm_metadata`/`_comm_meta`); thread the mapping as a param | +45 / −40 | widen |
| `pipeline_fine.py:536-539` | flat-rank → `(tp,cp,ep)` coords. `_hw_id_for_rank:275-291` does this via `cluster_coords` **only when a layout exists**; the `layout is None` branch (`:275`) has no coords at all | +12 | yes |
| `pipeline_fine.py:577-589` | placement split + `_comm_parallel_groups` dispatch | +10 | yes |
| `pipeline_fine.py` new `_attach_moe_parallel_post_group` | port of `block_program.py:285-372` | +70 | yes |
| hot-rank in **fine device space** | `block_program.py:223-235` returns a *block-local* rank (`tp + cp*tp + ep*tp*cp`); fine devices come from `layout.linearize(coords)` (`pipeline_fine.py:277-291`). **Not reusable** — needs a coordinate-space variant (`coords with ep=0`, or `tp=0,ep=0`) | +15 | yes |
| join-node key | `block_program.py:237` `moe_parallel_joins` is a per-build dict because it builds **one** layer; the fine builder builds mb × layer × direction × stage copies → token needs 4 more components. Sharing the helper means changing `block_program`'s key shape | +6 | yes |
| hot-before-cold ordering | `block_program.py:355-361` relies on the `ep` outer loop. The fine loop is flat `for tp_rank in range(par_degree)` with tp fastest, so ep=0 comes first — it happens to hold for both `"ep"` and `"tp_ep"` modes, **by coincidence of two unrelated loop orders**. Needs an assert | +4 | yes |
| `_create_transformer_comm_edge:727-762` | add the `name=` override (`{key}_rank{r}_to_hot{h}`) that `block_program.py:196` already has | +3 | — |
| `pipeline_fine.py:601-606,713` | ep-sync per-rank exploder (third copy) | +35 | yes |
| `program/validate.py` | new invariant: a grouped `CollectiveOp` must be instantiated on **every** member device. V5 (`validate.py:39-45`) explicitly defers this to the emitter | +20 | yes |
| `pipeline_fine.py:181-191` | decide on `is_moe_layer` (fix ⇒ memory golden regen, or keep the bug) | ±1 | — |
| `equiv/configs.py:201-215` | flattened MoE rows (currently `for backend in ("hybrid","hierarchical")`) + first-time golden capture | +10 | — |

**Total ≈ +300 LOC across 6 files, ~7 new special-case branches**, plus a judgement call on a pinned bug and new goldens.

## The counterfactual against 85894c6 — the same six sites

- `llm_execution.py:1631` — delete guard. −2.
- `PipelineGraphFlattener._expand_transformer_node` (`git show 85894c6:llm_execution.py:690-899`) — **identical** placement split, parallel grouping, hot/cold join, residual rewiring. Legacy `_hw_id_for_rank` had the same layout/flat-arithmetic split. ≈ +140.
- `_create_transformer_comm_edge` (`85894c6:llm_execution.py:901`) — name override. +3.
- MoE helpers: `_split_comm_keys` / `_comm_parallel_groups` / `_moe_hot_rank` / `_attach_moe_parallel_post_group` are **nested closures inside `Graph.construct_transformer_graph`** (`85894c6:simulate_train_graph.py:1336 / 1360 / 1404 / 1466`) — exactly as non-reusable as `block_program.py`'s. Same hoist-or-copy, ≈ +85.
- ep-sync single-instance attach — same bug at the same place, same ≈ +35 fix.
- `astrasim_lib/executor.py:266-347` `_assign_collective_labels` already had the `("tp","ep")` composite-axes case (`:306-317`) — **no edit in either world** (it's the same code; `legacy_lowering.py:236-244` is a verbatim port).
- Validation: nothing existed to extend. Goldens: same work.

**≈ +290 LOC across 4 files, the same 6 sites, the same 7 branches.**

## Ratio and the honest scoreboard

**≈ 1.0 : 1 on edit count, edit sites, and new branches.** The new core does not absorb this feature anywhere. Two genuine deltas, both small:

1. **Blocker 5 fails loudly.** `et_emit.py:485-534` raises `group-order postcondition violated`. In legacy the same construction produced a communicator where one member issues a collective and the others don't → the silent AstraSim deadlock documented in `CONTEXT.md` (exit 0, no wall-time lines). This is real, and it is the *only* thing the rewrite bought here.
2. `CommMeta` is typed, so the MoE fields are named attributes instead of `metadata.get("moe_component")`. Cosmetic at the use site.

Against that, one **regression** in defensibility: blocker 4 (the vanishing residual p2p) is *harder* to catch now than before, because `validate.py:31-38` was deliberately widened to bless nonzero-byte same-device transfers, and `et_emit.py:351` drops them without a word.

## Collateral evidence for the owner's fear (same lens, not the feature)

- **Name-based dispatch survived, sitting next to the typed field that replaced it.** `ComputeEvent` has `role`, set to `"softmax"`/`"optimizer"`/`"embedding"`/`"layer"`/`"recompute"` (`schedule.py:586,594,611,700,709,725,741,1070`). `pipeline_fine.py:413` uses `obj.role in ("layer","recompute")` — and then the very next branches are `pipeline_fine.py:417` `if "linear_softmax" in obj.name:` and `:426` `elif "optimizer" in obj.name:`, plus `:705` `and "optimizer" in child.name`. The rewrite built the type and then didn't use it. Also `transforms.py:224` `"attention" in str(child.name).lower()` (CP-overlap target selection, on *template-supplied* names) and `pipeline_fine.py:355` `"bwd" in edge.name` (kept as a fallback next to the typed `zero3_offset_hint`).
- **The `hw ± par_degree` ZeRO-3 hack is retyped, not removed.** `pipeline_fine.py:293-306` `_offset_stage_device` — a ±1 pp-coordinate hop that is "identical because the canonical layout's pp stride equals `tp*cp*ep`". Same hack, better comment.
- **Two builders, one template, zero shared chain code.** `block_program.py:365-500` and `pipeline_fine.py:536-595` both enumerate "per rank: per GEMM entry: node + comm edges" from the same `BlockTemplate`, and share nothing. That duplication *is* the blocker.
- **`legacy_lowering.py` (1095 lines) is permanent by design**, per its own docstring and `CONTEXT.md` ("its name records its converter lineage"). The 1100-line converter wasn't replaced by a general mechanism; it was renamed, kept, and declared load-bearing because the pinned emission order is defined over children-list adjacency order that the typed op list "cannot represent."

**Bottom line for the repo owner: your fear is correct for this feature.** The fine builder's MoE path is the legacy flattener's MoE path with new type names; memory replay's use of it proves only that the GEMM chain doesn't crash. Enabling flattened MoE requires the same seven scattered edits it required in 2025, in the same seven conceptual places — the difference being that one of them now throws an exception instead of hanging AstraSim, and another now hides a dropped collective behind a deliberately-relaxed invariant.