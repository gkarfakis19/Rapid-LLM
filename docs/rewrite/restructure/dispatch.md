## VERDICT: the rewrite is conservative. The graph didn't die — it was renamed.

`program/` is 7,849 LOC. Roughly half of it (`legacy_lowering.py` 1,095 + `pipeline_fine.py` 970 + the event-graph half of `schedule.py` + `transforms.py` 557) is a mutable `parents`/`children` object DAG with `getattr(obj, "attr", default)` access — **197 `getattr` calls across `program/`**. The typed `Program` exists, but it is a *derived artifact*, not the substrate.

---

### 1. Every consumer's real input is a proto graph in `meta.misc`, not the Program

DESIGN §1: *"every consumer (ET emission, analytical evaluation, memory replay, retiming, gmap, faults, viz) is a pure function over it."* Actual:

| Consumer | What it actually reads |
|---|---|
| `analytic_sim.evaluate_detailed` | `program/analytic_sim.py:162` → `meta.misc["coarse_proto_root"]`; the event loop runs over events, ops are a `uid_of[id(event)]` side table (`:163`, `:169`) |
| `memory_sim.simulate_memory` | `program/memory_sim.py:260` → `fine_proto_root(program)`; the entire replay walks `FineNode.children` |
| ET emission (coarse/hier) | `pipeline_coarse.py:310` re-lowers `coarse_proto_root` through `lower_to_program` — the Program's own op list is discarded |
| `retime.apply_block_timings` | writes `op.duration` **and** mirrors to `events[op.uid].duration` (`retime.py:159`); the docstring at `retime.py:41-47` admits *"The mirror is load-bearing since M6"* — the typed field is the shadow, the untyped event attribute is the truth |
| `viz` | proto graphs |

`Program.devices` is a declared field, and `analytic_sim.py:180-192` still re-derives the device count by walking the graph for `max(hw_id)`. That is textbook re-inference of a value that was carried.

### 2. `legacy_lowering.py` — the re-inference engine survived, and was *promoted*

DESIGN §5 lists `legacy_lowering.py` under **M8 Deletes**. It is still here, 1,095 lines, and its docstring (`:16`) now reads **"PERMANENT — kept at M8"**. Its own header (`:31-55`) states it is *"an exact reproduction of converter Steps 1-7"* — the same function CONTEXT.md:112 diagnosed as the root cause: *"the converter re-infers a global schedule it was never given."*

Against DESIGN constraint 3 (*"no re-inference of stages, groups, or dependencies downstream of construction"*), what it still re-infers:

- **Stages** (`:323`): `stage_ids = sorted({node.hw_id for node in compute_nodes})` — the device set recovered by scanning a graph the builder placed.
- **Per-edge placement** (`:350-381`): `local_hw_id` → first parent's `hw_id` → first child's `hw_id`, with `print()`-then-`raise` diagnostics carried over verbatim from the converter.
- **Dependencies** (`:412-541`): both backward walkers, still two near-duplicate functions, still carrying the `(id(obj), via_collective)` visited discipline and the *"Legacy quirk: the parent object itself is recorded as the pipeline edge"* hack (`:503-505`).
- **Communicators by name** (`:593-598`): `tp_collective_groups[info["name"]].append(edge)` — group identity keyed on the edge **name string**, exactly as CONTEXT.md:73 described (`_assign_collective_labels`, gids from 1000).

The `GroupKey` type in `ir.py:93` is real, but it is *produced by* the name-grouping pass (`_group_key_for`, `:739`), not declared by a builder. Nothing was inverted.

### 3. Name-string dispatch: transplanted line-for-line into `_FineExpander`

`pipeline_fine.py:211` says it plainly: *"Port of `PipelineGraphFlattener` (llm_execution.py 339-994) ... The traversal, branch structure, op-id counter discipline, and attach orders are kept verbatim."* Diffing `_clone` against `git show 85894c6:llm_execution.py`:

| New | Legacy | Status |
|---|---|---|
| `pipeline_fine.py:412` `if obj.role in ("layer","recompute")` | `:524` `base_name.startswith("transformer_layer") or ...("vit_block")` | **fixed** — the only branch converted |
| `pipeline_fine.py:417` `if "linear_softmax" in obj.name:` | `:529` identical | **transplanted** |
| `pipeline_fine.py:426` `elif "optimizer" in obj.name:` | `:538` identical | **transplanted** |
| `pipeline_fine.py:705` `and "optimizer" in child.name` | `:884` identical | **transplanted** |
| `transforms.py:224` `"attention" in str(getattr(child,"name","")).lower()` | `:217` identical | **transplanted** |

This is not "hard to fix." `program/schedule.py` **already stamps the field**: `role="softmax"` (`:586`), `"embedding"` (`:594`), `"softmax_b"` (`:709`), `"embedding_b"` (`:700`), `"optimizer"` (`:1070`) — and `pipeline_coarse.py:94-102` already maps all seven role strings to `OpRole`. The declared field is sitting one attribute away and the FINE builder reads `obj.name` instead. One branch got converted; the author stopped.

### 4. The `"bwd" in name` hack is half-typed, and the live half is name-sniffing

CONTEXT.md:66 calls out the legacy `"bwd" in name` direction hack by name. `pipeline_fine.py:353-356`:

```python
hint = getattr(edge, "zero3_offset_hint", None)
if hint is None:
    hint = "bwd" if "bwd" in edge.name else "fwd"
```

The `zero3_offset_hint` field is set at `schedule.py:870, 881, 1026` — only on `zero3_transformer_gather` events. And the schedule then **constructs names with `_fwd`/`_bwd` suffixes** (`schedule.py:864`, `:1022`, `:857`) so the fallback would still work. Today `tp_shard` is only true for `zero3_transformer_gather` (`train_timing.py:4496`), so the fallback is dormant — but the hack was not removed, it was kept alive behind a field that is only populated on the path that doesn't need it. The zero3 *embedding* gathers get no hint at all.

### 5. `FineNode`/`FineEdge` are open attribute bags — a regression vs. the coarse events

`ComputeEvent`/`CommEvent` (`schedule.py:312`, `:399`) have `__slots__`. `FineNode`/`FineEdge` (`pipeline_fine.py:82`, `:115`) do not, and 11 attributes are attached from outside the constructor: `direction, is_cross_layer, is_moe_layer, layer_index, local_hw_id, micro_batch_index, param_gather, recompute, stage_id, tp_rank, tp_shard`. That is *why* `pipeline_fine.py` needs 49 `getattr` calls and `transforms.py` 24. The docstring calls them *"typed stand-ins"*; they are `Node`/`Edge` with the serial numbers filed off.

Related: `GemmEntry` (`block.py:134`) carries no `mem_kind`. Both builders recover it from the entry **name string** — `pipeline_fine.py:560` and `block_program.py:408`: `mem_kind_from_op_name(entry_name)`, a dict lookup that `raise`s on an unknown name. The template's `name` is a load-bearing dispatch key.

### 6. `OpKey` — the identity mechanism DESIGN made mandatory — was never built

DESIGN §2.1 opens with *"`OpKey` gains a `device` component (critic gap 13)"*; §4 says *"`retime` writes `duration_by_dp` via `OpKey` lookups."* `grep -rn 'OpKey\|duration_by_dp' program/ tests/` returns **zero hits outside `docs/rewrite/panel/`**. Retime instead uses positional alignment `events[op.uid]` (`retime.py:159`) plus a `(dp_idx, op.device)` tuple lookup. Stable semantic identity — the thing that was supposed to replace name matching for retime/memory/fault lookups — does not exist.

Similarly `ProgramBuilder` (`ir.py:250`) documents itself: *"No production builder constructs Programs through this class today."* The typed construction API is test-only scaffolding; both real builders assemble `Program(...)` from graph walks.

### 7. Residual duck-typing at declared-type boundaries

- `retime.BlockTimings` (`retime.py:65`): *"Entries are duck-typed timing pairs with `forward`/`backward` attributes."* Declared duck-typing at a brand-new interface.
- `ScheduleSpec.from_pipeline_graph` (`schedule.py:186`): *"duck-typed attr reads"* against `ScheduleInputs`, which is a `@dataclass` five lines up with exactly those fields. `misc.get("num_layer", getattr(pipeline_graph, "num_layer", 0))` (`:196`) and the same for `num_batch` (`:199`) are dead legacy fallbacks — `ScheduleInputs` declares neither attribute.
- `transforms._mode_label` (`:110-114`): `for attr in ("value","name"): if hasattr(mode, attr)` on a `ParallelismMode` enum, then string-compares `== "tensor_sequence"`.
- `et_emit.renumber_control_priority` (`:136`, `:139`): classifies nodes by `name.endswith("_send_control")` — re-parsing a suffix the same function wrote 250 lines earlier from `is_control` (`:372`, `:380`). Value recovered by string search that was carried.
- MoE dispatch in `block_program.py:305-313`, `:224-235` is metadata-driven (`moe_component`, `moe_routing_mode`) — genuinely better than the legacy hot/cold name joins — but compares raw strings `"base_all_to_all"`/`"residual_p2p"`/`"ep"`/`"tp_ep"` with no enum.

### 8. Stale determinism justification in the Kahn toposort

`legacy_lowering.py:646-652`:

> *"Tasks/neighbors are iterated in RAW set order like legacy: op_ids are NOT unique ... **Our walkers insert the same objects in the same sequence as legacy's, so raw iteration reproduces the legacy tie-break exactly in-process.**"*

Legacy was deleted in M8. The invariant this comment appeals to no longer has a referent, and the code still iterates `stage_tasks[stage]` (a `set` of identity-hashed objects) at `:661` and `stage_adj[stage].get(task, set())` at `:668`. The same author fixed exactly this hazard in Step 11 (`:1039-1049`, tie broken by `task_uid`) and left it in Step 10. DESIGN §1 demands *"emission is fully deterministic."*

**Honesty on this one:** I tried to make it bite and could not. I confirmed 256 duplicate-`op_id` tasks per stage under `tp_overlap=0.5` (the `_split_tp_node_fine` head reuses its source's `op_id`, `transforms.py:200`), then re-ran `lower_to_program` with `for task in sorted(tasks, key=lambda t: -id(t))` and reversed neighbour iteration across **all 15 flattened golden specs × {no overlap, 0.5 overlap}** — every Program hash identical. The head→tail edge means tied ops are never simultaneously in the heap. So this is a latent hazard with a false justification, not a live bug. Worth noting that **no golden spec sets any overlap fraction** (`grep overlap equiv/configs.py` → nothing), so the entire `transforms.py` proto-rewrite path — 557 LOC, including the op-id-reuse quirk — is ungated by the 42-spec matrix and only covered by one synthetic unit test.

---

## What actually got better (unpadded)

1. **`_check_group_order_postcondition`** (`et_emit.py:487-540`) plus the label→two-member-sets `EmissionError` at `:248`. This is new, always-on, and converts the historical silent AstraSim deadlock into a loud pre-run failure. Best thing in the rewrite.
2. **Layer dispatch by `role`** — `startswith("transformer_layer")/("vit_block")` (legacy `:524`, `:2032`) is genuinely dead; `retime.py:96` selects on `OpRole.TRANSFORMER_LAYER`.
3. **Memory classification by `mem_kind`** — `memory_sim._is_transformer_block` (`:282`) uses `MemKind` membership; legacy `memory_estimation.py:707-721` `_is_non_flattened` sniffed `name.startswith("transformer_layer")`. Replaced by `meta.misc["granularity"]`.
4. **`TransferOp` as one object with two endpoints** (`ir.py:158`) makes the tag-divergence deadlock class unrepresentable. Real structural win.
5. **MoE hot/cold joins via `moe_component`/`parallel_group`** in `block_program.py` rather than name matching.

## Bottom line

The owner's fear is correct in substance. What landed is **a typed façade over a preserved object graph plus a verbatim converter**. The ET emitter is a genuine rewrite; the postcondition and `TransferOp` are genuine mechanism replacements. Everything upstream of emission — construction, placement, dependency derivation, communicator identity, and now permanently the emission ordering — is the legacy machinery with new class names, and the commit that was supposed to delete it (`M8`) instead re-labeled it "PERMANENT."

The single most damning artifact: `pipeline_fine.py:412-426`, where `obj.role` and `obj.name` are read three lines apart, for the same dispatch, in the same `if/elif` chain.