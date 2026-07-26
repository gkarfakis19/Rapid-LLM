# Fresh-eyes review of `program/` — rewrite_astra

I read `program/` cold, then the docs, then `git show 85894c6:` for the originals. Verdict first, then evidence.

## Verdict

**The owner's fear is substantially correct, but not uniformly.** One module (`et_emit.py`) is a genuine general mechanism that replaced a pile of re-inference. The rest of the "typed IR core" is a **second skin over the legacy graph, not a replacement for it.** The legacy `Node`/`Edge` mutable DAG was not deleted — it was **forked into three classes** (`ComputeEvent`/`CommEvent`, `FineNode`/`FineEdge`, `BlockRoot`) with the same `parents`/`children`/`hw_id`/`local_hw_id`/`op_id` surface, and the 1100-line converter was **renamed, not removed** (`program/legacy_lowering.py`, 1095 lines / 834 SLOC). Every consumer that actually does work — the analytical evaluator, the memory replay, the emission ordering pass — reads the *graph*, not the `Program`. The `Program` is a duration lookup table and an emission worklist.

Net size: legacy `simulate_train_graph.py` + `llm_execution.py` + `executor.py` = **5,324 SLOC**; today `program/` + `llm_execution.py` + `executor.py` = **6,501 SLOC**. The rewrite is ~22% *bigger*, spread over 3× the files, and `docs/rewrite/CONTEXT.md` calls it a deletion of ~5,600 LOC.

---

## The conceptual model I infer from the code alone

Reading only `program/`, this is what I believe the system is:

> A configuration is distilled into a `ScheduleSpec`, which is expanded into a **mutable event DAG** whose *object creation order* and *`children`-list append order* are the load-bearing specification. That DAG is either (a) walked by an event-loop simulator to produce a time, or (b) fed to a graph-analysis pass that re-infers stages, dependencies, communicator groups and a per-stage topological order, and flattens the result into a numbered op list whose index order *is* the Chakra ET emission order. A `Program` object exists alongside the DAG and mirrors part of it.

Notice what is **not** in that model: the `Program` IR. I could delete `ComputeOp.mem_kind`, `.recompute`, `.param_gather`, `.micro_batch`, `.layer` and nothing outside `pipeline_coarse.py` would notice — those fields have **zero readers** (`program/ir.py:124-129` written only at `program/pipeline_coarse.py:186-188`; grep for `op.mem_kind`, `op.recompute`, `op.param_gather`, `op.micro_batch` across the tree returns nothing). And `legacy_lowering.py:809-816` — the constructor for **every** FINE, BLOCK and hierarchical `ComputeOp` — sets only `uid, name, device, duration, deps, legacy_op_id`. So on the paths that actually reach AstraSim, the "typed placed-operation IR" carries no semantics at all.

The docstrings describe a different system than the code implements. `ir.py:16-24` promises "a list of placed operations… uid order is the global total order". `analytic_sim.py:211-247` and `memory_sim.py:450-524` both run their event loop over `meta.misc["coarse_proto_root"]` / `["fine_proto_root"]` and consult `program.ops` only to look up a float.

---

## The five things that confused me longest

**1. Why there are two `role` vocabularies and neither is used for dispatch.**
`ComputeEvent.role` is a *string* set at `schedule.py:586,594,611,700,709,725,1070` (`"softmax"`, `"optimizer"`, `"layer"`, …). `OpRole` is an *enum* in `ir.py:73-86`. `pipeline_coarse.py:94-102` maps one to the other. Then `pipeline_fine.py` dispatches on the string role for two cases and on **substring matching of the node name** for the rest, in the same `if/elif` chain:

```python
if obj.role in ("layer", "recompute"):        # pipeline_fine.py:412
...
if "linear_softmax" in obj.name:              # pipeline_fine.py:417
elif "optimizer" in obj.name:                 # pipeline_fine.py:426
...
and "optimizer" in child.name                 # pipeline_fine.py:705
```

`obj.role` is literally `"softmax"` and `"optimizer"` at those lines. The typed field was added and then not used. This is the single clearest instance of "hack transplanted into a new type". Same pattern at `transforms.py:224` (`"attention" in str(...).lower()`) and `pipeline_fine.py:355` (`"bwd" in edge.name` — kept as a live fallback *underneath* the new `zero3_offset_hint` field the schedule stamps at `schedule.py:446-447`, 870, 1026).

**2. Which of the two graphs I was looking at.**
`ComputeEvent`/`CommEvent` (`schedule.py:304-460`) and `FineNode`/`FineEdge` (`pipeline_fine.py:82-150`) are structurally identical, both duck-typed, both with `add_child`, both consumed by `lower_to_program` through `getattr`. Neither uses `__slots__` consistently (`ComputeEvent` does, `FineNode` doesn't), so `FineNode` grows attributes by assignment scattered across three files — `tp_rank`, `stage_id`, `is_moe_layer`, `tp_shard`, `cp_rank`, `ep_rank`, `local_hw_id`, `is_cross_layer` (`pipeline_fine.py:670`), plus `done`/`scheduled`/`finish_time` bolted on later by `memory_sim.py:246-248`. There is no one place that tells you what a `FineNode` has.

**3. `program/legacy_lowering.py`.** 1,095 lines named after the thing it was supposed to retire, containing: DFS collection, stage inference, dp-major rank arithmetic, axis-layout recovery, gmap traffic collection, per-edge stage attribution, **two near-duplicate 60-line dependency walkers** (`analyze_for_compute` at :412 and `analyze_for_collective` at :477 — the diff between them is ~6 lines), collective label assignment, per-stage Kahn toposort, SCOTCH remap, uid assignment, Step-11 transfer replay, and same-device transfer synthesis. `CONTEXT.md` constraint 3 says "One placement/scheduling source of truth: no re-inference of stages, groups, or dependencies downstream of construction." This file **is** the re-inference, and `pipeline_fine.py:48-53` declares it permanent. The FINE builder knows every rank, every stage, every dependency at construction time and then throws that away so a walker can rediscover it from `parents`/`children`.

**4. The determinism argument.** `legacy_lowering.py:646-671` iterates `stage_tasks[stage]` — a **`set` of objects** — to seed a heap, tie-broken by an insertion counter, and rationalizes it thus:

```
# Tasks/neighbors are iterated in RAW set order like legacy:
# op_ids are NOT unique (the tp-overlap head split reuses its source's
# op_id) ... Our walkers insert the same objects in the same sequence
# as legacy's, so raw iteration reproduces the legacy tie-break exactly
# in-process.
```

The invariant being preserved here is *bit-parity with code that no longer exists in the tree*. "in-process" is a tacit admission that cross-process reproducibility is not argued. Meanwhile `_replay_step11` at `legacy_lowering.py:1039-1049` hit the exact same tie and **did** fix it (by falling back to `task_uid`), with a comment explaining why the legacy behavior "is not reproducible across graph instances." The same defect, two hundred lines apart, resolved two different ways. `DESIGN.md` §"Constraints" 2 demands "determinism BY CONSTRUCTION."

**5. `meta.misc` as a type system.** `Program` is really three different types discriminated by `meta.misc["granularity"] in {"coarse","fine","block"}`, checked by string comparison in five places that raise `RuntimeError`: `analytic_sim.py:141`, `memory_sim.py:69`, `retime.py:88`, `et_emit.py:180`, `pipeline_coarse.py:308`. A COARSE `Program` is not emittable and `et_emit` has a special-case bail-out for it (`et_emit.py:180-188`) *because it would otherwise fail validation with a confusing message*. That is name-based dispatch that moved from node names to a dict key.

---

## Representations you must hold to follow one run (config → simulated time)

For `full_astrasim_flattened`, **eleven**, with two live simultaneously for most of the run:

1. `time_calc` (`train_timing.TimeCalculation`)
2. `ScheduleInputs` — 5 ints + three **untyped dicts** (`schedule.py:109-143`), one of them deliberately mutable for write-back
3. `ScheduleSpec` (frozen distillation of 2)
4. `BlockTemplate` / `GemmEntry` / `CommMeta`
5. `ComputeEvent`/`CommEvent` DAG
6. `FineNode`/`FineEdge` DAG (mutated in place twice more: overlap rewrite, `propagate_local_hw_ids`)
7. The lowering's eight intermediate maps (`compute_info`, `collective_info`, `edge_stage`, `pipeline_edge_map`, `stage_tasks`, `stage_adj`, `stage_order`, `task_uid`)
8. `Program` (which still carries 5 and 6 in `meta.misc`)
9. `RankLayout` **and** its `descriptor()` dict — serialized at `pipeline_fine.py:943` and re-parsed by `_extract_axis_layout` at `legacy_lowering.py:336` **in the same process**, then rebuilt into a `RankLayout` at `:831`
10. Per-rank `pb.Node` traces → `.et` + `manifest.json` + `comm_groups.json`
11. AstraSim stdout `sys[i], Wall time:` strings

Hybrid/hierarchical is worse: `_build_coarse_program` produces a `Program` that **cannot be emitted**, `retime` writes durations to *both* its ops and its events (`retime.py:124-127`), and `lower_coarse_for_emission` lowers the events into a **second, different `Program`**. Two `Program` objects from one build, with different uid orders, one of which is a decoy.

---

## Coherent job vs. grab-bag

**Coherent (would happily own):**
- `layout.py` (330) — genuinely unified three copies; the one place the rewrite did what it promised.
- `block.py` (182) — clean typed template.
- `et_emit.py` (644) — the real win. All AstraSim contract quirks in one file, with a loud always-on postcondition (`:485-533`) that converts the historical silent deadlock into an exception. This is the general mechanism the whole rewrite was supposed to be.
- `validate.py`, `shadow.py`, `viz.py`, `retime.py` — small, single-purpose.

**Grab-bag:**
- `legacy_lowering.py` — ten jobs (above).
- `schedule.py` (1079) — input carrier + spec + policy ports + two event classes + a 600-line graph constructor. `build_pipeline_events` is a near-line-for-line transplant of `construct_fwd_bwd_graph`, including `attach_parallel_edge` with `skip_non_comm_children`/`skip_comm_children` (`:553-567`, used at `:888`, `:891`, `:1030`, `:1032`).
- `pipeline_fine.py` (970) — proto classes + shared attribute constants *for another module* + the flattener + entry points.
- `transforms.py` (557) — two implementations of the same transform, one live (proto-level) and one with **zero production callers** (`apply_tp_overlap` + `_objectify`/`_successor_map`/`_renumber`, `:379-557`), retained "as the seed of the post-M9 API."

---

## Hostile list: specific things transplanted rather than replaced

| Claim in docs | Reality |
|---|---|
| "the 1100-line ET converter" deleted | `legacy_lowering.py` is 1,095 lines and permanent by design (`pipeline_fine.py:48-53`) |
| "name-based dispatch" replaced | `pipeline_fine.py:417,426,705`; `transforms.py:224`; `pipeline_fine.py:355`; `et_emit.py:135-139` |
| ZeRO-3 special cases generalized | `_ensure_zero3_per_rank_edges` (`pipeline_fine.py:315-395`) is byte-for-byte the legacy body: `transformer_mode: Any = ""` three-valued sentinel (`:333,345`), a branch kept "for fidelity" that the docstring admits never executes (`:329-338`), undivided bytes (`:360`), and the `±par_degree` hack merely re-expressed (`:290-303`, with the flat arithmetic still there as fallback) |
| ZeRO-3 + flattened special cases removed | Still an explicit sibling-scan through `parent.children` at `pipeline_fine.py:461-480` and `:608-613` |
| MoE special cases | `block_program.py:285-374` hot/cold rank join with `_moe_hot_rank`/`_moe_parallel_token` string-mode switches on `"ep"`/`"tp_ep"`; flattened MoE still rejected (`llm_execution.py:596`) |
| "typed placed-operation IR" | `ProgramBuilder` (`ir.py:250-399`, ~150 lines) has **no production callers** (its own docstring says so at `:255-257`); `OpRole.GEMM`/`JOIN` are stamped by nobody (`ir.py:81-86`); `Program.rank_for`/`device_index` are dead (`et_emit.py:199` defines its own); `TransferOp.is_control` is used only by a test while `et_emit.py:372` re-derives it inline |
| "id policies" abstraction | `emit_chakra(id_policy=...)` accepts exactly one value and raises `NotImplementedError` on any other (`et_emit.py:173-179`) |
| Bugs fixed | `pipeline_fine.py:156-193` deliberately preserves a **live bug**: `OVERLAP_NODE_COPY_ATTRS` omits `is_moe_layer`, so a tp-overlap head split of a MoE layer is misclassified as dense in `memory_sim`. Documented as "bug-compatible by design." |
| V6 (the invariant for the 85894c6 deadlock class) | Disabled on every production path — `validate_program(..., check_races=False)` in `legacy_lowering.py:851`, `et_emit.py:189`, `transforms.py:556` (`validate.py:21-26`) |

Also: `legacy_lowering.py:377-379` prints to stdout inside an error path in production code. And 22 docstring citations point at `executor.py:NNNN-NNNN` — line numbers in a file that no longer exists.

---

## Rating

**Legacy (`git show 85894c6:simulate_train_graph.py` etc.): 2/10.** One 2,300-line file, one `Graph` class, a 635-line `construct_fwd_bwd_graph`, a 1,100-line single-function converter with `# HACK!!!!` inline, zero validation, silent deadlocks.

**Current `program/`: 5.5/10.**

What genuinely improved and I would not give back: `et_emit.py` and its always-on group-order postcondition; `layout.py`; `validate.py`'s V1-V5; the `EmissionError` messages, which are the best-written thing in the tree (e.g. `et_emit.py:246-254`); `executor.py` shrinking 2,044 → 509 lines; the module-level docstrings, which are unusually honest about what they did and did not fix.

What keeps it from a 7: I would be inheriting **two** untyped mutable DAGs instead of one, plus a typed IR that shadows them without owning them; a 1,095-line re-inference pass declared permanent; and a correctness contract that is not "the schedule is right" but "the object creation order matches a deleted program." That last one is the ownership killer. If I am asked next week to add 1F1B, or interleaved pipelining as a real schedule instead of the closed form at `llm_execution.py:353-368`, I have to edit `build_pipeline_events` — and *any* change to `add_child` call order there silently changes emission order, hence ET node ids, hence AstraSim scheduling priorities, hence wall seconds. There is no invariant I can reason from; there is only a 42-spec golden matrix that will go red and tell me nothing about why.

The migration was executed with real discipline (shadow mode, per-stage differentials, zero golden regens). It was aimed at the wrong target: **preserving byte-parity with the deleted code became the design, and the design document's own M9 — the one commit where the legacy id policy dies and the hacks retire — is the only place where the actual rewrite happens.** Right now the tree is the "before" half of a two-step, with a locked doc asserting it is the end state (`__init__.py:18` "Post-migration (M8) module inventory").

**Cheapest high-value moves, in order:** (1) delete the name-substring dispatch at `pipeline_fine.py:417,426,705` — the `role` string is already there and this cannot change emission order; (2) merge `analyze_for_compute`/`analyze_for_collective`; (3) give `Program` an ordered successor list so `analytic_sim` and `memory_sim` can stop reading `meta.misc` proto roots, which is the single change that would make the `Program` load-bearing instead of decorative; (4) delete `ProgramBuilder`, `apply_tp_overlap`, `OpRole.GEMM/JOIN`, and `Program.rank_for` rather than carrying them as "reserved surface" — git remembers them.