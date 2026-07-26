# Verdict: the owner's fear is correct on this axis. Ratio ≈ 1:1.

The schedule/event layer does **not** admit a second generator. Four consumers absorb it for free — but they absorbed it for free in the legacy tree too, so the rewrite bought nothing there.

## What `build_pipeline_events` actually is

`program/schedule.py:478-1079` is a 602-line function; **60% of its non-comment lines appear verbatim in `git show 85894c6:simulate_train_graph.py` lines 688-1322** (after normalizing whitespace, `self.`→`spec.`, `_stamp(`). Same loops, same variable names (`gpu_index`, `first_transformer_layer`, `layer_entry_nodes`), same comments ("bwd flips mbs, so we attach to first one, not last"). `program/__init__.py:24-26` names it: *"the single **GPipe** schedule enumeration every builder consumes."* GPipe is baked into the shared abstraction by name.

---

## CONCRETE DIFF PLAN — new code

### Absorbed with zero edits (real, but not new)
| File | Why | Legacy equivalent |
|---|---|---|
| `program/analytic_sim.py` | generic list scheduler over the event DAG | `Graph.simulate` — also 0 edits |
| `program/memory_sim.py:322-373` | census keyed on `mem_kind`/`layer_index`/`is_moe_layer`/`fwd`, not names | **byte-identical** to legacy `simulate_memory._collect_graph_layout` (legacy_stg.py:1880-1895). Zero credit. |
| `program/pipeline_coarse.py` | 1:1 typed projection of whatever events exist | n/a (new, genuinely nice) |
| `program/retime.py` | keyed on `role`/`device`/`direction` | legacy name-prefix walk **would** have needed no edit either |
| `program/et_emit.py` | consumes uid order + `send_seq`/`recv_seq` | converter also shape-driven |

### Must be widened / branched (14 sites, 8 files)

1. **`schedule.py:163,206,229-238`** — `layers_per_stage: Tuple[int,...]` is *counts per stage*; `_layer_to_stage` (93-100) emits a monotone map. Round-robin chunk ownership is **unrepresentable**. Field type must change → breaks `tests/test_fine_builder_diff.py:285`, `tests/test_hier_builder_diff.py:353`. ~30 lines + new helper.
2. **`schedule.py:255-268` `should_emit_dp_comm`** — `return int(micro_batch_idx) == 0` with the comment *"Backward pass processes micro-batches in reverse, so b==0 corresponds to last_mb"*. **False under 1F1B** (mb 0's backward runs first). Silent wrong answer, not a crash. New branch, ~8 lines.
3. **`schedule.py:656-672`** (fwd GPipe cross-mb) — **replace**. `gpu_index += 1` / `first_transformer_layer[gpu_index-1]` assumes each stage boundary is crossed exactly once in layer order. With v chunks there are `pp*v` boundaries → index walks off the list. ~40 lines.
4. **`schedule.py:955-976`** (bwd GPipe cross-mb) — **replace**, same defect mirrored (`gpu_index = spec.pp - 1` decremented per boundary → negative; `hw_id == spec.pp - 1` last-stage test). ~40 lines.
5. **`schedule.py:1048-1072`** optimizer tail — `embedding_node_b[0]` / `_bwd_exit_node(0, min_l)`, i.e. microbatch 0, justified by *"bwd flips mbs"*. Wrong under 1F1B. Branch, ~15 lines.
6. **`schedule.py:778-793, 848-892, 978-1046`** ZeRO-3 lattice — `attach_parallel_edge(host, gather, skip_non_comm_children=True)` plus *"the next batch's embedding gather rides the non-comm children (legacy rule)"* (886-892) reads **the shape of the GPipe cross-mb edges**. Under 1F1B a layer node's non-comm children are different objects, so the gather silently attaches to the wrong place. ~60 lines of rework, or a `NotImplementedError` — i.e. the MoE-in-flattened bandage, repeated.
7. **`schedule.py:499-506`** `op_id` stamping is **creation order**, purely to reproduce the legacy counter. AstraSim issues the lowest node id first (CONTEXT.md: "node ids are therefore scheduling priorities"), so for a real 1F1B `op_id` must be the *schedule* priority. Either restructure creation order (invasive) or add `prio` and widen **`legacy_lowering.py:653-654 _stage_task_key`** — 4 lines in the pass DESIGN §1 promised would not exist.
8. **`pipeline_fine.py:290-303` + `352-356`** — `stage_delta = 1 if hint == "fwd" else -1`, the typed rehousing of legacy's `hw ± par_degree` *"HACK!!!!"*. Under interleaving the ZeRO-3 gather's peer stage is not pp±1. ~25-40 lines. (Line 355 still carries the raw `"bwd" in edge.name` fallback.)
9. **`llm_execution.py:353-368` + `487, 582, 703`** — gate the closed form off or double-count. 4 sites.
10-14. `config.py:1921/1959/2479`, `base_timing.py:378`, `train_timing.py:5065-5075`, `equiv/configs.py` + goldens, 2 test files. ~20 lines.

**New generator itself:** ~450-650 lines (warmup/steady/cooldown per device materialized as intra-device sequencing edges). Only ~150 lines of the fwd/bwd chain bodies are shareable, and only after a refactor that does not exist today — `build_pipeline_events` is one flat function with no seams.

**Total: ~600-800 lines; 14 edit sites across 8 files outside the generator.**

---

## CONCRETE DIFF PLAN — legacy (`85894c6`)

- `simulate_train_graph.py:688-1322` — same generator, same size. Its fwd GPipe block (840-856) and optimizer tail (1283-1318) are the *same code* as `schedule.py:656-672` / `1048-1072`.
- `simulate_train_graph.py:198-251` (`_compute_layers_per_stage`, `_stage_for_layer`, `_should_emit_dp_comm`) — same 3 edits.
- `llm_execution.py PipelineGraphFlattener._ensure_zero3_per_rank_edges` — same ±`par_degree` edit.
- `astrasim_lib/executor.py` per-stage Kahn keyed on `op_id` — same 1 edit.
- `llm_execution.py:1428-1440, 1474, 1626, 1736` — same 4 sites.
- `Graph.simulate` / `simulate_memory` — **0 edits, identically**.
- `config.py`/`base_timing.py`/`train_timing.py` — identical.

**Legacy: ~12 sites across 6 files, identical generator LOC.**

## Ratio: 14/12 ≈ 1.2. Call it 1:1.

The new core needs **as many scattered edits as the old one**, in the *same functions*, for the *same reasons*.

---

## Corroborating evidence the rewrite was conservative

- **`program/ir.py:250-257`**: `ProgramBuilder` — *"No production builder constructs Programs through this class today."* The IR's construction API is dead code; production construction runs through proto graphs + the converter.
- **`program/et_emit.py:174-179`**: `id_policy="program"` raises `NotImplementedError`. Only the byte-compatible legacy projection exists. The "two id policies, one temporary" plan (DESIGN §1) shipped only the temporary one.
- **`program/legacy_lowering.py`** (1095 lines) is the ~1100-line converter, renamed and declared *permanent*. It still re-derives stages from `hw_id`s (`:323`), re-attributes edges to stages (`:350-381`), re-runs **both** legacy dependency walkers (`:412-541`), and reconstructs communicators by **edge name** (`:594-598`). DESIGN constraint 3 — *"One placement/scheduling source of truth: no re-inference of stages, groups, or dependencies downstream of construction"* — is not met. CONTEXT.md:26-29 relabels this as a "deliberate non-goal."
- **`program/pipeline_fine.py:411-451`**: still `"linear_softmax" in obj.name`, `"optimizer" in obj.name`. `:355`: `hint = "bwd" if "bwd" in edge.name else "fwd"`. `:158-193`: two divergent metadata-copy tuples, one of which is documented as **bug-compatible** (`OVERLAP_NODE_COPY_ATTRS` drops `is_moe_layer`, so a tp-overlap head split misclassifies MoE memory).
- **`llm_execution.py:596-597`**: flattened MoE still `NotImplementedError` — DESIGN §2 constraint 4 ("make flattened-MoE possible later without violence") is untested, and the FINE builder's EP-sync path (`pipeline_fine.py` docstring: *"EP sync collectives (dormant)"*) is dead code nobody has run.

## What the rewrite genuinely bought
Typed events instead of untyped `Node`/`Edge`; the always-on group-order postcondition (`et_emit.py:463-464`) that converts the 85894c6 silent-deadlock class into a loud pre-AstraSim failure; the label-collision `EmissionError` (`et_emit.py:248-254`); and a coarse builder that is a real 1:1 projection. Those are worth having. None of them is generality, and none of them changes the interleaved-1F1B cost.