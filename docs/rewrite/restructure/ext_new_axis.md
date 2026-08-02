## VERDICT

The fear is substantially correct. The rewrite delivered **one** real generalization (rank linearization: 3 copies → `RankLayout`) and **one** real firewall (`et_emit.py` is genuinely axis-blind). Everything else that the extension exercise touches — cluster-axis decomposition, block-rank formula, MoE routing, overlap plumbing, group-membership inference, schedule enumeration — was **retyped, not replaced**, and in one case (overlap) the rewrite *added* a duplicate. Measured on the "add `sp`" feature the new core needs ~33 edit hunks vs the legacy's ~37. **Ratio ≈ 0.9 : 1.**

---

## Part 1 — Bandages that survived, with citations

**Name-string dispatch is still the dispatch mechanism, even where a typed field now exists.** `schedule.py` stamps `role=` on every event (`schedule.py:586,594,611,700,709,725,1070`). `pipeline_fine._clone` uses `role` for exactly one branch (`pipeline_fine.py:412`) and then falls back to legacy name sniffing for the rest: `"linear_softmax" in obj.name` (`:417`), `"optimizer" in obj.name` (`:426`), `"optimizer" in child.name` (`:705`). The typed field was added and not used. Same class: `"attention" in str(name).lower()` (`transforms.py:224`), `mode == "tensor_sequence"` string compare (`transforms.py:326,442`), `comm_interconnect_type == "tp"/"cp"/"dp"` (`transforms.py:351,368`, `pipeline_fine.py:603,747`).

**The `"bwd" in name` hack is still in the file.** `pipeline_fine.py:353-356` prefers `zero3_offset_hint`, then falls back to `hint = "bwd" if "bwd" in edge.name else "fwd"`. Typing a hack and keeping the hack as the fallback is not replacing it. The ±`par_degree` stage offset is likewise preserved verbatim in the no-layout path (`:300`), and un-divided ZeRO-3 bytes at `:360` (`per_rank_bytes = int(base_bytes)` — the comment calls it "legacy modeling quirk").

**Communicator membership is still *inferred*, and the inference contains a hand-written axis special case.** `legacy_lowering.py:243-248`:
```python
if axis == "ep":
    if tp_size > 1 and participants == tp_size * ep_size:
        composite_axes = ("tp", "ep")
```
This re-derives a communicator's shape from a participant *count* — the exact "no re-inference of groups downstream of construction" the design forbade (CONTEXT.md constraint 3). It is the only mechanism for a collective spanning two axes.

**`legacy_lowering.py` was scheduled for deletion at M8 (DESIGN §5) and is now declared permanent** (`legacy_lowering.py:16`, CONTEXT.md:24-29). It still contains: stage re-derivation from `hw_id` (`:323`), per-edge stage attribution (`:350-381`), **both** near-duplicate backward dependency walkers (`:412-475` and `:477-541`, ~130 lines, structurally identical) — i.e. the three specific converter sins CONTEXT.md §"What is being replaced" itemized as motivation for the rewrite. They are 1095 lines of converter, moved.

**"Flattening ceases to exist as a graph rewrite" (DESIGN §1) is false.** `_FineExpander` (`pipeline_fine.py:211`) is a clone-with-`_clone_cache`/`add_child`/`parents` graph rewrite whose own docstring says "traversal, branch structure, op-id counter discipline, and attach orders are kept verbatim". `FineNode`/`FineEdge` (`:82,115`) are `Node`/`Edge` with new names.

**"Every consumer is a pure function over the Program" (DESIGN §1) is false.** `memory_sim`, `analytic_sim`, and `lower_coarse_for_emission` all read the proto graph out of `meta.misc`, not the op list: `pipeline_fine.py:968` (`"fine_proto_root"`), `pipeline_coarse.py:256,310` (`"coarse_proto_root"`).

**`OpKey` does not exist.** DESIGN §2 amendment 1 is entirely about giving `OpKey` a `device` component; `grep -r OpKey` returns nothing. `ProgramBuilder` (`ir.py:250-257`) admits it has no production callers. `OpRole.GEMM`/`JOIN` (`ir.py:84-85`) are never stamped. The "typed IR" is, in production, three fields (`device`, `duration`, `deps`) plus `legacy_op_id`.

**Genuine wins, for calibration:** `et_emit.py` (0 axis literals; groups are member-device sets), `analytic_sim`/`memory_sim`/`validate`/`retime`/`viz` (hw_id ints + metadata; `memory_sim.py:27-28,283` explicitly "never node names"), `RankLayout` replacing 3 copies of the coords/linearize loop, and the always-on group-order postcondition (`et_emit.py:485-533`) which is real and new.

---

## Part 2 — Extension diff plan: add `sp` as a first-class axis (CURRENT code)

| File | Hunks | ~LOC | What breaks |
|---|---|---|---|
| `program/layout.py` | 4 | 25 | `CANONICAL_AXES:40`; `cluster_coords:43-74` — 4 hand-rolled divisor chains + 4 explicit kwargs, `ep`'s divisor `tp*cp` must become `tp*sp*cp`; `cluster_axes = ("tp","cp","ep")` at `:248,257`; 4 per-axis "must include" checks `:310-317` → 5; **duplicate canonical list at `:303`** must stay in sync with `:40` |
| `program/schedule.py` | 2-3 | 10-28 | `ScheduleInputs`/`ScheduleSpec` fields; `par_degree():284`. If sp needs a pipeline-level sync you copy the EP-sync block `:936-953` — an axis-specific `if spec.ep > 1 and spec.dp > 1` with hardcoded comm-key names, i.e. the precedent is "one bespoke `if` per axis" |
| `program/pipeline_fine.py` | 4 | 10 | `par_degree` `:234`; `_configure_rank_layout:252-260` product check + error text; `cluster_coords(...)` call `:279-287`; `_offset_stage_device:300`. **Small — the expander treats the cluster as one opaque flat rank and delegates. This is the one place the abstraction actually absorbs the feature.** |
| `program/block_program.py` | 8 | 30 | `_rank_id:188-189` is a **hand-written 3-axis linearization that duplicates `cluster_coords`** — a 4th axis means a 4th term that must match layout strides by hand; triple loop `:376-378` → quadruple; `_moe_hot_rank:223-228` and `_moe_parallel_token:230-235` enumerate axis combinations per `routing_mode` string → a new `"tp_sp_ep"` mode = 4 new branches; `sp_rank` attr `:218-219`; `TransformerBlockSpec:94-112`; `build_block_program` kwargs `:523-568` |
| `program/transforms.py` | 6 + 8 signatures | 45-60 | Mode-label dispatch `:326-331` and `:441-447`; the tp block `:332-355` and cp block `:358-374` are already near-duplicates — sp is a third copy; `"attention"` predicate `:224`. **The three floats `tp_overlap, tp_sp_overlap, cp_overlap` are plumbed positionally through 8 signatures** (`build_fine_root`, `build_fine_program`, `build_block_program`, `apply_overlap_to_fine_root`, `apply_tp_overlap`, `_build_transformer_block_programs`) and 3 call sites (`llm_execution.py:662-664,783-785,814-816`). No `OverlapSpec`, no per-axis map |
| `program/legacy_lowering.py` | 1 | 6 | Only the composite hack `:243-248` — an sp all-gather runs over the *tp* group, so you must add `if axis == "sp" and participants == tp_size*sp_size: composite = ("tp","sp")`. Group *discovery* (`:158-181`, `:597`) is axis-name-driven and needs **zero** edits — a real win, undercut by having to extend the hack next to it |
| `llm_execution.py` | 5 | 8 | `axis_sizes:126`; `_subset_descriptor(["tp","cp","ep"])` / `(["pp","dp"])` `:144-145` — the cluster/schedule partition is a string literal here **and again** at `layout.py:248`; 3 overlap kwarg blocks |
| `memory_estimation.py` | 3 | 6 | `axis_sizes:298`, `cluster_coords(...):313-321`, `par_degree:324`, `vocab_shard ceil(v/(tp*cp)):426`. **M0 unified the validation but left the call shape duplicated here** |
| `et_emit.py`, `analytic_sim.py`, `memory_sim.py`, `validate.py`, `retime.py`, `viz.py`, `gmap.py`, `fault_projection.py`, `layout_utils.py`, `executor.py` | **0** | **0** | Genuinely absorb it |

**Core subtotal: ~140-175 LOC across ~33 hunks in 8 files.**

Shared/out-of-scope cost, identical in both counterfactuals: `config.py` (`ScheduleConfig` fields `:1980-2011`, overlap block `:924-926,997-1013`, `tp_sp` is a **bool** today `:1984` and must become a degree), `astrasim_lib/config_generation.py` (5 hardcoded 5-axis dicts at `:477-482,492-497,558-563,563,582` — **untouched by the rewrite**), `train_timing.py` (~80-150 lines of CommSpec/comm_keys wiring; the 12 hardcoded pipeline `comm_metadata` entries at `:4401-4504`), `equiv/configs.py` (~25 row literals `dict(dp,tp,cp,pp,mb,tp_sp)`) + golden regen.

### Counterfactual: same feature against `85894c6`

- `llm_execution.py:1050-1160` `_build_rank_layout_descriptor` — same 4 hunks, ~25 LOC. **Δ 0.**
- `llm_execution.py:387-418` + `:975-1006` `PipelineGraphFlattener._configure_rank_layout`/`_hw_id_for_rank` — a second independent copy of the decomposition, 3 hunks / ~12 LOC. **Δ −3 hunks (real M0 win).**
- `memory_estimation.py:289-370` `_build_rank_layout`/`_hw_id_for_rank` — a third copy, 4 hunks / ~15 LOC vs 3 hunks / ~6 today. **Δ −1 hunk, −9 LOC.**
- `simulate_train_graph.py:1323-1600` `construct_transformer_graph`: `_rank_id:1377-1378` is **byte-identical** to `block_program.py:188-189`; triple loop `:1557-1559` identical; `_moe_hot_rank`/`_moe_parallel_token` identical. 8 hunks / ~30 LOC. **Δ 0.**
- `simulate_train_graph.py` EP-sync + Graph degrees: **Δ 0.**
- `llm_execution.py:96-292` overlap (`_split_tp_node`, `_split_cp_edge`, the two `_apply_*_transforms`): 5 hunks + ~3 signatures. Today: 6 hunks + 8 signatures, because the rewrite forked overlap into a proto-level implementation **and** a Program-level `apply_tp_overlap` that `transforms.py:69-81` admits has no production caller. **Δ +1 hunk, +5 signature edits.**
- `executor.py` `_assign_collective_labels` composite ep hack, `_build_axis_groups`, `_extract_axis_layout`: verbatim ancestors of `legacy_lowering.py:158-282`. **Δ 0.**
- `executor.py` emission Steps 8-12: dp-major, device-int based, same axis-blindness as `et_emit`. **Δ 0.**
- `config_generation`/`gmap`/`fault_projection`/`layout_utils`/`config`/`train_timing`: unchanged files. **Δ 0.**

**Legacy subtotal: ~165-200 LOC across ~37 hunks in 7 files.**

### Ratio

**~33 : ~37 hunks ≈ 0.9. ~150 : ~180 LOC ≈ 0.85.**

The entire delta is the M0 rank-layout dedup (−4 hunks), partially given back by the duplicated overlap pass (+1 hunk, +5 signatures). Every other axis-shaped decision — the block rank formula, the MoE routing-mode enumeration, the composite-group inference, the cluster-vs-schedule axis partition (3 copies: `layout.py:248`, `layout.py:303`, `llm_execution.py:144-145`), the three-float overlap plumbing — costs exactly what it cost before.

---

## Part 3 — The second variant is worse: 1 : 1

"A second pipeline dimension" (interleaved / virtual pipeline as a real axis) is today a **closed-form scalar multiply on the answer** (`llm_execution.py:353-368`), applied identically in all four modes.

Making it structural requires a new schedule generator. There is no seam for one. `build_pipeline_events` (`schedule.py:478-1079`) is a **600-line straight-line transcription** of `construct_fwd_bwd_graph` in which `spec.pp` simultaneously means device count, schedule depth, and array length: softmax pinned to `spec.pp - 1` (`:583,709`), `[-1] * spec.pp` walk arrays (`:956-957`), `for stage in range(spec.pp)` optimizer tail (`:1054`), `stage_for_layer` from `legacy_layers_per_stage` (`:83-90`). `ScheduleSpec` is a parameter bag, not a schedule: it has no event stream, no policy object, no strategy hook. The FLAT and PIPELINE builders both consume `events.root` directly (`pipeline_fine.py:882`, `pipeline_coarse.py:152`), so an alternative generator must reproduce the `add_child` order byte-for-byte or the pinned emission order breaks (`legacy_lowering.py:16-29`).

Legacy: `construct_fwd_bwd_graph`, 630 lines, same shape. **~600 new lines either way. Ratio 1 : 1.** DESIGN §5 lists "native 1F1B generator" as optional post-M9 work; nothing in the landed code prepares for it. The single most-cited motivation for an IR — "schedule is data, builders are pure functions of it" — produced one function with `spec.` prefixed onto the same locals.

---

## Part 4 — What a non-conservative rewrite would have had (and doesn't)

1. **An `AxisSpec` registry**: name, size, stride role, cluster-vs-schedule class, collective interconnect tag, overlap fraction. Would collapse `layout.py:40+248+303`, `llm_execution.py:144-145`, `memory_estimation.py:298`, `config_generation.py:477-563` into one table. Today the same 5-tuple is spelled out in ≥9 places.
2. **`RankLayout.linearize` as the *only* rank formula.** `block_program.py:188-189` and `memory_estimation.py:309-322` are two more.
3. **An `OverlapSpec: Mapping[axis, float]`** instead of three positional floats through 8 signatures.
4. **Groups declared at construction**, carried on the op. `GroupKey` exists (`ir.py:93`) but every production Program gets its groups from post-hoc inference in `legacy_lowering.py:196-282`, participant-count heuristic included.
5. **A schedule protocol** (`iter_events(spec) -> Iterator[Event]`) so GPipe / 1F1B / interleaved are three implementations, not one 600-line function.

None of these are hard. All five were in scope. The migration's own discipline — "zero golden regens through M8" (DESIGN §6) — is precisely what made them impossible: every one requires changing an emission order that is pinned to legacy creation sequence. The rewrite optimized for byte-identity and got byte-identity, including the bytes it was supposed to delete.