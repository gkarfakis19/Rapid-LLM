## Verdict up front

**Score: 4 / 10** on "general mechanism vs. pile of cases."

One module (`et_emit.py`) is a genuine general mechanism and deserves 8–9. Everything upstream of it — `schedule.py`, `pipeline_fine.py`, `block_program.py` — is a **mechanically re-typed transcription** of the deleted code. I measured it rather than eyeballing it: after folding renames (`self.num_batch`→`B`, `Node`→`ComputeEvent`, dict→dataclass accessors) and stripping docstrings/comments:

| new code | legacy original | normalized line identity |
|---|---|---|
| `schedule.build_pipeline_events` (478–1079) | `Graph.construct_fwd_bwd_graph` | **0.74** |
| `pipeline_fine._FineExpander` (304–851) | `PipelineGraphFlattener` (llm_execution 418–994) | **0.72** |
| `block_program.build_block_root` (119–497) | `Graph.construct_transformer_graph` | **0.89** |

For `_FineExpander` I diffed the *set* of semantically-new normalized lines. There are exactly **two**: the `role`-based layer dispatch and the `zero3_offset_hint`. Every other "new" line is a type annotation, a docstring, or `entry.get("name")` → `entry.name`. The owner's fear is correct.

---

## Special-case inventory

### A. TRANSPLANTED HACKS (would need another branch for a sibling case)

**A1. Name-substring dispatch survived, verbatim.** The rewrite claimed to type it away; it typed away exactly one of four sites.

```python
program/pipeline_fine.py:412   if obj.role in ("layer", "recompute"):      # ← the ONE that got fixed
program/pipeline_fine.py:417   if "linear_softmax" in obj.name:
program/pipeline_fine.py:426   elif "optimizer" in obj.name:
program/pipeline_fine.py:705   and "optimizer" in child.name
program/transforms.py:224      "attention" in str(getattr(child, "name", "")).lower()
```
Legacy (`git show 85894c6:llm_execution.py`, `_clone`): `if base_name.startswith("transformer_layer") or base_name.startswith("vit_block")` / `if "linear_softmax" in obj.name` / `elif "optimizer" in obj.name`. They replaced the *first* predicate with `role` and left the other three untouched, in the same `if/elif` chain. `ComputeEvent` already carries `role="softmax"` / `"optimizer"` (schedule.py:339, 586, 1070) — the data to fix them is sitting right there, unused. **Transplanted hack.** A new pinned-placement op (e.g. a router, a loss head) needs another `elif "..." in obj.name`.

**A2. `zero3_offset_hint` is a renamed hack, and the name hack is still the fallback.**
```python
program/pipeline_fine.py:353-356
    hint = getattr(edge, "zero3_offset_hint", None)
    if hint is None:
        hint = "bwd" if "bwd" in edge.name else "fwd"
    stage_delta = 1 if hint == "fwd" else -1
```
Legacy: `if not "bwd" in edge.name: # HACK!!!!` → `offset = self._par_degree`. The new version stamps `"fwd"`/`"bwd"` strings at schedule.py:870/881/1026 and then *still* falls back to substring-sniffing the name. The principled rule is trivially available: the gather's placement is the device of the compute it feeds — the builder has `target` in hand at schedule.py:874. Instead they encode "±1 pp stage" as a direction string. `_offset_stage_device` (pipeline_fine.py:290–303) is a mild improvement (a pp-coordinate hop instead of raw `±par_degree`) but keeps the raw arithmetic as a layout-less fallback at line 300. **Transplanted hack with a new name.**

**A3. Dead code transplanted for "fidelity."**
```python
program/pipeline_fine.py:336-338
    direction = getattr(edge, "direction", None)
    if direction and str(direction).lower() == "backward":
        wire_anchors = rank_heads
```
`CommEvent.direction` is *never assigned* in `schedule.py` — the only `.direction =` writes are on `ComputeEvent`s (schedule.py:615, 729, 745). The docstring at pipeline_fine.py:327–330 admits it: *"the branch is kept for fidelity."* Same for the `"bwd" in edge.name` fallback: only `zero3_transformer_gather` carries `tp_shard: True` (train_timing.py:4496), and all three of its creation sites stamp the hint — so line 355 can never fire either. Two dead branches carried into a "clean-sheet" builder.

**A4. Un-divided ZeRO-3 bytes + `rank_tails[0]` DP asymmetry, copied character-for-character.**
```python
program/pipeline_fine.py:360   per_rank_bytes = int(base_bytes)          # not / par_degree
program/pipeline_fine.py:629   self._attach(rank_tails[0], child_clone)  # DP collective on rank 0 only
```
DESIGN §6 pre-authorizes these as "pinned quirks." That's fine as a *modeling* decision, but they're modeled as inline literals inside the expansion loop, not as a named, swappable policy. Compare A2's treatment: they were willing to invent `zero3_offset_hint` for one quirk and not for these.

**A5. The ZeRO-2/ZeRO-3 lattice is a hardcoded key-string cascade.** `schedule.py:798–1046` is ~250 lines built on twelve literal comm-key strings (`"zero2_embedding_gather"`, `"zero3_transformer_gather"`, `"transformer_moe_ep_sync"`, …), each with its own `if key in spec.comm_metadata and spec.should_emit_dp_comm(key, b)` guard and its own bespoke attach. `attach_parallel_edge` (schedule.py:553) is copied verbatim including the two boolean escape hatches:
```python
def attach_parallel_edge(target, gather_edge, skip_non_comm_children=None, skip_comm_children=None):
```
These flags are named after *what to skip*, not after the rule they implement — the cross-device ZeRO-3 case (schedule.py:885–892 and 1029–1032) needs both, in a specific order, with a comment saying "legacy rule." A ZeRO-3 variant (per-layer prefetch depth, hybrid FSDP shard groups) needs a fourth copy of this block. **Fully transplanted.** Also cosmetic-hack retention:
```python
program/schedule.py:801-803
    postfix = "_all_reduce"
    if zero2_embedding_key in spec.comm_metadata or zero3_embedding_key in spec.comm_metadata:
        postfix = "_reduce_scatter"
```
The op name is derived by sniffing which keys exist, when `CommMeta.kind` already *is* the collective type.

**A6. Grad-accum / ZeRO-3 / `last_mb` coupling duplicated in two places.**
```python
program/schedule.py:264   if self.dp_microbatch_mode != "last_mb" or self.zero_stage >= 3: return True
program/schedule.py:937   apply_ep_all_mbs = spec.dp_microbatch_mode != "last_mb" or spec.zero_stage >= 3
```
The EP-sync branch (936–953) re-derives the same predicate rather than calling `should_emit_dp_comm`, because legacy did. Two copies of one policy; changing one silently desyncs the other.

**A7. MoE hot/cold routing is a two-value enum implemented as two `if`s, twice.**
```python
program/block_program.py:223-235
    def _moe_hot_rank(tp_idx, cp_idx, routing_mode):
        if routing_mode == "ep":     return _rank_id(tp_idx, cp_idx, 0)
        if routing_mode == "tp_ep":  return _rank_id(0, cp_idx, 0)
        raise ValueError(...)
    def _moe_parallel_token(tp_idx, cp_idx, routing_mode):
        if routing_mode == "ep":     return (routing_mode, cp_idx, tp_idx)
        if routing_mode == "tp_ep":  return (routing_mode, cp_idx)
```
Byte-identical to legacy. Plus the hot-rank ordering dependency is *still* an unenforced construction-order assumption with a runtime error instead of an invariant (block_program.py:356–361: *"the simulator assumes the first rank in the routing group is hot and must be constructed before colder ranks"*). A third routing mode = edits in two functions + the `_attach_moe_parallel_post_group` component switch at 305–313. **Transplanted, 0.89 identity.**

**A8. Flattened-MoE rejection is the same hardcoded `NotImplementedError`.**
```python
llm_execution.py:597  raise NotImplementedError("MoE is not supported with full AstraSim flattened execution.")
```
CONTEXT constraint 4 was *"the design must make flattened-MoE possible later without violence."* It isn't. The EP-sync `CommEvent` reaches `_expand_transformer_node`'s `other_children` path (it isn't `interconnect == "dp"` and isn't `PIPELINE`), so it falls to `self._attach(downstream_parents, child_clone)` at pipeline_fine.py:713 — **one** collective op with **one** `local_hw_id`, attached to all `par_degree` rank tails. That is not an EP collective; it's a single-device op with cluster-wide fan-in. The docstring admits the state at pipeline_fine.py:41–42 (*"EP sync collectives (dormant: flattened MoE is rejected upstream)"*). The rejection is load-bearing, not a policy choice.

**A9. Known bug deliberately preserved with a comment instead of fixed.**
```python
program/pipeline_fine.py:160-166 (comment) + 183-193 (OVERLAP_NODE_COPY_ATTRS)
# It OMITS ``is_moe_layer`` ... so a tp-overlap head split of a MoE FineNode
# loses the flag and ``program.memory_sim`` classifies the head as dense...
# Bug-compatible by design.
```
Two adjacent attribute tuples, `FINE_EXPANDER_COPY_ATTRS` and `OVERLAP_NODE_COPY_ATTRS`, differ by exactly `is_moe_layer`/`param_gather`/`cp_rank`, and the delta is a documented wrong answer. This one isn't even golden-pinned (MoE + tp_overlap isn't in the matrix) — it's latent-incorrect code kept for symmetry with a deleted file.

**A10. Duplicated dispatcher policy.** `effective_dp = 1 if run_type == "inference" else max(1, ...)` appears verbatim at `llm_execution.py:412`, `:607`, `:743`; `include_optimizer = ... != "nonfinal"` at `:395`, `:612`, `:746`. `ScheduleSpec` — the thing whose job is to "distill" the schedule — takes `include_backward`/`include_optimizer` as caller-supplied arguments (schedule.py:182–184) instead of owning the derivation. Three copies of the inference rule, three copies of the grad-accum rule.

### B. PRINCIPLED RULES (genuine wins — credit where due)

**B1. `et_emit.py` — the one module that actually generalized.** 644 lines, and the strings `zero3`, `moe`, `recompute`, `softmax`, `embedding` appear **zero** times. Every branch is an AstraSim-contract rule keyed on IR structure, not on features: `dp_count <= 1 and op.is_dp` (301), singleton-group → COMP no-op (313–320), `src_device == dst_device` elision (351), control predicate `size == 0 or comm_type != PIPELINE` (372). This is what the rest of the rewrite was supposed to look like.

**B2. The always-on group-order postcondition (et_emit.py:463–533) is new, not a port.** No equivalent exists in `85894c6:astrasim_lib/executor.py`. It converts the historical silent-deadlock class into a loud pre-simulation failure, and it's a structural property check (per-communicator issue-order projection), not a case list.

**B3. The label→two-communicators guard (et_emit.py:241–254) is new.** Legacy last-wins silently; this raises. Correct instinct, general form.

**B4. `retime.apply_block_timings` replaced name-prefix matching with `role is OpRole.TRANSFORMER_LAYER`** (retime.py:96). The legacy `_assign_transformer_durations` walked for `"transformer_layer"`/`"vit_block"` prefixes. Real, data-driven. (Note the irony: `block_prefix` at schedule.py:282 still does `startswith("vit")` on the *model type* to build those names.)

**B5. `memory_sim`'s `granularity` metadata** replaces the legacy `_is_non_flattened` name-sniffing DFS guard (`85894c6:memory_estimation.py:707`). Small but correctly principled.

**B6. `pipeline_coarse.py` is clean** — a 1:1 typed projection of events with a `_ROLE_MAP` table (94–102) instead of branches. It's the only builder that reads like an IR construction. Its weakness is structural, not local (see C1).

---

## The structural problem the inventory points at

The special cases are a symptom. The disease is that **`Program` is not the source of truth in any execution path.**

**C1. `legacy_lowering.py` was supposed to be deleted at M8; it was reclassified as permanent.** DESIGN §5's M8 row lists `legacy_lowering.py` under *Deletes*. Instead, legacy_lowering.py:16 now reads **"PERMANENT — kept at M8."** All three builders produce a mutable, `id()`-keyed, dynamically-attributed proto DAG (`FineNode`/`FineEdge`/`ComputeEvent`, with attributes bolted on at runtime: `gather_edge.tp_rank = r`, `edge_obj.is_cross_layer = True`) and hand it to a 1095-line reproduction of converter Steps 1–7. The typed IR is what falls out the *back* of that pass.

**C2. Consequently, DESIGN §1's headline claims are false as shipped:**
- *"flattening ceases to exist as a graph rewrite"* — `_FineExpander._clone` (pipeline_fine.py:403–513) is a clone-with-expansion graph rewrite with an `id()`-keyed `_clone_cache`, 0.72-identical to the flattener it "replaced."
- *"every consumer is a pure function over [the Program]"* — `analytic_sim` reads `meta.misc["coarse_proto_root"]`, `memory_sim` reads `meta.misc["fine_proto_root"]`, and hierarchical ET emission reads `meta.misc["coarse_proto_root"]` (pipeline_coarse.py:310). Three of five consumers bypass the op list. `Program.meta.misc` is a live Python object graph, not metadata.
- *"`OpKey` gains a `device` component"* (DESIGN §2 amendment 1) — `grep -rn OpKey program/ tests/` returns **nothing**. The abstraction was dropped without a note.
- The `program` id policy is `NotImplementedError` (et_emit.py:174–179), so the "two id policies won't survive" condition is unmet and every emission still depends on reproducing legacy creation-order coincidences.

**C3. "ONE GPipe schedule" branches on execution mode.** `ScheduleSpec.flattened_mode` (schedule.py:172) feeds `recompute_enabled` (schedule.py:274–278), so the *structure* of the enumerated schedule differs depending on which backend will consume it. The unification is nominal.

**C4. The evidence base for "it's equivalent" is thinner than it reads.** The four `*_builder_diff.py` suites no longer diff against legacy (it's deleted); they are now self-determinism sweeps, and **59 of 74 are env-gated off by default** (`RAPID_FINE_DIFF` / `RAPID_BLOCK_DIFF` / `RAPID_COARSE_DIFF` / `RAPID_HIER_DIFF`). `tests/test_fine_builder_diff.py:17-36` documents this honestly, which I credit — but it means the only live gate on all of the above is the 42-spec golden matrix, i.e. bit-exactness with the transplanted behavior. Nothing in CI can detect that a rule is a hack rather than a rule.

---

## Worst offenders, ranked

1. **`program/pipeline_fine.py:403–513, 516–725`** — the flattener, transplanted at 0.72 identity, still doing name-substring dispatch (417/426/705), still carrying two dead branches (336–338, 355), still hardcoding `rank_tails[0]` (629) and `int(base_bytes)` (360). This is the file the rewrite was ostensibly *about* and it is the least changed.
2. **`program/legacy_lowering.py`** — 1095 lines the plan said would be deleted, kept by redefining the goal. As long as it exists, the IR is a report, not a representation, and every builder must produce a legacy-shaped mutable DAG to feed it.
3. **`program/schedule.py:778–1072`** — the ZeRO/DP/EP lattice: ~290 lines, twelve literal key strings, `attach_parallel_edge` with its two skip-flags, duplicated grad-accum predicate. Adding a sharding strategy means editing six places.
4. **`program/block_program.py:223–374`** — MoE at 0.89 identity, including the unenforced construction-order assumption and the two-mode `if` cascade duplicated across two helpers.
5. **`llm_execution.py:597`** — the MoE rejection, which is not a policy but a load-bearing guard against a genuinely broken EP path in the FLAT builder, in direct contradiction of CONTEXT constraint 4.

**What a non-conservative rewrite would have looked like:** placement as a function of an op's semantic key and the layout (killing A1, A2, A4); the DP/ZeRO lattice as data — a table of `(when, what, where, attach-rule)` rows driven off `CommMeta` (killing A5, A6); EP/MoE as a per-rank collective descriptor the fine expander instantiates per rank (killing A7, A8); and the op list as the artifact the ordering pass *emits*, not the artifact it *consumes*. The `et_emit` module proves the team can write that kind of code. They wrote it once, at the very end of the pipeline, where the AstraSim contract forced their hand — and nowhere the goldens would have made them re-derive a number.