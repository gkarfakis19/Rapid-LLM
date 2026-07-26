## VERDICT: two-representation compromise. The typed IR is a side-car, not the source of truth.

DESIGN.md §1 claims "one `Program` of placed ops … every consumer is a pure function over it; flattening ceases to exist as a graph rewrite." Measured against the tree: **one consumer (`et_emit`) reads the op list. Every other consumer reads an adjacency-list graph carried in `Program.meta.misc`.** The rewrite retyped the objects and renamed the passes; it did not change the representation the system actually computes over.

---

### 1. Two decisive experiments (run against this tree)

**(a) The op list is literally deletable for the memory path.** Built a FINE program (76 ops), ran `memory_sim.simulate_memory`, then set `p.ops=[]; p.groups={}; p.devices=(); p.dp_count=99` and re-ran:

```
with full op list: (0.00306, 0.0)
with ops WIPED   : (0.00306, 0.0)     IDENTICAL: True
```

`program/memory_sim.py` contains **zero** references to `program.ops`. It takes a `Program` solely to `meta.misc.get("fine_proto_root")` (`memory_sim.py:73`) and then runs the legacy list scheduler over `event.children` / `parent.done` (`memory_sim.py:458-460`), mutating `done`/`scheduled`/`finish_time` attributes that aren't even declared on `FineNode` — the legacy mutable graph, verbatim. 529 LOC of consumer, 0 LOC of IR.

**(b) Writing through the typed IR is silently discarded at emission.** Built a COARSE program, set `op.duration=(0.777,)` on all 8 `TRANSFORMER_LAYER` ops (the exact thing `retime` exists to do), then called `lower_coarse_for_emission`:

```
durations in the LOWERED (emitted) program: [(1e-05,), (0.001,), (0.002,)]
0.777 present in emission? False
```

`pipeline_coarse.py:302-323` ignores `coarse_program.ops` entirely and re-lowers `meta.misc["coarse_proto_root"]`. The typed op is not the source of truth *even for the one field it uniquely owns*. `retime.py` only works because it writes the value **twice** — `op.duration = tuple(values)` then `events[op.uid].duration = ...` (`retime.py:126-127`) — an unchecked index-parallel coupling between `meta.misc["coarse_events"]` and uid order. That mirror is the load-bearing write; the op write feeds only `analytic_sim`.

---

### 2. Consumer trace (asked for explicitly)

| consumer | what it iterates | reads `program.ops`? |
|---|---|---|
| `et_emit.emit_chakra` | `ops = program.ops` (`et_emit.py:192`) | **Yes — the only one** |
| `analytic_sim.evaluate` | `event.children`/`child.parents` heap loop (`analytic_sim.py:217-249`), device discovery by DFS over `node.children` (`:180-190`) | Only `program.ops[uid_of[id(event)]].duration[0]` (`:169`) — a duration lookup |
| `memory_sim.simulate_memory` | `fine_proto_root` (`:73`), children lists (`:458`) | **No** |
| `viz.save_events_graph` | proto events, `hasattr(node,"comm_size_bytes")` duck-dispatch (`viz.py:85-131`) | **No — never receives a `Program`** |
| `retime.apply_block_timings` | `ops` + mandatory event mirror (`:127`) | Yes, but non-authoritative (see 1b) |
| `lower_coarse_for_emission` | `meta.misc["coarse_proto_root"]` | **No** |

**Producer side is worse: there is no path in the tree that constructs an emittable `Program` from typed ops.** All three builders — `pipeline_fine.py:950`, `block_program.py:585`, `pipeline_coarse.py:317` — build an untyped `parents`/`children` graph and hand it to `legacy_lowering.lower_to_program`. `ProgramBuilder` is documented dead: *"No production builder constructs Programs through this class today"* (`ir.py:254-258`). `et_emit` supports exactly one id policy — `legacy` — and raises `NotImplementedError` otherwise (`et_emit.py:173-178`). So the emitted bytes are, end to end, a function of the event graph plus a converter port. The op list is a typed *rendering* of that pipeline's intermediate state.

---

### 3. Is the proto graph the old untyped graph wearing a new name? Yes, provably.

`ComputeEvent`/`CommEvent` (`schedule.py:304`, `:393`) are attribute-for-attribute reconstructions of legacy `Node`/`Edge` (`git show 85894c6:simulate_train_graph.py:38,126`): same `name/op_id/hw_id/_duration_data/fwd/mem_kind/recompute`, same `comm_size_bytes/comm_type/participants/comm_interconnect_type/local_hw_id/is_dp`, same `add_child` mutating `children`+`parents`, same `_normalize_duration` / `duration` / `duration_profile` property triple. `FineNode`/`FineEdge` (`pipeline_fine.py:82,115`) are a *third* copy of the same shape. The docstrings admit it ("legacy `Node` stand-in", "so the flattening port reads them exactly as the legacy code did").

Transplant rate, measured by normalized line-match against the deleted originals:

| new code | legacy original | line-identical |
|---|---|---|
| `block_program.build_block_root` (119-504) | `Graph.construct_transformer_graph` | **83%** |
| `pipeline_fine._FineExpander` (211-857) | `PipelineGraphFlattener` (llm_execution.py 339-994) | **61%** |
| `legacy_lowering.lower_to_program` (290-860) | `convert_rapid_llm_graph_to_chakra_et` Steps 1-7 (executor.py 747-1334) | **54%** |

The `analyze_for_compute` walker diffs to *nothing but the deleted `debug_en` print statements*. `construct_transformer_graph` was not deleted; it was moved into `block_program.py` with `Node`→`FineNode`.

---

### 4. The hacks the owner feared: still there, individually cited

- **Name-based dispatch survives, in the new code:** `if "linear_softmax" in obj.name:` (`pipeline_fine.py:417`), `elif "optimizer" in obj.name:` (`:426`), `if ... "optimizer" in child.name` (`:697`). A typed `role` field exists on `ComputeEvent` and is used one line earlier (`:412 obj.role in ("layer","recompute")` — and that's a *string*, not `OpRole`), then abandoned for substring matching.
- **The `"bwd" in edge.name` hack is not replaced, it's shadowed:** `zero3_offset_hint` was added (`schedule.py:446`) but the consumer keeps the old rule as a live fallback: `hint = "bwd" if "bwd" in edge.name else "fwd"` (`pipeline_fine.py:355`).
- **ZeRO-3 flattened special cases: transplanted including dead code.** `pipeline_fine.py:622` assigns `per_rank_edges = self._ensure_zero3_per_rank_edges(...)` and never reads it — a verbatim copy of the same dead binding at legacy `llm_execution.py:799` (contrast `:477-480` and `:717-721`, which do register into `_clone_cache`). The `hw ± par_degree` "HACK!!!!" was re-expressed as a ±1 pp-coordinate hop (`:290-302`) but the comment concedes it is "the same device"; un-divided `int(base_bytes)` per-rank gather sizes remain (`:369`).
- **MoE special cases: transplanted with their construction-order coupling.** `block_program.py:283-373` reproduces `_moe_hot_rank`/`_moe_parallel_token`/`moe_component` string dispatch and still raises *"hot join … must be constructed before colder ranks"* — a builder-ordering dependency that a placed-op IR exists to make impossible. Flattened MoE is still rejected (`llm_execution.py:597`), so CONTEXT constraint 4 ("possible later without violence") remains unevidenced.
- **The legacy converter's fragility is preserved by design:** the Kahn heap deliberately iterates raw `set`s because "op_ids are NOT unique … the tie-counter sequence depends on set iteration order" (`legacy_lowering.py:645-651`). A deps-carrying IR would have made the tie-break explicit; instead the comment argues the nondeterminism happens to cancel.

---

### 5. The stated justification does not hold

Four module docstrings assert the same law: the event graph must stay because *"the FIFO/tie discipline depends on children-list adjacency order, which uid order cannot represent"* (`memory_sim.py:41-48`, `analytic_sim.py:55-58`, `pipeline_coarse.py:34-40`, `legacy_lowering.py:16-30`).

That is a statement about **this** IR, not about IRs. `ComputeOp`/`CollectiveOp` carry `deps` and no ordered successor list — and the lowering **actively destroys** the ordering information it would need: `unique_deps = tuple(dict.fromkeys(sorted(dep_uids)))` (`legacy_lowering.py:768`, `:787`). Two fields — an ordered `succs` tuple, or simply not sorting `deps` — would let every consumer run on ops. Similarly `analytic_sim` must read comm sizes off the events because `CollectiveOp.size_bytes` is narrowed to `int` at construction (`pipeline_coarse.py:222,238`) while "dp reducer sizes are floats and must stay floats" (`analytic_sim.py:49-51`). The COARSE program also drops `device` to a `-1` sentinel (`:236`) and registers `groups={}` (`:266`) — `GroupKey`/`CommGroup`, one of DESIGN §2's core abstractions, is *empty* on the coarse program. **A self-inflicted expressiveness gap in the IR is being reported as a law of nature and used to make the second graph permanent.**

Corroborating: `Program.validate` is not a real gate. Every production call is `validate_program(..., check_races=False)` (`legacy_lowering.py:851`, `et_emit.py:189`, `transforms.py:556`), so **V6 — the group-race / deadlock invariant, i.e. CONTEXT constraint 2's "deadlock-freedom by construction" — never runs outside tests** (`validate.py:21-26,196`). COARSE programs are built with validation off entirely and knowingly violate V1 and V5 (`pipeline_coarse.py:58-66`). The only real deadlock guard is `et_emit`'s group-order postcondition — enforced on emitted protobuf nodes, after the fact, not by the IR.

---

### 6. Quantification (asked for)

`program/` is 7,849 LOC.

| | LOC | % |
|---|---|---|
| **Second untyped graph + the pass that linearizes it** | **~4,290** | **55%** |
| `legacy_lowering.py` (whole) | 1,095 | |
| `_FineExpander` flattener port + `FineNode`/`FineEdge` | 776 | |
| `build_pipeline_events` + `ComputeEvent`/`CommEvent` | 776 | |
| `block_root` builder + `BlockRoot` | 429 | |
| `memory_sim` (0 ops refs) | 529 | |
| `analytic_sim` (ops = duration lookup only) | 260 | |
| proto-level overlap rewrites (`transforms.py:110-383`) | 274 | |
| `viz` (never sees a `Program`) | 151 | |
| **Typed IR + genuinely pure consumers** | **~1,790** | **23%** |
| `et_emit` 644 · `validate` 256 · `layout` 330 · IR dataclasses ~250 · `retime` 130 · `block` 182 | | |
| Aspirational / dead / test-only: `ProgramBuilder` 150, `apply_tp_overlap` + scaffolding 174 (`transforms.py:66-79`: "the proto-level port … is the only overlap implementation production uses"), `shadow.py` 286 | ~610 | 8% |

---

### Bottom line

The rewrite is real work — `et_emit` is a genuine improvement (one tag per transfer makes the tag-divergence bug unrepresentable; the always-on group-order postcondition is the right shape), `RankLayout` unification and `ScheduleSpec` are real de-duplication, and the golden matrix held. But on the lens you asked about: **the op list is decorative for every consumer except the emitter, and non-authoritative even where it holds unique data.** What was deleted was three *files*; what survives, under new class names, is the same untyped mutable DAG plus the same 1,100-line converter, now called an "emission ordering pass" and declared permanent. DESIGN.md §1's core claim ("one Program … every consumer is a pure function over it") is not honored. The owner's fear is correct, and the mechanism is precisely the one he named: the hacks were re-typed rather than replaced, and the second representation was blessed rather than eliminated.

**Cheapest path to actually closing it** (in order): add ordered successors (or stop sorting `deps`) and widen `size_bytes` to float in `ir.py` → port `analytic_sim`/`memory_sim` off `children`/`parents` → then `meta.misc["*_proto_root"]` has one remaining reader (`legacy_lowering`), and the M9 `program` id policy can delete it. Until then, `legacy_lowering.py` is not an ordering pass — it is the compiler, and `ir.py` is its debug dump.