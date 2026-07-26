**Verdict: the owner is right for the DP/ZeRO axis. The rewrite retyped the lattice; it did not replace it. Extension ratio ≈ 0.9 (new ~318 lines / 11 files vs legacy ~350 / 10).**

Plan written to `/u1/ee/karfakis/.claude/plans/warm-squishing-stream-agent-a0613c092a0550665.md`.

Hard evidence, before the exercise:

- Normalized diff of legacy `construct_fwd_bwd_graph` (`85894c6:simulate_train_graph.py:688-1322`, 550 lines) vs `program/schedule.py:478-1079` (540 lines): **282 identical lines, SequenceMatcher 0.517**. The delta is the rename `self.create_comm_edge(name, op_id, key)` → `_comm_event(name, key)`.
- `attach_parallel_edge` with `skip_non_comm_children` / `skip_comm_children` survives byte-identical at `program/schedule.py:553-567` (was `simulate_train_graph.py:713`).
- Name-string dispatch survives verbatim: `program/pipeline_fine.py:417` `if "linear_softmax" in obj.name`, `:426` `elif "optimizer" in obj.name`, `:705` `"optimizer" in child.name` (legacy `llm_execution.py:529,538,884`).
- The advertised de-hack is cosmetic. `program/pipeline_fine.py:353-356`:
  ```python
  hint = getattr(edge, "zero3_offset_hint", None)
  if hint is None:
      hint = "bwd" if "bwd" in edge.name else "fwd"   # legacy hack retained as fallback
  stage_delta = 1 if hint == "fwd" else -1            # ±par_degree hack, retyped
  ```
  The delta is computable in scope (`edge.local_hw_id` vs `stage_id`, used two frames up at `:716`) and isn't computed.
- Six ZeRO string keys hard-coded at `schedule.py:779-1046`, matched by six hard-coded blocks at `train_timing.py:4453-4507` and six float kwargs threaded through three signatures. No `ShardingPolicy` type exists anywhere in `program/`.

Where the feature breaks the new core:
- `schedule.py:873` `host = transformer_nodes[b][layer_idx-1]` and `:1018` `_bwd_exit_node(b, layer_idx+1)` — the `∓1` **is** the prefetch depth.
- `schedule.py:882-892`: `skip_non_comm_children=True` routes the gather's dependency through host's single `cross_layer` child. Correct only because D=1 makes host/target adjacent. At D=2, pp≥3, they are two hops apart and no child of host reaches target's device → new attach mode (~25 lines). **This is the "new lattice of special-cased edges" the prompt predicted, and the new code needs it too.**
- `pipeline_fine.py:377-388` copies the host tail's children onto the per-rank gather — at D≥2 that silently wires it to layer host+1's consumers.
- `ir.py:126` `param_gather: bool` + `memory_sim.py:484-505` (alloc at a node's issue, release at the **same** node's finish) cannot express "bytes held across 2 layers" at all. ~90 lines + an IR extension.

Genuine dividends, quantified: `ScheduleSpec` field (~4 lines saved), `_offset_stage_device`'s real coordinate hop (~15 saved vs writing it from scratch in legacy), and two assertions that cost 0 lines here — `et_emit.py:485` group-order postcondition (turns the silent-deadlock class loud) and `pipeline_coarse.py:158-167` unreachable-parent check. That's the whole win: ~19 lines and two guards.

Aggravating: `DESIGN.md` §1 claims "flattening ceases to exist as a graph rewrite", but `_FineExpander._clone` (`pipeline_fine.py:403-513`) is a clone-cache graph rewrite whose output is fed to `legacy_lowering.lower_to_program` (`pipeline_fine.py:950`). The docstrings at `schedule.py:478` and `pipeline_fine.py:211` are honest ("verbatim port", "kept verbatim"); DESIGN.md §1 is not.