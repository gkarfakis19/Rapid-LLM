# Rewrite context pack: RAPID-LLM graph construction → AstraSim execution

Authoritative shared context for the `rewrite_astra` effort. Read this before
designing or implementing. Everything here was verified against the code and
by experiment on 2026-07-26.

## Post-migration state (M8, current)

The migration described below is COMPLETE through M8: all four execution
modes, inference prefill + sampled decode, and memory estimation run on the
typed Program core, and the legacy graph machinery
(`simulate_train_graph.py` with `Node`/`Edge`/`Data_batch`/`Graph`,
`construct_fwd_bwd_graph`, the converter, the flattener) is deleted.
`train_timing._prepare_execution_graphs` assembles a
`program.schedule.ScheduleInputs` carrier (parallelism degrees +
comp_times/comm_metadata/misc_metadata); `LLMExecutionDispatcher` distills
it into a `ScheduleSpec`, enumerates ONE GPipe schedule
(`build_pipeline_events`), and builds per-mode Programs from it: COARSE
(`pipeline_coarse` + `analytic_sim`/`retime` for analytical/hybrid,
`lower_coarse_for_emission` for hierarchical emission), FINE
(`pipeline_fine` for flattened execution and the `memory_sim` replay), and
BLOCK (`block_program` for the transformer AstraSim runs) — all emitted
through `et_emit`. Two deliberate non-goals of the retirement: the proto
event graphs stay alive alongside the op lists (the analytical evaluator's
tie discipline, the memory replay's FIFO order, and the pinned emission
order are defined over children-list adjacency order), and
`program.legacy_lowering.lower_to_program` stays as the permanent
emission-ordering pass over those events (its name records its converter
lineage). Module inventory: `program/__init__.py`. Test-tier map (golden
gate, always-on suites, env-gated determinism sweeps, run environment):
`docs/rewrite/TESTING.md`.

Everything below this section is HISTORICAL: it documents the legacy
architecture as it existed when the rewrite began (the golden-gate
contract, AstraSim workload rules, and design constraints remain valid).

## What is being replaced (historical)

The path from "model + hardware config" to "simulated time" currently flows:

1. `train_timing.py` / `inference_timing.py` compute per-op timings and build
   TWO untyped mutable DAGs via `simulate_train_graph.Graph`:
   - a **pipeline graph** (`construct_fwd_bwd_graph`, ~630 lines): one node
     per (microbatch, layer, direction), `hw_id` = pipeline stage, GPipe
     cross-microbatch edges, DP/ZeRO-2/3 gather-edge lattice grafted on via
     `attach_parallel_edge` with cross-device special cases.
   - a **transformer graph** (`construct_transformer_graph`, ~350 lines): one
     layer's GEMM chain per (tp, cp, ep) rank, `hw_id = tp + cp*tp + ep*tp*cp`,
     MoE hot/cold-rank join nodes.
   DAG vertices are `Node` (compute), `Edge` (communication; a *vertex*, not
   an edge), `Data_batch` (root token). Adjacency via `parents`/`children`.
2. `LLMExecutionDispatcher` (`llm_execution.py`) derives the rank layout
   descriptor (axis order/sizes/strides; canonical order tp,cp,ep,pp,dp) from
   `hw_config.network_layout.dimensions`, and dispatches one of 4 modes:
   - `analytical`: `convert_comm_sizes_to_times` + `Graph.simulate`
     (list-scheduler with per-hw_id exclusivity).
   - `hybrid`: transformer graph → AstraSim (fwd+bwd separately, dp=1);
     measured times written back onto pipeline-graph layer nodes
     (per-DP tuples via `duration_profile`); pipeline simulated analytically.
   - `full_astrasim_hierarchical`: like hybrid, but the pipeline graph also
     goes through AstraSim (PP/DP axes only).
   - `full_astrasim_flattened`: `PipelineGraphFlattener` clones the pipeline
     graph while expanding each layer node into `tp*cp*ep` per-rank GEMM
     chains (name-string dispatch; ZeRO-3 sibling-scan; `"bwd" in name`
     direction hack; ±par_degree hw-id offsets), then `apply_overlap_transforms`
     splits nodes/edges for TP/CP overlap, then `_propagate_local_hw_ids`
     backfills edge placements. MoE is rejected in this mode.
3. `astrasim_lib/executor.py::convert_rapid_llm_graph_to_chakra_et` (~1100
   lines, one function) re-derives everything: stages from `hw_id`s, dp-major
   ranks (`rank = dp_idx * num_stages + stage_idx`), per-edge stage
   attribution, two near-duplicate backward dependency walkers, per-stage
   Kahn toposort, communicator reconstruction by edge-name grouping
   (`_assign_collective_labels`, TP group ids from 1000, `comm_groups.json`),
   optional SCOTCH stage remap (`gmap.py`), then per-rank Chakra ET emission
   with send/recv materialization and control-send renumbering.
   DP replication is phantom: one trace per stage is stamped per dp index.
4. `integration.py` shells out to the AstraSim analytical binary and parses
   `sys[i], Wall time:` lines. `memory_estimation.py` replays a **flattened**
   graph (any mode) for peak memory; it consumes `mem_kind`, `hw_id`,
   `duration`, `parents/children`, `fwd`, `recompute`, `param_gather`.

Interleaved pipelining is NOT in any graph: it is a closed-form correction
(`_pipeline_interleave_scale`) applied to the simulated total.

## The AstraSim workload contract (verified by experiment)

These rules were reverse-engineered from `astra-sim/workload/Workload.cc` +
`HardwareResource.cc` and confirmed with minimal hand-built bundles
(see `equiv/dlsim.py`, which replays them causally):

- Per rank, ONE compute op and ONE comm (SEND or COLL) op in flight at a
  time. RECV never occupies a slot and is posted as soon as deps are met.
- Among ready ops wanting a slot, the LOWEST node id issues first
  (`std::set` iteration). Node ids are therefore scheduling priorities.
- SEND completes unconditionally once issued (delivery buffered); RECV
  completes once the matching SEND (src, dst, tag) has been issued —
  early/late posting both fine.
- Collectives within a communicator group are matched by PER-RANK ISSUE
  ORDER (streams), not by name/tag/size. If two members of a group issue
  that group's collectives in different relative orders, the simulation
  deadlocks SILENTLY (exit 0, no wall-time lines).
- Single-member groups deadlock native ring collectives (legacy emits a
  zero-duration COMP no-op instead).
- AstraSim needs ≥2 ranks (legacy duplicates the ET for 1-rank runs and
  prunes the fake result).

Legacy deadlock bugs found & fixed on this branch (commit 85894c6):
per-stage toposort tie-breaking made group collective orders diverge across
member ranks; and cross-stage deps with no pipeline Edge object drew p2p
tags independently on send/recv side. Both are *design* consequences: the
converter re-infers a global schedule it was never given.

## The equivalence safety net (must stay green at every step)

`equiv/` + `tests/test_equiv_golden.py`: 42 golden specs (35 at rewrite
start; fault, gmap/mesh2d, GQA, ViT and ga2 rows were added since — 4
backends × parallelism rows × ZeRO-2/3 × grad-accum × recompute ×
MoE(hybrid/hier) × inference). Gates per spec: (1) per-rank op multisets,
(2) dependency-DAG Merkle hashes, (3) exact AstraSim per-rank wall
seconds, (4) end-to-end reported times, (5) dlsim completability.
Regenerating goldens is allowed only for deliberate, justified behavior
changes, committed with the diff. See `docs/rewrite/TESTING.md` for the
full test-tier map and run commands.

Environment: run via `./.venv/bin/python`, with
`LD_LIBRARY_PATH=/u1/ee/karfakis/gcc-10.2.0/lib64:...anaconda3/lib` for the
AstraSim binary (see astra-sim/build/astra_analytical/build/bin).

## Constraints on the new design

1. All four execution modes + inference (prefill & sampled decode) + memory
   estimation keep working; goldens stay green through every landing stage.
2. Deadlock-freedom and determinism BY CONSTRUCTION: per-group collective
   issue orders must be projections of one global total order; p2p pairs
   share one identity; emission is fully deterministic.
3. One placement/scheduling source of truth: no re-inference of stages,
   groups, or dependencies downstream of construction.
4. MoE currently works in hybrid/hierarchical; the design must make
   flattened-MoE *possible* later without violence.
5. The rank-layout descriptor (axis order/sizes/strides) remains the bridge
   to `config_generation.py` / faults / gmap; keep those consumers working
   (SCOTCH remap and fault projection may be adapted, not dropped).
6. Overlap transforms (tp/tp_sp/cp partial overlap) must be expressible.
7. Python 3.11, no new heavy deps. Tests: pytest; suite must stay fast
   (whole golden matrix currently ~15 s wall on 8 workers).
