# Rewrite context pack: RAPID-LLM graph construction → AstraSim execution

Authoritative shared context for the `rewrite_astra` effort. Read this before
designing or implementing.

- **§"Current architecture"** describes the tree as it is now (post-restructure,
  P5/P6 landed, rebaseline approved in `70938f8`). Verified against the code.
- **§"What is being replaced" is HISTORICAL** — the legacy architecture as it
  existed when the rewrite began. Keep it: every delta table in
  `docs/rewrite/restructure/` cites it, and `program/` docstrings still name its
  files and line numbers as provenance. Do **not** use it to reason about how
  the code works today; none of the modules it names still exist.
- **The three sections after it are STILL BINDING**: the AstraSim workload
  contract (reverse-engineered and confirmed by experiment — it is what
  `equiv/dlsim.py` replays), the equivalence safety net, and the constraints on
  the design. Each is marked in its own heading.

## Current architecture (post-restructure)

Five levels, **one representation**, **one construction path**. The normative
interface spec is `docs/rewrite/restructure/INTERFACES.md`; the module
inventory is the docstring of `program/__init__.py`.

```
L0  program/workload.py   WorkloadSpec / RunPolicy / DurationTable / CommSpecTable
L1  program/work.py       WorkItem, WorkSet, SyncRequirement (what work exists, no order)
    program/policies/     sharding · gradaccum · recompute · routing · overlap
L2  program/placement.py  Placement, Granularity (COARSE | FINE | BLOCK), BlockExpander
    program/block.py      BlockTemplate / CommMeta
    program/groups.py     CommunicatorFactory — members CONSTRUCTED from (axis, coords)
    program/layout.py     RankLayout
L3  program/schedule/     SchedulePolicy, Schedule, GPipeSchedule
L4  program/ir.py         Program: ops with explicit deps/groups/transfers
    program/build.py      build() — the ONLY Program constructor
    program/validate.py   invariants V1–V10
L5  program/et_emit.py    Program -> Chakra ET bundle (id policy = program order)
    program/analytic_sim.py · memory_sim.py · retime.py · mapping.py · viz.py · shadow.py
```

**The flow.** `train_timing` / `inference_timing` compute the timing and byte
*math* and hand it to `WorkloadSpec.from_timing` — the single producer seam
(INTERFACES §1.7), which reads every `misc_metadata` key as REQUIRED and raises
naming the key rather than defaulting. `LLMExecutionDispatcher`
(`llm_execution.py`) then calls `program.build.build()` once per artifact. The
four execution modes, the BLOCK measurements and the memory replay differ
**only** in the `Granularity` they request and in which L5 consumer reads the
result:

| mode | granularity | consumer |
|---|---|---|
| analytical | COARSE | `analytic_sim.evaluate` |
| hybrid | COARSE | BLOCK retime, then `analytic_sim.evaluate` |
| full_astrasim_hierarchical | COARSE | BLOCK retime, then AstraSim over (pp, dp) |
| full_astrasim_flattened | FINE | AstraSim over the full layout |
| transformer blocks | BLOCK | AstraSim, one bundle per direction |
| memory estimation | FINE | `memory_sim.simulate_memory` |

**Dependencies are computed, not walked.** `build()` is mechanical and makes no
policy decision of its own; it composes L1+L2+L3 in a fixed phase order and
derives every edge from a numbered rule (INTERFACES §4.3): **R1** data flow
inside a block chain from the `BlockTemplate`; **R2** cross-layer and the
stated intra-microbatch edges (`SOFTMAX/F(b) → SOFTMAX/B(b)`,
`LAYER/F(b,l) → RECOMPUTE(b,l)`), with a cross-device pair becoming a
`TransferOp`; **R3** device serialization implied by the `SchedulePolicy`,
added only when not already transitively implied, and **per DEVICE** — this is
the rule the rebaseline turned on (`REBASELINE.md` §2.1, §5.1); **R4** sync
deps from each `SyncRequirement`'s declared attach mode, materialized as
1-byte control pairs when they cross ranks (emission is *total*: an uncarriable
cross-rank dep is an `EmissionError`, never a dropped edge); **R5** the
optimizer after every backward item of its stage.

**Program order** = a deterministic function of (schedule step, device,
intra-step index), and it is simultaneously the ET node-id priority AstraSim
consumes and the tie discipline the analytical evaluator and the memory replay
use. There is no `legacy_op_id` / `send_seq` / `recv_seq` / `post_deps`.

**What is gone.** `simulate_train_graph.py`, `legacy_lowering.py`,
`schedule.py`'s `build_pipeline_events`, `pipeline_fine.py`,
`pipeline_coarse.py`, `block_program.py`, `transforms.py`, the proto event
graphs (`ComputeEvent`/`CommEvent`/`FineNode`/`FineEdge`), the clone cache, the
`"bwd" in name` dispatch, the `ScheduleInputs` dict-carrier,
`ScheduleSpec.from_pipeline_graph`, `TransformerBlockSpec`, `Op.succs`,
`et_emit`'s `legacy` id policy and `CrossRankDepWarning`. Docstrings across
`program/` still *cite* those files by name and line — that is deliberate
provenance for the delta tables, not a live reference; nothing imports them.

**Where the remaining seams are.** `restructure/BUG_LEDGER.md` is the authority
on which defects are fixed and which are filed-but-open; check it rather than
this file. The two that shape the code you will read are **A2** — whether a
stage-spanning collective is placed on cluster rank 0 only or on every cluster
rank (`SyncSpread` in `policies/sharding.py`), which is what
`build(check_group_membership=...)` and invariant **V7** hinge on — and **A6** —
the ZeRO-3 prefetch anchor chosen by layer arithmetic, which can land on
another device. Both move predictions, so each needs its own delta table and
owner approval before landing.

Test-tier map (golden gate, always-on suites, the one surviving env-gated
sweep, the ledger, run environment): `docs/rewrite/TESTING.md`.

---

The next section is HISTORICAL; the three after it are still binding.

## What is being replaced (historical)

The path from "model + hardware config" to "simulated time" USED TO flow (all
of this is deleted; kept for provenance):

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

## The AstraSim workload contract (STILL BINDING — verified by experiment)

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

## The equivalence safety net (STILL CURRENT — must stay green at every step)

`equiv/` + `tests/test_equiv_golden.py`: the golden matrix (35 specs at
rewrite start, 42 after the fault, gmap/mesh2d, GQA, ViT and ga2 rows —
`equiv.configs.build_matrix()` is the authority on the current count and
`--list` prints it). 4 backends × parallelism rows × ZeRO-2/3 × grad-accum ×
recompute × MoE × inference. Gates per spec: (1) per-rank op multisets,
(2) dependency-DAG Merkle hashes, (3) exact AstraSim per-rank wall
seconds, (4) end-to-end reported times, (5) dlsim completability.
The tiers are now split into named tests (T1 structural / T2 timing /
T3 contract + determinism / T4 ledger); regenerating goldens is allowed only
for deliberate, justified behavior changes, committed with the diff **and**
with a reviewed delta table. See `docs/rewrite/TESTING.md` for the full
test-tier map and run commands, and
`docs/rewrite/restructure/REBASELINE.md` for the one rebaseline that has been
approved so far.

Environment: run via `./.venv/bin/python`, with
`LD_LIBRARY_PATH=/u1/ee/karfakis/gcc-10.2.0/lib64:...anaconda3/lib` for the
AstraSim binary (see astra-sim/build/astra_analytical/build/bin).

## Constraints on the new design (STILL BINDING)

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
