# RAPID-LLM Rewrite Design: The Global Schedule as the Spine

Replacement design for the graph-construction → AstraSim execution pipeline. Verified against the code as of this branch (`simulate_train_graph.py`, `llm_execution.py`, `astrasim_lib/executor.py`, `train_timing.py::_prepare_execution_graphs`, `memory_estimation.py`, `equiv/`) and the context pack at `docs/rewrite/CONTEXT.md`.

---

## 0. The stance, argued

Today the system builds **mutable DAGs first** and then *re-infers* everything downstream: `convert_rapid_llm_graph_to_chakra_et` re-derives stages from `hw_id`s, re-walks dependencies twice, reconstructs communicators by name-grouping, and re-invents a per-stage order via Kahn toposort. Both legacy deadlock bugs (commit 85894c6) were consequences of exactly this: the converter re-infers a global schedule it was never given, so per-group collective orders and p2p tag pairing were only *accidentally* consistent.

The replacement inverts the data flow. The core object is a **`GlobalSchedule`**: an explicit, partially-ordered set of `Step`s (compute / collective / p2p) with a **defined, deterministic linearization**. Everything else is a projection:

- Each rank's Chakra ET is the schedule restricted to that rank. Per-group collective issue order on every member is the restriction of one shared order → consistent by construction. Every p2p is *one* step carrying both endpoints and one tag → pairing consistent by construction.
- The analytical simulator is a **schedule evaluator** (list-scheduler over steps), not a graph walker.
- Memory replay is the same evaluator with a memory probe attached.
- GPipe is a **schedule generator**; interleaved-1F1B becomes a second generator emitting the same step vocabulary — no more closed-form bubble correction for natively generated schedules.
- Hybrid/hierarchical retiming is a tag-indexed rewrite of step durations, not a graph traversal.

One honest caveat, followed where it leads: the golden gates pin **exact AstraSim wall seconds**, and AstraSim node ids are scheduling priorities. Legacy ids come from a *per-stage* Kahn sort that ignores cross-stage deps, plus a control-send renumber. A pure "ids = global-order projection" policy would change ids and therefore wall times on some specs. The design therefore separates **schedule order** (`sid`, always a global topological order — the semantic spine) from **ET id policy** (a per-rank emission ordering). Two policies exist: `legacy` (bit-compatible with today's converter; per-group order consistency *verified* at emission) and `sid` (pure projection; consistency *by construction*). Existing modes ship with `legacy` and never regen goldens; new capabilities (native 1F1B, flattened MoE) use `sid`. This is the maximal honest version of the stance: the global order is always the source of truth; the legacy id policy is a checked, documented projection quirk kept only for golden exactness.

---

## 1. Core abstractions

All in a new package `sched/`. Python 3.11, stdlib + existing deps only. `CollectiveType` and `MemKind` are reused from `timing_model.py` / `memory_estimation.py` unchanged.

### 1.1 `sched/layout.py` — rank layout (the bridge that stays)

```python
Axis = Literal["tp", "cp", "ep", "pp", "dp"]
CANONICAL_AXIS_ORDER: tuple[Axis, ...] = ("tp", "cp", "ep", "pp", "dp")

@dataclass(frozen=True)
class RankLayout:
    axis_order: tuple[Axis, ...]          # active subset, canonical order enforced
    axis_sizes: dict[Axis, int]           # all five keys present, each >= 1
    axis_strides: dict[Axis, int]         # derived in __post_init__ (row-major over axis_order)
    world_size: int                       # derived: product of sizes over axis_order

    def rank_of(self, coords: Mapping[Axis, int]) -> int
    def coords_of(self, rank: int) -> dict[Axis, int]
    def group_along(self, axes: tuple[Axis, ...], fixed: Mapping[Axis, int]) -> tuple[int, ...]
        # sorted member ranks varying `axes`, others fixed — communicator membership source of truth
    def sublayout(self, axes: Sequence[Axis]) -> "RankLayout"
        # e.g. transformer sublayout (tp,cp,ep); pipeline sublayout (pp,dp)
    def descriptor(self) -> dict          # legacy dict {"axis_order","axis_sizes","axis_strides","stage_span"}
                                          # for config_generation / gmap / fault consumers during migration
    @staticmethod
    def from_network_layout(network_layout, *, tp,cp,ep,pp,dp, enforce_cluster_leading: bool) -> "RankLayout"
        # verbatim port of LLMExecutionDispatcher._build_rank_layout_descriptor validation
        # (dim-size cross-checks, hierarchical leading-axes rule, optimize_2dmap extraction)
```

`from_network_layout` also returns the `optimize_2dmap` config (as `Optional[dict]`) so the dispatcher no longer owns that parsing.

### 1.2 `sched/ir.py` — steps and schedules

```python
class StepKind(Enum):
    COMPUTE = "compute"
    COLLECTIVE = "collective"
    P2P = "p2p"                # src==dst means "local sequencing edge" (legacy same-stage PIPELINE Edge)

@dataclass(frozen=True)
class StepTags:
    microbatch: Optional[int] = None
    layer: Optional[int] = None
    segment: Optional[str] = None       # "embedding"|"transformer"|"softmax"|"optimizer"|gemm entry name
    direction: Optional[str] = None     # "forward"|"backward"
    stage: Optional[int] = None         # pipeline stage id
    vstage: Optional[int] = None        # virtual stage for interleaved-1F1B; None for GPipe
    recompute: bool = False
    param_gather: bool = False
    is_moe: bool = False
    mem_kind: Optional[MemKind] = None

@dataclass(frozen=True)
class ComputeOp:
    name: str
    duration: float                                # seconds
    duration_profile: Optional[tuple[float, ...]] = None   # per-dp override (hybrid retiming)

@dataclass(frozen=True)
class CollectiveOp:
    name: str                                      # human-readable; NEVER used for matching
    kind: CollectiveType                           # ALL_REDUCE / ALL_GATHER / REDUCE_SCATTER / ALL_TO_ALL
    size_bytes: int
    group_id: int                                  # index into GroupTable
    axis: str                                      # interconnect label: "tp"|"cp"|"ep"|"dp"|"pp"
    participants: int                              # kept for network_model + gmap parity

@dataclass(frozen=True)
class P2POp:
    name: str
    size_bytes: int                                # 0 => control (emitted as 1 byte, "control" naming)
    src: int                                       # logical rank (see §1.4 on dp stamping)
    dst: int

@dataclass(frozen=True)
class Step:
    sid: int                    # dense index; after finalize(), steps[sid].sid == sid and it is a topo order
    kind: StepKind
    op: ComputeOp | CollectiveOp | P2POp
    rank: Optional[int]         # COMPUTE only; None otherwise
    deps: tuple[int, ...]       # sids of predecessors; finalize() guarantees all deps < sid
    prio: int                   # generator creation index — the legacy-op_id analogue; linearization key
    tags: StepTags

def footprint(step: Step, groups: "GroupTable") -> frozenset[int]:
    # COMPUTE -> {rank}; COLLECTIVE -> group members; P2P -> {src, dst}

@dataclass
class GroupTable:
    groups: list[tuple[int, ...]]                  # index == group_id; members sorted ascending
    _index: dict[tuple[int, ...], int]
    def intern(self, members: Sequence[int]) -> int   # dedup; first-use order fixes group_id

@dataclass(frozen=True)
class ScheduleMeta:
    kind: str                    # "pipeline" | "transformer" | "flattened"
    dp_stamped: int              # 1 = dp explicit in ranks; N>1 = emitter replicates per dp index (phantom DP)
    interleave: int = 1          # >1 only for schedules produced by the native 1F1B generator
    optimize_2dmap: Optional[dict] = None
    run_type: str = "training"

@dataclass
class GlobalSchedule:
    layout: RankLayout           # logical-rank space of this schedule (may be a sublayout)
    groups: GroupTable
    steps: list[Step]            # finalized
    meta: ScheduleMeta
    def rank_steps(self, rank: int) -> list[Step]                  # restriction, sid order
    def validate(self) -> None
        # (a) dense sids, deps < sid (acyclicity by construction)
        # (b) locality: every dep of step s has footprint intersecting footprint(s);
        #     for COMPUTE at rank r, every dep attached on r's ET must contain r (checked at emission)
        # (c) group sanity: members within layout.world_size
```

**Dep semantics for collectives.** A `COLLECTIVE` step is one step per *group instance* (all members). Its `deps` may reference predecessors on any member's chain; at ET emission, member rank `r` receives only the deps whose footprint contains `r`. This reproduces the legacy per-member chains (each rank's collective node depends on that rank's predecessor) with a single schedule object, which is what makes per-group ordering a projection.

### 1.3 `sched/builder.py`

```python
class ScheduleBuilder:
    def __init__(self, layout: RankLayout, meta: ScheduleMeta): ...
    # each returns the step's provisional sid (== creation index == prio)
    def compute(self, name, *, rank, duration, deps=(), tags=StepTags(), profile=None) -> int
    def collective(self, name, *, kind, size_bytes, members, axis, deps=(), tags=StepTags()) -> int
    def p2p(self, name, *, src, dst, size_bytes, deps=(), tags=StepTags()) -> int
    def finalize(self) -> GlobalSchedule
        # Deterministic linearization: Kahn over the dep DAG, ready-set popped by min prio
        # (heap keyed (prio, seq)). Creation order is already near-topological for all
        # generators; finalize makes it exactly topological and reproducible.
```

`finalize()` is the *defined linearization* of the stance: the schedule is a partial order; `sid` order is its canonical total order.

### 1.4 DP phantom stamping (unchanged semantics, now explicit)

Training pipeline/flattened schedules are built **dp-generic**: logical ranks are flattened hw ids (stage × tp·cp·ep coordinates), `meta.dp_stamped = dp_count`. The emitter replicates each rank's trace per dp index with `physical_rank = dp_idx * num_stages + stage_index` (dp-major, exactly today's rule), reads `ComputeOp.duration_profile[dp_idx]` when present, instantiates dp collectives with per-stage group `str(stage_idx + 1)`, and keeps p2p within one dp replica. Transformer schedules have `dp_stamped = 1`. Inference forces `dp_stamped = 1` (today's `dp_override=1`).

---

## 2. Module layout

```
sched/
  __init__.py                     public surface                                    ~30 LOC
  layout.py                       RankLayout + from_network_layout validation       ~220
  ir.py                           Step/ops/tags/GroupTable/GlobalSchedule           ~260
  builder.py                      ScheduleBuilder + finalize linearization          ~150
  generators/
    __init__.py
    pipeline_program.py           PipelineProgram + DpCommPolicy inputs (see §3.2)  ~120
    pipeline_gpipe.py             GPipe generator (replaces construct_fwd_bwd_graph) ~480
    pipeline_1f1b.py              native interleaved-1F1B generator (stage S6)      ~260
    transformer.py                microplan generator incl. MoE hot/cold joins      ~320
  expand.py                       pipeline schedule × microplans -> flattened sched ~280
  transforms.py                   overlap splits (tp/tp_sp/cp), retime, misc        ~220
  evaluate.py                     analytical schedule evaluator (+ probe hooks)     ~260
  memory.py                       MemoryProbe (training/inference policies)         ~140
  et_emit.py                      Chakra ET emission, id policies, comm_groups,
                                  manifest, singleton no-ops, control renumber      ~380
  gmap_feed.py                    feed GMapCollector from a schedule; permutation   ~120
  faults.py                       FaultSpace wiring on RankLayout; per-(dp,stage)
                                  transformer fault variant table                   ~100
  dispatch.py                     mode orchestration (replaces LLMExecutionDispatcher
                                  internals; keeps its public constructor/run API)  ~320
  viz.py                          schedule -> graphviz (RAPID_VISUALIZE_GRAPHS)     ~130
  compat.py                       TEMPORARY: legacy-Graph -> schedule checks,
                                  bundle-diff harness against equiv.canonical       ~200 (deleted at S5)
tests/sched/                      unit + parity tests                               ~700
```

Total new code ≈ 3,300 LOC replacing ≈ 4,600 LOC of legacy construction/conversion (deleted per §4).

Untouched modules: `astrasim_lib/{config_generation, integration, et_utils, gmap (core), fault_projection, layout_utils (until S5 deletion)}`, `timing_model.py`, `memory_estimation.py` (the `MemoryEstimator` byte accounting), all of `train_timing.py`'s timing math, `equiv/`.

---

## 3. Generators and consumers

### 3.1 Transformer microplan generator (`generators/transformer.py`)

Input: the exact `transformer_cfg["gemms"]` entries and `comm_metadata` that `_prepare_execution_graphs` builds today (`CommSpec`-derived; unchanged). Output: a `GlobalSchedule` over `layout.sublayout(("tp","cp","ep"))` for one direction (`forward`/`backward`), or a **microplan template** (parameterized step list) for use by `expand.py`.

Per (ep, cp, tp) rank, in legacy creation order: pre-comm chain → GEMM compute step (`param_gather = (idx == 0)`, `mem_kind = mem_kind_from_op_name`) → post-comm groups. MoE parallel post groups become: base all-to-all `COLLECTIVE` step, per-rank zero-duration join `COMPUTE` step, and cold→hot residual `P2P` steps — replacing the `moe_parallel_joins` dict plumbing with explicit steps. Collectives are one step per group instance with members from `layout.group_along(axis, fixed_coords)`.

### 3.2 GPipe pipeline generator (`generators/pipeline_gpipe.py`)

Input dataclass (assembled by `train_timing`/`inference_timing`, replacing the `comp_times`/`comm_metadata`/`misc_metadata` dict soup at the boundary but carrying identical values):

```python
@dataclass(frozen=True)
class PipelineProgram:
    num_microbatches: int
    num_layers: int
    layers_per_stage: tuple[int, ...]          # same remainder-first split as today
    seg_durations: dict[str, float]            # embedding_f/b, softmax_f/b, transformer_{f,b}_{dense,moe}, optimizer
    comm: dict[str, CommMeta]                  # cross_layer, embedding, softmax, transformer_{dense,moe},
                                               # zero2_*/zero3_* gathers, *_ep_sync — sizes/kinds/participants
    moe_layer_mask: tuple[bool, ...]
    include_backward: bool
    include_optimizer: bool
    recompute: RecomputePolicy                 # full_recomputation × (flattened_mode | pipeline_style)
    dp_policy: DpCommPolicy                    # dp, zero_stage, dp_microbatch_mode, grad_accum_cycle,
                                               #   ga_required_every_cycle flags  (== _should_emit_dp_comm)
    model_type: str
```

Output: `GlobalSchedule` over stage-granular logical ranks (`dp_stamped = dp`). It emits, in the **same creation order as `construct_fwd_bwd_graph`** (this fixes `prio`):

- per-(mb, layer) transformer `COMPUTE` steps, embedding/softmax steps, recompute steps;
- every legacy `Edge` as a `P2P` step — including same-stage zero-byte PIPELINE edges as `P2P(src==dst, size=0)` steps, so the analytical evaluator sees the same event population as `Graph.simulate` (see §3.4);
- GPipe cross-microbatch dependency edges as pure deps;
- DP/ZeRO-2/3 gather lattice as `COLLECTIVE` steps with `axis="dp"`, gated by `DpCommPolicy` (verbatim port of `_should_emit_dp_comm` including the `b == 0` backward-reversal rule and the nonfinal-cycle rules), with the cross-device ZeRO-3 attachment quirks (`skip_non_comm_children` / `skip_comm_children`) reproduced as explicit dep wiring;
- EP dense-sync collectives and optimizer steps per stage.

**Interleaved-1F1B (`pipeline_1f1b.py`, later stage)** consumes the same `PipelineProgram` plus `interleave: int` and emits the classic 1F1B slot pattern with `vstage` tags. Because it produces the same step vocabulary, *every* consumer (ET, evaluator, memory, retiming, expansion) works on it with zero changes; `meta.interleave > 1` disables the `_pipeline_interleave_scale` correction in `dispatch.py`. This is the payoff of the schedule-first design: pipelining strategies are pluggable generators, not graph surgeries.

### 3.3 Expansion (`expand.py`) — the flattener, reborn

`expand(pipeline_schedule, microplans: dict[key, MicroplanTemplate], layout: RankLayout) -> GlobalSchedule`

Replaces `PipelineGraphFlattener` entirely. For each pipeline `COMPUTE` step tagged `segment="transformer"`, splice in the (dense or MoE) microplan template instantiated at `hw = layout.rank_of(stage, tp, cp, ep)` — placement via `RankLayout`, deleting `_hw_id_for_rank`'s hand-rolled arithmetic. Cross-stage `P2P` steps split per parallel rank with `ceil(total / par_degree)` bytes (legacy quirk kept). ZeRO-3 `tp_shard` gathers become per-rank `COLLECTIVE` steps with legacy byte semantics (`int(base_bytes)` per rank, un-divided — quirk kept) and the ±stage-offset placement expressed as explicit stage coordinates instead of the `"bwd" in name` hack — the direction comes from `tags.direction`. Optimizer/softmax/embedding expansion follows the legacy rules (optimizer 1:1 per rank; softmax pinned to rank 0 of its stage). **No name-string dispatch anywhere: all decisions read `StepTags`.** Because MoE microplans are ordinary templates, flattened-MoE later is just "stop raising in dispatch" plus new golden specs (constraint 4 satisfied without violence).

### 3.4 Analytical evaluation (`evaluate.py`)

`evaluate(schedule, *, network_model, interconnect_params, probes=()) -> EvalResult(total_time, finish: list[float])`

The schedule evaluator replaces `convert_comm_sizes_to_times` + `Graph.simulate`:

- Comm durations computed lazily per step via `network_model.collective(kind, size_bytes, participants, ib, ll, ...)` with `interconnect_params[op.axis]` — identical formula and inputs.
- Scheduling discipline mirrors `Graph.simulate` exactly: heap of `(finish_time, counter)`; a FIFO ready list scanned in insertion order on every completion; `COMPUTE` steps require the free device slot of `step.rank`; `COLLECTIVE`/`P2P` steps never occupy a slot; per-step successors are stored in **creation order** so ready-list insertion order matches legacy child order.
- Probes get `on_issue(step, t)` / `on_finish(step, t)` callbacks. `MemoryProbe` (`memory.py`) ports `simulate_memory`'s policy verbatim: static per-device bytes from dense/MoE layer counts, persistent alloc on forward / release on backward, transient alloc+release, `param_gather` ephemeral alloc at issue and release at finish, training vs inference modes, optional per-GPU logs. `MemoryEstimator.build_memory_data` is unchanged and feeds the probe. Memory replay for **all four modes** runs the probe over the *expanded* schedule (replacing `build_flattened_root_for_memory`).
- `_pipeline_interleave_scale` is applied by `dispatch.py` to the evaluator's result exactly as today when `meta.interleave == 1` and configured interleave > 1.

### 3.5 ET emission (`et_emit.py`) — contract rules honored by construction

`emit_chakra(schedule, *, dp_count, output_dir, id_policy="legacy", stage_permutation=None) -> EmittedBundle(et_prefix, rank_ids, manifest_path, comm_groups_path)`

Emission per physical rank (after dp stamping and optional SCOTCH permutation, §3.7):

- **Phase A** — `COMP`/`COLL` nodes. Under `id_policy="sid"`: in sid order restricted to the rank. Under `id_policy="legacy"`: per-stage Kahn over intra-rank dep edges with min-`prio` priority (byte-compatible with legacy Step 10, since `prio` reproduces the legacy op_id sequence and cross-stage deps are excluded from ordering exactly as legacy does). `dp_count <= 1` skips non-tp-group collectives (legacy rule). Singleton communicator groups (group cardinality 1 after membership resolution) emit a zero-duration `COMP` no-op — the contract's single-member-deadlock rule is enforced in exactly one place, for every generator, forever.
- **Phase B** — p2p materialization: for each step with cross-rank deps, SEND on the source trace / RECV on the destination trace, created in legacy iteration order (stages in order, tasks in per-stage order, dp ascending, parents by `prio`), deduplicated via the same caches keyed on (src step, dst stage, dp). Tag = the p2p step's identity (one step ⇒ one tag on both sides — the tag-mismatch bug class is structurally impossible). Zero-byte or non-PIPELINE-kind transfers emit as 1-byte `*_send_control`/`*_recv_control` (legacy rule). `src == dst` p2p steps emit no nodes — they collapse into ordinary ctrl deps.
- **Phase C** — stable control-first renumber, a verbatim port of `_RankTrace._renumber_control_priority`.
- **Postcondition check (always on, both policies):** for every communicator group, extract each member's emitted sequence of collective step ids for that group and assert all sequences are identical. Under `id_policy="sid"` this holds by construction (each member's sequence is the same subsequence of the global order); under `legacy` it converts the historical "silent deadlock" into a loud emission-time failure.
- Attributes: `pg_name` for tp/cp/ep groups uses ids interned from 1000 in deterministic first-use order (label-sorted, token-sorted — matching `_assign_collective_labels` + `_TP_MEMBERS_TO_GID` output); dp collectives use per-stage group `str(stage_idx+1)`; `comm_groups.json` and `manifest.json` writers are ports of the existing ones (manifest ops sorted by the same `_manifest_op_key`, so cache keys are stable).

**Deadlock-freedom argument** (for `id_policy="sid"`; `legacy` is covered by the postcondition check plus the fact that legacy-equal bundles are exactly the bundles the goldens already prove complete):

Strong induction on sid over the finalized schedule. Assume every step with sid < k has completed. Step k's ET nodes on each involved rank have all ctrl deps satisfied (deps have smaller sid; RECV pairing partners are part of the same step). What can occupy a rank's single comm slot ahead of k? (a) a SEND — completes unconditionally once issued (contract), so it drains; (b) a COLL with a lower node id — under sid policy, lower id ⇒ smaller sid ⇒ completed by hypothesis; RECVs never occupy slots (contract). So k's nodes issue on every involved rank. If k is a COLL on group G, every member's previously issued G-collectives are precisely the G-steps with smaller sid — the *same set in the same order* on every member (projection of one total order) — so issue-order matching pairs them correctly and the collective completes. If k is a P2P, the SEND issues (deps met, slot drains), hence the RECV completes (early or late posting both fine per contract). Phase C moves only control SEND/RECVs to the front: it never reorders COLLs relative to each other (preserving per-group order) and only lets always-completing SENDs issue earlier. Acyclicity of the global dependency relation is structural: ET deps map to sid-decreasing schedule deps, and SEND→RECV pairs are intra-step. Hence every step completes; the bundle is dlsim-completable and AstraSim-deadlock-free.

### 3.6 Hybrid / hierarchical retiming (`transforms.py::retime`)

`retime(pipeline_schedule, table: RetimingTable)` where `RetimingTable` maps `(is_moe, direction, dp_idx, stage) -> seconds` with a baseline fallback — populated from the AstraSim runs of the transformer schedules (dense fwd/bwd, MoE fwd/bwd, per-(dp, stage) fault variants, exactly the runs `_run_transformer_astrasim` performs today). The rewrite selects steps by `tags.segment == "transformer"` and writes `duration` (dp==1) or `duration_profile` tuples (dp>1) — replacing `_apply_transformer_time`'s recursive graph walk with an index lookup. Hierarchical mode then feeds the retimed pipeline schedule to `emit_chakra` (PP/DP sublayout); hybrid feeds it to `evaluate`.

### 3.7 Fault projection and gmap/SCOTCH (`faults.py`, `gmap_feed.py`)

- `FaultSpace` continues to consume the axis layout; it is constructed from `RankLayout` (via `descriptor()` during migration, natively at S5). The per-(dp, stage) transformer fault map, coverage validation, and the analytical/hybrid rejection rule move verbatim from the dispatcher into `faults.py`.
- `gmap_feed.feed(schedule, collector)` replaces the converter's inline collection: iterate steps in sid order; `COLLECTIVE` steps with axis in `collector.subset_axes` → `record_collective(axis, stage, bytes, participants, double_weight = kind in {ALL_REDUCE, ALL_TO_ALL})`; cross-stage `P2P` data steps → `record_pipeline(src_stage, dst_stage, bytes)`. Identical multigraph, no graph walking.
- The SCOTCH permutation is applied **inside `emit_chakra`** as a stage→emission-slot relabeling (the semantics of `_remap_stages_for_mapping`). Because group memberships and p2p endpoint ranks are computed *after* the relabeling from `RankLayout` + permutation, communicators, `comm_groups.json`, and send/recv endpoints stay consistent by construction — the remap can never desynchronize from the traces.

### 3.8 Inference (prefill + sampled decode)

Unchanged in structure, simpler in plumbing: `inference_timing.calc_time` builds a `PipelineProgram` (+ microplans) with `include_backward=False` for prefill and a second one from decode timings (`seq_len=1`, decode GEMM shapes), then calls the same `dispatch.py` entry points. Decode memory replay runs the memory probe over the decode expansion with decode `memory_data`. `dp_stamped=1` reproduces `dp_override=1`; the 1-rank ET duplication + result pruning stays in the runner shell (§6).

### 3.9 Overlap transforms (`transforms.py`)

Schedule rewrites replacing `apply_overlap_transforms`:

- `split_compute_for_tp_overlap(schedule, overlap)`: a `COMPUTE` step with a following tp `COLLECTIVE` splits into head `(1-o)·d` / tail `o·d` steps; the collective's dep moves to the head; the tail depends on the head (o≥1: collective hoisted before the compute). Mirrors `_split_tp_node` including edge cases.
- `split_cp_collective_for_overlap(schedule, overlap)`: a cp `COLLECTIVE` preceding attention splits into `block` (`ceil(bytes·(1-o))`) and `ovlp` (remainder) steps with legacy rewiring (`_split_cp_edge` semantics).

Applied to transformer schedules (hybrid/hier) and to the expanded schedule (flattened), at the same points in the flow as today.

### 3.10 Dispatch (`dispatch.py`)

Keeps `LLMExecutionDispatcher`'s public constructor shape and `run(mode) -> ExecutionResult` so `train_timing`/`inference_timing` call sites barely change, but internally: build programs → generate schedules → per mode: `analytical` = evaluate(pipeline); `hybrid` = emit+run transformer schedules → retime → evaluate; `hierarchical` = same + emit+run retimed pipeline schedule; `flattened` = expand → overlap → emit+run. Rank-count validation, artifact dirs, grad-accum no-dp variant orchestration, and the interleave scale stay identical.

---

## 4. Migration plan

Every stage ends with the full golden matrix green (`tests/test_equiv_golden.py`, 35 specs, all five gates), run via `./.venv/bin/python -m pytest` with the documented `LD_LIBRARY_PATH`. **No golden regeneration through S5.** Dev-loop verification within each stage uses `sched/compat.py`: emit the new bundle and the legacy bundle for the same run and diff via `equiv.canonical.diff_bundles` + per-rank node-id sequence comparison (stricter than the gates, since gate 3 needs id-identical bundles).

**S0 — Core IR (no behavior change).** Land `layout.py`, `ir.py`, `builder.py`, unit tests. `LLMExecutionDispatcher._build_rank_layout_descriptor` delegates to `RankLayout.from_network_layout(...).descriptor()`; asserts dict-equality with the old code path in tests. *Deletes: nothing.*

**S1 — Transformer schedules own the hybrid/hier transformer bundles.** Land `generators/transformer.py`, `transforms.py` (overlap), `et_emit.py`, `faults.py` (fault-variant table only). `_run_transformer_astrasim` emits fwd/bwd (dense + MoE) bundles via `emit_chakra(id_policy="legacy")` instead of passing graph roots to `convert_rapid_llm_graph_to_chakra_et`; `run_astra_simulation_only_onepath` grows a schedule entry point sharing the same config/caching shell. Gates: all `train:hybrid:*`, `train:hierarchical:*` (transformer sub-bundles), `:moe:` specs, `inf:hierarchical:*`. *Deletes: `construct_transformer_graph` remains (analytical/flattened still read gemm templates via the old path) — nothing deleted yet.*

**S2 — GPipe generator + expansion own flattened ET and memory replay.** Land `generators/pipeline_gpipe.py`, `pipeline_program.py`, `expand.py`, `evaluate.py` (probe machinery), `memory.py`, `gmap_feed.py`. Flattened mode: program → GPipe schedule → expand → overlap → emit (`legacy` policy) → run. Memory replay in **all** modes switches to the memory probe over the expanded schedule. `train_timing`/`inference_timing` assemble `PipelineProgram` from the same numbers they compute today. Gates: all `train:flattened:*` incl. `zero2/zero3/ga2/recompute`, `inf:flattened:*`, plus memory outputs byte-compared against legacy in a one-off parity test. *Deletes: `PipelineGraphFlattener` (~640 LOC), `llm_execution.apply_overlap_transforms` + `_split_tp_node`/`_split_cp_edge` (~300), `build_flattened_root_for_memory`, `Graph.simulate_memory` (~400), `MemoryEstimator.simulate_peak`'s graph plumbing.*

**S3 — Hierarchical pipeline bundle + the converter dies.** Hierarchical mode emits the retimed pipeline schedule (PP/DP sublayout, dp stamping) via `emit_chakra`; `retime` replaces `_apply_transformer_time`/`_assign_transformer_durations`; gmap/SCOTCH permutation moves into emission via `gmap_feed`. Gates: `train:hierarchical:*`, `:moe:`, `inf:hierarchical:*`. *Deletes: `convert_rapid_llm_graph_to_chakra_et`, `_RankTrace`, `_assign_collective_labels`, `_build_axis_groups`, `_compute_stage_axis_coords`, `_remap_stages_for_mapping`, `_extract_axis_layout`, module-level `_LAST_TP_GROUPS`/`_TP_*` globals (~1,300 LOC of `executor.py`; the file shrinks to the run/caching shell), `astrasim_lib/graph_debug.py`.*

**S4 — Evaluator owns analytical + hybrid pipeline; legacy graphs die.** `analytical` and hybrid's pipeline phase run `evaluate` on the GPipe schedule; grad-accum no-dp variant becomes a second program (`DpCommPolicy` nonfinal). Parity harness compares evaluator vs `Graph.simulate` totals on every spec before flipping. Gates: `train:analytical:*`, `train:hybrid:*`, `inf:analytical:*`, ga2 rows. *Deletes: `construct_fwd_bwd_graph` (~630), `construct_transformer_graph` (~350), `Graph.simulate`, `convert_comm_sizes_to_times`, `extract_forward_graph`/`extract_backward_graph` (~320), `Node`/`Edge`/`Data_batch` classes, `visualize_graph` (replaced by `sched/viz.py`) — effectively all of `simulate_train_graph.py`; `LLMExecutionDispatcher`'s graph fields and `_prepare_execution_graphs`'s graph construction (the function shrinks to program/microplan assembly, ~-500 LOC in `train_timing.py`).*

**S5 — Cleanup.** Delete `sched/compat.py`; `FaultSpace`/`gmap` consume `RankLayout` natively; delete `axis_layout_from_descriptor` shims where obsolete. Gates: full matrix.

**S6 — New capabilities (additive; new goldens only).** `pipeline_1f1b.py` behind config (`pipeline_interleave` with `native_interleave: true`), `id_policy="sid"` for schedules from new generators, flattened MoE enabled. New golden specs captured for these configurations with the commit that introduces them; **all 35 existing specs untouched.**

---

## 5. Equivalence risk register

Legend: **M** = must match exactly (gate-relevant), **V** = verified by dedicated parity test, **J** = legitimate divergence, justified, canonical-invisible.

| # | Surface | Risk | Handling |
|---|---------|------|----------|
| 1 | COMP durations (gate 1/3) | Rounding drift | **M**: `int(round(sec * 1e6))` identical; per-dp profile indexing identical; profile-length == dp_count assertion kept. |
| 2 | Collective sizes (gate 1) | Byte-count drift in expansion | **M**: keep quirks verbatim — per-rank pipeline split `ceil(total/par_degree)`; ZeRO-3 per-rank gather = `int(base_bytes)` **un-divided**; `CommSpec` divisible-`count` splits unchanged. |
| 3 | Control sends (gate 1) | Which transfers are "control" | **M**: size==0 **or** non-PIPELINE kind ⇒ 1-byte control, same naming suffixes (names canonical-invisible but kept). |
| 4 | Singleton groups (gates 1,5) | No-op emission | **M**: centralized rule in `et_emit.py`: group cardinality ≤ 1 ⇒ zero-duration COMP; unit test per axis, incl. the MoE + dp>1 ep-projection case. |
| 5 | dp==1 collective skipping (gate 1) | Extra dp collectives appearing | **M**: emitter rule `dp_count <= 1 and not tp-grouped ⇒ skip` preserved. |
| 6 | Dep sets / DAG shape (gate 2) | Extra or missing ctrl deps vs. legacy walkers | **M** + **V**: generators encode legacy attachment quirks explicitly (via-collective gating: a compute depends on the collective, never through it; dp side-branch attached only to rank 0's tail; ZeRO-3 cross-device `skip_*_children` wiring; flattener's compute-anchor double-parent). `compat.py` diffs Merkle hashes per spec per stage during development; goldens final arbiter. |
| 7 | Per-rank node-id order (gate 3 — ids are priorities) | Wall-time shifts from id shifts | **M**: `id_policy="legacy"`: `prio` sequence == legacy op_id creation sequence (generators constructed to reproduce it, incl. flattener counter starting at 1); Phase A per-stage min-prio Kahn ignoring cross-stage deps; Phase B legacy send/recv creation order (stage order → task order → dp asc → parents by prio); Phase C stable control-first renumber. `compat.py` compares full per-rank id sequences, not just hashes. |
| 8 | P2P tags | Tag values differ (one-step tags vs op_id tags) | **J**: canonical payloads exclude tags; AstraSim matching only needs consistency+uniqueness, which one-step-one-tag guarantees; wall time unaffected. Node names likewise **J** (kept similar anyway). |
| 9 | comm_groups.json / pg ids | Different group-id numbering | **J** for gates (canonical resolves ids→member sets; cache manifest excludes ids) but **M by choice**: reproduce label-sorted interning from 1000 to keep artifacts diffable. |
| 10 | AstraSim wall seconds (gate 3) | Binary nondeterminism | Non-risk: binary+configs untouched; id/dep/size/group-identical bundles ⇒ identical `sys[i] Wall time`. Any mismatch ⇒ items 1–7 regressed. |
| 11 | Analytical total (gate 4) | List-scheduler tie-order divergence | **M** + **V**: evaluator replicates `Graph.simulate` discipline (FIFO ready-scan in insertion order, heap `(finish, counter)`, edges slot-free); same-stage zero-duration PIPELINE edges kept as explicit steps so the event population and counter sequence match; successor lists stored in legacy child-creation order. S4 parity harness runs both on all 35 specs pre-deletion; target zero regens. |
| 12 | End-to-end composition (gate 4) | Interleave scale / grad-accum formula / retime application drift | **M**: `total = no_dp·(ga−1) + final`, `× _pipeline_interleave_scale` applied at the same points; retime writes the same scalars/tuples the legacy write-back produced (V: table vs legacy `comp_times` diff in S3). |
| 13 | dlsim completability (gate 5) | New emission deadlocks | By construction under `sid` policy (§3.5 proof); under `legacy` policy bundles are id-identical to goldens already proven completable, plus the per-group order postcondition check runs on every emission. |
| 14 | Memory peaks (reported outputs, not golden-gated) | Probe vs `simulate_memory` divergence | **V**: one-off byte-level parity test (per-GPU static/current/peak) across the matrix at S2. |
| 15 | gmap/SCOTCH permutation | Different float accumulation order → different rounded weights | **V**: feed in sid order (== legacy per-stage op order); CSR arrays compared against legacy collector on a synthetic optimize_2dmap config (none in the golden matrix, so no gate exposure). |
| 16 | Grad-accum nonfinal variant (gate 1/4, `ga2` specs) | Dp-comm gating drift | **M**: `DpCommPolicy` is a line-for-line port of `_should_emit_dp_comm` + the ep-sync `apply_ep_all_mbs` rule, with table-driven unit tests over (mode, zero_stage, cycle, mb). |
| 17 | Inference rank counts | 1-rank duplication / pruning | **M**: kept in the runner shell unchanged (`expected_rank_count==1 ⇒` duplicate ET, prune result). |
| 18 | S6 native 1F1B / flattened MoE / `sid` policy | Different totals and bundles | **J** by definition: new configurations, new goldens captured with the enabling commit; defaults unchanged. |

---

## 6. What we deliberately do NOT change, and why

- **The AstraSim side of the wall**: the binary, `config_generation.py`, `integration.py` (execution, `sys[i] Wall time` parsing, manifest-keyed caching), `et_utils.py` protobuf encoding, remote-memory path. The contract is reverse-engineered and experimentally pinned; touching the far side would invalidate the safety net that makes this rewrite checkable.
- **The verified workload contract handling as observable behavior**: 1-byte control messages, control-first renumbering, singleton no-ops, the ≥2-rank duplication hack, dp-major `rank = dp·num_stages + stage` mapping, per-stage dp group ids, tp group ids from 1000. These are load-bearing for gate 3; the rewrite changes *where* they are implemented (one emitter), not *what* they do.
- **All timing mathematics**: `train_timing`/`inference_timing` op timings, `CommSpec` derivation and `COMMUNICATION_RULES`, `NetworkModel.collective`, energy. The rewrite is about *structure and scheduling*, not performance modeling; conflating the two would make golden failures undiagnosable.
- **`MemoryEstimator.build_memory_data` byte accounting** — only the replay engine changes.
- **The rank-layout descriptor semantics** (canonical `tp,cp,ep,pp,dp` order, dimension validation, hierarchical leading-axes rule): it is the contracted bridge to `config_generation`, faults, and gmap (constraint 5). It gets a typed home (`RankLayout`) but identical meaning.
- **`fault_projection.py` math and `gmap.py`'s SCOTCH pipeline** — only their *feeding* moves onto the schedule.
- **The GPipe interleave closed-form correction** for existing configurations. Replacing approximation with native 1F1B changes results by design; that is an S6 opt-in with fresh goldens, never a silent swap.
- **MoE hybrid/hierarchical flow** (separate dense/MoE transformer schedules + mask-based retiming): it works and is golden-pinned; flattened MoE is *added* (S6), not refactored into.
- **The equivalence harness itself** (`equiv/`, golden format, spec matrix): the referee does not change while the game is being replayed. New specs are appended at S6 only.