# RAPID-LLM Rewrite Design: Typed Placed-Operation IR

Replacement design for the graph-construction → AstraSim pipeline. Verified against `docs/rewrite/CONTEXT.md`, `simulate_train_graph.py`, `llm_execution.py`, `astrasim_lib/executor.py`, `train_timing.py` (`_build_comm_metadata` / `_prepare_execution_graphs` / `calc_time_llm`), `memory_estimation.py`, `simulate_inference_graph.py`, and `equiv/`.

## 0. Stance and why it holds

**One typed Program IR of placed operations, constructed directly from an explicit schedule plus GEMM templates. All emitters and simulators are pure functions over it.**

The legacy failure modes are all consequences of one decision: the DAG is untyped and *under-placed* (a layer node carries a stage `hw_id`; per-rank placement, communicator membership, p2p pairing, and issue order are re-derived downstream). The flattener clone-and-mutates that DAG with name-string dispatch (`"bwd" in name`, `startswith("transformer_layer")`, ±`par_degree` hw offsets), and the ET converter then re-infers stages, groups, dependency classes, and a per-stage schedule it was never given — which is exactly where both deadlock bugs lived (divergent per-group collective orders from toposort tie-breaks; unpaired p2p tags for edge-less cross-stage deps).

The fix is to make every property the converter re-infers a *constructed* property:

- **Placement** is a coordinate in the rank layout, not an integer to be decoded.
- **Comm groups** are member-coordinate sets stored in a table, referenced by id.
- **A p2p transfer is one object** with both endpoints; SEND/RECV/tag are projections of it. Tags cannot diverge because there is nothing to draw twice.
- **The program order is a global total order** (a topological order fixed at build time). Every rank's ET is a projection of it, so per-group collective issue orders agree across member ranks *by construction* — the exact property the AstraSim contract demands.
- **Flattening is not a graph rewrite.** The fine program is built by enumerating the same schedule the coarse program uses and inlining the transformer block template per (mb, layer, direction, rank). Coarse and fine are two granularities of one builder family over one schedule — no clone caches, no sibling scans, no placeholder hacks. MoE flattening becomes "inline the MoE block template," which the IR supports natively (its joins/residual p2ps are ordinary ops with explicit deps), satisfying constraint 4.

The one honest cost of this stance: legacy per-stage ET node order is a Kahn order keyed by mutation-order `op_id`s, and AstraSim node ids are scheduling priorities. The new path derives order from the schedule instead. Section 6 treats every place where that can produce a legitimately different (never wrong) bundle, with an exact-match strategy first and a justified-regen fallback.

---

## 1. Core abstractions

All in `program/ir.py` and `program/layout.py`. Python 3.11, stdlib only, `slots=True` throughout.

```python
AxisName = str  # canonical axes, always in this order: "tp","cp","ep","pp","dp"
CANONICAL_AXES: tuple[AxisName, ...] = ("tp", "cp", "ep", "pp", "dp")

@dataclass(frozen=True, slots=True)
class RankLayout:
    """THE placement bridge (CONTEXT constraint 5). Single implementation,
    replacing the three copies in llm_execution / memory_estimation / flattener."""
    axis_order: tuple[AxisName, ...]          # subset of CANONICAL_AXES, canonical order
    axis_sizes: dict[AxisName, int]           # size >= 1 per axis in axis_order
    axis_strides: dict[AxisName, int]         # row-major over axis_order (tp fastest)

    def linearize(self, coord: "DeviceCoord") -> int: ...
    def coords_of(self, rank: int) -> "DeviceCoord": ...
    def subset(self, axes: Sequence[AxisName]) -> "RankLayout": ...   # e.g. ("tp","cp","ep") / ("pp","dp")
    def num_ranks(self) -> int: ...
    def descriptor(self) -> dict: ...   # legacy dict {"axis_order","axis_sizes","axis_strides","stage_span"}
                                        # kept for config_generation / faults / gmap consumers

DP_ALL: int = -1   # sentinel: op replicated across the dp axis (legacy "phantom DP")

@dataclass(frozen=True, slots=True)
class DeviceCoord:
    """A point in the parallelism grid. Missing axes are implicitly 0.
    dp may be DP_ALL for dp-replicated ops (see Placement)."""
    axes: tuple[tuple[AxisName, int], ...]    # sorted canonically; values >= 0 or DP_ALL for "dp"
    def get(self, axis: AxisName, default: int = 0) -> int: ...
    def replace(self, **kw: int) -> "DeviceCoord": ...

@dataclass(frozen=True, slots=True)
class Placement:
    coord: DeviceCoord
    @property
    def dp_replicated(self) -> bool:          # coord.get("dp") == DP_ALL
        ...
```

DP replication stays *symbolic* (`DP_ALL`) in the IR, matching the legacy phantom-DP emission (one logical op stamped onto every dp replica, `rank = dp_idx * num_stages + stage_idx`). This keeps fine programs ~dp× smaller and makes the hybrid per-dp duration profile (`duration_by_dp`) a natural field rather than dp-many cloned ops.

```python
class OpKind(Enum):
    COMPUTE    = "compute"     # occupies its device's compute slot
    COLLECTIVE = "collective"  # occupies the comm slot; references a CommGroup
    P2P        = "p2p"         # ONE object for a src->dst transfer (send+recv+tag)
    CONTROL    = "control"     # zero-cost join/root; never emitted to ET as-is

class OpRole(Enum):
    # compute
    EMBED = "embed"; HEAD = "head"; LAYER = "layer"; RECOMPUTE_LAYER = "recompute_layer"
    GEMM = "gemm"; OPTIMIZER = "optimizer"; MOE_JOIN = "moe_join"; NOOP = "noop"
    # collectives
    GEMM_COLL = "gemm_coll"          # tp/cp/ep collective attached to a GEMM template slot
    GRAD_REDUCE = "grad_reduce"      # dp all-reduce / reduce-scatter (embedding/layer/softmax)
    ZERO2_GATHER = "zero2_gather"; ZERO3_GATHER = "zero3_gather"; EP_SYNC = "ep_sync"
    MOE_A2A = "moe_a2a"
    # p2p
    XLAYER = "xlayer"                # cross-stage activation/grad transfer
    MOE_RESIDUAL = "moe_residual"
    CTRL_DEP = "ctrl_dep"            # zero-byte cross-stage ordering dependency

@dataclass(frozen=True, slots=True)
class OpKey:
    """Stable semantic identity. Total function of the model+schedule, independent of
    build order. Used for: retiming write-back, memory attribution, fault variants,
    golden diagnostics, cross-granularity mapping (fine ops -> parent coarse key)."""
    role: OpRole
    direction: str | None = None       # "forward" | "backward" | None
    micro_batch: int | None = None
    layer: int | None = None           # absolute layer index
    gemm: str | None = None            # template entry name ("qkv_proj", "MLP", ...)
    comm_name: str | None = None       # CommSpec/comm_metadata key for collectives
    part: int | None = None            # overlap-split part (0=head/block, 1=tail/ovlp), multi-part comms
    seq: int = 0                       # disambiguator for repeated (role,...) tuples

@dataclass(frozen=True, slots=True)
class CommGroup:
    group_id: int                          # program-unique; emitter maps to wire ids
    axes: tuple[AxisName, ...]             # ("tp",), ("dp",), ("tp","ep") composite, ...
    members: tuple[DeviceCoord, ...]       # sorted; dp component may be DP_ALL (per-dp instantiated at emit)
    interconnect: str                      # link-param key for analytical mode: "tp"|"cp"|"ep"|"dp"|"pp"

@dataclass(frozen=True, slots=True)
class CommDesc:
    kind: CollectiveType                   # ALL_REDUCE / ALL_GATHER / REDUCE_SCATTER / ALL_TO_ALL
    size_bytes: int
    group_id: int
    participants: int                      # kept for analytical NetworkModel + gmap weights

@dataclass(frozen=True, slots=True)
class P2PDesc:
    src: Placement
    dst: Placement
    size_bytes: int                        # 0 => control dependency (1-byte low-priority-band emission)
    interconnect: str = "pp"

@dataclass(frozen=True, slots=True)
class MemAttrs:
    mem_kind: MemKind | None
    fwd: bool
    recompute: bool = False
    param_gather: bool = False
    is_moe_layer: bool = False

@dataclass(slots=True)
class Op:
    op_id: int                             # dense index into Program.ops == global schedule priority
    key: OpKey
    kind: OpKind
    placement: Placement                   # executing device; for P2P this is p2p.src
    deps: tuple[int, ...]                  # predecessor op_ids; INVARIANT: all deps < op_id
    duration_s: float = 0.0                # COMPUTE/CONTROL only
    duration_by_dp: tuple[float, ...] | None = None   # hybrid/hier retime override, len == dp_count
    comm: CommDesc | None = None           # kind == COLLECTIVE
    p2p: P2PDesc | None = None             # kind == P2P
    mem: MemAttrs | None = None            # fine granularity compute ops

class Granularity(Enum):
    COARSE = "coarse"   # one COMPUTE per (mb, layer, dir) on a stage coord — analytical/hybrid/hier
    FINE   = "fine"     # per-rank GEMM chains — flattened ET, memory replay
    BLOCK  = "block"    # one transformer layer x (tp,cp,ep) ranks, one direction — hybrid/hier AstraSim

@dataclass(slots=True)
class ProgramMeta:
    granularity: Granularity
    dp_count: int                          # effective dp for emission (1 for block/inference)
    grad_accum_cycle: str                  # "final" | "nonfinal"
    run_type: str                          # "training" | "inference"
    optimize_2dmap: dict | None = None     # gmap hint (was root._optimize_2dmap)
    interleave: int = 1                    # closed-form correction stays outside the program

@dataclass(slots=True)
class Program:
    layout: RankLayout                     # the axes THIS program is emitted over
    ops: list[Op]                          # list order IS the global total order
    groups: dict[int, CommGroup]
    meta: ProgramMeta
    key_index: dict[OpKey, int]            # built once; retime/memory lookups
```

**Builder API** (`ProgramBuilder` in `ir.py`): `add(kind, key, placement, deps=(), **fields) -> int` appends and returns `op_id`; it can only reference already-added ops, so `deps < op_id` and acyclicity hold by construction. `intern_group(axes, members, interconnect) -> group_id` dedups groups. `validate(program)` (in `validate.py`) re-checks: dep bounds, coordinate ranges vs layout, every COLLECTIVE's placement ∈ its group members, p2p src ≠ dst placement (unless control same-rank elision), key uniqueness, and that per-rank projections contain each instantiated group's collectives in identical relative order (defense in depth; true by construction).

**Schedule** (`schedule.py`) — the single source of ordering truth:

```python
@dataclass(frozen=True, slots=True)
class ScheduleSpec:      # distilled from TimeCalculationLLM + comm_metadata (built in train_timing)
    mb: int; num_layers: int; layers_per_stage: tuple[int, ...]
    pp: int; dp: int; tp: int; cp: int; ep: int
    moe_layer_mask: tuple[bool, ...]
    zero_stage: int; dp_microbatch_mode: str; grad_accum_cycle: str
    include_backward: bool; include_optimizer: bool
    full_recomputation: bool; pipeline_style_recompute: bool
    comp_times: Mapping[str, float]                    # embedding_f/b, head f/b, layer f/b dense/moe, optimizer
    comm_metadata: Mapping[str, CommMeta]              # typed port of _build_comm_metadata output
    block_templates: Mapping[str, BlockTemplate]       # "dense" -> ..., "moe" -> ... (fine/block only)

@dataclass(frozen=True, slots=True)
class BlockTemplate:     # typed port of transformer_cfg["gemms"] + transformer comm_metadata
    entries: tuple[GemmEntry, ...]         # name, fwd/bwd durations, pre/post comm keys (placement-resolved)
    comm_metadata: Mapping[str, CommMeta]  # size, CollectiveType, participants, interconnect, parallel_group,
                                           # moe_component, moe_routing_mode, placement, tp_shard
```

The GPipe emission order (fwd mb-major with cross-mb stage dependencies, bwd reversed, DP/ZeRO lattice per `_should_emit_dp_comm`, EP sync, optimizer tails) is written once in `schedule.py` as an ordered event stream; both pipeline builders consume it, guaranteeing coarse and fine agree on structure by sharing code, not by equivalence testing.

---

## 2. Module layout

New package `program/` (all consumers migrate onto it; nothing outside it constructs execution structure):

| File | Responsibility | est. LOC |
|---|---|---|
| `program/ir.py` | dataclasses above, `ProgramBuilder`, key index | 300 |
| `program/layout.py` | `RankLayout`, derivation from `hw_config.network_layout` (port of `_build_rank_layout_descriptor` incl. hierarchical-mode leading-axes validation and `optimize_2dmap` extraction), subsets, legacy-descriptor adapter | 260 |
| `program/schedule.py` | `ScheduleSpec`, GPipe event stream, DP/ZeRO/EP emission policy (`_should_emit_dp_comm` port), grad-accum final/nonfinal | 300 |
| `program/block.py` | `BlockTemplate` construction from `_prepare_execution_graphs`'s CommSpec machinery output; `build_block_program(template, direction, layout_tcep)` incl. MoE hot/cold joins + residual p2ps | 380 |
| `program/pipeline.py` | `build_pipeline_program(spec, COARSE)` and `(spec, FINE)`; FINE inlines `block.py` chains per (mb, layer, dir, rank); ZeRO-3 per-rank gather placement; per-rank cross-layer p2ps; optimizer per rank | 480 |
| `program/transforms.py` | pure Program→Program passes: `apply_tp_overlap`, `apply_cp_overlap` (port of `_split_tp_node` / `_split_cp_edge` semantics on ops), `remap_devices(program, stage_perm)` for SCOTCH, `for_dp(program, dp_idx)` selector | 280 |
| `program/et_emit.py` | pure `emit_ets(program, out_dir) -> EtBundle` (per-rank Chakra ET, comm_groups.json, manifest.json); all contract rules live here | 420 |
| `program/analytic_sim.py` | discrete-event replay of `Graph.simulate` semantics over COARSE programs; `convert_comm_sizes_to_times` equivalent (NetworkModel per collective, link params by `interconnect`) | 220 |
| `program/memory_sim.py` | port of `Graph.simulate_memory` over FINE programs (same event loop core as `analytic_sim`, plus `_MemorySnapshot`); consumes `MemAttrs` | 280 |
| `program/astra_run.py` | thin wrapper: emit → `generate_astrasim_configs_from_hw` → `run_cache_astrasim` → per-rank seconds; 1-rank duplicate-and-prune; artifact persistence/cleanup (port of `run_astra_simulation_only_onepath` minus conversion) | 180 |
| `program/retime.py` | hybrid/hier measurement orchestration: run block programs (dense/moe × fwd/bwd × per-(dp,stage) fault variants), write `duration_by_dp` onto coarse LAYER ops via `OpKey` | 180 |
| `program/gmap_pass.py` | traffic collection over a Program (collectives by axis, cross-stage p2ps) feeding existing `astrasim_lib/gmap.py` collector/SCOTCH; returns stage permutation for `transforms.remap_devices` | 140 |
| `program/faults.py` | adapter: `RankLayout` ↔ `FaultSpace` projections; per-(dp,stage) transformer fault map (port from `LLMExecutionDispatcher._initialize_fault_mappings` et al.) | 120 |
| `program/dispatch.py` | `ExecutionDispatcher`: the four modes + inference, replacing `LLMExecutionDispatcher` orchestration | 320 |
| `program/validate.py` | invariant checker (debug/test only) | 160 |
| `program/viz.py` | graphviz over Programs (replaces `visualize_graph` uses) | 120 |

Total new code ≈ 4,100 LOC. Legacy deleted by the end (Section 4): ≈ 5,600 LOC (`simulate_train_graph.py` 2,300; flattener + overlap + dispatcher graph plumbing ≈ 1,400 of `llm_execution.py`; converter ≈ 1,750 of `executor.py`; duplicated layout code ≈ 150).

---

## 3. Consumers on the new core

### 3.1 ET emission — contract rules by construction

`emit_ets(program, out_dir)` is a single pass over `program.ops` (global order), maintaining one `RankTrace` per (stage-coord, dp) rank with **two id bands**: band C (control p2ps: `P2PDesc.size_bytes == 0`) and band R (everything else). Pass 1 walks program order and appends each op's per-rank records into the band lists; pass 2 assigns ids: band C first (`0..k-1`, program order), band R after (`k..`, program order), then resolves deps through the id map. This reproduces the legacy `_renumber_control_priority` outcome without a rewrite step.

Per-op emission:

- **COMPUTE** at `(coord, dp)` (or every dp if `DP_ALL`): `COMP` node, `duration_micros = int(round(dur * 1e6))`, `dur = duration_by_dp[dp]` if present else `duration_s`. `CONTROL` ops emit nothing; their deps are forwarded transitively (dep resolution follows CONTROL ops to their nearest emitted ancestors, precomputed once).
- **COLLECTIVE**: instantiate the group per dp (substitute `DP_ALL`→dp in each member coord, linearize via `program.layout`), intern member-rank-tuples → wire group id (dp-axis stage groups: `stage_index+1`, exactly legacy; other groups: counter from 1000 in first-instantiation order). Rules honored here, in one place each:
  - instantiated member count ≤ 1 → zero-duration `COMP` no-op (legacy singleton-ring rule);
  - dp-axis collective with `meta.dp_count == 1` → skipped entirely (legacy Step-10 skip);
  - `pg_name` attr set from the interned id; `comm_groups.json` written from the intern table plus the dp stage groups.
- **P2P**: from the *one* op, emit `SEND(dst_rank, tag)` on the src rank and `RECV(src_rank, tag)` on the dst rank, `tag = op.op_id`, size `max(1, size_bytes)`, control ops named `*_send_control`/`*_recv_control` and routed to band C. Same-placement p2p (possible after remap-free coarse builds) degenerates to a direct dep. The RECV carries no deps beyond none (posted early — legal per contract); the consumer op deps on the RECV id; the SEND deps on the producer op id.
- **1-rank programs**: `astra_run.py` duplicates the trace and prunes the fake result (unchanged policy).
- **Manifest**: identical schema and sort key as legacy (`_manifest_op_key`), so `run_cache_astrasim` signatures stay compatible.

**Deadlock-freedom argument.**
1. *Acyclicity*: `deps < op_id` by construction ⇒ program order is a topological order ⇒ each rank's ET dep graph (a projection plus matched send/recv pairs, deps always pointing at earlier ops) is acyclic.
2. *Collective stream consistency*: for any instantiated communicator group G and collectives C1, C2 ∈ G, every member rank's trace contains C1 and C2 in the order induced by `op_id` — projections of one total order cannot disagree. By the verified contract (matching is per-rank issue order within a group), every member pairs the same logical collective at the same stream position ⇒ the divergent-order silent deadlock is impossible. (Legacy had to *re-establish* this property with per-stage Kahn + op_id tie-breaks; here it is not re-established, it is inherited.)
3. *P2P pairing*: send and recv of a transfer are projections of one op with one tag ⇒ (src, dst, tag) always matches; SEND completes unconditionally; RECV never occupies a slot; so p2p can only wait on its producer dep chain, which is acyclic.
4. *Progress*: consider the unfinished op with the smallest global `op_id` whose deps are all finished (exists by acyclicity). If COMP/SEND: it needs only its slot; slots are held by ops that finish in finite time (COMP by duration, SEND unconditionally, COLL by induction below). If COLL: by (2) every member's earlier group collectives are already matched/finished, and each member's copy is the smallest-id ready op on its rank wanting the comm slot or is behind finite-time occupants, so all members issue it in finite time; native rings with ≥2 members terminate. If RECV: completes once the matched SEND issues, which is an earlier-or-independent op that progresses by the same argument. Hence no reachable stuck state. Priority-band placement of control sends prevents the (performance, and in legacy practice deadlock-adjacent) starvation of tiny control messages behind heavy collectives; it does not carry the correctness burden. Gate (5) — dlsim completability — remains the mechanical check on every bundle.

### 3.2 Analytical simulation

`analytic_sim.simulate(program_coarse, network_model, link_params) -> (total_s, finish_times_by_key)`. Two phases, mirroring legacy exactly: (a) timing conversion — each COLLECTIVE op's duration := `network_model.collective(kind, size_bytes, participants, ib, ll, ...)` with `(ib, ll) = link_params[comm.group.interconnect]`; p2ps use their interconnect the same way (legacy `cross_layer` edges carry sizes and get converted identically); (b) event loop — heap keyed `(finish_time, insertion_counter)`, per-device boolean compute exclusivity for COMPUTE ops, COLLECTIVE/P2P non-exclusive, insertion-order ready list. Ready-list insertion order derives from child iteration order; in the new IR, "children" of an op are the ops listing it as a dep, iterated in `op_id` order — which equals the legacy construction/child order because both follow the same schedule emission sequence. `_pipeline_interleave_scale` stays a closed-form multiplier in `dispatch.py` (unchanged, per CONTEXT).

### 3.3 Memory replay

`memory_sim.simulate(program_fine, memory_data, mode)` ports `Graph.simulate_memory` onto the same event loop: static per-device init from dense/moe layer counts per device (now counted from `Op.mem.mem_kind`/`key.layer`/placement instead of name sniffing), persistent alloc on fwd COMPUTE completion, transient alloc+release, persistent release on bwd, `param_gather` ephemeral alloc at issue / release at completion. `MemoryEstimator.build_memory_data` is unchanged; `MemoryEstimator.simulate_peak` calls `memory_sim` and its "is this flattened?" guard becomes `program.meta.granularity is FINE`. The fine program used for memory is the same object used for flattened execution when that mode runs (dispatcher caches it), else built on demand — replacing `build_flattened_root_for_memory`.

### 3.4 Hybrid / hierarchical retiming

`retime.measure_blocks(spec, faults) -> BlockTimings`: builds block programs (dense/moe × fwd/bwd) over `layout.subset(("tp","cp","ep"))`, applies overlap transforms, runs each through `astra_run` with `dp_count=1`, plus one run per `(dp_idx, stage)` fault variant from `faults.transformer_stage_dp_faults`. `retime.apply(program_coarse, timings, dp_count)`: for every op with `key.role in {LAYER, RECOMPUTE_LAYER}`, set `duration_by_dp = tuple(variant_or_baseline(dp_idx, stage(op), moe(op), direction(op)))` — the typed replacement of `_assign_transformer_durations`' name-prefix walk. Hybrid then runs `analytic_sim` on the retimed coarse program; hierarchical emits the retimed coarse program over `layout.subset(("pp","dp"))` via `et_emit` and runs AstraSim (dp collectives get stage groups; cross-stage p2ps are real ops; `duration_by_dp` selects per-dp COMP durations — identical semantics to today's `duration_profile`).

### 3.5 Inference (prefill and sampled decode)

Prefill: forward-only `ScheduleSpec` (`include_backward=False`, no DP lattice) → same four modes; `run_type == "inference"` forces `meta.dp_count = 1` in `dispatch.py` (legacy `dp_override=1`). Decode: `DecodeGraph._execute_decode_step` already re-enters the standard dispatcher per sampled step with per-step GEMM shapes; it changes only its call target (`program.dispatch.ExecutionDispatcher`). Sample-point generation and trapezoid integration are untouched. Decode memory replay uses the FINE program of the sampled step with `kv_cache_tokens` in `memory_data`, as today.

### 3.6 Fault projection and gmap/SCOTCH remap

`faults.py` keeps `FaultSpace`/`FaultProjectionResult` intact (they already operate on the layout descriptor); the dispatcher-side plumbing (global/transformer/pipeline projections, coverage validation, per-(dp,stage) map, analytical/hybrid rejection) moves in as pure functions of `RankLayout` + `faulty_links`. gmap: `gmap_pass.collect(program)` records, in program order, every collective (axis, stage, bytes, participants, double-weight for AR/A2A) and every cross-stage p2p with bytes — replacing the converter's in-line collection — then `gmap.finalize_collection` runs SCOTCH as today, and the resulting stage permutation is applied as `transforms.remap_devices(program, perm)` *before* emission: a pure coordinate substitution on placements and group members (legacy achieves the same by rebinding traces/ranks mid-conversion). Emitter stays remap-unaware, as the legacy comment aspired to.

### 3.7 Overlap transforms

`transforms.apply_tp_overlap(program, mode, frac)`: for each COMPUTE op with a dependent same-key-scope tp COLLECTIVE (identified by `key.role == GEMM_COLL and comm.group.interconnect == "tp"`, not name matching), split the compute op into `part=0` (head, `(1-f)·d`) and `part=1` (tail, `f·d`), rewiring deps exactly as `_split_tp_node` (f ≥ 1 hoists the collective before the op). `apply_cp_overlap`: split cp collectives feeding attention GEMMs (`key.gemm == "attention"`, no string sniffing) into block/ovlp parts with ceil byte split, wiring per `_split_cp_edge`. Both run on BLOCK programs (hybrid/hier) and FINE programs (flattened) — same code, satisfying constraint 6. Splits produce new `OpKey`s differing only in `part`, keeping retime/memory lookups stable.

### 3.8 Rank layout derivation

`layout.derive(hw_config, tc, execution_mode) -> RankLayout` ports `_build_rank_layout_descriptor` verbatim (axis discovery from `network_layout.dimensions`, canonical reordering, size validation, hierarchical/hybrid leading-cluster-axes enforcement, active-axis checks, `optimize_2dmap` config). `config_generation.py`, `derive_axes_filter`, and fault tooling keep consuming `RankLayout.descriptor()` unchanged.

---

## 4. Migration plan

Every stage ends with the full 35-spec golden matrix green (`pytest tests/test_equiv_golden.py`, ~15 s / 8 workers, run with `./.venv/bin/python` and the documented `LD_LIBRARY_PATH`). Stages are individually landable commits. "Delete" means removed in that stage's commit, not deprecated.

**Stage 0 — scaffolding (no behavior change).** Land `ir.py`, `layout.py`, `validate.py` + unit tests. Nothing wired. Gates trivially green. *Deleted: nothing.*

**Stage 1 — layout unification.** `llm_execution._build_rank_layout_descriptor`, `memory_estimation._build_rank_layout`/`_hw_id_for_rank`, and `PipelineGraphFlattener._configure_rank_layout`/`_hw_id_for_rank` become calls into `RankLayout` (flattener keeps its interface via `layout.descriptor()`). Pure refactor; identical numerics. *Deleted: the three duplicated layout/linearization implementations (~150 LOC).*

**Stage 2 — block programs power hybrid/hierarchical transformer runs.** Land `schedule.py` (BlockTemplate part), `block.py`, `et_emit.py`, `astra_run.py`, `transforms.py` (overlap only), `retime.measure_blocks`. `_prepare_execution_graphs` builds `BlockTemplate`s from its existing CommSpec machinery *in addition to* legacy structures; `LLMExecutionDispatcher._run_transformer_astrasim` switches to `retime.measure_blocks`. Affected goldens: every `hier/*` bundle (fwd/bwd, moe, fault variants) — op multisets, DAG hashes, exact wall seconds, dlsim. Block chains are per-rank serial, so the DAG is forced and wall equality follows from equal ops+deps+groups (Section 6, R1). *Deleted: `construct_transformer_graph` (~350 LOC), `extract_forward_graph`/`extract_backward_graph` (~310 LOC), transformer-root plumbing in dispatcher/train_timing; transformer `Graph` objects survive only as template carriers for the still-legacy flattener (shim attribute set).*

**Stage 3 — fine programs: memory first, then flattened ET.**
*3a*: `pipeline.py` FINE + `memory_sim.py`; `build_flattened_root_for_memory` → FINE program; `simulate_peak` → `memory_sim`. Memory output is not hash-gated (gates pin times/bundles), but end-to-end results files must not change; verified by gate (4).
*3b*: `FULL_ASTRASIM_FLATTENED` emits the FINE program via `et_emit` + `astra_run` (with `gmap_pass` if `optimize_2dmap`). Affected goldens: all `flat` bundles + their exact per-rank seconds. This is the highest-risk stage; land behind `RAPID_NEW_CORE=1` with a differential runner comparing canonical bundles legacy-vs-new on all 35 specs, flip the default in the same PR only when clean (see R2–R5 for the permitted-divergence policy). *Deleted (at 3b): `PipelineGraphFlattener` (~670 LOC), `_propagate_local_hw_ids`, legacy `apply_overlap_transforms` + `_split_tp_node`/`_split_cp_edge` (~300 LOC, moved), memory-side flatten shims.*

**Stage 4 — coarse programs: analytical + hybrid pipeline phase.** `pipeline.py` COARSE + `analytic_sim.py` + `retime.apply`; `ANALYTICAL` and `HYBRID` modes run fully on the new core (including grad-accum final/nonfinal program pairs and the no-dp variant). Affected goldens: analytical/hybrid rows' end-to-end times (exact). *Deleted: `Graph.simulate`, `convert_comm_sizes_to_times`, `_apply_transformer_time`/`_assign_transformer_durations`, `duration_profile` machinery on `Node`.*

**Stage 5 — hierarchical pipeline phase; converter dies.** Retimed COARSE program over `("pp","dp")` → `et_emit` → `astra_run`; `gmap_pass` + `faults.py` complete. Affected goldens: `hier` pipeline bundles + wall seconds. *Deleted: `convert_rapid_llm_graph_to_chakra_et` and all converter helpers (`_RankTrace`, `_assign_collective_labels`, `_build_axis_groups`, `_compute_stage_axis_coords`, `_remap_stages_for_mapping`, `_write_comm_groups_json`, pipeline send/recv caches — ~1,750 LOC of `executor.py`; the file shrinks to the AstraSim run wrapper, itself superseded by `astra_run.py` and deleted), `construct_fwd_bwd_graph` (~630 LOC), `Graph`/`Node`/`Edge`/`Data_batch` and thus **`simulate_train_graph.py` entirely** (visualization moves to `program/viz.py`), `LLMExecutionDispatcher` (replaced by `program/dispatch.py`).*

**Stage 6 — inference decode + final sweep.** `DecodeGraph` drops its `Graph` inheritance (it only used the class as a config carrier) and calls the new dispatcher; `inference_timing.prepare_decode_graphs` returns `ScheduleSpec`+templates. Remove `flattened_mode` misc flags, pickle debug entrypoints, dead env plumbing. Affected goldens: inference specs (prefill/decode phase times, bundles). *Deleted: decode-side graph plumbing, remaining shims (~200 LOC).*

Golden regeneration is permitted only under the R2–R5 justifications below, committed together with the diff and a one-paragraph rationale per spec, per the CONTEXT policy.

---

## 5. Equivalence risk register

Legend: **match** = must be bit/hash-identical, engineered to be; **regen-if-justified** = a legitimate difference may appear; policy stated.

| # | Surface | Risk | Handling |
|---|---|---|---|
| R1 | Block-program bundles (hier fwd/bwd) | none structural: per-rank chains are serial, ops/deps/groups forced by template | **match** — Stage 2 gate; wall seconds follow from identical bundles + same binary/configs |
| R2 | Flattened per-rank **op multisets** | quirks must be reproduced: zero-dur COMP no-ops for singleton groups; 1-byte control sends/recvs; dp-collective skip at dp=1; `local_comp_time` forced to 0; optimizer expanded per rank; recompute chains; `param_gather` on first GEMM | **match** — each quirk is a single named rule in `et_emit`/`block.py`; differential runner diffs op_counts per rank |
| R3 | Flattened **DAG shape** | legacy oddities are pinned by dag_hash: DP collectives attach to `rank_tails[0]` only (asymmetric); per-rank pipeline edges also anchored to the nearest compute ancestor; ZeRO-3 cross-device gathers placed on the *next/prev stage's* ranks via the ±par_degree offset hack; `attach_parallel_edge` parent/child skip rules for ZeRO-3 lattice | **match** — FINE builder encodes each as an explicit, commented rule (`dp_attach_rank0=True`; gather placed at `coord.replace(pp=target_stage)`, which equals the ±par_degree offset because canonical order fixes pp stride = tp·cp·ep; parallel-edge wiring ported as spec'd functions). If any rule is later judged a modeling bug, fix + regen is a **separate, justified** commit, never bundled with migration |
| R4 | **Node-id order** within a rank (AstraSim priority) | legacy: per-stage Kahn keyed by flattener-mutation `op_id`, then control-band renumber, then Step-11 recv appends in parent-op_id order. New: program order + two bands. Both are topological orders of the same DAG but can interleave independent ops differently, which can change AstraSim wall times via lowest-id-first issue | dominant case **match**: builder emits in schedule order and legacy flattener op_ids were assigned in the same schedule-shaped traversal, so orders coincide on all current specs — verified empirically by the Stage-3 differential runner (canonical hashes are id-independent; wall seconds catch scheduling drift). Where a spec disagrees: first try `et_emit` ordering hook (sort band R by a legacy-compatible key); if a residual diff is a pure tie-break between independent ops with equal wall time, accept (gates pass); if wall time shifts, **regen-if-justified** with the dlsim trace attached showing both schedules valid |
| R5 | Exact AstraSim **per-rank wall seconds** | consequence of R2–R4 plus comm_groups/system config identity | **match** given R2–R4 match and unchanged `config_generation`/binary; per-rank seconds compared at full precision by the runner |
| R6 | `comm_groups.json` | ids are free (canonicalizer resolves to member sets) but dp stage groups must exist iff dp>1 and cover dp-major ranks | **match** by emitter policy (same id scheme retained anyway to keep artifacts diffable) |
| R7 | Analytical / hybrid **end-to-end times** | event-loop equality: heap tie order `(finish, counter)`, ready-list insertion order, per-device flags; float sum order of collective durations | **match** — `analytic_sim` replays the identical loop; children iterated in op_id order = legacy construction order; NetworkModel called once per comm op with identical args in identical order. Property test: legacy vs new finish-time maps on captured graphs during Stages 4 (before deletion) |
| R8 | Hybrid retime write-back | per-dp tuple order, fault-variant selection (`(dp_idx, stage)`), moe/dense split, fwd/bwd flag | **match** — `retime.apply` ports `_per_dp_durations` logic keyed by `OpKey` instead of name prefix; unit test cross-checks on a captured legacy graph |
| R9 | `duration_micros` rounding | `int(round(sec*1e6))`, clamp ≥ 0 | **match** — single helper in `et_emit` |
| R10 | Grad accumulation | two programs (final/nonfinal) with different comp_times (`linear_softmax_b`/`embedding_b` overrides) and DP-emission policy; total = (n−1)·nonfinal + final | **match** — `ScheduleSpec.grad_accum_cycle` port; dispatcher formula unchanged |
| R11 | MoE hybrid/hier | dense/moe block templates, hot/cold join + residual p2p topology, `_moe_parallel_token` grouping, EP-sync specs on `layernorm1` backward | **match** — `block.py` ports the exact rules; MoE-flattened remains rejected until a deliberate post-migration feature commit (then new goldens, not regens) |
| R12 | gmap/SCOTCH permutation | permutation depends on accumulated float pair-weights; accumulation order changes could flip SCOTCH ties | **match** — `gmap_pass` records in program order, which mirrors legacy object-discovery order on current specs; verified by comparing emitted `.grf` files byte-wise in the differential runner; divergence ⇒ **regen-if-justified** (mapping quality metrics attached) |
| R13 | Fault projection | link remap identical per axis subset; single-stage constraint errors preserved verbatim | **match** — `FaultSpace` untouched; adapter tested against captured projections |
| R14 | Inference | dp forced to 1 (duplicate-and-prune), decode sample points/integration, prefill phase times | **match** — policy unchanged, only call targets move |
| R15 | Memory replay | layer-per-device counting (now from keys, not names), zero-layer fallback (`dense=1`), moe/dense byte maps, param-gather ephemeral timing at issue vs completion | **match** — ported rules + smoke parity vs `validation_scripts/memory_estimator_smoke.py`; not hash-gated but protected by gate (4) end-to-end outputs |

---

## 6. Deliberately unchanged, and why

- **The AstraSim side of the fence**: binary, `integration.py` invocation/caching/`sys[i] Wall time` parsing, `config_generation.py`, `et_utils.py` protobuf helpers, remote-memory path. The rewrite targets construction/emission; touching execution plumbing would multiply the equivalence surface for zero design benefit.
- **The rank-layout contract** (canonical `tp,cp,ep,pp,dp`, dp-major ET rank mapping, descriptor dict shape): it is the bridge to `config_generation`, faults, and gmap (CONTEXT constraint 5). We unify its *implementation*, not its semantics.
- **Timing and sizing models**: all of `train_timing`'s compute/comm byte math, `COMMUNICATION_RULES`/CommSpec generation, `NetworkModel`, `MemoryEstimator.build_memory_data`. The IR consumes their outputs; changing them would confound modeling changes with structural migration.
- **The GPipe schedule + interleave closed form**: interleaving stays a post-hoc scale (`_pipeline_interleave_scale`) exactly because it is not in any legacy graph; putting it into the schedule is a modeling change requiring its own validation and golden regen, out of scope (the schedule module leaves an explicit seam for it).
- **Legacy modeling quirks pinned by goldens** (R2/R3 list: rank-0 DP attach, singleton no-ops, `local_comp_time` skip, 1-byte controls, dp-collective skip). Reproduced verbatim; each is one flagged rule so future deliberate fixes are one-line diffs plus a justified regen.
- **The equivalence harness itself** (`equiv/`, dlsim, golden format, 35-spec matrix): it is the referee; the referee does not migrate. dlsim in particular stays the executable spec of the AstraSim workload contract.
- **Execution-mode surface and CLI/config semantics**: four modes + inference + memory estimation, same YAML inputs, same results files — external users see nothing (constraint 1).
- **MoE-flattened stays unsupported at migration end**: the design makes it a bounded follow-up (inline the MoE block template in the FINE builder; the IR already expresses its joins), but enabling it is new behavior with new goldens, never smuggled into the rewrite.