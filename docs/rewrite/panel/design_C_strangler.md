# Design: Replacing RAPID-LLM graph construction → AstraSim execution
## Incremental strangler with hard seams

Everything below assumes the context pack (`docs/rewrite/CONTEXT.md`) and was written against the code as of commit 85894c6 on this branch. File references are relative to `/app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM`.

---

## 0. Strategy and its honest cost

**Stance: incremental strangler.** The legacy untyped `Node`/`Edge`/`Data_batch` graph stays the *interface* between construction and consumers for most of the migration. We replace internals one seam at a time, and each seam is a hard boundary with a typed contract:

1. **Seam 1 — the converter.** Introduce the typed IR (`Program`) and a *lowering pass* that extracts it from the legacy graph, quarantining all re-inference (stage attribution, dependency walkers, per-stage toposort, communicator reconstruction) into one deletable module. A clean, ~300-line ET emitter runs over the `Program`. The AstraSim contract is honored by emitter construction, not by scattered conventions.
2. **Seam 2 — the flattener.** Replace `PipelineGraphFlattener` with a direct builder that still produces legacy `Node`/`Edge` objects, but with *explicit* metadata (placement, direction, p2p identity, ZeRO-3 shard role) instead of name-string dispatch, `"bwd" in name` hacks, sibling scans, and hw-id offset arithmetic.
3. **Seams 3–4 — the graph constructors** (`construct_transformer_graph`, then `construct_fwd_bwd_graph`), same pattern.
4. **Last — retire the legacy types.** Builders emit `Program` natively; analytical simulation and memory replay are ported onto it; `simulate_train_graph.py`'s classes are deleted.

Every stage lands independently with all 35 golden gates green.

**Where this is slower/dirtier than a fresh IR — stated up front:**

- The lowering pass (Seam 1) is *new code that re-implements the ugliest legacy logic* (the two dependency walkers, ZeRO-3 edge attribution). It exists only to be deleted at the end. A fresh IR would never write it. We accept this because it is the only way to cut the emitter over while the legacy constructors are still the source, and because it is the module where equivalence bugs will surface — better there than in the emitter.
- The end-state analytical scheduler and memory replay must *emulate legacy quirks* (ready-list scan order, children-append order, heap tie counters) to keep end-to-end golden times bit-identical. A fresh IR could define cleaner scheduling semantics; we would then have to regen every analytical golden with only "trust me" as justification.
- The `Program` inherits some contract-shaped warts permanently: DP phantom replication as an emission-time parameter, the control-send renumbering pass, the 1-rank ET duplication hack. These are AstraSim-contract facts, not incidental legacy — a fresh IR would carry them too, but a fresh IR could at least hide them better behind the emitter. We keep them visible in emitter code with contract citations.
- Total migration LOC churn is higher than a big-bang (~2,900 new + ~450 transitional lowering/adapters that get deleted), but no single landing exceeds ~500 LOC of behavior-relevant change, and no landing ever leaves the goldens red.

---

## 1. Core abstractions

New package `workload/` (name chosen to avoid clashing with `astrasim_lib`). All types Python 3.11 dataclasses, no new dependencies. `CollectiveType` and `MemKind` are reused from `timing_model.py` / `memory_estimation.py` unchanged.

### 1.1 `workload/rank_layout.py`

Formalizes the descriptor dict that today floats around as `{"axis_order": ..., "axis_sizes": ..., "axis_strides": ..., "stage_span": ...}`.

```python
AXIS_CANONICAL_ORDER: Tuple[str, ...] = ("tp", "cp", "ep", "pp", "dp")

@dataclass(frozen=True)
class RankLayout:
    axis_order: Tuple[str, ...]          # subset of AXIS_CANONICAL_ORDER, in canonical order
    axis_sizes: Mapping[str, int]        # size >= 1 for every axis in axis_order
    axis_strides: Mapping[str, int]      # derived: span-products in axis_order

    # --- derived helpers (all pure) ---
    def span(self) -> int                                    # product of sizes
    def linear(self, coords: Mapping[str, int]) -> int       # coord -> flat id
    def coords(self, flat_id: int) -> Dict[str, int]         # flat id -> coord
    def subset(self, axes: Sequence[str]) -> "RankLayout"    # e.g. transformer view (tp,cp,ep)
    def axis_groups(self, axis: str) -> List[Tuple[int, ...]] # member flat-ids per group of `axis`
    def descriptor(self) -> Dict[str, Any]                   # legacy-dict compatibility view

@dataclass(frozen=True)
class LayoutBundle:
    full: RankLayout                       # tp,cp,ep,pp,dp
    transformer: Optional[RankLayout]      # subset ["tp","cp","ep"]
    pipeline: Optional[RankLayout]         # subset ["pp","dp"]
    optimize_2dmap: Optional[Dict[str, Any]]   # first-dim SCOTCH config, verbatim semantics
    axis_to_dimension: Mapping[str, int]       # for FaultSpace

def derive_layout_bundle(hw_network_layout, axis_sizes: Mapping[str, int],
                         execution_mode: ExecutionMode) -> LayoutBundle: ...
```

`derive_layout_bundle` is a verbatim extraction of `LLMExecutionDispatcher._build_rank_layout_descriptor` + `_axis_to_dimension_map` (including the hierarchical/hybrid leading-cluster-axes validation and all its error messages). `descriptor()` keeps `astrasim_lib.layout_utils.axis_layout_from_descriptor`, `fault_projection.FaultSpace`, `config_generation`, and `gmap` working without touching them.

### 1.2 `workload/ir.py` — the Program IR

```python
OpUid = int          # dense 0..N-1; uid IS the global total order AND topological order
DeviceId = int       # flattened hardware coordinate ("stage" in converter terms)

class OpRole(Enum):
    GENERIC = auto()
    TRANSFORMER_LAYER = auto()     # pipeline-graph layer node (retiming target)
    EMBEDDING = auto()
    SOFTMAX = auto()
    OPTIMIZER = auto()
    GEMM = auto()                  # flattened/transformer per-rank compute
    JOIN = auto()                  # zero-duration MoE join / structural node

class Direction(Enum):
    FORWARD = auto()
    BACKWARD = auto()

@dataclass(frozen=True)
class GroupKey:
    axis: str                       # "tp" | "cp" | "ep" | "dp" | "tp+ep" (composite)
    members: Tuple[DeviceId, ...]   # sorted device ids (pre-DP; DP stamped at emission)

@dataclass
class CommGroup:
    key: GroupKey
    label: str                      # human label; feeds pg naming, never matching

@dataclass
class ComputeOp:
    uid: OpUid
    name: str                       # diagnostics only; never dispatched on
    device: DeviceId
    duration: Tuple[float, ...]     # length 1, or length dp_count (per-DP profile)
    deps: Tuple[OpUid, ...]         # all < uid (validated)
    role: OpRole = OpRole.GENERIC
    direction: Direction = Direction.FORWARD
    mem_kind: Optional[MemKind] = None
    recompute: bool = False
    param_gather: bool = False
    micro_batch: Optional[int] = None
    layer: Optional[int] = None
    is_moe_layer: bool = False

@dataclass
class CollectiveOp:
    uid: OpUid
    name: str
    device: DeviceId                # owning device (legacy "edge stage")
    group: GroupKey
    coll: CollectiveType            # never PIPELINE
    size_bytes: int
    participants: int
    interconnect: str               # "tp"|"cp"|"ep"|"dp" — drives gmap axis + skip rules
    is_dp: bool                     # legacy dp-collective flag (emission skip when dp==1)
    deps: Tuple[OpUid, ...]

@dataclass
class TransferOp:                   # one logical p2p transfer: BOTH endpoints, ONE identity
    uid: OpUid                      # uid doubles as the p2p tag
    name: str
    src_device: DeviceId
    dst_device: DeviceId
    size_bytes: int                 # 0 => control (emitted as 1-byte send/recv, control class)
    producer: OpUid                 # op whose completion triggers the send
    deps: Tuple[OpUid, ...]         # == (producer,) normally

Op = Union[ComputeOp, CollectiveOp, TransferOp]

@dataclass
class ProgramMeta:
    label: str                                  # "flat", "hier/fwd", ...
    optimize_2dmap: Optional[Dict[str, Any]]    # replaces root._optimize_2dmap attr
    misc: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Program:
    layout: RankLayout              # the layout this Program's device ids live in
                                    # (transformer subset, pipeline subset, or full)
    dp_count: int                   # DP phantom-replication degree at emission
    devices: Tuple[DeviceId, ...]   # sorted; includes devices seen only via collectives
    ops: List[Op]                   # ops[uid].uid == uid, dependency-closed
    groups: Dict[GroupKey, CommGroup]
    meta: ProgramMeta

    def validate(self) -> None      # invariants below; raises ProgramInvariantError
```

**Validated invariants** (`Program.validate()`, run at every construction and before every emission):

- V1: `ops[i].uid == i`; every dep uid < op uid (DAG + topological order by construction).
- V2: every `CollectiveOp.group` is registered and `device ∈ group.members ⊆ devices`.
- V3: every `TransferOp` has `src_device != dst_device`, both in `devices`, `producer` on `src_device`.
- V4: every `ComputeOp.duration` has length 1 or `dp_count` (mirrors the converter's `duration_profile` check).
- V5 (static group-order consistency): for each group, the subsequence of that group's collectives in uid order is, per member device, exactly the collectives that the emitter will append there — trivially true by construction; asserted anyway.
- V6 (group-race warning, non-fatal): for each group and each pair of consecutive same-group collectives, warn if no dependency path connects them on any member device. This statically flags the class of bug fixed in 85894c6; the dynamic guarantee remains dlsim-gated (see §3.1).

### 1.3 `workload/emitted.py`

```python
@dataclass
class EmittedBundle:
    et_prefix: str
    rank_ids: List[int]
    manifest_path: str
    comm_groups: Dict[str, List[int]]      # gid -> global rank members (replaces module globals)
```

### 1.4 Builder helper

```python
class ProgramBuilder:
    """Append-only builder. add_* returns the OpUid; uids are the creation sequence.
    This is the single source of scheduling priority: creation order == uid order
    == per-rank ET id order (within priority class)."""
    def __init__(self, layout, dp_count, meta): ...
    def add_compute(self, **fields) -> OpUid: ...
    def add_collective(self, **fields) -> OpUid: ...
    def add_transfer(self, **fields) -> OpUid: ...
    def group(self, axis: str, members: Sequence[DeviceId], label: str) -> GroupKey: ...
    def finish(self) -> Program:   # sorts devices, validates, freezes
```

---

## 2. Module layout

```
workload/
  __init__.py            exports Program, ProgramBuilder, RankLayout, emit_chakra          (~30)
  rank_layout.py         RankLayout, LayoutBundle, derive_layout_bundle                    (~220)
  ir.py                  ops, Program, ProgramBuilder, validator                           (~380)
  legacy_lowering.py     legacy Node/Edge graph -> Program  [TRANSITIONAL, deleted S7]     (~500)
  et_emission.py         Program -> Chakra ET bundle + manifest + comm groups              (~330)
  placement_opt.py       gmap/SCOTCH traffic collection + stage permutation as a
                         Program -> Program transform                                      (~160)
  flatten_build.py       direct flattened-graph builder (S3: legacy types; S6: Program)    (~450)
  transformer_build.py   per-(tp,cp,ep)-rank GEMM-chain builder incl. MoE joins            (~300)
  pipeline_build.py      GPipe pipeline builder incl. ZeRO-2/3 lattice, grad-accum,
                         optimizer, recompute                                              (~500)
  overlap.py             tp/tp_sp/cp overlap expansion applied inside builders (S6)        (~160)
  retime.py              hybrid/hierarchical duration write-back onto Programs             (~90)
  analytical_sim.py      legacy-faithful list scheduler over Program (S6)                  (~220)
  memory_replay.py       legacy-faithful peak-memory replay over Program (S6)              (~280)
  adapters.py            program_to_legacy_graph / attach-to-root shims [TRANSITIONAL]     (~150)
tests/
  test_program_parity.py differential: legacy path vs new path per seam, on the
                         golden matrix configs (uses equiv.canonical, no AstraSim exec)    (~250)
  test_ir_invariants.py  unit tests for validator, layout, builder                         (~200)
```

Responsibility boundaries:

- `et_emission.py` is the **only** writer of `.et` files, `manifest.json`, `comm_groups.json`. It knows every AstraSim contract rule and cites each one in a comment block. It never inspects names for semantics.
- `legacy_lowering.py` is the **only** module allowed to re-infer anything (stages from `hw_id`/`local_hw_id`, deps via graph walks, groups from labels+coords). Everything downstream of it consumes declared facts only. Constraint 3 ("one placement source of truth") is met at end state by deleting this module; during migration it is the one fenced-off exception.
- `astrasim_lib/{integration,config_generation,et_utils,gmap,fault_projection,layout_utils}.py` are consumed, not modified (except `executor.py`, which shrinks to a wrapper and then moves).

---

## 3. Consumers on the new core

### 3.1 ET emission — contract rules honored by construction

`emit_chakra(program: Program, output_dir: str) -> EmittedBundle` runs four passes. DP-major rank mapping is preserved exactly: `num_stages = len(program.devices)`, `rank(d, dp) = dp * num_stages + index(d)`.

**Pass 1 (main ops, uid order).** For each op in `program.ops`:
- `ComputeOp` on device `d`: for each `dp_idx`, append `COMP` to `rank(d, dp_idx)` with `duration_micros = int(round(sec * 1e6))`, per-DP profile indexed when present (V4 guarantees length).
- `CollectiveOp`: if `dp_count <= 1 and op.is_dp`: skip (legacy Step-10 skip of stage collectives). Else for each `dp_idx`: resolve `(group, dp_idx) -> gid` (gids allocated from 1000 in sorted-label order, member-set-deduplicated — verbatim legacy allocation); if `len(members(gid)) <= 1` append a zero-duration `COMP` no-op (contract: single-member native ring collectives deadlock); else append `COMM_COLL` with `pg_name=gid`. DP grad-sync collectives without an axis group get the stage DP group `str(stage_index + 1)` when `dp_count > 1`, exactly as today.
- Dep resolution is one uniform dict `et_id[(uid, rank)] -> local_id`; first-seen-order dedup as today.

**Pass 2 (p2p materialization, uid order of the TransferOp).** For each `TransferOp`, for each `dp_idx`: append `SEND(size or 1, dst_rank, tag=uid)` to the src rank (ctrl-dep on the producer's ET id), append `RECV(size or 1, src_rank, tag=uid)` to the dst rank; wire the RECV's ET id into every consumer op's ctrl_deps. `size==0` transfers get `_send_control`/`_recv_control` names. TransferOp uids are assigned by lowering/builders at the *consumer's* position in creation order, reproducing legacy Step-11 append order (sends appear on the src trace in the order destination tasks requested them).

**Pass 3 (priority classes).** Per rank, stable-partition nodes into (control sends+recvs) then (everything else), remap ids and ctrl_deps — the exact `_RankTrace._renumber_control_priority` algorithm, kept because node ids are scheduling priorities and golden wall seconds depend on it.

**Pass 4 (write).** Encode ETs; build the manifest with the existing `_manifest_op_key` sort (cache-key stability); write `comm_groups.json` from `EmittedBundle.comm_groups` (module globals `_LAST_TP_GROUPS` etc. die here). The 1-rank case duplicates the ET and the caller prunes, as today.

**Why deadlock-free by construction** (contract rules from CONTEXT.md, verified against `Workload.cc`/`HardwareResource.cc` behavior):

- *Collective matching (per-rank issue order within a group).* A `CollectiveOp` is appended to **all** member ranks during one uid step of Pass 1, and Pass 3 never reorders `COMM_COLL` nodes relative to each other. Hence for every group, every member rank's static order of that group's collectives is the same projection of the single uid order (V5). Because ids are priorities, ties among simultaneously-ready group collectives resolve identically on every member. The residual dynamic risk — two same-group collectives with *no ordering path* on some member becoming ready in different orders due to timing — is exactly the legacy 85894c6 bug class; the new design (a) detects it statically (V6 warning), (b) keeps the dlsim completability gate as the executable check, and (c) offers `serialize_groups(program)` (adds the missing chain dep) for post-migration configs — never applied in exact-match mode because it changes the golden DAG.
- *P2P.* One `TransferOp` = one identity: `(src_rank, dst_rank, tag=uid)` is computed once and used by both endpoints. The legacy failure mode (send/recv drawing independent fallback tags) is unrepresentable. SEND completes unconditionally once issued; RECV never occupies a slot; every RECV's matching SEND exists (V3) and its producer chain is acyclic (V1), so every SEND eventually issues.
- *Singleton groups* are lowered to `COMP` no-ops before AstraSim ever sees them (Pass 1 rule), keeping dependency chains intact.
- *Engine minimums.* `>=2` ranks enforced by the duplication shim; all comm sizes `>= 1` byte for controls.
- *Global acyclicity.* The union of all per-rank ET dep edges is the image of Program dep edges (intra-rank ctrl deps) plus send→recv pairs (image of TransferOps). Program is a DAG (V1); the image is therefore acyclic across ranks, so the one-compute-slot/one-comm-slot scheduler always has an issuable op until completion.

Determinism: emission is a pure function of `Program` — no sets iterated in id() order, no module globals, no clock.

### 3.2 Analytical simulation

Stages S1–S5: untouched — legacy `Graph.simulate` keeps running on the legacy graphs the (new) builders produce.

Stage S6: `analytical_sim.simulate(program, dp_index=0) -> float` ports the list scheduler with legacy-faithful semantics, explicitly documented:
- event heap keyed `(finish_time, insertion_counter)`;
- one outstanding `ComputeOp` per device (`GPU_list`), collectives/transfers unconstrained;
- ready-list is scanned in append order after every completion; children append order in builders is pinned (see §5, R11);
- comm durations come from `convert_comm_sizes_to_times` logic relocated to a pure function `comm_time(op, network_model, interconnect_params)`.
The gate is end-to-end golden times for all `analytical` and `hybrid` specs; a differential test asserts equal totals between `Graph.simulate(legacy_root)` and `simulate(program)` across the matrix before the legacy path is deleted.

### 3.3 Memory replay

Stages S1–S5: untouched (replays the flattened legacy graph from the new flatten builder).

Stage S6: `memory_replay.replay(program, memory_data, mode) -> (time, per_device_peaks)`. Consumes `ComputeOp.{device, duration, mem_kind, direction, recompute, param_gather, is_moe_layer, layer}` — the exact field set `simulate_memory` reads today. The layer-count census (`_collect_graph_layout`: dense/moe forward layers per device) becomes a Program scan over `role == GEMM`/`TRANSFORMER_LAYER` forward ops keyed by `(device, layer)`. Static allocation, transient alloc/release, ZeRO-3 `param_gather` ephemeral accounting, and the `_MemorySnapshot` logging format are ported verbatim. Peak GiB is not directly golden-gated, so we add a temporary differential assertion (old vs new peak per device, exact float equality) over the matrix before deleting `simulate_memory`.

### 3.4 Hybrid / hierarchical retiming

Unchanged control flow: dense and MoE transformer Programs (forward and backward separately, `dp_override=1`, transformer-subset layout) are emitted and executed; per-(dp, stage) fault variants re-run with `faulty_links_override`.

`retime.apply_stage_durations(pipeline_program, dense, moe, stage_overrides, dp_count)` replaces `_apply_transformer_time`/`_assign_transformer_durations`: it rewrites `duration` on every `ComputeOp` with `role == TRANSFORMER_LAYER`, selecting dense vs MoE by `is_moe_layer` and building the per-DP tuple from `stage_overrides[(dp_idx, device_stage)]` exactly as today. No name matching (`startswith("transformer_layer")` dies). Durations are the only mutation retiming may perform — deps, uids, and groups are frozen, so retiming can never perturb ET structure (today it can't either, but only by accident).

During S1–S5 the legacy write-back keeps operating on legacy pipeline nodes; `retime.py` lands with S6.

### 3.5 Inference (prefill and sampled decode)

Prefill: `_prepare_execution_graphs(include_*_backward=False)` — the same builders with `Direction.FORWARD` only; forward/backward extraction (`extract_forward_graph`/`extract_backward_graph`) becomes unnecessary at S4/S5 because builders take `direction` parameters natively (they already do for the transformer graph; the pipeline builder gains `include_backward` it already has). Decode: `prepare_decode_graphs` feeds decode GEMM shapes/timings into the identical builder path; the `InferenceEngine` sampling loop and the `dp_override=1` inference rule in the dispatcher are untouched. KV-cache/memory data flows through `build_memory_data` unchanged.

### 3.6 Fault projection and gmap/SCOTCH remap

- **Faults**: `FaultSpace` keeps consuming `axis_layout_from_descriptor(layout.descriptor())`. Projection to transformer/pipeline subsets, the per-(dp, stage) transformer fault map, and the analytical/hybrid rejection are unchanged dispatcher logic; only the descriptor's producer changes (S0).
- **gmap**: today traffic collection is interleaved with conversion and `_remap_stages_for_mapping` mutates rank bookkeeping mid-flight. New: `placement_opt.optimize(program) -> Program` runs *before* emission — it scans `CollectiveOp`s (same axis filter, participant counts, ALL_REDUCE/ALL_TO_ALL double-weight rule) and `TransferOp`s (pipeline bytes), builds the same CSR via `gmap.GMapCollector`, calls `gmap.finalize_collection`, and applies the permutation by **relabeling device ids** through one bijection (devices list, group members, op devices). Emission stays permutation-unaware; `rank = dp * num_stages + index(remapped_device)` reproduces legacy rank assignment. Byte-identical traffic matrices are required (same accumulation order: uid order == legacy collection order) so SCOTCH returns the same permutation (R15).

### 3.7 Overlap transforms

S1–S5: `apply_overlap_transforms` keeps rewriting legacy graphs (transformer graphs at construction, flattened graph post-build), unchanged.

S6: overlap becomes builder-integrated (`overlap.py`): the GEMM template expansion emits `head`/`tail` compute pairs for tp/tp_sp overlap and `block`/`ovlp` collective pairs for cp overlap directly, with the same byte-split (`ceil(total*(1-overlap))`) and the same rewiring semantics (tp edge hoisted to head; attention consumes block edge; ovlp edge closes after attention; `overlap >= 1.0` degenerate cases preserved). Creation order is specified to match the legacy post-transform traversal so op uids and hence wall times match (R16); if exact order proves unreachable for some config, that spec's golden is regenerated with the justification template (§5).

### 3.8 Rank layout derivation

`derive_layout_bundle` (S0) is the single producer. Consumers: dispatcher (fault init, transformer/pipeline subsets), builders (device-id computation — replacing `_hw_id_for_rank` in the flattener *and* the duplicated copy in `memory_estimation.py`), emission (`axes_filter` via `derive_axes_filter`), config generation, gmap. The `hw_id = tp + cp*tp + ep*tp*cp` transformer convention and the dp-major rank formula are unchanged facts of the layout, now computed in exactly one place.

---

## 4. Migration plan

Rules for every stage: land behind a merge only when `pytest tests/test_equiv_golden.py` (35/35) and the new differential/unit tests pass; no golden regen unless the stage's plan explicitly budgets one (none below S6 does); each stage lists what legacy code it deletes.

**S0 — Layout extraction (pure refactor).**
Add `workload/rank_layout.py`. `LLMExecutionDispatcher._build_rank_layout_descriptor` and `memory_estimation._build_rank_layout`/`_hw_id_for_rank` delegate to it; `_astrasim_rank_layout` root attributes keep carrying `layout.descriptor()`.
*Deletes:* the duplicated layout code inside `memory_estimation.py` (~80 LOC); dispatcher method body shrinks to a call.

**S1 — IR + lowering, shadow mode.**
Add `ir.py`, `legacy_lowering.py`, `test_ir_invariants.py`. `lower_to_program(graph_root, dp_size, layout)` reproduces converter Steps 1–7 exactly: DFS collection order, stage attribution (`local_hw_id` → parent `hw_id` → child `hw_id`), both dependency walkers, stage-task assembly (including stages discovered only via collectives), per-stage Kahn toposort keyed by `op_id` with tie counters, label assignment by `(base_name, primary member set)`. Program uids are assigned in the exact per-rank append order legacy emission would use (concatenation of stage toposorts for main ops; consumer-ordered TransferOps). A shadow test lowers every golden-matrix graph and re-emits with a *prototype* emitter into a temp dir, then asserts `equiv.canonical` equality against the legacy converter's output (no AstraSim execution — fast).
*Deletes:* nothing.

**S2 — Emitter cutover.**
Add `et_emission.py`, `placement_opt.py`, `emitted.py`. `convert_rapid_llm_graph_to_chakra_et` becomes: `lower_to_program` → `placement_opt.optimize` (when `_optimize_2dmap` present) → `emit_chakra` → return `(et_prefix, rank_ids, manifest_path)`. `run_astra_simulation_only_onepath` and everything below it unchanged. Golden gates 1–5 all bind here (exact wall seconds require identical id ordering — this is why S1 pinned uid order).
*Deletes:* the ~1,150-line body of `convert_rapid_llm_graph_to_chakra_et`; `_RankTrace` moves into `et_emission.py`; `_assign_collective_labels`, `_build_axis_groups`, `_compute_stage_axis_coords`, both walkers move into `legacy_lowering.py` (annotated `TRANSITIONAL — delete at S7`); module globals `_LAST_TP_GROUPS`, `_TP_LABEL_DP_TO_ID`, `_TP_MEMBERS_TO_GID`, `_LAST_STAGE_GROUPS`; `_remap_stages_for_mapping`.

**S3 — Flattener replacement (legacy node types, explicit metadata).**
Add `flatten_build.py` producing a legacy `Node`/`Edge` flattened graph directly from (pipeline graph spec, GEMM template(s), `LayoutBundle.full`): per-rank chains, per-rank pipeline edges with `is_cross_layer=True` and explicit `(src_stage, dst_stage)`, ZeRO-3 per-rank gather edges placed by declared rank (no sibling scan, no `±par_degree` offsets, no `"bwd" in name`), optimizer/softmax expansion, `local_hw_id` set at creation. Creation order contract: identical to `PipelineGraphFlattener`'s clone traversal (documented per node class), so `op_id`s — and therefore toposort keys, tags, and wall seconds — match. Differential test: for every flattened golden spec, `lower_to_program(old) == lower_to_program(new)` field-by-field.
*Deletes:* `PipelineGraphFlattener` (~640 LOC), `_propagate_local_hw_ids`, `_ensure_zero3_per_rank_edges`, the flattener's `_hw_id_for_rank`.
*Also lands:* fast-path in `legacy_lowering.py` — when edges carry explicit stage/pair metadata (all S3+ graphs), stage attribution and pipeline-pair recovery become table lookups; the walker fallback remains for hybrid/hier graphs.

**S4 — Transformer graph builder.**
`transformer_build.py` replaces `construct_transformer_graph` (dense + MoE hot/cold joins + residual p2p), still emitting legacy types, taking `direction` explicitly so `extract_forward_graph`/`extract_backward_graph` are no longer needed for transformer roots (they already aren't — the constructor takes `direction`; the extractors are used for other flows and die at S5). Same creation order.
*Deletes:* `construct_transformer_graph` (~350 LOC), `_moe_hot_rank`/`_moe_parallel_token` helpers (rehomed).

**S5 — Pipeline graph builder.**
`pipeline_build.py` replaces `construct_fwd_bwd_graph`, in four internally-reviewed sub-landings (one PR each, goldens green after each): (a) dense GPipe forward/backward + cross-microbatch deps + optimizer; (b) DP/ZeRO-2 collective lattice with `_should_emit_dp_comm` semantics; (c) ZeRO-3 gather lattice including the cross-device `skip_non_comm_children`/`skip_comm_children` attachment rules, expressed as explicit "attach to pipeline-edge successors only / non-pipeline successors only" declarations; (d) grad-accum final/nonfinal variants, MoE ep-sync, recompute nodes. Each sub-landing has a graph-isomorphism differential test against the legacy constructor (node multiset + adjacency + op_id sequence).
*Deletes:* `construct_fwd_bwd_graph` (~630 LOC), `attach_parallel_edge`, `extract_forward_graph`/`extract_backward_graph` (~310 LOC; builders emit direction-scoped graphs natively).

**S6 — Native Program construction + consumer ports.**
Builders gain `to_program()` outputs (uid order = the already-pinned creation order); `adapters.program_to_legacy_graph` bridges `Graph.simulate`/`simulate_memory` while `analytical_sim.py` and `memory_replay.py` are brought to exact parity (differential tests on totals and per-device peaks); `retime.py` and builder-integrated `overlap.py` land; dispatcher switches consumers one at a time: (a) AstraSim paths take Programs directly (lowering skipped for S3+ graphs — `legacy_lowering` now used only by nothing), (b) analytical simulate, (c) memory replay. Budgeted golden regen: only if analytical scheduler parity is impossible for some spec after emulating ready-list semantics — treated as last resort with per-spec diff review; target is zero regens.
*Deletes:* `apply_overlap_transforms` legacy rewrites and `_split_tp_node`/`_split_cp_edge` (~300 LOC), `_apply_transformer_time`/`_assign_transformer_durations`, `adapters.py` at stage end.

**S7 — Retirement.**
*Deletes:* `legacy_lowering.py`; `simulate_train_graph.Node/Edge/Data_batch/Graph` (construction, `simulate`, `simulate_memory`, `convert_comm_sizes_to_times`, clone/reset helpers — ~1,600 LOC; `visualize_graph` is rewritten as a ~100-LOC Program renderer); `flatten_build`'s legacy-type emission path. `simulate_train_graph.py` is removed; `llm_execution.py` shrinks to the dispatcher (~600 LOC).

---

## 5. Equivalence risk register

Gate legend: **G1** per-rank op multisets, **G2** DAG Merkle hashes, **G3** exact AstraSim per-rank wall seconds, **G4** end-to-end times, **G5** dlsim completability. "Match" = must be bit/structure-identical; "Regen" = deliberate change, golden regenerated with committed justification.

| # | Divergence point | Gates hit | Handling |
|---|---|---|---|
| R1 | Per-stage toposort order (Kahn keyed by `op_id`, heap tie counters) | G3 (ids=priorities), G5 | Match: lowering reproduces the exact heap discipline incl. `itertools.count` tie sequence. Covered by S1 shadow test. |
| R2 | Communicator gid allocation order (labels sorted, member-set dedup, base 1000) and label suffixes `name_N` | none (canonical resolves gids to member sets) but `comm_groups.json` diffs | Match anyway (verbatim allocation) to keep artifact diffs quiet; correctness rests on member sets. |
| R3 | Control-send/recv renumbering | G3, G5 | Match: identical stable-partition algorithm, identical `_send_control` suffix rule (`size==0` or non-PIPELINE edge). |
| R4 | P2P tags: legacy uses `edge.op_id`, fallback counter from 1e6; new uses `uid` | none (canonical excludes tags; timing tag-independent) | Justified silent difference: tags only pair send/recv; pairing is per-TransferOp identical. Documented; no regen needed since no gate observes tags. |
| R5 | Manifest op ordering (`_manifest_op_key` sort) | none directly, but manifest is the AstraSim cache key | Match: same sort, else every golden run becomes a cache miss (slow suite) and `cache.json` layout shifts. |
| R6 | Singleton-group no-op lowering | G1, G2, G5 | Match: same predicate (resolved member count ≤ 1), same zero-duration COMP with `_noop` name. |
| R7 | 1-rank duplication + prune | G3 rank counts | Match: kept in the wrapper, `axes_filter=["synthetic2"]` path untouched. |
| R8 | dp-major rank order; stages discovered only via collectives extend `stage_to_ranks` mid-conversion | G1–G3 | Match: `Program.devices` includes collective-only devices; `num_stages` computed *before* extension exactly as legacy (`rank = dp * num_stages_initial + idx`) — this quirk is reproduced and unit-tested. |
| R9 | `dp_count<=1` skip of DP stage collectives; `is_dp` attribution | G1 | Match: `CollectiveOp.is_dp` carries the legacy flag; skip in emitter. |
| R10 | Flattener creation order → `op_id` sequence → toposort keys, tags, wall times | G3 | Match: S3 builder pins iteration order (mb → clone traversal order → per-rank r ascending → gemm chain order); differential test compares full `op_id` sequences, not just multisets. |
| R11 | Analytical scheduler event order (heap counter, ready-list scan order, children append order) | G4 | Match: S5 builders replicate children-append order of legacy constructors (asserted by adjacency-sequence differential test); S6 port emulates scan semantics. Fallback: keep legacy `Graph.simulate` + adapter until parity proven (adapter preserves child order). |
| R12 | Hybrid/hier forward/backward subgraph membership (`extract_*_graph` inclusion rules: comm edges, `local_comp` nodes) | G1, G2 (hier bundles), G4 | Match: direction-scoped builders must produce the same op multiset as extraction of the both-direction graph; differential test at S4/S5 lowers both and diffs Programs. |
| R13 | Memory replay peaks (layer census, ephemeral ordering) | reported values, capacity warnings | Match via temporary exact-equality differential test at S6; not golden-gated, so the test is the gate. |
| R14 | Wall-time float path: `int(round(sec*1e6))`, duration-profile indexing | G3 | Match: identical rounding expression; profile length validated (V4). |
| R15 | gmap traffic matrix accumulation order / `_scotch_weight` rounding → SCOTCH permutation | G3 (rank relabeling), G1 per-rank | Match: collection iterates uid order == legacy `collective_info` insertion order; CSR construction unchanged (reuses `GMapCollector`). Differential test compares emitted `.grf` bytes. |
| R16 | Overlap transforms moving into builders (S6) — creation order of head/tail and block/ovlp ops | G1–G3 | Match target: builder emits in legacy post-transform traversal order. If a config's order is unreachable, Regen: per-spec, with diff showing identical multiset+DAG and wall-second delta attributed solely to id permutation, reviewed and committed. |
| R17 | ctrl_deps ordering within a node (first-seen dedup order) | none (G2 sorts parent hashes; AstraSim treats deps as a set) | Match anyway (cheap), avoids byte-level ET churn in artifact diffs. |
| R18 | `serialize_groups` transform (new safety option) | G2 if applied | Never applied to golden-covered configs during migration; off by default; enabling it later for new configs is a Regen with the V6-warning evidence attached. |
| R19 | Node/edge *names* (new builders may normalize) | none (canonical ignores names) — but control-class detection is name-suffix based in Pass 3 | Emission classifies controls by `TransferOp.size_bytes == 0`, not by name; names are then free. The `_send_control` suffix is still emitted for human/debug parity. |

---

## 6. What is deliberately NOT changed, and why

- **AstraSim itself, `integration.py`, `config_generation.py`, `et_utils.py` encoding, and the cache (`run_cache_astrasim`, `cache.json`, manifest keying).** The contract was reverse-engineered against these exact binaries; touching them invalidates the experimental basis of the contract and every golden.
- **The verified contract behaviors as emitter policy:** dp-phantom replication (one trace per stage stamped per dp index), control-priority renumbering, singleton-group no-ops, the 2-rank minimum shim, gid base 1000, `comm_groups.json` schema. These are facts about AstraSim, not legacy debt.
- **Timing computation upstream:** all of `train_timing.py`'s op/GEMM timing, `_build_comm_metadata` byte counts, `COMMUNICATION_RULES`, precision math, and `MemoryEstimator.build_memory_data` sizing. The rewrite replaces *graph plumbing*, not the performance model; conflating the two would make every golden diff unattributable.
- **Scheduling semantics of the analytical simulator and the GPipe/ZeRO lattice shape.** They are reproduced, not redesigned — including known modeling choices (e.g. `local_comp_time` forced to 0 in `create_comm_edge`, backward `b==0` meaning "last microbatch"). Changing schedule semantics is future work that should be its own golden-regen event with a performance-model justification, not a side effect of a refactor.
- **Interleaved pipelining as the closed-form `_pipeline_interleave_scale` correction.** Putting virtual stages into the graph is a modeling change with its own validation burden; the Program does not preclude it later (device ids are opaque), but this effort does not attempt it.
- **The rank-layout canonical order (`tp,cp,ep,pp,dp`), dp-major rank formula, and transformer `hw_id` convention.** They are load-bearing across faults, gmap, `config_generation`, and the goldens; `RankLayout` freezes them as the single definition rather than changing them.
- **The `equiv/` harness, canonical form, and golden format.** It is the safety net; it must not move while everything else does.
- **The dispatcher's mode structure** (four modes + inference `dp_override=1` + grad-accum dual-dispatcher final/nonfinal flow). Orchestration is fine; only what it orchestrates changes.
- **MoE-in-flattened stays rejected until after S7.** The design makes it *possible without violence* — `flatten_build` selects per-layer GEMM templates via `is_moe_layer` and `transformer_build`'s MoE join/residual emission is layout-parameterized, so enabling it is "allow the template + add goldens", not a rearchitecture — but enabling it during migration would add an un-goldened behavior to the exact window where we need the ground to stay still.
- **SCOTCH/gmap algorithms and `FaultSpace` math.** Consumed through existing entry points; only the call site moves (into `placement_opt.py`) and only the input source changes (Program scan instead of converter-interleaved collection), byte-compatibly (R15).