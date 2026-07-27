# INTERFACES.md — the authoritative L0–L4 interface contract

> **Status: normative.** This document is the contract the restructure phases P2–P6 implement
> against. Later agents implement one level each, in parallel, and **must not need to negotiate**:
> anything not fixed here is fixed by the signatures below, and anything that contradicts them is a
> bug in the implementation, not in this file. Amendments go here first, with a dated note.
>
> Read with: `PLAN.md` (phases/gates), `BUG_LEDGER.md` (A/B/C/D classification),
> `../CONTEXT.md` (AstraSim workload contract), `source_of_truth.md` / `special_cases.md` /
> `ext_*.md` (the audit this replaces).

---

## Amendment log

| Date | § | Amendment | Why |
|---|---|---|---|
| 2026-07-26 | §1.2, §1.6, §1.7, §1.8, §3.4 | `BlockTemplates` carries `dense_comm` / `moe_comm` `CommSpecTable`s; `WorkloadSpec.comm` is **pipeline-level keys only**; new `WorkloadSpec.block_comm(layer)` / `all_comm_specs()`; `BlockExpander` resolves every block comm key through the template for `work.layer`; new invariant **W5** | **B1.** Block comm keys are named PER TEMPLATE (`train_timing._build_transformer_template` holds one `_register_specs` accumulator per call, and `_register_specs` itself RAISES on a byte conflict — `train_timing.py:4694-4711`). Measured on `train:hybrid:dp2tp2cp1pp2mb2sp1:moe:ep2`: `ep_dense_sync_layernorm1_backward` is **337,641,472** bytes dense and **67,141,632** MoE (5.0288x). One flat table keeps one → 5x wrong EP-sync bytes on both MoE golden rows; a mixed `moe_layer_mask` needs both at once and a flat table cannot express it. |
| 2026-07-26 | §2.2, §3.3, §4.4 | `dp` is not a communicator axis of a device layout: `Placement.group_layout` (= `layout` minus `dp`) backs the `CommunicatorFactory`; `groups.py::_spanned_axes` **raises `GroupError`** on `"dp"`; `CommunicatorFactory.groups_for` returns `Tuple[Optional[GroupKey], ...]` with `None` per instance device of a dp requirement; new `SyncRequirement.is_dp`, and `dp` may not be composited with a device axis; `DP_AXIS` lives in `program/types.py` | **B2.** `Placement`'s device space included `dp`, so a dp-axis requirement got a communicator whose members are not devices (COARSE `dp:(0,2)` against `devices() == (0,1)`; FINE `(0,4)`), and at BLOCK it degenerated to a singleton that `et_emit` substitutes with a zero-duration `*_noop` — the reducer disappears. Contradicts §3.3 and `ir.py:92-99`. |
| 2026-07-26 | §2.2, §4.3 R2 | `ByteSource.bytes_for` honors `instances`: `ByteSplit.CEIL_DIV_CLUSTER` is `ceil(total / instances)`, **not** `ceil(total / fw.spec.cluster_size())` | **B3.** `Placement.cluster_size()` is 1 at COARSE (the stage IS the device) while `fw.spec.cluster_size()` is always `tp*cp*ep`. Legacy COARSE uses the RAW cross-layer bytes (`pipeline_coarse.py:222,238`), legacy FINE divides (`pipeline_fine.py:645`). At `tp=2` the old call returned 2048.0 where COARSE wants 4096.0 → wrong `cross_layer` bytes on every coarse/hybrid/hierarchical row with `cluster_size > 1`. |
| 2026-07-26 | §3.2 | `SyncSpread.PER_CLUSTER_RANK` resolves against `cluster_devices(stage_of(place_on))`, not `devices_for(place_on)`; `PlacementError` if it ever yields fewer than `cluster_size` devices | **B4.** `PER_CLUSTER_RANK` collapsed to ONE device whenever `place_on`'s kind is pinned to cluster rank 0 by `LEGACY_PLACEMENT` (EMBEDDING/SOFTMAX) — exactly ZeRO-3 rows **S7** and **S13**. Legacy `_ensure_zero3_per_rank_edges` builds `hw_ids` for every `par_degree` rank (`pipeline_fine.py:471-480`). Invisible in the matrix only because every zero2/zero3 spec is `tp=cp=1`. |
| 2026-07-26 | §2.4, §4.1 | `StagePartition` shrinks to **`same_stage(a, b)`** only; `ContiguousStages` implements `LayerAssignment`'s `stage_of(layer)` / `layers_of` / `min_layer` and its WorkItem rule is renamed `stage_of_work` | **B5.** `StagePartition.stage_of(work: WorkItem)` and `LayerAssignment.stage_of(layer: LayerId)` shared a name with incompatible argument types, so **no object satisfied both** — yet `Placement.__init__` consumes a `LayerAssignment` and `ShardingContext` a `StagePartition`, from one stage partition. `same_stage` was the only member L1 ever called. |
| 2026-07-27 | §4.1 | `Schedule` gains **`implied_deps(devices_for)`** (and `device_projection`, `last_microbatch_of`, `check_permutation`); new `ScheduleDep(before, after, device)`; `LayerAssignment` gains `explicit()` / `contiguous_layers()` | **L3.** §4.1's `Schedule` declared only an order. AstraSim's one-slot rule constrains CONCURRENCY, not ORDER, so an order that is not in the DAG is not an order: legacy materializes the cross-microbatch edge explicitly (`git show 85894c6:simulate_train_graph.py:836-856`). The schedule must therefore DECLARE those deps and R3 materialize them. `devices_for` is injected rather than imported so L3 stays free of L2 (`ScheduleDep` is per DEVICE — at FINE a stage is `cluster_size` devices and a pinned kind lives on one). `last_microbatch_of` is what `GradAccumPolicy.for_schedule` already called (Class B 10h). |
| 2026-07-27 | §4.2, §3.1 | `build()` gains keyword-only `placement_policy`, `sync_order`, `validate`, `check_group_membership`; new `restrict_work_for(granularity, work)` and `check_granularity_preconditions(granularity, fw)` | **L4.** §3.1 says BLOCK is "one layer over the (tp,cp,ep) sublayout only, **no pipeline**", but §4.2 phase 1 enumerates the whole workload. BLOCK's device space carries neither `pp` nor `dp`, so pipeline work and data-parallel sync are not REPRESENTABLE there: the restriction and the `dp==pp==1` precondition state that instead of leaving it to whichever dispatcher built the spec. With the precondition in place no granularity-specific POLICY override is needed — at `dp == 1`, `sharding_policy_for` already answers `NullSharding` and `GradAccumPolicy.emits` is unconditionally False, so neither the ZeRO lattice nor the EP grad sync can leak into a block measurement. |
| 2026-07-27 | §4.3 R2 | The zero-byte-vs-payload decision of a cross-layer link is **per STAGE, not per device**: `size = 0 if placement.same_stage(producer, consumer) else ByteSource("cross_layer", CEIL_DIV_CLUSTER).bytes_for(fw, placement.cluster_size())` | **L4.** R2's pseudocode keys the transfer on `dp == dc` (device). At FINE the embedding is pinned to cluster rank 0 (Class B 10d) while layer 0 spans the stage, so the `EMBEDDING -> LAYER 0` link IS cross-device inside ONE stage — keying on the device invented a `cross_layer` payload between tp ranks (measured: 4 bogus 2 MB transfers on `train:analytical:dp1tp2cp1pp2mb2sp1`). Legacy decides it at the coarse level with `prev_node.hw_id == curr_node.hw_id` (`schedule.py:628,639,648,755,763,771`), i.e. per STAGE, and the FINE expansion inherits the zero-byte control edge. |
| 2026-07-27 | §4.3 R2 | The `RECOMPUTE -> LAYER/BACKWARD` link *inside one layer* is a plain same-device dep, not a `TransferOp` | **L4.** R2 says a layer's backward entry is its RECOMPUTE chain, but not what joins the two. Legacy wires it as a direct `recompute_node.add_child(transformer_node_b)` (`schedule.py:750`) with no `CommEvent` — the rematerialized activation never leaves the device that recomputed it. |
| 2026-07-27 | §4.7, §4.8 | `validate_program` gains `check_group_membership` (**V7**, opt-in) and always-on **V8**; `TransferOp.moe_component`, `CollectiveOp.axes`, `Op.work`, `Op.succs` land as defaulted fields | **L4.** V7 ("every member device of a communicator issues the group's collectives") is CONTRADICTED by BUG_LEDGER **A2** (`SyncSpread.CLUSTER_RANK_0`: one instance on cluster rank 0), which §7 preserves as the default. Any non-dp requirement whose group spans >1 device therefore fails V7 today — e.g. the `ep` grad sync at `tp*cp*ep > 1`. V7 cannot be fatal by default without making today's modeling content unbuildable; flipping the spread in P7 is exactly what lets the default flip. V8 needs `moe_component` on the op to be checkable at all. |
| 2026-07-28 | §4.7 | `TransferOp` gains `participants` / `interconnect`; `CollectiveOp` gains `comm_key` — all defaulted | **P6.** The analytical evaluator needs a p2p's ANALYTICAL timing surface (participant count + interconnect axis) and the memory replay needs to know which comm TABLE a collective came from. Both were `CommEvent` fields on the deleted proto graph and no IR field expressed them; re-deriving either from the op's NAME would be rule-§8.2 name dispatch. `comm_key` is what makes "the memory replay times the PIPELINE-LEVEL collectives only" (§1.6's namespace split) statable as data instead of as a name test. |
| 2026-07-28 | §4.2, §3.1 | `build()` and `restrict_work_for` gain keyword-only `directions: Optional[Sequence[Direction]]` | **P5.** §4.2's own amendment says "a caller measuring a single direction builds a single direction's workload" but nothing could express it: `RunPolicy.include_backward` is all-or-nothing, and AstraSim returns ONE makespan per bundle while the hybrid/hierarchical retiming needs a forward time AND a backward time. `directions` is the restriction, stamped onto `meta.misc["work_directions"]`. `WorkloadSpec.for_block(layout, moe)` is its companion: the BLOCK workload (1 layer, 1 microbatch, `pp=dp=1`, recompute off, one template in the dense slot). |
| 2026-07-28 | §4.6, §5 (T1 tie discipline) | **Program order (ascending uid) IS the consumer tie discipline.** The successors of a finished op are visited in ascending uid, the initially-ready set is seeded in ascending uid, the ready list stays FIFO | **P6's normative deliverable.** The legacy replays' outcome depended on children-list adjacency order — an artifact of the construction sequence, declared nowhere, so it could not survive the cutover. Measured before choosing: on all 10 fully-analytical specs plus every hybrid spec, uid order and `Op.succs` construction order give BIT-IDENTICAL totals, and both match the deleted proto-graph evaluator. Program order was therefore free, and it is the order AstraSim already consumes as node-id priority. Two legacy quirks are preserved and NAMED rather than rediscovered: `analytic_sim._ROOT_COMM_IS_UNTIMED` (the conversion pass only converted *children*, and the ZeRO-3 forward-entry gather IS the coarse root — timing it moves `train:analytical:*:zero3` by +5.76%), and "the memory replay times the pipeline-level collectives only" (the coarse conversion ran BEFORE flattening, so per-rank pipeline transfers and block-template comms were created at `duration=0`). |
| 2026-07-28 | §4.9, new `program/mapping.py` | The first-dimension SCOTCH remap is a POST-BUILD pass over the ops (`apply_first_dim_mapping`), not a step inside a lowering | **P5.** Steps 3/8 of `legacy_lowering.lower_to_program` read a mutable proto graph. Over the IR the whole remap is: collect per-axis collective bytes and cross-stage pipeline bytes off the ops, ask SCOTCH for a permutation, and reorder `Program.devices` — because `devices.index(d)` IS the stage index the dp-major rank formula uses and communicator members are DEVICE ids, so permuting the order re-derives every rank and every wire group with nothing to re-run. Op uids are NOT recomputed (BUG_LEDGER C7's split, kept: mapping is not scheduling). |
| 2026-07-28 | §4.8 (V6) | **V6 is narrowed to pairs whose EARLIER member has a successor** | **P5.** §4.8 predicted the false-positive class and asked P5 to consider exactly this. V6 is always-on in `build()` now, and a gradient reducer is a graph SINK by construction, so every pair of reducers of one group on one device tripped it. Two sinks cannot race: nothing observes which issued first for a program whose order is ONE global order projected onto every device (CONTEXT constraint 2). Production no longer has to silence `GroupRaceWarning` to get a clean log. |
| 2026-07-28 | §4.3 R4 / L5 | A SYNC dep may cross devices in the IR; **the EMITTER is where "a `ctrl_dep` is rank-local" lives** | **P5, and it cost one wrong attempt.** A ZeRO-3 gather inheriting a cross-stage anchor's deps (rows S6/S14) produces a dep between ops on different devices with no transfer carrying it. Filtering those out in `build()` looked right and moved `train:analytical:dp2tp1cp1pp2mb2sp0:zero3` by **-12.1%** — the analytical evaluator honors that edge, which is why legacy's total depended on it. So the IR keeps the edge (as legacy's coarse graph did) and `et_emit` drops it with a counted `CrossRankDepWarning`, because writing a foreign rank's node id aliases another trace (measured: it closed a cycle on that spec). Rank-locality is an ET fact, not an IR fact. |
| 2026-07-28 | §4.3 R2, L5 | **A same-device (elided) `TransferOp` must be replaced by a PLAIN dep on its own deps at emission** — eliding the op must not elide the edge | **P5.** `ir.py` claimed "the consumer already depends on the producer directly (legacy stage_deps behavior)", but in a `build()` program the ONLY path from producer to consumer is through the transfer. Dropping it let the ranks of one stage leave collective-issue lockstep: AstraSim reported `Hardware Resource sys.id=N has unreleased nodes` and returned zero time on `train:flattened:dp1tp2cp1pp2mb2sp1:recompute` — a silent deadlock that `equiv.dlsim` did NOT catch (it does not model dynamic issue order). Two sibling fixes landed with it: a tp-overlap split must move the transfer's `consumers` to the head along with the deps, and a chain step that depends on a block-template p2p must register as a consumer of it (the MoE cold-rank JOIN's ordering against its own residual send). |

---

## 0. Ground rules

### 0.1 Hard cutover

The owner does **not** care about old caches, old results files, or the legacy emission order.
Concretely, the following are **NOT** preserved and must not be reproduced by any implementer:

| Not preserved | Where it lives today |
|---|---|
| legacy node-id / op-id order | `schedule.py:499-505` `_stamp`, `pipeline_fine.py:264-266` `_next_op_id`, `legacy_lowering.py:645-670` Kahn tie-counter |
| `legacy_op_id`, `send_seq`, `recv_seq`, `post_deps`, `legacy_tag` | `program/ir.py:131,134,154,155,185,186,189` |
| op **names** (they are not part of canonical equivalence: `equiv/canonical.py:20-27`) | everywhere |
| AstraSim wall-clock drift induced by node-id priority changes | rebaselined once, later, with a reviewed delta table |

What **MUST** be preserved, per spec, exactly (T1):

1. per-rank op multisets: compute microseconds, collective (kind, size, member set), transfer
   (size, peer);
2. per-GPU memory peaks and the static/activation census;
3. analytical totals (bit-exact, 1e-9) for the 10 fully-analytical specs;
4. the AstraSim workload contract of `../CONTEXT.md` §"The AstraSim workload contract" —
   in particular: one global total order projected onto every communicator, one identity per p2p
   transfer, deterministic emission.

### 0.2 Policy rule (Class B)

Every **Class B** item in `BUG_LEDGER.md` is preserved as MODELING CONTENT but **must be expressed
as a NAMED, SWAPPABLE POLICY OBJECT with a `name` attribute** — never as an inline literal inside an
expansion loop. §7 is the binding table: every Class B row maps to exactly one named policy and its
default value. A reviewer must be able to flip any of them by constructing a different policy, with
no edit inside `build.py`.

Class **A** items are **not fixed in this wave**. They must, however, remain *expressible*: the
interface must admit the corrected behavior as a different policy value, so that P7 is a one-line
default change plus a delta table. §7 records, per A-item, which policy field carries it.

### 0.3 Module layout and landing order

```
L0  program/workload.py            WorkloadSpec, RunPolicy, DurationTable, CommSpecTable   (P2)
L1  program/work.py                WorkItem, WorkSet, SyncRequirement, AttachMode          (P2)
    program/policies/__init__.py   policy registry + `policies_for(spec)`                  (P2)
    program/policies/sharding.py   ShardingPolicy, DDP, ZeRO1, ZeRO2, ZeRO3                (P2)
    program/policies/gradaccum.py  GradAccumPolicy                                          (P2)
    program/policies/recompute.py  RecomputePolicy                                          (P2)
    program/policies/routing.py    MoERoutingPolicy, AxisRouting                            (P2)
    program/policies/overlap.py    OverlapPolicy, OverlapDecl                               (P2)
L2  program/placement.py           Placement, Granularity, CommunicatorFactory              (P3)
    program/block.py               BlockTemplate (extended: placement-aware expansion)      (P3)
L3  program/sched/policy.py        SchedulePolicy, Schedule, ScheduleSlot, LayerAssignment  (P4)
    program/sched/gpipe.py         GPipeSchedule                                            (P4)
L4  program/ir.py                  Program (ordering metadata deleted)                      (P5)
    program/build.py               build()                                                  (P5)
```

> **Package-name collision.** `program/schedule.py` still exists when L3 lands. The L3 package is
> created as **`program/sched/`** in P4 and renamed to `program/schedule/` in P5, in the same commit
> that deletes `program/schedule.py`. Do not create `program/schedule/` before then.

### 0.4 Shared type aliases

```python
# program/types.py  (new, ~25 LOC, no imports from program.* — breaks every cycle)
from typing import NewType, Tuple, Mapping

AxisName  = str                      # one of program.layout.CANONICAL_AXES
DeviceId  = NewType("DeviceId", int) # a linearized rank in a RankLayout
StageId   = NewType("StageId", int)  # a pipeline stage index (pp coordinate)
LayerId   = int
MicroBatch = int
CommKey   = str                      # index into CommSpecTable
Coords    = Mapping[AxisName, int]
```

`RankLayout` (`program/layout.py`) is unchanged and remains THE linearization. No implementer may
write a second rank formula; `block_program.py:188-189` and `memory_estimation.py:309-322` are
deleted, not ported.

---

## 1. L0 — `program/workload.py`

Replaces **both** `ScheduleInputs` (`schedule.py:108-143`) and the duck-typed
`ScheduleSpec.from_pipeline_graph` (`schedule.py:178-221`), which today silently accepts a wrong
object and yields `mb=0`/`num_layers=0` because every read is a `getattr(..., default)` /
`misc.get(..., getattr(...))` chain.

### 1.1 The core types

```python
# program/workload.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Mapping, Optional, Tuple

from timing_model import CollectiveType
from program.block import BlockTemplate
from program.layout import RankLayout
from program.types import AxisName, CommKey


class WorkloadError(ValueError):
    """A WorkloadSpec was constructed from inconsistent or incomplete inputs."""


class RunType(Enum):
    TRAINING = auto()
    INFERENCE = auto()


class GradAccumCycle(Enum):
    FINAL = auto()
    NONFINAL = auto()


class DpMicrobatchMode(Enum):
    EVERY_MB = auto()
    LAST_MB = auto()


@dataclass(frozen=True)
class ParallelDegrees:
    tp: int
    cp: int
    ep: int          # the GRAPH ep: time_calc.ep when use_moe else 1 (train_timing.py:4667)
    pp: int
    dp: int

    def cluster_size(self) -> int:
        """tp*cp*ep — the number of devices in one pipeline stage (>=1)."""
        return max(1, self.tp * self.cp * self.ep)

    def __post_init__(self) -> None: ...   # raises WorkloadError on any degree < 1


@dataclass(frozen=True)
class ModelShape:
    num_layers: int
    micro_batches: int                      # legacy misc["num_batch"]
    model_type: str                         # lowercase; "vit*" selects the ViT block naming
    moe_layer_mask: Tuple[bool, ...]        # () when dense-only; len == num_layers otherwise

    def is_moe_layer(self, layer: LayerId) -> bool: ...
    def __post_init__(self) -> None:
        # raises WorkloadError if num_layers <= 0, micro_batches <= 0, or
        # (moe_layer_mask and len(moe_layer_mask) != num_layers)
        ...
```

**`ModelShape.__post_init__` is the fix for schedule.py:186.** A wrong object can no longer produce
`mb=0`: `micro_batches <= 0` raises. There is no `getattr` fallback anywhere in L0.

### 1.2 CommSpec — the typed comm table

```python
@dataclass(frozen=True)
class CommSpec:
    """One entry of the comm table. BYTES AND KIND ARE COMPUTED UPSTREAM
    (train_timing) and are read-only here — the restructure does not touch
    byte/timing math (PLAN 'Owner decisions': blast radius)."""

    key: CommKey
    size_bytes: float                 # RAW; dp reducers are floats and MUST stay floats
    kind: CollectiveType              # ALL_REDUCE | REDUCE_SCATTER | ALL_GATHER | ALL_TO_ALL | PIPELINE
    axes: Tuple[AxisName, ...]        # COMMUNICATOR IDENTITY — declared, never inferred (§3.3)
    participants: int                 # ANALYTICAL participant count — see the note below
    ga_required_every_cycle: bool = False
    tp_shard: bool = False            # "this collective is instantiated per cluster rank"
    placement: Placement_ = "post"    # "pre" | "post"  (block-template comm keys only)
    local_comp_time: float = 0.0      # retained; Class B item 5 keeps it unused (§7)
    parallel_group: Optional[str] = None
    moe_component: Optional[str] = None       # "base_all_to_all" | "residual_p2p"
    moe_routing_mode: Optional[str] = None    # "ep" | "tp_ep"
    extra: Mapping[str, Any] = field(default_factory=dict)


class CommSpecTable(Mapping[CommKey, CommSpec]):
    """Immutable, insertion-ordered. `table.require(key)` raises WorkloadError
    (never KeyError) with the full key list in the message."""
    def require(self, key: CommKey) -> CommSpec: ...
    def keys_with_axis(self, axis: AxisName) -> Tuple[CommKey, ...]: ...

    @classmethod
    def from_legacy(cls, raw: Mapping[str, Mapping[str, Any]]) -> "CommSpecTable": ...

    # -- AMENDMENT 2026-07-26 (B1) ---------------------------------------
    @classmethod
    def from_block_template(cls, template: BlockTemplate) -> "CommSpecTable":
        """The comm table of ONE BlockTemplate (`template.comm_metadata`, the
        per-template `_register_specs` accumulator). Identity conversion via
        `CommSpec.from_comm_meta`. THERE IS ONE TABLE PER TEMPLATE — see §1.6."""
```

> **AMENDMENT 2026-07-26 (B1).** `CommSpec.from_comm_meta(meta: CommMeta) -> CommSpec` is added
> alongside `from_legacy`, and the shared `axes` derivation is factored into one
> `_declare_axes(key, interconnect, routing_mode)` helper used by both. `from_legacy` reads the raw
> `train_timing` dict; `from_comm_meta` reads the typed `program.block.CommMeta` that a
> `BlockTemplate` already carries. Neither defaults nor infers; a `CommMeta` with no
> `CollectiveType` raises `WorkloadError`.

> **`axes` vs `participants` are SEPARATE and must stay separate.** Today
> `COMMUNICATION_RULES[TENSOR_CONTEXT_HYBRID]['output_proj']['forward']` declares
> `participants='tp', interconnect='cp'` (`train_timing.py:151-154`) — the analytical participant
> count and the communicator axis genuinely differ. `participants` feeds
> `analytic_sim.convert_comm_sizes_to_times` (`analytic_sim.py:110`) and the gmap traffic weight
> (`legacy_lowering.py:586`); `axes` feeds the communicator (§3.3) and never the timing model.
> Collapsing them would change analytical totals. **Do not derive one from the other.**

Default `axes` derivation (`CommSpec.from_legacy`), which reproduces today's grouping exactly:

```python
axes = (interconnect_type,)                     # the normal case
axes = routing_policy.routing_axes()            # iff moe_routing_mode is set (§2.6)
```

This **deletes** the participant-count inference at `legacy_lowering.py:243-248`
(`if axis == "ep" and participants == tp*ep -> ("tp","ep")`): the composite case is now declared by
the MoE routing policy at the source (§2.6, §3.3).

### 1.3 DurationTable — the ONE mutable object, by design

`comp_times` is mutated after construction and re-read later. This is real, load-bearing behavior;
it is designed explicitly here rather than inherited by accident.

* **Writer:** `llm_execution._update_comp_times_from_timings` (`llm_execution.py:976-1004`) writes
  `transformer_f/b`, `transformer_f/b_dense`, `transformer_f/b_moe` after the AstraSim BLOCK runs
  (hybrid `:499`, hierarchical).
* **Reader-before:** `_run_hybrid` builds the COARSE program at `:497` **before** the write-back
  and then retimes it (`retime.apply_block_timings`, `:500`). It must see the PRISTINE analytical
  durations.
* **Reader-after:** `build_fine_program_for_memory` (`llm_execution.py:748`) constructs a fresh
  spec **after** the write-back and must see the UPDATED durations.

```python
class DurationTable:
    """The single mutable member of a WorkloadSpec. Every other field is frozen.

    Reads outside a snapshot are forbidden in builders: L1-L4 receive a
    `FrozenDurations` obtained from `snapshot()`, so a build can never observe
    a torn write. `revision` increments on every write-back and is stamped onto
    the built Program so stale reuse is detectable, not silent.
    """

    def __init__(self, values: Mapping[str, float]) -> None: ...

    @property
    def revision(self) -> int: ...

    def snapshot(self) -> "FrozenDurations": ...

    def write_block_timings(
        self,
        *,
        dense_forward: Optional[float] = None,
        dense_backward: Optional[float] = None,
        moe_forward: Optional[float] = None,
        moe_backward: Optional[float] = None,
    ) -> int:
        """The ONLY mutator. Writes the six legacy keys atomically and returns
        the new revision. Replaces the dict-poking at llm_execution.py:995-1004.
        Raises WorkloadError on a negative duration (the check now lives here,
        not at three call sites)."""


@dataclass(frozen=True)
class FrozenDurations(Mapping[str, float]):
    revision: int
    values: Mapping[str, float]
    def get_or(self, key: str, default: float = 0.0) -> float: ...
```

**Normative:** `build()` (§4.2) takes a `FrozenDurations`, never a `DurationTable`. Every built
`Program` carries `meta.misc["duration_revision"] = frozen.revision`. Any consumer that reuses a
cached Program across a write-back must compare revisions and rebuild on mismatch — this replaces
the implicit "the flattened program is identical to a fresh build" comment at
`llm_execution.py:726-731`.

### 1.4 RunPolicy — kills the triplicated derivations

```python
@dataclass(frozen=True)
class RunPolicy:
    """One home for the rules currently copied verbatim three times each
    (llm_execution.py:392-395 / :606-612 / :742-746 — audit A10)."""

    run_type: RunType
    grad_accum_cycle: GradAccumCycle
    dp_microbatch_mode: DpMicrobatchMode
    zero_stage: int
    pipeline_interleave: int = 1          # v; the closed-form bubble correction (Class B item 8)

    @property
    def include_backward(self) -> bool:
        return self.run_type is not RunType.INFERENCE

    @property
    def include_optimizer(self) -> bool:
        return self.grad_accum_cycle is not GradAccumCycle.NONFINAL

    def effective_dp(self, degrees: ParallelDegrees) -> int:
        return 1 if self.run_type is RunType.INFERENCE else max(1, degrees.dp)

    def retime_dp_count(self, degrees: ParallelDegrees) -> int:
        return self.effective_dp(degrees)     # legacy _retime_dp_count, same rule

    def interleave_scale(self, degrees: ParallelDegrees, shape: ModelShape) -> float:
        """Class B item 8, verbatim (llm_execution.py:353-368). Named here so
        the 1F1B student project can replace it with a real schedule."""
```

### 1.5 OverlapSpec — kills the three positional floats through eight signatures

```python
@dataclass(frozen=True)
class OverlapSpec:
    """Replaces `tp_overlap`/`tp_sp_overlap`/`cp_overlap` threaded positionally
    through build_fine_root, build_fine_program, build_block_program,
    apply_overlap_to_fine_root, apply_tp_overlap, _build_transformer_block_programs
    and three dispatcher call sites (ext_new_axis.md Part 4 item 3)."""

    parallelism_mode: Any                     # ParallelismMode enum from train_timing
    by_axis: Mapping[AxisName, float] = field(default_factory=dict)   # e.g. {"tp": 0.5, "cp": 0.25}

    def fraction(self, axis: AxisName) -> float:
        """0.0 when absent. `tp_sp` is NOT an axis: TENSOR_SEQUENCE mode selects
        by_axis['tp'] from the tp_sp_overlap input at construction, exactly like
        transforms.py:326-331 does today, but ONCE, in the constructor."""
```

### 1.6 WorkloadSpec

```python
@dataclass(frozen=True)
class WorkloadSpec:
    degrees: ParallelDegrees
    shape: ModelShape
    run: RunPolicy
    comm: CommSpecTable               # PIPELINE-LEVEL KEYS ONLY (amendment 2026-07-26)
    blocks: "BlockTemplates"          # dense (required) + moe (optional), each WITH its comm table
    overlap: OverlapSpec
    layout: RankLayout                # FULL layout; L2 derives every sublayout
    interconnect: Mapping[str, Tuple[float, float]]   # axis -> (bandwidth, latency)
    granularity_hint: "Granularity"   # what the dispatcher intends to build (§3.1)

    #: THE ONLY MUTABLE MEMBER. Never read directly by a builder — see §1.3.
    durations: DurationTable = field(compare=False)

    # -- derived, pure ---------------------------------------------------
    def is_moe_layer(self, layer: LayerId) -> bool: ...
    def cluster_size(self) -> int: ...
    def block_template(self, layer: LayerId) -> BlockTemplate: ...
    def block_comm(self, layer: LayerId) -> CommSpecTable: ...   # amendment 2026-07-26
    def all_comm_specs(self) -> Tuple[CommSpec, ...]: ...        # amendment 2026-07-26
    def freeze(self) -> "FrozenWorkload": ...


@dataclass(frozen=True)
class FrozenWorkload:
    """A WorkloadSpec + a duration snapshot. This — not WorkloadSpec — is what
    L1/L2/L3/L4 consume, so a build is a pure function of its input."""
    spec: WorkloadSpec
    durations: FrozenDurations
    # pass-throughs: degrees / shape / run / comm / blocks / layout /
    # is_moe_layer / cluster_size / block_comm / all_comm_specs


@dataclass(frozen=True)
class BlockTemplates:
    dense: BlockTemplate
    moe: Optional[BlockTemplate] = None
    # -- AMENDMENT 2026-07-26 (B1) ---------------------------------------
    dense_comm: Optional[CommSpecTable] = None   # derived from dense.comm_metadata if None
    moe_comm: Optional[CommSpecTable] = None     # derived from moe.comm_metadata if None

    def template_for(self, is_moe_layer: bool) -> BlockTemplate: ...
    def comm_for(self, is_moe_layer: bool) -> CommSpecTable: ...
    def tables(self) -> Tuple[CommSpecTable, ...]: ...   # scans only, never resolution
```

> **AMENDMENT 2026-07-26 (B1): `WorkloadSpec.comm` is ONE table but block comm keys are
> PER-TEMPLATE.**
>
> `train_timing._build_transformer_template` (`:4676-4959`) builds `transformer_comm_metadata` as a
> **local** accumulator, once per call, and is called twice — `use_moe_layer=False` with
> `ep_dense_sync_bytes_dense` and `use_moe_layer=True` with `ep_dense_sync_bytes_moe` (`:4961-4975`).
> `_make_ep_dense_sync_specs` (`:4725-4749`) names the spec
> `ep_dense_sync_{op}_{direction}` with **no dense/MoE discriminator**, so both templates register
> the same key with different bytes. Measured on `train:hybrid:dp2tp2cp1pp2mb2sp1:moe:ep2`:
>
> | key | dense | moe | ratio |
> |---|---|---|---|
> | `ep_dense_sync_layernorm1_backward` | 337,641,472 | 67,141,632 | 5.0288 |
>
> `_register_specs` (`:4691-4723`) **raises `ValueError`** on exactly this conflict, which is the
> upstream proof that the two namespaces are distinct. A flat union keeps one entry → 5x wrong
> EP-sync bytes on both MoE golden rows, and with a mixed `moe_layer_mask` both values are live
> simultaneously so no single-valued table can express the workload at all.
>
> Therefore:
>
> * `WorkloadSpec.comm` holds **only** what `train_timing._build_comm_metadata` (`:4380-4506`)
>   produces: `transformer_dense`, `transformer_moe`, `embedding`, `softmax`, `cross_layer`,
>   `transformer_{dense,moe}_ep_sync`, `zero2_*_gather`, `zero3_*_gather`.
> * every block-template key is resolved through `WorkloadSpec.block_comm(layer)` →
>   `BlockTemplates.comm_for(is_moe_layer)`.
> * a key declared in **both** namespaces with different `size_bytes` / `kind` / `axes` /
>   `participants` is a `WorkloadError` at construction. Identical content is allowed (test fixtures
>   share one dict); the dense and MoE tables disagreeing with **each other** is the point and is
>   never an error.
> * whole-workload SCANS (`routing_policy_for`, byte histograms, diagnostics) use
>   `all_comm_specs()`, which returns pipeline-level specs followed by each block table's. Note
>   `moe_routing_mode` is set **only** by `_make_moe_comm_specs` (`train_timing.py:601,617`), whose
>   specs are block-level — scanning `comm` alone finds nothing.

### 1.7 Field-by-field producer map (train_timing.py)

Every field, its type, and the exact expression in `train_timing._prepare_execution_graphs`
(`train_timing.py:4520-5117`) that produces it. **These expressions do not change** — only their
destination type does.

| WorkloadSpec path | Type | Producer (train_timing.py) |
|---|---|---|
| `degrees.tp` | `int` | `self.tp` — `:5084` |
| `degrees.cp` | `int` | `self.cp` — `:5085` |
| `degrees.ep` | `int` | `graph_ep = self.ep if self.use_moe else 1` — `:4667`, `:5086` |
| `degrees.pp` | `int` | `self.pp` — `:5083` |
| `degrees.dp` | `int` | `self.dp` — `:5082` |
| `shape.num_layers` | `int` | `misc_metadata["num_layer"] = self.num_layers` — `:5067` |
| `shape.micro_batches` | `int` | `misc_metadata["num_batch"] = self.mb` — `:5066` |
| `shape.model_type` | `str` | `misc_metadata["model_type"] = self.model_type` — `:5074` |
| `shape.moe_layer_mask` | `Tuple[bool,...]` | `moe_layer_mask = list(self.moe_layer_mask or [])` — `:4967`, `:5073` |
| `run.run_type` | `RunType` | `self.model.run_type` (today re-read in the dispatcher at `llm_execution.py:371`) |
| `run.grad_accum_cycle` | `GradAccumCycle` | `misc_metadata_final["grad_accum_cycle"]="final"` `:5077` / `_nonfinal="nonfinal"` `:5079` |
| `run.dp_microbatch_mode` | `DpMicrobatchMode` | `misc_metadata["dp_microbatch_mode"] = self.dp_microbatch` — `:5072` |
| `run.zero_stage` | `int` | `misc_metadata["dp_zero_stage"] = self.zero_stage` — `:5068` |
| `run.pipeline_interleave` | `int` | `self.pipeline_interleave` (today read in `llm_execution.py:363`) |
| `durations["embedding_f"]` | `float` | `node_breakdown['embedding_f']` — `:5025` |
| `durations["embedding_b"]` | `float` | `node_breakdown['embedding_b'] if include_pipeline_backward else 0.0` — `:5026` |
| `durations["linear_softmax_f"]` | `float` | `:5027` |
| `durations["linear_softmax_b"]` | `float` | `:5028` |
| `durations["transformer_f"]` / `_b` | `float` | `:5029-5030` |
| `durations["transformer_f_dense"]` / `_b_dense` | `float` | `:5031-5032` |
| `durations["transformer_f_moe"]` / `_b_moe` | `float` | `:5033-5034` |
| `durations["optimizer"]` | `float` | `self.get_data_parallel_reduction_llm(hidden_dim, intermediate_size)` — `:5035` (Class D 10b: per-STAGE, not per-layer; unchanged here) |
| `comm["transformer_dense"]` | `CommSpec` | `_build_comm_metadata` `:4397-4404` — kind `grad_collective` `:4392`, axes `("dp",)`, participants `self.dp` |
| `comm["transformer_moe"]` | `CommSpec` | `:4405-4412` |
| `comm["embedding"]` | `CommSpec` | `:4413-4420` |
| `comm["softmax"]` | `CommSpec` | `:4421-4428` |
| `comm["cross_layer"]` | `CommSpec` | `:4429-4435` — kind `PIPELINE`, axes `("pp",)`, participants 2 |
| `comm["transformer_dense_ep_sync"]` | `CommSpec` | `:4437-4444`, bytes `:4599-4608` |
| `comm["transformer_moe_ep_sync"]` | `CommSpec` | `:4445-4452`, bytes `:4609-4620` |
| `comm["zero2_{embedding,transformer,softmax}_gather"]` | `CommSpec` | `:4453-4479`, `ga_required_every_cycle=False` |
| `comm["zero3_{embedding,transformer,softmax}_gather"]` | `CommSpec` | `:4480-4507`, `ga_required_every_cycle=True`; `zero3_transformer_gather` also `tp_shard=True` `:4496` |
| `blocks.dense` | `BlockTemplate` | `_build_transformer_template(transformer_timings, use_moe_layer=False, ...)` — `:4961` |
| `blocks.moe` | `Optional[BlockTemplate]` | `:4972` when `has_moe_layers` |
| `blocks.dense_comm` | `CommSpecTable` | `CommSpecTable.from_block_template(blocks.dense)` — i.e. the `use_moe_layer=False` `transformer_comm_metadata` accumulator (**amendment 2026-07-26**) |
| `blocks.moe_comm` | `Optional[CommSpecTable]` | `CommSpecTable.from_block_template(blocks.moe)` — the `use_moe_layer=True` accumulator, a **separate** table (**amendment 2026-07-26**) |
| (block comm keys) | `CommSpec` | `_register_specs` `:4691-4723`; per-op rules `_make_comm_specs` `:4775-4871`; MoE `_make_moe_comm_specs` `:578-620`; EP dense sync `_make_ep_dense_sync_specs` `:4725-4749`. **These land on `blocks.dense_comm` / `blocks.moe_comm`, never on `WorkloadSpec.comm`.** |
| `overlap.by_axis` | `Mapping` | `time_calc.tp_overlap` / `.tp_sp_overlap` / `.cp_overlap` (today `llm_execution.py:662-664`, `:783-785`, `:814-816`) |
| `overlap.parallelism_mode` | `ParallelismMode` | `self.get_parallelism_mode()` — `:4641` |
| `layout` | `RankLayout` | `RankLayout.from_network_layout(...)` — `llm_execution.py:132` (moves into the L0 constructor call) |
| `interconnect` | `Mapping` | `_build_interconnect_params()` — `:4510-4517`, returned at `:5116` |
| `granularity_hint` | `Granularity` | `ExecutionMode` → `Granularity` map, in the dispatcher |

New signature:

```python
# train_timing.py
def _prepare_execution_graphs(self, ...) -> Tuple[
    WorkloadSpec,             # was ScheduleInputs (final cycle)
    Optional[WorkloadSpec],   # was ScheduleInputs (nonfinal grad-accum cycle)
    BlockTemplates,           # was TransformerBlockSpec (degrees now live on the spec)
]:
```

`interconnect_params` moves onto the spec, so the 4-tuple becomes a 3-tuple.
`TransformerBlockSpec` (`block_program.py:94-112`) is deleted: `tp/cp/ep/include_backward` are
already on `WorkloadSpec.degrees` / `WorkloadSpec.run`.

### 1.8 L0 invariants

| id | Invariant | Verified by |
|---|---|---|
| **W1** | No field of `WorkloadSpec` is read with `getattr(obj, name, default)` or `dict.get(k, fallback)` anywhere in `program/`. A malformed input raises `WorkloadError` at construction. | new `tests/test_workload_spec.py`: constructing from an object with no `micro_batches` raises (today: silently `mb=0`); grep gate in the same test asserting zero `getattr(` in `program/workload.py` |
| **W2** | `durations` is the only mutable member; `revision` strictly increases; a `FrozenDurations` never changes after `snapshot()`. | `tests/test_workload_spec.py::test_writeback_revision_and_snapshot_isolation` |
| **W3** | Every `CommKey` referenced by any policy exists in `CommSpecTable`, or `WorkloadError` names it. | `tests/test_workload_spec.py`; T1 byte histograms by kind/axis (a dropped key changes the histogram) |
| **W4** | `Program.meta.misc["duration_revision"]` equals the revision the program was built from. | T3 (contract tier) |
| **W5** | *(new, 2026-07-26)* A block-template comm key is resolved **only** through `WorkloadSpec.block_comm(layer)`; the pipeline-level and block namespaces never shadow each other with different content (`WorkloadError` at construction if they do); the dense and MoE block tables may disagree freely. | `tests/test_placement.py::test_b1_block_expander_resolves_comm_through_the_layer_template`, `::test_b1_pipeline_and_block_comm_namespaces_may_not_shadow`, `tests/test_policies.py::test_b1_block_comm_tables_are_per_template`, `::test_b1_moe_routing_mode_is_found_on_the_block_tables`; T1 byte histograms on the `moe:ep2` rows |

### 1.9 What L0 deletes

* `program/schedule.py:108-143` — `ScheduleInputs`.
* `program/schedule.py:151-296` — `ScheduleSpec` **in full**, including `from_pipeline_graph`
  (`:178-221`, the mb=0 hazard), `time` (`:224`), `stage_for_layer` (`:229`), `is_moe_layer`
  (`:240`), `is_nonfinal_grad_accum_cycle` (`:248`), `_dp_comm_required_every_cycle` (`:251`),
  `should_emit_dp_comm` (`:255`, → §2.5), `recompute_enabled` (`:270`, → §2.7), `block_prefix`
  (`:280`), `par_degree` (`:284`), `stage_min_layer` (`:287`).
* `program/block.py:94-112`-equivalent `TransformerBlockSpec` in `block_program.py:94-112`.
* `llm_execution.py:392-395`, `:606-612`, `:742-746` — three copies of
  `include_backward` / `include_optimizer` / `effective_dp`; `_run_type` (`:370`) and
  `_retime_dp_count` (`:373-378`).
* `program/block.py:75-116` `CommMeta.from_legacy` / `comm_metadata_from_legacy` — replaced by
  `CommSpec.from_legacy` / `CommSpecTable.from_legacy` in `workload.py` (the raw-dict shape at the
  train_timing seam is kept; only the destination type changes).

---

## 2. L1 — `program/work.py` + `program/policies/`

### 2.1 WorkItem

```python
# program/work.py
class WorkKind(Enum):
    EMBEDDING = auto()
    LAYER     = auto()
    RECOMPUTE = auto()      # a forward-direction rematerialization inside the backward pass
    SOFTMAX   = auto()
    OPTIMIZER = auto()


class Direction(Enum):
    FORWARD  = auto()
    BACKWARD = auto()


@dataclass(frozen=True, order=True)
class WorkItem:
    """WHAT work exists. No order, no device, no duration.

    Frozen + ordered, so a WorkItem IS its own reference (`WorkRef = WorkItem`)
    and can key dicts / sort deterministically. `order` uses field order, which
    is the tie-break of last resort and must never be load-bearing.
    """
    kind: WorkKind
    direction: Direction
    microbatch: Optional[MicroBatch] = None   # None only for OPTIMIZER
    layer: Optional[LayerId] = None           # None for EMBEDDING/SOFTMAX/OPTIMIZER
    stage: Optional[StageId] = None           # SET ONLY for OPTIMIZER, whose identity is
                                              # per-stage; None otherwise (L2 derives placement)


WorkRef = WorkItem


@dataclass(frozen=True)
class WorkSet:
    """The complete, ordered-by-`WorkItem` set of work for one workload."""
    items: Tuple[WorkItem, ...]

    def get(self, kind, direction, *, microbatch=None, layer=None, stage=None) -> Optional[WorkItem]: ...
    def require(self, ...) -> WorkItem: ...          # raises WorkError, never KeyError
    def layers(self, direction: Direction, microbatch: MicroBatch) -> Tuple[WorkItem, ...]: ...
    def __contains__(self, item: WorkItem) -> bool: ...


def enumerate_work(fw: FrozenWorkload, recompute: "RecomputePolicy") -> WorkSet:
    """The complete work enumeration. Replaces the node-creation half of
    build_pipeline_events (schedule.py:579-776). Pure, order-free:

      for b in range(mb):
        EMBEDDING/FORWARD(b);  LAYER/FORWARD(b,l) for l;  SOFTMAX/FORWARD(b)
        if include_backward:
          SOFTMAX/BACKWARD(b); LAYER/BACKWARD(b,l) for l; EMBEDDING/BACKWARD(b)
          RECOMPUTE/FORWARD(b,l) for each l where recompute.materializes(...)
      if include_backward and include_optimizer and durations['optimizer'] > 0:
        OPTIMIZER/BACKWARD(stage=s) for s in range(pp)   # one per stage — Class D 10b
    """
```

Derived facts (`is_moe_layer`, block template, duration key, mem_kind) are **functions of the
WorkItem + FrozenWorkload**, never fields — this is what kills `FINE_EXPANDER_COPY_ATTRS` /
`OVERLAP_NODE_COPY_ATTRS` (`pipeline_fine.py:172-203`) and, with them, bug **A3**
(`is_moe_layer` dropped on an overlap head split): a split op still refers to the same `WorkItem`,
so no attribute can be lost in a copy.

```python
def duration_key(item: WorkItem, fw: FrozenWorkload) -> str: ...
def mem_kind(item: WorkItem) -> Optional["MemKind"]: ...
def is_moe(item: WorkItem, fw: FrozenWorkload) -> bool: ...
```

### 2.2 SyncRequirement and the typed attach mode

```python
class AttachMode(Enum):
    """CLOSED vocabulary. Replaces attach_parallel_edge's two boolean escape
    hatches (schedule.py:553-567). Completeness proof: §2.3."""
    BEFORE      = auto()
    AFTER       = auto()
    PARALLEL_TO = auto()
    OVERLAP_WITH = auto()      # required by the OverlapPolicy (§2.8), not by the ZeRO lattice


class DepClass(Enum):
    """The four dependency classes of §4.3. Also the `via` vocabulary of
    PARALLEL_TO: it is exactly the successor-edge classes the requirement inherits."""
    DATA_FLOW = auto()   # tensor producer -> consumer (same-device dep OR TransferOp)
    SCHEDULE  = auto()   # device serialization implied by the SchedulePolicy
    SYNC      = auto()   # another SyncRequirement's attach


VIA_DATA_FLOW     = frozenset({DepClass.DATA_FLOW})
VIA_NON_DATA_FLOW = frozenset({DepClass.SCHEDULE, DepClass.SYNC})
VIA_ALL           = frozenset(DepClass)


class SyncSpread(Enum):
    """How many instances of the requirement exist inside one pipeline stage.
    NAMED POLICY for BUG_LEDGER A2 and B/10d — see §7."""
    STAGE            = auto()   # one instance; the stage IS the device (COARSE/BLOCK granularity)
    CLUSTER_RANK_0   = auto()   # one instance, on cluster rank 0 of the stage  [legacy default]
    PER_CLUSTER_RANK = auto()   # one instance per cluster rank                 [A2's fix; ZeRO-3 tp_shard]


class ByteSplit(Enum):
    WHOLE             = auto()  # each instance carries the full size_bytes  [Class B item 1]
    CEIL_DIV_CLUSTER  = auto()  # ceil(total / instances)      [pipeline_fine.py:645; Class D 10c]


@dataclass(frozen=True)
class ByteSource:
    key: CommKey
    split: ByteSplit = ByteSplit.WHOLE

    def bytes_for(self, fw: FrozenWorkload, instances: int) -> float:
        """`instances` = the number of copies the caller is about to materialize,
        i.e. `Placement.cluster_size()` (§4.3 R2, §4.4). RAISES on instances < 1."""
```

> **AMENDMENT 2026-07-26 (B3): `CEIL_DIV_CLUSTER` divides by `instances`, not by
> `fw.spec.cluster_size()`.** The two differ exactly where it matters: `Placement.cluster_size()`
> is **1** at COARSE (the stage IS the device, §3.1) while `fw.spec.cluster_size()` is always
> `tp*cp*ep`. Legacy COARSE uses the **raw** cross-layer byte count
> (`pipeline_coarse.py:222,238` — `int(event.comm_size_bytes)`) and legacy FINE divides it
> (`pipeline_fine.py:645` — `ceil(total / par_degree)`). Ignoring `instances` returned `2048.0`
> at `tp=2` where COARSE wants `4096.0`, i.e. wrong `cross_layer` bytes on every
> coarse/hybrid/hierarchical row with `cluster_size > 1`. The unit test hid it by passing
> `instances == cluster_size`; it now passes a differing value.

```python


class SyncPhase(Enum):
    FWD_ENTRY = auto()
    FWD       = auto()
    BWD_ENTRY = auto()
    BWD       = auto()
    GRAD      = auto()


@dataclass(frozen=True, order=True)
class SyncKey:
    """Identity of one requirement instance. Unique by construction."""
    comm_key: CommKey
    phase: SyncPhase
    microbatch: Optional[MicroBatch] = None
    layer: Optional[LayerId] = None


SyncAnchor = Union[WorkItem, SyncKey]


@dataclass(frozen=True)
class SyncRequirement:
    key: SyncKey

    # -- what --------------------------------------------------------------
    bytes: ByteSource
    kind: CollectiveType                 # from CommSpecTable[bytes.key].kind — NEVER sniffed
    axes: Tuple[AxisName, ...]           # from CommSpecTable[bytes.key].axes  — §3.3
    participants: int                    # ANALYTICAL count (§1.2 note)

    # -- where -------------------------------------------------------------
    place_on: WorkItem                   # the work whose device hosts this collective
    spread: SyncSpread = SyncSpread.CLUSTER_RANK_0

    # -- when --------------------------------------------------------------
    mode: AttachMode
    anchors: Tuple[SyncAnchor, ...]      # >=1; PARALLEL_TO may repeat at several anchors (§2.3 S6/S14)
    via: FrozenSet[DepClass] = VIA_ALL   # PARALLEL_TO only; ignored by BEFORE/AFTER/OVERLAP_WITH

    # -- how ---------------------------------------------------------------
    overlap: Optional["OverlapDecl"] = None

    # -- AMENDMENT 2026-07-26 (B2) -----------------------------------------
    @property
    def is_dp(self) -> bool:
        """`axes == (DP_AXIS,)`. A dp collective has NO GroupKey: its wire members
        are stamped at emission over pre-dp device ids (`ir.py:92-99`), so the
        builder materializes `CollectiveOp(group=None, is_dp=True)`."""

    def __post_init__(self) -> None:
        """Raises SyncError when:
        * anchors is empty;
        * mode is BEFORE/AFTER/OVERLAP_WITH and len(anchors) != 1;
        * mode is not PARALLEL_TO and via != VIA_ALL (a silent-misuse guard);
        * kind is CollectiveType.PIPELINE (a p2p transfer is not a SyncRequirement);
        * `DP_AXIS in axes` and `axes != (DP_AXIS,)` — dp is replicated at
          emission and can never be one axis of a device-layout communicator
          (amendment 2026-07-26)."""
```

`DP_AXIS = "dp"` lives in `program/types.py` (which imports nothing from `program.*`), so L1
(`SyncRequirement.is_dp`) and L2 (`Placement`, `CommunicatorFactory`) agree on the name without an
import edge and without either dispatching on a bare string literal (rule §8.2).

**Attach-mode semantics (normative).** `deps(x)` is the dep set of `x` in the *snapshot* defined in
§4.4; `succ(x, via)` are `x`'s successors reached by an edge whose `DepClass` is in `via`.

| Mode | Effect |
|---|---|
| `BEFORE(a)` | `deps(req) := deps(a)`; `deps(a) += req` — the requirement is spliced *in front of* `a`. Degenerate when `a` is a root (then `req` becomes the root). |
| `AFTER(a)` | `deps(req) := {a}` — the requirement is a sink hanging off `a`. `a` may be a `WorkItem` **or** a `SyncKey` (a requirement chained after another requirement). |
| `PARALLEL_TO(a, via)` | `deps(req) += deps(a)`; for each `s in succ(a, via)`: `deps(s) += req`. The requirement runs *concurrently with* `a`. Applied once per anchor in `anchors`; idempotent (matching `attach_parallel_edge`'s `not in` guards, `schedule.py:556,566`). |
| `OVERLAP_WITH(a)` | Defined by `OverlapDecl` (§2.8); realized at L4 by splitting `a` and re-parenting, not by adding a dep. |

### 2.3 Completeness proof: every existing call site maps to exactly one mode

The lattice under replacement is `schedule.py:778-1046` plus its deferred attaches at `:928-934`.
It uses exactly **12 literal comm keys** and performs exactly **16 create-and-attach operations**.
Every one is listed. `attach_parallel_edge(t, e)` with both flags falsy ⇒ `via = VIA_ALL`;
`skip_non_comm_children=True` keeps only children with `comm_type == PIPELINE` ⇒ `via = VIA_DATA_FLOW`
(cross-layer edges are *always* `CollectiveType.PIPELINE`: same-stage at `:629/:640/:649`, cross-stage
via `comm_metadata['cross_layer']['type'] = PIPELINE` at `train_timing.py:4431`);
`skip_comm_children=True` keeps only non-PIPELINE children ⇒ `via = VIA_NON_DATA_FLOW`.

| # | line | comm key | legacy attach expression | mode | anchors | via | spread |
|---|---|---|---|---|---|---|---|
| S1 | `:784-791` | `zero3_embedding_gather` | `gather.add_child(embedding_node[0])`, becomes `root_forward_entry` | `BEFORE` | `EMBEDDING/FWD(0)` | — | `CLUSTER_RANK_0` |
| S2 | `:807` → `:931` | `embedding` | `embedding_node_b[b].add_child(edge)` | `AFTER` | `EMBEDDING/BWD(b)` | — | `CLUSTER_RANK_0` |
| S3 | `:812` → `:818` | `zero2_embedding_gather` | `embedding_edge.add_child(gather)` | `AFTER` | `SyncKey(embedding, GRAD, b)` | — | `CLUSTER_RANK_0` |
| S4 | `:828` → `:934` | `transformer_dense` / `transformer_moe` | `_bwd_exit_node(b,l).add_child(reducer)` | `AFTER` | `LAYER/BWD(b,l)` | — | `CLUSTER_RANK_0` |
| S5 | `:840` → `:846` | `zero2_transformer_gather` | `reducer.add_child(gather)` | `AFTER` | `SyncKey(transformer_*, GRAD, b, l)` | — | `CLUSTER_RANK_0` |
| S6 | `:857` → `:890` | `zero3_embedding_gather` | `attach_parallel_edge(host, e, skip_comm_children=True)` at every fwd stage boundary | `PARALLEL_TO` | `LAYER/FWD(b, l-1)` for each cross-device `l` | `VIA_NON_DATA_FLOW` | `CLUSTER_RANK_0` |
| S7 | `:864` → `:871` | `zero3_transformer_gather` | `attach_parallel_edge(embedding_node[b], e)` | `PARALLEL_TO` | `EMBEDDING/FWD(b)` | `VIA_ALL` | `PER_CLUSTER_RANK` |
| S8 | `:875` → `:883` / `:888` | `zero3_transformer_gather` | same-device: `attach_parallel_edge(host, e)`; cross-device: `..., skip_non_comm_children=True` | `PARALLEL_TO` | `LAYER/FWD(b, l-1)` | `VIA_ALL` / `VIA_DATA_FLOW` | `PER_CLUSTER_RANK` |
| S9 | `:896` → `:929` | `softmax` | `softmax_node_b[b].add_child(edge)` | `AFTER` | `SOFTMAX/BWD(b)` | — | `CLUSTER_RANK_0` |
| S10 | `:906` → `:912` | `zero2_softmax_gather` | `softmax_edge.add_child(gather)` | `AFTER` | `SyncKey(softmax, GRAD, b)` | — | `CLUSTER_RANK_0` |
| S11 | `:920` → `:926` | `zero3_softmax_gather` | `attach_parallel_edge(transformer_nodes[b][L-1], e)` | `PARALLEL_TO` | `LAYER/FWD(b, L-1)` | `VIA_ALL` | `CLUSTER_RANK_0` |
| S12 | `:947` → `:953` | `transformer_dense_ep_sync` / `transformer_moe_ep_sync` | `_bwd_exit_node(b,l).add_child(ep_edge)` | `AFTER` | `LAYER/BWD(b,l)` | — | `CLUSTER_RANK_0` |
| S13 | `:989` → `:995` | `zero3_softmax_gather` | `attach_parallel_edge(softmax_node[-1], e)` | `PARALLEL_TO` | `SOFTMAX/FWD(mb-1)` | `VIA_ALL` | `CLUSTER_RANK_0` |
| S14 | `:1004` → `:1032` | `zero3_softmax_gather` | `attach_parallel_edge(host, e, skip_comm_children=True)` at every bwd cross-device boundary | `PARALLEL_TO` | `SOFTMAX/BWD(b)` or `LAYER/BWD(b, l+1)` | `VIA_NON_DATA_FLOW` | `CLUSTER_RANK_0` |
| S15 | `:1020` → `:1028` / `:1030` | `zero3_transformer_gather` | same-device / cross-device as S8 | `PARALLEL_TO` | `SOFTMAX/BWD(b)` if `l == L-1` else `LAYER/BWD(b, l+1)` | `VIA_ALL` / `VIA_DATA_FLOW` | `PER_CLUSTER_RANK` |
| S16 | `:1040` → `:1046` | `zero3_embedding_gather` | `attach_parallel_edge(_bwd_exit_node(b,0), e)` | `PARALLEL_TO` | `LAYER/BWD(b, 0)` | `VIA_ALL` | `CLUSTER_RANK_0` |

**Mode census:** `BEFORE` 1 (S1) · `AFTER` 7 (S2,S3,S4,S5,S9,S10,S12) · `PARALLEL_TO/VIA_ALL` 6
(S7,S8ᵃ,S11,S13,S15ᵃ,S16) · `PARALLEL_TO/VIA_DATA_FLOW` 2 (S8ᵇ,S15ᵇ) ·
`PARALLEL_TO/VIA_NON_DATA_FLOW` 2 (S6,S14). Total 16 attach operations over 12 keys. **No call site
requires a mode outside the vocabulary, and no vocabulary member is unused by the lattice**
(`OVERLAP_WITH` is used only by the OverlapPolicy — stated honestly, not padded).

**Two consequences that must be implemented, not discovered:**

1. *Orphan requirements.* S6 and S14 are created once per microbatch but attached only inside the
   cross-device branch. With no cross-device boundary they are created and never attached — today
   that consumes a legacy op id and produces an unreachable object the lowering never collects.
   **`build()` must drop any requirement whose resolved anchor set is empty** and must not
   materialize an op for it. Divergence class: **C** (op-id numbering only; op multiset unchanged).
2. *Multi-anchor PARALLEL_TO.* S6/S14 attach the *same* requirement at *several* anchors. `anchors`
   is therefore a tuple, and the attach is idempotent per (requirement, anchor).

### 2.4 ShardingPolicy

```python
# program/policies/sharding.py

# -- AMENDMENT 2026-07-26 (B5) -------------------------------------------
class StagePartition(Protocol):
    """The ONE thing L1 needs from L2's placement. `stage_of(work: WorkItem)` is
    REMOVED from this protocol: it collided with `LayerAssignment.stage_of(layer)`
    (§4.1) on both name and arity, so no object satisfied both — and `Placement`
    consumes a `LayerAssignment` while `ShardingContext` consumes a
    `StagePartition`, from ONE stage partition. `same_stage` was the only member
    L1 ever called (`ShardingContext.same_stage`, `ZeRO3._via`)."""
    def same_stage(self, a: WorkItem, b: WorkItem) -> bool: ...


@dataclass(frozen=True)
class ContiguousStages:
    """Satisfies BOTH seams. LayerAssignment surface (§4.1): `stage_of(layer)`,
    `layers_of(stage)`, `min_layer(stage)`, `contiguous()`/`legacy()`.
    StagePartition surface: `same_stage`, built on `stage_of_work(work)` — the
    WorkItem rule, renamed on 2026-07-26 to free `stage_of` for §4.1's signature.
    P4 replaces it with `program.sched.policy.LayerAssignment`, same members."""
    stage_of_layer: Tuple[StageId, ...]
    num_stages: int


@dataclass(frozen=True)
class ShardingContext:
    fw: FrozenWorkload
    work: WorkSet
    grad_accum: "GradAccumPolicy"
    stages: StagePartition                                        # same_stage only
    def spec_for(self, key: CommKey) -> Optional[CommSpec]: ...   # None when absent (= disabled)


class ShardingPolicy(Protocol):
    name: str                    # "ddp" | "zero1" | "zero2" | "zero3" | ...

    def requirements(self, work: WorkItem, ctx: ShardingContext) -> Sequence[SyncRequirement]:
        """All sync a single WorkItem induces. Called once per WorkItem.
        MUST be pure and MUST NOT depend on call order."""

    def workload_requirements(self, ctx: ShardingContext) -> Sequence[SyncRequirement]:
        """Sync that belongs to the workload rather than to one WorkItem —
        exactly the two entry gathers, S1 and S13. Called once."""
```

Concrete policies (constructors only; bodies are ~30 LOC each):

```python
@dataclass(frozen=True)
class DDP(ShardingPolicy):
    name: str = "ddp"
    # BACKWARD LAYER  -> AFTER  (transformer_dense|transformer_moe)     [S4]
    # BACKWARD EMBED  -> AFTER  (embedding)                            [S2]
    # BACKWARD SOFTMAX-> AFTER  (softmax)                              [S9]
    # kind comes from CommSpecTable (ALL_REDUCE at zero<2, REDUCE_SCATTER at zero>=2)

@dataclass(frozen=True)
class ZeRO1(DDP):
    name: str = "zero1"
    # Structurally identical to DDP: ZeRO-1 shards optimizer state only, which
    # changes memory, not the collective set (train_timing.py:5128-5132).

@dataclass(frozen=True)
class ZeRO2(DDP):
    name: str = "zero2"
    # DDP's reducers PLUS one AFTER(reducer) param all-gather per reducer:
    #   zero2_embedding_gather   [S3]
    #   zero2_transformer_gather [S5]
    #   zero2_softmax_gather     [S10]

@dataclass(frozen=True)
class ZeRO3(ZeRO2):
    name: str = "zero3"
    prefetch_depth: int = 1
    """THE +/-1 at schedule.py:873 (`transformer_nodes[b][layer_idx-1]`) and
    :1018 (`_bwd_exit_node(b, layer_idx+1)`) IS the prefetch depth. Named here
    so the ext_zero_policy exercise is a constructor argument, not a new lattice."""
    # per-layer parameter gathers:
    #   forward : PARALLEL_TO(LAYER/FWD(b, l - prefetch_depth), via=...)   [S7,S8]
    #   backward: PARALLEL_TO(LAYER/BWD(b, l + prefetch_depth), via=...)   [S15]
    #   embedding/softmax boundary gathers                                  [S6,S11,S13,S14,S16]
    #   entry gathers via workload_requirements()                           [S1,S13]

def sharding_policy_for(run: RunPolicy, degrees: ParallelDegrees) -> ShardingPolicy:
    """{0: DDP, 1: ZeRO1, 2: ZeRO2, 3: ZeRO3}[run.zero_stage]. dp <= 1 selects
    a NullSharding that emits nothing (today: `if spec.dp > 1` at schedule.py:797
    plus the dp<=1 early return in should_emit_dp_comm at :257)."""
```

**`via` selection is a rule, not a per-site literal.** Inside `ZeRO3.requirements`:

```python
via = VIA_ALL if placement.same_stage(host, target) else VIA_DATA_FLOW
```
and the companion boundary gather (S6/S14) is emitted with `VIA_NON_DATA_FLOW` at the same anchors.
The "legacy rule" comment at `schedule.py:885-892` becomes the docstring of one branch, once.

### 2.5 GradAccumPolicy — one implementation of a predicate that exists twice today

```python
# program/policies/gradaccum.py
@dataclass(frozen=True)
class GradAccumPolicy:
    """Verbatim semantics of Graph._should_emit_dp_comm. ONE implementation:
    schedule.py:255-268 and its re-derivation at schedule.py:937
    (`apply_ep_all_mbs = ... != "last_mb" or zero_stage >= 3`) are the same
    predicate written twice (audit A6)."""

    name: str = "legacy"
    dp: int = 1
    zero_stage: int = 0
    cycle: GradAccumCycle = GradAccumCycle.FINAL
    mode: DpMicrobatchMode = DpMicrobatchMode.EVERY_MB

    def emits(self, spec: CommSpec, microbatch: Optional[MicroBatch]) -> bool:
        if self.dp <= 1:
            return False
        if self.cycle is GradAccumCycle.NONFINAL:
            return spec.ga_required_every_cycle
        if spec.ga_required_every_cycle:
            return True
        if self.mode is not DpMicrobatchMode.LAST_MB or self.zero_stage >= 3:
            return True
        # Class B item 10h: backward walks microbatches in reverse, so b == 0
        # IS the last microbatch. Semantics correct, legacy comment confusing.
        return microbatch == 0
```

The EP-sync gate (`schedule.py:936-938`) becomes `grad_accum.emits(ep_spec, b)` on the same object.
The three-way split of EP-sync policy (attach point `train_timing.py:4762-4768`, gate
`schedule.py:936-953`, bytes `train_timing.py:4599-4620`) collapses to: bytes stay upstream (§0.1),
gate + attach both live in the routing policy (§2.6).

### 2.6 MoE routing policy — absorbs the two two-value if-chains

Replaces `block_program.py:223-228` (`_moe_hot_rank`) and `:230-235` (`_moe_parallel_token`), which
are byte-identical to legacy and duplicated per routing mode.

```python
# program/policies/routing.py
class MoERoutingPolicy(Protocol):
    name: str
    def routing_axes(self) -> Tuple[AxisName, ...]:
        """The axes the routing collective spans. THIS is the declared source of
        the composite ("tp","ep") communicator that legacy_lowering.py:243-248
        infers from `participants == tp_size * ep_size`."""
    def hot_coords(self, coords: Coords) -> Coords: ...
    def join_token(self, coords: Coords) -> Tuple[Any, ...]: ...


@dataclass(frozen=True)
class AxisRouting(MoERoutingPolicy):
    """ONE implementation covering both existing modes and any future one.

    hot_coords  = coords with every routing axis zeroed
    join_token  = the coords on the axes NOT routed over, in canonical order
    """
    name: str
    axes: Tuple[AxisName, ...]
    cluster_axes: Tuple[AxisName, ...] = ("tp", "cp", "ep")

    def hot_coords(self, coords):
        return {**coords, **{a: 0 for a in self.axes}}

    def join_token(self, coords):
        rest = tuple(a for a in self.cluster_axes if a not in self.axes)
        return (self.name,) + tuple(coords[a] for a in rest)


EP_ROUTING    = AxisRouting(name="ep",    axes=("ep",))          # training
TP_EP_ROUTING = AxisRouting(name="tp_ep", axes=("tp", "ep"))     # inference

ROUTING_MODES = {"ep": EP_ROUTING, "tp_ep": TP_EP_ROUTING}       # keyed by CommSpec.moe_routing_mode
```

Equivalence to the code being deleted, checked term by term:

| legacy | `AxisRouting` |
|---|---|
| `_moe_hot_rank("ep")   = _rank_id(tp, cp, 0)` | `hot_coords` zeroes `ep` ✓ |
| `_moe_hot_rank("tp_ep")= _rank_id(0, cp, 0)` | `hot_coords` zeroes `tp`,`ep` ✓ |
| `_moe_parallel_token("ep")    = ("ep", cp, tp)` | `("ep", tp, cp)` — same partition of ranks; the tuple is a **dict key**, so component order is inert ✓ |
| `_moe_parallel_token("tp_ep") = ("tp_ep", cp)` | `("tp_ep", cp)` ✓ |

`MoERoutingPolicy` also owns the EP grad-sync requirement (S12) and the `moe_component` dispatch
(`base_all_to_all` / `residual_p2p`, `block_program.py:305-313`), so a third routing mode is one
`AxisRouting(...)` row plus, at most, a new component name.

### 2.7 RecomputePolicy

```python
# program/policies/recompute.py
class RecomputePolicy(Protocol):
    name: str
    def materializes(self, layer: LayerId, microbatch: MicroBatch, fw: FrozenWorkload) -> bool: ...


@dataclass(frozen=True)
class NoRecompute(RecomputePolicy):
    name: str = "none"
    def materializes(self, *a, **k) -> bool: return False


@dataclass(frozen=True)
class FullRecompute(RecomputePolicy):
    """Every layer of every microbatch materializes a RECOMPUTE WorkItem
    (schedule.py:716-732)."""
    name: str = "full"
    def materializes(self, *a, **k) -> bool: return True


def recompute_policy_for(fw: FrozenWorkload, granularity: "Granularity") -> RecomputePolicy:
    """PRESERVES the existing predicate (schedule.py:270-278):

        include_backward AND full_recomputation AND (flattened_mode OR pipeline_style_recompute)

    but as an EXPLICIT dispatcher-time SELECTION rather than a `misc["flattened_mode"]`
    read inside the builder. Audit C3 ("the enumerated schedule's structure
    depends on which backend consumes it") is thereby made visible: the coupling
    is now one documented line in a factory, and a caller may override it."""
```

### 2.8 OverlapPolicy — overlap DECLARED on a requirement

Replaces both competing implementations: the proto rewrite `transforms.py:169-299`
(`_split_tp_node_fine` / `_split_cp_edge_fine`, the only one production uses) and the
Program-level `transforms.py:430-556` (`apply_tp_overlap`, no production caller).

```python
# program/policies/overlap.py
class OverlapAnchor(Enum):
    PRODUCER = auto()   # split the COMPUTE that produces the collective's input   (tp / tp_sp)
    CONSUMER = auto()   # split the COLLECTIVE by bytes; the consumer runs on the blocking part (cp)


@dataclass(frozen=True)
class OverlapDecl:
    fraction: float                       # in (0, 1]; >= 1.0 means "fully hoisted"
    anchor: OverlapAnchor
    blocking_consumer: Optional[str] = None
    """For CONSUMER anchoring: the BlockTemplate entry name that must wait for the
    blocking part. Today this is the substring test `"attention" in name.lower()`
    (transforms.py:224) against a TEMPLATE-SUPPLIED name; here it is DATA on the
    policy. Default "attention" reproduces today exactly."""

    def __post_init__(self) -> None:
        """Raises OverlapError when fraction <= 0, or when anchor is CONSUMER and
        blocking_consumer is None."""


class OverlapPolicy(Protocol):
    name: str
    def declare(self, spec: CommSpec, fw: FrozenWorkload) -> Optional[OverlapDecl]:
        """Called once per block-template comm key. Returning None means no overlap."""


@dataclass(frozen=True)
class AxisFractionOverlap(OverlapPolicy):
    """The only production policy. Reads OverlapSpec.by_axis; anchors tp/tp_sp
    collectives on PRODUCER and cp collectives on CONSUMER, matching
    transforms.py:326-374 — but as a table, not two near-duplicate blocks."""
    name: str = "axis_fraction"
    anchor_by_axis: Mapping[AxisName, OverlapAnchor] = MappingProxyType(
        {"tp": OverlapAnchor.PRODUCER, "cp": OverlapAnchor.CONSUMER}
    )
```

L4 realizes the declaration (§4.5). The `head` / `_block` / `_ovlp` op-id-reuse quirks
(`transforms.py:200`, `:255`, `:267`) are **deleted**: program order is computed (§4.6), so
duplicate ids cannot exist (audit item 10e, Class C).

### 2.9 L1 invariants

| id | Invariant | Verified by |
|---|---|---|
| **K1** | Every `SyncRequirement.anchors` entry resolves to a member of the `WorkSet` or to another emitted `SyncKey`. Unresolved ⇒ `SyncError`, never a silent drop — except the *declared* empty-anchor drop of §2.3 note 1, which is logged. | `tests/test_policies.py`; T1 per-group collective counts |
| **K2** | `AttachMode` is exhaustively handled: `build()`'s dispatch ends in `assert_never(mode)`. | mypy/`typing.assert_never` + `tests/test_policies.py::test_attach_mode_exhaustive` |
| **K3** | `ShardingPolicy.requirements` is pure and order-independent: shuffling the WorkSet iteration order yields the same requirement set. | `tests/test_policies.py::test_requirements_order_independent` |
| **K4** | The grad-accum predicate has exactly one implementation. | grep gate in `tests/test_policies.py` asserting `dp_microbatch_mode`/`last_mb` appears in exactly one module |
| **K5** | Every `SyncRequirement.kind` equals `CommSpecTable[req.bytes.key].kind` (no name-derived collective type). | `tests/test_policies.py`; T1 byte histogram by kind |
| **K6** | Every Class B row of §7 is reachable only through a named policy object; no literal `rank_tails[0]`, `int(base_bytes)`, `"ep"`/`"tp_ep"`, `±1` survives in an expansion loop. | code review + the grep gates above |
| **K7** | *(new, 2026-07-26)* `ByteSource.bytes_for` is a function of `instances`, and `instances` is what the caller materializes — `Placement.cluster_size()`, not `fw.spec.cluster_size()`. | `tests/test_policies.py::test_b3_byte_source_honors_instances_not_cluster_size`, `::test_byte_source_splits` (now passes `instances != cluster_size`) |
| **K8** | *(new, 2026-07-26)* `SyncRequirement.is_dp` is the single declaration that a collective has no communicator; `dp` never appears composited with a device axis. | `tests/test_policies.py::test_b2_dp_requirement_declares_itself_dp` |
| **K9** | *(new, 2026-07-26)* `StagePartition` and `LayerAssignment` are simultaneously satisfiable by one object. | `tests/test_policies.py::test_b5_stage_partition_protocol_is_satisfiable`, `tests/test_placement.py::test_b5_contiguous_stages_satisfies_placement_and_sharding_together` |

### 2.10 What L1 deletes

* `program/schedule.py:553-567` — `attach_parallel_edge` and both boolean escape hatches.
* `program/schedule.py:778-1046` — the entire 12-key DP/ZeRO/EP lattice (~270 LOC), incl. the
  collective-name sniffing at `:801-803`.
* `program/schedule.py:255-268` + the duplicate at `:937` — `should_emit_dp_comm`.
* `program/schedule.py:270-278` — `recompute_enabled`.
* `program/schedule.py:409`, `:445-447` — `CommEvent.zero3_offset_hint`; and its consumer
  `pipeline_fine.py:351-356` incl. the live `"bwd" in edge.name` fallback.
* `program/pipeline_fine.py:305-312` `_should_shard_zero3_transformer`; `:314-395`
  `_ensure_zero3_per_rank_edges` (incl. the dead `direction == "backward"` branch at `:336-338`,
  bug **A5**'s dead store at `:620-629`, and the un-divided `int(base_bytes)` at `:360` → now
  `ByteSplit.WHOLE`).
* `program/pipeline_fine.py:172-203` — `FINE_EXPANDER_COPY_ATTRS`, `OVERLAP_NODE_COPY_ATTRS`,
  `OVERLAP_EDGE_COPY_ATTRS` (the divergence that IS bug **A3**).
* `program/block_program.py:223-235` — `_moe_hot_rank`, `_moe_parallel_token`.
* `program/transforms.py:110-299` (proto overlap) and `:430-556` (`apply_tp_overlap`) — one
  declaration + one realizer replace both.

---

## 3. L2 — `program/placement.py` + `program/block.py`

### 3.1 Granularity is a parameter

```python
# program/placement.py
class Granularity(Enum):
    COARSE = auto()   # one op per WorkItem, on the stage device        (was pipeline_coarse.py)
    FINE   = auto()   # WorkItem -> BlockTemplate chain x cluster_size  (was pipeline_fine.py)
    BLOCK  = auto()   # one layer expanded over the (tp,cp,ep) sublayout only (was block_program.py)
```

| Granularity | device space | `WorkItem` maps to | `SyncSpread` default |
|---|---|---|---|
| `COARSE` | `layout.subset(("pp","dp"))`; device == stage | 1 `ComputeOp` | `STAGE` |
| `FINE` | full `layout`; device == `linearize(cluster_coords(rank, stage))` | `cluster_size` chains of `len(template.entries)` computes + their comm keys | `CLUSTER_RANK_0` (`PER_CLUSTER_RANK` iff `CommSpec.tp_shard`) |
| `BLOCK` | `layout.subset(("tp","cp","ep"))`; device == cluster rank | 1 chain per cluster rank, one layer only, no pipeline | `STAGE` |

### 3.2 Placement

```python
@dataclass(frozen=True)
class DeviceCoord:
    """A device as coordinates. `DeviceId` is `layout.linearize(coords)`."""
    coords: Coords
    def with_(self, **axes: int) -> "DeviceCoord": ...


class Placement:
    """WorkItem -> DeviceCoord(s). THE single placement authority: no consumer
    downstream re-derives a stage from an hw_id or a device from a name."""

    def __init__(
        self,
        fw: FrozenWorkload,
        granularity: Granularity,
        layers: "LayerAssignment",
    ) -> None: ...

    # -- device space -----------------------------------------------------
    @property
    def layout(self) -> RankLayout:
        """The layout DEVICE IDS LIVE IN for this granularity (already subset)."""
    @property
    def group_layout(self) -> RankLayout:
        """AMENDMENT 2026-07-26 (B2): `layout` WITHOUT `dp` — the space
        COMMUNICATOR MEMBERS live in, and what backs `communicators`.
        `RankLayout` always orders axes canonically with `dp` LAST and assigns
        row-major strides, so removing `dp` preserves every stride: device ids
        are bit-identical to `layout`'s. No communicator can span `dp`."""
    def devices(self) -> Tuple[DeviceId, ...]:
        """All devices, in `linearize` order. Never varies the dp coordinate."""
    def cluster_size(self) -> int: ...

    # -- placement --------------------------------------------------------
    def stage_of(self, work: WorkItem) -> StageId:
        """EMBEDDING -> 0; SOFTMAX -> pp-1; LAYER/RECOMPUTE -> layers.stage_of(work.layer);
        OPTIMIZER -> work.stage. Verbatim placement rules of schedule.py:583,592,603,1054."""
    def devices_for(self, work: WorkItem) -> Tuple[DeviceId, ...]:
        """COARSE/BLOCK: one device. FINE: `cluster_size` devices in cluster-rank
        order. Softmax at FINE returns ONE device (cluster rank 0) — Class B 10d."""
    def devices_for_sync(self, req: SyncRequirement) -> Tuple[DeviceId, ...]:
        """Resolves `req.spread` against `req.place_on`:
           STAGE            -> (stage device,)
           CLUSTER_RANK_0   -> (devices_for(place_on)[0],)
           PER_CLUSTER_RANK -> cluster_devices(stage_of(place_on))   # amended 2026-07-26
        This REPLACES `local_hw_id` + `propagate_local_hw_ids` entirely: placement
        is DECLARED at L1 and RESOLVED here, never back-propagated from a parent.
        RAISES PlacementError if PER_CLUSTER_RANK yields < cluster_size devices."""
    def same_stage(self, a: WorkItem, b: WorkItem) -> bool: ...   # satisfies StagePartition (§2.4)
```

> **AMENDMENT 2026-07-26 (B4): `PER_CLUSTER_RANK` resolves against the STAGE, not against
> `devices_for(place_on)`.** The two differ whenever `place_on`'s kind is pinned to cluster rank 0 by
> the `PlacementPolicy` — and `LEGACY_PLACEMENT` pins `EMBEDDING` and `SOFTMAX` (Class B 10d). That
> is exactly ZeRO-3 row **S7**, whose `zero3_transformer_gather` carries `tp_shard=True` (therefore
> `SyncSpread.PER_CLUSTER_RANK`, `train_timing.py:4496`) and is placed on `EMBEDDING/FORWARD`, and
> row **S13**, placed on `SOFTMAX/BACKWARD`. Delegating to `devices_for` collapsed the requirement to
> ONE device (probed at `zero3, dp2, tp2, pp2`: S7 → `(0,)` with `cluster_size == 2`) while legacy
> `_ensure_zero3_per_rank_edges` receives `hw_ids` built for **every** `par_degree` rank
> (`pipeline_fine.py:471-480`). The SPREAD of a collective is a property of the collective, not of
> the placement of the work it hangs off. Invisible in the golden matrix only because every
> zero2/zero3 spec is `tp=cp=1` (see the §3.2 reachability note below), so this is new development
> with new goldens — but it must be right before the ZeRO-3 per-rank region is exercised.

> **The ZeRO-3 `±1 pp` hop is deleted, not ported.** `pipeline_fine._offset_stage_device`
> (`:290-303`) exists only to move a gather from the *host*'s device to the *target*'s device. With
> `SyncRequirement.place_on = <the target WorkItem>` (§2.2) the device is looked up directly and no
> offset arithmetic exists. The two are identical because the ±1 direction was exactly host→target.
> Reachability check: the per-rank ZeRO-3 machinery is **dead in the golden matrix**
> (`BUG_LEDGER.md` reframe: zero2/zero3 specs are built only at `dp=2, tp=1, cp=1`, so
> `cluster_size == 1` and `_should_shard_zero3_transformer` always returns False), so this region is
> new development with new goldens, not pinned behavior.

> **Placement checkpoint (P3, measure before landing).** For the *non*-per-rank ZeRO-2/3 gathers and
> the DP reducers, FINE today declares `local_hw_id = <stage>` and then `propagate_local_hw_ids`
> (`pipeline_fine.py:796-850`) overwrites it with the first placed parent's device — i.e. cluster
> rank 0. `SyncSpread.CLUSTER_RANK_0` is chosen as the default precisely to reproduce that. The P3
> gate must confirm T1-exact per-rank collective multisets on all ZeRO-2/3 golden rows before
> `propagate_local_hw_ids` is deleted.

### 3.3 CommunicatorFactory — groups CONSTRUCTED, never inferred

```python
@dataclass(frozen=True)
class CommunicatorFactory:
    """Replaces the participant-count inference at legacy_lowering.py:196-282
    (incl. the hand-written composite hack at :243-248) and the axis-group
    reconstruction at :135-181."""

    layout: RankLayout       # = Placement.group_layout: NO `dp` (amendment 2026-07-26)

    def members(self, axes: Sequence[AxisName], anchor: DeviceId) -> Tuple[DeviceId, ...]:
        """All devices sharing `anchor`'s coordinates on every axis NOT in `axes`,
        ranging over the full product of `axes`. Sorted ascending.

            base = layout.coords_of(anchor)
            for combo in itertools.product(*(range(layout.axis_sizes[a]) for a in axes)):
                yield layout.linearize({**base, **dict(zip(axes, combo))})

        Composite axes are ORDINARY input: ("tp","ep") is one call, not a special
        case. The ("tp","ep") communicator that legacy recognizes only when
        `participants == tp_size * ep_size` is now DECLARED by the MoE routing
        policy (§2.6) and flows through CommSpec.axes (§1.2)."""

    def group_for(self, axes: Sequence[AxisName], anchor: DeviceId) -> GroupKey:
        """GroupKey(axis="+".join(axes), members=self.members(axes, anchor)).
        The axis string is diagnostics; MEMBERSHIP is the identity."""

    def groups_for(
        self, req: SyncRequirement, placement: Placement
    ) -> Tuple[Optional[GroupKey], ...]:
        """One entry per instance device of `req` (§3.2 devices_for_sync):
        a GroupKey, or `None` when `req.is_dp`.  (amendment 2026-07-26)"""
```

Rules that fall out and must be asserted:

* an axis with `axis_sizes[a] <= 1` contributes a **singleton** group — the emitter's zero-duration
  `*_noop` COMP substitution (`et_emit.py:313-320`) still applies; Class B 10g is unchanged;
* `dp` groups are still stamped at emission (`GroupKey.members` are pre-DP devices,
  `ir.py:93-104`) — `CommunicatorFactory` never sees a dp index;
* the axis-name string is never dispatched on. `et_emit.py` remains axis-blind (0 axis literals).

> **AMENDMENT 2026-07-26 (B2): `dp` is not a communicator axis of a device layout — asking for one
> is a `GroupError`.** The bullet above ("`CommunicatorFactory` never sees a dp index") was a
> *statement of intent* with nothing enforcing it, and it was false in three ways at once:
>
> | granularity | `Placement.layout` | `devices()` | `members(("dp",), 0)` was | wrong because |
> |---|---|---|---|---|
> | COARSE | `("pp","dp")` | `(0, 1)` | `(0, 2)` | member 2 is not a device |
> | FINE | `("tp","cp","ep","pp","dp")` | `(0,1,2,3)` | `(0, 4)` | member 4 is not a device |
> | BLOCK | `("tp","cp","ep")` | `(0, 1)` | `(0,)` | `_spanned_axes` skipped the absent axis → singleton → `et_emit` substitutes a zero-duration `*_noop`, **deleting the reducer** |
>
> Three changes, all required (any one alone still leaves a silent failure mode):
>
> 1. `CommunicatorFactory` is built over `Placement.group_layout` (= `layout` minus `dp`), so no
>    reachable layout carries `dp` at all. As a bonus `partition()` — which V7 uses — now iterates
>    exactly the device set instead of `devices * dp`.
> 2. `groups.py::_spanned_axes` **raises `GroupError`** on `"dp"`. Without this, (1) turns the COARSE
>    and FINE cases into the BLOCK case: a silent singleton, i.e. a deleted collective.
> 3. dp requirements carry `group=None, is_dp=True` (`SyncRequirement.is_dp`, §2.2), which is what
>    `CollectiveOp` already models (`is_dp` / `label is None`, `ir.py:146-151`) and what emission
>    already expects.

### 3.4 Block expansion (`program/block.py`, extended)

```python
@dataclass(frozen=True)
class ExpandedChain:
    """One WorkItem, one device: the ordered ops and the identity of its ends."""
    device: DeviceId
    steps: Tuple["ChainStep", ...]     # ChainStep = ComputeStep | CommStep
    entry: int                          # index into steps
    exit: int


class BlockExpander:
    def __init__(self, fw: FrozenWorkload, placement: Placement,
                 overlap: OverlapPolicy, routing: Optional[MoERoutingPolicy]) -> None: ...

    def _specs_for(self, work: WorkItem) -> CommSpecTable:
        """AMENDMENT 2026-07-26 (B1): `fw.spec.block_comm(work.layer)` — the comm
        table of the template that expands this layer. Block comm keys are named
        PER TEMPLATE (§1.6), so the expander must never read a flat union: it
        would silently pick whichever template registered first and be 5.03x
        wrong on `ep_dense_sync_*` for the MoE golden rows. A WorkItem with no
        layer resolves no block keys and raises."""

    def expand(self, work: WorkItem) -> Tuple[ExpandedChain, ...]:
        """COARSE: one single-step chain (duration = durations[duration_key(work)]).
        FINE/BLOCK: for each device, for each GemmEntry in template order
        (REVERSED for BACKWARD, pipeline_fine.py:542-543), resolving every comm
        key against `_specs_for(work)`:

            for key in pre_keys(entry, direction):  emit CommStep(key)     # placement="pre"
            emit ComputeStep(entry)
            for group in parallel_groups(post_keys(entry, direction)):     # placement="post"
                emit the group (a single CommStep, or the MoE hot/cold join)

        `pre_keys`/`post_keys` come from `CommSpec.placement`. THE FINE PATH MUST
        HONOR THEM (today it does not — block.py:126-127, pipeline_fine.py:577-589
        chain every key post). This is a NO-OP for all 42 goldens: no production
        comm rule sets placement="pre" (train_timing.py COMMUNICATION_RULES /
        MOE_COMMUNICATION_RULES / _make_moe_comm_specs all emit "post"; only
        tests/test_block_builder_diff.py:249 uses "pre"). Therefore P3 stays T1-exact."""
```

`parallel_groups` is the hoisted `_comm_parallel_groups` (`block_program.py:171-186`) and the MoE
hot/cold join is the hoisted `_attach_moe_parallel_post_group` (`block_program.py:285-374`), both
lifted out of the `build_block_root` closure so the FINE path can call them — this is the single
biggest blocker in `ext_moe_flat.md` and it is discharged by hoisting, not by a new branch.

### 3.5 L2 invariants

| id | Invariant | Verified by |
|---|---|---|
| **P1** | Every op has a device in `placement.layout`; `layout.coords_of(device)` succeeds. | `program/validate.py` V1 (extended) |
| **P2** | Every `GroupKey.members` equals `CommunicatorFactory.members(declared_axes, device)`. No group is derived from a participant count. | T1 parsed `comm_groups.json` member sets; grep gate: `participants ==` appears nowhere in `program/` |
| **P3** | **(new, V7)** Every grouped `CollectiveOp` is instantiated on **every** member device of its group. | new `validate.py` V7 + the always-on `et_emit` group-order postcondition (`et_emit.py:463-533`) |
| **P4** | The MoE hot rank of a routing group is `min(members)` under `layout.linearize`, so hot-before-cold construction is guaranteed, not assumed. | `tests/test_placement.py`; replaces the runtime error at `block_program.py:356-361` |
| **P5** | `devices_for` is injective on `(WorkItem, cluster_rank)`. | `tests/test_placement.py` |
| **P6** | *(new, 2026-07-26)* Every member of every `GroupKey` is an element of `placement.devices()`; `dp` cannot be spanned (`GroupError`); a dp requirement gets `None` per instance device. `group_layout` preserves every device id of `layout`. | `tests/test_placement.py::test_b2_dp_requirement_gets_no_communicator`, `::test_b2_group_partition_covers_exactly_the_device_set` |
| **P7** | *(new, 2026-07-26)* `SyncSpread.PER_CLUSTER_RANK` yields exactly `cluster_size` devices for **every** `place_on`, whatever the `PlacementPolicy` pins; fewer is a `PlacementError`. | `tests/test_placement.py::test_b4_per_cluster_rank_spans_the_stage_when_place_on_is_pinned`, `::test_b4_per_cluster_rank_raises_when_it_cannot_span_the_cluster` |

### 3.6 What L2 deletes

* `program/pipeline_fine.py` — **entire file, 970 LOC**.
* `program/block_program.py` — **entire file, 588 LOC** (its MoE helpers move to `block.py`).
* `program/pipeline_coarse.py` — **entire file, 325 LOC**.
* `program/pipeline_fine.py:268-303` `_hw_id_for_rank` / `_offset_stage_device`;
  `program/block_program.py:188-189` `_rank_id`; `memory_estimation.py:309-322` — the three
  surviving rank formulas.
* `program/pipeline_fine.py:796-850` `propagate_local_hw_ids`, and `local_hw_id` as a concept.
* `program/legacy_lowering.py:135-181` `_compute_stage_axis_coords` / `_build_axis_groups`, and
  `:196-282` `_assign_collective_labels_with_members` incl. the composite-ep hack at `:243-248`.
* `llm_execution.py:144-145` `_subset_descriptor(["tp","cp","ep"])` / `(["pp","dp"])` — the
  cluster/schedule axis partition, currently duplicated at `layout.py:248` and `layout.py:303`.

---

## 4. L3 / L4 — `program/sched/` and `program/build.py`

### 4.1 SchedulePolicy

```python
# program/sched/policy.py
@dataclass(frozen=True)
class LayerAssignment:
    """Explicit layer -> stage map. GENERALIZES `layers_per_stage: Tuple[int,...]`
    (schedule.py:163) from monotone COUNTS to an arbitrary map — the prerequisite
    for interleaved virtual stages (ext_1f1b.md item 1)."""
    stage_of_layer: Tuple[StageId, ...]        # len == num_layers
    num_stages: int

    def stage_of(self, layer: LayerId) -> StageId: ...
    def layers_of(self, stage: StageId) -> Tuple[LayerId, ...]: ...
    def min_layer(self, stage: StageId) -> Optional[LayerId]: ...   # the optimizer attach rule

    @classmethod
    def contiguous(cls, num_layers: int, pp: int) -> "LayerAssignment":
        """The legacy remainder-first split (schedule.legacy_layers_per_stage:83-90):
        base+1 for the first (num_layers % pp) stages. THE DEFAULT."""

    # -- AMENDMENT 2026-07-26 (B5) ---------------------------------------
    # `stage_of` here takes a LayerId. The L1 `StagePartition` protocol (§2.4)
    # no longer declares a `stage_of` at all, so the two are no longer in
    # conflict and one object can be passed to BOTH `Placement(layers=...)` and
    # `ShardingContext(stages=...)`. `program.policies.sharding.ContiguousStages`
    # is exactly that object today; P4 renames it, nothing else.


@dataclass(frozen=True)
class ScheduleSlot:
    index: int          # GLOBAL, dense, 0..N-1 — the schedule step
    stage: StageId
    work: WorkItem


@dataclass(frozen=True)
class ScheduleDep:                                  # -- AMENDMENT 2026-07-27
    """A dep the pipelining pattern implies, ON ONE DEVICE."""
    before: WorkItem
    after: WorkItem
    device: DeviceId
    reason: str = "device_serialization"


@dataclass(frozen=True)
class Schedule:
    policy: str
    layers: LayerAssignment
    slots: Tuple[ScheduleSlot, ...]

    def order_for(self, stage: StageId) -> Tuple[WorkItem, ...]:
        """The stage's execution order = slots with that stage, by `index`."""
    def index_of(self, work: WorkItem) -> int: ...

    # -- AMENDMENT 2026-07-27 -------------------------------------------
    def device_projection(
        self, devices_for: Callable[[WorkItem], Sequence[DeviceId]]
    ) -> Mapping[DeviceId, Tuple[WorkItem, ...]]: ...
    def implied_deps(
        self, devices_for: Callable[[WorkItem], Sequence[DeviceId]]
    ) -> Tuple[ScheduleDep, ...]:
        """The deps this pipelining pattern implies: adjacent pairs of each
        DEVICE's projection, in (after-slot, device) order. R3 materializes each
        one that is not already transitively implied (§4.3)."""
    def last_microbatch_of(self) -> MicroBatch: ...      # Class B 10h, computed
    def check_permutation(self, work: WorkSet) -> None: ...   # S3


class SchedulePolicy(Protocol):
    name: str
    def layer_assignment(self, fw: FrozenWorkload) -> LayerAssignment: ...
    def schedule(self, fw: FrozenWorkload, work: WorkSet) -> Schedule:
        """Total order over ALL work, globally indexed. MUST be a permutation of
        `work.items` (S3 below)."""


# program/sched/gpipe.py
@dataclass(frozen=True)
class GPipeSchedule(SchedulePolicy):
    """THE DEFAULT. Reproduces today's order:
       forward  : microbatch-major, stage-ascending, layer-ascending
                  (embedding, layers..., softmax)
       backward : microbatch-DESCENDING (schedule.py:693 `for b in reversed(range(B))`),
                  stage-descending, layer-descending, each layer preceded by its
                  RECOMPUTE item when materialized
       optimizer: after all backward work, stage-ascending (schedule.py:1054)"""
    name: str = "gpipe"
```

`program/sched/onefonebee.py` (student project) implements the same protocol and needs **no other
edit**: `GradAccumPolicy`'s `microbatch == 0` rule becomes wrong under 1F1B, so `GradAccumPolicy`
gains a `last_microbatch_of(schedule)` hook rather than a hardcoded `0` — declared here so the
student does not have to negotiate it.

> **AMENDMENT 2026-07-27 (L3): a `Schedule` must DECLARE the deps its pattern implies.**
> §4.1 as written declared only an order, and an order that is not in the DAG is not an order:
> **AstraSim's one-slot rule constrains concurrency, not order** — a rank runs one compute at a time
> but is free to pick any ready node. Legacy therefore materializes the cross-microbatch dependency
> explicitly (`git show 85894c6:simulate_train_graph.py:836-856`:
> `layer_exit_nodes[b][l].add_child(embedding_node[b+1])  # dependency: finish stage before next
> batch embedding starts`), and so must every `Schedule`. `implied_deps` is that declaration; R3
> (§4.3) materializes each dep that is not already transitively implied.
>
> `devices_for` is **injected** (it is `Placement.devices_for`) rather than imported, so L3 keeps
> knowing nothing about devices while `ScheduleDep` is still per DEVICE — which it must be, because
> at FINE a stage is `cluster_size` devices and a pinned kind (softmax, Class B 10d) lives on only
> one of them.
>
> **The interleaving seam is `LayerAssignment` and nothing else.** It is an explicit map, and no rule
> in L2/L3/L4 requires it to be monotone or contiguous: `Placement.stage_of` reads it per layer, R2
> compares the *devices* of consecutive layers (so a layer whose stage revisits an earlier device is
> an ordinary same-device link), and R3 projects `slots` through `devices_for` (so a device appearing
> twice in the layer order simply has more work in its projection). `LayerAssignment.explicit()` is
> the constructor for it and `contiguous_layers()` reports the answer into
> `Program.meta.misc["layer_assignment_contiguous"]`. A 1F1B policy is therefore a new
> `SchedulePolicy` and NOTHING else. **1F1B is NOT implemented in this wave.**

### 4.2 build()

```python
# program/build.py
def build(
    fw: FrozenWorkload,
    *,
    granularity: Granularity,
    sharding: ShardingPolicy,
    schedule_policy: SchedulePolicy,
    recompute: RecomputePolicy,
    routing: Optional[MoERoutingPolicy] = None,
    overlap: OverlapPolicy = AxisFractionOverlap(),
    grad_accum: Optional[GradAccumPolicy] = None,
    dp_count: Optional[int] = None,
    label: str = "",
    gmap_workdir: Optional[str] = None,
    # -- AMENDMENT 2026-07-27 (keyword-only, all defaulted) --------------
    placement_policy: PlacementPolicy = LEGACY_PLACEMENT,   # Class B 10d, swappable
    sync_order: SyncOrder = SyncOrder(),                    # §4.4, swappable
    validate: bool = True,
    check_group_membership: bool = False,                   # V7 — see §4.8
) -> Program:
    """Compose L1+L2+L3 into a Program. Mechanical, ~600 LOC, no policy decisions
    of its own. Phases, in this exact order:

      0. check_granularity_preconditions(granularity, fw)   (amendment 2026-07-27)
      1. work      = restrict_work_for(granularity,
                        enumerate_work(fw, recompute))                      (L1)
      2. schedule  = schedule_policy.schedule(fw, work)                     (L3)
      3. placement = Placement(fw, granularity, schedule.layers)            (L2)
      4. chains    = {item: BlockExpander(...).expand(item) for item in work} (L2)
      5. deps      = R1 + R2 (data-flow, cross-layer)                       (§4.3)
      6. deps     += R3 (device serialization from `schedule`)              (§4.3)
      7. reqs      = sharding.workload_requirements(ctx)
                   + [r for item in work for r in sharding.requirements(item, ctx)]
                   + routing requirements                                   (L1)
                   filtered by grad_accum.emits(...)
      8. deps     += R4 (sync attach, §4.4) and materialize the sync ops    (L2 for devices/groups)
      9. overlap   = realize every OverlapDecl (§4.5)
     10. order     = program order (§4.6); assign uids; build Program; validate

    `build()` NEVER inspects a name, a role string, or an axis literal."""
```

The dispatcher's four modes become four `build()` calls with different `granularity` and consumers —
`llm_execution.py:343-351` keeps its shape; `_build_coarse_program` / `build_fine_program` /
`build_block_program` / `lower_coarse_for_emission` all collapse into it.

> **AMENDMENT 2026-07-27 (L4): what a granularity REQUIRES, stated.** §3.1 defines BLOCK as "one
> layer expanded over the (tp,cp,ep) sublayout **only, no pipeline**" while phase 1 above enumerates
> the whole workload — a contradiction whose resolution was left to whichever dispatcher built the
> spec. Two named functions state it instead:
>
> * **`restrict_work_for(granularity, work)`** — BLOCK expresses `WorkKind.LAYER` / `RECOMPUTE` only.
>   Its device space carries no `pp`, so embedding / softmax / optimizer work has nowhere to live.
>   The answer is stamped into `Program.meta.misc["work_restricted"]`.
> * **`check_granularity_preconditions(granularity, fw)`** — BLOCK requires `degrees.dp == 1` and
>   `degrees.pp == 1`. This IS the legacy `dp_override=1` semantics (`block_program`: a transformer
>   block run is a SINGLE-REPLICA measurement) written as a precondition. With it in place **no
>   granularity-specific policy override is needed anywhere**: at `dp == 1`, `sharding_policy_for`
>   already answers `NullSharding` and `GradAccumPolicy.emits` is unconditionally False, so neither
>   the ZeRO lattice nor the EP grad sync (S12, which is the *routing* policy's requirement and would
>   otherwise slip past a sharding override) can leak into a block measurement.
>
> Consequence for a consumer: `build()` at BLOCK produces ONE program containing both directions,
> where `build_block_program(template, direction, ...)` produced one per direction. A caller
> measuring a single direction builds a single direction's workload.

### 4.3 Dependency rules

Deps are **computed**, never walked off a mutable graph. Exactly four classes
(`DepClass`, §2.2) and they are applied in this order.

**R1 — data-flow inside a block chain (`DepClass.DATA_FLOW`).**
Within one `ExpandedChain`, `steps[i]` depends on `steps[i-1]`. The chain shape comes from
`BlockTemplate` and **honors `CommSpec.placement`** (pre/post) — see §3.4. `entry`/`exit` are the
chain's first/last step and are the only handles R2/R3/R4 may reference.

**R2 — cross-layer, L → L+1 (`DepClass.DATA_FLOW`).**
For consecutive work in one microbatch's data-flow order
(`EMBEDDING → LAYER 0 … LAYER L-1 → SOFTMAX` forward; the mirror image backward, where a layer's
entry is its `RECOMPUTE` chain when present):

```
# -- AMENDED 2026-07-27: the SIZE is decided per STAGE, the ENDPOINTS per device
size = 0 if placement.same_stage(producer_work, consumer_work) else \
       ByteSource("cross_layer", CEIL_DIV_CLUSTER).bytes_for(fw, placement.cluster_size())

for each producer chain p (device dp) and consumer chain c (device dc):
    emit TransferOp(src=dp, dst=dc, size=size,
                    producer = p.exit, consumers = (c.entry,))
    deps(c.entry) += transfer
```

> **AMENDMENT 2026-07-26 (B3).** The `instances` argument is `placement.cluster_size()` — **1 at
> COARSE**, `tp*cp*ep` at FINE/BLOCK — which is what makes this one rule reproduce both
> `pipeline_coarse.py:222,238` (raw) and `pipeline_fine.py:645` (divided). It is NOT
> `fw.spec.cluster_size()`; see §2.2.

> **AMENDMENT 2026-07-27 (L4): the byte decision is per STAGE; a same-device link is still an OP.**
> The original pseudocode keyed both the size and the "plain dep vs transfer" choice on
> `dp == dc`. Both were wrong:
>
> 1. **Size.** `cross_layer` models PIPELINE activation movement. At FINE the embedding is pinned to
>    cluster rank 0 (Class B 10d) while layer 0 spans the stage, so `EMBEDDING → LAYER 0` is
>    *cross-device inside one stage* — keying the size on the device invented a `cross_layer` payload
>    between tp ranks (measured: 4 bogus 2 MB transfers on `train:analytical:dp1tp2cp1pp2mb2sp1`
>    before the fix). Legacy decides it at the COARSE level with
>    `prev_node.hw_id == curr_node.hw_id` (`schedule.py:628,639,648,755,763,771`) — hw_id IS the
>    stage — and the FINE expansion inherits the zero-byte control edge.
> 2. **Plain dep vs op.** The `dp == dc` branch above says "plain dep", but Class B item 9 (and §7's
>    binding table) say the same-stage zero-byte PIPELINE event's PRESENCE is load-bearing for the
>    analytical evaluator's ready-scan. R2 therefore ALWAYS emits a `TransferOp`; a same-device one
>    carries zero bytes and is elided at emission.
>
> **The `RECOMPUTE → LAYER/BACKWARD` link inside one layer is the one exception**: a plain
> same-device dep, no transfer. Legacy wires it as a direct
> `recompute_node.add_child(transformer_node_b)` (`schedule.py:750`) with no `CommEvent`, because the
> rematerialized activation never leaves the device that recomputed it. The *cross-layer* backward
> link (`LAYER/BWD(l) → entry(l-1)`) is an ordinary R2 transfer, and `entry(l-1)` is the RECOMPUTE
> chain when one exists (`_bwd_entry_node`, `schedule.py:682-687`).

At FINE the pairing is per cluster rank `r → r` (`pipeline_fine.py:659-694`); when one side is
pinned to a single device (embedding, softmax) that chain pairs with **every** chain of the other
side, which is what legacy does by giving one edge many children. The
*compute-anchor double-dep* (`pipeline_fine.py:672-687` — the SEND must fire off the last COMPUTE,
not off a trailing collective) becomes a rule of R2, stated once:
`TransferOp.deps` includes both `p.exit` and the nearest preceding `ComputeStep` on `p`'s chain when
`p.exit` is a collective.

Same-device transfers remain legal and are elided at emission (`ir.py` §2.2 amendment,
`et_emit.py:351`). **Note for P8:** `validate.py:31-38` was widened to bless nonzero-byte
same-device transfers, which is what silently drops the MoE `residual_p2p`
(`ext_moe_flat.md` blocker 4). L4 adds **V8**: a same-device transfer with `size_bytes > 0` whose
`CommSpec.moe_component` is set is an error, not a shrug.

**R3 — schedule serialization (`DepClass.SCHEDULE`). THE PRECISE RULE.**

Legacy hand-wires GPipe cross-microbatch edges at `schedule.py:655-672` (forward) and `:955-976`
(backward): at each stage boundary, `exit(b) → entry(b+1)`. Those edges are **not** modeled; they
are *derived*:

```python
# AMENDED 2026-07-27: the adjacent pairs come from the SCHEDULE, which declares
# them (`Schedule.implied_deps`, §4.1) — R3 only decides whether to materialize.
for dep in schedule.implied_deps(placement.devices_for):        # slot-monotone
    source = exit_node(dep.before, dep.device)
    target = entry_node(dep.after, dep.device)
    if not reaches(source, target):
        deps(target) += source                                  # DepClass.SCHEDULE
```

* **Per DEVICE, not per stage.** At FINE a stage is `cluster_size` devices; the projection is the
  stage's slot order restricted to the chains placed on that device.
* **`reaches(a, b)`** is transitive reachability in the graph built so far (R1 + R2 + the R3 edges
  already added). *(Amended 2026-07-27: implemented as a backward BFS from `b` with early exit, over
  the one `deps` structure — which already carries every edge class including a `TransferOp`'s
  producer/consumer wiring. The "per-device dominator" shortcut the original text suggested is
  unsound across devices: an R3 edge on device X can make a pair on device Y reachable, so the
  question is asked against the WHOLE graph, in `(after-slot, device)` order so earlier answers are
  visible to later ones. A hit costs almost nothing; a miss costs one ancestor walk, and R3 asks
  O(devices × slots) questions.)*
* **Redundancy elimination is the point.** Consecutive layers of the same microbatch on one device
  are already chained by R2, so `reaches` is True and no edge is added — reproducing legacy, which
  only ever wires the *microbatch boundary*.
* **The legacy special cases fall out and must not be re-coded:**
  * `schedule.py:663-664` "if the stage is 0, the successor is `embedding_node[b+1]`" — stage 0's
    device order is `[…, LAYER/FWD(b, last), EMBEDDING/FWD(b+1), …]`, so the adjacent pair *is*
    that edge;
  * `schedule.py:670-672` `softmax_node[b] → layer_entry_nodes[b+1][first]` — stage `pp-1`'s order
    is `[…, LAYER/FWD(b,·), SOFTMAX/FWD(b), LAYER/FWD(b+1,·), …]`;
  * `schedule.py:969-970` "if `hw_id == pp-1`, successor is `softmax_node_b[b-1]`" and `:976`
    `embedding_node_b[b] → _bwd_entry_node(b-1, first)` — the mirror images on the backward
    projection.
* **Zero-byte same-stage PIPELINE control events** (`schedule.py:629`, `:640`, `:649`, `:756`,
  `:764`, `:772`) are **not** R3 artifacts: they are R2 same-device links. Class B item 9 says their
  *presence* is load-bearing for the analytical evaluator's ready-scan; they are therefore emitted
  as same-device `TransferOp`s exactly as today (`legacy_lowering.py:1062-1090`).

**R4 — sync attach (`DepClass.SYNC`).** §4.4.

### 4.4 Sync-attach resolution (deterministic, order-declared)

`attach_parallel_edge` reads the *current* `parents`/`children` of a mutable object, so today the
result depends on which lattice block ran first (the forward ZeRO-3 block at `:848-892` runs before
the backward GPipe wiring at `:955-976`; the backward ZeRO-3 block at `:978-1046` runs after). That
is not reproducible from a declaration set, so it is **replaced by a declared order**:

```python
@dataclass(frozen=True)
class SyncOrder:
    """The total order in which SyncRequirements are resolved. A NAMED POLICY,
    not an accident of construction order."""
    name: str = "phase_major"
    def key(self, req: SyncRequirement) -> Tuple[Any, ...]:
        # (phase rank, microbatch, layer, comm_key) — phase rank:
        # FWD_ENTRY < FWD < GRAD < BWD_ENTRY < BWD
        ...
```

**Resolution rule (normative).** Requirements are resolved in `SyncOrder` order. For requirement
`r`, `deps(anchor)` and `succ(anchor, via)` are evaluated against the snapshot consisting of

* **all** R1 + R2 + R3 edges (the complete base graph), and
* the `DepClass.SYNC` edges contributed by requirements strictly earlier in `SyncOrder`.

A requirement whose `anchors` resolve to an empty set is **dropped** with a debug log (§2.3 note 1).
`AFTER(SyncKey)` requires the referenced requirement to be strictly earlier in `SyncOrder`;
violation is a `SyncError`, which makes the S3/S5/S10 chaining a checkable property instead of a
construction-order coincidence.

Materialization: each requirement becomes one `CollectiveOp` **per device** in
`placement.devices_for_sync(req)`, each carrying
`size_bytes = req.bytes.bytes_for(fw, len(devices))` and, from
`CommunicatorFactory.groups_for(req, placement)` (§3.3):

* a non-dp requirement: `group = group_for(req.axes, device)`, `is_dp = False`;
* a **dp** requirement (`req.is_dp`): `group = None`, `is_dp = True` — the wire members are stamped
  at emission over pre-dp device ids (`ir.py:92-99`), and `CollectiveOp.label` stays `None`.
  *(amendment 2026-07-26, B2)*

### 4.5 Overlap realization

For a `SyncRequirement` (or block-template `CommStep`) carrying an `OverlapDecl`:

* `OverlapAnchor.PRODUCER`, `0 < f < 1`: split the producing `ComputeOp` into
  `head = d*(1-f)` and `tail = d*f`; `deps(head) := deps(compute)`; `deps(tail) := {head}`;
  the collective moves to `deps(coll) := {head}`; every consumer of the collective gains
  `deps += tail`. (Verbatim `_split_tp_node_fine`, `transforms.py:190-218`.)
* `OverlapAnchor.PRODUCER`, `f >= 1`: hoist — the collective takes the compute's deps and the
  compute takes the collective's consumers. (`transforms.py:177-188`.)
* `OverlapAnchor.CONSUMER`, `0 < f < 1`: split the collective by bytes into
  `block = ceil(total*(1-f))` and `ovlp = total - block`; the `blocking_consumer` op depends on
  `block`; every other consumer depends on `ovlp` (or on `block` when `ovlp == 0`) and on the
  blocking consumer. (Verbatim `_split_cp_edge_fine`, `transforms.py:246-298`.)
* `OverlapAnchor.CONSUMER`, `f >= 1` or `total == 0`: the blocking consumer is re-parented onto the
  collective's predecessors. (`transforms.py:233-244`.)

Split ops get **fresh uids** from §4.6. The op-id reuse quirks (`transforms.py:200`, `:255`, `:267`)
do not survive.

**Retiming interaction:** a per-DP duration profile cannot be split (`transforms.py:468-478`).
`build()` therefore realizes overlap **before** any retime write-back can apply; `retime` operates on
a built Program and is forbidden from running before `build()` returns. Stated so no implementer
re-discovers it.

### 4.6 Program order

**Normative definition.** Let

```
key(op) = (slot_index(op), device_index(op.device), intra_index(op))
```

where

* `slot_index` = `Schedule.index_of(owning WorkItem)`; a sync op inherits its `place_on` item's
  index; a `TransferOp` inherits its producer's; a split op inherits its source's;
* `device_index` = position of `op.device` in `placement.devices()`;
* `intra_index` = the deterministic position within one `(WorkItem, device)` expansion:
  chain position for R1 steps; for sync ops, `(SyncOrder.key(req), instance_index)`; for transfers,
  R2 emission order.

Uids are assigned by **Kahn's algorithm over the complete dep graph, with a min-heap keyed on
`key(op)`**. This is total (keys are unique by construction), respects `dep < uid` (V1), is
schedule-major (so AstraSim node-id priority tracks the intended schedule — `../CONTEXT.md`: "node
ids are therefore scheduling priorities"), and is a pure function of the inputs.

`Program.meta.misc["program_order"] = "kahn(slot,device,intra)"`. There is **one** id policy;
`et_emit`'s `id_policy` parameter and its `NotImplementedError` (`et_emit.py:173-179`) are removed.

### 4.7 L4 Program changes

```python
# program/ir.py — DELETED fields
ComputeOp.legacy_op_id, ComputeOp.post_deps
CollectiveOp.legacy_op_id, CollectiveOp.post_deps
TransferOp.send_seq, TransferOp.recv_seq, TransferOp.legacy_tag

# program/ir.py — ADDED
CollectiveOp.axes: Tuple[AxisName, ...]      # declared communicator axes (diagnostics + validate)
ComputeOp.work: WorkItem                     # stable semantic identity (the never-built "OpKey")
CollectiveOp.work: Optional[WorkItem]
Op.succs: Tuple[OpUid, ...]                  # ORDERED successors — see below
TransferOp.moe_component: Optional[str]      # AMENDMENT 2026-07-27 — V8 needs it ON THE OP

# program/ir.py — CHANGED
CollectiveOp.size_bytes: float               # was int; dp reducer sizes are floats and must stay
                                             # floats (analytic_sim.py:49-51). The int() truncation
                                             # moves to the emitter, where AstraSim needs it.
ComputeOp.deps / CollectiveOp.deps           # NO LONGER SORTED. `dict.fromkeys(sorted(...))`
                                             # (legacy_lowering.py:768,787) destroys the adjacency
                                             # order that memory_sim/analytic_sim need; keeping
                                             # insertion order + adding `succs` is what lets P6
                                             # delete the proto graphs.
ProgramBuilder                               # becomes THE production construction API
```

`Op.work` is the stable semantic identity that `retime` / `memory_sim` / fault projection look up
by, replacing positional `events[op.uid]` mirroring (`retime.py:126-127`) and the name-prefix walks.

> **AMENDMENT 2026-07-27 (L4).** The ADDED fields land NOW, as defaulted fields, because `build()`
> populates them and V7/V8 read them; the DELETED ones stay until P5 deletes the legacy lowering that
> writes them. A legacy-lowered program leaves every added field at its default (`work=None`,
> `succs=()`, `axes=()`, `moe_component=None`), so nothing about the current path moves.
> `CollectiveOp.size_bytes` is now annotated `float` and `build()` stores a float: the `int()`
> truncation legacy applies at construction (`pipeline_coarse.py:238`,
> `legacy_lowering`) is what this type change stops mandating.

### 4.8 L3/L4 invariants

| id | Invariant | Verified by |
|---|---|---|
| **S1** | `Schedule.slots` indices are dense `0..N-1` and unique. | `tests/test_build.py::test_s1_*` |
| **S2** | `Schedule.order_for(stage)` is a total order and every WorkItem appears exactly once across all stages. | `tests/test_build.py::test_s2_*` |
| **S3** | `schedule()` output is a permutation of `work.items`. | `tests/test_build.py::test_s3_*`, `Schedule.check_permutation` |
| **D1** | **No redundant schedule dep**: for every added `DepClass.SCHEDULE` edge `a → b`, `b` was not reachable from `a` before the edge. | `tests/test_build.py::test_r3_no_redundant_schedule_edges` (removes the edge and asserts reachability changes; the edges are stamped onto `Program.meta.misc["schedule_edges"]` as uid pairs so D1 is a property of the ARTIFACT) |
| **D2** | Every op is reachable from a root and every non-root has ≥1 dep (no orphans — the S6/S14 drop happens *before* materialization). | `tests/test_build.py::test_o2_*`; `validate.py` (extended V1) |
| **O1** | Program order is deterministic: two builds from the same `FrozenWorkload` produce byte-identical Programs and canonically-identical ET bundles. | `tests/test_build.py::test_o1_*`; T3 deterministic re-emission compared **canonically** (`equiv/canonical.py`), not `filecmp` |
| **O2** | `dep < uid` for every op (V1). | `program/validate.py`; `tests/test_build.py::test_o2_*` |
| **V7** | *(new; **opt-in**, amended 2026-07-27)* every grouped `CollectiveOp` is instantiated on every member device of its group. | `program/validate.py` (`check_group_membership=True`); `tests/test_build.py::test_v7_is_exactly_the_a2_shape` |
| **V8** | *(new)* a same-device `TransferOp` with `size_bytes > 0` and a set `moe_component` is an error. | `program/validate.py` (always on); `tests/test_build.py::test_v8_*` |
| **V1–V6** | unchanged (`program/validate.py:67-256`); **V6 is promoted to always-on** in `build()` (today every production caller passes `check_races=False`, so CONTEXT constraint 2 is unenforced outside tests). | `tests/test_program_ir.py:144-258` |
| **G1** | The always-on group-order postcondition still holds. | `et_emit.py:463-533`, T3 |

> **AMENDMENT 2026-07-27 (L4): V7 is OPT-IN, and that is a finding, not an oversight.** V7 says every
> member device of a communicator issues the group's collectives. BUG_LEDGER **A2**
> (`SyncSpread.CLUSTER_RANK_0`) puts exactly ONE instance of a stage-spanning collective on cluster
> rank 0, and §7 preserves that as the DEFAULT. The two are contradictory for every non-dp requirement
> whose group spans more than one device — the `ep` grad sync at `tp*cp*ep > 1` is the live example.
> V7 therefore cannot be fatal by default without making today's modeling content unbuildable:
> `build(..., check_group_membership=True)` is the flag, its default is `False`, and **flipping
> `SyncSpread` to `PER_CLUSTER_RANK` in P7 is exactly what lets that default flip.** The
> contradiction is pinned in both directions by
> `tests/test_build.py::test_v7_is_exactly_the_a2_shape`.
>
> **V6 promoted to always-on has a known false-positive class.** V6 warns when two same-group
> collectives on one device have no ordering path. A gradient reducer is a SINK by construction (it
> hangs off a backward exit and nothing follows it), so every pair of reducers of one group on one
> device trips it. For a program whose order is ONE global order projected onto every device
> (CONTEXT constraint 2) that is not the race V6 describes. The warning stays — it is a warning — but
> a caller that builds in bulk should silence `GroupRaceWarning`, and P5 should consider narrowing V6
> to pairs where at least one member has a successor.

### 4.9 What L3/L4 delete

L3:

* `program/schedule.py:655-672` (forward GPipe cross-mb) and `:955-976` (backward) — the
  hand-wired cross-microbatch edges become R3.
* `program/schedule.py:83-100` `legacy_layers_per_stage` / `_layer_to_stage` → `LayerAssignment`.
* `program/schedule.py:1048-1072` — the optimizer tail's placement/attach logic (the WorkItem is
  enumerated at L1, the slot at L3, the dep by R3).

> **Verified 2026-07-27:** the optimizer attach falls out of R3 as claimed. Legacy attaches stage
> `s`'s optimizer to `embedding_node_b[0]` (stage 0) or `_bwd_exit_node(0, min_layer(s))`, and those
> ARE the last backward items in each stage's device projection, because GPipe's backward walk
> descends microbatches (so `b == 0` is last) and descends layers (so `min_layer(s)` is last on `s`).
> No optimizer-specific rule exists in `build.py`.

L4 (**the rebaseline** — one atomic cutover, no partial step):

* `program/legacy_lowering.py` — **entire file, 1095 LOC**, including both near-duplicate backward
  walkers (`:412-475`, `:477-541`), the stage re-derivation (`:323`), the per-edge stage attribution
  (`:350-381`), the name-keyed communicator reconstruction (`:593-598`), and the Kahn heap with its
  stale legacy-tie justification (`:645-670`).
* `program/schedule.py:300-460` `ComputeEvent` / `CommEvent`; `:463-471` `PipelineEvents`;
  `:478-1079` `build_pipeline_events`. **`program/schedule.py` is deleted in full** and the L3
  package is renamed `program/sched/` → `program/schedule/` in the same commit.
* `program/transforms.py:110-383` — the proto-level overlap rewrite and the object-graph
  scaffolding (`_objectify`, `_successor_map`, `_renumber`).
* `program/ir.py` ordering metadata (§4.7) and the `ProgramBuilder` "no production caller" docstring.
* `program/et_emit.py:173-179` — the `id_policy` branch and `NotImplementedError`;
  `:135-139` — the `_send_control`/`_recv_control` name-suffix classification, replaced by
  `TransferOp.is_control` (the value is carried; re-parsing it from a string it wrote is audit item 7).
* `Program.meta.misc["fine_proto_root"]`, `["coarse_proto_root"]`, `["coarse_events"]`,
  `["fine_pp"]`, `["coarse_pp"]` — the second representation.
* `llm_execution.py:596-597` and the dead `:620-621` — the flattened-MoE rejection (lands in P8;
  the interface above makes it a policy, not a guard).

L5 (P6, consumers): `analytic_sim` (`:162-192`), `memory_sim` (`:73`, `:458-460`) and `viz` read
`Program.ops` + `Op.succs`; the `meta.misc` proto readers disappear with the fields above.

---

## 5. Invariant → gate map (summary)

| Level | Invariants | Gate tier (PLAN §"Gate redesign") | Concrete test |
|---|---|---|---|
| L0 | W1–W5 | T1 (byte histograms), T3 | `tests/test_workload_spec.py` (new); W5 in `tests/test_placement.py` + `tests/test_policies.py` |
| L1 | K1–K9 | T1 (per-group collectives, byte histograms by kind/axis), T2 exact | `tests/test_policies.py` (new); `tests/test_equiv_golden.py` |
| L2 | P1–P7 | T1 (`comm_groups.json` member sets, per-rank op multisets), T3 (group-order postcondition) | `tests/test_placement.py` (new); `tests/test_program_layout.py` |
| L3 | S1–S3, D1 | T1 (critical-path length), T3 (dlsim completability) | `tests/test_build.py` (new — S1–S3, D1 and the L3 seams; the planned `test_schedule_policy.py` / `test_build_deps.py` split collapsed into one file because the L3 and L4 tests share ONE fixture, `tests/test_policies.py::make_workload`); `equiv/dlsim.py` |
| L4 | O1–O2, V1–V8, G1, D2 | T1 **exact** + T3 + T5; **T2 produces the reviewed delta table** | `tests/test_program_ir.py`, `program/validate.py`, `equiv/runner.py`; `tests/test_build.py` — the always-on rule tests PLUS an env-gated oracle (`RAPID_BUILD_DIFF=1`) that builds every golden spec at every granularity through `build()` AND through the current path and compares the T1 quantities, with a `DIVERGENCES` table each residual must map to |
| all | modeling content unchanged | T5 physical (`test_func_test.py`, `test_IMEC_*`, `test_koyeb_*`, `validation_scripts/*`) within existing error thresholds | unchanged |

Golden gate, every phase:

```
env RAPID_ASTRA_CACHE_MODE=NO_CACHE \
    LD_LIBRARY_PATH=/u1/ee/karfakis/gcc-10.2.0/lib64:/app/nanocad/projects/personal/gkarfakis/anaconda3/lib \
    ./.venv/bin/python -m pytest tests/test_equiv_golden.py -q      # 42/42
```

> **`NO_CACHE` is mandatory in every phase.** The AstraSim result-cache key hashes the *sorted*
> manifest, not the `.et` bytes (`astrasim_lib/integration.py:456-465`), so a reorg that changes
> emission order while preserving the op multiset gets a **cache hit on the old result** — goldens
> pass while the schedule silently changed. This is bug **A1** and it masks exactly the class of
> change this restructure makes.

---

## 6. Deletion ledger

| Lands | Deleted | LOC |
|---|---|---|
| L0 (P2) | `schedule.ScheduleInputs` + `ScheduleSpec`; `TransformerBlockSpec`; 3× run-policy duplication in `llm_execution` | ~230 |
| L1 (P2) | `attach_parallel_edge`; the 12-key lattice; `should_emit_dp_comm` ×2; `recompute_enabled`; `zero3_offset_hint` + `"bwd" in name`; `_ensure_zero3_per_rank_edges`; the 3 copy-attr tuples; `_moe_hot_rank`/`_moe_parallel_token`; both overlap implementations | ~800 |
| L2 (P3) | `pipeline_fine.py`, `block_program.py`, `pipeline_coarse.py`; `propagate_local_hw_ids`; 3 rank formulas; the group-inference block of `legacy_lowering` | ~2,050 |
| L3 (P4) | GPipe cross-mb wiring ×2; `legacy_layers_per_stage`/`_layer_to_stage`; optimizer-tail attach | ~120 |
| L4 (P5) | `legacy_lowering.py`; the rest of `schedule.py`; `transforms.py` proto half; IR ordering metadata; `id_policy`; the `meta.misc` proto roots | ~2,100 |
| **total** | | **~5,300 deleted / ~1,900 added → 7,849 → ~4,400** (PLAN target: 4,000–4,500) |

---

## 7. Class-B / Class-A binding table

Every Class B item is a named policy with a default; every Class A item is a policy *field* whose
default is today's (wrong) value, so P7 is a default change plus a delta table.

| Ledger | Item | Named policy | Default (= today) | The fix (P7) |
|---|---|---|---|---|
| B 1 | ZeRO-3 gather bytes un-divided | `ByteSource.split` | `ByteSplit.WHOLE` | n/a — correct for tp/cp |
| B 5 | `local_comp_time` zeroed | `CommSpec.local_comp_time` carried, never materialized; documented in `BlockExpander` | unused | n/a |
| B 8 | interleave scale multiplies the whole total | `RunPolicy.interleave_scale` | legacy closed form | replaced by a real 1F1B `SchedulePolicy` |
| B 9 | same-stage zero-byte PIPELINE events | R2 same-device `TransferOp` (§4.3) | emitted | n/a — presence is load-bearing |
| B 10d | softmax pinned to cluster rank 0 | `Placement.devices_for(SOFTMAX)` | one device | `PER_CLUSTER_RANK` |
| B 10g | singleton group no-op; 1-byte controls | `et_emit` (unchanged) | on | n/a |
| B 10h | `b == 0` means "last microbatch" | `GradAccumPolicy` + `SchedulePolicy.last_microbatch_of` | `0` under GPipe | correct under 1F1B by construction |
| **A2** | DP collective on `rank_tails[0]` only | `SyncRequirement.spread` | `SyncSpread.CLUSTER_RANK_0` | `SyncSpread.PER_CLUSTER_RANK` — which now genuinely spans the stage for every `place_on` (§3.2 amendment 2026-07-26); before B4's fix flipping the default would have been a no-op wherever `place_on` was a pinned kind |
| **A3** | `is_moe_layer` dropped on overlap head split | derived from `Op.work`, not copied | **already impossible** (§2.1) | — (fix lands with L1) |
| **A4** | collective-only stage rank collision | `Program.num_stages_initial()` | pre-extension count | post-extension count |
| **A5** | ZeRO-3 dead store with side effects | deleted with `_ensure_zero3_per_rank_edges` | — | — |
| **A1** | cache key omits DAG structure | `astrasim_lib/integration.py` | manifest-only | hash the `.et` bytes — **fix FIRST** |
| C 3 | manifest records `ALL_REDUCE` as `-1` | `et_emit` manifest | unchanged | — |
| C 7 | Step-11 pre-SCOTCH-remap iteration | deleted with `legacy_lowering` | — | — |
| C 10e | non-unique op ids from overlap splits | deleted by §4.6 | — | — |
| D 10b | optimizer once per stage | `enumerate_work` emits one `OPTIMIZER` per stage | per stage | owner call |
| D 10c | cross-layer activation bytes `/cluster_size` | `ByteSource.split` on `cross_layer` | `CEIL_DIV_CLUSTER`, divided by the caller's `instances` = `placement.cluster_size()` (§2.2 amendment 2026-07-26) | A/B per the ledger |

---

## 8. Rules for implementing agents

1. **Do not add a compatibility shim.** No `legacy_*` name, no `getattr(obj, name, default)` against
   a type defined in `program/`, no dict-carrier.
2. **Do not dispatch on a name, a substring, or an axis literal.** If a decision needs data, the data
   goes on `WorkItem`, `CommSpec`, or a policy object. `et_emit.py` is the calibration: 644 LOC, zero
   occurrences of `zero3`/`moe`/`recompute`/`softmax`/`embedding`.
3. **A policy is a `@dataclass(frozen=True)` with a `name: str`.** No module-level flags, no
   `if flag:` inside an expansion loop.
4. **Every new invariant goes in `program/validate.py`** with an id, and gets a test in the tier
   named in §5. An invariant enforced only at emission time is a last resort, not a design.
5. **Signatures above are binding.** Adding a keyword-only argument with a default is allowed;
   changing a name, a type, or a positional order is an amendment to this file first.
6. **`RAPID_ASTRA_CACHE_MODE=NO_CACHE` on every validation run**, in every phase, including local
   ones (`equiv/runner.py:234` currently defaults to `CACHE_READWRITE` and must be overridden).
   `tools/parallelism_sweep_cache.csv` is keyed on inputs only and will not self-invalidate — delete
   it when timing behavior changes.
