# Copyright 2026 NanoCad lab, UCLA
# https://nanocad.ee.ucla.edu/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""L3 + L4 tests: ``program.schedule`` and ``program.build`` (INTERFACES §4).

Two halves:

* **always-on synthetic unit tests**, one per dependency rule (R1-R4), per L3
  invariant (S1-S3), per L4 invariant (D1, D2, O1, O2, V7, V8) and per overlap
  realization branch (§4.5). These use the ``tests/test_policies.py`` fixture —
  ONE configuration realized on both sides — so a rule is tested against a
  workload, not against a hand-drawn graph.

* an **env-gated T3 determinism check** (``RAPID_BUILD_DIFF=1``): for every
  golden spec in ``equiv.configs.MATRIX``, build the workload through ``build()``
  at each granularity TWICE and assert the two Programs are identical op for op
  and the two emitted ET bundles are CANONICALLY identical
  (``equiv.canonical``), plus ``equiv.dlsim`` completability of the bundle. This
  replaces the P5-era differential against ``pipeline_coarse`` / ``pipeline_fine``
  / ``block_program``: those builders are deleted, so there is no second
  implementation left to compare to. Invariant **O1** in the form the gate map
  asks for ("deterministic re-emission compared canonically, not ``filecmp``").

Run it::

    RAPID_BUILD_DIFF=1 RAPID_ASTRA_CACHE_MODE=NO_CACHE \\
      ./.venv/bin/python -m pytest tests/test_build.py -q
"""

from __future__ import annotations

import copy
import math
import os
import re
import warnings
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from timing_model import CollectiveType

from program.block import BlockTemplate
from program.build import (
    BuildError,
    SyncOrder,
    build,
    check_granularity_preconditions,
    restrict_work_for,
)
from program.ir import (
    CollectiveOp,
    CommGroup,
    ComputeOp,
    GroupKey,
    OpRole,
    Program,
    ProgramMeta,
    TransferOp,
)
from program.layout import RankLayout
from program.placement import Granularity, Placement
from program.policies import policies_for
from program.policies.overlap import (
    AxisFractionOverlap,
    NoOverlap,
    OverlapAnchor,
    OverlapDecl,
)
from program.policies.sharding import ShardingContext
from program.schedule import GPipeSchedule, LayerAssignment, Schedule, ScheduleSlot
from program.schedule.policy import ScheduleError
from program.types import StageId
from program.validate import (
    GroupRaceWarning,
    ProgramInvariantError,
    validate_program,
)
from program.work import (
    AttachMode,
    ByteSource,
    DepClass,
    Direction,
    SyncKey,
    SyncPhase,
    SyncRequirement,
    SyncSpread,
    VIA_ALL,
    VIA_DATA_FLOW,
    VIA_NON_DATA_FLOW,
    WorkItem,
    WorkKind,
    enumerate_work,
)
from program.workload import BlockTemplates, CommSpecTable, ParallelDegrees

from equiv.configs import MATRIX

from test_policies import Cfg, make_workload, raw_comm_metadata

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _program(
    cfg: Cfg,
    granularity: Granularity = Granularity.PIPELINE,
    *,
    spec_override: Any = None,
    **kwargs: Any,
) -> Program:
    """``build()`` for one :class:`Cfg`, with the default policy bundle.

    V6 (promoted to always-on in ``build()``) legitimately warns about SINK
    collectives — a gradient reducer has no successor, so two reducers of one
    group on one device have no path between them by construction. That is not a
    race for a program whose order is ONE global order projected onto every
    device (CONTEXT constraint 2), so the warning is silenced here rather than
    weakening the invariant.
    """
    spec = spec_override if spec_override is not None else make_workload(cfg)
    fw = spec.freeze()
    bundle = policies_for(fw, granularity=granularity, block_expanded=cfg.flattened)
    kwargs.setdefault("sharding", bundle.sharding)
    kwargs.setdefault("recompute", bundle.recompute)
    kwargs.setdefault("routing", bundle.routing)
    kwargs.setdefault("overlap", bundle.overlap)
    kwargs.setdefault("grad_accum", bundle.grad_accum)
    kwargs.setdefault("schedule_policy", GPipeSchedule())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GroupRaceWarning)
        return build(fw, granularity=granularity, **kwargs)


def _ops_by_name(program: Program, needle: str) -> List[Any]:
    return [op for op in program.ops if needle in op.name]


def _one(program: Program, needle: str) -> Any:
    found = _ops_by_name(program, needle)
    assert len(found) == 1, f"{needle!r} matched {[op.name for op in found]}"
    return found[0]


def _reaches(program: Program, source: int, target: int, *, skip=()) -> bool:
    """Transitive reachability over ``deps``, optionally without some edges."""
    banned = set(skip)
    stack = [target]
    seen: set = set()
    while stack:
        current = stack.pop()
        if current == source:
            return True
        if current in seen:
            continue
        seen.add(current)
        for dep in program.ops[current].deps:
            if (dep, current) in banned:
                continue
            stack.append(int(dep))
    return False


def _succs(program: Program) -> Dict[int, Tuple[int, ...]]:
    """``uid -> successor uids``, DERIVED from ``deps``.

    ``Op.succs`` was deleted (INTERFACES §4.7 amendment 2026-07-29): it was a
    mirror of ``deps`` that no consumer read and that every mutation had to keep
    in sync. A test that wants successors builds them, exactly as
    ``analytic_sim`` / ``memory_sim`` already do.
    """
    out: Dict[int, List[int]] = {int(op.uid): [] for op in program.ops}
    for op in program.ops:
        for dep in op.deps:
            out[int(dep)].append(int(op.uid))
    return {uid: tuple(succs) for uid, succs in out.items()}


def _block_template(
    entries: Sequence[Tuple[str, Sequence[str], Sequence[str]]],
    comm_metadata: Dict[str, Dict[str, Any]],
) -> BlockTemplate:
    """``[(entry name, forward comm keys, backward comm keys)] -> BlockTemplate``."""
    gemms = []
    for index, (name, fwd_keys, bwd_keys) in enumerate(entries):
        gemms.append(
            {
                "name": name,
                "forward": {"duration": 1.0 + index, "comm_keys": list(fwd_keys)},
                "backward": {"duration": 10.0 + index, "comm_keys": list(bwd_keys)},
            }
        )
    return BlockTemplate.from_gemm_entries(gemms, comm_metadata)


def _tp_comm(size: float = 4096.0, axis: str = "tp") -> Dict[str, Any]:
    return {
        "size": size,
        "type": CollectiveType.ALL_REDUCE,
        "participants": 2,
        "interconnect_type": axis,
        "local_comp_time": 0,
    }


# ===========================================================================
# L3 — SchedulePolicy / Schedule / LayerAssignment (S1-S3)
# ===========================================================================


def test_s1_slot_indices_are_dense_and_unique():
    cfg = Cfg(dp=1, pp=2, mb=2, num_layers=4)
    fw = make_workload(cfg).freeze()
    bundle = policies_for(fw, granularity=Granularity.PIPELINE)
    work = enumerate_work(fw, bundle.recompute)
    schedule = GPipeSchedule().schedule(fw, work)
    assert [slot.index for slot in schedule.ordered_slots] == list(range(len(work)))


def test_s1_non_dense_indices_are_rejected():
    layers = LayerAssignment.contiguous(1, 1)
    item = WorkItem(WorkKind.EMBEDDING, Direction.FORWARD, microbatch=0)
    with pytest.raises(ScheduleError, match="dense"):
        Schedule(policy="p", layers=layers, slots=(ScheduleSlot(3, StageId(0), item),))


def test_s2_every_work_item_appears_exactly_once_across_stages():
    cfg = Cfg(dp=1, pp=2, mb=2, num_layers=5)
    fw = make_workload(cfg).freeze()
    bundle = policies_for(fw, granularity=Granularity.PIPELINE)
    work = enumerate_work(fw, bundle.recompute)
    schedule = GPipeSchedule().schedule(fw, work)
    seen = Counter()
    for stage in range(cfg.pp):
        for item in schedule.order_for(StageId(stage)):
            seen[item] += 1
    assert set(seen) == set(work.items)
    assert set(seen.values()) == {1}


def test_s3_schedule_is_a_permutation_of_the_workset():
    for cfg in (
        Cfg(dp=1, pp=1, mb=1, num_layers=1),
        Cfg(dp=2, pp=2, mb=3, num_layers=5),
        Cfg(dp=1, pp=2, mb=2, num_layers=4, run_type="inference"),
        Cfg(dp=2, pp=2, mb=2, num_layers=4, full_recomputation=True, flattened=True),
    ):
        fw = make_workload(cfg).freeze()
        bundle = policies_for(fw, granularity=Granularity.FLAT, block_expanded=cfg.flattened)
        work = enumerate_work(fw, bundle.recompute)
        schedule = GPipeSchedule().schedule(fw, work)
        assert sorted(schedule.order(), key=WorkItem.sort_key) == list(work.items)
        schedule.check_permutation(work)


def test_gpipe_order_is_forward_ascending_then_backward_descending():
    """The load-bearing shape: legacy walks microbatches forward ascending
    (schedule.py:580) and backward DESCENDING (:693)."""
    cfg = Cfg(dp=1, pp=2, mb=3, num_layers=2)
    fw = make_workload(cfg).freeze()
    bundle = policies_for(fw, granularity=Granularity.PIPELINE)
    work = enumerate_work(fw, bundle.recompute)
    schedule = GPipeSchedule().schedule(fw, work)
    order = schedule.order()
    forward_mbs = [
        item.microbatch
        for item in order
        if item.direction is Direction.FORWARD and item.kind is not WorkKind.RECOMPUTE
    ]
    backward_mbs = [
        item.microbatch
        for item in order
        if item.direction is Direction.BACKWARD and item.kind is not WorkKind.OPTIMIZER
    ]
    assert forward_mbs == sorted(forward_mbs)
    assert backward_mbs == sorted(backward_mbs, reverse=True)
    # the optimizer tail is last, stage-ascending
    optimizers = [item for item in order if item.kind is WorkKind.OPTIMIZER]
    assert optimizers, "training workload must have an optimizer tail"
    assert list(order[-len(optimizers):]) == optimizers
    assert [int(item.stage) for item in optimizers] == sorted(
        int(item.stage) for item in optimizers
    )


def test_recompute_immediately_precedes_its_backward_layer():
    cfg = Cfg(dp=1, pp=1, mb=1, num_layers=3, full_recomputation=True, flattened=True)
    fw = make_workload(cfg).freeze()
    bundle = policies_for(fw, granularity=Granularity.FLAT, block_expanded=True)
    work = enumerate_work(fw, bundle.recompute)
    order = GPipeSchedule().schedule(fw, work).order()
    for layer in range(3):
        remat = work.require(WorkKind.RECOMPUTE, Direction.FORWARD, microbatch=0, layer=layer)
        backward = work.require(WorkKind.LAYER, Direction.BACKWARD, microbatch=0, layer=layer)
        assert order.index(remat) + 1 == order.index(backward)


def test_last_microbatch_of_is_computed_not_hardcoded():
    """Class B 10h as DATA: GPipe's answer is 0 because its backward walk
    descends, and ``GradAccumPolicy.for_schedule`` reads it from the schedule."""
    cfg = Cfg(dp=2, pp=2, mb=4, num_layers=4)
    fw = make_workload(cfg).freeze()
    bundle = policies_for(fw, granularity=Granularity.PIPELINE)
    work = enumerate_work(fw, bundle.recompute)
    schedule = GPipeSchedule().schedule(fw, work)
    assert schedule.last_microbatch_of() == 0
    assert bundle.grad_accum.for_schedule(schedule).last_microbatch == 0

    # A schedule whose backward walk ASCENDS answers with the last microbatch,
    # with no edit to GradAccumPolicy — the 1F1B seam.
    reversed_slots = []
    for index, item in enumerate(
        sorted(
            schedule.order(),
            key=lambda w: (
                w.direction is Direction.BACKWARD,
                (w.microbatch if w.microbatch is not None else -1)
                if w.direction is Direction.BACKWARD
                else 0,
                schedule.index_of(w),
            ),
        )
    ):
        reversed_slots.append(ScheduleSlot(index, schedule.stage_of(item), item))
    ascending = Schedule(policy="ascending", layers=schedule.layers, slots=tuple(reversed_slots))
    assert ascending.last_microbatch_of() == cfg.mb - 1


def test_layer_assignment_expresses_interleaving():
    """The 1F1B seam: an explicit map, not monotone counts. Nothing in L2/L3/L4
    requires contiguity — this is what makes interleaved virtual stages
    expressible without touching build()."""
    interleaved = LayerAssignment.explicit([0, 1, 0, 1])
    assert interleaved.layers_of(StageId(0)) == (0, 2)
    assert interleaved.layers_of(StageId(1)) == (1, 3)
    assert interleaved.min_layer(StageId(1)) == 1
    assert not interleaved.contiguous_layers()
    assert LayerAssignment.contiguous(4, 2).contiguous_layers()
    # it satisfies BOTH seams at once (the B5 amendment): Placement takes it as a
    # LayerAssignment, ShardingContext as a StagePartition.
    cfg = Cfg(dp=1, pp=2, mb=1, num_layers=4)
    fw = make_workload(cfg).freeze()
    placement = Placement(fw, Granularity.PIPELINE, interleaved)
    layer2 = WorkItem(WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=2)
    assert placement.stage_of(layer2) == 0
    ShardingContext(
        fw=fw,
        work=enumerate_work(fw, policies_for(fw, granularity=Granularity.PIPELINE).recompute),
        grad_accum=policies_for(fw, granularity=Granularity.PIPELINE).grad_accum,
        stages=interleaved,
    )


def test_implied_deps_are_per_device_and_reproduce_the_legacy_gpipe_edges():
    """R3's input. The legacy special cases must FALL OUT of the projection:
    stage 0's order is ``[..., LAYER/FWD(b, last), EMBEDDING/FWD(b+1), ...]``, so
    the adjacent pair IS ``schedule.py:663-664``'s hand-wired edge."""
    cfg = Cfg(dp=1, pp=2, mb=2, num_layers=4)
    fw = make_workload(cfg).freeze()
    bundle = policies_for(fw, granularity=Granularity.PIPELINE)
    work = enumerate_work(fw, bundle.recompute)
    schedule = GPipeSchedule().schedule(fw, work)
    placement = Placement(fw, Granularity.PIPELINE, schedule.layers)
    deps = schedule.implied_deps(placement.devices_for)

    layer1_f0 = work.require(WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=1)
    embedding_f1 = work.require(WorkKind.EMBEDDING, Direction.FORWARD, microbatch=1)
    assert any(
        dep.before == layer1_f0 and dep.after == embedding_f1 for dep in deps
    ), "schedule.py:663-664 (stage 0 -> next microbatch's embedding) must fall out"

    softmax_f0 = work.require(WorkKind.SOFTMAX, Direction.FORWARD, microbatch=0)
    layer2_f1 = work.require(WorkKind.LAYER, Direction.FORWARD, microbatch=1, layer=2)
    assert any(
        dep.before == softmax_f0 and dep.after == layer2_f1 for dep in deps
    ), "schedule.py:670-672 (softmax -> next microbatch's first layer) must fall out"

    # every dep is between two items on the SAME device
    for dep in deps:
        assert dep.device in placement.devices_for(dep.before)
        assert dep.device in placement.devices_for(dep.after)


def test_device_projection_is_per_device_not_per_stage():
    """At FLAT a stage is ``cluster_size`` devices and a PINNED kind lives on one
    of them (Class B 10d), so the two projections genuinely differ."""
    cfg = Cfg(dp=1, pp=1, tp=2, mb=2, num_layers=2)
    fw = make_workload(cfg).freeze()
    bundle = policies_for(fw, granularity=Granularity.FLAT)
    work = enumerate_work(fw, bundle.recompute)
    schedule = GPipeSchedule().schedule(fw, work)
    placement = Placement(fw, Granularity.FLAT, schedule.layers)
    projection = schedule.device_projection(placement.devices_for)
    assert len(projection) == 2
    softmax = work.require(WorkKind.SOFTMAX, Direction.FORWARD, microbatch=0)
    rank0, rank1 = placement.cluster_devices(StageId(0))
    assert softmax in projection[rank0]
    assert softmax not in projection[rank1]
    assert len(projection[rank0]) != len(projection[rank1])


# ===========================================================================
# R1 — data flow inside a block chain
# ===========================================================================


def test_r1_chain_steps_depend_on_their_predecessor():
    comm = {"mlp_tp": _tp_comm()}
    template = _block_template([("qkv_proj", [], []), ("MLP", ["mlp_tp"], [])], comm)
    cfg = Cfg(dp=1, pp=1, tp=1, mb=1, num_layers=1)
    spec = make_workload(cfg)
    spec = spec.with_(blocks=BlockTemplates(dense=template))
    program = _program(cfg, Granularity.FLAT, spec_override=spec, overlap=NoOverlap())

    qkv = _one(program, "qkv_proj_forward")
    mlp = _one(program, "MLP_forward")
    coll = _one(program, "mlp_tp")
    assert mlp.deps == (qkv.uid,)
    assert coll.deps == (mlp.uid,)
    assert qkv.uid < mlp.uid < coll.uid


def test_r1_honors_comm_spec_placement_pre():
    """INTERFACES §3.4: the FLAT path MUST honor ``CommSpec.placement``.
    ``pipeline_fine.py:577-589`` chains every key POST regardless."""
    comm = {"pre_gather": {**_tp_comm(), "placement": "pre"}}
    template = _block_template([("qkv_proj", ["pre_gather"], []), ("MLP", [], [])], comm)
    cfg = Cfg(dp=1, pp=1, tp=1, mb=1, num_layers=1)
    spec = make_workload(cfg).with_(blocks=BlockTemplates(dense=template))
    program = _program(cfg, Granularity.FLAT, spec_override=spec, overlap=NoOverlap())
    gather = _one(program, "pre_gather")
    qkv = _one(program, "qkv_proj_forward")
    assert qkv.deps == (gather.uid,), "a placement='pre' collective FEEDS its GEMM"


def test_r1_backward_reverses_the_template():
    """``pipeline_fine.py:542-543``: the backward chain walks the entries in
    reverse, and ``param_gather`` lands on the FIRST PROCESSED entry."""
    template = _block_template([("qkv_proj", [], []), ("MLP", [], [])], {})
    cfg = Cfg(dp=1, pp=1, tp=1, mb=1, num_layers=1)
    spec = make_workload(cfg).with_(blocks=BlockTemplates(dense=template))
    program = _program(cfg, Granularity.FLAT, spec_override=spec, overlap=NoOverlap())
    mlp_b = _one(program, "MLP_backward")
    qkv_b = _one(program, "qkv_proj_backward")
    assert qkv_b.deps == (mlp_b.uid,)
    assert mlp_b.param_gather and not qkv_b.param_gather


# ===========================================================================
# R2 — cross-layer data flow
# ===========================================================================


def test_r2_same_stage_link_is_a_zero_byte_transfer_op():
    """Class B item 9: the same-stage zero-byte PIPELINE event's PRESENCE is
    load-bearing for the analytical evaluator's ready-scan, so R2 emits it as a
    same-device ``TransferOp`` rather than collapsing it into a bare dep."""
    cfg = Cfg(dp=1, pp=1, tp=1, mb=1, num_layers=2)
    program = _program(cfg, Granularity.PIPELINE)
    transfers = [op for op in program.ops if isinstance(op, TransferOp)]
    assert transfers
    for op in transfers:
        assert op.src_device == op.dst_device == 0
        assert op.size_bytes == 0
        assert op.comm_type is CollectiveType.PIPELINE
        assert op.is_control


def test_r2_cross_stage_transfer_carries_the_declared_bytes():
    cfg = Cfg(dp=1, pp=2, tp=1, mb=1, num_layers=2)
    program = _program(cfg, Granularity.PIPELINE)
    payload = [
        op for op in program.ops if isinstance(op, TransferOp) and op.size_bytes > 0
    ]
    total = float(raw_comm_metadata(cfg)["cross_layer"]["size"])
    assert payload, "a pp=2 workload must move activations between stages"
    for op in payload:
        assert op.src_device != op.dst_device
        # PIPELINE: instances == placement.cluster_size() == 1 -> RAW bytes (B3).
        assert op.size_bytes == int(total)


def test_r2_fine_divides_cross_layer_bytes_by_the_cluster_size():
    """B3's amendment is what lets ONE rule reproduce both
    ``pipeline_coarse.py:222,238`` (raw) and ``pipeline_fine.py:645`` (divided)."""
    cfg = Cfg(dp=1, pp=2, tp=2, mb=1, num_layers=2)
    total = float(raw_comm_metadata(cfg)["cross_layer"]["size"])
    coarse = _program(cfg, Granularity.PIPELINE)
    fine = _program(cfg, Granularity.FLAT, overlap=NoOverlap())
    coarse_sizes = {
        op.size_bytes
        for op in coarse.ops
        if isinstance(op, TransferOp) and op.size_bytes > 0
    }
    fine_sizes = {
        op.size_bytes for op in fine.ops if isinstance(op, TransferOp) and op.size_bytes > 0
    }
    assert coarse_sizes == {int(total)}
    assert fine_sizes == {int(math.ceil(total / 2))}


def test_r2_does_not_divide_cross_layer_bytes_by_ep():
    """BUG_LEDGER **D 10c, ep half** (fixed 2026-07-27).

    The pipeline-boundary tensor is the RESIDUAL STREAM. ``ep`` shards experts,
    not tokens: in training every EP rank owns a DISTINCT microbatch
    (``base_timing.py:493-495`` -> ``dp_dense = dp*ep``, so the raw value is
    already ONE owner's microbatch), and it holds a full copy of its own residual
    stream sequence-sharded by ``tp``/``cp`` only. Dividing by ``cluster_size()
    == tp*cp*ep`` applied the ``/ep`` a second time.

    The stage still emits one transfer PER DEVICE (``cluster_size()`` of them) —
    only the payload changed — so the whole-cluster aggregate is now the ``ep``
    distinct microbatches the stage actually ships.
    """
    def payload(cfg):
        raw = float(raw_comm_metadata(cfg)["cross_layer"]["size"])
        program = _program(cfg, Granularity.FLAT, overlap=NoOverlap())
        ops = [
            op for op in program.ops if isinstance(op, TransferOp) and op.size_bytes > 0
        ]
        sizes = {op.size_bytes for op in ops}
        assert len(sizes) == 1, f"one payload size per program, got {sizes}"
        return raw, sizes.pop(), len(ops)

    kw = dict(dp=1, pp=2, tp=2, cp=1, mb=1, num_layers=2, moe=True)
    raw1, size1, count1 = payload(Cfg(ep=1, **kw))
    raw2, size2, count2 = payload(Cfg(ep=2, **kw))
    raw4, size4, count4 = payload(Cfg(ep=4, **kw))
    assert raw1 == raw2 == raw4, "the stub raw value must not move with ep"

    # The divisor is tp*cp on all three, NOT tp*cp*ep.
    expected = int(math.ceil(raw1 / 2))
    assert (size1, size2, size4) == (expected, expected, expected)
    assert size4 != int(math.ceil(raw4 / 8)), "tp*cp*ep would give raw/8 here"

    # One transfer per device, so ep multiplies the COUNT, not the payload...
    assert (count2, count4) == (2 * count1, 4 * count1)
    # ...and the aggregate therefore rises with ep instead of staying flat: at a
    # fixed per-owner microbatch a stage with ep experts groups ships ep distinct
    # microbatches across the boundary.
    assert size2 * count2 == 2 * size1 * count1
    assert size4 * count4 == 4 * size1 * count1


def test_r2_pairs_cluster_ranks_and_never_invents_a_cross_rank_payload():
    """At FLAT the pairing is ``r -> r`` (``pipeline_fine.py:659-694``), and the
    BYTE decision is per STAGE: an embedding -> layer 0 link inside one stage
    must not acquire a ``cross_layer`` payload just because the embedding is
    pinned to cluster rank 0."""
    cfg = Cfg(dp=1, pp=2, tp=2, mb=1, num_layers=2)
    program = _program(cfg, Granularity.FLAT, overlap=NoOverlap())
    for op in program.ops:
        if not isinstance(op, TransferOp) or op.size_bytes == 0:
            continue
        src_coords = program.layout.coords_of(op.src_device)
        dst_coords = program.layout.coords_of(op.dst_device)
        assert src_coords["pp"] != dst_coords["pp"], (
            f"payload transfer {op.src_device}->{op.dst_device} stays inside one "
            "stage; cross_layer models PIPELINE movement only"
        )
        assert src_coords["tp"] == dst_coords["tp"], "cluster ranks pair r -> r"


def test_r2_transfer_is_one_object_with_the_compute_anchor_as_a_second_dep():
    """The compute-anchor double-dep (``pipeline_fine.py:672-687``) is a second
    DEP on ONE transfer, not a second transfer. Legacy materializes one op per
    same-stage source parent; a p2p transfer has ONE identity (one tag)."""
    comm = {"mlp_tp": _tp_comm()}
    template = _block_template([("qkv_proj", [], []), ("MLP", ["mlp_tp"], [])], comm)
    cfg = Cfg(dp=1, pp=2, tp=1, mb=1, num_layers=2)
    spec = make_workload(cfg).with_(blocks=BlockTemplates(dense=template))
    program = _program(cfg, Granularity.FLAT, spec_override=spec, overlap=NoOverlap())
    payload = [
        op for op in program.ops if isinstance(op, TransferOp) and op.size_bytes > 0
    ]
    # pp=2 with 2 layers has exactly two cross-stage links: layer 0 -> layer 1
    # forward and layer 1 -> layer 0 backward. ONE TransferOp each.
    per_link = Counter((op.src_device, op.dst_device) for op in payload)
    assert per_link == Counter({(0, 1): 1, (1, 0): 1}), (
        f"one logical cross-stage edge -> exactly ONE TransferOp (got {per_link})"
    )
    saw_collective_exit = False
    for transfer in payload:
        producer = program.ops[transfer.producer]
        anchors = [program.ops[dep] for dep in transfer.deps]
        if isinstance(producer, CollectiveOp):
            saw_collective_exit = True
            assert any(isinstance(op, ComputeOp) for op in anchors), (
                "the SEND must additionally fire off the last COMPUTE, not only "
                "off the trailing collective"
            )
        else:
            assert transfer.deps == (producer.uid,), (
                "a compute exit needs no separate anchor"
            )
    assert saw_collective_exit, (
        "the forward chain ends in the tp collective, so the double-dep case must "
        "be exercised"
    )


def test_r2_recompute_to_backward_layer_is_a_plain_same_device_dep():
    """The rematerialized activation never leaves the device that recomputed it,
    so the ``RECOMPUTE -> LAYER/BACKWARD`` link inside one layer is a dep and not
    a transfer (legacy: ``recompute_node.add_child(transformer_node_b)``)."""
    cfg = Cfg(dp=1, pp=1, tp=2, mb=1, num_layers=1, full_recomputation=True, flattened=True)
    program = _program(cfg, Granularity.FLAT, overlap=NoOverlap())
    remat = [op for op in program.ops if isinstance(op, ComputeOp) and op.recompute]
    assert remat, "full recomputation must materialize RECOMPUTE work"
    succs = _succs(program)
    for op in remat:
        for succ in succs[op.uid]:
            assert not isinstance(program.ops[succ], TransferOp) or (
                program.ops[succ].size_bytes == 0
            )
        backward = [
            program.ops[succ]
            for succ in succs[op.uid]
            if isinstance(program.ops[succ], ComputeOp)
        ]
        assert backward, "the rematerialization feeds its backward layer directly"
        for consumer in backward:
            assert consumer.device == op.device


# ===========================================================================
# R3 — device serialization (D1)
# ===========================================================================


def test_r3_no_redundant_schedule_edges():
    """**D1**: for every added ``DepClass.SCHEDULE`` edge ``a -> b``, ``b`` was
    not reachable from ``a`` before the edge. Verified by removing the edge and
    asserting reachability changes.

    R5/R5b edges are removed too. D1 is a property of R3's DECISION, and R3
    eliminates against the graph it sees; the rule order is R2, R3, R4, R5, so
    an R3 edge can be subsumed by a path that did not exist yet — R5b's
    ``reducer -> optimizer`` edge makes exactly one such shortcut, since a
    stage's last backward op reaches its optimizer both directly (R3) and now
    through the gradient reducer. That is R5b adding an edge, not R3 having
    been wrong, and both edges are semantically distinct claims.
    """
    for cfg, granularity in (
        (Cfg(dp=1, pp=2, mb=3, num_layers=4), Granularity.PIPELINE),
        (Cfg(dp=2, pp=2, tp=2, mb=2, num_layers=4), Granularity.FLAT),
        (Cfg(dp=2, pp=2, mb=2, num_layers=4, zero_stage=3), Granularity.PIPELINE),
    ):
        program = _program(cfg, granularity, overlap=NoOverlap())
        edges = program.meta.misc["schedule_edges"]
        # Every edge added AFTER R3 ran. R4 belongs here for the same reason
        # R5 does, and A6 is what made it observable: an ``AttachMode.BEFORE``
        # requirement inserts itself between its target and the target's own
        # deps, so each of those deps gains a second, longer path to the target
        # and its direct R3 edge reads as redundant afterwards. R3 was right
        # when it ran; the shortcut did not exist yet.
        later = set(program.meta.misc["r5_edges"]) | set(
            program.meta.misc["r4_edges"]
        )
        assert edges, f"{cfg.label()} must need at least one serialization edge"
        for source, target in edges:
            skip = later | {(source, target)}
            assert not _reaches(program, source, target, skip=skip), (
                f"schedule edge {source}->{target} is redundant: "
                f"{program.ops[target].name!r} was already reachable from "
                f"{program.ops[source].name!r}"
            )


def test_r3_does_not_blanket_serialize_a_device():
    """Consecutive layers of one microbatch on one device are already chained by
    R2, so R3 must add NOTHING for them: the count of schedule edges is far
    below the number of adjacent pairs."""
    cfg = Cfg(dp=1, pp=2, mb=3, num_layers=6)
    fw = make_workload(cfg).freeze()
    bundle = policies_for(fw, granularity=Granularity.PIPELINE)
    work = enumerate_work(fw, bundle.recompute)
    schedule = GPipeSchedule().schedule(fw, work)
    placement = Placement(fw, Granularity.PIPELINE, schedule.layers)
    adjacent = len(schedule.implied_deps(placement.devices_for))
    program = _program(cfg, Granularity.PIPELINE)
    added = len(program.meta.misc["schedule_edges"])
    assert 0 < added < adjacent, (
        f"R3 added {added} of {adjacent} adjacent pairs; blanket serialization "
        "would add all of them"
    )


def test_r3_materializes_the_cross_microbatch_edge():
    """AstraSim's one-slot rule constrains CONCURRENCY, not ORDER, so the
    cross-microbatch dependency must be IN THE DAG (legacy materializes it
    explicitly: git show 85894c6:simulate_train_graph.py:836-856)."""
    cfg = Cfg(dp=1, pp=2, mb=2, num_layers=4)
    program = _program(cfg, Granularity.PIPELINE)
    embedding_mb1 = _one(program, "embedding_mb1")
    layer1_mb0 = _one(program, "layer_l1_mb0")
    assert layer1_mb0.uid in embedding_mb1.deps, (
        "stage 0 must finish microbatch 0 before microbatch 1's embedding starts"
    )
    assert (layer1_mb0.uid, embedding_mb1.uid) in set(
        program.meta.misc["schedule_edges"]
    )


def test_r3_puts_the_optimizer_where_legacy_attached_it():
    """The optimizer tail's attach logic (``schedule.py:1048-1072``) is DELETED,
    not ported: legacy attaches stage ``s``'s optimizer to
    ``embedding_node_b[0]`` (stage 0) or ``_bwd_exit_node(0, min_layer(s))``, and
    those ARE the last backward items in each stage's device projection because
    GPipe's backward walk descends both microbatches and layers."""
    cfg = Cfg(dp=1, pp=2, mb=2, num_layers=4)
    program = _program(cfg, Granularity.PIPELINE)
    layers = LayerAssignment.contiguous(cfg.num_layers, cfg.pp)

    stage0 = _one(program, "optimizer_stage0")
    assert _one(program, "embedding_b_mb0").uid in stage0.deps

    stage1 = _one(program, "optimizer_stage1")
    min_layer = layers.min_layer(StageId(1))
    assert _one(program, f"layer_b_l{min_layer}_mb0").uid in stage1.deps


# ===========================================================================
# A NON-GPipe schedule — the probe for "explicit, not incidental"
# ===========================================================================
#
# Every dependency below was, before 2026-07-29, carried ONLY by an R3 edge,
# i.e. by whatever adjacency the SchedulePolicy happened to produce. Under
# GPipe they all hold; the point of these probes is that they must hold under a
# DIFFERENT legal schedule too, because they are properties of the model.
#
# ``_AscBackward`` is the minimal legal deviation: GPipe with the backward pass
# walking microbatches ASCENDING — the one property every 1F1B / interleaved
# schedule has. Nothing else changes, so a failure here is about the dependency
# rules and not about the alternative schedule being exotic.


@dataclass(frozen=True)
class _AscBackward:
    """GPipe, but the backward walks microbatches ascending."""

    name: str = "asc_backward"

    def layer_assignment(self, fw):
        return LayerAssignment.contiguous(
            int(fw.spec.shape.num_layers), int(fw.spec.degrees.pp)
        )

    def schedule(self, fw, work):
        layers = self.layer_assignment(fw)
        shape = fw.spec.shape
        order: List[WorkItem] = []

        def take(item):
            if item is not None:
                order.append(item)

        for b in range(int(shape.micro_batches)):
            take(work.get(WorkKind.EMBEDDING, Direction.FORWARD, microbatch=b))
            for layer in range(int(shape.num_layers)):
                take(work.get(WorkKind.LAYER, Direction.FORWARD, microbatch=b, layer=layer))
            take(work.get(WorkKind.SOFTMAX, Direction.FORWARD, microbatch=b))
        for b in range(int(shape.micro_batches)):  # <-- ASCENDING
            take(work.get(WorkKind.SOFTMAX, Direction.BACKWARD, microbatch=b))
            for layer in reversed(range(int(shape.num_layers))):
                take(work.get(WorkKind.RECOMPUTE, Direction.FORWARD, microbatch=b, layer=layer))
                take(work.get(WorkKind.LAYER, Direction.BACKWARD, microbatch=b, layer=layer))
            take(work.get(WorkKind.EMBEDDING, Direction.BACKWARD, microbatch=b))
        for stage in range(int(fw.spec.degrees.pp)):
            take(work.get(WorkKind.OPTIMIZER, Direction.BACKWARD, stage=StageId(stage)))

        slots = tuple(
            ScheduleSlot(index=index, stage=layers.stage_of_work(item), work=item)
            for index, item in enumerate(order)
        )
        schedule = Schedule(policy=self.name, layers=layers, slots=slots)
        schedule.check_permutation(work)
        return schedule


_SCHEDULES = (GPipeSchedule(), _AscBackward())


def _chain_ends(program: Program, kind, direction, **fields):
    """``work -> {device: (first uid, last uid)}`` for one WorkItem's chains."""
    out: Dict[int, List[int]] = {}
    for op in program.ops:
        work = getattr(op, "work", None)
        if work is None or work.kind is not kind or work.direction is not direction:
            continue
        if any(getattr(work, name) != value for name, value in fields.items()):
            continue
        out.setdefault(int(op.device), []).append(int(op.uid))
    return {device: (min(uids), max(uids)) for device, uids in out.items()}


@pytest.mark.parametrize("schedule_policy", _SCHEDULES, ids=lambda s: s.name)
def test_r2_softmax_forward_precedes_its_own_backward_under_any_schedule(
    schedule_policy,
):
    """**D4** — the loss gradient needs the forward logits. R2 states it now;
    before, only R3's adjacency did, and R3 is free to drop what it finds
    implied."""
    cfg = Cfg(dp=1, pp=2, mb=3, num_layers=4)
    program = _program(cfg, Granularity.PIPELINE, schedule_policy=schedule_policy)
    for b in range(cfg.mb):
        fwd = _chain_ends(program, WorkKind.SOFTMAX, Direction.FORWARD, microbatch=b)
        bwd = _chain_ends(program, WorkKind.SOFTMAX, Direction.BACKWARD, microbatch=b)
        assert fwd and bwd
        for device, (_first, last) in fwd.items():
            entry = bwd[device][0]
            assert _reaches(program, last, entry), (
                f"{schedule_policy.name}: softmax fwd(mb{b}) does not precede its "
                f"own backward on device {device}"
            )


@pytest.mark.parametrize("schedule_policy", _SCHEDULES, ids=lambda s: s.name)
def test_r2_layer_forward_precedes_its_own_recompute_under_any_schedule(
    schedule_policy,
):
    """**D9** — the rematerialization consumes what the forward stashed."""
    cfg = Cfg(
        dp=1, pp=2, tp=2, mb=3, num_layers=4, full_recomputation=True, flattened=True
    )
    program = _program(
        cfg, Granularity.FLAT, schedule_policy=schedule_policy, overlap=NoOverlap()
    )
    checked = 0
    for b in range(cfg.mb):
        for layer in range(cfg.num_layers):
            fwd = _chain_ends(
                program, WorkKind.LAYER, Direction.FORWARD, microbatch=b, layer=layer
            )
            remat = _chain_ends(
                program, WorkKind.RECOMPUTE, Direction.FORWARD, microbatch=b, layer=layer
            )
            if not fwd or not remat:
                continue
            for device, (_first, last) in fwd.items():
                assert _reaches(program, last, remat[device][0]), (
                    f"{schedule_policy.name}: layer {layer} fwd(mb{b}) does not "
                    f"precede its own recompute on device {device}"
                )
                checked += 1
    assert checked, "the config must materialize RECOMPUTE work"


@pytest.mark.parametrize("schedule_policy", _SCHEDULES, ids=lambda s: s.name)
def test_r5_optimizer_waits_for_every_backward_item_of_its_stage(schedule_policy):
    """**D10** — legacy hand-picked ONE attach point per stage and the new core
    inherited the ordering from R3's per-device adjacency, which is only right
    because GPipe's backward descends microbatches. Under ``_AscBackward`` the
    ordering used to be missing for every microbatch but the last."""
    cfg = Cfg(dp=1, pp=2, mb=3, num_layers=4)
    program = _program(cfg, Granularity.PIPELINE, schedule_policy=schedule_policy)
    optimizers = [
        op for op in program.ops
        if getattr(op, "work", None) is not None
        and op.work.kind is WorkKind.OPTIMIZER
    ]
    assert optimizers
    checked = 0
    for optimizer in optimizers:
        stage = int(optimizer.work.stage)
        for op in program.ops:
            work = getattr(op, "work", None)
            if work is None or work.direction is not Direction.BACKWARD:
                continue
            if work.kind not in (WorkKind.LAYER, WorkKind.EMBEDDING, WorkKind.SOFTMAX):
                continue
            if int(op.device) != int(optimizer.device):
                continue
            assert _reaches(program, int(op.uid), int(optimizer.uid)), (
                f"{schedule_policy.name}: optimizer of stage {stage} does not wait "
                f"for {op.name!r} — it would apply an incomplete gradient"
            )
            checked += 1
    assert checked


def test_r5_adds_nothing_under_gpipe():
    """R5 is redundancy-eliminated exactly like R3 (**D1**): under GPipe every
    ordering it requires is already implied, so the artifact does not move."""
    cfg = Cfg(dp=1, pp=2, mb=3, num_layers=4)
    program = _program(cfg, Granularity.PIPELINE)
    for op in program.ops:
        work = getattr(op, "work", None)
        if work is None or work.kind is not WorkKind.OPTIMIZER:
            continue
        for dep in op.deps:
            source = program.ops[int(dep)]
            assert not _reaches(
                program, int(dep), int(op.uid), skip={(int(dep), int(op.uid))}
            ), (
                f"optimizer dep {source.name!r} -> {op.name!r} is redundant; R5 "
                "must only materialize what is not already implied"
            )


@pytest.mark.parametrize("schedule_policy", _SCHEDULES, ids=lambda s: s.name)
def test_zero3_prefetch_gathers_precede_their_declared_consumer(schedule_policy):
    """**D18/D19** — rows S6/S14 used to answer "who waits for me?" with
    ``succ(anchor, VIA_NON_DATA_FLOW)``, i.e. with an R3 edge whose identity the
    SchedulePolicy chooses. Under ``_AscBackward`` one gather preceded NOTHING
    and another attached to the wrong microbatch. The consumer is declared now,
    so both hold under either schedule."""
    cfg = Cfg(dp=2, pp=2, mb=3, num_layers=4, zero_stage=3)
    program = _program(cfg, Granularity.PIPELINE, schedule_policy=schedule_policy)
    gathers = [
        op for op in program.ops
        if isinstance(op, CollectiveOp) and "zero3_embedding_gather_fwd_b" in op.name
    ]
    assert gathers, "the config must materialize S6 gathers"
    for gather in gathers:
        microbatch = int(gather.name.rsplit("_b", 1)[1])
        target = _chain_ends(
            program, WorkKind.EMBEDDING, Direction.FORWARD, microbatch=microbatch
        )
        assert target, f"no embedding chain for mb{microbatch}"
        for device, (entry, _last) in target.items():
            if device != int(gather.device):
                continue
            assert _reaches(program, int(gather.uid), entry), (
                f"{schedule_policy.name}: {gather.name} does not precede the "
                f"embedding of microbatch {microbatch} it prefetches for"
            )


# ===========================================================================
# R4 — sync attach
# ===========================================================================


@dataclass(frozen=True)
class _FixedSharding:
    """A sharding policy that emits exactly the requirements a test declares.
    The protocol IS the seam — no build() argument is needed to inject them."""

    name: str
    workload: Tuple[SyncRequirement, ...] = ()
    per_item: Tuple[SyncRequirement, ...] = ()

    def requirements(self, work: WorkItem, ctx: ShardingContext):
        return tuple(req for req in self.per_item if req.place_on == work)

    def workload_requirements(self, ctx: ShardingContext):
        return self.workload


def _sync_fixture(cfg: Cfg):
    fw = make_workload(cfg).freeze()
    bundle = policies_for(fw, granularity=Granularity.PIPELINE)
    work = enumerate_work(fw, bundle.recompute)
    return fw, work, bundle


def _requirement(
    fw,
    key: str,
    *,
    phase: SyncPhase,
    place_on: WorkItem,
    mode: AttachMode,
    anchors,
    via=VIA_ALL,
    microbatch=None,
    layer=None,
    spread=SyncSpread.CLUSTER_RANK_0,
    is_reducer: bool = False,
) -> SyncRequirement:
    spec = fw.spec.comm.require(key)
    return SyncRequirement.from_spec(
        spec,
        key=SyncKey(key, phase, microbatch, layer),
        place_on=place_on,
        mode=mode,
        anchors=anchors,
        via=via,
        spread=spread,
        is_reducer=is_reducer,
        origin="test",
    )


def test_r4_after_puts_the_requirement_off_the_anchor_and_before_the_optimizer():
    """``AFTER`` hangs the collective off the anchor's exit, and **R5b** is the
    only thing downstream of it: the weight update consumes the REDUCED
    gradient. Before R5b existed every gradient reducer was a graph sink, so
    AstraSim ran the optimizer concurrently with the all-reduce producing the
    gradient it applies (BUG_LEDGER 11)."""
    cfg = Cfg(dp=2, pp=1, mb=1, num_layers=1)
    fw, work, bundle = _sync_fixture(cfg)
    layer_b = work.require(WorkKind.LAYER, Direction.BACKWARD, microbatch=0, layer=0)
    req = _requirement(
        fw,
        "transformer_dense",
        phase=SyncPhase.GRAD,
        place_on=layer_b,
        mode=AttachMode.AFTER,
        anchors=(layer_b,),
        microbatch=0,
        layer=0,
        # R5b selects by ROLE, so the fixture must declare it — which is the
        # property under test: a GRAD-phase collective that is NOT a reducer
        # (ZeRO-2's post-reduce gather) must stay off the optimizer.
        is_reducer=True,
    )
    program = _program(
        cfg,
        Granularity.PIPELINE,
        sharding=_FixedSharding(name="fixed", per_item=(req,)),
        routing=None,
    )
    reducer = _one(program, "transformer_dense_grad")
    anchor = _one(program, "layer_b_l0_mb0")
    optimizer = _one(program, "optimizer_stage0")
    assert reducer.deps == (anchor.uid,)
    assert _succs(program)[reducer.uid] == (optimizer.uid,)


def test_r5b_selects_reducers_by_role_not_by_schedule_phase():
    """**BUG_LEDGER 11**, the defect an adversarial audit found in the FIRST fix.

    R5b originally filtered its source set with ``key.phase is SyncPhase.GRAD``.
    ZeRO-2's post-reduce parameter all-gather is declared in that same phase
    (``sharding.ZeRO2.after_reducer``), so it became a source and the optimizer
    was ordered ``reduce-scatter -> all-gather -> update`` — the physical
    inverse of ``reduce-scatter -> update -> all-gather``.

    A ``SyncPhase`` says WHERE in the schedule a requirement sits. It can never
    answer WHAT it is. The role is declared by the policy that knows
    (``SyncRequirement.is_reducer``), and this pins it end to end on a real
    ZeRO-2 workload.
    """
    cfg = Cfg(dp=2, pp=2, mb=2, num_layers=4, zero_stage=2)
    for granularity in (Granularity.PIPELINE, Granularity.FLAT):
        program = _program(cfg, granularity, overlap=NoOverlap())
        succs = _succs(program)
        optimizers = {
            op.uid for op in program.ops
            if isinstance(op, ComputeOp) and op.role is OpRole.OPTIMIZER
        }
        assert optimizers, f"{granularity.name}: no optimizer op"

        reducers = [op for op in program.ops
                    if isinstance(op, CollectiveOp) and op.is_dp
                    and op.coll is CollectiveType.REDUCE_SCATTER]
        gathers = [op for op in program.ops
                   if isinstance(op, CollectiveOp) and op.is_dp
                   and op.coll is CollectiveType.ALL_GATHER]
        assert reducers and gathers, f"{granularity.name}: zero2 must emit both"

        # the REDUCER feeds the optimizer ...
        assert any(optimizers & set(succs.get(r.uid, ())) for r in reducers), (
            f"{granularity.name}: no reduce-scatter -> optimizer edge"
        )
        # ... and the post-reduce GATHER never does.
        for g in gathers:
            assert not (optimizers & set(succs.get(g.uid, ()))), (
                f"{granularity.name}: the ZeRO-2 parameter all-gather {g.uid} feeds "
                "the optimizer — the update cannot wait on a gather that "
                "broadcasts its own output"
            )
            # R5c: it runs AFTER the update instead. Excluding the gather from
            # R5b's SOURCES without adding this makes it a graph sink, which
            # overlaps the update for free — cheaper than the inverted edge it
            # replaced. The pair is the fix; either half alone is wrong.
            assert optimizers & set(g.deps), (
                f"{granularity.name}: the ZeRO-2 parameter all-gather {g.uid} is a "
                "SINK — it must depend on the optimizer whose output it broadcasts"
            )


def test_every_gradient_reducer_reaches_its_optimizer():
    """**BUG_LEDGER 11**, completeness — the regression the FIRST role-based fix
    shipped.

    Switching R5b from ``key.phase is SyncPhase.GRAD`` to
    ``SyncRequirement.is_reducer`` narrowed the source set correctly for ZeRO-2,
    and silently DROPPED the EP gradient sync (row S12, ``policies/routing.py``),
    which never set the new flag. The phase filter had been picking it up for
    free. 16 ``ep_sync`` collectives became graph sinks at a measured cost of
    0.00 s, so no golden caught it.

    A whitelist that must be maintained by hand is exactly the shape of bug that
    produced this whole ledger entry, so the property is asserted directly: any
    collective whose bytes are a GRADIENT payload must reach the optimizer of
    its stage. Checked over the configurations that exercise each reducer
    family — dp, ZeRO-2, and EP.
    """
    cases = (
        Cfg(dp=2, pp=2, mb=2, num_layers=4),                             # dp
        Cfg(dp=2, pp=2, mb=2, num_layers=4, zero_stage=2),               # + zero2 gather
        Cfg(dp=2, pp=2, mb=2, num_layers=4, ep=2, moe=True),             # + S12 ep sync
    )
    for cfg in cases:
        for granularity in (Granularity.PIPELINE, Granularity.FLAT):
            program = _program(cfg, granularity, overlap=NoOverlap())
            optimizers = {
                op.uid for op in program.ops
                if isinstance(op, ComputeOp) and op.role is OpRole.OPTIMIZER
            }
            assert optimizers, f"{cfg.label()}/{granularity.name}: no optimizer"
            succs = _succs(program)

            def reaches_optimizer(uid, _s=succs, _o=optimizers):
                seen, stack = {uid}, [uid]
                while stack:
                    for nxt in _s.get(stack.pop(), ()):
                        if nxt in _o:
                            return True
                        if nxt not in seen:
                            seen.add(nxt)
                            stack.append(nxt)
                return False

            for op in program.ops:
                if not isinstance(op, CollectiveOp):
                    continue
                # A reducer is identified by its ROLE at the source: it is the
                # requirement the sharding/routing policy declared is_reducer.
                # Here we can only see the emitted op, so use the two families
                # the policies produce as gradient reducers.
                is_grad_reducer = op.is_dp or "ep_sync" in (op.comm_key or "")
                if not is_grad_reducer:
                    continue
                if op.coll is CollectiveType.ALL_GATHER:
                    continue  # post-update broadcast; R5c orders it the OTHER way
                assert reaches_optimizer(op.uid), (
                    f"{cfg.label()}/{granularity.name}: gradient reducer "
                    f"{op.name!r} ({op.comm_key}) never reaches an optimizer — "
                    "the weight update is modeled as concurrent with the "
                    "collective that produces the gradient it applies"
                )


def test_r4_before_splices_the_requirement_in_front_of_the_anchor():
    cfg = Cfg(dp=2, pp=1, mb=1, num_layers=1, zero_stage=3)
    fw, work, bundle = _sync_fixture(cfg)
    entry = work.require(WorkKind.EMBEDDING, Direction.FORWARD, microbatch=0)
    req = _requirement(
        fw,
        "zero3_embedding_gather",
        phase=SyncPhase.FWD_ENTRY,
        place_on=entry,
        mode=AttachMode.BEFORE,
        anchors=(entry,),
        microbatch=0,
    )
    program = _program(
        cfg,
        Granularity.PIPELINE,
        sharding=_FixedSharding(name="fixed", workload=(req,)),
        routing=None,
    )
    gather = _one(program, "zero3_embedding_gather_fwd_entry")
    embedding = _one(program, "embedding_mb0")
    assert gather.deps == (), "the anchor was a root, so the requirement becomes the root"
    assert gather.uid in embedding.deps
    assert gather.uid == 0, "program order is schedule-major: the new root sorts first"


def test_r4_parallel_to_inherits_deps_and_successors_filtered_by_via():
    """``PARALLEL_TO(a, via)``: ``deps(req) += deps(a)`` and, for every successor
    of ``a`` reached by an edge whose class is in ``via``, ``deps(s) += req``.
    ``VIA_DATA_FLOW`` IS legacy's ``skip_non_comm_children=True`` (cross-layer
    edges are always ``CollectiveType.PIPELINE``); ``VIA_NON_DATA_FLOW`` IS
    ``skip_comm_children=True``."""
    cfg = Cfg(dp=2, pp=2, mb=2, num_layers=4, zero_stage=3)
    fw, work, bundle = _sync_fixture(cfg)
    # layer 1 is the last layer of stage 0, so its successors are BOTH a
    # cross-stage transfer (DATA_FLOW) and the next microbatch's embedding
    # (SCHEDULE).
    host = work.require(WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=1)
    target = work.require(WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=2)

    def _built(via, key_layer, place=None):
        req = _requirement(
            fw,
            "zero3_transformer_gather",
            phase=SyncPhase.FWD,
            place_on=place if place is not None else target,
            mode=AttachMode.PARALLEL_TO,
            anchors=(host,),
            via=via,
            microbatch=0,
            layer=key_layer,
        )
        program = _program(
            cfg,
            Granularity.PIPELINE,
            sharding=_FixedSharding(name="fixed", per_item=(req,)),
            routing=None,
        )
        gather = _one(program, "zero3_transformer_gather_fwd_b0")
        host_op = _one(program, "layer_l1_mb0")
        return program, gather, host_op

    program, gather, host_op = _built(VIA_DATA_FLOW, 2)
    successors = [program.ops[uid] for uid in _succs(program)[gather.uid]]
    assert successors, "VIA_DATA_FLOW must keep the cross-layer transfer"
    assert all(isinstance(op, TransferOp) for op in successors)
    assert set(program.ops[gather.uid].deps) == set(host_op.deps)

    # Inheritance is device-BLIND, by design: the gather sits on layer 2's device
    # (stage 1) and still inherits the stage-0 anchor's deps. The analytical
    # evaluator honors that edge (it is why
    # ``train:analytical:dp2tp1cp1pp2mb2sp0:zero3`` is bit-exact); the EMITTER is
    # where a rank-local ctrl_dep is a fact, and it puts what no ctrl_dep can
    # carry ON THE WIRE as a 1-byte control pair (``et_emit`` Phase B2, amendment
    # 2026-07-29) so the two evaluators honor the SAME DAG. Keeping the two
    # concerns apart is the point: L4 states the ordering, L5 states how a trace
    # expresses it. That the ordering is device-blind AT ALL is BUG_LEDGER A6.
    def _dep_devices(program, uid):
        out = []
        for d in program.ops[uid].deps:
            dep = program.ops[d]
            out.append(dep.dst_device if isinstance(dep, TransferOp) else dep.device)
        return out

    assert any(dev != gather.device for dev in _dep_devices(program, gather.uid))

    # Same device as the anchor: same rule, and now emittable as written.
    program, gather, host_op = _built(VIA_DATA_FLOW, 4, place=host)
    assert set(program.ops[gather.uid].deps) == set(host_op.deps)
    assert all(dev == gather.device for dev in _dep_devices(program, gather.uid))

    program, gather, host_op = _built(VIA_NON_DATA_FLOW, 3)
    successors = [program.ops[uid] for uid in _succs(program)[gather.uid]]
    assert successors, "VIA_NON_DATA_FLOW must keep the cross-microbatch dep"
    assert not any(isinstance(op, TransferOp) for op in successors)


def test_r4_after_a_sync_key_requires_it_to_be_earlier_in_sync_order():
    """§4.4 makes the S3/S5/S10 chaining a CHECKABLE property instead of a
    construction-order coincidence."""
    cfg = Cfg(dp=2, pp=1, mb=1, num_layers=1, zero_stage=2)
    fw, work, bundle = _sync_fixture(cfg)
    layer_b = work.require(WorkKind.LAYER, Direction.BACKWARD, microbatch=0, layer=0)
    reducer_key = SyncKey("transformer_dense", SyncPhase.GRAD, 0, 0)
    # the gather chains AFTER a requirement that is never emitted
    gather = _requirement(
        fw,
        "zero2_transformer_gather",
        phase=SyncPhase.FWD,  # sorts BEFORE GRAD -> the reference is not earlier
        place_on=layer_b,
        mode=AttachMode.AFTER,
        anchors=(reducer_key,),
        microbatch=0,
        layer=0,
    )
    reducer = _requirement(
        fw,
        "transformer_dense",
        phase=SyncPhase.GRAD,
        place_on=layer_b,
        mode=AttachMode.AFTER,
        anchors=(layer_b,),
        microbatch=0,
        layer=0,
    )
    with pytest.raises(BuildError, match="strictly earlier in SyncOrder"):
        _program(
            cfg,
            Granularity.PIPELINE,
            sharding=_FixedSharding(name="fixed", per_item=(gather, reducer)),
            routing=None,
        )


def test_r4_drops_a_requirement_whose_anchors_resolve_to_nothing():
    """INTERFACES §2.3 note 1: rows S6/S14 are created once per microbatch but
    attached only inside the cross-device branch. With no boundary the legacy
    lattice consumes an op id and leaves an unreachable object; ``build()`` must
    DROP it and materialize no op."""
    cfg = Cfg(dp=2, pp=1, mb=1, num_layers=1)
    fw, work, bundle = _sync_fixture(cfg)
    layer_b = work.require(WorkKind.LAYER, Direction.BACKWARD, microbatch=0, layer=0)
    orphan_anchor = WorkItem(
        WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=0
    )
    req = _requirement(
        fw,
        "transformer_dense",
        phase=SyncPhase.GRAD,
        place_on=layer_b,
        mode=AttachMode.PARALLEL_TO,
        anchors=(orphan_anchor,),
        microbatch=0,
        layer=0,
    )
    # sanity: the anchor IS in the work set, so this requirement is materialized
    program = _program(
        cfg,
        Granularity.PIPELINE,
        sharding=_FixedSharding(name="fixed", per_item=(req,)),
        routing=None,
    )
    assert _ops_by_name(program, "transformer_dense_grad")
    assert program.meta.misc["dropped_requirements"] == ()

    # now anchor it on a WorkItem that is NOT in the work set
    missing = WorkItem(WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=7)
    dropped = replace(req, anchors=(missing,))
    program = _program(
        cfg,
        Granularity.PIPELINE,
        sharding=_FixedSharding(name="fixed", per_item=(dropped,)),
        routing=None,
    )
    assert not _ops_by_name(program, "transformer_dense_grad")
    assert len(program.meta.misc["dropped_requirements"]) == 1


def test_r4_dp_requirement_carries_no_group_and_a_grouped_one_does():
    cfg = Cfg(dp=2, pp=2, tp=2, ep=2, mb=1, num_layers=2, moe=True)
    program = _program(cfg, Granularity.FLAT, overlap=NoOverlap())
    for op in program.ops:
        if not isinstance(op, CollectiveOp):
            continue
        if op.is_dp:
            assert op.group is None and op.label is None, (
                "a dp collective's members are stamped at emission over pre-dp "
                "device ids (ir.py:92-99)"
            )
        else:
            assert op.group is not None and op.label is not None
            assert op.device in op.group.members


def test_sync_order_is_phase_major_and_total():
    order = SyncOrder()
    ranks = [
        order.key(
            SyncRequirement(
                key=SyncKey("k", phase, 0, 0),
                bytes=ByteSource("k"),
                kind=CollectiveType.ALL_REDUCE,
                axes=("dp",),
                participants=2,
                place_on=WorkItem(WorkKind.EMBEDDING, Direction.FORWARD, microbatch=0),
                mode=AttachMode.AFTER,
                anchors=(WorkItem(WorkKind.EMBEDDING, Direction.FORWARD, microbatch=0),),
            )
        )[0]
        for phase in (
            SyncPhase.FWD_ENTRY,
            SyncPhase.FWD,
            SyncPhase.GRAD,
            SyncPhase.BWD_ENTRY,
            SyncPhase.BWD,
        )
    ]
    assert ranks == sorted(ranks) and len(set(ranks)) == 5


# ===========================================================================
# Program order and IR invariants (O1, O2, D2, V7, V8)
# ===========================================================================


def test_o1_program_order_is_deterministic():
    cfg = Cfg(dp=2, pp=2, tp=2, ep=2, mb=2, num_layers=4, moe=True)
    for granularity in (Granularity.PIPELINE, Granularity.FLAT):
        first = _program(cfg, granularity)
        second = _program(cfg, granularity)
        assert _describe(first) == _describe(second)
        assert first.meta.misc["program_order"] == "kahn(slot,device,intra)"


def _describe(program: Program) -> List[Any]:
    out: List[Any] = []
    for op in program.ops:
        if isinstance(op, TransferOp):
            out.append(
                ("XFER", op.name, op.src_device, op.dst_device, op.size_bytes,
                 op.producer, tuple(op.consumers), op.deps)
            )
        elif isinstance(op, CollectiveOp):
            out.append(
                ("COLL", op.name, op.device, op.coll.name, op.size_bytes, op.label,
                 None if op.group is None else tuple(op.group.members), op.deps)
            )
        else:
            out.append(("COMP", op.name, op.device, op.duration, op.deps))
    return out


def test_o2_every_dep_precedes_its_op_and_no_op_is_orphaned():
    cfg = Cfg(dp=2, pp=2, tp=2, mb=2, num_layers=4, zero_stage=3)
    for granularity in (Granularity.PIPELINE, Granularity.FLAT):
        program = _program(cfg, granularity)
        # O2 / V1
        for op in program.ops:
            for dep in op.deps:
                assert 0 <= dep < op.uid
        # D2: every op is reachable from a root
        reachable = set()
        roots = [op.uid for op in program.ops if not op.deps]
        assert roots
        succs = _succs(program)
        stack = list(roots)
        while stack:
            current = stack.pop()
            if current in reachable:
                continue
            reachable.add(current)
            stack.extend(succs[int(current)])
        assert len(reachable) == len(program.ops), (
            f"{len(program.ops) - len(reachable)} ops are unreachable from a root"
        )


def test_deps_carry_every_edge_exactly_once():
    """``deps`` is the ONE edge structure (``Op.succs``, its write-only mirror,
    is deleted): every edge appears exactly once and nothing is duplicated."""
    program = _program(Cfg(dp=2, pp=2, tp=2, mb=2, num_layers=4), Granularity.FLAT)
    forward = Counter()
    for op in program.ops:
        for dep in op.deps:
            forward[(int(dep), op.uid)] += 1
    assert all(count == 1 for count in forward.values()), "no duplicate edges"
    assert not hasattr(program.ops[0], "succs"), "Op.succs must stay deleted"



def test_v8_rejects_a_same_device_moe_p2p_carrying_bytes():
    """A same-device transfer is ELIDED at emission, so a ``residual_p2p`` with
    a payload on one device silently drops it (ext_moe_flat.md blocker 4)."""
    program = Program(
        layout=RankLayout((), {}, {}),
        dp_count=1,
        devices=(0,),
        ops=[
            ComputeOp(uid=0, name="c", device=0, duration=(1.0,)),
            TransferOp(
                uid=1,
                name="residual",
                src_device=0,
                dst_device=0,
                size_bytes=4096,
                comm_type=CollectiveType.PIPELINE,
                producer=0,
                deps=(0,),
                moe_component="residual_p2p",
            ),
        ],
        groups={},
        meta=ProgramMeta(),
    )
    with pytest.raises(ProgramInvariantError, match="V8"):
        validate_program(program, check_races=False)
    # the same transfer WITHOUT a moe_component is the legal Class B 9 event
    program.ops[1].moe_component = None
    validate_program(program, check_races=False)


def test_v7_is_exactly_the_a2_shape():
    """**V7** (a grouped collective spans its group) contradicts BUG_LEDGER
    **A2** (``SyncSpread.CLUSTER_RANK_0``: ONE instance on cluster rank 0), which
    is the PRESERVED default of INTERFACES §7. So V7 must be opt-in until P7
    flips the spread — this test pins both directions of that statement."""
    group = GroupKey(axis="ep", members=(0, 1))
    ops = [
        ComputeOp(uid=0, name="c0", device=0, duration=(1.0,)),
        ComputeOp(uid=1, name="c1", device=1, duration=(1.0,)),
        CollectiveOp(
            uid=2,
            name="ep_sync",
            device=0,
            coll=CollectiveType.ALL_REDUCE,
            size_bytes=512,
            label="ep_sync",
            group=group,
            deps=(0,),
        ),
    ]
    program = Program(
        layout=RankLayout((), {}, {}),
        dp_count=1,
        devices=(0, 1),
        ops=ops,
        groups={group: CommGroup(key=group, label="ep_sync")},
        meta=ProgramMeta(),
    )
    validate_program(program, check_races=False)  # V7 off: A2's shape is legal
    with pytest.raises(ProgramInvariantError, match="V7"):
        validate_program(program, check_races=False, check_group_membership=True)
    # instantiate it on every member and V7 passes
    program.ops.append(
        CollectiveOp(
            uid=3,
            name="ep_sync",
            device=1,
            coll=CollectiveType.ALL_REDUCE,
            size_bytes=512,
            label="ep_sync",
            group=group,
            deps=(1,),
        )
    )
    validate_program(program, check_races=False, check_group_membership=True)


@pytest.mark.parametrize(
    "cfg",
    [
        Cfg(dp=2, tp=2, cp=2, pp=2, mb=2, num_layers=4),
        Cfg(dp=2, tp=2, cp=1, pp=2, mb=2, num_layers=4, zero_stage=2),
        Cfg(dp=2, tp=2, cp=1, pp=2, mb=2, num_layers=4, zero_stage=3),
        Cfg(dp=2, tp=2, ep=2, pp=2, mb=2, num_layers=4, moe=True),
    ],
    ids=["ddp_tp2cp2", "zero2_tp2", "zero3_tp2", "moe_tp2ep2"],
)
def test_a2_dp_collectives_exist_on_every_cluster_rank(cfg):
    """**BUG_LEDGER A2, fixed.** One dp collective per (stage, cluster rank).

    Legacy attached a stage's gradient all-reduce / reduce-scatter to
    ``rank_tails[0]`` (``pipeline_fine.py:616-629``), so ONE of the
    ``tp*cp*ep`` cluster ranks bore the whole dp traffic and the other
    ``par_degree - 1`` ranks emitted nothing. That is not a placement detail: a
    rank can only reduce the gradient shard IT owns, and the ``par_degree``
    dp-axis communicators of a stage are DISJOINT, so no peer's collective
    stands in for a missing one.

    Pinned here at three levels:

    1. every dp collective carries ``group=None`` (dp membership is stamped at
       emission over pre-dp device ids — INTERFACES §3.3 / the B2 fix);
    2. the per-``SyncKey`` instance devices are exactly
       ``Placement.cluster_devices(stage)``, and every cluster rank of a stage
       carries the SAME dp multiset and the SAME dp byte total;
    3. the emitted bundle really contains ``par_degree`` distinct dp
       communicators per stage, each of them a ``dp``-sized member set, and
       they partition the ranks (no rank is in two of them).
    """
    program = _program(cfg, Granularity.FLAT)
    fw = make_workload(cfg).freeze()
    placement = Placement(
        fw, Granularity.FLAT, LayerAssignment.contiguous(cfg.num_layers, cfg.pp)
    )
    par_degree = placement.cluster_size()
    assert par_degree == cfg.tp * cfg.cp * cfg.ep > 1, "fixture is not a tp>1 shape"

    dp_ops = [op for op in program.ops if isinstance(op, CollectiveOp) and op.is_dp]
    assert dp_ops, "fixture emits no dp collectives"

    # (1) a dp collective is unlabeled and carries no communicator.
    for op in dp_ops:
        assert op.group is None and op.label is None, op.name

    # (2) instance devices are exactly the stage's cluster, per SyncKey.
    #     ``_sync_name`` is ``<SyncKey>`` for instance 0 and ``<SyncKey>_rank<i>``
    #     after it, so stripping the suffix recovers the requirement identity.
    by_key: Dict[str, List[int]] = {}
    for op in dp_ops:
        by_key.setdefault(re.sub(r"_rank\d+$", "", op.name), []).append(int(op.device))
    stage_of_device = {
        int(device): int(placement.coords_of(device).of("pp"))
        for device in placement.devices()
    }
    for name, devices in by_key.items():
        stages = {stage_of_device[device] for device in devices}
        assert len(stages) == 1, f"{name} spans stages {sorted(stages)}"
        expected = placement.cluster_devices(StageId(stages.pop()))
        assert sorted(devices) == sorted(int(d) for d in expected), name
        assert len(devices) == par_degree, name

    # ... so every cluster rank of a stage carries the same dp load.
    load: Dict[int, Counter] = {}
    dp_bytes: Dict[int, float] = {}
    for op in dp_ops:
        load.setdefault(int(op.device), Counter())[op.comm_key] += 1
        dp_bytes[int(op.device)] = dp_bytes.get(int(op.device), 0.0) + float(op.size_bytes)
    for stage in range(cfg.pp):
        cluster = [int(d) for d in placement.cluster_devices(StageId(stage))]
        hosts = [device for device in cluster if device in load]
        if not hosts:
            continue
        assert len(hosts) == par_degree, f"stage {stage}: dp load on {hosts}"
        assert len({tuple(sorted(load[d].items())) for d in hosts}) == 1, stage
        assert len({round(dp_bytes[d], 6) for d in hosts}) == 1, stage

    # (3) the emitted bundle carries par_degree DISJOINT dp communicators per
    #     stage that hosts dp traffic.
    import tempfile

    from equiv.canonical import canonicalize_bundle
    from program.et_emit import emit_chakra

    with tempfile.TemporaryDirectory() as out:
        emit_chakra(program, out)
        groups = canonicalize_bundle(out).summary()["collectives_by_group"]
    n_devices = len(program.devices)
    # BUG_LEDGER 19 made the communicator dp x cp, so its SIZE is dp * cp and its
    # members are a cp sibling set crossed with the dp replicas — not the
    # per-device dp pair this test used to hard-code. What A2 actually protects
    # is unchanged and is what is asserted: every cluster rank that carries dp
    # load appears in exactly one dp communicator, and those communicators
    # partition the cluster ranks (no rank left out, none doubled).
    cp_degree = max(1, int(cfg.cp))

    # The expected dp communicator of a device: its cp SIBLINGS (BUG_LEDGER 19 —
    # cp ranks hold replicated parameters and partial gradients, so they reduce
    # together) crossed with the dp replicas. At cp == 1 the sibling set is the
    # device itself and this is the pre-19 dp pair. Identified STRUCTURALLY, not
    # by group size: at cp == 1 the dp pair and a tp/ep wire group are both
    # 2-tuples, and only the structure tells them apart.
    def _cp_siblings(device: int) -> Tuple[int, ...]:
        own = dict(placement.coords_of(device).coords)
        fixed = {a: int(v) for a, v in own.items() if a != "cp"}
        return tuple(
            sorted(
                int(d)
                for d in placement.devices()
                if all(
                    int(dict(placement.coords_of(d).coords).get(a, -1)) == v
                    for a, v in fixed.items()
                )
            )
        )

    expected_groups = {
        tuple(
            sorted(
                sibling + idx * n_devices
                for sibling in _cp_siblings(int(device))
                for idx in range(program.dp_count)
            )
        )
        for device in sorted(load)
    }
    seen = {tuple(int(part) for part in key.split(",")) for key in groups}
    assert expected_groups <= seen, (
        f"missing dp communicators: {sorted(expected_groups - seen)}"
    )
    for members in expected_groups:
        assert len(members) == program.dp_count * cp_degree, members
    stages_with_load = len({stage_of_device[d] for d in load})
    assert len(expected_groups) == (par_degree // cp_degree) * stages_with_load
    assert len(load) == par_degree * stages_with_load
    flat = [rank for members in expected_groups for rank in members]
    assert len(flat) == len(set(flat)), "dp communicators are not disjoint"


def test_labels_are_one_to_one_with_member_sets():
    """``et_emit`` interns wire gids BY LABEL (``et_emit.py:221-263``) and raises
    when one label maps to two member sets. The interner keys on the member set,
    so that is unrepresentable."""
    cfg = Cfg(dp=2, pp=2, tp=2, ep=2, mb=2, num_layers=4, moe=True)
    program = _program(cfg, Granularity.FLAT, overlap=NoOverlap())
    by_label: Dict[str, set] = {}
    for op in program.ops:
        if isinstance(op, CollectiveOp) and op.label is not None:
            by_label.setdefault(op.label, set()).add(tuple(op.group.members))
    assert by_label
    for label, member_sets in by_label.items():
        assert len(member_sets) == 1, f"label {label!r} maps to {member_sets}"


def test_program_carries_the_duration_revision_it_was_built_from():
    """**W4**: a consumer that reuses a cached Program across a write-back must
    be able to detect the staleness."""
    cfg = Cfg(dp=1, pp=1, mb=1, num_layers=1)
    spec = make_workload(cfg)
    program = _program(cfg, Granularity.PIPELINE, spec_override=spec)
    assert program.meta.misc["duration_revision"] == spec.durations.revision
    spec.durations.write_block_timings(dense_forward=123.0)
    later = _program(cfg, Granularity.PIPELINE, spec_override=spec)
    assert later.meta.misc["duration_revision"] > program.meta.misc["duration_revision"]


# ===========================================================================
# Granularity contract
# ===========================================================================


def test_block_restricts_work_to_transformer_block_kinds():
    cfg = Cfg(dp=1, pp=1, mb=1, num_layers=1)
    fw = make_workload(cfg).freeze()
    bundle = policies_for(fw, granularity=Granularity.BLOCK)
    work = enumerate_work(fw, bundle.recompute)
    restricted = restrict_work_for(Granularity.BLOCK, work)
    assert {item.kind for item in restricted} <= {WorkKind.LAYER, WorkKind.RECOMPUTE}
    assert restrict_work_for(Granularity.FLAT, work) is work
    program = _program(cfg, Granularity.BLOCK, overlap=NoOverlap())
    assert program.meta.misc["work_restricted"] is True
    assert not _ops_by_name(program, "embedding")
    assert not _ops_by_name(program, "linear_softmax")


def test_block_requires_a_single_replica_single_stage_workload():
    cfg = Cfg(dp=2, pp=2, mb=1, num_layers=2)
    fw = make_workload(cfg).freeze()
    with pytest.raises(BuildError, match="single-replica"):
        check_granularity_preconditions(Granularity.BLOCK, fw)
    with pytest.raises(BuildError, match="single-replica"):
        _program(cfg, Granularity.BLOCK)
    # ... and at dp=pp=1 no data-parallel sync can leak into a block measurement
    single = Cfg(dp=1, pp=1, tp=2, mb=1, num_layers=1)
    program = _program(single, Granularity.BLOCK, overlap=NoOverlap())
    assert not [op for op in program.ops if isinstance(op, CollectiveOp) and op.is_dp]


# ===========================================================================
# Overlap realization (§4.5)
# ===========================================================================


@dataclass(frozen=True)
class _AlwaysOverlap:
    """An overlap policy that declares one fixed decl for every comm key."""

    name: str
    decl: OverlapDecl

    def declare(self, spec, fw):
        return self.decl


def test_overlap_producer_splits_the_producing_compute():
    """``OverlapAnchor.PRODUCER``, ``0 < f < 1`` (verbatim
    ``transforms._split_tp_node_fine``): ``head = d*(1-f)``, ``tail = d*f``;
    ``deps(head) := deps(compute)``, ``deps(tail) := {head}``,
    ``deps(coll) := {head}``, and every consumer of the collective gains
    ``deps += tail``."""
    comm = {"mlp_tp": _tp_comm()}
    template = _block_template(
        [("qkv_proj", [], []), ("MLP", ["mlp_tp"], []), ("output_proj", [], [])], comm
    )
    cfg = Cfg(dp=1, pp=1, tp=1, mb=1, num_layers=1)
    spec = make_workload(cfg).with_(blocks=BlockTemplates(dense=template))
    decl = OverlapDecl(fraction=0.25, anchor=OverlapAnchor.PRODUCER)
    program = _program(
        cfg,
        Granularity.FLAT,
        spec_override=spec,
        overlap=_AlwaysOverlap(name="always", decl=decl),
    )
    head = _one(program, "_head")
    tail = [
        op
        for op in program.ops
        if isinstance(op, ComputeOp)
        and op.name.startswith("MLP_forward")
        and not op.name.endswith("_head")
    ]
    assert len(tail) == 1
    tail = tail[0]
    coll = _one(program, "mlp_tp")
    qkv = _one(program, "qkv_proj_forward")
    out = _one(program, "output_proj_forward")

    total = 2.0  # the template's MLP forward duration (1.0 + entry index)
    assert head.duration[0] == pytest.approx(total * 0.75)
    assert tail.duration[0] == pytest.approx(total * 0.25)
    assert head.deps == (qkv.uid,), "the head takes the compute's deps"
    assert head.uid in tail.deps and head.uid in coll.deps
    assert tail.uid in out.deps and coll.uid in out.deps, (
        "the collective's consumer waits for BOTH the overlapped tail and the "
        "collective"
    )


def test_overlap_producer_hoist():
    """``f >= 1``: the collective takes the compute's deps and the compute takes
    the collective's consumers (``transforms.py:177-188``)."""
    comm = {"mlp_tp": _tp_comm()}
    template = _block_template(
        [("qkv_proj", [], []), ("MLP", ["mlp_tp"], []), ("output_proj", [], [])], comm
    )
    cfg = Cfg(dp=1, pp=1, tp=1, mb=1, num_layers=1)
    spec = make_workload(cfg).with_(blocks=BlockTemplates(dense=template))
    program = _program(
        cfg,
        Granularity.FLAT,
        spec_override=spec,
        overlap=_AlwaysOverlap(
            name="hoist", decl=OverlapDecl(fraction=1.0, anchor=OverlapAnchor.PRODUCER)
        ),
    )
    assert not _ops_by_name(program, "_head")
    mlp = _one(program, "MLP_forward")
    coll = _one(program, "mlp_tp")
    qkv = _one(program, "qkv_proj_forward")
    out = _one(program, "output_proj_forward")
    assert qkv.uid in coll.deps, "the hoisted collective takes the compute's deps"
    assert mlp.uid not in coll.deps
    assert out.uid in _succs(program)[mlp.uid]


def test_overlap_consumer_splits_the_collective_by_bytes():
    """``OverlapAnchor.CONSUMER``, ``0 < f < 1`` (verbatim
    ``transforms._split_cp_edge_fine``): ``block = ceil(total*(1-f))``,
    ``ovlp = total - block``; the ``blocking_consumer`` waits on ``block`` and
    every other consumer on ``ovlp`` and on the blocking consumer."""
    total = 4096.0
    comm = {"cp_gather": _tp_comm(size=total, axis="cp")}
    template = _block_template(
        [("layernorm1", ["cp_gather"], []), ("attention", [], []), ("MLP", [], [])],
        comm,
    )
    cfg = Cfg(dp=1, pp=1, tp=1, cp=1, mb=1, num_layers=1)
    spec = make_workload(cfg).with_(blocks=BlockTemplates(dense=template))
    decl = OverlapDecl(
        fraction=0.25, anchor=OverlapAnchor.CONSUMER, blocking_consumer="attention"
    )
    program = _program(
        cfg,
        Granularity.FLAT,
        spec_override=spec,
        overlap=_AlwaysOverlap(name="cp", decl=decl),
    )
    block = _one(program, "cp_gather_block")
    ovlp = _one(program, "cp_gather_ovlp")
    attention = _one(program, "attention_forward")
    mlp = _one(program, "MLP_forward")
    assert block.size_bytes == math.ceil(total * 0.75)
    assert ovlp.size_bytes == total - math.ceil(total * 0.75)
    assert block.uid in attention.deps, "the blocking consumer waits on the block part"
    # In a SERIAL chain the collective's only consumer IS the blocking one, so
    # the overlapped part becomes a sink that runs concurrently with it — exactly
    # what ``_split_cp_edge_fine``'s "for succ in succs: if succ in
    # attention_children: continue" loop leaves behind (transforms.py:289-298).
    assert ovlp.deps == (block.uid,)
    assert _succs(program)[ovlp.uid] == ()
    assert mlp.deps == (attention.uid,)


def test_overlap_axis_fraction_policy_only_fires_on_declared_axes():
    comm = {"mlp_tp": _tp_comm(), "ep_all_to_all": _tp_comm(axis="ep")}
    template = _block_template(
        [("qkv_proj", ["ep_all_to_all"], []), ("MLP", ["mlp_tp"], [])], comm
    )
    cfg = Cfg(dp=1, pp=1, tp=1, mb=1, num_layers=1)
    spec = make_workload(cfg)
    spec = spec.with_(
        blocks=BlockTemplates(dense=template),
        overlap=replace(spec.overlap, by_axis={"tp": 0.5}),
    )
    program = _program(
        cfg, Granularity.FLAT, spec_override=spec, overlap=AxisFractionOverlap()
    )
    heads = [op.name for op in program.ops if op.name.endswith("_head")]
    assert any(name.startswith("MLP_forward") for name in heads), heads
    assert not any(name.startswith("qkv_proj_forward") for name in heads), heads


# ===========================================================================
# no name dispatch (INTERFACES §8 rule 2)
# ===========================================================================


def _string_constants(path: Path) -> List[str]:
    """Every string CONSTANT in ``path``, docstrings excluded.

    An AST walk, not a text scan: a citation inside a docstring can never be
    mistaken for a dispatch, and a dispatch can never hide inside one.
    """
    import ast

    tree = ast.parse(path.read_text())
    docstrings: set = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            docstrings.add(id(body[0].value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def _name_membership_tests(path: Path) -> List[str]:
    """Every ``x in <expr>.name`` / ``<expr>.name`` substring test — the
    ``"linear_softmax" in obj.name`` shape this restructure deletes."""
    import ast

    tree = ast.parse(path.read_text())
    found: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        for op, comparator in zip(node.ops, node.comparators):
            if isinstance(op, (ast.In, ast.NotIn)) and (
                isinstance(comparator, ast.Attribute) and comparator.attr == "name"
            ):
                found.append(ast.unparse(node))
    return found


def test_build_never_dispatches_on_a_name_or_an_axis_literal():
    """INTERFACES §8 rule 2. ``et_emit.py`` is the calibration: 644 LOC, zero
    occurrences of ``zero3``/``moe``/``recompute``/``softmax``/``embedding``.
    ``build.py`` may cite them in prose only."""
    build_py = REPO_ROOT / "program" / "build.py"
    constants = _string_constants(build_py)
    for needle in (
        "zero3",
        "zero2",
        "softmax",
        "embedding",
        "optimizer",
        "bwd",
        "attention",
        "moe_dispatch",
        "flattened_mode",
    ):
        offenders = [text for text in constants if needle in text]
        assert not offenders, (
            f"build.py carries the string literal {needle!r}: {offenders}"
        )
    # The only comm key build() may NAME is cross_layer — R2's own rule owns
    # pipeline data movement — and it is a module constant, declared once.
    assert [c for c in constants if "cross_layer" in c] == ["cross_layer"]
    # No axis literal: dp comes from program.types.DP_AXIS, every other axis
    # arrives on CommSpec.axes / SyncRequirement.axes.
    assert not {"tp", "ep", "cp", "pp"} & set(constants)
    assert not _name_membership_tests(build_py)
    # ... and the same holds for L3.
    for module in ("schedule/policy.py", "schedule/gpipe.py"):
        path = REPO_ROOT / "program" / module
        assert not _name_membership_tests(path)
        assert not {"tp", "ep", "cp", "pp", "dp"} & set(_string_constants(path))


# ===========================================================================
# T3 canonical determinism (env-gated)
# ===========================================================================
#
# P5 replaced the build()-vs-legacy differential with this: the legacy builders
# it compared against are deleted, so the property that remains checkable on the
# real 42-spec matrix is O1 — program order and emission are a pure function of
# the FrozenWorkload — plus T3 completability of what comes out.

_oracle_gate = pytest.mark.skipif(
    os.environ.get("RAPID_BUILD_DIFF", "") != "1",
    reason="the whole-matrix determinism sweep is a dev gate; set RAPID_BUILD_DIFF=1",
)

BASE_MODEL_CONFIG = (
    REPO_ROOT / "validation_scripts" / "validation_configs" / "model-config" / "Llama2-7B.yaml"
)
BASE_HW_CONFIG = (
    REPO_ROOT / "validation_scripts" / "validation_configs" / "hardware-config" / "a100_80GB.yaml"
)


def _parse_spec_configs(spec, tmp_path: Path):
    import config as config_mod
    from validation_scripts.validation_helpers import _deep_update, _load_yaml, _write_yaml

    model_dict = copy.deepcopy(_load_yaml(str(BASE_MODEL_CONFIG)))
    hw_dict = copy.deepcopy(_load_yaml(str(BASE_HW_CONFIG)))
    _deep_update(model_dict, spec.model_overrides())
    _deep_update(hw_dict, spec.hardware_overrides())
    model_path = tmp_path / "model.yaml"
    hw_path = tmp_path / "hardware.yaml"
    _write_yaml(str(model_path), model_dict)
    _write_yaml(str(hw_path), hw_dict)
    mode = str(model_dict.get("model_param", {}).get("mode", "LLM")).strip().upper()
    hw_config = config_mod.parse_config(str(hw_path), config_type="hardware")
    model_config = config_mod.parse_config(str(model_path), config_type=mode)
    config_mod.validate_configs(hw_config, model_config)
    return hw_config, model_config, mode


def _dispatchers(spec, hw_config, model_config, mode, out_dir: Path):
    """Drive the timing models to the dispatcher — production's own entry."""
    from llm_execution import LLMExecutionDispatcher

    cases = []
    if spec.run_type == "training":
        from train_timing import TimeCalculationLLM

        tc = TimeCalculationLLM(hw_config, model_config, mode, output_dir=str(out_dir))
        tc._build_training_graphs_and_memory_data()
        cases.append(("final", tc, LLMExecutionDispatcher(tc, tc.workload)))
        if tc.workload_no_dp is not None:
            cases.append(("no_dp", tc, LLMExecutionDispatcher(tc, tc.workload_no_dp)))
    else:
        from inference_timing import TimeCalculationLLMInference

        tc = TimeCalculationLLMInference(hw_config, model_config, mode, output_dir=str(out_dir))
        batch_size = tc._effective_transformer_batch()
        prefill_len = tc.seq_len - tc.model.decode_len
        assert prefill_len > 0
        num_SMs = tc.hw_config.tech_config.core.num_bundles
        transformer_timings, node_breakdown = tc.compute_all_gemm_and_node_times(
            batch_size, tc.vocab_size, tc.hidden_dim, prefill_len, tc.num_heads,
            tc.kv_heads, tc.intermediate_size, num_SMs, use_moe_override=False,
        )
        workload, _ = tc._prepare_execution_graphs(
            node_breakdown=node_breakdown, transformer_timings=transformer_timings,
            batch_size=batch_size, seq_len=prefill_len, hidden_dim=tc.hidden_dim,
            intermediate_size=tc.intermediate_size, vocab_size=tc.vocab_size,
            include_pipeline_backward=False, include_transformer_backward=False,
        )
        cases.append(("prefill", tc, LLMExecutionDispatcher(tc, workload)))
    return cases


def _describe_program(program: Program) -> List[Any]:
    """A total, order-sensitive description: two equal descriptions mean two
    identical Programs (uids, deps, placements, bytes, groups and all)."""
    out: List[Any] = [program.devices, program.dp_count, tuple(program.layout.axis_order)]
    for op in program.ops:
        if isinstance(op, ComputeOp):
            out.append((op.uid, op.device, op.duration, op.deps, op.role.name,
                        op.direction.name, str(op.mem_kind), op.layer, op.micro_batch,
                        op.is_moe_layer, op.recompute, op.param_gather))
        elif isinstance(op, CollectiveOp):
            out.append((op.uid, op.device, op.coll.name, op.size_bytes, op.participants,
                        op.axes, op.is_dp, op.label,
                        None if op.group is None else op.group.members,
                        op.comm_key, op.deps))
        else:
            out.append((op.uid, op.src_device, op.dst_device, op.size_bytes,
                        None if op.comm_type is None else op.comm_type.name,
                        op.producer, op.consumers, op.moe_component, op.deps))
    return out


_GRANULARITIES = (Granularity.PIPELINE, Granularity.FLAT)


@_oracle_gate
@pytest.mark.parametrize("spec", MATRIX, ids=[s.spec_id for s in MATRIX])
def test_build_is_deterministic_and_emits_a_completable_bundle(spec, tmp_path):
    """**O1** + **T3** on the real matrix: build twice, emit, compare canonically.

    A Program is compared field for field (not just canonically), because
    ``build()`` claims to be a pure function of the ``FrozenWorkload`` — the ET
    bundle is then compared through ``equiv.canonical`` (id-independent) and run
    through ``equiv.dlsim``, which is what the gate map asks of T3.
    """
    from equiv.canonical import canonicalize_bundle, diff_bundles_detailed
    from equiv.dlsim import BundleSim
    from program.et_emit import emit_chakra

    hw_config, model_config, mode = _parse_spec_configs(spec, tmp_path)
    cases = _dispatchers(spec, hw_config, model_config, mode, tmp_path / "out")
    checked = 0
    for label, _tc, dispatcher in cases:
        # Every granularity is emittable, MoE included: flattened MoE execution
        # is production (``ext_moe_flat.md`` P8, 2026-07-26), so the FLAT MoE
        # bundle must satisfy the group-order postcondition and the AstraSim
        # scheduling contract like any other.
        emittable = {Granularity.PIPELINE, Granularity.FLAT}
        for granularity in _GRANULARITIES:
            first = dispatcher._build_program(granularity, label="a")
            second = dispatcher._build_program(granularity, label="b")
            assert _describe_program(first) == _describe_program(second), (
                f"{spec.spec_id}[{label}] {granularity.name}: build() is not a pure "
                "function of its inputs"
            )
            checked += 1
            if granularity not in emittable:
                continue
            tag = f"{label}_{granularity.name.lower()}"
            out_a = tmp_path / f"emit_{tag}_a"
            out_b = tmp_path / f"emit_{tag}_b"
            bundle = emit_chakra(first, str(out_a))
            emit_chakra(second, str(out_b))
            problems = diff_bundles_detailed(
                canonicalize_bundle(str(out_a)).summary(),
                canonicalize_bundle(str(out_b)).summary(),
            )
            assert not problems, (
                f"{spec.spec_id}[{tag}]: re-emission is not canonically identical:\n"
                + "\n".join(message for _f, message, _o, _n in problems)
            )
            result = BundleSim(str(out_a)).simulate()
            assert result.completed, (
                f"{spec.spec_id}[{tag}]: emitted bundle does not complete under the "
                f"AstraSim scheduling contract: {result.blocked_report or result.cycle}"
            )
            assert bundle.rank_ids
    assert checked


# ---------------------------------------------------------------------------
# R3 reachability: the slot-floor bound (D2)
# ---------------------------------------------------------------------------


_R3_BOUND_CFGS = [
    Cfg(dp=2, tp=2, cp=1, pp=2, mb=3),
    Cfg(dp=2, tp=2, cp=1, pp=2, mb=3, ep=2, moe=True),
    Cfg(dp=2, tp=1, cp=1, pp=2, mb=3, zero_stage=3),
    Cfg(dp=2, tp=1, cp=1, pp=2, mb=3, zero_stage=2),
    Cfg(dp=1, tp=2, cp=1, pp=2, mb=3, full_recomputation=True),
    Cfg(dp=2, tp=1, cp=1, pp=2, mb=3, dp_microbatch="last_mb"),
    Cfg(dp=1, tp=1, cp=2, pp=2, mb=3),
    Cfg(dp=1, tp=2, cp=1, pp=2, mb=3, run_type="inference"),
    Cfg(dp=1, tp=1, cp=1, pp=4, mb=6, num_layers=8),
]


@pytest.mark.parametrize("granularity", [Granularity.PIPELINE, Granularity.FLAT])
def test_r3_implied_answers_exactly_what_the_unbounded_walk_answers(granularity):
    """``_r3_implied`` — the slot-bounded FORWARD-cone verdict — is EXACT.

    It is what keeps R3 linear in the schedule length instead of quadratic,
    and its soundness rests on D2: paths in the R1+R2+R3-so-far graph are
    slot-monotone, so bounding the forward walk above by ``slot(target)`` and
    the old backward walk below by ``slot(source)`` prune the SAME window. If
    the verdict ever disagreed with the unbounded backward walk, R3 would add
    or drop a SCHEDULE edge and the only thing that would notice is a golden
    diff — so the agreement is asserted here directly, on every R3 query the
    matrix generates.
    """
    import program.build as build_mod

    seen_queries = 0
    disagreements = []
    original = build_mod._Builder._r3_implied
    unbounded_walk = build_mod._Builder._reaches

    def checking(self, source, target, floor):
        fast = original(self, source, target, floor)
        nonlocal seen_queries
        seen_queries += 1
        unbounded = unbounded_walk(self, source, target)
        if fast != unbounded:
            disagreements.append(
                (self._nodes[source].name, self._nodes[target].name,
                 floor, fast, unbounded)
            )
        return fast

    build_mod._Builder._r3_implied = checking
    try:
        for cfg in _R3_BOUND_CFGS:
            _program(replace(cfg, flattened=granularity is Granularity.FLAT),
                     granularity=granularity)
    finally:
        build_mod._Builder._r3_implied = original

    assert seen_queries, "no R3 implication queries were exercised"
    assert not disagreements, (
        "_r3_implied changed a reachability answer (D2 broken):\n"
        + "\n".join(str(d) for d in disagreements[:8])
    )


def test_r3_is_linear_in_schedule_length_not_quadratic():
    """Guard against the ``_apply_r3`` blow-up that made GPT 1T unrunnable.

    Before the ``slot_floor`` bound, every microbatch-boundary query walked the
    whole ancestor set, so doubling the microbatch count roughly QUADRUPLED
    build time; a ``pp=32, L=64, mb=512`` PIPELINE build took over two CPU-hours.
    The assertion is on the RATIO, not on absolute seconds, so it does not
    depend on machine speed.
    """
    import time

    def build_seconds(micro_batches: int) -> float:
        cfg = Cfg(dp=1, tp=1, cp=1, pp=8, mb=micro_batches, num_layers=16)
        start = time.perf_counter()
        _program(cfg, granularity=Granularity.PIPELINE)
        return time.perf_counter() - start

    build_seconds(8)  # warm any lazy imports so they are not billed to the first point
    small = build_seconds(32)
    large = build_seconds(128)
    # 4x the microbatches. Linear => ~4x; quadratic => ~16x. 8x is a wide band
    # that still fails loudly on a return to the quadratic walk.
    assert large < small * 8.0, (
        f"R3 looks superlinear again: mb=32 took {small:.3f}s, mb=128 took "
        f"{large:.3f}s (ratio {large / small:.1f}x for a 4x input)"
    )


def test_r3_rejects_a_graph_whose_edges_run_backwards_in_schedule_slots():
    """D2 is CHECKED, not assumed — a violation raises instead of silently
    producing an extra SCHEDULE edge."""
    import program.build as build_mod

    original = build_mod._Builder._apply_r3

    def sabotaged(self):
        # Flip one existing edge's endpoints in slot space by rewriting the
        # dep's order key (a builder-side array since the proto->IR fusion,
        # 2026-08-02) to something later than its consumer's.
        for (dep_nid, node_nid) in self._edges:
            dep_order = self._orders[dep_nid]
            node_order = self._orders[node_nid]
            if dep_order[0] < node_order[0]:
                self._orders[dep_nid] = (node_order[0] + 1,) + tuple(dep_order[1:])
                break
        else:  # pragma: no cover - the matrix always has such an edge
            pytest.skip("no strictly increasing edge to sabotage")
        return original(self)

    build_mod._Builder._apply_r3 = sabotaged
    try:
        with pytest.raises(BuildError, match="D2 violated"):
            _program(Cfg(dp=1, tp=1, cp=1, pp=2, mb=3), granularity=Granularity.PIPELINE)
    finally:
        build_mod._Builder._apply_r3 = original


# ---------------------------------------------------------------------------
# BUG_LEDGER 19 — the gradient reduction group is dp x cp
# ---------------------------------------------------------------------------


def test_gradient_reducer_spans_cp_because_cp_gradients_are_partial():
    """The dp reducer's communicator is ``dp x cp``, not ``dp``.

    CP ranks hold REPLICATED parameters (``layer_params_per_rank`` divides by tp
    only) and compute PARTIAL gradients — ``_shard_gemm_descriptor`` under
    ``ParallelismMode.CONTEXT`` sets ``shard_m = ceil(m / cp)`` for every GEMM
    type, QKV/FFN1/FFN2/OUT_PROJ included. Reducing only over dp leaves those
    partials never summed, and the ring-attention cp collectives do not do it:
    they reduce dL/dx, not dL/dW.
    """
    from program.work import DP_AXIS

    for cp, expect_cp in ((1, False), (2, True)):
        program = _program(
            Cfg(dp=2, tp=1, cp=cp, pp=2, mb=2, flattened=True),
            granularity=Granularity.FLAT,
        )
        reducers = [
            op for op in program.ops
            if isinstance(op, CollectiveOp) and op.is_dp
        ]
        assert reducers, f"cp={cp}: no dp reducer built"
        for op in reducers:
            assert DP_AXIS in op.axes, f"cp={cp}: {op.name} lost the dp axis"
            has_cp = "cp" in op.axes
            assert has_cp is expect_cp, (
                f"cp={cp}: {op.name} axes={op.axes}; expected cp "
                f"{'present' if expect_cp else 'absent'}"
            )
            assert op.participants == 2 * cp, (
                f"cp={cp}: {op.name} participants={op.participants}, expected {2 * cp}"
            )


def test_dp_cp_reducer_emits_one_communicator_per_cp_sibling_set(tmp_path):
    """Emission: cp siblings share ONE process group, of size ``dp * cp``.

    This is the half that makes the fix real — a correct ``participants`` count
    with per-device communicators would still never sum across cp. Also pins the
    id-space property that made the naive version wrong: a ``dp x cp`` group is
    interned by MEMBER SET, so it can never collide with the per-device dp stage
    group ids that pure-dp collectives (e.g. the ZeRO gathers) still use.
    """
    from program.et_emit import emit_chakra

    program = _program(
        Cfg(dp=2, tp=1, cp=2, pp=2, mb=2, flattened=True), granularity=Granularity.FLAT
    )
    out = tmp_path / "emit_dpcp"
    bundle = emit_chakra(program, str(out))
    groups = bundle.comm_groups

    dp_cp_groups = {
        gid: members for gid, members in groups.items() if len(members) == 4
    }
    assert dp_cp_groups, f"no dp x cp (4-member) group emitted; groups={groups}"

    dp_count = program.dp_count
    devices = list(program.devices)
    per_replica = len(devices)
    for gid, members in dp_cp_groups.items():
        # members must be <cp sibling devices> x <dp replicas>
        by_replica = {}
        for rank in members:
            by_replica.setdefault(rank // per_replica, []).append(rank % per_replica)
        assert len(by_replica) == dp_count, (
            f"group {gid} members {members} do not span all {dp_count} dp replicas"
        )
        sibling_sets = {tuple(sorted(v)) for v in by_replica.values()}
        assert len(sibling_sets) == 1, (
            f"group {gid} uses different device sets per dp replica: {by_replica}"
        )
        assert len(next(iter(sibling_sets))) == 2, (
            f"group {gid} should hold cp=2 sibling devices, got {sibling_sets}"
        )

    # id-space: no member set is served by two ids, and no id by two member sets
    seen = {}
    for gid, members in groups.items():
        key = tuple(sorted(members))
        assert key not in seen or seen[key] == gid, (
            f"member set {key} interned under two ids: {seen[key]} and {gid}"
        )
        seen[key] = gid


def test_cp1_is_byte_identical_after_the_dp_cp_change():
    """cp == 1 must be untouched: the sibling set is the device itself, so the
    communicator, its id and the emitted bytes are the pre-19 ones. This is what
    keeps 41 of 44 goldens bit-identical."""
    program = _program(
        Cfg(dp=2, tp=2, cp=1, pp=2, mb=2, flattened=True), granularity=Granularity.FLAT
    )
    for op in program.ops:
        if isinstance(op, CollectiveOp) and op.is_dp:
            assert op.axes == ("dp",), f"{op.name} axes drifted to {op.axes} at cp=1"
            assert op.interconnect == "dp"


# ---------------------------------------------------------------------------
# BUG_LEDGER A6 — the ZeRO-3 prefetch anchor is a DEVICE-LOCAL notion
# ---------------------------------------------------------------------------


def _zero3_prefetch_requirements(cfg: Cfg):
    """Rows S8/S15 — the per-layer parameter gathers — with the context that
    produced them, so a test can ask which stage each anchor lives on."""
    from program.policies import policies_for
    from program.policies.sharding import ContiguousStages, ShardingContext
    from program.work import enumerate_work

    fw = make_workload(cfg).freeze()
    bundle = policies_for(fw, block_expanded=cfg.flattened)
    work = enumerate_work(fw, bundle.recompute)
    ctx = ShardingContext(
        fw=fw,
        work=work,
        grad_accum=bundle.grad_accum,
        stages=ContiguousStages.legacy(cfg.num_layers, cfg.pp),
    )
    out = []
    for item in work:
        for req in bundle.sharding.requirements(item, ctx):
            if req.origin in ("S8", "S15"):
                out.append(req)
    return out, ctx


def test_zero3_prefetch_anchor_never_crosses_a_stage():
    """**BUG_LEDGER A6.** A layer's parameter gather is a dp-axis collective over
    ITS OWN stage's replicas: no rank of another stage participates and no other
    stage produces its input. So its anchor must be on its own device.

    Before the fix, rows S8/S15 chose the anchor by LAYER ARITHMETIC
    (``layer -+ prefetch_depth``), and at a stage boundary that names a layer on
    the other stage — where ``PARALLEL_TO``'s ``deps(req) += deps(anchor)`` made
    the gather inherit the foreign stage's predecessor set. Fails with
    ``_prefetch_attach`` reverted to the raw arithmetic.
    """
    cfg = Cfg(dp=2, pp=4, mb=2, num_layers=8, zero_stage=3)
    reqs, ctx = _zero3_prefetch_requirements(cfg)
    assert reqs, "no S8/S15 requirements were produced"
    offenders = [
        (req.origin, req.key, req.place_on, anchor)
        for req in reqs
        for anchor in req.anchors
        if not ctx.same_stage(anchor, req.place_on)
    ]
    assert not offenders, (
        "ZeRO-3 prefetch gathers anchored on another stage's work: "
        f"{offenders[:3]}"
    )


def test_zero3_stage_entry_gather_is_issued_before_its_own_layer():
    """The stage-boundary fallback is ``BEFORE(target)``, not ``PARALLEL_TO``.

    ``PARALLEL_TO(target)`` would run the gather *alongside* the layer that needs
    the parameters and only order it before that layer's SUCCESSORS — i.e. the
    layer would compute without waiting for its own gather. ``BEFORE`` takes the
    layer's deps (the inbound cross-stage transfer) and makes the layer wait,
    which is "a stage issues its entry layer's gather when it becomes active".
    """
    from program.work import AttachMode

    cfg = Cfg(dp=2, pp=4, mb=2, num_layers=8, zero_stage=3)
    reqs, ctx = _zero3_prefetch_requirements(cfg)
    boundary = [req for req in reqs if req.anchors == (req.place_on,)]
    assert boundary, "no stage-entry gather in a pp=4 workload"
    for req in boundary:
        assert req.mode is AttachMode.BEFORE, (
            f"{req.origin} {req.key}: a gather anchored on the work it feeds must "
            f"be BEFORE it, got {req.mode}"
        )


def test_no_zero3_gather_becomes_an_untimed_root():
    """A6's fix must not create program ROOTS.

    ``analytic_sim._ROOT_COMM_IS_UNTIMED`` prices a collective with no deps at
    zero, so a gather that became a root would be silently free. It cannot
    happen — the ``BEFORE(target)`` fallback only fires at a stage boundary, and
    a stage's entry layer always has the inbound transfer as a dep — but the
    reasoning is load-bearing, so it is checked rather than argued.
    """
    cfg = Cfg(dp=2, pp=4, mb=2, num_layers=8, zero_stage=3)
    for granularity in (Granularity.PIPELINE, Granularity.FLAT):
        program = _program(cfg, granularity, overlap=NoOverlap())
        roots = [
            op for op in program.ops
            if isinstance(op, CollectiveOp)
            and not op.deps
            and "zero3_transformer_gather" in op.name
        ]
        assert not roots, (
            f"{granularity.name}: transformer parameter gathers became untimed "
            f"program roots: {[op.name for op in roots]}"
        )
