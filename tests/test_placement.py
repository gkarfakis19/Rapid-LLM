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

"""L2 — placement, communicator construction and block expansion (INTERFACES §3).

Three tiers, all intrinsic since P5 (the legacy oracles this file used to run
differentially against — ``pipeline_fine._FineExpander``,
``pipeline_fine.build_fine_program``, ``block_program.build_block_root`` and
``legacy_lowering._build_axis_groups`` — are DELETED; the differentials they
gated on passed before the deletion, and the properties they proved are asserted
directly here now):

1. **device space** — ``Placement.devices``/``devices_for``/``cluster_rank_of``
   per granularity, the placement POLICY (Class B 10d) and the layout/degree
   consistency check;
2. **communicators** — members CONSTRUCTED from the ``RankLayout``, the declared
   composite ``("tp","ep")`` routing group (never inferred from a participant
   count), singleton and absent-axis groups, and ``devices_for_sync`` for every
   ``SyncSpread``;
3. **block expansion** — chain shape, ``CommSpec.placement`` (pre/post), the MoE
   hot/cold routing join, and the B1/B2/B4/B5 amendments.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pytest

from timing_model import CollectiveType

from program.block import BlockTemplate
from program.groups import CommunicatorFactory, GroupError, canonical_axis_label
from program.layout import CANONICAL_AXES, RankLayout
from program.placement import (
    LEGACY_PLACEMENT,
    BlockExpander,
    CommStep,
    ComputeStep,
    DeviceCoord,
    Granularity,
    Placement,
    PlacementError,
    PlacementPolicy,
    StepKind,
    WorkSpread,
    canonical_layout,
)
from program.policies.recompute import FullRecompute, NoRecompute
from program.policies.routing import EP_ROUTING, TP_EP_ROUTING
from program.work import (
    Direction,
    WorkItem,
    WorkKind,
    enumerate_work,
)
from program.workload import (
    BlockTemplates,
    CommSpecTable,
    DpMicrobatchMode,
    DurationTable,
    GradAccumCycle,
    ModelShape,
    OverlapSpec,
    ParallelDegrees,
    RunPolicy,
    RunType,
    WorkloadSpec,
)


# ---------------------------------------------------------------------------
# L3 stand-in: LayerAssignment (program/sched/policy.py lands in P4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LayerAssignment:
    """The contract's ``LayerAssignment`` (INTERFACES §4.1) reduced to the three
    members L2 consumes. Uses the legacy remainder-first split so the FLAT
    oracle sees the same layer->stage map as ``ScheduleSpec.stage_for_layer``.

    ``layers_of`` joined ``stage_of``/``min_layer`` when the fused per-stage
    optimizer node started being priced from the layer set its stage owns
    (BUG_LEDGER 10b) — ``BlockExpander._expand_single`` reads it."""

    stage_of_layer: Tuple[int, ...]
    num_stages: int

    @classmethod
    def contiguous(cls, num_layers: int, pp: int) -> "_LayerAssignment":
        counts = _contiguous_counts(num_layers, pp)
        mapping: List[int] = []
        for stage, count in enumerate(counts):
            mapping.extend([stage] * count)
        return cls(stage_of_layer=tuple(mapping), num_stages=max(1, int(pp)))

    def stage_of(self, layer: int) -> int:
        return self.stage_of_layer[layer]

    def layers_of(self, stage: int) -> Tuple[int, ...]:
        return tuple(
            layer
            for layer, assigned in enumerate(self.stage_of_layer)
            if int(assigned) == int(stage)
        )

    def min_layer(self, stage: int) -> Optional[int]:
        for layer, assigned in enumerate(self.stage_of_layer):
            if assigned == stage:
                return layer
        return None


# ---------------------------------------------------------------------------
# Synthetic workload fixtures
# ---------------------------------------------------------------------------

#: A two-GEMM dense block with one tp collective and one cp collective. Both
#: are declared ``placement="post"``, matching every production comm rule.
DENSE_COMM: Mapping[str, Mapping[str, Any]] = {
    "qkv_all_reduce": {
        "size": 4096,
        "type": CollectiveType.ALL_REDUCE,
        "participants": 2,
        "interconnect_type": "tp",
        "placement": "post",
    },
    "attn_all_gather": {
        "size": 2048,
        "type": CollectiveType.ALL_GATHER,
        "participants": 2,
        "interconnect_type": "cp",
        "placement": "post",
    },
    "cross_layer": {
        "size": 8192,
        "type": CollectiveType.PIPELINE,
        "participants": 2,
        "interconnect_type": "pp",
    },
}

DENSE_GEMMS: Tuple[Mapping[str, Any], ...] = (
    {
        "name": "qkv_proj",
        "forward": {"duration": 1.0e-4, "comm_keys": ["qkv_all_reduce"]},
        "backward": {"duration": 2.0e-4, "comm_keys": ["attn_all_gather"]},
    },
    {
        "name": "MLP",
        "forward": {"duration": 3.0e-4, "comm_keys": []},
        "backward": {"duration": 4.0e-4, "comm_keys": ["qkv_all_reduce"]},
    },
)

DURATIONS: Mapping[str, float] = {
    "embedding_f": 1.1e-5,
    "embedding_b": 1.2e-5,
    "linear_softmax_f": 1.3e-5,
    "linear_softmax_b": 1.4e-5,
    "transformer_f": 1.0e-3,
    "transformer_b": 2.0e-3,
    "transformer_f_dense": 1.0e-3,
    "transformer_b_dense": 2.0e-3,
    "optimizer": 5.0e-5,
}


@dataclass(frozen=True)
class Grid:
    """One parallelism grid of the golden matrix (dp forced to 1 so the FLAT
    oracle contains no DP/ZeRO collectives — those are L1's lattice, not L2's
    placement)."""

    label: str
    tp: int
    cp: int
    ep: int
    pp: int
    mb: int
    num_layers: int
    recompute: bool = False
    include_backward: bool = True


#: The parallelism grids of the flattened golden rows (equiv/configs.MATRIX),
#: plus the MoE grid's (tp,ep) shape and one inference-shaped row.
GRIDS: Tuple[Grid, ...] = (
    Grid("dp1tp1cp1pp1mb1", tp=1, cp=1, ep=1, pp=1, mb=1, num_layers=2),
    Grid("dp1tp2cp1pp2mb2", tp=2, cp=1, ep=1, pp=2, mb=2, num_layers=4),
    Grid("dp1tp1cp2pp2mb2", tp=1, cp=2, ep=1, pp=2, mb=2, num_layers=4),
    Grid("dp1tp2cp2pp2mb2", tp=2, cp=2, ep=1, pp=2, mb=2, num_layers=4),
    Grid("dp1tp4cp1pp1mb1:mesh2d", tp=4, cp=1, ep=1, pp=1, mb=1, num_layers=2),
    Grid("dp1tp2cp1ep2pp2mb2:moegrid", tp=2, cp=1, ep=2, pp=2, mb=2, num_layers=4),
    Grid("dp1tp2cp1pp2mb2:recompute", tp=2, cp=1, ep=1, pp=2, mb=2, num_layers=4, recompute=True),
    Grid(
        "inf:dp1tp2cp1pp2mb2",
        tp=2, cp=1, ep=1, pp=2, mb=2, num_layers=4, include_backward=False,
    ),
    Grid("dp1tp3cp1pp3mb2:ragged", tp=3, cp=1, ep=1, pp=3, mb=2, num_layers=5),
)


def _contiguous_counts(num_layers: int, pp: int) -> Tuple[int, ...]:
    """The remainder-first contiguous split, as a COUNT tuple.

    Replaces the deleted ``schedule.legacy_layers_per_stage``; the authority is
    ``LayerAssignment.contiguous`` (L3) and this derives the counts from it, so
    there is still exactly one implementation of the split.
    """
    from program.schedule.policy import LayerAssignment

    assignment = LayerAssignment.contiguous(num_layers, pp)
    return tuple(len(assignment.layers_of(stage)) for stage in range(pp))


def _template(
    gemms: Tuple[Mapping[str, Any], ...] = DENSE_GEMMS,
    comm: Mapping[str, Mapping[str, Any]] = DENSE_COMM,
) -> BlockTemplate:
    return BlockTemplate.from_gemm_entries(list(gemms), comm)


def _workload(
    grid: Grid,
    *,
    dp: int = 1,
    comm: Mapping[str, Mapping[str, Any]] = DENSE_COMM,
    gemms: Tuple[Mapping[str, Any], ...] = DENSE_GEMMS,
    moe_gemms: Optional[Tuple[Mapping[str, Any], ...]] = None,
    moe_comm: Optional[Mapping[str, Mapping[str, Any]]] = None,
    pipeline_comm: Optional[Mapping[str, Mapping[str, Any]]] = None,
    moe_layer_mask: Tuple[bool, ...] = (),
    layout: Optional[RankLayout] = None,
) -> Tuple[WorkloadSpec, BlockTemplate]:
    degrees = ParallelDegrees(tp=grid.tp, cp=grid.cp, ep=grid.ep, pp=grid.pp, dp=dp)
    shape = ModelShape(
        num_layers=grid.num_layers,
        micro_batches=grid.mb,
        model_type="gpt",
        moe_layer_mask=moe_layer_mask,
    )
    run = RunPolicy(
        run_type=RunType.TRAINING if grid.include_backward else RunType.INFERENCE,
        grad_accum_cycle=GradAccumCycle.FINAL,
        dp_microbatch_mode=DpMicrobatchMode.EVERY_MB,
        zero_stage=0,
        full_recomputation=grid.recompute,
        pipeline_style_recompute=grid.recompute,
    )
    dense = _template(gemms, comm)
    moe = (
        _template(moe_gemms, comm if moe_comm is None else moe_comm)
        if moe_gemms is not None
        else None
    )
    spec = WorkloadSpec(
        degrees=degrees,
        shape=shape,
        run=run,
        comm=CommSpecTable.from_legacy(comm if pipeline_comm is None else pipeline_comm),
        blocks=BlockTemplates(dense=dense, moe=moe),
        overlap=OverlapSpec(parallelism_mode=None, by_axis={}),
        layout=layout
        if layout is not None
        else canonical_layout(
            {"tp": grid.tp, "cp": grid.cp, "ep": grid.ep, "pp": grid.pp, "dp": dp}
        ),
        interconnect={},
        granularity_hint=Granularity.FLAT,
        durations=DurationTable(DURATIONS),
    )
    return spec, dense


def _placement(
    grid: Grid,
    granularity: Granularity,
    **kwargs: Any,
) -> Tuple[Placement, Any]:
    spec, _template_obj = _workload(grid, **kwargs)
    fw = spec.freeze()
    layers = _LayerAssignment.contiguous(grid.num_layers, grid.pp)
    return Placement(fw, granularity, layers), fw


# ---------------------------------------------------------------------------
# 1. Device sets per granularity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grid", GRIDS, ids=[g.label for g in GRIDS])
def test_device_sets_per_granularity(grid: Grid) -> None:
    cluster = grid.tp * grid.cp * grid.ep

    coarse, _ = _placement(grid, Granularity.PIPELINE)
    assert coarse.cluster_size() == 1
    assert coarse.devices() == tuple(range(grid.pp))
    # At PIPELINE the device IS the stage.
    for stage in range(grid.pp):
        assert coarse.stage_device(stage) == stage

    fine, _ = _placement(grid, Granularity.FLAT)
    assert fine.cluster_size() == cluster
    assert fine.devices() == tuple(range(grid.pp * cluster))

    block, _ = _placement(grid, Granularity.BLOCK)
    assert block.cluster_size() == cluster
    assert block.devices() == tuple(range(cluster))
    # BLOCK has no pipeline: every stage collapses onto the same cluster.
    for stage in range(grid.pp):
        assert block.cluster_devices(stage) == tuple(range(cluster))


@pytest.mark.parametrize("grid", GRIDS, ids=[g.label for g in GRIDS])
def test_activation_shard_size_excludes_ep(grid: Grid) -> None:
    """BUG_LEDGER **D 10c, ep half** (fixed 2026-07-27).

    ``cluster_size()`` counts the stage's DEVICES (``tp*cp*ep``);
    ``activation_shard_size()`` counts how many ways a TOKEN-INDEXED tensor is
    cut across them (``tp*cp``). They differ by exactly ``ep``, because an EP
    rank owns a distinct microbatch and therefore holds a whole copy of its own
    residual stream. Only the latter may divide ``cross_layer`` bytes.
    """
    shards = grid.tp * grid.cp
    cluster = grid.tp * grid.cp * grid.ep

    coarse, _ = _placement(grid, Granularity.PIPELINE)
    # PIPELINE: the stage IS the device, so the raw value is already per-device.
    assert coarse.activation_shard_size() == 1

    for granularity in (Granularity.FLAT, Granularity.BLOCK):
        placement, _ = _placement(grid, granularity)
        assert placement.activation_shard_size() == shards
        assert placement.cluster_size() == cluster
        assert (
            placement.cluster_size()
            == placement.activation_shard_size() * grid.ep
        )


# P5: ``test_fine_device_ids_match_legacy_rank_formula`` is DELETED with
# ``pipeline_fine._FineExpander._hw_id_for_rank``. The rank formula it compared
# against no longer exists; ``RankLayout`` is the one linearization and is
# gated directly by ``tests/test_program_layout.py``.


@pytest.mark.parametrize("grid", GRIDS, ids=[g.label for g in GRIDS])
def test_cluster_rank_roundtrip(grid: Grid) -> None:
    """P5: ``devices_for`` is injective on ``(WorkItem, cluster_rank)``."""
    placement, _ = _placement(grid, Granularity.FLAT)
    seen: Dict[int, Tuple[int, int]] = {}
    for stage in range(grid.pp):
        for rank in range(placement.cluster_size()):
            device = placement.device_for(stage, rank)
            assert device not in seen, f"device {device} claimed twice"
            seen[device] = (stage, rank)
            assert placement.cluster_rank_of(device) == rank
            assert placement.coords_of(device).of("pp") == stage
    assert sorted(seen) == list(placement.devices())


def test_placement_rules_key_off_workkind() -> None:
    """Embedding at device 0, softmax pinned to cluster rank 0 of the last
    stage, optimizer expanded per cluster rank — with no name in sight."""
    grid = Grid("rules", tp=2, cp=2, ep=1, pp=2, mb=2, num_layers=4)
    placement, _ = _placement(grid, Granularity.FLAT)
    cluster = placement.cluster_size()

    embedding = WorkItem(WorkKind.EMBEDDING, Direction.FORWARD, microbatch=0)
    assert placement.stage_of(embedding) == 0
    assert placement.devices_for(embedding) == (0,)

    softmax = WorkItem(WorkKind.SOFTMAX, Direction.FORWARD, microbatch=1)
    assert placement.stage_of(softmax) == grid.pp - 1
    assert placement.devices_for(softmax) == (placement.device_for(grid.pp - 1, 0),)

    optimizer = WorkItem(WorkKind.OPTIMIZER, Direction.BACKWARD, stage=1)
    assert placement.stage_of(optimizer) == 1
    assert placement.devices_for(optimizer) == placement.cluster_devices(1)
    assert len(placement.devices_for(optimizer)) == cluster

    layer = WorkItem(WorkKind.LAYER, Direction.BACKWARD, microbatch=0, layer=3)
    assert placement.stage_of(layer) == 1
    assert placement.devices_for(layer) == placement.cluster_devices(1)

    recompute = WorkItem(WorkKind.RECOMPUTE, Direction.FORWARD, microbatch=0, layer=3)
    assert placement.same_stage(layer, recompute)


def test_placement_policy_is_swappable_class_b_10d() -> None:
    """BUG_LEDGER Class B 10d is a named policy field, not an inline literal."""
    grid = Grid("b10d", tp=2, cp=2, ep=1, pp=2, mb=1, num_layers=2)
    softmax = WorkItem(WorkKind.SOFTMAX, Direction.FORWARD, microbatch=0)

    legacy, fw = _placement(grid, Granularity.FLAT)
    assert legacy.policy.name == "legacy"
    assert len(legacy.devices_for(softmax)) == 1

    sharded_policy = PlacementPolicy(
        name="softmax_sharded",
        spread_by_kind={
            **LEGACY_PLACEMENT.spread_by_kind,
            WorkKind.SOFTMAX: WorkSpread.PER_CLUSTER_RANK,
        },
    )
    sharded = Placement(
        fw,
        Granularity.FLAT,
        _LayerAssignment.contiguous(grid.num_layers, grid.pp),
        policy=sharded_policy,
    )
    assert sharded.devices_for(softmax) == sharded.cluster_devices(grid.pp - 1)
    assert len(sharded.devices_for(softmax)) == 4


def test_placement_rejects_bad_input() -> None:
    grid = GRIDS[1]
    placement, fw = _placement(grid, Granularity.FLAT)
    with pytest.raises(PlacementError):
        placement.device_for(grid.pp, 0)
    with pytest.raises(PlacementError):
        placement.device_for(0, placement.cluster_size())
    with pytest.raises(PlacementError, match="must be a Granularity"):
        Placement(fw, "fine", _LayerAssignment.contiguous(grid.num_layers, grid.pp))  # type: ignore[arg-type]
    with pytest.raises(PlacementError, match="does not declare a spread"):
        PlacementPolicy(name="partial", spread_by_kind={WorkKind.LAYER: WorkSpread.PER_CLUSTER_RANK})


def test_placement_rejects_layout_degree_mismatch() -> None:
    grid = Grid("mismatch", tp=2, cp=1, ep=1, pp=2, mb=1, num_layers=2)
    spec, _ = _workload(
        grid,
        layout=canonical_layout({"tp": 4, "cp": 1, "ep": 1, "pp": 2, "dp": 1}),
    )
    with pytest.raises(PlacementError, match="Inconsistent tensor/context/expert"):
        Placement(
            spec.freeze(),
            Granularity.FLAT,
            _LayerAssignment.contiguous(grid.num_layers, grid.pp),
        )


def test_device_coord_is_hashable_and_shiftable() -> None:
    grid = GRIDS[1]
    placement, _ = _placement(grid, Granularity.FLAT)
    coord = placement.coords_of(placement.device_for(0, 1))
    assert isinstance(coord, DeviceCoord)
    assert {coord: "x"}[DeviceCoord(dict(coord.coords))] == "x"
    hopped = coord.with_(pp=1)
    assert placement.device_of(hopped) == placement.device_for(1, 1)
    with pytest.raises(PlacementError):
        coord.of("nope")


# ---------------------------------------------------------------------------
# 2. Group construction
# ---------------------------------------------------------------------------


# P5: ``_legacy_axis_groups`` / ``test_single_axis_groups_match_legacy`` are
# DELETED with ``legacy_lowering._build_axis_groups`` /
# ``_compute_stage_axis_coords``. Group membership is CONSTRUCTED from the
# ``RankLayout`` now (invariants P2/P3) and is gated by the tests below, which
# assert members directly instead of against a second implementation.


def test_composite_tp_ep_group_is_declared_not_inferred() -> None:
    """The ``("tp","ep")`` communicator that ``legacy_lowering.py:243-248``
    recovers via ``participants == tp_size * ep_size`` is one ordinary call."""
    grid = Grid("tp_ep", tp=2, cp=2, ep=2, pp=2, mb=1, num_layers=2)
    placement, _ = _placement(grid, Granularity.FLAT)
    factory = placement.communicators

    # The legacy composite rule: every device sharing all coords EXCEPT tp/ep.
    def reference(device: int) -> Tuple[int, ...]:
        base = placement.layout.coords_of(device)
        keep = tuple(ax for ax in placement.layout.axis_order if ax not in ("tp", "ep"))
        out = []
        for other in placement.devices():
            other_coords = placement.layout.coords_of(other)
            if all(base[ax] == other_coords[ax] for ax in keep):
                out.append(other)
        return tuple(sorted(out))

    for device in placement.devices():
        members = factory.members(("tp", "ep"), device)
        assert members == reference(device)
        assert len(members) == grid.tp * grid.ep
        # Axis order is inert: membership is the identity.
        assert factory.members(("ep", "tp"), device) == members

    assert canonical_axis_label(("ep", "tp")) == "tp+ep"
    assert factory.group_for(("tp", "ep"), 0).axis == "tp+ep"
    # And the composite partition covers every device exactly once.
    partition = factory.partition(("tp", "ep"))
    flat = [device for group in partition for device in group]
    assert sorted(flat) == list(placement.devices())


def test_singleton_and_absent_axis_groups() -> None:
    """Class B 10g: a size-1 axis (or one the layout does not carry) yields a
    singleton group, which the emitter turns into a zero-duration noop."""
    grid = Grid("singleton", tp=2, cp=1, ep=1, pp=1, mb=1, num_layers=2)
    placement, _ = _placement(grid, Granularity.BLOCK)
    factory = placement.communicators
    assert factory.spans(("tp",)) == 2
    assert factory.spans(("cp",)) == 1
    assert factory.spans(("ep", "cp")) == 1
    for device in placement.devices():
        assert factory.members(("cp",), device) == (device,)
        # "pp" is not an axis of the BLOCK layout at all.
        assert factory.members(("pp",), device) == (device,)
    with pytest.raises(GroupError):
        factory.members((), 0)
    with pytest.raises(GroupError):
        factory.members(("nonsense",), 0)


def test_group_members_are_layout_derived_not_count_derived() -> None:
    """P2: two collectives with the same participant count but different
    declared axes get different member sets."""
    grid = Grid("p2", tp=2, cp=2, ep=1, pp=2, mb=1, num_layers=2)
    placement, _ = _placement(grid, Granularity.FLAT)
    factory = placement.communicators
    tp_group = factory.members(("tp",), 3)
    cp_group = factory.members(("cp",), 3)
    assert len(tp_group) == len(cp_group) == 2
    assert tp_group != cp_group


# ---------------------------------------------------------------------------
# 3. FLAT placement oracle: build_fine_program
# ---------------------------------------------------------------------------


# P5: the FLAT placement ORACLE (``_schedule_spec`` / ``_oracle_profile`` /
# ``_l2_profile`` / ``test_fine_placement_matches_build_fine_program``) is
# DELETED with ``pipeline_fine.build_fine_program``. It was the differential
# that had to pass before ``pipeline_fine.py`` could be deleted; it did, and it
# was. What survives is the intrinsic shape check below plus the end-to-end
# gate in ``tests/test_equiv_golden.py``.


@pytest.mark.parametrize("grid", GRIDS, ids=[g.label for g in GRIDS])
def test_fine_chains_are_well_formed(grid: Grid) -> None:
    """Every chain lives on a device of the placement layout (P1) and every
    LAYER expansion produces exactly one chain per cluster rank."""
    spec, _ = _workload(grid)
    fw = spec.freeze()
    placement = Placement(
        fw, Granularity.FLAT, _LayerAssignment.contiguous(grid.num_layers, grid.pp)
    )
    expander = BlockExpander(fw, placement)
    devices = set(placement.devices())

    for item in enumerate_work(fw, NoRecompute()):
        chains = expander.expand(item)
        seen = set()
        for chain in chains:
            assert chain.device in devices
            assert chain.device not in seen
            seen.add(chain.device)
            assert chain.entry == 0
            assert chain.exit == len(chain.steps) - 1
            for step in chain.steps[1:]:
                assert step.deps, "only the chain head may be dep-free"
        if item.kind in (WorkKind.LAYER, WorkKind.RECOMPUTE, WorkKind.OPTIMIZER):
            assert len(chains) == placement.cluster_size()
        else:
            assert len(chains) == 1


def test_coarse_expansion_is_one_op_per_workitem() -> None:
    grid = Grid("coarse", tp=2, cp=1, ep=1, pp=2, mb=2, num_layers=4)
    spec, _ = _workload(grid)
    fw = spec.freeze()
    placement = Placement(
        fw, Granularity.PIPELINE, _LayerAssignment.contiguous(grid.num_layers, grid.pp)
    )
    expander = BlockExpander(fw, placement)
    for item in enumerate_work(fw, NoRecompute()):
        chains = expander.expand(item)
        assert len(chains) == 1
        assert len(chains[0].steps) == 1
        assert chains[0].device == placement.stage_device(placement.stage_of(item))
    # Durations come from the FrozenDurations, keyed by L1's duration_key.
    embedding = WorkItem(WorkKind.EMBEDDING, Direction.FORWARD, microbatch=0)
    assert expander.expand(embedding)[0].steps[0].duration == DURATIONS["embedding_f"]


@pytest.mark.parametrize("granularity", [Granularity.PIPELINE, Granularity.FLAT])
def test_optimizer_node_is_priced_for_every_layer_its_stage_owns(
    granularity: Granularity,
) -> None:
    """BUG_LEDGER 10b, at the expander. ONE fused optimizer node per stage
    (structure unchanged), priced for ALL the layers that stage owns.

    The ``:ragged`` grid is the point: ``L=5, pp=3`` splits remainder-first into
    ``(2, 2, 1)``, so the stage-0 node is twice the stage-2 node and no single
    global scalar reproduces both. Every cluster rank of a stage carries the
    same duration (the apply-grad price is already per-rank).
    """
    grid = Grid("dp1tp3cp1pp3mb2:ragged", tp=3, cp=1, ep=1, pp=3, mb=2, num_layers=5)
    counts = _contiguous_counts(grid.num_layers, grid.pp)
    assert counts == (2, 2, 1)

    placement, fw = _placement(grid, granularity)
    expander = BlockExpander(fw, placement)
    per_layer = DURATIONS["optimizer"]
    for stage, layer_count in enumerate(counts):
        item = WorkItem(WorkKind.OPTIMIZER, Direction.BACKWARD, stage=stage)
        chains = expander.expand(item)
        assert chains
        for chain in chains:
            assert len(chain.steps) == 1
            assert chain.steps[0].duration == pytest.approx(layer_count * per_layer)

    # The total per pipeline replica is L x the per-layer price, independent of pp.
    total = sum(
        expander.expand(WorkItem(WorkKind.OPTIMIZER, Direction.BACKWARD, stage=stage))[0]
        .steps[0]
        .duration
        for stage in range(grid.pp)
    )
    assert total == pytest.approx(grid.num_layers * per_layer)


def test_block_expansion_covers_every_cluster_rank() -> None:
    grid = Grid("block", tp=2, cp=2, ep=1, pp=2, mb=1, num_layers=2)
    spec, _ = _workload(grid)
    fw = spec.freeze()
    placement = Placement(
        fw, Granularity.BLOCK, _LayerAssignment.contiguous(grid.num_layers, grid.pp)
    )
    expander = BlockExpander(fw, placement)
    layer = WorkItem(WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=0)
    chains = expander.expand(layer)
    assert tuple(chain.device for chain in chains) == placement.devices()
    assert len(chains) == grid.tp * grid.cp


# ---------------------------------------------------------------------------
# 4. CommSpec.placement ("pre"/"post") — the FLAT-path divergence
# ---------------------------------------------------------------------------


def test_fine_expansion_honors_pre_placement() -> None:
    """INTERFACES §3.4: the FLAT path MUST honor ``CommSpec.placement``.
    ``pipeline_fine.py:577-589`` chains every key post; this is one of the six
    flattened-MoE blockers. No production comm rule sets "pre", so the 42
    goldens are unaffected."""
    comm = {
        "pre_gather": {
            "size": 512,
            "type": CollectiveType.ALL_GATHER,
            "participants": 2,
            "interconnect_type": "tp",
            "placement": "pre",
        },
        "post_reduce": {
            "size": 1024,
            "type": CollectiveType.ALL_REDUCE,
            "participants": 2,
            "interconnect_type": "tp",
            "placement": "post",
        },
        "cross_layer": DENSE_COMM["cross_layer"],
    }
    gemms = (
        {
            "name": "qkv_proj",
            "forward": {"duration": 1e-4, "comm_keys": ["pre_gather", "post_reduce"]},
            "backward": {"duration": 2e-4, "comm_keys": []},
        },
    )
    grid = Grid("pre", tp=2, cp=1, ep=1, pp=1, mb=1, num_layers=1)
    spec, _ = _workload(grid, comm=comm, gemms=gemms)
    fw = spec.freeze()
    placement = Placement(fw, Granularity.FLAT, _LayerAssignment.contiguous(1, 1))
    chain = BlockExpander(fw, placement).expand(
        WorkItem(WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=0)
    )[0]

    kinds = [
        (step.kind, getattr(step, "comm_key", getattr(step, "entry_name", None)))
        for step in chain.steps
    ]
    assert kinds == [
        (StepKind.COMM, "pre_gather"),
        (StepKind.COMPUTE, "qkv_proj"),
        (StepKind.COMM, "post_reduce"),
    ]
    # The GEMM depends on the pre collective, not the other way around.
    assert chain.steps[1].deps == (0,)
    assert chain.steps[2].deps == (1,)


def test_unregistered_comm_key_raises() -> None:
    gemms = (
        {
            "name": "qkv_proj",
            "forward": {"duration": 1e-4, "comm_keys": ["missing_key"]},
            "backward": {"duration": 2e-4, "comm_keys": []},
        },
    )
    grid = Grid("missing", tp=1, cp=1, ep=1, pp=1, mb=1, num_layers=1)
    spec, _ = _workload(grid, gemms=gemms)
    fw = spec.freeze()
    placement = Placement(fw, Granularity.FLAT, _LayerAssignment.contiguous(1, 1))
    with pytest.raises(PlacementError, match="missing_key"):
        BlockExpander(fw, placement).expand(
            WorkItem(WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=0)
        )


# ---------------------------------------------------------------------------
# 5. MoE hot/cold routing join
# ---------------------------------------------------------------------------


MOE_COMM: Mapping[str, Mapping[str, Any]] = {
    **DENSE_COMM,
    "moe_a2a": {
        "size": 16384,
        "type": CollectiveType.ALL_TO_ALL,
        "participants": 2,
        "interconnect_type": "ep",
        "placement": "post",
        "parallel_group": "moe_route",
        "moe_component": "base_all_to_all",
        "moe_routing_mode": "ep",
    },
    "moe_residual": {
        "size": 4096,
        "type": CollectiveType.PIPELINE,
        "participants": 2,
        "interconnect_type": "ep",
        "placement": "post",
        "parallel_group": "moe_route",
        "moe_component": "residual_p2p",
        "moe_routing_mode": "ep",
    },
}

MOE_GEMMS: Tuple[Mapping[str, Any], ...] = (
    {
        "name": "qkv_proj",
        "forward": {"duration": 1.0e-4, "comm_keys": ["qkv_all_reduce"]},
        "backward": {"duration": 2.0e-4, "comm_keys": []},
    },
    {
        "name": "moe_dispatch",
        "forward": {"duration": 5.0e-4, "comm_keys": ["moe_a2a", "moe_residual"]},
        "backward": {"duration": 6.0e-4, "comm_keys": []},
    },
)


def _moe_placement(tp: int, ep: int, routing):
    grid = Grid("moe", tp=tp, cp=1, ep=ep, pp=1, mb=1, num_layers=1)
    mode = routing.name
    comm = {
        key: ({**value, "moe_routing_mode": mode} if "moe_component" in value else value)
        for key, value in MOE_COMM.items()
    }
    spec, _ = _workload(
        grid,
        comm=comm,
        gemms=MOE_GEMMS,
        moe_gemms=MOE_GEMMS,
        moe_layer_mask=(True,),
    )
    fw = spec.freeze()
    placement = Placement(fw, Granularity.FLAT, _LayerAssignment.contiguous(1, 1))
    return fw, placement, BlockExpander(fw, placement, routing=routing)


def test_moe_hot_rank_is_min_of_routing_group() -> None:
    """P4: hot rank == ``min(members)``, so hot-before-cold is guaranteed by
    construction rather than asserted at runtime (block_program.py:356-361)."""
    fw, placement, expander = _moe_placement(tp=2, ep=2, routing=EP_ROUTING)
    chains = expander.expand(
        WorkItem(WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=0)
    )
    assert len(chains) == 4

    by_device = {chain.device: chain for chain in chains}
    factory = placement.communicators
    for device, chain in by_device.items():
        members = factory.members(("ep",), device)
        hot = min(members)
        join = [step for step in chain.steps if step.kind is StepKind.JOIN]
        assert len(join) == 1
        residual = [
            step for step in chain.comm_steps() if step.comm_key == "moe_residual"
        ]
        if device == hot:
            # The hot rank carries no residual p2p; cold ranks send to it.
            assert residual == []
        else:
            assert len(residual) == 1
            assert residual[0].extra_consumers
            target = residual[0].extra_consumers[0]
            assert target.device == hot
            assert by_device[hot].steps[target.index].kind is StepKind.JOIN
    # Legacy _moe_hot_rank("ep") == _rank_id(tp, cp, 0): the ep=0 device.
    for device in placement.devices():
        coords = placement.coords_of(device)
        expected_hot = placement.device_of(coords.with_(ep=0))
        assert min(factory.members(("ep",), device)) == expected_hot


def test_moe_tp_ep_routing_uses_the_composite_group() -> None:
    """Legacy ``_moe_hot_rank("tp_ep") == _rank_id(0, cp, 0)``."""
    fw, placement, expander = _moe_placement(tp=2, ep=2, routing=TP_EP_ROUTING)
    factory = placement.communicators
    for device in placement.devices():
        members = factory.members(("tp", "ep"), device)
        assert len(members) == 4
        coords = placement.coords_of(device)
        assert min(members) == placement.device_of(coords.with_(tp=0, ep=0))
    chains = expander.expand(
        WorkItem(WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=0)
    )
    residual_senders = [
        chain.device
        for chain in chains
        if any(step.comm_key == "moe_residual" for step in chain.comm_steps())
    ]
    assert residual_senders == [1, 2, 3]  # everything but the single hot rank 0


def test_moe_group_without_routing_policy_raises() -> None:
    grid = Grid("moe_norouting", tp=1, cp=1, ep=2, pp=1, mb=1, num_layers=1)
    spec, _ = _workload(
        grid, comm=MOE_COMM, gemms=MOE_GEMMS, moe_gemms=MOE_GEMMS, moe_layer_mask=(True,)
    )
    fw = spec.freeze()
    placement = Placement(fw, Granularity.FLAT, _LayerAssignment.contiguous(1, 1))
    with pytest.raises(PlacementError, match="MoERoutingPolicy"):
        BlockExpander(fw, placement).expand(
            WorkItem(WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=0)
        )
    with pytest.raises(PlacementError, match="tp_ep"):
        BlockExpander(fw, placement, routing=TP_EP_ROUTING).expand(
            WorkItem(WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=0)
        )


# P5: the BLOCK placement ORACLE (``_block_oracle_profile`` /
# ``_block_l2_profile`` and the three ``*_matches_build_block_root`` tests) is
# DELETED with ``block_program.build_block_root``, for the same reason as the
# FLAT oracle above.


# ---------------------------------------------------------------------------
# 6. Sync spread resolution (the propagate_local_hw_ids replacement)
# ---------------------------------------------------------------------------


def test_devices_for_sync_resolves_every_spread() -> None:
    from program.work import (
        AttachMode,
        ByteSource,
        SyncKey,
        SyncPhase,
        SyncRequirement,
        SyncSpread,
    )

    grid = Grid("sync", tp=2, cp=2, ep=1, pp=2, mb=1, num_layers=2)
    spec, _ = _workload(grid)
    fw = spec.freeze()
    placement = Placement(
        fw, Granularity.FLAT, _LayerAssignment.contiguous(grid.num_layers, grid.pp)
    )
    layer = WorkItem(WorkKind.LAYER, Direction.BACKWARD, microbatch=0, layer=1)

    def _req(spread: SyncSpread) -> SyncRequirement:
        return SyncRequirement(
            key=SyncKey(comm_key="qkv_all_reduce", phase=SyncPhase.GRAD, layer=1, microbatch=0),
            bytes=ByteSource(key="qkv_all_reduce"),
            kind=CollectiveType.ALL_REDUCE,
            axes=("tp",),
            participants=2,
            place_on=layer,
            spread=spread,
            mode=AttachMode.AFTER,
            anchors=(layer,),
        )

    stage = placement.stage_of(layer)
    assert placement.devices_for_sync(_req(SyncSpread.STAGE)) == (
        placement.stage_device(stage),
    )
    assert placement.devices_for_sync(_req(SyncSpread.CLUSTER_RANK_0)) == (
        placement.device_for(stage, 0),
    )
    assert placement.devices_for_sync(_req(SyncSpread.PER_CLUSTER_RANK)) == (
        placement.cluster_devices(stage)
    )
    # And the communicator of that requirement is constructed, not inferred.
    groups = placement.communicators.groups_for(_req(SyncSpread.PER_CLUSTER_RANK), placement)
    assert len(groups) == placement.cluster_size()
    for group in groups:
        assert group.axis == "tp"
        assert len(group.members) == grid.tp


# ---------------------------------------------------------------------------
# 7. REGRESSION: the five Wave-1 composition defects (B1-B5)
# ---------------------------------------------------------------------------
#
# Each test below FAILS on the pre-fix tree. See the amendment notes in
# docs/rewrite/restructure/INTERFACES.md dated 2026-07-26.


def _sync_req(
    *,
    comm_key: str,
    axes: Tuple[str, ...],
    place_on: WorkItem,
    spread: Any,
    kind: CollectiveType = CollectiveType.ALL_REDUCE,
    participants: int = 2,
) -> Any:
    from program.work import AttachMode, ByteSource, SyncKey, SyncPhase, SyncRequirement

    return SyncRequirement(
        key=SyncKey(comm_key=comm_key, phase=SyncPhase.GRAD, microbatch=0, layer=0),
        bytes=ByteSource(key=comm_key),
        kind=kind,
        axes=axes,
        participants=participants,
        place_on=place_on,
        spread=spread,
        mode=AttachMode.AFTER,
        anchors=(place_on,),
    )


# -- B2: dp is not a communicator axis of a device layout --------------------

DP_COMM: Mapping[str, Mapping[str, Any]] = {
    "transformer_dense": {
        "size": 1000.5,
        "type": CollectiveType.ALL_REDUCE,
        "participants": 2,
        "interconnect_type": "dp",
    },
    "cross_layer": DENSE_COMM["cross_layer"],
}


@pytest.mark.parametrize(
    "granularity", [Granularity.PIPELINE, Granularity.FLAT, Granularity.BLOCK]
)
def test_b2_dp_requirement_gets_no_communicator(granularity: Granularity) -> None:
    """B2. ``Placement``'s device space carries ``dp``, so a dp-axis requirement
    used to receive a communicator whose members are NOT devices of the program
    (PIPELINE: ``dp:(0, 2)`` against ``devices() == (0, 1)``), and at BLOCK — where
    the layout has no dp at all — it silently degenerated to a singleton that
    ``et_emit`` substitutes with a zero-duration ``*_noop``, i.e. the reducer
    disappears. Contradicts INTERFACES §3.3 and ``ir.py:92-99``.

    Post-fix: ``groups_for`` returns ``None`` per instance device (the builder
    stamps ``group=None, is_dp=True``) and asking the factory for a dp group at
    all is a :class:`GroupError`.
    """
    from program.work import SyncSpread

    grid = Grid("b2", tp=2, cp=1, ep=1, pp=2, mb=1, num_layers=2)
    placement, _fw = _placement(grid, granularity, dp=2, comm=DP_COMM)

    # (a) THE DEFECT: a dp requirement resolves to instance devices but NO
    # GroupKey. Pre-fix this returned GroupKey(axis="dp", members=(0, 2)) at
    # PIPELINE — member 2 is not in devices() == (0, 1) — and GroupKey(members=(0,))
    # at BLOCK, a singleton et_emit substitutes with a zero-duration noop.
    layer = WorkItem(WorkKind.LAYER, Direction.BACKWARD, microbatch=0, layer=0)
    spread = (
        SyncSpread.STAGE if granularity is not Granularity.FLAT else SyncSpread.CLUSTER_RANK_0
    )
    req = _sync_req(
        comm_key="transformer_dense", axes=("dp",), place_on=layer, spread=spread
    )
    devices = placement.devices_for_sync(req)
    assert devices
    groups = placement.communicators.groups_for(req, placement)
    assert groups == tuple(None for _ in devices), (
        f"{granularity.name}: a dp collective must carry no GroupKey, got {groups}"
    )
    assert req.is_dp is True

    # (b) dp is not in the group layout, and it is a hard error to span it.
    assert "dp" not in placement.group_layout.axis_order
    with pytest.raises(GroupError, match="dp"):
        placement.communicators.members(("dp",), placement.devices()[0])
    with pytest.raises(GroupError, match="dp"):
        placement.communicators.group_for(("dp",), placement.devices()[0])

    # (c) every member of every non-dp group IS a device of this program.
    known = set(placement.devices())
    for axis in ("tp", "cp", "ep", "pp"):
        if axis not in placement.group_layout.axis_order:
            continue
        for device in placement.devices():
            for member in placement.communicators.members((axis,), device):
                assert member in known, (
                    f"{granularity.name}: group over {axis!r} at device {device} "
                    f"has member {member} which is not a device {sorted(known)}"
                )

    # (d) dropping dp from the group layout does not move a single device id.
    for device in placement.devices():
        coords = {
            axis: value
            for axis, value in placement.layout.coords_of(device).items()
            if axis != "dp"
        }
        assert placement.group_layout.linearize(coords) == device


def test_b2_group_partition_covers_exactly_the_device_set() -> None:
    """B2, the same defect seen through ``partition`` (which V7 uses): with dp in
    the layout, ``partition`` iterated ``num_ranks()`` = ``devices * dp`` and
    invented groups outside the device space."""
    grid = Grid("b2part", tp=2, cp=2, ep=1, pp=2, mb=1, num_layers=2)
    placement, _ = _placement(grid, Granularity.FLAT, dp=2, comm=DP_COMM)
    assert placement.group_layout.num_ranks() == len(placement.devices())
    for axis in ("tp", "cp", "pp"):
        flat = [d for group in placement.communicators.partition((axis,)) for d in group]
        assert sorted(flat) == list(placement.devices()), axis


# -- B4: PER_CLUSTER_RANK must span the stage --------------------------------


def test_b4_per_cluster_rank_spans_the_stage_when_place_on_is_pinned() -> None:
    """B4. ``SyncSpread.PER_CLUSTER_RANK`` degraded to ONE device whenever
    ``place_on``'s kind is pinned to cluster rank 0 by ``LEGACY_PLACEMENT``
    (EMBEDDING and SOFTMAX are) — which is exactly ZeRO-3 row **S7**
    (``zero3_transformer_gather``, ``tp_shard=True``, placed on
    ``EMBEDDING/FORWARD``) and row **S13** (placed on ``SOFTMAX/BACKWARD``).
    Legacy ``_ensure_zero3_per_rank_edges`` builds ``hw_ids`` for every
    ``par_degree`` rank (``pipeline_fine.py:471-480``).

    Invisible in the golden matrix only because every zero2/zero3 spec is
    ``tp=cp=1``.
    """
    from program.work import SyncSpread

    grid = Grid("b4", tp=2, cp=1, ep=1, pp=2, mb=1, num_layers=2)
    placement, _ = _placement(grid, Granularity.FLAT, dp=2, comm=DP_COMM)
    assert placement.cluster_size() == 2

    pinned = {
        "S7": WorkItem(WorkKind.EMBEDDING, Direction.FORWARD, microbatch=0),
        "S13": WorkItem(WorkKind.SOFTMAX, Direction.BACKWARD, microbatch=0),
    }
    for row, place_on in pinned.items():
        # LEGACY_PLACEMENT pins these kinds: devices_for is ONE device...
        assert len(placement.devices_for(place_on)) == 1, row
        req = _sync_req(
            comm_key="transformer_dense",
            axes=("dp",),
            place_on=place_on,
            spread=SyncSpread.PER_CLUSTER_RANK,
        )
        # ... but a PER_CLUSTER_RANK collective still exists on every rank.
        stage = placement.stage_of(place_on)
        assert placement.devices_for_sync(req) == placement.cluster_devices(stage), row
        assert len(placement.devices_for_sync(req)) == placement.cluster_size(), row

    # A non-pinned host is unaffected (the two used to agree only here).
    layer = WorkItem(WorkKind.LAYER, Direction.BACKWARD, microbatch=0, layer=1)
    req = _sync_req(
        comm_key="transformer_dense",
        axes=("dp",),
        place_on=layer,
        spread=SyncSpread.PER_CLUSTER_RANK,
    )
    assert placement.devices_for_sync(req) == placement.devices_for(layer)


def test_b4_per_cluster_rank_raises_when_it_cannot_span_the_cluster() -> None:
    """B4's declared guard: a PER_CLUSTER_RANK collective that resolves to fewer
    than ``cluster_size`` devices is a build error, never a silent single
    instance."""
    from program.work import SyncSpread

    grid = Grid("b4guard", tp=2, cp=1, ep=1, pp=2, mb=1, num_layers=2)
    placement, _ = _placement(grid, Granularity.FLAT, dp=2, comm=DP_COMM)
    layer = WorkItem(WorkKind.LAYER, Direction.BACKWARD, microbatch=0, layer=0)
    req = _sync_req(
        comm_key="transformer_dense",
        axes=("dp",),
        place_on=layer,
        spread=SyncSpread.PER_CLUSTER_RANK,
    )

    class _ShrunkCluster:
        """A stand-in whose cluster is smaller than the placement's."""

        def __init__(self, inner: Placement) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        def cluster_devices(self, stage: Any) -> Tuple[int, ...]:
            return self._inner.cluster_devices(stage)[:1]

    shrunk = _ShrunkCluster(placement)
    with pytest.raises(PlacementError, match="PER_CLUSTER_RANK"):
        Placement.devices_for_sync(shrunk, req)  # type: ignore[arg-type]


# -- B1: block comm keys are per template -----------------------------------
#
# Measured on train:hybrid:dp2tp2cp1pp2mb2sp1:moe:ep2 (the golden MoE rows):
# ep_dense_sync_layernorm1_backward is 337,641,472 bytes in the DENSE template
# and 67,141,632 in the MoE one — a 5.0288x ratio. train_timing's own
# _register_specs RAISES on exactly this conflict (train_timing.py:4694-4711).

EP_SYNC_KEY = "ep_dense_sync_layernorm1_backward"
EP_SYNC_DENSE_BYTES = 337_641_472
EP_SYNC_MOE_BYTES = 67_141_632


def _ep_sync_entry(size: int) -> Mapping[str, Any]:
    return {
        "size": size,
        "type": CollectiveType.ALL_REDUCE,
        "participants": 2,
        "interconnect_type": "ep",
        "placement": "post",
    }


def test_b1_block_expander_resolves_comm_through_the_layer_template() -> None:
    """B1. ``WorkloadSpec.comm`` was ONE flat table while block-template comm
    keys are named PER TEMPLATE, so the dense and MoE tables collide: a union
    keeps one entry and both MoE golden rows get 5.03x-wrong EP-sync bytes. With
    a mixed ``moe_layer_mask`` both values are needed simultaneously and a flat
    table cannot express it at all.
    """
    dense_comm = {**DENSE_COMM, EP_SYNC_KEY: _ep_sync_entry(EP_SYNC_DENSE_BYTES)}
    moe_comm = {**DENSE_COMM, EP_SYNC_KEY: _ep_sync_entry(EP_SYNC_MOE_BYTES)}
    gemms = (
        {
            "name": "layernorm1",
            "forward": {"duration": 1e-4, "comm_keys": []},
            "backward": {"duration": 2e-4, "comm_keys": [EP_SYNC_KEY]},
        },
    )
    #: layer 0 dense, layer 1 MoE — the mixed mask a flat table cannot express.
    grid = Grid("b1", tp=1, cp=1, ep=2, pp=1, mb=1, num_layers=2)
    spec, _ = _workload(
        grid,
        comm=dense_comm,
        gemms=gemms,
        moe_gemms=gemms,
        moe_comm=moe_comm,
        pipeline_comm={"cross_layer": DENSE_COMM["cross_layer"]},
        moe_layer_mask=(False, True),
    )
    fw = spec.freeze()

    # (a) THE DEFECT: the expansion of each layer resolves its OWN table.
    # Pre-fix the expander read the ONE flat ``WorkloadSpec.comm``, so a
    # block-template key was either absent from it or present exactly once —
    # both layers then got the same bytes (or a PlacementError).
    placement = Placement(fw, Granularity.FLAT, _LayerAssignment.contiguous(2, 1))
    expander = BlockExpander(fw, placement)
    by_layer = {}
    for layer in (0, 1):
        chain = expander.expand(
            WorkItem(WorkKind.LAYER, Direction.BACKWARD, microbatch=0, layer=layer)
        )[0]
        steps = [s for s in chain.comm_steps() if s.comm_key == EP_SYNC_KEY]
        assert len(steps) == 1, layer
        by_layer[layer] = steps[0].spec.size_bytes
    assert by_layer == {0: float(EP_SYNC_DENSE_BYTES), 1: float(EP_SYNC_MOE_BYTES)}

    # (b) both tables exist, keyed by MoE-ness, and they disagree by 5.03x.
    dense_spec = fw.spec.block_comm(0).require(EP_SYNC_KEY)
    moe_spec = fw.spec.block_comm(1).require(EP_SYNC_KEY)
    assert dense_spec.size_bytes == float(EP_SYNC_DENSE_BYTES)
    assert moe_spec.size_bytes == float(EP_SYNC_MOE_BYTES)
    assert dense_spec.size_bytes / moe_spec.size_bytes == pytest.approx(5.0288, abs=1e-4)

    # (c) the pipeline table does NOT carry block keys.
    assert EP_SYNC_KEY not in fw.spec.comm


def test_b1_pipeline_and_block_comm_namespaces_may_not_shadow() -> None:
    """B1's guard: a pipeline-level key that shadows a block key with different
    content is the flat-table defect again, so it is a construction error."""
    from program.workload import WorkloadError

    dense_comm = {**DENSE_COMM, EP_SYNC_KEY: _ep_sync_entry(EP_SYNC_DENSE_BYTES)}
    grid = Grid("b1shadow", tp=1, cp=1, ep=2, pp=1, mb=1, num_layers=1)
    with pytest.raises(WorkloadError, match="must not shadow"):
        _workload(
            grid,
            comm=dense_comm,
            pipeline_comm={
                "cross_layer": DENSE_COMM["cross_layer"],
                EP_SYNC_KEY: _ep_sync_entry(EP_SYNC_MOE_BYTES),
            },
        )


# -- B5: one object satisfies both stage seams ------------------------------


def test_b5_contiguous_stages_satisfies_placement_and_sharding_together() -> None:
    """B5. ``StagePartition.stage_of(work: WorkItem)`` and
    ``LayerAssignment.stage_of(layer: LayerId)`` shared a name with incompatible
    argument types, so NO object satisfied both — yet ``Placement`` consumes a
    ``LayerAssignment`` and ``ShardingContext`` a ``StagePartition``, and P3 must
    pass one stage partition to both.
    """
    from program.policies.gradaccum import GradAccumPolicy
    from program.policies.sharding import ContiguousStages, ShardingContext, ZeRO3

    grid = Grid("b5", tp=2, cp=1, ep=1, pp=2, mb=2, num_layers=4)
    spec, _ = _workload(grid, dp=2, comm=DP_COMM)
    fw = spec.freeze()
    stages = ContiguousStages.legacy(grid.num_layers, grid.pp)

    # (a) the LayerAssignment surface: Placement accepts it verbatim.
    placement = Placement(fw, Granularity.FLAT, stages)
    for layer in range(grid.num_layers):
        item = WorkItem(WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=layer)
        assert placement.stage_of(item) == stages.stage_of(layer)
    for stage in range(grid.pp):
        assert stages.layers_of(stage)
        assert stages.min_layer(stage) == stages.layers_of(stage)[0]
    assert stages.contiguous(grid.num_layers, grid.pp) == stages

    # (b) the StagePartition surface: the SAME object drives ShardingContext.
    work = enumerate_work(fw, NoRecompute())
    ctx = ShardingContext(
        fw=fw,
        work=work,
        grad_accum=GradAccumPolicy(dp=2, zero_stage=3, mode=DpMicrobatchMode.EVERY_MB),
        stages=stages,
    )
    first = WorkItem(WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=0)
    last = WorkItem(WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=grid.num_layers - 1)
    assert ctx.same_stage(first, first)
    assert not ctx.same_stage(first, last)
    assert ZeRO3().requirements(first, ctx) is not None

    # (c) and Placement itself is still a valid StagePartition (INTERFACES §3.2).
    assert placement.same_stage(first, first)
    assert placement.same_stage(first, last) is False
    ShardingContext(fw=fw, work=work, grad_accum=ctx.grad_accum, stages=placement)
