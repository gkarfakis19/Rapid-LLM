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

"""L2 tests: ``program/placement.py`` + ``program/groups.py`` (INTERFACES §3.5).

Three tiers:

1. **device sets per granularity** — COARSE/FINE/BLOCK over the parallelism
   grids of the golden matrix, plus a direct differential against the legacy
   rank formula ``_FineExpander._hw_id_for_rank`` (the thing L2 deletes);
2. **group construction** — single-axis groups differentially against
   ``legacy_lowering._build_axis_groups``, the composite ``("tp","ep")`` group
   that ``legacy_lowering.py:243-248`` recovers from a participant count, and
   the singleton (size-1 axis) case;
3. **the FINE placement oracle** — for every golden parallelism grid, the
   ``(device -> op multiset)`` mapping produced by ``Placement`` +
   ``BlockExpander`` must equal the one ``program.pipeline_fine.
   build_fine_program`` produces from the equivalent ``ScheduleSpec``.

Tier 3 is the load-bearing one: ``build_fine_program`` is the oracle L2 must
reproduce before ``pipeline_fine.py`` can be deleted in P3.

Run:
    ./.venv/bin/python -m pytest tests/test_placement.py -q
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
from program.schedule import ScheduleSpec, legacy_layers_per_stage
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
    """The contract's ``LayerAssignment`` (INTERFACES §4.1) reduced to the two
    members L2 consumes. Uses the legacy remainder-first split so the FINE
    oracle sees the same layer->stage map as ``ScheduleSpec.stage_for_layer``."""

    stage_of_layer: Tuple[int, ...]
    num_stages: int

    @classmethod
    def contiguous(cls, num_layers: int, pp: int) -> "_LayerAssignment":
        counts = legacy_layers_per_stage(num_layers, pp)
        mapping: List[int] = []
        for stage, count in enumerate(counts):
            mapping.extend([stage] * count)
        return cls(stage_of_layer=tuple(mapping), num_stages=max(1, int(pp)))

    def stage_of(self, layer: int) -> int:
        return self.stage_of_layer[layer]

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
    """One parallelism grid of the golden matrix (dp forced to 1 so the FINE
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
    moe = _template(moe_gemms, comm) if moe_gemms is not None else None
    spec = WorkloadSpec(
        degrees=degrees,
        shape=shape,
        run=run,
        comm=CommSpecTable.from_legacy(comm),
        blocks=BlockTemplates(dense=dense, moe=moe),
        overlap=OverlapSpec(parallelism_mode=None, by_axis={}),
        layout=layout
        if layout is not None
        else canonical_layout(
            {"tp": grid.tp, "cp": grid.cp, "ep": grid.ep, "pp": grid.pp, "dp": dp}
        ),
        interconnect={},
        granularity_hint=Granularity.FINE,
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

    coarse, _ = _placement(grid, Granularity.COARSE)
    assert coarse.cluster_size() == 1
    assert coarse.devices() == tuple(range(grid.pp))
    # At COARSE the device IS the stage.
    for stage in range(grid.pp):
        assert coarse.stage_device(stage) == stage

    fine, _ = _placement(grid, Granularity.FINE)
    assert fine.cluster_size() == cluster
    assert fine.devices() == tuple(range(grid.pp * cluster))

    block, _ = _placement(grid, Granularity.BLOCK)
    assert block.cluster_size() == cluster
    assert block.devices() == tuple(range(cluster))
    # BLOCK has no pipeline: every stage collapses onto the same cluster.
    for stage in range(grid.pp):
        assert block.cluster_devices(stage) == tuple(range(cluster))


@pytest.mark.parametrize("grid", GRIDS, ids=[g.label for g in GRIDS])
def test_fine_device_ids_match_legacy_rank_formula(grid: Grid) -> None:
    """Differential against ``_FineExpander._hw_id_for_rank`` — the hand-rolled
    rank formula L2 deletes (pipeline_fine.py:268-286)."""
    from program.pipeline_fine import _FineExpander

    spec, template = _workload(grid)
    fw = spec.freeze()
    placement = Placement(
        fw, Granularity.FINE, _LayerAssignment.contiguous(grid.num_layers, grid.pp)
    )
    expander = _FineExpander(
        _schedule_spec(grid, template), {"dense": template}, fw.spec.layout
    )

    for stage in range(grid.pp):
        for rank in range(placement.cluster_size()):
            assert placement.device_for(stage, rank) == expander._hw_id_for_rank(stage, rank)


@pytest.mark.parametrize("grid", GRIDS, ids=[g.label for g in GRIDS])
def test_cluster_rank_roundtrip(grid: Grid) -> None:
    """P5: ``devices_for`` is injective on ``(WorkItem, cluster_rank)``."""
    placement, _ = _placement(grid, Granularity.FINE)
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
    placement, _ = _placement(grid, Granularity.FINE)
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

    legacy, fw = _placement(grid, Granularity.FINE)
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
        Granularity.FINE,
        _LayerAssignment.contiguous(grid.num_layers, grid.pp),
        policy=sharded_policy,
    )
    assert sharded.devices_for(softmax) == sharded.cluster_devices(grid.pp - 1)
    assert len(sharded.devices_for(softmax)) == 4


def test_placement_rejects_bad_input() -> None:
    grid = GRIDS[1]
    placement, fw = _placement(grid, Granularity.FINE)
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
            Granularity.FINE,
            _LayerAssignment.contiguous(grid.num_layers, grid.pp),
        )


def test_device_coord_is_hashable_and_shiftable() -> None:
    grid = GRIDS[1]
    placement, _ = _placement(grid, Granularity.FINE)
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


def _legacy_axis_groups(layout: RankLayout, num_devices: int):
    """The legacy group table (``legacy_lowering._compute_stage_axis_coords`` +
    ``_build_axis_groups``) at dp_count=1, where stage == device."""
    from program.legacy_lowering import _build_axis_groups, _compute_stage_axis_coords

    axis_order = list(layout.axis_order)
    axis_sizes = dict(layout.axis_sizes)
    stage_ids = list(range(num_devices))
    coords = _compute_stage_axis_coords(stage_ids, axis_order, axis_sizes)
    stage_to_ranks = {stage: [stage] for stage in stage_ids}
    return _build_axis_groups(axis_order, axis_sizes, coords, stage_to_ranks), coords


@pytest.mark.parametrize("grid", GRIDS, ids=[g.label for g in GRIDS])
def test_single_axis_groups_match_legacy(grid: Grid) -> None:
    placement, _ = _placement(grid, Granularity.FINE)
    factory = placement.communicators
    devices = placement.devices()
    groups, coords = _legacy_axis_groups(placement.layout, len(devices))

    for axis in ("tp", "cp", "ep", "pp"):
        size = placement.layout.axis_sizes[axis]
        for device in devices:
            members = factory.members((axis,), device)
            if size <= 1:
                assert members == (device,), f"{axis} size 1 must give a singleton"
                continue
            key_base = tuple(
                (ax, coords[device].get(ax, 0))
                for ax in placement.layout.axis_order
                if ax != axis
            )
            expected = tuple(sorted(groups[axis][(0, key_base)]))
            assert members == expected, f"axis {axis} device {device}"
            assert len(members) == size


def test_composite_tp_ep_group_is_declared_not_inferred() -> None:
    """The ``("tp","ep")`` communicator that ``legacy_lowering.py:243-248``
    recovers via ``participants == tp_size * ep_size`` is one ordinary call."""
    grid = Grid("tp_ep", tp=2, cp=2, ep=2, pp=2, mb=1, num_layers=2)
    placement, _ = _placement(grid, Granularity.FINE)
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
    placement, _ = _placement(grid, Granularity.FINE)
    factory = placement.communicators
    tp_group = factory.members(("tp",), 3)
    cp_group = factory.members(("cp",), 3)
    assert len(tp_group) == len(cp_group) == 2
    assert tp_group != cp_group


# ---------------------------------------------------------------------------
# 3. FINE placement oracle: build_fine_program
# ---------------------------------------------------------------------------


def _schedule_spec(grid: Grid, template: BlockTemplate) -> ScheduleSpec:
    """The legacy ``ScheduleSpec`` equivalent of :func:`_workload`'s grid."""
    return ScheduleSpec(
        mb=grid.mb,
        num_layers=grid.num_layers,
        pp=grid.pp,
        dp=1,
        tp=grid.tp,
        cp=grid.cp,
        ep=grid.ep,
        layers_per_stage=legacy_layers_per_stage(grid.num_layers, grid.pp),
        moe_layer_mask=(),
        zero_stage=0,
        dp_microbatch_mode="every_mb",
        grad_accum_cycle="final",
        include_backward=grid.include_backward,
        include_optimizer=True,
        full_recomputation=grid.recompute,
        pipeline_style_recompute=grid.recompute,
        flattened_mode=True,
        model_type="gpt",
        comp_times=dict(DURATIONS),
        comm_metadata=dict(template.comm_metadata),
    )


def _oracle_profile(grid: Grid, template: BlockTemplate, layout: RankLayout):
    """``device -> Counter(op descriptor)`` from the legacy FINE builder.

    TransferOps are excluded: cross-layer p2p is dependency rule R2 of
    ``build()`` (INTERFACES §4.3), not L2 placement. dp is 1 in every grid, so
    every ``CollectiveOp`` present is a block-template collective.
    """
    from program.ir import CollectiveOp, ComputeOp
    from program.pipeline_fine import build_fine_program

    program = build_fine_program(
        _schedule_spec(grid, template), {"dense": template}, layout, dp_count=1
    )
    profile: Dict[int, Counter] = defaultdict(Counter)
    for op in program.ops:
        if isinstance(op, ComputeOp):
            profile[op.device][("COMP", round(float(op.duration[0]), 15))] += 1
        elif isinstance(op, CollectiveOp):
            profile[op.device][
                ("COLL", op.coll.name, int(op.size_bytes))
            ] += 1
    return {device: dict(counter) for device, counter in profile.items()}


def _l2_profile(fw, placement: Placement, recompute) -> Dict[int, Dict[Any, int]]:
    """``device -> Counter(op descriptor)`` from Placement + BlockExpander."""
    expander = BlockExpander(fw, placement)
    work = enumerate_work(fw, recompute)
    profile: Dict[int, Counter] = defaultdict(Counter)
    for item in work:
        for chain in expander.expand(item):
            for step in chain.steps:
                if isinstance(step, ComputeStep):
                    profile[chain.device][("COMP", round(float(step.duration), 15))] += 1
                else:
                    profile[chain.device][
                        ("COLL", step.spec.kind.name, int(step.spec.size_bytes))
                    ] += 1
    return {device: dict(counter) for device, counter in profile.items()}


@pytest.mark.parametrize("grid", GRIDS, ids=[g.label for g in GRIDS])
def test_fine_placement_matches_build_fine_program(grid: Grid) -> None:
    """THE ORACLE. ``Placement`` + ``BlockExpander`` must reproduce the legacy
    FINE builder's ``device -> op multiset`` mapping exactly."""
    spec, template = _workload(grid)
    fw = spec.freeze()
    placement = Placement(
        fw, Granularity.FINE, _LayerAssignment.contiguous(grid.num_layers, grid.pp)
    )
    recompute = FullRecompute() if grid.recompute else NoRecompute()

    expected = _oracle_profile(grid, template, fw.spec.layout)
    actual = _l2_profile(fw, placement, recompute)

    assert sorted(actual) == sorted(expected), (
        f"{grid.label}: device sets differ\n"
        f"  L2:     {sorted(actual)}\n"
        f"  legacy: {sorted(expected)}"
    )
    for device in sorted(expected):
        assert actual[device] == expected[device], (
            f"{grid.label}: device {device} op multiset differs\n"
            f"  L2:     {sorted(actual[device].items())}\n"
            f"  legacy: {sorted(expected[device].items())}"
        )


@pytest.mark.parametrize("grid", GRIDS, ids=[g.label for g in GRIDS])
def test_fine_chains_are_well_formed(grid: Grid) -> None:
    """Every chain lives on a device of the placement layout (P1) and every
    LAYER expansion produces exactly one chain per cluster rank."""
    spec, _ = _workload(grid)
    fw = spec.freeze()
    placement = Placement(
        fw, Granularity.FINE, _LayerAssignment.contiguous(grid.num_layers, grid.pp)
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
        fw, Granularity.COARSE, _LayerAssignment.contiguous(grid.num_layers, grid.pp)
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
# 4. CommSpec.placement ("pre"/"post") — the FINE-path divergence
# ---------------------------------------------------------------------------


def test_fine_expansion_honors_pre_placement() -> None:
    """INTERFACES §3.4: the FINE path MUST honor ``CommSpec.placement``.
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
    placement = Placement(fw, Granularity.FINE, _LayerAssignment.contiguous(1, 1))
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
    placement = Placement(fw, Granularity.FINE, _LayerAssignment.contiguous(1, 1))
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
    placement = Placement(fw, Granularity.FINE, _LayerAssignment.contiguous(1, 1))
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
    placement = Placement(fw, Granularity.FINE, _LayerAssignment.contiguous(1, 1))
    with pytest.raises(PlacementError, match="MoERoutingPolicy"):
        BlockExpander(fw, placement).expand(
            WorkItem(WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=0)
        )
    with pytest.raises(PlacementError, match="tp_ep"):
        BlockExpander(fw, placement, routing=TP_EP_ROUTING).expand(
            WorkItem(WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=0)
        )


def _block_oracle_profile(template: BlockTemplate, direction: str, grid: Grid):
    """``device -> Counter`` from the legacy BLOCK builder ``build_block_root``.

    This is the ONLY existing implementation of the MoE hot/cold join and of
    ``placement`` (pre/post) handling, so it is the oracle for both.
    """
    from program.block_program import build_block_root
    from program.pipeline_fine import FineEdge, FineNode

    root = build_block_root(template, direction, tp=grid.tp, cp=grid.cp, ep=grid.ep)
    profile: Dict[int, Counter] = defaultdict(Counter)
    seen = set()
    stack = [root]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        stack.extend(getattr(obj, "children", []))
        if isinstance(obj, FineNode):
            profile[int(obj.hw_id)][("COMP", round(float(obj.duration), 15))] += 1
        elif isinstance(obj, FineEdge):
            profile[int(obj.local_hw_id)][
                ("COLL", obj.comm_type.name, int(obj.comm_size_bytes))
            ] += 1
    return {device: dict(counter) for device, counter in profile.items()}


def _block_l2_profile(fw, placement: Placement, routing=None):
    expander = BlockExpander(fw, placement, routing=routing)
    profile: Dict[int, Counter] = defaultdict(Counter)
    for chain in expander.expand(
        WorkItem(WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=0)
    ):
        for step in chain.steps:
            if isinstance(step, ComputeStep):
                profile[chain.device][("COMP", round(float(step.duration), 15))] += 1
            else:
                profile[chain.device][
                    ("COLL", step.spec.kind.name, int(step.spec.size_bytes))
                ] += 1
    return {device: dict(counter) for device, counter in profile.items()}


BLOCK_GRIDS: Tuple[Grid, ...] = (
    Grid("block:tp1", tp=1, cp=1, ep=1, pp=1, mb=1, num_layers=1),
    Grid("block:tp2", tp=2, cp=1, ep=1, pp=1, mb=1, num_layers=1),
    Grid("block:tp2cp2", tp=2, cp=2, ep=1, pp=1, mb=1, num_layers=1),
    Grid("block:tp2ep2", tp=2, cp=1, ep=2, pp=1, mb=1, num_layers=1),
    Grid("block:tp2cp2ep2", tp=2, cp=2, ep=2, pp=1, mb=1, num_layers=1),
)


@pytest.mark.parametrize("grid", BLOCK_GRIDS, ids=[g.label for g in BLOCK_GRIDS])
def test_block_placement_matches_build_block_root_dense(grid: Grid) -> None:
    spec, template = _workload(grid)
    fw = spec.freeze()
    placement = Placement(fw, Granularity.BLOCK, _LayerAssignment.contiguous(1, 1))
    assert _block_l2_profile(fw, placement) == _block_oracle_profile(
        template, "forward", grid
    )


def test_block_pre_post_placement_matches_build_block_root() -> None:
    """``split_comm_keys`` differentially against ``block_program._split_comm_keys``
    — the only legacy implementation that honors ``placement``."""
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
        {
            "name": "MLP",
            "forward": {"duration": 3e-4, "comm_keys": ["pre_gather"]},
            "backward": {"duration": 4e-4, "comm_keys": []},
        },
    )
    grid = Grid("block:prepost", tp=2, cp=1, ep=1, pp=1, mb=1, num_layers=1)
    spec, template = _workload(grid, comm=comm, gemms=gemms)
    fw = spec.freeze()
    placement = Placement(fw, Granularity.BLOCK, _LayerAssignment.contiguous(1, 1))

    assert _block_l2_profile(fw, placement) == _block_oracle_profile(
        template, "forward", grid
    )
    # ... and the ORDER matches too, not just the multiset.
    chain = BlockExpander(fw, placement).expand(
        WorkItem(WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=0)
    )[0]
    assert [step.kind for step in chain.steps] == [
        StepKind.COMM,
        StepKind.COMPUTE,
        StepKind.COMM,
        StepKind.COMM,
        StepKind.COMPUTE,
    ]


@pytest.mark.parametrize(
    "grid",
    [g for g in BLOCK_GRIDS if g.ep > 1],
    ids=[g.label for g in BLOCK_GRIDS if g.ep > 1],
)
def test_block_moe_join_matches_build_block_root(grid: Grid) -> None:
    """The MoE hot/cold join, differentially: same ops on the same devices as
    ``block_program._attach_moe_parallel_post_group``, with ``_moe_hot_rank`` /
    ``_moe_parallel_token`` replaced by ``min(members)`` over a constructed
    communicator."""
    spec, template = _workload(
        grid,
        comm=MOE_COMM,
        gemms=MOE_GEMMS,
        moe_gemms=MOE_GEMMS,
        moe_layer_mask=(True,),
    )
    fw = spec.freeze()
    placement = Placement(fw, Granularity.BLOCK, _LayerAssignment.contiguous(1, 1))
    assert _block_l2_profile(fw, placement, routing=EP_ROUTING) == _block_oracle_profile(
        template, "forward", grid
    )


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
        fw, Granularity.FINE, _LayerAssignment.contiguous(grid.num_layers, grid.pp)
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
