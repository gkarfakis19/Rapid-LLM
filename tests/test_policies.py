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

"""L1 policy tests (INTERFACES.md §2.9 invariants K1-K6).

The correctness oracle while nothing is wired: for the SAME configuration,
build (a) the legacy event DAG via ``program.schedule.build_pipeline_events``
and (b) the L1 policy requirement set, then compare the **multiset of comm
keys**. Byte values, collective kinds, participants and axes are compared too;
op NAMES are not (they are not part of canonical equivalence,
``equiv/canonical.py:20-27``).

Oracle detail: the legacy builder identifies a comm event only by name, so the
harness swaps in a recording ``comm_metadata`` mapping that stamps a unique
sentinel ``size_bytes`` per ``__getitem__``; the sentinel is carried onto the
created ``CommEvent`` and maps it back to its comm key. Comm events that were
*created but never attached* (rows S6/S14 with no cross-device boundary —
INTERFACES §2.3 note 1) are excluded by walking the event DAG undirected from
the compute seeds, because the legacy lowering never collects them either: they
consume a legacy op id and nothing else (Class **C**).
"""

from __future__ import annotations

import dataclasses
import itertools
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from timing_model import CollectiveType

from program.block import BlockTemplate, comm_metadata_from_legacy
from program.policies import policies_for
from program.policies.gradaccum import GradAccumPolicy, grad_accum_policy_for
from program.policies.overlap import (
    AxisFractionOverlap,
    NoOverlap,
    OverlapAnchor,
    OverlapDecl,
    OverlapError,
)
from program.policies.recompute import (
    FullRecompute,
    NoRecompute,
    RecomputeError,
    recompute_policy_for,
)
from program.policies.routing import (
    EP_ROUTING,
    ROUTING_MODES,
    TP_EP_ROUTING,
    AxisRouting,
    MoEComponent,
    RoutingError,
    component_of,
    ep_sync_requirements,
    routing_for_mode,
    routing_policy_for,
)
from program.policies.sharding import (
    DDP,
    ContiguousStages,
    NullSharding,
    ShardingContext,
    ZeRO1,
    ZeRO2,
    ZeRO3,
    sharding_policy_for,
)
from program.schedule import ScheduleSpec, build_pipeline_events, legacy_layers_per_stage
from program.work import (
    AttachMode,
    ByteSource,
    ByteSplit,
    DepClass,
    Direction,
    SyncError,
    SyncKey,
    SyncPhase,
    SyncRequirement,
    SyncSpread,
    VIA_ALL,
    VIA_DATA_FLOW,
    VIA_NON_DATA_FLOW,
    WorkItem,
    WorkKind,
    attach_mode_arity,
    duration_for,
    duration_key,
    enumerate_work,
    is_moe,
    mem_kind,
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROGRAM_DIR = PROJECT_ROOT / "program"

ALL_REDUCE = CollectiveType.ALL_REDUCE
ALL_GATHER = CollectiveType.ALL_GATHER
REDUCE_SCATTER = CollectiveType.REDUCE_SCATTER
PIPELINE = CollectiveType.PIPELINE


# ---------------------------------------------------------------------------
# Fixture: one config -> (legacy ScheduleSpec, L0 WorkloadSpec)
# ---------------------------------------------------------------------------


COMP_TIMES: Dict[str, float] = {
    "embedding_f": 11.0,
    "embedding_b": 22.0,
    "linear_softmax_f": 13.0,
    "linear_softmax_b": 26.0,
    "transformer_f": 100.0,
    "transformer_b": 200.0,
    "transformer_f_dense": 100.0,
    "transformer_b_dense": 200.0,
    "transformer_f_moe": 150.0,
    "transformer_b_moe": 300.0,
    "optimizer": 7.0,
}


@dataclass(frozen=True)
class Cfg:
    """A parallelism/run configuration, expressed once and realized on both
    sides of the oracle."""

    dp: int = 1
    pp: int = 2
    tp: int = 1
    cp: int = 1
    ep: int = 1
    mb: int = 3
    num_layers: int = 4
    zero_stage: int = 0
    dp_microbatch: str = "every_mb"
    grad_accum_cycle: str = "final"
    moe: bool = False
    run_type: str = "training"
    full_recomputation: bool = False
    pipeline_style_recompute: bool = False
    flattened: bool = False

    @property
    def include_backward(self) -> bool:
        return self.run_type != "inference"

    @property
    def include_optimizer(self) -> bool:
        return self.grad_accum_cycle != "nonfinal"

    @property
    def moe_layer_mask(self) -> Tuple[bool, ...]:
        if not self.moe:
            return ()
        return tuple(idx % 2 == 0 for idx in range(self.num_layers))

    def label(self) -> str:
        return (
            f"dp{self.dp}-pp{self.pp}-ep{self.ep}-z{self.zero_stage}"
            f"-{'moe' if self.moe else 'dense'}-{self.grad_accum_cycle}"
            f"-{self.dp_microbatch}-mb{self.mb}-L{self.num_layers}"
            f"-{self.run_type}{'-rc' if self.full_recomputation else ''}"
        )


def raw_comm_metadata(cfg: Cfg) -> Dict[str, Dict[str, Any]]:
    """Mirror of ``train_timing._build_comm_metadata`` (:4380-4506) — the same
    keys, kinds, participants, interconnects and ``ga_required_every_cycle``
    flags, with byte values stubbed (bytes are upstream math; the seam is the
    dict shape)."""
    grad = REDUCE_SCATTER if (cfg.zero_stage >= 2 and cfg.dp > 1) else ALL_REDUCE
    md: Dict[str, Dict[str, Any]] = {
        "transformer_dense": {
            "size": 1000.5,
            "type": grad,
            "participants": cfg.dp,
            "interconnect_type": "dp",
            "local_comp_time": 3.5,
            "ga_required_every_cycle": False,
        },
        "transformer_moe": {
            "size": 2000.5,
            "type": grad,
            "participants": cfg.dp,
            "interconnect_type": "dp",
            "local_comp_time": 4.5,
            "ga_required_every_cycle": False,
        },
        "embedding": {
            "size": 3000.25,
            "type": grad,
            "participants": cfg.dp,
            "interconnect_type": "dp",
            "local_comp_time": 0,
            "ga_required_every_cycle": False,
        },
        "softmax": {
            "size": 4000.75,
            "type": grad,
            "participants": cfg.dp,
            "interconnect_type": "dp",
            "local_comp_time": 0,
            "ga_required_every_cycle": False,
        },
        "cross_layer": {
            "size": 4096,
            "type": PIPELINE,
            "participants": 2,
            "interconnect_type": "pp",
            "local_comp_time": 0,
        },
    }
    if cfg.ep > 1 and cfg.include_backward:
        for key in ("transformer_dense_ep_sync", "transformer_moe_ep_sync"):
            md[key] = {
                "size": 512,
                "type": grad,
                "participants": cfg.ep,
                "interconnect_type": "ep",
                "local_comp_time": 0,
            }
    if cfg.zero_stage == 2 and cfg.dp > 1:
        for key in (
            "zero2_embedding_gather",
            "zero2_transformer_gather",
            "zero2_softmax_gather",
        ):
            md[key] = {
                "size": 777,
                "type": ALL_GATHER,
                "participants": cfg.dp,
                "interconnect_type": "dp",
                "local_comp_time": 0,
                "ga_required_every_cycle": False,
            }
    if cfg.zero_stage >= 3 and cfg.dp > 1:
        md["zero3_embedding_gather"] = {
            "size": 888,
            "type": ALL_GATHER,
            "participants": cfg.dp,
            "interconnect_type": "dp",
            "local_comp_time": 0,
            "ga_required_every_cycle": True,
        }
        md["zero3_transformer_gather"] = {
            "size": 999,
            "type": ALL_GATHER,
            "participants": cfg.dp,
            "interconnect_type": "dp",
            "local_comp_time": 0,
            "tp_shard": True,
            "ga_required_every_cycle": True,
        }
        md["zero3_softmax_gather"] = {
            "size": 1111,
            "type": ALL_GATHER,
            "participants": cfg.dp,
            "interconnect_type": "dp",
            "local_comp_time": 0,
            "ga_required_every_cycle": True,
        }
    return md


class RecordingCommMetadata(dict):
    """A ``comm_metadata`` mapping that records every ``__getitem__`` and
    stamps a unique sentinel ``size_bytes`` so the created ``CommEvent`` can be
    mapped back to the comm key it came from.

    ``build_pipeline_events`` reads the table exactly once per keyed comm event
    (``_comm_event``, schedule.py:507-527); presence tests use ``in`` and the
    grad-accum flag uses ``.get``, neither of which is recorded.
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        super().__init__(data)
        self.reads: List[str] = []
        self.sentinel_to_key: Dict[float, str] = {}

    def __getitem__(self, key: str) -> Any:
        meta = super().__getitem__(key)
        self.reads.append(key)
        sentinel = float(len(self.reads))  # 1-based; 0 is the zero-byte control edges
        self.sentinel_to_key[sentinel] = key
        return dataclasses.replace(meta, size_bytes=sentinel)


def make_schedule_spec(cfg: Cfg) -> Tuple[ScheduleSpec, RecordingCommMetadata]:
    recorder = RecordingCommMetadata(comm_metadata_from_legacy(raw_comm_metadata(cfg)))
    spec = ScheduleSpec(
        mb=cfg.mb,
        num_layers=cfg.num_layers,
        pp=cfg.pp,
        dp=cfg.dp,
        tp=cfg.tp,
        cp=cfg.cp,
        ep=cfg.ep,
        layers_per_stage=legacy_layers_per_stage(cfg.num_layers, cfg.pp),
        moe_layer_mask=cfg.moe_layer_mask,
        zero_stage=cfg.zero_stage,
        dp_microbatch_mode=cfg.dp_microbatch,
        grad_accum_cycle=cfg.grad_accum_cycle,
        include_backward=cfg.include_backward,
        include_optimizer=cfg.include_optimizer,
        full_recomputation=cfg.full_recomputation,
        pipeline_style_recompute=cfg.pipeline_style_recompute,
        flattened_mode=cfg.flattened,
        model_type="gpt",
        comp_times=dict(COMP_TIMES),
        comm_metadata=recorder,
    )
    return spec, recorder


def _block_template(
    comm_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    backward_comm_keys: Tuple[str, ...] = (),
) -> BlockTemplate:
    """A two-GEMM block template with its OWN comm table.

    Block-template comm keys live on the template, not in
    ``WorkloadSpec.comm`` (INTERFACES §1.6 amendment 2026-07-26): the dense and
    MoE templates name their keys independently and may declare the same key
    with different bytes.
    """
    return BlockTemplate.from_gemm_entries(
        [
            {
                "name": "layernorm1",
                "forward": {"duration": 1.0, "comm_keys": []},
                "backward": {"duration": 2.0, "comm_keys": list(backward_comm_keys)},
            },
            {
                "name": "attention",
                "forward": {"duration": 3.0, "comm_keys": []},
                "backward": {"duration": 4.0, "comm_keys": []},
            },
        ],
        comm_metadata or {},
    )


def make_workload(cfg: Cfg) -> WorkloadSpec:
    raw = raw_comm_metadata(cfg)
    templates = BlockTemplates(
        dense=_block_template(),
        moe=_block_template() if cfg.moe else None,
    )
    return WorkloadSpec(
        degrees=ParallelDegrees(tp=cfg.tp, cp=cfg.cp, ep=cfg.ep, pp=cfg.pp, dp=cfg.dp),
        shape=ModelShape(
            num_layers=cfg.num_layers,
            micro_batches=cfg.mb,
            model_type="gpt",
            moe_layer_mask=cfg.moe_layer_mask,
        ),
        run=RunPolicy(
            run_type=RunType.parse(cfg.run_type),
            grad_accum_cycle=GradAccumCycle.parse(cfg.grad_accum_cycle),
            dp_microbatch_mode=DpMicrobatchMode.parse(cfg.dp_microbatch),
            zero_stage=cfg.zero_stage,
            full_recomputation=cfg.full_recomputation,
            pipeline_style_recompute=cfg.pipeline_style_recompute,
        ),
        comm=CommSpecTable.from_legacy(raw),
        blocks=templates,
        overlap=OverlapSpec(parallelism_mode="single", by_axis={}),
        layout=None,
        interconnect={"dp": (1.0, 0.0), "pp": (1.0, 0.0)},
        granularity_hint=None,
        durations=DurationTable(COMP_TIMES),
    )


def policy_requirements(cfg: Cfg) -> List[SyncRequirement]:
    """The complete L1 requirement set for ``cfg``, exactly as ``build()``
    phase 7 (INTERFACES §4.2) will assemble it."""
    fw = make_workload(cfg).freeze()
    bundle = policies_for(fw, block_expanded=cfg.flattened)
    work = enumerate_work(fw, bundle.recompute)
    ctx = ShardingContext(
        fw=fw,
        work=work,
        grad_accum=bundle.grad_accum,
        stages=ContiguousStages.legacy(cfg.num_layers, cfg.pp),
    )
    out: List[SyncRequirement] = list(bundle.sharding.workload_requirements(ctx))
    for item in work:
        out.extend(bundle.sharding.requirements(item, ctx))
        if bundle.routing is not None:
            out.extend(ep_sync_requirements(item, ctx))
    return out


# ---------------------------------------------------------------------------
# Oracle: the legacy lattice's attached comm events
# ---------------------------------------------------------------------------


def _walk_events(events: Any) -> List[Any]:
    """Every event ATTACHED to the schedule, reached undirected (children and
    parents) from the compute seeds. Comm events created but never attached
    (S6/S14 with no cross-device boundary) have neither, so they are excluded —
    exactly like ``legacy_lowering._collect_objects``, which never sees them."""
    seeds: List[Any] = [events.root]
    seeds.extend(events.embedding)
    seeds.extend(events.softmax)
    for row in events.layers:
        seeds.extend(row)

    seen: Dict[int, Any] = {}
    stack = [s for s in seeds if s is not None]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen[id(obj)] = obj
        for neighbour in list(getattr(obj, "children", [])) + list(getattr(obj, "parents", [])):
            if id(neighbour) not in seen:
                stack.append(neighbour)
    return list(seen.values())


def legacy_comm_keys(cfg: Cfg) -> Tuple[Counter, Counter]:
    """``(attached, created)`` comm-key multisets of the legacy lattice.

    ``cross_layer`` is excluded from both: it is data flow (R2), never a
    :class:`SyncRequirement` (``SyncRequirement`` rejects
    ``CollectiveType.PIPELINE`` outright).
    """
    spec, recorder = make_schedule_spec(cfg)
    events = build_pipeline_events(spec)

    attached: Counter = Counter()
    for obj in _walk_events(events):
        sentinel = getattr(obj, "comm_size_bytes", None)
        if not isinstance(sentinel, float):
            continue
        key = recorder.sentinel_to_key.get(sentinel)
        if key is not None and key != "cross_layer":
            attached[key] += 1

    created = Counter(k for k in recorder.reads if k != "cross_layer")
    return attached, created


# ---------------------------------------------------------------------------
# The config matrix
# ---------------------------------------------------------------------------


def _matrix() -> List[Cfg]:
    cfgs: List[Cfg] = []
    for dp, zero, ep, moe, cycle, mode in itertools.product(
        (1, 2),
        (0, 1, 2, 3),
        (1, 2),
        (False, True),
        ("final", "nonfinal"),
        ("every_mb", "last_mb"),
    ):
        if ep > 1 and not moe:
            # graph_ep == time_calc.ep only when use_moe (train_timing.py:4667),
            # so ep>1 without MoE templates is not a producible workload.
            continue
        cfgs.append(
            Cfg(dp=dp, zero_stage=zero, ep=ep, moe=moe, grad_accum_cycle=cycle, dp_microbatch=mode)
        )
    # pipeline shapes: pp=1 (no stage boundary -> rows S6/S14 are unattachable),
    # pp=4 with L=4 (a boundary at every layer), L < pp, and single-microbatch.
    for pp, num_layers, mb in ((1, 4, 3), (4, 4, 3), (2, 4, 1), (3, 2, 2), (2, 5, 4)):
        for dp, zero in itertools.product((1, 2), (0, 2, 3)):
            cfgs.append(Cfg(dp=dp, zero_stage=zero, pp=pp, num_layers=num_layers, mb=mb))
            cfgs.append(
                Cfg(dp=dp, zero_stage=zero, pp=pp, num_layers=num_layers, mb=mb, moe=True, ep=2)
            )
    # recompute on/off (changes the backward entry item, hence S15's anchors)
    for zero in (0, 2, 3):
        cfgs.append(
            Cfg(dp=2, zero_stage=zero, full_recomputation=True, flattened=True)
        )
        cfgs.append(
            Cfg(dp=2, zero_stage=zero, full_recomputation=True, pipeline_style_recompute=True)
        )
    # inference (no backward -> no lattice at all, even at dp>1)
    for dp in (1, 2):
        cfgs.append(Cfg(dp=dp, run_type="inference", zero_stage=3))
        cfgs.append(Cfg(dp=dp, run_type="inference", zero_stage=0, moe=True, ep=2))
    return cfgs


MATRIX = _matrix()


# ---------------------------------------------------------------------------
# K1/K5 — the completeness oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", MATRIX, ids=[c.label() for c in MATRIX])
def test_policy_comm_keys_match_legacy_lattice(cfg: Cfg) -> None:
    """The union of the policies' outputs is exactly the multiset of comm keys
    ``build_pipeline_events`` attaches for the same config."""
    attached, _created = legacy_comm_keys(cfg)
    mine = Counter(req.comm_key for req in policy_requirements(cfg))
    assert mine == attached, (
        f"{cfg.label()}\n"
        f"  policy-only : {sorted((mine - attached).items())}\n"
        f"  legacy-only : {sorted((attached - mine).items())}"
    )


@pytest.mark.parametrize("cfg", MATRIX, ids=[c.label() for c in MATRIX])
def test_requirement_specs_match_the_comm_table(cfg: Cfg) -> None:
    """K5 + the byte seam: kind/axes/participants/bytes on every requirement are
    the table's values, never re-derived."""
    fw = make_workload(cfg).freeze()
    for req in policy_requirements(cfg):
        spec = fw.spec.comm.require(req.comm_key)
        assert req.kind is spec.kind
        assert req.axes == spec.axes
        assert req.participants == spec.participants
        assert req.size_bytes(fw) == pytest.approx(spec.size_bytes)
        assert req.kind is not CollectiveType.PIPELINE


@pytest.mark.parametrize("cfg", MATRIX, ids=[c.label() for c in MATRIX])
def test_sync_keys_are_unique(cfg: Cfg) -> None:
    """K1: one requirement instance per :class:`SyncKey` — the identity is
    unique by construction, so ``AFTER(SyncKey)`` chaining is resolvable."""
    keys = [req.key for req in policy_requirements(cfg)]
    duplicates = [k for k, n in Counter(keys).items() if n > 1]
    assert not duplicates, f"{cfg.label()}: duplicate SyncKeys {duplicates}"


@pytest.mark.parametrize("cfg", MATRIX, ids=[c.label() for c in MATRIX])
def test_anchors_resolve(cfg: Cfg) -> None:
    """K1: every anchor is a member of the WorkSet or an earlier-emitted
    SyncKey; nothing is silently dropped."""
    fw = make_workload(cfg).freeze()
    bundle = policies_for(fw, block_expanded=cfg.flattened)
    work = enumerate_work(fw, bundle.recompute)
    reqs = policy_requirements(cfg)
    emitted = {req.key for req in reqs}
    for req in reqs:
        assert req.place_on in work, f"{req!r} placed on a non-existent WorkItem"
        for anchor in req.anchors:
            if isinstance(anchor, SyncKey):
                assert anchor in emitted, f"{req!r} chains onto unemitted {anchor!r}"
                assert req.mode is AttachMode.AFTER
            else:
                assert anchor in work, f"{req!r} anchors on a non-existent WorkItem"


def test_orphan_requirements_are_the_documented_class_c_divergence() -> None:
    """INTERFACES §2.3 note 1: rows S6/S14 are CREATED once per microbatch but
    attached only inside the cross-device branch. With no stage boundary the
    legacy lattice builds an object it never attaches — it consumes a legacy op
    id and is never lowered. The L1 model simply does not emit it (Class C:
    op-id numbering only, op multiset unchanged)."""
    cfg = Cfg(dp=2, zero_stage=3, pp=1, num_layers=4, mb=3)
    attached, created = legacy_comm_keys(cfg)
    orphans = created - attached
    # pp == 1: no forward boundary (mb-1 S6 orphans) and no backward boundary
    # (mb-1 S14 orphans).
    assert orphans == Counter(
        {"zero3_embedding_gather": cfg.mb - 1, "zero3_softmax_gather": cfg.mb - 1}
    )
    assert Counter(req.comm_key for req in policy_requirements(cfg)) == attached

    # ... and with a stage boundary there are no orphans at all.
    cfg2 = dataclasses.replace(cfg, pp=2)
    attached2, created2 = legacy_comm_keys(cfg2)
    assert created2 == attached2


# ---------------------------------------------------------------------------
# K2 — attach-mode coverage / exhaustiveness
# ---------------------------------------------------------------------------


def test_attach_mode_exhaustive() -> None:
    """K2: the dispatch over :class:`AttachMode` is exhaustive (the witness
    ends in ``typing.assert_never``)."""
    for mode in AttachMode:
        assert attach_mode_arity(mode) in (1, None)

    class _Fake:
        pass

    with pytest.raises(AssertionError):
        attach_mode_arity(_Fake())  # type: ignore[arg-type]


def test_attach_mode_census_reproduces_the_interfaces_table() -> None:
    """INTERFACES §2.3: 16 create-and-attach operations over 12 comm keys, with
    a fixed mode census. A ZeRO-3 + MoE + EP config exercises every row."""
    cfg = Cfg(dp=2, zero_stage=3, pp=2, num_layers=4, mb=3, moe=True, ep=2)
    reqs = policy_requirements(cfg)

    rows = Counter(req.origin for req in reqs)
    expected_rows = {
        "S1",  # zero3 embedding fwd entry (BEFORE)
        "S2",  # embedding reducer
        "S4",  # transformer reducer
        "S6",  # zero3 embedding fwd, cross-device only
        "S7",  # zero3 transformer, layer < prefetch_depth
        "S8",  # zero3 transformer, layer >= prefetch_depth
        "S9",  # softmax reducer
        "S11",  # zero3 softmax fwd
        "S12",  # EP grad sync
        "S13",  # zero3 softmax bwd entry
        "S14",  # zero3 softmax bwd, cross-device only
        "S15",  # zero3 transformer bwd
        "S16",  # zero3 embedding bwd
    }
    assert set(rows) == expected_rows

    modes = Counter(req.mode for req in reqs)
    assert modes[AttachMode.BEFORE] == 1  # S1 only
    assert modes[AttachMode.AFTER] == rows["S2"] + rows["S4"] + rows["S9"] + rows["S12"]
    assert modes[AttachMode.PARALLEL_TO] == (
        rows["S6"] + rows["S7"] + rows["S8"] + rows["S11"]
        + rows["S13"] + rows["S14"] + rows["S15"] + rows["S16"]
    )
    assert AttachMode.OVERLAP_WITH not in modes  # OverlapPolicy-only, honestly

    via = Counter(req.via for req in reqs if req.mode is AttachMode.PARALLEL_TO)
    assert via[VIA_NON_DATA_FLOW] == rows["S6"] + rows["S14"]
    assert via[VIA_DATA_FLOW] > 0  # the cross-device S8/S15 instances
    assert via[VIA_ALL] > 0

    # every non-PARALLEL_TO requirement carries the neutral via
    for req in reqs:
        if req.mode is not AttachMode.PARALLEL_TO:
            assert req.via == VIA_ALL


def test_zero2_chains_after_its_reducer() -> None:
    """Rows S3/S5/S10: the ZeRO-2 parameter gather is ``AFTER`` the reducer's
    :class:`SyncKey`, not after a WorkItem — a checkable property, not a
    construction-order coincidence."""
    cfg = Cfg(dp=2, zero_stage=2, pp=2, num_layers=4, mb=2)
    reqs = policy_requirements(cfg)
    gathers = [r for r in reqs if r.comm_key.startswith("zero2_")]
    assert gathers
    reducer_keys = {r.key for r in reqs if not r.comm_key.startswith("zero2_")}
    for gather in gathers:
        assert gather.mode is AttachMode.AFTER
        (anchor,) = gather.anchors
        assert isinstance(anchor, SyncKey)
        assert anchor in reducer_keys
        assert gather.origin in {"S3", "S5", "S10"}


def test_zero3_spread_is_tp_shard_driven() -> None:
    """A2 / B 10d: ``SyncSpread`` is DATA (``CommSpec.tp_shard``), so flipping
    the legacy ``rank_tails[0]``-only attach is a policy change."""
    cfg = Cfg(dp=2, zero_stage=3, pp=2, num_layers=4, mb=2)
    for req in policy_requirements(cfg):
        expected = (
            SyncSpread.PER_CLUSTER_RANK
            if req.comm_key == "zero3_transformer_gather"
            else SyncSpread.CLUSTER_RANK_0
        )
        assert req.spread is expected, req


def test_prefetch_depth_is_a_constructor_argument() -> None:
    """The +/-1 at schedule.py:873/:1018 IS the prefetch depth (ext_zero_policy
    is a constructor argument, not a new lattice)."""
    cfg = Cfg(dp=2, zero_stage=3, pp=1, num_layers=6, mb=2)
    fw = make_workload(cfg).freeze()
    work = enumerate_work(fw, NoRecompute())
    ctx = ShardingContext(
        fw=fw,
        work=work,
        grad_accum=grad_accum_policy_for(fw),
        stages=ContiguousStages.legacy(cfg.num_layers, cfg.pp),
    )
    for depth in (1, 2, 3):
        policy = ZeRO3(prefetch_depth=depth)
        reqs = [r for item in work for r in policy.requirements(item, ctx)]
        fwd = {
            r.key.layer: r
            for r in reqs
            if r.comm_key == "zero3_transformer_gather" and r.key.phase is SyncPhase.FWD
            and r.key.microbatch == 0
        }
        assert sorted(fwd) == list(range(cfg.num_layers))
        # layers shallower than the depth prefetch off the embedding (S7)
        for layer in range(depth):
            assert fwd[layer].origin == "S7"
        for layer in range(depth, cfg.num_layers):
            assert fwd[layer].origin == "S8"
            (anchor,) = fwd[layer].anchors
            assert anchor.layer == layer - depth
        # the requirement count never changes with the depth
        assert len(fwd) == cfg.num_layers

    with pytest.raises(Exception):
        ZeRO3(prefetch_depth=0)


# ---------------------------------------------------------------------------
# K3 — purity / order independence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cfg",
    [
        Cfg(dp=2, zero_stage=3, pp=2, moe=True, ep=2),
        Cfg(dp=2, zero_stage=2, pp=4, num_layers=4),
        Cfg(dp=2, zero_stage=0, pp=1, num_layers=3, mb=2),
    ],
    ids=lambda c: c.label(),
)
def test_requirements_order_independent(cfg: Cfg) -> None:
    """K3: shuffling the WorkSet iteration order yields the same requirement
    set — the policies are pure functions of ``(WorkItem, ShardingContext)``."""
    fw = make_workload(cfg).freeze()
    bundle = policies_for(fw, block_expanded=False)
    work = enumerate_work(fw, bundle.recompute)
    ctx = ShardingContext(
        fw=fw,
        work=work,
        grad_accum=bundle.grad_accum,
        stages=ContiguousStages.legacy(cfg.num_layers, cfg.pp),
    )

    def collect(items: List[WorkItem]) -> set:
        out = set()
        for item in items:
            for req in bundle.sharding.requirements(item, ctx):
                out.add(req)
            for req in ep_sync_requirements(item, ctx):
                out.add(req)
        return out

    baseline = collect(list(work))
    rng = random.Random(20260726)
    for _ in range(5):
        shuffled = list(work)
        rng.shuffle(shuffled)
        assert collect(shuffled) == baseline


def test_null_sharding_for_dp1_and_inference() -> None:
    for cfg in (
        Cfg(dp=1, zero_stage=3),
        Cfg(dp=2, zero_stage=3, run_type="inference"),
    ):
        fw = make_workload(cfg).freeze()
        policy = sharding_policy_for(fw.spec.run, fw.spec.degrees)
        assert isinstance(policy, NullSharding)
        assert policy_requirements(cfg) == []


def test_sharding_policy_selection() -> None:
    def pick(zero: int, dp: int = 2, run_type: str = "training"):
        fw = make_workload(Cfg(dp=dp, zero_stage=zero, run_type=run_type)).freeze()
        return sharding_policy_for(fw.spec.run, fw.spec.degrees)

    assert isinstance(pick(0), DDP) and pick(0).name == "ddp"
    assert isinstance(pick(1), ZeRO1) and pick(1).name == "zero1"
    assert isinstance(pick(2), ZeRO2) and pick(2).name == "zero2"
    assert isinstance(pick(3), ZeRO3) and pick(3).name == "zero3"
    # every legacy `zero_stage >= 3` test clamps
    assert isinstance(pick(4), ZeRO3)
    # ZeRO-1 is structurally DDP (train_timing.py:5128-5132)
    assert type(ZeRO1()).__mro__[1] is DDP


def test_zero1_emits_exactly_ddps_collectives() -> None:
    base = Cfg(dp=2, zero_stage=0, pp=2, num_layers=4, mb=2)
    one = dataclasses.replace(base, zero_stage=1)
    assert Counter(r.comm_key for r in policy_requirements(base)) == Counter(
        r.comm_key for r in policy_requirements(one)
    )


# ---------------------------------------------------------------------------
# K4/K6/W1 — grep gates
# ---------------------------------------------------------------------------

L1_MODULES = [
    PROGRAM_DIR / "types.py",
    PROGRAM_DIR / "workload.py",
    PROGRAM_DIR / "work.py",
    PROGRAM_DIR / "policies" / "__init__.py",
    PROGRAM_DIR / "policies" / "sharding.py",
    PROGRAM_DIR / "policies" / "gradaccum.py",
    PROGRAM_DIR / "policies" / "recompute.py",
    PROGRAM_DIR / "policies" / "routing.py",
    PROGRAM_DIR / "policies" / "overlap.py",
]


def _code_lines(path: Path) -> List[str]:
    """Source lines with comments and docstring bodies removed, so a grep gate
    tests the CODE and not the citations in the prose."""
    text = path.read_text()
    text = re.sub(r'"""(?:.|\n)*?"""', '""', text)
    text = re.sub(r"'''(?:.|\n)*?'''", "''", text)
    out = []
    for line in text.splitlines():
        stripped = re.sub(r"#.*$", "", line)
        if stripped.strip():
            out.append(stripped)
    return out


def test_grad_accum_predicate_has_exactly_one_implementation() -> None:
    """K4: the ``last_mb`` / ``dp_microbatch_mode`` predicate lives in exactly
    one L1 module. Today it is written twice (schedule.py:255-268 and its
    re-derivation at :937 for the EP sync)."""
    hits = {
        path.name
        for path in L1_MODULES
        if any("LAST_MB" in line for line in _code_lines(path))
    }
    assert hits == {"gradaccum.py", "workload.py"}, hits
    # workload.py only DEFINES the enum; the predicate is gradaccum's alone.
    gradaccum = _code_lines(PROGRAM_DIR / "policies" / "gradaccum.py")
    assert sum(1 for line in gradaccum if "DpMicrobatchMode.LAST_MB" in line) == 1


def test_no_getattr_in_workload() -> None:
    """W1: a malformed input raises ``WorkloadError`` at construction; there is
    no ``getattr(obj, name, default)`` chain anywhere in L0."""
    for path in (PROGRAM_DIR / "workload.py", PROGRAM_DIR / "work.py"):
        offenders = [line for line in _code_lines(path) if "getattr(" in line]
        assert not offenders, f"{path.name}: {offenders}"


def test_no_participant_count_inference() -> None:
    """P2 precondition: communicator membership is never derived from a
    participant count (legacy_lowering.py:243-248)."""
    for path in L1_MODULES:
        offenders = [
            line
            for line in _code_lines(path)
            if re.search(r"participants\s*==", line)
        ]
        assert not offenders, f"{path.name}: {offenders}"


def test_no_name_or_axis_literal_dispatch_in_expansion() -> None:
    """K6 / INTERFACES §8 rule 2: the axis and routing-mode literals appear only
    in the declared TABLES (``ROUTING_MODES``, the overlap anchor table, the
    reducer/gather key tables), never in an ``if`` inside a policy method."""
    offenders: List[Tuple[str, str]] = []
    for path in L1_MODULES:
        for line in _code_lines(path):
            if re.search(r"^\s*(el)?if\b.*['\"](tp_ep|ep|tp|cp|dp|pp)['\"]", line):
                offenders.append((path.name, line.strip()))
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# Routing (INTERFACES §2.6)
# ---------------------------------------------------------------------------


def _legacy_moe_hot_rank(tp_idx, cp_idx, mode, *, tp, cp):
    """Verbatim block_program.py:223-228."""
    def rank_id(t, c, e):
        return t + c * tp + e * tp * cp

    if mode == "ep":
        return rank_id(tp_idx, cp_idx, 0)
    if mode == "tp_ep":
        return rank_id(0, cp_idx, 0)
    raise ValueError(mode)


def _legacy_moe_parallel_token(tp_idx, cp_idx, mode):
    """Verbatim block_program.py:230-235."""
    if mode == "ep":
        return (mode, cp_idx, tp_idx)
    if mode == "tp_ep":
        return (mode, cp_idx)
    raise ValueError(mode)


@pytest.mark.parametrize("mode", ["ep", "tp_ep"])
def test_axis_routing_matches_the_deleted_if_chains(mode: str) -> None:
    """The two two-value if-chains, term by term. The join token's COMPONENT
    ORDER is inert (it is a dict key), so equality is checked as "induces the
    same partition of ranks"."""
    tp, cp, ep = 2, 2, 2
    policy = routing_for_mode(mode)

    legacy_hot: Dict[Tuple[int, int, int], int] = {}
    new_hot: Dict[Tuple[int, int, int], int] = {}
    legacy_groups: Dict[Any, set] = {}
    new_groups: Dict[Any, set] = {}
    for ep_idx, cp_idx, tp_idx in itertools.product(range(ep), range(cp), range(tp)):
        rank = tp_idx + cp_idx * tp + ep_idx * tp * cp
        coords = {"tp": tp_idx, "cp": cp_idx, "ep": ep_idx}
        legacy_hot[(tp_idx, cp_idx, ep_idx)] = _legacy_moe_hot_rank(
            tp_idx, cp_idx, mode, tp=tp, cp=cp
        )
        hot = policy.hot_coords(coords)
        new_hot[(tp_idx, cp_idx, ep_idx)] = (
            hot["tp"] + hot["cp"] * tp + hot["ep"] * tp * cp
        )
        legacy_groups.setdefault(
            _legacy_moe_parallel_token(tp_idx, cp_idx, mode), set()
        ).add(rank)
        new_groups.setdefault(policy.join_token(coords), set()).add(rank)

    assert new_hot == legacy_hot
    assert sorted(map(sorted, new_groups.values())) == sorted(
        map(sorted, legacy_groups.values())
    )
    # the hot rank of a group is min(members) — P4, by construction
    for members in new_groups.values():
        assert policy.hot_device(sorted(members)) == min(members)


def test_routing_axes_declare_the_composite_communicator() -> None:
    """The ``("tp","ep")`` communicator that ``legacy_lowering.py:243-248``
    infers from ``participants == tp*ep`` is DECLARED here and flows through
    ``CommSpec.axes``."""
    assert EP_ROUTING.routing_axes() == ("ep",)
    assert TP_EP_ROUTING.routing_axes() == ("tp", "ep")
    assert set(ROUTING_MODES) == {"ep", "tp_ep"}

    raw = {
        "moe_dispatch_forward_base_all_to_all": {
            "size": 4096,
            "type": CollectiveType.ALL_TO_ALL,
            "participants": 8,
            "interconnect_type": "ep",
            "placement": "post",
            "parallel_group": "moe_dispatch_forward_moe_parallel_group",
            "moe_component": "base_all_to_all",
            "moe_routing_mode": "tp_ep",
        }
    }
    table = CommSpecTable.from_legacy(raw)
    spec = table.require("moe_dispatch_forward_base_all_to_all")
    assert spec.axes == ("tp", "ep")
    assert spec.participants == 8  # analytical count stays independent of axes
    assert component_of(spec) is MoEComponent.BASE_ALL_TO_ALL


def test_a_third_routing_mode_is_a_table_row() -> None:
    """The extension test: a new mode is one :class:`AxisRouting` value."""
    cp_ep = AxisRouting(name="cp_ep", axes=("cp", "ep"))
    assert cp_ep.routing_axes() == ("cp", "ep")
    assert cp_ep.hot_coords({"tp": 1, "cp": 1, "ep": 1}) == {"tp": 1, "cp": 0, "ep": 0}
    assert cp_ep.join_token({"tp": 1, "cp": 1, "ep": 1}) == ("cp_ep", 1)

    with pytest.raises(RoutingError):
        routing_for_mode("nope")
    with pytest.raises(RoutingError):
        AxisRouting(name="bad", axes=("dp",))
    with pytest.raises(RoutingError):
        AxisRouting(name="bad", axes=())


def test_routing_policy_selection() -> None:
    dense = make_workload(Cfg(dp=2)).freeze()
    assert routing_policy_for(dense) is None
    with_ep = make_workload(Cfg(dp=2, moe=True, ep=2)).freeze()
    assert routing_policy_for(with_ep) is EP_ROUTING


def test_ep_sync_gate_is_the_grad_accum_predicate() -> None:
    """The EP gate written a second time at schedule.py:937 is one call to
    :meth:`GradAccumPolicy.emits` here (audit A6)."""
    for cycle, mode, zero, expect_any in (
        ("final", "every_mb", 0, True),
        ("final", "last_mb", 0, True),  # only microbatch 0
        ("final", "last_mb", 3, True),  # zero>=3 forces every mb
        ("nonfinal", "every_mb", 0, False),
    ):
        cfg = Cfg(dp=2, moe=True, ep=2, grad_accum_cycle=cycle, dp_microbatch=mode, zero_stage=zero)
        ep_reqs = [r for r in policy_requirements(cfg) if r.comm_key.endswith("_ep_sync")]
        assert bool(ep_reqs) is expect_any, cfg.label()
        if cycle == "final" and mode == "last_mb" and zero == 0:
            assert {r.key.microbatch for r in ep_reqs} == {0}


# ---------------------------------------------------------------------------
# GradAccumPolicy
# ---------------------------------------------------------------------------


def test_grad_accum_policy_truth_table() -> None:
    fw = make_workload(Cfg(dp=2, zero_stage=3)).freeze()
    reducer = fw.spec.comm.require("transformer_dense")
    gather = fw.spec.comm.require("zero3_transformer_gather")
    assert reducer.ga_required_every_cycle is False
    assert gather.ga_required_every_cycle is True

    off = GradAccumPolicy(dp=1)
    assert off.emits(reducer, 0) is False
    assert off.emits(gather, 0) is False

    nonfinal = GradAccumPolicy(dp=2, cycle=GradAccumCycle.NONFINAL)
    assert nonfinal.emits(reducer, 0) is False
    assert nonfinal.emits(gather, 0) is True

    last_mb = GradAccumPolicy(dp=2, mode=DpMicrobatchMode.LAST_MB)
    assert last_mb.emits(reducer, 0) is True
    assert last_mb.emits(reducer, 1) is False

    zero3 = GradAccumPolicy(dp=2, mode=DpMicrobatchMode.LAST_MB, zero_stage=3)
    assert zero3.emits(reducer, 1) is True


def test_last_microbatch_is_a_named_attribute_not_a_literal() -> None:
    """Class B 10h: backward walks microbatches in reverse, so ``b == 0`` IS the
    last microbatch under GPipe. Under 1F1B it is not, so the constant is DATA
    and :meth:`GradAccumPolicy.for_schedule` rebinds it."""
    fw = make_workload(Cfg(dp=2)).freeze()
    policy = grad_accum_policy_for(fw)
    assert policy.last_microbatch == 0
    reducer = fw.spec.comm.require("embedding")

    last_mb = dataclasses.replace(policy, mode=DpMicrobatchMode.LAST_MB)
    assert last_mb.emits(reducer, 0) is True
    assert last_mb.emits(reducer, 2) is False

    class _OneFOneB:
        def last_microbatch_of(self):
            return 2

    rebound = last_mb.for_schedule(_OneFOneB())
    assert rebound.last_microbatch == 2
    assert rebound.emits(reducer, 0) is False
    assert rebound.emits(reducer, 2) is True


# ---------------------------------------------------------------------------
# Recompute (INTERFACES §2.7)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "run_type,full,pipeline_style,block_expanded,expected",
    [
        ("training", False, False, False, "none"),
        ("training", True, False, False, "none"),
        ("training", True, False, True, "full"),
        ("training", True, True, False, "full"),
        ("training", False, True, True, "none"),
        ("inference", True, True, True, "none"),
    ],
)
def test_recompute_predicate(
    run_type: str, full: bool, pipeline_style: bool, block_expanded: bool, expected: str
) -> None:
    """Verbatim schedule.py:270-278, now an explicit dispatcher-time selection."""
    cfg = Cfg(
        run_type=run_type,
        full_recomputation=full,
        pipeline_style_recompute=pipeline_style,
    )
    fw = make_workload(cfg).freeze()
    assert recompute_policy_for(fw, block_expanded=block_expanded).name == expected


def test_recompute_requires_an_explicit_selection() -> None:
    fw = make_workload(Cfg()).freeze()
    with pytest.raises(RecomputeError):
        recompute_policy_for(fw)


def test_recompute_materializes_work_items() -> None:
    cfg = Cfg(dp=1, mb=2, num_layers=3, full_recomputation=True, flattened=True)
    fw = make_workload(cfg).freeze()
    with_rc = enumerate_work(fw, FullRecompute())
    without = enumerate_work(fw, NoRecompute())
    assert len(with_rc.of_kind(WorkKind.RECOMPUTE)) == cfg.mb * cfg.num_layers
    assert without.of_kind(WorkKind.RECOMPUTE) == ()


# ---------------------------------------------------------------------------
# Overlap (INTERFACES §2.8)
# ---------------------------------------------------------------------------


def _overlap_workload(mode: str, **fractions: float) -> Any:
    spec = make_workload(Cfg())
    return spec.with_(
        overlap=OverlapSpec.from_legacy(mode, **fractions)
    ).freeze()


def test_overlap_spec_resolves_tp_sp_once() -> None:
    """``tp_sp`` is not an axis: the mode -> fraction selection of
    transforms.py:326-331 happens once, at construction."""
    seq = OverlapSpec.from_legacy("tensor_sequence", tp_overlap=0.1, tp_sp_overlap=0.5)
    assert seq.fraction("tp") == 0.5
    tensor = OverlapSpec.from_legacy("tensor", tp_overlap=0.1, tp_sp_overlap=0.5)
    assert tensor.fraction("tp") == 0.1
    hybrid = OverlapSpec.from_legacy(
        "tensor_context_hybrid", tp_overlap=0.25, cp_overlap=0.75
    )
    assert hybrid.fraction("tp") == 0.25 and hybrid.fraction("cp") == 0.75
    ctx_only = OverlapSpec.from_legacy("context", tp_overlap=0.25, cp_overlap=0.75)
    assert ctx_only.fraction("tp") == 0.0 and ctx_only.fraction("cp") == 0.75
    single = OverlapSpec.from_legacy("single", tp_overlap=0.9, cp_overlap=0.9)
    assert single.by_axis == {}
    assert single.fraction("tp") == 0.0


def test_overlap_declaration_anchors() -> None:
    fw = _overlap_workload("tensor_context_hybrid", tp_overlap=0.25, cp_overlap=0.75)
    policy = AxisFractionOverlap()

    tp_spec = dataclasses.replace(fw.spec.comm.require("transformer_dense"), axes=("tp",))
    decl = policy.declare(tp_spec, fw)
    assert decl == OverlapDecl(fraction=0.25, anchor=OverlapAnchor.PRODUCER)
    assert decl.blocking_consumer is None

    cp_spec = dataclasses.replace(fw.spec.comm.require("transformer_dense"), axes=("cp",))
    decl = policy.declare(cp_spec, fw)
    assert decl.anchor is OverlapAnchor.CONSUMER
    assert decl.fraction == 0.75
    # the "attention" substring test of transforms.py:224 is DATA here
    assert decl.blocking_consumer == "attention"

    dp_spec = fw.spec.comm.require("transformer_dense")
    assert dp_spec.axes == ("dp",)
    assert policy.declare(dp_spec, fw) is None

    assert NoOverlap().declare(tp_spec, fw) is None


def test_overlap_declaration_validation() -> None:
    with pytest.raises(OverlapError):
        OverlapDecl(fraction=0.0, anchor=OverlapAnchor.PRODUCER)
    with pytest.raises(OverlapError):
        OverlapDecl(fraction=-1.0, anchor=OverlapAnchor.PRODUCER)
    with pytest.raises(OverlapError):
        OverlapDecl(fraction=0.5, anchor=OverlapAnchor.CONSUMER)
    assert OverlapDecl(fraction=1.0, anchor=OverlapAnchor.PRODUCER).is_hoist
    assert not OverlapDecl(fraction=0.9, anchor=OverlapAnchor.PRODUCER).is_hoist


def test_overlap_policy_selection() -> None:
    assert isinstance(policies_for(make_workload(Cfg()), block_expanded=False).overlap, NoOverlap)
    spec = make_workload(Cfg()).with_(
        overlap=OverlapSpec.from_legacy("tensor", tp_overlap=0.5)
    )
    assert isinstance(
        policies_for(spec, block_expanded=False).overlap, AxisFractionOverlap
    )


# ---------------------------------------------------------------------------
# WorkItem / WorkSet / SyncRequirement
# ---------------------------------------------------------------------------


def test_enumerate_work_shape() -> None:
    cfg = Cfg(dp=1, pp=2, mb=3, num_layers=4, moe=True, ep=1)
    fw = make_workload(cfg).freeze()
    work = enumerate_work(fw, NoRecompute())
    assert len(work.of_kind(WorkKind.EMBEDDING)) == 2 * cfg.mb
    assert len(work.of_kind(WorkKind.SOFTMAX)) == 2 * cfg.mb
    assert len(work.of_kind(WorkKind.LAYER)) == 2 * cfg.mb * cfg.num_layers
    # Class D 10b: one OPTIMIZER per pipeline stage, not per layer.
    assert len(work.of_kind(WorkKind.OPTIMIZER)) == cfg.pp
    assert work.items == tuple(sorted(work.items, key=WorkItem.sort_key))

    inference = make_workload(dataclasses.replace(cfg, run_type="inference")).freeze()
    inf_work = enumerate_work(inference, NoRecompute())
    assert all(item.direction is Direction.FORWARD for item in inf_work)
    assert inf_work.of_kind(WorkKind.OPTIMIZER) == ()

    nonfinal = make_workload(dataclasses.replace(cfg, grad_accum_cycle="nonfinal")).freeze()
    assert enumerate_work(nonfinal, NoRecompute()).of_kind(WorkKind.OPTIMIZER) == ()


def test_derived_facts_are_functions_not_fields() -> None:
    """A3 is unrepresentable: MoE-ness, duration key and mem kind are LOOKED UP
    from the ``WorkItem``, so a split op cannot lose them in a copy."""
    cfg = Cfg(dp=1, moe=True, num_layers=4)
    fw = make_workload(cfg).freeze()
    moe_layer = WorkItem(WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=0)
    dense_layer = WorkItem(WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=1)
    assert is_moe(moe_layer, fw) and not is_moe(dense_layer, fw)
    assert duration_key(moe_layer, fw) == "transformer_f_moe"
    assert duration_key(dense_layer, fw) == "transformer_f_dense"
    assert duration_for(moe_layer, fw) == COMP_TIMES["transformer_f_moe"]
    recomputed = dataclasses.replace(moe_layer, kind=WorkKind.RECOMPUTE)
    assert duration_key(recomputed, fw) == "transformer_f_moe"
    assert mem_kind(recomputed) is mem_kind(moe_layer)

    # the dense/moe fallback to the undifferentiated key, stated once
    stripped = make_workload(cfg)
    stripped = stripped.with_(
        durations=DurationTable(
            {k: v for k, v in COMP_TIMES.items() if not k.endswith(("_dense", "_moe"))}
        )
    ).freeze()
    assert duration_for(dense_layer, stripped) == COMP_TIMES["transformer_f"]


def test_work_item_validation() -> None:
    from program.work import WorkError

    with pytest.raises(WorkError):
        WorkItem(WorkKind.OPTIMIZER, Direction.BACKWARD)  # no stage
    with pytest.raises(WorkError):
        WorkItem(WorkKind.LAYER, Direction.FORWARD, microbatch=0)  # no layer
    with pytest.raises(WorkError):
        WorkItem(WorkKind.EMBEDDING, Direction.FORWARD, microbatch=0, layer=1)
    with pytest.raises(WorkError):
        WorkItem(WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=0, stage=1)
    with pytest.raises(WorkError):
        WorkItem(WorkKind.RECOMPUTE, Direction.BACKWARD, microbatch=0, layer=0)

    # ordering is total and never raises on None fields
    items = [
        WorkItem(WorkKind.OPTIMIZER, Direction.BACKWARD, stage=0),
        WorkItem(WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=1),
        WorkItem(WorkKind.EMBEDDING, Direction.FORWARD, microbatch=2),
    ]
    assert sorted(items) == sorted(items, key=WorkItem.sort_key)


def test_work_set_lookup_raises_not_keyerrors() -> None:
    from program.work import WorkError

    fw = make_workload(Cfg(dp=1)).freeze()
    work = enumerate_work(fw, NoRecompute())
    assert work.get(WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=0) is not None
    assert work.get(WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=99) is None
    with pytest.raises(WorkError):
        work.require(WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=99)


def _dummy_requirement(**overrides: Any) -> SyncRequirement:
    fw = make_workload(Cfg(dp=2)).freeze()
    spec = fw.spec.comm.require("embedding")
    item = WorkItem(WorkKind.EMBEDDING, Direction.BACKWARD, microbatch=0)
    kwargs: Dict[str, Any] = dict(
        key=SyncKey("embedding", SyncPhase.GRAD, 0),
        place_on=item,
        mode=AttachMode.AFTER,
        anchors=(item,),
    )
    kwargs.update(overrides)
    return SyncRequirement.from_spec(spec, **kwargs)


def test_sync_requirement_validation() -> None:
    _dummy_requirement()  # the happy path

    with pytest.raises(SyncError):
        _dummy_requirement(anchors=())
    item = WorkItem(WorkKind.EMBEDDING, Direction.BACKWARD, microbatch=0)
    other = WorkItem(WorkKind.EMBEDDING, Direction.BACKWARD, microbatch=1)
    with pytest.raises(SyncError):
        _dummy_requirement(mode=AttachMode.BEFORE, anchors=(item, other))
    with pytest.raises(SyncError):
        _dummy_requirement(via=VIA_DATA_FLOW)  # AFTER ignores via
    with pytest.raises(SyncError):
        _dummy_requirement(mode=AttachMode.PARALLEL_TO, anchors=(SyncKey("embedding", SyncPhase.GRAD, 0),))
    with pytest.raises(SyncError):
        _dummy_requirement(key=SyncKey("softmax", SyncPhase.GRAD, 0))

    # a PIPELINE collective is data flow, never a SyncRequirement
    fw = make_workload(Cfg(dp=2)).freeze()
    with pytest.raises(SyncError):
        SyncRequirement.from_spec(
            fw.spec.comm.require("cross_layer"),
            key=SyncKey("cross_layer", SyncPhase.FWD, 0),
            place_on=item,
            mode=AttachMode.AFTER,
            anchors=(item,),
        )

    # PARALLEL_TO admits any number of anchors >= 1 (rows S6/S14)
    multi = _dummy_requirement(
        key=SyncKey("embedding", SyncPhase.FWD, 0),
        mode=AttachMode.PARALLEL_TO,
        anchors=(item, other),
        via=VIA_NON_DATA_FLOW,
    )
    assert len(multi.anchors) == 2


def test_byte_source_splits() -> None:
    """B 1 (``WHOLE``) and D 10c (``CEIL_DIV_CLUSTER``) are named policy values,
    not literals in an expansion loop.

    ``instances`` is deliberately DIFFERENT from ``fw.spec.cluster_size()`` here:
    passing the two equal is what hid defect B3 (``bytes_for`` divided by
    ``cluster_size`` and ignored its argument).
    """
    fw = make_workload(Cfg(dp=2, tp=2, cp=2, ep=1)).freeze()
    assert fw.spec.cluster_size() == 4
    whole = ByteSource("cross_layer", ByteSplit.WHOLE)
    split = ByteSource("cross_layer", ByteSplit.CEIL_DIV_CLUSTER)
    total = fw.spec.comm.require("cross_layer").size_bytes
    assert whole.bytes_for(fw, 2) == total
    assert split.bytes_for(fw, 2) == pytest.approx(total / 2)

    odd = dataclasses.replace(fw.spec.comm.require("cross_layer"), size_bytes=4097.0)
    table = CommSpecTable(
        [odd] + [s for k, s in fw.spec.comm.items() if k != "cross_layer"]
    )
    fw_odd = fw.spec.with_(comm=table).freeze()
    assert split.bytes_for(fw_odd, 4) == 1025.0  # ceil, per pipeline_fine.py:645
    assert split.bytes_for(fw_odd, 2) == 2049.0


# ---------------------------------------------------------------------------
# REGRESSION: the five Wave-1 composition defects (B1-B5)
# ---------------------------------------------------------------------------
#
# Each test below FAILS on the pre-fix tree. See the amendment notes in
# docs/rewrite/restructure/INTERFACES.md dated 2026-07-26.


def test_b3_byte_source_honors_instances_not_cluster_size() -> None:
    """B3. ``ByteSource.bytes_for`` ignored its ``instances`` argument:
    ``CEIL_DIV_CLUSTER`` always divided by ``fw.spec.cluster_size()``.

    The two differ exactly where it matters. ``Placement.cluster_size()`` is 1
    at COARSE (the stage IS the device) while ``fw.spec.cluster_size()`` is
    always ``tp*cp*ep``, and legacy COARSE uses the RAW cross-layer byte count
    (``pipeline_coarse.py:222,238``) where legacy FINE divides
    (``pipeline_fine.py:645``). At ``tp=2`` the pre-fix call returned 2048.0
    where COARSE wants 4096.0 — wrong cross_layer bytes on every
    coarse/hybrid/hierarchical row with ``cluster_size > 1``.
    """
    fw = make_workload(Cfg(dp=1, tp=2, cp=1, ep=1)).freeze()
    assert fw.spec.cluster_size() == 2
    raw = fw.spec.comm.require("cross_layer").size_bytes
    assert raw == 4096.0

    split = ByteSource("cross_layer", ByteSplit.CEIL_DIV_CLUSTER)
    # COARSE: Placement.cluster_size() == 1 -> the RAW value.
    assert split.bytes_for(fw, 1) == raw
    # FINE: cluster_size instances -> divided.
    assert split.bytes_for(fw, 2) == raw / 2
    # ... and the divisor tracks `instances`, not the workload's cluster.
    assert split.bytes_for(fw, 4) == raw / 4
    assert split.bytes_for(fw, 8) == raw / 8

    # WHOLE never divides, whatever the instance count (BUG_LEDGER Class B 1).
    whole = ByteSource("cross_layer", ByteSplit.WHOLE)
    assert {whole.bytes_for(fw, n) for n in (1, 2, 4, 8)} == {raw}

    with pytest.raises(SyncError):
        split.bytes_for(fw, 0)


def test_b1_block_comm_tables_are_per_template() -> None:
    """B1. ``WorkloadSpec.comm`` is ONE flat table but block-template comm keys
    are named PER TEMPLATE, so the dense and MoE templates collide.

    ``train_timing._register_specs`` (:4694-4711) RAISES on this very conflict,
    and it is real: on ``train:hybrid:dp2tp2cp1pp2mb2sp1:moe:ep2``,
    ``ep_dense_sync_layernorm1_backward`` is 337,641,472 bytes dense and
    67,141,632 MoE (5.0288x). Both values are needed simultaneously under a
    mixed ``moe_layer_mask``.
    """
    dense_bytes, moe_bytes = 337_641_472, 67_141_632
    key = "ep_dense_sync_layernorm1_backward"

    def entry(size: int) -> Dict[str, Any]:
        return {
            "size": size,
            "type": ALL_REDUCE,
            "participants": 2,
            "interconnect_type": "ep",
            "placement": "post",
        }

    cfg = Cfg(dp=2, tp=2, cp=1, ep=2, pp=2, num_layers=4, moe=True)
    templates = BlockTemplates(
        dense=_block_template({key: entry(dense_bytes)}, backward_comm_keys=(key,)),
        moe=_block_template({key: entry(moe_bytes)}, backward_comm_keys=(key,)),
    )
    spec = make_workload(cfg).with_(blocks=templates)
    fw = spec.freeze()

    # cfg.moe_layer_mask alternates, so both templates are live at once.
    mask = cfg.moe_layer_mask
    assert set(mask) == {True, False}

    for layer, is_moe_layer in enumerate(mask):
        table = fw.spec.block_comm(layer)
        expected = moe_bytes if is_moe_layer else dense_bytes
        assert table.require(key).size_bytes == float(expected), layer

    # The two tables are distinct objects with distinct content ...
    assert templates.comm_for(False) is not templates.comm_for(True)
    assert (
        templates.comm_for(False).require(key).size_bytes
        / templates.comm_for(True).require(key).size_bytes
        == pytest.approx(5.0288, abs=1e-4)
    )
    # ... and the pipeline-level table carries no block key at all.
    assert key not in fw.spec.comm
    # A whole-workload SCAN still sees both (routing_policy_for uses this).
    assert sorted(
        s.size_bytes for s in fw.spec.all_comm_specs() if s.key == key
    ) == [float(moe_bytes), float(dense_bytes)]


def test_b1_moe_routing_mode_is_found_on_the_block_tables() -> None:
    """B1 consequence: ``moe_routing_mode`` is only ever set by
    ``train_timing._make_moe_comm_specs`` (:601,:617), whose specs are registered
    on the BLOCK template — never in ``_build_comm_metadata``. Once the block
    tables stop being unioned into ``WorkloadSpec.comm``, a scan of that table
    alone finds nothing."""
    a2a = {
        "moe_dispatch_forward_base_all_to_all": {
            "size": 16384,
            "type": CollectiveType.ALL_TO_ALL,
            "participants": 2,
            "interconnect_type": "ep",
            "placement": "post",
            "parallel_group": "moe_route",
            "moe_component": "base_all_to_all",
            "moe_routing_mode": "tp_ep",
        }
    }
    cfg = Cfg(dp=2, tp=2, ep=2, moe=True)
    templates = BlockTemplates(
        dense=_block_template(),
        moe=_block_template(a2a, backward_comm_keys=tuple(a2a)),
    )
    fw = make_workload(cfg).with_(blocks=templates).freeze()
    assert not any(s.moe_routing_mode for s in fw.spec.comm.values())
    assert routing_policy_for(fw) is TP_EP_ROUTING


def test_b2_dp_requirement_declares_itself_dp() -> None:
    """B2 (L1 half): a dp collective is DECLARED as having no communicator.

    ``GroupKey.members`` are pre-DP device ids (``ir.py:92-99``); dp replication
    is stamped at emission. A dp requirement therefore lowers to
    ``CollectiveOp(group=None, is_dp=True)`` and ``dp`` may never appear as one
    axis of a device-layout communicator.
    """
    fw = make_workload(Cfg(dp=2, tp=2)).freeze()
    item = WorkItem(WorkKind.LAYER, Direction.BACKWARD, microbatch=0, layer=0)

    dp_req = SyncRequirement.from_spec(
        fw.spec.comm.require("transformer_dense"),
        key=SyncKey("transformer_dense", SyncPhase.GRAD, 0, 0),
        place_on=item,
        mode=AttachMode.AFTER,
        anchors=(item,),
    )
    assert dp_req.axes == ("dp",)
    assert dp_req.is_dp is True

    tp_spec = dataclasses.replace(
        fw.spec.comm.require("transformer_dense"), axes=("tp",)
    )
    tp_req = SyncRequirement.from_spec(
        tp_spec,
        key=SyncKey("transformer_dense", SyncPhase.GRAD, 0, 0),
        place_on=item,
        mode=AttachMode.AFTER,
        anchors=(item,),
    )
    assert tp_req.is_dp is False

    # A composite communicator including dp is a declaration bug, not a group.
    mixed = dataclasses.replace(
        fw.spec.comm.require("transformer_dense"), axes=("dp", "tp")
    )
    with pytest.raises(SyncError, match="dp"):
        SyncRequirement.from_spec(
            mixed,
            key=SyncKey("transformer_dense", SyncPhase.GRAD, 0, 0),
            place_on=item,
            mode=AttachMode.AFTER,
            anchors=(item,),
        )


def test_b5_stage_partition_protocol_is_satisfiable() -> None:
    """B5. ``StagePartition`` and ``LayerAssignment`` both declared ``stage_of``
    with incompatible argument types (``WorkItem`` vs ``LayerId``), so no object
    satisfied both — yet L1's ``ShardingContext`` needs the first and L2's
    ``Placement`` the second, from the same stage partition.

    Post-fix: ``StagePartition`` declares ONLY ``same_stage``; ``ContiguousStages``
    carries ``LayerAssignment``'s ``stage_of(layer)`` / ``layers_of`` /
    ``min_layer`` and exposes the WorkItem rule as ``stage_of_work``.
    """
    from program.policies.sharding import StagePartition

    # The protocol declares exactly one member, and it is not `stage_of`.
    declared = {
        name
        for name, value in vars(StagePartition).items()
        if not name.startswith("_") and callable(value)
    }
    assert declared == {"same_stage"}

    stages = ContiguousStages.legacy(5, 3)
    # LayerAssignment surface: keyed on a LayerId.
    assert [stages.stage_of(layer) for layer in range(5)] == [0, 0, 1, 1, 2]
    assert stages.layers_of(0) == (0, 1)
    assert stages.min_layer(2) == 4
    with pytest.raises(Exception):
        stages.stage_of(5)

    # StagePartition surface: keyed on WorkItems, via stage_of_work.
    embedding = WorkItem(WorkKind.EMBEDDING, Direction.FORWARD, microbatch=0)
    softmax = WorkItem(WorkKind.SOFTMAX, Direction.FORWARD, microbatch=0)
    layer4 = WorkItem(WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=4)
    optimizer = WorkItem(WorkKind.OPTIMIZER, Direction.BACKWARD, stage=1)
    assert stages.stage_of_work(embedding) == 0
    assert stages.stage_of_work(softmax) == 2
    assert stages.stage_of_work(layer4) == 2
    assert stages.stage_of_work(optimizer) == 1
    assert stages.same_stage(softmax, layer4)
    assert not stages.same_stage(embedding, softmax)

    # The SAME object drives both seams: ShardingContext(stages=...) uses
    # same_stage, Placement(layers=...) uses stage_of(layer). Neither shadows the
    # other any more. (The Placement half is exercised in tests/test_placement.py
    # ::test_b5_contiguous_stages_satisfies_placement_and_sharding_together.)
    fw = make_workload(Cfg(dp=2, pp=3, num_layers=5)).freeze()
    ctx = ShardingContext(
        fw=fw,
        work=enumerate_work(fw, NoRecompute()),
        grad_accum=grad_accum_policy_for(fw),
        stages=stages,
    )
    assert ctx.same_stage(softmax, layer4)


def test_dep_class_via_sets() -> None:
    """``skip_non_comm_children`` / ``skip_comm_children`` are gone; the
    successor-edge classes they selected are now named sets."""
    assert VIA_DATA_FLOW == frozenset({DepClass.DATA_FLOW})
    assert VIA_NON_DATA_FLOW == frozenset({DepClass.SCHEDULE, DepClass.SYNC})
    assert VIA_ALL == frozenset(DepClass)
    assert VIA_DATA_FLOW | VIA_NON_DATA_FLOW == VIA_ALL


def test_policy_bundle_names() -> None:
    bundle = policies_for(make_workload(Cfg(dp=2, zero_stage=3, moe=True, ep=2)), block_expanded=True)
    assert bundle.names == {
        "sharding": "zero3",
        "grad_accum": "legacy",
        "recompute": "none",
        "overlap": "none",
        "routing": "ep",
    }
    # every policy is a frozen dataclass carrying a name
    for policy in (bundle.sharding, bundle.grad_accum, bundle.recompute, bundle.overlap, bundle.routing):
        assert dataclasses.is_dataclass(policy)
        assert isinstance(policy.name, str) and policy.name
        with pytest.raises(dataclasses.FrozenInstanceError):
            policy.name = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Structural oracle: the ATTACHMENT, not just the existence
# ---------------------------------------------------------------------------
#
# The comm-key multiset above proves every collective EXISTS. This second
# oracle proves each one attaches the way the legacy lattice attached it — i.e.
# that the typed `via` reproduces `skip_non_comm_children` / `skip_comm_children`
# site by site. It observes `attach_parallel_edge`'s effect by logging every
# `add_child` call and replaying it, so nothing is re-derived from the code
# under test.

_ZERO3_NAME_PATTERNS = (
    (re.compile(r"^(zero3_embedding_gather)_b(\d+)_fwd_entry$"), SyncPhase.FWD_ENTRY, False),
    (re.compile(r"^(zero3_softmax_gather)_b(\d+)_bwd_entry$"), SyncPhase.BWD_ENTRY, False),
    (re.compile(r"^(zero3_transformer_gather)_b(\d+)_layer(\d+)_fwd$"), SyncPhase.FWD, True),
    (re.compile(r"^(zero3_transformer_gather)_b(\d+)_layer(\d+)_bwd$"), SyncPhase.BWD, True),
    (re.compile(r"^(zero3_embedding_gather)_b(\d+)_fwd$"), SyncPhase.FWD, False),
    (re.compile(r"^(zero3_embedding_gather)_b(\d+)_bwd$"), SyncPhase.BWD, False),
    (re.compile(r"^(zero3_softmax_gather)_b(\d+)_fwd$"), SyncPhase.FWD, False),
    (re.compile(r"^(zero3_softmax_gather)_b(\d+)_bwd$"), SyncPhase.BWD, False),
)


def _sync_key_from_legacy_name(name: str) -> Optional[SyncKey]:
    for pattern, phase, has_layer in _ZERO3_NAME_PATTERNS:
        match = pattern.match(name)
        if match is None:
            continue
        key = match.group(1)
        microbatch = int(match.group(2))
        layer = int(match.group(3)) if has_layer else None
        return SyncKey(key, phase, microbatch, layer)
    return None


def _legacy_events_with_log(cfg: Cfg, monkeypatch: Any) -> Tuple[Any, List[Tuple[Any, Any]]]:
    """Build the legacy event DAG while logging every ``add_child`` call."""
    import program.schedule as legacy_schedule

    log: List[Tuple[Any, Any]] = []

    def _wrap(cls: Any) -> None:
        original = cls.add_child

        def add_child(self: Any, obj: Any) -> None:
            log.append((self, obj))
            original(self, obj)

        monkeypatch.setattr(cls, "add_child", add_child)

    _wrap(legacy_schedule.ComputeEvent)
    _wrap(legacy_schedule.CommEvent)

    spec, _recorder = make_schedule_spec(cfg)
    return build_pipeline_events(spec), log


def _legacy_compute_index(events: Any, cfg: Cfg) -> Dict[WorkItem, Any]:
    """WorkItem -> the legacy ``ComputeEvent`` that realizes it."""
    import program.schedule as legacy_schedule

    index: Dict[WorkItem, Any] = {}
    for b, node in enumerate(events.embedding):
        index[WorkItem(WorkKind.EMBEDDING, Direction.FORWARD, microbatch=b)] = node
    for b, node in enumerate(events.softmax):
        index[WorkItem(WorkKind.SOFTMAX, Direction.FORWARD, microbatch=b)] = node
        for child in node.children:
            if isinstance(child, legacy_schedule.ComputeEvent) and child.role == "softmax_b":
                index[WorkItem(WorkKind.SOFTMAX, Direction.BACKWARD, microbatch=b)] = child

    for obj in _walk_events(events):
        if not isinstance(obj, legacy_schedule.ComputeEvent):
            continue
        if obj.layer_index is None or obj.micro_batch_index is None:
            continue
        kind = WorkKind.RECOMPUTE if obj.recompute else WorkKind.LAYER
        direction = Direction.FORWARD if obj.direction == "forward" else Direction.BACKWARD
        index[
            WorkItem(kind, direction, microbatch=obj.micro_batch_index, layer=obj.layer_index)
        ] = obj
    return index


def _is_pipeline(obj: Any) -> bool:
    return getattr(obj, "comm_type", None) is PIPELINE


@dataclass(frozen=True)
class _AttachCall:
    """One ``add_child`` burst: the parents an object was hung under, followed
    by the successors it inherited. ``attach_parallel_edge`` emits exactly this
    shape (schedule.py:553-567), so segmenting the log recovers the calls."""

    obj: Any
    start: int
    parents: Tuple[Any, ...]
    children: Tuple[Any, ...]


def _attach_calls(log: List[Tuple[Any, Any]], tracked: Any) -> List[_AttachCall]:
    """Segment the ``add_child`` log into the ``attach_parallel_edge`` calls
    that hung each TRACKED object.

    ``attach_parallel_edge(target, e)`` emits ``(p, e)`` for every parent of
    ``target`` and then ``(e, c)`` for every child that survives the filter,
    contiguously (schedule.py:553-567). An entry therefore belongs to ``e``'s
    call whenever ``e`` is its CHILD; a leading children-only run belongs to
    ``e`` when ``target`` had no parents.
    """
    tracked_ids = {id(obj) for obj in tracked}
    calls: List[_AttachCall] = []
    idx = 0
    total = len(log)
    while idx < total:
        parent, child = log[idx]
        if id(child) in tracked_ids:
            hung = child
            parent_end = idx
            while parent_end < total and log[parent_end][1] is hung:
                parent_end += 1
            child_end = parent_end
            while child_end < total and log[child_end][0] is hung:
                child_end += 1
            calls.append(
                _AttachCall(
                    obj=hung,
                    start=idx,
                    parents=tuple(p for p, _ in log[idx:parent_end]),
                    children=tuple(c for _, c in log[parent_end:child_end]),
                )
            )
            idx = child_end
            continue
        if id(parent) in tracked_ids:
            hung = parent
            child_end = idx
            while (
                child_end < total
                and log[child_end][0] is hung
                and id(log[child_end][1]) not in tracked_ids
            ):
                child_end += 1
            calls.append(
                _AttachCall(
                    obj=hung,
                    start=idx,
                    parents=(),
                    children=tuple(c for _, c in log[idx:child_end]),
                )
            )
            idx = child_end
            continue
        idx += 1
    return calls


@pytest.mark.parametrize(
    "cfg",
    [
        Cfg(dp=2, zero_stage=3, pp=2, num_layers=4, mb=3),
        Cfg(dp=2, zero_stage=3, pp=4, num_layers=4, mb=3),
        Cfg(dp=2, zero_stage=3, pp=2, num_layers=5, mb=4, moe=True, ep=2),
        Cfg(dp=2, zero_stage=3, pp=3, num_layers=2, mb=2),
        Cfg(dp=2, zero_stage=3, pp=1, num_layers=4, mb=3),
        Cfg(dp=2, zero_stage=3, pp=2, num_layers=4, mb=1),
        Cfg(
            dp=2,
            zero_stage=3,
            pp=2,
            num_layers=4,
            mb=3,
            full_recomputation=True,
            flattened=True,
        ),
    ],
    ids=lambda c: c.label(),
)
def test_via_reproduces_the_legacy_skip_flags(cfg: Cfg, monkeypatch: Any) -> None:
    """Row by row: the typed ``via`` selects exactly the successor edges the
    legacy boolean flags selected.

    ``skip_non_comm_children=True`` keeps only ``CollectiveType.PIPELINE``
    successors -> ``VIA_DATA_FLOW``; ``skip_comm_children=True`` keeps only the
    others -> ``VIA_NON_DATA_FLOW``; no flag -> ``VIA_ALL``, and the gather must
    then inherit the host's successors UNFILTERED (compared against a replay of
    the host's children as of that call).
    """
    events, log = _legacy_events_with_log(cfg, monkeypatch)
    compute = _legacy_compute_index(events, cfg)

    legacy_by_key: Dict[SyncKey, Any] = {}
    for obj in _walk_events(events):
        key = _sync_key_from_legacy_name(str(getattr(obj, "name", "")))
        if key is not None:
            legacy_by_key[key] = obj
    calls = _attach_calls(log, tracked=tuple(legacy_by_key.values()))

    reqs = {r.key: r for r in policy_requirements(cfg)}
    parallel = [
        r for r in reqs.values() if r.mode is AttachMode.PARALLEL_TO and r.key in legacy_by_key
    ]
    assert parallel, "no PARALLEL_TO requirement was exercised"

    checked_all = checked_data = checked_non_data = 0
    for req in parallel:
        gather = legacy_by_key[req.key]
        own = [call for call in calls if call.obj is gather]
        assert own, f"{req.key!r}: legacy gather was never attached"
        assert len(own) == len(req.anchors), (
            f"{req.key!r}: legacy attached it {len(own)} time(s) but the policy "
            f"declares {len(req.anchors)} anchor(s)"
        )

        for call in own:
            if req.via == VIA_DATA_FLOW:
                assert all(_is_pipeline(c) for c in call.children), (
                    f"{req.key!r} declares VIA_DATA_FLOW but legacy kept non-pipeline "
                    f"successors {[getattr(c, 'name', c) for c in call.children]}"
                )
                checked_data += 1
            elif req.via == VIA_NON_DATA_FLOW:
                assert not any(_is_pipeline(c) for c in call.children), (
                    f"{req.key!r} declares VIA_NON_DATA_FLOW but legacy kept pipeline "
                    f"successors {[getattr(c, 'name', c) for c in call.children]}"
                )
                checked_non_data += 1
            else:
                assert req.via == VIA_ALL
                if len(req.anchors) != 1:
                    continue
                host = compute.get(req.anchors[0])
                if host is None:
                    continue
                snapshot = tuple(
                    child for (parent, child) in log[: call.start] if parent is host
                )
                assert call.children == snapshot, (
                    f"{req.key!r} declares VIA_ALL but legacy filtered the host's "
                    f"successors: took {[getattr(c, 'name', c) for c in call.children]} "
                    f"of {[getattr(c, 'name', c) for c in snapshot]}"
                )
                checked_all += 1

    assert checked_all > 0
    if cfg.pp > 1 and cfg.num_layers >= cfg.pp:
        # a multi-stage config always exercises at least one cross-device site
        assert checked_data + checked_non_data > 0


def test_ep_sync_requires_the_graph_ep_degree() -> None:
    """schedule.py:936 gates the EP sync on ``spec.ep > 1`` (the GRAPH ep:
    ``time_calc.ep`` only when ``use_moe``). Upstream never registers the keys
    at ``ep == 1``, so the degree test is belt-and-braces — pinned here so it
    stays intentional rather than becoming dead code nobody dares delete."""
    cfg = Cfg(dp=2, moe=True, ep=2)
    fw = make_workload(cfg).freeze()
    work = enumerate_work(fw, NoRecompute())
    layer_bwd = work.require(WorkKind.LAYER, Direction.BACKWARD, microbatch=0, layer=0)

    ctx = ShardingContext(
        fw=fw,
        work=work,
        grad_accum=grad_accum_policy_for(fw),
        stages=ContiguousStages.legacy(cfg.num_layers, cfg.pp),
    )
    assert ep_sync_requirements(layer_bwd, ctx)

    # same comm table, graph ep degree of 1 -> no EP sync
    degraded = fw.spec.with_(
        degrees=ParallelDegrees(tp=1, cp=1, ep=1, pp=cfg.pp, dp=cfg.dp)
    ).freeze()
    ctx_ep1 = dataclasses.replace(ctx, fw=degraded)
    assert ep_sync_requirements(layer_bwd, ctx_ep1) == ()

    # forward work never induces an EP grad sync
    layer_fwd = work.require(WorkKind.LAYER, Direction.FORWARD, microbatch=0, layer=0)
    assert ep_sync_requirements(layer_fwd, ctx) == ()
