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

"""M5 coarse-builder / analytical-evaluator differential tests.

Two layers, both asserting EXACT float equality between the legacy
analytical pipeline (``convert_comm_sizes_to_times`` + ``Graph.simulate``
over the legacy Node/Edge graph) and the M5 replacement
(``build_coarse_program`` + ``analytic_sim.evaluate``), including the full
per-event finish-time maps and the converted comm durations:

* always-on synthetic differentials: hand-built pipeline configurations
  covering GPipe multi-stage schedules, DP reducers (float byte sizes),
  ZeRO-2/3 gather lattices (incl. the ZeRO-3 entry-edge root), EP sync,
  MoE layer masks, recompute nodes, grad-accum nonfinal cycles, forward
  -only (inference) graphs, and the hybrid per-DP retiming write-back
  (legacy ``_assign_transformer_durations`` reference vs
  ``program.retime.apply_block_timings``);
* an env-gated matrix differential (``RAPID_COARSE_DIFF=1``): for every
  analytical + hybrid golden spec (train + inference + ga2 final/no-dp +
  zero2/zero3 + moe hybrid) the real config pipeline graphs are compared
  end-to-end; hybrid specs additionally compare both write-back paths under
  synthetic AstraSim block timings (per-(dp, stage) fault overrides
  included) — no AstraSim binary involved.

The legacy side prefers the LIVE legacy methods while they exist
(pre-cutover evidence); after the M5 deletion it runs the frozen verbatim
reference copies below over the legacy graph, which
``construct_fwd_bwd_graph`` keeps building until M6. The reference copies
die with it.

Run the matrix differential:
    RAPID_COARSE_DIFF=1 ./.venv/bin/python -m pytest tests/test_coarse_builder_diff.py -q
"""

from __future__ import annotations

import copy
import os
from heapq import heappop, heappush
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

import simulate_train_graph as llm_simulation
from timing_model import CollectiveType
from equiv.configs import MATRIX

from program.analytic_sim import evaluate_detailed
from program.pipeline_coarse import build_coarse_program
from program.retime import BlockTimings, apply_block_timings
from program.schedule import CommEvent, ComputeEvent, ScheduleSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_CONFIG = (
    REPO_ROOT / "validation_scripts" / "validation_configs" / "model-config" / "Llama2-7B.yaml"
)
BASE_HW_CONFIG = (
    REPO_ROOT / "validation_scripts" / "validation_configs" / "hardware-config" / "a100_80GB.yaml"
)

COARSE_SPECS = [spec for spec in MATRIX if spec.backend in ("analytical", "hybrid")]

_diff_gate = pytest.mark.skipif(
    os.environ.get("RAPID_COARSE_DIFF", "") != "1",
    reason="coarse differential sweep is a dev gate; set RAPID_COARSE_DIFF=1 to run",
)


# ---------------------------------------------------------------------------
# Frozen legacy reference (verbatim semantics of the deleted
# Graph.convert_comm_sizes_to_times / Graph.simulate; dies at M6 with
# construct_fwd_bwd_graph).
# ---------------------------------------------------------------------------


def _reference_convert_comm_sizes_to_times(roots, network_model, interconnect_params):
    def traverse_and_convert(node, visited=None):
        if visited is None:
            visited = set()
        if id(node) in visited:
            return
        visited.add(id(node))

        for child in node.children:
            if hasattr(child, "comm_size_bytes") and child.comm_size_bytes > 0:
                interconnect_type = child.comm_interconnect_type
                if interconnect_type and interconnect_type in interconnect_params:
                    ib, ll = interconnect_params[interconnect_type]
                else:
                    raise ValueError(f"Invalid interconnect type: {interconnect_type}")
                if not isinstance(child.comm_type, CollectiveType):
                    raise TypeError(
                        f"Comm edge {getattr(child, 'name', '<unnamed>')} missing "
                        "CollectiveType comm_type"
                    )
                child.duration = network_model.collective(
                    kind=child.comm_type,
                    size_bytes=child.comm_size_bytes,
                    participants=child.participants,
                    ib=ib,
                    ll=ll,
                    local_bytes=0.0,
                    local_ops=0.0,
                    debug_label=f"{child.name}_conversion",
                )
            traverse_and_convert(child, visited)

    traverse_and_convert(roots)
    return roots


def _reference_reset_execution_state(root: Any) -> None:
    if root is None:
        return
    visited: Set[int] = set()
    stack: List[Any] = list(root) if isinstance(root, (list, tuple, set)) else [root]
    while stack:
        obj = stack.pop()
        obj_id = id(obj)
        if obj_id in visited:
            continue
        visited.add(obj_id)
        if hasattr(obj, "done"):
            obj.done = False
        if hasattr(obj, "scheduled"):
            obj.scheduled = False
        if hasattr(obj, "finish_time"):
            obj.finish_time = -1
        stack.extend(getattr(obj, "children", []))


def _reference_simulate(pp: int, root: Any) -> float:
    """Verbatim ``Graph.simulate`` port (Data_batch events are never present:
    the legacy constructor removed them before returning the root)."""
    time = 0
    counter = 0
    event_queue: List[Tuple[float, int, Any]] = []
    ready_list: List[Any] = []

    _reference_reset_execution_state(root)

    ready_list.append(root)
    root.scheduled = True
    base_devices = max(1, int(pp) if pp else 1)
    max_hw_id = -1
    visited_nodes: Set[int] = set()
    stack = list(root if isinstance(root, (list, tuple)) else [root])
    while stack:
        node = stack.pop()
        node_id = id(node)
        if node_id in visited_nodes:
            continue
        visited_nodes.add(node_id)
        hw_id = getattr(node, "hw_id", None)
        if hw_id is not None:
            try:
                hw_val = int(hw_id)
            except (TypeError, ValueError):
                hw_val = None
            if hw_val is not None and hw_val >= 0:
                max_hw_id = max(max_hw_id, hw_val)
        stack.extend(getattr(node, "children", []))
    if max_hw_id >= 0:
        base_devices = max(base_devices, max_hw_id + 1)

    GPU_list = [True for _ in range(base_devices)]

    heappush(event_queue, (root.duration, counter, root))
    ready_list.remove(root)
    counter = counter + 1

    while len(event_queue) > 0:
        time, _, event = heappop(event_queue)
        event.done = True
        event.scheduled = False
        event.finish_time = time

        for child in event.children:
            is_ready = True
            for parent in child.parents:
                if parent.done == False:  # noqa: E712 - legacy comparison kept
                    is_ready = False
            if is_ready and (child not in ready_list) and (not child.done) and (not child.scheduled):
                ready_list.append(child)

        if isinstance(event, llm_simulation.Node):
            GPU_list[int(event.hw_id)] = True

        for event in ready_list[:]:
            if isinstance(event, llm_simulation.Node):
                if GPU_list[int(event.hw_id)] == True:  # noqa: E712
                    new_time = time + event.duration
                    heappush(event_queue, (new_time, counter, event))
                    event.scheduled = True
                    counter = counter + 1
                    GPU_list[int(event.hw_id)] = False
                    ready_list.remove(event)
            elif isinstance(event, llm_simulation.Edge):
                new_time = time + event.duration
                heappush(event_queue, (new_time, counter, event))
                event.scheduled = True
                counter = counter + 1
                ready_list.remove(event)

    return time


def _reference_assign_transformer_durations(
    node: Any,
    visited: Set[int],
    stage_timings: Dict[Tuple[int, int], Any],
    stage_moe_timings: Dict[Tuple[int, int], Any],
    dense_timings: Optional[Any],
    moe_timings: Optional[Any],
    dp_count: int,
) -> None:
    """Verbatim ``LLMExecutionDispatcher._assign_transformer_durations``
    (the legacy name-prefix hybrid write-back), kept as the reference for
    the retiming differential."""
    if node is None:
        return
    node_id = id(node)
    if node_id in visited:
        return
    visited.add(node_id)

    if isinstance(node, llm_simulation.Node):
        base_name = str(getattr(node, "name", "") or "")
        if base_name.startswith("transformer_layer") or base_name.startswith("vit_block"):
            is_moe_layer = bool(getattr(node, "is_moe_layer", False))
            timing_source = moe_timings if is_moe_layer and moe_timings is not None else dense_timings
            if timing_source is None:
                timing_source = dense_timings
            if timing_source is not None:
                try:
                    hw_stage = int(getattr(node, "hw_id", None))
                except (TypeError, ValueError):
                    hw_stage = None

                values: List[float] = []
                is_forward = bool(getattr(node, "fwd", True))
                for dp_idx in range(dp_count):
                    default = timing_source.forward if is_forward else timing_source.backward
                    timing_override = None
                    if hw_stage is not None:
                        if is_moe_layer:
                            timing_override = stage_moe_timings.get((dp_idx, hw_stage))
                        else:
                            timing_override = stage_timings.get((dp_idx, hw_stage))
                    if timing_override:
                        values.append(timing_override.forward if is_forward else timing_override.backward)
                    else:
                        values.append(default)

                if dp_count > 1:
                    node.duration = tuple(values)
                else:
                    node.duration = values[0]

    for child in getattr(node, "children", []):
        _reference_assign_transformer_durations(
            child,
            visited,
            stage_timings,
            stage_moe_timings,
            dense_timings,
            moe_timings,
            dp_count,
        )


def _legacy_total(graph: Any, root: Any, network_model: Any, interconnect_params) -> float:
    """Legacy analytical total: LIVE legacy methods while they exist
    (pre-cutover evidence), frozen reference afterwards."""
    if hasattr(graph, "convert_comm_sizes_to_times") and hasattr(graph, "simulate"):
        graph.convert_comm_sizes_to_times(root, network_model, interconnect_params)
        return graph.simulate(root)
    _reference_convert_comm_sizes_to_times(root, network_model, interconnect_params)
    return _reference_simulate(int(getattr(graph, "pp", 1) or 1), root)


# ---------------------------------------------------------------------------
# Structural pairing (legacy graph <-> coarse events) + finish-map compare
# ---------------------------------------------------------------------------


def _assert_finish_maps_equal(legacy_root: Any, program: Any, finish_times: List[float]) -> int:
    """Pairwise DFS over the isomorphic graphs asserting names, kinds,
    children arity, converted comm durations, and finish times all match.
    Returns the number of paired events."""
    events = program.meta.misc["coarse_events"]
    events_root = program.meta.misc["coarse_proto_root"]
    uid_of = {id(event): uid for uid, event in enumerate(events)}

    paired = 0
    seen: Set[int] = set()
    stack: List[Tuple[Any, Any]] = [(legacy_root, events_root)]
    while stack:
        legacy_obj, event_obj = stack.pop()
        if id(legacy_obj) in seen:
            continue
        seen.add(id(legacy_obj))
        paired += 1

        assert getattr(legacy_obj, "name") == getattr(event_obj, "name")
        if isinstance(legacy_obj, llm_simulation.Node):
            assert isinstance(event_obj, ComputeEvent)
        else:
            assert isinstance(legacy_obj, llm_simulation.Edge)
            assert isinstance(event_obj, CommEvent)
            # Converted comm durations must agree exactly.
            assert legacy_obj.duration == event_obj.duration, (
                f"comm duration mismatch on '{legacy_obj.name}': "
                f"{legacy_obj.duration} != {event_obj.duration}"
            )

        legacy_finish = getattr(legacy_obj, "finish_time", -1)
        new_finish = finish_times[uid_of[id(event_obj)]]
        assert legacy_finish == new_finish, (
            f"finish-time mismatch on '{legacy_obj.name}': "
            f"legacy={legacy_finish} new={new_finish}"
        )

        legacy_children = list(legacy_obj.children)
        event_children = list(event_obj.children)
        assert len(legacy_children) == len(event_children), (
            f"children arity mismatch on '{legacy_obj.name}'"
        )
        stack.extend(zip(legacy_children, event_children))
    return paired


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


class FakeNetworkModel:
    """Deterministic pure stand-in for NetworkModel.collective."""

    _FACTORS = {
        CollectiveType.ALL_REDUCE: 1.0,
        CollectiveType.REDUCE_SCATTER: 0.75,
        CollectiveType.ALL_GATHER: 0.5,
        CollectiveType.ALL_TO_ALL: 1.25,
        CollectiveType.PIPELINE: 0.25,
    }

    def collective(
        self,
        *,
        kind,
        size_bytes,
        participants,
        ib,
        ll,
        local_bytes=0.0,
        local_ops=0.0,
        debug_label="",
        axis=None,
    ):
        factor = self._FACTORS.get(kind, 2.0)
        return ll + float(size_bytes) / float(ib) * factor + float(participants) * 1e-7


FAKE_INTERCONNECT = {
    "dp": (4.0e10, 1.0e-6),
    "pp": (2.0e10, 2.0e-6),
    "tp": (8.0e10, 0.5e-6),
    "cp": (6.0e10, 0.75e-6),
    "ep": (3.0e10, 1.5e-6),
}


def _synthetic_comm_metadata(dp: int, ep: int, zero_stage: int, with_ep_sync: bool) -> Dict[str, Dict[str, Any]]:
    grad_collective = (
        CollectiveType.REDUCE_SCATTER if (zero_stage >= 2 and dp > 1) else CollectiveType.ALL_REDUCE
    )
    metadata: Dict[str, Dict[str, Any]] = {
        "transformer_dense": {
            "size": 12345678.5,  # RAW float (legacy dp reducer sizes are floats)
            "type": grad_collective,
            "participants": dp,
            "interconnect_type": "dp",
            "local_comp_time": 0,
            "ga_required_every_cycle": False,
        },
        "transformer_moe": {
            "size": 23456789.25,
            "type": grad_collective,
            "participants": dp,
            "interconnect_type": "dp",
            "local_comp_time": 0,
            "ga_required_every_cycle": False,
        },
        "embedding": {
            "size": 524288,
            "type": grad_collective,
            "participants": dp,
            "interconnect_type": "dp",
            "local_comp_time": 0,
            "ga_required_every_cycle": False,
        },
        "softmax": {
            "size": 262144,
            "type": grad_collective,
            "participants": dp,
            "interconnect_type": "dp",
            "local_comp_time": 0,
            "ga_required_every_cycle": False,
        },
        "cross_layer": {
            "size": 1048576,
            "type": CollectiveType.PIPELINE,
            "participants": 2,
            "interconnect_type": "pp",
            "local_comp_time": 0,
        },
    }
    if zero_stage >= 2:
        for key, size in (
            ("zero2_embedding_gather", 111111.0),
            ("zero2_transformer_gather", 222222.0),
            ("zero2_softmax_gather", 333333.0),
        ):
            metadata[key] = {
                "size": size,
                "type": CollectiveType.ALL_GATHER,
                "participants": dp,
                "interconnect_type": "dp",
                "local_comp_time": 0,
                "ga_required_every_cycle": False,
            }
    if zero_stage >= 3:
        for key, size in (
            ("zero3_embedding_gather", 444444.0),
            ("zero3_transformer_gather", 555555.0),
            ("zero3_softmax_gather", 666666.0),
        ):
            metadata[key] = {
                "size": size,
                "type": CollectiveType.ALL_GATHER,
                "participants": dp,
                "interconnect_type": "dp",
                "local_comp_time": 0,
                "ga_required_every_cycle": True,
            }
        metadata["zero3_transformer_gather"]["tp_shard"] = True
    if with_ep_sync and ep > 1:
        metadata["transformer_dense_ep_sync"] = {
            "size": 777777,
            "type": grad_collective,
            "participants": ep,
            "interconnect_type": "ep",
            "local_comp_time": 0,
        }
        metadata["transformer_moe_ep_sync"] = {
            "size": 888888,
            "type": grad_collective,
            "participants": ep,
            "interconnect_type": "ep",
            "local_comp_time": 0,
        }
    return metadata


def _synthetic_case(
    *,
    dp: int = 1,
    pp: int = 1,
    tp: int = 1,
    cp: int = 1,
    ep: int = 1,
    mb: int = 1,
    layers: int = 2,
    zero_stage: int = 0,
    moe_layer_mask: Optional[List[bool]] = None,
    full_recomputation: bool = False,
    grad_accum_cycle: str = "final",
    dp_microbatch_mode: str = "every_mb",
    include_backward: bool = True,
    include_optimizer: bool = True,
    with_ep_sync: bool = False,
    model_type: str = "gpt",
):
    """Build (legacy graph, legacy root, ScheduleSpec) from one input set."""
    comp_times = {
        "linear_softmax_f": 0.002,
        "linear_softmax_b": 0.0025,
        "transformer_f": 0.010,
        "transformer_b": 0.021,
        "transformer_f_dense": 0.010,
        "transformer_b_dense": 0.021,
        "transformer_f_moe": 0.017,
        "transformer_b_moe": 0.033,
        "embedding_f": 0.001,
        "embedding_b": 0.0015,
        "optimizer": 0.004,
        "cross_layer_f": 0.0,
        "cross_layer_b": 0.0,
    }
    misc_metadata = {
        "num_batch": mb,
        "num_layer": layers,
        "dp_zero_stage": zero_stage,
        "full_recomputation": full_recomputation,
        "flattened_mode": False,
        "pipeline_style_recompute": full_recomputation,
        "dp_microbatch_mode": dp_microbatch_mode,
        "grad_accum_cycle": grad_accum_cycle,
        "moe_layer_mask": list(moe_layer_mask or []),
        "model_type": model_type,
    }
    comm_metadata = _synthetic_comm_metadata(dp, ep, zero_stage, with_ep_sync)

    graph = llm_simulation.Graph(
        mode="pipeline",
        dp=dp,
        pp=pp,
        tp=tp,
        cp=cp,
        ep=ep,
        comp_times=dict(comp_times),
        comm_metadata=comm_metadata,
        misc_metadata=misc_metadata,
    )
    root = graph.construct_fwd_bwd_graph(
        include_backward=include_backward,
        include_optimizer=include_optimizer,
    )
    spec = ScheduleSpec.from_pipeline_graph(
        graph,
        include_backward=include_backward,
        include_optimizer=include_optimizer,
    )
    return graph, root, spec


SYNTHETIC_CASES = {
    "single_stage": dict(dp=1, pp=1, mb=1, layers=2),
    "gpipe_dp": dict(dp=2, pp=2, mb=3, layers=4),
    "gpipe_dp_last_mb": dict(dp=2, pp=2, mb=2, layers=4, dp_microbatch_mode="last_mb"),
    "zero2": dict(dp=2, pp=2, mb=2, layers=4, zero_stage=2),
    "zero3": dict(dp=2, pp=2, mb=2, layers=4, zero_stage=3),
    "moe_ep_sync": dict(
        dp=2, pp=2, mb=2, layers=4, ep=2, with_ep_sync=True,
        moe_layer_mask=[False, True, False, True],
    ),
    "recompute": dict(dp=2, pp=2, mb=2, layers=4, full_recomputation=True),
    "nonfinal_ga": dict(
        dp=2, pp=2, mb=2, layers=4, zero_stage=3,
        grad_accum_cycle="nonfinal", include_optimizer=False,
    ),
    "forward_only": dict(dp=1, pp=2, mb=2, layers=4, include_backward=False),
    "uneven_layers": dict(dp=1, pp=2, mb=2, layers=3),
}


class _Timing:
    """TransformerTimings stand-in (forward/backward attrs)."""

    def __init__(self, forward: float, backward: float) -> None:
        self.forward = forward
        self.backward = backward


# ---------------------------------------------------------------------------
# Always-on synthetic differentials
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_name", sorted(SYNTHETIC_CASES))
def test_synthetic_analytical_parity(case_name):
    graph, root, spec = _synthetic_case(**SYNTHETIC_CASES[case_name])
    program = build_coarse_program(spec, None)

    network_model = FakeNetworkModel()
    legacy_total = _legacy_total(graph, root, network_model, FAKE_INTERCONNECT)
    result = evaluate_detailed(program, network_model, FAKE_INTERCONNECT)

    assert result.total_time == legacy_total
    paired = _assert_finish_maps_equal(root, program, result.finish_times)
    assert paired == len(program.ops)


@pytest.mark.parametrize("case_name", ["gpipe_dp", "moe_ep_sync", "recompute"])
def test_synthetic_hybrid_retime_parity(case_name):
    cfg = SYNTHETIC_CASES[case_name]
    graph, root, spec = _synthetic_case(**cfg)
    program = build_coarse_program(spec, None)

    dp_count = int(cfg.get("dp", 1))
    dense = _Timing(0.00125, 0.0025)
    moe = _Timing(0.0035, 0.00475) if cfg.get("moe_layer_mask") else None
    stage_dense = {(0, 0): _Timing(0.0015, 0.00275), (1, 1): _Timing(0.00175, 0.003)}
    stage_moe = {(0, 1): _Timing(0.004, 0.005)} if moe else {}

    # Legacy write-back (reference copy of _assign_transformer_durations).
    _reference_assign_transformer_durations(
        root, set(), stage_dense, stage_moe, dense, moe, dp_count
    )
    # New write-back (metadata-selected, per-DP tuples).
    retimed = apply_block_timings(
        program,
        BlockTimings(dense=dense, moe=moe, stage_dense=stage_dense, stage_moe=stage_moe),
        dp_count,
    )
    assert retimed > 0

    network_model = FakeNetworkModel()
    legacy_total = _legacy_total(graph, root, network_model, FAKE_INTERCONNECT)
    result = evaluate_detailed(program, network_model, FAKE_INTERCONNECT)

    assert result.total_time == legacy_total
    _assert_finish_maps_equal(root, program, result.finish_times)


def test_coarse_program_shape():
    """Typed-surface checks: roles, metadata, transfers, zero-byte events."""
    from program.ir import CollectiveOp, ComputeOp, Direction, OpRole, TransferOp

    _, _, spec = _synthetic_case(dp=2, pp=2, mb=2, layers=4, zero_stage=3)
    program = build_coarse_program(spec, None, dp_count=2)

    assert program.meta.misc["granularity"] == "coarse"
    assert program.dp_count == 2
    assert program.devices == (0, 1)
    assert len(program.meta.misc["coarse_events"]) == len(program.ops)

    layer_ops = [
        op
        for op in program.ops
        if isinstance(op, ComputeOp) and op.role is OpRole.TRANSFORMER_LAYER
    ]
    # 2 micro-batches x 4 layers x fwd+bwd (no recompute in this case).
    assert len(layer_ops) == 16
    forward_layers = [op for op in layer_ops if op.direction is Direction.FORWARD]
    assert {(op.micro_batch, op.layer) for op in forward_layers} == {
        (b, l) for b in range(2) for l in range(4)
    }
    # Layer placement follows the legacy remainder-first split (2 per stage).
    for op in layer_ops:
        assert op.device == (0 if op.layer < 2 else 1)
        assert op.is_moe_layer is False

    transfers = [op for op in program.ops if isinstance(op, TransferOp)]
    same_stage = [op for op in transfers if op.src_device == op.dst_device]
    cross_stage = [op for op in transfers if op.src_device != op.dst_device]
    assert same_stage and cross_stage
    # Same-placement zero-byte events are preserved as ops (DESIGN §2.2).
    assert all(op.size_bytes == 0 for op in same_stage)
    assert all(op.size_bytes > 0 for op in cross_stage)

    collectives = [op for op in program.ops if isinstance(op, CollectiveOp)]
    assert collectives
    assert all(op.coll is not CollectiveType.PIPELINE for op in collectives)
    # ZeRO-3 lattice is present (dp-flagged, all-gather).
    assert any(op.coll is CollectiveType.ALL_GATHER and op.is_dp for op in collectives)


def test_evaluator_reads_duration_index_zero():
    """Legacy Node.duration property semantics: profiles evaluate at [0]."""
    _, _, spec = _synthetic_case(dp=2, pp=2, mb=2, layers=4)
    program_scalar = build_coarse_program(spec, None)
    program_profile = build_coarse_program(spec, None)

    dense = _Timing(0.005, 0.009)
    # Profile whose dp0 entries equal the scalar retime, dp1 entries differ:
    # the evaluator must ignore every index but 0.
    apply_block_timings(program_scalar, BlockTimings(dense=dense), dp_count=1)
    apply_block_timings(
        program_profile,
        BlockTimings(dense=dense, stage_dense={(1, 0): _Timing(1.0, 2.0), (1, 1): _Timing(3.0, 4.0)}),
        dp_count=2,
    )

    network_model = FakeNetworkModel()
    total_scalar = evaluate_detailed(program_scalar, network_model, FAKE_INTERCONNECT).total_time
    total_profile = evaluate_detailed(program_profile, network_model, FAKE_INTERCONNECT).total_time
    assert total_scalar == total_profile


# ---------------------------------------------------------------------------
# Env-gated matrix differential (real configs, no AstraSim)
# ---------------------------------------------------------------------------


def _sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)


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


def _graph_cases(spec, hw_config, model_config, mode, out_dir: Path):
    """Drive the time-calculation objects far enough to obtain the pipeline
    graphs + dispatcher(s); returns [(label, tc, dispatcher)]."""
    from llm_execution import LLMExecutionDispatcher

    cases = []
    if spec.run_type == "training":
        from train_timing import TimeCalculationLLM

        tc = TimeCalculationLLM(hw_config, model_config, mode, output_dir=str(out_dir))
        tc._build_training_graphs_and_memory_data()

        dispatcher = LLMExecutionDispatcher(
            time_calc=tc,
            pipeline_graph=tc.pipeline_graph,
            pipeline_root=tc.pipeline_root,
            interconnect_params=tc.pipeline_interconnect,
            transformer_blocks=tc.transformer_blocks,
            no_data_parallel=False,
        )
        cases.append(("final", tc, dispatcher))

        if getattr(tc, "gradient_accumulation_steps", 1) > 1:
            dispatcher_no_dp = LLMExecutionDispatcher(
                time_calc=tc,
                pipeline_graph=tc.pipeline_graph_no_dp,
                pipeline_root=tc.pipeline_root_no_dp,
                interconnect_params=tc.pipeline_interconnect,
                transformer_blocks=tc.transformer_blocks_no_dp or tc.transformer_blocks,
                no_data_parallel=True,
            )
            cases.append(("no_dp", tc, dispatcher_no_dp))
    else:
        from inference_timing import TimeCalculationLLMInference

        tc = TimeCalculationLLMInference(hw_config, model_config, mode, output_dir=str(out_dir))
        batch_size = tc._effective_transformer_batch()
        decode_len = tc.model.decode_len
        prefill_len = tc.seq_len - decode_len
        assert prefill_len > 0, "coarse differential requires a prefill phase"
        num_SMs = tc.hw_config.tech_config.core.num_bundles
        transformer_timings, node_breakdown = tc.compute_all_gemm_and_node_times(
            batch_size,
            tc.vocab_size,
            tc.hidden_dim,
            prefill_len,
            tc.num_heads,
            tc.kv_heads,
            tc.intermediate_size,
            num_SMs,
            use_moe_override=False,
        )
        (
            pipeline_graph,
            pipeline_root,
            _,
            _,
            transformer_blocks,
            interconnect_params,
        ) = tc._prepare_execution_graphs(
            node_breakdown=node_breakdown,
            transformer_timings=transformer_timings,
            batch_size=batch_size,
            seq_len=prefill_len,
            hidden_dim=tc.hidden_dim,
            intermediate_size=tc.intermediate_size,
            vocab_size=tc.vocab_size,
            include_pipeline_backward=False,
            include_transformer_backward=False,
        )
        dispatcher = LLMExecutionDispatcher(
            time_calc=tc,
            pipeline_graph=pipeline_graph,
            pipeline_root=pipeline_root,
            interconnect_params=interconnect_params,
            transformer_blocks=transformer_blocks,
        )
        cases.append(("prefill", tc, dispatcher))
    return cases


@_diff_gate
@pytest.mark.parametrize("spec", COARSE_SPECS, ids=lambda s: s.spec_id)
def test_matrix_coarse_parity(spec, tmp_path):
    from llm_execution import TransformerTimings

    hw_config, model_config, mode = _parse_spec_configs(spec, tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cases = _graph_cases(spec, hw_config, model_config, mode, out_dir)
    assert cases

    for label, tc, dispatcher in cases:
        graph = dispatcher.pipeline_graph
        root = dispatcher.pipeline_root
        network_model = tc.network_model
        interconnect_params = dispatcher.interconnect_params

        # ---- analytical arm (pristine comp_times on both sides) ----------
        program = dispatcher._build_coarse_program()
        # Hybrid arm program must ALSO see pristine comp_times: build now.
        program_hybrid = (
            dispatcher._build_coarse_program() if spec.backend == "hybrid" else None
        )

        legacy_total = _legacy_total(graph, root, network_model, interconnect_params)
        result = evaluate_detailed(program, network_model, interconnect_params)
        assert result.total_time == legacy_total, (
            f"{spec.spec_id}[{label}]: analytical totals diverge: "
            f"legacy={legacy_total!r} new={result.total_time!r}"
        )
        paired = _assert_finish_maps_equal(root, program, result.finish_times)
        assert paired == len(program.ops)

        # ---- hybrid arm: identical synthetic block timings through both
        # write-back paths (no AstraSim involved) --------------------------
        if spec.backend == "hybrid":
            dp = max(1, getattr(tc, "dp", 1))
            pp = max(1, getattr(tc, "pp", 1))
            dense = TransformerTimings(forward=0.00125, backward=0.0025)
            moe = (
                TransformerTimings(forward=0.0035, backward=0.00475)
                if spec.use_moe
                else None
            )
            stage_dense = {(0, 0): TransformerTimings(forward=0.0015, backward=0.00275)}
            if dp > 1 and pp > 1:
                stage_dense[(1, 1)] = TransformerTimings(forward=0.00175, backward=0.003)
            stage_moe = (
                {(0, min(1, pp - 1)): TransformerTimings(forward=0.004, backward=0.005)}
                if moe
                else {}
            )

            # Legacy write-back onto the legacy graph.
            dispatcher._transformer_stage_timings = dict(stage_dense)
            dispatcher._transformer_stage_moe_timings = dict(stage_moe)
            dispatcher._apply_transformer_time(dense, moe)

            # New write-back onto the coarse program.
            retimed = apply_block_timings(
                program_hybrid,
                dispatcher._collect_block_timings(dense, moe),
                dispatcher._retime_dp_count(),
            )
            assert retimed > 0

            legacy_total_h = _legacy_total(graph, root, network_model, interconnect_params)
            result_h = evaluate_detailed(program_hybrid, network_model, interconnect_params)
            assert result_h.total_time == legacy_total_h, (
                f"{spec.spec_id}[{label}]: hybrid totals diverge: "
                f"legacy={legacy_total_h!r} new={result_h.total_time!r}"
            )
            _assert_finish_maps_equal(root, program_hybrid, result_h.finish_times)
