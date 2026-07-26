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

"""M5/M6 coarse-builder tests.

The M5 differential (legacy ``convert_comm_sizes_to_times`` +
``Graph.simulate`` vs ``build_coarse_program`` + ``analytic_sim.evaluate``,
EXACT totals + full finish-time maps + converted comm durations; the
hybrid ``_assign_transformer_durations`` write-back vs
``program.retime.apply_block_timings``) ran green pre-cutover on all 10
synthetic cases and the analytical+hybrid matrix specs (16 specs as of
M8; differential cases counted arms — ga2 final+no_dp, synthetic
per-(dp, stage) retime overrides included), then
again post-cutover against frozen verbatim reference copies of the deleted
legacy methods. Per that plan, the reference copies died at M6 together
with the legacy hierarchical retime write-back they mirrored
(``_apply_transformer_time``/``_assign_transformer_durations``). The
always-on structural differential against the LIVE legacy constructor
(add_child-order isomorphism to the ``construct_fwd_bwd_graph`` Node/Edge
graph, op_id sequence included) ran green through M6 and died at M7 with
``construct_fwd_bwd_graph`` itself.

What remains:

* always-on synthetic tests pinning the coarse Program's typed surface,
  the evaluator's legacy duration-index-0 semantics, and the schedule
  events' determinism + ``op_id`` integrity (unique creation-order stamps
  aligned with the op list — the hierarchical emission lowering keys its
  per-stage Kahn toposort and Step-11 transfer replay on them);
* an env-gated determinism sweep (``RAPID_COARSE_DIFF=1``): every
  analytical + hybrid golden spec builds and evaluates the coarse program
  twice (hybrid: retimed under synthetic block timings) — totals AND full
  finish-time maps must be identical across builds.

Run the matrix sweep:
    RAPID_COARSE_DIFF=1 ./.venv/bin/python -m pytest tests/test_coarse_builder_diff.py -q
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

from timing_model import CollectiveType
from equiv.configs import MATRIX

from program.analytic_sim import evaluate_detailed
from program.pipeline_coarse import build_coarse_program
from program.retime import BlockTimings, apply_block_timings
from program.schedule import CommEvent, ComputeEvent, ScheduleInputs, ScheduleSpec

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
    reason="coarse determinism sweep is a dev gate; set RAPID_COARSE_DIFF=1 to run",
)


# ---------------------------------------------------------------------------
# Event-stream description (determinism + op_id integrity)
# ---------------------------------------------------------------------------


def _describe_event_stream(program: Any) -> List[Tuple[str, str, Optional[int], int]]:
    """DFS-preorder description of the coarse schedule events: (kind, name,
    op_id, children arity). Two builds from the same ScheduleSpec must
    produce identical streams — the emission lowering keys its per-stage
    Kahn toposort and Step-11 transfer replay on the op_ids."""
    events_root = program.meta.misc["coarse_proto_root"]

    out: List[Tuple[str, str, Optional[int], int]] = []
    seen: Set[int] = set()
    stack: List[Any] = [events_root]
    while stack:
        event_obj = stack.pop()
        if id(event_obj) in seen:
            continue
        seen.add(id(event_obj))
        kind = "compute" if isinstance(event_obj, ComputeEvent) else "comm"
        if kind == "comm":
            assert isinstance(event_obj, CommEvent)
        out.append((kind, event_obj.name, event_obj.op_id, len(event_obj.children)))
        stack.extend(reversed(event_obj.children))
    return out


def _assert_op_id_integrity(program: Any) -> None:
    """Every event carries a unique op_id stamp and the DFS stream is
    aligned 1:1 with the coarse op list."""
    stream = _describe_event_stream(program)
    op_ids = [op_id for _, _, op_id, _ in stream]
    assert all(op_id is not None for op_id in op_ids)
    assert len(set(op_ids)) == len(op_ids), "op_ids must be unique"
    assert len(stream) == len(program.ops)


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
    """Build (ScheduleInputs, ScheduleSpec) from one input set."""
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

    inputs = ScheduleInputs(
        dp=dp,
        pp=pp,
        tp=tp,
        cp=cp,
        ep=ep,
        comp_times=dict(comp_times),
        comm_metadata=comm_metadata,
        misc_metadata=misc_metadata,
    )
    spec = ScheduleSpec.from_pipeline_graph(
        inputs,
        include_backward=include_backward,
        include_optimizer=include_optimizer,
    )
    return inputs, spec


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
# Always-on synthetic tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_name", sorted(SYNTHETIC_CASES))
def test_synthetic_coarse_events_deterministic(case_name):
    """Two builds from the same ScheduleSpec produce identical schedule
    event streams (names, kinds, children order, op_id stamps), and the
    op_id stamps are unique and 1:1 with the coarse op list. (Successor of
    the legacy-constructor isomorphism differential, which was verified
    green through M6 and died at M7 with ``construct_fwd_bwd_graph``.)"""
    _, spec = _synthetic_case(**SYNTHETIC_CASES[case_name])
    program_a = build_coarse_program(spec, None)
    program_b = build_coarse_program(spec, None)

    _assert_op_id_integrity(program_a)
    assert _describe_event_stream(program_a) == _describe_event_stream(program_b)


def test_coarse_program_shape():
    """Typed-surface checks: roles, metadata, transfers, zero-byte events."""
    from program.ir import CollectiveOp, ComputeOp, Direction, OpRole, TransferOp

    _, spec = _synthetic_case(dp=2, pp=2, mb=2, layers=4, zero_stage=3)
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
    _, spec = _synthetic_case(dp=2, pp=2, mb=2, layers=4)
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
# Env-gated matrix determinism sweep (real configs, no AstraSim)
# ---------------------------------------------------------------------------


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
            interconnect_params=tc.pipeline_interconnect,
            transformer_blocks=tc.transformer_blocks,
            no_data_parallel=False,
        )
        cases.append(("final", tc, dispatcher))

        if getattr(tc, "gradient_accumulation_steps", 1) > 1:
            dispatcher_no_dp = LLMExecutionDispatcher(
                time_calc=tc,
                pipeline_graph=tc.pipeline_graph_no_dp,
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
        assert prefill_len > 0, "coarse sweep requires a prefill phase"
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
            interconnect_params=interconnect_params,
            transformer_blocks=transformer_blocks,
        )
        cases.append(("prefill", tc, dispatcher))
    return cases


@_diff_gate
@pytest.mark.parametrize("spec", COARSE_SPECS, ids=lambda s: s.spec_id)
def test_matrix_coarse_determinism(spec, tmp_path):
    """Build + evaluate the coarse program twice per case: event streams and
    op_id stamps must be identical across builds, and totals + full
    finish-time maps must be identical across builds (hybrid additionally
    retimed under synthetic block timings)."""
    from llm_execution import TransformerTimings

    hw_config, model_config, mode = _parse_spec_configs(spec, tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cases = _graph_cases(spec, hw_config, model_config, mode, out_dir)
    assert cases

    for label, tc, dispatcher in cases:
        network_model = tc.network_model
        interconnect_params = dispatcher.interconnect_params

        # Event-stream determinism + op_id integrity across builds.
        program = dispatcher._build_coarse_program()
        _assert_op_id_integrity(program)
        assert _describe_event_stream(program) == _describe_event_stream(
            dispatcher._build_coarse_program()
        ), f"{spec.spec_id}[{label}]: event streams diverge across builds"

        timings = None
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
            timings = BlockTimings(
                dense=dense, moe=moe, stage_dense=stage_dense, stage_moe=stage_moe
            )

        results = []
        for _ in range(2):
            program = dispatcher._build_coarse_program()
            if timings is not None:
                retimed = apply_block_timings(
                    program, timings, dispatcher._retime_dp_count()
                )
                assert retimed > 0
            results.append(evaluate_detailed(program, network_model, interconnect_params))

        assert results[0].total_time == results[1].total_time, (
            f"{spec.spec_id}[{label}]: totals diverge across builds"
        )
        assert results[0].finish_times == results[1].finish_times, (
            f"{spec.spec_id}[{label}]: finish-time maps diverge across builds"
        )
