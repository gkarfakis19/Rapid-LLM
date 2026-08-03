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

"""L0 invariants W1-W4 (INTERFACES.md §1.8)."""

from __future__ import annotations

import pytest

from timing_model import CollectiveType

from program.workload import (
    CommSpec,
    CommSpecTable,
    DpMicrobatchMode,
    DurationTable,
    FrozenDurations,
    GradAccumCycle,
    ModelShape,
    OverlapSpec,
    ParallelDegrees,
    RunPolicy,
    RunType,
    WorkloadError,
    ceil_div,
)

from tests.test_policies import Cfg, make_workload, raw_comm_metadata


# ---------------------------------------------------------------------------
# W1 — a malformed input raises at construction (the schedule.py:186 hazard)
# ---------------------------------------------------------------------------


def test_wrong_object_can_no_longer_produce_mb_zero() -> None:
    """The schedule.py:186 hazard, now unrepresentable.

    ``ScheduleSpec.from_pipeline_graph`` read every field through a
    ``getattr(obj, name, default)`` chain, so an object with empty
    ``misc_metadata`` yielded ``mb == 0`` / ``num_layers == 0`` and built an
    empty schedule in silence. That constructor is deleted; ``ModelShape``
    raises instead, and ``WorkloadSpec.from_timing`` — the one producer seam —
    goes through it.
    """
    from program.workload import WorkloadError as _WE

    with pytest.raises(WorkloadError, match="micro_batches"):
        ModelShape(num_layers=4, micro_batches=0, model_type="gpt")
    with pytest.raises(WorkloadError, match="num_layers"):
        ModelShape(num_layers=0, micro_batches=4, model_type="gpt")
    with pytest.raises(WorkloadError, match="micro_batches"):
        ModelShape(num_layers=4, micro_batches=None, model_type="gpt")  # type: ignore[arg-type]


def test_model_shape_validation() -> None:
    shape = ModelShape(
        num_layers=3, micro_batches=2, model_type="GPT", moe_layer_mask=(1, 0, 1)
    )
    assert shape.model_type == "gpt"
    assert shape.moe_layer_mask == (True, False, True)
    assert shape.is_moe_layer(0) and not shape.is_moe_layer(1)
    assert shape.has_moe_layers
    with pytest.raises(WorkloadError):
        shape.is_moe_layer(9)
    with pytest.raises(WorkloadError):
        ModelShape(num_layers=3, micro_batches=2, model_type="gpt", moe_layer_mask=(True,))

    dense = ModelShape(num_layers=3, micro_batches=2, model_type="gpt")
    assert not dense.has_moe_layers
    assert not dense.is_moe_layer(0)
    assert not dense.is_moe_layer(99)  # no mask -> no lookup, no error


def test_parallel_degrees_validation() -> None:
    degrees = ParallelDegrees(tp=2, cp=2, ep=2, pp=2, dp=2)
    assert degrees.cluster_size() == 8
    assert degrees.of("tp") == 2 and degrees.of("dp") == 2
    with pytest.raises(WorkloadError):
        degrees.of("zz")
    for bad in ({"tp": 0}, {"dp": -1}, {"pp": True}):
        kwargs = dict(tp=1, cp=1, ep=1, pp=1, dp=1)
        kwargs.update(bad)
        with pytest.raises(WorkloadError):
            ParallelDegrees(**kwargs)  # type: ignore[arg-type]


def test_run_policy_derivations_have_one_home() -> None:
    """The three copies at llm_execution.py:392-395 / :606-612 / :742-746
    (audit A10) collapse to these properties."""
    degrees = ParallelDegrees(tp=1, cp=1, ep=1, pp=4, dp=8)
    shape = ModelShape(num_layers=8, micro_batches=8, model_type="gpt")

    training = RunPolicy(
        run_type=RunType.TRAINING,
        grad_accum_cycle=GradAccumCycle.FINAL,
        dp_microbatch_mode=DpMicrobatchMode.EVERY_MB,
        zero_stage=0,
    )
    assert training.include_backward and training.include_optimizer
    assert training.effective_dp(degrees) == 8
    assert training.retime_dp_count(degrees) == 8
    assert training.interleave_scale(degrees, shape) == 1.0

    inference = RunPolicy(
        run_type=RunType.INFERENCE,
        grad_accum_cycle=GradAccumCycle.FINAL,
        dp_microbatch_mode=DpMicrobatchMode.EVERY_MB,
        zero_stage=0,
    )
    assert not inference.include_backward
    assert inference.effective_dp(degrees) == 1

    nonfinal = RunPolicy(
        run_type=RunType.TRAINING,
        grad_accum_cycle=GradAccumCycle.NONFINAL,
        dp_microbatch_mode=DpMicrobatchMode.EVERY_MB,
        zero_stage=0,
    )
    assert nonfinal.include_backward and not nonfinal.include_optimizer
    assert nonfinal.is_nonfinal_grad_accum_cycle

    # Class B 8: the closed-form bubble correction, verbatim
    interleaved = RunPolicy(
        run_type=RunType.TRAINING,
        grad_accum_cycle=GradAccumCycle.FINAL,
        dp_microbatch_mode=DpMicrobatchMode.EVERY_MB,
        zero_stage=0,
        pipeline_interleave=4,
    )
    assert interleaved.interleave_scale(degrees, shape) == pytest.approx(
        (8 + 3 / 4) / 11
    )
    flat = ParallelDegrees(tp=1, cp=1, ep=1, pp=1, dp=1)
    assert interleaved.interleave_scale(flat, shape) == 1.0


def test_run_policy_rejects_untyped_inputs() -> None:
    kwargs = dict(
        run_type=RunType.TRAINING,
        grad_accum_cycle=GradAccumCycle.FINAL,
        dp_microbatch_mode=DpMicrobatchMode.EVERY_MB,
        zero_stage=0,
    )
    for field, value in (
        ("run_type", "training"),
        ("grad_accum_cycle", "final"),
        ("dp_microbatch_mode", "every_mb"),
        ("zero_stage", -1),
        ("pipeline_interleave", 0),
    ):
        bad = dict(kwargs)
        bad[field] = value
        with pytest.raises(WorkloadError):
            RunPolicy(**bad)  # type: ignore[arg-type]


def test_enum_parsers_reject_unknown_strings() -> None:
    assert RunType.parse("Inference") is RunType.INFERENCE
    assert GradAccumCycle.parse("NONFINAL") is GradAccumCycle.NONFINAL
    assert DpMicrobatchMode.parse("last_mb") is DpMicrobatchMode.LAST_MB
    for parser, value in (
        (RunType.parse, "sampling"),
        (GradAccumCycle.parse, "middle"),
        (DpMicrobatchMode.parse, "first_mb"),
    ):
        with pytest.raises(WorkloadError):
            parser(value)


# ---------------------------------------------------------------------------
# W2 — the duration table is the only mutable member
# ---------------------------------------------------------------------------


def test_writeback_revision_and_snapshot_isolation() -> None:
    table = DurationTable({"transformer_f": 1.0, "transformer_b": 2.0, "optimizer": 3.0})
    assert table.revision == 0
    before = table.snapshot()
    assert before["transformer_f"] == 1.0

    revision = table.write_block_timings(dense_forward=10.0, dense_backward=20.0)
    assert revision == 1 and table.revision == 1

    # the pre-write-back snapshot is unchanged (the hybrid PIPELINE build must see
    # PRISTINE analytical durations - llm_execution.py:497 vs :500)
    assert before["transformer_f"] == 1.0
    assert before.revision == 0

    after = table.snapshot()
    assert after["transformer_f"] == 10.0
    assert after["transformer_f_dense"] == 10.0
    assert after["transformer_b_dense"] == 20.0
    assert after.revision == 1

    table.write_block_timings(moe_forward=30.0, moe_backward=40.0)
    latest = table.snapshot()
    assert table.revision == 2
    assert latest["transformer_f_moe"] == 30.0
    assert latest["transformer_f"] == 10.0  # untouched by a moe-only write-back


def test_writeback_rejects_negative_durations() -> None:
    table = DurationTable({"transformer_f": 1.0})
    with pytest.raises(WorkloadError):
        table.write_block_timings(dense_forward=-1.0)
    assert table.revision == 0  # atomic: nothing was written
    with pytest.raises(WorkloadError):
        DurationTable({"embedding_f": -1.0})


def test_duration_table_filters_non_numeric_and_defaults_explicitly() -> None:
    table = DurationTable({"a": 1, "b": "nope", "c": True, "d": 2.5})
    frozen = table.snapshot()
    assert set(frozen) == {"a", "d"}
    assert frozen.get_or("a") == 1.0
    assert frozen.get_or("missing") == 0.0
    assert frozen.get_or("missing", 7.0) == 7.0
    with pytest.raises(WorkloadError, match="missing"):
        frozen["missing"]
    assert isinstance(frozen, FrozenDurations)
    assert len(frozen) == 2


def test_frozen_workload_is_a_pure_input() -> None:
    spec = make_workload(Cfg(dp=2))
    fw = spec.freeze()
    assert fw.durations.revision == 0
    spec.durations.write_block_timings(dense_forward=999.0)
    assert fw.durations["transformer_f_dense"] != 999.0
    assert spec.freeze().durations["transformer_f_dense"] == 999.0
    assert spec.freeze().durations.revision == 1


# ---------------------------------------------------------------------------
# W3 — the comm table names every missing key
# ---------------------------------------------------------------------------


def test_comm_table_require_raises_workload_error_not_keyerror() -> None:
    table = CommSpecTable.from_legacy(raw_comm_metadata(Cfg(dp=2, zero_stage=3)))
    assert table.require("embedding").kind is CollectiveType.REDUCE_SCATTER
    with pytest.raises(WorkloadError) as excinfo:
        table.require("no_such_key")
    message = str(excinfo.value)
    assert "no_such_key" in message and "embedding" in message
    assert not isinstance(excinfo.value, KeyError)


def test_comm_table_preserves_insertion_order_and_axis_lookup() -> None:
    raw = raw_comm_metadata(Cfg(dp=2, zero_stage=3))
    table = CommSpecTable.from_legacy(raw)
    assert list(table) == list(raw)
    assert set(table.keys_with_axis("dp")) == {
        k for k, v in raw.items() if v["interconnect_type"] == "dp"
    }
    assert table.keys_with_axis("pp") == ("cross_layer",)


def test_comm_spec_keeps_float_bytes_and_separates_axes_from_participants() -> None:
    """dp reducer sizes are floats and MUST stay floats (analytic_sim.py:49-51);
    the ``int()`` truncation belongs to the emitter."""
    table = CommSpecTable.from_legacy(raw_comm_metadata(Cfg(dp=4)))
    spec = table.require("transformer_dense")
    assert isinstance(spec.size_bytes, float) and spec.size_bytes == 1000.5
    assert spec.axes == ("dp",) and spec.participants == 4

    # TENSOR_CONTEXT_HYBRID output_proj: participants='tp', interconnect='cp'
    hybrid = CommSpec.from_legacy(
        "output_proj_forward",
        {
            "size": 32,
            "type": CollectiveType.REDUCE_SCATTER,
            "participants": 8,
            "interconnect_type": "cp",
        },
    )
    assert hybrid.axes == ("cp",)
    assert hybrid.participants == 8  # NOT derived from the axis


def test_comm_spec_validation() -> None:
    with pytest.raises(WorkloadError, match="type"):
        CommSpec.from_legacy("x", {"size": 1, "interconnect_type": "dp"})
    with pytest.raises(WorkloadError, match="CollectiveType"):
        CommSpec.from_legacy("x", {"size": 1, "type": "all_reduce", "interconnect_type": "dp"})
    with pytest.raises(WorkloadError, match="communicator axis"):
        CommSpec.from_legacy("x", {"size": 1, "type": CollectiveType.ALL_REDUCE})
    with pytest.raises(WorkloadError, match="placement"):
        CommSpec.from_legacy(
            "x",
            {
                "size": 1,
                "type": CollectiveType.ALL_REDUCE,
                "interconnect_type": "dp",
                "placement": "sideways",
            },
        )
    with pytest.raises(WorkloadError, match="Duplicate"):
        CommSpecTable(
            [
                CommSpec(key="a", size_bytes=1, kind=CollectiveType.ALL_REDUCE, axes=("dp",), participants=1),
                CommSpec(key="a", size_bytes=1, kind=CollectiveType.ALL_REDUCE, axes=("dp",), participants=1),
            ]
        )
    with pytest.raises(WorkloadError, match="disagrees"):
        CommSpecTable(
            [("b", CommSpec(key="a", size_bytes=1, kind=CollectiveType.ALL_REDUCE, axes=("dp",), participants=1))]
        )


def test_comm_spec_carries_class_b_item_5_local_comp_time() -> None:
    """Class B 5: ``local_comp_time`` is carried as data and never
    materialized — the assignment was already dead in legacy and attaching it
    per dp edge would double-count by ``layers_per_stage``."""
    table = CommSpecTable.from_legacy(raw_comm_metadata(Cfg(dp=2)))
    assert table.require("transformer_dense").local_comp_time == 3.5
    assert table.require("embedding").local_comp_time == 0.0


# ---------------------------------------------------------------------------
# WorkloadSpec assembly
# ---------------------------------------------------------------------------


def test_workload_spec_rejects_untyped_members() -> None:
    spec = make_workload(Cfg(dp=2, moe=True, ep=2))
    for field, value in (
        ("degrees", (1, 1, 1, 1, 1)),
        ("shape", None),
        ("run", "training"),
        ("comm", {}),
        ("blocks", None),
        ("overlap", 0.5),
        ("durations", {}),
    ):
        with pytest.raises(WorkloadError):
            spec.with_(**{field: value})


def test_workload_spec_requires_a_moe_template_for_moe_layers() -> None:
    from program.workload import BlockTemplates

    spec = make_workload(Cfg(moe=True))
    assert spec.blocks.moe is not None
    assert spec.block_template(0) is spec.blocks.moe
    assert spec.block_template(1) is spec.blocks.dense
    with pytest.raises(WorkloadError, match="MoE"):
        spec.with_(blocks=BlockTemplates(dense=spec.blocks.dense, moe=None))


def test_overlap_spec_validation() -> None:
    assert OverlapSpec(parallelism_mode="tensor", by_axis={"tp": 0.5}).fraction("tp") == 0.5
    assert OverlapSpec(parallelism_mode="tensor").fraction("cp") == 0.0
    with pytest.raises(WorkloadError):
        OverlapSpec(parallelism_mode="tensor", by_axis={"tp": -0.1})

    import enum

    class _Mode(enum.Enum):
        TENSOR_SEQUENCE = "tensor_sequence"

    resolved = OverlapSpec.from_legacy(_Mode.TENSOR_SEQUENCE, tp_sp_overlap=0.4)
    assert resolved.mode_label == "tensor_sequence"
    assert resolved.fraction("tp") == 0.4


def test_ceil_div_matches_the_legacy_guard() -> None:
    assert ceil_div(4096, 4) == 1024.0
    assert ceil_div(4097, 4) == 1025.0
    assert ceil_div(10, 0) == 10.0  # max(1, divisor), pipeline_fine.py:645


# ---------------------------------------------------------------------------
# sw_param.estimate_memory (2026-08-02)
# ---------------------------------------------------------------------------


def test_estimate_memory_is_declared_in_sw_param():
    """``sw_param.estimate_memory: false`` must reach ``SWConfig`` — the first
    wiring read ``self.sw_config`` on the TimeCalculation object, an attribute
    that does not exist there (the config lives on ``hw_config``), so the
    getattr chain always answered the default and the lever was API-only."""
    import yaml
    from pathlib import Path
    from config import SWConfig

    hw_path = (
        Path(__file__).resolve().parents[1]
        / "validation_scripts"
        / "validation_configs"
        / "hardware-config"
        / "A100_SXM4_80GB_base.yaml"
    )
    base = yaml.safe_load(hw_path.read_text())["sw_param"]
    assert SWConfig.from_dict(dict(base)).estimate_memory is True
    assert SWConfig.from_dict({**base, "estimate_memory": False}).estimate_memory is False
    assert SWConfig.from_dict({**base, "estimate_memory": "false"}).estimate_memory is False
    assert SWConfig.from_dict({**base, "estimate_memory": True}).estimate_memory is True


def test_validate_graph_is_declared_in_sw_param():
    """Opt-in build-time validation (2026-08-02): OFF by default in production
    — it re-proves invariants build() holds by construction — and switchable
    via sw_param.validate_graph or RAPID_VALIDATE_GRAPH. The wire-level
    emission postconditions are unaffected and always on."""
    import yaml
    from pathlib import Path
    from config import SWConfig

    hw_path = (
        Path(__file__).resolve().parents[1]
        / "validation_scripts"
        / "validation_configs"
        / "hardware-config"
        / "A100_SXM4_80GB_base.yaml"
    )
    base = yaml.safe_load(hw_path.read_text())["sw_param"]
    assert SWConfig.from_dict(dict(base)).validate_graph is False
    assert SWConfig.from_dict({**base, "validate_graph": True}).validate_graph is True
    assert SWConfig.from_dict({**base, "validate_graph": "true"}).validate_graph is True
    assert SWConfig.from_dict({**base, "validate_graph": "false"}).validate_graph is False


def test_deadlock_errors_name_the_validation_switch():
    """The failure a user actually sees must tell them how to diagnose it."""
    from pathlib import Path

    import llm_execution

    src = Path(llm_execution.__file__).read_text()
    for needle in (
        "returned non-positive duration",
        "returned no per-rank timings",
    ):
        idx = src.find(needle)
        assert idx != -1, needle
        assert "RAPID_VALIDATE_GRAPH" in src[idx : idx + 400], (
            f"error {needle!r} does not name the validation switch"
        )
