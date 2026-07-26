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

"""M6 hierarchical-pipeline-emission tests.

The M6 pre-cutover differential compared, for every hierarchical golden
spec (5 train rows + ``moe:ep2`` + ``fault_tp`` + ``fault_pp``, plus the 3
inference hier specs incl. ViT — inference covering BOTH the prefill graph
and a representative ``prepare_decode_graphs`` decode-step graph), the
legacy pipeline path (legacy ``construct_fwd_bwd_graph`` root [+
``_apply_transformer_time`` retime write-back] -> ``lower_to_program`` ->
``emit_chakra``) against the M6 replacement (``build_coarse_program`` [+
``apply_block_timings``] -> ``lower_coarse_for_emission`` ->
``emit_chakra``) in three arms: un-retimed, synthetic block timings (incl.
per-(dp, stage) fault-variant overrides), and REAL block timings from
``_run_transformer_astrasim`` (AstraSim binary). All 11 specs matched on
all arms: identical Program op streams AND ``compare_et_bundles``-clean
AND byte-identical bundle files. The write-back arms died with
``_apply_transformer_time``/``_assign_transformer_durations`` in the same
change.

What remains:

* an always-on synthetic test pinning the emission-lowered coarse Program
  shape over a ("pp","dp") sublayout (no configs, no AstraSim);
* an env-gated sweep (``RAPID_HIER_DIFF=1``) over the hierarchical golden
  matrix keeping the halves that still have two live implementations:
  the UN-RETIMED cross-differential (legacy pipeline root, which
  ``_prepare_execution_graphs`` keeps building until M7, vs the coarse
  schedule events — both lowered through the shared pass) plus a retimed
  build/emit determinism check (two builds byte-identical).

Run:
    RAPID_HIER_DIFF=1 ./.venv/bin/python -m pytest tests/test_hier_builder_diff.py -q
"""

from __future__ import annotations

import copy
import filecmp
import os
from pathlib import Path

import pytest

from equiv.configs import MATRIX  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_CONFIG = (
    REPO_ROOT / "validation_scripts" / "validation_configs" / "model-config" / "Llama2-7B.yaml"
)
BASE_HW_CONFIG = (
    REPO_ROOT / "validation_scripts" / "validation_configs" / "hardware-config" / "a100_80GB.yaml"
)

HIER_SPECS = [spec for spec in MATRIX if spec.backend == "hierarchical"]

_diff_gate = pytest.mark.skipif(
    os.environ.get("RAPID_HIER_DIFF", "") != "1",
    reason="hierarchical emission sweep is a dev gate; set RAPID_HIER_DIFF=1 to run",
)


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


def _inference_prefill_case(tc, out_dir: Path):
    from llm_execution import LLMExecutionDispatcher

    batch_size = tc._effective_transformer_batch()
    decode_len = tc.model.decode_len
    prefill_len = tc.seq_len - decode_len
    assert prefill_len > 0, "hier sweep requires a prefill phase"
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
    return LLMExecutionDispatcher(
        time_calc=tc,
        pipeline_graph=pipeline_graph,
        pipeline_root=pipeline_root,
        interconnect_params=interconnect_params,
        transformer_blocks=transformer_blocks,
    )


def _inference_decode_case(tc):
    """A representative decode-step graph (what DecodeGraph re-enters the
    dispatcher with, per sampled step)."""
    import llm_util
    from llm_execution import LLMExecutionDispatcher

    batch_size = tc._effective_transformer_batch()
    decode_gemm_shapes = llm_util.process_decode_gemm_shapes(
        tc,
        batch_size=batch_size,
        current_seq_len=tc.seq_len,
        d_model=tc.hidden_dim,
        num_heads=tc.num_heads,
        kv_heads=tc.kv_heads,
        intermediate_size=tc.moe_intermediate_size if tc.use_moe else tc.intermediate_size,
        vocab_size=tc.vocab_size,
        model_type=tc.model_type,
    )
    (
        decode_pipeline_graph,
        decode_pipeline_root,
        _,
        _,
        decode_transformer_blocks,
        decode_interconnect_params,
    ), _ = tc.prepare_decode_graphs(
        batch_size=batch_size,
        total_seq_len=tc.seq_len,
        gemm_shapes=decode_gemm_shapes,
    )
    return LLMExecutionDispatcher(
        time_calc=tc,
        pipeline_graph=decode_pipeline_graph,
        pipeline_root=decode_pipeline_root,
        interconnect_params=decode_interconnect_params,
        transformer_blocks=decode_transformer_blocks,
    )


def _graph_cases(spec, hw_config, model_config, mode, out_dir: Path):
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
    else:
        from inference_timing import TimeCalculationLLMInference

        tc = TimeCalculationLLMInference(hw_config, model_config, mode, output_dir=str(out_dir))
        cases.append(("prefill", tc, _inference_prefill_case(tc, out_dir)))
        if int(getattr(tc.model, "decode_len", 0) or 0) > 0:
            cases.append(("decode", tc, _inference_decode_case(tc)))
    return cases


def _effective_dp(tc) -> int:
    run_type = str(getattr(getattr(tc, "model", None), "run_type", "training")).lower()
    return 1 if run_type == "inference" else max(1, getattr(tc, "dp", 1))


def _legacy_root_bundle(dispatcher, tc, out_dir: Path):
    """Un-retimed legacy arm: pipeline root -> lower -> emit — the pre-M6
    `_run_full_astrasim_hierarchical` plumbing (root attach included).
    Lives until `construct_fwd_bwd_graph` stops producing roots (M7)."""
    from program.et_emit import emit_chakra
    from program.legacy_lowering import lower_to_program

    out_dir.mkdir(parents=True, exist_ok=True)
    root = dispatcher.pipeline_root
    layout = getattr(dispatcher, "_pipeline_rank_layout", None)
    if layout:
        setattr(root, "_astrasim_rank_layout", layout)
    elif hasattr(root, "_astrasim_rank_layout"):
        delattr(root, "_astrasim_rank_layout")
    if dispatcher._first_dim_optimize_cfg:
        setattr(root, "_optimize_2dmap", dict(dispatcher._first_dim_optimize_cfg))
    elif hasattr(root, "_optimize_2dmap"):
        delattr(root, "_optimize_2dmap")
    program = lower_to_program(
        root,
        _effective_dp(tc),
        getattr(root, "_astrasim_rank_layout", None),
        gmap_workdir=str(out_dir),
    )
    bundle = emit_chakra(program, str(out_dir), id_policy="legacy")
    return program, bundle


def _coarse_bundle(dispatcher, coarse_program, out_dir: Path):
    """New arm: (retimed) coarse program -> lower_coarse_for_emission ->
    emit, the M6 `_run_full_astrasim_hierarchical` plumbing."""
    from program.et_emit import emit_chakra
    from program.pipeline_coarse import lower_coarse_for_emission

    out_dir.mkdir(parents=True, exist_ok=True)
    optimize_cfg = (
        dict(dispatcher._first_dim_optimize_cfg) if dispatcher._first_dim_optimize_cfg else None
    )
    program = lower_coarse_for_emission(
        coarse_program,
        layout_descriptor=dispatcher._pipeline_rank_layout or None,
        optimize_2dmap=optimize_cfg,
        gmap_workdir=str(out_dir),
    )
    bundle = emit_chakra(program, str(out_dir), id_policy="legacy")
    return program, bundle


def _describe_ops(program):
    from program.ir import CollectiveOp, TransferOp

    out = []
    for op in program.ops:
        if isinstance(op, TransferOp):
            out.append(
                ("XFER", op.name, op.src_device, op.dst_device, op.size_bytes,
                 op.producer, tuple(op.consumers), op.send_seq, op.recv_seq, op.legacy_tag)
            )
        elif isinstance(op, CollectiveOp):
            out.append(
                ("COLL", op.name, op.device, int(op.size_bytes), op.label,
                 tuple(op.group.members) if op.group else None, op.deps, op.legacy_op_id)
            )
        else:
            out.append(("COMP", op.name, op.device, op.duration, op.deps, op.legacy_op_id))
    return out


def _bundle_files(bundle_dir: Path):
    names = sorted(p.name for p in bundle_dir.iterdir() if p.is_file())
    return [n for n in names if n.endswith(".et") or n.endswith(".json")]


def _assert_bundles_match(spec_id, label, arm, dir_a: Path, dir_b: Path, bundle_a, bundle_b,
                          reference_label="legacy", candidate_label="coarse"):
    from program.shadow import compare_et_bundles

    problems = compare_et_bundles(
        reference_dir=str(dir_a),
        candidate_dir=str(dir_b),
        reference_ranks=[int(r) for r in bundle_a.rank_ids],
        candidate_ranks=[int(r) for r in bundle_b.rank_ids],
        reference_manifest=bundle_a.manifest_path,
        candidate_manifest=bundle_b.manifest_path,
        reference_groups={k: list(v) for k, v in bundle_a.comm_groups.items()},
        candidate_groups={k: list(v) for k, v in bundle_b.comm_groups.items()},
        reference_label=reference_label,
        candidate_label=candidate_label,
    )
    assert not problems, (
        f"{spec_id} [{label}/{arm}]: bundle mismatch:\n" + "\n".join(problems)
    )

    files_a = _bundle_files(dir_a)
    files_b = _bundle_files(dir_b)
    assert files_a == files_b, (
        f"{spec_id} [{label}/{arm}]: bundle file sets differ: {files_a} vs {files_b}"
    )
    for name in files_a:
        assert filecmp.cmp(dir_a / name, dir_b / name, shallow=False), (
            f"{spec_id} [{label}/{arm}]: bundle file {name} not byte-identical"
        )


def _synthetic_block_timings(spec, tc):
    """Synthetic retime inputs (per-(dp, stage) fault-style overrides)."""
    from llm_execution import TransformerTimings
    from program.retime import BlockTimings

    dp = max(1, getattr(tc, "dp", 1))
    pp = max(1, getattr(tc, "pp", 1))
    dense = TransformerTimings(forward=0.00125, backward=0.0025)
    moe = TransformerTimings(forward=0.0035, backward=0.00475) if spec.use_moe else None
    stage_dense = {(0, 0): TransformerTimings(forward=0.0015, backward=0.00275)}
    if dp > 1 and pp > 1:
        stage_dense[(1, 1)] = TransformerTimings(forward=0.00175, backward=0.003)
    stage_moe = (
        {(0, min(1, pp - 1)): TransformerTimings(forward=0.004, backward=0.005)}
        if moe
        else {}
    )
    return BlockTimings(dense=dense, moe=moe, stage_dense=stage_dense, stage_moe=stage_moe)


# ---------------------------------------------------------------------------
# Always-on synthetic emission-shape pin
# ---------------------------------------------------------------------------


def test_hier_emission_program_shape_synthetic():
    """Pin the emission-lowered coarse Program over a ("pp","dp") sublayout:
    devices = stages in emission order, dp stage groups on emit, retimed
    per-DP durations flowing through the events' duration_profile."""
    from program.ir import ComputeOp, TransferOp
    from program.pipeline_coarse import build_coarse_program, lower_coarse_for_emission
    from program.retime import BlockTimings, apply_block_timings
    from program.schedule import ScheduleSpec
    from timing_model import CollectiveType

    comm_metadata = {
        "transformer_dense": {
            "size": 12345678.5,
            "type": CollectiveType.ALL_REDUCE,
            "participants": 2,
            "interconnect_type": "dp",
            "local_comp_time": 0,
        },
        "embedding": {
            "size": 524288,
            "type": CollectiveType.ALL_REDUCE,
            "participants": 2,
            "interconnect_type": "dp",
            "local_comp_time": 0,
        },
        "softmax": {
            "size": 262144,
            "type": CollectiveType.ALL_REDUCE,
            "participants": 2,
            "interconnect_type": "dp",
            "local_comp_time": 0,
        },
        "cross_layer": {
            "size": 1048576,
            "type": CollectiveType.PIPELINE,
            "participants": 2,
            "interconnect_type": "pp",
            "local_comp_time": 0,
        },
    }
    from program.block import comm_metadata_from_legacy

    spec = ScheduleSpec(
        mb=2, num_layers=4, pp=2, dp=2, tp=1, cp=1, ep=1,
        layers_per_stage=(2, 2), moe_layer_mask=(), zero_stage=0,
        dp_microbatch_mode="every_mb", grad_accum_cycle="final",
        include_backward=True, include_optimizer=True,
        full_recomputation=False, pipeline_style_recompute=False,
        flattened_mode=False, model_type="gpt",
        comp_times={
            "linear_softmax_f": 0.002, "linear_softmax_b": 0.0025,
            "transformer_f": 0.010, "transformer_b": 0.021,
            "embedding_f": 0.001, "embedding_b": 0.0015,
            "optimizer": 0.004,
        },
        comm_metadata=comm_metadata_from_legacy(comm_metadata),
    )
    coarse = build_coarse_program(spec, None, dp_count=2)

    class _T:
        def __init__(self, forward, backward):
            self.forward = forward
            self.backward = backward

    retimed = apply_block_timings(
        coarse,
        BlockTimings(dense=_T(0.005, 0.009), stage_dense={(1, 1): _T(0.006, 0.010)}),
        dp_count=2,
    )
    assert retimed == 16  # 2 mb x 4 layers x fwd+bwd

    descriptor = {
        "axis_order": ["pp", "dp"],
        "axis_sizes": {"pp": 2, "dp": 2},
        "axis_strides": {"pp": 1, "dp": 2},
    }
    program = lower_coarse_for_emission(coarse, layout_descriptor=descriptor)

    # Emission order: stage devices sorted, dp-major ranks derived from them.
    assert program.devices == (0, 1)
    assert program.dp_count == 2
    assert program.layout.axis_order == ("pp", "dp")

    # Retimed layers carry the per-DP profile; stage-1 dp-1 override applied.
    layer_ops = [
        op for op in program.ops
        if isinstance(op, ComputeOp) and op.name.startswith("transformer_layer")
        and not op.name.endswith("_recompute")
    ]
    assert len(layer_ops) == 16
    for op in layer_ops:
        assert len(op.duration) == 2
        fwd = not op.name.endswith("_b")
        if op.device == 1:
            assert op.duration == ((0.005, 0.006) if fwd else (0.009, 0.010))
        else:
            assert op.duration == ((0.005, 0.005) if fwd else (0.009, 0.009))

    # Cross-stage pipeline transfers exist and carry the legacy op_id tags.
    transfers = [op for op in program.ops if isinstance(op, TransferOp)]
    cross = [op for op in transfers if op.src_device != op.dst_device]
    assert cross and all(op.legacy_tag is not None for op in cross)

    # dp collectives (is_dp, unlabeled) survive to emission; per-stage Kahn
    # ordering means every op's deps precede it.
    for op in program.ops:
        deps = getattr(op, "deps", ())
        assert all(dep < op.uid for dep in deps)


# ---------------------------------------------------------------------------
# Env-gated matrix sweep
# ---------------------------------------------------------------------------


@_diff_gate
@pytest.mark.parametrize("spec", HIER_SPECS, ids=[s.spec_id for s in HIER_SPECS])
def test_hier_pipeline_emission_sweep(spec, tmp_path):
    from program.retime import apply_block_timings

    hw_config, model_config, mode = _parse_spec_configs(spec, tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    cases = _graph_cases(spec, hw_config, model_config, mode, run_dir)
    assert cases, f"{spec.spec_id}: no pipeline cases produced"

    for label, tc, dispatcher in cases:
        # ---- un-retimed cross-differential (two live implementations) ----
        coarse_plain = dispatcher._build_coarse_program()
        dir_a = tmp_path / f"legacy_{_sanitize(label)}_plain"
        dir_b = tmp_path / f"coarse_{_sanitize(label)}_plain"
        program_a, bundle_a = _legacy_root_bundle(dispatcher, tc, dir_a)
        program_b, bundle_b = _coarse_bundle(dispatcher, coarse_plain, dir_b)
        assert _describe_ops(program_a) == _describe_ops(program_b), (
            f"{spec.spec_id} [{label}/plain]: op streams diverge"
        )
        _assert_bundles_match(spec.spec_id, label, "plain", dir_a, dir_b, bundle_a, bundle_b)

        # ---- retimed determinism (build + retime + lower + emit twice) ---
        timings = _synthetic_block_timings(spec, tc)
        dirs = [tmp_path / f"retimed_{_sanitize(label)}_{i}" for i in (0, 1)]
        outputs = []
        for out_dir in dirs:
            coarse = dispatcher._build_coarse_program()
            retimed = apply_block_timings(coarse, timings, dispatcher._retime_dp_count())
            assert retimed > 0
            outputs.append(_coarse_bundle(dispatcher, coarse, out_dir))
        (program_0, bundle_0), (program_1, bundle_1) = outputs
        assert _describe_ops(program_0) == _describe_ops(program_1), (
            f"{spec.spec_id} [{label}/retimed]: two builds produced different op streams"
        )
        _assert_bundles_match(
            spec.spec_id, label, "retimed", dirs[0], dirs[1], bundle_0, bundle_1,
            reference_label="build0", candidate_label="build1",
        )
