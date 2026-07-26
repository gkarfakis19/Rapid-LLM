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

"""M3a fine-builder differential (dev gate; set ``RAPID_FINE_DIFF=1``).

For every flattened-family spec in the golden matrix (``equiv.configs``,
backend == "flattened" — this includes the zero2/zero3/ga2/recompute/gqa/
mesh2d/fault variants and the flattened inference prefill rows), build the
legacy in-process graphs (TimeCalculationLLM / TimeCalculationLLMInference
driven exactly far enough to obtain the pipeline root + transformer graph;
no AstraSim execution), then produce BOTH ET bundles:

* legacy: ``PipelineGraphFlattener.build`` -> ``apply_overlap_transforms``
  -> ``_propagate_local_hw_ids`` -> ``convert_rapid_llm_graph_to_chakra_et``
  (lower + emit) — the RAPID_LEGACY_FLATTEN=1 path;
* fine:   ``ScheduleSpec``/``BlockTemplate`` -> ``build_fine_program`` ->
  ``program.transforms.apply_overlap_transforms`` -> ``emit_chakra`` — the
  default path,

and compare them with the shadow comparator's standard (per-rank node
sequences in final id order: type + payload + sorted deps; manifest bytes;
comm_groups; tag pairing bijective — ``program.shadow.compare_et_bundles``).

Grad-accumulation specs additionally compare the no-DP (nonfinal-cycle)
graph pair. Decode graphs are structurally identical to prefill (forward-
only, seq_len=1 with decode GEMM shapes) and are covered by the prefill
rows plus the golden gates.

Run:
    RAPID_FINE_DIFF=1 ./.venv/bin/python -m pytest tests/test_fine_builder_diff.py -q
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RAPID_FINE_DIFF", "") != "1",
    reason="fine-builder differential is a dev gate; set RAPID_FINE_DIFF=1 to run",
)

from equiv.configs import MATRIX  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_CONFIG = (
    REPO_ROOT / "validation_scripts" / "validation_configs" / "model-config" / "Llama2-7B.yaml"
)
BASE_HW_CONFIG = (
    REPO_ROOT / "validation_scripts" / "validation_configs" / "hardware-config" / "a100_80GB.yaml"
)

FLATTENED_SPECS = [spec for spec in MATRIX if spec.backend == "flattened"]


def _sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)


def _parse_spec_configs(spec, tmp_path: Path):
    """Merge the spec overrides onto the base validation configs and parse."""
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
    """Drive the legacy time-calculation objects far enough to obtain the
    pipeline root(s) + transformer graph(s); returns [(label, tc, dispatcher)].
    """
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
            transformer_graph=tc.transformer_graph,
            transformer_forward_root=tc.transformer_forward_root,
            transformer_backward_root=tc.transformer_backward_root,
            no_data_parallel=False,
        )
        cases.append(("final", tc, dispatcher))

        if getattr(tc, "gradient_accumulation_steps", 1) > 1:
            dispatcher_no_dp = LLMExecutionDispatcher(
                time_calc=tc,
                pipeline_graph=tc.pipeline_graph_no_dp,
                pipeline_root=tc.pipeline_root_no_dp,
                interconnect_params=tc.pipeline_interconnect,
                transformer_graph=tc.transformer_graph_no_dp or tc.transformer_graph,
                transformer_forward_root=tc.transformer_forward_root_no_dp
                or tc.transformer_forward_root,
                transformer_backward_root=tc.transformer_backward_root_no_dp
                or tc.transformer_backward_root,
                no_data_parallel=True,
            )
            cases.append(("no_dp", tc, dispatcher_no_dp))
    else:
        from inference_timing import TimeCalculationLLMInference

        tc = TimeCalculationLLMInference(hw_config, model_config, mode, output_dir=str(out_dir))
        # Replicate the prefill graph construction of calc_time() up to (and
        # including) _prepare_execution_graphs — no AstraSim involved.
        batch_size = tc._effective_transformer_batch()
        decode_len = tc.model.decode_len
        prefill_len = tc.seq_len - decode_len
        assert prefill_len > 0, "differential requires a prefill phase"
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
            transformer_graph,
            transformer_forward_root,
            transformer_backward_root,
            _moe_graph,
            _moe_fwd,
            _moe_bwd,
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
            transformer_graph=transformer_graph,
            transformer_forward_root=transformer_forward_root,
            transformer_backward_root=transformer_backward_root,
        )
        cases.append(("prefill", tc, dispatcher))
    return cases


def _effective_dp(tc) -> int:
    run_type = str(getattr(getattr(tc, "model", None), "run_type", "training")).lower()
    return 1 if run_type == "inference" else max(1, getattr(tc, "dp", 1))


def _legacy_bundle(dispatcher, tc, out_dir: Path):
    """The RAPID_LEGACY_FLATTEN path up to (and including) ET conversion."""
    from astrasim_lib.executor import convert_rapid_llm_graph_to_chakra_et
    from llm_execution import PipelineGraphFlattener, apply_overlap_transforms

    flattener = PipelineGraphFlattener(
        pipeline_graph=dispatcher.pipeline_graph,
        transformer_graph=dispatcher.transformer_graph,
        moe_transformer_graph=None,
        rank_layout=dispatcher._rank_layout,
    )
    flattened_root = flattener.build(dispatcher.pipeline_root)
    flattened_root = apply_overlap_transforms(
        flattened_root,
        tc.get_parallelism_mode(),
        getattr(tc, "tp_overlap", 0.0),
        getattr(tc, "tp_sp_overlap", 0.0),
        getattr(tc, "cp_overlap", 0.0),
    )
    flattener._propagate_local_hw_ids(flattened_root)
    setattr(flattened_root, "_astrasim_rank_layout", dispatcher._rank_layout)
    dispatcher._attach_optimize_hint(flattened_root)

    out_dir.mkdir(parents=True, exist_ok=True)
    _prefix, rank_ids, manifest = convert_rapid_llm_graph_to_chakra_et(
        flattened_root,
        _effective_dp(tc),
        str(out_dir),
    )
    return rank_ids, manifest


def _fine_bundle(dispatcher, tc, out_dir: Path):
    """The default (M3a) path up to (and including) ET emission."""
    from program.block import BlockTemplate
    from program.et_emit import emit_chakra
    from program.layout import RankLayout
    from program.pipeline_fine import build_fine_program
    from program.schedule import ScheduleSpec

    run_type = str(getattr(getattr(tc, "model", None), "run_type", "training")).lower()
    include_backward = run_type != "inference"
    misc = getattr(dispatcher.pipeline_graph, "misc_metadata", None) or {}
    include_optimizer = str(misc.get("grad_accum_cycle", "final") or "final").lower() != "nonfinal"

    spec_obj = ScheduleSpec.from_pipeline_graph(
        dispatcher.pipeline_graph,
        include_backward=include_backward,
        include_optimizer=include_optimizer,
    )
    block_templates = {"dense": BlockTemplate.from_transformer_graph(dispatcher.transformer_graph)}

    layout_obj = None
    if dispatcher._rank_layout and dispatcher._rank_layout.get("axis_order"):
        layout_obj = RankLayout(
            axis_order=tuple(dispatcher._rank_layout.get("axis_order", [])),
            axis_sizes=dict(dispatcher._rank_layout.get("axis_sizes", {})),
            axis_strides=dict(dispatcher._rank_layout.get("axis_strides", {})),
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    program = build_fine_program(
        spec_obj,
        block_templates,
        layout_obj,
        no_data_parallel=dispatcher.no_data_parallel,
        dp_count=_effective_dp(tc),
        optimize_2dmap=(
            dict(dispatcher._first_dim_optimize_cfg)
            if dispatcher._first_dim_optimize_cfg
            else None
        ),
        gmap_workdir=str(out_dir),
        parallelism_mode=tc.get_parallelism_mode(),
        tp_overlap=getattr(tc, "tp_overlap", 0.0),
        tp_sp_overlap=getattr(tc, "tp_sp_overlap", 0.0),
        cp_overlap=getattr(tc, "cp_overlap", 0.0),
    )
    bundle = emit_chakra(program, str(out_dir), id_policy="legacy")
    return bundle.rank_ids, bundle.manifest_path


def _describe_ops(program):
    from program.ir import CollectiveOp, TransferOp

    out = []
    for op in program.ops:
        if isinstance(op, TransferOp):
            out.append(
                ("XFER", op.name, op.src_device, op.dst_device, op.size_bytes,
                 op.producer, tuple(op.consumers), op.send_seq, op.recv_seq)
            )
        elif isinstance(op, CollectiveOp):
            out.append(
                ("COLL", op.name, op.device, int(op.size_bytes), op.label,
                 tuple(op.group.members) if op.group else None, op.deps)
            )
        else:
            out.append(("COMP", op.name, op.device, op.duration, op.deps, op.legacy_op_id))
    return out


def test_program_level_tp_overlap_matches_proto_level_when_exact():
    """The Program->Program overlap passes are exact when split computes
    feed no cross-device consumers (single stage, single microbatch — see
    program/transforms.py's documented limitations). Verify against the
    proto-level (load-bearing) application on a synthetic tp=2 block."""
    from timing_model import CollectiveType
    from program.block import BlockTemplate
    from program.pipeline_fine import build_fine_program
    from program.schedule import ScheduleSpec
    from program.transforms import apply_tp_overlap

    comm_metadata = {
        "qkv_proj_forward_all_reduce": {
            "size": 4096,
            "type": CollectiveType.ALL_REDUCE,
            "participants": 2,
            "interconnect_type": "tp",
            "local_comp_time": 0,
        },
    }
    gemms = [
        {
            "name": "qkv_proj",
            "forward": {"duration": 1e-4, "comm_keys": ["qkv_proj_forward_all_reduce"]},
            "backward": {"duration": 2e-4, "comm_keys": []},
        },
        {
            "name": "MLP",
            "forward": {"duration": 3e-4, "comm_keys": []},
            "backward": {"duration": 4e-4, "comm_keys": []},
        },
    ]
    template = BlockTemplate.from_gemm_entries(gemms, comm_metadata)
    spec = ScheduleSpec(
        mb=1, num_layers=1, pp=1, dp=1, tp=2, cp=1, ep=1,
        layers_per_stage=(1,), moe_layer_mask=(), zero_stage=0,
        dp_microbatch_mode="every_mb", grad_accum_cycle="final",
        include_backward=True, include_optimizer=False,
        full_recomputation=False, pipeline_style_recompute=False,
        flattened_mode=True, model_type="gpt",
        comp_times={
            "embedding_f": 1e-5, "embedding_b": 1e-5,
            "linear_softmax_f": 1e-5, "linear_softmax_b": 1e-5,
            "transformer_f": 1e-3, "transformer_b": 2e-3,
        },
        comm_metadata={},
    )
    templates = {"dense": template}

    proto_level = build_fine_program(
        spec, templates, None,
        parallelism_mode="tensor", tp_overlap=0.6, tp_sp_overlap=0.0, cp_overlap=0.0,
    )
    program_level = build_fine_program(spec, templates, None)
    program_level = apply_tp_overlap(program_level, "tensor", 0.6, 0.0)

    assert _describe_ops(proto_level) == _describe_ops(program_level)


@pytest.mark.parametrize("spec", FLATTENED_SPECS, ids=[s.spec_id for s in FLATTENED_SPECS])
def test_fine_builder_matches_legacy_flatten(spec, tmp_path):
    from program.shadow import compare_et_bundles, load_comm_groups

    hw_config, model_config, mode = _parse_spec_configs(spec, tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    for label, tc, dispatcher in _graph_cases(spec, hw_config, model_config, mode, run_dir):
        legacy_dir = tmp_path / f"legacy_{_sanitize(label)}"
        fine_dir = tmp_path / f"fine_{_sanitize(label)}"

        legacy_ranks, legacy_manifest = _legacy_bundle(dispatcher, tc, legacy_dir)
        fine_ranks, fine_manifest = _fine_bundle(dispatcher, tc, fine_dir)

        problems = compare_et_bundles(
            reference_dir=str(legacy_dir),
            candidate_dir=str(fine_dir),
            reference_ranks=list(legacy_ranks),
            candidate_ranks=list(fine_ranks),
            reference_manifest=legacy_manifest,
            candidate_manifest=fine_manifest,
            reference_groups=load_comm_groups(str(legacy_dir)),
            candidate_groups=load_comm_groups(str(fine_dir)),
            reference_label="legacy",
            candidate_label="fine",
        )
        assert not problems, (
            f"{spec.spec_id} [{label}]: fine builder diverged from the legacy "
            f"flatten path (dirs: {legacy_dir} vs {fine_dir}):\n"
            + "\n".join(f"  - {p}" for p in problems)
        )
