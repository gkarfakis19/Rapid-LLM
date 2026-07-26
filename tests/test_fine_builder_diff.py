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

"""M3 fine-builder tests.

The M3a legacy-flattener differential is gone with the legacy flattener
(M3b deleted ``PipelineGraphFlattener`` and the ``RAPID_LEGACY_FLATTEN``
path). What remains:

* an always-on unit test pinning the Program-level tp-overlap pass against
  the proto-level (load-bearing) application on a synthetic block;
* a determinism gate (``RAPID_FINE_DIFF=1``): for every flattened-family
  spec in the golden matrix, build the FINE program twice and emit twice —
  the op streams must compare equal and the emitted ET bundles (rank
  ``.et`` files, manifest, ``comm_groups.json``) must be byte-identical.
  This is the process-internal replacement for the old differential: the
  builder has no hidden iteration-order dependence (the legacy Step-11
  set-order tie was resolved deterministically in M3a).

Run:
    RAPID_FINE_DIFF=1 ./.venv/bin/python -m pytest tests/test_fine_builder_diff.py -q
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

FLATTENED_SPECS = [spec for spec in MATRIX if spec.backend == "flattened"]

_determinism_gate = pytest.mark.skipif(
    os.environ.get("RAPID_FINE_DIFF", "") != "1",
    reason="fine-builder determinism sweep is a dev gate; set RAPID_FINE_DIFF=1 to run",
)


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
        # Replicate the prefill graph construction of calc_time() up to (and
        # including) _prepare_execution_graphs — no AstraSim involved.
        batch_size = tc._effective_transformer_batch()
        decode_len = tc.model.decode_len
        prefill_len = tc.seq_len - decode_len
        assert prefill_len > 0, "determinism sweep requires a prefill phase"
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


def _effective_dp(tc) -> int:
    run_type = str(getattr(getattr(tc, "model", None), "run_type", "training")).lower()
    return 1 if run_type == "inference" else max(1, getattr(tc, "dp", 1))


def _fine_bundle(dispatcher, tc, out_dir: Path):
    """The default flattened path up to (and including) ET emission."""
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
    block_templates = {"dense": dispatcher.transformer_blocks.dense}

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
    return program, bundle


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


def _bundle_files(bundle_dir: Path):
    names = sorted(p.name for p in bundle_dir.iterdir() if p.is_file())
    return [n for n in names if n.endswith(".et") or n.endswith(".json") or n.endswith(".txt")]


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


@_determinism_gate
@pytest.mark.parametrize("spec", FLATTENED_SPECS, ids=[s.spec_id for s in FLATTENED_SPECS])
def test_fine_builder_deterministic(spec, tmp_path):
    """Build twice, emit twice: op streams equal, ET bundles byte-equal."""
    hw_config, model_config, mode = _parse_spec_configs(spec, tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    for label, tc, dispatcher in _graph_cases(spec, hw_config, model_config, mode, run_dir):
        dir_a = tmp_path / f"fine_a_{_sanitize(label)}"
        dir_b = tmp_path / f"fine_b_{_sanitize(label)}"

        program_a, _bundle_a = _fine_bundle(dispatcher, tc, dir_a)
        program_b, _bundle_b = _fine_bundle(dispatcher, tc, dir_b)

        assert _describe_ops(program_a) == _describe_ops(program_b), (
            f"{spec.spec_id} [{label}]: two builds produced different op streams"
        )

        files_a = _bundle_files(dir_a)
        files_b = _bundle_files(dir_b)
        assert files_a == files_b, (
            f"{spec.spec_id} [{label}]: bundle file sets differ: {files_a} vs {files_b}"
        )
        for name in files_a:
            assert filecmp.cmp(dir_a / name, dir_b / name, shallow=False), (
                f"{spec.spec_id} [{label}]: bundle file {name} is not byte-identical "
                f"across two builds ({dir_a} vs {dir_b})"
            )
