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

"""M4 block-builder tests.

The M4 legacy differential is gone with ``construct_transformer_graph``:
before the cutover it compared, for every hybrid / hierarchical golden spec
(17 specs — incl. both ``:moe:ep2`` specs, ``fault_tp``/``fault_pp``,
``inf:hierarchical`` and the ViT spec), the legacy path (transformer graph
-> ``lower_to_program`` -> ``emit_chakra``) against the new path
(``build_block_program`` -> ``emit_chakra``) with
``program.shadow.compare_et_bundles`` on every transformer program the
dispatcher runs (dense fwd/bwd, MoE fwd/bwd) — all 17 specs matched.

What remains (the M3a pattern):

* an always-on unit test pinning the BLOCK Program shape on a synthetic
  template (per-rank serial chains, dp_count == 1, pre/post placement);
* a determinism gate (``RAPID_BLOCK_DIFF=1``): for every hybrid /
  hierarchical spec in the golden matrix, build the transformer BLOCK
  programs twice and emit twice — op streams must compare equal and the
  emitted ET bundles must be byte-identical.

Run:
    RAPID_BLOCK_DIFF=1 ./.venv/bin/python -m pytest tests/test_block_builder_diff.py -q
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

BLOCK_SPECS = [spec for spec in MATRIX if spec.backend in ("hybrid", "hierarchical")]

_diff_gate = pytest.mark.skipif(
    os.environ.get("RAPID_BLOCK_DIFF", "") != "1",
    reason="block-builder determinism sweep is a dev gate; set RAPID_BLOCK_DIFF=1 to run",
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


def _transformer_cases(spec, hw_config, model_config, mode, out_dir: Path):
    """Drive the time-calculation objects far enough to obtain the
    TransformerBlockSpec bundle(s) plus the transformer sublayout; returns
    (tc, layout_descriptor, [(label, template, degrees, direction)]).
    """
    from llm_execution import LLMExecutionDispatcher

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
        block_specs = [("", tc.transformer_blocks)]
        if tc.transformer_blocks_no_dp is not None:
            block_specs.append(("no_dp_", tc.transformer_blocks_no_dp))
    else:
        from inference_timing import TimeCalculationLLMInference

        tc = TimeCalculationLLMInference(hw_config, model_config, mode, output_dir=str(out_dir))
        batch_size = tc._effective_transformer_batch()
        decode_len = tc.model.decode_len
        prefill_len = tc.seq_len - decode_len
        assert prefill_len > 0, "block determinism sweep requires a prefill phase"
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
        moe_transformer_timings = None
        moe_node_breakdown = None
        if tc.use_moe and any(getattr(tc, "moe_layer_mask", []) or []):
            moe_transformer_timings, moe_node_breakdown = tc.compute_all_gemm_and_node_times(
                batch_size,
                tc.vocab_size,
                tc.hidden_dim,
                prefill_len,
                tc.num_heads,
                tc.kv_heads,
                tc.moe_intermediate_size,
                num_SMs,
                use_moe_override=True,
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
            moe_node_breakdown=moe_node_breakdown,
            moe_transformer_timings=moe_transformer_timings,
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
        block_specs = [("", transformer_blocks)]

    cases = []
    for prefix, blocks in block_specs:
        assert blocks is not None, f"{spec.spec_id}: missing transformer block spec"
        degrees = (blocks.tp, blocks.cp, blocks.ep)
        directions = ["forward"] + (["backward"] if blocks.include_backward else [])
        for direction in directions:
            cases.append((f"{prefix}dense_{direction}", blocks.dense, degrees, direction))
            if blocks.moe is not None:
                cases.append((f"{prefix}moe_{direction}", blocks.moe, degrees, direction))

    layout_descriptor = getattr(dispatcher, "_transformer_rank_layout", None)
    return tc, layout_descriptor, cases


def _emit_block(tc, template, degrees, direction, layout_descriptor, out_dir: Path):
    from program.block_program import build_block_program
    from program.et_emit import emit_chakra

    tp, cp, ep = degrees
    out_dir.mkdir(parents=True, exist_ok=True)
    program = build_block_program(
        template,
        direction,
        layout_descriptor,
        tp=tp,
        cp=cp,
        ep=ep,
        parallelism_mode=tc.get_parallelism_mode(),
        tp_overlap=getattr(tc, "tp_overlap", 0.0),
        tp_sp_overlap=getattr(tc, "tp_sp_overlap", 0.0),
        cp_overlap=getattr(tc, "cp_overlap", 0.0),
    )
    return program, emit_chakra(program, str(out_dir), id_policy="legacy")


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


def test_block_program_shape_synthetic():
    """Always-on pin of the BLOCK program shape: per-rank serial chains with
    pre/post comm placement, dp_count == 1, granularity tag."""
    from timing_model import CollectiveType
    from program.block import BlockTemplate
    from program.block_program import build_block_program, build_block_root
    from program.ir import CollectiveOp, ComputeOp
    from program.pipeline_fine import FineNode

    comm_metadata = {
        "layernorm1_forward_all_gather": {
            "size": 512,
            "type": CollectiveType.ALL_GATHER,
            "participants": 2,
            "interconnect_type": "tp",
            "placement": "pre",
        },
        "MLP_forward_all_reduce": {
            "size": 4096,
            "type": CollectiveType.ALL_REDUCE,
            "participants": 2,
            "interconnect_type": "tp",
            "placement": "post",
        },
    }
    gemms = [
        {
            "name": "layernorm1",
            "forward": {"duration": 1e-4, "comm_keys": ["layernorm1_forward_all_gather"]},
            "backward": {"duration": 2e-4, "comm_keys": []},
        },
        {
            "name": "MLP",
            "forward": {"duration": 3e-4, "comm_keys": ["MLP_forward_all_reduce"]},
            "backward": {"duration": 4e-4, "comm_keys": []},
        },
    ]
    template = BlockTemplate.from_gemm_entries(gemms, comm_metadata)
    program = build_block_program(template, "forward", None, tp=2, cp=1, ep=1)

    assert program.dp_count == 1
    assert program.meta.misc.get("granularity") == "block"
    assert program.devices == (0, 1)

    # Per rank: pre AG -> layernorm1 -> MLP -> post AR, serial.
    for device in program.devices:
        ops = [op for op in program.ops if getattr(op, "device", None) == device]
        names = [op.name for op in ops]
        assert names == [
            "layernorm1_forward_all_gather",
            f"layernorm1_fwd_rank{device}",
            f"MLP_fwd_rank{device}",
            "MLP_forward_all_reduce",
        ]
        assert isinstance(ops[0], CollectiveOp) and not ops[0].deps
        for prev, cur in zip(ops, ops[1:]):
            assert cur.deps == (prev.uid,)
        assert isinstance(ops[1], ComputeOp) and isinstance(ops[2], ComputeOp)

    # param_gather lives on the proto elements (memory-path attribute; the
    # lowering never carried it into ET emission).
    root = build_block_root(template, "forward", tp=2, cp=1, ep=1)
    seen = set()
    stack = list(root.children)
    gemm_nodes = []
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, FineNode):
            gemm_nodes.append(obj)
        stack.extend(getattr(obj, "children", []))
    flags = {node.name: node.param_gather for node in gemm_nodes}
    assert flags == {
        "layernorm1_fwd_rank0": True,
        "MLP_fwd_rank0": False,
        "layernorm1_fwd_rank1": True,
        "MLP_fwd_rank1": False,
    }


@_diff_gate
@pytest.mark.parametrize("spec", BLOCK_SPECS, ids=[s.spec_id for s in BLOCK_SPECS])
def test_block_builder_deterministic(spec, tmp_path):
    """Build twice, emit twice: op streams equal, ET bundles byte-equal."""
    hw_config, model_config, mode = _parse_spec_configs(spec, tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    tc, layout_descriptor, cases = _transformer_cases(spec, hw_config, model_config, mode, run_dir)
    assert cases, f"{spec.spec_id}: no transformer cases produced"

    for label, template, degrees, direction in cases:
        dir_a = tmp_path / f"block_a_{_sanitize(label)}"
        dir_b = tmp_path / f"block_b_{_sanitize(label)}"

        program_a, _bundle_a = _emit_block(tc, template, degrees, direction, layout_descriptor, dir_a)
        program_b, _bundle_b = _emit_block(tc, template, degrees, direction, layout_descriptor, dir_b)

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
