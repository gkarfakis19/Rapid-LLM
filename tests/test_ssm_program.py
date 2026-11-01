"""Fast schema/IR/composition evidence; archive matrix is in the slow lane."""
import copy
import json
import os
from pathlib import Path
import subprocess

import pytest
import yaml

import config
from inference_timing import TimeCalculationLLMInference
from llm_execution import LLMExecutionDispatcher
from memory_estimation import MemoryEstimator
from program.ir import CollectiveOp, ComputeOp
from program.groups import CommunicatorFactory
from program.layout import RankLayout
from ssm_gpu import local_dims

ROOT = Path(__file__).resolve().parents[1]
HW = ROOT / "validation_scripts/validation_configs/hardware-config/a100_80GB.yaml"


def model_data(kinds=None, *, moe=False):
    data = yaml.safe_load((ROOT / "configs/model-config/mamba2_base_inf.yaml").read_text())
    mp = data["model_param"]
    mp.update(global_batch_size=4, seq_len=8, decode_len=8, hidden_dim=64,
              intermediate_size=128, vocab_size=128, num_layers=4)
    mp["ssm"].update(d_state=8, n_heads=4, d_head=32, n_groups=2, chunk_size=4)
    if kinds is None:
        mp.pop("layer_plan")
        mp.pop("ssm")
        kinds = ["attention"] * 4
    else:
        mp["layer_plan"] = dict(layer_types=kinds, ffn_per_layer=True)
        if "ssm" not in kinds:
            mp.pop("ssm")
    if "attention" in kinds:
        mp["attention"] = dict(attention_type="mha", num_heads=4,
                               use_flashattention=False)
    if moe:
        mp["moe"].update(num_experts=4, top_k=2, moe_intermediate_size=128,
                         moe_layer_freq=2, first_k_dense_replace=0)
    return data


def calculator(tmp_path, kinds=None, *, tp=1, pp=1, moe=False, data=None):
    hw = config.parse_config(str(HW), "hardware")
    hw.sch_config.tp = tp
    hw.sch_config.pp = pp
    hw.sch_config.cp = 1
    hw.sch_config.mb = 1
    hw.sch_config.tp_sp = True
    hw.sch_config.inference.moe_dp = 1
    hw.execution_backend.model = "analytical"
    data = model_data(kinds, moe=moe) if data is None else data
    model = config.ModelConfig(config.LLMConfig.from_dict(data["model_param"]),
                              config.LLMInferenceConfig(sample_every=-1))
    return TimeCalculationLLMInference(hw, model, "LLM", output_dir=str(tmp_path))


def workload(tc):
    (spec, _), _ = tc.prepare_decode_graphs(batch_size=4, total_seq_len=8)
    return spec


def test_all_attention_is_existing_path_bitwise(tmp_path):
    a = calculator(tmp_path / "plain")
    b = calculator(tmp_path / "explicit", ["attention"] * 4)
    plain, explicit = workload(a), workload(b)
    assert plain.blocks == explicit.blocks
    assert plain.durations.snapshot() == explicit.durations.snapshot()
    assert LLMExecutionDispatcher(a, plain).build_flat_program_for_memory().ops == LLMExecutionDispatcher(b, explicit).build_flat_program_for_memory().ops
    for tc in (calculator(tmp_path / "a"), calculator(tmp_path / "b", ["attention"] * 4)):
        result = LLMExecutionDispatcher(tc, workload(tc)).run(tc.execution_mode)
        if tc.layer_plan is None:
            expected = result.total_time
        else:
            assert result.total_time == expected


def test_all_ssm_pattern_and_explicit_are_bitwise(tmp_path):
    data = model_data(["ssm"] * 4)
    pattern = copy.deepcopy(data)
    pattern["model_param"]["layer_plan"] = dict(pattern=["ssm"], ffn_per_layer=True)
    a = calculator(tmp_path / "a", data=data)
    b = calculator(tmp_path / "b", data=pattern)
    wa, wb = workload(a), workload(b)
    assert wa.blocks == wb.blocks
    assert wa.durations.snapshot() == wb.durations.snapshot()
    assert LLMExecutionDispatcher(a, wa).build_flat_program_for_memory().ops == LLMExecutionDispatcher(b, wb).build_flat_program_for_memory().ops
    assert LLMExecutionDispatcher(a, wa).run(a.execution_mode).total_time == LLMExecutionDispatcher(b, wb).run(b.execution_mode).total_time


@pytest.mark.parametrize("tp,pp", [(1, 1), (2, 1), (2, 2)])
def test_mixed_layers_compose_ops_memory_and_total(tmp_path, tp, pp):
    kinds = ["ssm", "attention", "ssm", "attention"]
    a = calculator(tmp_path / "a", ["attention"] * 4, tp=tp, pp=pp)
    s = calculator(tmp_path / "s", ["ssm"] * 4, tp=tp, pp=pp)
    m = calculator(tmp_path / "m", kinds, tp=tp, pp=pp)
    wa, ws, wm = workload(a), workload(s), workload(m)
    for i, kind in enumerate(kinds):
        single = wa if kind == "attention" else ws
        assert wm.block_template(i) == single.block_template(i)
        assert dict(wm.block_comm(i)) == dict(single.block_comm(i))
    memory = [MemoryEstimator(t).build_memory_data(mode="inference", batch_size=4,
              seq_len=1, kv_cache_tokens=8) for t in (a, s, m)]
    for i, kind in enumerate(kinds):
        entry = memory[2]["by_layer"][i]
        if kind == "ssm":
            assert entry == memory[1]["by_layer"][i]
        else:
            assert entry["weight_mem_per_layer"] == memory[0]["weight_mem_per_layer_dense"]
            assert entry["transient_bytes_by_kind"] == memory[0]["transient_bytes_by_kind_dense"]
            assert entry["kv_cache_bytes_per_layer"] == memory[0]["kv_cache_bytes_per_layer"]
    # One microbatch/no endpoints: the pipeline serially visits the schedule.
    result = LLMExecutionDispatcher(m, wm).run(m.execution_mode)
    expected = 2 * wa.durations.snapshot().get_or("transformer_f", 0)
    expected += 2 * ws.durations.snapshot().get_or("transformer_f", 0)
    if pp == 1:
        # Exact evaluation order is a layer-by-layer sum, not count*price.
        da = wa.durations.snapshot().get_or("transformer_f", 0)
        ds = ws.durations.snapshot().get_or("transformer_f", 0)
        assert result.total_time == sum([ds, da, ds, da])
    dispatcher = LLMExecutionDispatcher(m, wm)
    program = dispatcher.build_flat_program_for_memory()
    for i, kind in enumerate(kinds):
        names = [op.name for op in program.ops if isinstance(op, ComputeOp) and op.layer == i]
        assert any("ssm_scan" in name for name in names) == (kind == "ssm")
    MemoryEstimator(m).simulate_peak(program, memory[2], mode="inference")


@pytest.mark.parametrize("groups,tp,expected", [(8, 4, [(0,), (1,), (2,), (3,)]),
    (2, 4, [(0, 1), (0, 1), (2, 3), (2, 3)]), (1, 4, [(0, 1, 2, 3)] * 4)])
def test_rms_subgroups_are_coordinate_partitions(tmp_path, groups, tp, expected):
    tc = calculator(tmp_path, ["ssm"] * 4, tp=tp)
    # Use sufficient heads for all group cases.
    tc.ssm = config.SSMBlockConfig.from_dict(dict(variant="mamba2", expand=4,
        d_state=8, d_conv=4, n_groups=groups, n_heads=8, d_head=32, chunk_size=4))
    dims = local_dims(tc)
    factory = CommunicatorFactory(RankLayout(("tp", "pp"), {"tp": tp, "pp": 2}, {"tp": 1, "pp": tp}))
    assert [factory.tp_subgroup(rank, dims.tp_shard_factor).members for rank in range(tp)] == expected
    assert factory.tp_subgroup(tp, dims.tp_shard_factor).members == tuple(tp + i for i in expected[0])


def test_rms_and_ffn_collectives_exist_in_ir(tmp_path):
    tc = calculator(tmp_path, ["ssm"] * 4, tp=4)
    tc.ssm = config.SSMBlockConfig.from_dict(dict(variant="mamba2", expand=2,
        d_state=8, d_conv=4, n_groups=2, n_heads=4, d_head=32, chunk_size=4))
    spec = workload(tc)
    program = LLMExecutionDispatcher(tc, spec).build_flat_program_for_memory()
    stats = [op for op in program.ops if isinstance(op, CollectiveOp) and op.name == "ssm_rms_stats"]
    assert len(stats) == 16
    assert {op.group.members for op in stats} == {(0, 1), (2, 3)}
    assert {op.size_bytes for op in stats} == {8}
    assert {op.participants for op in stats} == {2}
    ffn = [op for op in program.ops if isinstance(op, CollectiveOp) and op.name == "ffn2_tp"]
    assert len(ffn) == 16
    assert {op.size_bytes for op in ffn} == {512}


def test_conv_history_and_state_are_resident_per_owned_layer(tmp_path, monkeypatch):
    from program.memory_sim import _MemorySnapshot
    seen = {}
    original = _MemorySnapshot.add_static
    def capture(self, device, size_bytes, timestamp=0.0):
        seen[device] = size_bytes
        return original(self, device, size_bytes, timestamp)
    monkeypatch.setattr(_MemorySnapshot, "add_static", capture)
    tc = calculator(tmp_path, ["ssm"] * 4, tp=2, pp=2)
    spec = workload(tc)
    memory = MemoryEstimator(tc).build_memory_data(mode="inference", batch_size=4, seq_len=1)
    d = local_dims(tc)
    layer = memory["by_layer"][0]
    assert layer["ssm_state_bytes_per_layer"] == 4 * d.d_ssm_local * 8 * 2
    assert layer["conv_state_bytes_per_layer"] == 4 * d.conv_width_local * 3 * 2
    program = LLMExecutionDispatcher(tc, spec).build_flat_program_for_memory()
    _, peak = MemoryEstimator(tc).simulate_peak(program, memory, mode="inference")
    per_device = 2 * (layer["weight_mem_per_layer"] + layer["ssm_cache_bytes_per_layer"])
    assert seen == {i: per_device + memory["extra_static_bytes_per_device"].get(i, 0.0) for i in range(4)}
    assert peak > 2 * (layer["weight_mem_per_layer"] + layer["ssm_cache_bytes_per_layer"]) / 1024**3


@pytest.mark.parametrize("kinds", [["ssm"] * 4, ["ssm", "attention", "ssm", "attention"]])
def test_memory_only_matches_inference_memory(tmp_path, kinds):
    import llm_util
    data = model_data(kinds)
    data["model_param"]["decode_len"] = 4
    tc = calculator(tmp_path / "inference", data=data)
    tc.calc_time()
    result = llm_util.estimate_inference_memory(tc.hw_config, tc._raw_model_config,
        output_dir=str(tmp_path / "memory_only"))
    assert result["max_peak_gb"] == tc.memory_peak_gb


def test_prefill_reuses_ssd_law_per_sequence(tmp_path):
    from cim_timing import ssd_chunked_scan_work
    tc = calculator(tmp_path, ["ssm"] * 4, tp=2)
    tc._prepare_layer_plan_graphs(batch_size=4, seq_len=5, total_seq_len=8)
    d = local_dims(tc)
    work = ssd_chunked_scan_work(5, d_inner=d.d_ssm_local, d_state=d.state_size,
        n_groups=d.groups_local, n_heads=d.heads_local, chunk_size=4, act_bytes=2)
    assert tc.ssm_kernel_evidence["scan_flops"] == 4 * work.ops + 2 * d.d_ssm_local * 4 * 5
    assert tc.ssm_kernel_evidence["scan_mem_bytes"] == 4 * work.state_bytes + 4 * 5 * d.d_ssm_local * tc.precision.stats
    # A one-token prefill is not a decode at the final context length.
    mixed = calculator(tmp_path / "one-token", ["ssm", "attention", "ssm", "attention"])
    mixed._prepare_layer_plan_graphs(batch_size=4, seq_len=1, total_seq_len=8)
    expected, _ = mixed.compute_all_gemm_and_node_times(4, mixed.vocab_size,
        mixed.hidden_dim, 1, mixed.num_heads, mixed.kv_heads,
        mixed.intermediate_size, mixed.hw_config.tech_config.core.num_bundles,
        use_moe_override=False)
    assert mixed.layer_plan_operation_evidence["dense"][0] == expected


def test_moe_ssm_and_attention_compose(tmp_path):
    tc = calculator(tmp_path, ["ssm", "attention", "ssm", "attention"], moe=True, tp=2)
    spec = workload(tc)
    assert spec.shape.block_kinds == ("ssm_moe", "dense", "ssm_moe", "dense")
    ops = tc.layer_plan_operation_evidence
    for key in ("ffn1", "ffn2", "gelu", "router", "moe_dispatch", "moe_combine"):
        assert ops["ssm_moe"][0][key] == ops["moe"][0][key]
    memory = MemoryEstimator(tc).build_memory_data(mode="inference", batch_size=4,
        seq_len=1, kv_cache_tokens=8)
    program = LLMExecutionDispatcher(tc, spec).build_flat_program_for_memory()
    assert LLMExecutionDispatcher(tc, spec).run(tc.execution_mode).total_time > 0
    MemoryEstimator(tc).simulate_peak(program, memory, mode="inference")


@pytest.mark.slow
@pytest.mark.parametrize("name,hw_name", [
    ("mamba2_2p7b_decode_inf", "fws_cim_mamba2_frontier"),
    ("granite_4_0_h_1b_decode_inf", "fws_cim_granite_1b_frontier"),
])
def test_same_yaml_gpu_and_mapped_cim(tmp_path, name, hw_name):
    import fws_mapping
    import fws_eval
    from program.fws_build import build_fws_program
    path = ROOT / f"configs/model-config/{name}.yaml"
    model = config.parse_config(str(path), "LLM")
    gpu = config.parse_config(str(ROOT / "configs/hardware-config/a100_80GB.yaml"), "hardware")
    gpu.sch_config.tp = 1
    gpu.sch_config.pp = 1
    gpu.sch_config.mb = 1
    gpu.execution_backend.model = "analytical"
    config.validate_configs(gpu, model)
    tc = TimeCalculationLLMInference(gpu, model, "LLM", output_dir=str(tmp_path))
    (spec, _), _ = tc.prepare_decode_graphs(batch_size=1, total_seq_len=model.model_config.seq_len)
    assert LLMExecutionDispatcher(tc, spec).run(tc.execution_mode).total_time > 0
    cim = config.parse_config(str(ROOT / f"configs/hardware-config/{hw_name}.yaml"), "hardware")
    config.validate_configs(cim, model)
    mapping = fws_mapping.build_mapping(cim, model)
    evaluation = fws_eval.evaluate_fws(build_fws_program(mapping, decode_steps=1))
    assert evaluation.timeline.total_time > 0


@pytest.mark.parametrize("kinds", [["ssm"] * 4, ["ssm", "attention", "ssm", "attention"]])
def test_one_shared_yaml_runs_both_device_laws(tmp_path, kinds):
    import fws_mapping
    import fws_eval
    from program.fws_build import build_fws_program
    data = model_data(kinds)
    data["model_param"].update(global_batch_size=1, seq_len=2, decode_len=1)
    path = tmp_path / "shared.yaml"
    path.write_text(yaml.safe_dump(data))
    model = config.parse_config(str(path), "LLM")
    gpu = config.parse_config(str(ROOT / "configs/hardware-config/a100_80GB.yaml"), "hardware")
    gpu.sch_config.tp = gpu.sch_config.pp = gpu.sch_config.mb = 1
    gpu.execution_backend.model = "analytical"
    config.validate_configs(gpu, model)
    tc = TimeCalculationLLMInference(gpu, model, "LLM", output_dir=str(tmp_path / "gpu"))
    assert tc.calc_total_inference_time()["total_inference_time"] > 0
    cim = config.parse_config(str(ROOT / "configs/hardware-config/fws_cim_mamba2_frontier.yaml"), "hardware")
    config.validate_configs(cim, model)
    result = fws_eval.evaluate_fws(build_fws_program(fws_mapping.build_mapping(cim, model), decode_steps=1))
    assert result.timeline.total_time > 0


@pytest.mark.parametrize("name,ssm_count,attn_count", [
    ("Granite-4.0-H-Small-32B_inf", 36, 4), ("Jamba2-Mini-52B-A12B_inf", 28, 4),
])
def test_named_hybrid_schedules_price_each_kind(tmp_path, name, ssm_count, attn_count):
    data = yaml.safe_load((ROOT / f"configs/model-config/{name}.yaml").read_text())
    tc = calculator(tmp_path, data=data)
    spec = workload(tc)
    kinds = spec.shape.block_kinds
    assert sum(k.startswith("ssm") for k in kinds) == ssm_count
    assert sum(k in ("dense", "moe") for k in kinds) == attn_count
    for i, kind in enumerate(kinds):
        template = spec.block_template(i)
        assert any("ssm_scan" == op.name for op in template.entries) == kind.startswith("ssm")


@pytest.mark.slow
def test_all_archive_references(tmp_path):
    subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "tools/compare_ssm_goldens.py"),
        "--output", str(tmp_path / "archive")], check=True, cwd=ROOT)
