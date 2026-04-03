from pathlib import Path
import tempfile

import pytest

from astrasim_lib import is_astrasim_available
import config
from inference_timing import TimeCalculationLLMInference
from simulate_train_graph import Edge, Graph, Node
from train_timing import MoECommDecomposition
from timing_model import CollectiveType
from train_timing import TimeCalculationLLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HW_BASE = PROJECT_ROOT / "validation_scripts" / "validation_configs" / "hardware-config" / "a100_80GB.yaml"


def _build_model_param(
    *,
    run_type: str,
    expert_imbalance_factor=None,
    num_experts: int = 4,
    global_batch_size: int = 8,
):
    model_param = {
        "mode": "LLM",
        "run_type": run_type,
        "tied_embeddings": False,
        "model_type": "llama",
        "global_batch_size": int(global_batch_size),
        "gradient_accumulation_steps": 1,
        "seq_len": 8,
        "decode_len": 4 if run_type == "inference" else None,
        "hidden_dim": 64,
        "attention": {
            "attention_type": "mha",
            "num_heads": 8,
            "kv_heads": 8,
            "use_flashattention": False,
            "attention_tile_size": 128,
        },
        "intermediate_size": 128,
        "vocab_size": 512,
        "num_layers": 2,
        "moe": {
            "num_experts": int(num_experts),
            "top_k": 1,
            "moe_intermediate_size": 32,
            "n_shared_experts": 0,
            "moe_layer_freq": 1,
            "first_k_dense_replace": 0,
        },
    }
    if run_type != "inference":
        model_param.pop("decode_len")
    if expert_imbalance_factor is not None:
        model_param["moe"]["expert_imbalance_factor"] = expert_imbalance_factor
    return model_param


def _build_hw_config(*, run_type: str, tp: int, ep: int, moe_dp: int) -> config.HWConfig:
    hw_config = config.parse_config(str(HW_BASE), config_type="hardware")
    hw_config.sch_config.tp = int(tp)
    hw_config.sch_config.cp = 1
    hw_config.sch_config.pp = 1
    hw_config.sch_config.mb = 1
    hw_config.sch_config.tp_sp = bool(tp > 1)
    hw_config.sch_config.train.dp = 1
    hw_config.sch_config.train.ep = int(ep)
    hw_config.sch_config.train.tp_ep = True
    hw_config.sch_config.inference.replica_count = 1
    hw_config.sch_config.inference.moe_dp = int(moe_dp)
    if str(run_type).lower() == "inference":
        hw_config.inference_config = hw_config.sch_config.inference
    return hw_config


def _walk_graph(root):
    seen = set()
    stack = [root]
    objects = []
    while stack:
        obj = stack.pop()
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        objects.append(obj)
        stack.extend(getattr(obj, "children", []))
    return objects


def _et_name_to_id_map(text: str):
    mapping = {}
    for line in text.splitlines():
        if not line.startswith("- id="):
            continue
        parts = line.split()
        node_id = int(parts[1].split("=")[1])
        name = parts[2].split("=", 1)[1]
        mapping[name] = node_id
    return mapping


def _et_ctrl_deps_map(text: str):
    deps = {}
    for line in text.splitlines():
        if not line.startswith("- ") or "<- [" not in line:
            continue
        lhs, rhs = line[2:].split(" <- [", 1)
        node_id = int(lhs)
        dep_ids = [int(token.strip()) for token in rhs.rstrip("]").split(",") if token.strip()]
        deps[node_id] = dep_ids
    return deps


def _find_et_node_id(mapping, prefix: str) -> int:
    matches = [node_id for name, node_id in mapping.items() if name.startswith(prefix)]
    if len(matches) != 1:
        raise KeyError(f"Expected exactly one ET node starting with '{prefix}', found {len(matches)}")
    return matches[0]


def test_moe_expert_imbalance_factor_defaults_to_balanced():
    llm_config = config.LLMConfig.from_dict(_build_model_param(run_type="training"))
    assert llm_config.expert_imbalance_factor == pytest.approx(1.0)


def test_moe_expert_imbalance_factor_rejects_values_below_one():
    with pytest.raises(ValueError, match="expert_imbalance_factor"):
        config.LLMConfig.from_dict(
            _build_model_param(run_type="training", expert_imbalance_factor=0.9)
        )


def test_moe_expert_imbalance_factor_rejects_values_above_num_experts():
    with pytest.raises(ValueError, match="cannot exceed model_param.moe.num_experts"):
        config.LLMConfig.from_dict(
            _build_model_param(run_type="training", expert_imbalance_factor=4.5, num_experts=4)
        )


def test_training_moe_expert_imbalance_factor_inflates_routed_load():
    hw_config = _build_hw_config(run_type="training", tp=1, ep=2, moe_dp=1)

    balanced_model = config.ModelConfig(
        model_config=config.LLMConfig.from_dict(_build_model_param(run_type="training")),
        inference_config=None,
    )
    imbalanced_model = config.ModelConfig(
        model_config=config.LLMConfig.from_dict(
            _build_model_param(run_type="training", expert_imbalance_factor=1.5)
        ),
        inference_config=None,
    )

    config.validate_configs(hw_config, balanced_model)
    balanced_tc = TimeCalculationLLM(hw_config, balanced_model, "LLM")
    balanced = balanced_tc._moe_routed_tokens_per_expert(
        balanced_tc._effective_transformer_batch(),
        balanced_tc.seq_len,
        allow_padding=balanced_tc._moe_allow_padding(),
    )

    hw_config_hot = _build_hw_config(run_type="training", tp=1, ep=2, moe_dp=1)
    config.validate_configs(hw_config_hot, imbalanced_model)
    hot_tc = TimeCalculationLLM(hw_config_hot, imbalanced_model, "LLM")
    hot = hot_tc._moe_routed_tokens_per_expert(
        hot_tc._effective_transformer_batch(),
        hot_tc.seq_len,
        allow_padding=hot_tc._moe_allow_padding(),
    )

    assert hot[2] > balanced[2]
    assert hot[1] > balanced[1]
    assert hot[4] > balanced[4]


def test_hot_expert_profile_derives_cold_and_hot_rank_factors():
    hw_config = _build_hw_config(run_type="training", tp=1, ep=5, moe_dp=1)
    model = config.ModelConfig(
        model_config=config.LLMConfig.from_dict(
            _build_model_param(
                run_type="training",
                expert_imbalance_factor=1.5,
                num_experts=10,
                global_batch_size=25,
            )
        ),
        inference_config=None,
    )
    config.validate_configs(hw_config, model)
    tc = TimeCalculationLLM(hw_config, model, "LLM")

    profile = tc._moe_imbalance_profile(tc.experts_per_gpu)

    assert tc.experts_per_gpu == 2
    assert profile.hot_expert_factor == pytest.approx(1.5)
    assert profile.cold_expert_factor == pytest.approx((10.0 - 1.5) / 9.0)
    assert profile.hot_rank_factor == pytest.approx((1.5 + ((10.0 - 1.5) / 9.0)) / 2.0)


def test_multi_expert_hot_rank_compute_uses_hot_and_cold_buckets():
    hw_config = _build_hw_config(run_type="training", tp=1, ep=5, moe_dp=1)
    model = config.ModelConfig(
        model_config=config.LLMConfig.from_dict(
            _build_model_param(
                run_type="training",
                expert_imbalance_factor=1.5,
                num_experts=10,
                global_batch_size=25,
            )
        ),
        inference_config=None,
    )
    config.validate_configs(hw_config, model)
    tc = TimeCalculationLLM(hw_config, model, "LLM")

    buckets = tc._moe_routed_compute_buckets(tokens_per_expert_balanced=16, experts_per_rank=tc.experts_per_gpu)

    assert buckets == [(1, 24, "hot"), (1, 15, "cold")]
    assert tc._moe_apply_expert_imbalance(tokens_local=32, experts_per_rank=tc.experts_per_gpu) == 39


def test_inference_moe_expert_imbalance_factor_inflates_decode_load():
    hw_config = _build_hw_config(run_type="inference", tp=2, ep=1, moe_dp=2)

    balanced_model_cfg = config.LLMConfig.from_dict(_build_model_param(run_type="inference"))
    inference_cfg = config.LLMInferenceConfig(sample_every=-1)
    balanced_model = config.ModelConfig(model_config=balanced_model_cfg, inference_config=inference_cfg)

    imbalanced_model_cfg = config.LLMConfig.from_dict(
        _build_model_param(run_type="inference", expert_imbalance_factor=1.5)
    )
    imbalanced_model = config.ModelConfig(model_config=imbalanced_model_cfg, inference_config=inference_cfg)

    config.validate_configs(hw_config, balanced_model)
    balanced_tc = TimeCalculationLLMInference(hw_config, balanced_model, "LLM")
    balanced = balanced_tc._moe_routed_tokens_per_expert(
        balanced_tc._effective_transformer_batch(),
        1,
        allow_padding=balanced_tc._moe_allow_padding(),
    )

    hw_config_hot = _build_hw_config(run_type="inference", tp=2, ep=1, moe_dp=2)
    config.validate_configs(hw_config_hot, imbalanced_model)
    hot_tc = TimeCalculationLLMInference(hw_config_hot, imbalanced_model, "LLM")
    hot = hot_tc._moe_routed_tokens_per_expert(
        hot_tc._effective_transformer_batch(),
        1,
        allow_padding=hot_tc._moe_allow_padding(),
    )

    assert hot[2] > balanced[2]
    assert hot[1] > balanced[1]
    assert hot[4] > balanced[4]


def test_training_moe_comm_decomposition_uses_ep_groups():
    hw_config = _build_hw_config(run_type="training", tp=1, ep=5, moe_dp=1)
    model = config.ModelConfig(
        model_config=config.LLMConfig.from_dict(
            _build_model_param(
                run_type="training",
                expert_imbalance_factor=1.5,
                num_experts=10,
                global_batch_size=25,
            )
        ),
        inference_config=None,
    )
    config.validate_configs(hw_config, model)
    tc = TimeCalculationLLM(hw_config, model, "LLM")

    decomposition = tc._moe_comm_decomposition(total_bytes=1000, experts_per_rank=tc.experts_per_gpu)

    assert decomposition.routing_mode == "ep"
    assert decomposition.routing_group == 5
    assert decomposition.total_bytes == 1000
    assert decomposition.base_all_to_all_bytes == 944
    assert decomposition.residual_total_bytes == 56
    assert decomposition.residual_sender_count == 4
    assert decomposition.residual_per_sender_bytes == 14


def test_inference_moe_comm_decomposition_uses_pooled_tp_ep_groups():
    hw_config = _build_hw_config(run_type="inference", tp=2, ep=1, moe_dp=2)
    model = config.ModelConfig(
        model_config=config.LLMConfig.from_dict(
            _build_model_param(
                run_type="inference",
                expert_imbalance_factor=1.5,
                num_experts=8,
                global_batch_size=8,
            )
        ),
        inference_config=config.LLMInferenceConfig(sample_every=-1),
    )
    config.validate_configs(hw_config, model)
    tc = TimeCalculationLLMInference(hw_config, model, "LLM")

    decomposition = tc._moe_comm_decomposition(total_bytes=1000, experts_per_rank=tc.experts_per_gpu)

    assert decomposition.routing_mode == "tp_ep"
    assert decomposition.routing_group == 4
    assert decomposition.residual_sender_count == 3
    assert decomposition.base_all_to_all_bytes == 928
    assert decomposition.residual_total_bytes == 72
    assert decomposition.residual_per_sender_bytes == 24


def test_training_moe_comm_graph_uses_local_joins_and_hot_join_fan_in_per_tp_slice():
    graph = Graph(
        mode="transformer",
        dp=1,
        pp=1,
        tp=2,
        cp=1,
        ep=3,
        comp_times={
            "transformer": {
                "gemms": [
                    {
                        "name": "moe_dispatch",
                        "forward": {
                            "duration": 0.0,
                            "comm_keys": [
                                "moe_dispatch_forward_base_all_to_all",
                                "moe_dispatch_forward_residual_p2p",
                            ],
                        },
                        "backward": {"duration": 0.0, "comm_keys": []},
                    }
                ]
            }
        },
        comm_metadata={
            "moe_dispatch_forward_base_all_to_all": {
                "size": 944,
                "type": CollectiveType.ALL_TO_ALL,
                "participants": 3,
                "interconnect_type": "ep",
                "placement": "post",
                "parallel_group": "dispatch_fwd_group",
                "moe_component": "base_all_to_all",
                "moe_routing_mode": "ep",
            },
            "moe_dispatch_forward_residual_p2p": {
                "size": 28,
                "type": CollectiveType.PIPELINE,
                "participants": 2,
                "interconnect_type": "ep",
                "placement": "post",
                "parallel_group": "dispatch_fwd_group",
                "moe_component": "residual_p2p",
                "moe_routing_mode": "ep",
            },
        },
        misc_metadata={},
    )
    root = graph.construct_transformer_graph(direction="forward")
    objects = _walk_graph(root)

    joins = [obj for obj in objects if isinstance(obj, Node) and "dispatch_fwd_group_join" in obj.name]
    residuals = [obj for obj in objects if isinstance(obj, Edge) and "residual_p2p_rank" in obj.name]

    assert len(joins) == 6
    assert sorted(node.hw_id for node in joins) == [0, 1, 2, 3, 4, 5]
    assert len(residuals) == 4
    parent_counts = {join.hw_id: len(join.parents) for join in joins}
    assert parent_counts == {0: 3, 1: 3, 2: 2, 3: 2, 4: 2, 5: 2}


def test_inference_moe_comm_graph_uses_local_joins_with_single_pooled_hot_rank():
    graph = Graph(
        mode="transformer",
        dp=1,
        pp=1,
        tp=2,
        cp=1,
        ep=2,
        comp_times={
            "transformer": {
                "gemms": [
                    {
                        "name": "moe_combine",
                        "forward": {
                            "duration": 0.0,
                            "comm_keys": [
                                "moe_combine_forward_base_all_to_all",
                                "moe_combine_forward_residual_p2p",
                            ],
                        },
                        "backward": {"duration": 0.0, "comm_keys": []},
                    }
                ]
            }
        },
        comm_metadata={
            "moe_combine_forward_base_all_to_all": {
                "size": 928,
                "type": CollectiveType.ALL_TO_ALL,
                "participants": 4,
                "interconnect_type": "ep",
                "placement": "post",
                "parallel_group": "combine_fwd_group",
                "moe_component": "base_all_to_all",
                "moe_routing_mode": "tp_ep",
            },
            "moe_combine_forward_residual_p2p": {
                "size": 24,
                "type": CollectiveType.PIPELINE,
                "participants": 2,
                "interconnect_type": "ep",
                "placement": "post",
                "parallel_group": "combine_fwd_group",
                "moe_component": "residual_p2p",
                "moe_routing_mode": "tp_ep",
            },
        },
        misc_metadata={},
    )
    root = graph.construct_transformer_graph(direction="forward")
    objects = _walk_graph(root)

    joins = [obj for obj in objects if isinstance(obj, Node) and "combine_fwd_group_join" in obj.name]
    residuals = [obj for obj in objects if isinstance(obj, Edge) and "residual_p2p_rank" in obj.name]

    assert len(joins) == 4
    assert sorted(node.hw_id for node in joins) == [0, 1, 2, 3]
    assert len(residuals) == 3
    parent_counts = {join.hw_id: len(join.parents) for join in joins}
    assert parent_counts == {0: 4, 1: 2, 2: 2, 3: 2}


def test_training_graph_build_path_registers_moe_parallel_comm_specs():
    hw_config = _build_hw_config(run_type="training", tp=2, ep=3, moe_dp=1)
    model = config.ModelConfig(
        model_config=config.LLMConfig.from_dict(
            _build_model_param(
                run_type="training",
                expert_imbalance_factor=1.5,
                num_experts=6,
                global_batch_size=36,
            )
        ),
        inference_config=None,
    )
    config.validate_configs(hw_config, model)
    tc = TimeCalculationLLM(hw_config, model, "LLM")

    tc._build_training_graphs_and_memory_data()
    objects = _walk_graph(tc.transformer_forward_root_moe)

    base_edges = [obj for obj in objects if isinstance(obj, Edge) and "base_all_to_all" in obj.name]
    residual_edges = [obj for obj in objects if isinstance(obj, Edge) and "residual_p2p" in obj.name]
    joins = [obj for obj in objects if isinstance(obj, Node) and "_join_rank" in obj.name]

    assert len(base_edges) == 12
    assert len(residual_edges) == 8
    assert len(joins) == 12


def test_astra_backed_pipeline_timing_uses_direct_point_to_point_formula():
    hw_config = config.parse_config(str(HW_BASE), config_type="hardware")
    hw_config.sch_config.tp = 2
    hw_config.sch_config.cp = 1
    hw_config.sch_config.pp = 1
    hw_config.sch_config.mb = 1
    hw_config.sch_config.tp_sp = True
    hw_config.sch_config.train.dp = 1
    hw_config.sch_config.train.ep = 1
    hw_config.sch_config.inference.replica_count = 1
    hw_config.sch_config.inference.moe_dp = 2
    hw_config.execution_backend.model = "astra"
    hw_config.execution_backend.astra.mode = "full_astrasim_hierarchical"
    hw_config.inference_config = hw_config.sch_config.inference

    model = config.ModelConfig(
        model_config=config.LLMConfig.from_dict(
            _build_model_param(
                run_type="inference",
                expert_imbalance_factor=1.5,
                num_experts=8,
                global_batch_size=16,
            )
        ),
        inference_config=config.LLMInferenceConfig(sample_every=-1),
    )
    config.validate_configs(hw_config, model)
    tc = TimeCalculationLLMInference(hw_config, model, "LLM")

    decomposition = MoECommDecomposition(
        total_bytes=196,
        base_all_to_all_bytes=0,
        residual_total_bytes=196,
        residual_per_sender_bytes=196,
        residual_sender_count=1,
        routing_group=2,
        routing_mode="tp_ep",
        hot_rank_factor=1.0,
        cold_rank_factor=1.0,
    )

    direct_time = tc._moe_comm_time(decomposition, debug_label="unit_residual_only", axis=None)

    assert direct_time > 0.0


@pytest.mark.skipif(not is_astrasim_available(), reason="AstraSim ET conversion is unavailable")
def test_training_hierarchical_et_blocks_cold_local_join_on_residual_send():
    from astrasim_lib.executor import _dump_et_text, convert_rapid_llm_graph_to_chakra_et

    graph = Graph(
        mode="transformer",
        dp=1,
        pp=1,
        tp=2,
        cp=1,
        ep=3,
        comp_times={
            "transformer": {
                "gemms": [
                    {
                        "name": "moe_dispatch",
                        "forward": {
                            "duration": 0.0,
                            "comm_keys": [
                                "moe_dispatch_forward_base_all_to_all",
                                "moe_dispatch_forward_residual_p2p",
                            ],
                        },
                        "backward": {"duration": 0.0, "comm_keys": []},
                    }
                ]
            }
        },
        comm_metadata={
            "moe_dispatch_forward_base_all_to_all": {
                "size": 944,
                "type": CollectiveType.ALL_TO_ALL,
                "participants": 3,
                "interconnect_type": "ep",
                "placement": "post",
                "parallel_group": "dispatch_fwd_group",
                "moe_component": "base_all_to_all",
                "moe_routing_mode": "ep",
            },
            "moe_dispatch_forward_residual_p2p": {
                "size": 28,
                "type": CollectiveType.PIPELINE,
                "participants": 2,
                "interconnect_type": "ep",
                "placement": "post",
                "parallel_group": "dispatch_fwd_group",
                "moe_component": "residual_p2p",
                "moe_routing_mode": "ep",
            },
        },
        misc_metadata={},
    )

    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as tmpdir:
        root = graph.construct_transformer_graph(direction="forward")
        _, rank_ids, _ = convert_rapid_llm_graph_to_chakra_et(root, dp_size=1, output_dir=tmpdir)
        et_paths = [str(Path(tmpdir) / f"llm_graph.{rank}.et") for rank in rank_ids]
        _dump_et_text(et_paths)

        cold_text = Path(tmpdir, "llm_graph.2.et.txt").read_text(encoding="utf-8")
        hot_text = Path(tmpdir, "llm_graph.0.et.txt").read_text(encoding="utf-8")

        cold_ids = _et_name_to_id_map(cold_text)
        cold_deps = _et_ctrl_deps_map(cold_text)
        hot_ids = _et_name_to_id_map(hot_text)
        hot_deps = _et_ctrl_deps_map(hot_text)

        cold_join = cold_ids["dispatch_fwd_group_join_rank2_8"]
        cold_send = cold_ids["moe_dispatch_forward_residual_p2p_rank2_to_hot0_send_dp0"]
        cold_base = _find_et_node_id(cold_ids, "moe_dispatch_forward_base_all_to_all")
        assert cold_send in cold_deps[cold_join]
        assert cold_base in cold_deps[cold_join]

        hot_join = hot_ids["dispatch_fwd_group_join_rank0_2"]
        hot_recv_rank2 = hot_ids["moe_dispatch_forward_residual_p2p_rank2_to_hot0_recv_dp0"]
        hot_recv_rank4 = hot_ids["moe_dispatch_forward_residual_p2p_rank4_to_hot0_recv_dp0"]
        hot_base = _find_et_node_id(hot_ids, "moe_dispatch_forward_base_all_to_all")
        assert hot_base in hot_deps[hot_join]
        assert hot_recv_rank2 in hot_deps[hot_join]
        assert hot_recv_rank4 in hot_deps[hot_join]


@pytest.mark.skipif(not is_astrasim_available(), reason="AstraSim ET conversion is unavailable")
def test_inference_hierarchical_et_blocks_cold_local_join_on_residual_send():
    from astrasim_lib.executor import _dump_et_text, convert_rapid_llm_graph_to_chakra_et

    graph = Graph(
        mode="transformer",
        dp=1,
        pp=1,
        tp=2,
        cp=1,
        ep=2,
        comp_times={
            "transformer": {
                "gemms": [
                    {
                        "name": "moe_combine",
                        "forward": {
                            "duration": 0.0,
                            "comm_keys": [
                                "moe_combine_forward_base_all_to_all",
                                "moe_combine_forward_residual_p2p",
                            ],
                        },
                        "backward": {"duration": 0.0, "comm_keys": []},
                    }
                ]
            }
        },
        comm_metadata={
            "moe_combine_forward_base_all_to_all": {
                "size": 928,
                "type": CollectiveType.ALL_TO_ALL,
                "participants": 4,
                "interconnect_type": "ep",
                "placement": "post",
                "parallel_group": "combine_fwd_group",
                "moe_component": "base_all_to_all",
                "moe_routing_mode": "tp_ep",
            },
            "moe_combine_forward_residual_p2p": {
                "size": 24,
                "type": CollectiveType.PIPELINE,
                "participants": 2,
                "interconnect_type": "ep",
                "placement": "post",
                "parallel_group": "combine_fwd_group",
                "moe_component": "residual_p2p",
                "moe_routing_mode": "tp_ep",
            },
        },
        misc_metadata={},
    )

    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as tmpdir:
        root = graph.construct_transformer_graph(direction="forward")
        _, rank_ids, _ = convert_rapid_llm_graph_to_chakra_et(root, dp_size=1, output_dir=tmpdir)
        et_paths = [str(Path(tmpdir) / f"llm_graph.{rank}.et") for rank in rank_ids]
        _dump_et_text(et_paths)

        cold_text = Path(tmpdir, "llm_graph.1.et.txt").read_text(encoding="utf-8")
        hot_text = Path(tmpdir, "llm_graph.0.et.txt").read_text(encoding="utf-8")

        cold_ids = _et_name_to_id_map(cold_text)
        cold_deps = _et_ctrl_deps_map(cold_text)
        hot_ids = _et_name_to_id_map(hot_text)
        hot_deps = _et_ctrl_deps_map(hot_text)

        cold_join = cold_ids["combine_fwd_group_join_rank1_5"]
        cold_send = cold_ids["moe_combine_forward_residual_p2p_rank1_to_hot0_send_dp0"]
        cold_base = _find_et_node_id(cold_ids, "moe_combine_forward_base_all_to_all")
        assert cold_send in cold_deps[cold_join]
        assert cold_base in cold_deps[cold_join]

        hot_join = hot_ids["combine_fwd_group_join_rank0_2"]
        hot_base = _find_et_node_id(hot_ids, "moe_combine_forward_base_all_to_all")
        hot_recv_rank1 = hot_ids["moe_combine_forward_residual_p2p_rank1_to_hot0_recv_dp0"]
        hot_recv_rank2 = hot_ids["moe_combine_forward_residual_p2p_rank2_to_hot0_recv_dp0"]
        hot_recv_rank3 = hot_ids["moe_combine_forward_residual_p2p_rank3_to_hot0_recv_dp0"]
        assert hot_base in hot_deps[hot_join]
        assert hot_recv_rank1 in hot_deps[hot_join]
        assert hot_recv_rank2 in hot_deps[hot_join]
        assert hot_recv_rank3 in hot_deps[hot_join]
