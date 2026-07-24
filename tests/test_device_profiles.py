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

"""Tests for per-device throttle profiles (device_profiles).

Covers:
  1. schema/negative-path validation and every mode gate,
  2. binary-free graph-level injection (tp2 + tp_overlap>0: op_key survives the
     overlap split; uniform == baseline; additive stage-node math; dp>1 tuples),
  3. AstraSim end-to-end equivalence (uniform all-1.0 == no profiles; uniform
     scale == globally scaled config) at <= 1 us per rank,
  4. hetero sanity (slowed device shifts the makespan and the per-device
     schedule idle) + device_metrics.json schema validation,
  5. flattened-inference uniform-profile == scaled-config equivalence,
  6. astra-cache distinctness across profiles sharing one cache file,
  7. gradient-accumulation run records.

AstraSim-dependent tests skip unless RAPID_ASTRASIM_BINARY is set (or the
repo-local submodule binary exists) AND the binary actually executes (e.g.
LD_LIBRARY_PATH provides the required libstdc++).
"""

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
HW_PROFILE_EXAMPLE = REPO_ROOT / "configs" / "hardware-config" / "a100_80GB_flat_pp2_device_profiles.yaml"
HW_ANALYTIC_THERMAL = REPO_ROOT / "configs" / "hardware-config" / "a100_80GB_legacy_thermal_port.yaml"
MODEL_TRAIN_BASE = REPO_ROOT / "validation_scripts" / "validation_configs" / "model-config" / "Llama2-7B.yaml"
MODEL_INF_BASE = REPO_ROOT / "validation_scripts" / "validation_configs" / "model-config" / "Llama2-7B_inf.yaml"
PROFILE_UNIFORM_FILE = REPO_ROOT / "configs" / "device-profiles" / "uniform_baseline.yaml"
PROFILE_HOT_FILE = REPO_ROOT / "configs" / "device-profiles" / "hot_center_example.yaml"

RANK_WALL_TOL_S = 1e-6  # <= 1 us per rank (A11)


# ---------------------------------------------------------------------------
# AstraSim availability
# ---------------------------------------------------------------------------

def _resolve_astra_binary():
    binary = os.environ.get("RAPID_ASTRASIM_BINARY")
    if binary and os.path.exists(binary):
        return binary
    local = (
        REPO_ROOT
        / "astra-sim" / "build" / "astra_analytical" / "build" / "bin"
        / "AstraSim_Analytical_Congestion_Aware"
    )
    if local.exists():
        return str(local)
    return None


def _astra_env():
    """Environment for subprocesses that must run AstraSim, or None."""
    binary = _resolve_astra_binary()
    if binary is None:
        return None
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["RAPID_ASTRASIM_BINARY"] = binary
    # Chakra protobufs live in the same astra-sim tree as the binary:
    # <astra-sim>/build/astra_analytical/build/bin/<binary>
    astra_root = Path(binary).resolve().parents[4]
    chakra_paths = [
        str(astra_root / "extern" / "graph_frontend" / "chakra" / "schema" / "protobuf"),
        str(astra_root / "extern" / "graph_frontend" / "chakra" / "src" / "third_party" / "utils"),
    ]
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(chakra_paths + ([existing] if existing else []))
    return env


def _binary_executes(env) -> bool:
    try:
        proc = subprocess.run(
            [env["RAPID_ASTRASIM_BINARY"]],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
    except Exception:
        return False
    blob = (proc.stdout or "") + (proc.stderr or "")
    return "GLIBCXX" not in blob


_ASTRA_ENV = _astra_env()
HAVE_ASTRA = _ASTRA_ENV is not None and _binary_executes(_ASTRA_ENV)
needs_astra = pytest.mark.skipif(
    not HAVE_ASTRA,
    reason="AstraSim binary unavailable (set RAPID_ASTRASIM_BINARY and LD_LIBRARY_PATH, or build the submodule)",
)


# ---------------------------------------------------------------------------
# Config factories
# ---------------------------------------------------------------------------

def _base_hw_dict():
    raw = yaml.safe_load(HW_PROFILE_EXAMPLE.read_text(encoding="utf-8"))
    raw.pop("device_profiles", None)
    return raw


def _tiny_model_dict(*, run_type="training", num_layers=4, decode_len=64, ga_steps=1):
    base_path = MODEL_INF_BASE if run_type == "inference" else MODEL_TRAIN_BASE
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    model_param = base["model_param"]
    model_param.update(
        {
            "run_type": run_type,
            "model_type": "llama",
            "global_batch_size": 8 * ga_steps,
            "gradient_accumulation_steps": ga_steps,
            "seq_len": 192 if run_type == "inference" else 128,
            "hidden_dim": 512,
            "intermediate_size": 1024,
            "vocab_size": 1024,
            "num_layers": num_layers,
        }
    )
    model_param["attention"].update(
        {
            "attention_type": "gqa",
            "num_heads": 8,
            "head_dim": 64,
            "kv_heads": 4,
            "use_flashattention": False,
            "attention_tile_size": 128,
        }
    )
    if run_type == "inference":
        model_param["decode_len"] = decode_len
        base.setdefault("inference_param", {})["sample_every"] = 32
    return base


def _write_yaml(tmp_path: Path, name: str, payload) -> Path:
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle)
    return path


def _hw_with(tmp_path, name, *, device_profiles=None, parallelism=None, freq_mult=None,
             hbm_bw_mult=None, l2_bw_mult=None, network_patch=None, sw_patch=None):
    raw = _base_hw_dict()
    if parallelism:
        raw["parallelism"].update(parallelism)
    if device_profiles is not None:
        raw["device_profiles"] = device_profiles
    if freq_mult is not None:
        raw["tech_param"]["core"]["operating_frequency"] = float(
            raw["tech_param"]["core"]["operating_frequency"]
        ) * freq_mult
    if hbm_bw_mult is not None:
        # base config uses '1986 GB'; scale numerically in bytes
        raw["tech_param"]["DRAM"]["bandwidth"] = float(1986 * (1024 ** 3) * hbm_bw_mult)
    if l2_bw_mult is not None:
        # base config uses '7050 GB'; scale numerically in bytes
        raw["tech_param"]["SRAM-L2"]["bandwidth"] = float(7050 * (1024 ** 3) * l2_bw_mult)
    if network_patch:
        raw["network"]["dimensions"][0].update(network_patch)
    if sw_patch:
        raw["sw_param"].update(sw_patch)
    return _write_yaml(tmp_path, name, raw)


def _run_perf(tmp_dir: Path, hw_config: Path, model_config: Path, *, env=None,
              device_profiles: Path = None, expect_failure=False):
    run_env = dict(env or os.environ)
    run_env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "run_perf.py"),
        "--hardware_config", str(hw_config),
        "--model_config", str(model_config),
    ]
    if device_profiles is not None:
        cmd += ["--device_profiles", str(device_profiles)]
    result = subprocess.run(
        cmd, cwd=str(tmp_dir), env=run_env, capture_output=True, text=True, timeout=560
    )
    if expect_failure:
        assert result.returncode != 0, f"run_perf unexpectedly succeeded:\n{result.stdout[-2000:]}"
        return result
    assert result.returncode == 0, (
        f"run_perf failed (rc={result.returncode})\n"
        f"stdout tail:\n{result.stdout[-3000:]}\nstderr tail:\n{result.stderr[-3000:]}"
    )
    return result


def _load_metrics(tmp_dir: Path) -> dict:
    path = tmp_dir / "output" / "LLM" / "device_metrics.json"
    assert path.exists(), f"device_metrics.json missing under {tmp_dir}"
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Schema / negative paths
# ---------------------------------------------------------------------------

class TestSchema:
    def _parse(self, raw):
        import config

        return config.parse_device_profiles(raw)

    def test_sample_configs_parse(self):
        import config
        from device_profiles import load_device_profiles_yaml

        hw = config.parse_config(str(HW_PROFILE_EXAMPLE), "hardware")
        assert hw.device_profiles is not None
        assert hw.device_profiles.resolve(0) == "hot"
        assert hw.device_profiles.resolve(7) == "nominal"
        for path in (PROFILE_UNIFORM_FILE, PROFILE_HOT_FILE):
            parsed = load_device_profiles_yaml(str(path))
            assert parsed.profiles

    def test_absent_block_is_none(self):
        assert self._parse(None) is None
        assert self._parse({}) is None

    @pytest.mark.parametrize(
        "raw, fragment",
        [
            ("not-a-mapping", "must be a mapping"),
            ({"profiles": {}, "devices": {"default": "p"}}, "non-empty mapping"),
            ({"profiles": {"p": {"bogus": 1.0}}, "devices": {"default": "p"}}, "unknown field"),
            ({"profiles": {"p": {"frequency_scale": 0.0}}, "devices": {"default": "p"}}, "> 0"),
            ({"profiles": {"p": {"frequency_scale": -1.0}}, "devices": {"default": "p"}}, "> 0"),
            ({"profiles": {"p": {"hbm_bandwidth_scale": float("nan")}}, "devices": {"default": "p"}}, "finite"),
            ({"profiles": {"p": {"frequency_scale": "hotplate"}}, "devices": {"default": "p"}}, "must be a number"),
            ({"profiles": {"p": {}}, "devices": {"default": "q"}}, "unknown profile"),
            ({"profiles": {"p": {}}, "devices": {"x": "p"}}, "integer hw_id"),
            ({"profiles": {"p": {}}, "devices": {-1: "p"}}, ">= 0"),
            ({"profiles": {"p": {}}, "devices": {}}, "at least one hw_id"),
            ({"profiles": {"p": {}}, "devices": {"default": "p"}, "dp_devices": {"zzz": "p"}}, "<hw_id>,<dp_idx>"),
            ({"profiles": {"p": {}}, "devices": {"default": "p"}, "dp_devices": {"0,1": "q"}}, "unknown profile"),
            ({"profiles": {"p": {}}, "devices": {"default": "p"}, "unknown_section": {}}, "unknown section"),
        ],
    )
    def test_malformed_blocks_raise(self, raw, fragment):
        from config import DeviceProfileError

        with pytest.raises(DeviceProfileError) as excinfo:
            self._parse(raw)
        assert fragment in str(excinfo.value)

    def test_loader_rejects_empty_file(self, tmp_path):
        from config import DeviceProfileError
        from device_profiles import load_device_profiles_yaml

        empty = tmp_path / "empty.yaml"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(DeviceProfileError):
            load_device_profiles_yaml(str(empty))

    def test_resolution_order(self):
        parsed = self._parse(
            {
                "profiles": {"a": {}, "b": {"frequency_scale": 0.5}, "c": {"frequency_scale": 0.9}},
                "devices": {0: "a", "default": "c"},
                "dp_devices": {"0,1": "b"},
            }
        )
        assert parsed.resolve(0, 0) == "a"
        assert parsed.resolve(0, 1) == "b"  # dp_devices beats devices
        assert parsed.resolve(3, 0) == "c"  # default fallback


# ---------------------------------------------------------------------------
# Mode gates (no binary)
# ---------------------------------------------------------------------------

UNIFORM_BLOCK = {"profiles": {"nominal": {}}, "devices": {"default": "nominal"}}
HOT_ALL_BLOCK = {
    "profiles": {"hot": {"frequency_scale": 0.7, "hbm_bandwidth_scale": 0.85}},
    "devices": {"default": "hot"},
}


def _make_time_calc(hw_path, model_path, out_dir):
    import config
    from train_timing import TimeCalculationLLM

    hw = config.parse_config(str(hw_path), "hardware")
    model = config.parse_config(str(model_path), "LLM")
    config.validate_configs(hw, model)
    return TimeCalculationLLM(hw, model, "LLM", output_dir=str(out_dir))


class TestModeGates:
    def test_analytical_mode_rejected(self, tmp_path):
        import config
        from train_timing import TimeCalculationLLM

        hw = config.parse_config(str(HW_ANALYTIC_THERMAL), "hardware")
        hw.device_profiles = config.parse_device_profiles(UNIFORM_BLOCK)
        model_path = _write_yaml(tmp_path, "model.yaml", _tiny_model_dict())
        model = config.parse_config(str(model_path), "LLM")
        with pytest.raises(ValueError, match="full_astrasim_flattened"):
            TimeCalculationLLM(hw, model, "LLM", output_dir=str(tmp_path / "out"))

    def test_hybrid_mode_rejected(self, tmp_path):
        raw = _base_hw_dict()
        raw["execution_backend"]["astra"]["mode"] = "hybrid"
        raw["device_profiles"] = UNIFORM_BLOCK
        hw_path = _write_yaml(tmp_path, "hw.yaml", raw)
        model_path = _write_yaml(tmp_path, "model.yaml", _tiny_model_dict())
        with pytest.raises(ValueError, match="full_astrasim_flattened"):
            _make_time_calc(hw_path, model_path, tmp_path / "out")

    def test_dp_devices_requires_dp_gt_1(self, tmp_path):
        block = dict(UNIFORM_BLOCK)
        block = {**UNIFORM_BLOCK, "dp_devices": {"0,0": "nominal"}}
        hw_path = _hw_with(tmp_path, "hw.yaml", device_profiles=block)
        model_path = _write_yaml(tmp_path, "model.yaml", _tiny_model_dict())
        with pytest.raises(ValueError, match="dp > 1"):
            _make_time_calc(hw_path, model_path, tmp_path / "out")

    def test_dp_devices_rejected_for_inference(self, tmp_path):
        block = {**UNIFORM_BLOCK, "dp_devices": {"0,0": "nominal"}}
        hw_path = _hw_with(tmp_path, "hw.yaml", device_profiles=block)
        model_path = _write_yaml(tmp_path, "model.yaml", _tiny_model_dict(run_type="inference"))
        import config
        from inference_timing import TimeCalculationLLMInference

        hw = config.parse_config(str(hw_path), "hardware")
        model = config.parse_config(str(model_path), "LLM")
        with pytest.raises(ValueError, match="training-only"):
            TimeCalculationLLMInference(hw, model, "LLM", output_dir=str(tmp_path / "out"))

    def test_gemm_mode_rejected(self, tmp_path):
        import run_perf

        hw_path = _hw_with(tmp_path, "hw.yaml", device_profiles=UNIFORM_BLOCK)
        with pytest.raises(ValueError, match="GEMM"):
            run_perf.run_GEMM(
                exp_hw_config_path=str(hw_path),
                exp_model_config_path=str(REPO_ROOT / "configs" / "model-config" / "GEMM.yaml"),
                exp_dir=str(tmp_path / "out"),
                mode="GEMM",
            )

    def test_vit_mode_rejected(self, tmp_path):
        import run_perf

        hw_path = _hw_with(tmp_path, "hw.yaml", device_profiles=UNIFORM_BLOCK)
        with pytest.raises(ValueError, match="VIT|ViT"):
            run_perf.run_LLM(
                exp_hw_config_path=str(hw_path),
                exp_model_config_path=str(
                    REPO_ROOT / "configs" / "model-config" / "vit_base_inf.yaml"
                ),
                exp_dir=str(tmp_path / "out"),
                mode="VIT",
            )

    def test_moe_rejected(self, tmp_path):
        import config

        hw_path = _hw_with(tmp_path, "hw.yaml", device_profiles=UNIFORM_BLOCK)
        model_dict = _tiny_model_dict()
        model_dict["model_param"]["moe"] = {
            "num_experts": 4,
            "top_k": 1,
            "moe_intermediate_size": 1024,
            "n_shared_experts": 0,
            "moe_layer_freq": 1,
            "first_k_dense_replace": 0,
        }
        model_path = _write_yaml(tmp_path, "model.yaml", model_dict)
        hw = config.parse_config(str(hw_path), "hardware")
        model = config.parse_config(str(model_path), "LLM")
        # MoE + flattened is rejected at config validation, before profiles even
        # come into play (pre-feature gate, still worth pinning) ...
        with pytest.raises(NotImplementedError, match="MoE"):
            config.validate_configs(hw, model)

    def test_moe_rejected_by_dispatcher_gate(self, tmp_path, monkeypatch):
        """The DISPATCHER profiles+MoE gate (A11) fires for programmatic construction.

        The config-level rejection above predates device profiles, so it cannot
        cover the profiles-specific gate: build a dense flattened tc WITH
        profiles, then present it as MoE to the dispatcher. The error must name
        device_profiles (not the generic flattened-MoE rejection).
        """
        _stub_collectives(monkeypatch)
        hw_path = _hw_with(tmp_path, "hw.yaml", device_profiles=UNIFORM_BLOCK)
        model_path = _write_yaml(tmp_path, "model.yaml", _tiny_model_dict())
        tc = _make_time_calc(hw_path, model_path, tmp_path / "out")
        tc._build_training_graphs_and_memory_data()
        from llm_execution import LLMExecutionDispatcher

        def _dispatch(**extra):
            return LLMExecutionDispatcher(
                time_calc=tc,
                pipeline_graph=tc.pipeline_graph,
                pipeline_root=tc.pipeline_root,
                interconnect_params=tc.pipeline_interconnect,
                transformer_graph=tc.transformer_graph,
                transformer_forward_root=tc.transformer_forward_root,
                transformer_backward_root=tc.transformer_backward_root,
                **extra,
            )

        # use_moe branch of the gate.
        tc.use_moe = True
        with pytest.raises(ValueError, match="device_profiles do not support MoE"):
            _dispatch()
        # moe_transformer_graph branch of the same gate.
        tc.use_moe = False
        with pytest.raises(ValueError, match="device_profiles do not support MoE"):
            _dispatch(
                moe_transformer_graph=tc.transformer_graph,
                moe_transformer_forward_root=tc.transformer_forward_root,
                moe_transformer_backward_root=tc.transformer_backward_root,
            )
        # Sanity: without MoE markers the same construction succeeds.
        _dispatch()

    def test_optimize_2dmap_rejected(self, tmp_path, monkeypatch):
        import base_timing

        monkeypatch.setattr(
            base_timing.NetworkModel,
            "_astra_collective",
            lambda self, kind, part, size, axis=None: 1e-6,
        )
        hw_path = _hw_with(
            tmp_path,
            "hw.yaml",
            device_profiles=UNIFORM_BLOCK,
            parallelism={"tp": 2, "tp_sp": True},
            network_patch={
                "size": 4,
                "topology": {
                    "type": "Mesh2D",
                    "bandwidth": "400 Gb",
                    "latency": 5e-6,
                    "energy_per_bit": 8e-12,
                    "util": 1.0,
                    "optimize_2dmap": True,
                },
            },
        )
        model_path = _write_yaml(tmp_path, "model.yaml", _tiny_model_dict())
        tc = _make_time_calc(hw_path, model_path, tmp_path / "out")
        tc._build_training_graphs_and_memory_data()
        from llm_execution import LLMExecutionDispatcher

        with pytest.raises(ValueError, match="optimize_2dmap"):
            LLMExecutionDispatcher(
                time_calc=tc,
                pipeline_graph=tc.pipeline_graph,
                pipeline_root=tc.pipeline_root,
                interconnect_params=tc.pipeline_interconnect,
                transformer_graph=tc.transformer_graph,
                transformer_forward_root=tc.transformer_forward_root,
                transformer_backward_root=tc.transformer_backward_root,
            )

    def test_cli_override_replaces_block(self, tmp_path):
        import config
        import run_perf

        hw = config.parse_config(str(HW_PROFILE_EXAMPLE), "hardware")
        assert "hot" in hw.device_profiles.profiles
        run_perf._apply_device_profiles_override(hw, str(PROFILE_UNIFORM_FILE))
        assert sorted(hw.device_profiles.profiles) == ["nominal"]
        assert hw.device_profiles.default_profile == "nominal"


# ---------------------------------------------------------------------------
# 2. Graph-level injection (binary-free; A2)
# ---------------------------------------------------------------------------

def _stub_collectives(monkeypatch):
    import base_timing

    monkeypatch.setattr(
        base_timing.NetworkModel,
        "_astra_collective",
        lambda self, kind, part, size, axis=None: 1e-6,
    )


def _build_injected_root(hw_path, model_path, out_dir):
    from llm_execution import LLMExecutionDispatcher

    tc = _make_time_calc(hw_path, model_path, out_dir)
    tc._build_training_graphs_and_memory_data()
    dispatcher = LLMExecutionDispatcher(
        time_calc=tc,
        pipeline_graph=tc.pipeline_graph,
        pipeline_root=tc.pipeline_root,
        interconnect_params=tc.pipeline_interconnect,
        transformer_graph=tc.transformer_graph,
        transformer_forward_root=tc.transformer_forward_root,
        transformer_backward_root=tc.transformer_backward_root,
    )
    root, hw_ids, flattener, effective_dp = dispatcher._build_and_inject_flattened_root()
    return tc, root, hw_ids, flattener, effective_dp


def _collect_compute_nodes(root):
    """All nonzero-duration compute nodes, sorted deterministically."""
    from device_profiles import _iter_graph_nodes

    nodes = [
        node
        for node in _iter_graph_nodes(root)
        if getattr(node, "hw_id", None) is not None
        and node.hw_id >= 0
        and float(node.duration or 0.0) > 0.0
    ]
    return sorted(nodes, key=lambda n: (str(n.name), int(getattr(n, "op_id", 0))))


TP2_PARALLELISM = {"tp": 2, "tp_sp": True, "pp": 2, "mb": 2}


class TestGraphLevelInjection:
    def test_tp2_overlap_all_nodes_scaled(self, tmp_path, monkeypatch):
        """tp2 + tp_sp_overlap>0: heads AND tails of split nodes are re-priced."""
        _stub_collectives(monkeypatch)
        model_path = _write_yaml(tmp_path, "model.yaml", _tiny_model_dict())
        hw_base = _hw_with(tmp_path, "hw_base.yaml", parallelism=TP2_PARALLELISM)
        hw_hot = _hw_with(
            tmp_path, "hw_hot.yaml", parallelism=TP2_PARALLELISM, device_profiles=HOT_ALL_BLOCK
        )

        _, base_root, base_hw_ids, _, _ = _build_injected_root(hw_base, model_path, tmp_path / "b")
        tc_hot, hot_root, hot_hw_ids, _, _ = _build_injected_root(hw_hot, model_path, tmp_path / "h")

        assert sorted(base_hw_ids) == sorted(hot_hw_ids) == [0, 1, 2, 3]
        base_nodes = _collect_compute_nodes(base_root)
        hot_nodes = _collect_compute_nodes(hot_root)
        assert len(base_nodes) == len(hot_nodes) > 0

        head_nodes = [n for n in hot_nodes if str(n.name).endswith("_head")]
        assert head_nodes, "tp_sp_overlap > 0 must produce _head split nodes"

        # Every nonzero compute node carries op_key post-transform (A2).
        for node in hot_nodes:
            assert getattr(node, "op_key", None) is not None, f"missing op_key on {node.name}"

        bank = tc_hot.get_device_profile_bank(build=False)
        assert bank is not None and "hot" in bank.entries

        scaled_up = 0
        for base_node, hot_node in zip(base_nodes, hot_nodes):
            assert str(base_node.name) == str(hot_node.name)
            base_d = float(base_node.duration)
            hot_d = float(hot_node.duration)
            op_name = hot_node.op_key[0]
            if op_name in ("embedding", "linear_softmax", "optimizer"):
                continue  # additive ops checked separately
            assert hot_d >= base_d * (1.0 - 1e-12), (
                f"node {hot_node.name} ({hot_node.op_key}) got FASTER under the hot profile"
            )
            if hot_d > base_d * (1.0 + 1e-9):
                scaled_up += 1
        # The hot profile throttles freq and HBM BW: virtually every GEMM slows.
        assert scaled_up >= 0.9 * len([n for n in hot_nodes if n.op_key[0] not in ("embedding", "linear_softmax", "optimizer")])

        # Heads specifically must be scaled (the Dec-2025 failure mode).
        base_by_key = {}
        for node in base_nodes:
            base_by_key.setdefault(str(node.name), []).append(float(node.duration))
        for head in head_nodes:
            baselines = base_by_key.get(str(head.name))
            assert baselines, f"head node {head.name} missing from baseline graph"
            assert float(head.duration) > min(baselines) * (1.0 + 1e-9), (
                f"head node {head.name} was not re-priced"
            )

    def test_uniform_profile_matches_baseline_graph(self, tmp_path, monkeypatch):
        """All-1.0 profiles leave every duration bit-identical (graph level)."""
        _stub_collectives(monkeypatch)
        model_path = _write_yaml(tmp_path, "model.yaml", _tiny_model_dict())
        hw_base = _hw_with(tmp_path, "hw_base.yaml", parallelism=TP2_PARALLELISM)
        hw_unif = _hw_with(
            tmp_path, "hw_unif.yaml", parallelism=TP2_PARALLELISM, device_profiles=UNIFORM_BLOCK
        )
        _, base_root, _, _, _ = _build_injected_root(hw_base, model_path, tmp_path / "b")
        _, unif_root, _, _, _ = _build_injected_root(hw_unif, model_path, tmp_path / "u")
        base_nodes = _collect_compute_nodes(base_root)
        unif_nodes = _collect_compute_nodes(unif_root)
        assert len(base_nodes) == len(unif_nodes)
        for base_node, unif_node in zip(base_nodes, unif_nodes):
            assert str(base_node.name) == str(unif_node.name)
            assert float(base_node.duration) == float(unif_node.duration)

    def test_additive_stage_node_math(self, tmp_path, monkeypatch):
        """embedding/linear_softmax/optimizer: new = base + (compute_k - compute_base)."""
        _stub_collectives(monkeypatch)
        model_path = _write_yaml(tmp_path, "model.yaml", _tiny_model_dict())
        hw_base = _hw_with(tmp_path, "hw_base.yaml", parallelism=TP2_PARALLELISM)
        hw_hot = _hw_with(
            tmp_path, "hw_hot.yaml", parallelism=TP2_PARALLELISM, device_profiles=HOT_ALL_BLOCK
        )
        _, base_root, _, _, _ = _build_injected_root(hw_base, model_path, tmp_path / "b")
        tc_hot, hot_root, _, _, _ = _build_injected_root(hw_hot, model_path, tmp_path / "h")
        bank = tc_hot.get_device_profile_bank(build=False)
        entry = bank.entries["hot"]
        baseline = bank.baseline

        base_nodes = _collect_compute_nodes(base_root)
        hot_nodes = _collect_compute_nodes(hot_root)
        checked = {"embedding": 0, "linear_softmax": 0, "optimizer": 0}
        for base_node, hot_node in zip(base_nodes, hot_nodes):
            op_key = getattr(hot_node, "op_key", None)
            if op_key is None or op_key[0] not in checked:
                continue
            base_d = float(base_node.duration)
            hot_d = float(hot_node.duration)
            if op_key[0] == "optimizer":
                delta = entry.optimizer_time_s - baseline.optimizer_time_s
            else:
                delta = entry.ops[tuple(op_key)][0] - baseline.ops[tuple(op_key)][0]
            assert hot_d == pytest.approx(base_d + delta, rel=1e-12, abs=1e-15), (
                f"additive math violated for {hot_node.name} {op_key}"
            )
            checked[op_key[0]] += 1
        assert all(count > 0 for count in checked.values()), checked

    def test_dp2_duration_tuples(self, tmp_path, monkeypatch):
        """dp>1: per-node tuples of length dp; dp_devices targets one replica."""
        _stub_collectives(monkeypatch)
        model_path = _write_yaml(tmp_path, "model.yaml", _tiny_model_dict())
        block = {
            "profiles": {"hot": {"frequency_scale": 0.7, "hbm_bandwidth_scale": 0.85}, "nominal": {}},
            "devices": {"default": "nominal"},
            "dp_devices": {"0,1": "hot"},
        }
        hw_path = _hw_with(
            tmp_path,
            "hw.yaml",
            parallelism={"pp": 2, "mb": 2, "train": {"dp": 2, "ep": 1, "tp_ep": True}},
            device_profiles=block,
        )
        _, root, hw_ids, _, effective_dp = _build_injected_root(hw_path, model_path, tmp_path / "d")
        assert effective_dp == 2
        from device_profiles import _iter_graph_nodes

        tuple_nodes = 0
        for node in _iter_graph_nodes(root):
            if getattr(node, "hw_id", None) is None or node.hw_id < 0:
                continue
            if node.duration_profile is None:
                continue
            profile = node.duration_profile
            assert len(profile) == 2
            tuple_nodes += 1
            if node.hw_id == 0:
                # dp replica 1 on hw 0 is throttled; additive ops may share the
                # delta sign, multiplicative ops must strictly slow down.
                assert profile[1] >= profile[0]
            else:
                assert profile[0] == profile[1]
        assert tuple_nodes > 0
        hot_seen = any(
            node.hw_id == 0 and node.duration_profile and node.duration_profile[1] > node.duration_profile[0]
            for node in _iter_graph_nodes(root)
            if getattr(node, "hw_id", None) is not None and node.hw_id >= 0 and node.duration_profile
        )
        assert hot_seen, "dp_devices '0,1' produced no throttled durations on hw 0 / dp 1"

    def test_unresolved_hw_id_raises(self, tmp_path, monkeypatch):
        _stub_collectives(monkeypatch)
        from config import DeviceProfileError

        model_path = _write_yaml(tmp_path, "model.yaml", _tiny_model_dict())
        block = {"profiles": {"hot": {"frequency_scale": 0.7}}, "devices": {0: "hot"}}  # no default
        hw_path = _hw_with(tmp_path, "hw.yaml", device_profiles=block)
        with pytest.raises(DeviceProfileError, match="hw_id=1"):
            _build_injected_root(hw_path, model_path, tmp_path / "x")

    def test_device_key_outside_graph_raises(self, tmp_path, monkeypatch):
        _stub_collectives(monkeypatch)
        from config import DeviceProfileError

        model_path = _write_yaml(tmp_path, "model.yaml", _tiny_model_dict())
        block = {
            "profiles": {"hot": {"frequency_scale": 0.7}},
            "devices": {0: "hot", 5: "hot", "default": "hot"},
        }
        hw_path = _hw_with(tmp_path, "hw.yaml", device_profiles=block)
        with pytest.raises(DeviceProfileError, match=r"\[5\]"):
            _build_injected_root(hw_path, model_path, tmp_path / "x")


# ---------------------------------------------------------------------------
# device_metrics.json schema validator
# ---------------------------------------------------------------------------

def _validate_device_metrics_schema(metrics: dict, *, expect_num_layers=None,
                                    expect_pp_only=False):
    """Validate device_metrics.json.

    Beyond presence/type checks, this asserts VALUES:
    - makespan/per-device busy+wall consistency with the weighted `runs` records,
    - hosts_lm_head on exactly ONE device per dp replica,
    - with `expect_num_layers`: layers_hosted matches the front-loaded
      `_stage_for_layer` partition (base + 1 for the first `remainder` stages),
    - with `expect_pp_only` (pp-and-dp-only topologies, tp=cp=ep=1): exact axis
      coordinates per rank (pp == hw_id, dp == dp_idx, other axes 0).
    """
    assert metrics["schema_version"] == 1
    assert metrics["execution_mode"] == "full_astrasim_flattened"
    assert metrics["run_type"] in ("training", "inference")
    assert isinstance(metrics["dp_count"], int) and metrics["dp_count"] >= 1
    assert isinstance(metrics["num_devices"], int)
    assert metrics["num_devices"] == len(metrics["devices"]) > 0
    assert metrics["makespan_s"] > 0.0
    assert metrics["pipeline_interleave_scale"] > 0.0
    assert isinstance(metrics["profiles"], dict)
    assert isinstance(metrics["runs"], list) and metrics["runs"]

    # makespan is exactly the weighted sum of the per-run raw totals.
    expected_makespan = sum(
        float(run["weight"]) * float(run["total_raw_s"]) for run in metrics["runs"]
    )
    assert metrics["makespan_s"] == pytest.approx(expected_makespan, rel=1e-12)

    ranks = set()
    for device in metrics["devices"]:
        for key in (
            "rank", "hw_id", "dp_idx", "coords", "profile", "compute_busy_s",
            "wall_time_s", "sched_idle_frac", "layers_hosted", "hosts_lm_head",
            "kernel_idle_layer_s", "kernel_idle_global_s", "kernel_idle_frac_thermal",
        ):
            assert key in device, f"device entry missing {key}"
        assert device["rank"] not in ranks
        ranks.add(device["rank"])
        assert 0.0 <= device["sched_idle_frac"] <= 1.0
        assert device["compute_busy_s"] >= 0.0
        assert device["wall_time_s"] >= 0.0
        assert device["kernel_idle_layer_s"] >= 0.0
        assert device["kernel_idle_global_s"] >= 0.0
        assert 0.0 <= device["kernel_idle_frac_thermal"] <= 1.0
        assert isinstance(device["coords"], dict) and "dp" in device["coords"]
        if metrics["profiles"]:
            assert device["profile"] in metrics["profiles"]

        # Combined busy/wall are exactly the weighted per-run values.
        expected_busy = 0.0
        expected_wall = 0.0
        for run in metrics["runs"]:
            per_rank = {d["rank"]: d for d in run["devices"]}
            assert device["rank"] in per_rank, (
                f"run '{run['name']}' missing rank {device['rank']}"
            )
            expected_busy += float(run["weight"]) * per_rank[device["rank"]]["compute_busy_s"]
            expected_wall += float(run["weight"]) * per_rank[device["rank"]]["wall_time_s"]
        assert device["compute_busy_s"] == pytest.approx(expected_busy, rel=1e-12, abs=1e-15)
        assert device["wall_time_s"] == pytest.approx(expected_wall, rel=1e-12, abs=1e-15)

    # The lm head lands on exactly one hw_id — one device per dp replica (A8).
    # (dp>1 flattened runs carry per-dp duration tuples on shared ranks, so the
    # replica set is taken from the emitted dp_idx values, not dp_count.)
    lm_hosts = [d for d in metrics["devices"] if d["hosts_lm_head"]]
    assert len({d["hw_id"] for d in lm_hosts}) == 1
    for dp_idx in sorted({d["dp_idx"] for d in metrics["devices"]}):
        replica_hosts = [d for d in lm_hosts if d["dp_idx"] == dp_idx]
        assert len(replica_hosts) == 1, (
            f"dp replica {dp_idx} must host the lm head on exactly one device, "
            f"got {len(replica_hosts)}"
        )

    if expect_num_layers is not None:
        # layers_hosted follows the front-loaded _stage_for_layer partition (A10).
        stages = sorted({d["hw_id"] for d in metrics["devices"]})
        pp = len(stages)
        base, remainder = divmod(int(expect_num_layers), pp)
        expected_per_stage = [base + (1 if s < remainder else 0) for s in range(pp)]
        assert sum(expected_per_stage) == expect_num_layers
        for device in metrics["devices"]:
            stage = int(device["coords"].get("pp", 0))
            assert device["layers_hosted"] == expected_per_stage[stage], (
                f"rank {device['rank']} (pp={stage}) hosts {device['layers_hosted']} "
                f"layers, expected {expected_per_stage[stage]} of {expected_per_stage}"
            )

    if expect_pp_only:
        # pp/dp-only topology: hw_id IS the pp coordinate; all other axes are 0.
        for device in metrics["devices"]:
            coords = device["coords"]
            assert coords["dp"] == device["dp_idx"]
            assert coords.get("pp", 0) == device["hw_id"], (
                f"rank {device['rank']}: pp coord {coords.get('pp')} != hw_id "
                f"{device['hw_id']}"
            )
            for axis, value in coords.items():
                if axis not in ("pp", "dp"):
                    assert value == 0, f"axis {axis} expected 0, got {value}"


# ---------------------------------------------------------------------------
# 3-5, 7. AstraSim end-to-end tests
# ---------------------------------------------------------------------------

@needs_astra
class TestAstraEquivalence:
    def _run(self, tmp_path, label, hw_path, model_path):
        run_dir = tmp_path / label
        run_dir.mkdir()
        _run_perf(run_dir, hw_path, model_path, env=_ASTRA_ENV)
        return run_dir

    def test_uniform_profiles_equal_no_profiles(self, tmp_path):
        # num_layers=5 on pp2 exercises the front-loaded remainder ([3, 2]).
        model_path = _write_yaml(tmp_path, "model.yaml", _tiny_model_dict(num_layers=5))
        hw_base = _hw_with(tmp_path, "hw_base.yaml")
        hw_unif = _hw_with(tmp_path, "hw_unif.yaml", device_profiles=UNIFORM_BLOCK)
        base_dir = self._run(tmp_path, "base", hw_base, model_path)
        unif_dir = self._run(tmp_path, "unif", hw_unif, model_path)
        base_metrics = _load_metrics(base_dir)
        unif_metrics = _load_metrics(unif_dir)
        _validate_device_metrics_schema(base_metrics, expect_num_layers=5, expect_pp_only=True)
        _validate_device_metrics_schema(unif_metrics, expect_num_layers=5, expect_pp_only=True)
        assert abs(base_metrics["makespan_s"] - unif_metrics["makespan_s"]) <= RANK_WALL_TOL_S
        base_devices = {d["rank"]: d for d in base_metrics["devices"]}
        unif_devices = {d["rank"]: d for d in unif_metrics["devices"]}
        assert sorted(base_devices) == sorted(unif_devices)
        for rank in base_devices:
            delta = abs(base_devices[rank]["wall_time_s"] - unif_devices[rank]["wall_time_s"])
            assert delta <= RANK_WALL_TOL_S, f"rank {rank} wall time drifted by {delta:.3e}s"
            # Identity profiles must also reproduce the profile-free JSON
            # kernel-idle values exactly (training analog of the inference
            # identity test below).
            for key in ("kernel_idle_layer_s", "kernel_idle_global_s", "kernel_idle_frac_thermal"):
                assert unif_devices[rank][key] == pytest.approx(
                    base_devices[rank][key], rel=1e-9, abs=1e-15
                ), f"rank {rank} {key} differs between identity-profiles and no-profiles"

    def test_uniform_scale_equals_scaled_config(self, tmp_path):
        model_path = _write_yaml(tmp_path, "model.yaml", _tiny_model_dict())
        scale = 0.8
        profile_block = {
            "profiles": {"slow": {"frequency_scale": scale}},
            "devices": {"default": "slow"},
        }
        hw_prof = _hw_with(tmp_path, "hw_prof.yaml", device_profiles=profile_block)
        hw_scaled = _hw_with(tmp_path, "hw_scaled.yaml", freq_mult=scale)
        prof_dir = self._run(tmp_path, "prof", hw_prof, model_path)
        scaled_dir = self._run(tmp_path, "scaled", hw_scaled, model_path)
        prof_metrics = _load_metrics(prof_dir)
        scaled_metrics = _load_metrics(scaled_dir)
        assert abs(prof_metrics["makespan_s"] - scaled_metrics["makespan_s"]) <= RANK_WALL_TOL_S
        prof_devices = {d["rank"]: d for d in prof_metrics["devices"]}
        scaled_devices = {d["rank"]: d for d in scaled_metrics["devices"]}
        for rank in prof_devices:
            delta = abs(prof_devices[rank]["wall_time_s"] - scaled_devices[rank]["wall_time_s"])
            assert delta <= RANK_WALL_TOL_S, f"rank {rank} wall time drifted by {delta:.3e}s"

    def test_uniform_mixed_scales_equal_scaled_config(self, tmp_path):
        """Invariant 3 beyond frequency_scale: a profile mixing frequency, HBM-BW
        and L2-BW scales must equal the identically-scaled hardware config.

        A bank regression that silently drops any of the non-frequency scales
        (e.g. hbm_bandwidth_scale not applied to the profile hw variant) fails
        this test; the frequency-only case cannot catch it.
        """
        model_path = _write_yaml(tmp_path, "model.yaml", _tiny_model_dict())
        freq, hbm_bw, l2_bw = 0.9, 0.8, 0.85
        profile_block = {
            "profiles": {
                "throttled": {
                    "frequency_scale": freq,
                    "hbm_bandwidth_scale": hbm_bw,
                    "l2_bandwidth_scale": l2_bw,
                }
            },
            "devices": {"default": "throttled"},
        }
        hw_prof = _hw_with(tmp_path, "hw_prof.yaml", device_profiles=profile_block)
        hw_scaled = _hw_with(
            tmp_path, "hw_scaled.yaml", freq_mult=freq, hbm_bw_mult=hbm_bw, l2_bw_mult=l2_bw
        )
        prof_dir = self._run(tmp_path, "prof", hw_prof, model_path)
        scaled_dir = self._run(tmp_path, "scaled", hw_scaled, model_path)
        prof_metrics = _load_metrics(prof_dir)
        scaled_metrics = _load_metrics(scaled_dir)
        _validate_device_metrics_schema(prof_metrics, expect_num_layers=4, expect_pp_only=True)
        assert abs(prof_metrics["makespan_s"] - scaled_metrics["makespan_s"]) <= RANK_WALL_TOL_S
        prof_devices = {d["rank"]: d for d in prof_metrics["devices"]}
        scaled_devices = {d["rank"]: d for d in scaled_metrics["devices"]}
        assert sorted(prof_devices) == sorted(scaled_devices)
        for rank in prof_devices:
            delta = abs(prof_devices[rank]["wall_time_s"] - scaled_devices[rank]["wall_time_s"])
            assert delta <= RANK_WALL_TOL_S, f"rank {rank} wall time drifted by {delta:.3e}s"
        # Guard against the degenerate all-knobs-dropped case: the throttled run
        # must actually be slower than an unthrottled baseline would be — check
        # against the profile-free base config makespan.
        hw_base = _hw_with(tmp_path, "hw_base.yaml")
        base_dir = self._run(tmp_path, "base", hw_base, model_path)
        base_metrics = _load_metrics(base_dir)
        assert prof_metrics["makespan_s"] > base_metrics["makespan_s"] * 1.01

    def test_cli_device_profiles_override_end_to_end(self, tmp_path):
        """--device_profiles must REPLACE the hw-YAML block through the real CLI.

        The hw config embeds the hot/nominal block; the CLI passes the uniform
        override. The run's JSON profiles echo must show ONLY the override's
        profiles, proving argparse -> run_LLM -> _apply_device_profiles_override
        end-to-end.
        """
        model_path = _write_yaml(tmp_path, "model.yaml", _tiny_model_dict())
        hw_path = _hw_with(tmp_path, "hw.yaml", device_profiles=HOT_ALL_BLOCK)
        run_dir = tmp_path / "cli"
        run_dir.mkdir()
        result = _run_perf(
            run_dir, hw_path, model_path, env=_ASTRA_ENV,
            device_profiles=PROFILE_UNIFORM_FILE,
        )
        assert "REPLACES" in (result.stdout + result.stderr)
        metrics = _load_metrics(run_dir)
        _validate_device_metrics_schema(metrics, expect_num_layers=4, expect_pp_only=True)
        # The hot block is gone; only the override's nominal profile remains.
        assert sorted(metrics["profiles"]) == ["nominal"]
        for device in metrics["devices"]:
            assert device["profile"] == "nominal"

    def test_inference_identity_profiles_json_kernel_idle_matches_no_profiles(self, tmp_path):
        """Regression: profiles-ON inference JSON must keep the PREFILL kernel idle.

        The prefill re-pricing bank is cleared by the decode-shaped memory
        estimation pass; the writer must use the prefill snapshot instead.
        Identity profiles therefore must reproduce the profiles-OFF JSON
        kernel-idle values exactly: profiles-ON = prefill per-profile idle +
        integrated decode per-profile idle.
        """
        import re

        model_path = _write_yaml(
            tmp_path, "model.yaml", _tiny_model_dict(run_type="inference")
        )
        hw_base = _hw_with(tmp_path, "hw_base.yaml")
        hw_unif = _hw_with(tmp_path, "hw_unif.yaml", device_profiles=UNIFORM_BLOCK)
        base_dir = self._run(tmp_path, "base", hw_base, model_path)
        unif_dir = self._run(tmp_path, "unif", hw_unif, model_path)
        base_metrics = _load_metrics(base_dir)
        unif_metrics = _load_metrics(unif_dir)
        _validate_device_metrics_schema(base_metrics, expect_num_layers=4, expect_pp_only=True)
        _validate_device_metrics_schema(unif_metrics, expect_num_layers=4, expect_pp_only=True)
        assert base_metrics["run_type"] == unif_metrics["run_type"] == "inference"

        # Non-vacuity: this workload has real prefill idle, so equality cannot
        # be satisfied by BOTH sides dropping the prefill term.
        text = (unif_dir / "output" / "LLM" / "LLM_inference_results.txt").read_text()
        prefill_layer = float(re.findall(r"Prefill Idle Layer Time:\s*([\d.eE+-]+)s", text)[-1])
        decode_layer = float(re.findall(r"Decode Idle Layer Time:\s*([\d.eE+-]+)s", text)[-1])
        assert prefill_layer > 0.0

        base_devices = {d["rank"]: d for d in base_metrics["devices"]}
        unif_devices = {d["rank"]: d for d in unif_metrics["devices"]}
        assert sorted(base_devices) == sorted(unif_devices)
        for rank in base_devices:
            for key in ("kernel_idle_layer_s", "kernel_idle_global_s", "kernel_idle_frac_thermal"):
                assert unif_devices[rank][key] == pytest.approx(
                    base_devices[rank][key], rel=1e-9, abs=1e-15
                ), f"rank {rank} {key}: identity-profiles != no-profiles"

        # And the values decompose as prefill + integrated decode: summed over
        # ranks, per-layer idle * layers_hosted totals (prefill+decode) * L.
        num_layers = 4
        total_layer_json = sum(d["kernel_idle_layer_s"] for d in unif_metrics["devices"])
        assert total_layer_json == pytest.approx(
            (prefill_layer + decode_layer) * num_layers, rel=1e-6
        ), "JSON kernel_idle_layer_s does not decompose as prefill + integrated decode"

    def test_hetero_sanity_and_schema(self, tmp_path):
        model_path = _write_yaml(tmp_path, "model.yaml", _tiny_model_dict())
        hw_base = _hw_with(tmp_path, "hw_base.yaml")
        hot_block = {
            "profiles": {"hot": {"frequency_scale": 0.5}, "nominal": {}},
            "devices": {0: "hot", "default": "nominal"},
        }
        hw_hot = _hw_with(tmp_path, "hw_hot.yaml", device_profiles=hot_block)
        base_dir = self._run(tmp_path, "base", hw_base, model_path)
        hot_dir = self._run(tmp_path, "hot", hw_hot, model_path)
        base_metrics = _load_metrics(base_dir)
        hot_metrics = _load_metrics(hot_dir)
        _validate_device_metrics_schema(hot_metrics, expect_num_layers=4, expect_pp_only=True)

        # Slowing one device gates the makespan.
        assert hot_metrics["makespan_s"] > base_metrics["makespan_s"] * 1.01

        base_devices = {d["hw_id"]: d for d in base_metrics["devices"]}
        hot_devices = {d["hw_id"]: d for d in hot_metrics["devices"]}
        slowed_delta = hot_devices[0]["sched_idle_frac"] - base_devices[0]["sched_idle_frac"]
        fast_delta = hot_devices[1]["sched_idle_frac"] - base_devices[1]["sched_idle_frac"]
        # The slowed device spends MORE time computing (its schedule idle drops
        # vs its own baseline); the fast device waits longer (idle rises).
        # Note: comparing the two devices' absolute idle fractions within one
        # run is confounded by baseline pipeline asymmetry (stage 1 hosts the
        # lm head + optimizer), so the invariant is on the per-device deltas.
        assert slowed_delta < 0.0, f"slowed device idle did not drop (delta={slowed_delta:.4f})"
        assert fast_delta > 0.0, f"fast device idle did not rise (delta={fast_delta:.4f})"
        assert slowed_delta < fast_delta
        assert hot_devices[0]["profile"] == "hot"
        assert hot_devices[1]["profile"] == "nominal"
        assert hot_devices[0]["compute_busy_s"] > base_devices[0]["compute_busy_s"]

    def test_inference_uniform_profile_equals_scaled_config(self, tmp_path):
        """A4 regression: decode must be re-priced through decode's own path."""
        model_path = _write_yaml(
            tmp_path, "model.yaml", _tiny_model_dict(run_type="inference")
        )
        scale = 0.8
        profile_block = {
            "profiles": {"slow": {"frequency_scale": scale}},
            "devices": {"default": "slow"},
        }
        hw_prof = _hw_with(tmp_path, "hw_prof.yaml", device_profiles=profile_block)
        hw_scaled = _hw_with(tmp_path, "hw_scaled.yaml", freq_mult=scale)
        prof_dir = self._run(tmp_path, "prof", hw_prof, model_path)
        scaled_dir = self._run(tmp_path, "scaled", hw_scaled, model_path)

        prof_metrics = _load_metrics(prof_dir)
        scaled_metrics = _load_metrics(scaled_dir)
        _validate_device_metrics_schema(prof_metrics, expect_num_layers=4, expect_pp_only=True)
        assert prof_metrics["run_type"] == "inference"

        # Prefill per-rank walls are single-run values: <= 1 us per rank.
        prof_prefill = {r["name"]: r for r in prof_metrics["runs"]}["prefill"]
        scaled_prefill = {r["name"]: r for r in scaled_metrics["runs"]}["prefill"]
        for prof_dev, scaled_dev in zip(prof_prefill["devices"], scaled_prefill["devices"]):
            assert prof_dev["rank"] == scaled_dev["rank"]
            assert abs(prof_dev["wall_time_s"] - scaled_dev["wall_time_s"]) <= RANK_WALL_TOL_S

        # Decode totals are trapezoid-weighted over decode_len steps; the us
        # quantization scales with the weight sum (decode_len).
        decode_len = 64
        decode_tol = RANK_WALL_TOL_S * decode_len
        prof_decode = {r["name"]: r for r in prof_metrics["runs"]}["decode"]
        scaled_decode = {r["name"]: r for r in scaled_metrics["runs"]}["decode"]
        assert abs(prof_decode["total_raw_s"] - scaled_decode["total_raw_s"]) <= decode_tol
        assert abs(prof_metrics["makespan_s"] - scaled_metrics["makespan_s"]) <= decode_tol + RANK_WALL_TOL_S

        # And the reported phase times agree.
        def _phase_times(run_dir):
            text = (run_dir / "output" / "LLM" / "LLM_inference_results.txt").read_text()
            import re

            prefill = float(re.findall(r"Prefill Time:\s*([\d.eE+-]+)s", text)[-1])
            decode = float(re.findall(r"Decode Time:\s*([\d.eE+-]+)s", text)[-1])
            return prefill, decode

        prof_prefill_t, prof_decode_t = _phase_times(prof_dir)
        scaled_prefill_t, scaled_decode_t = _phase_times(scaled_dir)
        assert abs(prof_prefill_t - scaled_prefill_t) <= RANK_WALL_TOL_S
        assert abs(prof_decode_t - scaled_decode_t) <= decode_tol

    def test_grad_accum_records_both_runs(self, tmp_path):
        # GA=3 on purpose: GA=2 makes (GA-1)=1, where the weighted A5 formulas
        # are indistinguishable from unweighted sums and a weight-dropping
        # regression would pass unnoticed.
        ga_steps = 3
        model_path = _write_yaml(tmp_path, "model.yaml", _tiny_model_dict(ga_steps=ga_steps))
        # ZeRO-2 rejects GA>1; run this case with plain DDP sharding.
        hw_hot = _hw_with(
            tmp_path, "hw.yaml", device_profiles=HOT_ALL_BLOCK, sw_patch={"dp_zero_stage": 0}
        )
        run_dir = self._run(tmp_path, "ga", hw_hot, model_path)
        metrics = _load_metrics(run_dir)
        _validate_device_metrics_schema(metrics, expect_num_layers=4, expect_pp_only=True)
        assert metrics["gradient_accumulation_steps"] == ga_steps
        runs = {r["name"]: r for r in metrics["runs"]}
        assert set(runs) == {"no_dp", "final"}
        assert runs["no_dp"]["weight"] == float(ga_steps - 1) == 2.0  # GA-1
        assert runs["final"]["weight"] == 1.0
        # Combined makespan is the WEIGHTED sum: no_dp*(GA-1) + final. Both run
        # totals are nonzero, so an unweighted-sum regression cannot pass.
        assert runs["no_dp"]["total_raw_s"] > 0.0
        assert runs["final"]["total_raw_s"] > 0.0
        expected = 2.0 * runs["no_dp"]["total_raw_s"] + runs["final"]["total_raw_s"]
        unweighted = runs["no_dp"]["total_raw_s"] + runs["final"]["total_raw_s"]
        assert metrics["makespan_s"] == pytest.approx(expected, rel=1e-12)
        assert metrics["makespan_s"] > unweighted * (1.0 + 1e-9)
        for device in metrics["devices"]:
            per_run = {
                name: next(d for d in runs[name]["devices"] if d["rank"] == device["rank"])
                for name in ("no_dp", "final")
            }
            # busy = no_dp*2 + final (A5).
            combined_busy = 2.0 * per_run["no_dp"]["compute_busy_s"] + per_run["final"]["compute_busy_s"]
            assert per_run["no_dp"]["compute_busy_s"] > 0.0
            assert device["compute_busy_s"] == pytest.approx(combined_busy, rel=1e-12)
            assert device["compute_busy_s"] > (
                per_run["no_dp"]["compute_busy_s"] + per_run["final"]["compute_busy_s"]
            ) * (1.0 + 1e-9)


# ---------------------------------------------------------------------------
# 6. Astra-cache distinctness
# ---------------------------------------------------------------------------

_CACHE_DRIVER = """
import json, os, sys
sys.path.insert(0, sys.argv[5])
os.environ["RAPID_ASTRA_CACHE_MODE"] = "CACHE_READWRITE"
os.environ["RAPID_PERSIST_ASTRASIM_ARTIFACTS"] = "1"
import config
from train_timing import TimeCalculationLLM

model = config.parse_config(sys.argv[3], "LLM")
out_dir = sys.argv[4]
totals = []
for hw_path in (sys.argv[1], sys.argv[2]):
    hw = config.parse_config(hw_path, "hardware")
    config.validate_configs(hw, model)
    tc = TimeCalculationLLM(hw, model, "LLM", output_dir=out_dir)
    totals.append(tc.calc_time_llm())
cache = json.load(open(os.path.join(out_dir, "astra_flat", "cache.json")))
print("CACHE_RESULT", totals[0], totals[1], len(cache))
"""


@needs_astra
def test_astra_cache_distinct_profiles(tmp_path):
    """Two different profiles sharing one cache dir must yield different totals."""
    model_path = _write_yaml(tmp_path, "model.yaml", _tiny_model_dict())
    hw_hot = _hw_with(tmp_path, "hw_hot.yaml", device_profiles=HOT_ALL_BLOCK)
    hw_nom = _hw_with(tmp_path, "hw_nom.yaml", device_profiles=UNIFORM_BLOCK)
    out_dir = tmp_path / "shared_out"
    out_dir.mkdir()
    result = subprocess.run(
        [
            sys.executable, "-c", _CACHE_DRIVER,
            str(hw_hot), str(hw_nom), str(model_path), str(out_dir), str(REPO_ROOT),
        ],
        cwd=str(tmp_path),
        env=_ASTRA_ENV,
        capture_output=True,
        text=True,
        timeout=560,
    )
    assert result.returncode == 0, (
        f"cache driver failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
    )
    line = [ln for ln in result.stdout.splitlines() if ln.startswith("CACHE_RESULT")][-1]
    _, total_hot, total_nominal, cache_entries = line.split()
    total_hot, total_nominal = float(total_hot), float(total_nominal)
    # A stale cache hit would return the hot total for the nominal run.
    assert abs(total_hot - total_nominal) > RANK_WALL_TOL_S
    assert total_hot > total_nominal
    assert int(cache_entries) >= 2
