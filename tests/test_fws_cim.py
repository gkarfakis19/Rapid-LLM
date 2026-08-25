"""FWS-CIM device-class tests.

Stage 1: config schema (CIMConfig tree, device_class, vision.num_prefix_tokens
override), YAML template parsing, and the validation gates (transformer
inference only; no training / flash attention / MLA / astra; pass 2 admits
dense and MoE LLMs with decode and the cim_sram/cim_dram KV stories).
Stage 2: device laws (cim_timing.CimDeviceModel) — SA closed form vs recorded
OPTIMA cycles, analog stage times, array counts / areas, N_mult compensation.
Stage 3: integration through the real entry point (run_perf.py subprocess) —
T1 smoke vs recorded targets, tp=2 variant, GPU-baseline inertness.
"""

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import cim_timing
import config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HW_DIR = PROJECT_ROOT / "configs" / "hardware-config"
MODEL_DIR = PROJECT_ROOT / "configs" / "model-config"

FWS_T1 = HW_DIR / "fws_cim_optima_t1.yaml"
FWS_T2 = HW_DIR / "fws_cim_optima_t2.yaml"
FWS_T3 = HW_DIR / "fws_cim_optima_t3.yaml"
A100 = HW_DIR / "a100_80GB.yaml"
VIT_HUGE_64 = MODEL_DIR / "vit_huge_story_64_inf.yaml"
VIT_G_64 = MODEL_DIR / "vit_g_64_inf.yaml"
VIT_HUGE_196 = MODEL_DIR / "vit_huge_story_196_inf.yaml"
VIT_BASE = MODEL_DIR / "vit_base_inf.yaml"
LLAMA2_7B_INF = MODEL_DIR / "Llama2-7B_inf.yaml"
GEMM_MODEL = MODEL_DIR / "GEMM.yaml"
DEEPSEEK_V3_INF = MODEL_DIR / "DeepSeekV3_inf.yaml"
FWS_LLAMA7B = HW_DIR / "fws_cim_llama7b.yaml"
FWS_LLAMA7B_KVDRAM = HW_DIR / "fws_cim_llama7b_kvdram.yaml"
FWS_MOE = HW_DIR / "fws_cim_moe.yaml"
LLAMA2_7B_FWS_INF = MODEL_DIR / "llama2_7b_fws_inf.yaml"
MOE_SMALL_FWS_INF = MODEL_DIR / "moe_small_fws_inf.yaml"
#: QIF P1 model-matrix rows: hybrid block kinds vs the attention-only rows.
HYBRID_MATRIX_INF = {
    "ssm": MODEL_DIR / "granite_4_0_h_tiny_inf.yaml",
    "linear_attn": MODEL_DIR / "qwen3_5_4b_inf.yaml",
    "short_conv": MODEL_DIR / "lfm2_2p6b_inf.yaml",
}
ATTENTION_ONLY_MATRIX_INF = (
    MODEL_DIR / "hunyuan_7b_inf.yaml",
    MODEL_DIR / "smollm3_3b_inf.yaml",
)


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _hw_from_dict(config_dict):
    """Mirror parse_config for an in-memory hardware dict (convert + from_dict)."""
    config_dict = copy.deepcopy(config_dict)
    config.convert(config_dict)
    return config.HWConfig.from_dict(config_dict)


def _model_from_yaml(path, mutate=None):
    """Mirror parse_config for a model YAML, with an optional dict mutation."""
    config_dict = _load_yaml(path)
    if mutate is not None:
        mutate(config_dict)
    config.convert(config_dict)
    model_config = config.LLMConfig.from_dict(config_dict["model_param"])
    inference_config = None
    if model_config.run_type == "inference":
        inference_config = config.LLMInferenceConfig.from_dict(config_dict.get("inference_param"))
    return config.ModelConfig(model_config=model_config, inference_config=inference_config)


# ---------------------------------------------------------------------------
# Parsing: fws_cim hardware YAML templates
# ---------------------------------------------------------------------------


def test_fws_cim_t1_parses_expected_values():
    hw = config.parse_config(str(FWS_T1), "hardware")
    assert hw.device_class == "fws_cim"
    cim = hw.cim_config
    assert cim is not None

    analog = cim.analog
    assert analog.rows == 1280
    assert analog.cols_adc == 320
    assert analog.adc_mux == 4
    assert analog.cols == 1280  # cols_adc * adc_mux
    assert analog.slice_cycles == 2
    assert analog.analog_clock_mhz == 100
    assert analog.energy_per_vec_pj == pytest.approx(9534.9389)
    assert analog.shots_per_output == 2
    assert analog.area_mm2_per_array == pytest.approx(1.403065)

    fabric = cim.fabric
    assert fabric.model == "sa"
    assert fabric.rows == 32
    assert fabric.cols == 64
    assert fabric.num_arrays == 2
    assert fabric.replicas == 1
    assert fabric.clock_ghz == pytest.approx(0.95)
    # fill_drain_penalty_cycles is absent in the YAML -> defaults to 3*rows.
    assert fabric.fill_drain_penalty_cycles == 96
    assert fabric.softmax_lanes == 1
    assert fabric.softmax_pipeline_depth == 20

    chip = cim.chip
    assert chip.arrays_per_chip == 400
    assert chip.layers_per_chip == 32

    # Stub sw_param calibration: deterministic fabric has no launch overhead.
    assert hw.sw_config.kernel_launch_overhead == 0
    # fws_cim configs pass hardware validation.
    config.validate_hw_config(hw)


def test_fws_cim_t2_t3_parse_expected_values():
    hw2 = config.parse_config(str(FWS_T2), "hardware")
    a2 = hw2.cim_config.analog
    assert (a2.rows, a2.cols_adc, a2.adc_mux) == (1408, 704, 2)
    assert a2.energy_per_vec_pj == pytest.approx(10138.5868)
    assert a2.area_mm2_per_array == pytest.approx(2.014071)
    assert hw2.cim_config.fabric.clock_ghz == pytest.approx(1.0)
    assert hw2.cim_config.chip.arrays_per_chip == 0  # capacity unchecked
    assert hw2.cim_config.chip.layers_per_chip == 40

    hw3 = config.parse_config(str(FWS_T3), "hardware")
    a3 = hw3.cim_config.analog
    assert (a3.rows, a3.cols_adc, a3.adc_mux) == (1280, 80, 16)
    assert a3.energy_per_vec_pj == pytest.approx(12470.0063)
    assert a3.area_mm2_per_array == pytest.approx(1.135296)
    assert hw3.cim_config.fabric.clock_ghz == pytest.approx(0.95)
    assert hw3.cim_config.chip.layers_per_chip == 32

    for hw in (hw2, hw3):
        assert hw.device_class == "fws_cim"
        config.validate_hw_config(hw)


def test_fws_cim_single_chip_ring_dims_and_parallelism():
    # DESIGN landmines: analytical mode requires ring dims; the parallelism
    # and network.overlap blocks must survive parsing even for a single chip.
    hw = config.parse_config(str(FWS_T1), "hardware")
    sch = hw.sch_config
    assert (sch.tp, sch.pp, sch.mb, sch.cp) == (1, 1, 1, 1)
    for dim in hw.network_layout.dimensions:
        assert str(dim.topology_type).lower() == "ring"


def test_layers_per_chip_accepts_explicit_list():
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["cim"]["chip"]["layers_per_chip"] = [16, 16]
    hw = _hw_from_dict(hw_dict)
    assert hw.cim_config.chip.layers_per_chip == (16, 16)


def test_fill_drain_penalty_explicit_value_wins():
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["cim"]["fabric"]["fill_drain_penalty_cycles"] = 42
    hw = _hw_from_dict(hw_dict)
    assert hw.cim_config.fabric.fill_drain_penalty_cycles == 42


# ---------------------------------------------------------------------------
# Default inertness: existing configs keep existing behavior
# ---------------------------------------------------------------------------


def test_device_class_absent_defaults_to_gpu():
    hw = config.parse_config(str(A100), "hardware")
    assert hw.device_class == "gpu"
    assert hw.cim_config is None
    config.validate_hw_config(hw)


def test_existing_a100_vit_pair_still_validates():
    hw = config.parse_config(str(A100), "hardware")
    model = config.parse_config(str(VIT_BASE), "VIT")
    config.validate_configs(hw, model)


def test_unknown_device_class_rejected():
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["device_class"] = "tpu"
    with pytest.raises(ValueError, match="device_class"):
        _hw_from_dict(hw_dict)


# ---------------------------------------------------------------------------
# Scope rejection list (each error must name the offending setting)
# ---------------------------------------------------------------------------


def test_fws_cim_rejects_training():
    hw = config.parse_config(str(FWS_T1), "hardware")

    def to_training(d):
        d["model_param"]["run_type"] = "training"

    model = _model_from_yaml(VIT_HUGE_64, mutate=to_training)
    with pytest.raises(ValueError, match="run_type"):
        config.validate_configs(hw, model)


def test_fws_cim_rejects_non_transformer_model():
    # Pass 2 admits LLM inference (see the pass-2 section below); a
    # non-transformer (GEMM) model config is still rejected.
    hw = config.parse_config(str(FWS_T1), "hardware")
    model = config.parse_config(str(GEMM_MODEL), "GEMM")
    with pytest.raises(ValueError, match="transformer"):
        config.validate_configs(hw, model)


def test_vit_decode_len_still_rejected_at_parse():
    # Pass 2 admits decode for LLMs (the fws_cim gate no longer rejects
    # decode_len); a ViT YAML with decode_len > 0 is still rejected at parse
    # time with a message that names the setting.
    def set_decode(d):
        d["model_param"]["decode_len"] = 1

    with pytest.raises(ValueError, match="decode_len"):
        _model_from_yaml(VIT_HUGE_64, mutate=set_decode)


def test_fws_cim_rejects_flashattention():
    hw = config.parse_config(str(FWS_T1), "hardware")

    def set_flash(d):
        d["model_param"]["attention"]["use_flashattention"] = True
        d["model_param"]["attention"]["attention_tile_size"] = 64

    model = _model_from_yaml(VIT_HUGE_64, mutate=set_flash)
    with pytest.raises(ValueError, match="use_flashattention"):
        config.validate_configs(hw, model)


def test_fws_cim_rejects_astra_backend():
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["execution_backend"]["model"] = "astra"
    hw = _hw_from_dict(hw_dict)
    with pytest.raises(ValueError, match="analytical"):
        config.validate_hw_config(hw)


def test_fws_cim_rejects_missing_cim_block():
    hw_dict = _load_yaml(FWS_T1)
    del hw_dict["cim"]
    hw = _hw_from_dict(hw_dict)
    with pytest.raises(ValueError, match="cim"):
        config.validate_hw_config(hw)


def test_fws_cim_rejects_pp_gt_1():
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["parallelism"]["pp"] = 2
    hw = _hw_from_dict(hw_dict)
    with pytest.raises(ValueError, match="pp"):
        config.validate_hw_config(hw)


@pytest.mark.parametrize("fabric_model", ["sa", "gpu_native"])
def test_fws_cim_rejects_cp_gt_1(fabric_model):
    # Both fabric modes must reject cp at the same config seam: gpu_native
    # never reaches the sa pricing path's cp check, so without this gate a
    # cp=2 run would complete and silently ignore cp.
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["parallelism"]["cp"] = 2
    hw_dict["cim"]["fabric"]["model"] = fabric_model
    hw = _hw_from_dict(hw_dict)
    with pytest.raises(ValueError, match="cp"):
        config.validate_hw_config(hw)


def test_cim_block_without_fws_cim_device_class_warns(capsys):
    hw_dict = _load_yaml(FWS_T1)
    del hw_dict["device_class"]
    hw = _hw_from_dict(hw_dict)
    assert hw.device_class == "gpu"
    assert hw.cim_config is not None
    config.validate_hw_config(hw)  # warns, does not raise
    captured = capsys.readouterr()
    assert "cim" in captured.out
    assert "device_class" in captured.out


# ---------------------------------------------------------------------------
# model_param.vision.num_prefix_tokens override
# ---------------------------------------------------------------------------


def test_prefix_override_changes_derived_seq_len():
    for path, expected_seq in ((VIT_HUGE_64, 64), (VIT_G_64, 64), (VIT_HUGE_196, 196)):
        model = config.parse_config(str(path), "VIT").model_config
        assert model.num_prefix_tokens == 0
        assert model.seq_len == expected_seq


def test_prefix_default_behavior_unchanged():
    # 224px/16 patches = 196 tokens + 1 default CLS token -> 197.
    model = config.parse_config(str(VIT_BASE), "VIT").model_config
    assert model.num_prefix_tokens == 1
    assert model.seq_len == 197


def test_prefix_default_for_dinov3_is_five():
    def to_dinov3(d):
        d["model_param"]["model_type"] = "vit_dinov3"
        del d["model_param"]["vision"]["num_prefix_tokens"]

    model = _model_from_yaml(VIT_HUGE_64, mutate=to_dinov3).model_config
    assert model.num_prefix_tokens == 5
    assert model.seq_len == 64 + 5


def test_prefix_override_rejects_negative():
    def set_negative(d):
        d["model_param"]["vision"]["num_prefix_tokens"] = -1

    with pytest.raises(ValueError, match="num_prefix_tokens"):
        _model_from_yaml(VIT_HUGE_64, mutate=set_negative)


# ---------------------------------------------------------------------------
# Stage 2: device laws (cim_timing.CimDeviceModel)
# ---------------------------------------------------------------------------


def _cim_device(hw_path, model_path):
    hw = config.parse_config(str(hw_path), "hardware")
    model = config.parse_config(str(model_path), "VIT").model_config
    return cim_timing.CimDeviceModel(hw, model)


@pytest.fixture(scope="module")
def cim_t1():
    return _cim_device(FWS_T1, VIT_HUGE_64)


@pytest.fixture(scope="module")
def cim_t2():
    return _cim_device(FWS_T2, VIT_G_64)


@pytest.fixture(scope="module")
def cim_t3():
    return _cim_device(FWS_T3, VIT_HUGE_196)


def _fake_tc(**overrides):
    """Minimal TimeCalculation surface price_gemm reads.

    Pass 2 prices attention at the CALL DIMS, so price_gemm reads exactly:
    batch_size, kv_heads, tp, cp (all set by TimeCalculation.__init__).
    Non-decode weight-GEMM pricing uses none of them (only dim1 as
    received); decode weight ops additionally read batch_size (M = B).
    """
    attrs = dict(batch_size=1, kv_heads=16, tp=1, cp=1)
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


# --- SA closed form vs ALL recorded ScaleSim cycle counts (DESIGN section 5) ---


def test_sa_closed_form_reproduces_recorded_cycles(cim_t1, cim_t2, cim_t3):
    # T1 (R32 C64): QK M=64 N=64 K=80*16; PV M=64 N=80 K=64*16.
    assert cim_t1.sa_cycles(64, 64, 1280) == 2747  # = 2*(1280+94)-1
    assert cim_t1.sa_cycles(64, 80, 1024) == 4471  # = 2*2*(1024+94)-1
    # T2 (head_dim 88): QK K=88*16=1408; PV K=64*16=1024.
    assert cim_t2.sa_cycles(64, 64, 1408) == 3003
    assert cim_t2.sa_cycles(64, 88, 1024) == 4471
    # T3 (seq 196): QK K=1280; PV K=196*16=3136.
    assert cim_t3.sa_cycles(196, 196, 1280) == 38471  # = 7*4*(1280+94)-1
    assert cim_t3.sa_cycles(196, 80, 3136) == 45219   # = 7*2*(3136+94)-1


def test_folded_attention_derives_recorded_shapes(cim_t1, cim_t2, cim_t3):
    # attention_timing folds heads into K itself from the model params.
    for dev, qk, pv in ((cim_t1, 2747, 4471), (cim_t2, 3003, 4471), (cim_t3, 38471, 45219)):
        att = dev.attention_timing()
        assert att.heads_chip == 16
        assert att.heads_per_replica == 16
        assert att.qk_cycles == qk
        assert att.pv_cycles == pv
        assert att.total_cycles == max(qk, pv) + 96  # fill_drain = 3*rows


# --- Analog law: stage times, K/N-independence -------------------------------


def test_analog_stage_times_match_recorded_periods(cim_t1, cim_t2, cim_t3):
    assert cim_t1.vec_latency_s == pytest.approx(80e-9, rel=1e-12)   # mux4*slice2/100MHz
    assert cim_t1.analog_gemm_time(64) == pytest.approx(5.12e-6, rel=1e-9)
    assert cim_t2.analog_gemm_time(64) == pytest.approx(2.56e-6, rel=1e-9)
    assert cim_t3.analog_gemm_time(196) == pytest.approx(62.72e-6, rel=1e-9)


def test_layer_stage_times_s1_s3_s4_s5_are_the_analog_law(cim_t1):
    stages = cim_t1.layer_stage_times()
    analog = cim_t1.analog_gemm_time(64)
    for key in ("S1_qkv", "S3_o_proj", "S4_ffn1", "S5_ffn2"):
        assert stages[key] == pytest.approx(analog, rel=1e-12)
    # S2 = max(T_sa, T_softmax) = (4471+96)/0.95 GHz here (SA-bound).
    assert stages["S2_attention"] == pytest.approx(4567 / 0.95e9, rel=1e-12)


def test_pipeline_period_and_bottleneck(cim_t1, cim_t2, cim_t3):
    period1, bottleneck1 = cim_t1.pipeline_period()
    assert period1 == pytest.approx(5.12e-6, rel=1e-9)
    assert bottleneck1 == "S1_qkv"
    assert 1.0 / period1 == pytest.approx(195312.5, rel=1e-9)

    period2, bottleneck2 = cim_t2.pipeline_period()
    assert period2 == pytest.approx(4.567e-6, rel=1e-9)
    assert bottleneck2 == "S2_attention"
    assert 1.0 / period2 == pytest.approx(218962.11955331726, rel=1e-9)

    period3, bottleneck3 = cim_t3.pipeline_period()
    assert period3 == pytest.approx(62.72e-6, rel=1e-9)
    assert bottleneck3 == "S1_qkv"


def test_block_latency_matches_optima_with_peripheral_stage(cim_t1, cim_t2, cim_t3):
    # OPTIMA's recorded block latency sums a sixth trailing stage equal to one
    # analog stage time (see cim_timing module docstring); Rapid-LLM lists
    # endpoint stages separately, so compare block_latency + one analog stage.
    for dev, seq, recorded_us in ((cim_t1, 64, 30.407368421052634), (cim_t2, 64, 17.367), (cim_t3, 196, 361.3)):
        optima_block = dev.block_latency() + dev.analog_gemm_time(seq)
        assert optima_block == pytest.approx(recorded_us * 1e-6, rel=1e-9)


# --- Array counting and area -------------------------------------------------


def test_arrays_per_layer_recorded_counts(cim_t1, cim_t2, cim_t3):
    assert cim_t1.per_layer_stage_arrays() == {"qkv": 3, "o_proj": 1, "ffn1": 4, "ffn2": 4}
    assert cim_t1.arrays_per_layer() == 12
    # T2: ceil(6144/1408) = 5 for each FFN matrix.
    assert cim_t2.per_layer_stage_arrays() == {"qkv": 3, "o_proj": 1, "ffn1": 5, "ffn2": 5}
    assert cim_t2.arrays_per_layer() == 14
    # T3 stores the same weights as T1 (mux 16 keeps cols_adc*adc_mux = 1280).
    assert cim_t3.arrays_per_layer() == 12


def test_endpoint_arrays(cim_t1):
    # patch embed: K = 3*16*16 = 768, N = 1280 -> 1; head: K=1280, N=1000 -> 1.
    assert cim_t1.endpoint_arrays() == {"patch_embed": 1, "vit_head": 1}
    assert cim_t1.transformer_stack_arrays() == 384
    assert cim_t1.total_arrays() == 386


def test_ctt_stack_area_matches_recorded(cim_t1, cim_t2, cim_t3):
    # OPTIMA's recorded CTT area counts transformer-stack arrays only.
    assert cim_t1.stack_area_mm2() == pytest.approx(538.7768, abs=1e-3)
    assert cim_t2.stack_area_mm2() == pytest.approx(1127.8796, abs=1e-3)
    assert cim_t3.stack_area_mm2() == pytest.approx(435.9537, abs=1e-3)
    # Endpoint arrays add on top for the full-chip figure.
    assert cim_t1.total_area_mm2() == pytest.approx(386 * 1.403065, rel=1e-12)


# --- price_gemm: role classification, laws, N_mult compensation --------------


def test_price_gemm_weight_law_is_kn_independent(cim_t1):
    tc = _fake_tc()
    t = cim_t1.price_gemm("qkv_projection_f", 64, 1280, 3840, tc)
    assert t == pytest.approx(5.12e-6, rel=1e-12)
    # K/N never matter — tp-sharded dims give the same time.
    assert cim_t1.price_gemm("ffn_f", 64, 1280, 5120, tc) == pytest.approx(t, rel=1e-12)
    assert cim_t1.price_gemm("ffn2_f", 64, 2560, 1280, tc) == pytest.approx(t, rel=1e-12)
    # vit head arrives with M=1 (pooled CLS token).
    assert cim_t1.price_gemm("vit_head_f", 1, 1280, 1000, tc) == pytest.approx(80e-9, rel=1e-12)


def test_price_gemm_backward_twin_priced_on_arriving_dims(cim_t1):
    # Backward twins are priced during inference then discarded; same law,
    # whatever dims arrive, no crash.
    tc = _fake_tc()
    t = cim_t1.price_gemm("qkv_projection_b", 1280, 64, 3840, tc)
    assert t == pytest.approx(1280 * cim_t1.vec_latency_s, rel=1e-12)
    t_att_b = cim_t1.price_gemm("attention_score_b", 64, 64, 80, tc)
    assert t_att_b == pytest.approx(cim_t1.price_gemm("attention_score_f", 64, 80, 64, tc), rel=1e-12)


def test_price_gemm_nmult_compensation_tp1(cim_t1):
    # SINGLE mode (tp=1): the caller multiplies the one-head-shape price by
    # batch = B*kv_heads (train_timing single_gpu_gemm_forward); the folded
    # per-chip total must be restored exactly.
    tc = _fake_tc(batch_size=1, kv_heads=16, tp=1)
    t_chip = cim_t1.attention_timing(seq_len=64, head_dim=80, kv_heads=16, tp=1).sa_time_s
    assert t_chip == pytest.approx(4567 / 0.95e9, rel=1e-12)
    for name in ("attention_score_f", "attention_output_f"):
        priced = cim_t1.price_gemm(name, 64, 80, 64, tc)
        assert priced * 1 * 16 == pytest.approx(t_chip, rel=1e-12)


def test_price_gemm_nmult_compensation_tp2_tensor_sequence(cim_t1):
    # TENSOR_SEQUENCE (tp=2): callers multiply by batch * (1/tp) = B*kv_heads/tp
    # (train_timing._tensor_parallelism_gemm_forward); heads_chip = ceil(16/2).
    tc = _fake_tc(batch_size=1, kv_heads=16, tp=2)
    att = cim_t1.attention_timing(seq_len=64, head_dim=80, kv_heads=16, tp=2)
    assert att.heads_chip == math.ceil(16 / 2)
    assert att.qk_cycles == cim_t1.sa_cycles(64, 64, 80 * 8)
    assert att.pv_cycles == cim_t1.sa_cycles(64, 80, 64 * 8)
    priced = cim_t1.price_gemm("attention_score_f", 64, 80, 64, tc)
    assert priced * (1 * 16 / 2) == pytest.approx(att.sa_time_s, rel=1e-12)


def test_price_gemm_rejects_context_parallelism(cim_t1):
    tc = _fake_tc(cp=2)
    with pytest.raises(ValueError, match="cp"):
        cim_t1.price_gemm("attention_score_f", 64, 80, 64, tc)


def test_price_gemm_gpu_native_passes_attention_through():
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["cim"]["fabric"]["model"] = "gpu_native"
    hw = _hw_from_dict(hw_dict)
    model = config.parse_config(str(VIT_HUGE_64), "VIT").model_config
    dev = cim_timing.CimDeviceModel(hw, model)
    tc = _fake_tc()
    # ACT_GEMMs return None -> the native tile/roofline path continues.
    assert dev.price_gemm("attention_score_f", 64, 80, 64, tc) is None
    assert dev.price_gemm("attention_output_f", 64, 64, 80, tc) is None
    # Weight GEMMs still take the analog law.
    assert dev.price_gemm("qkv_projection_f", 64, 1280, 3840, tc) == pytest.approx(5.12e-6, rel=1e-12)


def test_price_gemm_unknown_name_warns_once_and_uses_analog_law(cim_t1, capsys):
    tc = _fake_tc()
    t_f = cim_t1.price_gemm("mystery_op_f", 64, 128, 256, tc)
    t_b = cim_t1.price_gemm("mystery_op_b", 32, 128, 256, tc)
    assert t_f == pytest.approx(64 * cim_t1.vec_latency_s, rel=1e-12)
    assert t_b == pytest.approx(32 * cim_t1.vec_latency_s, rel=1e-12)
    captured = capsys.readouterr()
    assert captured.out.count("mystery_op") == 1  # one-time warning per base op


# --- Softmax lanes law -------------------------------------------------------


def test_softmax_lanes_law(cim_t1):
    # cycles = pipeline_depth + ceil(S*heads_chip/lanes) - 1, lanes = 1*1.
    assert cim_t1.softmax_cycles(64, 16) == 20 + math.ceil(64 * 16 / 1) - 1 == 1043
    att = cim_t1.attention_timing()
    assert att.softmax_time_s == pytest.approx(1043 / 0.95e9, rel=1e-12)
    # SA-bound here: stage-2 time = max(T_sa, T_softmax) = T_sa.
    assert att.stage_time_s == pytest.approx(att.sa_time_s, rel=1e-12)


# --- Energy law (analog part only; pass-1 energy is PARTIAL) ----------------


def test_analog_energy_law(cim_t1):
    e = cim_t1.layer_stage_energy_pj()
    # E = M_tokens * E_vec * shots * arrays per stage per layer.
    assert e["qkv"] == pytest.approx(64 * 9534.9389 * 2 * 3, rel=1e-12)
    assert e["ffn1"] == pytest.approx(64 * 9534.9389 * 2 * 4, rel=1e-12)
    assert cim_t1.transformer_stack_energy_pj() == pytest.approx(32 * sum(e.values()), rel=1e-12)
    ep = cim_t1.endpoint_energy_pj()
    # Head runs at M = batch_size = 1 on one array.
    assert ep["vit_head"] == pytest.approx(1 * 9534.9389 * 2 * 1, rel=1e-12)
    assert ep["patch_embed"] == pytest.approx(64 * 9534.9389 * 2 * 1, rel=1e-12)


# --- Chip placement and capacity --------------------------------------------


def test_chip_capacity_check(cim_t1, cim_t2):
    # T1: single chip, 32*12 + 2 endpoint arrays = 386 <= 400.
    assert cim_t1.chip_layer_counts() == (32,)
    assert cim_t1.chip_array_usage() == (386,)
    cim_t1.validate_capacity()  # passes
    # T2: arrays_per_chip = 0 -> capacity unchecked even though usage is 562.
    assert cim_t2.chip_array_usage() == (14 * 40 + 2,)
    cim_t2.validate_capacity()  # unchecked


def test_chip_capacity_overflow_raises():
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["cim"]["chip"]["arrays_per_chip"] = 300
    hw = _hw_from_dict(hw_dict)
    model = config.parse_config(str(VIT_HUGE_64), "VIT").model_config
    dev = cim_timing.CimDeviceModel(hw, model)
    with pytest.raises(ValueError, match="arrays_per_chip"):
        dev.validate_capacity()


def test_chip_layers_list_placement():
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["cim"]["chip"]["layers_per_chip"] = [16, 16]
    hw = _hw_from_dict(hw_dict)
    model = config.parse_config(str(VIT_HUGE_64), "VIT").model_config
    dev = cim_timing.CimDeviceModel(hw, model)
    assert dev.chip_layer_counts() == (16, 16)
    # patch embed on chip 0, head on the last chip.
    assert dev.chip_array_usage() == (16 * 12 + 1, 16 * 12 + 1)

    hw_dict["cim"]["chip"]["layers_per_chip"] = [16, 8]
    hw_bad = _hw_from_dict(hw_dict)
    dev_bad = cim_timing.CimDeviceModel(hw_bad, model)
    with pytest.raises(ValueError, match="layers_per_chip"):
        dev_bad.chip_layer_counts()


# ---------------------------------------------------------------------------
# Stage 3: integration through the real entry point (run_perf.py subprocess)
# ---------------------------------------------------------------------------

# Every run_perf subprocess writes into a PER-TEST artifact root, passed with
# --output_dir. The repo's own output/ tree is shared state: two pytest sessions
# in one checkout (or a session running beside a DSE sweep) used to overwrite
# each other's run directory mid-assertion. Nothing else passes the flag, so an
# ordinary user run still lands in <repo>/output/<MODE>, byte for byte.
_PERF_OUT = {"factory": None, "root": None}


@pytest.fixture(autouse=True)
def _perf_output_root(tmp_path_factory):
    _PERF_OUT["factory"] = tmp_path_factory
    _PERF_OUT["root"] = None
    yield
    _PERF_OUT["factory"] = None
    _PERF_OUT["root"] = None


def _perf_root():
    """This test's run_perf artifact root, created on first use."""
    if _PERF_OUT["root"] is None:
        assert _PERF_OUT["factory"] is not None, "the _perf_output_root fixture did not run"
        _PERF_OUT["root"] = _PERF_OUT["factory"].mktemp("run_perf")
    return _PERF_OUT["root"]


def _perf_out(mode, *parts):
    """A path inside this test's run_perf artifact root: <root>/<MODE>/..."""
    root = _PERF_OUT["root"]
    assert root is not None, (
        "no run_perf subprocess has run in this test, so it has no artifact root"
    )
    return root.joinpath(mode, *parts)


# Saved bit-identical baseline of a100_80GB.yaml + vit_base_inf.yaml
# (DESIGN section 5 regression gate). Committed under tests/baselines so the
# gate cannot silently rot; override with RAPID_A100_VIT_BASELINE to compare
# against a different recording.
BASELINE_A100_VIT = Path(
    os.environ.get(
        "RAPID_A100_VIT_BASELINE",
        str(PROJECT_ROOT / "tests" / "baselines" / "baseline_a100_vit_results.txt"),
    )
)


def _run_perf_subprocess(hw_path, model_path):
    """Run run_perf.py exactly as a user would, into this test's own artifact root.

    One root per TEST, not per call: the P6.2 byte-identity gate runs run_perf
    twice and compares the second run's files against bytes read from the first.
    """
    _perf_root()
    return subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "run_perf.py"),
            "--hardware_config",
            str(hw_path),
            "--model_config",
            str(model_path),
            "--output_dir",
            str(_PERF_OUT["root"]),
        ],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=600,
    )


def test_run_perf_t1_end_to_end_smoke():
    proc = _run_perf_subprocess(FWS_T1, VIT_HUGE_64)
    assert proc.returncode == 0, proc.stdout[-4000:]

    # Machine-readable report: recorded OPTIMA T1 targets (cycle counts exact,
    # times <= 0.1% relative per DESIGN section 5).
    assert _perf_out("VIT", "fws_cim_report.json").exists()
    report = json.loads(_perf_out("VIT", "fws_cim_report.json").read_text())
    assert report["device_class"] == "fws_cim"
    assert report["qk_cycles"] == 2747
    assert report["sv_cycles"] == 4471
    assert report["stages"]["S2_attention"]["qk_cycles"] == 2747
    assert report["stages"]["S2_attention"]["sv_cycles"] == 4471
    assert report["period_us"] == pytest.approx(5.12, rel=1e-3)
    assert report["bottleneck_stage"] == "S1_qkv"
    assert report["fps"] == pytest.approx(195312.5, rel=1e-3)
    assert report["block_latency_us"] == pytest.approx(25.287368421052634, rel=1e-3)
    assert report["arrays_per_layer"] == 12
    assert report["arrays_total"] == 386
    assert report["chips"] == [
        {
            "chip": 0,
            "num_layers": 32,
            "layer_range": [0, 31],
            "arrays_used": 386,
            "arrays_per_chip": 400,
            "occupancy": pytest.approx(386 / 400),
        }
    ]
    assert report["boundary"]["count"] == 0  # single chip, no p2p hops
    assert report["area_mm2"]["transformer_stack"] == pytest.approx(538.7768, rel=1e-3)

    # sa mode: the sequential per-op sum double-counts the folded QK+PV run;
    # both outputs must disclose it.
    assert "twice per layer" in report["sequential_latency_note"]

    # Readable section rides on the results txt.
    results_text = _perf_out("VIT", "LLM_inference_results.txt").read_text()
    assert "FWS-CIM spatial pipeline" in results_text
    assert "bottleneck: S1_qkv" in results_text
    assert "PARTIAL" in results_text  # energy is labeled partial
    assert "twice per layer" in results_text  # sequential double count disclosed


def test_run_perf_t1_tp2_variant_runs(tmp_path):
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["parallelism"]["tp"] = 2
    hw_path = tmp_path / "fws_cim_optima_t1_tp2.yaml"
    hw_path.write_text(yaml.safe_dump(hw_dict))

    proc = _run_perf_subprocess(hw_path, VIT_HUGE_64)
    assert proc.returncode == 0, proc.stdout[-4000:]
    report = json.loads(_perf_out("VIT", "fws_cim_report.json").read_text())
    assert report["tp"] == 2
    # heads_chip = ceil(16/2) = 8 folded into K: qk = 2*(8*80+94)-1.
    assert report["qk_cycles"] == 1467
    assert report["period_us"] > 0
    # tp >= 2 is a system of tp shard devices: the report discloses the
    # system totals (per-shard chips/area stay in the pass-1 fields; the
    # census is unsharded, so the system multiplies by tp).
    assert report["tp_shards"] == 2
    assert report["system_chips"] == 2 * len(report["chips"])
    assert report["system_area_mm2"] == pytest.approx(
        2 * report["area_mm2"]["total"], rel=1e-12
    )


def test_run_perf_gpu_results_unchanged_vs_baseline():
    # DESIGN section 5 regression gate: bit-identical, no tolerance. A missing
    # baseline FAILS (never skips): a silent skip would quietly disarm the
    # suite's only GPU bit-identity gate.
    assert BASELINE_A100_VIT.exists(), (
        f"saved A100 ViT baseline missing at {BASELINE_A100_VIT} "
        "(broken checkout, or a bad RAPID_A100_VIT_BASELINE override)"
    )
    # SEED the run directory first. Every test now runs into its own root,
    # so an unseeded "the CIM report is absent" assertion would hold on an
    # empty tmp dir no matter what run_perf did — a gate that cannot fail.
    # Planting a stale report makes the assertion prove what it names: that
    # run_perf RECREATES exp_dir and a GPU run leaves no CIM report behind.
    stale = _perf_root() / "VIT" / "fws_cim_report.json"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text('{"device_class": "stale, from a previous run"}')

    proc = _run_perf_subprocess(A100, VIT_BASE)
    assert proc.returncode == 0, proc.stdout[-4000:]
    assert _perf_out("VIT", "LLM_inference_results.txt").read_bytes() == BASELINE_A100_VIT.read_bytes()
    # GPU runs never emit the CIM report, and the stale one is GONE: run_perf
    # recreates exp_dir rather than writing over whatever it finds.
    assert not _perf_out("VIT", "fws_cim_report.json").exists()


# ---------------------------------------------------------------------------
# Pass 2 stage 1: LLM/decode admission, KV stories, new cim schema
# ---------------------------------------------------------------------------


def test_fws_cim_accepts_dense_llm_inference_with_decode():
    hw = config.parse_config(str(FWS_LLAMA7B), "hardware")
    model = config.parse_config(str(LLAMA2_7B_FWS_INF), "LLM")
    assert not model.model_config.is_vit
    assert model.model_config.decode_len == 256
    assert not model.model_config.use_flashattention
    config.validate_configs(hw, model)  # dense LLM + decode admitted


def test_fws_cim_accepts_moe_llm_inference():
    hw = config.parse_config(str(FWS_MOE), "hardware")
    model = config.parse_config(str(MOE_SMALL_FWS_INF), "LLM")
    assert model.model_config.use_moe
    assert model.model_config.num_experts == 16
    assert model.model_config.top_k == 2
    assert model.model_config.decode_len == 32
    config.validate_configs(hw, model)  # MoE LLM + decode admitted


def test_fws_cim_rejects_mla_with_seam_reason():
    # DeepSeekV3_inf is an MLA model; the gate names the setting and the
    # reason (the MLA path discards per-op GEMM times), and fires before the
    # flash gate (the YAML also sets use_flashattention: true).
    hw = config.parse_config(str(FWS_LLAMA7B), "hardware")
    model = config.parse_config(str(DEEPSEEK_V3_INF), "LLM")
    with pytest.raises(ValueError, match="mla") as excinfo:
        config.validate_configs(hw, model)
    assert "attention_type" in str(excinfo.value)


@pytest.mark.parametrize("block_kind", sorted(HYBRID_MATRIX_INF))
def test_fws_cim_rejects_hybrid_layer_plan_with_seam_reason(block_kind):
    # QIF P1.4: a model whose layer plan names a block kind the device laws do
    # not cover is rejected by name, and the message points at the plans that
    # will price it (P2-P4) instead of silently pricing it as attention.
    hw = config.parse_config(str(FWS_LLAMA7B), "hardware")
    model = config.parse_config(str(HYBRID_MATRIX_INF[block_kind]), "LLM")
    with pytest.raises(ValueError, match="fws_cim") as excinfo:
        config.validate_configs(hw, model)
    message = str(excinfo.value)
    assert "layer_plan" in message
    assert block_kind in message
    assert "P2" in message and "P4" in message


def test_fws_cim_hybrid_gate_fires_before_the_mla_gate():
    # A hybrid MLA model must name the layer plan, not the attention type:
    # the layer plan is the reason nothing can price it.
    hw = config.parse_config(str(FWS_LLAMA7B), "hardware")

    def to_mla(d):
        d["model_param"]["attention"] = {
            "attention_type": "mla",
            "num_heads": 12,
            "kv_lora_rank": 256,
            "q_lora_rank": 768,
            "qk_nope_head_dim": 64,
            "qk_rope_head_dim": 32,
            "v_head_dim": 64,
            "use_flashattention": False,
        }

    model = _model_from_yaml(HYBRID_MATRIX_INF["ssm"], mutate=to_mla)
    with pytest.raises(ValueError, match="layer_plan"):
        config.validate_configs(hw, model)


@pytest.mark.parametrize("model_path", ATTENTION_ONLY_MATRIX_INF, ids=lambda p: p.stem)
def test_fws_cim_accepts_attention_only_matrix_rows(model_path):
    # The two runs-now rows of the matrix are plain GQA transformers; a
    # layer plan whose every entry is attention must not trip the gate.
    hw = config.parse_config(str(FWS_LLAMA7B), "hardware")
    model = config.parse_config(str(model_path), "LLM")
    assert model.model_config.hybrid_block_kinds == ()
    config.validate_configs(hw, model)


def test_fws_cim_llm_flash_still_rejected():
    hw = config.parse_config(str(FWS_LLAMA7B), "hardware")
    model = config.parse_config(str(LLAMA2_7B_INF), "LLM")  # flash: true
    with pytest.raises(ValueError, match="use_flashattention"):
        config.validate_configs(hw, model)


def test_fws_cim_llm_training_still_rejected():
    hw = config.parse_config(str(FWS_LLAMA7B), "hardware")

    def to_training(d):
        d["model_param"]["run_type"] = "training"

    model = _model_from_yaml(LLAMA2_7B_FWS_INF, mutate=to_training)
    with pytest.raises(ValueError, match="run_type"):
        config.validate_configs(hw, model)


def test_fws_cim_rejects_decode_only_run():
    # decode_len == seq_len passes the generic decode_len <= seq_len gate but
    # leaves prefill_len = 0: calc_time would skip the prefill branch and
    # with it the ENTIRE FWS spatial report (the authoritative device
    # output), silently exiting 0. The gate rejects it loudly, matching the
    # DSE's rejection of the same input.
    hw = config.parse_config(str(FWS_LLAMA7B), "hardware")

    def decode_only(d):
        d["model_param"]["decode_len"] = d["model_param"]["seq_len"]

    model = _model_from_yaml(LLAMA2_7B_FWS_INF, mutate=decode_only)
    with pytest.raises(ValueError, match="prefill_len"):
        config.validate_configs(hw, model)


def test_gpu_device_with_cim_kvcache_story_warns(capsys):
    # Mirror of the ignored-cim-block warning: a CIM KV story on a GPU
    # config is inert (the GPU path never reads kvcache_type), so the
    # config gate must say so instead of silently pricing hbm_only.
    hw_dict = _load_yaml(A100)
    hw_dict.setdefault("inference", {})["kvcache_type"] = "cim_dram"
    hw = _hw_from_dict(hw_dict)
    capsys.readouterr()
    config.validate_hw_config(hw)
    out = capsys.readouterr().out
    assert "[WARNING]" in out
    assert "kvcache_type" in out and "cim_dram" in out
    assert "ignored" in out
    # hbm_only on a GPU config stays warning-free.
    hw_dict["inference"]["kvcache_type"] = "hbm_only"
    config.validate_hw_config(_hw_from_dict(hw_dict))
    assert "[WARNING]" not in capsys.readouterr().out


# --- KV stories (inference.kvcache_type) ------------------------------------


def test_pass1_templates_use_cim_sram_story():
    for path in (FWS_T1, FWS_T2, FWS_T3):
        hw = config.parse_config(str(path), "hardware")
        assert hw.inference_config.kvcache_type == "cim_sram"
        config.validate_hw_config(hw)


def test_fws_cim_rejects_hbm_only_kvcache():
    hw_dict = _load_yaml(FWS_LLAMA7B)
    hw_dict["inference"]["kvcache_type"] = "hbm_only"
    hw = _hw_from_dict(hw_dict)
    with pytest.raises(ValueError, match="no HBM"):
        config.validate_hw_config(hw)


def test_fws_cim_rejects_cim_dram_without_kv_dram_block():
    hw_dict = _load_yaml(FWS_LLAMA7B)  # has no cim.kv_dram block
    hw_dict["inference"]["kvcache_type"] = "cim_dram"
    hw = _hw_from_dict(hw_dict)
    with pytest.raises(ValueError, match="kv_dram"):
        config.validate_hw_config(hw)


def test_kvcache_type_value_set_enforced():
    hw_dict = _load_yaml(FWS_LLAMA7B)
    hw_dict["inference"]["kvcache_type"] = "flash_kv"
    with pytest.raises(ValueError, match="kvcache_type"):
        _hw_from_dict(hw_dict)


def test_kv_dram_block_parses_and_cim_sram_warns_when_unused(capsys):
    hw = config.parse_config(str(FWS_LLAMA7B_KVDRAM), "hardware")
    assert hw.inference_config.kvcache_type == "cim_dram"
    kv = hw.cim_config.kv_dram
    assert kv.capacity_bytes == 8589934592
    assert kv.bandwidth_bytes_per_s == 100000000000
    assert kv.energy_per_bit_pj == pytest.approx(2.0)
    config.validate_hw_config(hw)  # cim_dram + kv_dram: no warning
    assert "kv_dram" not in capsys.readouterr().out

    hw_dict = _load_yaml(FWS_LLAMA7B_KVDRAM)
    hw_dict["inference"]["kvcache_type"] = "cim_sram"
    hw_sram = _hw_from_dict(hw_dict)
    config.validate_hw_config(hw_sram)  # warns (kv_dram ignored), never raises
    assert "kv_dram" in capsys.readouterr().out


def test_kv_dram_rejects_unit_strings():
    # convert() rewrites "<int> <Unit>" strings elsewhere in the YAML; any
    # string that survives into the cim block must be rejected.
    hw_dict = _load_yaml(FWS_LLAMA7B_KVDRAM)
    hw_dict["cim"]["kv_dram"]["capacity_bytes"] = "8 GiB"
    with pytest.raises(ValueError, match="plain number"):
        _hw_from_dict(hw_dict)


# --- cim.chip additions: layers_per_chip "auto", moe_expert_parallel --------


def test_layers_per_chip_auto_parses():
    hw_dict = _load_yaml(FWS_LLAMA7B)
    hw_dict["cim"]["chip"]["layers_per_chip"] = "auto"
    hw = _hw_from_dict(hw_dict)
    assert hw.cim_config.chip.layers_per_chip == "auto"
    config.validate_hw_config(hw)


def test_layers_per_chip_auto_requires_arrays_per_chip():
    hw_dict = _load_yaml(FWS_LLAMA7B)
    hw_dict["cim"]["chip"]["layers_per_chip"] = "auto"
    hw_dict["cim"]["chip"]["arrays_per_chip"] = 0
    with pytest.raises(ValueError, match="arrays_per_chip"):
        _hw_from_dict(hw_dict)


def test_layers_per_chip_other_strings_rejected():
    hw_dict = _load_yaml(FWS_LLAMA7B)
    hw_dict["cim"]["chip"]["layers_per_chip"] = "all"
    with pytest.raises(ValueError, match="layers_per_chip"):
        _hw_from_dict(hw_dict)


def test_moe_expert_parallel_parses_and_rejects_zero():
    # Absent -> default 1 (the pass-1 templates never set it).
    hw = config.parse_config(str(FWS_T1), "hardware")
    assert hw.cim_config.chip.moe_expert_parallel == 1

    hw_dict = _load_yaml(FWS_MOE)
    hw_dict["cim"]["chip"]["moe_expert_parallel"] = 4
    hw4 = _hw_from_dict(hw_dict)
    assert hw4.cim_config.chip.moe_expert_parallel == 4

    hw_dict["cim"]["chip"]["moe_expert_parallel"] = 0
    with pytest.raises(ValueError, match="moe_expert_parallel"):
        _hw_from_dict(hw_dict)


# --- cim.dse: parse-only candidate space for the DSE tool -------------------


def test_dse_block_parses_structurally():
    hw_dict = _load_yaml(FWS_LLAMA7B)
    hw_dict["cim"]["dse"] = {
        "mux_candidates": [1, 2, 4, 8, 16],
        "variants": [
            {
                "adc_mux": 4,
                "cols_adc": 320,
                "energy_per_vec_pj": 9534.9389,
                "area_mm2_per_array": 1.403065,
            },
            {
                "adc_mux": 16,
                "cols_adc": 80,
                "rows": 1280,
                "energy_per_vec_pj": 12470.0063,
                "area_mm2_per_array": 1.135296,
            },
        ],
        "tp_candidates": "auto",
        "max_chips": 0,
    }
    hw = _hw_from_dict(hw_dict)
    dse = hw.cim_config.dse
    assert dse.mux_candidates == (1, 2, 4, 8, 16)
    assert len(dse.variants) == 2
    v0, v1 = dse.variants
    assert (v0.adc_mux, v0.cols_adc) == (4, 320)
    assert v0.rows is None  # inherits cim.analog.rows
    assert v0.energy_per_vec_pj == pytest.approx(9534.9389)
    assert (v1.adc_mux, v1.cols_adc, v1.rows) == (16, 80, 1280)
    assert dse.tp_candidates == "auto"
    assert dse.max_chips == 0
    config.validate_hw_config(hw)  # parse-only block; nothing consumes it yet

    hw_dict["cim"]["dse"]["tp_candidates"] = [1, 2, 4]
    hw_list = _hw_from_dict(hw_dict)
    assert hw_list.cim_config.dse.tp_candidates == (1, 2, 4)

    hw_dict["cim"]["dse"]["tp_candidates"] = "divisors"
    with pytest.raises(ValueError, match="tp_candidates"):
        _hw_from_dict(hw_dict)

    hw_dict["cim"]["dse"]["tp_candidates"] = "auto"
    hw_dict["cim"]["dse"]["variants"] = [{"cols_adc": 320}]  # missing adc_mux
    with pytest.raises(ValueError, match="adc_mux"):
        _hw_from_dict(hw_dict)


def test_dse_absent_stays_none():
    hw = config.parse_config(str(FWS_LLAMA7B), "hardware")
    assert hw.cim_config.dse is None


# --- new hardware templates parse and validate ------------------------------


def test_fws_llama7b_templates_parse_and_validate():
    hw = config.parse_config(str(FWS_LLAMA7B), "hardware")
    assert hw.device_class == "fws_cim"
    assert hw.cim_config.analog.rows == 4096
    assert hw.cim_config.analog.cols == 4096  # cols_adc * adc_mux
    assert hw.cim_config.chip.layers_per_chip == 8
    assert hw.cim_config.chip.moe_expert_parallel == 1
    assert hw.inference_config.kvcache_type == "cim_sram"
    config.validate_hw_config(hw)

    hw_kv = config.parse_config(str(FWS_LLAMA7B_KVDRAM), "hardware")
    assert hw_kv.inference_config.kvcache_type == "cim_dram"
    assert hw_kv.cim_config.kv_dram is not None
    config.validate_hw_config(hw_kv)


def test_fws_moe_template_parses_and_validates():
    hw = config.parse_config(str(FWS_MOE), "hardware")
    assert hw.cim_config.analog.rows == 1024
    assert hw.cim_config.chip.arrays_per_chip == 300
    assert hw.cim_config.chip.moe_expert_parallel == 1
    config.validate_hw_config(hw)


# ---------------------------------------------------------------------------
# Pass 2 stage 2: device-model laws (DESIGN2 section 1)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cim_llama():
    """fws_cim_llama7b + llama2_7b_fws_inf: H=4096, 32 MHA heads, d=128,
    vocab 32000; analog rows=4096 cols=4096 mux=4 slice=2 @100MHz
    (vec_latency 80 ns); fabric R=32 C=64 fd=96 @0.95GHz."""
    return _cim_device(FWS_LLAMA7B, LLAMA2_7B_FWS_INF)


@pytest.fixture(scope="module")
def cim_moe():
    """fws_cim_moe + moe_small_fws_inf: H=1024, 16 q heads / 4 kv heads
    (GQA, shared_heads=4), d=64, E=16 top_k=2 I_moe=512 shared=1 gated,
    layer 0 dense (I=2048), layers 1..11 MoE; rows=1024 cols=1024."""
    return _cim_device(FWS_MOE, MOE_SMALL_FWS_INF)


def _moe_hand_params(**overrides):
    """DESIGN2 section 4 hand case: E=8, top_k=2, I_moe=512, shared=1, gated."""
    kwargs = dict(
        hidden_dim=1024,
        intermediate_size=2048,
        num_layers=4,
        num_heads=16,
        kv_heads=4,
        head_dim=64,
        seq_len=256,
        batch_size=2,
        gated_mlp=True,
        patch_dim=0,
        num_classes=0,
        num_experts=8,
        top_k=2,
        moe_intermediate_size=512,
        n_shared_experts=1,
        expert_imbalance_factor=1.5,
        moe_layer_mask=(False, True, True, True),
        vocab_size=32000,
    )
    kwargs.update(overrides)
    return cim_timing.CimModelParams(**kwargs)


def _moe_hand_device(hw_mutate=None, **param_overrides):
    hw_dict = _load_yaml(FWS_MOE)
    if hw_mutate is not None:
        hw_mutate(hw_dict)
    hw = _hw_from_dict(hw_dict)
    return cim_timing.CimDeviceModel(hw, _moe_hand_params(**param_overrides))


# --- 1.1: call-dims attention law is identical to the pass-1 law -------------


def test_attention_call_dims_equivalent_to_old_law_on_t1_t2_t3(cim_t1, cim_t2, cim_t3):
    # Old law (pass 1): QK = sa(S, S, d*h_rep), PV = sa(S, d, S*h_rep),
    # total = max + fill_drain. New law: for a call (m, k, n) the pair is
    # sa(m, n, k*h_rep) and sa(m, k, n*h_rep). MHA prefill score call
    # (m=S, k=d, n=S) must reproduce the recorded cycles EXACTLY.
    cases = (
        (cim_t1, 64, 80, 2747, 4471),    # T1: recorded QK/PV
        (cim_t2, 64, 88, 3003, 4471),    # T2
        (cim_t3, 196, 80, 38471, 45219), # T3
    )
    for dev, s, d, qk, pv in cases:
        old = dev.attention_timing(seq_len=s, head_dim=d)
        score = dev.attention_call_timing(m=s, k=d, n=s)
        assert (score.qk_cycles, score.pv_cycles) == (qk, pv)
        assert score.total_cycles == old.total_cycles == max(qk, pv) + 96
        assert score.sa_time_s == old.sa_time_s
        assert score.softmax_cycles == old.softmax_cycles
        # Output-orientation call (m=S, k=S, n=d): labels swap, max is
        # symmetric under k<->n, so the folded total is identical.
        out = dev.attention_call_timing(m=s, k=s, n=d)
        assert (out.qk_cycles, out.pv_cycles) == (pv, qk)
        assert out.total_cycles == score.total_cycles
        assert out.sa_time_s == score.sa_time_s


def test_gqa_prefill_score_call_folds_shared_heads(cim_moe):
    # GQA prefill score call: m = S*shared_heads = 512*4 = 2048, k = d = 64,
    # n = S = 512; heads_chip = kv_heads = 4 (tp=1).
    # QK = ceil(2048/32)*ceil(512/64)*(64*4+94) - 1 = 64*8*350 - 1 = 179199
    # PV = ceil(2048/32)*ceil(64/64)*(512*4+94) - 1 = 64*1*2142 - 1 = 137087
    att = cim_moe.prefill_attention_timing()
    assert att.heads_chip == 4
    assert att.qk_cycles == 179199
    assert att.pv_cycles == 137087
    assert att.total_cycles == 179199 + 96


# --- 1.4: decode attention closed form and monotone context growth -----------


def test_decode_attention_closed_form_and_monotone_growth(cim_llama):
    # DESIGN2 section 4 shape: sh=1, d=128, kv_heads=32, tp=1, R=32, C=64.
    # score call m=sh=1, k=128, n=ctx; K+R+C-2 = K+94; fd = 96:
    #  ctx=129:  qk = 1*ceil(129/64)*(128*32+94)-1 = 3*4190-1  = 12569
    #            pv = 1*ceil(128/64)*(129*32+94)-1 = 2*4222-1  = 8443
    #  ctx=512:  qk = 8*4190-1  = 33519; pv = 2*(16384+94)-1   = 32955
    #  ctx=4096: qk = 64*4190-1 = 268159; pv = 2*(131072+94)-1 = 262331
    expected = {
        129: (12569, 8443, 12569 + 96),
        512: (33519, 32955, 33519 + 96),
        4096: (268159, 262331, 268159 + 96),
    }
    totals = []
    for ctx, (qk, pv, total) in expected.items():
        att = cim_llama.decode_attention_timing(ctx, batch_size=1)
        assert (att.qk_cycles, att.pv_cycles, att.total_cycles) == (qk, pv, total)
        assert att.sa_time_s == pytest.approx(total / 0.95e9, rel=1e-12)
        # decode softmax is B-scaled: tokens_q = B*sh = 1;
        # cycles = 20 + ceil(1*32/1) - 1 = 51.
        assert att.softmax_cycles == 51
        totals.append(att.total_cycles)
    assert totals[0] < totals[1] < totals[2]  # monotone in context


def test_decode_softmax_is_b_scaled(cim_llama):
    # tokens_q = B*shared_heads = 8*1; cycles = 20 + ceil(8*32/1) - 1 = 275.
    att = cim_llama.decode_attention_timing(512, batch_size=8)
    assert att.softmax_cycles == 20 + math.ceil(8 * 32 / 1) - 1 == 275


def test_decode_attention_sa_is_b_folded(cim_llama):
    # The wavefront carries B streams, so the SA prices all B: streams fold
    # into the contraction exactly like heads. ctx=512, B=4:
    #   qk = 1*ceil(512/64)*(128*32*4+94)-1 = 8*16478-1 = 131823
    #   pv = 1*ceil(128/64)*(512*32*4+94)-1 = 2*65630-1 = 131259
    #   total = 131823 + 96 = 131919.
    att1 = cim_llama.decode_attention_timing(512, batch_size=1)
    att4 = cim_llama.decode_attention_timing(512, batch_size=4)
    assert att4.qk_cycles == 131823 == cim_llama.sa_cycles(1, 512, 128 * 32 * 4)
    assert att4.pv_cycles == 131259 == cim_llama.sa_cycles(1, 128, 512 * 32 * 4)
    assert att4.total_cycles == 131823 + 96
    assert att4.sa_time_s > att1.sa_time_s  # B streams cost more silicon time
    # B=1 stays the pass-1 closed form (33519 + 96 cycles at ctx=512).
    assert att1.total_cycles == 33519 + 96


def test_prefill_attention_sa_is_b_folded(cim_llama):
    # LLM prefill wavefront = B streams: S2 must price the same B streams
    # whose tokens S1/S3/S4/S5 price (tokens = B*S). S=1792, B=4, MHA:
    #   qk = 56*28*(128*32*4+94)-1 = 1568*16478-1 = 25837503
    #   pv = 56*2*(1792*32*4+94)-1 = 112*229470-1 = 25700639
    # softmax defaults to the B-scaled query rows: tokens_q = B*S = 7168,
    # cycles = 20 + ceil(7168*32/1) - 1 = 229395.
    att = cim_llama.prefill_attention_timing(seq_len=1792, streams=4)
    assert att.qk_cycles == 25837503
    assert att.pv_cycles == 25700639
    assert att.total_cycles == 25837503 + 96
    assert att.softmax_cycles == 229395
    # streams=1 reproduces the pass-1 law bit-exactly.
    att1 = cim_llama.prefill_attention_timing(seq_len=1792, streams=1)
    assert att1.qk_cycles == 6569919
    assert att1.total_cycles == 6569919 + 96
    # layer_stage_times carries streams to S2 (the report convention).
    stages = cim_llama.layer_stage_times(1792, tokens=4 * 1792, streams=4)
    assert stages["S2_attention"] == pytest.approx(att.stage_time_s, rel=1e-12)


def test_decode_sustained_throughput_kv_bandwidth_cap(cim_llama):
    # All resident wavefronts read the ONE KV tier concurrently, so the
    # sustained rate can never imply more than tier_bw / bytes-per-token.
    # Resident rate = 2 wavefronts * 4 / 9 us = 888888.9 tok/s; the tier
    # (1e9 B/s, 1e6 B KV read per generated token) sustains 1000 tok/s.
    r = cim_llama.decode_sustained_throughput(
        9e-6, 2e-6, 8, batch_size=4,
        kv_read_bandwidth_bytes_per_s=1e9, kv_bytes_per_token=1e6,
    )
    assert r.limiting_factor == "kv_bandwidth"
    assert r.tokens_per_s == pytest.approx(1e9 / 1e6, rel=1e-12)
    # A tier fast enough not to bind leaves the kv_capacity result intact.
    r2 = cim_llama.decode_sustained_throughput(
        9e-6, 2e-6, 8, batch_size=4,
        kv_read_bandwidth_bytes_per_s=1e13, kv_bytes_per_token=1e6,
    )
    assert r2.limiting_factor == "kv_capacity"
    assert r2.tokens_per_s == pytest.approx(2 * 4 / 9e-6, rel=1e-12)


# --- 1.2: decode/MoE op-name classification --------------------------------


def test_decode_names_map_to_intended_laws(cim_llama, capsys):
    # Dense decode weight ops arrive per-stream (m=1, B streams outside the
    # call, never multiplied back by the caller): M = B. vec_latency = 80 ns.
    tc = _fake_tc(batch_size=4, kv_heads=32)
    vec = cim_llama.vec_latency_s
    for name in (
        "decode_qkv_proj_f",
        "decode_output_projection_f",
        "decode_ffn1_f",
        "decode_ffn2_f",
    ):
        assert cim_llama.price_gemm(name, 1, 4096, 4096, tc) == pytest.approx(4 * vec, rel=1e-12)
    # Prefill qkv_projection_* still matches (qkv_proj is a prefix of it).
    assert cim_llama.price_gemm("qkv_projection_f", 64, 4096, 12288, tc) == pytest.approx(64 * vec, rel=1e-12)
    # router: weight law at M as received (name carries no decode_ prefix).
    assert cim_llama.price_gemm("router_f", 256, 4096, 16, tc) == pytest.approx(256 * vec, rel=1e-12)
    # MoE bucket ops are token-true: never rescaled by B, even under decode_.
    assert cim_llama.price_gemm("decode_ffn1_f_hot", 3, 4096, 1024, tc) == pytest.approx(3 * vec, rel=1e-12)
    assert cim_llama.price_gemm("ffn1_f_shared", 7, 4096, 1024, tc) == pytest.approx(7 * vec, rel=1e-12)
    # The balanced ("uniform") bucket carries its suffix too: its per-expert
    # M already folds B, so the decode M = B rule must not re-scale it.
    assert cim_llama.price_gemm("decode_ffn1_f_uniform", 1, 4096, 1024, tc) == pytest.approx(1 * vec, rel=1e-12)
    # The decode lm_head GEMM carries the decode_ prefix: M = B (DESIGN2 1.5).
    assert cim_llama.price_gemm("decode_linear_softmax_f", 1, 4096, 32000, tc) == pytest.approx(4 * vec, rel=1e-12)
    # moe_dispatch / moe_combine never reach get_gemm_time; if they ever do,
    # they are priced silently (analog law), NEVER warned about.
    cim_llama.price_gemm("moe_dispatch_f", 8, 1, 4096, tc)
    cim_llama.price_gemm("moe_combine_f", 8, 1, 4096, tc)
    assert "[WARNING]" not in capsys.readouterr().out


def test_decode_attention_time_grows_with_context(cim_llama):
    # Kill the pass-1 silent-fallback bug class forever: decode attention is
    # priced by the call-dims act law (context-dependent), not the analog
    # M-law fallback.
    tc = _fake_tc(batch_size=1, kv_heads=32)
    t129 = cim_llama.price_gemm("decode_attention_score_f", 1, 128, 129, tc)
    t512 = cim_llama.price_gemm("decode_attention_score_f", 1, 128, 512, tc)
    t4096 = cim_llama.price_gemm("decode_attention_score_f", 1, 128, 4096, tc)
    assert t129 < t512 < t4096
    # N_mult compensation: caller multiplies by B*kv_heads = 32.
    assert t512 * 32 == pytest.approx((33519 + 96) / 0.95e9, rel=1e-12)
    # Output call (m=1, k=ctx, n=d) is the same folded stage total.
    t512_out = cim_llama.price_gemm("decode_attention_output_f", 1, 512, 128, tc)
    assert t512_out == pytest.approx(t512, rel=1e-12)


# --- 1.3: MoE array counts and stage laws ------------------------------------


def test_moe_layer_arrays_hand_case():
    # rows=1024, cols=1024. H=1024, GQA q16/kv4 d=64, E=8, I_moe=512, gated,
    # shared=1:
    #   qkv: N=(16+2*4)*64=1536 -> ceil(1024/1024)*ceil(1536/1024) = 2
    #   o_proj: arrays(1024,1024) = 1;  router: arrays(1024, 8) = 1
    #   per-expert pair: ffn1 FUSED arrays(1024, 2*512=1024)=1 + ffn2
    #   arrays(512,1024)=1 -> 2; routed: 8*1 + 8*1; shared: 1 + 1.
    dev = _moe_hand_device()
    assert dev.moe_layer_stage_arrays() == {
        "qkv": 2,
        "o_proj": 1,
        "router": 1,
        "ffn1_routed": 8,
        "ffn2_routed": 8,
        "ffn1_shared": 1,
        "ffn2_shared": 1,
    }
    assert dev.arrays_per_moe_layer() == 22
    assert dev.moe_routed_arrays() == 16
    assert dev.moe_expert_ffn_arrays() == 2
    # Dense layer (I=2048, gated FUSED): 2 + 1 + arrays(1024,4096)=4 +
    # arrays(2048,1024)=2 = 9. Stack: 1 dense + 3 MoE = 9 + 3*22 = 75.
    assert dev.arrays_per_layer() == 9
    assert dev.transformer_stack_arrays() == 75


def test_moe_template_layer_arrays(cim_moe):
    # E=16 point (fws_cim_moe.yaml comment, corrected for GQA qkv=2):
    # MoE layer 2+1+1+16+16+1+1 = 38; dense layer 0 (I=2048) = 9.
    assert cim_moe.arrays_per_moe_layer() == 38
    assert cim_moe.arrays_per_layer() == 9
    assert cim_moe.transformer_stack_arrays() == 9 + 11 * 38  # == 427
    # lm_head endpoint: ceil(1024/1024)*ceil(32000/1024) = 32, last chip.
    assert cim_moe.endpoint_arrays() == {"lm_head": 32}
    # 6+6 layer split: chip0 = dense 9 + 5*38 = 199; chip1 = 6*38 + 32 = 260.
    assert cim_moe.chip_array_usage() == (199, 260)
    cim_moe.validate_capacity()  # 260 <= 300


def test_moe_routed_stage_law_includes_alpha():
    # tokens_owner = B*S = 2*256 = 512; alpha=1.5, top_k=2, E=8:
    # tokens_hot = ceil(512*2*1.5/8) = 192 -> T_routed = 192 * 80 ns.
    # shared: 512 * 80 ns; stage = max = shared here.
    dev = _moe_hand_device()
    vec = dev.vec_latency_s
    assert dev.moe_tokens_hot(512) == 192
    assert dev.moe_routed_ffn_time(512) == pytest.approx(192 * vec, rel=1e-12)
    assert dev.moe_shared_ffn_time(512) == pytest.approx(512 * vec, rel=1e-12)
    assert dev.moe_ffn_stage_time(512) == pytest.approx(512 * vec, rel=1e-12)
    # alpha=1 (balanced): tokens_hot = ceil(512*2/8) = 128.
    dev_bal = _moe_hand_device(expert_imbalance_factor=1.0)
    assert dev_bal.moe_tokens_hot(512) == 128
    # No shared experts: the routed law binds the stage.
    dev_ns = _moe_hand_device(n_shared_experts=0)
    assert dev_ns.moe_shared_ffn_time(512) == 0.0
    assert dev_ns.moe_ffn_stage_time(512) == pytest.approx(192 * vec, rel=1e-12)


def test_moe_identity_reduces_to_dense_ffn_stage():
    # E=1, top_k=1, alpha=1, shared=0: tokens_hot = ceil(T*1*1/1) = T, so the
    # MoE FFN stage time equals the dense analog law EXACTLY (identity).
    dev = _moe_hand_device(
        num_experts=1, top_k=1, n_shared_experts=0, expert_imbalance_factor=1.0
    )
    for tokens in (1, 64, 512, 1000):
        assert dev.moe_ffn_stage_time(tokens) == dev.analog_gemm_time(tokens)
    # num_experts == 1 also collapses the layer classes to dense.
    assert dev.layer_class_mask() == (False, False, False, False)
    assert dev.transformer_stack_arrays() == 4 * dev.arrays_per_layer()


def test_moe_expert_parallel_divides_arrays_and_dispatch_time():
    def set_k(d):
        d["cim"]["chip"]["moe_expert_parallel"] = 4

    dev_k1 = _moe_hand_device()
    dev_k4 = _moe_hand_device(hw_mutate=set_k)
    # Expert pool: 3 MoE layers * 4 chips; each holds ceil(16/4) = 4 routed
    # arrays; the layer chip keeps 22 - 16 = 6 arrays for its MoE layers.
    assert dev_k1.moe_expert_pool() == (0, 0)
    assert dev_k4.moe_expert_pool() == (12, 4)
    # fws_cim_moe.yaml maps layers 6+... but this 4-layer model fits chip 0:
    # dense 9 + 3 MoE; k=1 keeps routed arrays on the layer chip.
    assert dev_k1.chip_array_usage()[0] >= dev_k4.chip_array_usage()[0]
    # Dispatch bytes (each way): tokens_owner*top_k*H*act = 512*2*1024*2
    # = 2097152; base p2p on (1 GB/s, 1 us) = 2097152/1e9 + 1e-6 s.
    bytes_each_way = dev_k1.moe_dispatch_bytes(512, 2)
    assert bytes_each_way == 2097152.0
    base = bytes_each_way / 1e9 + 1e-6
    assert dev_k1.moe_dispatch_time(512, 2, 1e9, 1e-6) == pytest.approx(base, rel=1e-12)
    assert dev_k4.moe_dispatch_time(512, 2, 1e9, 1e-6) == pytest.approx(base / 4, rel=1e-12)


def test_moe_layer_stage_times_table():
    # tokens_owner = 512: S1/S3/S4_router = 512*80ns; FFN stages = max(routed,
    # shared) = 512*80ns (shared-bound at alpha=1.5).
    dev = _moe_hand_device()
    stages = dev.moe_layer_stage_times(seq_len=256, tokens_owner=512)
    t_owner = dev.analog_gemm_time(512)
    assert stages["S1_qkv"] == pytest.approx(t_owner, rel=1e-12)
    assert stages["S3_o_proj"] == pytest.approx(t_owner, rel=1e-12)
    assert stages["S4_router"] == pytest.approx(t_owner, rel=1e-12)
    assert stages["S5_ffn1_moe"] == pytest.approx(dev.moe_ffn_stage_time(512), rel=1e-12)
    assert stages["S6_ffn2_moe"] == stages["S5_ffn1_moe"]
    # S2 uses the GQA prefill score call at seq_len, not tokens_owner.
    assert stages["S2_attention"] == dev.prefill_attention_timing(seq_len=256).stage_time_s


# --- 1.4: KV stories ---------------------------------------------------------


def test_kv_bytes_mirror_memory_estimation_sharded_law(cim_llama):
    # memory_estimation.py per-device GQA law: B * ceil(kv/tp) * d * tokens *
    # prec * 2. Llama: kv=32, d=128. ctx=1000, prec=2, tp=2:
    # per-stream-layer = 2*ceil(32/2)*128*1000*2 = 8192000 bytes.
    assert cim_llama.kv_bytes_per_stream_layer(1000, 2, tp=2) == 8192000.0
    # Whole model (32 layers): 262144000; B=3 step read (per layer): 24576000.
    assert cim_llama.kv_bytes_per_stream(1000, 2, tp=2) == 262144000.0
    assert cim_llama.kv_read_bytes(1000, 2, batch_size=3, tp=2) == 24576000.0
    # The unsharded traffic helper would overstate by tp — guard the factor.
    assert cim_llama.kv_bytes_per_stream_layer(1000, 2, tp=1) == 2 * 8192000.0


def test_kv_max_streams_and_max_context(cim_llama):
    # capacity 8 GiB = 8589934592.
    # max_streams(ctx=1000, prec=2, tp=2) = floor(8589934592/262144000) = 32.
    assert cim_llama.kv_max_streams(8589934592, 1000, 2, tp=2) == 32
    # max_context(B=4, prec=2, tp=1): per token = 4*32*(2*32*128*2)
    # = 4*32*16384 = 2097152 -> floor(8589934592/2097152) = 4096.
    assert cim_llama.kv_max_context(8589934592, 2, batch_size=4, tp=1) == 4096


def test_kv_story_bandwidth_resolution():
    hw_dict = _load_yaml(FWS_LLAMA7B_KVDRAM)
    hw = _hw_from_dict(hw_dict)
    model = config.parse_config(str(LLAMA2_7B_FWS_INF), "LLM").model_config
    dev = cim_timing.CimDeviceModel(hw, model)
    # cim_dram reads the cim.kv_dram block; cim_sram takes the stub tier bw.
    assert dev.kv_story_bandwidth("cim_dram") == 100000000000.0
    assert dev.kv_story_bandwidth("cim_sram", 123.0) == 123.0
    with pytest.raises(ValueError, match="no HBM"):
        dev.kv_story_bandwidth("hbm_only")
    with pytest.raises(ValueError, match="cim.kv_dram"):
        _cim_device(FWS_LLAMA7B, LLAMA2_7B_FWS_INF).kv_story_bandwidth("cim_dram")


def test_decode_s2_kv_bandwidth_binds_when_low(cim_llama):
    # ctx=512, B=1, tp=1, prec=2: kv_read = 2*32*128*512*2 = 8388608 bytes.
    # kv_bw = 1 GB/s -> 8.388608 ms >> sa (33615/0.95e9 = 35.4 us) >> softmax.
    s2 = cim_llama.decode_s2_timing(512, 1e9, 2, batch_size=1)
    assert s2.kv_read_bytes == 8388608.0
    assert s2.bound == "kv_read"
    assert s2.stage_time_s == pytest.approx(8388608.0 / 1e9, rel=1e-12)
    assert s2.attention.sa_time_s == pytest.approx((33519 + 96) / 0.95e9, rel=1e-12)
    # A fast KV tier hands the stage back to the fabric law.
    s2_fast = cim_llama.decode_s2_timing(512, 1e15, 2, batch_size=1)
    assert s2_fast.bound == "sa"
    assert s2_fast.stage_time_s == pytest.approx(s2_fast.attention.sa_time_s, rel=1e-12)


def test_cim_dram_energy_helper():
    hw = _hw_from_dict(_load_yaml(FWS_LLAMA7B_KVDRAM))  # 2 pJ/bit
    model = config.parse_config(str(LLAMA2_7B_FWS_INF), "LLM").model_config
    dev = cim_timing.CimDeviceModel(hw, model)
    assert dev.kv_dram_energy_pj(1000) == pytest.approx(1000 * 8 * 2.0, rel=1e-12)
    # No kv_dram block -> zero (cim_sram story keeps KV in the stub tier).
    dev_sram = _cim_device(FWS_LLAMA7B, LLAMA2_7B_FWS_INF)
    assert dev_sram.kv_dram_energy_pj(1000) == 0.0


# --- 1.5: model-shaped endpoints and boundary helpers ------------------------


def test_lm_head_arrays_law(cim_llama):
    # H=4096, V=32000, rows=4096, cols_adc*mux=4096:
    # ceil(4096/4096) * ceil(32000/4096) = 1 * 8 = 8.
    assert cim_llama.endpoint_arrays() == {"lm_head": 8}
    # Llama per-layer arrays under the FUSED gated law (uses_gated_mlp source,
    # not model_type=='vit_dinov3'): ffn1 = ceil(2*11008/4096) = 6.
    assert cim_llama.per_layer_stage_arrays() == {"qkv": 3, "o_proj": 1, "ffn1": 6, "ffn2": 3}
    assert cim_llama.arrays_per_layer() == 13
    # 4 chips x 8 layers: last chip carries the lm_head (8*13 + 8 = 112).
    assert cim_llama.chip_array_usage() == (104, 104, 104, 112)


def test_lm_head_stage_times_prefill_and_decode(cim_llama):
    # Prefill M = B*S = 4*2048; decode M = B = 4; vec_latency = 80 ns.
    vec = cim_llama.vec_latency_s
    prefill = cim_llama.endpoint_stage_times()
    assert list(prefill) == ["lm_head"]
    assert prefill["lm_head"] == pytest.approx(4 * 2048 * vec, rel=1e-12)
    decode = cim_llama.endpoint_stage_times(decode=True)
    assert decode["lm_head"] == pytest.approx(4 * vec, rel=1e-12)
    # The embedding lookup is a report note, never an endpoint stage.
    assert "embedding" not in prefill
    assert cim_llama.embedding_note is not None
    assert "disable_embedding_unembedding" in cim_llama.embedding_note


def test_vit_endpoints_unchanged_and_no_lm_head(cim_t1):
    # ViT keeps the pass-1 endpoint shape; vocab_size aliases num_classes and
    # must NOT create an lm_head entry.
    assert cim_t1.endpoint_arrays() == {"patch_embed": 1, "vit_head": 1}
    assert cim_t1.embedding_note is None


def test_boundary_bytes_and_p2p_helper(cim_llama):
    # tokens * hidden * act_bytes: prefill B*S = 8192 tokens at 2 bytes.
    assert cim_llama.boundary_bytes(8192, 2) == 8192 * 4096 * 2
    assert cim_timing.CimDeviceModel.p2p_time_s(1e6, 1e9, 1e-6) == pytest.approx(
        1e6 / 1e9 + 1e-6, rel=1e-12
    )
    assert cim_timing.CimDeviceModel.p2p_time_s(0, 1e9, 1e-6) == 0.0
    assert cim_timing.CimDeviceModel.p2p_time_s(1e6, 0, 1e-6) == float("inf")


def _llama_auto_device(arrays_per_chip=None):
    hw_dict = _load_yaml(FWS_LLAMA7B)
    hw_dict["cim"]["chip"]["layers_per_chip"] = "auto"
    if arrays_per_chip is not None:
        hw_dict["cim"]["chip"]["arrays_per_chip"] = arrays_per_chip
    hw = _hw_from_dict(hw_dict)
    model = config.parse_config(str(LLAMA2_7B_FWS_INF), "LLM").model_config
    return cim_timing.CimDeviceModel(hw, model)


def test_layers_per_chip_auto_derives_greedy_split():
    # Greedy capacity-first packing (DESIGN2 section 2): 13 arrays/layer at
    # capacity 120 packs 9 layers/chip (117); the last chip carries the
    # remaining 5 layers + the 8 lm_head arrays (73). Layer order preserved.
    dev = _llama_auto_device()
    assert dev.layers_per_chip_is_auto
    assert dev.chip_layer_counts() == (9, 9, 9, 5)
    assert dev.chip_array_usage() == (117, 117, 117, 73)
    dev.validate_capacity()  # feasible by construction


def test_layers_per_chip_auto_tail_endpoint_forces_new_chip():
    # Capacity 26 packs 2 layers/chip, but the LAST layer must fit next to
    # the 8 lm_head arrays: 13+13+8 > 26 pushes it to its own chip (13+8=21).
    dev = _llama_auto_device(arrays_per_chip=26)
    counts = dev.chip_layer_counts()
    assert sum(counts) == 32
    assert counts[-2:] == (1, 1)
    assert all(c == 2 for c in counts[:-2])
    assert dev.chip_array_usage()[-1] == 13 + 8
    dev.validate_capacity()


def test_layers_per_chip_auto_single_layer_overflow_raises():
    # One layer alone (13 arrays) exceeds capacity 12: actionable hard error.
    dev = _llama_auto_device(arrays_per_chip=12)
    with pytest.raises(ValueError, match="auto"):
        dev.chip_layer_counts()


def test_layers_per_chip_auto_requires_capacity_at_device_model():
    # Config parsing already rejects auto with arrays_per_chip == 0; the
    # device model repeats the check for directly-constructed instances.
    hw_dict = _load_yaml(FWS_LLAMA7B)
    hw = _hw_from_dict(hw_dict)
    object.__setattr__(hw.cim_config.chip, "layers_per_chip", "auto")
    object.__setattr__(hw.cim_config.chip, "arrays_per_chip", 0)
    model = config.parse_config(str(LLAMA2_7B_FWS_INF), "LLM").model_config
    dev = cim_timing.CimDeviceModel(hw, model)
    with pytest.raises(ValueError, match="arrays_per_chip"):
        dev.chip_layer_counts()


def test_layers_per_chip_auto_moe_hand_case():
    # Hand MoE stack (dense 9 + 3 MoE x 22 + lm_head 32 = 107) fits one chip
    # at capacity 300; moe_expert_parallel=4 moves the routed arrays (16) to
    # the expert pool, shrinking the layer chip to 9 + 3*6 + 32 = 59.
    def set_auto(d):
        d["cim"]["chip"]["layers_per_chip"] = "auto"

    dev = _moe_hand_device(hw_mutate=set_auto)
    assert dev.chip_layer_counts() == (4,)
    assert dev.chip_array_usage() == (107,)

    def set_auto_k4(d):
        d["cim"]["chip"]["layers_per_chip"] = "auto"
        d["cim"]["chip"]["moe_expert_parallel"] = 4

    dev_k4 = _moe_hand_device(hw_mutate=set_auto_k4)
    assert dev_k4.chip_layer_counts() == (4,)
    assert dev_k4.chip_array_usage() == (9 + 3 * 6 + 32,)
    assert dev_k4.moe_expert_pool() == (12, 4)


# ---------------------------------------------------------------------------
# Pass 2 stage 3: report laws (all-class period, decode tables, KV capacity)
# ---------------------------------------------------------------------------


def test_kv_story_capacity_resolution():
    hw = _hw_from_dict(_load_yaml(FWS_LLAMA7B_KVDRAM))
    model = config.parse_config(str(LLAMA2_7B_FWS_INF), "LLM").model_config
    dev = cim_timing.CimDeviceModel(hw, model)
    assert dev.kv_story_capacity("cim_dram") == 8589934592.0
    assert dev.kv_story_capacity("cim_sram", 123.0) == 123.0
    with pytest.raises(ValueError, match="no HBM"):
        dev.kv_story_capacity("hbm_only")
    with pytest.raises(ValueError, match="cim.kv_dram"):
        _cim_device(FWS_LLAMA7B, LLAMA2_7B_FWS_INF).kv_story_capacity("cim_dram")


def test_all_stage_times_covers_all_layer_classes():
    # Mixed stack (1 dense + 3 MoE layers): both class tables appear (MoE
    # keys prefixed moe_) plus the lm_head endpoint; the period is the max
    # over ALL of them. All-dense models keep the pass-1 table exactly.
    dev = _moe_hand_device()
    times = dev.all_stage_times(tokens=512)
    assert "S1_qkv" in times and "moe_S1_qkv" in times
    assert "moe_S5_ffn1_moe" in times and "lm_head" in times
    assert times["S1_qkv"] == pytest.approx(dev.analog_gemm_time(512), rel=1e-12)
    assert times["moe_S5_ffn1_moe"] == pytest.approx(dev.moe_ffn_stage_time(512), rel=1e-12)
    period, bottleneck = dev.pipeline_period(tokens=512)
    assert period == pytest.approx(max(times.values()), rel=1e-12)
    assert bottleneck == max(times, key=times.get)
    # Dense-only model: no moe_ keys (pass-1 shape).
    dev_dense = _moe_hand_device(num_experts=1, top_k=1, n_shared_experts=0)
    assert not any(k.startswith("moe_") for k in dev_dense.all_stage_times())


def test_decode_stage_tables_direct_law(cim_llama):
    # Dense decode table at ctx=512, B=4, fast KV (sa-bound): weight stages
    # at M = B; S2 = the decode law with the B streams folded into the
    # contraction (B=4: qk = 8*(128*32*4+94)-1 = 131823, pv =
    # 2*(512*32*4+94)-1 = 131259, total = 131823+96 = 131919 cycles).
    stages = cim_llama.decode_layer_stage_times(512, 1e15, 2, batch_size=4)
    vec = cim_llama.vec_latency_s
    for key in ("S1_qkv", "S3_o_proj", "S4_ffn1", "S5_ffn2"):
        assert stages[key] == pytest.approx(4 * vec, rel=1e-12)
    assert stages["S2_attention"] == pytest.approx((131823 + 96) / 0.95e9, rel=1e-12)
    # decode_all_stage_times adds the lm_head endpoint at M = B.
    all_stages = cim_llama.decode_all_stage_times(512, 1e15, 2, batch_size=4)
    assert all_stages["lm_head"] == pytest.approx(4 * vec, rel=1e-12)
    assert not any(k.startswith("moe_") for k in all_stages)
    period, bottleneck = cim_llama.decode_pipeline_period(512, 1e15, 2, batch_size=4)
    assert bottleneck == "S2_attention"
    assert period == pytest.approx(all_stages["S2_attention"], rel=1e-12)
    # A slow KV tier hands the decode stage to the kv_read term.
    period_kv, bottleneck_kv = cim_llama.decode_pipeline_period(512, 1e9, 2, batch_size=4)
    assert bottleneck_kv == "S2_attention"
    assert period_kv == pytest.approx(
        cim_llama.decode_s2_timing(512, 1e9, 2, batch_size=4).stage_time_s, rel=1e-12
    )


def test_decode_moe_stage_table_tokens_owner_is_batch():
    # MoE decode: tokens_owner = B; FFN stages take the parallel-expert law
    # at B tokens; router/qkv/o_proj at M = B.
    dev = _moe_hand_device()
    stages = dev.decode_moe_layer_stage_times(300, 1e15, 2, batch_size=8)
    vec = dev.vec_latency_s
    assert stages["S1_qkv"] == pytest.approx(8 * vec, rel=1e-12)
    assert stages["S4_router"] == pytest.approx(8 * vec, rel=1e-12)
    assert stages["S5_ffn1_moe"] == pytest.approx(dev.moe_ffn_stage_time(8), rel=1e-12)
    assert stages["S6_ffn2_moe"] == stages["S5_ffn1_moe"]
    # Mixed stack: decode_all_stage_times carries both classes + lm_head.
    all_stages = dev.decode_all_stage_times(300, 1e15, 2, batch_size=8)
    assert "S1_qkv" in all_stages and "moe_S1_qkv" in all_stages
    assert "lm_head" in all_stages


def test_decode_sustained_throughput_kv_bound_llama_hand_case(cim_llama):
    # Hand case recomputed from the llama7b smoke's own reported quantities
    # (fws_cim_llama7b + llama2_7b_fws_inf, final context 2048, B=4):
    #   step latency = 17809.945001027958 us  (decode.contexts[-1].step_latency_us)
    #   period       = 555.1484210526315 us   (decode.contexts[-1].period_us)
    #   kv max_streams = 8                    (8 GiB cim_sram tier / 1 GiB per stream)
    # Derivation:
    #   W_full = ceil(17809.945001027958 / 555.1484210526315)
    #          = ceil(32.0814...) = 33 wavefronts to fill the pipeline;
    #   W_kv   = floor(8 / 4) = 2 wavefronts the KV capacity holds;
    #   sustained = min(33, 2) * 4 / 17809.945001027958 us
    #             = 8 / 17809.945001027958e-6 s = 449.1872377785701 tok/s.
    r = cim_llama.decode_sustained_throughput(
        17809.945001027958e-6, 555.1484210526315e-6, 8, batch_size=4
    )
    assert r.wavefronts_full == 33
    assert r.wavefronts_kv == 2
    assert r.limiting_factor == "kv_capacity"
    assert r.tokens_per_s == pytest.approx(8 / 17809.945001027958e-6, rel=1e-12)
    assert r.tokens_per_s == pytest.approx(449.1872377785701, rel=1e-12)
    # KV binds well below the fabric ceiling B / period = 7205.28 tok/s.
    assert r.tokens_per_s < 4 / 555.1484210526315e-6


def test_decode_sustained_throughput_fabric_bound(cim_llama):
    # W_kv = floor(200/4) = 50 >= W_full = ceil(9/2) = 5: the pipeline
    # fills and the fabric limits. The raw resident rate W_full * B / step
    # = 5 * 4 / 9 us would EXCEED B / period (the ceil rounds the partial
    # fifth wavefront up), and a pipeline with period P cannot sustain more
    # than B / P — so the law caps the fabric-limited figure at exactly
    # B / period.
    r = cim_llama.decode_sustained_throughput(9e-6, 2e-6, 200, batch_size=4)
    assert r.wavefronts_full == 5
    assert r.wavefronts_kv == 50
    assert r.limiting_factor == "fabric"
    assert r.tokens_per_s == pytest.approx(4 / 2e-6, rel=1e-12)  # == B / period
    assert r.tokens_per_s <= 4 / 2e-6  # never above the fabric ceiling


def test_decode_sustained_throughput_infeasible_when_kv_below_batch(cim_llama):
    # kv_max_streams (3) < B (4): W_kv = 0 — the KV capacity holds no full
    # wavefront; sustained is 0 and the limit says infeasible.
    r = cim_llama.decode_sustained_throughput(9e-6, 2e-6, 3, batch_size=4)
    assert r.wavefronts_full == 5
    assert r.wavefronts_kv == 0
    assert r.limiting_factor == "infeasible"
    assert r.tokens_per_s == 0.0


def test_decode_sustained_throughput_defaults_to_model_batch(cim_llama):
    # batch_size omitted: the law uses params.batch_size (4 for the llama
    # smoke model), matching the explicit call.
    assert int(cim_llama.params.batch_size) == 4
    implicit = cim_llama.decode_sustained_throughput(10e-6, 2e-6, 8)
    explicit = cim_llama.decode_sustained_throughput(10e-6, 2e-6, 8, batch_size=4)
    assert implicit == explicit


def test_decode_sustained_throughput_rejects_nonpositive_inputs(cim_llama):
    with pytest.raises(ValueError, match="all > 0"):
        cim_llama.decode_sustained_throughput(0.0, 2e-6, 8, batch_size=4)
    with pytest.raises(ValueError, match="all > 0"):
        cim_llama.decode_sustained_throughput(10e-6, 0.0, 8, batch_size=4)
    with pytest.raises(ValueError, match="all > 0"):
        cim_llama.decode_sustained_throughput(10e-6, 2e-6, 8, batch_size=0)


# ---------------------------------------------------------------------------
# Pass 2 stage 3: end-to-end smokes (DESIGN2 section 4; output/LLM run dir)
# ---------------------------------------------------------------------------



def test_run_perf_llama7b_dense_llm_smoke():
    proc = _run_perf_subprocess(FWS_LLAMA7B, LLAMA2_7B_FWS_INF)
    assert proc.returncode == 0, proc.stdout[-4000:]

    # The report writer must land in the LLM run dir, not output/VIT.
    assert _perf_out("LLM", "fws_cim_report.json").exists()
    report = json.loads(_perf_out("LLM", "fws_cim_report.json").read_text())
    assert report["device_class"] == "fws_cim"
    assert report["layer_classes"] == {"dense": 32, "moe": 0}
    assert report["tokens_owner"] == 4 * 1792  # B * prefill_len

    # Generalized endpoint list: lm_head on the last chip; embedding note.
    lm_head = report["endpoint_stages"]["lm_head"]
    assert lm_head["chip"] == 3
    assert lm_head["arrays"] == 8
    assert lm_head["m_tokens"] == 4 * 1792
    assert "disable_embedding_unembedding" in report["embedding_note"]
    assert [c["arrays_used"] for c in report["chips"]] == [104, 104, 104, 112]

    # KV section coherence (cim_sram; stub tier 8 GB = 8 GiB).
    kv = report["kv"]
    assert kv["story"] == "cim_sram"
    assert kv["final_context"] == 2048
    # per stream: num_layers * 2 * kv_heads * head_dim * ctx * 2 bytes.
    assert kv["bytes_per_stream"] == 32 * 2 * 32 * 128 * 2048 * 2
    assert kv["total_bytes"] == 4 * kv["bytes_per_stream"]
    assert kv["max_streams"] == int(kv["capacity_bytes"] // kv["bytes_per_stream"]) == 8
    assert kv["max_context_at_batch"] == 4096
    assert kv["fits"] is True

    # Decode section: DIRECT LAW EVALUATION at {prefill+1, midpoint, final}.
    decode = report["decode"]
    assert decode["decode_len"] == 256
    assert decode["prefill_len"] == 1792
    labels = [e["label"] for e in decode["contexts"]]
    assert labels == ["first", "midpoint", "final"]
    contexts = [e["context"] for e in decode["contexts"]]
    assert contexts == [1793, 1920, 2048]
    # Decode attention grows with context (kills the silent-fallback class).
    sa_times = [e["s2"]["sa_time_us"] for e in decode["contexts"]]
    assert sa_times[0] < sa_times[1] < sa_times[2]
    kv_reads = [e["s2"]["kv_read_time_us"] for e in decode["contexts"]]
    assert kv_reads[0] < kv_reads[1] < kv_reads[2]
    periods = [e["period_us"] for e in decode["contexts"]]
    assert periods[0] < periods[2]
    final = decode["contexts"][-1]
    assert final["bottleneck_stage"] == "S2_attention"
    assert decode["aggregate_tokens_per_s_final"] == pytest.approx(
        4 / (final["period_us"] * 1e-6), rel=1e-9
    )
    assert final["step_latency_us"] > 32 * final["stages_us"]["S2_attention"]
    assert "sample_every" in decode["integration_note"]  # staircase caveat

    # Fabric ceiling vs sustained reconciliation at the final context:
    # W_full = ceil(step latency / period) = ceil(32.32) = 33 wavefronts to
    # fill the pipeline; W_kv = floor(8 KV streams / B=4) = 2 wavefronts the
    # KV capacity holds; sustained = 2 * 4 / step latency << B / period.
    assert decode["wavefronts_full"] == 33
    assert decode["wavefronts_full"] == math.ceil(
        final["step_latency_us"] / final["period_us"]
    )
    assert decode["wavefronts_kv"] == kv["max_streams"] // 4 == 2
    assert decode["decode_throughput_limit"] == "kv_capacity"
    assert decode["sustained_tokens_per_s"] == pytest.approx(
        2 * 4 / (final["step_latency_us"] * 1e-6), rel=1e-9
    )
    assert decode["sustained_tokens_per_s"] < decode["aggregate_tokens_per_s_final"]
    assert "fabric ceiling" in decode["aggregate_note"].lower()

    # Readable section rides on the LLM results txt.
    results_text = _perf_out("LLM", "LLM_inference_results.txt").read_text()
    assert "FWS-CIM spatial pipeline" in results_text
    assert "KV cache (story: cim_sram)" in results_text
    assert "Decode (direct law evaluation" in results_text
    # Both throughput lines: the relabeled fabric ceiling with its stream
    # requirement (33 wavefronts x B=4 = 132 streams) and the sustained line.
    assert "fabric ceiling, needs >= 132 streams in flight" in results_text
    assert "sustained at KV capacity (max 8 streams => 2 wavefronts of B=4)" in results_text
    assert "[limited by kv_capacity]" in results_text
    assert "twice per layer" in results_text
    # cim_sram: the activation-feasibility label says KV is included.
    mem_text = _perf_out("LLM", "memory-summary", "memory_capacity_comparison.txt").read_text()
    assert "includes the KV cache under kvcache_type: cim_sram" in mem_text


def test_run_perf_llama7b_kvdram_variant_side_check():
    proc = _run_perf_subprocess(FWS_LLAMA7B_KVDRAM, LLAMA2_7B_FWS_INF)
    assert proc.returncode == 0, proc.stdout[-4000:]

    report = json.loads(_perf_out("LLM", "fws_cim_report.json").read_text())
    kv = report["kv"]
    assert kv["story"] == "cim_dram"
    assert kv["capacity_bytes"] == 8589934592.0
    assert kv["bandwidth_bytes_per_s"] == 100000000000.0
    assert kv["fits"] is True  # 4 GiB needed vs 8 GiB tier
    # The 100 GB/s tier binds decode S2 (kv_read >> sa here).
    assert report["decode"]["contexts"][-1]["s2"]["bound"] == "kv_read"
    # KV capacity still caps the streams: floor(8 / B=4) = 2 wavefronts...
    assert report["decode"]["wavefronts_kv"] == 2
    # ...but the ONE 100 GB/s tier serves every resident wavefront's kv
    # reads concurrently, so the tighter cap is the tier bandwidth over the
    # per-token KV read: 1e11 B/s / 1 GiB per generated token = 93.13 tok/s
    # (the 2-wavefront resident rate alone would imply ~2x the tier).
    assert report["decode"]["decode_throughput_limit"] == "kv_bandwidth"
    assert report["decode"]["sustained_tokens_per_s"] == pytest.approx(
        1e11 / report["kv"]["bytes_per_stream"], rel=1e-9
    )
    assert "[limited by kv_bandwidth]" in _perf_out("LLM", "LLM_inference_results.txt").read_text()
    # cim_dram KV traffic energy joins the PARTIAL figure.
    assert report["energy_partial_pj"]["kv_dram_traffic"] > 0

    # Side capacity check surfaces in the capacity report text (WARN path
    # never raises; this config has headroom).
    mem_text = _perf_out("LLM", "memory-summary", "memory_capacity_comparison.txt").read_text()
    assert "KV DRAM tier (kvcache_type: cim_dram)" in mem_text
    assert "KV DRAM headroom" in mem_text
    assert "excluded here" in mem_text  # SRAM check stays activations-only


def test_run_perf_llama7b_kvdram_overflow_warns_never_raises(tmp_path):
    # Shrink the KV tier below the 4 GiB the run needs: the side check WARNS
    # (matching the memory-pass precedent) and the run still exits 0.
    hw_dict = _load_yaml(FWS_LLAMA7B_KVDRAM)
    hw_dict["cim"]["kv_dram"]["capacity_bytes"] = 1073741824  # 1 GiB
    hw_path = tmp_path / "fws_cim_llama7b_kvdram_small.yaml"
    hw_path.write_text(yaml.safe_dump(hw_dict))

    proc = _run_perf_subprocess(hw_path, LLAMA2_7B_FWS_INF)
    assert proc.returncode == 0, proc.stdout[-4000:]
    assert "[WARNING]: fws_cim: KV cache at the final context does not fit" in proc.stdout
    report = json.loads(_perf_out("LLM", "fws_cim_report.json").read_text())
    assert report["kv"]["fits"] is False
    assert report["kv"]["max_streams"] == 1  # floor(1 GiB / 1 GiB per stream)
    # Sustained reconciliation on the overflow path: 1 stream < B=4 means
    # zero KV wavefronts — 0 tok/s sustained, flagged infeasible (still WARN,
    # never fatal).
    assert report["decode"]["wavefronts_kv"] == 0
    assert report["decode"]["decode_throughput_limit"] == "infeasible"
    assert report["decode"]["sustained_tokens_per_s"] == 0.0
    mem_text = _perf_out("LLM", "memory-summary", "memory_capacity_comparison.txt").read_text()
    assert "[WARN] KV DRAM capacity exceeded by 3.00 GiB" in mem_text
    results_text = _perf_out("LLM", "LLM_inference_results.txt").read_text()
    assert "[WARNING] KV capacity exceeded" in results_text
    assert "[infeasible: KV holds no full wavefront of B]" in results_text


def test_run_perf_moe_smoke():
    proc = _run_perf_subprocess(FWS_MOE, MOE_SMALL_FWS_INF)
    assert proc.returncode == 0, proc.stdout[-4000:]

    report = json.loads(_perf_out("LLM", "fws_cim_report.json").read_text())
    assert report["layer_classes"] == {"dense": 1, "moe": 11}

    # Expert arrays in the capacity census.
    moe = report["moe"]
    assert moe["stage_arrays"] == {
        "qkv": 2,
        "o_proj": 1,
        "router": 1,
        "ffn1_routed": 16,
        "ffn2_routed": 16,
        "ffn1_shared": 1,
        "ffn2_shared": 1,
    }
    assert moe["arrays_per_moe_layer"] == 38
    assert report["arrays_per_layer"] == 9  # dense layer, gated FUSED law
    assert [c["arrays_used"] for c in report["chips"]] == [199, 260]

    # Dispatch/combine boundary entries on the ep link.
    assert moe["dispatch"]["count"] == 22  # 11 MoE layers x (dispatch+combine)
    assert moe["dispatch"]["link"] == "ep"
    # tokens_owner = B * prefill_len = 4 * (512 - 32) = 1920 owner tokens.
    assert moe["dispatch"]["bytes_each_way"] == 4 * 480 * 2 * 1024 * 2
    assert moe["dispatch"]["time_us_each_way"] > 0
    # The ep dim (50 GB) deliberately differs from pp (100 GB), so the
    # each-way time only matches the ep-link p2p law: bytes/bw + latency.
    # A regression that priced dispatch/combine on the pp link fails here.
    assert moe["dispatch"]["time_us_each_way"] == pytest.approx(
        (moe["dispatch"]["bytes_each_way"] / (50 * 2**30) + 1e-6) * 1e6, rel=1e-9
    )
    assert report["moe_stages"]["S5_ffn1_moe"]["arrays_per_layer"] == 17

    # Dispatch/combine energy joins the PARTIAL interconnect term on the ep
    # link: interconnect = pp boundary (1 x 3,932,160 B) + ep dispatch
    # (11 layers x 2 ways x 7,864,320 B), both at 1e-12 J/bit.
    energy = report["energy_partial_pj"]
    expected_dispatch_pj = 22 * moe["dispatch"]["bytes_each_way"] * 8.0 * 1e-12 * 1e12
    assert energy["interconnect_moe_dispatch"] == pytest.approx(
        expected_dispatch_pj, rel=1e-9
    )
    assert energy["interconnect"] == pytest.approx(
        (3932160 * 8.0 * 1e-12 * 1e12) + expected_dispatch_pj, rel=1e-9
    )

    # Serialized-experts disclosure joins the sequential note.
    assert "serialized" in report["sequential_latency_note"]
    assert "twice per layer" in report["sequential_latency_note"]

    results_text = _perf_out("LLM", "LLM_inference_results.txt").read_text()
    assert "MoE layer stages" in results_text
    assert "MoE dispatch/combine (p2p over the ep link)" in results_text
    assert "serialized" in results_text


def test_run_perf_moe_expert_parallel_pool_accounting(tmp_path):
    hw_dict = _load_yaml(FWS_MOE)
    hw_dict["cim"]["chip"]["moe_expert_parallel"] = 4
    hw_path = tmp_path / "fws_cim_moe_ep4.yaml"
    hw_path.write_text(yaml.safe_dump(hw_dict))

    proc = _run_perf_subprocess(hw_path, MOE_SMALL_FWS_INF)
    assert proc.returncode == 0, proc.stdout[-4000:]
    report = json.loads(_perf_out("LLM", "fws_cim_report.json").read_text())
    pool = report["moe"]["expert_pool"]
    # 11 MoE layers x 4 chips; ceil(32 routed arrays / 4) = 8 per chip; the
    # layer chips keep 38 - 32 = 6 arrays per MoE layer.
    assert pool["chips"] == 44
    assert pool["arrays_used_per_chip"] == 8
    assert [c["arrays_used"] for c in report["chips"]] == [9 + 5 * 6, 6 * 6 + 32]
    assert "MoE expert pool: 44 dedicated chips" in _perf_out("LLM", "LLM_inference_results.txt").read_text()


def test_run_perf_llama7b_auto_placement(tmp_path):
    hw_dict = _load_yaml(FWS_LLAMA7B)
    hw_dict["cim"]["chip"]["layers_per_chip"] = "auto"
    hw_path = tmp_path / "fws_cim_llama7b_auto.yaml"
    hw_path.write_text(yaml.safe_dump(hw_dict))

    proc = _run_perf_subprocess(hw_path, LLAMA2_7B_FWS_INF)
    assert proc.returncode == 0, proc.stdout[-4000:]
    report = json.loads(_perf_out("LLM", "fws_cim_report.json").read_text())
    assert report["placement"] == "auto"
    assert report["derived_layers_per_chip"] == [9, 9, 9, 5]
    assert [c["arrays_used"] for c in report["chips"]] == [117, 117, 117, 73]
    assert "auto -> derived [9, 9, 9, 5]" in _perf_out("LLM", "LLM_inference_results.txt").read_text()


def test_run_perf_gpu_native_llm_decode_runs(tmp_path):
    # gpu_native: the native tile machinery prices the skinny decode ops
    # against the stub tech_param; the spatial report still runs (its laws
    # come from CimDeviceModel directly).
    hw_dict = _load_yaml(FWS_LLAMA7B)
    hw_dict["cim"]["fabric"]["model"] = "gpu_native"
    hw_path = tmp_path / "fws_cim_llama7b_gpunative.yaml"
    hw_path.write_text(yaml.safe_dump(hw_dict))

    proc = _run_perf_subprocess(hw_path, LLAMA2_7B_FWS_INF)
    assert proc.returncode == 0, proc.stdout[-4000:]
    report = json.loads(_perf_out("LLM", "fws_cim_report.json").read_text())
    assert report["fabric_model"] == "gpu_native"
    assert report["decode"] is not None
    # No folded-run double count under gpu_native, and no MoE here: the
    # sequential note stays empty (pass-1 behavior).
    assert report["sequential_latency_note"] is None


# ---------------------------------------------------------------------------
# Pass 2B: DSE tool (tools/fws_cim_dse.py) — closed-form sweep on
# CimDeviceModel, chips derived via the auto placement, T1 pinned.
# ---------------------------------------------------------------------------

# The recorded OPTIMA T1 array point as a cim.dse.variants entry (rows /
# slice_cycles / analog_clock_mhz inherit cim.analog, which IS the T1 point
# in fws_cim_optima_t1.yaml).
T1_VARIANT = {
    "adc_mux": 4,
    "cols_adc": 320,
    "energy_per_vec_pj": 9534.9389,
    "area_mm2_per_array": 1.403065,
}


def _load_dse_module():
    """Import tools/fws_cim_dse.py in-process (tools/ is not a package)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "fws_cim_dse_under_test", str(PROJECT_ROOT / "tools" / "fws_cim_dse.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_t1_dse_yaml(tmp_path, variants, tp_candidates=(1,), **dse_extra):
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["cim"]["dse"] = dict(
        {"variants": list(variants), "tp_candidates": list(tp_candidates)},
        **dse_extra,
    )
    hw_path = tmp_path / "fws_cim_t1_dse.yaml"
    hw_path.write_text(yaml.safe_dump(hw_dict, sort_keys=False))
    return hw_path


def test_layers_per_chip_auto_uniform_matches_manual_split():
    # DESIGN2 section 2: when the capacity packs exactly the manual uniform
    # split, auto must reproduce it. Llama 13 arrays/layer at capacity 112:
    # greedy packs 8 layers/chip (104; a 9th would hit 117) and the last chip
    # carries 8 layers + 8 lm_head arrays = 112 — identical to the shipped
    # manual spec layers_per_chip: 8.
    dev_auto = _llama_auto_device(arrays_per_chip=112)
    hw_manual = _load_yaml(FWS_LLAMA7B)
    hw_manual["cim"]["chip"]["arrays_per_chip"] = 112
    model = config.parse_config(str(LLAMA2_7B_FWS_INF), "LLM").model_config
    dev_manual = cim_timing.CimDeviceModel(_hw_from_dict(hw_manual), model)
    assert dev_manual.chip_layer_counts() == (8, 8, 8, 8)
    assert dev_auto.chip_layer_counts() == dev_manual.chip_layer_counts()
    assert dev_auto.chip_array_usage() == dev_manual.chip_array_usage() == (104, 104, 104, 112)
    dev_auto.validate_capacity()


def test_layers_per_chip_auto_overflow_error_names_layer_class():
    # The over-capacity error must name the layer class and both counts.
    dev = _llama_auto_device(arrays_per_chip=12)
    with pytest.raises(ValueError, match=r"layer 0 \(dense\).*needs 13.*arrays_per_chip = 12"):
        dev.chip_layer_counts()

    def shrink(d):
        d["cim"]["chip"]["layers_per_chip"] = "auto"
        d["cim"]["chip"]["arrays_per_chip"] = 20  # dense layer 0 (9) fits; MoE (22) does not

    dev_moe = _moe_hand_device(hw_mutate=shrink)
    with pytest.raises(ValueError, match=r"layer 1 \(MoE\).*needs 22"):
        dev_moe.chip_layer_counts()


def test_dse_moe_expert_parallel_schema():
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["cim"]["dse"] = {
        "variants": [dict(T1_VARIANT)],
        "moe_expert_parallel": [1, 2, 4],
    }
    hw = _hw_from_dict(hw_dict)
    assert hw.cim_config.dse.moe_expert_parallel == (1, 2, 4)
    # Default when absent: no spreading.
    hw_dict["cim"]["dse"].pop("moe_expert_parallel")
    assert _hw_from_dict(hw_dict).cim_config.dse.moe_expert_parallel == (1,)
    # Entries must be integers >= 1.
    hw_dict["cim"]["dse"]["moe_expert_parallel"] = [0]
    with pytest.raises(ValueError, match="moe_expert_parallel"):
        _hw_from_dict(hw_dict)


def test_dse_t1_pinned_selection(tmp_path):
    # DESIGN2 section 5 validation: a sweep whose variants contain exactly
    # the T1 array point at tp=[1] must select a candidate whose period /
    # fps / arrays equal the recorded T1 targets.
    dse = _load_dse_module()
    hw_path = _write_t1_dse_yaml(tmp_path, [T1_VARIANT])
    out_dir = tmp_path / "dse_out"
    emitted = tmp_path / "t1_selected.yaml"
    rc = dse.main(
        [
            "--hardware_config", str(hw_path),
            "--model_config", str(VIT_HUGE_64),
            "--output-dir", str(out_dir),
            "--emit-config", str(emitted),
        ]
    )
    assert rc == 0
    payload = json.loads((out_dir / "dse_report.json").read_text())
    assert payload["workload"] == "prefill"
    assert payload["throughput_unit"] == "fps"
    selected = payload["selected"]
    assert selected is not None and selected["id"] == payload["selected_id"]
    assert selected["knobs"]["adc_mux"] == 4
    assert selected["knobs"]["tp"] == 1
    metrics = selected["metrics"]
    # Recorded T1 targets (test_run_perf_t1_end_to_end_smoke).
    assert metrics["period_us"] == pytest.approx(5.12, rel=1e-6)
    assert metrics["fps"] == pytest.approx(195312.5, rel=1e-6)
    assert metrics["throughput"] == pytest.approx(195312.5, rel=1e-6)
    assert metrics["arrays_total"] == 386
    assert metrics["chips"] == 1
    assert metrics["derived_layers_per_chip"] == [32]
    assert metrics["area_mm2_total"] == pytest.approx(386 * 1.403065, rel=1e-9)
    # The single feasible candidate is the whole Pareto front.
    assert payload["front_ids"] == [selected["id"]]
    # Artifacts: markdown + emitted runnable YAML with the DERIVED split.
    assert (out_dir / "dse_report.md").exists()
    emitted_dict = _load_yaml(emitted)
    assert emitted_dict["cim"]["chip"]["layers_per_chip"] == [32]
    assert emitted_dict["cim"]["analog"]["adc_mux"] == 4
    assert emitted_dict["parallelism"]["tp"] == 1
    # The emitted config parses as a valid fws_cim hardware config.
    emitted_hw = _hw_from_dict(emitted_dict)
    config.validate_hw_config(emitted_hw)


def test_dse_dominance_keeps_t1_on_front(tmp_path):
    # A variants list where the T1 point dominates (higher throughput, lower
    # area) must keep it on the front and drop the dominated point.
    dominated = {
        "adc_mux": 8,          # vec_cycles 16 -> half the throughput
        "cols_adc": 320,
        "energy_per_vec_pj": 9534.9389,
        "area_mm2_per_array": 5.0,  # larger area despite fewer arrays
    }
    dse = _load_dse_module()
    # Dominated point FIRST: both candidates are feasible, so a selection
    # that returned the first feasible candidate instead of applying the
    # objective would pick it and fail the selected_id assert below.
    hw_path = _write_t1_dse_yaml(tmp_path, [dominated, T1_VARIANT])
    out_dir = tmp_path / "dse_out"
    rc = dse.main(
        [
            "--hardware_config", str(hw_path),
            "--model_config", str(VIT_HUGE_64),
            "--output-dir", str(out_dir),
        ]
    )
    assert rc == 0
    payload = json.loads((out_dir / "dse_report.json").read_text())
    by_mux = {c["knobs"]["adc_mux"]: c for c in payload["candidates"]}
    assert by_mux[4]["ok"] and by_mux[8]["ok"]  # both feasible, one dominated
    assert by_mux[4]["metrics"]["throughput"] > by_mux[8]["metrics"]["throughput"]
    assert by_mux[4]["metrics"]["area_mm2_total"] < by_mux[8]["metrics"]["area_mm2_total"]
    assert payload["front_ids"] == [by_mux[4]["id"]]
    assert payload["selected_id"] == by_mux[4]["id"]


def test_dse_infeasible_only_exits_nonzero(tmp_path, capsys):
    # Every candidate infeasible -> nonzero exit with the best violation
    # surfaced (stage tag + actionable message; nothing silently dropped).
    tiny = {
        "adc_mux": 1,
        "cols_adc": 8,
        "rows": 64,   # one layer alone needs tens of thousands of arrays
        "energy_per_vec_pj": 1.0,
        "area_mm2_per_array": 0.001,
    }
    dse = _load_dse_module()
    hw_path = _write_t1_dse_yaml(tmp_path, [tiny])
    out_dir = tmp_path / "dse_out"
    rc = dse.main(
        [
            "--hardware_config", str(hw_path),
            "--model_config", str(VIT_HUGE_64),
            "--output-dir", str(out_dir),
        ]
    )
    assert rc != 0
    out = capsys.readouterr().out
    assert "no feasible candidate" in out
    assert "cannot place layer" in out  # the best violation is surfaced
    payload = json.loads((out_dir / "dse_report.json").read_text())
    assert payload["selected_id"] is None
    assert payload["num_valid"] == 0
    assert payload["best_violation"]["fail_stage"] == "placement"
    assert all(not c["ok"] for c in payload["candidates"])
    # Every infeasible candidate carries its stage tag + message.
    assert all(c["fail_stage"] and c["fail_message"] for c in payload["candidates"])


def test_dse_tp_auto_selection_beats_first_valid(tmp_path):
    # End-to-end tp_candidates: auto on the MoE model (num_heads 16,
    # num_experts 16, moe_dp 1): auto must resolve to the divisors
    # [1, 2, 4, 8, 16]. Selection coverage (decode workload, so the ranking
    # key is sustained_tokens_per_s): tp 4/8/16 saturate (ceil(kv_heads/tp)
    # == 1) and tie on the sustained figure, but tp now COSTS silicon —
    # chips and area multiply by the tp shard count — so the min-chips
    # tie-break must pick tp=4, the THIRD valid candidate. A selection that
    # ignored the objective and returned the first feasible candidate
    # (tp=1, lower throughput) fails here.
    dse = _load_dse_module()
    hw_dict = _load_yaml(FWS_MOE)
    hw_dict["cim"]["dse"] = {
        "variants": [
            {
                "adc_mux": 4,
                "cols_adc": 256,
                "energy_per_vec_pj": 6102.361,
                "area_mm2_per_array": 0.897962,
            }
        ],
        "tp_candidates": "auto",
    }
    hw_path = tmp_path / "fws_cim_moe_tp_auto.yaml"
    hw_path.write_text(yaml.safe_dump(hw_dict, sort_keys=False))
    out_dir = tmp_path / "dse_out"
    rc = dse.main(
        [
            "--hardware_config", str(hw_path),
            "--model_config", str(MOE_SMALL_FWS_INF),
            "--output-dir", str(out_dir),
        ]
    )
    assert rc == 0
    payload = json.loads((out_dir / "dse_report.json").read_text())
    assert payload["tp_candidates"] == [1, 2, 4, 8, 16]
    assert payload["dse_echo"]["tp_candidates"] == "auto"
    assert payload["dse_echo"]["resolved_tp_candidates"] == [1, 2, 4, 8, 16]
    valid = [c for c in payload["candidates"] if c["ok"]]
    assert len(valid) == 5
    by_tp = {c["knobs"]["tp"]: c for c in valid}
    # tp 4, 8, 16 saturate (ceil(kv_heads/tp) == 1) and tie exactly; tp 1
    # and 2 are strictly slower.
    thr = {tp: by_tp[tp]["metrics"]["throughput"] for tp in by_tp}
    assert thr[4] == pytest.approx(thr[8], rel=1e-12)
    assert thr[4] == pytest.approx(thr[16], rel=1e-12)
    assert thr[1] < thr[2] < thr[4]
    # The decode ranking key (sustained_tokens_per_s) shows the same
    # saturation pattern, so the min-tp tie-break decides here too.
    sust = {tp: by_tp[tp]["metrics"]["sustained_tokens_per_s"] for tp in by_tp}
    assert sust[4] == pytest.approx(sust[8], rel=1e-12)
    assert sust[4] == pytest.approx(sust[16], rel=1e-12)
    assert sust[1] < sust[2] < sust[4]
    # tp is not a free knob: the system is tp shard devices, so chips and
    # area scale linearly with tp (per-shard census x tp; the per-shard
    # placement is tp-invariant because the census does not shard weights).
    for tp in (2, 4, 8, 16):
        assert by_tp[tp]["metrics"]["tp_shards"] == tp
        assert by_tp[tp]["metrics"]["chips"] == tp * by_tp[1]["metrics"]["chips"]
        assert by_tp[tp]["metrics"]["area_mm2_total"] == pytest.approx(
            tp * by_tp[1]["metrics"]["area_mm2_total"], rel=1e-9
        )
    # Decode selection ranks on the sustained figure; the rule text says so.
    assert "sustained" in payload["selection_rule"]
    # The winner is tp=4: sustained ties for 4/8/16, and the min-chips
    # tie-break (8 < 16 < 32 chips) decides — NOT the first valid
    # candidate (tp=1).
    assert payload["selected_id"] == by_tp[4]["id"]
    assert payload["selected_id"] != valid[0]["id"]


def test_dse_decode_selection_uses_sustained_metric(tmp_path):
    # The honest-headline rule (decode workloads select on
    # sustained_tokens_per_s): two llama7b variants that tie EXACTLY on the
    # fabric-ceiling tok/s (the decode period is S2/fabric-bound, so adc_mux
    # never touches it) but differ on the sustained figure. The mux-8 point
    # has FEWER chips and LESS area (higher mux packs more columns per
    # array), so the OLD ceiling-based rule would pick it on the min-chips
    # tie-break; the sustained key must pick the mux-4 point (smaller step
    # latency => more tokens per resident wavefront). The winner is
    # deliberately listed SECOND.
    mux4 = {
        "adc_mux": 4,
        "cols_adc": 1024,
        "energy_per_vec_pj": 97637.774,
        "area_mm2_per_array": 14.367386,
    }
    mux8 = dict(mux4, adc_mux=8)
    dse = _load_dse_module()
    hw_dict = _load_yaml(FWS_LLAMA7B)
    hw_dict["cim"]["dse"] = {"variants": [mux8, mux4], "tp_candidates": [1]}
    hw_path = tmp_path / "fws_cim_llama_sustained.yaml"
    hw_path.write_text(yaml.safe_dump(hw_dict, sort_keys=False))
    out_dir = tmp_path / "dse_out"
    rc = dse.main(
        [
            "--hardware_config", str(hw_path),
            "--model_config", str(LLAMA2_7B_FWS_INF),
            "--output-dir", str(out_dir),
        ]
    )
    assert rc == 0
    payload = json.loads((out_dir / "dse_report.json").read_text())
    assert payload["workload"] == "decode"
    by_mux = {c["knobs"]["adc_mux"]: c for c in payload["candidates"]}
    m4, m8 = by_mux[4]["metrics"], by_mux[8]["metrics"]
    assert by_mux[4]["ok"] and by_mux[8]["ok"]
    # Exact ceiling tie; mux8 wins every old tie-break (chips, area).
    assert m4["throughput"] == m8["throughput"]
    assert m8["chips"] < m4["chips"]
    assert m8["area_mm2_total"] < m4["area_mm2_total"]
    # Both are KV-capacity-bound at 2 wavefronts; mux4 sustains more.
    assert m4["decode_throughput_limit"] == m8["decode_throughput_limit"] == "kv_capacity"
    assert m4["wavefronts_kv"] == m8["wavefronts_kv"] == 2
    assert m4["sustained_tokens_per_s"] > m8["sustained_tokens_per_s"]
    # Same figure as the smoke's hand case: 2 wavefronts * B=4 / step latency.
    assert m4["sustained_tokens_per_s"] == pytest.approx(449.1872377785701, rel=1e-9)
    # The sustained key decides: mux4 wins despite mux8's chips/area edge.
    assert payload["selected_id"] == by_mux[4]["id"]
    # The front's throughput axis is the same key, so both points are
    # non-dominated (mux4 sustains more, mux8 is smaller) and the winner
    # sits on the front it was chosen from.
    assert sorted(payload["front_ids"]) == sorted(c["id"] for c in payload["candidates"])
    assert payload["selected_id"] in payload["front_ids"]


def test_dse_objective_min_chips_end_to_end(tmp_path):
    # --objective min_chips end to end, on a sweep where the two objectives
    # genuinely disagree: at arrays_per_chip 350 the T1 point needs 2 chips
    # (higher fps), the mux-8 point packs all 32 layers on 1 chip (half the
    # fps). throughput must pick the T1 point; min_chips must pick the
    # 1-chip point — which is also the SECOND candidate, so a
    # first-feasible-wins selection bug fails under min_chips too.
    mux8 = {
        "adc_mux": 8,
        "cols_adc": 320,
        "energy_per_vec_pj": 9534.9389,
        "area_mm2_per_array": 1.403065,
    }
    dse = _load_dse_module()
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["cim"]["chip"]["arrays_per_chip"] = 350
    hw_dict["cim"]["dse"] = {
        "variants": [dict(T1_VARIANT), mux8],
        "tp_candidates": [1],
    }
    hw_path = tmp_path / "fws_cim_t1_objectives.yaml"
    hw_path.write_text(yaml.safe_dump(hw_dict, sort_keys=False))

    payloads = {}
    for objective in ("throughput", "min_chips"):
        out_dir = tmp_path / f"dse_out_{objective}"
        rc = dse.main(
            [
                "--hardware_config", str(hw_path),
                "--model_config", str(VIT_HUGE_64),
                "--output-dir", str(out_dir),
                "--objective", objective,
            ]
        )
        assert rc == 0
        payloads[objective] = json.loads((out_dir / "dse_report.json").read_text())

    for payload in payloads.values():
        by_mux = {c["knobs"]["adc_mux"]: c for c in payload["candidates"]}
        assert by_mux[4]["ok"] and by_mux[8]["ok"]
        assert by_mux[4]["metrics"]["chips"] == 2
        assert by_mux[8]["metrics"]["chips"] == 1
        assert by_mux[4]["metrics"]["fps"] == pytest.approx(195312.5, rel=1e-6)
        assert by_mux[8]["metrics"]["fps"] == pytest.approx(97656.25, rel=1e-6)
        # A real trade-off: both points sit on the throughput-vs-area front.
        assert sorted(payload["front_ids"]) == sorted(c["id"] for c in payload["candidates"])

    thr = payloads["throughput"]
    chips = payloads["min_chips"]
    assert thr["objective"] == "throughput"
    assert chips["objective"] == "min_chips"
    assert thr["selected"]["knobs"]["adc_mux"] == 4
    assert chips["selected"]["knobs"]["adc_mux"] == 8
    assert chips["selected"]["metrics"]["chips"] == 1
    # min_chips picks the SECOND candidate, not the first valid one.
    assert chips["selected_id"] != chips["candidates"][0]["id"]


def test_dse_mux_candidates_cross_check(tmp_path, capsys):
    # cim.dse.mux_candidates never expands the sweep: it is a cross-check.
    # A variant whose adc_mux is missing from the list is a config error
    # (exit 2, clean message); a consistent list passes and is echoed.
    dse = _load_dse_module()
    args_tail = [
        "--model_config", str(VIT_HUGE_64),
        "--output-dir", str(tmp_path / "dse_out"),
    ]

    # Consistent: sweep runs and the echo records the list.
    hw_ok = _write_t1_dse_yaml(tmp_path, [T1_VARIANT], mux_candidates=[1, 2, 4, 8, 16])
    rc = dse.main(["--hardware_config", str(hw_ok)] + args_tail)
    assert rc == 0
    payload = json.loads((tmp_path / "dse_out" / "dse_report.json").read_text())
    assert payload["dse_echo"]["mux_candidates"] == [1, 2, 4, 8, 16]

    # Inconsistent: the T1 variant's adc_mux 4 is not in [8, 16].
    hw_bad = _write_t1_dse_yaml(tmp_path, [T1_VARIANT], mux_candidates=[8, 16])
    capsys.readouterr()
    rc = dse.main(["--hardware_config", str(hw_bad)] + args_tail)
    assert rc == 2
    out = capsys.readouterr().out
    assert "[FWS-CIM DSE] error:" in out
    assert "variants[0].adc_mux = 4" in out
    assert "mux_candidates" in out


def test_dse_config_parse_error_exits_2_cleanly(tmp_path, capsys):
    # A hardware-YAML config error (here: a variant missing the required
    # adc_mux) must follow the documented exit contract — code 2 with the
    # clean "[FWS-CIM DSE] error:" line — not a raw traceback.
    dse = _load_dse_module()
    broken = {"cols_adc": 320, "energy_per_vec_pj": 1.0, "area_mm2_per_array": 1.0}
    hw_path = _write_t1_dse_yaml(tmp_path, [broken])
    capsys.readouterr()
    rc = dse.main(
        [
            "--hardware_config", str(hw_path),
            "--model_config", str(VIT_HUGE_64),
            "--output-dir", str(tmp_path / "dse_out"),
        ]
    )
    assert rc == 2
    out = capsys.readouterr().out
    assert "[FWS-CIM DSE] error:" in out
    assert "adc_mux" in out


# ---------------------------------------------------------------------------
# QIF P2: the macro resource model — cards, tiles, active-column-set pricing,
# bit slicing, the per-macro digital pool (D10-D14, ADJ-4, D23: no accuracy)
# ---------------------------------------------------------------------------

#: Every shipped fws_cim template with the model it is paired with.
SHIPPED_FWS_PAIRS = (
    (FWS_T1, VIT_HUGE_64, "VIT"),
    (FWS_T2, VIT_G_64, "VIT"),
    (FWS_T3, VIT_HUGE_196, "VIT"),
    (FWS_LLAMA7B, LLAMA2_7B_FWS_INF, "LLM"),
    (FWS_LLAMA7B_KVDRAM, LLAMA2_7B_FWS_INF, "LLM"),
    (FWS_MOE, MOE_SMALL_FWS_INF, "LLM"),
)


def _device_from_dicts(hw_dict, model_path, mode="VIT"):
    """Device model from a mutated hardware dict plus a shipped model YAML."""
    hw = _hw_from_dict(hw_dict)
    model = config.parse_config(str(model_path), mode).model_config
    return cim_timing.CimDeviceModel(hw, model)


def _sliced_t1_device(**card_overrides):
    """T1 with an explicit CTT card: 2 bits/cell of an 8-bit weight => n_s = 4."""
    hw_dict = _load_yaml(FWS_T1)
    card = {
        "kind": "analog_macro",
        "device": "ctt",
        "bits_per_cell": 2,
        "weight_bits": 8,
        "bank_depth": 1,
    }
    card.update(card_overrides)
    hw_dict["cim"]["cards"] = {"ctt_sliced": card}
    return _device_from_dicts(hw_dict, VIT_HUGE_64)


# --- P2.1 device cards -----------------------------------------------------


@pytest.mark.parametrize("hw_path,model_path,mode", SHIPPED_FWS_PAIRS)
def test_shipped_configs_synthesize_an_inert_card(hw_path, model_path, mode):
    # Backward compatibility is the whole contract: a YAML with no cim.cards
    # block gets a synthesized card whose every knob is inert, wrapping the
    # very cim.analog / cim.fabric objects the laws already read.
    hw = config.parse_config(str(hw_path), "hardware")
    cim = hw.cim_config
    card = cim.analog_card
    assert card.name == "default" and card.device == "ctt"
    assert card.params is cim.analog
    assert cim.digital_card.fabric is cim.fabric
    assert card.n_slices == 1 and not card.slicing_enabled
    assert card.allocation_granularity == cim.analog.adc_mux  # whole macro
    assert card.stack_3d_height == 1
    assert card.validity == ()
    assert (card.pool_clock_ghz, card.pool_energy_per_add_pj, card.pool_area_mm2_per_adder) == (
        0.0,
        0.0,
        0.0,
    )
    assert cim.allocation is None  # dedicated per matrix

    model = config.parse_config(str(model_path), mode).model_config
    dev = cim_timing.CimDeviceModel(hw, model)
    assert dev.analog is cim.analog and dev.fabric is cim.fabric
    assert dev.n_slices == 1
    assert dev.column_sets_per_tile == cim.analog.adc_mux


def test_card_block_parses_structural_knobs():
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["cim"]["cards"] = {
        "ctt_2bpc": {
            "kind": "analog_macro",
            "device": "ctt",
            "bits_per_cell": 2,
            "weight_bits": 8,
            "slicing": "chained_macros",
            "bank_depth": 2,
            "stack_3d_height": 4,
            "validity": [{"bits_per_cell": 2, "weight_bits": 8, "mux": 4}],
            "pool_clock_ghz": 1.5,
            "pool_energy_per_add_pj": 0.25,
            "pool_area_mm2_per_adder": 0.001,
        },
        "shared_digital": {"kind": "digital_chiplet", "area_mm2": 12.0},
    }
    hw = _hw_from_dict(hw_dict)
    card = hw.cim_config.analog_card
    assert card.name == "ctt_2bpc"
    assert (card.bits_per_cell, card.weight_bits) == (2, 8)
    assert card.n_slices == 4 and card.slicing_enabled
    assert card.slicing == "chained_macros"
    assert card.allocation_granularity == 2
    assert card.stack_3d_height == 4
    assert card.pool_clock_ghz == pytest.approx(1.5)
    digital = hw.cim_config.digital_card
    assert digital.name == "shared_digital"
    assert digital.area_mm2 == pytest.approx(12.0)
    # The chiplet card wraps the fabric block verbatim: the SA and softmax
    # laws are ITS laws and read the same parameters (D13).
    assert digital.fabric is hw.cim_config.fabric


def test_card_n_slices_law_and_opt_in():
    # n_s = ceil(weight_bits / bits_per_cell); slicing is present iff
    # bits_per_cell < weight_bits (ADJ-4).
    for bpc, wbits, expected in ((2, 8, 4), (3, 8, 3), (4, 8, 2), (8, 8, 1), (16, 8, 1)):
        hw_dict = _load_yaml(FWS_T1)
        hw_dict["cim"]["cards"] = {
            "c": {"kind": "analog_macro", "bits_per_cell": bpc, "weight_bits": wbits}
        }
        card = _hw_from_dict(hw_dict).cim_config.analog_card
        assert card.n_slices == expected
        assert card.slicing_enabled == (expected > 1)


def test_card_bits_per_cell_and_weight_bits_declared_together():
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["cim"]["cards"] = {"c": {"kind": "analog_macro", "bits_per_cell": 2}}
    with pytest.raises(ValueError, match="declared together"):
        _hw_from_dict(hw_dict)


def test_card_bank_depth_must_divide_mux():
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["cim"]["cards"] = {"c": {"kind": "analog_macro", "bank_depth": 3}}
    with pytest.raises(ValueError, match="bank_depth"):
        _hw_from_dict(hw_dict)


def test_card_validity_menu_refuses_unlisted_points():
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["cim"]["cards"] = {
        "c": {
            "kind": "analog_macro",
            "bits_per_cell": 2,
            "weight_bits": 8,
            "validity": [{"bits_per_cell": 2, "weight_bits": 8, "mux": 4}],
        }
    }
    dev = _device_from_dicts(hw_dict, VIT_HUGE_64)
    dev.check_card_point()  # the card's own point is on the menu
    with pytest.raises(cim_timing.CardValidityError, match="does not admit"):
        dev.check_card_point(bits_per_cell=3, weight_bits=8, mux=4)
    with pytest.raises(cim_timing.CardValidityError, match="does not admit"):
        dev.check_card_point(bits_per_cell=2, weight_bits=8, mux=8)

    # The card's OWN point must be on its own menu, or the config is refused.
    hw_dict["cim"]["cards"]["c"]["bits_per_cell"] = 4
    with pytest.raises(ValueError, match="validity menu"):
        _hw_from_dict(hw_dict)


def test_card_without_validity_menu_admits_everything():
    # An empty menu means the card declares none; nothing is refused, which
    # is what keeps every shipped YAML working unchanged.
    hw = config.parse_config(str(FWS_T1), "hardware")
    model = config.parse_config(str(VIT_HUGE_64), "VIT").model_config
    dev = cim_timing.CimDeviceModel(hw, model)
    dev.check_card_point(bits_per_cell=3, weight_bits=17, mux=999)


@pytest.mark.parametrize("device", ["reram", "mram"])
def test_empty_device_slots_ship_no_numbers(device):
    # ADJ-4: ReRAM/MRAM are NAMED EMPTY SLOTS. The schema admits them; no
    # parameters are shipped, so nothing is inherited into one.
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["cim"]["cards"] = {"second": {"kind": "analog_macro", "device": device}}
    with pytest.raises(ValueError, match="NAMED EMPTY CARD SLOT"):
        _hw_from_dict(hw_dict)

    # With a complete params block of its own it parses — the schema is
    # genuinely device-generic, and every number came from the user.
    hw_dict["cim"]["cards"]["second"]["params"] = {
        "rows": 512,
        "cols_adc": 128,
        "adc_mux": 2,
        "slice_cycles": 1,
        "analog_clock_mhz": 200,
        "energy_per_vec_pj": 1.0,
        "area_mm2_per_array": 0.5,
    }
    cim = _hw_from_dict(hw_dict).cim_config
    card = cim.analog_card
    assert card.device == device
    assert card.params.rows == 512 and card.params.adc_mux == 2
    assert card.params is not cim.analog  # its own parameter set, nothing inherited


def test_custom_device_family_is_admitted_and_inherits_nothing():
    # D14 says device CARDS, not device rewrites — but an open `device` field
    # cannot tell a new device from a typo. 'custom' is the explicit escape
    # hatch: it is admitted by name and pays the empty-slot price.
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["cim"]["cards"] = {"second": {"kind": "analog_macro", "device": "custom"}}
    with pytest.raises(ValueError, match="must supply a complete 'params' block"):
        _hw_from_dict(hw_dict)

    hw_dict["cim"]["cards"]["second"]["params"] = {
        "rows": 256,
        "cols_adc": 64,
        "adc_mux": 4,
        "slice_cycles": 2,
        "analog_clock_mhz": 150,
        "energy_per_vec_pj": 2.0,
        "area_mm2_per_array": 0.25,
    }
    cim = _hw_from_dict(hw_dict).cim_config
    card = cim.analog_card
    assert card.device == "custom"
    assert card.params.rows == 256 and card.params.adc_mux == 4
    assert card.params is not cim.analog  # nothing inherited under another name


def test_a_typo_device_is_still_refused_and_names_the_escape_hatch():
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["cim"]["cards"] = {"c": {"kind": "analog_macro", "device": "crt"}}
    with pytest.raises(ValueError, match="device: 'custom'"):
        _hw_from_dict(hw_dict)


def test_card_properties_refuse_a_hand_built_config_by_name():
    # cards=None is unreachable through from_dict (it always synthesizes a
    # library); it is reachable by assembling the dataclass by hand, and used
    # to surface as a bare NoneType AttributeError.
    hw = config.parse_config(str(FWS_T1), "hardware")
    cim = hw.cim_config
    bare = config.CIMConfig(analog=cim.analog, fabric=cim.fabric, chip=cim.chip)
    assert bare.cards is None
    for name in ("analog_card", "digital_card"):
        with pytest.raises(config.MissingCardLibraryError, match="carries no card library"):
            getattr(bare, name)
    # AttributeError is one of its bases, so the one optional probe on the card
    # path keeps its default instead of turning into a traceback.
    assert getattr(bare, "analog_card", None) is None


def test_unknown_card_kind_and_device_rejected():
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["cim"]["cards"] = {"c": {"kind": "quantum"}}
    with pytest.raises(ValueError, match="kind must be one of"):
        _hw_from_dict(hw_dict)
    hw_dict["cim"]["cards"] = {"c": {"kind": "analog_macro", "device": "flash"}}
    with pytest.raises(ValueError, match="device must be one of"):
        _hw_from_dict(hw_dict)


def test_card_default_selection_names_a_declared_card():
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["cim"]["cards"] = {
        "a": {"kind": "analog_macro"},
        "b": {"kind": "analog_macro", "bank_depth": 1},
        "default_analog": "b",
    }
    assert _hw_from_dict(hw_dict).cim_config.analog_card.name == "b"
    hw_dict["cim"]["cards"]["default_analog"] = "missing"
    with pytest.raises(ValueError, match="names no analog_macro card"):
        _hw_from_dict(hw_dict)


def test_stack_height_divides_footprint_only(cim_t1):
    assert cim_t1.macro_footprint_mm2() == pytest.approx(cim_t1.analog.area_mm2_per_array)
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["cim"]["cards"] = {"c": {"kind": "analog_macro", "stack_3d_height": 4}}
    dev = _device_from_dicts(hw_dict, VIT_HUGE_64)
    assert dev.macro_footprint_mm2() == pytest.approx(dev.analog.area_mm2_per_array / 4)
    # Silicon area and energy are untouched: height divides FOOTPRINT only.
    assert dev.total_area_mm2() == pytest.approx(cim_t1.total_area_mm2())
    assert dev.transformer_stack_energy_pj() == pytest.approx(
        cim_t1.transformer_stack_energy_pj()
    )


# --- P2.2 tiles: object, enumerator, capacity ------------------------------


TILE_SHAPES = ((1280, 3840), (1280, 1280), (1280, 5120), (5120, 1280), (1000, 7), (4096, 11008))


@pytest.mark.parametrize("k,n", TILE_SHAPES)
def test_tile_enumerator_reproduces_the_array_census(cim_t1, k, n):
    owner = cim_timing.TileOwner(model="vit_huge", layer=3, op="qkv", shard=0)
    tiles = cim_t1.enumerate_tiles(k, n, owner)
    macros = {tile.site.macro_id for tile in tiles}
    # The census law IS the tile enumerator at the shipped defaults.
    assert len(tiles) == cim_t1.arrays(k, n)
    assert len(macros) == cim_t1.arrays(k, n)
    assert cim_t1.tile_macro_count(k, n) == cim_t1.arrays(k, n)
    # Every tile has a named owner and a site (D21).
    mux = cim_t1.analog.adc_mux
    for tile in tiles:
        assert tile.owner is owner
        assert tile.slice_index == 0
        assert tile.site.column_sets == tuple(range(mux))  # dedicated: the whole macro
        assert tile.site.row_end == tile.k_end and tile.site.row_start == tile.k_start
    # The tiles partition the matrix: no gap, no overlap.
    covered = sorted((tile.k_start, tile.k_end, tile.n_start, tile.n_end) for tile in tiles)
    assert len(set(covered)) == len(covered)
    assert sum((t.k_end - t.k_start) * (t.n_end - t.n_start) for t in tiles) == k * n


def test_tile_owner_label_names_model_layer_op_expert_shard():
    owner = cim_timing.TileOwner(model="moe_small", layer=7, op="ffn1", expert=3, shard=1)
    assert owner.label == "moe_small.L7.ffn1.e3.s1"
    assert cim_timing.TileOwner(model="m", layer=0, op="qkv").label == "m.L0.qkv.s0"


def test_finer_banks_let_tiles_co_reside_in_one_macro(cim_t1):
    # ADJ-4: the mux slot is the smallest allocatable unit. One column set per
    # tile lets four owners share a mux-4 macro.
    mux = cim_t1.analog.adc_mux
    tiles = cim_t1.enumerate_tiles(
        1280, 4 * cim_t1.analog.cols_adc, cim_timing.TileOwner(op="qkv"), column_sets_per_tile=1
    )
    assert len(tiles) == 4
    assert {tile.site.macro_id for tile in tiles} == {0}
    assert [tile.site.column_sets for tile in tiles] == [(i,) for i in range(mux)]
    cim_t1.validate_macro_capacity(tiles)


def test_enumerator_rejects_a_bank_that_does_not_divide_mux(cim_t1):
    with pytest.raises(ValueError, match="smallest allocatable unit"):
        cim_t1.enumerate_tiles(1280, 1280, cim_timing.TileOwner(), column_sets_per_tile=3)


def test_macro_capacity_overflow_is_a_named_hard_error(cim_t1):
    mux = cim_t1.analog.adc_mux
    site = cim_timing.TileSite(macro_id=0, row_start=0, row_end=1280, column_sets=(0,))
    tile_a = cim_timing.Tile(
        owner=cim_timing.TileOwner(model="a", op="qkv"),
        k_start=0, k_end=1280, n_start=0, n_end=320, slice_index=0, site=site,
    )
    tile_b = cim_timing.Tile(
        owner=cim_timing.TileOwner(model="b", op="ffn1"),
        k_start=0, k_end=1280, n_start=0, n_end=320, slice_index=0, site=site,
    )
    # Two owners on one column set: the column set holds one tile.
    with pytest.raises(cim_timing.MacroCapacityError, match="claimed by both"):
        cim_t1.validate_macro_capacity([tile_a, tile_b])
    # A column set the card does not have.
    over = cim_timing.Tile(
        owner=cim_timing.TileOwner(model="c", op="ffn2"),
        k_start=0, k_end=1280, n_start=0, n_end=320, slice_index=0,
        site=cim_timing.TileSite(macro_id=0, row_start=0, row_end=1280, column_sets=(mux,)),
    )
    with pytest.raises(cim_timing.MacroCapacityError, match="column sets"):
        cim_t1.validate_macro_capacity([over])
    # A legal full macro passes.
    cim_t1.validate_macro_capacity(cim_t1.enumerate_tiles(1280, 1280, tile_a.owner))


def test_allocation_block_parses_and_capacity_checks(cim_t1):
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["cim"]["allocation"] = {
        "column_sets_per_tile": 1,
        "assignments": [
            {"model": "vit", "layer": 0, "op": "qkv", "macro": 0, "column_sets": [0, 1]},
            {"model": "vit", "layer": 1, "op": "ffn1", "macro": 0, "column_sets": [2, 3]},
        ],
    }
    dev = _device_from_dicts(hw_dict, VIT_HUGE_64)
    assert dev.column_sets_per_tile == 1
    pairs = dev.allocation_sites()
    assert [owner.label for owner, _ in pairs] == ["vit.L0.qkv.s0", "vit.L1.ffn1.s0"]
    dev.validate_allocation()  # two owners, four disjoint column sets, one macro

    # Overlapping claims are the named hard error, and it fires while the
    # hardware config is PARSED: a user-written allocation is validated on the
    # run path (D10), not only when a caller reaches for the device model.
    hw_dict["cim"]["allocation"]["assignments"][1]["column_sets"] = [1, 2]
    with pytest.raises(cim_timing.MacroCapacityError, match="claimed by both"):
        _hw_from_dict(hw_dict)


def test_allocation_out_of_range_column_set_is_refused_at_parse_time():
    hw_dict = _load_yaml(FWS_T1)
    mux = int(hw_dict["cim"]["analog"]["adc_mux"])
    hw_dict["cim"]["allocation"] = {
        "column_sets_per_tile": 1,
        "assignments": [
            {"model": "vit", "layer": 0, "op": "qkv", "macro": 0, "column_sets": [mux + 95]},
        ],
    }
    with pytest.raises(cim_timing.MacroCapacityError, match="column sets \\(mux slots\\)"):
        _hw_from_dict(hw_dict)


def test_allocation_granularity_must_divide_mux():
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["cim"]["allocation"] = {"column_sets_per_tile": 3}
    with pytest.raises(ValueError, match="smallest allocatable unit"):
        _hw_from_dict(hw_dict)


def test_allocation_absent_is_dedicated_per_matrix(cim_t1):
    # ABSENT means today's behavior, bit-identically: the default bank is the
    # whole macro and one owner activates every mux slot.
    assert cim_t1.allocation is None
    assert cim_t1.column_sets_per_tile == cim_t1.analog.adc_mux
    tiles = cim_t1.enumerate_tiles(1280, 3840, cim_timing.TileOwner(op="qkv"))
    assert cim_t1.active_column_sets(tiles) == cim_t1.analog.adc_mux


# --- P2.3 active-column-set pricing (ADJ-4) --------------------------------


@pytest.mark.parametrize("hw_path,model_path,mode", SHIPPED_FWS_PAIRS)
def test_full_occupancy_identity_reproduces_the_pass_one_charge(hw_path, model_path, mode):
    # THE required identity: one owner filling every mux slot pays exactly the
    # pass-1 charge — same time (bit-identical, not approximate) and the same
    # analog energy the census law gives.
    hw = config.parse_config(str(hw_path), "hardware")
    model = config.parse_config(str(model_path), mode).model_config
    dev = cim_timing.CimDeviceModel(hw, model)
    m_tokens = 64
    for stage, (k, n) in dev.per_layer_stage_shapes().items():
        tiles = dev.enumerate_tiles(k, n, cim_timing.TileOwner(model=stage, op=stage))
        cost = dev.price_tiled_op(m_tokens, tiles)
        assert cost.active_column_sets == dev.analog.adc_mux, stage
        assert cost.time_s == dev.analog_gemm_time(m_tokens), stage
        assert cost.macros == dev.arrays(k, n), stage
        assert cost.energy_pj == pytest.approx(
            dev.analog_stage_energy_pj(m_tokens, dev.arrays(k, n)), rel=1e-12
        ), stage


#: The one shipped parity stage whose own N does not fill a macro. The
#: dedicated default still gives it the whole macro, so the parity charge is
#: unmoved — but the configuration is not physically full-occupancy there.
UNDERFILLED_PARITY_STAGES = ("router",)


@pytest.mark.parametrize("hw_path,model_path,mode", SHIPPED_FWS_PAIRS)
def test_parity_configs_are_full_occupancy_configurations(hw_path, model_path, mode):
    # ADJ-4 open question 4. The enumerator's default hands one owner a whole
    # macro, so asserting active_column_sets == mux would assert a tautology.
    # The load-bearing statement is PHYSICAL and independent of the enumerator:
    # each stage's own N spans at least mux column sets, so nothing under-fills
    # a macro — with the MoE router named as the single exception.
    hw = config.parse_config(str(hw_path), "hardware")
    model = config.parse_config(str(model_path), mode).model_config
    dev = cim_timing.CimDeviceModel(hw, model)
    assert dev.n_slices == 1 and dev.allocation is None
    shapes = dict(dev.per_layer_stage_shapes())
    p = dev.params
    if p.use_moe:
        shapes["router"] = (p.hidden_dim, p.num_experts)
        shapes["moe_ffn1"] = (p.hidden_dim, p.ffn1_fold * p.moe_intermediate)
        shapes["moe_ffn2"] = (p.moe_intermediate, p.hidden_dim)
    if p.lm_head_enabled:
        shapes["lm_head"] = (p.hidden_dim, p.vocab_size)
    if p.is_vit_shaped:
        shapes["patch_embed"] = (p.patch_dim, p.hidden_dim)
    cols_adc = int(dev.analog.cols_adc)
    mux = int(dev.analog.adc_mux)
    for stage, (k, n) in shapes.items():
        physical_sets = -(-int(n) // cols_adc)
        if stage in UNDERFILLED_PARITY_STAGES:
            assert physical_sets < mux, stage
        else:
            assert physical_sets >= mux, stage
        # And the dedicated default charges the whole macro either way, which
        # is why active-set pricing leaves every parity number where it was.
        tiles = dev.enumerate_tiles(k, n, cim_timing.TileOwner(op=stage))
        assert dev.active_column_sets(tiles) == mux, stage


def test_active_column_set_pricing_charges_only_the_sets_it_activates(cim_t1):
    # The adopted law change: a half-occupied macro pays half the mux depth.
    mux = cim_t1.analog.adc_mux
    owner = cim_timing.TileOwner(model="vit", layer=0, op="qkv")
    half = cim_t1.enumerate_tiles(
        1280, 2 * cim_t1.analog.cols_adc, owner, column_sets_per_tile=1
    )
    assert cim_t1.active_column_sets(half) == mux // 2
    assert cim_t1.price_tiled_op(64, half).time_s == pytest.approx(
        cim_t1.analog_gemm_time(64) / 2, rel=1e-12
    )
    # And the co-resident owner pays its own sets, not the whole macro's.
    other = cim_timing.TileOwner(model="vit", layer=1, op="ffn1")
    rest = tuple(
        cim_timing.Tile(
            owner=other, k_start=t.k_start, k_end=t.k_end, n_start=t.n_start, n_end=t.n_end,
            slice_index=0,
            site=cim_timing.TileSite(
                macro_id=0, row_start=t.k_start, row_end=t.k_end,
                column_sets=tuple(c + mux // 2 for c in t.site.column_sets),
            ),
        )
        for t in half
    )
    cim_t1.validate_macro_capacity(half + rest)  # they co-reside legally
    assert cim_t1.active_column_sets(rest) == mux // 2


def test_analog_op_time_is_the_generalized_vector_latency(cim_t1):
    # vec_latency = mux * slice_cycles / f generalizes to active sets.
    for sets in range(1, cim_t1.analog.adc_mux + 1):
        expected = (
            17 * sets * cim_t1.analog.slice_cycles / (cim_t1.analog.analog_clock_mhz * 1e6)
        )
        assert cim_t1.analog_op_time(17, sets) == pytest.approx(expected, rel=1e-12)
    # The identity is bit-identical, not approximate, at every token count:
    # analog_op_time groups its arithmetic exactly as vec_latency_s does.
    for m_tokens in (1, 17, 64, 197, 1792):
        assert cim_t1.analog_op_time(m_tokens, cim_t1.analog.adc_mux) == (
            cim_t1.analog_gemm_time(m_tokens)
        )


def test_macros_hosting_an_op_fire_in_parallel(cim_t1):
    # The charge is the WORST macro's active sets, not the sum: the K/N-free
    # law says every macro holding the op's tiles converts concurrently.
    owner = cim_timing.TileOwner(op="ffn1")
    wide = cim_t1.enumerate_tiles(1280, 40 * cim_t1.analog.cols_adc, owner)
    assert len({t.site.macro_id for t in wide}) == 10
    assert cim_t1.active_column_sets(wide) == cim_t1.analog.adc_mux
    assert cim_t1.price_tiled_op(64, wide).time_s == cim_t1.analog_gemm_time(64)


# --- P2.4 bit slicing, priced (D11) ----------------------------------------


def test_column_set_slicing_multiplies_stored_column_demand():
    dev = _sliced_t1_device()
    assert dev.n_slices == 4
    for k, n in TILE_SHAPES[:4]:
        tiles = dev.enumerate_tiles(k, n, cim_timing.TileOwner(op="qkv"))
        macros = {tile.site.macro_id for tile in tiles}
        # Stored-column demand x n_s (the column-set arrangement).
        assert len(macros) == dev.arrays(k, n) * dev.n_slices
        assert sorted({t.slice_index for t in tiles}) == [0, 1, 2, 3]
        dev.validate_macro_capacity(tiles)


def test_slicing_off_forces_the_degenerate_census(cim_t1):
    dev = _sliced_t1_device()
    # n_slices = 1 collapses the sliced enumerator back onto the census.
    assert dev.tile_macro_count(1280, 3840, n_slices=1, column_sets_per_tile=4) == cim_t1.arrays(
        1280, 3840
    )
    assert dev.reduction_descriptors(
        dev.enumerate_tiles(1280, 3840, cim_timing.TileOwner(), n_slices=1)
    ) == ()


def test_sliced_op_still_pays_one_adc_pass_per_active_set():
    # One ADC pass per column set: the four slices of an output block occupy
    # the four mux slots of one macro, so the op pays mux * slice_cycles —
    # the full-occupancy charge, with four times the stored columns.
    dev = _sliced_t1_device()
    owner = cim_timing.TileOwner(model="vit", layer=0, op="qkv")
    tiles = dev.enumerate_tiles(1280, 3840, owner)
    cost = dev.price_tiled_op(64, tiles)
    assert cost.active_column_sets == dev.analog.adc_mux
    assert cost.time_s == dev.analog_gemm_time(64)
    assert cost.energy_pj == pytest.approx(
        dev.n_slices * dev.analog_stage_energy_pj(64, dev.arrays(1280, 3840)), rel=1e-12
    )


def test_reduction_descriptors_are_emitted_per_slice_group():
    dev = _sliced_t1_device()
    owner = cim_timing.TileOwner(model="vit", layer=2, op="ffn1")
    tiles = dev.enumerate_tiles(1280, 3840, owner)
    descriptors = dev.reduction_descriptors(tiles)
    # One tree per (row block, output block): 1 x ceil(3840/320) here.
    assert len(descriptors) == 12
    for descriptor in descriptors:
        assert descriptor.kind == "shift_add_tree"
        assert descriptor.arrangement == "column_sets"
        assert descriptor.owner is owner
        assert descriptor.n_slices == 4
        assert descriptor.adds_per_output == 3          # n_s - 1
        assert descriptor.depth == 2                    # ceil(log2(n_s))
        assert descriptor.output_lanes == dev.analog.cols_adc
        assert len(descriptor.operand_tiles) == 4
        assert [t.slice_index for t in descriptor.operand_tiles] == [0, 1, 2, 3]
        # Column-set arrangement: all four slices sit in the sink macro, so
        # the tree is local to that macro's digital pool and nothing travels.
        assert descriptor.local and descriptor.transport_partials == 0
        assert {t.site.macro_id for t in descriptor.operand_tiles} == {
            descriptor.site_macro_id
        }


def test_column_set_slice_groups_never_straddle_a_macro():
    # D11 defines the column-set arrangement as a slice group reduced by ONE
    # macro's digital pool. Linear packing broke that whenever n_s did not
    # divide the macro's tile slots (n_s = 3 into 4 slots), and the descriptors
    # then charged p2p transport for an arrangement that needs none.
    dev = _sliced_t1_device(bits_per_cell=3, weight_bits=8)
    assert dev.n_slices == 3
    tiles = dev.enumerate_tiles(1280, 1280, cim_timing.TileOwner(op="qkv"))
    by_group = {}
    for tile in tiles:
        by_group.setdefault((tile.k_start, tile.n_start), set()).add(tile.site.macro_id)
    assert all(len(macros) == 1 for macros in by_group.values()), by_group
    for descriptor in dev.reduction_descriptors(tiles):
        assert descriptor.local and descriptor.transport_partials == 0
    dev.validate_macro_capacity(tiles)


def test_column_set_arrangement_refuses_a_group_it_cannot_keep_local():
    # A whole-macro bank leaves one tile slot, so four slices cannot be local.
    # A named refusal, not a silently non-local placement.
    dev = _sliced_t1_device(bank_depth=0)
    with pytest.raises(ValueError, match="cannot keep a slice group local"):
        dev.enumerate_tiles(1280, 1280, cim_timing.TileOwner(op="qkv"))


def test_bank_switch_cost_is_a_declared_card_figure(cim_t1):
    # AUDIT finding 3: OPTIMA priced bank switching at zero silently. Zero is
    # still the shipped number, but it is now the card's declared number.
    assert cim_t1.card.bank_switch_cycles == 0
    dev = _sliced_t1_device(bank_switch_cycles=3, bits_per_cell=8, weight_bits=8)
    assert dev.n_slices == 1
    mux = int(dev.analog.adc_mux)
    base_cycles = mux * int(dev.analog.slice_cycles)
    expected = 64 * ((base_cycles + (mux - 1) * 3) / (float(dev.analog.analog_clock_mhz) * 1e6))
    assert dev.analog_op_time(64, mux) == pytest.approx(expected, rel=1e-12)
    # One active set switches nothing.
    assert dev.analog_op_time(64, 1) == cim_t1.analog_op_time(64, 1)


def test_shared_digital_chiplet_area_is_accounted_and_never_smeared():
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["cim"]["cards"] = {
        "ctt": {"kind": "analog_macro", "device": "ctt"},
        "sa": {"kind": "digital_chiplet", "area_mm2": 12.0},
    }
    dev = _device_from_dicts(hw_dict, VIT_HUGE_64)
    assert dev.shared_digital_area_mm2() == pytest.approx(12.0)
    # The analog total is the OPTIMA parity accounting and does not absorb it.
    assert dev.system_area_mm2() == pytest.approx(dev.total_area_mm2() + 12.0)
    assert dev.total_area_mm2() < dev.system_area_mm2()


def test_digital_chiplet_energy_knob_is_refused_by_name():
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["cim"]["cards"] = {"sa": {"kind": "digital_chiplet", "energy_per_op_pj": 1.0}}
    with pytest.raises(ValueError, match="energy_per_op_pj is not a card field"):
        _hw_from_dict(hw_dict)


def test_derived_pool_sizing_is_reported(cim_t1):
    # D12: the derived sizing is REPORTED. The line names lanes, adders, the
    # rate they consume and the footprint share, and it moves no number.
    sizing = cim_t1.digital_pool_sizing()
    line = cim_t1.report_digital_pool(sizing)
    assert str(sizing.lanes) in line and str(sizing.adders) in line
    assert "derived, D12" in line


def test_a_real_run_reports_the_derived_pool_sizing():
    # And the report reaches a user: an ordinary run prints it. On stdout only
    # — the report file and the results txt stay the frozen closed-form
    # accounting (ADJ-8), which the P6.2 byte-identity gate checks.
    proc = _run_perf_subprocess(FWS_T1, VIT_HUGE_64)
    assert proc.returncode == 0, proc.stdout[-4000:]
    assert "[FWS-CIM] per-macro digital pool (derived, D12)" in proc.stdout


def test_chained_macro_arrangement_puts_partials_on_the_p2p_law():
    dev = _sliced_t1_device(slicing="chained_macros")
    owner = cim_timing.TileOwner(model="vit", layer=2, op="ffn2")
    tiles = dev.enumerate_tiles(1280, 3840, owner, arrangement="chained_macros")
    descriptors = dev.reduction_descriptors(tiles)
    assert len(descriptors) == 12
    descriptor = descriptors[0]
    assert descriptor.arrangement == "chained_macros"
    assert not descriptor.local
    assert descriptor.transport_partials == descriptor.n_slices - 1
    assert len({t.site.macro_id for t in descriptor.operand_tiles}) == descriptor.n_slices
    cost = dev.price_reduction(descriptor, 64, act_bytes=2.0)
    assert cost.transport_bytes == pytest.approx(64 * descriptor.output_lanes * 3 * 2.0)
    # The partials ride the existing p2p law; the tree waits at the sink.
    assert dev.p2p_time_s(cost.transport_bytes, 100e9, 1e-6) > 0


def test_shift_add_tree_pricing_law():
    dev = _sliced_t1_device(pool_energy_per_add_pj=0.25)
    descriptor = dev.reduction_descriptors(
        dev.enumerate_tiles(1280, 1280, cim_timing.TileOwner(op="qkv"))
    )[0]
    pool = dev.digital_pool_sizing()
    m_tokens = 64
    cost = dev.price_reduction(descriptor, m_tokens)
    results = m_tokens * descriptor.output_lanes
    # ~(n_s - 1) adds per output; pipelined, one result per lane per cycle
    # after a fill of depth ~log2(n_s).
    assert cost.adds == results * (descriptor.n_slices - 1)
    assert cost.cycles == descriptor.depth + math.ceil(results / pool.lanes) - 1
    assert cost.time_s == pytest.approx(cost.cycles / pool.pool_clock_hz, rel=1e-12)
    assert cost.energy_pj == pytest.approx(cost.adds * 0.25, rel=1e-12)
    assert cost.transport_bytes == 0.0


def test_reduction_cost_terms_are_zero_until_a_card_declares_them():
    # ADJ-4 / D23 honesty: no invented numbers. A card that declares no pool
    # energy reports zero energy, exactly as area_mm2_per_array: 0 does.
    dev = _sliced_t1_device()
    descriptor = dev.reduction_descriptors(
        dev.enumerate_tiles(1280, 1280, cim_timing.TileOwner(op="qkv"))
    )[0]
    assert dev.price_reduction(descriptor, 64).energy_pj == 0.0
    assert dev.digital_pool_sizing().area_mm2 == 0.0


# --- P2.5 the per-macro digital pool (D12) ---------------------------------


def test_digital_pool_lanes_come_from_the_peak_result_rate(cim_t1):
    sizing = cim_t1.digital_pool_sizing()
    expected_rate = (
        cim_t1.analog.cols_adc * cim_t1.analog.analog_clock_mhz * 1e6 / cim_t1.analog.slice_cycles
    )
    assert sizing.result_rate_per_s == pytest.approx(expected_rate, rel=1e-12)
    assert sizing.pool_clock_hz == cim_t1.f_fabric_hz  # inherits the digital card's clock
    assert sizing.lanes == math.ceil(expected_rate / sizing.pool_clock_hz)
    # No slicing => no shift-add tree to hold.
    assert sizing.adders == 0 and sizing.area_mm2 == 0.0
    assert not sizing.disclose and sizing.note is None


def test_digital_pool_adders_scale_with_the_slice_count():
    dev = _sliced_t1_device(pool_area_mm2_per_adder=0.001, pool_clock_ghz=2.0)
    sizing = dev.digital_pool_sizing()
    assert sizing.pool_clock_hz == pytest.approx(2.0e9)
    assert sizing.lanes == math.ceil(sizing.result_rate_per_s / 2.0e9)
    assert sizing.adders == sizing.lanes * (dev.n_slices - 1)
    assert sizing.area_mm2 == pytest.approx(sizing.adders * 0.001, rel=1e-12)


def test_pool_disclosure_threshold_prints_and_changes_nothing(capsys):
    # ADJ-4: 1/5 of the served macro footprint is a DISCLOSURE threshold.
    small = _sliced_t1_device(pool_area_mm2_per_adder=1e-6)
    assert small.digital_pool_sizing().area_share < cim_timing.POOL_DISCLOSURE_AREA_SHARE
    capsys.readouterr()
    assert small.disclose_digital_pool() is None
    assert capsys.readouterr().out == ""

    loud = _sliced_t1_device(pool_area_mm2_per_adder=0.02)
    sizing = loud.digital_pool_sizing()
    assert sizing.area_share >= cim_timing.POOL_DISCLOSURE_AREA_SHARE and sizing.disclose
    capsys.readouterr()
    note = loud.disclose_digital_pool(sizing)
    out = capsys.readouterr().out
    assert "[NOTE]" in out and "disclosure threshold 20%" in out
    assert note is not None and "not a constraint" in note
    # Reported, never a constraint: nothing else moves.
    assert loud.analog_gemm_time(64) == _sliced_t1_device().analog_gemm_time(64)
    assert loud.total_area_mm2() == _sliced_t1_device().total_area_mm2()


def test_pool_share_uses_the_stacked_footprint():
    flat = _sliced_t1_device(pool_area_mm2_per_adder=0.001)
    stacked = _sliced_t1_device(pool_area_mm2_per_adder=0.001, stack_3d_height=4)
    assert stacked.digital_pool_sizing().area_share == pytest.approx(
        4 * flat.digital_pool_sizing().area_share, rel=1e-12
    )


# ---------------------------------------------------------------------------
# P6.2 degenerate-reduction gates: slicing off + single owner + no allocation
# block reproduces today's full report, bit-identically.
# ---------------------------------------------------------------------------


P6_2_GATE_PAIRS = (
    (FWS_T1, VIT_HUGE_64, "VIT"),
    (FWS_LLAMA7B, LLAMA2_7B_FWS_INF, "LLM"),
    (FWS_MOE, MOE_SMALL_FWS_INF, "LLM"),
)


def _inert_p2_blocks(hw_dict):
    """An explicit card + allocation block that must change nothing.

    Slicing off (no bits_per_cell / weight_bits), one owner per macro
    (bank = the whole mux depth), and an allocation block that names the same
    granularity the absent block implies.
    """
    mux = int(hw_dict["cim"]["analog"]["adc_mux"])
    hw_dict["cim"]["cards"] = {
        "ctt_explicit": {"kind": "analog_macro", "device": "ctt", "bank_depth": mux},
        "shared_digital": {"kind": "digital_chiplet"},
    }
    hw_dict["cim"]["allocation"] = {"column_sets_per_tile": mux}
    return hw_dict


@pytest.mark.parametrize("hw_path,model_path,mode", P6_2_GATE_PAIRS)
def test_p6_2_degenerate_reduction_reproduces_today_bit_identically(
    hw_path, model_path, mode, tmp_path
):
    proc = _run_perf_subprocess(hw_path, model_path)
    assert proc.returncode == 0, proc.stdout[-4000:]
    report_path = _perf_out(mode, "fws_cim_report.json")
    results_path = _perf_out(mode, "LLM_inference_results.txt")
    baseline_report = report_path.read_bytes()
    baseline_results = results_path.read_bytes()

    explicit_path = tmp_path / f"{Path(hw_path).stem}_p2_explicit.yaml"
    explicit_path.write_text(yaml.safe_dump(_inert_p2_blocks(_load_yaml(hw_path))))
    proc = _run_perf_subprocess(explicit_path, model_path)
    assert proc.returncode == 0, proc.stdout[-4000:]

    # Byte-for-byte: the P2 layer is inert on today's configurations.
    assert report_path.read_bytes() == baseline_report
    assert results_path.read_bytes() == baseline_results


@pytest.mark.parametrize("hw_path,model_path,mode", P6_2_GATE_PAIRS)
def test_p6_2_degenerate_laws_match_the_pass_one_laws(hw_path, model_path, mode):
    # The same gate at law level, where a bit-identity claim is exact rather
    # than formatted: every pass-1 law the report prints is untouched by the
    # explicit-but-inert card and allocation blocks.
    base = _device_from_dicts(_load_yaml(hw_path), model_path, mode)
    explicit = _device_from_dicts(_inert_p2_blocks(_load_yaml(hw_path)), model_path, mode)
    assert explicit.n_slices == 1
    assert explicit.column_sets_per_tile == base.analog.adc_mux
    assert explicit.vec_cycles == base.vec_cycles
    assert explicit.vec_latency_s == base.vec_latency_s
    assert explicit.total_arrays() == base.total_arrays()
    assert explicit.total_area_mm2() == base.total_area_mm2()
    assert explicit.chip_layer_counts() == base.chip_layer_counts()
    assert explicit.chip_array_usage() == base.chip_array_usage()
    assert dict(explicit.all_stage_times()) == dict(base.all_stage_times())
    assert explicit.pipeline_period() == base.pipeline_period()
    assert explicit.transformer_stack_energy_pj() == base.transformer_stack_energy_pj()
    # And the tiled path reproduces the untiled one, stage by stage.
    for stage, (k, n) in base.per_layer_stage_shapes().items():
        tiles = explicit.enumerate_tiles(k, n, cim_timing.TileOwner(op=stage))
        assert explicit.price_tiled_op(64, tiles).time_s == base.analog_gemm_time(64), stage
        assert len(tiles) == base.arrays(k, n), stage


# ---------------------------------------------------------------------------
# ADJ-10: the attention fabric and the softmax pipeline derive too
# ---------------------------------------------------------------------------


def test_the_fold_split_reproduces_the_pass_one_law_at_two_arrays(cim_t1):
    """ADJ-10's generalization must be INERT at the declared width.

    NEW GATE. Every shipped card declares ``num_arrays: 2`` and every OPTIMA
    parity configuration is priced through this law, so a fold split that moved
    a cycle at 2 arrays would move a validated number. At 2 the only partition
    is (1, 1), ``ceil(folds / 1) == folds``, and both expressions are the
    pass-1 ones character for character.
    """
    assert cim_t1.fabric_num_arrays == 2
    timing = cim_t1.attention_timing(seq_len=64, head_dim=64, kv_heads=16, tp=1)
    h_rep = timing.heads_per_replica
    assert (timing.qk_arrays, timing.pv_arrays) == (1, 1)
    assert timing.qk_folds_per_array == timing.pv_folds_per_array == timing.folds
    assert timing.qk_cycles == cim_t1.sa_cycles(64, 64, 64 * h_rep)
    assert timing.pv_cycles == cim_t1.sa_cycles(64, 64, 64 * h_rep)


def test_more_arrays_buy_more_concurrent_folds_and_then_stop(cim_t1):
    """The fold law, hand-computed, including where it stops buying.

    NEW GATE (ADJ-10). ``cim_t1`` is the OPTIMA T1 point: 32 x 64 arrays,
    ViT-Huge at 16 kv heads and head_dim 64, one replica, so a prefill call at
    seq 64 folds 16 heads into K.

        R = 32, C = 64, m = n = 64, k = 64, folds = 16
        QK(a) = ceil(64/32) * ceil(64/64) * (64 * ceil(16/a) + 32 + 64 - 2) - 1
              = 2 * (64 * ceil(16/a) + 94) - 1

        a = 1 -> 2 * (1024 + 94) - 1 = 2235
        a = 2 -> 2 * ( 512 + 94) - 1 = 1211
        a = 4 -> 2 * ( 256 + 94) - 1 =  699
        a = 8 -> 2 * ( 128 + 94) - 1 =  443
        a = 16 -> 2 * (  64 + 94) - 1 =  315
        a = 17 -> ceil(16/17) = 1, still 315: the seventeenth array carries no
                  fold, and 315 is the DECLARED 32 x 64 geometry's own floor.
    """
    demand = cim_timing.AttentionCallDemand(
        m=64, k=64, n=64, folds=16, softmax_tokens=64, heads_chip=16
    )
    expected = {1: 2235, 2: 1211, 4: 699, 8: 443, 16: 315}
    for group, cycles in expected.items():
        # A group of `group` arrays on each side: num_arrays = 2 * group makes
        # the balanced partition exactly that, and the two runs are symmetric
        # here (m = n, k = n), so both sides land on the same number.
        qk, pv, _sm, a_qk, a_pv = cim_t1.attention_cycles_at(demand, 2 * group, 1)
        assert (a_qk, a_pv) == (group, group), group
        assert qk == pv == cycles, group
    assert demand.saturation_arrays == 32
    # AND THE SATURATION IS THE POINT: past 2 x folds nothing moves, so the
    # derivation has somewhere to stop that is not a cap. The 34-array and
    # 4096-array partitions are TIES (every extra array is idle whichever
    # group it joins), and the documented tie-break takes the smallest a_qk —
    # deterministic, and it costs no cycle.
    assert cim_t1.attention_cycles_at(demand, 32, 1)[:2] == (315, 315)
    assert cim_t1.attention_cycles_at(demand, 34, 1)[:2] == (315, 315)
    assert cim_t1.attention_cycles_at(demand, 4096, 1)[:2] == (315, 315)
    assert cim_t1.attention_cycles_at(demand, 34, 1)[3] == 16


def test_the_fold_split_minimises_the_serialised_pair(cim_t1):
    """The partition's objective, stated and checked on an ASYMMETRIC call.

    NEW GATE (ADJ-10). The lowering runs qk -> softmax -> pv in SERIES, so the
    stage's measured fabric time is the SUM and the partition minimises the
    sum. On a call where the two runs are lopsided the split must therefore be
    lopsided too, and a balanced split must be measurably worse.
    """
    demand = cim_timing.AttentionCallDemand(
        m=8, k=64, n=2048, folds=8, softmax_tokens=8, heads_chip=8
    )
    qk, pv, _sm, a_qk, a_pv = cim_t1.attention_cycles_at(demand, 6, 1)
    assert a_qk + a_pv == 6
    # Exhaustive: no other partition of 6 gives a smaller sum.
    best = min(
        cim_t1.sa_cycles(8, 2048, 64 * -(-8 // left))
        + cim_t1.sa_cycles(8, 64, 2048 * -(-8 // (6 - left)))
        for left in range(1, 6)
    )
    assert qk + pv == best
    # ... and the chosen split really is the argmin, not a coincidence of ties.
    assert (
        cim_t1.sa_cycles(8, 2048, 64 * -(-8 // a_qk))
        + cim_t1.sa_cycles(8, 64, 2048 * -(-8 // a_pv))
    ) == best


def test_a_card_that_pins_the_fabric_rides_a_disclosure_and_skips_the_derivation():
    """ADJ-10 keeps a DECLARED width legal on ONE machine, with its name on it.

    NEW GATE. D31 made a declared ``vector_lanes`` an OVERRIDE that rides a
    disclosure; ADJ-10 gives the fabric the same seam, under its own card
    fields, and the disclosure must say the word OVERRIDE and point at the
    axis-level refusal. A machine that pins nothing must carry no such note —
    a disclosure that always fires discloses nothing.
    """
    granite_hw = HW_DIR / "fws_cim_granite_tiny.yaml"
    raw = _load_yaml(granite_hw)
    plain = _hw_from_dict(raw)
    plain_card = plain.cim_config.cards.digital_card
    assert plain_card.has_fabric_override is False
    pinned_raw = copy.deepcopy(raw)
    card = pinned_raw["cim"]["cards"]["sa"]
    card["fabric_num_arrays"] = 6
    card["fabric_softmax_lanes"] = 3
    pinned = _hw_from_dict(pinned_raw)
    pinned_card = pinned.cim_config.cards.digital_card
    assert pinned_card.has_fabric_override is True
    assert pinned_card.fabric_num_arrays_effective == 6
    assert pinned_card.fabric_softmax_lanes_effective == 3
    model = config.parse_config(
        str(MODEL_DIR / "granite_4_0_h_tiny_inf.yaml"), "LLM"
    ).model_config
    plain_device = cim_timing.CimDeviceModel(plain, model)
    pinned_device = cim_timing.CimDeviceModel(pinned, model)
    assert plain_device.fabric_sizing_disclosures() == ()
    assert plain_device.fabric_num_arrays == 2
    notes = pinned_device.fabric_sizing_disclosures()
    assert len(notes) == 1 and "OVERRIDE" in notes[0]
    assert "REFUSED_AXES" in notes[0]
    assert pinned_device.fabric_num_arrays == 6
    assert pinned_device.softmax_width == 3
    # The pin WINS over a derivation, which is what makes it an override.
    fake = cim_timing.DerivedFabricSizing(
        analog_beat_s=1e-5, clock_hz=1e9, rows=32, cols=64, replicas=1,
        num_arrays=64, softmax_lanes=64, declared_num_arrays=2,
        declared_softmax_lanes=1, binding_stage=0, per_stage=(),
    )
    pinned_device.install_derived_fabric(fake)
    assert pinned_device.fabric_num_arrays == 6
    plain_device.install_derived_fabric(fake)
    assert plain_device.fabric_num_arrays == 64


def test_the_fabric_derivation_refuses_an_empty_demand_rather_than_inventing_one(cim_t1):
    """ADJ-10 derives FROM measured demand; with none there is no width.

    NEW GATE, and it is the ADJ-4 rule applied to a second engine: a run with
    no attention call in the beat asks nothing of the fabric, and answering
    with a number anyway would be an invention rather than a derivation.
    """
    with pytest.raises(cim_timing.FabricSizingError) as excinfo:
        cim_t1.derive_fabric_sizing(1e-5, ())
    assert "nothing to size" in str(excinfo.value)
    # A demand whose every entry is call-less is the same case, by the same name.
    with pytest.raises(cim_timing.FabricSizingError):
        cim_t1.derive_fabric_sizing(
            1e-5, (cim_timing.FabricDemand(stage=0, calls=(), analog_time_s=1e-6),)
        )


def test_the_fabric_derivation_stops_at_the_smallest_width_that_meets_the_target(cim_t1):
    """No margin (D28): reachable targets get the SMALLEST width, not the floor.

    NEW GATE. The saturation rule only applies where the analog m-pass cannot
    be reached. Where it CAN, ADJ-10 must behave exactly like ADJ-9 and stop at
    the first integer that fits — this test hands the same call a target it can
    reach and checks that the answer is smaller than the saturation width and
    that one array fewer would miss.
    """
    demand = cim_timing.FabricDemand(
        stage=0,
        calls=(
            cim_timing.AttentionCallDemand(
                m=64, k=64, n=64, folds=16, softmax_tokens=64, heads_chip=16
            ),
        ),
        # 699 + 699 + softmax fits in 1600 cycles at 4 arrays a side but not at 2.
        analog_time_s=1600 / cim_t1.f_fabric_hz,
    )
    sizing = cim_t1.derive_fabric_sizing(1e-3, (demand,), compose=False)
    assert sizing.num_arrays == 8
    assert sizing.saturated_stages == ()
    assert sizing.analog_bound_stages == (0,)
    assert sizing.target_ratio < 1.0
    row = sizing.binding_row
    assert row.qk_cycles == row.pv_cycles == 699
    assert row.used_cycles <= row.budget_cycles
    # One array fewer misses the budget, which is what "smallest" means.
    narrower = cim_t1._stage_attention_cycles(demand.calls, 7, row.softmax_width)[0]
    assert narrower > row.budget_cycles
    # ONLY THE COUNT MOVED: the array geometry is the declared one.
    assert (sizing.rows, sizing.cols) == (cim_t1.fabric.rows, cim_t1.fabric.cols)
