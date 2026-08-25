"""QIF P2.6 — digital op laws for the shared chiplet and the per-macro pool.

Wave B extends P2's law layer with the op kinds P1 admitted: Mamba-2 SSD /
chunked scan, Mamba-1 selective scan, the gated delta rule (DeltaNet /
RWKV-7 class) with RG-LRU as its subset, the short depthwise conv absorbed
into the per-macro pool (ADJ-3), sliding-window attention, and the MLA
pricing seam (D6).

Every law here has at least one HAND-COMPUTED point, written out in the test
comments so a reader can redo the arithmetic without running anything.

Two invariants guard the whole section:
  * PHYSICS — no law may exceed the declared engine's ``lanes * clock``
    ops/s (the DESIGN2 section-8 erratum precedent, applied before the fact).
  * DEGENERACY — the P2.5 pool sizing is bit-identical when no conv is
    declared, and the P1 hybrid/MLA refusals are untouched (this wave prices
    laws; integration un-rejects nothing).
"""

import copy
import math
from pathlib import Path

import pytest
import yaml

import cim_timing
import config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HW_DIR = PROJECT_ROOT / "configs" / "hardware-config"
MODEL_DIR = PROJECT_ROOT / "configs" / "model-config"

FWS_T1 = HW_DIR / "fws_cim_optima_t1.yaml"
VIT_HUGE_64 = MODEL_DIR / "vit_huge_story_64_inf.yaml"
GRANITE_H_TINY = MODEL_DIR / "granite_4_0_h_tiny_inf.yaml"
QWEN35_4B = MODEL_DIR / "qwen3_5_4b_inf.yaml"
LFM2_2P6B = MODEL_DIR / "lfm2_2p6b_inf.yaml"
MINICPM3_4B = MODEL_DIR / "minicpm3_4b_inf.yaml"


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _hw_from_dict(config_dict):
    config_dict = copy.deepcopy(config_dict)
    config.convert(config_dict)
    return config.HWConfig.from_dict(config_dict)


def _device_from_dicts(hw_dict, model_path=VIT_HUGE_64, mode="VIT"):
    hw = _hw_from_dict(hw_dict)
    model = config.parse_config(str(model_path), mode).model_config
    return cim_timing.CimDeviceModel(hw, model)


def _engine_t1(**digital_card):
    """T1 with a shared-digital-chiplet card that DECLARES a vector engine.

    Defaults: 1024 lanes; clock and pipeline depth inherited from cim.fabric
    (0.95 GHz, depth 20) so the inheritance path is the one under test.
    """
    hw_dict = _load_yaml(FWS_T1)
    card = {"kind": "digital_chiplet", "vector_lanes": 1024}
    card.update(digital_card)
    hw_dict["cim"]["cards"] = {
        "ctt": {"kind": "analog_macro", "device": "ctt"},
        "sa": card,
    }
    return _device_from_dicts(hw_dict)


@pytest.fixture(scope="module")
def engine():
    return _engine_t1()


@pytest.fixture(scope="module")
def bare_t1():
    """The shipped T1 pair: no cards block at all, so no vector engine."""
    hw = config.parse_config(str(FWS_T1), "hardware")
    model = config.parse_config(str(VIT_HUGE_64), "VIT").model_config
    return cim_timing.CimDeviceModel(hw, model)


# ---------------------------------------------------------------------------
# Engine capability knobs on the shared digital chiplet card (ADJ-4)
# ---------------------------------------------------------------------------


def test_vector_lanes_have_no_default_and_refuse_by_name(bare_t1):
    # ADJ-4: the systolic array is a matmul engine and softmax_lanes is a
    # softmax pipeline. Neither may stand in for scan lanes.
    #
    # WAVE F (D31): the lane count is no longer DECLARED, it is DERIVED from
    # the pipeline beat. That moves the refusal without weakening it — a call
    # OUTSIDE a mapped run has no beat to derive from, so it still gets a named
    # refusal rather than a borrowed number, and the message now says which of
    # the two is missing.
    assert bare_t1.digital_card.has_vector_engine is False
    assert bare_t1.derived_engine is None
    assert bare_t1.vector_lanes_provenance == "undetermined"
    with pytest.raises(cim_timing.EngineCapabilityError, match="has no vector engine"):
        _ = bare_t1.vector_lanes
    with pytest.raises(cim_timing.EngineCapabilityError, match="D31 sizes the scan/vector engine FROM THE PIPELINE BEAT"):
        _ = bare_t1.vector_lanes
    with pytest.raises(cim_timing.EngineCapabilityError, match="vector_lanes"):
        bare_t1.price_ssm_scan(1, d_inner=8, d_state=4, n_groups=1, n_heads=1)


def test_vector_clock_and_depth_inherit_the_chiplet_and_say_so(engine):
    # Honest defaults: one chiplet declares one clock, and softmax_pipeline_depth
    # is the only elementwise-pipeline depth it declares.
    assert engine.vector_lanes == 1024
    assert engine.vector_clock_hz == pytest.approx(0.95e9)
    assert engine.vector_pipeline_depth == 20
    notes = engine.vector_engine_disclosures()
    assert any("vector_clock_ghz undeclared" in n for n in notes)
    assert any("vector_pipeline_depth undeclared" in n for n in notes)
    assert any("state_bytes_per_cycle undeclared" in n for n in notes)


def test_declared_vector_clock_and_depth_win_and_drop_their_disclosures():
    dev = _engine_t1(vector_clock_ghz=2.0, vector_pipeline_depth=7)
    assert dev.vector_clock_hz == pytest.approx(2.0e9)
    assert dev.vector_pipeline_depth == 7
    notes = dev.vector_engine_disclosures()
    assert not any("vector_clock_ghz undeclared" in n for n in notes)
    assert not any("vector_pipeline_depth undeclared" in n for n in notes)


def test_undeclared_state_bandwidth_reports_traffic_and_bounds_nothing(engine):
    cost = engine.price_ssm_scan(1024, d_inner=512, d_state=64, n_groups=1, n_heads=8)
    assert cost.work.state_bytes > 0          # always REPORTED
    assert cost.state_time_s == 0.0           # and it bounds nothing
    assert cost.time_s == cost.arith_time_s
    assert any("state_bytes_per_cycle undeclared" in n for n in cost.disclosures)


def test_declared_state_bandwidth_can_bind_the_op():
    # A deliberately starved state path: 1 byte/cycle against a 512x64 state.
    dev = _engine_t1(vector_lanes=1 << 20, state_bytes_per_cycle=1.0)
    cost = dev.price_ssm_scan(4, d_inner=512, d_state=64, n_groups=1, n_heads=8)
    assert cost.state_time_s > cost.arith_time_s
    assert cost.time_s == cost.state_time_s


# ---------------------------------------------------------------------------
# 1+2. Mamba-2 SSD / chunked scan and Mamba-1 selective scan
# ---------------------------------------------------------------------------


def test_recurrent_scan_work_hand_computed():
    # HAND COMPUTED. d_inner = 8, d_state = 4, G = 2, H = 4, one token.
    #   state elements = 8 * 4 = 32
    #   state_decay    = 32 muls
    #   input_gate     = H * N = 4 * 4 = 16 muls
    #   outer_write    = 32 muls
    #   state_combine  = 32 adds
    #   readout_mul    = 32 muls
    #   readout_reduce = 32 adds
    #   muls = 32 + 16 + 32 + 32 = 112 ; adds = 32 + 32 = 64 ; ops = 176
    #   state bytes at 2 B/elem = 1 * 2 * 32 * 2 = 128
    work = cim_timing.ssm_recurrent_scan_work(
        1, d_inner=8, d_state=4, n_groups=2, n_heads=4, act_bytes=2.0
    )
    assert work.term("state_decay") == (32.0, 0.0)
    assert work.term("input_gate") == (16.0, 0.0)
    assert work.term("outer_write") == (32.0, 0.0)
    assert work.term("state_combine") == (0.0, 32.0)
    assert work.term("readout_mul") == (32.0, 0.0)
    assert work.term("readout_reduce") == (0.0, 32.0)
    assert work.muls == 112.0
    assert work.adds == 64.0
    assert work.ops == 176.0
    assert work.state_bytes == 128.0
    assert work.validated == cim_timing.LAW_OPTIMA_M3


def test_recurrent_scan_work_is_linear_in_tokens():
    kwargs = dict(d_inner=8, d_state=4, n_groups=2, n_heads=4)
    one = cim_timing.ssm_recurrent_scan_work(1, **kwargs)
    many = cim_timing.ssm_recurrent_scan_work(37, **kwargs)
    assert many.ops == pytest.approx(37 * one.ops)
    assert many.state_bytes == pytest.approx(37 * one.state_bytes)


def test_recurrent_scan_reproduces_the_optima_m3_census_term_for_term():
    """OPTIMA spot point: mamba_giant at mux 16 (compiler_mamba_giant_22nm.md).

    THE OPERATING POINT IS READ OFF THE RECORDED SCENARIO, not assumed. Chain:
    scenarios/compiler_mamba_giant.yaml -> config_refs.digital ->
    configs/digital_tile_config.yaml -> ``clock_GHz: 0.8``. That scenario's
    ONLY override is interconnect bandwidth, so the digital clock really is
    0.8 GHz, not the 1.0 GHz baseline. (An earlier version of this test read
    the point at 1.0 GHz and reported a residual tiling ceil of 520/512; the
    correction removes the residual entirely.)

    Recorded there: D = 2048, expand = 2 -> d_ssm = 4096, H = 64, G = 8,
    N = 128, analog_clock 100 MHz (configs/analog_config.yaml), vec_cycles =
    mux_factor 16 x slice_cycles 2 = 32, digital clock 0.8 GHz.

    HAND COMPUTED from create_s3_scan_collection's own arithmetic:
        effective_hidden_ratio = 1/32,  clock_ratio = 0.8*1000/100 = 8
        scale = 1/256  ->  reuse = 256 microcycles per token
        L = 4096/8 = 512, g_h = 64/8 = 8
        target_TL = round(sqrt(256*512/128)) = round(32.0) = 32
        T_L = closest divisor of 256 to 32 = 32 ; T_N = 256/32 = 8
        L' = ceil(512/32) = 16 ; U = ceil(128/8) = 16
    Its per-group-per-token census is therefore
        decay = outer = combine = L'*U*reuse = 16*16*256 = 65536
        beta  = g_h*U*reuse     = 8*16*256             = 32768

    Ours, per group: 512*128 = 65536 and 8*128 = 1024. The decay/outer/combine
    terms are EQUAL — the TILING CEIL (L'*T_L/L)*(U*T_N/N) is exactly 1.0 at
    the recorded point, so the port is exact with NO residual. The beta term
    differs by exactly T_L = 32 because OPTIMA recomputes dt*B inside every
    channel tile: an artifact of ITS tiling of a SIZED engine, not per-token
    work. We do not charge it, and this test is where that is said out loud.
    """
    d_ssm, H, G, N = 4096, 64, 8, 128
    reuse, T_L, T_N, Lp, U = 256, 32, 8, 16, 16
    L, gh = d_ssm // G, H // G
    optima_decay = Lp * U * reuse
    optima_beta = gh * U * reuse
    assert (optima_decay, optima_beta) == (65536, 32768)

    work = cim_timing.ssm_recurrent_scan_work(
        1, d_inner=d_ssm, d_state=N, n_groups=G, n_heads=H
    )
    ours_decay = work.term("state_decay")[0] / G
    ours_beta = work.term("input_gate")[0] / G
    assert ours_decay == 65536.0
    assert ours_beta == 1024.0

    tiling_ceil = (Lp * T_L / L) * (U * T_N / N)
    assert tiling_ceil == 1.0
    assert optima_decay == pytest.approx(ours_decay * tiling_ceil, rel=1e-12)
    for term in ("outer_write", "state_combine", "readout_mul", "readout_reduce"):
        muls, adds = work.term(term)
        assert (muls + adds) / G == 65536.0
    assert optima_beta == pytest.approx(ours_beta * T_L, rel=1e-12)


def test_mamba1_is_the_unchunked_subset_with_one_gate_term_changed():
    # HAND COMPUTED. d_inner = 8, d_state = 4, one token, no head structure:
    #   the gate is dt_i * x_i per CHANNEL -> 8 muls (not H*N).
    #   muls = 32 + 8 + 32 + 32 = 104 ; adds = 64 ; ops = 168
    work = cim_timing.ssm_recurrent_scan_work(
        1, d_inner=8, d_state=4, variant="mamba1"
    )
    assert work.term("input_gate") == (8.0, 0.0)
    assert work.ops == 168.0
    assert work.validated == cim_timing.LAW_OPTIMA_M3_SUBSET
    # Every other term is identical to the mamba2 law: it IS the same law.
    m2 = cim_timing.ssm_recurrent_scan_work(1, d_inner=8, d_state=4, n_groups=2, n_heads=4)
    for term in ("state_decay", "outer_write", "state_combine", "readout_mul", "readout_reduce"):
        assert work.term(term) == m2.term(term)


def test_mamba1_refuses_a_head_structure_it_does_not_have():
    with pytest.raises(ValueError, match="mamba1 selective scan has no head structure"):
        cim_timing.ssm_recurrent_scan_work(1, d_inner=8, d_state=4, variant="mamba1", n_heads=4)


def test_scan_work_refuses_indivisible_groups_and_heads():
    with pytest.raises(ValueError, match="must be >= 1 and divide"):
        cim_timing.ssm_recurrent_scan_work(1, d_inner=9, d_state=4, n_groups=2, n_heads=2)
    with pytest.raises(ValueError, match="divisible by"):
        cim_timing.ssm_recurrent_scan_work(1, d_inner=8, d_state=4, n_groups=2, n_heads=3)


def test_chunked_ssd_work_hand_computed():
    # HAND COMPUTED. d_inner = 8, d_state = 4, G = 2, H = 4, Q = 2, T = 4.
    #   L = 4, N = 4, g_h = 2, chunks = 2, scale = chunks*G = 4
    #   decay_cumprod     = 4 * 2*2                = 16 muls
    #   intra_scores      = 4 * 2*2*4              = 64 muls / 64 adds
    #   intra_mask        = 4 * 2*2*2              = 32 muls
    #   intra_output      = 4 * 2*2*4              = 64 muls / 64 adds
    #   chunk_state_scale = 4 * 2*4                = 32 muls
    #   chunk_state       = 4 * 2*4*4              = 128 muls / 128 adds
    #   state_passing     = 4 * 4*4                = 64 muls / 64 adds
    #   inter_output      = 4 * 2*4*4              = 128 muls / 128 adds
    #   combine_outputs   = 4 * 2*4                = 32 adds
    #   muls = 16+64+32+64+32+128+64+128 = 528 ; adds = 64+64+128+64+128+32 = 480
    work = cim_timing.ssd_chunked_scan_work(
        4, d_inner=8, d_state=4, n_groups=2, n_heads=4, chunk_size=2, act_bytes=2.0
    )
    assert work.term("decay_cumprod") == (16.0, 0.0)
    assert work.term("intra_scores") == (64.0, 64.0)
    assert work.term("intra_mask") == (32.0, 0.0)
    assert work.term("intra_output") == (64.0, 64.0)
    assert work.term("chunk_state_scale") == (32.0, 0.0)
    assert work.term("chunk_state") == (128.0, 128.0)
    assert work.term("state_passing") == (64.0, 64.0)
    assert work.term("inter_output") == (128.0, 128.0)
    assert work.term("combine_outputs") == (0.0, 32.0)
    assert work.muls == 528.0
    assert work.adds == 480.0
    assert work.ops == 1008.0
    assert work.validated == cim_timing.LAW_UNVALIDATED


def test_chunking_moves_the_state_once_per_chunk_not_once_per_token():
    # The point of SSD: state traffic falls by exactly the chunk size.
    kwargs = dict(d_inner=8, d_state=4, n_groups=2, n_heads=4, act_bytes=2.0)
    recurrent = cim_timing.ssm_recurrent_scan_work(4, **kwargs)
    chunked = cim_timing.ssd_chunked_scan_work(4, chunk_size=2, **kwargs)
    assert recurrent.state_bytes == 512.0
    assert chunked.state_bytes == 256.0
    assert recurrent.state_bytes == 2 * chunked.state_bytes


def test_chunking_trades_arithmetic_for_state_traffic():
    # Chunking is not a free win, and the law must not read like one: the
    # intra-chunk terms are QUADRATIC in Q, so a bigger chunk buys state
    # traffic with arithmetic. Granite-4.0-H-Tiny dims, 2048 prefill tokens.
    kwargs = dict(d_inner=3072, d_state=128, n_groups=1, n_heads=48)
    recurrent = cim_timing.ssm_recurrent_scan_work(2048, **kwargs)
    small = cim_timing.ssd_chunked_scan_work(2048, chunk_size=32, **kwargs)
    big = cim_timing.ssd_chunked_scan_work(2048, chunk_size=512, **kwargs)
    assert big.state_bytes < small.state_bytes < recurrent.state_bytes
    assert small.ops < recurrent.ops        # the chunk pays off at Q = 32
    assert big.ops > recurrent.ops          # and stops paying off well before 512


def test_chunked_ssd_charges_the_tail_chunk_full_and_says_so():
    kwargs = dict(d_inner=8, d_state=4, n_groups=2, n_heads=4, chunk_size=4)
    five = cim_timing.ssd_chunked_scan_work(5, **kwargs)
    eight = cim_timing.ssd_chunked_scan_work(8, **kwargs)
    assert dict(five.detail)["chunks"] == 2.0
    assert dict(five.detail)["tokens_charged"] == 8.0
    assert five.ops == eight.ops           # the tail chunk is charged full
    assert "TAIL CHUNK IS CHARGED" in cim_timing.ssd_chunked_scan_work.__doc__


def test_chunked_law_refuses_an_unchunked_call_by_name():
    with pytest.raises(ValueError, match="is not chunked; call"):
        cim_timing.ssd_chunked_scan_work(
            4, d_inner=8, d_state=4, n_groups=2, n_heads=4, chunk_size=1
        )


def test_scan_dispatch_uses_the_recurrence_for_decode_and_mamba1(engine):
    common = dict(d_inner=8, d_state=4, n_groups=2, n_heads=4, chunk_size=256)
    # A decode step is one token: chunking cannot apply, so the recurrence wins.
    assert engine.ssm_scan_work(1, **common).law == "ssm_recurrent_scan[mamba2]"
    assert engine.ssm_scan_work(512, **common).law == "ssd_chunked_scan"
    assert engine.ssm_scan_work(512, **{**common, "chunk_size": 1}).law == (
        "ssm_recurrent_scan[mamba2]"
    )
    m1 = engine.ssm_scan_work(512, d_inner=8, d_state=4, variant="mamba1", chunk_size=256)
    assert m1.law == "ssm_recurrent_scan[mamba1]"


def test_price_ssm_block_reads_a_parsed_ssm_config(engine):
    # Granite-4.0-H-Tiny (ADJ-1 headline): d = 1536, expand 2 -> d_inner 3072,
    # d_state 128, n_groups 1, n_heads 48, chunk 256.
    model = config.parse_config(str(GRANITE_H_TINY), "LLM").model_config
    ssm = model.ssm
    assert (ssm.variant, ssm.d_state, ssm.n_groups, ssm.n_heads, ssm.chunk_size) == (
        "mamba2", 128, 1, 48, 256,
    )
    assert ssm.resolve_d_inner(model.hidden_dim) == 3072
    chunked = engine.price_ssm_block(ssm, model.hidden_dim, 2048)
    assert chunked.law == "ssd_chunked_scan"
    decode = engine.price_ssm_block(ssm, model.hidden_dim, 1)
    assert decode.law == "ssm_recurrent_scan[mamba2]"
    # HAND COMPUTED decode work: state = 3072*128 = 393216 elements.
    #   muls = 3*393216 + 48*128 = 1179648 + 6144 = 1185792
    #   adds = 2*393216 = 786432 ; ops = 1972224
    assert decode.work.muls == 1185792.0
    assert decode.work.adds == 786432.0
    assert decode.work.ops == 1972224.0
    # Timed on the declared engine: 1024 lanes @ 0.95 GHz, depth 20.
    #   cycles = 20 + ceil(1972224/1024) - 1 = 20 + 1926 - 1 = 1945
    assert decode.arith_cycles == 1945
    assert decode.time_s == pytest.approx(1945 / 0.95e9, rel=1e-12)


# ---------------------------------------------------------------------------
# 3. Delta rule (gated DeltaNet / RWKV-7) and RG-LRU
# ---------------------------------------------------------------------------


def test_delta_rule_recurrent_work_hand_computed():
    # HAND COMPUTED. 2 key heads x d_k 4, 4 value heads x d_v 3, one token.
    #   value_dim = 4*3 = 12 ; d_v effective = 12/2 = 6
    #   state = 2 * 4 * 6 = 48 elements ( = key_head_dim * value_dim = 4*12 )
    #   per = 48 ; heads*d_v = 12
    #   read_old_value 48/48 ; delta_correct 12/12 ; state_decay 48/0
    #   rank1_write 48/48 ; output_read 48/48 ; output_gate 12/0
    #   muls = 48+12+48+48+48+12 = 216 ; adds = 48+12+48+48 = 156 ; ops = 372
    work = cim_timing.delta_rule_work(
        1, num_key_heads=2, key_head_dim=4, num_value_heads=4, value_head_dim=3
    )
    assert dict(work.detail)["state_elements"] == 48.0
    assert dict(work.detail)["d_v_effective"] == 6.0
    assert work.term("read_old_value") == (48.0, 48.0)
    assert work.term("delta_correct") == (12.0, 12.0)
    assert work.term("state_decay") == (48.0, 0.0)
    assert work.term("rank1_write") == (48.0, 48.0)
    assert work.term("output_read") == (48.0, 48.0)
    assert work.term("output_gate") == (12.0, 0.0)
    assert work.muls == 216.0
    assert work.adds == 156.0
    assert work.state_bytes == 96.0
    assert work.validated == cim_timing.LAW_UNVALIDATED


def test_delta_rule_gates_are_real_terms_that_can_be_switched_off():
    base = dict(num_key_heads=2, key_head_dim=4, num_value_heads=4, value_head_dim=3)
    gated = cim_timing.delta_rule_work(1, **base)
    no_out = cim_timing.delta_rule_work(1, output_gate=False, **base)
    no_decay = cim_timing.delta_rule_work(1, decay_gate=False, **base)
    assert gated.ops - no_out.ops == 12.0     # heads * d_v
    assert gated.ops - no_decay.ops == 48.0   # heads * d_k * d_v
    with pytest.raises(KeyError):
        no_out.term("output_gate")


def test_delta_rule_chunked_work_hand_computed():
    # HAND COMPUTED. Same dims, Q = 4, T = 4 -> chunks = 1, scale = heads = 2.
    #   tri = 4*3/2 = 6 ; tri_incl = 4*5/2 = 10 ; ut = 4*3*2/6 = 4 ; kv = 24
    #   decay_cumprod  2*4*4        = 32
    #   gate_scale     2*2*4*4      = 64
    #   kk_scores      2*6*4        = 48 / 48
    #   ut_transform   2*4          = 8 / 8
    #   w_pseudo_value 2*10*6       = 120 / 120
    #   u_pseudo_key   2*10*4       = 80 / 80
    #   state_read     2*4*24       = 192 / 192
    #   state_write    2*5*24       = 240 / 240
    #   output_inter   2*4*24       = 192 / 192
    #   output_intra   2*6*(4+6)    = 120 / 120
    #   output_gate    2*4*6        = 48
    #   muls = 32+64+48+8+120+80+192+240+192+120+48 = 1144
    #   adds = 48+8+120+80+192+240+192+120          = 1000
    work = cim_timing.delta_rule_work(
        4, num_key_heads=2, key_head_dim=4, num_value_heads=4, value_head_dim=3, chunk_size=4
    )
    assert work.term("ut_transform") == (8.0, 8.0)
    assert work.term("state_write") == (240.0, 240.0)
    assert work.term("output_intra") == (120.0, 120.0)
    assert work.muls == 1144.0
    assert work.adds == 1000.0
    assert work.ops == 2144.0
    assert work.law == "delta_rule_chunked"
    assert work.validated == cim_timing.LAW_UNVALIDATED
    # The state moves once per chunk, exactly as in SSD.
    assert work.state_bytes == 96.0
    recurrent = cim_timing.delta_rule_work(
        4, num_key_heads=2, key_head_dim=4, num_value_heads=4, value_head_dim=3
    )
    assert recurrent.state_bytes == 4 * work.state_bytes


def test_delta_rule_law_declares_itself_unvalidated_in_its_docstring():
    # D21 honesty: no numeric reference exists for this law anywhere, and the
    # law says so where a reader will find it.
    doc = cim_timing.delta_rule_work.__doc__
    assert "UNVALIDATED" in doc
    assert "NO NUMERIC REFERENCE EXISTS" in doc
    assert "UNVALIDATED" in cim_timing.rg_lru_work.__doc__


def test_delta_rule_refuses_key_heads_that_do_not_divide_value_heads():
    with pytest.raises(ValueError, match="must divide num_value_heads"):
        cim_timing.delta_rule_work(
            1, num_key_heads=3, key_head_dim=4, num_value_heads=4, value_head_dim=3
        )


def test_price_linear_attention_block_reads_a_parsed_config(engine):
    # Qwen3.5-4B: 16 key heads x 128, 32 value heads x 128, gated both ways.
    model = config.parse_config(str(QWEN35_4B), "LLM").model_config
    la = model.linear_attention
    assert (la.num_key_heads, la.key_head_dim, la.num_value_heads, la.value_head_dim) == (
        16, 128, 32, 128,
    )
    cost = engine.price_linear_attention_block(la, 1)
    # HAND COMPUTED. value_dim = 32*128 = 4096 ; d_v eff = 4096/16 = 256
    #   state = 16 * 128 * 256 = 524288 ( = key_head_dim * value_dim = 128*4096 )
    #   heads*d_v = 16*256 = 4096
    #   muls = 4*524288 + 4096 + 4096 = 2105344 ; adds = 3*524288 + 4096 = 1576960
    assert dict(cost.work.detail)["state_elements"] == 524288.0
    assert cost.work.muls == 2105344.0
    assert cost.work.adds == 1576960.0


def test_rg_lru_is_a_strict_subset_of_the_delta_rule():
    # HAND COMPUTED. width = 8, one token: 2+1+1+2 = 6 muls and 1 add per
    # channel -> muls 48, adds 8, ops 56.
    rg = cim_timing.rg_lru_work(1, width=8)
    assert rg.muls == 48.0
    assert rg.adds == 8.0
    assert rg.ops == 56.0
    # The same state size expressed as a delta rule with a diagonal state
    # (d_k = 1) charges 6*8 muls + 4*8 adds = 80 ops: strictly more, because
    # RG-LRU has no outer-product write and no read-and-remove step.
    delta = cim_timing.delta_rule_work(
        1, num_key_heads=8, key_head_dim=1, num_value_heads=8, value_head_dim=1
    )
    assert dict(delta.detail)["state_elements"] == dict(rg.detail)["state_elements"]
    assert delta.ops == 80.0
    assert rg.ops < delta.ops


# ---------------------------------------------------------------------------
# The physics invariant: no law outruns the silicon its card declares
# ---------------------------------------------------------------------------


PEAK_CASES = [
    ("ssd_chunk", dict(law="ssm", tokens=2048, d_inner=3072, d_state=128, n_groups=1,
                       n_heads=48, chunk_size=256)),
    ("ssd_decode", dict(law="ssm", tokens=1, d_inner=3072, d_state=128, n_groups=1,
                        n_heads=48, chunk_size=256)),
    ("mamba1", dict(law="ssm", tokens=777, d_inner=4096, d_state=16, variant="mamba1")),
    ("delta_chunk", dict(law="delta", tokens=2048, num_key_heads=16, key_head_dim=128,
                         num_value_heads=32, value_head_dim=128, chunk_size=64)),
    ("delta_decode", dict(law="delta", tokens=1, num_key_heads=16, key_head_dim=128,
                          num_value_heads=32, value_head_dim=128)),
    ("rg_lru", dict(law="rg_lru", tokens=1024, width=2560)),
]


@pytest.mark.parametrize("name,case", PEAK_CASES, ids=[c[0] for c in PEAK_CASES])
@pytest.mark.parametrize("lanes", [1, 7, 1024, 65536])
def test_no_law_exceeds_the_declared_arithmetic_peak(name, case, lanes):
    """DESIGN2 section-8 erratum precedent, applied before the fact.

    One lane retires one scalar operation per vector cycle, so the priced
    op/s of EVERY law must be <= lanes * clock. This is the only hard
    guarantee the unvalidated laws carry, so it is checked over every law,
    a decode and a prefill shape, and four lane counts.
    """
    dev = _engine_t1(vector_lanes=lanes)
    case = dict(case)
    kind = case.pop("law")
    tokens = case.pop("tokens")
    if kind == "ssm":
        cost = dev.price_ssm_scan(tokens, **case)
    elif kind == "delta":
        cost = dev.price_delta_rule(tokens, **case)
    else:
        cost = dev.price_rg_lru(tokens, **case)
    peak = lanes * dev.vector_clock_hz
    assert cost.ops_per_s <= peak * (1 + 1e-12)
    assert cost.arith_cycles >= math.ceil(cost.work.ops / lanes)
    assert cost.time_s >= cost.work.ops / peak


# ---------------------------------------------------------------------------
# 4. Short depthwise conv absorbed into the per-macro pool (ADJ-3)
# ---------------------------------------------------------------------------


def test_short_conv_ops_per_result_is_k_muls_plus_k_minus_one_adds():
    assert cim_timing.short_conv_ops_per_result(1) == 1
    assert cim_timing.short_conv_ops_per_result(3) == 5    # LFM2 L = 3
    assert cim_timing.short_conv_ops_per_result(4) == 7    # Mamba2 / Qwen3.5 k = 4
    with pytest.raises(ValueError, match="kernel_size must be >= 1"):
        cim_timing.short_conv_ops_per_result(0)


def test_pool_sizing_without_a_conv_is_bit_identical_to_p2_5(bare_t1):
    # DEGENERACY: the extension must move no P2.5 number.
    sizing = bare_t1.digital_pool_sizing()
    assert (sizing.conv_kernel, sizing.conv_channels, sizing.conv_lanes) == (0, 0, 0)
    assert sizing.conv_ops_per_s == 0.0
    assert sizing.conv_area_mm2 == 0.0
    assert sizing.total_lanes == sizing.lanes
    assert sizing.area_mm2 == sizing.adders * float(bare_t1.card.pool_area_mm2_per_adder)
    line = bare_t1.report_digital_pool(sizing)
    assert "Absorbed short depthwise conv" not in line


def test_short_conv_pool_sizing_hand_computed(bare_t1):
    # HAND COMPUTED on T1: cols_adc 320, adc_mux 4, slice_cycles 2,
    # analog clock 100 MHz, pool clock inherits the fabric's 0.95 GHz.
    #   result rate    = 320 * 100e6 / 2 = 1.6e10 results/s
    #   shift-add lanes= ceil(1.6e10 / 0.95e9) = ceil(16.84) = 17
    #   stored columns = 320 * 4 = 1280
    #   k = 4 over 1280 channels -> duty = 1.0, ops/result = 7
    #   conv ops/s     = 1.6e10 * 1.0 * 7 = 1.12e11
    #   conv lanes     = ceil(1.12e11 / 0.95e9) = ceil(117.89) = 118
    sizing = bare_t1.digital_pool_sizing(conv_kernel=4, conv_channels=1280)
    assert bare_t1.macro_result_rate_per_s() == pytest.approx(1.6e10)
    assert sizing.lanes == 17
    assert sizing.conv_ops_per_result == 7
    assert sizing.conv_column_duty == pytest.approx(1.0)
    assert sizing.conv_ops_per_s == pytest.approx(1.12e11)
    assert sizing.conv_lanes == 118
    assert sizing.total_lanes == 135
    line = bare_t1.report_digital_pool(sizing)
    assert "Absorbed short depthwise conv (ADJ-3): k = 4 over 1280 channels" in line
    assert "118 conv lanes" in line


def test_short_conv_duty_scales_with_the_macro_columns_it_actually_serves(bare_t1):
    # HAND COMPUTED. 640 of 1280 stored columns are conv channels -> duty 0.5,
    # conv ops/s = 1.6e10 * 0.5 * 7 = 5.6e10, lanes = ceil(58.947) = 59.
    half = bare_t1.digital_pool_sizing(conv_kernel=4, conv_channels=640)
    assert half.conv_column_duty == pytest.approx(0.5)
    assert half.conv_lanes == 59
    # A block wider than one macro saturates at the whole-macro worst case.
    wide = bare_t1.digital_pool_sizing(conv_kernel=4, conv_channels=99999)
    full = bare_t1.digital_pool_sizing(conv_kernel=4, conv_channels=1280)
    assert wide.conv_column_duty == 1.0
    assert wide.conv_lanes == full.conv_lanes == 118


def test_short_conv_area_joins_the_one_pool_area_accounting():
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["cim"]["cards"] = {
        "ctt": {"kind": "analog_macro", "device": "ctt", "pool_area_mm2_per_adder": 1e-4},
    }
    dev = _device_from_dicts(hw_dict)
    sizing = dev.digital_pool_sizing(conv_kernel=4, conv_channels=1280)
    # One accounting per metric (D21): area_mm2 is adders + conv lanes, and
    # conv_area_mm2 is the named component, never a second total.
    assert sizing.conv_area_mm2 == pytest.approx(118 * 1e-4)
    assert sizing.area_mm2 == pytest.approx((sizing.adders + sizing.conv_lanes) * 1e-4)


def test_conv_kernel_without_channels_is_refused_not_invented(bare_t1):
    with pytest.raises(ValueError, match="not a number this model may invent"):
        bare_t1.digital_pool_sizing(conv_kernel=4)


def test_pool_sizing_absorbs_a_parsed_short_conv_block(bare_t1):
    # LFM2-2.6B: kernel 3 over conv_dim channels.
    model = config.parse_config(str(LFM2_2P6B), "LLM").model_config
    short_conv = model.short_conv
    assert short_conv.kernel_size == 3
    sizing = bare_t1.digital_pool_sizing(
        conv_kernel=short_conv.kernel_size, conv_channels=short_conv.conv_dim
    )
    assert sizing.conv_ops_per_result == 5
    assert sizing.conv_lanes > 0


# ---------------------------------------------------------------------------
# 5. Sliding-window attention: the folded SA law at n = window
# ---------------------------------------------------------------------------


def test_window_context_is_the_whole_new_content(bare_t1):
    assert bare_t1.window_context(4096, 512) == 512
    assert bare_t1.window_context(256, 512) == 256      # context shorter than the window
    assert bare_t1.window_context(4096, None) == 4096   # global layer
    assert bare_t1.window_context(4096, 0) == 4096      # inert window


def test_sliding_window_decode_is_the_existing_law_at_the_capped_context(bare_t1):
    swa = bare_t1.sliding_window_decode_timing(4096, 512)
    direct = bare_t1.decode_attention_timing(512)
    assert swa == direct                         # a THIN wrapper, not a new law
    # A window at or above the context changes nothing at all.
    assert bare_t1.sliding_window_decode_timing(256, 512) == bare_t1.decode_attention_timing(256)
    assert bare_t1.sliding_window_decode_timing(4096, None) == bare_t1.decode_attention_timing(4096)


def test_sliding_window_is_strictly_cheaper_than_the_full_context(bare_t1):
    swa = bare_t1.sliding_window_decode_timing(4096, 512)
    full = bare_t1.decode_attention_timing(4096)
    assert swa.stage_time_s < full.stage_time_s
    assert swa.total_cycles < full.total_cycles


def test_sliding_window_prefill_caps_n_and_discloses_the_rectangle_relaxation(bare_t1):
    swa = bare_t1.sliding_window_prefill_timing(seq_len=2048, window=256)
    direct = bare_t1.attention_call_timing(
        m=2048 * bare_t1.params.shared_heads, k=bare_t1.params.head_dim, n=256
    )
    assert swa == direct
    assert swa.total_cycles < bare_t1.prefill_attention_timing(seq_len=2048).total_cycles
    assert "RELAXATION, disclosed" in (
        cim_timing.CimDeviceModel.sliding_window_prefill_timing.__doc__
    )


# ---------------------------------------------------------------------------
# 6. MLA pricing seam and the KV-replication area consequence (D6)
# ---------------------------------------------------------------------------


def test_mla_is_priced_through_the_sa_law_at_the_latent_call_dims(bare_t1):
    # MiniCPM3-4B: 40 heads, kv_lora_rank 256, qk_rope_head_dim 32.
    # The score call contracts over 256 + 32 = 288 against the context.
    timing = bare_t1.mla_attention_timing(
        1024, num_heads=40, kv_lora_rank=256, qk_rope_head_dim=32
    )
    direct = bare_t1.attention_call_timing(
        m=40, k=288, n=1024, kv_heads=1, softmax_tokens=40, streams=1
    )
    assert timing == direct                      # one attention law, not two
    assert timing.heads_chip == 1


def test_mla_tp_hostility_is_a_number_not_a_sentence(bare_t1):
    # D6: the KV latent is ONE shared rank, so heads_chip stays 1 at every tp
    # and tp buys no attention-side reduction whatsoever.
    dims = dict(num_heads=40, kv_lora_rank=256, qk_rope_head_dim=32)
    tp1 = bare_t1.mla_attention_timing(1024, tp=1, **dims)
    tp4 = bare_t1.mla_attention_timing(1024, tp=4, **dims)
    assert tp1.heads_chip == tp4.heads_chip == 1
    assert tp1.total_cycles == tp4.total_cycles
    assert tp1.stage_time_s == tp4.stage_time_s
    # A GQA model of the same shape DOES shrink with tp — the contrast is the
    # finding, so it is asserted rather than asserted-about.
    gqa1 = bare_t1.decode_attention_timing(1024, kv_heads=8, tp=1)
    gqa4 = bare_t1.decode_attention_timing(1024, kv_heads=8, tp=4)
    assert gqa4.heads_chip < gqa1.heads_chip


def test_mla_replication_area_hand_computed(bare_t1):
    # HAND COMPUTED on MiniCPM3-4B dims against the T1 card (rows 1280,
    # stored cols = cols_adc*adc_mux = 1280, area 1.403065 mm2 per array):
    #   W_DKV 2560 x 288  -> ceil(2560/1280)*ceil(288/1280)  = 2*1 = 2 arrays
    #   W_DQ  2560 x 768  -> 2*1 = 2 arrays
    #   W_UK   256 x 2560 -> 1*2 = 2 arrays
    #   W_UV   256 x 2560 -> 1*2 = 2 arrays                    total 8 arrays
    #   latent bytes @2B = (2560*288 + 2560*768)*2 = 5,406,720
    #   up-proj bytes @2B = (256*2560 + 256*2560)*2 = 2,621,440
    #   area per shard = 8 * 1.403065 = 11.22452 mm2
    #   tp = 4 pays (tp-1) further copies: 24 arrays, 33.67356 mm2
    rep = bare_t1.mla_kv_replication(
        hidden_dim=2560,
        num_heads=40,
        kv_lora_rank=256,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
        q_lora_rank=768,
        tp=4,
        weight_bytes=2.0,
    )
    assert [name for name, _, _, _ in rep.matrices] == ["W_DKV", "W_DQ", "W_UK", "W_UV"]
    assert [arrays for _, _, _, arrays in rep.matrices] == [2, 2, 2, 2]
    assert rep.latent_bytes_per_shard == 5406720.0
    assert rep.up_projection_bytes_per_shard == 2621440.0
    assert rep.replicated_bytes_per_shard == 8028160.0
    assert rep.replicated_arrays_per_shard == 8
    assert rep.replicated_area_mm2_per_shard == pytest.approx(8 * 1.403065)
    assert rep.extra_replicated_arrays == 24
    assert rep.extra_replicated_bytes == 3 * 8028160.0
    assert rep.extra_replicated_area_mm2 == pytest.approx(24 * 1.403065)


def test_mla_replication_costs_nothing_extra_at_tp_1(bare_t1):
    rep = bare_t1.mla_kv_replication(
        hidden_dim=2560, num_heads=40, kv_lora_rank=256, qk_nope_head_dim=64,
        qk_rope_head_dim=32, v_head_dim=64, q_lora_rank=768, tp=1, weight_bytes=2.0,
    )
    assert rep.extra_replicated_bytes == 0.0
    assert rep.extra_replicated_area_mm2 == 0.0
    assert rep.replicated_arrays_per_shard == 8


def test_mla_seams_read_the_parsed_minicpm3_config(bare_t1):
    model = config.parse_config(str(MINICPM3_4B), "LLM").model_config
    attention = model.attention
    assert attention.attention_type == "mla"
    rep = bare_t1.mla_kv_replication_from_config(
        attention, model.hidden_dim, tp=2, layers=model.num_layers, weight_bytes=2.0
    )
    # 62 layers of the per-layer figure computed above.
    assert rep.layers == 62
    assert rep.replicated_bytes_per_shard == 62 * 8028160.0
    assert rep.replicated_arrays_per_shard == 62 * 8
    timing = bare_t1.mla_attention_timing_from_config(attention, 1024, batch_size=2, tp=2)
    assert timing.heads_chip == 1
    line = bare_t1.report_mla_replication(rep)
    assert "MLA KV replication (D6)" in line
    assert "W_DKV, W_DQ, W_UK, W_UV" in line
    assert "Area PAID to replication" in line


def test_mla_seams_refuse_a_non_mla_attention_block(bare_t1):
    model = config.parse_config(str(VIT_HUGE_64), "VIT").model_config
    with pytest.raises(ValueError, match="attention_type must be 'mla'"):
        bare_t1.mla_kv_replication_from_config(model.attention, model.hidden_dim)


# ---------------------------------------------------------------------------
# This wave prices laws; it un-rejects nothing (P1.4 seams stay closed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_path", [GRANITE_H_TINY, QWEN35_4B, LFM2_2P6B, MINICPM3_4B]
)
def test_hybrid_and_mla_rows_are_still_refused_on_fws_cim(model_path):
    hw = config.parse_config(str(FWS_T1), "hardware")
    model_config = config.parse_config(str(model_path), "LLM")
    with pytest.raises(ValueError):
        config.validate_model_config(model_config, hw)


# ---------------------------------------------------------------------------
# Card schema: the engine knobs parse, refuse, and stay inert when absent
# ---------------------------------------------------------------------------


def test_engine_capability_knobs_parse_off_the_card():
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["cim"]["cards"] = {
        "sa": {
            "kind": "digital_chiplet",
            "vector_lanes": 512,
            "vector_clock_ghz": 1.5,
            "vector_pipeline_depth": 6,
            "state_bytes_per_cycle": 64.0,
        }
    }
    card = _hw_from_dict(hw_dict).cim_config.cards.digital_card
    assert card.vector_lanes == 512
    assert card.has_vector_engine is True
    assert card.vector_clock_ghz_effective == pytest.approx(1.5)
    assert card.vector_pipeline_depth_effective == 6
    assert card.state_bytes_per_cycle == pytest.approx(64.0)


def test_engine_knobs_are_inert_on_a_synthesized_card(bare_t1):
    # A YAML with no cim.cards block declares no engine, and every fabric law
    # that existed before this wave is untouched by that.
    card = bare_t1.digital_card
    assert card.vector_lanes == 0
    assert card.has_vector_engine is False
    assert card.vector_clock_ghz_effective == pytest.approx(bare_t1.fabric.clock_ghz)
    assert card.vector_pipeline_depth_effective == bare_t1.fabric.softmax_pipeline_depth
    assert card.state_bytes_per_cycle == 0.0
    # The attention laws still work: only the vector engine is missing.
    assert bare_t1.decode_attention_timing(64).total_cycles > 0


def test_negative_vector_lanes_are_refused_at_parse():
    hw_dict = _load_yaml(FWS_T1)
    hw_dict["cim"]["cards"] = {"sa": {"kind": "digital_chiplet", "vector_lanes": -4}}
    with pytest.raises(ValueError, match="vector_lanes"):
        _hw_from_dict(hw_dict)


def test_every_priced_op_carries_its_provenance_label(engine):
    # D21: a report can never present an unvalidated law as a checked number,
    # because the label travels with the cost.
    ssd = engine.price_ssm_scan(1, d_inner=8, d_state=4, n_groups=2, n_heads=4)
    chunked = engine.price_ssm_scan(
        512, d_inner=8, d_state=4, n_groups=2, n_heads=4, chunk_size=64
    )
    m1 = engine.price_ssm_scan(1, d_inner=8, d_state=4, variant="mamba1")
    delta = engine.price_delta_rule(
        1, num_key_heads=2, key_head_dim=4, num_value_heads=4, value_head_dim=3
    )
    rg = engine.price_rg_lru(1, width=8)
    assert ssd.validated == cim_timing.LAW_OPTIMA_M3
    assert m1.validated == cim_timing.LAW_OPTIMA_M3_SUBSET
    assert chunked.validated == delta.validated == rg.validated == cim_timing.LAW_UNVALIDATED
    for cost in (ssd, chunked, m1, delta, rg):
        assert cost.validated == cost.work.validated
        assert cost.disclosures


def test_zero_token_calls_charge_no_arithmetic(engine):
    cost = engine.price_ssm_scan(0, d_inner=8, d_state=4, n_groups=2, n_heads=4)
    assert cost.work.ops == 0.0
    assert cost.ops_per_s == 0.0


def test_pool_sizing_only_absorbs_the_work_a_caller_declares(bare_t1):
    """A named LIMIT, not a claim of completeness (P1 appendix, D12).

    The P1 op appendix asks P2 to confirm the per-macro pool still never binds
    once SSM blocks raise the elementwise:GEMM ratio. The honest answer is
    structural, and this test pins both halves of it:

    * the sizing law grows lanes from OPS-PER-RESULT demand, so any demand a
      caller declares is absorbed by construction — the short depthwise conv
      (ADJ-3) is the first such demand and it works;
    * a demand NOBODY declares is sized as one lane-op per emitted result.
      SSM norms and gates (RMSNorm, per-head GroupNorm, SiLU, sigmoid) are
      more than that, and no defensible ops-per-result figure for them ships
      here, so P2.6 does NOT claim the pool is proven non-binding for them.

    Inventing that figure to close the gap is exactly the failure ADJ-4
    forbids, so the gap is named instead.
    """
    plain = bare_t1.digital_pool_sizing()
    with_conv = bare_t1.digital_pool_sizing(conv_kernel=4, conv_channels=1280)
    # Declared demand is absorbed: lanes grow with ops per result.
    assert with_conv.total_lanes > plain.total_lanes
    assert with_conv.conv_ops_per_s == pytest.approx(
        plain.result_rate_per_s * with_conv.conv_ops_per_result
    )
    # Undeclared demand is not: the base sizing is one lane-op per result.
    assert plain.total_lanes == plain.lanes
    assert plain.lanes == math.ceil(plain.result_rate_per_s / plain.pool_clock_hz)
