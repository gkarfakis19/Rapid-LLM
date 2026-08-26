"""QIF P1.5: Granite-4.0-H-Tiny runs end to end on the mapped fws_cim path.

ADJ-1 made Granite-4.0-H-Tiny the headline model. P1.4 refused it — nothing
priced a Mamba-2 recurrence — and P2.6, P3 and P4 then built the laws, the
placement and the pricing. This module is the proof that the three met: 36 SSD
layers and 4 attention layers, MoE on every one of them, priced op by op.

The narrowing is the risky part, so most of these tests are about what stays
REFUSED. A rejection seam that opens one notch too wide does not crash; it
prints a number that silently assumes a law nobody wrote.

The physics audits are here too, in the DESIGN2 §8 style: a law that exceeds a
declared peak is wrong even when nothing fails, and an analog op whose duration
does not match the M-law by hand is wrong even when the report is self-
consistent.
"""

from __future__ import annotations

import copy
import json
import math
import os

import pytest
import yaml

import cim_timing
import config
import fws_atlas_export
import fws_eval
import fws_mapping
from program.fws_build import build_fws_program

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HW_DIR = os.path.join(PROJECT_ROOT, "configs", "hardware-config")
MODEL_DIR = os.path.join(PROJECT_ROOT, "configs", "model-config")

GRANITE_HW = os.path.join(HW_DIR, "fws_cim_granite_tiny.yaml")
GRANITE_MODEL = os.path.join(MODEL_DIR, "granite_4_0_h_tiny_inf.yaml")
QWEN_MODEL = os.path.join(MODEL_DIR, "qwen3_5_4b_inf.yaml")
FWS_LLAMA7B = os.path.join(HW_DIR, "fws_cim_llama7b.yaml")
A100 = os.path.join(HW_DIR, "a100_80GB.yaml")


def _hw(path, mutate=None):
    with open(path) as handle:
        raw = copy.deepcopy(yaml.safe_load(handle))
    if mutate is not None:
        mutate(raw)
    config.convert(raw)
    return config.HWConfig.from_dict(raw)


@pytest.fixture(scope="module")
def granite():
    mapping = fws_mapping.build_mapping(
        _hw(GRANITE_HW), config.parse_config(GRANITE_MODEL, "LLM")
    )
    return fws_eval.evaluate_fws(build_fws_program(mapping))


# ---------------------------------------------------------------------------
# The narrowing: what opened, and everything that did not
# ---------------------------------------------------------------------------


def test_granite_validates_on_the_mapped_fws_cim_path():
    hw = _hw(GRANITE_HW)
    assert hw.mapping_config is not None  # the mapping block IS the narrowing
    config.validate_configs(hw, config.parse_config(GRANITE_MODEL, "LLM"))


def test_the_same_granite_config_is_still_refused_without_a_mapping_block():
    """The closed form has no stage for a recurrence and must keep saying so."""
    hw = _hw(GRANITE_HW, lambda raw: raw.pop("mapping"))
    with pytest.raises(ValueError) as excinfo:
        config.validate_configs(hw, config.parse_config(GRANITE_MODEL, "LLM"))
    message = str(excinfo.value)
    assert "ssm" in message
    assert "mapping:" in message  # the message names the way forward


def test_the_gpu_path_rejection_is_untouched():
    """P1.4's GPU refusal is not scoped to fws_cim and must not move."""
    hw = config.parse_config(A100, "hardware")
    with pytest.raises(ValueError) as excinfo:
        config.validate_configs(hw, config.parse_config(GRANITE_MODEL, "LLM"))
    assert "ssm" in str(excinfo.value)


@pytest.mark.parametrize(
    "field, override",
    [
        # attention.output_gate is NO LONGER on this list: C1 places W_g as an
        # analog tile and the gate multiply as pool work, so the mapped path
        # prices it. tests/test_qif_qwen_e2e.py holds it end to end, and it
        # stays refused on the GPU and unmapped paths there.
        (
            "shared_weight_groups",
            {"shared_weight_groups": [{"name": "depth_shared", "layers": [0, 1]}]},
        ),
        ("ffn_dims", {"ffn_dims": {"shared_expert": 1024, "default": 999}}),
    ],
)
def test_inputs_the_mapped_path_still_cannot_price_stay_refused(field, override):
    """The narrowing lifted a NAMED list. Everything else keeps its name.

    ``ffn_dims`` is the sharp one: ``shared_expert`` is priced now (it reaches
    the array census and the mapping's stage shapes), and any OTHER key still
    only moves llm_util's parameter count — so the refusal must survive on the
    key, not on the block.
    """
    with open(GRANITE_MODEL) as handle:
        raw = copy.deepcopy(yaml.safe_load(handle))
    model_param = raw["model_param"]
    for key, value in override.items():
        if key == "attention":
            model_param["attention"] = {**model_param["attention"], **value}
        else:
            model_param[key] = value
    # Parse through the normal file path so the schema validation is the real one.
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(raw, handle)
        path = handle.name
    try:
        model = config.parse_config(path, "LLM")
        with pytest.raises(ValueError) as excinfo:
            config.validate_configs(_hw(GRANITE_HW), model)
        message = str(excinfo.value)
        assert field.split(".")[-1] in message
    finally:
        os.unlink(path)


def test_a_block_kind_nobody_prices_would_still_be_refused_on_the_mapped_path():
    """The mapped narrowing is a NAMED list, not `hybrids are fine now`."""
    priced = set(config._FWS_CIM_MAPPED_BLOCK_KINDS)
    assert priced == {"attention", "ssm", "short_conv", "linear_attn"}
    # Every kind on the list must be one the P1 schema can express, or the
    # list is naming something that cannot arrive.
    assert priced <= set(config._BLOCK_KINDS)


def test_an_ssm_run_without_a_declared_vector_engine_derives_one_from_the_beat():
    """D31 (WAVE F): an undeclared engine inside a mapped run is DERIVED, not refused.

    REGIME-CHANGE REWRITE. The old claim was "a mapped Granite run whose card
    declares no vector_lanes refuses by name (ADJ-4: scan lanes have no
    default)". D31 retires vector_lanes as an input and makes the DEFAULT
    "derive it from the beat the analog stages set", so the new claim is: the
    run completes, the width is derived, and it is REPORTED with its provenance
    and the stage that bound it. ADJ-4 is not weakened — the refusal survives
    wherever there is no beat to derive from, which
    tests/test_qif_digital_ops.py::test_vector_lanes_have_no_default_and_refuse_by_name
    still pins on a bare device.
    """
    def _no_cards(raw):
        # Dropping the cards drops this machine's `bank_depth: 1` with them, and
        # a placement with no banking needs MORE macro slots for the same
        # weights (556, against the 484 the shipped banked machine is built to).
        # The capacity moves with the placement — that is ADJ-13 — so the test
        # declares the one its own machine needs rather than borrowing a number
        # derived for a different one.
        raw["cim"].pop("cards")
        raw["cim"]["chip"]["arrays_per_chip"] = 556

    hw = _hw(GRANITE_HW, _no_cards)
    mapping = fws_mapping.build_mapping(hw, config.parse_config(GRANITE_MODEL, "LLM"))
    evaluation = fws_eval.evaluate_fws(build_fws_program(mapping))
    sizing = mapping.device.derived_engine
    assert sizing is not None
    assert sizing.vector_lanes >= 1
    assert mapping.device.vector_lanes == sizing.vector_lanes
    assert mapping.device.vector_lanes_provenance == cim_timing.PROVENANCE_DERIVED_COUNT
    # The width HOLDS the analog beat: the binding stage's own priced vector
    # time fits inside it, and one lane fewer would not (no margin, D28).
    binding = next(r for r in sizing.per_stage if r.stage == sizing.binding_stage)
    assert binding.time_s <= sizing.analog_beat_s
    assert binding.used_cycles <= binding.budget_cycles
    # ... and it is REPORTED, which is the other half of D31.
    row = next(
        d for d in evaluation.disclosures if d.constraint == "derived_engine_sizing"
    )
    assert str(sizing.vector_lanes) in row.value
    assert "D31" in row.reason
    # This card names no synthesis library, so nothing is composed for it (D32
    # composes from MEASURED blocks or not at all).
    assert sizing.composition is None


# ---------------------------------------------------------------------------
# The run: every block kind reaches a named law
# ---------------------------------------------------------------------------


def test_every_hybrid_block_kind_is_priced_by_a_named_law(granite):
    laws = {}
    for cost in granite.pricing.costs:
        laws.setdefault(cost.block, set()).add(cost.basis.split("(")[0].strip())
    assert "ssm_scan" in laws, sorted(laws)
    assert any("price_ssm_block" in basis for basis in laws["ssm_scan"])
    # WAVE F (D29/D30): a mapped run lowers DECODE only — the streams arrive
    # already prefilled — so attention is priced by the decode law, and there
    # is no lm_head block at all because D30 drops the endpoints entirely.
    assert any("decode_attention_timing" in b for b in laws["attention_qk"])
    assert "lm_head" not in laws
    for block in ("ssm_in_proj", "ssm_out_proj", "ffn1_routed", "ffn2_shared", "o_proj"):
        assert any("price_tiled_op" in b for b in laws[block]), block
    # No op may be priced by an unnamed law, ever.
    assert all(cost.basis for cost in granite.pricing.costs)


def test_the_layer_plan_reaches_the_placement_layer_by_layer(granite):
    """36 Mamba layers and 4 attention layers, in the pattern the YAML declares."""
    scan_layers = {
        cost.layer for cost in granite.pricing.costs if cost.block == "ssm_scan"
    }
    attention_layers = {
        cost.layer for cost in granite.pricing.costs if cost.block == "attention_qk"
    }
    assert len(scan_layers) == 36
    assert sorted(attention_layers) == [5, 15, 25, 35]
    assert not (scan_layers & attention_layers)


def test_the_short_conv_rides_the_pool_and_never_becomes_a_timed_op(granite):
    """ADJ-3: Mamba-2's d_conv = 4 is absorbed into the pool sizing, not timed."""
    assert granite.mapping.model.ssm.d_conv == 4
    assert not [cost for cost in granite.pricing.costs if cost.block == "short_conv"]
    # And the absorption is DISCLOSED rather than silent.
    assert any(
        item.constraint == "macro_pool_helper_ops" for item in granite.disclosures
    )


def test_the_shared_expert_is_wider_than_a_routed_one_everywhere(granite):
    """Granite's shared MLP is 1024 wide beside 64 routed experts at 512.

    llm_util's parameter census already read ``ffn_dims.shared_expert``; the
    array census and the mapping now read the SAME number, so one model does
    not carry two widths (D21).
    """
    params = granite.mapping.device.params
    assert params.moe_intermediate == 512
    assert params.shared_expert_intermediate == 1024
    census = granite.mapping.device.moe_layer_stage_arrays()
    assert census["ffn1_shared"] == 2  # ceil(2 * 1024 / 1536)
    assert census["ffn1_routed"] == 64 * 1  # ceil(2 * 512 / 1536)
    shapes = {
        (shape.op, shape.n)
        for shape in fws_mapping._moe_stages(granite.mapping.device)
        if shape.op == "ffn1_shared"
    }
    assert shapes == {("ffn1_shared", 2048)}


def test_the_shared_expert_width_defaults_to_the_routed_width(granite):
    """Degenerate identity: a model with no ffn_dims moves no array count."""
    device = granite.mapping.device
    params = device.params
    plain = cim_timing.CimModelParams(
        **{**params.__dict__, "shared_intermediate_size": 0}
    )
    assert plain.shared_expert_intermediate == plain.moe_intermediate


# ---------------------------------------------------------------------------
# Physics audits (DESIGN2 §8 style): peaks, and the M-law by hand
# ---------------------------------------------------------------------------


def test_no_op_exceeds_the_engine_peak_its_law_declares(granite):
    """A law that outruns its own silicon is wrong even when parity passes."""
    device = granite.mapping.device
    peak_ops_per_s = device.vector_lanes * device.vector_clock_hz
    checked = 0
    for cost in granite.pricing.costs:
        ops = float((cost.detail or {}).get("ops", 0.0))
        if ops <= 0 or cost.duration_s <= 0:
            continue
        checked += 1
        assert ops / cost.duration_s <= peak_ops_per_s * (1 + 1e-9), (
            cost.block,
            ops / cost.duration_s,
            peak_ops_per_s,
        )
    assert checked > 0, "no vector-engine op was audited"


def test_the_scan_engine_runs_at_peak_and_is_no_longer_what_binds(granite):
    """ADJ-10 REWRITE of an ADJ-9 rewrite. The scan is still not the binder.

    OLD CLAIM (ADJ-9's version of this test): the scan fits inside its stage's
    analog m-pass, and what sets the beat is the attention fabric at 18.60 us
    of `attention_qk` — more than 40x the scan and more than 10x the whole
    analog m-pass of its stage.

    NEW CLAIM: ADJ-10 derives the fabric too, so the attention term falls from
    18.60 us to 6.88 us (`num_arrays` 2 -> 8, one fold per array) and the
    ratios move with it: `attention_qk` is now ~17x the scan, not 40x. The two
    halves that do NOT move are the ones that matter — the scan still runs at
    its declared peak, and the analog m-pass STILL does not bind, because the
    attention floor is 4.3x it. What changed is that the excess is now the
    DECLARED array geometry's own floor rather than an underived width.

        ops = 1 972 224, lanes = 5479, depth = 20, clock = 0.95 GHz
        cycles = 20 + ceil(ops / lanes) - 1, at the derived width
        peak-relative = ops / (lanes x cycles), which stays above 0.94
    """
    device = granite.mapping.device
    peak = device.vector_lanes * device.vector_clock_hz
    scan = next(
        cost for cost in granite.pricing.costs if cost.block == "ssm_scan"
    )
    ops = float(scan.detail["ops"])
    cycles = device.vector_pipeline_depth + math.ceil(ops / device.vector_lanes) - 1
    # ADJ-13/ADJ-14 MOVED THE SHIPPED MACHINE. Granite now ships the design
    # its own sweep chose - 4 layers per chip, 10 chiplets, bank_depth 1, and
    # a chip BUILT to that placement (484 slots, not a declared 640). It is a
    # strictly better machine (65325 tokens/s at 10784 mm2, against 62661 at
    # 12526), so the reference numbers below moved with it.
    assert cycles == 322
    utilisation = ops / scan.duration_s / peak
    assert utilisation == pytest.approx(ops / (device.vector_lanes * cycles), rel=1e-12)
    assert 0.94 < utilisation <= 1.0
    # THE FIXED POINT: the scan no longer dwarfs the analog side it shares a
    # stage with. It is under a QUARTER of the stage's analog m-pass, which is
    # what "analog-bound by construction" means op by op. (It was under a fifth
    # on the machine that declared 640 slots; banking shortened the analog pass
    # too, so the ratio moved with both terms.)
    sizing = device.derived_engine
    for row in sizing.per_stage:
        assert row.time_s <= row.analog_time_s
    # Against the STAGE's analog m-pass, which is what the claim above is
    # about. It used to be read against a single ssm_in_proj op as a proxy;
    # banking made that op a 2-cycle m-pass (20 ns), so the proxy stopped
    # standing for the stage and the comparison is now taken against the
    # quantity the sentence names.
    stage_analog = max(row.analog_time_s for row in sizing.per_stage)
    assert scan.duration_s < stage_analog / 4
    # And what DOES set the beat is measured and named — still the attention
    # fabric, but at the floor ADJ-10 walked it to rather than at a width
    # nobody derived. The ratio to the scan fell from >40x on the pre-ADJ-10
    # machine to ~20x here.
    qk = next(cost for cost in granite.pricing.costs if cost.block == "attention_qk")
    assert 15 < qk.duration_s / scan.duration_s < 25
    # The PROBE's own evidence is what the derivation ran against, and it is a
    # DIFFERENT quantity from what binds the machine that gets built. Both are
    # reported and neither is merged into the other (D21).
    setter = sizing.beat_setting_stage
    assert setter["analog_is_largest_term"] is False
    assert setter["terms"][0]["block"] == "attention_qk"
    assert setter["terms"][0]["device_class"] == "shared_digital"
    assert "the evidence the derivation RAN AGAINST" in setter["basis"]
    binding = granite.binding_term
    assert binding["terms"][0]["block"] == "attention_qk"
    assert binding["terms"][0]["busy_s"] == pytest.approx(qk.duration_s, rel=1e-12)
    # The machine that gets built is FASTER than the one the derivation looked
    # at, on exactly the term ADJ-10 derived.
    assert binding["terms"][0]["busy_s"] < 0.4 * setter["terms"][0]["busy_s"]


def test_the_attention_fabric_derives_by_integer_copies_of_the_measured_block(granite):
    """ADJ-10's core claim, hand-computed on the shipped Granite machine.

    NEW GATE (ADJ-10). `cim.fabric.num_arrays` used to be a DECLARED 2 and the
    beat-setter; it is now DERIVED to 8 and `softmax_lanes` from 1 to 12, both
    in integer copies of MEASURED synthesis blocks. The array's 32 x 64
    geometry is NOT derived and must not become derivable.

    THE HAND COMPUTATION (Granite decode, one attention layer, stream at
    context 1800; rows R = 32, cols C = 64, f = 0.95 GHz, fill/drain = 96):

        heads_chip = kv_heads = 4, replicas = 1 -> heads_per_replica = 4
        streams = 1 (D29: local batch is always 1) -> folds = 4 x 1 = 4
        score call dims: m = shared_heads = 12/4 = 3, k = head_dim = 128,
                         n = context = 1800

      AT THE DECLARED num_arrays = 2, groups (1, 1), all 4 folds on one array:
        QK = ceil(3/32) * ceil(1800/64) * (128*4 + 32 + 64 - 2) - 1
           = 1 * 29 * 606 - 1 = 17573 cycles  (+96 = 17669 -> 18.60 us)
        PV = ceil(3/32) * ceil(128/64) * (1800*4 + 94) - 1
           = 1 * 2 * 7294 - 1 = 14587 cycles          (-> 15.35 us)

      AT THE DERIVED num_arrays = 8, groups (4, 4), ONE fold per array:
        QK = 1 * 29 * (128 + 94) - 1 = 29 * 222 - 1 = 6437  (+96 -> 6.88 us)
        PV = 1 * 2 * (1800 + 94) - 1 = 2 * 1894 - 1 = 3787       (-> 3.99 us)

      AND THAT IS THE FLOOR. A ninth array joins one of the groups and carries
      NO fold (ceil(4/5) = ceil(4/4) = 1), so it buys nothing. What is left is
      one array's own walk: 29 column passes of ceil(1800/64), each costing
      k + R + C - 2 cycles. Only a wider C would shorten it, and C is the
      geometry the systolic law was validated at — a card change, not a
      derivation (ADJ-4).
    """
    device = granite.mapping.device
    fabric = device.derived_fabric
    assert fabric is not None
    assert device.fabric.num_arrays == 2 and device.fabric.softmax_lanes == 1
    assert fabric.num_arrays == 8 and device.fabric_num_arrays == 8
    assert fabric.softmax_lanes == 12 and device.softmax_width == 12
    assert device.fabric_provenance == "derived-count"
    # ONLY THE COUNT IS DERIVED.
    assert (fabric.rows, fabric.cols) == (32, 64)
    assert (device.fabric.rows, device.fabric.cols) == (32, 64)
    # The hand computation above, taken through the pricing law itself.
    timing = device.decode_attention_timing(1800, batch_size=1, tp=1)
    assert (timing.heads_chip, timing.heads_per_replica, timing.folds) == (4, 4, 4)
    assert (timing.qk_arrays, timing.pv_arrays) == (4, 4)
    assert (timing.qk_folds_per_array, timing.pv_folds_per_array) == (1, 1)
    assert timing.qk_cycles == 6437
    assert timing.pv_cycles == 3787
    # A NINTH ARRAY BUYS NOTHING: the saturation claim, measured.
    from cim_timing import AttentionCallDemand

    call = AttentionCallDemand(
        m=3, k=128, n=1800, folds=4, softmax_tokens=3, heads_chip=4,
        fill_drain_cycles=96,
    )
    assert call.saturation_arrays == 8
    at_8 = device.attention_cycles_at(call, 8, 12)
    at_9 = device.attention_cycles_at(call, 9, 12)
    at_64 = device.attention_cycles_at(call, 64, 12)
    assert at_8[:3] == at_9[:3] == at_64[:3] == (6437, 3787, 20)
    # And the declared-width law is the pass-1 law, bit for bit.
    at_2 = device.attention_cycles_at(call, 2, 1)
    assert at_2[:3] == (17573, 14587, 31)
    assert at_2[3:] == (1, 1)


def test_the_fabric_derivation_names_its_floor_instead_of_clamping_it(granite):
    """ADJ-10: an unreachable target SATURATES and is NAMED, never clamped.

    NEW GATE. Granite's attention stages cannot reach their own 1.60 us analog
    m-pass at any array count — the floor is 10.88 us — so every one of them is
    counted in `saturated_stages`, carries `target_kind` naming the reason, and
    keeps the SATURATION width. The thing this test forbids is the two ways of
    lying about it: clamping the width down to something that "fits" a wider
    budget, and padding it up past the point where copies stop buying time.
    """
    fabric = granite.mapping.device.derived_fabric
    assert fabric.saturated_stages == fabric.unreachable_stages
    assert len(fabric.saturated_stages) == fabric_stages(fabric)
    assert fabric.analog_bound_stages == ()
    for row in fabric.per_stage:
        assert row.target_kind == "fold_concurrency_saturated_below_analog_time"
        # SATURATED means: at the installed width the stage is already at its
        # floor, and the floor is above the target. Both halves, measured.
        assert row.time_s == pytest.approx(row.floor_time_s, rel=1e-12)
        assert row.floor_time_s > row.analog_time_s
        assert row.num_arrays == 8
    # NOT PADDED: the ratio to the target is reported ABOVE 1 rather than
    # capped, which is the number a reader needs to see the residual at all.
    assert fabric.target_ratio == pytest.approx(8.5033, rel=1e-4)
    # The disclosure says the words, so the artifact carries the finding.
    text = " ".join(fabric.disclosures)
    assert "FOLD CONCURRENCY SATURATES" in text
    assert "never clamped and never padded" in text


def fabric_stages(fabric):
    return len(fabric.per_stage)


def test_the_other_derived_width_is_at_the_analog_floor_too(granite):
    """ADJ-9 covers EVERY derived engine width, and D12's pool is the other one.

    NEW GATE. The scan/vector engine is the width ADJ-9 retargets, but it is
    not the only DERIVED one: D12 sizes each analog macro's digital pool from
    that macro's own peak RESULT RATE, which is already the analog floor by
    construction — a pool that consumes exactly what its macro emits can never
    be the term that sets a stage's time. This test is the measurement that
    says so on the shipped machine rather than leaving it to the derivation's
    docstring, and it is the reason ADJ-9 needed no second retarget.

    The comparison is deliberately harsh: the SUM of the stage's pool-op
    durations (not their union) against the stage's analog m-pass. Pools run in
    parallel across macros, so the sum over-counts by roughly the macro count,
    and the claim survives it anyway.
    """
    import fws_eval as _fws_eval

    select, _ = _fws_eval._measurement_slice(granite.serving)
    sizing = granite.mapping.device.derived_engine
    analog_by_stage = {row.stage: row.analog_time_s for row in sizing.per_stage}
    pool_by_stage = {}
    for annotation, cost in zip(
        _fws_eval.annotations_of(granite.program), granite.pricing.costs
    ):
        if cost.device_class != "macro_pool" or not select(annotation):
            continue
        stage = int(annotation.stage)
        pool_by_stage[stage] = pool_by_stage.get(stage, 0.0) + float(cost.duration_s)
    assert pool_by_stage, "the run prices no pool work at all"
    for stage, pool_s in pool_by_stage.items():
        analog_s = analog_by_stage[stage]
        assert analog_s > 0.0
        assert pool_s < analog_s, (stage, pool_s, analog_s)
    # D12's derivation is REPORTED beside the number, which is the other half
    # of the decision (the sizing is a report, never a constraint).
    pool = granite.mapping.device.digital_pool_sizing()
    assert pool.lanes >= 1 and pool.result_rate_per_s > 0
    assert pool.lanes == math.ceil(pool.result_rate_per_s / pool.pool_clock_hz)


def test_an_analog_op_matches_the_M_law_computed_by_hand(granite):
    """duration = M * slice_cycles / f_analog * active_column_sets.

    Recomputed here from the YAML's own numbers, not from CimDeviceModel, so a
    change inside the law has to move this line too (the pass-2A spot-check
    pattern).
    """
    device = granite.mapping.device
    slice_cycles = int(device.analog.slice_cycles)
    f_analog = float(device.analog.analog_clock_mhz) * 1e6
    # WAVE F: every op of a filled-pipeline run is a DECODE op at M = 1, so the
    # law is checked on the three blocks rather than on three phases.
    for block, phase in (("qkv", "decode"), ("ffn1_routed", "decode"), ("ssm_out_proj", "decode")):
        cost = next(
            c
            for c in granite.pricing.costs
            if c.block == block and c.phase == phase and c.kind == "weight_gemm"
        )
        tokens = float(cost.basis.split("m_tokens=")[1].split(",")[0])
        columns = float(cost.detail["active_column_sets"])
        assert cost.duration_s == pytest.approx(
            tokens * slice_cycles / f_analog * columns, rel=1e-12
        )


def test_a_routed_expert_sees_the_routed_token_count_by_hand(granite):
    """ceil(m * top_k * alpha / E) at the ONE token a stage holds per beat.

    WAVE F (D29): the local batch is 1, so the hot-expert count is
    ceil(1 * 6 / 64) = 1 — one token reaches an expert or it does not, and the
    law rounds up to the expert that gets it. The retired regime asked the same
    question at m = B x S = 7168 and got 672.
    """
    params = granite.mapping.device.params
    owner_tokens = 1.0
    expected = math.ceil(owner_tokens * params.top_k * params.expert_imbalance_factor
                         / params.num_experts)
    assert expected == 1
    assert granite.mapping.device.moe_tokens_hot(1) == 1
    cost = next(c for c in granite.pricing.costs if c.block == "ffn1_routed")
    assert "m_tokens=%d" % expected in cost.basis


# ---------------------------------------------------------------------------
# The artifacts: the report and the atlas
# ---------------------------------------------------------------------------


def test_the_report_is_one_document_with_every_component_labeled(granite):
    document = fws_eval.report_document(granite)
    assert document["schema"] == fws_eval.REPORT_SCHEMA
    energy = document["evaluation"]["energy"]
    assert energy["total_pj"] == pytest.approx(
        sum(component["energy_pj"] for component in energy["components"])
    )
    for component in energy["components"]:
        assert component["coverage"] in (
            fws_eval.COVERAGE_COVERED,
            fws_eval.COVERAGE_PARTIAL,
            fws_eval.COVERAGE_UNCOVERED,
        )
    keys = {metric["key"] for metric in document["evaluation"]["metrics"]}
    assert "sys.fws.tokens_per_s" in keys
    # WAVE F (D29): the metric set is the filled pipeline's. There is no
    # requests/s figure at all — a request occupies one stream for decode_len x
    # D beats and NONE completes inside the lowered window, so the only
    # request-scale number is the labeled extrapolation, which is what P4 3
    # allows and what D21 requires (one accounting per metric).
    assert keys == {
        "sys.fws.beat",
        "sys.fws.tokens_per_s",
        "sys.fws.per_stream_tokens_per_s",
        "sys.fws.resident_streams",
        "sys.fws.per_token_latency",
    }
    assert not any(key.startswith("sys.fws.requests_per_s") for key in keys)
    extrapolation = document["evaluation"]["extrapolation"]
    assert extrapolation["extrapolated"] is True
    assert "DECODE_LEN x D x BEAT" in extrapolation["basis"]


def test_the_declared_vector_engine_relaxations_ride_the_report(granite):
    """Every knob the card did NOT declare is named in the artifact (D21).

    Two separate conditions, asserted separately: the disclosure EXISTS under a
    ``vector_engine:`` constraint, and that entry's REASON names the undeclared
    knob. (An earlier version asserted `A or B` where both branches were the
    same joined string, so it passed without distinguishing the two shapes.)
    """
    engine = [
        item for item in granite.disclosures if item.constraint.startswith("vector_engine:")
    ]
    assert engine, "an inherited or undeclared engine knob must be disclosed"
    named = {
        knob: [item for item in engine if knob in item.constraint]
        for knob in ("vector_clock_ghz", "vector_pipeline_depth", "state_bytes_per_cycle")
    }
    for knob, entries in named.items():
        assert entries, knob
        assert all(knob in item.reason for item in entries), knob
        assert all(len(item.reason) > 80 for item in entries), knob


def test_the_atlas_export_covers_every_placed_macro(granite):
    document = fws_atlas_export.export_atlas(
        [granite.mapping],
        title="Granite-4.0-H-Tiny",
        duty_cycles=[granite.atlas_duty_cycles()],
        extra_metrics=[granite.atlas_metrics()],
    )
    assert document["schema"] == "fws_atlas/1"
    # EVERY PLACED MACRO AND EVERY TILE IS EXPORTED. Macros are no longer >=
    # tiles: this machine ships bank_depth 1, so one macro holds several tiles
    # (that is what banking IS), and the count that has to close is the tiles.
    # What the macro side pins instead is coverage: every macro a tile sits on
    # is in the document.
    assert len(document["tiles"]) == len(granite.mapping.tiles)
    exported = {macro["id"] for macro in document["macros"]}
    placed = {
        macro["id"] for macro in document["macros"]
        if macro.get("tiles")
    }
    assert placed and placed <= exported
    assert len(placed) == len(
        {slot.macro_id for slot in granite.mapping.macros if slot.tiles}
    )
    # D23: accuracy is never a factor, anywhere, including in a viz payload.
    assert "accuracy" not in json.dumps(document).lower()


# ---------------------------------------------------------------------------
# Qwen3.5-4B: the linear-attention row (ADJ-1's second P0)
# ---------------------------------------------------------------------------


def test_qwen3_5_4b_linear_attention_prices_on_the_mapped_path():
    """ADJ-1's second P0 no longer skips: it runs.

    P1.5 left it refused BY NAME on attention.output_gate — the delta law was
    wired but no stage placed the gate matrix. C1 places W_g as an ordinary
    analog tile, so this test now proves the delta law prices every linear
    layer of a real model instead of proving a refusal. The end-to-end proof of
    the GATE lives in tests/test_qif_qwen_e2e.py; what is asserted here is the
    thing this module has always asserted — the law, on its consumer.
    """
    hw = _hw(os.path.join(HW_DIR, "fws_cim_qwen3_5_4b.yaml"))
    model = config.parse_config(QWEN_MODEL, "LLM")
    config.validate_configs(hw, model)
    mapping = fws_mapping.build_mapping(hw, model)
    evaluation = fws_eval.evaluate_fws(build_fws_program(mapping))
    delta = [cost for cost in evaluation.pricing.costs if cost.block == "delta_rule"]
    assert delta
    assert all("price_linear_attention_block" in cost.basis for cost in delta)
    peak = mapping.device.vector_lanes * mapping.device.vector_clock_hz
    for cost in delta:
        ops = float((cost.detail or {}).get("ops", 0.0))
        if ops > 0 and cost.duration_s > 0:
            assert ops / cost.duration_s <= peak * (1 + 1e-9)


def test_the_granite_run_places_no_gate_because_granite_declares_none(granite):
    """The C1 narrowing is guarded by the model's own declaration.

    Granite-4.0-H-Tiny sets no attn_output_gate, so its placement, its report
    and its atlas must be exactly what they were before the gate existed.
    """
    assert not bool(granite.mapping.model.attention.output_gate)
    assert not [o for o in granite.mapping.owners() if o.op == "attn_gate_proj"]
    assert not [c for c in granite.pricing.costs if c.block == "attn_output_gate"]


def test_the_shipped_granite_atlas_validates_through_the_real_loader(tmp_path):
    """The emitted artifact, judged by atlas.html's own ATLAS.validate (P5.3).

    The document under docs/qif/atlas is a deliverable someone will open. A
    schema it does not satisfy is a broken page, not a failing test.
    """
    from tests.test_qif_atlas import _run_core

    document = os.path.join(
        PROJECT_ROOT, "docs", "qif", "atlas", "granite_4_0_h_tiny.json"
    )
    # A missing artifact FAILS (never skips). The repo's own precedent is
    # test_run_perf_gpu_results_unchanged_vs_baseline: a silent skip would let
    # deleting the file turn this gate green, which disarms it.
    assert os.path.exists(document), (
        "docs/qif/atlas/granite_4_0_h_tiny.json is the ADJ-1 headline artifact "
        "and is committed; regenerate it with the command in its provenance block"
    )
    out = _run_core(
        """
        const v = ATLAS.validate(doc);
        console.log(JSON.stringify({
          errors: v.errors.map(e => e.rule + ' ' + e.field),
          warnings: v.warnings.map(w => w.rule + ' ' + w.field),
        }));
        """,
        tmp_path,
        document=__import__("pathlib").Path(document),
    )
    verdict = json.loads(out)
    assert verdict["errors"] == []
    assert verdict["warnings"] == []


def test_the_atlas_draws_a_service_wire_to_every_analog_chip():
    """NEW GATE (results.html pipeline map): the `svc` links, and their bytes.

    The atlas drew the `act` boundary between consecutive analog chips and
    NOTHING for the relationship between an analog chip and the shared digital
    chiplet that runs its attention and its scan (D13) — so the picture showed
    a pipeline with no wires to the silicon doing half its work. One `svc` row
    now exists per (chiplet -> analog chip) service relationship, read off the
    LOWERED DAG (each shared-digital op names its chiplet and the layer it
    serves; the placement names the chip that holds that layer), with bytes
    MEASURED off P4's timeline.
    """
    path = os.path.join(PROJECT_ROOT, "docs", "qif", "atlas", "granite_4_0_h_tiny.json")
    document = json.load(open(path, encoding="utf-8"))
    svc = [link for link in document["links"] if link["role"] == "svc"]
    analog_chips = [c for c in document["chips"] if c["pool"] == "analog"]
    digital_chips = {c["id"] for c in document["chips"] if c["pool"] == "digital"}
    # One wire per analog chip, and every wire runs from a chiplet to a chip.
    assert len(svc) == len(analog_chips) == 10
    assert {link["to"] for link in svc} == {c["id"] for c in analog_chips}
    assert {link["from"] for link in svc} <= digital_chips
    for link in svc:
        assert link["per"] == "beat"
        assert link["bytes"] >= 0.0
        assert "MEASURED on beat" in link["basis"]
        assert "operands out" in link["basis"] and "results back" in link["basis"]
    # The attention stages carry real bytes; the SSD-only stages carry none,
    # and the basis says WHICH component is unpriced rather than the total
    # absorbing an estimate (D21, D28).
    carrying = [link for link in svc if link["bytes"] > 0]
    empty = [link for link in svc if link["bytes"] == 0]
    assert len(carrying) == 4 and len(empty) == 6
    assert {link["bytes"] for link in carrying} == {8192.0}
    for link in svc:
        assert "UNPRICED COMPONENT: ssm_scan" in link["basis"]
        assert "absent op, not a measured zero" in link["basis"]
    for link in carrying:
        assert "qkv 5120 B" in link["basis"] and "o_proj 3072 B" in link["basis"]
    # And the byte total is the evaluator's own accounting, not a second one:
    # 5120 out + 3072 back is exactly what the lowering annotated.
    assert all(
        link["bytes"] == 5120.0 + 3072.0 for link in carrying
    )


def test_the_shipped_granite_atlas_carries_p4s_duty_cycles():
    """--priced is what makes it the FULL artifact rather than a placement."""
    path = os.path.join(PROJECT_ROOT, "docs", "qif", "atlas", "granite_4_0_h_tiny.json")
    assert os.path.exists(path), (
        "docs/qif/atlas/granite_4_0_h_tiny.json is committed; a missing artifact "
        "must fail this gate, never skip it"
    )
    with open(path) as handle:
        document = json.load(handle)
    assert "--priced" in document["provenance"]["reference_run"]["command"]
    duty = [
        macro["duty_cycle"]
        for macro in document["macros"]
        if macro.get("duty_cycle") is not None
    ]
    assert duty, "the priced atlas carries no duty cycle"
    assert all(0.0 <= value <= 1.0 for value in duty)


def test_the_shipped_granite_report_is_what_a_fresh_run_produces(granite):
    """The committed artifact cannot rot: it is regenerated and compared.

    The report is a pure function of the two YAMLs — no clock, no path, no
    ordering by hash — so byte equality is the right bar. A law change that
    moves a Granite number now has to move this file too, in the same commit.
    """
    path = os.path.join(
        PROJECT_ROOT, "docs", "qif", "atlas", "granite_4_0_h_tiny_report.json"
    )
    assert os.path.exists(path), (
        "docs/qif/atlas/granite_4_0_h_tiny_report.json is committed; a missing "
        "artifact must fail this gate, never skip it"
    )
    shipped = open(path, encoding="utf-8").read()
    # BYTES, against the exact serialization inference_timing._write_fws_dag_report
    # uses (json.dump(..., indent=2), no trailing newline). Comparing the parsed
    # objects would prove value equality only.
    #
    # WAVE C AUDIT: what this gate proves is byte equality with
    # fws_eval.report_document, regenerated in-process. No run_perf subprocess
    # runs here, so it is not evidence about run_perf's serialization, key order
    # or indent -- the status files and the plan page now say so.
    assert shipped == json.dumps(fws_eval.report_document(granite), indent=2)


def test_the_shipped_granite_atlas_is_what_a_fresh_run_produces(tmp_path):
    """The priced atlas cannot rot either — regenerate and compare BYTES.

    tests/test_qif_mapping.py gives p3_llama7b_tp2.json exactly this gate. The
    Granite atlas is the ADJ-1 HEADLINE artifact and had none, so a law change
    could move a Granite number and leave the shipped picture stale.
    """
    import tools.fws_emit_atlas as emit
    import fws_atlas_export

    _mapping, _rows, document = emit.build(GRANITE_HW, GRANITE_MODEL, priced=True)
    regenerated = fws_atlas_export.write_atlas_json(
        document, os.path.join(str(tmp_path), "granite.json")
    )
    path = os.path.join(
        PROJECT_ROOT, "docs", "qif", "atlas", "granite_4_0_h_tiny.json"
    )
    assert regenerated == open(path, encoding="utf-8").read(), (
        "docs/qif/atlas/granite_4_0_h_tiny.json is stale; regenerate it with "
        "the command in its provenance block"
    )


def test_the_granite_artifacts_name_the_model_and_not_its_pricing_carrier():
    """D21's named owner has to be a name a reader can TRUST.

    granite_4_0_h_tiny_inf.yaml declares model_type: llama because the SwiGLU
    gated MLP is the arithmetic carrier. That is a pricing choice, not the
    model's identity, and every owner, label and title names the model.
    """
    atlas = json.load(
        open(
            os.path.join(
                PROJECT_ROOT, "docs", "qif", "atlas", "granite_4_0_h_tiny.json"
            ),
            encoding="utf-8",
        )
    )
    assert "granite" in atlas["title"].lower()
    system = atlas["systems"][0]
    assert "granite" in system["label"].lower()
    assert system["models"][0]["id"] == "granite_4_0_h_tiny"
    assert system["models"][0]["label"] == "granite_4_0_h_tiny"
    assert atlas["tiles"]
    assert all(tile["owner"]["model"] == "granite_4_0_h_tiny" for tile in atlas["tiles"])
    # The carrier is reconciled, not hidden.
    carrier = [
        item for item in atlas["relaxations"] if item["constraint"] == "model_type_carrier"
    ]
    assert carrier and "llama" in carrier[0]["value"]

    report = json.load(
        open(
            os.path.join(
                PROJECT_ROOT, "docs", "qif", "atlas", "granite_4_0_h_tiny_report.json"
            ),
            encoding="utf-8",
        )
    )
    assert report["model"]["model_id"] == "granite_4_0_h_tiny"
    assert report["model"]["model_type"] == "llama"


def test_the_atlas_and_the_report_size_the_pool_the_same_way(granite):
    """One quantity, one number, across two artifacts (D21).

    macros[].digital_pool and the report's P4.5 pool table are the same thing.
    Before this gate the atlas printed P2.5's per-unit derivation and the report
    printed it multiplied by the measured concurrency, so one macro carried two
    act_lanes figures under one field name in one run.
    """
    atlas = json.load(
        open(
            os.path.join(
                PROJECT_ROOT, "docs", "qif", "atlas", "granite_4_0_h_tiny.json"
            ),
            encoding="utf-8",
        )
    )
    sized = {
        int(entry.macro_id): entry.scaled.total_lanes for entry in granite.pools
    }
    assert sized, "the Granite run measures pool demand on some macro"
    checked = 0
    for macro in atlas["macros"]:
        pool = macro.get("digital_pool")
        if pool is None:
            continue
        macro_id = int(macro["id"].rsplit(".", 1)[1])
        if macro_id not in sized:
            continue
        assert pool["act_lanes"] == sized[macro_id], macro["id"]
        checked += 1
    assert checked == len(sized)
    # A pool area of zero is an ABSENT LAW, and the artifact says which.
    zero = [
        macro["digital_pool"]
        for macro in atlas["macros"]
        if macro.get("digital_pool") and macro["digital_pool"]["area_mm2"] == 0.0
    ]
    assert zero
    assert all("UNCOVERED" in entry["basis"] for entry in zero)


def test_the_granite_headline_numbers_are_the_ones_reported(granite):
    """Pins what the P1.5 status report states, so the two cannot drift apart.

    WAVE F (D29/D30). Granite is a FILLED PIPELINE now: 10 stages (one per
    analog chiplet), D = 10 resident streams, one token out per beat, local
    batch 1, and no lm_head (66 macros gone, 5610 -> 5544). There is no prefill
    figure any more — the streams arrive already prefilled (D25) — and the
    headline is 1 / beat, not tokens over a batched step.
    """
    document = fws_eval.report_document(granite)
    mapping = document["mapping"]
    assert mapping["analog_chips"] == 10
    # ADJ-13/ADJ-14 MOVED THE SHIPPED MACHINE. Granite now ships the design
    # its own sweep chose - 4 layers per chip, 10 chiplets, bank_depth 1, and
    # a chip BUILT to that placement (484 slots, not a declared 640). It is a
    # strictly better machine (65325 tokens/s at 10784 mm2, against 62661 at
    # 12526), so the reference numbers below moved with it.
    # bank_depth 1 lets the dense packer save 716 macros outright, which is
    # where 5544 became 4828 (the TILE count is unchanged at 19304: the same
    # weights, packed into fewer macros).
    assert mapping["macros_holding_tiles"] == 4828
    assert mapping["tiles"] == 19304
    assert mapping["shared_digital_chiplets"] == 10
    assert mapping["serving_regime"] == "filled_pipeline"
    assert mapping["resident_streams"] == 10
    metrics = {metric["key"]: metric["value"] for metric in document["evaluation"]["metrics"]}
    assert "sys.fws.prefill_latency" not in metrics
    # ADJ-13 REWRITE. The machine this asserts on is the one Granite's own
    # sweep selected, not a hand-declared one: 4 layers per chip, 10 shared
    # digital chiplets, bank_depth 1, and a chip BUILT to that placement (484
    # macro slots). Beat 15.31 us / 65325.1 tokens/s at 10784 mm2.
    #
    # The trail behind it: 62661.5 tokens/s at 15.96 us and 12526 mm2 (ADJ-10,
    # the same laws on a chip that DECLARED 640 slots and enumerated 156 empty
    # ones per chip); 25617.6 at 39.04 us (ADJ-9, fabric left at its declared 2
    # arrays); 15103.6 at 66.21 us (a 220-lane engine sized against the analog
    # BEAT); 22737 at 43.98 us (a card that DECLARED 1024 lanes nobody
    # measured).
    #
    # +144.6% tokens/s for +92.8 mm2 of digital silicon, which is +0.70% of the
    # machine — the same trade ADJ-9 was adjudicated on, on the engine ADJ-9
    # did not size. The scan width does NOT move (5479 either way): its target
    # is the analog m-pass, and the analog m-pass did not move.
    assert metrics["sys.fws.beat"] == pytest.approx(1.5308054e-05, rel=1e-3)
    assert metrics["sys.fws.tokens_per_s"] == pytest.approx(65325.1, rel=1e-3)
    assert metrics["sys.fws.per_stream_tokens_per_s"] == pytest.approx(6532.51, rel=1e-3)
    assert metrics["sys.fws.resident_streams"] == 10.0
    # The engine is REPORTED, which is the other half of D31, and its silicon is
    # COMPOSED from the measured 22nm synthesis library (D32).
    silicon = document["evaluation"]["digital_silicon"]
    assert silicon["vector_lanes"] == 6509
    assert silicon["vector_lanes_provenance"] == "derived-count"
    assert silicon["library"]["technology"] == "22nm"
    assert silicon["digital_area_mm2_total"] == pytest.approx(319.55105, rel=1e-4)
    # ADJ-9's fixed point, on the shipped headline machine: all ten stages.
    sizing = silicon["derived_engine_sizing"]
    assert sizing["sizing_target"] == "analog_stage_time"
    assert sizing["analog_bound_stages"] == list(range(10))
    assert sizing["unreachable_stages"] == []
    assert sizing["engine_duty_at_target"] == pytest.approx(0.996904, rel=1e-4)
    # ADJ-10's fixed point beside it, and ADJ-10's honest failure: the fabric
    # is derived to 8 arrays and 12 softmax lanes, and it still cannot reach
    # the analog floor on any of its four attention stages.
    fabric = silicon["derived_fabric_sizing"]
    assert silicon["fabric_num_arrays"] == 8
    assert silicon["fabric_num_arrays_provenance"] == "derived-count"
    assert silicon["fabric_softmax_lanes"] == 12
    assert fabric["num_arrays_declared"] == 2 and fabric["softmax_lanes_declared"] == 1
    assert fabric["array_rows"] == 32 and fabric["array_cols"] == 64
    assert fabric["array_geometry_provenance"] == "declared-count"
    assert fabric["saturated_stages"] == [1, 3, 6, 8]
    assert fabric["analog_bound_stages"] == []
    # WHAT BINDS THE MACHINE THAT GETS BUILT, in the artifact, with numbers.
    binding = silicon["binding_term"]
    # Which of the four attention stages wins is a tie broken by the span
    # measurement; the point the gate is making is that an ATTENTION stage
    # binds, and that it is one of the saturated ones.
    assert binding["stage"] in fabric["saturated_stages"]
    assert binding["analog_is_largest_term"] is False
    assert binding["terms"][0]["block"] == "attention_qk"
    assert binding["terms"][0]["busy_s"] == pytest.approx(6.876842e-06, rel=1e-4)
    assert binding["analog_time_s"] == pytest.approx(1.32e-06, rel=1e-6)
    named = next(
        item for item in document["disclosures"] if item["constraint"] == "binding_term"
    )
    assert "THE NEXT REAL LIMIT IS THE DECLARED ARRAY GEOMETRY" in named["reason"]
    # The energy is the WINDOW's, and the window is D - 1 fill beats plus the
    # steady sample: a longer window is more beats of real work, not a
    # different machine. It is reported per run, never per token here.
    #
    # WAVE F (D32): it rose from 1.449e9 pJ because the shared digital chiplet
    # is no longer an UNCOVERED component. Its ops are now charged the composed
    # engine's measured power for their own duration, which is a term that
    # always existed and used to report zero.
    assert document["evaluation"]["energy"]["total_pj"] == pytest.approx(
        1.0628268e10, rel=1e-3
    )
    digital = next(
        c
        for c in document["evaluation"]["energy"]["components"]
        if c["key"] == "shared_digital_chiplet"
    )
    assert digital["coverage"] == "partial" and digital["energy_pj"] > 0
    # THE GEORGE CONSTRAINT, on the headline model: every one of the 10 stages
    # holds all 10 streams' state and KV for its own 4 layers.
    residency = document["evaluation"]["state_residency"]
    assert residency["resident_streams"] == 10 and residency["stages"] == 10
    assert residency["max_stage_state_bytes"] == pytest.approx(60413952.0)
    # The store is BUILT to the bill, so the verdict is "sized" and never a
    # violation; what it costs shows up in silicon, not in a refusal.
    assert residency["verdict"] == "sized"
    assert residency["store_macros"] > 0
    assert residency["store_area_mm2"] > 0
