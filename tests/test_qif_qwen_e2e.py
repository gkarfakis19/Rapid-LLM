"""QIF C1: Qwen3.5-4B runs end to end on the mapped fws_cim path.

ADJ-1 made Qwen3.5-4B the SECOND P0 model. Wave B wired the delta-rule law and
placed its linear-attention block, and the model still did not run: it declares
``attn_output_gate``, and P1.4 refused that field by name because no stage
placed the gate matrix. The C1 adjudication closes that gap the boring way --
the output gate is an ORDINARY WEIGHT MATRIX. P3 places it as an analog tile
beside qkv and o_proj, P4 prices it with the same analog law every other weight
matrix gets, and only the sigmoid and the elementwise multiply are per-macro
pool work (ADJ-3 / D12).

This module is the proof, held to the standard the Wave B auditor demanded of
``attention.window``: the narrowing is NOT tested at the schema level. Every
claim below is made against a real mapped run of the shipped pair of YAMLs --
the gate is placed, the gate is priced by the analog law, the pool multiply
rides the pool sizing, and everything the narrowing did not open stays refused.

It is also the first consumer of the delta-rule law, which is LAW_UNVALIDATED
by design. A run whose headline inherits an unchecked law has to SAY so, so the
label is asserted in the report and in the atlas rather than assumed.
"""

from __future__ import annotations

import copy
import json
import math
import os
import pathlib
import tempfile

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
ATLAS_DIR = os.path.join(PROJECT_ROOT, "docs", "qif", "atlas")

QWEN_HW = os.path.join(HW_DIR, "fws_cim_qwen3_5_4b.yaml")
QWEN_MODEL = os.path.join(MODEL_DIR, "qwen3_5_4b_inf.yaml")
GRANITE_HW = os.path.join(HW_DIR, "fws_cim_granite_tiny.yaml")
GRANITE_MODEL = os.path.join(MODEL_DIR, "granite_4_0_h_tiny_inf.yaml")
LLAMA_MODEL = os.path.join(MODEL_DIR, "llama2_7b_fws_inf.yaml")
FWS_LLAMA7B = os.path.join(HW_DIR, "fws_cim_llama7b.yaml")
A100 = os.path.join(HW_DIR, "a100_80GB.yaml")

ATLAS_JSON = os.path.join(ATLAS_DIR, "qwen3_5_4b.json")
REPORT_JSON = os.path.join(ATLAS_DIR, "qwen3_5_4b_report.json")


def _hw(path, mutate=None):
    with open(path) as handle:
        raw = copy.deepcopy(yaml.safe_load(handle))
    if mutate is not None:
        mutate(raw)
    config.convert(raw)
    return config.HWConfig.from_dict(raw)


def _model(path, mutate=None):
    """Parse a model YAML through the real entry point, optionally mutated.

    ``config.parse_config`` reads a FILE, so a mutated model is written back out
    and re-read: nothing here bypasses the parser the shipped runs use.
    """
    with open(path) as handle:
        raw = copy.deepcopy(yaml.safe_load(handle))
    if mutate is not None:
        mutate(raw)
    handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    with handle:
        yaml.safe_dump(raw, handle)
    return config.parse_config(handle.name, "LLM")


#: WAVE F (D29/D30). The SHIPPED Qwen point is a filled pipeline: staggered
#: streams, one token out per beat, local batch 1, no prefill in the DAG at all
#: (the streams arrive prefilled — D25/D29). That is the ``qwen`` fixture and
#: the shipped artifacts.
#:
#: Most of this file predates the pivot and asks LAW questions on a PREFILL
#: pass — the gated-attention tile priced by the analog M-law at m = 7168, the
#: delta-rule work count at that same m, the prefill split that is P1's finding
#: #1. Those laws did not change and the questions are still worth asking, so
#: they run on ``qwen_lockstep``: the same hardware and model under the RETIRED
#: regime, declared by name, with its batch-4 workload intact. Every test that
#: uses it says so in its own first line.
def _qwen_model(lockstep=False):
    model = config.parse_config(QWEN_MODEL, "LLM")
    if lockstep:
        # the workload as it was before D29/D30 removed the batch and the
        # endpoints from this surface
        model.model_config.global_batch_size = 4
        model.model_config.disable_embedding_unembedding = False
    return model


@pytest.fixture(scope="module")
def qwen():
    mapping = fws_mapping.build_mapping(_hw(QWEN_HW), _qwen_model())
    return fws_eval.evaluate_fws(build_fws_program(mapping))


def _lockstep_machine(raw):
    """The machine the retired regime's hand arithmetic was computed against.

    Two things separate it from the shipped machine, and both change what gets
    placed rather than how it is priced:

    * D30 drops the embedding and the lm_head on the shipped path and this
      fixture puts them back, so the last chip carries Qwen's 248320 x 2560
      lm_head and the placement is far larger.
    * The shipped machine ships `bank_depth: 1` (ADJ-13/ADJ-14 selected it),
      which allocates at bank granularity and re-tiles every weight matrix.

    The checks that ride this fixture derive their numbers BY HAND for the
    dedicated allocation on a 180-slot chip -- the gate spanning two macros at
    4 active column sets, the delta-rule law -- so the fixture pins that
    machine instead of following the shipped one. A hand-derived number is an
    INDEPENDENT check, and it stays independent only if the machine it was
    derived for is the machine it runs on.
    """
    raw["cim"]["chip"]["arrays_per_chip"] = 180
    cards = raw.get("cim", {}).get("cards") or {}
    for card in cards.values():
        if isinstance(card, dict):
            card.pop("bank_depth", None)


@pytest.fixture(scope="module")
def qwen_lockstep():
    """The RETIRED regime, by name: the prefill machine the law checks need."""
    mapping = fws_mapping.build_mapping(
        _hw(QWEN_HW, _lockstep_machine),
        _qwen_model(lockstep=True),
        regime=fws_mapping.REGIME_LOCKSTEP,
    )
    assert mapping.regime == fws_mapping.REGIME_LOCKSTEP
    return fws_eval.evaluate_fws(build_fws_program(mapping))


# ---------------------------------------------------------------------------
# The narrowing: what opened, and everything that did not
# ---------------------------------------------------------------------------


def test_qwen3_5_4b_validates_on_the_mapped_fws_cim_path():
    hw = _hw(QWEN_HW)
    assert hw.mapping_config is not None  # the mapping block IS the narrowing
    config.validate_configs(hw, config.parse_config(QWEN_MODEL, "LLM"))


def test_the_same_qwen_config_is_still_refused_without_a_mapping_block():
    """Drop `mapping:` and the closed form has no stage for either new thing."""

    def drop(raw):
        raw.pop("mapping", None)

    with pytest.raises(ValueError) as error:
        config.validate_configs(_hw(QWEN_HW, drop), config.parse_config(QWEN_MODEL, "LLM"))
    assert "linear_attn" in str(error.value)


def test_the_output_gate_is_refused_by_name_off_the_mapped_path():
    """The gate alone, on an attention-only model, on every other path.

    Qwen's own refusal names its layer plan first, so the gate is injected into
    a model whose plan is uncontroversial: that isolates the field under test.
    A GPU run and an UNMAPPED fws_cim run must both still refuse it by name,
    because neither has a stage that places W_g.
    """

    def gate(raw):
        raw["model_param"]["attention"]["output_gate"] = True

    model = _model(LLAMA_MODEL, gate)
    for hardware in (A100, FWS_LLAMA7B):
        with pytest.raises(ValueError) as error:
            config.validate_configs(_hw(hardware), model)
        assert "model_param.attention.output_gate" in str(error.value)


def test_the_same_injected_gate_runs_on_a_mapped_fws_cim_config():
    """The narrowing is the PAIR (fws_cim + a mapping block), nothing wider."""

    def gate(raw):
        raw["model_param"]["attention"]["output_gate"] = True

    def mapped(raw):
        # WAVE F: this llama7b point carries the batch-4 workload of the
        # retired regime, so it declares that regime by name. The question the
        # test asks — does the fws_cim + `mapping:` PAIR open the gate stage —
        # is the same under either one.
        raw["mapping"] = {
            "parallelism": {"tp": 1},
            "decode_window": 1,
            "regime": "lockstep",
        }
        # W_g adds ceil(4096/4096) = 1 macro per layer to the llama7b point.
        raw["cim"]["chip"]["arrays_per_chip"] = 128

    model = _model(LLAMA_MODEL, gate)
    hw = _hw(FWS_LLAMA7B, mapped)
    config.validate_configs(hw, model)
    mapping = fws_mapping.build_mapping(hw, model)
    assert [owner for owner in mapping.owners() if owner.op == "attn_gate_proj"]


def test_the_ungated_models_gain_no_stage_and_do_not_move(qwen):
    """Granite and llama declare no output gate, so nothing is placed for them.

    A narrowing that quietly adds a matrix to every attention model would move
    every shipped artifact in the tree. This is the gate that says it did not.
    """
    granite = fws_mapping.build_mapping(
        _hw(GRANITE_HW), config.parse_config(GRANITE_MODEL, "LLM")
    )
    assert not [owner for owner in granite.owners() if owner.op == "attn_gate_proj"]
    stages = fws_mapping._attention_stages(granite.device, granite.model)
    assert [stage.op for stage in stages] == ["qkv", "o_proj"]
    # ... and the same call WITH the gated model does grow, so the guard is the
    # model's declaration and not an accident of the device card.
    gated = fws_mapping._attention_stages(qwen.mapping.device, qwen.mapping.model)
    assert [stage.op for stage in gated] == ["qkv", "attn_gate_proj", "o_proj"]


# ---------------------------------------------------------------------------
# END TO END: the gate is PLACED, PRICED, and its pool work SIZES the pool
# ---------------------------------------------------------------------------


def test_the_gate_is_placed_as_an_analog_tile_on_every_attention_layer(qwen):
    """One W_g per gated-attention layer, at the census's own (K, N).

    llm_util's attention census already charges `hidden_dim * q_size` extra
    when `output_gate` is set. The placement reads THAT term at (K, N) instead
    of as a scalar, so the mapping and the parameter count are ONE accounting
    (D21) -- not `hidden x hidden`, which would place 2560 x 2560 while the
    census counted 2560 x 4096.
    """
    mapping = qwen.mapping
    params = mapping.device.params
    plan = tuple(mapping.model.layer_mixers)
    attention_layers = [i for i, kinds in enumerate(plan) if "attention" in kinds]
    assert len(attention_layers) == 8  # 3 linear : 1 full, 32 layers

    owners = [owner for owner in mapping.owners() if owner.op == "attn_gate_proj"]
    assert sorted(owner.layer for owner in owners) == attention_layers

    for owner in owners:
        tiles = mapping.tiles_for(owner)
        assert tiles
        # K = hidden_dim, N = num_heads * head_dim (2560 x 4096 for Qwen3.5-4B)
        assert min(tile.k_start for tile in tiles) == 0
        assert max(tile.k_end for tile in tiles) == params.hidden_dim == 2560
        assert max(tile.n_end for tile in tiles) == params.num_heads * params.head_dim == 4096
        # It sits on the same chip as the layer that owns it.
        chips = {mapping.macro(tile.site.macro_id).chip_id for tile in tiles}
        assert chips == {mapping.macro(
            mapping.tiles_for(
                next(o for o in mapping.owners()
                     if o.op == "qkv" and o.layer == owner.layer)
            )[0].site.macro_id
        ).chip_id}


def test_the_gate_matrix_is_the_parameter_censuss_own_term(qwen):
    """The placed gate weights equal the census term, element for element."""
    import llm_util

    model = qwen.mapping.model
    attention = model.attention
    gated = llm_util.attention_block_param_count(
        hidden_dim=int(model.hidden_dim), attention=attention
    )
    ungated = llm_util.attention_block_param_count(
        hidden_dim=int(model.hidden_dim),
        attention=copy.replace(attention, output_gate=False)
        if hasattr(copy, "replace")
        else __import__("dataclasses").replace(attention, output_gate=False),
    )
    placed = sum(
        (tile.k_end - tile.k_start) * (tile.n_end - tile.n_start)
        for owner in qwen.mapping.owners()
        if owner.op == "attn_gate_proj" and owner.layer == 3
        for tile in qwen.mapping.tiles_for(owner)
    )
    assert gated - ungated == placed == 2560 * 4096


def test_the_gate_is_priced_by_the_analog_law_and_matches_it_by_hand(qwen_lockstep):
    """HAND CHECK of the analog M-law on the gate op (DESIGN2 section-8 style).

    Prefill token count m = batch 4 x prefill 1792 = 7168. The gate tile on one
    macro spans 4 column sets (mux slots), and the card charges slice_cycles = 2
    analog cycles per ADC evaluation at 100 MHz:

        t = m * active_column_sets * slice_cycles / f_analog
          = 7168 * 4 * 2 / 1e8 s
          = 573.44 us

    Both of layer 3's gate ops (the matrix spans two macros) must equal that.
    """
    gate_ops = [
        cost
        for cost in qwen_lockstep.pricing.costs
        if cost.block == "attn_gate_proj" and cost.phase == "prefill" and cost.layer == 3
    ]
    assert len(gate_ops) == 2
    for cost in gate_ops:
        assert cost.device_class == "analog_macro"
        assert "price_tiled_op" in cost.basis  # the SAME law every weight GEMM gets
        assert float(cost.detail["active_column_sets"]) == 4.0
        assert cost.duration_s == pytest.approx(7168 * 4 * 2 / 1e8, rel=1e-12)
    # It also carries analog energy, so it is a real tile and not a bookkeeping op.
    assert all(cost.energy_pj > 0 for cost in gate_ops)


def test_the_gate_multiply_is_pool_work_and_rides_the_pool_sizing(qwen_lockstep):
    """ADJ-3 / D12: sigmoid + multiply is elementwise, so it is pool work.

    "Absorbed" is a SIZED claim, not a free one: the op costs 0 s because no
    card declares a per-element pool law, and its CONCURRENCY is measured and
    reported as the derived pool width (P4.5). This asserts both halves.
    """
    multiplies = [cost for cost in qwen_lockstep.pricing.costs if cost.block == "attn_output_gate"]
    # 8 gated layers x (1 prefill + 3 lowered decode steps)
    assert len(multiplies) == 8 * 4
    assert all(cost.device_class == "macro_pool" for cost in multiplies)
    assert all(cost.duration_s == 0.0 for cost in multiplies)

    # Every multiply runs on a macro that HOLDS W_g, which is the point of
    # putting it there: both operands are already local.
    gate_macros = {
        tile.site.macro_id
        for owner in qwen_lockstep.mapping.owners()
        if owner.op == "attn_gate_proj"
        for tile in qwen_lockstep.mapping.tiles_for(owner)
    }
    assert {cost.macro_id for cost in multiplies} <= gate_macros

    # And it reaches P4.5's sizing: each host macro has a pool report whose
    # measured ops are exactly the gate multiplies scheduled on it.
    pools = {report.macro_id: report for report in qwen_lockstep.pools}
    for macro_id in {cost.macro_id for cost in multiplies}:
        assert macro_id in pools, "a placed pool op that sizes no pool is a lost op"
        report = pools[macro_id]
        assert report.pool_ops == len([c for c in multiplies if c.macro_id == macro_id])
        assert report.peak_concurrent_demand >= 1
        assert report.scaled.lanes == report.per_unit.lanes * report.peak_concurrent_demand
        assert "attn_gate_proj" in report.owner


def test_the_gate_sits_between_the_attention_output_and_the_o_proj(qwen_lockstep):
    """The DAG order is the algorithm's order, not a convenient one."""
    program = qwen_lockstep.program
    by_uid = {a.uid: a for a in fws_eval.annotations_of(program)}
    multiply = next(
        a for a in by_uid.values()
        if a.block == "attn_output_gate" and a.phase == "prefill" and a.layer == 3
    )
    op = next(node for node in program.ops if node.uid == multiply.uid)
    upstream = {by_uid[dep].block for dep in op.deps if dep in by_uid}
    # It waits for BOTH operands: the gate projection and the attention output.
    assert "attn_gate_proj" in upstream
    assert "o_proj" in upstream  # the fabric -> macro hop carries that label
    # ... and the o_proj GEMM waits for the multiply.
    o_proj = next(
        node for node in program.ops
        if node.uid in by_uid
        and by_uid[node.uid].block == "o_proj"
        and by_uid[node.uid].kind == "weight_gemm"
        and by_uid[node.uid].phase == "prefill"
        and by_uid[node.uid].layer == 3
    )
    assert multiply.uid in o_proj.deps


# ---------------------------------------------------------------------------
# The delta rule: the first consumer, and it says the law is unchecked
# ---------------------------------------------------------------------------


def test_the_delta_rule_prices_every_linear_attention_layer(qwen_lockstep):
    plan = tuple(qwen_lockstep.mapping.model.layer_mixers)
    linear_layers = [i for i, kinds in enumerate(plan) if "linear_attn" in kinds]
    assert len(linear_layers) == 24
    delta = [
        cost for cost in qwen_lockstep.pricing.costs
        if cost.block == "delta_rule" and cost.phase == "prefill"
    ]
    assert sorted(cost.layer for cost in delta) == linear_layers
    assert all(cost.device_class == "shared_digital" for cost in delta)
    assert all("price_linear_attention_block" in cost.basis for cost in delta)


def test_a_delta_rule_op_matches_its_law_computed_by_hand(qwen_lockstep):
    """HAND CHECK of the delta-rule law on one prefill op.

    Qwen3.5-4B: 16 key heads x d_k 128; 32 value heads x 128 = 4096 value
    channels, so d_v per key head = 4096 / 16 = 256. One state slice is
    d_k * d_v = 32768 elements and the whole state is heads * d_k * d_v =
    524288. m = 4 x 1792 = 7168 tokens, chunk_size = 1 (the recurrent form; no
    config field declares a chunk size, which is disclosed).

    Terms, muls + adds, per token over all heads (per = 524288):
        read_old_value  2 * per      rank1_write    2 * per
        state_decay     1 * per      output_read    2 * per
      -> 7 * per                          = 3670016 per token
        delta_correct   2 * heads*d_v = 8192
        output_gate     1 * heads*d_v = 4096
      -> 12288 per token
    ops = 7168 * (7 * 524288 + 12288) = 26,394,755,072

    The engine retires `lanes` ops per cycle and drains its pipeline once:

        cycles = ceil(ops / lanes) + (pipeline_depth - 1)

    ADJ-9 REWRITE (D31-v2). The width moved twice and the LAW never did.
    OLD-OLD: `lanes = 1024`, DECLARED on the card, 25,776,147 cycles, 27.1328
    ms. OLD (D31-v1): 178 lanes, the smallest that held the analog BEAT,
    148,285,160 cycles, 156.0896 ms. NEW (D31-v2, ADJ-9): the width is derived
    to the ANALOG FLOOR — the smallest integer whose priced time fits the
    ANALOG M-PASS of the stage itself — which on this fixture's decode step is
    48.96 us of analog against 0.95 GHz, and the answer is 7676 lanes:

        cycles = ceil(26,394,755,072 / 7676) + 19 = 3,438,609 + 19
               = 3,438,628
        t      = cycles / 0.95e9 = 3.6196 ms

    The hand check is the same check; only the width it is evaluated at moved,
    and it moved because the TARGET moved from the slowest stage's time to this
    stage's own analog time. Note also WHICH time: this run is the retired
    lockstep fixture, so the engine is sized against its DECODE STEP and the
    PREFILL op below is then priced on decode-sized silicon — which is what the
    machine physically has, and which is why P7 is decode-only (D25) and these
    prefill rows are law checks rather than claims about prefill performance.
    """
    cost = next(
        c for c in qwen_lockstep.pricing.costs
        if c.block == "delta_rule" and c.phase == "prefill" and c.layer == 0
    )
    per = 16 * 128 * 256
    expected_ops = 7168 * (7 * per + 12288)
    assert expected_ops == 26_394_755_072
    assert float(cost.detail["ops"]) == float(expected_ops)
    device = qwen_lockstep.mapping.device
    assert device.vector_lanes == 7676
    assert device.vector_lanes_provenance == "derived-count"
    assert device.vector_clock_hz == 0.95e9
    cycles = math.ceil(expected_ops / 7676) + (device.vector_pipeline_depth - 1)
    assert cycles == 3_438_628
    assert float(cost.detail["arith_cycles"]) == float(cycles)
    assert cost.duration_s == pytest.approx(cycles / 0.95e9, rel=1e-12)
    # The pipeline drain is 19 cycles on 3.4 million: the op is lane-bound,
    # so it also sits within 1e-5 of the pure ops / (lanes * clock) bound.
    assert cost.duration_s == pytest.approx(
        expected_ops / (7676 * 0.95e9), rel=1e-5
    )
    # And 7676 is the SMALLEST width that holds the stage's own ANALOG m-pass:
    # one lane fewer does not fit, which is what "no margins" (D28) means as an
    # assertion, and the row is ANALOG-BOUND, which is what ADJ-9 means as one.
    sizing = device.derived_engine
    assert sizing.vector_lanes == 7676
    row = sizing.per_stage[0]
    assert row.target_kind == cim_timing.TARGET_ANALOG_STAGE
    assert row.used_cycles <= row.budget_cycles
    assert row.time_s <= row.analog_time_s
    assert row.analog_bound
    assert cim_timing.vector_cycles_at(
        [expected_ops], 7676, device.vector_pipeline_depth
    ) == cycles


def test_no_op_exceeds_the_engine_peak_its_law_declares(qwen):
    """Physics before self-consistency: nothing retires work faster than lanes.

    The analog side gets the same treatment through its own peak: a macro
    cannot evaluate more column sets per second than its clock allows.
    """
    device = qwen.mapping.device
    vector_peak = device.vector_lanes * device.vector_clock_hz
    for cost in qwen.pricing.costs:
        ops = float((cost.detail or {}).get("ops", 0.0))
        if ops > 0 and cost.duration_s > 0:
            assert ops / cost.duration_s <= vector_peak * (1 + 1e-9), cost.block
    # The analog side gets the same treatment through its own peak: a macro
    # cannot evaluate column sets faster than f_analog / slice_cycles.
    f_analog = float(device.analog.analog_clock_mhz) * 1e6
    set_peak = f_analog / int(device.analog.slice_cycles)
    by_uid = {a.uid: a for a in fws_eval.annotations_of(qwen.program)}
    for cost in qwen.pricing.costs:
        sets = float((cost.detail or {}).get("total_active_column_sets", 0.0))
        tokens = float(by_uid[cost.uid].tokens) if cost.uid in by_uid else 0.0
        if sets > 0 and cost.duration_s > 0 and tokens > 0:
            assert (sets * tokens) / cost.duration_s <= set_peak * (1 + 1e-9), cost.block


def test_the_unvalidated_law_label_rides_the_report_and_the_atlas(qwen_lockstep):
    """LAW_UNVALIDATED is part of the number, so it travels with the number.

    cim_timing labels every priced digital op; before C1 the label reached the
    op's basis string and stopped there, so a reader of either artifact saw a
    duration with no way to know that NO numeric reference exists for the law
    behind it. It is now a disclosure, which both artifacts carry.
    """
    assert cim_timing.LAW_UNVALIDATED == "unvalidated"
    delta = next(cost for cost in qwen_lockstep.pricing.costs if cost.block == "delta_rule")
    assert cim_timing.LAW_UNVALIDATED in delta.basis

    document = fws_eval.report_document(qwen_lockstep)
    entry = next(
        item for item in document["disclosures"]
        if item["constraint"] == "unvalidated_law:delta_rule_recurrent"
    )
    assert "UNVALIDATED" in entry["value"]

    with open(ATLAS_JSON, encoding="utf-8") as handle:
        atlas = json.load(handle)
    assert [
        item for item in atlas["relaxations"]
        if item["constraint"] == "unvalidated_law:delta_rule_recurrent"
    ]


def test_the_undeclared_delta_chunk_size_is_disclosed_and_not_invented(qwen):
    """Qwen3.5 publishes no chunk size, so P4 invents none and says so.

    AUDIT FIX (C1 finding 1). The disclosure used to call the recurrent form
    "the conservative direction". It is not established as such, and this test
    is the reason: on this card the chunked form's saving is state traffic,
    which is priced at ZERO, while its retired-op count EXCEEDS the recurrent
    one above Q ~ 34. A disclosure may not claim a direction the model cannot
    support, so the word is banned here rather than merely absent.
    """
    assert qwen.mapping.model.linear_attention.chunk_size is None
    entry = next(
        item for item in fws_eval.report_document(qwen)["disclosures"]
        if item["constraint"] == "delta_rule_chunk_size"
    )
    assert "recurrent" in entry["reason"].lower()
    assert "conservative" not in entry["reason"].lower()
    assert "upper bound" not in entry["reason"].lower()
    assert "not established" in entry["reason"].lower()


def test_the_chunked_delta_rule_is_not_cheaper_on_this_card(qwen):
    """The measurement behind the disclosure, not a restatement of it.

    Two independent facts, both asserted against the real laws:
    (1) state traffic is priced at ZERO on this card, so the chunked form's
        one saving buys nothing -- ``state_time_s`` is 0.0 and the duration is
        entirely arithmetic; and
    (2) in retired OPS the chunked form crosses the recurrent one at Q ~ 34,
        so it is DEARER at the chunk sizes reference gated-DeltaNet kernels
        actually use (64, 128).
    """
    device = qwen.mapping.device
    linear = qwen.mapping.model.linear_attention
    tokens = 7168.0

    recurrent = device.price_linear_attention_block(linear, tokens, chunk_size=1)
    assert recurrent.state_time_s == 0.0
    assert recurrent.time_s == recurrent.arith_time_s
    assert any("state_bytes_per_cycle undeclared" in d for d in recurrent.disclosures)

    base = recurrent.work.ops
    ratios = {
        q: device.price_linear_attention_block(linear, tokens, chunk_size=q).work.ops / base
        for q in (8, 32, 64, 128)
    }
    assert ratios[8] < 1.0 and ratios[32] < 1.0  # cheaper only for small Q
    assert ratios[64] > 1.10 and ratios[128] > 1.35  # dearer where kernels chunk
    assert ratios[64] < ratios[128]


def test_a_declared_delta_chunk_size_is_honored_and_relabels_the_disclosure():
    """The durable fix: the schema can now SAY the Q instead of leaving it out.

    Declaring ``model_param.linear_attention.chunk_size`` moves the prefill
    delta-rule ops to the chunked law and rewrites the disclosure to name the
    declared value. Decode stays recurrent -- one token is not a chunk.

    WAVE F: the question is about PREFILL, and a D29 mapped run lowers none
    (the streams arrive prefilled -- D25/D29), so the run declares the RETIRED
    regime by name. The chunked law itself is untouched by the pivot.
    """
    model = _model(
        QWEN_MODEL,
        lambda raw: raw["model_param"]["linear_attention"].update({"chunk_size": 64}),
    )
    assert model.model_config.linear_attention.chunk_size == 64
    evaluation = fws_eval.evaluate_fws(
        build_fws_program(
            fws_mapping.build_mapping(
                _hw(QWEN_HW), model, regime=fws_mapping.REGIME_LOCKSTEP
            )
        )
    )
    document = fws_eval.report_document(evaluation)
    entry = next(
        item for item in document["disclosures"]
        if item["constraint"] == "delta_rule_chunk_size"
    )
    assert entry["value"].startswith("64 (DECLARED)")
    assert "not established" not in entry["reason"].lower()
    # The chunked law is a DIFFERENT law with its own unvalidated label, so the
    # declaration is visible in the artifact and not only in one disclosure.
    constraints = {item["constraint"] for item in document["disclosures"]}
    assert "unvalidated_law:delta_rule_chunked" in constraints


def test_a_declared_delta_chunk_size_below_one_is_refused_by_name():
    with pytest.raises(ValueError, match=r"linear_attention\.chunk_size"):
        _model(
            QWEN_MODEL,
            lambda raw: raw["model_param"]["linear_attention"].update({"chunk_size": 0}),
        )


# ---------------------------------------------------------------------------
# The finding, and the shipped artifacts
# ---------------------------------------------------------------------------


def test_the_prefill_is_shared_digital_bound_and_the_split_is_the_finding(qwen_lockstep):
    """P1's finding #1 (D5), restated for a 3:1 linear:full stack.

    Qwen3.5-4B's prefill is owned by the shared digital chiplet twice over: the
    24 delta-rule ops and the 8 layers of full attention together are ~80% of
    the prefill critical path, and ONE gated-attention layer costs many times
    what ONE delta-rule layer costs at this context.
    """
    prefill = [cost for cost in qwen_lockstep.pricing.costs if cost.phase == "prefill"]
    latency = qwen_lockstep.metric("sys.fws.prefill_latency").value
    delta = sum(c.duration_s for c in prefill if c.block == "delta_rule")
    attention = sum(
        c.duration_s for c in prefill
        if c.block in ("attention_qk", "attention_pv", "attention_softmax")
    )
    analog = sum(c.duration_s for c in prefill if c.device_class == "analog_macro")

    # ADJ-10 REWRITE. This split has now been read at FOUR provisionings and it
    # says something different at each, which is the point of writing the
    # provisioning down beside it.
    #   1024 DECLARED lanes, 2 DECLARED arrays: delta 50-60%, attention 30-40%.
    #   178 lanes (D31-v1, scan sized to the analog BEAT), 2 arrays: delta
    #     80-95%, attention 5-20%, one attention layer CHEAPER than one delta.
    #   7676 lanes (D31-v2/ADJ-9, scan sized to the ANALOG FLOOR), 2 arrays:
    #     delta ~13%, ATTENTION ~67%, one attention layer ~15x one delta layer.
    #   7676 lanes AND 32 DERIVED arrays / 64 DERIVED softmax lanes (ADJ-10):
    #     the numbers below.
    # OLD CLAIM (ADJ-9's): this stack is bound by the attention systolic
    # fabric, whose geometry is DECLARED and which nothing derives.
    # NEW CLAIM: the fabric is derived too — 16 concurrent folds per side on
    # this batch — and the picture INVERTS. Attention falls from ~67% to ~13%
    # of the summed prefill work, delta rule RISES to ~34% (it is unchanged in
    # absolute terms; the denominator shrank), and one attention layer now
    # costs only ~1.1x one delta-rule layer instead of ~15x. The two digital
    # blocks together are under half the work and the ANALOG macros are the
    # largest term by far.
    #
    # The shares are sums of OP DURATIONS over parallel device classes, so they
    # are shares of WORK and not of one serial path; they do not sum to 1, they
    # can exceed 1 (hundreds of macros run concurrently), and neither is
    # claimed otherwise.
    assert 0.28 < delta / latency < 0.40
    assert 0.10 < attention / latency < 0.17
    assert 0.40 < (delta + attention) / latency < 0.55
    assert 1.4 < analog / latency < 1.9
    # per-layer: 24 linear layers vs 8 attention layers — and with the fabric
    # derived they are now within ~13% of each other, which is the reversal.
    assert 1.0 < (attention / 8) / (delta / 24) < 1.4


def test_the_shipped_qwen_report_is_what_a_fresh_run_produces(qwen):
    """Regenerate and compare BYTES, the Granite artifacts' own gate.

    The report is a pure function of the two YAMLs -- no clock, no path, no
    ordering by hash -- so byte equality is the right bar. A law change that
    moves a Qwen number now has to move this file too, in the same commit.
    """
    assert os.path.exists(REPORT_JSON), (
        "docs/qif/atlas/qwen3_5_4b_report.json is committed; a missing artifact "
        "must fail this gate, never skip it"
    )
    shipped = open(REPORT_JSON, encoding="utf-8").read()
    assert shipped == json.dumps(fws_eval.report_document(qwen), indent=2)


def test_the_shipped_qwen_atlas_is_what_a_fresh_run_produces(tmp_path):
    import tools.fws_emit_atlas as emit

    assert os.path.exists(ATLAS_JSON), (
        "docs/qif/atlas/qwen3_5_4b.json is committed; a missing artifact must "
        "fail this gate, never skip it"
    )
    _mapping, _rows, document = emit.build(QWEN_HW, QWEN_MODEL, priced=True)
    regenerated = fws_atlas_export.write_atlas_json(
        document, os.path.join(str(tmp_path), "qwen.json")
    )
    assert regenerated == open(ATLAS_JSON, encoding="utf-8").read(), (
        "docs/qif/atlas/qwen3_5_4b.json is stale; regenerate it with the "
        "command in its provenance block"
    )


def test_the_shipped_qwen_atlas_validates_through_the_real_loader(tmp_path):
    """Judged by atlas.html's own ATLAS.validate (P5.3), not by a local schema."""
    from tests.test_qif_atlas import _run_core

    assert os.path.exists(ATLAS_JSON)
    out = _run_core(
        """
        const v = ATLAS.validate(doc);
        console.log(JSON.stringify({
          errors: v.errors.map(e => e.rule + ' ' + e.field),
          warnings: v.warnings.map(w => w.rule + ' ' + w.field),
        }));
        """,
        tmp_path,
        document=pathlib.Path(ATLAS_JSON),
    )
    verdict = json.loads(out)
    assert verdict["errors"] == []
    assert verdict["warnings"] == []


def test_the_qwen_artifacts_name_the_model_and_not_its_pricing_carrier():
    """D21's named owner has to be a name a reader can TRUST.

    qwen3_5_4b_inf.yaml declares model_type: llama because the SwiGLU gated MLP
    is the arithmetic carrier. That is a pricing choice, not the model's
    identity, and every owner, label and title names the model.
    """
    with open(ATLAS_JSON, encoding="utf-8") as handle:
        atlas = json.load(handle)
    assert "qwen3_5_4b" in atlas["title"]
    system = atlas["systems"][0]
    assert system["models"][0]["id"] == "qwen3_5_4b"
    assert atlas["tiles"]
    assert all(tile["owner"]["model"] == "qwen3_5_4b" for tile in atlas["tiles"])
    carrier = [
        item for item in atlas["relaxations"] if item["constraint"] == "model_type_carrier"
    ]
    assert carrier and "llama" in carrier[0]["value"]

    with open(REPORT_JSON, encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["model"]["model_id"] == "qwen3_5_4b"
    assert report["model"]["model_type"] == "llama"


def test_the_gate_reaches_the_atlas_as_a_drawn_tile():
    """An artifact someone opens has to show the thing the wave added."""
    with open(ATLAS_JSON, encoding="utf-8") as handle:
        atlas = json.load(handle)
    gate_tiles = [t for t in atlas["tiles"] if t["owner"]["block"] == "attn_gate_proj"]
    # ONE W_g PER GATED LAYER, however the allocation tiles it. The count of
    # TILES is a property of the allocation granularity -- this machine ships
    # bank_depth 1, so a matrix is cut at bank granularity and spans more,
    # smaller tiles than the dedicated allocation did -- so what is pinned is
    # the thing the wave added: every gated layer has a gate on the picture,
    # and no ungated layer does.
    gated = {t["owner"]["layer"] for t in gate_tiles}
    assert len(gated) == 8
    assert gate_tiles and len(gate_tiles) % len(gated) == 0
    assert "attn_gate_proj" in atlas["systems"][0]["models"][0]["blocks"]


def test_accuracy_is_never_a_factor_in_the_qwen_artifacts():
    """D23, asserted on the artifacts a reader actually receives."""
    for path in (ATLAS_JSON, REPORT_JSON):
        assert "accuracy" not in open(path, encoding="utf-8").read().lower()
