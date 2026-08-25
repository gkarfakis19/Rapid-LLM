"""QIF P7.3: the dense-packing MAPPING MODE, and utilization as an output.

P7.2 built the packer; this module is about what a RUN does with it. Four
things are pinned here and every one of them is a wrong number rather than a
crash if it rots:

  1. ``mapping.packing`` — the placement law a config DECLARES. The default is
     today's dedicated placement, so nothing shipped moves; dense is opt-in.
  2. Co-residency. Dense packing puts several tensors, and several LAYERS, in
     one macro, and what that costs is MEASURED here, not asserted — including
     what SAME-LAYER co-residency does cost. The measurement is taken on the
     LOCKSTEP machine (see the regime note below), where the layers run in
     sequence; D29's filled pipeline splits the same quantity into a
     within-stage half and a cross-stage half, and tests/test_qif_pipeline.py
     owns both claims, including a machine where the cross-stage term is
     positive.
  3. The accumulator ops. A K stack that lives in one macro reduces on that
     macro's pool, priced by P7.2's law from the packer's own descriptor.
  4. Utilization. Per device class, on every run, idle devices inside the
     average (D28: idle silicon is a finding, never a footnote).

THE GEOMETRIES the hand arithmetic below is done on:

  * configs/hardware-config/fws_cim_moe.yaml — rows = 1024, cols_adc = 256,
    adc_mux = 4. A macro stores 1024 x 1024 cells; at ``bank_depth: 1`` a bank
    is 256 columns wide and a macro holds 4 of them.
  * the same file with ``cols_adc: 512`` — a macro stores 1024 x 2048, a bank
    is 512 wide, and a tensor narrower than a macro's stored width can now
    leave two of its K blocks in ONE macro under the dedicated placement,
    which is the case that has no accumulator until the packer describes it.
  * configs/hardware-config/fws_cim_granite_tiny_dse_selected.yaml — the Wave D
    winner (rows = 1536, cols_adc = 384, adc_mux = 4, bank_depth = 1), which is
    the shipped config where Invariant W actually changes the macro count.
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path

import pytest
import yaml

import config
import fws_eval
import fws_mapping
from cim_timing import DeadFoldError
from fws_mapping import MappingError
from program.fws_build import LAW_K_ACCUMULATION, annotations_of, build_fws_program

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HW_DIR = PROJECT_ROOT / "configs" / "hardware-config"
MODEL_DIR = PROJECT_ROOT / "configs" / "model-config"

FWS_MOE = HW_DIR / "fws_cim_moe.yaml"
FWS_GRANITE_DSE = HW_DIR / "fws_cim_granite_tiny_dse_selected.yaml"
MOE_SMALL = MODEL_DIR / "moe_small_fws_inf.yaml"
GRANITE = MODEL_DIR / "granite_4_0_h_tiny_inf.yaml"

#: The P7.3 artifact: Granite-4.0-H-Tiny decode under both placement laws.
PACKING_ARTIFACT = PROJECT_ROOT / "docs" / "qif" / "folding" / "granite_dense_vs_dedicated.json"

#: WAVE F (D29/D30). A DECLARED `mapping:` block is now a FILLED PIPELINE, and
#: the filled pipeline refuses a declared batch and declared endpoints by name.
#:
#: EVERY fixture in this file is DELIBERATELY LOCKSTEP, and the reason is the
#: subject: this module is about WHICH WEIGHTS LAND IN WHICH BANK and what that
#: placement costs, which is a question about the packer and not about the
#: serving regime — the same packing produces the same tiles, the same banks
#: and the same accumulators under either one. Keeping the fixtures on the
#: batch-4 lockstep workload is what keeps this file's hand arithmetic (445
#: macros, 1776 bank passes, 684 accumulator ops) checkable against the numbers
#: it was computed from. The regime-DEPENDENT claims — the beat, D, the state
#: bill, and the within-stage / cross-stage split of bank sharing — belong to
#: tests/test_qif_pipeline.py and are made there on real D29 machines.
#:
#: A P7.9 correction: this note previously said the RUN tests "run the D29
#: machine through `_d29_model`". They never did — the helper it named was
#: never called from anywhere, and the fixtures below build from
#: configs/hardware-config/fws_cim_moe.yaml, which declares no `mapping:`
#: block and therefore lowers under lockstep. The helper is gone and the note
#: now describes what the file actually runs.
LOCKSTEP = {"regime": "lockstep"}


def _hw(path, mutate=None):
    raw = copy.deepcopy(yaml.safe_load(Path(path).read_text()))
    if mutate is not None:
        mutate(raw)
    config.convert(raw)
    return config.HWConfig.from_dict(raw)


def _fine_bank(depth=1, cols_adc=None):
    """Mutator: a card that allocates in single mux slots (and, optionally, a
    wider ADC so a tensor fits inside one macro's stored width)."""

    def mutate(raw):
        if cols_adc is not None:
            raw["cim"]["analog"]["cols_adc"] = cols_adc
        raw["cim"]["cards"] = {
            "ctt": {"kind": "analog_macro", "device": "ctt", "bank_depth": depth}
        }

    return mutate


def _mapping(hw_path, model_path, mutate=None, **kwargs):
    hw = _hw(hw_path, mutate)
    model = config.parse_config(str(model_path), "LLM")
    return fws_mapping.build_mapping(hw, model, **kwargs)


def _run(hw_path, model_path, mutate=None, **kwargs):
    mapping = _mapping(hw_path, model_path, mutate, **kwargs)
    return fws_eval.evaluate_fws(build_fws_program(mapping), mapping)


@pytest.fixture(scope="module")
def moe_dense():
    """MoE-small on a bank_depth = 1 card, packed densely. 445 macros."""
    return _run(FWS_MOE, MOE_SMALL, _fine_bank(1), packing=fws_mapping.PACKING_DENSE)


@pytest.fixture(scope="module")
def moe_dedicated():
    return _run(FWS_MOE, MOE_SMALL, _fine_bank(1))


@pytest.fixture(scope="module")
def stacked_dense():
    """The geometry where a K stack fits INSIDE one macro (cols_adc = 512)."""
    return _run(
        FWS_MOE, MOE_SMALL, _fine_bank(1, cols_adc=512), packing=fws_mapping.PACKING_DENSE
    )


@pytest.fixture(scope="module")
def stacked_dedicated():
    return _run(FWS_MOE, MOE_SMALL, _fine_bank(1, cols_adc=512))


@pytest.fixture(scope="module")
def granite_packing_document():
    """The P7.3 artifact, REBUILT: Granite decode under both laws."""
    evaluations = {}
    for packing in (fws_mapping.PACKING_DEDICATED, fws_mapping.PACKING_DENSE):
        mapping = _mapping(FWS_GRANITE_DSE, GRANITE, packing=packing)
        evaluations[packing] = fws_eval.evaluate_fws(
            build_fws_program(mapping), mapping
        )
    return fws_eval.packing_comparison_document(
        evaluations[fws_mapping.PACKING_DEDICATED],
        evaluations[fws_mapping.PACKING_DENSE],
        hardware_config="configs/hardware-config/fws_cim_granite_tiny_dse_selected.yaml",
        model_config="configs/model-config/granite_4_0_h_tiny_inf.yaml",
    )


# ---------------------------------------------------------------------------
# 1. mapping.packing — the law a config declares (P7.3 item 1)
# ---------------------------------------------------------------------------


def test_the_default_packing_is_todays_placement():
    # The whole point of "opt-in this wave": a config that says nothing about
    # packing gets the placement it got yesterday, byte for byte.
    spec = config.MappingSystemConfig.from_dict({}, "mapping")
    assert spec.packing == fws_mapping.PACKING_DEDICATED
    mapping = _mapping(FWS_MOE, MOE_SMALL, _fine_bank(1))
    assert mapping.packing == fws_mapping.PACKING_DEDICATED
    assert mapping.packings == {}


def test_a_declared_dense_packing_reaches_the_mapper():
    # The config surface is the seam P7.2 left for this: `mapping.packing`
    # names the law and build_mapping is what runs it.
    def declare_dense(raw):
        _fine_bank(1)(raw)
        raw.setdefault("mapping", {}).update(LOCKSTEP, packing="dense")

    mapping = _mapping(FWS_MOE, MOE_SMALL, declare_dense)
    assert mapping.packing == fws_mapping.PACKING_DENSE
    assert mapping.packings, "a dense mapping carries its per-chip packings"
    assert mapping.packing_summary()["walk"] == "k_inner"


def test_the_keyword_overrides_the_declared_law():
    # ONE law reaches the packer, and which one is a sentence, not a
    # precedence table: the config declares it, an explicit keyword wins.
    def declare_dense(raw):
        _fine_bank(1)(raw)
        raw.setdefault("mapping", {}).update(LOCKSTEP, packing="dense")

    mapping = _mapping(
        FWS_MOE, MOE_SMALL, declare_dense, packing=fws_mapping.PACKING_DEDICATED
    )
    assert mapping.packing == fws_mapping.PACKING_DEDICATED
    assert mapping.packing_summary() == {}


def test_an_unknown_packing_is_refused_by_name():
    with pytest.raises(ValueError, match="not a placement law this mapper has"):
        config.MappingSystemConfig.from_dict({"packing": "sparse"}, "mapping")
    with pytest.raises(MappingError, match="not a placement law this mapper has"):
        _mapping(FWS_MOE, MOE_SMALL, _fine_bank(1), packing="sparse")


def test_the_config_surface_and_the_mapper_cannot_drift():
    # Two spellings of one list is two accountings of one fact (D21). The
    # config parser imports the mapper's tuple; this pins that it still does.
    assert config._parse_packing("dense", "mapping") == fws_mapping.PACKING_DENSE
    assert config._parse_packing(None, "mapping") == fws_mapping.PACKING_DEDICATED
    for mode in fws_mapping.PACKING_MODES:
        assert config._parse_packing(mode, "mapping") == mode


def test_a_declared_dense_prefill_is_refused_by_name():
    # D25 / D26: prefill folding is on the DOA register and the refusal IS the
    # implementation. Reaching it through the CONFIG must hit the same wall.
    with pytest.raises(DeadFoldError, match="prefill_folding"):
        _mapping(
            FWS_MOE,
            MOE_SMALL,
            _fine_bank(1),
            packing=fws_mapping.PACKING_DENSE,
            phase="prefill",
        )


def test_a_pd_half_declares_its_own_law_and_the_prefill_half_is_refused():
    # D16 is two inventories; each declares its own placement. A decode half
    # may pack densely (Llama2-7B at bank_depth 1: 408 macros against 424, and
    # only 3.477% of the weight space wasted) while the prefill half keeps
    # today's placement — and asking the PREFILL half for dense packing hits
    # the DOA register, through the config, by name.
    raw = copy.deepcopy(yaml.safe_load((HW_DIR / "fws_cim_llama7b_mapped.yaml").read_text()))
    raw["cim"]["cards"] = {"ctt": {"kind": "analog_macro", "device": "ctt", "bank_depth": 1}}
    raw["mapping"] = {
        "pd": {
            # WAVE F: the halves declare LOCKSTEP by name. The question here is
            # which PLACEMENT LAW each half runs under, which the regime does
            # not change, and the hand numbers below (408 macros against 424)
            # are the batch-4 Llama2-7B placement they were computed on.
            "prefill": dict(LOCKSTEP, layers_per_chip=8, shared_chiplets=2),
            "decode": dict(
                LOCKSTEP,
                layers_per_chip=4,
                shared_chiplets=1,
                decode_window=2,
                packing="dense",
            ),
        }
    }
    config.convert(raw)
    hw = config.HWConfig.from_dict(raw)
    model = config.parse_config(str(MODEL_DIR / "llama2_7b_fws_inf.yaml"), "LLM")
    pair = fws_mapping.build_pd_pair(hw, model)
    assert pair.prefill.packing == fws_mapping.PACKING_DEDICATED
    assert pair.decode.packing == fws_mapping.PACKING_DENSE
    summary = pair.decode.packing_summary()
    assert (summary["macros"], summary["dedicated_macros"]) == (408, 424)
    assert summary["waste_pct"] == pytest.approx(3.477328431372549)

    raw["mapping"]["pd"]["prefill"]["packing"] = "dense"
    with pytest.raises(DeadFoldError, match="prefill_folding"):
        fws_mapping.build_pd_pair(config.HWConfig.from_dict(raw), model)


def test_an_emitter_override_carries_the_declared_law(tmp_path):
    # tools/fws_emit_atlas.py rebuilds the spec field by field when a flag
    # overrides one of them. A rebuild that forgot the packing would quietly
    # draw the DEDICATED machine under a dense config's title, which is a
    # wrong picture rather than an error.
    import tools.fws_emit_atlas as emit

    spec = config.MappingSystemConfig.from_dict({"packing": "dense"}, "mapping")
    assert emit._override_spec(spec, layers_per_chip=4).packing == "dense"
    assert emit._override_spec(spec, shared_chiplets=1).packing == "dense"
    assert emit._override_spec(spec).packing == "dense"


def test_the_mapping_summary_names_the_law_it_placed_under():
    # Two mappings of one model can differ ONLY by this word, so a summary
    # that omitted it would make them look identical.
    dedicated = _mapping(FWS_MOE, MOE_SMALL, _fine_bank(1))
    dense = _mapping(FWS_MOE, MOE_SMALL, _fine_bank(1), packing=fws_mapping.PACKING_DENSE)
    assert dedicated.summary()["packing"] == "dedicated"
    assert dense.summary()["packing"] == "dense"
    assert dense.summary()["macros_holding_tiles"] < dedicated.summary()["macros_holding_tiles"]


# ---------------------------------------------------------------------------
# 2. Cross-tensor and cross-layer bank sharing (P7.3 item 1, Fact 3)
# ---------------------------------------------------------------------------


def test_dense_packing_actually_shares_banks_across_tensors_and_layers(moe_dense):
    # Without this the contention test below would prove nothing: it would be
    # measuring a machine where no macro has a co-resident.
    mapping = moe_dense.mapping
    cross_tensor = [
        macro
        for macro in mapping.macros
        if macro.tiles and len({tile.owner for tile in macro.tiles}) > 1
    ]
    cross_layer = [
        macro
        for macro in mapping.macros
        if macro.tiles and len({tile.owner.layer for tile in macro.tiles}) > 1
    ]
    assert len(cross_tensor) == 328
    assert len(cross_layer) == 8
    # ... and the sharing is real weights on both sides, never a padded bank.
    for macro in cross_layer:
        assert all(tile.logical_columns > 0 for tile in macro.tiles)
    # The report says the same thing, from the same count (D21).
    assert moe_dense.bank_sharing["macros_sharing_banks_across_tensors"] == 328
    assert moe_dense.bank_sharing["macros_sharing_banks_across_layers"] == 8


def test_cross_layer_bank_sharing_costs_exactly_zero_on_a_lockstep_decode_step(
    moe_dense,
):
    # MEASURED, ON THE LOCKSTEP MACHINE. Every op of the lowered decode step
    # that started later than it was ready has its wait attributed to whatever
    # else held its device. Not one picosecond of that wait belongs to a
    # resident of another layer, because THIS regime runs the layers in
    # sequence.
    #
    # P7.9 RETITLE: this used to open "FACT 3, MEASURED". P7_folding.html marks
    # Fact 3 [RETIRED] — D29's filled pipeline fires every stage every beat, so
    # cross-layer sharing is free only WITHIN a stage. The measurement below is
    # unchanged and still true of the machine it is taken on; only the claim it
    # is offered as has narrowed, from a law about decode to a fact about the
    # lockstep regime. The live D29 pair lives in tests/test_qif_pipeline.py.
    sharing = moe_dense.bank_sharing
    assert sharing["measured"] is True
    assert sharing["cross_layer_delay_s"] == 0.0
    # POSITIVE CONTROL. The instrument is not blind: on the same step it does
    # see a tensor's own blocks waiting for one another (the macro walking its
    # banks), so the zero above is a measurement and not an empty sum.
    assert sharing["delayed_ops"] > 0
    assert sharing["same_owner_delay_s"] > 0.0


def test_same_layer_co_residency_is_priced_and_disclosed_not_hidden(moe_dense):
    # Two tensors of ONE layer can be concurrent (a routed expert beside a
    # shared expert), and the packer can land them in one macro. That DOES
    # serialize, the timeline already prices it, and the run says so by name.
    sharing = moe_dense.bank_sharing
    assert sharing["cross_tensor_same_layer_delay_s"] > 0.0
    banners = [
        item
        for item in moe_dense.disclosures
        if item.constraint == "same_layer_bank_sharing_serializes"
    ]
    assert len(banners) == 1
    assert "CROSS-LAYER" in banners[0].reason


def test_the_dedicated_placement_shares_no_banks_at_all(moe_dedicated):
    # The control: under the per-tensor law a macro has ONE owner, so the
    # sharing question does not arise and the block reports zeros rather than
    # a number with no co-residency behind it.
    assert moe_dedicated.bank_sharing["macros_sharing_banks_across_tensors"] == 0
    assert moe_dedicated.bank_sharing["cross_layer_delay_s"] == 0.0
    assert moe_dedicated.bank_sharing["cross_tensor_same_layer_delay_s"] == 0.0


# ---------------------------------------------------------------------------
# 3. Every bank pass, priced once per decode step (P7.3 item 2, Fact 1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["moe_dense", "moe_dedicated"])
def test_each_macro_walks_its_occupied_banks_once_per_decode_step(fixture, request):
    evaluation = request.getfixturevalue(fixture)
    passes = evaluation.bank_passes
    assert passes["measured"] is True
    assert passes["macros_walked_exactly_once"] == passes["macros_holding_tiles"]
    assert passes["column_sets_never_read"] == 0.0
    assert passes["column_sets_read_more_than_once"] == 0.0
    # The charge is the placement's own column-set count — no more, no less.
    occupied = sum(
        macro.claimed_column_sets for macro in evaluation.mapping.macros if macro.tiles
    )
    assert passes["column_set_passes_charged"] == float(occupied)
    assert passes["column_sets_occupied"] == float(occupied)


def test_dense_packing_charges_the_same_bank_passes_on_fewer_macros(
    moe_dense, moe_dedicated
):
    # THE INVARIANT W STATEMENT, on a timeline. The same weights are read the
    # same number of times per token; what changes is how many macros hold
    # them. 1776 banks is the model, 445 vs 459 macros is the packing.
    assert moe_dense.bank_passes["column_set_passes_charged"] == 1776.0
    assert moe_dedicated.bank_passes["column_set_passes_charged"] == 1776.0
    assert moe_dense.bank_passes["macros_holding_tiles"] == 445
    assert moe_dedicated.bank_passes["macros_holding_tiles"] == 459
    assert moe_dense.mapping.packing_summary()["banks_used"] == 1776


def test_one_analog_op_holds_tiles_on_exactly_one_macro(moe_dense):
    # The bank-pass reading takes each op's charge as ONE macro's column sets.
    # That is what the builder emits; if it ever stopped being true the
    # accounting above would be wrong rather than approximate.
    for annotation in annotations_of(moe_dense.program):
        if annotation.tiles and annotation.kind == "weight_gemm":
            assert len({tile.site.macro_id for tile in annotation.tiles}) == 1


# ---------------------------------------------------------------------------
# 4. The accumulator ops the lowered DAG carries (P7.3 item 2)
# ---------------------------------------------------------------------------


def _accumulations(evaluation):
    annotations = annotations_of(evaluation.program)
    return [a for a in annotations if a.law == LAW_K_ACCUMULATION]


def test_every_local_k_stack_gets_one_accumulator_op_per_pass(stacked_dense):
    # The geometry (cols_adc = 512, bank_depth = 1): ffn2 is K x N =
    # 2048 x 1024 on rows = 1024 and a 512-wide bank, so it is 2 K blocks x 2
    # output blocks, and the K-INNER stream lands both K blocks of an output
    # block in ONE macro. That is the local stack, and the packer describes it.
    mapping = stacked_dense.mapping
    summary = mapping.packing_summary()
    assert summary["local_k_stacks"] == 1
    ops = _accumulations(stacked_dense)
    # One op per stack per lowered pass: the prefill plus the decode window.
    passes = 1 + stacked_dense.serving.decode_steps
    assert len(ops) == summary["local_k_stacks"] * passes
    for annotation in ops:
        assert annotation.device_class == "macro_pool"
        assert annotation.k_stack is not None
        assert annotation.k_stack.local and annotation.k_stack.depth > 1
        # ON THE OWNING MACRO'S POOL: the partials never leave the macro that
        # produced them, which is the whole point of the K-inner walk.
        assert annotation.macro_id == annotation.k_stack.sink_macro_id
        assert annotation.device_id == mapping.macro(annotation.macro_id).pool_device


def test_the_accumulator_op_is_priced_by_hand(stacked_dense):
    # HAND-COMPUTED. The stack is depth 2, 512 columns wide, and the decode
    # step carries batch = 4 tokens, so the pass emits 4 x 512 = 2048 results.
    # P2.5 sizes this macro's pool at 27 lanes on a 0.95 GHz clock, so the
    # final drain is ceil(2048 / 27) = 76 cycles = 76 / 0.95e9 = 8.0e-8 s.
    # Every earlier add hides under the next ADC pass; depth 2 has none.
    costs = {cost.uid: cost for cost in stacked_dense.pricing.costs}
    decode_ops = [
        a for a in _accumulations(stacked_dense) if a.phase == "decode" and a.step == 0
    ]
    assert decode_ops
    for annotation in decode_ops:
        cost = costs[annotation.uid]
        assert cost.detail["depth"] == 2.0
        assert cost.detail["width"] == 512.0
        assert cost.detail["drain_cycles"] == 76.0
        assert cost.detail["hidden_adds"] == 0.0
        assert cost.detail["stall_s"] == 0.0
        assert cost.duration_s == pytest.approx(76 / 0.95e9, rel=1e-12)
        # (depth - 1) adds per output result: 2048 x 1.
        assert cost.detail["partial_adds"] == 2048.0


def test_a_spread_k_stack_is_never_given_an_accumulator_op(stacked_dense):
    # A stack whose blocks sit on different macros is priced END TO END by the
    # row-block partial-sum law the builder already emits. A second op for it
    # would be a second accounting of one metric (D21). This packing has one
    # of each, so the two shapes are told apart on ONE timeline.
    summary = stacked_dense.mapping.packing_summary()
    assert summary["local_k_stacks"] == 1 and summary["spread_k_stacks"] == 1
    spread = [
        stack
        for packing in stacked_dense.mapping.packings.values()
        for stack in packing.spread_k_stacks
    ]
    assert len(spread) == 1
    accumulated = {
        (a.k_stack.owner, a.k_stack.n_start) for a in _accumulations(stacked_dense)
    }
    assert (spread[0].owner, spread[0].n_start) not in accumulated
    # ... and the shape that DOES price it is on the same timeline: a row-block
    # partial sum on the sink macro's pool.
    rowsums = [
        a
        for a in annotations_of(stacked_dense.program)
        if a.kind == "reduction" and "row-block partials summed" in a.note
    ]
    assert rowsums


def test_the_pricer_refuses_an_accumulation_it_cannot_price(stacked_dense):
    # Belt and braces on the seam: an accumulation op carrying no descriptor,
    # or carrying a SPREAD stack, is a contradiction and not a zero.
    from dataclasses import replace as _replace

    pricer = fws_eval._Pricer(stacked_dense.mapping, stacked_dense.serving)
    annotation = _accumulations(stacked_dense)[0]
    with pytest.raises(MappingError, match="carrying no descriptor"):
        pricer.price(_replace(annotation, k_stack=None))
    spread = _replace(
        annotation.k_stack,
        macro_ids=(annotation.k_stack.sink_macro_id, annotation.k_stack.sink_macro_id + 1),
    )
    with pytest.raises(MappingError, match="SPANS macros"):
        pricer.price(_replace(annotation, k_stack=spread))


def test_a_local_stack_the_packer_never_described_is_disclosed(stacked_dedicated):
    # The dedicated placement can build a local K stack too (same geometry,
    # per-tensor placement), and nothing describes it, so nothing prices it.
    # The gap is a BANNER rather than a silent zero, and the same run under
    # dense packing is where the adds get charged.
    assert stacked_dedicated.program.meta.misc["fws_undescribed_local_k_stacks"] == 2
    banners = [
        item
        for item in stacked_dedicated.disclosures
        if item.constraint == "unpriced_local_k_stack"
    ]
    assert len(banners) == 1
    assert "mapping.packing: dense" in banners[0].reason
    assert _accumulations(stacked_dedicated) == []


def test_the_accumulator_energy_is_its_own_itemized_term(stacked_dense):
    # One component, one coverage label (ADJ-6). No card declares a per-add
    # energy, so the term is UNCOVERED and prints 0 rather than an invented
    # figure — and the TIME above is fully priced regardless.
    components = {item.key: item for item in stacked_dense.energy}
    assert "digital_accumulation" in components
    assert components["digital_accumulation"].coverage == fws_eval.COVERAGE_UNCOVERED
    assert components["digital_accumulation"].energy_pj == 0.0
    assert any(
        item.constraint == "accumulator_energy" for item in stacked_dense.disclosures
    )


# ---------------------------------------------------------------------------
# 5. Utilization as a required output (P7.3 item 3, D28)
# ---------------------------------------------------------------------------


def test_every_report_carries_per_device_class_utilization(moe_dense):
    document = fws_eval.report_document(moe_dense)
    rows = document["evaluation"]["utilization"]
    classes = [row["device_class"] for row in rows]
    assert classes == ["analog_macro", "macro_pool", "shared_digital", "link"]
    for row in rows:
        assert set(row) >= {
            "devices",
            "devices_used",
            "idle_devices",
            "idle_share",
            "ops",
            "busy_s",
            "mean_occupancy",
            "median_occupancy",
            "max_occupancy",
            "min_occupancy",
            "basis",
        }
    assert document["evaluation"]["utilization_basis"]


def test_utilization_averages_over_idle_silicon_too(moe_dense):
    # THE NUMBER THAT HIDES THE FINDING is the mean over the busy devices
    # only. This is the mean over every device of the class, so provisioning
    # 600 macros and using 459 of them shows up as a smaller number.
    row = moe_dense.utilization_of("analog_macro")
    devices = [
        item for item in moe_dense.occupancy if item.device_class == "analog_macro"
    ]
    assert row.devices == len(devices) == 600
    assert row.idle_devices == sum(1 for item in devices if not item.ops)
    assert row.idle_devices > 0
    assert row.mean_occupancy == pytest.approx(
        math.fsum(item.occupancy for item in devices) / len(devices), rel=1e-12
    )
    used = [item.occupancy for item in devices if item.ops]
    assert row.mean_occupancy < math.fsum(used) / len(used)


def test_the_class_row_is_the_same_accounting_as_the_device_rows(moe_dense):
    for row in moe_dense.utilization:
        if row.device_class == "link":
            continue
        devices = [
            item for item in moe_dense.occupancy if item.device_class == row.device_class
        ]
        assert row.busy_s == pytest.approx(
            math.fsum(item.busy_s for item in devices), rel=1e-12
        )
        assert row.ops == sum(item.ops for item in devices)
        assert row.max_occupancy == max(item.occupancy for item in devices)


def test_the_link_row_says_it_is_a_demand_ratio_and_not_an_occupancy(moe_dense):
    # D17 gives the fabric no congestion model, so a link occupies nothing on
    # the timeline. Printing its number as an occupancy would be a claim the
    # model does not make.
    row = moe_dense.utilization_of("link")
    assert row.devices > 0
    assert "DEMAND ratio" in row.basis
    assert "no congestion model" in row.basis


def test_low_utilization_is_a_finding_the_report_surfaces(moe_dense):
    # D28. The banner is UNCONDITIONAL: no threshold decides when the reader
    # is allowed to see it, and it names the class that actually binds.
    banners = [
        item
        for item in moe_dense.disclosures
        if item.constraint == "device_class_utilization"
    ]
    assert len(banners) == 1
    assert "analog_macro" in banners[0].value and "shared_digital" in banners[0].value
    assert "idle" in banners[0].value
    assert "BINDING class here is shared_digital" in banners[0].reason
    # And the analog side really is the idle silicon this run should report.
    assert moe_dense.utilization_of("analog_macro").mean_occupancy < 0.01
    assert moe_dense.utilization_of("shared_digital").mean_occupancy > 0.2


def test_the_text_report_prints_the_utilization_table(moe_dense):
    text = "\n".join(fws_eval.render_report(fws_eval.report_document(moe_dense)))
    assert "Utilization by device class" in text
    assert "idle silicon is a finding" in text
    assert "Bank passes in decode step" in text
    assert "Bank sharing in decode step" in text
    assert "Packing (Invariant W, D27)" in text


def test_the_atlas_export_seam_carries_the_same_rows(moe_dense):
    # One accounting, two consumers: the atlas gets the rows the report
    # printed, not a second derivation of them (D21).
    document = fws_eval.report_document(moe_dense)
    assert moe_dense.atlas_utilization() == document["evaluation"]["utilization"]
    assert moe_dense.atlas_packing() == document["evaluation"]["packing"]


def test_waste_is_reported_under_dense_and_refused_under_dedicated(
    moe_dense, moe_dedicated
):
    dense = fws_eval.report_document(moe_dense)["evaluation"]["packing"]
    assert dense["waste_reported"] is True
    assert dense["packing"] == "dense"
    assert dense["waste_pct"] == pytest.approx(
        100.0 * dense["waste_cells"] / dense["committed_cells"], rel=1e-12
    )
    assert dense["real_cells"] + dense["remainder_cells"] + dense["tail_cells"] == (
        dense["committed_cells"]
    )
    dedicated = fws_eval.report_document(moe_dedicated)["evaluation"]["packing"]
    assert dedicated["waste_reported"] is False
    assert "waste_pct" not in dedicated
    assert "would be invented" in dedicated["basis"]


# ---------------------------------------------------------------------------
# 6. Granite end to end, with a regenerate-and-compare gate (P7.3 item 4)
# ---------------------------------------------------------------------------


def test_granite_dense_packing_saves_macros_at_the_same_bank_passes(
    granite_packing_document,
):
    # THE HEADLINE, on the shipped Wave D winner (bank_depth = 1). Today's
    # layers_per_chip placement reaches 5544 macros; Invariant W reaches 4828
    # for the same weights, against a GLOBAL CELL FLOOR of 2876. What is left
    # above the floor is chip granularity plus the 40.442% dimension-mismatch
    # remainder, which is the metric, not a rounding.
    #
    # WAVE F: every count here fell by the LM HEAD and by nothing else — D30
    # drops it entirely, which is 66 macros under either placement law (5610 ->
    # 5544, 4894 -> 4828) and 65 off the cell floor. The SAVING is unchanged at
    # 716 macros: the endpoint filled whole macros, so packing it densely never
    # bought anything.
    document = granite_packing_document
    dedicated, dense = document["points"]
    assert dedicated["macros_holding_tiles"] == 5544
    assert dense["macros_holding_tiles"] == 4828
    assert document["delta"]["macros_saved"] == 716
    assert document["delta"]["cell_floor_macros"] == 2876
    assert document["delta"]["macros_above_floor"] == 4828 - 2876
    assert document["delta"]["waste_pct"] == pytest.approx(40.44203949185308)
    # Fact 1 under D29: the same weights are read the same number of times per
    # BEAT — every stage fires every beat, so one beat reads the whole model
    # once and one token leaves.
    assert dense["serving_regime"] == dedicated["serving_regime"] == "filled_pipeline"
    assert dense["bank_passes"]["unit"] == dedicated["bank_passes"]["unit"] == "beat"
    assert (
        dense["bank_passes"]["column_set_passes_charged"]
        == dedicated["bank_passes"]["column_set_passes_charged"]
        == 19304.0
    )
    # ... and every macro of both placements walked its own banks exactly once.
    for point in (dedicated, dense):
        assert (
            point["bank_passes"]["macros_walked_exactly_once"]
            == point["bank_passes"]["macros_holding_tiles"]
        )


def test_granite_dense_packing_moves_the_decode_step_by_the_accumulator_trade(
    granite_packing_document,
):
    # The BEAT is NOT assignment-invariant to the last digit and P7 says why:
    # stacking K blocks in one macro buys an in-macro accumulator and sells a
    # partial-sum transport. Dense is 684 accumulator ops richer and 0.103%
    # faster per beat; both numbers are on the artifact.
    #
    # WAVE F: the op count rose from 288 with the WINDOW, not with the packing —
    # the filled pipeline lowers D - 1 fill beats plus the steady ones, so the
    # same 72 local K stacks fire on more traversals (684 = 72 stacks x the
    # traversals the window lowers, minus the ones truncated at its edge). The
    # per-beat trade is what moved: the accumulator saving is now measured
    # against a beat 17x shorter than the retired regime's step, so the same
    # trade is a smaller PERCENTAGE of it.
    document = granite_packing_document
    dedicated, dense = document["points"]
    assert dedicated["accumulator_ops"] == 0
    assert dense["accumulator_ops"] == 684
    assert document["delta"]["median_decode_step_delta_s"] < 0
    assert document["delta"]["median_decode_step_delta_pct"] == pytest.approx(
        -0.10260106919069317
    )
    assert dense["tokens_per_s"] > dedicated["tokens_per_s"]


def test_the_granite_packing_artifact_states_its_terms(granite_packing_document):
    document = granite_packing_document
    assert document["schema"] == fws_eval.PACKING_COMPARISON_SCHEMA
    assert "DECODE ONLY (D25)" in document["note"]
    assert document["provenance"]["invented_fields"] == []
    assert document["provenance"]["hardware_config"].endswith(
        "fws_cim_granite_tiny_dse_selected.yaml"
    )
    assert document["model"]["model_id"] == "granite_4_0_h_tiny"
    assert document["model"]["phase"] == "decode"
    # Utilization travels WITH the comparison: a point without it would be a
    # (area, throughput) pair with no way to see the idle silicon (D28).
    for point in document["points"]:
        classes = [row["device_class"] for row in point["utilization"]]
        assert classes == ["analog_macro", "macro_pool", "shared_digital", "link"]


def test_the_comparison_refuses_a_pairing_that_is_not_one_experiment(
    moe_dense, moe_dedicated
):
    # ONE model, ONE machine, two laws. The argument order IS the comparison's
    # meaning, and a pair that differs by anything else would answer a question
    # nobody asked — so both are checked rather than assumed.
    with pytest.raises(MappingError, match="DEDICATED evaluation first"):
        fws_eval.packing_comparison_document(
            moe_dense, moe_dense, hardware_config="x", model_config="y"
        )

    def shorter_window(raw):
        _fine_bank(1)(raw)
        raw.setdefault("mapping", {}).update(LOCKSTEP, decode_window=1)

    windowed = _run(
        FWS_MOE, MOE_SMALL, shorter_window, packing=fws_mapping.PACKING_DENSE
    )
    with pytest.raises(MappingError, match="serve different points"):
        fws_eval.packing_comparison_document(
            moe_dedicated, windowed, hardware_config="x", model_config="y"
        )


def test_the_granite_packing_artifact_is_what_a_fresh_run_produces(
    granite_packing_document,
):
    """The committed artifact cannot rot: it is regenerated and compared.

    BYTES, not parsed values — the document is a pure function of the two
    YAMLs (no clock, no path, no hash), so byte equality is the right bar. A
    law change that moves a Granite number now has to move this file too, in
    the same commit.
    """
    regenerated = json.dumps(granite_packing_document, indent=2)
    if os.environ.get("FWS_WRITE_PACKING_ARTIFACT"):
        PACKING_ARTIFACT.write_text(regenerated, encoding="utf-8")
    # A missing artifact FAILS, never skips: deleting the file must not turn
    # this gate green.
    assert PACKING_ARTIFACT.exists(), (
        "docs/qif/folding/granite_dense_vs_dedicated.json is committed; regenerate "
        "it with the command in its provenance block"
    )
    assert regenerated == PACKING_ARTIFACT.read_text(encoding="utf-8"), (
        "docs/qif/folding/granite_dense_vs_dedicated.json is stale; regenerate it "
        "with the command in its provenance block"
    )
