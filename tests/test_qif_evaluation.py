"""QIF P4 evaluation tests: pricing, projections, verdicts and the reports.

Five things can rot silently in an evaluator and every one of them is a
plausible-looking wrong number rather than a crash:

1. the heterogeneous device layer moving the GPU evaluator (the 221 goldens
   catch the numbers; the tests here catch the SEMANTICS — capacity 1 is the
   old exclusivity, a root still does not occupy its device);
2. a metric quietly becoming a formula instead of a projection of the
   timeline (the AUDIT-4 failure mode);
3. an extrapolation being promoted into the metric set;
4. a capacity check passing by default because nobody declared the capacity;
5. an energy total drifting away from the sum of its printed parts.

Every test below pins one of those.
"""

from __future__ import annotations

import copy
import json
import math
import os

import pytest
import yaml

import config
import fws_atlas_export
import fws_eval
import fws_mapping
from program.analytic_sim import DeviceResource, evaluate, evaluate_detailed
from program.fws_build import annotations_of, build_fws_program
from program.ir import ProgramBuilder, ProgramMeta
from program.layout import RankLayout

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HW_DIR = os.path.join(PROJECT_ROOT, "configs", "hardware-config")
MODEL_DIR = os.path.join(PROJECT_ROOT, "configs", "model-config")

FWS_LLAMA7B = os.path.join(HW_DIR, "fws_cim_llama7b.yaml")
FWS_LLAMA7B_MAPPED = os.path.join(HW_DIR, "fws_cim_llama7b_mapped.yaml")
FWS_MOE = os.path.join(HW_DIR, "fws_cim_moe.yaml")
FWS_T1 = os.path.join(HW_DIR, "fws_cim_optima_t1.yaml")
LLAMA2_7B = os.path.join(MODEL_DIR, "llama2_7b_fws_inf.yaml")
MOE_SMALL = os.path.join(MODEL_DIR, "moe_small_fws_inf.yaml")
VIT_HUGE_64 = os.path.join(MODEL_DIR, "vit_huge_story_64_inf.yaml")


def _hw(path, mutate=None):
    with open(path) as handle:
        raw = copy.deepcopy(yaml.safe_load(handle))
    if mutate is not None:
        mutate(raw)
    config.convert(raw)
    return config.HWConfig.from_dict(raw)


def _mapping(hw_path, model_path, mode="LLM", mutate=None, **kwargs):
    hw = _hw(hw_path, mutate)
    model = config.parse_config(str(model_path), mode)
    return fws_mapping.build_mapping(hw, model, **kwargs)


@pytest.fixture(scope="module")
def llama_eval():
    mapping = _mapping(FWS_LLAMA7B, LLAMA2_7B)
    return fws_eval.evaluate_fws(build_fws_program(mapping, decode_steps=3))


@pytest.fixture(scope="module")
def vit_eval():
    mapping = _mapping(FWS_T1, VIT_HUGE_64, mode="VIT")
    return fws_eval.evaluate_fws(build_fws_program(mapping))


@pytest.fixture(scope="module")
def llama_eval_full_window():
    """A run whose lowered window COVERS the declared decode length.

    Nothing is truncated here, so nothing is extrapolated and a request really
    does complete inside the timeline. It is the control for the windowed
    fixtures: it proves the honest renaming is the truncation talking and not a
    blanket relabel.
    """
    mapping = _mapping(FWS_MOE, MOE_SMALL)
    decode_len = int(mapping.model.decode_len)
    return fws_eval.evaluate_fws(
        build_fws_program(mapping, decode_steps=decode_len)
    )


@pytest.fixture(scope="module")
def moe_eval():
    mapping = _mapping(FWS_MOE, MOE_SMALL)
    return fws_eval.evaluate_fws(build_fws_program(mapping, decode_steps=2))


# ---------------------------------------------------------------------------
# P4.1 — the heterogeneous device layer, capacity 1 = the old semantics
# ---------------------------------------------------------------------------


def _toy_program(durations, deps, devices):
    """A hand-built PIPELINE program: one compute op per entry."""
    builder = ProgramBuilder(
        layout=RankLayout((), {}, {}), dp_count=1, meta=ProgramMeta(label="toy")
    )
    for index, duration in enumerate(durations):
        builder.add_compute(f"op{index}", devices[index], duration, deps=deps[index])
    program = builder.finish(devices=tuple(sorted(set(devices))), validate=True)
    program.meta.misc["granularity"] = "pipeline"
    return program


def test_capacity_one_is_the_old_exclusivity_bit_identically():
    """Three independent ops on ONE device serialize, with or without resources."""
    program = _toy_program([1.0, 2.0, 4.0], [(), (), ()], [0, 0, 0])
    plain = evaluate_detailed(program, None, {}, durations=[1.0, 2.0, 4.0])
    with_resources = evaluate_detailed(
        program,
        None,
        {},
        durations=[1.0, 2.0, 4.0],
        resources={0: DeviceResource(0, capacity=1, owner="macro 0")},
    )
    assert plain.finish_times == with_resources.finish_times
    assert plain.total_time == with_resources.total_time
    # The legacy root rule: roots are pushed at t=0 WITHOUT occupying the
    # device, so all three finish at their own duration.
    assert plain.finish_times == [1.0, 2.0, 4.0]


def test_a_root_completion_never_lifts_a_device_above_its_capacity():
    """The ``min(capacity, free + 1)`` release — the one place counters could drift.

    op0 is a root (does not occupy device 0). op1..op3 are its children on the
    same device. If op0's completion had incremented a counter instead of
    re-asserting a boolean, device 0 would briefly hold TWO free slots and two
    children would start at once.
    """
    program = _toy_program(
        [1.0, 2.0, 2.0, 2.0], [(), (0,), (0,), (0,)], [0, 0, 0, 0]
    )
    result = evaluate_detailed(program, None, {}, durations=[1.0, 2.0, 2.0, 2.0])
    assert result.finish_times == [1.0, 3.0, 5.0, 7.0]


def test_capacity_above_one_is_a_real_resource_not_a_relabeling():
    program = _toy_program(
        [1.0, 2.0, 2.0, 2.0], [(), (0,), (0,), (0,)], [0, 0, 0, 0]
    )
    result = evaluate_detailed(
        program,
        None,
        {},
        durations=[1.0, 2.0, 2.0, 2.0],
        resources={0: DeviceResource(0, capacity=2, owner="chiplet with two lanes")},
    )
    # Two children run concurrently, the third waits for a lane.
    assert result.finish_times == [1.0, 3.0, 3.0, 5.0]


def test_a_zero_capacity_device_is_refused_by_name():
    with pytest.raises(ValueError) as excinfo:
        DeviceResource(3, capacity=0)
    assert "not a capacity" in str(excinfo.value)


def test_supplied_durations_must_be_uid_indexed():
    program = _toy_program([1.0], [()], [0])
    with pytest.raises(ValueError) as excinfo:
        evaluate_detailed(program, None, {}, durations=[1.0, 2.0])
    assert "uid-indexed" in str(excinfo.value)


def test_evaluate_without_the_new_arguments_is_unchanged():
    """The default call still converts comm sizes and still demands PIPELINE."""
    program = _toy_program([1.0, 2.0], [(), (0,)], [0, 1])
    assert evaluate(program, None, {}) == 3.0
    program.meta.misc["granularity"] = "flat"
    with pytest.raises(RuntimeError):
        evaluate(program, None, {})


def test_every_device_carries_a_named_owner(llama_eval):
    """D21: every macro has a named owner — checked, not hoped for."""
    resources = llama_eval.pricing.resources
    assert resources, "the mapping declared no devices"
    assert all(resource.owner for resource in resources.values())
    assert all(resource.basis for resource in resources.values())
    macro_owned = [
        resource
        for device_id, resource in resources.items()
        if resource.device_class == "analog_macro"
    ]
    assert macro_owned
    assert all(
        "unowned macro slot" in resource.owner or "." in resource.owner
        for resource in macro_owned
    )


def test_every_op_is_priced_through_a_named_law(llama_eval):
    for cost in llama_eval.pricing.costs:
        assert cost.basis, f"op {cost.uid} carries no pricing basis"
        assert cost.duration_s >= 0.0
    priced = {cost.kind for cost in llama_eval.pricing.costs}
    assert {"weight_gemm", "pool", "fabric", "transfer"} <= priced


def test_transfers_are_priced_but_never_occupy_a_device(llama_eval):
    """D17: transfers are priced, never contended."""
    transfers = [c for c in llama_eval.pricing.costs if c.kind == "transfer"]
    assert any(cost.duration_s > 0 for cost in transfers)
    busy = sum(row.busy_s for row in llama_eval.occupancy)
    compute_time = sum(
        cost.duration_s for cost in llama_eval.pricing.costs if cost.kind != "transfer"
    )
    # The busy union can only be <= the summed compute durations, and transfer
    # time is nowhere in it.
    assert busy <= compute_time + 1e-9


# ---------------------------------------------------------------------------
# P4.2 — every metric is a projection of the one timeline
# ---------------------------------------------------------------------------


def test_throughput_is_tokens_over_the_observed_span_of_the_same_timeline(llama_eval):
    """The projection-not-formula gate: recompute the headline from the object."""
    timeline = llama_eval.timeline
    costs = llama_eval.pricing.costs
    prefill_finish = max(
        timeline.finish_times[c.uid] for c in costs if c.phase == "prefill"
    )
    decode_finish = max(
        timeline.finish_times[c.uid] for c in costs if c.phase == "decode"
    )
    steps = len(llama_eval.decode_step_times_s)
    expected = (llama_eval.serving.batch * steps) / (decode_finish - prefill_finish)
    assert llama_eval.metric("sys.fws.tokens_per_s").value == pytest.approx(
        expected, rel=0, abs=0
    )


def test_request_latency_is_read_once_off_the_critical_path(llama_eval):
    """One read of one timeline — under whichever name the span honestly has.

    ``llama_eval`` is a WINDOWED decode run, so the makespan is the span of the
    lowered window and not a request's end-to-end latency; the metric therefore
    ships as ``lowered_window_latency``. The arithmetic being pinned here is the
    same either way: makespan minus first issue, read once.
    """
    timeline = llama_eval.timeline
    starts = [t for t in timeline.start_times if t >= 0]
    expected = timeline.total_time - min(starts)
    assert llama_eval.extrapolation["extrapolated"] is True
    assert llama_eval.metric("sys.fws.lowered_window_latency").value == expected
    keys = {metric.key for metric in llama_eval.metrics}
    assert "sys.fws.request_latency" not in keys


def test_a_truncated_window_never_claims_a_completed_request(llama_eval):
    """The window is not a request, and no metric may say it is (D21).

    A windowed run lowers a few decode steps of many, so ZERO of the batch's
    requests complete inside the timeline. Publishing that span as
    "end-to-end request latency" and the batch over it as "completed requests
    per second" would put a second, contradicting value for request latency in
    the same artifact beside ``extrapolation.extrapolated_request_latency_s``.
    """
    assert llama_eval.extrapolation["extrapolated"] is True
    for metric in llama_eval.metrics:
        label = metric.label.lower()
        if "request" not in label:
            continue
        # Every surviving "request" label must say it is NOT a whole request.
        assert "not a" in label, metric.key
    window = llama_eval.metric("sys.fws.requests_per_s_lowered_window")
    assert window.unit == "requests/s"
    assert "ZERO of them COMPLETE" in window.basis
    # And it is never back-derived from the extrapolation (P4 §3).
    extrapolated = float(llama_eval.extrapolation["extrapolated_request_latency_s"])
    assert window.value != pytest.approx(llama_eval.serving.batch / extrapolated)


def test_an_unwindowed_run_keeps_the_unqualified_names(llama_eval_full_window):
    """The rename is the TRUNCATION talking, not a blanket relabel.

    When the window covers the declared decode length nothing is extrapolated,
    a request really does complete, and the metrics carry their plain names.
    """
    evaluation = llama_eval_full_window
    assert evaluation.extrapolation["extrapolated"] is False
    keys = {metric.key for metric in evaluation.metrics}
    assert "sys.fws.request_latency" in keys
    assert "sys.fws.requests_per_s" in keys
    assert "sys.fws.lowered_window_latency" not in keys
    assert "sys.fws.requests_per_s_lowered_window" not in keys
    assert evaluation.metric("sys.fws.request_latency").label == (
        "end-to-end request latency"
    )


def test_decode_step_times_are_differences_of_terminal_finishes(llama_eval):
    timeline = llama_eval.timeline
    costs = llama_eval.pricing.costs
    terminals = []
    previous = max(timeline.finish_times[c.uid] for c in costs if c.phase == "prefill")
    for step in range(len(llama_eval.decode_step_times_s)):
        finish = max(
            timeline.finish_times[c.uid]
            for c in costs
            if c.phase == "decode" and c.step == step
        )
        terminals.append(finish - previous)
        previous = finish
    assert list(llama_eval.decode_step_times_s) == terminals
    assert llama_eval.metric("sys.fws.decode_step_first").value == terminals[0]
    assert llama_eval.metric("sys.fws.decode_step_last").value == terminals[-1]


def test_the_decode_series_moves_forward_in_time(llama_eval):
    """A negative step time means the DAG lost the prefill -> decode edge."""
    assert all(value > 0 for value in llama_eval.decode_step_times_s)
    assert llama_eval.metric("sys.fws.prefill_latency").value > 0


def test_no_metric_is_a_period_times_a_count(llama_eval):
    """The extrapolation exists, is labeled, and never enters metrics[]."""
    assert llama_eval.extrapolation["extrapolated"] is True
    assert "period" not in {metric.key for metric in llama_eval.metrics}
    keys = {metric.key for metric in llama_eval.metrics}
    assert not any("extrapolat" in key for key in keys)
    assert "EXTRAPOLATION" in str(llama_eval.extrapolation["basis"])
    disclosed = [
        item for item in llama_eval.disclosures if item.constraint == "decode_series_window"
    ]
    assert disclosed, "a windowed decode series must disclose the window (ADJ-6)"


def test_the_ceiling_sustained_pair_is_gone_from_the_dag_report(llama_eval):
    """ADJ-6 retires the pair: no published FIGURE is a ceiling or a sustained rate."""
    document = fws_eval.report_document(llama_eval)
    published = {metric["key"] for metric in document["evaluation"]["metrics"]}
    published |= {metric["label"].lower() for metric in document["evaluation"]["metrics"]}
    assert not any("ceiling" in name or "sustained" in name for name in published)
    assert "limiting_factor" not in json.dumps(document["evaluation"])
    text = "\n".join(fws_eval.render_report(document))
    # It survives only as prose explaining what retired, never as a number.
    assert "retired with the closed form" in text


def test_headline_and_requests_per_second_are_both_printed(llama_eval):
    headline = llama_eval.metric("sys.fws.tokens_per_s")
    assert "HEADLINE" in headline.label and headline.unit == "tokens/s"
    # ADJ-6 prints a requests/s figure beside the headline. On a windowed run it
    # is the window rate and says so; the pair is printed either way.
    beside = [
        metric for metric in llama_eval.metrics if metric.unit == "requests/s"
    ]
    assert len(beside) == 1
    assert beside[0].key.startswith("sys.fws.requests_per_s")


def test_occupancy_and_duty_cycles_are_owner_attributed_fractions(llama_eval):
    assert llama_eval.duty_cycles
    assert all(0.0 <= value <= 1.0 for value in llama_eval.duty_cycles.values())
    busy_rows = [row for row in llama_eval.occupancy if row.ops]
    assert busy_rows
    for row in busy_rows:
        assert 0.0 <= row.occupancy <= 1.0
        assert row.owner
    macro_rows = {
        row.macro_id: row
        for row in llama_eval.occupancy
        if row.device_class == "analog_macro"
    }
    for macro_id, duty in llama_eval.duty_cycles.items():
        assert duty == macro_rows[macro_id].occupancy


def test_a_prefill_only_workload_reports_no_decode_metrics(vit_eval):
    keys = {metric.key for metric in vit_eval.metrics}
    assert "sys.fws.prefill_latency" in keys
    assert not any("decode" in key for key in keys)
    assert vit_eval.decode_step_times_s == ()


# ---------------------------------------------------------------------------
# P4.3 — memory feasibility
# ---------------------------------------------------------------------------


def test_memory_verdicts_name_the_offender_and_the_quantity(llama_eval):
    assert llama_eval.memory
    violated = [v for v in llama_eval.memory if v.status == "VIOLATED"]
    assert violated, "the Llama2-7B point overruns the declared 8 GiB tier"
    for verdict in violated:
        assert verdict.owner and verdict.scope
        assert verdict.high_water_bytes > verdict.capacity_bytes
        assert verdict.disclosure
        assert math.isclose(
            verdict.high_water_bytes,
            math.fsum(verdict.contributors.values()),
            rel_tol=1e-12,
        )


def test_an_undeclared_capacity_is_not_a_pass():
    """A missing capacity field says 'undeclared', never 'fits' (D21)."""

    def mutate(raw):
        raw["tech_param"]["DRAM"]["size"] = 0

    mapping = _mapping(FWS_LLAMA7B, LLAMA2_7B, mutate=mutate)
    evaluation = fws_eval.evaluate_fws(build_fws_program(mapping, decode_steps=1))
    statuses = {verdict.status for verdict in evaluation.memory}
    assert statuses == {"undeclared"}
    assert all(verdict.capacity_bytes == 0 for verdict in evaluation.memory)


def test_the_shared_kv_tier_is_disclosed_not_assumed(llama_eval):
    names = {item.constraint for item in llama_eval.disclosures}
    assert "kv_shares_the_activation_tier" in names
    assert "activation_tier_is_per_chip" in names


def test_kv_bytes_ride_the_p2_law_and_the_timeline_context(llama_eval):
    """The KV figure is P2's law at the context the timeline actually reached."""
    mapping = llama_eval.mapping
    kv_precision = float(mapping.hw.sw_config.precision.kv_cache)
    context = llama_eval.serving.prefill_len + len(llama_eval.decode_step_times_s)
    per_layer = mapping.device.kv_bytes_per_stream_layer(context, kv_precision, 1)
    total_kv = math.fsum(
        verdict.contributors.get("kv_cache", 0.0) for verdict in llama_eval.memory
    )
    layers = int(mapping.device.params.num_layers)
    assert total_kv == pytest.approx(
        llama_eval.serving.batch * layers * per_layer, rel=1e-12
    )
    # KV lives where the act x act attention runs: the shared digital chiplets
    # (D13), never on the analog chips that only hold weights.
    holders = {
        verdict.scope for verdict in llama_eval.memory
        if verdict.contributors.get("kv_cache", 0.0) > 0
    }
    digital = {f"chip {chip.chip_id}" for chip in mapping.digital_chips()}
    assert holders and holders <= digital


# ---------------------------------------------------------------------------
# P4.4 — itemized energy with per-component coverage labels
# ---------------------------------------------------------------------------


def test_the_energy_total_is_the_sum_of_the_printed_parts(llama_eval):
    assert llama_eval.energy
    assert llama_eval.total_energy_pj == math.fsum(
        component.energy_pj for component in llama_eval.energy
    )


def test_every_energy_component_carries_its_own_coverage_label(llama_eval):
    allowed = {
        fws_eval.COVERAGE_COVERED,
        fws_eval.COVERAGE_PARTIAL,
        fws_eval.COVERAGE_UNCOVERED,
    }
    keys = {component.key for component in llama_eval.energy}
    assert {
        "analog_arrays",
        "link_traffic",
        "macro_pool_helpers",
        "shared_digital_chiplet",
    } <= keys
    for component in llama_eval.energy:
        assert component.coverage in allowed
        assert component.basis
    analog = next(c for c in llama_eval.energy if c.key == "analog_arrays")
    assert analog.coverage == fws_eval.COVERAGE_COVERED and analog.energy_pj > 0
    fabric = next(c for c in llama_eval.energy if c.key == "shared_digital_chiplet")
    assert fabric.coverage == fws_eval.COVERAGE_UNCOVERED and fabric.energy_pj == 0.0


def test_a_declared_pool_energy_knob_flips_the_reduction_coverage():
    def mutate(raw):
        raw["cim"]["cards"] = {
            "ctt_costed": {"kind": "analog_macro", "pool_energy_per_add_pj": 0.05}
        }

    mapping = _mapping(FWS_LLAMA7B, LLAMA2_7B, mutate=mutate)
    evaluation = fws_eval.evaluate_fws(build_fws_program(mapping, decode_steps=1))
    reduction = next(
        c for c in evaluation.energy if c.key == "digital_reduction"
    )
    assert reduction.coverage == fws_eval.COVERAGE_COVERED
    assert reduction.energy_pj > 0


def test_analog_energy_is_p2s_own_number(llama_eval):
    """No re-derivation: the component is the sum of price_tiled_op energies."""
    expected = math.fsum(
        cost.energy_pj
        for cost in llama_eval.pricing.costs
        if cost.energy_component == "analog_arrays"
    )
    analog = next(c for c in llama_eval.energy if c.key == "analog_arrays")
    assert analog.energy_pj == expected


# ---------------------------------------------------------------------------
# P4.5 — pool sizing FROM the timeline
# ---------------------------------------------------------------------------


def test_pool_sizing_scales_p2s_derivation_by_the_measured_concurrency(llama_eval):
    assert llama_eval.pools
    for pool in llama_eval.pools:
        assert pool.peak_concurrent_demand >= 1
        assert pool.scaled.lanes == pool.per_unit.lanes * pool.peak_concurrent_demand
        assert pool.scaled.adders == pool.per_unit.adders * pool.peak_concurrent_demand
        assert pool.owner
        assert "FWS-CIM" in pool.report_line
    assert max(p.peak_concurrent_demand for p in llama_eval.pools) >= 1


def test_an_absurd_derived_width_prints_as_an_absurd_derived_width():
    """Nothing is clamped: a slow pool clock yields an enormous lane count."""

    def mutate(raw):
        raw["cim"]["cards"] = {
            "ctt_slow_pool": {"kind": "analog_macro", "pool_clock_ghz": 0.000001}
        }

    mapping = _mapping(FWS_LLAMA7B, LLAMA2_7B, mutate=mutate)
    evaluation = fws_eval.evaluate_fws(build_fws_program(mapping, decode_steps=1))
    worst = max(evaluation.pools, key=lambda pool: pool.scaled.lanes)
    assert worst.scaled.lanes > 10_000_000
    assert str(worst.scaled.lanes) in worst.report_line


def test_the_pool_demand_is_measured_not_assumed(llama_eval):
    """The sizing rides on real pool ops, and their count is reported."""
    pool_ops = sum(pool.pool_ops for pool in llama_eval.pools)
    priced_pool_ops = sum(
        1
        for cost in llama_eval.pricing.costs
        if cost.device_class == "macro_pool" and cost.macro_id >= 0
    )
    assert pool_ops == priced_pool_ops > 0


# ---------------------------------------------------------------------------
# P4.6 — the reports
# ---------------------------------------------------------------------------


def test_the_report_has_one_producer_per_block_and_one_version(llama_eval):
    document = fws_eval.report_document(llama_eval)
    assert document["schema"] == fws_eval.REPORT_SCHEMA
    assert document["model"]["producer"].startswith("P1")
    assert document["mapping"]["producer"].startswith("P3")
    assert document["evaluation"]["producer"].startswith("fws_eval")
    assert document["disclosures"]


def test_every_metric_in_the_report_names_its_accounting_basis(llama_eval):
    document = fws_eval.report_document(llama_eval)
    for metric in document["evaluation"]["metrics"]:
        assert metric["basis"] and metric["unit"] and metric["label"]
    assert document["evaluation"]["occupancy_basis"]
    assert document["evaluation"]["duty_cycle_basis"]
    assert document["evaluation"]["energy"]["total_basis"]
    assert document["evaluation"]["digital_pool"]["basis"]


def test_the_text_section_is_rendered_from_the_json_and_from_nothing_else(llama_eval):
    document = fws_eval.report_document(llama_eval)
    before = "\n".join(fws_eval.render_report(document))
    assert "FWS-CIM placed-DAG evaluation" in before
    assert "%.6g" % llama_eval.metric("sys.fws.tokens_per_s").value in before
    # Change the JSON, and the text changes with it: the renderer computes
    # nothing of its own.
    document["evaluation"]["metrics"][0]["value"] = 1234.5
    after = "\n".join(fws_eval.render_report(document))
    assert "1234.5" in after and before != after


def test_the_report_carries_one_accounting_of_each_metric(llama_eval):
    document = fws_eval.report_document(llama_eval)
    keys = [metric["key"] for metric in document["evaluation"]["metrics"]]
    assert len(keys) == len(set(keys))
    text = "\n".join(fws_eval.render_report(document))
    # The closed-form vocabulary must not appear beside the DAG numbers.
    for banned in ("Pipeline period", "S1..S5", "inferences/s"):
        assert banned not in text


def test_the_report_discloses_the_surrounding_results_file_totals(llama_eval):
    document = fws_eval.report_document(llama_eval)
    names = {item["constraint"] for item in document["disclosures"]}
    assert "surrounding_results_file_totals" in names
    assert "op_durations" not in names, "P3's unpriced disclosure retires once P4 prices"


def test_a_mapped_run_publishes_the_dag_report_and_not_the_closed_form(tmp_path):
    import inference_timing

    hw = config.parse_config(FWS_LLAMA7B_MAPPED, config_type="hardware")
    model = config.parse_config(LLAMA2_7B, config_type="LLM")
    config.validate_configs(hw, model)
    tc = inference_timing.TimeCalculationLLMInference(
        hw, model, "LLM", output_dir=str(tmp_path)
    )
    tc.calc_total_inference_time()
    assert (tmp_path / "fws_qif_report.json").exists()
    assert not (tmp_path / "fws_cim_report.json").exists()
    with open(tmp_path / "fws_qif_report.json") as handle:
        document = json.load(handle)
    assert document["schema"] == fws_eval.REPORT_SCHEMA
    assert tc.fws_cim_report_lines
    assert "placed-DAG evaluation" in "\n".join(tc.fws_cim_report_lines)


def test_an_unmapped_run_still_publishes_the_legacy_closed_form(tmp_path):
    """ADJ-6 demotes the closed form; the P6.2 gates still pin its path."""
    import inference_timing

    hw = config.parse_config(FWS_LLAMA7B, config_type="hardware")
    model = config.parse_config(LLAMA2_7B, config_type="LLM")
    config.validate_configs(hw, model)
    assert getattr(hw, "mapping_config", None) is None
    tc = inference_timing.TimeCalculationLLMInference(
        hw, model, "LLM", output_dir=str(tmp_path)
    )
    tc.calc_total_inference_time()
    assert (tmp_path / "fws_cim_report.json").exists()
    assert not (tmp_path / "fws_qif_report.json").exists()


# ---------------------------------------------------------------------------
# P4 -> P5: duty cycles and labeled metrics reach the atlas
# ---------------------------------------------------------------------------


def test_the_atlas_export_takes_duty_cycles_from_the_evaluation(llama_eval):
    mapping = llama_eval.mapping
    plain = fws_atlas_export.export_atlas([mapping], title="plain")
    assert all(macro["duty_cycle"] is None for macro in plain["macros"])

    timed = fws_atlas_export.export_atlas(
        [mapping],
        title="timed",
        duty_cycles=[llama_eval.atlas_duty_cycles()],
        extra_metrics=[llama_eval.atlas_metrics()],
    )
    drawn = [
        macro["duty_cycle"] for macro in timed["macros"] if macro["duty_cycle"] is not None
    ]
    assert drawn
    # R27 of the frozen fws_atlas/1 loader: a duty cycle is a bar, so it must
    # be inside [0, 1] or the picture lies.
    assert all(0.0 <= value <= 1.0 for value in drawn)
    keys = {metric["key"] for metric in timed["metrics"]}
    assert "sys.fws.tokens_per_s" in keys
    assert all(metric["basis"] for metric in timed["metrics"])


def test_atlas_duty_cycles_are_one_entry_per_mapping(llama_eval):
    with pytest.raises(ValueError) as excinfo:
        fws_atlas_export.export_atlas(
            [llama_eval.mapping],
            title="broken",
            duty_cycles=[{}, {}],
        )
    assert "ONE entry per mapping" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Cross-workload smoke: the pricing table has a row for every placed op
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["llama_eval", "vit_eval", "moe_eval"])
def test_every_workload_prices_every_placed_op(fixture, request):
    evaluation = request.getfixturevalue(fixture)
    annotations = annotations_of(evaluation.program)
    assert len(evaluation.pricing.costs) == len(annotations)
    assert len(evaluation.pricing.durations) == len(evaluation.program.ops)
    assert evaluation.makespan_s > 0
    document = fws_eval.report_document(evaluation)
    json.dumps(document)  # the report must be serializable, always
    assert fws_eval.render_report(document)


def test_the_moe_run_prices_its_router_and_expert_stages(moe_eval):
    """The MoE point runs ep = 1, so experts stay on their layer's chip."""
    assert int(moe_eval.mapping.degrees["ep"]) == 1
    blocks = {cost.block for cost in moe_eval.pricing.costs}
    assert "router" in blocks
    assert {"ffn1_routed", "ffn2_routed"} <= blocks
    routed = [c for c in moe_eval.pricing.costs if c.block == "ffn1_routed"]
    assert routed and all(cost.duration_s > 0 for cost in routed)


def test_expert_parallel_dispatch_is_priced_over_the_ep_link():
    """ep > 1 moves experts off their layer's chip; those bytes ride the ep law."""
    def mutate(raw):
        raw["cim"]["chip"]["moe_expert_parallel"] = 2

    mapping = _mapping(FWS_MOE, MOE_SMALL, mutate=mutate)
    evaluation = fws_eval.evaluate_fws(build_fws_program(mapping, decode_steps=1))
    dispatch = [
        cost
        for cost, annotation in zip(
            evaluation.pricing.costs, annotations_of(evaluation.program)
        )
        if annotation.boundary_id.startswith("ep.dispatch")
    ]
    combine = [
        annotation
        for annotation in annotations_of(evaluation.program)
        if annotation.boundary_id.startswith("ep.combine")
    ]
    assert dispatch and combine
    assert all("over the ep link" in cost.basis for cost in dispatch)


# ---------------------------------------------------------------------------
# P1.5's attention.window un-rejection, driven END TO END (not law-level)
# ---------------------------------------------------------------------------


def _windowed_model(window_size=512, interval=6, last_global=True):
    """llama2_7b_fws_inf with model_param.attention.window declared.

    P1.5 narrowed ``_unpriced_model_inputs`` so a MAPPED fws_cim run no longer
    refuses ``attention.window``, and fws_eval._window_of plus the P2.6
    sliding-window laws implement it — but no shipped model YAML declares a
    window, so the branch had law-level and schema-level coverage and no run.
    ASSUMPTION (recorded on the board): the window is injected into a real
    config here rather than shipped as a new model YAML, because the natural
    carrier is a Gemma-3-class model and its dims cannot be verified against
    the published config.json from this tree (ADJ-4: no invented numbers).
    """
    with open(LLAMA2_7B) as handle:
        raw = copy.deepcopy(yaml.safe_load(handle))
    raw["model_param"]["attention"]["window"] = {
        "window_size": window_size,
        "local_global_interval": interval,
        "last_layer_global": last_global,
    }
    config.convert(raw)
    model_config = config.LLMConfig.from_dict(raw["model_param"])
    inference = config.LLMInferenceConfig.from_dict(raw.get("inference_param"))
    return config.ModelConfig(model_config=model_config, inference_config=inference)


@pytest.fixture(scope="module")
def windowed_eval():
    hw = _hw(FWS_LLAMA7B_MAPPED)
    model = _windowed_model()
    config.validate_configs(hw, model)  # the narrowing, exercised as a run does it
    mapping = fws_mapping.build_mapping(hw, model)
    return fws_eval.evaluate_fws(build_fws_program(mapping, decode_steps=2))


def test_a_declared_window_splits_the_layers_local_and_global(windowed_eval):
    """(i + 1) % local_global_interval == 0 is global, plus last_layer_global."""
    layers = int(windowed_eval.mapping.device.params.num_layers)
    expected_global = {
        layer
        for layer in range(layers)
        if (layer + 1) % 6 == 0 or layer == layers - 1
    }
    priced = {}
    for cost in windowed_eval.pricing.costs:
        if cost.block != "attention_qk" or cost.phase != "prefill":
            continue
        priced.setdefault(int(cost.layer), cost)
    assert priced, "the windowed run prices prefill attention"
    seen_global = {
        layer
        for layer, cost in priced.items()
        if "sliding_window" not in cost.basis
    }
    assert seen_global == expected_global & set(priced)
    assert set(priced) - seen_global, "some layer must be LOCAL, or nothing is tested"


def test_a_local_layer_is_cheaper_than_a_global_one(windowed_eval):
    """The window is applied, not merely parsed: it shortens the context."""
    qk = {}
    for cost in windowed_eval.pricing.costs:
        if cost.block == "attention_qk" and cost.phase == "prefill":
            qk.setdefault(int(cost.layer), cost)
    local = [cost for cost in qk.values() if "sliding_window" in cost.basis]
    globals_ = [cost for cost in qk.values() if "sliding_window" not in cost.basis]
    assert local and globals_
    assert max(c.duration_s for c in local) < min(c.duration_s for c in globals_)


def test_both_window_laws_are_reached_prefill_and_decode(windowed_eval):
    bases = {
        cost.phase: {c.basis for c in windowed_eval.pricing.costs
                     if c.block == "attention_qk" and c.phase == cost.phase
                     and "sliding_window" in c.basis}
        for cost in windowed_eval.pricing.costs
        if cost.block == "attention_qk"
    }
    assert any("sliding_window_prefill_timing" in b for b in bases.get("prefill", ()))
    assert any("sliding_window_decode_timing" in b for b in bases.get("decode", ()))


def test_the_window_pattern_rides_the_report_as_a_disclosure(windowed_eval):
    """A relaxation a reader cannot see in the artifact is not disclosed."""
    document = fws_eval.report_document(windowed_eval)
    named = [
        item
        for item in document["disclosures"]
        if item["constraint"] == "sliding_window_layer_pattern"
    ]
    assert named, "the local/global pattern P4 ASSUMED must be in the artifact"
    assert "local_global_interval" in named[0]["reason"]


def test_an_undeclared_window_prices_every_layer_at_full_context(llama_eval):
    """The degenerate default: no window block, no windowed law, anywhere."""
    assert getattr(llama_eval.mapping.model.attention, "window", None) is None
    assert not any(
        "sliding_window" in cost.basis for cost in llama_eval.pricing.costs
    )


# ---------------------------------------------------------------------------
# One constraint, one entry; one relaxation, one disclosure (D21)
# ---------------------------------------------------------------------------


def test_no_constraint_key_is_disclosed_twice(llama_eval):
    """Two entries under one key read as two claims, which is the D21 sin.

    P3 declares ``intra_chip_activation_movement`` generically and P4 declares
    it again with the measured byte count. Both survive, merged under one entry
    naming both producers' reasons.
    """
    document = fws_eval.report_document(llama_eval)
    keys = [item["constraint"] for item in document["disclosures"]]
    assert len(keys) == len(set(keys)), [k for k in keys if keys.count(k) > 1]
    merged = [
        item
        for item in document["disclosures"]
        if item["constraint"] == "intra_chip_activation_movement"
    ]
    assert len(merged) == 1
    # Nothing a producer said is dropped by the merge.
    assert "ALSO DISCLOSED" in merged[0]["reason"]
    assert "bytes moving inside a chip" in merged[0]["reason"]


def test_merge_disclosures_keeps_the_first_value_and_every_reason():
    items = (
        fws_mapping.Relaxation(constraint="c", value="first", reason="a" * 90),
        fws_mapping.Relaxation(constraint="c", value="second", reason="b" * 90),
        fws_mapping.Relaxation(constraint="d", value="only", reason="c" * 90),
    )
    merged = fws_eval.merge_disclosures(items)
    assert [entry["constraint"] for entry in merged] == ["c", "d"]
    assert merged[0]["value"] == "first"
    assert "a" * 90 in merged[0]["reason"]
    assert "b" * 90 in merged[0]["reason"]


def test_raising_a_device_capacity_discloses_the_optimistic_root_release(llama_eval):
    """Above capacity 1 the executor is optimistic, and the artifact says so.

    A root op never occupies a slot, yet its completion still releases one, so
    with capacity C and k non-root ops running a root's finish can leave C+1 ops
    in flight. At capacity 1 that release is the legacy idempotent assignment
    and the 221 goldens pin it, so nothing fires on any shipped run.
    """
    program = llama_eval.program
    default = fws_eval.price_program(program, llama_eval.mapping)
    assert not any(
        item.constraint == "device_capacity_root_release" for item in default.disclosures
    )
    device_id = next(iter(default.resources))
    raised = fws_eval.price_program(
        program, llama_eval.mapping, capacity_overrides={device_id: 2}
    )
    named = [
        item
        for item in raised.disclosures
        if item.constraint == "device_capacity_root_release"
    ]
    assert named, "a raised capacity is a relaxation and must be disclosed (D21)"
    assert "optimistic" in named[0].reason
    assert str(device_id) in named[0].value
