"""QIF P6.3 bridge-gate tests: the frozen closed form against the placed DAG.

A1 made the DAG the model and demoted the pass-1/2 spatial-pipeline report to a
legacy reference. ADJ-8 is what makes that move honest: on the degenerate case
both paths can express, the DAG must REPRODUCE the closed form — discrete
quantities exactly, continuous ones at 1e-3.

Four things can rot here, and all four are silent:

1. the gate passing because it compares the DAG to itself rather than to the
   artifact ``run_perf`` writes;
2. an exclusion widening until the gate compares nothing (the AUDIT's finding 5,
   the flagship rows that ran with the constraints switched off);
3. the projection leaking into ``fws_eval`` and becoming a SECOND accounting of
   a metric the DAG report already publishes (AUDIT finding 4);
4. a bug the gate already caught coming back.

Every test below pins one of those. The three upstream bugs the gate caught on
its first run each keep a named regression test at the bottom of this file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import yaml

import config
import fws_bridge
import fws_eval
import fws_mapping
from program.fws_build import annotations_of, build_fws_program

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HW_DIR = os.path.join(PROJECT_ROOT, "configs", "hardware-config")
MODEL_DIR = os.path.join(PROJECT_ROOT, "configs", "model-config")

#: ADJ-8's set: the three ViT tiers, the dense-LLM point, the MoE smoke.
POINTS = (
    ("T1", "fws_cim_optima_t1.yaml", "vit_huge_story_64_inf.yaml", "VIT"),
    ("T2", "fws_cim_optima_t2.yaml", "vit_g_64_inf.yaml", "VIT"),
    ("T3", "fws_cim_optima_t3.yaml", "vit_huge_story_196_inf.yaml", "VIT"),
    ("Llama7B", "fws_cim_llama7b.yaml", "llama2_7b_fws_inf.yaml", "LLM"),
    ("MoE", "fws_cim_moe.yaml", "moe_small_fws_inf.yaml", "LLM"),
)
_POINT_BY_LABEL = {point[0]: point for point in POINTS}


def _closed_form(hw_path, model_path, mode, output_root):
    """The REAL closed-form artifact: run_perf, no ``mapping:`` block, one JSON.

    Deliberately a subprocess. Reading the number the evaluator would have
    produced in-process would compare the DAG to a re-derivation of itself,
    which is exactly the failure mode this gate exists to prevent.

    ``output_root`` is a per-session temp dir handed to ``--output_dir``: the
    repo's own ``output/`` tree is shared state, and a second pytest session in
    the same checkout used to overwrite this gate's artifact mid-read.
    """
    proc = subprocess.run(
        [
            sys.executable,
            os.path.join(PROJECT_ROOT, "run_perf.py"),
            "--hardware_config",
            hw_path,
            "--model_config",
            model_path,
            "--output_dir",
            str(output_root),
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-4000:]
    report_path = os.path.join(str(output_root), mode, "fws_cim_report.json")
    with open(report_path) as handle:
        return json.load(handle)


def _dag(hw_path, model_path, mode):
    with open(hw_path) as handle:
        raw = yaml.safe_load(handle)
    config.convert(raw)
    hw = config.HWConfig.from_dict(raw)
    # ADJ-10 PIN, and it is the same pin the validator's bridge gate applies.
    # The mapped path DERIVES the attention fabric's array count and the softmax
    # pipeline's lane count; the frozen closed form has no timeline to derive
    # them from and prices the DECLARED cim.fabric seed. Without the pin the two
    # sides of this gate describe TWO MACHINES and every attention row compares a
    # derived fabric against a declared one. Pinning uses the ordinary card
    # override, so the run discloses it, and it drops no row and moves no
    # tolerance — every attention cycle count below is still compared EXACTLY.
    # Named in fws_bridge.EXCLUSIONS as
    # derived_fabric_pinned_to_the_declared_seed; the derivation itself is gated
    # by tests/test_fws_cim.py and the two e2e suites.
    digital = hw.cim_config.cards.digital_card
    digital.fabric_num_arrays = int(digital.fabric.num_arrays)
    digital.fabric_softmax_lanes = int(digital.fabric.softmax_lanes)
    mapping = fws_mapping.build_mapping(hw, config.parse_config(model_path, mode))
    evaluation = fws_eval.evaluate_fws(build_fws_program(mapping))
    device = mapping.device
    # The pin is a claim about the run, so it is CHECKED here rather than
    # assumed: the derivation must not have run, and the widths must be the
    # declared ones.
    assert device.derived_fabric is None
    assert device.fabric_num_arrays == int(digital.fabric.num_arrays)
    assert device.fabric_softmax_lanes == int(digital.fabric.softmax_lanes)
    assert len(device.fabric_sizing_disclosures()) == 1
    return evaluation


@pytest.fixture(scope="module")
def bridged(tmp_path_factory):
    """Every point, run once: {label: (closed_form, evaluation, rows)}."""
    out = {}
    output_root = tmp_path_factory.mktemp("bridge_run_perf")
    for label, hw_yaml, model_yaml, mode in POINTS:
        hw_path = os.path.join(HW_DIR, hw_yaml)
        model_path = os.path.join(MODEL_DIR, model_yaml)
        closed_form = _closed_form(hw_path, model_path, mode, output_root)
        evaluation = _dag(hw_path, model_path, mode)
        out[label] = (
            closed_form,
            evaluation,
            fws_bridge.bridge_rows(evaluation, closed_form, label),
        )
    return out


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", [point[0] for point in POINTS])
def test_the_dag_reproduces_the_closed_form_on_the_degenerate_case(bridged, label):
    _closed, _evaluation, rows = bridged[label]
    failures = [
        "%s: expected %r, measured %r" % (row.name, row.expected, row.measured)
        for row in rows
        if not row.ok
    ]
    assert not failures, "\n".join(failures)
    assert rows, "the gate produced no rows for %s" % label


@pytest.mark.parametrize("label", [point[0] for point in POINTS])
def test_the_gate_compares_discrete_quantities_exactly(bridged, label):
    """No tolerance may hide behind an integer. Cycles, arrays, chips: exact."""
    _closed, _evaluation, rows = bridged[label]
    discrete = [
        row
        for row in rows
        if any(
            token in row.name
            for token in (
                "cycles",
                "macros",
                "chips",
                "stage set",
                "layer classes",
                "bottleneck",
                "boundaries",
                "fold identity",
            )
        )
    ]
    assert discrete
    assert all(row.kind == "exact" for row in discrete), [
        row.name for row in discrete if row.kind != "exact"
    ]


def test_the_gate_covers_every_quantity_adj_8_names(bridged):
    """op/stage sets, QK/PV cycles, arrays, chips, bottleneck; then the numbers."""
    rows = bridged["MoE"][2] + bridged["T1"][2] + bridged["Llama7B"][2]
    names = " | ".join(row.name for row in rows)
    for required in (
        "stage set",
        "QK cycles",
        "PV cycles",
        "macros per dense layer",
        "macros per MoE layer",
        "total analog macros",
        "analog chips",
        "bottleneck stage",
        "period_us",
        "block latency_us",
        "end-to-end latency_us",
        "boundary 0 time_us",
        "total analog area mm2",
        "analog energy pJ",
        "decode: period_us",
    ):
        assert required in names, required


def test_a_wrong_dag_fails_the_gate(bridged):
    """The gate has teeth: perturb ONE priced op and a row must go red.

    A gate nobody can fail is a gate that checks nothing. The perturbation is
    the smallest one that matters — a single analog op running 1 % slow.
    """
    from dataclasses import replace

    closed_form, evaluation, rows = bridged["T1"]
    assert all(row.ok for row in rows)
    costs = list(evaluation.pricing.costs)
    index = next(
        i for i, cost in enumerate(costs) if cost.block == "qkv" and cost.phase == "prefill"
    )
    costs[index] = replace(costs[index], duration_s=costs[index].duration_s * 1.01)
    broken = replace(
        evaluation, pricing=replace(evaluation.pricing, costs=tuple(costs))
    )
    bad = [row for row in fws_bridge.bridge_rows(broken, closed_form, "T1") if not row.ok]
    assert bad, "a 1 % error in one analog op passed the bridge gate"


# ---------------------------------------------------------------------------
# Exclusions: disclosed, named, and bounded
# ---------------------------------------------------------------------------


def test_every_exclusion_is_named_with_its_reason():
    assert fws_bridge.EXCLUSIONS
    names = [name for name, _reason in fws_bridge.EXCLUSIONS]
    assert len(set(names)) == len(names)
    for name, reason in fws_bridge.EXCLUSIONS:
        assert name and name == name.lower().replace(" ", "_")
        # A reason short enough to be a label is not a disclosure.
        assert len(reason) > 80, name


def test_a_row_that_restricts_what_it_compares_names_the_exclusion(bridged):
    """A note on a row must cite an exclusion that exists, not prose."""
    known = {name for name, _reason in fws_bridge.EXCLUSIONS}
    seen = set()
    for _label, (_closed, _evaluation, rows) in bridged.items():
        for row in rows:
            if "exclusion" not in row.note:
                continue
            cited = row.note.split("exclusion ", 1)[1].split(":")[0].strip()
            assert cited in known, row.name
            seen.add(cited)
    # The gate must actually be exercising the exclusions it declares.
    assert {"attention_fold", "fabric_fill_drain", "energy_scope"} <= seen


def test_the_excluded_set_is_the_closed_forms_blind_spots_not_a_blanket():
    """Named, specific exclusions — never one word that covers everything."""
    text = " ".join(name for name, _reason in fws_bridge.EXCLUSIONS)
    assert "partial" not in text
    assert "approximate" not in text
    assert len(fws_bridge.EXCLUSIONS) >= 6


# ---------------------------------------------------------------------------
# The projection is a bridge artifact, never a metric (D21)
# ---------------------------------------------------------------------------


def test_the_dag_report_publishes_no_period_and_no_bottleneck(bridged):
    """One accounting per metric: the period retired with the closed form (A1).

    The bridge reconstructs a period-equivalent to compare against a report that
    assumed a steady state. If that number ever appeared in the DAG report as
    well, the artifact would carry two accountings of one metric — the exact
    AUDIT finding 4 shape.
    """
    _closed, evaluation, _rows = bridged["Llama7B"]
    document = fws_eval.report_document(evaluation)
    banned = ("period", "bottleneck", "ceiling", "sustained")
    published = [
        (metric["key"] + " " + metric["label"]).lower()
        for metric in document["evaluation"]["metrics"]
    ]
    published += [key.lower() for key in document["evaluation"]]
    for name in published:
        assert not any(word in name for word in banned), name
    # The words are allowed — and wanted — inside a basis string that explains
    # what the number is NOT. What must never exist is a published quantity.
    prose = json.dumps(document)
    assert "retired with the closed form" in prose


def test_a_stage_duration_is_the_MAX_of_its_concurrent_ops_not_the_sum(bridged):
    """The closed form's stage is concurrent hardware; the projection must agree."""
    _closed, evaluation, _rows = bridged["T1"]
    projection = fws_bridge.project(evaluation, label="T1")
    ffn1 = projection.dense_stages["S4_ffn1"]
    ops = [
        cost
        for cost in evaluation.pricing.costs
        if cost.block == "ffn1" and cost.layer == 0 and cost.phase == "prefill"
    ]
    assert ffn1.macros > 1  # otherwise the test proves nothing
    assert ffn1.duration_s == pytest.approx(max(cost.duration_s for cost in ops))
    assert ffn1.duration_s < sum(cost.duration_s for cost in ops)


def test_the_tie_break_follows_the_closed_forms_own_stage_order(bridged):
    """T1's analog stages tie at 5.12 us; both paths must name S1_qkv.

    ``pipeline_period`` takes ``max(times, key=times.get)``, which returns the
    FIRST maximum in ``all_stage_times`` insertion order. The projection builds
    that same order, so a tie cannot break to a different name on one side.
    """
    closed_form, evaluation, _rows = bridged["T1"]
    projection = fws_bridge.project(evaluation, label="T1")
    times = projection.stage_times_s
    assert list(times)[:5] == list(fws_bridge._DENSE_ORDER)
    tied = [name for name, value in times.items() if value == projection.period_s]
    assert len(tied) > 1, "the tie this test exists for is gone"
    assert projection.bottleneck == closed_form["bottleneck_stage"] == "S1_qkv"


def test_the_attention_fold_identity_holds_from_the_dags_own_op_cycles(bridged):
    """max(QK op, PV op + fill/drain, softmax op) == the closed form's fold.

    The two paths compose P2's attention law differently and the bridge refuses
    to average over that: it compares the law's outputs and reconstructs the
    fold. Recomputed here from the report, independently of fws_bridge.
    """
    for label in ("T1", "T2", "T3", "Llama7B", "MoE"):
        closed_form, evaluation, _rows = bridged[label]
        projection = fws_bridge.project(evaluation, label=label)
        stages = closed_form["stages"] or closed_form["moe_stages"]
        attention = stages["S2_attention"]
        fill_drain = projection.fill_drain_cycles
        expected = max(
            attention["qk_cycles"] + fill_drain,
            attention["sv_cycles"] + fill_drain,
            attention["softmax_cycles"],
        )
        assert projection.fold_cycles == expected == attention["fabric_total_cycles"]


def test_the_gate_reads_the_artifact_run_perf_wrote(bridged):
    """The closed-form side is a file on disk, not a second in-process call."""
    closed_form, _evaluation, _rows = bridged["T1"]
    assert closed_form["device_class"] == "fws_cim"
    # Values pinned by the existing 152-check validator; if the gate ever read
    # a re-derived object instead of the artifact, these would drift together
    # with the DAG and nobody would notice.
    assert closed_form["qk_cycles"] == 2747
    assert closed_form["sv_cycles"] == 4471
    assert closed_form["arrays_total"] == 386


# ---------------------------------------------------------------------------
# Regressions: the three upstream bugs the gate caught on its first run
# ---------------------------------------------------------------------------


def test_the_vit_head_is_priced_at_one_token_per_image(bridged):
    """P3 lowering: the ViT classification head reads the pooled token, not S.

    P2's ``endpoint_stage_times`` prices it ``analog_gemm_time(B)``. Lowering it
    over the whole sequence charged it S times its work and moved T1's analog
    energy by one full-sequence array pass.
    """
    for label in ("T1", "T2", "T3"):
        closed_form, evaluation, _rows = bridged[label]
        head = [
            cost
            for cost in evaluation.pricing.costs
            if cost.block == "vit_head" and cost.phase == "prefill"
        ]
        assert head
        expected_us = closed_form["endpoint_stages"]["vit_head"]["time_us"]
        assert max(cost.duration_s for cost in head) * 1e6 == pytest.approx(
            expected_us, rel=1e-9
        )
        assert "m_tokens=%d" % evaluation.serving.batch in head[0].basis
        # patch_embed still sees every token: the fix is the HEAD, not both.
        patch = next(
            cost for cost in evaluation.pricing.costs if cost.block == "patch_embed"
        )
        assert "m_tokens=%d" % evaluation.serving.prefill_len in patch.basis


def test_a_routed_expert_is_priced_at_its_routed_tokens(bridged):
    """P3 lowering: top-k routing means an expert sees tokens_hot, not the batch.

    Pricing every expert at the full owner batch charged the MoE layer E/top_k
    times its analog work. P2's ``moe_tokens_hot`` is the count both paths use.
    """
    _closed, evaluation, _rows = bridged["MoE"]
    device = evaluation.mapping.device
    owner_tokens = evaluation.serving.batch * evaluation.serving.prefill_len
    hot = device.moe_tokens_hot(owner_tokens)
    assert hot < owner_tokens  # otherwise the test proves nothing
    routed = [
        cost
        for cost in evaluation.pricing.costs
        if cost.block == "ffn1_routed" and cost.phase == "prefill"
    ]
    shared = [
        cost
        for cost in evaluation.pricing.costs
        if cost.block == "ffn1_shared" and cost.phase == "prefill"
    ]
    assert routed and shared
    assert all("m_tokens=%d" % hot in cost.basis for cost in routed)
    # A SHARED expert still sees every token — the distinction is the point.
    assert all("m_tokens=%d" % owner_tokens in cost.basis for cost in shared)
    # And the dispatch bytes stay on the aggregate: one expert's share is not
    # what crosses the link.
    annotations = {a.uid: a for a in annotations_of(evaluation.program)}
    dispatch = [
        annotations[cost.uid]
        for cost in evaluation.pricing.costs
        if annotations[cost.uid].boundary_id.startswith("ep.dispatch")
    ]
    for annotation in dispatch:
        assert annotation.tokens == owner_tokens


def test_prefill_attention_uses_the_gqa_score_call(bridged):
    """P4 pricing: a GQA stack's query heads must not vanish into the fold.

    ``attention_timing`` is the MHA-only view (m = S); the GQA score call is
    ``m = S * shared_heads``. They coincide at kv_heads == num_heads, so the
    dense-LLM point could never have caught this — the MoE point (16 heads over
    4 KV groups) undercharged attention by exactly 4x.
    """
    _closed, evaluation, _rows = bridged["MoE"]
    device = evaluation.mapping.device
    assert device.params.shared_heads > 1  # otherwise the test proves nothing
    qk = next(
        cost
        for cost in evaluation.pricing.costs
        if cost.block == "attention_qk" and cost.phase == "prefill"
    )
    expected = device.prefill_attention_timing(
        seq_len=evaluation.serving.prefill_len, tp=1, streams=evaluation.serving.batch
    )
    mha_view = device.attention_timing(
        seq_len=evaluation.serving.prefill_len, tp=1, streams=evaluation.serving.batch
    )
    # shared_heads more query rows, so shared_heads more systolic tile passes
    # (exactly, up to the law's single trailing -1 cycle per call).
    assert expected.qk_cycles == pytest.approx(
        mha_view.qk_cycles * device.params.shared_heads, rel=1e-4
    )
    assert int(qk.detail["cycles"]) == expected.qk_cycles + int(
        device.fabric.fill_drain_penalty_cycles
    )
    assert "prefill_attention_timing" in qk.basis
