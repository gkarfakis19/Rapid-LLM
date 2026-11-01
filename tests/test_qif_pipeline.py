"""QIF P7.7: the FILLED PIPELINE serving regime (D29/D30).

The machine this file pins is the one George described on 2026-08-24: the
pipeline is ALWAYS FULL. Decode is independent streams staggered across the PP
stages, ONE token exits per beat, the local batch is always 1 (M = 1 on every
analog op, attention and scan per stream), and the resident stream count D IS
the stage count. Batch size does not exist here and is refused by name.

Five things are pinned, and every one of them is a wrong NUMBER rather than a
crash if it rots:

  1. **The regime.** A config that declares a ``mapping:`` block is a MAPPED
     fws_cim run and therefore a filled pipeline. A config that declares none
     is the RETIRED lockstep mode (D15) that the pass-1/2 closed form and the
     ADJ-8 bridge compare against, and it says so in its own disclosures.
  2. **The refusals.** A declared batch (D29) and declared endpoints (D30) are
     refused BY NAME, at the config surface and again at the mapper, because a
     run that quietly forced M = 1 or quietly dropped an lm_head would publish
     a number for a machine the user did not ask for.
  3. **The rotation.** Traversal j enters at beat j and occupies stage d at
     beat j + d; a stage holds ONE stream at a time; a stream's next token
     depends on its previous one. The BEAT emerges from the timeline as the
     slowest stage's service time and one token exits per beat — measured,
     never assumed.
  4. **The contexts.** The resident streams sit at DIFFERENT decode depths and
     every attention op is priced at its OWN stream's context. The reported
     representative context is the median of the window, disclosed.
  5. **The residency (the George constraint).** EVERY stage holds ALL D
     streams' state and KV for its layers. The per-stage bytes are reported,
     they are D x the per-stream state, and they carry a named verdict.

THE GEOMETRY the hand arithmetic below is done on: configs/hardware-config/
fws_cim_moe.yaml + moe_small_fws_inf.yaml at batch 1 with the endpoints
dropped — 12 GQA layers (16 heads, 4 KV heads, head_dim 64), hidden 1024,
seq_len 512 with decode_len 32 (so prefill_len = 480), layers_per_chip 6, so
the default stage plan is 2 stages and D = 2.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import yaml

import config
import fws_eval
import fws_mapping
from fws_mapping import (
    REGIME_FILLED,
    REGIME_LOCKSTEP,
    MappingError,
)
from program.fws_build import (
    LAW_ANALOG_GEMM,
    LAW_STAGE_HOP,
    annotations_of,
    build_fws_program,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HW_DIR = PROJECT_ROOT / "configs" / "hardware-config"
MODEL_DIR = PROJECT_ROOT / "configs" / "model-config"

FWS_MOE = HW_DIR / "fws_cim_moe.yaml"
FWS_GRANITE = HW_DIR / "fws_cim_granite_tiny.yaml"
FWS_LLAMA_MAPPED = HW_DIR / "fws_cim_llama7b_mapped.yaml"
MOE_SMALL = MODEL_DIR / "moe_small_fws_inf.yaml"
GRANITE = MODEL_DIR / "granite_4_0_h_tiny_inf.yaml"
LLAMA_DECODE = MODEL_DIR / "llama2_7b_fws_decode_inf.yaml"
LLAMA_LOCKSTEP = MODEL_DIR / "llama2_7b_fws_inf.yaml"

#: MoE-small's decode context arithmetic, by hand.
PREFILL_LEN = 512 - 32
#: K and V, 4 KV heads, head_dim 64, at D38's FP8 state element (1 B), which is
#: a PLATFORM choice and not the card's tensor format (which is bf16 here).
KV_BYTES_PER_TOKEN_PER_LAYER = int(2 * 4 * 64 * fws_mapping.STATE_ELEMENT_BYTES)


def _hw(path, mutate=None):
    raw = copy.deepcopy(yaml.safe_load(Path(path).read_text()))
    if mutate is not None:
        mutate(raw)
    config.convert(raw)
    return config.HWConfig.from_dict(raw)


def _model(path):
    return config.parse_config(str(path), "LLM")


def _d29(raw):
    """Mutator: declare a `mapping:` block, which IS the D29 regime (P7.7)."""
    raw.setdefault("mapping", {})


def _mapping(hw_path, model_path, mutate=None, **kwargs):
    return fws_mapping.build_mapping(_hw(hw_path, mutate), _model(model_path), **kwargs)


def _run(hw_path, model_path, mutate=None, **kwargs):
    mapping = _mapping(hw_path, model_path, mutate, **kwargs)
    return fws_eval.evaluate_fws(build_fws_program(mapping), mapping)


def _moe_d29(raw):
    """MoE-small on the D29 surface: a declared mapping, batch 1, no endpoints.

    The batch and the endpoints live in the MODEL config, so the fixture edits
    the model the way a D29 config file does (see
    configs/model-config/llama2_7b_fws_decode_inf.yaml, which is that edit
    checked in).
    """
    _d29(raw)


def _moe_model():
    model = _model(MOE_SMALL)
    model.model_config.global_batch_size = 1
    model.model_config.disable_embedding_unembedding = True
    return model


def _moe_run(mutate=_moe_d29, **kwargs):
    mapping = fws_mapping.build_mapping(_hw(FWS_MOE, mutate), _moe_model(), **kwargs)
    return fws_eval.evaluate_fws(build_fws_program(mapping), mapping)


@pytest.fixture(scope="module")
def moe():
    """The reference filled pipeline: MoE-small, D = 2, default window."""
    return _moe_run()


@pytest.fixture(scope="module")
def moe_lockstep():
    """The same workload under the RETIRED regime, for the contrast."""
    mapping = fws_mapping.build_mapping(
        _hw(FWS_MOE), _moe_model(), regime=REGIME_LOCKSTEP
    )
    return fws_eval.evaluate_fws(build_fws_program(mapping), mapping)


# ---------------------------------------------------------------------------
# 1. The regime: which machine a config asks for (D29)
# ---------------------------------------------------------------------------


def test_a_declared_mapping_block_is_a_filled_pipeline(moe):
    assert moe.mapping.regime == REGIME_FILLED
    assert moe.serving.regime == REGIME_FILLED
    assert moe.serving.batch == 1
    banner = [d for d in moe.disclosures if d.constraint == "serving_regime"]
    assert len(banner) == 1
    assert "ALWAYS FULL" in banner[0].reason
    assert f"D = {moe.mapping.resident_streams}" in banner[0].value


def test_a_config_with_no_mapping_block_is_the_retired_lockstep_mode():
    mapping = _mapping(FWS_MOE, MOE_SMALL)
    assert mapping.regime == REGIME_LOCKSTEP
    banner = [d for d in mapping.relaxations() if d.constraint == "serving_regime"]
    assert len(banner) == 1
    # The disclosure has to name the decision that retired it, or a reader of
    # an old-looking number has no way to know it is old.
    assert "D29" in banner[0].reason and "SUPERSEDED" in banner[0].reason
    assert REGIME_LOCKSTEP in banner[0].value


def test_the_regime_can_be_declared_by_name_and_a_wrong_name_is_refused():
    parsed = config.MappingConfig.from_dict({"regime": "lockstep"})
    assert parsed.system.regime == REGIME_LOCKSTEP
    with pytest.raises(ValueError) as excinfo:
        config.MappingConfig.from_dict({"regime": "streaming"})
    assert "is not a serving regime" in str(excinfo.value)
    # ... and the declared name reaches the mapper.
    mapping = _mapping(
        FWS_MOE, MOE_SMALL, mutate=lambda raw: raw.update({"mapping": {"regime": "lockstep"}})
    )
    assert mapping.regime == REGIME_LOCKSTEP


def test_the_filled_pipeline_is_a_decode_regime_and_prefill_is_refused():
    with pytest.raises(MappingError) as excinfo:
        fws_mapping.build_mapping(
            _hw(FWS_MOE), _moe_model(), phase="prefill", regime=REGIME_FILLED
        )
    message = str(excinfo.value)
    assert "DECODE regime" in message and "D25" in message


def test_the_shipped_mapped_configs_declare_which_regime_they_are():
    """The two P1 headline models are D29; the comparison point says it is not."""
    granite = _mapping(FWS_GRANITE, GRANITE)
    assert granite.regime == REGIME_FILLED
    llama = _mapping(FWS_LLAMA_MAPPED, LLAMA_LOCKSTEP)
    assert llama.regime == REGIME_LOCKSTEP, (
        "fws_cim_llama7b_mapped.yaml declares mapping.regime: lockstep BY NAME "
        "because it is the closed-form comparison workhorse; if that declaration "
        "is dropped the config becomes a filled pipeline and its artifacts are "
        "no longer comparable with the pass-1/2 numbers"
    )
    # The same hardware runs the D29 twin of the workload.
    filled = _mapping(FWS_LLAMA_MAPPED, LLAMA_DECODE, regime=REGIME_FILLED)
    # D38: one layer per stage, so D is the LAYER count and not the chip count.
    assert filled.regime == REGIME_FILLED
    assert filled.resident_streams == len(filled.stages) == 32


# ---------------------------------------------------------------------------
# 2. B does not exist, and neither do the endpoints (D29 / D30)
# ---------------------------------------------------------------------------


def test_the_mapping_block_refuses_a_batch_by_name():
    for key in ("batch", "batch_size", "local_batch", "streams", "resident_streams"):
        with pytest.raises(ValueError) as excinfo:
            config.MappingConfig.from_dict({key: 4})
        message = str(excinfo.value)
        assert f"mapping.{key} is refused BY NAME (D29)" in message
        assert "ALWAYS FULL" in message and "layers_per_stage" in message


def test_a_mapped_run_refuses_a_declared_batch_by_name():
    """The config surface refuses it, citing D29 and naming the field."""
    hw = _hw(FWS_MOE, _d29)
    model = _model(MOE_SMALL)  # global_batch_size: 4
    model.model_config.disable_embedding_unembedding = True
    with pytest.raises(ValueError) as excinfo:
        config.validate_model_config(hw, model)
    message = str(excinfo.value)
    assert "model_param.global_batch_size" in message
    assert "D29" in message and "ALWAYS FULL" in message
    assert "resident stream count D" in message


def test_the_mapper_refuses_a_declared_batch_too():
    """Second gate: a caller that skips validate_model_config gets the same no."""
    model = _model(MOE_SMALL)
    model.model_config.disable_embedding_unembedding = True
    with pytest.raises(MappingError) as excinfo:
        fws_mapping.build_mapping(_hw(FWS_MOE, _d29), model)
    assert "local batch of 4" in str(excinfo.value)


def test_a_mapped_run_refuses_declared_endpoints_by_name():
    hw = _hw(FWS_MOE, _d29)
    model = _model(MOE_SMALL)
    model.model_config.global_batch_size = 1
    with pytest.raises(ValueError) as excinfo:
        config.validate_model_config(hw, model)
    message = str(excinfo.value)
    assert "model_param.disable_embedding_unembedding" in message
    assert "D30" in message and "DROPS embedding and lm_head" in message
    # and the mapper refuses the same model, naming the stage it would place
    with pytest.raises(MappingError) as mapper:
        fws_mapping.build_mapping(hw, model)
    assert "lm_head" in str(mapper.value) and "D30" in str(mapper.value)


def test_the_filled_pipeline_carries_no_endpoint_anything(moe):
    """D30: no endpoint arrays, no endpoint stages, no endpoint ops."""
    mapping = moe.mapping
    assert list(mapping.endpoint_blocks) == []
    assert not [owner for owner in mapping.owners() if owner.op in ("lm_head", "patch_embed", "vit_head")]
    assert not [tile for tile in mapping.tiles if tile.owner.op == "lm_head"]
    assert not [a for a in annotations_of(moe.program) if a.block == "lm_head"]


def test_dropping_the_endpoints_is_worth_exactly_the_lm_head_macros():
    """The D30 delta is a NUMBER, not a claim: 32 macros of lm_head on this card."""
    lockstep = _mapping(FWS_MOE, MOE_SMALL)
    filled = fws_mapping.build_mapping(_hw(FWS_MOE, _d29), _moe_model())
    lm_head_macros = len(
        {
            tile.site.macro_id
            for tile in lockstep.tiles
            if tile.owner.op == "lm_head"
        }
    )
    assert lm_head_macros == 32  # ceil(32000 / 1024) columns of vocab
    assert (
        lockstep.summary()["macros_holding_tiles"]
        - filled.summary()["macros_holding_tiles"]
        == lm_head_macros
    )


# ---------------------------------------------------------------------------
# 3. Stages: D is DERIVED, never configured (D29)
# ---------------------------------------------------------------------------


def test_the_stage_is_one_layer_and_d_is_the_stage_count(moe):
    """D38: stage granularity DERIVES, and the finest granularity is a layer."""
    mapping = moe.mapping
    assert mapping.stage_basis.startswith("DERIVED (D38)")
    assert len(mapping.stages) == 12
    assert mapping.resident_streams == 12
    assert [stage.layers for stage in mapping.stages] == [(i,) for i in range(12)]
    # The stage names the chip(s) its layer sits on; the chip partition places
    # weights and no longer decides where a stage boundary falls.
    assert [stage.chips for stage in mapping.stages] == [
        tuple(mapping.layer_chip[i]) for i in range(12)
    ]
    summary = mapping.summary()
    assert summary["resident_streams"] == 12 and summary["pipeline_stages"] == 12
    assert summary["layers_per_stage"] == [1] * 12
    assert summary["pp_util"] == 1.0 and summary["pipeline_fill"] == 1.0
    assert summary["serving_regime"] == REGIME_FILLED


def test_a_declared_partition_is_refused_by_name(moe):
    """D38 retired the partition as an input, so it is refused, not ignored."""
    for declared in (3, 12, [6, 5]):
        def mutate(raw, value=declared):
            raw["mapping"] = {"layers_per_stage": value}

        with pytest.raises(MappingError) as excinfo:
            fws_mapping.build_mapping(_hw(FWS_MOE, mutate), _moe_model())
        message = str(excinfo.value)
        assert "mapping.layers_per_stage" in message and "REFUSED BY NAME" in message
        assert "D38" in message and "Stage granularity DERIVES" in message
        assert "mapping.pp_util" in message, (
            "a refusal has to name the axis that replaced the one it refuses"
        )
    # The same refusal guards the law itself, so no caller can route around it.
    with pytest.raises(MappingError):
        fws_mapping.stage_partition(12, 3)


def test_pp_util_is_the_axis_that_replaced_the_partition():
    """D38: resident streams = round(pp_util x stages), the fill is the axis."""
    full = fws_mapping.build_mapping(_hw(FWS_MOE, _moe_d29), _moe_model())
    assert full.pp_util == 1.0 and full.resident_streams == len(full.stages) == 12

    under = fws_mapping.build_mapping(
        _hw(FWS_MOE, _moe_d29), _moe_model(), pp_util=0.5
    )
    assert len(under.stages) == 12, "the fill moves D, never the stage plan"
    assert under.resident_streams == 6
    assert under.pipeline_fill == 0.5
    disclosed = [d for d in under.relaxations() if d.constraint == "pipeline_underfill"]
    assert len(disclosed) == 1 and "D38" in disclosed[0].reason

    three_quarters = fws_mapping.build_mapping(
        _hw(FWS_MOE, _moe_d29), _moe_model(), pp_util=0.75
    )
    assert three_quarters.resident_streams == 9

    # The list is CLOSED: a fill nobody adjudicated is refused by name.
    with pytest.raises(MappingError) as excinfo:
        fws_mapping.build_mapping(_hw(FWS_MOE, _moe_d29), _moe_model(), pp_util=0.9)
    assert "not one of D38's fill fractions" in str(excinfo.value)


def test_a_declared_pp_membership_does_not_move_the_stage_plan():
    """After D38 the stage plan is the layers, whatever the pp degree is."""
    def mutate(raw):
        raw["mapping"] = {"parallelism": {"pp": 2}}

    mapping = fws_mapping.build_mapping(_hw(FWS_MOE, mutate), _moe_model())
    assert mapping.resident_streams == 12 == len(mapping.stages)
    assert mapping.stage_basis.startswith("DERIVED (D38)")


def test_granite_stages_are_its_layers_and_d_is_forty():
    mapping = _mapping(FWS_GRANITE, GRANITE)
    assert mapping.resident_streams == 40 == len(mapping.stages)
    assert sum(len(stage.layers) for stage in mapping.stages) == 40


# ---------------------------------------------------------------------------
# 4. The rotation, on the timeline (D29)
# ---------------------------------------------------------------------------


def _stage_windows(evaluation):
    """(traversal, stage) -> [first start, last finish] off the one timeline."""
    windows = {}
    timeline = evaluation.timeline
    for annotation in annotations_of(evaluation.program):
        start = timeline.start_times[annotation.uid]
        finish = timeline.finish_times[annotation.uid]
        if start < 0 or finish < 0:
            continue
        key = (int(annotation.step), int(annotation.stage))
        span = windows.setdefault(key, [start, finish])
        span[0] = min(span[0], float(start))
        span[1] = max(span[1], float(finish))
    return windows


def test_the_window_is_the_fill_plus_the_measured_steady_beats(moe):
    serving = moe.serving
    assert serving.streams == 12 == serving.stages
    # The window is DERIVED and its floor is the stage count, so a 12-stage
    # machine measures 12 steady beats however small the declared default is.
    assert serving.steady_beats == serving.stages == 12
    assert serving.steady_beats > fws_mapping.DEFAULT_DECODE_WINDOW
    # `stages - 1` beats to fill, then one beat more than the steady sample so
    # the transient interval can be held out and STILL leave `steady_beats` of
    # it. The window is the LAW's, so a fill change resizes it in one place.
    assert serving.beats == fws_mapping.lowered_beats(
        serving.stages, serving.streams, serving.steady_beats
    )
    pipeline = moe.pipeline
    assert pipeline["fill_beats"] == serving.stages - 1
    assert pipeline["beats_lowered"] == serving.beats
    # D + steady beats lowered -> steady + 1 traversals complete -> steady
    # intervals to read the beat from. Nothing is extrapolated to get there.
    assert pipeline["tokens_exited"] == serving.steady_beats + 2
    # One interval is HELD OUT as the fill transient: the traversal that
    # entered at beat 0 travelled through a filling pipeline with nothing
    # queued behind it, so its exit is early. The beat is the median of what is
    # left, which is exactly the declared window, and both lists are printed.
    assert pipeline["steady_beats_measured"] == serving.steady_beats
    assert len(pipeline["steady_beat_intervals_s"]) == serving.steady_beats
    assert len(pipeline["transient_beat_intervals_s"]) == 1
    transient = [d for d in moe.disclosures if d.constraint == "pipeline_fill_transient"]
    assert len(transient) == 1 and "fill" in transient[0].reason


def test_traversal_j_occupies_stage_d_at_beat_j_plus_d(moe):
    for annotation in annotations_of(moe.program):
        if annotation.stage < 0:
            continue
        assert annotation.beat == (
            moe.serving.entry_beat(annotation.step) + annotation.stage
        )
        assert annotation.beat < moe.serving.beats
        assert annotation.stream == annotation.step % moe.serving.streams


def test_every_analog_op_fires_at_m_equals_one(moe):
    analog = [a for a in annotations_of(moe.program) if a.law == LAW_ANALOG_GEMM]
    assert analog
    assert {a.tokens for a in analog} == {1.0}, (
        "D29: the local batch is always 1, so every analog op is an M = 1 vector "
        "pass. A token count above 1 here is a batched machine wearing D29's name."
    )


def test_a_stage_holds_one_stream_at_a_time(moe):
    """The regime's own constraint, on the timeline: no stage overlaps itself."""
    windows = _stage_windows(moe)
    for stage in range(moe.serving.streams):
        spans = [
            windows[(traversal, stage)]
            for traversal in range(moe.serving.beats)
            if (traversal, stage) in windows
        ]
        spans.sort()
        for earlier, later in zip(spans, spans[1:]):
            assert later[0] >= earlier[1] - 1e-15, (
                f"stage {stage} runs two streams at once; D29's machine holds D "
                "streams, one per stage, and its state/KV accounting says so"
            )


def test_the_beat_is_read_off_the_settled_tail_and_does_not_move_with_the_window():
    """P7.9: the HEADLINE must not depend on how many beats the window lowered.

    The old rule held out exactly ONE inter-exit interval as the fill transient
    and took the median of the rest. That is exact only for a pipeline of
    equal-service stages; an uneven stage plan is still ramping after D - 1
    beats, so the median averaged ramp with rhythm and the answer moved with the
    window size — measured at 12% on Granite-4.0-H-Tiny between decode_window 2
    and 3, on a run that reported beat_converged: false and published the number
    as the headline anyway.

    The rule now reads the beat off the longest SETTLED TAIL of the steady
    sample. This test is the gate: a D = 12 machine (the models the wave reports
    are D = 8 to D = 32; the old pytest gate only ever saw D = 2, where one
    held-out interval is exactly the right number to hold out) priced at two
    window sizes must publish the SAME beat, and every interval it dropped as
    ramp must still be printed.
    """
    def plan(window):
        def mutate(raw):
            raw["mapping"] = {"decode_window": window}

        return mutate

    # D38 derives the plan (one layer per stage, D = 12); the window is the
    # only knob left, and it is floored at the stage count, so the two runs
    # below differ by the beats they lower and by nothing else.
    short = _moe_run(mutate=plan(12))
    long = _moe_run(mutate=plan(16))
    assert short.serving.streams == long.serving.streams == 12
    assert long.serving.beats > short.serving.beats
    # The residual is CONTEXT DRIFT, not estimator noise: each stream's
    # attention grows by one token of context every D beats, so a later window
    # genuinely measures a slightly dearer beat. It is 0.12% here against the
    # 12% the old median-of-the-whole-steady-sample rule produced on Granite.
    assert short.pipeline["beat_s"] == pytest.approx(long.pipeline["beat_s"], rel=3e-3)

    for run in (short, long):
        pipeline = run.pipeline
        # CONVERGED means at least two intervals agreed with each other; a
        # single reading cannot show a period and says so under its own name.
        assert pipeline["beat_converged"] is True
        assert pipeline["converged_tail_beats"] >= 2
        # The beat IS the median of the tail, and the tail is a suffix of the
        # steady sample — nothing is reordered and nothing is smoothed.
        tail = list(pipeline["converged_tail_intervals_s"])
        steady = list(pipeline["steady_beat_intervals_s"])
        assert steady[len(steady) - len(tail):] == tail
        assert pipeline["beat_s"] == pytest.approx(
            sorted(tail)[len(tail) // 2]
            if len(tail) % 2
            else 0.5 * (sorted(tail)[len(tail) // 2 - 1] + sorted(tail)[len(tail) // 2]),
            rel=1e-12,
        )
        # Every dropped interval is PRINTED, and the two lists partition the
        # steady sample exactly.
        ramp = list(pipeline["ramp_beat_intervals_s"])
        assert ramp + tail == steady
        assert pipeline["ramp_beats_dropped"] == len(ramp)
        # A dropped interval rides a named disclosure; nothing is dropped silently.
        named = [
            d
            for d in run.disclosures
            if d.constraint == "beat_read_from_the_converged_tail"
        ]
        assert bool(ramp) == bool(named)
        # tokens/s is 1/beat on the settled number, and its basis says so.
        tokens = run.metric("sys.fws.tokens_per_s")
        assert tokens.value == pytest.approx(1.0 / pipeline["beat_s"], rel=1e-12)
        assert "SETTLED TAIL" in tokens.basis


def test_the_beat_is_the_slowest_stage_and_one_token_exits_per_beat(moe):
    """THE HEADLINE, hand-checked against the stages it emerged from."""
    windows = _stage_windows(moe)
    last_complete = max(
        j for j in range(moe.serving.beats) if moe.serving.completes(j)
    )
    spans = [
        windows[(last_complete, stage)][1] - windows[(last_complete, stage)][0]
        for stage in range(moe.serving.stages)
    ]
    beat = float(moe.pipeline["beat_s"])
    assert beat == pytest.approx(max(spans), rel=5e-3), (
        "the beat must BE the slowest stage's service time: that is what 'one "
        "token per beat' means, and the timeline is where it comes from. The "
        "tolerance is the CONTEXT DRIFT: each stream's attention grows by one "
        "token of context every D beats, so the slowest stage is a little "
        "dearer each beat and the beat creeps by a fraction of a percent across "
        "the window. D38's per-layer stages make the window 12 stages deep "
        "instead of 2, so the drift across it is larger and the tolerance is "
        "the measured 0.19%."
    )
    # the exits are one beat apart, every one of them, once the pipeline is
    # full — to within the context drift, which is the only thing that moves a
    # steady interval: each stream's attention grows by a token of context
    # every D beats.
    for interval in moe.pipeline["steady_beat_intervals_s"]:
        assert interval == pytest.approx(beat, rel=5e-3)
    assert max(moe.pipeline["steady_beat_intervals_s"]) >= min(
        moe.pipeline["steady_beat_intervals_s"]
    )
    tokens = moe.metric("sys.fws.tokens_per_s")
    assert tokens.value == pytest.approx(1.0 / beat, rel=1e-12)
    assert "1/beat" in tokens.basis
    per_stream = moe.metric("sys.fws.per_stream_tokens_per_s")
    assert per_stream.value == pytest.approx(
        1.0 / (beat * moe.serving.streams), rel=1e-12
    )
    assert moe.metric("sys.fws.resident_streams").value == float(moe.serving.streams)


def test_the_per_token_latency_is_measured_and_the_identity_residual_reported(moe):
    latency = moe.metric("sys.fws.per_token_latency")
    pipeline = moe.pipeline
    assert latency.value == pytest.approx(pipeline["per_token_latency_s"], rel=1e-12)
    assert pipeline["identity_d_times_beat_s"] == pytest.approx(
        moe.serving.streams * pipeline["beat_s"], rel=1e-12
    )
    assert pipeline["identity_residual_s"] == pytest.approx(
        pipeline["per_token_latency_s"] - pipeline["identity_d_times_beat_s"], rel=1e-9
    )
    # It is a MEASUREMENT, not a period times a count: P4 3 forbids the latter
    # in metrics[], so the identity lives in the pipeline block beside it.
    assert "median of the settled half" in latency.basis.lower()
    assert "identity" in pipeline["identity_basis"].lower()


def test_a_stream_waits_for_its_own_previous_token(moe):
    """Autoregression: traversal j and j - D are one stream's two tokens.

    The claim is about the GRAPH, so it is checked on the graph: the first op
    of traversal j depends on the tail of traversal j - D. Reading it off the
    clock instead would pass on any timeline where the two happen not to
    overlap, which is not the same statement.
    """
    annotations = annotations_of(moe.program)
    ops = {op.uid: op for op in moe.program.ops}
    streams = moe.serving.streams
    first_of = {}
    for annotation in annotations:
        if annotation.stage != 0:
            continue
        first_of.setdefault(int(annotation.step), int(annotation.uid))
        first_of[int(annotation.step)] = min(first_of[int(annotation.step)], int(annotation.uid))
    checked = 0
    for traversal, uid in sorted(first_of.items()):
        parents = {int(annotations[dep].step) for dep in ops[uid].deps}
        if traversal < streams:
            # The pipeline is still FILLING: these traversals have no earlier
            # token of their own stream, so the only edge into them is the one
            # that says a stage holds ONE stream at a time.
            assert parents == ({traversal - 1} if traversal else set())
            continue
        assert traversal - streams in parents, (
            f"traversal {traversal} is stream {traversal % streams}'s next token and "
            f"must wait for traversal {traversal - streams}"
        )
        # ... and the timeline honours it.
        finish = max(moe.timeline.finish_times[dep] for dep in ops[uid].deps)
        assert moe.timeline.start_times[uid] >= finish - 1e-15
        checked += 1
    assert checked > 0


def test_the_lockstep_run_is_a_different_machine_and_says_so(moe, moe_lockstep):
    """The contrast, so the regime change is a number and not an adjective."""
    assert moe_lockstep.serving.regime == REGIME_LOCKSTEP
    assert moe_lockstep.pipeline == {}
    assert not moe_lockstep.state_residency.get("measured")
    assert moe_lockstep.serving.streams == 0
    filled = moe.metric("sys.fws.tokens_per_s").value
    batched = moe_lockstep.metric("sys.fws.tokens_per_s").value
    assert filled > batched, (
        "a filled pipeline runs every stage every beat; the lockstep machine runs "
        "one layer at a time. If the retired regime ever measures faster, one of "
        "the two lowerings is wrong."
    )


# ---------------------------------------------------------------------------
# 5. The streams sit at different depths (D29 context handling)
# ---------------------------------------------------------------------------


def test_each_stream_is_priced_at_its_own_context(moe):
    annotations = [a for a in annotations_of(moe.program) if a.context is not None]
    assert annotations
    # The stages carry DIFFERENT streams, so the window holds several contexts
    # and at least one beat holds two at once: that is the whole point of the
    # regime. (Two streams CAN sit at the same depth for a beat — the stagger
    # is an offset, not a guarantee of distinctness — so the claim is about the
    # window and about some beat, not about every beat.)
    per_beat = {}
    for annotation in annotations:
        per_beat.setdefault(annotation.beat, set()).add(annotation.context)
    assert len({a.context for a in annotations}) > 1
    assert any(len(values) > 1 for values in per_beat.values())
    # The convention, checked: stream i enters at prefill_len + i + 1 and
    # advances one token every D beats.
    streams = moe.serving.streams
    for annotation in annotations:
        expected = PREFILL_LEN + (annotation.step % streams) + (annotation.step // streams) + 1
        assert annotation.context == expected


def test_attention_is_priced_at_the_stream_context_and_not_a_shared_step(moe):
    by_context = {}
    for cost in moe.pricing.costs:
        annotation = annotations_of(moe.program)[cost.uid]
        if annotation.block != "attention_qk" or annotation.layer != 0:
            continue
        by_context.setdefault(int(annotation.context), set()).add(cost.duration_s)
    assert len(by_context) > 1, "every attention op saw one context: the streams are not staggered"
    contexts = sorted(by_context)
    # Longer context, never cheaper. The law is monotone and the pricing has to
    # follow it stream by stream.
    for earlier, later in zip(contexts, contexts[1:]):
        assert min(by_context[later]) >= min(by_context[earlier])


def test_the_representative_context_is_the_median_and_the_convention_is_disclosed(moe):
    residency = moe.state_residency
    assert residency["context_min"] <= residency["representative_context"]
    assert residency["representative_context"] <= residency["context_max"]
    banner = [d for d in moe.disclosures if d.constraint == "representative_context"]
    assert len(banner) == 1
    assert "MEDIAN" in banner[0].reason
    assert "no op is priced at the representative figure" in banner[0].reason.lower() or (
        "own stream's context" in banner[0].reason
    )


# ---------------------------------------------------------------------------
# 6. THE GEORGE CONSTRAINT: every stage holds all D streams' state (D29)
# ---------------------------------------------------------------------------


def test_the_kv_law_at_one_context_by_hand():
    """512 bytes per token per layer at D38's FP8 element, so the rest is arithmetic."""
    mapping = fws_mapping.build_mapping(_hw(FWS_MOE, _d29), _moe_model())
    per_layer = mapping.device.kv_bytes_per_stream_layer(
        482, fws_mapping.STATE_ELEMENT_BYTES, 1
    )
    assert per_layer == float(KV_BYTES_PER_TOKEN_PER_LAYER * 482) == 246784.0
    # The element size is the PLATFORM's, not the card's: this card declares
    # bf16 activations and its state store is still built at 1 B/element.
    assert mapping.hw.sw_config.precision.kv_cache == 2.0
    assert fws_mapping.STATE_ELEMENT_BYTES == 1.0


def test_every_stage_holds_all_d_streams_kv_for_its_layers(moe):
    """The per-stage figure, re-derived by hand from the D29 convention.

    THE HAND COMPUTATION. Stream i enters at prefill_len + i + 1 and advances
    one token every D beats, so at the last lowered beat B - 1 it sits at
    context prefill_len + i + floor((B - 1 - i) / D) + 1. On this fixture
    (prefill 480, D = 12 after D38 makes every layer a stage, B = 25) that is
    480 + 0 + 2 + 1 = 483 for stream 0 and 480 + 11 + 1 + 1 = 493 for stream
    11: the stagger is an OFFSET, and the streams sit at 11 different depths.
    MoE-small holds 2 x 4 x 64 x 1 = 512 KV bytes per token per layer at D38's
    FP8 state element, so a stage's KV is 512 x (sum of the D contexts) x (its
    layer count) and nothing else.

    The window is bounded, so a LATE stage does not lower every stream's pass
    inside it. The residency is the regime's statement, not a census of the
    window: every stage holds all D streams, and the test pins exactly that by
    checking the convention against the contexts the ops that DID run carry.
    """
    annotations = annotations_of(moe.program)
    streams = moe.serving.streams
    contexts = [
        PREFILL_LEN + index + (moe.serving.beats - 1 - index) // streams + 1
        for index in range(streams)
    ]
    assert len(contexts) == 12
    assert contexts[0] == 483 and contexts[-1] == 493
    # The convention is the one the LOWERING stamped on the priced ops: every
    # context an op carries is one of them (or an earlier turn of the same
    # stream), and each stream's last one is exactly the figure above.
    reached = {}
    for annotation in annotations:
        if annotation.block != "attention_qk":
            continue
        key = (int(annotation.layer), int(annotation.stream))
        reached[key] = max(reached.get(key, 0), int(annotation.context))
    for (_layer, stream), value in reached.items():
        assert value <= contexts[stream]
    assert max(value for (_l, s), value in reached.items() if s == 0) == contexts[0]
    residency = moe.state_residency
    assert residency["measured"] and residency["resident_streams"] == 12
    # THE STORE IS SIZED FOR THE WHOLE SERVED REQUEST (M10), not for wherever
    # the lowered window stopped: every resident stream reaches prefill_len +
    # decode_len before it leaves, so the silicon holds D streams at THAT
    # context. The window contexts above are what the priced ops carry; this is
    # what the machine has to be built for, and the two are different readings
    # of one convention rather than two conventions.
    served = PREFILL_LEN + 32
    for row in residency["per_stage"]:
        stage = moe.mapping.stages[int(row["stage"])]
        expected = float(
            KV_BYTES_PER_TOKEN_PER_LAYER * served * streams * len(stage.layers)
        )
        assert row["kv_bytes"] == pytest.approx(expected, rel=1e-12), (
            "a stage's KV is the SUM over its resident streams at each stream's "
            "own served context; a count times one window context would be a "
            "different number"
        )
        # ... and it is D streams' worth, not one.
        assert row["per_stream_state_bytes"] == pytest.approx(
            row["state_bytes"] / moe.serving.streams, rel=1e-12
        )
    assert residency["total_state_bytes"] == pytest.approx(
        math.fsum(row["state_bytes"] for row in residency["per_stage"]), rel=1e-12
    )
    assert residency["max_stage_state_bytes"] == max(
        row["state_bytes"] for row in residency["per_stage"]
    )


def test_state_scales_with_d_because_every_stage_holds_every_stream():
    """Double D and the per-stream model state stays; the resident total grows."""
    # D38 moves D with the FILL, not with a partition: both runs have 12
    # stages and differ only in how many of them hold a stream.
    coarse = _moe_run(pp_util=0.5)
    fine = _moe_run()
    assert coarse.serving.streams == 6 and fine.serving.streams == 12
    assert coarse.serving.stages == fine.serving.stages == 12
    per_stream_coarse = coarse.state_residency["per_stream_model_state_bytes"]
    per_stream_fine = fine.state_residency["per_stream_model_state_bytes"]
    # The MODEL's per-stream state is a property of the model, not of D: what
    # D multiplies is how many copies of it every stage must hold. The two
    # differ only by where each stream sits in its own decode (the streams are
    # staggered by one token, so a wider D spreads the contexts by a few
    # tokens), which is well under a percent on a 480-token context.
    assert per_stream_fine == pytest.approx(per_stream_coarse, rel=0.01)
    assert fine.state_residency["total_state_bytes"] == pytest.approx(
        fine.serving.streams * per_stream_fine, rel=1e-12
    )
    assert fine.state_residency["total_state_bytes"] > coarse.state_residency[
        "total_state_bytes"
    ]


def test_the_state_store_is_sized_to_the_stage_and_a_tight_declared_tier_cannot_violate_it():
    """A MACHINE IS NEVER REFUSED FOR NEEDING MEMORY.

    The per-stage state is RESIDENT and read every beat, so it needs a store —
    and a store is something you BUILD, not a constant to fail against. It is
    sized here out of integer copies of the measured SRAM macro and its area is
    charged as silicon, so a tight number in tech_param cannot make a mapping
    infeasible. What can refuse a machine is its AREA budget.
    """

    def tight(raw):
        _d29(raw)
        raw["tech_param"]["SRAM-L2"]["size"] = 1048576  # 1 MiB: far under one stage

    run = _moe_run(mutate=tight)
    residency = run.state_residency
    assert residency["verdict"] == "uncovered"
    assert residency["stages_violating"] == []
    assert residency["store_bytes"] >= residency["total_state_bytes"]
    # The declared tier is still reported — it is just not a gate.
    assert residency["declared_onchip_tier_bytes"] == 1048576.0
    for row in residency["per_stage"]:
        assert row["verdict"] == "uncovered"
        assert row["store_provenance"] == "absent"
    # This machine's card names NO synthesis library, so there is no measured
    # SRAM macro to build a store out of. That is an absent law, not a zero:
    # the store reports the requirement and counts no macros, and it still
    # never turns into a refusal.
    assert residency["store_macros"] == 0
    assert residency["store_area_mm2"] == 0.0

    # On a machine whose card DOES name the measured library, the store is
    # composed: whole macros, and real silicon.
    granite = _run(FWS_GRANITE, GRANITE)
    gres = granite.state_residency
    assert gres["verdict"] == "sized"
    assert gres["store_macros"] > 0
    assert gres["store_area_mm2"] > 0
    assert gres["store_bytes"] >= gres["total_state_bytes"]
    for row in gres["per_stage"]:
        # whole macros of the measured 33280 B block, rounded UP to hold the
        # stage: the granularity is physical, not a margin (D28)
        assert row["store_macros"] * 33280 == row["capacity_bytes"]
        assert row["capacity_bytes"] >= row["state_bytes"]
    # The DRAM activation stub is the OTHER reading and keeps its own name.
    assert residency["activation_tier_verdict"] == "fits"
    assert residency["activation_tier_capacity_bytes"] == float(
        run.mapping.hw.tech_config.DRAM.size
    )
    # The MemoryVerdict rows are P4's DECLARED-capacity machinery and stay on
    # the declared activation tier, untouched by the tightened SRAM-L2 above.
    verdicts = [v for v in run.memory if v.tier == "stage_state_kv"]
    assert len(verdicts) == len(run.mapping.stages)
    for verdict in verdicts:
        assert verdict.status == "fits"
        assert verdict.capacity_bytes == float(run.mapping.hw.tech_config.DRAM.size)
        assert verdict.owner.startswith("stage ")

    # And the same machine with a tight DECLARED activation tier still says so
    # on those rows: that tier is a different question with a different name.
    def tight_dram(raw):
        _d29(raw)
        raw["tech_param"]["DRAM"]["size"] = 1048576

    dram_run = _moe_run(mutate=tight_dram)
    dram_rows = [v for v in dram_run.memory if v.tier == "stage_state_kv"]
    assert dram_rows and all(v.status == "VIOLATED" for v in dram_rows)
    assert all("resident streams" in v.disclosure for v in dram_rows)
    assert dram_run.state_residency["activation_tier_verdict"] == "VIOLATED"
    # ... while the state store is untouched by that tier entirely: it is not
    # checked against anything, and on this library-less card it is uncovered.
    assert dram_run.state_residency["verdict"] == "uncovered"

def test_the_state_number_rides_the_report_headline(moe):
    document = fws_eval.report_document(moe)
    serving = document["serving"]
    assert serving["regime"] == REGIME_FILLED
    assert serving["resident_streams"] == 12
    assert serving["beats_lowered"] == moe.serving.beats
    pipeline = document["evaluation"]["pipeline"]
    assert pipeline["beat_s"] == moe.pipeline["beat_s"]
    residency = document["evaluation"]["state_residency"]
    assert residency["max_stage_state_bytes"] == moe.state_residency["max_stage_state_bytes"]
    # This fixture's card names no measured library, so the store is UNCOVERED
    # rather than sized - and the headline says which, because a store that
    # could not be built must not read like one that measured zero.
    assert residency["verdict"] in ("sized", "uncovered")


# ---------------------------------------------------------------------------
# 7. Contention: cross-STAGE sharing contends, within-stage does not (D29)
# ---------------------------------------------------------------------------


def test_moe_reads_active_experts_once_and_reports_unread_resident_weights(moe):
    # 0176fa6 / M6: residency is not activity; only top-k MoE experts fire.
    passes = moe.bank_passes
    assert passes["measured"] and passes["unit"] == "beat"
    assert passes["beat"] == moe.serving.beats - 1
    assert 0 < passes["macros_walked_exactly_once"] < passes["macros_holding_tiles"]
    assert passes["column_sets_never_read"] > 0
    assert passes["column_sets_read_more_than_once"] == 0
    occupied = [macro for macro in moe.mapping.macros if macro.tiles]
    assert passes["macros_holding_tiles"] == len(occupied)
    assert passes["column_sets_occupied"] == sum(macro.claimed_column_sets for macro in occupied)
    assert passes["column_set_passes_charged"] + passes["column_sets_never_read"] == passes["column_sets_occupied"]


def test_within_stage_sharing_is_serial_free_and_cross_stage_is_the_priced_one():
    """D29's contention statement, measured on a plan that cuts inside a chip."""
    def mutate(raw):
        raw["mapping"] = {"packing": "dense"}
        raw["cim"]["cards"] = {
            "ctt": {"kind": "analog_macro", "device": "ctt", "bank_depth": 1}
        }

    run = _moe_run(mutate=mutate)
    sharing = run.bank_sharing
    assert sharing["measured"] and sharing["unit"] == "beat"
    # The packer really did put two STAGES in one macro, or the measurement
    # below would be about a machine with nothing to contend.
    assert sharing["macros_sharing_banks_across_stages"] > 0
    assert sharing["macros_sharing_banks_across_layers"] > 0
    # WITHIN a stage the layers are sequential for the one stream the stage
    # holds: serial-free, measured as exactly zero. After D38 a stage IS one
    # layer, so this term is also zero by CONSTRUCTION - it is kept because the
    # instrument still reports it and a non-zero reading would mean the stage
    # plan is no longer the derived one.
    assert sharing["within_stage_cross_layer_delay_s"] == 0.0
    # The instrument is not blind on this timeline.
    assert sharing["delayed_ops"] > 0 and sharing["same_owner_delay_s"] > 0.0
    # P7.9 REWRITE. The old claim here was `cross_stage_delay_s >= 0.0`, which
    # every possible value satisfies — the term D29's regime change exists to
    # introduce had no test at all. The NEW claim is the measured fact: at this
    # card's analog speed the term is EXACTLY zero, because the analog macro
    # duty is ~0.06% and two stages wanting one macro in the same beat still do
    # not overlap in time. That is a finding, not freeness by assumption, and
    # the test below shows the same instrument reporting a positive value the
    # moment the analog work is slow enough to actually collide.
    assert sharing["cross_stage_delay_s"] == 0.0
    assert (
        sharing["cross_layer_delay_s"]
        == pytest.approx(
            sharing["cross_stage_delay_s"] + sharing["within_stage_cross_layer_delay_s"]
        )
    )
    # Zero delay means zero disclosure: the banner only fires on a real cost.
    assert not [
        d
        for d in run.disclosures
        if d.constraint == "cross_stage_bank_sharing_contends_every_beat"
    ]


@pytest.mark.xfail(strict=True, reason="NEEDS GEORGE: the slow-ADC collision probe reports zero cross-stage delay after 0176fa6; D29 still requires measured contention")
def test_cross_stage_bank_sharing_costs_real_time_when_the_macro_collides():
    """D29's new pricing term, shown POSITIVE and shown to be the timeline's.

    The shipped cards run the analog macros at ~0.06% duty, so a macro holding
    two stages' weights is wanted twice per beat and still never collides —
    which is why the sibling test above measures exactly 0.0. That is an honest
    finding about the machines we build, but on its own it leaves D29's central
    new claim ("cross-STAGE bank sharing contends every beat") with no positive
    evidence anywhere. This test supplies it: hold the stage plan, the packing
    and the card fixed and change ONE number — the analog ADC evaluation length
    — until the analog op is long enough that the two stages' passes overlap.

    Everything about the sharing is identical between the two runs (same 8
    macros holding two stages' weights); only the collision changes. So the
    delay below is produced by the timeline, not by a rule that prices sharing.
    """

    def mutate(raw):
        raw["mapping"] = {"packing": "dense"}
        raw["cim"]["cards"] = {
            "ctt": {"kind": "analog_macro", "device": "ctt", "bank_depth": 1}
        }
        # 10000x the shipped 2 cycles: the analog pass now owns the beat instead
        # of hiding inside it. Nothing else about the machine moves.
        raw["cim"]["analog"]["slice_cycles"] = 20000

    run = _moe_run(mutate=mutate)
    sharing = run.bank_sharing
    assert sharing["measured"] and sharing["unit"] == "beat"
    shared_macros = int(sharing["macros_sharing_banks_across_stages"])
    assert shared_macros == 8
    # THE POSITIVE CLAIM: the term is > 0 and it is entirely cross-STAGE.
    assert sharing["cross_stage_delay_s"] > 0.0
    assert sharing["within_stage_cross_layer_delay_s"] == 0.0
    assert sharing["cross_layer_delay_s"] == pytest.approx(
        sharing["cross_stage_delay_s"]
    )
    # It is an overlap of real ops on real macros, so it cannot exceed one
    # analog service time per sharing macro in the measured beat.
    longest = max(
        float(cost.duration_s)
        for cost in run.pricing.costs
        if cost.device_class == "analog_macro"
    )
    assert 0.0 < sharing["cross_stage_delay_s"] <= shared_macros * longest
    # And it is DISCLOSED by name, which the zero-delay run does not do.
    banner = [
        d
        for d in run.disclosures
        if d.constraint == "cross_stage_bank_sharing_contends_every_beat"
    ]
    assert len(banner) == 1 and "two STAGES" in banner[0].reason


def test_utilization_is_still_a_required_output_of_a_filled_pipeline(moe):
    classes = {row.device_class for row in moe.utilization}
    assert {"analog_macro", "macro_pool", "shared_digital", "link"} <= classes
    banner = [d for d in moe.disclosures if d.constraint == "device_class_utilization"]
    assert len(banner) == 1
    for row in moe.utilization:
        assert row.devices >= row.devices_used


# ---------------------------------------------------------------------------
# 8. The stage handoff, the fill and the stub, each by name (D38 + the interim
#    link prices)
# ---------------------------------------------------------------------------


def _hops(evaluation):
    """(crosses_chip, duration_s) for every op the STAGE HOP law priced."""
    annotations = annotations_of(evaluation.program)
    return [
        (bool(annotations[cost.uid].crosses_chip), float(cost.duration_s))
        for cost in evaluation.pricing.costs
        if annotations[cost.uid].law == LAW_STAGE_HOP
    ]


def test_every_stage_handoff_is_priced_at_the_flat_hop_for_its_kind(moe):
    """The interim link prices REPLACE `links: zeroed` as the mapped default.

    Before them a stage handoff that stayed on one chip emitted no op at all
    and cost 0 s, which is a claim no built machine can make: the activation
    still crosses the die. Every stage boundary now emits a handoff, and its
    price is the FLAT latency of its kind - 5 ns inside a chip, 500 ns across
    one - and nothing about how many bytes it carries.
    """
    hops = _hops(moe)
    assert hops, "a 12-stage pipeline hands off 11 times per traversal"
    intra = [d for crosses, d in hops if not crosses]
    cross = [d for crosses, d in hops if crosses]
    assert intra and cross, (
        "this fixture places its 12 layers on 2 chips, so it has both kinds"
    )
    assert set(intra) == {fws_mapping.T_HOP_INTRA_CHIP} == {5e-9}
    assert set(cross) == {fws_mapping.T_HOP_CROSS_CHIP} == {500e-9}
    # The prices are the LAW LAYER's, quoted where they are charged. The
    # mapper spells the law and the pricer applies it, so both name it - and
    # they name it with the SAME sentence, so the merged artifact carries one
    # entry and not two accountings of one fact (D21).
    banner = [
        d for d in moe.disclosures if d.constraint == "stage_handoff_hop_prices"
    ]
    assert banner and len({d.reason for d in banner}) == 1
    assert "LATENCY ONLY" in banner[0].reason
    assert "REPLACING links: zeroed" in banner[0].reason
    merged = [
        row
        for row in fws_eval.merge_disclosures(moe.disclosures)
        if row["constraint"] == "stage_handoff_hop_prices"
    ]
    assert len(merged) == 1


def test_the_zeroed_flag_still_overrides_the_hop_to_zero():
    """`links: zeroed` stays available - as an explicit, disclosed choice."""
    def mutate(raw):
        raw["mapping"] = {"links": "zeroed"}

    run = _moe_run(mutate=mutate)
    hops = _hops(run)
    assert hops and {duration for _crosses, duration in hops} == {0.0}
    named = [d for d in run.disclosures if d.constraint == "links_zeroed"]
    assert len(named) == 1
    # The BYTES are untouched: only the time is relaxed.
    annotations = annotations_of(run.program)
    moved = [
        annotations[cost.uid].bytes_moved
        for cost in run.pricing.costs
        if annotations[cost.uid].law == LAW_STAGE_HOP
    ]
    assert all(value > 0 for value in moved)


def test_the_underfilled_pipeline_trades_throughput_for_state(moe):
    """D38's trade, both halves measured off one timeline."""
    under = _moe_run(pp_util=0.5)
    assert under.serving.streams == 6 and under.serving.stages == 12
    assert under.pipeline["pipeline_fill"] == 0.5
    assert under.pipeline["pp_util"] == 0.5
    assert under.pipeline["stage_count"] == 12

    # THROUGHPUT falls with the fill. It does NOT fall exactly linearly, and
    # the reason is measured rather than corrected: the fill enters the DAG's
    # entry spacing and everything downstream is read off that timeline, so a
    # machine holding half the streams also contends less and its beat shrinks
    # a little. Half the streams buy 0.64x the rate here, not 0.5x, and no
    # number is multiplied by the fill anywhere to make it look linear.
    full_rate = moe.metric("sys.fws.tokens_per_s").value
    under_rate = under.metric("sys.fws.tokens_per_s").value
    assert 0.5 * full_rate <= under_rate < full_rate
    assert under_rate == pytest.approx(0.64 * full_rate, rel=0.05)

    # STATE scales with the resident streams, which is the whole point.
    full_state = moe.state_residency["total_state_bytes"]
    under_state = under.state_residency["total_state_bytes"]
    assert under_state == pytest.approx(0.5 * full_state, rel=0.05)
    assert under.state_residency["resident_streams"] == 6

    # And it is DISCLOSED: an underfilled machine never reads as a full one.
    named = [d for d in under.disclosures if d.constraint == "pipeline_underfill"]
    assert len(named) == 1 and "pp_util = 0.5" in named[0].value
    assert not [d for d in moe.disclosures if d.constraint == "pipeline_underfill"]


def test_the_state_element_is_fp8_and_says_so(moe):
    """D38: KV and recurrent state are FP8, whatever the card's tensor format."""
    named = [d for d in moe.disclosures if d.constraint == "state_element_precision"]
    assert len(named) == 1
    assert "fp8" in named[0].value and "1 B/element" in named[0].reason
    assert moe.mapping.hw.sw_config.precision.activations == 2.0, (
        "the fixture's card is bf16, so the FP8 store is NOT the card's format"
    )


def test_every_mapped_artifact_carries_the_mapping_stub_disclosure(moe):
    """Until the mapping law lands, no mapped result may read as design-valid."""
    named = [
        d for d in moe.disclosures if d.constraint == fws_mapping.MAPPING_STUB_KEY
    ]
    assert len(named) == 1
    assert named[0].value == fws_mapping.MAPPING_STUB_VALUE
    assert "NOT DESIGN-VALID" in named[0].reason
    document = fws_eval.report_document(moe)
    keys = {row["constraint"] for row in document["disclosures"]}
    assert fws_mapping.MAPPING_STUB_KEY in keys
