"""QIF P3.7: the MAPPED-PATH design-space sweep (`tools/fws_qif_dse.py`).

D10 promises that allocation is "swept by the DSE". Four things can rot
silently in a sweep and every one of them is a WRONG ANSWER rather than a
crash, so the suite pins all four:

* a candidate that stops being evaluated on the real mapped path (a closed
  form, a reused timeline or a cached neighbour would all still print a
  number),
* an infeasible candidate that is dropped instead of recorded with the stage
  it died at and the refusal's own message,
* a selection or a front that stops being the rule the report prints,
* an emitted config that no longer reproduces the point the sweep selected.

The cheap vehicle is Llama2-7B on the mapped config (a sub-second candidate);
the checked-in Granite-4.0-H-Tiny demo sweep is regenerated and compared byte
for byte in one deliberately slow test, because a fixture nobody regenerates
is a screenshot.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from tests.qif_artifacts import qif_artifact

import config as config_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HW_DIR = PROJECT_ROOT / "configs" / "hardware-config"
MODEL_DIR = PROJECT_ROOT / "configs" / "model-config"
DSE_DIR = PROJECT_ROOT / "docs" / "qif" / "dse"

LLAMA_MAPPED = HW_DIR / "fws_cim_llama7b_mapped.yaml"
LLAMA_MODEL = MODEL_DIR / "llama2_7b_fws_decode_inf.yaml"
GRANITE_DSE = HW_DIR / "fws_cim_granite_tiny_dse.yaml"
GRANITE_MODEL = MODEL_DIR / "granite_4_0_h_tiny_inf.yaml"
GRANITE_SELECTED = HW_DIR / "fws_cim_granite_tiny_dse_selected.yaml"

#: The capacity x bank-depth demo sweep. FROZEN by ADJ-13: `arrays_per_chip`
#: is refused by name now (a chip is BUILT to its placement), so this sweep can
#: never be re-walked and its rows are read as history, not regenerated.
DEMO_DIR = DSE_DIR / "granite_capacity_banks_retired"
#: The Wave-D sweep it replaced (ADJ-9): frozen on the same grounds.
RETIRED_DEMO_DIR = DSE_DIR / "granite_lanes_banks_retired"
DEMO_JSON = DEMO_DIR / "dse_report.json"
DEMO_MD = DEMO_DIR / "dse_report.md"


def _load_tool():
    """Import tools/fws_qif_dse.py in-process (tools/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "fws_qif_dse_under_test", str(PROJECT_ROOT / "tools" / "fws_qif_dse.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DSE = _load_tool()


# ---------------------------------------------------------------------------
# Fixtures: the cheap mapped vehicle
# ---------------------------------------------------------------------------


def _llama_base():
    """Llama2-7B mapped, plus the two device cards a card axis needs.

    The shipped mapped config declares no `cim.cards` block, so its cards are
    SYNTHESIZED. A card knob has nowhere to be written in that file, which is
    a refusal the tool owns (tested below); a config that wants the axis
    declares the cards, exactly as this fixture does.
    """
    raw = yaml.safe_load(LLAMA_MAPPED.read_text())
    # 0176fa6 / D38: fill is meaningful only on the batch-one pipeline twin.
    raw["mapping"].pop("regime", None)
    raw["cim"]["cards"] = {
        "ctt": {"kind": "analog_macro", "device": "ctt"},
        "sa": {"kind": "digital_chiplet", "vector_lanes": 1024},
    }
    # Give this vehicle enough activation memory to exercise selection.
    # The memory-stage test independently supplies an explicit small tier.
    raw["tech_param"]["DRAM"]["size"] = "32 GB"
    return raw


def _write(tmp_path, raw, sweep, name="fws_cim_sweep.yaml"):
    raw = copy.deepcopy(raw)
    raw[DSE.DSE_BLOCK] = sweep
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def _run(tmp_path, sweep, *, raw=None, model=LLAMA_MODEL, **kwargs):
    path = _write(tmp_path, raw or _llama_base(), sweep)
    return DSE.run_sweep(
        str(path), str(model), output_dir=str(tmp_path / "out"), quiet=True, **kwargs
    )


def _evaluate_directly(raw, knobs, model_path):
    """The same point, built by hand through P3 -> lowering -> P4.

    This is the control the sweep is measured against: if a candidate's
    headline ever stops being a reading off ITS OWN timeline, this comparison
    is what notices.
    """
    import fws_eval
    import fws_mapping
    from program.fws_build import build_fws_program

    candidate_raw = DSE.apply_point(
        copy.deepcopy(raw), knobs, "ctt", "sa", "test fixture"
    )
    candidate_raw, _slots = DSE._derive_chip_capacity(
        candidate_raw, config_module.parse_config(str(model_path), "LLM"), None, DSE._hw_from_raw
    )
    converted = copy.deepcopy(candidate_raw)
    config_module.convert(converted)
    hw = config_module.HWConfig.from_dict(converted)
    model = config_module.parse_config(str(model_path), "LLM")
    mapping = fws_mapping.build_mapping(hw, model)
    evaluation = fws_eval.evaluate_fws(build_fws_program(mapping))
    return evaluation, mapping


# ---------------------------------------------------------------------------
# The sweep block: strict-keyed by the tool that owns it
# ---------------------------------------------------------------------------


def test_missing_sweep_block_is_refused_by_name(tmp_path):
    path = tmp_path / "no_sweep.yaml"
    path.write_text(yaml.safe_dump(_llama_base(), sort_keys=False))
    with pytest.raises(DSE.QifDseUsageError) as excinfo:
        DSE.run_sweep(str(path), str(LLAMA_MODEL), output_dir=str(tmp_path / "out"))
    message = str(excinfo.value)
    assert DSE.DSE_BLOCK in message
    # The message must name the axes, or the reader has to read the source.
    assert all(axis in message for axis in ("macro_y", "macro_z", "pp_util"))
    # ADJ-9: and it must NOT offer the axis it refuses.
    assert "vector_lanes" not in message


def test_unknown_key_and_unknown_axis_are_refused(tmp_path):
    with pytest.raises(DSE.QifDseUsageError) as excinfo:
        _run(tmp_path, {"axes": {"bank_depth": [1]}, "objectiv": "throughput"})
    assert "objectiv" in str(excinfo.value)

    with pytest.raises(DSE.QifDseUsageError) as excinfo:
        _run(tmp_path, {"axes": {"cache_line_bytes": [1, 2]}})
    message = str(excinfo.value)
    assert "cache_line_bytes" in message
    # An unknown axis names what IS swept and what each axis moves.
    assert "cim.cards.<analog card>.params.adcs" in message


def test_axis_values_must_be_a_non_empty_unique_list(tmp_path):
    # 0176fa6 / DECISIONS D35-REVISED, D36 and D38: only Y, Z and fill are axes.
    for values, message in [([], "NON-EMPTY"), (0.5, "NON-EMPTY"), ([1.0, 0.5, 1.0], "repeats")]:
        with pytest.raises(DSE.QifDseUsageError, match=message):
            _run(tmp_path, {"axes": {"pp_util": values}})


def test_partition_axis_is_retired_for_every_former_spelling(tmp_path):
    # 0176fa6 / DECISIONS D35-REVISED, D36 and D38: only Y, Z and fill are axes.
    for value in (8, [8, 8, 8, 8], "auto"):
        with pytest.raises(DSE.QifDseUsageError) as excinfo:
            DSE.SweepSpec.from_raw({DSE.DSE_BLOCK: {"axes": {"layers_per_chip": [value]}}}, "test")
        assert "D38" in str(excinfo.value) and "pp_util" in str(excinfo.value)


def test_objective_must_be_one_of_the_printed_rules(tmp_path):
    # D38: use a legal axis so a parked-axis error cannot mask the objective.
    with pytest.raises(DSE.QifDseUsageError, match="objective"):
        _run(tmp_path, {"axes": {"pp_util": [1.0]}, "objective": "min_energy"})
    assert set(DSE.SELECTION_RULES) == {"throughput", "min_silicon"}


def test_cross_product_is_complete_and_declaration_ordered():
    # 0176fa6 / DECISIONS D35-REVISED, D36 and D38: only Y, Z and fill are axes.
    spec = DSE.SweepSpec.from_raw(
        {DSE.DSE_BLOCK: {"axes": {"macro_z": [8, 16, 32, 64], "pp_util": [1.0, 0.5]}}}, "test"
    )
    points = list(spec.points)
    assert spec.size == 8 == len(points)
    assert points[0] == {"macro_z": 8, "pp_util": 1.0}
    assert points[1] == {"macro_z": 8, "pp_util": 0.5}
    assert points[-1] == {"macro_z": 64, "pp_util": 0.5}
    assert [p["macro_z"] for p in points] == [8, 8, 16, 16, 32, 32, 64, 64]


def test_the_engine_width_is_refused_as_an_axis_by_name():
    """ADJ-9: D31-v2 derives the engine, so sweeping it would sweep the answer.

    NEW GATE. Wave F LABELLED this axis (RETIRED_AXES) and let it run, and the
    labelled axis went on producing a checked-in artifact whose whole spread
    came from overriding the derivation. ADJ-9 closed it: the refusal is by
    name, at parse, exactly as D26 refuses a dead fold.
    """
    assert "vector_lanes" in DSE.REFUSED_AXES
    # ADJ-10 put the fabric's two composable widths in the same register.
    assert "num_arrays" in DSE.REFUSED_AXES
    assert "softmax_lanes" in DSE.REFUSED_AXES
    assert "vector_lanes" not in DSE.AXIS_TARGETS
    with pytest.raises(DSE.QifDseUsageError) as excinfo:
        DSE.SweepSpec.from_raw(
            {DSE.DSE_BLOCK: {"axes": {"vector_lanes": [512, 4096]}}}, "test"
        )
    message = str(excinfo.value)
    assert "REFUSED" in message
    assert "vector_lanes" in message
    assert "ADJ-9" in message and "DERIVED" in message
    # A DECLARED width on one machine is still a legal override (D31); what is
    # refused is making it an axis. The message says so rather than leaving a
    # reader to guess which of the two it just hit.
    assert "OVERRIDE" in message


def test_every_axis_writes_exactly_one_documented_field():
    # 0176fa6 / DECISIONS D35-REVISED, D36 and D38: only Y, Z and fill are axes.
    base = _llama_base()
    base["cim"]["cards"]["ctt"].update(
        calc_profile="george_cima_mux1",
        params={k: base["cim"]["analog"][k] for k in ("rows", "cols", "adcs", "shots_per_output")},
    )
    fields = {
        "macro_z": ("adcs", 512),
        "macro_y": ("cols", 2048),
        "pp_util": ("pp_util", 0.5),
    }
    assert set(fields) == set(DSE.AXIS_TARGETS)
    for axis, (key, value) in fields.items():
        moved = DSE.apply_point(base, {axis: value}, "ctt", "sa", "test")
        restored = copy.deepcopy(moved)
        if axis == "pp_util":
            assert moved["mapping"][key] == value
            restored["mapping"].pop(key)
        else:
            assert moved["cim"]["cards"]["ctt"]["params"][key] == value
            # Macro axes also synchronize the fallback law from the same sheet.
            restored["cim"]["cards"]["ctt"]["params"] = copy.deepcopy(base["cim"]["cards"]["ctt"]["params"])
            restored["cim"]["analog"] = copy.deepcopy(base["cim"]["analog"])
        assert restored == base, axis


def test_card_axis_is_refused_when_the_config_declares_no_card(tmp_path):
    """The shipped mapped config synthesizes its cards; the tool refuses to
    invent a card block rather than sweeping a device nobody described."""
    raw = yaml.safe_load(LLAMA_MAPPED.read_text())
    assert "cards" not in raw["cim"]
    with pytest.raises(DSE.QifDseUsageError) as excinfo:
        _run(tmp_path, {"axes": {"macro_z": [8, 16]}}, raw=raw)
    message = str(excinfo.value)
    assert "cim.cards" in message and "drop the axis" in message


def test_sweep_block_leaves_the_config_runnable_and_never_reaches_the_parser(tmp_path):
    """`mapping_dse` is a TOOL input. A config carrying it still parses (so
    run_perf runs the base point unchanged), and the emitted winner drops it."""
    path = _write(tmp_path, _llama_base(), {"axes": {"pp_util": [1.0]}})
    raw = yaml.safe_load(path.read_text())
    assert DSE.DSE_BLOCK in raw
    converted = copy.deepcopy(raw)
    config_module.convert(converted)
    hw = config_module.HWConfig.from_dict(converted)
    assert hw.device_class == "fws_cim" and hw.mapping_config is not None
    assert DSE.DSE_BLOCK not in DSE._base_without_sweep(raw)

    exit_code, payload = DSE.run_sweep(
        str(path),
        str(LLAMA_MODEL),
        output_dir=str(tmp_path / "out"),
        emit_config=str(tmp_path / "selected.yaml"),
        quiet=True,
    )
    assert exit_code == 0
    emitted = yaml.safe_load((tmp_path / "selected.yaml").read_text())
    assert DSE.DSE_BLOCK not in emitted
    assert emitted["mapping"]["pp_util"] == 1.0


# ---------------------------------------------------------------------------
# The real mapped path, per candidate
# ---------------------------------------------------------------------------


def test_every_candidate_is_priced_on_its_own_mapped_timeline(tmp_path):
    # 0176fa6 / DECISIONS D35-REVISED, D36 and D38: only Y, Z and fill are axes.
    base = _llama_base()
    exit_code, payload = _run(tmp_path, {"axes": {"pp_util": [1.0, 0.5]}}, raw=base)
    assert exit_code == 0 and payload["num_valid"] == 2
    assert payload["evaluation_path"].startswith("fws_mapping.build_mapping")
    for candidate in payload["candidates"]:
        evaluation, mapping = _evaluate_directly(base, candidate["knobs"], LLAMA_MODEL)
        assert candidate["metrics"]["tokens_per_s"] == evaluation.metric("sys.fws.tokens_per_s").value
        assert candidate["metrics"]["makespan_s"] == evaluation.makespan_s
        assert candidate["placement"]["tiles"] == mapping.summary()["tiles"]
    # D38's live fill axis must actually move the headline, so copying a
    # neighbour's evaluation cannot pass just by agreeing with itself.
    assert len({c["metrics"]["tokens_per_s"] for c in payload["candidates"]}) == 2
    assert len({c["metrics"]["makespan_s"] for c in payload["candidates"]}) == 2


def test_parked_banking_axis_is_refused_before_any_candidate_runs(tmp_path):
    # 0176fa6 / DECISIONS D35-REVISED, D36 and D38: only Y, Z and fill are axes.
    with pytest.raises(DSE.QifDseUsageError) as excinfo:
        _run(tmp_path, {"axes": {"bank_depth": [1, 2, 4]}})
    assert "D38" in str(excinfo.value) and "PARK" in str(excinfo.value)
    assert not (tmp_path / "out" / "dse_report.json").exists()


def test_allocation_alias_cannot_bypass_the_parked_bank_axis(tmp_path):
    # 0176fa6 / DECISIONS D35-REVISED, D36 and D38: only Y, Z and fill are axes.
    with pytest.raises(DSE.QifDseUsageError) as excinfo:
        _run(tmp_path, {"axes": {"bank_depth": [4], "column_sets_per_tile": [1]}})
    message = str(excinfo.value)
    assert "bank_depth" in message and "column_sets_per_tile" in message
    assert "D38" in message and "PARK" in message


# ---------------------------------------------------------------------------
# Infeasibility: recorded, never dropped
# ---------------------------------------------------------------------------


def test_infeasible_candidate_carries_its_stage_and_the_refusal_message(tmp_path):
    # D38 replaces the retired bank axis with fill. Size a real area budget
    # at the underfilled point: the fully resident store must then exceed it.
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    base = _llama_base()
    base["cim"]["cards"]["sa"]["synthesis_library"] = "22nm"
    _, probe = _run(probe_dir, {"axes": {"pp_util": [0.5]}}, raw=base)
    budget = probe["candidates"][0]["silicon"]["total_silicon_mm2"]
    exit_code, payload = _run(
        tmp_path, {"axes": {"pp_util": [1.0, 0.5]}, "max_silicon_mm2": budget}, raw=base
    )
    assert exit_code == 0
    assert payload["num_candidates"] == 2 and payload["num_valid"] == 1
    failed, feasible = payload["candidates"]
    assert feasible["ok"] and feasible["knobs"]["pp_util"] == 0.5
    assert failed["fail_stage"] == "budget"
    assert "max_silicon_mm2" in failed["fail_message"]
    for stage in ("config", "mapping", "lowering", "pricing"):
        assert failed["stages"][stage] == {"ok": True}
    assert payload["selected"]["id"] == feasible["id"]


def test_card_refusal_is_recorded_and_the_sweep_continues(tmp_path, monkeypatch):
    # D38 parks bank_depth as an INPUT. Inject a bad card at the config seam
    # for one legal fill point to retain the sweep's continuation PROPERTY;
    # the actual card parser supplies the refusal, not a mocked result.
    apply = DSE.apply_point

    def first_card_refuses(raw, point, *args, **kwargs):
        applied = apply(raw, point, *args, **kwargs)
        if point["pp_util"] == 0.5:
            applied["cim"]["cards"]["ctt"]["bank_depth"] = 3
        return applied

    monkeypatch.setattr(DSE, "apply_point", first_card_refuses)
    code, payload = _run(tmp_path, {"axes": {"pp_util": [0.5, 1.0]}})
    assert code == 0 and payload["num_candidates"] == 2 and payload["num_valid"] == 1
    failed, feasible = payload["candidates"]
    assert failed["ok"] is False
    assert failed["fail_stage"] == "config"
    assert "must divide the card's adc_mux" in failed["fail_message"]
    assert failed["stages"] == {"config": {"ok": False, "message": failed["fail_message"]}}
    assert feasible["ok"] and feasible["knobs"]["pp_util"] == 1.0
    assert feasible["stages"]["pricing"] == {"ok": True}
    assert payload["selected"]["id"] == feasible["id"]


def test_memory_verdict_is_a_stage(tmp_path):
    """An explicit small activation tier produces a measured P4.3 refusal.

    The control does not depend on a particular historical placement's high
    water. Its failure must still carry the memory stage, scope and bytes.
    """
    raw = yaml.safe_load(LLAMA_MAPPED.read_text())
    raw["tech_param"]["DRAM"]["size"] = "1 MB"
    raw["cim"]["cards"] = {
        "ctt": {"kind": "analog_macro", "device": "ctt"},
        "sa": {"kind": "digital_chiplet", "vector_lanes": 1024},
    }
    exit_code, payload = _run(tmp_path, {"axes": {"pp_util": [1.0]}}, raw=raw)
    assert exit_code == 3
    candidate = payload["candidates"][0]
    assert candidate["fail_stage"] == "memory"
    assert candidate["stages"]["pricing"] == {"ok": True}
    assert "VIOLATED" in candidate["fail_message"]
    assert "activation_sram+kv" in candidate["fail_message"]
    memory = candidate["memory"]
    # regression anchor at 3cdce5c, not a validation target; reason: the
    # checkpoint's four analog and four shared-digital chiplets each own a
    # declared activation tier; this explicit 1 MiB tier refuses all eight.
    assert memory["verdicts"] == memory["violated"] == 8
    assert memory["undeclared"] == 0
    rows = memory["violated_verdicts"]
    assert len(rows) == memory["violated"]
    assert {row["scope"] for row in rows} == {f"chip {i}" for i in range(8)}
    for row in rows:
        chip = int(row["scope"].split()[1])
        assert row["tier"] == "activation_sram+kv"
        expected_owner = (
            f"analog chiplet {chip}, tp shard 0 (backbone, analog)" if chip < 4
            else f"shared digital chiplet {chip - 4} (shared_digital, digital)"
        )
        assert row["owner"] == expected_owner
        assert row["capacity_bytes"] == 1048576
        assert row["high_water_bytes"] == (2862612480 if chip < 4 else 293994496)
        assert row["high_water_bytes"] > row["capacity_bytes"]
        assert row["ratio"] == row["high_water_bytes"] / row["capacity_bytes"]
    assert memory["worst_ratio"] == max(row["ratio"] for row in rows)
    # The metrics of an infeasible-by-memory candidate are still recorded:
    # it got a timeline, and the report prints what that timeline said.
    assert candidate["metrics"]["tokens_per_s"] > 0


def test_budget_caps_are_a_stage_of_their_own(tmp_path):
    exit_code, payload = _run(
        tmp_path, {"axes": {"pp_util": [1.0]}, "max_chips": 1}
    )
    assert exit_code == 3
    candidate = payload["candidates"][0]
    assert candidate["fail_stage"] == "budget"
    assert "max_chips" in candidate["fail_message"]
    assert candidate["stages"]["mapping"] == {"ok": True}

    exit_code, payload = _run(
        tmp_path, {"axes": {"pp_util": [1.0]}, "max_silicon_mm2": 1.0}
    )
    assert exit_code == 3
    assert payload["candidates"][0]["fail_stage"] == "budget"
    assert "max_silicon_mm2" in payload["candidates"][0]["fail_message"]


def test_all_infeasible_sweep_exits_three_and_still_writes_its_artifacts(tmp_path):
    exit_code, payload = _run(tmp_path, {"axes": {"pp_util": [1.0, 0.5]}, "max_chips": 1})
    assert exit_code == 3
    assert payload["selected"] is None and payload["front_ids"] == []
    assert payload["front_shape"] == "empty"
    out = tmp_path / "out"
    assert (out / "dse_report.json").exists()
    report = (out / "dse_report.md").read_text()
    assert "NONE — every candidate is infeasible." in report
    assert "Best violation" in report


# ---------------------------------------------------------------------------
# Front and selection
# ---------------------------------------------------------------------------


def _fake(cid, tokens, silicon, chips=1, tiles=1, slots=64, chiplets=2, uncovered=()):
    return {
        "id": cid,
        "ok": True,
        "metrics": {"tokens_per_s": tokens},
        "silicon": {
            "total_silicon_mm2": silicon,
            "analog_macro_slots": slots,
            "uncovered_terms": list(uncovered),
        },
        "placement": {
            "analog_chips": chips,
            "tiles": tiles,
            "shared_digital_chiplets": chiplets,
        },
    }


def test_pareto_front_is_two_axis_dominance_and_the_shape_is_read_off_the_data():
    spread = [_fake("a", 100.0, 10.0), _fake("b", 200.0, 20.0), _fake("c", 150.0, 30.0)]
    front = DSE.pareto_front_ids(spread)
    assert front == ["a", "b"]  # c is dominated by b on both axes
    shape, note = DSE.front_shape(spread, front)
    assert shape == "spread" and "real silicon-for-throughput trade" in note

    # Two candidates that tie on BOTH axes are both non-dominated, and that
    # is a TIE, not a trade: a knob moved neither number.
    tied = [_fake("a", 100.0, 10.0), _fake("b", 100.0, 10.0)]
    front = DSE.pareto_front_ids(tied)
    assert front == ["a", "b"]
    shape, note = DSE.front_shape(tied, front)
    assert shape == "tied" and "INDISTINGUISHABLE" in note

    flat = [_fake("a", 100.0, 10.0, tiles=7), _fake("b", 200.0, 10.0, tiles=99)]
    front = DSE.pareto_front_ids(flat)
    assert front == ["b"]
    shape, note = DSE.front_shape(flat, front, axes=("column_sets_per_tile", "bank_depth"))
    assert shape == "flat_area" and "SAME total silicon" in note
    # The note DERIVES the flatness from the two counts the accounting
    # multiplies, and it names the axes it read them over.
    assert "ENUMERATED analog macro slot count is 64" in note
    assert "shared digital chiplet count is 2" in note
    assert "column_sets_per_tile, bank_depth" in note
    # ... and it never claims a mechanism the payload contradicts. These two
    # candidates place 7 and 99 tiles; a note asserting the same tiles are
    # placed either way, or that the analog term is the model's weights,
    # would be prose that the row beside it refutes.
    assert "same tiles are placed either way" not in note
    assert "the model's weights" not in note

    # An UNCOVERED term in the accounting rides the note, so a reader cannot
    # mistake a 0 mm2 term for a measured zero.
    covered_note = DSE.front_shape(flat, front, axes=("bank_depth",))[1]
    assert "UNCOVERED" not in covered_note
    uncov = [
        _fake("a", 100.0, 10.0, uncovered=["shared digital chiplet area"]),
        _fake("b", 200.0, 10.0, uncovered=["shared digital chiplet area"]),
    ]
    note = DSE.front_shape(uncov, ["b"], axes=("bank_depth",))[1]
    assert "UNCOVERED" in note and "shared digital chiplet area" in note

    chain = [_fake("a", 100.0, 20.0), _fake("b", 200.0, 10.0)]
    front = DSE.pareto_front_ids(chain)
    assert front == ["b"]
    shape, note = DSE.front_shape(chain, front)
    assert shape == "dominated_chain"


def test_chip_split_is_intent_and_the_sweep_names_its_replacement(tmp_path):
    # 0176fa6 / DECISIONS D35-REVISED, D36 and D38: only Y, Z and fill are axes.
    with pytest.raises(DSE.QifDseUsageError) as excinfo:
        _run(tmp_path, {"axes": {"layers_per_chip": [8, 4]}})
    assert "D38" in str(excinfo.value)
    assert "pp_util" in str(excinfo.value)

def test_the_headline_key_is_one_per_sweep_and_is_named():
    """ADJ-6's headline is tokens/s at the decode terminal. A prefill-only
    workload publishes none, so the sweep falls back to the request rate off
    the SAME timeline and the report names which key it ranked on — one key
    per sweep, never two mixed inside one ordering."""
    assert DSE.HEADLINE_KEYS == ("tokens_per_s", "requests_per_s")
    decode = {"metrics": {"tokens_per_s": 6110.9, "requests_per_s": 9.8}}
    prefill = {"metrics": {"tokens_per_s": None, "requests_per_s": 2.5}}
    silent = {"metrics": {"tokens_per_s": None, "requests_per_s": None}}
    assert DSE.headline_key_of(decode) == "tokens_per_s"
    assert DSE.headline_key_of(prefill) == "requests_per_s"
    assert DSE.headline_key_of(silent) is None
    assert DSE.headline_text(decode) == "6110.9 tokens/s"
    assert DSE.headline_text(prefill) == "2.5 requests/s"
    assert DSE._headline(decode) == 6110.9
    assert DSE._headline(prefill) == 2.5


def test_selection_is_lexicographic_and_the_rule_is_printed(tmp_path):
    # 0176fa6 / DECISIONS D35-REVISED, D36 and D38: only Y, Z and fill are axes.
    rules = {
        "throughput": "lexicographic: max headline tokens/s (ADJ-6, off each candidate's own "
        "timeline), then min total silicon, then min analog chips, then min "
        "placed tiles, then first in declaration order",
        "min_silicon": "lexicographic: min total silicon, then max headline tokens/s (ADJ-6), "
        "then min analog chips, then min placed tiles, then first in declaration order",
    }
    for objective, rule in rules.items():
        exit_code, payload = _run(tmp_path, {"axes": {"pp_util": [0.5, 1.0]}, "objective": objective})
        assert exit_code == 0 and payload["num_valid"] == 2
        valid = [c for c in payload["candidates"] if c["ok"]]
        # Independent tuple ordering, including declaration order on a tie.
        if objective == "throughput":
            key = lambda pair: (-pair[1]["metrics"]["tokens_per_s"], pair[1]["silicon"]["total_silicon_mm2"], pair[0])
        else:
            key = lambda pair: (pair[1]["silicon"]["total_silicon_mm2"], -pair[1]["metrics"]["tokens_per_s"], pair[0])
        assert payload["selected_id"] == min(enumerate(valid), key=key)[1]["id"]
        assert payload["sweep"]["objective"] == objective
        assert payload["selection_rule"] == rule
        assert payload["selected_id"] in payload["front_ids"]


def test_silicon_names_its_terms_and_labels_the_uncovered_ones(tmp_path):
    exit_code, payload = _run(tmp_path, {"axes": {"pp_util": [1.0]}})
    silicon = payload["candidates"][0]["silicon"]
    slots = payload["candidates"][0]["placement"]["analog_macro_slots"]
    assert silicon["analog_macro_silicon_mm2"] == slots * silicon["macro_footprint_mm2"]
    assert silicon["total_silicon_mm2"] == (
        silicon["analog_macro_silicon_mm2"] + silicon["shared_digital_silicon_mm2"]
    )
    # The shared digital card declares no area law here; that is an ABSENT
    # law, and the artifact must say so rather than print a measured zero.
    assert silicon["shared_digital_silicon_mm2"] == 0.0
    assert any("shared digital chiplet area" in term for term in silicon["uncovered_terms"])
    assert "UNCOVERED" in silicon["basis"]


def test_report_markdown_names_the_selection_the_front_and_the_failures(tmp_path):
    _run(tmp_path, {"axes": {"pp_util": [1.0, 0.5]}, "max_chips": 1})
    report = (tmp_path / "out" / "dse_report.md").read_text()
    assert "## Selected design point" in report
    assert "## Pareto front (headline tokens/s vs total silicon)" in report
    assert "Front shape" in report
    assert "## Infeasible candidates" in report
    assert "[budget]" in report
    assert "## Disclosures" in report
    assert "nothing sampled" in report
    # Every candidate is a row, feasible or not.
    for candidate in ("c000", "c001"):
        assert "| " + candidate + " |" in report


# ---------------------------------------------------------------------------
# The emitted config and the verify round trip
# ---------------------------------------------------------------------------


def test_emit_config_and_verify_round_trip(tmp_path):
    """The emitted YAML must be a machine, and run_perf must reproduce the
    selected point from it. Both sides run the same evaluator, so anything
    but an exact match means the EMITTED CONFIG lost the candidate."""
    exit_code, payload = _run(
        tmp_path,
        {"axes": {"pp_util": [1.0, 0.5]}},
        emit_config=str(tmp_path / "selected.yaml"),
        verify=True,
    )
    assert exit_code == 0
    verify = payload["verify"]
    assert verify["pass"] is True
    names = {check["name"] for check in verify["checks"]}
    assert {"tokens_per_s", "analog_chips", "tiles", "analog_macro_slots"} <= names
    for check in verify["checks"]:
        assert check["ok"], check
        assert check["rel_err"] == 0.0
    assert verify["report_path"] == "output/LLM/fws_qif_report.json"
    # The verify run's own output tree lands INSIDE the sweep's output dir,
    # which is what keeps a sweep from clobbering the repo's output/LLM.
    assert (tmp_path / "out" / "output" / "LLM" / "fws_qif_report.json").exists()
    assert not (tmp_path / "out" / "output" / "VIT").exists()


# ---------------------------------------------------------------------------
# The checked-in demo sweep
# ---------------------------------------------------------------------------


def _normalized(payload):
    """The payload minus what is a property of the MACHINE, not the sweep.

    Wall times differ on every host and the emitted-config path differs
    between the checked-in run and a regeneration into a temp directory.
    Everything else — every metric, every count, every message — must match
    exactly, because the same inputs must produce the same artifact.
    """
    def strip(node):
        if isinstance(node, dict):
            out = {}
            for key, value in node.items():
                if key in ("wall_s", "total_wall_s", "emitted_config"):
                    continue
                if key in ("hardware_config", "model_config"):
                    # The recorded input paths are whatever the CALLER passed.
                    # The checked-in artifact was written from the repo root
                    # with relative paths; this test passes absolute ones, so
                    # that a regeneration does not depend on the process cwd.
                    # The file identity is what must match.
                    out[key] = Path(value).name
                    continue
                out[key] = strip(value)
            return out
        if isinstance(node, list):
            return [strip(item) for item in node]
        return node

    return strip(payload)


def test_checked_in_demo_sweep_is_the_sweep_george_asked_for():
    """Content gate on the FROZEN artifact (cheap): the axes, the scaling and
    the pick, read as history.

    ADJ-13 FROZE THIS SWEEP. `arrays_per_chip` is refused by name now — a chip
    is BUILT to the placement it holds — so this cross product can never be
    re-walked, and what it measured is preserved rather than regenerated. What
    it measured is worth preserving: it is the clearest statement in the repo
    of why the axis had to go.

    ADJ-9 REWRITE, AXES AND FINDING BOTH.
    OLD AXES: `vector_lanes {512,1024,2048,4096} x bank_depth {1,2}`.
    OLD FINDING: at a fixed bank depth the headline rises strictly with every
    doubling of the DECLARED lane count, sub-linearly (1.28x over 8x lanes).
    NEW AXES: `arrays_per_chip {604,640,965,1207} x bank_depth {1,2}` — both
    D31-legal, because D31-v2 derives the engine to the analog floor and
    tools/fws_qif_dse.py refuses the lane axis by name.
    NEW FINDING, and it is a stronger one: at a fixed bank depth CHIP CAPACITY
    MOVES NO THROUGHPUT AT ALL. All four capacities measure the identical
    tokens/s, and the silicon doubles across them. That is Invariant W (D27)
    drawn on a cross product: the analog macro count is the global cell floor,
    so capacity buys enumerated-but-unowned SLOTS and nothing else.

    The frozen Wave-D artifact this replaced is preserved at
    docs/qif/dse/granite_lanes_banks_retired/ — it is the +27%-for-+0.78%
    comparison ADJ-9 itself quotes, and it is labelled as a machine whose
    digital side was chosen rather than derived.
    """
    payload = json.loads(qif_artifact(DEMO_JSON).read_text())
    assert payload["sweep"]["axes"] == {
        "arrays_per_chip": [604, 640, 965, 1207],
        "bank_depth": [1, 2],
    }
    assert payload["num_candidates"] == 8 and payload["num_valid"] == 8
    assert payload["model_id"] == "Granite-4.0-H-Tiny"
    # ADJ-9: the refused axis is refused, and the payload says so at sweep
    # level rather than only in a source file.
    assert "vector_lanes" not in payload["sweep"]["axes"]
    assert "vector_lanes" in payload["sweep"]["refused_axes"]
    refusal = [
        d for d in payload["disclosures"]
        if d["constraint"] == "digital_provisioning_is_not_an_axis"
    ]
    assert len(refusal) == 1 and "ADJ-9" in refusal[0]["reason"]

    # THE FINDING: at a fixed bank depth, capacity moves the silicon and NOT
    # the headline. Every candidate derives its own engine (D31-v2), and the
    # derived width does not depend on how many empty slots a chip enumerates.
    for bank in (1, 2):
        rows = sorted(
            (c for c in payload["candidates"] if c["knobs"]["bank_depth"] == bank),
            key=lambda c: c["knobs"]["arrays_per_chip"],
        )
        assert len(rows) == 4
        rates = {c["metrics"]["tokens_per_s"] for c in rows}
        assert len(rates) == 1, rates
        lanes = {c["derived_digital"]["vector_lanes"] for c in rows}
        assert len(lanes) == 1, lanes
        silicon = [c["silicon"]["total_silicon_mm2"] for c in rows]
        assert all(b > a for a, b in zip(silicon, silicon[1:])), silicon
        # ... and the term that moves is the ANALOG one, because capacity
        # enumerates slots (D27). The digital side is constant along the row.
        digital = {
            round(c["silicon"]["shared_digital_silicon_mm2"], 9) for c in rows
        }
        assert len(digital) == 1, digital
    # The whole sweep spans 2x the silicon for ZERO extra throughput at a
    # fixed bank depth: the number that used to read "8x the lanes is
    # sub-linear" now reads "2x the capacity is flat".
    biggest = max(c["silicon"]["total_silicon_mm2"] for c in payload["candidates"])
    smallest = min(c["silicon"]["total_silicon_mm2"] for c in payload["candidates"])
    assert biggest / smallest > 1.9
    slowest = min(c["metrics"]["tokens_per_s"] for c in payload["candidates"])
    fastest = max(c["metrics"]["tokens_per_s"] for c in payload["candidates"])
    assert 1.0 < fastest / slowest < 1.02
    assert all(
        c["metrics"]["serving_regime"] == "filled_pipeline"
        and c["metrics"]["resident_streams"] == 10.0
        for c in payload["candidates"]
    )
    # ADJ-9's fixed point holds at every point of the sweep, not only on the
    # shipped machine: every stage of every candidate is analog-bound.
    for candidate in payload["candidates"]:
        derived = candidate["derived_digital"]
        assert derived["sizing_target"] == "analog_stage_time"
        assert derived["analog_bound_stages"] == derived["stages_sized"] == 10
        assert derived["vector_lanes_provenance"] == "derived-count"

    assert payload["headline_metric"] == "tokens_per_s"
    assert payload["selected"]["knobs"] == {"arrays_per_chip": 604, "bank_depth": 1}

    # The front is a real 2-point trade, and it is the BANKING axis that makes
    # it: bank_depth 2 packs a shorter analog m-pass, derives a narrower engine
    # and lands cheaper AND slower than bank_depth 1 at the same capacity.
    assert payload["front_shape"] == "spread"
    assert payload["front_ids"] == ["c000", "c001"]
    assert payload["selected_id"] in payload["front_ids"]
    cheap = next(c for c in payload["candidates"] if c["id"] == "c001")
    fast = next(c for c in payload["candidates"] if c["id"] == "c000")
    assert cheap["knobs"]["bank_depth"] == 2 and fast["knobs"]["bank_depth"] == 1
    assert cheap["silicon"]["total_silicon_mm2"] < fast["silicon"]["total_silicon_mm2"]
    assert cheap["metrics"]["tokens_per_s"] < fast["metrics"]["tokens_per_s"]
    assert (
        cheap["derived_digital"]["vector_lanes"]
        < fast["derived_digital"]["vector_lanes"]
    )
    # ... and every mm2 of THAT trade is digital: the analog term is identical
    # on the two, because Invariant W pins it to the same capacity.
    assert (
        cheap["silicon"]["analog_macro_silicon_mm2"]
        == fast["silicon"]["analog_macro_silicon_mm2"]
    )
    assert all(
        c["silicon"]["shared_digital_area_provenance"] == "composed-measured"
        for c in payload["candidates"]
    )
    assert payload["verify"]["pass"] is True
    assert all(check["ok"] for check in payload["verify"]["checks"])


def test_checked_in_selected_config_is_the_winner_and_runs_the_mapped_path():
    """The emitted machine is the selection, and it is a MAPPED config: the
    `mapping:` block is what makes run_perf price a DAG instead of the frozen
    closed form (ADJ-6).

    ADJ-9 REWRITE. OLD CLAIM: the emitted card declares `vector_lanes: 4096`.
    NEW CLAIM: it declares NO vector_lanes at all — the winning point no
    longer has a lane count to write down, because the width is derived from
    the machine's own analog m-pass every time it runs.
    """
    payload = json.loads(qif_artifact(DEMO_JSON).read_text())
    raw = yaml.safe_load(GRANITE_SELECTED.read_text())
    assert DSE.DSE_BLOCK not in raw
    assert "vector_lanes" not in raw["cim"]["cards"]["sa"]
    assert raw["cim"]["chip"]["arrays_per_chip"] == 604
    assert raw["cim"]["cards"]["ctt"]["bank_depth"] == 1
    assert "mapping" in raw
    converted = copy.deepcopy(raw)
    config_module.convert(converted)
    hw = config_module.HWConfig.from_dict(converted)
    assert hw.mapping_config is not None
    assert payload["emitted_config"].endswith("fws_cim_granite_tiny_dse_selected.yaml")


def test_the_retired_wave_d_sweep_is_frozen_and_says_so():
    """ADJ-9: the historical comparison is KEPT, and kept LABELLED.

    NEW GATE. The Wave-D sweep is the measurement ADJ-9 was adjudicated on
    (+27% tokens/s for +0.78% silicon from declaring lane widths), and
    docs/qif/atlas/fixture_frontier_granite.json is grounded on its eight rows,
    so deleting it would delete the evidence. It cannot be regenerated either:
    its config declared an axis the tool now refuses by name. A frozen artifact
    with no label is a rotting one, so the label is what this pins.
    """
    payload = json.loads(qif_artifact(RETIRED_DEMO_DIR / "dse_report.json").read_text())
    retired = payload["retired"]
    assert retired["decision"].startswith("ADJ-9")
    assert "FROZEN" in retired["status"]
    assert "REFUSES that axis by name" in retired["status"]
    assert "granite_capacity_banks" in retired["successor"]
    assert "DERIVED" in retired["read_it_as"]
    # It really is the lane sweep, and the tool really would refuse it now.
    assert payload["sweep"]["axes"] == {
        "vector_lanes": [512, 1024, 2048, 4096],
        "bank_depth": [1, 2],
    }
    with pytest.raises(DSE.QifDseUsageError):
        DSE.SweepSpec.from_raw({DSE.DSE_BLOCK: payload["sweep"]}, "retired")
    # And the comparison it is kept FOR is still readable off it.
    at_depth_one = sorted(
        (c for c in payload["candidates"] if c["knobs"]["bank_depth"] == 1),
        key=lambda c: c["knobs"]["vector_lanes"],
    )
    rates = [c["metrics"]["tokens_per_s"] for c in at_depth_one]
    silicon = [c["silicon"]["total_silicon_mm2"] for c in at_depth_one]
    assert rates[-1] / rates[0] == pytest.approx(1.273, rel=1e-2)
    assert silicon[-1] / silicon[0] == pytest.approx(1.0078, rel=1e-3)


def test_the_frozen_demo_sweep_can_never_be_re_walked(tmp_path):
    """ADJ-13 REPLACES THE REGENERATE GATE.

    This used to re-run the eight Granite candidates and compare them to the
    checked-in artifact. That gate cannot exist any more, and its absence is
    the point: the sweep's own axis is refused by name, so re-walking it is
    refused too. What is pinned instead is that the refusal happens, that it
    NAMES the axis and the reason, and that the frozen artifact is still on
    disk to be read.
    """
    assert qif_artifact(DEMO_JSON).exists(), "the frozen artifact is history and stays checked in"
    raw = yaml.safe_load(GRANITE_DSE.read_text())
    assert "arrays_per_chip" in raw[DSE.DSE_BLOCK]["axes"], (
        "the frozen sweep config is preserved verbatim, refused axis and all"
    )
    with pytest.raises(DSE.QifDseUsageError) as excinfo:
        DSE.run_sweep(
            str(GRANITE_DSE),
            str(GRANITE_MODEL),
            model_id="Granite-4.0-H-Tiny",
            output_dir=str(tmp_path / "out"),
            quiet=True,
        )
    message = str(excinfo.value)
    assert "arrays_per_chip" in message
    assert "ADJ-13" in message and "built to" in message.lower() or "derived" in message.lower()

def test_demo_config_is_a_frozen_sweep_with_retired_axes():
    # 0176fa6 / DECISIONS D35-REVISED, D36 and D38: only Y, Z and fill are axes.
    # ADJ-13 froze this demo; D33 removed a privileged shipped point.
    sweep = yaml.safe_load(GRANITE_DSE.read_text())
    assert set(sweep[DSE.DSE_BLOCK]["axes"]) == {"arrays_per_chip", "bank_depth"}
    with pytest.raises(DSE.QifDseUsageError, match="REFUSED"):
        DSE.SweepSpec.from_raw(sweep, str(GRANITE_DSE))


def test_front_note_never_asserts_a_mechanism_the_candidates_refute():
    """The front note is DERIVED, not a constant string.

    On the demo sweep the two bank depths place different tile counts (19304
    and 10972 under the dense law this config declares), so a note claiming the
    same tiles are placed either way would be contradicted by the table two
    sections above it in its own report. The note may only name quantities that
    are actually constant.
    """
    payload = json.loads(qif_artifact(DEMO_JSON).read_text())
    note = payload["front_note"]
    tiles = {c["placement"]["tiles"] for c in payload["candidates"] if c["ok"]}
    assert len(tiles) > 1, "this gate is only meaningful while the tiles differ"
    assert "same tiles are placed either way" not in note
    assert "the model's weights" not in note

    # WAVE F REWRITE (D32, P7.8). OLD CLAIM: the note names the constant slot
    # count and chiplet count, and it names every uncovered silicon term —
    # both of which were clauses of the FLAT_AREA note, the only shape this
    # sweep could produce while the chiplet declared no area. NEW CLAIM: the
    # note is derived from whatever shape the front actually has, and it must
    # still never assert a mechanism the candidates refute. The flat_area
    # clauses are asserted when the front IS flat, so neither branch rots.
    shape = payload["front_shape"]
    slots = {c["silicon"]["analog_macro_slots"] for c in payload["candidates"] if c["ok"]}
    chiplets = {
        c["placement"]["shared_digital_chiplets"] for c in payload["candidates"] if c["ok"]
    }
    # On this FROZEN sweep the chiplet count is constant and the slot count is
    # not, because the sweep declared chip capacity directly and enumerating
    # slots is exactly what that did. ADJ-13 is the answer to what this shows:
    # the four capacities measure the identical throughput and differ only in
    # enumerated-but-unowned slots, so capacity was never a design choice — it
    # is arithmetic, and it derives.
    assert len(chiplets) == 1
    assert len(slots) == 4
    uncovered = payload["silicon_coverage"]["uncovered_terms"]
    if shape == "flat_area":
        assert len(slots) == 1
        assert f"slot count is {slots.pop()}" in note
        assert f"chiplet count is {chiplets.pop()}" in note
        for term in uncovered:
            assert term in note
    else:
        # A spread note may only claim a trade that the numbers contain.
        assert shape == "spread", shape
        corners = {
            (
                round(c["metrics"]["tokens_per_s"], 12),
                round(c["silicon"]["total_silicon_mm2"], 12),
            )
            for c in payload["candidates"]
            if c["id"] in payload["front_ids"]
        }
        assert len(corners) > 1, "a 'spread' note over one corner would be a tie"
        assert f"{len(payload['front_ids'])} non-dominated points" in note
        assert f"{len(corners)} distinct" in note
        assert not uncovered, (
            "the composed silicon accounting covers every term on this sweep; if a "
            "term goes uncovered again the note has to name it"
        )


def test_banking_moves_active_column_sets_and_not_wasted_columns():
    """The MECHANISM behind the banking win, pinned to the numbers.

    ADJ-4 prices ACTIVE column sets. A finer bank therefore cuts the ENERGY of
    the analog arrays; it does NOT recover stranded columns, and it does not
    buy a single mm2 of silicon. The unowned-column census and the enumerated
    SLOT count are identical across the whole sweep, so any prose crediting the
    energy win to wasted columns or to a smaller machine is refuted here.

    P7.9 REWRITE. OLD CLAIM: the MACRO COUNT is identical across the sweep too.
    NEW CLAIM: it is not, and that is Invariant W working. That sweep ran under
    the DEDICATED law, where every tensor rounds up to its own whole macros and
    a finer bank changes nothing; under the DENSE law this config now declares
    (D27), a finer bank lets one macro's banks hold blocks of DIFFERENT tensors,
    so bank_depth 1 places 4828 macros against bank_depth 2's 5488. WHY THE
    ENERGY CLAIM SURVIVES ANYWAY: the silicon a point buys is its enumerated
    SLOTS, not the macros a mapping fills, so the two wins stay separate —
    energy from the active-column-set law, macros from Invariant W — and
    neither is the other's cause.

    ADJ-9 REWRITE. OLD CLAIM: the enumerated slot count and the unowned-column
    census are constant across the WHOLE sweep, and the silicon moves with
    vector_lanes. NEW CLAIM: both censuses are constant across the BANK DEPTHS
    at a fixed capacity, and they move with the capacity axis that replaced the
    lane axis — which is the same statement about banking, scoped to the axis
    that actually holds capacity still.
    """
    payload = json.loads(qif_artifact(DEMO_JSON).read_text())
    valid = [c for c in payload["candidates"] if c["ok"]]
    at_capacity = [c for c in valid if c["knobs"]["arrays_per_chip"] == 604]
    assert len(at_capacity) == 2
    assert len({c["placement"]["unowned_columns"] for c in at_capacity}) == 1
    assert len({c["placement"]["analog_macro_slots"] for c in at_capacity}) == 1
    assert len({c["silicon"]["total_silicon_mm2"] for c in valid}) > 1, (
        "the silicon still moves across this sweep — with arrays_per_chip, and "
        "only marginally with the bank depth"
    )
    # The macro count moves with the BANK DEPTH and with nothing else.
    by_depth = {}
    for cand in valid:
        by_depth.setdefault(cand["knobs"]["bank_depth"], set()).add(
            cand["placement"]["macros_holding_tiles"]
        )
    assert {depth: sorted(v) for depth, v in by_depth.items()} == {
        1: [4828],
        2: [5488],
    }
    # Fewer macros is LESS WASTE, measured in cells (D27), not fewer slots.
    waste = {c["knobs"]["bank_depth"]: c["packing"]["waste_pct"] for c in valid}
    assert waste[1] < waste[2]

    def component(cand, name):
        return next(
            e["energy_pj"] for e in cand["metrics"]["energy_components"]
            if e["component"] == name
        )

    bank1 = next(
        c for c in valid if c["knobs"] == {"arrays_per_chip": 604, "bank_depth": 1}
    )
    bank2 = next(
        c for c in valid if c["knobs"] == {"arrays_per_chip": 604, "bank_depth": 2}
    )
    # ADJ-9 REWRITE. OLD CLAIM: the WHOLE energy delta between the two bank
    # depths is the analog arrays. NEW CLAIM: it is two terms, and the second
    # one is a consequence of ADJ-9 rather than a defect. A finer bank shortens
    # the analog m-pass a stage spends, and under D31-v2 that m-pass IS the
    # engine's sizing target — so bank_depth 1 derives a WIDER engine (6509
    # lanes against 6125), whose composed power is higher. The links still do
    # not move, and the analog term still moves the way the active-column-set
    # law says. The two terms have opposite signs and the total is their sum.
    assert component(bank1, "link_traffic") == component(bank2, "link_traffic")
    assert component(bank1, "analog_arrays") < component(bank2, "analog_arrays")
    assert component(bank1, "shared_digital_chiplet") > component(
        bank2, "shared_digital_chiplet"
    )
    assert (
        bank1["derived_digital"]["vector_lanes"]
        > bank2["derived_digital"]["vector_lanes"]
    )
    delta = bank2["metrics"]["total_energy_pj"] - bank1["metrics"]["total_energy_pj"]
    array_delta = component(bank2, "analog_arrays") - component(bank1, "analog_arrays")
    engine_delta = component(bank2, "shared_digital_chiplet") - component(
        bank1, "shared_digital_chiplet"
    )
    assert abs(delta - (array_delta + engine_delta)) < 1e-3
    # The analog term is still the DOMINANT half of the move; the derived
    # engine's is a 6.5% counterweight, not a wash.
    assert abs(engine_delta) < 0.1 * abs(array_delta)
    # A finer bank makes MORE tiles, each narrower — not the same tiles.
    assert bank1["placement"]["tiles"] > bank2["placement"]["tiles"]


def test_an_unrankable_sweep_still_writes_both_artifacts(tmp_path, monkeypatch):
    """The docstrings promise the artifacts are always written once evaluation
    has started. A sweep whose valid candidates publish different headline
    keys used to RAISE after every candidate had been priced, throwing the
    whole run away. It now records the conflict, selects nothing, writes both
    reports and exits 2."""
    real = DSE.headline_key_of
    calls = {"n": 0}

    def alternating(candidate):
        calls["n"] += 1
        return real(candidate) if calls["n"] % 2 else "requests_per_s"

    monkeypatch.setattr(DSE, "headline_key_of", alternating)
    exit_code, payload = _run(tmp_path, {"axes": {"pp_util": [1.0, 0.5]}})
    assert exit_code == 2
    assert payload["selected"] is None and payload["front_ids"] == []
    assert payload["front_shape"] == "unrankable"
    assert payload["headline_metric"] is None
    assert "DIFFERENT headline metrics" in payload["headline_conflict"]
    assert payload["num_valid"] == 2  # both were priced; nothing was dropped
    out = tmp_path / "out"
    assert (out / "dse_report.json").exists()
    report = (out / "dse_report.md").read_text()
    assert "UNRANKABLE" in report
    assert all(c["id"] in report for c in payload["candidates"])


def test_disclosures_are_one_entry_per_constraint_and_name_their_candidates():
    """One constraint, one row — per candidate and for the sweep.

    A candidate's list is the evaluator's plus the mapping's, and both carry
    several of the same constraints; printing one twice reads as two
    findings. The sweep-level union keeps the first value, so it must also
    name which candidates carried it and flag any candidate that disagrees —
    otherwise a per-candidate quantity is presented as a sweep-level one.
    """
    payload = json.loads(qif_artifact(DEMO_JSON).read_text())
    for cand in payload["candidates"]:
        keys = [item["constraint"] for item in cand["disclosures"]]
        assert len(keys) == len(set(keys)), (cand["id"], keys)
    union = payload["disclosures"]
    keys = [item["constraint"] for item in union]
    assert len(keys) == len(set(keys))
    valid_ids = [c["id"] for c in payload["candidates"] if c["ok"]]
    for item in union:
        assert item["candidates"], item
        assert set(item["candidates"]) <= set(valid_ids)
        # A constraint CAN legitimately differ between candidates — P7.3's
        # per-device-class utilization does, because provisioning is what this
        # sweep moves. What must never happen is the difference being merged
        # away: every disagreeing candidate is named beside the entry, with
        # its own value and reason, which is what stops a per-candidate
        # quantity from being read as a sweep-level one.
        for variant in item.get("varies_by_candidate", ()):
            assert variant["id"] in valid_ids, variant
            assert variant["value"] and variant["reason"], variant
    utilization = [i for i in union if i["constraint"] == "device_class_utilization"]
    assert len(utilization) == 1, "D28's utilization finding is on every candidate"
    assert utilization[0]["varies_by_candidate"], (
        "the candidates provision different digital widths, so their utilization "
        "differs; the union entry must carry the differences rather than present "
        "one candidate's numbers as the sweep's"
    )


@pytest.fixture
def frozen_demo_markdown(tmp_path):
    # The Markdown was never committed. Rendering the archived JSON does not
    # rerun its refused axes, and the committed renderer is deterministic.
    payload = json.loads(qif_artifact(DEMO_JSON).read_text())
    rendered = DSE.render_markdown(payload)
    assert rendered == DSE.render_markdown(payload)
    path = tmp_path / "retired_dse_report.md"
    path.write_text(rendered)
    return path


def test_markdown_carries_the_silicon_accountings_coverage(frozen_demo_markdown):
    """D21 wants the relaxation disclosed in the ARTIFACT, and the MD is the
    artifact a human reads. A `0` shared-digital term is an ABSENT law, not a
    measured zero, and the MD must say so on every front shape — not only
    when the flat_area note happens to mention it."""
    report = frozen_demo_markdown.read_text()
    assert report.startswith("# QIF P3.7 — mapped-path DSE report")
    payload = json.loads(qif_artifact(DEMO_JSON).read_text())
    assert "silicon_accounting" in report
    uncovered = payload["silicon_coverage"]["uncovered_terms"]

    # WAVE F REWRITE (D32, P7.8). OLD CLAIM: the MD always carries a
    # `silicon_uncovered` line saying the 0 mm2 shared-digital term is an
    # ABSENT law and not a measured zero. That claim depended on the term
    # being absent. NEW CLAIM: the MD states the accounting's PROVENANCE on
    # every run — an uncovered term is still named as an absent law, and a
    # covered one names the measured library it was composed from. What is
    # forbidden is a silent number, in either direction.
    if uncovered:
        assert "silicon_uncovered" in report
        assert "not a measured zero" in report
        for term in uncovered:
            assert term in report
    else:
        assert "silicon_uncovered" not in report
        assert "COMPOSED area" in report
        assert "D32" in report
    assert payload["selected"]["silicon"]["basis"] in report
