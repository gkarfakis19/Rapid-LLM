"""The TEMPORARY zero-links relaxation: ``mapping.links`` (George, 2026-08-27).

WHAT IT IS. A mapped fws_cim run declares how its INTER-CHIP boundary handoffs
are priced. ``priced`` is the machine — every handoff pays the D17 p2p law, and
a stage holds its slot until the handoff lands, so a link can set the beat.
``zeroed`` prices those handoffs at 0 s: every link and boundary floor vacates
and the beat reads COMPUTE ONLY.

WHAT IT IS NOT. It is not a byte relaxation. Bytes are a fact of the mapping —
the lowering annotates them, the boundary table and the energy accounting read
them, and a zeroed run reports exactly the bytes the priced run reports. Only
the TIME is relaxed, the relaxation is named ``links_zeroed``, and that name
rides every artifact the run writes.

The five things pinned here, each a wrong NUMBER rather than a crash if it rots:

  1. **The default.** A config that names no mode gets ``priced``, and naming
     ``priced`` explicitly produces the SAME document byte for byte. The
     relaxation cannot move a shipped number by existing.
  2. **The refusal.** Any other spelling is refused BY NAME, with the legal set
     and the reason, at the config surface and again at the mapper.
  3. **The pricing.** Every inter-chip transfer costs exactly 0.0 s, and every
     COMPUTE op keeps the duration the priced run gave it.
  4. **The disclosure.** ``links_zeroed`` is on the evaluation, on the report
     document, on the atlas relaxations and on the DSE report (candidate and
     sweep level) — and on none of the priced ones.
  5. **The binding.** With free links, nothing link-shaped may be what sets the
     pace: every stage's outgoing link is 0 s, the stage floor is its compute,
     and the binding term is not a link.

THE FIXTURE is the cheapest mapped fws_cim machine the suite already has: the
MoE-small filled pipeline of tests/test_qif_pipeline.py (fws_cim_moe.yaml +
moe_small_fws_inf.yaml at batch 1 with the endpoints dropped — 12 layers,
layers_per_chip 6, 2 stages, D = 2). No new machine is invented here.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import yaml

import config
import fws_eval
import fws_mapping
from fws_mapping import LINKS_PRICED, LINKS_ZEROED, LINK_MODES, MappingError
from program.fws_build import annotations_of, build_fws_program

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HW_DIR = PROJECT_ROOT / "configs" / "hardware-config"
MODEL_DIR = PROJECT_ROOT / "configs" / "model-config"

FWS_MOE = HW_DIR / "fws_cim_moe.yaml"
MOE_SMALL = MODEL_DIR / "moe_small_fws_inf.yaml"


def _hw(mutate=None):
    raw = copy.deepcopy(yaml.safe_load(FWS_MOE.read_text()))
    # A declared `mapping:` block IS the D29 mapped run (P7.7).
    raw.setdefault("mapping", {})
    if mutate is not None:
        mutate(raw)
    config.convert(raw)
    return config.HWConfig.from_dict(raw)


def _moe_model():
    model = config.parse_config(str(MOE_SMALL), "LLM")
    model.model_config.global_batch_size = 1
    model.model_config.disable_embedding_unembedding = True
    return model


def _run(mutate=None, **kwargs):
    mapping = fws_mapping.build_mapping(_hw(mutate), _moe_model(), **kwargs)
    program = build_fws_program(mapping)
    return mapping, program, fws_eval.evaluate_fws(program, mapping)


def _links(value):
    def mutate(raw):
        raw["mapping"]["links"] = value

    return mutate


@pytest.fixture(scope="module")
def priced():
    """The machine: every inter-chip handoff pays the D17 p2p law."""
    return _run()


@pytest.fixture(scope="module")
def zeroed():
    """The same machine with its inter-chip handoffs priced at 0 s."""
    return _run(_links("zeroed"))


def _metric(evaluation, key):
    for item in evaluation.metrics:
        if item.key == key:
            return float(item.value)
    raise AssertionError(f"no metric {key!r} on this evaluation")


def _hops(program, evaluation):
    """``{boundary_id -> [(uid, bytes, seconds)]}`` for the INTER-CHIP hops."""
    costs = {cost.uid: cost for cost in evaluation.pricing.costs}
    out = {}
    for annotation in annotations_of(program):
        if annotation.kind != "transfer" or not annotation.crosses_chip:
            continue
        cost = costs[annotation.uid]
        out.setdefault(annotation.boundary_id, []).append(
            (int(annotation.uid), float(annotation.bytes_moved), float(cost.duration_s))
        )
    return out


def _bytes_in(document, path=""):
    """Every ``*bytes*`` number in a report document, keyed by its path."""
    out = {}
    if isinstance(document, dict):
        for key, value in document.items():
            here = f"{path}.{key}"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if "bytes" in str(key).lower():
                    out[here] = float(value)
            out.update(_bytes_in(value, here))
    elif isinstance(document, (list, tuple)):
        for index, value in enumerate(document):
            out.update(_bytes_in(value, f"{path}[{index}]"))
    return out


# ---------------------------------------------------------------------------
# 1. The knob: default priced, 'zeroed' spelled by name
# ---------------------------------------------------------------------------


def test_the_default_link_mode_is_priced_and_zeroed_is_spelled_by_name():
    """No shipped config names a mode, so every shipped config is PRICED."""
    assert config.MappingSystemConfig.from_dict({}, "mapping").links == LINKS_PRICED
    assert config.MappingSystemConfig().links == LINKS_PRICED
    assert (
        config.MappingSystemConfig.from_dict({"links": "priced"}, "mapping").links
        == LINKS_PRICED
    )
    assert (
        config.MappingSystemConfig.from_dict({"links": "zeroed"}, "mapping").links
        == LINKS_ZEROED
    )
    # Spelled the way the other named laws are: whitespace and case are the
    # user's typing, not a different mode.
    assert (
        config.MappingSystemConfig.from_dict({"links": " Zeroed "}, "mapping").links
        == LINKS_ZEROED
    )
    assert LINK_MODES == (LINKS_PRICED, LINKS_ZEROED)


def test_any_other_link_mode_is_refused_by_name_with_the_reason():
    """A mode this model does not have is a named refusal, never a fallback."""
    with pytest.raises(ValueError) as excinfo:
        config.MappingSystemConfig.from_dict({"links": "free"}, "mapping")
    message = str(excinfo.value)
    assert "mapping.links" in message
    assert "'free'" in message
    for legal in LINK_MODES:
        assert legal in message
    # The REASON, quoted: what each mode does and what the relaxation costs.
    assert "0 s" in message and "links_zeroed" in message and "BYTES" in message

    with pytest.raises(ValueError) as excinfo:
        config.MappingSystemConfig.from_dict({"links": 0}, "mapping")
    assert "mapping.links" in str(excinfo.value)

    # And again at the mapper, for a caller that hands the mode in directly.
    with pytest.raises(MappingError) as mapper_error:
        fws_mapping.build_mapping(_hw(), _moe_model(), links="free")
    assert "links = 'free'" in str(mapper_error.value)
    for legal in LINK_MODES:
        assert legal in str(mapper_error.value)


def test_the_priced_path_is_the_default_path_and_does_not_move(priced):
    """Declaring ``links: priced`` produces the document the default produces."""
    mapping, _program, evaluation = priced
    assert mapping.links == LINKS_PRICED
    _explicit_mapping, _explicit_program, explicit = _run(_links("priced"))
    assert _explicit_mapping.links == LINKS_PRICED
    assert json.dumps(
        fws_eval.report_document(explicit), default=str
    ) == json.dumps(fws_eval.report_document(evaluation), default=str)


# ---------------------------------------------------------------------------
# 2. The pricing: 0 s on the boundary, the bytes untouched
# ---------------------------------------------------------------------------


def test_every_inter_chip_handoff_is_priced_at_exactly_zero(priced, zeroed):
    _mapping, priced_program, priced_eval = priced
    _zmapping, zeroed_program, zeroed_eval = zeroed
    priced_hops = _hops(priced_program, priced_eval)
    zeroed_hops = _hops(zeroed_program, zeroed_eval)

    # The fixture HAS inter-chip hops, or this test proves nothing.
    assert priced_hops and sum(len(rows) for rows in priced_hops.values()) > 1
    assert sorted(priced_hops) == sorted(zeroed_hops)

    priced_seconds = sum(row[2] for rows in priced_hops.values() for row in rows)
    zeroed_seconds = sum(row[2] for rows in zeroed_hops.values() for row in rows)
    assert priced_seconds > 0.0, "the priced run must charge for its handoffs"
    assert zeroed_seconds == 0.0
    for boundary, rows in zeroed_hops.items():
        for uid, _bytes_moved, seconds in rows:
            assert seconds == 0.0, f"{boundary} op {uid} was charged {seconds} s"


def test_the_bytes_are_a_fact_and_the_zeroed_run_reports_the_priced_ones(
    priced, zeroed
):
    _mapping, priced_program, priced_eval = priced
    _zmapping, zeroed_program, zeroed_eval = zeroed
    priced_hops = _hops(priced_program, priced_eval)
    zeroed_hops = _hops(zeroed_program, zeroed_eval)

    # Per boundary, per op: the same bytes on the same ops.
    for boundary, rows in priced_hops.items():
        assert [(uid, size) for uid, size, _s in rows] == [
            (uid, size) for uid, size, _s in zeroed_hops[boundary]
        ]
    # And on the cost records, which is what the energy accounting reads.
    priced_bytes = [float(cost.bytes_moved) for cost in priced_eval.pricing.costs]
    zeroed_bytes = [float(cost.bytes_moved) for cost in zeroed_eval.pricing.costs]
    assert priced_bytes == zeroed_bytes

    # 0176fa6 / ADJ-16: removing hop latency changes concurrent memory high
    # water and derived engine demand. Traffic bytes remain the invariant;
    # memory-residency and width-dependent bytes are different quantities.
    priced_document = _bytes_in([asdict(row) for row in _mapping.boundary_table()])
    zeroed_document = _bytes_in([asdict(row) for row in _zmapping.boundary_table()])
    assert priced_document and priced_document == zeroed_document


def test_compute_pricing_is_untouched_and_all_stage_hops_are_zeroed(priced, zeroed):
    """0176fa6's flat intra-chip hops join the existing boundary relaxation."""
    _mapping, priced_program, priced_eval = priced
    _zmapping, _zprogram, zeroed_eval = zeroed
    annotations = annotations_of(priced_program)
    priced_costs = {cost.uid: cost for cost in priced_eval.pricing.costs}
    zeroed_costs = {cost.uid: cost for cost in zeroed_eval.pricing.costs}
    assert set(priced_costs) == set(zeroed_costs)
    moved = 0
    for uid, cost in priced_costs.items():
        annotation = annotations[uid]
        other = zeroed_costs[uid]
        if annotation.kind == "transfer" and (annotation.crosses_chip or cost.block == "boundary"):
            assert other.duration_s == 0.0
            moved += 1
            continue
        assert other.duration_s == cost.duration_s, (
            f"{annotation.kind} op {uid} ({cost.block}) moved from "
            f"{cost.duration_s} s to {other.duration_s} s and it is not an "
            "inter-chip handoff"
        )
        assert other.basis == cost.basis
    assert moved > 0


def test_the_zeroed_run_is_at_least_as_fast_as_the_priced_one(priced, zeroed):
    _mapping, _program, priced_eval = priced
    _zmapping, _zprogram, zeroed_eval = zeroed
    priced_tokens = _metric(priced_eval, "sys.fws.tokens_per_s")
    zeroed_tokens = _metric(zeroed_eval, "sys.fws.tokens_per_s")
    assert zeroed_tokens >= priced_tokens
    # Free links can only remove time from the beat, never add it.
    assert float(zeroed_eval.pipeline["beat_s"]) <= float(
        priced_eval.pipeline["beat_s"]
    )


# ---------------------------------------------------------------------------
# 3. The disclosure, on every artifact
# ---------------------------------------------------------------------------


def _links_zeroed_entries(items, key=lambda item: item.constraint):
    return [item for item in items if key(item) == "links_zeroed"]


def test_the_relaxation_is_named_on_every_artifact_of_a_zeroed_run(priced, zeroed):
    _mapping, _program, priced_eval = priced
    _zmapping, _zprogram, zeroed_eval = zeroed

    # The evaluation itself.
    entries = _links_zeroed_entries(zeroed_eval.disclosures)
    assert len(entries) == 1
    reason = entries[0].reason
    assert "TEMPORARY" in reason and "2026-08-27" in reason and "George" in reason
    assert "0 s" in reason
    assert "BYTES" in reason
    assert "COMPUTE ONLY" in reason
    assert "VACATES" in reason
    assert entries[0].value

    # The report document (what the runs write to disk).
    document = fws_eval.report_document(zeroed_eval)
    carried = _links_zeroed_entries(
        document["disclosures"], key=lambda item: item["constraint"]
    )
    assert len(carried) == 1
    assert carried[0]["reason"] == reason

    # The atlas relaxations (what tools/fws_emit_atlas.py hands the atlas).
    atlas = _links_zeroed_entries(
        zeroed_eval.atlas_relaxations(), key=lambda item: item["constraint"]
    )
    assert len(atlas) == 1
    assert atlas[0]["reason"] == reason

    # And NOWHERE on a priced run: an unrelaxed machine may not carry a
    # relaxation it did not take.
    assert not _links_zeroed_entries(priced_eval.disclosures)
    assert not _links_zeroed_entries(
        fws_eval.report_document(priced_eval)["disclosures"],
        key=lambda item: item["constraint"],
    )
    assert not _links_zeroed_entries(
        priced_eval.atlas_relaxations(), key=lambda item: item["constraint"]
    )


# ---------------------------------------------------------------------------
# 4. The binding: with free links, no link can set the pace
# ---------------------------------------------------------------------------


def test_no_link_binds_a_zeroed_run(zeroed):
    """A link that costs 0 s cannot be what limits the machine.

    This is asserted, not special-cased in the code: the beat law is untouched
    and simply reads the durations it is given.
    """
    _mapping, _program, evaluation = zeroed
    occupancy = evaluation.pipeline["stage_occupancy"]
    rows = occupancy["per_stage"]
    assert rows
    for row in rows:
        assert float(row["outgoing_link_s"]) == 0.0
    # The stage floor is COMPUTE ONLY: it is exactly the slowest stage's own
    # compute, with nothing added for a handoff.
    assert float(occupancy["floor_s"]) == max(
        float(row["compute_s_per_beat"]) for row in rows
    )
    # And the binding term names a compute class, never the link.
    terms = list(evaluation.binding_term.get("terms") or ())
    assert terms
    assert terms[0]["device_class"] != "link"
    for term in terms:
        if term["device_class"] == "link":
            assert float(term["busy_s"]) == 0.0


# ---------------------------------------------------------------------------
# 5. The third artifact: the DSE report
# ---------------------------------------------------------------------------


def _load_dse_tool():
    """Import tools/fws_qif_dse.py in-process (tools/ is not a package)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "fws_qif_dse_links_relaxation",
        str(PROJECT_ROOT / "tools" / "fws_qif_dse.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _one_point_sweep(tmp_path, links):
    """A ONE-CANDIDATE sweep over the shipped mapped Llama config.

    One point, because what is under test is which disclosures a candidate's
    artifacts carry, not the search. The DRAM tier is raised exactly as
    tests/test_qif_dse_allocation.py raises it, so the point is feasible and
    the sweep-level union (which spans the VALID candidates) is non-empty.
    """
    dse = _load_dse_tool()
    raw = yaml.safe_load(
        (HW_DIR / "fws_cim_llama7b_mapped.yaml").read_text()
    )
    raw["tech_param"]["DRAM"]["size"] = "32 GB"
    raw.setdefault("mapping", {})["links"] = links
    # 0176fa6 / D38: a one-point sweep uses the current fill axis.
    raw[dse.DSE_BLOCK] = {"axes": {"pp_util": [1.0]}}
    path = tmp_path / "fws_cim_links_sweep.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    exit_code, payload = dse.run_sweep(
        str(path),
        str(MODEL_DIR / "llama2_7b_fws_decode_inf.yaml"),
        output_dir=str(tmp_path / "out"),
        quiet=True,
    )
    assert exit_code == 0
    return payload, json.loads((tmp_path / "out" / "dse_report.json").read_text())


def test_the_dse_report_of_a_zeroed_sweep_carries_the_relaxation(tmp_path):
    payload, report = _one_point_sweep(tmp_path, "zeroed")
    assert payload["candidates"]
    for candidate in payload["candidates"]:
        constraints = [item["constraint"] for item in candidate.get("disclosures", ())]
        assert "links_zeroed" in constraints
    sweep_level = [
        item for item in report["disclosures"] if item["constraint"] == "links_zeroed"
    ]
    assert len(sweep_level) == 1
    assert "2026-08-27" in sweep_level[0]["reason"]


def test_the_dse_report_of_a_priced_sweep_carries_no_relaxation(tmp_path):
    _payload, report = _one_point_sweep(tmp_path, "priced")
    assert not [
        item for item in report["disclosures"] if item["constraint"] == "links_zeroed"
    ]


# ---------------------------------------------------------------------------
# 6. The energy law: only the TIME is relaxed
# ---------------------------------------------------------------------------


def _link_traffic(evaluation):
    components = fws_eval.report_document(evaluation)["evaluation"]["energy"][
        "components"
    ]
    rows = [item for item in components if item["key"] == "link_traffic"]
    assert len(rows) == 1
    return rows[0]


def _zero_energy_per_bit(node):
    """Every ``energy_per_bit`` in a raw config set to 0: a card that has none."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "energy_per_bit":
                node[key] = 0.0
            else:
                _zero_energy_per_bit(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _zero_energy_per_bit(value)


def test_the_energy_law_is_untouched_by_the_time_relaxation(priced, zeroed):
    """Same pJ, same coverage word. A relaxed TIME may not relabel an ENERGY.

    The module's energy law says a component that says ``uncovered``
    CONTRIBUTES 0 and never contributes an invented number. A zeroed run still
    charges the card's pJ/bit for every byte it moves, so its link energy is
    covered exactly where the priced run's is, and the coverage token may not
    move: it would be a moved artifact value with no named cause (the
    ``links_zeroed`` disclosure names a TIME relaxation and nothing else).
    """
    _mapping, _program, priced_eval = priced
    _zmapping, _zprogram, zeroed_eval = zeroed
    priced_row = _link_traffic(priced_eval)
    zeroed_row = _link_traffic(zeroed_eval)
    assert priced_row["energy_pj"] > 0.0
    assert zeroed_row["energy_pj"] == priced_row["energy_pj"]
    assert zeroed_row["coverage"] == priced_row["coverage"]
    # Per cost record, not only in the aggregate.
    priced_costs = {cost.uid: cost for cost in priced_eval.pricing.costs}
    for cost in zeroed_eval.pricing.costs:
        other = priced_costs[cost.uid]
        assert cost.energy_pj == other.energy_pj
        assert cost.coverage == other.coverage


def test_a_card_with_no_energy_per_bit_says_so_in_either_link_mode():
    """``zero-because-free`` and ``zero-because-uncovered`` stay distinguishable.

    A network dimension that declares ``energy_per_bit = 0`` gets the named
    ``link_energy`` disclosure whether its TIME was charged or relaxed. Without
    it a zeroed run would print a 0 pJ link term with no named reason.
    """
    def mutate(raw):
        _zero_energy_per_bit(raw)

    def with_links(value):
        def inner(raw):
            mutate(raw)
            raw["mapping"]["links"] = value

        return inner

    for mode in LINK_MODES:
        _mapping, _program, evaluation = _run(with_links(mode))
        row = _link_traffic(evaluation)
        assert row["energy_pj"] == 0.0
        assert row["coverage"] == "uncovered"
        named = [
            item for item in evaluation.disclosures if item.constraint == "link_energy"
        ]
        assert len(named) == 1, f"links = {mode!r} lost the link_energy disclosure"
        assert "energy_per_bit = 0" in named[0].reason
        assert "invented" in named[0].reason


# ---------------------------------------------------------------------------
# 7. The D16 handoff is an inter-chip transfer too
# ---------------------------------------------------------------------------


LLAMA7B = HW_DIR / "fws_cim_llama7b.yaml"
LLAMA7B_MODEL = MODEL_DIR / "llama2_7b_fws_inf.yaml"


def _pd_pair(prefill_links=None, decode_links=None):
    """The shipped Llama PD pair, each half declaring its own link mode."""
    raw = copy.deepcopy(yaml.safe_load(LLAMA7B.read_text()))
    prefill = {"regime": "lockstep"}
    decode = {"regime": "lockstep"}
    if prefill_links is not None:
        prefill["links"] = prefill_links
    if decode_links is not None:
        decode["links"] = decode_links
    raw["mapping"] = {"pd": {"prefill": prefill, "decode": decode}}
    config.convert(raw)
    hw = config.HWConfig.from_dict(raw)
    model = config.parse_config(str(LLAMA7B_MODEL), "LLM")
    return fws_mapping.build_pd_pair(hw, model)


def test_the_pd_handoff_is_priced_by_the_declared_link_mode():
    """The D16 handoff rides the pp link, so the mode has to reach it.

    It is the LARGEST inter-chip hop a PD machine has. A zeroed run that
    printed it at its p2p time would print a nonzero inter-chip handoff beside
    a disclosure saying every inter-chip transfer is 0 s.
    """
    default = _pd_pair()
    explicit = _pd_pair(LINKS_PRICED, LINKS_PRICED)
    relaxed = _pd_pair(LINKS_ZEROED, LINKS_ZEROED)

    # The default path does not move.
    assert default.handoff.time_s > 0.0
    assert explicit.handoff.time_s == default.handoff.time_s
    assert explicit.handoff.basis == default.handoff.basis

    # The relaxed one is 0 s on the SAME bytes, and says why.
    assert relaxed.handoff.total_bytes == default.handoff.total_bytes > 0
    assert relaxed.handoff.kv_bytes == default.handoff.kv_bytes
    assert relaxed.handoff.state_bytes == default.handoff.state_bytes
    assert relaxed.handoff.time_s == 0.0
    assert "links_zeroed" in relaxed.handoff.basis
    assert "BYTES" in relaxed.handoff.basis
    # The byte count is still in the summary the artifacts read.
    summary = relaxed.summary()
    assert summary["handoff_time_s"] == 0.0
    assert summary["handoff_bytes"] == default.summary()["handoff_bytes"]
    assert "0 s" in relaxed.report()


def test_two_halves_that_price_one_wire_two_ways_are_refused_by_name():
    """One wire, one law: a disagreement is a named refusal, not a winner."""
    with pytest.raises(MappingError) as excinfo:
        _pd_pair(LINKS_ZEROED, LINKS_PRICED)
    message = str(excinfo.value)
    assert "mapping.pd.prefill.links = 'zeroed'" in message
    assert "mapping.pd.decode.links = 'priced'" in message
    assert "ONE wire" in message
