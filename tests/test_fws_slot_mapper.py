# Copyright 2026 NanoCad lab, UCLA
# https://nanocad.ee.ucla.edu/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The slot mapper against its own law (D39), and its checker against liars.

Two halves. The LAW TESTS assert the mapper's output obeys D39 on every point
of the toy sweep: the beat is mux x 20 ns, slot equals tick modulo mux, the hop
and seam gaps hold, D is Little's law on L, and the ragged waste is exactly the
one the geometry forces. The MUTATION TESTS corrupt a good document one way at
a time and assert the independent checker catches each: a checker that passes
a lie is worth nothing.
"""

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import fws_slot_mapper as slotmap  # noqa: E402
from fws_slot_mapper import BundleSpec, SlotPlatform  # noqa: E402

import fws_toy_slotmap as toytool  # noqa: E402

CONFIG = PROJECT_ROOT / "configs" / "toy" / "toy_slotmap.yaml"
TICK_NS = 20.0


# --- fixtures ---------------------------------------------------------------


@pytest.fixture(scope="module")
def toy():
    raw, spec, bundles = toytool.load_config(CONFIG)
    return raw, spec, bundles


@pytest.fixture(scope="module")
def mapped(toy):
    """Every toy point: platform, mapping, ledgers, document, verdict."""
    raw, spec, bundles = toy
    out = {}
    for point in raw["points"]:
        platform = SlotPlatform(
            rows=int(spec["rows"]),
            z=int(point["z"]),
            y=int(point["y"]),
            chip_macros=int(spec["chip_macros"]),
        )
        mapping = slotmap.map_chain(platform, bundles)
        area = slotmap.macro_area_mm2(platform)
        led = slotmap.ledgers(mapping, area_mm2=area)
        doc = slotmap.to_document(
            mapping,
            title="toy %s" % point["id"],
            authored="2026-09-01",
            author="tests",
            command="pytest",
            layers=int(spec["layers"]),
            area_mm2=area,
        )
        doc["slots"]["check"] = slotmap.check_document(doc)
        out[str(point["id"])] = (platform, mapping, led, doc)
    return out


def _ids(mapped):
    return sorted(mapped)


# --- the law ----------------------------------------------------------------


def test_beat_is_mux_times_twenty_nanoseconds(mapped):
    for pid, (platform, _m, led, doc) in mapped.items():
        assert platform.mux == platform.y // platform.z, pid
        assert led["beat_ticks"] == platform.mux, pid
        assert led["beat_ns"] == pytest.approx(platform.mux * TICK_NS), pid
        assert led["throughput_tokens_per_s"] == pytest.approx(
            1.0 / (platform.mux * TICK_NS * 1e-9)
        ), pid
        assert doc["slots"]["platform"]["beat_ns"] == pytest.approx(platform.mux * TICK_NS)


def test_slot_is_phase_everywhere(mapped):
    """SLOT = PHASE: slot j fires at ticks == j (mod mux), and only those."""
    for pid, (platform, _m, _led, doc) in mapped.items():
        seen = 0
        for macro in doc["slots"]["macros"]:
            phases = set()
            for entry in macro["slots"]:
                assert entry["tick"] % platform.mux == entry["slot"], (pid, macro["id"])
                assert entry["slot"] not in phases, (pid, macro["id"])
                phases.add(entry["slot"])
                seen += 1
        assert seen == _led_of(doc)["slots_used"], pid


def _led_of(doc):
    return doc["slots"]["ledgers"]


def test_no_macro_fires_twice_in_one_tick_over_three_tokens(mapped):
    for pid, (platform, _m, _led, doc) in mapped.items():
        beat = platform.beat_ticks
        for token in range(3):
            busy = set()
            for macro in doc["slots"]["macros"]:
                for entry in macro["slots"]:
                    key = (macro["id"], entry["tick"] + token * beat)
                    assert key not in busy, (pid, key)
                    busy.add(key)


def test_hop_gap_and_seam_gap_hold_along_the_chain(mapped):
    for pid, (platform, _m, _led, doc) in mapped.items():
        bundles = sorted(doc["slots"]["bundles"], key=lambda b: b["order"])
        for prev, nxt in zip(bundles, bundles[1:]):
            same = prev["chip"] == nxt["chip"]
            need = prev["end_tick"] + platform.hop_gap_ticks
            if not same:
                need += platform.seam_ticks
            assert nxt["start_tick"] >= need, (pid, nxt["id"])
            assert nxt["seam_before"] is (not same), (pid, nxt["id"])
        assert bundles[0]["seam_before"] is False, pid


def test_a_bundle_never_splits_across_chips(mapped):
    for pid, (_p, _m, _led, doc) in mapped.items():
        chip_of = {m["id"]: m["chip"] for m in doc["slots"]["macros"]}
        for bundle in doc["slots"]["bundles"]:
            chips = {chip_of[mid] for mid in bundle["macros"]}
            assert chips == {bundle["chip"]}, (pid, bundle["id"])


def test_only_the_z512_point_wastes_ragged_cells(mapped):
    """qkv is 768 columns. Against Z = 512 that is 256 columns x 512 rows."""
    for pid, (platform, _m, led, doc) in mapped.items():
        ragged = {b["id"]: b["ragged_columns"] for b in doc["slots"]["bundles"]}
        blocks = {b["id"]: b["block"] for b in doc["slots"]["bundles"]}
        if platform.z == 512:
            qkv = [bid for bid, blk in blocks.items() if blk == "qkv"]
            assert qkv, pid
            assert all(ragged[bid] == 256 for bid in qkv), pid
            assert all(ragged[bid] == 0 for bid in ragged if bid not in qkv), pid
            assert led["cells_ragged"] == len(qkv) * 256 * platform.rows
        else:
            assert set(ragged.values()) == {0}, pid
            assert led["cells_ragged"] == 0, pid


def test_bundle_columns_is_the_whole_bundle_across_its_k_blocks(mapped):
    """SCHEMA.md, fws_slot/1: ``columns`` is the bundle's real columns, ALL
    K-BLOCKS TOGETHER. ffn2 is four K-blocks of 512 column-slots, so it reads
    2048 and not 512, and the toy chain totals 16128 columns over 12 bundles.
    The atlas prints this figure beside the slot-layer count, and the two only
    agree on this reading: 2048 columns over Z = 64 is the 32 slot-layers the
    map holds.
    """
    for pid, (platform, _m, _led, doc) in mapped.items():
        bundles = doc["slots"]["bundles"]
        assert len(bundles) == 12, pid
        assert sum(b["columns"] for b in bundles) == 16128, pid
        for b in bundles:
            kb = b["k_blocks"]
            assert b["columns"] % kb == 0, (pid, b["id"])
            per = b["columns"] // kb
            assert b["slot_layers"] == math.ceil(per / platform.z) * kb, (pid, b["id"])
            # every committed column-slot is real weight or declared ragged
            assert b["slot_layers"] * platform.z == b["columns"] + b["ragged_columns"]
        ffn2 = [b for b in bundles if b["block"] == "ffn2"]
        assert ffn2, pid
        for b in ffn2:
            assert b["k_blocks"] == 4 and b["reduce"] is True, pid
            assert b["columns"] == 2048, pid


def test_a_k_blocked_bundle_gives_each_k_block_its_own_column_chunk(mapped):
    """The bundle's column range is 0..columns and K-block k owns the k-th
    contiguous chunk of it, so no two K-blocks claim the same span and the
    checker can prove the cover from the document alone."""
    for pid, (platform, _m, _led, doc) in mapped.items():
        spans = {}
        for macro in doc["slots"]["macros"]:
            for entry in macro["slots"]:
                spans.setdefault(entry["bundle"], []).append(
                    (entry["kblock"], entry["columns"]["start"], entry["columns"]["count"])
                )
        for bundle in doc["slots"]["bundles"]:
            per = bundle["columns"] // bundle["k_blocks"]
            covered = []
            for kblock, start, count in spans[bundle["id"]]:
                assert kblock * per <= start < (kblock + 1) * per, (pid, bundle["id"])
                covered.extend(range(start, start + count))
            assert sorted(covered) == list(range(bundle["columns"])), (pid, bundle["id"])


def test_waste_is_ragged_plus_stranded_and_its_bound_is_zero(mapped):
    for pid, (platform, _m, led, doc) in mapped.items():
        assert led["cells_wasted"] == led["cells_ragged"] + led["cells_stranded"], pid
        assert led["cells_stranded"] == led["slots_stranded"] * platform.z * platform.rows
        # SCHEMA.md: the fraction is read against cells_total, every cell the
        # map commits, so the basis the atlas prints beside it ("N of M cells")
        # and the percentage above it are the same division.
        assert led["cells_total"] == led["slots_total"] * platform.z * platform.rows
        assert led["waste_frac"] == pytest.approx(led["cells_wasted"] / led["cells_total"]), pid
        assert led["waste_min"] == 0, pid


def test_d_is_little_law_on_l(mapped):
    """D = ceil(L / beat), and Little's law CLOSES on the figures one strip prints.

    A user's next token re-enters on the first bundle's own phase, and that
    phase comes round once per beat. One user's period is therefore D beats,
    and the rate is the throughput shared over the D sequences in flight. No
    user out-runs the machine. One over the chain latency is the CEILING of
    that rate: a beat that divides L reaches it, every other beat falls short.
    """
    for pid, (platform, _m, led, doc) in mapped.items():
        assert led["D"] == math.ceil(led["L_ticks"] / platform.beat_ticks), pid
        assert led["D_min"] == math.ceil(led["L_min_ticks"] / platform.beat_ticks), pid
        assert led["per_user_tokens_per_s"] == pytest.approx(
            1.0 / (led["D"] * platform.beat_s)
        ), pid
        assert led["D"] * led["per_user_tokens_per_s"] == pytest.approx(
            led["throughput_tokens_per_s"]
        ), pid
        assert led["per_user_tokens_per_s"] <= led["throughput_tokens_per_s"] * (
            1 + 1e-9
        ), pid
        assert led["per_user_ceiling_tokens_per_s"] == pytest.approx(
            1.0 / (led["L_ticks"] * TICK_NS * 1e-9)
        ), pid
        assert (
            led["per_user_ceiling_tokens_per_s"]
            >= led["per_user_tokens_per_s"] * (1 - 1e-9)
        ), pid
        if led["L_ticks"] % platform.beat_ticks == 0:
            assert led["per_user_ceiling_tokens_per_s"] == pytest.approx(
                led["per_user_tokens_per_s"]
            ), pid


def test_the_latency_bound_is_a_bound(mapped):
    """L_min = 2n - 1 ticks for n bundles plus the seams capacity forces. It is a floor.

    The law states 2 ticks per bundle. The tight floor is one tick lower, 2n -
    1, because L counts the last firing plus one and the hop gap only separates
    the n - 1 joins. No toy point comes near either figure. This guard is what
    catches a future chain that beats the printed floor: it fails here instead
    of printing a latency under its own bound.
    """
    for pid, (_p, mapping, led, _doc) in mapped.items():
        assert led["L_min_ticks"] >= 2 * len(mapping.bundles) - 1, pid
        assert led["L_min_ticks"] <= led["L_ticks"], pid
        assert led["D_min"] <= led["D"], pid
        assert led["seams_min"] <= led["seams"], pid


def test_l_is_the_last_firing_tick_plus_one(mapped):
    for pid, (_p, _m, led, doc) in mapped.items():
        bundles = sorted(doc["slots"]["bundles"], key=lambda b: b["order"])
        assert bundles[0]["start_tick"] == 0, pid
        assert led["L_ticks"] == bundles[-1]["end_tick"] + 1, pid
        assert led["L_ns"] == pytest.approx(led["L_ticks"] * TICK_NS), pid


def test_every_toy_point_checks(mapped):
    for pid, (_p, _m, _led, doc) in mapped.items():
        verdict = slotmap.check_document(doc)
        assert verdict["ok"], (pid, verdict["violations"])
        assert doc["slots"]["check"]["ok"] is True, pid


# --- the document contract --------------------------------------------------


REQUIRED_TOP_LEVEL = (
    "schema", "title", "provenance", "cards", "systems", "chips", "macros",
    "tiles", "links", "metrics", "relaxations",
)


def test_every_point_document_carries_the_atlas_required_fields(mapped):
    for pid, (_p, _m, _led, doc) in mapped.items():
        for key in REQUIRED_TOP_LEVEL:
            assert key in doc, (pid, key)
        assert doc["schema"] == "fws_atlas/1", pid
        prov = doc["provenance"]
        for key in ("authored", "author", "invented_fields"):
            assert key in prov, (pid, key)
        assert prov["invented_fields"] == [], pid
        assert doc["slots"]["schema"] == "fws_slot/1", pid
        for relaxation in doc["relaxations"]:
            for field in ("constraint", "value", "reason"):
                assert isinstance(relaxation.get(field), str) and relaxation[field]
        for metric in doc["metrics"]:
            assert isinstance(metric["value"], float) and math.isfinite(metric["value"])
            assert metric["basis"] and metric["label"]


def test_containment_is_stated_in_both_directions(mapped):
    for pid, (platform, _m, _led, doc) in mapped.items():
        macros = {m["id"]: m for m in doc["macros"]}
        tiles = {t["id"]: t for t in doc["tiles"]}
        for chip in doc["chips"]:
            assert chip["capacity"]["macro_slots"] == len(chip["macros"]), pid
            for mid in chip["macros"]:
                assert macros[mid]["chip"] == chip["id"], pid
        for mid, macro in macros.items():
            chip = next(c for c in doc["chips"] if c["id"] == macro["chip"])
            assert mid in chip["macros"], pid
            for tid in macro["tiles"]:
                assert tiles[tid]["macro"] == mid, pid
            if not macro["tiles"]:
                assert isinstance(macro["reserved"], str) and macro["reserved"], pid
        for tid, tile in tiles.items():
            assert tid in macros[tile["macro"]]["tiles"], pid
        # the two halves describe the same machine
        slot_macros = {m["id"]: m for m in doc["slots"]["macros"]}
        assert set(slot_macros) == set(macros), pid
        for mid, sm in slot_macros.items():
            assert len(sm["slots"]) == len(macros[mid]["tiles"]), pid
            assert sm["chip"] == macros[mid]["chip"], pid


def test_the_toy_emits_no_fake_digital_or_state_numbers(mapped):
    for pid, (_p, _m, _led, doc) in mapped.items():
        assert doc["links"] == [], pid
        assert all(m["digital_pool"] is None for m in doc["macros"]), pid
        assert all(card["kind"] == "analog_macro" for card in doc["cards"].values()), pid


# --- the checker against liars ----------------------------------------------


@pytest.fixture()
def good(mapped):
    return copy.deepcopy(mapped["p02"][3])


def _first_occupied(doc):
    for macro in doc["slots"]["macros"]:
        if macro["slots"]:
            return macro
    raise AssertionError("no occupied macro in the document")


def test_checker_catches_a_double_booked_slot(good):
    macro = _first_occupied(good)
    entry = copy.deepcopy(macro["slots"][0])
    macro["slots"].append(entry)
    macro["free_slots"] = [s for s in macro["free_slots"] if s != entry["slot"]]
    verdict = slotmap.check_document(good)
    assert not verdict["ok"]
    assert any("books slot" in v for v in verdict["violations"]), verdict


def test_checker_catches_a_tick_off_its_phase(good):
    macro = _first_occupied(good)
    macro["slots"][0]["tick"] += 1
    verdict = slotmap.check_document(good)
    assert not verdict["ok"]
    assert any("slot is phase" in v for v in verdict["violations"]), verdict


def test_checker_catches_a_violated_hop_gap(good):
    bundles = sorted(good["slots"]["bundles"], key=lambda b: b["order"])
    target = bundles[1]
    target["start_tick"] = bundles[0]["end_tick"]
    verdict = slotmap.check_document(good)
    assert not verdict["ok"]
    assert any(v.startswith("(d)") for v in verdict["violations"]), verdict


def test_checker_catches_a_violated_seam_gap(good):
    bundles = sorted(good["slots"]["bundles"], key=lambda b: b["order"])
    crossing = [b for b in bundles if b["seam_before"]]
    if not crossing:
        pytest.skip("this point crosses no seam")
    target = crossing[0]
    prev = bundles[bundles.index(target) - 1]
    target["start_tick"] = prev["end_tick"] + good["slots"]["platform"]["hop_gap_ticks"]
    verdict = slotmap.check_document(good)
    assert not verdict["ok"]
    assert any("the chain allows no earlier than" in v for v in verdict["violations"]), verdict


def test_checker_catches_a_dropped_tile(good):
    macro = _first_occupied(good)
    tid = next(m for m in good["macros"] if m["id"] == macro["id"])["tiles"][0]
    good["tiles"] = [t for t in good["tiles"] if t["id"] != tid]
    verdict = slotmap.check_document(good)
    assert not verdict["ok"]
    assert any(v.startswith("(g)") for v in verdict["violations"]), verdict


def test_checker_catches_a_dropped_slot_layer(good):
    macro = _first_occupied(good)
    dropped = macro["slots"].pop()
    macro["free_slots"] = sorted(macro["free_slots"] + [dropped["slot"]])
    verdict = slotmap.check_document(good)
    assert not verdict["ok"]
    assert any(v.startswith("(c)") for v in verdict["violations"]), verdict


def test_checker_catches_a_wrong_declared_d(good):
    good["slots"]["ledgers"]["D"] += 1
    verdict = slotmap.check_document(good)
    assert not verdict["ok"]
    assert any("ledgers.D declares" in v for v in verdict["violations"]), verdict


def test_checker_catches_a_wrong_declared_waste(good):
    good["slots"]["ledgers"]["cells_wasted"] += 512
    verdict = slotmap.check_document(good)
    assert not verdict["ok"]
    assert any("cells_wasted" in v for v in verdict["violations"]), verdict


def test_checker_catches_a_broken_beat_law(good):
    good["slots"]["platform"]["mux"] += 1
    verdict = slotmap.check_document(good)
    assert not verdict["ok"]
    assert any(v.startswith("(a)") for v in verdict["violations"]), verdict


def test_checker_catches_a_broken_containment(good):
    good["slots"]["chips"][0]["macros"] = good["slots"]["chips"][0]["macros"][:-1]
    verdict = slotmap.check_document(good)
    assert not verdict["ok"]
    assert any(v.startswith("(g)") for v in verdict["violations"]), verdict


# --- the mapper refuses what it cannot map ----------------------------------


def test_a_fractional_mux_is_refused():
    with pytest.raises(slotmap.SlotMapError):
        SlotPlatform(rows=512, z=48, y=2048, chip_macros=6)


def test_seconds_passed_as_nanoseconds_are_refused():
    with pytest.raises(slotmap.SlotMapError):
        SlotPlatform(rows=512, z=64, y=2048, chip_macros=6, tick_ns=20e-9)


def test_a_bundle_wider_than_one_chip_is_refused():
    platform = SlotPlatform(rows=512, z=64, y=512, chip_macros=1)
    wide = [BundleSpec(id="b.000", order=0, layer=0, block="ffn1", columns=2048)]
    with pytest.raises(slotmap.SlotMapError):
        slotmap.map_chain(platform, wide)


# --- the tool ---------------------------------------------------------------


def test_the_tool_writes_a_checked_manifest_report_and_points(tmp_path):
    out = tmp_path / "toy_slotmap_frontier"
    manifest = tmp_path / "toy_slotmap.json"
    code = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tools" / "fws_toy_slotmap.py"),
         "--config", str(CONFIG), "--out", str(out), "--manifest", str(manifest)],
        capture_output=True, text=True, timeout=600, cwd=str(PROJECT_ROOT),
    )
    assert code.returncode == 0, code.stderr

    report = json.loads((out / "dse_report.json").read_text())
    assert report["mode"] == "TOY"
    assert report["headline_metric"] == "tokens_per_s"
    assert report["num_candidates"] == len(report["candidates"]) == 6
    assert report["selected_id"] in report["front_ids"]
    assert report["disclosures"], "the toy always discloses that it is a toy"

    for cand in report["candidates"]:
        assert cand["ok"] is True
        assert cand["atlas_point"] == "points/%s.json" % cand["id"]
        assert set(cand["knobs"]) == {"macro_z", "macro_y", "mux"}
        assert cand["metrics"]["tokens_per_s"] > 0
        assert cand["silicon"]["total_silicon_mm2"] > 0
        assert cand["placement"]["analog_chips"] >= 1

        # A published block's own numbers must reproduce its own percentage.
        pack = cand["packing"]
        assert pack["wasted_cells"] == pack["ragged_cells"] + pack["stranded_cells"]
        assert pack["waste_pct"] == pytest.approx(
            100.0 * pack["wasted_cells"] / pack["cells_total"]
        )
        # Little's law, in the block that publishes both figures.
        pipe = cand["pipeline"]
        assert pipe["resident_streams"] * pipe["per_stream_tokens_per_s"] == pytest.approx(
            pipe["tokens_per_s"]
        )
        assert (
            pipe["per_stream_ceiling_tokens_per_s"]
            >= pipe["per_stream_tokens_per_s"] * (1 - 1e-9)
        )
        # absent, never a zero standing in for a fact the toy does not have
        assert "derived_digital" not in cand
        assert "state_bill" not in cand
        assert "shared_digital_chiplets" not in cand["placement"]

        doc = json.loads((out / "points" / ("%s.json" % cand["id"])).read_text())
        assert doc["schema"] == "fws_atlas/1"
        assert doc["run_id"] == report["run_id"]
        assert doc["dse_point"]["candidate_id"] == cand["id"]
        assert doc["slots"]["check"]["ok"] is True
        assert slotmap.check_document(doc)["ok"]

    row = json.loads(manifest.read_text())["members"][0]
    assert row["mode"] == "TOY"
    assert row["name"] == "toy_slotmap"
    assert row["model_id"] == "Toy slot-map"
    assert row["run_id"] == report["run_id"]
    assert row["selected_id"] == report["selected_id"]
    assert row["headline"].endswith("tokens/s")


def test_the_tool_refuses_to_write_a_point_that_does_not_check(tmp_path, monkeypatch):
    """A mapping nobody can verify is not a mapping: nothing is written."""
    out = tmp_path / "frontier"
    monkeypatch.setattr(
        slotmap, "check_document",
        lambda doc: {"ok": False, "violations": ["planted violation"]},
    )
    code = toytool.run(str(CONFIG), out, tmp_path / "manifest.json")
    assert code == 1
    assert not (out / "points").exists()
    assert not (out / "dse_report.json").exists()


# --- the real loader judges the real documents ------------------------------


def test_the_atlas_loader_accepts_every_toy_point(mapped, tmp_path):
    """The page's own loader, headless. A document it refuses is not a mapping."""
    import shutil

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the atlas loader test needs it")
    html = (PROJECT_ROOT / "docs" / "qif" / "atlas" / "atlas.html").read_text(
        encoding="utf-8"
    )
    core = html[html.index("// ==ATLAS-CORE-START==") : html.index("// ==ATLAS-CORE-END==")]
    for pid, (_p, _m, _led, doc) in mapped.items():
        path = tmp_path / ("%s.json" % pid)
        path.write_text(json.dumps(doc))
        script = tmp_path / ("drive_%s.js" % pid)
        script.write_text(
            "const fs = require('fs');\n"
            "const doc = JSON.parse(fs.readFileSync(%r, 'utf8'));\n" % str(path)
            + "const ATLAS = eval('(function(){' + "
            + json.dumps(core)
            + " + '; return ATLAS; })()');\n"
            + "const v = ATLAS.validate(doc);\n"
            + "console.log(JSON.stringify({e: v.errors, w: v.warnings}));\n",
            encoding="utf-8",
        )
        run = subprocess.run(
            [node, str(script)], capture_output=True, text=True, timeout=120,
            cwd=str(PROJECT_ROOT),
        )
        assert run.returncode == 0, run.stderr
        result = json.loads(run.stdout.strip())
        assert result["e"] == [], (pid, result["e"])
        assert result["w"] == [], (pid, result["w"])


def test_the_atlas_slot_fixture_is_what_a_fresh_build_produces():
    """The atlas proves its slot rules on a document, and a document nobody can
    rebuild drifts from the contract it illustrates - this one did, twice. It is
    emitted by `python tools/fws_toy_slotmap.py --fixture` and compared BYTE for
    byte, so a contract change that is not carried into it fails here."""
    import json as _json

    fixture = PROJECT_ROOT / "docs" / "qif" / "atlas" / "fixture_slotmap.json"
    assert fixture.exists(), "the atlas slot fixture is missing"
    doc, _led = toytool.build_fixture(CONFIG)
    fresh = _json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    on_disk = fixture.read_text(encoding="utf-8")
    # `authored` is today's date and moves on its own; nothing else may.
    import re as _re

    strip = lambda blob: _re.sub(r'"authored": "[0-9-]+"', '"authored": "-"', blob)
    assert strip(fresh) == strip(on_disk), (
        "docs/qif/atlas/fixture_slotmap.json is stale; rebuild it with "
        "python tools/fws_toy_slotmap.py --fixture"
    )
    assert doc["slots"]["check"]["ok"] is True


def test_the_temporary_fixture_bundle_is_gone():
    """The slot view is read on the real sweep. The three-point fixture bundle
    was scaffolding while the two halves were built apart, and a second toy in
    ../dse is a second answer to one question."""
    dse = PROJECT_ROOT / "docs" / "qif" / "dse"
    assert not (dse / "fixture_slotmap.json").exists()
    assert not (dse / "fixture_slotmap_frontier").exists()
