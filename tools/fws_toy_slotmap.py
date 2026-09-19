#!/usr/bin/env python
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

"""Run the TOY slot-map sweep and write the artifacts the results page reads.

The toy is a MAPPER DEBUG MODEL (D39's first implementation). It sweeps one
axis pair -- the macro's ADC count Z and its column count Y -- over a chain of
twelve bundles, and for each point it writes ONE ``fws_atlas/1`` document
carrying the ``fws_slot/1`` schedule, a sweep report and a one-member bundle
manifest.

REFUSAL IS THE POINT. Every document is handed to
``fws_slot_mapper.check_document``, which re-derives the whole schedule from
the document alone. A point whose check fails is NOT WRITTEN: this tool prints
the violations and exits non-zero, because a mapping nobody can verify is not
a mapping.

Fields the toy cannot honestly fill are ABSENT from the report, never zero. It
places no digital chiplet and holds no per-sequence state, so it publishes no
digital engine, no state bill, no energy and no boundary table.
"""

import argparse
import datetime
import hashlib
import json
import math
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fws_slot_mapper as slotmap  # noqa: E402
from fws_slot_mapper import BundleSpec, SlotPlatform  # noqa: E402

TOOL = "fws_toy_slotmap"
TOOL_VERSION = "fws_toy_slotmap/1.0-d39-toy"
REPORT_SCHEMA = "fws_toy_slotmap/1"
MANIFEST_SCHEMA = "fws_run_bundle/1"
DEFAULT_CONFIG = "configs/toy/toy_slotmap.yaml"
DEFAULT_OUT = "docs/qif/dse/toy_slotmap_frontier"
DEFAULT_MANIFEST = "docs/qif/dse/toy_slotmap.json"
DEFAULT_FIXTURE = "docs/qif/atlas/fixture_slotmap.json"

#: Quoted verbatim in the results page's disclosures panel.
DISCLOSURES = [
    {
        "constraint": "toy_model",
        "value": "a mapper debug chain, not a machine",
        "reason": (
            "This sweep exists to exercise the slot mapper and its independent "
            "checker. Its twelve bundles are three blocks of qkv, out, ffn1 and "
            "ffn2 at a hidden size chosen to equal the macro height, so every "
            "bundle's K fits one macro exactly. No number here describes any real "
            "model, and none of it should be quoted as a result."
        ),
    },
    {
        "constraint": "chip_capacity",
        "value": "6 macros per chip, a debug override",
        "reason": (
            "The chip capacity is passed to the mapper directly by the toy so that "
            "seams and stranded slots happen on a short chain. It is not the die "
            "area budget the real platform work derives, and it must not be read "
            "as one."
        ),
    },
    {
        "constraint": "bandwidth_provisioned",
        "value": "assumed sufficient, not derived",
        "reason": (
            "Wire hops, reduce trees and chip seams add latency only; they never "
            "bound the beat. The mapping assumes the fabric carries every handoff "
            "inside the two-tick hop and the twenty-five-tick seam it is charged. "
            "No bandwidth number is derived anywhere here, and the assumption is "
            "recorded rather than hidden."
        ),
    },
    {
        "constraint": "unpriced_terms",
        "value": "no digital chiplet, no state store, no energy, no bytes",
        "reason": (
            "The toy places analog weight macros and schedules their firings. It "
            "composes no digital engine, sizes no state store, prices no energy "
            "and counts no bytes across the seams it reports. Those panels are "
            "empty because the quantity is absent, not because it measured zero."
        ),
    },
    {
        "constraint": "silicon_counted",
        "value": "the macros the mapping occupies",
        "reason": (
            "Silicon is the count of macros holding at least one slot-layer times "
            "the macro area the analog sheet port derives at that geometry. Empty "
            "macro slots on the chips the mapper opened are enumerated and "
            "reported, and they are not charged."
        ),
    },
]

#: The same statements, in the shape the atlas document's relaxations[] takes.
RELAXATIONS = DISCLOSURES


def load_config(path):
    """Parse the toy YAML into the chain, the chip capacity and the points."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or "toy" not in raw or "points" not in raw:
        raise SystemExit(f"{path}: not a toy slot-map config (needs toy: and points:).")
    toy = raw["toy"]
    bundles = []
    order = 0
    for layer in range(int(toy["layers"])):
        for spec in toy["bundles"]:
            bundles.append(
                BundleSpec(
                    id="b.%03d" % order,
                    order=order,
                    layer=layer,
                    block=str(spec["block"]),
                    columns=int(spec["columns"]),
                    k_blocks=int(spec.get("k_blocks", 1)),
                    reduce=bool(spec.get("reduce", False)),
                )
            )
            order += 1
    return raw, toy, bundles


def run_id_of(config_path):
    """A content hash over what this run was made of.

    Everything that can change a number is in it: this tool, the mapper module
    and the config. ``generated_at`` is deliberately outside, so two runs of
    the same inputs carry the same id.
    """
    h = hashlib.sha256()
    h.update(TOOL_VERSION.encode())
    for path in (Path(__file__), REPO_ROOT / "fws_slot_mapper.py", Path(config_path)):
        h.update(Path(path).read_bytes())
    return h.hexdigest()[:16]


def front_ids_of(points):
    """Non-dominated under (max tokens/s, min mm2), in declaration order."""
    front = []
    for a in points:
        dominated = False
        for b in points:
            if b is a:
                continue
            if (
                b["tokens_per_s"] >= a["tokens_per_s"]
                and b["mm2"] <= a["mm2"]
                and (b["tokens_per_s"] > a["tokens_per_s"] or b["mm2"] < a["mm2"])
            ):
                dominated = True
                break
        if not dominated:
            front.append(a["id"])
    return front


def knee_of(points, front_ids):
    """The point furthest above the chord of the front, or None under three."""
    front = [p for p in points if p["id"] in front_ids]
    front.sort(key=lambda p: p["mm2"])
    if len(front) < 3:
        return None
    lo, hi = front[0], front[-1]
    span_x = max(1e-30, abs(hi["tokens_per_s"] - lo["tokens_per_s"]))
    span_y = max(1e-30, abs(hi["mm2"] - lo["mm2"]))

    def norm(p):
        return ((p["tokens_per_s"] - lo["tokens_per_s"]) / span_x, (p["mm2"] - lo["mm2"]) / span_y)

    ax, ay = norm(lo)
    bx, by = norm(hi)
    best, best_d = None, -1.0
    for p in front:
        px, py = norm(p)
        area = abs((bx - ax) * (py - ay) - (px - ax) * (by - ay))
        d = area / max(1e-30, math.hypot(bx - ax, by - ay))
        if d > best_d:
            best, best_d = p, d
    return {
        "point": best["id"],
        "distance": best_d,
        "basis": (
            "furthest above the chord of the front, measured on the front's own "
            "spans between the cheapest and the fastest point. Past it, silicon "
            "buys throughput at worse than the front's average rate."
        ),
        "points": [p["id"] for p in front],
    }


def build_point(point, toy, bundles, run, config_path, out_dir, authored):
    """Map one (Z, Y) point, check its document, and return both."""
    platform = SlotPlatform(
        rows=int(toy["rows"]),
        z=int(point["z"]),
        y=int(point["y"]),
        chip_macros=int(toy["chip_macros"]),
    )
    mapping = slotmap.map_chain(platform, bundles)
    area = slotmap.macro_area_mm2(platform)
    led = slotmap.ledgers(mapping, area_mm2=area)
    pid = str(point["id"])
    command = (
        "python tools/fws_toy_slotmap.py --config %s --out %s   (point %s: Z = %d, "
        "Y = %d, mux = %d)" % (config_path, out_dir, pid, platform.z, platform.y, platform.mux)
    )
    doc = slotmap.to_document(
        mapping,
        title="Toy slot map - point %s, macro %s" % (pid, platform.label),
        subtitle=(
            "A mapper debug chain: %d bundles, %d slot-layers, beat %g ns, chain "
            "latency %d ticks. Placed and scheduled in one decision, then verified "
            "from this document alone."
            % (len(mapping.bundles), led["slots_used"], led["beat_ns"], led["L_ticks"])
        ),
        authored=authored,
        author="fws_toy_slotmap.build_point, from fws_slot_mapper.map_chain",
        command=command,
        model_id=str(toy["model_id"]),
        model_label=str(toy["label"]),
        layers=int(toy["layers"]),
        intermediate_size=2048,
        note=(
            "A DEBUG TOY. Every figure is derived from the enumerated slot table in "
            "this document; nothing is carried over from another point. Read "
            "relaxations[] before quoting anything: this chain is not a model."
        ),
        relaxations=RELAXATIONS,
        run=run,
        dse_point={
            "report": "dse_report.json",
            "candidate_id": pid,
            "knobs": {"macro_z": platform.z, "macro_y": platform.y, "mux": platform.mux},
            "knobs_derived": {
                "beat_ns": platform.beat_ns,
                "chip_macros": platform.chip_macros,
                "slots_per_chip": platform.slots_per_chip,
            },
            "headline": "%g tokens/s" % led["throughput_tokens_per_s"],
            "total_silicon_mm2": led["analog_area_mm2"],
            "front_membership": (
                "not stated here: whether this point is on the front is a property "
                "of the sweep, and dse_report.json says it once."
            ),
        },
        area_mm2=area,
    )
    verdict = slotmap.check_document(doc)
    doc["slots"]["check"] = verdict
    return doc, led, platform, verdict


def candidate_row(pid, platform, led, note):
    """One report candidate. Absent means absent: no zero stands in for a fact."""
    return {
        "id": pid,
        "ok": True,
        "atlas_point": "points/%s.json" % pid,
        "knobs": {"macro_z": platform.z, "macro_y": platform.y, "mux": platform.mux},
        "knobs_derived": {
            "beat_ns": platform.beat_ns,
            "chip_macros": platform.chip_macros,
            "slots_per_chip": platform.slots_per_chip,
        },
        "note": note,
        "placement": {
            "analog_chips": led["chips_used"],
            "analog_macro_slots": led["slots_total"] // platform.mux,
            "macros_holding_tiles": led["macros_used"],
            "unowned_macro_slots": led["slots_total"] // platform.mux - led["macros_used"],
            "tiles": led["slots_used"],
            "slots_used": led["slots_used"],
            "slots_stranded": led["slots_stranded"],
            "seams": led["seams"],
        },
        "silicon": {
            "analog_macro_slots": led["slots_total"] // platform.mux,
            "macro_footprint_mm2": led["macro_area_mm2"],
            "analog_macro_silicon_mm2": led["analog_area_mm2"],
            "total_silicon_mm2": led["analog_area_mm2"],
            "basis": (
                "%d macro(s) holding at least one slot-layer x %.6g mm2, the macro "
                "area the analog sheet port derives at %s. The empty slots on the "
                "chips the mapper opened are enumerated and reported, and they are "
                "not charged." % (led["macros_used"], led["macro_area_mm2"], platform.label)
            ),
        },
        "packing": {
            "law": "slot",
            "waste_pct": 100.0 * led["waste_frac"],
            "real_cells": led["cells_used"],
            # The denominator the percentage uses, stated. It is every cell the
            # map commits: slots_total x Z x rows over the macros it occupies,
            # free slots included. cells_used + wasted_cells is the smaller
            # count of the OCCUPIED slots alone, and dividing by that gives a
            # different percentage, so the block publishes the one it used.
            "cells_total": led["cells_total"],
            "wasted_cells": led["cells_wasted"],
            "ragged_cells": led["cells_ragged"],
            "stranded_cells": led["cells_stranded"],
            "basis": (
                "waste in CELLS: the ragged remainder of every bundle's last "
                "slot-layer against Z, plus every slot stranded when a bundle moved "
                "whole to a fresh chip. waste_pct is those wasted cells over "
                "cells_total, every cell the mapping commits, free slots included. "
                "The bound is zero."
            ),
        },
        "pipeline": {
            "beat_s": platform.beat_s,
            "beat_ticks": platform.beat_ticks,
            "tokens_per_s": led["throughput_tokens_per_s"],
            # resident_streams x per_stream_tokens_per_s == tokens_per_s.
            "per_stream_tokens_per_s": led["per_user_tokens_per_s"],
            "per_stream_ceiling_tokens_per_s": led["per_user_ceiling_tokens_per_s"],
            "per_token_latency_s": led["L_ns"] * 1e-9,
            "resident_streams": led["D"],
        },
        "metrics": {
            "tokens_per_s": led["throughput_tokens_per_s"],
            "beat_s": platform.beat_s,
            "per_stream_tokens_per_s": led["per_user_tokens_per_s"],
            "per_token_latency_s": led["L_ns"] * 1e-9,
        },
        "ledgers": {
            "L_ticks": led["L_ticks"],
            "L_min_ticks": led["L_min_ticks"],
            "D": led["D"],
            "D_min": led["D_min"],
            "seams": led["seams"],
            "seams_min": led["seams_min"],
            "stall_ticks": led["stall_ticks"],
            "waste_frac": led["waste_frac"],
            "waste_min": led["waste_min"],
        },
        "utilization": [
            {
                "device_class": "analog_macro",
                "devices": led["slots_total"] // platform.mux,
                "idle_devices": led["slots_total"] // platform.mux - led["macros_used"],
                "mean_occupancy": led["macro_duty_mean"],
                "max_occupancy": led["macro_duty_max"],
            }
        ],
        "disclosures": DISCLOSURES,
    }


def run(config_path, out_dir, manifest_path, *, authored=None, generated_at=None):
    """Run every point, refuse any that does not check, write the artifacts."""
    raw, toy, bundles = load_config(config_path)
    out_dir = Path(out_dir)
    now = generated_at or datetime.datetime.now(datetime.timezone.utc).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")
    authored = authored or now[:10]
    rid = run_id_of(config_path)
    run_block = {
        "run_id": rid,
        "tool": TOOL,
        "tool_version": TOOL_VERSION,
        "generated_at": now,
        "basis": (
            "the run id is a sha256 over this tool's version, the content of "
            "tools/fws_toy_slotmap.py, fws_slot_mapper.py and the toy config. Every "
            "artifact this run writes carries it, so a reader holding two of them "
            "can tell whether they describe one generation or two. generated_at is "
            "outside the hash."
        ),
    }

    docs, cands, rows = {}, [], []
    failures = []
    for point in raw["points"]:
        doc, led, platform, verdict = build_point(
            point, toy, bundles, run_block, config_path, out_dir, authored
        )
        pid = str(point["id"])
        if not verdict["ok"]:
            failures.append((pid, verdict["violations"]))
            continue
        docs[pid] = doc
        cands.append(candidate_row(pid, platform, led, str(point.get("note", ""))))
        rows.append(
            {
                "id": pid,
                "z": platform.z,
                "y": platform.y,
                "mux": platform.mux,
                "beat_ns": platform.beat_ns,
                "tokens_per_s": led["throughput_tokens_per_s"],
                "mm2": led["analog_area_mm2"],
                "led": led,
            }
        )

    if failures:
        print("[toy] REFUSED: %d point(s) did not check, and no point was written."
              % len(failures), file=sys.stderr)
        for pid, violations in failures:
            print("  %s:" % pid, file=sys.stderr)
            for line in violations:
                print("    - %s" % line, file=sys.stderr)
        return 1

    front = front_ids_of(rows)
    knee = knee_of(rows, front)
    selected = knee["point"] if knee else front[0]

    report = {
        "schema": REPORT_SCHEMA,
        "tool": TOOL,
        "run_id": rid,
        "run": run_block,
        "config": str(config_path),
        "model_id": str(toy["model_id"]),
        "mode": "TOY",
        "objective": "throughput",
        "headline_metric": "tokens_per_s",
        "selection_rule": (
            "the knee of the front: the point furthest above the chord between the "
            "cheapest and the fastest non-dominated point."
        ),
        "evaluation_path": (
            "fws_slot_mapper.map_chain -> ledgers -> to_document -> check_document, "
            "per point. Every document is verified from itself before it is written."
        ),
        "sweep": {
            "block": "toy_slotmap",
            "label": str(toy["label"]),
            "axes": {
                "macro_z": sorted({int(p["z"]) for p in raw["points"]}),
                "macro_y": sorted({int(p["y"]) for p in raw["points"]}),
            },
            "chain": [
                {
                    "id": b.id,
                    "order": b.order,
                    "layer": b.layer,
                    "block": b.block,
                    "columns": b.columns,
                    "k_blocks": b.k_blocks,
                    "reduce": b.reduce,
                }
                for b in bundles
            ],
        },
        "num_candidates": len(cands),
        "num_valid": len(cands),
        "candidates": cands,
        "front_ids": front,
        "front_note": (
            "%d non-dominated point(s) over %d evaluated. x is tokens per second, "
            "which is one over the beat, and y is the silicon of the macros the "
            "mapping occupies." % (len(front), len(cands))
        ),
        "selected_id": selected,
        "atlas_points": {
            "schema": slotmap.ATLAS_SCHEMA,
            "dir": "points",
            "run_id": rid,
            "enabled": True,
            "candidates": len(cands),
            "emitted": [c["id"] for c in cands],
            "basis": (
                "one fws_atlas/1 document per point, carrying the fws_slot/1 tick "
                "schedule and the checker's verdict."
            ),
        },
        "disclosures": DISCLOSURES,
    }
    if knee:
        report["knee"] = knee

    points_dir = out_dir / "points"
    points_dir.mkdir(parents=True, exist_ok=True)
    for pid, doc in docs.items():
        (points_dir / ("%s.json" % pid)).write_text(json.dumps(doc, indent=2) + "\n")
    (out_dir / "dse_report.json").write_text(json.dumps(report, indent=2) + "\n")

    headline = "%g tokens/s" % next(r for r in rows if r["id"] == selected)["tokens_per_s"]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "bundle": str(toy["name"]),
        "label": str(toy["label"]),
        "generated_at": now,
        "members": [
            {
                "name": str(toy["name"]),
                "model_id": str(toy["model_id"]),
                "label": str(toy["label"]),
                "mode": "TOY",
                "dir": _repo_relative(out_dir),
                "run_id": rid,
                "selected_id": selected,
                "headline": headline,
            }
        ],
    }
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print("[toy] run %s - %d point(s), all checked" % (rid, len(cands)))
    print("[toy] report   %s" % (out_dir / "dse_report.json"))
    print("[toy] manifest %s (selected %s, %s)" % (manifest_path, selected, headline))
    print(ledger_table(rows))
    return 0


def _repo_relative(path):
    """The manifest names a REPO-RELATIVE dir, the spelling the page prefixes."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def ledger_table(rows):
    """The per-point ledgers, one line each, in declaration order."""
    head = (
        "point   Z     Y   mux  beat ns  macros chips seams  L tk  D  Dmin  waste %  "
        "stalls      tokens/s   per-user tok/s      mm2"
    )
    out = [head, "-" * len(head)]
    for r in rows:
        led = r["led"]
        out.append(
            "%-5s %4d %5d %5d %8.1f %7d %5d %5d %6d %3d %5d %8.3f %7d %13.0f %16.0f %8.3f"
            % (
                r["id"], r["z"], r["y"], r["mux"], r["beat_ns"],
                led["macros_used"], led["chips_used"], led["seams"], led["L_ticks"],
                led["D"], led["D_min"], 100.0 * led["waste_frac"], led["stall_ticks"],
                led["throughput_tokens_per_s"], led["per_user_tokens_per_s"],
                led["analog_area_mm2"],
            )
        )
    return "\n".join(out)


def build_fixture(config_path):
    """The atlas's slot fixture: ONE block of the chain on 512 x 512 : 64.

    The atlas needs a small slot document to prove its loader rules on. That
    document used to be typed by hand, and a hand document drifts from the
    contract it illustrates: this one carried K-block spans that restarted
    inside one bundle and a waste fraction read against a denominator the
    sentence beside it did not use. It is emitted here instead, from the same
    chain and the same mapper the sweep uses, so the two cannot disagree.
    """
    _raw, toy, bundles = load_config(config_path)
    per_block = len(bundles) // int(toy["layers"])
    one_block = [
        BundleSpec(id=b.id, order=b.order, layer=0, block=b.block,
                   columns=b.columns, k_blocks=b.k_blocks, reduce=b.reduce)
        for b in bundles[:per_block]
    ]
    platform = SlotPlatform(rows=int(toy["rows"]), y=512, z=64,
                            chip_macros=int(toy["chip_macros"]))
    mapping = slotmap.map_chain(platform, one_block)
    area = slotmap.macro_area_mm2(platform)
    led = slotmap.ledgers(mapping, area_mm2=area)
    doc = slotmap.to_document(
        mapping,
        title="Toy slot map - one block on %s" % platform.label,
        subtitle=(
            "One block of the toy debug chain - qkv 768 columns, out 512, ffn1 2048 "
            "and ffn2 as four K-blocks of 512 whose partials reduce - placed slot by "
            "slot on macros of %d rows x %d columns behind %d ADCs. mux = %d, so the "
            "beat is %g ns. ffn1 does not fit the slots the first chiplet has left, so "
            "it moves whole and pays a crossing, and ffn2 does the same: %d chiplets, "
            "%d crossings, %d stranded slots."
            % (platform.rows, platform.y, platform.z, platform.mux, led["beat_ns"],
               led["chips_used"], led["seams"], led["slots_stranded"])
        ),
        authored=datetime.date.today().isoformat(),
        author="fws_toy_slotmap.build_fixture, from fws_slot_mapper.map_chain",
        command="python tools/fws_toy_slotmap.py --fixture",
        model_id=str(toy["model_id"]),
        model_label="Toy slot map (atlas fixture)",
        layers=1,
        intermediate_size=2048,
        note=(
            "THE ATLAS'S SLOT FIXTURE. One block of the toy debug chain, emitted by the "
            "same mapper the sweep uses, so the loader's slot rules are proved against a "
            "document that obeys the contract. It is not a result: read relaxations[] "
            "before quoting anything."
        ),
        relaxations=RELAXATIONS,
        run={
            "run_id": "fixture_slotmap_1",
            "tool": "fws_toy_slotmap.build_fixture",
            "tool_version": "the slot fixture the atlas loader is proved on",
            "generated_at": "2026-09-01T09:00:00Z",
            "basis": (
                "one mapping of one block, rebuilt whenever the slot contract moves so "
                "the fixture and the sweep cannot disagree"
            ),
        },
        area_mm2=area,
    )
    doc["slots"]["check"] = slotmap.check_document(doc)
    return doc, led


def write_fixture(config_path, path):
    doc, led = build_fixture(config_path)
    if not doc["slots"]["check"]["ok"]:
        raise SystemExit(
            "[toy] REFUSED: the fixture does not check.\n  "
            + "\n  ".join(doc["slots"]["check"]["violations"][:8])
        )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[toy] fixture %s - %d bundles, %d chiplets, %d crossings, %.1f%% waste"
          % (out, len(doc["slots"]["bundles"]), led["chips_used"], led["seams"],
             100 * led["waste_frac"]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(REPO_ROOT / DEFAULT_CONFIG))
    ap.add_argument("--out", default=str(REPO_ROOT / DEFAULT_OUT))
    ap.add_argument("--manifest", default=str(REPO_ROOT / DEFAULT_MANIFEST))
    ap.add_argument("--fixture", nargs="?", const=str(REPO_ROOT / DEFAULT_FIXTURE),
                    default=None,
                    help="rebuild the atlas's slot fixture instead of the sweep")
    args = ap.parse_args(argv)
    if args.fixture:
        return write_fixture(args.config, args.fixture)
    return run(args.config, args.out, args.manifest)


if __name__ == "__main__":
    sys.exit(main())
