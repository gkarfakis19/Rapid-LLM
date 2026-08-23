"""QIF P5 system-atlas tests.

The atlas is one self-contained HTML file with an embedded copy of the
fixture, so three things can rot silently between suite runs: the embedded
blob can drift from the sibling fixture, the fixture can stop conforming to
the frozen fws_atlas/1 schema, and the loader can pick up a JavaScript syntax
error that renders the page blank. The suite catches all three.

The loader tests shell out to node (skipped when node is absent, the
tests/test_webui_browser.py precedent) and evaluate only the block between
the ATLAS-CORE markers, which carries no DOM.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATLAS_DIR = PROJECT_ROOT / "docs" / "qif" / "atlas"
ATLAS_HTML = ATLAS_DIR / "atlas.html"
FIXTURE = ATLAS_DIR / "fixture_llama7b_tp2.json"
SCHEMA_MD = ATLAS_DIR / "SCHEMA.md"

CORE_START = "// ==ATLAS-CORE-START=="
CORE_END = "// ==ATLAS-CORE-END=="
EMBED_RE = re.compile(
    r'<script[^>]*type="application/json"[^>]*id="atlas-embedded"[^>]*>(.*?)</script>',
    re.DOTALL,
)


def _html() -> str:
    return ATLAS_HTML.read_text(encoding="utf-8")


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _node() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the atlas loader tests need it")
    return node


def _run_core(driver: str, tmp_path: Path) -> str:
    """Evaluate the ATLAS core block headlessly and run a driver against it."""
    html = _html()
    core = html[html.index(CORE_START) : html.index(CORE_END)]
    script = tmp_path / "driver.js"
    script.write_text(
        "const fs = require('fs');\n"
        "const doc = JSON.parse(fs.readFileSync(%r, 'utf8'));\n" % str(FIXTURE)
        + "const ATLAS = eval('(function(){' + "
        + json.dumps(core)
        + " + '; return ATLAS; })()');\n"
        + textwrap.dedent(driver),
        encoding="utf-8",
    )
    result = subprocess.run(
        [_node(), str(script)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


# --- P5.1 the fixture -------------------------------------------------------


def test_fixture_is_an_fws_atlas_1_document():
    doc = _fixture()
    assert doc["schema"] == "fws_atlas/1"
    for key in ("title", "provenance", "cards"):
        assert key in doc, key
    for key in ("systems", "chips", "macros", "tiles", "links", "metrics", "relaxations"):
        assert isinstance(doc[key], list) and doc[key], key
    provenance = doc["provenance"]
    assert provenance["authored"] and provenance["author"]
    assert isinstance(provenance["invented_fields"], list)


def test_fixture_declares_no_accuracy_field_anywhere():
    # D23 is hard: this is a performance model. The schema has no accuracy
    # field and no document may smuggle one in under another name.
    banned = ("accuracy", "perplexity", "top1", "top_1", "bleu", "error_rate")
    blob = FIXTURE.read_text(encoding="utf-8").lower()
    for word in banned:
        assert word not in blob, word


def test_embedded_blob_is_the_fixture_verbatim():
    # SCHEMA.md's own instruction is "when the fixture changes, re-embed it".
    # A stale embed is a document that draws the wrong mapping under the
    # right title, which is exactly the silent divergence the note warns of.
    match = EMBED_RE.search(_html())
    assert match is not None, "atlas.html carries no atlas-embedded blob"
    assert json.loads(match.group(1)) == _fixture()


# --- P5.3 the loader --------------------------------------------------------


def test_the_whole_inline_script_parses(tmp_path):
    # A syntax error anywhere in atlas.html renders the deliverable blank with
    # nothing else to notice it: the page is one file and there is no build.
    html = _html()
    start = html.index(CORE_START)
    open_tag = html.rindex("<script>", 0, start)
    end_tag = html.index("</script>", start)
    script = tmp_path / "atlas_inline.js"
    script.write_text(html[open_tag + len("<script>") : end_tag], encoding="utf-8")
    result = subprocess.run(
        [_node(), "--check", str(script)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr


def test_loader_parses_and_the_fixture_validates_clean(tmp_path):
    out = _run_core(
        """
        const v = ATLAS.validate(doc);
        console.log(JSON.stringify({
          errors: v.errors.map(e => e.rule + ' ' + e.field),
          warnings: v.warnings.map(w => w.rule + ' ' + w.field),
        }));
        """,
        tmp_path,
    )
    report = json.loads(out)
    assert report["errors"] == []
    assert report["warnings"] == []


def test_self_test_corrupts_once_per_rule_and_every_refusal_fires(tmp_path):
    out = _run_core(
        """
        const rows = ATLAS.selfTest(doc);
        console.log(JSON.stringify({
          total: rows.length,
          cases: ATLAS.CORRUPTIONS.length,
          failed: rows.filter(r => !r.pass).map(r => r.rule + ' ' + r.what),
        }));
        """,
        tmp_path,
    )
    report = json.loads(out)
    assert report["failed"] == []
    # The clean document plus exactly one case per corruption.
    assert report["total"] == report["cases"] + 1


def test_every_rule_the_loader_can_emit_is_documented_in_schema_md(tmp_path):
    # The refusal panel tells the reader "See SCHEMA.md for the rule numbers",
    # so a rule id that fires but is undocumented sends the reader nowhere.
    out = _run_core(
        """
        console.log(JSON.stringify(Array.from(new Set(ATLAS.CORRUPTIONS.map(c => c.rule))).sort()));
        """,
        tmp_path,
    )
    rules = json.loads(out)
    html = _html()
    emitted = set(re.findall(r'[EW]\("(R\d+)"', html))
    assert emitted <= set(rules), sorted(emitted - set(rules))
    schema = SCHEMA_MD.read_text(encoding="utf-8")
    documented = set(re.findall(r"^\| (\d+) \|", schema, re.MULTILINE))
    missing = sorted(int(r[1:]) for r in emitted if r[1:] not in documented)
    assert not missing, "rules missing from SCHEMA.md: %s" % missing


@pytest.mark.parametrize(
    "rule,field,mutation",
    [
        ("R1", "provenance", "delete d.provenance;"),
        ("R24", "cards.ctt_4k_1024x4.slicing", "d.cards.ctt_4k_1024x4.slicing = true;"),
        ("R25", "slice", "d.tiles[0].slice = {index: 9, of: 2, bits: 4};"),
        ("R26", "role", 'd.systems[0].role = "mixed";'),
        ("R27", "duty_cycle", "d.macros[0].duty_cycle = 5.0;"),
        ("R6", "basis", 'd.links[0].basis = "";'),
        ("R18", "value", 'd.metrics[0].value = "lots";'),
    ],
)
def test_declared_schema_rules_are_enforced_not_just_documented(rule, field, mutation, tmp_path):
    out = _run_core(
        """
        const d = JSON.parse(JSON.stringify(doc));
        %s
        const v = ATLAS.validate(d);
        console.log(JSON.stringify(v.errors.map(e => e.rule + ' ' + e.field)));
        """
        % mutation,
        tmp_path,
    )
    hits = [e for e in json.loads(out) if e.startswith(rule + " ") and field in e]
    assert hits, json.loads(out)
