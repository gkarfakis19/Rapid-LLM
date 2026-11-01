"""The results-root contract is independent of any particular report schema."""

import pytest

from tests.qif_artifacts import REPO_ROOT, qif_artifact


def test_default_results_root(monkeypatch):
    monkeypatch.delenv("RAPID_QIF_RESULTS", raising=False)
    relative = "docs/qif/atlas/example.json"
    assert qif_artifact(relative, required=False) == REPO_ROOT.parent / "Rapid-LLM-results" / "qif" / relative


def test_override_reads_exact_bytes(monkeypatch, tmp_path):
    monkeypatch.setenv("RAPID_QIF_RESULTS", str(tmp_path))
    relative = "docs/qif/atlas/example.json"
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b'{"historical": true}\n')
    assert qif_artifact(REPO_ROOT / relative).read_bytes() == target.read_bytes()


def test_missing_result_skips_with_path_and_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("RAPID_QIF_RESULTS", str(tmp_path))
    with pytest.raises(pytest.skip.Exception) as missing:
        qif_artifact("docs/qif/atlas/missing.json")
    assert "RAPID_QIF_RESULTS" in str(missing.value)
    assert str(tmp_path / "docs/qif/atlas/missing.json") in str(missing.value)
