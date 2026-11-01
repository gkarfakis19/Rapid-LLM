"""Read generated QIF evidence from the results checkout, never from git."""

import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def qif_artifact(path, *, required=True):
    """Resolve required evidence, failing unless missing results are opted out.

    ``required=False`` is for collection-time discovery and standalone gate
    presence probes. Required results are part of the test environment.
    """
    relative = Path(path)
    if relative.is_absolute():
        relative = relative.relative_to(REPO_ROOT)
    root = Path(os.environ.get("RAPID_QIF_RESULTS", REPO_ROOT.parent / "Rapid-LLM-results" / "qif"))
    resolved = root / relative
    if required and not resolved.exists():
        message = (
            f"QIF artifact missing: {resolved}; set RAPID_QIF_RESULTS to the results root "
            "(skip missing evidence only with RAPID_QIF_ALLOW_MISSING=1)"
        )
        if os.environ.get("RAPID_QIF_ALLOW_MISSING") == "1":
            pytest.skip(message)
        pytest.fail(message, pytrace=False)
    return resolved
