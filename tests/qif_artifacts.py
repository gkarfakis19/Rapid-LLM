"""Read generated QIF evidence from the results checkout, never from git."""

import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def qif_artifact(path, *, required=True):
    """Resolve a repo-relative artifact; only its consuming test may skip.

    ``required=False`` is for collection-time discovery and standalone gate
    presence probes. Pytest readers use the default for an explicit skip.
    """
    relative = Path(path)
    if relative.is_absolute():
        relative = relative.relative_to(REPO_ROOT)
    root = Path(os.environ.get("RAPID_QIF_RESULTS", REPO_ROOT.parent / "Rapid-LLM-results" / "qif"))
    resolved = root / relative
    if required and not resolved.exists():
        pytest.skip(f"QIF artifact missing: {resolved}; set RAPID_QIF_RESULTS to the results root")
    return resolved
