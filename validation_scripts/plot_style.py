"""Shared plotting style constants for validation figures."""

from __future__ import annotations

from typing import Dict

IEEE_HALF_COLUMN_WIDTH_IN = 3.5
IEEE_FONT_SIZE_PT = 8
IEEE_AXIS_TITLE_SIZE_PT = 10
IEEE_TITLE_SIZE_PT = 10
IEEE_DPI = 200


def ieee_rc_params() -> Dict[str, float]:
    """Matplotlib rc params for consistent IEEE-style sizing."""
    return {
        "font.size": IEEE_FONT_SIZE_PT,
        "axes.titlesize": IEEE_TITLE_SIZE_PT,
        "axes.labelsize": IEEE_AXIS_TITLE_SIZE_PT,
        "xtick.labelsize": IEEE_FONT_SIZE_PT,
        "ytick.labelsize": IEEE_FONT_SIZE_PT,
        "legend.fontsize": IEEE_FONT_SIZE_PT,
        "legend.title_fontsize": IEEE_FONT_SIZE_PT,
        "figure.titlesize": IEEE_TITLE_SIZE_PT,
    }
