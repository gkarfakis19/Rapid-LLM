#!/usr/bin/env python3
"""Run the coherent 8.86G HF training slice for rebuttal exploration."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from validation_scripts import rebuttal_hf_train_sanity as hf

hf.SELECTED_JOB_IDS = [
    14098566,
    14098563,
    14098568,
    14098562,
    14098567,
    14098629,
    14098640,
    14098635,
    14098628,
    14098631,
    14098634,
    14098627,
    14098630,
]
hf.OUT_ROOT = hf.SCRIPT_DIR / "rebuttal_hf_train_mid8_16_slice"
hf.CASE_ROOT = hf.OUT_ROOT / "case_configs"
hf.RESULTS_CSV = hf.OUT_ROOT / "hf_train_mid8_16_results.csv"
hf.SUMMARY_TXT = hf.OUT_ROOT / "summary.txt"


if __name__ == "__main__":
    raise SystemExit(hf.main())
