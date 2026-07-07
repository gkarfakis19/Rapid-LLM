#!/usr/bin/env python3
"""Presentation-constrained factor sets: d = n = u (merged mem/net factor).

Evaluates a pre-registered grid of 'presentable' candidates — round compute
values (including slightly above each device's granular golden) x a merged
memory/network utilization u, launch fixed at the universal 6us — on the
SAME canonized golden calibration/holdout splits, all real simulations.
Reports the menu; selection is the user's call. No split re-search: this is
an evaluation of constrained candidates against the existing protocol splits.
"""
from __future__ import annotations
import json, math, os, sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from validation_scripts import device_testbench as dtb

GRID = {
    "H100_SXM5": [(c, u, u, 6.0e-6) for c in (0.600, 0.625, 0.650)
                  for u in (0.75, 0.80, 0.85)],
    "A100_SXM4": [(1.000, u, u, 6.0e-6) for u in (0.70, 0.75, 0.80)],
}

def main():
    workers = int(os.environ.get("RAPID_VALIDATION_WORKERS", "16"))
    results = {}
    for device, combos in GRID.items():
        golden = yaml.safe_load((dtb.CONFIG_ROOT / f"{device}_golden_calib.yaml").read_text())
        calib_ids = set(golden["calibration_points"]); holdout_ids = set(golden["holdout_points"])
        points = [p for p in dtb.collect_points(device) if p["fit_eligible"]]
        points = [p for p in points if p["point_id"] in calib_ids | holdout_ids]
        calib = [p for p in points if p["point_id"] in calib_ids]
        holdout = [p for p in points if p["point_id"] in holdout_ids]
        out_root = dtb.OUT_BASE / device
        runs_path = out_root / "runs.json"
        runs = json.loads(runs_path.read_text()) if runs_path.exists() else {}
        jobs = [(p, combo) for p in points for combo in combos]
        import random; random.Random(0).shuffle(jobs)
        print(f"--- {device} ---", flush=True)
        dtb._collect(jobs, workers, runs, runs_path, out_root)
        missing = [(p, c) for p, c in jobs
                   if not (runs.get(p["point_id"], {}).get(dtb._combo_key(*c), math.nan) > 0)]
        if missing:
            dtb._collect(missing, 3, runs, runs_path, out_root)
        for p in points:
            p["runs"] = runs.get(p["point_id"], {})
        print(f"{'combo':<28} {'calib':>7} {'holdout':>8}  per-source")
        for combo in combos:
            srcs = "  ".join(
                f"{s}={dtb._mape([p for p in points if p['source']==s], combo):.1f}%"
                for s in sorted({p['source'] for p in points}))
            print(f"c={combo[0]:.3f} u={combo[1]:.2f} l=6us   "
                  f"{dtb._mape(calib, combo):6.2f}% {dtb._mape(holdout, combo):7.2f}%  {srcs}",
                  flush=True)
        results[device] = {dtb._combo_key(*c): (dtb._mape(calib, c), dtb._mape(holdout, c)) for c in combos}
    (dtb.OUT_BASE / "presentable_factors_study.json").write_text(json.dumps(results, indent=1))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
