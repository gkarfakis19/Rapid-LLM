#!/usr/bin/env python3
"""Backend (extended_roofline) x merged mem/net study — the paper factor set.

Design: per device, residual compute c x merged utilization u (dram = net = u)
at launch 6us, on the canonized golden splits, backend ON for every run.
Separate cache root (device_testbench/<device>_opmodel/) shared with the
residual study. All real simulations.
"""
from __future__ import annotations
import json, math, os, random, sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from validation_scripts import device_testbench as dtb

# extended_roofline is the built-in default backend since the opmodel vendoring;
# no environment configuration is required.

GRID = {
    # boundary-extension probes added after round 1 (optima sat on u edges)
    "H100_SXM5": [(c, u, u, 6.0e-6) for c in (0.80, 0.85, 0.90)
                  for u in (0.75, 0.80, 0.85, 0.90)],
    "A100_SXM4": [(c, u, u, 6.0e-6) for c in (0.95, 1.00)
                  for u in (0.70, 0.75, 0.80, 0.85)],
}

def main():
    workers = int(os.environ.get("RAPID_VALIDATION_WORKERS", "14"))
    for device, combos in GRID.items():
        golden = yaml.safe_load((dtb.CONFIG_ROOT / f"{device}_golden_calib.yaml").read_text())
        calib_ids = set(golden["calibration_points"]); holdout_ids = set(golden["holdout_points"])
        points = [p for p in dtb.collect_points(device) if p["fit_eligible"]]
        points = [p for p in points if p["point_id"] in calib_ids | holdout_ids]
        calib = [p for p in points if p["point_id"] in calib_ids]
        holdout = [p for p in points if p["point_id"] in holdout_ids]
        out_root = dtb.OUT_BASE / f"{device}_opmodel"
        runs_path = out_root / "runs.json"
        runs = json.loads(runs_path.read_text()) if runs_path.exists() else {}
        jobs = [(p, c) for p in points for c in combos]
        random.Random(0).shuffle(jobs)
        print(f"--- {device}: {len(points)} pts x {len(combos)} merged combos (backend ON) ---", flush=True)
        dtb._collect(jobs, workers, runs, runs_path, out_root)
        missing = [(p, c) for p, c in jobs
                   if not (runs.get(p["point_id"], {}).get(dtb._combo_key(*c), math.nan) > 0)]
        if missing:
            print(f"retry pass: {len(missing)} at 3 workers", flush=True)
            dtb._collect(missing, 3, runs, runs_path, out_root)
        for p in points:
            p["runs"] = runs.get(p["point_id"], {})
        print(f"{'combo':<26} {'calib':>7} {'holdout':>8}  per-source", flush=True)
        for c in combos:
            srcs = "  ".join(
                f"{s}={dtb._mape([p for p in points if p['source']==s], c):.1f}%"
                for s in sorted({p['source'] for p in points}))
            print(f"c={c[0]:.2f} u={c[1]:.2f} l=6us  {dtb._mape(calib, c):6.2f}% "
                  f"{dtb._mape(holdout, c):7.2f}%  {srcs}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
