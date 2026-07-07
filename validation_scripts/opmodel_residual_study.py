#!/usr/bin/env python3
"""Residual-compute-derate study with the extended-roofline GEMM backend.

Question: with the (SMEM-prologue-fixed) opmodel extended-roofline backend
providing shape-dependent GEMM efficiency, how much per-device flat compute
derate remains necessary, and where does MAPE land vs the native goldens?

Protocol:
- Devices: H100_SXM5, A100_SXM4 (A100_PCIe dropped per project decision).
- Backend ON for every run (RAPID_GEMM_BACKEND=extended_roofline must be set
  by the caller); skinny GEMV ops (min(M,N) < 128) use the native path.
- Design per fit point: residual compute scale c in {0.80..1.00 step 0.05}
  at the device's golden (dram, net, launch), plus dram/net/launch lo-hi
  axials at c = 0.90. All real simulations; separate cache root
  (device_testbench/<device>_opmodel/runs.json) so native results stay
  untouched.
- Fit on the device's canonized golden calibration point list; evaluate the
  canonized holdout list. Per-source MAPE reported per candidate c so the
  "do we need the derate" question is answered directly.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from validation_scripts import device_testbench as dtb  # noqa: E402

assert os.environ.get("RAPID_GEMM_BACKEND") == "extended_roofline", (
    "run with RAPID_GEMM_BACKEND=extended_roofline and RAPID_OPMODEL_PATH set"
)

DEVICES = ("A100_SXM4", "H100_SXM5")
C_AXIS = (0.80, 0.85, 0.90, 0.95, 1.00)
C_MID = 0.90

GOLDEN_DNL = {  # (dram, net, launch) from the native goldens
    # Re-anchored 2026-07-07 to the post-methodology-fix goldens (physical
    # NVLink, all-hierarchical inference; commit 38c993f).
    "H100_SXM5": (0.90, 0.70, 6.0e-6),
    "A100_SXM4": (0.70, 0.80, 6.0e-6),
}
DNL_AXIALS = {  # lo/hi probes per knob
    "H100_SXM5": {"dram": (0.80, 1.00), "net": (0.60, 0.85), "launch": (3.0e-6, 12.0e-6)},
    "A100_SXM4": {"dram": (0.60, 0.80), "net": (0.65, 0.95), "launch": (3.0e-6, 12.0e-6)},
}


def _design(device):
    d0, n0, l0 = GOLDEN_DNL[device]
    combos = [(c, d0, n0, l0) for c in C_AXIS]
    ax = DNL_AXIALS[device]
    for d in ax["dram"]:
        combos.append((C_MID, d, n0, l0))
    for n in ax["net"]:
        combos.append((C_MID, d0, n, l0))
    for l in ax["launch"]:
        combos.append((C_MID, d0, n0, l))
    seen, out = set(), []
    for combo in combos:
        key = dtb._combo_key(*combo)
        if key not in seen:
            seen.add(key)
            out.append(combo)
    return out


def run_device(device, workers=7):
    golden = yaml.safe_load(
        (dtb.CONFIG_ROOT / f"{device}_golden_calib.yaml").read_text(encoding="utf-8")
    )
    calib_ids = set(golden["calibration_points"])
    holdout_ids = set(golden["holdout_points"])
    points = [p for p in dtb.collect_points(device) if p["fit_eligible"]]
    points = [p for p in points if p["point_id"] in calib_ids | holdout_ids]
    calib = [p for p in points if p["point_id"] in calib_ids]
    holdout = [p for p in points if p["point_id"] in holdout_ids]

    out_root = dtb.OUT_BASE / f"{device}_opmodel"
    out_root.mkdir(parents=True, exist_ok=True)
    runs_path = out_root / "runs.json"
    runs = json.loads(runs_path.read_text(encoding="utf-8")) if runs_path.exists() else {}

    design = _design(device)
    jobs = [(p, combo) for p in points for combo in design]
    # Spread the heavy points (GPT-1T/530B graph conversions peak at tens of
    # GB) across the run instead of queueing each point's combos back to back.
    random.Random(0).shuffle(jobs)
    print(f"--- {device}: {len(points)} fit points x {len(design)} combos ---", flush=True)
    dtb._collect(jobs, workers, runs, runs_path, out_root)
    # Low-concurrency retry pass for OOM-killed stragglers.
    for p in points:
        p["runs"] = runs.get(p["point_id"], {})
    missing = [(p, combo) for p, combo in jobs
               if not (runs.get(p["point_id"], {}).get(dtb._combo_key(*combo), math.nan) > 0)]
    if missing:
        print(f"retry pass: {len(missing)} failed combos at 3 workers", flush=True)
        dtb._collect(missing, 3, runs, runs_path, out_root)
    for p in points:
        p["runs"] = runs.get(p["point_id"], {})

    best = min(design, key=lambda combo: dtb._mape(calib, combo))
    print(f"\n[{device}] BACKEND GOLDEN combo: c={best[0]:.2f} d={best[1]:.2f} "
          f"n={best[2]:.2f} launch={best[3]*1e6:.1f}us")
    print(f"  calib MAPE {dtb._mape(calib, best):.2f}%  holdout MAPE {dtb._mape(holdout, best):.2f}%")
    print(f"  (native golden: calib {golden['calib_mape_pct']}%  holdout {golden['holdout_mape_pct']}%, "
          f"compute={golden['compute_util']})")
    d0, n0, l0 = GOLDEN_DNL[device]
    print(f"  residual-c sweep at golden (d,n,l) — calib/holdout MAPE and per-source:")
    for c in C_AXIS:
        combo = (c, d0, n0, l0)
        line = f"    c={c:.2f}: calib {dtb._mape(calib, combo):6.2f}%  holdout {dtb._mape(holdout, combo):6.2f}%"
        for source in sorted({p['source'] for p in points}):
            subset = [p for p in points if p['source'] == source]
            line += f"  {source}={dtb._mape(subset, combo):.1f}%"
        print(line, flush=True)
    return best


def main():
    workers = int(os.environ.get("RAPID_VALIDATION_WORKERS", "18"))
    for device in DEVICES:
        run_device(device, workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
