#!/usr/bin/env python3
"""Study: ONE global DRAM util and ONE global network util across all devices.

Question: how much per-device holdout MAPE do we lose if dram_util and
network_util are shared across H100_SXM5 / A100_SXM4 / A100_PCIe (so the
per-device table reduces to compute + launch), instead of per-device values?

Protocol (real simulations only):
1. Candidate global pairs: dram in {0.70, 0.75, 0.80} x net in {0.80, 0.85}
   (the envelope of the three unconstrained goldens).
2. For each device, simulate every fit point at (c*, dram, net, l*) for all
   six pairs, with compute/launch frozen at the device's unconstrained golden
   (every earlier sweep showed compute insensitive to dram/net changes within
   these ranges; validated in step 4).
3. Choose the global pair minimizing the UNWEIGHTED MEAN of the three
   per-device calibration MAPEs (device-balanced, so H100's larger point
   count does not dominate), using the canonized golden calibration lists.
4. Compute re-check: at the winning pair, simulate compute +/- one axis step
   per device and let compute refit; report if it moves.
5. Report per-device calibration/holdout MAPE at the chosen global pair vs
   the unconstrained goldens.

Runs share the device_testbench caches (ordinary combo keys), so nothing is
ever re-simulated.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from validation_scripts import device_testbench as dtb  # noqa: E402

DEVICES = ("H100_SXM5", "A100_SXM4", "A100_PCIe")
GLOBAL_PAIRS = [(d, n) for d in (0.70, 0.75, 0.80) for n in (0.80, 0.85)]


def _golden(device: str) -> dict:
    path = dtb.CONFIG_ROOT / f"{device}_golden_calib.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_device(device: str):
    golden = _golden(device)
    calib_ids = set(golden["calibration_points"])
    holdout_ids = set(golden["holdout_points"])
    points = [p for p in dtb.collect_points(device) if p["fit_eligible"]]
    points = [p for p in points if p["point_id"] in calib_ids | holdout_ids]
    out_root = dtb.OUT_BASE / device
    runs_path = out_root / "runs.json"
    runs = json.loads(runs_path.read_text(encoding="utf-8")) if runs_path.exists() else {}
    return {
        "device": device,
        "golden": golden,
        "c_star": float(golden["compute_util"]),
        "l_star": float(golden["kernel_launch_overhead_s"]),
        "points": points,
        "calib": [p for p in points if p["point_id"] in calib_ids],
        "holdout": [p for p in points if p["point_id"] in holdout_ids],
        "out_root": out_root,
        "runs_path": runs_path,
        "runs": runs,
    }


def _refresh(ctx) -> None:
    for p in ctx["points"]:
        p["runs"] = ctx["runs"].get(p["point_id"], {})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("RAPID_VALIDATION_WORKERS", "6")))
    args = parser.parse_args()

    ctxs = [_load_device(d) for d in DEVICES]

    # Phase 1+2: simulate the six global pairs per device (frozen c*, l*)
    for ctx in ctxs:
        combos = [(ctx["c_star"], d, n, ctx["l_star"]) for d, n in GLOBAL_PAIRS]
        jobs = [(p, combo) for p in ctx["points"] for combo in combos]
        print(f"--- {ctx['device']}: collecting {len(GLOBAL_PAIRS)} global pairs ---",
              flush=True)
        dtb._collect(jobs, args.workers, ctx["runs"], ctx["runs_path"], ctx["out_root"])
        _refresh(ctx)

    # Phase 3: choose the pair minimizing the device-balanced mean calib MAPE
    def mean_calib_mape(pair):
        d, n = pair
        return statistics.fmean(
            dtb._mape(ctx["calib"], (ctx["c_star"], d, n, ctx["l_star"])) for ctx in ctxs
        )

    print("\nGlobal-pair scores (device-balanced mean calibration MAPE):")
    for pair in GLOBAL_PAIRS:
        per_dev = ", ".join(
            f"{ctx['device']}={dtb._mape(ctx['calib'], (ctx['c_star'], *pair, ctx['l_star'])):.2f}%"
            for ctx in ctxs
        )
        print(f"  dram={pair[0]:.2f} net={pair[1]:.2f}: mean={mean_calib_mape(pair):.2f}%  ({per_dev})")
    best_pair = min(GLOBAL_PAIRS, key=mean_calib_mape)
    d_bar, n_bar = best_pair
    print(f"\nWinning global pair: dram={d_bar:.2f}, net={n_bar:.2f}")

    # Phase 4: per-device compute re-check at the winning pair
    results = []
    for ctx in ctxs:
        cs = dtb.KNOB_AXES[ctx["device"]]["compute"]
        step = round(cs[1] - cs[0], 3)
        c_candidates = [ctx["c_star"]]
        for c in (round(ctx["c_star"] - step, 3), round(ctx["c_star"] + step, 3)):
            if 0.05 < c <= 1.0:
                c_candidates.append(c)
        combos = [(c, d_bar, n_bar, ctx["l_star"]) for c in c_candidates]
        jobs = [(p, combo) for p in ctx["points"] for combo in combos]
        dtb._collect(jobs, args.workers, ctx["runs"], ctx["runs_path"], ctx["out_root"])
        _refresh(ctx)
        best_combo = min(combos, key=lambda combo: dtb._mape(ctx["calib"], combo))
        res = {
            "device": ctx["device"],
            "global_pair": {"dram_util": d_bar, "network_util": n_bar},
            "compute_util": best_combo[0],
            "compute_moved": abs(best_combo[0] - ctx["c_star"]) > 1e-9,
            "launch_s": ctx["l_star"],
            "calib_mape": dtb._mape(ctx["calib"], best_combo),
            "holdout_mape": dtb._mape(ctx["holdout"], best_combo),
            "unconstrained_calib_mape": float(ctx["golden"]["calib_mape_pct"]),
            "unconstrained_holdout_mape": float(ctx["golden"]["holdout_mape_pct"]),
            "per_workload": {
                w: dtb._mape([p for p in ctx["points"] if p["workload"] == w], best_combo)
                for w in ("train", "inference")
            },
        }
        results.append(res)
        print(
            f"\n[{ctx['device']}] GLOBAL-PAIR RESULT: compute={best_combo[0]:.3f}"
            f"{' (moved)' if res['compute_moved'] else ''} dram={d_bar:.2f} net={n_bar:.2f} "
            f"launch={ctx['l_star']*1e6:.0f}us\n"
            f"  constrained:   calib {res['calib_mape']:.2f}%  holdout {res['holdout_mape']:.2f}%\n"
            f"  unconstrained: calib {res['unconstrained_calib_mape']:.2f}%  "
            f"holdout {res['unconstrained_holdout_mape']:.2f}%\n"
            f"  train {res['per_workload']['train']:.2f}%  "
            f"inference {res['per_workload']['inference']:.2f}%"
        )

    out = dtb.OUT_BASE / "global_dram_net_study.json"
    out.write_text(json.dumps({"pair": best_pair, "results": results}, indent=1),
                   encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
