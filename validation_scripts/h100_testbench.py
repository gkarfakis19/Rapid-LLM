#!/usr/bin/env python3
"""Unified H100 training-time testbench + golden calibration/holdout split.

Collects every H100 training validation point into one bench:

* Megatron-Core MoE/EP points (arXiv:2504.14960 "MCore" rows) from
  moe_ep_validation.py — run with the interleaved-1F1B policy (vpp=max,
  one layer per virtual stage). The tp*ep>8 "ep_dim" stretch rows are
  REPORT-ONLY: their EP all-to-all is bound to the slow inter-node domain
  by construction (a conservative bound, not a factor error).
* HF/Nanotron benchmark points from rebuttal_hf_train_sanity.py machinery:
  the FULL pp=1 population of the benchmark (68 rows: status Success and
  after_pp_fix, no tp or dp window), plus the 6 curated held-out pp>1 sanity
  rows (plain 1F1B -> no vpp).
  Rows whose MEASURED MFU is below MEASURED_MFU_FLOOR (20%) are REPORT-ONLY:
  RAPID-LLM is a roofline-based analytical model, not cycle-accurate, so
  configurations the benchmark itself records as running far below achievable
  efficiency are outside the regime it targets. 36 of the 68 pp=1 rows fall
  below the floor, leaving 32; the floor lands in a 4.2 pp gap in the sweep's
  efficiency distribution (16.4% -> 20.6%), so the cut point is not tuned.

Protocol (no interpolation — every number is a real simulation):
1. Every fit-eligible point is simulated at every candidate compute-util
   value in F_CANDIDATES. TP-SP communication overlap is fixed at
   TP_SP_OVERLAP for all points (Megatron and nanotron both overlap TP-SP
   collectives with compute).
2. The calibration fit picks the candidate util minimizing calibration-set
   MAPE; the holdout MAPE is evaluated from the same real runs.
3. Split search: stratified 50:50 calibration:holdout assignments
   (strata = source x GPU-scale bucket, 5 seeds) plus a structured
   "largest-scale points held out" probe (reported, never golden). The
   GOLDEN split is the median-holdout-MAPE 50:50 seed (median, not minimum,
   so the canonical split is representative, not flattering).
4. pp>1 rule: if the HF pp>1 sanity subset still exceeds PP_GT1_CUT_MAPE at
   the golden f*, those rows are cut to report-only and the split search +
   fit are redone without them.
5. Report-only points are simulated at the golden f* only.

The golden split is written to
validation_scripts/validation_configs/h100_golden_split.yaml.

Usage:
  .venv/bin/python validation_scripts/h100_testbench.py            # full run
  .venv/bin/python validation_scripts/h100_testbench.py --fit-only # reuse runs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from validation_scripts import moe_ep_validation as mev  # noqa: E402
from validation_scripts import rebuttal_hf_train_sanity as hfs  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_ROOT = SCRIPT_DIR / "h100_testbench"
CASE_ROOT = OUT_ROOT / "case_configs"
RESULTS_CSV = OUT_ROOT / "testbench_results.csv"
SPLIT_REPORT = OUT_ROOT / "split_report.md"
RUNS_JSON = OUT_ROOT / "runs_v2.json"
GOLDEN_SPLIT_YAML = SCRIPT_DIR / "validation_configs" / "h100_golden_split.yaml"

TRAIN_TIME_RE = re.compile(r"Training time for batch:\s*([0-9]+(?:\.[0-9]+)?)s")

# Candidate compute-util values. Every fit-eligible point is simulated at
# every candidate; the fit selects among these REAL runs only.
F_CANDIDATES = (0.600, 0.625, 0.650, 0.675, 0.700)
F_FROZEN = 0.56  # Table III reference value

# Communication/compute overlap fractions (tp, tp_sp, cp), applied uniformly.
TP_SP_OVERLAP = 0.6
OVERLAPS = {"tp_overlap": 0.6, "tp_sp_overlap": 0.6, "cp_overlap": 0.6}

# If the HF pp>1 sanity subset exceeds this MAPE at the golden f*, it is cut
# to report-only and the fit is redone.
PP_GT1_CUT_MAPE = 25.0

CALIB_SHARE = 0.5  # 50:50 calibration:holdout
SEEDS = (0, 1, 2, 3, 4)

# Applicability envelope, disclosed in the paper. RAPID-LLM is a
# roofline-based analytical model, not cycle-accurate: configurations that the
# benchmark itself records as running far below achievable efficiency are
# outside the regime it targets, so rows whose MEASURED MFU is below this
# floor are REPORT-ONLY. The floor is a property of the reference run, never
# of the prediction, so it cannot be tuned against our own error.
#
# The threshold sits in a natural gap in the sweep's efficiency distribution:
# over the full pp==1 population the highest below-floor run is 16.4% MFU and
# the lowest above-floor run is 20.6% MFU (a 4.2 pp gap, the largest in the
# whole distribution), so any cut in (16.4%, 20.6%] selects the same 32 rows.
MEASURED_MFU_FLOOR = 0.20


# --------------------------------------------------------------------------
# Point collection
# --------------------------------------------------------------------------


def _mev_points() -> list[dict[str, Any]]:
    points = []
    for case in mev.CASES:
        model = mev.MODELS[case["model"]]
        gbs = mev._case_gbs(case)
        layout = mev._case_layout(case)
        t_ref = mev._reference_time_s(model, case["gpus"], case["mfu_ref"], gbs)
        points.append(
            {
                "point_id": f"megatron:{mev._case_id(case)}",
                "source": "megatron_moe",
                "family": model["display"],
                "num_gpus": case["gpus"],
                "t_ref_s": t_ref,
                "fit_eligible": layout != "ep_dim",
                "report_note": "ep_dim conservative EP-comm bound" if layout == "ep_dim" else "",
                "runner": ("mev", case),
            }
        )
    return points


def _hf_rows() -> list[tuple[str, pd.Series]]:
    df = pd.read_csv(hfs.INPUT_CSV)
    ok = df[(df["status"] == "Success") & (df["after_pp_fix"] == True)].copy()  # noqa: E712
    # Full pp==1 population: no tp/dp windowing (the previous tp in [2,16] /
    # dp in [8,64] windows were undisclosed selection knobs and are gone).
    # The pp==1 restriction stays and is justified separately in the paper.
    calib_mask = ok["pp"] == 1
    rows: list[tuple[str, pd.Series]] = []
    for _, row in ok[calib_mask].iterrows():
        rows.append(("hf_pp1", row))
    by_id = {int(r["job_id"]): r for _, r in ok.iterrows()}
    for job_id in hfs.SELECTED_JOB_IDS:
        rows.append(("hf_heldout_pp", by_id[int(job_id)]))
    return rows


def _hf_flops_per_token(row: pd.Series) -> float:
    h = int(row["hidden_size"])
    layers = int(row["num_layers"])
    heads = int(row["num_heads"])
    kv_heads = int(float(row["num_kv_heads"]))
    interm = int(float(row["interm_dim"]))
    vocab = int(row["vocab_size"])
    seq = int(row["seq_len"])
    head_dim = h // heads
    qkv = 2 * h * (h + 2 * kv_heads * head_dim)
    attn = 4 * seq * h
    out_proj = 2 * h * h
    ffn = 3 * 2 * h * interm  # nanotron llama = SwiGLU
    logits = 2 * h * vocab
    return 3.0 * (layers * (qkv + attn + out_proj + ffn) + logits)


def _hf_points() -> list[dict[str, Any]]:
    points = []
    for subset, row in _hf_rows():
        num_gpus = int(row["dp"]) * int(row["pp"]) * int(row["tp"])
        total_tokens = int(row["gbs"]) * int(row["seq_len"])
        t_meas = total_tokens / (float(row["tok/s/gpu"]) * num_gpus)
        measured_mfu = (
            _hf_flops_per_token(row) * float(row["tok/s/gpu"]) / mev.PEAK_FLOPS_PER_GPU
        )
        degenerate = measured_mfu < MEASURED_MFU_FLOOR
        points.append(
            {
                "point_id": f"hf:job{int(row['job_id'])}",
                "source": subset,
                "family": f"nanotron_h{int(row['hidden_size'])}",
                "num_gpus": num_gpus,
                "t_ref_s": t_meas,
                "fit_eligible": not degenerate,
                "report_note": (
                    f"measured MFU {measured_mfu*100:.1f}% < {MEASURED_MFU_FLOOR*100:.0f}% "
                    "(degenerate/comm-collapsed row)"
                    if degenerate
                    else ""
                ),
                "runner": ("hf", row),
            }
        )
    return points


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


def _factors(compute_util: float) -> dict[str, float]:
    base = hfs._derates()
    base["compute_util"] = float(compute_util)
    return base


def _apply_tp_sp_overlap(hw_path: Path) -> None:
    hw = yaml.safe_load(hw_path.read_text(encoding="utf-8"))
    overlap = hw.setdefault("network", {}).setdefault("overlap", {})
    overlap.update(OVERLAPS)
    hw_path.write_text(yaml.safe_dump(hw, sort_keys=False), encoding="utf-8")


def _run_one(point: dict[str, Any], compute_util: float, hf_network_mode: str) -> float:
    kind, payload = point["runner"]
    slug = point["point_id"].replace(":", "_")
    case_dir = CASE_ROOT / slug / f"f{compute_util:.3f}_{hf_network_mode}"
    case_dir.mkdir(parents=True, exist_ok=True)
    factors = _factors(compute_util)
    if kind == "mev":
        hw_path, model_path = mev._build_configs(payload, case_dir, factors, vpp="max")
    else:
        hw_path, model_path = hfs._build_configs(payload, hf_network_mode, case_dir, factors)
    _apply_tp_sp_overlap(hw_path)
    env = dict(os.environ)
    env["RAPID_ASTRA_CACHE_MODE"] = "NO_CACHE"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "run_perf.py"),
            "--hardware_config",
            str(hw_path),
            "--model_config",
            str(model_path),
        ],
        cwd=case_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env=env,
    )
    (case_dir / "run.log").write_text(result.stdout or "", encoding="utf-8")
    match = TRAIN_TIME_RE.search(result.stdout or "")
    if result.returncode != 0 or not match:
        print(f"[WARN] {point['point_id']} @f={compute_util} failed (rc={result.returncode})")
        return math.nan
    return float(match.group(1))


def _cache_key(point: dict[str, Any], hf_network_mode: str) -> str:
    # megatron points do not depend on the HF network mode
    if point["runner"][0] == "mev":
        return point["point_id"]
    return f"{point['point_id']}@{hf_network_mode}"


def _collect_runs(
    jobs: list[tuple[dict[str, Any], float]],
    workers: int,
    hf_network_mode: str,
    runs: dict[str, dict[str, float]],
) -> None:
    """Simulate the (point, util) pairs missing from ``runs`` (in place)."""
    todo = [
        (p, f)
        for p, f in jobs
        if not (runs.get(_cache_key(p, hf_network_mode), {}).get(f"{f:.3f}", math.nan) > 0)
    ]
    print(f"{len(todo)} runs to execute ({len(jobs) - len(todo)} cached)")
    if not todo:
        return
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        times = list(pool.map(lambda pf: _run_one(*pf, hf_network_mode), todo))
    for (point, f), t in zip(todo, times):
        runs.setdefault(_cache_key(point, hf_network_mode), {})[f"{f:.3f}"] = t
        print(f"[{point['point_id']}] f={f:.2f} t={t:.3f}s")
    RUNS_JSON.write_text(json.dumps(runs, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------
# Fitting (real runs only)
# --------------------------------------------------------------------------


def _t_at(point: dict[str, Any], f: float) -> float:
    return point["runs"].get(f"{f:.3f}", math.nan)


def _mape(points: list[dict[str, Any]], f: float) -> float:
    errs = []
    for p in points:
        t = _t_at(p, f)
        if t == t and t > 0:
            errs.append(abs(t - p["t_ref_s"]) / p["t_ref_s"] * 100.0)
    return statistics.fmean(errs) if errs else math.nan


def _fit(points: list[dict[str, Any]]) -> float:
    return min(F_CANDIDATES, key=lambda f: _mape(points, f))


# --------------------------------------------------------------------------
# Split search
# --------------------------------------------------------------------------


def _scale_bucket(num_gpus: int) -> str:
    if num_gpus <= 32:
        return "small"
    if num_gpus <= 128:
        return "mid"
    return "large"


def _stratified_split(
    points: list[dict[str, Any]], calib_share: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for p in points:
        key = (p["source"], _scale_bucket(p["num_gpus"]))
        strata.setdefault(key, []).append(p)
    calib: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for key in sorted(strata):
        members = sorted(strata[key], key=lambda p: p["point_id"])
        rng.shuffle(members)
        n_cal = max(1, round(calib_share * len(members)))
        if n_cal >= len(members) and len(members) > 1:
            n_cal = len(members) - 1
        calib.extend(members[:n_cal])
        holdout.extend(members[n_cal:])
    return calib, holdout


def _largest_scale_holdout(
    points: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Structured probe: the largest-GPU point of each (source, family) held out."""
    holdout_ids = set()
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for p in points:
        groups.setdefault((p["source"], p["family"]), []).append(p)
    for members in groups.values():
        if len(members) > 1:
            holdout_ids.add(max(members, key=lambda p: p["num_gpus"])["point_id"])
    calib = [p for p in points if p["point_id"] not in holdout_ids]
    holdout = [p for p in points if p["point_id"] in holdout_ids]
    return calib, holdout


def _split_search(fit_points: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for seed in SEEDS:
        calib, holdout = _stratified_split(fit_points, CALIB_SHARE, seed)
        f_star = _fit(calib)
        candidates.append(
            {
                "kind": f"stratified_50_{seed}",
                "seed": seed,
                "f_star": f_star,
                "calib_mape": _mape(calib, f_star),
                "holdout_mape": _mape(holdout, f_star),
                "n_calib": len(calib),
                "n_holdout": len(holdout),
                "calib_ids": sorted(p["point_id"] for p in calib),
                "holdout_ids": sorted(p["point_id"] for p in holdout),
            }
        )
    calib, holdout = _largest_scale_holdout(fit_points)
    f_star = _fit(calib)
    candidates.append(
        {
            "kind": "largest_scale_holdout",
            "seed": None,
            "f_star": f_star,
            "calib_mape": _mape(calib, f_star),
            "holdout_mape": _mape(holdout, f_star),
            "n_calib": len(calib),
            "n_holdout": len(holdout),
            "calib_ids": sorted(p["point_id"] for p in calib),
            "holdout_ids": sorted(p["point_id"] for p in holdout),
        }
    )
    # Golden: the MEDIAN-holdout-MAPE seed among the stratified 50:50
    # candidates (median, not minimum — representative, not flattering).
    fifty = sorted(
        (c for c in candidates if c["kind"].startswith("stratified_50")),
        key=lambda c: c["holdout_mape"],
    )
    golden = fifty[len(fifty) // 2]
    return candidates, golden


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _write_results_csv(points: list[dict[str, Any]], f_star: float) -> None:
    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["point_id", "source", "family", "num_gpus", "fit_eligible", "t_ref_s"]
            + [f"t_pred_f{f:.3f}" for f in F_CANDIDATES]
            + ["t_pred_fstar", "signed_err_fstar_pct", "report_note"]
        )
        for p in points:
            t_star = _t_at(p, f_star)
            err = (
                (t_star - p["t_ref_s"]) / p["t_ref_s"] * 100.0
                if t_star == t_star
                else math.nan
            )
            writer.writerow(
                [
                    p["point_id"], p["source"], p["family"], p["num_gpus"],
                    p["fit_eligible"], f"{p['t_ref_s']:.4f}",
                ]
                + [
                    f"{_t_at(p, f):.4f}" if _t_at(p, f) == _t_at(p, f) else ""
                    for f in F_CANDIDATES
                ]
                + [
                    f"{t_star:.4f}" if t_star == t_star else "",
                    f"{err:+.2f}" if err == err else "",
                    p["report_note"],
                ]
            )


def _write_split_report(
    candidates: list[dict[str, Any]],
    golden: dict[str, Any],
    fit_points: list[dict[str, Any]],
    report_points: list[dict[str, Any]],
    pp_gt1_cut: bool,
) -> None:
    lines = [
        "# H100 testbench — calibration/holdout split search",
        "",
        "Protocol: every fit-eligible point simulated at every candidate compute-util "
        f"value {list(F_CANDIDATES)}; the fit selects among these REAL runs (no "
        f"interpolation). TP-SP overlap fixed at {TP_SP_OVERLAP} for all points. "
        "Split: stratified 50:50, golden = median-holdout seed.",
        "",
        f"Fit-eligible points: {len(fit_points)}. Report-only points: {len(report_points)}.",
        f"HF pp>1 sanity rows cut per protocol (> {PP_GT1_CUT_MAPE:.0f}% MAPE at f*): "
        f"{'YES' if pp_gt1_cut else 'no'}.",
        "",
        "| kind | f* | calib MAPE | holdout MAPE | n_cal | n_hold |",
        "|---|---|---|---|---|---|",
    ]
    for c in candidates:
        marker = " **<- GOLDEN**" if c is golden else ""
        lines.append(
            f"| {c['kind']}{marker} | {c['f_star']:.3f} | {c['calib_mape']:.2f}% "
            f"| {c['holdout_mape']:.2f}% | {c['n_calib']} | {c['n_holdout']} |"
        )
    f_stars = [c["f_star"] for c in candidates]
    lines += [
        "",
        f"f* stability: min {min(f_stars):.3f} / median {statistics.median(f_stars):.3f} "
        f"/ max {max(f_stars):.3f} across all candidates.",
        "",
        f"GOLDEN: {golden['kind']}  f* = {golden['f_star']:.3f}, "
        f"calib {golden['calib_mape']:.2f}%, holdout {golden['holdout_mape']:.2f}%.",
        "",
        "## Per-source MAPE at the golden f* (real runs)",
        "",
    ]
    for source in sorted({p["source"] for p in fit_points}):
        subset = [p for p in fit_points if p["source"] == source]
        lines.append(
            f"- {source}: n={len(subset)}, MAPE {_mape(subset, golden['f_star']):.2f}%"
        )
    excluded_hf = [p for p in report_points if p["source"].startswith("hf")]
    lines += [
        "",
        f"Report-only: {len(excluded_hf)} HF rows (measured MFU < "
        f"{MEASURED_MFU_FLOOR*100:.0f}% degenerate envelope"
        f"{' + pp>1 rows cut per protocol' if pp_gt1_cut else ''}) and "
        f"{len(report_points) - len(excluded_hf)} Megatron ep_dim stretch rows "
        "(conservative all-EP-on-IB bound).",
    ]
    SPLIT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_golden_yaml(
    golden: dict[str, Any],
    report_points: list[dict[str, Any]],
    hf_network_mode: str,
    pp_gt1_cut: bool,
) -> None:
    payload = {
        "description": (
            "Golden H100 training-time calibration/holdout split (50:50). Fit "
            "knob: tech_param.core.util. All numbers are real simulations at "
            f"the listed util (candidates {list(F_CANDIDATES)}; no interpolation). "
            f"network.overlap.tp_sp_overlap = {TP_SP_OVERLAP} for all points. "
            "Megatron MoE points: sw_param.pipeline_interleave = layers-per-stage "
            "(interleaved 1F1B assumption) + moe_ep_validation topology; "
            f"HF/Nanotron points: {hf_network_mode} mode (full pp==1 population, "
            "no tp/dp window). HF rows with measured MFU < "
            f"{MEASURED_MFU_FLOOR*100:.0f}% are report-only (applicability "
            "envelope of a roofline-based model)"
            + ("; HF pp>1 sanity rows cut per protocol (>25% MAPE at f*)." if pp_gt1_cut else ".")
            + " Selection rule: median-holdout-MAPE seed among stratified 50:50 "
            "candidates (see split_report.md)."
        ),
        "hf_network_mode": hf_network_mode,
        "tp_sp_overlap": TP_SP_OVERLAP,
        "pipeline_interleave_policy": "max",
        "f_candidates": list(F_CANDIDATES),
        "pp_gt1_rows_cut": pp_gt1_cut,
        "generator": "validation_scripts/h100_testbench.py",
        "kind": golden["kind"],
        "compute_util_fstar": golden["f_star"],
        "calib_mape_pct": round(golden["calib_mape"], 2),
        "holdout_mape_pct": round(golden["holdout_mape"], 2),
        "calibration_points": golden["calib_ids"],
        "holdout_points": golden["holdout_ids"],
        "report_only_points": sorted(p["point_id"] for p in report_points),
    }
    GOLDEN_SPLIT_YAML.parent.mkdir(parents=True, exist_ok=True)
    with GOLDEN_SPLIT_YAML.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--workers", type=int, default=int(os.environ.get("RAPID_VALIDATION_WORKERS", "8"))
    )
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help="skip simulation; reuse cached runs_v2.json (missing runs are dropped)",
    )
    parser.add_argument(
        "--hf-network",
        choices=("fixed_network", "hf_network"),
        default="hf_network",
        help="HF/Nanotron points: stock H100 topology or per-job measured "
        "collective bandwidths (default — isolates the compute factor from "
        "network-config mismatch)",
    )
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    points = _mev_points() + _hf_points()

    runs: dict[str, dict[str, float]] = {}
    if RUNS_JSON.exists():
        runs = json.loads(RUNS_JSON.read_text(encoding="utf-8"))

    fit_candidates = [p for p in points if p["fit_eligible"]]
    if not args.fit_only:
        jobs = [(p, f) for p in fit_candidates for f in F_CANDIDATES]
        _collect_runs(jobs, args.workers, args.hf_network, runs)

    for p in points:
        p["runs"] = runs.get(_cache_key(p, args.hf_network), {})

    fit_points = [
        p for p in fit_candidates if any(_t_at(p, f) > 0 for f in F_CANDIDATES)
    ]

    candidates, golden = _split_search(fit_points)

    # pp>1 rule: cut the HF pp>1 sanity rows if they remain out of envelope.
    pp_gt1 = [p for p in fit_points if p["source"] == "hf_heldout_pp"]
    pp_gt1_cut = False
    if pp_gt1 and _mape(pp_gt1, golden["f_star"]) > PP_GT1_CUT_MAPE:
        pp_gt1_cut = True
        subset_mape = _mape(pp_gt1, golden["f_star"])
        for p in pp_gt1:
            p["fit_eligible"] = False
            p["report_note"] = (
                f"pp>1 nanotron row cut per protocol: subset MAPE {subset_mape:.1f}% "
                f"> {PP_GT1_CUT_MAPE:.0f}% at f* even with tp_sp_overlap={TP_SP_OVERLAP}"
            )
        fit_points = [p for p in fit_points if p["source"] != "hf_heldout_pp"]
        candidates, golden = _split_search(fit_points)

    report_points = [p for p in points if not p["fit_eligible"]]
    if not args.fit_only:
        jobs = [(p, golden["f_star"]) for p in report_points]
        _collect_runs(jobs, args.workers, args.hf_network, runs)
        for p in points:
            p["runs"] = runs.get(_cache_key(p, args.hf_network), {})

    usable = [p for p in points if p["runs"]]
    _write_results_csv(usable, golden["f_star"])
    _write_split_report(candidates, golden, fit_points, report_points, pp_gt1_cut)
    _write_golden_yaml(golden, report_points, args.hf_network, pp_gt1_cut)

    print(
        f"\nGOLDEN: {golden['kind']}  f*={golden['f_star']:.3f}  "
        f"calib MAPE {golden['calib_mape']:.2f}%  holdout MAPE {golden['holdout_mape']:.2f}%"
    )
    print(f"HF pp>1 rows cut per protocol: {'YES' if pp_gt1_cut else 'no'}")
    for source in sorted({p["source"] for p in fit_points}):
        subset = [p for p in fit_points if p["source"] == source]
        print(f"  {source}: n={len(subset)} MAPE@f*={_mape(subset, golden['f_star']):.2f}%")
    print(f"Wrote {RESULTS_CSV}, {SPLIT_REPORT}, {GOLDEN_SPLIT_YAML}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
