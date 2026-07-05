#!/usr/bin/env python3
"""Run the Table III factor sensitivity test on neighboring validation cases.

This expands the four-case rebuttal sensitivity test to every row in the same
small validation CSVs that supplied those four anchors. It still performs only a
model-output sensitivity characterization: no new measured ground truth, no new
benchmarks, no recalibration, and the same one-at-a-time +/-10% factor variants.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shlex
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import validation_scripts.rebuttal_factor_sensitivity as base  # noqa: E402


OUT_ROOT = SCRIPT_DIR / "rebuttal_factor_sensitivity_neighbors"
CASE_ROOT = OUT_ROOT / "case_configs"
RESULTS_CSV = OUT_ROOT / "factor_sensitivity_results.csv"
SUMMARY_MD = OUT_ROOT / "factor_sensitivity_summary.md"
COMMANDS_TXT = OUT_ROOT / "commands.txt"


def _slug(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]+", "_", text.lower())).strip("_")


def _read_csv_rows(path: Path, *, encoding: str = "utf-8") -> list[dict[str, str]]:
    with path.open(newline="", encoding=encoding) as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _set_output_paths() -> None:
    base.OUT_ROOT = OUT_ROOT
    base.CASE_ROOT = CASE_ROOT
    base.RESULTS_CSV = RESULTS_CSV
    base.SUMMARY_MD = SUMMARY_MD
    base.COMMANDS_TXT = COMMANDS_TXT


def _format_shell_command(run: base.PlannedRun) -> str:
    return (
        f"cd {shlex.quote(str(run.run_dir))} && "
        f"{shlex.quote(str(base.PYTHON))} {shlex.quote(str(base.RUN_PERF))} "
        f"--hardware_config {shlex.quote(str(run.hardware_path))} "
        f"--model_config {shlex.quote(str(run.model_path))}"
    )


def _with_shell_commands(planned: list[base.PlannedRun]) -> list[base.PlannedRun]:
    return [replace(run, command=_format_shell_command(run)) for run in planned]


def _write_commands(planned: list[base.PlannedRun]) -> None:
    COMMANDS_TXT.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{run.case.case_id},{run.variant}: {run.command}" for run in planned]
    COMMANDS_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _refresh_existing_result_commands(planned: list[base.PlannedRun]) -> None:
    if not RESULTS_CSV.exists():
        return
    command_by_key = {(run.case.case_id, run.variant): run.command for run in planned}
    with RESULTS_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if not fieldnames or "command" not in fieldnames:
        return
    changed = False
    for row in rows:
        command = command_by_key.get((row["case_id"], row["variant"]))
        if command and row.get("command") != command:
            row["command"] = command
            changed = True
    if changed:
        with RESULTS_CSV.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _build_nvidia_training_neighbors() -> list[base.CaseSpec]:
    from validation_scripts import nvidia_train_validation as ntv

    source_csv = SCRIPT_DIR / "train_validation_data" / "nvidia_train_validation_cases.csv"
    default_hw = SCRIPT_DIR / "validation_configs" / "hardware-config" / "a100_80GB_train_validation.yaml"
    cases: list[base.CaseSpec] = []
    for row in _read_csv_rows(source_csv):
        model = row["model"]
        dp = int(row["dp"])
        tp = int(row["tp"])
        pp = int(row["pp"])
        cp = int(row["cp"])
        batch = int(row["batch"])
        mb = int(row["mb"])
        tp_sp = str(row["tp_sp"]).strip().lower() in {"1", "true", "yes", "y", "t"}
        recomputation = row["recomputation"]
        base_model_path = ntv._resolve_model_config(model)  # type: ignore[attr-defined]
        base_hw_path = ntv._resolve_hw_config(row, default_hw)  # type: ignore[attr-defined]
        model_overrides, hw_overrides = ntv._build_overrides(row)  # type: ignore[attr-defined]
        model_cfg = base._merge_configs(
            base._load_yaml(base_model_path),
            model_overrides,
            merge_lists_by_id=True,
        )
        hw_cfg = base._merge_configs(
            base._load_yaml(base_hw_path),
            hw_overrides,
            merge_lists_by_id=True,
        )
        case_id = _slug(
            "train_{}_{}_b{}_mb{}_dp{}_tp{}_pp{}_cp{}_tpsp{}_{}".format(
                row["device"],
                model,
                batch,
                mb,
                dp,
                tp,
                pp,
                cp,
                int(tp_sp),
                recomputation,
            )
        )
        cases.append(
            base.CaseSpec(
                case_id=case_id,
                regime="NVIDIA training validation neighbor",
                source_csv=source_csv,
                model=model,
                training_or_inference="training",
                gpu_type="A100 SXM4 80GB",
                device_type="A100_SXM4",
                num_gpus=dp * tp * pp * cp,
                parallelism=(
                    f"DP={dp}, TP={tp}, PP={pp}, CP={cp}, "
                    f"TP_SP={tp_sp}, recomputation={recomputation}, batch={batch}, mb={mb}"
                ),
                model_config=model_cfg,
                hardware_config=hw_cfg,
            )
        )
    return cases


def _build_inference_neighbors() -> list[base.CaseSpec]:
    from validation_scripts import nvidia_inf

    source_csv = SCRIPT_DIR / "imec_data" / "A100_inf.csv"
    cases: list[base.CaseSpec] = []
    for idx, row in enumerate(_read_csv_rows(source_csv, encoding="utf-8-sig")):
        if row.get("device") != "A100":
            continue
        model = row["model"]
        tp = int(row["TP"])
        spec, hw_path, model_path = nvidia_inf._build_spec(  # type: ignore[attr-defined]
            "A100",
            model,
            tp,
            idx,
            network_ignored=False,
            fit_model=False,
        )
        model_cfg = base._merge_configs(
            base._load_yaml(Path(model_path)),
            spec.model_overrides or {},
            merge_lists_by_id=True,
        )
        hw_cfg = base._merge_configs(
            base._load_yaml(Path(hw_path)),
            spec.hardware_overrides or {},
            merge_lists_by_id=True,
        )
        case_id = _slug(f"inf_a100_{model}_tp{tp}")
        cases.append(
            base.CaseSpec(
                case_id=case_id,
                regime="A100 inference validation neighbor",
                source_csv=source_csv,
                model=model,
                training_or_inference="inference",
                gpu_type="A100 SXM4 80GB",
                device_type="A100_SXM4",
                num_gpus=tp,
                parallelism="DP=1, TP={}, PP=1, CP=1, TP_SP=True, batch=1, seq=400, decode=200".format(tp),
                model_config=model_cfg,
                hardware_config=hw_cfg,
            )
        )
    return cases


def _build_uci_training_neighbors() -> list[base.CaseSpec]:
    from validation_scripts import uci_train_validation as uci

    source_csv = SCRIPT_DIR / "train_validation_data" / "uci_train.csv"
    cases: list[base.CaseSpec] = []
    for row in _read_csv_rows(source_csv):
        if str(row.get("Status", "")).strip().upper() not in {"", "SUCCESS"}:
            continue
        variant = row["variant"]
        tp = int(row["TP"])
        pp = int(row["PP"])
        cp = int(row["CP"])
        dp = int(row["DP"])
        mb = uci.GLOBAL_BATCH_SIZE // (uci.MICRO_BATCH_SIZE * dp)
        eff_mb = mb if pp > 1 else 1
        network_override, mapping_desc = uci._build_network_override(  # type: ignore[attr-defined]
            tp,
            pp,
            cp,
            dp,
            "full_astrasim_hierarchical",
        )
        model_overrides = {
            "model_param": {
                "seq_len": 4096,
                "run_type": "training",
            }
        }
        hw_overrides: dict[str, Any] = {
            "parallelism": {
                "tp": tp,
                "tp_sp": False,
                "cp": cp,
                "pp": pp,
                "mb": eff_mb,
                "train": {"dp": dp, "ep": 1, "tp_ep": True},
                "inference": {"replica_count": 1, "moe_dp": 1},
            },
            "sw_param": {
                "dp_zero_stage": 0 if variant.upper() == "DDP" else 3,
            },
        }
        hw_overrides.update(network_override)
        model_cfg = base._merge_configs(
            base._load_yaml(uci.DEFAULT_MODEL_CONFIG),
            model_overrides,
            merge_lists_by_id=False,
        )
        hw_cfg = base._merge_configs(
            base._load_yaml(uci.DEFAULT_HW_CONFIG),
            hw_overrides,
            merge_lists_by_id=False,
        )
        case_id = _slug(f"uci_train_{variant}_tp{tp}_pp{pp}_cp{cp}_dp{dp}")
        cases.append(
            base.CaseSpec(
                case_id=case_id,
                regime="UCI PCIe training validation neighbor",
                source_csv=source_csv,
                model="Llama 2-7B",
                training_or_inference="training",
                gpu_type="A100 PCIe 80GB",
                device_type="A100_PCIe",
                num_gpus=tp * pp * cp * dp,
                parallelism=(
                    f"{variant}; DP={dp}, TP={tp}, PP={pp}, CP={cp}, "
                    f"TP_SP=False, mb={eff_mb}; {mapping_desc}"
                ),
                model_config=model_cfg,
                hardware_config=hw_cfg,
            )
        )
    return cases


def _build_neighbor_cases() -> list[base.CaseSpec]:
    cases = (
        _build_nvidia_training_neighbors()
        + _build_inference_neighbors()
        + _build_uci_training_neighbors()
    )
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        duplicates = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
        raise ValueError(f"Duplicate case IDs: {duplicates}")
    return cases


def _range_by_factor(rows: list[dict[str, str]]) -> dict[str, tuple[float, float] | None]:
    return base._range_by_factor(rows)


def _range_text(value: tuple[float, float] | None) -> str:
    return base._range_text(value)


def _find_row(rows: list[dict[str, str]], case_id: str, variant: str) -> dict[str, str] | None:
    return base._find_row(rows, case_id, variant)


def _fmt_runtime(value: str) -> str:
    return base._fmt_runtime(value)


def _fmt_change(value: str) -> str:
    return base._fmt_change(value)


def _verify_neighbors(rows: list[dict[str, str]], planned: list[base.PlannedRun]) -> dict[str, bool]:
    case_ids = sorted({run.case.case_id for run in planned})
    expected_rows = len(case_ids) * len(base.VARIANTS)
    per_case_ok = all(sum(row["case_id"] == case_id for row in rows) == len(base.VARIANTS) for case_id in case_ids)
    baseline_ok = all(
        sum(row["case_id"] == case_id and row["variant"] == "baseline" for row in rows) == 1
        for case_id in case_ids
    )
    no_fitted = all(
        ".fitted.yaml" not in row["hardware_config"] and ".fitted.yaml" not in row["command"]
        for row in rows
    )
    launch_ok = True
    allowed_only = True
    for run in planned:
        hw = base._load_yaml(run.hardware_path)
        launch_ok = launch_ok and float(hw["sw_param"]["kernel_launch_overhead"]) == 6e-6
    for case_id in case_ids:
        baseline_run = next(run for run in planned if run.case.case_id == case_id and run.variant == "baseline")
        baseline_hw = base._load_yaml(baseline_run.hardware_path)
        baseline_model = base._load_yaml(baseline_run.model_path)
        for run in [run for run in planned if run.case.case_id == case_id]:
            if base._load_yaml(run.model_path) != baseline_model:
                allowed_only = False
            hw = base._load_yaml(run.hardware_path)
            stripped = _strip_allowed_factor_fields(hw)
            baseline_stripped = _strip_allowed_factor_fields(baseline_hw)
            if stripped != baseline_stripped:
                allowed_only = False
    return {
        "csv_has_expected_lines_including_header": len(rows) + 1 == expected_rows + 1,
        "each_case_has_7_rows": per_case_ok,
        "each_case_has_one_baseline": baseline_ok,
        "no_fitted_yaml_referenced": no_fitted,
        "launch_overhead_fixed_6e-6": launch_ok,
        "only_allowed_factor_fields_change_within_case": allowed_only,
    }


def _strip_allowed_factor_fields(config: Mapping[str, Any]) -> dict[str, Any]:
    data = yaml.safe_load(yaml.safe_dump(dict(config), sort_keys=True))
    data.get("sw_param", {}).pop("kernel_launch_overhead", None)
    data.get("tech_param", {}).get("core", {}).pop("util", None)
    data.get("tech_param", {}).get("DRAM", {}).pop("util", None)
    for dim in data.get("network", {}).get("dimensions", []):
        if isinstance(dim, dict):
            dim.get("topology", {}).pop("util", None)
    return data


def _write_summary(
    rows: list[dict[str, str]],
    planned: list[base.PlannedRun],
    derates: Mapping[str, Any],
) -> None:
    case_ids = sorted({run.case.case_id for run in planned}, key=lambda cid: next(run.order for run in planned if run.case.case_id == cid))
    ranges = _range_by_factor(rows)
    failures = [row for row in rows if row["status"] != "success"]
    source_counts: dict[str, int] = {}
    for case_id in case_ids:
        row = _find_row(rows, case_id, "baseline")
        if row is not None:
            source_counts[row["source_csv"]] = source_counts.get(row["source_csv"], 0) + 1
    sentence = (
        "As a broader neighbor-case sensitivity check, we applied the same fixed Table III "
        f"one-at-a-time +/-10% perturbation to {len(case_ids)} already-submitted validation "
        "configurations from the same source CSVs as the four anchor cases. Across successful "
        f"perturbations, predicted runtime changed by {_range_text(ranges['compute'])} for compute, "
        f"{_range_text(ranges['memory'])} for memory, and {_range_text(ranges['communication'])} "
        "for communication; no factors were recalibrated."
    )

    lines = [
        "# Neighbor-Case Table III Factor Sensitivity",
        "",
        "This is a model-output sensitivity characterization of RAPID-LLM using already-submitted validation configurations. It does not add new measured ground truth, new benchmarks, new workloads, or recalibration.",
        "",
        "## Selection Rule",
        "",
        "Neighbor cases are defined as all rows in the same validation CSVs that supplied the original four anchor cases:",
        "",
    ]
    for source, count in sorted(source_counts.items()):
        lines.append(f"- `{source}`: {count} cases")
    lines += [
        "",
        f"Total cases: `{len(case_ids)}`",
        f"Total planned simulator rows: `{len(planned)}` (`{len(case_ids)}` cases x `{len(base.VARIANTS)}` variants)",
        "",
        "## Frozen Table III Factors",
        "",
        f"- A100 SXM4: compute `{derates['A100_SXM4']['compute']:.2f}`, memory `{derates['A100_SXM4']['memory']:.2f}`, communication `{derates['A100_SXM4']['communication']:.2f}`",
        f"- A100 PCIe: compute `{derates['A100_PCIe']['compute']:.2f}`, memory `{derates['A100_PCIe']['memory']:.2f}`, communication `{derates['A100_PCIe']['communication']:.2f}`",
        f"- Launch overhead fixed at `{float(derates['launch_overhead_s']):.8g}` seconds",
        "- No `.fitted.yaml` hardware config was used.",
        "",
        "## Per-Case Sensitivity",
        "",
        "| case_id | source | model | mode | GPUs | parallelism | baseline runtime | compute -10% / +10% | memory -10% / +10% | communication -10% / +10% | max abs change |",
        "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for case_id in case_ids:
        baseline = _find_row(rows, case_id, "baseline")
        values = {
            variant: _find_row(rows, case_id, variant)
            for variant in (
                "compute_m10",
                "compute_p10",
                "memory_m10",
                "memory_p10",
                "communication_m10",
                "communication_p10",
            )
        }
        abs_values = [
            float(row["absolute_runtime_change_percent"])
            for row in rows
            if row["case_id"] == case_id and row["absolute_runtime_change_percent"]
        ]
        max_abs = f"{max(abs_values):.2f}%" if abs_values else "FAILED"
        source = Path(baseline["source_csv"]).name if baseline else ""
        lines.append(
            "| `{}` | `{}` | {} | {} | {} | {} | {} | {} / {} | {} / {} | {} / {} | {} |".format(
                case_id,
                source,
                baseline["model"] if baseline else "",
                baseline["training_or_inference"] if baseline else "",
                baseline["num_gpus"] if baseline else "",
                baseline["parallelism"] if baseline else "",
                _fmt_runtime(baseline["predicted_runtime_s"] if baseline else ""),
                _fmt_change(values["compute_m10"]["signed_runtime_change_percent"] if values["compute_m10"] else ""),
                _fmt_change(values["compute_p10"]["signed_runtime_change_percent"] if values["compute_p10"] else ""),
                _fmt_change(values["memory_m10"]["signed_runtime_change_percent"] if values["memory_m10"] else ""),
                _fmt_change(values["memory_p10"]["signed_runtime_change_percent"] if values["memory_p10"] else ""),
                _fmt_change(values["communication_m10"]["signed_runtime_change_percent"] if values["communication_m10"] else ""),
                _fmt_change(values["communication_p10"]["signed_runtime_change_percent"] if values["communication_p10"] else ""),
                max_abs,
            )
        )

    lines += [
        "",
        "## Observed Ranges",
        "",
        f"- Compute factor +/-10%: {_range_text(ranges['compute'])} absolute runtime change across successful perturbations.",
        f"- Memory factor +/-10%: {_range_text(ranges['memory'])} absolute runtime change across successful perturbations.",
        f"- Communication factor +/-10%: {_range_text(ranges['communication'])} absolute runtime change across successful perturbations.",
        "",
        "## Failed Runs",
        "",
    ]
    if failures:
        lines += ["| case_id | variant | error |", "|---|---|---|"]
        for row in failures:
            lines.append(f"| `{row['case_id']}` | `{row['variant']}` | {row['error']} |")
    else:
        lines.append("None.")

    lines += [
        "",
        "## Rebuttal-Ready Sentence",
        "",
        sentence,
        "",
        "## Output Files",
        "",
        f"- CSV: `{base._rel(RESULTS_CSV)}`",
        f"- Commands: `{base._rel(COMMANDS_TXT)}`",
        f"- Generated configs: `{base._rel(CASE_ROOT)}`",
        "",
        "## Verification",
        "",
    ]
    verification = _verify_neighbors(rows, planned)
    for name, ok in verification.items():
        lines.append(f"- `{name}`: {'PASS' if ok else 'FAIL'}")
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_plan(planned: list[base.PlannedRun]) -> None:
    case_count = len({run.case.case_id for run in planned})
    print(f"Planned cases: {case_count}")
    print(f"Planned runs: {len(planned)}")
    for run in planned:
        print(f"{run.order:03d}: {run.case.case_id} {run.variant}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4, help="Parallel RAPID-LLM runs.")
    parser.add_argument("--dry-run", action="store_true", help="Materialize configs and commands only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _set_output_paths()
    if not base.PYTHON.exists():
        raise FileNotFoundError(f"Expected repo Python at {base.PYTHON}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    mpl_dir = OUT_ROOT / ".matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    derates = base._load_derates()
    cases = _build_neighbor_cases()
    planned = _with_shell_commands(base._plan_runs(cases, derates))
    base._LAST_PLANNED_RUNS = planned
    _write_commands(planned)
    _refresh_existing_result_commands(planned)
    _print_plan(planned)
    if args.dry_run:
        print(f"Wrote commands: {base._rel(COMMANDS_TXT)}")
        return 0

    results = base._run_all(planned, args.workers)
    rows = base._rows_from_results(planned, results, derates)
    base._write_csv(rows)
    _write_summary(rows, planned, derates)
    print(f"Wrote CSV: {base._rel(RESULTS_CSV)}")
    print(f"Wrote summary: {base._rel(SUMMARY_MD)}")
    print(f"Wrote commands: {base._rel(COMMANDS_TXT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
