#!/usr/bin/env python3
"""
Parse MosaicML MPT benchmark tables in README.md into per-section validation CSVs.
"""

import argparse
import csv
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, List, NamedTuple, Sequence


SECTION_TO_FILE = {
    "H100 80GB BF16 (Large Scale, >= 128 GPUs)": "h100_80gb_bf16_large_scale.csv",
    "H100 80GB BF16": "h100_80gb_bf16.csv",
    "H100 80GB FP8": "h100_80gb_fp8.csv",
    "A100 80GB with 1600 Gbps node-node interconnect (RoCE)": "a100_80gb_roce.csv",
    "A100 40GB with 1600 Gbps node-node interconnect (RoCE)": "a100_40gb_roce.csv",
}

EXPECTED_ROW_COUNTS = {
    "h100_80gb_bf16_large_scale.csv": 18,
    "h100_80gb_bf16.csv": 52,
    "h100_80gb_fp8.csv": 7,
    "a100_80gb_roce.csv": 61,
    "a100_40gb_roce.csv": 78,
}

SOURCE_TO_OUTPUT = {
    "Model": "model_size",
    "SeqLen (T)": "seq_len",
    "# GPUs": "num_gpus",
    "GPU": "gpu",
    "MicroBatchSize": "micro_batch_size",
    "GradAccum": "gradient_accumulation_steps",
    "GlobalBatchSize": "global_batch_size",
    "GlobalBatchSize (T)": "global_batch_tokens",
    "Precision": "precision",
    "Sharding Strategy": "sharding_strategy",
    "Activation Checkpointing": "activation_checkpointing",
    "Throughput (T/s)": "throughput_tokens_per_s",
}

OUTPUT_COLUMNS = [
    "model_size",
    "seq_len",
    "num_gpus",
    "gpu",
    "micro_batch_size",
    "gradient_accumulation_steps",
    "global_batch_size",
    "global_batch_tokens",
    "precision",
    "sharding_strategy",
    "activation_checkpointing",
    "throughput_tokens_per_s",
    "inferred_total_latency_s",
]

NUMERIC_COLUMNS = {
    "seq_len",
    "num_gpus",
    "micro_batch_size",
    "gradient_accumulation_steps",
    "global_batch_size",
    "global_batch_tokens",
    "throughput_tokens_per_s",
}


class SectionTable(NamedTuple):
    section: str
    headers: List[str]
    rows: List[List[str]]


def _split_md_row(line: str) -> List[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_separator_row(cells: Sequence[str]) -> bool:
    if not cells:
        return False
    return all(re.fullmatch(r"[:\- ]+", cell or "") is not None for cell in cells)


def _parse_int(value: str, *, field: str, section: str, row_index: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid integer for '{field}' in section '{section}', row {row_index}: {value!r}"
        ) from exc


def _parse_bool_text(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return "True"
    if lowered in {"false", "0", "no", "n"}:
        return "False"
    return value.strip()


def _decimal_to_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def extract_target_tables(readme_text: str) -> Dict[str, SectionTable]:
    tables: Dict[str, SectionTable] = {}
    current_section = ""
    in_target_table = False
    headers: List[str] = []
    rows: List[List[str]] = []

    def flush() -> None:
        nonlocal in_target_table, headers, rows
        if in_target_table and current_section in SECTION_TO_FILE:
            tables[current_section] = SectionTable(
                section=current_section,
                headers=headers[:],
                rows=rows[:],
            )
        in_target_table = False
        headers = []
        rows = []

    for raw_line in readme_text.splitlines():
        line = raw_line.rstrip("\n")
        if line.startswith("## "):
            flush()
            current_section = line[3:].strip()
            continue

        if current_section not in SECTION_TO_FILE:
            continue

        stripped = line.strip()
        if stripped.startswith("|"):
            cells = _split_md_row(stripped)
            if not in_target_table:
                headers = cells
                rows = []
                in_target_table = True
                continue
            if _is_separator_row(cells):
                continue
            rows.append(cells)
        elif in_target_table and stripped == "":
            flush()

    flush()
    return tables


def transform_rows(table: SectionTable, warnings: List[str]) -> List[Dict[str, str]]:
    missing_headers = [src for src in SOURCE_TO_OUTPUT if src not in table.headers]
    if missing_headers:
        raise ValueError(
            f"Section '{table.section}' missing expected headers: {', '.join(missing_headers)}"
        )

    header_index = {h: i for i, h in enumerate(table.headers)}
    transformed: List[Dict[str, str]] = []

    for row_idx, row_cells in enumerate(table.rows, start=1):
        source_row: Dict[str, str] = {}
        for src in SOURCE_TO_OUTPUT:
            idx = header_index[src]
            source_row[src] = row_cells[idx].strip() if idx < len(row_cells) else ""

        out: Dict[str, str] = {}
        for src_col, out_col in SOURCE_TO_OUTPUT.items():
            raw = source_row[src_col]
            if out_col in NUMERIC_COLUMNS:
                out[out_col] = str(
                    _parse_int(raw, field=out_col, section=table.section, row_index=row_idx)
                )
            elif out_col == "activation_checkpointing":
                out[out_col] = _parse_bool_text(raw)
            else:
                out[out_col] = raw.strip()

        gbt = int(out["global_batch_tokens"])
        tps = int(out["throughput_tokens_per_s"])
        if tps <= 0:
            warnings.append(
                f"Section '{table.section}' row {row_idx} has non-positive throughput_tokens_per_s={tps};"
                " leaving inferred_total_latency_s blank."
            )
            out["inferred_total_latency_s"] = ""
        else:
            latency = Decimal(gbt) / Decimal(tps)
            out["inferred_total_latency_s"] = _decimal_to_text(latency)

        transformed.append(out)

    return transformed


def write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def verify_outputs(out_dir: Path) -> None:
    total_rows = 0
    for filename, expected_count in EXPECTED_ROW_COUNTS.items():
        path = out_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Expected output CSV not found: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if list(reader.fieldnames or []) != OUTPUT_COLUMNS:
                raise ValueError(f"Schema mismatch in {path}: {reader.fieldnames!r}")
            rows = list(reader)

        if len(rows) != expected_count:
            raise ValueError(
                f"Row count mismatch for {path.name}: expected {expected_count}, got {len(rows)}"
            )
        total_rows += len(rows)

        for row_idx, row in enumerate(rows, start=1):
            for col in NUMERIC_COLUMNS:
                try:
                    Decimal(row[col])
                except (InvalidOperation, KeyError) as exc:
                    raise ValueError(
                        f"Non-numeric value in {path.name} row {row_idx}, column '{col}': {row.get(col)!r}"
                    ) from exc

            latency_text = row.get("inferred_total_latency_s", "")
            tps = Decimal(row["throughput_tokens_per_s"])
            gbt = Decimal(row["global_batch_tokens"])
            if tps <= 0:
                if latency_text:
                    raise ValueError(
                        f"Expected blank latency for non-positive throughput in {path.name} row {row_idx}"
                    )
                continue
            expected_latency = gbt / tps
            try:
                latency = Decimal(latency_text)
            except InvalidOperation as exc:
                raise ValueError(
                    f"Invalid inferred_total_latency_s in {path.name} row {row_idx}: {latency_text!r}"
                ) from exc
            if abs(latency - expected_latency) > Decimal("1e-9"):
                raise ValueError(
                    f"Latency mismatch in {path.name} row {row_idx}: "
                    f"{latency_text} vs expected {expected_latency}"
                )

    if total_rows != 216:
        raise ValueError(f"Total row count mismatch: expected 216, got {total_rows}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path(__file__).resolve().parent / "README.md",
        help="Path to the source README markdown file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for output CSV files.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip post-write validation checks.",
    )
    args = parser.parse_args()

    readme_text = args.readme.read_text(encoding="utf-8")
    tables = extract_target_tables(readme_text)

    missing_sections = [section for section in SECTION_TO_FILE if section not in tables]
    if missing_sections:
        raise ValueError(f"Missing section tables in README: {missing_sections}")

    warnings: List[str] = []
    for section, filename in SECTION_TO_FILE.items():
        rows = transform_rows(tables[section], warnings=warnings)
        write_csv(args.out_dir / filename, rows)
        print(f"Wrote {filename}: {len(rows)} rows")

    if warnings:
        for warning in warnings:
            print(f"WARNING: {warning}", file=sys.stderr)

    if not args.skip_verify:
        verify_outputs(args.out_dir)
        print("Verification passed: schema, row counts, numeric fields, and latency checks.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
