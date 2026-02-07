#!/usr/bin/env python3
"""Compare GEMMCalc-predicted memory traffic vs NCU bucketized metrics for aten::mm.

For each `*_summary.csv` in a run directory (e.g.
`Rapid-LLM/energy/imec/Llama2-7B/prompt_32_predict_4`), this script:
  - extracts all `aten::mm` rows from the summary (unique (cpu_op,input_dims) groups)
  - parses `input_dims` as `[[M, K], [K, N]]`
  - runs `GEMMCalc.run(M, K, N)` once per unique shape
  - scales each `AccessBytes` by the summary row count and sums totals
  - loads the corresponding `bucketized_ncu/<step>_bucketized_ncu.csv` and extracts
    the `aten::mm` bucket totals
  - writes a per-step comparison CSV and an SVG bar chart comparing:
      DRAM load/store (32B sectors), L2 load/store (32B sectors),
      and SHMEM load/store (bytes)

Defaults assume the A100 hardware config:
  `Rapid-LLM/configs/hardware-config/a100_80GB_no_parallelism.yaml`
"""

import argparse
import ast
import csv
import sys
import importlib.util
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

if sys.version_info < (3, 10):
    raise SystemExit(
        "Rapid-LLM currently requires Python 3.10+ (this interpreter is {}.{}.{}). "
        "Try: python3.11 Rapid-LLM/energy/mem_accesses.py ...".format(
            sys.version_info.major, sys.version_info.minor, sys.version_info.micro
        )
    )


RAPID_LLM_ROOT = Path(__file__).resolve().parents[1]
if str(RAPID_LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(RAPID_LLM_ROOT))

try:
    import config  # noqa: E402
except ModuleNotFoundError as e:  # pragma: no cover
    if str(e).strip().endswith("No module named 'yaml'") or getattr(e, "name", None) == "yaml":
        raise SystemExit(
            "Missing dependency: PyYAML. Install it for this Python interpreter, e.g.\n"
            "  python3.12 -m pip install PyYAML\n"
            "then rerun this script."
        )
    raise

from gemm_calc import GEMMCalc  # noqa: E402
from hw_component import Core, MemoryHierarchy  # noqa: E402
from tile import AccessBytes  # noqa: E402


Shape = Tuple[int, int, int]  # (M, K, N)

DEFAULT_RUN_DIR = (
    RAPID_LLM_ROOT / "energy/imec/Llama2-7B/prompt_32_predict_4"
)

NCU_DRAM_LOAD_COL = "DRAM Load Accesses"
NCU_DRAM_STORE_COL = "DRAM Store Accesses"
NCU_L2_LOAD_COL = "L2 Load Accesses"
NCU_L2_STORE_COL = "L2 Store Accesses"
NCU_SHMEM_LD_COL = "SHMEM LD Bytes"
NCU_SHMEM_LDSM_COL = "SHMEM LDSM Bytes"
NCU_SHMEM_ST_COL = "SHMEM ST Bytes"

DEEPCALC_LABEL = "DeepFlow"
GEMMCALC_LABEL = "GEMMCalc"
NCU_LABEL = "NCU"

DRAM_PJ_PER_SECTOR_32B = 2090.0
L2_PJ_PER_SECTOR_32B = 368.0
SHMEM_PJ_PER_32B = 82.1


def _parse_mm_input_dims(value: str) -> Optional[Shape]:
    """Parse input dims string for an aten::mm entry.

    Expected format in CSV: '[[M, K], [K, N]]'.
    Returns (M, K, N) or None if parsing/validation fails.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None

    try:
        dims = ast.literal_eval(s)
    except Exception:
        try:
            dims = ast.literal_eval("[" + s + "]")
        except Exception:
            return None

    if not isinstance(dims, (list, tuple)) or len(dims) < 2:
        return None
    a, b = dims[0], dims[1]
    if (
        not isinstance(a, (list, tuple))
        or not isinstance(b, (list, tuple))
        or len(a) != 2
        or len(b) != 2
    ):
        return None

    try:
        m = int(a[0])
        k1 = int(a[1])
        k2 = int(b[0])
        n = int(b[1])
    except Exception:
        return None

    if m <= 0 or k1 <= 0 or k2 <= 0 or n <= 0:
        return None
    if k1 != k2:
        return None
    return (m, k1, n)


def _iter_mm_shapes_and_counts(summary_csv: Path) -> Iterable[Tuple[Shape, int]]:
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return
        for row in reader:
            if (row.get("cpu_op_name") or "") != "aten::mm":
                continue
            shape = _parse_mm_input_dims(row.get("input_dims"))
            if shape is None:
                continue
            try:
                rows = int(float(row.get("rows") or 0))
            except Exception:
                rows = 0
            if rows <= 0:
                continue
            yield shape, rows


def _sum_levels(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
    if len(a) != len(b):
        raise ValueError("Mismatched number of memory levels: {} vs {}".format(len(a), len(b)))
    return tuple(x + y for x, y in zip(a, b))


def _accesses_for_shape(calc: GEMMCalc, shape: Shape) -> AccessBytes:
    m, k, n = shape
    best_gemm, _best_time = calc.run(m, k, n)
    return best_gemm.mem_accesses


def _parse_number(value) -> float:
    if value is None:
        return 0.0
    s = str(value).strip().strip('"')
    if not s or s.lower() in ("n/a", "na", "nan", "none"):
        return 0.0
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return 0.0


def _sectors_32b(byte_count: int) -> int:
    return (int(byte_count) + 31) // 32


def _scale_access(access: AccessBytes, factor: int) -> AccessBytes:
    f = int(factor)
    return AccessBytes(
        reads=tuple(int(x) * f for x in access.reads),
        writes=tuple(int(x) * f for x in access.writes),
    )


def _find_mem_level_indices(mem_layers) -> Dict[str, int]:
    idx: Dict[str, int] = {}
    for i, layer in enumerate(mem_layers):
        layer_type = getattr(layer, "type", None)
        name = layer.__class__.__name__
        if name == "DRAM" or layer_type == "DRAM":
            idx["dram"] = i
        elif layer_type == "SRAM-L2":
            idx["l2"] = i
        elif layer_type == "SRAM-L1":
            idx["l1"] = i
    return idx


def _read_bucketized_mm_metrics(bucketized_csv: Path) -> Dict[str, float]:
    with bucketized_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Empty bucketized NCU CSV: %s" % bucketized_csv)
        need = [
            "cpu_op_name",
            NCU_DRAM_LOAD_COL,
            NCU_DRAM_STORE_COL,
            NCU_L2_LOAD_COL,
            NCU_L2_STORE_COL,
            NCU_SHMEM_LD_COL,
            NCU_SHMEM_LDSM_COL,
            NCU_SHMEM_ST_COL,
        ]
        for col in need:
            if col not in reader.fieldnames:
                raise ValueError("Missing column %s in %s" % (col, bucketized_csv))
        for row in reader:
            if (row.get("cpu_op_name") or "").strip() != "aten::mm":
                continue
            return {
                "dram_load_sectors": _parse_number(row.get(NCU_DRAM_LOAD_COL)),
                "dram_store_sectors": _parse_number(row.get(NCU_DRAM_STORE_COL)),
                "l2_load_sectors": _parse_number(row.get(NCU_L2_LOAD_COL)),
                "l2_store_sectors": _parse_number(row.get(NCU_L2_STORE_COL)),
                "shmem_ld_bytes": _parse_number(row.get(NCU_SHMEM_LD_COL)),
                "shmem_ldsm_bytes": _parse_number(row.get(NCU_SHMEM_LDSM_COL)),
                "shmem_st_bytes": _parse_number(row.get(NCU_SHMEM_ST_COL)),
            }
    raise ValueError("aten::mm row not found in %s" % bucketized_csv)


def _fmt_sci(v: float) -> str:
    try:
        if v == 0:
            return "0"
        return "{:.3e}".format(float(v))
    except Exception:
        return str(v)


def _write_svg_mm_compare(out_path: Path, title: str, values: Dict[str, float]) -> None:
    """Write a simple 3-panel grouped bar chart as SVG (no external deps)."""
    # Panels: DRAM (sectors), L2 (sectors), SHMEM (bytes)
    methods = [
        (GEMMCALC_LABEL, "#4C78A8"),
        (DEEPCALC_LABEL, "#9ECAE9"),
        (NCU_LABEL, "#F58518"),
    ]
    panels = [
        {
            "name": "DRAM (32B sectors)",
            "cats": [
                (
                    "Load",
                    values["pred_dram_load"],
                    values["df_dram_load"],
                    values["ncu_dram_load"],
                ),
                (
                    "Store",
                    values["pred_dram_store"],
                    values["df_dram_store"],
                    values["ncu_dram_store"],
                ),
            ],
        },
        {
            "name": "L2 (32B sectors)",
            "cats": [
                ("Load", values["pred_l2_load"], values["df_l2_load"], values["ncu_l2_load"]),
                ("Store", values["pred_l2_store"], values["df_l2_store"], values["ncu_l2_store"]),
            ],
        },
        {
            "name": "SHMEM (bytes)",
            "cats": [
                (
                    "Load",
                    values["pred_shmem_load"],
                    values["df_shmem_load"],
                    values["ncu_shmem_load"],
                ),
                (
                    "Store",
                    values["pred_shmem_store"],
                    values["df_shmem_store"],
                    values["ncu_shmem_store"],
                ),
            ],
        },
    ]

    width = 1200
    height = 520
    margin_top = 50
    margin_bottom = 80
    panel_gap = 30
    panel_w = int((width - 2 * 40 - 2 * panel_gap) / 3)
    panel_h = height - margin_top - margin_bottom
    left0 = 40
    top0 = margin_top

    def esc(s: str) -> str:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    parts: List[str] = []
    parts.append(
        '<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'.format(
            w=width, h=height
        )
    )
    parts.append('<rect x="0" y="0" width="{w}" height="{h}" fill="white" />'.format(w=width, h=height))
    parts.append(
        '<text x="{x}" y="{y}" font-size="18" font-family="sans-serif" font-weight="600">{t}</text>'.format(
            x=left0, y=30, t=esc(title)
        )
    )

    legend_x = width - 330
    legend_y = 20
    cursor = 0
    for label, color in methods:
        parts.append(
            '<rect x="{x}" y="{y}" width="12" height="12" fill="{c}"/>'.format(
                x=legend_x + cursor, y=legend_y, c=color
            )
        )
        parts.append(
            '<text x="{x}" y="{y}" font-size="12" font-family="sans-serif">{t}</text>'.format(
                x=legend_x + cursor + 18, y=legend_y + 11, t=esc(label)
            )
        )
        cursor += 110

    for p_idx, panel in enumerate(panels):
        x0 = left0 + p_idx * (panel_w + panel_gap)
        y0 = top0
        # Panel frame
        parts.append(
            '<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#ffffff" stroke="#dddddd" />'.format(
                x=x0, y=y0, w=panel_w, h=panel_h
            )
        )
        parts.append(
            '<text x="{x}" y="{y}" font-size="14" font-family="sans-serif" font-weight="600">{t}</text>'.format(
                x=x0 + 8, y=y0 + 18, t=esc(panel["name"])
            )
        )
        # Plot area
        plot_left = x0 + 45
        plot_right = x0 + panel_w - 10
        plot_top = y0 + 30
        plot_bottom = y0 + panel_h - 35
        plot_w = plot_right - plot_left
        plot_h = plot_bottom - plot_top

        # Axes
        parts.append('<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="#333" />'.format(
            x=plot_left, y1=plot_top, y2=plot_bottom
        ))
        parts.append('<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#333" />'.format(
            x1=plot_left, x2=plot_right, y=plot_bottom
        ))

        cats = panel["cats"]
        max_v = 1.0
        for _c, a, b, c in cats:
            max_v = max(max_v, float(a), float(b), float(c))

        # Y ticks
        ticks = 4
        for i in range(ticks + 1):
            frac = i / ticks
            val = max_v * (1.0 - frac)
            y = plot_top + plot_h * frac
            parts.append('<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#eee" />'.format(
                x1=plot_left, x2=plot_right, y=y
            ))
            parts.append(
                '<text x="{x}" y="{y}" font-size="10" font-family="monospace" text-anchor="end" dominant-baseline="middle">{t}</text>'.format(
                    x=plot_left - 6, y=y, t=esc(_fmt_sci(val))
                )
            )

        # Bars
        group_w = plot_w / len(cats)
        bar_w = min(24.0, group_w * 0.18)
        offsets = [-bar_w - 6, 0, bar_w + 6]
        for c_idx, (c_name, v_pred, v_df, v_ncu) in enumerate(cats):
            gx = plot_left + group_w * c_idx + group_w * 0.5
            # label
            parts.append(
                '<text x="{x}" y="{y}" font-size="11" font-family="sans-serif" text-anchor="middle">{t}</text>'.format(
                    x=gx, y=plot_bottom + 20, t=esc(c_name)
                )
            )
            vals = [float(v_pred), float(v_df), float(v_ncu)]
            for idx_m, v in enumerate(vals):
                h = (v / max_v) * plot_h
                x_bar = gx + offsets[idx_m]
                y_bar = plot_bottom - h
                parts.append(
                    '<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{c}" />'.format(
                        x=x_bar,
                        y=y_bar,
                        w=bar_w,
                        h=h,
                        c=methods[idx_m][1],
                    )
                )

    parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")


def _energy_uJ_from_accesses(dram_sectors_32b: float, l2_sectors_32b: float, shmem_bytes: float) -> Dict[str, float]:
    dram_uJ = (float(dram_sectors_32b) * DRAM_PJ_PER_SECTOR_32B) / 1e6
    l2_uJ = (float(l2_sectors_32b) * L2_PJ_PER_SECTOR_32B) / 1e6
    shmem_sectors_32b = float(shmem_bytes) / 32.0
    shmem_uJ = (shmem_sectors_32b * SHMEM_PJ_PER_32B) / 1e6
    total_uJ = dram_uJ + l2_uJ + shmem_uJ
    return {
        "dram_uJ": dram_uJ,
        "l2_uJ": l2_uJ,
        "shmem_uJ": shmem_uJ,
        "total_uJ": total_uJ,
    }


def _write_svg_energy_breakdown(out_path: Path, title: str, energies_by_method: Dict[str, Dict[str, float]]) -> None:
    """Write a stacked energy breakdown chart (uJ) as SVG."""
    methods = [
        (GEMMCALC_LABEL, "#4C78A8"),
        (DEEPCALC_LABEL, "#9ECAE9"),
        (NCU_LABEL, "#F58518"),
    ]
    comps = [
        ("DRAM", "dram_uJ", "#E45756"),
        ("L2", "l2_uJ", "#72B7B2"),
        ("SHMEM", "shmem_uJ", "#54A24B"),
    ]

    width = 820
    height = 460
    margin = 50
    top = 55
    bottom = 70
    plot_left = margin
    plot_right = width - margin
    plot_top = top
    plot_bottom = height - bottom
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top

    def esc(s: str) -> str:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    totals = []
    for label, _color in methods:
        total = float(energies_by_method.get(label, {}).get("total_uJ", 0.0))
        totals.append(total)
    y_max = max(totals + [1.0])
    y_max *= 1.10

    parts: List[str] = []
    parts.append(
        '<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'.format(
            w=width, h=height
        )
    )
    parts.append('<rect x="0" y="0" width="{w}" height="{h}" fill="white" />'.format(w=width, h=height))
    parts.append(
        '<text x="{x}" y="30" font-size="18" font-family="sans-serif" font-weight="600">{t}</text>'.format(
            x=plot_left, t=esc(title)
        )
    )
    parts.append(
        '<text x="{x}" y="48" font-size="12" font-family="sans-serif" fill="#444">{t}</text>'.format(
            x=plot_left,
            t=esc(
                "Energy model: DRAM={:.0f} pJ/sector, L2={:.0f} pJ/sector, SHMEM={:.1f} pJ/32B".format(
                    DRAM_PJ_PER_SECTOR_32B, L2_PJ_PER_SECTOR_32B, SHMEM_PJ_PER_32B
                )
            ),
        )
    )

    # axes
    parts.append(
        '<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="#333" />'.format(
            x=plot_left, y1=plot_top, y2=plot_bottom
        )
    )
    parts.append(
        '<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#333" />'.format(
            x1=plot_left, x2=plot_right, y=plot_bottom
        )
    )

    # y ticks
    ticks = 4
    for i in range(ticks + 1):
        frac = i / ticks
        val = y_max * (1.0 - frac)
        y = plot_top + plot_h * frac
        parts.append(
            '<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#eee" />'.format(
                x1=plot_left, x2=plot_right, y=y
            )
        )
        parts.append(
            '<text x="{x}" y="{y}" font-size="10" font-family="monospace" text-anchor="end" dominant-baseline="middle">{t}</text>'.format(
                x=plot_left - 8, y=y, t=esc(_fmt_sci(val))
            )
        )

    # legend (components)
    lx = plot_right - 250
    ly = 18
    for idx_c, (name, _key, color) in enumerate(comps):
        parts.append(
            '<rect x="{x}" y="{y}" width="12" height="12" fill="{c}"/>'.format(
                x=lx + idx_c * 85, y=ly, c=color
            )
        )
        parts.append(
            '<text x="{x}" y="{y}" font-size="12" font-family="sans-serif">{t}</text>'.format(
                x=lx + idx_c * 85 + 18, y=ly + 11, t=esc(name)
            )
        )

    # bars
    bar_group_w = plot_w / len(methods)
    bar_w = min(80.0, bar_group_w * 0.5)
    for idx_m, (label, outline) in enumerate(methods):
        x_center = plot_left + bar_group_w * idx_m + bar_group_w * 0.5
        x0 = x_center - bar_w / 2.0
        y0 = plot_bottom
        method_energy = energies_by_method.get(label, {})

        # stacked components
        for _cname, key, color in comps:
            v = float(method_energy.get(key, 0.0))
            h = (v / y_max) * plot_h
            y0 = y0 - h
            parts.append(
                '<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{c}" stroke="none" />'.format(
                    x=x0, y=y0, w=bar_w, h=h, c=color
                )
            )

        # outline + label + total
        parts.append(
            '<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{c}" stroke-width="1.5" />'.format(
                x=x0, y=plot_top, w=bar_w, h=plot_bottom - plot_top, c=outline
            )
        )
        parts.append(
            '<text x="{x}" y="{y}" font-size="12" font-family="sans-serif" text-anchor="middle">{t}</text>'.format(
                x=x_center, y=plot_bottom + 22, t=esc(label)
            )
        )
        total = float(method_energy.get("total_uJ", 0.0))
        parts.append(
            '<text x="{x}" y="{y}" font-size="11" font-family="monospace" text-anchor="middle" fill="#333">{t} uJ</text>'.format(
                x=x_center, y=plot_top - 8, t=esc(_fmt_sci(total))
            )
        )

    parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")


def _load_deepflow_timecalc():
    repo_root = RAPID_LLM_ROOT.parent
    perf_path = repo_root / "DeepFlow" / "nanoCAD" / "perf.py"
    if not perf_path.exists():
        raise RuntimeError("DeepFlow perf.py not found at: %s" % perf_path)
    spec = importlib.util.spec_from_file_location("deepflow_nanoCAD_perf", str(perf_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load DeepFlow perf.py: %s" % perf_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    tc = getattr(mod, "TimeCalculation", None)
    if tc is None:
        raise RuntimeError("DeepFlow perf.py did not define TimeCalculation")
    return tc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=str,
        default=str(DEFAULT_RUN_DIR),
        help="Run directory containing *_summary.csv and bucketized_ncu/ (default: %s)." % str(DEFAULT_RUN_DIR),
    )
    parser.add_argument(
        "--bucketized-dir",
        type=str,
        default=None,
        help="Directory containing *_bucketized_ncu.csv (default: <run-dir>/bucketized_ncu).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to write comparison CSVs/SVGs (default: <run-dir>/mm_access_compare).",
    )
    parser.add_argument(
        "--hardware-config",
        type=str,
        default=str(RAPID_LLM_ROOT / "configs" / "hardware-config" / "a100_80GB_no_parallelism.yaml"),
        help="Hardware config YAML used to build Core + MemoryHierarchy for GEMMCalc.",
    )
    parser.add_argument(
        "--flashattn-enable",
        action="store_true",
        default=False,
        help="Enable FlashAttention-aware roofline (not typically used for aten::mm).",
    )
    parser.add_argument(
        "--no-shape-details",
        action="store_true",
        default=False,
        help="Disable writing per-unique-shape GEMMCalc access detail CSVs.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        print("Run dir not found or not a dir: %s" % run_dir, file=sys.stderr)
        return 2

    bucketized_dir = Path(args.bucketized_dir) if args.bucketized_dir else (run_dir / "bucketized_ncu")
    if not bucketized_dir.exists() or not bucketized_dir.is_dir():
        print("bucketized_ncu dir not found: %s" % bucketized_dir, file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "mm_access_compare")
    out_dir.mkdir(parents=True, exist_ok=True)

    hw_path = Path(args.hardware_config)
    if not hw_path.exists():
        print("Hardware config not found: %s" % hw_path, file=sys.stderr)
        return 2

    hw_config = config.parse_config(str(hw_path), config_type="hardware")
    core = Core(hw_config)
    memory_hierarchy = MemoryHierarchy(hw_config, core=core)
    mem_layers = memory_hierarchy.mem_layer
    num_levels = len(mem_layers)
    idx = _find_mem_level_indices(mem_layers)
    missing_idx = [k for k in ("dram", "l2", "l1") if k not in idx]
    if missing_idx:
        print("Could not locate memory levels in hardware config: %s" % ", ".join(missing_idx), file=sys.stderr)
        return 2

    dtype_bytes = int(getattr(hw_config.sw_config.precision, "activations", 0) or 0)
    if dtype_bytes <= 0:
        print(
            "Invalid dtype bytes from hw_config.sw_config.precision.activations: %s" % dtype_bytes,
            file=sys.stderr,
        )
        return 2

    gemm_calc = GEMMCalc(
        core=core,
        mem_hierarchy=mem_layers,
        dtype_bytes=dtype_bytes,
        flashattn_enable=bool(args.flashattn_enable),
    )

    # DeepFlow GEMM model (best-tile) for comparison of access counts.
    DeepFlowTimeCalc = _load_deepflow_timecalc()
    deepflow_tc = DeepFlowTimeCalc.gemm_only_from_rapid_llm(
        core=core,
        mem_layers=mem_layers,
        precision_bytes=dtype_bytes,
        kernel_launch_overhead_s=float(hw_config.sw_config.kernel_launch_overhead),
        debug=False,
    )

    summary_csvs = sorted(run_dir.glob("*_summary.csv"))
    if not summary_csvs:
        print("No *_summary.csv files found under: %s" % run_dir, file=sys.stderr)
        return 2

    shape_to_access: Dict[Shape, AccessBytes] = {}
    shape_to_df_access: Dict[Shape, Tuple[List[float], List[float]]] = {}
    failures = 0

    combined_energies = {
        GEMMCALC_LABEL: {"dram_uJ": 0.0, "l2_uJ": 0.0, "shmem_uJ": 0.0, "total_uJ": 0.0},
        DEEPCALC_LABEL: {"dram_uJ": 0.0, "l2_uJ": 0.0, "shmem_uJ": 0.0, "total_uJ": 0.0},
        NCU_LABEL: {"dram_uJ": 0.0, "l2_uJ": 0.0, "shmem_uJ": 0.0, "total_uJ": 0.0},
    }

    for summary_csv in summary_csvs:
        step = summary_csv.stem
        if step.endswith("_summary"):
            step = step[: -len("_summary")]

        bucketized_csv = bucketized_dir / ("%s_bucketized_ncu.csv" % step)
        if not bucketized_csv.exists():
            failures += 1
            print("Missing bucketized NCU CSV for %s: %s" % (step, bucketized_csv), file=sys.stderr)
            continue

        shapes_counts = list(_iter_mm_shapes_and_counts(summary_csv))
        if not shapes_counts:
            print("Step %s: no aten::mm rows in %s" % (step, summary_csv), file=sys.stderr)
            continue

        shape_to_count: Dict[Shape, int] = {}
        for shape, count in shapes_counts:
            shape_to_count[shape] = shape_to_count.get(shape, 0) + int(count)

        unique_shapes = sorted(shape_to_count.keys())
        for shape in unique_shapes:
            if shape not in shape_to_access:
                shape_to_access[shape] = _accesses_for_shape(gemm_calc, shape)

        if not args.no_shape_details:
            detail_csv = out_dir / ("%s_mm_shape_accesses_gemmcalc.csv" % step)
            with detail_csv.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", step])
                w.writerow(["summary_csv", str(summary_csv)])
                w.writerow(["hardware_config", str(hw_path)])
                w.writerow(["dtype_bytes", dtype_bytes])
                w.writerow([])
                w.writerow(
                    [
                        "M",
                        "K",
                        "N",
                        "rows",
                        "dram_load_sectors",
                        "dram_store_sectors",
                        "l2_load_sectors",
                        "l2_store_sectors",
                        "shmem_load_bytes",
                        "shmem_store_bytes",
                    ]
                )
                for m, k, n in unique_shapes:
                    count = int(shape_to_count[(m, k, n)])
                    access = shape_to_access[(m, k, n)]
                    # per GEMM
                    p_dram_r = float(_sectors_32b(int(access.reads[idx["dram"]])))
                    p_dram_w = float(_sectors_32b(int(access.writes[idx["dram"]])))
                    p_l2_r = float(_sectors_32b(int(access.reads[idx["l2"]])))
                    p_l2_w = float(_sectors_32b(int(access.writes[idx["l2"]])))
                    p_sh_r = float(int(access.reads[idx["l1"]]))
                    p_sh_w = float(int(access.writes[idx["l1"]]))
                    w.writerow(
                        [
                            m,
                            k,
                            n,
                            count,
                            int(p_dram_r),
                            int(p_dram_w),
                            int(p_l2_r),
                            int(p_l2_w),
                            int(p_sh_r),
                            int(p_sh_w),
                        ]
                    )
            print("Step %s: wrote %s" % (step, detail_csv), file=sys.stderr)

        reads: Tuple[int, ...] = tuple([0] * num_levels)
        writes: Tuple[int, ...] = tuple([0] * num_levels)
        total_rows = 0
        for shape, count in shapes_counts:
            total_rows += int(count)
            scaled = _scale_access(shape_to_access[shape], count)
            reads = _sum_levels(reads, scaled.reads)
            writes = _sum_levels(writes, scaled.writes)
        total_access = AccessBytes(reads=reads, writes=writes)

        # GEMMCalc predicted
        pred_dram_load = float(_sectors_32b(int(total_access.reads[idx["dram"]])))
        pred_dram_store = float(_sectors_32b(int(total_access.writes[idx["dram"]])))
        pred_l2_load = float(_sectors_32b(int(total_access.reads[idx["l2"]])))
        pred_l2_store = float(_sectors_32b(int(total_access.writes[idx["l2"]])))
        pred_shmem_load = float(int(total_access.reads[idx["l1"]]))
        pred_shmem_store = float(int(total_access.writes[idx["l1"]]))

        # DeepFlow (best-tile) predicted
        df_reads = [0.0] * num_levels
        df_writes = [0.0] * num_levels
        for shape, count in shapes_counts:
            if shape not in shape_to_df_access:
                m, k, n = shape
                _t, _order, _tile, _tot, reads, writes = deepflow_tc.getGEMMTime(
                    m, k, n, "aten::mm", return_accesses=True
                )
                # reads/writes are per-level byte counts for a single GEMM.
                shape_to_df_access[shape] = (list(reads), list(writes))
            r0, w0 = shape_to_df_access[shape]
            for li in range(num_levels):
                df_reads[li] += float(r0[li]) * float(count)
                df_writes[li] += float(w0[li]) * float(count)

        df_dram_load = float(_sectors_32b(int(df_reads[idx["dram"]])))
        df_dram_store = float(_sectors_32b(int(df_writes[idx["dram"]])))
        df_l2_load = float(_sectors_32b(int(df_reads[idx["l2"]])))
        df_l2_store = float(_sectors_32b(int(df_writes[idx["l2"]])))
        df_shmem_load = float(df_reads[idx["l1"]])
        df_shmem_store = float(df_writes[idx["l1"]])

        # NCU measured (bucketized)
        try:
            ncu = _read_bucketized_mm_metrics(bucketized_csv)
        except Exception as e:
            failures += 1
            print("Step %s: failed to read bucketized metrics (%s)" % (step, str(e)), file=sys.stderr)
            continue

        ncu_shmem_load = float(ncu["shmem_ld_bytes"] + ncu["shmem_ldsm_bytes"])
        ncu_shmem_store = float(ncu["shmem_st_bytes"])

        values = {
            "pred_dram_load": pred_dram_load,
            "pred_dram_store": pred_dram_store,
            "pred_l2_load": pred_l2_load,
            "pred_l2_store": pred_l2_store,
            "pred_shmem_load": pred_shmem_load,
            "pred_shmem_store": pred_shmem_store,
            "df_dram_load": df_dram_load,
            "df_dram_store": df_dram_store,
            "df_l2_load": df_l2_load,
            "df_l2_store": df_l2_store,
            "df_shmem_load": df_shmem_load,
            "df_shmem_store": df_shmem_store,
            "ncu_dram_load": float(ncu["dram_load_sectors"]),
            "ncu_dram_store": float(ncu["dram_store_sectors"]),
            "ncu_l2_load": float(ncu["l2_load_sectors"]),
            "ncu_l2_store": float(ncu["l2_store_sectors"]),
            "ncu_shmem_load": ncu_shmem_load,
            "ncu_shmem_store": ncu_shmem_store,
        }

        # Energy breakdown (uJ), stacked bar plot.
        energies = {
            GEMMCALC_LABEL: _energy_uJ_from_accesses(
                dram_sectors_32b=values["pred_dram_load"] + values["pred_dram_store"],
                l2_sectors_32b=values["pred_l2_load"] + values["pred_l2_store"],
                shmem_bytes=values["pred_shmem_load"] + values["pred_shmem_store"],
            ),
            DEEPCALC_LABEL: _energy_uJ_from_accesses(
                dram_sectors_32b=values["df_dram_load"] + values["df_dram_store"],
                l2_sectors_32b=values["df_l2_load"] + values["df_l2_store"],
                shmem_bytes=values["df_shmem_load"] + values["df_shmem_store"],
            ),
            NCU_LABEL: _energy_uJ_from_accesses(
                dram_sectors_32b=values["ncu_dram_load"] + values["ncu_dram_store"],
                l2_sectors_32b=values["ncu_l2_load"] + values["ncu_l2_store"],
                shmem_bytes=values["ncu_shmem_load"] + values["ncu_shmem_store"],
            ),
        }
        for method, d in energies.items():
            for k in ("dram_uJ", "l2_uJ", "shmem_uJ", "total_uJ"):
                combined_energies[method][k] += float(d.get(k, 0.0))

        out_energy_svg = out_dir / ("%s_mm_energy_breakdown.svg" % step)
        _write_svg_energy_breakdown(
            out_path=out_energy_svg,
            title="%s aten::mm energy breakdown (%s vs %s vs %s)" % (step, GEMMCALC_LABEL, DEEPCALC_LABEL, NCU_LABEL),
            energies_by_method=energies,
        )

        # Write per-step comparison CSV
        out_csv = out_dir / ("%s_mm_access_compare.csv" % step)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", step])
            w.writerow(["summary_csv", str(summary_csv)])
            w.writerow(["bucketized_ncu_csv", str(bucketized_csv)])
            w.writerow(["mm_unique_shapes", len(unique_shapes)])
            w.writerow(["mm_rows_total", total_rows])
            w.writerow([])
            w.writerow(["metric", GEMMCALC_LABEL, DEEPCALC_LABEL, NCU_LABEL])
            w.writerow(
                [
                    "dram_load_sectors_32B",
                    _fmt_sci(values["pred_dram_load"]),
                    _fmt_sci(values["df_dram_load"]),
                    _fmt_sci(values["ncu_dram_load"]),
                ]
            )
            w.writerow(
                [
                    "dram_store_sectors_32B",
                    _fmt_sci(values["pred_dram_store"]),
                    _fmt_sci(values["df_dram_store"]),
                    _fmt_sci(values["ncu_dram_store"]),
                ]
            )
            w.writerow(
                [
                    "l2_load_sectors_32B",
                    _fmt_sci(values["pred_l2_load"]),
                    _fmt_sci(values["df_l2_load"]),
                    _fmt_sci(values["ncu_l2_load"]),
                ]
            )
            w.writerow(
                [
                    "l2_store_sectors_32B",
                    _fmt_sci(values["pred_l2_store"]),
                    _fmt_sci(values["df_l2_store"]),
                    _fmt_sci(values["ncu_l2_store"]),
                ]
            )
            w.writerow(
                [
                    "shmem_load_bytes",
                    _fmt_sci(values["pred_shmem_load"]),
                    _fmt_sci(values["df_shmem_load"]),
                    _fmt_sci(values["ncu_shmem_load"]),
                ]
            )
            w.writerow(
                [
                    "shmem_store_bytes",
                    _fmt_sci(values["pred_shmem_store"]),
                    _fmt_sci(values["df_shmem_store"]),
                    _fmt_sci(values["ncu_shmem_store"]),
                ]
            )

        out_svg = out_dir / ("%s_mm_access_compare.svg" % step)
        _write_svg_mm_compare(
            out_path=out_svg,
            title="%s aten::mm memory access (%s vs %s vs %s)" % (step, GEMMCALC_LABEL, DEEPCALC_LABEL, NCU_LABEL),
            values=values,
        )
        print("Step %s: wrote %s, %s, %s" % (step, out_csv, out_svg, out_energy_svg), file=sys.stderr)

    combined_energy_svg = out_dir / "combined_mm_energy_breakdown.svg"
    _write_svg_energy_breakdown(
        out_path=combined_energy_svg,
        title="combined aten::mm energy breakdown (%s vs %s vs %s)" % (GEMMCALC_LABEL, DEEPCALC_LABEL, NCU_LABEL),
        energies_by_method=combined_energies,
    )
    print("Wrote %s" % combined_energy_svg, file=sys.stderr)

    if failures:
        print("Completed with %d failure(s). Output dir: %s" % (failures, out_dir), file=sys.stderr)
        return 1
    print("Done. Output dir: %s" % out_dir, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
