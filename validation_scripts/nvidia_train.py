import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .validation_helpers import (
  ValidationSpec,
  run_validation_suite,
  parse_inference_time,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_PERF = os.path.join(PROJECT_ROOT, "run_perf.py")
VALIDATION_CONFIG_ROOT = os.path.join(PROJECT_ROOT, "validation_scripts", "validation_configs")
HARDWARE_CONFIG_PATH = os.path.join(VALIDATION_CONFIG_ROOT, "hardware-config")
MODEL_CONFIG_PATH = os.path.join(VALIDATION_CONFIG_ROOT, "model-config")
DATA_DIR = Path(__file__).parent / "imec_data"

HW_CONFIGS = {
  "A100": "a100_80GB.yaml",
}

MODEL_CONFIGS = {
  "Llama 2-7B": "Llama2-7B_inf.yaml",
}


def _load_data(csv_path: Path):
  from validation_scripts.nvidia_inf import _load_data as _load_data_inf  # reuse parsing behavior
  return _load_data_inf(csv_path)


@lru_cache(maxsize=None)
def _load_device_data(device: str):
  csv_path = DATA_DIR / f"{device}_train.csv"
  if not csv_path.exists():
    raise FileNotFoundError(f"Validation CSV not found for device {device}: {csv_path}")
  return _load_data(csv_path)


def _iter_tests(df):
  for _, row in df.iterrows():
    yield (row["device"], row["model"], row["batch"], row["dp"], row["tp"], row["pp"], row["cp"])


def _build_spec(device: str, model: str, batch: int, dp: int, tp: int, pp: int, cp: int, idx: int) -> Tuple[ValidationSpec, str, str]:
  label = f"{device} {model} bs={batch} dp={dp} tp={tp} pp={pp} cp={cp}"

  model_overrides = {
    "model_param": {
      "global_batch_size": int(batch),
    }
  }

  hw_overrides = {
    "parallelism": {
      "dp": int(dp),
      "tp": int(tp),
      "tp_sp": False,
      "cp": int(cp),
      "lp": int(pp),
      "mb": max(1, int(pp)),  # micro-batches align with pipeline by default
    }
  }

  spec = ValidationSpec(
    label=label,
    model_overrides=model_overrides,
    hardware_overrides=hw_overrides,
    model_config_path=os.path.join(MODEL_CONFIG_PATH, MODEL_CONFIGS.get(model)),
    hardware_config_path=os.path.join(HARDWARE_CONFIG_PATH, HW_CONFIGS.get(device)),
    metadata={
      "device": device,
      "model": model,
      "batch": int(batch),
      "dp": int(dp),
      "tp": int(tp),
      "pp": int(pp),
      "cp": int(cp),
    },
    order=idx,
  )
  return spec, spec.model_config_path, spec.hardware_config_path  # type: ignore[return-value]


def _lookup_actual(device: str, model: str, batch: int, dp: int, tp: int, pp: int, cp: int) -> float:
  data = _load_device_data(device)
  matches = data.loc[
    (data["device"] == device)
    & (data["model"] == model)
    & (data["batch"] == int(batch))
    & (data["dp"] == int(dp))
    & (data["tp"] == int(tp))
    & (data["pp"] == int(pp))
    & (data["cp"] == int(cp)),
    "actual",
  ]
  if matches.empty:
    raise ValueError(f"No reference training time for {device} {model} bs={batch} dp={dp} tp={tp} pp={pp} cp={cp}")
  return float(matches.values[0])


def compute_pct_errors(results, actual_lookup: Dict[Tuple[str, int, int, int, int], float]):
  rows: List[Dict[str, object]] = []
  for res in results:
    meta = res.spec.metadata or {}
    model = meta.get("model")
    batch = int(meta.get("batch")) if "batch" in meta else None
    dp = int(meta.get("dp")) if "dp" in meta else None
    tp = int(meta.get("tp")) if "tp" in meta else None
    pp = int(meta.get("pp")) if "pp" in meta else None
    cp = int(meta.get("cp")) if "cp" in meta else None
    key = (model, batch, dp, tp, pp, cp)
    inf_time = float(res.metrics.get("inference_time_s", float("nan"))) if res.success else float("nan")
    actual = actual_lookup.get(key, float("nan"))
    if math.isnan(inf_time) or actual == 0 or math.isnan(actual):
      pct_error = float("nan")
    else:
      pct_error = abs(inf_time - actual) / actual * 100.0
    rows.append(
      {
        "device": meta.get("device"),
        "model": model,
        "batch": batch,
        "dp": dp,
        "tp": tp,
        "pp": pp,
        "cp": cp,
        "inference_time_s": inf_time,
        "actual_inference_time_s": actual,
        "pct_error": pct_error,
        "success": res.success,
        "error": res.error,
      }
    )
  return rows


def build_specs_for_device(
  device: str,
  *,
  models: Optional[Iterable[str]] = None,
) -> Tuple[List[ValidationSpec], Dict[Tuple[str, int, int, int, int, int], float], str, str]:
  data = _load_device_data(device)
  model_filter = set(models) if models is not None else None
  specs: List[ValidationSpec] = []
  actual_lookup: Dict[Tuple[str, int, int, int, int, int], float] = {}
  base_model_path: Optional[str] = None
  hw_config_path: Optional[str] = None
  idx = 0
  for device_val, model, batch, dp, tp, pp, cp in _iter_tests(data):
    if model_filter and model not in model_filter:
      continue
    spec, model_path, hw_path = _build_spec(device_val, model, batch, dp, tp, pp, cp, idx)
    specs.append(spec)
    actual_lookup[(model, int(batch), int(dp), int(tp), int(pp), int(cp))] = _lookup_actual(
      device_val, model, batch, dp, tp, pp, cp
    )
    base_model_path = base_model_path or model_path
    hw_config_path = hw_config_path or hw_path
    idx += 1
  if not specs:
    raise ValueError(f"No validation specs generated for device={device} (models={models}).")
  return specs, actual_lookup, base_model_path, hw_config_path


def run(
  device: str = "A100",
  models: Optional[Sequence[str]] = None,
  emit_logs: bool = True,
):
  specs, actual_lookup, base_model_path, hw_config_path = build_specs_for_device(
    device, models=models
  )

  validation_results = run_validation_suite(
    specs,
    base_model_config_path=base_model_path,
    base_hardware_config_path=hw_config_path,
    result_parser=parse_inference_time,
    run_perf_path=RUN_PERF,
  )

  rows = compute_pct_errors(validation_results, actual_lookup)
  pct_errors = [r["pct_error"] for r in rows if not math.isnan(r["pct_error"])]

  if emit_logs:
    for row in rows:
      block = [
        f"\n=== Result (device={row['device']}, model={row['model']}, bs={row['batch']}, dp={row['dp']}, tp={row['tp']}, pp={row['pp']}, cp={row['cp']}) ==="
      ]
      if row["success"] and not math.isnan(row["pct_error"]):
        block.append(f"  DeepFlow time:  {float(row['inference_time_s']):.2f}s")
        block.append(f"  Actual time:    {float(row['actual_inference_time_s']):.2f}s")
        block.append(f"  Percent Error:  {float(row['pct_error']):.2f}%")
      else:
        block.append(f"  DeepFlow run failed. {(row.get('error') or '')}".rstrip())
      print("\n".join(block))

  avg_abs_error = sum(pct_errors) / len(pct_errors) if pct_errors else float("nan")
  if emit_logs:
    print("Average absolute percent error across all training tests: {:.2f}%".format(avg_abs_error))
  return {"avg_abs_error": avg_abs_error, "rows": rows}


if __name__ == "__main__":
  if __package__ is None:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
  run(device="A100", emit_logs=True)
