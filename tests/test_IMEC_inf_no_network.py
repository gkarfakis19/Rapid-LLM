import math
from pathlib import Path

import pandas as pd
import pytest

from validation_scripts import nvidia_inf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "validation_scripts/imec_data"
DEFAULT_THRESHOLD = 100.0
# Override per (device, model) if needed. Falls back to DEFAULT_THRESHOLD.
# Actual + 0.5% margin of error, rounded up to nearest 0.5%.
GROUP_THRESHOLDS = {
  ("A100", "Llama 2-7B"): 5.5,   # 4.62 + 0.5 = 5.12 -> 5.5
  ("A100", "Llama 2-13B"): 3.0,  # 2.44 + 0.5 = 2.94 -> 3.0
  ("A100", "Llama 2-70B"): 13.0, # 12.09 + 0.5 = 12.59 -> 13.0
}


def _load_params():
  params = []
  for device in ("A100", "H100"):
    csv_path = DATA_DIR / f"{device}_inf.csv"
    if not csv_path.exists():
      continue
    df = pd.read_csv(csv_path, comment="/")
    df["TP"] = df["TP"].astype(int)
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    df = df.dropna(subset=["device", "model", "TP", "actual"])
    for model, group in df.groupby("model"):
      tp_actuals = [(int(tp), float(act)) for tp, act in zip(group["TP"], group["actual"])]
      marks = ()
      if device == "H100":
        marks = (pytest.mark.xfail(reason="Pending model refinements", strict=False),)
      params.append(
        pytest.param(
          device,
          model,
          tp_actuals,
          marks=marks,
          id=f"{device}-{model}",
        )
      )
  return params


PARAMS = _load_params()


@pytest.mark.parametrize("device,model,tp_actuals", PARAMS)
def test_avg_abs_error_below_threshold_network_ignored(device, model, tp_actuals, record_validation):
  label = f"{device} | {model}"
  threshold = GROUP_THRESHOLDS.get((device, model), DEFAULT_THRESHOLD)
  result = nvidia_inf.run(
    enable_plot=False,
    network_ignored=True,
    device=device,
    models=[model],
    emit_logs=False,
  )
  rows = result["rows"]
  assert rows, f"{label} produced no validation rows"

  pct_errors = []
  for row in rows:
    pct_error = row["pct_error"]
    assert not math.isnan(pct_error), f"{label}-TP{row['tp']} produced no valid measurements"
    pct_errors.append(pct_error)
  val_label = f"{label} avg_abs_error"
  avg_error = sum(pct_errors) / len(pct_errors)
  record_validation(val_label, avg_error, expected_pct=threshold)
  assert avg_error <= threshold, f"{label} avg error {avg_error:.2f}% exceeds {threshold}%"
