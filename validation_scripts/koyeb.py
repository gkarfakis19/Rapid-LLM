#!/usr/bin/env python3
import os
import sys
import re
import tempfile
import subprocess
import yaml
import math
from typing import Dict, Tuple, List
import seaborn as sns

import matplotlib.pyplot as plt


# Llama-3.1-8B Instruct - A100 SXM (from Koyeb matrix, using actual Input/Output tokens)
# https://www.koyeb.com/docs/hardware/gpu-benchmarks
#
# Keys below are NORMALIZED to (prefill_len, decode_len, batch_size):
#   prefill_len = (input_tokens_agg / batch_size) rounded to nearest int
#   decode_len  = (decode_tokens_agg / batch_size) rounded to nearest int
# Values are tuples: (total_decode_time_seconds, expected_tokens_per_second_aggregate)
#
# Groups:
# - small  = "token shape 512x512"
# - medium = "token shape 1024x1024"
# - large  = "token shape 4096x1024"
# 
# We compare Deepflow's average token/s (calculated from decode time) with Koyeb's expected tokens/s.


# We need to convert the Koyeb data to the same format as the Deepflow data.
# Koyeb data is input_tokens_agg, decode_tokens_agg so we divided by batch size.
KOYEB_DATA: Dict[str, Dict[Tuple[int, int, int], Tuple[float, float]]] = {
  "small": {
    # From: (1484, 512, 1) -> (1484, 512, 1)
    (1484, 512, 1): (6.50, 78.77),
    # From: (11896, 4096, 8) -> (1487, 512, 8)
    (1487, 512, 8): (6.79, 603.29),
    # From: (48416, 16384, 32) -> (1513, 512, 32)
    (1513, 512, 32): (8.48, 1932.73),
  },
  "medium": {
    # From: (2948, 467, 1) -> (2948, 467, 1)
    (2948, 467, 1): (5.84, 79.99),
    # From: (23504, 3934, 8) -> (2938, 492, 8)
    (2938, 492, 8): (8.53, 461.46),
    # From: (94176, 23798, 32) -> (2943, 744, 32)
    (2943, 744, 32): (17.22, 1382.08),
  },
  "large": {
    # From: (11525, 449, 1) -> (11525, 449, 1)
    (11525, 449, 1): (5.97, 75.26),
    # From: (92608, 5923, 8) -> (11576, 740, 8)
    (11576, 740, 8): (13.67, 433.16),
    # From: (370592, 640, 32) -> (11581, 20, 32)
    (11581, 20, 32): (1.34, 477.82),
  },
}
sns.set()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_PERF = os.path.join(PROJECT_ROOT, "run_perf.py")
HARDWARE_CONFIG = os.path.join(PROJECT_ROOT, "configs/hardware-config/a100_80GB_no_parallelism.yaml")
BASE_MODEL_CONFIG = os.path.join(PROJECT_ROOT, "configs/model-config/Llama3.1-8B_inf.yaml")

def _load_yaml(path: str) -> dict:
  with open(path, "r") as f:
    return yaml.safe_load(f)

def _write_yaml(path: str, data: dict) -> None:
  with open(path, "w") as f:
    yaml.safe_dump(data, f, sort_keys=False)


def _update_model_config(base_cfg: dict, *, prefill_len: int, decode_len: int, batch_size: int) -> dict:
  cfg = dict(base_cfg)  # shallow copy is fine; we reassign primitives
  model_param = dict(cfg.get("model_param", {}))
  # seq_len = prefill + decode
  model_param["seq_len"] = int(prefill_len + decode_len)
  model_param["decode_len"] = int(decode_len)
  model_param["batch_size"] = int(batch_size)
  cfg["model_param"] = model_param
  return cfg


def _run_deepflow_with_temp_model(prefill_len: int, decode_len: int, batch_size: int) -> Tuple[str, float]:
  base_cfg = _load_yaml(BASE_MODEL_CONFIG)
  updated_cfg = _update_model_config(base_cfg, prefill_len=prefill_len, decode_len=decode_len, batch_size=batch_size)
  # if .tmp does not exist, make it
  if not os.path.exists("./tmp"):
    os.makedirs("./tmp")
  fd, tmp_path = tempfile.mkstemp(prefix="koyeb_model_", suffix=".yaml", dir="./tmp")
  os.close(fd)

  try:
    _write_yaml(tmp_path, updated_cfg)
    cmd = [
      sys.executable,
      RUN_PERF,
      "--hardware_config", HARDWARE_CONFIG,
      "--model_config", tmp_path,
    ]

    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    output = proc.stdout

    # Parse decode time from line: [prefill] time: Xs, [decode] time: Ys, [total] time: Zs
    decode_time = None
    m = re.search(r"\[prefill\]\s*time:\s*[0-9.]+s,\s*\[decode\]\s*time:\s*([0-9.]+)s,\s*\[total\]\s*time:\s*[0-9.]+s", output)
    if m:
      decode_time = float(m.group(1))

    if decode_time is None:
      raise RuntimeError("Failed to parse decode time from DeepFlow output")
    return output, decode_time
  finally:
    try:
      os.remove(tmp_path)
    except Exception:
      pass


def _collect_points_in_order(data: Dict[str, Dict[Tuple[int, int, int], Tuple[float, float]]]) -> List[Tuple[str, Tuple[int, int, int], Tuple[float, float]]]:
  ordered = []
  for size in ["small", "medium", "large"]:
    if size not in data:
      continue
    # sort by batch size ascending, then by prefill_len
    keys = sorted(list(data[size].keys()), key=lambda k: (k[2], k[0], k[1]))
    for k in keys:
      ordered.append((size, k, data[size][k]))
  return ordered


def run_all_and_plot() -> None:
  experiments = _collect_points_in_order(KOYEB_DATA)
  if not experiments:
    print("No experiments to run.")
    return

  labels: List[str] = []
  expected_tokps: List[float] = []
  predicted_tokps: List[float] = []

  for size, (prefill_len, decode_len, batch_size), (expected_decode_time_s, expected_tokps_agg) in experiments:
    label = f"{size} bs={batch_size}"
    print(f"\n=== Running {label} (prefill={prefill_len}, decode={decode_len}, batch={batch_size}) ===")
    try:
      console, decode_time_s = _run_deepflow_with_temp_model(prefill_len, decode_len, batch_size)
      # Aggregate tokens per second using overall definition: total tokens / decode_time
      total_tokens_agg = int(decode_len) * int(batch_size)  # dp=1 for no-parallelism hardware config
      predicted_tokps_agg = float(total_tokens_agg) / float(decode_time_s) if decode_time_s > 0 else 0.0
      print(f"Predicted agg tok/s = {predicted_tokps_agg:.2f} (decode_time={decode_time_s:.2f}s)")
    except Exception as e:
      print(f"ERROR running DeepFlow for {label}: {e}")
      predicted_tokps_agg = float("nan")

    labels.append(label)
    expected_tokps.append(expected_tokps_agg)
    predicted_tokps.append(predicted_tokps_agg)

  # Plot
  plt.figure(figsize=(12, 5))
  x = list(range(len(labels)))
  plt.plot(x, predicted_tokps, marker="o", linestyle="-", color="#1f77b4", label="DeepFlow predicted tok/s (aggregate)")
  plt.plot(x, expected_tokps, marker="o", linestyle="--", color="#ff7f0e", alpha=0.7, label="Expected tok/s (Koyeb)")
  plt.xticks(x, labels, rotation=30, ha="right")
  plt.ylabel("Tokens per second (aggregate)")
  plt.title("Koyeb vs DeepFlow (A100 SXM, no parallelism)")
  plt.grid(True, linestyle=":", alpha=0.4)
  plt.legend()

  out_dir = os.path.join(PROJECT_ROOT, "output", "validation")
  os.makedirs(out_dir, exist_ok=True)
  out_path = os.path.join(out_dir, "koyeb_a100_sxm_no_parallelism.png")
  plt.tight_layout()
  plt.savefig(out_path, dpi=160)
  print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
  run_all_and_plot()
