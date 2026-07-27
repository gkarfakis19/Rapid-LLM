"""Generate GenZ baseline latencies for the H100 inference-validation cases.

Mirrors the methodology of GenZ-LLM-Analyzer/main.py (which produced the A100
CSVs in this directory):
  * System built from Systems/system_configs.py entry for the device
  * interchip link 25 GB/s, 1 us latency, FullyConnected topology, GenZ collectives
  * everything else at tool defaults
  * e2e = TTFT + TPOT * output_tokens

System efficiency follows the GenZ paper's empirically-measured per-hardware
factors (Bambhaniya et al., "Demystifying Platform Requirements for Diverse LLM
Inference Use Cases", arXiv:2406.01698): "The average empirically-measured
efficiency factors used for different hardware configurations are V100: 0.45,
A100: 0.4, 1xH100: 0.55, 2xH100: 0.64, 4xH100: 0.66, and 8xH100: 0.75."

The 0.4 figure is the A100 value only. Using it for H100 would understate GenZ's
efficiency and inflate its predicted runtimes, so the H100 runs select the factor
by GPU count. The A100 CSVs in this directory stay at 0.4 (verified: they
reproduce bit-identically at 0.4 with these settings).

Usage:
    PYTHONPATH=<genz_checkout> .venv/bin/python validation_scripts/run_genz_h100.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

GENZ_ROOT = os.environ.get(
    "GENZ_ROOT", "/app/nanocad/projects/personal/yaoe888/GenZ-LLM-Analyzer"
)
if GENZ_ROOT not in sys.path:
    sys.path.insert(0, GENZ_ROOT)

# GenZ imports IPython purely for notebook plotting helpers; stub it if absent.
try:  # pragma: no cover
    import IPython  # noqa: F401
except ImportError:  # pragma: no cover
    import types

    _ip = types.ModuleType("IPython")
    _disp = types.ModuleType("IPython.display")
    _disp.display = lambda *a, **k: None
    _ip.display = _disp
    sys.modules["IPython"] = _ip
    sys.modules["IPython.display"] = _disp

from GenZ import decode_moddeling, prefill_moddeling  # noqa: E402
from GenZ.system import System  # noqa: E402
from Systems.system_configs import system_configs  # noqa: E402

INTERCONNECT_BW_GBPS = 25.0
INTERCONNECT_LATENCY_US = 1.0
INTERCONNECT_TOPOLOGY = "FullyConnected"
COLLECTIVE_STRATEGY = "GenZ"
# GenZ paper's empirically-measured efficiency factors, keyed by H100 GPU count.
H100_SYSTEM_EFF = {1: 0.55, 2: 0.64, 4: 0.66, 8: 0.75}
# GenZ's precision table has no 'fp16' key; 'bf16' carries the identical
# 2-byte / 1x-compute multipliers, so it is the faithful stand-in for the
# fp16 H100 measurements.
BITS = "bf16"

DEVICE = "H100_GPU"
SCRIPT_DIR = Path(__file__).resolve().parent
IMEC_DATA_DIR = SCRIPT_DIR / "imec_data"
NVIDIA_DATA_DIR = SCRIPT_DIR / "nvidia_data"

IMEC_MODEL_MAP = {
    "Llama 2-7B": "llama2_7b",
    "Llama 2-13B": "llama2_13b",
    "Llama 2-70B": "llama2_70b",
}
NVIDIA_MODEL_LABEL = "Llama 3.3-70B"
NVIDIA_MODEL_GENZ = "llama3_70b"
NVIDIA_TP = 4

COLUMNS = [
    "suite",
    "device",
    "model",
    "genz_model",
    "nodes",
    "tp",
    "pp",
    "ep",
    "batch_size",
    "input_tokens",
    "output_tokens",
    "ttft_s",
    "tpot_s",
    "e2e_s",
    "error",
    # Auditability: the GenZ efficiency factor actually applied to this row.
    "system_eff",
]


def _system_eff_for(num_gpus: int) -> float:
    if num_gpus not in H100_SYSTEM_EFF:
        raise ValueError(
            f"No GenZ paper efficiency factor published for {num_gpus}xH100; "
            f"known counts: {sorted(H100_SYSTEM_EFF)}"
        )
    return H100_SYSTEM_EFF[num_gpus]


def _build_system(num_nodes: int, system_eff: float) -> System:
    base = system_configs[DEVICE]
    return System(
        flops=base["Flops"],
        off_chip_mem_size=base["Memory_size"] * 1024,
        offchip_mem_bw=base["Memory_BW"],
        bits=BITS,
        compute_efficiency=system_eff,
        memory_efficiency=system_eff,
        interchip_link_bw=INTERCONNECT_BW_GBPS,
        interchip_link_latency=INTERCONNECT_LATENCY_US,
        num_nodes=num_nodes,
        topology=INTERCONNECT_TOPOLOGY,
        collective_strategy=COLLECTIVE_STRATEGY,
    )


def _run(suite, model_label, genz_model, tp, batch_size, input_tokens, output_tokens):
    pp = ep = 1
    nodes = tp * pp * ep
    hierarchy = f"TP{{{tp}}}_EP{{{ep}}}_PP{{{pp}}}"
    system_eff = _system_eff_for(nodes)
    system = _build_system(nodes, system_eff)
    result = {
        "suite": suite,
        "device": DEVICE,
        "model": model_label,
        "genz_model": genz_model,
        "nodes": nodes,
        "tp": tp,
        "pp": pp,
        "ep": ep,
        "batch_size": batch_size,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "ttft_s": None,
        "tpot_s": None,
        "e2e_s": None,
        "error": None,
        "system_eff": system_eff,
    }
    try:
        prefill = prefill_moddeling(
            model=genz_model,
            batch_size=batch_size,
            input_tokens=input_tokens,
            system_name=system,
            system_eff=system_eff,
            bits=BITS,
            tensor_parallel=tp,
            pipeline_parallel=pp,
            expert_parallel=ep,
            parallelism_hierarchy=hierarchy,
            debug=False,
        )
        ttft_s = prefill["Latency"] / 1000.0
        result["ttft_s"] = ttft_s
    except Exception as exc:
        result["error"] = f"prefill_failed: {exc}"
        return result
    try:
        decode = decode_moddeling(
            model=genz_model,
            batch_size=batch_size,
            Bb=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            system_name=system,
            system_eff=system_eff,
            bits=BITS,
            tensor_parallel=tp,
            pipeline_parallel=pp,
            expert_parallel=ep,
            parallelism_hierarchy=hierarchy,
            debug=False,
        )
        tpot_s = decode["Latency"] / 1000.0
        result["tpot_s"] = tpot_s
        result["e2e_s"] = ttft_s + tpot_s * max(output_tokens, 1)
    except Exception as exc:
        result["error"] = f"decode_failed: {exc}"
    return result


def main() -> int:
    # ---- IMEC H100 Llama 2 TP sweep (batch 1, 200 in / 200 out) ----
    imec = pd.read_csv(IMEC_DATA_DIR / "H100_inf.csv", comment="/")
    imec = imec[imec["device"] == "H100"].copy()
    imec["TP"] = imec["TP"].astype(int)
    imec = imec[imec["TP"] <= 8]
    imec_rows = []
    for _, row in imec.iterrows():
        label = str(row["model"])
        genz_model = IMEC_MODEL_MAP.get(label)
        if not genz_model:
            print(f"Skipping IMEC model with no GenZ mapping: {label}")
            continue
        tp = int(row["TP"])
        print(f"IMEC {label} TP={tp}")
        imec_rows.append(_run("IMEC", label, genz_model, tp, 1, 200, 200))

    # ---- NIM H100 Llama 3.3-70B, TP4 ----
    nim = pd.read_csv(NVIDIA_DATA_DIR / "4xH100_fp16_Llama3_3-70B.csv")
    nim_rows = []
    for _, row in nim.iterrows():
        bs = int(row["Concurrency"])
        it = int(row["Input Tokens"])
        ot = int(row["Output Tokens"])
        print(f"NVIDIA {NVIDIA_MODEL_LABEL} TP={NVIDIA_TP} batch={bs} in={it} out={ot}")
        nim_rows.append(
            _run("NVIDIA", NVIDIA_MODEL_LABEL, NVIDIA_MODEL_GENZ, NVIDIA_TP, bs, it, ot)
        )

    imec_out = IMEC_DATA_DIR / "H100_inf_genz.csv"
    pd.DataFrame(imec_rows, columns=COLUMNS).to_csv(imec_out, index=False)
    print(f"Wrote {len(imec_rows)} IMEC rows to {imec_out}")

    nim_out = NVIDIA_DATA_DIR / "4xH100_fp16_Llama3_3-70B_genz.csv"
    pd.DataFrame(nim_rows, columns=COLUMNS).to_csv(nim_out, index=False)
    print(f"Wrote {len(nim_rows)} NVIDIA rows to {nim_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
