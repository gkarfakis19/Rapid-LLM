#!/usr/bin/env python3
"""llm-analysis baseline for the dense-train validation figure (reproducible).

Runs STOCK llm-analysis (github.com/cli99/llm-analysis @ d841e40, unmodified)
for the 7 GPT train-validation rows at the documented protocol
(train_validation_data/llm_analysis_configs/PROVENANCE.txt), then applies the
post-emission correction of the paper's appendix, using only quantities the
tool itself emits:

    t_corr = t_iter - 3 * n_acc * max(0, t_fwd_dp_allgather - t_fwd_compute)
    t_fwd_compute = t_attn + t_mlp + t_layernorm + t_tp_comm

(stock llm-analysis floors each layer's forward latency at a ZeRO-3-style
data-parallel weight all-gather regardless of the sharding setting; the factor
3 reflects its backward = 2x forward modeling). Prints stock and corrected
predictions plus MAPE/worst for both — the corrected values are the ROWS in
plot_train_comparison_llmanalysis.py.

Requires an llm-analysis checkout at LLMA (pip deps: fire), at stock d841e40;
the tool's code is never patched.
"""
import os
import sys
from pathlib import Path

LLMA = "/app/nanocad/projects/ispass_deepflow/deepflow_astra_dev/llm-analysis"
CFG = str(Path(__file__).resolve().parent / "train_validation_data" / "llm_analysis_configs")
sys.path.insert(0, LLMA)

import logging

logging.disable(logging.CRITICAL)
from llm_analysis.analysis import train  # noqa: E402

# label, model json, actual_s, dp, tp, pp, gas(=mb), recompute (4=full, 1=selective)
CASES = [
    ("GPT 1T korthi full",       "megatron-gpt-1t.json",   87.9,  1, 8, 64, 512, 4),
    ("GPT 1T korthi selective",  "megatron-gpt-1t.json",   69.1,  1, 8, 64, 512, 1),
    ("GPT 1T selene full",       "megatron-gpt-1t.json",  100.7,  6, 8, 64, 512, 4),
    ("GPT 310B selene full",     "megatron-gpt-310b.json", 34.1, 15, 8, 16, 144, 4),
    ("GPT 530B selene full",     "megatron-gpt-530b.json", 51.2,  9, 8, 35, 280, 4),
    ("GPT 175B korthi full",     "megatron-gpt-175b.json", 16.9,  1, 8,  8,  64, 4),
    ("GPT 175B korthi selective","megatron-gpt-175b.json", 12.9,  1, 8,  8,  64, 1),
]


def main() -> None:
    stock_errs, corr_errs = [], []
    for label, mj, actual, dp, tp, pp, gas, rec in CASES:
        s = train(
            model_name=os.path.join(CFG, mj),
            gpu_name="a100-sxm-80gb",
            dtype_name="w16a16e16",
            batch_size_per_gpu=1,
            gradient_accumulation_steps=gas,
            dp_size=dp, tp_size=tp, pp_size=pp,
            seq_len=2048,
            activation_recomputation=rec,
            flops_efficiency=0.5,
        )
        t_iter = s["latency_per_iter"]
        compute = (s["latency_fwd_attn"] + s["latency_fwd_mlp"]
                   + s["latency_fwd_layernorm"] + s["latency_fwd_tp_comm"])
        delta = max(0.0, s["latency_fwd_sharded_dp_comm"] - compute)
        t_corr = t_iter - 3 * gas * delta
        e_stock = (t_iter - actual) / actual * 100
        e_corr = (t_corr - actual) / actual * 100
        stock_errs.append(abs(e_stock))
        corr_errs.append(abs(e_corr))
        print(f"{label:28s} actual {actual:7.2f}  stock {t_iter:8.2f} ({e_stock:+6.1f}%)"
              f"  corrected {t_corr:8.2f} ({e_corr:+6.1f}%)")
    print(f"stock:     MAPE {sum(stock_errs)/len(stock_errs):.2f}%  worst {max(stock_errs):.1f}%")
    print(f"corrected: MAPE {sum(corr_errs)/len(corr_errs):.2f}%  worst {max(corr_errs):.1f}%")


if __name__ == "__main__":
    main()
