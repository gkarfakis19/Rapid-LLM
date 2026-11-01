"""Convert archive/ssm-gpu-old model YAMLs to the shared layer-plan schema.

This is an explicit migration, never a runtime alias. See docs/ssm/SSM.md.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


OLD_SSM_KEYS = {
    "state_size": "d_state", "conv_kernel": "d_conv",
    "num_heads": "n_heads", "headdim": "d_head", "head_dim": "d_head",
    "expand": "expand", "n_groups": "n_groups", "chunk_size": "chunk_size",
}


def convert_model(data: dict, *, chunk_size: int = 256) -> dict:
    """Preserve old dimensions and intended schedule, including interval=0.

    The archive did not price/use chunk_size. 256 is the explicit migration
    default (NEW prefill assumption). Conflicting duplicate values are refused.
    The archive treats every non-GPT model, including Mamba2, as SwiGLU.
    """
    result = copy.deepcopy(data)
    model = result["model_param"]
    if "layer_plan" in model:
        raise ValueError("Already has model_param.layer_plan; conversion requires an old model")
    old_type = model["model_type"].lower()
    pure_type = old_type in {"mamba", "mamba2", "mamba2_ssm", "ssm"}
    linear = model.pop("linear", {}) or {}
    if isinstance(linear, str):
        linear = {"linear_type": linear}
    if linear and linear.pop("linear_type", "mamba2") != "mamba2":
        raise ValueError("Only archive Mamba2 linear blocks can be converted")
    old_ssm = model.pop("ssm", {}) or {}
    ssm = {"variant": "mamba2"}
    for old, new in OLD_SSM_KEYS.items():
        values = []
        for block in (model, old_ssm, linear):
            if old in block:
                values.append(block.pop(old))
        if values:
            if len(set(values)) > 1 or (new in ssm and ssm[new] != values[0]):
                raise ValueError(f"Conflicting old SSM values for {old} -> ssm.{new}")
            ssm[new] = values[0]
    if old_ssm or linear:
        raise ValueError(f"Unknown old SSM keys: {sorted(set(old_ssm) | set(linear))}")
    ssm.setdefault("chunk_size", chunk_size)
    layers = int(model["num_layers"])
    schedule = model.pop("hybrid_schedule", None)
    kinds = ["ssm" if pure_type else "attention"] * layers
    if schedule is not None:
        if "linear_layers" in schedule or "full_attn_layers" in schedule:
            attn = set(schedule.get("full_attn_layers", []))
            recur = set(schedule.get("linear_layers", []))
            if attn & recur or attn | recur != set(range(1, layers + 1)):
                raise ValueError("hybrid_schedule must partition all 1-based layers")
            kinds = ["attention" if i in attn else "ssm" for i in range(1, layers + 1)]
        else:
            interval = int(schedule["full_attention_interval"])
            if interval < 0:
                raise ValueError("full_attention_interval must be >= 0")
            kinds = ["attention" if interval and i % interval == 0 else "ssm"
                     for i in range(1, layers + 1)]
    if "ssm" not in kinds:
        raise ValueError("Old model has no SSM layers to migrate")
    model["model_id"] = model.get("model_id", old_type)
    model["model_type"] = "llama"
    model["layer_plan"] = {"layer_types": kinds,
                           "ffn_per_layer": int(model["intermediate_size"]) > 0}
    model["ssm"] = ssm
    if "attention" not in kinds:
        model.pop("attention", None)
    # The shared schema requires a positive FFN width even for a mixer-only stack.
    if int(model["intermediate_size"]) == 0:
        model["intermediate_size"] = int(model["hidden_dim"]) * int(ssm["expand"])
    if model.get("run_type") != "inference":
        raise ValueError("SSM conversion requires model_param.run_type: inference")
    result.setdefault("inference_param", {"sample_every": -1})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()
    args.destination.write_text(yaml.safe_dump(convert_model(
        yaml.safe_load(args.source.read_text()), chunk_size=args.chunk_size), sort_keys=False))


if __name__ == "__main__":
    main()
