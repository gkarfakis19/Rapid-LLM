#!/usr/bin/env python3
# Copyright 2026 NanoCad lab, UCLA
# https://nanocad.ee.ucla.edu/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert a Hugging Face transformer config.json into a RAPID-LLM LLM YAML config."""

import argparse
import json
import re
import sys
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.parse
import urllib.request

import yaml

HF_CONFIG_MAX_BYTES = 2 * 1024 * 1024
HF_REPO_ID_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
HF_REVISION_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SUPPORTED_HF_MODEL_TYPES = {
    "deepseek_v3",
    "deepseekv3",
    "gpt2",
    "gpt_bigcode",
    "gpt_j",
    "gpt_neox",
    "gptj",
    "glm4",
    "glm4_moe",
    "llama",
    "mpt",
    "opt",
    "phi3",
    "qwen2",
}


def _validate_repo_id(model_id: str) -> str:
    text = str(model_id or "").strip().strip("/")
    parts = text.split("/")
    if len(parts) not in {1, 2} or any(not part for part in parts):
        raise SystemExit("Model id must look like 'org/model' or 'model'.")
    for part in parts:
        if not HF_REPO_ID_SEGMENT_RE.fullmatch(part) or "--" in part or ".." in part:
            raise SystemExit("Model id may contain only letters, numbers, '.', '_', and '-' in one or two path segments.")
    return "/".join(parts)


def _validate_revision(revision: str = "main") -> str:
    text = str(revision or "main").strip().strip("/")
    if not text:
        return "main"
    parts = text.split("/")
    if any(not part for part in parts) or any(part in {".", ".."} for part in parts):
        raise SystemExit("Revision may not contain empty, '.', or '..' path segments.")
    for part in parts:
        if not HF_REVISION_SEGMENT_RE.fullmatch(part) or "--" in part or ".." in part:
            raise SystemExit("Revision may contain only letters, numbers, '/', '.', '_', and '-'.")
    return "/".join(parts)


def _hf_config_url(model_id: str, revision: str = "main") -> str:
    safe_model = "/".join(urllib.parse.quote(part, safe="") for part in _validate_repo_id(model_id).split("/"))
    safe_revision = "/".join(urllib.parse.quote(part, safe="") for part in _validate_revision(revision).split("/"))
    return f"https://huggingface.co/{safe_model}/resolve/{safe_revision}/config.json"


def _fetch_hf_config(model_id: str, revision: str = "main") -> dict:
    url = _hf_config_url(model_id, revision)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            final_host = urllib.parse.urlparse(resp.geturl()).netloc.lower()
            if final_host not in {"huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co"}:
                raise SystemExit(f"Hugging Face redirected config.json to unsupported host '{final_host}'.")
            raw_payload = resp.read(HF_CONFIG_MAX_BYTES + 1)
            if len(raw_payload) > HF_CONFIG_MAX_BYTES:
                raise SystemExit("Hugging Face config.json is too large to import safely.")
            payload = raw_payload.decode("utf-8")
    except urllib.error.HTTPError as exc:  # pragma: no cover - runtime fetch
        raise SystemExit(f"Failed to fetch config.json for '{model_id}' ({exc.code} {exc.reason}).")
    except urllib.error.URLError as exc:  # pragma: no cover - runtime fetch
        raise SystemExit(f"Failed to reach Hugging Face: {exc.reason}.")

    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON returned for '{model_id}': {exc}.")


def _first(cfg: dict, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    return default


def _infer_model_type(model_type_field: Optional[str]) -> Tuple[str, Optional[str]]:
    alias = None
    if not model_type_field:
        raise SystemExit("config.json is missing model_type; only explicit supported Hugging Face model families can be imported.")
    lowered = model_type_field.lower()
    normalized_for_support = lowered.replace("-", "_")
    if normalized_for_support not in SUPPORTED_HF_MODEL_TYPES:
        supported = ", ".join(sorted(SUPPORTED_HF_MODEL_TYPES))
        raise SystemExit(f"Unsupported Hugging Face model_type '{model_type_field}'. Supported importer model types: {supported}.")
    normalized = lowered.replace("-", "").replace("_", "")
    if "glm" in normalized:
        return "glm4_moe", "glm"
    if "qwen2" in normalized:
        return "llama", "qwen2"
    if "phi3" in normalized:
        return "llama", "phi3"
    if "deepseek" in normalized:
        return "deepseek_v3", "deepseek"
    if "llama" in lowered:
        return "llama", alias
    if "gpt" in lowered or "opt" in lowered or "mpt" in lowered:
        return "gpt", alias
    raise SystemExit(f"Unsupported Hugging Face model_type '{model_type_field}'.")


def _language_config(cfg: dict) -> dict:
    lang_cfg = cfg.get("language_config", {})
    return lang_cfg if isinstance(lang_cfg, dict) else {}


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_truthy_flag(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _validate_supported_config_features(cfg: dict, alias: Optional[str]) -> None:
    lang_cfg = _language_config(cfg)
    if alias == "phi3":
        sliding_candidates = [cfg.get("sliding_window"), lang_cfg.get("sliding_window")]
        if any(_is_positive_int(value) for value in sliding_candidates):
            raise SystemExit(
                "Unsupported Phi-3 config: sliding-window attention is not modeled by RAPID-LLM, "
                "so this Hugging Face model cannot be imported safely."
            )
    if alias == "qwen2":
        sliding_flags = [cfg.get("use_sliding_window"), lang_cfg.get("use_sliding_window")]
        if any(_is_truthy_flag(value) for value in sliding_flags):
            raise SystemExit(
                "Unsupported Qwen2 config: active sliding-window attention is not modeled by RAPID-LLM, "
                "so this Hugging Face model cannot be imported safely."
            )


def _build_yaml_config(cfg: dict, args: argparse.Namespace, model_type: str) -> dict:
    hidden_dim = _first(
        cfg,
        "hidden_size",
        "d_model",
        "dim",
    )
    if hidden_dim is None:
        raise SystemExit("Unable to deduce hidden dimension from config.json (missing hidden_size / d_model / dim).")

    num_layers = _first(cfg, "num_hidden_layers", "n_layer", "num_layers")
    if num_layers is None:
        raise SystemExit("Unable to deduce number of layers from config.json.")

    num_heads = _first(cfg, "num_attention_heads", "n_head", "num_heads")
    if num_heads is None:
        raise SystemExit("Unable to deduce number of attention heads from config.json.")

    lang_cfg = cfg.get("language_config", {})

    head_dim = _first(
        cfg,
        "head_dim",
        "attention_head_dim",
        "head_size",
        "kv_channels",
        default=_first(
            lang_cfg,
            "head_dim",
            "attention_head_dim",
            "head_size",
            "kv_channels",
            default=None,
        ),
    )

    kv_heads = _first(
        cfg,
        "num_key_value_heads",
        "num_kv_heads",
        "n_kv_heads",
        default=_first(
            lang_cfg,
            "num_key_value_heads",
            "num_kv_heads",
            "n_kv_heads",
            default=None,
        ),
    )

    attention_type = "mha"
    attention_block: Dict[str, Any] = {
        "attention_type": attention_type,
        "num_heads": int(num_heads),
    }
    if model_type == "deepseek_v3":
        attention_block["attention_type"] = "mla"
        mla_field_names = (
            "kv_lora_rank",
            "q_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
        )
        for field_name in mla_field_names:
            raw_value = _first(cfg, field_name, default=_first(lang_cfg, field_name, default=None))
            if raw_value is None:
                raise SystemExit(
                    f"Unable to deduce {field_name} for DeepSeek-V3 MLA attention from config.json."
                )
            try:
                attention_block[field_name] = int(raw_value)
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    f"Invalid {field_name} value for DeepSeek-V3 MLA attention: {raw_value!r}."
                ) from exc
    elif kv_heads is not None and int(kv_heads) > 0 and int(kv_heads) != int(num_heads):
        attention_block["attention_type"] = "gqa"
        attention_block["kv_heads"] = int(kv_heads)
    else:
        attention_block["kv_heads"] = None

    if model_type == "glm4_moe":
        if head_dim is None:
            raise SystemExit(
                "Unable to deduce head_dim for model_type 'glm4_moe' from config.json."
            )
        try:
            attention_block["head_dim"] = int(head_dim)
        except (TypeError, ValueError):
            raise SystemExit(
                f"Invalid head_dim value for model_type 'glm4_moe': {head_dim!r}."
            )

    use_flashattention = args.use_flashattention
    if use_flashattention is None:
        use_flashattention = bool(
            _first(
                cfg,
                "flash_attention",
                "use_flash_attention",
                default=_first(lang_cfg, "flash_attention", "use_flash_attention", default=False),
            )
        )
    attention_block["use_flashattention"] = bool(use_flashattention)

    tile_size = args.flash_tile_size
    if tile_size is None:
        tile_size = _first(
            cfg,
            "flash_attention_block_size",
            "attention_tile_size",
            default=_first(
                lang_cfg,
                "flash_attention_block_size",
                "attention_tile_size",
                default=None,
            ),
        )
    if tile_size is not None:
        try:
            tile_size = int(tile_size)
        except (TypeError, ValueError):
            tile_size = None
    attention_block["attention_tile_size"] = tile_size

    intermediate_size = _first(
        cfg,
        "intermediate_size",
        "mlp_dim",
        default=_first(lang_cfg, "intermediate_size", "mlp_dim", default=None),
    )

    seq_len = args.seq_len
    if seq_len is None:
        seq_len = _first(
            cfg,
            "max_position_embeddings",
            "n_positions",
            "max_sequence_length",
            "sequence_length",
            default=2048,
        )


    tied_embeddings = bool(
        _first(cfg, "tie_word_embeddings", default=_first(lang_cfg, "tie_word_embeddings", default=True))
    )

    vocab_size = _first(cfg, "vocab_size", default=_first(lang_cfg, "vocab_size", default=None))
    if vocab_size is None:
        raise SystemExit("Unable to deduce vocab_size from config.json.")

    intermediate_size_value = int(intermediate_size) if intermediate_size is not None else 4 * int(hidden_dim)

    moe_intermediate_size = _first(
        cfg,
        "moe_intermediate_size",
        default=_first(lang_cfg, "moe_intermediate_size", default=intermediate_size_value),
    )
    try:
        moe_intermediate_size_value = int(moe_intermediate_size)
    except (TypeError, ValueError):
        moe_intermediate_size_value = intermediate_size_value

    num_experts = _first(
        cfg,
        "num_local_experts",
        "num_experts",
        "expert_count",
        "n_routed_experts",
        default=_first(
            lang_cfg,
            "num_local_experts",
            "num_experts",
            "expert_count",
            "n_routed_experts",
            default=1,
        ),
    )
    try:
        num_experts_value = max(1, int(num_experts))
    except (TypeError, ValueError):
        num_experts_value = 1

    top_k = _first(
        cfg,
        "num_experts_per_token",
        "num_experts_per_tok",
        "router_top_k",
        "gate_top_k",
        default=_first(
            lang_cfg,
            "num_experts_per_token",
            "num_experts_per_tok",
            "router_top_k",
            "gate_top_k",
            default=1,
        ),
    )
    try:
        top_k_value = max(1, int(top_k))
    except (TypeError, ValueError):
        top_k_value = 1
    top_k_value = min(top_k_value, num_experts_value)
    n_shared_experts = _first(
        cfg,
        "n_shared_experts",
        "num_shared_experts",
        default=_first(lang_cfg, "n_shared_experts", "num_shared_experts", default=0),
    )
    try:
        n_shared_experts_value = max(0, int(n_shared_experts))
    except (TypeError, ValueError):
        n_shared_experts_value = 0

    moe_layer_freq = _first(
        cfg,
        "moe_layer_freq",
        "moe_layer_frequency",
        default=_first(lang_cfg, "moe_layer_freq", "moe_layer_frequency", default=1),
    )
    try:
        moe_layer_freq = int(moe_layer_freq)
    except (TypeError, ValueError):
        moe_layer_freq = 1

    first_k_dense_replace = _first(
        cfg,
        "first_k_dense_replace",
        "first_k_dense_layers",
        default=_first(lang_cfg, "first_k_dense_replace", "first_k_dense_layers", default=0),
    )
    try:
        first_k_dense_replace = int(first_k_dense_replace)
    except (TypeError, ValueError):
        first_k_dense_replace = 0
    if num_experts_value <= 1:
        # Default to dense-only when MoE parameters are disabled.
        first_k_dense_replace = int(num_layers)

    yaml_dict = {
        "model_param": {
            "mode": "LLM",
            "run_type": args.run_type,
            "tied_embeddings": tied_embeddings,
            "model_type": model_type,
            "global_batch_size": args.global_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "seq_len": int(seq_len),
            "decode_len": args.decode_len,
            "hidden_dim": int(hidden_dim),
            "attention": attention_block,
            "intermediate_size": intermediate_size_value,
            "moe": {
                "num_experts": num_experts_value,
                "top_k": top_k_value,
                "moe_intermediate_size": moe_intermediate_size_value,
                "n_shared_experts": n_shared_experts_value,
                "moe_layer_freq": moe_layer_freq,
                "first_k_dense_replace": first_k_dense_replace,
            },
            "vocab_size": int(vocab_size),
            "num_layers": int(num_layers),
        },
    }

    return yaml_dict


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", help="Hugging Face model identifier, e.g. meta-llama/Llama-2-7b-hf")
    parser.add_argument("--revision", default="main", help="Model revision/branch to use (default: main)")
    parser.add_argument("--global-batch-size", type=int, default=1, help="Training global batch size to encode in the config (default: 1)")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation micro-steps per optimizer update (default: 1)",
    )
    parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length. Defaults to HF config max position embeddings if not provided")
    parser.add_argument("--decode-len", type=int, default=0, help="Decode sequence length (default: 0)")
    parser.add_argument("--run-type", default="training", help="Config run_type field ('training' or 'inference')")
    parser.add_argument("--use-flashattention", type=_str_to_bool_or_none, default=None, help="Override FlashAttention usage (true/false)")
    parser.add_argument("--flash-tile-size", type=int, default=None, help="Override FlashAttention tile size")
    parser.add_argument("--output", "-o", default=None, help="Path to write YAML (defaults to stdout)")
    return parser.parse_args(argv)


def _str_to_bool_or_none(value: str) -> Optional[bool]:
    lowered = value.strip().lower()
    if lowered in {"none", "null", "auto"}:
        return None
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value!r}")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cfg = _fetch_hf_config(args.model_id, revision=args.revision)
    orig_model_type = cfg.get("model_type")
    inferred_model_type, alias = _infer_model_type(orig_model_type)
    _validate_supported_config_features(cfg, alias)
    yaml_config = _build_yaml_config(cfg, args, inferred_model_type)

    yaml_dump = yaml.dump(
        yaml_config,
        default_flow_style=False,
        sort_keys=False,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(yaml_dump)
    else:
        sys.stdout.write(yaml_dump)

    if alias:
        print(
            f"[INFO] Mapping Hugging Face model_type '{orig_model_type}' to RAPID-LLM model_type '{inferred_model_type}'."
        )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
