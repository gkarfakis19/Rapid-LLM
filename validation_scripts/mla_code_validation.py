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

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import llm_util
from inference_timing import TimeCalculationLLMInference
from train_timing import GemmType, TimeCalculationLLM


MEGATRON_ROOT = PROJECT_ROOT / "tmp" / "reference_repos" / "Megatron-LM"
DEEPSPEED_ROOT = PROJECT_ROOT / "tmp" / "reference_repos" / "DeepSpeed"
HW_BASE = (
    PROJECT_ROOT
    / "validation_scripts"
    / "validation_configs"
    / "hardware-config"
    / "a100_80GB.yaml"
)
OUT_DIR = PROJECT_ROOT / "tmp" / "mla_code_validation"
OUT_JSON = OUT_DIR / "results.json"


@dataclass
class ToyCase:
    name: str
    run_type: str
    tp: int
    cp: int
    phase: str = "default"
    tp_sp: bool = False


@dataclass
class CheckResult:
    name: str
    status: str
    details: str
    rapid_value: Optional[float | str] = None
    reference_value: Optional[float | str] = None
    case: Optional[str] = None


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout.strip()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _find_required(text: str, pattern: str, desc: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise AssertionError(f"Could not find required Megatron/DeepSpeed code fact: {desc}")
    return match.group(0)


def _build_hw_config(*, tp: int = 1, cp: int = 1, tp_sp: bool = False) -> config.HWConfig:
    hw_config = config.parse_config(str(HW_BASE), config_type="hardware")
    hw_config.sch_config.tp = tp
    hw_config.sch_config.cp = cp
    hw_config.sch_config.pp = 1
    hw_config.sch_config.mb = 1
    hw_config.sch_config.tp_sp = tp_sp
    hw_config.sch_config.train.dp = 1
    hw_config.sch_config.train.ep = 1
    hw_config.sch_config.train.tp_ep = True
    return hw_config


def _build_mla_model(*, run_type: str = "training") -> config.ModelConfig:
    model_param = {
        "mode": "LLM",
        "run_type": run_type,
        "tied_embeddings": False,
        "model_type": "llama",
        "global_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "full_recomputation": False,
        "seq_len": 8,
        "decode_len": 4 if run_type == "inference" else 0,
        "hidden_dim": 64,
        "attention": {
            "attention_type": "mla",
            "num_heads": 4,
            "head_dim": 16,
            "use_flashattention": False,
            "attention_tile_size": 64,
            "kv_lora_rank": 4,
            "q_lora_rank": 8,
            "qk_nope_head_dim": 12,
            "qk_rope_head_dim": 4,
            "v_head_dim": 8,
        },
        "intermediate_size": 128,
        "vocab_size": 512,
        "num_layers": 2,
        "moe": {
            "num_experts": 1,
            "top_k": 1,
            "moe_intermediate_size": 128,
            "n_shared_experts": 0,
            "moe_layer_freq": 1,
            "first_k_dense_replace": 0,
        },
    }
    llm_config = config.LLMConfig.from_dict(model_param)
    inference_cfg = config.LLMInferenceConfig(sample_every=-1) if run_type == "inference" else None
    return config.ModelConfig(model_config=llm_config, inference_config=inference_cfg)


def _rapid_tc(case: ToyCase):
    hw_config = _build_hw_config(tp=case.tp, cp=case.cp, tp_sp=case.tp_sp)
    if case.run_type == "inference":
        hw_config.inference_config = hw_config.sch_config.inference
    model = _build_mla_model(run_type=case.run_type)
    config.validate_configs(hw_config, model)
    if case.run_type == "inference":
        return TimeCalculationLLMInference(hw_config, model, "LLM")
    return TimeCalculationLLM(hw_config, model, "LLM")


def _rapid_runtime_shapes(case: ToyCase) -> Dict[str, tuple]:
    tc = _rapid_tc(case)
    if case.run_type == "inference" and case.phase == "decode":
        return llm_util.process_decode_gemm_shapes(
            tc,
            batch_size=4,
            current_seq_len=8,
            d_model=64,
            num_heads=4,
            kv_heads=4,
            intermediate_size=128,
            vocab_size=512,
            model_type=tc.model_type,
        )
    return llm_util.process_gemm_shapes(
        tc,
        batch_size=4,
        seq_len=8,
        d_model=64,
        num_heads=4,
        kv_heads=4,
        intermediate_size=128,
        vocab_size=512,
    )


def _rapid_local_output_elements(case: ToyCase, gemm_key: str, gemm_type: GemmType) -> int:
    tc = _rapid_tc(case)
    shapes = _rapid_runtime_shapes(case)
    return int(tc._shard_gemm_descriptor(shapes[gemm_key], gemm_type).output_elements())


def _rapid_mla_cache_token_bytes_per_rank(case: ToyCase) -> float:
    precision = 2
    total = llm_util.mla_kv_cache_token_bytes(
        batch_size=4,
        kv_lora_rank=4,
        qk_rope_head_dim=4,
        precision_bytes=precision,
    )
    return float(total) / float(case.tp)


def _megatron_reference_shapes(case: ToyCase) -> Dict[str, int]:
    batch_size = 4
    if case.run_type == "inference" and case.phase == "decode":
        seq_local = 1
    else:
        seq_local = math.ceil(8 / case.cp)
    heads_local = 4 // case.tp
    return {
        "d_proj_q_local_elems": batch_size * seq_local * 8,
        "d_proj_kv_plus_rope_local_elems": batch_size * seq_local * (4 + 4),
        "q_up_local_elems": batch_size * seq_local * heads_local * (12 + 4),
        "kv_up_local_elems": batch_size * seq_local * heads_local * (12 + 8),
        "output_proj_input_local_elems": batch_size * seq_local * heads_local * 8,
    }


def _megatron_cache_bytes_per_rank(*, tp: int, latent_cache: bool) -> float:
    precision = 2
    batch_size = 4
    heads_local = 4 // tp
    if latent_cache:
        return float(batch_size * (4 + 4) * precision)
    return float(batch_size * heads_local * (12 + 4 + 8) * precision)


def _check_reference_code_facts() -> List[CheckResult]:
    checks: List[CheckResult] = []

    megatron_specs = _read_text(
        MEGATRON_ROOT / "megatron/core/models/gpt/gpt_layer_specs.py"
    )
    megatron_mla = _read_text(
        MEGATRON_ROOT / "megatron/core/transformer/multi_latent_attention.py"
    )
    megatron_cfg = _read_text(
        MEGATRON_ROOT / "megatron/core/transformer/transformer_config.py"
    )
    megatron_ctx = _read_text(
        MEGATRON_ROOT / "megatron/core/inference/contexts/dynamic_context.py"
    )
    deepspeed_autotp = _read_text(DEEPSPEED_ROOT / "deepspeed/module_inject/autotp_config.py")

    checks.append(
        CheckResult(
            name="Megatron default down-projection spec",
            status="PASS",
            details=_find_required(
                megatron_specs,
                r"linear_q_down_proj=backend\.linear\(\),\s*\n\s*linear_q_up_proj=.*\n\s*linear_kv_down_proj=backend\.linear\(\)",
                "Megatron default MLA uses backend.linear() for q/kv down projections",
            ),
        )
    )
    checks.append(
        CheckResult(
            name="Megatron default cache_mla_latents default",
            status="PASS",
            details=_find_required(
                megatron_cfg,
                r"cache_mla_latents:\s*bool\s*=\s*False",
                "Megatron MLA cache_mla_latents defaults to False",
            ),
        )
    )
    checks.append(
        CheckResult(
            name="Megatron latent cache shape",
            status="PASS",
            details=_find_required(
                megatron_ctx,
                r"self\.kv_reduced_dim\s*=\s*model_config\.kv_lora_rank\s*\+\s*model_config\.qk_pos_emb_head_dim",
                "Megatron latent cache stores kv_lora_rank + qk_pos_emb_head_dim",
            ),
        )
    )
    checks.append(
        CheckResult(
            name="Megatron default prefill/training path is unabsorbed",
            status="PASS",
            details=_find_required(
                megatron_mla,
                r"kv,\s*_ = self\.linear_kv_up_proj\(kv_compressed\)",
                "Megatron expands kv_compressed back to full per-head K/V before attention",
            ),
        )
    )
    checks.append(
        CheckResult(
            name="Megatron decode latent-cache path uses absorbed decode helper",
            status="PASS",
            details=_find_required(
                megatron_mla,
                r"use_absorption\s*=\s*\(\s*self\.config\.cache_mla_latents[\s\S]*?inference_context\.is_decode_only\(\)\s*\)",
                "Megatron absorption is gated by cache_mla_latents and decode-only inference",
            ),
        )
    )
    checks.append(
        CheckResult(
            name="DeepSpeed TP rules skip low-rank down projections",
            status="PASS",
            details=_find_required(
                deepspeed_autotp,
                r"patterns=\[r\"\.\*\\\.self_attn\\\.\(q_a_proj\|kv_a_proj_with_mqa\)\\\.weight\$\"\],\s*\n\s*partition_type=PartitionType\.SKIP",
                "DeepSpeed AutoTP does not shard DeepSeek low-rank down projections",
            ),
        )
    )
    checks.append(
        CheckResult(
            name="DeepSpeed TP rules shard latent up projections column-wise",
            status="PASS",
            details=_find_required(
                deepspeed_autotp,
                r"patterns=\[r\"\.\*\\\.self_attn\\\.\(q_b_proj\|kv_b_proj\)\\\.weight\$\"\],\s*\n\s*partition_type=PartitionType\.COLUMN",
                "DeepSpeed AutoTP column-shards latent up projections",
            ),
        )
    )
    checks.append(
        CheckResult(
            name="DeepSpeed TP rules shard output row-wise",
            status="PASS",
            details=_find_required(
                deepspeed_autotp,
                r"patterns=\[r\"\.\*\\\.self_attn\\\.o_proj\\\.weight\$\"\],\s*\n\s*partition_type=PartitionType\.ROW",
                "DeepSpeed AutoTP row-shards the output projection",
            ),
        )
    )
    return checks


def _compare_param_totals() -> CheckResult:
    rapid_total = llm_util.mla_attention_param_groups(
        hidden_dim=64,
        num_heads=4,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=8,
    )["total"]
    megatron_total = (
        64 * 8
        + 64 * (4 + 4)
        + 8 * 4 * (12 + 4)
        + 4 * 4 * (12 + 8)
        + (4 * 8 * 64)
    )
    status = "PASS" if rapid_total == megatron_total else "MISMATCH"
    return CheckResult(
        name="Parameter total",
        status=status,
        details="RAPID parameter total vs Megatron-equivalent algebraic total",
        rapid_value=rapid_total,
        reference_value=megatron_total,
    )


def _compare_case(case: ToyCase) -> List[CheckResult]:
    results: List[CheckResult] = []
    ref = _megatron_reference_shapes(case)

    rapid_q_down = _rapid_local_output_elements(case, "D_proj_q", GemmType.QKV)
    rapid_kv_down = (
        _rapid_local_output_elements(case, "D_proj_kv", GemmType.QKV)
        + _rapid_local_output_elements(case, "K_rope_proj", GemmType.QKV)
    )
    status_q = "PASS" if rapid_q_down == ref["d_proj_q_local_elems"] else "MISMATCH"
    status_kv = (
        "PASS" if rapid_kv_down == ref["d_proj_kv_plus_rope_local_elems"] else "MISMATCH"
    )
    results.append(
        CheckResult(
            name="Local q down-projection output elements",
            status=status_q,
            details="Megatron default keeps q_lora_rank unsharded across TP in q_down_proj",
            rapid_value=rapid_q_down,
            reference_value=ref["d_proj_q_local_elems"],
            case=case.name,
        )
    )
    results.append(
        CheckResult(
            name="Local kv down-projection output elements",
            status=status_kv,
            details="Megatron default keeps kv_lora_rank + rope unsharded across TP in kv_down_proj",
            rapid_value=rapid_kv_down,
            reference_value=ref["d_proj_kv_plus_rope_local_elems"],
            case=case.name,
        )
    )

    if case.run_type == "inference":
        rapid_cache = _rapid_mla_cache_token_bytes_per_rank(case)
        megatron_full = _megatron_cache_bytes_per_rank(tp=case.tp, latent_cache=False)
        megatron_latent = _megatron_cache_bytes_per_rank(tp=case.tp, latent_cache=True)
        cache_status = (
            "PASS"
            if math.isclose(rapid_cache, megatron_full) or math.isclose(rapid_cache, megatron_latent)
            else "MISMATCH"
        )
        results.append(
            CheckResult(
                name="Per-rank inference KV-cache bytes per token",
                status=cache_status,
                details="RAPID current MLA cache bytes compared against both Megatron full-cache default and Megatron latent-cache decode mode",
                rapid_value=rapid_cache,
                reference_value=f"default_full={megatron_full}, latent_cache={megatron_latent}",
                case=case.name,
            )
        )

    shapes = _rapid_runtime_shapes(case)
    if case.run_type == "training":
        results.append(
            CheckResult(
                name="Training/prefill attention core contract",
                status="MISMATCH",
                details=(
                    "RAPID uses absorbed latent attention_score_1/2 + latent context + absorbed output, "
                    "while Megatron default training path materializes full per-head K/V via linear_kv_up_proj "
                    "before core attention."
                ),
                rapid_value=str(
                    {
                        "attention_score_1": shapes["attention_score_1"],
                        "attention_score_2": shapes["attention_score_2"],
                        "attention_ctx_latent": shapes["attention_ctx_latent"],
                        "output_proj": shapes["output_proj"],
                    }
                ),
                reference_value="Megatron default path: q_up -> kv_up -> core_attention(query, key, value) -> row-parallel output",
                case=case.name,
            )
        )
    elif case.phase == "prefill":
        results.append(
            CheckResult(
                name="Inference prefill attention core contract",
                status="MISMATCH",
                details=(
                    "RAPID uses absorbed latent attention_score_1/2 + latent context + absorbed output, "
                    "while Megatron default inference prefill expands kv_compressed through linear_kv_up_proj "
                    "and runs core attention on full per-head Q/K/V."
                ),
                rapid_value=str(
                    {
                        "attention_score_1": shapes["attention_score_1"],
                        "attention_score_2": shapes["attention_score_2"],
                        "attention_ctx_latent": shapes["attention_ctx_latent"],
                        "output_proj": shapes["output_proj"],
                    }
                ),
                reference_value="Megatron prefill default path: q_up -> kv_up -> core_attention(query, key, value) -> row-parallel output",
                case=case.name,
            )
        )
    else:
        results.append(
            CheckResult(
                name="Inference decode Flash/absorption contract",
                status="MISMATCH",
                details=(
                    "Megatron only uses absorption in decode when cache_mla_latents is enabled, "
                    "and decode-only MLA uses a dedicated FlashMLA kernel path. RAPID currently models "
                    "absorbed latent cache semantics directly without this mode split."
                ),
                rapid_value=str(
                    {
                        "attention_score_1": shapes["attention_score_1"],
                        "attention_ctx_latent": shapes["attention_ctx_latent"],
                        "output_proj": shapes["output_proj"],
                    }
                ),
                reference_value="Megatron default decode uses full KV unless cache_mla_latents=true; FlashMLA only on decode latent-cache path",
                case=case.name,
            )
        )
    return results


def run_validation() -> Dict[str, object]:
    if not MEGATRON_ROOT.exists():
        raise FileNotFoundError(f"Megatron reference repo missing: {MEGATRON_ROOT}")
    if not DEEPSPEED_ROOT.exists():
        raise FileNotFoundError(f"DeepSpeed reference repo missing: {DEEPSPEED_ROOT}")

    commits = {
        "megatron": _git_commit(MEGATRON_ROOT),
        "deepspeed": _git_commit(DEEPSPEED_ROOT),
    }

    code_fact_checks = _check_reference_code_facts()
    checks: List[CheckResult] = []
    checks.extend(code_fact_checks)
    checks.append(_compare_param_totals())

    toy_cases = [
        ToyCase(name="train_single", run_type="training", tp=1, cp=1),
        ToyCase(name="train_tp2", run_type="training", tp=2, cp=1),
        ToyCase(name="train_cp2", run_type="training", tp=1, cp=2),
        ToyCase(name="train_tp2_cp2", run_type="training", tp=2, cp=2),
        ToyCase(name="infer_prefill_single", run_type="inference", tp=1, cp=1, phase="prefill"),
        ToyCase(name="infer_prefill_tp2", run_type="inference", tp=2, cp=1, phase="prefill"),
        ToyCase(name="infer_prefill_cp2", run_type="inference", tp=1, cp=2, phase="prefill"),
        ToyCase(name="infer_prefill_tp2_cp2", run_type="inference", tp=2, cp=2, phase="prefill"),
        ToyCase(name="infer_decode_single", run_type="inference", tp=1, cp=1, phase="decode"),
        ToyCase(name="infer_decode_tp2", run_type="inference", tp=2, cp=1, phase="decode"),
        ToyCase(name="infer_decode_cp2", run_type="inference", tp=1, cp=2, phase="decode"),
        ToyCase(name="infer_decode_tp2_cp2", run_type="inference", tp=2, cp=2, phase="decode"),
    ]
    for case in toy_cases:
        checks.extend(_compare_case(case))

    status_counts: Dict[str, int] = {}
    for item in checks:
        status_counts[item.status] = status_counts.get(item.status, 0) + 1

    payload = {
        "commits": commits,
        "status_counts": status_counts,
        "checks": [asdict(item) for item in checks],
    }
    return payload


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = run_validation()
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote MLA code-validation results to {OUT_JSON}")
    print(json.dumps(payload["status_counts"], indent=2, sort_keys=True))
    mismatches = [item for item in payload["checks"] if item["status"] != "PASS"]
    if mismatches:
        print("\nKey mismatches:")
        for item in mismatches:
            case = f" [{item['case']}]" if item.get("case") else ""
            print(f"- {item['name']}{case}: {item['details']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
