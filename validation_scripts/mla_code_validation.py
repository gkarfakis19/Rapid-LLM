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
    model = _build_mla_model(
        run_type=case.run_type,
    )
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
    total = llm_util.attention_kv_cache_token_bytes(
        "mla",
        batch_size=4,
        kv_heads=4,
        head_dim=16,
        precision_bytes=2,
        kv_lora_rank=4,
        num_heads=4,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=8,
    )
    return float(total) / float(max(1, case.cp))


def _megatron_reference_shapes(case: ToyCase) -> Dict[str, int]:
    batch_size = 4
    if case.run_type == "inference" and case.phase == "decode":
        # Dynamic decode with THD context parallel partitions active query tokens across CP
        # ranks. With one active token per request, this reduces the local request count.
        seq_local = 1
        batch_local = int(math.ceil(float(batch_size) / float(max(1, case.cp))))
        # After the CP all-gather into attention, each rank produces outputs for the full
        # decode batch while still using only its local TP head slice.
        attention_output_batch = batch_size
    else:
        seq_local = math.ceil(8 / case.cp)
        batch_local = batch_size
        attention_output_batch = batch_local
    heads_local = 4 // case.tp
    return {
        "d_proj_q_local_elems": batch_local * seq_local * 8,
        "d_proj_kv_plus_rope_local_elems": batch_local * seq_local * (4 + 4),
        "q_up_local_elems": batch_local * seq_local * heads_local * (12 + 4),
        "kv_up_local_elems": batch_local * seq_local * heads_local * (12 + 8),
        "output_proj_input_local_elems": attention_output_batch * seq_local * heads_local * 8,
    }


def _vllm_latent_cache_bytes_per_rank(*, cp: int) -> float:
    precision = 2
    batch_local = math.ceil(4 / max(1, cp))
    return float(batch_local * (4 + 4) * precision)


def _check_reference_code_facts() -> List[CheckResult]:
    checks: List[CheckResult] = []

    megatron_specs = _read_text(
        MEGATRON_ROOT / "megatron/core/models/gpt/gpt_layer_specs.py"
    )
    megatron_mla = _read_text(
        MEGATRON_ROOT / "megatron/core/transformer/multi_latent_attention.py"
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
    shapes = _rapid_runtime_shapes(case)

    rapid_q_down = _rapid_local_output_elements(case, "D_proj_q", GemmType.MLA_DOWN_PROJ)
    rapid_kv_down = _rapid_local_output_elements(case, "D_proj_kv", GemmType.MLA_DOWN_PROJ)
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

    rapid_q_up = _rapid_local_output_elements(case, "U_proj_q", GemmType.QKV)
    results.append(
        CheckResult(
            name="Local q up-projection output elements",
            status="PASS" if rapid_q_up == ref["q_up_local_elems"] else "MISMATCH",
            details="Megatron TP column-shards the MLA q up-projection over local heads.",
            rapid_value=rapid_q_up,
            reference_value=ref["q_up_local_elems"],
            case=case.name,
        )
    )
    if case.run_type == "inference" and case.phase == "decode":
        rapid_q_absorb = _rapid_local_output_elements(case, "Q_absorb", GemmType.QKV)
        rapid_v_up = _rapid_local_output_elements(case, "U_proj_v", GemmType.QKV)
        rapid_output_input = _rapid_local_output_elements(case, "attention_output", GemmType.ATTENTION_OUTPUT)
        batch_local = math.ceil(4 / max(1, case.cp))
        heads_local = 4 // case.tp
        results.append(
            CheckResult(
                name="Local query-absorb output elements",
                status="PASS" if rapid_q_absorb == batch_local * heads_local * 4 else "MISMATCH",
                details="vLLM-style MLA decode absorbs q_nope into latent query space before attention.",
                rapid_value=rapid_q_absorb,
                reference_value=batch_local * heads_local * 4,
                case=case.name,
            )
        )
        results.append(
            CheckResult(
                name="Local value up-projection output elements",
                status="PASS" if rapid_v_up == batch_local * heads_local * 8 else "MISMATCH",
                details="vLLM-style MLA decode applies the value up-projection after latent attention.",
                rapid_value=rapid_v_up,
                reference_value=batch_local * heads_local * 8,
                case=case.name,
            )
        )
        results.append(
            CheckResult(
                name="Local latent attention-output elements",
                status="PASS" if rapid_output_input == 4 * heads_local * 4 else "MISMATCH",
                details="vLLM-style MLA decode emits latent per-head outputs before the value up-projection.",
                rapid_value=rapid_output_input,
                reference_value=4 * heads_local * 4,
                case=case.name,
            )
        )
    else:
        rapid_kv_up = _rapid_local_output_elements(case, "U_proj_kv", GemmType.QKV)
        rapid_output_input = _rapid_local_output_elements(case, "attention_output", GemmType.ATTENTION_OUTPUT)
        results.append(
            CheckResult(
                name="Local kv up-projection output elements",
                status="PASS" if rapid_kv_up == ref["kv_up_local_elems"] else "MISMATCH",
                details="Megatron TP column-shards the MLA kv up-projection over local heads.",
                rapid_value=rapid_kv_up,
                reference_value=ref["kv_up_local_elems"],
                case=case.name,
            )
        )
        results.append(
            CheckResult(
                name="Local attention-output elements",
                status="PASS" if rapid_output_input == ref["output_proj_input_local_elems"] else "MISMATCH",
                details="Megatron default MLA attention emits local-head value vectors into the row-parallel output projection.",
                rapid_value=rapid_output_input,
                reference_value=ref["output_proj_input_local_elems"],
                case=case.name,
            )
        )

    if case.run_type == "inference":
        rapid_cache = _rapid_mla_cache_token_bytes_per_rank(case)
        reference_cache = _vllm_latent_cache_bytes_per_rank(cp=case.cp)
        cache_status = "PASS" if math.isclose(rapid_cache, reference_cache) else "MISMATCH"
        results.append(
            CheckResult(
                name="Per-rank inference KV-cache bytes per token",
                status=cache_status,
                details="RAPID MLA cache bytes compared against the vLLM-style latent-cache inference contract.",
                rapid_value=rapid_cache,
                reference_value=reference_cache,
                case=case.name,
            )
        )

    if case.run_type == "training":
        results.append(
            CheckResult(
                name="Training/prefill attention core contract",
                status="PASS" if "U_proj_q" in shapes and "U_proj_kv" in shapes and "attention_score_1" not in shapes else "MISMATCH",
                details=(
                    "Megatron default training MLA uses q_down/kv_down -> q_up/kv_up -> full per-head attention -> row-parallel output."
                ),
                rapid_value=str(
                    {
                        "U_proj_q": shapes.get("U_proj_q"),
                        "U_proj_kv": shapes.get("U_proj_kv"),
                        "attention_score_1": shapes.get("attention_score_1"),
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
                status="PASS" if "U_proj_q" in shapes and "U_proj_kv" in shapes and "attention_score_1" not in shapes else "MISMATCH",
                details=(
                    "Megatron default inference prefill expands kv_compressed through linear_kv_up_proj and runs core attention on full per-head Q/K/V."
                ),
                rapid_value=str(
                    {
                        "U_proj_q": shapes.get("U_proj_q"),
                        "U_proj_kv": shapes.get("U_proj_kv"),
                        "attention_score_1": shapes.get("attention_score_1"),
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
                name="Inference decode latent-cache contract",
                status=(
                    "PASS"
                    if "U_proj_q" in shapes
                    and "Q_absorb" in shapes
                    and "U_proj_v" in shapes
                    and "U_proj_kv" not in shapes
                    and "attention_score_1" not in shapes
                    else "MISMATCH"
                ),
                details=(
                    "vLLM-style MLA decode uses latent-cache attention with absorbed query projection and post-attention value up-projection."
                ),
                rapid_value=str(
                    {
                        "U_proj_q": shapes.get("U_proj_q"),
                        "Q_absorb": shapes.get("Q_absorb"),
                        "U_proj_v": shapes.get("U_proj_v"),
                        "U_proj_kv": shapes.get("U_proj_kv"),
                        "attention_score_1": shapes.get("attention_score_1"),
                        "output_proj": shapes["output_proj"],
                    }
                ),
                reference_value="vLLM decode uses q_up -> q_absorb -> latent attention(kv_c + k_pe) -> v_up -> output",
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
