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

"""The named config matrix pinned by the golden-equivalence harness.

Self-contained (no imports from tests/) so the same module works across
older commits when bisecting. Base configs are the small validation configs
(Llama2-7B-ish on a100_80GB) for fast turnaround.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

BACKEND_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "analytical": {"model": "analytical"},
    "hybrid": {"model": "astra", "astra": {"mode": "hybrid"}},
    "hierarchical": {"model": "astra", "astra": {"mode": "full_astrasim_hierarchical"}},
    "flattened": {"model": "astra", "astra": {"mode": "full_astrasim_flattened"}},
}

# Statuses a spec's legacy behavior can be pinned at.
STATUS_OK = "ok"                      # legacy runs and produces a result
STATUS_LEGACY_DEADLOCK = "legacy_deadlock"  # legacy AstraSim run never completes (zero-time)
STATUS_LEGACY_ERROR = "legacy_error"  # legacy raises before/while running


@dataclass
class EquivSpec:
    spec_id: str
    run_type: str  # "training" | "inference"
    backend: str
    dp: int = 1
    ep: int = 1
    tp: int = 1
    cp: int = 1
    pp: int = 1
    mb: int = 1
    tp_sp: bool = False
    zero_stage: int = 0
    grad_accum: int = 1
    full_recomputation: bool = False
    use_moe: bool = False
    model_type: str = "gpt"
    attention_type: str = "mha"
    num_layers: Optional[int] = None  # default: 2 * pp
    extra_model: Dict[str, Any] = field(default_factory=dict)
    extra_hw: Dict[str, Any] = field(default_factory=dict)

    def layers(self) -> int:
        return self.num_layers if self.num_layers is not None else 2 * self.pp

    def model_overrides(self) -> Dict[str, Any]:
        model_param: Dict[str, Any] = {
            "run_type": self.run_type,
            "model_type": self.model_type,
            "global_batch_size": 8,
            "gradient_accumulation_steps": int(self.grad_accum),
            "seq_len": 256 if self.run_type == "inference" else 128,
            "hidden_dim": 4096,
            "intermediate_size": 11008,
            "vocab_size": 32000,
            "num_layers": self.layers(),
            "attention": {
                "attention_type": self.attention_type,
                "num_heads": 32,
                "head_dim": 128,
                "kv_heads": 8 if self.attention_type == "gqa" else 32,
                "use_flashattention": False,
                "attention_tile_size": 128,
            },
        }
        if self.run_type == "inference":
            model_param["decode_len"] = 64
        if self.use_moe:
            model_param["moe"] = {
                "num_experts": 4,
                "top_k": 1,
                "moe_intermediate_size": 11008,
                "n_shared_experts": 0,
                "moe_layer_freq": 1,
                "first_k_dense_replace": 0,
            }
        model_param.update(self.extra_model)
        return {"model_param": model_param}

    def hardware_overrides(self) -> Dict[str, Any]:
        inference_replica = self.dp if self.run_type == "inference" else 1
        inference_moe_dp = self.dp if (self.run_type == "inference" and self.use_moe) else 1
        overrides: Dict[str, Any] = {
            "parallelism": {
                "tp": self.tp,
                "tp_sp": bool(self.tp_sp),
                "cp": self.cp,
                "pp": self.pp,
                "mb": self.mb,
                "train": {"dp": self.dp, "ep": self.ep, "tp_ep": True},
                "inference": {
                    "replica_count": inference_replica,
                    "moe_dp": inference_moe_dp,
                },
            },
            "execution_backend": BACKEND_OVERRIDES[self.backend],
        }
        if self.zero_stage:
            overrides.setdefault("sw_param", {})["dp_zero_stage"] = int(self.zero_stage)
        if self.full_recomputation:
            overrides.setdefault("sw_param", {})["full_recomputation"] = True
        if self.backend == "analytical":
            overrides["network"] = {
                "dimensions": [{"id": "dim0", "topology": {"type": "Ring"}}]
            }
        overrides.update(self.extra_hw)
        return overrides


def _base_rows() -> List[Dict[str, Any]]:
    return [
        dict(dp=1, tp=1, cp=1, pp=1, mb=1, tp_sp=False),
        dict(dp=2, tp=1, cp=1, pp=2, mb=2, tp_sp=False),
        dict(dp=1, tp=2, cp=1, pp=2, mb=2, tp_sp=True),
        dict(dp=1, tp=1, cp=2, pp=2, mb=2, tp_sp=False),
        dict(dp=2, tp=2, cp=2, pp=2, mb=2, tp_sp=True),
    ]


def _row_tag(row: Dict[str, Any]) -> str:
    return (
        f"dp{row['dp']}tp{row['tp']}cp{row['cp']}pp{row['pp']}"
        f"mb{row['mb']}sp{int(row['tp_sp'])}"
    )


def build_matrix() -> List[EquivSpec]:
    specs: List[EquivSpec] = []

    # Core: every backend on every base parallelism row (training, zero-0).
    for backend in BACKEND_OVERRIDES:
        for row in _base_rows():
            specs.append(
                EquivSpec(
                    spec_id=f"train:{backend}:{_row_tag(row)}",
                    run_type="training",
                    backend=backend,
                    **row,
                )
            )

    # ZeRO stages on the flattened + analytical paths (dp>1 rows).
    for backend in ("analytical", "flattened"):
        for zero in (2, 3):
            row = dict(dp=2, tp=1, cp=1, pp=2, mb=2, tp_sp=False)
            specs.append(
                EquivSpec(
                    spec_id=f"train:{backend}:{_row_tag(row)}:zero{zero}",
                    run_type="training",
                    backend=backend,
                    zero_stage=zero,
                    **row,
                )
            )

    # Gradient accumulation (exercises the no-DP graph variant).
    for backend in ("analytical", "flattened"):
        row = dict(dp=2, tp=1, cp=1, pp=2, mb=2, tp_sp=False)
        specs.append(
            EquivSpec(
                spec_id=f"train:{backend}:{_row_tag(row)}:ga2",
                run_type="training",
                backend=backend,
                grad_accum=2,
                **row,
            )
        )

    # Full recomputation (flattened-mode recompute nodes).
    row = dict(dp=1, tp=2, cp=1, pp=2, mb=2, tp_sp=True)
    specs.append(
        EquivSpec(
            spec_id=f"train:flattened:{_row_tag(row)}:recompute",
            run_type="training",
            backend="flattened",
            full_recomputation=True,
            **row,
        )
    )

    # MoE on the modes that support it (hybrid/hierarchical; flattened rejects MoE).
    for backend in ("hybrid", "hierarchical"):
        row = dict(dp=2, tp=2, cp=1, pp=2, mb=2, tp_sp=True)
        specs.append(
            EquivSpec(
                spec_id=f"train:{backend}:{_row_tag(row)}:moe:ep2",
                run_type="training",
                backend=backend,
                ep=2,
                use_moe=True,
                model_type="glm4_moe",
                **row,
            )
        )

    # Inference (prefill + decode) on a couple of rows.
    for backend in ("analytical", "flattened", "hierarchical"):
        for row in (
            dict(dp=1, tp=1, cp=1, pp=1, mb=1, tp_sp=False),
            dict(dp=2, tp=2, cp=1, pp=2, mb=2, tp_sp=True),
        ):
            specs.append(
                EquivSpec(
                    spec_id=f"inf:{backend}:{_row_tag(row)}",
                    run_type="inference",
                    backend=backend,
                    **row,
                )
            )

    ids = [s.spec_id for s in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate spec ids in equivalence matrix")
    return specs


MATRIX: List[EquivSpec] = build_matrix()
MATRIX_BY_ID: Dict[str, EquivSpec] = {s.spec_id: s for s in MATRIX}
