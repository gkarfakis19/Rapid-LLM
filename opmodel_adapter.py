"""Optional extended-roofline GEMM backend (eyao600/op-model integration).

Experimental adapter that routes RAPID-LLM's per-GEMM kernel time through the
`opmodel` extended-roofline model (shape-dependent effective utilization from
CTA tiling, wave quantization, L2 reuse, and per-level bandwidth limits)
instead of the native tile model. Everything else — tile selection, memory
access accounting, launch overhead, communication, scheduling — stays native.

Opt-in via environment variables (no config-schema change while experimental):

    RAPID_GEMM_BACKEND=extended_roofline   enable the backend
    RAPID_OPMODEL_PATH=/path/to/op-model   repo checkout (its src/ is imported)

Semantics:
- The opmodel HardwareSpec is DERIVED from the RAPID hardware config (clock,
  SM count, tensor FLOP rate, L2 and DRAM size/bandwidth), so there is a
  single source of truth for device parameters.
- opmodel is queried with ideal utilizations and zero fixed overhead; its
  latency already embeds shape-dependent efficiency. RAPID's
  tech_param.core.util is applied as a residual global scale (latency / util)
  so the calibrated factor keeps one clear meaning, and RAPID's own
  kernel_launch_overhead is added by the caller exactly as for the native
  backend (no double counting).
- bf16/fp16 (2-byte) precision only; other precisions fall back to the native
  tile model. FlashAttention paths always stay native.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Any, Optional

_BACKEND_ENV = "RAPID_GEMM_BACKEND"
_PATH_ENV = "RAPID_OPMODEL_PATH"


def backend_requested() -> bool:
    return os.environ.get(_BACKEND_ENV, "").strip().lower() == "extended_roofline"


class ExtendedRooflineGemmBackend:
    """Per-TimeCalculation adapter instance (caches model + hardware spec)."""

    def __init__(self, hw_config: Any) -> None:
        path = os.environ.get(_PATH_ENV, "").strip()
        if path:
            src = os.path.join(path, "src")
            if os.path.isdir(src) and src not in sys.path:
                sys.path.insert(0, src)
        try:
            from opmodel import DType, LocalOp, OpKind, Phase, TensorRole, TensorSpec
            from opmodel.hardware import _parse_hardware
            from opmodel.registry import create_model
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                f"{_BACKEND_ENV}=extended_roofline requires the opmodel package; "
                f"set {_PATH_ENV} to an eyao600/op-model checkout."
            ) from exc

        self._api = {
            "DType": DType,
            "LocalOp": LocalOp,
            "OpKind": OpKind,
            "Phase": Phase,
            "TensorRole": TensorRole,
            "TensorSpec": TensorSpec,
        }
        self._model = create_model("extended_roofline")
        self._hardware = _parse_hardware(self._hardware_dict(hw_config))
        self._warned_precision = False

    @staticmethod
    def _hardware_dict(hw_config: Any) -> dict:
        core = hw_config.tech_config.core
        freq = float(core.operating_frequency)
        num_sms = int(core.num_bundles)
        tensor_bf16 = (
            freq
            * num_sms
            * float(core.num_mcu_per_bundle)
            * float(core.nominal_flop_rate_per_mcu)
        )
        dram = hw_config.tech_config.DRAM
        l2 = hw_config.tech_config.SRAML2
        levels = [
            {
                "name": "l2",
                "size_bytes": int(l2.size),
                "bandwidth_bytes_per_s": float(l2.bandwidth),
                "energy_j_per_byte": 3.8e-11,
            },
            {
                "name": "hbm",
                "size_bytes": int(dram.size),
                "bandwidth_bytes_per_s": float(dram.bandwidth),
                "energy_j_per_byte": 1.05e-10,
            },
        ]
        return {
            "name": "rapid_derived",
            "kind": "gpu",
            "static_power_w": 1.0,
            "compute": {
                "clock_hz": freq,
                # RAPID adds its own kernel_launch_overhead; avoid double count.
                "device_fixed_overhead_cycles": 0,
                "num_sms": num_sms,
                "fma_dims": [16, 8, 16],
                "dataflow": "output_stationary",
                "vector_flops_per_s": {"fp32": tensor_bf16 / 16.0, "bf16": tensor_bf16 / 8.0},
                "tensor_flops_per_s": {
                    "tf32": tensor_bf16 / 2.0,
                    "bf16": tensor_bf16,
                    "fp16": tensor_bf16,
                    "int8": tensor_bf16 * 2.0,
                },
                "vector_energy_j_per_flop": {"fp32": 2.0e-11, "bf16": 2.0e-11},
                "tensor_energy_j_per_flop": {
                    "tf32": 2.5e-12,
                    "bf16": 1.3e-12,
                    "fp16": 1.3e-12,
                    "int8": 6.3e-13,
                },
            },
            "memory": {"levels": levels},
            "utilization": {
                "vector": 1.0,
                "tensor": 1.0,
                "memory": {level["name"]: 1.0 for level in levels},
            },
        }

    def gemm_latency_s(
        self, m: int, k: int, n: int, precision_bytes: float, batch: int = 1
    ) -> Optional[float]:
        """Kernel latency for A[m,k] x B[k,n] (batched if batch > 1).

        Returns None when the backend cannot model the request (caller falls
        back to the native tile model).
        """
        if abs(float(precision_bytes) - 2.0) > 1e-6:
            if not self._warned_precision:
                warnings.warn(
                    "extended_roofline GEMM backend supports 2-byte precisions "
                    "only; falling back to the native tile model.",
                    stacklevel=2,
                )
                self._warned_precision = True
            return None
        m, k, n, batch = int(m), int(k), int(n), max(1, int(batch))
        if min(m, k, n) < 1:
            return None
        api = self._api
        dt = api["DType"].BF16
        role = api["TensorRole"]
        if batch > 1:
            tensors = (
                api["TensorSpec"](role.INPUT, (batch, m, k), dt),
                api["TensorSpec"](role.WEIGHT, (batch, k, n), dt),
                api["TensorSpec"](role.OUTPUT, (batch, m, n), dt),
            )
            kind = api["OpKind"].BATCHED_GEMM
        else:
            tensors = (
                api["TensorSpec"](role.INPUT, (m, k), dt),
                api["TensorSpec"](role.WEIGHT, (k, n), dt),
                api["TensorSpec"](role.OUTPUT, (m, n), dt),
            )
            kind = api["OpKind"].GEMM
        op = api["LocalOp"](
            name="rapid_gemm", kind=kind, phase=api["Phase"].TRAIN_FWD, tensors=tensors
        )
        profile = self._model.predict(op, self._hardware)
        latency = float(profile.latency_s)
        if latency <= 0.0 or latency != latency:
            return None
        return latency


def maybe_create(hw_config: Any) -> Optional[ExtendedRooflineGemmBackend]:
    """Create the backend when requested via env; None otherwise."""
    if not backend_requested():
        return None
    return ExtendedRooflineGemmBackend(hw_config)
