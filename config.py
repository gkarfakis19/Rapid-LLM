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

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import yaml as _yaml
from yaml import YAMLError as _YAMLError


_PRECISION_DTYPE_BYTES = {
    "mxfp4": 4.25 / 8.0,
    "int4": 0.5,
    "fp8": 1.0,
    "fp16": 2.0,
    "half": 2.0,
    "bf16": 2.0,
    "fp32": 4.0,
    "single": 4.0,
}


from program import _env_flag

_MOE_PADDING_WARNED = False
_MLA_DECODE_FLASH_WARNED = False
_VIT_MODEL_TYPES = {"vit", "vit_dinov3"}


def _is_vit_model_type(model_type: str) -> bool:
    return str(model_type or "").strip().lower() in _VIT_MODEL_TYPES


def _vit_default_num_prefix_tokens(model_type: str) -> int:
    return 5 if str(model_type or "").strip().lower() == "vit_dinov3" else 1


def _vit_effective_num_prefix_tokens(vision: Optional["ViTConfig"], model_type: str) -> int:
    """Prefix-token count for a ViT run.

    The explicit model_param.vision.num_prefix_tokens override wins when given;
    otherwise the model-family default applies (vit -> 1 CLS token,
    vit_dinov3 -> 5), which preserves prior behavior exactly.
    """
    if vision is not None and vision.num_prefix_tokens is not None:
        return int(vision.num_prefix_tokens)
    return int(_vit_default_num_prefix_tokens(model_type))


@dataclass(frozen=True)
class PrecisionConfig:
    tensor: float
    kv_cache: float
    parameters: float
    gradients: float
    grad_communication: float
    optimizer_states: float
    stats: float
    master_parameters: float

    @property
    def activations(self) -> float:
        return self.tensor

    @property
    def tensor_format(self) -> float:
        return self.tensor

    @property
    def requires_master_copy(self) -> bool:
        return self.master_parameters > 0.0

def _coerce_precision_value(value, *, tensor_bytes: Optional[float] = None, allow_as_tensor: bool = False) -> float:
    if isinstance(value, (int, float)):
        if value <= 0:
            raise ValueError("precision byte size must be positive")
        return float(value)

    if not isinstance(value, str):
        raise TypeError(f"Unsupported precision specification type: {type(value)!r}")

    normalized = value.strip().lower()
    if allow_as_tensor and normalized == "as_tensor_format":
        return float(tensor_bytes)

    if normalized in _PRECISION_DTYPE_BYTES:
        return _PRECISION_DTYPE_BYTES[normalized]

    parsed = float(normalized)
    if parsed <= 0:
        raise ValueError("precision byte size must be positive")
    return parsed


def _parse_precision_block(spec: dict) -> PrecisionConfig:
    """Parse precision configuration by directly reading each precision type.

    Each field can be:
    - A numeric byte count (e.g., 2.0, 4.0)
    - A dtype string (e.g., "mxfp4", "int4", "fp8", "fp16", "bf16", "fp32")
    - "as_tensor_format" (only for kv_cache, parameters, gradients, grad_communication)
    """
    tensor_bytes = _coerce_precision_value(spec["tensor_format"])

    # Parse each precision field directly
    kv_cache_bytes = _coerce_precision_value(
        spec["kv_cache"],
        tensor_bytes=tensor_bytes,
        allow_as_tensor=True,
    )

    parameter_bytes = _coerce_precision_value(
        spec.get("parameters", "as_tensor_format"),
        tensor_bytes=tensor_bytes,
        allow_as_tensor=True,
    )

    gradient_bytes = _coerce_precision_value(
        spec.get("gradients", "fp32"),  # Default to FP32 for accumulated gradients
        tensor_bytes=tensor_bytes,
        allow_as_tensor=True,
    )

    grad_comm_bytes = _coerce_precision_value(
        spec.get("grad_communication", "as_tensor_format"),
        tensor_bytes=tensor_bytes,
        allow_as_tensor=True,
    )

    optimizer_bytes = _coerce_precision_value(
        spec.get("optimizer_states", "fp32"),  # Default to FP32 for optimizer states
        tensor_bytes=tensor_bytes,
        allow_as_tensor=True,
    )

    stats_bytes = _coerce_precision_value(
        spec.get("stats", "fp32"),  # Default to FP32 for stats
        tensor_bytes=tensor_bytes,
        allow_as_tensor=True,
    )

    # master_parameters can be 0 (no master copy) or a positive value
    master_raw = spec.get("master_parameters", 0.0)
    if isinstance(master_raw, (int, float)) and master_raw == 0.0:
        master_bytes = 0.0
    else:
        master_bytes = _coerce_precision_value(
            master_raw,
            tensor_bytes=tensor_bytes,
            allow_as_tensor=False,
        )

    return PrecisionConfig(
        tensor=tensor_bytes,
        kv_cache=kv_cache_bytes,
        parameters=parameter_bytes,
        gradients=gradient_bytes,
        grad_communication=grad_comm_bytes,
        optimizer_states=optimizer_bytes,
        stats=stats_bytes,
        master_parameters=master_bytes,
    )


@dataclass
class CoreConfig:
    nominal_power_per_mcu: float
    nominal_flop_rate_per_mcu: float
    nominal_energy_per_flop: float
    nominal_voltage: float
    threshold_voltage: float
    margin_voltage: float
    operating_area_per_mcu: float
    num_mcu_per_bundle: int
    FMA_dims: tuple
    dataflow: str
    util: float
    num_bundles: int = None
    operating_frequency: float = None
    nominal_frequency: float = None
    nominal_area_per_mcu: float = None

    @classmethod
    def from_dict(cls, core_config_dict):
        return cls(
            nominal_power_per_mcu=core_config_dict.get("nominal_power_per_mcu", 0.1),
            nominal_flop_rate_per_mcu=core_config_dict["nominal_flop_rate_per_mcu"],
            nominal_energy_per_flop=core_config_dict["nominal_energy_per_flop"],
            nominal_voltage=core_config_dict.get("nominal_voltage", 0.1),
            threshold_voltage=core_config_dict.get("threshold_voltage", 0.1),
            margin_voltage=core_config_dict.get("margin_voltage", 0.1),
            operating_area_per_mcu=core_config_dict.get("operating_area_per_mcu", 0.1),
            num_mcu_per_bundle=core_config_dict["num_mcu_per_bundle"],
            FMA_dims=(core_config_dict["FMA_d1"], core_config_dict["FMA_d2"]),
            dataflow=core_config_dict["dataflow"],
            util=core_config_dict["util"],
            num_bundles=core_config_dict.get("num_bundles", None),
            operating_frequency=core_config_dict.get("operating_frequency", None),
            nominal_frequency=core_config_dict.get("nominal_frequency", None),
            nominal_area_per_mcu=core_config_dict.get("nominal_area_per_mcu", None),
        )


@dataclass
class DRAMConfig:
    dynamic_energy_per_bit: float
    static_power_per_bit: float
    area_per_bit: float
    stack_capacity: float
    area_per_stack: float
    latency: float
    mem_ctrl_area: float
    nominal_voltage: float
    threshold_voltage: float
    margin_voltage: float
    num_links_per_mm: int
    num_links_per_stack: int
    max_voltage: float
    util: float
    size: float = None
    bandwidth: float = None
    num_stacks: int = None
    operating_frequency: float = None
    nominal_frequency: float = None

    @classmethod
    def from_dict(cls, dram_config_dict):
        return cls(
            dynamic_energy_per_bit=dram_config_dict["dynamic_energy_per_bit"],
            static_power_per_bit=dram_config_dict.get("static_power_per_bit", 0.1),
            area_per_bit=dram_config_dict.get("area_per_bit", 0.1),
            stack_capacity=dram_config_dict.get("stack_capacity", 0.1),
            area_per_stack=dram_config_dict.get("area_per_stack", 0.1),
            latency=dram_config_dict["latency"],
            mem_ctrl_area=dram_config_dict.get("mem_ctrl_area", 0.1),
            nominal_voltage=dram_config_dict.get("nominal_voltage", 0.1),
            threshold_voltage=dram_config_dict.get("threshold_voltage", 0.1),
            margin_voltage=dram_config_dict.get("margin_voltage", 0.1),
            num_links_per_mm=dram_config_dict.get("num_links_per_mm", 1),
            num_links_per_stack=dram_config_dict.get("num_links_per_stack", 1),
            max_voltage=dram_config_dict.get("max_voltage", 0.1),
            util=dram_config_dict["util"],
            size=dram_config_dict.get("size", None),
            bandwidth=dram_config_dict.get("bandwidth", None),
            num_stacks=dram_config_dict.get("num_stacks", None),
            operating_frequency=dram_config_dict.get("operating_frequency", None),
            nominal_frequency=dram_config_dict.get("nominal_frequency", None),
        )


@dataclass
class SRAMConfig:
    dynamic_energy_per_bit: float
    static_power_per_bit: float
    area_per_bit: float
    bank_capacity: float
    controller_area_per_link: float
    latency: float
    overhead: float
    util: float
    size: float = None
    bandwidth: float = None

    @classmethod
    def from_dict(cls, sram_config_dict):
        return cls(
            dynamic_energy_per_bit=sram_config_dict["dynamic_energy_per_bit"],
            static_power_per_bit=sram_config_dict.get("static_power_per_bit", 0.1),
            area_per_bit=sram_config_dict.get("area_per_bit", 0.1),
            bank_capacity=sram_config_dict.get("bank_capacity", 0.1),
            controller_area_per_link=sram_config_dict.get("controller_area_per_link", 0.1),
            latency=sram_config_dict["latency"],
            overhead=sram_config_dict.get("overhead", 0.1),
            util=sram_config_dict["util"],
            size=sram_config_dict.get("size", None),
            bandwidth=sram_config_dict.get("bandwidth", None),
        )


@dataclass
class TechConfig:
    core: CoreConfig
    DRAM: DRAMConfig
    SRAML2: SRAMConfig
    SRAML1: SRAMConfig
    SRAMR: SRAMConfig

    @classmethod
    def from_dict(cls, tech_config_dict):
        return cls(
            core=CoreConfig.from_dict(tech_config_dict["core"]),
            DRAM=DRAMConfig.from_dict(tech_config_dict["DRAM"]),
            SRAML2=SRAMConfig.from_dict(tech_config_dict["SRAM-L2"]),
            SRAML1=SRAMConfig.from_dict(tech_config_dict["SRAM-L1"]),
            SRAMR=SRAMConfig.from_dict(tech_config_dict["SRAM-R"]),
        )


@dataclass
class AreaBreakdownConfig:
    proc_chip_area_budget: float
    core: float
    DRAM: float
    L2: float
    L1: float
    reg_mem: float
    node_area_budget: float
    network: "NetworkAreaConfig"

    @classmethod
    def from_dict(cls, area_config_dict):
        return cls(
            proc_chip_area_budget=area_config_dict["proc_chip_area_budget"],
            core=area_config_dict["core"],
            DRAM=area_config_dict["DRAM"],
            L2=area_config_dict["L2"],
            L1=area_config_dict["L1"],
            reg_mem=area_config_dict["reg_mem"],
            node_area_budget=area_config_dict["device_area_budget"],
            network=NetworkAreaConfig.from_dict(area_config_dict["network"]),
        )


@dataclass
class PerimeterBreakdownConfig:
    DRAM: float
    inter_node: float
    intra_node: float

    @classmethod
    def from_dict(cls, perimeter_config_dict):
        return cls(
            DRAM=perimeter_config_dict["DRAM"],
            inter_node=perimeter_config_dict["inter_node"],
            intra_node=perimeter_config_dict["intra_node"],
        )


@dataclass
class NetworkAreaConfig:
    inter_node: float
    intra_node: float

    @classmethod
    def from_dict(cls, network_config_dict):
        return cls(
            inter_node=network_config_dict["inter_node"],
            intra_node=network_config_dict["intra_node"],
        )


@dataclass
class PowerBreakdownConfig:
    TDP: float
    core: float
    DRAM: float
    L2: float
    L1: float
    reg_mem: float
    network: "NetworkPowerConfig"

    @classmethod
    def from_dict(cls, power_config_dict):
        return cls(
            TDP=power_config_dict["TDP"],
            core=power_config_dict["core"],
            DRAM=power_config_dict["DRAM"],
            L2=power_config_dict["L2"],
            L1=power_config_dict["L1"],
            reg_mem=power_config_dict["reg_mem"],
            network=NetworkPowerConfig.from_dict(power_config_dict["network"]),
        )


@dataclass
class NetworkPowerConfig:
    inter_node: float
    intra_node: float

    @classmethod
    def from_dict(cls, network_power_config_dict):
        return cls(
            inter_node=network_power_config_dict["inter_node"],
            intra_node=network_power_config_dict["intra_node"],
        )


SUPERPOD_ALLOWED_DIMENSION_INDEX = 1
SUPERPOD_ALLOWED_PARALLELISMS = {"pp", "dp"}


@dataclass(frozen=True)
class NetworkDimensionLayout:
    id: str
    label: str
    size: int
    topology_type: str
    bandwidth: object
    util: float
    latency: float
    size_2d: Optional[Tuple[int, int]] = None
    collective_override: Dict[str, str] = field(default_factory=dict)
    parallelisms: Tuple[str, ...] = field(default_factory=tuple)
    energy_per_bit: float = 0.0
    optimize_2dmap: bool = False
    superpod_variant: Optional[str] = None
    superpod_leaf_size: Optional[int] = None
    superpod_leaf_switches_per_su: Optional[int] = None
    superpod_spine_switches_per_su: Optional[int] = None

    @classmethod
    def from_raw(
        cls,
        raw: dict,
        *,
        parallelism_params: Dict[str, object],
        index: int,
    ) -> "NetworkDimensionLayout":
        if not isinstance(raw, dict):
            raise TypeError("each network dimension must be a mapping")

        raw_id = raw.get("id")
        dim_id = str(raw_id) if raw_id is not None else f"dim{index}"
        label = str(raw.get("label", dim_id))

        if "size" not in raw:
            raise ValueError(f"network dimension '{label}' is missing required field 'size'")
        size_raw = raw["size"]
        size_mode: str
        size_value: Optional[int] = None
        size_2d: Optional[Tuple[int, int]] = None
        tuple_entries: Optional[Tuple[object, object]] = None
        if isinstance(size_raw, str):
            normalized_size = size_raw.strip()
            if normalized_size.startswith("(") and normalized_size.endswith(")"):
                tuple_entries = tuple(
                    part.strip() for part in normalized_size[1:-1].split(",")
                )  # type: ignore[assignment]
                size_mode = "tuple"
            elif normalized_size.lower() == "auto":
                size_mode = "auto_scalar"
            else:
                try:
                    size_value = int(size_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"network dimension '{label}' size must be an integer or tuple") from exc
                if size_value < 1:
                    raise ValueError(f"network dimension '{label}' size must be >= 1")
                size_mode = "scalar"
        elif isinstance(size_raw, (list, tuple)):
            if len(size_raw) != 2:
                raise ValueError(f"network dimension '{label}' 2D size must have exactly two entries")
            tuple_entries = tuple(size_raw)  # type: ignore[assignment]
            size_mode = "tuple"
        else:
            size_mode = "auto_scalar"

        topo_dict = raw.get("topology")
        if not isinstance(topo_dict, dict):
            raise ValueError(f"network dimension '{label}' requires a 'topology' mapping")
        if "type" not in topo_dict:
            raise ValueError(f"network dimension '{label}' topology missing required 'type'")
        topo_type = str(topo_dict["type"])
        normalized_topo = topo_type.lower().replace("-", "").replace("_", "")
        is_2d_topo = normalized_topo in {"mesh2d", "torus2d", "kingmesh2d", "fcring2d"}
        is_superpod = normalized_topo == "superpod"

        superpod_variant: Optional[str] = None
        superpod_leaf_size: Optional[int] = None
        superpod_leaf_switches_per_su: Optional[int] = None
        superpod_spine_switches_per_su: Optional[int] = None

        if is_superpod:
            raw_variant = topo_dict.get("superpod_variant")
            if raw_variant is None:
                raise ValueError(
                    f"network dimension '{label}' SuperPOD topology requires 'superpod_variant' set to 'h100'"
                )
            superpod_variant = str(raw_variant).strip().lower()
            if superpod_variant != "h100":
                raise ValueError(
                    f"network dimension '{label}' SuperPOD superpod_variant must be 'h100' (got {raw_variant!r})"
                )

            def _parse_superpod_int(field: str, default: int) -> int:
                raw_value = topo_dict.get(field, default)
                if raw_value is None:
                    return default
                try:
                    parsed = int(raw_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"network dimension '{label}' SuperPOD {field} must be an integer"
                    ) from exc
                if parsed < 1:
                    raise ValueError(
                        f"network dimension '{label}' SuperPOD {field} must be > 0"
                    )
                return parsed

            superpod_leaf_size = _parse_superpod_int("leaf_size", 32)
            superpod_leaf_switches_per_su = _parse_superpod_int("leaf_switches_per_su", 8)
            superpod_spine_switches_per_su = _parse_superpod_int("spine_switches_per_su", 4)

        if "bandwidth" not in topo_dict:
            raise ValueError(f"network dimension '{label}' topology missing required 'bandwidth'")
        bandwidth_raw = topo_dict["bandwidth"]
        bandwidth = parse_bandwidth_string(bandwidth_raw)
        if isinstance(bandwidth, (list, tuple)):
            if not (is_2d_topo or is_superpod):
                raise ValueError(
                    f"network dimension '{label}' bandwidth tuple is only supported for 2D topologies or SuperPOD"
                )
            if len(bandwidth) != 2:
                raise ValueError(
                    f"network dimension '{label}' bandwidth tuple must have exactly two entries"
                )
            for entry in bandwidth:
                if entry is None:
                    raise ValueError(
                        f"network dimension '{label}' bandwidth tuple entries must be numeric"
                    )
                if float(entry) <= 0:
                    raise ValueError(
                        f"network dimension '{label}' bandwidth tuple entries must be > 0"
                    )
        else:
            if bandwidth is None:
                raise ValueError(
                    f"network dimension '{label}' bandwidth must be numeric"
                )
            if float(bandwidth) <= 0:
                raise ValueError(
                    f"network dimension '{label}' bandwidth must be > 0"
                )
            if is_superpod:
                raise ValueError(
                    f"network dimension '{label}' SuperPOD bandwidth must be a two-entry list/tuple"
                )

        try:
            util = float(topo_dict.get("util", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"network dimension '{label}' topology util must be numeric"
            ) from exc
        if util <= 0:
            raise ValueError(f"network dimension '{label}' topology util must be > 0")

        if "energy_per_bit" not in topo_dict:
            raise ValueError(
                f"network dimension '{label}' topology missing required 'energy_per_bit'"
            )
        try:
            energy_per_bit = float(topo_dict["energy_per_bit"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"network dimension '{label}' energy_per_bit must be numeric"
            ) from exc
        if energy_per_bit < 0:
            raise ValueError(
                f"network dimension '{label}' energy_per_bit must be >= 0"
            )

        if "latency" not in topo_dict:
            raise ValueError(f"network dimension '{label}' topology missing required 'latency'")
        try:
            latency = float(topo_dict["latency"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"network dimension '{label}' latency must be numeric") from exc

        raw_optimize = topo_dict.get("optimize_2dmap", False)
        if raw_optimize not in (True, False):
            raise ValueError(
                f"network dimension '{label}' topology optimize_2dmap must be a boolean when provided"
            )
        optimize_2dmap = bool(raw_optimize)
        if optimize_2dmap:
            if normalized_topo not in {"mesh2d", "torus2d", "kingmesh2d", "fcring2d"}:
                raise ValueError(
                    f"network dimension '{label}' sets optimize_2dmap but topology type '{topo_type}'"
                    " is not Mesh2D/Torus2D/KingMesh2D/FC-Ring2D"
                )

        collectives_raw = raw.get("collective_override")
        if collectives_raw is None and "collectives" in raw:
            collectives_raw = raw.get("collectives")
        if not collectives_raw:
            collectives_raw = {}
        if not isinstance(collectives_raw, dict):
            raise ValueError(f"network dimension '{label}' collective_override must be a mapping if provided")
        collective_override = {str(k): str(v) for k, v in collectives_raw.items()}

        parallelisms_raw = raw.get("parallelisms", [])
        if parallelisms_raw is None:
            parallelisms_raw = []
        if not isinstance(parallelisms_raw, Sequence) or isinstance(parallelisms_raw, (str, bytes)):
            raise ValueError(
                f"network dimension '{label}' parallelisms must be a sequence of names"
            )

        normalized_parallelisms: List[str] = []
        alias_map: Dict[str, str] = {}
        for entry in parallelisms_raw:
            name = str(entry).strip()
            if not name:
                raise ValueError(f"network dimension '{label}' has an empty parallelism name")
            normalized = name.lower()
            normalized_parallelisms.append(normalized)
            alias_map[normalized] = name

        if is_superpod:
            if index != SUPERPOD_ALLOWED_DIMENSION_INDEX:
                raise ValueError(
                    f"network dimension '{label}' SuperPOD is only supported on Dimension {SUPERPOD_ALLOWED_DIMENSION_INDEX}"
                )
            if set(normalized_parallelisms) != SUPERPOD_ALLOWED_PARALLELISMS:
                readable = [alias_map.get(name, name) for name in normalized_parallelisms]
                raise ValueError(
                    f"network dimension '{label}' SuperPOD must be assigned exactly to PP and DP parallelisms "
                    f"(got {readable})"
                )

        computed_product: Optional[int] = None
        auto_product: Optional[int] = None

        if size_mode in {"auto_scalar", "tuple"}:
            auto_product = _compute_dimension_parallelism_product(
                dimension_label=label,
                normalized_names=tuple(normalized_parallelisms),
                alias_map=alias_map,
                parallelism_params=parallelism_params,
            )
            if auto_product < 1:
                raise ValueError(f"network dimension '{label}' inferred size must be >= 1")

        if size_mode == "auto_scalar":
            size_value = auto_product
            computed_product = auto_product
        elif size_mode == "tuple":
            assert tuple_entries is not None
            resolved: List[Optional[int]] = [None, None]
            auto_entries = 0
            for idx, entry in enumerate(tuple_entries):
                if isinstance(entry, str):
                    normalized = entry.strip().lower()
                    if normalized == "auto":
                        auto_entries += 1
                        if auto_entries > 1:
                            raise ValueError(
                                f"network dimension '{label}' 2D size may include at most one 'auto' entry"
                            )
                        resolved[idx] = None
                        continue
                    if normalized in parallelism_params:
                        try:
                            factor = int(parallelism_params[normalized])
                        except (TypeError, ValueError) as exc:
                            alias = alias_map.get(normalized, normalized)
                            raise ValueError(
                                f"network dimension '{label}' 2D size parallelism '{alias}' must be an integer"
                            ) from exc
                        if factor < 1:
                            alias = alias_map.get(normalized, normalized)
                            raise ValueError(
                                f"network dimension '{label}' 2D size parallelism '{alias}' must be >= 1"
                            )
                        resolved[idx] = factor
                        continue
                    try:
                        factor = int(entry)
                    except (TypeError, ValueError) as exc:
                        alias = alias_map.get(normalized, normalized)
                        raise ValueError(
                            f"network dimension '{label}' 2D size entry '{alias}' must be an integer, "
                            "parallelism name, or 'auto'"
                        ) from exc
                    if factor < 1:
                        raise ValueError(
                            f"network dimension '{label}' 2D size entries must be >= 1"
                        )
                    resolved[idx] = factor
                else:
                    try:
                        factor = int(entry)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"network dimension '{label}' 2D size entries must be integers, parallelism names, or 'auto'"
                        ) from exc
                    if factor < 1:
                        raise ValueError(f"network dimension '{label}' 2D size entries must be >= 1")
                    resolved[idx] = factor

            known_product = 1
            for val in resolved:
                if val:
                    known_product *= val

            if auto_entries:
                if auto_product is None:
                    auto_product = _compute_dimension_parallelism_product(
                        dimension_label=label,
                        normalized_names=tuple(normalized_parallelisms),
                        alias_map=alias_map,
                        parallelism_params=parallelism_params,
                    )
                if known_product == 0:
                    raise ValueError(f"network dimension '{label}' 2D size known entries must be > 0")
                if auto_product % known_product != 0:
                    raise ValueError(
                        f"network dimension '{label}' 2D size mismatch: auto product {auto_product} "
                        f"is not divisible by provided factors {tuple_entries}"
                    )
                auto_value = auto_product // known_product
                if auto_value < 1:
                    raise ValueError(
                        f"network dimension '{label}' 2D size auto-resolved entry must be >= 1 (got {auto_value})"
                    )
                for idx in range(2):
                    if resolved[idx] is None:
                        resolved[idx] = auto_value
                        break
            else:
                if auto_product is None:
                    auto_product = _compute_dimension_parallelism_product(
                        dimension_label=label,
                        normalized_names=tuple(normalized_parallelisms),
                        alias_map=alias_map,
                        parallelism_params=parallelism_params,
                    )
                if known_product != auto_product:
                    raise ValueError(
                        f"network dimension '{label}' 2D size mismatch: provided shape {tuple_entries} "
                        f"product {known_product} does not match parallelism product {auto_product}"
                    )

            if resolved[0] is None or resolved[1] is None:
                raise ValueError(f"network dimension '{label}' 2D size could not be fully resolved")

            size_2d = (int(resolved[0]), int(resolved[1]))
            size_value = int(size_2d[0]) * int(size_2d[1])
            computed_product = auto_product
        else:
            computed_product = None

        if size_value is None:
            raise ValueError(f"network dimension '{label}' size could not be resolved")

        _validate_dimension_parallelisms(
            dimension_label=label,
            dimension_size=int(size_value) if size_value is not None else 0,
            normalized_names=tuple(normalized_parallelisms),
            alias_map=alias_map,
            parallelism_params=parallelism_params,
            expected_product=computed_product,
        )

        if is_superpod:
            total_boxes = int(size_value)
            leaf_size = int(superpod_leaf_size or 0)
            if leaf_size < 1:
                raise ValueError(
                    f"network dimension '{label}' SuperPOD leaf_size must be > 0"
                )
            if total_boxes % leaf_size != 0:
                divisors: List[int] = []
                root = int(math.isqrt(total_boxes)) if total_boxes >= 1 else 0
                for candidate in range(1, root + 1):
                    if total_boxes % candidate != 0:
                        continue
                    divisors.append(candidate)
                    other = total_boxes // candidate
                    if other != candidate:
                        divisors.append(other)
                divisors = sorted(divisors)
                raise ValueError(
                    f"network dimension '{label}' SuperPOD size mismatch: N_boxes={total_boxes} "
                    f"is not divisible by leaf_size={leaf_size}. Suggested leaf_size divisors: {divisors}"
                )
            num_su = total_boxes // leaf_size
            if num_su <= 1:
                raise ValueError(
                    f"network dimension '{label}' SuperPOD requires more than 1 SU (got {num_su})."
                )

        return cls(
            id=dim_id,
            label=label,
            size=int(size_value) if size_value is not None else 0,
            size_2d=size_2d,
            topology_type=topo_type,
            bandwidth=bandwidth,
            util=util,
            latency=latency,
            collective_override=collective_override,
            parallelisms=tuple(normalized_parallelisms),
            energy_per_bit=energy_per_bit,
            optimize_2dmap=optimize_2dmap,
            superpod_variant=superpod_variant,
            superpod_leaf_size=superpod_leaf_size,
            superpod_leaf_switches_per_su=superpod_leaf_switches_per_su,
            superpod_spine_switches_per_su=superpod_spine_switches_per_su,
        )

    @property
    def effective_bandwidth(self) -> float:
        bw = self.bandwidth
        if isinstance(bw, (list, tuple)):
            if not bw:
                return 0.0
            return float(bw[0]) * float(self.util)
        return float(bw) * float(self.util)


def _validate_dimension_parallelisms(
    *,
    dimension_label: str,
    dimension_size: int,
    normalized_names: Tuple[str, ...],
    alias_map: Dict[str, str],
    parallelism_params: Dict[str, object],
    expected_product: Optional[int] = None,
) -> None:
    if not normalized_names:
        return

    product = expected_product
    if product is None:
        product = _compute_dimension_parallelism_product(
            dimension_label=dimension_label,
            normalized_names=normalized_names,
            alias_map=alias_map,
            parallelism_params=parallelism_params,
        )

    if product != dimension_size:
        readable = [alias_map.get(name, name) for name in normalized_names]
        raise ValueError(
            f"Network dimension '{dimension_label}' size mismatch: declared size {dimension_size} "
            f"but parallelism factors ({readable}) imply {product}"
        )


def _compute_dimension_parallelism_product(
    *,
    dimension_label: str,
    normalized_names: Tuple[str, ...],
    alias_map: Dict[str, str],
    parallelism_params: Dict[str, object],
) -> int:
    product = 1
    for name in normalized_names:
        if name not in parallelism_params:
            alias = alias_map.get(name, name)
            raise ValueError(
                f"network dimension '{dimension_label}' references parallelism '{alias}' "
                "which is not defined in parallelism"
            )
        value = parallelism_params[name]
        if value in (None, False):
            alias = alias_map.get(name, name)
            raise ValueError(
                f"network dimension '{dimension_label}' parallelism '{alias}' must have a "
                "positive parallelism factor"
            )
        try:
            factor = int(value)
        except (TypeError, ValueError) as exc:
            alias = alias_map.get(name, name)
            raise ValueError(
                f"parallelism.{alias} must be an integer to compute network dimension sizes"
            ) from exc
        if factor < 1:
            alias = alias_map.get(name, name)
            raise ValueError(
                f"parallelism.{alias} must be >= 1 to compute network dimension sizes"
            )
        product *= factor

    return product


def _parse_network_layout(
    network_spec,
    parallelism_params: Dict[str, object],
) -> Tuple[Tuple[NetworkDimensionLayout, ...], Tuple[Tuple[int, int, float], ...], "NetworkOverlapConfig"]:
    if network_spec is None:
        raise ValueError("network section must be specified and include overlap settings")

    if not isinstance(network_spec, dict):
        raise ValueError("network must be provided as a mapping to supply overlap settings")

    faulty_links: Tuple[Tuple[int, int, float], ...] = _parse_faulty_links("network", network_spec.get("faulty_links", []))
    overlap_config = _parse_network_overlap(network_spec.get("overlap"))
    dimensions_spec = network_spec.get("dimensions")
    if dimensions_spec is None:
        raise ValueError("network.dimensions must be specified when network is a mapping")

    if not isinstance(dimensions_spec, Sequence) or isinstance(dimensions_spec, (str, bytes)):
        raise ValueError("network.dimensions must be a sequence of dimension mappings")

    dimensions: List[NetworkDimensionLayout] = []
    for index, entry in enumerate(dimensions_spec):
        dimensions.append(
            NetworkDimensionLayout.from_raw(
                entry,
                parallelism_params=parallelism_params,
                index=index,
            )
        )
    for idx, dim in enumerate(dimensions):
        topo_name = str(getattr(dim, "topology_type", "")).strip().lower()
        topo_name = topo_name.replace("-", "").replace("_", "")
        if idx > 0 and topo_name in {"mesh2d", "torus2d", "kingmesh2d", "fcring2d"}:
            raise ValueError(
                f"2D topology '{dim.topology_type}' is only supported on the first network dimension."
            )
    return tuple(dimensions), faulty_links, overlap_config


@dataclass(frozen=True)
class NetworkOverlapConfig:
    tp_overlap: float
    tp_sp_overlap: float
    cp_overlap: float


@dataclass(frozen=True)
class NetworkLayoutConfig:
    dimensions: Tuple[NetworkDimensionLayout, ...]
    faulty_links: Tuple[Tuple[int, int, float], ...] = field(default_factory=tuple)
    parallelism_map: Dict[str, NetworkDimensionLayout] = field(default_factory=dict)
    overlap_config: "NetworkOverlapConfig" = None

    def primary_dimension(self) -> Optional[NetworkDimensionLayout]:
        return self.dimensions[0] if self.dimensions else None

    def dimension_for_parallelism(self, name: str) -> Optional[NetworkDimensionLayout]:
        normalized = str(name).strip().lower()
        if normalized in self.parallelism_map:
            return self.parallelism_map[normalized]
        return self.primary_dimension()

    def link_for_parallelism(self, name: str) -> Tuple[float, float]:
        dim = self.dimension_for_parallelism(name)
        if dim is None:
            return 0.0, 0.0
        return dim.effective_bandwidth, dim.latency


def _parse_faulty_links(owner_label: str, faulty_links_raw) -> Tuple[Tuple[int, int, float], ...]:
    entries: List[Tuple[int, int, float]] = []
    if not faulty_links_raw:
        return tuple()
    if not isinstance(faulty_links_raw, Sequence) or isinstance(faulty_links_raw, (str, bytes)):
        raise ValueError(
            f"{owner_label} faulty_links must be a sequence of [src, dst, weight] entries"
        )
    for idx, entry in enumerate(faulty_links_raw):
        if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes)) or len(entry) != 3:
            raise ValueError(
                f"{owner_label} faulty_links[{idx}] must be a three-item sequence [src, dst, weight]"
            )
        src_raw, dst_raw, weight_raw = entry
        try:
            src = int(src_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{owner_label} faulty_links[{idx}][0] must be an integer endpoint"
            ) from exc
        try:
            dst = int(dst_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{owner_label} faulty_links[{idx}][1] must be an integer endpoint"
            ) from exc
        if src < 0 or dst < 0:
            raise ValueError(
                f"{owner_label} faulty_links[{idx}] endpoints must be >= 0"
            )
        try:
            weight = float(weight_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{owner_label} faulty_links[{idx}][2] must be a numeric reliability weight"
            ) from exc
        if weight < 0.0 or weight > 1.0:
            raise ValueError(
                f"{owner_label} faulty_links[{idx}] weight must be between 0.0 and 1.0"
            )
        entries.append((src, dst, weight))
    return tuple(entries)

def _parse_network_overlap(overlap_raw) -> "NetworkOverlapConfig":
    if not isinstance(overlap_raw, dict):
        raise ValueError("network.overlap must be a mapping with tp_overlap, tp_sp_overlap, and cp_overlap")
    required_fields = ("tp_overlap", "tp_sp_overlap", "cp_overlap")
    values = {}
    for field in required_fields:
        if field not in overlap_raw:
            raise ValueError(f"network.overlap missing required field '{field}'")
        try:
            val = float(overlap_raw[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"network.overlap.{field} must be numeric") from exc
        if val < 0.0 or val > 1.0:
            raise ValueError(f"network.overlap.{field} must be between 0.0 and 1.0")
        values[field] = val
    return NetworkOverlapConfig(
        tp_overlap=values["tp_overlap"],
        tp_sp_overlap=values["tp_sp_overlap"],
        cp_overlap=values["cp_overlap"],
    )


def _build_network_layout_config(
    dimensions: Sequence[NetworkDimensionLayout],
    faulty_links: Sequence[Tuple[int, int, float]] = (),
    overlap_config: Optional["NetworkOverlapConfig"] = None,
) -> NetworkLayoutConfig:
    parallelism_map: Dict[str, NetworkDimensionLayout] = {}
    for dim in dimensions:
        for pname in dim.parallelisms:
            if pname in parallelism_map:
                raise ValueError(
                    f"parallelism '{pname}' assigned to multiple network dimensions"
                )
            parallelism_map[pname] = dim
    return NetworkLayoutConfig(
        dimensions=tuple(dimensions),
        faulty_links=tuple(faulty_links),
        parallelism_map=parallelism_map,
        overlap_config=overlap_config,
    )


_PARALLELISM_DEFAULTS: Dict[str, object] = {
    "auto": False,
    "pp": 1,
    "mb": 1,
    "tp": 1,
    "cp": 1,
    "tp_sp": False,
}


@dataclass
class TrainParallelismConfig:
    dp: int
    ep: int
    tp_ep: bool

    @classmethod
    def from_dict(cls, train_block: Optional[Dict[str, object]]) -> "TrainParallelismConfig":
        train_block = _require_mapping("parallelism.train", train_block or {})
        dp = _coerce_int(_require_field("parallelism.train", train_block, "dp"), "parallelism.train.dp")
        ep = _coerce_int(_require_field("parallelism.train", train_block, "ep"), "parallelism.train.ep")
        tp_ep = _coerce_bool(_require_field("parallelism.train", train_block, "tp_ep"), "parallelism.train.tp_ep")
        return cls(dp=dp, ep=ep, tp_ep=tp_ep)


@dataclass
class InferenceParallelismConfig:
    replica_count: int
    moe_dp: int

    @classmethod
    def from_dict(cls, inference_block: Optional[Dict[str, object]]) -> "InferenceParallelismConfig":
        inference_block = _require_mapping("parallelism.inference", inference_block or {})
        replica_count = _coerce_int(
            _require_field("parallelism.inference", inference_block, "replica_count"),
            "parallelism.inference.replica_count",
        )
        moe_dp = _coerce_int(
            _require_field("parallelism.inference", inference_block, "moe_dp"),
            "parallelism.inference.moe_dp",
        )
        return cls(replica_count=replica_count, moe_dp=moe_dp)


@dataclass
class MemoryConfig:
    type: str
    scope: str

    @classmethod
    def from_dict(cls, d):
        return cls(
            type=d["type"],
            scope=d["scope"],
        )


@dataclass
class MemoryHierarchyConfig:
    num_levels: int
    mem_hr: list

    @classmethod
    def from_dict(cls, d):
        num_levels = len(d)
        mem_hr = [None] * num_levels
        for level in range(num_levels):
            m = MemoryConfig.from_dict(d["l" + str(level)])
            mem_hr[level] = m
        return cls(
            num_levels=num_levels,
            mem_hr=mem_hr,
        )


def _require_mapping(context: str, value: object) -> Dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return value


def _require_field(context: str, data: Dict[str, object], field: str) -> object:
    if field not in data:
        raise ValueError(f"{context}.{field} must be specified")
    return data[field]


def _parse_str_field(context: str, data: Dict[str, object], field: str) -> str:
    value = _require_field(context, data, field)
    return str(value).strip()


def _parse_int_field(context: str, data: Dict[str, object], field: str, *, min_value: Optional[int] = 1) -> int:
    value = _require_field(context, data, field)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}.{field} must be an integer (got {value!r})") from exc
    if min_value is not None and parsed < min_value:
        raise ValueError(f"{context}.{field} must be >= {min_value}")
    return parsed


def _coerce_bool(value: object, context: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"{context} must be a boolean (got {value!r})")


def _parse_bool_field(context: str, data: Dict[str, object], field: str) -> bool:
    value = _require_field(context, data, field)
    return _coerce_bool(value, f"{context}.{field}")


def _parse_positive_pair(value: object, context: str) -> Tuple[int, int]:
    if isinstance(value, int):
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"{context} must be > 0")
        return (parsed, parsed)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            first = int(value[0])
            second = int(value[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context} must be an int or a two-entry list/tuple of ints") from exc
        if first <= 0 or second <= 0:
            raise ValueError(f"{context} entries must be > 0")
        return (first, second)
    raise ValueError(f"{context} must be an int or a two-entry list/tuple of ints")


@dataclass
class GEMMConfig:
    mode: str
    M: int
    K: int
    N: int
    backward: bool
    gemm_shard_axis: str

    @classmethod
    def from_dict(cls, model_dict: Dict[str, object]) -> "GEMMConfig":
        model_dict = _require_mapping("model_param", model_dict)
        mode_raw = _parse_str_field("model_param", model_dict, "mode")
        mode = mode_raw.upper()
        if mode != "GEMM":
            raise ValueError(f"model_param.mode must be 'GEMM' for GEMM configs (got {mode_raw!r})")
        M = _parse_int_field("model_param", model_dict, "M")
        K = _parse_int_field("model_param", model_dict, "K")
        N = _parse_int_field("model_param", model_dict, "N")
        backward = _coerce_bool(model_dict.get("backward", False), "model_param.backward")
        axis_raw = _parse_str_field("model_param", model_dict, "gemm_shard_axis")
        axis = axis_raw.strip().lower()
        if axis not in {"row", "col"}:
            raise ValueError(
                "model_param.gemm_shard_axis must be 'row' or 'col' "
                f"(got {axis_raw!r})"
            )
        return cls(
            mode=mode,
            M=M,
            K=K,
            N=N,
            backward=backward,
            gemm_shard_axis=axis,
        )
@dataclass
class LLMAttentionConfig:
    attention_type: str
    num_heads: int
    kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    q_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    use_flashattention: bool = False
    attention_tile_size: Optional[int] = None
    #: Qwen3.5-style gated attention: the query projection also emits a
    #: per-head output gate, doubling its output width.
    output_gate: bool = False
    #: Sliding-window pattern (Gemma 3/4, Phi-4-mini-flash); None = full.
    window: Optional["AttentionWindowConfig"] = None

    @classmethod
    def from_dict(cls, attention_dict: Dict[str, object]) -> "LLMAttentionConfig":
        attention_dict = _require_mapping("model_param.attention", attention_dict)

        attn_type_raw = _parse_str_field("model_param.attention", attention_dict, "attention_type")
        attn_type = attn_type_raw.strip().lower()
        if attn_type not in {"mha", "gqa", "mla"}:
            raise ValueError(
                "model_param.attention.attention_type must be one of "
                f"'mha', 'gqa', or 'mla' (got {attn_type_raw!r})"
            )

        num_heads = _parse_int_field("model_param.attention", attention_dict, "num_heads")
        head_dim_raw = attention_dict.get("head_dim", None)
        if head_dim_raw is not None:
            try:
                head_dim = int(head_dim_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"model_param.attention.head_dim must be an integer when provided (got {head_dim_raw!r})"
                ) from exc
            if head_dim <= 0:
                raise ValueError("model_param.attention.head_dim must be a positive integer")
        else:
            head_dim = None
        kv_heads_raw = attention_dict.get("kv_heads", None)
        if attn_type == "gqa":
            if kv_heads_raw is None:
                raise ValueError(
                    "model_param.attention.kv_heads must be specified when attention_type='gqa'"
                )
            try:
                kv_heads = int(kv_heads_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"model_param.attention.kv_heads must be an integer when attention_type='gqa' (got {kv_heads_raw!r})"
                ) from exc
            if kv_heads <= 0:
                raise ValueError("model_param.attention.kv_heads must be a positive integer")
            if num_heads % kv_heads != 0:
                raise ValueError(
                    f"model_param.attention.kv_heads={kv_heads} must divide num_heads={num_heads}"
                )
        else:
            kv_heads = num_heads

        def _parse_optional_positive_int(raw_value: object, field_name: str) -> Optional[int]:
            if raw_value is None:
                return None
            try:
                parsed = int(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"model_param.attention.{field_name} must be an integer when provided "
                    f"(got {raw_value!r})"
                ) from exc
            if parsed <= 0:
                raise ValueError(
                    f"model_param.attention.{field_name} must be a positive integer"
                )
            return parsed

        kv_lora_rank = _parse_optional_positive_int(
            attention_dict.get("kv_lora_rank", None),
            "kv_lora_rank",
        )
        q_lora_rank = _parse_optional_positive_int(
            attention_dict.get("q_lora_rank", None),
            "q_lora_rank",
        )
        qk_nope_head_dim = _parse_optional_positive_int(
            attention_dict.get("qk_nope_head_dim", None),
            "qk_nope_head_dim",
        )
        qk_rope_head_dim = _parse_optional_positive_int(
            attention_dict.get("qk_rope_head_dim", None),
            "qk_rope_head_dim",
        )
        v_head_dim = _parse_optional_positive_int(
            attention_dict.get("v_head_dim", None),
            "v_head_dim",
        )
        if attn_type == "mla":
            if kv_lora_rank is None:
                raise ValueError(
                    "model_param.attention.kv_lora_rank must be specified when attention_type='mla'"
                )
            if q_lora_rank is None:
                raise ValueError(
                    "model_param.attention.q_lora_rank must be specified when attention_type='mla'"
                )
            if qk_nope_head_dim is None:
                raise ValueError(
                    "model_param.attention.qk_nope_head_dim must be specified when attention_type='mla'"
                )
            if qk_rope_head_dim is None:
                raise ValueError(
                    "model_param.attention.qk_rope_head_dim must be specified when attention_type='mla'"
                )
            if v_head_dim is None:
                raise ValueError(
                    "model_param.attention.v_head_dim must be specified when attention_type='mla'"
                )
        raw_flash = attention_dict.get("use_flashattention", False)
        use_flashattention = _coerce_bool(
            raw_flash,
            "model_param.attention.use_flashattention",
        )

        attention_tile_size = attention_dict.get("attention_tile_size", None)
        if use_flashattention:
            if attention_tile_size is None:
                raise ValueError(
                    "model_param.attention.attention_tile_size must be specified when flash attention is enabled"
                )
            try:
                attention_tile_size = int(attention_tile_size)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "model_param.attention.attention_tile_size must be an integer when flash attention is enabled "
                    f"(got {attention_tile_size!r})"
                ) from exc
            if attention_tile_size <= 0:
                raise ValueError(
                    "model_param.attention.attention_tile_size must be a positive integer when flash attention is enabled"
                )
        else:
            attention_tile_size = None

        output_gate = _coerce_bool(
            attention_dict.get("output_gate", False),
            "model_param.attention.output_gate",
        )
        window_raw = attention_dict.get("window", None)
        window = None if window_raw is None else AttentionWindowConfig.from_dict(window_raw)

        return cls(
            attention_type=attn_type,
            num_heads=num_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=q_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            use_flashattention=use_flashattention,
            attention_tile_size=attention_tile_size,
            output_gate=output_gate,
            window=window,
        )


@dataclass
class MoEConfig:
    num_experts: int
    top_k: int
    moe_intermediate_size: int
    n_shared_experts: int
    expert_imbalance_factor: float
    moe_layer_freq: int
    first_k_dense_replace: int

    @classmethod
    def from_dict(
        cls,
        moe_dict: Dict[str, object],
        *,
        validate: bool = True,
        fallback_intermediate_size: Optional[int] = None,
    ) -> "MoEConfig":
        moe_dict = _require_mapping("model_param.moe", moe_dict)
        imbalance_help = (
            "model_param.moe.expert_imbalance_factor must be a finite float >= 1.0. "
            "Use 1.0 to preserve the current perfectly balanced MoE behavior; "
            "values > 1.0 model one routed expert as hotter than the balanced average expert load."
        )

        def _parse_expert_imbalance_factor(value: object) -> float:
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(imbalance_help) from exc
            if not math.isfinite(parsed):
                raise ValueError(imbalance_help)
            if parsed < 1.0:
                raise ValueError(
                    "model_param.moe.expert_imbalance_factor must be >= 1.0. "
                    "Use 1.0 to preserve the current perfectly balanced MoE behavior; "
                    "values > 1.0 model one routed expert as hotter than the balanced average expert load."
                )
            return parsed

        if not validate:
            def _lenient_int(value: object, default: int) -> int:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return default

            if fallback_intermediate_size is None:
                fallback_intermediate_size = 1
            fallback_intermediate_size = _lenient_int(fallback_intermediate_size, 1)

            num_experts = _lenient_int(moe_dict.get("num_experts", 1), 1)
            top_k = _lenient_int(moe_dict.get("top_k", 1), 1)
            moe_intermediate_size = _lenient_int(
                moe_dict.get("moe_intermediate_size", fallback_intermediate_size),
                fallback_intermediate_size,
            )
            n_shared_experts = _lenient_int(moe_dict.get("n_shared_experts", 0), 0)
            expert_imbalance_factor = _parse_expert_imbalance_factor(
                moe_dict.get("expert_imbalance_factor", 1.0)
            )
            moe_layer_freq = _lenient_int(moe_dict.get("moe_layer_freq", 1), 1)
            first_k_dense_replace = _lenient_int(moe_dict.get("first_k_dense_replace", 0), 0)
        else:
            num_experts = _parse_int_field("model_param.moe", moe_dict, "num_experts")
            top_k = _parse_int_field("model_param.moe", moe_dict, "top_k")
            moe_intermediate_size = _parse_int_field("model_param.moe", moe_dict, "moe_intermediate_size")
            n_shared_experts = _coerce_int(
                moe_dict.get("n_shared_experts", 0),
                "model_param.moe.n_shared_experts",
                min_value=0,
            )
            expert_imbalance_factor = _parse_expert_imbalance_factor(
                moe_dict.get("expert_imbalance_factor", 1.0)
            )
            moe_layer_freq = _coerce_int(
                moe_dict.get("moe_layer_freq", 1),
                "model_param.moe.moe_layer_freq",
                min_value=1,
            )
            first_k_dense_replace = _coerce_int(
                moe_dict.get("first_k_dense_replace", 0),
                "model_param.moe.first_k_dense_replace",
                min_value=0,
            )

            if top_k > num_experts:
                raise ValueError("model_param.moe.top_k cannot exceed model_param.moe.num_experts")

        if expert_imbalance_factor > float(num_experts):
            raise ValueError(
                "model_param.moe.expert_imbalance_factor cannot exceed model_param.moe.num_experts. "
                "RAPID models imbalance as one hot routed expert and the remaining routed experts "
                "sharing the leftover tokens uniformly."
            )

        # TODO: Shared experts are modeled as replicated across EP for now.
        return cls(
            num_experts=num_experts,
            top_k=top_k,
            moe_intermediate_size=moe_intermediate_size,
            n_shared_experts=n_shared_experts,
            expert_imbalance_factor=expert_imbalance_factor,
            moe_layer_freq=moe_layer_freq,
            first_k_dense_replace=first_k_dense_replace,
        )


@dataclass
class ViTConfig:
    image_size: Tuple[int, int]
    patch_size: Tuple[int, int]
    #: Optional override for the prefix-token count (CLS/register tokens).
    #: None keeps the model-family default (vit -> 1, vit_dinov3 -> 5).
    num_prefix_tokens: Optional[int] = None

    @property
    def patch_dim(self) -> int:
        return int(self.patch_size[0]) * int(self.patch_size[1]) * 3

    @property
    def num_patches(self) -> int:
        return int(self.image_size[0] // self.patch_size[0]) * int(self.image_size[1] // self.patch_size[1])

    @classmethod
    def from_dict(cls, vision_dict: Dict[str, object]) -> "ViTConfig":
        vision_dict = _require_mapping("model_param.vision", vision_dict)
        image_size = _parse_positive_pair(_require_field("model_param.vision", vision_dict, "image_size"), "model_param.vision.image_size")
        patch_size = _parse_positive_pair(_require_field("model_param.vision", vision_dict, "patch_size"), "model_param.vision.patch_size")
        if image_size[0] % patch_size[0] != 0 or image_size[1] % patch_size[1] != 0:
            raise ValueError(
                "model_param.vision.image_size must be divisible by model_param.vision.patch_size"
            )
        num_prefix_tokens_raw = vision_dict.get("num_prefix_tokens", None)
        num_prefix_tokens = None
        if num_prefix_tokens_raw is not None:
            num_prefix_tokens = _coerce_int(
                num_prefix_tokens_raw,
                "model_param.vision.num_prefix_tokens",
                min_value=0,
            )
        allowed_keys = {"image_size", "patch_size", "num_prefix_tokens"}
        extra_keys = sorted(set(vision_dict.keys()) - allowed_keys)
        if extra_keys:
            raise ValueError(
                "model_param.vision only supports image_size, patch_size and num_prefix_tokens for ViT configs. "
                f"Unsupported keys: {', '.join(extra_keys)}"
            )

        return cls(
            image_size=image_size,
            patch_size=patch_size,
            num_prefix_tokens=num_prefix_tokens,
        )


# ---------------------------------------------------------------------------
# Block-typed layer plans (QIF P1.1)
#
# Hybrid models (Mamba2 / gated-DeltaNet / short-conv backbones) break the
# "one uniform layer repeated num_layers times" assumption the rest of this
# schema is built on. These classes DESCRIBE such a model; nothing prices it
# yet, so validate_model_config rejects a hybrid layer plan on every device
# class until the P2-P4 pricing laws land.
# ---------------------------------------------------------------------------

_BLOCK_KINDS = ("attention", "ssm", "linear_attn", "short_conv", "ffn", "moe")
_MIXER_BLOCK_KINDS = ("attention", "ssm", "linear_attn", "short_conv")
#: Block kinds with no device law yet (P2-P4 owns their pricing).
_HYBRID_BLOCK_KINDS = ("ssm", "linear_attn", "short_conv")
_FFN_DIM_KEYS = _BLOCK_KINDS + ("shared_expert", "default")
_LAYER_PLAN_STRUCTURES = ("sequential", "parallel_branch")
_SSM_VARIANTS = ("mamba1", "mamba2")

#: HuggingFace `layer_types` spellings -> RAPID block kinds. hf_to_config.py
#: ingests the HF field verbatim, so both spellings must parse here.
_HF_BLOCK_KIND_ALIASES = {
    "attention": "attention",
    "full_attention": "attention",
    "sliding_attention": "attention",
    "mamba": "ssm",
    "mamba2": "ssm",
    "ssm": "ssm",
    "linear_attention": "linear_attn",
    "linear_attn": "linear_attn",
    "conv": "short_conv",
    "short_conv": "short_conv",
    "ffn": "ffn",
    "mlp": "ffn",
    "moe": "moe",
}


def _reject_unknown_keys(context: str, data: Dict[str, object], allowed: Sequence[str]) -> None:
    extra_keys = sorted(set(data.keys()) - set(allowed))
    if extra_keys:
        raise ValueError(
            f"{context} does not support the keys: {', '.join(extra_keys)}. "
            f"Supported keys: {', '.join(sorted(allowed))}"
        )


def _parse_block_kind(value: object, context: str) -> str:
    key = str(value).strip().lower()
    if key not in _HF_BLOCK_KIND_ALIASES:
        raise ValueError(
            f"{context} must name a block kind ({', '.join(_BLOCK_KINDS)}) or a "
            f"HuggingFace layer_types spelling ({', '.join(sorted(_HF_BLOCK_KIND_ALIASES))}); "
            f"got {value!r}"
        )
    return _HF_BLOCK_KIND_ALIASES[key]


def _parse_block_kind_list(value: object, context: str) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{context} must be a non-empty list of block kinds")
    return tuple(_parse_block_kind(item, f"{context} entries") for item in value)


@dataclass(frozen=True)
class AttentionWindowConfig:
    """Sliding-window attention pattern (model_param.attention.window).

    ``local_global_interval`` = N means one global layer every N attention
    layers (Gemma 3 uses 6 -> 5 local : 1 global); 1 makes every layer
    global and the window inert.
    """

    window_size: int
    local_global_interval: int
    last_layer_global: bool

    @classmethod
    def from_dict(cls, window_dict: Dict[str, object]) -> "AttentionWindowConfig":
        window_dict = _require_mapping("model_param.attention.window", window_dict)
        _reject_unknown_keys(
            "model_param.attention.window",
            window_dict,
            ("window_size", "local_global_interval", "last_layer_global"),
        )
        window_size = _parse_int_field("model_param.attention.window", window_dict, "window_size")
        local_global_interval = _coerce_int(
            window_dict.get("local_global_interval", 1),
            "model_param.attention.window.local_global_interval",
            min_value=1,
        )
        last_layer_global = _coerce_bool(
            window_dict.get("last_layer_global", False),
            "model_param.attention.window.last_layer_global",
        )
        return cls(
            window_size=window_size,
            local_global_interval=local_global_interval,
            last_layer_global=last_layer_global,
        )


@dataclass(frozen=True)
class SSMBlockConfig:
    """State-space (Mamba) mixer block parameters (model_param.ssm).

    ``variant: mamba2`` is the chunked SSD form (head-structured, needs
    n_heads / d_head / n_groups / chunk_size); ``mamba1`` is the per-token
    selective scan (needs dt_rank, has no head structure).
    """

    variant: str
    d_state: int
    d_conv: int
    d_inner: Optional[int] = None
    expand: Optional[int] = None
    n_groups: Optional[int] = None
    n_heads: Optional[int] = None
    d_head: Optional[int] = None
    chunk_size: Optional[int] = None
    dt_rank: Optional[int] = None

    def resolve_d_inner(self, hidden_dim: int) -> int:
        """Mixer width: the explicit d_inner when given, else expand * hidden."""
        if self.d_inner is not None:
            return int(self.d_inner)
        return int(self.expand) * int(hidden_dim)

    @classmethod
    def from_dict(cls, ssm_dict: Dict[str, object]) -> "SSMBlockConfig":
        ssm_dict = _require_mapping("model_param.ssm", ssm_dict)
        allowed = (
            "variant", "d_state", "d_conv", "d_inner", "expand",
            "n_groups", "n_heads", "d_head", "chunk_size", "dt_rank",
        )
        _reject_unknown_keys("model_param.ssm", ssm_dict, allowed)

        variant = str(ssm_dict.get("variant", "mamba2")).strip().lower()
        if variant not in _SSM_VARIANTS:
            raise ValueError(
                "model_param.ssm.variant must be one of "
                f"{', '.join(_SSM_VARIANTS)} (got {ssm_dict.get('variant')!r})"
            )

        d_state = _parse_int_field("model_param.ssm", ssm_dict, "d_state")
        d_conv = _parse_int_field("model_param.ssm", ssm_dict, "d_conv")
        d_inner = None
        if ssm_dict.get("d_inner") is not None:
            d_inner = _coerce_int(ssm_dict["d_inner"], "model_param.ssm.d_inner", min_value=1)
        expand = None
        if ssm_dict.get("expand") is not None:
            expand = _coerce_int(ssm_dict["expand"], "model_param.ssm.expand", min_value=1)
        if d_inner is None and expand is None:
            raise ValueError(
                "model_param.ssm requires d_inner or expand: the mixer width is "
                "d_inner when given, else expand * hidden_dim."
            )
        if d_inner is not None and expand is not None:
            raise ValueError(
                "model_param.ssm accepts d_inner OR expand, not both. Published "
                "configs that carry both disagree (Falcon-H1 sets mamba_d_ssm 3072 "
                "next to expand 2 at hidden 3072; Falcon-Mamba sets expand 16 next "
                "to a d_inner of 8192), so the width must be stated once."
            )

        def _optional(field: str) -> Optional[int]:
            if ssm_dict.get(field) is None:
                return None
            return _coerce_int(ssm_dict[field], f"model_param.ssm.{field}", min_value=1)

        n_groups = _optional("n_groups")
        n_heads = _optional("n_heads")
        d_head = _optional("d_head")
        chunk_size = _optional("chunk_size")
        dt_rank = _optional("dt_rank")

        if variant == "mamba2":
            missing = [
                name
                for name, value in (
                    ("n_groups", n_groups),
                    ("n_heads", n_heads),
                    ("d_head", d_head),
                    ("chunk_size", chunk_size),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "model_param.ssm.variant: mamba2 (chunked SSD) requires "
                    f"{', '.join('model_param.ssm.' + name for name in missing)}"
                )
        else:
            if dt_rank is None:
                raise ValueError(
                    "model_param.ssm.variant: mamba1 (selective scan) requires "
                    "model_param.ssm.dt_rank"
                )
            head_fields = [
                name
                for name, value in (
                    ("n_groups", n_groups),
                    ("n_heads", n_heads),
                    ("d_head", d_head),
                    ("chunk_size", chunk_size),
                )
                if value is not None
            ]
            if head_fields:
                raise ValueError(
                    "model_param.ssm.variant: mamba1 has no head structure and no "
                    "chunking; remove "
                    f"{', '.join('model_param.ssm.' + name for name in head_fields)} "
                    "or set variant: mamba2."
                )

        return cls(
            variant=variant,
            d_state=d_state,
            d_conv=d_conv,
            d_inner=d_inner,
            expand=expand,
            n_groups=n_groups,
            n_heads=n_heads,
            d_head=d_head,
            chunk_size=chunk_size,
            dt_rank=dt_rank,
        )


@dataclass(frozen=True)
class LinearAttentionBlockConfig:
    """Linear-attention (gated DeltaNet / WKV) block (model_param.linear_attention).

    Key and value head counts and dims are independent: Qwen3.5 carries 16
    key heads x 128 next to 32 value heads x 128.

    ``chunk_size`` is the linear-attention counterpart of ``ssm.chunk_size``:
    the Q of the chunked UT-transform kernel. It is OPTIONAL and has NO
    DEFAULT. Absent means the recurrent (token-by-token) form is priced, which
    is what every published gated-DeltaNet config forces today -- none of them
    publishes a chunk size. A default would be an invented number (ADJ-4), and
    it would not be a free one: the two forms differ by up to 1.38x in retired
    ops at this model's dims, in EITHER direction depending on Q.
    """

    num_key_heads: int
    key_head_dim: int
    num_value_heads: int
    value_head_dim: int
    conv_kernel: int
    output_gate: bool = True
    decay_gate: bool = True
    chunk_size: Optional[int] = None

    @property
    def key_dim(self) -> int:
        return int(self.num_key_heads) * int(self.key_head_dim)

    @property
    def value_dim(self) -> int:
        return int(self.num_value_heads) * int(self.value_head_dim)

    @classmethod
    def from_dict(cls, la_dict: Dict[str, object]) -> "LinearAttentionBlockConfig":
        context = "model_param.linear_attention"
        la_dict = _require_mapping(context, la_dict)
        allowed = (
            "num_key_heads", "key_head_dim", "num_value_heads", "value_head_dim",
            "conv_kernel", "output_gate", "decay_gate", "chunk_size",
        )
        _reject_unknown_keys(context, la_dict, allowed)

        num_key_heads = _parse_int_field(context, la_dict, "num_key_heads")
        key_head_dim = _parse_int_field(context, la_dict, "key_head_dim")
        num_value_heads = _parse_int_field(context, la_dict, "num_value_heads")
        value_head_dim = _parse_int_field(context, la_dict, "value_head_dim")
        conv_kernel = _parse_int_field(context, la_dict, "conv_kernel")
        if num_value_heads % num_key_heads != 0:
            raise ValueError(
                f"{context}.num_key_heads={num_key_heads} must divide "
                f"{context}.num_value_heads={num_value_heads}"
            )
        output_gate = _coerce_bool(la_dict.get("output_gate", True), f"{context}.output_gate")
        decay_gate = _coerce_bool(la_dict.get("decay_gate", True), f"{context}.decay_gate")
        chunk_size = None
        if la_dict.get("chunk_size") is not None:
            chunk_size = _parse_int_field(context, la_dict, "chunk_size")
            if chunk_size < 1:
                raise ValueError(
                    f"{context}.chunk_size={chunk_size} must be >= 1. 1 declares the "
                    "recurrent form explicitly; omit the field to leave it undeclared."
                )
        return cls(
            num_key_heads=num_key_heads,
            key_head_dim=key_head_dim,
            num_value_heads=num_value_heads,
            value_head_dim=value_head_dim,
            conv_kernel=conv_kernel,
            output_gate=output_gate,
            decay_gate=decay_gate,
            chunk_size=chunk_size,
        )


@dataclass(frozen=True)
class ShortConvBlockConfig:
    """Short causal depthwise-conv mixer (model_param.short_conv).

    ``double_gated`` is the LFM2 form: the input projection emits three
    conv_dim-wide streams (two gates plus the convolved stream).
    """

    kernel_size: int
    conv_dim: int
    double_gated: bool = True

    @property
    def in_proj_streams(self) -> int:
        return 3 if self.double_gated else 1

    @classmethod
    def from_dict(cls, conv_dict: Dict[str, object]) -> "ShortConvBlockConfig":
        context = "model_param.short_conv"
        conv_dict = _require_mapping(context, conv_dict)
        _reject_unknown_keys(context, conv_dict, ("kernel_size", "conv_dim", "double_gated"))
        return cls(
            kernel_size=_parse_int_field(context, conv_dict, "kernel_size"),
            conv_dim=_parse_int_field(context, conv_dict, "conv_dim"),
            double_gated=_coerce_bool(
                conv_dict.get("double_gated", True), f"{context}.double_gated"
            ),
        )


@dataclass(frozen=True)
class SharedWeightGroup:
    """Depth-shared weights (ADJ-3): one stored tensor set, fired by many layers.

    Zamba2 reuses two attention blocks across depth with a per-invocation
    LoRA projector. Under FWS the shared block is stored once, so the group
    is a real area statement, not an accounting note.
    """

    name: str
    layers: Tuple[int, ...]
    lora_rank: Optional[int] = None

    @classmethod
    def from_dict(cls, group_dict: Dict[str, object], *, index: int, num_layers: int) -> "SharedWeightGroup":
        context = f"model_param.shared_weight_groups[{index}]"
        group_dict = _require_mapping(context, group_dict)
        _reject_unknown_keys(context, group_dict, ("name", "layers", "lora_rank"))
        name = _parse_str_field(context, group_dict, "name")
        if not name:
            raise ValueError(f"{context}.name must be a non-empty string")
        layers_raw = _require_field(context, group_dict, "layers")
        if not isinstance(layers_raw, (list, tuple)) or len(layers_raw) < 2:
            raise ValueError(
                f"{context}.layers must be a list of at least two layer indices "
                "(a group of one shares nothing)"
            )
        layers = tuple(
            _coerce_int(item, f"{context}.layers entries", min_value=0) for item in layers_raw
        )
        if len(set(layers)) != len(layers):
            raise ValueError(f"{context}.layers must not repeat a layer index")
        for layer_idx in layers:
            if layer_idx >= num_layers:
                raise ValueError(
                    f"{context}.layers entry {layer_idx} is out of range for "
                    f"model_param.num_layers={num_layers}"
                )
        lora_rank = None
        if group_dict.get("lora_rank") is not None:
            lora_rank = _coerce_int(group_dict["lora_rank"], f"{context}.lora_rank", min_value=1)
        return cls(name=name, layers=tuple(sorted(layers)), lora_rank=lora_rank)


@dataclass(frozen=True)
class LayerPlanConfig:
    """Per-layer block structure (model_param.layer_plan).

    ``structure: sequential`` names one mixer kind per layer, either as an
    explicit ``layer_types`` list of length num_layers or as a ``pattern``
    repeated over the depth. ``structure: parallel_branch`` is the Falcon-H1
    template: every layer runs all of ``branches`` side by side.

    ``ffn_per_layer`` says each mixer layer carries its own FFN / MoE block,
    which is true of every hybrid except the pure-SSM stacks (Falcon-Mamba
    has no MLP at all).
    """

    structure: str
    layer_types: Tuple[str, ...]
    branches: Tuple[str, ...]
    pattern: Optional[Tuple[str, ...]]
    ffn_per_layer: bool

    @property
    def layer_mixers(self) -> Tuple[Tuple[str, ...], ...]:
        """Block kinds each layer runs, in layer order."""
        if self.structure == "parallel_branch":
            return tuple(self.branches for _ in range(len(self.layer_types)))
        return tuple((kind,) for kind in self.layer_types)

    @property
    def block_kinds(self) -> Tuple[str, ...]:
        present = set()
        for kinds in self.layer_mixers:
            present.update(kinds)
        return tuple(kind for kind in _BLOCK_KINDS if kind in present)

    @property
    def hybrid_block_kinds(self) -> Tuple[str, ...]:
        present = set(self.block_kinds)
        return tuple(kind for kind in _HYBRID_BLOCK_KINDS if kind in present)

    def count(self, kind: str) -> int:
        return sum(1 for kinds in self.layer_mixers if kind in kinds)

    @classmethod
    def from_dict(cls, plan_dict: Dict[str, object], *, num_layers: int) -> "LayerPlanConfig":
        context = "model_param.layer_plan"
        plan_dict = _require_mapping(context, plan_dict)
        _reject_unknown_keys(
            context, plan_dict, ("structure", "layer_types", "pattern", "branches", "ffn_per_layer")
        )
        structure = str(plan_dict.get("structure", "sequential")).strip().lower()
        if structure not in _LAYER_PLAN_STRUCTURES:
            raise ValueError(
                f"{context}.structure must be one of {', '.join(_LAYER_PLAN_STRUCTURES)} "
                f"(got {plan_dict.get('structure')!r})"
            )

        has_types = plan_dict.get("layer_types") is not None
        has_pattern = plan_dict.get("pattern") is not None
        has_branches = plan_dict.get("branches") is not None
        has_ffn_flag = plan_dict.get("ffn_per_layer") is not None

        if structure == "parallel_branch":
            if has_types or has_pattern:
                raise ValueError(
                    f"{context}.structure: parallel_branch describes ONE layer template; "
                    f"remove {context}.layer_types / {context}.pattern and list the "
                    f"concurrent blocks in {context}.branches."
                )
            if not has_branches:
                raise ValueError(
                    f"{context}.branches must be specified when "
                    f"{context}.structure is 'parallel_branch'"
                )
            if has_ffn_flag:
                raise ValueError(
                    f"{context}.ffn_per_layer is not allowed with "
                    f"structure: parallel_branch — {context}.branches already names "
                    "every block the layer runs."
                )
            branches = _parse_block_kind_list(plan_dict["branches"], f"{context}.branches")
            if len(set(branches)) != len(branches):
                raise ValueError(f"{context}.branches must not repeat a block kind")
            return cls(
                structure=structure,
                layer_types=tuple("parallel" for _ in range(num_layers)),
                branches=branches,
                pattern=None,
                ffn_per_layer=False,
            )

        if has_branches:
            raise ValueError(
                f"{context}.branches is only valid with structure: parallel_branch"
            )
        if has_types == has_pattern:
            raise ValueError(
                f"{context} requires exactly one of {context}.layer_types "
                f"(an explicit list of length num_layers) or {context}.pattern "
                "(a block sequence repeated over the depth)."
            )
        pattern = None
        if has_pattern:
            pattern = _parse_block_kind_list(plan_dict["pattern"], f"{context}.pattern")
            if num_layers % len(pattern) != 0:
                raise ValueError(
                    f"{context}.pattern of length {len(pattern)} does not tile "
                    f"model_param.num_layers={num_layers}; use {context}.layer_types "
                    "for an irregular plan."
                )
            layer_types = tuple(pattern[idx % len(pattern)] for idx in range(num_layers))
        else:
            layer_types = _parse_block_kind_list(plan_dict["layer_types"], f"{context}.layer_types")
            if len(layer_types) != num_layers:
                raise ValueError(
                    f"{context}.layer_types has {len(layer_types)} entries but "
                    f"model_param.num_layers={num_layers}; the explicit list must "
                    "name every layer."
                )
        ffn_per_layer = _coerce_bool(
            plan_dict.get("ffn_per_layer", True), f"{context}.ffn_per_layer"
        )
        if ffn_per_layer and any(kind in ("ffn", "moe") for kind in layer_types):
            raise ValueError(
                f"{context} names a dedicated 'ffn'/'moe' layer while "
                f"{context}.ffn_per_layer is true, which would count the FFN twice. "
                f"Set {context}.ffn_per_layer: false for a plan with standalone FFN layers."
            )
        return cls(
            structure=structure,
            layer_types=layer_types,
            branches=(),
            pattern=pattern,
            ffn_per_layer=ffn_per_layer,
        )


def _parse_ffn_dims(ffn_dims_raw: object) -> Dict[str, int]:
    """model_param.ffn_dims: per-block-kind FFN widths.

    Keys are block kinds plus 'shared_expert' (the always-on shared MLP of a
    Granite-style MoE layer) and 'default'; anything unnamed falls back to
    model_param.intermediate_size.
    """
    context = "model_param.ffn_dims"
    ffn_dims = _require_mapping(context, ffn_dims_raw)
    _reject_unknown_keys(context, ffn_dims, _FFN_DIM_KEYS)
    if not ffn_dims:
        raise ValueError(f"{context} must not be empty; omit it to use model_param.intermediate_size")
    return {
        str(key): _coerce_int(value, f"{context}.{key}", min_value=1)
        for key, value in ffn_dims.items()
    }


def _validate_block_schema(
    *,
    model_type: str,
    layer_plan: Optional[LayerPlanConfig],
    attention: Optional[LLMAttentionConfig],
    ssm: Optional[SSMBlockConfig],
    linear_attention: Optional[LinearAttentionBlockConfig],
    short_conv: Optional[ShortConvBlockConfig],
    hidden_dim: int,
) -> None:
    """Cross-validate the block-typed schema (QIF P1.1).

    Every block kind the layer plan names must carry its parameter block, and
    every parameter block present must be reachable from the layer plan — a
    dead block group is a silent modeling error, not a harmless extra.
    """
    block_groups = (
        ("ssm", "model_param.ssm", ssm),
        ("linear_attn", "model_param.linear_attention", linear_attention),
        ("short_conv", "model_param.short_conv", short_conv),
    )

    if _is_vit_model_type(model_type):
        named = [name for _, name, value in block_groups if value is not None]
        if layer_plan is not None:
            named.append("model_param.layer_plan")
        if named:
            raise ValueError(
                "ViT configs are uniform encoders and do not support the "
                f"block-typed schema; remove {', '.join(sorted(named))}."
            )

    if layer_plan is None:
        if attention is None:
            raise ValueError("model_param.attention must be specified")
        dead = [name for _, name, value in block_groups if value is not None]
        if dead:
            raise ValueError(
                f"{', '.join(sorted(dead))} requires model_param.layer_plan to say "
                "which layers run the block; without a layer plan every layer is a "
                "plain attention layer and the block group is dead config."
            )
        return

    kinds = set(layer_plan.block_kinds)
    if "attention" in kinds and attention is None:
        raise ValueError(
            "model_param.attention must be specified: model_param.layer_plan "
            "declares attention blocks."
        )
    if "attention" not in kinds and attention is not None:
        raise ValueError(
            "model_param.attention is set but model_param.layer_plan declares no "
            "attention block; remove the attention block or add attention layers."
        )
    for kind, name, value in block_groups:
        if kind in kinds and value is None:
            raise ValueError(
                f"{name} must be specified: model_param.layer_plan declares "
                f"{kind!r} blocks."
            )
        if kind not in kinds and value is not None:
            raise ValueError(
                f"{name} is set but model_param.layer_plan declares no {kind!r} "
                "block; remove it or add those layers."
            )

    if ssm is not None:
        d_inner = ssm.resolve_d_inner(hidden_dim)
        if ssm.variant == "mamba2":
            heads_width = int(ssm.n_heads) * int(ssm.d_head)
            if heads_width != d_inner:
                raise ValueError(
                    "model_param.ssm: n_heads * d_head must equal the mixer width "
                    f"(n_heads={ssm.n_heads} * d_head={ssm.d_head} = {heads_width}, "
                    f"d_inner = {d_inner})"
                )


class NoAttentionBlockError(AttributeError, ValueError):
    """Asked for an attention-only quantity on a model that has no attention.

    A pure-recurrence stack (Falcon-Mamba, and any layer plan whose block kinds
    exclude ``attention``) legally carries ``attention = None``, so
    ``LLMConfig.num_heads`` and ``LLMConfig.head_dim`` have no answer to give.
    They used to fail with a bare ``AttributeError: 'NoneType' object has no
    attribute 'num_heads'``, which names neither the config nor the reason.

    It subclasses BOTH ``AttributeError`` and ``ValueError`` on purpose: every
    ``getattr(model, "head_dim", None)`` / ``hasattr`` probe in the legacy
    timing path keeps swallowing it exactly as before (no priced number moves),
    while a caller that wants the refusal can catch it by name and a user sees
    a sentence instead of a NoneType.
    """


@dataclass
class LLMConfig:
    mode: str
    run_type: str
    model_type: str
    tied_embeddings: bool
    disable_embedding_unembedding: bool
    num_layers: int
    hidden_dim: int
    global_batch_size: int
    gradient_accumulation_steps: int
    seq_len: int
    decode_len: Optional[int]
    intermediate_size: Optional[int]
    vocab_size: int
    n_tokens: int
    attention: Optional[LLMAttentionConfig]
    moe: MoEConfig
    vision: Optional[ViTConfig] = None
    #: Block-typed layer structure (QIF P1.1). None = the uniform transformer
    #: layer the rest of this schema assumes.
    layer_plan: Optional[LayerPlanConfig] = None
    ssm: Optional[SSMBlockConfig] = None
    linear_attention: Optional[LinearAttentionBlockConfig] = None
    short_conv: Optional[ShortConvBlockConfig] = None
    ffn_dims: Dict[str, int] = field(default_factory=dict)
    shared_weight_groups: Tuple[SharedWeightGroup, ...] = ()
    #: The model's OWN name (QIF P1). ``model_type`` is a PRICING CARRIER — it
    #: selects the FFN/attention arithmetic — and several supported models
    #: declare a carrier that is not their name (Granite-4.0-H-Tiny declares
    #: ``llama`` for the SwiGLU gated MLP). Anything that shows a reader a model
    #: name (tile owners, atlas labels, the DAG report) uses THIS. Empty means
    #: "no separate name declared", and consumers fall back to ``model_type``,
    #: which is exactly the behaviour before this field existed.
    model_id: str = ""

    @property
    def num_heads(self) -> int:
        if self.attention is None:
            raise NoAttentionBlockError(
                "model_param.attention.num_heads was asked for, but this model has no "
                "attention block: its layer plan declares only "
                f"{list(self.block_kinds)}. A pure-recurrence stack has no heads; price "
                "its mixer through the ssm / linear_attention block instead."
            )
        return self.attention.num_heads

    @property
    def layer_mixers(self) -> Tuple[Tuple[str, ...], ...]:
        """Block kinds each layer runs. A plain transformer is all-attention."""
        if self.layer_plan is not None:
            return self.layer_plan.layer_mixers
        return tuple(("attention",) for _ in range(self.num_layers))

    @property
    def block_kinds(self) -> Tuple[str, ...]:
        if self.layer_plan is not None:
            return self.layer_plan.block_kinds
        return ("attention",)

    @property
    def hybrid_block_kinds(self) -> Tuple[str, ...]:
        """Block kinds present that no device law prices yet (P2-P4 owns them)."""
        if self.layer_plan is None:
            return ()
        return self.layer_plan.hybrid_block_kinds

    @property
    def has_hybrid_blocks(self) -> bool:
        return bool(self.hybrid_block_kinds)

    @property
    def ffn_per_layer(self) -> bool:
        if self.layer_plan is None:
            return True
        return bool(self.layer_plan.ffn_per_layer)

    def ffn_dim_for(self, block_kind: str) -> int:
        """FFN width for one block kind, falling back to intermediate_size."""
        for key in (str(block_kind), "default"):
            if key in self.ffn_dims:
                return int(self.ffn_dims[key])
        return int(self.intermediate_size)

    @property
    def head_dim(self) -> int:
        if self.attention is None:
            raise NoAttentionBlockError(
                "model_param.attention.head_dim was asked for, but this model has no "
                "attention block: its layer plan declares only "
                f"{list(self.block_kinds)}. A pure-recurrence stack has no heads; the "
                "recurrent state width lives in the ssm / linear_attention block."
            )
        if self.attention.head_dim is not None:
            return int(self.attention.head_dim)
        return self.hidden_dim // self.num_heads

    @property
    def use_flashattention(self) -> bool:
        return bool(getattr(self.attention, "use_flashattention", False))

    @property
    def kv_lora_rank(self) -> Optional[int]:
        return getattr(self.attention, "kv_lora_rank", None)

    @property
    def q_lora_rank(self) -> Optional[int]:
        return getattr(self.attention, "q_lora_rank", None)

    @property
    def qk_nope_head_dim(self) -> Optional[int]:
        return getattr(self.attention, "qk_nope_head_dim", None)

    @property
    def qk_rope_head_dim(self) -> Optional[int]:
        return getattr(self.attention, "qk_rope_head_dim", None)

    @property
    def v_head_dim(self) -> Optional[int]:
        return getattr(self.attention, "v_head_dim", None)

    @property
    def use_moe(self) -> bool:
        return self.num_experts > 1 and self.num_moe_layers > 0

    @property
    def num_experts(self) -> int:
        return self.moe.num_experts

    @property
    def top_k(self) -> int:
        return self.moe.top_k

    @property
    def num_moe_layers(self) -> int:
        return sum(self.moe_layer_mask)

    @property
    def n_shared_experts(self) -> int:
        return self.moe.n_shared_experts

    @property
    def expert_imbalance_factor(self) -> float:
        return self.moe.expert_imbalance_factor

    @property
    def moe_intermediate_size(self) -> int:
        return self.moe.moe_intermediate_size

    @property
    def moe_layer_freq(self) -> int:
        return self.moe.moe_layer_freq

    @property
    def first_k_dense_replace(self) -> int:
        return self.moe.first_k_dense_replace

    @property
    def moe_params_enabled(self) -> bool:
        return self.num_experts > 1 and self.num_moe_layers > 0

    @property
    def moe_layer_mask(self) -> List[bool]:
        if self.num_experts <= 1:
            return [False for _ in range(self.num_layers)]
        mask: List[bool] = []
        for layer_idx in range(self.num_layers):
            if layer_idx < self.first_k_dense_replace:
                mask.append(False)
                continue
            if self.moe_layer_freq <= 0:
                mask.append(False)
                continue
            mask.append(((layer_idx - self.first_k_dense_replace) % self.moe_layer_freq) == 0)
        return mask


    @property
    def grad_accumulation_steps(self) -> int:
        """Backward-compatible alias for gradient accumulation steps."""
        return self.gradient_accumulation_steps

    @property
    def is_vit(self) -> bool:
        return _is_vit_model_type(self.model_type)

    @property
    def num_classes(self) -> int:
        return 0 if not self.is_vit else int(max(0, self.vocab_size))

    @property
    def image_size(self) -> Optional[Tuple[int, int]]:
        return None if self.vision is None else tuple(self.vision.image_size)

    @property
    def patch_size(self) -> Optional[Tuple[int, int]]:
        return None if self.vision is None else tuple(self.vision.patch_size)

    @property
    def in_chans(self) -> int:
        return 3

    @property
    def num_prefix_tokens(self) -> int:
        return 0 if self.vision is None else _vit_effective_num_prefix_tokens(self.vision, self.model_type)

    @property
    def num_patches(self) -> int:
        return 0 if self.vision is None else int(self.vision.num_patches)

    @property
    def patch_dim(self) -> int:
        return 0 if self.vision is None else int(self.vision.patch_dim)

    @property
    def swiglu_mlp(self) -> bool:
        return self.model_type == "vit_dinov3"

    @classmethod
    def from_dict(cls, model_dict: Dict[str, object]) -> "LLMConfig":
        model_dict = _require_mapping("model_param", model_dict)

        mode_raw = _parse_str_field("model_param", model_dict, "mode")
        mode = mode_raw.strip().upper()
        if str(mode).upper() not in {"LLM", "VIT"}:
            raise ValueError(
                f"model_param.mode must be 'LLM' or 'ViT' for LLM-based configs (got {mode_raw!r})"
            )

        run_type_raw = _parse_str_field("model_param", model_dict, "run_type")
        run_type = run_type_raw.strip().lower()
        if run_type not in {"training", "inference"}:
            raise ValueError(
                f"model_param.run_type must be either 'training' or 'inference' (got {run_type_raw!r})"
            )

        tied_embeddings = _coerce_bool(
            _require_field("model_param", model_dict, "tied_embeddings"),
            "model_param.tied_embeddings",
        )

        model_type_raw = _parse_str_field("model_param", model_dict, "model_type")
        model_type = model_type_raw.strip().lower()
        if model_type in {"glm4", "glm"}:
            model_type = "glm4_moe"
        if model_type not in {"gpt", "llama", "deepseek_v3", "glm4_moe", "vit", "vit_dinov3"}:
            raise ValueError(
                "model_param.model_type must be one of 'gpt', 'llama', 'deepseek_v3', 'vit', 'vit_dinov3', or 'glm4_moe' "
                f"(got {model_type_raw!r})"
            )

        model_id = str(model_dict.get("model_id", "") or "").strip()

        attention_raw = model_dict.get("attention", None)
        attention = None if attention_raw is None else LLMAttentionConfig.from_dict(attention_raw)

        num_layers = _parse_int_field("model_param", model_dict, "num_layers")
        hidden_dim = _parse_int_field("model_param", model_dict, "hidden_dim")

        layer_plan = None
        if model_dict.get("layer_plan") is not None:
            layer_plan = LayerPlanConfig.from_dict(
                model_dict["layer_plan"], num_layers=num_layers
            )
        ssm = None
        if model_dict.get("ssm") is not None:
            ssm = SSMBlockConfig.from_dict(model_dict["ssm"])
        linear_attention = None
        if model_dict.get("linear_attention") is not None:
            linear_attention = LinearAttentionBlockConfig.from_dict(model_dict["linear_attention"])
        short_conv = None
        if model_dict.get("short_conv") is not None:
            short_conv = ShortConvBlockConfig.from_dict(model_dict["short_conv"])
        ffn_dims: Dict[str, int] = {}
        if model_dict.get("ffn_dims") is not None:
            ffn_dims = _parse_ffn_dims(model_dict["ffn_dims"])
        shared_weight_groups: Tuple[SharedWeightGroup, ...] = ()
        if model_dict.get("shared_weight_groups") is not None:
            groups_raw = model_dict["shared_weight_groups"]
            if not isinstance(groups_raw, (list, tuple)) or not groups_raw:
                raise ValueError(
                    "model_param.shared_weight_groups must be a non-empty list of "
                    "{name, layers[, lora_rank]} mappings"
                )
            shared_weight_groups = tuple(
                SharedWeightGroup.from_dict(entry, index=idx, num_layers=num_layers)
                for idx, entry in enumerate(groups_raw)
            )
            seen_layers: Dict[int, str] = {}
            for group in shared_weight_groups:
                for layer_idx in group.layers:
                    if layer_idx in seen_layers:
                        raise ValueError(
                            f"model_param.shared_weight_groups: layer {layer_idx} appears in "
                            f"both {seen_layers[layer_idx]!r} and {group.name!r}; a layer may "
                            "belong to at most one share group."
                        )
                    seen_layers[layer_idx] = group.name

        _validate_block_schema(
            model_type=model_type,
            layer_plan=layer_plan,
            attention=attention,
            ssm=ssm,
            linear_attention=linear_attention,
            short_conv=short_conv,
            hidden_dim=hidden_dim,
        )

        global_batch_size = _parse_int_field("model_param", model_dict, "global_batch_size")
        grad_accum_raw = model_dict.get("gradient_accumulation_steps", None)
        if grad_accum_raw is None:
            grad_accum_raw = model_dict.get("gradient_accumulation_step", None)
        if grad_accum_raw is None:
            gradient_accumulation_steps = 1
        else:
            try:
                gradient_accumulation_steps = int(grad_accum_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"model_param.gradient_accumulation_steps must be an integer (got {grad_accum_raw!r})"
                ) from exc
            if gradient_accumulation_steps <= 0:
                raise ValueError("model_param.gradient_accumulation_steps must be a positive integer")
        vision = None
        if _is_vit_model_type(model_type):
            vision = ViTConfig.from_dict(_require_field("model_param", model_dict, "vision"))
        elif "vision" in model_dict:
            raise ValueError(
                "model_param.vision is only supported when model_param.model_type is a ViT family model"
            )

        seq_len_raw = model_dict.get("seq_len", None)
        if seq_len_raw is None:
            if vision is None:
                raise ValueError("model_param.seq_len must be specified")
            seq_len = int(vision.num_patches) + _vit_effective_num_prefix_tokens(vision, model_type)
        else:
            try:
                seq_len = int(seq_len_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"model_param.seq_len must be an integer (got {seq_len_raw!r})") from exc
            if seq_len <= 0:
                raise ValueError("model_param.seq_len must be >= 1")
            if vision is not None:
                min_vit_seq_len = int(vision.num_patches) + _vit_effective_num_prefix_tokens(vision, model_type)
                if seq_len < min_vit_seq_len:
                    raise ValueError(
                        "model_param.seq_len must be >= the ViT-derived token count "
                        f"({min_vit_seq_len} from image_size/patch_size/prefix tokens, got {seq_len})"
                    )

        vocab_size_raw = model_dict.get("vocab_size", None)
        if _is_vit_model_type(model_type):
            if vocab_size_raw is None:
                raise ValueError(
                    "model_param.vocab_size must be specified for ViT configs and represents the classifier head size "
                    "(use 0 for a headless feature encoder)."
                )
            vocab_size = _coerce_int(vocab_size_raw, "model_param.vocab_size", min_value=0)
        else:
            vocab_size = _parse_int_field("model_param", model_dict, "vocab_size")

        intermediate_raw = model_dict.get("intermediate_size", None)
        if intermediate_raw is None:
            raise ValueError("model_param.intermediate_size must be specified")
        else:
            try:
                intermediate_size = int(intermediate_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"model_param.intermediate_size must be an integer (got {intermediate_raw!r})"
                ) from exc
            if intermediate_size <= 0:
                raise ValueError("model_param.intermediate_size must be >= 1")

        if attention is None:
            pass
        elif model_type == "glm4_moe":
            if attention.head_dim is None:
                raise ValueError(
                    "model_param.attention.head_dim must be specified when model_type is 'glm4_moe'"
                )
        elif attention.head_dim is not None:
            # A DECLARED head_dim is authoritative. It used to have to equal
            # hidden_dim // num_heads unless the model declared a hybrid layer
            # plan, and the model window D3 covers made that rule wrong for
            # plain transformers too: Gemma 3 4B is 2560 / 8 heads with
            # head_dim 256, Hunyuan-4B is 3072 / 32 heads with head_dim 128,
            # and neither declares a layer plan of any kind. The field was
            # therefore refusable on exactly the models it exists for, while
            # every CONSUMER of it — llm_util.attention_dim_sizes and the GEMM
            # descriptors, memory_estimation, train_timing and
            # CimDeviceModel.per_layer_stage_shapes — already reads the
            # declared value and sizes q / k / v / o from num_heads * head_dim.
            # The guard is replaced by proof that the declared value is HONORED
            # (tests/test_qif_model_matrix.py), which is what it was standing in
            # for. hidden_dim // num_heads is not computed here at all, so the
            # divisibility requirement below applies only when head_dim is
            # absent and the value has to be derived.
            pass
        else:
            if hidden_dim % attention.num_heads != 0:
                raise ValueError(
                    "model_param.hidden_dim must be divisible by attention.num_heads when "
                    "model_param.attention.head_dim is not provided"
                )

        decode_len = model_dict.get("decode_len", None)
        if decode_len is not None:
            try:
                decode_len = int(decode_len)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"model_param.decode_len must be an integer when provided (got {decode_len!r})"
                ) from exc
            if decode_len < 0:
                raise ValueError("model_param.decode_len must be >= 0")

        if run_type == "inference" and decode_len is None:
            if _is_vit_model_type(model_type):
                decode_len = 0
            else:
                raise ValueError("model_param.decode_len must be specified when run_type is 'inference'")

        disable_embedding_unembedding = _coerce_bool(
            model_dict.get("disable_embedding_unembedding", model_dict.get("disable_embedding_unembedding_ops", False)),
            "model_param.disable_embedding_unembedding",
        )
        if _is_vit_model_type(model_type):
            if decode_len not in (0, None):
                raise ValueError("model_param.decode_len must be 0 for ViT runs.")
            if disable_embedding_unembedding:
                raise ValueError(
                    "ViT does not support model_param.disable_embedding_unembedding=true. "
                    "Patch embedding and final ViT head/pooling are first-class modeled ops."
                )

        moe_block = model_dict.get("moe", {})
        if moe_block is None:
            moe_block = {}
        moe_candidate = MoEConfig.from_dict(
            moe_block,
            validate=False,
            fallback_intermediate_size=intermediate_size,
        )

        def _count_moe_layers(
            *,
            num_layers: int,
            moe_layer_freq: int,
            first_k_dense_replace: int,
        ) -> int:
            count = 0
            for layer_idx in range(num_layers):
                if layer_idx < first_k_dense_replace:
                    continue
                if moe_layer_freq <= 0:
                    continue
                if ((layer_idx - first_k_dense_replace) % moe_layer_freq) == 0:
                    count += 1
            return count

        moe_enabled = (
            moe_candidate.num_experts > 1
            and _count_moe_layers(
                num_layers=num_layers,
                moe_layer_freq=moe_candidate.moe_layer_freq,
                first_k_dense_replace=moe_candidate.first_k_dense_replace,
            )
            > 0
        )
        moe = moe_candidate
        if moe_enabled:
            moe = MoEConfig.from_dict(moe_block)

        return cls(
            mode=mode,
            run_type=run_type,
            model_type=model_type,
            tied_embeddings=tied_embeddings,
            disable_embedding_unembedding=disable_embedding_unembedding,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            global_batch_size=global_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            seq_len=seq_len,
            decode_len=decode_len,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            n_tokens=0,
            attention=attention,
            moe=moe,
            vision=vision,
            layer_plan=layer_plan,
            ssm=ssm,
            linear_attention=linear_attention,
            short_conv=short_conv,
            ffn_dims=ffn_dims,
            shared_weight_groups=shared_weight_groups,
            model_id=model_id,
        )


@dataclass
class LLMInferenceConfig:
    sample_every: int = -1

    @classmethod
    def from_dict(cls, inference_dict: Optional[Dict[str, object]]) -> "LLMInferenceConfig":
        if not inference_dict:
            return cls(sample_every=-1)
        inference_dict = _require_mapping("inference_param", inference_dict)
        raw = inference_dict.get("sample_every", -1)
        try:
            sample_every = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"inference_param.sample_every must be an integer (got {raw!r})"
            ) from exc
        return cls(sample_every=sample_every)
def _coerce_int(value: object, context: str, *, min_value: Optional[int] = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be an integer (got {value!r})") from exc
    if min_value is not None and parsed < min_value:
        raise ValueError(f"{context} must be >= {min_value}")
    return parsed


@dataclass
class SWConfig:
    kernel_launch_overhead: float
    precision: PrecisionConfig
    h2d_bandwidth: float
    dp_zero_stage: int
    full_recomputation: bool
    dp_microbatch: str
    const_mem_offset: float
    grad_acc_overhead: float
    #: Whether calc_time also runs the memory-estimation pass (the FLAT
    #: program build + replay). It is a separate OUTPUT, not a step of the
    #: timing computation; a caller that only wants a time can turn it off
    #: (sw_param.estimate_memory: false) and an analytical/hierarchical run
    #: skips its dominant wall-clock phase.
    estimate_memory: bool = True
    #: Whether build() runs the V1-V8/V6/V7 graph validation on production
    #: runs. OFF by default: it re-proves invariants the builder holds by
    #: construction; the wire-level emission postconditions (the actual
    #: deadlock guards) are always on. Deadlock-shaped runtime errors name
    #: this switch (sw_param.validate_graph / RAPID_VALIDATE_GRAPH).
    validate_graph: bool = False
    # Interleaved 1F1B (virtual pipeline) stages per rank. 1 = GPipe-style
    # schedule (the graph's native shape). v > 1 analytically rescales the
    # pipeline time by (mb + (pp-1)/v) / (mb + pp - 1), which is exact under
    # the simulator's own uniform-stage-time assumption.
    pipeline_interleave: int = 1

    @classmethod
    def from_dict(cls, sw_block: Dict[str, object]) -> "SWConfig":
        sw_block = _require_mapping("sw_param", sw_block)
        precision_spec = _require_field("sw_param", sw_block, "precision")
        precision_config = _parse_precision_block(precision_spec)
        kernel_launch_overhead = float(_require_field("sw_param", sw_block, "kernel_launch_overhead"))
        h2d_bandwidth = float(sw_block.get("h2d_bandwidth", -1))
        dp_zero_stage = _coerce_int(sw_block.get("dp_zero_stage", 0), "sw_param.dp_zero_stage", min_value=0)
        full_recomputation = _coerce_bool(
            sw_block.get("full_recomputation", False),
            "sw_param.full_recomputation",
        )
        dp_microbatch_raw = sw_block.get("dp_microbatch", "every_mb")
        dp_microbatch = str(dp_microbatch_raw).strip().lower()
        if dp_microbatch not in {"every_mb", "last_mb"}:
            raise ValueError("sw_param.dp_microbatch must be 'every_mb' or 'last_mb'")
        const_mem_offset_raw = sw_block.get("const_mem_offset", 0.0)
        if const_mem_offset_raw is None:
            const_mem_offset = 0.0
        else:
            try:
                const_mem_offset = float(const_mem_offset_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"sw_param.const_mem_offset must be a float-compatible value (got {const_mem_offset_raw!r})"
                ) from exc
        grad_acc_overhead_raw = sw_block.get("grad_acc_overhead", 0.0)
        if grad_acc_overhead_raw is None:
            grad_acc_overhead = 0.0
        else:
            try:
                grad_acc_overhead = float(grad_acc_overhead_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"sw_param.grad_acc_overhead must be a float-compatible value (got {grad_acc_overhead_raw!r})"
                ) from exc
        pipeline_interleave = _coerce_int(
            sw_block.get("pipeline_interleave", 1),
            "sw_param.pipeline_interleave",
            min_value=1,
        )
        estimate_memory_raw = sw_block.get("estimate_memory", True)
        if isinstance(estimate_memory_raw, str):
            estimate_memory = estimate_memory_raw.strip().lower() not in {"false", "0", "no", "off"}
        else:
            estimate_memory = bool(estimate_memory_raw)
        validate_graph_raw = sw_block.get("validate_graph", False)
        if isinstance(validate_graph_raw, str):
            validate_graph = validate_graph_raw.strip().lower() in {"true", "1", "yes", "on"}
        else:
            validate_graph = bool(validate_graph_raw)
        return cls(
            kernel_launch_overhead=kernel_launch_overhead,
            precision=precision_config,
            h2d_bandwidth=h2d_bandwidth,
            dp_zero_stage=dp_zero_stage,
            full_recomputation=full_recomputation,
            dp_microbatch=dp_microbatch,
            const_mem_offset=const_mem_offset,
            grad_acc_overhead=grad_acc_overhead,
            estimate_memory=estimate_memory,
            validate_graph=validate_graph,
            pipeline_interleave=pipeline_interleave,
        )


@dataclass
class SchedulingConfig:
    auto: bool
    pp: int
    mb: int
    tp: int
    cp: int
    tp_sp: bool
    train: TrainParallelismConfig
    inference: InferenceParallelismConfig

    @classmethod
    def from_dict(cls, parallelism_block: Optional[Dict[str, object]]) -> "SchedulingConfig":
        if parallelism_block is None:
            parallelism_block = {}
        parallelism_block = _require_mapping("parallelism", parallelism_block)
        params = dict(_PARALLELISM_DEFAULTS)
        params.update(parallelism_block)
        auto = _coerce_bool(params.get("auto", False), "parallelism.auto")
        tp_sp = _coerce_bool(params.get("tp_sp", False), "parallelism.tp_sp")
        pp = _coerce_int(params.get("pp", 1), "parallelism.pp")
        mb = _coerce_int(params.get("mb", 1), "parallelism.mb")
        tp = _coerce_int(params.get("tp", 1), "parallelism.tp")
        cp = _coerce_int(params.get("cp", 1), "parallelism.cp")
        train_cfg = TrainParallelismConfig.from_dict(_require_field("parallelism", parallelism_block, "train"))
        inference_cfg = InferenceParallelismConfig.from_dict(
            _require_field("parallelism", parallelism_block, "inference")
        )
        return cls(
            auto=auto,
            pp=pp,
            mb=mb,
            tp=tp,
            cp=cp,
            tp_sp=tp_sp,
            train=train_cfg,
            inference=inference_cfg,
        )


@dataclass
class FullConfig:
    model_config: object
    sw_config: SWConfig
    tech_config: TechConfig
    power_breakdown: PowerBreakdownConfig
    sch_config: SchedulingConfig
    area_breakdown: AreaBreakdownConfig
    perimeter_breakdown: PerimeterBreakdownConfig
    memory_hierarchy: MemoryHierarchyConfig
    network_layout: NetworkLayoutConfig


@dataclass
class ExecutionBackendAstraCollectives:
    all_gather: str = "auto"
    all_reduce: str = "auto"
    reduce_scatter: str = "auto"
    all_to_all: str = "auto"

    @classmethod
    def from_dict(cls, coll_dict: Optional[Dict[str, object]]) -> "ExecutionBackendAstraCollectives":
        if not coll_dict:
            return cls()
        coll_dict = _require_mapping("execution_backend.astra.collectives", coll_dict)
        return cls(
            all_gather=str(coll_dict.get("all_gather", "auto")),
            all_reduce=str(coll_dict.get("all_reduce", "auto")),
            reduce_scatter=str(coll_dict.get("reduce_scatter", "auto")),
            all_to_all=str(coll_dict.get("all_to_all", "auto")),
        )


@dataclass
class ExecutionBackendAstraSysOptions:
    endpoint_delay: Optional[int] = None
    active_chunks_per_dimension: Optional[int] = None
    preferred_dataset_splits: Optional[int] = None
    collective_arbitration: Optional[str] = None

    @classmethod
    def from_dict(cls, sys_dict: Optional[Dict[str, object]]) -> Optional["ExecutionBackendAstraSysOptions"]:
        if sys_dict is None:
            return None
        sys_dict = _require_mapping("execution_backend.astra.sys_options", sys_dict)
        endpoint_delay = sys_dict.get("endpoint_delay", None)
        active_chunks = sys_dict.get("active_chunks_per_dimension", None)
        preferred_splits = sys_dict.get("preferred_dataset_splits", None)
        collective_arbitration = sys_dict.get("collective_arbitration", None)
        if collective_arbitration is not None:
            collective_arbitration = str(collective_arbitration).strip().lower()
            allowed = {"off", "last_resort", "best_effort", "strict", "on", "true", "false", "0", "1"}
            if collective_arbitration not in allowed:
                raise ValueError(
                    "execution_backend.astra.sys_options.collective_arbitration "
                    f"must be one of {sorted(allowed)} (got {collective_arbitration!r})"
                )
        return cls(
            endpoint_delay=None if endpoint_delay is None else _coerce_int(endpoint_delay, "execution_backend.astra.sys_options.endpoint_delay", min_value=0),
            active_chunks_per_dimension=None if active_chunks is None else _coerce_int(active_chunks, "execution_backend.astra.sys_options.active_chunks_per_dimension", min_value=1),
            preferred_dataset_splits=None if preferred_splits is None else _coerce_int(preferred_splits, "execution_backend.astra.sys_options.preferred_dataset_splits", min_value=1),
            collective_arbitration=collective_arbitration,
        )


@dataclass
class ExecutionBackendAstra:
    backend: str
    mode: str
    collectives: ExecutionBackendAstraCollectives
    sys_options: Optional[ExecutionBackendAstraSysOptions]

    @classmethod
    def from_dict(cls, astra_dict: Optional[Dict[str, object]]) -> "ExecutionBackendAstra":
        astra_dict = _require_mapping("execution_backend.astra", astra_dict or {})
        backend = str(astra_dict.get("backend", "analytical"))
        mode = str(astra_dict.get("mode", "hybrid"))
        collectives = ExecutionBackendAstraCollectives.from_dict(astra_dict.get("collectives"))
        sys_options = ExecutionBackendAstraSysOptions.from_dict(astra_dict.get("sys_options"))
        return cls(
            backend=backend,
            mode=mode,
            collectives=collectives,
            sys_options=sys_options,
        )


@dataclass
class ExecutionBackend:
    model: str
    astra: Optional[ExecutionBackendAstra]

    @classmethod
    def from_dict(cls, backend_dict: Optional[Dict[str, object]]) -> "ExecutionBackend":
        backend_dict = _require_mapping("execution_backend", backend_dict or {})
        model = str(backend_dict.get("model", "analytical"))
        astra_cfg = backend_dict.get("astra", {}) if model == "astra" else None
        astra = ExecutionBackendAstra.from_dict(astra_cfg) if astra_cfg is not None else None
        return cls(model=model, astra=astra)


#: Valid inference.kvcache_type values. hbm_only is the GPU default;
#: cim_sram / cim_dram are the fws_cim KV stories (the device has no HBM).
_KVCACHE_TYPES = ("hbm_only", "cim_sram", "cim_dram")


@dataclass
class InferenceHWConfig:
    kvcache_type: str

    @classmethod
    def from_dict(cls, inference_dict: Optional[Dict[str, object]]) -> "InferenceHWConfig":
        if not inference_dict:
            return cls(kvcache_type="hbm_only")
        inference_dict = _require_mapping("inference", inference_dict)
        kvcache_type = str(inference_dict.get("kvcache_type", "hbm_only")).strip().lower()
        if kvcache_type not in _KVCACHE_TYPES:
            raise ValueError(
                "inference.kvcache_type must be one of 'hbm_only', 'cim_sram', or 'cim_dram' "
                f"(got {inference_dict.get('kvcache_type')!r})"
            )
        return cls(kvcache_type=kvcache_type)


def _parse_cim_float(block: Dict[str, object], context: str, field: str, *, default=None, min_value: float = 0.0, strict: bool = False):
    """Parse a plain-number float from a cim sub-block.

    The global convert() preprocessor rewrites any "<int> <Unit>" string in the
    YAML into byte counts, so cim blocks must use plain numbers only; a
    leftover string here means a unit string slipped in, which we reject.
    """
    if default is None:
        raw = _require_field(context, block, field)
    else:
        raw = block.get(field, default)
    if isinstance(raw, str):
        raise ValueError(
            f"{context}.{field} must be a plain number, not a string (got {raw!r}). "
            "Unit strings like '100 MB' are rewritten to byte counts elsewhere in the YAML "
            "and are not allowed inside the cim block."
        )
    try:
        parsed = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}.{field} must be a number (got {raw!r})") from exc
    if strict and parsed <= min_value:
        raise ValueError(f"{context}.{field} must be > {min_value} (got {parsed})")
    if not strict and parsed < min_value:
        raise ValueError(f"{context}.{field} must be >= {min_value} (got {parsed})")
    return parsed


@dataclass
class CIMAnalogConfig:
    """Analog fixed-weight-stationary array parameters (cim.analog).

    Modeling assumptions: an array stores rows x (cols_adc * adc_mux) weights;
    one input vector is evaluated in adc_mux * slice_cycles analog cycles;
    shots_per_output affects energy only, never time. All values are plain
    numbers (no unit strings — see convert()).
    """

    rows: int
    cols_adc: int
    adc_mux: int
    slice_cycles: int
    analog_clock_mhz: float
    energy_per_vec_pj: float   # per array per input vector (already includes mux); 0 reports zero analog energy
    shots_per_output: int      # energy only, never time
    area_mm2_per_array: float  # 0 disables area reporting

    @property
    def cols(self) -> int:
        """Stored weight columns per array (cols_adc * adc_mux)."""
        return int(self.cols_adc) * int(self.adc_mux)

    @classmethod
    def from_dict(cls, analog_dict: Optional[Dict[str, object]]) -> "CIMAnalogConfig":
        analog_dict = _require_mapping("cim.analog", analog_dict)
        return cls(
            rows=_parse_int_field("cim.analog", analog_dict, "rows"),
            cols_adc=_parse_int_field("cim.analog", analog_dict, "cols_adc"),
            adc_mux=_parse_int_field("cim.analog", analog_dict, "adc_mux"),
            slice_cycles=_parse_int_field("cim.analog", analog_dict, "slice_cycles"),
            analog_clock_mhz=_parse_cim_float(analog_dict, "cim.analog", "analog_clock_mhz", strict=True),
            energy_per_vec_pj=_parse_cim_float(analog_dict, "cim.analog", "energy_per_vec_pj", default=0.0),
            shots_per_output=_coerce_int(
                analog_dict.get("shots_per_output", 1), "cim.analog.shots_per_output", min_value=1
            ),
            area_mm2_per_array=_parse_cim_float(analog_dict, "cim.analog", "area_mm2_per_array", default=0.0),
        )


@dataclass
class CIMFabricConfig:
    """Digital attention-sidecar fabric parameters (cim.fabric).

    model 'sa' prices act-act attention GEMMs on a closed-form systolic-array
    law; 'gpu_native' lets them fall through to the native tile/roofline
    machinery priced against the stub tech_param.
    """

    model: str                      # "sa" | "gpu_native"
    rows: int
    cols: int
    num_arrays: int                 # QK and PV run concurrently on these
    replicas: int                   # attention array replicas
    clock_ghz: float
    fill_drain_penalty_cycles: int  # default 3 * rows when absent
    softmax_lanes: int
    softmax_pipeline_depth: int

    @classmethod
    def from_dict(cls, fabric_dict: Optional[Dict[str, object]]) -> "CIMFabricConfig":
        fabric_dict = _require_mapping("cim.fabric", fabric_dict)
        model = str(fabric_dict.get("model", "sa")).strip().lower()
        if model not in {"sa", "gpu_native"}:
            raise ValueError(
                f"cim.fabric.model must be 'sa' or 'gpu_native' (got {fabric_dict.get('model')!r})"
            )
        rows = _parse_int_field("cim.fabric", fabric_dict, "rows")
        return cls(
            model=model,
            rows=rows,
            cols=_parse_int_field("cim.fabric", fabric_dict, "cols"),
            num_arrays=_coerce_int(fabric_dict.get("num_arrays", 2), "cim.fabric.num_arrays", min_value=1),
            replicas=_coerce_int(fabric_dict.get("replicas", 1), "cim.fabric.replicas", min_value=1),
            clock_ghz=_parse_cim_float(fabric_dict, "cim.fabric", "clock_ghz", strict=True),
            fill_drain_penalty_cycles=_coerce_int(
                fabric_dict.get("fill_drain_penalty_cycles", 3 * rows),
                "cim.fabric.fill_drain_penalty_cycles",
                min_value=0,
            ),
            softmax_lanes=_coerce_int(fabric_dict.get("softmax_lanes", 1), "cim.fabric.softmax_lanes", min_value=1),
            softmax_pipeline_depth=_coerce_int(
                fabric_dict.get("softmax_pipeline_depth", 20),
                "cim.fabric.softmax_pipeline_depth",
                min_value=1,
            ),
        )


@dataclass
class CIMChipConfig:
    """Chip-level mapping parameters (cim.chip)."""

    arrays_per_chip: int  # capacity limit for validation; 0 = unchecked
    #: Uniform layers per chip (int), an explicit per-chip list, or "auto"
    #: (derive the chip count and per-chip layer split greedily from
    #: arrays_per_chip; the mapping becomes an output, not an input).
    layers_per_chip: Union[int, Tuple[int, ...], str]
    #: Chips each MoE layer's routed experts spread over (parallel links);
    #: 1 keeps every routed expert of a layer on that layer's chip.
    moe_expert_parallel: int = 1

    @classmethod
    def from_dict(cls, chip_dict: Optional[Dict[str, object]]) -> "CIMChipConfig":
        chip_dict = _require_mapping("cim.chip", chip_dict)
        arrays_per_chip = _coerce_int(
            chip_dict.get("arrays_per_chip", 0), "cim.chip.arrays_per_chip", min_value=0
        )
        layers_raw = _require_field("cim.chip", chip_dict, "layers_per_chip")
        if isinstance(layers_raw, str):
            if layers_raw.strip().lower() != "auto":
                raise ValueError(
                    "cim.chip.layers_per_chip must be an integer, a list of integers, or 'auto' "
                    f"(got {layers_raw!r})"
                )
            if arrays_per_chip <= 0:
                raise ValueError(
                    "cim.chip.layers_per_chip: 'auto' derives the per-chip layer split from the "
                    "array capacity, which requires cim.chip.arrays_per_chip > 0 "
                    f"(got arrays_per_chip={arrays_per_chip})."
                )
            layers_per_chip: Union[int, Tuple[int, ...], str] = "auto"
        elif isinstance(layers_raw, (list, tuple)):
            if not layers_raw:
                raise ValueError("cim.chip.layers_per_chip list must not be empty")
            layers_per_chip = tuple(
                _coerce_int(item, "cim.chip.layers_per_chip entries", min_value=1) for item in layers_raw
            )
        else:
            layers_per_chip = _coerce_int(layers_raw, "cim.chip.layers_per_chip", min_value=1)
        moe_expert_parallel = _coerce_int(
            chip_dict.get("moe_expert_parallel", 1), "cim.chip.moe_expert_parallel", min_value=1
        )
        return cls(
            arrays_per_chip=arrays_per_chip,
            layers_per_chip=layers_per_chip,
            moe_expert_parallel=moe_expert_parallel,
        )


@dataclass
class CIMKvDramConfig:
    """Optional per-device KV-cache DRAM tier (cim.kv_dram).

    Required iff inference.kvcache_type is 'cim_dram'; ignored (with a
    warning) under 'cim_sram'. All values are plain numbers (no unit
    strings — see convert()).
    """

    capacity_bytes: float
    bandwidth_bytes_per_s: float
    energy_per_bit_pj: float  # 0 reports zero KV DRAM energy

    @classmethod
    def from_dict(cls, kv_dict: Optional[Dict[str, object]]) -> "CIMKvDramConfig":
        kv_dict = _require_mapping("cim.kv_dram", kv_dict)
        return cls(
            capacity_bytes=_parse_cim_float(kv_dict, "cim.kv_dram", "capacity_bytes", strict=True),
            bandwidth_bytes_per_s=_parse_cim_float(
                kv_dict, "cim.kv_dram", "bandwidth_bytes_per_s", strict=True
            ),
            energy_per_bit_pj=_parse_cim_float(kv_dict, "cim.kv_dram", "energy_per_bit_pj", default=0.0),
        )


@dataclass
class CIMDseVariant:
    """One analog array design point for the DSE sweep (a cim.dse.variants entry).

    rows / slice_cycles / analog_clock_mhz default to None, meaning "inherit
    the corresponding cim.analog value".
    """

    adc_mux: int
    cols_adc: int
    energy_per_vec_pj: float
    area_mm2_per_array: float
    rows: Optional[int] = None
    slice_cycles: Optional[int] = None
    analog_clock_mhz: Optional[float] = None

    @classmethod
    def from_dict(cls, variant_dict: Optional[Dict[str, object]], index: int) -> "CIMDseVariant":
        context = f"cim.dse.variants[{index}]"
        variant_dict = _require_mapping(context, variant_dict)
        rows_raw = variant_dict.get("rows")
        slice_raw = variant_dict.get("slice_cycles")
        clock_present = "analog_clock_mhz" in variant_dict
        return cls(
            adc_mux=_parse_int_field(context, variant_dict, "adc_mux"),
            cols_adc=_parse_int_field(context, variant_dict, "cols_adc"),
            energy_per_vec_pj=_parse_cim_float(variant_dict, context, "energy_per_vec_pj", default=0.0),
            area_mm2_per_array=_parse_cim_float(variant_dict, context, "area_mm2_per_array", default=0.0),
            rows=None if rows_raw is None else _coerce_int(rows_raw, f"{context}.rows"),
            slice_cycles=None if slice_raw is None else _coerce_int(slice_raw, f"{context}.slice_cycles"),
            analog_clock_mhz=(
                _parse_cim_float(variant_dict, context, "analog_clock_mhz", strict=True)
                if clock_present
                else None
            ),
        )


@dataclass
class CIMDseConfig:
    """Parse-only `cim.dse` block: the candidate space for the DSE tool.

    Validated structurally here; consumed only by tools/fws_cim_dse.py
    (pass 2B). The simulator itself never reads it.
    """

    #: Cross-check only, never a knob: when set, the DSE tool requires every
    #: variant's adc_mux to appear here (typo guard); the sweep itself
    #: enumerates variants.
    mux_candidates: Tuple[int, ...]
    variants: Tuple[CIMDseVariant, ...]
    #: "auto" = divisors of the model's num_heads; or an explicit list.
    tp_candidates: Union[str, Tuple[int, ...]]
    max_chips: int  # 0 = unbounded
    #: MoE expert-spreading candidates (cim.chip.moe_expert_parallel values
    #: the DSE sweeps); the knob applies only when the model is MoE
    #: (DESIGN2 section 5). Default: no spreading.
    moe_expert_parallel: Tuple[int, ...] = (1,)

    @classmethod
    def from_dict(cls, dse_dict: Optional[Dict[str, object]]) -> "CIMDseConfig":
        dse_dict = _require_mapping("cim.dse", dse_dict)
        mux_raw = dse_dict.get("mux_candidates", ())
        if not isinstance(mux_raw, (list, tuple)):
            raise ValueError(
                f"cim.dse.mux_candidates must be a list of integers (got {mux_raw!r})"
            )
        mux_candidates = tuple(
            _coerce_int(item, "cim.dse.mux_candidates entries") for item in mux_raw
        )
        variants_raw = dse_dict.get("variants", ())
        if not isinstance(variants_raw, (list, tuple)):
            raise ValueError(
                f"cim.dse.variants must be a list of array-point mappings (got {variants_raw!r})"
            )
        variants = tuple(
            CIMDseVariant.from_dict(item, index) for index, item in enumerate(variants_raw)
        )
        tp_raw = dse_dict.get("tp_candidates", "auto")
        if isinstance(tp_raw, str):
            if tp_raw.strip().lower() != "auto":
                raise ValueError(
                    f"cim.dse.tp_candidates must be 'auto' or a list of integers (got {tp_raw!r})"
                )
            tp_candidates: Union[str, Tuple[int, ...]] = "auto"
        elif isinstance(tp_raw, (list, tuple)):
            tp_candidates = tuple(
                _coerce_int(item, "cim.dse.tp_candidates entries") for item in tp_raw
            )
        else:
            raise ValueError(
                f"cim.dse.tp_candidates must be 'auto' or a list of integers (got {tp_raw!r})"
            )
        moe_ep_raw = dse_dict.get("moe_expert_parallel", (1,))
        if not isinstance(moe_ep_raw, (list, tuple)):
            raise ValueError(
                "cim.dse.moe_expert_parallel must be a list of integers >= 1 "
                f"(got {moe_ep_raw!r})"
            )
        moe_expert_parallel = tuple(
            _coerce_int(item, "cim.dse.moe_expert_parallel entries", min_value=1)
            for item in moe_ep_raw
        ) or (1,)
        return cls(
            mux_candidates=mux_candidates,
            variants=variants,
            tp_candidates=tp_candidates,
            max_chips=_coerce_int(dse_dict.get("max_chips", 0), "cim.dse.max_chips", min_value=0),
            moe_expert_parallel=moe_expert_parallel,
        )


# ---------------------------------------------------------------------------
# Device cards (QIF P2.1 — D14, ADJ-4)
#
# A card is a parameter set plus STRUCTURAL KNOBS: the parameters set the
# numbers, the knobs change the shape of a law. `cim.analog` and `cim.fabric`
# already carry the shipped cards' parameter halves, so the card block
# PROMOTES them instead of replacing them: when `cim.cards` is absent the
# library is synthesized from those two blocks with inert knob defaults, and
# every shipped YAML parses unchanged and prices identically.
# ---------------------------------------------------------------------------

#: Device families the analog-macro card admits. `ctt` is the populated card;
#: `reram` / `mram` are NAMED EMPTY SLOTS (ADJ-4) — the schema exists and no
#: parameters are shipped, so a card on those families must supply its own
#: `params` block in full. Nothing is ever inherited into an empty slot.
#:
#: `custom` is the ESCAPE HATCH, and it is explicit on purpose. D14 says device
#: cards, not device rewrites — a family the list has never heard of is a real
#: card the tool should be able to price. But an OPEN field cannot tell a new
#: device from a typo, and silently admitting `crt` as a device would be the
#: worse failure. So a caller declares `device: custom` deliberately, and pays
#: for it: a custom card inherits NOTHING and must state every parameter itself,
#: exactly like an empty slot. Anything else is still refused by name.
CIM_CUSTOM_DEVICE_FAMILY = "custom"
CIM_DEVICE_FAMILIES = ("ctt", "reram", "mram", CIM_CUSTOM_DEVICE_FAMILY)
CIM_EMPTY_CARD_SLOTS = ("reram", "mram")
#: Where the bit slices of one weight word live (P2.4).
CIM_SLICING_ARRANGEMENTS = ("column_sets", "chained_macros")
CIM_CARD_KINDS = ("analog_macro", "digital_chiplet")


@dataclass(frozen=True)
class CIMCardValidityPoint:
    """One admitted (bits_per_cell, weight_bits, mux) point of a card menu.

    The menu is the honesty device: a device does not support every
    bits-per-cell, so the tool REFUSES an unlisted point instead of
    interpolating one.
    """

    bits_per_cell: int
    weight_bits: int
    mux: int

    @classmethod
    def from_dict(cls, point_dict: object, context: str) -> "CIMCardValidityPoint":
        point_dict = _require_mapping(context, point_dict)
        return cls(
            bits_per_cell=_parse_int_field(context, point_dict, "bits_per_cell"),
            weight_bits=_parse_int_field(context, point_dict, "weight_bits"),
            mux=_parse_int_field(context, point_dict, "mux"),
        )

    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.bits_per_cell, self.weight_bits, self.mux)


@dataclass
class CIMAnalogCardConfig:
    """A named analog-macro device card: `cim.analog` plus structural knobs.

    Knob semantics (each one changes a law, not just a constant):
      * bits_per_cell / weight_bits — bit slicing is present IFF
        ``bits_per_cell < weight_bits`` (ADJ-4). Both 0 means the card
        declares no cell width and slicing is off (``n_slices == 1``).
      * slicing — where the slices of one weight word live.
      * bank_depth — mux slots per allocatable bank, i.e. the allocation
        granularity the card admits. Default ``adc_mux`` = the whole macro,
        which is today's dedicated-per-matrix behavior.
      * stack_3d_height — divides the FOOTPRINT a macro occupies on the
        package. It never divides silicon area or energy.
      * validity — the admitted (bits_per_cell, weight_bits, mux) menu. An
        empty menu means the card declares none, and nothing is refused.
      * pool_* — per-macro digital pool cost knobs (D12, P2.5). 0 means the
        card declares no number and the pool term reports zero, exactly as
        ``area_mm2_per_array: 0`` disables area reporting.
    """

    name: str
    device: str
    params: CIMAnalogConfig
    bits_per_cell: int = 0
    weight_bits: int = 0
    slicing: str = "column_sets"
    bank_depth: int = 0          # 0 -> params.adc_mux (the whole macro)
    stack_3d_height: int = 1
    validity: Tuple[CIMCardValidityPoint, ...] = ()
    pool_clock_ghz: float = 0.0        # 0 -> inherit the digital card's clock
    pool_energy_per_add_pj: float = 0.0
    pool_area_mm2_per_adder: float = 0.0
    #: Analog cycles lost switching from one active column set to the next.
    #: DEFAULT 0 — no bank-switch cost is shipped for any device, which is a
    #: DISCLOSED relaxation (AUDIT finding 3: OPTIMA priced switching at zero
    #: silently; this states it). A card that knows its number declares it.
    bank_switch_cycles: int = 0

    @property
    def n_slices(self) -> int:
        """Bit slices per weight word: ceil(weight_bits / bits_per_cell)."""
        if self.bits_per_cell <= 0 or self.weight_bits <= 0:
            return 1
        if self.bits_per_cell >= self.weight_bits:
            return 1
        return -(-int(self.weight_bits) // int(self.bits_per_cell))

    @property
    def slicing_enabled(self) -> bool:
        """Slicing is present iff bits_per_cell < weight_bits (ADJ-4)."""
        return self.n_slices > 1

    @property
    def column_sets_per_macro(self) -> int:
        """Column sets (mux slots) one macro carries — the ADC passes it owns."""
        return int(self.params.adc_mux)

    @property
    def stored_columns_per_set(self) -> int:
        """Stored weight columns in one column set."""
        return int(self.params.cols_adc)

    @property
    def allocation_granularity(self) -> int:
        """Mux slots in one allocatable bank (bank_depth, or the whole macro)."""
        return self.bank_depth if self.bank_depth > 0 else self.column_sets_per_macro

    def admits(self, bits_per_cell: int, weight_bits: int, mux: int) -> bool:
        """True when the point is on the card's menu (or no menu is declared)."""
        if not self.validity:
            return True
        return (int(bits_per_cell), int(weight_bits), int(mux)) in {
            point.as_tuple() for point in self.validity
        }

    @classmethod
    def synthesized(cls, analog: CIMAnalogConfig) -> "CIMAnalogCardConfig":
        """The card implied by a `cim.analog` block with no `cim.cards`.

        Every knob is inert: no slicing, whole-macro allocation, no stacking,
        no menu, no declared pool costs.
        """
        return cls(name="default", device="ctt", params=analog)

    @classmethod
    def from_dict(
        cls,
        card_dict: Dict[str, object],
        name: str,
        analog_default: Optional[CIMAnalogConfig],
    ) -> "CIMAnalogCardConfig":
        context = f"cim.cards.{name}"
        device = str(card_dict.get("device", "ctt")).strip().lower()
        if device not in CIM_DEVICE_FAMILIES:
            raise ValueError(
                f"{context}.device must be one of {list(CIM_DEVICE_FAMILIES)} "
                f"(got {card_dict.get('device')!r}). An unlisted family is refused so a "
                "typo cannot become a device; a genuinely new device declares "
                "device: 'custom' and supplies a complete 'params' block."
            )
        params_dict = card_dict.get("params")
        if params_dict is None:
            if device == CIM_CUSTOM_DEVICE_FAMILY:
                raise ValueError(
                    f"{context}: device 'custom' names a device this tool ships no "
                    "parameters for, so the card must supply a complete 'params' block "
                    "of its own. Nothing is inherited into a custom card — inheriting "
                    "CTT numbers under another device's name is exactly the silent "
                    "mis-pricing the family list exists to prevent."
                )
            if device in CIM_EMPTY_CARD_SLOTS:
                raise ValueError(
                    f"{context}: device '{device}' is a NAMED EMPTY CARD SLOT — the schema "
                    "exists but no parameters are shipped for it, so the card must supply a "
                    "complete 'params' block of its own. Nothing is inherited into an empty slot."
                )
            if analog_default is None:
                raise ValueError(
                    f"{context}: no 'params' block and no cim.analog block to inherit from."
                )
            params = analog_default
        else:
            params = CIMAnalogConfig.from_dict(_require_mapping(f"{context}.params", params_dict))
        bits_per_cell = _coerce_int(
            card_dict.get("bits_per_cell", 0), f"{context}.bits_per_cell", min_value=0
        )
        weight_bits = _coerce_int(
            card_dict.get("weight_bits", 0), f"{context}.weight_bits", min_value=0
        )
        if (bits_per_cell > 0) != (weight_bits > 0):
            raise ValueError(
                f"{context}: bits_per_cell and weight_bits must be declared together "
                f"(got bits_per_cell={bits_per_cell}, weight_bits={weight_bits}); slicing is "
                "present iff bits_per_cell < weight_bits."
            )
        slicing = str(card_dict.get("slicing", "column_sets")).strip().lower()
        if slicing not in CIM_SLICING_ARRANGEMENTS:
            raise ValueError(
                f"{context}.slicing must be one of {list(CIM_SLICING_ARRANGEMENTS)} "
                f"(got {card_dict.get('slicing')!r})"
            )
        bank_depth = _coerce_int(
            card_dict.get("bank_depth", 0), f"{context}.bank_depth", min_value=0
        )
        if bank_depth > 0 and int(params.adc_mux) % bank_depth != 0:
            raise ValueError(
                f"{context}.bank_depth = {bank_depth} must divide the card's adc_mux "
                f"= {params.adc_mux}: a bank is a whole number of mux slots and the mux "
                "slot is the smallest allocatable unit."
            )
        stack_3d_height = _coerce_int(
            card_dict.get("stack_3d_height", 1), f"{context}.stack_3d_height", min_value=1
        )
        validity_raw = card_dict.get("validity", ())
        if not isinstance(validity_raw, (list, tuple)):
            raise ValueError(
                f"{context}.validity must be a list of "
                f"(bits_per_cell, weight_bits, mux) mappings (got {validity_raw!r})"
            )
        validity = tuple(
            CIMCardValidityPoint.from_dict(item, f"{context}.validity[{index}]")
            for index, item in enumerate(validity_raw)
        )
        card = cls(
            name=name,
            device=device,
            params=params,
            bits_per_cell=bits_per_cell,
            weight_bits=weight_bits,
            slicing=slicing,
            bank_depth=bank_depth,
            stack_3d_height=stack_3d_height,
            validity=validity,
            pool_clock_ghz=_parse_cim_float(
                card_dict, context, "pool_clock_ghz", default=0.0
            ),
            pool_energy_per_add_pj=_parse_cim_float(
                card_dict, context, "pool_energy_per_add_pj", default=0.0
            ),
            pool_area_mm2_per_adder=_parse_cim_float(
                card_dict, context, "pool_area_mm2_per_adder", default=0.0
            ),
            bank_switch_cycles=_coerce_int(
                card_dict.get("bank_switch_cycles", 0),
                f"{context}.bank_switch_cycles",
                min_value=0,
            ),
        )
        if validity and not card.admits(bits_per_cell, weight_bits, int(params.adc_mux)):
            raise ValueError(
                f"{context}: the card's own point (bits_per_cell={bits_per_cell}, "
                f"weight_bits={weight_bits}, mux={params.adc_mux}) is not on its validity "
                "menu. The menu lists what the device admits; the tool refuses an unlisted "
                "point instead of interpolating one."
            )
        return card


@dataclass
class CIMDigitalChipletCardConfig:
    """A named SHARED DIGITAL CHIPLET card: `cim.fabric` plus its cost knobs.

    The attention (systolic-array) and softmax-lane laws are this card's laws
    (D13); `cim_timing.CimDeviceModel` reads them from the same
    :class:`CIMFabricConfig` object the card wraps, so promoting the block to
    a card changes no number. `area_mm2` is the chiplet's own silicon, reported
    by `CimDeviceModel.shared_digital_area_mm2` and added to
    `CimDeviceModel.system_area_mm2`; it defaults to 0, which reports zero
    exactly as `area_mm2_per_array: 0` does. There is no energy knob: no law
    prices a fabric op's energy yet, and a cost knob nothing reads is worse
    than a missing one.

    The ENGINE CAPABILITY knobs below (QIF P2.6) describe the chiplet's
    VECTOR/SCAN engine — the unit the SSM scan, delta-rule and RG-LRU laws
    are timed on, as distinct from the systolic array that runs attention.
    """

    name: str
    fabric: CIMFabricConfig
    area_mm2: float = 0.0
    #: --- ENGINE CAPABILITY knobs (QIF P2.6, digital op laws) ---------------
    #: The chiplet's VECTOR/SCAN engine: the unit every non-attention digital
    #: op law (SSD scan, selective scan, delta rule, RG-LRU) is timed on. One
    #: lane retires ONE scalar arithmetic operation (a multiply OR an add) per
    #: vector cycle — the peak is `vector_lanes * vector_clock`, and every law
    #: is bound by it by construction.
    #:
    #: `vector_lanes` has NO default on purpose (ADJ-4, no invented numbers).
    #: The systolic array's rows x cols is a MATMUL engine, not a scan engine,
    #: and `softmax_lanes` is a softmax pipeline, not a general vector unit —
    #: deriving scan lanes from either would invent silicon. A card that does
    #: not declare it makes `cim_timing.EngineCapabilityError` the answer to
    #: every scan/delta-rule pricing call, which is the honest answer.
    vector_lanes: int = 0
    #: 0 -> inherit `fabric.clock_ghz`. Honest: the vector engine sits on THIS
    #: chiplet, and the chiplet declares exactly one clock.
    vector_clock_ghz: float = 0.0
    #: 0 -> inherit `fabric.softmax_pipeline_depth`. Honest by the same rule:
    #: it is the only elementwise-pipeline depth the chiplet declares, and a
    #: scan lane is the same class of unit as a softmax lane. DISCLOSED by
    #: `cim_timing.CimDeviceModel.vector_engine_disclosures`.
    vector_pipeline_depth: int = 0
    #: Recurrent-state bytes the engine can read+write per vector cycle.
    #: 0 = UNDECLARED: the state traffic is REPORTED by every scan law and
    #: BOUNDS nothing, which is a disclosed relaxation, not a silent zero
    #: (AUDIT finding 3 is the precedent for saying so out loud).
    state_bytes_per_cycle: float = 0.0

    @property
    def has_vector_engine(self) -> bool:
        """True when the card declares a vector/scan engine at all."""
        return int(self.vector_lanes) > 0

    @property
    def vector_clock_ghz_effective(self) -> float:
        """Vector-engine clock: the declared one, else the chiplet's clock."""
        return float(self.vector_clock_ghz) if self.vector_clock_ghz > 0 else float(self.fabric.clock_ghz)

    @property
    def vector_pipeline_depth_effective(self) -> int:
        """Vector-engine fill depth: the declared one, else the softmax depth."""
        if self.vector_pipeline_depth > 0:
            return int(self.vector_pipeline_depth)
        return int(self.fabric.softmax_pipeline_depth)

    @classmethod
    def synthesized(cls, fabric: CIMFabricConfig) -> "CIMDigitalChipletCardConfig":
        return cls(name="default", fabric=fabric)

    @classmethod
    def from_dict(
        cls,
        card_dict: Dict[str, object],
        name: str,
        fabric_default: Optional[CIMFabricConfig],
    ) -> "CIMDigitalChipletCardConfig":
        context = f"cim.cards.{name}"
        if "energy_per_op_pj" in card_dict:
            raise ValueError(
                f"{context}.energy_per_op_pj is not a card field: no law prices a shared "
                "digital chiplet op's energy yet, and the schema does not carry a cost knob "
                "nothing reads. Fabric energy lands with P4 (evaluation)."
            )
        params_dict = card_dict.get("params")
        if params_dict is None:
            if fabric_default is None:
                raise ValueError(
                    f"{context}: no 'params' block and no cim.fabric block to inherit from."
                )
            fabric = fabric_default
        else:
            fabric = CIMFabricConfig.from_dict(_require_mapping(f"{context}.params", params_dict))
        return cls(
            name=name,
            fabric=fabric,
            area_mm2=_parse_cim_float(card_dict, context, "area_mm2", default=0.0),
            vector_lanes=_coerce_int(
                card_dict.get("vector_lanes", 0), f"{context}.vector_lanes", min_value=0
            ),
            vector_clock_ghz=_parse_cim_float(
                card_dict, context, "vector_clock_ghz", default=0.0
            ),
            vector_pipeline_depth=_coerce_int(
                card_dict.get("vector_pipeline_depth", 0),
                f"{context}.vector_pipeline_depth",
                min_value=0,
            ),
            state_bytes_per_cycle=_parse_cim_float(
                card_dict, context, "state_bytes_per_cycle", default=0.0
            ),
        )


@dataclass
class CIMCardLibrary:
    """The parsed `cim.cards` block, or the library `cim.analog`/`cim.fabric` imply.

    `default_analog` / `default_digital` name the cards the device model uses;
    a synthesized library names both "default".
    """

    analog: Dict[str, CIMAnalogCardConfig]
    digital: Dict[str, CIMDigitalChipletCardConfig]
    default_analog: str
    default_digital: str

    @property
    def analog_card(self) -> CIMAnalogCardConfig:
        return self.analog[self.default_analog]

    @property
    def digital_card(self) -> CIMDigitalChipletCardConfig:
        return self.digital[self.default_digital]

    @classmethod
    def synthesized(
        cls, analog: CIMAnalogConfig, fabric: CIMFabricConfig
    ) -> "CIMCardLibrary":
        return cls(
            analog={"default": CIMAnalogCardConfig.synthesized(analog)},
            digital={"default": CIMDigitalChipletCardConfig.synthesized(fabric)},
            default_analog="default",
            default_digital="default",
        )

    @classmethod
    def from_dict(
        cls,
        cards_dict: Optional[Dict[str, object]],
        analog_default: Optional[CIMAnalogConfig],
        fabric_default: Optional[CIMFabricConfig],
    ) -> "CIMCardLibrary":
        cards_dict = _require_mapping("cim.cards", cards_dict)
        analog_cards: Dict[str, CIMAnalogCardConfig] = {}
        digital_cards: Dict[str, CIMDigitalChipletCardConfig] = {}
        default_analog = None
        default_digital = None
        for name, entry in cards_dict.items():
            if name in ("default_analog", "default_digital"):
                continue
            entry = _require_mapping(f"cim.cards.{name}", entry)
            kind = str(entry.get("kind", "analog_macro")).strip().lower()
            if kind == "analog_macro":
                analog_cards[str(name)] = CIMAnalogCardConfig.from_dict(
                    entry, str(name), analog_default
                )
            elif kind == "digital_chiplet":
                digital_cards[str(name)] = CIMDigitalChipletCardConfig.from_dict(
                    entry, str(name), fabric_default
                )
            else:
                raise ValueError(
                    f"cim.cards.{name}.kind must be one of {list(CIM_CARD_KINDS)} "
                    f"(got {entry.get('kind')!r})"
                )
        if not analog_cards:
            if analog_default is None:
                raise ValueError(
                    "cim.cards declares no analog_macro card and there is no cim.analog "
                    "block to synthesize one from."
                )
            analog_cards["default"] = CIMAnalogCardConfig.synthesized(analog_default)
        if not digital_cards:
            if fabric_default is None:
                raise ValueError(
                    "cim.cards declares no digital_chiplet card and there is no cim.fabric "
                    "block to synthesize one from."
                )
            digital_cards["default"] = CIMDigitalChipletCardConfig.synthesized(fabric_default)
        default_analog = str(cards_dict.get("default_analog", next(iter(analog_cards))))
        default_digital = str(cards_dict.get("default_digital", next(iter(digital_cards))))
        if default_analog not in analog_cards:
            raise ValueError(
                f"cim.cards.default_analog = {default_analog!r} names no analog_macro card "
                f"(have {list(analog_cards)})"
            )
        if default_digital not in digital_cards:
            raise ValueError(
                f"cim.cards.default_digital = {default_digital!r} names no digital_chiplet "
                f"card (have {list(digital_cards)})"
            )
        return cls(
            analog=analog_cards,
            digital=digital_cards,
            default_analog=default_analog,
            default_digital=default_digital,
        )


@dataclass(frozen=True)
class CIMTileAssignment:
    """One user-written tile placement (a `cim.allocation.assignments` entry).

    The owner half names the tile (D21: every tile has a named owner); the
    site half places it. `column_sets` are mux-slot ids inside `macro`.
    """

    model: str
    layer: int
    op: str
    expert: int
    shard: int
    #: The bit slice this entry places, or None when the entry does not say.
    #: DECLARED means CHECKED: entries for one owner are consumed in declaration
    #: order against the enumerator's block order, so a declared slice_index is
    #: verified against the tile the entry lands on and a mismatch is refused by
    #: name (fws_mapping._apply_user_allocation). It was parsed and then dropped
    #: on the floor before, which let a user write slice 3 and silently place
    #: slice 0. Absent is the unconstrained default; it was never a claim of
    #: slice 0, which is why the default is None and not 0.
    slice_index: Optional[int]
    macro: int
    column_sets: Tuple[int, ...]

    @classmethod
    def from_dict(cls, entry: object, index: int) -> "CIMTileAssignment":
        context = f"cim.allocation.assignments[{index}]"
        entry = _require_mapping(context, entry)
        sets_raw = _require_field(context, entry, "column_sets")
        if not isinstance(sets_raw, (list, tuple)) or not sets_raw:
            raise ValueError(
                f"{context}.column_sets must be a non-empty list of mux-slot ids "
                f"(got {sets_raw!r})"
            )
        column_sets = tuple(
            _coerce_int(item, f"{context}.column_sets entries", min_value=0) for item in sets_raw
        )
        if len(set(column_sets)) != len(column_sets):
            raise ValueError(f"{context}.column_sets repeats a mux-slot id: {list(column_sets)}")
        return cls(
            model=str(entry.get("model", "")),
            layer=_coerce_int(entry.get("layer", 0), f"{context}.layer", min_value=0),
            op=str(_require_field(context, entry, "op")),
            expert=_coerce_int(entry.get("expert", -1), f"{context}.expert", min_value=-1),
            shard=_coerce_int(entry.get("shard", 0), f"{context}.shard", min_value=0),
            slice_index=(
                None
                if entry.get("slice_index") is None
                else _coerce_int(entry["slice_index"], f"{context}.slice_index", min_value=0)
            ),
            macro=_coerce_int(_require_field(context, entry, "macro"), f"{context}.macro", min_value=0),
            column_sets=tuple(sorted(column_sets)),
        )


@dataclass
class CIMAllocationConfig:
    """The optional `cim.allocation` block: user-specified tile allocation (D10).

    ABSENT means today's dedicated-per-matrix behavior, bit-identically: one
    owner per macro, every mux slot of the macro activated by that owner.
    """

    #: Mux slots per tile; None inherits the card's bank_depth.
    column_sets_per_tile: Optional[int] = None
    assignments: Tuple[CIMTileAssignment, ...] = ()

    @classmethod
    def from_dict(cls, alloc_dict: Optional[Dict[str, object]]) -> "CIMAllocationConfig":
        alloc_dict = _require_mapping("cim.allocation", alloc_dict)
        raw = alloc_dict.get("column_sets_per_tile")
        column_sets_per_tile = (
            None
            if raw is None
            else _coerce_int(raw, "cim.allocation.column_sets_per_tile", min_value=1)
        )
        assignments_raw = alloc_dict.get("assignments", ())
        if not isinstance(assignments_raw, (list, tuple)):
            raise ValueError(
                "cim.allocation.assignments must be a list of tile placements "
                f"(got {assignments_raw!r})"
            )
        return cls(
            column_sets_per_tile=column_sets_per_tile,
            assignments=tuple(
                CIMTileAssignment.from_dict(item, index)
                for index, item in enumerate(assignments_raw)
            ),
        )


class MissingCardLibraryError(AttributeError, ValueError):
    """Asked for the active device card on a :class:`CIMConfig` with no library.

    ``CIMConfig.from_dict`` ALWAYS builds one (synthesized from `cim.analog` +
    `cim.fabric` when `cim.cards` is absent), so this can only be reached by
    constructing ``CIMConfig`` directly and leaving ``cards=None`` — a test
    double or a caller assembling the dataclass by hand. It used to surface as
    a bare ``AttributeError: 'NoneType' object has no attribute 'analog_card'``.

    It subclasses AttributeError as well as ValueError so the one `getattr`
    probe on the card path (fws_atlas_export's optional area) keeps its default
    instead of turning a hand-built config into a traceback.
    """


@dataclass
class CIMConfig:
    """Top-level `cim:` block for device_class: fws_cim hardware configs."""

    analog: CIMAnalogConfig
    fabric: CIMFabricConfig
    chip: CIMChipConfig
    #: Optional KV DRAM tier; required iff inference.kvcache_type == cim_dram.
    kv_dram: Optional[CIMKvDramConfig] = None
    #: Optional DSE candidate space; parse-only (tools/fws_cim_dse.py).
    dse: Optional[CIMDseConfig] = None
    #: Device-card library (P2.1). Always present: synthesized from analog +
    #: fabric when `cim.cards` is absent, so shipped YAMLs price identically.
    cards: Optional[CIMCardLibrary] = None
    #: User-specified tile allocation (P2.2, D10). None = dedicated per matrix.
    allocation: Optional[CIMAllocationConfig] = None

    def _require_cards(self, which: str) -> CIMCardLibrary:
        if self.cards is None:
            raise MissingCardLibraryError(
                f"cim.{which} was asked for, but this CIMConfig carries no card library "
                "(cards=None). CIMConfig.from_dict always builds one — synthesizing it "
                "from cim.analog and cim.fabric when the cim.cards block is absent — so "
                "a None library means the object was assembled by hand; pass "
                "CIMCardLibrary.synthesized(analog, fabric)."
            )
        return self.cards

    @property
    def analog_card(self) -> CIMAnalogCardConfig:
        """The active analog-macro card (the synthesized one when none is named)."""
        return self._require_cards("analog_card").analog_card

    @property
    def digital_card(self) -> CIMDigitalChipletCardConfig:
        """The active shared-digital-chiplet card."""
        return self._require_cards("digital_card").digital_card

    @classmethod
    def from_dict(cls, cim_dict: Optional[Dict[str, object]]) -> "CIMConfig":
        cim_dict = _require_mapping("cim", cim_dict)
        kv_dram_dict = cim_dict.get("kv_dram")
        dse_dict = cim_dict.get("dse")
        cards_dict = cim_dict.get("cards")
        allocation_dict = cim_dict.get("allocation")
        analog = CIMAnalogConfig.from_dict(_require_field("cim", cim_dict, "analog"))
        fabric = CIMFabricConfig.from_dict(_require_field("cim", cim_dict, "fabric"))
        cards = (
            CIMCardLibrary.synthesized(analog, fabric)
            if cards_dict is None
            else CIMCardLibrary.from_dict(cards_dict, analog, fabric)
        )
        allocation = (
            None if allocation_dict is None else CIMAllocationConfig.from_dict(allocation_dict)
        )
        if allocation is not None and allocation.column_sets_per_tile is not None:
            mux = int(cards.analog_card.params.adc_mux)
            if mux % allocation.column_sets_per_tile != 0:
                raise ValueError(
                    f"cim.allocation.column_sets_per_tile = {allocation.column_sets_per_tile} "
                    f"must divide the active card's adc_mux = {mux}: the mux slot is the "
                    "smallest allocatable unit."
                )
        if allocation is not None and allocation.assignments:
            # A user-written allocation is priced AND validated by the tool
            # (D10), so the capacity check runs where the block is parsed —
            # not only when a test reaches for the device model.
            import cim_timing as _cim_timing

            _cim_timing.check_allocation_capacity(
                int(cards.analog_card.column_sets_per_macro), allocation.assignments
            )
        return cls(
            analog=analog,
            fabric=fabric,
            chip=CIMChipConfig.from_dict(_require_field("cim", cim_dict, "chip")),
            kv_dram=None if kv_dram_dict is None else CIMKvDramConfig.from_dict(kv_dram_dict),
            dse=None if dse_dict is None else CIMDseConfig.from_dict(dse_dict),
            cards=cards,
            allocation=allocation,
        )



# ---------------------------------------------------------------------------
# The `mapping:` block (QIF P3.1, ADJ-5)
#
# A NEW TOP-LEVEL BLOCK, not an extension of `cim.chip`: the concept widened
# from "how many layers sit on a chip" to "which macros, chips, shard groups
# and devices exist and who owns them", and a new name is the honest way to
# say so (P3 plan-page open question 5, adjudicated by ADJ-5).
#
# ABSENT means the DERIVED DEDICATED MAPPING: chips, macros and layer
# assignment reproduce today's `cim.chip.layers_per_chip` semantics exactly.
# Present means the user DECLARES the placement and the tool validates it
# (D10); every refusal names the offending setting.
# ---------------------------------------------------------------------------


#: Parallelism axes a mapping annotates. `cp` is deliberately absent: P3
#: annotates tp / ep / pp only (D20), and an axis nothing colors is an axis
#: nothing checks.
MAPPING_AXES: Tuple[str, ...] = ("tp", "ep", "pp")


@dataclass(frozen=True)
class MappingParallelism:
    """Declared tp / ep / pp degrees. ``None`` = inherit the derived degree.

    ``tp`` and ``ep`` must agree with the hardware settings that already carry
    them (`parallelism.tp`, `cim.chip.moe_expert_parallel`) — two spellings of
    one degree is two accountings of one number (D21), so a disagreement is a
    named error rather than a precedence rule.

    ``pp`` is the exception and it is deliberate: ADJ-5 makes pp an INDEPENDENT
    annotation over chips, so a mapping may declare `pp > 1` while the hardware
    keeps `parallelism.pp: 1` (which the fws_cim run path requires). The two
    are different axes that share a name; the mapping's pp never reaches the
    GPU rank grid.
    """

    tp: Optional[int] = None
    ep: Optional[int] = None
    pp: Optional[int] = None

    @classmethod
    def from_dict(cls, raw: object, context: str) -> "MappingParallelism":
        if raw is None:
            return cls()
        raw = _require_mapping(context, raw)
        _reject_unknown_keys(context, raw, MAPPING_AXES)
        values = {}
        for axis in MAPPING_AXES:
            if raw.get(axis) is None:
                values[axis] = None
            else:
                values[axis] = _coerce_int(raw[axis], f"{context}.{axis}", min_value=1)
        return cls(**values)


@dataclass(frozen=True)
class MappingSystemConfig:
    """One system's declared placement (the whole block, or one PD half).

    Every field is optional; an omitted field is DERIVED and the derivation is
    reported. A declared field is checked against the placement the tool
    builds, so the block is a claim the tool can refuse — never an input that
    silently wins over the machine it describes.
    """

    #: Total ANALOG chips. Declared, then checked against the enumerated
    #: placement: a chip count that disagrees with the chips actually built is
    #: a named error, and a non-integer never parses (D21: chips are integers).
    chips: Optional[int] = None
    #: Macro slots per analog chip. None inherits `cim.chip.arrays_per_chip`.
    macros_per_chip: Optional[int] = None
    #: Shared digital chiplets (D13). A CONFIG INPUT per ADJ-5; the derived
    #: suggestion is always reported next to the declared value.
    shared_chiplets: Optional[int] = None
    parallelism: MappingParallelism = field(default_factory=MappingParallelism)
    #: Layer -> chip assignment, `cim.chip.layers_per_chip` spelling (int, list
    #: or "auto"). None inherits the cim block, which is what makes an absent
    #: `mapping:` reproduce today's placement.
    layers_per_chip: Optional[Union[int, Tuple[int, ...], str]] = None
    #: Per-analog-chip axis indices, e.g. `{"pp": [0, 0, 1, 1]}`. An axis the
    #: user does not list is derived. This is the membership half of ADJ-5:
    #: chip index is NOT implicitly pp.
    membership: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    #: Decode steps the DAG builder lowers (ADJ-6's bounded window). None
    #: leaves the builder's default; the truncation is always disclosed.
    decode_window: Optional[int] = None

    _KEYS = (
        "chips",
        "macros_per_chip",
        "shared_chiplets",
        "parallelism",
        "layers_per_chip",
        "membership",
        "decode_window",
    )

    @classmethod
    def from_dict(cls, raw: object, context: str) -> "MappingSystemConfig":
        raw = _require_mapping(context, raw)
        _reject_unknown_keys(context, raw, cls._KEYS)
        layers_raw = raw.get("layers_per_chip")
        layers: Optional[Union[int, Tuple[int, ...], str]]
        if layers_raw is None:
            layers = None
        elif isinstance(layers_raw, str):
            if layers_raw.strip().lower() != "auto":
                raise ValueError(
                    f"{context}.layers_per_chip must be an integer, a list of integers "
                    f"or 'auto' (got {layers_raw!r})"
                )
            layers = "auto"
        elif isinstance(layers_raw, (list, tuple)):
            if not layers_raw:
                raise ValueError(f"{context}.layers_per_chip list must not be empty")
            layers = tuple(
                _coerce_int(item, f"{context}.layers_per_chip entries", min_value=1)
                for item in layers_raw
            )
        else:
            layers = _coerce_int(layers_raw, f"{context}.layers_per_chip", min_value=1)

        membership_raw = raw.get("membership")
        membership: Dict[str, Tuple[int, ...]] = {}
        if membership_raw is not None:
            membership_raw = _require_mapping(f"{context}.membership", membership_raw)
            _reject_unknown_keys(f"{context}.membership", membership_raw, MAPPING_AXES)
            for axis in MAPPING_AXES:
                entries = membership_raw.get(axis)
                if entries is None:
                    continue
                if not isinstance(entries, (list, tuple)) or not entries:
                    raise ValueError(
                        f"{context}.membership.{axis} must be a non-empty list of "
                        f"per-chip {axis} indices (got {entries!r})"
                    )
                membership[axis] = tuple(
                    _coerce_int(item, f"{context}.membership.{axis} entries", min_value=0)
                    for item in entries
                )
        return cls(
            chips=(
                None
                if raw.get("chips") is None
                else _coerce_int(raw["chips"], f"{context}.chips", min_value=1)
            ),
            macros_per_chip=(
                None
                if raw.get("macros_per_chip") is None
                else _coerce_int(
                    raw["macros_per_chip"], f"{context}.macros_per_chip", min_value=1
                )
            ),
            shared_chiplets=(
                None
                if raw.get("shared_chiplets") is None
                else _coerce_int(
                    raw["shared_chiplets"], f"{context}.shared_chiplets", min_value=0
                )
            ),
            parallelism=MappingParallelism.from_dict(
                raw.get("parallelism"), f"{context}.parallelism"
            ),
            layers_per_chip=layers,
            membership=membership,
            decode_window=(
                None
                if raw.get("decode_window") is None
                else _coerce_int(raw["decode_window"], f"{context}.decode_window", min_value=1)
            ),
        )


@dataclass(frozen=True)
class MappingConfig:
    """The parsed `mapping:` block (P3.1).

    Either one unified system, or a PD pair (D16): two separate inventories,
    two mappings, one handoff priced as bytes. Setting both halves equal
    reproduces the unified machine, which is a test, not a claim.
    """

    system: MappingSystemConfig = field(default_factory=MappingSystemConfig)
    prefill: Optional[MappingSystemConfig] = None
    decode: Optional[MappingSystemConfig] = None

    @property
    def is_pd(self) -> bool:
        return self.prefill is not None and self.decode is not None

    @classmethod
    def from_dict(cls, raw: object) -> "MappingConfig":
        context = "mapping"
        raw = _require_mapping(context, raw)
        _reject_unknown_keys(context, raw, MappingSystemConfig._KEYS + ("pd",))
        pd_raw = raw.get("pd")
        prefill = decode = None
        if pd_raw is not None:
            pd_raw = _require_mapping(f"{context}.pd", pd_raw)
            _reject_unknown_keys(f"{context}.pd", pd_raw, ("prefill", "decode"))
            missing = [half for half in ("prefill", "decode") if pd_raw.get(half) is None]
            if missing:
                raise ValueError(
                    f"{context}.pd requires BOTH prefill and decode (missing: "
                    f"{', '.join(missing)}). PD disaggregation is two inventories and "
                    "one handoff (D16); one half alone is not a machine."
                )
            prefill = MappingSystemConfig.from_dict(pd_raw["prefill"], f"{context}.pd.prefill")
            decode = MappingSystemConfig.from_dict(pd_raw["decode"], f"{context}.pd.decode")
        unified_keys = {k: v for k, v in raw.items() if k != "pd"}
        return cls(
            system=MappingSystemConfig.from_dict(unified_keys, context),
            prefill=prefill,
            decode=decode,
        )


@dataclass
class HWConfig:
    sw_config: SWConfig
    tech_config: TechConfig
    power_breakdown: PowerBreakdownConfig
    sch_config: SchedulingConfig
    area_breakdown: AreaBreakdownConfig
    perimeter_breakdown: PerimeterBreakdownConfig
    memory_hierarchy: MemoryHierarchyConfig
    network_layout: NetworkLayoutConfig
    execution_backend: ExecutionBackend
    inference_config: InferenceHWConfig
    #: Device class of the accelerator: "gpu" (default, existing behavior) or
    #: "fws_cim" (fixed-weight-stationary compute-in-memory).
    device_class: str = "gpu"
    #: Parsed `cim:` block; None unless the YAML provides one.
    cim_config: Optional[CIMConfig] = None
    #: Parsed `mapping:` block (P3.1, ADJ-5); None = the derived dedicated
    #: mapping, which reproduces today's `cim.chip.layers_per_chip` placement.
    mapping_config: Optional[MappingConfig] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, object]) -> "HWConfig":
        config_dict = _require_mapping("hardware_config", config_dict)
        sw_config = SWConfig.from_dict(_require_field("hardware_config", config_dict, "sw_param"))
        sch_config = SchedulingConfig.from_dict(config_dict.get("parallelism", {}))
        scheduling_for_network = {
            "auto": sch_config.auto,
            "dp": sch_config.train.dp,
            "ep": sch_config.train.ep,
            "pp": sch_config.pp,
            "mb": sch_config.mb,
            "tp": sch_config.tp,
            "cp": sch_config.cp,
            "tp_sp": sch_config.tp_sp,
        }
        network_dimensions, network_faults, network_overlap = _parse_network_layout(
            config_dict.get("network"),
            scheduling_for_network,
        )
        if not network_dimensions:
            raise ValueError("network section must define at least one dimension")
        network_layout_config = _build_network_layout_config(
            network_dimensions,
            network_faults,
            network_overlap,
        )
        tech_config = TechConfig.from_dict(_require_field("hardware_config", config_dict, "tech_param"))

        if "power_breakdown" in config_dict:
            power_config = PowerBreakdownConfig.from_dict(config_dict["power_breakdown"])
        else:
            power_config = PowerBreakdownConfig(
                TDP=1.0, core=1.0, DRAM=1.0, L2=1.0, L1=1.0, reg_mem=1.0,
                network=NetworkPowerConfig(inter_node=1.0, intra_node=1.0)
            )
        if "area_breakdown" in config_dict:
            area_config = AreaBreakdownConfig.from_dict(config_dict["area_breakdown"])
        else:
            area_config = AreaBreakdownConfig(
                proc_chip_area_budget=1.0, core=1.0, DRAM=1.0, L2=1.0, L1=1.0, reg_mem=1.0,
                node_area_budget=1.0, network=NetworkAreaConfig(inter_node=1.0, intra_node=1.0)
            )
        if "perimeter_breakdown" in config_dict:
            perimeter_config = PerimeterBreakdownConfig.from_dict(config_dict["perimeter_breakdown"])
        else:
            perimeter_config = PerimeterBreakdownConfig(DRAM=0.1, inter_node=0.1, intra_node=0.1)

        memory_hierarchy_config = MemoryHierarchyConfig.from_dict(
            _require_field("hardware_config", config_dict, "memory_hierarchy")
        )
        execution_backend = ExecutionBackend.from_dict(config_dict.get("execution_backend", {}))
        inference_config = InferenceHWConfig.from_dict(config_dict.get("inference"))

        device_class = str(config_dict.get("device_class", "gpu")).strip().lower()
        if device_class not in {"gpu", "fws_cim"}:
            raise ValueError(
                f"device_class must be 'gpu' or 'fws_cim' (got {config_dict.get('device_class')!r})"
            )
        cim_dict = config_dict.get("cim")
        cim_config = CIMConfig.from_dict(cim_dict) if cim_dict is not None else None
        mapping_dict = config_dict.get("mapping")
        mapping_config = (
            None if mapping_dict is None else MappingConfig.from_dict(mapping_dict)
        )
        if mapping_config is not None and device_class != "fws_cim":
            raise ValueError(
                "the mapping: block describes an FWS-CIM placement (macros, chips, "
                "shard groups, shared digital chiplets) and requires "
                f"device_class: fws_cim (got {device_class!r})."
            )

        return cls(
            sw_config=sw_config,
            tech_config=tech_config,
            power_breakdown=power_config,
            sch_config=sch_config,
            area_breakdown=area_config,
            perimeter_breakdown=perimeter_config,
            memory_hierarchy=memory_hierarchy_config,
            network_layout=network_layout_config,
            execution_backend=execution_backend,
            inference_config=inference_config,
            device_class=device_class,
            cim_config=cim_config,
            mapping_config=mapping_config,
        )

@dataclass
class ModelConfig:
    model_config: object
    inference_config: Optional["LLMInferenceConfig"]


def _convert_scalar_string(value: str):
    try:
        return float(value)
    except ValueError:
        pass

    digit = [int(s) for s in value.split() if s.isdigit()]
    order = [str(s) for s in value.split() if not s.isdigit()]
    if not (order and digit):
        return value

    prefix = order[0][0]
    bit = order[0][1] if len(order[0]) > 1 else "B"
    mult = 1

    if prefix == "K":
        mult = 1024
    elif prefix == "M":
        mult = 1024 * 1024
    elif prefix == "G":
        mult = 1024 * 1024 * 1024
    elif prefix == "T":
        mult = 1024 * 1024 * 1024 * 1024
    else:
        raise ValueError(f"Unknown prefix '{prefix}' while parsing value '{value}'")

    if bit == "b":
        mult = mult / 8  # Capacity is expected in Bytes
    elif bit != "B":
        raise ValueError(f"Unknown type '{bit}' while parsing value '{value}'")

    return digit[0] * mult


def _convert_value(value):
    if isinstance(value, dict):
        convert(value)
        return value
    if isinstance(value, list):
        return [_convert_value(item) for item in value]
    if isinstance(value, str):
        try:
            return _convert_scalar_string(value)
        except ValueError:
            return value
    return value


def convert(d):
    if not isinstance(d, dict):
        return d
    for key, val in list(d.items()):
        d[key] = _convert_value(val)
    return d


def parse_bandwidth_string(value):
    """Parse bandwidth/size string (e.g., '300 GB', '1986 GB') to bytes.

    This function uses the same logic as convert() to parse bandwidth strings.
    Returns the numeric value if already a number, or None if value is None.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(parse_bandwidth_string(item) for item in value)

    if not isinstance(value, str):
        return float(value)

    digit = [int(s) for s in value.split() if s.isdigit()]
    order = [str(s) for s in value.split() if not s.isdigit()]

    if not order or not digit:
        # If no units found, try to parse as float
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Cannot parse bandwidth value: {value}")

    assert len(order) >= 1
    assert len(digit) >= 1

    prefix = order[0][0]
    bit = order[0][1] if len(order[0]) > 1 else 'B'  # Default to Bytes
    mult = 1

    if prefix == "K":
        mult = 1024
    elif prefix == "M":
        mult = 1024 * 1024
    elif prefix == "G":
        mult = 1024 * 1024 * 1024
    elif prefix == "T":
        mult = 1024 * 1024 * 1024 * 1024
    else:
        raise ValueError(f"Unknown prefix: {prefix} in bandwidth value: {value}")

    if bit == "b":
        mult = mult / 8  # Convert bits to Bytes
    elif bit == "B":
        mult = mult
    else:
        raise ValueError(f"Unknown type: {bit} in bandwidth value: {value}")

    return digit[0] * mult


def parse_config(filename, config_type):
    """Parse a yaml configuration file for this experiment.
    Args:
            filename (str): Path to the configuration file
    Returns:
            FullConfig: Contains dataset, model, optimization, training and
            scheduling configurations
    """
    with open(filename, "r") as f:
        try:
            config_dict = _yaml.safe_load(f)
        except _YAMLError as exc:
            hint = (
                f"Failed to parse YAML config '{filename}'. "
                "Please check indentation and required sections like 'attention' and parameters such as 'moe.num_experts'."
            )
            raise ValueError(hint) from exc
        # print(config_dict)
        convert(config_dict)
    if config_type == "hardware":
        config = HWConfig.from_dict(config_dict)
    elif config_type == "GEMM":
        model_config = GEMMConfig.from_dict(config_dict["model_param"])
        config = ModelConfig(model_config=model_config, inference_config=None)
    elif str(config_type).upper() in {"LLM", "VIT"}:
        model_config = LLMConfig.from_dict(config_dict["model_param"])
        inference_config = None
        if model_config.run_type == "inference":
            inference_config = LLMInferenceConfig.from_dict(config_dict.get("inference_param"))
        config = ModelConfig(model_config=model_config, inference_config=inference_config)
    else:
        raise ValueError("Invalid config type: {}".format(config_type))
    
    return config


def validate_hw_config(hw_config: HWConfig) -> None:
    backend = getattr(hw_config, "execution_backend", None)
    model = getattr(backend, "model", "analytical") if backend else "analytical"
    network_layout = getattr(hw_config, "network_layout", None)
    if str(model).lower() == "analytical" and network_layout:
        for dim in getattr(network_layout, "dimensions", ()):
            topo = str(getattr(dim, "topology_type", "ring")).lower()
            if topo != "ring":
                raise RuntimeError(
                    "Non-ring network topologies are not supported in analytical mode. "
                    "Only execution_backend.model='astra' (requires a valid AstraSim install) supports non-ring networks."
                )

    device_class = str(getattr(hw_config, "device_class", "gpu")).lower()
    cim_config = getattr(hw_config, "cim_config", None)
    if device_class == "fws_cim":
        if cim_config is None:
            raise ValueError(
                "device_class: fws_cim requires a top-level 'cim:' block (analog/fabric/chip) "
                "in the hardware config. Add the cim block or set device_class: gpu."
            )
        if str(model).lower() == "astra":
            raise ValueError(
                "device_class: fws_cim supports only the analytical backend; "
                "set execution_backend.model: analytical (got 'astra')."
            )
        pp = int(getattr(getattr(hw_config, "sch_config", None), "pp", 1) or 1)
        if pp != 1:
            raise ValueError(
                "device_class: fws_cim requires parallelism.pp = 1: chip placement comes from "
                f"cim.chip.layers_per_chip, not pipeline stages (got pp={pp})."
            )
        # Rejected here (not only in the sa-fabric pricing path) so that
        # fabric.model: gpu_native cannot silently ignore cp.
        cp = int(getattr(getattr(hw_config, "sch_config", None), "cp", 1) or 1)
        if cp != 1:
            raise ValueError(
                "device_class: fws_cim does not support context parallelism; "
                f"set parallelism.cp: 1 (got cp={cp})."
            )
        kvcache_type = str(
            getattr(getattr(hw_config, "inference_config", None), "kvcache_type", "hbm_only")
        ).strip().lower()
        if kvcache_type == "hbm_only":
            raise ValueError(
                "device_class: fws_cim requires inference.kvcache_type: cim_sram or cim_dram — "
                "the device has no HBM (got 'hbm_only')."
            )
        if kvcache_type == "cim_dram" and cim_config.kv_dram is None:
            raise ValueError(
                "inference.kvcache_type: cim_dram requires a cim.kv_dram block "
                "(capacity_bytes, bandwidth_bytes_per_s[, energy_per_bit_pj]) in the "
                "hardware config."
            )
        if kvcache_type == "cim_sram" and cim_config.kv_dram is not None:
            print(
                "[WARNING]: hardware config defines cim.kv_dram but inference.kvcache_type is "
                "'cim_sram'; the kv_dram block is ignored. Set kvcache_type: cim_dram to use it."
            )
    elif cim_config is not None:
        # Unknown top-level YAML keys are silently ignored, so a missing/typo'd
        # device_class would otherwise turn a CIM config into a silent GPU run.
        print(
            "[WARNING]: hardware config contains a 'cim:' block but device_class is "
            f"'{device_class}'; the cim block is ignored. Set device_class: fws_cim to use it."
        )
    if device_class != "fws_cim":
        # Mirror of the ignored-cim-block warning above: a CIM KV story on a
        # non-CIM device is inert (the GPU path never reads kvcache_type), so
        # say so instead of silently pricing the KV cache as hbm_only.
        kvcache_type = str(
            getattr(getattr(hw_config, "inference_config", None), "kvcache_type", "hbm_only")
        ).strip().lower()
        if kvcache_type in ("cim_sram", "cim_dram"):
            print(
                f"[WARNING]: inference.kvcache_type is '{kvcache_type}' but "
                f"device_class is '{device_class}'; the CIM KV story is ignored "
                "and the run behaves as kvcache_type: hbm_only. Set "
                "device_class: fws_cim to use it."
            )


#: Layer-plan block kinds the MAPPED fws_cim path prices END TO END: P3 places
#: them on a device and P4 prices every op through a NAMED P2 law (QIF P1.5).
#: ``ssm`` rides ``price_ssm_block`` (SSD / selective scan) on the shared
#: digital chiplet, ``short_conv`` is absorbed into the per-macro pool sizing
#: (ADJ-3), ``linear_attn`` rides ``price_linear_attention_block`` (the gated
#: delta rule), and ``attention`` rides the fabric laws (D13). A kind reaches
#: this tuple only when a run of a real model produced a priced op for it.
_FWS_CIM_MAPPED_BLOCK_KINDS = ("attention", "ssm", "short_conv", "linear_attn")


def _validate_fws_cim_model(model: object, *, mapped: bool = False) -> None:
    """Scope gate for device_class: fws_cim — transformer inference only.

    Pass 2 admits dense and MoE LLMs (attention_type mha/gqa) with
    autoregressive decode next to the pass-1 ViT class. Fixed-weight-
    stationary arrays admit no weight writes, so training is permanently out
    (not deferred); flash attention, MLA, and the astra backend stay out of
    scope. Every rejection names the offending setting.

    ``mapped`` is the QIF P1.5 narrowing (ADJ-1). A run that declares a
    ``mapping:`` block takes the placed-DAG path, where the hybrid block kinds
    in :data:`_FWS_CIM_MAPPED_BLOCK_KINDS` ARE priced — so the hybrid refusal
    lifts for exactly those kinds and for nothing else. The UNMAPPED fws_cim
    path keeps the full refusal: the closed-form spatial report has no stage
    for a recurrence and would silently price a Mamba layer as attention.
    """
    if not isinstance(model, LLMConfig):
        model_type = getattr(model, "model_type", type(model).__name__)
        raise ValueError(
            "device_class: fws_cim supports only transformer (LLM or ViT) inference; "
            f"got a non-transformer model config ({model_type!r})."
        )
    run_type = str(getattr(model, "run_type", "training")).lower()
    if run_type != "inference":
        raise ValueError(
            "device_class: fws_cim does not support model_param.run_type: training — "
            "fixed-weight-stationary arrays admit no weight writes. Set run_type: inference."
        )
    hybrid_kinds = tuple(getattr(model, "hybrid_block_kinds", ()) or ())
    unpriced_kinds = (
        tuple(kind for kind in hybrid_kinds if kind not in _FWS_CIM_MAPPED_BLOCK_KINDS)
        if mapped
        else hybrid_kinds
    )
    if unpriced_kinds:
        raise ValueError(
            "device_class: fws_cim does not support model_param.layer_plan block kinds "
            f"{', '.join(unpriced_kinds)} — the analog macro laws price weight GEMMs and "
            "the digital fabric prices attention, and neither covers a recurrence, a "
            "delta-rule state, or a depthwise short convolution. "
            + (
                "The mapped path prices "
                + ", ".join(_FWS_CIM_MAPPED_BLOCK_KINDS)
                + "; these kinds are not priced anywhere."
                if mapped
                else "Pricing for these blocks lands with the MAPPED path: declare a "
                "`mapping:` block in the hardware config and the run is placed by P3 "
                "(mapping) and priced op by op by P4 (evaluation) through the P2 "
                "(macro resource model) laws. The unmapped closed-form report has no "
                "stage for them."
            )
        )
    # A PARALLEL-BRANCH layer template (Falcon-H1: attention || Mamba2 || MLP
    # inside one layer) has every law it needs — the block kinds are all on the
    # mapped list — and no LOWERING. program/fws_build._layer walks a layer's
    # mixers in order and CHAINS them, feeding each branch the previous one's
    # output, and it never sums the branch outputs. A parallel model lowered
    # that way is priced as a deeper sequential model with the wrong dataflow:
    # a wrong number, not a missing one, which is exactly what P1.4's rule
    # exists to prevent. Refused by name until the DAG builder can express a
    # branch.
    layer_plan = getattr(model, "layer_plan", None)
    if layer_plan is not None and str(getattr(layer_plan, "structure", "")) == "parallel_branch":
        raise ValueError(
            "device_class: fws_cim does not support model_param.layer_plan.structure: "
            "parallel_branch — every LAW the branches need exists (the block kinds are "
            "priced), but the DAG lowering does not: program/fws_build lowers a layer's "
            "mixers SEQUENTIALLY, chaining each branch onto the previous branch's output "
            "and never summing them, so a parallel-branch model would be priced as a "
            "deeper sequential one. Pricing lands when the builder can express a branch; "
            "until then only sequential layer plans run."
        )
    attention_type = str(
        getattr(getattr(model, "attention", None), "attention_type", "mha")
    ).lower()
    if attention_type == "mla":
        raise ValueError(
            "device_class: fws_cim does not support model_param.attention.attention_type: mla — "
            "the MLA inference path discards per-op GEMM times and keeps only memory accesses "
            "(_mla_component_forward_stats), so the CIM intercept's zero-memory-access results "
            "would feed an aggregate roofline instead of the CIM laws; MLA needs its own "
            "pricing seam. Use attention_type: mha or gqa."
        )
    if bool(getattr(model, "use_flashattention", False)):
        raise ValueError(
            "device_class: fws_cim prices attention on the digital fabric and does not support "
            "model_param.attention.use_flashattention: true; set it to false."
        )
    # A decode-only run (prefill_len <= 0) would skip calc_time's prefill
    # branch and with it the entire FWS spatial report — the authoritative
    # device output — while still exiting 0. Reject it loudly, exactly like
    # the DSE does for the same input.
    seq_len = int(getattr(model, "seq_len", 0) or 0)
    decode_len = int(getattr(model, "decode_len", 0) or 0)
    if seq_len > 0 and decode_len >= seq_len:
        raise ValueError(
            "device_class: fws_cim requires prefill_len = seq_len - decode_len > 0 "
            "— the FWS spatial report (like the DSE) prices the prefill wavefront "
            f"(got seq_len={seq_len}, decode_len={decode_len}). Reduce "
            "model_param.decode_len."
        )


def _unpriced_model_inputs(model: "LLMConfig", *, mapped_fws: bool = False) -> Tuple[str, ...]:
    """Modeling inputs that parse but that no timing path prices yet (P1.4).

    Same rule as the hybrid block-kind gate: pricing a declared input as if it
    were absent is a wrong number rather than a missing one. Only the parameter
    census (`llm_util`) reads these; the pricing seam lands with P2-P4.

    ``mapped_fws`` is the QIF P1.5 narrowing: on the placed-DAG path two of
    these inputs ARE priced now, and each is lifted for a named reason rather
    than as a block. Everything else stays refused, on every path.
    """
    unpriced: List[str] = []
    attention = getattr(model, "attention", None)
    window = getattr(attention, "window", None)
    if window is not None and int(window.local_global_interval) != 1 and not mapped_fws:
        # PRICED on the mapped path: fws_eval reads the pattern per layer and
        # calls sliding_window_prefill_timing / sliding_window_decode_timing at
        # the capped context (P2.6 5).
        unpriced.append(
            "model_param.attention.window (every attention op is priced against "
            "the full KV context, so a local layer would be charged as global)"
        )
    if bool(getattr(attention, "output_gate", False)) and not mapped_fws:
        # PRICED on the mapped path (QIF C1): the gate projection is an
        # ORDINARY weight matrix, so P3 places it as an analog tile beside the
        # qkv/o_proj pair (hidden x num_heads*head_dim — llm_util's own census
        # term, so the placement and the parameter count stay ONE accounting)
        # and P4 prices it with the same analog law every other weight matrix
        # gets. The sigmoid and the elementwise multiply are per-macro pool
        # work (ADJ-3 / D12), sized from the timeline like every other pool op.
        # The closed-form and GPU paths have no stage for the matrix and keep
        # refusing it by name.
        unpriced.append(
            "model_param.attention.output_gate (no stage prices the gate projection)"
        )
    if tuple(getattr(model, "shared_weight_groups", ()) or ()):
        unpriced.append(
            "model_param.shared_weight_groups (depth-shared weights are stored once "
            "but every path prices one weight set per layer)"
        )
    ffn_dims = dict(getattr(model, "ffn_dims", {}) or {})
    if mapped_fws:
        # PRICED on the mapped path: the shared-expert width reaches the array
        # census and the mapping's stage shapes, the same number llm_util's
        # parameter census already reads. Any OTHER key still only moves the
        # census, so it stays refused BY NAME.
        ffn_dims = {key: value for key, value in ffn_dims.items() if key != "shared_expert"}
    if ffn_dims:
        unpriced.append(
            "model_param.ffn_dims keys "
            + ", ".join(sorted(ffn_dims))
            + " (every dense FFN stage is priced at model_param.intermediate_size)"
        )
    return tuple(unpriced)


def validate_model_config(hw_config: HWConfig, model_config: ModelConfig) -> None:
    sch = getattr(hw_config, "sch_config", None)
    if sch is None:
        raise ValueError("hardware parallelism settings are missing")

    pp = sch.pp
    mb = sch.mb
    tp = sch.tp
    cp = sch.cp
    train_dp = sch.train.dp
    train_ep = sch.train.ep

    model = model_config.model_config

    device_class = str(getattr(hw_config, "device_class", "gpu")).lower()
    # QIF P1.5 (ADJ-1): a `mapping:` block is what makes a run take the
    # placed-DAG path, and that path prices block kinds the closed form cannot
    # express. The narrowing is scoped to exactly that pair — fws_cim AND a
    # declared mapping — so the GPU paths and the unmapped closed-form path
    # keep the P1.4 refusal verbatim.
    mapped_fws = device_class == "fws_cim" and (
        getattr(hw_config, "mapping_config", None) is not None
    )
    if device_class == "fws_cim":
        _validate_fws_cim_model(model, mapped=mapped_fws)

    if isinstance(model, GEMMConfig):
        if tp > 1:
            if model.gemm_shard_axis == "row" and (model.K % tp != 0):
                raise ValueError("GEMM row sharding requires K divisible by tp")
            if model.gemm_shard_axis == "col" and (model.N % tp != 0):
                raise ValueError("GEMM col sharding requires N divisible by tp")
        return

    if not isinstance(model, LLMConfig):
        raise ValueError("Unsupported model config type for validation")

    # Every timing path in this repo prices ONE uniform transformer layer
    # repeated num_layers times. A hybrid layer plan would be priced as if its
    # SSM / linear-attention / short-conv blocks were attention layers, which
    # is a wrong number rather than a missing one — so refuse it outright on
    # every device class. Pricing lands with P2-P4.
    unpriced_kinds = (
        tuple(
            kind
            for kind in model.hybrid_block_kinds
            if kind not in _FWS_CIM_MAPPED_BLOCK_KINDS
        )
        if mapped_fws
        else tuple(model.hybrid_block_kinds)
    )
    if unpriced_kinds:
        raise ValueError(
            "model_param.layer_plan declares block kinds "
            f"{', '.join(unpriced_kinds)}, which no device path prices yet: "
            f"device_class {device_class!r} would "
            "price them as plain attention layers. Pricing for these blocks lands with "
            "the fws_cim MAPPED path: a hardware config that declares a `mapping:` "
            "block is placed by P3 (mapping) and priced by P4 (evaluation) through the "
            "P2 (macro resource model) laws. Until then only attention-only layer "
            "plans run."
        )

    unpriced = _unpriced_model_inputs(model, mapped_fws=mapped_fws)
    if unpriced:
        raise ValueError(
            "model_param declares modeling inputs no device path prices yet: "
            + "; ".join(unpriced)
            + ". device_class "
            + repr(device_class)
            + " would price the model as if they were absent, which is a wrong number "
            "rather than a missing one. Pricing lands with the fws_cim MAPPED path: "
            "P3 (mapping) places the model and P4 (evaluation) prices it through the "
            "P2 (macro resource model) laws. Until then remove the field."
        )

    if model.use_moe and model.top_k > model.num_experts:
        raise ValueError("model_param.moe.top_k cannot exceed model_param.moe.num_experts")

    run_type = str(getattr(model, "run_type", "training")).lower()
    attention_type = str(getattr(getattr(model, "attention", None), "attention_type", "mha")).lower()
    use_flashattention = bool(getattr(model, "use_flashattention", False))
    if run_type == "inference" and attention_type == "mla":
        decode_len = int(getattr(model, "decode_len", 0) or 0)
        if use_flashattention and decode_len > 0:
            global _MLA_DECODE_FLASH_WARNED
            if not _MLA_DECODE_FLASH_WARNED:
                print(
                    "[WARNING]: MLA inference models the latent-cache decode path without FlashMLA-specific kernels. "
                    "model_param.attention.use_flashattention only affects MLA prefill; "
                    "decode continues to use the non-flash latent-cache path."
                )
                _MLA_DECODE_FLASH_WARNED = True
    if run_type == "inference":
        replica_count = sch.inference.replica_count
        moe_dp = sch.inference.moe_dp

        if mb > 1:
            print(
                f"[WARNING]: LLM inference configured with mb={mb} (>1). \n "
                "Pipeline micro-batching is ill-defined for autoregressive decode and should be avoided."
            )
        if getattr(hw_config.sw_config, "dp_zero_stage", 0) >= 3 and replica_count > 1:
            raise ValueError(
                "ZeRO-3 data parallelism is not supported for inference runs "
                "(dp_zero_stage must be <3 or replica_count=1)."
            )
        if not model.use_moe and moe_dp > 1:
            raise ValueError(
                "parallelism.inference.moe_dp must be 1 when MoE is disabled."
            )
        if model.decode_len is not None and model.decode_len > model.seq_len:
            raise ValueError("model_param.decode_len must be <= seq_len for inference")
        if model.is_vit:
            if model.decode_len not in (0, None):
                raise ValueError("model_param.decode_len must be 0 for ViT inference")
            if model.use_moe:
                raise ValueError("ViT inference does not support MoE/EP. Set model_param.moe.num_experts=1.")
    elif model.is_vit:
        if model.decode_len not in (0, None):
            raise ValueError("model_param.decode_len must be 0 for ViT training")
        if model.use_moe:
            raise ValueError("ViT training does not support MoE/EP. Set model_param.moe.num_experts=1.")
        if train_ep > 1:
            raise ValueError("ViT training does not support EP. Set parallelism.train.ep=1.")
    else:
        if cp > 1 and train_ep > 1:
            raise ValueError(
                "Unsupported parallelism combination: cp > 1 with ep > 1 is not allowed. "
                "Set parallelism.cp=1 or parallelism.train.ep=1."
            )

    if model.gradient_accumulation_steps > 1:
        zero_stage = int(getattr(hw_config.sw_config, "dp_zero_stage", 0) or 0)
        # ZeRO-2 communication is still modeled with a coarse final-step approximation.
        # ZeRO-3 is supported via per-DP-op grad-acc scheduling in pipeline graph construction.
        if zero_stage == 2:
            raise ValueError(
                "Gradient accumulation steps > 1 is not supported with ZeRO-2 (dp_zero_stage == 2). "
                "Use ZeRO-1/DP or ZeRO-3."
            )

    if model.global_batch_size % model.gradient_accumulation_steps != 0:
        raise ValueError(
            "Global batch size must be divisible by gradient accumulation steps"
        )
    if run_type != "inference":
        batch_size = model.global_batch_size // model.gradient_accumulation_steps
        dp_dense = train_dp * train_ep if model.use_moe else train_dp
        if batch_size % dp_dense != 0:
            if model.use_moe:
                raise ValueError(f"Batch size must be divisible by dp*ep when MoE is enabled: {batch_size} % {dp_dense} != 0")
            raise ValueError(f"Batch size must be divisible by data parallelism degree: {batch_size} % {train_dp} != 0")
        mini_batch = batch_size // dp_dense
        if mini_batch % mb != 0:
            raise ValueError(f"Batch size must be divisible by micro-batch size: {mini_batch} % {mb} != 0")

        pipeline_interleave = int(
            getattr(getattr(hw_config, "sw_config", None), "pipeline_interleave", 1) or 1
        )
        if pipeline_interleave > 1 and pp > 1:
            num_layers = int(getattr(model, "num_layers", 0) or 0)
            if num_layers % (pp * pipeline_interleave) != 0:
                raise ValueError(
                    "sw_param.pipeline_interleave requires num_layers to divide evenly into "
                    f"pp * interleave virtual stages: num_layers={num_layers}, pp={pp}, "
                    f"pipeline_interleave={pipeline_interleave}."
                )

        if not model.use_moe and train_ep > 1:
            raise ValueError(
                "parallelism.train.ep must be 1 when MoE is disabled. "
                "Set parallelism.train.ep=1 or enable MoE."
            )

    if model.use_moe:
        allow_moe_padding = _env_flag("RAPID_ALLOW_MOE_EXPERT_PADDING")
        if run_type == "inference":
            moe_ranks = tp * max(1, moe_dp)
            if moe_ranks > model.num_experts:
                raise ValueError(
                    "MoE routing group size cannot exceed the number of MoE experts "
                    f"(moe_group={moe_ranks}, tp={tp}, moe_dp={moe_dp})."
                )
            if model.num_experts % moe_ranks != 0:
                if allow_moe_padding:
                    global _MOE_PADDING_WARNED
                    if not _MOE_PADDING_WARNED:
                        print(
                            "[WARNING]: MoE expert count is not divisible by tp*moe_dp for "
                            "inference; padding experts to enable simulation. "
                            "Set RAPID_ALLOW_MOE_EXPERT_PADDING=0 to enforce divisibility."
                        )
                        _MOE_PADDING_WARNED = True
                else:
                    raise ValueError(
                        "Number of MoE experts must be divisible by the MoE routing group size "
                        f"(moe_group={moe_ranks}, tp={tp}, moe_dp={moe_dp})."
                    )
        else:
            tp_sp = bool(getattr(sch, "tp_sp", False))
            if tp > 1 and train_ep > 1 and not tp_sp:
                raise ValueError(
                    "MoE with tp>1 and ep>1 requires sequence parallelism. "
                    "Set parallelism.tp_sp=true."
                )
            moe_ranks = train_ep
            if moe_ranks > model.num_experts:
                raise ValueError(
                    "MoE routing group size cannot exceed the number of MoE experts "
                    f"(moe_group={moe_ranks}, tp={tp}, ep={train_ep})."
                )
            if model.num_experts % moe_ranks != 0:
                raise ValueError(
                    "Number of MoE experts must be divisible by the MoE routing group size "
                    f"(moe_group={moe_ranks}, tp={tp}, ep={train_ep})."
                )
            effective_batch = mini_batch
            if pp > 1:
                effective_batch = mini_batch // mb
            elif dp_dense <= 1:
                effective_batch = batch_size
            seq_per_rank = math.ceil(model.seq_len / max(1, cp))
            tokens_owner = int(effective_batch) * int(seq_per_rank)
            tokens_dispatched = tokens_owner * int(model.top_k)
            if tokens_dispatched % moe_ranks != 0:
                raise ValueError(
                    "MoE routed tokens must divide evenly across the MoE routing group for batched expert GEMMs\n"
                    f"(tokens_owner = effective_batch * seq_per_rank = {effective_batch} * {seq_per_rank})\n"
                    f"(tokens_dispatched = tokens_owner * top_k = {tokens_owner} * {model.top_k})\n"
                    f"(tokens_dispatched={tokens_dispatched} % moe_group={moe_ranks} != 0)"
                )
            tokens_local = tokens_dispatched // moe_ranks
            experts_per_rank = model.num_experts // moe_ranks
            if tokens_local % experts_per_rank != 0:
                raise ValueError(
                    "MoE routed tokens per rank must divide evenly across experts for batched expert GEMMs\n"
                    f"(tokens_owner = effective_batch * seq_per_rank = {effective_batch} * {seq_per_rank})\n"
                    f"(tokens_dispatched = tokens_owner * top_k = {tokens_owner} * {model.top_k})\n"
                    f"(tokens_local = tokens_dispatched // moe_ranks = {tokens_dispatched} // {moe_ranks})\n"
                    f"(tokens_local={tokens_local} % experts_per_rank={experts_per_rank} != 0)\n"
                )
        if run_type != "inference" and getattr(hw_config.sw_config, "dp_zero_stage", 0) >= 2:
            raise NotImplementedError("MoE with ZeRO-2/3 (dp_zero_stage >= 2) is not supported yet.")
        network_layout = getattr(hw_config, "network_layout", None)
        faulty_links = getattr(network_layout, "faulty_links", ()) if network_layout else ()
        if faulty_links:
            raise ValueError(
                "MoE with faulty links is not supported yet. Please disable faults or MoE."
            )


def validate_configs(hw_config: HWConfig, model_config: ModelConfig) -> None:
    validate_hw_config(hw_config)
    validate_model_config(hw_config, model_config)
