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

"""L0 — ``WorkloadSpec`` and friends (INTERFACES.md §1).

Replaces **both** ``program.schedule.ScheduleInputs`` and the duck-typed
``ScheduleSpec.from_pipeline_graph``, which today silently accepts a wrong
object and yields ``mb=0`` / ``num_layers=0`` because every read is a
``getattr(..., default)`` / ``misc.get(..., getattr(...))`` chain.

Hard rules this module enforces (INTERFACES §1.8):

* **W1** — no ``getattr`` at all in this file, and no ``dict.get`` fallback
  against a type defined in ``program/``. Malformed input raises
  :class:`WorkloadError` at construction. The only ``Mapping.get`` calls here
  read the *raw legacy dict* at the ``train_timing`` seam (``from_legacy``),
  which is untyped by construction and is exactly the boundary the typed
  constructors defend.
* **W2** — :class:`DurationTable` is the single mutable member of a
  :class:`WorkloadSpec`; builders never see it, they see a
  :class:`FrozenDurations` snapshot, so a build is a pure function of input.
* **W3** — :meth:`CommSpecTable.require` raises ``WorkloadError`` naming every
  known key, never ``KeyError``.

**Byte/timing math stays upstream.** ``CommSpec.size_bytes`` /
``.participants`` / ``.kind`` are computed by ``train_timing`` and are
read-only here (PLAN "Owner decisions": blast radius). Policies decide
*existence*, *axis* and *attachment* — never bytes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Tuple,
)

from timing_model import CollectiveType

from program.block import BlockTemplate
from program.layout import RankLayout
from program.types import AxisName, CommKey, LayerId

__all__ = [
    "WorkloadError",
    "RunType",
    "GradAccumCycle",
    "DpMicrobatchMode",
    "ParallelDegrees",
    "ModelShape",
    "CommSpec",
    "CommSpecTable",
    "DurationTable",
    "FrozenDurations",
    "RunPolicy",
    "OverlapSpec",
    "BlockTemplates",
    "WorkloadSpec",
    "FrozenWorkload",
    "DURATION_KEYS",
]


class WorkloadError(ValueError):
    """A WorkloadSpec was constructed from inconsistent or incomplete inputs."""


class RunType(Enum):
    TRAINING = auto()
    INFERENCE = auto()

    @classmethod
    def parse(cls, value: Any) -> "RunType":
        """Parse the ``model.run_type`` string at the train_timing seam."""
        if isinstance(value, RunType):
            return value
        text = str(value).strip().lower()
        if text == "inference":
            return cls.INFERENCE
        if text in ("training", "train"):
            return cls.TRAINING
        raise WorkloadError(f"Unknown run_type {value!r} (expected 'training' or 'inference')")


class GradAccumCycle(Enum):
    FINAL = auto()
    NONFINAL = auto()

    @classmethod
    def parse(cls, value: Any) -> "GradAccumCycle":
        if isinstance(value, GradAccumCycle):
            return value
        text = str(value).strip().lower()
        if text == "nonfinal":
            return cls.NONFINAL
        if text == "final":
            return cls.FINAL
        raise WorkloadError(
            f"Unknown grad_accum_cycle {value!r} (expected 'final' or 'nonfinal')"
        )


class DpMicrobatchMode(Enum):
    EVERY_MB = auto()
    LAST_MB = auto()

    @classmethod
    def parse(cls, value: Any) -> "DpMicrobatchMode":
        if isinstance(value, DpMicrobatchMode):
            return value
        text = str(value).strip().lower()
        if text == "last_mb":
            return cls.LAST_MB
        if text == "every_mb":
            return cls.EVERY_MB
        raise WorkloadError(
            f"Unknown dp_microbatch_mode {value!r} (expected 'every_mb' or 'last_mb')"
        )


# ---------------------------------------------------------------------------
# Parallelism / model shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParallelDegrees:
    """The five canonical parallelism degrees (INTERFACES §1.1)."""

    tp: int
    cp: int
    ep: int  #: the GRAPH ep: ``time_calc.ep`` when use_moe else 1 (train_timing.py:4667)
    pp: int
    dp: int

    def cluster_size(self) -> int:
        """``tp*cp*ep`` — the number of devices in one pipeline stage (>= 1)."""
        return max(1, self.tp * self.cp * self.ep)

    def of(self, axis: AxisName) -> int:
        """Degree of one canonical axis. Raises on an unknown axis name."""
        table = {"tp": self.tp, "cp": self.cp, "ep": self.ep, "pp": self.pp, "dp": self.dp}
        if axis not in table:
            raise WorkloadError(
                f"Unknown parallelism axis {axis!r} (known: {sorted(table)})"
            )
        return table[axis]

    def __post_init__(self) -> None:
        for name in ("tp", "cp", "ep", "pp", "dp"):
            value = self.__dict__[name]
            if not isinstance(value, int) or isinstance(value, bool):
                raise WorkloadError(f"ParallelDegrees.{name} must be an int (got {value!r})")
            if value < 1:
                raise WorkloadError(f"ParallelDegrees.{name} must be >= 1 (got {value})")


@dataclass(frozen=True)
class ModelShape:
    """Model/pipeline shape. ``__post_init__`` is the fix for schedule.py:186:
    a wrong object can no longer produce ``mb=0`` — it raises."""

    num_layers: int
    micro_batches: int  #: legacy ``misc["num_batch"]``
    model_type: str  #: lowercase; "vit*" selects the ViT block naming
    moe_layer_mask: Tuple[bool, ...] = ()  #: () when dense-only; else len == num_layers

    def is_moe_layer(self, layer: LayerId) -> bool:
        if not self.moe_layer_mask:
            return False
        if layer < 0 or layer >= len(self.moe_layer_mask):
            raise WorkloadError(
                f"Layer {layer} is out of range for {len(self.moe_layer_mask)} layers"
            )
        return bool(self.moe_layer_mask[layer])

    @property
    def has_moe_layers(self) -> bool:
        return any(self.moe_layer_mask)

    def __post_init__(self) -> None:
        if not isinstance(self.num_layers, int) or self.num_layers <= 0:
            raise WorkloadError(
                f"ModelShape.num_layers must be a positive int (got {self.num_layers!r}); "
                "this is the schedule.py:186 hazard — a wrong input object used to "
                "silently produce num_layers=0"
            )
        if not isinstance(self.micro_batches, int) or self.micro_batches <= 0:
            raise WorkloadError(
                f"ModelShape.micro_batches must be a positive int (got {self.micro_batches!r}); "
                "this is the schedule.py:186 hazard — a wrong input object used to "
                "silently produce mb=0"
            )
        if self.moe_layer_mask and len(self.moe_layer_mask) != self.num_layers:
            raise WorkloadError(
                f"ModelShape.moe_layer_mask has {len(self.moe_layer_mask)} entries "
                f"but num_layers is {self.num_layers}"
            )
        object.__setattr__(self, "model_type", str(self.model_type).lower())
        object.__setattr__(self, "moe_layer_mask", tuple(bool(v) for v in self.moe_layer_mask))


# ---------------------------------------------------------------------------
# CommSpec / CommSpecTable — the typed comm table
# ---------------------------------------------------------------------------


#: Legal ``CommSpec.placement`` values (block-template chain position).
COMM_PLACEMENTS: Tuple[str, ...] = ("pre", "post")


@dataclass(frozen=True)
class CommSpec:
    """One entry of the comm table.

    BYTES AND KIND ARE COMPUTED UPSTREAM (``train_timing``) and are read-only
    here — the restructure does not touch byte/timing math.

    ``axes`` vs ``participants`` are SEPARATE and must stay separate:
    ``COMMUNICATION_RULES[TENSOR_CONTEXT_HYBRID]['output_proj']['forward']``
    declares ``participants='tp', interconnect='cp'`` (train_timing.py:151-154)
    — the analytical participant count and the communicator axis genuinely
    differ. ``participants`` feeds ``analytic_sim.convert_comm_sizes_to_times``
    and the gmap traffic weight; ``axes`` feeds the communicator and never the
    timing model. **Do not derive one from the other.**
    """

    key: CommKey
    size_bytes: float  #: RAW; dp reducers are floats and MUST stay floats
    kind: CollectiveType
    axes: Tuple[AxisName, ...]  #: COMMUNICATOR IDENTITY — declared, never inferred
    participants: int  #: ANALYTICAL participant count — see the class docstring
    ga_required_every_cycle: bool = False
    tp_shard: bool = False  #: "this collective is instantiated per cluster rank"
    placement: str = "post"  #: "pre" | "post" (block-template comm keys only)
    #: Class B item 5: carried, never materialized. The legacy assignment was
    #: already dead (the same quantity is modeled once per stage as the
    #: ``optimizer`` node); attaching it per dp edge would double-count by
    #: ``layers_per_stage``. Kept as DATA so a reviewer can flip it.
    local_comp_time: float = 0.0
    parallel_group: Optional[str] = None
    moe_component: Optional[str] = None  #: "base_all_to_all" | "residual_p2p"
    moe_routing_mode: Optional[str] = None  #: "ep" | "tp_ep"
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CollectiveType):
            raise WorkloadError(
                f"CommSpec[{self.key!r}].kind must be a CollectiveType "
                f"(got {type(self.kind).__name__})"
            )
        if self.placement not in COMM_PLACEMENTS:
            raise WorkloadError(
                f"CommSpec[{self.key!r}].placement must be one of {COMM_PLACEMENTS} "
                f"(got {self.placement!r})"
            )
        if not self.axes:
            raise WorkloadError(f"CommSpec[{self.key!r}] declares no communicator axes")
        if int(self.participants) < 1:
            raise WorkloadError(
                f"CommSpec[{self.key!r}].participants must be >= 1 (got {self.participants})"
            )
        object.__setattr__(self, "size_bytes", float(self.size_bytes))
        object.__setattr__(self, "participants", int(self.participants))
        object.__setattr__(self, "axes", tuple(str(a) for a in self.axes))

    # -- the train_timing seam -------------------------------------------
    @classmethod
    def from_legacy(cls, key: CommKey, data: Mapping[str, Any]) -> "CommSpec":
        """Build one spec from a raw ``comm_metadata`` entry.

        This is the ONE untyped boundary: ``data`` is a plain dict produced by
        ``train_timing._build_comm_metadata`` / ``_register_specs``. Required
        fields are read positively (``KeyError`` -> ``WorkloadError``); only
        genuinely optional legacy fields use ``Mapping.get``.

        Default ``axes`` derivation, reproducing today's grouping exactly::

            axes = (interconnect_type,)          # the normal case
            axes = routing_policy.routing_axes() # iff moe_routing_mode is set

        The second line **deletes** the participant-count inference at
        ``legacy_lowering.py:243-248``
        (``if axis == "ep" and participants == tp*ep -> ("tp","ep")``): the
        composite communicator is now DECLARED by the MoE routing policy at the
        source (INTERFACES §2.6, §3.3).
        """
        if "type" not in data:
            raise WorkloadError(f"comm_metadata[{key!r}] has no 'type' (CollectiveType)")
        kind = data["type"]
        if not isinstance(kind, CollectiveType):
            raise WorkloadError(
                f"comm_metadata[{key!r}]['type'] must be a CollectiveType "
                f"(got {type(kind).__name__})"
            )
        interconnect = data.get("interconnect_type")
        routing_mode = data.get("moe_routing_mode")
        if routing_mode is not None:
            # Lazy import: program.policies imports program.work -> program.workload,
            # so a module-level import here would close the cycle. The routing
            # table is DATA (INTERFACES §2.6) and this is its only consumer at L0.
            from program.policies.routing import routing_for_mode

            axes = routing_for_mode(str(routing_mode)).routing_axes()
        else:
            if interconnect is None:
                raise WorkloadError(
                    f"comm_metadata[{key!r}] has neither 'interconnect_type' nor "
                    "'moe_routing_mode'; the communicator axis cannot be declared"
                )
            axes = (str(interconnect),)

        known = {
            "size",
            "type",
            "participants",
            "interconnect_type",
            "local_comp_time",
            "tp_shard",
            "ga_required_every_cycle",
            "placement",
            "parallel_group",
            "moe_component",
            "moe_routing_mode",
        }
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            key=key,
            size_bytes=float(data.get("size", 0) or 0.0),
            kind=kind,
            axes=axes,
            participants=int(data.get("participants", 1) or 1),
            ga_required_every_cycle=bool(data.get("ga_required_every_cycle", False)),
            tp_shard=bool(data.get("tp_shard", False)),
            placement=str(data.get("placement", "post")),
            local_comp_time=float(data.get("local_comp_time", 0) or 0.0),
            parallel_group=data.get("parallel_group"),
            moe_component=data.get("moe_component"),
            moe_routing_mode=(None if routing_mode is None else str(routing_mode)),
            extra=extra,
        )


class CommSpecTable(Mapping[CommKey, CommSpec]):
    """Immutable, insertion-ordered mapping of ``CommKey -> CommSpec``."""

    __slots__ = ("_data",)

    def __init__(self, specs: Any = ()) -> None:
        data: Dict[CommKey, CommSpec] = {}
        items = specs.items() if isinstance(specs, Mapping) else specs
        for entry in items:
            if isinstance(entry, CommSpec):
                spec = entry
                key = spec.key
            else:
                key, spec = entry
            if not isinstance(spec, CommSpec):
                raise WorkloadError(
                    f"CommSpecTable[{key!r}] must be a CommSpec (got {type(spec).__name__})"
                )
            if key != spec.key:
                raise WorkloadError(
                    f"CommSpecTable key {key!r} disagrees with CommSpec.key {spec.key!r}"
                )
            if key in data:
                raise WorkloadError(f"Duplicate CommSpec key {key!r}")
            data[key] = spec
        self._data = data

    # -- Mapping ----------------------------------------------------------
    def __getitem__(self, key: CommKey) -> CommSpec:
        return self._data[key]

    def __iter__(self) -> Iterator[CommKey]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"CommSpecTable({list(self._data)})"

    # -- typed accessors --------------------------------------------------
    def require(self, key: CommKey) -> CommSpec:
        """W3: raises :class:`WorkloadError` (never ``KeyError``) naming every
        known key, so a dropped comm key is a loud build failure."""
        if key not in self._data:
            raise WorkloadError(
                f"Comm key {key!r} is not registered. Known keys: {sorted(self._data)}"
            )
        return self._data[key]

    def keys_with_axis(self, axis: AxisName) -> Tuple[CommKey, ...]:
        return tuple(k for k, s in self._data.items() if axis in s.axes)

    @classmethod
    def from_legacy(cls, raw: Mapping[str, Mapping[str, Any]]) -> "CommSpecTable":
        """Convert a raw ``comm_metadata`` dict (insertion order preserved)."""
        return cls(
            [(name, CommSpec.from_legacy(name, data)) for name, data in (raw or {}).items()]
        )


# ---------------------------------------------------------------------------
# DurationTable — the ONE mutable object, by design (INTERFACES §1.3)
# ---------------------------------------------------------------------------


#: The compute-duration keys the pipeline schedule reads (train_timing:5025-5035).
DURATION_KEYS: Tuple[str, ...] = (
    "embedding_f",
    "embedding_b",
    "linear_softmax_f",
    "linear_softmax_b",
    "transformer_f",
    "transformer_b",
    "transformer_f_dense",
    "transformer_b_dense",
    "transformer_f_moe",
    "transformer_b_moe",
    "optimizer",
)

#: The six keys the AstraSim BLOCK write-back replaces atomically
#: (llm_execution.py:995-1004).
_BLOCK_WRITEBACK_KEYS: Tuple[str, ...] = (
    "transformer_f",
    "transformer_b",
    "transformer_f_dense",
    "transformer_b_dense",
    "transformer_f_moe",
    "transformer_b_moe",
)


@dataclass(frozen=True)
class FrozenDurations(Mapping[str, float]):
    """An immutable snapshot of a :class:`DurationTable`. This — not the table
    — is what L1..L4 read, so a build can never observe a torn write."""

    revision: int
    values: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", dict(self.values))

    def __getitem__(self, key: str) -> float:
        if key not in self.values:
            raise WorkloadError(
                f"Duration key {key!r} is not present. Known keys: {sorted(self.values)}"
            )
        return self.values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def get_or(self, key: str, default: float = 0.0) -> float:
        """Explicit, named default read — the ONLY defaulted duration read.

        Ports ``ScheduleSpec.time(key, default)`` (schedule.py:224). The
        default is passed by the caller at the one site that legitimately has
        one (the ``transformer_f_dense`` -> ``transformer_f`` fallback), never
        buried in an expansion loop.
        """
        value = self.values.get(key)
        return float(value) if value is not None else float(default)


class DurationTable:
    """The single mutable member of a :class:`WorkloadSpec`.

    Load-bearing, by design (INTERFACES §1.3):

    * **writer** — ``llm_execution._update_comp_times_from_timings`` after the
      AstraSim BLOCK runs;
    * **reader-before** — ``_run_hybrid`` builds the COARSE program before the
      write-back and must see PRISTINE analytical durations;
    * **reader-after** — ``build_fine_program_for_memory`` constructs a fresh
      spec after the write-back and must see the UPDATED durations.

    ``revision`` increments on every write-back and is stamped onto the built
    Program (``meta.misc["duration_revision"]``) so stale reuse is detectable,
    not silent.
    """

    __slots__ = ("_values", "_revision")

    def __init__(self, values: Mapping[str, float]) -> None:
        cleaned: Dict[str, float] = {}
        for key, value in (values or {}).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                # Legacy comp_times carried non-numeric entries; ScheduleSpec
                # filtered them (schedule.py:191-195). Same filter, one place.
                continue
            numeric = float(value)
            if numeric < 0.0:
                raise WorkloadError(f"Duration {key!r} is negative ({numeric})")
            cleaned[str(key)] = numeric
        self._values = cleaned
        self._revision = 0

    @property
    def revision(self) -> int:
        return self._revision

    def snapshot(self) -> FrozenDurations:
        return FrozenDurations(revision=self._revision, values=dict(self._values))

    def write_block_timings(
        self,
        *,
        dense_forward: Optional[float] = None,
        dense_backward: Optional[float] = None,
        moe_forward: Optional[float] = None,
        moe_backward: Optional[float] = None,
    ) -> int:
        """The ONLY mutator. Writes the six legacy keys atomically and returns
        the new revision. Replaces the dict-poking at llm_execution.py:995-1004
        and centralizes the negative-duration check that lives at three call
        sites today."""
        for name, value in (
            ("dense_forward", dense_forward),
            ("dense_backward", dense_backward),
            ("moe_forward", moe_forward),
            ("moe_backward", moe_backward),
        ):
            if value is not None and float(value) < 0.0:
                raise WorkloadError(f"AstraSim transformer time {name} must be >= 0 (got {value})")

        staged = dict(self._values)
        if dense_forward is not None:
            staged["transformer_f"] = float(dense_forward)
            staged["transformer_f_dense"] = float(dense_forward)
        if dense_backward is not None:
            staged["transformer_b"] = float(dense_backward)
            staged["transformer_b_dense"] = float(dense_backward)
        if moe_forward is not None:
            staged["transformer_f_moe"] = float(moe_forward)
        if moe_backward is not None:
            staged["transformer_b_moe"] = float(moe_backward)

        self._values = staged
        self._revision += 1
        return self._revision

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"DurationTable(rev={self._revision}, keys={sorted(self._values)})"


# ---------------------------------------------------------------------------
# RunPolicy / OverlapSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunPolicy:
    """One home for the rules currently copied verbatim three times each
    (llm_execution.py:392-395 / :606-612 / :742-746 — audit A10).

    AMENDMENT to INTERFACES §1.7 (dated 2026-07-26): ``full_recomputation`` and
    ``pipeline_style_recompute`` are carried here. The §1.7 producer table omits
    them, but ``recompute_policy_for`` (§2.7) needs both — they are
    ``misc_metadata["full_recomputation"]`` / ``["pipeline_style_recompute"]``
    (train_timing.py:5068-5072). ``misc["flattened_mode"]`` is deliberately NOT
    carried: it becomes the dispatcher's granularity selection (§2.7).
    """

    run_type: RunType
    grad_accum_cycle: GradAccumCycle
    dp_microbatch_mode: DpMicrobatchMode
    zero_stage: int
    pipeline_interleave: int = 1  #: v; the closed-form bubble correction (Class B item 8)
    full_recomputation: bool = False
    pipeline_style_recompute: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.run_type, RunType):
            raise WorkloadError("RunPolicy.run_type must be a RunType")
        if not isinstance(self.grad_accum_cycle, GradAccumCycle):
            raise WorkloadError("RunPolicy.grad_accum_cycle must be a GradAccumCycle")
        if not isinstance(self.dp_microbatch_mode, DpMicrobatchMode):
            raise WorkloadError("RunPolicy.dp_microbatch_mode must be a DpMicrobatchMode")
        if int(self.zero_stage) < 0:
            raise WorkloadError(f"RunPolicy.zero_stage must be >= 0 (got {self.zero_stage})")
        if int(self.pipeline_interleave) < 1:
            raise WorkloadError(
                f"RunPolicy.pipeline_interleave must be >= 1 (got {self.pipeline_interleave})"
            )

    @property
    def include_backward(self) -> bool:
        return self.run_type is not RunType.INFERENCE

    @property
    def include_optimizer(self) -> bool:
        return self.grad_accum_cycle is not GradAccumCycle.NONFINAL

    @property
    def is_nonfinal_grad_accum_cycle(self) -> bool:
        return self.grad_accum_cycle is GradAccumCycle.NONFINAL

    def effective_dp(self, degrees: ParallelDegrees) -> int:
        return 1 if self.run_type is RunType.INFERENCE else max(1, degrees.dp)

    def retime_dp_count(self, degrees: ParallelDegrees) -> int:
        """Legacy ``_retime_dp_count`` (llm_execution.py:373-378), same rule."""
        return self.effective_dp(degrees)

    def interleave_scale(self, degrees: ParallelDegrees, shape: ModelShape) -> float:
        """Class B item 8, verbatim (llm_execution.py:353-368).

        Named here so the 1F1B student project can replace it with a real
        ``SchedulePolicy``. Quantified bias (BUG_LEDGER B/8): both the additive
        DP tail scaling and the ignored ``v``x p2p hop count make totals too
        low; worst at small mb, large pp, large v, large tail share.
        """
        v = int(self.pipeline_interleave)
        pp = int(degrees.pp)
        mb = int(shape.micro_batches)
        if v <= 1 or pp <= 1:
            return 1.0
        return (mb + (pp - 1) / float(v)) / float(mb + pp - 1)


@dataclass(frozen=True)
class OverlapSpec:
    """Replaces ``tp_overlap``/``tp_sp_overlap``/``cp_overlap`` threaded
    positionally through eight signatures (ext_new_axis.md Part 4 item 3).

    ``tp_sp`` is NOT an axis: ``TENSOR_SEQUENCE`` mode selects ``by_axis['tp']``
    from the ``tp_sp_overlap`` input **at construction**, exactly like
    transforms.py:326-331 does today, but ONCE, in :meth:`from_legacy`.
    """

    parallelism_mode: Any  #: ParallelismMode enum from train_timing
    by_axis: Mapping[AxisName, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cleaned: Dict[AxisName, float] = {}
        for axis, value in (self.by_axis or {}).items():
            fraction = float(value)
            if fraction < 0.0:
                raise WorkloadError(f"OverlapSpec.by_axis[{axis!r}] must be >= 0 (got {fraction})")
            cleaned[str(axis)] = fraction
        object.__setattr__(self, "by_axis", cleaned)

    def fraction(self, axis: AxisName) -> float:
        """0.0 when absent."""
        value = self.by_axis.get(axis)
        return float(value) if value is not None else 0.0

    @property
    def mode_label(self) -> str:
        mode = self.parallelism_mode
        if isinstance(mode, Enum):
            return str(mode.value).lower()
        return str(mode).lower()

    @classmethod
    def from_legacy(
        cls,
        parallelism_mode: Any,
        *,
        tp_overlap: float = 0.0,
        tp_sp_overlap: float = 0.0,
        cp_overlap: float = 0.0,
    ) -> "OverlapSpec":
        """The mode -> axis-fraction selection of transforms.py:326-374, ONCE."""
        probe = cls(parallelism_mode=parallelism_mode, by_axis={})
        label = probe.mode_label
        if label == "tensor_sequence":
            tp_value = float(tp_sp_overlap)
        elif label in ("tensor", "tensor_context_hybrid"):
            tp_value = float(tp_overlap)
        else:
            tp_value = 0.0
        cp_value = float(cp_overlap) if label in ("context", "tensor_context_hybrid") else 0.0
        by_axis: Dict[AxisName, float] = {}
        if tp_value > 0.0:
            by_axis["tp"] = tp_value
        if cp_value > 0.0:
            by_axis["cp"] = cp_value
        return cls(parallelism_mode=parallelism_mode, by_axis=by_axis)


# ---------------------------------------------------------------------------
# WorkloadSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockTemplates:
    dense: BlockTemplate
    moe: Optional[BlockTemplate] = None

    def __post_init__(self) -> None:
        if not isinstance(self.dense, BlockTemplate):
            raise WorkloadError("BlockTemplates.dense must be a BlockTemplate")
        if self.moe is not None and not isinstance(self.moe, BlockTemplate):
            raise WorkloadError("BlockTemplates.moe must be a BlockTemplate or None")


@dataclass(frozen=True)
class WorkloadSpec:
    """The complete, typed workload description (INTERFACES §1.6)."""

    degrees: ParallelDegrees
    shape: ModelShape
    run: RunPolicy
    comm: CommSpecTable
    blocks: BlockTemplates
    overlap: OverlapSpec
    layout: Optional[RankLayout]  #: FULL layout; L2 derives every sublayout
    interconnect: Mapping[str, Tuple[float, float]]  #: axis -> (bandwidth, latency)
    granularity_hint: Any  #: what the dispatcher intends to build (L2 Granularity)

    #: THE ONLY MUTABLE MEMBER. Never read directly by a builder — see §1.3.
    durations: DurationTable = field(compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.degrees, ParallelDegrees):
            raise WorkloadError("WorkloadSpec.degrees must be a ParallelDegrees")
        if not isinstance(self.shape, ModelShape):
            raise WorkloadError("WorkloadSpec.shape must be a ModelShape")
        if not isinstance(self.run, RunPolicy):
            raise WorkloadError("WorkloadSpec.run must be a RunPolicy")
        if not isinstance(self.comm, CommSpecTable):
            raise WorkloadError("WorkloadSpec.comm must be a CommSpecTable")
        if not isinstance(self.blocks, BlockTemplates):
            raise WorkloadError("WorkloadSpec.blocks must be a BlockTemplates")
        if not isinstance(self.overlap, OverlapSpec):
            raise WorkloadError("WorkloadSpec.overlap must be an OverlapSpec")
        if not isinstance(self.durations, DurationTable):
            raise WorkloadError("WorkloadSpec.durations must be a DurationTable")
        if self.shape.has_moe_layers and self.blocks.moe is None:
            raise WorkloadError(
                "WorkloadSpec.shape declares MoE layers but blocks.moe is None"
            )
        object.__setattr__(self, "interconnect", dict(self.interconnect or {}))

    # -- derived, pure ----------------------------------------------------
    def is_moe_layer(self, layer: LayerId) -> bool:
        return self.shape.is_moe_layer(layer)

    def cluster_size(self) -> int:
        return self.degrees.cluster_size()

    def block_template(self, layer: LayerId) -> BlockTemplate:
        if self.shape.is_moe_layer(layer):
            if self.blocks.moe is None:
                raise WorkloadError(f"Layer {layer} is MoE but no MoE BlockTemplate was supplied")
            return self.blocks.moe
        return self.blocks.dense

    def freeze(self) -> "FrozenWorkload":
        return FrozenWorkload(spec=self, durations=self.durations.snapshot())

    def with_(self, **changes: Any) -> "WorkloadSpec":
        """Typed ``dataclasses.replace`` (used by tests and by the dispatcher's
        final/nonfinal grad-accum cycle pair)."""
        return replace(self, **changes)


@dataclass(frozen=True)
class FrozenWorkload:
    """A :class:`WorkloadSpec` plus a duration snapshot. This — not
    ``WorkloadSpec`` — is what L1/L2/L3/L4 consume, so a build is a pure
    function of its input."""

    spec: WorkloadSpec
    durations: FrozenDurations

    # -- convenience pass-throughs (additive; no behavior of their own) ----
    @property
    def degrees(self) -> ParallelDegrees:
        return self.spec.degrees

    @property
    def shape(self) -> ModelShape:
        return self.spec.shape

    @property
    def run(self) -> RunPolicy:
        return self.spec.run

    @property
    def comm(self) -> CommSpecTable:
        return self.spec.comm

    @property
    def layout(self) -> Optional[RankLayout]:
        return self.spec.layout

    def is_moe_layer(self, layer: LayerId) -> bool:
        return self.spec.is_moe_layer(layer)

    def cluster_size(self) -> int:
        return self.spec.cluster_size()


def ceil_div(total: float, divisor: int) -> float:
    """``ceil(total / divisor)`` with the legacy guard ``max(1, divisor)``
    (pipeline_fine.py:645). One implementation, used by ``ByteSplit``."""
    return float(math.ceil(float(total) / float(max(1, int(divisor)))))
