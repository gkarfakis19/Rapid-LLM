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
    List,
    Mapping,
    Optional,
    Tuple,
)

from timing_model import CollectiveType

from program.block import BlockTemplate, CommMeta
from program.layout import RankLayout
from program.axes import CANONICAL_AXES, axis_sizes_from
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
        table = axis_sizes_from(self)
        if axis not in table:
            raise WorkloadError(
                f"Unknown parallelism axis {axis!r} (known: {sorted(table)})"
            )
        return table[axis]

    def __post_init__(self) -> None:
        for name in CANONICAL_AXES:
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


def _declare_axes(
    key: CommKey,
    interconnect: Optional[Any],
    routing_mode: Optional[Any],
) -> Tuple[AxisName, ...]:
    """The default ``CommSpec.axes`` derivation, reproducing today's grouping::

        axes = (interconnect_type,)          # the normal case
        axes = routing_policy.routing_axes() # iff moe_routing_mode is set

    The second line **deletes** the participant-count inference at
    ``legacy_lowering.py:243-248``
    (``if axis == "ep" and participants == tp*ep -> ("tp","ep")``): the
    composite communicator is now DECLARED by the MoE routing policy at the
    source (INTERFACES §2.6, §3.3).

    One implementation, shared by :meth:`CommSpec.from_legacy` (raw dict) and
    :meth:`CommSpec.from_comm_meta` (typed :class:`~program.block.CommMeta`).
    """
    if routing_mode is not None:
        # Lazy import: program.policies imports program.work -> program.workload,
        # so a module-level import here would close the cycle. The routing
        # table is DATA (INTERFACES §2.6) and this is its only consumer at L0.
        from program.policies.routing import routing_for_mode

        return routing_for_mode(str(routing_mode)).routing_axes()
    if interconnect is None:
        raise WorkloadError(
            f"comm_metadata[{key!r}] has neither 'interconnect_type' nor "
            "'moe_routing_mode'; the communicator axis cannot be declared"
        )
    # A composite axis is DECLARED, never recovered from a participant count.
    # ``'dp*cp'`` is the gradient reducer's communicator when context
    # parallelism is on (BUG_LEDGER 19): cp ranks hold REPLICATED parameters and
    # compute PARTIAL gradients from different sequence chunks, so the reduction
    # group is dp x cp exactly as in Megatron. Spelled as one table value so the
    # composite stays a property of the comm rule, not of this function.
    text = str(interconnect)
    if "*" in text:
        return tuple(part for part in text.split("*") if part)
    return (text,)


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
        axes = _declare_axes(key, interconnect, routing_mode)

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

    @classmethod
    def from_comm_meta(cls, meta: CommMeta) -> "CommSpec":
        """Build one spec from a typed :class:`~program.block.CommMeta`.

        ``BlockTemplate.comm_metadata`` is the per-template comm table
        ``train_timing._build_transformer_template`` produced (one
        ``_register_specs`` accumulator per template). It is an identity
        conversion — no defaulting, no inference — and it is what makes
        :class:`BlockTemplates` able to carry the dense and MoE tables
        SEPARATELY (INTERFACES §1.6 amendment, 2026-07-26).
        """
        if not isinstance(meta, CommMeta):
            raise WorkloadError(
                f"CommSpec.from_comm_meta expects a CommMeta (got {type(meta).__name__})"
            )
        if not isinstance(meta.kind, CollectiveType):
            raise WorkloadError(
                f"BlockTemplate comm entry {meta.name!r} has no CollectiveType "
                f"(got {type(meta.kind).__name__})"
            )
        return cls(
            key=meta.name,
            size_bytes=float(meta.size_bytes or 0.0),
            kind=meta.kind,
            axes=_declare_axes(meta.name, meta.interconnect, meta.moe_routing_mode),
            participants=max(1, int(meta.participants or 1)),
            ga_required_every_cycle=bool(meta.ga_required_every_cycle),
            tp_shard=bool(meta.tp_shard),
            placement=str(meta.placement),
            local_comp_time=float(meta.local_comp_time or 0.0),
            parallel_group=meta.parallel_group,
            moe_component=meta.moe_component,
            moe_routing_mode=(
                None if meta.moe_routing_mode is None else str(meta.moe_routing_mode)
            ),
            extra=dict(meta.extra or {}),
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

    @classmethod
    def from_block_template(cls, template: BlockTemplate) -> "CommSpecTable":
        """The comm table of ONE :class:`~program.block.BlockTemplate`.

        Block-template comm keys are named PER TEMPLATE
        (``train_timing._register_specs``, whose accumulator is local to
        ``_build_transformer_template``), so the dense and MoE tables are
        SEPARATE tables and may legitimately declare the same key with
        different bytes — e.g. ``ep_dense_sync_layernorm1_backward`` is
        337,641,472 bytes dense and 67,141,632 bytes MoE on the
        ``train:*:moe:ep2`` golden rows. Unioning them would be a 5.03x byte
        error, and with a mixed ``moe_layer_mask`` both are needed at once.
        """
        return cls(
            [
                (name, CommSpec.from_comm_meta(meta))
                for name, meta in (template.comm_metadata or {}).items()
            ]
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
    #: PER-LAYER apply-grad prices. The fused per-stage optimizer node scales
    #: these by the layers its own stage owns (``program.work.optimizer_duration``,
    #: BUG_LEDGER 10b), so neither is a whole-stage duration.
    "optimizer",
    "optimizer_moe",
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
    * **reader-before** — ``_run_hybrid`` builds the PIPELINE program before the
      write-back and must see PRISTINE analytical durations;
    * **reader-after** — ``build_flat_program_for_memory`` constructs a fresh
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

    AMENDMENT to INTERFACES §1.7 (dated 2026-07-26): ``full_recomputation``
    is carried here for ``recompute_policy_for`` (§2.7). ``misc["flattened_mode"]``
    is deliberately NOT carried: it became the dispatcher's granularity
    selection, and then (2026-08-02) not even that — see recompute.py.

    ``pipeline_style_recompute`` was DELETED 2026-08-02: train_timing.py:297
    hardwired it to ``bool(full_recomputation)`` with no config path, so the
    legacy predicate ``full_recomputation AND (flattened_mode OR
    pipeline_style_recompute)`` was ``full_recomputation`` in every reachable
    state, and the granularity coupling it appeared to carry never existed.
    """

    run_type: RunType
    grad_accum_cycle: GradAccumCycle
    dp_microbatch_mode: DpMicrobatchMode
    zero_stage: int
    pipeline_interleave: int = 1  #: v; the closed-form bubble correction (Class B item 8)
    full_recomputation: bool = False

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
    """The dense (required) and MoE (optional) block templates AND their
    per-template comm tables.

    AMENDMENT to INTERFACES §1.6 (dated 2026-07-26). ``dense_comm`` /
    ``moe_comm`` are new. Block-template comm keys are named per template
    (``train_timing._build_transformer_template`` holds one ``_register_specs``
    accumulator per call, and ``_register_specs`` itself RAISES on a byte
    conflict — train_timing.py:4694-4711), so ONE flat table cannot express
    them: measured on ``train:hybrid:dp2tp2cp1pp2mb2sp1:moe:ep2``,
    ``ep_dense_sync_layernorm1_backward`` is 337,641,472 bytes in the dense
    template and 67,141,632 in the MoE one (5.03x). A mixed ``moe_layer_mask``
    needs both simultaneously.

    ``WorkloadSpec.comm`` therefore keeps only PIPELINE-LEVEL keys (the dp
    reducers, the ZeRO gathers, ``cross_layer``, the EP grad syncs) and every
    block-template key is resolved through this object, keyed on whether the
    layer is MoE.
    """

    dense: BlockTemplate
    moe: Optional[BlockTemplate] = None
    #: comm table of ``dense``; derived from ``dense.comm_metadata`` when absent.
    dense_comm: Optional[CommSpecTable] = None
    #: comm table of ``moe``; derived from ``moe.comm_metadata`` when absent.
    moe_comm: Optional[CommSpecTable] = None

    def __post_init__(self) -> None:
        if not isinstance(self.dense, BlockTemplate):
            raise WorkloadError("BlockTemplates.dense must be a BlockTemplate")
        if self.moe is not None and not isinstance(self.moe, BlockTemplate):
            raise WorkloadError("BlockTemplates.moe must be a BlockTemplate or None")
        for field_name, table in (("dense_comm", self.dense_comm), ("moe_comm", self.moe_comm)):
            if table is not None and not isinstance(table, CommSpecTable):
                raise WorkloadError(
                    f"BlockTemplates.{field_name} must be a CommSpecTable or None"
                )
        if self.dense_comm is None:
            object.__setattr__(
                self, "dense_comm", CommSpecTable.from_block_template(self.dense)
            )
        if self.moe is None:
            if self.moe_comm is not None:
                raise WorkloadError(
                    "BlockTemplates.moe_comm was supplied without a moe BlockTemplate"
                )
        elif self.moe_comm is None:
            object.__setattr__(
                self, "moe_comm", CommSpecTable.from_block_template(self.moe)
            )

    def template_for(self, is_moe_layer: bool) -> BlockTemplate:
        if is_moe_layer:
            if self.moe is None:
                raise WorkloadError(
                    "A MoE layer was requested but BlockTemplates.moe is None"
                )
            return self.moe
        return self.dense

    def comm_for(self, is_moe_layer: bool) -> CommSpecTable:
        """The comm table of the template that expands ``is_moe_layer``."""
        if is_moe_layer:
            if self.moe_comm is None:
                raise WorkloadError(
                    "A MoE layer was requested but BlockTemplates.moe_comm is None"
                )
            return self.moe_comm
        if self.dense_comm is None:  # pragma: no cover - __post_init__ fills it
            raise WorkloadError("BlockTemplates.dense_comm is None")
        return self.dense_comm

    def tables(self) -> Tuple[CommSpecTable, ...]:
        """Every block comm table, dense first. Diagnostics / whole-workload
        scans (e.g. ``routing_policy_for``), never expansion."""
        out = [self.comm_for(False)]
        if self.moe_comm is not None:
            out.append(self.moe_comm)
        return tuple(out)


#: The ``misc_metadata`` keys :meth:`WorkloadSpec.from_timing` reads. Every one
#: is written unconditionally by the single producer
#: (``train_timing._prepare_execution_graphs``, which the inference model
#: inherits), so a missing key means the producer changed and the workload is
#: malformed — not that a default applies (**W1**).
REQUIRED_MISC_KEYS: Tuple[str, ...] = (
    "dp_microbatch_mode",
    "dp_zero_stage",
    "full_recomputation",
    "moe_layer_mask",
    "model_type",
    "num_batch",
    "num_layer",
)


def _require_misc(misc_metadata: Mapping[str, Any], key: str) -> Any:
    """``misc_metadata[key]``, with a WorkloadError that names the producer."""
    if key not in misc_metadata:
        raise WorkloadError(
            f"misc_metadata is missing required key {key!r} "
            f"(required: {list(REQUIRED_MISC_KEYS)}; produced by "
            "train_timing._prepare_execution_graphs — INTERFACES §1.7). W1 "
            "forbids a silent default here."
        )
    return misc_metadata[key]


@dataclass(frozen=True)
class WorkloadSpec:
    """The complete, typed workload description (INTERFACES §1.6).

    ``comm`` holds the **pipeline-level** comm keys only (the dp reducers, the
    ZeRO gathers, ``cross_layer``, the EP grad syncs — i.e. exactly what
    ``train_timing._build_comm_metadata`` produces). Block-template keys live on
    :class:`BlockTemplates` and are resolved per layer through
    :meth:`block_comm`, because they are named per template and the dense/MoE
    tables genuinely disagree (INTERFACES §1.6 amendment, 2026-07-26).
    """

    degrees: ParallelDegrees
    shape: ModelShape
    run: RunPolicy
    #: PIPELINE-LEVEL comm keys only — see the class docstring.
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
        self._check_pipeline_block_key_disjointness()

    def _check_pipeline_block_key_disjointness(self) -> None:
        """A key present in BOTH the pipeline table and a block table must mean
        the same thing in both.

        The dense and MoE block tables may disagree with each other — that is
        the whole point of :class:`BlockTemplates` — but a pipeline key that
        silently shadows a block key with different bytes is the flat-table
        defect all over again, so it is a construction error.
        """
        for table in self.blocks.tables():
            for key, block_spec in table.items():
                pipeline_spec = self.comm.get(key)
                if pipeline_spec is None:
                    continue
                if (
                    pipeline_spec.size_bytes != block_spec.size_bytes
                    or pipeline_spec.kind is not block_spec.kind
                    or pipeline_spec.axes != block_spec.axes
                    or pipeline_spec.participants != block_spec.participants
                ):
                    raise WorkloadError(
                        f"Comm key {key!r} is declared both pipeline-level and "
                        "block-level with different content "
                        f"(pipeline: {pipeline_spec.size_bytes} bytes / "
                        f"{pipeline_spec.kind.name} / {pipeline_spec.axes}; block: "
                        f"{block_spec.size_bytes} bytes / {block_spec.kind.name} / "
                        f"{block_spec.axes}). Pipeline and block comm namespaces "
                        "must not shadow each other."
                    )

    # -- derived, pure ----------------------------------------------------
    def is_moe_layer(self, layer: LayerId) -> bool:
        return self.shape.is_moe_layer(layer)

    def cluster_size(self) -> int:
        return self.degrees.cluster_size()

    def block_template(self, layer: LayerId) -> BlockTemplate:
        return self.blocks.template_for(self.shape.is_moe_layer(layer))

    def block_comm(self, layer: LayerId) -> CommSpecTable:
        """The comm table the block expansion of ``layer`` resolves against.

        THE fix for the dense/MoE comm-key collision: a block comm key is only
        ever looked up through the template that declared it.
        """
        return self.blocks.comm_for(self.shape.is_moe_layer(layer))

    def all_comm_specs(self) -> Tuple[CommSpec, ...]:
        """Every declared spec: pipeline-level first, then each block table.

        For whole-workload SCANS only (``routing_policy_for``, diagnostics,
        byte histograms). Never for resolution — a key may legitimately appear
        more than once here with different bytes.
        """
        out: List[CommSpec] = list(self.comm.values())
        for table in self.blocks.tables():
            out.extend(table.values())
        return tuple(out)

    def freeze(self) -> "FrozenWorkload":
        return FrozenWorkload(spec=self, durations=self.durations.snapshot())

    def with_(self, **changes: Any) -> "WorkloadSpec":
        """Typed ``dataclasses.replace`` (used by tests and by the dispatcher's
        final/nonfinal grad-accum cycle pair)."""
        return replace(self, **changes)

    # -- THE producer seam (INTERFACES §1.7) ------------------------------
    @classmethod
    def from_timing(
        cls,
        *,
        tp: int,
        cp: int,
        ep: int,
        pp: int,
        dp: int,
        run_type: Any,
        comp_times: Mapping[str, Any],
        comm_metadata: Mapping[str, Mapping[str, Any]],
        misc_metadata: Mapping[str, Any],
        blocks: BlockTemplates,
        overlap: OverlapSpec,
        interconnect: Mapping[str, Tuple[float, float]],
        grad_accum_cycle: Any = GradAccumCycle.FINAL,
        pipeline_interleave: int = 1,
        layout: Optional[RankLayout] = None,
        granularity_hint: Any = None,
        durations: Optional[DurationTable] = None,
    ) -> "WorkloadSpec":
        """THE §1.7 producer map, in ONE place.

        This is the seam ``train_timing`` / ``inference_timing`` hand across:
        every field is read from the raw timing-model outputs exactly once, with
        no ``getattr(..., fallback)`` against a ``program/`` type (**W1**). It
        replaces the five duplicated dispatcher-construction flows
        (``train_timing.calc_time_llm`` / ``estimate_memory_only``,
        ``inference_timing``, ``llm_util.estimate_inference_memory``,
        ``simulate_inference_graph``, ``huggingface_bench_validation``), each of
        which re-derived ``include_backward``, ``include_optimizer`` and the
        effective dp for itself.

        ``dp`` is the DECLARED data-parallel degree; the stored
        :attr:`degrees` carries the EFFECTIVE one, because inference is a
        single-replica measurement (legacy ``dp_override=1``, three copies at
        ``llm_execution.py:412``/``:607``/``:743``). Making it the degree — not a
        per-call override — is what lets ``sharding_policy_for`` answer
        ``NullSharding`` and ``GradAccumPolicy.emits`` answer False with no
        run-type special case anywhere downstream.

        ``layout`` is normally left ``None`` here and filled in by the dispatcher
        (which owns the hardware network layout); use :meth:`with_`.

        **AMENDMENT 2026-07-29 (W1).** Every ``misc_metadata`` key this seam
        reads is REQUIRED. It used to read all eight through
        ``dict.get(key, fallback)``, which is the same silent-default hazard W1
        deletes on the ``getattr`` side and which W1's own text already forbids:
        a producer that stops writing ``num_layer`` yielded a 0-layer model, and
        one that stops writing ``model_type`` yielded ``""`` — which the ViT
        naming path dispatches on. ``_require_misc`` names the missing key and
        the producer instead.
        """
        run = RunPolicy(
            run_type=RunType.parse(run_type),
            grad_accum_cycle=GradAccumCycle.parse(grad_accum_cycle),
            dp_microbatch_mode=DpMicrobatchMode.parse(
                _require_misc(misc_metadata, "dp_microbatch_mode")
            ),
            zero_stage=int(_require_misc(misc_metadata, "dp_zero_stage") or 0),
            pipeline_interleave=max(1, int(pipeline_interleave or 1)),
            full_recomputation=bool(_require_misc(misc_metadata, "full_recomputation")),
        )
        declared = max(1, int(dp or 1))
        degrees = ParallelDegrees(
            tp=max(1, int(tp or 1)),
            cp=max(1, int(cp or 1)),
            ep=max(1, int(ep or 1)),
            pp=max(1, int(pp or 1)),
            dp=1 if run.run_type is RunType.INFERENCE else declared,
        )
        shape = ModelShape(
            num_layers=int(_require_misc(misc_metadata, "num_layer") or 0),
            micro_batches=int(_require_misc(misc_metadata, "num_batch") or 0),
            model_type=str(_require_misc(misc_metadata, "model_type") or ""),
            moe_layer_mask=tuple(
                bool(value)
                for value in (_require_misc(misc_metadata, "moe_layer_mask") or [])
            ),
        )
        return cls(
            degrees=degrees,
            shape=shape,
            run=run,
            comm=CommSpecTable.from_legacy(comm_metadata or {}),
            blocks=blocks,
            overlap=overlap,
            layout=layout,
            interconnect=dict(interconnect or {}),
            granularity_hint=granularity_hint,
            durations=(
                durations if durations is not None else DurationTable(comp_times or {})
            ),
        )

    def for_block(self, *, layout: Optional[RankLayout], moe: bool) -> "WorkloadSpec":
        """The BLOCK workload: ONE layer of ONE template over the (tp,cp,ep)
        sublayout (INTERFACES §3.1).

        BLOCK's device space carries neither ``pp`` nor ``dp``, so a block run is
        a single-stage, single-replica, single-microbatch measurement of one
        transformer block — which is exactly what the legacy
        ``build_block_program(template, direction, ...)`` measured with
        ``dp_override=1``. Recompute is off because a block measurement times one
        template chain; the rematerialization chain IS that chain.

        ``moe`` selects which template the single layer uses, by declaring the
        one-entry ``moe_layer_mask``. The comm table is the *template's* own
        (INTERFACES §1.6 amendment B1), which is why the dense and MoE block
        runs cannot share a workload.
        """
        if moe and self.blocks.moe is None:
            raise WorkloadError(
                "for_block(moe=True) needs BlockTemplates.moe, which is None"
            )
        blocks = (
            BlockTemplates(dense=self.blocks.moe, dense_comm=self.blocks.moe_comm)
            if moe
            else BlockTemplates(dense=self.blocks.dense, dense_comm=self.blocks.dense_comm)
        )
        return replace(
            self,
            degrees=ParallelDegrees(
                tp=self.degrees.tp,
                cp=self.degrees.cp,
                ep=self.degrees.ep,
                pp=1,
                dp=1,
            ),
            shape=ModelShape(
                num_layers=1,
                micro_batches=1,
                model_type=self.shape.model_type,
                moe_layer_mask=(),
            ),
            run=replace(self.run, full_recomputation=False),
            blocks=blocks,
            layout=layout,
        )


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
        """PIPELINE-LEVEL comm keys only; block keys go through
        :meth:`block_comm`."""
        return self.spec.comm

    @property
    def blocks(self) -> BlockTemplates:
        return self.spec.blocks

    @property
    def layout(self) -> Optional[RankLayout]:
        return self.spec.layout

    def is_moe_layer(self, layer: LayerId) -> bool:
        return self.spec.is_moe_layer(layer)

    def cluster_size(self) -> int:
        return self.spec.cluster_size()

    def block_comm(self, layer: LayerId) -> CommSpecTable:
        return self.spec.block_comm(layer)

    def all_comm_specs(self) -> Tuple[CommSpec, ...]:
        return self.spec.all_comm_specs()


def ceil_div(total: float, divisor: int) -> float:
    """``ceil(total / divisor)`` with the legacy guard ``max(1, divisor)``
    (pipeline_fine.py:645). One implementation, used by ``ByteSplit``."""
    return float(math.ceil(float(total) / float(max(1, int(divisor)))))
