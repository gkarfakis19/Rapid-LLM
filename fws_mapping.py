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

"""The mapping object — QIF P3 (plan `docs/qif/plans/P3_mapping.html`).

Four relations and one annotation set (P3 §1). Everything else is derived.

===========  ==========================  =====================================
Relation     From -> to                  Where it lives here
===========  ==========================  =====================================
Residency    tile -> macro column set    :attr:`FwsMapping.tiles`, sited by
                                         P2's ``enumerate_tiles`` (or by
                                         ``cim.allocation`` when the user
                                         wrote one)
Assembly     macro -> chip               :attr:`FwsMapping.chips`; chip ids are
                                         INTEGERS and capacity is a hard error
Package      chip -> package graph       fully-connected p2p (D17); the drawn
                                         edges are the boundary table
Execution    op -> device                :class:`FwsDevice` + the placed DAG
                                         (``program/fws_build.py``)
Annotation   entity -> (tp, ep, pp)      :class:`ShardCoord`, with membership
                                         CONSTRUCTED by
                                         ``program.groups.CommunicatorFactory``
===========  ==========================  =====================================

Three stances bind the module.

* **A3 — the macro is the atomic resource, the tile is the allocation unit.**
  A macro's stored columns are claimed by column SETS (mux slots); a set holds
  one tile. Unowned columns are legal and REPORTED (ADJ-5), never rounded away.
* **ADJ-5 — a chip hosts many macros; chip = chiplet = package unit.** ``pp``
  is an INDEPENDENT annotation: a chip index is not implicitly a pipeline
  stage. The shared-chiplet count is a config input and the derived suggestion
  is reported next to it. PD is two separate inventories.
* **D21 — honesty invariants.** Chips are integers. Every macro names an owner
  (a tile owner, or a reservation string). One accounting per metric: this
  module never prices an op, and it never restates a number the law layer
  (P2, ``cim_timing``) already owns.

**P3 does not price.** The DAG this mapping feeds carries zero durations by
construction — P4 owns pricing (A1, P4 §1). A boundary rate needs a time base
and P3 has no timeline, so :meth:`FwsMapping.boundary_table` takes the period
as a DECLARED input with a basis string, and reports bytes and required link
time when no period is supplied. Nothing here invents a period.

The `TileSite` row convention, pinned
------------------------------------
``cim_timing.TileSite.row_start``/``row_end`` had two producer conventions and
no validation (the Wave A leftover). This module pins ONE and validates it:

    **a site's row range is the half-open K range of the OWNER'S weight matrix
    that the site holds** — exactly what ``enumerate_tiles`` emits, so
    ``site.row_start == tile.k_start`` and ``site.row_end == tile.k_end``, and
    the span never exceeds the card's rows.

:func:`validate_tile_row_convention` enforces it on every tile this module
places, so a producer that drifts fails a test rather than a drawing.
``CimDeviceModel.allocation_sites`` is the other producer: it fills the range
with ``(0, rows)`` because a `cim.allocation` entry DECLARES no K range at all.
That output feeds the capacity check, which reads column sets only and never
looks at rows — so this module never consumes it as a row range. A user
allocation reaches a tile here through :func:`_apply_user_allocation`, which
keeps the enumerator's rows and overrides only the site's macro and column
sets.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cim_timing
from cim_timing import CimDeviceModel, Tile, TileOwner, TileSite
from program.groups import CommunicatorFactory, canonical_axis_label
from program.ir import GroupKey
from program.layout import RankLayout
from program.placement import DeviceCoord


class MappingError(ValueError):
    """A mapping could not be built, or a declared mapping is inconsistent.

    Carries the DSE's stage-tag convention (P3 exports, P6): the message opens
    with ``[<stage>]`` so a downstream consumer can group failures by the
    relation that refused — ``residency``, ``assembly``, ``package``,
    ``execution`` or ``annotation``.
    """

    def __init__(self, stage: str, message: str) -> None:
        self.stage = str(stage)
        super().__init__(f"[{self.stage}] {message}")


# ---------------------------------------------------------------------------
# Device classes (A2) and the WIDENED device coordinate
# ---------------------------------------------------------------------------

#: The three device classes ops run on, plus the link a transfer rides. A2 and
#: D12/D13 name them; P4's pricing table prices each one from a P2 law.
DEVICE_CLASSES: Tuple[str, ...] = (
    "analog_macro",      # weight GEMM only
    "macro_pool",        # that macro's digital pool: reductions, acts, conv
    "shared_digital",    # act x act, attention, scan / state update (D13)
    "link",              # not a device: the p2p a transfer rides (D17)
)

DEVICE_CLASS_CODE: Mapping[str, int] = {
    name: index for index, name in enumerate(DEVICE_CLASSES)
}

#: The axes a widened device coordinate carries. ``tp``/``ep``/``pp`` are the
#: rewrite's own axes; ``dev_class`` and ``host_macro`` are the widening P3
#: needs (P3 §4: "a device also has a class and a host macro. That is a wider
#: coordinate, not a new IR").
#:
#: The widening is ADDITIVE BY CONSTRUCTION: ``program.placement.DeviceCoord``
#: stores a free-form ``{axis: int}`` mapping, so a coordinate with extra axes
#: is an ordinary DeviceCoord and the GPU rank grid — which never carries these
#: axes — is untouched. Nothing in ``program/`` is edited to express this.
FWS_DEVICE_AXES: Tuple[str, ...] = ("tp", "ep", "pp", "dev_class", "host_macro")

#: Axes a mapping annotates (D20). ``cp`` is deliberately absent.
MAPPING_AXES: Tuple[str, ...] = ("tp", "ep", "pp")

#: The pinned meaning of ``TileSite.row_start``/``row_end`` (see module doc).
TILE_ROW_CONVENTION = "owner_matrix_k_range"

#: Stride used for chip-local macro ids while the slot count is still being
#: derived; it only has to exceed any one chip's usage before the shift.
_LOCAL_BASE_STRIDE = 1_000_000

#: ADJ-6: decode steps the DAG lowers by default. A longer run is a window
#: plus a DISCLOSED extrapolation, never a silent truncation.
DEFAULT_DECODE_WINDOW = 2


@dataclass(frozen=True)
class ShardCoord:
    """One entity's (tp, ep, pp) coordinates — the annotation set of P3 §1."""

    tp: int = 0
    ep: int = 0
    pp: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {"tp": int(self.tp), "ep": int(self.ep), "pp": int(self.pp)}

    def of(self, axis: str) -> int:
        return int(self.as_dict()[axis])


# ---------------------------------------------------------------------------
# Weight-matrix inventory: the GEMM stages a layer puts on analog macros
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageShape:
    """One weight matrix a block contributes, before tp sharding.

    ``shard_axis`` is the Megatron split: ``n`` is column-parallel (the output
    dimension splits), ``k`` is row-parallel (the input dimension splits, and
    the op's partial sums need a tp reduction), ``none`` is replicated.
    """

    op: str
    k: int
    n: int
    shard_axis: str = "n"
    expert: int = -1
    routed: bool = False           # a routed-expert stage (leaves the layer chip when ep > 1)

    def sharded(self, tp: int) -> "StageShape":
        if tp <= 1 or self.shard_axis == "none":
            return self
        dim = self.n if self.shard_axis == "n" else self.k
        if dim % tp != 0:
            raise MappingError(
                "residency",
                f"tp = {tp} does not divide the {self.shard_axis.upper()} dimension "
                f"{dim} of weight matrix {self.op!r} (K x N = {self.k} x {self.n}). "
                "Set parallelism.tp (or mapping.parallelism.tp) to a divisor, or "
                "change the model dimension that produced it.",
            )
        if self.shard_axis == "n":
            return replace(self, n=dim // tp)
        return replace(self, k=dim // tp)


def _attention_stages(device: CimDeviceModel, model=None) -> Tuple[StageShape, ...]:
    shapes = device.per_layer_stage_shapes()
    stages = [
        StageShape("qkv", *shapes["qkv"], shard_axis="n"),
        StageShape("o_proj", *shapes["o_proj"], shard_axis="k"),
    ]
    attention = getattr(model, "attention", None) if model is not None else None
    if attention is not None and bool(getattr(attention, "output_gate", False)):
        # Gated attention (Qwen3.5). The output gate is an ORDINARY weight
        # matrix: W_g maps the block input to one scalar per attention output
        # channel, so its shape is (hidden, num_heads * head_dim) — exactly the
        # extra `hidden_dim * q_size` term llm_util's attention census already
        # counts for `output_gate`, read at (K, N) instead of as a scalar. It
        # is COLUMN-parallel, which splits the query heads exactly the way the
        # o_proj's row split does, so a tp shard holds the gate of the heads it
        # owns and the multiply needs no extra reduction.
        p = device.params
        stages.insert(1, StageShape(
            "attn_gate_proj", p.hidden_dim, p.num_heads * p.head_dim, shard_axis="n"
        ))
    return tuple(stages)


def _ffn_stages(device: CimDeviceModel) -> Tuple[StageShape, ...]:
    shapes = device.per_layer_stage_shapes()
    return (
        StageShape("ffn1", *shapes["ffn1"], shard_axis="n"),
        StageShape("ffn2", *shapes["ffn2"], shard_axis="k"),
    )


def _moe_stages(device: CimDeviceModel) -> Tuple[StageShape, ...]:
    """Router, routed experts and shared experts of one MoE layer.

    The shapes are ``moe_layer_stage_arrays``' own, restated as (K, N) pairs
    per expert instead of as an array total — the same accounting at a finer
    grain, which is what a placement needs.
    """
    p = device.params
    i_moe = p.moe_intermediate
    stages: List[StageShape] = [
        # The router is tiny and every tp shard needs the whole of it.
        StageShape("router", p.hidden_dim, p.num_experts, shard_axis="none"),
    ]
    for expert in range(p.num_experts):
        stages.append(
            StageShape(
                "ffn1_routed", p.hidden_dim, p.ffn1_fold * i_moe,
                shard_axis="n", expert=expert, routed=True,
            )
        )
        stages.append(
            StageShape(
                "ffn2_routed", i_moe, p.hidden_dim,
                shard_axis="k", expert=expert, routed=True,
            )
        )
    # The shared expert may be WIDER than a routed one (Granite-4.0-H-Tiny: 64
    # routed experts at 512 beside one shared MLP at 1024). P2 owns that width.
    i_shared = p.shared_expert_intermediate
    for expert in range(p.n_shared_experts):
        stages.append(
            StageShape(
                "ffn1_shared", p.hidden_dim, p.ffn1_fold * i_shared,
                shard_axis="n", expert=expert,
            )
        )
        stages.append(
            StageShape(
                "ffn2_shared", i_shared, p.hidden_dim,
                shard_axis="k", expert=expert,
            )
        )
    return tuple(stages)


def _hybrid_stages(model, block_kind: str) -> Tuple[StageShape, ...]:
    """Weight MATRICES of one hybrid mixer block (P1 schema, D1).

    These are exactly the matrix terms of ``llm_util``'s raw-parameter census
    for the same block — one accounting of the weights, read at (K, N) instead
    of as a scalar total. The census's remaining terms are NOT matrices and so
    are not tiles: the depthwise conv kernels run on the per-macro pool (ADJ-3)
    and the ``A_log`` / ``D`` / ``dt_bias`` vectors are pool state.
    """
    hidden = int(model.hidden_dim)
    if block_kind == "ssm":
        ssm = model.ssm
        if ssm is None:
            raise MappingError(
                "residency",
                "model_param.layer_plan names an 'ssm' block but model_param.ssm "
                "is absent; the mapper cannot invent the mixer dimensions.",
            )
        d_inner = int(ssm.resolve_d_inner(hidden))
        if str(ssm.variant).lower() == "mamba2":
            in_n = 2 * d_inner + 2 * int(ssm.n_groups) * int(ssm.d_state) + int(ssm.n_heads)
            return (
                StageShape("ssm_in_proj", hidden, in_n, shard_axis="n"),
                StageShape("ssm_out_proj", d_inner, hidden, shard_axis="k"),
            )
        dt_rank = int(ssm.dt_rank)
        return (
            StageShape("ssm_in_proj", hidden, 2 * d_inner, shard_axis="n"),
            StageShape(
                "ssm_x_proj", d_inner, dt_rank + 2 * int(ssm.d_state), shard_axis="none"
            ),
            StageShape("ssm_dt_proj", dt_rank, d_inner, shard_axis="n"),
            StageShape("ssm_out_proj", d_inner, hidden, shard_axis="k"),
        )
    if block_kind == "linear_attn":
        la = model.linear_attention
        if la is None:
            raise MappingError(
                "residency",
                "model_param.layer_plan names a 'linear_attn' block but "
                "model_param.linear_attention is absent.",
            )
        key_dim = int(la.key_dim)
        value_dim = int(la.value_dim)
        qkv_out = 2 * key_dim + value_dim + (value_dim if bool(la.output_gate) else 0)
        gate_out = int(la.num_value_heads) * (2 if bool(la.decay_gate) else 1)
        return (
            StageShape("la_qkv_proj", hidden, qkv_out, shard_axis="n"),
            StageShape("la_gate_proj", hidden, gate_out, shard_axis="none"),
            StageShape("la_out_proj", value_dim, hidden, shard_axis="k"),
        )
    if block_kind == "short_conv":
        conv = model.short_conv
        if conv is None:
            raise MappingError(
                "residency",
                "model_param.layer_plan names a 'short_conv' block but "
                "model_param.short_conv is absent.",
            )
        conv_dim = int(conv.conv_dim)
        return (
            StageShape(
                "conv_in_proj", hidden, int(conv.in_proj_streams) * conv_dim, shard_axis="n"
            ),
            StageShape("conv_out_proj", conv_dim, hidden, shard_axis="k"),
        )
    raise MappingError("residency", f"unknown mixer block kind {block_kind!r}")


# ---------------------------------------------------------------------------
# Placement records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MacroSlot:
    """One enumerated macro slot: the atomic resource, with its named owners."""

    macro_id: int
    chip_id: int
    slot: int                       # index within the chip
    pool: str                       # "analog" | "digital"
    card_id: str
    shard: ShardCoord
    tiles: Tuple[Tile, ...] = ()
    reserved: Optional[str] = None
    analog_device: int = -1
    pool_device: int = -1

    @property
    def owners(self) -> Tuple[TileOwner, ...]:
        seen: "OrderedDict[TileOwner, None]" = OrderedDict()
        for tile in self.tiles:
            seen.setdefault(tile.owner, None)
        return tuple(seen)

    @property
    def held_columns(self) -> int:
        """Stored columns tiles actually hold (their LOGICAL widths)."""
        return sum(tile.logical_columns for tile in self.tiles)

    @property
    def claimed_column_sets(self) -> int:
        return sum(tile.site.num_column_sets for tile in self.tiles)


@dataclass(frozen=True)
class ChipRecord:
    """One chip = one chiplet = one package unit (ADJ-5). The id is an INTEGER."""

    chip_id: int
    label: str
    pool: str                       # "analog" | "digital"
    shard: ShardCoord
    macro_slots: int
    macro_ids: Tuple[int, ...] = ()
    layers: Tuple[int, ...] = ()    # layers resident on this chip (analog only)
    role: str = "backbone"          # "backbone" | "expert_pool" | "shared_digital"


@dataclass(frozen=True)
class FwsDevice:
    """One device in the inventory (A2). ``device_id`` is the IR's DeviceId."""

    device_id: int
    device_class: str
    chip_id: int
    macro_id: int                   # -1 when the device is not macro-hosted
    shard: ShardCoord
    role_key: Tuple[int, int, str]  # (chip index within shard, slot, class)

    @property
    def coord(self) -> DeviceCoord:
        """The WIDENED device coordinate: shard axes + class + host macro.

        ``host_macro`` is present only on a macro-hosted device; asking a
        shared digital chiplet for its host macro raises ``PlacementError``
        naming the axis, which is the honest answer rather than a sentinel.
        """
        coords = dict(self.shard.as_dict())
        coords["dev_class"] = DEVICE_CLASS_CODE[self.device_class]
        if self.macro_id >= 0:
            coords["host_macro"] = int(self.macro_id)
        return DeviceCoord(coords)


@dataclass(frozen=True)
class BoundaryRow:
    """One boundary of the P3.5 table: bytes, rate, and a NAMED verdict.

    A violated boundary is REPORTED, not priced (D17): with no congestion
    model there is nothing to charge it to, so the DAG time stays optimistic
    and :attr:`disclosure` says exactly that.

    The verdict is ``required_time_s > period_s`` — the p2p law's own answer,
    latency included. ``ratio`` is the separate bytes-rate comparison
    (``required_rate / available_rate``) and can sit below 1 on a violated row
    when the link's LATENCY is what does not fit. Two numbers, two questions;
    neither is derived from the other.
    """

    boundary_id: str
    role: str                       # tp | ep | pp | pd | act
    src_chip: int
    dst_chip: int
    bytes_per_unit: float
    unit: str                       # "prefill" | "decode_step"
    label: str
    basis: str
    bandwidth_bytes_per_s: float
    latency_s: float
    required_time_s: float
    crosses_link: bool = True
    period_s: Optional[float] = None
    required_rate_bytes_per_s: Optional[float] = None
    available_rate_bytes_per_s: float = 0.0
    ratio: Optional[float] = None
    violated: bool = False
    disclosure: Optional[str] = None


@dataclass(frozen=True)
class Relaxation:
    """A disclosed relaxation (D21): it renders as a banner, never a footnote."""

    constraint: str
    value: str
    reason: str


# ---------------------------------------------------------------------------
# The mapping object
# ---------------------------------------------------------------------------


class FwsMapping:
    """The placement of one system: tiles, macros, chips, devices, groups.

    Construct with :func:`build_mapping`. The object is a plain record with
    derived views; it computes no time and prices no op.
    """

    def __init__(
        self,
        *,
        device: CimDeviceModel,
        hw_config,
        model_config,
        system_id: str,
        role: str,
        phase: str,
        label: str,
        model_id: str,
        degrees: Mapping[str, int],
        chips: Sequence[ChipRecord],
        macros: Sequence[MacroSlot],
        devices: Sequence[FwsDevice],
        tiles: Sequence[Tile],
        blocks: Sequence[str],
        endpoint_blocks: Sequence[str],
        layer_chip: Mapping[int, Tuple[int, ...]],
        shared_chiplet_suggestion: int,
        decode_window: int,
        notes: Sequence[str],
        relaxations: Sequence[Relaxation],
    ) -> None:
        self.device = device
        self.hw = hw_config
        self.model = model_config
        self.system_id = system_id
        self.role = role
        self.phase = phase
        self.label = label
        self.model_id = model_id
        self.degrees = dict(degrees)
        self.chips = tuple(chips)
        self.macros = tuple(macros)
        self.devices = tuple(devices)
        self.tiles = tuple(tiles)
        self.blocks = tuple(blocks)
        self.endpoint_blocks = tuple(endpoint_blocks)
        self.layer_chip = {int(k): tuple(v) for k, v in layer_chip.items()}
        self.shared_chiplet_suggestion = int(shared_chiplet_suggestion)
        self.decode_window = int(decode_window)
        self.notes = tuple(notes)
        self._relaxations = tuple(relaxations)

        sizes = {axis: max(1, int(self.degrees[axis])) for axis in MAPPING_AXES}
        strides: Dict[str, int] = {}
        span = 1
        for axis in MAPPING_AXES:
            strides[axis] = span
            span *= sizes[axis]
        self._shard_layout = RankLayout(
            axis_order=MAPPING_AXES, axis_sizes=sizes, axis_strides=strides
        )
        self._communicators = CommunicatorFactory(self._shard_layout)
        self._chip_by_id = {chip.chip_id: chip for chip in self.chips}
        self._macro_by_id = {macro.macro_id: macro for macro in self.macros}
        self._device_by_id = {dev.device_id: dev for dev in self.devices}
        self._device_by_role = {
            (self.shard_id(dev.shard), dev.role_key): dev.device_id for dev in self.devices
        }
        self._tiles_by_owner: "OrderedDict[TileOwner, List[Tile]]" = OrderedDict()
        for tile in self.tiles:
            self._tiles_by_owner.setdefault(tile.owner, []).append(tile)
        self._macro_of_tile = {}
        for macro in self.macros:
            for tile in macro.tiles:
                self._macro_of_tile[id(tile)] = macro.macro_id

    # -- the shard grid: membership CONSTRUCTED, never inferred (P3.3) ----

    @property
    def shard_layout(self) -> RankLayout:  # noqa: D401 - see the cache below
        """The (tp, ep, pp) grid the annotations live in.

        This is a ``RankLayout`` in the rewrite's own sense, and the ids it
        linearizes are SHARD ids — the coordinate space, not the device space.
        A device is then found by (shard id, role); membership is therefore
        built from (axis, coords) exactly as ``program/groups.py`` requires,
        and never from a participant count.
        """
        return self._shard_layout

    @property
    def communicators(self) -> CommunicatorFactory:
        """The ONE factory (its memo is why it is built once, not per call)."""
        return self._communicators

    def shard_id(self, shard: ShardCoord) -> int:
        return int(self.shard_layout.linearize(shard.as_dict()))

    def shard_of_id(self, shard_id: int) -> ShardCoord:
        coords = self.shard_layout.coords_of(int(shard_id))
        return ShardCoord(**{axis: int(coords[axis]) for axis in MAPPING_AXES})

    @property
    def primary_group(self) -> str:
        """The axis whose index colors this mapping (ADJ-7).

        The first axis of ``tp, ep, pp`` with a degree above 1; ``tp`` when the
        machine has no parallelism at all. The rule is a property of the FILE,
        not of what is on screen, so a filter never repaints anything.
        """
        for axis in MAPPING_AXES:
            if int(self.degrees.get(axis, 1)) > 1:
                return axis
        return "tp"

    def device_group(self, axis: str, device_id: int) -> GroupKey:
        """The communicator a device belongs to on ``axis`` (P3.3).

        Members are the devices playing the SAME ROLE in the sibling shards
        that ``CommunicatorFactory.members`` enumerates for the anchor's
        coordinates. Nothing is re-derived downstream: this key is what the
        DAG registers and what P4 reads.
        """
        if axis not in MAPPING_AXES:
            raise MappingError("annotation", f"{axis!r} is not one of {MAPPING_AXES}")
        dev = self._device_by_id[int(device_id)]
        anchor = self.shard_id(dev.shard)
        members = []
        for sibling in self.communicators.members((axis,), anchor):
            peer = self._device_by_role.get((int(sibling), dev.role_key))
            if peer is not None:
                members.append(int(peer))
        return GroupKey(axis=canonical_axis_label((axis,)), members=tuple(sorted(members)))

    # -- lookups ----------------------------------------------------------

    def chip(self, chip_id: int) -> ChipRecord:
        return self._chip_by_id[int(chip_id)]

    def macro(self, macro_id: int) -> MacroSlot:
        return self._macro_by_id[int(macro_id)]

    def device_record(self, device_id: int) -> FwsDevice:
        return self._device_by_id[int(device_id)]

    def macro_of_tile(self, tile: Tile) -> int:
        return int(self._macro_of_tile[id(tile)])

    def tiles_for(self, owner: TileOwner) -> Tuple[Tile, ...]:
        return tuple(self._tiles_by_owner.get(owner, ()))

    def owners(self) -> Tuple[TileOwner, ...]:
        return tuple(self._tiles_by_owner)

    def analog_chips(self) -> Tuple[ChipRecord, ...]:
        return tuple(chip for chip in self.chips if chip.pool == "analog")

    def digital_chips(self) -> Tuple[ChipRecord, ...]:
        return tuple(chip for chip in self.chips if chip.pool == "digital")

    def devices_of_class(self, device_class: str) -> Tuple[FwsDevice, ...]:
        return tuple(dev for dev in self.devices if dev.device_class == device_class)

    def shared_digital_devices(self, shard: Optional[ShardCoord] = None) -> Tuple[FwsDevice, ...]:
        devs = self.devices_of_class("shared_digital")
        if shard is None:
            return devs
        same = tuple(dev for dev in devs if dev.shard.tp == shard.tp)
        return same or devs

    # -- derived reports --------------------------------------------------

    def macro_occupancy(self, macro_id: int) -> float:
        macro = self.macro(macro_id)
        stored = self._stored_columns(macro)
        if stored <= 0:
            return 0.0
        return macro.held_columns / stored

    def chip_occupancy(self, chip_id: int) -> float:
        chip = self.chip(chip_id)
        if chip.macro_slots <= 0:
            return 0.0
        used = sum(1 for mid in chip.macro_ids if self.macro(mid).tiles)
        return used / chip.macro_slots

    def _stored_columns(self, macro: MacroSlot) -> int:
        if macro.pool == "digital":
            return int(self.device.digital_card.fabric.cols)
        card = self.device.card
        return int(card.stored_columns_per_set) * int(card.column_sets_per_macro)

    def unowned_macro_slots(self) -> int:
        """Enumerated macro slots holding no tile. Legal, and reported (ADJ-5)."""
        return sum(1 for macro in self.macros if macro.pool == "analog" and not macro.tiles)

    def unowned_columns(self) -> int:
        """Stored analog columns no tile holds — the granularity that is paid for.

        Counts BOTH kinds of gap: a macro slot nothing claims, and the columns
        of a claimed column set that the tile's logical width does not reach.
        A mux slot is the smallest allocatable unit (ADJ-4), so the second kind
        cannot be handed to another owner; it is unowned all the same, and
        rounding it away would be the AUDIT finding-1 sin at column scale.
        """
        total = 0
        for macro in self.macros:
            if macro.pool != "analog":
                continue
            total += self._stored_columns(macro) - macro.held_columns
        return int(total)

    def summary(self) -> Dict[str, object]:
        analog = self.analog_chips()
        return OrderedDict(
            (
                ("system", self.system_id),
                ("role", self.role),
                ("phase", self.phase),
                ("parallelism", dict(self.degrees)),
                ("primary_group", self.primary_group),
                ("analog_chips", len(analog)),
                ("shared_digital_chiplets", len(self.digital_chips())),
                ("shared_digital_suggestion", self.shared_chiplet_suggestion),
                ("macro_slots", sum(chip.macro_slots for chip in self.chips)),
                ("analog_macro_slots", sum(chip.macro_slots for chip in analog)),
                ("macros_holding_tiles", sum(1 for m in self.macros if m.tiles)),
                ("unowned_macro_slots", self.unowned_macro_slots()),
                ("unowned_columns", self.unowned_columns()),
                ("tiles", len(self.tiles)),
                ("devices", len(self.devices)),
                ("owners", len(self._tiles_by_owner)),
                ("decode_window", self.decode_window),
            )
        )

    def relaxations(self, boundary_rows: Sequence[BoundaryRow] = ()) -> Tuple[Relaxation, ...]:
        """Every relaxation this mapping carries, disclosed (D21, AUDIT 5)."""
        out = list(self._relaxations)
        if any(row.violated for row in boundary_rows):
            names = ", ".join(row.boundary_id for row in boundary_rows if row.violated)
            out.append(
                Relaxation(
                    constraint="boundary_bandwidth",
                    value=f"violated and NOT priced: {names}",
                    reason=(
                        "D17 fixes the network at fully-connected p2p with no congestion "
                        "model, so a boundary whose required rate exceeds the declared "
                        "link rate is REPORTED, not charged. Every DAG time that crosses "
                        "one of these boundaries is optimistic by an amount this model "
                        "cannot compute."
                    ),
                )
            )
        return tuple(out)

    # -- the boundary table (P3.5) ---------------------------------------

    def boundary_table(
        self,
        *,
        period_s: Optional[float] = None,
        period_basis: str = "",
        tokens: Optional[int] = None,
    ) -> Tuple[BoundaryRow, ...]:
        """Per-boundary bytes and rates against the DECLARED link bandwidth.

        ``period_s`` is the time base a RATE needs. P3 has no timeline (A1: the
        DAG is P4's to price), so the period is an INPUT with a stated
        ``period_basis``; without one the rows still carry bytes and the link
        time they require, and every rate field stays ``None`` rather than
        being invented.
        """
        if period_s is not None and not period_basis:
            raise MappingError(
                "package",
                "boundary_table(period_s=...) needs period_basis: a rate whose time "
                "base is unnamed is a number nobody can check (D21).",
            )
        dev = self.device
        precision = self.hw.sw_config.precision
        act_bytes = float(precision.activations)
        layout = self.hw.network_layout
        batch = int(dev.params.batch_size)
        seq_len = int(dev.params.seq_len)
        decode_len = int(getattr(self.model, "decode_len", 0) or 0)
        prefill_len = max(0, seq_len - decode_len)
        if tokens is None:
            tokens = batch * prefill_len if self.phase == "prefill" else batch
        unit = "prefill" if self.phase == "prefill" else "decode_step"
        rows: List[BoundaryRow] = []

        # 1. chip-boundary activations, one row per consecutive chip pair
        #    inside a shard (act, ADJ-5: a chip boundary is never called pp).
        pp_bw, pp_lat = layout.link_for_parallelism("pp")
        boundary_bytes = dev.boundary_bytes(tokens, act_bytes)
        for shard_chips in self._backbone_chips_by_tp_shard().values():
            for src, dst in zip(shard_chips, shard_chips[1:]):
                rows.append(
                    self._row(
                        boundary_id=f"act.c{src.chip_id}->c{dst.chip_id}",
                        role="act",
                        src=src.chip_id,
                        dst=dst.chip_id,
                        size=boundary_bytes,
                        unit=unit,
                        label="activation handoff across the chip boundary",
                        basis=(
                            f"cim_timing.boundary_bytes(tokens = {tokens}, act_bytes = "
                            f"{act_bytes}) = tokens x hidden {dev.params.hidden_dim} x act"
                        ),
                        bw=pp_bw,
                        lat=pp_lat,
                        period_s=period_s,
                    )
                )

        # 2. tp collectives: the row-parallel stages (o_proj, ffn2) leave
        #    partial sums that a tp all-reduce sums.
        tp = int(self.degrees.get("tp", 1))
        if tp > 1:
            tp_bw, tp_lat = layout.link_for_parallelism("tp")
            payload = float(tokens) * float(dev.params.hidden_dim) * act_bytes
            ring = 2.0 * (tp - 1) / tp
            for shard_chips in self._backbone_chips_by_tp_shard().values():
                for chip in shard_chips:
                    peers = self._tp_siblings(chip)
                    if not peers:
                        continue
                    reductions = 2 * len(chip.layers)  # o_proj and ffn2 per layer
                    size = payload * ring * reductions
                    rows.append(
                        self._row(
                            boundary_id=f"tp.c{chip.chip_id}",
                            role="tp",
                            src=chip.chip_id,
                            dst=peers[0].chip_id,
                            size=size,
                            unit=unit,
                            label=f"tp all-reduce of {reductions} row-parallel stages",
                            basis=(
                                f"ring all-reduce volume per link 2(tp-1)/tp = {ring:.6g} x "
                                f"payload {payload:.6g} B x {reductions} row-parallel stages "
                                "(o_proj and ffn2 of every resident layer). D17 fixes the "
                                "fabric at fully-connected p2p, so this is the collective's "
                                "declared per-link volume and not a topology claim."
                            ),
                            bw=tp_bw,
                            lat=tp_lat,
                            period_s=period_s,
                        )
                    )

        # 3. expert dispatch and combine (ep), only when the experts left the
        #    layer chip. With ep = 1 the routed experts sit on the layer chip
        #    and there is NO boundary — which the report says out loud.
        ep = int(self.degrees.get("ep", 1))
        if ep > 1 and dev.params.use_moe:
            ep_bw, ep_lat = layout.link_for_parallelism("ep")
            dispatch = dev.moe_dispatch_bytes(tokens, act_bytes)
            for chip in self.analog_chips():
                if chip.role != "expert_pool":
                    continue
                source = self._layer_chip_for_expert_chip(chip)
                if source is None:
                    continue
                for direction in ("dispatch", "combine"):
                    rows.append(
                        self._row(
                            boundary_id=f"ep.{direction}.c{source}->c{chip.chip_id}",
                            role="ep",
                            src=source if direction == "dispatch" else chip.chip_id,
                            dst=chip.chip_id if direction == "dispatch" else source,
                            size=dispatch / ep,
                            unit=unit,
                            label=f"MoE expert {direction}",
                            basis=(
                                f"cim_timing.moe_dispatch_bytes(tokens = {tokens}) = tokens x "
                                f"top_k {dev.params.top_k} x hidden {dev.params.hidden_dim} x "
                                f"act, spread over ep = {ep} expert chips"
                            ),
                            bw=ep_bw,
                            lat=ep_lat,
                            period_s=period_s,
                        )
                    )

        # 4. KV traffic. It crosses no p2p link in either KV story (cim_sram
        #    is the chip's own activation SRAM; cim_dram is a per-chip DRAM
        #    tier), so it is checked against the KV TIER's declared bandwidth
        #    and marked as not crossing a link. One accounting, named.
        kv_rows = self._kv_rows(unit=unit, period_s=period_s)
        rows.extend(kv_rows)
        return tuple(rows)

    def _kv_rows(self, *, unit: str, period_s: Optional[float]) -> Tuple[BoundaryRow, ...]:
        dev = self.device
        inference = getattr(self.hw, "inference_config", None)
        story = str(getattr(inference, "kvcache_type", "hbm_only") or "hbm_only").lower()
        if story not in ("cim_sram", "cim_dram") or dev.params.is_vit_shaped:
            return ()
        precision = self.hw.sw_config.precision
        kv_precision = float(precision.kv_cache)
        tech_dram = self.hw.tech_config.DRAM
        bandwidth = dev.kv_story_bandwidth(story, float(getattr(tech_dram, "bandwidth", 0.0) or 0.0))
        seq_len = int(dev.params.seq_len)
        decode_len = int(getattr(self.model, "decode_len", 0) or 0)
        # Prefill WRITES the KV of the tokens it consumes; a decode step READS
        # the whole context. Same law, two operating points — and the label
        # says which one, because they are not the same number.
        context = max(1, seq_len - decode_len) if self.phase == "prefill" else seq_len
        traffic = "write" if self.phase == "prefill" else "read"
        tp = max(1, int(self.degrees.get("tp", 1)))
        rows = []
        for chip in self.analog_chips():
            if chip.role != "backbone" or not chip.layers:
                continue
            per_layer = dev.kv_read_bytes(context, kv_precision, dev.params.batch_size, tp)
            size = per_layer * len(chip.layers)
            rows.append(
                self._row(
                    boundary_id=f"kv.c{chip.chip_id}",
                    role="act",
                    src=chip.chip_id,
                    dst=chip.chip_id,
                    size=size,
                    unit=unit,
                    label=f"KV {traffic} traffic of {len(chip.layers)} resident layers",
                    basis=(
                        f"cim_timing.kv_read_bytes(context = {context}, tp = {tp}) x "
                        f"{len(chip.layers)} resident layers, against the {story} tier's "
                        "declared bandwidth. KV crosses no p2p link in either KV story."
                    ),
                    bw=bandwidth,
                    lat=0.0,
                    period_s=period_s,
                    crosses_link=False,
                )
            )
        return tuple(rows)

    def _row(
        self,
        *,
        boundary_id: str,
        role: str,
        src: int,
        dst: int,
        size: float,
        unit: str,
        label: str,
        basis: str,
        bw: float,
        lat: float,
        period_s: Optional[float],
        crosses_link: bool = True,
    ) -> BoundaryRow:
        required_time = CimDeviceModel.p2p_time_s(size, bw, lat)
        required_rate = None
        ratio = None
        violated = False
        disclosure = None
        if period_s is not None and period_s > 0:
            required_rate = float(size) / float(period_s)
            ratio = required_rate / bw if bw > 0 else float("inf")
            violated = required_time > period_s
            if violated:
                disclosure = (
                    f"boundary {boundary_id} needs {required_rate:.6g} B/s but the "
                    f"declared link carries {bw:.6g} B/s (ratio {ratio:.4g}); the p2p law "
                    f"needs {required_time:.6g} s against a period of {period_s:.6g} s, "
                    "latency included. D17 gives this model no congestion term, so the "
                    "excess is REPORTED and NOT priced: the DAG time is optimistic here."
                )
        return BoundaryRow(
            boundary_id=boundary_id,
            role=role,
            src_chip=int(src),
            dst_chip=int(dst),
            bytes_per_unit=float(size),
            unit=unit,
            label=label,
            basis=basis,
            bandwidth_bytes_per_s=float(bw),
            latency_s=float(lat),
            required_time_s=float(required_time),
            crosses_link=bool(crosses_link),
            period_s=period_s,
            required_rate_bytes_per_s=required_rate,
            available_rate_bytes_per_s=float(bw),
            ratio=ratio,
            violated=violated,
            disclosure=disclosure,
        )

    def _backbone_chips_by_tp_shard(self) -> "OrderedDict[int, Tuple[ChipRecord, ...]]":
        """Backbone chips of each tp shard, in chip order — the chip CHAIN.

        Keyed by the tp index and not by the whole shard id on purpose: pp
        annotates that chain, it does not cut it (ADJ-5). Grouping by the full
        coordinate would hide the boundary between two pp stages, which is the
        one boundary a pipeline most needs drawn.
        """
        by_shard: "OrderedDict[int, List[ChipRecord]]" = OrderedDict()
        for chip in self.analog_chips():
            if chip.role != "backbone":
                continue
            by_shard.setdefault(int(chip.shard.tp), []).append(chip)
        return OrderedDict((k, tuple(v)) for k, v in by_shard.items())

    def _tp_siblings(self, chip: ChipRecord) -> Tuple[ChipRecord, ...]:
        groups = self._backbone_chips_by_tp_shard()
        position = [c.chip_id for c in groups[int(chip.shard.tp)]].index(chip.chip_id)
        out = []
        for sibling in self.communicators.members(("tp",), self.shard_id(chip.shard)):
            tp_index = self.shard_of_id(int(sibling)).tp
            if tp_index == chip.shard.tp:
                continue
            peers = groups.get(int(tp_index))
            if peers and position < len(peers):
                out.append(peers[position])
        return tuple(out)

    def _layer_chip_for_expert_chip(self, chip: ChipRecord) -> Optional[int]:
        if not chip.layers:
            return None
        layer = chip.layers[0]
        for candidate in self.analog_chips():
            if candidate.role == "backbone" and layer in candidate.layers:
                if candidate.shard.tp == chip.shard.tp:
                    return candidate.chip_id
        return None

    # -- the text report --------------------------------------------------

    def report(self, boundary_rows: Sequence[BoundaryRow] = ()) -> str:
        """The mapping as text: placement, occupancy, boundaries, disclosures."""
        summary = self.summary()
        lines = [
            f"[FWS-CIM] mapping '{self.label}' (P3): {summary['analog_chips']} analog chips, "
            f"{summary['shared_digital_chiplets']} shared digital chiplets, "
            f"{summary['tiles']} tiles on {summary['macros_holding_tiles']} of "
            f"{summary['analog_macro_slots']} analog macro slots",
            f"[FWS-CIM]   parallelism tp/ep/pp = "
            f"{self.degrees['tp']}/{self.degrees['ep']}/{self.degrees['pp']}; "
            f"primary group axis = {self.primary_group} (ADJ-7)",
            f"[FWS-CIM]   unowned: {summary['unowned_macro_slots']} macro slots, "
            f"{summary['unowned_columns']} stored columns "
            "(legal and reported, ADJ-5)",
            f"[FWS-CIM]   shared digital chiplets: declared "
            f"{summary['shared_digital_chiplets']}, derived suggestion "
            f"{self.shared_chiplet_suggestion} (ADJ-5: the count is a config input; "
            "the suggestion is one chiplet per analog chip, which is the concurrency "
            "the closed-form spatial pipeline assumes)",
        ]
        for note in self.notes:
            lines.append(f"[FWS-CIM]   note: {note}")
        for chip in self.analog_chips():
            lines.append(
                f"[FWS-CIM]   chip {chip.chip_id} ({chip.role}, tp{chip.shard.tp}/"
                f"ep{chip.shard.ep}/pp{chip.shard.pp}): layers {list(chip.layers)}, "
                f"{sum(1 for m in chip.macro_ids if self.macro(m).tiles)}/{chip.macro_slots} "
                f"slots owned, occupancy {self.chip_occupancy(chip.chip_id):.3f}"
            )
        for row in boundary_rows:
            verdict = "VIOLATED" if row.violated else "ok"
            where = "p2p link" if row.crosses_link else "on-chip tier (crosses no link)"
            lines.append(
                f"[FWS-CIM]   boundary {row.boundary_id} ({row.role}): "
                f"{row.bytes_per_unit:.6g} B per {row.unit}, needs "
                f"{row.required_time_s * 1e6:.6f} us on a "
                f"{row.bandwidth_bytes_per_s:.6g} B/s {where} -> {verdict}"
            )
            if row.disclosure:
                lines.append(f"[FWS-CIM]   [NOTE] {row.disclosure}")
        for relaxation in self.relaxations(boundary_rows):
            lines.append(
                f"[FWS-CIM]   [RELAXATION] {relaxation.constraint} = {relaxation.value}: "
                f"{relaxation.reason}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_tile_row_convention(tiles: Sequence[Tile], rows: int) -> None:
    """Pin and check the ``TileSite`` row convention (the Wave A leftover).

    A site's row range is the half-open K range of the owner's weight matrix
    it holds. Anything else — a physical row extent, a 0-based placeholder, a
    range wider than the card's rows — is refused by name here rather than
    silently drawn later.
    """
    for tile in tiles:
        site = tile.site
        if site.row_start != tile.k_start or site.row_end != tile.k_end:
            raise MappingError(
                "residency",
                f"tile {tile.owner.label} on macro {site.macro_id} declares site rows "
                f"[{site.row_start}, {site.row_end}) but the tile holds matrix rows "
                f"[{tile.k_start}, {tile.k_end}). The pinned convention is "
                f"{TILE_ROW_CONVENTION!r}: a site's row range IS the owner matrix's "
                "K range.",
            )
        if not (0 <= site.row_start < site.row_end):
            raise MappingError(
                "residency",
                f"tile {tile.owner.label} on macro {site.macro_id} has an empty or "
                f"negative row range [{site.row_start}, {site.row_end}).",
            )
        if site.row_end - site.row_start > int(rows):
            raise MappingError(
                "residency",
                f"tile {tile.owner.label} on macro {site.macro_id} spans "
                f"{site.row_end - site.row_start} rows but the card stores {rows}.",
            )


# ---------------------------------------------------------------------------
# The builder: (workload, platform) -> mapping
# ---------------------------------------------------------------------------


def _shard_shapes(device: CimDeviceModel, model, layer_idx: int, tp: int) -> Tuple[StageShape, ...]:
    """Weight matrices layer ``layer_idx`` puts on analog macros, tp-sharded."""
    mixers = tuple(getattr(model, "layer_mixers", ())) or (("attention",),)
    kinds = mixers[layer_idx] if layer_idx < len(mixers) else mixers[-1]
    stages: List[StageShape] = []
    for kind in kinds:
        if kind == "attention":
            stages.extend(_attention_stages(device, model))
        elif kind in ("ffn", "moe"):
            pass  # handled by the FFN half below
        else:
            stages.extend(_hybrid_stages(model, kind))
    if device.layer_class_mask()[layer_idx]:
        stages.extend(_moe_stages(device))
    else:
        stages.extend(_ffn_stages(device))
    return tuple(stage.sharded(tp) for stage in stages)


def _endpoint_shapes(device: CimDeviceModel, tp: int) -> Tuple[Tuple[str, StageShape], ...]:
    """(position, stage) for the model-shaped endpoints; position is head/tail."""
    p = device.params
    out: List[Tuple[str, StageShape]] = []
    if p.is_vit_shaped:
        if p.patch_dim > 0:
            out.append(("head", StageShape("patch_embed", p.patch_dim, p.hidden_dim, "n")))
        if p.num_classes > 0:
            out.append(("tail", StageShape("vit_head", p.hidden_dim, p.num_classes, "k")))
    elif p.lm_head_enabled:
        out.append(("tail", StageShape("lm_head", p.hidden_dim, p.vocab_size, "n")))
    return tuple((where, stage.sharded(tp)) for where, stage in out)


def _apply_user_allocation(
    tiles: Sequence[Tile], entries: Sequence[object], owner_label: str
) -> Tuple[Tile, ...]:
    """Override the enumerator's SITE with the user's `cim.allocation` entries.

    The config declares a macro and a set of mux slots per tile; it declares no
    K range, so the enumerator's rows (the pinned convention) are kept and only
    the site's macro and column sets move. Entries for one owner are consumed
    in declaration order against the enumerator's block order — the only rule
    that makes a row/column-partitioned matrix expressible at all — and a count
    mismatch is refused by name.
    """
    if len(entries) != len(tiles):
        raise MappingError(
            "residency",
            f"cim.allocation names {len(entries)} placement(s) for {owner_label} but "
            f"the matrix tiles into {len(tiles)}. Entries for one owner are consumed in "
            "declaration order against the enumerator's block order, so the counts must "
            "match exactly.",
        )
    placed = []
    for index, (tile, entry) in enumerate(zip(tiles, entries)):
        declared_slice = getattr(entry, "slice_index", None)
        if declared_slice is not None and int(declared_slice) != int(tile.slice_index):
            # A DECLARED slice is a claim about which tile this entry places, and
            # declaration order is what actually decides. When the two disagree
            # the user's intent is unknowable, so it is refused rather than
            # silently overruled (the field used to be parsed and dropped).
            raise MappingError(
                "residency",
                f"cim.allocation entry {index} for {owner_label} declares "
                f"slice_index = {int(declared_slice)} but declaration order places it on "
                f"slice {int(tile.slice_index)}. Entries for one owner are consumed in "
                "declaration order against the enumerator's block order; reorder the "
                "entries or drop slice_index, which is unconstrained when absent.",
            )
        placed.append(
            replace(
                tile,
                site=TileSite(
                    macro_id=int(entry.macro),
                    row_start=tile.k_start,
                    row_end=tile.k_end,
                    column_sets=tuple(entry.column_sets),
                ),
            )
        )
    return tuple(placed)


def build_mapping(
    hw_config,
    model_config,
    *,
    spec=None,
    system_id: str = "sys.fws",
    role: str = "unified",
    phase: Optional[str] = None,
    label: Optional[str] = None,
    model_id: Optional[str] = None,
) -> FwsMapping:
    """``(workload, platform) -> mapping`` — the seam P3 fixes now.

    ``spec`` is a ``config.MappingSystemConfig``; ``None`` takes the hardware
    config's ``mapping:`` block, and an ABSENT block derives the DEDICATED
    mapping, which reproduces today's ``cim.chip.layers_per_chip`` placement.
    Config-declared allocation is the first implementation of this signature;
    the ``layers_per_chip: auto`` packer is the second; automatic search is a
    later third (D10). The workload argument is a single model today and the
    signature takes a platform, never a count.
    """
    import config as _config

    model = getattr(model_config, "model_config", model_config)
    device = CimDeviceModel(hw_config, model)
    if spec is None:
        mapping_config = getattr(hw_config, "mapping_config", None)
        spec = (
            mapping_config.system
            if mapping_config is not None
            else _config.MappingSystemConfig()
        )
    p = device.params
    decode_len = int(getattr(model, "decode_len", 0) or 0)
    if phase is None:
        phase = "decode" if decode_len > 0 else "prefill"
    if phase not in ("prefill", "decode"):
        raise MappingError("execution", f"phase must be 'prefill' or 'decode' (got {phase!r})")
    # The model's own name wins, then the config's declared name, then the
    # PRICING CARRIER as the last resort (which is what this was before a model
    # could declare a name, so nothing already shipped moves).
    model_id = (
        model_id
        or str(getattr(model, "model_id", "") or "")
        or str(getattr(model, "model_type", "model"))
    )
    # The label names the MODEL, not the carrier: a page titled "llama decode
    # mapping" over Granite-4.0-H-Tiny tiles tells a reader the wrong model.
    label = label or f"{model_id} {phase} mapping"
    notes: List[str] = []
    relaxations: List[Relaxation] = [
        Relaxation(
            constraint="network_congestion",
            value="not modeled",
            reason=(
                "A1 / D17: the fabric is fully-connected p2p, analytical. Boundary "
                "bytes are placed and reported; no queueing, routing or topology term "
                "exists anywhere in this mapping."
            ),
        ),
        Relaxation(
            constraint="intra_chip_activation_movement",
            value="a dependency, not priced bytes",
            reason=(
                "D17 declares one network law and it is the p2p fabric BETWEEN chips. "
                "Activations moving between macros of one chip are carried as DAG "
                "dependencies with their byte count annotated and no link charge: no "
                "on-chip interconnect law is declared anywhere, and inventing one would "
                "be a number no card supports."
            ),
        ),
        Relaxation(
            constraint="op_durations",
            value="unpriced (0.0)",
            reason=(
                "P3 places; P4 prices (A1). Every op in the emitted DAG carries a zero "
                "duration and the annotation P4's pricing table needs. A number here "
                "would be a second accounting of a metric P4 owns (D21)."
            ),
        ),
    ]
    carrier = str(getattr(model, "model_type", "") or "")
    if carrier and carrier != model_id:
        # D21's named-owner invariant is about a name a reader can TRUST. When
        # the pricing carrier and the model's own id differ (Granite-4.0-H-Tiny
        # declares model_type: llama for the SwiGLU gated MLP), every tile owner
        # and every label reads the id, and the carrier is disclosed here rather
        # than left to contradict it.
        relaxations.append(
            Relaxation(
                constraint="model_type_carrier",
                value=f"model_id {model_id!r}, model_param.model_type {carrier!r}",
                reason=(
                    f"This model is priced through the {carrier!r} carrier: model_type "
                    "selects the FFN/attention arithmetic, not the model's identity. "
                    f"Owners, labels and the atlas all name {model_id!r}; the carrier is "
                    "named here so the two spellings are reconciled and not read as a "
                    "contradiction."
                ),
            )
        )

    # --- degrees, checked against the settings that already carry them ----
    sch = hw_config.sch_config
    tp_hw = max(1, int(getattr(sch, "tp", 1) or 1))
    tp = int(spec.parallelism.tp) if spec.parallelism.tp is not None else tp_hw
    if spec.parallelism.tp is not None and tp != tp_hw:
        raise MappingError(
            "annotation",
            f"mapping.parallelism.tp = {tp} disagrees with parallelism.tp = {tp_hw}. "
            "One degree, one setting (D21); change the hardware parallelism block or "
            "drop the mapping override.",
        )
    ep_hw = device.moe_expert_parallel
    ep = int(spec.parallelism.ep) if spec.parallelism.ep is not None else ep_hw
    if spec.parallelism.ep is not None and ep != ep_hw:
        raise MappingError(
            "annotation",
            f"mapping.parallelism.ep = {ep} disagrees with cim.chip.moe_expert_parallel "
            f"= {ep_hw}. One degree, one setting (D21).",
        )
    pp = int(spec.parallelism.pp) if spec.parallelism.pp is not None else 1
    if pp > 1:
        notes.append(
            f"mapping.parallelism.pp = {pp} annotates chips only. ADJ-5 makes pp an "
            "INDEPENDENT annotation (a chip index is not implicitly a pipeline stage), "
            "and the hardware parallelism.pp stays 1, which the fws_cim run path "
            "requires. The two axes share a name and nothing else."
        )
    degrees = {"tp": tp, "ep": ep, "pp": pp}

    # --- layer -> chip ----------------------------------------------------
    if spec.layers_per_chip is None:
        counts = device.chip_layer_counts()
        placement_source = "cim.chip.layers_per_chip"
    else:
        counts = _layer_counts_from_spec(spec.layers_per_chip, device)
        placement_source = "mapping.layers_per_chip"
    backbone_per_shard = len(counts)
    layers_of_chip: List[Tuple[int, ...]] = []
    cursor = 0
    for count in counts:
        layers_of_chip.append(tuple(range(cursor, cursor + count)))
        cursor += count

    # --- macro slots per chip --------------------------------------------
    declared_slots = int(getattr(device.chip, "arrays_per_chip", 0) or 0)
    if spec.macros_per_chip is not None:
        macro_slots = int(spec.macros_per_chip)
    elif declared_slots > 0:
        macro_slots = declared_slots
    else:
        macro_slots = 0  # derived below, once the placement is known
    if spec.macros_per_chip is not None and declared_slots > 0 and macro_slots != declared_slots:
        raise MappingError(
            "assembly",
            f"mapping.macros_per_chip = {macro_slots} disagrees with "
            f"cim.chip.arrays_per_chip = {declared_slots}. A macro slot count has one "
            "home (D21).",
        )

    # --- expert pool ------------------------------------------------------
    num_moe_layers = p.num_moe_layers if p.use_moe else 0
    expert_chips_per_shard = num_moe_layers * ep if ep > 1 and num_moe_layers else 0
    if ep > 1 and p.use_moe and p.num_experts % ep != 0:
        raise MappingError(
            "assembly",
            f"cim.chip.moe_expert_parallel = {ep} does not divide num_experts = "
            f"{p.num_experts}: an expert chip would hold a fraction of an expert.",
        )

    # --- pp membership ----------------------------------------------------
    analog_chips_per_shard = backbone_per_shard + expert_chips_per_shard
    total_analog = tp * analog_chips_per_shard
    pp_of_chip = _pp_membership(spec, pp, tp, backbone_per_shard, expert_chips_per_shard)

    if spec.chips is not None and int(spec.chips) != total_analog:
        raise MappingError(
            "assembly",
            f"mapping.chips = {spec.chips} but the placement enumerates {total_analog} "
            f"analog chips ({tp} tp shard(s) x [{backbone_per_shard} backbone + "
            f"{expert_chips_per_shard} expert-pool] chips from {placement_source}). "
            "Chips are enumerated objects, never a quotient (D21).",
        )

    # --- residency: place the tiles --------------------------------------
    card = device.card
    rows = int(card.params.rows)
    allocation_entries = _allocation_by_owner(device)
    if allocation_entries and macro_slots <= 0:
        raise MappingError(
            "residency",
            "cim.allocation names macro ids, which are GLOBAL slot ids in this "
            "placement, but no slot count is declared (cim.chip.arrays_per_chip = 0 "
            "and mapping.macros_per_chip is absent). Declare one so a macro id means "
            "the same thing to the config and to the mapper.",
        )
    chips: List[ChipRecord] = []
    macros: List[MacroSlot] = []
    all_tiles: List[Tile] = []
    layer_chip: Dict[int, List[int]] = {}
    next_chip_id = 0
    macro_usage: Dict[int, int] = {}
    auto_cursor: Dict[int, int] = {}

    def _place(owner: TileOwner, stage: StageShape, chip_id: int, chip_macro_base: int):
        """Enumerate one matrix's tiles onto the chip's next free macros.

        Two cursors, on purpose: ``auto_cursor`` is where the ENUMERATOR packs
        next, and ``macro_usage`` is the highest slot any tile reached. A user
        allocation that jumps to slot 119 must not drag the enumerator's cursor
        with it — the chip would then "need" 220 slots to hold 105 tiles. Two
        tiles landing on one column set is caught by P2's capacity check with
        its own named error, which is where that failure belongs.
        """
        used = auto_cursor.get(chip_id, 0)
        tiles = device.enumerate_tiles(
            stage.k, stage.n, owner, first_macro_id=chip_macro_base + used
        )
        entries = allocation_entries.get(_owner_key(owner))
        if entries:
            tiles = _apply_user_allocation(tiles, entries, owner.label)
            for tile in tiles:
                if not (chip_macro_base <= tile.site.macro_id < chip_macro_base + macro_slots):
                    raise MappingError(
                        "residency",
                        f"cim.allocation puts {owner.label} on macro "
                        f"{tile.site.macro_id}, which is outside chip {chip_id}'s slot "
                        f"range [{chip_macro_base}, {chip_macro_base + macro_slots}). A "
                        "tile lives on the chip its owner's layer is assigned to.",
                    )
        validate_tile_row_convention(tiles, rows)
        reach = max(tile.site.macro_id for tile in tiles) - chip_macro_base + 1
        if not entries:
            auto_cursor[chip_id] = max(used, reach)
        macro_usage[chip_id] = max(macro_usage.get(chip_id, 0), reach)
        all_tiles.extend(tiles)
        return tiles

    moe_layer_indices = [idx for idx, is_moe in enumerate(device.layer_class_mask()) if is_moe]
    tiles_by_chip: Dict[int, List[Tile]] = {}
    macro_base_of_chip: Dict[int, int] = {}
    chip_specs: List[Tuple[int, str, ShardCoord, Tuple[int, ...], str]] = []
    for tp_idx in range(tp):
        for chip_index in range(backbone_per_shard):
            shard = ShardCoord(tp=tp_idx, ep=0, pp=pp_of_chip[len(chip_specs)])
            chip_specs.append(
                (next_chip_id, f"analog chiplet {chip_index}, tp shard {tp_idx}", shard,
                 layers_of_chip[chip_index], "backbone")
            )
            next_chip_id += 1
        for expert_slot in range(expert_chips_per_shard):
            layer = moe_layer_indices[expert_slot // ep]
            ep_index = expert_slot % ep
            shard = ShardCoord(tp=tp_idx, ep=ep_index, pp=pp_of_chip[len(chip_specs)])
            chip_specs.append(
                (next_chip_id, f"expert-pool chiplet {expert_slot}, tp shard {tp_idx}",
                 shard, (layer,), "expert_pool")
            )
            next_chip_id += 1

    endpoints = _endpoint_shapes(device, tp)
    #: Chips are enumerated slot ranges: chip i owns macro ids
    #: [i * macro_slots, (i+1) * macro_slots). When the slot count is derived
    #: (arrays_per_chip = 0) it is not known yet, so placement runs on a
    #: per-chip LOCAL base and the ids are shifted into their ranges below.
    slots_known = macro_slots > 0
    for chip_index, (chip_id, chip_label, shard, chip_layers, chip_role) in enumerate(chip_specs):
        chip_macro_base = chip_index * macro_slots if slots_known else chip_index * _LOCAL_BASE_STRIDE
        macro_base_of_chip[chip_id] = chip_macro_base
        macro_usage[chip_id] = 0
        auto_cursor[chip_id] = 0
        tiles_by_chip[chip_id] = []
        shard_index = int(_linearize(degrees, shard))
        chip_index_in_shard = [c[0] for c in chip_specs if c[2].tp == shard.tp].index(chip_id)
        if chip_role == "backbone":
            if chip_index_in_shard == 0:
                for where, stage in endpoints:
                    if where != "head":
                        continue
                    owner = TileOwner(model_id, 0, stage.op, -1, shard_index)
                    tiles_by_chip[chip_id].extend(_place(owner, stage, chip_id, chip_macro_base))
            for layer in chip_layers:
                layer_chip.setdefault(layer, []).append(chip_id)
                for stage in _shard_shapes(device, model, layer, tp):
                    if stage.routed and ep > 1:
                        continue  # the routed experts left for the expert pool
                    owner = TileOwner(model_id, layer, stage.op, stage.expert, shard_index)
                    tiles_by_chip[chip_id].extend(_place(owner, stage, chip_id, chip_macro_base))
            if chip_index_in_shard == backbone_per_shard - 1:
                for where, stage in endpoints:
                    if where != "tail":
                        continue
                    owner = TileOwner(model_id, p.num_layers - 1, stage.op, -1, shard_index)
                    tiles_by_chip[chip_id].extend(_place(owner, stage, chip_id, chip_macro_base))
        else:
            layer = chip_layers[0]
            layer_chip.setdefault(layer, []).append(chip_id)
            per_chip = p.num_experts // ep
            first = shard.ep * per_chip
            for stage in _shard_shapes(device, model, layer, tp):
                if not stage.routed or not (first <= stage.expert < first + per_chip):
                    continue
                owner = TileOwner(model_id, layer, stage.op, stage.expert, shard_index)
                tiles_by_chip[chip_id].extend(_place(owner, stage, chip_id, chip_macro_base))

    # macro slots: derive when the config declares none, and disclose it.
    peak_usage = max(macro_usage.values()) if macro_usage else 0
    if macro_slots <= 0:
        macro_slots = max(1, peak_usage)
        relaxations.append(
            Relaxation(
                constraint="mapping.macros_per_chip",
                value=f"derived = {macro_slots}",
                reason=(
                    "cim.chip.arrays_per_chip is 0 (capacity checking disabled), so the "
                    "chip's slot count is derived from the placement itself and every "
                    "chip is exactly full. No capacity was checked."
                ),
            )
        )
    if not slots_known:
        # Shift each chip's locally-numbered tiles into its enumerated range.
        shifted: List[Tile] = []
        for chip_index, (chip_id, _label, _shard, _layers, _role) in enumerate(chip_specs):
            delta = chip_index * macro_slots - macro_base_of_chip[chip_id]
            macro_base_of_chip[chip_id] = chip_index * macro_slots
            moved = [
                replace(tile, site=replace(tile.site, macro_id=tile.site.macro_id + delta))
                for tile in tiles_by_chip[chip_id]
            ]
            tiles_by_chip[chip_id] = moved
            shifted.extend(moved)
        all_tiles = shifted
    for chip_id, used in macro_usage.items():
        if used > macro_slots:
            raise MappingError(
                "assembly",
                f"chip {chip_id} needs {used} macro slots but the chip declares "
                f"{macro_slots} (cim.chip.arrays_per_chip / mapping.macros_per_chip). "
                "Reduce layers_per_chip, raise the slot count, or shard further.",
            )

    # --- enumerate macros (every slot, reserved ones included) ------------
    next_device_id = 0
    card_id = _analog_card_id(device)
    for chip_index, (chip_id, chip_label, shard, chip_layers, chip_role) in enumerate(chip_specs):
        base = macro_base_of_chip[chip_id]
        by_macro = OrderedDict()
        for tile in tiles_by_chip[chip_id]:
            by_macro.setdefault(tile.site.macro_id, []).append(tile)
        macro_ids = []
        for slot in range(macro_slots):
            macro_id = base + slot
            resident = tuple(by_macro.get(macro_id, ()))
            macros.append(
                MacroSlot(
                    macro_id=macro_id,
                    chip_id=chip_id,
                    slot=slot,
                    pool="analog",
                    card_id=card_id,
                    shard=shard,
                    tiles=resident,
                    reserved=(
                        None
                        if resident
                        else "unallocated macro slot on an enumerated chip "
                             "(ADJ-5: unowned capacity is legal and reported)"
                    ),
                    analog_device=next_device_id,
                    pool_device=next_device_id + 1,
                )
            )
            macro_ids.append(macro_id)
            next_device_id += 2
        chips.append(
            ChipRecord(
                chip_id=chip_id,
                label=chip_label,
                pool="analog",
                shard=shard,
                macro_slots=macro_slots,
                macro_ids=tuple(macro_ids),
                layers=tuple(chip_layers),
                role=chip_role,
            )
        )

    # --- shared digital chiplets (D13, ADJ-5) -----------------------------
    suggestion = total_analog
    shared = int(spec.shared_chiplets) if spec.shared_chiplets is not None else suggestion
    if shared <= 0 and _needs_shared_digital(model):
        raise MappingError(
            "execution",
            "mapping.shared_chiplets = 0 but the workload has act x act compute "
            "(attention, scan or delta-rule state) which D13 puts on the shared digital "
            "chiplet. There is no device to place those ops on.",
        )
    if shared != suggestion:
        notes.append(
            f"shared digital chiplets: {shared} declared against a derived suggestion of "
            f"{suggestion} (one per analog chip, the concurrency the closed-form spatial "
            "pipeline assumes). ADJ-5 makes the count a config input; the suggestion is "
            "reported and binds nothing."
        )
    digital_card_id = _digital_card_id(device)
    next_macro_id = len(chip_specs) * macro_slots
    engines_per_chiplet = max(
        1, int(device.fabric.num_arrays) * max(1, int(getattr(device.fabric, "replicas", 1) or 1))
    )
    for index in range(shared):
        chip_id = next_chip_id
        next_chip_id += 1
        shard = ShardCoord(tp=index % tp if tp > 1 else 0, ep=0, pp=0)
        macro_ids = []
        for slot in range(engines_per_chiplet):
            macro_id = next_macro_id
            next_macro_id += 1
            macros.append(
                MacroSlot(
                    macro_id=macro_id,
                    chip_id=chip_id,
                    slot=slot,
                    pool="digital",
                    card_id=digital_card_id,
                    shard=shard,
                    tiles=(),
                    reserved=(
                        "shared digital chiplet engine: act x act GEMM, softmax, scan and "
                        "state update (D13). It stores no weights, so its stored-column "
                        "occupancy is 0."
                    ),
                    analog_device=-1,
                    pool_device=-1,
                )
            )
            macro_ids.append(macro_id)
        chips.append(
            ChipRecord(
                chip_id=chip_id,
                label=f"shared digital chiplet {index}",
                pool="digital",
                shard=shard,
                macro_slots=engines_per_chiplet,
                macro_ids=tuple(macro_ids),
                layers=(),
                role="shared_digital",
            )
        )

    # --- devices (A2): two per analog macro, one per shared chiplet -------
    devices: List[FwsDevice] = []
    chip_position: Dict[Tuple[int, int], int] = {}
    for chip in chips:
        key = (chip.shard.tp, 0 if chip.pool == "analog" else 1)
        position = chip_position.get(key, 0)
        chip_position[key] = position + 1
        if chip.pool == "analog":
            for macro_id in chip.macro_ids:
                macro = next(m for m in macros if m.macro_id == macro_id)
                devices.append(
                    FwsDevice(
                        device_id=macro.analog_device,
                        device_class="analog_macro",
                        chip_id=chip.chip_id,
                        macro_id=macro_id,
                        shard=chip.shard,
                        role_key=(position, macro.slot, "analog_macro"),
                    )
                )
                devices.append(
                    FwsDevice(
                        device_id=macro.pool_device,
                        device_class="macro_pool",
                        chip_id=chip.chip_id,
                        macro_id=macro_id,
                        shard=chip.shard,
                        role_key=(position, macro.slot, "macro_pool"),
                    )
                )
        else:
            devices.append(
                FwsDevice(
                    device_id=next_device_id,
                    device_class="shared_digital",
                    chip_id=chip.chip_id,
                    macro_id=-1,
                    shard=chip.shard,
                    role_key=(position, 0, "shared_digital"),
                )
            )
            next_device_id += 1

    # --- final capacity check over the placed sites (P2's named error) ----
    device.validate_macro_capacity(all_tiles)
    device.validate_allocation()

    decode_window = int(spec.decode_window or DEFAULT_DECODE_WINDOW)
    if phase == "decode" and decode_len > decode_window:
        relaxations.append(
            Relaxation(
                constraint="decode_window",
                value=f"{decode_window} of {decode_len} steps lowered",
                reason=(
                    "ADJ-6: the DAG grows linearly in the decode length, so a bounded "
                    "window is lowered and the remaining steps are an extrapolation the "
                    "consumer must declare. P3 lowers the window and states the bound; "
                    "it extrapolates nothing."
                ),
            )
        )

    blocks = sorted({tile.owner.op for tile in all_tiles})
    endpoint_blocks = sorted({stage.op for _, stage in endpoints})
    return FwsMapping(
        device=device,
        hw_config=hw_config,
        model_config=model,
        system_id=system_id,
        role=role,
        phase=phase,
        label=label,
        model_id=model_id,
        degrees=degrees,
        chips=chips,
        macros=macros,
        devices=devices,
        tiles=all_tiles,
        blocks=blocks,
        endpoint_blocks=endpoint_blocks,
        layer_chip=layer_chip,
        shared_chiplet_suggestion=suggestion,
        decode_window=decode_window,
        notes=notes,
        relaxations=relaxations,
    )


def _linearize(degrees: Mapping[str, int], shard: ShardCoord) -> int:
    span = 1
    out = 0
    for axis in MAPPING_AXES:
        out += shard.of(axis) * span
        span *= max(1, int(degrees[axis]))
    return out


def _owner_key(owner: TileOwner) -> Tuple[str, int, str, int, int]:
    return (owner.model, owner.layer, owner.op, owner.expert, owner.shard)


def _allocation_by_owner(device: CimDeviceModel) -> Dict[Tuple, List[object]]:
    allocation = device.allocation
    if allocation is None:
        return {}
    by_owner: Dict[Tuple, List[object]] = {}
    for entry in allocation.assignments:
        key = (entry.model, entry.layer, entry.op, entry.expert, entry.shard)
        by_owner.setdefault(key, []).append(entry)
    return by_owner


def _layer_counts_from_spec(spec_value, device: CimDeviceModel) -> Tuple[int, ...]:
    num_layers = int(device.params.num_layers)
    if isinstance(spec_value, str):
        return device.derive_auto_layers_per_chip()
    if isinstance(spec_value, tuple):
        if sum(spec_value) != num_layers:
            raise MappingError(
                "assembly",
                f"mapping.layers_per_chip {list(spec_value)} sums to {sum(spec_value)} but "
                f"the model has num_layers = {num_layers}.",
            )
        return spec_value
    per_chip = int(spec_value)
    num_chips = math.ceil(num_layers / per_chip)
    counts = [per_chip] * (num_chips - 1)
    counts.append(num_layers - per_chip * (num_chips - 1))
    return tuple(counts)


def _pp_membership(
    spec, pp: int, tp: int, backbone: int, expert: int
) -> Tuple[int, ...]:
    """Per-analog-chip pp index: declared membership, or derived by layer order."""
    per_shard = backbone + expert
    total = tp * per_shard
    declared = spec.membership.get("pp") if spec is not None else None
    if declared is not None:
        if len(declared) != total:
            raise MappingError(
                "annotation",
                f"mapping.membership.pp lists {len(declared)} entries but the placement "
                f"enumerates {total} analog chips. Membership is per chip, in chip-id "
                "order.",
            )
        bad = [index for index in declared if index >= pp]
        if bad:
            raise MappingError(
                "annotation",
                f"mapping.membership.pp names stage(s) {sorted(set(bad))} but "
                f"mapping.parallelism.pp = {pp} declares stages 0..{pp - 1}.",
            )
        missing = sorted(set(range(pp)) - set(declared))
        if missing:
            raise MappingError(
                "annotation",
                f"mapping.parallelism.pp = {pp} but no chip is assigned to stage(s) "
                f"{missing}. An empty pipeline stage is a degree nothing uses.",
            )
        return tuple(int(index) for index in declared)
    if pp > backbone:
        raise MappingError(
            "annotation",
            f"mapping.parallelism.pp = {pp} exceeds the {backbone} backbone chips per tp "
            "shard, so the derived membership would leave a stage empty. Declare "
            "mapping.membership.pp, or reduce pp.",
        )
    out: List[int] = []
    for _ in range(tp):
        for chip_index in range(backbone):
            out.append(chip_index * pp // backbone)
        for _ in range(expert):
            out.append(0)
    return tuple(out)


def _needs_shared_digital(model) -> bool:
    kinds = set(getattr(model, "block_kinds", ("attention",)) or ("attention",))
    return bool(kinds & {"attention", "ssm", "linear_attn"})


def _analog_card_id(device: CimDeviceModel) -> str:
    cards = getattr(device.cim, "cards", None)
    name = getattr(cards, "default_analog", None) if cards is not None else None
    return str(name or "analog_macro")


def _digital_card_id(device: CimDeviceModel) -> str:
    cards = getattr(device.cim, "cards", None)
    name = getattr(cards, "default_digital", None) if cards is not None else None
    return str(name or "shared_digital")


# ---------------------------------------------------------------------------
# PD disaggregation (P3.4, D16)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PDHandoff:
    """The one coupling between a prefill system and a decode system (D16).

    Bytes only: the KV of the prefilled context plus the recurrent/hidden state
    that seeds decode. No precision conversion is modeled — a seventh-order
    effect, explicitly out of scope.
    """

    kv_bytes: float
    state_bytes: float
    bandwidth_bytes_per_s: float
    latency_s: float
    time_s: float
    basis: str

    @property
    def total_bytes(self) -> float:
        return float(self.kv_bytes) + float(self.state_bytes)


@dataclass(frozen=True)
class PDPair:
    """Two inventories, two mappings, one handoff (D16, ADJ-5)."""

    prefill: FwsMapping
    decode: FwsMapping
    handoff: PDHandoff

    def summary(self) -> Dict[str, object]:
        return OrderedDict(
            (
                ("prefill", self.prefill.summary()),
                ("decode", self.decode.summary()),
                ("handoff_bytes", self.handoff.total_bytes),
                ("handoff_kv_bytes", self.handoff.kv_bytes),
                ("handoff_state_bytes", self.handoff.state_bytes),
                ("handoff_time_s", self.handoff.time_s),
            )
        )

    def report(self) -> str:
        lines = [self.prefill.report(), self.decode.report()]
        lines.append(
            f"[FWS-CIM]   PD handoff: {self.handoff.total_bytes:.6g} B "
            f"(KV {self.handoff.kv_bytes:.6g} + state {self.handoff.state_bytes:.6g}) "
            f"-> {self.handoff.time_s * 1e6:.6f} us on the declared p2p link. "
            f"{self.handoff.basis}"
        )
        return "\n".join(lines)


def pd_handoff_bytes(mapping: FwsMapping) -> Tuple[float, float, str]:
    """(KV bytes, state bytes, basis) one request hands from prefill to decode."""
    dev = mapping.device
    model = mapping.model
    precision = mapping.hw.sw_config.precision
    kv_precision = float(precision.kv_cache)
    act_bytes = float(precision.activations)
    p = dev.params
    seq_len = int(p.seq_len)
    decode_len = int(getattr(model, "decode_len", 0) or 0)
    prefill_len = max(1, seq_len - decode_len)
    batch = int(p.batch_size)
    kv_bytes = 0.0
    if not p.is_vit_shaped:
        # tp = 1 on purpose: the handoff carries the WHOLE KV of the request.
        # How each side shards it is that side's mapping, not the wire's.
        kv_bytes = batch * dev.kv_bytes_per_stream(prefill_len, kv_precision, 1)
    state_bytes = float(batch) * float(p.hidden_dim) * act_bytes
    ssm = getattr(model, "ssm", None)
    ssm_layers = 0
    for kinds in getattr(model, "layer_mixers", ()) or ():
        if "ssm" in kinds:
            ssm_layers += 1
    if ssm is not None and ssm_layers:
        d_inner = int(ssm.resolve_d_inner(int(p.hidden_dim)))
        state_bytes += float(batch) * d_inner * int(ssm.d_state) * ssm_layers * act_bytes
    basis = (
        f"KV = batch {batch} x kv_bytes_per_stream(context = {prefill_len}, tp = 1); "
        f"state = batch x hidden {p.hidden_dim} x act"
        + (f" + {ssm_layers} SSM recurrent states" if ssm_layers else "")
        + ". No precision conversion is modeled (D16)."
    )
    return kv_bytes, state_bytes, basis


def build_pd_pair(
    hw_config,
    model_config,
    *,
    prefill_spec=None,
    decode_spec=None,
    label: str = "PD pair",
) -> PDPair:
    """Two systems and the priced handoff (D16). Equal specs = one machine.

    ``None`` specs take the hardware config's ``mapping.pd`` block. Setting the
    two halves equal reproduces the non-disaggregated machine exactly: both
    inventories are the unified mapping, and the only extra quantity is the
    handoff byte count.
    """
    mapping_config = getattr(hw_config, "mapping_config", None)
    if prefill_spec is None and mapping_config is not None:
        prefill_spec = mapping_config.prefill
    if decode_spec is None and mapping_config is not None:
        decode_spec = mapping_config.decode
    if prefill_spec is None or decode_spec is None:
        raise MappingError(
            "assembly",
            "a PD pair needs BOTH a prefill spec and a decode spec (mapping.pd.prefill "
            "and mapping.pd.decode). D16 is two inventories and one handoff; one half "
            "alone is not a machine.",
        )
    prefill = build_mapping(
        hw_config,
        model_config,
        spec=prefill_spec,
        system_id="sys.prefill",
        role="prefill",
        phase="prefill",
        label=f"{label} — prefill system",
    )
    decode = build_mapping(
        hw_config,
        model_config,
        spec=decode_spec,
        system_id="sys.decode",
        role="decode",
        phase="decode",
        label=f"{label} — decode system",
    )
    kv_bytes, state_bytes, basis = pd_handoff_bytes(prefill)
    bandwidth, latency = hw_config.network_layout.link_for_parallelism("pp")
    total = kv_bytes + state_bytes
    handoff = PDHandoff(
        kv_bytes=kv_bytes,
        state_bytes=state_bytes,
        bandwidth_bytes_per_s=bandwidth,
        latency_s=latency,
        time_s=CimDeviceModel.p2p_time_s(total, bandwidth, latency),
        basis=basis,
    )
    return PDPair(prefill=prefill, decode=decode, handoff=handoff)
