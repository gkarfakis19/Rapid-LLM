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

"""The SLOT MAPPER: placement and schedule are ONE decision (D39).

D39 (docs/qif/DECISIONS.md) is the law this module implements, and nothing
else. Its terms, restated so the code can be read against them:

* A macro is ``rows x Y : Z`` -- ``Y`` total columns behind ``Z`` ADCs, so
  ``mux = Y / Z`` is DERIVED. One ADC firing is one TICK of 20 ns.
* ``beat = mux x 20 ns``. The platform point ``(Z, Y)`` fixes it. The mapper
  cannot move it. Throughput = ``1 / beat`` tokens/s. Lowering mux buys
  CONCURRENCY, not speed.
* SLOT = PHASE. Each ADC rotates through its mux slots once per beat, so slot
  index ``j`` fires at every tick ``== j (mod mux)``. A SLOT-LAYER is ``Z``
  columns of ONE bundle sitting in one slot index of one macro; all ``Z`` ADCs
  of that macro fire that slot together on ONE input vector.
* A token hops bundle to bundle along the rotations. Consecutive bundles need
  ``hop_gap_ticks = 2`` (one tick to finish converting, at most one tick of
  wire and reduce), plus ``seam_ticks = 25`` (500 ns) when the next bundle
  sits on a different chip.
* ``L`` is the per-token chain latency in ticks: the last firing tick of the
  last bundle, plus one, counted from tick 0 of the first. ``D = ceil(L /
  beat_ticks)`` in-flight sequences by Little's law.
* A user's next token re-enters on the first bundle's own phase, and that
  phase comes round once per beat. So one user's period is ``D x beat`` and
  the per-user rate is ``throughput / D``. Little's law then CLOSES on the
  ledger strip: ``D x per-user = throughput``, exactly. ``1 / (L x 20 ns)``
  is the CEILING of that rate, reached only when the beat divides ``L``.
  D39 writes the ceiling as the rate. Written that way, a chain shorter than
  one beat prints one user out-running the whole machine (``D = 1`` and a
  per-user rate above the throughput), so the document carries the two
  figures under their own names: ``per_user_tokens_per_s`` is what a user
  gets, ``per_user_ceiling_tokens_per_s`` is the latency ceiling.
* THE MAPPER CONTROLS EXACTLY THREE LEDGERS: cell waste, ``L`` (and through it
  ``D``), and seams. Every one is reported against its bound.
* NO NUMBER IS TRUSTED FROM THEORY. :func:`to_document` emits the explicit
  tick schedule and :func:`check_document` re-derives it from the DOCUMENT
  ALONE -- it never reads builder state.

This module is standalone on purpose. It imports nothing from ``fws_mapping``
or ``fws_eval``; the only outside call is ``cim_analog_calc`` for the macro's
area, which is a sheet port and not a mapping decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "SlotMapError",
    "SlotPlatform",
    "BundleSpec",
    "SlotPlacement",
    "MacroSlots",
    "BundleResult",
    "SlotMapping",
    "map_chain",
    "ledgers",
    "macro_area_mm2",
    "to_document",
    "check_document",
]

#: The extension schema this module writes inside an ``fws_atlas/1`` document.
SLOT_SCHEMA = "fws_slot/1"
#: The atlas contract the document as a whole conforms to.
ATLAS_SCHEMA = "fws_atlas/1"


class SlotMapError(ValueError):
    """A platform, a bundle chain or a document this mapper refuses."""


# --------------------------------------------------------------------------
# the platform and the work
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotPlatform:
    """One macro geometry and the chip it is packed into (D39).

    ``tick_ns`` is NANOSECONDS -- the same number the document carries as
    ``slots.platform.tick_ns``. One ADC firing is one tick.
    """

    rows: int
    z: int
    y: int
    chip_macros: int
    tick_ns: float = 20.0
    hop_gap_ticks: int = 2
    seam_ticks: int = 25

    def __post_init__(self) -> None:
        for name in ("rows", "z", "y", "chip_macros"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise SlotMapError(
                    f"SlotPlatform.{name} must be a positive integer, got {value!r}."
                )
        if self.y % self.z:
            raise SlotMapError(
                f"SlotPlatform: mux = Y / Z = {self.y} / {self.z} is not a whole "
                "number. The mux is DERIVED and a fractional one has no meaning: "
                "an ADC carries a whole number of muxed columns."
            )
        if self.tick_ns < 1e-3:
            raise SlotMapError(
                f"SlotPlatform.tick_ns = {self.tick_ns!r} is NANOSECONDS, not "
                "seconds. One tick is 20 ns, so pass 20.0."
            )
        if self.hop_gap_ticks < 0 or self.seam_ticks < 0:
            raise SlotMapError("hop_gap_ticks and seam_ticks are tick counts >= 0.")

    @property
    def mux(self) -> int:
        """``Y / Z`` -- how many slots one ADC rotates through per beat."""
        return self.y // self.z

    @property
    def beat_ticks(self) -> int:
        """``mux`` ticks. The beat is OCCUPANCY ONLY; latency never sets it."""
        return self.mux

    @property
    def beat_ns(self) -> float:
        return self.mux * self.tick_ns

    @property
    def tick_s(self) -> float:
        return self.tick_ns * 1e-9

    @property
    def beat_s(self) -> float:
        return self.beat_ns * 1e-9

    @property
    def slots_per_macro(self) -> int:
        return self.mux

    @property
    def slots_per_chip(self) -> int:
        return self.chip_macros * self.mux

    @property
    def label(self) -> str:
        """``rows x Y : Z`` -- how a macro is named everywhere."""
        return f"{self.rows}x{self.y}:{self.z}"


@dataclass(frozen=True)
class BundleSpec:
    """One link of the chain: the columns a token must fire through together.

    ``columns`` is the bundle's stored column-slots, ALL K-BLOCKS TOGETHER,
    which is the meaning ``slots.bundles[].columns`` carries in the document
    (docs/qif/atlas/SCHEMA.md, fws_slot/1). ``k_blocks`` above one means the
    matrix is taller than the macro: it is cut into that many row-blocks of
    equal output width, each owning its own input slice and its own column-
    slots, and their partials REDUCE (``reduce``). The K-blocks may fire on
    different macros at the same tick. The bundle's column range is
    ``0 .. columns``, and K-block k owns the k-th contiguous chunk of it, so
    ``columns`` must divide by ``k_blocks``.
    """

    id: str
    order: int
    layer: int
    block: str
    columns: int
    k_blocks: int = 1
    reduce: bool = False

    def columns_per_kblock(self) -> int:
        """The output width one K-block stores. Every K-block has the same."""
        return self.columns // self.k_blocks

    def chunks_per_kblock(self, z: int) -> int:
        """Slot-layers one K-block needs: its width over Z, rounded up."""
        return math.ceil(self.columns_per_kblock() / z)

    def slot_layers(self, z: int) -> int:
        """``ceil(columns per K-block / Z) x k_blocks`` -- what this bundle owns."""
        return self.chunks_per_kblock(z) * self.k_blocks

    def ragged_columns(self, z: int) -> int:
        """Column-slots the ragged last layer of every K-block leaves empty."""
        per = self.columns_per_kblock()
        return (self.chunks_per_kblock(z) * z - per) * self.k_blocks


# --------------------------------------------------------------------------
# what the mapper produces
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotPlacement:
    """One occupied slot: WHERE a slot-layer sits and WHEN it fires."""

    macro: int
    slot: int
    tick: int
    bundle: str
    layer: int
    block: str
    kblock: int
    col_start: int
    col_count: int


@dataclass
class MacroSlots:
    """One macro's slot table."""

    index: int
    chip: int
    index_on_chip: int
    placed: Dict[int, SlotPlacement] = field(default_factory=dict)
    free: List[int] = field(default_factory=list)
    stranded: List[int] = field(default_factory=list)


@dataclass
class BundleResult:
    """One bundle after placement: its chip, its macros and its ticks."""

    spec: BundleSpec
    chip: int
    seam_before: bool
    t_min: int
    placements: List[SlotPlacement]

    @property
    def start_tick(self) -> int:
        return min(p.tick for p in self.placements)

    @property
    def end_tick(self) -> int:
        return max(p.tick for p in self.placements)

    @property
    def serial_ticks(self) -> int:
        return self.end_tick - self.start_tick + 1

    @property
    def stall_ticks(self) -> int:
        return self.start_tick - self.t_min

    @property
    def macros(self) -> List[int]:
        return sorted({p.macro for p in self.placements})


@dataclass
class SlotMapping:
    """A placed and scheduled chain."""

    platform: SlotPlatform
    bundles: List[BundleResult]
    macros: List[MacroSlots]
    chips: List[List[int]]


# --------------------------------------------------------------------------
# the greedy law mapper
# --------------------------------------------------------------------------


def map_chain(platform: SlotPlatform, bundles: Sequence[BundleSpec]) -> SlotMapping:
    """Place and schedule the chain in one decision (D39).

    THE ALGORITHM, exactly.

    1. Bundles are taken in chain order. The first bundle opens chip 0 and its
       ``t_min`` is 0. Afterwards ``t_min = prev.end_tick + hop_gap_ticks``,
       plus ``seam_ticks`` when this bundle sits on a different chip than the
       previous one.
    2. A bundle NEVER splits across chips. If the current chip has fewer free
       slots than the bundle needs, every free slot still on that chip is
       marked STRANDED, a fresh chip opens, and the WHOLE bundle moves there
       with ``seam_before = True``.
    3. Each slot-layer of the bundle, in order (K-block outer, column chunk
       inner), takes the (macro on the current chip, free slot ``j``) pair that
       minimises ``fire(j) = t_min + ((j - t_min) mod mux)`` -- the next tick at
       which that slot's phase comes round. Ties break to the macro with the
       MOST free slots (that is the spread rule), then to the lowest macro
       index. Two slot-layers of one bundle share a macro only when this rule
       yields it.
    4. ``start_tick`` / ``end_tick`` are the min and max firing ticks of the
       bundle, ``serial_ticks = end - start + 1`` and ``stall_ticks =
       start_tick - t_min``.

    Because every slot on a macro is taken at most once and ``fire(j) == j (mod
    mux)``, no macro ever fires twice in one tick. That is a property of the
    construction, and :func:`check_document` re-proves it from the document.
    """
    specs = sorted(bundles, key=lambda b: b.order)
    if not specs:
        raise SlotMapError("map_chain needs at least one bundle.")
    seen = set()
    for spec in specs:
        if spec.id in seen:
            raise SlotMapError(f"bundle id {spec.id!r} appears twice.")
        seen.add(spec.id)
        if spec.columns <= 0 or spec.k_blocks <= 0:
            raise SlotMapError(
                f"bundle {spec.id!r}: columns and k_blocks are positive counts."
            )
        if spec.columns % spec.k_blocks:
            raise SlotMapError(
                f"bundle {spec.id!r}: columns is the stored column-slots of the "
                f"whole bundle, and its {spec.k_blocks} K-blocks share it "
                f"equally, so {spec.columns} must divide by {spec.k_blocks}."
            )
        need = spec.slot_layers(platform.z)
        if need > platform.slots_per_chip:
            raise SlotMapError(
                f"bundle {spec.id!r} needs {need} slot-layers but one chip holds "
                f"{platform.slots_per_chip} ({platform.chip_macros} macros x mux "
                f"{platform.mux}). A bundle is never split across chips, so this "
                "chain cannot be mapped on this platform."
            )

    macros: List[MacroSlots] = []
    chips: List[List[int]] = []
    results: List[BundleResult] = []

    def open_chip() -> int:
        idx = len(chips)
        ids: List[int] = []
        for k in range(platform.chip_macros):
            m = MacroSlots(
                index=len(macros),
                chip=idx,
                index_on_chip=k,
                free=list(range(platform.mux)),
            )
            macros.append(m)
            ids.append(m.index)
        chips.append(ids)
        return idx

    current = -1
    prev: Optional[BundleResult] = None
    for spec in specs:
        need = spec.slot_layers(platform.z)
        if current < 0:
            current = open_chip()
            seam_before = False
        else:
            free_now = sum(len(macros[i].free) for i in chips[current])
            if free_now < need:
                for i in chips[current]:
                    macros[i].stranded.extend(macros[i].free)
                    macros[i].stranded.sort()
                    macros[i].free = []
                current = open_chip()
                seam_before = True
            else:
                seam_before = False

        if prev is None:
            t_min = 0
        else:
            t_min = prev.end_tick + platform.hop_gap_ticks
            if seam_before:
                t_min += platform.seam_ticks

        placements: List[SlotPlacement] = []
        per = spec.columns_per_kblock()
        chunks = spec.chunks_per_kblock(platform.z)
        for kblock in range(spec.k_blocks):
            base = kblock * per
            for chunk in range(chunks):
                start = base + chunk * platform.z
                count = min(platform.z, base + per - start)
                best: Optional[Tuple[Tuple[int, int, int], MacroSlots, int]] = None
                for i in chips[current]:
                    m = macros[i]
                    for slot in m.free:
                        fire = t_min + ((slot - t_min) % platform.mux)
                        key = (fire, -len(m.free), m.index)
                        if best is None or key < best[0]:
                            best = (key, m, slot)
                if best is None:  # pragma: no cover - step 2 guarantees room
                    raise SlotMapError(
                        f"bundle {spec.id!r}: no free slot on chip {current}."
                    )
                _, macro, slot = best
                place = SlotPlacement(
                    macro=macro.index,
                    slot=slot,
                    tick=t_min + ((slot - t_min) % platform.mux),
                    bundle=spec.id,
                    layer=spec.layer,
                    block=spec.block,
                    kblock=kblock,
                    col_start=start,
                    col_count=count,
                )
                macro.free.remove(slot)
                macro.placed[slot] = place
                placements.append(place)

        result = BundleResult(
            spec=spec,
            chip=current,
            seam_before=seam_before,
            t_min=t_min,
            placements=placements,
        )
        results.append(result)
        prev = result

    return SlotMapping(platform=platform, bundles=results, macros=macros, chips=chips)


# --------------------------------------------------------------------------
# the three ledgers, each against its bound
# --------------------------------------------------------------------------


def macro_area_mm2(platform: SlotPlatform, *, profile: str = "george_cima_mux1") -> float:
    """The macro's silicon, DERIVED from the sheet port at ``rows x Y : Z``."""
    from cim_analog_calc import derive_analog_card

    card = derive_analog_card(
        platform.rows,
        platform.y,
        platform.z,
        profile=profile,
        context=f"toy slot map, macro {platform.label}",
    )
    return float(card.area_mm2_per_array)


def ledgers(mapping: SlotMapping, *, area_mm2: Optional[float] = None) -> Dict[str, Any]:
    """The three ledgers D39 gives the mapper, each next to its bound.

    Waste is CELLS: the ragged remainder of every bundle's last slot-layer
    against ``Z``, plus every slot STRANDED when a bundle moved whole to a
    fresh chip. ``L`` is the chain latency in ticks and ``D = ceil(L /
    beat_ticks)``. Seams are counted, never priced into the beat.
    """
    p = mapping.platform
    if area_mm2 is None:
        area_mm2 = macro_area_mm2(p)

    placements = [q for m in mapping.macros for q in m.placed.values()]
    slots_used = len(placements)
    slots_stranded = sum(len(m.stranded) for m in mapping.macros)
    slots_total = len(mapping.macros) * p.mux
    macros_used = sum(1 for m in mapping.macros if m.placed)

    real_columns = sum(q.col_count for q in placements)
    cells_used = real_columns * p.rows
    cells_ragged = (slots_used * p.z - real_columns) * p.rows
    cells_stranded = slots_stranded * p.z * p.rows
    cells_wasted = cells_ragged + cells_stranded
    cells_total = slots_total * p.z * p.rows

    first = mapping.bundles[0]
    last = mapping.bundles[-1]
    l_ticks = last.end_tick - first.start_tick + 1
    seams = sum(1 for b in mapping.bundles if b.seam_before)
    chips_min = math.ceil(slots_used / p.slots_per_chip)
    # The tight floor under D39: a chain of n bundles that each fire in a
    # single tick lands at 2n - 1, because L counts the last firing plus one
    # and the hop gap (2 ticks) only separates the n - 1 joins. Plus one seam
    # per chip that capacity forces. A valid mapping can never print an L
    # below this figure; the checker re-derives it from the document.
    l_min_ticks = (2 * len(mapping.bundles) - 1
                   + p.seam_ticks * max(0, chips_min - 1))
    d_seq = math.ceil(l_ticks / p.beat_ticks)

    return {
        "beat_ns": p.beat_ns,
        "beat_ticks": p.beat_ticks,
        "L_ticks": l_ticks,
        "L_ns": l_ticks * p.tick_ns,
        "D": d_seq,
        "D_min": math.ceil(l_min_ticks / p.beat_ticks),
        "L_min_ticks": l_min_ticks,
        "macros_used": macros_used,
        "chips_used": len(mapping.chips),
        "slots_total": slots_total,
        "slots_used": slots_used,
        "slots_stranded": slots_stranded,
        "cells_total": cells_total,
        "cells_used": cells_used,
        "cells_ragged": cells_ragged,
        "cells_stranded": cells_stranded,
        "cells_wasted": cells_wasted,
        # SCHEMA.md, fws_slot/1: the waste is a share of EVERY cell the map
        # commits, which is every macro it occupies, free slots included. A
        # free slot is not waste, but its silicon is bought and the fraction
        # is read against the whole array the mapping paid for.
        "waste_frac": (cells_wasted / cells_total) if cells_total else 0.0,
        "waste_min": 0,
        "seams": seams,
        "seams_min": max(0, chips_min - 1),
        "stall_ticks": sum(b.stall_ticks for b in mapping.bundles),
        "throughput_tokens_per_s": 1.0 / p.beat_s,
        # A user's next token re-enters on the first bundle's phase, and that
        # phase comes round once per beat. The period is D beats, so the rate
        # is the throughput shared over the D sequences in flight and Little's
        # law closes: D x per_user == throughput. 1 / L is the CEILING of it,
        # and only a beat that divides L reaches the ceiling.
        "per_user_tokens_per_s": 1.0 / (d_seq * p.beat_s),
        "per_user_ceiling_tokens_per_s": 1.0 / (l_ticks * p.tick_s),
        "macro_duty_mean": (slots_used / slots_total) if slots_total else 0.0,
        "macro_duty_max": max((len(m.placed) / p.mux) for m in mapping.macros),
        "analog_area_mm2": macros_used * float(area_mm2),
        "macro_area_mm2": float(area_mm2),
    }


# --------------------------------------------------------------------------
# the document
# --------------------------------------------------------------------------


def _chip_id(i: int) -> str:
    return f"chip.{i:02d}"


def _macro_id(i: int) -> str:
    return f"mac.{i:03d}"


def _tile_id(i: int) -> str:
    return f"tile.{i:04d}"


_GROUPS = {"tp": 0, "ep": 0, "pp": 0}


def to_document(
    mapping: SlotMapping,
    *,
    title: str,
    authored: str,
    author: str,
    command: str,
    subtitle: Optional[str] = None,
    model_id: str = "Toy slot-map",
    model_label: Optional[str] = None,
    layers: Optional[int] = None,
    intermediate_size: Optional[int] = None,
    note: str = "",
    grounded: Optional[Iterable[str]] = None,
    relaxations: Optional[Iterable[Dict[str, str]]] = None,
    run: Optional[Dict[str, Any]] = None,
    dse_point: Optional[Dict[str, Any]] = None,
    area_mm2: Optional[float] = None,
) -> Dict[str, Any]:
    """One ``fws_atlas/1`` document carrying the ``fws_slot/1`` extension.

    The atlas half is the PLACEMENT: one tile per occupied slot, its columns a
    real span inside the macro (slot ``j`` owns columns ``[j*Z, (j+1)*Z)``).
    The ``slots`` half is the SCHEDULE: the same table again with the firing
    tick of token 0 on every entry, the bundles in chain order, and the
    ledgers. Containment is stated both ways in both halves, which is what
    makes a broken export visible.

    Every field the toy cannot honestly fill is ABSENT. The toy places no
    digital chiplet and holds no per-sequence state, so it emits no digital
    card, no link and no state figure -- never a zero standing in for one.
    """
    p = mapping.platform
    led = ledgers(mapping, area_mm2=area_mm2)
    card_id = f"analog_{p.rows}x{p.y}x{p.z}"
    system_id = "sys.toy"
    blocks = sorted({b.spec.block for b in mapping.bundles})
    n_layers = layers if layers is not None else 1 + max(b.spec.layer for b in mapping.bundles)

    # tiles, in layout order: macro index, then slot index. Slot j of a macro
    # owns stored columns [j*Z, (j+1)*Z), so a tile's span is real and a ragged
    # last layer is a short span rather than a padded one.
    tiles: List[Dict[str, Any]] = []
    macro_tiles: Dict[int, List[str]] = {m.index: [] for m in mapping.macros}
    for m in mapping.macros:
        for slot in sorted(m.placed):
            q = m.placed[slot]
            tid = _tile_id(len(tiles))
            macro_tiles[m.index].append(tid)
            tiles.append({
                "id": tid,
                "macro": _macro_id(m.index),
                "owner": {"model": model_id, "layer": q.layer, "block": q.block, "expert": None},
                "shape": {"K": p.rows, "N": q.col_count},
                "slice": None,
                "groups": dict(_GROUPS),
                "primary_group": "tp",
                "columns": {"start": slot * p.z, "count": q.col_count},
            })

    atlas_macros = [{
        "id": _macro_id(m.index),
        "chip": _chip_id(m.chip),
        "card": card_id,
        "tiles": list(macro_tiles[m.index]),
        "occupancy": sum(q.col_count for q in m.placed.values()) / p.y,
        "duty_cycle": len(m.placed) / p.mux,
        "digital_pool": None,
        "reserved": None if m.placed else (
            "no slot-layer landed here: the chain was placed in order and moved "
            "on before this macro was reached"
        ),
    } for m in mapping.macros]

    atlas_chips = [{
        "id": _chip_id(ci),
        "system": system_id,
        "pool": "analog",
        "label": f"analog chiplet {ci} - {p.chip_macros} macros of {p.label}",
        "capacity": {"macro_slots": p.chip_macros},
        "groups": dict(_GROUPS),
        "primary_group": "tp",
        "occupancy": sum(1 for i in ids if mapping.macros[i].placed) / p.chip_macros,
        "macros": [_macro_id(i) for i in ids],
    } for ci, ids in enumerate(mapping.chips)]

    slot_macros = [{
        "id": _macro_id(m.index),
        "chip": _chip_id(m.chip),
        "index": m.index,
        "index_on_chip": m.index_on_chip,
        "slots": [{
            "slot": s,
            "bundle": m.placed[s].bundle,
            "layer": m.placed[s].layer,
            "block": m.placed[s].block,
            "kblock": m.placed[s].kblock,
            "columns": {"start": m.placed[s].col_start, "count": m.placed[s].col_count},
            "tick": m.placed[s].tick,
        } for s in sorted(m.placed)],
        "free_slots": sorted(m.free),
        "stranded_slots": sorted(m.stranded),
    } for m in mapping.macros]

    slot_bundles = [{
        "id": b.spec.id,
        "order": b.spec.order,
        "layer": b.spec.layer,
        "block": b.spec.block,
        "columns": b.spec.columns,
        "k_blocks": b.spec.k_blocks,
        "reduce": bool(b.spec.reduce),
        "slot_layers": b.spec.slot_layers(p.z),
        "ragged_columns": b.spec.ragged_columns(p.z),
        "chip": _chip_id(b.chip),
        "macros": [_macro_id(i) for i in b.macros],
        "start_tick": b.start_tick,
        "end_tick": b.end_tick,
        "serial_ticks": b.serial_ticks,
        "stall_ticks": b.stall_ticks,
        "seam_before": b.seam_before,
    } for b in mapping.bundles]

    area = led["macro_area_mm2"]
    metrics = _metrics(p, led)
    model: Dict[str, Any] = {
        "id": model_id,
        "label": model_label or model_id,
        "layers": n_layers,
        "hidden_dim": p.rows,
        "blocks": blocks,
        "endpoint_blocks": [],
    }
    if intermediate_size is not None:
        model["intermediate_size"] = intermediate_size

    doc: Dict[str, Any] = {
        "schema": ATLAS_SCHEMA,
        "title": title,
        "provenance": {
            "authored": authored,
            "author": author,
            "reference_run": {
                "command": command,
                "report": "dse_report.json of this run",
                "parallelism": "tp = 1, ep = 1, pp = 1",
            },
            "grounded": list(grounded)
            if grounded is not None
            else [
                "chips[], macros[] and tiles[] are the enumerated placement the slot "
                "mapper produced: one tile per OCCUPIED slot",
                "slots.macros[].slots carry the firing tick of token 0, and slot index "
                "equals tick modulo mux on every one of them",
                "slots.ledgers are re-derived from those tables by an independent "
                "checker before this document is written",
                "cards[].area_mm2 is the analog sheet port evaluated at this macro "
                "geometry",
            ],
            "invented_fields": [],
            "note": note,
        },
        "cards": {
            card_id: {
                "label": f"analog macro, {p.rows} rows x {p.y} columns : {p.z} ADCs",
                "kind": "analog_macro",
                "capacity": {
                    "rows": p.rows,
                    "cols_adc": p.z,
                    "mux": p.mux,
                    "stored_cols": p.y,
                },
                "area_mm2": area,
                "bits_per_cell": 8,
                "weight_bits": 8,
                "slicing": False,
                "basis": (
                    "area_mm2 is the analog sheet port evaluated at "
                    f"{p.rows} rows x {p.y} columns behind {p.z} ADCs, so mux = "
                    f"{p.mux} is derived. One cell stores one 8-bit weight, so the "
                    "card does not slice."
                ),
            }
        },
        "systems": [
            {
                "id": system_id,
                "role": "unified",
                "phase": "decode",
                "label": title,
                "parallelism": {"tp": 1, "ep": 1, "pp": 1},
                "models": [model],
            }
        ],
        "chips": atlas_chips,
        "macros": atlas_macros,
        "tiles": tiles,
        "links": [],
        "metrics": metrics,
        "relaxations": list(relaxations or []),
        "slots": {
            "schema": SLOT_SCHEMA,
            "platform": {
                "rows": p.rows,
                "z": p.z,
                "y": p.y,
                "mux": p.mux,
                "tick_ns": p.tick_ns,
                "beat_ticks": p.beat_ticks,
                "beat_ns": p.beat_ns,
                "hop_gap_ticks": p.hop_gap_ticks,
                "seam_ticks": p.seam_ticks,
                "seam_ns": p.seam_ticks * p.tick_ns,
                "chip_macros": p.chip_macros,
                "macro_area_mm2": area,
                "label": p.label,
            },
            "chips": [
                {
                    "id": _chip_id(ci),
                    "index": ci,
                    "macros": [_macro_id(i) for i in ids],
                    "capacity_macros": p.chip_macros,
                }
                for ci, ids in enumerate(mapping.chips)
            ],
            "macros": slot_macros,
            "bundles": slot_bundles,
            "ledgers": {k: v for k, v in led.items() if k != "macro_area_mm2"},
            "check": {"ok": None, "violations": ["not checked yet"]},
        },
    }
    if subtitle:
        doc["subtitle"] = subtitle
    if run:
        doc["run"] = dict(run)
        if run.get("run_id"):
            doc["run_id"] = run["run_id"]
    if dse_point:
        doc["dse_point"] = dict(dse_point)
    return doc


def _metrics(p: SlotPlatform, led: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The headline strip. Every figure names where it came from."""
    beat = f"beat = mux {p.mux} x tick {p.tick_ns:g} ns; the macro geometry fixes it"
    sched = "read off the emitted tick schedule, which an independent checker verified"
    rows = [
        ("toy.beat", "beat, one token exits per beat", led["beat_ns"], "ns", beat),
        ("toy.tokens_per_s", "steady throughput (HEADLINE)",
         led["throughput_tokens_per_s"], "tokens/s", "1 / beat. " + beat),
        ("toy.per_user_tokens_per_s", "one user's token rate",
         led["per_user_tokens_per_s"], "tokens/s",
         f"1 / (D {led['D']} x beat {led['beat_ns']:g} ns): a user's next token "
         f"re-enters on the first bundle's phase, and that phase comes round "
         f"once per beat, so D users share the throughput and D x this rate is "
         f"the throughput, " + sched),
        ("toy.per_user_ceiling_tokens_per_s", "ceiling on one user's token rate",
         led["per_user_ceiling_tokens_per_s"], "tokens/s",
         f"1 / (L {led['L_ticks']} ticks x {p.tick_ns:g} ns): the rate a user "
         f"would get if the next token could start the instant the last one "
         f"left, and only a beat that divides L reaches it, " + sched),
        ("toy.chain_latency_ticks", "chain latency L", led["L_ticks"], "ticks",
         "last firing tick of the last bundle plus one, from tick 0 of the first; " + sched),
        ("toy.chain_latency_ticks", "chain latency lower bound", led["L_min_ticks"], "ticks",
         "two ticks per bundle plus one seam per unavoidable chip crossing, "
         "relaxing the chip capacity to a bin-packing bound"),
        ("toy.resident_sequences", "sequences in flight D", led["D"], "sequences",
         "ceil(L / beat ticks): latency does not set the beat, it sets how many "
         "sequences are in flight"),
        ("toy.resident_sequences", "sequences in flight at the latency bound",
         led["D_min"], "sequences", "ceil(L lower bound / beat ticks)"),
        ("toy.cells_wasted", "cells committed that hold no weight", led["cells_wasted"], "cells",
         "ragged remainder of each bundle against Z, plus every slot stranded when "
         "a bundle moved whole to a fresh chip"),
        ("toy.waste_frac", "share of committed cells that are waste", led["waste_frac"],
         "fraction", "wasted cells over every cell the mapping commits; the bound is zero"),
        ("toy.seams", "chip crossings on the chain", led["seams"], "seams",
         f"bundles whose chip differs from the previous bundle's; each costs "
         f"{p.seam_ticks} ticks of latency and nothing of the beat"),
        ("toy.seams", "chip crossings at the capacity bound", led["seams_min"], "seams",
         "one fewer than the fewest chips these slot-layers could fit in"),
        ("toy.analog_chips", "enumerated analog chiplets", led["chips_used"], "chips",
         "the mapper opened them one by one; a chip count here is never a division"),
        ("toy.macros_owned", "macros holding at least one slot-layer", led["macros_used"],
         "macros", "occupied slots grouped by macro"),
        ("toy.slots_used", "occupied slots", led["slots_used"], "slots",
         "one per slot-layer, one tile each"),
        ("toy.slots_used", "slots stranded behind a moved bundle", led["slots_stranded"],
         "slots", "free slots left on a chip when the next bundle did not fit whole"),
        ("toy.analog_silicon", "silicon of the macros the mapping occupies",
         led["analog_area_mm2"], "mm2",
         f"{led['macros_used']} macros x {led['macro_area_mm2']:.6g} mm2 from the "
         "analog sheet port"),
        ("toy.stall_ticks", "ticks a bundle waited for its phase", led["stall_ticks"],
         "ticks", "summed start_tick minus earliest legal tick over the chain"),
    ]
    return [
        {"key": k, "label": lbl, "value": float(val), "unit": u, "basis": b}
        for k, lbl, val, u, b in rows
    ]


# --------------------------------------------------------------------------
# the INDEPENDENT checker
# --------------------------------------------------------------------------


def check_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Re-derive the whole schedule from the DOCUMENT ALONE (D39).

    This function never sees a :class:`SlotMapping`. It reads the emitted
    tables and re-proves, in order: the beat law, one firing per macro per
    tick with slot equal to tick modulo mux, every bundle's slot-layer count
    and column conservation, the hop and seam gaps along the chain, three
    tokens of the rotation, every ledger, and containment in both directions
    between chips, macros, slots and tiles.

    Returns ``{"ok": bool, "violations": [str]}``. Callers write it into
    ``doc["slots"]["check"]`` as a separate step, so the document carries the
    verdict of a reader, not of its own builder.
    """
    v: List[str] = []

    def bad(msg: str) -> None:
        v.append(msg)

    slots = doc.get("slots")
    if not isinstance(slots, dict):
        return {"ok": False, "violations": ["document carries no slots extension"]}
    if slots.get("schema") != SLOT_SCHEMA:
        bad(f"slots.schema is {slots.get('schema')!r}, expected {SLOT_SCHEMA!r}")

    plat = slots.get("platform") or {}
    try:
        rows = int(plat["rows"])
        z = int(plat["z"])
        y = int(plat["y"])
        mux = int(plat["mux"])
        beat_ticks = int(plat["beat_ticks"])
        tick_ns = float(plat["tick_ns"])
        hop = int(plat["hop_gap_ticks"])
        seam_ticks = int(plat["seam_ticks"])
        chip_macros = int(plat["chip_macros"])
    except (KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "violations": [f"slots.platform is unreadable: {exc}"]}

    # (a) the beat law
    if z <= 0 or y % z:
        bad(f"(a) mux = Y/Z = {y}/{z} is not a whole number")
    elif mux != y // z:
        bad(f"(a) declared mux {mux} but Y/Z = {y // z}")
    if beat_ticks != mux:
        bad(f"(a) beat_ticks {beat_ticks} is not mux {mux}")
    if abs(float(plat.get("beat_ns", -1)) - mux * tick_ns) > 1e-9:
        bad(f"(a) beat_ns {plat.get('beat_ns')} is not mux {mux} x tick {tick_ns}")

    smacros = slots.get("macros") or []
    schips = slots.get("chips") or []
    sbundles = sorted(slots.get("bundles") or [], key=lambda b: b.get("order", 0))
    if not smacros or not schips or not sbundles:
        return {"ok": False, "violations": v + ["slots tables are empty"]}

    # (b) one slot index per macro, and slot == tick mod mux
    fires: Dict[str, Dict[int, str]] = {}
    used_slots = 0
    for m in smacros:
        mid = m.get("id")
        seen: Dict[int, Any] = {}
        by_tick: Dict[int, int] = {}
        fires[mid] = {}
        for entry in m.get("slots") or []:
            slot, tick = int(entry["slot"]), int(entry["tick"])
            used_slots += 1
            if slot in seen:
                bad(f"(b) macro {mid} books slot {slot} twice")
            seen[slot] = entry
            if not 0 <= slot < mux:
                bad(f"(b) macro {mid} slot {slot} is outside 0..{mux - 1}")
            if tick % mux != slot:
                bad(
                    f"(b) macro {mid} slot {slot} fires at tick {tick}, and "
                    f"{tick} mod {mux} = {tick % mux}: slot is phase"
                )
            phase = tick % mux
            if phase in by_tick:
                bad(f"(b) macro {mid} fires twice on phase {phase}")
            by_tick[phase] = slot
            fires[mid][slot] = entry
        overlap = set(m.get("free_slots") or []) & set(m.get("stranded_slots") or [])
        if overlap:
            bad(f"(b) macro {mid} lists slot(s) {sorted(overlap)} as free AND stranded")
        allslots = sorted(list(seen) + list(m.get("free_slots") or []) + list(m.get("stranded_slots") or []))
        if allslots != list(range(mux)):
            bad(
                f"(b) macro {mid} accounts for {len(allslots)} slots, and the macro "
                f"rotates through {mux}"
            )

    # (c) slot-layer counts and column conservation, per bundle
    by_bundle: Dict[str, List[Dict[str, Any]]] = {}
    for m in smacros:
        for entry in m.get("slots") or []:
            by_bundle.setdefault(entry["bundle"], []).append(dict(entry, _macro=m["id"]))
    for b in sbundles:
        bid = b["id"]
        cols, kb = int(b["columns"]), int(b["k_blocks"])
        # ``columns`` is the whole bundle, all K-blocks together (SCHEMA.md).
        # The K-blocks share it equally and own contiguous chunks of it, so the
        # count per K-block is the total over their number.
        if kb <= 0 or cols <= 0 or cols % kb:
            bad(f"(c) bundle {bid} declares {cols} columns over {kb} K-blocks, which do not divide")
            continue
        per = cols // kb
        want = math.ceil(per / z) * kb
        if int(b["slot_layers"]) != want:
            bad(f"(c) bundle {bid} declares {b['slot_layers']} slot-layers, needs {want}")
        got = by_bundle.get(bid, [])
        if len(got) != want:
            bad(f"(c) bundle {bid} has {len(got)} occupied slots, needs {want}")
        if int(b.get("ragged_columns", -1)) != (math.ceil(per / z) * z - per) * kb:
            bad(f"(c) bundle {bid} declares the wrong ragged column count")
        for k in range(kb):
            base, cover = k * per, [0] * per
            for entry in got:
                if int(entry["kblock"]) != k:
                    continue
                start, count = int(entry["columns"]["start"]), int(entry["columns"]["count"])
                rel = start - base
                if rel < 0 or rel % z or count <= 0 or count > z or rel + count > per:
                    bad(f"(c) bundle {bid} K-block {k} has an illegal span {start}+{count}")
                    continue
                for c in range(rel, rel + count):
                    cover[c] += 1
            if any(n != 1 for n in cover):
                bad(
                    f"(c) bundle {bid} K-block {k} does not cover its {per} columns "
                    "exactly once"
                )
        chips_here = {e["_macro"] for e in got}
        declared = set(b.get("macros") or [])
        if chips_here != declared:
            bad(f"(c) bundle {bid} declares macros {sorted(declared)}, occupies {sorted(chips_here)}")
        if got:
            start = min(int(e["tick"]) for e in got)
            end = max(int(e["tick"]) for e in got)
            if int(b["start_tick"]) != start or int(b["end_tick"]) != end:
                bad(f"(c) bundle {bid} declares ticks {b['start_tick']}..{b['end_tick']}, schedule says {start}..{end}")
            if int(b["serial_ticks"]) != end - start + 1:
                bad(f"(c) bundle {bid} declares the wrong serial_ticks")

    # (d) the chain: hop gap, and the seam when the chip changes
    macro_chip = {m["id"]: m.get("chip") for m in smacros}
    for prev, nxt in zip(sbundles, sbundles[1:]):
        same_chip = prev.get("chip") == nxt.get("chip")
        need = int(prev["end_tick"]) + hop + (0 if same_chip else seam_ticks)
        if int(nxt["start_tick"]) < need:
            bad(
                f"(d) bundle {nxt['id']} starts at tick {nxt['start_tick']} but the "
                f"chain allows no earlier than {need}"
            )
        if bool(nxt.get("seam_before")) != (not same_chip):
            bad(f"(d) bundle {nxt['id']} declares seam_before {nxt.get('seam_before')} on a chip change of {not same_chip}")
        if int(nxt.get("stall_ticks", -1)) != int(nxt["start_tick"]) - need:
            bad(f"(d) bundle {nxt['id']} declares the wrong stall_ticks")
    if bool(sbundles[0].get("seam_before")):
        bad("(d) the first bundle opens the chain and crosses no seam")
    if int(sbundles[0].get("stall_ticks", -1)) != int(sbundles[0]["start_tick"]):
        bad("(d) the first bundle's stall is its own start tick, counted from tick 0")
    for b in sbundles:
        for mid in b.get("macros") or []:
            if macro_chip.get(mid) != b.get("chip"):
                bad(f"(d) bundle {b['id']} sits on chip {b.get('chip')} but uses macro {mid} of chip {macro_chip.get(mid)}")

    # (e) three tokens of the rotation
    for token in range(3):
        busy: Dict[Tuple[str, int], int] = {}
        for m in smacros:
            for entry in m.get("slots") or []:
                tick = int(entry["tick"]) + token * beat_ticks
                key = (m["id"], tick)
                if key in busy:
                    bad(f"(e) token {token}: macro {m['id']} fires twice at tick {tick}")
                busy[key] = entry["slot"]
        for m in smacros:
            for entry in m.get("slots") or []:
                tick = int(entry["tick"]) + token * beat_ticks
                if busy.get((m["id"], tick)) is None:
                    bad(f"(e) token {token}: macro {m['id']} slot {entry['slot']} never fires")
                if tick % mux != int(entry["slot"]):
                    bad(f"(e) token {token}: macro {m['id']} slot {entry['slot']} fires off its phase")

    # (f) the ledgers, re-derived
    led = slots.get("ledgers") or {}
    macros_used = sum(1 for m in smacros if m.get("slots"))
    stranded = sum(len(m.get("stranded_slots") or []) for m in smacros)
    real_cols = sum(int(e["columns"]["count"]) for m in smacros for e in m.get("slots") or [])
    ragged = (used_slots * z - real_cols) * rows
    stranded_cells = stranded * z * rows
    wasted = ragged + stranded_cells
    l_ticks = int(sbundles[-1]["end_tick"]) - int(sbundles[0]["start_tick"]) + 1
    seams = sum(1 for b in sbundles if b.get("seam_before"))
    chips_min = math.ceil(used_slots / (chip_macros * mux)) if mux else 0
    l_min = 2 * len(sbundles) - 1 + seam_ticks * max(0, chips_min - 1)
    want = {
        "L_ticks": l_ticks,
        "L_ns": l_ticks * tick_ns,
        "D": math.ceil(l_ticks / beat_ticks) if beat_ticks else 0,
        "L_min_ticks": l_min,
        "D_min": math.ceil(l_min / beat_ticks) if beat_ticks else 0,
        "macros_used": macros_used,
        "chips_used": len(schips),
        "slots_total": len(smacros) * mux,
        "slots_used": used_slots,
        "slots_stranded": stranded,
        "cells_total": len(smacros) * mux * z * rows,
        "cells_used": real_cols * rows,
        "cells_ragged": ragged,
        "cells_stranded": stranded_cells,
        "cells_wasted": wasted,
        "waste_frac": (wasted / (len(smacros) * mux * z * rows)) if smacros else 0.0,
        "seams": seams,
        "seams_min": max(0, chips_min - 1),
        "stall_ticks": sum(int(b.get("stall_ticks", 0)) for b in sbundles),
        "beat_ticks": beat_ticks,
        "beat_ns": mux * tick_ns,
        "macro_duty_mean": (used_slots / (len(smacros) * mux)) if smacros and mux else 0.0,
        "macro_duty_max": max(len(m.get("slots") or []) / mux for m in smacros),
        "throughput_tokens_per_s": 1.0 / (mux * tick_ns * 1e-9),
        # throughput / D, so Little's law closes on the emitted ledgers.
        "per_user_tokens_per_s": (
            1.0 / (math.ceil(l_ticks / beat_ticks) * beat_ticks * tick_ns * 1e-9)
            if beat_ticks else 0.0
        ),
        "per_user_ceiling_tokens_per_s": 1.0 / (l_ticks * tick_ns * 1e-9),
    }
    for key, expect in want.items():
        if key not in led:
            bad(f"(f) ledgers do not declare {key}")
            continue
        got = led[key]
        if isinstance(expect, float):
            if not isinstance(got, (int, float)) or abs(float(got) - expect) > max(
                1e-9, abs(expect) * 1e-9
            ):
                bad(f"(f) ledgers.{key} declares {got}, the tables give {expect}")
        elif int(got) != int(expect):
            bad(f"(f) ledgers.{key} declares {got}, the tables give {expect}")

    # (g) containment, both directions, and tiles against the slot table
    chip_lists = {c["id"]: list(c.get("macros") or []) for c in schips}
    for c in schips:
        if int(c.get("capacity_macros", -1)) != len(c.get("macros") or []):
            bad(f"(g) chip {c['id']} declares {c.get('capacity_macros')} macro slots and lists {len(c.get('macros') or [])}")
        for mid in c.get("macros") or []:
            if macro_chip.get(mid) != c["id"]:
                bad(f"(g) chip {c['id']} lists macro {mid}, whose chip is {macro_chip.get(mid)}")
    for m in smacros:
        if m["id"] not in chip_lists.get(m.get("chip"), []):
            bad(f"(g) macro {m['id']} names chip {m.get('chip')}, which does not list it")

    tiles = doc.get("tiles") or []
    amacros = {m["id"]: m for m in doc.get("macros") or []}
    tile_by_macro: Dict[str, List[Dict[str, Any]]] = {}
    for t in tiles:
        mid = t.get("macro")
        if mid not in amacros:
            bad(f"(g) tile {t.get('id')} names macro {mid}, which is not in macros[]")
            continue
        if t.get("id") not in (amacros[mid].get("tiles") or []):
            bad(f"(g) macro {mid} does not list tile {t.get('id')}")
        tile_by_macro.setdefault(mid, []).append(t)
    for mid, am in amacros.items():
        for tid in am.get("tiles") or []:
            if not any(t.get("id") == tid for t in tiles):
                bad(f"(g) macro {mid} lists tile {tid}, which is not in tiles[]")
    for m in smacros:
        entries = m.get("slots") or []
        ts = tile_by_macro.get(m["id"], [])
        if len(ts) != len(entries):
            bad(f"(g) macro {m['id']} holds {len(entries)} occupied slots and {len(ts)} tiles")
            continue
        spans = sorted((int(t["columns"]["start"]), int(t["columns"]["count"])) for t in ts)
        want_spans = sorted(
            (int(e["slot"]) * z, int(e["columns"]["count"])) for e in entries
        )
        if spans != want_spans:
            bad(f"(g) macro {m['id']}: tile column spans do not match its slot table")
    return {"ok": not v, "violations": v}
