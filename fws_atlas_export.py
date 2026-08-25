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

"""`fws_atlas/1` export — QIF P3.6.

One function, one contract: turn a real :class:`fws_mapping.FwsMapping` (or a
PD pair) into the document `docs/qif/atlas/SCHEMA.md` freezes, and let the P5
loader refuse it if it is wrong. The schema is FROZEN; this module conforms to
it and never negotiates with it.

Two consequences of being a REAL producer rather than a fixture:

* ``provenance.invented_fields`` is **empty**. Every value here is computed
  from the mapping or read from a device card; the ``fixture_invented`` marks
  that carried the hand-authored fixture drop away.
* ``macros[].duty_cycle`` is ``null`` and there is no time metric anywhere.
  P3 places; P4 prices (A1). A zero would draw as an empty bar and read as a
  measurement, so the field says "unknown" in the only way the schema has.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from fws_mapping import BoundaryRow, FwsMapping, PDPair, Relaxation

ATLAS_SCHEMA = "fws_atlas/1"
AUTHORED = "2026-08-23"
AUTHOR = "fws_atlas_export.export_atlas (QIF P3.6), from fws_mapping.build_mapping"


def _card_block(mapping: FwsMapping) -> Dict[str, object]:
    device = mapping.device
    card = device.card
    fabric = device.digital_card.fabric
    cols_adc = int(card.stored_columns_per_set)
    mux = int(card.column_sets_per_macro)
    # A card that declares neither width says so with 0/0 (P2.1: "both 0 means
    # the card declares no cell width and slicing is off"). The atlas needs a
    # positive stored-weight width, so an undeclared one falls back to the
    # config's PARAMETER precision and the basis string names the fallback.
    declared_cell = int(getattr(card, "bits_per_cell", 0) or 0)
    declared_weight = int(getattr(card, "weight_bits", 0) or 0)
    weight_bits = declared_weight or max(
        1, int(round(float(mapping.hw.sw_config.precision.parameters) * 8))
    )
    bits_per_cell = declared_cell if declared_cell > 0 else None
    slicing = bool(getattr(card, "slicing_enabled", False))
    analog_area = device.macro_footprint_mm2()
    digital_area = getattr(device.digital_card, "area_mm2", 0.0) or 0.0
    return OrderedDict(
        (
            (
                _analog_card_key(mapping),
                OrderedDict(
                    (
                        (
                            "label",
                            f"analog macro, {card.params.rows} x ({cols_adc} ADC x {mux} mux)",
                        ),
                        ("kind", "analog_macro"),
                        (
                            "capacity",
                            OrderedDict(
                                (
                                    ("rows", int(card.params.rows)),
                                    ("cols_adc", cols_adc),
                                    ("mux", mux),
                                    ("stored_cols", cols_adc * mux),
                                )
                            ),
                        ),
                        ("area_mm2", float(analog_area) if analog_area else None),
                        ("bits_per_cell", None if bits_per_cell is None else int(bits_per_cell)),
                        ("weight_bits", weight_bits),
                        ("slicing", bool(slicing)),
                        (
                            "basis",
                            "cim.analog / cim.cards of the hardware config; area_mm2 = "
                            "CimDeviceModel.macro_footprint_mm2() (P2.5: area per array "
                            "divided by the card's 3D height); weight_bits "
                            + (
                                "= cim.cards.weight_bits"
                                if declared_weight
                                else "is undeclared on this card and falls back to "
                                     "sw_param.precision.parameters"
                            ),
                        ),
                    )
                ),
            ),
            (
                _digital_card_key(mapping),
                OrderedDict(
                    (
                        (
                            "label",
                            f"shared digital chiplet, {fabric.rows} x {fabric.cols} "
                            f"{fabric.model} engine",
                        ),
                        ("kind", "digital_engine"),
                        (
                            "capacity",
                            OrderedDict(
                                (
                                    ("rows", int(fabric.rows)),
                                    ("cols_adc", int(fabric.cols)),
                                    ("mux", 1),
                                    ("stored_cols", int(fabric.cols)),
                                )
                            ),
                        ),
                        ("area_mm2", float(digital_area) if digital_area else None),
                        ("bits_per_cell", None),
                        (
                            "weight_bits",
                            max(1, int(mapping.hw.sw_config.precision.activations * 8)),
                        ),
                        ("slicing", False),
                        (
                            "basis",
                            "cim.fabric / cim.cards of the hardware config. It stores no "
                            "weights: weight_bits is the activation width it computes on, "
                            "and area_mm2 is null when the card declares none — the atlas "
                            "draws the unknown hatch rather than a guess",
                        ),
                    )
                ),
            ),
        )
    )


def _analog_card_key(mapping: FwsMapping) -> str:
    card = mapping.device.card
    return f"analog_{card.params.rows}x{card.stored_columns_per_set}x{card.column_sets_per_macro}"


def _digital_card_key(mapping: FwsMapping) -> str:
    fabric = mapping.device.digital_card.fabric
    return f"digital_{fabric.model}_{fabric.rows}x{fabric.cols}"


def _model_block(mapping: FwsMapping) -> Dict[str, object]:
    p = mapping.device.params
    model = mapping.model
    decode_len = int(getattr(model, "decode_len", 0) or 0)
    attention = getattr(model, "attention", None)
    return OrderedDict(
        (
            ("id", mapping.model_id),
            # The label is the model's OWN id, never the config's ``model_type``.
            # ``model_type`` is a pricing carrier (Granite-4.0-H-Tiny declares
            # ``llama`` for the SwiGLU gated MLP), so printing it as the model
            # name tells a reader the wrong model. When the two differ the
            # mapping raises a ``model_type_carrier`` disclosure that names both.
            ("label", str(mapping.model_id)),
            ("layers", int(p.num_layers)),
            ("hidden_dim", int(p.hidden_dim)),
            ("intermediate_size", int(p.intermediate_size)),
            ("vocab_size", int(p.vocab_size)),
            (
                "attention",
                OrderedDict(
                    (
                        (
                            "type",
                            str(getattr(attention, "attention_type", "mha")).lower()
                            if attention is not None
                            else "none",
                        ),
                        ("num_heads", int(p.num_heads)),
                        ("head_dim", int(p.head_dim)),
                    )
                ),
            ),
            ("precision", str(mapping.hw.sw_config.precision.tensor_format)),
            ("blocks", list(mapping.blocks)),
            ("endpoint_blocks", list(mapping.endpoint_blocks)),
            (
                "serving",
                OrderedDict(
                    (
                        ("batch", int(p.batch_size)),
                        ("prefill_len", max(0, int(p.seq_len) - decode_len)),
                        ("decode_len", decode_len),
                        ("seq_len", int(p.seq_len)),
                    )
                ),
            ),
        )
    )


def _ids(mapping: FwsMapping) -> Tuple[str, Dict[int, str], Dict[int, str]]:
    system = mapping.system_id
    chip_ids = {
        chip.chip_id: f"{system}.chip.{chip.chip_id:03d}" for chip in mapping.chips
    }
    macro_ids = {
        macro.macro_id: f"{system}.mac.{macro.macro_id:06d}" for macro in mapping.macros
    }
    return system, chip_ids, macro_ids


def _pool_basis(pool) -> str:
    """Why this macro's pool is this wide, and what a zero area MEANS.

    Every ``digital_pool`` block carries a basis, so the per-unit derivation and
    P4.5's timeline-measured width are never mistaken for each other, and a
    ``0.0`` area reads as an ABSENT LAW rather than a measured zero (the energy
    block's coverage vocabulary, applied to the one other declared-zero here).
    """
    return (
        "P2.5's per-unit derivation (D12): the width that keeps the pool "
        "non-blocking for ONE concurrent op. area_mm2 is "
        + (
            f"{float(pool.area_mm2):.6g} mm2"
            if float(pool.area_mm2) > 0
            else "0.0 = UNCOVERED: no card declares a pool area law, so this is "
            "an absent law and not a measured zero"
        )
    )


def _system_document(
    mapping: FwsMapping,
    boundary_rows: Sequence[BoundaryRow],
    duty_cycles: Optional[Mapping[int, float]] = None,
    pool_sizing: Optional[Mapping[int, Mapping[str, object]]] = None,
) -> Dict[str, object]:
    """The chips / macros / tiles / links of one system, with atlas ids.

    ``duty_cycles`` is P4's ``{macro_id: busy union / makespan}`` measurement
    (P4.2). Absent, every ``macros[].duty_cycle`` stays ``null`` — P3 has no
    timeline and a 0.0 would draw as a measured empty bar.

    ``pool_sizing`` is P4.5's timeline-measured per-macro pool width. Absent,
    ``macros[].digital_pool`` carries P2.5's per-unit derivation, which is what
    a placement-only export can honestly say. Present, it carries the SAME
    number the report's pool table prints for that macro (D21: one accounting
    per metric, across artifacts as well as within one).
    """
    system, chip_ids, macro_ids = _ids(mapping)
    primary = mapping.primary_group
    device = mapping.device
    pool = device.digital_pool_sizing()
    endpoint_blocks = set(mapping.endpoint_blocks)
    card_analog = _analog_card_key(mapping)
    card_digital = _digital_card_key(mapping)
    analog_card = device.card
    slicing = device.n_slices > 1

    chips: List[Dict[str, object]] = []
    macros: List[Dict[str, object]] = []
    tiles: List[Dict[str, object]] = []
    tile_index = 0

    for chip in mapping.chips:
        chips.append(
            OrderedDict(
                (
                    ("id", chip_ids[chip.chip_id]),
                    ("system", system),
                    ("pool", chip.pool),
                    (
                        "label",
                        f"{chip.label}"
                        + (f", layers {list(chip.layers)}" if chip.layers else ""),
                    ),
                    ("capacity", OrderedDict((("macro_slots", int(chip.macro_slots)),))),
                    ("groups", chip.shard.as_dict()),
                    ("primary_group", primary),
                    ("occupancy", mapping.chip_occupancy(chip.chip_id)),
                    ("macros", [macro_ids[mid] for mid in chip.macro_ids]),
                )
            )
        )

    for macro in mapping.macros:
        macro_tile_ids: List[str] = []
        for tile in macro.tiles:
            tile_id = f"{system}.tile.{tile_index:06d}"
            tile_index += 1
            macro_tile_ids.append(tile_id)
            owner = tile.owner
            columns_start = min(tile.site.column_sets) * int(
                analog_card.stored_columns_per_set
            )
            tiles.append(
                OrderedDict(
                    (
                        ("id", tile_id),
                        ("macro", macro_ids[macro.macro_id]),
                        (
                            "owner",
                            OrderedDict(
                                (
                                    ("model", mapping.model_id),
                                    (
                                        "layer",
                                        None if owner.op in endpoint_blocks else int(owner.layer),
                                    ),
                                    ("block", owner.op),
                                    ("expert", None if owner.expert < 0 else int(owner.expert)),
                                )
                            ),
                        ),
                        (
                            "shape",
                            OrderedDict(
                                (("K", int(tile.rows_used)), ("N", int(tile.logical_columns)))
                            ),
                        ),
                        (
                            "slice",
                            None
                            if not slicing
                            else OrderedDict(
                                (
                                    ("index", int(tile.slice_index)),
                                    ("of", int(device.n_slices)),
                                    ("bits", int(analog_card.bits_per_cell)),
                                )
                            ),
                        ),
                        ("groups", macro.shard.as_dict()),
                        ("primary_group", primary),
                        (
                            "columns",
                            OrderedDict(
                                (
                                    ("start", int(columns_start)),
                                    ("count", int(tile.logical_columns)),
                                )
                            ),
                        ),
                    )
                )
            )
        macros.append(
            OrderedDict(
                (
                    ("id", macro_ids[macro.macro_id]),
                    ("chip", chip_ids[macro.chip_id]),
                    ("card", card_analog if macro.pool == "analog" else card_digital),
                    ("tiles", macro_tile_ids),
                    ("occupancy", mapping.macro_occupancy(macro.macro_id)),
                    # P4 owns duty cycles; a 0.0 here would draw as a measured
                    # empty bar, so the field says unknown the only way it can
                    # until P4 hands one in.
                    (
                        "duty_cycle",
                        None
                        if duty_cycles is None or macro.macro_id not in duty_cycles
                        else float(duty_cycles[macro.macro_id]),
                    ),
                    (
                        "digital_pool",
                        None
                        if macro.pool != "analog"
                        else (
                            OrderedDict(pool_sizing[macro.macro_id])
                            if pool_sizing is not None
                            and macro.macro_id in pool_sizing
                            else OrderedDict(
                                (
                                    ("shift_add_units", int(pool.adders)),
                                    ("act_lanes", int(pool.total_lanes)),
                                    ("area_mm2", float(pool.area_mm2)),
                                    ("basis", _pool_basis(pool)),
                                )
                            )
                        ),
                    ),
                    ("reserved", macro.reserved),
                )
            )
        )

    links: List[Dict[str, object]] = []
    for row in boundary_rows:
        if not row.crosses_link or row.src_chip == row.dst_chip:
            continue
        links.append(
            OrderedDict(
                (
                    ("id", f"{system}.link.{row.boundary_id}"),
                    ("from", chip_ids[row.src_chip]),
                    ("to", chip_ids[row.dst_chip]),
                    ("role", row.role),
                    ("bytes", float(row.bytes_per_unit)),
                    ("per", row.unit),
                    ("label", row.label),
                    ("basis", row.basis),
                )
            )
        )
    return {"chips": chips, "macros": macros, "tiles": tiles, "links": links}


def _metrics(mapping: FwsMapping, boundary_rows: Sequence[BoundaryRow]) -> List[Dict[str, object]]:
    """Placement figures only. P3 has no timeline and prints no time (A1)."""
    summary = mapping.summary()
    footprint = mapping.device.macro_footprint_mm2()
    slots = int(summary["analog_macro_slots"])
    out = [
        OrderedDict(
            (
                ("key", f"{mapping.system_id}.analog_chips"),
                ("label", "enumerated analog chiplets"),
                ("value", int(summary["analog_chips"])),
                ("unit", "chips"),
                (
                    "basis",
                    "fws_mapping.build_mapping enumerated them one by one; a chip count "
                    "here is never a division (D21, AUDIT finding 1)",
                ),
            )
        ),
        OrderedDict(
            (
                ("key", f"{mapping.system_id}.macros_owned"),
                ("label", "analog macro slots holding at least one tile"),
                ("value", int(summary["macros_holding_tiles"])),
                ("unit", "macros"),
                ("basis", "placed tiles grouped by macro (A3)"),
            )
        ),
        OrderedDict(
            (
                ("key", f"{mapping.system_id}.macros_unowned"),
                ("label", "enumerated analog macro slots holding no tile"),
                ("value", int(summary["unowned_macro_slots"])),
                ("unit", "macros"),
                ("basis", "ADJ-5: unowned capacity is legal, and it is reported"),
            )
        ),
        OrderedDict(
            (
                ("key", f"{mapping.system_id}.unowned_columns"),
                ("label", "stored analog columns no tile holds"),
                ("value", int(summary["unowned_columns"])),
                ("unit", "columns"),
                (
                    "basis",
                    "stored columns of every enumerated analog macro minus the logical "
                    "widths the resident tiles hold; counts both empty slots and the "
                    "padding inside a claimed column set (ADJ-4: a mux slot is the "
                    "smallest allocatable unit)",
                ),
            )
        ),
        OrderedDict(
            (
                ("key", f"{mapping.system_id}.tiles"),
                ("label", "placed tiles"),
                ("value", int(summary["tiles"])),
                ("unit", "tiles"),
                ("basis", "CimDeviceModel.enumerate_tiles over every placed weight matrix"),
            )
        ),
        OrderedDict(
            (
                ("key", f"{mapping.system_id}.enumerated_macro_silicon"),
                ("label", "silicon of every enumerated analog macro slot"),
                ("value", float(slots * footprint)),
                ("unit", "mm2"),
                (
                    "basis",
                    f"{slots} enumerated analog macro slots x "
                    f"macro_footprint_mm2 = {footprint:.6g}. This is NOT "
                    "CimDeviceModel.total_area_mm2, which counts the arrays a model "
                    "needs; this counts the slots the machine has, unowned ones "
                    "included. Two names, two quantities (D21)",
                ),
            )
        ),
    ]
    crossing = [row for row in boundary_rows if row.crosses_link and row.src_chip != row.dst_chip]
    if crossing:
        out.append(
            OrderedDict(
                (
                    ("key", f"{mapping.system_id}.boundary_bytes"),
                    ("label", f"bytes crossing all drawn links per {crossing[0].unit}"),
                    ("value", float(sum(row.bytes_per_unit for row in crossing))),
                    ("unit", "bytes"),
                    (
                        "basis",
                        "sum of the P3.5 boundary table's link-crossing rows; each row "
                        "carries its own byte law in links[].basis",
                    ),
                )
            )
        )
    return out


def _merge_disclosures(
    entries: Sequence[Tuple[str, str, str, str]], system_ids: Sequence[str]
) -> List[Dict[str, object]]:
    """One entry per CONSTRAINT KEY, naming every system that disclosed it.

    ``entries`` is ``(system_id, constraint, value, reason)`` in document order.
    A constraint EVERY system discloses identically speaks for the document and
    is printed as it stands. Anything else — a constraint only one half of a PD
    pair raises, or one whose halves lower different decode windows — becomes
    ONE entry whose value names the systems it belongs to. Two entries under one
    key would read as two claims, and dropping the second would lose a
    disclosure; D21 allows neither.

    A single-system document has exactly one contributor per entry and it is
    always the whole document, so this is a no-op there and the shipped
    exports do not move by a byte.
    """
    everyone = list(dict.fromkeys(system_ids))
    order: List[str] = []
    variants: Dict[str, "OrderedDict[Tuple[str, str], List[str]]"] = OrderedDict()
    for system_id, constraint, value, reason in entries:
        if constraint not in variants:
            variants[constraint] = OrderedDict()
            order.append(constraint)
        contributors = variants[constraint].setdefault((value, reason), [])
        if system_id not in contributors:
            contributors.append(system_id)
    out: List[Dict[str, object]] = []
    for constraint in order:
        items = list(variants[constraint].items())
        (value, reason), contributors = items[0]
        if len(items) == 1 and set(contributors) == set(everyone):
            merged_value, merged_reason = value, reason
        else:
            merged_value = "; ".join(
                f"{', '.join(who)}: {text}" for (text, _why), who in items
            )
            merged_reason = reason + "".join(
                f" ({', '.join(who)}: {why})" for (_text, why), who in items[1:]
            )
        out.append(
            OrderedDict(
                (
                    ("constraint", constraint),
                    ("value", merged_value),
                    ("reason", merged_reason),
                )
            )
        )
    return out


def _handoff_endpoint(mapping: FwsMapping, *, last: bool):
    """Which chip the PD handoff is drawn from / to, and why (D16).

    The KV a prefill system hands over lives in the shared digital chiplets'
    activation tier (D13; P4.3 puts KV there), so the handoff is drawn between
    those chiplets when the inventory has any, and falls back to the backbone
    end of the chain when it has none. The choice is a DRAWING decision and
    says so in the link's basis: the byte count is the handoff's own, and no
    endpoint here recomputes it.
    """
    digital = [chip for chip in mapping.chips if chip.pool == "digital"]
    if digital:
        chip = digital[-1] if last else digital[0]
        return chip, f"{mapping.system_id}'s shared digital chiplet {chip.chip_id}"
    chip = mapping.chips[-1] if last else mapping.chips[0]
    return chip, (
        f"{mapping.system_id}'s {'last' if last else 'first'} chip "
        f"({chip.chip_id}) — this inventory declares no shared digital chiplet"
    )


def export_atlas(
    mappings: Sequence[FwsMapping],
    *,
    boundary_rows: Optional[Sequence[Sequence[BoundaryRow]]] = None,
    title: str,
    subtitle: str = "",
    reference_command: str = "",
    handoff=None,
    duty_cycles: Optional[Sequence[Optional[Mapping[int, float]]]] = None,
    extra_metrics: Optional[Sequence[Sequence[Mapping[str, object]]]] = None,
    pool_sizing: Optional[Sequence[Optional[Mapping[int, Mapping[str, object]]]]] = None,
    extra_relaxations: Optional[Sequence[Sequence[Mapping[str, object]]]] = None,
    service_links: Optional[Sequence[Sequence[Mapping[str, object]]]] = None,
) -> Dict[str, object]:
    """Build one ``fws_atlas/1`` document from one or two real mappings.

    ``mappings`` is one unified system, or the prefill/decode pair of a PD
    machine (D16), which the atlas renders side by side.

    ``duty_cycles`` and ``extra_metrics`` are P4's contributions, one entry per
    mapping: ``FwsEvaluation.atlas_duty_cycles()`` fills
    ``macros[].duty_cycle`` and ``FwsEvaluation.atlas_metrics()`` appends
    labeled timing metrics beside the placement figures. Omitting both leaves
    the document byte-identical to a placement-only export — P3 has no timeline
    (A1) and must never print one.

    ``service_links`` is P4's too: ``FwsEvaluation.atlas_service_links()``
    returns the ``svc`` rows — one per (shared digital chiplet -> analog chip)
    service relationship, carrying the MEASURED per-beat bytes that chip's act
    x act work moves in both directions. They are PRICED rows and therefore
    appear only on a priced export; a placement-only document has no timeline
    to measure them on and prints none.
    """
    if not mappings:
        raise ValueError("export_atlas needs at least one mapping")
    rows_per_system = (
        [tuple(mapping.boundary_table()) for mapping in mappings]
        if boundary_rows is None
        else [tuple(rows) for rows in boundary_rows]
    )
    systems: List[Dict[str, object]] = []
    chips: List[Dict[str, object]] = []
    macros: List[Dict[str, object]] = []
    tiles: List[Dict[str, object]] = []
    links: List[Dict[str, object]] = []
    metrics: List[Dict[str, object]] = []
    relaxations: List[Relaxation] = []
    # (system_id, constraint, value, reason) of P3's placement disclosures and
    # of P4's, kept apart because a constraint P3 already states is P3's to
    # word: the priced path adds detail to it, never a second entry.
    placement_disclosures: List[Tuple[str, str, str, str]] = []
    priced_disclosures: List[Tuple[str, str, str, str]] = []
    cards: Dict[str, object] = OrderedDict()

    duty_per_system = list(duty_cycles or [None] * len(mappings))
    metrics_per_system = list(extra_metrics or [()] * len(mappings))
    pool_per_system = list(pool_sizing or [None] * len(mappings))
    relax_per_system = list(extra_relaxations or [()] * len(mappings))
    service_per_system = list(service_links or [()] * len(mappings))
    if (
        len(duty_per_system) != len(mappings)
        or len(metrics_per_system) != len(mappings)
        or len(pool_per_system) != len(mappings)
        or len(relax_per_system) != len(mappings)
        or len(service_per_system) != len(mappings)
    ):
        raise ValueError(
            "export_atlas: duty_cycles / extra_metrics / pool_sizing / "
            "extra_relaxations / service_links take ONE entry per mapping, so a PD "
            "pair cannot silently borrow the prefill system's timeline for its "
            "decode half."
        )
    priced = any(entry is not None for entry in duty_per_system) or any(
        bool(entry) for entry in metrics_per_system
    )
    system_ids = [mapping.system_id for mapping in mappings]

    for index, (mapping, rows) in enumerate(zip(mappings, rows_per_system)):
        cards.update(_card_block(mapping))
        systems.append(
            OrderedDict(
                (
                    ("id", mapping.system_id),
                    ("role", mapping.role),
                    ("phase", mapping.phase),
                    ("label", mapping.label),
                    ("parallelism", dict(mapping.degrees)),
                    ("models", [_model_block(mapping)]),
                )
            )
        )
        block = _system_document(
            mapping, rows, duty_per_system[index], pool_per_system[index]
        )
        chips.extend(block["chips"])
        macros.extend(block["macros"])
        tiles.extend(block["tiles"])
        links.extend(block["links"])
        # The svc rows are already atlas-shaped: they are MEASURED off P4's
        # timeline (bytes per beat), which _system_document has no access to.
        links.extend(OrderedDict(entry) for entry in service_per_system[index])
        metrics.extend(_metrics(mapping, rows))
        metrics.extend(OrderedDict(entry) for entry in metrics_per_system[index])
        for relaxation in mapping.relaxations(rows):
            # A PRICED document retires ``op_durations``: it says every op
            # carries a zero duration, which stopped being true the moment P4's
            # timeline entered this file. Disclosing a gap that no longer exists
            # is its own dishonesty.
            if priced and relaxation.constraint == "op_durations":
                continue
            if relaxation not in relaxations:
                relaxations.append(relaxation)
            # Recorded even when an earlier system said exactly the same thing:
            # the merge needs every contributor to know whether a constraint
            # speaks for the document or for one half of it.
            placement_disclosures.append(
                (
                    mapping.system_id,
                    relaxation.constraint,
                    relaxation.value,
                    relaxation.reason,
                )
            )
        # P4's disclosures ride the same document as its metrics. Merged by
        # CONSTRAINT KEY, not by identity: one constraint listed twice with two
        # wordings reads as two claims (D21). The merge spans SYSTEMS as well as
        # producers — a PD pair prices two timelines, and its halves disclose
        # the same constraints with different numbers.
        for entry in relax_per_system[index]:
            key = str(entry.get("constraint", ""))
            if key:
                priced_disclosures.append(
                    (
                        mapping.system_id,
                        key,
                        str(entry.get("value", "")),
                        str(entry.get("reason", "")),
                    )
                )

    if handoff is not None and len(mappings) == 2:
        src = mappings[0]
        dst = mappings[1]
        src_chip, src_why = _handoff_endpoint(src, last=True)
        dst_chip, dst_why = _handoff_endpoint(dst, last=False)
        links.append(
            OrderedDict(
                (
                    ("id", "link.pd_handoff"),
                    ("from", f"{src.system_id}.chip.{src_chip.chip_id:03d}"),
                    ("to", f"{dst.system_id}.chip.{dst_chip.chip_id:03d}"),
                    ("role", "pd"),
                    ("bytes", float(handoff.total_bytes)),
                    ("per", "request"),
                    ("label", "prefill -> decode handoff (D16): the KV of the request plus the state that seeds decode"),
                    (
                        "basis",
                        handoff.basis
                        + f" Drawn from {src_why} to {dst_why}: the KV tier lives in "
                        "the shared digital chiplets' activation SRAM (D13, P4.3), so "
                        "those are the endpoints the bytes actually leave and enter. "
                        "The ENDPOINT CHOICE is a drawing decision; the byte count is "
                        "fws_mapping.pd_handoff_bytes and nothing here recomputes it.",
                    ),
                )
            )
        )

    document = OrderedDict(
        (
            ("schema", ATLAS_SCHEMA),
            ("title", title),
            ("subtitle", subtitle),
            (
                "provenance",
                OrderedDict(
                    (
                        ("authored", AUTHORED),
                        ("author", AUTHOR),
                        (
                            "reference_run",
                            OrderedDict(
                                (
                                    ("command", reference_command),
                                    ("report", "none: this document IS the producer's output"),
                                    (
                                        "parallelism",
                                        ", ".join(
                                            f"{axis} = {mappings[0].degrees[axis]}"
                                            for axis in ("tp", "ep", "pp")
                                        ),
                                    ),
                                )
                            ),
                        ),
                        (
                            "grounded",
                            [
                                "chips[], macros[] and tiles[] are the enumerated placement "
                                "fws_mapping.build_mapping produced",
                                "cards[] are the P2 device cards of the hardware config",
                                (
                                    "macros[].digital_pool is P4.5's TIMELINE-MEASURED "
                                    "per-macro pool sizing (D12) — the same number the "
                                    "run's pool table prints, not a second derivation"
                                    if priced
                                    else "macros[].digital_pool is the derived per-macro "
                                    "pool sizing (P2.5, D12)"
                                ),
                                (
                                    "links[] are the P3.5 boundary table's "
                                    "link-crossing rows, plus one MEASURED svc row per "
                                    "(shared digital chiplet -> analog chip) service "
                                    "relationship whose bytes are read off P4's "
                                    "timeline"
                                    if priced
                                    else "links[] are the P3.5 boundary table's "
                                    "link-crossing rows"
                                ),
                                (
                                    "metrics[] are placement counts plus P4's timing "
                                    "metrics, every one read off ONE timeline and every "
                                    "one naming its basis"
                                    if priced
                                    else "metrics[] are placement counts; every one names "
                                    "its basis"
                                ),
                            ],
                        ),
                        # A real producer declares an EMPTY invented list: nothing
                        # here is a placeholder of the right shape.
                        ("invented_fields", []),
                        (
                            "note",
                            (
                                "P3 placed and P4 priced (A1). Every time-valued field "
                                "here — macros[].duty_cycle and the timing metrics — is "
                                "read off the ONE timeline of the run named above; "
                                "nothing here is a period or a closed form. Read "
                                "relaxations[] before quoting a number: a windowed "
                                "decode run discloses its window there. The "
                                "hand-authored fixture's placeholder marks are absent "
                                "because every value here is computed."
                                if priced
                                else "P3 places and P4 prices (A1). This document "
                                "carries no time: macros[].duty_cycle is null and no "
                                "metric is a rate or a period. The hand-authored "
                                "fixture's placeholder marks are absent because every "
                                "value here is computed."
                            ),
                        ),
                    )
                ),
            ),
            ("cards", cards),
            ("systems", systems),
            ("chips", chips),
            ("macros", macros),
            ("tiles", tiles),
            ("links", links),
            ("metrics", metrics),
            (
                "relaxations",
                _merge_disclosures(placement_disclosures, system_ids)
                + _merge_disclosures(
                    [
                        item
                        for item in priced_disclosures
                        if item[1] not in {entry.constraint for entry in relaxations}
                    ],
                    system_ids,
                ),
            ),
        )
    )
    return document


def export_pd_atlas(
    pair: PDPair,
    *,
    title: str,
    subtitle: str = "",
    reference_command: str = "",
    duty_cycles: Optional[Sequence[Optional[Mapping[int, float]]]] = None,
    extra_metrics: Optional[Sequence[Sequence[Mapping[str, object]]]] = None,
    pool_sizing: Optional[Sequence[Optional[Mapping[int, Mapping[str, object]]]]] = None,
    extra_relaxations: Optional[Sequence[Sequence[Mapping[str, object]]]] = None,
    service_links: Optional[Sequence[Sequence[Mapping[str, object]]]] = None,
):
    """The two-system document of a PD machine (D16, ADJ-7).

    The P4 arguments are the same per-mapping sequences ``export_atlas`` takes,
    in prefill-then-decode order. Each half is priced on its OWN timeline: a
    PD pair is two inventories (D16), so borrowing one half's duty cycles for
    the other would be one measurement printed as two.
    """
    return export_atlas(
        [pair.prefill, pair.decode],
        title=title,
        subtitle=subtitle,
        reference_command=reference_command,
        handoff=pair.handoff,
        duty_cycles=duty_cycles,
        extra_metrics=extra_metrics,
        pool_sizing=pool_sizing,
        extra_relaxations=extra_relaxations,
        service_links=service_links,
    )


def write_atlas_json(document: Dict[str, object], path) -> str:
    """Write the document deterministically; the same mapping is the same file."""
    text = json.dumps(document, indent=1, ensure_ascii=False) + "\n"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return text
