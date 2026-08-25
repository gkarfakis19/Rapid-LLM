"""QIF P3 mapping tests: the mapping object, the placed DAG, the atlas export.

Five things can rot silently in a mapping and every one of them is a wrong
picture rather than a crash: a chip that stops being an enumerated object, a
macro that loses its named owner, a shard group that gets re-inferred instead
of constructed, a boundary that gets absorbed instead of reported, and an
export that stops conforming to the frozen fws_atlas/1 schema. The suite pins
all five.

The atlas conformance tests drive the P5 loader itself
(``tests/test_qif_atlas._run_core``, skipped when node is absent) against the
document THIS workstream emits, so the same 27 rules judge the fixture and the
producer.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

import cim_timing
import config
import fws_atlas_export
import fws_mapping
from fws_mapping import MappingError
from program.fws_build import (
    LAW_ANALOG_GEMM,
    LAW_FABRIC,
    LAW_LINK,
    LAW_POOL,
    LAW_SLICE_REDUCTION,
    ServingPoint,
    annotations_of,
    build_fws_program,
)
from program.groups import CommunicatorFactory
from program.placement import DeviceCoord, PlacementError
from test_qif_atlas import _run_core  # the P5 loader harness, driven on OUR export

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HW_DIR = PROJECT_ROOT / "configs" / "hardware-config"
MODEL_DIR = PROJECT_ROOT / "configs" / "model-config"
ATLAS_DIR = PROJECT_ROOT / "docs" / "qif" / "atlas"

FWS_LLAMA7B = HW_DIR / "fws_cim_llama7b.yaml"
FWS_MOE = HW_DIR / "fws_cim_moe.yaml"
FWS_T1 = HW_DIR / "fws_cim_optima_t1.yaml"
LLAMA2_7B_FWS_INF = MODEL_DIR / "llama2_7b_fws_inf.yaml"
MOE_SMALL_FWS_INF = MODEL_DIR / "moe_small_fws_inf.yaml"
VIT_HUGE_64 = MODEL_DIR / "vit_huge_story_64_inf.yaml"
QWEN35 = MODEL_DIR / "qwen3_5_4b_inf.yaml"
LFM2 = MODEL_DIR / "lfm2_2p6b_inf.yaml"

#: The checked-in P3 export the atlas can be opened with.
P3_ATLAS = ATLAS_DIR / "p3_llama7b_tp2.json"

#: WAVE F (D29). A DECLARED `mapping:` block now means a FILLED PIPELINE, and
#: the filled pipeline refuses a declared batch and declared endpoints BY NAME.
#: The fixtures in this file are PLACEMENT fixtures over the batch-4 Llama2-7B
#: and MoE-small workloads of the retired regime — chips, tiles, groups, pp
#: annotation, PD inventories — none of which the regime changes. They
#: therefore declare the regime they mean, by name, instead of being rewritten
#: into a different machine: the placement questions this file asks have the
#: same answers under both regimes, and the D29 regime has its own file
#: (tests/test_qif_pipeline.py).
LOCKSTEP = {"regime": "lockstep"}

#: The pairs the mapping is built over throughout this file.
PAIRS = (
    (FWS_LLAMA7B, LLAMA2_7B_FWS_INF, "LLM"),
    (FWS_MOE, MOE_SMALL_FWS_INF, "LLM"),
    (FWS_T1, VIT_HUGE_64, "VIT"),
)


def _load_yaml(path):
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def _hw(path, mutate=None):
    raw = copy.deepcopy(_load_yaml(path))
    if mutate is not None:
        mutate(raw)
    config.convert(raw)
    return config.HWConfig.from_dict(raw)


def _mapping(hw_path, model_path, mode="LLM", mutate=None, **kwargs):
    hw = _hw(hw_path, mutate)
    model = config.parse_config(str(model_path), mode)
    return fws_mapping.build_mapping(hw, model, **kwargs)


@pytest.fixture(scope="module")
def llama_mapping():
    return _mapping(FWS_LLAMA7B, LLAMA2_7B_FWS_INF)


@pytest.fixture(scope="module")
def llama_tp2_mapping():
    def mutate(raw):
        raw["parallelism"]["tp"] = 2
        raw["mapping"] = {"chips": 8, "shared_chiplets": 2, "parallelism": {"tp": 2}, **LOCKSTEP}

    return _mapping(FWS_LLAMA7B, LLAMA2_7B_FWS_INF, mutate=mutate)


# ---------------------------------------------------------------------------
# P3.1 — the mapping: config block
# ---------------------------------------------------------------------------


def test_mapping_block_parses_every_declared_field():
    raw = {
        "chips": 8,
        "macros_per_chip": 120,
        "shared_chiplets": 2,
        "parallelism": {"tp": 2, "ep": 1, "pp": 2},
        "layers_per_chip": [8, 8, 8, 8],
        "membership": {"pp": [0, 0, 1, 1, 0, 0, 1, 1]},
        "decode_window": 3,
    }
    parsed = config.MappingConfig.from_dict(raw)
    assert parsed.system.chips == 8
    assert parsed.system.parallelism.tp == 2 and parsed.system.parallelism.pp == 2
    assert parsed.system.layers_per_chip == (8, 8, 8, 8)
    assert parsed.system.membership["pp"] == (0, 0, 1, 1, 0, 0, 1, 1)
    assert parsed.system.decode_window == 3
    assert not parsed.is_pd


@pytest.mark.parametrize(
    "raw,needle",
    [
        ({"chiplets": 4}, "chiplets"),
        ({"chips": 0}, "mapping.chips must be >= 1"),
        ({"chips": "many"}, "mapping.chips must be an integer"),
        ({"parallelism": {"dp": 2}}, "mapping.parallelism does not support"),
        ({"layers_per_chip": "greedy"}, "layers_per_chip must be an integer"),
        ({"membership": {"tp": []}}, "membership.tp must be a non-empty list"),
        ({"pd": {"prefill": {}}}, "requires BOTH prefill and decode"),
    ],
)
def test_mapping_block_refuses_by_name(raw, needle):
    with pytest.raises(ValueError) as excinfo:
        config.MappingConfig.from_dict(raw)
    assert needle in str(excinfo.value)


def test_mapping_block_needs_the_fws_cim_device_class():
    raw = copy.deepcopy(_load_yaml(HW_DIR / "a100_80GB.yaml"))
    raw["mapping"] = {"chips": 2}
    config.convert(raw)
    with pytest.raises(ValueError) as excinfo:
        config.HWConfig.from_dict(raw)
    assert "requires device_class: fws_cim" in str(excinfo.value)


def test_absent_mapping_block_is_the_derived_dedicated_mapping(llama_mapping):
    # ABSENT means today's placement, and "today's placement" is a checkable
    # statement: cim.chip.layers_per_chip's chip split and the array census's
    # per-chip usage, macro for macro.
    hw = _hw(FWS_LLAMA7B)
    assert hw.mapping_config is None
    device = llama_mapping.device
    counts = device.chip_layer_counts()
    usage = device.chip_array_usage()
    analog = llama_mapping.analog_chips()
    assert len(analog) == len(counts)
    for chip, layer_count, arrays in zip(analog, counts, usage):
        assert len(chip.layers) == layer_count
        owned = sum(1 for mid in chip.macro_ids if llama_mapping.macro(mid).tiles)
        assert owned == arrays


@pytest.mark.parametrize("hw_path,model_path,mode", PAIRS)
def test_derived_mapping_reproduces_the_array_census_on_every_shipped_pair(
    hw_path, model_path, mode
):
    mapping = _mapping(hw_path, model_path, mode)
    device = mapping.device
    owned = sum(1 for macro in mapping.macros if macro.pool == "analog" and macro.tiles)
    assert owned == sum(device.chip_array_usage())
    assert owned == device.total_arrays()


@pytest.mark.parametrize("hw_path,model_path,mode", PAIRS)
def test_every_macro_names_an_owner_and_chips_are_integers(hw_path, model_path, mode):
    # D21: chips are integers and every macro has a named owner. A macro with
    # tiles names them through the tiles; a macro without tiles names its
    # reservation. A macro with neither is the anonymous resource this catches.
    mapping = _mapping(hw_path, model_path, mode)
    for chip in mapping.chips:
        assert isinstance(chip.chip_id, int)
        assert isinstance(chip.macro_slots, int) and chip.macro_slots > 0
        assert len(chip.macro_ids) == chip.macro_slots
    for macro in mapping.macros:
        assert bool(macro.tiles) != bool(macro.reserved), macro.macro_id
        for tile in macro.tiles:
            assert tile.owner.model and tile.owner.op


def test_capacity_overflow_is_a_named_hard_error():
    with pytest.raises(MappingError) as excinfo:
        _mapping(
            FWS_LLAMA7B,
            LLAMA2_7B_FWS_INF,
            mutate=lambda raw: raw["cim"]["chip"].update({"arrays_per_chip": 8}),
        )
    message = str(excinfo.value)
    assert message.startswith("[assembly]")
    assert "macro slots but the chip declares 8" in message


def test_declared_chip_count_is_checked_against_the_enumerated_placement():
    with pytest.raises(MappingError) as excinfo:
        _mapping(
            FWS_LLAMA7B,
            LLAMA2_7B_FWS_INF,
            mutate=lambda raw: raw.update({"mapping": {"chips": 7, **LOCKSTEP}}),
        )
    assert "enumerates 4 analog chips" in str(excinfo.value)
    assert "never a quotient" in str(excinfo.value)


def test_one_degree_has_one_home():
    with pytest.raises(MappingError) as excinfo:
        _mapping(
            FWS_LLAMA7B,
            LLAMA2_7B_FWS_INF,
            mutate=lambda raw: raw.update({"mapping": {"parallelism": {"tp": 2}, **LOCKSTEP}}),
        )
    assert "disagrees with parallelism.tp = 1" in str(excinfo.value)


def test_unowned_capacity_is_legal_and_reported(llama_mapping):
    # ADJ-5. The llama7b chips declare 120 slots and use 104, so the report
    # must SAY 16 per chip rather than shrink the chip to fit.
    summary = llama_mapping.summary()
    assert summary["unowned_macro_slots"] > 0
    assert summary["unowned_columns"] > 0
    report = llama_mapping.report()
    assert "unowned:" in report and "legal and reported" in report


def test_pp_is_an_independent_annotation_not_the_chip_index():
    # ADJ-5: chip index is not implicitly pp. Declaring pp = 2 over four chips
    # annotates them; the hardware parallelism.pp stays 1.
    mapping = _mapping(
        FWS_LLAMA7B,
        LLAMA2_7B_FWS_INF,
        mutate=lambda raw: raw.update({"mapping": {"parallelism": {"pp": 2}, **LOCKSTEP}}),
    )
    assert mapping.hw.sch_config.pp == 1
    assert mapping.degrees["pp"] == 2
    stages = [chip.shard.pp for chip in mapping.analog_chips()]
    assert stages == [0, 0, 1, 1]
    assert any("INDEPENDENT annotation" in note for note in mapping.notes)
    # pp annotates the chip chain; it does not cut it. The boundary between
    # the two stages is still a drawn boundary.
    act = [row for row in mapping.boundary_table() if row.boundary_id.startswith("act.")]
    assert [row.boundary_id for row in act] == ["act.c0->c1", "act.c1->c2", "act.c2->c3"]


def test_declared_pp_membership_is_validated_by_name():
    def mapping_with(membership):
        return _mapping(
            FWS_LLAMA7B,
            LLAMA2_7B_FWS_INF,
            mutate=lambda raw: raw.update(
                {"mapping": {"parallelism": {"pp": 2}, "membership": {"pp": membership}, **LOCKSTEP}}
            ),
        )

    with pytest.raises(MappingError) as short:
        mapping_with([0, 1])
    assert "lists 2 entries but the placement enumerates 4" in str(short.value)
    with pytest.raises(MappingError) as out_of_range:
        mapping_with([0, 1, 2, 1])
    assert "declares stages 0..1" in str(out_of_range.value)
    with pytest.raises(MappingError) as empty:
        mapping_with([0, 0, 0, 0])
    assert "no chip is assigned to stage(s) [1]" in str(empty.value)
    assert [chip.shard.pp for chip in mapping_with([1, 1, 0, 0]).analog_chips()] == [1, 1, 0, 0]


def test_shared_chiplet_count_is_an_input_with_a_reported_suggestion():
    mapping = _mapping(
        FWS_LLAMA7B,
        LLAMA2_7B_FWS_INF,
        mutate=lambda raw: raw.update({"mapping": {"shared_chiplets": 1, **LOCKSTEP}}),
    )
    assert len(mapping.digital_chips()) == 1
    assert mapping.shared_chiplet_suggestion == len(mapping.analog_chips())
    assert any("derived suggestion" in note for note in mapping.notes)
    with pytest.raises(MappingError) as excinfo:
        _mapping(
            FWS_LLAMA7B,
            LLAMA2_7B_FWS_INF,
            mutate=lambda raw: raw.update({"mapping": {"shared_chiplets": 0, **LOCKSTEP}}),
        )
    assert "act x act compute" in str(excinfo.value)


# ---------------------------------------------------------------------------
# P3.1 — residency through cim.allocation (D10)
# ---------------------------------------------------------------------------


def _llama_allocation(entries):
    def mutate(raw):
        raw["cim"]["allocation"] = {"assignments": entries}

    return mutate


def test_a_user_allocation_places_the_tiles_it_names():
    # D10: allocation is user-specified, priced and validated by the tool. The
    # enumerator supplies the K range (the pinned row convention); the config
    # supplies the macro and the column sets, and nothing else.
    entries = [
        {"model": "llama", "layer": 0, "op": "o_proj", "macro": 119, "column_sets": [0, 1, 2, 3]}
    ]
    mapping = _mapping(FWS_LLAMA7B, LLAMA2_7B_FWS_INF, mutate=_llama_allocation(entries))
    owner = cim_timing.TileOwner("llama", 0, "o_proj", -1, 0)
    tiles = mapping.tiles_for(owner)
    assert len(tiles) == 1
    assert tiles[0].site.macro_id == 119
    assert tiles[0].site.column_sets == (0, 1, 2, 3)
    assert tiles[0].site.row_start == tiles[0].k_start
    assert mapping.macro(119).tiles == tiles


def test_a_user_allocation_with_the_wrong_tile_count_is_refused_by_name():
    entries = [
        {"model": "llama", "layer": 0, "op": "ffn2", "macro": 100, "column_sets": [0, 1, 2, 3]}
    ]
    with pytest.raises(MappingError) as excinfo:
        _mapping(FWS_LLAMA7B, LLAMA2_7B_FWS_INF, mutate=_llama_allocation(entries))
    assert "names 1 placement(s)" in str(excinfo.value)
    assert "tiles into 3" in str(excinfo.value)


def test_a_user_allocation_outside_the_owners_chip_is_refused_by_name():
    entries = [
        {"model": "llama", "layer": 0, "op": "o_proj", "macro": 200, "column_sets": [0, 1, 2, 3]}
    ]
    with pytest.raises(MappingError) as excinfo:
        _mapping(FWS_LLAMA7B, LLAMA2_7B_FWS_INF, mutate=_llama_allocation(entries))
    assert "outside chip 0's slot range" in str(excinfo.value)


# ---------------------------------------------------------------------------
# P3.2 — the TileSite row convention (the Wave A leftover), pinned
# ---------------------------------------------------------------------------


def test_the_row_convention_is_pinned_and_validated(llama_mapping):
    # Wave A left TileSite.row_start/row_end with two producer conventions and
    # no validation. P3 pins ONE: a site's row range IS the owner matrix's K
    # range. Every placed tile obeys it, and a tile that does not is refused.
    rows = int(llama_mapping.device.card.params.rows)
    fws_mapping.validate_tile_row_convention(llama_mapping.tiles, rows)
    for tile in llama_mapping.tiles:
        assert tile.site.row_start == tile.k_start
        assert tile.site.row_end == tile.k_end
        assert 0 <= tile.site.row_start < tile.site.row_end <= tile.site.row_start + rows

    physical = [
        cim_timing.Tile(
            owner=tile.owner,
            k_start=tile.k_start,
            k_end=tile.k_end,
            n_start=tile.n_start,
            n_end=tile.n_end,
            slice_index=tile.slice_index,
            site=cim_timing.TileSite(
                macro_id=tile.site.macro_id,
                row_start=0,
                row_end=rows,
                column_sets=tile.site.column_sets,
            ),
        )
        for tile in llama_mapping.tiles
        if tile.k_start > 0
    ]
    assert physical, "the llama placement must carry a row-partitioned matrix"
    with pytest.raises(MappingError) as excinfo:
        fws_mapping.validate_tile_row_convention(physical, rows)
    assert "owner_matrix_k_range" in str(excinfo.value)


# ---------------------------------------------------------------------------
# P3.2 — the placed op DAG
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def llama_program(llama_mapping):
    return build_fws_program(llama_mapping, decode_steps=1)


def test_the_dag_is_a_valid_program_with_one_annotation_per_op(llama_program):
    llama_program.validate()
    annotations = annotations_of(llama_program)
    assert len(annotations) == len(llama_program.ops)
    for op, annotation in zip(llama_program.ops, annotations):
        assert annotation.uid == op.uid


def test_every_op_is_placed_on_a_device_that_exists(llama_program, llama_mapping):
    inventory = {dev.device_id for dev in llama_mapping.devices}
    assert set(llama_program.devices) == inventory
    for annotation in annotations_of(llama_program):
        if annotation.kind == "transfer":
            assert annotation.src_device in inventory
            assert annotation.dst_device in inventory
        else:
            assert annotation.device_id in inventory


def test_p3_places_and_prices_nothing(llama_program):
    # A1 / D21: one accounting per metric, and the duration accounting is P4's.
    for op in llama_program.ops:
        if hasattr(op, "duration"):
            assert op.duration == (0.0,)
    assert "P4 prices" in llama_program.meta.misc["fws_duration_basis"]


def test_weight_gemms_land_on_analog_macros_and_act_x_act_on_the_shared_chiplet(llama_program):
    # A2 / D13: the device class of an op is a placement decision, and this is
    # the whole of it stated as an assertion.
    by_block = {}
    for annotation in annotations_of(llama_program):
        by_block.setdefault(annotation.block, set()).add(
            (annotation.device_class, annotation.law)
        )
    for block in ("qkv", "o_proj", "ffn1"):
        assert by_block[block] <= {
            ("analog_macro", LAW_ANALOG_GEMM),
            ("link", LAW_LINK),
        }, block
    for block in ("attention_qk", "attention_softmax", "attention_pv"):
        assert by_block[block] == {("shared_digital", LAW_FABRIC)}, block
    for block in ("norm", "residual", "activation"):
        assert by_block[block] == {("macro_pool", LAW_POOL)}, block


def test_one_weight_gemm_op_per_macro_per_output_block(llama_program, llama_mapping):
    # The structural contract with P4: an op is priceable because it names the
    # tiles on ONE macro. Counting them per layer is how a lost or duplicated
    # placement becomes a test failure.
    device = llama_mapping.device
    per_layer = {}
    for annotation in annotations_of(llama_program):
        if annotation.kind != "weight_gemm" or annotation.phase != "prefill":
            continue
        if annotation.layer is None or annotation.block in ("lm_head",):
            continue
        per_layer.setdefault(annotation.layer, []).append(annotation)
    arrays_per_layer = device.arrays_per_layer()
    for layer, ops in per_layer.items():
        assert len(ops) == arrays_per_layer, layer
        assert len({op.device_id for op in ops}) == len(ops)


def test_the_dag_dependencies_run_in_pipeline_order(llama_program):
    # Dependency sanity: every dep precedes its op (validate_program's V1), and
    # the layer a dep belongs to never runs ahead of its consumer's layer.
    annotations = annotations_of(llama_program)
    for op in llama_program.ops:
        for dep in op.deps:
            assert dep < op.uid
            producer = annotations[dep]
            consumer = annotations[op.uid]
            if producer.layer is not None and consumer.layer is not None:
                if producer.phase == consumer.phase and producer.step == consumer.step:
                    assert producer.layer <= consumer.layer


def test_attention_depends_on_qkv_and_o_proj_depends_on_attention(llama_program):
    annotations = annotations_of(llama_program)
    by_name = {op.name: op for op in llama_program.ops}
    qk = by_name["prefill0.L0.attn.qk"]
    reached = {annotations[dep].block for dep in qk.deps}
    assert reached == {"qkv"}
    o_proj = [
        op
        for op in llama_program.ops
        if op.name.startswith("prefill0.L0.o_proj") and annotations[op.uid].kind == "weight_gemm"
    ]
    assert o_proj
    for op in o_proj:
        assert any(annotations[dep].block == "o_proj" for dep in op.deps)


def test_every_transfer_carries_bytes_and_a_boundary_id(llama_program):
    transfers = [a for a in annotations_of(llama_program) if a.kind == "transfer"]
    assert transfers
    for annotation in transfers:
        assert annotation.bytes_moved > 0
        assert annotation.boundary_id
        assert annotation.device_class == "link"


def test_each_tp_shard_lowers_its_own_chain(llama_tp2_mapping):
    # A tp shard has its own norms, its own attention on its own shared
    # chiplet, its own FFN. Lowering one shard and reusing its pool ops for the
    # other would place work no device does.
    program = build_fws_program(llama_tp2_mapping, decode_steps=1)
    annotations = annotations_of(program)
    pool_by_shard = {}
    for annotation in annotations:
        if annotation.kind == "pool":
            pool_by_shard.setdefault(annotation.groups["tp"], []).append(annotation)
    assert sorted(pool_by_shard) == [0, 1]
    assert len(pool_by_shard[0]) == len(pool_by_shard[1])
    fabric = {a.groups["tp"]: a.device_id for a in annotations if a.kind == "fabric"}
    assert sorted(fabric) == [0, 1]
    assert fabric[0] != fabric[1]
    tp_transfers = [a for a in annotations if a.boundary_id.startswith("tp.")]
    assert tp_transfers
    for annotation in tp_transfers:
        assert annotation.bytes_moved > 0
        assert "tp" in annotation.group_keys


def test_expert_parallelism_moves_the_experts_and_prices_the_hops():
    # ep > 1 takes the routed experts off the layer chip (cim.chip's own
    # moe_expert_pool law), so the dispatch and combine become real boundaries
    # instead of the nothing they are at ep = 1.
    def mutate(raw):
        raw["cim"]["chip"]["moe_expert_parallel"] = 2
        raw["cim"]["chip"]["arrays_per_chip"] = 400

    mapping = _mapping(FWS_MOE, MOE_SMALL_FWS_INF, mutate=mutate)
    assert mapping.degrees["ep"] == 2
    assert mapping.primary_group == "ep"
    expert_chips = [chip for chip in mapping.analog_chips() if chip.role == "expert_pool"]
    assert expert_chips
    assert {chip.shard.ep for chip in expert_chips} == {0, 1}
    for chip in expert_chips:
        blocks = {
            tile.owner.op
            for macro_id in chip.macro_ids
            for tile in mapping.macro(macro_id).tiles
        }
        assert blocks <= {"ffn1_routed", "ffn2_routed"}, blocks

    program = build_fws_program(mapping, decode_steps=1)
    hops = [a for a in annotations_of(program) if a.boundary_id.startswith("ep.")]
    assert hops
    assert {a.boundary_id.split(".")[1] for a in hops} == {"dispatch", "combine"}
    device = mapping.device
    act_bytes = float(mapping.hw.sw_config.precision.activations)
    expected = device.moe_dispatch_bytes(device.params.batch_size, act_bytes) / 2
    dispatch = [a for a in hops if ".dispatch." in a.boundary_id and a.phase == "decode"]
    assert dispatch and all(a.bytes_moved == expected for a in dispatch)
    rows = [row for row in mapping.boundary_table() if row.role == "ep"]
    assert rows and all(row.bytes_per_unit == expected for row in rows)


def test_bit_slicing_puts_a_real_shift_add_op_on_the_hosting_macros_pool(tmp_path):
    # D11: the shift-and-add tree is priced, never absorbed. It is P4's
    # "P2 priced reduction" row, and it only exists when a card slices — so
    # this is the one configuration that proves the row is reachable.
    def mutate(raw):
        raw["cim"]["chip"]["arrays_per_chip"] = 4000
        raw["cim"]["cards"] = {
            "ctt": {
                "kind": "analog_macro",
                "device": "ctt",
                "bits_per_cell": 2,
                "weight_bits": 8,
                "bank_depth": 1,
            }
        }

    mapping = _mapping(FWS_T1, VIT_HUGE_64, mode="VIT", mutate=mutate)
    assert mapping.device.n_slices == 4
    assert {tile.slice_index for tile in mapping.tiles} == {0, 1, 2, 3}
    program = build_fws_program(mapping)
    trees = [
        a
        for a in annotations_of(program)
        if a.kind == "reduction" and a.law == LAW_SLICE_REDUCTION
    ]
    assert trees
    for annotation in trees:
        assert annotation.device_class == "macro_pool"
        # the tree runs on the pool of the macro that holds its slices
        assert {tile.site.macro_id for tile in annotation.tiles} == {annotation.macro_id}
        assert len({tile.slice_index for tile in annotation.tiles}) == 4

    document = fws_atlas_export.export_atlas([mapping], title="sliced")
    assert document["cards"][next(iter(document["cards"]))]["slicing"] is True
    assert all(tile["slice"]["of"] == 4 for tile in document["tiles"])
    # R24/R25: the card's slicing flag and every tile's slice role are the
    # loader's business, so let the loader say it.
    report = _validate_with_the_p5_loader(document, tmp_path)
    assert report["errors"] == []
    # WAVE C AUDIT: this fixture declares arrays_per_chip = 4000 purely to make
    # the slicing row reachable, and P5's draw budget is now counted in RECTS
    # rather than macro slots -- 4000 macros with their tile bands and duty bars
    # are well past 10,000 rects, so the atlas says out loud that it will not
    # draw this chip. That is the rule working, not a mapping defect, and it is
    # the ONLY thing the loader may complain about here.
    assert report["warnings"] == ["R23 chips[sys.fws.chip.000].macros"]


def test_the_decode_series_is_a_bounded_window(llama_mapping):
    # ADJ-6: a window, and the truncation is disclosed rather than assumed.
    serving = ServingPoint.from_mapping(llama_mapping)
    assert serving.decode_len == 256
    assert serving.decode_steps == fws_mapping.DEFAULT_DECODE_WINDOW
    program = build_fws_program(llama_mapping, decode_steps=3)
    steps = {a.step for a in annotations_of(program) if a.phase == "decode"}
    assert steps == {0, 1, 2}
    disclosed = [r for r in llama_mapping.relaxations() if r.constraint == "decode_window"]
    assert disclosed and "extrapolates nothing" in disclosed[0].reason


@pytest.mark.parametrize("hw_path,model_path,mode", PAIRS)
def test_every_shipped_pair_lowers_into_a_valid_dag(hw_path, model_path, mode):
    mapping = _mapping(hw_path, model_path, mode)
    program = build_fws_program(mapping, decode_steps=1)
    program.validate()
    annotations = annotations_of(program)
    assert annotations
    assert {a.device_class for a in annotations} <= {
        "analog_macro",
        "macro_pool",
        "shared_digital",
        "link",
    }


@pytest.mark.parametrize("model_path,fabric_block", [(QWEN35, "delta_rule"), (LFM2, "short_conv")])
def test_hybrid_blocks_place_on_the_devices_their_decisions_name(model_path, fabric_block):
    # ADJ-3 puts the short conv on the per-macro pool; D13 puts the delta-rule
    # state update on the shared chiplet. The run path still REFUSES these
    # models (P1.4's gate holds until pricing lands); the mapper can place them.
    def mutate(raw):
        raw["cim"]["chip"]["arrays_per_chip"] = 4000
        raw["cim"]["chip"]["layers_per_chip"] = 40

    mapping = _mapping(FWS_LLAMA7B, model_path, mutate=mutate)
    program = build_fws_program(mapping, decode_steps=1)
    placed = {
        a.block: a.device_class for a in annotations_of(program) if a.block == fabric_block
    }
    assert placed, fabric_block
    expected = "macro_pool" if fabric_block == "short_conv" else "shared_digital"
    assert placed[fabric_block] == expected


# ---------------------------------------------------------------------------
# P3.2 — the widened device coordinate
# ---------------------------------------------------------------------------


def test_the_device_coordinate_is_widened_additively(llama_mapping):
    # P3 §4: a device also has a class and a host macro. DeviceCoord already
    # stores a free-form {axis: int} mapping, so the widening needs no edit to
    # program/placement.py and the GPU rank grid never sees these axes.
    analog = llama_mapping.devices_of_class("analog_macro")[0]
    coord = analog.coord
    assert isinstance(coord, DeviceCoord)
    assert coord.of("dev_class") == fws_mapping.DEVICE_CLASS_CODE["analog_macro"]
    assert coord.of("host_macro") == analog.macro_id
    assert coord.of("tp") == 0

    fabric = llama_mapping.devices_of_class("shared_digital")[0]
    with pytest.raises(PlacementError) as excinfo:
        fabric.coord.of("host_macro")
    assert "host_macro" in str(excinfo.value)

    # A GPU coordinate is untouched: same class, none of the new axes.
    gpu = DeviceCoord({"tp": 1, "pp": 3})
    assert sorted(gpu.coords) == ["pp", "tp"]
    assert gpu.with_(pp=4).of("pp") == 4


# ---------------------------------------------------------------------------
# P3.3 — tp/ep/pp annotations, CONSTRUCTED
# ---------------------------------------------------------------------------


def test_every_chip_macro_and_op_carries_the_three_axes(llama_tp2_mapping):
    for chip in llama_tp2_mapping.chips:
        assert set(chip.shard.as_dict()) == {"tp", "ep", "pp"}
    for macro in llama_tp2_mapping.macros:
        assert macro.shard.tp in (0, 1)
    program = build_fws_program(llama_tp2_mapping, decode_steps=1)
    for annotation in annotations_of(program):
        assert set(annotation.groups) == {"tp", "ep", "pp"}
        assert annotation.primary_group == "tp"


def test_group_membership_is_constructed_from_axis_and_coords(llama_tp2_mapping):
    # P3.3 / groups.py invariant P2: members come from (axis, coords) via
    # CommunicatorFactory and are never re-inferred from a participant count.
    mapping = llama_tp2_mapping
    factory = CommunicatorFactory(mapping.shard_layout)
    assert factory.spans(("tp",)) == 2
    device = mapping.devices_of_class("analog_macro")[0]
    key = mapping.device_group("tp", device.device_id)
    assert len(key.members) == 2
    assert tuple(sorted(key.members)) == key.members
    peers = [mapping.device_record(member) for member in key.members]
    assert {peer.shard.tp for peer in peers} == {0, 1}
    # every member plays the SAME role in its own shard
    assert len({peer.role_key for peer in peers}) == 1
    # and the shard ids the factory enumerated are exactly the tp siblings
    anchor = mapping.shard_id(device.shard)
    assert set(factory.members(("tp",), anchor)) == {
        mapping.shard_id(peer.shard) for peer in peers
    }


def test_the_dag_registers_the_groups_it_annotates_with(llama_tp2_mapping):
    program = build_fws_program(llama_tp2_mapping, decode_steps=1)
    used = set()
    for annotation in annotations_of(program):
        for key in annotation.group_keys.values():
            used.add(key)
    assert used
    assert used <= set(program.groups)
    for key in program.groups:
        assert tuple(sorted(key.members)) == key.members


def test_primary_group_names_the_first_axis_with_a_degree(llama_mapping, llama_tp2_mapping):
    # ADJ-7: P3 emits primary_group; the atlas invents no precedence rule.
    assert llama_mapping.degrees == {"tp": 1, "ep": 1, "pp": 1}
    assert llama_mapping.primary_group == "tp"
    assert llama_tp2_mapping.primary_group == "tp"
    pp_mapping = _mapping(
        FWS_LLAMA7B,
        LLAMA2_7B_FWS_INF,
        mutate=lambda raw: raw.update({"mapping": {"parallelism": {"pp": 2}, **LOCKSTEP}}),
    )
    assert pp_mapping.primary_group == "pp"


def test_a_group_axis_the_mapping_does_not_carry_is_refused(llama_mapping):
    device = llama_mapping.devices[0]
    with pytest.raises(MappingError) as excinfo:
        llama_mapping.device_group("cp", device.device_id)
    assert "not one of" in str(excinfo.value)


# ---------------------------------------------------------------------------
# P3.4 — the PD pair
# ---------------------------------------------------------------------------


def _pd_pair(prefill=None, decode=None):
    def mutate(raw):
        raw["mapping"] = {
            "pd": {
                "prefill": dict(prefill or {}, **LOCKSTEP),
                "decode": dict(decode or {}, **LOCKSTEP),
            }
        }

    hw = _hw(FWS_LLAMA7B, mutate)
    model = config.parse_config(str(LLAMA2_7B_FWS_INF), "LLM")
    return fws_mapping.build_pd_pair(hw, model)


def test_pd_is_two_inventories_two_mappings_one_handoff():
    pair = _pd_pair()
    assert pair.prefill.role == "prefill" and pair.prefill.phase == "prefill"
    assert pair.decode.role == "decode" and pair.decode.phase == "decode"
    assert pair.prefill.system_id != pair.decode.system_id
    assert pair.handoff.total_bytes > 0
    assert "No precision conversion is modeled" in pair.handoff.basis


def test_equal_pd_specs_reproduce_the_unified_machine():
    # D16's own test: setting both specs equal reproduces the non-disaggregated
    # machine. Both inventories must BE the unified mapping, placement included.
    unified = _mapping(FWS_LLAMA7B, LLAMA2_7B_FWS_INF)
    pair = _pd_pair()
    for half in (pair.prefill, pair.decode):
        assert half.summary()["analog_chips"] == unified.summary()["analog_chips"]
        assert half.summary()["macro_slots"] == unified.summary()["macro_slots"]
        assert half.summary()["tiles"] == unified.summary()["tiles"]
        assert [tile.site for tile in half.tiles] == [tile.site for tile in unified.tiles]
        assert [macro.reserved for macro in half.macros] == [
            macro.reserved for macro in unified.macros
        ]


def test_the_handoff_is_kv_plus_state_bytes_from_the_model():
    pair = _pd_pair()
    device = pair.prefill.device
    precision = pair.prefill.hw.sw_config.precision
    prefill_len = int(device.params.seq_len) - int(pair.prefill.model.decode_len)
    expected_kv = device.params.batch_size * device.kv_bytes_per_stream(
        prefill_len, float(precision.kv_cache), 1
    )
    expected_state = (
        device.params.batch_size * device.params.hidden_dim * float(precision.activations)
    )
    assert pair.handoff.kv_bytes == expected_kv
    assert pair.handoff.state_bytes == expected_state
    assert pair.handoff.time_s == cim_timing.CimDeviceModel.p2p_time_s(
        expected_kv + expected_state,
        pair.handoff.bandwidth_bytes_per_s,
        pair.handoff.latency_s,
    )


def test_pd_halves_may_differ_and_each_keeps_its_own_inventory():
    pair = _pd_pair(prefill={"shared_chiplets": 4}, decode={"shared_chiplets": 1})
    assert len(pair.prefill.digital_chips()) == 4
    assert len(pair.decode.digital_chips()) == 1


def test_a_pd_pair_needs_both_halves():
    hw = _hw(FWS_LLAMA7B)
    model = config.parse_config(str(LLAMA2_7B_FWS_INF), "LLM")
    with pytest.raises(MappingError) as excinfo:
        fws_mapping.build_pd_pair(hw, model)
    assert "BOTH a prefill spec and a decode spec" in str(excinfo.value)


# ---------------------------------------------------------------------------
# P3.5 — the boundary table
# ---------------------------------------------------------------------------


def test_boundary_bytes_come_from_the_law_that_already_owns_them(llama_mapping):
    rows = llama_mapping.boundary_table()
    device = llama_mapping.device
    act_bytes = float(llama_mapping.hw.sw_config.precision.activations)
    tokens = device.params.batch_size  # decode phase
    expected = device.boundary_bytes(tokens, act_bytes)
    act_rows = [row for row in rows if row.role == "act" and row.boundary_id.startswith("act.")]
    assert len(act_rows) == len(llama_mapping.analog_chips()) - 1
    for row in act_rows:
        assert row.bytes_per_unit == expected
        assert row.basis and row.unit == "decode_step"


def test_a_rate_needs_a_named_time_base(llama_mapping):
    with pytest.raises(MappingError) as excinfo:
        llama_mapping.boundary_table(period_s=1e-3)
    assert "period_basis" in str(excinfo.value)
    rows = llama_mapping.boundary_table()
    assert all(row.required_rate_bytes_per_s is None for row in rows)
    assert all(row.required_time_s >= 0 for row in rows)


def test_a_violated_boundary_is_named_reported_and_not_priced(llama_mapping):
    rows = llama_mapping.boundary_table(
        period_s=1e-9, period_basis="a deliberately impossible period, for the test"
    )
    violated = [row for row in rows if row.violated]
    assert violated
    for row in violated:
        assert row.boundary_id in row.disclosure
        assert "REPORTED and NOT priced" in row.disclosure
        assert row.required_rate_bytes_per_s > row.available_rate_bytes_per_s
        assert row.ratio > 1
    disclosed = [
        r for r in llama_mapping.relaxations(rows) if r.constraint == "boundary_bandwidth"
    ]
    assert disclosed and "optimistic" in disclosed[0].reason
    assert "[RELAXATION] boundary_bandwidth" in llama_mapping.report(rows)


def test_a_boundary_that_crosses_no_link_says_so(llama_mapping):
    rows = llama_mapping.boundary_table()
    kv = [row for row in rows if row.boundary_id.startswith("kv.")]
    assert kv
    for row in kv:
        assert row.crosses_link is False
        assert "crosses no p2p link" in row.basis


def test_tp_boundaries_appear_only_when_tp_does(llama_mapping, llama_tp2_mapping):
    assert not [row for row in llama_mapping.boundary_table() if row.role == "tp"]
    tp_rows = [row for row in llama_tp2_mapping.boundary_table() if row.role == "tp"]
    assert len(tp_rows) == len(llama_tp2_mapping.analog_chips())
    for row in tp_rows:
        assert "ring all-reduce" in row.basis
        assert row.src_chip != row.dst_chip


def test_the_report_states_every_relaxation_it_carries(llama_mapping):
    report = llama_mapping.report(llama_mapping.boundary_table())
    for constraint in ("network_congestion", "op_durations", "intra_chip_activation_movement"):
        assert f"[RELAXATION] {constraint}" in report


# ---------------------------------------------------------------------------
# P3.6 — the atlas export
# ---------------------------------------------------------------------------


def _validate_with_the_p5_loader(document, tmp_path):
    path = tmp_path / "atlas.json"
    fws_atlas_export.write_atlas_json(document, path)
    out = _run_core(
        """
        const v = ATLAS.validate(doc);
        console.log(JSON.stringify({
          errors: v.errors.map(e => e.rule + ' ' + e.field + ': ' + e.message),
          warnings: v.warnings.map(w => w.rule + ' ' + w.field),
        }));
        """,
        tmp_path,
        path,
    )
    return json.loads(out)


@pytest.mark.parametrize("hw_path,model_path,mode", PAIRS)
def test_the_export_passes_the_p5_loader_on_every_shipped_pair(
    hw_path, model_path, mode, tmp_path
):
    mapping = _mapping(hw_path, model_path, mode)
    document = fws_atlas_export.export_atlas([mapping], title=f"{hw_path.stem} placement")
    report = _validate_with_the_p5_loader(document, tmp_path)
    assert report["errors"] == []
    assert report["warnings"] == []


def test_the_tp2_export_passes_the_p5_loader(llama_tp2_mapping, tmp_path):
    document = fws_atlas_export.export_atlas([llama_tp2_mapping], title="llama7b tp=2")
    report = _validate_with_the_p5_loader(document, tmp_path)
    assert report["errors"] == []
    # tp = 2 puts two (axis, index) hue keys on screen; the palette holds four,
    # so the over-palette warning must not fire either.
    assert report["warnings"] == []


def test_the_pd_export_draws_two_systems_and_one_handoff_link(tmp_path):
    pair = _pd_pair()
    document = fws_atlas_export.export_pd_atlas(pair, title="llama7b PD")
    assert [system["role"] for system in document["systems"]] == ["prefill", "decode"]
    handoff = [link for link in document["links"] if link["role"] == "pd"]
    assert len(handoff) == 1
    assert handoff[0]["bytes"] == pair.handoff.total_bytes
    report = _validate_with_the_p5_loader(document, tmp_path)
    assert report["errors"] == []
    assert report["warnings"] == []


def test_a_real_producer_declares_no_invented_fields(llama_mapping):
    # SCHEMA.md's fixture_invented convention: a placeholder must never leave
    # the file wearing the clothes of a result. Nothing here is a placeholder,
    # so the list is empty and no object carries the mark.
    document = fws_atlas_export.export_atlas([llama_mapping], title="t")
    assert document["provenance"]["invented_fields"] == []
    assert "fixture_invented" not in json.dumps(document)


def test_the_export_carries_no_time_because_p3_has_no_timeline(llama_mapping):
    document = fws_atlas_export.export_atlas([llama_mapping], title="t")
    assert all(macro["duty_cycle"] is None for macro in document["macros"])
    for metric in document["metrics"]:
        assert metric["unit"] not in ("s", "us", "tok/s", "fps", "B/s")
        assert metric["basis"]


def test_the_export_declares_no_accuracy_field_anywhere(llama_mapping):
    # D23 is hard, and it binds the producer as well as the fixture.
    banned = ("accuracy", "perplexity", "top1", "top_1", "bleu", "error_rate")
    blob = json.dumps(fws_atlas_export.export_atlas([llama_mapping], title="t")).lower()
    for word in banned:
        assert word not in blob, word


def test_the_export_states_occupancy_the_loader_recomputes(llama_mapping):
    # The atlas computes only to refuse: it recomputes occupancy from the
    # enumerated spans and refuses a disagreement (R16/R17). This asserts the
    # producer's arithmetic directly, so a mismatch names the producer.
    document = fws_atlas_export.export_atlas([llama_mapping], title="t")
    stored = int(llama_mapping.device.card.stored_columns_per_set) * int(
        llama_mapping.device.card.column_sets_per_macro
    )
    tiles_by_macro = {}
    for tile in document["tiles"]:
        tiles_by_macro.setdefault(tile["macro"], []).append(tile)
    for macro in document["macros"]:
        if macro["card"].startswith("digital"):
            continue
        held = sum(tile["columns"]["count"] for tile in tiles_by_macro.get(macro["id"], ()))
        assert macro["occupancy"] == pytest.approx(held / stored, abs=1e-12)


def test_the_checked_in_export_is_the_one_the_tool_emits_today(tmp_path):
    # The atlas can be opened with this file, so a stale copy is a wrong
    # picture under the right title — the same failure the P5 embedded-blob
    # test guards. Regenerating and comparing is the whole check.
    import tools.fws_emit_atlas as emit

    _mapping_obj, _rows, document = emit.build(
        str(FWS_LLAMA7B),
        str(LLAMA2_7B_FWS_INF),
        tp=2,
        shared_chiplets=2,
    )
    regenerated = fws_atlas_export.write_atlas_json(document, tmp_path / "p3.json")
    assert regenerated == P3_ATLAS.read_text(encoding="utf-8"), (
        "docs/qif/atlas/p3_llama7b_tp2.json is stale; regenerate it with "
        "tools/fws_emit_atlas.py (the command is in its provenance block)"
    )


def test_the_checked_in_export_passes_the_p5_loader(tmp_path):
    out = _run_core(
        """
        const v = ATLAS.validate(doc);
        console.log(JSON.stringify({
          errors: v.errors.map(e => e.rule + ' ' + e.field + ': ' + e.message),
          warnings: v.warnings.map(w => w.rule + ' ' + w.field),
        }));
        """,
        tmp_path,
        P3_ATLAS,
    )
    report = json.loads(out)
    assert report["errors"] == []
    assert report["warnings"] == []


# ---------------------------------------------------------------------------
# A3 / D10's core concept, exercised: two OWNERS sharing one macro's columns
# ---------------------------------------------------------------------------


def _cosited_mapping():
    """Two different owners on one macro, on DISJOINT column sets.

    A3 says a macro's stored columns can host tiles from different layers,
    experts or models, and D10 makes that allocation user-specified. Every
    SHIPPED card is sized so one matrix fills a macro, so no shipped export puts
    two tiles on one macro and the legal co-siting path had no test at all — only
    the REFUSAL path (a doubly-claimed column set) did. This is the legal half.

    ``column_sets_per_tile: 1`` makes the mux slot the tile, and the one
    assignment moves layer 1's router — a 16-wide matrix that otherwise wastes a
    whole macro — onto two free column sets of layer 0's qkv macro.
    """
    raw = copy.deepcopy(yaml.safe_load(FWS_MOE.read_text()))
    raw["cim"]["allocation"] = {
        "column_sets_per_tile": 1,
        "assignments": [
            {
                "model": "llama",
                "layer": 1,
                "op": "router",
                "expert": -1,
                "shard": 0,
                "slice_index": 0,
                "macro": 1,
                "column_sets": [2],
            }
        ],
    }
    config.convert(raw)
    hw = config.HWConfig.from_dict(raw)
    model = config.parse_config(str(MOE_SMALL_FWS_INF), "LLM")
    return fws_mapping.build_mapping(hw, model)


def test_two_owners_can_share_one_macros_columns():
    mapping = _cosited_mapping()
    by_macro = {}
    for tile in mapping.tiles:
        by_macro.setdefault(tile.site.macro_id, []).append(tile)
    shared = {
        macro_id: tiles
        for macro_id, tiles in by_macro.items()
        if len({tile.owner.label for tile in tiles}) > 1
    }
    assert shared, "the point of this config is a macro with two owners"
    for macro_id, tiles in shared.items():
        claimed = [set(tile.site.column_sets) for tile in tiles]
        # Disjoint: sharing a macro is legal, sharing a COLUMN SET is not.
        for i, first in enumerate(claimed):
            for second in claimed[i + 1 :]:
                assert not (first & second), macro_id
        owners = {tile.owner.label for tile in tiles}
        assert len(owners) == 2
        assert all(owner for owner in owners)  # D21: every tile names its owner


def test_a_shared_macros_occupancy_counts_both_owners():
    mapping = _cosited_mapping()
    stored = int(mapping.device.card.stored_columns_per_set)
    per_macro = int(mapping.device.card.column_sets_per_macro)
    tiles = [tile for tile in mapping.tiles if tile.site.macro_id == 1]
    assert len({tile.owner.label for tile in tiles}) == 2
    macro = mapping.macro(1)
    # Three column sets claimed of four, by two owners.
    assert macro.claimed_column_sets == 3
    assert macro.claimed_column_sets < per_macro
    # held_columns are the LOGICAL widths: qkv fills its two sets, the 16-wide
    # router does not fill its one. Both are counted, neither is rounded up.
    assert macro.held_columns == 2 * stored + 16
    assert macro.held_columns < macro.claimed_column_sets * stored
    # The unowned columns of a SHARED macro are still reported, not absorbed.
    analog = [slot for slot in mapping.macros if slot.pool == "analog"]
    expected = sum(
        stored * per_macro - slot.held_columns for slot in analog
    )
    assert mapping.summary()["unowned_columns"] == expected
    # The shared macro contributes its OWN gap and nothing is double counted.
    assert (stored * per_macro - macro.held_columns) > 0


def test_a_co_sited_export_lets_the_loaders_overlap_rule_fire(tmp_path):
    """R13 ('two tiles overlap on one macro') needs a document it CAN corrupt.

    Driven through the P5 loader's own selfTest: on the shipped one-tile-per-
    macro exports that corruption cannot be constructed and the row reports
    "nothing fired", which is a coverage gap in the artifact rather than a
    conformance failure. This document closes it.
    """
    mapping = _cosited_mapping()
    rows = mapping.boundary_table()
    document = fws_atlas_export.export_atlas(
        [mapping],
        boundary_rows=[rows],
        title="co-sited macros (A3 / D10)",
        reference_command="tests/test_qif_mapping.py::_cosited_mapping",
    )
    path = tmp_path / "cosited.json"
    fws_atlas_export.write_atlas_json(document, path)
    out = _run_core(
        """
        const v = ATLAS.validate(doc);
        const rows = ATLAS.selfTest(doc);
        console.log(JSON.stringify({
          errors: v.errors.map(e => e.rule + ' ' + e.field),
          warnings: v.warnings.map(w => w.rule + ' ' + w.field),
          failed: rows.filter(r => !r.pass).map(r => r.rule + ' ' + r.what),
          rules: rows.map(r => r.rule),
        }));
        """,
        tmp_path,
        document=path,
    )
    report = json.loads(out)
    assert report["errors"] == []
    assert report["warnings"] == []
    # The whole point: the overlap corruption is CONSTRUCTIBLE here, so it fires.
    # (R22 still cannot be constructed on a ONE-shared-chiplet document; that is
    # a separate, named coverage gap and not what this test is about.)
    unfired = {row.split(" ", 1)[0] for row in report["failed"]}
    assert "R13" not in unfired, report["failed"]
    assert unfired <= {"R22"}, report["failed"]


# ---------------------------------------------------------------------------
# cim.allocation.slice_index — parsed AND honored (Wave D minors sweep)
# ---------------------------------------------------------------------------


def _slice_entry(slice_index, macro):
    """One `cim.allocation.assignments` entry, parsed by the real schema."""
    entry = {"model": "vit", "layer": 0, "op": "qkv", "macro": macro, "column_sets": [0]}
    if slice_index is not None:
        entry["slice_index"] = slice_index
    return config.CIMTileAssignment.from_dict(entry, 0)


def _slice_tiles(count):
    owner = cim_timing.TileOwner(model="vit", layer=0, op="qkv")
    return tuple(
        cim_timing.Tile(
            owner=owner,
            k_start=0,
            k_end=64,
            n_start=0,
            n_end=64,
            slice_index=s,
            site=cim_timing.TileSite(macro_id=s, row_start=0, row_end=64, column_sets=(0,)),
        )
        for s in range(count)
    )


def test_allocation_slice_index_defaults_to_unconstrained():
    # It used to default to 0, which made "absent" indistinguishable from a
    # claim of slice 0 — and on a 4-slice card every entry would then have
    # claimed the same slice.
    assert _slice_entry(None, 7).slice_index is None
    assert _slice_entry(3, 7).slice_index == 3


def test_allocation_honors_a_declared_slice_index():
    tiles = _slice_tiles(4)
    entries = [_slice_entry(s, 10 + s) for s in range(4)]
    placed = fws_mapping._apply_user_allocation(tiles, entries, "vit.L0.qkv.s0")
    assert [tile.site.macro_id for tile in placed] == [10, 11, 12, 13]
    assert [tile.slice_index for tile in placed] == [0, 1, 2, 3]


def test_allocation_refuses_a_slice_index_declaration_order_contradicts():
    # Declaration order is what places a tile. A declared slice that disagrees
    # with it is a contradiction, not a preference: refuse it by name.
    tiles = _slice_tiles(4)
    entries = [_slice_entry(s, 10 + s) for s in range(4)]
    entries[2] = _slice_entry(3, 12)
    with pytest.raises(MappingError, match="slice_index = 3 but declaration order"):
        fws_mapping._apply_user_allocation(tiles, entries, "vit.L0.qkv.s0")


def test_allocation_without_a_slice_index_still_places_every_slice():
    tiles = _slice_tiles(4)
    entries = [_slice_entry(None, 20 + s) for s in range(4)]
    placed = fws_mapping._apply_user_allocation(tiles, entries, "vit.L0.qkv.s0")
    assert [tile.site.macro_id for tile in placed] == [20, 21, 22, 23]


def _sliced_moe_mapping(assignments):
    """A SLICING card plus a user allocation, driven through the real path.

    The three tests above call the private placer with hand-built tiles. That
    proves the rule but not that it is REACHABLE: every shipped card stores a
    weight in one cell (`n_slices == 1`), so no shipped config ever produces a
    multi-slice tile for an allocation entry to contradict. This fixture
    declares a 2-bit cell on 8-bit weights, which slices every matrix four
    ways, and then goes through `config.HWConfig.from_dict` +
    `fws_mapping.build_mapping` like a user's file would.
    """
    raw = copy.deepcopy(yaml.safe_load(FWS_MOE.read_text()))
    raw["cim"]["cards"] = {
        "ctt": {
            "kind": "analog_macro",
            "device": "ctt",
            "bits_per_cell": 2,
            "weight_bits": 8,
            "bank_depth": 1,
        }
    }
    # Four slices of every matrix need four times the slots.
    raw["cim"]["chip"]["arrays_per_chip"] = 1200
    raw["cim"]["allocation"] = {"assignments": assignments}
    config.convert(raw)
    hw = config.HWConfig.from_dict(raw)
    model = config.parse_config(str(MOE_SMALL_FWS_INF), "LLM")
    return fws_mapping.build_mapping(hw, model)


def _router_slice_assignments(macro=1100):
    """Layer 1's router: a 16-wide matrix, so exactly one tile per slice."""
    return [
        {
            "model": "llama",
            "layer": 1,
            "op": "router",
            "expert": -1,
            "shard": 0,
            "slice_index": s,
            "macro": macro,
            "column_sets": [s],
        }
        for s in range(4)
    ]


def test_a_declared_slice_index_reaches_the_placer_through_a_real_config():
    mapping = _sliced_moe_mapping(_router_slice_assignments())
    assert mapping.device.n_slices == 4
    placed = [
        tile
        for tile in mapping.tiles
        if tile.owner.op == "router" and tile.owner.layer == 1
    ]
    assert [tile.slice_index for tile in placed] == [0, 1, 2, 3]
    assert {tile.site.macro_id for tile in placed} == {1100}
    assert [tile.site.column_sets for tile in placed] == [(0,), (1,), (2,), (3,)]


def test_a_contradicting_slice_index_is_refused_through_a_real_config():
    assignments = _router_slice_assignments()
    assignments[2]["slice_index"] = 3
    with pytest.raises(MappingError, match="slice_index = 3 but declaration order"):
        _sliced_moe_mapping(assignments)
