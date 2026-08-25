"""QIF P7.2: the dense packer (Invariant W) and the accumulator laws.

Four things can rot silently here and every one of them is a wrong NUMBER
rather than a crash: a bank that stops holding real weights, a waste figure
that stops being itemized, an accumulator that gets charged twice (or never),
and a dead fold that quietly grows machinery. This suite pins all four, and
every law carries a hand-computed point in its comment.

THE GEOMETRY the hand arithmetic below is done on (configs/hardware-config/
fws_cim_llama7b.yaml): rows = 4096, cols_adc = 1024, adc_mux = 4. So a macro
stores 4096 x 4096 = 16,777,216 cells; a whole-macro bank (the shipped
bank_depth) is 4096 wide and holds all of them; a single-mux-slot bank
(bank_depth: 1) is 1024 wide and holds 4,194,304.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import yaml

import cim_timing
import config
import fws_eval
import fws_mapping
from cim_timing import CANONICAL_WALK, DEAD_FOLDS, DeadFoldError, PackRequest, TileOwner
from fws_mapping import MappingError
from program.fws_build import build_fws_program

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HW_DIR = PROJECT_ROOT / "configs" / "hardware-config"
MODEL_DIR = PROJECT_ROOT / "configs" / "model-config"

FWS_LLAMA7B = HW_DIR / "fws_cim_llama7b.yaml"
FWS_MOE = HW_DIR / "fws_cim_moe.yaml"
FWS_GRANITE = HW_DIR / "fws_cim_granite_tiny.yaml"
FWS_QWEN = HW_DIR / "fws_cim_qwen3_5_4b.yaml"
FWS_T1 = HW_DIR / "fws_cim_optima_t1.yaml"

LLAMA2_7B = MODEL_DIR / "llama2_7b_fws_inf.yaml"
MOE_SMALL = MODEL_DIR / "moe_small_fws_inf.yaml"
GRANITE = MODEL_DIR / "granite_4_0_h_tiny_inf.yaml"
QWEN = MODEL_DIR / "qwen3_5_4b_inf.yaml"
VIT_HUGE_64 = MODEL_DIR / "vit_huge_story_64_inf.yaml"


def _hw(path, mutate=None):
    raw = copy.deepcopy(yaml.safe_load(Path(path).read_text()))
    if mutate is not None:
        mutate(raw)
    config.convert(raw)
    return config.HWConfig.from_dict(raw)


def _device(path, model_path, mode="LLM", mutate=None):
    hw = _hw(path, mutate)
    model = config.parse_config(str(model_path), mode)
    return cim_timing.CimDeviceModel(hw, model.model_config)


def _mapping(hw_path, model_path, mode="LLM", mutate=None, **kwargs):
    hw = _hw(hw_path, mutate)
    model = config.parse_config(str(model_path), mode)
    return fws_mapping.build_mapping(hw, model, **kwargs)


def _bank_depth(depth):
    """Mutator: declare a card whose allocation granularity is `depth` slots."""

    def mutate(raw):
        cards = raw["cim"].setdefault("cards", {})
        if cards:
            name = next(iter(cards))
            cards[name]["bank_depth"] = depth
        else:
            cards["ctt"] = {"kind": "analog_macro", "device": "ctt", "bank_depth": depth}

    return mutate


OWNER_FFN1 = TileOwner("llama", 0, "ffn1", -1, 0)
OWNER_FFN2 = TileOwner("llama", 0, "ffn2", -1, 0)


@pytest.fixture(scope="module")
def llama_device():
    return _device(FWS_LLAMA7B, LLAMA2_7B)


# ---------------------------------------------------------------------------
# 1. The canonical walk
# ---------------------------------------------------------------------------


def test_dense_blocks_walk_k_inner_and_clip_at_the_remainder(llama_device):
    # HAND-COMPUTED. K x N = 8192 x 8192 on rows = 4096, whole-macro bank width
    # 4096: 2 row blocks x 2 column blocks. K-INNER means the K blocks of ONE
    # output block come out consecutively, so the order is
    #   (k 0..4096, n 0..4096), (k 4096..8192, n 0..4096),
    #   (k 0..4096, n 4096..8192), (k 4096..8192, n 4096..8192).
    blocks = llama_device.dense_blocks(8192, 8192)
    assert blocks == (
        (0, 4096, 0, 4096),
        (4096, 8192, 0, 4096),
        (0, 4096, 4096, 8192),
        (4096, 8192, 4096, 8192),
    )
    # The far edge CLIPS: 22016 columns into 4096-wide banks is 5 full blocks
    # and a 22016 - 5*4096 = 1536-wide remainder, which is the dimension
    # mismatch Invariant W reports rather than rounds.
    tail = llama_device.dense_blocks(4096, 22016)[-1]
    assert tail == (0, 4096, 20480, 22016)
    assert tail[3] - tail[2] == 1536


def test_a_non_canonical_walk_is_refused_by_name(llama_device):
    # D26: the register BINDS. The refusal is the entire implementation.
    with pytest.raises(DeadFoldError) as excinfo:
        llama_device.dense_blocks(4096, 4096, walk="n_inner")
    assert "non_canonical_walk" in str(excinfo.value)
    with pytest.raises(DeadFoldError) as excinfo:
        llama_device.dense_pack([PackRequest(OWNER_FFN1, 4096, 4096)], walk="slice_major")
    assert "K-INNER" in str(excinfo.value)


def test_every_dead_fold_refuses_and_nothing_else_may_be_declared_dead():
    assert set(DEAD_FOLDS) == {
        "non_canonical_walk",
        "slicing_x_folding",
        "replication",
        "prefill_folding",
    }
    for name in DEAD_FOLDS:
        with pytest.raises(DeadFoldError) as excinfo:
            cim_timing.refuse_dead_fold(name, context="test")
        assert name in str(excinfo.value) and "D2" in str(excinfo.value)
    # A fold is dead on the register or it is not dead: refusing something that
    # is not listed is itself an error, so the register cannot be bypassed.
    with pytest.raises(ValueError) as excinfo:
        cim_timing.refuse_dead_fold("multi_tenant_folding")
    assert "not on the DOA register" in str(excinfo.value)
    assert not isinstance(excinfo.value, DeadFoldError)


def test_slicing_x_folding_is_refused_at_the_packer(llama_device):
    # A card that slices places its slice groups through enumerate_tiles (P2.4).
    sliced = _device(
        FWS_LLAMA7B,
        LLAMA2_7B,
        mutate=lambda raw: raw["cim"].__setitem__(
            "cards",
            {"ctt": {"kind": "analog_macro", "device": "ctt",
                     "bits_per_cell": 2, "weight_bits": 8, "bank_depth": 1}},
        ),
    )
    assert sliced.n_slices == 4
    with pytest.raises(DeadFoldError) as excinfo:
        sliced.dense_pack([PackRequest(OWNER_FFN1, 4096, 4096)])
    assert "slicing_x_folding" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 2. Invariant W: the packing and its waste accounting
# ---------------------------------------------------------------------------


def test_whole_macro_bank_packing_is_hand_computable(llama_device):
    # HAND-COMPUTED, whole-macro banks (the shipped bank_depth), bank width 4096:
    #   ffn1 4096 x 22016 -> 1 row block x ceil(22016/4096) = 6 -> 6 banks
    #   ffn2 11008 x 4096 -> ceil(11008/4096) = 3 row blocks x 1 -> 3 banks
    #   banks = 9, and one bank IS one macro here, so macros = 9.
    #   real cells    = 4096*22016 + 11008*4096 = 90,177,536 + 45,088,768
    #                 = 135,266,304
    #   committed     = 9 * 16,777,216 = 150,994,944
    #   remainder     = ffn1's N tail (6*4096 - 22016 = 2560 cols x 4096 rows
    #                   = 10,485,760) + ffn2's K tail (3*4096 - 11008 = 1280
    #                   rows x 4096 cols = 5,242,880) = 15,728,640
    #   tail          = 0 (the stream ends exactly on a macro boundary)
    #   waste         = 15,728,640 / 150,994,944 = 10.4166...%
    packing = llama_device.dense_pack(
        [PackRequest(OWNER_FFN1, 4096, 22016), PackRequest(OWNER_FFN2, 11008, 4096)]
    )
    assert packing.banks_used == 9
    assert packing.banks_per_macro == 1
    assert packing.macro_count == 9
    assert packing.real_cells == 135_266_304
    assert packing.committed_cells == 150_994_944
    assert packing.remainder_cells == 15_728_640
    assert packing.tail_cells == 0
    assert packing.waste_pct == pytest.approx(100 * 15_728_640 / 150_994_944)
    assert packing.waste_pct == pytest.approx(10.4166666, abs=1e-6)


def test_single_slot_banks_pack_four_tensor_blocks_to_a_macro(llama_device):
    # HAND-COMPUTED, bank = 1 mux slot, bank width 1024:
    #   ffn1 4096 x 22016 -> 1 x ceil(22016/1024) = 22 banks
    #   ffn2 11008 x 4096 -> 3 x 4 = 12 banks;  banks = 34
    #   banks_per_macro = 4  ->  macros = ceil(34/4) = 9
    #   block cells = 34 * 4096 * 1024 = 142,606,336
    #   remainder   = 142,606,336 - 135,266,304 = 7,340,032
    #   tail        = 9*16,777,216 - 142,606,336 = 8,388,608 (the last macro
    #                 holds 2 of its 4 banks)
    # The TOTAL waste is unchanged at 15,728,640 cells; what changes is WHERE
    # it is, and the itemization is the whole point of reporting two terms.
    packing = llama_device.dense_pack(
        [PackRequest(OWNER_FFN1, 4096, 22016), PackRequest(OWNER_FFN2, 11008, 4096)],
        column_sets_per_tile=1,
    )
    assert (packing.banks_used, packing.banks_per_macro, packing.macro_count) == (34, 4, 9)
    assert packing.remainder_cells == 7_340_032
    assert packing.tail_cells == 8_388_608
    assert packing.waste_cells == 15_728_640
    # The stream never idles a bank it has not reached: every macro but the last
    # is full, and the last one is exactly the remainder of the stream.
    fills = {fill.macro_id: fill.banks_used for fill in packing.macro_fills}
    assert [fills[m] for m in range(8)] == [4] * 8
    assert fills[8] == 34 - 32


def test_the_waste_accounting_closes_exactly(llama_device):
    # ONE accounting (D21): real + remainder + tail == committed, exactly, with
    # no rounding anywhere. Integers, so this is an equality and not a tolerance.
    for bank in (1, 2, 4):
        packing = llama_device.dense_pack(
            [
                PackRequest(OWNER_FFN1, 4096, 22016),
                PackRequest(OWNER_FFN2, 11008, 4096),
                PackRequest(TileOwner("llama", 1, "qkv", -1, 0), 4096, 12288),
            ],
            column_sets_per_tile=bank,
        )
        assert (
            packing.real_cells + packing.remainder_cells + packing.tail_cells
            == packing.committed_cells
        )
        assert packing.waste_cells == packing.remainder_cells + packing.tail_cells


def test_the_global_cell_floor_is_a_floor_nothing_beats(llama_device):
    # D27: analog_macros ~= ceil(total_model_cells / cells_per_macro). The floor
    # is a LOWER BOUND — a packing reaches it or pays a stated delta, and it can
    # never come in under it.
    for bank in (1, 2, 4):
        packing = llama_device.dense_pack(
            [PackRequest(OWNER_FFN1, 4096, 22016), PackRequest(OWNER_FFN2, 11008, 4096)],
            column_sets_per_tile=bank,
        )
        assert packing.cell_floor_macros == math.ceil(
            packing.real_cells / packing.cells_per_macro
        )
        assert packing.floor_delta_macros >= 0
        assert packing.macro_count >= packing.cell_floor_macros
    # HAND-COMPUTED: 135,266,304 real cells / 16,777,216 cells per macro = 8.06,
    # so the floor is 9 macros and this packing sits exactly on it.
    packing = llama_device.dense_pack(
        [PackRequest(OWNER_FFN1, 4096, 22016), PackRequest(OWNER_FFN2, 11008, 4096)]
    )
    assert packing.cell_floor_macros == 9
    assert packing.floor_delta_macros == 0


def test_the_stream_is_contiguous_across_tensors(llama_device):
    # Invariant W's mechanism: a tensor that ends mid-macro is followed by the
    # NEXT tensor's first block in the very next bank. Under bank = 1 slot, ffn1
    # ends at bank 22 (macro 5, slot 2), so ffn2's first block is macro 5 slot 3
    # — a macro holding two owners, which is the whole point.
    packing = llama_device.dense_pack(
        [PackRequest(OWNER_FFN1, 4096, 22016), PackRequest(OWNER_FFN2, 11008, 4096)],
        column_sets_per_tile=1,
    )
    ffn1_last = [t for t in packing.tiles if t.owner == OWNER_FFN1][-1]
    ffn2_first = [t for t in packing.tiles if t.owner == OWNER_FFN2][0]
    assert (ffn1_last.site.macro_id, ffn1_last.site.column_sets) == (5, (1,))
    assert (ffn2_first.site.macro_id, ffn2_first.site.column_sets) == (5, (2,))
    shared = [f for f in packing.macro_fills if f.macro_id == 5][0]
    assert set(shared.owners) == {OWNER_FFN1, OWNER_FFN2}


def test_an_empty_dimension_is_refused_rather_than_packed_as_nothing(llama_device):
    with pytest.raises(ValueError) as excinfo:
        llama_device.dense_blocks(4096, 0)
    assert "holds no weights" in str(excinfo.value)


def test_a_bank_finer_than_the_card_admits_is_refused(llama_device):
    with pytest.raises(ValueError) as excinfo:
        llama_device.dense_pack(
            [PackRequest(OWNER_FFN1, 4096, 4096)], column_sets_per_tile=3
        )
    assert "smallest allocatable unit" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 3. The K stacks and the accumulator laws
# ---------------------------------------------------------------------------


def test_k_inner_puts_a_k_stack_in_one_macro_and_says_when_it_cannot(llama_device):
    # HAND-COMPUTED (bank = 1 slot, 4 banks per macro). ffn1 fills banks 0..21;
    # ffn2's 4 output blocks each stack 3 K blocks, at stream indices
    #   n=0:    22,23,24 -> macros 5,5,6   SPREAD (it straddles the boundary)
    #   n=1024: 25,26,27 -> macros 6,6,6   LOCAL
    #   n=2048: 28,29,30 -> macros 7,7,7   LOCAL
    #   n=3072: 31,32,33 -> macros 7,8,8   SPREAD
    # This is the K-inner walk earning its keep: the stack is local WHENEVER the
    # macro boundary lets it be, and the two that straddle are named, not hidden.
    packing = llama_device.dense_pack(
        [PackRequest(OWNER_FFN1, 4096, 22016), PackRequest(OWNER_FFN2, 11008, 4096)],
        column_sets_per_tile=1,
    )
    stacks = [s for s in packing.k_stacks if s.owner == OWNER_FFN2]
    assert [s.depth for s in stacks] == [3, 3, 3, 3]
    assert [s.macro_ids for s in stacks] == [(5, 5, 6), (6, 6, 6), (7, 7, 7), (7, 8, 8)]
    assert [s.local for s in stacks] == [False, True, True, False]
    # Every ffn1 block is its own output block: depth 1, nothing to accumulate.
    assert {s.depth for s in packing.k_stacks if s.owner == OWNER_FFN1} == {1}


def test_the_in_macro_accumulator_law_is_hand_computable(llama_device):
    # HAND-COMPUTED on the llama7b card at m = 1 token, bank width 1024:
    #   pool result rate = cols_adc * f_analog / slice_cycles
    #                    = 1024 * 1e8 / 2 = 5.12e10 results/s
    #   pool lanes       = ceil(5.12e10 / 0.95e9) = ceil(53.89) = 54  (P2.5)
    #   results          = m * width = 1 * 1024 = 1024
    #   drain cycles     = ceil(1024 / 54) = 19   (54*18 = 972 < 1024 <= 1026)
    #   drain            = 19 / 0.95e9 = 2.0e-8 s
    #   ADC pass         = m * slice_cycles / f_analog = 2 / 1e8 = 2.0e-8 s
    #   stall            = max(0, drain - pass) = 0  -- and it is 0 BY THE POOL
    #                      SIZING LAW, which sizes lanes to consume exactly this
    #                      result rate. The derived zero is the point.
    #   depth 3 -> 2 adds per output: add 1 hides under pass 3, add 2 is the
    #              tail. So partial_adds = 2048, hidden_adds = 1024,
    #              time = drain = 2.0e-8 s.
    packing = llama_device.dense_pack(
        [PackRequest(OWNER_FFN1, 4096, 22016), PackRequest(OWNER_FFN2, 11008, 4096)],
        column_sets_per_tile=1,
    )
    pool = llama_device.digital_pool_sizing()
    assert pool.lanes == 54 and pool.pool_clock_hz == pytest.approx(0.95e9)
    local = [s for s in packing.k_stacks if s.depth > 1 and s.local][0]
    cost = llama_device.price_accumulation(local, 1.0, act_bytes=2.0, pool=pool)
    assert cost.width == 1024 and cost.depth == 3
    assert cost.drain_cycles == 19
    assert cost.stall_s == 0.0
    assert cost.time_s == pytest.approx(19 / 0.95e9)
    assert cost.time_s == pytest.approx(llama_device.analog_op_time(1.0, 1), rel=1e-12)
    assert cost.partial_adds == 2048.0
    assert cost.hidden_adds == 1024.0
    assert cost.priced_here is True
    assert cost.transport_bytes == 0.0
    # Energy is an honest zero: no card declares pool_energy_per_add_pj, so the
    # term reports zero and DISCLOSES that it did (never an invented figure).
    assert cost.energy_pj == 0.0
    assert any("pool_energy_per_add_pj" in note for note in cost.disclosures)


def test_a_depth_one_stack_costs_nothing_at_all(llama_device):
    packing = llama_device.dense_pack([PackRequest(OWNER_FFN1, 4096, 22016)])
    flat = packing.k_stacks[0]
    cost = llama_device.price_accumulation(flat, 8.0)
    assert flat.depth == 1
    assert (cost.time_s, cost.energy_pj, cost.partial_adds) == (0.0, 0.0, 0.0)
    assert "nothing accumulates" in cost.law


def test_a_spread_stack_is_charged_by_the_existing_row_block_law_only(llama_device):
    # D21, one accounting per metric. P3 already emits the partial transfer and
    # the pool rowsum for a stack whose K blocks sit on different macros, and P4
    # prices them with price_reduction. So this law charges it ZERO and reports
    # the transport bytes for reconciliation.
    # HAND-COMPUTED: the n=0 stack spans macros (5,5,6) = 2 distinct macros, so
    # 1 partial word per output crosses, and at m=1, width 1024, 2 B/activation
    # the transport is 1 * 1024 * 1 * 2 = 2048 bytes.
    packing = llama_device.dense_pack(
        [PackRequest(OWNER_FFN1, 4096, 22016), PackRequest(OWNER_FFN2, 11008, 4096)],
        column_sets_per_tile=1,
    )
    spread = [s for s in packing.k_stacks if s.depth > 1 and not s.local][0]
    cost = llama_device.price_accumulation(spread, 1.0, act_bytes=2.0)
    assert cost.priced_here is False
    assert cost.transport_partials == 1
    assert cost.transport_bytes == 2048.0
    assert (cost.time_s, cost.energy_pj, cost.partial_adds) == (0.0, 0.0, 0.0)
    assert "row_block_partial_sum" in cost.law
    assert any("two accountings" in note for note in cost.disclosures)


def test_the_aggregate_charges_the_worst_macro_and_one_accumulator_per_macro(
    llama_device,
):
    # HAND-COMPUTED. 8192 x 2048 at bank width 1024: 2 row blocks x 2 column
    # blocks = 4 banks = exactly one macro. Both output blocks are LOCAL stacks
    # of depth 2 on macro 0, so:
    #   * accumulators = 1, not 2 — under the K-inner walk a stack's banks are
    #     consecutive, so its partial is consumed before the next stack starts
    #     and exactly one accumulator is live per macro.
    #   * time = the two tails SUM (one macro walks them in sequence)
    #     = 2 * 19 / 0.95e9 = 4.0e-8 s.
    #   * adders per accumulator = min(bank width 1024, pool lanes 54) = 54.
    owner = TileOwner("llama", 0, "square", -1, 0)
    packing = llama_device.dense_pack(
        [PackRequest(owner, 8192, 2048)], column_sets_per_tile=1
    )
    assert packing.macro_count == 1 and packing.banks_used == 4
    assert [s.depth for s in packing.k_stacks] == [2, 2]
    assert all(s.local for s in packing.k_stacks)
    aggregate = llama_device.price_packing_accumulation(packing, 1.0, act_bytes=2.0)
    assert aggregate.accumulators == 1
    assert aggregate.adders_per_accumulator == 54
    assert aggregate.local_stacks == 2 and aggregate.spread_stacks == 0
    assert aggregate.time_s == pytest.approx(2 * 19 / 0.95e9)
    # Area is an honest zero until a card declares the adder cost, and the
    # accumulator's holding REGISTER is a named gap rather than a guess.
    assert aggregate.area_mm2 == 0.0
    assert any("pool_area_mm2_per_adder" in note for note in aggregate.disclosures)
    assert any("holding REGISTER" in note for note in aggregate.disclosures)


def test_a_declared_adder_cost_reaches_the_accumulator_area_and_energy():
    # The knobs are inert by default and LOAD-BEARING when declared: 1e-4 mm2
    # per adder over 54 adders on 1 accumulator = 5.4e-3 mm2, and 0.5 pJ per add
    # over 2 * (1 * 1024) = 2048 adds = 1024 pJ.
    device = _device(
        FWS_LLAMA7B,
        LLAMA2_7B,
        mutate=lambda raw: raw["cim"].__setitem__(
            "cards",
            {
                "ctt": {
                    "kind": "analog_macro",
                    "device": "ctt",
                    "bank_depth": 1,
                    "pool_area_mm2_per_adder": 1e-4,
                    "pool_energy_per_add_pj": 0.5,
                }
            },
        ),
    )
    owner = TileOwner("llama", 0, "square", -1, 0)
    packing = device.dense_pack([PackRequest(owner, 8192, 1024)])
    aggregate = device.price_packing_accumulation(packing, 1.0)
    assert aggregate.accumulators == 1
    assert aggregate.area_mm2 == pytest.approx(54 * 1e-4)
    assert aggregate.energy_pj == pytest.approx(1024 * 1 * 0.5)
    assert not any("pool_area_mm2_per_adder" in n for n in aggregate.disclosures)


# ---------------------------------------------------------------------------
# 4. The degenerate identity (P7.2 item 3)
# ---------------------------------------------------------------------------


DEGENERATE_PAIRS = (
    (FWS_LLAMA7B, LLAMA2_7B, "LLM"),
    (FWS_MOE, MOE_SMALL, "LLM"),
    (FWS_GRANITE, GRANITE, "LLM"),
    (FWS_QWEN, QWEN, "LLM"),
)


@pytest.mark.parametrize("hw_path,model_path,mode", DEGENERATE_PAIRS)
def test_dense_packing_reproduces_the_shipped_placement_tile_for_tile(
    hw_path, model_path, mode
):
    # THE DEGENERATE IDENTITY. Every shipped card admits WHOLE-MACRO allocation
    # (bank_depth absent -> adc_mux), so a dense bank IS a macro: each tensor
    # still starts on a fresh macro and the stream coincides with the per-tensor
    # placement. Every shipped tensor is also single-row-block or
    # single-column-block, so the K-inner walk and the enumerator's walk agree
    # block for block and the identity is TILE-FOR-TILE, not merely numeric.
    dedicated = _mapping(hw_path, model_path, mode)
    dense = _mapping(hw_path, model_path, mode, packing=fws_mapping.PACKING_DENSE)
    assert dense.tiles == dedicated.tiles
    assert dense.packing_summary()["macros_saved"] == 0
    assert dense.packing_summary()["macros"] == dense.packing_summary()["dedicated_macros"]
    # ... and the identity is stated in the artifact, not left to be noticed.
    packing_relaxations = [
        r for r in dense.relaxations() if r.constraint == "packing"
    ]
    assert len(packing_relaxations) == 1
    assert "degenerate identity holds" in packing_relaxations[0].reason


@pytest.mark.parametrize(
    "hw_path,model_path", [(FWS_LLAMA7B, LLAMA2_7B), (FWS_MOE, MOE_SMALL)]
)
def test_the_degenerate_identity_survives_the_whole_priced_path(hw_path, model_path):
    # Tile-for-tile identity is worth nothing if a number downstream moves, so
    # this drives the SAME path a run drives: place, build the DAG, price it.
    numbers = []
    for packing in (fws_mapping.PACKING_DEDICATED, fws_mapping.PACKING_DENSE):
        mapping = _mapping(hw_path, model_path, packing=packing)
        evaluation = fws_eval.evaluate_fws(build_fws_program(mapping), mapping)
        numbers.append((evaluation.makespan_s, mapping.summary()["macro_slots"],
                        mapping.unowned_columns()))
    assert numbers[0] == numbers[1]  # bit-exact, not approximately


def test_the_dedicated_packing_reports_no_packing_summary(llama_device):
    # A summary of a packing that was never run would be an invented number.
    mapping = _mapping(FWS_LLAMA7B, LLAMA2_7B)
    assert mapping.packing == fws_mapping.PACKING_DEDICATED
    assert mapping.packings == {}
    assert mapping.packing_summary() == {}


# ---------------------------------------------------------------------------
# 5. Where the macro count DOES change, the delta is computed and disclosed
# ---------------------------------------------------------------------------


def test_a_finer_bank_lets_invariant_w_fill_macros_and_the_delta_is_disclosed():
    # Granite-4.0-H-Tiny, the headline model (ADJ-1), on a card that admits
    # single-mux-slot allocation. The shipped 5544-macro placement is 48.1%
    # empty weight space; dense packing at bank_depth 1 reaches 4828 macros for
    # the same weights, against a GLOBAL CELL FLOOR of 2876.
    #
    # WAVE F (D30): these four numbers each fell by exactly the LM HEAD, which
    # D30 drops entirely — ceil(100352 / 1536) = 66 macros of vocabulary
    # projection on the last chip, off both placements (5610 -> 5544 dedicated,
    # 4894 -> 4828 dense). The floor's numerator loses the head's CELLS, which
    # is 1536 x 100352 = 154,140,672 and not the 66 whole macros' worth of
    # cells the head occupies, so the floor falls by 65 (2941 -> 2876) rather
    # than by 66 — the difference is the ceiling, as the assertion below says. The SAVING is unchanged at 716 macros, because the
    # endpoint filled whole macros under either law and packing it densely
    # never bought anything.
    dense = _mapping(
        FWS_GRANITE, GRANITE, mutate=_bank_depth(1), packing=fws_mapping.PACKING_DENSE
    )
    summary = dense.packing_summary()
    assert summary["macros"] == 4828
    assert summary["dedicated_macros"] == 5544
    assert summary["macros_saved"] == 716
    assert summary["cell_floor_macros"] == 2876
    assert summary["real_cells"] + summary["remainder_cells"] + summary["tail_cells"] == (
        summary["committed_cells"]
    )
    assert summary["waste_pct"] == pytest.approx(40.4420395, abs=1e-6)
    # DISCLOSED, never silent (D21): the delta rides in a relaxation banner.
    relaxation = [r for r in dense.relaxations() if r.constraint == "packing"][0]
    assert "716 macros" in relaxation.reason
    assert "cell floor 2876" in relaxation.value
    # The shipped card's own waste is a separate, larger number — the card's
    # granularity speaking, which is exactly what the note says.
    shipped = _mapping(FWS_GRANITE, GRANITE, packing=fws_mapping.PACKING_DENSE)
    assert shipped.packing_summary()["waste_pct"] == pytest.approx(48.1338684, abs=1e-6)
    assert any("WHOLE-MACRO allocation only" in note for note in shipped.notes)


def test_the_endpoint_macros_are_gone_and_that_is_the_whole_delta():
    """WAVE F (D30): the packer carries no endpoint arrays, and the count says so.

    The check is the arithmetic of the drop, not a re-assertion of the new
    numbers: the LOCKSTEP mapping of the same model still places the lm_head,
    the D29 mapping does not, and the difference is exactly the macros that
    endpoint owned.
    """
    hw = _hw(FWS_GRANITE, _bank_depth(1))
    with_endpoints = config.parse_config(str(GRANITE), "LLM")
    # The shipped config now declares D30's drop, so the COMPARISON has to put
    # the endpoint back: this is the placement the same model had before D30.
    with_endpoints.model_config.disable_embedding_unembedding = False
    with_endpoints.model_config.global_batch_size = 4
    lockstep = fws_mapping.build_mapping(
        hw,
        with_endpoints,
        packing=fws_mapping.PACKING_DENSE,
        regime=fws_mapping.REGIME_LOCKSTEP,
    )
    filled = _mapping(
        FWS_GRANITE, GRANITE, mutate=_bank_depth(1), packing=fws_mapping.PACKING_DENSE
    )
    assert filled.regime == fws_mapping.REGIME_FILLED
    assert "lm_head" in lockstep.endpoint_blocks and not filled.endpoint_blocks
    dropped = lockstep.packing_summary()["macros"] - filled.packing_summary()["macros"]
    assert dropped == 66 == math.ceil(100352 / 1536)
    assert (
        lockstep.packing_summary()["cell_floor_macros"]
        - filled.packing_summary()["cell_floor_macros"]
    ) == 65  # the floor is a ceiling over cells, so it drops by ceil, not by 66


def test_a_dense_mapping_still_prices_end_to_end_and_holds_its_invariants():
    # The packed placement is not a report: it goes through the DAG builder and
    # the evaluator like any other, with macros holding several owners.
    dense = _mapping(
        FWS_MOE, MOE_SMALL, mutate=_bank_depth(1), packing=fws_mapping.PACKING_DENSE
    )
    dedicated = _mapping(FWS_MOE, MOE_SMALL, mutate=_bank_depth(1))
    summary = dense.packing_summary()
    assert summary["macros_saved"] == 14
    # P2's capacity invariant still holds over the packed sites, and a macro
    # really does hold more than one owner now.
    dense.device.validate_macro_capacity(dense.tiles)
    shared = [m for m in dense.macros if len(m.owners) > 1]
    assert shared
    evaluation = fws_eval.evaluate_fws(build_fws_program(dense), dense)
    assert evaluation.makespan_s > 0
    # Fewer macros hold the same weights, so the packed run cannot need MORE
    # analog silicon than the placement it replaces.
    assert sum(1 for m in dense.macros if m.tiles) <= sum(
        1 for m in dedicated.macros if m.tiles
    )


def test_dense_packing_keeps_a_layers_tensors_on_its_own_chip():
    # Locality by preference (P7.2 item 1): the packer never reorders, so a
    # chip's stream holds exactly the tensors that chip's layers own.
    dense = _mapping(
        FWS_GRANITE, GRANITE, mutate=_bank_depth(1), packing=fws_mapping.PACKING_DENSE
    )
    for chip in dense.analog_chips():
        owners = {
            tile.owner
            for macro_id in chip.macro_ids
            for tile in dense.macro(macro_id).tiles
        }
        layers = {owner.layer for owner in owners}
        # The endpoint tiles (lm_head) ride the last layer's index, which is on
        # the chip that owns that layer, so the chip's layer set contains them.
        assert layers <= set(chip.layers)


# ---------------------------------------------------------------------------
# 6. The refusals at the mapping seam
# ---------------------------------------------------------------------------


def test_a_workload_with_no_decode_cannot_be_packed_densely():
    # The ViT tiers are a PREFILL workload (decode_len 0), so build_mapping
    # infers phase 'prefill' and D25 refuses the packing without anyone having
    # to ask for a prefill fold by name. That is the register reaching the
    # workload, not just the keyword.
    with pytest.raises(DeadFoldError) as excinfo:
        _mapping(FWS_T1, VIT_HUGE_64, "VIT", packing=fws_mapping.PACKING_DENSE)
    assert "prefill_folding" in str(excinfo.value)
    # ... and the same pair packs nothing and moves nothing under the default.
    assert _mapping(FWS_T1, VIT_HUGE_64, "VIT").packing_summary() == {}


def test_prefill_dense_packing_is_refused_by_name():
    # D25: DECODE ONLY. The refusal names the register entry.
    with pytest.raises(DeadFoldError) as excinfo:
        _mapping(
            FWS_LLAMA7B, LLAMA2_7B, packing=fws_mapping.PACKING_DENSE, phase="prefill"
        )
    assert "prefill_folding" in str(excinfo.value)
    assert "DECODE ONLY (D25)" in str(excinfo.value)


def test_an_unknown_packing_law_is_refused_by_name():
    with pytest.raises(MappingError) as excinfo:
        _mapping(FWS_LLAMA7B, LLAMA2_7B, packing="greedy")
    assert "'dedicated'" in str(excinfo.value) and "'dense'" in str(excinfo.value)


def test_a_declared_allocation_and_dense_packing_cannot_both_place_a_tile():
    def mutate(raw):
        _bank_depth(1)(raw)
        raw["cim"]["allocation"] = {
            "assignments": [
                {"model": "llama", "layer": 0, "op": "qkv", "macro": 0, "column_sets": [0]}
            ]
        }

    with pytest.raises(MappingError) as excinfo:
        _mapping(FWS_LLAMA7B, LLAMA2_7B, mutate=mutate, packing=fws_mapping.PACKING_DENSE)
    assert "Two placements of one tile" in str(excinfo.value)


def test_a_packing_is_re_packed_at_the_enumerated_base_not_shifted():
    # When the config declares no slot count (arrays_per_chip: 0) the mapper
    # places on a chip-LOCAL base and only then learns the enumerated one. A
    # packing is a bank STREAM from a base, so it is re-packed rather than
    # shifted, and its fills and K stacks then name the ids the mapping ships.
    def mutate(raw):
        _bank_depth(1)(raw)
        raw["cim"]["chip"]["arrays_per_chip"] = 0

    dense = _mapping(
        FWS_MOE, MOE_SMALL, mutate=mutate, packing=fws_mapping.PACKING_DENSE
    )
    enumerated = {macro.macro_id for macro in dense.macros}
    assert all(tile.site.macro_id in enumerated for tile in dense.tiles)
    for chip_id, packing in dense.packings.items():
        chip = dense.chip(chip_id)
        assert packing.first_macro_id == chip.macro_ids[0]
        assert {fill.macro_id for fill in packing.macro_fills} <= set(chip.macro_ids)
        assert {tile.site.macro_id for tile in packing.tiles} <= set(chip.macro_ids)
    dense.device.validate_macro_capacity(dense.tiles)
    assert dense.packing_summary()["macros_saved"] == 14


def test_the_waste_metric_reaches_the_report(llama_device):
    # D27 makes waste% a FIRST-CLASS reported metric, so it gets its own line
    # in the mapping report rather than living only inside a relaxation.
    dense = _mapping(
        FWS_MOE, MOE_SMALL, mutate=_bank_depth(1), packing=fws_mapping.PACKING_DENSE
    )
    line = [l for l in dense.report().splitlines() if "packing: dense" in l]
    assert len(line) == 1
    assert "global cell floor" in line[0] and "waste" in line[0]
    assert "14 saved" in line[0]
    # The dedicated mapping prints no packing line at all.
    assert not [
        l for l in _mapping(FWS_MOE, MOE_SMALL).report().splitlines() if "packing:" in l
    ]


def test_the_canonical_walk_is_the_one_the_mapper_uses():
    dense = _mapping(FWS_MOE, MOE_SMALL, packing=fws_mapping.PACKING_DENSE)
    assert dense.packing_summary()["walk"] == CANONICAL_WALK == "k_inner"
    assert all(p.walk == CANONICAL_WALK for p in dense.packings.values())
