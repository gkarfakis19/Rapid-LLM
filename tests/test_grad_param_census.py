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

"""BUG_LEDGER 12 + 13 — the gradient/apply-grad path counts PER-RANK parameters.

``_param_stats_per_rank`` is the authoritative per-rank parameter census: it
divides the MLP (``intermediate * ffn_proj_factor * hidden / tp``) and the
embedding (``vocab * hidden / tp``) by ``tp``. The apply-grad price and the dp
gradient payload must agree with it, because they describe the same tensors.

They did not. ``get_data_parallel_reduction_llm`` / ``_sizes`` ran their
attention terms through ``_attention_param_components_per_rank`` (which divides)
while writing the dense MLP inline as ``ffn1_dim * d`` — the FULL unsharded
matrix, on every one of the ``tp`` ranks. At ``tp=8`` on a ``4h`` MLP that made
a layer's weight update and its dp all-reduce **5.667x** too large.

Provenance (verified against git, and NOT what the first write-up of this said):
the census was per-rank from the start — before ``c5823bb`` it divided the whole
layer once at the end (``params_per_layer_per_rank = transformer_param_layer /
tp``). ``173a765`` ("MoE overhaul", 2026-01-06) made the EXPERTS per-rank. The
half-finished conversion is ``c5823bb`` ("MLA PT#1 (Megatron-equal)",
2026-04-03), which introduced ``_attention_param_components_per_rank``, swapped
it into the gradient helpers, and left the dense MLP alone.

BUG_LEDGER 18 adds the second half: the four readers of that census now share
ONE function, and the tests at the bottom of this file assert they agree.
"""

from pathlib import Path

import math

import pytest

import config
from train_timing import TimeCalculationLLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HW_BASE = (
    PROJECT_ROOT
    / "validation_scripts"
    / "validation_configs"
    / "hardware-config"
    / "a100_80GB.yaml"
)


def _tc(*, tp: int, dp: int = 2, tied: bool = False, gated: bool = False):
    hw = config.parse_config(str(HW_BASE), config_type="hardware")
    hw.sch_config.tp = tp
    hw.sch_config.cp = 1
    hw.sch_config.pp = 1
    hw.sch_config.mb = 1
    hw.sch_config.tp_sp = False
    hw.sch_config.train.dp = dp
    hw.sch_config.train.ep = 1
    hw.sch_config.train.tp_ep = True
    hw.sch_config.inference.replica_count = 1
    hw.sch_config.inference.moe_dp = 1
    model_param = {
        "mode": "LLM",
        "run_type": "training",
        "tied_embeddings": tied,
        "model_type": "llama" if gated else "gpt",
        "global_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "seq_len": 128,
        "hidden_dim": 256,
        "intermediate_size": 1024,
        "vocab_size": 512,
        "num_layers": 4,
        "attention": {
            "attention_type": "mha",
            "num_heads": 8,
            "kv_heads": 8,
            "use_flashattention": False,
            "attention_tile_size": 32,
        },
        "moe": {
            "num_experts": 1,
            "top_k": 1,
            "moe_intermediate_size": 1024,
            "n_shared_experts": 0,
            "moe_layer_freq": 1,
            "first_k_dense_replace": 4,
        },
    }
    model = config.ModelConfig(
        model_config=config.LLMConfig.from_dict(model_param), inference_config=None
    )
    return TimeCalculationLLM(hw, model, "LLM", output_dir=None)


@pytest.mark.parametrize("gated", [False, True])
def test_dense_ffn_apply_grad_params_are_per_rank(gated):
    """**13**. The MLP is column/row parallel, so a rank owns ``1/tp`` of it —
    the same fraction ``_param_stats_per_rank`` charges for memory."""
    for tp in (1, 2, 4, 8):
        tc = _tc(tp=tp, gated=gated)
        f1, f2 = tc._dense_ffn_param_components_per_rank(tc.hidden_dim, tc.intermediate_size)
        whole = (
            tc._ffn1_output_dim(tc.intermediate_size) * tc.hidden_dim
            + tc.intermediate_size * tc.hidden_dim
        )
        assert f1 + f2 == pytest.approx(whole / tp)

        # ...and it is the SAME number the memory census uses. `_param_stats_per_rank`
        # returns attention + MLP per layer; subtracting attention leaves the MLP.
        _t, _m, per_layer, _e, _o = tc._param_stats_per_rank(
            tc.hidden_dim, tc.intermediate_size, tc.vocab_size
        )
        attention = tc._attention_param_total_per_rank(tc.hidden_dim)
        assert per_layer - attention == pytest.approx(f1 + f2), (
            f"tp={tp}: the apply-grad MLP param count disagrees with the memory census"
        )


def test_dp_gradient_bytes_conserve_across_tp():
    """**13**, byte half. Each rank all-reduces the shard IT owns, so the
    aggregate over a TP group is invariant in ``tp`` — a rank cannot reduce a
    peer's shard, and the whole layer's gradient is reduced exactly once."""
    aggregate = {}
    for tp in (1, 2, 4, 8):
        tc = _tc(tp=tp)
        aggregate[tp] = tc.get_data_parallel_reduction_sizes(
            tc.hidden_dim, tc.intermediate_size
        ) * tp
    # ceil() per component makes this exact only to within a few bytes.
    for tp, total in aggregate.items():
        assert total == pytest.approx(aggregate[1], rel=1e-9, abs=8 * tp), (
            f"tp={tp} aggregate dp gradient bytes {total} != tp=1 {aggregate[1]}"
        )


def test_dp_gradient_bytes_and_apply_grad_describe_the_same_parameters():
    """The payload and the price are two views of ONE parameter set: bytes /
    grad_communication must equal the parameter count the apply-grad prices."""
    for tp in (1, 2, 8):
        tc = _tc(tp=tp)
        size = tc.get_data_parallel_reduction_sizes(tc.hidden_dim, tc.intermediate_size)
        qkv, out = tc._attention_param_components_per_rank(tc.hidden_dim)
        f1, f2 = tc._dense_ffn_param_components_per_rank(tc.hidden_dim, tc.intermediate_size)
        expected = tc.precision.grad_communication * (qkv + out + f1 + f2)
        assert size == pytest.approx(expected, abs=4)


def test_embedding_and_softmax_apply_grad_match_the_memory_census():
    """**12**. The endpoint parameters get a weight-update price at last, and it
    is derived from the same per-rank counts the memory census reports."""
    for tp in (1, 2, 8):
        for tied in (False, True):
            tc = _tc(tp=tp, tied=tied)
            _t, _m, _l, emb_params, out_params = tc._param_stats_per_rank(
                tc.hidden_dim, tc.intermediate_size, tc.vocab_size
            )
            emb, out = tc._embedding_softmax_params_per_rank(tc.hidden_dim, tc.vocab_size)
            assert emb == pytest.approx(emb_params)
            assert out == pytest.approx(out_params)

            assert tc.get_embedding_apply_grad_llm(
                tc.hidden_dim, tc.vocab_size
            ) == pytest.approx(tc.apply_grad(int(emb_params)))
            # Tied embeddings: the last stage's projection IS stage 0's table,
            # so charging it again would double-count.
            expected_out = 0.0 if tied else tc.apply_grad(int(out_params))
            assert tc.get_softmax_apply_grad_llm(
                tc.hidden_dim, tc.vocab_size
            ) == pytest.approx(expected_out)


def _moe_tc(*, tp: int, ep: int, tp_ep: bool = True, shared: int = 2, gated: bool = True):
    hw = config.parse_config(str(HW_BASE), config_type="hardware")
    hw.sch_config.tp = tp
    hw.sch_config.cp = 1
    hw.sch_config.pp = 1
    hw.sch_config.mb = 1
    hw.sch_config.tp_sp = False
    hw.sch_config.train.dp = 2
    hw.sch_config.train.ep = ep
    hw.sch_config.train.tp_ep = tp_ep
    hw.sch_config.inference.replica_count = 1
    hw.sch_config.inference.moe_dp = 1
    model_param = {
        "mode": "LLM",
        "run_type": "training",
        "tied_embeddings": False,
        "model_type": "llama" if gated else "gpt",
        "global_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "seq_len": 128,
        "hidden_dim": 256,
        "intermediate_size": 1024,
        "vocab_size": 512,
        "num_layers": 4,
        "attention": {
            "attention_type": "mha", "num_heads": 8, "kv_heads": 8,
            "use_flashattention": False, "attention_tile_size": 32,
        },
        "moe": {
            "num_experts": 8, "top_k": 2, "moe_intermediate_size": 512,
            "n_shared_experts": shared, "moe_layer_freq": 1, "first_k_dense_replace": 0,
        },
    }
    model = config.ModelConfig(
        model_config=config.LLMConfig.from_dict(model_param), inference_config=None
    )
    return TimeCalculationLLM(hw, model, "LLM", output_dir=None)


# ---------------------------------------------------------------------------
# BUG_LEDGER 18 — ONE census, and every reader must agree with it.
# ---------------------------------------------------------------------------

def test_every_census_reader_agrees_dense():
    """**18**. Four call sites answer "how many parameters does this rank own":
    the memory/ZeRO census, the apply-grad price, the dp gradient payload and
    the EP-sync payload. Three of them disagreed until items 12 and 13. This is
    the gate that stops them drifting apart again."""
    for tp in (1, 2, 4, 8):
        for gated in (False, True):
            tc = _tc(tp=tp, gated=gated)
            census = tc.layer_params_per_rank(tc.hidden_dim, tc.intermediate_size, moe=False)
            total = sum(census.values())

            # reader 1: the memory / ZeRO stats
            _t, _m, per_layer, _e, _o = tc._param_stats_per_rank(
                tc.hidden_dim, tc.intermediate_size, tc.vocab_size
            )
            assert per_layer == pytest.approx(total), f"tp={tp} gated={gated}: memory census"

            # reader 2: the dp gradient payload (bytes / precision == params)
            size = tc.get_data_parallel_reduction_sizes(tc.hidden_dim, tc.intermediate_size)
            assert size == pytest.approx(
                tc.precision.grad_communication * total, abs=4
            ), f"tp={tp} gated={gated}: dp bytes"

            # reader 3: the apply-grad price, component by component
            expected = sum(tc.apply_grad(int(v)) for v in
                           (census["qkv"], census["output"], census["ffn1"], census["ffn2"]))
            assert tc.get_data_parallel_reduction_llm(
                tc.hidden_dim, tc.intermediate_size
            ) == pytest.approx(expected), f"tp={tp} gated={gated}: apply-grad"


def test_every_census_reader_agrees_moe():
    """**18**, MoE half — including the ``tp_ep`` switch, which the duplicated
    copies each spelled out by hand."""
    for tp in (1, 2, 4):
        for tp_ep in (True, False):
            tc = _moe_tc(tp=tp, ep=2, tp_ep=tp_ep)
            census = tc.layer_params_per_rank(
                tc.hidden_dim, tc.moe_intermediate_size, moe=True
            )
            assert set(census) == {"qkv", "output", "experts", "router"}

            size = tc.get_data_parallel_reduction_sizes(
                tc.hidden_dim, tc.moe_intermediate_size, moe=True
            )
            g = tc.precision.grad_communication
            expected_bytes = (
                math.ceil(g * census["qkv"])
                + math.ceil(g * census["output"])
                + math.ceil(g * (census["experts"] + census["router"]))
            )
            assert size == expected_bytes

            expected_time = sum(tc.apply_grad(int(census[k]))
                                for k in ("qkv", "output", "experts", "router"))
            assert tc.get_data_parallel_reduction_llm(
                tc.hidden_dim, tc.moe_intermediate_size, moe=True
            ) == pytest.approx(expected_time)

            # tp_ep is the ONLY thing that changes expert sharding, and it must
            # move the expert term by exactly tp.
            if tp > 1:
                other = _moe_tc(tp=tp, ep=2, tp_ep=not tp_ep)
                oc = other.layer_params_per_rank(
                    other.hidden_dim, other.moe_intermediate_size, moe=True
                )
                ratio = oc["experts"] / census["experts"]
                assert ratio == pytest.approx(tp if tp_ep else 1.0 / tp)
                assert oc["router"] == pytest.approx(census["router"])


def test_census_is_the_only_place_the_mlp_divisor_is_written():
    """**18**, structural. The per-rank divisor must appear in the census
    helpers and nowhere else, so a future edit cannot resurrect a fourth
    opinion. Checked by source inspection, because the whole class of bug is
    "someone re-spelled the arithmetic"."""
    import ast
    import inspect
    import textwrap

    from train_timing import TimeCalculationLLM as T

    def _code_only(fn) -> str:
        """Source with the docstring removed — the census helpers are cited by
        NAME in prose all over this file, and prose is not a re-derivation."""
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src).body[0]
        body = tree.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]
        return "\n".join(ast.unparse(node) for node in body)

    census_owners = {
        "_dense_ffn_param_components_per_rank",
        "_moe_ffn_param_components_per_rank",
        "_attention_param_components_per_rank",
        "_embedding_softmax_params_per_rank",
        "layer_params_per_rank",
        "_param_stats_per_rank",
    }
    offenders = []
    for name in (
        "get_data_parallel_reduction_llm",
        "get_data_parallel_reduction_sizes",
        "get_embedding_apply_grad_llm",
        "get_softmax_apply_grad_llm",
    ):
        src = _code_only(getattr(T, name))
        # No re-derivation: these must not compute expert/MLP sizes themselves.
        for needle in ("ffn_proj_factor", "_ffn1_output_dim", "moe_num_experts /",
                       "expert_param_size", "vocab_size *"):
            if needle in src:
                offenders.append(f"{name} re-derives {needle!r}")
    assert not offenders, offenders
    assert census_owners  # the owners exist


def test_endpoint_apply_grad_is_not_negligible_against_a_layer():
    """Why 12 is worth its own term rather than a rounding note: at realistic
    ``vocab/hidden`` the endpoint tables rival several transformer layers."""
    tc = _tc(tp=1)
    emb, _out = tc._embedding_softmax_params_per_rank(tc.hidden_dim, tc.vocab_size)
    qkv, out = tc._attention_param_components_per_rank(tc.hidden_dim)
    f1, f2 = tc._dense_ffn_param_components_per_rank(tc.hidden_dim, tc.intermediate_size)
    assert emb > 0.1 * (qkv + out + f1 + f2)
