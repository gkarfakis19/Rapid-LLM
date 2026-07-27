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
matrix. That was the residue of a half-finished conversion: before ``173a765``
NOTHING in those helpers was per-rank (QKV was literally ``d * 3 * d``); the MoE
overhaul converted attention and the experts and left the dense MLP behind. At
``tp=8`` on a ``4h`` MLP it made a layer's weight update and its dp all-reduce
**5.667x** too large.
"""

from pathlib import Path

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


def test_endpoint_apply_grad_is_not_negligible_against_a_layer():
    """Why 12 is worth its own term rather than a rounding note: at realistic
    ``vocab/hidden`` the endpoint tables rival several transformer layers."""
    tc = _tc(tp=1)
    emb, _out = tc._embedding_softmax_params_per_rank(tc.hidden_dim, tc.vocab_size)
    qkv, out = tc._attention_param_components_per_rank(tc.hidden_dim)
    f1, f2 = tc._dense_ffn_param_components_per_rank(tc.hidden_dim, tc.intermediate_size)
    assert emb > 0.1 * (qkv + out + f1 + f2)
