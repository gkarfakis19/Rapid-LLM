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

"""Which degree shards the residual stream — RESIDENCY vs TRANSPORT.

Two functions answer a question that *sounds* identical:

* ``train_timing._sequence_parallel_degree()`` — how much of the residual is
  RESIDENT on this device (drives the memory census);
* ``Placement.activation_shard_size()`` — how much of it CROSSES a pipeline
  boundary (drives the ``cross_layer`` payload).

They disagree in 6 of 12 cells, and until this file existed nobody could say
which cells were intended. Every cell is now justified here, so a future reader
gets an answer instead of two numbers.

There is exactly ONE cell that is a gap rather than a position, and it is
marked. See ``docs/rewrite/restructure/BUG_LEDGER.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))

from program.placement import Granularity, Placement  # noqa: E402
from program.schedule.gpipe import GPipeSchedule  # noqa: E402
from test_policies import Cfg, make_workload  # noqa: E402


def _transport_degree(tp: int, cp: int) -> int:
    cfg = Cfg(dp=1, tp=tp, cp=cp, ep=1, pp=2, mb=2)
    fw = make_workload(cfg).freeze()
    placement = Placement(fw, Granularity.FLAT, GPipeSchedule().layer_assignment(fw))
    return placement.activation_shard_size()


def _residency_degree(tp: int, cp: int, tp_sp: bool) -> int:
    """Mirror of ``train_timing._sequence_parallel_degree`` (kept in lockstep by
    :func:`test_residency_mirror_matches_train_timing`)."""
    degree = 1
    if tp_sp and tp > 1:
        degree *= tp
    if cp > 1:
        degree *= cp
    return degree


#: (tp, cp, tp_sp) -> (residency, transport, why they differ)
CONTRACT = {
    # --- agree: nothing is sharded, or only cp is -------------------------
    (1, 1, False): (1, 1, "no sharding on either leg"),
    (1, 1, True): (1, 1, "sp with tp=1 shards nothing"),
    (1, 2, False): (2, 2, "cp shards the sequence for residency AND transport"),
    (1, 2, True): (2, 2, "same; sp adds nothing at tp=1"),
    (2, 1, True): (2, 2, "sp shards the residual by tp on both legs"),
    (4, 1, True): (4, 4, "same at tp=4"),
    # --- differ BY DESIGN: plain TP replicates in memory, splits on the wire
    (2, 1, False): (1, 2, "PLAIN TP (10c): resident whole, sent split by "
                          "--scatter-gather-tensors-in-pipeline"),
    (4, 1, False): (1, 4, "PLAIN TP (10c), tp=4"),
    (2, 2, False): (2, 4, "PLAIN TP + cp: resident /cp, sent /(tp*cp)"),
    (4, 2, False): (2, 8, "PLAIN TP + cp, tp=4"),
    # --- WAS the gap, FIXED 2026-08-02: sp + cp shard on BOTH axes ---------
    (2, 2, True): (4, 4, "sp AND cp both shard the sequence, so residency is "
                         "tp*cp — the two legs AGREE here now. Before the fix "
                         "_sequence_parallel_degree's elif chain could name one "
                         "axis and returned cp alone."),
    (4, 2, True): (8, 8, "as above at tp=4."),
}

#: Empty. It used to hold the two ``sp and cp`` cells, where residency was
#: UNDER-sharded because the elif chain could not name two axes at once.
#: :meth:`train_timing.TimeCalculationLLM._sequence_parallel_degree` is a
#: PRODUCT now, so every remaining divergence is the deliberate plain-TP one.
KNOWN_GAP_CELLS: set = set()


@pytest.mark.parametrize("cell", sorted(CONTRACT))
def test_activation_sharding_contract(cell):
    tp, cp, tp_sp = cell
    expected_residency, expected_transport, why = CONTRACT[cell]
    assert _residency_degree(tp, cp, tp_sp) == expected_residency, why
    assert _transport_degree(tp, cp) == expected_transport, why


def test_the_only_unjustified_divergence_is_the_named_gap():
    """Every residency/transport disagreement is either the documented plain-TP
    position or one of the two named gap cells. A NEW disagreement fails here."""
    unexplained = []
    for (tp, cp, tp_sp), (res, trans, _why) in CONTRACT.items():
        if res == trans:
            continue
        plain_tp_position = (not tp_sp) and tp > 1
        if plain_tp_position or (tp, cp, tp_sp) in KNOWN_GAP_CELLS:
            continue
        unexplained.append((tp, cp, tp_sp, res, trans))
    assert not unexplained, f"unexplained residency/transport divergence: {unexplained}"


def test_residency_mirror_matches_train_timing():
    """The mirror above must not drift from the real function."""
    import train_timing

    src = train_timing.TimeCalculationLLM._sequence_parallel_degree
    for tp in (1, 2, 4):
        for cp in (1, 2):
            for tp_sp in (False, True):
                stub = type("S", (), {"tp": tp, "cp": cp, "tp_sp": tp_sp})()
                assert src(stub) == _residency_degree(tp, cp, tp_sp), (tp, cp, tp_sp)


def test_transport_leg_is_the_axis_registry():
    """The transport leg is not hand-written arithmetic — it is the set of axes
    declaring ``shards_activations`` (``program.axes``)."""
    from program.axes import activation_sharding_axes

    assert activation_sharding_axes() == ("tp", "cp")
    for tp in (1, 2, 4):
        for cp in (1, 2):
            assert _transport_degree(tp, cp) == tp * cp
