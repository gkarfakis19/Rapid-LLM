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

"""``program.axes`` is THE parallelism-axis registry — nothing re-derives it.

Adding an axis was the one extension exercise the restructure left expensive
(ratio 0.43) because every axis fact was a separate hand-written literal. These
tests are what keep it collapsed: they fail if a module grows its own copy.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from program.axes import (
    AXES,
    AXIS_BY_NAME,
    CANONICAL_AXES,
    CLUSTER_AXES,
    PIPELINE_AXES,
    REPLICA_AXES,
    AxisRole,
    AxisSpec,
    activation_sharding_axes,
    axes_needing_gradient_reduction,
    axis_sizes_from,
    cluster_strides,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_roles_partition_the_axes():
    assert set(CLUSTER_AXES) | set(PIPELINE_AXES) | set(REPLICA_AXES) == set(CANONICAL_AXES)
    assert len(CLUSTER_AXES) + len(PIPELINE_AXES) + len(REPLICA_AXES) == len(CANONICAL_AXES)
    assert CANONICAL_AXES == ("tp", "cp", "ep", "pp", "dp")


def test_gradient_reduction_query_is_bug_ledger_19():
    """The cp bug, stated as a property rather than noticed.

    An axis needs its gradients reduced exactly when it splits the work (so each
    rank's gradient is partial) AND replicates the parameters (so the partials
    sum to one tensor). tp and ep each own a distinct shard; dp is the reduction
    axis itself.
    """
    assert axes_needing_gradient_reduction() == ("cp",)
    assert AXIS_BY_NAME["cp"].shards_work and AXIS_BY_NAME["cp"].replicates_parameters
    for name in ("tp", "ep"):
        assert not AXIS_BY_NAME[name].replicates_parameters


def test_work_dp_companion_axes_is_the_query_not_a_copy():
    from program.work import _DP_COMPANION_AXES

    assert _DP_COMPANION_AXES == axes_needing_gradient_reduction()


def test_activation_sharding_is_tp_cp_and_not_ep():
    """BUG_LEDGER 10c: EP ranks own DISTINCT tokens, so the pipeline-boundary
    payload is already one owner's microbatch and must not be divided again."""
    assert activation_sharding_axes() == ("tp", "cp")
    assert not AXIS_BY_NAME["ep"].shards_activations


def test_placement_activation_shard_size_uses_the_registry():
    import sys

    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from test_policies import Cfg, make_workload
    from program.placement import Granularity, Placement
    from program.schedule.gpipe import GPipeSchedule

    for cfg, expected in (
        (Cfg(dp=1, tp=2, cp=2, ep=1, pp=2, mb=2), 4),
        (Cfg(dp=1, tp=2, cp=1, ep=2, pp=2, mb=2, moe=True), 2),  # ep excluded
        (Cfg(dp=1, tp=1, cp=1, ep=1, pp=2, mb=2), 1),
    ):
        fw = make_workload(cfg).freeze()
        placement = Placement(
            fw, Granularity.FLAT, GPipeSchedule().layer_assignment(fw)
        )
        assert placement.activation_shard_size() == expected, cfg


def test_cluster_strides_are_row_major_over_cluster_axes():
    sizes = {"tp": 2, "cp": 3, "ep": 5, "pp": 7, "dp": 11}
    strides = cluster_strides(CANONICAL_AXES, sizes)
    assert strides == {"tp": 1, "cp": 2, "ep": 6}
    # absent axes are skipped, and the running product skips with them
    assert cluster_strides(("tp", "ep"), sizes) == {"tp": 1, "ep": 2}


def test_axis_sizes_from_reads_every_axis_through_the_registry():
    class Degrees:
        tp, cp, ep, pp, dp = 2, 3, 4, 5, 6

    assert axis_sizes_from(Degrees()) == {"tp": 2, "cp": 3, "ep": 4, "pp": 5, "dp": 6}


def test_replicating_without_sharding_work_is_rejected():
    """A CLUSTER rank that holds the same parameters AND does the same work is
    pure duplicate compute — the registry refuses to describe one."""
    with pytest.raises(ValueError, match="replicates parameters without sharding"):
        AxisSpec("xx", AxisRole.CLUSTER, "xx", "nonsense", replicates_parameters=True)


def test_no_module_reintroduces_a_hand_written_axis_tuple():
    """Source-level guard: the canonical axis list appears ONCE, in the registry.

    This is the check that keeps the 0.43 extension ratio from creeping back —
    a second literal ``("tp", "cp", "ep", ...)`` anywhere in ``program/`` is a
    module re-deriving what ``program.axes`` already declares.
    """
    axis_names = set(CANONICAL_AXES)
    offenders = []
    for path in sorted((REPO_ROOT / "program").rglob("*.py")):
        if path.name == "axes.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Tuple, ast.List)):
                continue
            elts = node.elts
            if len(elts) < 3:
                continue
            if not all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in elts):
                continue
            values = [e.value for e in elts]
            if len(set(values) & axis_names) >= 3:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} {values}")
    assert not offenders, (
        "hand-written axis tuples found; derive them from program.axes instead:\n  "
        + "\n  ".join(offenders)
    )
