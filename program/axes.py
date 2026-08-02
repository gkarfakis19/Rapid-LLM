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

"""L0 — THE parallelism-axis registry.

Adding a parallelism axis was the ONE extension exercise the restructure did not
make cheap: measured at ratio **0.43**, against 0.03 for flattened MoE, because
every axis fact was spelled out by hand in a different module —

* the canonical order and the axis set  (``layout.CANONICAL_AXES``, a literal);
* the cluster/pipeline split            (``layout.py:248-260``, ``:303``,
  ``placement.CLUSTER_AXES``, all literal tuples);
* the coordinate decomposition          (``layout.cluster_coords``, one
  hand-written ``if "<axis>" in axis_order`` block per axis with its own stride
  arithmetic);
* the "layout must declare this axis" checks (``layout.py:310-315``, three
  near-identical blocks);
* which axes a granularity materializes (``placement._GRANULARITY_AXES``);
* which axes shard an activation       (``placement.activation_shard_size``,
  a literal ``tp * cp``);
* which axes a dp collective may span  (``work._DP_COMPANION_AXES``).

Each of those is a *property of an axis*, so each belongs on the axis. This
module is that one place: a new axis is a new :class:`AxisSpec` row plus a
degree field, and every list above re-derives.

**The payoff is not only brevity.** BUG_LEDGER 19 — cp gradients are partial and
were never summed — is exactly the conjunction
``shards_work and replicates_parameters``, which no other axis has. Written down
here, that bug is a readable property of a table row rather than something you
have to notice. :func:`axes_needing_gradient_reduction` is the query, and
``work._DP_COMPANION_AXES`` is now its result instead of a hand-kept list.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Mapping, Sequence, Tuple

AxisName = str

__all__ = [
    "AxisName",
    "AxisRole",
    "AxisSpec",
    "AXES",
    "AXIS_BY_NAME",
    "CANONICAL_AXES",
    "CLUSTER_AXES",
    "PIPELINE_AXES",
    "REPLICA_AXES",
    "DP_AXIS",
    "axes_with_role",
    "axes_needing_gradient_reduction",
    "activation_sharding_axes",
    "axis_sizes_from",
    "cluster_strides",
]


class AxisRole(Enum):
    """Where an axis lives relative to a pipeline stage's device cluster."""

    #: inside one stage's cluster of devices (tp, cp, ep). Devices differing
    #: only in a CLUSTER axis are peers of the same stage.
    CLUSTER = auto()
    #: across stages (pp). This axis IS the stage index.
    PIPELINE = auto()
    #: outside the device space entirely (dp). A dp collective carries no
    #: ``GroupKey``; replication is stamped at emission over pre-dp device ids.
    REPLICA = auto()


@dataclass(frozen=True)
class AxisSpec:
    """Everything the core needs to know about one parallelism axis."""

    name: AxisName
    role: AxisRole
    #: attribute on ``workload.ParallelDegrees`` holding this axis's degree.
    degree_attr: str
    #: human name, for diagnostics ("tensor parallelism"). Kept on the axis so a
    #: generic check can still produce the specific message a reader needs.
    label: str

    #: This axis splits the WORK (tokens / sequence positions) across its ranks,
    #: so each rank computes over a strict subset. tp splits tensors, not work,
    #: and is therefore False; cp splits the sequence; ep splits the tokens.
    shards_work: bool = False

    #: This axis's ranks hold the SAME parameters. tp and ep each own a distinct
    #: shard of the weights, so they are False; cp ranks are full replicas.
    replicates_parameters: bool = False

    #: This axis shards a pipeline-boundary ACTIVATION (the residual stream), so
    #: it divides the cross-layer transfer payload. BUG_LEDGER 10c: this is
    #: ``tp * cp`` and pointedly NOT ``ep`` — EP ranks own DISTINCT tokens, so
    #: the raw byte count is already one owner's microbatch.
    shards_activations: bool = False

    def __post_init__(self) -> None:
        if self.replicates_parameters and not self.shards_work:
            # A replica that computes over the SAME work is a plain replica
            # (that is dp, which is REPLICA-role and outside the device space).
            # Inside the device space, replicated parameters only make sense
            # when the work is split -- otherwise the rank is redundant.
            raise ValueError(
                f"AxisSpec({self.name!r}) replicates parameters without sharding "
                "work; such a rank would be pure duplicate compute"
            )


#: THE registry. Order is canonical order — tp fastest-varying, dp last —
#: which is what makes ``RankLayout``'s row-major strides and the
#: dp-excluded ``subset`` stride-preserving (INTERFACES §3.3).
AXES: Tuple[AxisSpec, ...] = (
    AxisSpec("tp", AxisRole.CLUSTER, "tp", "tensor parallelism", shards_work=False,
             replicates_parameters=False, shards_activations=True),
    AxisSpec("cp", AxisRole.CLUSTER, "cp", "context parallelism", shards_work=True,
             replicates_parameters=True, shards_activations=True),
    AxisSpec("ep", AxisRole.CLUSTER, "ep", "expert parallelism", shards_work=True,
             replicates_parameters=False, shards_activations=False),
    AxisSpec("pp", AxisRole.PIPELINE, "pp", "pipeline parallelism"),
    AxisSpec("dp", AxisRole.REPLICA, "dp", "data parallelism", shards_work=True),
)

AXIS_BY_NAME: Mapping[AxisName, AxisSpec] = {axis.name: axis for axis in AXES}


def axes_with_role(*roles: AxisRole) -> Tuple[AxisName, ...]:
    """Axis names with any of ``roles``, in canonical order."""
    wanted = set(roles)
    return tuple(axis.name for axis in AXES if axis.role in wanted)


CANONICAL_AXES: Tuple[AxisName, ...] = tuple(axis.name for axis in AXES)
CLUSTER_AXES: Tuple[AxisName, ...] = axes_with_role(AxisRole.CLUSTER)
PIPELINE_AXES: Tuple[AxisName, ...] = axes_with_role(AxisRole.PIPELINE)
REPLICA_AXES: Tuple[AxisName, ...] = axes_with_role(AxisRole.REPLICA)

#: The single REPLICA axis, named for the many call sites that want it directly.
DP_AXIS: AxisName = REPLICA_AXES[0]


def axes_needing_gradient_reduction() -> Tuple[AxisName, ...]:
    """In-device-space axes whose ranks hold PARTIAL gradients of SHARED weights.

    **This is BUG_LEDGER 19 stated as a query.** Such an axis splits the work
    (so each rank's gradient is partial) while replicating the parameters (so
    the partials are contributions to one tensor). They must therefore be summed,
    which means the gradient reducer's communicator spans them alongside dp.

    Returns ``('cp',)`` today. ``tp``/``ep`` are excluded because each owns a
    distinct parameter shard, and ``dp`` is excluded because it is the reduction
    axis itself, not a companion.
    """
    return tuple(
        axis.name
        for axis in AXES
        if axis.role is AxisRole.CLUSTER
        and axis.shards_work
        and axis.replicates_parameters
    )


def activation_sharding_axes() -> Tuple[AxisName, ...]:
    """Axes that divide a pipeline-boundary activation (BUG_LEDGER 10c)."""
    return tuple(axis.name for axis in AXES if axis.shards_activations)


def axis_sizes_from(degrees: object) -> Dict[AxisName, int]:
    """``{axis: degree}`` read off a ``ParallelDegrees`` through the registry.

    Replaces the hand-written dict literal at ``placement.py:306-310``; a new
    axis needs no edit here.
    """
    return {
        axis.name: max(1, int(getattr(degrees, axis.degree_attr, 1) or 1))
        for axis in AXES
    }


def cluster_strides(
    axis_order: Sequence[AxisName], sizes: Mapping[AxisName, int]
) -> Dict[AxisName, int]:
    """Row-major strides for the CLUSTER axes present in ``axis_order``.

    The generalization of ``layout.cluster_coords``' hand-written ladder
    (``tp_rank % tp``, ``(tp_rank // tp) % cp``, ``(tp_rank // (tp*cp)) % ep``):
    each cluster axis's stride is the product of the sizes of the cluster axes
    before it in canonical order. Stated once, it is correct for any number of
    cluster axes rather than for exactly three.
    """
    strides: Dict[AxisName, int] = {}
    running = 1
    for name in CLUSTER_AXES:
        if name not in axis_order:
            continue
        strides[name] = running
        running *= max(1, int(sizes.get(name, 1)))
    return strides
