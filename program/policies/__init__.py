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

"""L1 — the policy registry (INTERFACES.md §0.3, §2).

Every policy is a ``@dataclass(frozen=True)`` with a ``name: str``. A reviewer
must be able to flip any BUG_LEDGER Class B choice by CONSTRUCTING a different
policy, with no edit inside ``build.py`` — :func:`policies_for` is the only
place the defaults are chosen, and :class:`PolicyBundle` is what ``build()``
receives.

Class B / Class A binding table (INTERFACES §7), as implemented:

===========  ===========================================================
ledger id    named policy attribute
===========  ===========================================================
B 1          ``work.ByteSource.split`` (default ``ByteSplit.WHOLE``)
B 5          ``workload.CommSpec.local_comp_time`` (carried, unused)
B 8          ``workload.RunPolicy.interleave_scale``
B 9          R2 same-device ``TransferOp`` (L4)
B 10d        ``work.SyncSpread`` (default ``CLUSTER_RANK_0``)
B 10g        ``et_emit`` (unchanged)
B 10h        ``gradaccum.GradAccumPolicy.last_microbatch``
A2           ``work.SyncSpread`` / ``sharding._spread_for``
A3           derived from ``WorkItem`` — already impossible
10b (fixed)  ``work.enumerate_work`` (one FUSED OPTIMIZER per stage) +
             ``work.optimizer_duration`` (priced over the stage's OWN layers)
10c (fixed)  ``work.ByteSplit.CEIL_DIV_CLUSTER`` on ``cross_layer``, divided
             by ``placement.Placement.activation_shard_size()`` (``tp*cp``) —
             NOT by ``cluster_size()`` (``tp*cp*ep``)
===========  ===========================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from program.policies.gradaccum import GradAccumPolicy, grad_accum_policy_for
from program.policies.overlap import (
    AxisFractionOverlap,
    NoOverlap,
    OverlapAnchor,
    OverlapDecl,
    OverlapPolicy,
    overlap_policy_for,
)
from program.policies.recompute import (
    FullRecompute,
    NoRecompute,
    RecomputePolicy,
    recompute_policy_for,
)
from program.policies.routing import (
    ROUTING_MODES,
    AxisRouting,
    MoERoutingPolicy,
    routing_for_mode,
    routing_policy_for,
)
from program.policies.sharding import (
    DDP,
    SHARDING_POLICIES,
    ContiguousStages,
    NullSharding,
    ShardingContext,
    ShardingPolicy,
    StagePartition,
    ZeRO1,
    ZeRO2,
    ZeRO3,
    sharding_policy_for,
)

__all__ = [
    "PolicyBundle",
    "policies_for",
    # sharding
    "ShardingPolicy",
    "ShardingContext",
    "StagePartition",
    "ContiguousStages",
    "NullSharding",
    "DDP",
    "ZeRO1",
    "ZeRO2",
    "ZeRO3",
    "SHARDING_POLICIES",
    "sharding_policy_for",
    # grad accumulation
    "GradAccumPolicy",
    "grad_accum_policy_for",
    # recompute
    "RecomputePolicy",
    "NoRecompute",
    "FullRecompute",
    "recompute_policy_for",
    # routing
    "MoERoutingPolicy",
    "AxisRouting",
    "ROUTING_MODES",
    "routing_for_mode",
    "routing_policy_for",
    # overlap
    "OverlapPolicy",
    "OverlapAnchor",
    "OverlapDecl",
    "NoOverlap",
    "AxisFractionOverlap",
    "overlap_policy_for",
]


@dataclass(frozen=True)
class PolicyBundle:
    """The complete policy selection for one build. Every member is a named,
    swappable object; ``build()`` takes them as keyword arguments and makes no
    policy decision of its own (INTERFACES §4.2)."""

    sharding: ShardingPolicy
    grad_accum: GradAccumPolicy
    recompute: RecomputePolicy
    overlap: OverlapPolicy
    routing: Optional[MoERoutingPolicy] = None

    @property
    def names(self) -> dict:
        """Diagnostics: the selected policy names, for ``Program.meta.misc``."""
        return {
            "sharding": self.sharding.name,
            "grad_accum": self.grad_accum.name,
            "recompute": self.recompute.name,
            "overlap": self.overlap.name,
            "routing": None if self.routing is None else self.routing.name,
        }


def policies_for(
    spec: Any,
    *,
    granularity: Any = None,
    block_expanded: Optional[bool] = None,
) -> PolicyBundle:
    """The DEFAULT policy selection for a workload — today's behavior, named.

    ``spec`` may be a :class:`~program.workload.WorkloadSpec` or an already
    frozen :class:`~program.workload.FrozenWorkload`. ``granularity`` /
    ``block_expanded`` select the recompute policy (INTERFACES §2.7); the
    coupling is one documented argument, not a ``misc["flattened_mode"]`` read
    inside a builder.

    No granularity-specific SHARDING override is needed: a BLOCK workload is a
    single-replica block measurement (``degrees.dp == 1``, which ``build()``
    enforces — INTERFACES §3.1 and the legacy ``dp_override=1``), and
    ``sharding_policy_for`` already answers :class:`NullSharding` at ``dp <= 1``.
    """
    from program.workload import WorkloadSpec

    fw = spec.freeze() if isinstance(spec, WorkloadSpec) else spec
    return PolicyBundle(
        sharding=sharding_policy_for(fw.spec.run, fw.spec.degrees),
        grad_accum=grad_accum_policy_for(fw),
        recompute=recompute_policy_for(
            fw, granularity, block_expanded=block_expanded
        ),
        overlap=overlap_policy_for(fw),
        routing=routing_policy_for(fw),
    )
