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

"""L1 — overlap DECLARED on a requirement (INTERFACES.md §2.8).

This module **declares** overlap; it does not rewrite anything. Neither of the
two competing rewrite implementations in :mod:`program.transforms` is ported:

* the proto rewrite ``_split_tp_node_fine`` / ``_split_cp_edge_fine``
  (transforms.py:169-299, the only one production uses), and
* the Program-level ``apply_tp_overlap`` (transforms.py:430-556, no production
  caller)

are both replaced by one declaration here plus one realizer at L4
(INTERFACES §4.5). The consequences that fall out:

* the ``head`` / ``_block`` / ``_ovlp`` op-id-reuse quirks (transforms.py:200,
  :255, :267 — Class C item 10e) are deleted, because program order is computed
  (§4.6) and split ops get fresh uids;
* the ``"attention" in name.lower()`` substring test (transforms.py:224) becomes
  DATA on the policy (:attr:`OverlapDecl.blocking_consumer`);
* the mode -> axis-fraction selection (transforms.py:326-374) happens ONCE, in
  :meth:`~program.workload.OverlapSpec.from_legacy`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from types import MappingProxyType
from typing import Mapping, Optional, Protocol

from program.types import AxisName
from program.workload import CommSpec, FrozenWorkload

__all__ = [
    "OverlapError",
    "OverlapAnchor",
    "OverlapDecl",
    "OverlapPolicy",
    "NoOverlap",
    "AxisFractionOverlap",
    "overlap_policy_for",
]


class OverlapError(ValueError):
    """An overlap declaration is malformed."""


class OverlapAnchor(Enum):
    """What gets split to create the overlap."""

    #: split the COMPUTE that produces the collective's input (tp / tp_sp)
    PRODUCER = auto()
    #: split the COLLECTIVE by bytes; the blocking consumer waits on the
    #: blocking part (cp)
    CONSUMER = auto()


@dataclass(frozen=True)
class OverlapDecl:
    """Overlap as data hanging off a comm key / requirement."""

    fraction: float  #: in (0, 1]; >= 1.0 means "fully hoisted"
    anchor: OverlapAnchor
    blocking_consumer: Optional[str] = None
    """For CONSUMER anchoring: the ``BlockTemplate`` entry name that must wait
    for the blocking part. Today this is the substring test
    ``"attention" in name.lower()`` (transforms.py:224) against a
    TEMPLATE-SUPPLIED name; here it is DATA on the policy. The default
    ``"attention"`` reproduces today exactly."""

    def __post_init__(self) -> None:
        if not isinstance(self.anchor, OverlapAnchor):
            raise OverlapError("OverlapDecl.anchor must be an OverlapAnchor")
        if float(self.fraction) <= 0.0:
            raise OverlapError(
                f"OverlapDecl.fraction must be > 0 (got {self.fraction}); "
                "absence of overlap is declared by returning None"
            )
        if self.anchor is OverlapAnchor.CONSUMER and self.blocking_consumer is None:
            raise OverlapError(
                "OverlapDecl(anchor=CONSUMER) requires a blocking_consumer "
                "(the BlockTemplate entry that waits on the blocking part)"
            )
        object.__setattr__(self, "fraction", float(self.fraction))

    @property
    def is_hoist(self) -> bool:
        """``fraction >= 1.0``: the collective is fully hoisted rather than
        split (transforms.py:177-188 / :233-244)."""
        return self.fraction >= 1.0


class OverlapPolicy(Protocol):
    name: str

    def declare(self, spec: CommSpec, fw: FrozenWorkload) -> Optional[OverlapDecl]:
        """Called once per block-template comm key. ``None`` means no overlap."""


@dataclass(frozen=True)
class NoOverlap(OverlapPolicy):
    name: str = "none"

    def declare(self, spec: CommSpec, fw: FrozenWorkload) -> Optional[OverlapDecl]:
        return None


#: axis -> what to split. transforms.py anchors tp/tp_sp collectives on the
#: PRODUCER compute and cp collectives on the CONSUMER, as a table not two
#: near-duplicate blocks.
_DEFAULT_ANCHOR_BY_AXIS: Mapping[AxisName, OverlapAnchor] = MappingProxyType(
    {"tp": OverlapAnchor.PRODUCER, "cp": OverlapAnchor.CONSUMER}
)


@dataclass(frozen=True)
class AxisFractionOverlap(OverlapPolicy):
    """The only production policy.

    Reads :class:`~program.workload.OverlapSpec.by_axis` — which already
    resolved ``tp_sp`` into ``tp`` at construction — and selects the anchor from
    :attr:`anchor_by_axis`. The axis a collective overlaps on is its FIRST
    declared communicator axis, i.e. the legacy ``comm_interconnect_type``;
    note this is deliberately NOT the participant axis (they differ for
    ``TENSOR_CONTEXT_HYBRID`` ``output_proj``, train_timing.py:151-154, and
    legacy also selected on the interconnect).
    """

    name: str = "axis_fraction"
    anchor_by_axis: Mapping[AxisName, OverlapAnchor] = field(
        default_factory=lambda: _DEFAULT_ANCHOR_BY_AXIS
    )
    #: Class B: the default blocking consumer for CONSUMER-anchored overlap.
    blocking_consumer: str = "attention"

    def overlap_axis(self, spec: CommSpec) -> AxisName:
        return spec.axes[0]

    def declare(self, spec: CommSpec, fw: FrozenWorkload) -> Optional[OverlapDecl]:
        axis = self.overlap_axis(spec)
        anchor = self.anchor_by_axis.get(axis)
        if anchor is None:
            return None
        fraction = fw.spec.overlap.fraction(axis)
        if fraction <= 0.0:
            return None
        return OverlapDecl(
            fraction=fraction,
            anchor=anchor,
            blocking_consumer=(
                self.blocking_consumer if anchor is OverlapAnchor.CONSUMER else None
            ),
        )


def overlap_policy_for(fw: FrozenWorkload) -> OverlapPolicy:
    """``AxisFractionOverlap`` whenever any axis declares a fraction, else
    :class:`NoOverlap` — so "no overlap" is a policy object, not a falsy float
    threaded through eight signatures."""
    if any(value > 0.0 for value in fw.spec.overlap.by_axis.values()):
        return AxisFractionOverlap()
    return NoOverlap()
