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

"""L1 — recompute policy (INTERFACES.md §2.7).

Which layers materialize a ``RECOMPUTE`` :class:`~program.work.WorkItem`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

from program.types import LayerId, MicroBatch
from program.workload import FrozenWorkload

__all__ = [
    "RecomputeError",
    "RecomputePolicy",
    "NoRecompute",
    "FullRecompute",
    "recompute_policy_for",
]


class RecomputeError(ValueError):
    """A recompute policy could not be selected."""


class RecomputePolicy(Protocol):
    name: str

    def materializes(
        self, layer: LayerId, microbatch: MicroBatch, fw: FrozenWorkload
    ) -> bool: ...


@dataclass(frozen=True)
class NoRecompute(RecomputePolicy):
    name: str = "none"

    def materializes(self, layer: LayerId, microbatch: MicroBatch, fw: FrozenWorkload) -> bool:
        return False


@dataclass(frozen=True)
class FullRecompute(RecomputePolicy):
    """Every layer of every microbatch materializes a RECOMPUTE WorkItem
    (schedule.py:716-732)."""

    name: str = "full"

    def materializes(self, layer: LayerId, microbatch: MicroBatch, fw: FrozenWorkload) -> bool:
        return True


def _granularity_expands_blocks(granularity: Any) -> bool:
    """``granularity is not Granularity.PIPELINE`` — the legacy ``flattened_mode``.

    L2's ``Granularity`` (``program/placement.py``) lands in P3; until then
    callers pass ``block_expanded=`` explicitly and this import is never
    reached. Deliberately an identity comparison against the enum, not a name
    or string test (INTERFACES §8 rule 2).
    """
    from program.placement import Granularity  # lazy; lands in P3

    return granularity is not Granularity.PIPELINE


def recompute_policy_for(
    fw: FrozenWorkload,
    granularity: Any = None,
    *,
    block_expanded: Optional[bool] = None,
) -> RecomputePolicy:
    """PRESERVES the existing predicate (schedule.py:270-278)::

        include_backward AND full_recomputation
                         AND (flattened_mode OR pipeline_style_recompute)

    but as an EXPLICIT dispatcher-time SELECTION rather than a
    ``misc["flattened_mode"]`` read inside the builder. Audit **C3** ("the
    enumerated schedule's structure depends on which backend consumes it") is
    thereby made visible: the coupling is now one documented line in a factory,
    and a caller may override it by passing a policy directly.

    ``block_expanded`` IS legacy ``misc["flattened_mode"]`` — the dispatcher
    knows it as ``granularity is not Granularity.PIPELINE`` and passes it here.
    """
    if block_expanded is None:
        if granularity is None:
            raise RecomputeError(
                "recompute_policy_for needs either a granularity or block_expanded=; "
                "the flattened_mode coupling is now an explicit dispatcher decision"
            )
        block_expanded = _granularity_expands_blocks(granularity)

    run = fw.spec.run
    enabled = (
        run.include_backward
        and run.full_recomputation
        and (bool(block_expanded) or run.pipeline_style_recompute)
    )
    return FullRecompute() if enabled else NoRecompute()
