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


def recompute_policy_for(
    fw: FrozenWorkload,
    granularity: Any = None,
    *,
    block_expanded: Optional[bool] = None,
) -> RecomputePolicy:
    """``FullRecompute`` iff the run trains with full recomputation.

    The legacy predicate (schedule.py:270-278) was::

        include_backward AND full_recomputation
                         AND (flattened_mode OR pipeline_style_recompute)

    and audit **C3** flagged the third conjunct as a granularity coupling
    ("the enumerated schedule's structure depends on which backend consumes
    it"). It never was one (established 2026-08-02): ``train_timing.py:297``
    hardwired ``pipeline_style_recompute = bool(full_recomputation)`` with no
    config path, so whenever the outer ``full_recomputation`` conjunct held,
    the disjunct held too, and the predicate reduced to the first two terms
    in every reachable state. The flag and the disjunct are deleted; RECOMPUTE
    work items exist per (run, model) — never per backend.

    ``granularity`` / ``block_expanded`` are ACCEPTED AND IGNORED so the
    dispatcher call sites did not all have to change in the same commit; both
    are deprecated.
    """
    run = fw.spec.run
    enabled = run.include_backward and run.full_recomputation
    return FullRecompute() if enabled else NoRecompute()
