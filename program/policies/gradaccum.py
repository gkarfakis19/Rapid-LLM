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

"""L1 — grad-accumulation emission gate (INTERFACES.md §2.5).

ONE implementation of a predicate that exists twice today:
``ScheduleSpec.should_emit_dp_comm`` (schedule.py:255-268) and its
re-derivation for the EP sync at schedule.py:937
(``apply_ep_all_mbs = dp_microbatch_mode != "last_mb" or zero_stage >= 3``) are
the same predicate written twice (audit **A6**). Every DP/ZeRO/EP requirement
in :mod:`program.policies.sharding` and :mod:`program.policies.routing` goes
through :meth:`GradAccumPolicy.emits`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

from program.types import MicroBatch
from program.workload import (
    CommSpec,
    DpMicrobatchMode,
    FrozenWorkload,
    GradAccumCycle,
)

__all__ = ["GradAccumPolicy", "grad_accum_policy_for"]


@dataclass(frozen=True)
class GradAccumPolicy:
    """Verbatim semantics of ``Graph._should_emit_dp_comm``.

    ``last_microbatch`` is BUG_LEDGER **Class B item 10h** as a NAMED
    ATTRIBUTE: the backward pass walks microbatches in reverse, so under GPipe
    ``b == 0`` IS the last microbatch. The semantics are correct and the legacy
    comment is merely confusing — but under 1F1B the constant is wrong, so it is
    data here and :meth:`for_schedule` sets it from the schedule
    (INTERFACES §4.1: ``SchedulePolicy.last_microbatch_of``).
    """

    name: str = "legacy"
    dp: int = 1
    zero_stage: int = 0
    cycle: GradAccumCycle = GradAccumCycle.FINAL
    mode: DpMicrobatchMode = DpMicrobatchMode.EVERY_MB
    #: Class B 10h — which microbatch index "last_mb" means under the schedule.
    last_microbatch: int = 0

    def emits(self, spec: CommSpec, microbatch: Optional[MicroBatch]) -> bool:
        if self.dp <= 1:
            return False
        if self.cycle is GradAccumCycle.NONFINAL:
            return spec.ga_required_every_cycle
        if spec.ga_required_every_cycle:
            return True
        if self.mode is not DpMicrobatchMode.LAST_MB or self.zero_stage >= 3:
            return True
        return microbatch == self.last_microbatch

    def for_schedule(self, schedule: Any) -> "GradAccumPolicy":
        """Rebind ``last_microbatch`` from a :class:`SchedulePolicy`'s schedule.

        GPipe's answer is ``0``; the 1F1B student project overrides
        ``last_microbatch_of`` and this policy needs no edit (INTERFACES §4.1).
        """
        resolved = schedule.last_microbatch_of()
        return replace(self, last_microbatch=int(resolved))


def grad_accum_policy_for(fw: FrozenWorkload) -> GradAccumPolicy:
    """The single construction site: every field is a typed WorkloadSpec read."""
    run = fw.spec.run
    return GradAccumPolicy(
        name="legacy",
        dp=int(fw.spec.degrees.dp),
        zero_stage=int(run.zero_stage),
        cycle=run.grad_accum_cycle,
        mode=run.dp_microbatch_mode,
    )
