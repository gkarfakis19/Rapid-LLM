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

"""Hybrid retiming write-back onto COARSE Programs (M5).

:func:`apply_block_timings` replaces the coarse-program half of the legacy
``LLMExecutionDispatcher._assign_transformer_durations`` recursive graph
walk: instead of matching node-name prefixes (``transformer_layer`` /
``vit_block``), layer COMPUTE ops are selected by metadata — ``role ==
TRANSFORMER_LAYER`` (which covers both the layer ops and the
pipeline-style recompute ops, exactly the set the legacy name-prefix walk
matched, recompute suffixes included) — and the per-DP duration tuple is
built from ``(dp_idx, stage)`` fault-variant overrides with the
dense/MoE baseline fallback, verbatim legacy semantics:

* MoE layers use the MoE baseline when present, else the dense baseline;
  dense layers with no dense baseline are SKIPPED (keep their analytical
  duration) — the legacy ``timing_source is None -> return`` rule;
* per-DP override lookup keys on ``(dp_idx, op.device)`` — the coarse op's
  device IS the legacy ``hw_id`` stage;
* forward/backward selection follows ``op.direction`` (recompute ops are
  forward, like the legacy ``node.fwd`` read);
* ``dp_count > 1`` writes a length-``dp_count`` tuple, else a single-entry
  tuple — the evaluator reads index 0 either way (legacy ``Node.duration``
  property, DESIGN.md §2 amendment 4).

The write-back is mirrored onto the aligned schedule events
(``meta.misc["coarse_events"]``, legacy scalar-vs-tuple convention). The
mirror is load-bearing since M6: the hierarchical pipeline emission lowers
the coarse proto root (``program.pipeline_coarse.lower_coarse_for_emission``),
and the lowering reads the events' ``duration_profile``/``duration`` —
exactly the legacy ``Node`` duration surface the deleted
``_apply_transformer_time``/``_assign_transformer_durations`` walk wrote.
``RAPID_VISUALIZE_GRAPHS`` renders the same retimed durations; the
analytical evaluator itself reads only the ops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from program.ir import ComputeOp, Direction, OpRole, Program


@dataclass(frozen=True)
class BlockTimings:
    """Transformer block timings from the AstraSim BLOCK-program runs.

    Entries are duck-typed timing pairs with ``forward``/``backward``
    attributes in seconds (the dispatcher passes its ``TransformerTimings``
    instances straight through). ``stage_dense``/``stage_moe`` map
    ``(dp_idx, stage)`` to fault-variant timings.
    """

    dense: Optional[Any] = None
    moe: Optional[Any] = None
    stage_dense: Mapping[Tuple[int, int], Any] = field(default_factory=dict)
    stage_moe: Mapping[Tuple[int, int], Any] = field(default_factory=dict)


def apply_block_timings(
    coarse_program: Program,
    timings: BlockTimings,
    dp_count: int,
) -> int:
    """Write per-DP duration tuples onto the layer COMPUTE ops.

    Returns the number of retimed ops (0 when no baseline applied — the
    legacy no-op case).
    """
    if not isinstance(coarse_program, Program):
        raise TypeError(
            f"apply_block_timings expects a Program (got {type(coarse_program).__name__})"
        )
    if coarse_program.meta.misc.get("granularity") != "coarse":
        raise RuntimeError("apply_block_timings requires a COARSE program")

    dp_count = max(1, int(dp_count))
    events = coarse_program.meta.misc.get("coarse_events")
    retimed = 0

    for op in coarse_program.ops:
        if not isinstance(op, ComputeOp) or op.role is not OpRole.TRANSFORMER_LAYER:
            continue

        is_moe_layer = bool(op.is_moe_layer)
        timing_source = (
            timings.moe if is_moe_layer and timings.moe is not None else timings.dense
        )
        if timing_source is None:
            timing_source = timings.dense
        if timing_source is None:
            # Legacy rule: no baseline -> keep the analytical duration.
            continue

        hw_stage = int(op.device)
        is_forward = op.direction is Direction.FORWARD
        overrides = timings.stage_moe if is_moe_layer else timings.stage_dense

        values = []
        for dp_idx in range(dp_count):
            default = timing_source.forward if is_forward else timing_source.backward
            timing_override = overrides.get((dp_idx, hw_stage))
            if timing_override:
                values.append(
                    float(timing_override.forward if is_forward else timing_override.backward)
                )
            else:
                values.append(float(default))

        op.duration = tuple(values)
        if events is not None:
            # Mirror for visualization (legacy tuple-vs-scalar convention).
            events[op.uid].duration = tuple(values) if dp_count > 1 else values[0]
        retimed += 1

    return retimed
