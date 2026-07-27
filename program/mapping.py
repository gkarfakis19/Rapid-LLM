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

"""First-dimension SCOTCH remap over a built :class:`~program.ir.Program` (P5).

The gmap traffic collection and the SCOTCH stage permutation used to be Steps 3
and 8 inside ``legacy_lowering.lower_to_program``, reading a mutable proto graph.
They are a **post-build, emission-side** pass over the ops:

1. collect per-axis collective bytes and cross-stage pipeline bytes off the
   ops (``astrasim_lib.gmap`` collector API, unchanged);
2. ask SCOTCH for a permutation of the first-dimension vertices;
3. reorder ``Program.devices``.

Step 3 IS the remap: ``Program.devices.index(d)`` is the stage index the
dp-major rank formula uses (``Program.rank_for``), and communicator members are
DEVICE ids, so permuting the device order re-derives every wire group and every
rank without touching an op. Legacy re-ran ``_build_axis_groups`` +
``_assign_collective_labels`` for the same reason; here there is nothing to
re-run.

Op uids are **not** recomputed. Program order is assigned by ``build()`` from
the pre-remap ``placement.devices()`` order (INTERFACES §4.6) and the remap is a
mapping decision, not a scheduling one — the same split legacy had (BUG_LEDGER
C7: Step 11 iterated the pre-remap order, "endpoints are correct").
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Dict, Mapping, Optional, Tuple

from astrasim_lib import gmap
from timing_model import CollectiveType

from program.ir import CollectiveOp, Program, TransferOp

__all__ = ["apply_first_dim_mapping"]

#: Collectives whose gmap weight is doubled (legacy ``should_double``).
_DOUBLE_WEIGHT = (CollectiveType.ALL_REDUCE, CollectiveType.ALL_TO_ALL)


def _axis_layout(program: Program) -> Tuple[Dict[str, int], Dict[int, Dict[str, int]]]:
    layout = program.layout
    axis_sizes = {
        str(axis): max(1, int(layout.axis_sizes.get(axis, 1)))
        for axis in layout.axis_order
    }
    coords = {
        int(device): {
            str(axis): int(value) for axis, value in layout.coords_of(int(device)).items()
        }
        for device in program.devices
    }
    return axis_sizes, coords


def apply_first_dim_mapping(
    program: Program,
    *,
    optimize_2dmap: Optional[Mapping[str, Any]],
    workdir: Optional[str] = None,
) -> Program:
    """Reorder ``program.devices`` by the SCOTCH first-dimension mapping.

    A no-op (returns ``program`` unchanged) when ``optimize_2dmap`` is falsy or
    the layout carries no axes. ``workdir`` receives the ``first_dim_comm.*``
    artifacts; a temp dir is used and removed when omitted.
    """
    if not optimize_2dmap:
        return program
    axis_sizes, stage_axis_coords = _axis_layout(program)
    if not axis_sizes:
        return program

    tmpdir: Optional[str] = None
    if workdir is None:
        tmpdir = tempfile.mkdtemp(prefix="rapid_gmap_")
        workdir = tmpdir
    os.makedirs(workdir, exist_ok=True)
    try:
        collector = gmap.begin_collection(
            optimize_2dmap, axis_sizes, stage_axis_coords, workdir
        )
        if collector is None:  # pragma: no cover - begin_collection guards
            return program

        for op in program.ops:
            if isinstance(op, CollectiveOp):
                axis = str(op.interconnect or "").strip().lower()
                if not axis or axis not in collector.subset_axes:
                    continue
                collector.record_collective(
                    axis=axis,
                    stage_id=int(op.device),
                    size_bytes=int(op.size_bytes),
                    participant_count=int(op.participants),
                    double_weight=op.coll in _DOUBLE_WEIGHT,
                )
            elif isinstance(op, TransferOp):
                if not collector.include_pipeline:
                    continue
                if op.comm_type is not CollectiveType.PIPELINE:
                    continue
                if int(op.src_device) == int(op.dst_device):
                    continue
                collector.record_pipeline(
                    src_stage=int(op.src_device),
                    dst_stage=int(op.dst_device),
                    size_bytes=int(op.size_bytes),
                )

        result = gmap.finalize_collection(collector)
        if not result:
            return program
        permutation = result.permutation
        if len(permutation) != collector.vertex_count:
            raise ValueError(
                "Permutation length does not match first-dimension vertex count."
            )

        def _sort_key(device: int) -> Tuple[int, ...]:
            coords = collector.stage_axis_coords.get(device, {})
            higher = tuple(
                int(coords.get(axis, 0)) for axis in collector.replication_axes
            )
            return higher + (permutation[collector.local_index_for_stage(device)],)

        program.devices = tuple(sorted(program.devices, key=_sort_key))
        program.meta.misc["first_dim_mapping"] = tuple(int(v) for v in permutation)
        return program
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)
