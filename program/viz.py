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

"""Graphviz rendering of a :class:`~program.ir.Program` (P6).

Reads the **Program**; the proto event graphs are gone. The classification is
the legacy renderer's, restated over typed fields instead of ``hasattr``
probes:

* :class:`~program.ir.CollectiveOp` / :class:`~program.ir.TransferOp` -> comm:
  green when ``is_dp``, white for EVERY ``PIPELINE`` comm op (including
  byte-carrying cross-stage ``cross_layer`` transfers — the legacy renderer
  never checked bytes), yellow for the remaining collectives. A collective's
  label shows its owning device (the legacy ``local_hw_id``);
* :class:`~program.ir.ComputeOp` -> compute: lightblue forward / lightcoral
  backward, labelled with the scalar duration or the per-DP profile written by
  :func:`program.retime.apply_block_timings` (grouped by equal values, the
  legacy format).

Comm durations come from ``meta.misc[analytic_sim.COMM_DURATIONS_KEY]`` when the
analytical conversion ran (it is what the evaluator used), else 0 — the legacy
renderer read the durations the conversion pass had written onto the events.

Edges are drawn in program order (ascending uid on both endpoints), the tie
discipline :mod:`program.analytic_sim` documents.

:func:`save_program_graph` preserves the legacy output contract byte-for-byte:
``<output_folder><filename>.svg`` via :func:`util.graphviz_submit` (async when
``RAPID_VISUALIZE_GRAPHS`` is set, else inline), with the same "Graph saved to"
completion message.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from graphviz import Digraph

import util
from timing_model import CollectiveType

from program.analytic_sim import COMM_DURATIONS_KEY
from program.ir import CollectiveOp, ComputeOp, Direction, Program, TransferOp

__all__ = ["save_program_graph", "visualize_program"]


def _format_duration(value: float, profile: Optional[Tuple[float, ...]] = None) -> str:
    def _format_single(entry: float) -> str:
        ms = entry * 1e3
        if abs(ms) > 1000:
            return f"{entry:.2f}s"
        return f"{ms:.2f}ms"

    if not profile:
        return _format_single(value)

    groups: List[Dict[str, Any]] = []
    for idx, entry in enumerate(profile):
        matched = False
        for group in groups:
            if math.isclose(entry, group["value"], rel_tol=1e-9, abs_tol=1e-12):
                group["indices"].append(idx)
                matched = True
                break
        if not matched:
            groups.append({"value": entry, "indices": [idx]})

    if len(groups) == 1:
        return _format_single(groups[0]["value"])

    parts = []
    for group in groups:
        indices = ",".join(str(i) for i in group["indices"])
        parts.append(f"{{{indices}}}: {_format_single(group['value'])}")
    return ", ".join(parts)


def _op_color(op: Any) -> str:
    if isinstance(op, ComputeOp):
        return "lightblue" if op.direction is Direction.FORWARD else "lightcoral"
    if isinstance(op, CollectiveOp):
        if op.is_dp:
            return "green"
        if op.coll is CollectiveType.PIPELINE:
            return "white"
        return "yellow"
    if isinstance(op, TransferOp):
        return "white" if op.comm_type is CollectiveType.PIPELINE else "yellow"
    return "mediumorchid"


def _op_label(op: Any, duration: float) -> str:
    if isinstance(op, ComputeOp):
        profile = tuple(op.duration) if len(op.duration) > 1 else None
        value = op.duration[0] if op.duration else 0.0
        return f"{op.name}\n(hw_id={op.device}, dur={_format_duration(value, profile)})"
    display = _format_duration(duration)
    if isinstance(op, CollectiveOp):
        return f"{op.name}\n(local_hw_id={op.device}, dur={display})"
    return f"{op.name}\n(dur={display})"


def visualize_program(program: Program) -> Digraph:
    """Render a Program's dependency DAG to a Digraph."""
    if not isinstance(program, Program):
        raise TypeError(
            f"visualize_program expects a Program (got {type(program).__name__})"
        )
    stored: Optional[Sequence[float]] = program.meta.misc.get(COMM_DURATIONS_KEY)

    dot = Digraph(comment="Computation Graph", format="svg")
    for op in program.ops:
        duration = 0.0 if stored is None else float(stored[op.uid])
        dot.node(
            str(op.uid),
            label=_op_label(op, duration),
            style="filled",
            fillcolor=_op_color(op),
            shape="box",
        )
    for op in program.ops:
        for dep in sorted(op.deps):
            dot.edge(str(dep), str(op.uid))
    return dot


def save_program_graph(
    program: Program, output_folder: str = "output/LLM/", filename: str = "graph"
) -> None:
    """Render + write ``<output_folder><filename>.svg`` (legacy
    ``Graph.save_graph`` contract, async via :func:`util.graphviz_submit`)."""
    os.makedirs(output_folder, exist_ok=True)

    base_path = os.path.normpath(f"{output_folder}{filename}")
    svg_path = f"{base_path}.svg"
    display_path = util.relpath_display(svg_path)
    printstr = f" | Graph saved to    {display_path}"

    def _render_graph() -> None:
        dot = visualize_program(program)
        dot.render(base_path, format="svg", cleanup=True)

    util.graphviz_submit(f"{filename}.svg", _render_graph, print_message=printstr)
