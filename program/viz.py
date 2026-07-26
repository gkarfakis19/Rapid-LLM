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

"""Graphviz rendering of schedule/fine proto event graphs (M8).

Port of the legacy ``simulate_train_graph.visualize_graph`` /
``Graph.save_graph`` pair over the Program builders' proto events — the
coarse schedule events (:class:`program.schedule.ComputeEvent` /
``CommEvent``) and the fine builder's :class:`program.pipeline_fine.
FineNode`/``FineEdge``. The classification is duck-typed exactly like the
legacy renderer's event branches:

* ``comm_size_bytes`` present -> comm event: green when ``is_dp``, white
  for EVERY ``PIPELINE`` comm event — including byte-carrying cross-stage
  ``cross_layer`` transfers; the legacy renderer never checked bytes —
  yellow for the remaining (non-dp, non-PIPELINE) collectives; the label
  shows ``local_hw_id`` when placed;
* ``hw_id`` present -> compute event: lightblue forward / lightcoral
  backward; the label shows the scalar duration or the per-DP
  ``duration_profile`` written by :func:`program.retime.apply_block_timings`
  (grouped by equal values, the legacy format).

``save_events_graph`` preserves the legacy output contract byte-for-byte:
``<output_folder><filename>.svg`` via :func:`util.graphviz_submit` (async
when ``RAPID_VISUALIZE_GRAPHS`` is set, else inline), with the same
"Graph saved to" completion message.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

from graphviz import Digraph

import util
from timing_model import CollectiveType


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


def _node_color(node: Any) -> str:
    if hasattr(node, "comm_size_bytes"):
        if getattr(node, "is_dp", False):
            return "green"
        if getattr(node, "comm_type", None) == CollectiveType.PIPELINE:
            return "white"
        return "yellow"
    if hasattr(node, "hw_id"):
        return "lightblue" if getattr(node, "fwd", True) else "lightcoral"
    return "mediumorchid"


def _node_label(node: Any) -> str:
    if hasattr(node, "comm_size_bytes"):
        duration_display = _format_duration(node.duration)
        if getattr(node, "local_hw_id", None) is not None:
            return f"{node.name}\n(local_hw_id={node.local_hw_id}, dur={duration_display})"
        return f"{node.name}\n(dur={duration_display})"
    if hasattr(node, "hw_id"):
        # ComputeEvent ports the legacy Node duration property pair
        # (scalar ``duration`` + optional per-DP ``duration_profile``).
        profile = getattr(node, "duration_profile", None)
        duration = node.duration
        if profile is None and isinstance(duration, (tuple, list)):
            profile = tuple(duration)
        value = profile[0] if profile else duration
        return f"{node.name}\n(hw_id={node.hw_id}, dur={_format_duration(value, profile)})"
    return str(node)


def visualize_events(roots: Any) -> Digraph:
    """Render an event DAG (coarse schedule or fine proto) to a Digraph."""

    dot = Digraph(comment="Computation Graph", format="svg")
    visited = set()

    def _visit(node: Any) -> None:
        if node in visited:
            return
        visited.add(node)
        node_id = str(id(node))
        dot.node(node_id, label=_node_label(node), style="filled", fillcolor=_node_color(node), shape="box")
        for child in getattr(node, "children", []):
            child_id = str(id(child))
            dot.edge(node_id, child_id)
            _visit(child)

    iterable = roots if isinstance(roots, (list, tuple, set)) else [roots]
    for root in iterable:
        _visit(root)
    return dot


def save_events_graph(roots: Any, output_folder: str = "output/LLM/", filename: str = "graph") -> None:
    """Render + write ``<output_folder><filename>.svg`` (legacy
    ``Graph.save_graph`` contract, async via ``util.graphviz_submit``)."""
    os.makedirs(output_folder, exist_ok=True)

    base_path = os.path.normpath(f"{output_folder}{filename}")
    svg_path = f"{base_path}.svg"
    display_path = util.relpath_display(svg_path)
    printstr = f" | Graph saved to    {display_path}"

    def _render_graph() -> None:
        dot = visualize_events(roots)
        dot.render(base_path, format="svg", cleanup=True)

    util.graphviz_submit(f"{filename}.svg", _render_graph, print_message=printstr)
