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

"""``RankLayout`` — THE placement bridge (DESIGN.md §2, CONTEXT constraint 5).

Single implementation replacing the three legacy copies of the rank-layout /
hw-id linearization logic:

* ``LLMExecutionDispatcher._build_rank_layout_descriptor`` (llm_execution.py)
* ``PipelineGraphFlattener._configure_rank_layout`` / ``_hw_id_for_rank``
  (llm_execution.py)
* the ``_build_rank_layout`` / ``_hw_id_for_rank`` closures in
  ``MemoryEstimator.build_memory_data`` (memory_estimation.py)

The validation logic and every error message are a verbatim port of the legacy
code; the rank-layout *semantics* are frozen (DESIGN.md §7 — implementation
unifies, meaning frozen).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from program.axes import (  # noqa: F401 - re-exported for existing importers
    AXES,
    AXIS_BY_NAME,
    CANONICAL_AXES,
    CLUSTER_AXES,
    PIPELINE_AXES,
    REPLICA_AXES,
    AxisName,
    AxisRole,
    cluster_strides,
)


def cluster_coords(
    axis_order: Sequence[AxisName],
    tp_rank: int,
    stage_id: int,
    *,
    sizes: Mapping[AxisName, int],
) -> Dict[AxisName, int]:
    """Decompose a flat transformer-cluster rank + pipeline stage into coords.

    Verbatim port of the shared decomposition in
    ``PipelineGraphFlattener._hw_id_for_rank`` and memory estimation's
    ``_hw_id_for_rank``: ``tp_rank`` is split row-major into tp/cp/ep
    coordinates (tp fastest) and ``stage_id`` becomes the pp coordinate, with
    the legacy pp bounds check. Only axes present in ``axis_order`` receive a
    coordinate; ``sizes`` is supplied by the caller because the two legacy call
    sites use different fallbacks for axes absent from the layout.

    Both loops are driven by ``program.axes`` roles, so a new axis needs no
    edit here and none in this signature.
    """
    coords: Dict[AxisName, int] = {}
    # One rule for every CLUSTER axis, from program.axes: coordinate =
    # (flat_rank // stride) % size, stride = product of the earlier cluster
    # axes' sizes. The three hand-written lines this replaces were that formula
    # spelled out for tp/cp/ep specifically.
    for name, stride in cluster_strides(axis_order, sizes).items():
        coords[name] = (tp_rank // max(1, stride)) % max(1, int(sizes[name]))
    for name in PIPELINE_AXES:
        if name not in axis_order:
            continue
        extent = max(1, int(sizes.get(name, 1)))
        if stage_id < 0 or stage_id >= extent:
            raise ValueError(
                f"stage_id {stage_id} is out of range for {name}={extent}"
            )
        coords[name] = stage_id % extent
    return coords


@dataclass(frozen=True, slots=True)
class RankLayout:
    """A rank layout: ordered parallelism axes with sizes and strides.

    ``axis_order`` is a subset of :data:`CANONICAL_AXES` in canonical order.
    ``axis_strides`` is row-major over ``axis_order`` (first axis
    fastest-varying, i.e. stride 1). ``axis_sizes`` may contain axes beyond
    ``axis_order``: the legacy full descriptor always carries all five
    canonical axis sizes while ``axis_order`` lists only the axes present in
    the network layout — :meth:`descriptor` must reproduce that dict shape
    exactly.
    """

    axis_order: Tuple[AxisName, ...]
    axis_sizes: Dict[AxisName, int]
    axis_strides: Dict[AxisName, int]

    def num_ranks(self) -> int:
        """Product of the sizes of the axes in ``axis_order`` (>= 1)."""
        span = 1
        for axis in self.axis_order:
            span *= self.axis_sizes.get(axis, 1)
        return span

    def linearize(self, coords: Mapping[AxisName, int]) -> int:
        """Map per-axis coordinates to a flat rank id.

        Missing axes default to coordinate 0. Verbatim port of the legacy
        linearization loops (including error messages) in
        ``PipelineGraphFlattener._hw_id_for_rank`` and memory estimation's
        ``_hw_id_for_rank``.
        """
        linear_rank = 0
        for axis in self.axis_order:
            coord = coords.get(axis, 0)
            size = self.axis_sizes.get(axis, 1)
            if coord < 0 or coord >= size:
                raise ValueError(f"Coordinate {coord} for axis '{axis}' is out of range <{size}")
            stride = self.axis_strides.get(axis)
            if stride is None:
                raise KeyError(f"Rank layout stride missing for axis '{axis}'")
            linear_rank += coord * stride
        return linear_rank

    def coords_of(self, rank: int) -> Dict[AxisName, int]:
        """Inverse of :meth:`linearize` for ranks in ``[0, num_ranks())``."""
        rank_int = int(rank)
        if rank_int < 0 or rank_int >= self.num_ranks():
            raise ValueError(f"Rank {rank_int} is out of range for layout with {self.num_ranks()} ranks")
        coords: Dict[AxisName, int] = {}
        for axis in self.axis_order:
            size = self.axis_sizes.get(axis, 1)
            stride = self.axis_strides.get(axis)
            if stride is None:
                raise KeyError(f"Rank layout stride missing for axis '{axis}'")
            coords[axis] = (rank_int // stride) % size
        return coords

    def subset(self, axes: Sequence[AxisName]) -> "RankLayout":
        """Restrict the layout to ``axes``, recomputing row-major strides.

        Verbatim port of ``_subset_layout`` in
        ``LLMExecutionDispatcher._build_rank_layout_descriptor``: the subset
        keeps the relative order of ``axis_order``, its sizes dict contains
        only the subset axes, and strides restart from 1. The legacy helper
        returned ``None`` for an empty subset; here an empty-``axis_order``
        layout is returned instead and callers translate.
        """
        selected = [axis for axis in self.axis_order if axis in axes and self.axis_sizes.get(axis, 1) >= 1]
        strides: Dict[AxisName, int] = {}
        span = 1
        for axis in selected:
            strides[axis] = span
            span *= self.axis_sizes[axis]
        return RankLayout(
            axis_order=tuple(selected),
            axis_sizes={axis: self.axis_sizes[axis] for axis in selected},
            axis_strides=strides,
        )

    def descriptor(self) -> Dict[str, Any]:
        """Legacy descriptor dict: ``{"axis_order","axis_sizes","axis_strides","stage_span"}``.

        ``axis_order`` is a list, ``axis_sizes``/``axis_strides`` are plain
        dicts preserving insertion order, and ``stage_span`` is the product of
        the ``axis_order`` sizes — exactly the shape built by the legacy
        ``_build_rank_layout_descriptor`` / ``_subset_layout``. This dict is
        the compatibility surface consumed by config_generation / faults /
        gmap; do not change its shape.
        """
        return {
            "axis_order": list(self.axis_order),
            "axis_sizes": dict(self.axis_sizes),
            "axis_strides": dict(self.axis_strides),
            "stage_span": self.num_ranks(),
        }

    @classmethod
    def from_network_layout(
        cls,
        network_layout: Any,
        axis_sizes: Mapping[AxisName, int],
        execution_mode_enforce_cluster: bool,
        *,
        extract_optimize_2dmap: bool = True,
        unsupported_axis_context: str = "AstraSim integration",
        empty_axis_order_is_none: bool = False,
    ) -> Tuple[Optional["RankLayout"], Optional[Dict[str, Any]]]:
        """Build a ``RankLayout`` from a hardware network layout.

        Verbatim port of the validation in
        ``LLMExecutionDispatcher._build_rank_layout_descriptor``
        (llm_execution.py), including exact error messages and the
        ``optimize_2dmap`` config extraction. ``axis_sizes`` must map every
        supported axis name to its (already clamped, >= 1) size; the caller
        computes it because the two legacy call sites source the sizes
        differently.

        Keyword-only flags reproduce the small observable differences of the
        memory-estimation copy (``MemoryEstimator.build_memory_data``):

        * ``extract_optimize_2dmap=False`` skips the optimize_2dmap scan
          (memory estimation never performed it);
        * ``unsupported_axis_context`` is interpolated into the unsupported
          axis message ("AstraSim integration" vs "memory estimation");
        * ``empty_axis_order_is_none=True`` returns ``(None, cfg)`` when no
          axis appears in the network layout, *before* the active-axis
          checks (memory estimation's early ``return None``).

        Returns ``(layout, optimize_cfg)`` where ``optimize_cfg`` is the
        legacy first-dimension optimize_2dmap dict or ``None``.
        """
        dimensions = getattr(network_layout, "dimensions", None) if network_layout is not None else None
        if not dimensions:
            raise ValueError("Network layout with at least one dimension is required to build a RankLayout.")

        optimize_cfg: Optional[Dict[str, Any]] = None
        if extract_optimize_2dmap:
            for idx, dim in enumerate(dimensions):
                if getattr(dim, "optimize_2dmap", False):
                    if idx != 0:
                        raise ValueError("optimize_2dmap is only supported on the first network dimension.")
                    if optimize_cfg is not None:
                        raise ValueError("Multiple network dimensions requested optimize_2dmap; only one is supported.")
                    topo_type = getattr(dim, "topology_type", None)
                    if not topo_type:
                        raise ValueError("optimize_2dmap requires a topology type on the target dimension.")
                    size_value = getattr(dim, "size", None)
                    if size_value is None:
                        raise ValueError("optimize_2dmap requires an explicit dimension size.")
                    dims_value = getattr(dim, "size_2d", None)
                    if dims_value is not None:
                        dims_value = (int(dims_value[0]), int(dims_value[1]))
                    optimize_cfg = {
                        "dimension_index": idx,
                        "topology": str(topo_type),
                        "size": int(size_value),
                        "parallelisms": tuple(getattr(dim, "parallelisms", ()) or ()),
                    }
                    if dims_value:
                        optimize_cfg["dims"] = dims_value

        # Enforce axis ordering for hierarchical/hybrid modes: the active
        # {'tp','cp','ep'} axes must occupy the leading active network
        # dimensions (one dim, or split across consecutive leading dims — e.g.
        # dim0=[tp,cp] NVLink + dim1=[ep] inter-node), before any dimension
        # carrying active 'pp'/'dp'. This matches the assumptions in the
        # hierarchical graphs where a stage is a TP/CP/EP cluster replicated
        # across PP (and potentially DP) axes; sub-graph simulations can only
        # include whole network dimensions, never a slice of one.
        if execution_mode_enforce_cluster:
            cluster_axes = sorted(axis for axis in CLUSTER_AXES if axis_sizes[axis] > 1)
            covered: List[str] = []
            for dim in dimensions:
                if sorted(set(covered)) == cluster_axes:
                    break
                if int(getattr(dim, "size", 1)) <= 1:
                    continue
                dim_axes_l = [str(axis).strip().lower() for axis in getattr(dim, "parallelisms", ())]
                cluster_here = [
                    axis for axis in dim_axes_l if axis in CLUSTER_AXES and axis_sizes[axis] > 1
                ]
                sched_here = [
                    axis for axis in dim_axes_l if axis in PIPELINE_AXES + REPLICA_AXES and axis_sizes[axis] > 1
                ]
                if sched_here:
                    raise ValueError(
                        "For hierarchical/hybrid AstraSim modes, the active TP/CP/EP axes must "
                        "occupy the leading active network dimensions (before any dimension "
                        "carrying active PP/DP) to represent the transformer cluster. "
                        f"Dimension '{getattr(dim, 'label', getattr(dim, 'id', '<unnamed>'))}' carries "
                        f"{sched_here} while the cluster axes {cluster_axes} are not yet fully mapped "
                        f"(covered so far: {sorted(set(covered))})."
                    )
                covered.extend(cluster_here)
            if cluster_axes and sorted(set(covered)) != cluster_axes:
                raise ValueError(
                    "For hierarchical/hybrid AstraSim modes, the leading active network dimensions "
                    f"must jointly carry the active TP/CP/EP axes {cluster_axes} "
                    f"(found only {sorted(set(covered))})."
                )

        axis_order: List[str] = []
        for dim in dimensions:
            dim_axes = [str(axis).strip().lower() for axis in getattr(dim, "parallelisms", ())]
            declared = int(getattr(dim, "size", 1))

            for name in dim_axes:
                if name not in axis_sizes:
                    raise ValueError(
                        f"Unsupported parallelism axis '{name}' in network layout. "
                        f"Supported axes for {unsupported_axis_context} are: tp, cp, ep, pp, dp."
                    )
                if name not in axis_order:
                    axis_order.append(name)

            expected = 1
            for axis_name in dim_axes:
                expected *= axis_sizes.get(axis_name, 1)
            if expected != declared:
                raise ValueError(
                    f"Network dimension '{getattr(dim, 'label', getattr(dim, 'id', '<unnamed>'))}' "
                    f"size mismatch: declared {declared}, but parallelism factors imply {expected}."
                )

        if axis_order:
            ordered_axes = list(CANONICAL_AXES)
            axis_order = [axis for axis in ordered_axes if axis in axis_order]

        if empty_axis_order_is_none and not axis_order:
            return None, optimize_cfg

        # Ensure the layout covers active parallel axes
        # One check per CLUSTER axis, driven by program.axes rather than three
        # near-identical hand-written blocks.
        for _name in CLUSTER_AXES + PIPELINE_AXES:
            if int(axis_sizes.get(_name, 1)) > 1 and _name not in axis_order:
                raise ValueError(
                    f"Network layout must include {_name!r} when "
                    f"{AXIS_BY_NAME[_name].label} > 1."
                )


        axis_strides: Dict[str, int] = {}
        span = 1
        for axis in axis_order:
            axis_strides[axis] = span
            span *= axis_sizes[axis]

        layout = cls(
            axis_order=tuple(axis_order),
            axis_sizes=dict(axis_sizes),
            axis_strides=axis_strides,
        )
        return layout, optimize_cfg
