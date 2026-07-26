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

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, TYPE_CHECKING

from astrasim_lib import run_astra_simulation_only_onepath
from astrasim_lib.fault_projection import FaultProjectionResult, FaultSpace
from astrasim_lib.layout_utils import axis_layout_from_descriptor
from program import _env_flag
from program.block_program import TransformerBlockSpec
from program.layout import RankLayout
from program.schedule import ScheduleInputs
from util import log_message

if TYPE_CHECKING:
    from train_timing import TimeCalculationLLM




class ExecutionMode(Enum):
    ANALYTICAL = "analytical"
    HYBRID = "hybrid"
    FULL_ASTRASIM_HIERARCHICAL = "full_astrasim_hierarchical"
    FULL_ASTRASIM_FLATTENED = "full_astrasim_flattened"
    
    
@dataclass
class ExecutionResult:
    # M7/M8 note: the legacy ``graph_root``/``mode`` compat fields were
    # retired — no caller ever read them (callers consume ``total_time``
    # only; the executed Programs live on the dispatcher as
    # ``coarse_program``/``fine_program``).
    total_time: float


@dataclass
class TransformerTimings:
    forward: float
    backward: float
    
class LLMExecutionDispatcher:
    def __init__(
        self,
        time_calc: TimeCalculationLLM,  # somehow this works? TODO: fix this at some point so its not annotated by IDE.
        pipeline_graph: ScheduleInputs,
        interconnect_params: Dict[str, Tuple[float, float]],
        transformer_blocks: Optional[TransformerBlockSpec] = None,
        no_data_parallel: bool = False,
    ) -> None:
        self.time_calc = time_calc
        #: The pipeline schedule inputs (M7: ``program.schedule.
        #: ScheduleInputs``, the typed carrier that replaced the legacy
        #: ``simulate_train_graph.Graph`` object — same attribute surface:
        #: parallelism degrees + comp_times/comm_metadata/misc_metadata).
        self.pipeline_graph = pipeline_graph
        self.interconnect_params = interconnect_params
        #: BLOCK template bundle (M4): dense/MoE BlockTemplates + cluster
        #: degrees, replacing the legacy transformer Graph + fwd/bwd roots.
        self.transformer_blocks = transformer_blocks
        #: FINE Program built by the flattened execution path (reused by the
        #: memory path) and the memory path's own cached build (M3b).
        self.fine_program: Optional[Any] = None
        self._memory_fine_program: Optional[Any] = None
        #: COARSE Program evaluated by the analytical/hybrid modes (M5).
        self.coarse_program: Optional[Any] = None
        #: True once _run_pipeline_with_analytical_comm converted the coarse
        #: graph's comm sizes to times in place (analytical/hybrid modes) —
        #: the legacy memory flatten inherited those durations, so the FINE
        #: memory build replays the conversion on its coarse events.
        self._comm_sizes_converted = False
        self._transformer_rank_layout: Dict[str, Any] = {}
        self._pipeline_rank_layout: Dict[str, Any] = {}
        self._network_dimensions: Tuple[Any, ...] = tuple()
        self._axis_dimension_map: Dict[str, int] = {}
        self._transformer_stage_dp_faults: Dict[Tuple[int, int], Tuple[Tuple[int, int, float], ...]] = {}
        self._transformer_stage_timings: Dict[Tuple[int, int], TransformerTimings] = {}
        self._transformer_stage_moe_timings: Dict[Tuple[int, int], TransformerTimings] = {}
        self._transformer_baseline_timings: Optional[TransformerTimings] = None
        self._transformer_moe_baseline_timings: Optional[TransformerTimings] = None
        self.no_data_parallel = bool(no_data_parallel)
        self._rank_layout = self._build_rank_layout_descriptor()
        self._first_dim_optimize_cfg: Optional[Dict[str, Any]] = getattr(self, "_first_dim_optimize_cfg", None)
        self._fault_space: Optional[FaultSpace] = None
        self._fault_projections: Dict[str, FaultProjectionResult] = {}
        self._initialize_fault_mappings()

    def _build_rank_layout_descriptor(self) -> Dict[str, Any]:
        hw_config = getattr(self.time_calc, "hw_config", None)
        layout = getattr(hw_config, "network_layout", None)
        dimensions = getattr(layout, "dimensions", None) if layout is not None else None
        if not dimensions:
            return {}
        self._network_dimensions = tuple(dimensions)

        def _safe_int(value: Any, default: int = 1) -> int:
            try:
                candidate = int(value)
            except (TypeError, ValueError):
                candidate = default
            return max(1, candidate)

        tp_size = _safe_int(getattr(self.pipeline_graph, "tp", getattr(self.time_calc, "tp", 1)))
        cp_size = _safe_int(getattr(self.pipeline_graph, "cp", getattr(self.time_calc, "cp", 1)))
        ep_size = _safe_int(getattr(self.time_calc, "ep", 1))
        pp_size = _safe_int(getattr(self.pipeline_graph, "pp", getattr(self.time_calc, "pp", 1)))
        dp_size = _safe_int(getattr(self.time_calc, "dp", 1))

        axis_sizes: Dict[str, int] = {"tp": tp_size, "cp": cp_size, "ep": ep_size, "pp": pp_size, "dp": dp_size}

        enforce_layout = self.time_calc.execution_mode in {
            ExecutionMode.HYBRID,
            ExecutionMode.FULL_ASTRASIM_HIERARCHICAL,
        }
        rank_layout, optimize_cfg = RankLayout.from_network_layout(layout, axis_sizes, enforce_layout)
        self._first_dim_optimize_cfg = optimize_cfg

        def _subset_descriptor(allowed: Sequence[str]) -> Optional[Dict[str, Any]]:
            subset = rank_layout.subset(allowed)
            if not subset.axis_order:
                return None
            return subset.descriptor()

        # Transformer graphs only encode TP/CP axes; pipeline graphs encode PP
        # (DP replicas are handled externally). Store these subsets so callers
        # can attach the appropriate layout before invoking AstraSim.
        self._transformer_rank_layout = _subset_descriptor(["tp", "cp", "ep"])
        self._pipeline_rank_layout = _subset_descriptor(["pp", "dp"])

        return rank_layout.descriptor()

    def _log_fault_summary(self, axis_order: Sequence[str], axis_sizes: Mapping[str, int]) -> None:
        if not axis_order:
            return
        space = self._fault_space
        if space is None or not space.entries:
            return
        network_dims = getattr(self, "_network_dimensions", tuple())
        if not network_dims:
            log_message("[RAPID-LLM][faults] Hardware dimensions unavailable; skipping fault summary.")
            return

        def coords_to_dict(coords: Tuple[Tuple[str, int], ...]) -> Dict[str, int]:
            return {name: value for name, value in coords}

        entries = space.entries
        for dim_index, dim in enumerate(network_dims):
            axes = [str(axis).strip().lower() for axis in getattr(dim, "parallelisms", ())]
            if not axes:
                continue
            replication_axes: List[str] = []
            for future_dim in network_dims[dim_index + 1 :]:
                replication_axes.extend(
                    str(axis).strip().lower() for axis in getattr(future_dim, "parallelisms", ()) if axis
                )
            total_clusters = 1
            for axis in replication_axes:
                total_clusters *= max(1, axis_sizes.get(axis, 1))
            axes_label = ", ".join(axes) or "<none>"
            table_rows: List[Tuple[str, str, float]] = []
            for entry in entries:
                if not any(axis in axes for axis in entry.affected_axes):
                    continue
                src_dict = coords_to_dict(entry.src_coords)
                dst_dict = coords_to_dict(entry.dst_coords)
                src_label = f"{entry.original[0]} (" + ", ".join(f"{axis}={src_dict.get(axis, 0)}" for axis in axes) + ")"
                dst_label = f"{entry.original[1]} (" + ", ".join(f"{axis}={dst_dict.get(axis, 0)}" for axis in axes) + ")"
                higher_coords = " ".join(f"{axis}={src_dict.get(axis, 0)}" for axis in replication_axes)
                table_rows.append((higher_coords, f"{src_label} <-> {dst_label}", float(entry.original[2])))

            if not table_rows:
                log_message(f"  • dim{dim_index} ({axes_label}) → Affected HW clusters: 0 / {total_clusters}", category="faults")
                continue

            log_message(
                f"  • dim{dim_index} ({axes_label}) → Affected HW clusters: {len(table_rows)} / {total_clusters or 1}",
                category="faults",
            )
            header_axes = " ".join(replication_axes) if replication_axes else ""
            log_message(f"  {header_axes:<12} | {'source ↔ dest':<40} | derate", category="faults")
            log_message(f"  {'-' * max(len(header_axes),12)}-+-{'-' * 40}-+-------", category="faults")

            for higher_str, pair, derate in table_rows:
                log_message(
                    f"  {higher_str:<12} | {pair:<40} | {derate:>6.2f}",
                    category="faults",
                )
            log_message(f"  {'-' * max(len(header_axes),12)}-+-{'-' * 40}-+-------", category="faults")




    def _initialize_fault_mappings(self) -> None:
        hw_config = getattr(self.time_calc, "hw_config", None)
        network_layout = getattr(hw_config, "network_layout", None)
        faulty_links: Tuple[Tuple[int, int, float], ...] = tuple(
            getattr(network_layout, "faulty_links", ()) or ()
        )
        if faulty_links and self.time_calc.execution_mode in {ExecutionMode.ANALYTICAL, ExecutionMode.HYBRID}:
            raise ValueError("Faulty links require full AstraSim execution; analytical/hybrid modes are not supported.")
        axis_layout = axis_layout_from_descriptor(self._rank_layout)
        axis_dim_map = self._axis_to_dimension_map(network_layout)
        self._axis_dimension_map = dict(axis_dim_map)
        self._fault_space = FaultSpace(
            axis_layout,
            faulty_links,
            axis_to_dimension=axis_dim_map,
        )
        self._fault_projections = self._build_fault_projections_from_space(self._fault_space)
        self._validate_fault_coverage()
        self._transformer_stage_dp_faults = self._build_transformer_stage_dp_fault_map()
        axis_order = self._rank_layout.get("axis_order", []) if isinstance(self._rank_layout, dict) else []
        axis_sizes = self._rank_layout.get("axis_sizes", {}) if isinstance(self._rank_layout, dict) else {}
        self._log_fault_summary(axis_order, axis_sizes)

    def _axis_to_dimension_map(self, network_layout) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        if network_layout is None:
            return mapping
        dimensions = getattr(network_layout, "dimensions", None)
        if not dimensions:
            return mapping
        for idx, dim in enumerate(dimensions):
            for axis in getattr(dim, "parallelisms", ()) or ():
                normalized = str(axis).strip().lower()
                if normalized:
                    mapping[normalized] = idx
        return mapping

    def _build_fault_projections_from_space(
        self,
        space: FaultSpace,
    ) -> Dict[str, FaultProjectionResult]:
        projections: Dict[str, FaultProjectionResult] = {}
        if not space.entries:
            return projections

        global_axes = tuple(space.layout.axis_order)
        if global_axes:
            projections["global"] = space.project(global_axes)

        transformer_axes: Tuple[str, ...] = tuple(
            self._transformer_rank_layout.get("axis_order", ())
        ) if self._transformer_rank_layout else tuple()
        if transformer_axes:
            projections["transformer"] = space.project(transformer_axes)

        pipeline_axes: Tuple[str, ...] = tuple(
            self._pipeline_rank_layout.get("axis_order", ())
        ) if self._pipeline_rank_layout else tuple()
        if pipeline_axes:
            projections["pipeline"] = space.project(pipeline_axes)

        return projections

    def _validate_fault_coverage(self) -> None:
        if self._fault_space is None:
            return
        covered: Set[Tuple[int, int, float]] = set()
        for label in ("transformer", "pipeline"):
            proj = self._fault_projections.get(label)
            if proj:
                covered.update(proj.covered_originals)
        uncovered = [
            entry.original
            for entry in self._fault_space.entries
            if entry.original not in covered
        ]
        if uncovered:
            formatted = ", ".join(str(item) for item in uncovered)
            raise ValueError(
                "Faulty links do not map to any transformer or pipeline axis subset: "
                f"{formatted}"
            )

    def _axis_value_from_coords(self, coords: Tuple[Tuple[str, int], ...], axis: str) -> int:
        for name, value in coords:
            if name == axis:
                return int(value)
        layout = self._rank_layout or {}
        if isinstance(layout, dict):
            layout_axes = layout.get("axis_order", [])
            if axis not in layout_axes:
                return 0
            axis_sizes = layout.get("axis_sizes", {})
        else:
            axis_sizes = {}
        if axis_sizes.get(axis, 1) <= 1:
            return 0
        raise ValueError(f"Axis '{axis}' not present in coordinate tuple for faulty link.")

    def _build_transformer_stage_dp_fault_map(self) -> Dict[Tuple[int, int], Tuple[Tuple[int, int, float], ...]]:
        projection = self._fault_projection_for("transformer")
        if not projection:
            return {}
        stage_dp_faults: Dict[Tuple[int, int], List[Tuple[int, int, float]]] = {}
        for detail in projection.entries:
            src_stage = self._axis_value_from_coords(detail.src_coords, "pp")
            dst_stage = self._axis_value_from_coords(detail.dst_coords, "pp")
            if src_stage != dst_stage:
                raise ValueError(
                    "Transformer faulty link spans multiple pipeline stages; "
                    "hierarchical execution requires faults limited to a single stage."
                )
            src_dp = self._axis_value_from_coords(detail.src_coords, "dp")
            dst_dp = self._axis_value_from_coords(detail.dst_coords, "dp")
            affected_dp = {src_dp, dst_dp}
            for dp_idx in affected_dp:
                stage_dp_faults.setdefault((dp_idx, src_stage), []).append(detail.remapped)
        return {key: tuple(links) for key, links in stage_dp_faults.items()}

    def _fault_projection_for(self, label: str) -> Optional[FaultProjectionResult]:
        return self._fault_projections.get(label)

    def _fault_links_for(self, label: str) -> Tuple[Tuple[int, int, float], ...]:
        projection = self._fault_projection_for(label)
        if projection is None:
            return tuple()
        return projection.remapped_links

    def _fault_override(self, label: str) -> Optional[Tuple[Tuple[int, int, float], ...]]:
        if self._fault_space is None:
            return None
        return self._fault_links_for(label)

    def run(self, mode: ExecutionMode) -> ExecutionResult:
        if mode == ExecutionMode.ANALYTICAL:
            return self._run_pipeline_with_analytical_comm(ExecutionMode.ANALYTICAL)
        if mode == ExecutionMode.HYBRID:
            return self._run_hybrid()
        if mode == ExecutionMode.FULL_ASTRASIM_HIERARCHICAL:
            return self._run_full_astrasim_hierarchical()
        if mode == ExecutionMode.FULL_ASTRASIM_FLATTENED:
            return self._run_full_astrasim_flattened()

    def _pipeline_interleave_scale(self) -> float:
        """Analytical interleaved-1F1B (virtual pipeline) bubble correction.

        The pipeline graph is built with a GPipe-style schedule whose span is
        (mb + pp - 1) uniform slots. Interleaving each rank's layers into v
        virtual stages shrinks the bubble to (pp - 1) / v slots, so the total
        scales by (mb + (pp - 1) / v) / (mb + pp - 1). This is exact under the
        simulator's own uniform-stage-time assumption; the DP grad-sync tail
        is scaled along with it, bounding the error by the (small) comm share.
        """
        v = int(getattr(self.time_calc, "pipeline_interleave", 1) or 1)
        pp = int(getattr(self.time_calc, "pp", 1) or 1)
        mb = int(getattr(self.time_calc, "mb", 1) or 1)
        if v <= 1 or pp <= 1:
            return 1.0
        return (mb + (pp - 1) / float(v)) / float(mb + pp - 1)

    def _run_type(self) -> str:
        return str(getattr(getattr(self.time_calc, "model", None), "run_type", "training")).lower()

    def _retime_dp_count(self) -> int:
        """Per-DP duration-profile length (legacy ``_apply_transformer_time``
        rule: inference forces 1, training uses the dp degree)."""
        if self._run_type() == "inference":
            return 1
        return max(1, getattr(self.time_calc, "dp", 1))

    def _build_coarse_program(self) -> Any:
        """Build the COARSE pipeline Program (M5) for analytical/hybrid.

        Reads the SAME inputs the legacy ``construct_fwd_bwd_graph`` call in
        ``_prepare_execution_graphs`` consumed: the pipeline graph's
        comp_times/comm_metadata/misc_metadata, ``include_backward`` from the
        run type and ``include_optimizer`` from the grad-accum cycle (the
        nonfinal no-DP graph is built without the optimizer tail).
        """
        from program.pipeline_coarse import build_coarse_program
        from program.schedule import ScheduleSpec

        run_type = self._run_type()
        include_backward = run_type != "inference"
        misc = getattr(self.pipeline_graph, "misc_metadata", None) or {}
        include_optimizer = str(misc.get("grad_accum_cycle", "final") or "final").lower() != "nonfinal"

        spec = ScheduleSpec.from_pipeline_graph(
            self.pipeline_graph,
            include_backward=include_backward,
            include_optimizer=include_optimizer,
        )

        layout_obj: Optional[RankLayout] = None
        pipeline_layout = getattr(self, "_pipeline_rank_layout", None)
        if pipeline_layout and pipeline_layout.get("axis_order"):
            layout_obj = RankLayout(
                axis_order=tuple(pipeline_layout.get("axis_order", [])),
                axis_sizes=dict(pipeline_layout.get("axis_sizes", {})),
                axis_strides=dict(pipeline_layout.get("axis_strides", {})),
            )

        effective_dp = 1 if run_type == "inference" else max(1, getattr(self.time_calc, "dp", 1))
        return build_coarse_program(
            spec,
            layout_obj,
            dp_count=effective_dp,
            label="coarse_no_dp" if self.no_data_parallel else "coarse",
        )

    def _collect_block_timings(
        self,
        timings: Optional[TransformerTimings],
        moe_timings: Optional[TransformerTimings],
    ) -> "Any":
        """Bundle the AstraSim block timings for ``program.retime`` (same
        baseline fallbacks as the legacy ``_apply_transformer_time``)."""
        from program.retime import BlockTimings

        return BlockTimings(
            dense=timings or self._transformer_baseline_timings,
            moe=moe_timings or self._transformer_moe_baseline_timings,
            stage_dense=dict(getattr(self, "_transformer_stage_timings", {})),
            stage_moe=dict(getattr(self, "_transformer_stage_moe_timings", {})),
        )

    def _run_pipeline_with_analytical_comm(
        self,
        declared_mode: ExecutionMode,
        coarse_program: Optional[Any] = None,
    ) -> ExecutionResult:
        """Analytical pipeline evaluation over the COARSE Program (M5).

        Replaces the legacy ``convert_comm_sizes_to_times`` +
        ``Graph.simulate`` pair: ``program.pipeline_coarse`` builds the typed
        coarse program from the same schedule events, and
        ``program.analytic_sim.evaluate`` replays the exact legacy
        conversion + list-scheduler discipline over it. The hybrid mode
        passes its retimed program in ``coarse_program``.
        """
        from program import analytic_sim

        if declared_mode == ExecutionMode.HYBRID:
            if self.no_data_parallel:
                filename = "/hybrid_graph_no_dp"
            else:
                filename = "/hybrid_graph"
        else: # must be "ANALYTICAL"
            if self.no_data_parallel:
                filename = "/analytical_graph_no_dp"
            else:
                filename = "/analytical_graph"

        if coarse_program is None:
            coarse_program = self._build_coarse_program()
        self.coarse_program = coarse_program

        total_time = analytic_sim.evaluate(
            coarse_program,
            self.time_calc.network_model,
            self.interconnect_params,
        )
        #: the memory-path FINE build replays the comm-size conversion on
        #: its own coarse events (legacy flatten inherited the durations).
        self._comm_sizes_converted = True

        if _env_flag("RAPID_VISUALIZE_GRAPHS"):
            # Render the coarse schedule events (converted comm durations +
            # retimed compute durations, like the legacy timed graph).
            from program.viz import save_events_graph

            save_events_graph(
                coarse_program.meta.misc["coarse_proto_root"],
                self.time_calc.output_dir,
                filename,
            )

        total_time *= self._pipeline_interleave_scale()
        return ExecutionResult(total_time=total_time)

    def _run_hybrid(self) -> ExecutionResult:
        from program.retime import apply_block_timings

        transformer_time, moe_transformer_time = self._run_transformer_astrasim()

        # Build from the pristine analytical comp_times (the legacy pipeline
        # graph predated the write-back), then retime the layer ops.
        coarse_program = self._build_coarse_program()
        if transformer_time is not None or moe_transformer_time is not None:
            self._update_comp_times_from_timings(transformer_time, moe_transformer_time)
            apply_block_timings(
                coarse_program,
                self._collect_block_timings(transformer_time, moe_transformer_time),
                self._retime_dp_count(),
            )
        return self._run_pipeline_with_analytical_comm(
            ExecutionMode.HYBRID, coarse_program=coarse_program
        )

    def _run_full_astrasim_hierarchical(self) -> ExecutionResult:
        """Hierarchical pipeline phase over the retimed COARSE Program (M6).

        Replaces the legacy pipeline-graph AstraSim path (attach
        ``_astrasim_rank_layout``/``_optimize_2dmap`` to ``pipeline_root``,
        retime via the ``_apply_transformer_time`` name-prefix walk, feed
        the legacy Node graph to the converter): the coarse Program is
        built from the schedule events, retimed with
        ``program.retime.apply_block_timings`` (same per-(dp, stage)
        fault-variant override semantics), and lowered for emission over
        the ("pp","dp") pipeline sublayout by
        ``program.pipeline_coarse.lower_coarse_for_emission`` — sharing the
        emission-order pass with the fine builder. Bundle labels/dirs
        (``astra_hier``), the pipeline fault override, the interleave scale
        and the non-positive-duration error are unchanged.
        """
        from program.pipeline_coarse import lower_coarse_for_emission
        from program.retime import apply_block_timings

        transformer_time, moe_transformer_time = self._run_transformer_astrasim()

        if not self.pipeline_graph:
            raise RuntimeError("Pipeline graph is not available for AstraSim execution")

        # Build from the pristine analytical comp_times (the legacy pipeline
        # graph predated the write-back too), then retime the layer ops.
        # Inference dp_override=1 lives in the builder: the coarse Program's
        # dp_count is the effective dp (_build_coarse_program).
        coarse_program = self._build_coarse_program()
        if transformer_time is not None or moe_transformer_time is not None:
            self._update_comp_times_from_timings(transformer_time, moe_transformer_time)
            apply_block_timings(
                coarse_program,
                self._collect_block_timings(transformer_time, moe_transformer_time),
                self._retime_dp_count(),
            )
        self.coarse_program = coarse_program

        # Use hierarchical artifact directory when persisting artifacts
        artifact_dir = self.time_calc.output_dir
        if self.time_calc.persist_astrasim_artifacts:
            artifact_dir = os.path.join(self.time_calc.output_dir, "astra_hier")

        if _env_flag("RAPID_VISUALIZE_GRAPHS"):
            # Render the coarse schedule events (retimed durations mirrored
            # by apply_block_timings, like the legacy retimed graph).
            from program.viz import save_events_graph

            filename = "/pipeline_graph_hierarchical_no_dp" if self.no_data_parallel else "/pipeline_graph_hierarchical"
            save_events_graph(
                coarse_program.meta.misc["coarse_proto_root"],
                self.time_calc.output_dir,
                filename,
            )

        optimize_cfg = dict(self._first_dim_optimize_cfg) if self._first_dim_optimize_cfg else None
        gmap_workdir = artifact_dir if (optimize_cfg and self.time_calc.persist_astrasim_artifacts) else None
        program = lower_coarse_for_emission(
            coarse_program,
            layout_descriptor=self._pipeline_rank_layout or None,
            optimize_2dmap=optimize_cfg,
            gmap_workdir=gmap_workdir,
        )

        per_rank_sec, max_sec = run_astra_simulation_only_onepath(
            program,
            self.time_calc,
            artifact_dir,
            persist_artifacts=self.time_calc.persist_astrasim_artifacts,
            faulty_links_override=self._fault_override("pipeline"),
        )
        if max_sec <= 0:
            raise RuntimeError("AstraSim pipeline execution returned non-positive duration")
        max_sec *= self._pipeline_interleave_scale()
        return ExecutionResult(total_time=max_sec)

    def _run_full_astrasim_flattened(self) -> ExecutionResult:
        """Flattened execution via ``program.pipeline_fine.build_fine_program``.

        Builds the flattened Program directly from the pipeline graph's
        ScheduleSpec + the transformer graph's BlockTemplate (no legacy
        flattener clone), applies the proto-level overlap transforms, and
        feeds the Program straight into ``run_astra_simulation_only_onepath``.
        The Program is cached on ``self.fine_program`` so the memory path
        (``build_fine_program_for_memory``) reuses it. Flattened *execution*
        keeps rejecting MoE (the memory path accepts it — DESIGN.md §4).
        """
        if self.transformer_blocks is not None and self.transformer_blocks.moe is not None:
            raise NotImplementedError("MoE is not supported with full AstraSim flattened execution.")
        if not self.pipeline_graph:
            raise RuntimeError("Pipeline schedule inputs are not available for flattening")
        if self.transformer_blocks is None:
            raise RuntimeError("Transformer graph metadata is required for flattening")

        from program.pipeline_fine import build_fine_program
        from program.schedule import ScheduleSpec

        run_type = str(getattr(getattr(self.time_calc, "model", None), "run_type", "training")).lower()
        effective_dp = 1 if run_type == "inference" else max(1, getattr(self.time_calc, "dp", 1))
        include_backward = run_type != "inference"
        # _prepare_execution_graphs builds the final-cycle graph with the
        # optimizer and the nonfinal (grad-accum no-DP) graph without it.
        misc = getattr(self.pipeline_graph, "misc_metadata", None) or {}
        include_optimizer = str(misc.get("grad_accum_cycle", "final") or "final").lower() != "nonfinal"

        spec = ScheduleSpec.from_pipeline_graph(
            self.pipeline_graph,
            include_backward=include_backward,
            include_optimizer=include_optimizer,
        )
        block_templates = {"dense": self.transformer_blocks.dense}
        if self.transformer_blocks.moe is not None:  # pragma: no cover - rejected above
            block_templates["moe"] = self.transformer_blocks.moe

        layout_obj: Optional[RankLayout] = None
        if self._rank_layout and self._rank_layout.get("axis_order"):
            layout_obj = RankLayout(
                axis_order=tuple(self._rank_layout.get("axis_order", [])),
                axis_sizes=dict(self._rank_layout.get("axis_sizes", {})),
                axis_strides=dict(self._rank_layout.get("axis_strides", {})),
            )

        if _env_flag("RAPID_VISUALIZE_GRAPHS"):
            # Render the un-expanded pipeline schedule (the coarse events —
            # add_child-order isomorphic to the retired legacy pre-flatten
            # pipeline root, same filename).
            from program.schedule import build_pipeline_events
            from program.viz import save_events_graph

            filename = "/pipeline_graph_pre_flatten_no_dp" if self.no_data_parallel else "/pipeline_graph_pre_flatten"
            save_events_graph(
                build_pipeline_events(spec).root,
                self.time_calc.output_dir,
                filename,
            )

        # Use flattened artifact directory when persisting artifacts
        artifact_dir = self.time_calc.output_dir
        if self.time_calc.persist_astrasim_artifacts:
            artifact_dir = os.path.join(self.time_calc.output_dir, "astra_flat")

        optimize_cfg = dict(self._first_dim_optimize_cfg) if self._first_dim_optimize_cfg else None
        gmap_workdir = artifact_dir if (optimize_cfg and self.time_calc.persist_astrasim_artifacts) else None

        program = build_fine_program(
            spec,
            block_templates,
            layout_obj,
            no_data_parallel=self.no_data_parallel,
            dp_count=effective_dp,
            optimize_2dmap=optimize_cfg,
            gmap_workdir=gmap_workdir,
            parallelism_mode=self.time_calc.get_parallelism_mode(),
            tp_overlap=getattr(self.time_calc, "tp_overlap", 0.0),
            tp_sp_overlap=getattr(self.time_calc, "tp_sp_overlap", 0.0),
            cp_overlap=getattr(self.time_calc, "cp_overlap", 0.0),
        )
        self.fine_program = program

        # Inference dp_override=1 semantics live in the builder: the FINE
        # Program's dp_count is the effective dp (M6 removed the executor's
        # dp_override plumbing along with the legacy-graph entry).
        per_rank_sec, max_sec = run_astra_simulation_only_onepath(
            program,
            self.time_calc,
            artifact_dir,
            persist_artifacts=self.time_calc.persist_astrasim_artifacts,
            rank_layout=self._rank_layout or None,
        )

        if not per_rank_sec:
            raise RuntimeError("AstraSim flattened execution returned no per-rank timings")

        expected_rank_count = effective_dp * len(program.compute_devices())

        # Special case: If expected rank count is 1, then 2 is fine, but we prune the extra result
        # this is done, since astrasim backend only supports >1 ranks, so we generate extra fake result for that case.
        if expected_rank_count == 1:
            if len(per_rank_sec) > 2:
                raise RuntimeError(
                    "AstraSim rank count mismatch for flattened execution: "
                    f"expected {expected_rank_count}, got {len(per_rank_sec)}"
                )
            per_rank_sec = per_rank_sec[:1]
        if len(per_rank_sec) != expected_rank_count:
            raise RuntimeError(
                "AstraSim rank count mismatch for flattened execution: "
                f"expected {expected_rank_count}, got {len(per_rank_sec)}"
            )

        if max_sec <= 0:
            raise RuntimeError("AstraSim flattened execution returned non-positive duration")

        return ExecutionResult(
            total_time=max_sec * self._pipeline_interleave_scale(),
        )

    def build_fine_program_for_memory(self) -> Any:
        """Build (and cache) the FINE Program the memory replay consumes.

        Replacement of ``build_flattened_root_for_memory`` (M3b): the fine
        program is built from this dispatcher's own coarse pipeline graph —
        with the MoE block template when the spec has MoE layers (the legacy
        flattener flattened MoE for memory even though flattened *execution*
        rejects it) — and cached per dispatcher, exactly like the legacy
        flattened root was. When the flattened execution path already built
        the fine program, that Program is reused verbatim.

        No SCOTCH/gmap hint is attached (the legacy memory flatten never
        carried ``_optimize_2dmap``). For the analytical/hybrid modes the
        coarse events replay ``convert_comm_sizes_to_times`` first, because
        the legacy flatten cloned the already-converted coarse edge
        durations into the memory graph.
        """
        if self._memory_fine_program is not None:
            return self._memory_fine_program

        if self.fine_program is not None:
            # Flattened execution already built the FINE program from the
            # same schedule/template/layout inputs; the proto root is
            # identical to a fresh build (gmap only affects Program stage
            # numbering, never the proto graph).
            self._memory_fine_program = self.fine_program
            return self._memory_fine_program

        if not self.pipeline_graph:
            raise RuntimeError("Pipeline schedule inputs are not available for memory flattening")
        if self.transformer_blocks is None:
            raise RuntimeError("Transformer graph metadata is required for memory flattening")

        from program.pipeline_fine import build_fine_program
        from program.schedule import ScheduleSpec

        run_type = str(getattr(getattr(self.time_calc, "model", None), "run_type", "training")).lower()
        effective_dp = 1 if run_type == "inference" else max(1, getattr(self.time_calc, "dp", 1))
        include_backward = run_type != "inference"
        misc = getattr(self.pipeline_graph, "misc_metadata", None) or {}
        include_optimizer = str(misc.get("grad_accum_cycle", "final") or "final").lower() != "nonfinal"

        spec = ScheduleSpec.from_pipeline_graph(
            self.pipeline_graph,
            include_backward=include_backward,
            include_optimizer=include_optimizer,
        )
        block_templates = {"dense": self.transformer_blocks.dense}
        if self.transformer_blocks.moe is not None:
            block_templates["moe"] = self.transformer_blocks.moe

        layout_obj: Optional[RankLayout] = None
        if self._rank_layout and self._rank_layout.get("axis_order"):
            layout_obj = RankLayout(
                axis_order=tuple(self._rank_layout.get("axis_order", [])),
                axis_sizes=dict(self._rank_layout.get("axis_sizes", {})),
                axis_strides=dict(self._rank_layout.get("axis_strides", {})),
            )

        events_hook = None
        if self._comm_sizes_converted:
            from program.analytic_sim import convert_comm_sizes_to_times

            def events_hook(events_root: Any) -> None:
                convert_comm_sizes_to_times(
                    events_root,
                    self.time_calc.network_model,
                    self.interconnect_params,
                )

        program = build_fine_program(
            spec,
            block_templates,
            layout_obj,
            no_data_parallel=self.no_data_parallel,
            dp_count=effective_dp,
            parallelism_mode=self.time_calc.get_parallelism_mode(),
            tp_overlap=getattr(self.time_calc, "tp_overlap", 0.0),
            tp_sp_overlap=getattr(self.time_calc, "tp_sp_overlap", 0.0),
            cp_overlap=getattr(self.time_calc, "cp_overlap", 0.0),
            events_hook=events_hook,
        )
        self._memory_fine_program = program
        return program

    def _build_transformer_block_programs(
        self,
        template: Any,
        *,
        include_backward: bool,
        label: str,
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """Build the (forward, backward) BLOCK Programs for one template.

        The overlap transforms are applied inside ``build_block_program`` at
        the same point in the flow as the legacy path (train_timing applied
        them to the freshly constructed transformer roots).
        """
        from program.block_program import build_block_program

        blocks = self.transformer_blocks
        layout = getattr(self, "_transformer_rank_layout", None)
        tc = self.time_calc
        common = dict(
            tp=blocks.tp,
            cp=blocks.cp,
            ep=blocks.ep,
            parallelism_mode=tc.get_parallelism_mode(),
            tp_overlap=getattr(tc, "tp_overlap", 0.0),
            tp_sp_overlap=getattr(tc, "tp_sp_overlap", 0.0),
            cp_overlap=getattr(tc, "cp_overlap", 0.0),
        )
        forward_program = build_block_program(
            template, "forward", layout, label=f"{label}_forward", **common
        )
        backward_program = None
        if include_backward:
            backward_program = build_block_program(
                template, "backward", layout, label=f"{label}_backward", **common
            )
        return forward_program, backward_program

    def _run_transformer_astrasim(
        self,
    ) -> Tuple[Optional[TransformerTimings], Optional[TransformerTimings]]:
        blocks = self.transformer_blocks
        has_dense = blocks is not None and blocks.dense is not None
        has_moe = blocks is not None and blocks.moe is not None
        if not has_dense and not has_moe:
            if getattr(self, "_transformer_stage_dp_faults", {}):
                raise ValueError("Transformer faults require transformer graph metadata, but none is available.")
            return None, None

        persist = self.time_calc.persist_astrasim_artifacts
        os.makedirs(self.time_calc.output_dir, exist_ok=True)
        self._transformer_stage_timings = {}
        self._transformer_stage_moe_timings = {}
        self._transformer_baseline_timings = None
        self._transformer_moe_baseline_timings = None

        baseline_timings: Optional[TransformerTimings] = None
        if has_dense:
            forward_program, backward_program = self._build_transformer_block_programs(
                blocks.dense, include_backward=blocks.include_backward, label="block_dense"
            )
            # Baseline run (no transformer faults)
            baseline_fwd_dir, baseline_bwd_dir = self._transformer_artifact_dirs(label=None, persist=persist)
            baseline_timings, baseline_fwd_per_rank, baseline_bwd_per_rank = self._execute_transformer_run(
                baseline_fwd_dir,
                baseline_bwd_dir,
                forward_program=forward_program,
                backward_program=backward_program,
                faulty_links_override=(),
            )
            self._transformer_baseline_timings = baseline_timings
            self.time_calc.transformer_astrasim_per_rank_forward = baseline_fwd_per_rank
            self.time_calc.transformer_astrasim_per_rank_backward = baseline_bwd_per_rank
            self.time_calc.transformer_astrasim_time_forward = baseline_timings.forward
            self.time_calc.transformer_astrasim_time_backward = baseline_timings.backward

            # Per-stage fault runs (dense only): the fault links change the
            # AstraSim network configs, never the emitted bundle, so the same
            # Programs are re-emitted per variant (legacy reused the roots).
            stage_dp_faults = getattr(self, "_transformer_stage_dp_faults", {})
            for fault_index, ((dp_idx, stage_id), fault_links) in enumerate(sorted(stage_dp_faults.items())):
                label = f"fault{fault_index}_dp{dp_idx}_stage{stage_id}"
                stage_fwd_dir, stage_bwd_dir = self._transformer_artifact_dirs(label=label, persist=persist)
                stage_timings, _, _ = self._execute_transformer_run(
                    stage_fwd_dir,
                    stage_bwd_dir,
                    forward_program=forward_program,
                    backward_program=backward_program,
                    faulty_links_override=fault_links,
                )
                self._transformer_stage_timings[(dp_idx, stage_id)] = stage_timings

        moe_timings: Optional[TransformerTimings] = None
        if has_moe:
            moe_forward_program, moe_backward_program = self._build_transformer_block_programs(
                blocks.moe, include_backward=blocks.include_backward, label="block_moe"
            )
            stage_dp_faults = getattr(self, "_transformer_stage_dp_faults", {})
            moe_fwd_dir, moe_bwd_dir = self._transformer_artifact_dirs(label="moe", persist=persist)
            moe_timings, moe_fwd_per_rank, moe_bwd_per_rank = self._execute_transformer_run(
                moe_fwd_dir,
                moe_bwd_dir,
                forward_program=moe_forward_program,
                backward_program=moe_backward_program,
                faulty_links_override=(),
            )
            self._transformer_moe_baseline_timings = moe_timings
            self.time_calc.transformer_astrasim_per_rank_forward_moe = moe_fwd_per_rank
            self.time_calc.transformer_astrasim_per_rank_backward_moe = moe_bwd_per_rank
            self.time_calc.transformer_astrasim_time_forward_moe = moe_timings.forward
            self.time_calc.transformer_astrasim_time_backward_moe = moe_timings.backward
            for fault_index, ((dp_idx, stage_id), fault_links) in enumerate(sorted(stage_dp_faults.items())):
                label = f"moe_fault{fault_index}_dp{dp_idx}_stage{stage_id}"
                stage_fwd_dir, stage_bwd_dir = self._transformer_artifact_dirs(label=label, persist=persist)
                stage_timings, _, _ = self._execute_transformer_run(
                    stage_fwd_dir,
                    stage_bwd_dir,
                    forward_program=moe_forward_program,
                    backward_program=moe_backward_program,
                    faulty_links_override=fault_links,
                )
                self._transformer_stage_moe_timings[(dp_idx, stage_id)] = stage_timings

        return baseline_timings, moe_timings

    def _transformer_artifact_dirs(self, label: Optional[str], persist: bool) -> Tuple[str, str]:
        if not persist:
            base_dir = self.time_calc.output_dir
            os.makedirs(base_dir, exist_ok=True)
            return base_dir, base_dir
        base_dir = os.path.join(self.time_calc.output_dir, "astra_hier")
        os.makedirs(base_dir, exist_ok=True)
        if label is None:
            fwd_dir = os.path.join(base_dir, "fwd")
            bwd_dir = os.path.join(base_dir, "bwd")
        else:
            fwd_dir = os.path.join(base_dir, f"{label}_fwd")
            bwd_dir = os.path.join(base_dir, f"{label}_bwd")
        os.makedirs(fwd_dir, exist_ok=True)
        os.makedirs(bwd_dir, exist_ok=True)
        return fwd_dir, bwd_dir

    def _execute_transformer_run(
        self,
        artifact_dir_fwd: str,
        artifact_dir_bwd: str,
        *,
        forward_program: Optional[Any],
        backward_program: Optional[Any],
        faulty_links_override: Optional[Tuple[Tuple[int, int, float], ...]],
    ) -> Tuple[TransformerTimings, Optional[List[float]], Optional[List[float]]]:
        """Run the (forward, backward) BLOCK Programs through AstraSim.

        The Program entry of ``run_astra_simulation_only_onepath`` reads
        ``Program.dp_count`` (1 for block programs — the legacy
        ``dp_override=1`` semantics live in the builder now).
        """
        fwd_per_rank = None
        bwd_per_rank = None
        fwd_max = 0
        bwd_max = 0

        if forward_program is not None:
            fwd_per_rank, fwd_max = run_astra_simulation_only_onepath(
                forward_program,
                self.time_calc,
                artifact_dir_fwd,
                persist_artifacts=self.time_calc.persist_astrasim_artifacts,
                faulty_links_override=faulty_links_override,
            )
            if fwd_max <= 0:
                raise RuntimeError("AstraSim transformer forward execution returned non-positive duration")

        if backward_program is not None:
            bwd_per_rank, bwd_max = run_astra_simulation_only_onepath(
                backward_program,
                self.time_calc,
                artifact_dir_bwd,
                persist_artifacts=self.time_calc.persist_astrasim_artifacts,
                faulty_links_override=faulty_links_override,
            )
            if bwd_max < 0:
                raise RuntimeError("AstraSim transformer backward execution returned non-positive duration")

        return TransformerTimings(forward=fwd_max, backward=bwd_max), fwd_per_rank, bwd_per_rank

    def _update_comp_times_from_timings(
        self,
        timings: Optional[TransformerTimings],
        moe_timings: Optional[TransformerTimings] = None,
    ) -> None:
        """Write the AstraSim transformer baselines into the pipeline graph's
        ``comp_times`` (shared hybrid/hierarchical half of the legacy
        ``_apply_transformer_time``; the memory-path FINE build reads these).
        """
        if timings is None and moe_timings is None:
            return
        if timings is not None and (timings.forward < 0 or timings.backward < 0):
            raise ValueError("AstraSim transformer times must be positive")
        if moe_timings is not None and (moe_timings.forward < 0 or moe_timings.backward < 0):
            raise ValueError("AstraSim transformer times must be positive")

        baseline_timings = timings or self._transformer_baseline_timings
        moe_baseline_timings = moe_timings or self._transformer_moe_baseline_timings

        comp_times = getattr(self.pipeline_graph, "comp_times", None)
        if isinstance(comp_times, dict):
            if baseline_timings:
                comp_times["transformer_f"] = baseline_timings.forward
                comp_times["transformer_b"] = baseline_timings.backward
                comp_times["transformer_f_dense"] = baseline_timings.forward
                comp_times["transformer_b_dense"] = baseline_timings.backward
            if moe_baseline_timings:
                comp_times["transformer_f_moe"] = moe_baseline_timings.forward
                comp_times["transformer_b_moe"] = moe_baseline_timings.backward

    # M6 note: the legacy-graph retime walk (_apply_transformer_time +
    # _assign_transformer_durations, the name-prefix recursion over legacy
    # pipeline Node graphs) is deleted — every mode retimes through
    # program.retime.apply_block_timings on the coarse Program now.
