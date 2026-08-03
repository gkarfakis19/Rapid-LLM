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

"""The execution dispatcher: ONE construction path for every mode (P5/P6).

``program.build.build()`` is the only Program constructor. The four execution
modes, the memory replay and the BLOCK measurements differ ONLY in the
:class:`~program.placement.Granularity` they ask for and in what they do with
the result:

======================  ===========  ===========================================
mode                    granularity  consumer
======================  ===========  ===========================================
ANALYTICAL              PIPELINE       ``program.analytic_sim.evaluate``
HYBRID                  PIPELINE       BLOCK retime, then ``analytic_sim.evaluate``
FULL_ASTRASIM_HIER..    PIPELINE       BLOCK retime, then AstraSim over (pp,dp)
FULL_ASTRASIM_FLAT..    FLAT         AstraSim over the full layout
transformer blocks      BLOCK        AstraSim, one bundle per direction
memory                  FLAT         ``program.memory_sim.simulate_memory``
======================  ===========  ===========================================

The dispatcher takes a **typed** :class:`~program.workload.WorkloadSpec` (L0),
built by ``WorkloadSpec.from_timing`` — the single home of INTERFACES §1.7's
producer map. ``ScheduleInputs``, ``ScheduleSpec.from_pipeline_graph`` and
``TransformerBlockSpec`` are gone, and so is every per-caller re-derivation of
``include_backward`` / ``include_optimizer`` / the effective dp: those live on
:class:`~program.workload.RunPolicy`, together with the interleave scale
(Class B item 8).
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, TYPE_CHECKING

from astrasim_lib import run_astra_simulation_only_onepath
from astrasim_lib.fault_projection import FaultProjectionResult, FaultSpace
from astrasim_lib.layout_utils import axis_layout_from_descriptor
from program import _env_flag
from program.ir import Program
from program.layout import RankLayout
from program.placement import Granularity
from program.work import Direction
from program.workload import WorkloadSpec
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
    total_time: float


@dataclass
class TransformerTimings:
    forward: float
    backward: float


class LLMExecutionDispatcher:
    def __init__(
        self,
        time_calc: "TimeCalculationLLM",
        workload: WorkloadSpec,
    ) -> None:
        if not isinstance(workload, WorkloadSpec):
            raise TypeError(
                "LLMExecutionDispatcher requires a typed program.workload.WorkloadSpec "
                f"(got {type(workload).__name__}); build it with WorkloadSpec.from_timing"
            )
        self.time_calc = time_calc
        #: True for the grad-accumulation non-final cycle (the graph built
        #: without the optimizer tail and with the skippable dp comms dropped).
        #: DERIVED from the workload's own grad-accum cycle — it used to be a
        #: separate constructor flag that every caller had to keep in sync.
        self.no_data_parallel = bool(workload.run.is_nonfinal_grad_accum_cycle)

        self._network_dimensions: Tuple[Any, ...] = tuple()
        self._transformer_rank_layout: Optional[Dict[str, Any]] = None
        self._pipeline_rank_layout: Optional[Dict[str, Any]] = None
        self._axis_dimension_map: Dict[str, int] = {}
        self._first_dim_optimize_cfg: Optional[Dict[str, Any]] = None
        self._rank_layout = self._build_rank_layout_descriptor(workload)
        #: the workload with the hardware layout attached (the dispatcher owns
        #: the network layout; ``from_timing`` leaves ``layout=None``).
        self.workload = workload.with_(layout=_layout_of(self._rank_layout))
        self.interconnect_params: Dict[str, Tuple[float, float]] = dict(
            self.workload.interconnect
        )

        #: FLAT Program built by the flattened execution path (reused by the
        #: memory path) and the memory path's own cached build.
        self.flat_program: Optional[Program] = None
        self._memory_flat_program: Optional[Program] = None
        #: PIPELINE Program evaluated/emitted by the other three modes.
        self.pipeline_program: Optional[Program] = None
        #: True once a mode ran the analytical comm-size conversion: the memory
        #: replay then times the dp/ZeRO collectives too (memory_sim docstring).
        self._comm_sizes_converted = False

        self._transformer_stage_dp_faults: Dict[Tuple[int, int], Tuple[Tuple[int, int, float], ...]] = {}
        self._transformer_stage_timings: Dict[Tuple[int, int], TransformerTimings] = {}
        self._transformer_stage_moe_timings: Dict[Tuple[int, int], TransformerTimings] = {}
        self._transformer_baseline_timings: Optional[TransformerTimings] = None
        self._transformer_moe_baseline_timings: Optional[TransformerTimings] = None
        self._fault_space: Optional[FaultSpace] = None
        self._fault_projections: Dict[str, FaultProjectionResult] = {}
        self._initialize_fault_mappings()

    # ==================================================================
    # layout / faults  (unchanged surface)
    # ==================================================================
    def _build_rank_layout_descriptor(self, workload: WorkloadSpec) -> Dict[str, Any]:
        hw_config = getattr(self.time_calc, "hw_config", None)
        layout = getattr(hw_config, "network_layout", None)
        dimensions = getattr(layout, "dimensions", None) if layout is not None else None
        if not dimensions:
            return {}
        self._network_dimensions = tuple(dimensions)

        degrees = workload.degrees
        axis_sizes: Dict[str, int] = {
            "tp": degrees.tp,
            "cp": degrees.cp,
            # the HARDWARE ep axis (``time_calc.ep``), which is the graph ep
            # only when MoE is active; Placement cross-checks the product.
            "ep": max(1, int(getattr(self.time_calc, "ep", 1) or 1)),
            "pp": degrees.pp,
            "dp": max(1, int(getattr(self.time_calc, "dp", 1) or 1)),
        }

        enforce_layout = self.time_calc.execution_mode in {
            ExecutionMode.HYBRID,
            ExecutionMode.FULL_ASTRASIM_HIERARCHICAL,
        }
        rank_layout, optimize_cfg = RankLayout.from_network_layout(
            layout, axis_sizes, enforce_layout
        )
        self._first_dim_optimize_cfg = optimize_cfg

        def _subset_descriptor(allowed: Sequence[str]) -> Optional[Dict[str, Any]]:
            subset = rank_layout.subset(allowed)
            if not subset.axis_order:
                return None
            return subset.descriptor()

        # Transformer graphs only encode TP/CP/EP axes; pipeline graphs encode PP
        # (DP replicas are handled externally).
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

    # ==================================================================
    # THE one construction path
    # ==================================================================
    def _build_program(
        self,
        granularity: Granularity,
        *,
        label: str,
        workload: Optional[WorkloadSpec] = None,
        gmap_workdir: Optional[str] = None,
        directions: Optional[Tuple[Direction, ...]] = None,
    ) -> Program:
        """``build()`` — the ONLY Program constructor (INTERFACES §4.2).

        Policy selection is :func:`program.policies.policies_for`; ``build()``
        makes no policy decision of its own, and this method makes none either.
        ``GroupRaceWarning`` is silenced because V6 is always-on in ``build()``
        and a gradient reducer is a graph SINK by construction — two reducers of
        one group on one device therefore trip a warning whose race cannot
        happen under CONTEXT constraint 2 (INTERFACES §4.8).
        """
        from program.build import build
        from program.policies import policies_for
        from program.schedule.gpipe import GPipeSchedule
        from program.validate import GroupRaceWarning

        spec = workload if workload is not None else self.workload
        fw = spec.freeze()
        bundle = policies_for(fw, granularity=granularity)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", GroupRaceWarning)
            return build(
                fw,
                granularity=granularity,
                sharding=bundle.sharding,
                schedule_policy=GPipeSchedule(),
                recompute=bundle.recompute,
                routing=bundle.routing,
                overlap=bundle.overlap,
                grad_accum=bundle.grad_accum,
                label=label,
                gmap_workdir=gmap_workdir,
                directions=directions,
                # Build-time validation (V1-V8, V6 races, V7 group membership)
                # is OPT-IN for production runs (sw_param.validate_graph /
                # RAPID_VALIDATE_GRAPH) — it re-proves invariants build() holds
                # by construction and costs ~a minute at GPT-1T scale. The
                # wire-level emission postconditions in et_emit (per-group
                # issue order, V9 no-lost-successors) stay ALWAYS-ON: they are
                # cheap and are the actual last line against silent AstraSim
                # deadlocks. Every deadlock-shaped runtime error names the
                # switch. build()'s own default stays True, so unit tests and
                # direct library callers keep full validation.
                validate=bool(getattr(self.time_calc, "validate_graph", True)),
            )

    def _emission_program(
        self,
        granularity: Granularity,
        *,
        label: str,
        artifact_dir: str,
        workload: Optional[WorkloadSpec] = None,
        directions: Optional[Tuple[Direction, ...]] = None,
    ) -> Program:
        """Build + apply the first-dimension SCOTCH remap (P5: the remap is a
        post-build pass over the ops — see :mod:`program.mapping`)."""
        from program.mapping import apply_first_dim_mapping

        optimize_cfg = dict(self._first_dim_optimize_cfg) if self._first_dim_optimize_cfg else None
        gmap_workdir = (
            artifact_dir if (optimize_cfg and self.time_calc.persist_astrasim_artifacts) else None
        )
        program = self._build_program(
            granularity,
            label=label,
            workload=workload,
            gmap_workdir=gmap_workdir,
            directions=directions,
        )
        return apply_first_dim_mapping(
            program, optimize_2dmap=optimize_cfg, workdir=gmap_workdir
        )

    def _interleave_scale(self) -> float:
        """Class B item 8, now a :class:`~program.workload.RunPolicy` method."""
        return self.workload.run.interleave_scale(
            self.workload.degrees, self.workload.shape
        )

    def _pipeline_label(self) -> str:
        return "pipeline_no_dp" if self.no_data_parallel else "pipeline"

    # ==================================================================
    # modes
    # ==================================================================
    def run(self, mode: ExecutionMode) -> ExecutionResult:
        if mode == ExecutionMode.ANALYTICAL:
            return self._run_pipeline_with_analytical_comm(ExecutionMode.ANALYTICAL)
        if mode == ExecutionMode.HYBRID:
            return self._run_hybrid()
        if mode == ExecutionMode.FULL_ASTRASIM_HIERARCHICAL:
            return self._run_full_astrasim_hierarchical()
        if mode == ExecutionMode.FULL_ASTRASIM_FLATTENED:
            return self._run_full_astrasim_flattened()
        raise ValueError(f"Unknown execution mode {mode!r}")

    def _collect_block_timings(
        self,
        timings: Optional[TransformerTimings],
        moe_timings: Optional[TransformerTimings],
    ) -> Any:
        """Bundle the AstraSim block timings for :mod:`program.retime`."""
        from program.retime import BlockTimings

        return BlockTimings(
            dense=timings or self._transformer_baseline_timings,
            moe=moe_timings or self._transformer_moe_baseline_timings,
            stage_dense=dict(self._transformer_stage_timings),
            stage_moe=dict(self._transformer_stage_moe_timings),
        )

    def _run_pipeline_with_analytical_comm(
        self,
        declared_mode: ExecutionMode,
        pipeline_program: Optional[Program] = None,
    ) -> ExecutionResult:
        """Analytical evaluation of the PIPELINE Program."""
        from program import analytic_sim

        if declared_mode == ExecutionMode.HYBRID:
            filename = "/hybrid_graph_no_dp" if self.no_data_parallel else "/hybrid_graph"
        else:  # ANALYTICAL
            filename = "/analytical_graph_no_dp" if self.no_data_parallel else "/analytical_graph"

        if pipeline_program is None:
            pipeline_program = self._build_program(
                Granularity.PIPELINE, label=self._pipeline_label()
            )
        self.pipeline_program = pipeline_program

        total_time = analytic_sim.evaluate(
            pipeline_program,
            self.time_calc.network_model,
            self.interconnect_params,
        )
        #: the memory replay times the dp/ZeRO collectives once a mode has run
        #: the conversion (legacy: the flatten inherited the converted durations).
        self._comm_sizes_converted = True

        if _env_flag("RAPID_VISUALIZE_GRAPHS"):
            from program.viz import save_program_graph

            save_program_graph(pipeline_program, self.time_calc.output_dir, filename)

        total_time *= self._interleave_scale()
        return ExecutionResult(total_time=total_time)

    def _run_hybrid(self) -> ExecutionResult:
        from program.retime import apply_block_timings

        transformer_time, moe_transformer_time = self._run_transformer_astrasim()

        # Build from the PRISTINE analytical durations (the coarse program
        # predates the write-back), then retime the layer ops.
        pipeline_program = self._build_program(
            Granularity.PIPELINE, label=self._pipeline_label()
        )
        if transformer_time is not None or moe_transformer_time is not None:
            self._write_block_durations(transformer_time, moe_transformer_time)
            apply_block_timings(
                pipeline_program,
                self._collect_block_timings(transformer_time, moe_transformer_time),
                self.workload.run.retime_dp_count(self.workload.degrees),
            )
        return self._run_pipeline_with_analytical_comm(
            ExecutionMode.HYBRID, pipeline_program=pipeline_program
        )

    def _run_full_astrasim_hierarchical(self) -> ExecutionResult:
        """Hierarchical pipeline phase: the PIPELINE Program IS the emitted one.

        There is no lowering step any more — ``build()`` at PIPELINE produces the
        program the emitter consumes, over the ("pp","dp") sublayout that
        ``Placement`` derives for that granularity.
        """
        from program.retime import apply_block_timings

        transformer_time, moe_transformer_time = self._run_transformer_astrasim()

        artifact_dir = self.time_calc.output_dir
        if self.time_calc.persist_astrasim_artifacts:
            artifact_dir = os.path.join(self.time_calc.output_dir, "astra_hier")

        pipeline_program = self._emission_program(
            Granularity.PIPELINE, label=self._pipeline_label(), artifact_dir=artifact_dir
        )
        if transformer_time is not None or moe_transformer_time is not None:
            self._write_block_durations(transformer_time, moe_transformer_time)
            apply_block_timings(
                pipeline_program,
                self._collect_block_timings(transformer_time, moe_transformer_time),
                self.workload.run.retime_dp_count(self.workload.degrees),
            )
        self.pipeline_program = pipeline_program

        if _env_flag("RAPID_VISUALIZE_GRAPHS"):
            from program.viz import save_program_graph

            filename = (
                "/pipeline_graph_hierarchical_no_dp"
                if self.no_data_parallel
                else "/pipeline_graph_hierarchical"
            )
            save_program_graph(pipeline_program, self.time_calc.output_dir, filename)

        per_rank_sec, max_sec = run_astra_simulation_only_onepath(
            pipeline_program,
            self.time_calc,
            artifact_dir,
            persist_artifacts=self.time_calc.persist_astrasim_artifacts,
            faulty_links_override=self._fault_override("pipeline"),
            rank_layout=self._pipeline_rank_layout or None,
        )
        if max_sec <= 0:
            raise RuntimeError("AstraSim pipeline execution returned non-positive duration Re-run with sw_param.validate_graph: true (or RAPID_VALIDATE_GRAPH=1) for build-time graph diagnostics.")
        max_sec *= self._interleave_scale()
        return ExecutionResult(total_time=max_sec)

    def _run_full_astrasim_flattened(self) -> ExecutionResult:
        """Flattened execution: the FLAT Program straight into AstraSim.

        MoE included: ``build()`` expands the MoE block template through the
        same :class:`~program.placement.BlockExpander` at every granularity, so
        the hot/cold routing joins, the per-device all-to-all and the
        ``residual_p2p`` cold->hot transfers are realized here exactly as they
        are in the BLOCK builder (INTERFACES §3.4; ``ext_moe_flat.md`` P8).
        """
        artifact_dir = self.time_calc.output_dir
        if self.time_calc.persist_astrasim_artifacts:
            artifact_dir = os.path.join(self.time_calc.output_dir, "astra_flat")

        program = self._emission_program(
            Granularity.FLAT,
            label="flat_no_dp" if self.no_data_parallel else "flat",
            artifact_dir=artifact_dir,
        )
        self.flat_program = program

        if _env_flag("RAPID_VISUALIZE_GRAPHS"):
            # The un-expanded pipeline schedule, under the legacy filename: the
            # PIPELINE build of the same workload (there is no proto graph to
            # render any more — the coarse Program *is* that schedule).
            from program.viz import save_program_graph

            filename = (
                "/pipeline_graph_pre_flatten_no_dp"
                if self.no_data_parallel
                else "/pipeline_graph_pre_flatten"
            )
            save_program_graph(
                self._build_program(Granularity.PIPELINE, label=self._pipeline_label()),
                self.time_calc.output_dir,
                filename,
            )

        per_rank_sec, max_sec = run_astra_simulation_only_onepath(
            program,
            self.time_calc,
            artifact_dir,
            persist_artifacts=self.time_calc.persist_astrasim_artifacts,
            rank_layout=self._rank_layout or None,
        )

        if not per_rank_sec:
            raise RuntimeError("AstraSim flattened execution returned no per-rank timings Re-run with sw_param.validate_graph: true (or RAPID_VALIDATE_GRAPH=1) for build-time graph diagnostics.")

        effective_dp = self.workload.run.effective_dp(self.workload.degrees)
        expected_rank_count = effective_dp * len(program.compute_devices())

        # Special case: if the expected rank count is 1 then 2 is fine, but we
        # prune the extra result — the AstraSim backend only supports > 1 ranks,
        # so the executor duplicates the single trace.
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
            raise RuntimeError("AstraSim flattened execution returned non-positive duration Re-run with sw_param.validate_graph: true (or RAPID_VALIDATE_GRAPH=1) for build-time graph diagnostics.")

        return ExecutionResult(total_time=max_sec * self._interleave_scale())

    # ==================================================================
    # memory
    # ==================================================================
    def build_flat_program_for_memory(self) -> Program:
        """Build (and cache) the FLAT Program the memory replay consumes.

        Reuses the flattened path's Program when it built one (same workload,
        same granularity). The comm-duration vector the replay needs is stamped
        here, because the dispatcher is the only thing that knows whether a mode
        ran the analytical conversion (:mod:`program.memory_sim`).
        """
        if self._memory_flat_program is not None:
            return self._memory_flat_program

        program = self.flat_program
        if program is None:
            program = self._build_program(
                Granularity.FLAT,
                label="flat_no_dp" if self.no_data_parallel else "flat",
            )
        self._stamp_memory_comm_durations(program)
        self._memory_flat_program = program
        return program

    def _stamp_memory_comm_durations(self, program: Program) -> None:
        from program.analytic_sim import COMM_DURATIONS_KEY, comm_durations

        if not self._comm_sizes_converted:
            program.meta.misc.pop(COMM_DURATIONS_KEY, None)
            return
        program.meta.misc[COMM_DURATIONS_KEY] = comm_durations(
            program,
            self.time_calc.network_model,
            self.interconnect_params,
            collective_keys=set(self.workload.comm),
        )

    # ==================================================================
    # BLOCK measurements
    # ==================================================================
    def _build_transformer_block_programs(
        self, *, moe: bool, label: str
    ) -> Tuple[Optional[Program], Optional[Program]]:
        """The (forward, backward) BLOCK Programs for one template.

        Two bundles, because AstraSim returns ONE makespan per bundle and the
        retiming needs a forward time AND a backward time — INTERFACES §4.2's
        "a caller measuring a single direction builds a single direction's
        workload" (``build(directions=...)``).
        """
        block_workload = self.workload.for_block(
            layout=_layout_of(self._transformer_rank_layout), moe=moe
        )
        forward = self._build_program(
            Granularity.BLOCK,
            label=f"{label}_forward",
            workload=block_workload,
            directions=(Direction.FORWARD,),
        )
        backward = None
        if self.workload.run.include_backward:
            backward = self._build_program(
                Granularity.BLOCK,
                label=f"{label}_backward",
                workload=block_workload,
                directions=(Direction.BACKWARD,),
            )
        return forward, backward

    def _run_transformer_astrasim(
        self,
    ) -> Tuple[Optional[TransformerTimings], Optional[TransformerTimings]]:
        blocks = self.workload.blocks
        has_moe = blocks.moe is not None

        persist = self.time_calc.persist_astrasim_artifacts
        os.makedirs(self.time_calc.output_dir, exist_ok=True)
        self._transformer_stage_timings = {}
        self._transformer_stage_moe_timings = {}
        self._transformer_baseline_timings = None
        self._transformer_moe_baseline_timings = None

        forward_program, backward_program = self._build_transformer_block_programs(
            moe=False, label="block_dense"
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

        # Per-stage fault runs (dense only): the fault links change the AstraSim
        # network configs, never the emitted bundle, so the same Programs are
        # re-emitted per variant.
        stage_dp_faults = self._transformer_stage_dp_faults
        for fault_index, ((dp_idx, stage_id), fault_links) in enumerate(sorted(stage_dp_faults.items())):
            variant = f"fault{fault_index}_dp{dp_idx}_stage{stage_id}"
            stage_fwd_dir, stage_bwd_dir = self._transformer_artifact_dirs(label=variant, persist=persist)
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
                moe=True, label="block_moe"
            )
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
                variant = f"moe_fault{fault_index}_dp{dp_idx}_stage{stage_id}"
                stage_fwd_dir, stage_bwd_dir = self._transformer_artifact_dirs(label=variant, persist=persist)
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
        forward_program: Optional[Program],
        backward_program: Optional[Program],
        faulty_links_override: Optional[Tuple[Tuple[int, int, float], ...]],
    ) -> Tuple[TransformerTimings, Optional[List[float]], Optional[List[float]]]:
        """Run the (forward, backward) BLOCK Programs through AstraSim."""
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
                rank_layout=self._transformer_rank_layout or None,
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
                rank_layout=self._transformer_rank_layout or None,
            )
            if bwd_max < 0:
                raise RuntimeError("AstraSim transformer backward execution returned non-positive duration")

        return TransformerTimings(forward=fwd_max, backward=bwd_max), fwd_per_rank, bwd_per_rank

    def _write_block_durations(
        self,
        timings: Optional[TransformerTimings],
        moe_timings: Optional[TransformerTimings] = None,
    ) -> None:
        """Write the AstraSim transformer baselines into the workload's
        :class:`~program.workload.DurationTable` — the one mutable member, whose
        ``revision`` is stamped onto every Program built after the write."""
        if timings is None and moe_timings is None:
            return
        baseline = timings or self._transformer_baseline_timings
        moe_baseline = moe_timings or self._transformer_moe_baseline_timings
        self.workload.durations.write_block_timings(
            dense_forward=None if baseline is None else baseline.forward,
            dense_backward=None if baseline is None else baseline.backward,
            moe_forward=None if moe_baseline is None else moe_baseline.forward,
            moe_backward=None if moe_baseline is None else moe_baseline.backward,
        )


def _layout_of(descriptor: Optional[Mapping[str, Any]]) -> Optional[RankLayout]:
    """Descriptor dict -> :class:`RankLayout` (``None`` when there is none)."""
    if not descriptor or not descriptor.get("axis_order"):
        return None
    return RankLayout(
        axis_order=tuple(descriptor["axis_order"]),
        axis_sizes=dict(descriptor["axis_sizes"]),
        axis_strides=dict(descriptor["axis_strides"]),
    )
