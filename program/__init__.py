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

"""``program`` — the typed placed-operation IR core (docs/rewrite/DESIGN.md).

Post-migration (M8) module inventory:

* :mod:`program.ir` — the Program IR (``ComputeOp``/``CollectiveOp``/
  ``TransferOp``/``Program``);
* :mod:`program.layout` — the unified ``RankLayout`` (M0);
* :mod:`program.validate` — invariants V1-V6;
* :mod:`program.schedule` — ``ScheduleInputs`` (the pipeline input carrier,
  M7), ``ScheduleSpec`` and ``build_pipeline_events`` (the single GPipe
  schedule enumeration every builder consumes);
* :mod:`program.block` — ``BlockTemplate``/``CommMeta``;
* :mod:`program.block_program` — ``build_block_program`` (BLOCK Programs
  for the hybrid/hierarchical transformer AstraSim runs, M4);
* :mod:`program.pipeline_fine` — ``build_fine_program`` (the direct
  flattened builder, M3a; also feeds the memory replay);
* :mod:`program.pipeline_coarse` — ``build_coarse_program`` (COARSE
  pipeline Programs, M5) + ``lower_coarse_for_emission`` (the hierarchical
  pipeline emission entry, M6);
* :mod:`program.analytic_sim` — the analytical evaluator (M5);
* :mod:`program.retime` — ``apply_block_timings`` (per-DP retime
  write-back, M5/M6);
* :mod:`program.transforms` — TP/TP-SP/CP overlap passes;
* :mod:`program.memory_sim` — ``simulate_memory`` (peak-memory replay over
  FINE proto graphs, M3b);
* :mod:`program.legacy_lowering` — the emission-ordering pass (event DAG ->
  validated emission-ordered Program; permanent, see its docstring);
* :mod:`program.et_emit` — Program -> Chakra ET bundle;
* :mod:`program.viz` — ``RAPID_VISUALIZE_GRAPHS`` event-graph rendering
  (M8, port of the retired ``simulate_train_graph`` visualizer);
* :mod:`program.shadow` — ``compare_et_bundles``, the standard bundle
  comparator used by the differential/determinism tests.

Heavier submodules are imported lazily by their consumers; this package
import stays light.
"""

from program.layout import CANONICAL_AXES, RankLayout, cluster_coords

__all__ = ["CANONICAL_AXES", "RankLayout", "cluster_coords"]
