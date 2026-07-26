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

Shipped so far: :mod:`program.layout` (M0, the unified ``RankLayout``), the
M1 shadow-mode stack — :mod:`program.ir` (the Program IR),
:mod:`program.validate` (invariants V1-V6), :mod:`program.legacy_lowering`
(legacy graph -> Program, transitional) and :mod:`program.et_emit`
(Program -> Chakra ET bundle) — and the M3a FINE builder:
:mod:`program.schedule` (``ScheduleSpec`` + the GPipe event enumeration),
:mod:`program.block` (``BlockTemplate``/``CommMeta``),
:mod:`program.pipeline_fine` (``build_fine_program``, the direct flattened
builder that replaced ``PipelineGraphFlattener``),
:mod:`program.transforms` (TP/TP-SP/CP overlap), the M3b memory replay
:mod:`program.memory_sim` (``simulate_memory`` over FINE Programs, the
replacement of ``Graph.simulate_memory``), and the M4 BLOCK builder
:mod:`program.block_program` (``build_block_program``, the replacement of
``Graph.construct_transformer_graph`` powering the hybrid/hierarchical
transformer AstraSim runs). Heavier submodules are imported lazily by
their consumers; this package import stays light.
"""

from program.layout import CANONICAL_AXES, RankLayout, cluster_coords

__all__ = ["CANONICAL_AXES", "RankLayout", "cluster_coords"]
