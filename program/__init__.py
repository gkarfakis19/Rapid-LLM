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

Post-cutover (P5/P6) module inventory. **One representation** (the IR) and
**one construction path** (``build()``); the proto graphs, the three granularity
builders and the emission-ordering pass are gone.

L0  * :mod:`program.workload` — ``WorkloadSpec``/``RunPolicy``/``DurationTable``/
      ``CommSpecTable`` (the typed workload; ``WorkloadSpec.from_timing`` is THE
      producer seam);
L1  * :mod:`program.work` — ``WorkItem``/``WorkSet``/``SyncRequirement``;
    * :mod:`program.policies` — sharding / grad-accum / recompute / routing /
      overlap, one named swappable object each;
L2  * :mod:`program.placement` — ``Placement``, ``Granularity``,
      ``BlockExpander``; :mod:`program.block` — ``BlockTemplate``/``CommMeta``;
    * :mod:`program.groups` — ``CommunicatorFactory`` (members CONSTRUCTED);
    * :mod:`program.layout` — the unified ``RankLayout``;
L3  * :mod:`program.schedule` — ``SchedulePolicy``/``Schedule``/``GPipeSchedule``;
L4  * :mod:`program.ir` — the Program IR; :mod:`program.build` — ``build()``,
      the ONLY Program constructor; :mod:`program.validate` — V1-V8;
L5  * :mod:`program.et_emit` — Program -> Chakra ET bundle (id policy = program
      order, the only policy);
    * :mod:`program.mapping` — the first-dimension SCOTCH remap over a built
      Program;
    * :mod:`program.analytic_sim` — the analytical evaluator + the byte->time
      conversion; :mod:`program.memory_sim` — the peak-memory replay;
      :mod:`program.retime` — the per-DP block retime write-back;
      :mod:`program.viz` — ``RAPID_VISUALIZE_GRAPHS`` rendering;
    * :mod:`program.shadow` — ``compare_et_bundles``, the bundle comparator.

Heavier submodules are imported lazily by their consumers; this package
import stays light.
"""

import os

from program.layout import CANONICAL_AXES, RankLayout, cluster_coords

__all__ = ["CANONICAL_AXES", "RankLayout", "cluster_coords", "_env_flag"]


def _env_flag(name: str) -> bool:
    """Truthy env-flag check (``RAPID_*`` switches). Single shared home —
    formerly duplicated byte-identically in config.py, run_perf.py,
    train_timing.py, llm_execution.py and program/memory_sim.py."""
    value = os.environ.get(name)
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized not in {"", "0", "false", "no"}
