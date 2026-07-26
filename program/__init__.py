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

Migration stage M0 ships only :mod:`program.layout` (the unified
``RankLayout``); further modules (``ir``, ``schedule``, ``et_emit``, ...)
land in later stages.
"""

from program.layout import CANONICAL_AXES, RankLayout, cluster_coords

__all__ = ["CANONICAL_AXES", "RankLayout", "cluster_coords"]
