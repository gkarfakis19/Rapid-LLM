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

"""Shared type aliases (INTERFACES.md §0.4).

Deliberately imports nothing from ``program.*`` — this module exists to break
every import cycle between L0 (:mod:`program.workload`), L1
(:mod:`program.work`, :mod:`program.policies`) and L2/L3.
"""

from __future__ import annotations

from typing import Mapping, NewType

AxisName = str  #: one of program.layout.CANONICAL_AXES
DeviceId = NewType("DeviceId", int)  #: a linearized rank in a RankLayout
StageId = NewType("StageId", int)  #: a pipeline stage index (the pp coordinate)
LayerId = int
MicroBatch = int
CommKey = str  #: index into a CommSpecTable
Coords = Mapping[AxisName, int]

__all__ = [
    "AxisName",
    "DeviceId",
    "StageId",
    "LayerId",
    "MicroBatch",
    "CommKey",
    "Coords",
]
