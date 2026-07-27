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

"""L3 — schedule policies (INTERFACES.md §0.3, §4.1).

The package is ``program/sched/`` and not ``program/schedule/`` on purpose:
``program/schedule.py`` still exists while L3 lands. P5 renames this package to
``program/schedule/`` in the same commit that deletes that module (INTERFACES
§0.3 "Package-name collision").
"""

from __future__ import annotations

from program.schedule.gpipe import GPipeSchedule
from program.schedule.policy import (
    LayerAssignment,
    Schedule,
    ScheduleDep,
    ScheduleError,
    SchedulePolicy,
    ScheduleSlot,
)

__all__ = [
    "GPipeSchedule",
    "LayerAssignment",
    "Schedule",
    "ScheduleDep",
    "ScheduleError",
    "SchedulePolicy",
    "ScheduleSlot",
]
