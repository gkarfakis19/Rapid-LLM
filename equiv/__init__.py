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

"""Golden-equivalence harness for the AstraSim execution rewrite.

The harness pins the observable behavior of a RAPID-LLM checkout so that any
reimplementation of the graph-construction / AstraSim-conversion pipeline can
be proven equivalent:

- ``equiv.configs``  — the named config matrix under test.
- ``equiv.canonical`` — implementation-independent canonical form of Chakra ET
  bundles (op multisets + dependency-DAG hashes; pg ids resolved to members).
- ``equiv.runner``  — subprocess runner that executes one spec with isolated
  caches and persisted artifacts, then collects metrics + canonical bundles.
- ``equiv.capture`` — CLI to record golden JSON files for the current code.
- ``tests/test_equiv_golden.py`` — pytest gates comparing current code to the
  recorded goldens at four levels (ops, DAG, AstraSim wall times, totals).
"""
