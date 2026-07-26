# Test-tier map: RAPID-LLM rewrite (`rewrite_astra`)

The discoverable map of every test tier guarding the Program-core rewrite.
Companion to `docs/rewrite/CONTEXT.md` (architecture/history) and
`docs/rewrite/DESIGN.md` (the design the gates pin).

## Run environment

All commands use the repo virtualenv interpreter and the AstraSim binary's
runtime libraries:

```sh
export LD_LIBRARY_PATH=/u1/ee/karfakis/gcc-10.2.0/lib64:/app/nanocad/projects/personal/gkarfakis/anaconda3/lib
./.venv/bin/python -m pytest ...
```

- Interpreter: `$REPO_ROOT/.venv/bin/python` (NOT the system python).
- `LD_LIBRARY_PATH` is required by the AstraSim analytical binary
  (`astra-sim/build/astra_analytical/build/bin`); without it every
  AstraSim-backed test errors out at simulation time.

## Tier 1 — the golden gate (must stay green at every step)

```sh
./.venv/bin/python -m pytest tests/test_equiv_golden.py -q          # whole matrix
./.venv/bin/python -m pytest tests/test_equiv_golden.py -q -k flat  # subset
```

- 42 golden specs (`equiv.configs.MATRIX`: 15 flattened, 11 hierarchical,
  10 analytical, 6 hybrid), recorded in `tests/golden_equiv/`.
- Five compare levels per spec (`equiv.runner.compare_observation`):
  1. per-rank op multisets,
  2. dependency-DAG Merkle hashes,
  3. exact AstraSim per-rank wall seconds,
  4. end-to-end reported times,
  5. dlsim causal completability (a golden-completed bundle must not
     deadlock in the `equiv/dlsim.py` replay).
- Regeneration is allowed ONLY for deliberate, justified behavior changes:

  ```sh
  ./.venv/bin/python -m equiv.capture                       # full matrix
  ./.venv/bin/python -m equiv.capture --specs id1 id2 ...   # subset
  ./.venv/bin/python -m equiv.capture --list                # show matrix ids
  ```

  Commit the regenerated `tests/golden_equiv/*.json` together with the code
  diff and the justification.

## Tier 2 — always-on unit/builder suites

Run with the whole test tree (webui browser/remote/service/worker-runner
tests need extra services and are usually excluded):

```sh
./.venv/bin/python -m pytest tests/ -q \
    --ignore=tests/test_webui_browser.py \
    --ignore=tests/test_webui_remote.py \
    --ignore=tests/test_webui_service.py \
    --ignore=tests/test_webui_worker_runner.py
```

Highlights:

- `tests/test_program_ir.py` — Program IR invariants V1-V5, the V6 race
  warning, all ET-emitter contract rules (dp skip, singleton no-op,
  control renumber, group-order postcondition incl. the interning and
  non-member-rank guards), legacy lowering quirks.
- `tests/test_program_layout.py` — `RankLayout` / descriptor / gmap config
  extraction.
- `tests/test_equiv_unit.py` — equivalence-harness unit tests (dlsim etc.).
- `tests/test_*_builder_diff.py` — the always-on synthetic halves of the
  four builder suites (see Tier 3 for their env-gated sweeps).
- Modeling/config suites: `test_mla_modeling.py`, `test_vit_modeling.py`,
  `test_moe_expert_imbalance.py`, `test_deepseek_v3_config.py`,
  `test_func_test.py`, `test_IMEC_*`, `test_koyeb_inf_single_gpu.py`,
  `test_webui_formatting.py`.

## Tier 3 — env-gated determinism sweeps (dev gates)

Each builder suite carries a sweep over its golden-matrix family that
builds + emits twice inside one process and requires identical op streams
and byte-identical ET bundles. They are skipped unless their env flag is
set:

```sh
RAPID_FINE_DIFF=1   ./.venv/bin/python -m pytest tests/test_fine_builder_diff.py -q   # flattened specs
RAPID_BLOCK_DIFF=1  ./.venv/bin/python -m pytest tests/test_block_builder_diff.py -q  # hybrid + hierarchical specs
RAPID_COARSE_DIFF=1 ./.venv/bin/python -m pytest tests/test_coarse_builder_diff.py -q # analytical + hybrid specs
RAPID_HIER_DIFF=1   ./.venv/bin/python -m pytest tests/test_hier_builder_diff.py -q   # hierarchical specs
```

Known limitation (documented in the suites): both builds share one
interpreter, so PYTHONHASHSEED-dependent iteration order is identical
across them — hash-seed-order regressions are caught only by the golden
gate's cross-run byte comparisons.

## Notes

- The golden matrix takes ~15 s wall on 8 workers (`-n 8` with
  pytest-xdist) and 2-4 minutes serially.
- AstraSim artifacts cache under `astra_cache/`; set
  `RAPID_ASTRA_CACHE_MODE` to control reuse.
