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
- The harness FORCES `RAPID_ASTRA_CACHE_MODE=NO_CACHE`
  (`equiv.runner.CACHE_MODE`). A harness that can read a stored AstraSim
  result gates the cache, not the code.
- One test per gate tier, per spec (`equiv.runner`, restructure P0):

  | test | tier | what it pins |
  |---|---|---|
  | `test_structural` | **T1** | `compare_structural`: per-rank op multisets, dependency-DAG Merkle hashes, **compute-microsecond totals**, **byte histograms by op kind and by interconnect axis**, **collectives grouped by resolved member set**, **transfer counts**, **payload-weighted critical-path length**, **`manifest.json` content + digest**, **`comm_groups.json` member sets**, **per-GPU memory peaks / capacity / headroom / warnings**. Every quantity is invariant under node-id renumbering. |
  | `test_timing` | **T2** | `compare_timing`: exact AstraSim per-rank wall seconds (from `astra_runs.json`) and end-to-end reported times. |
  | `test_contract` | **T3** | `compare_contract`: dlsim causal completability and the p2p tag-pairing bijection (`program.shadow.p2p_pairing_problems`). Fails on ANY candidate violation — a deadlocking bundle is a bug even when the golden deadlocked too. |
  | `test_reemission_deterministic` | **T3** | `compare_determinism`: a second, independent run must re-emit CANONICALLY identical bundles (not `filecmp`) and report the same total. |
  | `test_bug_ledger_entries_are_live` | **T4** | dead ledger entries must be pruned. |
  | `test_matches_golden` | all | the union, in one assertion. |

- **T4 — declare the change before you land it.** Put the expected
  difference in `tests/golden_equiv/bug_ledger.json`, keyed per
  `(spec, level, field)` with `{old, new, commit, justification}`
  (schema and matching rules: `equiv/ledger.py`). A listed mismatch is
  reported as a RECORDED EXCEPTION instead of a failure, so the rest of the
  matrix keeps gating while one bug fix lands. Entries expire the moment
  `old` stops matching (i.e. once the goldens are recaptured) and must then
  be pruned.
- Regeneration is allowed ONLY for deliberate, justified behavior changes:

  ```sh
  ./.venv/bin/python -m equiv.capture                       # full matrix
  ./.venv/bin/python -m equiv.capture --specs id1 id2 ...   # subset
  ./.venv/bin/python -m equiv.capture --list                # show matrix ids
  ```

  Commit the regenerated `tests/golden_equiv/*.json` together with the code
  diff and the justification. When the recapture is supposed to ADD pins
  without moving any (a gate rebuild), prove it:

  ```sh
  cp -r tests/golden_equiv /tmp/golden_before      # before recapturing
  ./.venv/bin/python tools/check_golden_superset.py /tmp/golden_before tests/golden_equiv
  ```

  which classifies every leaf as ADDED / CHANGED / REMOVED and fails on
  anything but ADDED (`git_rev` is capture provenance and is excluded).

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
- `tests/test_equiv_unit.py` — equivalence-harness unit tests: canonical
  form + diff field keys, dlsim, memory-summary parsing, the p2p pairing
  property, and the T4 ledger's matching/expiry rules.
- `tests/test_astrasim_cache_key.py` — the AstraSim result-cache key covers
  the emitted DAG (BUG_LEDGER A1): equal manifests with different
  dependencies / p2p tags / rank assignment must NOT collide, while the
  workload signature stays stable as the per-bundle run identity.
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
across them. That class is covered by the always-on
`test_reemission_deterministic`, which compares two SEPARATE processes'
bundles canonically; these sweeps remain useful while they still compare
builders that are scheduled for deletion.

## Notes

- The golden matrix takes ~30 s wall on 8 workers (two independent runs of
  all 42 specs: one for T1/T2/T3-vs-golden, one for the determinism check).
- `run_perf.py` HONORS an inherited `RAPID_ASTRA_CACHE_MODE`; it only
  supplies a default when the variable is unset. AstraSim artifacts cache
  under `astra_cache/` (safe to delete: it is regenerated, and the cache key
  changed with the A1 fix).
- Every persisted bundle directory carries `astra_runs.json` (per-invocation
  wall seconds, written in every cache mode) and `comm_axes.json` (group id
  -> parallelism axes; a sidecar AstraSim never reads).
