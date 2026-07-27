# Test-tier map: RAPID-LLM rewrite (`rewrite_astra`)

The discoverable map of every test tier guarding the Program-core rewrite.
Companion to `docs/rewrite/CONTEXT.md` (architecture/history) and
`docs/rewrite/DESIGN.md` (the design the gates pin).

> **Changed at the P5/P6 cutover.** The four `RAPID_{FINE,BLOCK,COARSE,HIER}_DIFF`
> builder sweeps **no longer exist**. They were differentials between two
> builders, and there is only one builder now (`program.build.build()`), so
> there is nothing left to difference. What replaced them, and where each
> property they held now lives, is §Tier 3.

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

- The golden matrix (`equiv.configs.build_matrix()` — 42 specs at the P5/P6
  rebaseline: 15 flattened, 11 hierarchical, 10 analytical, 6 hybrid),
  recorded in `tests/golden_equiv/`. Ask the code, not this file, for the
  current count: `./.venv/bin/python -m equiv.capture --list`.
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

## Tier 2 — always-on unit suites, one per abstraction level

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

- `tests/test_workload_spec.py` — **L0**: `WorkloadSpec.from_timing`, the
  single producer seam. Every `misc_metadata` key is REQUIRED and a missing
  one must raise **naming the key**, not silently default.
- `tests/test_policies.py` — **L1**: sharding (DDP / ZeRO-1/2/3), recompute,
  routing, overlap-as-declaration, grad-accum; plus the **W1 grep gate** that
  fails on a silent `dict.get(k, fallback)` reappearing in the producer seam.
- `tests/test_placement.py` — **L2**: `Placement`, `Granularity`,
  `BlockExpander`, and communicator membership **constructed** from
  `(axis, fixed coords)` rather than inferred from participant counts.
- `tests/test_build.py` — **L3/L4**: `build()`'s dependency rules R1–R5, each
  §4.3 probe parametrized over GPipe **and** `_AscBackward` (a non-GPipe
  schedule), so a dependency that survives only by schedule adjacency fails
  here. Also hosts the Tier 3 sweep.
- `tests/test_program_ir.py` — Program IR invariants V1–V10, the V6 race
  warning, and all ET-emitter contract rules (dp skip, singleton no-op,
  control renumber, group-order postcondition incl. the interning and
  non-member-rank guards, the SEND's compute anchor, cross-rank
  materialization, and the **V9** no-lost-successor postcondition).
- `tests/test_program_layout.py` — `RankLayout` / descriptor / gmap config
  extraction.
- `tests/test_equiv_unit.py` — equivalence-harness unit tests: canonical
  form + diff field keys, dlsim, memory-summary parsing, the p2p pairing
  property, and the T4 ledger's matching/expiry rules.
- `tests/test_astrasim_cache_key.py` — the AstraSim result-cache key covers
  the emitted DAG (BUG_LEDGER A1): equal manifests with different
  dependencies / p2p tags / rank assignment must NOT collide, while the
  workload signature stays stable as the per-bundle run identity.
- Modeling/config suites: `test_mla_modeling.py`, `test_vit_modeling.py`,
  `test_moe_expert_imbalance.py`, `test_deepseek_v3_config.py`,
  `test_func_test.py`, `test_IMEC_*`, `test_koyeb_inf_single_gpu.py`,
  `test_webui_formatting.py`.

## Tier 3 — the env-gated build sweep (dev gate)

**There is exactly one.** `tests/test_build.py` carries a whole-matrix sweep
that, for every golden spec × `{COARSE, FINE}`, builds twice inside one
process and asserts **O1** (INTERFACES §4.1) field-for-field on the two
`Program`s, then canonically on the two emitted bundles, then runs `dlsim`
on the result. It is skipped unless its env flag is set:

```sh
env RAPID_BUILD_DIFF=1 RAPID_ASTRA_CACHE_MODE=NO_CACHE \
    LD_LIBRARY_PATH=/u1/ee/karfakis/gcc-10.2.0/lib64:/app/nanocad/projects/personal/gkarfakis/anaconda3/lib \
    ./.venv/bin/python -m pytest tests/test_build.py -q -p no:randomly
```

**95 passed** at the P5/P6 revision the delta table describes
(`docs/rewrite/restructure/REBASELINE.md` §8); the number grows as probes are
added, so treat it as "all green", not as a pinned constant.

### What replaced the four deleted sweeps

`RAPID_FINE_DIFF`, `RAPID_BLOCK_DIFF`, `RAPID_COARSE_DIFF` and
`RAPID_HIER_DIFF` were deleted in `f06ef6f` together with
`pipeline_fine.py` / `block_program.py` / `pipeline_coarse.py` /
`legacy_lowering.py` — the builders they compared. Every property they held is
now held somewhere stronger:

| the old sweep asserted | now held by | why it is stronger |
|---|---|---|
| two builds in one process produce identical op streams | `RAPID_BUILD_DIFF` sweep (above), over **both** granularities in one suite | one builder, so the sweep tests determinism rather than agreement-between-duplicates |
| byte-identical ET bundles | `test_reemission_deterministic` (**always on**, T3) | two SEPARATE processes, compared **canonically** rather than by `filecmp`, so it also catches PYTHONHASHSEED-dependent iteration order — the documented blind spot of the old in-process sweeps |
| the new builder matched the legacy builder on the matrix | `test_structural` (**T1**) + `test_timing` (**T2**) against `tests/golden_equiv/` | pins against recorded *output*, which survives the deletion of the thing being compared to |
| — (not previously covered) | `test_contract` (**T3**): dlsim completability + p2p tag bijection, failing on ANY candidate violation | a deadlocking bundle is a bug even when the golden deadlocked too |
| — (not previously covered) | `test_bug_ledger_entries_are_live` (**T4**) | a declared exception that has stopped applying is itself a failure |

The tier split — **T1** structural / **T2** predictions / **T3** contracts +
determinism / **T4** ledger / **T5** physical — is the P0 gate rebuild
(`restructure/PLAN.md` §"Gate redesign"). `equiv/runner.py`'s single
`compare_observation` was split into `compare_structural` /
`compare_timing` / `compare_contract` / `compare_determinism` so that each
tier fails independently and a T2-only movement is legible as a T2-only
movement.

### The ledger and `equiv.capture`, together

These two are one workflow, and the order matters:

1. **Before** landing a change that will move a pinned value, declare it in
   `tests/golden_equiv/bug_ledger.json` (T4, schema in `equiv/ledger.py`).
   The matrix keeps gating everything else while the declared field is
   reported as a RECORDED EXCEPTION.
2. Write the delta table. `docs/rewrite/restructure/REBASELINE.md` is the
   worked example: every moved row, its **verified** cause, and the per-rank
   movement — not just `total_time`, because T2 gates `per_rank_sec`
   element-wise and the per-rank Δ is routinely an order of magnitude larger.
3. Get owner approval for any **modeling** claim in it.
4. Recapture with `equiv.capture`, then **prune**: the recapture makes every
   entry's `old` stop matching, `test_bug_ledger_entries_are_live` goes red,
   and the ledger must return to `{"entries": []}` — its enforced steady
   state. (At the rebaseline this fired at scale: 449 entries → 0.)

## Notes

- The golden matrix takes ~40 s wall on 8 workers (two independent runs of
  the whole matrix: one for T1/T2/T3-vs-golden, one for the determinism
  check).
- `astra_runs.json` entries are keyed and **sorted by workload signature**,
  which includes `manifest.json`. A change to the 1-byte control rows
  therefore RE-SORTS a multi-run bundle's list (inference prefill + decode
  samples, grad-accum's no-DP + final dispatchers). When reading a diff on
  such a bundle, pair runs **by role**, not by index — see
  `restructure/REBASELINE.md` §3.2.
- `run_perf.py` HONORS an inherited `RAPID_ASTRA_CACHE_MODE`; it only
  supplies a default when the variable is unset. AstraSim artifacts cache
  under `astra_cache/` (safe to delete: it is regenerated, and the cache key
  changed with the A1 fix).
- Every persisted bundle directory carries `astra_runs.json` (per-invocation
  wall seconds, written in every cache mode) and `comm_axes.json` (group id
  -> parallelism axes; a sidecar AstraSim never reads).
