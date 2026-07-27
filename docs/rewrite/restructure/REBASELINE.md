# REBASELINE.md — the P5/P6 hard-cutover delta table

> **Status: the reviewed delta table PLAN.md §P5 requires.** `build()` is now the only
> Program constructor for all four execution modes, the BLOCK measurements and the memory
> replay; the legacy spine (`legacy_lowering.py`, `schedule.py`, `pipeline_coarse.py`,
> `pipeline_fine.py`, `block_program.py`, `transforms.py`) is deleted. **The goldens are NOT
> recaptured here.** They are left failing on **T2 only**, for the 10 specs listed below, so
> the owner reviews the movement before it becomes the new baseline.
>
> Run: `env RAPID_ASTRA_CACHE_MODE=NO_CACHE LD_LIBRARY_PATH=... ./.venv/bin/python -m pytest
> tests/test_equiv_golden.py -q`

---

## 1. Gate status

| Tier | Result |
|---|---|
| **T1 structural** (canonical op multisets, DAG hashes, byte histograms, `manifest.json`, `comm_groups.json` member sets, per-GPU memory peaks) | **42/42 green**, with 475 declared T4 exceptions in **3 classes**, all non-semantic — §2 |
| **T2 predictions** (AstraSim per-rank wall seconds + end-to-end totals) | **32/42 green**, 10 moved — §3. The 10 analytical specs are **bit-exact**; every hybrid and hierarchical spec is bit-exact too |
| **T3 contracts** (dlsim completability, p2p tag bijection, group-order postcondition, deterministic re-emission) | **42/42 green**; 63/63 bundles complete — §4 |
| **T4 ledger** | 475 live entries, every one pinning `old -> new`; `test_bug_ledger_entries_are_live` green |
| **T5 physical** (`test_func_test`, `test_IMEC_*`, `test_koyeb_*`, `validation_scripts`) | **234/234 functionality tests, every error threshold unchanged** |

Everything else in the suite: `1331 passed, 42 skipped, 7 xfailed, 2 xpassed`; the only
failures are `test_timing` on the 10 specs of §3 and `test_matches_golden` on the same 10
(that gate is the *union* of T1–T4, so it fails exactly when its T2 component does).

---

## 2. The T1-exactness proof

**What is bit-identical, on every one of the 42 specs.** These are the quantities PLAN.md's
T1 row names as the modeling content, and not one of them appears in the ledger — i.e. the
harness compared them and found them equal:

| Quantity (as `equiv.canonical` computes it) | Status |
|---|---|
| `compute_micros` per rank and `compute_micros_total` | identical |
| `bytes_by_kind[COLL]` per rank and per bundle | identical |
| `collectives_by_group` — every collective grouped by its **resolved wire member set**, with counts, bytes and per-kind breakdown | identical |
| `byte_hist_by_kind[COLL]` | identical |
| **payload** SEND/RECV multisets — every `(size, peer)` with `size > 1` | identical |
| `bytes_by_axis` for every non-`p2p` axis | identical |
| `manifest.json` `COMP` / `COMM` rows, `npus`, and every `SEND:`/`RECV:` row except the 1-byte ones | identical |
| per-GPU memory peaks | identical on 40 of 42; the 2 MoE specs move by BUG_LEDGER **A3**, below |
| **analytical totals** (the 10 fully-analytical specs, which emit no ETs) | **bit-exact to the last digit**, verified independently of the harness — see §2.2 |

**What moved.** 475 ledger entries over 21 specs, in three classes:

### 2.1 `P5-CONTROL-TRANSFER` — 467 entries

`build()` emits a strict **subset** of the legacy **1-byte control** SEND/RECV pairs. Measured
directly, old bundle vs new bundle, on `train:flattened:dp1tp1cp2pp2mb2sp0` (the whole
difference, all four ranks):

```
COLL   12 / 12  ==      COMP 53 / 53  ==      payload SEND/RECV  8 / 8 x 2097152 B  ==
control (1 B) SEND/RECV   13  ->  8
```

Two causes, both contract decisions already reviewed at the L4 wave
(`tests/test_build.py::DIVERGENCES`, class `CONTROL_TRANSFER_SUBSET`):

1. **One p2p is one identity.** `legacy_lowering` recorded `same_device_edges` per
   *(edge, same-stage source parent)* and Step 11 created one `TransferOp` per key, so one
   logical p2p whose producer chain ended in a collective became **two** ops with two tags.
   `build()` emits one object with both producers as deps. Same endpoints, same payload.
2. **R3 is per DEVICE** (INTERFACES §4.3). Legacy attached the COARSE cross-microbatch GPipe
   edge to *every cluster rank of a stage* and lowered each cross-device pair to a zero-byte
   control message; `build()` records the boundary as a device-local dep, which needs no wire.
   Under AstraSim's collective semantics the cross-rank ordering is already enforced: every
   device of a stage issues the same per-layer block collectives, and a collective cannot
   complete until all members issue it, so the next microbatch's first op on any device is
   already behind that barrier.

Everything else in this class is **derived** from those counts: `ops_hash`, `dag_hash`,
`n_nodes`, `critical_path_nodes` / `critical_path_weight`, `bytes_by_kind[SEND|RECV]` (which
differ by exactly the number of dropped 1-byte pairs), `byte_hist_by_kind`, `bytes_by_axis/p2p`,
`n_transfers`, `n_recv`, `manifest_sha256`, and the `manifest ranks/N/ops/SEND:1|RECV:1` rows.

The critical-path numbers move most visibly (e.g. rank 1 of
`train:flattened:dp1tp1cp2pp2mb2sp0`: 18 -> 34 nodes) because the *encoding* of one stage's
serialization changed, not the serialization: legacy expressed it partly through cross-rank
1-byte messages, `build()` expresses it as explicit same-rank deps. The DAG got **more**
explicit, not less.

### 2.2 `A3` — 4 entries (2 specs)

`train:hybrid:dp2tp2cp1pp2mb2sp1:moe:ep2` and `train:hierarchical:dp2tp2cp1pp2mb2sp1:moe:ep2`:
`simulated_peak_memory_usage_per_gpu` **4.0 -> 2.68 GiB** (headroom 76.0 -> 77.32).

This is BUG_LEDGER **A3**, which INTERFACES §7 records as *"already impossible"* in the new
design: legacy's `OVERLAP_NODE_COPY_ATTRS` omitted `is_moe_layer`, so the head of a tp-overlap
split MoE layer was filed under *dense* and the census counted the same layer twice. `build()`
derives every field of a split op from the same `WorkItem`, so the flag cannot be lost. The
ledger's predicted direction and shape hold exactly: the reported peak **decreases**, by one
dense layer's worth per split MoE layer per device. Timing unaffected.

### 2.3 `P5-GID-NUMBERING` — 4 entries (1 spec)

`train:hierarchical:dp2tp2cp1pp2mb2sp1:moe:ep2`, `comm_groups.json`: gids 1000..1003 hold the
member sets `[0],[2],[1],[3]` instead of `[0],[1],[2],[3]`. All four are **singletons** (the
ep-axis communicator of a COARSE program, whose device space carries no ep axis —
`ABSENT_AXIS_SINGLETON_GROUP`), and a singleton wire group is emitted as a zero-duration
`*_noop` COMP, never as a collective (Class B 10g). The *set* of member sets is unchanged;
only which gid each interned under moved, because gids are allocated in sorted-label order and
labels are now assigned in program order. `collectives_by_group` — the T1 quantity keyed on the
member set rather than the gid — is identical.

### 2.4 Analytical bit-exactness, proved separately

The 10 fully-analytical specs emit no ETs, so nothing about them may move for an id-order
reason. Verified by evaluating the COARSE program of every analytical **and** hybrid case
through the new `program.analytic_sim` and comparing against the deleted proto-graph evaluator
on the same run, `repr`-identical:

```
train:analytical:dp1tp1cp1pp1mb1sp0        0.026110988503097537
train:analytical:dp2tp1cp1pp2mb2sp0        0.032949500056949785
train:analytical:dp1tp2cp1pp2mb2sp1        0.026015998614471965
train:analytical:dp1tp1cp2pp2mb2sp0        0.028468371962254373
train:analytical:dp2tp2cp2pp2mb2sp1        0.025442552723068690
train:analytical:dp2tp1cp1pp2mb2sp0:zero2  0.038826875056949790
train:analytical:dp2tp1cp1pp2mb2sp0:zero3  0.169817756271639750
train:analytical:dp2tp1cp1pp2mb2sp0:ga2    0.036514388841312460  (+ no_dp cycle 0.026418347776941083)
inf:analytical:dp1tp1cp1pp1mb1sp0          0.010549615481964376
inf:analytical:dp2tp2cp1pp2mb2sp1          0.010150620465687321
```

Two findings came out of that measurement and are now named in `program/analytic_sim.py`:

* **the tie discipline is free.** `Op.succs` (construction order) and ascending uid give
  *bit-identical* totals on every case, so P6's normative choice — **program order is the tie
  discipline** — costs nothing and is the same order AstraSim consumes as node-id priority.
* **a comm op that is a program ROOT carries no time** is load-bearing and preserved
  (`_ROOT_COMM_IS_UNTIMED`). The legacy conversion pass only ever converted a node's
  *children*, and the ZeRO-3 forward-entry gather *is* the coarse root. Timing it moves
  `train:analytical:dp2tp1cp1pp2mb2sp0:zero3` by **+5.76%**.

---

## 3. T2 — the delta table

Every spec, `total_time` and the `max_sec` of every bundle it produces. Rows marked **T2** are
the ones left failing.

| spec | golden total_time | new total_time | Δ% | bundle | golden max_sec | new max_sec | Δ% |
|---|---:|---:|---:|---|---:|---:|---:|
| `inf:analytical:dp1tp1cp1pp1mb1sp0` | 0.14 | 0.14 | 0 | — | — | — | — |
| `inf:analytical:dp2tp2cp1pp2mb2sp1` | 0.26 | 0.26 | 0 | — | — | — | — |
| `inf:flattened:dp1tp1cp1pp1mb1sp0` | 0.14 | 0.14 | 0 | flat | 0.010553 | 0.010553 | 0 |
| `inf:flattened:dp2tp2cp1pp2mb2sp1` **T2** | 0.23 | 0.22 | -4.348 | flat | 0.009449092 | 0.009179177 | -2.857 |
| `inf:hierarchical:dp1tp1cp1pp1mb1sp0` | 0.14 | 0.14 | 0 | hier | 0.010553 | 0.010553 | 0 |
| | | | | hier/fwd | 0.003839 | 0.003839 | 0 |
| `inf:hierarchical:dp1tp2cp1pp2mb2sp0:vit` | 0.01 | 0.01 | 0 | hier | 0.0099195 | 0.0099195 | 0 |
| | | | | hier/fwd | 0.001428 | 0.001428 | 0 |
| `inf:hierarchical:dp2tp2cp1pp2mb2sp1` | 0.23 | 0.23 | 0 | hier | 0.009412375 | 0.009412375 | 0 |
| | | | | hier/fwd | 0.001095 | 0.001095 | 0 |
| `train:analytical:dp1tp1cp1pp1mb1sp0` | 0.02611099 | 0.02611099 | 0 | — | — | — | — |
| `train:analytical:dp1tp1cp2pp2mb2sp0` | 0.02846837 | 0.02846837 | 0 | — | — | — | — |
| `train:analytical:dp1tp2cp1pp2mb2sp1` | 0.026016 | 0.026016 | 0 | — | — | — | — |
| `train:analytical:dp2tp1cp1pp2mb2sp0` | 0.0329495 | 0.0329495 | 0 | — | — | — | — |
| `train:analytical:dp2tp1cp1pp2mb2sp0:ga2` | 0.06293274 | 0.06293274 | 0 | — | — | — | — |
| `train:analytical:dp2tp1cp1pp2mb2sp0:zero2` | 0.03882688 | 0.03882688 | 0 | — | — | — | — |
| `train:analytical:dp2tp1cp1pp2mb2sp0:zero3` | 0.16981776 | 0.16981776 | 0 | — | — | — | — |
| `train:analytical:dp2tp2cp2pp2mb2sp1` | 0.02544255 | 0.02544255 | 0 | — | — | — | — |
| `train:flattened:dp1tp1cp1pp1mb1sp0` | 0.026111 | 0.026111 | 0 | flat | 0.026111 | 0.026111 | 0 |
| `train:flattened:dp1tp1cp2pp2mb2sp0` **T2** | 0.02814551 | 0.02811051 | -0.124 | flat | 0.028145506 | 0.028110506 | -0.124 |
| `train:flattened:dp1tp2cp1pp2mb2sp0:fault_tp` **T2** | 0.02530694 | 0.02508541 | -0.875 | flat | 0.025306937 | 0.025085409 | -0.875 |
| `train:flattened:dp1tp2cp1pp2mb2sp1` **T2** | 0.02481312 | 0.0245076 | -1.231 | flat | 0.024813124 | 0.024507598 | -1.231 |
| `train:flattened:dp1tp2cp1pp2mb2sp1:gqa` **T2** | 0.02017512 | 0.0198696 | -1.514 | flat | 0.020175124 | 0.019869598 | -1.514 |
| `train:flattened:dp1tp2cp1pp2mb2sp1:recompute` **T2** | 0.03023712 | 0.02991249 | -1.074 | flat | 0.030237124 | 0.029912487 | -1.074 |
| `train:flattened:dp1tp4cp1pp1mb1sp0:mesh2d` **T2** | 0.01296883 | 0.01292894 | -0.308 | flat | 0.012968834 | 0.012928942 | -0.308 |
| `train:flattened:dp1tp4cp1pp1mb1sp0:mesh2d_gmap` **T2** | 0.01296883 | 0.01292894 | -0.308 | flat | 0.012968834 | 0.012928942 | -0.308 |
| `train:flattened:dp2tp1cp1pp2mb2sp0` | 0.05275879 | 0.05275879 | 0 | flat | 0.052758791 | 0.052758791 | 0 |
| `train:flattened:dp2tp1cp1pp2mb2sp0:ga2` | 0.08043081 | 0.08043081 | 0 | flat | 0.054020687 | 0.054020687 | 0 |
| `train:flattened:dp2tp1cp1pp2mb2sp0:zero2` | 0.09278284 | 0.09278284 | 0 | flat | 0.092782841 | 0.092782841 | 0 |
| `train:flattened:dp2tp1cp1pp2mb2sp0:zero3` **T2** | 0.22019728 | 0.22320511 | +1.366 | flat | 0.220197279 | 0.22320511 | +1.366 |
| `train:flattened:dp2tp2cp2pp2mb2sp1` **T2** | 0.04498711 | 0.04231276 | -5.945 | flat | 0.044987106 | 0.042312763 | -5.945 |
| `train:hierarchical:dp1tp1cp1pp1mb1sp0` | 0.026111 | 0.026111 | 0 | hier | 0.026111 | 0.026111 | 0 |
| | | | | hier/bwd | 0.005707 | 0.005707 | 0 |
| | | | | hier/fwd | 0.002897 | 0.002897 | 0 |
| `train:hierarchical:dp1tp1cp2pp2mb2sp0` | 0.0280755 | 0.0280755 | 0 | hier | 0.0280755 | 0.0280755 | 0 |
| | | | | hier/bwd | 0.002144233 | 0.002144233 | 0 |
| | | | | hier/fwd | 0.000963634 | 0.000963634 | 0 |
| `train:hierarchical:dp1tp2cp1pp2mb2sp0:fault_pp` | 0.025541 | 0.025541 | 0 | hier | 0.025541 | 0.025541 | 0 |
| | | | | hier/bwd | 0.001822 | 0.001822 | 0 |
| | | | | hier/fwd | 0.000934 | 0.000934 | 0 |
| `train:hierarchical:dp1tp2cp1pp2mb2sp0:fault_tp` | 0.0252285 | 0.0252285 | 0 | hier | 0.0252285 | 0.0252285 | 0 |
| | | | | hier/bwd | 0.001822 | 0.001822 | 0 |
| | | | | hier/fault0_dp0_stage0_bwd | 0.001822 | 0.001822 | 0 |
| | | | | hier/fault0_dp0_stage0_fwd | 0.000934 | 0.000934 | 0 |
| | | | | hier/fwd | 0.000934 | 0.000934 | 0 |
| `train:hierarchical:dp1tp2cp1pp2mb2sp1` | 0.0247425 | 0.0247425 | 0 | hier | 0.0247425 | 0.0247425 | 0 |
| | | | | hier/bwd | 0.001776 | 0.001776 | 0 |
| | | | | hier/fwd | 0.000899 | 0.000899 | 0 |
| `train:hierarchical:dp2tp1cp1pp2mb2sp0` | 0.05275879 | 0.05275879 | 0 | hier | 0.052758791 | 0.052758791 | 0 |
| | | | | hier/bwd | 0.001971 | 0.001971 | 0 |
| | | | | hier/fwd | 0.000945 | 0.000945 | 0 |
| `train:hierarchical:dp2tp2cp1pp2mb2sp1:moe:ep2` | 0.05111506 | 0.05111506 | 0 | hier | 0.051115055 | 0.051115055 | 0 |
| | | | | hier/bwd | 0.001497924 | 0.001497924 | 0 |
| | | | | hier/fwd | 0.00039 | 0.00039 | 0 |
| | | | | hier/moe_bwd | 0.001528024 | 0.001528024 | 0 |
| | | | | hier/moe_fwd | 0.000767948 | 0.000767948 | 0 |
| `train:hierarchical:dp2tp2cp2pp2mb2sp1` | 0.04161979 | 0.04161979 | 0 | hier | 0.041619791 | 0.041619791 | 0 |
| | | | | hier/bwd | 0.001369337 | 0.001369337 | 0 |
| | | | | hier/fwd | 0.000366215 | 0.000366215 | 0 |
| `train:hybrid:dp1tp1cp1pp1mb1sp0` | 0.02611045 | 0.02611045 | 0 | hier/bwd | 0.005707 | 0.005707 | 0 |
| | | | | hier/fwd | 0.002897 | 0.002897 | 0 |
| `train:hybrid:dp1tp1cp2pp2mb2sp0` | 0.02807599 | 0.02807599 | 0 | hier/bwd | 0.002144233 | 0.002144233 | 0 |
| | | | | hier/fwd | 0.000963634 | 0.000963634 | 0 |
| `train:hybrid:dp1tp2cp1pp2mb2sp1` | 0.02474348 | 0.02474348 | 0 | hier/bwd | 0.001776 | 0.001776 | 0 |
| | | | | hier/fwd | 0.000899 | 0.000899 | 0 |
| `train:hybrid:dp2tp1cp1pp2mb2sp0` | 0.0270885 | 0.0270885 | 0 | hier/bwd | 0.001971 | 0.001971 | 0 |
| | | | | hier/fwd | 0.000945 | 0.000945 | 0 |
| `train:hybrid:dp2tp2cp1pp2mb2sp1:moe:ep2` | 0.02044628 | 0.02044628 | 0 | hier/bwd | 0.001497924 | 0.001497924 | 0 |
| | | | | hier/fwd | 0.00039 | 0.00039 | 0 |
| | | | | hier/moe_bwd | 0.001528024 | 0.001528024 | 0 |
| | | | | hier/moe_fwd | 0.000767948 | 0.000767948 | 0 |
| `train:hybrid:dp2tp2cp2pp2mb2sp1` | 0.01703159 | 0.01703159 | 0 | hier/bwd | 0.001369337 | 0.001369337 | 0 |
| | | | | hier/fwd | 0.000366215 | 0.000366215 | 0 |

### Summary of the movement

| | `total_time` (10 moved specs) | `max_sec` (10 moved bundles) |
|---|---:|---:|
| min | **-5.945%** | -5.945% |
| median | **-0.974%** | -0.975% |
| max | **+1.366%** | +1.366% |
| median absolute | 1.152% | 1.152% |

**Which specs move, and why exactly those.** Every moved spec is `flattened` — the FINE path,
the only one whose *emission order* changed materially. Analytical and hybrid never emit a
pipeline bundle; hierarchical does, and its wall seconds are **bit-identical on all 34
bundles of the 11 hierarchical specs** (and on all 14 BLOCK bundles of the
6 hybrid specs), because the COARSE pipeline DAG is deep and narrow enough that node-id
priority does not change the makespan. The FINE bundles are wide (a stage is `cluster_size`
ranks running GEMM chains with tp/cp collectives between them), so priority reordering plus the
dropped 1-byte control messages move the AstraSim schedule by ~1%.

The two extremes are worth naming:

* `train:flattened:dp2tp2cp2pp2mb2sp1` **-5.945%** — 16 ranks, the widest bundle in the matrix
  and the one with the most control transfers removed (rank 0: 14 -> 4 SEND, 14 -> 4 RECV).
* `train:flattened:dp2tp1cp1pp2mb2sp0:zero3` **+1.366%** — the only spec that got *slower*. It
  is also the only one where the emitter now reports dropped cross-rank dep edges
  (`et_emit.CrossRankDepWarning`, 4 edges): the ZeRO-3 gathers inherit a cross-stage anchor's
  deps, which the analytical evaluator honors and a rank-local `ctrl_dep` cannot express. Per
  BUG_LEDGER's own reframe this region is "unvalidated code, not pinned behavior" —
  `_should_shard_zero3_transformer` is dead at `tp=cp=1`, which every zero2/zero3 spec is.

**Not caused by the cutover:** `RAPID_ASTRA_CACHE_MODE=NO_CACHE` was already forced by the
harness (A1), and `tools/parallelism_sweep_cache.csv` is keyed on inputs only — delete it before
trusting a sweep.

---

## 4. T3 — dlsim completability, per bundle

**63 bundles across 42 specs; every one completes.** `equiv.dlsim` replays each bundle under
the AstraSim scheduling contract (per-rank issue order, collective rendezvous, p2p tag
matching) and reports no blocked rank and no cycle anywhere; `program.shadow.p2p_pairing_problems`
reports a bijection on every bundle. In addition:

* the always-on **group-order postcondition** in `et_emit` passes on every emitted bundle (it
  is a precondition of emission, not a test);
* `test_reemission_deterministic` — two independent whole-matrix runs, compared canonically —
  is green on all 42;
* the new `tests/test_build.py::test_build_is_deterministic_and_emits_a_completable_bundle`
  (`RAPID_BUILD_DIFF=1`, 42 specs x {COARSE, FINE}) asserts **O1** field-for-field on the
  Program and then canonically on the bundle, plus dlsim: **86 passed**. This is what replaced
  the four `RAPID_*_DIFF` builder differentials whose comparison targets are deleted.

One bundle is deliberately **built but not emitted**: FINE + MoE. Flattened MoE execution is
rejected in production (`llm_execution._run_full_astrasim_flattened`) and lands in P8; the FINE
MoE program exists only for the memory replay, and its group-order postcondition does not hold
yet (`ext_moe_flat.md`). The determinism sweep therefore emits COARSE for those two specs and
FINE for the other 40.

---

## 5. What the owner is being asked to approve

1. The **T2 movement of §3** on 10 flattened specs (min -5.945%, median -0.974%, max +1.366%),
   after which `python -m equiv.capture` recaptures and the T4 ledger below is pruned in the
   same commit.
2. The **T4 ledger** as the record of the T1 residual: 467 `P5-CONTROL-TRANSFER` + 4 `A3` +
   4 `P5-GID-NUMBERING` entries. `A3` is a Class A bug the interface declares unreproducible,
   so approving it is approving the fix; the other two classes are encodings, not content.
3. `program/analytic_sim.py`'s two named preserved quirks (§2.4): `_ROOT_COMM_IS_UNTIMED`, and
   "the memory replay times the pipeline-level collectives only".

Once (1) lands, `tests/golden_equiv/bug_ledger.json` must go back to `"entries": []` — that is
its steady state, and `test_bug_ledger_entries_are_live` enforces it.

---

## 6. Test inventory delta (so the collected count is not a surprise)

`pytest --collect-only` over the gated suite goes **1598 -> 1403**. Every removal is a
comparison against a deleted target; nothing that tested a PROPERTY was dropped.

| Removed | n | Why |
|---|---:|---|
| `tests/test_coarse_builder_diff.py` | 28 | the four `RAPID_*_DIFF` builder differentials the task authorizes deleting: their comparison targets (`pipeline_coarse`, `pipeline_fine`, `block_program` + `legacy_lowering`) no longer exist |
| `tests/test_hier_builder_diff.py` | 12 | " |
| `tests/test_fine_builder_diff.py` | 16 | " |
| `tests/test_block_builder_diff.py` | 18 | " |
| `tests/test_policies.py` — `test_policy_comm_keys_match_legacy_lattice` (whole config matrix), `test_via_reproduces_the_legacy_skip_flags`, `test_orphan_requirements_...` | 174 | replayed `schedule.build_pipeline_events` and logged `attach_parallel_edge`'s `add_child` calls; both are deleted. The orphan row was rewritten intrinsically (`test_no_orphan_requirements_at_pp1`) |
| `tests/test_placement.py` — the `_FineExpander._hw_id_for_rank`, `_build_axis_groups`, `build_fine_program` and `build_block_root` oracles | 35 | ditto; these were the differentials that had to pass BEFORE P3 could delete those builders |
| `tests/test_program_ir.py` — two `lower_to_program` quirk tests | 2 | rewritten as `test_same_device_transfer_edge_survives_elision` (which is the bug §2.1's recursion fixes) and `test_rank_formula_uses_the_declared_stage_count` (which keeps A4 expressible) |

| Added | n | What |
|---|---:|---|
| `tests/test_build.py` always-on | 45 | unchanged from the L3/L4 wave, plus the two rows the P5 amendments required (`directions`, the device-blind PARALLEL_TO inheritance) |
| `tests/test_build.py::test_build_is_deterministic_and_emits_a_completable_bundle` | 42 (env-gated, `RAPID_BUILD_DIFF=1`) | **replaces all four builder differentials**: 42 specs x {COARSE, FINE}, Program compared field-for-field for **O1**, bundle compared canonically, then `equiv.dlsim`. `86 passed in 64s` |
| `tests/test_policies.py`, `tests/test_program_ir.py` intrinsic replacements | 3 | above |

The 42 `skipped` in a default run are exactly that env-gated sweep.

## 7. One warning class is expected in the logs

`et_emit.CrossRankDepWarning` fires on the two `zero3` specs (4 edges each): a ZeRO-3 gather
inherits a cross-stage anchor's deps, which no wire carries. See §2.4 and the INTERFACES
amendment dated 2026-07-28 for why the edge stays in the IR and is dropped at emission rather
than the other way round. It is a `UserWarning` subclass so it is visible without being fatal;
`build()`'s `GroupRaceWarning` silencing does not hide it.
