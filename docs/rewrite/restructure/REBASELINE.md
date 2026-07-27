# REBASELINE.md — the P5/P6 hard-cutover delta table

> **Status: the reviewed delta table PLAN.md §P5 requires — SECOND revision, 2026-07-29.**
> `build()` is the only Program constructor for all four execution modes, the BLOCK measurements
> and the memory replay; the legacy spine is deleted. **The goldens are NOT recaptured here.**
>
> Run: `env RAPID_ASTRA_CACHE_MODE=NO_CACHE LD_LIBRARY_PATH=... ./.venv/bin/python -m pytest
> tests/test_equiv_golden.py -q`

> ## What changed since the first revision
>
> The first revision explained all 10 moved specs with one story: *`build()` emits a strict SUBSET
> of legacy's 1-byte control SEND/RECV pairs, so fewer ops compete for the single comm slot, and
> the orderings those pairs carried are already enforced elsewhere.*
>
> **That story was false, and dropping a dependency also makes things faster.** Three independent
> forensics (closure preservation, the ZeRO-3 asymmetry, a rule-by-rule dependency audit) proved
> that on 8 of the 10 rows the dominant cause was a *different, undeclared* change: **two real
> ordering constraints were being thrown away at emission**. They are fixed. The T2 movement fell
> by an order of magnitude:
>
> | | first revision | **now** |
> |---|---:|---:|
> | largest down-mover | **−5.945%** | **−0.215%** |
> | median absolute | 1.152% | **0.141%** |
> | the one up-mover (`zero3`) | +1.366% | **+0.002%** |
> | specs whose `total_time` matches the golden exactly | 32 / 42 | **33 / 42** |
> | T4 ledger entries | 475 over 21 specs | **449 over 12 specs** |
> | `et_emit.CrossRankDepWarning` in a production emission | 2 specs | **the class is deleted** |
>
> Nothing in this document is "already enforced elsewhere" any more. Every row below names a
> **verified** cause, and §4 is the proof.

---

## 1. Gate status

| Tier | Result |
|---|---|
| **T1 structural** (canonical op multisets, DAG hashes, byte histograms, `manifest.json`, `comm_groups.json` member sets, per-GPU memory peaks) | **42/42 green**, with 449 declared T4 exceptions over **12** specs in 4 classes, all non-semantic — §2 |
| **T2 predictions** (AstraSim per-rank wall seconds + end-to-end totals) | **32/42 green**, 10 moved — §3. All 10 analytical specs, all 11 hierarchical specs (34 bundles), all 6 hybrid specs (14 BLOCK bundles) and 4 of the 14 flattened bundles are **bit-exact** |
| **T3 contracts** (dlsim completability, p2p tag bijection, group-order postcondition, **the new no-lost-successor postcondition**, deterministic re-emission) | **42/42 green**; 63/63 bundles complete — §6 |
| **T4 ledger** | 449 live entries, every one pinning `old -> new`; `test_bug_ledger_entries_are_live` green |
| **T5 physical** (`test_func_test`, `test_IMEC_*`, `test_koyeb_*`, `validation_scripts`) | **234/234 functionality tests, every error threshold unchanged** |

Whole gated suite: `20 failed, **1346** passed, 42 skipped, 7 xfailed, 2 xpassed`; the pre-fix
baseline was `20 failed, 1331 passed, ...` and the +15 are the new probes of §8. The only failures
are `test_timing` on the 10 specs of §3 plus `test_matches_golden` on the same 10 (that gate is the
*union* of T1–T4, so it fails exactly when its T2 component does).

---

## 2. The T1 residual

**What is bit-identical, on every one of the 42 specs** (unchanged from the first revision, and
now on strictly more specs):

| Quantity (as `equiv.canonical` computes it) | Status |
|---|---|
| `compute_micros` per rank and `compute_micros_total` | identical |
| `bytes_by_kind[COLL]` per rank and per bundle | identical |
| `collectives_by_group` — every collective grouped by its **resolved wire member set** | identical |
| `byte_hist_by_kind[COLL]` | identical |
| **payload** SEND/RECV multisets — every `(size, peer)` with `size > 1` | identical |
| `bytes_by_axis` for every non-`p2p` axis | identical |
| `manifest.json` `COMP` / `COMM` rows, `npus`, and every `SEND:`/`RECV:` row except the 1-byte ones | identical |
| per-GPU memory peaks | identical on 40 of 42; the 2 MoE specs move by BUG_LEDGER **A3** |
| **analytical totals** (the 10 fully-analytical specs) | **bit-exact to the last digit** |
| **all 34 hierarchical bundles and all 14 hybrid BLOCK bundles** | **now fully identical, hashes included** |

**What moved.** 449 ledger entries over 12 specs, in four classes.

### 2.1 `P5-CONTROL-TRANSFER` — 414 entries, 9 specs

`build()` emits fewer 1-byte control SEND/RECV pairs than legacy. Two causes, both now **measured**
rather than asserted:

**(1) One p2p is one identity.** `legacy_lowering` recorded `same_device_edges` per *(edge,
same-stage source parent)* and Step 11 created one `TransferOp` per key, so one logical p2p whose
producer chain ended in a collective became **two** ops with **two** tags. `build()` emits one
object with both producers as deps — and, since 2026-07-29, its SEND carries **both** as
`ctrl_deps`. That is what makes the merge ordering-neutral; before the fix it was not (see §4.1).
This cause alone explains `mesh2d` / `mesh2d_gmap` (`pp=1, mb=1`, 15 → 12 wires).

**(2) R3 is per DEVICE** (INTERFACES §4.3). Legacy attached the COARSE cross-microbatch GPipe
boundary to *every cluster rank of a stage* and lowered each cross-device pair to a zero-byte
control message; `build()` records the boundary as a device-local dep, which needs no wire.

> **Correction.** The first revision justified (2) with *"the ordering is already enforced: a
> collective cannot complete until all members issue it, so the next microbatch's first op on any
> device is already behind that barrier."* **That is wrong.** The collective barrier is
> `S(coll_j,k) ≤ E(coll_i,k)`; it orders nothing that sits *after* the last collective of a
> microbatch on the peer rank. Isolated on the one moved spec with no other cause
> (`train:flattened:dp1tp1cp2pp2mb2sp0`, `tp=1, cp=2`), **73 ordering pairs are lost and 0 added**,
> all 73 attributable to the removed wires. Two witnesses from the ETs:
>
> * legacy r0 `SEND tag=262 deps=[embedding_b_188]` → r1 `RECV` is the **sole** dep of r1's
>   `optimizer_stage0_rank1_190` (3447 µs). Lost: `r0:embedding_b ⇒ r1:optimizer`,
>   `r0:layernorm1_backward_mb0_l0_rank0 ⇒ r1:optimizer`, `r0:qkv_proj_backward_mb0_l0_rank0 ⇒
>   r1:optimizer`.
> * legacy r0 `SEND tag=260 deps=[embedding_b_155]` gated r1's `MLP_backward_mb0_l1`. Lost:
>   `r0:attention_backward_reduce_scatter_1 ⇒ r1:{output_proj_backward_all_gather_1,
>   MLP_backward_mb0_l1, …}`.
>
> **The right description is "a spurious constraint was removed", not "the ordering is already
> enforced".** INTERFACES §4.3 states R3 per DEVICE and legacy's cross-product over the cluster
> ranks of a stage is an artifact of its per-stage wiring — but that is a **modeling** claim to
> approve, not an encoding claim to wave through. It is the whole of the residual T2 movement on
> these 9 specs (−0.116% to −0.215%), and §5.1 is the owner decision it needs.

Everything else in this class is **derived** from those counts: `ops_hash`, `n_nodes`,
`critical_path_nodes` / `critical_path_weight`, `bytes_by_kind[SEND|RECV]`, `byte_hist_by_kind`,
`bytes_by_axis/p2p`, `n_transfers`, `n_recv`, `manifest_sha256`, and the `manifest ranks/SEND:1 |
RECV:1` rows. `critical_path_*` also moves because R2 now **states** the microbatch's
`SOFTMAX/FORWARD → SOFTMAX/BACKWARD` and `LAYER/FORWARD → RECOMPUTE` edges (§4.3); those orderings
were already implied, so the AstraSim wall clock does not move with them.

### 2.2 `P5-ZERO3-CROSS-STAGE-SYNC` — 27 entries, 1 spec

`train:flattened:dp2tp1cp1pp2mb2sp0:zero3`, `byte_hist RECV:1` **8 → 16**. `SEND:1` is now
**identical to golden (16)**. See §4.2 — this is the class that used to be a dropped edge and a
warning.

### 2.3 `A3` — 4 entries, 2 specs

`train:hybrid:dp2tp2cp1pp2mb2sp1:moe:ep2` and `train:hierarchical:dp2tp2cp1pp2mb2sp1:moe:ep2`:
`simulated_peak_memory_usage_per_gpu` **4.0 → 2.68 GiB** (headroom 76.0 → 77.32). BUG_LEDGER **A3**,
which INTERFACES §7 records as *"already impossible"* in the new design: legacy's
`OVERLAP_NODE_COPY_ATTRS` omitted `is_moe_layer`, so the head of a tp-overlap split MoE layer was
filed under *dense* and the census counted the same layer twice. The ledger's predicted direction
and shape hold exactly. Timing unaffected. **Unchanged from the first revision.**

### 2.4 `P5-GID-NUMBERING` — 4 entries, 1 spec

`train:hierarchical:dp2tp2cp1pp2mb2sp1:moe:ep2`, `comm_groups.json`: gids 1000..1003 hold the member
sets `[0],[2],[1],[3]` instead of `[0],[1],[2],[3]`. All four are **singletons** (the ep-axis
communicator of a COARSE program, whose device space carries no ep axis), and a singleton wire group
is emitted as a zero-duration `*_noop` COMP, never as a collective (Class B 10g). The *set* of member
sets is unchanged; only which gid each interned under moved. `collectives_by_group` — the T1 quantity
keyed on the member set — is identical. **Unchanged from the first revision.**

### 2.5 The 16 entries that are GONE

The first revision carried 16 `dag_hash` entries over 10 specs
(`train:hierarchical:{dp2tp1cp1pp2mb2sp0, dp1tp2cp1pp2mb2sp1, dp1tp1cp2pp2mb2sp0,
dp2tp2cp2pp2mb2sp1, dp2tp2cp1pp2mb2sp1:moe:ep2, dp1tp2cp1pp2mb2sp0:fault_tp,
dp1tp2cp1pp2mb2sp0:fault_pp}`, `train:flattened:{dp2tp1cp1pp2mb2sp0, :zero2, :ga2}`) filed under
`P5-CONTROL-TRANSFER`. **They were not control transfers.** They were the microbatch's
`SOFTMAX/FORWARD(b) → SOFTMAX/BACKWARD(b)` data-flow edge, which `build()` carried only through an
R3 adjacency and therefore only for the microbatch the schedule happens to put adjacently.
Stating it in R2 (§4.3) makes every one of those bundles **bit-exact with its golden again**, so the
entries are deleted rather than re-filed. That is the "10 misattributed entries" the cutover
verification flagged, resolved.

### 2.6 Analytical bit-exactness, proved separately

The 10 fully-analytical specs emit no ETs. Verified by evaluating the COARSE program of every
analytical **and** hybrid case through `program.analytic_sim`, `repr`-identical to the golden:

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

Two preserved quirks are named in `program/analytic_sim.py`:

* **the tie discipline is free.** Construction order and ascending uid give *bit-identical* totals on
  every case, so P6's normative choice — **program order is the tie discipline** — costs nothing and
  is the same order AstraSim consumes as node-id priority.
* **a comm op that is a program ROOT carries no time** is load-bearing and preserved
  (`_ROOT_COMM_IS_UNTIMED`). Timing it moves `train:analytical:dp2tp1cp1pp2mb2sp0:zero3` by **+5.76%**.
  This is a **precondition of BUG_LEDGER A6** (§5.2): re-anchoring the ZeRO-3 prefetch legitimately
  makes a stage's first gather a root, and it would otherwise become free.

---

## 3. T2 — the delta table

Every moved row, with its **verified** cause. Rows marked **T2** are the ones left failing.
Everything not listed is bit-exact on `total_time` and on every bundle's per-rank wall seconds.

| spec | golden `total_time` | new | Δ% | bundle | golden `max_sec` | new | Δ% | **verified cause** |
|---|---:|---:|---:|---|---:|---:|---:|---|
| `train:flattened:dp1tp4cp1pp1mb1sp0:mesh2d` **T2** | 0.01296883 | 0.01295383 | **−0.116** | flat | 0.012968834 | 0.012953834 | −0.116 | §2.1 cause (1) — 3 duplicate-identity control wires |
| `train:flattened:dp1tp4cp1pp1mb1sp0:mesh2d_gmap` **T2** | 0.01296883 | 0.01295383 | **−0.116** | flat | 0.012968834 | 0.012953834 | −0.116 | as above |
| `train:flattened:dp1tp1cp2pp2mb2sp0` **T2** | 0.02814551 | 0.02811051 | **−0.124** | flat | 0.028145506 | 0.028110506 | −0.124 | §2.1 cause (2) ONLY — 5 cross-cluster-rank R3 wires, 73 lost pairs |
| `train:flattened:dp1tp2cp1pp2mb2sp0:fault_tp` **T2** | 0.02530694 | 0.02527194 | **−0.138** | flat | 0.025306937 | 0.025271937 | −0.138 | §2.1 causes (1)+(2) — 6 wires |
| `train:flattened:dp1tp2cp1pp2mb2sp1` **T2** | 0.02481312 | 0.02477812 | **−0.141** | flat | 0.024813124 | 0.024778124 | −0.141 | §2.1 causes (1)+(2) — 7 wires |
| `train:flattened:dp1tp2cp1pp2mb2sp1:gqa` **T2** | 0.02017512 | 0.02014012 | **−0.173** | flat | 0.020175124 | 0.020140124 | −0.173 | §2.1 causes (1)+(2) — 7 wires |
| `train:flattened:dp2tp2cp2pp2mb2sp1` **T2** | 0.04498711 | 0.04489122 | **−0.213** | flat | 0.044987106 | 0.044891222 | −0.213 | §2.1 causes (1)+(2) — 42 wires, 16 ranks |
| `train:flattened:dp1tp2cp1pp2mb2sp1:recompute` **T2** | 0.03023712 | 0.03017212 | **−0.215** | flat | 0.030237124 | 0.030172124 | −0.215 | §2.1 causes (1)+(2) — 39 wires |
| `train:flattened:dp2tp1cp1pp2mb2sp0:zero3` **T2** | 0.22019728 | 0.22020228 | **+0.002** | flat | 0.220197279 | 0.220202279 | +0.002 | §4.2 — the 8 cross-stage ZeRO-3 sync pairs are now all PAIRED (5 µs) |
| `inf:flattened:dp2tp2cp1pp2mb2sp1` **T2** | 0.23 | 0.23 | **0** | flat (prefill) | 0.009449092 | 0.009434092 | −0.159 | §2.1 causes (1)+(2) — 3 wires |
| | | | | flat (decode 1) | 0.003508733 | 0.003475733 | −0.940 | as above |
| | | | | flat (decode 2) | 0.003490733 | 0.003493733 | +0.086 | as above |

**Every other spec is 0.000%**, including all 10 analytical, all 11 hierarchical (34 bundles), all 6
hybrid (14 BLOCK bundles), and the flattened `dp1tp1cp1pp1mb1sp0`, `dp2tp1cp1pp2mb2sp0`, `:ga2`,
`:zero2` and `inf:dp1tp1cp1pp1mb1sp0`.

### 3.1 Summary of the movement

| | `total_time` | `max_sec` (flattened bundles) |
|---|---:|---:|
| min | **−0.215%** | −0.940% (an inference decode sample) |
| median | **−0.141%** | −0.141% |
| max | **+0.002%** | +0.086% |
| median absolute | **0.141%** | 0.141% |

Compare the first revision: min −5.945%, median −0.974%, max +1.366%, median absolute 1.152%.

### 3.2 Two rows need reading carefully

* **`inf:flattened:dp2tp2cp1pp2mb2sp1`.** Its `total_time` now *matches* the golden (0.23), but the
  golden records inference totals to **2 decimal places**, so a `total_time` match on this row proves
  much less than it does elsewhere — the honest numbers are the per-bundle ones above and the
  `phase decode` line (0.224 → 0.223). The row still fails T2 for a second, non-modeling reason:
  `astra_runs.json` entries are keyed and **sorted by the workload signature**, which includes
  `manifest.json`, which contains the 1-byte SEND/RECV rows. Changing those rows re-sorts the run
  list, so `astra[flat] run 0: signature differs` even where the wall seconds agree. That is a
  harness artifact of a multi-run bundle (prefill + 2 decode samples); it is reported, not hidden.
* **`train:flattened:dp2tp1cp1pp2mb2sp0:zero3`.** +0.002% is **5 microseconds on 220 ms**, and it is
  the price of making the emitted DAG equal the IR that the analytical evaluator honors (§4.2). The
  previous revision's +1.366% was four dropped edges.

---

## 4. The three verified causes

### 4.1 The cross-stage transfer's compute anchor was thrown away at emission

`program/build.py::_emit_cross_layer` deliberately records **two** DATA_FLOW deps on a pipeline
transfer, and says why:

```python
self._add_dep(producer_nid, nid, DepClass.DATA_FLOW)
# The compute-anchor double-dep (``pipeline_fine.py:672-687``): the SEND
# must fire off the last COMPUTE, not off a trailing collective.
anchor = self._nearest_compute(prod, producer_nid)
if anchor is not None and anchor != producer_nid:
    self._add_dep(anchor, nid, DepClass.DATA_FLOW)
```

`program/et_emit.py` Phase B emitted `transfer.producer` and **nothing else**. The anchor was
dropped silently: not a `cross_rank_drop`, no warning, no ledger row. The IR was right; the wire was
wrong.

**Why it matters at `tp > 1`.** With `sp1` the tp/sp `OverlapDecl` has `fraction >= 1.0`, so
`_split_compute`'s hoist branch runs the reduce-scatter *in parallel with* the MLP compute — anchor
and producer become **siblings**, not ancestor and descendant. So the emitted SEND of the
inter-stage activation had **no dependency at all on the compute that produced that activation**,
and the layer's last COMPUTE became a dependency **sink**:

| | legacy SEND deps | P5 (before the fix) |
|---|---|---|
| `dp1tp2cp1pp2mb2sp1` r0 | `MLP_forward_mb0_l1_rank0_31` (COMP) | `MLP_forward_reduce_scatter` (COLL) |
| `dp2tp2cp2pp2mb2sp1` r4 | `layernorm1_backward_mb1_l2_rank0_329` | `layernorm1_backward_all_gather` |
| `mesh2d` r1 | `MLP_forward_mb0_l1_rank1_49` | `MLP_forward_all_reduce` |

Newly-sunk semantic ops before the fix: `inf:dp2tp2` 3 ops / 336 µs, `fault_tp` 3 / 552 µs,
`dp1tp2cp1pp2mb2sp1(+gqa)` 3 / 579 µs, `mesh2d(_gmap)` 3 / 522 µs, `dp2tp2cp2pp2mb2sp1` 14 / 952 µs,
`tp=1` specs **0**. That is exactly why all 5 zero-movement flattened controls were `tp=1` and all 8
`tp>1` flattened specs moved.

**Causal proof (measured before the fix landed).** Patching each emitted bundle by adding back only
the dropped local producer edges — monotone, so node ids and hence AstraSim priorities stay
byte-identical — and re-running the real AstraSim binary recovered 62–95% of the movement:

| spec | golden | P5 | + restored edges | Δ% before | Δ% after | edges |
|---|---:|---:|---:|---:|---:|---:|
| `dp2tp2cp2pp2mb2sp1` | 0.044987106 | 0.042312763 | 0.044857194 | −5.945% | **−0.289%** | +32 |
| `dp1tp2cp1pp2mb2sp0:fault_tp` | 0.025306937 | 0.025085409 | 0.025243765 | −0.875% | **−0.250%** | +4 |
| `dp1tp2cp1pp2mb2sp1` | 0.024813124 | 0.024507598 | 0.024710861 | −1.231% | **−0.412%** | +8 |
| `dp1tp2cp1pp2mb2sp1:gqa` | 0.020175124 | 0.019869598 | 0.020072861 | −1.514% | **−0.507%** | +8 |
| `mesh2d` / `mesh2d_gmap` | 0.012968834 | 0.012928942 | 0.012953834 | −0.308% | **−0.116%** | +6 |

**The fix** (INTERFACES §4.3 R2 amendment 2026-07-29): a SEND's `ctrl_deps` are the union of
`transfer.producer` and every other member of `transfer.deps` that resolves on the SEND's own rank.
The anchor is on the SEND's own rank, so this was a plain omission, never an inexpressibility.

### 4.2 The four ZeRO-3 cross-rank edges were not ordering-redundant either

`et_emit` dropped any dep whose two ends live on different ranks with no transfer carrying it, and
warned with: *"The ordering they encode is carried by the shared per-stage collectives (a collective
cannot complete until every member issues it)."*

**Disproved.** The four edges are `zero3_transformer_gather_{fwd_b0_l2, fwd_b1_l2, bwd_b1_l1,
bwd_b0_l1}` ← a `cross_layer` transfer on the **other** pipeline stage (rows **S8** and **S15**, not
S6/S14 as §7 of the previous revision and `et_emit.py:296` both claimed — verified by instrumenting
`SyncRequirement.from_spec` and reading `req.origin`). At `dp2 tp1 cp1 pp2`, the gather's wire
communicator is `comm_groups.json` gid `2 = [1,3]` (stage 1's dp pair) and gid `1 = [0,2]` is stage
0's: **no collective has members on both stages**, so no collective can carry a stage-0 → stage-1
ordering. Reachability with the direct edge removed is `False` in both directions for all four. All
four gathers became **program roots** (`deps=[]`) that AstraSim issues at t=0, which is the whole of
the old `+1.366%`.

**Legacy put them on the wire.** At HEAD~1 the lowering materializes **eight** zero-byte
`cross_layer_rank0` `TransferOp`s for this shape; the four whose consumer sits on `dst_device`
produce a paired SEND/RECV, and the other four produce **unpaired** SENDs that nothing waits on —
which is exactly the golden's `SEND:1 = 16, RECV:1 = 8`.

**The fix** (INTERFACES §4.3 R4 amendment 2026-07-29): emission is **total**. A cross-rank sync dep
becomes a 1-byte control SEND/RECV pair (`_SYNC_TAG_BASE`), in **both** directions; an edge that
cannot be carried raises `EmissionError`. `CrossRankDepWarning` is deleted, so the class cannot
silently return. `SEND:1` is now identical to the golden (16); `RECV:1` is 16 vs the golden's 8,
because the eight legacy SENDs that carried no ordering now carry it. Cost: **5 µs, +0.002%**,
down from +1.366%. The analytical golden stays bit-exact (the IR is untouched), so the two
evaluators finally honor the same DAG.

The *underlying* defect — the S8/S15 prefetch host is chosen by layer arithmetic and lands on
another device — is filed as **BUG_LEDGER A6** and is deliberately **not** fixed here (§5.2).

### 4.3 Four real dependencies were carried only by a schedule adjacency

R3 materializes an adjacent pair of a device's projection *only when it is not already implied*
(`build.py:995`). Any dependency whose sole carrier is an R3 edge is therefore one adjacency away
from disappearing, and its identity is chosen by the `SchedulePolicy`. Four such dependencies were
found; all four are now stated by a rule, and each is regression-probed against **a non-GPipe
schedule** (`tests/test_build.py::_AscBackward` — GPipe with the backward walking microbatches
ascending, the one property every 1F1B/interleaved schedule has):

| dependency | now stated by | measured effect on the artifact |
|---|---|---|
| `SOFTMAX/FORWARD(b) → SOFTMAX/BACKWARD(b)` | **R2** | +1 edge per pipeline; **removes 16 ledger `dag_hash` rows over 10 specs** (§2.5); AstraSim wall clock unchanged |
| `LAYER/FORWARD(b,l) → RECOMPUTE(b,l)` | **R2** | +33 edges on the one recompute spec, all already implied; AstraSim wall clock unchanged |
| `OPTIMIZER(stage s)` after every backward item of stage `s` | **R5** (new) | **0 edges** under GPipe — redundancy-eliminated exactly like R3. Under `_AscBackward`, 6 of 9 backward items were previously un-ordered before the optimizer |
| a ZeRO-3 prefetch gather → the work it prefetches for (rows **S6/S14**) | `SyncRequirement.consumers`, resolved through `ShardingContext.next_after(hosts, kind, direction)` over `Schedule.order()` | **0 structural change on all 42 specs** — under GPipe it names exactly the op `via` already found. Under `_AscBackward` one gather previously preceded *nothing at all* and another attached to the wrong microbatch |

Two adjacent bookkeeping defects were fixed with them: `meta.misc["schedule_edges"]` was **stale**
wherever overlap ran (6/13 pairs on `dp1tp2cp1pp2mb2sp1`, 14/25 on `dp2tp2cp2pp2mb2sp1`) because
`_split_compute`/`_split_collective` re-parent *after* R3 — it is derived from the surviving edge
table now; and those re-parents re-added every moved edge as `DATA_FLOW`, erasing the `DepClass`
that `PARALLEL_TO`'s `via` filter dispatches on.

**What was checked and found sound.** `_apply_r2` stops at SOFTMAX/FORWARD and restarts at
SOFTMAX/BACKWARD, and R3 does drop the explicit `linear_softmax[mb] → linear_softmax_b[mb]` dep when
it finds it transitively implied. On all 15 flattened bundles, in both the plain-dependency and the
collective-rendezvous closure, **every** `(softmax_fwd[mb], softmax_bwd[mb])` pair is present in the
new closure: 0 missing. The R3 drop at `build.py:995` is sound; what was missing was the *statement*,
which is why §2.5's 16 entries existed.

---

## 5. What the owner is being asked to approve

### 5.1 The T2 movement of §3 — and the one modeling claim under it

Nine flattened specs move by **−0.116% to −0.215%** and one by **+0.002%**. Median absolute
**0.141%**, down from 1.152%.

**The single claim that needs an owner decision** is §2.1 cause (2): *legacy's cross-cluster-rank
GPipe control wires expressed a synchronization the model does not require, and INTERFACES §4.3 is
right to state R3 per DEVICE.* This is a **removed over-constraint**, proven not to be implied by
any surviving path (§2.1). If you accept it, the nine `−0.1x%` rows are correct and the goldens
should be recaptured. If you do not, R3 must fan the boundary across the cluster ranks of a stage
again, and those rows will return to 0.000%.

Everything else in §3 is now accounted for by a fix, not by a preference.

### 5.2 The T4 ledger as the record of the T1 residual

449 entries over 12 specs: 414 `P5-CONTROL-TRANSFER` + 27 `P5-ZERO3-CROSS-STAGE-SYNC` + 4 `A3` +
4 `P5-GID-NUMBERING`. `A3` is a Class-A bug the interface declares unreproducible, so approving it is
approving the fix; the other three are encodings and wire-level accounting, not modeling content.
Once §5.1 lands, `python -m equiv.capture` recaptures and
`tests/golden_equiv/bug_ledger.json` goes back to `"entries": []` — its steady state, enforced by
`test_bug_ledger_entries_are_live`.

### 5.3 Two follow-ups filed, not fixed

* **BUG_LEDGER A6** *(new, Class A)* — the ZeRO-3 prefetch anchor is chosen by layer arithmetic and
  lands on another device (rows S8/S15). A layer-`L` parameter all-gather is a dp-axis collective
  over `stage(L)`'s replicas; a stage-0 op is not its producer in any sense. Fixing it means
  re-anchoring on the gather's own device, which **moves the analytical zero3 golden** (≈ −4.3%
  measured with a crude local anchor) and requires `analytic_sim._ROOT_COMM_IS_UNTIMED` to be
  settled first, because a stage's first gather then legitimately becomes a program root. That is a
  deliberate modeling change with its own delta table — not a cutover artifact — so P5 makes the two
  evaluators agree on the *current* anchor instead (§4.2).
* **BUG_LEDGER D 11** *(new, Class D)* — **nothing orders the optimizer after its stage's gradient
  reducer**, at HEAD and at HEAD~1 alike (0 of 9 COARSE / 0 of 18 FINE paths). R5 deliberately stops
  at the backward COMPUTE items. If the weight update must also wait for the all-reduce that produces
  the gradients it applies, add the reducer's `SyncKey` to R5's source set; totals will **increase**,
  most at large `dp` and small `pp`.

### 5.4 The preserved analytical quirks

`program/analytic_sim.py`'s two named quirks (§2.6): `_ROOT_COMM_IS_UNTIMED`, and "the memory replay
times the pipeline-level collectives only".

---

## 6. T3 — contracts, per bundle

**63 bundles across 42 specs; every one completes.** `equiv.dlsim` replays each bundle under the
AstraSim scheduling contract and reports no blocked rank and no cycle;
`program.shadow.p2p_pairing_problems` reports a bijection on every bundle. In addition:

* the always-on **group-order postcondition** in `et_emit` passes on every emitted bundle;
* the always-on **no-lost-successor postcondition** (**V9**, new — `et_emit._check_no_lost_successors`)
  passes on every emitted bundle. This is the gate §4.1's defect would have tripped: an op with
  successors in the Program must keep at least one in the emitted trace. A dropped ordering edge is
  invisible to the op multisets, the byte histograms, `dlsim` and the group-order check, and moves
  only the AstraSim wall clock — which is precisely why it survived the first revision;
* emission is **total for cross-rank deps** (**V10**, new): materialized as a 1-byte control pair or
  `EmissionError`. `CrossRankDepWarning` is deleted;
* `test_reemission_deterministic` — two independent whole-matrix runs, compared canonically — is
  green on all 42;
* `tests/test_build.py::test_build_is_deterministic_and_emits_a_completable_bundle`
  (`RAPID_BUILD_DIFF=1`, 42 specs × {COARSE, FINE}) asserts **O1** field-for-field on the Program and
  then canonically on the bundle, plus dlsim.

One bundle is deliberately **built but not emitted**: FINE + MoE. Flattened MoE execution is rejected
in production and lands in P8; the FINE MoE program exists only for the memory replay, and its
group-order postcondition does not hold yet (`ext_moe_flat.md`).

---

## 7. Code and doc changes behind this revision

| Change | Site |
|---|---|
| SEND carries every same-rank dep of the transfer (the compute anchor) | `program/et_emit.py` Phase B |
| Cross-rank sync deps materialized as 1-byte control pairs; `CrossRankDepWarning` deleted; `EmissionError` when uncarriable | `program/et_emit.py` Phase B2 |
| **V9** no-lost-successor postcondition, always on | `program/et_emit.py::_check_no_lost_successors` |
| R2 states `SOFTMAX/F(b) → SOFTMAX/B(b)` and `LAYER/F(b,l) → RECOMPUTE(b,l)` | `program/build.py::_apply_r2` |
| **R5** — the optimizer's gradient dependency, redundancy-eliminated | `program/build.py::_apply_r5` |
| `SyncRequirement.consumers`; `ShardingContext.order` / `next_after`; S6/S14 declare their consumer | `program/work.py`, `program/policies/sharding.py`, `program/build.py::_attach` |
| `schedule_edges` derived from the edge table; overlap re-parenting preserves `DepClass` | `program/build.py::_finish`, `::_reparent_dep` |
| `Op.succs` deleted (write-only mirror of `deps`) | `program/ir.py`, `program/build.py::_finish` |
| `from_timing` reads every `misc_metadata` key as REQUIRED (`REQUIRED_MISC_KEYS`) | `program/workload.py` |
| W1 grep gate extended to `dict.get(k, fallback)` in the producer seam | `tests/test_policies.py` |
| Non-GPipe regression probes (`_AscBackward`) for all four §4.3 dependencies | `tests/test_build.py` |
| Emitter regression probes (anchor on the SEND, cross-rank materialization, V9) | `tests/test_program_ir.py` |
| 8 dated amendment rows; **R5**, **D3**, **V9**, **V10**; `_prepare_execution_graphs` is a 2-tuple | `docs/rewrite/restructure/INTERFACES.md` |
| **A6** (Class A) and **D 11** (Class D) filed | `docs/rewrite/restructure/BUG_LEDGER.md` |

### Two numeric errors in the first revision, corrected

* *"rank 0: 14 -> 4 SEND, 14 -> 4 RECV"* on `train:flattened:dp2tp2cp2pp2mb2sp1` — the measured
  per-rank counts were **14 → 8**, and the per-bundle 1-byte totals were 90 → 48. (After the fixes
  the bundle totals are the same 90 → 48; the movement they cause is now −0.213%, not −5.945%,
  because the anchor edge is no longer lost with them.)
* the `inf:flattened:*` Δ% quotes were computed from `total_time`, which the golden records to **2
  decimal places**; every inference Δ% now carries that caveat and the per-bundle `max_sec` numbers
  are given alongside (§3.2).

**Not caused by the cutover:** `RAPID_ASTRA_CACHE_MODE=NO_CACHE` was already forced by the harness
(A1), and `tools/parallelism_sweep_cache.csv` is keyed on inputs only — delete it before trusting a
sweep.

---

## 8. Test inventory delta

`pytest --collect-only` over the gated suite is **1417** (1403 before this revision, 1598 at the
cutover; every removal was a comparison against a deleted target). This revision **adds** 14 tests
and removes none — plus one rewritten in place:

| Added | n | What |
|---|---:|---|
| `tests/test_build.py` — the four §4.3 dependency probes, parametrized over GPipe and `_AscBackward`, plus `test_r5_adds_nothing_under_gpipe` | 9 | **D3**: every real data dependency is stated by a rule, and survives a non-GPipe schedule |
| `tests/test_program_ir.py` — `test_cross_device_send_carries_every_same_rank_dep_of_the_transfer`, `test_cross_rank_dep_is_materialized_as_a_control_pair`, `test_uncarriable_cross_rank_dep_is_an_error_not_a_shrug`, `test_emission_postcondition_catches_a_lost_successor` | 4 | §4.1, §4.2, **V9**, **V10** |
| `tests/test_policies.py` — `test_no_silent_defaults_in_the_producer_seam`, `test_missing_misc_metadata_key_raises_and_names_the_key` | 2 | **W1**, the `dict.get` half |

`tests/test_build.py::test_succs_mirror_deps` is rewritten as
`test_deps_carry_every_edge_exactly_once` (the mirror it asserted no longer exists; the property —
one edge structure, no duplicate edges — is kept and strengthened).
