# Student projects on the restructured `program/` core

Nine projects, expanded from the table in `PLAN.md` §"Student projects the restructure enables"
into briefs someone can start from on day one.

Every project has the same shape, and that shape is the point of the restructure: **you implement
one named object behind a stable protocol, and the existing gates prove you broke nothing.** You do
not edit `build()`, you do not edit `et_emit.py`, and you do not touch the timing or byte math in
`train_timing.py`. If your project seems to require any of those, that is a finding — write it up,
because it means the abstraction boundary is in the wrong place.

Read first: `docs/rewrite/CONTEXT.md` (§"Current architecture"),
`docs/rewrite/restructure/INTERFACES.md` (normative — §4.1–§4.6 are the dependency rules),
`docs/rewrite/TESTING.md` (how to run the gates).

---

## How much work is this, honestly

Before the restructure, an adversarial audit ran four extension exercises against the *legacy* code
and found the "new" core was no cheaper to extend: ratio ≈ **0.9–1.0 : 1**. The restructure was
then judged by re-running those same four exercises as **executable experiments**, not paper plans.
The measured extension ratios (new LOC ÷ what the same extension cost against legacy) are:

| exercise | ratio | what that means for you |
|---|---:|---|
| **flattened MoE** | **0.03** | Effectively free. Zero edits to `program/`; it fell out of granularity-as-a-parameter. |
| **native 1F1B** | **0.25** | **151 LOC in one new file**, validated end-to-end during the audit. |
| **ZeRO-3 prefetch depth** | **0.3** | `prefetch_depth` is already a constructor field on `ZeRO3`. |
| **a new parallelism axis** | **0.43** | **The weakest boundary.** No `AxisSpec` registry exists; axes are still named in several places. |

Two honest caveats, because a ratio is not a schedule:

1. **A low ratio measures the code you must write, not the work you must do.** Every project below
   is dominated by *validation* — building the evidence that the number your policy produces is
   right — and validation is not in the ratio. Budget accordingly: for most of these, the
   implementation is a week and the validation is the term.
2. **0.43 is a warning, not a bargain.** If your project touches the axis vocabulary
   (`program/types.py`, `groups.py::_spanned_axes`, `layout.py::CANONICAL_AXES`), expect to spend
   real time on plumbing that the restructure did *not* generalize. That is stated in
   `ext_new_axis.md` and it has not changed.

---

## Project 1 — Native interleaved / 1F1B pipeline schedule

**Goal.** Replace the closed-form bubble multiplier (`RunPolicy.interleave_scale`,
`program/workload.py:712`) with a real schedule: virtual stages, a non-contiguous layer→stage map,
and a 1F1B slot order. Today interleaving is *not in any graph* — it is a scalar correction applied
to a simulated total, so it cannot interact with communication, memory, or overlap.

**Interface you implement.** `program/schedule/onefonebee.py`, a new file implementing
`program.schedule.policy.SchedulePolicy`:

```python
class SchedulePolicy(Protocol):
    name: str
    def layer_assignment(self, fw: FrozenWorkload) -> LayerAssignment: ...
    def schedule(self, fw: FrozenWorkload, work: WorkSet) -> Schedule: ...
```

`LayerAssignment.explicit(stage_of_layer, num_stages)` already accepts an **arbitrary** map — that
generalization from `layers_per_stage` counts to a map was done *for this project*
(`schedule/policy.py:105-160`). `Schedule` enforces its own invariants in `__post_init__`: slot
indices dense and unique (**S1**), every `WorkItem` appearing exactly once (**S2**), the slot set a
permutation of `work.items` (**S3**). You also declare `Schedule.implied_deps` — the per-device
serialization the pattern implies — and `build()` materializes each one that is not already
transitively implied (R3, invariant **D1**).

**Read the module docstring at `program/schedule/policy.py:44-63` before anything else.** It states
exactly why this is a new `SchedulePolicy` and nothing else: `Placement.stage_of` reads the map per
layer; R2 compares *devices* of consecutive layers, so a stage revisiting a device is an ordinary
same-device link; R3 projects `Schedule.slots` per device, so a device appearing twice just has a
longer projection. The one place GPipe's shape leaks — `GradAccumPolicy`'s "`b == 0` is the last
microbatch" — is already a data field rebound through `GradAccumPolicy.for_schedule` →
`Schedule.last_microbatch_of`, computed from slot order.

> **This project is NOT "make it work at all".** A **151-LOC reference implementation in one new
> file** was written and validated end-to-end during the restructure audit — it built, emitted,
> passed `dlsim`, and produced a plausible bubble reduction. That is where the 0.25 ratio comes
> from. Your project is **"make it production-quality and validate it against measured hardware"**:
> the reference has no interleaving-degree parameter sweep, no memory-peak story (1F1B changes peak
> activation memory, which is half the reason anyone uses it), no comparison against the closed-form
> multiplier it replaces, and no validation against a real trace. Those four things are the work.

**Acceptance criteria.**
- The whole golden gate stays **green with an empty ledger** — your policy is opt-in, so GPipe runs
  must be bit-identical. This is the hard constraint: `GPipeSchedule` is the default and nothing you
  do may perturb it.
- Your policy passes the same non-GPipe probes GPipe does: `tests/test_build.py`'s §4.3 dependency
  tests are parametrized over `_AscBackward` precisely so a schedule-order change cannot silently
  drop a dependency. Add your policy as a third parameter and they must pass unchanged.
- `dlsim` completability and the p2p tag bijection on every emitted bundle (T3).
- **The real deliverable:** a delta table for a 1F1B-vs-GPipe *and* 1F1B-vs-`interleave_scale`
  comparison, plus peak-memory numbers from `memory_sim`, plus one validation point against
  measured hardware in the T5 harness.

**Difficulty: L.** Hardest part is not the schedule — it is proving the bubble and memory numbers
are right.

**First step.** `./.venv/bin/python -c "from program.schedule import LayerAssignment;
print(LayerAssignment.explicit([0,1,0,1], num_stages=2).contiguous_layers())"`. Then copy
`program/schedule/gpipe.py`, rename the class, and make `layer_assignment` return an interleaved
map with `schedule()` unchanged. Build one spec and diff the Program against GPipe's: everything
that changes at that point changes *because the map changed*, which is the cleanest possible
introduction to the whole core.

---

## Project 2 — Fast surrogate simulator

**Goal.** `equiv/dlsim.py` today answers a boolean: *does this bundle complete?* Give it a clock and
a communication cost model so it answers *how long does it take?* — then quantify its accuracy and
speedup against the real AstraSim binary across the whole golden matrix. AstraSim dominates every
sweep's wall time; a surrogate within a few percent unlocks Project 3.

**Interface you implement.** A consumer over the IR. Two existing pieces do most of the work:
`equiv/dlsim.py::BundleSim` already replays a bundle under the exact AstraSim scheduling contract
(one compute + one comm slot per rank, lowest node id first, RECV never occupying a slot,
per-communicator stream matching — all of it verified by experiment and written down in
`CONTEXT.md` §"The AstraSim workload contract"). And `program.analytic_sim.comm_durations(program,
network_model, interconnect_params)` is already a free function returning per-uid comm seconds. You
are wiring an existing cost model into an existing replayer.

**Acceptance criteria.**
- `dlsim`'s existing boolean behavior is unchanged — `test_contract` must stay green on every spec.
  Add the clock; do not perturb the causal replay.
- An accuracy report over the matrix: per spec, surrogate makespan vs the pinned
  `astra_times[bundle].max_sec` **and** vs the pinned `per_rank_sec` vector. Report per-rank error,
  not just makespan error: the rebaseline showed per-rank movement running ~an order of magnitude
  larger than end-to-end movement (`REBASELINE.md` §3.1), so a surrogate that matches makespan while
  getting the rank distribution wrong is a surrogate that will mislead Project 3.
- A measured speedup, run with `RAPID_ASTRA_CACHE_MODE=NO_CACHE` on both sides.

**Difficulty: M–L.** The replayer and the cost model both exist; the modelling judgement is in
congestion and overlap, which is where your error will come from.

**First step.** `./.venv/bin/python -m equiv.dlsim --help`, then run it on one persisted bundle
directory and read `BundleSim`'s issue loop. Add a `time` field to `_Op` and make the makespan the
max finish time with all durations set to their measured values from the ET — that reproduces
AstraSim's answer *exactly* on the trivial cases and tells you where the real work is.

---

## Project 3 — Auto-parallelism search (capstone)

**Goal.** Search the configuration space `(tp, cp, ep, pp, dp, zero level, recompute, overlap
fraction, schedule)` for the best predicted time subject to a memory cap, using Project 2 as the
inner loop.

**Interface you implement.** No new protocol — you *compose* the existing L1+L2+L3 policy objects.
The entire search space is already constructor arguments to one function:

```python
build(fw, granularity=..., sharding=..., schedule_policy=..., recompute=...,
      routing=..., overlap=..., grad_accum=..., placement_policy=...)
```

Every knob is a named swappable object: `sharding_policy_for`, `recompute_policy_for`,
`overlap_policy_for`, `routing_policy_for` in `program/policies/`, `GPipeSchedule` in
`program/schedule/`. `tools/parallelism_sweep.py` is the existing (non-searching) sweep driver to
learn the harness from.

**Acceptance criteria.**
- Any configuration your search proposes must actually **build and emit**: run the top-k through
  `build()` + `et_emit` + `dlsim` and require completability. A search that proposes unbuildable
  configurations is not a search.
- Validate the top-k against the real AstraSim binary, and report surrogate-vs-real ranking
  agreement (not just error) — a search only needs the *ordering* to be right.
- Respect a memory cap using `memory_sim.simulate_memory`, and say what the cap does to the answer.
- **Delete `tools/parallelism_sweep_cache.csv` before trusting any sweep.** It is keyed on inputs
  only and will not self-invalidate (`PLAN.md` §"Stale-cache trap").

**Difficulty: L (capstone).** Depends on Project 2 existing, or on tolerating AstraSim's wall time.

**First step.** Enumerate a small legal space and just *build* every point, timing nothing: count
how many configurations `build()` accepts and how many raise. The failures are as interesting as
the successes — they are the shape of the feasible region.

---

## Project 4 — Overlap as first-class scheduling

**Goal.** Today overlap is a *fraction*: `AxisFractionOverlap` declares an `OverlapDecl(fraction,
anchor)` per comm key and `build()` realizes it by splitting a compute or a collective. That is a
heuristic. Replace it with explicit asynchronous communication: an `issue` op and a `wait` op as
separate scheduled work, so the overlap that happens is the overlap the schedule *achieves* rather
than the overlap a fraction *asserts*.

**Interface you implement.** `program/policies/overlap.py`:

```python
class OverlapPolicy(Protocol):
    name: str
    def declare(self, spec: CommSpec, fw: FrozenWorkload) -> Optional[OverlapDecl]: ...
```

`OverlapDecl` carries a fraction and an `OverlapAnchor` (`PRODUCER` for tp/tp_sp, `CONSUMER` for cp
— a table at `overlap.py:_DEFAULT_ANCHOR_BY_AXIS`, replacing two near-duplicate rewrite blocks in
the deleted `transforms.py`). Extending the declaration vocabulary to async issue/wait is a change
to `OverlapDecl` plus its realization step in `build()` (§4.5) — this is the one project that
legitimately touches `build()`, and you should agree the shape of that change before writing it.

> **Read this before scoping.** The audit found that **no golden spec sets any overlap fraction**
> (`dispatch.md`: `grep overlap equiv/configs.py` → nothing). The entire overlap path is therefore
> **ungated by the 42-spec matrix**. Your first deliverable is not the feature — it is *golden
> coverage for the feature you are about to change*. Add overlap rows to `equiv/configs.py`,
> capture them, and only then start.

**Acceptance criteria.**
- New golden specs with non-zero overlap fractions, captured and green, **before** any behavior
  change. This is a `check_golden_superset.py` situation: adding pins must not move pins.
- The existing matrix stays bit-identical (no spec sets overlap, so any movement there is a bug).
- Validation against measured overlap efficiency on real hardware — the T5 suites are the only
  tests with physical meaning.

**Difficulty: M**, with an L-sized tail if you take the golden-coverage work seriously. You should.

**First step.** `grep -n "overlap" equiv/configs.py` — confirm the gap yourself. Then add one
flattened spec with `tp_overlap=0.5` and run `equiv.capture --specs` on it.

---

## Project 5 — FSDP-style sharding with prefetch depth

**Goal.** Implement what PyTorch FSDP actually does — forward and backward prefetch with a
configurable depth, and the reshard-after-forward choice — as a fifth `ShardingPolicy` beside DDP /
ZeRO-1 / ZeRO-2 / ZeRO-3, then validate it against real FSDP traces.

**Interface you implement.** `program/policies/sharding.py`:

```python
class ShardingPolicy(Protocol):
    name: str
    def requirements(self, work: WorkItem, ctx: ShardingContext) -> Sequence[SyncRequirement]: ...
    def workload_requirements(self, ctx: ShardingContext) -> Sequence[SyncRequirement]: ...
```

`requirements` is called once per `WorkItem` and **must be pure and order-independent** (invariant
**K3**). You return `SyncRequirement`s carrying collective kind, axis, byte source, participants,
and a typed **attach mode** from a closed vocabulary (`before` / `after` / `overlap_with` /
`parallel_to(target, edge_class)`) — that vocabulary replaced the
`skip_non_comm_children` / `skip_comm_children` boolean escape hatches, and staying inside it is the
discipline that makes your policy composable.

Start from `ZeRO3`: `prefetch_depth: int = 1` is already a constructor field
(`sharding.py:502`), explicitly named there so this exercise is a constructor argument rather than a
code edit. It is what earns the **0.3** ratio. `ZeRO3` also demonstrates the two structural pieces
you will need: `_backward_host` (which work item hosts a prefetch that runs `depth` layers ahead)
and `SyncRequirement.consumers` resolved through
`ShardingContext.next_after(hosts, kind, direction)` over `Schedule.order()` — i.e. the prefetch
names its consumer *through the schedule*, so it survives a non-GPipe order.

> **Know about BUG_LEDGER A6 before you start.** The existing ZeRO-3 prefetch anchor is chosen by
> layer arithmetic and can land on **another device** (`REBASELINE.md` §4.2, §5.3). It is filed,
> deliberately unfixed, and it interacts with `analytic_sim._ROOT_COMM_IS_UNTIMED` (a stage's first
> gather legitimately becomes a program root under a correct anchor, and would otherwise be free).
> Fixing A6 as part of this project is legitimate and welcome — but it **moves predictions** and
> therefore needs its own delta table and owner approval. Do not fold it silently into a new policy.

**Acceptance criteria.**
- Golden gate green with an empty ledger: your policy is selected by `sharding_policy_for` only for
  new configurations, so every existing spec must be bit-identical.
- New golden rows for the FSDP configurations, captured as an additive superset.
- The `_AscBackward` probes in `tests/test_build.py` pass for your policy — a prefetch that only
  works under GPipe adjacency is exactly the defect R3/§4.3 was written to catch.
- Validation against measured FSDP traces.

**Difficulty: M.** The cheapest real modelling project in the list.

**First step.** `./.venv/bin/python -m pytest tests/test_policies.py -q -k zero3`, then instantiate
`ZeRO3(prefetch_depth=2)` and diff the resulting `SyncRequirement` set against `prefetch_depth=1`.
That diff is your project's entire surface area, made visible in ten minutes.

---

## Project 6 — Congestion- and topology-aware analytical collectives

**Goal.** The analytical evaluator converts bytes to seconds with ring formulas
(`NetworkModel.collective(kind, size_bytes, participants, ib, ll, ...)`). Go beyond that: model
topology (the Mesh2D specs already exist in the matrix), congestion, and contention between
concurrent collectives — then compare against AstraSim's congestion-aware backend.

**Interface you implement.** `program/analytic_sim.py::comm_durations` is already a pure function of
`(kind, size, participants, ib, ll)` per op, and its docstring says so explicitly — the legacy
recursion order is unobservable and is not reproduced. That purity is your seam: you are replacing
the `network_model.collective` call, not the evaluator.

**Acceptance criteria.**
- **The 10 fully-analytical specs are bit-exact today, to 17 significant figures** (`REBASELINE.md`
  §2.6 lists all 10 values). Your model must be opt-in, and with it off, those 10 numbers must not
  move by one ulp.
- Two preserved quirks are named in `program/analytic_sim.py` and are load-bearing:
  `_ROOT_COMM_IS_UNTIMED` (a comm op that is a program ROOT carries no time — turning it on moves
  `train:analytical:dp2tp1cp1pp2mb2sp0:zero3` by **+5.76%**) and "the memory replay times the
  pipeline-level collectives only". If your model changes either, that is a modelling change with
  its own delta table.
- A comparison against AstraSim's congestion-aware backend on the Mesh2D specs.

**Difficulty: M.** Well-bounded seam; the modelling is the whole difficulty.

**First step.** Read `comm_durations` (60 lines) and print its output for
`train:analytical:dp2tp2cp2pp2mb2sp1`. Every number your project changes is in that tuple.

---

## Project 7 — Memory-aware recompute policy search

**Goal.** Recompute is currently all-or-nothing (`NoRecompute` / `FullRecompute`). Minimize
predicted time subject to a peak-memory cap by choosing *which* layers rematerialize — selective
recompute, which is what production frameworks actually do.

**Interface you implement.** `program/policies/recompute.py`:

```python
class RecomputePolicy(Protocol):
    name: str
    def materializes(self, layer: LayerId, microbatch: MicroBatch, fw: FrozenWorkload) -> bool: ...
```

That is the whole protocol, and it is already per-`(layer, microbatch)` — `FullRecompute` just
returns `True`. A selective policy is a different return value. The two evaluators you optimize
against are `program.memory_sim.simulate_memory(program, memory_data) -> (finish_time, peak_gib)`
and `program.analytic_sim.evaluate(program, network_model, interconnect_params) -> float`.

**Acceptance criteria.**
- `NoRecompute` and `FullRecompute` behavior unchanged; the golden gate green with an empty ledger.
  T1 pins **per-GPU memory peaks**, so a memory regression is a gate failure, not a silent drift.
- A Pareto front (time vs peak memory) for at least one real model shape, with the two existing
  policies as its endpoints — if your front does not contain them, it is wrong.
- Recompute interacts with R2: `LAYER/FORWARD(b,l) → RECOMPUTE(b,l)` is a *stated* rule
  (`REBASELINE.md` §4.3). A per-layer policy must not perturb it.

**Difficulty: M.** Clean protocol, real optimization content, immediately useful.

**First step.** `./.venv/bin/python -m pytest tests/test_equiv_golden.py -q -k recompute`, then
write a policy returning `layer % 2 == 0` and read the peak-memory delta out of the run's memory
summary. You will have a two-point Pareto front within the hour.

---

## Project 8 — MoE routing and load imbalance from real traces

**Goal.** The MoE model is a one-hot hot/cold-rank imbalance factor. Replace it with a routing model
fit to measured expert-assignment traces: real token distributions, real capacity-factor drops, real
per-expert skew.

**Interface you implement.** `program/policies/routing.py`:

```python
class MoERoutingPolicy(Protocol):
    name: str
    def routing_axes(self) -> Tuple[AxisName, ...]: ...
    def hot_coords(self, coords: Coords) -> Coords: ...
    def join_token(self, coords: Coords) -> Tuple[Any, ...]: ...
```

`AxisRouting` is the single existing implementation covering both production modes (`hot_coords` =
coords with every routing axis zeroed; `join_token` = the coords on the non-routed axes). The byte
math is upstream in `train_timing.py` and stays there — `_moe_routed_tokens_per_expert` and
`_moe_expert_imbalance_factor` are the existing imbalance model, and `tests/test_moe_expert_imbalance.py`
already pins their behavior in 15 tests. Your policy shapes *placement and grouping*; if you also
need to change how many tokens an expert gets, that is a `train_timing` change and it needs its own
justification.

**Acceptance criteria.**
- The existing MoE golden specs stay green; `tests/test_moe_expert_imbalance.py` stays green.
- New golden rows for trace-driven routing, captured additively.
- Validation against the trace you fit to, held out properly — `validation_scripts/moe_ep_validation.py`
  is the existing MoE validation entry point.
- **Know the terrain:** `ext_moe_flat.md` records that the memory replay does **not** validate MoE
  comm topology (it reads no `comm_type`, no participants, no group), and that flattened MoE
  execution has its own history (`REBASELINE.md` §6: the FLAT MoE program is built for the memory
  replay but its group-order postcondition does not hold yet). Check the current state of flattened
  MoE in `equiv/configs.py` before assuming either way.

**Difficulty: M.** Bounded protocol; the difficulty is entirely in trace acquisition and fitting.

**First step.** `./.venv/bin/python -m pytest tests/test_moe_expert_imbalance.py -q`, then read
`AxisRouting` (55 lines). Print `hot_coords` and `join_token` for every device of an `ep=2` spec —
the routing model is a hundred lines of code and it will fit in your head today.

---

## Project 9 — New hardware or model family, validated

**Goal.** Add a device (or a model family) and close the loop against measured data: predicted time
and predicted memory within the existing error thresholds.

**Interface you implement.** Configuration plus the **T5 physical harness** — `test_func_test.py`,
`test_IMEC_*`, `test_koyeb_*`, and `validation_scripts/`. `PLAN.md` §"Gate redesign" is blunt about
why this matters: these *"are the only tests with physical meaning and are the truest 'predictions
must not move' contract."* Everything else in the gate proves the code is self-consistent; T5 is the
only thing that proves it is *right*.

Existing examples to copy from, in increasing order of ambition:
`validation_scripts/h100_testbench.py` (a device), `validation_scripts/nvidia_train_validation.py`
and `uci_train_validation.py` (measured-vs-predicted with thresholds),
`test_vit_modeling.py` / `test_mla_modeling.py` / `test_deepseek_v3_config.py` (a model family).

**Acceptance criteria.**
- Every existing T5 threshold unchanged. You add rows; you do not relax bounds.
- Your new configuration within a stated error threshold against measured data, with the
  measurement method written down.
- If your model family needs a new parallelism axis, read `ext_new_axis.md` first — that is the
  **0.43** ratio, the weakest boundary in the system, and the reason is that no `AxisSpec` registry
  was built. Budget for plumbing in `program/types.py`, `groups.py::_spanned_axes` and
  `layout.py::CANONICAL_AXES`. (Building that registry is itself a good, self-contained project, and
  it would turn 0.43 into something respectable.)

**Difficulty: S–M**, and by far the best first project on this list. It teaches the whole pipeline
end-to-end without requiring you to change any of it.

**First step.** `./.venv/bin/python -m pytest tests/test_func_test.py -q` (234 tests), then open
`validation_scripts/h100_testbench.py` and change one device parameter. Watch which thresholds move.

---

## Ground rules for all nine

1. **The golden gate is not negotiable.**
   ```sh
   env RAPID_ASTRA_CACHE_MODE=NO_CACHE \
       LD_LIBRARY_PATH=/u1/ee/karfakis/gcc-10.2.0/lib64:/app/nanocad/projects/personal/gkarfakis/anaconda3/lib \
       ./.venv/bin/python -m pytest tests/test_equiv_golden.py -q
   ```
   Green, with `tests/golden_equiv/bug_ledger.json` at `{"entries": []}`.
2. **Your policy is opt-in.** The default configuration must produce byte-identical output. If it
   does not, you have changed the model, not extended it.
3. **Declare before you land.** If your change is *supposed* to move a pinned number, put it in the
   T4 ledger first, write the delta table, get it approved, recapture, then prune the ledger back to
   empty. `REBASELINE.md` is the worked example of what a delta table looks like when it is done
   properly — including the part where the first two revisions were wrong and said so.
4. **Report per-rank, not just totals.** T2 gates `per_rank_sec` element-wise, and per-rank movement
   routinely runs an order of magnitude larger than end-to-end movement (`REBASELINE.md` §3.1:
   −12.554% on a rank where `total_time` moved −0.213%). A summary statistic can hide a real change.
5. **Adding pins is not moving pins — prove it.**
   ```sh
   cp -r tests/golden_equiv /tmp/golden_before
   ./.venv/bin/python -m equiv.capture --specs <your new specs>
   ./.venv/bin/python tools/check_golden_superset.py /tmp/golden_before tests/golden_equiv
   ```
6. **Never trust a cached AstraSim result.** Always `RAPID_ASTRA_CACHE_MODE=NO_CACHE`, and delete
   `tools/parallelism_sweep_cache.csv` before any sweep.
