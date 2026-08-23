# QIF FWS-CIM system — plan of plans

Status: draft for George's review, 2026-08-23.
Branch: `fws_cim_dev` (based on main = the completed astra rewrite).

## 0. What this document is

This is the index and contract for the whole project, not a technical plan.
Each workstream (WS) below gets its own detailed plan at activation time,
written to `docs/qif/plans/WS<n>_<slug>.md` and reviewed before any build
starts — the same discipline as `docs/rewrite/`. This document fixes the
decomposition, the interfaces between workstreams, the acceptance gates, and
the priorities. When a detailed plan contradicts this document, this document
is updated first.

## 1. Mission

Implement the QIF proposal (`QIF_Proposal_final.docx`) with Rapid-LLM as the
base: a co-design framework for a heterogeneous, massively tileable chiplet
system — many identical 22 nm CTT-CiM chiplets holding stationary weights,
a few advanced-node digital chiplets running dynamic kernels — executing
billion-parameter ViT / SSM / VLM inference at the edge with deterministic
latency. The framework must answer placement, scheduling, sizing, and
co-design questions the OPTIMA model cannot, and must keep every claim
validated.

Priorities, per George (2026-08-23):

1. **Single-tenant system first.** One workload (possibly a chained VLM) on
   the full heterogeneous fabric, end to end.
2. **Multi-tenancy second** — but every interface built from now on must
   make it cheap to add (§3, the readiness contract).

## 2. Ground truth — what exists today

| Asset | Where | What it gives us |
|---|---|---|
| fws_cim device class | `cim_timing.py` + seams (commit `1b14d58`) | Validated analog + fabric laws: ViT, dense LLM, GQA, decode + KV stories, MoE experts-per-chip; 123 tests; 152-check validator |
| Program IR + policies | `program/` (the rewrite, now main) | Typed WorkloadSpec → placement → schedule → Program → consumers; placement/schedule as swappable policies; emission + deadlock discipline (dlsim) |
| DSE seed | `tools/fws_cim_dse.py` | Closed-form sweep, Pareto front, constraint reporting, `--emit-config`/`--verify` round trip |
| Standing gates | `tests/` | 221 golden-equivalence specs; GPU bit-identity baselines; physics-audit precedent |
| OPTIMA reference | `/app/nanocad/projects/cim_ctt_big_optima/perf_model/` | Recorded ViT targets (matched); **Mamba2 prefill+decode model** (not yet absorbed); multi-tenant area pools; ScaleSim recordings |
| Accuracy assets | `/app/nanocad/projects/cim_ctt_big_arch/` | ADC-bit / MX (microscaling) QAT studies — the accuracy surfaces WS6 imports |
| Heterogeneity sketch | `origin/heterogeneous` branch | device_profiles schema, per-instance GEMM re-pricing — concept salvage only (built on the deleted legacy flattener) |
| Weight-streaming baseline | A100 paths + validator rows | The proposal's Figure-1 contrast, for free, hardware-validated |

## 3. The shape of the build

Five standing rules. Every detailed plan inherits them.

1. **One law module.** Every physical timing/energy/area law lives in the CIM
   law layer (today `cim_timing.py`; WS3 may grow it into a `cim/` package).
   Consumers — reports, DSE, the Program IR — call it. No law is ever
   duplicated at a consumer. This rule already caught real bugs; it scales.
2. **Two-speed evaluation.** Tier F: closed-form steady-state laws — the DSE
   inner loop, sub-second. Tier S: Program-IR event-level simulation — owns
   contention, pipeline fill/drain, tails, and determinism claims. Standing
   cross-check: on any workload both tiers can express, they agree within
   0.1% (the existing DSE `--verify` pattern, generalized). This two-tier
   structure is the framework's defining design idea.
3. **Mapping is a policy pair.** The proposal's "compiler" is a placement
   policy plus a schedule policy over the Program IR — exactly what the
   rewrite made first-class. No mapping logic hides inside reports or laws.
4. **Gates before growth.** Every workstream lands with its own validator
   appendix and tests. GPU paths stay bit-identical forever. Physics audits
   (arithmetic-peak style) repeat at every law change.
5. **Everything speaks sets** — the multi-tenancy readiness contract:
   - Every new interface takes a *set* of workloads/tenants; the singleton is
     a degenerate case, never a separate code path.
   - Shared resources (digital pool, links, state buffers) are modeled as
     pools with named owners from day one; pool size 1 reproduces today's
     numbers bit-identically.
   - Arrays and weight shards carry ownership ids (tenant, model, layer);
     placement is a relation, not an array index.
   - Wherever Tier S runs, timing outputs are series/distributions; scalars
     are projections of them — so tail latency later costs a report, not a
     refactor.
   - No law may silently assume the machine belongs to one model. Such
     assumptions are named validators that WS8 relaxes deliberately.

## 4. Workstreams

| WS | Name | One-line mission | Depends on | Priority |
|---|---|---|---|---|
| 1 | SSM first-class | Mamba prefill + decode as a validated FWS workload | — | now |
| 2 | Workload sets | Chained VLM (ViT frames → SSM sequence) as one run | 1 | now |
| 3 | Platform model | Two-chiplet fabric: counts, links, shared digital pool, 3D variant, cost | — | now |
| 4 | Mapping compiler | Placement + schedule policies over workload set × platform | 2, 3 | next |
| 5 | Two-speed evaluation | Tier S event simulation joined to Tier F closed forms | 3 | next |
| 6 | Co-design DSE | Macro × chiplets × mapping × quantization sweeps; full energy | 3, 4 | continuous |
| 7 | Validation & calibration | Standing parity program; silicon-calibration seam | all | standing |
| 8 | Multi-tenancy | Residency, shared-pool scheduling, SLO tails | 2–5 stable | second |

### WS1 — SSM first-class

The proposal's token engine is an SSM; Mamba-Base, Mamba-Giant, and the VLM
language backend are headline rows. Rapid-LLM's SSM support is second-class
and the FWS passes skipped it. Build: Mamba(2) workload description (scan /
state update, conv, gating, projections), FWS pricing (weight GEMMs on the
analog law; state update as a new digital-fabric law class beside SA
attention and softmax lanes), decode without KV — recurrent state instead.
Reference and target: OPTIMA's Mamba2 model, the only decode OPTIMA ever had.
**Gate:** OPTIMA Mamba2 parity rows in the validator, same exactness
discipline as the ViT T1–T3 targets.

### WS2 — Workload sets (VLM chaining)

The proposal's flagship workload is 11 ViT frames feeding one 2048-token SSM
sequence. Build: a workload-set abstraction above today's one-config entry
point — a set of models, a dataflow between them, a request mix — and the
sequences-per-second (SPS) metric. This is also the load-bearing
multi-tenancy hook: a tenant set is the same abstraction with more members.
**Gate:** a chain of one model reproduces today's runs bit-identically; the
VLM chain produces an end-to-end SPS figure with per-model occupancy.

### WS3 — Platform model

From device-class to platform: a chiplet inventory (N analog CTT chiplets, M
digital chiplets — exactly two chiplet types, the proposal's cost story),
package-level links, the digital fabric as a **shared pool** rather than a
per-chip sidecar, the 8-tall 3D-stack variant as an area transform, and a
cost model rewarding replication. Arrays carry ownership ids (§3.5).
**Gate:** a pool of size 1 with today's geometry reproduces current fws_cim
numbers bit-identically; Table-1-shaped chiplet counts are derivable from
the schema.

### WS4 — Mapping compiler

Grow `layers_per_chip: auto` + DSE selection into explicit policies: stage
boundaries, weight-shard→tile assignment (tp/pp/ep as placement onto analog
chiplets), digital-pool scheduling, bandwidth/buffer feasibility with named
violations. Interface fixed from day one: (workload set, platform) → mapping
— even while the only supported set is a singleton. Heuristic first; search
sophistication is a later, separately planned upgrade.
**Gate:** reproduces today's auto placement on singleton workloads; every
infeasibility carries a stage tag and message (the DSE convention).

### WS5 — Two-speed evaluation

Tier F already exists (CimDeviceModel); this workstream builds Tier S: the
platform mapping lowered to a Program and executed event-level, so pipeline
fill/drain, shared-pool contention, chained-model handoff, and tail latency
become observable. Reuses the rewrite's IR, schedule policies, and dlsim
discipline — this is where the rewrite investment pays off.
**Gate:** Tier S matches Tier F within 0.1% on every workload both express;
contention and fill/drain are visible only in Tier S, by construction.

### WS6 — Co-design DSE

Grow `fws_cim_dse.py` into the proposal's "comprehensive DSE": axes = macro
parameters (rows / cols / mux / ADC), chiplet counts and pool sizing, mapping
knobs, quantization format (W5A8 / W4A8 / MXFP4 — accuracy imported from
`cim_ctt_big_arch` as constraint surfaces, never re-simulated), 2D vs 3D.
Objectives: TOPS/W, TOPS/mm², FPS / tok/s / SPS, cost. Includes the energy
completion program: the PARTIAL label goes only when the component library
(fabric, SRAM, helpers, ADC — Samyak's macro data) lands.
**Gate:** the proposal's Table 1 configurations reproduced in-framework with
reconciled, disclosed deltas; every sweep still verifies against a run.

### WS7 — Validation & calibration (standing)

The standing program: OPTIMA parity (ViT now, Mamba2 with WS1), GPU
bit-identity, the A100 weight-streaming contrast, physics audits at every law
change — plus a calibration seam so measured silicon (macro tapeout, then
system silicon in 2027) replaces simulated macro parameters without touching
laws. **Gate:** every push green; the seam demonstrated by swapping one
parameter set end to end.

### WS8 — Multi-tenancy (second priority, explicit)

Deferred scope: multi-model residency on multiplexed arrays, cross-tenant
digital-pool scheduling, deterministic tail SLOs, admission/what-fits
questions. Not deferred: the hooks in §3.5, which WS1–WS5 must honor and
WS7 must test (pool-of-1 and singleton-set degeneracy checks are standing
gates from day one). **Entry criterion:** WS2–WS5 stable on the
single-tenant VLM. The detailed plan is written then, not now.

## 5. Sequencing and the QIF timeline

Now → late Fall 2026 (proposal: "system pipeline for large workloads,
scheduling, weight-to-array allocation; macro DSE"):

1. **WS1 ∥ WS3** — independent, start both. Mamba parity is the single
   highest-information next step; the platform schema unblocks everything.
2. **WS2** once WS1 prices SSM stages — first end-to-end VLM SPS number.
3. **WS4 + WS5** — mapper and Tier S together (they co-define the IR
   lowering); WS6 grows an axis whenever its workstream lands.

Winter 2027 (proposal: "finalized system, reproducible tool flows"):
WS8 build-out, consolidation, and the reproducibility pass. The proposal's
Spring-2026 "system simulator + analog model integration" milestone is
effectively delivered by the fws_cim passes; we are building the Fall
milestones now, which matches the calendar.

## 6. Non-goals

- **Training** — physically excluded by FWS; permanent.
- **Accuracy simulation** — imported from `cim_ctt_big_arch`, never rebuilt.
- **Cycle-accurate NoC/NoP simulation** — link-level analytical models unless
  a validation failure forces more fidelity.
- **Runtime software for real silicon** — this is a modeling and co-design
  framework; the "tool flow" deliverable is reproducible studies.
- **New GPU serving features** — A100 paths are frozen baselines.

## 7. Open questions for George

1. **Routing fidelity.** The proposal says "routing plan that respects
   bandwidth and buffering." Are analytical per-link p2p models enough
   (current stance), or does the package need an explicit NoP model?
2. **Thermal.** Motivation only, or a modeled output (per-chiplet power
   density)? The `george_thermal_26` work suggests you may want the latter.
3. **Transformer-LLM decode/KV.** Keep as a maintained contrast axis (my
   recommendation — it is the argument *for* the SSM choice) or freeze?
4. **Quantization × timing.** Does MXFP4 change analog timing (bit-slice /
   `slice_cycles` interaction) or is it accuracy-and-density only?
5. **Mamba2 validation targets.** Which OPTIMA Mamba2 recordings exist, and
   for which configs? WS1's gate needs them enumerated.
6. **Table 1 provenance.** Which runs produced the proposal's Table 1, so
   WS6 can reconcile rather than guess?
7. **Digital chiplet scope.** Fixed-function engine (SA + softmax + state
   update, priced by fabric laws) or programmable (needs a kernel cost
   model)? This decides how WS1 prices the scan.

## 8. Immediate next moves

1. George answers §7 (items 5–7 block WS1/WS3 detail plans; 1–4 can wait).
2. Write `docs/qif/plans/WS1_ssm.md` and `WS3_platform.md`; review; build.
3. Recon pass over OPTIMA's Mamba2 model and recordings (read-only, scoped)
   to enumerate WS1's validation targets before the plan is written.
