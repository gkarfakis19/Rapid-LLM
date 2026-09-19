# QIF build — the decisions register

The binding record of George's decisions from the 2026-08-23 planning
sessions. Every superplan conforms to this document. If a plan needs to
contradict it, the plan says so explicitly and the contradiction goes to
George — it is never resolved silently.

## Mission

Implement the QIF proposal with Rapid-LLM as the base: a modeling and
mapping framework for a heterogeneous chiplet system — analog FWS-CIM
macro chiplets (CTT first, device-generic by design) plus shared digital
chiplets — running ≤7B-total-parameter edge inference (ViTs, LLMs, SSMs,
hybrids). The plan-of-plans board tracks progress; it is a plan George
reads, not a process that enforces anything.

## Architecture stance (the big three)

- **A1 — DAG-first. There is no period law.** The system builds the op DAG
  (the rewrite's program IR) with per-op durations from the macro laws and
  evaluates it fully analytically. AstraSim is OFF. The network assumption
  is fully-connected p2p with no congestion modeling. Contention and
  occupancy emerge from the DAG plus resource assignment — never from a
  closed-form spatial-pipeline formula. The pass-1/2 closed-form report
  survives only as a validation/legacy reference.
- **A2 — Heterogeneous device layer in the DAG executor.** Ops are placed
  on devices: analog macro devices and digital devices (per-macro pools
  and shared digital chiplets), QIF-proposal style. The executor prices
  each op on its assigned device.
- **A3 — The macro is the atomic resource; the tile is the allocation
  unit.** A macro's stored columns (cols × mux slots) can host tiles from
  the same layer, different layers, different experts, or different
  models. Mapping is the assignment of tiles to macros, macros to chips,
  and shards to groups.

## Decisions

Workloads:

- **D1** Support MAJOR MODELS WHOLESALE. Primitives (scan, SSD, delta
  rule, sliding window, conv, norms) may appear as an internal
  decomposition aid, never as the support target.
- **D2** Size cutoff: ≤ 7B TOTAL parameters. Total, not active — the
  binding constraint is area (weights own cells). Corollary: big-total /
  small-active MoE is exactly the wrong shape for FWS.
- **D3** Model window: the last ~15 months (≈ May 2025 – Aug 2026).
- **D4** ViTs = BERT-class = trivially supported; any attention flavor;
  not interesting; do not spend bandwidth.
- **D5** Pure transformer LLMs are IN. They will predictably lean on heavy
  digital co-compute; that is a finding the tool should expose, not a
  blocker.
- **D6** MLA is modeled, even though it is TP-hostile and its KV
  replication hurts area (our weakest metric). If it is ugly, show
  exactly how ugly.
- **D7** DiT/diffusion is OUT of phase 1 (even though VLA will want it
  later). Audio/speech is out.
- **D8** VLM chaining and VLA are ADDRESS-ONLY extensions, listed on the
  board, empty by design. VLA is qualitatively different — latency-first,
  closed-loop (one microbatch in the pipeline), aggressive sharing — and
  gets rethought when its time comes. Keep the core rewrite-free for it;
  spend no bandwidth now.

Macro and platform:

- **D9** No PWS (partial weight stationary) anywhere in phase 1.
- **D10** Banking/multiplexing is SPATIAL column allocation (A3), the
  core mapping concept. Allocation is user-specified in config,
  priced and validated by the tool, swept by the DSE. Automatic
  allocation inside the mapper comes later.
- **D11** Bit-slicing is natively supported and PRICED: low bits-per-cell
  makes an active column set yield partial results; full results compose
  via column sets and/or chained macros with digital shift-and-add trees
  that cost real area, time, and energy.
- **D12** Each analog macro carries a small digital pool (shift/add
  trees, activations, elementwise). It is SIZED so it never blocks the
  pipeline, and the derived sizing is REPORTED. It is not a DSE
  constraint for now; revisit only if sizing grows embarrassing.
- **D13** Attention and other act×act compute goes on the SHARED digital
  chiplet, not the per-macro pools.
- **D14** Device cards, not CTT hardcoding: a macro card = parameter set
  plus structural knobs (bits-per-cell, slicing, bank/mux depth, 3D
  height). ReRAM/MRAM are cards, not rewrites. Accuracy is never a
  factor anywhere (D23).

Serving and system:

- **D15** Decode serving stays simple: batched, same-length requests —
  each request prefills S tokens and decodes N. No sustained-vs-ceiling
  elaboration beyond what exists.
- **D16** PD disaggregation is FOLDED INTO MAPPING and is simple: a
  separate prefill system spec and decode system spec (own models, own
  pool counts), plus a handoff priced as bytes over the p2p law. No
  precision-conversion modeling (7th-order effect, out of scope).
- **D17** Network: fully-connected p2p, analytical, no congestion (A1).

Process and scope:

- **D18** Anti-overengineering is a standing directive. No lifecycle
  machinery, no gates-as-process, no ceremony. Plans are documents George
  reads; the board is a page with links and notes.
- **D19** Objectives stay simple: make the whole system work. No bespoke
  area/cost model. Paper framing is deferred entirely.
- **D20** Placement is LOGICAL only (macro → chip, chip → package graph).
  The atlas (viz) colors by parallelism group: tp / ep / pp.
- **D21** Validation stance per `docs/qif/plans/AUDIT_optima.md`: OPTIMA
  stage laws, array censuses, and single-model recordings are references;
  OPTIMA's mapper and multi-tenant layer are NEVER targets. Honesty
  invariants: one accounting per metric; constraint relaxations must be
  disclosed in the artifact; chips are integers; every macro has a named
  owner.
- **D22** Keep the far future cheap: no exclusive-ownership assumptions
  in resource abstractions; latency composition stays honest in the DAG
  (fill/drain is naturally representable) so VLA later adds content, not
  rewrites.

## The superplan set

| ID | File | Title | One-line scope |
|---|---|---|---|
| P1 | `plans/P1_model_matrix.html` | The Model Matrix | Which models we support wholesale (D1–D8), their architecture facts, and the config-schema gaps |
| P2 | `plans/P2_macro_resource_model.html` | The Macro Resource Model | Device cards, tiles, mux/banking, bit-slicing + priced reduction, per-macro digital pool, occupancy semantics (A3, D9–D14) |
| P3 | `plans/P3_mapping.html` | Mapping | Tiles→macros→chips→system; tp/ep/pp shard groups; heterogeneous op placement; PD disagg; DAG output (A1–A3, D16) |
| P4 | `plans/P4_evaluation.html` | Evaluation | Pricing a mapped DAG analytically; metrics; reports; what survives of the closed forms (A1, D15) |
| P5 | `plans/P5_system_atlas.html` | The System Atlas | The interactive mapping visualization — package→chip→macro, colored by shard group (D20) |
| P6 | `plans/P6_validation.html` | Validation | What is a target vs provisional; the bridge gate from closed forms to DAG; honesty invariants (D21) |
| — | (board only) | VLM chaining / VLA flow | Named, empty, future (D8) |

## Assets

- This repo (`fws_cim_dev` branch): `docs/fws_cim/` (device class, pass
  1+2), `docs/rewrite/` (program IR), `cim_timing.py`, `tools/fws_cim_dse.py`,
  `validation_scripts/validate_fws_cim_vs_optima.py`.
- `docs/qif/plans/RECON_optima.md` and `AUDIT_optima.md` — what OPTIMA is
  and what to trust.
- OPTIMA itself (read-only, bounded reads):
  `/app/nanocad/projects/cim_ctt_big_optima/perf_model/`.
- QIF proposal: `/app/nanocad/projects/ispass_deepflow/deepflow_astra_dev/Rapid-LLM/QIF_Proposal_final.docx`.

## Adjudications (2026-08-23, evening — George delegated best-guess authority; only non-trivial items go back to him)

- **D23 (GEORGE, HARD): Accuracy is NEVER a factor.** This is a performance
  model only. No accuracy measurement, import, or reporting anywhere.
  This voids D14's accuracy-import clause, P2's card accuracy field, and
  P6's accuracy row.
- **ADJ-1 (P1):** headline model = Granite-4.0-H-Tiny; Qwen3.5-4B second P0.
- **ADJ-2 (P1):** D2 counts RAW params → Gemma 4 E4B out, E2B in. Hard 7B
  cutoff: RWKV-7 7.2B and LFM2.5-8B are reference rows, not supported.
  LLaDA-class diffusion LMs out (D7 spirit). MLA carrier = MiniCPM3-4B.
- **ADJ-3 (P1):** short depthwise conv runs on the per-macro digital pool.
  Depth-shared weights (Zamba2) are expressed in the model config (P1 owns).
- **ADJ-4 (P2):** active-column-set pricing ADOPTED (full-occupancy identity
  must hold; parity gates unmoved). Slicing is per-card opt-in (present iff
  bits_per_cell < weight_bits). Mux slot = smallest allocatable unit. Pool
  revisit threshold = 1/5 of served macro footprint (disclosure only).
  Second card: schema slots only, no invented numbers. The shared digital
  chiplet card is a P2 card (audit resolution stands).
- **ADJ-5 (P3):** a chip hosts many macros; chip = chiplet = package unit.
  pp is an independent annotation (chip index is not implicitly pp).
  Shared-chiplet count is a config input, with a derived suggestion
  reported. PD = two separate inventories. New `mapping:` config block.
  Unowned columns are legal and reported.
- **ADJ-6 (P4):** headline = tokens/s at the decode terminal (requests/s
  also printed); the ceiling/sustained pair retires from the DAG report
  (legacy only). Per-step decode series up to a bound, then window +
  disclosed extrapolation. Energy: per-component coverage labels. Legacy
  closed-form report reachable via the validation script only.
- **ADJ-7 (P5):** separate atlas.json; fixture = Llama2-7B tp=2 decode;
  P3 emits primary_group; embedded-blob single file is first-class.
- **ADJ-8 (P6):** bridge = ViT tiers + Llama7b + MoE smoke at 1e-3
  (discrete quantities exact); on disagreement the DAG wins and the closed
  form is annotated; the closed-form path stays in tree, frozen, to hold
  the bridge.

## Amendment (2026-08-24, George — the folding pivot)

- **D24 (GEORGE, HARD): Folding is the core.** Spatial-vs-temporal folding of
  tensors onto fixed macros is a per-tensor MAPPING decision, and choosing it
  per model is the project's center. This promotes the mapping optimizer from
  "future search" to NOW, amending D10's "automatic allocation later" and
  D19's "objectives simple" for this scope only. Plan: `plans/P7_folding.html`.
- **D25 (GEORGE, HARD): Decode only for P7.** Prefill folding is out (SSD /
  concurrent-layer interplay); every P7 artifact states the decode-only
  assumption. Serving regime stays D15.
- **D26: The DOA register binds.** P7's dead-fold list (slicing x folding,
  replication, non-canonical walks, prefill folds) is refused BY NAME in
  code — machinery for a dead fold is a conformance violation, not initiative.

- **D27 (GEORGE, HARD): NEVER waste weight space.** Every mux bank holds real
  weights; dense packing is the invariant; waste exists only at
  dimension-mismatch remainders and is a reported metric. Analog macro count
  = the global cell floor. Replication that idles or duplicates for
  throughput decode cannot use violates this.
- **D28 (GEORGE): no safety margins; the P7 objective is the (total area,
  decode throughput) Pareto frontier, with SYSTEM SIZING (digital chiplet
  count, engine widths, pool widths) as first-class sweep axes — low
  utilization anywhere is a provisioning bug the DSE must expose, not a fact
  to accept.

## Amendment (2026-08-24 evening, George — the filled-pipeline pivot)

- **D29 (GEORGE, HARD): the pipeline is ALWAYS FULL.** Decode = independent
  streams staggered across PP stages, one token exits per beat
  (throughput = 1/beat; per-stream rate = 1/(D·beat)). LOCAL BATCH IS ALWAYS
  1 — batch size does not exist as a concept; analog macros fire at M=1 and
  attention/scan are per-stream. Resident streams D = stage count. EVERY
  stage holds all D streams' state/KV for its layers: state memory scales
  with D and is a first-class reported quantity and feasibility check.
  Supersedes D15's batched-lockstep semantics for fws_cim. Consequence:
  cross-STAGE bank sharing now contends every beat (priced by the timeline);
  within-stage sharing stays serial-free.
- **D30 (GEORGE): embedding and lm_head are DROPPED entirely** —
  disable_embedding_unembedding is the standing default for all QIF runs;
  no endpoint arrays, no endpoint stages, no notes.
- **D31 (GEORGE): the scan/vector engine is NEVER a swept or declared free
  knob.** Its sizing is DERIVED: start from the OPTIMA engine work model and
  scale units to achieve full utilization at the pipeline beat; report the
  derived size. vector_lanes as a design point is retired (an explicit
  override stays possible but rides a disclosure).
- **D32 (GEORGE): digital area/energy come from the measured OPTIMA
  synthesis library** — cim_ctt_big_optima/perf_model/configs/
  digital_hw_components_22nm.yaml (generated from rtl/reports_22nm_1000_efh):
  FP_ADD/FP_MULT/BF16_EXP/RECIP/systolic/buffer/register blocks composed
  into engine area and power, first-order. Declared placeholders are
  replaced by block compositions.

- **ADJ-9 (2026-08-25, adjudicated from George's own words; he may overrule):
  D31-v2 — derive the engine to the ANALOG floor.** "Full util" means the
  analog side: the digital engine is sized UP until the analog m-pass time
  is the beat-setter (digital per-stage time <= analog stage time), not the
  smallest engine that keeps itself busy. Rationale: the audit measured
  +27% tokens/s for +0.78% area on Granite — digital silicon is nearly free
  against the Invariant-W analog floor, so the machine must never be left
  digital-bound by a minimal derivation. The Wave-D demo sweep's
  vector_lanes axis retires when this lands.

- **ADJ-10 (2026-08-25, adjudicated from George's doctrine; he may overrule):
  ALL composable digital engines derive to the analog floor.** ADJ-9's scan
  derivation left the attention SA fabric as the beat-setter (declared
  geometry, underived). Extension: every digital engine whose width is a
  composition of MEASURED synthesis blocks derives up until the analog
  m-pass binds — SA fabric by integer copies of the measured GEMMINI 32x32
  block (rows/cols stay as measured; inventing geometry stays refused),
  softmax by its measured-lane composition, scan per ADJ-9, pool per D12.
  Whatever binds after that must be a REAL limit (unscalable declared
  geometry or a bandwidth term) and is named in the artifact. The frontier
  sweeps only knobs with genuine physical trades left.

---

## ADJ-11 — A machine is never refused for needing memory. The state store is SIZED, and AREA is the one budget.

**What changed.** The per-stage residency cap (`mapping_dse.max_stage_state_bytes`)
is REFUSED BY NAME. It used to refuse a stage plan whose resident state exceeded
a declared on-chip tier, which is the wrong question: a decode pipeline's state
is not a constraint to check, it is a store to BUILD.

**The law.** `compose_state_sram` (cim_timing) sizes the store to the state a
mapping actually holds, in integer copies of the measured 22nm SRAM macro
(`SRAM_MACRO_256x1040`: 256 b x 1040 words = 33 280 B, 53 006.39 um2, copied
from OPTIMA's `sram_config.yaml` with its provenance, same convention as D32).
The rounding up to whole macros is physical block granularity, not a margin
(D28). Only the macro's LEAKAGE power is carried: the source measures read and
write energy per word, and this repo has no access law to apply them with, so
the store's dynamic energy is a NAMED GAP rather than an estimate.

**Where it lands.** `silicon.state_sram_silicon_mm2` is a fourth named term in
the machine's silicon, inside `digital_silicon_mm2` and inside
`total_silicon_mm2`. `state_residency.verdict` is now `sized` — never
`VIOLATED` — and each stage row carries `store_macros`, `store_area_mm2` and a
`capacity_bytes` that IS the store built for it. The declared tier
(`tech_param.SRAM-L2.size`) is still reported as `declared_onchip_tier_bytes`,
but it gates nothing.

**What refuses a machine now.** `mapping_dse.max_silicon_mm2`, declared at
20 000 mm2 on every sweep. It is an appliance-level statement about the box the
system ships in, deliberately the SAME number for every model, and it is
checked ONCE — after pricing, because both the derived digital engine and the
sized store exist only then. A point refused by area is still fully priced, so
it still reports the throughput it would have delivered and still draws on the
design-space plot; the refusal names all four silicon terms.

**Measured consequence.** Granite-4.0-H-Tiny: 12 940 SRAM macros = 686 mm2
(+5.5% silicon). Falcon-Mamba-7B: the finer stage plan holds 4x the state
(427.5 mm2 vs 106.9 mm2 of store) and delivers 4x the throughput
(241 362 vs 60 340 tokens/s) — a trade that was previously REFUSED and is now
priced. Falcon-Mamba at `arrays_per_chip` 880 or above exceeds the area budget
outright, refused with its terms named.

## ADJ-12 — The two simplest workloads run: a ViT encoder and a pure-SSM stack.

A suspicious mapping number on a hybrid could come from either mixer, so the
debugging shapes come first.

**ViT** (`configs/hardware-config/fws_cim_vit.yaml`, the OPTIMA T1 machine plus a
`mapping:` block): an encoder, uniform layers, NO KV and no decode step. It runs
under `regime: lockstep` because the filled pipeline is a decode regime with
nothing to stagger, and it refuses `packing: dense`, which is decode-only.
ViT-Huge-story at seq 64: 1 chip, and the slot count is DERIVED from the
placement (386 tiles, 386 slots) rather than the 400 first recorded here -
those 14 unowned slots are exactly the waste ADJ-13 exists to remove. Read
the shipped sweep for the current throughput.

**Pure SSM** (`configs/hardware-config/fws_cim_mamba.yaml` + the new
`falcon_mamba_7b_decode_inf.yaml` D29 twin): 64 Mamba-1 layers, no attention
block at all. It used to CRASH rather than refuse — three sites read attention
attributes unguarded (`CimModelParams.from_model`, and two in
`base_timing.TimeCalculation.__init__`). A model with no attention has no heads:
the count is 0 and every attention law is already gated on a layer declaring an
attention block. REGENERATED after ADJ-13/ADJ-14 and the conv fixes: read the shipped sweep
(docs/qif/dse/falcon_mamba_7b_frontier/dse_report.json) for the current
selected point. The number first recorded here (241 362 tokens/s at 14 788
mm2) was measured on a machine whose chip declared its capacity and whose
every layer carried a phantom FFN; both are gone, so it describes nothing
that exists and is kept only as the reading it was.

## ADJ-13 — The machine sizes itself. Chip capacity and chip counts DERIVE; the sweep keeps only the free choices.

**The error this fixes.** The sweep declared `arrays_per_chip` (a chip's macro
slot count) and `shared_chiplets` (how many digital chiplets exist) as AXES, so
a human had to guess them and the frontier filled up with points that were not
designs at all. Under the requirement, the assembly check refuses the point and
it teaches nothing. Over the requirement, the machine enumerates empty slots and
is CHARGED SILICON for them, so the curve bought area with no throughput and
called it a design point. Granite's shipped config declared 640 slots per chip
and needed 556: 84 empty slots on each of 10 chips, 1697 mm2 of silicon — 13%
of the machine — bought for nothing and drawn on the frontier as if it were a
choice.

**Chip capacity is arithmetic, not a choice.** A chip offers exactly the macro
slots the layers it holds need. The sweep places the layers once against a
capacity that cannot bind (`_CAPACITY_PROBE`), reads what the busiest chip
needed, and builds the chip to it. What stays free is the PARTITION —
`layers_per_chip` / `layers_per_stage` — which is a real design choice with a
real trade: a finer partition raises the resident stream count D and the
throughput with it, and makes every stage hold more state.

**The digital chiplet count is what makes the pipeline balanced, and it
derives from the stage plan.** One chiplet per pipeline stage, each with its
engines sized up to that stage's own analog pass (D31/ADJ-9/ADJ-10). That is
the balance condition, and it is not a knob: the PARTITION decides it, so
declaring it separately could only starve the pipeline or buy idle chiplets.

**The packing question is REPORTED, not applied — and the reason is a caught
mistake.** How few chiplets could carry this work if the stages time-shared
them inside one beat? First-fit over the stages sorted heaviest-first says
FOUR for Granite (three chiplets at 95.7% of the beat, one at 75.6%) where the
machine builds ten. The first implementation APPLIED that number and charged
area for four chiplets. It was wrong: the engine widths are derived per STAGE,
so a chiplet serving three stages would have been sized for one of them and
silently under-provisioned. `--verify` caught it — it re-ran the emitted config
and measured a machine 3.5x slower than the sweep claimed (Qwen: 47148 vs
13314 tokens/s). Sharing a chiplet across stages needs an engine sizing that
knows it is shared; until that law exists, the row says what the headroom is
and the machine does not take it.

**Measured consequence.** Granite-4.0-H-Tiny declared 640 macro slots per chip
and needs 556, so 84 slots on each of 10 chips were being enumerated and
charged for nothing. The
clearest case is the ViT machine, whose swept capacities put its frontier at
8980 mm2; built to its placement the same machine is 542 mm2, which means 94%
of that curve was empty silicon that a human had picked.

**What the sweep is left with:** `layers_per_chip` (the partition) and
`bank_depth` (the fold). Everything else on this machine now derives: the
engine widths to each stage's analog pass (D31/ADJ-9/ADJ-10), the state store to
the state held (ADJ-11), the chip capacity to the weights and the chiplet count
to the beat (here). `arrays_per_chip` and `shared_chiplets` join REFUSED_AXES
and are refused BY NAME, with the reason quoted.

## ADJ-14 — An engine is sized for the stages its chiplet serves, so the CHIPLET COUNT is a frontier axis.

**What was missing.** ADJ-13 left the shared digital chiplet count derived at
one per pipeline stage, and refused it as an axis, because the engine widths
were derived PER STAGE: a chiplet asked to serve three stages would have been
sized for one of them. That is not a modelling nicety — `--verify` measured it,
re-running an emitted config with a shared count and finding a machine 3.5x
slower than the sweep claimed (Qwen: 47148 vs 13314 tokens/s).

**The law.** A shared chiplet runs the digital work of every stage it serves,
one after another, inside one beat. So both derivations (the scan/vector engine
of D31/ADJ-9 and the attention fabric of ADJ-10) now take a STAGE GROUPING and
widen until the busiest chiplet's SUMMED per-stage time fits the ANALOG PACE of
the stages it serves — the same target ADJ-9 uses for one stage, read over the
set. With one stage per chiplet the loop exits immediately and every previous
number is unchanged. The grouping is CONTIGUOUS, so a chiplet serves neighbours
in the pipeline and reads its operands locally.

**Where widening stops.** It stops where it stops HELPING. If doubling the
width no longer lowers the busiest chiplet's time, more silicon is not buying
speed: the search halts at that saturation width and the disclosure says the
chiplet is the binding term. This is the same saturation ADJ-10 already names
for a single stage — the attention fabric runs out of folds to run in parallel
— read over a shared chiplet. Without this guard the search bought 8192 arrays
and 63000 mm2 of silicon that changed nothing.

**What it buys.** The count is a real design choice with a real trade, so it is
an AXIS (`mapping.shared_chiplets`) and the frontier explores it. Granite at
one chiplet per stage: 62661 tokens/s, 294.2 mm2 of digital. At 5 chiplets:
40761 tokens/s, 241.5 mm2. At 2: 40702 tokens/s, 211.9 mm2 — 28% less digital
silicon for 35% less throughput, with the scan engine widening 5479 -> 30342
lanes and the attention fabric refusing to widen at all because it is
saturated.

**On Granite the near-tie is the finding.** Its 23-point sweep puts
`5 layers/chip x 5 chiplets` at 65253 tokens/s and 10575 mm2 against the
selected `4 x 10` at 65325 and 10784 — 0.1% of the throughput for 209 mm2 and
half the digital chiplets. Both are on the front, and the lexicographic rule
takes throughput first, so the sweep ships the faster one and the cheaper one
is one row below it rather than invisible. The same table shows 20 chiplets
costing 615.7 mm2 of digital silicon against 10 chiplets' 307.8 for IDENTICAL
throughput, which is what over-provisioning a count looks like once the count
is swept instead of assumed.

**And it finds designs a human would not have picked.** On Qwen3.5-4B the sweep
selects FOUR chiplets over eight (one per stage) — 43434 tokens/s at 4456 mm2
against 42214 at 4495. Sharing is both faster and smaller there, because one
card carries one engine width: sizing it for a shared chiplet widens the engine
for every stage on the machine. That is the kind of point the frontier exists
to find, and it is exactly the point a hand-declared count would have missed.

## ADJ-15 — A convolution maps terribly onto a crossbar. Where its WORK runs is a swept axis; where its WEIGHTS live was never a choice.

**The physics.** A short depthwise conv is `k` taps per channel with no channel
mixing. A weight-stationary analog macro holding one would activate a handful
of cells out of millions — the worst possible use of the one resource Invariant
W says is never wasted. So the conv's WEIGHTS are not tiled onto analog arrays,
under any setting, and never were: `_hybrid_stages` excludes them from the tile
set by name.

**What was wrong.** The WORK had one hardcoded home — the per-macro digital
pool (ADJ-3) — and two bugs hid its cost. The pool read conv dimensions only
from a `short_conv:` block, so a Mamba stack that declares the same operator as
`ssm.d_conv` sized ZERO conv lanes; and the recurrent mixers never emitted a
conv op at all. The mapper had already dropped the weights *because* the pool
runs them, so the operator was charged on neither side. It is now read from all
three schemas that declare it (`short_conv`, `ssm.d_conv`,
`linear_attention.conv_kernel`) and the conv's rolling window — `d_inner x
(d_conv - 1)` per stream — is in the state bill it was missing from.

**The choice, and it is a real one.** `cim.allocation.conv_placement`:

* `macro_pool` — absorbed into every macro's own pool. No beat time, and the
  silicon is bought on EVERY enumerated macro slot.
* `shared_digital` — a TIMED op on the shared digital chiplet, priced on the
  vector engine by the same `channels x (2k-1)` per-result law the pool sizing
  uses (one law, two placements, D21). Costs beat time and engine width; the
  silicon is bought once per chiplet.

Neither is a default worth assuming, so it is an AXIS. Measured on
Falcon-Mamba: `macro_pool` costs 1448.6 mm2 of pool silicon, `shared_digital`
costs 6300 mm2 of extra vector engine (the conv's serialized work forces ADJ-9
to widen it to hold the analog pace) for 1% less throughput. The pool wins on
that model — the opposite of the intuition that replicating an operator
thousands of times must be the expensive option — and a model with a narrow
conv and a busy pool can flip it. A third value is refused by name.

## ADJ-16 — The beat is the larger of two derived floors, and the measurement window is derived from the stage count.

**What was wrong.** The beat was the MEASURED spacing between token exits and
nothing else. Three defects hid behind that.

*The links were free.* `_chip_boundary` took `deps[-1]` as the producer, which
in the filled pipeline is the STAGE-HOLD edge — an op already on the
DESTINATION chip. The link law saw producer and consumer on one chip, called
the hop intra-chip, and charged 0 s. 589 of Falcon-Mamba's 620 handoffs and 72
of Granite's 385 cost nothing. The producer is now the last dep that actually
lives on the source chip.

*The exits were read even when they had not converged.* Granite's last two
steady intervals disagree by 51% at window 3 and still by 62% at window 14. A
beat read off that is the ramp, not the rhythm.

*Nothing checked the answer against the machine's own capacity.* Falcon-Mamba
reported 1,240,209 tokens/s with **71.7 tokens in flight in a pipeline built for
32 streams**.

**The rule.** Two floors, both physical, both measured off the timeline:

1. **Stage occupancy.** No stage can be busy more than one whole beat, and a
   stage holds its slot until its OUTGOING LINK completes. There is no buffer
   to park a sequence between stages, so a link that does not fit inside a beat
   is not hidden by pipelining — it sets the pace. Occupancy is a RATE (busy
   time over beats lowered) and does not move with the window.
2. **Little's law.** A traversal takes L and the machine has D stream slots, so
   a token cannot be replaced faster than L / D. L is the MEDIAN of the settled
   traversals, never a single sample — one sample swung the beat 2x with the
   window.

The beat is the larger. The measured exit spacing is used only when it
CONVERGED and only if it is slower than both floors — a converged, slower
rhythm means the machine really stalls on dependencies, and that is honest.
The artifact names which floor binds (`beat_binding`).

**Why the link is folded into its stage rather than made a stage of its own.**
Counting the 31 inter-chip links as pipeline positions would raise D from 32 to
63 and roughly double the state store. The machine cannot buffer sequences in
flight, so those positions do not exist: the honest reading is that a stage is
not done until it has handed off.

**The window is DERIVED.** `decode_window` is raised to the stage count, so
every stage is observed in steady state at least once. A declared window below
that is raised and the raise is disclosed; a larger one is honoured. It changes
NO silicon (the store is sized from the served request, ADJ-11 + M10), and a
declared 3, 6 or 10 now produce the identical beat.

**Measured.** Falcon-Mamba 1,240,209 -> **534,701 tokens/s**, 31.64 tokens in
flight against 32 slots, identity residual -0.0115. The independent Little's-law
ceiling computed by the numbers audit was 538,345 — two methods, one answer.
Granite and Qwen were already legal and stay legal (6.53/10 and 8.00/8).

## D33 (GEORGE, HARD, 2026-08-28): There is NO SHIPPED POINT. The frontier is the entry, and the selection binds the page.

George, verbatim: "WE SHOULD NOT HAVE A SHIPPED POINT AT ALL!!! THAT IS WHY WE
SHOW FRONTIER FIRST. THEN WE CLICK ON A POINT AND EVERYTHING UPDATES TO SHOW
THAT." Consequences, binding on the atlas and on every artifact that feeds it:

- The atlas opens on the MODEL SELECTOR, then the PARETO FRONTIER
  (x = decode tokens/s, y = total silicon mm2). Nothing renders above it but
  the selector.
- Every other element of the page — headline, hardware cards, beat, tables,
  map, drilldown — describes the SELECTED point and rebinds on click. A
  panel that cannot be produced for the selected point states that absence
  for THAT point; it never falls back to another point's content.
- The sweep's lexicographic pick survives only as the INITIAL selection and
  carries no privileged name. The words "shipped design" leave the page.
- The DSE emits per-candidate data sufficient to bind the whole page for
  every FRONT point; the deep spatial map may be lazy-emitted per point,
  and a point without one says so and names the command that emits it.

## D34 (GEORGE, HARD, 2026-08-28): Stage mismatch is a mapper FAILURE, not a fact. The mapper eliminates it even at area cost.

George, verbatim: "one of the stages is much longer than others. this causes
pipeline waste and sets the beat, right? this should *never* happen unless we
are reaching degenerate cases... The entire point of the mapper is to *try*
and eliminate mismatch even if it comes at an area cost. thats the core
philosophy." Consequences:

- Every priced candidate computes a STAGE-BALANCE verdict: max stage busy
  time over the median, the stages that exceed it, and WHAT makes them long
  (block breakdown). A ratio beyond the disclosed threshold is a WARNING
  printed to the sweep's stdout (visible during regeneration) and carried on
  the candidate and in the report — never silent.
- The warning names the reachable bound: the beat a balanced partition could
  achieve given the point's own unavoidable floors (a saturated fabric stage
  cannot be shrunk by repartitioning; everything else can be packed toward
  it).
- The follow-on build is a balance-aware partitioner: non-uniform
  layers_per_stage packed toward equal stage time (the mapping schema
  already admits non-uniform stage plans). Measured case that forced this:
  Granite-4.0-H-1B at 2 layers/chip — stages 2/7/12/17 (the attention
  layers) run 28.7x the median stage's digital time and set the beat while
  16 stages idle.

## D35 (GEORGE + adjudicated data, 2026-08-29): The macro is a PLATFORM CONSTANT. The sweepable list is closed.

George: "the 2kx2k macros are too large... Macro size should not be
[sweepable]. It should be a fixed 512x512 or 1024x1024." Until now the cards
set rows = the model's hidden_dim — the hardware changed shape per workload,
which is not a buildable chiplet platform and quietly flattered every
comparison. Adjudicated on sheet data (George may overrule): **1024x1024**,
which wins net silicon per useful cell on all four subset_v1 models (14-36%
over 512x512: the 55% peripheral overhead of the smaller macro dominates its
cleaner tiling) and stays inside the OPTIMA-characterized array family, which
is exactly what is wrong with 2048+: extrapolated analog physics. Weights TILE
onto the fixed macro (ceil row/col splits, the priced K-accumulator);
dimension-mismatch remainders are D27's reported waste metric, never hidden.

THE CLOSED LISTS — a knob is in exactly one:

FIXED (platform constants; never swept, never model-derived):
  macro geometry 1024x1024 · analog clock 100 MHz · slice structure and the
  vector-mux tie (sheet profile) · shots_per_output · the measured 22nm
  synthesis blocks · the measured SRAM macro · the measured GEMMINI block.

SWEPT (the design space; exactly these):
  adc_capacity_mux · layers_per_chip / layers_per_stage · bank_depth ·
  shared_chiplets · conv_placement.

DERIVED (from the point; never declared):
  analog macro count (cell floor via tiling) · chip capacity · chiplet
  count/balance · D · engine widths incl. context-split S · SA geometry R/C ·
  softmax/vector/combine lanes · the state store.

REFUSED BY NAME (axes and config spellings):
  arrays_per_chip · num_arrays · vector_lanes · softmax_lanes ·
  analog_clock_mhz · macro rows/cols in any spelling, including the retired
  rows-equals-hidden convention.

## D35-REVISED (GEORGE, 2026-08-29, supersedes D35's square-macro reading): The platform sweep is (Z, Y) in the constraint-native box. Mux is DERIVED and is the lens, never an axis.

The macro is X x Y : Z — X rows (height, the analog sheet's constrained
dimension), Y total CTT columns (storage tile), Z ADCs (the FLOP producers;
muxed columns are storage). mux = Y/Z, always derived notation.

- X is FIXED per platform (512 or 1024; George's device call, pending).
- THE PLATFORM AXES: Z in pow2 {8 .. X} (never more ADCs than rows) and
  Y in pow2 {X .. 8X}. Y >= Z always (mux = 1 is legal). The 8X ceiling is
  adjudicated, not sacred: an EDGE-CROWDING check reports by name when a
  model's front crowds the Y = 8X wall, and extending to 16X is then a
  one-line data-justified change.
- MUX IS NEVER SWEPT. It is derived, printed in every point's X x Y : Z
  label, and is the plot's LENS: points colored by mux on a monotone
  single-hue ramp; a two-handle RANGE control above the plot whose bounds
  are the realized min/max mux among plotted points, defaulting to full
  range, handles snapping only to realized values, controlling visibility.
  Y gets button/toggle filters beside it. Rationale (measured, sheet):
  total system ADC throughput = cells/(X*mux) — Y cancels — so mux is the
  one density label; but the CONSTRAINTS are native to Z (converter count,
  output interface, calibration) and Y (wordline length, tile granularity),
  so the sweep walks the axis-aligned legal box and no absurd corner hides
  in a derived quantity.
- EVERY AXIS IS REFEREED: the per-axis vitality check (an axis must
  contribute >= 2 distinct values to the non-dominated set across a bundle
  run, else it is flagged for demotion by name) applies to Z and Y exactly
  as to the mapping axes.

WITHIN A TAPE-OUT (Y, Z), the mapping sweep is exactly:
  layers_per_chip (partition) · bank_depth (ON PROBATION: it won depth-1
  nearly everywhere on the old cards; the fixed-macro remainder economy is
  its last chance, the vitality check is its judge) · shared_chiplets ·
  conv_placement.
Named exclusions, each with its promotion condition: tensor orientation
(presumptively monotone; promote if remainder data shows both orientations
winning), tp (presumptively link-dominated at <=7B; promote on measurement),
PD split (returns with prefill), workload params (serving, not design).
Everything else DERIVES (ADJ-9/10/13/14, the SA rule, context split, state
store) — the doctrine stands: digital derives to the analog floor.

## D36 (GEORGE, 2026-08-29): The two-stage DSE. Digital is a derived FUNCTION of the analog point; grouping is chiplet ATTACHMENT, never execution.

- The GROUP = the model's minimal repeating motif (auto-derived from
  layer_plan; 1 for homogeneous models). Execution is UNTOUCHED: the
  pipeline stays per-layer staggered (D29). The group is the pattern at
  which digital chiplets attach: every chiplet serves one group's stages
  and therefore sees the SAME per-beat work mix — uniform chiplets by
  construction, full pipelining preserved.
- STAGE 1 (the megasweep): analog + mapping only — (Z, Y) x partition x
  bank_depth x conv_placement. Under the standing doctrine (digital derives
  to the analog floor; no saturation cliff post context-split; links
  zeroed) THROUGHPUT IS A FUNCTION OF THE ANALOG POINT ALONE.
- STAGE 2 (a function, not a search): per point, derive the group's
  chiplet — engines to the group's summed per-beat digital work at the
  analog pace, state store INSIDE the chiplet and inside its area.
  Packaging arithmetic, not axes: chiplet area band [50, 400] mm2
  (configurable, hard-set for now); over 400 -> chips_per_group =
  ceil(area/400); under 50 -> merge groups per chiplet toward the band.
- shared_chiplets is DEMOTED from axis to a STANDING LINEARITY CHECK
  (price one point at 1x/2x/4x sharing; assert linearity within a few
  percent). The nonlinearity that once made it a real axis was fabric
  saturation, which the context-split law removed; if any future law
  reintroduces a cliff the check fails loudly and the axis returns.
- THE NAMED EXCEPTION: the day a card declares state_bytes_per_cycle and
  an operand-bandwidth floor binds, digital sets the beat and stage 1 must
  fold in the (closed-form) bandwidth beat-check. Until then it is a named
  assumption.
- Within-digital sweeps (mix, node, clock) return only when fidelity earns
  them. 22nm measured library only; clock fixed at 1 GHz.

## D38 (GEORGE, 2026-08-29): The partition is INTENT, not an axis. The fill fraction is the axis.

- layers_per_stage RETIRES as a sweep axis. The machine is always maximally
  disaggregated: local microbatch 1, finest motif-compatible stages,
  everything time-matched by area sizing (the derive-to-floor doctrine is
  what makes 100% utilization possible). Stage granularity DERIVES.
- THE NEW AXIS: pp_util in {1.0, 0.75, 0.5} — a deliberately UNDERFILLED
  pipeline (resident streams = pp_util x stages). Trades state/KV area
  (linear in resident streams) against throughput (linear in fill). Usually
  dominated; for state-limited large LLMs it is the escape valve. This
  amends D29's "always full": full is the default, underfill is a priced,
  disclosed choice.
- KV and recurrent state are FP8 (8 bits/element) for now.
- conv_placement RETIRES: the conv's work is ALWAYS on the shared digital
  chiplet. Not an axis.
- bank_depth is PARKED for ground-up reformulation (the mux-slot allocation
  question, next).

## D37 (GEORGE, 2026-08-29; recorded late — the code cited it before this entry existed, which an audit caught): The analog chiplet is a DENSE CARRIER. No axis lives at chip level.

Agreed in the same session as D35-REVISED/D36, written down now:

- The analog chiplet is a fixed grid of identical macros, capacity
  C = floor(400 mm2 / macro_footprint) — the same area cap as D36's digital
  band, derived per platform point, never swept.
- The packer fills chiplets DENSE, in pipeline order, CROSSING stage
  boundaries freely: stages are logical, silicon is packed. Per-stage
  rounding waste does not exist; the only chip-level waste is ONE partial
  tail chiplet per system, reported (packing tail), never multiplied by the
  stage count. Invariant W holds.
- Chip-level sweep axes: NONE. Chiplet capacity would fail the vitality
  test on arrival (with links cheap, silicon is C-invariant under dense
  packing and the tail shrinks monotonically with C). PROMOTION CONDITION:
  analog area-per-chiplet becomes an axis when BOTH links and per-chiplet
  overhead/yield are priced — a real counter-pressure pair.
- Mid-chiplet stage boundaries are a FEATURE once links are priced: a
  handoff inside a chiplet is an intra-chip hop, not a package link.

## Amendment to the link prices (2026-08-29, George + adjudicated value): intra-chip hop = 5 ns (was 50 ns placeholder). Cross-chip stays 500 ns, no fancy packaging assumed.
Basis: tightly packed macros on one 22 nm die; ~20 mm die edge at ~0.1 ns/mm
repeated wire -> under 2 ns typical flight; 5 ns flat is conservative margin.

## D39 (GEORGE, 2026-08-31): The beat is OCCUPANCY ONLY: beat = mux x 20 ns. Latency never sets the beat; it sets the state bill through Little's law. The mapper's ledgers are waste, L and seams.

Settled in the 2026-08-30/31 mapping sessions, in George's "simpler universe"
(hidden_size == macro height; column-height folding is a later add-on):

- COUNTING, NOT SCHEDULING. Every dense column fires once per token, Z at a
  time, so a macro owes mux = Y/Z firings per token. beat = mux x t_slice
  (20 ns). The platform point (Z, Y) fixes the beat; the mapper cannot move
  it. (George: "the beat is mux * 20ns.")
- THROUGHPUT = 1/beat. Lowering mux buys CONCURRENCY, not speed. Per-user
  token rate = 1/L, where L is the per-token chain latency. (George: "when we
  lower mux, we are not actually making the pipeline any faster at all.")
- LATENCY IS PIPELINED. Wire hops, reduce-tree depth and seams never bound
  the beat (bandwidth is ASSUMED PROVISIONED - requirements are low here;
  recorded as an assumption and disclosed on every artifact). Latency sets
  the in-flight sequence count D = ceil(L / beat) (Little's law), and the
  state bill = D x state per sequence. This SUPERSEDES the D29/D38 spelling
  D = round(pp_util x stages): that is the stage-synchronous schedule's
  number (one live sequence per stage) and overstates D by about mux/2 for
  a spread machine. pp_util keeps its meaning as the fill fraction of D.
- SLOT = PHASE. Each ADC rotates through its mux slots once per beat: slot
  index j fires at ticks == j (mod mux). Placement and schedule are ONE
  decision. A token hops slot to slot along the rotations (a wavefront); one
  hop needs >= 2 ticks (fire, then wire + reduce <= 1 tick). Temporal depth
  d on one macro costs d ticks of L, never beat.
- THE MAPPER CONTROLS EXACTLY THREE LEDGERS: cell waste (ragged widths
  against Z), L (-> D -> SRAM), and seams (500 ns = 25 ticks each). Rules:
  spread each bundle across macros on ONE chip; place in chain order; never
  split a bundle across chips. The only objective is zero waste; everything
  else is measured and reported against its bound (L_min = 2n - 1 ticks for
  n bundles, plus 25 per seam that chip capacity forces: the hop gap of 2
  separates the n - 1 joins and L counts the last firing plus one).
- NO NUMBER IS TRUSTED FROM THEORY. Every mapping emits an explicit tick
  schedule, and an INDEPENDENT checker verifies it from the document alone:
  one firing per macro per tick, every slot-layer once per token, hop gap
  and seam gap honoured, columns conserved.
- Supersedes the in-beat link accounting of the 2026-08-29 link-price
  amendment and the stub's beat law pp_lat = t_serial + t_reduce + t_hop:
  those terms now enter L, not the beat.
- FIRST IMPLEMENTATION: a standalone slot mapper (fws_slot_mapper.py) and a
  TOY debug model (about two analog chips, a few layers, no digital) shown
  in the atlas under ?bundle=toy_slotmap. subset_v1 results stay
  design-invalid until the mapper replaces the stub (mapping_stub stands).
