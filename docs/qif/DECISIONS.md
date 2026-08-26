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
ViT-Huge-story at seq 64: 1 chip, 400 slots, 386 tiles, 779 requests/s.

**Pure SSM** (`configs/hardware-config/fws_cim_mamba.yaml` + the new
`falcon_mamba_7b_decode_inf.yaml` D29 twin): 64 Mamba-1 layers, no attention
block at all. It used to CRASH rather than refuse — three sites read attention
attributes unguarded (`CimModelParams.from_model`, and two in
`base_timing.TimeCalculation.__init__`). A model with no attention has no heads:
the count is 0 and every attention law is already gated on a layer declaring an
attention block. 241 362 tokens/s at 14 788 mm2 on the selected point.

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
