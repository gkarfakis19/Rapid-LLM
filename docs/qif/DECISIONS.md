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
