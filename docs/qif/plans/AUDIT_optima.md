# Audit: the OPTIMA model/mapper — what to trust

Manual audit (single-reader, no agent fleet), 2026-08-23, prompted by
George's distrust of the model "especially around multi-tenancy." Companion
to `RECON_optima.md`. Labels: **[contradiction]** internal inconsistency;
**[physics]** internally consistent but physically questionable;
**[gap]** the capability does not exist despite appearances.

## Verdict up front

George is right. OPTIMA splits cleanly in two:

- **The stage laws are trustworthy** — analog M-law, array census, per-stage
  byte flows, the digital engine library. Keep them as numeric references
  and parity targets (that is all our fws_cim validation ever used).
- **The "mapper" and the multi-tenant layer are not a mapper and not a
  model.** Area-division arithmetic plus a serial time-sharing formula.
  Port nothing from them; treat the proposal Table 1 VLM rows as
  provisional numbers, not targets.

## Findings, ranked

### 1. [gap] There is no mapper

`_chip_accounting` (compiler.py:120) derives chip counts by dividing
required area by usable chip area — in `fractional_area` mode it reports
**non-integer chips**. No layer or stage is ever assigned to a specific
chip; there is no adjacency, no per-link traffic derivation, no notion of
which boundary crosses a package link. "Layers/chip (analog): 3" in the
reports is division, not placement. The proposal's "compile-time placement
and scheduling problem" has **no existing implementation to port** — the
mapping compiler must be built natively (on the program IR), with OPTIMA
supplying only stage laws and area inputs.

### 2. [physics] Multi-tenant throughput is serial time-sharing

`_model_sequence_throughput` (compiler.py:248):
`SPS = 1 / Σ (weight_i / fps_i)`. The tenants never run concurrently — on a
spatial machine whose premise is that co-resident weights stream
concurrently. For the VLM max-perf row: concurrent steady state would be
`min(127.551K/11, 11.837K) ≈ 11.6K SPS`; the report says 5.858K — a ~2x
understatement *if* each tenant's arrays are dedicated (the area accounting
charges for them). If instead arrays are truly time-shared via banking,
serialization is right but then the area story is wrong (see 3). The model
cannot represent the co-resident-and-concurrent case the proposal describes.
Latency-shaped math is being presented as throughput.

### 3. [physics, self-declared] Partial banking is an aggregate smear

`_apply_partial_banking` (compiler.py:840) computes a bank_factor area/energy
delta and smears a *fraction* of it evenly across all layers; per-stage
numbers stay "fully unbanked" (its own comment says so); identical chips and
interleavable floorplans are assumed; and banking is asserted to leave
vector timing unchanged — dubious, since stored-columns-per-ADC is exactly
the mux/timing trade the rest of the model prices. Bank-switch cost: absent.

### 4. [contradiction] Two TOPS accountings disagree by ~10x in one artifact

pipeline.py's multi-tenant report carries "Combined TOPS 602,880 / 35.1
TOPS/W / 89.7 TOPS/mm²" = `Σ weight × raw_TOPS` — every tenant at full rate
simultaneously, while the SPS on the same page says they serialize. The
compiler's `seq_tops` (59,377 = SPS x ops-per-sequence) is the honest
figure, and it is the one Table 1 used. Both sit in the recorded outputs;
any future reader can quote the inflated one.

### 5. [evidence hygiene] The flagship rows ran with constraints disabled

`compiler_vlm_vit_mamba_5nm_2d3d8.yaml` (and max_perf) override
`max_link_bandwidth_GBs: 9999`, `max_io_bandwidth_GBs: 9999`. The bandwidth
constraint machinery exists (`_bandwidth_constraints`) and was switched off
for exactly the headline Table 1 configurations. Whether those rows pass
with real limits is unknown.

### 6. [stance] The digital side is sized-to-target, never timed

The scan engine derives its tiling and unit counts from a reuse target
(`microcycles_per_token=100`, hardcoded at the stages_mamba.py call site);
helper collections raise if they would bind a stage. Digital load can only
cost area, never time — contention is impossible **by construction**. Fine
as DSE intuition; vacuous as validation of digital timing; useless as a
scheduling reference. (This is the OPTIMA absorption contract taken to its
limit; our Tier S exists precisely to model what this cannot.)

### 7. [gap] The bandwidth model is aggregate-only

`_bandwidth_constraints` checks stage-boundary totals divided evenly across
chips (perfect balance assumed), plus one ring-allreduce estimate for tp.
No topology, no routing, no buffering, no latency anywhere. The proposal's
"activation routing plan that respects bandwidth and buffering constraints"
does not exist in any current asset.

### 8. Minor (parity-relevant, not damning)

- `_analog_vec_time_us` quantizes analog time to the digital clock with a
  +1 cycle — replicate for Mamba parity or gates miss by one cycle.
- `opt_prefill` is dead code (hardcoded False).
- mux must divide num_heads for the Mamba dt array (hard assert).
- "TP groups" in reports has unclear semantics vs tp.degree — resolve
  before using tp recordings as targets.

## Consequences for the plan

1. **Mapping is greenfield.** Build it on Rapid-LLM's program IR; OPTIMA
   contributes stage laws, area/energy inputs, and single-model recordings
   only. This aligns with the mapping-first rescope.
2. **Validation stance:** OPTIMA parity gates remain law-level and
   single-model. No mapping-level or multi-tenant OPTIMA number is a
   target. Table 1 VLM rows are provisional until our own model reproduces
   or corrects them — expect corrections (findings 2, 5).
3. **Honesty rules for our own tools** (so we never reproduce findings 4-5):
   a report never carries two inconsistent accountings of one metric; a
   constraint can be relaxed only if the artifact discloses the relaxation;
   chips are integers.
