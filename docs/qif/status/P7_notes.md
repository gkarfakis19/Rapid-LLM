# P7 status notes — the long form

`docs/qif/status/P7.json` carries the 2-4 sentence claim per deliverable, as the
status protocol asks. This file is where the detail those entries used to hold
lives: the assumptions, the regime-change rewrites and the files each wave
touched outside its own set. Nothing here was deleted from the register — it was
moved, verbatim, on 2026-08-24 (P7.9), and every entry in P7.json points here.


## P7.1 — Fold taxonomy + DOA register (decode-only; dead folds refused by name)

Revised after George's corrections: Invariant W (weight space never wasted - dense packing,
waste% reported) pins analog at the global cell floor; decode analog time is assignment-
invariant at the floor; the frontier is SYSTEM SIZING (digital provisioning vs
area/throughput); no safety margins. DOA register updated accordingly.


## P7.2 — Dense packer (Invariant W) + K-inner accumulator pricing + waste% metric; degenerate identity

cim_timing.py gained the dense packer (dense_blocks / dense_pack -> DensePacking) that
streams every tensor's (K,N) blocks into consecutive BANKS across a chip with no per-tensor
rounding, so a bank idles only when the stream runs out, and reports ONE waste accounting
itemized into remainder (dimension mismatch) + tail (unfilled last macro), closing exactly
as real + remainder + tail == committed against the GLOBAL CELL FLOOR ceil(real_cells /
cells_per_macro). The K-INNER walk is the only one that exists (CANONICAL_WALK); DEAD_FOLDS
+ refuse_dead_fold are the DOA register in code and non_canonical_walk / slicing_x_folding /
replication / prefill_folding are refusals with no machinery behind them (D26), with dense
prefill refused at the mapping seam (D25). price_accumulation prices an in-macro K stack --
(depth-1) adds per output, all but the last hidden under the following ADC pass, the tail
being the final drain, and stall DERIVED to zero because P2.5 sizes the pool lanes to the
macro's own result rate -- while a SPREAD stack is charged nothing here because the existing
row-block partial-sum law already prices it end to end (D21), and price_packing_accumulation
charges the worst macro and counts ONE accumulator per macro, which is what the canonical
walk buys. fws_mapping.build_mapping took a packing='dedicated'|'dense' KEYWORD (config.py
is E2's; this is the seam it will call): the degenerate identity is TILE-FOR-TILE on
Llama2-7B, MoE-small, Granite-4.0-H-Tiny and Qwen3.5-4B with a bit-identical makespan
through the DAG and the evaluator, because every shipped card admits whole-macro allocation
only; where the card admits a finer bank the delta is computed and disclosed (Granite at
bank_depth 1: 4894 macros against 5610 dedicated, 716 saved, waste 47.579% -> 39.910%, floor
2941). ASSUMPTIONS: accumulator energy and area report 0 when pool_energy_per_add_pj /
pool_area_mm2_per_adder are undeclared (honest zero, disclosed, load-bearing when a card
declares them) and the accumulator's holding register is a NAMED GAP rather than an invented
number; the packer never reorders tensors, so chip locality is the caller's offer order and
remains a preference, not a search.


## P7.3 — Dense packing as a mapping mode; cross-layer bank sharing; utilization as a required output

mapping.packing is the config surface (config.py parses the name, fws_mapping owns the list,
so the two cannot drift) with 'dedicated' still the default and an explicit keyword
overriding a declared law; a dense PREFILL is refused by name through it (D25/D26). The
lowered DAG now carries an accumulator op per LOCAL K stack on the OWNING macro's pool,
priced by P7.2's price_accumulation from the packer's own KStack descriptor
(LAW_K_ACCUMULATION, its own itemized energy term), while a SPREAD stack is left to the
existing row-block law - one accounting per shape (D21). Fact 1 is MEASURED per run
(evaluation.bank_passes): every macro of every shipped run walked its occupied column sets
exactly once in the lowered decode step, dense and dedicated alike (Granite 19566 passes
either way). Fact 3 is measured too (evaluation.bank_sharing): with 328 macros holding
several tensors and 8 holding several LAYERS, the delay attributable to another layer's
residents is exactly 0.0 while the same instrument sees the macro's own bank walk - the zero
is a measurement, not an assertion. UTILIZATION is now a required output: per-device-class
rows (analog macros, pools, shared chiplets, links) in every report JSON, in the text
section, and in the atlas via atlas_metrics(), with idle devices INSIDE the mean, plus an
unconditional device_class_utilization banner naming the binding class (D28 - Granite's
headline run reports analog macros at 0.008% mean against shared digital at 8.8%). Waste%
rides the same report as the packing block (refused, by name, under the dedicated law where
no packer ran). END TO END on the shipped Wave D winner (bank_depth 1): Granite decode
reaches 4894 macros against today's layers_per_chip 5610 (716 saved, floor 2941, waste
39.910%) with the median decode step 652.963 us against 654.563 us (-0.244%, the
accumulator-vs-partial-transport trade, 288 accumulator ops) - checked in as
docs/qif/folding/granite_dense_vs_dedicated.json behind a regenerate-and-compare BYTE gate.
DISCLOSED: co-residency is free only ACROSS layers - two concurrent tensors of ONE layer
sharing a macro serialize on its ADC path (1.44 us on the MoE fixture) and that is now a
named banner rather than a footnote; a local K stack under the DEDICATED law carries no
accumulator (only the packer emits the descriptor) and is disclosed as unpriced rather than
absorbed; the link utilization row is a DEMAND ratio, not an occupancy, because D17 declares
no congestion model; the shipped artifacts were regenerated for the new blocks and their
byte gates re-pass (granite/qwen reports, granite/qwen/pd atlases, and the Wave D demo
sweep, whose numbers are unchanged and which gained only the per-candidate utilization
banner). THREE FILES OUTSIDE MY SET, all small, all disclosed: tools/fws_emit_atlas.py
rebuilt the mapping spec field by field and DROPPED a declared packing whenever a
--layers_per_chip / --shared_chiplets flag was given (it now carries it, with a test); and
the validator's P3.7 row 'every sweep disclosure names the candidates that carried it'
asserted that NO union entry disagreed across candidates, which stopped being true the
moment utilization became a disclosure - the row now checks what its own claim says, that
every disagreement is FLAGGED and itemized (candidate id, value, reason), and it prints '18
rows labelled, 1 with a flagged disagreement'. Validator 423/423, exit 0.


## P7.4 — Frontier v2 (D29-D32): stage granularity x mux depth x chip capacity vs (area, tokens/s), digital derived per point, DAG-verified

THE FRONTIER IS REGIME v2 (D29-D32), and the axes changed with the regime. STAGE GRANULARITY
is the axis that buys throughput: the default stage plan is one stage per analog chip, so
mapping.layers_per_chip moves the chip split, the stage count, D, the beat and the per-stage
state bill AT ONCE. CHIP CAPACITY (cim.chip.arrays_per_chip) is the AREA axis, because
Invariant W (D27) pins the analog macro count at the global cell floor and the only way a
point spends analog silicon is by enumerating slots no tensor lands on. MUX/BANK DEPTH is
the third, refused BY THE CARD where the card's adc_mux does not divide it.
mapping.layers_per_stage is a fourth axis for a stage plan decoupled from the chip split.
DIGITAL PROVISIONING IS NOT AN AXIS: D31's engine is derived per point and D32 prices it, so
vector_lanes now lives in RETIRED_AXES -- still spellable, because the frozen Wave-D demo
sweep is built on it and the validator reruns that artifact, but every candidate that sets
it carries the retirement in its own notes and a sweep-level retired_axis_swept disclosure,
and neither frontier declares one. THE WALK IS A LADDER, not a cross product
(mapping_dse.search: ladder): from a DECLARED balanced initializer it trims idle silicon
first, then climbs ONE axis ONE rung at a time toward throughput (best tokens/s per mm2),
trims each accepted rung before ranking the next, and finally descends from the trimmed
initializer trading throughput for area. Stage granularity and chip capacity are COUPLED --
a chip must hold the layers its stage owns -- so a step the placement refuses is retried
once on the declared repair_axis at the smallest capacity that admits it, and both the
refusal and the repaired point stay in the report (4 repaired steps on Granite, 4 on Qwen).
Every point the walk LOOKS at is placed by P3 and priced by P4 on its own timeline,
including the ones it declines; the walk is deterministic, per-point wall time is printed,
and the front it reports is the non-dominated set of the VISITED points, which is a LOWER
BOUND on the front of the declared space -- said by name in a
front_is_over_the_visited_points disclosure rather than implied. PER POINT the report now
carries D29's own quantities: the beat, tokens/s = 1/beat, D (= the stage count, DERIVED),
the per-stage state and KV bill for all D streams with its verdict, per-device-class
utilization with idle devices inside the mean (D28), and the DERIVED digital side (lane
count, provenance, composed mm2 and W). THE SILICON ACCOUNTING GAINED ITS THIRD TERM: total
= analog macro slots x footprint + shared digital chiplets + PER-MACRO DIGITAL POOLS, the
last two composed from the measured synthesis library (D32) and the pool term previously
missing -- it is priced per enumerated macro SLOT, which is exactly the term the chip-
capacity axis moves, so a frontier without it under-reported every point by a whole device
class. A NEW LAST STAGE, `state`, is D29's feasibility check: the per-stage residency
against mapping_dse.max_stage_state_bytes, run AFTER pricing so a refused point still
reports its throughput and its bill, and refused with the stage, its layers and its bytes
named. Stage order is now config < mapping < budget < lowering < pricing < memory < state,
and the memory stage's violated verdicts are recorded by scope/tier/owner rather than
counted. ASSUMPTIONS / DISCLOSED: (1) the ladder is a walk, so an unvisited point is ABSENT,
never estimated -- the disclosure says so and the trail says where the walk stopped and why;
(2) the repair only fires on a MAPPING-stage refusal and only upward on the repair axis,
because no capacity fixes a residency violation; (3) max_stage_state_bytes is a DECLARED
sweep budget, not a measured capacity and not a margin (D28) -- the evaluator's own
residency verdict against the declared activation tier rides every row beside it under its
own name, two budgets on one measurement; (4) a declared budget on a lockstep run is REFUSED
rather than passed, because that regime measures no per-stage residency and an unmade check
is not a pass; (5) the knee is the front point furthest above the chord of its own extremes,
normalized over the front's own spans, and a straight front or a front with no interior
names NO point rather than an endpoint. FILES OUTSIDE MY SET, all disclosed:
tests/test_qif_dse_allocation.py's one-axis-one-field map gained layers_per_chip's sibling
layers_per_stage; configs/hardware-config/fws_cim_granite_tiny_dse_selected.yaml was
regenerated (it was STALE from before D31/D32: it carried headline 6110.95 tokens/s and no
synthesis_library, which its own test does not check);
docs/qif/dse/granite_lanes_banks/dse_report.json was regenerated for the pool term, the
state stage and the retired-axis disclosure, and its front, its pick and its throughputs are
unmoved.


## P7.5 — Granite + Qwen decode frontier curves with D, the state bill and per-device utilization, regenerate-gated

TWO CURVES, CHECKED IN, REGENERATE-GATED. Each frontier config is the SHIPPED machine plus
the mapping_dse block and nothing else (a test pins the two documents equal outside it), so
a curve can never price a machine that no longer exists. GRANITE-4.0-H-TINY (ADJ-1 headline,
docs/qif/dse/granite_4_0_h_tiny_frontier/): 16 of 24 visited points valid over a 168-point
declared space in 965 s; front spread over 5 points, 6322 tokens/s at 11334.0 mm2
(layers_per_chip 20, D = 2) up to 15291 tokens/s at 11358.7 mm2 (layers_per_chip 4, D = 10);
KNEE `c016` (layers_per_chip 10, arrays_per_chip 1400, D = 4, 12645 tokens/s at 11340.2 mm2,
41.0 MiB/stage). ITS BOUND IS MEMORY, NOT SILICON: with the config's own 64 MiB L2 tier
declared as the per-stage budget, 2 stage plan(s) are refused BY RESIDENCY with the stage
and the bytes named -- both layers_per_chip 2 points the walk reached (arrays_per_chip 640
and 560) hold 85.4 MiB on their worst stage at D = 20, against the 64.0 MiB budget, and 4 of
their 20 stages are over it -- so the climb stops at a stage plan the machine can hold
rather than at one it can feed. QWEN3.5-4B (docs/qif/dse/qwen3_5_4b_frontier/): 25 of 30
visited points valid over 105 declared in 58 s; front spread over 5 points, 1825 tokens/s at
3600.53 mm2 (D = 2) up to 14846 tokens/s at 3693.18 mm2 (D = 32) -- 8.1x the throughput for
2.6% more silicon -- KNEE `c008` (layers_per_chip 4, arrays_per_chip 80, D = 8), which is
the granularity the shipped machine already declares. THE PICTURE BOTH CURVES DRAW: the
analog term is IDENTICAL on every point of both fronts (11314.32 mm2 on Granite, 3591.85 mm2
on Qwen) because Invariant W pins it at the cell floor, so every mm2 of the frontier's
spread is DIGITAL, and the quantity that actually moves with stage granularity is the STATE
BILL (41.0 to 85.4 MiB per stage on Granite, 68.1 to 226.1 on Qwen). Qwen's frontier
DECLARES NO residency budget and says why in the config: its per-stage bill is 68.2 MiB at
the shipped granularity, already above the same 64 MiB L2 tier, and the bill is very nearly
INVARIANT to the stage plan (splitting stages divides the layers and multiplies D by the
same factor), so declaring that tier would refuse every point and produce no frontier at
all. That Qwen does not fit its own declared L2 at any stage plan is the finding, stated
rather than tuned away. GATES: tests/test_qif_frontier.py is 30 tests -- the ladder's own
arithmetic on a synthetic space with hand-computed rungs (trim, climb, descend, repair, an
infeasible initializer), the knee by hand on a four-point front (distances 0, 1/3, 4/15, 0),
the three-term silicon sum, D29's per-point quantities and the residency refusal on a real
MoE-small timeline with the budget set one byte under that plan's own measured bill, and a
regenerate-and-compare gate per curve that re-walks the whole ladder and reruns the --verify
round trip. --emit-config + --verify pass on both curves (8 and 8 cross-checks, counts exact
and times within 0.1%). Validator +39 rows (20 on Granite, 19 on Qwen -- the budget rows
differ because Qwen declares none -- none removed). ASSUMPTIONS / DISCLOSED: (1)
dse_report.md is NOT tracked (`*.md` is gitignored repo-wide), so the JSON is the artifact
and the tests render the markdown from the checked-in payload rather than reading the file;
(2) the frontier configs are two NEW files outside this builder's named set, each pinned
equal to its shipped machine outside the sweep block; (3) the Granite gate re-walks a ladder
whose fine stage plans are expensive to price, so it is the slowest test in the suite (965 s
of sweep plus the verify run) and nothing in it is sampled to make it faster; (4) no
fws_frontier/1 atlas document is emitted -- P7.6 asked for one and the two fields it named
(a composed shared-digital area and per-point utilization) are now on every row, but the
explorer loads that document from docs/qif/atlas/, which is another builder's directory.


## P7.6 — Folding card + macro time-lane view + knee explorer in the atlas

atlas.html gained three views behind two optional atlas blocks and one new document: the
FOLDING CARD (fws_fold/1 - one tensor, its mapping family side by side, spatial blocks as
boxes, bank passes as time lanes inside a box, the accumulator drawn where the K partials
meet, and area/time/energy/waste under each variant), the MACRO TIME-LANE drill-in
(fws_pass/1 - a macro's mux banks as lanes of one decode step, reusing the column-map
machinery), and the FRONTIER EXPLORER (fws_frontier/1 - area vs decode tokens/s, one mark
per complete provisioning, per-point utilization and waste in the table beside it, the knee
ringed, and a point clickable through to its atlas where the artifact exists). Both fixtures
are grounded: fixture_granite_folding.json copies 8 macro slots, their tiles and the cards
out of granite_4_0_h_tiny.json ROW FOR ROW (a test asserts byte-identity, which is the
regenerate gate a hand fixture can have) and folds the real L0 ssm_out_proj (K 3072 x N 1536
= 4718592 cells, floor exactly 2 slots: K-spread with 12288 B/step of partial transport vs
K-stack with an in-macro accumulator, identical cells, slots, 8 passes and 0 waste - P7 Fact
1 drawn) and the real L0 router (1536 x 64 owning a whole slot: 2260992 cells of waste,
itemized in P7.2's own remainder + tail vocabulary, against 24576 when it shares a bank with
ssm_in_proj's real 304-column remainder); fixture_frontier_granite.json carries the eight
REAL Wave-D candidates plus the shipped Granite atlas run as the one clickable point. 30 new
refusals (F1-F13, T1-T9, X1-X14) and 4 view rules (V10-V13) are in the self-test panel and
run headlessly by 14 new tests; the DOA register is enforced twice - a variant naming a dead
fold is refused, and an artifact that drops one of the register's entries is refused - and a
test pins the viewer's register as a superset of cim_timing.DEAD_FOLDS so the two copies
cannot drift. ASSUMPTIONS / DISCLOSED: no producer emits fws_fold/1, fws_pass/1 or
fws_frontier/1 yet, so both fixtures are fixtures; the folding card's SECONDS are a marked
placeholder (measured macro duty x measured decode step / 4 banks) while every pass COUNT,
cell count and waste figure is exact arithmetic on the copied shapes; every fold ENERGY is
reported UNCOVERED because the run itself discloses reduction_energy as an absent law; the
frontier's x axis exists only because of a marked placeholder shared-digital area term (the
card declares no area_mm2 - the reason Wave D's front collapsed to flat_area) and its
shared-digital utilization is null-with-a-reason, which is the field D28 most needs from
P7.4; its waste is the COLUMN census (the only one emitted), disclosed as smaller than D27's
cell census. E3's artifact did not exist under docs/qif/dse/ at hand-off, so the frontier
fixture is not wired to a real sweep output - fws_frontier/1 is published in SCHEMA.md as
the contract to emit.


## P7.7 — Filled-pipeline regime (D29/D30): staggered streams, one token per beat, D-stream state residency

THE PIPELINE IS ALWAYS FULL (D29) and the endpoints are gone (D30). mapping partitions the
layers into PP STAGES - mapping.layers_per_stage (int or list), else the declared pp
membership, else ONE STAGE PER CHIP - and D = the stage count, DERIVED and reported in
summary(), never configured: the serving surface refuses batch / batch_size / local_batch /
streams / resident_streams BY NAME in the mapping block, refuses
model_param.global_batch_size != 1 on any mapped fws_cim run (config.validate_model_config,
citing D29) and refuses it again at the mapper, and refuses
model_param.disable_embedding_unembedding: false by name (D30) with the mapper naming the
endpoint stage it would otherwise place. THE REGIME IS KEYED TO THE `mapping:` BLOCK: a
hardware config that declares one is a MAPPED run and therefore a filled pipeline; a config
that declares none (the ADJ-8 bridge points, every unmapped fixture) lowers under the
RETIRED lockstep regime and carries a serving_regime disclosure that says so by name, which
is the degenerate mode the license allows. THE LOWERING is a rotation: traversal j enters at
beat j and occupies stage d at beat j+d, a stage holds ONE stream at a time (the edge that
makes D streams resident rather than unbounded), a stream's next token waits for its
previous one, every analog op fires at M = 1, and the window is D-1 fill beats plus
decode_window steady beats so the steady state is REACHED and not assumed. THE BEAT EMERGES:
it is the median interval between token EXITS (finishes of completed traversals) with the
first interval held out as the fill transient and printed beside it, and a test pins beat ==
the slowest stage's measured service time. tokens/s = 1/beat (headline), per-stream = 1/(D x
beat), per-token latency is the MEASURED traversal with D x beat printed beside it as D29's
identity plus the residual (an uneven stage plan measures SHORTER than D x beat because a
stage is released when it is done, and that is stated rather than smoothed). STATE/KV
RESIDENCY (the George constraint) is a first-class block and a named verdict per stage:
every stage holds ALL D streams' state and KV for its layers, KV summed over the streams at
each stream's OWN context (streams are staggered one token apart; the representative context
is the MEDIAN of the window and the convention is disclosed), and the same per-(chip, layer)
quantities feed the per-chip verdicts - one accounting, two scopes. CONTENTION follows D29:
bank passes and bank sharing are measured per BEAT, cross-STAGE sharing is its own measured
term with its own banner, and within-stage sharing is measured at exactly 0.0 (serial-free).
GRANITE-4.0-H-TINY, the ADJ-1 headline: D = 10, beat 43.98 us, 22737 tokens/s against the
retired regime's 4486, 2274 tokens/s per stream, 5544 macros (the lm_head's 66 gone), 57.6
MiB of state on the worst stage for all 10 streams at a representative context of 1797.5
tokens, verdict fits. QWEN3.5-4B: D = 8, 12407 tokens/s, 68.2 MiB per stage. THE WINDOW is D
- 1 fill beats plus one more than the declared steady sample, so the transient interval can
be held out and STILL leave decode_window steady intervals to take the median of.
tests/test_qif_pipeline.py is 35 new tests with the hand arithmetic on MoE-small (1024 KV
bytes per token per layer, contexts 482/483 from the convention). Validator 436/436 (13 new
P7.7 rows, none removed). REGIME-CHANGE REWRITES (each old claim -> new claim -> why): (a)
test_qif_folding_mapping's 'cross-layer bank sharing costs exactly zero' kept its
measurement but the EVALUATOR now splits that term into within_stage (still 0.0, serial-
free) and cross_stage (contends every beat, its own banner) - the old zero was a fact about
a machine where layers ran one at a time; (b) the batch-4 accumulator hand computation (4 x
512 = 2048 results) is now the M = 1 machine and the granite comparison moved to 5544/4828
macros, 612 accumulator ops, -0.103% per beat - the counts fell by the lm_head (66 macros)
and the percentage fell because the beat is 17x shorter than the retired step; (c) the atlas
frontier fixture's knee moved from 1024 lanes to the smallest swept width and its law is now
a FIT (4e-5) rather than an identity, because the D29 beat is a MAX over stages while the
lockstep step WAS the scan chain; (d) the Granite DSE demo's '8x lanes buys 1.79x' became
1.277x for the same reason, stated in the test; (e) qwen's prefill law checks (gated
attention at m = 7168, the delta-rule work count, the prefill split) moved to a
qwen_lockstep fixture declared BY NAME, because a D29 run lowers no prefill at all; (f) the
P3 placement, PD and window tests declare LOCKSTEP by name and keep their batch-4 fixtures -
placement is regime-independent and rewriting them would have changed what they test.
ASSUMPTIONS / DISCLOSED: (1) the regime default is keyed to the DECLARED `mapping:` block,
which is config.py's own definition of a mapped fws_cim run - a caller-passed spec does not
by itself declare a regime, and `regime=` overrides everything; (2) configs/hardware-
config/fws_cim_llama7b_mapped.yaml DECLARES mapping.regime: lockstep, because the PD
picture, the P3.7 DSE round trip and the Llama2-7B closed-form comparison all read it
against pass-1/2 numbers that model exactly the batched-synchronous machine - its D29 twin
is the new configs/model-config/llama2_7b_fws_decode_inf.yaml, which the filled-pipeline
tests run on the same hardware; (3) the PD prefill half stays lockstep by PHASE (a prefill
has no stream to stagger) and an explicit filled+prefill request is refused by name; (4) a
stage's state is judged against ONE declared tier size, applied per stage, which is
PESSIMISTIC for a stage spanning chips and is disclosed; (5) resident KV uses the D29
context convention rather than a census of the lowered window, so a bounded window cannot
under-report a late stage's residency - the convention is the same one stamped on the priced
ops; (6) the fill transient is IN the DAG and priced, and only its exit interval is held out
of the beat. FILES OUTSIDE MY SET, all disclosed: tools/fws_qif_dse.py carries the new
pipeline metrics per candidate (beat, resident streams, per-stream rate, per-token latency,
max stage state, state verdict) and falls back from decode_step_median to the beat;
tools/fws_emit_atlas.py's _override_spec rebuilt the mapping spec field by field and DROPPED
the declared regime exactly as it once dropped the declared packing - it is now
dataclasses.replace, so no future field can be lost;
validation_scripts/validate_fws_cim_vs_optima.py gained check_qif_filled_pipeline (13 rows,
nothing removed); docs/qif/atlas/atlas.html's embedded frontier blob was refreshed from the
regenerated fixture (data only, no code).


## P7.8 — Library pricing + derived sizing

DIGITAL SILICON IS NOW MEASURED, AND THE ENGINE IS NO LONGER A CHOICE. (D32)
configs/hardware-config/digital_components_22nm.yaml and _12nm.yaml are the OPTIMA synthesis
library COPIED INTO THIS REPO value for value, 13 blocks each, every block carrying
`provenance: measured via synthesis, <tech>` and a header that names the source file, the
source reports directory (rtl/reports_<tech>_1000_efh) and the generator - and says out loud
that nothing in this repo fetches the source at runtime, because a run must never depend on
another project being present on a shared mount. cim_timing gained the loader
(SynthesisLibrary.load, cached by path) and it REFUSES BY NAME: a block the library does not
name gets a SynthesisLibraryError that names the block, lists the blocks that exist and
states the no-default rule (ADJ-4); an unknown technology dies at config parse
(cim.cards.<card>.synthesis_library), not inside a composition. (D32, composition) The
chiplet's area AND power are now compositions of measured blocks, in OPTIMA's collection
pattern and priced by our own laws: the SA fabric is ceil(rows/32) x ceil(cols/32)
GEMMINI_SYS_ARRAY blocks per array per replica plus one TRANSPOSER per replica, the softmax
pipeline is OPTIMA's create_softmax_collection term for term (FP_COMP w-1, FP_ADD 2w-1,
FP_MULT 2w, BF16_EXP w, BF16_RECIP 1, whose declared depth 20 is the same 20 our card
defaults to), the scan engine is lanes x (FP_MULT + FP_ADD + M_REG), and the per-macro pool
is its D12-derived adders and conv lanes plus one M_REG per lane - which retires P7.2's 'the
accumulator's holding register is a named gap'. Every composed row carries TWO provenance
labels because they answer different questions: unit_provenance (always measured-synthesis)
and count_provenance (derived-count or declared-count). (D31) vector_lanes is RETIRED as an
input. evaluate_fws runs a PROBE pass in which every vector op is COUNTED and costs nothing,
so the beat it measures is the one the ANALOG stages set - the circularity is broken by
construction, not by iteration - then derives the smallest integer width whose own priced
time fits that beat and prices the run again on it. The derivation INVERTS price_vector_work
exactly (same depth, same ceil, same per-call fill), so the engine sized is the engine
priced; the width is the max over stages because one card carries one width, and every non-
binding stage's slack is printed (D28). A beat shorter than the engine's own pipeline fill
is REFUSED by name rather than clamped, because lanes shorten the streaming term and never
the fill. An explicit vector_lanes still works as an OVERRIDE and rides a disclosure that
prints what the beat would have derived instead. GRANITE, the ADJ-1 headline: the card
declares no engine, 220 lanes are derived (against the 1024 somebody had written down), the
engine runs at 99.970% duty inside the 37.84 us analog beat, and the measured beat is 66.21
us for 15104 tokens/s against the retired 22737. That fall is the point of D31, not a
regression: the old number was a machine nobody had sized. Digital silicon totals 52.59 mm2
(10 chiplets x 3.711 mm2 composed + 6400 pool instances at 0.002418 mm2) and 62.94 W,
reported per component in evaluation.digital_silicon. QWEN3.5-4B: 169 lanes derived, beat
138.05 us, 7244 tokens/s against the retired 12407, 34.17 mm2 of digital silicon. THE
FRONTIER UN-FLATTENED. P7.6 said the front collapsed for want of a declared shared-digital
area; with the composed one the Granite demo sweep's front_shape moved from flat_area to
SPREAD - four non-dominated points over four distinct (throughput, silicon) corners, 12976
-> 13077 mm2 against 20148 -> 25648 tokens/s - and every mm2 of the spread is digital,
because Invariant W pins the analog floor. On the atlas frontier the DERIVED point is the
cheapest on the front. ENERGY: shared_digital_chiplet stopped being UNCOVERED; an op is
charged its engine's composed power for its own duration, labelled PARTIAL with both
directions of the approximation named (one average power per block, no dynamic/static split,
and an idle chiplet's leakage charged at zero). ASSUMPTIONS / DISCLOSED: (1) a vector LANE
holds both an FP_MULT and an FP_ADD, because the pricing law lets it retire either on any
cycle and the work laws report one ops total with no schedule - a mix-split engine would be
smaller and is NOT modelled, and the stance is stated rather than padded; (2) OPTIMA's per-
collection overhead fractions (0.0-0.2) are RECORDED in OPTIMA_COLLECTION_OVERHEADS and NOT
applied (D28), and its unexplained 9 TRANSPOSERs per replica are not copied - one per
replica is, with the difference named; (3) the block count ceils where OPTIMA keeps the
ratio, so a 40x64 fabric pays for 4 whole blocks and the idle remainder is disclosed; (4)
the composed chiplet is the SA fabric + softmax + scan engine only - activation SRAM,
interconnect and control have no block in the library - so it is a LOWER BOUND and says so;
(5) 'holds the beat' means digital-per-beat <= the ANALOG beat, i.e. the engine is at most
CO-BOUND and never binding: it does NOT make the measured beat equal the analog beat and
cannot, because a D29 stage holds one stream at a time and its analog passes and its scan
run in series; (6) a run that prices no scan op composes no scan engine (zero demand derives
zero silicon) and the composition says so; (7) the probe doubles the timeline work of a
derived-mode run. REGIME-CHANGE REWRITES (old claim -> new claim -> why), each named in the
source: (a) granite_e2e's 'a mapped run with no declared vector engine REFUSES by name' ->
'it DERIVES one from the beat and reports it', because D31 makes derivation the default; the
ADJ-4 refusal survives wherever there is no beat and test_qif_digital_ops still pins it on a
bare device; (b) the Granite headline 43.98 us / 22737 tokens/s -> 66.21 us / 15104, because
the width stopped being declared; (c) qwen's delta-rule HAND CHECK moved from 1024 lanes /
25776147 cycles to 178 / 148285160 - same law, same arithmetic, derived width - and its
prefill split (delta 50-60% -> 87%, attention 30-40% -> 10%, per-layer ratio REVERSED)
because a decode-sized engine is what a prefill actually runs on; (d) the DSE demo's
front_shape flat_area / front_ids == [selected] -> SPREAD over four points, and 'the
accounting always names an uncovered term' -> 'the accounting always names its PROVENANCE',
because D32 covered the term that was absent; (e) the atlas frontier fixture's
shared_digital_mm2 stopped being its one invented field and the ref.atlas point joined the
Pareto front; (f) P7.7's validator row 'every STEADY exit interval IS the beat (1e-3)' ->
'the exits CONVERGE (monotone, last two agree to 1e-3) and a sample that still contains ramp
says so BY NAME' - the old claim held while a fast declared engine made the stages nearly
equal; a derived engine makes the digital half comparable to the analog half and the UNEVEN
stage plan (4- and 3-layer stages) then needs more than D-1 beats to settle. That is now a
reported quantity (pipeline.beat_converged, steady_beat_spread) plus a
pipeline_beat_still_converging disclosure, not a widened tolerance; the pytest twin on MoE-
small was already green and is untouched. FILES OUTSIDE MY SET, all disclosed: fws_eval.py
(the probe/derive two-pass, the composed fabric energy, the digital_silicon report block,
the convergence disclosure); tools/fws_qif_dse.py (the silicon basis now names composed-vs-
declared provenance and carries shared_digital_area_provenance); configs/hardware-
config/fws_cim_granite_tiny_dse.yaml (re-synced to the shipped Granite point, which its own
gate requires) - its mapping_dse block still SWEEPS vector_lanes, which D31 retires; the
axis is left standing because that sweep and its artifact are P7.4/P7.5's deliverable, every
candidate now rides the override disclosure, and retiring the axis belongs to that wave;
docs/qif/atlas/fixture_frontier_granite.json + fixture_granite_folding.json + atlas.html's
embedded blob (data only, regrounded on the regenerated sources). REGENERATED: granite/qwen
reports and atlases, the PD atlas, the DSE demo sweep and its emitted config, the packing
comparison. Validator 454/454 (18 new P7.8 rows, none removed),
tests/test_qif_synthesis_library.py is 21 new tests with the hand arithmetic on the vector
lane (220 x 2829.418 um2 = 0.62247196 mm2), the softmax census, the GEMMINI count and the
lane derivation (24 lanes hold a 43.98 us beat for 1e6 ops; 23 do not).


## P7.9 — Adversarial-audit fixes (F1–F3)

THE THREE REAL DEFECTS, and what each one cost.

(1) THE FRONTIER'S MISSING ENGINE (critical). `tools/fws_qif_dse.py` composed the
shared-digital chiplet at the MAPPING stage, before `evaluate_fws` derives and
installs this point's vector engine (D31). `CimDeviceModel.resolved_vector_lanes()`
was therefore `None` and the composition returned the SA fabric plus softmax and
nothing else — 0.6 to 1.3 mm2 of scan engine per chiplet dropped from every row of
both shipped curves, while each row still asserted
`shared_digital_area_provenance: composed-measured` and `uncovered_terms: []`. It
contradicted the sibling artifact `docs/qif/atlas/granite_4_0_h_tiny_report.json`,
which composed the engine for the same machine (D21). No gate could see it: the
validator checked the provenance STRING and the internal sum, and the
--emit-config/--verify round trip checked eight quantities of which none was an
area. The silicon accounting is now FINALISED AFTER PRICING, off the same device
the evaluator just sized; the declared `max_silicon_mm2` budget is re-checked there
and refuses by name if the derived engine pushes a point over; the verify round
trip cross-checks `shared_digital_area_mm2_per_chiplet` and `vector_lanes` against
the simulator's own `digital_silicon` block; and a new validator row asserts the
two agree on every valid point.

(2) THE WINDOW-SENSITIVE BEAT (critical). `_build_pipeline_metrics` held out
exactly ONE inter-exit interval as the fill transient and took the median of the
rest. That is exact only for a pipeline of equal-service stages; Granite's stage
plan is uneven, so the sample still contained ramp, the run reported
`beat_converged: false` with a 21.6% spread — and published the headline anyway.
Measured consequence: 15104 tokens/s at `decode_window` 3 against 16935 at
`decode_window` 2, a 12.1% move in the ADJ-1 headline from a knob that should not
touch it. The beat is now the median of the longest SETTLED TAIL of the steady
sample (`_converged_beat_tail`): the longest suffix that agrees with its own
median to 1e-3. Every dropped interval is printed as `ramp_beat_intervals_s` and
rides a `beat_read_from_the_converged_tail` relaxation; `beat_converged` now means
two intervals were SEEN to agree, and a run where none do rides
`pipeline_beat_NOT_converged` and says so in the headline metric's own basis. The
D31 derivation carries the PROBE beat's convergence too, under
`derived_engine_probe_beat`, because a still-ramping probe beat is a tighter cycle
budget and therefore systematically over-sizes the engine. Granite's beat moved
15103.6 -> 15102.7 tokens/s (0.006%) and now converges; Qwen is unchanged.

(3) INVARIANT W WAS NOT IN FORCE (critical). No shipped config declared
`mapping.packing`, so every artifact ran the DEDICATED law while the frontier's
own status entry said "the analog term is identical on every point ... because
Invariant W pins it at the cell floor". Both halves were wrong: D27's law was not
running, and the analog term is constant because the LADDER settled on one slot
count, not because a floor pins it. `packing: dense` is now declared on
`fws_cim_granite_tiny.yaml`, `fws_cim_qwen3_5_4b.yaml`, both `_frontier.yaml`
files and `fws_cim_granite_tiny_dse.yaml` (which a gate pins equal to the shipped
card). On the whole-macro cards dense and dedicated coincide tile for tile, so no
headline number moved; on the frontier, where `bank_depth` 1 is swept, the laws
SEPARATE and the curves were regenerated under the law D27 mandates. Every sweep
point now carries its own CELL census — macros, the global cell floor and waste% —
so what Invariant W is worth at each point is a number and not a claim. The
Granite frontier's `arrays_per_chip` rungs were re-derived for the dense law
(120.625 macros per layer at bank_depth 1, measured at two granularities) because
the old rungs were the dedicated law's per-layer counts and no longer described
any tight capacity.

ALSO FIXED, from the same audits: per-stage residency is judged against the
declared ON-CHIP tier (`tech_param.SRAM-L2.size`), where D29's resident state
actually lives, with the 8 GiB activation stub beside it as
`activation_tier_verdict` — the P4 `MemoryVerdict` rows keep the activation tier
so the DSE's capacity gate is unchanged; the cross-stage bank-sharing test stopped
asserting `>= 0.0` (true of every value) and now asserts the measured zero on the
shipped card AND has a positive twin on a machine where the shared macro really
collides; `tests/test_qif_folding_mapping.py`'s regime note said the run fixtures
go through a `_d29_model` helper that was never called from anywhere — the helper
is gone and the note says what the file runs (deliberately lockstep, because
packing is regime-independent); the "FACT 3, MEASURED" title was retired, since
P7_folding.html marks Fact 3 [RETIRED]; the DOA register's `replication` and
`prefill_folding` rationales were restated on D27 and D25 rather than on D15 and
Fact 3; `synthesis_library` refuses a file path at the config surface, so no run
can reach outside the repo for the D32 library; `OPTIMA_COLLECTION_OVERHEADS`
gained the 0.4 depthwise-conv pad it was missing and both library headers now say
0.0-0.4, name M_REG's own TODO in the source and name the two measured blocks no
composition uses; the D32 pool term names the second, demand-scaled pool sizing
beside it; the atlas frontier fixture gained its third silicon term, D27's cell
census, and a corrected label on the DERIVED ninth point; and P7 Fact 2's "the
0.06% analog duty was a mis-provisioned config" is corrected on the plan page,
because the D29/D31 runs reproduce it.

NOT FIXED, and why. The (area, throughput) trade a wider-than-derived engine
would buy (+27% tokens/s for +0.78% area on the Wave-D demo) is a DECISION for
George, not a defect: D31 is explicit that the engine is derived and never swept,
and the sizing criterion is disclosed. It is raised in the report rather than
acted on.


## P7.10 — ADJ-9: derive the engine to the ANALOG FLOOR (D31-v2)

THE TARGET MOVED, AND NOTHING ELSE DID. D31-v1 sized the scan/vector engine to the
analog BEAT — the slowest stage's probe time — so a stage faster than the beat got an
engine sized against somebody else's stage and became its own binding term. ADJ-9
retargets it to each stage's OWN analog m-pass: `derive_engine_sizing` now takes an
`EngineDemand.analog_time_s` per stage and answers the smallest integer width for which
`vector_cycles_at(ops, lanes, depth) <= floor(analog_time_s x clock)` on EVERY stage,
which is the same law inverted against a 24x tighter budget. The law, the probe pass,
the two-pass argument and the no-margin rule are untouched; only the number the
inversion is evaluated against changed.

THE TARGET IS A MEASUREMENT, AND IT IS WIDTH-INVARIANT. `_engine_demand` reads each
stage's analog time off the PROBE timeline as the UNION of the busy intervals of that
stage's `analog_macro` ops in the measurement beat — a union and never a sum, because
macros of one stage fire in parallel and a sum would report Granite's 1.6 us stage as
44.5 us. Two properties make it the right target: an analog op's duration does not move
with a lane count and the union ignores the gaps the engine's own time opens between
those ops, so D31's two-pass argument survives intact and no iteration is needed; and
the union EXCLUDES the gaps, so it is the strictly smaller of the two candidate
readings (union vs. first-issue-to-last-finish span) and therefore the tighter target.
Sizing against the span would let the engine hide inside time the analog side is not
working, which is the margin D28 forbids.

WHERE THE CRITERION CANNOT BE REACHED, IT SAYS SO. Two cases fall back to the analog
beat and are counted in `unreachable_stages` and named in the disclosures by
`target_kind`: a stage with NO analog work in the beat (`no_analog_work_in_stage`) has
no analog time to be bound by at any width, and a stage whose analog time is shorter
than the engine's own pipeline FILL (`analog_stage_time_below_engine_fill`) cannot be
held at any width because lanes shorten the streaming term and never the fill. Neither
is clamped and neither is padded. On the shipped Granite and Qwen machines neither case
occurs: 10 of 10 and 8 of 8 stages are analog-bound.

THE NUMBERS. GRANITE-4.0-H-TINY: 5479 lanes derived against D31-v1's 220, binding stage
2 at 99.737% duty inside its own 1.6 us analog m-pass, measured beat 66.21 -> 39.04 us
= 15103.6 -> 25617.6 tokens/s (+69.6%), digital silicon 52.587 -> 201.386 mm2 and
62.94 -> 287.50 W, total 12983.2 -> 13132.0 mm2 (+1.15%). QWEN3.5-4B: 7971 lanes
against 169, binding stage 0 at 99.931%, 138.05 -> 70.71 us = 7244 -> 14143.1 tokens/s
(+95.2%), digital 34.17 -> 210.772 mm2, total 8115.8 -> 8292.4 mm2 (+2.18%). Both
headline artifacts, both atlases, the PD atlas, the packing comparison, both frontier
curves and the demo sweep were regenerated.

THE FINDING, AND IT CHANGES THE PHYSICS STORY. ADJ-9 asks for the analog side to be the
binding term "wherever physically reachable". The derivation guarantees the half it can
— the DERIVED engine fits inside every stage's analog m-pass — and it cannot guarantee
the other half, because a stage also runs silicon D31 derives no width for. A new
MEASURED block says which: `derived_engine_sizing.beat_setting_stage` names the stage
with the longest span in the probe slice and itemizes its terms by op block, and an
`analog_floor_reachability` disclosure carries the verdict. On Granite the beat-setting
stage spends 18.60 us on `attention_qk` and 15.35 us on `attention_pv` against 1.60 us
of analog m-pass; on Qwen it is 34.23 us and 30.71 us against 1.52 us. THE MACHINE IS
BOUND BY THE ATTENTION SYSTOLIC FABRIC, whose geometry is DECLARED card knobs
(`cim.fabric` rows x cols x num_arrays, softmax_lanes) and a D28 sweep axis, not a D31
derived engine. The consequence is visible on the atlas frontier: the ADJ-9-derived
Granite point (5479 lanes, 13132.0 mm2, 25617.6 tokens/s) is WIDER than the retired
sweep's widest declared point (4096 lanes, 13092.9 mm2, 25673.8 tokens/s) and is
DOMINATED by it — past the width at which the scan fits under the attention term, lanes
buy area and no throughput. That is not a defect in the derivation and not an argument
against ADJ-9; it is what ADJ-9 buys and what it does not, and it is the question P7.9's
open item now becomes: whether SA geometry joins the frontier's axes.

THE AXIS IS REFUSED BY NAME. `vector_lanes` moved from `RETIRED_AXES` (spellable,
labelled) to `REFUSED_AXES` in tools/fws_qif_dse.py and out of `AXIS_TARGETS` and
`apply_point` entirely: a sweep that declares it now raises at parse with the axis named
and ADJ-9 quoted, exactly as D26 refuses a dead fold. An explicit
`cim.cards.<card>.vector_lanes` on ONE machine is still a legal override riding its own
disclosure (D31); what is refused is making it an axis. The Wave-D demo sweep is FROZEN
and labelled at `docs/qif/dse/granite_lanes_banks_retired/` — its payload carries a
`retired` block naming the decision, why it is kept and how to read it — because it is
the +27%-for-+0.78% comparison ADJ-9 was adjudicated on and the atlas frontier fixture is
grounded on its eight rows. Its D31-legal successor is
`docs/qif/dse/granite_capacity_banks/`: the same config over `arrays_per_chip`
{604, 640, 965, 1207} x `bank_depth` {1, 2}, 8 of 8 valid, and its finding INVERTS the
old one — at a fixed bank depth CHIP CAPACITY MOVES NO THROUGHPUT AT ALL (26051.9
tokens/s on all four) and doubles the silicon (12433 -> 24631 mm2), which is Invariant W
(D27) drawn on a cross product. The one real trade left on it is the BANKING axis, and
it is now a two-term trade: a finer bank shortens the analog m-pass, which under D31-v2
IS the sizing target, so bank_depth 1 derives a WIDER engine (6509 lanes against 6125)
that is faster and costs more.

THE ATLAS GAINED `svc` LINKS (charter addendum). The atlas drew the `act` chip boundary
and nothing at all for the relationship between an analog chip and the shared digital
chiplet that runs its attention and its scan (D13), so the pipeline map had no wires to
the silicon doing half the work. `FwsEvaluation.atlas_service_links` emits one `svc` row
per (chiplet -> analog chip) service relationship, read off the LOWERED DAG rather than
by re-implementing the builder's assignment rule: every shared-digital op names its
chiplet and the LAYER it serves, and the placement names the chip that holds that layer.
The bytes are MEASURED — every priced `link` op in the measurement beat whose two
endpoints are an analog chip and a chiplet, summed with the two directions kept apart
(Granite: 5120 B of operands out + 3072 B of results back per beat on each of the four
attention stages). SIX OF TEN GRANITE WIRES CARRY 0 B AND SAY WHY: `_recurrent_block`
places the scan on the chiplet with no transfer op on either side, so the recurrent
hops are an UNPRICED COMPONENT named in every basis string (`_UNPRICED_SERVICE_BLOCKS`)
rather than a total absorbing an estimate. `role: svc` is documented in
docs/qif/atlas/SCHEMA.md, accepted by atlas.html's loader with its own dash pattern and
legend row, and an undocumented role still refuses (R6).

REGIME-CHANGE REWRITES (old claim -> new claim -> why), each named in the source:
(a) synthesis_library's `test_derive_vector_lanes_is_the_exact_inverse_of_the_pricing_law`
    24 lanes against a 43.98 us BEAT -> 667 lanes against a 1.6 us analog M-PASS, same
    law, 28x tighter budget, with the old number kept beside it as what it replaced;
(b) `test_one_card_carries_one_width_and_the_slack_stays_visible` -> 
    `test_every_stage_is_sized_to_its_own_analog_m_pass`: the binding stage is no longer
    the one with the most work, it is the one with the worst work/analog-time ratio;
(c) granite_e2e's `test_the_scan_engine_is_the_binding_resource_and_says_so` ->
    `..._runs_at_peak_and_is_no_longer_what_binds`: the peak claim survives at 0.9498
    (a wider engine pays the same fill over fewer streaming cycles) and the binding claim
    is replaced by the measured attention term;
(d) the Granite headline 66.21 us / 15103.6 -> 39.04 us / 25617.6, 220 -> 5479 lanes,
    52.587 -> 201.386 mm2, energy 9.27615e9 -> 9.42996e9 pJ;
(e) qwen's delta-rule HAND CHECK 178 lanes / 148,285,160 cycles -> 7676 / 3,438,628 on
    the lockstep fixture, and its prefill split REVERSES AGAIN — delta 87% -> 13%,
    attention 10% -> 67%, analog 6% -> 63% — because a properly sized scan engine leaves
    the stack attention-bound, which is the same finding as (c) on the other model;
(f) the demo sweep's axes and finding (above), and its banking energy delta, which was
    "the whole delta is analog_arrays" and is now two terms of opposite sign because the
    derived width follows the bank depth;
(g) the atlas frontier fixture's ninth point: "the derived point is the cheapest one on
    the frontier" -> it is DOMINATED, and the relaxation
    `the_eight_swept_points_sweep_a_RETIRED_axis` became `..._a_REFUSED_axis`.

ASSUMPTIONS / DISCLOSED: (1) the analog stage time is a UNION of busy intervals, so it
measures the stage's analog OCCUPANCY and not its analog critical path — the two differ
by the gaps, and the union is the smaller and therefore the stricter target; (2) it is
read on the PROBE pass, in which vector ops cost nothing, so a stage whose analog ops
would OVERLAP differently once the engine's time is priced could measure a slightly
different union in pass B — the DAG serializes them on every shipped card and the
quantity is invariant there; (3) the fallback for an unreachable stage is the analog
BEAT, which is the widest budget this derivation uses, and it is named rather than
clamped; (4) D31 still derives only the scan/vector engine — the per-macro pool is
already sized to the macro's own result rate (D12) and is therefore at the analog floor
by construction, and the SA fabric and softmax pipeline are declared card geometry; (5)
the `svc` link bytes cover only traffic this repo LOWERS as a link op, and the recurrent
hops are named as absent rather than estimated; (6) the derived engine's duty against
the MEASURED beat is now very low (Granite's binding stage spends 1.596 us of a 39.04 us
beat) — that is idle silicon, it is what ADJ-9 bought, and D28 requires it to be visible,
which the per-device-class utilization banner and the beat_setting_stage block both do.


## P7.11 — ADJ-10: ALL composable digital engines derive to the ANALOG FLOOR

WHAT ADJ-9 LEFT BEHIND. ADJ-9 derived the scan/vector engine to each stage's own analog
m-pass and then MEASURED that the machine was still not analog-bound: the beat-setting
Granite stage spent 18.60 us on `attention_qk` and 15.35 us on `attention_pv` against a
1.60 us analog m-pass (Qwen 34.23 / 30.71 against 1.52). That fabric was DECLARED card
geometry — `cim.fabric.num_arrays: 2`, `softmax_lanes: 1` — and no derivation sized it.
ADJ-10 is the extension: every digital engine whose width is a COMPOSITION OF MEASURED
SYNTHESIS BLOCKS derives up until the analog m-pass binds, or until copies stop buying
time. The SA fabric derives by integer copies of the measured `GEMMINI_SYS_ARRAY` 32x32
block and the softmax pipeline by copies of OPTIMA's measured per-lane census. `rows` and
`cols` per array are NOT derived and are not swept: they are the geometry the systolic
closed form was validated at, bit-exact against recorded ScaleSim outputs, and the
fill/drain surrogate is `3 x rows` — deriving a new geometry would move a VALIDATED law
onto an unvalidated shape (ADJ-4). The per-macro pool needed no retarget: D12 already
sizes it from the macro's own result rate, which is the analog floor by construction.

THE FOLD LAW, EXTENDED HONESTLY. The pass-1 law folds heads and streams into K to model
back-to-back dual-buffered runs: `QK = sa(m, n, k x folds)`, `PV = sa(m, k, n x folds)`
with `folds = heads_per_replica x streams`. ADJ-10 makes `num_arrays` a COUNT OF
CONCURRENT FOLDS: the arrays split into a QK group of `a_qk` and a PV group of `a_pv`,
the folds are dealt across the group, and an array in a group of `a` carries
`ceil(folds / a)` of them, so its contraction dim is that many folds instead of all of
them. `fold_group_split` picks the partition that MINIMISES `QK + PV`, because the
lowering places three SERIAL ops (qk -> softmax -> pv) and the sum is what the timeline
measures and what the criterion is taken on; ties break toward the balanced split and
then toward the smaller `a_qk`, so the answer is deterministic. AT `num_arrays = 2` THE
SPLIT IS (1, 1) AND THE LAW IS THE PASS-1 LAW BIT FOR BIT, which is why no OPTIMA parity
number moves — every shipped card declares 2.

WHERE IT STOPS, AND WHY THAT IS NOT A CAP. Fold concurrency SATURATES at `2 x folds`:
past that another copy of the measured block carries no fold. What is left is one array's
own walk — `ceil(n / cols)` column passes, each costing `k + rows + cols - 2` cycles —
and no COUNT of measured blocks shortens it. Stages where the analog m-pass is still not
reached at saturation are counted in `saturated_stages`, named by
`target_kind = fold_concurrency_saturated_below_analog_time`, and given the SATURATION
width. THIS IS DELIBERATELY NOT ADJ-9's RULE. ADJ-9 hands an unreachable scan stage the
analog BEAT as its budget, which is right there because its unreachable cases (no analog
work at all; analog time under the engine's own pipeline fill) have no floor to walk to.
The fabric has one, and falling back to the wider beat budget would derive a NARROWER
fabric than the machine can use — leaving measured throughput on the table for silicon
that buys time, the trade ADJ-9's own rationale rejects. Neither rule pads and neither
clamps.

THE NUMBERS. GRANITE-4.0-H-TINY: `num_arrays` 2 -> 8 and `softmax_lanes` 1 -> 12; the
scan width does NOT move (5479 either way — its target is the analog m-pass, which did
not move). Beat 39.04 -> 15.96 us = 25617.6 -> 62661.5 tokens/s (+144.6%); shared
chiplet 18.5908 -> 27.8703 mm2, digital total 201.386 -> 294.181 mm2, machine
13132.0 -> 13224.8 mm2 (+0.71%). QWEN3.5-4B: 2 -> 8 arrays, 1 -> 16 softmax lanes, 7971
lanes unchanged; 70.71 -> 24.56 us = 14143.1 -> 40717.1 tokens/s (+187.9%), digital
210.772 -> 285.216 mm2, machine 8292.4 -> 8366.9 mm2 (+0.90%). Both headline reports and
atlases, the PD atlas, the packing comparison, both frontier curves, the capacity/banks
demo sweep and the two hand fixtures were regenerated.

THE FINDING, AND IT IS THE HEADLINE. THE MACHINE IS STILL NOT ANALOG-BOUND, AND WHAT
BINDS IT NOW IS A REAL LIMIT WITH ARITHMETIC BEHIND IT: THE DECLARED 32 x 64 ARRAY
GEOMETRY. Granite's beat-setting stage 6 spends 6.877 us on `attention_qk` against a
1.600 us analog m-pass — 4.3x, down from 11.6x — and that 6.877 us is the FLOOR:
`folds = heads_per_replica 4 x streams 1 = 4`, so at 8 arrays each of the 4 QK arrays
carries exactly one fold and sees `k = head_dim = 128`; the run costs
`ceil(3/32) x ceil(1800/64) x (128 + 32 + 64 - 2) - 1 = 1 x 29 x 222 - 1 = 6437` cycles
plus the 96-cycle fill/drain, at 0.95 GHz. A ninth array carries no fold. Qwen is the
same shape at `k = 256`: `29 x 350 - 1 = 10149` cycles = 10.784 us against 1.520 us.
The residue is the `ceil(n / cols) = 29` COLUMN PASSES of a 64-column array, and only two
things move it: a wider `cols` (a CARD CHANGE a human makes, because rows x cols is the
validated geometry) or splitting one fold's N across arrays (a DATAFLOW CLAIM — how the K
matrix is broadcast, how the column tiles merge — that no recorded reference in this repo
covers). Both are named and neither is taken. A new MEASURED block carries it:
`evaluation.digital_silicon.binding_term` names the beat-setting stage of the PRICED
timeline — the machine that gets BUILT, every derived width installed — and itemises its
terms by op block, with a `binding_term` disclosure that says which real limit it is in
words. It is a DIFFERENT quantity from `derived_engine_sizing.beat_setting_stage`, which
describes the machine the derivation ran AGAINST, and the two are never merged (D21).

WHAT IT REVERSES. The atlas frontier fixture's ninth point was DOMINATED under ADJ-9
(5479 lanes, 13132.0 mm2, 25617.6 tokens/s against c006's 4096 declared lanes at
13092.9 mm2 and 25673.8). With the fabric derived it measures 62661.5 tokens/s — 2.4x the
fastest swept point — for 131.9 mm2 more silicon, so it is BACK ON THE FRONT and is now
its fastest point. The eight swept points are frozen at the declared 2-array fabric and
can never be re-walked, which is exactly what makes the comparison worth drawing.

THE AXES. `num_arrays` and `softmax_lanes` join `vector_lanes` in
`tools/fws_qif_dse.py REFUSED_AXES`: a sweep that declares one raises at parse with the
axis named and ADJ-10 quoted. `rows` and `cols` are in neither register — neither swept
nor derived. A SINGLE MACHINE may still pin the two derived widths through the new card
knobs `cim.cards.<card>.fabric_num_arrays` / `fabric_softmax_lanes`, which skip the
derivation and ride `CimDeviceModel.fabric_sizing_disclosures` naming themselves an
OVERRIDE (the same seam D31 gives `vector_lanes`, and it does not print what the
derivation would have returned, because a pinned card never runs one — the P7.9
correction applies for the same reason).

THE BRIDGE GATE, AND WHY IT MOVED WITHOUT WEAKENING. The ADJ-8 bridge compares the frozen
CLOSED FORM against the placed DAG on the degenerate case. The closed form has no timeline
and prices the DECLARED fabric; the DAG now derives one, so the two sides began describing
TWO MACHINES and every attention row compared a derived fabric against a declared one.
The gate now PINS both widths on the DAG side to the config's own declared seed, using the
ordinary card override, and the pin is named in `fws_bridge.EXCLUSIONS` as
`derived_fabric_pinned_to_the_declared_seed`. No row is dropped, no tolerance moves, every
attention cycle count is still compared EXACTLY, and the gate GAINED five rows (one per
bridge point) asserting the pin took effect — 519 checks became 524.

ASSUMPTIONS / DISCLOSED: (1) a stage's attention calls SUM, because the lowering puts a
chip's fabric ops on one chiplet and decode runs a stage's layers in sequence; a stage
plan spread over several chips would make the sum an upper bound, and the docstring says
so; (2) the partition minimises the SERIAL sum while the closed-form
`attention_call_timing` still reports `max(qk, pv) + fill_drain` over the same partition —
that is the already-disclosed `attention_op_folding` divergence (ADJ-8: the DAG wins), and
at saturation, where both shipped machines land, the two objectives agree because each run
is at its own floor; (3) `qk_arrays` / `pv_arrays` on a per-stage row are the LAST call's
partition and a report field only — a stage's calls are its attention layers and share
their dims on every shipped machine; (4) fws_mapping still enumerates
`num_arrays x replicas` engine SLOTS per chiplet from the DECLARED seed, because it places
before any beat is measured; those slots hold no tile, carry no device and price no time,
and the derivation's own disclosures reconcile the two spellings by name; (5) the fabric
derivation is taken on the same probe pass and the same per-stage analog m-pass
measurement as ADJ-9's, and the order (fabric first, then scan) moves no number because
both targets are properties of the analog side; (6) `replicas` is untouched and remains a
declared knob — it replicates the WHOLE fabric including the softmax pipeline, where
`num_arrays` is the finer grain, and ADJ-10 names `num_arrays` as the quantity to derive.
