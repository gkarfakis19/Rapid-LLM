{
  "summary": "Audit every sizing/derivation law in the QIF perf model against one methodology: minimum hardware, matched to the analog beat",
  "agentCount": 13,
  "logs": [
    "memory-store: 38 laws read, 7 findings",
    "fabric-sa: 28 laws read, 6 findings",
    "analog-beat: 51 laws read, 7 findings",
    "shared-engine: 30 laws read, 8 findings",
    "counts-dse: 29 laws read, 7 findings",
    "macro-pool: 30 laws read, 6 findings",
    "survivors: 21 findings over 206 laws (20 refuted)"
  ],
  "result": {
    "methodology": "# THE QIF SIZING METHODOLOGY, THE COMPLIANCE TABLE, AND THE VIOLATIONS

Synthesis of six audit lanes (206 laws) and 21 adversarially verified findings.
Repo: `/app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM`

---

# 1. THE METHODOLOGY

*Written to be pasted into `docs/qif/DECISIONS.md` as the governing decision. It sharpens the owner's M1–M5 and adds five rules the inventory proved were needed. Every rule has a test. A reader must be able to take any law in the repo and return a verdict.*

## ADJ-16 — THE SIZING METHODOLOGY (supersedes the informal M1–M5)

**Scope.** These rules govern every law that produces a count, a width, a duration, an area, a power or an energy. Accuracy is out of scope (D23).

---

### M1 — The analog side sets the beat.

The macro count comes from the weights alone: enough macros to hold every weight once, and no more (Invariant W, D27). The beat comes from the card geometry, the muxing and the mapping. Nothing else may set the beat.

If something else does set the beat, the tool **reports that fact**. It does not widen digital blocks until the fact goes away.

**Test.** Change the weights, the card geometry, the muxing or the mapping. The macro count and the beat must move. Change anything else. They must not. The artifact must name the term that binds the **priced** timeline, measured on the priced timeline, never on a probe.

---

### M2 — A block is sized to the work of one beat.

Size every digital block from the work it must do inside one beat. The basis is `work_this_run_per_beat / beat_length`.

Forbidden bases: a device peak streaming rate, a declared constant, a round number, a duty the regime never reaches.

**Test.** Name the numerator and the denominator out loud. The numerator must be work this run measured. The denominator must be a beat or a stage's own window. A law that divides a per-second device rate by a clock frequency fails M2.

---

### M3 — Counts follow from work, never from declaration.

The count of chips, chiplets, slots, lanes and blocks is the count that carries work. A block with no demand is not composed. A slot that holds nothing is not charged. A chiplet no chip can address is not built. A capacity is the capacity that is used.

**Exception.** A measured physical constant is a fact, not a declaration. Array rows, array columns, block geometry and clock frequency are facts.

**Test.** Set the demand for the block to zero. If its count does not fall to zero, the law fails M3.

---

### M4 — Matched means matched, in both directions, and at the device.

A block above the beat's demand is waste and is a defect. A block below the demand is a violation and is a defect. \"Conservative\" is not a defence (D28).

Matching is judged **at the device, not at the block**. Blocks that share one queue share one budget. Sum the per-beat times of every op class a device runs, then compare that sum to one target.

**Test.** For every derived width print `work_per_beat / capacity_per_beat`. It must be near 1. For every device print the sum of the times of all op classes placed on it. That sum must fit one budget, once.

---

### M5 — Every number comes from a measured law, or is named absent — in structure.

A term with no law says so. The statement is a **structured field**, not prose, and it reaches the shipped artifact. A measured zero and an unasked question never read alike.

- A zero-count row is printed as zero. It is never dropped from the census.
- A fallback target keeps its own name. It never reads as the matched target.
- A gate that cannot fire is not a gate.
- Every published number carries a provenance: `measured`, `derived`, `declared` or `absent`.

**Test.** Grep the shipped artifact for the disclosure text. If the text is not in the artifact, the disclosure does not exist.

---

### M6 — Resident is not active. Count what fires.

The per-beat work census is the work the filled pipeline performs in one beat. Weights are resident every beat; weights are not read every beat. A sub-token load is never rounded up to a full pass.

**Test.** For each op the lowering emits, ask how many times the regime fires it per beat. Under D29 local batch is 1, so a MoE layer fires `top_k` experts, not `E`.

---

### M7 — One quantity, one name, one law.

Each quantity has exactly one implementation. Two derivations never share a key name. A law is never re-implemented inline beside its caller. A second law for the same quantity is deleted, or it becomes the only law. A name states the quantity it holds.

**Test.** Grep the quantity. If two sites compute it, the model fails M7.

---

### M8 — A total sums one unit. A bound is stated in the unit it bounds.

Never sum a package footprint with a silicon area. Never bound a macro-slot count with an array count. A budget compares like with like.

**Test.** For every sum, print the unit of each term. For every bound, print the unit of the bound and the unit of the thing bounded.

---

### M9 — The time law and the silicon census describe the same machine.

If a timing law retires N results per cycle, the composition builds N units. If an op runs on a subset of a device, the energy law charges that subset. Silicon a timing law assumes is silicon the machine buys. A capacity that gates feasibility has an area law.

**Test.** For each timing law, list the hardware it assumes and find each item in the composition. For each energy charge, list the blocks active during that op.

---

### M10 — An analysis knob never sizes hardware.

The measurement window, the probe pass, the lowered beat count and the decode window are analysis choices. They may size a **reported metric** (ADJ-6). They never size built silicon, a chip capacity, or a refusal.

**Test.** Change the analysis knob. Any built quantity that moves fails M10.

---

### The one conflict this methodology creates, and the ruling it needs

**ADJ-9 / ADJ-10 as written conflict with M2 and M4.** They size every digital engine to a **stage's analog m-pass window**, not to the beat. On Granite that window is 1.32e-6 s against a 1.53e-5 s beat — an 11.6x tighter target than the pipeline imposes. The stated rationale, \"digital silicon is nearly free against the analog floor, so never leave the machine digital-bound\", is a safety-margin argument, which D28 forbids by name. The evidence that it over-shoots is on the shipped points: derived `vector_lanes` of 39,033 and 722,944, and a Falcon candidate carrying 8,194 mm² of shared-digital silicon — more than its entire analog macro array.

**Recommended ruling (M1 + M4).** Size each device against **the achieved beat minus the co-resident occupancy on that device**, iterated to a fixed point with the existing two-pass probe. If the analog side then does not bind, report that as an M1 result and name what does bind. Widening digital until analog \"would\" bind hides the M1 answer instead of producing it.

This ruling is the owner's to make. Everything below is scored against M1–M10 with ADJ-9's target marked as the conflict, not silently as a pass.

---

# 2. THE COMPLIANCE TABLE

Deduplicated: 206 inventory entries → 181 distinct laws. Verdicts are against M1–M10.
`OK` = compliant. `OK(fact)` = a measured physical constant (M3 exception). `OK(named)` = an absence correctly named. `VIOL` = non-compliant.

## 2.1 Analog beat and the weight census (`cim_timing.py`, `fws_mapping.py`, `program/fws_build.py`)

| Law | file:line | Sizes against now | Verdict | Rule |
|---|---|---|---|---|
| `vec_cycles` | cim_timing 3117 | `adc_mux * slice_cycles` | OK(fact) | M1 |
| `vec_latency_s` | cim_timing 3122 | vec_cycles / analog clock | OK | M1 |
| `analog_gemm_time` | cim_timing 3127 | `m_tokens * vec_latency_s`, K/N-independent | OK | M1 |
| `analog_op_time` (ADJ-4) | cim_timing 5912 | active column sets × slice_cycles + bank switches | OK | M1 |
| `active_column_sets` | cim_timing 5934 | max over macros of sets converted | OK | M1 |
| `price_tiled_op` | cim_timing 5947 | max-macro sets; energy per vec × shots × sets/mux | OK | M1 |
| `analog_stage_energy_pj` | cim_timing 3135 | closed-form parity twin | OK | M1 |
| `arrays(k,n)` | cim_timing 3131 | `ceil(k/rows)*ceil(n/cols)` | OK | M1 |
| `enumerate_tiles` | cim_timing 5715 | row/col blocks, `mux//bank` tiles per macro | OK | M1 |
| `tile_macro_count` | cim_timing 5809 | distinct macro ids over tiles | OK | M1 |
| `bank_columns` / `_packing_bank` | cim_timing 6050 | `stored_columns_per_set * bank` | OK(fact) | M1 |
| `dense_blocks` | cim_timing 6070 | card rows × bank width | OK(fact) | M3 |
| `dense_pack` (Invariant W) | cim_timing 6113 | `ceil(banks_used / banks_per_macro)` | OK | M1 |
| `cell_floor_macros` | cim_timing 6256 | `ceil(real_cells / cells_per_macro)` | OK | M1 |
| `packing_summary` | fws_mapping 906 | sum over per-chip packings | OK | M1 |
| `price_accumulation` drain/stall | cim_timing 6264/6330 | `ceil(results/lanes)` vs one analog pass | OK* | M2 |
| `price_reduction` | cim_timing 6016/6017 | `depth + ceil(results/lanes) - 1` | **VIOL** | M9 |
| `macro_result_rate_per_s` | cim_timing 6467 | peak back-to-back ADC rate 1.92e10/s | **VIOL** | M2 |
| `digital_pool_sizing.lanes` | cim_timing 6514 | `ceil(peak_rate / pool_clock)` | **VIOL** | M2 |
| `digital_pool_sizing.conv_lanes` | cim_timing 6528 | peak rate × conv duty (known 106x) | **VIOL** | M2,M4 |
| `compose_macro_pool` M_REG | cim_timing 2281 | `lanes + conv_lanes` per enumerated slot | **VIOL** | M2 |
| `moe_tokens_hot` | cim_timing 3645 | `ceil(tokens * top_k * α / E)` → 1 | **VIOL** | M6 |
| `_routed_tokens` / `_ffn_block` | fws_build 1202 | all E routed owners fire per beat | **VIOL** | M6 |
| `moe_routed_ffn_time` / `moe_ffn_stage_time` | cim_timing 3655 | closed-form parity path | OK | M1 |
| `_stage_pass` (D29 drive) | fws_build 511 | M = 1 per stage per traversal | OK | M6 |
| `_filled` residency edges | fws_build 466 | stage tail serialisation, `j - streams` | OK | M1 |
| `_weight_stage` granularity | fws_build 659 | one op per (output block, macro) | OK | M1 |
| `_build_bank_passes` measurement | fws_eval 2007 | measured sets vs claimed sets | OK | M5 |
| `bank_passes.basis` \"Fact 1\" sentence | fws_eval 2007 | asserts every bank read once per beat | **VIOL** | M5,M6 |
| `_union_busy_s` | fws_eval 3620 | union of analog busy intervals | OK | M4 |
| `_engine_demand/_fabric_demand.analog_time_s` | fws_eval 3643 | stage analog union on probe slice | OK† | M4 |
| `_analog_beat_of` (probe beat) | fws_eval 3564 | median settled inter-exit interval | OK(named) | M5 |
| `DerivedEngineSizing.analog_beat_s` (name) | cim_timing 2483 | probe beat, published as \"analog beat\" | **VIOL** | M7 |
| `_beat_setting_stage_terms` | fws_eval 3762 | max stage span on probe slice | OK | M1 |
| `binding_term` (priced slice) | fws_eval 3762 | union-busy on priced timeline | OK | M1 |
| `candidate['binding_device_class']` (duty) | dse 1484 | `max(..., key=max_occupancy)` | **VIOL** | M7 |
| `derived_digital['binding_device_class']` | dse 1048 | binding_term's largest term | **VIOL** | M7 |
| `_build_utilization` | fws_eval 1582 | busy_s / makespan, idles included | OK | M4 |
| `build_mapping.macro_slots` | fws_mapping 1956 | `max(macro_usage)` applied to every chip | **VIOL** | M3,M4 |
| `_slots_needed` | dse 714 | highest slot index + 1 | OK | M3 |
| `_derive_chip_capacity` probe bound | dse 758 | `total_arrays()` (array census) | **VIOL** | M8 |
| `_derive_chip_capacity` write-back | dse 764 | busiest chip's need, uniform | **VIOL** | M3,M4 |
| `chip_layer_counts` | fws_mapping 1720 | declared or auto partition | OK | M3 |
| `derive_auto_layers_per_chip` | cim_timing 5510 | packs against probe's array bound | **VIOL** | M8 |
| `validate_capacity` | cim_timing 5600 | usage vs declared arrays_per_chip | OK | M3 |
| `_shard_shapes` / stage shape census | fws_mapping 1422 | layer plan; conv/vectors excluded by name | OK | M1 |
| `_endpoint_shapes` + D30 refusal | fws_mapping 1452 | endpoints refused under REGIME_FILLED | OK | M1 |
| `unowned_macro_slots` / `unowned_columns` | fws_mapping 886 | reported, binds nothing | OK | M5 |
| `macro_footprint_mm2` | cim_timing 5686 | `area_per_array / stack_3d_height` | OK | M8 |
| `stack/total/system_area_mm2` (OPTIMA parity) | cim_timing 4211 | model array census × area | OK(named) | M7 |
| `_silicon` analog term | dse 882 | slots × **footprint**, unowned included | **VIOL** | M8,M3 |
| `candidate['packing']` basis string | dse 1197 | asserts cause, drops the evidence fields | **VIOL** | M5 |
| `engines_per_chiplet` | fws_mapping 2060 | declared seed geometry, prices nothing | OK(named) | M5 |
| `check_allocation_capacity` / `_check_sites` | cim_timing 785 | one column set holds one tile | OK(fact) | M3 |
| `DEAD_FOLDS` / `refuse_dead_fold` | cim_timing 849 | replication refused by name (D27) | OK | M1 |

\\* sound only while the lane law it rests on is sound. † the *reading* is correct; the *scope* of the target is the M4-device violation below.

## 2.2 Per-macro digital pool

| Law | file:line | Sizes against now | Verdict | Rule |
|---|---|---|---|---|
| `digital_pool_sizing.adders` | cim_timing 6515 | `lanes * (n_slices - 1)` = 0 always | **VIOL** | M9 |
| `digital_pool_sizing.area/area_share/disclose` | cim_timing 6535 | `(adders + conv_lanes) * knob` = 0 | **VIOL** | M9,M5 |
| `POOL_DISCLOSURE_AREA_SHARE` | cim_timing 781 | 0.2 gate on a structurally zero numerator | **VIOL** | M5 |
| `n_slices` | cim_timing 5655 | card `bits_per_cell`, absent → 1 | OK(fact) | M3 |
| `compose_macro_pool` FP_ADD row | cim_timing 2278 | `sizing.adders` = 0, row dropped | **VIOL** | M9,M5 |
| `compose_macro_pool` FP_MULT row | cim_timing 2279 | conv lanes, absence named | OK(named) | M5 |
| `macro_pool_composition` | cim_timing 4428 | docstring asserts honest sizing | **VIOL** | M5 |
| `reduction_descriptors` | cim_timing 5978 | derived from placed tiles; empty when no slicing | OK | M2 |
| `price_accumulation` energy | cim_timing 6363 | knob 0, absence named | OK(named) | M5 |
| `price_packing_accumulation` adders | cim_timing 6405 | `min(cells/rows, lanes)` = 21 | **VIOL** | M7 |
| `price_packing_accumulation` area | cim_timing 6431 | demand-based hosts × 0 knob | OK(named) | M3 |
| `report_digital_pool` | cim_timing 6567 | asserts \"the pool constrains nothing\" | **VIOL** | M5 |
| `DigitalPoolSizing.total_lanes` | cim_timing 775 | derived, no consumer | OK(inert) | M7 |
| `FwsPricer._reduction` | fws_eval 489 | per-unit width | OK | M2 |
| `FwsPricer._row_block_descriptor` | fws_eval 549 | operands from placement | OK | M2 |
| `FwsPricer._accumulation` | fws_eval 421 | packer's KStack; spread stacks refused | OK | M7 |
| `FwsPricer._pool_helper` | fws_eval 574 | duration 0, energy 0, no silicon | **VIOL** | M9 |
| `_peak_overlap` | fws_eval 2131 | closed intervals; chains read as concurrency | **VIOL** | M2 |
| `_build_pools` width | fws_eval 3510 | per-unit width × peak overlap | **VIOL** | M2 |
| `_build_pools` conv kwargs | fws_eval 3505 | conv-inclusive here, conv-free in silicon | **VIOL** | M7 |
| `_digital_silicon_block` per_macro_pool | fws_eval 4706 | pool area × **enumerated slots** | **VIOL** | M3 |
| `_digital_silicon_block` digital totals | fws_eval 4760 | sum of three named terms | **VIOL** | M9 |
| `_derive_engine` (the matched pattern) | fws_eval 3864 | two-pass probe, demand-derived | OK | M2 |

## 2.3 Shared digital chiplet — scan / vector engine

| Law | file:line | Sizes against now | Verdict | Rule |
|---|---|---|---|---|
| `vector_cycles_at` | cim_timing 2612 | per-call fill + streaming, summed | OK | M2 |
| `derive_vector_lanes` | cim_timing 2628 | monotone bisection; refuses, never clamps | OK | M2,M4 |
| `derive_engine_sizing` target (ADJ-9) | cim_timing 5117 | the stage's **analog window**, not the beat | **CONFLICT** | M2,M4 |
| `derive_engine_sizing` fallbacks | cim_timing 5118/5126 | probe beat, both kinds named | OK(named) | M5 |
| one card, one width | cim_timing 5262 | max over stages, re-priced, slack printed | OK | M4 |
| ADJ-14 `_group_target` | cim_timing 5171 | max analog window over the group | **CONFLICT** | M4 |
| ADJ-14 `_group_time` | cim_timing 5183 | serial sum over the group's stages | OK | M4 |
| ADJ-14 `_grow` bisection | cim_timing 5198 | smallest sufficient width above ADJ-9 | OK | M2 |
| ADJ-14 saturation guard | cim_timing 5219 | `gain` differences two group identities | **VIOL** | M7 |
| ADJ-14 live-group filter | cim_timing 5232 | computed, used as a boolean only | **VIOL** | M5 |
| ADJ-14 widening verdict | cim_timing 5246 | prose only; never reaches the artifact | **VIOL** | M5 |
| `DerivedEngineSizing.summary` | cim_timing 2527/2540 | literal `\"analog_stage_time\"` | **VIOL** | M5 |
| `price_vector_work` | cim_timing 6789 | `depth + ceil(ops/lanes) - 1` | OK | M2 |
| `EngineDemand` (calls kept separate) | cim_timing 2388 | one entry per priced call | OK | M2 |
| `compose_vector_engine` | cim_timing 2065 | lanes × measured 22 nm blocks | OK | M9 |
| `_engine_demand` op set + target | fws_eval 3643 | vector ops only; attention excluded from target | **VIOL** | M4 |
| `_stage_chiplet_groups` | fws_eval 3285 | same mapping the lowering places with | OK | M7 |
| `device_resources` capacity | fws_eval 289 | capacity 1 per chiplet | OK | M4 |
| `_digital_silicon_block` chiplet term | fws_eval 4698 | `chiplet_area × declared chiplets` | **VIOL** | M3 |
| mapping shared-chiplet resolution | fws_mapping 2050 | declared knob; suggestion \"binds nothing\" | **VIOL** | M3 |
| shared chiplet enumeration | fws_mapping 2079 | `range(shared)` unchecked | **VIOL** | M3 |
| `chiplet_index_of_chip` | fws_mapping 815 | contiguous blocks | OK | M7 |
| `_derive_shared_chiplets` (count) | dse 777/809 | first-fit against the **beat** | **VIOL** | M7 |
| DSE engine point record | dse 1025 | no engine counterpart to fabric saturation | **VIOL** | M5 |

## 2.4 Attention fabric and softmax

| Law | file:line | Sizes against now | Verdict | Rule |
|---|---|---|---|---|
| `sa_cycles` | cim_timing 3244 | measured 32×64 geometry | OK(fact) | M3 |
| `fold_group_split` | cim_timing 2689 | minimises qk + pv (serial ops) | OK | M9 |
| `saturation_arrays` | cim_timing 2772 | `2 × folds`, derivative zero | OK | M2 |
| `saturation_softmax_width` | cim_timing 2777 | softmax **rows** | **VIOL** | M9 |
| `softmax_cycles` | cim_timing 3256/3267 | rows / width; `n` never enters | **VIOL** | M9 |
| `attention_cycles_at` (inline copy) | cim_timing 4568/4595 | same law re-implemented inline | **VIOL** | M7,M9 |
| `attention_call_timing` | cim_timing 3271 | `folds = h_rep × streams`, batch 1 | OK | M6 |
| `decode_attention_timing` | cim_timing 3350 | per-stream context off the annotation | OK | M6 |
| `_stage_attention_cycles` | cim_timing 4605 | qk + pv + softmax + fill, summed | OK | M4 |
| `derive_fabric_sizing` array ladder | cim_timing 4718 | integer GEMMINI copies vs stage target | OK‡ | M2 |
| `derive_fabric_sizing` saturated branch | cim_timing 4714 | both axes jump to saturation | **VIOL** | M4,M5 |
| one card, one fabric | cim_timing 4759 | max over stages, re-priced with slack | OK | M4 |
| lanes-per-replica rounding | cim_timing 4761 | `ceil(width/replicas)`; leftover 0 today | OK | M8 |
| ADJ-14 `_group_attention_time` | cim_timing 4781/4787 | arrays searched, lanes frozen | **VIOL** | M4 |
| `compose_sa_fabric` blocks | cim_timing 2172 | tiles × arrays × replicas, remainder disclosed | OK | M9 |
| `compose_sa_fabric` TRANSPOSER | cim_timing 2237 | 1 per replica; OPTIMA's 9 refused by name | OK(named) | M5 |
| `compose_softmax_engine` census | cim_timing 2121 | OPTIMA row-width unit at width w | OK | M9 |
| `fabric_num_arrays` / `fabric_softmax_lanes` | cim_timing 3155/3179 | falls through to the declared config value | **VIOL** | M3 |
| `shared_digital_compositions` | cim_timing 4450/4458 | fabric + softmax composed unconditionally | **VIOL** | M3 |
| `sa_fabric_composition` provenance stamp | cim_timing 4318 | `derived-count` on a declared count | **VIOL** | M5 |
| `_fabric_demand` | fws_eval 3702 | one row per `attention_qk` cost | OK | M2 |
| `_derive_engine` fabric guard | fws_eval 3961 | no attention → no derivation | OK | M3 |
| `_attention` op pricing | fws_eval 780 | qk/pv/softmax cycles at f_fabric | OK | M9 |
| `fill_drain_penalty_cycles` | config 3146 | `3 × rows`, named a surrogate | OK(named) | M5 |
| `_engine_power_w(\"attention\")` | fws_eval 733 | whole fabric + whole softmax, per op | **VIOL** | M9 |
| `_attention_block` lowering | fws_build 1026 | one qk→softmax→pv chain per layer per beat | OK | M6 |
| `num_arrays`/`softmax_lanes` retired as axes | dse 254 | refused by name (ADJ-10) | OK | M3 |

‡ compliant against its target; the target itself is the ADJ-9 conflict.

## 2.5 State store, KV and movement

| Law | file:line | Sizes against now | Verdict | Rule |
|---|---|---|---|---|
| `compose_state_sram` | cim_timing 2320 | `ceil(bytes / per_macro)` | OK | M9 |
| `sram_macro_bytes` | cim_timing 2308 | measured rows × cols / 8; raises if absent | OK(fact) | M5 |
| `state_sram_composition` | cim_timing 4442 | one stage's bill | OK | M3 |
| `_sized_store` | fws_eval 3234 | composition, or `provenance: absent` | OK(named) | M5 |
| `_build_state_residency` | fws_eval 3355 | `state + kv` → store | **VIOL** | M7,M10 |
| resident stream multiplicity | fws_eval 2923 | `streams = stage count` (D29) | OK | M6 |
| `_resident_contexts` | fws_eval 2850 | contexts reached in the **lowered window** | **VIOL** | M10 |
| `kv_bytes_per_stream_layer` | cim_timing 3806 | `2 × kv_heads/tp × d × ctx × bytes` | OK | M2 |
| KV summation over residents | fws_eval 2983 | sum at each stream's own depth | OK | M6 |
| `_state_residency` ssm branch | fws_eval 2825 | `d_inner × d_state` + conv window | OK | M9 |
| `_state_residency` delta_rule branch | fws_eval 2845 | `key_heads × d_k × value_head_dim` | **VIOL** | M7 |
| `_delta_state_dims` (the correct law) | cim_timing 1471 | `key_head_dim × value_dim` | OK | M7 |
| `_state_residency` short_conv | fws_eval 2847 | falls through to `return 0.0` | **VIOL** | M5 |
| `VectorWork.state_bytes` | cim_timing 1350/1593 | reported traffic per op | OK | M5 |
| `price_vector_work` state term | cim_timing 6836 | knob absent, bounds nothing, disclosed | OK(named) | M5 |
| `_attention` KV bandwidth | fws_eval 832 | no KV term at all | **VIOL** | M5 |
| `decode_s2_timing` | cim_timing 3930 | a KV-bandwidth law, never called | **VIOL** | M7 |
| `kv_story_capacity` | cim_timing 3900 | declared tier | OK | M3 |
| `kv_story_bandwidth` | cim_timing 3870 | defined, never read | **VIOL** | M5 |
| activation high-water | fws_eval 2912 | replayed intervals off the priced timeline | OK | M2 |
| activation tier capacity + its silicon | fws_eval 2997 | declared 8 GB/chip, no area law | **VIOL** | M9 |
| declared on-chip tier (SRAM-L2) | fws_eval 3108 | reported, gates nothing | OK(named) | M5 |
| state store leakage power | fws_eval 4740 | leakage only; dynamic named absent | OK(named) | M5 |
| `_build_silicon` state-store term | fws_eval 4733 | sum of per-stage store areas | OK | M8 |
| `_silicon` state-store term (DSE) | dse 911/918 | one store per stage, gated against evaluator | OK | M7 |
| `max_stage_state_bytes` | dse 481 | refused by name (ADJ-11) | OK | M3 |
| DSE memory gate | dse 1501 | verdicts vs the declared activation tier | **VIOL** | M9 |
| DSE state gate (retired) | dse 1546 | reports only | OK | M5 |
| `_transfer` p2p | fws_eval 958 | bytes / declared link; refuses when bw ≤ 0 | OK | M5 |
| intra-chip activation movement | fws_eval 959 | 0 s, disclosed as unpriced | OK(named) | M5 |
| recurrent-mixer analog↔chiplet hop | fws_build 1124 | no transfer op; named only in a byte census | **VIOL** | M5 |
| attention analog↔chiplet hops | fws_build 1008 | priced by the p2p law | OK | M9 |
| svc link census | fws_eval 1276 | measured bytes; unpriced blocks named | OK(named) | M5 |
| `boundary_table.required_time_s` | fws_mapping 1041 | bytes over the declared pp link | OK | M2 |
| `binding_term` link verdict | fws_eval 4137 | largest union-busy term, class named | OK | M1 |
| `pd_handoff_bytes` | fws_mapping 2513/2529 | hidden + SSM only; LA and conv absent, unnamed | **VIOL** | M5 |
| PD handoff time on the timeline | fws_build 84 | `LAW_PD_LINK` assigned to no op | **VIOL** | M5 |
| `kv_max_streams` | cim_timing 3843 | report only | OK | M5 |

## 2.6 Counts, budgets and the sweep

| Law | file:line | Sizes against now | Verdict | Rule |
|---|---|---|---|---|
| `_derive_shared_chiplets` built duty | dse 827 | duty keyed by **stage**, not by chiplet | **VIOL** | M7,M5 |
| `_silicon` shared-digital term | dse 884 | `per_chiplet × declared chiplets` | **VIOL** | M3 |
| `_silicon` macro-pool term | dse 899 | pool × slots, conv-free, \"composed-measured\" | **VIOL** | M3,M5 |
| `_silicon` uncovered_terms | dse 922 | appends only on a missing library | **VIOL** | M5 |
| `_silicon` totals | dse 975 | duplicates `fws_eval._digital_silicon_block` | **VIOL** | M7 |
| area budget gate (`max_silicon_mm2`) | dse 1408 | a mixed-unit total (footprint + areas) | **VIOL** | M8 |
| chip-count budget gate (`max_chips`) | dse 1219 | declared cap, unbounded on every sweep | OK | M3 |
| REFUSED_AXES `arrays_per_chip` | dse 241 | refused by name | OK | M3 |
| REFUSED_AXES engine widths | dse 262 | refused by name (D31/ADJ-9/ADJ-10) | OK | M3 |
| AXIS_TARGETS `shared_chiplets` | dse 209 | unbounded above the stage count | **VIOL** | M3 |
| AXIS_TARGETS `conv_placement` | dse 214 | one arm priced, one arm not | **VIOL** | M5 |
| AXIS_TARGETS `layers_per_chip` | dse 206 | the real throughput/state trade | OK | M1 |
| AXIS_TARGETS `bank_depth` / `column_sets` | dse 204 | precedence recorded as a note | OK | M7 |
| `pareto_front_ids` | dse 1661 | 2-axis dominance | OK | M4 |
| `_selection_key` | dse 2257 | throughput, then area, then chips | OK | M4 |
| `knee_of` | dse 2104 | max distance above the chord | OK | M4 |
| ladder trim `_trim_accept` | dse 1963 | less silicon and not less throughput (D28) | OK | M4 |
| `emit_selected_config` | dse 2726 | derived capacity written to both homes | OK | M3 |
| `verify_selected` | dse 2766 | timing metrics only; no silicon term | **VIOL** | M5 |
| `shared_chiplet_suggestion` | fws_mapping 2049 | `total_analog` (one per analog chip) | **VIOL** | M3,M7 |

**Totals.** 181 distinct laws: **129 compliant** (including 21 correctly named absences and 11 physical facts), **50 non-compliant**, **2 in methodology conflict** (the ADJ-9/ADJ-14 target).

---

# 3. THE VIOLATIONS, RANKED AND GROUPED

Ranked by how much each distorts the answer the tool exists to give. Grouped so that violations sharing a root cause are fixed in one edit.

---

## V1 — The per-beat work census counts residency, not firing (M6). **CRITICAL**

**Laws:** `program/fws_build.py:1202` (`_routed_tokens` / `_ffn_block`), `cim_timing.py:3645` (`moe_tokens_hot`), `fws_eval.py:2007` (the Fact-1 basis sentence).

- **Sizes:** the analog m-passes charged per beat, the per-beat column-set pass census, and the analog array energy.
- **Now:** every routed expert of every MoE layer fires a full m-pass every beat. `_stage_owners` filters by layer/op/tp only — there is no `top_k` selection anywhere in the lowering. `moe_tokens_hot` then rounds a sub-token load up: `ceil(1 × 6 × 1.0 / 64) = 1`.
- **Should be:** under D29 local batch is 1, so exactly `top_k` routed experts fire per MoE layer per beat. A sub-token load never becomes a full pass.
- **Magnitude (Granite-4.0-H-Tiny):** firings 64 vs 6 = 10.67x. Column-set passes 19,304 charged vs 3,064 honest = 6.30x. Analog array energy 1.259e9 pJ reported vs ~2.0e8 pJ honest — an over-count of 1.059e9 pJ, **84.1 % of the analog energy term and 9.97 % of the run total**.
- **Fix:** lower exactly `top_k` routed owners per MoE layer per beat (a fixed representative set is honest; only the count is a modelling fact). Delete the `ceil()` path from the filled-pipeline lowering. Keep `moe_tokens_hot` for the retired closed-form path only. Then rewrite the `bank_passes` basis sentence: after the fix `column_sets_never_read > 0` on a MoE model, and the honest Invariant-W statement is *weights resident*, not *weights fired*.
- **Moves shipped numbers:** yes — analog energy, the pass census, and the Fact-1 disclosure on every MoE point.

---

## V2 — The state bill and the timing law disagree about the same state (M7), and the store is sized from an analysis knob (M10). **CRITICAL**

Two defects in one function; fix together.

**V2a — delta-rule state is `num_value_heads / num_key_heads` too small.** `fws_eval.py:2845` computes `key_heads × key_head_dim × value_head_dim`; the repo's own `_delta_state_dims` (`cim_timing.py:1471`) says the state is `key_head_dim × value_dim`. Both run in the same evaluation. The conv rolling window is also absent, although ADJ-15 claims it is in the bill.
- **Magnitude (Qwen3.5-4B):** 524,288 B/stream billed vs 1,085,440 B/stream correct = **2.07x under**. 539 SRAM macros missing per LA stage × 24 stages = 12,936 macros = **685.7 mm² (18.7 % of the store, 8.4 % of the machine)**.
- **Fix:** call `_delta_state_dims` in the `delta_rule` branch; add the conv window for the `linear_attention.conv_kernel` and `short_conv` schemas.

**V2b — the store is sized to the context the lowered window reached.** `fws_eval.py:2850` (`_resident_contexts`) derives contexts from `beats = streams + decode_window + 1`. Raising `mapping.decode_window` raises the built SRAM macro count.
- **Magnitude:** Granite +616 macros = 32.65 mm² (+4.8 % of the store). Qwen +7,542 macros = **399.7 mm² (+10.9 % of the store, +4.9 % of the machine)**.
- **Fix:** size the store from the **served** context (`prefill_len + decode_len`). Keep `_resident_contexts` for the priced ops, where the window reading is honest. Metrics may be windowed (ADJ-6); silicon may not (M10).

**V2c — short-conv mixer state returns 0 with no name** (`fws_eval.py:2847`). **V2d — under `cim_dram` the KV is both a declared off-chip tier and composed on-chip macros** (`fws_eval.py:3357`; latent, no shipped config triggers it). Fix both inside the same function.

---

## V3 — The pool's timing law and the pool's silicon census describe different machines (M9). **CRITICAL**

**Laws:** `cim_timing.py:6515` (`adders = lanes × (n_slices - 1)`), `cim_timing.py:2278` (FP_ADD row), `cim_timing.py:6016` / `6330` (the timing laws that assume `lanes` parallel adders), `cim_timing.py:6405` (a second, disagreeing adder law), `cim_timing.py:6535`/`781` (an area law and a disclosure gate with a structurally zero numerator), `fws_eval.py:574` (5,248 pool ops priced at zero duration and zero silicon).

- **Now:** `n_slices` is 1 on every shipped card, so the composed pool contains **zero arithmetic**. The zero-count FP_ADD row is dropped from the census entirely. Meanwhile `drain_cycles = ceil(results/lanes)` and `cycles = depth + ceil(results/lanes) - 1` retire one result per lane per cycle, which requires `lanes` parallel adders, and the timeline runs 39,003,136 adds through those pools.
- **Should be:** `adders ≥ lanes`. 21 is not a guess: at 20 lanes the drain stalls (2.105e-8 s > 2.0e-8 s analog pass), at 21 the stall is exactly 0.0. The repo's own `price_packing_accumulation` already derives 21.
- **Magnitude (Falcon-Mamba):** 75,264 units. Composed today 8.67 mm² / 9.20 W. At the floor the timing law proves: 115.35 mm² / 159.90 W. **13.3x under in area, 17.4x under in power — +7.2 % of all digital area and +21.8 % of all digital power.**
- **Fix:** `adders = lanes` as the floor, with `lanes × (n_slices - 1)` as an additional bit-slice contributor. Never drop a zero-count row. Delete the second adder law at 6405 or make it the only one.
- **Moves shipped numbers:** yes — digital area and power on every point.

---

## V4 — The softmax is sized on rows and built as a row width (M9). **CRITICAL**

**Laws:** `cim_timing.py:3267` and its inline twin at `4595-4602`; `saturation_softmax_width` at `2777`.

- **Now:** `cycles = depth + ceil(tokens_q × heads_chip / width) - 1`. The call's `n` never enters. On Qwen the derivation believes 16 lanes finish the whole call in one pass.
- **Should be:** softmax **elements**: `tokens_q × heads_chip × n` = 16 × 1793 = 28,688 exponentials per call per beat. `compose_softmax_engine` builds one reciprocal *shared by the width* — that is a row-width unit consuming w elements, and the two readings cannot be reconciled.
- **Magnitude:** duty under-counted **1793x**. Charged 20 cycles vs 1,812 honest (90.6x). Stage 17,828 → 19,620 cycles = **+4.7 % on the priced beat**. Composed softmax area 0.106 mm² vs 186.57 mm² at the honest width.
- **Fix:** carry `n` into `AttentionCallDemand` and price elements, in **one** place (delete the inline copy at 4595 — M7).
- **Moves shipped numbers:** yes — the beat, the fabric derivation, and shared-digital area on every attention model.

---

## V5 — A bound is stated in the wrong unit, so valid machines are refused (M8). **CRITICAL**

**Law:** `tools/fws_qif_dse.py:758` — the probe capacity is `total_arrays()`, an **array census**, asserted to be an upper bound on **macro slots**.

- **Magnitude (Qwen3.5-4B):** bound 544, requirement 584 — **7.35 % short**. The probe itself raises `MappingError`. 2 of 32 candidates lost, both at the cheap end. Priced on the current tree the lost single-stage point costs **3,469.95 mm² against the current cheapest front point at 3,583.59 mm² — 113.64 mm² cheaper**, a genuine Pareto corner replaced by a hole and by the report's headline `best_violation`.
- **Fix:** run the probe with `arrays_per_chip = 0` (that path already derives slots from the placement). If a numeric bound is wanted, express it in macro slots.
- **Related, same edit:** `dse:764` applies the busiest chip's requirement uniformly to every chip, and `fws_mapping.py:1956` does the same with `max(macro_usage)` — both enumerate empty slots on the lighter chips (M3/M4). `cim_timing.py:5510` packs `layers_per_chip: 'auto'` against the same invented bound (M8).
- **Moves shipped numbers:** yes — the Qwen frontier gains a cheaper corner.

---

## V6 — Silicon is charged for hardware nothing can reach or use (M3). **MAJOR**

Four laws, one pattern. Fix together.

| | Law | Charged now | Should be |
|---|---|---|---|
| V6a | `fws_eval.py:4698`, `fws_mapping.py:2050`/`2079`, `dse:884`, `dse:209` | `per_chiplet × declared shared_chiplets`; `chiplet_index_of_chip` returns `min(position, count-1)`, so indices above the chip count serve nothing | chiplets that carry stage work, at most one per stage |
| V6b | `cim_timing.py:4458` + `3179` | SA fabric and softmax composed unconditionally; with no attention the width falls through to the **declared** config value | gate on measured attention demand, exactly as the scan engine is gated one line below |
| V6c | `fws_eval.py:4706`, `dse:899` | pool area × **every enumerated macro slot** | slots that hold tiles and sink pool work |
| V6d | `cim_timing.py:4318` | `count_provenance: \"derived-count\"` stamped on a declared count | `declared-count` |

- **Magnitude:** V6a — Qwen at `layers_per_chip=16`, 2 stages: 30 idle chiplets = **1,143.65 mm², 24.2 % of total silicon**, at byte-identical throughput (10,555.390828008 tokens/s at 2, 8 and 32 chiplets). Granite `{4,1,20}` vs `{4,1,10}`: +307.85 mm² and +425.6 W at 0.00 % throughput change. 30 % of valid Qwen candidates sit above their chip count. V6b — 12.35 mm² and 14.84 W per pure-recurrence machine (2.7 % of its shared digital area) for an operator the run never emits.
- **Fix:** clamp or refuse `shared_chiplets` above the stage count at `fws_mapping.py:2050`; gate the attention compositions on demand; charge pool silicon against macros that hold tiles. Sweeping **below** the stage count stays legitimate — ADJ-14's trade is real and measured (per-chiplet area rises 9.79 → 113.53 mm² as the count falls).
- **Moves shipped numbers:** yes — candidate silicon and the frontier's area axis; no *selected* machine changes (no over-count point reaches any shipped front).

---

## V7 — Widths sized against a peak rate the machine never reaches (M2). **MAJOR**

**Laws:** `cim_timing.py:6467` (`macro_result_rate_per_s`), `6514` (`lanes`), `6528` (`conv_lanes` — known), `2281` (M_REG), `fws_eval.py:2131`/`3510` (`_peak_overlap` × per-unit width).

- **Now:** `lanes = ceil(1.92e10 / pool_clock)`. Measured filled-pipeline macro duty is 4.6 % (Falcon), 0.356 % (Granite), 0.145 % (Qwen). The published width is then multiplied by a \"peak concurrent demand\" count that counts **dependency chains** as concurrency: the three intervals behind a peak of 3 are `residual_attn → norm_ffn → residual_ffn`, a strict producer-consumer chain, all at the same instant.
- **Should be:** genuine overlap (`closed=False` returns peak 1 on all 2,848 macros — exactly the per-unit width, which is separately proven minimal-matched at 21 lanes).
- **Magnitude:** 704 of 2,848 macros (24.7 %) publish an inflated width, exported per-macro to the atlas as `act_lanes`. Qwen: 80 of 464 (17.2 %). No priced total moves. The conv arm of the same rate basis is the known 106x / 19 %-of-silicon finding.
- **Nuance worth keeping:** the 21-lane per-unit width is *simultaneously* the peak-rate answer **and** the matched-to-one-fire answer on the CTT card. That coincidence is why the defect hid. State the matched basis explicitly so a different card does not inherit the coincidence.

---

## V8 — Energy charges idle silicon at full synthesis power (M9). **MAJOR**

**Law:** `fws_eval.py:733` — `_engine_power_w(\"attention\")` returns the whole SA fabric plus the whole softmax pipeline, multiplied by each of the three serial ops in turn.

- **Should be:** the blocks active during that op. The derivation already stores `qk_arrays`, `pv_arrays` and `softmax_width` per stage.
- **Magnitude:** charged 2.797e-4 J per stage per beat; active-only 1.381e-4 J. **2.03x over-count**, roughly independent of the split.
- **Extra defect:** this contradicts the neighbouring disclosure, which tells the reader \"an idle engine costs 0 pJ here, which is a floor\". Idle silicon costs zero in one place and full power in another (D21).

---

## V9 — Two engines on one queue are sized against the same budget independently (M4). **MAJOR**

**Laws:** `fws_eval.py:3643` (`_engine_demand`) and `3707` (`_fabric_demand`); `cim_timing.py:4705` and `5117` both set `target_s = analog_s`.

- The lowering places the SSM scan, the short conv and all three attention ops on **the same** chiplet device, whose capacity is 1. Each criterion certifies its own half against the whole analog window. Both can pass while the queue runs at up to 2x the analog pace — and with conv on the chiplet a third term joins with no budget of its own.
- **The repo already knows better:** `dse:806-811` sums `engine_row.time_s + fabric_row.time_s` for its packing law.
- **Fix:** subtract the co-resident engine's measured per-beat time from each budget, or derive jointly. Both times are already measured and already grouped per chiplet.

---

## V10 — Two derivations share one key name, so the tool's answer to M1 is wrong in every table (M7). **MAJOR**

**Laws:** `dse:1484` (duty-cycle version) and `dse:1048` (beat-setting version), both writing `binding_device_class`.

- The frontier prints only the duty-cycle version, under a header that reads \"binding class\". On a weight-stationary machine with hundreds of macros each firing once per beat, `max_occupancy` **can never select `analog_macro`** (Falcon peak 4.6 % vs shared_digital 31.4 %; Granite 0.554 % vs 76.4 %).
- **Magnitude:** outright self-contradiction inside one artifact on the **selected** point of `falcon_mamba_7b_frontier`: the prose says \"the ANALOG FLOOR is the limit\" and the table five lines below says `shared_digital`. 8 of 27 candidates disagree; the top-level field reads `shared_digital` on every valid candidate of all four models.
- **Fix:** rename the duty field `busiest_device_class` and print `derived_digital.binding_device_class` in the Candidates and utilization tables.

---

## V11 — Fallbacks, saturation and absences are reported in prose that never arrives (M5). **MAJOR**

Five laws, one pattern.

| | Law | Defect |
|---|---|---|
| V11a | `cim_timing.py:5246`, `2540` | ADJ-14's widening verdict lives only in a disclosure string that `fws_eval.py:4195` never emits. `grep -c 'ADJ-14'` = **0** in every shipped `dse_report.{json,md}`, while `ADJ-9`/`D31-v2` appear 257/192 times. `summary()` hardcodes `sizing_target = \"analog_stage_time\"`. Falcon c018 provisions **722,944 lanes and 8,194.40 mm²** — more than its entire analog array — against a target the search abandoned, and every machine-readable field reads like a matched machine. |
| V11b | `dse:827` | `built_duty` is keyed by **stage**, not by chiplet. Measured Qwen 16-stage/2-chiplet case: reported mean 0.1237 against a measured 0.7348 (**5.9x understatement**), reported max 0.2464 against 0.9798. The correct bin duty is already in the same dict (`detail['duty']` = [0.9966, 0.9833]) and is discarded. The prose still says \"one per stage\" ten lines below the ADJ-14 comment that says the opposite, producing output such as \"the 4 chiplets this machine builds (one per stage) could carry their work on 7\". |
| V11c | `dse:899`, `922`, `962` | The conv pool's absence is named in `cim_timing` and surfaced by `fws_eval` as `conv_pool_unpriced`, and the DSE reads neither. It publishes `uncovered_terms: []` and `macro_pool_area_provenance: \"composed-measured\"`. Consequence: the two arms of the `conv_placement` axis are ranked against each other on **different accountings** — the shared-digital arm is charged (Qwen 39.740 vs 38.122 mm²/chiplet), the macro-pool arm is charged nothing and named nothing. The axis decides by 91.03 mm² on Falcon. **Do not add the conv-inclusive area to the total** — until the width law is settled that is the 106x over-provision. Report that the axis is not rankable on `total_silicon_mm2`. |
| V11d | `cim_timing.py:6567`, `781`, `4428` | `report_digital_pool` asserts \"the pool constrains nothing (D12)\" while `binding_term` lists `macro_pool` among the beat-setting terms and the beat moves 6.3 % (21→16 lanes) and 5.0 % (21→32). `POOL_DISCLOSURE_AREA_SHARE` is a gate whose numerator is structurally zero. `macro_pool_composition`'s docstring asserts honest sizing and contradicts the comment 13 lines above it. |
| V11e | `dse:2766` | `verify_selected` round-trips timing metrics only. The frontier's **second axis is never verified**. |

---

## V12 — Lesser and latent defects

| | Law | Defect | Magnitude |
|---|---|---|---|
| V12a | `dse:882`, `cim_timing.py:5686`, `dse:1408` | `analog_macro_silicon_mm2` is a **footprint** (`area / stack_3d_height`) summed with three true areas into `total_silicon_mm2`, the one budget that refuses machines. Three separate docstrings forbid exactly this. | **Zero today** (every card leaves `stack_3d_height = 1`). At h = 2 the Granite total falls 10,783.9 → ~5,894.5 mm² with no silicon removed. Latent unit error in the ADJ-11 gate. |
| V12b | `cim_timing.py:5219` | The ADJ-14 saturation guard differences `_group_time` of a group **identity that can change between iterations**, so a false \"saturated\" stop is possible. Counterexample constructed; no shipped point shown to take it. Fix: measure progress on `_worst_over`, which is provably monotone. Same edit in the fabric copy at 4797. | Not computable. |
| V12c | `dse:975` vs `fws_eval.py:4758` | The whole silicon composition exists **twice**, in two files, with identical formulas. They agree to 0.0 on the current tree — they did not in the shipped artifacts (Qwen c007 delta 405.99 mm², exactly the conv term). | D21 duplication; a live divergence risk. |
| V12d | `fws_mapping.py:2049` vs `dse:777` | Two \"suggested chiplet count\" laws on one candidate row: `total_analog` (one per analog chip) and the beat-packed first-fit. Granite c000: **10 vs 4**. Qwen c014: 32 vs 7. The chip-count heuristic is also the binding **default** on any run without `mapping.shared_chiplets`. | 2.5x–4.6x disagreement, printed side by side with no basis string. |
| V12e | `fws_build.py:1124` | The recurrent-mixer analog↔chiplet hop takes a cross-chip dependency with **no transfer op**. Named only in a byte census, never in the time accounting. | The beat is optimistic by that movement, unquantified. |
| V12f | `fws_mapping.py:2529`, `fws_build.py:84` | The PD handoff omits linear-attention and conv state entirely and does not name the omission; `LAW_PD_LINK` is assigned to no op, so the handoff time never reaches the timeline and nothing says so. | Qwen: 25.2 MB fast-weight + 0.88 MB conv per request, 0.26 ms at the declared link. Report-only path. |
| V12g | `fws_eval.py:2997`, `dse:1501` | The activation tier is a declared 8 GB/chip that **gates feasibility and has no area law**, while the state store beside it is composed from measured macros. | Inconsistent accounting between two stores in one report. |
| V12h | `cim_timing.py:3930`, `3870` | `decode_s2_timing` (a KV-bandwidth law) and `kv_story_bandwidth` exist and are never called; `_attention` carries no KV bandwidth term and does not name its absence. | Dead second laws. |

---

# 4. THE SYSTEMIC ROOT CAUSES

Three patterns produced 46 of the 50 violations. They are worth more than the individual fixes.

## RC1 — Residency is silently substituted for activity.

The model knows, everywhere, what is **present**: enumerated macro slots, resident weights, declared chiplets, a card's peak result rate, a model's array census. It rarely computes what is **active in one beat**. Whenever the active quantity was hard to reach, a resident quantity was used instead, and the substitution was never labelled.

Produced: V1 (all 64 experts fire because all 64 are resident), V6a/V6b/V6c (chiplets, fabrics and pools charged because they are enumerated), V7 (lanes sized to a peak rate because that rate is a card property and the duty is not), V5 (an array census used as a slot bound because the census was already computed), V11b (duty keyed by stage because stages are the thing already enumerated).

**Why it recurs:** there is no single object in the codebase that means \"the work of one beat\". Demand is re-derived ad hoc in `_engine_demand`, `_fabric_demand`, `_build_pools`, `_derive_shared_chiplets` and the lowering, each with its own filter.

**The structural fix:** build one `BeatWorkCensus` on the measurement slice — per device, per op class, per macro: firings, ops, bytes, busy seconds. Every sizing law, every silicon census and every charge reads that one object. Any law that reads a *card constant* or a *config count* where a census field exists is then a visible, greppable violation. M2, M3 and M6 all become testable against a single artifact.

## RC2 — One quantity is implemented in two places, and the two drift.

The repo repeatedly has the right law already. `_delta_state_dims` is correct and unused by the residency. `price_packing_accumulation` derives 21 adders while the composition ships 0. `_derive_shared_chiplets` sums both engines' time while the two sizing laws do not. `attention_cycles_at` re-implements `softmax_cycles` inline. The DSE recomputes the whole silicon block. Two suggestions and two binding classes ride one candidate record.

Produced: V2a, V3, V4 (the inline twin), V9, V10, V12c, V12d, V12h — and the *unit* drift in V12a, where a divisor was applied in one place and forgotten in the sum.

**Why it recurs:** the derivation, the pricer, the composer and the sweep each grew their own copy of a number so that each could be read independently. D21 exists and is stated, but nothing enforces it, and no field carries a unit.

**The structural fix:** make it mechanical. (a) Every published number carries `{value, unit, provenance, law_site}`. A sum asserts a single unit. (b) A derived width is passed to the composer **as the sizing object**, never as a loose integer, so a composition cannot be built from a different width than the one priced. (c) A CI check that greps for the same key name written at two sites. (d) Delete the second implementation rather than reconciling it — every one of the pairs above has an obvious winner.

## RC3 — Absence, fallback and approximation are communicated in prose, and the prose does not arrive.

The project's disclosure culture is real and unusually good — but it lives in docstrings, basis strings and `disclosures` tuples that are assembled by one module and never emitted by the next. `ADJ-14` appears **zero times** in every shipped report. `uncovered_terms` is `[]` while the conv is unpriced. A zero-count FP_ADD row vanishes from the census instead of printing as zero. A fallback target keeps the literal `\"analog_stage_time\"`. A gate is written whose numerator can never exceed it. Worse, several strings assert the **opposite** of what the code does — \"every stored weight is read once per beat\", \"the pool constrains nothing\", \"an idle engine costs 0 pJ\", \"one chiplet per stage\" — so a reader who checks the disclosure is actively misled.

Produced: V11a–V11e, V12f, V12g, and the reason V1, V3 and V6b went unnoticed for so long.

**Why it recurs:** disclosures are strings, not data. Nothing fails when one is dropped, and nothing fails when one goes stale.

**The structural fix:** disclosures become typed records with a key, a status and a value, carried through every layer that touches the object they describe (mapping → eval → DSE → report), and asserted present in the shipped artifact by the round-trip verifier. A basis string is then *generated from* the record, never hand-written beside it. Add M5's provenance requirement to the schema so a number with no provenance cannot be published at all.

**A fourth pattern, smaller but load-bearing:** ADJ-9's \"size to the analog window\" rationale — *this silicon is cheap, so build more of it* — is a safety margin wearing an engineering argument. D28 forbids margins. It is the source of the 39,033- and 722,944-lane engines. It needs the owner's explicit ruling (see §1).

---

# 5. THE ORDERED WORK PLAN

Highest distortion first. **[SHIPPED]** marks a fix that moves numbers in the checked-in reports and therefore needs the artifacts regenerated.

### Phase 0 — the ruling (blocks Phase 3)
0. **Decide ADJ-9's target.** Beat-minus-co-resident-occupancy at a fixed point (recommended), or keep the analog-window target as an explicit, disclosed over-provision with its ratio published per stage. Nothing in Phase 3 can be sized until this is settled. *No code.*

### Phase 1 — the census and the state bill (largest distortions, isolated edits)
1. **V1 — MoE fires `top_k`, not `E`.** `fws_build.py:1202`, `cim_timing.py:3645`, rewrite the `bank_passes` basis. **[SHIPPED: analog energy −84 %, pass census 6.3x, Fact-1 text]**
2. **V2a — delta-rule state uses `_delta_state_dims`; add the conv window.** `fws_eval.py:2845`. **[SHIPPED: +685.7 mm² on Qwen, +8.4 % machine]**
3. **V2b — store sized from the served context, not the lowered window.** `fws_eval.py:2850`/`3355`. **[SHIPPED: +399.7 mm² Qwen, +32.7 mm² Granite]**
4. **V5 — probe with `arrays_per_chip = 0`; state bounds in macro slots.** `dse:758`, then `dse:764` and `fws_mapping.py:1956` per-chip rather than uniform. **[SHIPPED: Qwen frontier gains a 3,469.95 mm² corner]**
5. **V2c/V2d — short-conv state law; KV excluded from the SRAM store under `cim_dram`.** *Latent; no shipped number.*

### Phase 2 — make time and silicon agree
6. **V3 — `adders = lanes`; never drop a zero-count row; delete the duplicate adder law.** `cim_timing.py:6515`, `2278`, `6405`. **[SHIPPED: +7.2 % digital area, +21.8 % digital power]**
7. **V4 — softmax priced in elements, in one place.** `cim_timing.py:3267`; delete the inline copy at `4595`; fix `saturation_softmax_width`. **[SHIPPED: beat +4.7 %, fabric derivation, shared-digital area]**
8. **V8 — attention energy charged to the active array group.** `fws_eval.py:733`; reconcile or delete the idle-leakage sentence. **[SHIPPED: attention energy ÷2]**
9. **V12a — the ADJ-11 budget compares silicon with silicon.** `dse:882`; keep the footprint under its own name. *Zero today, latent.*
10. **V12g/V12h — give the activation tier an area law or stop gating on it; delete the dead KV-bandwidth laws or call them.**

### Phase 3 — counts follow from work (needs Phase 0)
11. **V6a — refuse or clamp `shared_chiplets` above the stage count; charge only chiplets that serve a stage; fix the \"one per stage\" note.** `fws_mapping.py:2050`/`2079`, `fws_eval.py:4698`, `dse:884`/`209`. **[SHIPPED: candidate silicon; up to 24 % on swept points; no selected machine changes]**
12. **V6b — gate the SA fabric and softmax compositions on measured attention demand; emit the twin absence sentence; fix the provenance stamp.** `cim_timing.py:4458`, `4318`. **[SHIPPED: −12.35 mm² / −14.84 W on pure-recurrence machines]**
13. **V6c/V7 — pool width from measured per-beat work, charged on macros that hold tiles; `_peak_overlap` half-open.** `cim_timing.py:6514`, `fws_eval.py:2131`/`3510`/`4706`. **[SHIPPED: atlas `act_lanes` on 24.7 % of macros; pool silicon basis]**
14. **V9 — one budget per device: subtract co-resident occupancy before deriving each engine.** `fws_eval.py:3643`/`3707`, `cim_timing.py:4705`/`5117`. **[SHIPPED: engine and fabric widths]**
15. **V12b — ADJ-14 progress measured on `_worst_over`.** `cim_timing.py:5219` and `4797`.

### Phase 4 — the disclosure spine (RC3), then the reporting fixes
16. **Typed disclosure records carried mapping → eval → DSE → artifact; provenance on every published number; a unit on every summed term.** This is the structural fix and it makes 17–21 one-liners.
17. **V11a — `shared_widened` on `DerivedEngineSizing`; emit `sizing.disclosures`; add the engine counterpart to `fabric_saturated_stages`.**
18. **V11b — bin duty by chiplet (or publish the bins' own duty); delete the \"one per stage\" prose.**
19. **V11c — carry `conv_pool_unpriced` into `uncovered_terms`; stop claiming `composed-measured`; state that `conv_placement` is not rankable on total silicon.**
20. **V10 — rename the duty field `busiest_device_class`; print the beat-setting one in both tables.**
21. **V11d/V11e/V12d — retire the false \"constrains nothing\" claims; verify silicon in the round trip; retire the chip-count chiplet suggestion as the mapper default.**
22. **V12c — one silicon composition, called from both places.**
23. **V12e/V12f — price the recurrent-mixer hop or name it in the time accounting; complete the PD handoff state terms or name their absence; state that the handoff is report-only.**

**Regression gate to add with Phase 1:** the `BeatWorkCensus` from RC1 becomes a test fixture. Every sizing law is asserted to read from it, and `work_per_beat / capacity_per_beat` is printed per derived width in the report. That single number makes M2 and M4 auditable at a glance and would have caught V1, V3, V4, V7 and V9 before they shipped.

---

# 6. WHERE THIS AUDIT IS THIN — READ THE COMPLIANCE TABLE WITH THESE CAVEATS

I am confident in the 21 verified findings; each carries a re-derivation or a live run. I am **less** confident that the inventory is complete. The following areas were not covered by any lane, and their entries in the table above are therefore *unchallenged*, not *cleared*:

1. **The card's own declared constants were never audited.** `energy_per_vec_pj`, `area_mm2_per_array`, `analog_clock_mhz`, `slice_cycles`, `shots_per_output` are the multiplicands under every analog number in the tool. No lane asked where they came from or whether they are measured. M5 has not been applied to the card.
2. **Energy is thinly covered.** Only two energy laws were examined in depth (analog array energy, fabric power). Pool energy rides a knob that is 0 on every card; the state store charges leakage only; link energy, ADC energy detail and DAC/driver energy were not inventoried at all. The reported total energy is likely governed by absences nobody has counted.
3. **The interconnect is barely modelled and barely audited.** Only the p2p transfer law is priced. Intra-chip movement is declared 0. One cross-chip hop takes a dependency with no transfer. No lane audited the network, and `binding_term` can name `link` — so the term that may bind a future machine is the least examined one.
4. **The prefill path and the PD pair are report-only and lightly checked.** `LAW_PD_LINK` reaches no op. Everything in §2.5 about the handoff is untested against a run.
5. **The 22 nm synthesis library was taken as given.** The `M_REG` block that today constitutes the *entire* composed pool carries the comment `\"Register for MAMBA BF16. TODO!!!!!\"`. Fixing V3 makes FP_ADD dominate that census, which raises the stakes on library provenance considerably.
6. **The zero-duration op class was not audited.** 5,248 norm / activation / residual ops are priced at exactly 0 s and 0 pJ by declaration (D12). That declaration rests on the pool sizing that V3 and V7 both show is wrong. It should be re-examined after Phase 2, not before.
7. **Model coverage is four points.** V1 rests on Granite, V2a and V4 on Qwen, V3 and V7 on Falcon-Mamba. A pure-transformer point (D5 says these are IN and will \"lean on heavy digital co-compute\") is not in the shipped set, and that is precisely the shape most likely to expose the M4-device violation V9.
8. **The methodology conflict in §1 is my reading, not a verified defect.** ADJ-9 is an explicit owner decision. I have argued it conflicts with M2, M4 and D28 and shown the widths it produces. Whether that is a defect or the owner's intent is the owner's call, and I have not treated it as settled.",
    "laws": 206,
    "findings": 21,
    "refuted": 20
  },
  "workflowProgress": [
    {
      "type": "workflow_phase",
      "index": 1,
      "title": "Audit"
    },
    {
      "type": "workflow_phase",
      "index": 2,
      "title": "Verify"
    },
    {
      "type": "workflow_phase",
      "index": 3,
      "title": "Methodology"
    },
    {
      "type": "workflow_agent",
      "index": 1,
      "label": "audit:analog-beat",
      "phaseIndex": 1,
      "phaseTitle": "Audit",
      "agentId": "a76cd8338a720a158",
      "model": "claude-opus-5[1m]",
      "state": "done",
      "startedAt": 1787799788618,
      "queuedAt": 1787799786283,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "You are auditing a research performance model for a chiplet system: many analog
compute-in-memory (CIM) weight-stationary macro chiplets, plus shared digital
chiplets, running transformer / state-space / hybrid inference in DECODE.

REPO ROOT: /app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM
Work ONLY inside that directory. This is a shared NFS mount: never run an
unbound…",
      "lastProgressAt": 1787800697541,
      "tokens": 214519,
      "toolCalls": 89,
      "durationMs": 908823,
      "resultPreview": "{\"findings\":[{\"title\":\"Every routed MoE expert fires a full analog m-pass every beat; the filled pipeline routes top_k of E\",\"file\":\"program/fws_build.py\",\"line\":1202,\"methodology_rule\":\"M2\",\"what_it_sizes\":\"the number of analog m-passes (and the analog array energy, per-macro pool demand and expert dispatch/combine traffic) charged per beat for a MoE layer's routed experts\",\"rate_basis_now\":\"stag…"
    },
    {
      "type": "workflow_agent",
      "index": 2,
      "label": "audit:macro-pool",
      "phaseIndex": 1,
      "phaseTitle": "Audit",
      "agentId": "af0570814fe4e3fc3",
      "model": "claude-opus-5[1m]",
      "state": "done",
      "startedAt": 1787799788313,
      "queuedAt": 1787799786284,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "You are auditing a research performance model for a chiplet system: many analog
compute-in-memory (CIM) weight-stationary macro chiplets, plus shared digital
chiplets, running transformer / state-space / hybrid inference in DECODE.

REPO ROOT: /app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM
Work ONLY inside that directory. This is a shared NFS mount: never run an
unbound…",
      "lastProgressAt": 1787801557704,
      "tokens": 133765,
      "toolCalls": 83,
      "durationMs": 1768986,
      "resultPreview": "{\"findings\":[{\"title\":\"The composed per-macro pool contains ZERO arithmetic: adders = lanes*(n_slices-1) is structurally 0 on every shipped card, while the pool's own timing law assumes `lanes` parallel adders\",\"file\":\"cim_timing.py\",\"line\":2278,\"methodology_rule\":\"M5\",\"what_it_sizes\":\"the measured FP_ADD blocks in one analog macro's digital pool (the silicon that actually performs the shift-add t…"
    },
    {
      "type": "workflow_agent",
      "index": 3,
      "label": "audit:shared-engine",
      "phaseIndex": 1,
      "phaseTitle": "Audit",
      "agentId": "adfe280e2085eb246",
      "model": "claude-opus-5[1m]",
      "state": "done",
      "startedAt": 1787799789169,
      "queuedAt": 1787799786284,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "You are auditing a research performance model for a chiplet system: many analog
compute-in-memory (CIM) weight-stationary macro chiplets, plus shared digital
chiplets, running transformer / state-space / hybrid inference in DECODE.

REPO ROOT: /app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM
Work ONLY inside that directory. This is a shared NFS mount: never run an
unbound…",
      "lastProgressAt": 1787800761438,
      "tokens": 152695,
      "toolCalls": 68,
      "durationMs": 972269,
      "resultPreview": "{\"findings\":[{\"title\":\"The scan engine is sized against the stage's analog m-pass window, not against the beat the filled pipeline imposes (ADJ-9's target)\",\"file\":\"/app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM/cim_timing.py\",\"line\":5117,\"methodology_rule\":\"M2\",\"what_it_sizes\":\"vector_lanes - the scan/vector lane count of every shared digital chiplet\",\"rate_basis_now\":…"
    },
    {
      "type": "workflow_agent",
      "index": 4,
      "label": "audit:fabric-sa",
      "phaseIndex": 1,
      "phaseTitle": "Audit",
      "agentId": "aa9e68670ea766df5",
      "model": "claude-opus-5[1m]",
      "state": "done",
      "startedAt": 1787799787816,
      "queuedAt": 1787799786284,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "You are auditing a research performance model for a chiplet system: many analog
compute-in-memory (CIM) weight-stationary macro chiplets, plus shared digital
chiplets, running transformer / state-space / hybrid inference in DECODE.

REPO ROOT: /app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM
Work ONLY inside that directory. This is a shared NFS mount: never run an
unbound…",
      "lastProgressAt": 1787800650716,
      "tokens": 143846,
      "toolCalls": 64,
      "durationMs": 862899,
      "resultPreview": "{\"findings\":[{\"title\":\"Softmax pipeline is sized against the softmax ROW COUNT while its composed width is a row WIDTH: the decode duty is under-counted by exactly the context length (1793x on the shipped Qwen point)\",\"file\":\"cim_timing.py\",\"line\":3267,\"methodology_rule\":\"M2\",\"what_it_sizes\":\"cim.fabric.softmax_lanes - the width of the softmax pipeline composed on every shared digital chiplet (FP_…"
    },
    {
      "type": "workflow_agent",
      "index": 5,
      "label": "audit:memory-store",
      "phaseIndex": 1,
      "phaseTitle": "Audit",
      "agentId": "ab3d62522b43291a4",
      "model": "claude-opus-5[1m]",
      "state": "done",
      "startedAt": 1787799789298,
      "queuedAt": 1787799786284,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "You are auditing a research performance model for a chiplet system: many analog
compute-in-memory (CIM) weight-stationary macro chiplets, plus shared digital
chiplets, running transformer / state-space / hybrid inference in DECODE.

REPO ROOT: /app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM
Work ONLY inside that directory. This is a shared NFS mount: never run an
unbound…",
      "lastProgressAt": 1787800640843,
      "tokens": 174190,
      "toolCalls": 77,
      "durationMs": 851544,
      "resultPreview": "{\"findings\":[{\"title\":\"The recurrent mixer's analog<->chiplet chip crossing is not on the timeline at all: every scan/delta layer's operands and results cross a chip boundary for zero time, while the identical attention hop is priced\",\"file\":\"program/fws_build.py\",\"line\":1124,\"methodology_rule\":\"M1\",\"what_it_sizes\":\"the beat itself - the wall time a pipeline stage spends per token\",\"rate_basis_now…"
    },
    {
      "type": "workflow_agent",
      "index": 6,
      "label": "audit:counts-dse",
      "phaseIndex": 1,
      "phaseTitle": "Audit",
      "agentId": "a0a3b54aadca3dfe3",
      "model": "claude-opus-5[1m]",
      "state": "done",
      "startedAt": 1787799788127,
      "queuedAt": 1787799786284,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "You are auditing a research performance model for a chiplet system: many analog
compute-in-memory (CIM) weight-stationary macro chiplets, plus shared digital
chiplets, running transformer / state-space / hybrid inference in DECODE.

REPO ROOT: /app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM
Work ONLY inside that directory. This is a shared NFS mount: never run an
unbound…",
      "lastProgressAt": 1787800823652,
      "tokens": 204072,
      "toolCalls": 64,
      "durationMs": 1034935,
      "resultPreview": "{\"findings\":[{\"title\":\"The chip-capacity probe bound is not a bound: total_arrays() is below what the placement needs, so the coarsest partition is refused as infeasible\",\"file\":\"tools/fws_qif_dse.py\",\"line\":758,\"methodology_rule\":\"M3\",\"what_it_sizes\":\"cim.chip.arrays_per_chip — the macro slot count a chip offers (via the probe placement that measures the requirement)\",\"rate_basis_now\":\"CimDeviceM…"
    },
    {
      "type": "workflow_agent",
      "index": 7,
      "label": "verify:memory-store",
      "phaseIndex": 2,
      "phaseTitle": "Verify",
      "agentId": "a5a0d76a37c748b26",
      "model": "claude-opus-5[1m]",
      "state": "done",
      "startedAt": 1787800644355,
      "queuedAt": 1787800642856,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "You are auditing a research performance model for a chiplet system: many analog
compute-in-memory (CIM) weight-stationary macro chiplets, plus shared digital
chiplets, running transformer / state-space / hybrid inference in DECODE.

REPO ROOT: /app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM
Work ONLY inside that directory. This is a shared NFS mount: never run an
unbound…",
      "lastProgressAt": 1787801236399,
      "tokens": 121951,
      "toolCalls": 65,
      "durationMs": 592044,
      "resultPreview": "{\"findings\":[{\"title\":\"The delta-rule (linear attention) resident state in the store's bill is exactly num_value_heads/num_key_heads too small (2x on Qwen3.5) and omits the conv rolling window, contradicting the repo's own state law\",\"file\":\"fws_eval.py\",\"line\":2845,\"methodology_rule\":\"M4\",\"what_it_sizes\":\"the per-stage byte bill that ADJ-11 builds the on-chip SRAM store from, hence `silicon.state…"
    },
    {
      "type": "workflow_agent",
      "index": 8,
      "label": "verify:fabric-sa",
      "phaseIndex": 2,
      "phaseTitle": "Verify",
      "agentId": "a78c29be810ffe3ee",
      "model": "claude-opus-5[1m]",
      "state": "done",
      "startedAt": 1787800653937,
      "queuedAt": 1787800652411,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "You are auditing a research performance model for a chiplet system: many analog
compute-in-memory (CIM) weight-stationary macro chiplets, plus shared digital
chiplets, running transformer / state-space / hybrid inference in DECODE.

REPO ROOT: /app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM
Work ONLY inside that directory. This is a shared NFS mount: never run an
unbound…",
      "lastProgressAt": 1787801258853,
      "tokens": 110352,
      "toolCalls": 51,
      "durationMs": 604916,
      "resultPreview": "{\"findings\":[{\"title\":\"Softmax pipeline is sized against the softmax ROW COUNT while its composed width is a row WIDTH: the decode duty is under-counted by exactly the context length (1793x on the shipped Qwen point)\",\"file\":\"cim_timing.py\",\"line\":3267,\"methodology_rule\":\"M2\",\"what_it_sizes\":\"cim.fabric.softmax_lanes - the width of the softmax pipeline composed on every shared digital chiplet (FP_…"
    },
    {
      "type": "workflow_agent",
      "index": 9,
      "label": "verify:analog-beat",
      "phaseIndex": 2,
      "phaseTitle": "Verify",
      "agentId": "a63c3bbaf43be62d7",
      "model": "claude-opus-5[1m]",
      "state": "done",
      "startedAt": 1787800700990,
      "queuedAt": 1787800699586,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "You are auditing a research performance model for a chiplet system: many analog
compute-in-memory (CIM) weight-stationary macro chiplets, plus shared digital
chiplets, running transformer / state-space / hybrid inference in DECODE.

REPO ROOT: /app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM
Work ONLY inside that directory. This is a shared NFS mount: never run an
unbound…",
      "lastProgressAt": 1787801108950,
      "tokens": 90516,
      "toolCalls": 43,
      "durationMs": 407959,
      "resultPreview": "{\"findings\":[{\"title\":\"Every routed MoE expert fires a full analog m-pass every beat; the filled pipeline routes top_k of E\",\"file\":\"program/fws_build.py\",\"line\":1202,\"methodology_rule\":\"M2\",\"what_it_sizes\":\"the number of analog m-passes (and hence the analog array energy and the per-beat column-set pass census) charged per beat for a MoE layer's routed experts\",\"rate_basis_now\":\"stage_tokens = se…"
    },
    {
      "type": "workflow_agent",
      "index": 10,
      "label": "verify:shared-engine",
      "phaseIndex": 2,
      "phaseTitle": "Verify",
      "agentId": "a91be81f8f25cf3aa",
      "model": "claude-opus-5[1m]",
      "state": "done",
      "startedAt": 1787800764510,
      "queuedAt": 1787800763039,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "You are auditing a research performance model for a chiplet system: many analog
compute-in-memory (CIM) weight-stationary macro chiplets, plus shared digital
chiplets, running transformer / state-space / hybrid inference in DECODE.

REPO ROOT: /app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM
Work ONLY inside that directory. This is a shared NFS mount: never run an
unbound…",
      "lastProgressAt": 1787801597709,
      "tokens": 123463,
      "toolCalls": 44,
      "durationMs": 833199,
      "resultPreview": "{\"findings\":[{\"title\":\"Shared digital chiplets are built and charged the full derived engine width even when no chip can ever reach them\",\"file\":\"/app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM/fws_eval.py\",\"line\":4698,\"methodology_rule\":\"M3\",\"what_it_sizes\":\"the number of shared digital chiplets whose engine silicon and power are charged to the machine\",\"rate_basis_now\"…"
    },
    {
      "type": "workflow_agent",
      "index": 11,
      "label": "verify:counts-dse",
      "phaseIndex": 2,
      "phaseTitle": "Verify",
      "agentId": "a042ac4447a438450",
      "model": "claude-opus-5[1m]",
      "state": "done",
      "startedAt": 1787800827951,
      "queuedAt": 1787800826667,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "You are auditing a research performance model for a chiplet system: many analog
compute-in-memory (CIM) weight-stationary macro chiplets, plus shared digital
chiplets, running transformer / state-space / hybrid inference in DECODE.

REPO ROOT: /app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM
Work ONLY inside that directory. This is a shared NFS mount: never run an
unbound…",
      "lastProgressAt": 1787801608970,
      "tokens": 128893,
      "toolCalls": 53,
      "durationMs": 781018,
      "resultPreview": "{\"findings\":[{\"title\":\"The chip-capacity probe bound is not a bound: total_arrays() is below what the placement needs, so the coarsest partition is refused as infeasible\",\"file\":\"tools/fws_qif_dse.py\",\"line\":758,\"methodology_rule\":\"M3\",\"what_it_sizes\":\"cim.chip.arrays_per_chip — the macro slot count a chip offers (via the probe placement that measures the requirement)\",\"rate_basis_now\":\"CimDeviceM…"
    },
    {
      "type": "workflow_agent",
      "index": 12,
      "label": "verify:macro-pool",
      "phaseIndex": 2,
      "phaseTitle": "Verify",
      "agentId": "a0f25cb53ee66fe93",
      "model": "claude-opus-5[1m]",
      "state": "done",
      "startedAt": 1787801561541,
      "queuedAt": 1787801560223,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "promptPreview": "You are auditing a research performance model for a chiplet system: many analog
compute-in-memory (CIM) weight-stationary macro chiplets, plus shared digital
chiplets, running transformer / state-space / hybrid inference in DECODE.

REPO ROOT: /app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM
Work ONLY inside that directory. This is a shared NFS mount: never run an
unbound…",
      "lastProgressAt": 1787802130259,
      "tokens": 89847,
      "toolCalls": 44,
      "durationMs": 568717,
      "resultPreview": "{\"findings\":[{\"title\":\"The composed per-macro pool contains ZERO arithmetic: adders = lanes*(n_slices-1) is structurally 0 on every shipped card, while the pool's own timing law drains 384 results per fire through `lanes` parallel adders\",\"file\":\"cim_timing.py\",\"line\":2278,\"methodology_rule\":\"M5\",\"what_it_sizes\":\"the measured FP_ADD blocks in one analog macro's digital pool (the silicon that perfo…"
    },
    {
      "type": "workflow_agent",
      "index": 13,
      "label": "methodology",
      "phaseIndex": 3,
      "phaseTitle": "Methodology",
      "agentId": "a9ed34cee2bfef423",
      "model": "claude-opus-5[1m]",
      "state": "done",
      "startedAt": 1787802134790,
      "queuedAt": 1787802131981,
      "attempt": 1,
      "lastToolName": "Bash",
      "lastToolSummary": "cd /app/nanocad/projects/ispass_deepflow/deepflow_astra_dev…",
      "promptPreview": "You are auditing a research performance model for a chiplet system: many analog
compute-in-memory (CIM) weight-stationary macro chiplets, plus shared digital
chiplets, running transformer / state-space / hybrid inference in DECODE.

REPO ROOT: /app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM
Work ONLY inside that directory. This is a shared NFS mount: never run an
unbound…",
      "lastProgressAt": 1787802683665,
      "tokens": 106279,
      "toolCalls": 3,
      "durationMs": 548874,
      "resultPreview": "# THE QIF SIZING METHODOLOGY, THE COMPLIANCE TABLE, AND THE VIOLATIONS

Synthesis of six audit lanes (206 laws) and 21 adversarially verified findings.
Repo: `/app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM`

---

# 1. THE METHODOLOGY

*Written to be pasted into `docs/qif/DECISIONS.md` as the governing decision. It sharpens the owner's M1–M5 and adds five rules the inven…"
    }
  ],
  "totalTokens": 1794388,
  "totalToolCalls": 748
}