Verification complete. Findings below.

## Fact-check results

Verified against `/app/nanocad/projects/ispass_deepflow/deepflow_astra_dev_rewrite/Rapid-LLM` (CONTEXT.md, `simulate_train_graph.py`, `llm_execution.py`, `astrasim_lib/executor.py`, `train_timing.py`, `memory_estimation.py`, `equiv/configs.py`).

**Claims that check out (all designs):** per-stage Kahn keyed by `op_id` with `itertools.count` tie sequence (executor.py:1301-1334); p2p tags = `edge.op_id` with fallback counter from 1e6 (executor.py:1408-1422); gid base 1000 with **label-sorted, token-sorted** allocation (executor.py:1571-1590); dp stage groups `str(stage_idx+1)` (executor.py:1683-1685); singleton-group zero-duration `_noop` COMP (executor.py:1649-1668); `dp_count<=1` skip of non-tp collectives (executor.py:1601); control = `size==0 OR comm_type != PIPELINE` → 1 byte + `_send_control`/`_recv_control` + name-suffix-based renumber (executor.py:1441-1445, 368-430); `int(round(sec*1e6))` (executor.py:1706); `duration_profile` length==dp_count check (executor.py:803-815); ZeRO-3 `"bwd" in name` ±par_degree hack and `int(base_bytes)` un-divided (llm_execution.py:457-495); DP collectives attached to `rank_tails[0]` only (llm_execution.py:806); pipeline per-rank `ceil(total/par_degree)` split + compute-anchor double-parent (llm_execution.py:824-877); flattener op_id counter first value 1 (llm_execution.py:971-973); `param_gather = (gemm_idx==0)` (llm_execution.py:745); `local_comp_time` forced to 0 (simulate_train_graph.py:284-289); `_should_emit_dp_comm` incl. `b==0` backward rule (simulate_train_graph.py:239-250); `Graph.simulate` heap `(finish, counter)` + FIFO ready-scan + `GPU_list` node-only exclusivity (simulate_train_graph.py:1680-1793); duplicated layout code in `memory_estimation.py:289,341`; `diff_bundles`, `GMapCollector.record_collective/record_pipeline`, `FaultSpace`, `axis_layout_from_descriptor`, `memory_estimator_smoke.py` all exist.

## Concrete factual errors

**Design A**
1. **Dead code presented as live plumbing.** `extract_forward_graph`/`extract_backward_graph` (simulate_train_graph.py:297,455) have **zero call sites anywhere in the repo** — transformer graphs are built with explicit `direction=` (train_timing.py:4987-4990). Stage 2's framing of them as transformer-root plumbing being replaced is wrong (deleting them is trivially safe, but the design misstates current dataflow).
2. **Gid allocation policy misdescribed.** A says "counter from 1000 in first-instantiation order" while R6 claims exact artifact match. Legacy allocates in **sorted-label, sorted-token** order (executor.py:1579-1590). First-instantiation program order differs in general; canonical-invisible but inconsistent with A's own "match" claim.
3. **"Memory ... verified by gate (4)"** — gate 4 pins end-to-end *times* (CONTEXT.md:88). Memory peaks are not gate-protected; only A's smoke-parity test covers them.
4. **Deadlock proof overstated.** Static per-rank id order being a projection of one total order does *not* force dynamic issue-order consistency: a higher-id same-group collective that becomes ready earlier issues first and occupies the comm slot (contract: lowest-id-first applies only among *simultaneously ready* ops). A's progress argument step 4 is circular for exactly this interleaving. Practically bounded by the dlsim gate, but "impossible by construction" is not proven.

**Design B**
1. **"None in the golden matrix, so no gate exposure" for optimize_2dmap is false.** `equiv/configs.py:285-298` defines `train:flattened:*:mesh2d_gmap` with `"optimize_2dmap": True`. SCOTCH permutation divergence at S2 would break gates 1-3 on a real golden spec, not just a synthetic parity test. This is B's only hard factual error, but it sits exactly on the S2 critical path.
2. **Same deadlock-proof gap as A** under the `sid` policy: "What can occupy the comm slot ahead of k? ... a COLL with a lower node id" omits the higher-sid COLL issued earlier because it was ready earlier. B's legacy-policy stance (id-identical to bundles the goldens already prove completable, plus the postcondition check) is honest; the sid-policy proof is not airtight.
3. Minor: `et_emit.py` at ~380 LOC hosting two full id policies, p2p materialization, renumber, comm_groups, and manifest is optimistic — legacy Steps 10-11 plus `_RankTrace` alone exceed that.

**Design C**
1. **"The extractors are used for other flows" is false** (§3.5) — as above, `extract_*_graph` is dead code. Consequently risk R12 defends a phantom equivalence surface (wasted differential-test budget, not a safety hole).
2. Everything else I spot-checked in C survived, including deep cuts the others miss: the mid-conversion `stage_to_ranks` extension using **pre-extension** `num_stages` for collective-only stages (executor.py:1251-1262, C's R8) — a real quirk neither A nor B models, and a latent bundle-diff risk for both of their clean device-count models.

## Scores

| | A (ir-first) | B (schedule-first) | C (strangler) |
|---|---|---|---|
| (a) correctness-by-construction | **9** | 8 | 8 |
| (b) migration safety | 7 | 8 | **9** |
| (c) end-state simplicity | **8** | 7 | 8 |
| (d) consumer coverage | **9** | 8 | 8 |
| (e) extensibility | 7 | **9** | 6 |
| **Total** | **40** | **40** | **39** |

Rationale under the maintainer lens:
- **(a)** A's single total order with ETs as pure projections is the cleanest inheritance of the contract property; B's everyday path is the `legacy` id policy (correctness inherited from goldens plus a checker, not from construction); C's end state matches A structurally and is the only design that names the residual dynamic collective race honestly (V6 + `serialize_groups` + dlsim).
- **(b)** C's shadow-lowering + emitter-first cutover is the safest sequencing in the field: the riskiest artifact (the emitter) cuts over while construction is still bit-identical legacy. B's dual id policy is purpose-built for gate 3 (exact wall seconds). A bets gate 3 on program order *coinciding* with legacy mutation order, "verified empirically" by a runner that doesn't exist yet, with regen as the escape hatch — the least-verified assumption of any design.
- **(c)** Two years out: A gives one order, one builder family, typed `OpKey` lookups — debugging a wrong number is an index lookup, a deadlock is one total order plus dlsim. B permanently carries two id policies and a hidden invariant that generator *creation order* is load-bearing (any generator refactor that reorders creation shifts `prio` → ids → wall seconds); every future investigation starts with "which policy emitted this?". C's designed end state is the leanest IR (3 op classes), but it is reached only at S6-S7 — if the strangle stalls at S5, George owns new builders emitting legacy `Node`/`Edge` plus a lowering pass indefinitely, which is worse than either alternative's failure mode.
- **(d)** All three cover the full consumer set; A is most explicit per consumer; B's probe-based memory unification is elegant but the gmap error is precisely a consumer-coverage slip; C defers overlap to S6 with a regen escape hatch (R16).
- **(e)** B is the only design where native interleaved-1F1B is *designed* (same step vocabulary, `vstage` tags, correction disabled) rather than deferred; flattened MoE is easy in all three. C explicitly declines 1F1B.

## Verdict

**1. Design A (ir-first)** — narrowly. The end state is the one George would actually enjoy owning: one typed program, one total order, every legacy quirk a single named rule, consumers as pure functions, and the fewest permanent concepts per debugging session. Its weakness is concentrated and mitigable: the gate-3 node-id bet at Stage 3b. It should be landed **with B's mitigation grafted on** — a temporary legacy-compatible id policy (deleted at migration end, unlike B's permanent one) rather than relying on order coincidence plus regens.

**2. Design B (schedule-first)** — a very close second and the winner if the org weights landing certainty and native 1F1B over end-state purity. Its honest treatment of "ids are priorities" is the most clear-eyed engineering in any of the three documents; its cost is that the split brain and the legacy-Kahn projection are *permanent* for all 35 existing configs, and generator creation order stays load-bearing forever.

**3. Design C (strangler)** — the most factually precise document and the safest per-landing plan, but it spends ~450 LOC re-implementing the legacy walkers only to delete them, requires exact-order reproduction twice per seam (legacy types at S3-S5, then Program uids at S6), delivers no new capability, and its payoff is entirely back-loaded onto the stages most likely to be deferred.

**Steal from each:**
- **From A:** typed `OpKey` semantic identity — retime write-back, memory attribution, fault variants, and cross-granularity mapping become index lookups instead of name-prefix matching (`startswith("transformer_layer")`, `"bwd" in name` all die at once).
- **From B:** the separation of schedule order from ET id policy, with a bit-compatible `legacy` emission mode plus the always-on per-group issue-order postcondition check that turns the historical silent deadlock into a loud emission-time failure.
- **From C:** the shadow-mode lowering harness — lower legacy graphs into the new IR and canonical-diff prototype emissions against the legacy converter *before* any cutover, so "new emitter correct" is proven independently of "new builders correct" (and take C's quirk catalog with it; its R8 collective-only-stage rank quirk at `astrasim_lib/executor.py:1251-1262` is a live landmine for A's and B's clean device models).