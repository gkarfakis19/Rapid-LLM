# MoE One-Hot-Expert Model

This document is the single authoritative description of RAPID-LLM's current
MoE hot-expert behavior.

It explains what `model_param.moe.expert_imbalance_factor` means, what exact
one-hot-expert assumption the simulator makes, how that assumption affects
compute and communication in training and inference, and what the model still
does not capture.

This document supersedes the earlier split across separate "improve plan",
"communication plan", and "limitations" notes.

## Short Version

The current model is not a general MoE routing simulator.

It is a first-order approximation built around one explicit assumption:

- exactly one routed expert is hot
- every other routed expert is uniformly colder
- the first rank in the MoE routing group owns that hot expert
- the hot rank is the one that determines routed MoE latency

Everything else in the model is derived from that assumption.

## Config Semantics

The relevant config knob is:

- `model_param.moe.expert_imbalance_factor`

Its meaning is:

- `1.0`: perfectly balanced routed experts
- `> 1.0`: one routed expert is hotter than the balanced average expert load

Validation enforces:

- `expert_imbalance_factor >= 1.0`
- `expert_imbalance_factor <= num_experts`

Why the upper bound exists:

- the model assumes one hot expert and all remaining experts share the leftover
  routed load uniformly
- if the factor exceeded `num_experts`, the implied cold-expert load would go
  negative, which is nonsensical

## Core One-Hot-Expert Assumption

Let:

- `E`: total routed experts
- `e`: experts per rank
- `alpha`: configured `expert_imbalance_factor`

The simulator assumes:

- one expert has load factor `alpha`
- the other `E - 1` experts all share the remaining routed load uniformly

From conservation of average expert load, the cold expert factor is:

```text
beta = (E - alpha) / (E - 1)
```

If multiple experts share a rank, the implied hot-rank factor is:

```text
gamma_hot_rank = (alpha + (e - 1) * beta) / e
```

Meaning:

- `alpha`: hot expert factor
- `beta`: cold expert factor
- `gamma_hot_rank`: worst-case rank factor implied by the hot expert and its
  colder local peers

## Concrete Example

Example:

- `10` experts total
- `5` ranks in the routing group
- `2` experts per rank
- `expert_imbalance_factor = 1.5`

Then:

```text
beta = (10 - 1.5) / 9 = 0.944...
gamma_hot_rank = (1.5 + 0.944...) / 2 = 1.222...
```

So the intended interpretation is:

- hot expert load: `1.5x`
- all cold experts: `0.944x`
- hot rank effective load: `1.222x`

This distinction matters:

- expert hotness is not the same thing as rank hotness when `experts_per_rank > 1`

## Training vs Inference Routing Groups

The one-hot assumption is applied over the simulator's logical MoE routing
group. That group differs between training and inference.

### Training

In training:

- the full graph expands over `tp * cp * ep` ranks
- MoE dispatch and combine use routing group size `ep`
- with `tp > 1`, each TP slice has its own EP routing group

So with:

- `tp = 3`
- `ep = 5`
- `cp = 1`

the simulator models:

- three separate 5-rank MoE routing groups

not:

- one literal 3-to-5 producer/consumer exchange

Hot-rank convention in training:

- the first rank in each EP group is the hot rank
- equivalently, for fixed `(tp_idx, cp_idx)`, the hot rank is `ep_idx = 0`

### Inference

In inference:

- `ep` is overloaded to represent `moe_dp`
- the MoE routing group size is `tp * moe_dp`
- the routing group is the pooled TP-by-MoE-DP product within a CP slice

So with:

- `tp = 3`
- `moe_dp = 5`
- `cp = 1`

the simulator models:

- one 15-rank MoE routing pool per CP slice

not:

- separate 5-rank expert groups inside each TP slice

Hot-rank convention in inference:

- the first rank in the pooled `tp * moe_dp` group is the hot rank
- equivalently, for fixed `cp_idx`, the hot rank is `tp_idx = 0, ep_idx = 0`

## Compute Model

The routed expert FFN compute path is where the one-hot-expert refinement
matters most.

### Balanced Case

In the balanced model, every local routed expert on a rank is treated as having
the same token count.

That worked reasonably well when:

- `experts_per_rank == 1`

It became unfair when:

- `experts_per_rank > 1`

because the old model effectively flattened the hot rank into an evenly
inflated local expert shape.

### Current Hot Compute Model

Today, when `experts_per_rank > 1`, the hot rank is modeled with two routed
expert buckets:

- one hot local expert bucket
- one cold bucket covering the remaining local experts

So the hot rank is represented more like:

```text
[alpha, beta, beta, ...]
```

instead of:

```text
[gamma_hot_rank, gamma_hot_rank, gamma_hot_rank, ...]
```

This is still an approximation, but it is much fairer than evenly inflating all
local experts.

### What This Means Operationally

Compute is still not modeling explicit expert identity over time.

Instead, it is modeling the worst-case routed MoE FFN shape implied by the
one-hot assumption.

So the compute model should be interpreted as:

- one-hot-expert load imbalance
- with a hot/cold local expert split on the hot rank

not:

- full per-expert trace-based scheduling

## Communication Model

Communication is still the least faithful part of the model, because the
simulator and Astra-Sim interface do not carry a true MoE sender-by-destination
traffic matrix.

### Why We Changed It

The older hot-MoE approximation inflated the MoE dispatch/combine all-to-all
payload directly using the hot-rank load.

That was unsatisfying because:

1. it changed total routed byte volume
2. it still looked like a symmetric all-to-all to the communication model

So it was:

- pessimistic on total bytes
- optimistic on real traffic skew

at the same time

### Current Communication Decomposition

The current model keeps total routed bytes conserved and decomposes each MoE
dispatch or combine into:

1. a reduced base all-to-all
2. a residual hot-path point-to-point transfer

Let `B_total` be the balanced physical routed activation bytes for one MoE
dispatch or combine. Then:

```text
B_a2a_base = floor(beta * B_total)
B_residual_total = B_total - B_a2a_base
B_residual_per_sender = ceil(B_residual_total / (routing_group - 1))
```

Interpretation:

- every rank participates in the reduced base all-to-all
- every non-hot rank contributes one residual transfer
- the hot rank receives the residual transfers

Communication uses `beta`, not `gamma_hot_rank`, for the base all-to-all
reduction, because the base all-to-all is meant to represent the symmetric
"cold-rank" exchange.

### Direct Timing Interpretation

The direct timing path treats the base A2A and residual P2P as concurrent and
charges the MoE communication time using the hot-path completion behavior,
rather than symmetrically inflating all routed bytes.

### Graph-Level Interpretation

The transformer graph explicitly represents this decomposition.

For each MoE dispatch or combine:

- the previous compute node fans out to the base all-to-all edge
- non-hot ranks also fan out to a residual `PIPELINE` edge
- every rank inserts a zero-duration local join node

Blocking behavior:

- on non-hot ranks, the local join waits for:
  - the base all-to-all
  - the local residual send completion
- on the hot rank, the local join waits for:
  - the base all-to-all
  - all incoming residual receives

This is important. The model does not serialize:

- base A2A
- then residual P2P

It models them as concurrent paths whose completion feeds the local post-comm
barrier.

### Why Residual Uses `PIPELINE`

There is no dedicated point-to-point primitive in the current timing model.
So the residual path is represented with `CollectiveType.PIPELINE`.

That is an approximation:

- in analytical mode, it behaves like a point-to-point transfer
- in hierarchical Astra-Sim conversion, it becomes SEND/RECV nodes

This should be read as:

- "nearest available primitive for an explicit residual transfer"

not:

- "full arbitrary MoE traffic matrix support"

## What Is Implemented Across Modes

The one-hot-expert model is applied consistently across:

- training forward
- training backward where relevant to MoE routed work
- inference prefill
- inference decode

The grouping logic is mode-aware:

- training uses EP groups per TP slice
- inference uses pooled `tp * moe_dp` groups per CP slice

The compute refinement is shared:

- training and inference reuse the same one-hot profile logic
- multi-expert hot ranks use hot/cold routed expert buckets

The communication refinement is also shared:

- conserved byte volume
- reduced base A2A
- residual hot-path P2P

## What The Model Does Not Capture

The current model still does not attempt to capture several important real-world
effects.

- no true sender-by-destination traffic matrix
- no congestion-aware skewed all-to-all behavior
- no hotspot-aware link-level network effects
- no capacity overflow, token dropping, reroute, or backup-expert policies
- no router-side permutation, bucketization, metadata exchange, or unpacking
  micro-model
- no per-layer or per-step routing trace
- no arbitrary expert placement beyond the fixed "first rank is hot" convention
- no heterogeneous expert structure; all experts still share one expert shape

So this is still a first-order model.

## Where It Is Most Faithful

This approximation is most defensible when:

- one routed expert is meaningfully hotter than the rest
- the rest of the experts are roughly uniform
- the dominant system effect is hot-rank completion time
- batch sizes are large enough that integer routing noise is not dominant

That tends to make it more trustworthy for:

- training
- inference prefill
- larger-batch decode

than for:

- tiny-batch decode
- step-by-step router microanalysis

## Small Decode Caveat

Decode can still be coarse when token counts are very small.

The one-hot expert split eventually has to become integer token counts before it
turns into GEMM shapes and routed bytes. At small token counts, a continuous
imbalance factor may either:

- collapse to no visible skew
- or snap to a coarse integer pattern

So decode trends are still useful, but they should not be over-interpreted as a
faithful micro-model of per-token router behavior.

## Practical Interpretation

The current feature should be described as:

- first-order one-hot-expert imbalance support for train and inference
- improved fairness for `experts_per_rank > 1`
- approximate communication decomposition using reduced A2A plus residual
  hot-path P2P

It should not be described as:

- exact MoE routing simulation
- congestion-aware skewed all-to-all modeling
- arbitrary mismatched-expert support

## Bottom Line

RAPID-LLM's current hot-MoE behavior is a one-hot-expert approximation.

It assumes:

- one routed expert is hot
- all others are uniformly colder
- the first rank in the routing group owns the hot expert

From that assumption, it derives:

- hot and cold expert compute behavior
- the hot-rank compute shape when multiple experts share a rank
- a conserved-volume MoE communication decomposition

This gives a materially better first-order model than the older uniformly
inflated hot-rank approximation, while staying inside the current simulator and
Astra-Sim interfaces.
