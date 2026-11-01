# Shared SSM schema and GPU inference

The sole SSM/hybrid schema is `model_param.layer_plan` plus typed block
parameters. `model_type` selects FFN arithmetic, not the mixer schedule.
`run_type: inference` in the model YAML is the rule for both GPU and CIM;
neither path silently converts training to inference. GPU SSM training is
out of scope. Existing CIM laws and pricing are retained.

## Migration from archive/ssm-gpu-old (fdbb0ea)

Run `./.venv/bin/python tools/convert_ssm_config.py old.yaml new.yaml`.
Runtime parsing refuses retired keys by name. It does not alias them.

| Archive key | Shared key | Meaning |
|---|---|---|
| state_size | ssm.d_state | recurrent state width per channel |
| conv_kernel | ssm.d_conv | depthwise causal filter length |
| num_heads | ssm.n_heads | mixer heads, not attention heads |
| headdim or head_dim | ssm.d_head | mixer channels per head |
| expand | ssm.expand | mixer width = expand * hidden_dim |
| n_groups | ssm.n_groups | B/C and RMS groups |
| linear.linear_type: mamba2 | ssm.variant: mamba2 | head-structured SSD |
| hybrid_schedule | layer_plan.layer_types | explicit per-layer kinds |
| chunk_size | ssm.chunk_size | SSD prefill chunk length |

The first six dimensions may formerly appear at model level, in `ssm`, or
in `linear`. The converter rejects conflicting duplicates. The archive has
no chunk-size timing input; migration declares 256 explicitly. This is a
NEW prefill assumption, not a recovered old parameter.
Explicit old schedules are 1-based; the converter writes a complete ordered
list. Interval 0 means all SSM. The archive's `int(interval or 1)` bug is not
preserved. Pure Mamba uses the GPT/GELU FFN carrier; Granite/Jamba use
llama/SwiGLU, as in the archive. Mixer-only models set `ffn_per_layer: false`;
their required positive intermediate_size is unused.

Conv history contains `(d_conv - 1)` cached inputs per channel, including
the Mamba2 X/B/C channels. The archive memory helper used `d_conv`, one
extra slot. CIM already uses `d_conv - 1`; its values remain unchanged.

The archive only records RMS subgroup **counts**, not rank IDs. For TP=T,
groups=G, let g=gcd(G,T), local groups=G/g, subgroup size s=T/g.
The shared GPU convention assigns contiguous TP coordinates
`[floor(t/s)*s, ..., floor(t/s)*s+s-1]`, fixing PP/EP/CP coordinates.
Statistics are all-reduced only when s>1, with B*tokens*(G/g)*stats_bytes
per rank. This preserves the archive's counts and gcd law. Rank membership
is now explicit in the program IR. NEEDS GEORGE: approve this explicit
membership convention; the archive did not establish rank IDs.

## Implementation and evidence

Implementation and validation results are recorded below as the port lands.
