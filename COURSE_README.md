# Course projects in RAPID-LLM

Read this before you start your project. [PROJECTS.md](PROJECTS.md) says what each project must
build and evaluate; this file says where that lives in the code. For setup and running a config,
see [README.md](README.md), including "How a decode step is priced".

All project inputs go in the model YAML. The hardware configs are shared: do not edit them.

## Helpers

Four small hooks are in the code. Each is off by default and changes no number until used.

- **H1 `project_param`** (all projects). Add a top-level block to the model YAML, next to
  `model_param`:
  ```yaml
  project_param:
    my_switch: true
    my_number: 4
  ```
  It is read as written (no unit conversion) and is available on the timing object as
  `self.project_param`, a dict (`{}` when absent). See `_read_project_param` in config.py.
- **H2 `query_tokens`** (Projects 3 and 4): new tokens per decode call. It is passed from
  `build_decode_graph` down the decode path to `autoregressive_decoder_gemm`, and appears at the
  decode-memory call in `calc_time`. The shapes still use a literal 1, so any other value raises
  `NotImplementedError`. Deciding what changes when it is not 1 is your job.
- **H3 `kv_cache_bytes_by_layer`** (Project 2). If `build_memory_data` adds this list (one entry
  per layer), the memory replay in simulate_train_graph.py sums it over the layers each GPU holds
  instead of using `kv_cache_bytes_per_layer` times the layer count.
- **H4 `block_timings_by_kind` / `layer_block_kinds`** (Project 2): layers of different kinds.
  `_prepare_execution_graphs` (train_timing.py) takes `layer_block_kinds`, one kind name per
  layer. `"dense"` and `"moe"` are the two templates it already builds; any other name must be a
  key of `block_timings_by_kind`, mapped to a `(transformer_timings, node_breakdown)` pair built
  like the dense one (for example by `_build_decode_transformer_results`). Each layer then takes
  `transformer_time_f` of its own kind. Both arguments are already passed, empty, in
  `prepare_decode_graphs` (decode) and `calc_time` (prefill).

## Where to start, by project

### Project 1: attention/FFN disaggregation (Qwen3-30B-A3B)

Estimate what it would cost to run decode attention and the feed-forward (expert) part of each
layer on separate GPUs. Do it analytically from single-GPU runs: prefill as usual, then decode with
only the attention side of each layer, and with only the FFN side, and add the data transfers
between the two sides yourself. Report time per token with and without overlap, and the largest
batch each side can hold. No multi-GPU or routing code is needed.

- Prefill-only and decode-only runs need no code: set the output to 0 (`decode_len: 0`, `seq_len`
  = prompt) for prefill only, or the prompt to 0 (`seq_len` = `decode_len`) for decode only. With
  a prompt of 0, decode starts from an empty context; to keep the prompt's KV context, subtract a
  prefill-only run from a full run instead.
- RAPID-LLM has no switch for attention-only or FFN-only time. You have to do surgery inside the
  model so that a run reports only one side. Start at `_build_decode_transformer_results`
  (inference_timing.py): the per-layer decode time is `transformer_time_f` in its
  `node_breakdown`, not the sum of the operation table.
- `MemoryEstimator.build_memory_data` (memory_estimation.py): the weights, KV cache, embedding and
  output head each side would hold.
- `project_param` (H1) may be useful.


### Project 2: sliding-window attention (Gemma 3 4B)

Gemma 3 has 5 local layers, which attend only to the last w = 1024 tokens, for every global layer.
Today RAPID-LLM refuses such a model. Make it price local layers in decode, prefill and memory,
and compare with dense attention: time per token, prefill time, peak memory and largest batch.

- `AttentionWindowConfig` (config.py) holds the parsed window and the local/global layer rule; the
  refusal is in `validate_model_config`. The window reaches the timing object as
  `self._raw_model_config.model_config.attention.window`.
- H4: price a local layer once (decode: `_build_decode_transformer_results` at a context of
  min(S, w)), name each layer's kind in `layer_block_kinds`, and pass both where
  `prepare_decode_graphs` and `calc_time` call `_prepare_execution_graphs`.
- `flash_attention_kernel_forward` (train_timing.py) for prefill; H3 for per-layer KV memory;
  `_generate_sample_points` decides which decode steps are priced.


### Project 3: block diffusion (Llama-3.1-8B)

A block-diffusion model produces its output in blocks of b tokens, refining each block over R
rounds, and every call works on all b positions of the block at once. Model this and compare with
normal decode: total time, number of model calls and peak memory.

- H2 `query_tokens`, from `build_decode_graph` down to `autoregressive_decoder_gemm` (llm_util.py),
  and `output_seq_len` in `_build_decode_transformer_results`.
- The decode loop: `build_decode_graph`, `_generate_sample_points` and `_integrate_decode_samples`
  (simulate_inference_graph.py).
- `project_param` (H1) may be useful.


### Project 4: MTP speculative decoding (MiMo-7B)

With D multi-token-prediction (MTP) modules, each round makes D cheap guesses and then one pass of
the full model checks D + 1 positions; each guess is accepted with probability p. The MiMo YAML
models only the base model. Model the rounds and compare with normal decode: time per output
token, speedup and peak memory.

- H2 for the checking pass (the same shape work as Project 3).
- `transformer_time_f` and `linear_softmax_f` from `_build_decode_transformer_results`;
  `get_gemm_time` (base_timing.py) prices any new matrix multiply.
- `memory_data["extra_static_bytes_per_device"]` (memory_estimation.py) for extra weights and KV.
- `project_param` (H1) may be useful.
