# RAPID-LLM (course branch)

RAPID-LLM estimates how long a large language model takes to run on given hardware, and how much
GPU memory it needs. It does not run the model. It turns the model shape (layers, hidden size,
heads, experts) into matrix multiplications and other operations, prices each one from the GPU's
compute rate and memory bandwidth, adds communication time when several GPUs work together, and
adds everything up.

This course branch runs on the analytical backend. You will change the core code for one of four
projects. This file names functions, not line numbers.

## Setup (Python 3.12)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt pytest
pytest -q                           # should pass
```

## Running one config

```bash
python run_perf.py --hardware_config configs/hardware-config/course_project/a100_80GB_1gpu.yaml \
                   --model_config configs/model-config/course_project/Llama3.1-8B_inf.yaml
```

Results go to `output/LLM/`: `LLM_inference_results.txt` repeats the summary, and
`memory-summary/memory_capacity_comparison.txt` gives the prefill, decode and maximum peak memory
per GPU. The lines that matter on screen:

```
[decode] sample step 0: seq_len=1025, time=0.0121s
[decode] sample step 1023: seq_len=2048, time=0.0122s
[prefill] time: 0.0661s, [decode] time: 12.4618s, [total] time: 12.5279s
[kv-cache] prefill_store=0.1 GiB, decode_store=0.1 GiB, decode_fetch=192.0 GiB
LLM time to first token: 0.08s
Aggregate decode throughput tok/s (batch=1, replica_count=1): start=82.40, mid(token 512)=82.17, end=81.94
```

- `sample step`: one decode step (one new token for every sequence) at that context length.
- `prefill` processes the prompt once; `decode` generates all output tokens; time to first token
  is prefill plus the first decode step.
- `kv-cache`: key/value bytes written in prefill and decode and read in decode. `tok/s`: tokens
  per second over the whole batch at the start, middle and end of decoding.

## Configs

Put all your settings in the model config. The hardware configs are shared: do not edit them.

`configs/model-config/course_project/` holds the four project models, and
`configs/hardware-config/course_project/` the GPUs used in the course.

| Model config | What it is |
|---|---|
| `course_project/qwen3_30b_a3b_inf.yaml` | Qwen3-30B-A3B, mixture of experts (Project 1) |
| `course_project/gemma_3_4b_inf.yaml` | Gemma 3 4B text model with sliding window (Project 2) |
| `course_project/Llama3.1-8B_inf.yaml` | Llama-3.1-8B, dense (Project 3) |
| `course_project/mimo_7b_inf.yaml` | MiMo-7B base model, dense (Project 4) |
| `Llama3.1-70B_inf.yaml`, `Llama3.1-405B_inf.yaml` | larger dense Llama models |
| `DeepSeekV3_inf.yaml`, `GLM4.7_358B_inf.yaml` | large mixture-of-experts models |
| `minicpm3_4b_inf.yaml` | a small dense model with MLA attention |

| Hardware config | What it is |
|---|---|
| `course_project/a100_80GB_1gpu.yaml` | one A100 80 GB (all project evaluations) |
| `course_project/a100_80GB.yaml` | two A100 80 GB, tensor parallel (tp: 2) |
| `course_project/H100_SXM5_80GB.yaml` | one H100 SXM5 80 GB |
| `waferscale_20v100_80hbm.yaml` | 20 V100-class GPUs in one node, tensor parallel |

The Gemma file is refused until Project 2 is done; delete its `window:` block to run it as a dense
model. Units: in the YAML files `GB` means 2^30 bytes (as a bandwidth, 2^30 bytes per second);
latencies are in seconds.

## Changing the inputs

In the model YAML, under `model_param`: `global_batch_size` is the number of sequences served
together (B), `decode_len` the decode (output) tokens per sequence, and `seq_len` the prefill plus
decode tokens; to change only the output length, change both by the same amount. Under
`inference_param`, `sample_every: -1` (the default) prices only the first and last decode step;
`sample_every: N` also prices every Nth step.

Your own inputs (link bandwidth and latency, window size, block size, number of guesses, and so
on) go in a `project_param:` block in the model YAML; see H1 below.

## How a decode step is priced

1. `TimeCalculationLLMInference.calc_total_inference_time` (inference_timing.py) runs prefill
   and peak memory in `calc_time`, then decode in `calc_decode_time`, which hands over to
   `DecodeGraph.build_decode_graph` (simulate_inference_graph.py).
2. `build_decode_graph` picks the sampled steps (`_generate_sample_points`; by default only the
   first and last step), computes the GEMM shapes for the current context length
   (`llm_util.process_decode_gemm_shapes`, which calls `autoregressive_decoder_gemm`) and calls
   `_execute_decode_step`.
3. `_execute_decode_step` makes a fresh timing object and calls `prepare_decode_graphs`
   (inference_timing.py).
4. `prepare_decode_graphs` calls `_build_decode_transformer_results` for a dense layer and, for
   MoE models, an MoE layer. Each call returns a table of operations (`qkv_proj`, `attention`,
   `ffn1`, ...) and a `node_breakdown` holding the per-layer time `transformer_time_f` and the
   output-head time `linear_softmax_f`.
5. `_prepare_execution_graphs` (train_timing.py) turns these into a graph of layers
   (simulate_train_graph.py), and `LLMExecutionDispatcher.run` (llm_execution.py) returns the
   step time.
6. `_integrate_decode_samples` joins the sampled steps with straight lines (trapezoid rule) to
   get the total decode time.

Peak memory: `calc_time` calls `MemoryEstimator.build_memory_data` (memory_estimation.py) for
weights, KV cache and activations, then `simulate_peak`, which replays the layer graph
(simulate_train_graph.py). Going over the GPU's memory prints a warning; it does not stop the run.

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
  `self.project_param`, a dict (`{}` when absent). Nothing checks the names, so a typo is
  silently ignored. See `_read_project_param` in config.py.
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

## Projects

For your project, first run exactly the evaluation given below, then run case studies of your own
that show off your feature. All evaluations use `course_project/a100_80GB_1gpu.yaml`. The pointers below
are where to start reading, not a recipe.

## Project 1: attention/FFN disaggregation (Qwen3-30B-A3B)

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
- Use `project_param` (H1) to choose the side.

Evaluate: `qwen3_30b_a3b_inf.yaml`, batch 8, prompt 512, output 128 and 256, link 100 GB/s with
5 µs latency.

## Project 2: sliding-window attention (Gemma 3 4B)

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

Evaluate: `gemma_3_4b_inf.yaml` (w = 1024, 5 local : 1 global), batch 4, prompt 8192, output 128
and 256.

## Project 3: block diffusion (Llama-3.1-8B)

A block-diffusion model produces its output in blocks of b tokens, refining each block over R
rounds, and every call works on all b positions of the block at once. Model this and compare with
normal decode: total time, number of model calls and peak memory.

- H2 `query_tokens`, from `build_decode_graph` down to `autoregressive_decoder_gemm` (llm_util.py),
  and `output_seq_len` in `_build_decode_transformer_results`.
- The decode loop: `build_decode_graph`, `_generate_sample_points` and `_integrate_decode_samples`
  (simulate_inference_graph.py).
- Put b and R in `project_param` (H1).

Evaluate: `Llama3.1-8B_inf.yaml`, batch 1, prompt 512, output 128 and 256, b = 8, R = 4.

## Project 4: MTP speculative decoding (MiMo-7B)

With D multi-token-prediction (MTP) modules, each round makes D cheap guesses and then one pass of
the full model checks D + 1 positions; each guess is accepted with probability p. The MiMo YAML
models only the base model. Model the rounds and compare with normal decode: time per output
token, speedup and peak memory.

- H2 for the checking pass (the same shape work as Project 3).
- `transformer_time_f` and `linear_softmax_f` from `_build_decode_transformer_results`;
  `get_gemm_time` (base_timing.py) prices any new matrix multiply.
- `memory_data["extra_static_bytes_per_device"]` (memory_estimation.py) for extra weights and KV.
- Put D and p in `project_param` (H1).

Evaluate: `mimo_7b_inf.yaml`, batch 1, prompt 512, output 128 and 256, D = 3, p = 0.8.
