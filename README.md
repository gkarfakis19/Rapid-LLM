# RAPID-LLM (course branch)

RAPID-LLM estimates how long a large language model takes to run on given hardware, and how much
GPU memory it needs. It does not run the model. It turns the model shape (layers, hidden size,
heads, experts) into matrix multiplications and other operations, prices each one from the GPU's
compute rate and memory bandwidth, adds communication time when several GPUs work together, and
adds everything up.

This course branch runs on the analytical backend. This file covers setup, running a config and
how RAPID-LLM prices a decode step. For the course projects, read
[COURSE_README.md](COURSE_README.md): the helpers we added and where to start in the code.

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
on) go in a `project_param:` block in the model YAML; see H1 in [COURSE_README.md](COURSE_README.md).

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
