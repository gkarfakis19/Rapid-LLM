# ECE201C RAPID-LLM projects

Repository: https://github.com/gkarfakis19/Rapid-LLM (branch `course`)

**Before you start, read [COURSE_README.md](COURSE_README.md).** It lists the helpers we added to
RAPID-LLM for these projects and where each project starts in the code. [README.md](README.md)
covers setup and running a config.

Each project adds one idea from current LLM serving or model design to RAPID-LLM. First, run
exactly the evaluation we specify below. Then go further with case studies of your own that show
off your feature. Change the hardware, the model, the batch size or the sequence lengths, and find
where your feature helps, where it stops helping, and why. Ask what it means for hardware design:
does it shift a workload from compute-bound to memory-bound, change how much memory a GPU needs,
or put more weight on the network? The project slides shown in class have examples, but we
encourage you to think outside the box. Show why your feature is interesting and important, and
how it changes the trade-offs between hardware and model architecture.

All evaluations use `configs/hardware-config/course_project/a100_80GB_1gpu.yaml` (one A100 80 GB).
Put all your settings in the model config; do not edit the hardware configs.

## Project 1: Attention/FFN disaggregation (AFD)

Estimate what splitting decode attention and the FFN onto separate GPUs would cost. Run prefill as
usual, then time the attention and FFN parts of decode separately, and add the cost of moving data
between them at every layer. Report time per token, with and without overlap, and the largest batch
each side can hold.

**Evaluate:** `configs/model-config/course_project/qwen3_30b_a3b_inf.yaml`, batch 32, prompt 2048,
output 128 and 256, link 100 GB/s with 5 µs latency.

## Project 2: Sliding-window attention

Add sliding-window attention for GPU runs: local layers attend only to the last w tokens, and global
layers to everything. Report time per token, prefill time, peak memory and largest batch, each
compared with dense attention.

**Evaluate:** `configs/model-config/course_project/gemma_3_4b_inf.yaml` (w = 1024, 5 local : 1
global), batch 16, prompt 8192, output 128 and 256.

## Project 3: Block diffusion LLM

Model a block diffusion LLM: output is generated in blocks of b tokens, each refined over R rounds.
Report total time, number of model calls and peak memory, each compared with normal decode.

**Evaluate:** `configs/model-config/course_project/Llama3.1-8B_inf.yaml`, batch 1, 16 and 64,
prompt 512, output 128 and 256, b = 8, R = 4.

## Project 4: Speculative decoding with MTP

Model speculative decoding with D multi-token-prediction (MTP) modules that guess tokens and one
full-model pass that checks them. Report time per output token, speedup over normal decode, and
peak memory.

**Evaluate:** `configs/model-config/course_project/mimo_7b_inf.yaml`, batch 16, prompt 2048 and 8192,
output 128 and 256, D = 3, acceptance p = 0.8.
