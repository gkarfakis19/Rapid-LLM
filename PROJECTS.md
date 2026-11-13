# ECE201C RAPID-LLM projects

Repository: https://github.com/gkarfakis19/Rapid-LLM (branch `course`)

**Before you start, read [COURSE_README.md](COURSE_README.md).** It lists the helpers we added to
RAPID-LLM for these projects and where each project starts in the code. [README.md](README.md)
covers setup and running a config.

Each project adds one idea from current LLM serving or model design to RAPID-LLM. First, run
exactly the evaluation we specify below. Then go further with case studies of your own that show
off your feature. Change the hardware, the model, the batch size or the sequence lengths, and find
where your feature helps, where it stops helping, and why. The project slides shown in class have
examples, but we encourage you to think outside the box.

All modeling is inference only, on one GPU. All evaluations use
`configs/hardware-config/course_project/a100_80GB_1gpu.yaml` (one A100 80 GB). Put all your
settings in the model config; do not edit the hardware configs.

## Project 1: Attention/FFN disaggregation (AFD)

Estimate what it would cost to run decode attention and the feed-forward (FFN, here the experts)
on separate GPUs. Attention reads the KV cache, which grows with batch and context; the FFN reads
the expert weights. Run prefill as usual, then time the attention and FFN parts of each decode step
separately. At every layer the hidden state (B × H values) goes from the attention GPU to the FFN
GPU and back; add the cost of these transfers yourself (latency + bytes / bandwidth). Report time
per token with no overlap (attention, then transfer, then FFN, then transfer) and with perfect
overlap (the slowest of the three), and the largest batch each side can hold: the attention GPU
keeps the attention weights and the KV cache, the FFN GPU keeps the expert weights. No routing
between FFN GPUs is needed.

**Evaluate:** `configs/model-config/course_project/qwen3_30b_a3b_inf.yaml`, batch 32, prompt 2048,
output 128 and 256, link 100 GB/s with 5 µs latency.

## Project 2: Sliding-window attention

Gemma 3 mixes local and global attention layers: 5 local layers, then 1 global, repeating. A local
layer attends only to the last w tokens; a global layer attends to everything. Add this to RAPID-LLM
for GPU runs. In decode, a local layer reads at most w tokens of KV cache. In prefill, a local layer
computes only the band of width w instead of the full triangle. In memory, a local layer keeps only
w tokens of KV cache. Global layers are unchanged. Report time per token, prefill time, peak memory
and the largest batch that fits, each compared with dense attention (the same model with every
layer global).

**Evaluate:** `configs/model-config/course_project/gemma_3_4b_inf.yaml` (w = 1024, 5 local : 1
global), batch 16, prompt 8192, output 128 and 256.

## Project 3: Block diffusion LLM

A block diffusion LLM writes its output in blocks of b tokens. Each block starts as b unknown
positions and is refined over R model calls; every call processes all b positions at once and reads
the KV cache of all earlier text. When the block is done it joins the context, and the next block
starts. So a decode call handles b tokens instead of one: the weights and KV cache are read once
per call, but the compute grows with b. Model this and compare with normal decode (one token per
call). Report total time, number of model calls and peak memory.

**Evaluate:** `configs/model-config/course_project/Llama3.1-8B_inf.yaml`, batch 1, 16 and 64,
prompt 512, output 128 and 256, b = 8, R = 4.

## Project 4: Speculative decoding with MTP

Multi-token prediction (MTP) adds D small modules to the model. Each module is one transformer
layer plus a projection from 2H to H, and shares the model's embedding and output head. In each
round the D modules guess the next D tokens one after another, then one pass of the full model
checks all D + 1 positions at once. Each guess is accepted with probability p, and a round stops at
the first wrong guess, so a round produces 1 + p + p² + … + p^D tokens on average. Model the rounds,
including the cost of the guesses, the D + 1-token check, and the memory for the MTP layers and
their KV cache. Report time per output token, speedup over normal decode, and peak memory.

**Evaluate:** `configs/model-config/course_project/mimo_7b_inf.yaml`, batch 32 and 64, prompt 2048,
output 128 and 256, D = 3, acceptance p = 0.8 and 0.6.
