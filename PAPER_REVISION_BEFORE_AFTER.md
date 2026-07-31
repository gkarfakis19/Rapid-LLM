# RAPID-LLM paper revision: unresolved before/after items

Source: `tmp/actual_paper/690154783a2b7835ba730fbe/TACO/main.tex`

Accepted edits have been moved into the manuscript and removed from this file.
Only unresolved or explicitly deferred items remain.

## 4. Section 4 workload description: DeepFlow provenance

**What still needs deciding:** We should weaken the DeepFlow ancestry claim
without defining RAPID-LLM by what it is not. The replacement must begin by
positively explaining RAPID-LLM (for example, “RAPID-LLM builds on…” or
“RAPID-LLM expands…”).

**Relevant PI comments:**

- DeepFlow accounts for only roughly 5--10\% of the current code.
- Cite DeepFlow, but do not present RAPID-LLM as fundamentally based on it.
- Focus on RAPID-LLM's capability and new modeling rather than implementation
  ancestry.

### BEFORE

```latex
RAPID-LLM builds on DeepFlow~\cite{pal2023deepflow}, extending it from LSTM-based models to modern decoder-only LLMs.
```

-----

### AFTER

Unresolved. The previous proposed replacement was rejected.

## 7. Section 4.1 memory model: choose the right level of detail

**What still needs deciding:** The PI asked for concrete memory-model details,
including KV-cache sizing, but it is unclear whether this warrants several
equations. We need a compact replacement that makes the model reproducible
without overwhelming Section 4.1.

**Relevant PI comments:**

- Specify the full memory model, including the KV-cache size expression.
- Treat memory modeling as a distinct part of Section 4.1.

### BEFORE

```latex
\paragraph{Memory modeling.}
RAPID-LLM also checks whether a given configuration fits in device memory. For each GPU, the frontend first estimates static memory usage for parameters, optimizer state, gradients, and (for inference) KV caches under the chosen parallelism, precision (mixed vs FP32) and ZeRO~\cite{rajbhandari2020zero} (stages 1,2,3) sharding settings. It then performs a simulated traversal of the computation graph under a specified recomputation policy (full vs selective vs off \cite{korthikanti2023reducing}), tracking live activations over time and recording the peak activation footprint per GPU for the given hybrid parallelism configuration. It can then prune configurations that exceed the total memory capacity of the GPU, restricting analysis to memory-feasible designs.
```

-----

### AFTER

Unresolved. The previous three-equation expansion was rejected as likely too
heavy.

## 11. Section 5.3 training validation: FSDP error explanation

**What still needs deciding:** Whether and how to explain the higher FSDP
validation error. The PI did explicitly request an explanation, but the
previous causal language was inferential and has not been accepted.

**Relevant PI comments:**

- “have some explanation of why FSDP error is large”

### BEFORE

```latex
Across all 52 rows, spanning MPT 760M to 70B, sequence lengths 512 to 65k, and 8 to 64 GPUs, RAPID-LLM reaches a MAPE of 13.2\% (Figure~\ref{fig:mpt_train}).
```

-----

### AFTER

Unresolved. If retained, this should be one evidence-backed sentence and should
not speculate about an unverified cause.

## 13. Section 6.2 fault study

**What still needs deciding:** The experimental framing and final prose need a
separate pass. Do not apply the prior replacement.

**Relevant PI comments:**

- Soft-versus-hard framing is misleading when routing responses differ.
- The central conclusion should concern degradation detection and routing
  response, not simply nominal fault severity.
- Retain Llama if it preserves the best-versus-second-best reversal; GLM may be
  removed if it adds no separate insight.
- Correct the activation-sharding discussion from DP to CP/TP.

### BEFORE

```latex
\subsection{Faulty links and fault-aware parallelism selection}
```

The current subsection and its figures remain unchanged in the manuscript.

-----

### AFTER

Deferred. The previous proposed subsection replacement was rejected.

## 14. Section 6.4 3D-stack study

**What still needs deciding:** The section needs a more restrained, coherent
revision. Do not apply the prior setup table or rewritten conclusion.

**Relevant PI comments:**

- Use “inference throughput,” not “serving throughput.”
- Explain the setup, including parallelism, batch size, network topology, and
  per-GPU-count normalization.
- State the 20-$\mu$s$ result precisely.
- Do not make universal bottleneck claims.

### BEFORE

The current 3D-stack subsection remains unchanged in the manuscript.

-----

### AFTER

Deferred. The previous proposed replacement was rejected.

## 15. Limitations

**What still needs deciding:** Whether the fault limitation should change at
all. Do not apply the prior narrowing.

**Relevant PI comments:**

- Routing response, rather than the fault label alone, determines the observed
  impact.

### BEFORE

```latex
On the network side, the backend is analytical and congestion-aware with explicit multidimensional links and fault-aware routing, but it does not model all packet-level dynamics (e.g., transport-level congestion control and detailed switch-buffer behavior). Likewise, our fault model captures soft and hard link faults and their rerouting consequences, but does not represent correlated or time-varying failures such as bursts, multi-link events, or switch failures.
```

-----

### AFTER

Deferred. The previous proposed replacement was rejected.

## Other unresolved production items

- Several figure labels are too small to read.
- Figures 8 and 9 may be reduced or consolidated to recover space.
