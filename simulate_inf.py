"""
LLM Inference Simulation - Extended Graph Generation for Prefill + Decode

This module extends DeepFlow's LLM training simulation to support full inference
workflows including both prefill and autoregressive decode phases.
"""

import math
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass

from simulate_LLM import Graph, Node, Edge
import LLM_util

def _measure_decode_gemm(
    *,
    temp_time_calc,
    name: str,
    gemm: Tuple[int, ...],
) -> Dict[str, float]:
    if len(gemm) == 3:
        gemm_args = gemm
    elif len(gemm) == 4:
        batch, seq_len, k_dim, n_dim = gemm
        gemm_args = (batch * seq_len, k_dim, n_dim)
    else:
        raise ValueError(f"Invalid GEMM shape for {name}: {gemm}")

    gemm_time, reduction_time = temp_time_calc._distributed_gemm_forward(gemm_args, f"decode_{name}_f")

    return {
        "forward": gemm_time + reduction_time,
        "backward": 0.0,
        "forward_gemm": gemm_time,
        "forward_reduction": reduction_time,
        "backward_gemm": 0.0,
        "backward_reduction": 0.0,
    }


def _run_decode_step_execution(
    *,
    temp_time_calc,
    gemm_results: Dict[str, Dict[str, float]],
    gemm_shapes: Dict[str, Tuple[int, ...]],
    batch_size: int,
    seq_len: int,
    total_seq_len: int,
) -> float:
    from time_calculation_LLM import ExecutionMode, LLMExecutionDispatcher

    layer_time, linear_time = temp_time_calc.compute_decode_layer_time(
        gemm_results=gemm_results,
        batch_size=batch_size,
        seq_len=seq_len,
        total_seq_len=total_seq_len,
    )

    node_breakdown = {
        "transformer_time_f": layer_time,
        "transformer_time_b": 0.0,
        "linear_softmax_f": linear_time,
        "linear_softmax_b": 0.0,
        "embedding_f": 0.0,
        "embedding_b": 0.0,
    }

    (
        pipeline_graph,
        pipeline_root,
        transformer_graph,
        transformer_forward_root,
        _,
        interconnect_params,
    ) = temp_time_calc._prepare_execution_graphs(
        node_breakdown=node_breakdown,
        gemm_results=gemm_results,
        batch_size=batch_size,
        seq_len=total_seq_len,
        hidden_dim=temp_time_calc.hidden_dim,
        num_heads=temp_time_calc.num_heads,
        ffn_dim=temp_time_calc.ffn_dim,
        vocab_size=temp_time_calc.vocab_size,
        include_pipeline_backward=False,
        include_transformer_backward=False,
    )

    dispatcher = LLMExecutionDispatcher(
        time_calc=temp_time_calc,
        pipeline_graph=pipeline_graph,
        pipeline_root=pipeline_root,
        interconnect_params=interconnect_params,
        transformer_graph=transformer_graph,
        transformer_forward_root=transformer_forward_root,
        transformer_backward_root=None,
    )

    execution_mode = getattr(temp_time_calc, "execution_mode", ExecutionMode.ANALYTICAL)
    result = dispatcher.run(execution_mode)
    return result.total_time



@dataclass
class InferenceConfig:
    """Configuration for inference simulation parameters."""
    batch_size: int
    seq_len: int  # prefill sequence length
    decode_len: int  # number of decode steps
    hidden_dim: int
    num_heads: int
    ffn_dim: int
    vocab_size: int
    num_layers: int
    dp: int = 1  # data parallel
    lp: int = 1  # layer parallel
    kp1: int = 1  # tensor parallel dim 1
    kp2: int = 1  # tensor parallel dim 2
    tp_mode: str = "row_col"
    kv_cache_enabled: bool = True
    # Decode sampling configuration
    sample_every: int = 32  # Sample every N decode steps
    force_sample_last: bool = True  # Always sample final step


@dataclass
class DecodeStep:
    """Represents a single decode step in autoregressive generation."""
    step_id: int
    generated_tokens: int  # number of newly generated tokens so far (1-indexed)
    total_seq_len: int  # prefill length + generated tokens
    gemm_shapes: Dict[str, Tuple[int, ...]]
    kv_cache_size: int


@dataclass
class DecodeSample:
    """Represents a sampled decode step with execution results."""
    step_id: int
    current_seq_len: int
    execution_time: float
    graph_root: Any
    kv_cache_tokens: int


class DecodeGraph(Graph):
    """
    Graph builder for autoregressive decode phase.

    Extends the base Graph class to handle step-by-step token generation
    with evolving sequence lengths and KV-cache considerations.
    """

    def __init__(self, config: InferenceConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.decode_steps: List[DecodeStep] = []
        self.hw_config = None
        self.model_config = None

    def _set_execution_context(self, hw_config, model_config):
        """Set hardware and model configs for proper execution integration."""
        self.hw_config = hw_config
        self.model_config = model_config

    def build_decode_graph(self) -> Tuple[float, List[DecodeSample]]:
        """
        Build decode phase using sample-based approach for efficiency.

        Instead of simulating every decode step, we sample at regular intervals
        and integrate between sample points using linear interpolation.

        Returns:
            Tuple of (total_decode_time, list_of_decode_samples)
        """
        # Generate sample points
        sample_points = self._generate_sample_points()

        # Execute graphs at sample points
        decode_samples = []
        for step_id in sample_points:
            generated_tokens = step_id + 1
            total_seq_len = self.config.seq_len + generated_tokens

            gemm_shapes = LLM_util.process_decode_gemm_shapes(
                batch_size=self.config.batch_size,
                current_seq_len=total_seq_len,
                d_model=self.config.hidden_dim,
                num_heads=self.config.num_heads,
                ffn_dim=self.config.ffn_dim,
                vocab_size=self.config.vocab_size,
                kv_cache_enabled=self.config.kv_cache_enabled,
                option="multiply_batch_into_m"
            )

            decode_step = DecodeStep(
                step_id=step_id,
                generated_tokens=generated_tokens,
                total_seq_len=total_seq_len,
                gemm_shapes=gemm_shapes,
                kv_cache_size=total_seq_len,  # KV cache bytes still approximated by token count; refine later
            )
            self.decode_steps.append(decode_step)

            sample_time = self._execute_decode_step_graph(
                step_id=step_id,
                total_seq_len=total_seq_len,
                gemm_shapes=gemm_shapes,
            )

            decode_samples.append(
                DecodeSample(
                    step_id=step_id,
                    current_seq_len=total_seq_len,
                    execution_time=sample_time,
                    graph_root=None,
                    kv_cache_tokens=total_seq_len,
                )
            )

        total_decode_time = self._integrate_decode_samples(decode_samples)

        return total_decode_time, decode_samples

    def _generate_sample_points(self) -> List[int]:
        """Generate decode step sample points based on sampling configuration."""
        sample_points = []

        # Always sample step 0 (first decode step)
        sample_points.append(0)

        # Sample at regular intervals
        for step_id in range(self.config.sample_every - 1, self.config.decode_len, self.config.sample_every):
            sample_points.append(step_id)

        # Always sample the last step if configured
        last_step = self.config.decode_len - 1
        if self.config.force_sample_last and last_step not in sample_points:
            sample_points.append(last_step)

        return sorted(sample_points)

    def _execute_decode_step_graph(
        self,
        *,
        step_id: int,
        total_seq_len: int,
        gemm_shapes: Dict[str, Tuple[int, ...]],
    ) -> float:
        """Execute decode step using appropriate DeepFlow execution mode."""
        from time_calculation_LLM import (
            ExecutionMode,
            Graph as DFGraph,
            LLMExecutionDispatcher,
            TimeCalculationLLM,
        )

        if not self.hw_config or not self.model_config:
            raise RuntimeError("Hardware config and model config are required for decode step execution.")

        temp_time_calc = TimeCalculationLLM(
            hw_config=self.hw_config,
            model_config=self.model_config,
            mode="LLM",
            output_dir="/tmp",
        )

        gemm_results = {}
        for name, shape in (
            ("qkv_proj", gemm_shapes["qkv_proj"]),
            ("attention_score", gemm_shapes["attention_score"]),
            ("attention_output", gemm_shapes["attention_output"]),
            ("output_proj", gemm_shapes["output_proj"]),
            ("ffn1", gemm_shapes["ffn1"]),
            ("ffn2", gemm_shapes["ffn2"]),
            ("linear", gemm_shapes["linear"]),
        ):
            gemm_results[name] = _measure_decode_gemm(
                temp_time_calc=temp_time_calc,
                name=name,
                gemm=shape,
            )

        return _run_decode_step_execution(
            temp_time_calc=temp_time_calc,
            gemm_results=gemm_results,
            gemm_shapes=gemm_shapes,
            batch_size=self.config.batch_size,
            seq_len=self.config.seq_len,
            total_seq_len=total_seq_len,
        )

    def _integrate_decode_samples(self, samples: List[DecodeSample]) -> float:
        """
        Integrate execution times between sample points using linear interpolation.

        Since attention cost grows linearly, we can use trapezoid rule integration
        to get accurate total time from sparse samples.
        """
        if not samples:
            raise ValueError("No decode samples available for integration")

        total_time = 0.0

        for idx, sample in enumerate(samples):
            if idx == 0:
                total_time += sample.execution_time
                continue

            prev_sample = samples[idx - 1]
            step_gap = sample.step_id - prev_sample.step_id
            if step_gap <= 0:
                raise ValueError("Sample points must be strictly increasing")

            midpoint = 0.5 * (prev_sample.execution_time + sample.execution_time)
            total_time += midpoint * step_gap

        last_sample = samples[-1]
        remaining_steps = self.config.decode_len - (last_sample.step_id + 1)
        if remaining_steps > 0:
            total_time += remaining_steps * last_sample.execution_time

        return total_time



class InferenceEngine:
    """
    Main orchestrator for complete LLM inference simulation.

    Manages the transition from prefill to decode phases and coordinates
    the overall inference execution using existing DeepFlow infrastructure.
    """

    def __init__(self, config: InferenceConfig, hw_config=None, model_config=None):
        self.config = config
        self.hw_config = hw_config
        self.model_config = model_config
        self.prefill_graph: Optional[Graph] = None
        self.decode_graph: Optional[DecodeGraph] = None


    def _build_decode_graph(self) -> Tuple[float, List[DecodeSample]]:
        """
        Build decode phase using sample-based approach with proper DeepFlow integration.

        Returns:
            Tuple of (total_decode_time, decode_samples)
        """
        self.decode_graph = DecodeGraph(
            config=self.config,
            mode="inference",
            dp=self.config.dp,
            lp=self.config.lp,
            kp1=self.config.kp1,
            kp2=self.config.kp2,
            tp_mode=self.config.tp_mode,
            comp_times={},
            comm_metadata={},
            misc_metadata={}
        )

        if hasattr(self.decode_graph, '_set_execution_context'):
            self.decode_graph._set_execution_context(self.hw_config, self.model_config)

        return self.decode_graph.build_decode_graph()


