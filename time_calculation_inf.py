"""LLM inference prefill time-calculation entry points."""

from __future__ import annotations

import os

from time_calculation_LLM import LLMExecutionDispatcher, TimeCalculationLLM
from simulate_inf import InferenceEngine, InferenceConfig, DecodeGraph


class TimeCalculationLLMInference(TimeCalculationLLM):
    """Inference-specialized facade for ``TimeCalculationLLM``."""

    def __init__(self, hw_config, model_config, mode, output_dir: str | None = None):
        super().__init__(hw_config, model_config, mode, output_dir)

    def calc_time(self) -> float:
        batch_size = self._effective_transformer_batch()
        vocab_size = self.vocab_size
        hidden_dim = self.hidden_dim
        seq_len = self.seq_len
        num_heads = self.num_heads
        ffn_mult = self.ffn_mult
        ffn_dim = self.hidden_dim * ffn_mult if ffn_mult else self.ffn_dim

        self.readjust_type()

        gemm_results, node_breakdown = self.compute_all_gemm_and_node_times(
            batch_size,
            vocab_size,
            hidden_dim,
            seq_len,
            num_heads,
            ffn_dim,
        )

        (
            pipeline_graph,
            pipeline_root,
            transformer_graph,
            transformer_forward_root,
            _,
            interconnect_params,
        ) = self._prepare_execution_graphs(
            node_breakdown=node_breakdown,
            gemm_results=gemm_results,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            vocab_size=vocab_size,
            include_pipeline_backward=False,
            include_transformer_backward=False,
        )

        self.pipeline_graph = pipeline_graph
        self.pipeline_root = pipeline_root
        self.pipeline_interconnect = interconnect_params
        self.transformer_graph = transformer_graph
        self.transformer_forward_root = transformer_forward_root
        self.transformer_backward_root = None
        self.transformer_analytical_time_forward = node_breakdown.get("transformer_time_f")
        self.transformer_analytical_time_backward = None

        dispatcher = LLMExecutionDispatcher(
            time_calc=self,
            pipeline_graph=self.pipeline_graph,
            pipeline_root=self.pipeline_root,
            interconnect_params=self.pipeline_interconnect,
            transformer_graph=self.transformer_graph,
            transformer_forward_root=self.transformer_forward_root,
            transformer_backward_root=self.transformer_backward_root,
        )
        mode = self.execution_mode
        try:
            result = dispatcher.run(mode)
        except NotImplementedError as exc:
            raise NotImplementedError(
                f"{exc}. Selected execution mode '{mode.value}'."
            ) from exc

        self.pipeline_graph = dispatcher.pipeline_graph
        self.pipeline_root = result.graph_root
        self.pipeline_interconnect = dispatcher.interconnect_params

        total_time = result.total_time

        graph_folder = self.output_dir.rstrip(os.sep) + os.sep
        if self._generate_graphs and self.transformer_forward_root is not None:
            self.transformer_graph.save_graph(
                self.transformer_forward_root,
                graph_folder,
                "transformer_graph_forward_inference",
            )
        if self._generate_graphs and self.pipeline_root is not None:
            self.pipeline_graph.save_graph(
                self.pipeline_root,
                graph_folder,
                "pipeline_graph_inference",
            )

        return total_time

    def calc_decode_time(self) -> float:
        """
        Calculate autoregressive decode phase execution time.

        Calculate autoregressive decode phase execution time using sample-based approach.

        Returns:
            float: Total decode phase execution time
        """
        # Get inference sampling configuration
        inference_config_dict = getattr(getattr(self, "model", None), "inference", {})
        sample_every = inference_config_dict.get('sample_every', 32)
        force_sample_last = inference_config_dict.get('force_sample_last', True)

        # Create inference configuration from model parameters
        inference_config = InferenceConfig(
            batch_size=self._effective_transformer_batch(),
            seq_len=self.seq_len,
            decode_len=getattr(getattr(self, "model", None), "decode_len", 512),
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            ffn_dim=self.hidden_dim * self.ffn_mult if self.ffn_mult else self.ffn_dim,
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            dp=self.dp,
            lp=self.lp,
            kp1=self.kp1 if hasattr(self, 'kp1') else 1,
            kp2=self.kp2 if hasattr(self, 'kp2') else 1,
            tp_mode=self.t if hasattr(self, 't') else "row_col",
            kv_cache_enabled=inference_config_dict.get('kv_cache_enabled', True),
            sample_every=sample_every,
            force_sample_last=force_sample_last
        )


        # Create inference engine with proper hardware and model configs
        inference_engine = InferenceEngine(
            config=inference_config,
            hw_config=self.hw_config,
            model_config=self.model_config
        )

        # Build decode phase using sample-based approach with real DeepFlow integration
        decode_time, decode_samples = inference_engine._build_decode_graph()
        print(f"Decode simulation: {len(decode_samples)} samples for {inference_config.decode_len} steps (real timing enabled)")

        return decode_time

    def calc_total_inference_time(self) -> dict:
        """
        Calculate complete inference time including prefill + decode phases.

        Returns:
            dict: Breakdown of inference timing components
        """
        # Calculate prefill time (existing functionality)
        prefill_time = self.calc_time()

        # Calculate decode time (new functionality)
        decode_time = self.calc_decode_time()

        total_time = prefill_time + decode_time

        return {
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_inference_time": total_time
        }



__all__ = ["TimeCalculationLLMInference"]
