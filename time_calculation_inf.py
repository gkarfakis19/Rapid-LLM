"""LLM inference prefill time-calculation entry points."""

from __future__ import annotations

import os

from time_calculation_LLM import LLMExecutionDispatcher, TimeCalculationLLM


class TimeCalculationLLMInference(TimeCalculationLLM):
    """Inference-specialized facade for ``TimeCalculationLLM``."""

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


__all__ = ["TimeCalculationLLMInference"]
