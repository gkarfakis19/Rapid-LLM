# Copyright 2026 NanoCad lab, UCLA
# https://nanocad.ee.ucla.edu/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LLM Inference Simulation - Extended Graph Generation for Prefill + Decode

This module extends RAPID-LLM's LLM training simulation to support full inference
workflows including both prefill and autoregressive decode phases.
"""

import math
import os
import shutil
from typing import Any, Callable, Dict, List, Tuple, Optional
from dataclasses import dataclass

from simulate_train_graph import Graph
import llm_util
from train_timing import LLMExecutionDispatcher

@dataclass
class InferenceConfig:
    """Configuration for inference simulation parameters."""
    batch_size: int
    seq_len: int  # prefill sequence length
    decode_len: int  # number of decode steps
    hidden_dim: int
    num_heads: int
    kv_heads: int
    intermediate_size: int
    vocab_size: int
    num_layers: int
    use_moe: bool 
    num_experts: int
    top_k: int

    
    moe_dp: int = 1  # inference expert pool expansion (used to size MoE routing group)
    pp: int = 1  # layer parallel
    tp: int = 1  # tensor parallel degree
    cp: int = 1  # context parallel degree
    tp_sp: bool = False  # sequence-parallel toggle
    # Decode sampling configuration
    sample_every: int = 32  # Sample every N decode steps

@dataclass
class DecodeSample:
    """Represents a sampled decode step with execution results."""
    step_id: int
    current_seq_len: int
    execution_time: float
    execution_energy: float
    execution_idle_time: float
    execution_idle_layer_time: float
    execution_idle_global_time: float
    graph_root: Any
    kv_cache_tokens: int


def _decode_sample_weights(step_ids: List[int], decode_len: int) -> List[float]:
    """Per-sample weights reproducing `_integrate_decode_samples`'s trapezoid rule.

    total = sum(w_i * x_i) equals seed + midpoint-segment + flat-tail
    integration for any per-sample scalar series x.
    """
    if not step_ids:
        return []
    weights = [0.0] * len(step_ids)
    weights[0] += 1.0
    for idx in range(1, len(step_ids)):
        gap = step_ids[idx] - step_ids[idx - 1]
        weights[idx - 1] += 0.5 * gap
        weights[idx] += 0.5 * gap
    weights[-1] += max(0, decode_len - (step_ids[-1] + 1))
    return weights


class DecodeGraph(Graph):
    """
    Graph builder for autoregressive decode phase.

    Extends the base Graph class to handle step-by-step token generation
    with evolving sequence lengths and KV-cache considerations.
    """

    def __init__(
        self,
        config: InferenceConfig,
        hw_config,
        model_config,
        time_calc_cls: Callable[..., Any],
        use_moe,
        num_experts,
        top_k,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.config = config
        self.hw_config = hw_config
        self.model_config = model_config
        self.time_calc_cls = time_calc_cls
        self._decode_sample_root: Optional[str] = None
        self.use_moe = config.use_moe
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        # Mirror MoE configuration expected by shared GEMM utilities
        self.moe_num_experts = int(getattr(config, "num_experts", 1) or 1)
        self.moe_top_k = int(getattr(config, "top_k", 1) or 1)
        model_cfg = getattr(model_config, "model_config", None)
        self.moe_intermediate_size = int(
            getattr(model_cfg, "moe_intermediate_size", getattr(config, "intermediate_size", 0))
        )
        self.n_shared_experts = int(getattr(model_cfg, "n_shared_experts", 0) or 0)
        self.head_dim = getattr(model_cfg, "head_dim", None)
        attention_cfg = getattr(model_cfg, "attention", None)
        self.attention_type = getattr(attention_cfg, "attention_type", "mha")
        self.kv_lora_rank = getattr(attention_cfg, "kv_lora_rank", None)
        self.q_lora_rank = getattr(attention_cfg, "q_lora_rank", None)
        self.qk_nope_head_dim = getattr(attention_cfg, "qk_nope_head_dim", None)
        self.qk_rope_head_dim = getattr(attention_cfg, "qk_rope_head_dim", None)
        self.v_head_dim = getattr(attention_cfg, "v_head_dim", None)
        self.run_type = str(getattr(model_cfg, "run_type", "inference")).lower()
        # Integrated per-device decode metrics (flattened mode only); consumed
        # by device_metrics.json via TimeCalculationLLMInference.calc_decode_time.
        self._per_device_totals: Optional[Dict[str, Any]] = None

    def build_decode_graph(self) -> Tuple[float, float, float, List[DecodeSample], float, float]:
        """
        Build decode phase using sample-based approach for efficiency.

        Instead of simulating every decode step, we sample at regular intervals
        and integrate between sample points using linear interpolation.

        Returns:
            Tuple of:
              (total_decode_time, total_decode_energy, total_decode_idle, decode_samples,
               total_decode_idle_layer, total_decode_idle_global)
        """
        # Determine decode steps we actually simulate
        sample_points = self._generate_sample_points()

        # Execute graphs at sample points
        decode_samples = []
        sample_device_records: List[Optional[Dict[str, Any]]] = []
        for step_id in sample_points:
            generated_tokens = step_id + 1
            total_seq_len = self.config.seq_len + generated_tokens

            gemm_shapes = llm_util.process_decode_gemm_shapes(
                self,
                batch_size=self.config.batch_size,
                current_seq_len=total_seq_len,
                d_model=self.config.hidden_dim,
                num_heads=self.config.num_heads,
                kv_heads=self.config.kv_heads,
                intermediate_size=self.moe_intermediate_size if self.use_moe else self.config.intermediate_size,
                vocab_size=self.config.vocab_size,
                model_type=self.model_config.model_config.model_type,
            )

            (
                sample_time,
                sample_energy,
                sample_idle_time,
                sample_idle_layer,
                sample_idle_global,
                sample_device_record,
            ) = self._execute_decode_step(
                step_id=step_id,
                total_seq_len=total_seq_len,
                gemm_shapes=gemm_shapes,
            )
            sample_device_records.append(sample_device_record)

            decode_samples.append(
                DecodeSample(
                    step_id=step_id,
                    current_seq_len=total_seq_len,
                    execution_time=sample_time,
                    execution_energy=sample_energy,
                    execution_idle_time=sample_idle_time,
                    execution_idle_layer_time=sample_idle_layer,
                    execution_idle_global_time=sample_idle_global,
                    graph_root=None,
                    kv_cache_tokens=total_seq_len,
                )
            )

        self._per_device_totals = self._integrate_per_device_records(
            decode_samples, sample_device_records
        )

        (
            total_decode_time,
            total_decode_energy,
            total_decode_idle,
            total_decode_idle_layer,
            total_decode_idle_global,
        ) = self._integrate_decode_samples(decode_samples)

        return (
            total_decode_time,
            total_decode_energy,
            total_decode_idle,
            decode_samples,
            total_decode_idle_layer,
            total_decode_idle_global,
        )

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
        if last_step not in sample_points:
            sample_points.append(last_step)

        return sorted(sample_points)

    def _execute_decode_step(
        self,
        *,
        step_id: int,
        total_seq_len: int,
        gemm_shapes: Dict[str, Tuple[int, ...]],
    ) -> Tuple[float, float, float, float, float, Optional[Dict[str, Any]]]:
        """Execute decode step using appropriate RAPID-LLM execution mode."""

        if not self.hw_config or not self.model_config:
            raise RuntimeError("Hardware config and model config are required for decode step execution.")
        if not self.time_calc_cls:
            raise RuntimeError("time_calc_cls must be provided for decode execution.")

        temp_time_calc = self.time_calc_cls(
            hw_config=self.hw_config,
            model_config=self.model_config,
            mode="LLM",
            output_dir="./output/LLM/",
        )

        base_dir = temp_time_calc.output_dir.rstrip(os.sep)
        sample_dir = None
        if self._debug_graphs_enabled():
            sample_dir = os.path.join(base_dir, "decode_samples", f"step_{step_id:04d}")
            root_dir = os.path.dirname(sample_dir)
            if self._decode_sample_root is None:
                self._decode_sample_root = root_dir
                if os.path.isdir(root_dir):
                    shutil.rmtree(root_dir, ignore_errors=True)
            os.makedirs(sample_dir, exist_ok=True)
            gemm_shapes_file = os.path.join(sample_dir, "decode_gemm_shapes.txt")
            with open(gemm_shapes_file, "w", encoding="utf-8") as f:
                for key, shape in gemm_shapes.items():
                    f.write(f"{key}: {shape}\n")
            prev_output_dir = temp_time_calc.output_dir
            temp_time_calc.output_dir = sample_dir

        execution_graphs, energy = temp_time_calc.prepare_decode_graphs(
            batch_size=self.config.batch_size,
            total_seq_len=total_seq_len,
            gemm_shapes=gemm_shapes,
        )
        (
            pipeline_graph,
            pipeline_root,
            _,
            _,
            transformer_graph,
            transformer_forward_root,
            transformer_backward_root,
            moe_transformer_graph,
            moe_transformer_forward_root,
            moe_transformer_backward_root,
            interconnect_params,
        ) = execution_graphs

        dispatcher = LLMExecutionDispatcher(
            time_calc=temp_time_calc,
            pipeline_graph=pipeline_graph,
            pipeline_root=pipeline_root,
            interconnect_params=interconnect_params,
            transformer_graph=transformer_graph,
            transformer_forward_root=transformer_forward_root,
            transformer_backward_root=transformer_backward_root,
            moe_transformer_graph=moe_transformer_graph,
            moe_transformer_forward_root=moe_transformer_forward_root,
            moe_transformer_backward_root=moe_transformer_backward_root,
        )

        result = dispatcher.run(temp_time_calc.execution_mode)
        # Fresh temp_time_calc counters start at zero (no reset needed); everything
        # recorded on this instance came from pricing this decode step.
        idle_time = temp_time_calc.get_idle_time_seconds()
        idle_breakdown = temp_time_calc.get_idle_breakdown_seconds()
        idle_layer_time = float(idle_breakdown.get("layer", 0.0))
        idle_global_time = float(idle_breakdown.get("global", 0.0))
        # Per-device record for device_metrics.json (flattened mode only). Each
        # per-sample temp instance carries its own flattened run record and, with
        # device profiles active, its own per-step re-pricing bank.
        device_record = getattr(temp_time_calc, "flattened_run_final", None)
        if device_record is not None:
            bank = temp_time_calc.get_device_profile_bank(build=False)
            if bank is not None:
                device_record = dict(device_record)
                device_record["profile_idle_layer_s"] = {
                    name: float(entry.idle_layer_s) for name, entry in bank.entries.items()
                }
                device_record["profile_idle_global_s"] = {
                    name: float(entry.idle_global_s) for name, entry in bank.entries.items()
                }
        if sample_dir:
            temp_time_calc.output_dir = prev_output_dir
            print(
                f"[decode] sample step {step_id}: seq_len={total_seq_len}, "
                f"time={result.total_time:.4f}s, artifacts={sample_dir}"
            )
        else:
            print(
                f"[decode] sample step {step_id}: seq_len={total_seq_len}, "
                f"time={result.total_time:.4f}s"
            )
        return result.total_time, energy, idle_time, idle_layer_time, idle_global_time, device_record

    def _integrate_per_device_records(
        self,
        samples: List[DecodeSample],
        records: List[Optional[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        """Integrate per-sample flattened device records over the decode phase.

        Uses the same trapezoid weights as `_integrate_decode_samples`, applied
        per rank (wall/busy) and per profile (kernel idle). Returns None unless
        every sampled step produced a flattened run record.
        """
        if not samples or len(samples) != len(records):
            return None
        if any(record is None for record in records):
            return None

        weights = _decode_sample_weights(
            [sample.step_id for sample in samples], self.config.decode_len
        )

        def _weighted_dict_sum(key: str) -> Dict[Any, float]:
            totals: Dict[Any, float] = {}
            for weight, record in zip(weights, records):
                for entry_key, value in (record.get(key) or {}).items():
                    totals[entry_key] = totals.get(entry_key, 0.0) + weight * float(value)
            return totals

        first = records[0]
        totals: Dict[str, Any] = {
            "total_raw_s": sum(
                weight * float(record.get("total_raw_s", 0.0))
                for weight, record in zip(weights, records)
            ),
            "per_rank_wall_s": _weighted_dict_sum("per_rank_wall_s"),
            "per_rank_busy_s": _weighted_dict_sum("per_rank_busy_s"),
            "rank_meta": dict(first.get("rank_meta") or {}),
            "rank_layout": dict(first.get("rank_layout") or {}),
            "layers_per_stage": list(first.get("layers_per_stage") or []),
            "lm_head_hw_id": first.get("lm_head_hw_id"),
            "effective_dp": int(first.get("effective_dp", 1) or 1),
            "hw_ids": list(first.get("hw_ids") or []),
            "interleave_scale": float(first.get("interleave_scale", 1.0) or 1.0),
        }
        if any("profile_idle_layer_s" in record for record in records):
            totals["profile_idle_layer_s"] = _weighted_dict_sum("profile_idle_layer_s")
            totals["profile_idle_global_s"] = _weighted_dict_sum("profile_idle_global_s")
        return totals

    def _integrate_decode_samples(self, samples: List[DecodeSample]) -> Tuple[float, float, float, float, float]:
        """
        Integrate execution times between sample points using linear interpolation.

        Since attention cost grows linearly, we can use trapezoid rule integration
        to get accurate total time from sparse samples.
        """
        if not samples:
            raise ValueError("No decode samples available for integration")

        total_time = 0.0
        total_energy = 0.0
        total_idle = 0.0
        total_idle_layer = 0.0
        total_idle_global = 0.0

        for idx, sample in enumerate(samples):
            if idx == 0:
                total_time += sample.execution_time
                total_energy += sample.execution_energy
                total_idle += sample.execution_idle_time
                total_idle_layer += sample.execution_idle_layer_time
                total_idle_global += sample.execution_idle_global_time
                if self._debug_graphs_enabled():
                    print(
                        f"[decode] integration seed step {sample.step_id:02d}: "
                        f"time={sample.execution_time:.4f}s, energy={sample.execution_energy:.4f}J"
                    )
                continue

            prev_sample = samples[idx - 1]
            step_gap = sample.step_id - prev_sample.step_id
            if step_gap <= 0:
                raise ValueError("Sample points must be strictly increasing")

            midpoint = 0.5 * (prev_sample.execution_time + sample.execution_time)
            segment_time = midpoint * step_gap
            total_time += segment_time

            midpoint_energy = 0.5 * (prev_sample.execution_energy + sample.execution_energy)
            segment_energy = midpoint_energy * step_gap
            total_energy += segment_energy

            midpoint_idle = 0.5 * (prev_sample.execution_idle_time + sample.execution_idle_time)
            segment_idle = midpoint_idle * step_gap
            total_idle += segment_idle
            midpoint_idle_layer = 0.5 * (
                prev_sample.execution_idle_layer_time + sample.execution_idle_layer_time
            )
            segment_idle_layer = midpoint_idle_layer * step_gap
            total_idle_layer += segment_idle_layer
            midpoint_idle_global = 0.5 * (
                prev_sample.execution_idle_global_time + sample.execution_idle_global_time
            )
            segment_idle_global = midpoint_idle_global * step_gap
            total_idle_global += segment_idle_global

            print(
                f"[decode] integration segment {prev_sample.step_id}->{sample.step_id}: "
                f"width={step_gap}, midpoint={midpoint:.4f}s, contribution={segment_time:.4f}s"
            )

        last_sample = samples[-1]
        remaining_steps = self.config.decode_len - (last_sample.step_id + 1)
        if remaining_steps > 0:
            tail_time = remaining_steps * last_sample.execution_time
            total_time += tail_time
            total_energy += remaining_steps * last_sample.execution_energy
            total_idle += remaining_steps * last_sample.execution_idle_time
            total_idle_layer += remaining_steps * last_sample.execution_idle_layer_time
            total_idle_global += remaining_steps * last_sample.execution_idle_global_time
            print(
                f"[decode] integration tail from {last_sample.step_id} covering {remaining_steps} steps: "
                f"contribution={tail_time:.4f}s"
            )

        print(
            f"[decode] total interpolated decode time: {total_time:.4f}s, energy: {total_energy:.4f}J "
            f"from {len(samples)} samples"
        )

        return total_time, total_energy, total_idle, total_idle_layer, total_idle_global

    def _debug_graphs_enabled(self) -> bool:
        flag = os.environ.get("RAPID_VISUALIZE_GRAPHS")
        if flag is None:
            return False
        return flag.strip().lower() not in {"", "0", "false", "no"}



class InferenceEngine:
    """
    Main orchestrator for complete LLM inference simulation.

    Manages the transition from prefill to decode phases and coordinates
    the overall inference execution using existing RAPID-LLM infrastructure.
    """

    def __init__(
        self,
        config: InferenceConfig,
        hw_config=None,
        model_config=None,
        *,
        time_calc_cls: Optional[Callable[..., Any]] = None,
    ):
        self.config = config
        self.hw_config = hw_config
        self.model_config = model_config
        self.prefill_graph: Optional[Graph] = None
        self.decode_graph: Optional[DecodeGraph] = None
        self.time_calc_cls = time_calc_cls


    def _build_decode_graph(self) -> Tuple[float, float, float, List[DecodeSample], float, float]:
        """
        Build decode phase using sample-based approach with proper RAPID-LLM integration.

        Returns:
            Tuple of:
              (total_decode_time, total_decode_energy, total_decode_idle, decode_samples,
               total_decode_idle_layer, total_decode_idle_global)
        """
        if self.time_calc_cls is None:
            raise RuntimeError("InferenceEngine requires time_calc_cls for decode graph building.")

        self.decode_graph = DecodeGraph(
            config=self.config,
            mode="inference",
            dp=1,
            pp=self.config.pp,
            tp=self.config.tp,
            cp=self.config.cp,
            ep=self.config.moe_dp,
            comp_times={},
            comm_metadata={},
            misc_metadata={
                "sequence_parallel": self.config.tp_sp,
            },
            hw_config=self.hw_config,
            model_config=self.model_config,
            time_calc_cls=self.time_calc_cls,
            use_moe=self.config.use_moe,
            num_experts=self.config.num_experts,
            top_k=self.config.top_k,
        )

        return self.decode_graph.build_decode_graph()
