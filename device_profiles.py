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

"""Per-device hardware throttle profiles (heterogeneous execution support).

This module implements the re-pricing bank and flattened-graph injection for
the `device_profiles:` hardware-config feature:

* :func:`load_device_profiles_yaml` — load the standalone ``--device_profiles``
  override file (same schema as the hardware-YAML block).
* :func:`build_timing_bank` — for each distinct profile, deepcopy the owning
  ``TimeCalculation``'s hardware config, apply the six throttle scales, build a
  fresh timing instance, and re-price the owner's op set through the SAME
  pricing entry point the owner used (``compute_all_gemm_and_node_times`` for
  training/prefill, ``_build_decode_transformer_results`` for decode). Per-op
  durations therefore go through the full roofline/tiling model under each
  profile's parameters — no flat duration multipliers.
* :func:`apply_profiles_to_flattened_root` — rewrite the durations of the
  flattened AstraSim graph per device: multiplicative per-op ratios for
  pure-compute GEMM nodes; additive compute-delta for embedding/linear_softmax
  stage nodes (their durations embed analytic comm time, which must not
  scale); re-priced optimizer deltas for optimizer nodes.
* :func:`write_device_metrics_json` — emit ``device_metrics.json`` with
  per-device schedule/kernel idle metrics for every flattened-mode run.

Known modeling choice (documented): each profile re-runs tile/kernel selection
under its own scaled parameters, i.e. the model picks its best kernel at that
operating point. Real DVFS does not re-tile mid-run; both are approximations
within the model's fidelity, and this choice keeps the uniform-profile
equivalence exact (identical code path to a globally scaled config).
"""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import yaml

import simulate_train_graph as llm_simulation
from config import (
    DEVICE_PROFILE_SCALE_FIELDS,
    DeviceProfileError,
    DeviceProfileSpec,
    DeviceProfilesConfig,
    parse_device_profiles,
)
from util import log_message

# Execution-mode value string for the only mode that supports device profiles.
FLATTENED_MODE_VALUE = "full_astrasim_flattened"

# Stage-level ops whose flattened node durations embed analytic comm time
# (train_timing node_breakdown uses OperationTiming.total_*_time). Profiles
# must not scale the comm part, so these are injected additively:
#   new = base + (compute_profile - compute_baseline)
STAGE_ADDITIVE_OPS = ("embedding", "linear_softmax")

# Optimizer nodes are HBM-bandwidth-bound and re-priced through
# get_data_parallel_reduction_llm per profile (a transformer-aggregate ratio
# would be systematically wrong for them).
OPTIMIZER_OP = "optimizer"

DEVICE_METRICS_FILENAME = "device_metrics.json"


def load_device_profiles_yaml(path: str) -> DeviceProfilesConfig:
    """Load a standalone device-profiles YAML (CLI ``--device_profiles`` override)."""
    expanded = os.path.expandvars(os.path.expanduser(str(path)))
    try:
        with open(expanded, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except OSError as exc:
        raise DeviceProfileError(f"Cannot read --device_profiles file '{path}': {exc}") from exc
    except yaml.YAMLError as exc:
        raise DeviceProfileError(f"--device_profiles file '{path}' is not valid YAML: {exc}") from exc
    if raw is None:
        raise DeviceProfileError(f"--device_profiles file '{path}' is empty")
    if not isinstance(raw, dict):
        raise DeviceProfileError(
            f"--device_profiles file '{path}' must contain a mapping with `profiles:` and `devices:`"
        )
    # Accept either the bare schema or a file wrapping it in `device_profiles:`.
    if "device_profiles" in raw and "profiles" not in raw:
        raw = raw["device_profiles"]
    parsed = parse_device_profiles(raw)
    if parsed is None:
        raise DeviceProfileError(f"--device_profiles file '{path}' defines no profiles")
    return parsed


# ---------------------------------------------------------------------------
# Re-pricing bank
# ---------------------------------------------------------------------------

@dataclass
class ProfileTimingEntry:
    """Re-priced timings for one profile (or the owner's baseline pricing)."""

    name: str
    spec: Optional[DeviceProfileSpec]
    # (op_name, direction) -> (compute_time_s, comm_time_s)
    ops: Dict[Tuple[str, str], Tuple[float, float]]
    optimizer_time_s: Optional[float]
    idle_layer_s: float
    idle_global_s: float


class HeterogeneousTimingBank:
    """Per-profile re-priced op durations + kernel-idle counters."""

    def __init__(
        self,
        profiles_cfg: DeviceProfilesConfig,
        baseline: ProfileTimingEntry,
        entries: Dict[str, ProfileTimingEntry],
        pricing_kind: str,
    ) -> None:
        self.profiles_cfg = profiles_cfg
        self.baseline = baseline
        self.entries = entries
        self.pricing_kind = pricing_kind

    def entry_for(self, profile_name: str) -> ProfileTimingEntry:
        try:
            return self.entries[profile_name]
        except KeyError as exc:
            raise DeviceProfileError(
                f"internal: timing bank has no entry for profile '{profile_name}' "
                f"(known: {sorted(self.entries)})"
            ) from exc


def _op_table_from_timings(timings: Mapping[str, Any]) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """Build the (op, direction) -> (compute, comm) table mirroring the graph entries.

    Every OperationTiming in ``timings`` is included directly, plus the "MLP"
    group (ffn1 + gelu + ffn2) exactly as ``_prepare_execution_graphs`` builds
    its transformer entries.
    """
    table: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for name, op in timings.items():
        if op is None:
            continue
        forward = getattr(op, "forward", None)
        if forward is not None:
            table[(str(name), "forward")] = (float(forward.compute_time), float(forward.comm_time))
        backward = getattr(op, "backward", None)
        if backward is not None:
            table[(str(name), "backward")] = (float(backward.compute_time), float(backward.comm_time))
    members = [timings.get(member) for member in ("ffn1", "gelu", "ffn2")]
    if all(member is not None for member in members):
        if all(getattr(member, "forward", None) is not None for member in members):
            table[("MLP", "forward")] = (
                sum(float(member.forward.compute_time) for member in members),
                sum(float(member.forward.comm_time) for member in members),
            )
        if all(getattr(member, "backward", None) is not None for member in members):
            table[("MLP", "backward")] = (
                sum(float(member.backward.compute_time) for member in members),
                sum(float(member.backward.comm_time) for member in members),
            )
    return table


def apply_profile_scales_to_hw_config(hw_config: Any, spec: DeviceProfileSpec) -> None:
    """Multiply the six throttle scales into a (deepcopied) hardware config."""
    tech = hw_config.tech_config
    core = tech.core
    if getattr(core, "operating_frequency", None) is None:
        raise DeviceProfileError(
            "device_profiles require tech_param.core.operating_frequency to be set "
            "explicitly; the voltage-derived nominal_frequency path is not supported."
        )
    core.operating_frequency = float(core.operating_frequency) * spec.frequency_scale
    dram = tech.DRAM
    if getattr(dram, "bandwidth", None) is None:
        raise DeviceProfileError(
            "device_profiles require tech_param.DRAM.bandwidth to be set explicitly; "
            "link-derived DRAM bandwidth is not supported."
        )
    dram.bandwidth = float(dram.bandwidth) * spec.hbm_bandwidth_scale
    dram.latency = float(dram.latency) * spec.hbm_latency_scale
    tech.SRAML2.bandwidth = float(tech.SRAML2.bandwidth) * spec.l2_bandwidth_scale
    tech.SRAML1.bandwidth = float(tech.SRAML1.bandwidth) * spec.l1_bandwidth_scale
    tech.SRAMR.bandwidth = float(tech.SRAMR.bandwidth) * spec.register_bandwidth_scale


def _price_profile_variant(owner: Any, context: Dict[str, Any], spec: DeviceProfileSpec) -> ProfileTimingEntry:
    """Re-price the owner's op set under one profile via a fresh timing instance."""
    hw_k = copy.deepcopy(owner.hw_config)
    # A6: profile-variant instances must never construct banks or dispatch.
    hw_k.device_profiles = None
    apply_profile_scales_to_hw_config(hw_k, spec)

    variant_dir = os.path.join(owner.output_dir, f"profile_{spec.name}")
    tc_k = type(owner)(hw_k, owner._raw_model_config, owner.mode, output_dir=variant_dir)
    if getattr(tc_k.hw_config, "device_profiles", None) is not None:
        raise DeviceProfileError("internal: profile variant instance still carries device_profiles")

    kind = context["kind"]
    if kind in ("training", "prefill"):
        timings, _ = tc_k.compute_all_gemm_and_node_times(
            *context["compute_all_args"], use_moe_override=False
        )
    elif kind == "decode":
        decode_args = context["decode_args"]
        # A4: decode re-prices through decode's own pricing path with this
        # sample step's total_seq_len (K-times-per-sampled-step cost is
        # accepted; `inference_param.sample_every` controls it).
        timings, _ = tc_k._build_decode_transformer_results(
            batch_size=decode_args["batch_size"],
            total_seq_len=decode_args["total_seq_len"],
            use_moe_layer=False,
            gemm_shapes=None,
        )
    else:
        raise DeviceProfileError(f"internal: unknown device-profile pricing kind '{kind}'")

    optimizer_args = context.get("optimizer_args")
    optimizer_time = (
        float(tc_k.get_data_parallel_reduction_llm(*optimizer_args))
        if optimizer_args is not None
        else None
    )
    idle = tc_k.get_idle_breakdown_seconds()
    return ProfileTimingEntry(
        name=spec.name,
        spec=spec,
        ops=_op_table_from_timings(timings),
        optimizer_time_s=optimizer_time,
        idle_layer_s=float(idle.get("layer", 0.0)),
        idle_global_s=float(idle.get("global", 0.0)),
    )


def build_timing_bank(owner: Any) -> HeterogeneousTimingBank:
    """Build the per-profile re-pricing bank for ``owner`` (a TimeCalculationLLM).

    The owner must carry ``hw_config.device_profiles`` and a
    ``_device_profile_pricing_context`` recorded at pricing time (see
    train_timing/inference_timing). Identical scale tuples are de-duplicated;
    identity (all-1.0) profiles reuse the owner's baseline pricing exactly.
    """
    profiles_cfg = getattr(owner.hw_config, "device_profiles", None)
    if not profiles_cfg:
        raise DeviceProfileError("internal: timing bank requested without device_profiles")
    context = getattr(owner, "_device_profile_pricing_context", None)
    if not context:
        raise DeviceProfileError(
            "internal: device-profile pricing context missing — op pricing has not run yet"
        )

    baseline_ops = _op_table_from_timings(context["timings"])
    optimizer_args = context.get("optimizer_args")
    baseline_optimizer = (
        float(owner.get_data_parallel_reduction_llm(*optimizer_args))
        if optimizer_args is not None
        else None
    )
    baseline_idle = owner.get_idle_breakdown_seconds()
    baseline = ProfileTimingEntry(
        name="__baseline__",
        spec=None,
        ops=baseline_ops,
        optimizer_time_s=baseline_optimizer,
        idle_layer_s=float(baseline_idle.get("layer", 0.0)),
        idle_global_s=float(baseline_idle.get("global", 0.0)),
    )

    entries: Dict[str, ProfileTimingEntry] = {}
    by_scale_tuple: Dict[Tuple[float, ...], ProfileTimingEntry] = {}
    for name, spec in profiles_cfg.profiles.items():
        scale_tuple = spec.scale_tuple()
        if spec.is_identity():
            # Identity profiles reuse the baseline pricing exactly — keeps the
            # uniform all-1.0 equivalence bit-exact and avoids K re-pricings.
            entries[name] = ProfileTimingEntry(
                name=name,
                spec=spec,
                ops=baseline.ops,
                optimizer_time_s=baseline.optimizer_time_s,
                idle_layer_s=baseline.idle_layer_s,
                idle_global_s=baseline.idle_global_s,
            )
            continue
        cached = by_scale_tuple.get(scale_tuple)
        if cached is not None:
            entries[name] = ProfileTimingEntry(
                name=name,
                spec=spec,
                ops=cached.ops,
                optimizer_time_s=cached.optimizer_time_s,
                idle_layer_s=cached.idle_layer_s,
                idle_global_s=cached.idle_global_s,
            )
            continue
        entry = _price_profile_variant(owner, context, spec)
        by_scale_tuple[scale_tuple] = entry
        entries[name] = entry

    return HeterogeneousTimingBank(
        profiles_cfg=profiles_cfg,
        baseline=baseline,
        entries=entries,
        pricing_kind=str(context["kind"]),
    )


# ---------------------------------------------------------------------------
# Flattened-graph injection
# ---------------------------------------------------------------------------

def _iter_graph_nodes(root: Any) -> Iterable[Any]:
    """Yield every simulate_train_graph.Node reachable from ``root`` once."""
    stack: List[Any] = list(root) if isinstance(root, (list, tuple)) else [root]
    visited: Set[int] = set()
    while stack:
        obj = stack.pop()
        obj_id = id(obj)
        if obj_id in visited:
            continue
        visited.add(obj_id)
        if isinstance(obj, llm_simulation.Node):
            yield obj
        for child in getattr(obj, "children", []) or []:
            stack.append(child)


def _priced_duration(
    base: float,
    op_key: Tuple[str, str],
    entry: ProfileTimingEntry,
    baseline: ProfileTimingEntry,
) -> float:
    op_name = op_key[0]
    if op_name == OPTIMIZER_OP:
        # Re-priced optimizer duration delta (HBM-BW-bound; A3).
        if entry.optimizer_time_s is None or baseline.optimizer_time_s is None:
            raise DeviceProfileError(
                "internal: optimizer re-pricing missing from timing bank "
                f"(pricing kind lacks optimizer args; op_key={op_key!r})"
            )
        return base + (entry.optimizer_time_s - baseline.optimizer_time_s)

    profile_op = entry.ops.get(op_key)
    baseline_op = baseline.ops.get(op_key)
    if profile_op is None or baseline_op is None:
        raise DeviceProfileError(
            f"op_key {op_key!r} does not resolve in the device-profile timing bank "
            f"(known ops: {sorted({k[0] for k in baseline.ops})}). "
            "This indicates the bank was priced through a different pricing path "
            "than the graph being injected."
        )
    if op_name in STAGE_ADDITIVE_OPS:
        # Stage-level node durations embed analytic comm time; only the compute
        # component responds to the profile (A3): base + (compute_k - compute_base).
        return base + (profile_op[0] - baseline_op[0])

    # Pure-compute per-GEMM node: multiplicative per-op ratio. This is
    # transform-safe (overlap splits scale head and tail identically) and
    # includes the kernel-launch overhead correctly because both numerator and
    # denominator are whole-op compute times.
    base_compute = baseline_op[0]
    if base_compute <= 0.0:
        return base
    return base * (profile_op[0] / base_compute)


def _log_injection_summary(
    profiles_cfg: DeviceProfilesConfig,
    profile_grid: Dict[Tuple[int, int], str],
    hw_ids: Sequence[int],
    effective_dp: int,
    scaled_nodes: int,
    run_label: str,
) -> None:
    distinct = sorted({name for name in profile_grid.values()})
    label_suffix = f" [{run_label}]" if run_label else ""
    log_message(
        f"[RAPID-LLM][profiles] Device throttle profiles active{label_suffix}: "
        f"{len(distinct)} distinct profile(s) across {len(hw_ids)} device(s) x dp={effective_dp}; "
        f"{scaled_nodes} compute nodes re-priced.",
        category="profiles",
    )
    header = (
        f"  {'hw_id':>5} {'dp':>3} | {'profile':<12} | "
        + " ".join(f"{field.split('_scale')[0]:>9}" for field in DEVICE_PROFILE_SCALE_FIELDS)
    )
    log_message(header, category="profiles")
    log_message("  " + "-" * (len(header) - 2), category="profiles")
    for hw_id in sorted(hw_ids):
        for dp_idx in range(effective_dp):
            name = profile_grid[(int(hw_id), dp_idx)]
            spec = profiles_cfg.profiles[name]
            scales = " ".join(f"{value:>9.4f}" for value in spec.scale_tuple())
            log_message(
                f"  {int(hw_id):>5} {dp_idx:>3} | {name:<12} | {scales}",
                category="profiles",
            )


def apply_profiles_to_flattened_root(
    root: Any,
    *,
    bank: HeterogeneousTimingBank,
    profiles_cfg: DeviceProfilesConfig,
    unique_hw_ids: Iterable[int],
    effective_dp: int,
    run_label: str = "",
) -> Dict[str, Any]:
    """Rewrite flattened-graph compute durations per device profile.

    Must be called AFTER ``apply_overlap_transforms`` (the TP-overlap split
    collapses tuple durations) and BEFORE ``run_astra_simulation_only_onepath``.
    For ``effective_dp == 1`` durations stay scalar; for dp > 1 each node gets a
    per-dp duration tuple consumed by the executor's ``duration_profile`` path.
    """
    hw_set = {int(h) for h in unique_hw_ids}
    if not hw_set:
        raise DeviceProfileError("internal: flattened graph exposes no hardware ids")
    effective_dp = max(1, int(effective_dp))

    extra_devices = sorted(set(profiles_cfg.devices) - hw_set)
    if extra_devices:
        raise DeviceProfileError(
            f"device_profiles.devices lists hw_id(s) {extra_devices} that do not exist in "
            f"the flattened graph (flattened device set: {sorted(hw_set)})."
        )
    bad_dp_keys = sorted(
        key for key in profiles_cfg.dp_devices
        if key[0] not in hw_set or key[1] >= effective_dp
    )
    if bad_dp_keys:
        raise DeviceProfileError(
            f"device_profiles.dp_devices entries {bad_dp_keys} do not exist in this run "
            f"(flattened device set: {sorted(hw_set)}, dp={effective_dp})."
        )

    profile_grid: Dict[Tuple[int, int], str] = {}
    for hw_id in sorted(hw_set):
        for dp_idx in range(effective_dp):
            profile_grid[(hw_id, dp_idx)] = profiles_cfg.resolve(hw_id, dp_idx)

    scaled_nodes = 0
    for node in _iter_graph_nodes(root):
        hw_id = getattr(node, "hw_id", None)
        if hw_id is None or hw_id < 0:
            continue
        if getattr(node, "flatten_placeholder", False):
            continue
        if node.duration_profile is not None:
            raise DeviceProfileError(
                f"internal: node '{getattr(node, 'name', '<unnamed>')}' already carries a "
                "duration tuple before device-profile injection"
            )
        base = float(node.duration or 0.0)
        if base <= 0.0:
            # Zero-duration no-ops (disabled embedding/unembedding, placeholder
            # anchors) are whitelisted keyless.
            continue
        op_key = getattr(node, "op_key", None)
        if op_key is None:
            raise DeviceProfileError(
                f"internal: compute node '{getattr(node, 'name', '<unnamed>')}' "
                f"(hw_id={hw_id}, duration={base:.3e}s) carries no op_key; the flattener "
                "must stamp op identity on every nonzero-duration compute node."
            )
        values: List[float] = []
        for dp_idx in range(effective_dp):
            entry = bank.entry_for(profile_grid[(int(hw_id), dp_idx)])
            values.append(_priced_duration(base, tuple(op_key), entry, bank.baseline))
        if effective_dp == 1:
            node.duration = values[0]
        else:
            node.duration = tuple(values)
        scaled_nodes += 1

    _log_injection_summary(
        profiles_cfg, profile_grid, sorted(hw_set), effective_dp, scaled_nodes, run_label
    )
    return {
        "profile_grid": profile_grid,
        "scaled_nodes": scaled_nodes,
        "effective_dp": effective_dp,
        "hw_ids": sorted(hw_set),
    }


# ---------------------------------------------------------------------------
# device_metrics.json
# ---------------------------------------------------------------------------

# Warn once per process when the rank-layout decode fails: every device in the
# run hits the identical failure, so per-device repeats would only flood the log.
_layout_decode_warning_emitted = False


def _device_coords(hw_id: int, dp_idx: int, rank_layout: Mapping[str, Any]) -> Optional[Dict[str, int]]:
    """Decode per-device axis coordinates from the flattened rank layout.

    Returns None when the layout cannot be decoded. Consumers must treat the
    placement as UNKNOWN (the JSON carries null coords/layers_hosted) — never
    silently default the device to pipeline stage 0.
    """
    global _layout_decode_warning_emitted
    try:
        from astrasim_lib.layout_utils import axis_layout_from_descriptor, decode_axis_coordinates

        layout = axis_layout_from_descriptor(rank_layout)
        coords = {axis: int(value) for axis, value in decode_axis_coordinates(int(hw_id), layout)}
    except Exception as exc:
        if not _layout_decode_warning_emitted:
            _layout_decode_warning_emitted = True
            log_message(
                "[device-metrics] WARNING: failed to decode device axis coordinates "
                f"from the flattened rank layout ({exc!r}). device_metrics.json will "
                "carry null coords/layers_hosted/kernel_idle_layer_s/"
                "kernel_idle_frac_thermal so consumers can distinguish 'placement "
                "unknown' from 'stage 0'.",
                category="results",
            )
        return None
    coords["dp"] = int(dp_idx)
    return coords


def write_device_metrics_json(
    time_calc: Any,
    exp_dir: str,
    *,
    run_type: str,
    inference_metrics: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Write ``device_metrics.json`` for a flattened-mode run (profiles or not).

    Training: GA-combined metrics over the no_dp and final runs (A5); each run
    also appears in a ``runs`` sub-array. Inference: prefill + trapezoid-
    integrated decode. ``makespan_s`` is the RAW AstraSim total (pre
    ``_pipeline_interleave_scale``); ``pipeline_interleave_scale`` is recorded
    so consumers can reconcile with the reported Total Time (A7).

    Returns the written path, or None when the run was not flattened-mode.
    """
    mode_value = getattr(getattr(time_calc, "execution_mode", None), "value", None)
    if mode_value != FLATTENED_MODE_VALUE:
        return None

    run_type = str(run_type or "training").lower()
    final_record = getattr(time_calc, "flattened_run_final", None)
    runs: List[Tuple[str, float, Mapping[str, Any]]] = []
    if run_type == "training":
        ga_steps = int(getattr(time_calc, "gradient_accumulation_steps", 1) or 1)
        no_dp_record = getattr(time_calc, "flattened_run_no_dp", None)
        if ga_steps > 1 and no_dp_record:
            runs.append(("no_dp", float(ga_steps - 1), no_dp_record))
        if final_record:
            runs.append(("final", 1.0, final_record))
    else:
        ga_steps = 1
        if final_record:
            runs.append(("prefill", 1.0, final_record))
        decode_totals = getattr(time_calc, "_decode_per_device_totals", None)
        if decode_totals:
            runs.append(("decode", 1.0, decode_totals))
    if not runs:
        return None

    # Static layout info: prefer a record that carries rank metadata.
    reference = next((rec for _, _, rec in runs if rec.get("rank_meta")), runs[0][2])
    rank_meta: Dict[int, Mapping[str, Any]] = {
        int(rank): meta for _, _, rec in runs for rank, meta in (rec.get("rank_meta") or {}).items()
    }
    rank_layout = reference.get("rank_layout") or {}
    layers_per_stage = list(reference.get("layers_per_stage") or [])
    lm_head_hw_id = reference.get("lm_head_hw_id")
    effective_dp = int(reference.get("effective_dp", 1) or 1)
    interleave_scale = float(reference.get("interleave_scale", 1.0) or 1.0)

    makespan_s = sum(weight * float(rec.get("total_raw_s", 0.0)) for _, weight, rec in runs)

    def _combined(rank: int, key: str) -> float:
        return sum(
            weight * float((rec.get(key) or {}).get(rank, 0.0)) for _, weight, rec in runs
        )

    profiles_cfg = getattr(getattr(time_calc, "hw_config", None), "device_profiles", None)
    bank = None
    if profiles_cfg is not None:
        bank = time_calc.get_device_profile_bank(build=False)

    # Per-profile (or baseline) kernel-idle counters. These are pricing-time
    # counters: recorded once per layer TYPE per pricing pass, independent of
    # gradient-accumulation/microbatch counts (same semantics as the results-txt
    # idle lines). Per device they scale by layers_hosted (+ the global bucket
    # on the lm-head device only, A8/A10).
    def _kernel_idle_components(profile_name: Optional[str]) -> Tuple[float, float]:
        if profiles_cfg is not None:
            layer_s = 0.0
            global_s = 0.0
            if run_type == "training":
                if bank is not None and profile_name is not None:
                    entry = bank.entry_for(profile_name)
                    layer_s, global_s = entry.idle_layer_s, entry.idle_global_s
            else:
                # prefill per-profile idle (if prefill ran) + integrated decode
                # per-profile idle. The prefill term comes from the snapshot
                # taken right after the prefill dispatcher run — the live bank
                # on the instance is cleared by prepare_decode_graphs when the
                # decode-shaped memory-estimation graph records its own pricing
                # context, so reading it here would silently drop prefill.
                prefill_idle = getattr(time_calc, "_prefill_profile_idle", None) or {}
                decode_totals = getattr(time_calc, "_decode_per_device_totals", None) or {}
                if profile_name is not None:
                    layer_s += float((prefill_idle.get("layer") or {}).get(profile_name, 0.0))
                    global_s += float((prefill_idle.get("global") or {}).get(profile_name, 0.0))
                    layer_s += float((decode_totals.get("profile_idle_layer_s") or {}).get(profile_name, 0.0))
                    global_s += float((decode_totals.get("profile_idle_global_s") or {}).get(profile_name, 0.0))
            return layer_s, global_s
        if run_type == "training":
            breakdown = time_calc.get_idle_breakdown_seconds()
            return float(breakdown.get("layer", 0.0)), float(breakdown.get("global", 0.0))
        metrics = inference_metrics or {}
        layer_s = float(getattr(time_calc, "_prefill_idle_layer_time_s", 0.0)) + float(
            metrics.get("decode_idle_layer_time", 0.0)
        )
        global_s = float(getattr(time_calc, "_prefill_idle_global_time_s", 0.0)) + float(
            metrics.get("decode_idle_global_time", 0.0)
        )
        return layer_s, global_s

    devices: List[Dict[str, Any]] = []
    for rank in sorted(rank_meta):
        meta = rank_meta[rank]
        hw_id = int(meta.get("stage", rank))
        dp_idx = int(meta.get("dp", 0))
        coords = _device_coords(hw_id, dp_idx, rank_layout)
        if coords is None:
            # Layout decode failed: placement is unknown. Emit nulls rather than
            # guessing stage 0 (which would assign stage-0 layer counts — and
            # therefore wrong kernel idle — to every device).
            layers_hosted = None
        else:
            pp_coord = int(coords.get("pp", 0))
            if layers_per_stage and 0 <= pp_coord < len(layers_per_stage):
                layers_hosted = int(layers_per_stage[pp_coord])
            else:
                layers_hosted = 0
        hosts_lm_head = lm_head_hw_id is not None and hw_id == int(lm_head_hw_id)
        profile_name = profiles_cfg.resolve(hw_id, dp_idx) if profiles_cfg is not None else None
        busy_s = _combined(rank, "per_rank_busy_s")
        wall_s = _combined(rank, "per_rank_wall_s")
        if makespan_s > 0.0:
            sched_idle_frac = min(1.0, max(0.0, 1.0 - busy_s / makespan_s))
        else:
            sched_idle_frac = 0.0
        idle_layer_unit_s, idle_global_unit_s = _kernel_idle_components(profile_name)
        kernel_idle_global_s = idle_global_unit_s if hosts_lm_head else 0.0
        if layers_hosted is None:
            kernel_idle_layer_s = None
            kernel_idle_frac_thermal = None
        else:
            kernel_idle_layer_s = idle_layer_unit_s * layers_hosted
            kernel_idle_frac_thermal = (
                (kernel_idle_layer_s + kernel_idle_global_s) / makespan_s
                if makespan_s > 0.0
                else 0.0
            )
        devices.append(
            {
                "rank": int(rank),
                "hw_id": hw_id,
                "dp_idx": dp_idx,
                "coords": coords,
                "profile": profile_name,
                "compute_busy_s": busy_s,
                "wall_time_s": wall_s,
                "sched_idle_frac": sched_idle_frac,
                "layers_hosted": layers_hosted,
                "hosts_lm_head": bool(hosts_lm_head),
                "kernel_idle_layer_s": kernel_idle_layer_s,
                "kernel_idle_global_s": kernel_idle_global_s,
                "kernel_idle_frac_thermal": kernel_idle_frac_thermal,
            }
        )

    runs_payload = [
        {
            "name": name,
            "weight": weight,
            "total_raw_s": float(rec.get("total_raw_s", 0.0)),
            "devices": [
                {
                    "rank": int(rank),
                    "compute_busy_s": float((rec.get("per_rank_busy_s") or {}).get(rank, 0.0)),
                    "wall_time_s": float((rec.get("per_rank_wall_s") or {}).get(rank, 0.0)),
                }
                for rank in sorted(rank_meta)
            ],
        }
        for name, weight, rec in runs
    ]

    payload = {
        "schema_version": 1,
        "execution_mode": FLATTENED_MODE_VALUE,
        "run_type": run_type,
        "dp_count": effective_dp,
        "num_devices": len(devices),
        "gradient_accumulation_steps": ga_steps,
        "pipeline_interleave_scale": interleave_scale,
        "makespan_s": makespan_s,
        "total_time_s": makespan_s * interleave_scale,
        "profiles": profiles_cfg.profiles_as_dict() if profiles_cfg is not None else {},
        "runs": runs_payload,
        "devices": devices,
        "notes": (
            "makespan_s is the raw AstraSim total (pre pipeline_interleave_scale), "
            "GA-combined for training and prefill+integrated-decode for inference. "
            "sched_idle_frac = clamp(1 - compute_busy_s/makespan_s, 0, 1); "
            "collective/SEND/RECV time lands in the idle complement (comm is not "
            "separable from the ET), so thermal consumers must not treat it as pure "
            "idle-power time. kernel_idle_* derive from pricing-time counters "
            "(once per layer type; GA/microbatch-independent); embedding and "
            "pointwise ops are uninstrumented, so stage-0 devices under-report "
            "kernel idle."
        ),
    }

    os.makedirs(exp_dir, exist_ok=True)
    path = os.path.join(exp_dir, DEVICE_METRICS_FILENAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
    return path
