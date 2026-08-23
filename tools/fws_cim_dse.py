"""FWS-CIM design-space exploration (DESIGN2 pass 2B, section 5).

Rederives the OPTIMA compiler concepts on :class:`cim_timing.CimDeviceModel`,
the single source of truth for every FWS-CIM law. Evaluation is CLOSED FORM
ONLY — the tool never runs run_perf per candidate. The one optional run_perf
invocation (``--verify``) executes the SELECTED point once and cross-checks
period and throughput against the closed form (must match <= 0.1%; the DSE
and the simulator share CimDeviceModel, so this guards drift).

Knobs (enumerated exhaustively; the product is small by design):
  - array variant: ``cim.dse.variants`` — each entry is a complete analog
    array point after inheritance (rows / slice_cycles / analog_clock_mhz
    fall back to ``cim.analog``). ``cim.dse.mux_candidates`` never expands
    the sweep: when set it is a CROSS-CHECK — every variant's adc_mux must
    appear in it, or the tool exits with a config error (typo guard).
  - tp: ``cim.dse.tp_candidates``; ``auto`` derives divisors of the model's
    num_heads (the run path shards heads with ceil and imposes no
    intermediate-size divisibility gate; the only extra hard gate is the
    MoE routing-group divisibility ``num_experts % (tp * moe_dp) == 0``,
    which auto mirrors). tp >= 2 evaluates a SYSTEM of tp shard devices:
    the timing laws shard kv heads / KV bytes per device, and the chips /
    arrays / area metrics (and the max_chips constraint) multiply the
    per-shard figures by tp (the census does not shard weight matrices,
    so each shard is counted at the full per-device figure — a
    conservative upper bound). tp therefore costs real silicon in the
    selection instead of acting as a free throughput knob.
  - moe_expert_parallel: ``cim.dse.moe_expert_parallel`` (MoE models only;
    default [1]).
  - chips are DERIVED per candidate via the ``layers_per_chip: auto``
    greedy placement — mapping is an output, never a knob.
  - the digital fabric is FIXED (pass 2B does not resize it).

Constraints per candidate (every infeasible candidate is recorded with a
stage tag and message — fail-fast, self-explaining; nothing is silently
dropped). Stage order, used to rank the "best violation" (the candidate
that got furthest):
  validation < placement < capacity < max_chips < boundary_bandwidth
  < dispatch_bandwidth < kv_capacity

Metrics per candidate mirror the FWS spatial report
(inference_timing._write_fws_cim_report) by composing the same
CimDeviceModel laws: period, throughput (fps for prefill/ViT workloads,
tok/s at the final decode context for decode workloads — a FABRIC CEILING
B/period, plus the sustained figure
``CimDeviceModel.decode_sustained_throughput`` that caps the resident
wavefronts by the KV stream capacity and the rate by B/period and the KV
tier bandwidth), single-item latency, chips (backbone + MoE expert pool,
x tp shard devices), total analog array area, arrays utilization, and the
PARTIAL energy per inference (analog + boundary interconnect incl. MoE
dispatch/combine on the ep link + cim_dram KV traffic).

Pareto front: throughput vs total array area over the valid candidates
(2-axis dominance). Both the front's throughput axis and the final
selection use the workload's ranking key: ``sustained_tokens_per_s`` on
decode workloads (the honest headline — the fabric ceiling stays as
info), fps otherwise. Selection is lexicographic per ``--objective``:
  throughput (default): max throughput, then min chips, min area, min tp,
                        min adc_mux.
  min_chips:            min chips, then max throughput, min area, min tp,
                        min adc_mux.

Artifacts: ``<output-dir>/dse_report.md`` and ``dse_report.json``;
``--emit-config`` additionally writes a complete runnable fws_cim hardware
YAML for the selected point (base YAML with the chosen variant's analog
fields, the DERIVED layers_per_chip list, tp in parallelism, and
moe_expert_parallel in cim.chip).

Exit codes: 0 ok; 2 usage/config error; 3 no feasible candidate (the best
violation is printed); 4 --verify mismatch or verify-run failure.
"""

import argparse
import copy
import json
import math
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml

import cim_timing
import config


# Constraint stages in evaluation order; index = how far a candidate got.
CONSTRAINT_STAGES = (
    "validation",
    "placement",
    "capacity",
    "max_chips",
    "boundary_bandwidth",
    "dispatch_bandwidth",
    "kv_capacity",
)

VERIFY_REL_TOL = 1e-3  # <= 0.1% (DESIGN2 section 5)


class DseUsageError(ValueError):
    """Tool-level configuration error (exit code 2)."""


def _load_yaml(path):
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def _hw_from_raw_dict(raw_dict):
    """Mirror config.parse_config for an in-memory hardware dict."""
    converted = copy.deepcopy(raw_dict)
    config.convert(converted)
    return config.HWConfig.from_dict(converted)


def _model_mode(model_path):
    model_dict = _load_yaml(model_path)
    model_param = (model_dict or {}).get("model_param") or {}
    mode = str(model_param.get("mode", "")).strip().upper()
    if mode not in ("LLM", "VIT"):
        raise DseUsageError(
            f"model config {model_path} must set model_param.mode to LLM or VIT "
            f"(got {mode!r}); the DSE covers transformer inference only."
        )
    return mode


def _divisors(n):
    n = int(n)
    return [d for d in range(1, n + 1) if n % d == 0]


def derive_tp_candidates(dse_cfg, model, moe_dp=1):
    """tp candidates: explicit list, or 'auto' = divisors of num_heads.

    Auto additionally mirrors the run path's only extra hard gate for MoE
    inference: num_experts % (tp * moe_dp) == 0 (config.validate_model_config).
    Head sharding itself is ceil-based, so no other divisibility applies.
    """
    if isinstance(dse_cfg.tp_candidates, tuple):
        return [int(t) for t in dse_cfg.tp_candidates]
    num_heads = int(model.num_heads)
    cands = _divisors(num_heads)
    if bool(getattr(model, "use_moe", False)):
        num_experts = int(getattr(model, "num_experts", 1) or 1)
        group = max(1, int(moe_dp))
        cands = [t for t in cands if num_experts % (t * group) == 0]
    return cands


def _variant_analog_fields(variant, base_analog):
    """Complete analog point for one variant after cim.analog inheritance."""
    return {
        "adc_mux": int(variant.adc_mux),
        "cols_adc": int(variant.cols_adc),
        "rows": int(base_analog.rows if variant.rows is None else variant.rows),
        "slice_cycles": int(
            base_analog.slice_cycles if variant.slice_cycles is None else variant.slice_cycles
        ),
        "analog_clock_mhz": float(
            base_analog.analog_clock_mhz
            if variant.analog_clock_mhz is None
            else variant.analog_clock_mhz
        ),
        "energy_per_vec_pj": float(variant.energy_per_vec_pj),
        "area_mm2_per_array": float(variant.area_mm2_per_array),
    }


def _candidate_hw_raw(base_raw, analog_fields, tp, moe_expert_parallel):
    """Raw (unconverted) hardware dict for one candidate / the emitted YAML."""
    raw = copy.deepcopy(base_raw)
    raw["cim"]["analog"].update(analog_fields)
    raw["cim"]["chip"]["layers_per_chip"] = "auto"
    raw["cim"]["chip"]["moe_expert_parallel"] = int(moe_expert_parallel)
    raw.setdefault("parallelism", {})["tp"] = int(tp)
    return raw


def _fail(candidate, stage, message):
    candidate["ok"] = False
    if candidate["fail_stage"] is None:
        candidate["fail_stage"] = stage
        candidate["fail_message"] = message
    candidate["constraints"][stage] = {"ok": False, "message": message}


def evaluate_candidate(
    cand_id,
    base_raw,
    variant_index,
    analog_fields,
    tp,
    moe_expert_parallel,
    model_config,
    max_chips,
):
    """Closed-form evaluation of one (variant, tp, moe_expert_parallel) point.

    All laws come from CimDeviceModel; the compositions (period over the
    stage tables, boundary/dispatch checks, KV capacity, partial energy)
    mirror inference_timing._write_fws_cim_report exactly.
    """
    model = model_config.model_config
    candidate = {
        "id": cand_id,
        "variant_index": variant_index,
        "knobs": dict(analog_fields, tp=int(tp), moe_expert_parallel=int(moe_expert_parallel)),
        "ok": True,
        "fail_stage": None,
        "fail_message": None,
        "constraints": {},
        "metrics": {},
    }

    hw_raw = _candidate_hw_raw(base_raw, analog_fields, tp, moe_expert_parallel)

    # Stage: validation (parse + the run path's own hardware/model gates).
    try:
        hw = _hw_from_raw_dict(hw_raw)
        config.validate_hw_config(hw)
        config.validate_model_config(hw, model_config)
        dev = cim_timing.CimDeviceModel(hw, model)
    except (ValueError, KeyError, TypeError) as exc:
        _fail(candidate, "validation", str(exc))
        return candidate, None, None
    candidate["constraints"]["validation"] = {"ok": True}

    # Stage: placement (auto greedy derivation; chips become an output).
    try:
        chip_layers = dev.chip_layer_counts()
    except ValueError as exc:
        _fail(candidate, "placement", str(exc))
        return candidate, dev, None
    candidate["constraints"]["placement"] = {"ok": True}

    # Stage: capacity (layer chips + the MoE expert pool).
    try:
        dev.validate_capacity()
    except ValueError as exc:
        _fail(candidate, "capacity", str(exc))
        return candidate, dev, None
    candidate["constraints"]["capacity"] = {"ok": True}

    backbone_chips = len(chip_layers)
    pool_chips, pool_arrays_each = dev.moe_expert_pool()
    # tp >= 2 runs a SYSTEM of tp shard devices (the timing laws shard kv
    # heads / KV bytes per device). The census does not shard weight
    # matrices, so each shard is counted at the full per-device figure — a
    # conservative upper bound — and every system resource metric (chips,
    # arrays, area) multiplies by tp. This is what makes tp a real cost
    # knob instead of a free throughput multiplier.
    tp_shards = max(1, int(tp))
    chips_per_shard = backbone_chips + pool_chips
    chips_total = tp_shards * chips_per_shard

    # Stage: max_chips (0 = unbounded).
    if max_chips > 0 and chips_total > max_chips:
        _fail(
            candidate,
            "max_chips",
            f"candidate needs {chips_total} chips ({backbone_chips} backbone "
            f"+ {pool_chips} expert pool, x {tp_shards} tp shard devices) "
            f"but cim.dse.max_chips = {max_chips}.",
        )
    else:
        candidate["constraints"]["max_chips"] = {"ok": True}

    # ---- metrics (mirroring the FWS spatial report compositions) ----------
    params = dev.params
    is_vit = params.is_vit_shaped
    decode_len = int(getattr(model, "decode_len", 0) or 0)
    final_context = int(model.seq_len)
    prefill_len = final_context - decode_len
    batch_size = int(params.batch_size)
    tokens_owner = None if is_vit else batch_size * prefill_len
    tokens_owner_num = prefill_len if tokens_owner is None else tokens_owner
    # The LLM wavefront carries B streams: S2 prices streams = B (mirrors
    # the report; ViT keeps the pass-1 one-image wavefront).
    streams = 1 if is_vit else batch_size

    period_s, bottleneck = dev.pipeline_period(
        prefill_len, tp, batch_size, tokens=tokens_owner, streams=streams
    )
    fps = 1.0 / period_s if period_s > 0 else float("inf")

    mask = dev.layer_class_mask()
    num_moe_layers = sum(mask)
    num_dense_layers = len(mask) - num_moe_layers
    dense_block_s = (
        sum(
            dev.layer_stage_times(
                prefill_len, tp, tokens=tokens_owner, streams=streams
            ).values()
        )
        if num_dense_layers > 0
        else 0.0
    )
    moe_block_s = (
        sum(
            dev.moe_layer_stage_times(
                prefill_len, tp, tokens_owner=tokens_owner, streams=streams
            ).values()
        )
        if num_moe_layers > 0
        else 0.0
    )
    endpoint_s = sum(dev.endpoint_stage_times(prefill_len, batch_size).values())

    # Boundary p2p transfers between consecutive backbone chips (pp link).
    precision = hw.sw_config.precision
    act_bytes = float(precision.activations)
    layout = hw.network_layout
    pp_bw, pp_lat = layout.link_for_parallelism("pp")
    num_boundaries = backbone_chips - 1
    boundary_bytes = dev.boundary_bytes(batch_size * prefill_len, act_bytes)
    boundary_time_s = (
        dev.p2p_time_s(boundary_bytes, pp_bw, pp_lat) if num_boundaries > 0 else 0.0
    )

    # MoE dispatch/combine over the ep link.
    dispatch_time_s = 0.0
    if num_moe_layers > 0:
        ep_bw, ep_lat = layout.link_for_parallelism("ep")
        dispatch_time_s = dev.moe_dispatch_time(tokens_owner_num, act_bytes, ep_bw, ep_lat)

    prefill_e2e_s = (
        num_dense_layers * dense_block_s
        + num_moe_layers * moe_block_s
        + endpoint_s
        + num_boundaries * boundary_time_s
        + num_moe_layers * 2.0 * dispatch_time_s
    )

    # KV story (LLM only; ViT runs disable the KV cache by mode).
    kv_story = str(
        getattr(getattr(hw, "inference_config", None), "kvcache_type", "hbm_only")
    ).strip().lower()
    kv_precision = float(precision.kv_cache)
    tech_dram = hw.tech_config.DRAM
    sram_size = float(getattr(tech_dram, "size", 0.0) or 0.0)
    sram_bw = float(getattr(tech_dram, "bandwidth", 0.0) or 0.0)
    kv_active = (not is_vit) and kv_story in ("cim_sram", "cim_dram")
    kv_bw = None
    kv_capacity = None
    kv_total = None
    if kv_active:
        kv_bw = dev.kv_story_bandwidth(kv_story, sram_bw)
        kv_capacity = dev.kv_story_capacity(kv_story, sram_size)
        kv_total = float(batch_size) * dev.kv_bytes_per_stream(final_context, kv_precision, tp)

    # Throughput per workload type.
    decode_period_s = None
    sustained = None
    single_item_latency_s = prefill_e2e_s
    if decode_len > 0 and kv_active:
        workload = "decode"
        decode_period_s, _ = dev.decode_pipeline_period(
            final_context, kv_bw, kv_precision, batch_size, tp
        )
        throughput = batch_size / decode_period_s if decode_period_s > 0 else float("inf")
        throughput_unit = "tok/s"

        def _decode_step_latency(ctx):
            dense_d = (
                sum(
                    dev.decode_layer_stage_times(ctx, kv_bw, kv_precision, batch_size, tp).values()
                )
                if num_dense_layers > 0
                else 0.0
            )
            moe_d = (
                sum(
                    dev.decode_moe_layer_stage_times(
                        ctx, kv_bw, kv_precision, batch_size, tp
                    ).values()
                )
                if num_moe_layers > 0
                else 0.0
            )
            endpoint_d = sum(
                dev.endpoint_stage_times(batch_size=batch_size, decode=True).values()
            )
            boundary_d = 0.0
            if num_boundaries > 0:
                boundary_d = dev.p2p_time_s(
                    dev.boundary_bytes(batch_size, act_bytes), pp_bw, pp_lat
                )
            dispatch_d = 0.0
            if num_moe_layers > 0:
                dispatch_d = dev.moe_dispatch_time(batch_size, act_bytes, ep_bw, ep_lat)
            return (
                num_dense_layers * dense_d
                + num_moe_layers * moe_d
                + endpoint_d
                + num_boundaries * boundary_d
                + num_moe_layers * 2.0 * dispatch_d
            )

        # Trapezoid over the first/final step latencies (the decode laws are
        # stepwise in context; stated as an approximation, like the report).
        step_first = _decode_step_latency(prefill_len + 1)
        step_final = _decode_step_latency(final_context)
        single_item_latency_s = prefill_e2e_s + decode_len * 0.5 * (step_first + step_final)
        # Sustained throughput at the final context: the B/period figure
        # above is a fabric ceiling; the KV stream capacity caps the
        # resident wavefronts, and the law additionally caps the rate at
        # B/period and at the KV tier bandwidth over the per-token KV read
        # (CimDeviceModel.decode_sustained_throughput — mirrors the report).
        sustained = dev.decode_sustained_throughput(
            step_final,
            decode_period_s,
            dev.kv_max_streams(kv_capacity, final_context, kv_precision, tp),
            batch_size,
            kv_read_bandwidth_bytes_per_s=kv_bw,
            kv_bytes_per_token=dev.kv_bytes_per_stream(
                final_context, kv_precision, tp
            ),
        )
    else:
        workload = "prefill"
        throughput = fps
        throughput_unit = "fps"

    # Area / arrays / utilization (system totals: per-shard figure x tp
    # shards; utilization is a per-shard ratio and is tp-invariant).
    arrays_total = tp_shards * dev.total_arrays()
    area_total = tp_shards * dev.total_area_mm2()
    capacity = int(dev.chip.arrays_per_chip)
    chip_usage = dev.chip_array_usage()
    placed_arrays = sum(chip_usage) + pool_chips * pool_arrays_each
    utilization = (
        placed_arrays / float(chips_per_shard * capacity)
        if capacity > 0 and chips_per_shard > 0
        else None
    )

    # PARTIAL energy per inference (report composition: analog stack +
    # endpoints + boundary interconnect + cim_dram KV traffic).
    pp_dim = layout.dimension_for_parallelism("pp")
    pp_energy_per_bit_j = float(getattr(pp_dim, "energy_per_bit", 0.0) or 0.0)
    interconnect_pj = boundary_bytes * 8.0 * pp_energy_per_bit_j * num_boundaries * 1e12
    # MoE dispatch/combine are boundary transfers too: price their bytes on
    # the ep link's energy_per_bit (mirrors the report; bytes are
    # link-count-invariant under moe_expert_parallel).
    if num_moe_layers > 0:
        ep_dim = layout.dimension_for_parallelism("ep")
        ep_energy_per_bit_j = float(getattr(ep_dim, "energy_per_bit", 0.0) or 0.0)
        dispatch_bytes_each_way = dev.moe_dispatch_bytes(tokens_owner_num, act_bytes)
        interconnect_pj += (
            dispatch_bytes_each_way * 8.0 * ep_energy_per_bit_j * num_moe_layers * 2.0 * 1e12
        )
    analog_stack_pj = dev.transformer_stack_energy_pj(tokens_owner_num)
    analog_endpoints_pj = sum(dev.endpoint_energy_pj(prefill_len, batch_size).values())
    kv_dram_pj = 0.0
    if kv_active and kv_story == "cim_dram":
        per_token_layer = dev.kv_bytes_per_stream_layer(1, kv_precision, tp)
        decode_ctx_sum = 0.0
        if decode_len > 0:
            decode_ctx_sum = (
                float(decode_len) * float(prefill_len)
                + float(decode_len) * (decode_len + 1) / 2.0
            )
        kv_read_bytes_total = (
            float(batch_size) * params.num_layers * per_token_layer * decode_ctx_sum
        )
        kv_dram_pj = dev.kv_dram_energy_pj(kv_total + kv_read_bytes_total)
    energy_partial_pj = analog_stack_pj + analog_endpoints_pj + interconnect_pj + kv_dram_pj

    candidate["metrics"] = {
        "workload": workload,
        "period_us": period_s * 1e6,
        "bottleneck_stage": bottleneck,
        "fps": fps,
        "decode_period_us": None if decode_period_s is None else decode_period_s * 1e6,
        # "throughput" on decode workloads is the FABRIC CEILING B/period
        # (kept as info); "sustained_tokens_per_s" reconciles it with the
        # KV stream capacity and is the decode selection key.
        "throughput": throughput,
        "throughput_unit": throughput_unit,
        "sustained_tokens_per_s": None if sustained is None else sustained.tokens_per_s,
        "wavefronts_full": None if sustained is None else sustained.wavefronts_full,
        "wavefronts_kv": None if sustained is None else sustained.wavefronts_kv,
        "decode_throughput_limit": None if sustained is None else sustained.limiting_factor,
        "single_item_latency_us": single_item_latency_s * 1e6,
        # chips / arrays_total / area are SYSTEM totals (x tp shard
        # devices); backbone_chips / expert_pool_chips / the derived layer
        # split describe ONE shard device.
        "chips": chips_total,
        "tp_shards": tp_shards,
        "backbone_chips": backbone_chips,
        "expert_pool_chips": pool_chips,
        # One-way MoE dispatch (== combine) time on the ep link at the owner
        # token count — the same figure the dispatch_bandwidth constraint
        # compares against the period. CimDeviceModel.moe_dispatch_time
        # divides by moe_expert_parallel (k parallel links). None for
        # models with no MoE layers.
        "dispatch_time_us": None if num_moe_layers == 0 else dispatch_time_s * 1e6,
        "derived_layers_per_chip": [int(c) for c in chip_layers],
        "arrays_total": int(arrays_total),
        "area_mm2_total": area_total,
        "arrays_utilization": utilization,
        "energy_partial_pj": energy_partial_pj,
    }

    # Stage: boundary bandwidth vs the pipeline period (report law: a
    # transfer longer than the period makes the pipeline bandwidth-bound).
    if num_boundaries > 0 and boundary_time_s > period_s:
        _fail(
            candidate,
            "boundary_bandwidth",
            f"chip-boundary transfer time ({boundary_time_s * 1e6:.6f} us) exceeds "
            f"the pipeline period ({period_s * 1e6:.6f} us); bandwidth-bound on the pp link.",
        )
    else:
        candidate["constraints"]["boundary_bandwidth"] = {"ok": True}

    # Stage: MoE dispatch/combine bandwidth vs the period (ep link).
    if num_moe_layers > 0 and dispatch_time_s > period_s:
        _fail(
            candidate,
            "dispatch_bandwidth",
            f"MoE dispatch/combine transfer time ({dispatch_time_s * 1e6:.6f} us) "
            f"exceeds the pipeline period ({period_s * 1e6:.6f} us); bandwidth-bound "
            "on the ep link.",
        )
    else:
        candidate["constraints"]["dispatch_bandwidth"] = {"ok": True}

    # Stage: KV capacity at the model's final context and batch (story-aware).
    if kv_active:
        if kv_total > kv_capacity:
            _fail(
                candidate,
                "kv_capacity",
                f"KV cache at context {final_context} needs "
                f"{kv_total / 1024 ** 3:.2f} GiB ({batch_size} streams) but the "
                f"{kv_story} capacity is {kv_capacity / 1024 ** 3:.2f} GiB.",
            )
        else:
            candidate["constraints"]["kv_capacity"] = {"ok": True}
    else:
        candidate["constraints"]["kv_capacity"] = {"ok": True, "message": "no KV story (ViT)"}

    return candidate, dev, hw_raw


def _selection_throughput(m):
    """The throughput key selection ranks on.

    Decode workloads rank on sustained_tokens_per_s — the honest headline
    that caps the fabric ceiling by the KV stream capacity. Prefill/ViT
    workloads (sustained is None) rank on the fps figure in "throughput".
    """
    sustained = m.get("sustained_tokens_per_s")
    return m["throughput"] if sustained is None else sustained


def _selection_key(candidate, objective):
    m = candidate["metrics"]
    if objective == "min_chips":
        return (
            m["chips"],
            -_selection_throughput(m),
            m["area_mm2_total"],
            candidate["knobs"]["tp"],
            candidate["knobs"]["adc_mux"],
        )
    return (
        -_selection_throughput(m),
        m["chips"],
        m["area_mm2_total"],
        candidate["knobs"]["tp"],
        candidate["knobs"]["adc_mux"],
    )


SELECTION_RULES = {
    "throughput": (
        "lexicographic: max throughput, then min chips, then min total array "
        "area, then min tp, then min adc_mux"
    ),
    "min_chips": (
        "lexicographic: min chips, then max throughput, then min total array "
        "area, then min tp, then min adc_mux"
    ),
}

# On decode workloads the ranking key is the sustained figure, not the
# fabric-ceiling tok/s; run_dse substitutes this label into the rule text.
DECODE_THROUGHPUT_LABEL = (
    "sustained throughput (sustained_tokens_per_s; the fabric-ceiling tok/s "
    "stays as info)"
)


def pareto_front_ids(valid_candidates):
    """Non-dominated ids under (max throughput, min total array area).

    The throughput axis is the same key selection ranks on
    (:func:`_selection_throughput`): sustained_tokens_per_s on decode
    workloads, fps otherwise — so the selected candidate always sits on
    the front it is chosen from.
    """
    front = []
    for cand in valid_candidates:
        m = cand["metrics"]
        thr = _selection_throughput(m)
        dominated = False
        for other in valid_candidates:
            if other is cand:
                continue
            om = other["metrics"]
            othr = _selection_throughput(om)
            if (
                othr >= thr
                and om["area_mm2_total"] <= m["area_mm2_total"]
                and (othr > thr or om["area_mm2_total"] < m["area_mm2_total"])
            ):
                dominated = True
                break
        if not dominated:
            front.append(cand["id"])
    return front


def _best_violation(candidates):
    """The failure of the candidate that got furthest (highest stage index)."""
    failed = [c for c in candidates if not c["ok"]]
    if not failed:
        return None
    order = {stage: idx for idx, stage in enumerate(CONSTRAINT_STAGES)}
    best = max(failed, key=lambda c: order.get(c["fail_stage"], -1))
    return best


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    return obj


def _fmt(value, spec=".6g"):
    if value is None:
        return "-"
    if isinstance(value, float):
        return format(value, spec)
    return str(value)


def _write_markdown(path, payload):
    lines = []
    lines.append("# FWS-CIM DSE report")
    lines.append("")
    lines.append(f"- hardware config: `{payload['hardware_config']}`")
    lines.append(f"- model config: `{payload['model_config']}`")
    lines.append(f"- workload: {payload['workload']} (throughput unit: {payload['throughput_unit']})")
    lines.append(f"- objective: {payload['objective']}")
    lines.append(f"- selection rule: {payload['selection_rule']}")
    lines.append(
        "- evaluation: closed form on cim_timing.CimDeviceModel only "
        "(no run_perf per candidate); chips derived via layers_per_chip: auto; "
        "digital fabric fixed"
    )
    lines.append("")

    selected = payload["selected"]
    lines.append("## Selected mapping")
    lines.append("")
    if selected is None:
        best = payload.get("best_violation")
        lines.append("NONE — every candidate is infeasible.")
        if best is not None:
            lines.append(
                f"Best violation (candidate {best['id']}, stage {best['fail_stage']}): "
                f"{best['fail_message']}"
            )
    else:
        k = selected["knobs"]
        m = selected["metrics"]
        lines.append(f"- candidate id: {selected['id']}")
        lines.append(
            f"- array variant: adc_mux={k['adc_mux']}, cols_adc={k['cols_adc']}, "
            f"rows={k['rows']}, slice_cycles={k['slice_cycles']}, "
            f"analog_clock_mhz={_fmt(k['analog_clock_mhz'])}"
        )
        lines.append(
            f"- energy_per_vec_pj={_fmt(k['energy_per_vec_pj'])}, "
            f"area_mm2_per_array={_fmt(k['area_mm2_per_array'])}"
        )
        lines.append(f"- tp={k['tp']}, moe_expert_parallel={k['moe_expert_parallel']}")
        shards = int(m.get("tp_shards", 1) or 1)
        lines.append(
            f"- chips: {m['chips']} ({m['backbone_chips']} backbone + "
            f"{m['expert_pool_chips']} expert pool"
            + (f", x {shards} tp shard devices" if shards > 1 else "")
            + f"); derived layers_per_chip {m['derived_layers_per_chip']}"
        )
        lines.append(
            f"- period {_fmt(m['period_us'])} us (bottleneck {m['bottleneck_stage']}); "
            f"throughput {_fmt(m['throughput'])} {payload['throughput_unit']}"
            + (" (fabric ceiling)" if m.get("sustained_tokens_per_s") is not None else "")
            + f"; single-item latency {_fmt(m['single_item_latency_us'])} us"
        )
        if m.get("sustained_tokens_per_s") is not None:
            lines.append(
                f"- sustained {_fmt(m['sustained_tokens_per_s'])} tok/s at KV capacity "
                f"(wavefronts: full {m['wavefronts_full']}, kv {m['wavefronts_kv']}; "
                f"limited by {m['decode_throughput_limit']}) — the decode selection key"
            )
        lines.append(
            f"- arrays {m['arrays_total']} (utilization {_fmt(m['arrays_utilization'], '.4g')}); "
            f"total array area {_fmt(m['area_mm2_total'])} mm2; "
            f"PARTIAL energy/inference {_fmt(m['energy_partial_pj'])} pJ"
        )
    lines.append("")

    lines.append("## Candidates")
    lines.append("")
    header = (
        "| id | mux | cols_adc | rows | tp | moe_ep | ok | fail stage | chips | "
        "period_us | throughput | sustained | area_mm2 | util | energy_pJ |"
    )
    lines.append(header)
    lines.append("|" + "---|" * 15)
    for cand in payload["candidates"]:
        k = cand["knobs"]
        m = cand.get("metrics") or {}
        lines.append(
            f"| {cand['id']} | {k['adc_mux']} | {k['cols_adc']} | {k['rows']} "
            f"| {k['tp']} | {k['moe_expert_parallel']} "
            f"| {'yes' if cand['ok'] else 'NO'} | {cand['fail_stage'] or '-'} "
            f"| {_fmt(m.get('chips'))} | {_fmt(m.get('period_us'))} "
            f"| {_fmt(m.get('throughput'))} "
            f"| {_fmt(m.get('sustained_tokens_per_s'))} "
            f"| {_fmt(m.get('area_mm2_total'))} "
            f"| {_fmt(m.get('arrays_utilization'), '.4g')} "
            f"| {_fmt(m.get('energy_partial_pj'), '.4g')} |"
        )
    lines.append("")

    lines.append(
        "## Pareto front (throughput vs total array area, valid candidates; "
        "decode workloads use the sustained figure as the throughput axis)"
    )
    lines.append("")
    if payload["front_ids"]:
        for cid in payload["front_ids"]:
            cand = next(c for c in payload["candidates"] if c["id"] == cid)
            m = cand["metrics"]
            label = (
                "sustained"
                if m.get("sustained_tokens_per_s") is not None
                else "throughput"
            )
            lines.append(
                f"- {cid}: {label} {_fmt(_selection_throughput(m))} "
                f"{payload['throughput_unit']}, area {_fmt(m['area_mm2_total'])} mm2"
            )
    else:
        lines.append("(empty — no valid candidate)")
    lines.append("")

    failures = [c for c in payload["candidates"] if not c["ok"]]
    lines.append("## Infeasible candidates")
    lines.append("")
    if failures:
        for cand in failures:
            lines.append(f"- {cand['id']} [{cand['fail_stage']}]: {cand['fail_message']}")
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("## Config echo (cim.dse)")
    lines.append("")
    lines.append("```yaml")
    lines.append(yaml.safe_dump(payload["dse_echo"], sort_keys=False).rstrip())
    lines.append("```")
    lines.append("")
    path.write_text("\n".join(lines))


def emit_selected_config(base_raw, selected, out_path):
    """Write the complete runnable fws_cim hardware YAML for the selection.

    Base YAML with the chosen variant's analog fields, the DERIVED
    layers_per_chip list (an explicit mapping, so the emitted config is
    self-describing), tp in parallelism, and moe_expert_parallel in
    cim.chip.
    """
    k = selected["knobs"]
    analog_fields = {
        name: k[name]
        for name in (
            "adc_mux",
            "cols_adc",
            "rows",
            "slice_cycles",
            "analog_clock_mhz",
            "energy_per_vec_pj",
            "area_mm2_per_array",
        )
    }
    raw = _candidate_hw_raw(base_raw, analog_fields, k["tp"], k["moe_expert_parallel"])
    raw["cim"]["chip"]["layers_per_chip"] = [
        int(c) for c in selected["metrics"]["derived_layers_per_chip"]
    ]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return out_path


def verify_selected(emitted_path, model_path, mode, output_dir, selected, workload):
    """Run run_perf ONCE on the emitted config and cross-check the closed form.

    The subprocess runs with cwd = the DSE output dir, so run_perf's
    output/<MODE> tree lands inside the DSE artifacts and never clobbers the
    repo's own output directory (the run-order landmine from pass 1).
    Returns a result dict; sets "pass": False on any mismatch (> 0.1%).
    """
    result = {
        "ran": True,
        "emitted_config": str(emitted_path),
        "pass": False,
        "checks": [],
        "message": None,
    }
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "run_perf.py"),
            "--hardware_config",
            str(Path(emitted_path).resolve()),
            "--model_config",
            str(Path(model_path).resolve()),
        ],
        cwd=str(output_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=1200,
    )
    if proc.returncode != 0:
        result["message"] = (
            f"run_perf failed with exit code {proc.returncode}; last output:\n"
            + proc.stdout[-4000:]
        )
        return result
    report_path = Path(output_dir) / "output" / mode / "fws_cim_report.json"
    if not report_path.exists():
        result["message"] = f"run_perf wrote no report at {report_path}"
        return result
    report = json.loads(report_path.read_text())

    m = selected["metrics"]
    checks = [("period_us", m["period_us"], float(report["period_us"]))]
    if workload == "decode":
        sim_decode = report.get("decode") or {}
        sim_toks = float(sim_decode.get("aggregate_tokens_per_s_final") or 0.0)
        checks.append(("tokens_per_s", m["throughput"], sim_toks))
        # The sustained (KV-capped) figure must round-trip too — it is the
        # decode selection key.
        sim_sustained = float(sim_decode.get("sustained_tokens_per_s") or 0.0)
        checks.append(
            ("sustained_tokens_per_s", m["sustained_tokens_per_s"], sim_sustained)
        )
    else:
        checks.append(("fps", m["throughput"], float(report["fps"])))

    ok = True
    for name, dse_value, sim_value in checks:
        rel = abs(dse_value - sim_value) / abs(sim_value) if sim_value != 0 else float("inf")
        check_ok = rel <= VERIFY_REL_TOL
        ok = ok and check_ok
        result["checks"].append(
            {"name": name, "dse": dse_value, "simulator": sim_value, "rel_err": rel, "ok": check_ok}
        )
    result["pass"] = ok
    if not ok:
        result["message"] = (
            "closed-form vs run_perf mismatch > 0.1%: "
            + "; ".join(
                f"{c['name']}: dse {c['dse']:.9g} vs sim {c['simulator']:.9g} "
                f"(rel {c['rel_err']:.3e})"
                for c in result["checks"]
                if not c["ok"]
            )
        )
    return result


def run_dse(
    hardware_config,
    model_config_path,
    output_dir=None,
    objective="throughput",
    emit_config=None,
    verify=False,
):
    """Run the sweep; return (exit_code, payload dict). Artifacts are always
    written to the output dir (even for an all-infeasible sweep)."""
    hw_path = Path(hardware_config)
    model_path = Path(model_config_path)
    if objective not in SELECTION_RULES:
        raise DseUsageError(f"unknown --objective {objective!r}; use throughput or min_chips.")

    base_raw = _load_yaml(hw_path)
    if not isinstance(base_raw, dict) or "cim" not in base_raw:
        raise DseUsageError(
            f"{hw_path} is not a fws_cim hardware config (no cim block); the DSE "
            "needs a device_class: fws_cim YAML with a cim.dse block."
        )
    base_hw = _hw_from_raw_dict(base_raw)
    if str(getattr(base_hw, "device_class", "gpu")).lower() != "fws_cim":
        raise DseUsageError(f"{hw_path} must set device_class: fws_cim.")
    dse_cfg = base_hw.cim_config.dse
    if dse_cfg is None or not dse_cfg.variants:
        raise DseUsageError(
            f"{hw_path} has no cim.dse.variants — the DSE candidate space is "
            "empty. Add a cim.dse block with at least one array variant."
        )
    # cim.dse.mux_candidates is a cross-check, never a knob: the sweep
    # enumerates variants only. When the list is set, a variant whose
    # adc_mux is missing from it is a config error (typo guard — a user
    # expecting a mux sweep must express it as variants).
    mux_candidates = [int(m) for m in dse_cfg.mux_candidates]
    if mux_candidates:
        for index, variant in enumerate(dse_cfg.variants):
            if int(variant.adc_mux) not in mux_candidates:
                raise DseUsageError(
                    f"cim.dse.variants[{index}].adc_mux = {int(variant.adc_mux)} "
                    f"is not in cim.dse.mux_candidates {mux_candidates}. "
                    "mux_candidates is a cross-check on the variant list (the "
                    "sweep enumerates variants, never mux_candidates); add the "
                    "mux value to the list or drop the variant."
                )
    if int(base_hw.cim_config.chip.arrays_per_chip) <= 0:
        raise DseUsageError(
            "the DSE derives chips via layers_per_chip: auto, which requires "
            f"cim.chip.arrays_per_chip > 0 (got {base_hw.cim_config.chip.arrays_per_chip})."
        )

    mode = _model_mode(model_path)
    model_config = config.parse_config(str(model_path), mode)
    model = model_config.model_config

    if output_dir is None:
        output_dir = REPO_ROOT / "output" / "fws_cim_dse" / f"{hw_path.stem}__{model_path.stem}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    moe_dp = int(getattr(base_hw.sch_config.inference, "moe_dp", 1) or 1)
    tp_candidates = derive_tp_candidates(dse_cfg, model, moe_dp=moe_dp)
    if not tp_candidates:
        raise DseUsageError(
            "cim.dse.tp_candidates resolved to an empty list "
            f"(tp_candidates={dse_cfg.tp_candidates!r}, num_heads={model.num_heads})."
        )
    is_moe_model = bool(getattr(model, "use_moe", False))
    moe_ep_candidates = list(dse_cfg.moe_expert_parallel) if is_moe_model else [1]

    decode_len = int(getattr(model, "decode_len", 0) or 0)
    if int(model.seq_len) - decode_len <= 0:
        raise DseUsageError(
            "the DSE (like the FWS spatial report) requires prefill_len = "
            f"seq_len - decode_len > 0 (got seq_len={model.seq_len}, "
            f"decode_len={decode_len})."
        )
    workload = "decode" if decode_len > 0 and mode != "VIT" else "prefill"
    throughput_unit = "tok/s" if workload == "decode" else "fps"

    candidates = []
    cand_index = 0
    for variant_index, variant in enumerate(dse_cfg.variants):
        analog_fields = _variant_analog_fields(variant, base_hw.cim_config.analog)
        for tp in tp_candidates:
            for k in moe_ep_candidates:
                cand_id = f"c{cand_index:03d}"
                cand_index += 1
                candidate, _, _ = evaluate_candidate(
                    cand_id,
                    base_raw,
                    variant_index,
                    analog_fields,
                    tp,
                    k,
                    model_config,
                    int(dse_cfg.max_chips),
                )
                candidates.append(candidate)

    valid = [c for c in candidates if c["ok"]]
    front_ids = pareto_front_ids(valid)
    selected = min(valid, key=lambda c: _selection_key(c, objective)) if valid else None
    best_violation = _best_violation(candidates)

    selection_rule = SELECTION_RULES[objective]
    if workload == "decode":
        selection_rule = selection_rule.replace(
            "max throughput", "max " + DECODE_THROUGHPUT_LABEL
        )

    payload = {
        "tool": "fws_cim_dse",
        "hardware_config": str(hw_path),
        "model_config": str(model_path),
        "mode": mode,
        "workload": workload,
        "throughput_unit": throughput_unit,
        "objective": objective,
        "selection_rule": selection_rule,
        "tp_candidates": [int(t) for t in tp_candidates],
        "moe_expert_parallel_candidates": [int(k) for k in moe_ep_candidates],
        "max_chips": int(dse_cfg.max_chips),
        "num_candidates": len(candidates),
        "num_valid": len(valid),
        "candidates": candidates,
        "front_ids": front_ids,
        "selected_id": None if selected is None else selected["id"],
        "selected": selected,
        "best_violation": None
        if best_violation is None
        else {
            "id": best_violation["id"],
            "fail_stage": best_violation["fail_stage"],
            "fail_message": best_violation["fail_message"],
        },
        "dse_echo": {
            "mux_candidates": mux_candidates,
            "variants": [
                _variant_analog_fields(v, base_hw.cim_config.analog) for v in dse_cfg.variants
            ],
            "tp_candidates": (
                "auto" if not isinstance(dse_cfg.tp_candidates, tuple) else list(dse_cfg.tp_candidates)
            ),
            "resolved_tp_candidates": [int(t) for t in tp_candidates],
            "moe_expert_parallel": [int(k) for k in moe_ep_candidates],
            "max_chips": int(dse_cfg.max_chips),
        },
        "verify": None,
    }

    exit_code = 0
    if selected is None:
        exit_code = 3
    else:
        if emit_config is None and verify:
            emit_config = output_dir / "selected_config.yaml"
        if emit_config is not None:
            emitted = emit_selected_config(base_raw, selected, emit_config)
            payload["emitted_config"] = str(emitted)
            if verify:
                verify_result = verify_selected(
                    emitted, model_path, mode, output_dir, selected, workload
                )
                payload["verify"] = verify_result
                if not verify_result["pass"]:
                    exit_code = 4

    json_path = output_dir / "dse_report.json"
    json_path.write_text(json.dumps(_json_safe(payload), indent=2))
    _write_markdown(output_dir / "dse_report.md", payload)
    payload["output_dir"] = str(output_dir)
    return exit_code, payload


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="fws_cim_dse",
        description=(
            "FWS-CIM design-space exploration: closed-form sweep over "
            "cim.dse array variants x tp x moe_expert_parallel on "
            "CimDeviceModel; chips derived via layers_per_chip: auto."
        ),
    )
    parser.add_argument(
        "--hardware_config",
        required=True,
        help="fws_cim hardware YAML with a cim.dse block (the candidate space).",
    )
    parser.add_argument("--model_config", required=True, help="Model YAML (LLM or VIT inference).")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Artifact directory (default output/fws_cim_dse/<hw-stem>__<model-stem>/).",
    )
    parser.add_argument(
        "--emit-config",
        default=None,
        help="Write a complete runnable fws_cim hardware YAML for the selected point.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Run run_perf once on the emitted config and cross-check period and "
            "throughput against the closed form (<= 0.1%%; nonzero exit on mismatch)."
        ),
    )
    parser.add_argument(
        "--objective",
        choices=sorted(SELECTION_RULES),
        default="throughput",
        help="Lexicographic selection objective (default: throughput).",
    )
    args = parser.parse_args(argv)

    try:
        exit_code, payload = run_dse(
            args.hardware_config,
            args.model_config,
            output_dir=args.output_dir,
            objective=args.objective,
            emit_config=args.emit_config,
            verify=args.verify,
        )
    except ValueError as exc:
        # DseUsageError subclasses ValueError; plain ValueErrors are the
        # config parsers' own errors (config.convert / HWConfig.from_dict /
        # the cim.dse schema), so both are usage/config errors: exit 2.
        print(f"[FWS-CIM DSE] error: {exc}")
        return 2

    out_dir = payload["output_dir"]
    print(f"[FWS-CIM DSE] {payload['num_valid']}/{payload['num_candidates']} candidates valid; "
          f"artifacts in {out_dir}")
    if payload["selected"] is None:
        best = payload["best_violation"]
        if best is not None:
            print(
                "[FWS-CIM DSE] no feasible candidate. Best violation "
                f"(candidate {best['id']}, stage {best['fail_stage']}): {best['fail_message']}"
            )
        else:
            print("[FWS-CIM DSE] no feasible candidate.")
        return exit_code
    selected = payload["selected"]
    m = selected["metrics"]
    print(
        f"[FWS-CIM DSE] selected {selected['id']} ({payload['selection_rule']}): "
        f"period {m['period_us']:.6g} us, throughput {m['throughput']:.6g} "
        f"{payload['throughput_unit']}, chips {m['chips']}, "
        f"area {m['area_mm2_total']:.6g} mm2"
    )
    if m.get("sustained_tokens_per_s") is not None:
        print(
            f"[FWS-CIM DSE] sustained {m['sustained_tokens_per_s']:.6g} tok/s "
            f"at KV capacity (wavefronts: full {m['wavefronts_full']}, "
            f"kv {m['wavefronts_kv']}; limited by {m['decode_throughput_limit']}); "
            "the throughput figure above is the fabric ceiling."
        )
    verify_result = payload.get("verify")
    if verify_result is not None:
        if verify_result["pass"]:
            print("[FWS-CIM DSE] verify: run_perf matches the closed form (<= 0.1%).")
        else:
            print(f"[FWS-CIM DSE] verify FAILED: {verify_result['message']}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
