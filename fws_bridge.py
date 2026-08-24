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

"""The bridge gate: the frozen closed form against the placed DAG — QIF P6.3.

A1 demoted the pass-1/2 spatial-pipeline report to a legacy reference and made
the DAG the model. ADJ-8 is the price of that move: before the closed form
stops being consulted, the DAG must REPRODUCE it on the degenerate case both
paths can express — one model, one owner per macro, no column sharing, no
bit-slicing, no PD split, tp = 1.

WHAT THIS MODULE IS
===================
A PROJECTION and a COMPARISON, and deliberately nothing else.

:func:`project` reads a priced :class:`~fws_eval.FwsEvaluation` and re-expresses
it in the closed form's vocabulary: named pipeline stages, one duration each, a
period, a bottleneck name. :func:`bridge_rows` holds that projection against a
``fws_cim_report.json`` and returns one :class:`BridgeRow` per comparison.

The projection is NOT a metric and must never become one. ``fws_eval`` publishes
timeline projections; this module publishes a steady-state re-reading of the same
priced ops, for the sole purpose of comparing against a report that assumed a
steady state. Two accountings of one metric never share an artifact (D21) — so
this one lives outside the artifact, in the validator.

THE PROJECTION RULE, STATED ONCE
================================
A closed-form *stage* is a set of ops the spatial pipeline runs at the same time
on dedicated hardware. So a stage's duration here is the MAXIMUM duration of the
DAG ops mapped to it, not their sum: in the degenerate case every analog op of a
stage sits on its own macro, so they are genuinely concurrent and the maximum is
what a filled pipeline sees.

The attention stage is the one place where the two paths compose the SAME law
outputs differently, and the bridge says so instead of averaging over it. P2's
attention law folds QK and PV onto the fabric arrays concurrently and runs the
softmax pipeline beside them; P3 places three DEPENDENT ops and P4 serializes
them (``fws_eval`` discloses this as ``attention_op_folding``). The bridge
compares the law OUTPUTS — the cycle counts — exactly, and reconstructs the
closed form's fold from the DAG's own op cycles through an identity that holds
by construction:

    closed-form folded stage = max(QK op cycles, PV op cycles + fill/drain,
                                   softmax op cycles)

because the DAG charges the array's fill/drain once, on the QK op. When that
identity fails, one of the two paths has changed its attention law, which is
exactly what the gate exists to catch.

WHAT THE GATE EXCLUDES, BY DESIGN
=================================
Every exclusion is named in :data:`EXCLUSIONS`, printed by the validator and
asserted by the tests, because an undisclosed exclusion is the AUDIT's finding 5
(the flagship rows that ran with the constraints switched off).

The excluded set is the closed form's blind spots, not the DAG's: fill/drain,
the attention fold above, the shared digital chiplet (D13 — the closed form
assumes a per-layer fabric), the per-macro pool ops the closed form absorbs into
its stages (D12), the ep dispatch hop the closed form charges even at ep = 1,
a KV-read-bound decode attention stage (a bandwidth story the DAG's fabric op
does not carry), contention, banking, bit-slicing, and the PD handoff. On every one of them the
DAG wins and the closed form carries the annotation (ADJ-8).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from fws_eval import FwsEvaluation
from program.fws_build import annotations_of

__all__ = [
    "BRIDGE_SCHEMA",
    "BridgeProjection",
    "BridgeRow",
    "BridgeStage",
    "EXCLUSIONS",
    "REL_TOL",
    "bridge_rows",
    "project",
]

#: One version field for the projection shape.
BRIDGE_SCHEMA = "fws_bridge/1"

#: The validator's own tolerance, reused verbatim: exact where the quantity is
#: discrete, 1e-3 relative where it is continuous (P6 §2).
REL_TOL = 1e-3

#: Closed-form stage name per DAG block, for a DENSE layer.
_DENSE_STAGE_OF_BLOCK: Mapping[str, str] = OrderedDict(
    (
        ("qkv", "S1_qkv"),
        ("attention_qk", "S2_attention"),
        ("attention_softmax", "S2_attention"),
        ("attention_pv", "S2_attention"),
        ("o_proj", "S3_o_proj"),
        ("ffn1", "S4_ffn1"),
        ("ffn2", "S5_ffn2"),
    )
)

#: Closed-form stage name per DAG block, for a MoE layer. Routed and shared
#: experts share one stage exactly as ``moe_ffn_stage_time`` = max(routed,
#: shared) does.
_MOE_STAGE_OF_BLOCK: Mapping[str, str] = OrderedDict(
    (
        ("qkv", "S1_qkv"),
        ("attention_qk", "S2_attention"),
        ("attention_softmax", "S2_attention"),
        ("attention_pv", "S2_attention"),
        ("o_proj", "S3_o_proj"),
        ("router", "S4_router"),
        ("ffn1_routed", "S5_ffn1_moe"),
        ("ffn1_shared", "S5_ffn1_moe"),
        ("ffn2_routed", "S6_ffn2_moe"),
        ("ffn2_shared", "S6_ffn2_moe"),
    )
)

#: The stage order the closed form's ``all_stage_times`` builds, which is also
#: its tie-break order: ``max(times, key=times.get)`` returns the FIRST maximum.
#: The projection builds the same order so a tie breaks to the same name.
_DENSE_ORDER = ("S1_qkv", "S2_attention", "S3_o_proj", "S4_ffn1", "S5_ffn2")
_MOE_ORDER = (
    "S1_qkv",
    "S2_attention",
    "S3_o_proj",
    "S4_router",
    "S5_ffn1_moe",
    "S6_ffn2_moe",
)

#: Endpoint blocks, in the closed form's own order.
_ENDPOINT_BLOCKS = ("patch_embed", "vit_head", "lm_head")

_ATTENTION_STAGE = "S2_attention"

#: Named, non-negotiable disclosures. The validator prints them; a test asserts
#: the tuple is non-empty and that every row of the gate names one of these when
#: it restricts what it compares.
EXCLUSIONS: Tuple[Tuple[str, str], ...] = (
    (
        "attention_fold",
        "P2's attention law runs QK and PV concurrently on the fabric arrays and the "
        "softmax pipeline beside them; P3 places three DEPENDENT ops and P4 serializes "
        "them. The gate compares the law's CYCLE COUNTS exactly and reconstructs the "
        "closed form's fold from the DAG's own op cycles; it never compares the two "
        "compositions (ADJ-8: the DAG wins, this is the annotation).",
    ),
    (
        "fabric_fill_drain",
        "The DAG charges the systolic array's fill/drain once, on the QK op. The gate "
        "names that penalty and removes it before comparing QK cycles, so the fold "
        "identity carries it instead of a tolerance absorbing it.",
    ),
    (
        "shared_digital_chiplet",
        "D13 puts act x act work on a shared digital chiplet per analog chip; the closed "
        "form assumes a dedicated per-layer fabric. Stage residency, chiplet occupancy "
        "and the analog->fabric hops have no closed-form counterpart and are not compared.",
    ),
    (
        "per_macro_pool_ops",
        "D12's pool ops — row-block partial-sum trees, norms, activations, residuals — are "
        "priced ops in the DAG and absorbed into the stage by the closed form (the OPTIMA "
        "sizing contract). They are excluded from the stage durations the gate compares.",
    ),
    (
        "ep_dispatch_at_ep_1",
        "The closed form charges a dispatch and a combine hop per MoE layer even at "
        "ep = 1, where the routed experts sit on the layer's own chip and nothing crosses "
        "a link. The DAG sends nothing. The gate subtracts the closed form's phantom term "
        "before comparing end-to-end latency and names the subtraction in the row.",
    ),
    (
        "energy_scope",
        "Only the analog-array term is bridged, prefill phase only — the scope the closed "
        "form's energy_partial_pj declares. The DAG's link and pool terms price hops the "
        "closed form has no notion of.",
    ),
    (
        "throughput_pair",
        "ADJ-6 retired the fabric-ceiling / sustained tokens-per-second pair from the DAG "
        "report. The gate compares the DECODE PERIOD the closed form derived it from; the "
        "pair itself is legacy and is never recomputed here.",
    ),
    (
        "kv_read_bound_decode",
        "When the closed form's decode attention is KV-READ bound rather than "
        "systolic-array bound, its stage time is a memory-bandwidth story the DAG's "
        "fabric op does not carry. The gate names the row and does not compare it: "
        "the gate goes red on DISAGREEMENT, never on inapplicability. No shipped "
        "bridge point is KV-read bound today; the row exists so a future one is a "
        "disclosed non-comparison instead of a failure.",
    ),
    (
        "degenerate_case_only",
        "Fill and drain of the SYSTEM pipeline, contention, column sharing, bit-slicing, "
        "banking and the PD handoff are absent from the degenerate case by construction. "
        "The closed form cannot express any of them; disagreement there is new content, "
        "not a failure (P6 §2).",
    ),
)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BridgeStage:
    """One closed-form stage, re-read off the priced DAG."""

    name: str
    duration_s: float
    macros: int
    ops: int
    basis: str


@dataclass(frozen=True)
class BridgeProjection:
    """The placed DAG in the closed form's vocabulary, for ONE phase/step."""

    schema: str
    label: str
    phase: str
    step: int
    dense_stages: "OrderedDict[str, BridgeStage]"
    moe_stages: "OrderedDict[str, BridgeStage]"
    endpoint_stages: "OrderedDict[str, BridgeStage]"
    qk_cycles: int
    pv_cycles: int
    softmax_cycles: int
    fold_cycles: int
    fill_drain_cycles: int
    num_dense_layers: int
    num_moe_layers: int
    analog_chips: int
    macros_total: int
    macros_stack: int
    boundary_times_s: Tuple[float, ...]
    analog_energy_pj: float
    area_stack_mm2: float
    area_total_mm2: float

    # -- derived, all from the fields above ------------------------------

    @property
    def stage_times_s(self) -> "OrderedDict[str, float]":
        """The closed form's ``all_stage_times`` shape, INCLUDING its key order.

        Dense stages, then ``moe_``-prefixed MoE stages, then endpoints — the
        order ``pipeline_period`` breaks ties in.
        """
        times: "OrderedDict[str, float]" = OrderedDict()
        for name, stage in self.dense_stages.items():
            times[name] = stage.duration_s
        for name, stage in self.moe_stages.items():
            times[f"moe_{name}"] = stage.duration_s
        for name, stage in self.endpoint_stages.items():
            times[name] = stage.duration_s
        return times

    @property
    def period_s(self) -> float:
        times = self.stage_times_s
        return times[self.bottleneck] if times else 0.0

    @property
    def bottleneck(self) -> str:
        times = self.stage_times_s
        return max(times, key=times.get) if times else ""

    @property
    def block_latency_s(self) -> float:
        """The closed form's ``block_latency``: the DENSE block, MoE as fallback."""
        if self.dense_stages:
            return sum(stage.duration_s for stage in self.dense_stages.values())
        return sum(stage.duration_s for stage in self.moe_stages.values())

    @property
    def end_to_end_s(self) -> float:
        """Layers x block + endpoints + chip boundaries.

        The closed form adds ``2 x dispatch`` per MoE layer on top; the DAG
        sends nothing at ep = 1, which the ``ep_dispatch_at_ep_1`` exclusion
        names and the comparison subtracts from the closed-form side.
        """
        dense_block = sum(stage.duration_s for stage in self.dense_stages.values())
        moe_block = sum(stage.duration_s for stage in self.moe_stages.values())
        return (
            self.num_dense_layers * dense_block
            + self.num_moe_layers * moe_block
            + sum(stage.duration_s for stage in self.endpoint_stages.values())
            + sum(self.boundary_times_s)
        )

    @property
    def macros_per_dense_layer(self) -> int:
        return sum(stage.macros for stage in self.dense_stages.values())

    @property
    def macros_per_moe_layer(self) -> int:
        return sum(stage.macros for stage in self.moe_stages.values())


@dataclass(frozen=True)
class BridgeRow:
    """One comparison. ``kind`` mirrors the validator's CheckTable methods."""

    name: str
    kind: str          # exact | close | bool
    expected: object
    measured: object
    ok: bool
    note: str = ""


# ---------------------------------------------------------------------------
# The projection
# ---------------------------------------------------------------------------


def _round_cycles(duration_s: float, f_hz: float) -> int:
    """Cycles behind a priced fabric op. The op was built as cycles / f."""
    return int(round(float(duration_s) * float(f_hz)))


def project(
    evaluation: FwsEvaluation,
    *,
    label: str = "",
    phase: str = "prefill",
    step: int = 0,
) -> BridgeProjection:
    """Re-read a priced DAG as the closed form's stage table.

    One representative layer per layer class carries the stage durations: in the
    degenerate case every layer of a class lowers to the same ops on the same
    shapes, and the closed form has exactly one stage table per class. The
    census fields (macros, chips, area, energy) span the whole program.
    """
    device = evaluation.mapping.device
    f_fabric = float(device.f_fabric_hz)
    fill_drain = int(device.fabric.fill_drain_penalty_cycles)
    annotations = {a.uid: a for a in annotations_of(evaluation.program)}

    mask = device.layer_class_mask()
    num_moe = sum(1 for flag in mask if flag)
    num_dense = len(mask) - num_moe
    dense_layer = next((i for i, flag in enumerate(mask) if not flag), None)
    moe_layer = next((i for i, flag in enumerate(mask) if flag), None)

    selected = [
        cost
        for cost in evaluation.pricing.costs
        if cost.phase == phase and int(cost.step) == int(step)
    ]

    dense_stages = _layer_stages(
        selected, dense_layer, _DENSE_STAGE_OF_BLOCK, _DENSE_ORDER, f_fabric, fill_drain
    )
    moe_stages = _layer_stages(
        selected, moe_layer, _MOE_STAGE_OF_BLOCK, _MOE_ORDER, f_fabric, fill_drain
    )
    endpoint_stages = _endpoint_stages(selected)

    attention_source = dense_stages or moe_stages
    qk = pv = softmax = fold = 0
    if _ATTENTION_STAGE in attention_source:
        qk, pv, softmax, fold = _attention_cycles(
            selected,
            dense_layer if dense_stages else moe_layer,
            f_fabric,
            fill_drain,
        )

    weight_costs = [
        cost for cost in evaluation.pricing.costs if cost.kind == "weight_gemm"
    ]
    macros_total = len({cost.macro_id for cost in weight_costs})
    macros_stack = len(
        {
            cost.macro_id
            for cost in weight_costs
            if cost.block not in _ENDPOINT_BLOCKS
        }
    )
    chips = len({cost.chip_id for cost in weight_costs})
    per_array = float(device.analog.area_mm2_per_array)

    return BridgeProjection(
        schema=BRIDGE_SCHEMA,
        label=label,
        phase=phase,
        step=int(step),
        dense_stages=dense_stages,
        moe_stages=moe_stages,
        endpoint_stages=endpoint_stages,
        qk_cycles=qk,
        pv_cycles=pv,
        softmax_cycles=softmax,
        fold_cycles=fold,
        fill_drain_cycles=fill_drain,
        num_dense_layers=num_dense,
        num_moe_layers=num_moe,
        analog_chips=chips,
        macros_total=macros_total,
        macros_stack=macros_stack,
        boundary_times_s=_boundary_times(selected, annotations),
        analog_energy_pj=sum(
            cost.energy_pj
            for cost in evaluation.pricing.costs
            if cost.phase == "prefill" and cost.energy_component == "analog_arrays"
        ),
        area_stack_mm2=macros_stack * per_array,
        area_total_mm2=macros_total * per_array,
    )


def _layer_stages(
    costs: Sequence[object],
    layer: Optional[int],
    stage_of_block: Mapping[str, str],
    order: Sequence[str],
    f_fabric: float,
    fill_drain: int,
) -> "OrderedDict[str, BridgeStage]":
    """Stage table for one representative layer, or empty when the class is absent."""
    stages: "OrderedDict[str, BridgeStage]" = OrderedDict()
    if layer is None:
        return stages
    grouped: Dict[str, List[object]] = {}
    for cost in costs:
        if cost.layer != layer or cost.block not in stage_of_block:
            continue
        # D12's pool ops and D17's transfers are named exclusions: the closed
        # form absorbs the first into its stages and does not model the second.
        if cost.kind not in ("weight_gemm", "fabric"):
            continue
        grouped.setdefault(stage_of_block[cost.block], []).append(cost)
    for name in order:
        members = grouped.get(name)
        if not members:
            continue
        if name == _ATTENTION_STAGE:
            qk, pv, softmax, fold = _attention_cycles_from(members, f_fabric, fill_drain)
            stages[name] = BridgeStage(
                name=name,
                duration_s=fold / f_fabric,
                macros=0,
                ops=len(members),
                basis=(
                    "P2's folded attention stage, rebuilt from the DAG's OWN op cycles: "
                    f"max(QK {qk + fill_drain}, PV {pv} + fill/drain {fill_drain}, "
                    f"softmax {softmax}) = {fold} cycles (exclusion: attention_fold)"
                ),
            )
            continue
        stages[name] = BridgeStage(
            name=name,
            duration_s=max(float(cost.duration_s) for cost in members),
            macros=len({cost.macro_id for cost in members}),
            ops=len(members),
            basis=(
                "the longest of the stage's concurrent analog ops; in the degenerate "
                "case each sits on its own macro, so a filled pipeline sees the maximum"
            ),
        )
    return stages


def _endpoint_stages(costs: Sequence[object]) -> "OrderedDict[str, BridgeStage]":
    stages: "OrderedDict[str, BridgeStage]" = OrderedDict()
    for block in _ENDPOINT_BLOCKS:
        members = [
            cost for cost in costs if cost.block == block and cost.kind == "weight_gemm"
        ]
        if not members:
            continue
        stages[block] = BridgeStage(
            name=block,
            duration_s=max(float(cost.duration_s) for cost in members),
            macros=len({cost.macro_id for cost in members}),
            ops=len(members),
            basis="the longest of the endpoint's concurrent analog ops",
        )
    return stages


def _attention_cycles_from(
    members: Sequence[object], f_fabric: float, fill_drain: int
) -> Tuple[int, int, int, int]:
    """(qk, pv, softmax, folded) cycles from the three priced attention ops.

    The QK op carries the array's fill/drain, so the QK LAW output is the op's
    cycles minus that penalty — named, not absorbed into a tolerance.
    """
    cycles = {
        cost.block: _round_cycles(cost.duration_s, f_fabric) for cost in members
    }
    qk_op = cycles.get("attention_qk", 0)
    pv = cycles.get("attention_pv", 0)
    softmax = cycles.get("attention_softmax", 0)
    qk = qk_op - fill_drain
    fold = max(qk_op, pv + fill_drain, softmax)
    return qk, pv, softmax, fold


def _attention_cycles(
    costs: Sequence[object], layer: Optional[int], f_fabric: float, fill_drain: int
) -> Tuple[int, int, int, int]:
    members = [
        cost
        for cost in costs
        if cost.layer == layer and cost.block.startswith("attention_")
    ]
    return _attention_cycles_from(members, f_fabric, fill_drain)


def _boundary_times(
    costs: Sequence[object], annotations: Mapping[int, object]
) -> Tuple[float, ...]:
    """Chip-to-chip activation hops, in chip order — the closed form's boundaries."""
    rows = []
    for cost in costs:
        annotation = annotations.get(cost.uid)
        if annotation is None or annotation.kind != "transfer":
            continue
        boundary = str(annotation.boundary_id)
        if not boundary.startswith("act.c") or "->c" not in boundary:
            continue
        rows.append((int(annotation.chip_id), float(cost.duration_s)))
    rows.sort()
    return tuple(duration for _chip, duration in rows)


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


def _exact(rows: List[BridgeRow], name: str, expected, measured, note: str = "") -> None:
    rows.append(BridgeRow(name, "exact", expected, measured, measured == expected, note))


def _close(
    rows: List[BridgeRow], name: str, expected, measured, note: str = "", rel_tol=REL_TOL
) -> None:
    try:
        rel = abs(float(measured) - float(expected)) / abs(float(expected))
        ok = rel <= rel_tol
    except (TypeError, ValueError, ZeroDivisionError):
        ok = False
    rows.append(BridgeRow(name, "close", expected, measured, ok, note))


def _bool(rows: List[BridgeRow], name: str, description, ok: bool, detail="") -> None:
    rows.append(BridgeRow(name, "bool", description, detail, bool(ok), ""))


def bridge_rows(
    evaluation: FwsEvaluation, closed_form: Mapping[str, object], label: str
) -> Tuple[BridgeRow, ...]:
    """Every P6.3 comparison for one config, prefill and (when present) decode.

    ``closed_form`` is a parsed ``fws_cim_report.json`` produced by the SAME
    hardware and model YAMLs with the ``mapping:`` block absent.
    """
    rows: List[BridgeRow] = []
    projection = project(evaluation, label=label)
    _prefill_rows(rows, projection, closed_form, label)
    decode = closed_form.get("decode")
    if decode and decode.get("contexts"):
        _decode_rows(rows, evaluation, decode, label)
    return tuple(rows)


def _prefill_rows(
    rows: List[BridgeRow],
    projection: BridgeProjection,
    closed_form: Mapping[str, object],
    label: str,
) -> None:
    cf_stages = closed_form.get("stages") or {}
    cf_moe_stages = closed_form.get("moe_stages") or {}
    cf_endpoints = closed_form.get("endpoint_stages") or {}

    # --- discrete: the op and stage SETS -------------------------------
    _exact(
        rows,
        "%s: dense stage set" % label,
        sorted(cf_stages),
        sorted(projection.dense_stages),
    )
    _exact(
        rows,
        "%s: MoE stage set" % label,
        sorted(cf_moe_stages),
        sorted(projection.moe_stages),
    )
    _exact(
        rows,
        "%s: endpoint stage set" % label,
        sorted(cf_endpoints),
        sorted(projection.endpoint_stages),
    )
    _exact(
        rows,
        "%s: layer classes" % label,
        closed_form["layer_classes"],
        {"dense": projection.num_dense_layers, "moe": projection.num_moe_layers},
    )

    # --- discrete: the array census, stage by stage --------------------
    for name, entry in cf_stages.items():
        _exact(
            rows,
            "%s: dense %s macros" % (label, name),
            int(entry["arrays_per_layer"]),
            projection.dense_stages[name].macros,
        )
    for name, entry in cf_moe_stages.items():
        _exact(
            rows,
            "%s: MoE %s macros" % (label, name),
            int(entry["arrays_per_layer"]),
            projection.moe_stages[name].macros,
        )
    for name, entry in cf_endpoints.items():
        _exact(
            rows,
            "%s: endpoint %s macros" % (label, name),
            int(entry["arrays"]),
            projection.endpoint_stages[name].macros,
        )
    if cf_stages:
        _exact(
            rows,
            "%s: macros per dense layer" % label,
            int(closed_form["arrays_per_layer"]),
            projection.macros_per_dense_layer,
        )
    if cf_moe_stages:
        _exact(
            rows,
            "%s: macros per MoE layer" % label,
            int(closed_form["arrays_per_moe_layer"]),
            projection.macros_per_moe_layer,
        )
    _exact(
        rows,
        "%s: transformer-stack macros" % label,
        int(closed_form["arrays_transformer_stack"]),
        projection.macros_stack,
    )
    _exact(
        rows,
        "%s: total analog macros" % label,
        int(closed_form["arrays_total"]),
        projection.macros_total,
    )
    _exact(
        rows,
        "%s: analog chips (integers, D21)" % label,
        len(closed_form["chips"]),
        projection.analog_chips,
    )

    # --- discrete: the attention law's own outputs ---------------------
    attention = (cf_stages or cf_moe_stages).get(_ATTENTION_STAGE, {})
    _exact(
        rows,
        "%s: QK cycles" % label,
        int(closed_form["qk_cycles"]),
        projection.qk_cycles,
        "exclusion fabric_fill_drain: the DAG's QK op carries "
        "%d fill/drain cycles, removed here" % projection.fill_drain_cycles,
    )
    _exact(
        rows,
        "%s: PV cycles" % label,
        int(closed_form["sv_cycles"]),
        projection.pv_cycles,
    )
    if "softmax_cycles" in attention:
        _exact(
            rows,
            "%s: softmax cycles" % label,
            int(attention["softmax_cycles"]),
            projection.softmax_cycles,
        )
    if "fabric_total_cycles" in attention:
        _exact(
            rows,
            "%s: attention fold identity" % label,
            int(attention["fabric_total_cycles"]),
            projection.fold_cycles,
            "exclusion attention_fold: max(QK op, PV op + fill/drain, softmax op)",
        )

    # --- discrete: the bottleneck is a NAME ----------------------------
    _exact(
        rows,
        "%s: bottleneck stage" % label,
        closed_form["bottleneck_stage"],
        projection.bottleneck,
    )

    # --- continuous: stage durations, period, latencies ----------------
    for name, entry in cf_stages.items():
        _close(
            rows,
            "%s: dense %s time_us" % (label, name),
            entry["time_us"],
            projection.dense_stages[name].duration_s * 1e6,
        )
    for name, entry in cf_moe_stages.items():
        _close(
            rows,
            "%s: MoE %s time_us" % (label, name),
            entry["time_us"],
            projection.moe_stages[name].duration_s * 1e6,
        )
    for name, entry in cf_endpoints.items():
        _close(
            rows,
            "%s: endpoint %s time_us" % (label, name),
            entry["time_us"],
            projection.endpoint_stages[name].duration_s * 1e6,
        )
    _close(
        rows,
        "%s: period_us (period-equivalent)" % label,
        closed_form["period_us"],
        projection.period_s * 1e6,
    )
    _close(
        rows,
        "%s: fps (1 / period)" % label,
        closed_form["fps"],
        1.0 / projection.period_s if projection.period_s > 0 else float("inf"),
    )
    _close(
        rows,
        "%s: block latency_us" % label,
        closed_form["block_latency_us"],
        projection.block_latency_s * 1e6,
    )

    dispatch_us = 0.0
    moe = closed_form.get("moe") or {}
    if moe:
        dispatch = moe.get("dispatch") or {}
        dispatch_us = float(dispatch.get("time_us_each_way", 0.0)) * int(
            dispatch.get("count", 0)
        )
    _close(
        rows,
        "%s: end-to-end latency_us" % label,
        closed_form["end_to_end_latency_us"] - dispatch_us,
        projection.end_to_end_s * 1e6,
        "exclusion ep_dispatch_at_ep_1: %.6g us of closed-form dispatch/combine "
        "subtracted" % dispatch_us,
    )

    # --- continuous: boundaries, area, energy --------------------------
    boundary = closed_form["boundary"]
    _exact(
        rows,
        "%s: chip boundaries" % label,
        int(boundary["count"]),
        len(projection.boundary_times_s),
    )
    for index, expected_us in enumerate(boundary["times_us"]):
        _close(
            rows,
            "%s: boundary %d time_us" % (label, index),
            expected_us,
            projection.boundary_times_s[index] * 1e6,
        )
    area = closed_form.get("area_mm2") or {}
    if area:
        _close(
            rows,
            "%s: transformer-stack area mm2" % label,
            area["transformer_stack"],
            projection.area_stack_mm2,
        )
        _close(
            rows,
            "%s: total analog area mm2" % label,
            area["total"],
            projection.area_total_mm2,
        )
    energy = closed_form["energy_partial_pj"]
    _close(
        rows,
        "%s: analog energy pJ (prefill)" % label,
        energy["analog_stack"] + energy["analog_endpoints"],
        projection.analog_energy_pj,
        "exclusion energy_scope: analog arrays only, the closed form's own scope",
    )


def _decode_rows(
    rows: List[BridgeRow],
    evaluation: FwsEvaluation,
    decode: Mapping[str, object],
    label: str,
) -> None:
    """Decode step 0 against the closed form's FIRST recorded context.

    The DAG's step 0 runs at context ``prefill_len + 1``, which is exactly the
    context the closed form labels ``first``. Later steps leave the lowered
    window at different contexts and are not compared here.
    """
    first = decode["contexts"][0]
    serving = evaluation.serving
    if serving.decode_steps <= 0:
        return
    _exact(
        rows,
        "%s decode: context of step 0" % label,
        int(first["context"]),
        int(serving.prefill_len) + 1,
    )
    s2 = first.get("s2") or {}
    if str(s2.get("bound", "sa")) != "sa":
        # The closed form's decode attention is KV-read bound here, which is a
        # bandwidth story the DAG's fabric op does not carry. This is an
        # INAPPLICABLE comparison, not a disagreement, so it is a named
        # exclusion (like fabric_fill_drain and energy_scope) rather than a red
        # gate: a legitimate configuration must not fail the validator.
        _bool(
            rows,
            "%s decode: attention stage comparability" % label,
            "exclusion kv_read_bound_decode: closed-form S2 bound is "
            "%r, not 'sa', so the stage table is NOT compared" % s2.get("bound"),
            True,
            "bound=%s" % s2.get("bound"),
        )
        return
    projection = project(evaluation, label=label, phase="decode", step=0)
    expected = first["stages_us"]
    measured = projection.stage_times_s
    _exact(
        rows,
        "%s decode: stage set" % label,
        sorted(expected),
        sorted(measured),
    )
    for name, expected_us in expected.items():
        _close(
            rows,
            "%s decode: %s time_us" % (label, name),
            expected_us,
            measured[name] * 1e6,
        )
    _close(
        rows,
        "%s decode: period_us" % label,
        first["period_us"],
        projection.period_s * 1e6,
        "exclusion throughput_pair: the tokens/s pair ADJ-6 retired is derived "
        "from this period and is not recomputed",
    )
    _exact(
        rows,
        "%s decode: bottleneck stage" % label,
        first["bottleneck_stage"],
        projection.bottleneck,
    )
    _exact(
        rows,
        "%s decode: QK cycles" % label,
        int(s2["qk_cycles"]),
        projection.qk_cycles,
    )
    _exact(
        rows,
        "%s decode: PV cycles" % label,
        int(s2["pv_cycles"]),
        projection.pv_cycles,
    )
    _exact(
        rows,
        "%s decode: attention fold identity" % label,
        int(s2["total_cycles"]),
        projection.fold_cycles,
    )
