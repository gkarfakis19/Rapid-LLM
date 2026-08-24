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

"""The PLACED OP DAG for an FWS-CIM inference workload — QIF P3.2.

``build_fws_program`` lowers one :class:`fws_mapping.FwsMapping` plus a serving
point into the rewrite's typed program IR: ``ComputeOp`` on a device,
``TransferOp`` on a link, explicit deps, one global order. It is a SECOND
builder standing beside :func:`program.build.build`, never a change to it — the
GPU program path and its 221 golden-equivalence specs are untouched by
construction, because nothing here is imported from there.

What lands where (P3 §3, A2, D12, D13)::

    weight GEMM (qkv, o_proj, ffn1/2, router, experts,        analog macro
      endpoints, SSM / linear-attn / short-conv projections)     device
    bit-slice + row-block shift-and-add reduction             that macro's pool
    norm, activation, gating, residual, short depthwise conv  that macro's pool
    attention QK^T / softmax / PV, every act x act GEMM,      shared digital
      SSM scan and state update, delta rule                     chiplet
    boundary activations, tp partials, expert dispatch,       p2p link
      PD handoff

**Every duration is 0.0 and that is the deliverable, not a gap.** P3 places and
P4 prices (A1; P4 §1 "One op, one device, one duration, one law source. The
mapper states the assignment; P4 asks P2 for the number"). A duration written
here would be a second accounting of a metric P4 owns (D21). What every op DOES
carry is the annotation P4's pricing table needs, in
:class:`FwsOpAnnotation`: the device and its class, the resident tiles, the
token count, the transferred bytes, the owner, the shard group, and the NAME OF
THE LAW that prices it — the row of P4 §1 this op belongs to.

The decode series is a BOUNDED WINDOW (ADJ-6): ``decode_steps`` steps are
lowered and the truncation is disclosed by the mapping's relaxation list. The
DAG never extrapolates.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from cim_timing import Tile, TileOwner
from fws_mapping import (
    MAPPING_AXES,
    FwsMapping,
    MappingError,
    ShardCoord,
)
from program.ir import GroupKey, Program, ProgramBuilder, ProgramMeta
from program.layout import RankLayout

#: The pricing-table rows of P4 §1, verbatim. An op names the law that prices
#: it; P4 dispatches on this string and never on an op name.
LAW_ANALOG_GEMM = "P2 analog vector law at the tile's column set"
LAW_SLICE_REDUCTION = "P2 priced reduction (D11)"
LAW_POOL = "P2 pool component costs (D12)"
LAW_FABRIC = "P2 fabric laws (D13)"
LAW_LINK = "bytes / link law (D17)"
LAW_PD_LINK = "bytes / link law (D16)"
LAW_COLLECTIVE = "analytical collective over the p2p law"

#: Set on every ComputeOp/TransferOp; P4 replaces it with a priced timeline.
UNPRICED = 0.0
DURATION_BASIS = (
    "unpriced: P3 places, P4 prices (A1). Every duration is 0.0 by construction; "
    "the annotation carries the inputs P4's pricing table needs."
)


@dataclass(frozen=True)
class FwsOpAnnotation:
    """Everything P4 needs to price one op, and nothing it must re-derive.

    ``law`` is the row of P4 §1's pricing contract this op belongs to;
    ``device_class`` is the resource it occupies (``link`` means it occupies
    none — D17 prices transfers, it does not schedule them).
    """

    uid: int
    kind: str                       # weight_gemm | reduction | pool | fabric | transfer
    law: str
    device_class: str
    device_id: int
    chip_id: int
    macro_id: int
    phase: str                      # prefill | decode
    step: int                       # decode step index; 0 for prefill
    layer: Optional[int]
    block: str
    tokens: float
    owner: Optional[TileOwner] = None
    tiles: Tuple[Tile, ...] = ()
    bytes_moved: float = 0.0
    src_device: int = -1
    dst_device: int = -1
    boundary_id: str = ""
    #: False when both endpoints sit on one chip. D17's p2p law prices the
    #: fabric between chips; movement inside a chip is a DEPENDENCY here, and
    #: the mapping discloses that as a relaxation rather than pricing it with a
    #: law nobody declared.
    crosses_chip: bool = True
    groups: Mapping[str, int] = field(default_factory=dict)
    primary_group: str = "tp"
    group_keys: Mapping[str, GroupKey] = field(default_factory=dict)
    note: str = ""


@dataclass
class ServingPoint:
    """The serving semantics the DAG lowers (D15).

    A batch of same-length requests: each prefills ``prefill_len`` tokens and
    decodes ``decode_len``. ``decode_steps`` is the lowered WINDOW (ADJ-6).
    """

    batch: int
    prefill_len: int
    decode_len: int
    decode_steps: int

    @classmethod
    def from_mapping(cls, mapping: FwsMapping, decode_steps: Optional[int] = None) -> "ServingPoint":
        params = mapping.device.params
        decode_len = int(getattr(mapping.model, "decode_len", 0) or 0)
        prefill_len = max(0, int(params.seq_len) - decode_len)
        window = mapping.decode_window if decode_steps is None else int(decode_steps)
        return cls(
            batch=int(params.batch_size),
            prefill_len=prefill_len,
            decode_len=decode_len,
            decode_steps=max(0, min(window, decode_len)),
        )


class _Lowering:
    """One pass over the mapping, one Program out."""

    def __init__(self, mapping: FwsMapping, serving: ServingPoint, label: str) -> None:
        self.mapping = mapping
        self.serving = serving
        self.builder = ProgramBuilder(
            layout=RankLayout((), {}, {}),
            dp_count=1,
            meta=ProgramMeta(label=label),
        )
        self.annotations: List[FwsOpAnnotation] = []
        self.act_bytes = float(mapping.hw.sw_config.precision.activations)
        self.groups: Dict[GroupKey, str] = OrderedDict()
        # tiles grouped by owner, by chip, in placement order
        self._owner_tiles: "OrderedDict[TileOwner, Tuple[Tile, ...]]" = OrderedDict()
        for tile in mapping.tiles:
            self._owner_tiles.setdefault(tile.owner, [])
            self._owner_tiles[tile.owner].append(tile)
        self._owner_tiles = OrderedDict(
            (owner, tuple(tiles)) for owner, tiles in self._owner_tiles.items()
        )

    # -- helpers ---------------------------------------------------------

    def _shard_index(self, shard: ShardCoord) -> int:
        return self.mapping.shard_id(shard)

    def _group_keys(self, device_id: int) -> Dict[str, GroupKey]:
        keys = {}
        for axis in MAPPING_AXES:
            if int(self.mapping.degrees.get(axis, 1)) <= 1:
                continue
            key = self.mapping.device_group(axis, device_id)
            keys[axis] = key
            label = f"{axis}{list(key.members)}"
            if key not in self.groups:
                self.groups[key] = label
                self.builder.group(key.axis, key.members, label)
        return keys

    def _annotate(self, uid: int, annotation: FwsOpAnnotation) -> int:
        assert uid == len(self.annotations), "annotations are uid-indexed"
        self.annotations.append(annotation)
        return uid

    def compute(
        self,
        name: str,
        device_id: int,
        deps: Sequence[int],
        *,
        kind: str,
        law: str,
        phase: str,
        step: int,
        layer: Optional[int],
        block: str,
        tokens: float,
        owner: Optional[TileOwner] = None,
        tiles: Sequence[Tile] = (),
        note: str = "",
    ) -> int:
        dev = self.mapping.device_record(device_id)
        uid = self.builder.add_compute(
            name, device_id, UNPRICED, deps=tuple(sorted(set(int(d) for d in deps))), layer=layer
        )
        return self._annotate(
            uid,
            FwsOpAnnotation(
                uid=uid,
                kind=kind,
                law=law,
                device_class=dev.device_class,
                device_id=device_id,
                chip_id=dev.chip_id,
                macro_id=dev.macro_id,
                phase=phase,
                step=step,
                layer=layer,
                block=block,
                tokens=float(tokens),
                owner=owner,
                tiles=tuple(tiles),
                groups=dict(dev.shard.as_dict()),
                primary_group=self.mapping.primary_group,
                group_keys=self._group_keys(device_id),
                note=note,
            ),
        )

    def transfer(
        self,
        name: str,
        producer: int,
        dst_device: int,
        size_bytes: float,
        *,
        boundary_id: str,
        law: str = LAW_LINK,
        phase: str,
        step: int,
        layer: Optional[int],
        block: str,
        tokens: float,
        note: str = "",
    ) -> int:
        src_device = self.annotations[producer].device_id
        src_chip = self.mapping.device_record(src_device).chip_id
        dst_chip = self.mapping.device_record(dst_device).chip_id
        uid = self.builder.add_transfer(
            name,
            src_device,
            dst_device,
            int(size_bytes),
            producer,
            participants=2,
            interconnect="pp",
        )
        dev = self.mapping.device_record(src_device)
        return self._annotate(
            uid,
            FwsOpAnnotation(
                uid=uid,
                kind="transfer",
                law=law,
                device_class="link",
                device_id=-1,
                chip_id=dev.chip_id,
                macro_id=-1,
                phase=phase,
                step=step,
                layer=layer,
                block=block,
                tokens=float(tokens),
                bytes_moved=float(size_bytes),
                src_device=src_device,
                dst_device=dst_device,
                boundary_id=boundary_id,
                crosses_chip=src_chip != dst_chip,
                groups=dict(dev.shard.as_dict()),
                primary_group=self.mapping.primary_group,
                group_keys=self._group_keys(src_device),
                note=note,
            ),
        )

    # -- the lowering ----------------------------------------------------

    def run(self) -> Program:
        serving = self.serving
        tail: List[int] = []
        if serving.prefill_len > 0:
            # The first decode step CONTINUES the request the prefill started
            # (D15: each request prefills S tokens and then decodes N), so the
            # prefill's tail seeds decode step 0 exactly as step k-1 seeds step
            # k. Dropping the edge would make prefill and decode independent
            # subgraphs racing for the same devices, and P4's "finish(step k) -
            # finish(step k-1)" projection would then read NEGATIVE on a
            # timeline where a prefill op outlives the decode chain.
            tail = self._phase(
                "prefill", 0, float(serving.batch * serving.prefill_len), []
            )
        for step in range(serving.decode_steps):
            tail = self._phase("decode", step, float(serving.batch), tail)
        devices = tuple(sorted(dev.device_id for dev in self.mapping.devices))
        program = self.builder.finish(devices=devices, validate=True)
        program.meta.misc["fws_annotations"] = tuple(self.annotations)
        program.meta.misc["fws_mapping"] = self.mapping
        program.meta.misc["fws_serving"] = self.serving
        program.meta.misc["fws_duration_basis"] = DURATION_BASIS
        program.meta.misc["fws_shard_layout"] = self.mapping.shard_layout
        return program

    def _phase(self, phase: str, step: int, tokens: float, seed: Sequence[int]) -> List[int]:
        """One prefill pass or one decode step, over every tp shard.

        Each tp shard runs its OWN chain: its own norms, its own attention on
        its own shared chiplet, its own FFN. The shards meet only where the
        row-parallel stages need a tp all-reduce, which is an explicit
        transfer. Lowering one shard and reusing its pool ops for the others
        would place work that no device does.
        """
        tails: List[int] = []
        for tp_idx in range(max(1, int(self.mapping.degrees.get("tp", 1)))):
            tails.extend(self._shard_phase(tp_idx, phase, step, tokens, seed))
        return tails

    def _shard_phase(
        self, tp_idx: int, phase: str, step: int, tokens: float, seed: Sequence[int]
    ) -> List[int]:
        mapping = self.mapping
        p = mapping.device.params
        pending: List[int] = list(seed)
        # endpoints that lead: the ViT patch embedding
        pending = self._endpoint(tp_idx, phase, step, tokens, pending, where="head")
        previous_chip: Optional[int] = None
        for layer in range(p.num_layers):
            chip_id = self._layer_chip(layer, tp_idx)
            if previous_chip is not None and chip_id != previous_chip:
                pending = self._chip_boundary(
                    previous_chip, chip_id, pending, phase, step, tokens, layer
                )
            pending = self._layer(layer, chip_id, tp_idx, pending, phase, step, tokens)
            previous_chip = chip_id
        pending = self._endpoint(tp_idx, phase, step, tokens, pending, where="tail")
        return pending

    def _layer_chip(self, layer: int, tp_idx: int) -> int:
        mapping = self.mapping
        for chip_id in mapping.layer_chip.get(layer, ()):
            chip = mapping.chip(chip_id)
            if chip.role == "backbone" and int(chip.shard.tp) == int(tp_idx):
                return chip_id
        raise MappingError(
            "execution", f"layer {layer} is on no backbone chip of tp shard {tp_idx}"
        )

    def _owner_tp(self, owner: TileOwner) -> int:
        return int(self.mapping.shard_of_id(int(owner.shard)).tp)

    def _endpoint(
        self, tp_idx: int, phase: str, step: int, tokens: float, deps: Sequence[int], *, where: str
    ) -> List[int]:
        mapping = self.mapping
        blocks = {"head": ("patch_embed",), "tail": ("lm_head", "vit_head")}[where]
        out: List[int] = list(deps)
        for owner, tiles in self._owner_tiles.items():
            if owner.op not in blocks or self._owner_tp(owner) != tp_idx:
                continue
            chip_id = mapping.chip(mapping.macro(tiles[0].site.macro_id).chip_id).chip_id
            produced = self._weight_stage(
                owner,
                tiles,
                chip_id,
                out,
                phase,
                step,
                self._endpoint_tokens(owner.op, tokens),
                layer=owner.layer,
            )
            out = produced
        return out

    def _endpoint_tokens(self, op: str, tokens: float) -> float:
        """M for an endpoint GEMM, per P2's ``endpoint_stage_times``.

        The ViT classification head reads ONE pooled token per image, so its M
        is the batch, not the sequence: P2 prices it ``analog_gemm_time(B)``
        while ``patch_embed`` and the LLM ``lm_head`` see every token. Lowering
        the head over the whole sequence charged it S times its work — caught
        by the P6.3 bridge on T1/T2/T3, where it moved analog energy by the
        cost of one full-sequence array pass.
        """
        if op != "vit_head":
            return tokens
        return float(self.serving.batch)

    def _chip_boundary(
        self,
        src_chip: int,
        dst_chip: int,
        deps: Sequence[int],
        phase: str,
        step: int,
        tokens: float,
        layer: int,
    ) -> List[int]:
        """The activation handoff across a chip boundary (role `act`, ADJ-5)."""
        mapping = self.mapping
        if not deps:
            return list(deps)
        producer = deps[-1]
        dst_macro = next(
            m for m in mapping.macros if m.chip_id == dst_chip and m.pool == "analog"
        )
        size = tokens * float(mapping.device.params.hidden_dim) * self.act_bytes
        uid = self.transfer(
            f"{phase}{step}.boundary.c{src_chip}->c{dst_chip}",
            producer,
            dst_macro.pool_device,
            size,
            boundary_id=f"act.c{src_chip}->c{dst_chip}",
            phase=phase,
            step=step,
            layer=layer,
            block="boundary",
            tokens=tokens,
            note="chip-boundary activation handoff; a chip boundary is never called pp (ADJ-5)",
        )
        return [uid]

    def _weight_stage(
        self,
        owner: TileOwner,
        tiles: Sequence[Tile],
        chip_id: int,
        deps: Sequence[int],
        phase: str,
        step: int,
        tokens: float,
        layer: Optional[int],
    ) -> List[int]:
        """One weight matrix: macro GEMM ops, then the reductions they need."""
        mapping = self.mapping
        prefix = f"{phase}{step}.L{layer}.{owner.op}"
        if owner.expert >= 0:
            prefix += f".e{owner.expert}"
        by_column: "OrderedDict[int, List[Tile]]" = OrderedDict()
        for tile in tiles:
            by_column.setdefault(tile.n_start, []).append(tile)
        outputs: List[int] = []
        for n_start, column_tiles in by_column.items():
            by_macro: "OrderedDict[int, List[Tile]]" = OrderedDict()
            for tile in column_tiles:
                by_macro.setdefault(tile.site.macro_id, []).append(tile)
            partials: List[int] = []
            for macro_id, macro_tiles in by_macro.items():
                macro = mapping.macro(macro_id)
                uid = self.compute(
                    f"{prefix}.n{n_start}.m{macro_id}",
                    macro.analog_device,
                    deps,
                    kind="weight_gemm",
                    law=LAW_ANALOG_GEMM,
                    phase=phase,
                    step=step,
                    layer=layer,
                    block=owner.op,
                    tokens=tokens,
                    owner=owner,
                    tiles=macro_tiles,
                )
                slices = {tile.slice_index for tile in macro_tiles}
                if len(slices) > 1:
                    # D11: the shift-and-add tree is a REAL op on this macro's
                    # pool, never absorbed into the analog charge.
                    uid = self.compute(
                        f"{prefix}.n{n_start}.m{macro_id}.shift_add",
                        macro.pool_device,
                        [uid],
                        kind="reduction",
                        law=LAW_SLICE_REDUCTION,
                        phase=phase,
                        step=step,
                        layer=layer,
                        block=owner.op,
                        tokens=tokens,
                        owner=owner,
                        tiles=macro_tiles,
                        note=f"{len(slices)} bit slices composed on the host macro's pool",
                    )
                partials.append(uid)
            if len(partials) > 1:
                # Several ROW blocks of one output block: real partial sums.
                sink = mapping.macro(next(iter(by_macro)))
                width = max(tile.logical_columns for tile in column_tiles)
                moved = []
                for uid in partials[1:]:
                    moved.append(
                        self.transfer(
                            f"{prefix}.n{n_start}.partial{uid}",
                            uid,
                            sink.pool_device,
                            tokens * width * self.act_bytes,
                            boundary_id=f"partial.L{layer}.{owner.op}.n{n_start}",
                            phase=phase,
                            step=step,
                            layer=layer,
                            block=owner.op,
                            tokens=tokens,
                            note="row-block partial sum crossing to the sink macro's pool",
                        )
                    )
                outputs.append(
                    self.compute(
                        f"{prefix}.n{n_start}.rowsum",
                        sink.pool_device,
                        [partials[0]] + moved,
                        kind="reduction",
                        law=LAW_POOL,
                        phase=phase,
                        step=step,
                        layer=layer,
                        block=owner.op,
                        tokens=tokens,
                        owner=owner,
                        tiles=column_tiles,
                        note=f"{len(partials)} row-block partials summed",
                    )
                )
            else:
                outputs.extend(partials)
        return outputs

    def _pool_op(
        self, name: str, macro_id: int, deps: Sequence[int], *, block: str, **kwargs
    ) -> int:
        macro = self.mapping.macro(macro_id)
        return self.compute(
            name,
            macro.pool_device,
            deps,
            kind="pool",
            law=LAW_POOL,
            block=block,
            **kwargs,
        )

    def _fabric_op(
        self, name: str, chip_id: int, deps: Sequence[int], *, block: str, **kwargs
    ) -> int:
        shard = self.mapping.chip(chip_id).shard
        engines = self.mapping.shared_digital_devices(shard)
        if not engines:
            raise MappingError(
                "execution",
                "an act x act op needs a shared digital chiplet (D13) and the mapping "
                "declares none.",
            )
        engine = engines[chip_id % len(engines)]
        return self.compute(
            name,
            engine.device_id,
            deps,
            kind="fabric",
            law=LAW_FABRIC,
            block=block,
            **kwargs,
        )

    def _tp_exchange(
        self,
        name: str,
        producer: int,
        deps: Sequence[int],
        *,
        phase: str,
        step: int,
        layer: Optional[int],
        block: str,
        tokens: float,
    ) -> List[int]:
        """The tp all-reduce of a row-parallel stage, as explicit p2p bytes."""
        mapping = self.mapping
        tp = int(mapping.degrees.get("tp", 1))
        if tp <= 1:
            return [producer]
        src_device = self.annotations[producer].device_id
        key = mapping.device_group("tp", src_device)
        payload = tokens * float(mapping.device.params.hidden_dim) * self.act_bytes
        volume = payload * 2.0 * (tp - 1) / tp
        out = [producer]
        for member in key.members:
            if int(member) == int(src_device):
                continue
            out.append(
                self.transfer(
                    f"{name}.tp{member}",
                    producer,
                    int(member),
                    volume / max(1, len(key.members) - 1),
                    boundary_id=f"tp.L{layer}.{block}",
                    law=LAW_COLLECTIVE,
                    phase=phase,
                    step=step,
                    layer=layer,
                    block=block,
                    tokens=tokens,
                    note=(
                        f"tp all-reduce of a row-parallel stage over group "
                        f"{key.axis}{list(key.members)}; ring volume 2(tp-1)/tp"
                    ),
                )
            )
        return out

    def _layer(
        self,
        layer: int,
        chip_id: int,
        tp_idx: int,
        deps: Sequence[int],
        phase: str,
        step: int,
        tokens: float,
    ) -> List[int]:
        mapping = self.mapping
        model = mapping.model
        mixers = tuple(getattr(model, "layer_mixers", ())) or (("attention",),)
        kinds = mixers[layer] if layer < len(mixers) else mixers[-1]
        owners = [
            owner
            for owner in self._owner_tiles
            if owner.layer == layer
            and owner.op not in ("patch_embed", "lm_head", "vit_head")
            and self._owner_tp(owner) == tp_idx
        ]
        if not owners:
            return list(deps)
        first_macro = self._owner_tiles[owners[0]][0].site.macro_id
        common = dict(phase=phase, step=step, layer=layer, tokens=tokens)

        pending = list(deps)
        pending = [self._pool_op(f"{phase}{step}.L{layer}.norm_in", first_macro, pending,
                                 block="norm", **common)]

        mixer_out = pending
        for kind in kinds:
            if kind == "attention":
                mixer_out = self._attention_block(
                    layer, chip_id, tp_idx, mixer_out, phase, step, tokens
                )
            elif kind == "ssm":
                mixer_out = self._recurrent_block(
                    layer, chip_id, tp_idx, mixer_out, phase, step, tokens,
                    projections=("ssm_in_proj", "ssm_x_proj", "ssm_dt_proj"),
                    out_projection="ssm_out_proj",
                    fabric_op="ssm_scan",
                    note="SSM recurrence / SSD chunked scan and state update (D13)",
                )
            elif kind == "linear_attn":
                mixer_out = self._recurrent_block(
                    layer, chip_id, tp_idx, mixer_out, phase, step, tokens,
                    projections=("la_qkv_proj", "la_gate_proj"),
                    out_projection="la_out_proj",
                    fabric_op="delta_rule",
                    note="gated delta-rule state update (D13)",
                )
            elif kind == "short_conv":
                mixer_out = self._short_conv_block(
                    layer, chip_id, tp_idx, mixer_out, phase, step, tokens
                )
            elif kind in ("ffn", "moe"):
                continue
            else:
                raise MappingError("execution", f"no device placement for block kind {kind!r}")

        pending = [
            self._pool_op(f"{phase}{step}.L{layer}.residual_attn", first_macro, mixer_out,
                          block="residual", **common)
        ]
        pending = [
            self._pool_op(f"{phase}{step}.L{layer}.norm_ffn", first_macro, pending,
                          block="norm", **common)
        ]
        pending = self._ffn_block(layer, chip_id, tp_idx, pending, phase, step, tokens)
        pending = [
            self._pool_op(f"{phase}{step}.L{layer}.residual_ffn", first_macro, pending,
                          block="residual", **common)
        ]
        return pending

    def _routed_tokens(self, tokens: float) -> float:
        """Tokens ONE routed expert sees, from P2's ``moe_tokens_hot``.

        ``ceil(tokens * top_k * alpha / E)`` — the hot-expert load the closed
        form's ``moe_routed_ffn_time`` prices. It is never re-derived here.
        """
        return float(self.mapping.device.moe_tokens_hot(int(math.ceil(tokens))))

    def _stage_owners(self, layer: int, op: str, tp_idx: int) -> Tuple[TileOwner, ...]:
        return tuple(
            owner
            for owner in self._owner_tiles
            if owner.layer == layer and owner.op == op and self._owner_tp(owner) == tp_idx
        )

    def _run_stage(
        self,
        layer: int,
        op: str,
        tp_idx: int,
        deps: Sequence[int],
        phase: str,
        step: int,
        tokens: float,
    ) -> List[int]:
        out: List[int] = []
        for owner in self._stage_owners(layer, op, tp_idx):
            tiles = self._owner_tiles[owner]
            chip_id = self.mapping.macro(tiles[0].site.macro_id).chip_id
            out.extend(
                self._weight_stage(owner, tiles, chip_id, deps, phase, step, tokens, layer=layer)
            )
        return out

    def _attention_block(
        self,
        layer: int,
        chip_id: int,
        tp_idx: int,
        deps: Sequence[int],
        phase: str,
        step: int,
        tokens: float,
    ) -> List[int]:
        mapping = self.mapping
        common = dict(phase=phase, step=step, layer=layer, tokens=tokens)
        qkv = self._run_stage(layer, "qkv", tp_idx, deps, phase, step, tokens)
        if not qkv:
            return list(deps)
        # Gated attention (Qwen3.5): W_g reads the SAME block input the qkv
        # projection reads, so it runs beside it on its own macros. It is an
        # ordinary analog weight stage; only the sigmoid and the multiply are
        # pool work, and they wait for the attention output below.
        gate = self._run_stage(layer, "attn_gate_proj", tp_idx, deps, phase, step, tokens)
        p = mapping.device.params
        # analog -> shared digital hop: the act x act work leaves the macro (D13)
        heads = max(1, p.num_heads // max(1, int(mapping.degrees.get("tp", 1))))
        qkv_bytes = tokens * (heads + 2 * max(1, p.kv_heads // max(1, int(mapping.degrees["tp"])))) \
            * p.head_dim * self.act_bytes
        hops = [
            self.transfer(
                f"{phase}{step}.L{layer}.qkv->fabric.{uid}",
                uid,
                self._fabric_device(chip_id).device_id,
                qkv_bytes / max(1, len(qkv)),
                boundary_id=f"act.L{layer}.qkv_to_fabric",
                phase=phase,
                step=step,
                layer=layer,
                block="qkv",
                tokens=tokens,
                note="analog -> shared-digital hop that attention needs (D13)",
            )
            for uid in qkv
        ]
        qk = self._fabric_op(f"{phase}{step}.L{layer}.attn.qk", chip_id, hops,
                             block="attention_qk", **common)
        softmax = self._fabric_op(f"{phase}{step}.L{layer}.attn.softmax", chip_id, [qk],
                                  block="attention_softmax", **common)
        pv = self._fabric_op(f"{phase}{step}.L{layer}.attn.pv", chip_id, [softmax],
                             block="attention_pv", **common)
        o_owners = self._stage_owners(layer, "o_proj", tp_idx)
        back: List[int] = [pv]
        if o_owners:
            sink = mapping.macro(self._owner_tiles[o_owners[0]][0].site.macro_id)
            # With a gate the attention output lands on the pool that holds the
            # gate, because that is where both operands of the multiply are.
            gate_macro = self.annotations[gate[-1]].macro_id if gate else -1
            sink_device = (
                mapping.macro(gate_macro).pool_device if gate else sink.analog_device
            )
            back = [
                self.transfer(
                    f"{phase}{step}.L{layer}.fabric->{'gate' if gate else 'o_proj'}",
                    pv,
                    sink_device,
                    tokens * heads * p.head_dim * self.act_bytes,
                    boundary_id=(
                        f"act.L{layer}.fabric_to_gate" if gate
                        else f"act.L{layer}.fabric_to_o_proj"
                    ),
                    phase=phase,
                    step=step,
                    layer=layer,
                    block="o_proj",
                    tokens=tokens,
                    note=(
                        "shared-digital -> macro-pool hop returning the attention "
                        "output to the pool that holds its gate (D13)"
                        if gate else
                        "shared-digital -> analog hop returning the attention output (D13)"
                    ),
                )
            ]
        if gate:
            # sigmoid(W_g x) * attention_out: elementwise over the owned query
            # channels, on the per-macro pool that already holds W_g's output
            # (ADJ-3 / D12). Its CONCURRENCY is what sizes that pool (P4.5).
            back = [
                self._pool_op(
                    f"{phase}{step}.L{layer}.attn.output_gate",
                    self.annotations[gate[-1]].macro_id,
                    list(back) + list(gate),
                    block="attn_output_gate",
                    note=(
                        "gated attention output: sigmoid of the gate projection times "
                        "the attention output, elementwise on the per-macro pool "
                        "(ADJ-3 / D12)"
                    ),
                    **common,
                )
            ]
        out = self._run_stage(layer, "o_proj", tp_idx, back, phase, step, tokens)
        return self._reduce_row_parallel(out, layer, "o_proj", phase, step, tokens)

    def _recurrent_block(
        self,
        layer: int,
        chip_id: int,
        tp_idx: int,
        deps: Sequence[int],
        phase: str,
        step: int,
        tokens: float,
        *,
        projections: Sequence[str],
        out_projection: str,
        fabric_op: str,
        note: str,
    ) -> List[int]:
        common = dict(phase=phase, step=step, layer=layer, tokens=tokens)
        produced: List[int] = list(deps)
        for op in projections:
            stage = self._run_stage(layer, op, tp_idx, produced, phase, step, tokens)
            if stage:
                produced = stage
        scan = self._fabric_op(
            f"{phase}{step}.L{layer}.{fabric_op}", chip_id, produced,
            block=fabric_op, note=note, **common
        )
        out = self._run_stage(layer, out_projection, tp_idx, [scan], phase, step, tokens)
        return self._reduce_row_parallel(out, layer, out_projection, phase, step, tokens)

    def _short_conv_block(
        self,
        layer: int,
        chip_id: int,
        tp_idx: int,
        deps: Sequence[int],
        phase: str,
        step: int,
        tokens: float,
    ) -> List[int]:
        """ADJ-3: the short depthwise conv runs on the per-macro digital pool."""
        common = dict(phase=phase, step=step, layer=layer, tokens=tokens)
        produced = self._run_stage(
            layer, "conv_in_proj", tp_idx, deps, phase, step, tokens
        ) or list(deps)
        macro_id = self.annotations[produced[-1]].macro_id
        conv = self._pool_op(
            f"{phase}{step}.L{layer}.short_conv", macro_id, produced,
            block="short_conv",
            note="short causal depthwise conv absorbed by the per-macro pool (ADJ-3)",
            **common,
        )
        out = self._run_stage(layer, "conv_out_proj", tp_idx, [conv], phase, step, tokens)
        return self._reduce_row_parallel(out, layer, "conv_out_proj", phase, step, tokens)

    def _ffn_block(
        self,
        layer: int,
        chip_id: int,
        tp_idx: int,
        deps: Sequence[int],
        phase: str,
        step: int,
        tokens: float,
    ) -> List[int]:
        mapping = self.mapping
        common = dict(phase=phase, step=step, layer=layer, tokens=tokens)
        if mapping.device.layer_class_mask()[layer]:
            router = self._run_stage(layer, "router", tp_idx, deps, phase, step, tokens)
            gate = router or list(deps)
            produced: List[int] = []
            for stage_in, stage_out in (("ffn1_routed", "ffn2_routed"),
                                        ("ffn1_shared", "ffn2_shared")):
                routed = stage_in.endswith("_routed")
                # A ROUTED expert sees only the tokens the router sent it, not
                # the whole owner batch: P2's moe_tokens_hot is that count
                # (ceil(tokens * top_k * alpha / E), the one-hot imbalance
                # contract). Shared experts do see every token. Pricing every
                # expert at the full batch charged the layer E/top_k times its
                # analog work — caught by the P6.3 bridge on fws_cim_moe, where
                # the closed form's moe_routed_ffn_time uses the same law.
                # The dispatch/combine BYTES stay on the aggregate `tokens`:
                # moe_dispatch_bytes prices the whole A2A, not one expert's
                # share.
                stage_tokens = self._routed_tokens(tokens) if routed else tokens
                stage_common = dict(common, tokens=stage_tokens)
                dispatched = (
                    self._expert_dispatch(layer, stage_in, tp_idx, gate, phase, step, tokens)
                    if routed
                    else gate
                )
                first = self._run_stage(
                    layer, stage_in, tp_idx, dispatched, phase, step, stage_tokens
                )
                if not first:
                    continue
                act = self._activations(
                    f"{phase}{step}.L{layer}.{stage_in}.act", first, **stage_common
                )
                second = self._run_stage(
                    layer, stage_out, tp_idx, act, phase, step, stage_tokens
                )
                reduced = self._reduce_row_parallel(
                    second, layer, stage_out, phase, step, stage_tokens
                )
                if routed:
                    reduced = self._expert_combine(
                        layer, stage_out, tp_idx, reduced, phase, step, tokens
                    )
                produced.extend(reduced)
            return produced or gate
        first = self._run_stage(layer, "ffn1", tp_idx, deps, phase, step, tokens)
        if not first:
            return list(deps)
        act = self._activations(f"{phase}{step}.L{layer}.ffn1.act", first, **common)
        second = self._run_stage(layer, "ffn2", tp_idx, act, phase, step, tokens)
        return self._reduce_row_parallel(second, layer, "ffn2", phase, step, tokens)

    def _activations(self, name: str, produced: Sequence[int], **common) -> List[int]:
        """One activation op per PRODUCING MACRO, on that macro's own pool (D12).

        Each op depends on every producer, not only on its own macro's: a gated
        MLP multiplies two halves that the column split may have put on
        different macros, so the gate cannot fire before both arrive. Placing
        one activation for the whole stage would instead put the work on a pool
        that does not serve those columns.
        """
        out: List[int] = []
        for macro_id in dict.fromkeys(self.annotations[uid].macro_id for uid in produced):
            if macro_id < 0:
                continue
            out.append(
                self._pool_op(
                    f"{name}.m{macro_id}", macro_id, produced, block="activation", **common
                )
            )
        return out or list(produced)

    def _expert_chips(self, layer: int, tp_idx: int) -> Tuple[int, ...]:
        """The chips that hold this layer's routed experts, when ep moved them."""
        mapping = self.mapping
        return tuple(
            chip_id
            for chip_id in mapping.layer_chip.get(layer, ())
            if mapping.chip(chip_id).role == "expert_pool"
            and int(mapping.chip(chip_id).shard.tp) == int(tp_idx)
        )

    def _expert_dispatch(
        self,
        layer: int,
        block: str,
        tp_idx: int,
        deps: Sequence[int],
        phase: str,
        step: int,
        tokens: float,
    ) -> List[int]:
        """Routed tokens crossing to the expert-pool chips (D17, one hop each).

        With ep = 1 the routed experts sit on the layer's own chip and there is
        no boundary at all, which is the honest answer rather than a zero-byte
        transfer nobody sends.
        """
        chips = self._expert_chips(layer, tp_idx)
        if not chips or not deps:
            return list(deps)
        mapping = self.mapping
        ep = max(1, int(mapping.degrees.get("ep", 1)))
        act_bytes = self.act_bytes
        size = mapping.device.moe_dispatch_bytes(tokens, act_bytes) / ep
        out: List[int] = []
        for chip_id in chips:
            sink = next(m for m in mapping.macros if m.chip_id == chip_id and m.pool == "analog")
            out.append(
                self.transfer(
                    f"{phase}{step}.L{layer}.dispatch.c{chip_id}",
                    deps[-1],
                    sink.analog_device,
                    size,
                    boundary_id=f"ep.dispatch.L{layer}.c{chip_id}",
                    law=LAW_COLLECTIVE,
                    phase=phase,
                    step=step,
                    layer=layer,
                    block=block,
                    tokens=tokens,
                    note="MoE expert dispatch: balanced A2A sizing over the ep link",
                )
            )
        return out

    def _expert_combine(
        self,
        layer: int,
        block: str,
        tp_idx: int,
        produced: Sequence[int],
        phase: str,
        step: int,
        tokens: float,
    ) -> List[int]:
        """Expert outputs returning to the layer chip — dispatch, backwards."""
        chips = self._expert_chips(layer, tp_idx)
        if not chips or not produced:
            return list(produced)
        mapping = self.mapping
        ep = max(1, int(mapping.degrees.get("ep", 1)))
        size = mapping.device.moe_dispatch_bytes(tokens, self.act_bytes) / ep
        home = self._layer_chip(layer, tp_idx)
        sink = next(m for m in mapping.macros if m.chip_id == home and m.pool == "analog")
        out: List[int] = []
        for uid in produced:
            source_chip = self.annotations[uid].chip_id
            if source_chip == home:
                out.append(uid)
                continue
            out.append(
                self.transfer(
                    f"{phase}{step}.L{layer}.combine.c{source_chip}",
                    uid,
                    sink.pool_device,
                    size / len(chips),
                    boundary_id=f"ep.combine.L{layer}.c{source_chip}",
                    law=LAW_COLLECTIVE,
                    phase=phase,
                    step=step,
                    layer=layer,
                    block=block,
                    tokens=tokens,
                    note="MoE expert combine: the dispatch law, backwards",
                )
            )
        return out

    def _reduce_row_parallel(
        self, produced: Sequence[int], layer: int, block: str, phase: str, step: int, tokens: float
    ) -> List[int]:
        """A row-parallel stage's tp partials need a tp all-reduce (P4 §1)."""
        if int(self.mapping.degrees.get("tp", 1)) <= 1 or not produced:
            return list(produced)
        out: List[int] = []
        for uid in produced:
            out.extend(
                self._tp_exchange(
                    f"{phase}{step}.L{layer}.{block}.allreduce",
                    uid,
                    (),
                    phase=phase,
                    step=step,
                    layer=layer,
                    block=block,
                    tokens=tokens,
                )
            )
        return out

    def _fabric_device(self, chip_id: int):
        shard = self.mapping.chip(chip_id).shard
        engines = self.mapping.shared_digital_devices(shard)
        if not engines:
            raise MappingError(
                "execution",
                "the workload needs a shared digital chiplet (D13) and the mapping "
                "declares none.",
            )
        return engines[chip_id % len(engines)]


def build_fws_program(
    mapping: FwsMapping,
    *,
    serving: Optional[ServingPoint] = None,
    decode_steps: Optional[int] = None,
    label: Optional[str] = None,
) -> Program:
    """Lower one mapping into a placed, annotated, UNPRICED program IR.

    This is the P3 export to P4: ops with explicit deps, a device, tiles, bytes,
    an owner and a shard group. It is not a period and it never becomes one.
    """
    serving = serving or ServingPoint.from_mapping(mapping, decode_steps)
    lowering = _Lowering(mapping, serving, label or f"fws:{mapping.label}")
    return lowering.run()


def annotations_of(program: Program) -> Tuple[FwsOpAnnotation, ...]:
    """The uid-indexed annotation table of an FWS program."""
    return tuple(program.meta.misc.get("fws_annotations", ()))


def annotation_of(program: Program, uid: int) -> FwsOpAnnotation:
    return annotations_of(program)[int(uid)]
