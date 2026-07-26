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

"""M1 SHADOW verification — removed at M2 cutover.

Called from the tail of ``convert_rapid_llm_graph_to_chakra_et`` when
``RAPID_SHADOW_EMIT`` is truthy: re-derives the bundle through
``lower_to_program`` + ``emit_chakra`` into ``<output_dir>/__shadow__/`` and
compares it against the legacy bundle just written:

(a) per-rank node sequences in final id order — node type + payload +
    sorted ctrl-dep ids must be identical, where payload is
    ``duration_micros`` for COMP, ``(comm_type, comm_size, resolved pg
    member set)`` for COLL, and ``(comm_size, dst/src rank)`` for
    SEND/RECV. Names and tag VALUES are excluded (design C R4/R19), but tag
    PAIRING must be a bijection: every shadow ``((src,dst), tag)`` class
    of node positions must coincide with exactly one legacy class;
(b) ``manifest.json`` byte-equal;
(c) ``comm_groups.json`` equal as parsed JSON (the legacy file is not yet
    written at hook time, so the reference is produced by calling the
    legacy ``_write_comm_groups_json`` into a scratch dir).

Any mismatch raises ``RuntimeError`` with a precise (rank, node index,
field) diff; on success the ``__shadow__`` directory is deleted so the
equivalence harness never sees it as an extra bundle.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

from astrasim_lib.et_utils import chakra_decode, chakra_open, pb

_MAX_PROBLEMS = 25


def _load_et_nodes(path: str) -> List["pb.Node"]:
    fh = chakra_open(path)
    try:
        meta = pb.GlobalMetadata()
        if not chakra_decode(fh, meta):
            raise ValueError(f"{path}: missing Chakra GlobalMetadata")
        nodes: List["pb.Node"] = []
        while True:
            node = pb.Node()
            if not chakra_decode(fh, node):
                break
            nodes.append(node)
        return nodes
    finally:
        fh.close()


def _attr_map(node: "pb.Node") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for attr in node.attr:
        which = attr.WhichOneof("value")
        if which is None:
            continue
        raw = getattr(attr, which)
        out[attr.name] = list(raw.values) if hasattr(raw, "values") else raw
    return out


def _payload(
    node: "pb.Node",
    comm_groups: Dict[str, List[int]],
    all_ranks: List[int],
) -> Tuple[Any, ...]:
    attrs = _attr_map(node)
    if node.type == pb.COMP_NODE:
        return ("COMP", int(node.duration_micros or 0))
    if node.type == pb.COMM_COLL_NODE:
        pg = attrs.get("pg_name")
        if pg is not None:
            members = tuple(sorted(int(r) for r in comm_groups.get(str(pg), ())))
        else:
            members = tuple(sorted(int(r) for r in all_ranks))
        return ("COLL", int(attrs.get("comm_type", -1)), int(attrs.get("comm_size", 0)), members)
    if node.type == pb.COMM_SEND_NODE:
        return ("SEND", int(attrs.get("comm_size", 0)), int(attrs.get("comm_dst", -1)))
    if node.type == pb.COMM_RECV_NODE:
        return ("RECV", int(attrs.get("comm_size", 0)), int(attrs.get("comm_src", -1)))
    return ("UNKNOWN", int(node.type))


def _p2p_identity(node: "pb.Node", rank: int) -> Optional[Tuple[Tuple[int, int], int]]:
    """((src_rank, dst_rank), tag) for SEND/RECV nodes; None otherwise."""
    attrs = _attr_map(node)
    if node.type == pb.COMM_SEND_NODE:
        return ((rank, int(attrs.get("comm_dst", -1))), int(attrs.get("comm_tag", -1)))
    if node.type == pb.COMM_RECV_NODE:
        return ((int(attrs.get("comm_src", -1)), rank), int(attrs.get("comm_tag", -1)))
    return None


def run_shadow_comparison(
    *,
    graph_root: Any,
    dp_size: int,
    output_dir: str,
    legacy_rank_ids: List[int],
    legacy_manifest_path: str,
    legacy_dp_count: int,
) -> None:
    # Imported lazily so this module stays inert unless the hook fires.
    from astrasim_lib import executor as legacy_executor
    from program.et_emit import emit_chakra
    from program.legacy_lowering import lower_to_program

    shadow_dir = os.path.join(output_dir, "__shadow__")
    if os.path.isdir(shadow_dir):
        shutil.rmtree(shadow_dir, ignore_errors=True)
    os.makedirs(shadow_dir, exist_ok=True)

    program = lower_to_program(
        graph_root,
        dp_size,
        getattr(graph_root, "_astrasim_rank_layout", None),
        gmap_workdir=shadow_dir,
    )
    bundle = emit_chakra(program, shadow_dir, id_policy="legacy")

    problems: List[str] = []

    # ---- rank id sets ----------------------------------------------------
    legacy_ranks = sorted(int(r) for r in legacy_rank_ids)
    shadow_ranks = sorted(int(r) for r in bundle.rank_ids)
    if legacy_ranks != shadow_ranks:
        problems.append(f"rank ids differ: legacy={legacy_ranks} shadow={shadow_ranks}")
        _raise(problems, shadow_dir)

    # ---- comm groups (c) -------------------------------------------------
    ref_dir = os.path.join(shadow_dir, "legacy_comm_groups_ref")
    os.makedirs(ref_dir, exist_ok=True)
    legacy_cg_path = legacy_executor._write_comm_groups_json(ref_dir, legacy_dp_count, list(legacy_ranks))
    legacy_groups: Dict[str, List[int]] = {}
    if legacy_cg_path is not None:
        with open(legacy_cg_path) as fh:
            legacy_groups = {str(k): sorted(int(r) for r in v) for k, v in json.load(fh).items()}
    shadow_groups = {str(k): sorted(int(r) for r in v) for k, v in bundle.comm_groups.items()}
    if legacy_groups != shadow_groups:
        for key in sorted(set(legacy_groups) | set(shadow_groups)):
            if legacy_groups.get(key) != shadow_groups.get(key):
                problems.append(
                    f"comm_groups[{key}]: legacy={legacy_groups.get(key)} "
                    f"shadow={shadow_groups.get(key)}"
                )

    # ---- per-rank node sequences (a) -------------------------------------
    shadow_pairs: Dict[Tuple[Tuple[int, int], int], List[Tuple[int, int]]] = {}
    legacy_pairs: Dict[Tuple[Tuple[int, int], int], List[Tuple[int, int]]] = {}
    for rank in legacy_ranks:
        legacy_path = os.path.join(output_dir, f"llm_graph.{rank}.et")
        shadow_path = os.path.join(shadow_dir, f"llm_graph.{rank}.et")
        try:
            legacy_nodes = _load_et_nodes(legacy_path)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"rank {rank}: failed to read legacy ET: {exc}")
            continue
        try:
            shadow_nodes = _load_et_nodes(shadow_path)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"rank {rank}: failed to read shadow ET: {exc}")
            continue
        if len(legacy_nodes) != len(shadow_nodes):
            problems.append(
                f"rank {rank}: node count legacy={len(legacy_nodes)} shadow={len(shadow_nodes)}"
            )
        for idx in range(min(len(legacy_nodes), len(shadow_nodes))):
            lnode, snode = legacy_nodes[idx], shadow_nodes[idx]
            if int(lnode.type) != int(snode.type):
                problems.append(
                    f"rank {rank} node {idx}: type legacy={int(lnode.type)} "
                    f"({lnode.name}) shadow={int(snode.type)} ({snode.name})"
                )
                continue
            lpay = _payload(lnode, legacy_groups, legacy_ranks)
            spay = _payload(snode, shadow_groups, shadow_ranks)
            if lpay != spay:
                problems.append(
                    f"rank {rank} node {idx}: payload legacy={lpay} ({lnode.name}) "
                    f"shadow={spay} ({snode.name})"
                )
            ldeps = sorted(int(d) for d in lnode.ctrl_deps)
            sdeps = sorted(int(d) for d in snode.ctrl_deps)
            if ldeps != sdeps:
                problems.append(
                    f"rank {rank} node {idx}: ctrl_deps legacy={ldeps} ({lnode.name}) "
                    f"shadow={sdeps} ({snode.name})"
                )
            lident = _p2p_identity(lnode, rank)
            sident = _p2p_identity(snode, rank)
            if lident is not None:
                legacy_pairs.setdefault(lident, []).append((rank, idx))
            if sident is not None:
                shadow_pairs.setdefault(sident, []).append((rank, idx))
            if len(problems) > _MAX_PROBLEMS:
                _raise(problems, shadow_dir)

    # ---- tag pairing bijection (a, tag clause) ---------------------------
    if not problems:
        legacy_by_positions = {tuple(sorted(v)): k for k, v in legacy_pairs.items()}
        shadow_to_legacy: Dict[Tuple[Tuple[int, int], int], Tuple[Tuple[int, int], int]] = {}
        legacy_matched: Dict[Tuple[Tuple[int, int], int], Tuple[Tuple[int, int], int]] = {}
        for skey, positions in shadow_pairs.items():
            pos_key = tuple(sorted(positions))
            lkey = legacy_by_positions.get(pos_key)
            if lkey is None:
                problems.append(
                    f"tag pairing: shadow tag class {skey} covers node positions "
                    f"{pos_key} which match no single legacy tag class"
                )
                continue
            if skey[0] != lkey[0]:
                problems.append(
                    f"tag pairing: shadow class {skey} and legacy class {lkey} "
                    "disagree on (src,dst) endpoints"
                )
                continue
            shadow_to_legacy[skey] = lkey
            prev = legacy_matched.get(lkey)
            if prev is not None and prev != skey:
                problems.append(
                    f"tag pairing: legacy tag class {lkey} matched by two shadow "
                    f"classes {prev} and {skey}"
                )
            legacy_matched[lkey] = skey
        if not problems and len(legacy_pairs) != len(shadow_pairs):
            problems.append(
                f"tag pairing: legacy has {len(legacy_pairs)} (src,dst,tag) classes, "
                f"shadow has {len(shadow_pairs)}"
            )

    # ---- manifest bytes (b) ----------------------------------------------
    try:
        with open(legacy_manifest_path, "rb") as fh:
            legacy_manifest = fh.read()
        with open(bundle.manifest_path, "rb") as fh:
            shadow_manifest = fh.read()
        if legacy_manifest != shadow_manifest:
            detail = _manifest_diff_detail(legacy_manifest, shadow_manifest)
            problems.append(f"manifest.json bytes differ: {detail}")
    except OSError as exc:
        problems.append(f"manifest comparison failed: {exc}")

    if problems:
        _raise(problems, shadow_dir)

    shutil.rmtree(shadow_dir, ignore_errors=True)


def _manifest_diff_detail(legacy_bytes: bytes, shadow_bytes: bytes) -> str:
    try:
        legacy = json.loads(legacy_bytes)
        shadow = json.loads(shadow_bytes)
    except json.JSONDecodeError:
        return "unparseable JSON on one side"
    for field in ("version", "npus"):
        if legacy.get(field) != shadow.get(field):
            return f"{field}: legacy={legacy.get(field)} shadow={shadow.get(field)}"
    legacy_ranks = legacy.get("ranks", {})
    shadow_ranks = shadow.get("ranks", {})
    if set(legacy_ranks) != set(shadow_ranks):
        return (
            f"rank keys differ: legacy-only={sorted(set(legacy_ranks) - set(shadow_ranks))} "
            f"shadow-only={sorted(set(shadow_ranks) - set(legacy_ranks))}"
        )
    for rank in sorted(legacy_ranks, key=int):
        lops, sops = legacy_ranks[rank], shadow_ranks[rank]
        if lops == sops:
            continue
        if len(lops) != len(sops):
            return f"rank {rank}: op count legacy={len(lops)} shadow={len(sops)}"
        for idx, (lop, sop) in enumerate(zip(lops, sops)):
            if lop != sop:
                return f"rank {rank} sorted-op {idx}: legacy={lop} shadow={sop}"
    return "byte-level difference with identical parsed content (formatting)"


def _raise(problems: List[str], shadow_dir: str) -> None:
    listed = problems[:_MAX_PROBLEMS]
    suffix = "" if len(problems) <= _MAX_PROBLEMS else f"\n... and {len(problems) - _MAX_PROBLEMS} more"
    raise RuntimeError(
        "[M1 shadow] lowering+emitter bundle diverged from the legacy converter "
        f"(shadow artifacts kept in {shadow_dir}; dump both with "
        "astrasim_lib.executor._dump_et_text to diff):\n"
        + "\n".join(f"  - {p}" for p in listed)
        + suffix
    )
