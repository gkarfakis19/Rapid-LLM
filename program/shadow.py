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

"""Chakra ET bundle comparison utilities.

:func:`compare_et_bundles` is the standard bundle comparator (born as the
M1 shadow-emit verifier, now the shared assertion core of the builder
differential/determinism tests). Two bundles are equivalent iff:

(a) per-rank node sequences in final id order — node type + payload +
    sorted ctrl-dep ids must be identical, where payload is
    ``duration_micros`` for COMP, ``(comm_type, comm_size, resolved pg
    member set)`` for COLL, and ``(comm_size, dst/src rank)`` for
    SEND/RECV. Names and tag VALUES are excluded (design C R4/R19), but tag
    PAIRING must be a bijection: every candidate ``((src,dst), tag)`` class
    of node positions must coincide with exactly one reference class;
(b) ``manifest.json`` byte-equal;
(c) ``comm_groups.json`` equal as parsed JSON.

M8 note: ``run_shadow_comparison`` — the M1 hook that re-derived a shadow
bundle through ``lower_to_program`` + ``emit_chakra`` and compared it
in-process — was dead since the M6 mode cutovers completed and is deleted;
:func:`load_comm_groups` stays as the bundle ``comm_groups.json`` reader.
"""

from __future__ import annotations

import json
import os
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


def load_comm_groups(bundle_dir: str) -> Dict[str, List[int]]:
    """Read a bundle's ``comm_groups.json`` (absent => no groups)."""
    path = os.path.join(bundle_dir, "comm_groups.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return {str(k): sorted(int(r) for r in v) for k, v in json.load(fh).items()}


def compare_et_bundles(
    *,
    reference_dir: str,
    candidate_dir: str,
    reference_ranks: List[int],
    candidate_ranks: List[int],
    reference_manifest: str,
    candidate_manifest: str,
    reference_groups: Dict[str, List[int]],
    candidate_groups: Dict[str, List[int]],
    reference_label: str = "legacy",
    candidate_label: str = "shadow",
    max_problems: int = _MAX_PROBLEMS,
) -> List[str]:
    """The standard bundle comparator (module docstring clauses a-c).

    Returns the list of problems (empty == equivalent). Shared by the M1
    shadow hook and the M3a fine-builder differential test.
    """
    problems: List[str] = []

    # ---- rank id sets ----------------------------------------------------
    reference_ranks = sorted(int(r) for r in reference_ranks)
    candidate_ranks = sorted(int(r) for r in candidate_ranks)
    if reference_ranks != candidate_ranks:
        problems.append(
            f"rank ids differ: {reference_label}={reference_ranks} "
            f"{candidate_label}={candidate_ranks}"
        )
        return problems

    # ---- comm groups (c) -------------------------------------------------
    reference_groups = {str(k): sorted(int(r) for r in v) for k, v in reference_groups.items()}
    candidate_groups = {str(k): sorted(int(r) for r in v) for k, v in candidate_groups.items()}
    if reference_groups != candidate_groups:
        for key in sorted(set(reference_groups) | set(candidate_groups)):
            if reference_groups.get(key) != candidate_groups.get(key):
                problems.append(
                    f"comm_groups[{key}]: {reference_label}={reference_groups.get(key)} "
                    f"{candidate_label}={candidate_groups.get(key)}"
                )

    # ---- per-rank node sequences (a) -------------------------------------
    candidate_pairs: Dict[Tuple[Tuple[int, int], int], List[Tuple[int, int]]] = {}
    reference_pairs: Dict[Tuple[Tuple[int, int], int], List[Tuple[int, int]]] = {}
    for rank in reference_ranks:
        reference_path = os.path.join(reference_dir, f"llm_graph.{rank}.et")
        candidate_path = os.path.join(candidate_dir, f"llm_graph.{rank}.et")
        try:
            reference_nodes = _load_et_nodes(reference_path)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"rank {rank}: failed to read {reference_label} ET: {exc}")
            continue
        try:
            candidate_nodes = _load_et_nodes(candidate_path)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"rank {rank}: failed to read {candidate_label} ET: {exc}")
            continue
        if len(reference_nodes) != len(candidate_nodes):
            problems.append(
                f"rank {rank}: node count {reference_label}={len(reference_nodes)} "
                f"{candidate_label}={len(candidate_nodes)}"
            )
        for idx in range(min(len(reference_nodes), len(candidate_nodes))):
            lnode, snode = reference_nodes[idx], candidate_nodes[idx]
            if int(lnode.type) != int(snode.type):
                problems.append(
                    f"rank {rank} node {idx}: type {reference_label}={int(lnode.type)} "
                    f"({lnode.name}) {candidate_label}={int(snode.type)} ({snode.name})"
                )
                continue
            lpay = _payload(lnode, reference_groups, reference_ranks)
            spay = _payload(snode, candidate_groups, candidate_ranks)
            if lpay != spay:
                problems.append(
                    f"rank {rank} node {idx}: payload {reference_label}={lpay} ({lnode.name}) "
                    f"{candidate_label}={spay} ({snode.name})"
                )
            ldeps = sorted(int(d) for d in lnode.ctrl_deps)
            sdeps = sorted(int(d) for d in snode.ctrl_deps)
            if ldeps != sdeps:
                problems.append(
                    f"rank {rank} node {idx}: ctrl_deps {reference_label}={ldeps} ({lnode.name}) "
                    f"{candidate_label}={sdeps} ({snode.name})"
                )
            lident = _p2p_identity(lnode, rank)
            sident = _p2p_identity(snode, rank)
            if lident is not None:
                reference_pairs.setdefault(lident, []).append((rank, idx))
            if sident is not None:
                candidate_pairs.setdefault(sident, []).append((rank, idx))
            if len(problems) > max_problems:
                return problems

    # ---- tag pairing bijection (a, tag clause) ---------------------------
    if not problems:
        reference_by_positions = {tuple(sorted(v)): k for k, v in reference_pairs.items()}
        candidate_to_reference: Dict[Tuple[Tuple[int, int], int], Tuple[Tuple[int, int], int]] = {}
        reference_matched: Dict[Tuple[Tuple[int, int], int], Tuple[Tuple[int, int], int]] = {}
        for skey, positions in candidate_pairs.items():
            pos_key = tuple(sorted(positions))
            lkey = reference_by_positions.get(pos_key)
            if lkey is None:
                problems.append(
                    f"tag pairing: {candidate_label} tag class {skey} covers node positions "
                    f"{pos_key} which match no single {reference_label} tag class"
                )
                continue
            if skey[0] != lkey[0]:
                problems.append(
                    f"tag pairing: {candidate_label} class {skey} and {reference_label} class "
                    f"{lkey} disagree on (src,dst) endpoints"
                )
                continue
            candidate_to_reference[skey] = lkey
            prev = reference_matched.get(lkey)
            if prev is not None and prev != skey:
                problems.append(
                    f"tag pairing: {reference_label} tag class {lkey} matched by two "
                    f"{candidate_label} classes {prev} and {skey}"
                )
            reference_matched[lkey] = skey
        if not problems and len(reference_pairs) != len(candidate_pairs):
            problems.append(
                f"tag pairing: {reference_label} has {len(reference_pairs)} (src,dst,tag) "
                f"classes, {candidate_label} has {len(candidate_pairs)}"
            )

    # ---- manifest bytes (b) ----------------------------------------------
    try:
        with open(reference_manifest, "rb") as fh:
            reference_bytes = fh.read()
        with open(candidate_manifest, "rb") as fh:
            candidate_bytes = fh.read()
        if reference_bytes != candidate_bytes:
            detail = _manifest_diff_detail(reference_bytes, candidate_bytes)
            problems.append(f"manifest.json bytes differ: {detail}")
    except OSError as exc:
        problems.append(f"manifest comparison failed: {exc}")

    return problems


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
