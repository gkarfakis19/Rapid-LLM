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

"""Canonical, implementation-independent form of Chakra ET bundles.

Two implementations of the RAPID-LLM -> AstraSim conversion are considered
equivalent on a bundle when, for every rank:

1. the multiset of operations matches (kind + payload), and
2. the dependency DAG matches structurally (Merkle hash over payloads and
   parent hashes).

Payloads deliberately exclude everything an implementation is free to choose
differently: node names, node ids, op ids, p2p tags, and communicator-group
*ids* (groups are resolved to their member-rank sets via comm_groups.json).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from astrasim_lib.et_utils import chakra_decode, chakra_open, pb

# Payload tuples are one of:
#   ("COMP", duration_micros)
#   ("COLL", collective_enum, size_bytes, members: Tuple[int, ...])
#   ("SEND", size_bytes, dst_rank)
#   ("RECV", size_bytes, src_rank)
Payload = Tuple[Any, ...]

_RANK_ET_RE = re.compile(r"^(?P<prefix>.+)\.(?P<rank>\d+)\.et$")


@dataclass
class CanonicalNode:
    payload: Payload
    parents: List[int]  # indices into the rank's node list (original id space)
    name: str  # kept for diagnostics only; never part of hashes


@dataclass
class CanonicalRank:
    rank: int
    nodes: List[CanonicalNode]
    dag_hash: str = ""
    ops_hash: str = ""
    op_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class CanonicalBundle:
    prefix: str
    ranks: Dict[int, CanonicalRank]
    bundle_hash: str = ""

    def summary(self) -> Dict[str, Any]:
        return {
            "bundle_hash": self.bundle_hash,
            "n_ranks": len(self.ranks),
            "ranks": {
                str(rank): {
                    "dag_hash": cr.dag_hash,
                    "ops_hash": cr.ops_hash,
                    "op_counts": dict(sorted(cr.op_counts.items())),
                    "n_nodes": len(cr.nodes),
                }
                for rank, cr in sorted(self.ranks.items())
            },
        }


def _attr_map(node: "pb.Node") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for attr in node.attr:
        which = attr.WhichOneof("value")
        if which is None:
            continue
        raw = getattr(attr, which)
        out[attr.name] = list(raw.values) if hasattr(raw, "values") else raw
    return out


def _load_et_nodes(path: str) -> List["pb.Node"]:
    fh = chakra_open(path)
    try:
        meta = pb.GlobalMetadata()
        if not chakra_decode(fh, meta):
            raise ValueError(f"{path}: missing Chakra GlobalMetadata")
        nodes: List[pb.Node] = []
        while True:
            node = pb.Node()
            if not chakra_decode(fh, node):
                break
            nodes.append(node)
        return nodes
    finally:
        fh.close()


def _load_comm_groups(bundle_dir: str) -> Dict[str, List[int]]:
    path = os.path.join(bundle_dir, "comm_groups.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        raw = json.load(fh)
    return {str(k): sorted(int(r) for r in v) for k, v in raw.items()}


def _payload_for(
    node: "pb.Node",
    comm_groups: Dict[str, List[int]],
    all_ranks: Sequence[int],
) -> Payload:
    attrs = _attr_map(node)
    if node.type == pb.COMP_NODE:
        return ("COMP", int(node.duration_micros or 0))
    if node.type == pb.COMM_COLL_NODE:
        pg = attrs.get("pg_name")
        if pg is not None:
            members = tuple(comm_groups.get(str(pg), ()))
            if not members:
                raise ValueError(
                    f"ET node {node.id} ({node.name}) references pg_name={pg} "
                    "but comm_groups.json has no such group"
                )
        else:
            members = tuple(sorted(int(r) for r in all_ranks))
        return (
            "COLL",
            int(attrs.get("comm_type", -1)),
            int(attrs.get("comm_size", 0)),
            members,
        )
    if node.type == pb.COMM_SEND_NODE:
        return ("SEND", int(attrs.get("comm_size", 0)), int(attrs.get("comm_dst", -1)))
    if node.type == pb.COMM_RECV_NODE:
        return ("RECV", int(attrs.get("comm_size", 0)), int(attrs.get("comm_src", -1)))
    raise ValueError(f"Unsupported ET node type {node.type} for node {node.id}")


def _hash_str(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def canonicalize_rank(
    rank: int,
    et_path: str,
    comm_groups: Dict[str, List[int]],
    all_ranks: Sequence[int],
) -> CanonicalRank:
    pb_nodes = _load_et_nodes(et_path)
    by_id = {int(n.id): n for n in pb_nodes}

    nodes: List[CanonicalNode] = []
    id_to_index: Dict[int, int] = {}
    for idx, node in enumerate(pb_nodes):
        id_to_index[int(node.id)] = idx
        parents = [int(d) for d in node.ctrl_deps if int(d) in by_id]
        nodes.append(
            CanonicalNode(
                payload=_payload_for(node, comm_groups, all_ranks),
                parents=parents,
                name=node.name or "",
            )
        )

    # Merkle hash over the DAG: a node's hash covers its payload and the
    # sorted multiset of its parents' hashes. Iterate in dependency order.
    node_hash: Dict[int, str] = {}
    remaining = set(range(len(nodes)))
    progressed = True
    while remaining and progressed:
        progressed = False
        for idx in sorted(remaining):
            parent_idxs = [id_to_index[p] for p in nodes[idx].parents]
            if any(p not in node_hash for p in parent_idxs):
                continue
            parent_hashes = sorted(node_hash[p] for p in parent_idxs)
            node_hash[idx] = _hash_str(
                repr(nodes[idx].payload) + "|" + ",".join(parent_hashes)
            )
            remaining.discard(idx)
            progressed = True
    if remaining:
        raise ValueError(f"{et_path}: cycle detected in ctrl_deps ({len(remaining)} nodes unresolved)")

    dag_hash = _hash_str(",".join(sorted(node_hash.values())))
    ops_sorted = sorted(repr(n.payload) for n in nodes)
    ops_hash = _hash_str(";".join(ops_sorted))
    op_counts: Dict[str, int] = {}
    for n in nodes:
        op_counts[n.payload[0]] = op_counts.get(n.payload[0], 0) + 1

    return CanonicalRank(
        rank=rank, nodes=nodes, dag_hash=dag_hash, ops_hash=ops_hash, op_counts=op_counts
    )


def canonicalize_bundle(bundle_dir: str, prefix: str = "llm_graph") -> CanonicalBundle:
    """Canonicalize all ``<prefix>.<rank>.et`` files in ``bundle_dir``."""
    rank_paths: Dict[int, str] = {}
    for entry in os.listdir(bundle_dir):
        match = _RANK_ET_RE.match(entry)
        if match and match.group("prefix") == prefix:
            rank_paths[int(match.group("rank"))] = os.path.join(bundle_dir, entry)
    if not rank_paths:
        raise FileNotFoundError(f"No {prefix}.<rank>.et files in {bundle_dir}")

    comm_groups = _load_comm_groups(bundle_dir)
    all_ranks = sorted(rank_paths)
    ranks = {
        rank: canonicalize_rank(rank, path, comm_groups, all_ranks)
        for rank, path in sorted(rank_paths.items())
    }
    bundle_hash = _hash_str(
        ",".join(f"{rank}:{cr.dag_hash}" for rank, cr in sorted(ranks.items()))
    )
    return CanonicalBundle(prefix=prefix, ranks=ranks, bundle_hash=bundle_hash)


def diff_bundles(
    golden: Dict[str, Any], candidate: Dict[str, Any], label: str = ""
) -> List[str]:
    """Human-readable differences between two ``CanonicalBundle.summary()`` dicts."""
    problems: List[str] = []
    tag = f"[{label}] " if label else ""
    if golden["n_ranks"] != candidate["n_ranks"]:
        problems.append(
            f"{tag}rank count differs: golden={golden['n_ranks']} candidate={candidate['n_ranks']}"
        )
        return problems
    for rank, gold in golden["ranks"].items():
        cand = candidate["ranks"].get(rank)
        if cand is None:
            problems.append(f"{tag}rank {rank} missing from candidate")
            continue
        if gold["ops_hash"] != cand["ops_hash"]:
            problems.append(
                f"{tag}rank {rank}: op multiset differs "
                f"(golden counts {gold['op_counts']}, candidate {cand['op_counts']})"
            )
        elif gold["dag_hash"] != cand["dag_hash"]:
            problems.append(
                f"{tag}rank {rank}: same ops but dependency DAG differs"
            )
    return problems
