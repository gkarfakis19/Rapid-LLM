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

Restructure T1 (structural tier) adds ID-INDEPENDENT *quantities* alongside
those hashes, so a divergence reports what moved instead of "hash differs":

* per-rank compute-microsecond totals and byte totals per op kind,
* bundle-level byte histograms per op kind and per **interconnect axis**
  (from the emitter's ``comm_axes.json`` sidecar, when present),
* collectives grouped by their RESOLVED member set,
* the transfer count,
* the payload-weighted critical-path length over the per-rank DAG,
* ``manifest.json`` content (the AstraSim cache-key input, recorded
  losslessly as a per-rank multiset) and its byte digest,
* the parsed ``comm_groups.json`` member sets.

Every one of those is invariant under node-id renumbering, so they gate the
restructure phases that deliberately change emission order.
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

#: op kinds that carry bytes (COMP carries microseconds instead)
_BYTE_KINDS = ("COLL", "SEND", "RECV")

#: axis label used for collectives emitted without a communicator group
_WORLD_AXIS = "__world__"


@dataclass
class CanonicalNode:
    payload: Payload
    parents: List[int]  # indices into the rank's node list (original id space)
    name: str  # kept for diagnostics only; never part of hashes

    @property
    def kind(self) -> str:
        return str(self.payload[0])

    @property
    def weight(self) -> int:
        """Payload weight: microseconds for COMP, bytes for comm ops."""
        return int(self.payload[1]) if len(self.payload) > 1 else 0


@dataclass
class CanonicalRank:
    rank: int
    nodes: List[CanonicalNode]
    dag_hash: str = ""
    ops_hash: str = ""
    op_counts: Dict[str, int] = field(default_factory=dict)
    compute_micros: int = 0
    bytes_by_kind: Dict[str, int] = field(default_factory=dict)
    critical_path_nodes: int = 0
    critical_path_weight: int = 0

    def summary(self) -> Dict[str, Any]:
        return {
            "dag_hash": self.dag_hash,
            "ops_hash": self.ops_hash,
            "op_counts": dict(sorted(self.op_counts.items())),
            "n_nodes": len(self.nodes),
            "compute_micros": self.compute_micros,
            "bytes_by_kind": dict(sorted(self.bytes_by_kind.items())),
            "critical_path_nodes": self.critical_path_nodes,
            "critical_path_weight": self.critical_path_weight,
        }


@dataclass
class CanonicalBundle:
    prefix: str
    ranks: Dict[int, CanonicalRank]
    bundle_hash: str = ""
    comm_groups: Dict[str, List[int]] = field(default_factory=dict)
    comm_axes: Dict[str, List[str]] = field(default_factory=dict)
    manifest: Optional[Dict[str, Any]] = None
    manifest_sha256: Optional[str] = None

    # -- derived, ID-independent aggregates --------------------------------

    def _collectives_by_group(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for cr in self.ranks.values():
            for node in cr.nodes:
                if node.kind != "COLL":
                    continue
                _kind, coll_enum, size, members = node.payload
                key = ",".join(str(m) for m in members)
                entry = out.setdefault(key, {"n_ops": 0, "bytes": 0, "by_kind": {}})
                entry["n_ops"] += 1
                entry["bytes"] += int(size)
                kind_entry = entry["by_kind"].setdefault(str(int(coll_enum)), {"n": 0, "bytes": 0})
                kind_entry["n"] += 1
                kind_entry["bytes"] += int(size)
        for entry in out.values():
            entry["by_kind"] = dict(sorted(entry["by_kind"].items()))
        return dict(sorted(out.items()))

    def _axis_for_members(self, members: Sequence[int]) -> str:
        """Resolve a collective's interconnect axis from the emitter sidecar."""
        if not self.comm_axes:
            return ""
        key = ",".join(str(m) for m in members)
        axes = self.comm_axes.get(key)
        if not axes:
            return "unmapped"
        return "+".join(axes)

    def _bytes_by_axis(self) -> Dict[str, int]:
        if not self.comm_axes:
            return {}
        out: Dict[str, int] = {}
        for cr in self.ranks.values():
            for node in cr.nodes:
                if node.kind == "COLL":
                    axis = self._axis_for_members(node.payload[3]) or _WORLD_AXIS
                    out[axis] = out.get(axis, 0) + int(node.payload[2])
                elif node.kind in ("SEND", "RECV"):
                    out["p2p"] = out.get("p2p", 0) + int(node.payload[1])
        return dict(sorted(out.items()))

    def _byte_hist_by_kind(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for cr in self.ranks.values():
            for node in cr.nodes:
                if node.kind not in _BYTE_KINDS:
                    continue
                size = int(node.payload[2] if node.kind == "COLL" else node.payload[1])
                bucket = out.setdefault(node.kind, {})
                bucket[str(size)] = bucket.get(str(size), 0) + 1
        return {
            kind: dict(sorted(sizes.items(), key=lambda kv: int(kv[0])))
            for kind, sizes in sorted(out.items())
        }

    def summary(self) -> Dict[str, Any]:
        bytes_by_kind: Dict[str, int] = {}
        compute_micros_total = 0
        n_send = n_recv = 0
        for cr in self.ranks.values():
            compute_micros_total += cr.compute_micros
            for kind, total in cr.bytes_by_kind.items():
                bytes_by_kind[kind] = bytes_by_kind.get(kind, 0) + total
            n_send += cr.op_counts.get("SEND", 0)
            n_recv += cr.op_counts.get("RECV", 0)
        out: Dict[str, Any] = {
            "bundle_hash": self.bundle_hash,
            "n_ranks": len(self.ranks),
            "ranks": {str(rank): cr.summary() for rank, cr in sorted(self.ranks.items())},
            "compute_micros_total": compute_micros_total,
            "bytes_by_kind": dict(sorted(bytes_by_kind.items())),
            "byte_hist_by_kind": self._byte_hist_by_kind(),
            "n_transfers": n_send,
            "n_recv": n_recv,
            "collectives_by_group": self._collectives_by_group(),
            "comm_groups": {gid: list(members) for gid, members in sorted(self.comm_groups.items())},
        }
        axis_bytes = self._bytes_by_axis()
        if axis_bytes:
            out["bytes_by_axis"] = axis_bytes
        if self.manifest is not None:
            out["manifest"] = self.manifest
            out["manifest_sha256"] = self.manifest_sha256
        return out


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


def _load_comm_axes(bundle_dir: str, comm_groups: Dict[str, List[int]]) -> Dict[str, List[str]]:
    """Read the emitter's ``comm_axes.json`` sidecar, keyed by MEMBER SET.

    The sidecar maps communicator-group ids to the parallelism axes they were
    constructed for. Group *ids* are an implementation choice, so the axis map
    is re-keyed here by the resolved member set (the same identity the op
    payloads use). Absent sidecar => no axis attribution ("where available").
    """
    path = os.path.join(bundle_dir, "comm_axes.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    groups = raw.get("groups") if isinstance(raw, dict) else None
    if not isinstance(groups, dict):
        return {}
    by_members: Dict[str, List[str]] = {}
    for gid, entry in groups.items():
        members = comm_groups.get(str(gid))
        if members is None:
            continue
        axes = entry.get("axes") if isinstance(entry, dict) else None
        if not axes:
            continue
        key = ",".join(str(m) for m in members)
        merged = set(by_members.get(key, ())) | {str(a) for a in axes}
        by_members[key] = sorted(merged)
    return by_members


def _manifest_op_key(op: Any) -> str:
    if isinstance(op, (list, tuple)):
        return ":".join("null" if part is None else str(part) for part in op)
    return str(op)


def _load_manifest(bundle_dir: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Parse ``manifest.json`` into a canonical, lossless, compact form.

    The manifest is by construction a per-rank *sorted multiset* of ops (it is
    the AstraSim cache-key input, ``program/et_emit.py::_write_manifest``), so
    recording it as a per-rank ``op -> count`` map loses nothing and keeps the
    golden records small. The raw byte digest is pinned alongside it.
    """
    path = os.path.join(bundle_dir, "manifest.json")
    if not os.path.exists(path):
        return None, None
    with open(path, "rb") as fh:
        raw_bytes = fh.read()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    try:
        raw = json.loads(raw_bytes)
    except json.JSONDecodeError:
        return {"error": "unparseable manifest.json"}, digest
    ranks: Dict[str, Dict[str, int]] = {}
    for rank, ops in (raw.get("ranks") or {}).items():
        counts: Dict[str, int] = {}
        for op in ops:
            key = _manifest_op_key(op)
            counts[key] = counts.get(key, 0) + 1
        ranks[str(rank)] = dict(sorted(counts.items()))
    return (
        {
            "version": raw.get("version"),
            "npus": raw.get("npus"),
            "ranks": dict(sorted(ranks.items(), key=lambda kv: int(kv[0]))),
        },
        digest,
    )


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
    # The same relaxation computes the payload-weighted critical path.
    node_hash: Dict[int, str] = {}
    path_weight: Dict[int, int] = {}
    path_nodes: Dict[int, int] = {}
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
            path_weight[idx] = nodes[idx].weight + max(
                [path_weight[p] for p in parent_idxs] or [0]
            )
            path_nodes[idx] = 1 + max([path_nodes[p] for p in parent_idxs] or [0])
            remaining.discard(idx)
            progressed = True
    if remaining:
        raise ValueError(f"{et_path}: cycle detected in ctrl_deps ({len(remaining)} nodes unresolved)")

    dag_hash = _hash_str(",".join(sorted(node_hash.values())))
    ops_sorted = sorted(repr(n.payload) for n in nodes)
    ops_hash = _hash_str(";".join(ops_sorted))
    op_counts: Dict[str, int] = {}
    compute_micros = 0
    bytes_by_kind: Dict[str, int] = {}
    for n in nodes:
        op_counts[n.payload[0]] = op_counts.get(n.payload[0], 0) + 1
        if n.kind == "COMP":
            compute_micros += int(n.payload[1])
        elif n.kind == "COLL":
            bytes_by_kind["COLL"] = bytes_by_kind.get("COLL", 0) + int(n.payload[2])
        elif n.kind in ("SEND", "RECV"):
            bytes_by_kind[n.kind] = bytes_by_kind.get(n.kind, 0) + int(n.payload[1])

    return CanonicalRank(
        rank=rank,
        nodes=nodes,
        dag_hash=dag_hash,
        ops_hash=ops_hash,
        op_counts=op_counts,
        compute_micros=compute_micros,
        bytes_by_kind=bytes_by_kind,
        critical_path_nodes=max(path_nodes.values()) if path_nodes else 0,
        critical_path_weight=max(path_weight.values()) if path_weight else 0,
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
    manifest, manifest_sha = _load_manifest(bundle_dir)
    return CanonicalBundle(
        prefix=prefix,
        ranks=ranks,
        bundle_hash=bundle_hash,
        comm_groups=comm_groups,
        comm_axes=_load_comm_axes(bundle_dir, comm_groups),
        manifest=manifest,
        manifest_sha256=manifest_sha,
    )


# ---------------------------------------------------------------------------
# Diffing
# ---------------------------------------------------------------------------

#: bundle-level summary keys compared as opaque values, in report order
_BUNDLE_SCALARS = (
    "compute_micros_total",
    "n_transfers",
    "n_recv",
    "manifest_sha256",
)
_BUNDLE_MAPS = (
    "bytes_by_kind",
    "bytes_by_axis",
    "byte_hist_by_kind",
    "collectives_by_group",
    "comm_groups",
)
#: per-rank summary keys compared as opaque values, in report order
_RANK_SCALARS = (
    "compute_micros",
    "critical_path_nodes",
    "critical_path_weight",
    "n_nodes",
)


#: one structural difference: (field key, human message, golden value, candidate value).
#: The field key is stable across runs and is what the T4 bug ledger is keyed on.
BundleDiff = Tuple[str, str, Any, Any]


def _diff_maps(
    tag: str, field_prefix: str, name: str, golden: Any, candidate: Any
) -> List[BundleDiff]:
    """Key-wise diff of two dict summaries; falls back to whole-value diff."""
    if not isinstance(golden, dict) or not isinstance(candidate, dict):
        if golden != candidate:
            return [
                (
                    f"{field_prefix}{name}",
                    f"{tag}{name}: golden={golden} candidate={candidate}",
                    golden,
                    candidate,
                )
            ]
        return []
    out: List[BundleDiff] = []
    for key in sorted(set(golden) | set(candidate), key=str):
        gval, cval = golden.get(key), candidate.get(key)
        if gval == cval:
            continue
        out.append(
            (
                f"{field_prefix}{name}/{key}",
                f"{tag}{name}[{key}]: golden={gval} candidate={cval}",
                gval,
                cval,
            )
        )
    return out


def diff_bundles_detailed(
    golden: Dict[str, Any], candidate: Dict[str, Any], label: str = ""
) -> List[BundleDiff]:
    """Structured differences between two ``CanonicalBundle.summary()`` dicts."""
    out: List[BundleDiff] = []
    tag = f"[{label}] " if label else ""
    if golden["n_ranks"] != candidate["n_ranks"]:
        return [
            (
                "n_ranks",
                f"{tag}rank count differs: golden={golden['n_ranks']} candidate={candidate['n_ranks']}",
                golden["n_ranks"],
                candidate["n_ranks"],
            )
        ]
    for rank, gold in golden["ranks"].items():
        cand = candidate["ranks"].get(rank)
        if cand is None:
            out.append((f"rank/{rank}", f"{tag}rank {rank} missing from candidate", rank, None))
            continue
        if gold["ops_hash"] != cand["ops_hash"]:
            out.append(
                (
                    f"rank/{rank}/ops_hash",
                    f"{tag}rank {rank}: op multiset differs "
                    f"(golden counts {gold['op_counts']}, candidate {cand['op_counts']})",
                    gold["ops_hash"],
                    cand["ops_hash"],
                )
            )
        elif gold["dag_hash"] != cand["dag_hash"]:
            out.append(
                (
                    f"rank/{rank}/dag_hash",
                    f"{tag}rank {rank}: same ops but dependency DAG differs",
                    gold["dag_hash"],
                    cand["dag_hash"],
                )
            )
        for key in _RANK_SCALARS:
            if key in gold and gold.get(key) != cand.get(key):
                out.append(
                    (
                        f"rank/{rank}/{key}",
                        f"{tag}rank {rank} {key}: golden={gold.get(key)} candidate={cand.get(key)}",
                        gold.get(key),
                        cand.get(key),
                    )
                )
        if "bytes_by_kind" in gold or "bytes_by_kind" in cand:
            out.extend(
                _diff_maps(
                    f"{tag}rank {rank} ",
                    f"rank/{rank}/",
                    "bytes_by_kind",
                    gold.get("bytes_by_kind"),
                    cand.get("bytes_by_kind"),
                )
            )
    for key in _BUNDLE_SCALARS:
        if key in golden and golden.get(key) != candidate.get(key):
            out.append(
                (
                    key,
                    f"{tag}{key}: golden={golden.get(key)} candidate={candidate.get(key)}",
                    golden.get(key),
                    candidate.get(key),
                )
            )
    for key in _BUNDLE_MAPS:
        if key in golden or key in candidate:
            out.extend(_diff_maps(tag, "", key, golden.get(key), candidate.get(key)))
    if "manifest" in golden or "manifest" in candidate:
        out.extend(_diff_manifest(tag, golden.get("manifest"), candidate.get("manifest")))
    return out


def diff_bundles(
    golden: Dict[str, Any], candidate: Dict[str, Any], label: str = ""
) -> List[str]:
    """Human-readable differences between two ``CanonicalBundle.summary()`` dicts."""
    return [message for _field, message, _old, _new in
            diff_bundles_detailed(golden, candidate, label=label)]


def _diff_manifest(tag: str, golden: Any, candidate: Any) -> List[BundleDiff]:
    if golden == candidate:
        return []
    if not isinstance(golden, dict) or not isinstance(candidate, dict):
        return [("manifest", f"{tag}manifest: golden={golden!r} candidate={candidate!r}",
                 golden, candidate)]
    out: List[BundleDiff] = []
    for key in ("version", "npus"):
        if golden.get(key) != candidate.get(key):
            out.append(
                (
                    f"manifest/{key}",
                    f"{tag}manifest {key}: golden={golden.get(key)} candidate={candidate.get(key)}",
                    golden.get(key),
                    candidate.get(key),
                )
            )
    gold_ranks = golden.get("ranks") or {}
    cand_ranks = candidate.get("ranks") or {}
    if set(gold_ranks) != set(cand_ranks):
        out.append(
            (
                "manifest/ranks",
                f"{tag}manifest rank keys: golden-only={sorted(set(gold_ranks) - set(cand_ranks))} "
                f"candidate-only={sorted(set(cand_ranks) - set(gold_ranks))}",
                sorted(gold_ranks),
                sorted(cand_ranks),
            )
        )
        return out
    for rank in sorted(gold_ranks, key=int):
        out.extend(
            _diff_maps(
                f"{tag}manifest rank {rank} ",
                f"manifest/ranks/{rank}/",
                "ops",
                gold_ranks[rank],
                cand_ranks[rank],
            )
        )
    return out
