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

"""Causal deadlock simulator for Chakra ET bundles under AstraSim's discipline.

Replays a bundle using AstraSim's scheduling contract (reverse-engineered from
``astra-sim/workload/Workload.cc`` + ``HardwareResource.cc``), ignoring time:

- Per rank, ONE compute op and ONE comm (SEND or COLL) op in flight at a time.
- Among ready ops competing for a slot, the lowest node id issues first.
- RECV ops never occupy a slot; they are posted as soon as deps are met.
- SEND completes unconditionally once issued (delivery is buffered).
- RECV completes once the matching SEND (src, dst, tag) has been issued.
- A collective completes when every member of its communicator group has
  issued its k-th collective for that group, where k is the per-rank issue
  ORDER of that group's collectives (AstraSim matches by order, not name).

If the bundle cannot run to completion under this contract, the real
simulator deadlocks silently. ``explain()`` reports the blocked frontier and
a wait-for cycle. This is both a legacy-debugging tool and a permanent
property check: every emitted bundle must satisfy ``simulate().completed``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .canonical import _attr_map, _load_et_nodes, pb  # reuse loaders


@dataclass
class _Op:
    rank: int
    node_id: int
    kind: str  # COMP | COLL | SEND | RECV
    name: str
    deps: List[int]
    # SEND/RECV: (src, dst, tag); COLL: (group_key, seq_assigned_at_issue)
    peer: Optional[Tuple[int, int, int]] = None
    group: Optional[str] = None
    coll_seq: Optional[int] = None


@dataclass
class SimResult:
    completed: bool
    done_counts: Dict[int, int]
    total_counts: Dict[int, int]
    blocked_report: List[str] = field(default_factory=list)
    cycle: List[str] = field(default_factory=list)


class BundleSim:
    def __init__(self, bundle_dir: str, prefix: str = "llm_graph") -> None:
        import json
        import os
        import re

        rank_re = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.et$")
        self.ranks: Dict[int, Dict[int, _Op]] = {}
        groups_path = os.path.join(bundle_dir, "comm_groups.json")
        self.comm_groups: Dict[str, List[int]] = {}
        if os.path.exists(groups_path):
            with open(groups_path) as fh:
                self.comm_groups = {
                    str(k): sorted(int(x) for x in v) for k, v in json.load(fh).items()
                }
        rank_files = {}
        for entry in os.listdir(bundle_dir):
            m = rank_re.match(entry)
            if m:
                rank_files[int(m.group(1))] = os.path.join(bundle_dir, entry)
        self.all_ranks = sorted(rank_files)
        for rank, path in rank_files.items():
            ops: Dict[int, _Op] = {}
            for node in _load_et_nodes(path):
                attrs = _attr_map(node)
                nid = int(node.id)
                deps = [int(d) for d in node.ctrl_deps]
                if node.type == pb.COMP_NODE:
                    ops[nid] = _Op(rank, nid, "COMP", node.name, deps)
                elif node.type == pb.COMM_COLL_NODE:
                    group = str(attrs.get("pg_name", "__world__"))
                    ops[nid] = _Op(rank, nid, "COLL", node.name, deps, group=group)
                elif node.type == pb.COMM_SEND_NODE:
                    peer = (rank, int(attrs["comm_dst"]), int(attrs["comm_tag"]))
                    ops[nid] = _Op(rank, nid, "SEND", node.name, deps, peer=peer)
                elif node.type == pb.COMM_RECV_NODE:
                    peer = (int(attrs["comm_src"]), rank, int(attrs["comm_tag"]))
                    ops[nid] = _Op(rank, nid, "RECV", node.name, deps, peer=peer)
                else:
                    raise ValueError(f"rank {rank} node {nid}: unknown type {node.type}")
            self.ranks[rank] = ops

    def group_members(self, group: str) -> List[int]:
        if group == "__world__":
            return list(self.all_ranks)
        members = self.comm_groups.get(group)
        if members is None:
            raise ValueError(f"pg_name {group} not found in comm_groups.json")
        return members

    def simulate(self) -> SimResult:
        done: Dict[int, Set[int]] = {r: set() for r in self.ranks}
        issued: Dict[int, Set[int]] = {r: set() for r in self.ranks}
        # (group, seq) -> set of ranks that issued that slot; seq per (rank, group)
        coll_issue_count: Dict[Tuple[int, str], int] = {}
        coll_streams: Dict[Tuple[str, int], Set[int]] = {}
        issued_sends: Set[Tuple[int, int, int]] = set()
        inflight_comm: Dict[int, Optional[int]] = {r: None for r in self.ranks}
        inflight_comp: Dict[int, Optional[int]] = {r: None for r in self.ranks}
        inflight_recvs: Dict[int, List[int]] = {r: [] for r in self.ranks}

        def deps_met(op: _Op) -> bool:
            return all(d in done[op.rank] for d in op.deps)

        def try_complete(rank: int) -> bool:
            """Complete any in-flight op whose completion condition holds."""
            progressed = False
            cid = inflight_comp[rank]
            if cid is not None:
                done[rank].add(cid)
                inflight_comp[rank] = None
                progressed = True
            mid = inflight_comm[rank]
            if mid is not None:
                op = self.ranks[rank][mid]
                if op.kind == "SEND":
                    done[rank].add(mid)
                    inflight_comm[rank] = None
                    progressed = True
                elif op.kind == "COLL":
                    stream = coll_streams[(op.group, op.coll_seq)]
                    if stream >= set(self.group_members(op.group)):
                        done[rank].add(mid)
                        inflight_comm[rank] = None
                        progressed = True
            still = []
            for rid in inflight_recvs[rank]:
                op = self.ranks[rank][rid]
                if op.peer in issued_sends:
                    done[rank].add(rid)
                    progressed = True
                else:
                    still.append(rid)
            inflight_recvs[rank] = still
            return progressed

        def try_issue(rank: int) -> bool:
            progressed = False
            ready = sorted(
                nid
                for nid, op in self.ranks[rank].items()
                if nid not in issued[rank] and deps_met(op)
            )
            for nid in ready:
                op = self.ranks[rank][nid]
                if op.kind == "RECV":
                    issued[rank].add(nid)
                    inflight_recvs[rank].append(nid)
                    progressed = True
                elif op.kind == "COMP":
                    if inflight_comp[rank] is None:
                        issued[rank].add(nid)
                        inflight_comp[rank] = nid
                        progressed = True
                else:  # SEND / COLL share the single comm slot
                    if inflight_comm[rank] is None:
                        issued[rank].add(nid)
                        inflight_comm[rank] = nid
                        if op.kind == "SEND":
                            issued_sends.add(op.peer)
                        else:
                            seq = coll_issue_count.get((rank, op.group), 0)
                            coll_issue_count[(rank, op.group)] = seq + 1
                            op.coll_seq = seq
                            coll_streams.setdefault((op.group, seq), set()).add(rank)
                        progressed = True
            return progressed

        progressed = True
        while progressed:
            progressed = False
            for rank in self.ranks:
                if try_complete(rank):
                    progressed = True
                if try_issue(rank):
                    progressed = True

        total = {r: len(ops) for r, ops in self.ranks.items()}
        counts = {r: len(done[r]) for r in self.ranks}
        completed = all(counts[r] == total[r] for r in self.ranks)
        result = SimResult(completed=completed, done_counts=counts, total_counts=total)
        if not completed:
            result.blocked_report, result.cycle = self._explain(
                done, issued, inflight_comm, inflight_recvs, coll_streams, issued_sends
            )
        return result

    def _explain(
        self,
        done: Dict[int, Set[int]],
        issued: Dict[int, Set[int]],
        inflight_comm: Dict[int, Optional[int]],
        inflight_recvs: Dict[int, List[int]],
        coll_streams: Dict[Tuple[str, int], Set[int]],
        issued_sends: Set[Tuple[int, int, int]],
    ) -> Tuple[List[str], List[str]]:
        report: List[str] = []
        # wait-for graph over ("rank:node_id") items
        waits: Dict[str, Set[str]] = {}

        def key(rank: int, nid: int) -> str:
            op = self.ranks[rank][nid]
            return f"r{rank}#{nid}:{op.kind}:{op.name}"

        for rank, ops in self.ranks.items():
            for nid, op in ops.items():
                if nid in done[rank]:
                    continue
                k = key(rank, nid)
                blockers: Set[str] = set()
                for dep in op.deps:
                    if dep not in done[rank]:
                        blockers.add(key(rank, dep))
                if nid in issued[rank]:
                    if op.kind == "RECV" and op.peer not in issued_sends:
                        src = op.peer[0]
                        matches = [
                            key(src, snid)
                            for snid, sop in self.ranks[src].items()
                            if sop.kind == "SEND" and sop.peer == op.peer
                        ]
                        if matches:
                            blockers.update(matches)
                        else:
                            report.append(f"{k}: NO MATCHING SEND for (src,dst,tag)={op.peer}")
                    if op.kind == "COLL":
                        stream = coll_streams.get((op.group, op.coll_seq), set())
                        missing = set(self.group_members(op.group)) - stream
                        for m in missing:
                            blockers.add(f"r{m}:<next {op.group} collective not yet issued>")
                else:
                    # not issued: maybe the slot is busy
                    if op.kind in ("SEND", "COLL") and inflight_comm[rank] is not None:
                        blockers.add(key(rank, inflight_comm[rank]))
                if blockers:
                    waits[k] = blockers

        frontier = [
            k
            for k, blockers in waits.items()
            if not any(b in waits for b in blockers)
        ]
        for rank in sorted(self.ranks):
            stuck = [k for k in waits if k.startswith(f"r{rank}#")]
            report.append(f"rank {rank}: {len(stuck)} not-done")
        # find a cycle with DFS
        color: Dict[str, int] = {}
        stack: List[str] = []
        cycle: List[str] = []

        def dfs(u: str) -> bool:
            color[u] = 1
            stack.append(u)
            for v in waits.get(u, ()):  # noqa: B007
                if v not in waits:
                    continue
                c = color.get(v, 0)
                if c == 0 and dfs(v):
                    return True
                if c == 1:
                    idx = stack.index(v)
                    cycle.extend(stack[idx:] + [v])
                    return True
            color[u] = 2
            stack.pop()
            return False

        for u in list(waits):
            if color.get(u, 0) == 0 and dfs(u):
                break
        return report, cycle


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir")
    parser.add_argument("--prefix", default="llm_graph")
    args = parser.parse_args(argv)

    sim = BundleSim(args.bundle_dir, prefix=args.prefix)
    result = sim.simulate()
    print(f"completed: {result.completed}")
    print(f"progress: { {r: f'{result.done_counts[r]}/{result.total_counts[r]}' for r in sorted(result.done_counts)} }")
    for line in result.blocked_report[:20]:
        print("  ", line)
    if result.cycle:
        print("wait-for cycle:")
        for item in result.cycle:
            print("   ->", item)
    return 0 if result.completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
