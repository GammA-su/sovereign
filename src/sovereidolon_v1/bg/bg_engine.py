from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from ..ledger.ledger import Ledger
from ..schemas import BGRevisionOp
from ..utils import read_jsonl, stable_hash, write_jsonl_line


@dataclass
class ActiveView:
    active_nodes: List[str]
    conflicts: Dict[str, Dict[str, Any]]
    context_hash: str
    policy_version: str
    active_view_hash: str


def compute_context_hash(assumptions: Dict[str, Any]) -> str:
    return stable_hash(assumptions)


class BGEngine:
    def __init__(self, run_dir: Path, ledger: Ledger) -> None:
        self.run_dir = run_dir
        self.ledger = ledger
        self.log_path = run_dir / "bg" / "revisions.jsonl"
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.active: Dict[str, bool] = {}
        self.retracted: set[str] = set()
        self.superseded: set[str] = set()
        self.conflict_sets: Dict[str, List[str]] = {}
        self.resolutions: Dict[str, str] = {}

    def apply(self, op: BGRevisionOp, record: bool = True) -> None:
        if op.op == "ASSERT":
            if op.node_id is None:
                raise ValueError("node_id required")
            node_id = op.node_id
            self.nodes[node_id] = op.payload
            self.active[node_id] = True
        elif op.op == "SUPERSEDE":
            if op.old_id is None:
                raise ValueError("old_id required")
            if op.old_id not in self.nodes:
                raise ValueError("Unknown old_id")
            self.superseded.add(op.old_id)
            self.active[op.old_id] = False
            if op.new_id:
                self.nodes[op.new_id] = op.payload
                self.active[op.new_id] = True
        elif op.op == "PATCH":
            if op.node_id is None or op.diff is None:
                raise ValueError("node_id and diff required")
            if op.node_id not in self.nodes:
                raise ValueError("Unknown node_id")
            patched_id = stable_hash({"base": op.node_id, "diff": op.diff})
            self.nodes[patched_id] = {"base": op.node_id, "diff": op.diff}
            self.active[op.node_id] = False
            self.active[patched_id] = True
            self.superseded.add(op.node_id)
        elif op.op == "RETRACT":
            if op.node_id is None:
                raise ValueError("node_id required")
            if op.node_id not in self.nodes:
                raise ValueError("Unknown node_id")
            self.retracted.add(op.node_id)
            self.active[op.node_id] = False
        elif op.op == "DECLARE_CONFLICT":
            if op.conflict_id is None:
                raise ValueError("conflict_id required")
            self.conflict_sets[op.conflict_id] = list(op.conflict_set)
        elif op.op == "RESOLVE_CONFLICT":
            if op.conflict_id is None or op.chosen_id is None:
                raise ValueError("conflict_id and chosen_id required")
            if op.conflict_id not in self.conflict_sets:
                raise ValueError("Unknown conflict_id")
            self.resolutions[op.conflict_id] = op.chosen_id
        else:
            raise ValueError("Unknown op")

        if record:
            write_jsonl_line(self.log_path, op.model_dump())
            self.ledger.append(
                "BG_OP_APPLIED",
                {
                    "op": op.op,
                    "node_id": op.node_id,
                    "old_id": op.old_id,
                    "new_id": op.new_id,
                    "conflict_id": op.conflict_id,
                    "witness_id": op.witness_id,
                },
            )

    def compute_active_view(self, context_hash: str, policy_version: str) -> ActiveView:
        active_nodes = {node_id for node_id, is_active in self.active.items() if is_active}
        active_nodes -= self.retracted
        active_nodes -= self.superseded

        conflicts_summary: Dict[str, Dict[str, Any]] = {}
        for conflict_id, members in self.conflict_sets.items():
            chosen = self.resolutions.get(conflict_id)
            conflicts_summary[conflict_id] = {
                "members": sorted(members),
                "chosen": chosen,
            }
            if chosen is None:
                active_nodes -= set(members)
            else:
                active_nodes -= set(member for member in members if member != chosen)

        active_list = sorted(active_nodes)
        active_view_hash = stable_hash(
            {
                "context_hash": context_hash,
                "policy_version": policy_version,
                "active_nodes": active_list,
                "conflicts": conflicts_summary,
            }
        )
        return ActiveView(
            active_nodes=active_list,
            conflicts=conflicts_summary,
            context_hash=context_hash,
            policy_version=policy_version,
            active_view_hash=active_view_hash,
        )

    @staticmethod
    def replay(log_path: Path, context_hash: str, policy_version: str) -> ActiveView:
        entries = read_jsonl(log_path)
        engine = BGEngine(log_path.parent.parent, Ledger(Path("/dev/null")))
        for entry in entries:
            op = BGRevisionOp(**entry)
            engine.apply(op, record=False)
        return engine.compute_active_view(context_hash, policy_version)

    @staticmethod
    def replay_from_ops(
        ops: List[BGRevisionOp], context_hash: str, policy_version: str
    ) -> ActiveView:
        engine = BGEngine(Path("."), Ledger(Path("/dev/null")))
        for op in ops:
            engine.apply(op, record=False)
        return engine.compute_active_view(context_hash, policy_version)
