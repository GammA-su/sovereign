from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List, Optional

from ..orchestrator.task import Task
from ..schemas import BGRevisionOp, VerifierVerdict
from ..utils import stable_hash


@dataclass
class ForgeDecision:
    decision: str
    reason: str
    mdl_score: int


class ForgeGate:
    def __init__(self) -> None:
        pass

    def decide(
        self,
        task: Task,
        program: Any,
        verdicts: List[VerifierVerdict],
        required_passed: bool,
    ) -> ForgeDecision:
        program_json = json.dumps(program.to_json(), sort_keys=True)
        mdl_score = len(program_json)
        if task.sealed and task.sealed.canary_token in program_json:
            return ForgeDecision(decision="QUARANTINE", reason="canary_echo", mdl_score=mdl_score)
        if not required_passed:
            return ForgeDecision(
                decision="REJECT", reason="required_lanes_failed", mdl_score=mdl_score
            )
        if task.sealed and task.family in task.sealed.withheld_families:
            return ForgeDecision(decision="REJECT", reason="family_disjoint", mdl_score=mdl_score)
        for verdict in verdicts:
            if verdict.soundness_grade == "HEURISTIC" and verdict.verdict == "PASS":
                return ForgeDecision(
                    decision="REJECT", reason="heuristic_only", mdl_score=mdl_score
                )
        return ForgeDecision(decision="PROMOTE", reason="passes", mdl_score=mdl_score)

    def build_promotion_op(self, program: Any, witness_id: str) -> BGRevisionOp:
        node_id = stable_hash(program.to_json())
        return BGRevisionOp(
            op="ASSERT", witness_id=witness_id, node_id=node_id, payload=program.to_json()
        )

    def build_reject_op(self, witness_id: str, reason: str) -> Optional[BGRevisionOp]:
        return BGRevisionOp(
            op="RETRACT",
            witness_id=witness_id,
            node_id=stable_hash({"reject": reason, "witness": witness_id}),
            reason=reason,
        )
