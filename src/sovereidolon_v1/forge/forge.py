from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List, Optional

from ..config import AdmissionPolicy
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
        admission_policy: AdmissionPolicy,
        controller_overhead_ratio: float,
        withheld_hits: int,
    ) -> ForgeDecision:
        program_json = json.dumps(program.to_json(), sort_keys=True)
        mdl_score = len(program_json)
        if task.sealed and task.sealed.canary_token in program_json:
            return ForgeDecision(decision="QUARANTINE", reason="canary_echo", mdl_score=mdl_score)
        if not required_passed:
            return ForgeDecision(
                decision="REJECT", reason="required_lanes_failed", mdl_score=mdl_score
            )
        if controller_overhead_ratio > admission_policy.max_controller_overhead:
            return ForgeDecision(
                decision="REJECT", reason="controller_overhead", mdl_score=mdl_score
            )
        if task.sealed and admission_policy.reject_on_withheld_hits and withheld_hits > 0:
            return ForgeDecision(decision="REJECT", reason="withheld_hits", mdl_score=mdl_score)
        if task.sealed and task.family in task.sealed.withheld_families:
            return ForgeDecision(decision="REJECT", reason="family_disjoint", mdl_score=mdl_score)
        has_strong_pass = any(
            verdict.verdict == "PASS" and verdict.soundness_grade in {"CERT", "BOUNDED"}
            for verdict in verdicts
        )
        has_heuristic_pass = any(
            verdict.verdict == "PASS" and verdict.soundness_grade == "HEURISTIC"
            for verdict in verdicts
        )
        if has_heuristic_pass and not has_strong_pass:
            return ForgeDecision(decision="REJECT", reason="heuristic_only", mdl_score=mdl_score)
        return ForgeDecision(decision="ADMIT", reason="passes", mdl_score=mdl_score)

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
