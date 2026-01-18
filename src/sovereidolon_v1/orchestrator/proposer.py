from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..codepatch.program import CodePatchProgram, compute_codepatch_hash
from ..jsonspec.program import JsonSpecProgram, compute_jsonspec_hash
from ..pyfunc.program import PyFuncProgram, compute_pyfunc_hash
from .task import Task


@dataclass(frozen=True)
class ProposerBudget:
    break_budget_attempts: int
    verify_budget_steps: int


@dataclass(frozen=True)
class StoreCandidate:
    program: Any
    program_hash: str
    domain: str
    source: str


@dataclass(frozen=True)
class ProposerInput:
    task: Task
    spec_signature: str
    store_candidates: List[StoreCandidate]
    budget: ProposerBudget


@dataclass
class ProposerOutput:
    candidate_program: Optional[Any]
    candidate_hash: str
    rationale: str
    predicted_lane_costs: Dict[str, int]
    risk_flags: List[str]
    source: str

    def report(self) -> Dict[str, Any]:
        return {
            "candidate_hash": self.candidate_hash,
            "rationale": self.rationale,
            "predicted_lane_costs": dict(self.predicted_lane_costs),
            "risk_flags": list(self.risk_flags),
            "source": self.source,
        }


class ProposerStub:
    def propose(self, request: ProposerInput) -> ProposerOutput:
        predicted_lane_costs = {"verify_ns": 0, "breaker_ns": 0}
        if request.store_candidates:
            candidate = sorted(
                request.store_candidates, key=lambda item: item.program_hash
            )[0]
            return ProposerOutput(
                candidate_program=candidate.program,
                candidate_hash=candidate.program_hash,
                rationale="warm_start_store",
                predicted_lane_costs=predicted_lane_costs,
                risk_flags=["WARM_START"],
                source="store",
            )

        task = request.task
        if task.task_type == "pyfunc":
            candidate_code = task.metadata.get("pyfunc", {}).get("candidate_program", "")
            if candidate_code:
                program = PyFuncProgram(candidate_code)
                return ProposerOutput(
                    candidate_program=program,
                    candidate_hash=compute_pyfunc_hash(program.code),
                    rationale="task_metadata",
                    predicted_lane_costs=predicted_lane_costs,
                    risk_flags=[],
                    source="metadata",
                )
        if task.task_type == "codepatch":
            candidate_patch = task.metadata.get("codepatch", {}).get("candidate_patch", "")
            if candidate_patch:
                patch_program = CodePatchProgram(candidate_patch)
                return ProposerOutput(
                    candidate_program=patch_program,
                    candidate_hash=compute_codepatch_hash(patch_program.patch),
                    rationale="task_metadata",
                    predicted_lane_costs=predicted_lane_costs,
                    risk_flags=[],
                    source="metadata",
                )
        if task.task_type == "jsonspec":
            candidate_spec = task.metadata.get("jsonspec", {}).get("candidate_program")
            if candidate_spec:
                json_program = JsonSpecProgram(candidate_spec)
                return ProposerOutput(
                    candidate_program=json_program,
                    candidate_hash=compute_jsonspec_hash(json_program.spec),
                    rationale="task_metadata",
                    predicted_lane_costs=predicted_lane_costs,
                    risk_flags=[],
                    source="metadata",
                )

        return ProposerOutput(
            candidate_program=None,
            candidate_hash="",
            rationale="cegis",
            predicted_lane_costs=predicted_lane_costs,
            risk_flags=["CEGIS_REQUIRED"],
            source="cegis",
        )
