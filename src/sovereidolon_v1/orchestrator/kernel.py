from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .task import Task


@dataclass
class Interpretation:
    interpretation_id: str
    description: str
    assumptions: Dict[str, Any]


class KernelStub:
    def propose_interpretations(self, task: Task) -> List[Interpretation]:
        interpretations: List[Interpretation] = []
        if task.task_type == "arith":
            interpretations.append(
                Interpretation(
                    interpretation_id="add",
                    description="Compute x + y",
                    assumptions={"goal": "add"},
                )
            )
            interpretations.append(
                Interpretation(
                    interpretation_id="mul",
                    description="Compute x * y",
                    assumptions={"goal": "mul"},
                )
            )
        elif task.task_type == "list":
            interpretations.append(
                Interpretation(
                    interpretation_id="sum",
                    description="Sum list elements",
                    assumptions={"goal": "sum"},
                )
            )
            interpretations.append(
                Interpretation(
                    interpretation_id="max",
                    description="Max of list elements",
                    assumptions={"goal": "max"},
                )
            )
        else:
            interpretations.append(
                Interpretation(
                    interpretation_id=task.goal,
                    description="Task-specific",
                    assumptions={"goal": task.goal},
                )
            )
        return interpretations[:4]

    def choose_interpretation(
        self, task: Task, interpretations: List[Interpretation]
    ) -> Interpretation:
        for interp in interpretations:
            if interp.assumptions.get("goal") == task.goal:
                return interp
        return interpretations[0]

    def propose_patch(self, failure_capsule: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "noop", "reason": "stub"}
