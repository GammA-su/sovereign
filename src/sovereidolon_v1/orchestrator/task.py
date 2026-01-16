from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..utils import read_json


class Example(BaseModel):
    inputs: Dict[str, Any]
    output: Any


class SealedTask(BaseModel):
    canary_token: str
    sealed_seed: int
    withheld_families: List[str] = Field(default_factory=list)


class Task(BaseModel):
    task_id: str
    family: str
    task_type: str
    goal: str
    inputs: Dict[str, str]
    output: str
    bounds: Dict[str, Any] = Field(default_factory=dict)
    examples: List[Example]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    sealed: Optional[SealedTask] = None

    def open_view(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "task_type": self.task_type,
            "goal": self.goal,
            "inputs": self.inputs,
            "output": self.output,
            "bounds": self.bounds,
            "examples": [example.model_dump() for example in self.examples],
            "metadata": self.metadata,
        }


def load_task(path: Path) -> Task:
    data = read_json(path)
    return Task(**data)
