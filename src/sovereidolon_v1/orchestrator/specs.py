from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from ..bvps.interpreter import eval_program
from ..orchestrator.task import Example, Task


@dataclass
class TaskSpec:
    task: Task

    def evaluate(self, inputs: Dict[str, Any]) -> Any:
        if self.task.task_type == "arith":
            x = int(inputs["x"])
            y = int(inputs["y"])
            if self.task.goal == "add":
                return x + y
            if self.task.goal == "mul":
                return x * y
            raise ValueError("unknown arith goal")
        if self.task.task_type == "list":
            xs = list(inputs["xs"])
            if self.task.goal == "sum":
                return int(sum(xs))
            if self.task.goal == "max":
                return int(max(xs)) if xs else 0
            raise ValueError("unknown list goal")
        if self.task.task_type == "bg":
            return True
        if self.task.task_type == "bool":
            arg_names = list(self.task.inputs.keys())
            a_val = bool(inputs[arg_names[0]])
            b_val = bool(inputs[arg_names[1]])
            if self.task.goal == "and":
                return a_val and b_val
            if self.task.goal == "or":
                return a_val or b_val
            if self.task.goal == "xor":
                return (a_val and not b_val) or (not a_val and b_val)
            raise ValueError("unknown bool goal")
        if self.task.task_type == "pyfunc":
            int_values = [
                int(value)
                for name, value in inputs.items()
                if self.task.inputs.get(name) == "Int"
            ]
            if self.task.goal == "sum":
                return sum(int_values)
            if self.task.goal == "max":
                return max(int_values) if int_values else 0
            if self.task.goal == "min":
                return min(int_values) if int_values else 0
            return sum(int_values)
        if self.task.task_type == "codepatch":
            return True
        raise ValueError("unknown task type")

    def random_inputs(self, rng: random.Random) -> Dict[str, Any]:
        if self.task.task_type == "arith":
            x_min, x_max = self.task.bounds.get("x", [-5, 5])
            y_min, y_max = self.task.bounds.get("y", [-5, 5])
            return {
                "x": rng.randint(int(x_min), int(x_max)),
                "y": rng.randint(int(y_min), int(y_max)),
            }
        if self.task.task_type == "list":
            len_min, len_max = self.task.bounds.get("xs_len", [0, 5])
            elem_min, elem_max = self.task.bounds.get("xs_elem", [-5, 5])
            length = rng.randint(int(len_min), int(len_max))
            return {
                "xs": [rng.randint(int(elem_min), int(elem_max)) for _ in range(length)]
            }
        if self.task.task_type == "bg":
            return {"state": {}}
        if self.task.task_type == "bool":
            inputs: Dict[str, Any] = {}
            for name in self.task.inputs.keys():
                inputs[name] = bool(rng.randint(0, 1))
            return inputs
        if self.task.task_type == "codepatch":
            return {}
        raise ValueError("unknown task type")

    def generate_inputs(self, count: int, seed: int) -> Iterable[Dict[str, Any]]:
        rng = random.Random(seed)
        for _ in range(count):
            yield self.random_inputs(rng)

    def find_counterexample(
        self,
        program: Any,
        tests: List[Example],
        budget: int,
        seed: int,
        step_limit: int,
    ) -> Optional[Example]:
        rng = random.Random(seed)
        for _ in range(budget):
            candidate = self.random_inputs(rng)
            expected = self.evaluate(candidate)
            actual, _ = eval_program(program, candidate, step_limit=step_limit)
            if actual != expected:
                return Example(inputs=candidate, output=expected)
        return None


def task_spec(task: Task) -> TaskSpec:
    return TaskSpec(task=task)
