from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from ..config import Settings
from ..orchestrator.specs import task_spec
from ..orchestrator.task import Example, Task
from ..utils import stable_hash
from .interpreter import INTERPRETER_HASH, eval_program
from .synth import candidate_programs


@dataclass
class CEGISResult:
    status: str
    program: Optional[Any]
    tests: List[Example]
    counterexamples: List[Example]
    ast_hash: str
    interpreter_hash: str
    trace_hashes: List[str]
    failure_reason: str = ""


def _passes_tests(program: Any, tests: List[Example], step_limit: int) -> bool:
    for example in tests:
        value, _ = eval_program(program, example.inputs, step_limit=step_limit)
        if value != example.output:
            return False
    return True


def run_cegis(task: Task, settings: Settings, rng_seed: int) -> CEGISResult:
    spec_fn = task_spec(task)
    tests = list(task.examples)
    counterexamples: List[Example] = []
    trace_hashes: List[str] = []

    for program in candidate_programs(task):
        if not _passes_tests(program, tests, step_limit=settings.verify_budget_steps):
            continue
        counterexample = spec_fn.find_counterexample(
            program,
            tests,
            budget=settings.break_budget_attempts,
            seed=rng_seed,
            step_limit=settings.verify_budget_steps,
        )
        if counterexample is not None:
            tests.append(counterexample)
            counterexamples.append(counterexample)
            continue
        for example in tests:
            _, trace_hash = eval_program(
                program, example.inputs, step_limit=settings.verify_budget_steps
            )
            trace_hashes.append(trace_hash)
        ast_hash = stable_hash(program.to_json())
        return CEGISResult(
            status="ok",
            program=program,
            tests=tests,
            counterexamples=counterexamples,
            ast_hash=ast_hash,
            interpreter_hash=INTERPRETER_HASH,
            trace_hashes=trace_hashes,
        )

    return CEGISResult(
        status="fail",
        program=None,
        tests=tests,
        counterexamples=counterexamples,
        ast_hash="",
        interpreter_hash=INTERPRETER_HASH,
        trace_hashes=trace_hashes,
        failure_reason="CEGIS_UNSAT",
    )
