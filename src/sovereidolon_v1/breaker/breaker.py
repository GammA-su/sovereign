from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..bvps.interpreter import eval_program
from ..codepatch.breaker import run_codepatch_breaker
from ..config import Settings
from ..jsonspec.program import JsonSpecProgram
from ..orchestrator.specs import TaskSpec
from ..orchestrator.task import Example, Task
from ..schemas import BreakerKPI
from ..utils import read_jsonl, write_jsonl_line
from .jsonspec_breaker import run_jsonspec_breaker
from .pyfunc_breaker import BreakerBudget, BreakerResult, run_pyfunc_breaker


def _fingerprint(inputs: Dict[str, Any]) -> List[str]:
    if "xs" in inputs:
        xs = list(inputs["xs"])
        return [
            f"len:{len(xs)}",
            f"sum:{sum(xs)}",
            f"max:{max(xs) if xs else 0}",
        ]
    return [f"x:{inputs.get('x')}", f"y:{inputs.get('y')}"]


def _jaccard(a: List[str], b: List[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _novelty_score(inputs: Dict[str, Any], history: List[List[str]]) -> float:
    if not history:
        return 1.0
    fp = _fingerprint(inputs)
    sims = [_jaccard(fp, past) for past in history]
    return 1.0 - max(sims)


def _mutate_inputs(base: Dict[str, Any], family: str, rng: random.Random) -> Dict[str, Any]:
    if "xs" in base:
        xs = list(base["xs"])
        if family == "permute":
            rng.shuffle(xs)
        elif family == "duplicate":
            xs = xs + xs
        elif family == "scale":
            xs = [x * 2 for x in xs]
        elif family == "shift":
            xs = [x + 1 for x in xs]
        return {"xs": xs}
    x = int(base["x"])
    y = int(base["y"])
    if family == "scale":
        return {"x": x * 2, "y": y * 2}
    if family == "shift":
        return {"x": x + 1, "y": y - 1}
    if family == "permute":
        return {"x": y, "y": x}
    return base


def _minimize_failure(
    inputs: Dict[str, Any], spec: TaskSpec, program: Any, step_limit: int
) -> tuple[Dict[str, Any], int]:
    if "xs" in inputs:
        xs = list(inputs["xs"])
        steps = 0
        while len(xs) > 1:
            steps += 1
            candidate_list = {"xs": xs[: len(xs) // 2]}
            expected = spec.evaluate(candidate_list)
            actual, _ = eval_program(program, candidate_list, step_limit)
            if expected != actual:
                xs = candidate_list["xs"]
            else:
                break
        return {"xs": xs}, steps
    x = int(inputs["x"])
    y = int(inputs["y"])
    steps = 0
    for dx, dy in [(0, 0), (x // 2, y // 2), (x - 1, y - 1)]:
        steps += 1
        candidate_int = {"x": dx, "y": dy}
        expected = spec.evaluate(candidate_int)
        actual, _ = eval_program(program, candidate_int, step_limit)
        if expected != actual:
            return candidate_int, steps
    return inputs, steps


class BreakerLab:
    def __init__(self, settings: Settings, run_dir: Path) -> None:
        self.settings = settings
        self.run_dir = run_dir
        self.novelty_path = run_dir / "capsules" / "novelty.jsonl"

    def _load_history(self) -> List[List[str]]:
        entries = read_jsonl(self.novelty_path)
        return [entry.get("fingerprint", []) for entry in entries]

    def _record_novelty(self, inputs: Dict[str, Any]) -> None:
        write_jsonl_line(self.novelty_path, {"fingerprint": _fingerprint(inputs)})

    def run(
        self,
        task: Task,
        program: Any,
        spec: TaskSpec,
        tests: List[Example],
        budget: int,
        seed: int,
    ) -> BreakerResult:
        if task.task_type == "pyfunc":
            program_path = self.run_dir / "artifacts" / "pyfunc" / "program.py"
            return run_pyfunc_breaker(
                task,
                program_path,
                BreakerBudget(attempt_budget=budget),
            )
        if task.task_type == "codepatch":
            return run_codepatch_breaker(
                task,
                program,
                BreakerBudget(attempt_budget=budget),
                self.run_dir,
            )
        if task.task_type == "jsonspec":
            if not isinstance(program, JsonSpecProgram):
                kpi = BreakerKPI(
                    CDR=0.0,
                    TMR=0.0,
                    NOVN=0.0,
                    WFHR=0.0,
                    window={"attempts": 0},
                    budget={"attempt_budget": budget},
                )
                report_data: Dict[str, Any] = {
                    "counterexample": None,
                    "minimized": None,
                    "attempts": 0,
                    "skipped": True,
                }
                return BreakerResult(
                    counterexample=None, minimized=None, kpi=kpi, report=report_data
                )
            return run_jsonspec_breaker(
                task,
                program,
                BreakerBudget(attempt_budget=budget),
                self.run_dir,
            )

        if task.task_type not in {"arith", "list"}:
            kpi = BreakerKPI(
                CDR=0.0,
                TMR=0.0,
                NOVN=0.0,
                WFHR=0.0,
                window={"attempts": 0},
                budget={"attempt_budget": budget},
            )
            report_payload: Dict[str, Any] = {
                "counterexample": None,
                "minimized": None,
                "attempts": 0,
                "skipped": True,
            }
            return BreakerResult(
                counterexample=None, minimized=None, kpi=kpi, report=report_payload
            )

        rng = random.Random(seed)
        start = time.time_ns()
        attempts = 0
        found: Optional[Example] = None
        minimized: Optional[Example] = None
        if task.sealed and task.sealed.withheld_families:
            withheld_families = list(task.sealed.withheld_families)
        else:
            withheld_families = list(self.settings.withheld_families)
        withheld_hits = 0
        withheld_trials = 0

        bases = [example.inputs for example in tests]
        families = ["permute", "duplicate", "scale", "shift"]

        for family in withheld_families:
            if attempts >= budget:
                break
            attempts += 1
            base = rng.choice(bases)
            mutated = _mutate_inputs(base, family, rng)
            expected = spec.evaluate(mutated)
            actual, _ = eval_program(program, mutated, self.settings.verify_budget_steps)
            withheld_trials += 1
            if expected != actual:
                withheld_hits += 1
                found = Example(inputs=mutated, output=expected)
                break

        if found is None:
            for _ in range(budget - attempts):
                attempts += 1
                base = rng.choice(bases)
                family = rng.choice(families)
                mutated = _mutate_inputs(base, family, rng)
                expected = spec.evaluate(mutated)
                actual, _ = eval_program(program, mutated, self.settings.verify_budget_steps)
                if family in withheld_families:
                    withheld_trials += 1
                    if expected != actual:
                        withheld_hits += 1
                if expected != actual:
                    found = Example(inputs=mutated, output=expected)
                    break

        if found is None:
            for _ in range(budget // 2):
                attempts += 1
                candidate = spec.random_inputs(rng)
                expected = spec.evaluate(candidate)
                actual, _ = eval_program(program, candidate, self.settings.verify_budget_steps)
                if expected != actual:
                    found = Example(inputs=candidate, output=expected)
                    break

        tmr = 0.0
        if found is not None:
            minimized_inputs, minimized_steps = _minimize_failure(
                found.inputs, spec, program, self.settings.verify_budget_steps
            )
            minimized = Example(inputs=minimized_inputs, output=spec.evaluate(minimized_inputs))
            tmr = minimized_steps / max(1, attempts)
        else:
            minimized_steps = 0

        history = self._load_history()[-self.settings.breaker_novelty_window :]
        nov_score = _novelty_score(found.inputs if found else tests[0].inputs, history)
        if found is not None:
            self._record_novelty(found.inputs)

        cdr = 1.0 / max(1, attempts) if found is not None else 0.0
        wfhr = withheld_hits / max(1, withheld_trials)
        kpi = BreakerKPI(
            CDR=cdr,
            TMR=tmr,
            NOVN=nov_score,
            WFHR=wfhr,
            window={"attempts": attempts},
            budget={"attempt_budget": budget},
        )
        report: Dict[str, Any] = {
            "counterexample": found.model_dump() if found else None,
            "minimized": minimized.model_dump() if minimized else None,
            "attempts": attempts,
            "duration_ns": time.time_ns() - start,
            "withheld_trials": withheld_trials,
            "withheld_hits": withheld_hits,
            "minimized_steps": minimized_steps,
        }
        return BreakerResult(counterexample=found, minimized=minimized, kpi=kpi, report=report)
