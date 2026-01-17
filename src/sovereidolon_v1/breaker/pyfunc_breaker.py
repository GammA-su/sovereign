from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..orchestrator.specs import TaskSpec, task_spec
from ..orchestrator.task import Example, Task
from ..pyfunc.program import compute_pyfunc_hash
from ..schemas import BreakerKPI
from ..utils import stable_hash

PYTHON_ROOT = Path(__file__).resolve().parents[2]


def _pyexec_env() -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": "",
        "PYTHONHOME": "",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
    }


def _pyfunc_command(program_path: Path, entrypoint: str) -> List[str]:
    script = (
        "import importlib, sys;"
        f"sys.path.insert(0, {str(PYTHON_ROOT)!r});"
        "import sovereidolon_v1.pyfunc.runner as runner;"
        "runner.main()"
    )
    return [
        sys.executable,
        "-I",
        "-c",
        script,
        "--program",
        str(program_path),
        "--entrypoint",
        entrypoint,
    ]


def _run_pyfunc_runner(
    program_path: Path, entrypoint: str, payload: Dict[str, Any], timeout: float = 1.0
) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as workdir:
        proc = subprocess.run(
            _pyfunc_command(program_path, entrypoint),
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=timeout,
            env=_pyexec_env(),
            cwd=workdir,
        )
    if proc.returncode != 0:
        raise RuntimeError("RESOURCE_LIMIT")
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("BAD_OUTPUT") from exc
    if not isinstance(data, dict):
        return {}
    return data


def _unique(values: Iterable[Any]) -> List[Any]:
    ordered: List[Any] = []
    seen = set()
    for value in values:
        key = json.dumps(value, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def _int_bounds(task: Task, name: str) -> tuple[int, int]:
    bounds = task.bounds.get(name)
    if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
        low = int(min(bounds[0], bounds[1]))
        high = int(max(bounds[0], bounds[1]))
        return low, high
    return -5, 5


def _boundary_values(task: Task, name: str, typ: str) -> List[Any]:
    if typ == "Int":
        low, high = _int_bounds(task, name)
        return _unique([low, high, 0, 1])
    if typ == "Bool":
        return [False, True]
    return ["", "a"]


def _candidate_cartesian(task: Task) -> List[Dict[str, Any]]:
    keys = sorted(task.inputs.keys())
    options: list[list[Any]] = []
    for name in keys:
        typ = task.inputs[name]
        options.append(_boundary_values(task, name, typ))
    return [dict(zip(keys, combo, strict=True)) for combo in product(*options)]


def _mutate_candidates(
    bases: Iterable[Dict[str, Any]], task: Task, commutative: bool
) -> List[Dict[str, Any]]:
    mutations: List[Dict[str, Any]] = []
    keys = sorted(task.inputs.keys())
    for base in bases:
        for key, value in base.items():
            if isinstance(value, int):
                low, high = _int_bounds(task, key)
                for delta in (-1, 1):
                    candidate = dict(base)
                    mutated = value + delta
                    if mutated < low:
                        mutated = low
                    if mutated > high:
                        mutated = high
                    candidate[key] = mutated
                    mutations.append(candidate)
            elif isinstance(value, str):
                candidate = dict(base)
                candidate[key] = value[: max(0, len(value) // 2)]
                mutations.append(candidate)
        if commutative and len(keys) >= 2:
            swapped = dict(base)
            first, second = keys[0], keys[1]
            swapped[first], swapped[second] = base.get(second), base.get(first)
            mutations.append(swapped)
    uniques: List[Dict[str, Any]] = []
    seen = set()
    for candidate in mutations:
        fingerprint = stable_hash(candidate)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        uniques.append(candidate)
    return uniques


def _random_candidate(task: Task, rng: random.Random) -> Dict[str, Any]:
    candidate: Dict[str, Any] = {}
    for name in sorted(task.inputs.keys()):
        typ = task.inputs[name]
        if typ == "Int":
            low, high = _int_bounds(task, name)
            candidate[name] = rng.randint(low, high)
        elif typ == "Bool":
            candidate[name] = rng.choice([False, True])
        else:
            length = rng.randint(0, 3)
            letters = [rng.choice(["a", "b", "c"]) for _ in range(length)]
            candidate[name] = "".join(letters)
    return candidate


@dataclass
class BreakerBudget:
    attempt_budget: int


@dataclass
class BreakerResult:
    counterexample: Optional[Example]
    minimized: Optional[Example]
    kpi: BreakerKPI
    report: Dict[str, Any]


def _seed_for(task: Task, program_hash: str) -> int:
    signature = {"spec": task.spec_signature(), "program_hash": program_hash}
    return int(stable_hash(signature)[:8], 16)


def _evaluate_candidate(
    candidate: Dict[str, Any],
    spec: TaskSpec,
    program_path: Path,
    entrypoint: str,
) -> tuple[bool, Any, Any, List[str]]:
    expected = spec.evaluate(candidate)
    payload = {"tests": [{"inputs": candidate, "output": expected}], "metamorphic": []}
    try:
        result = _run_pyfunc_runner(program_path, entrypoint, payload)
    except subprocess.TimeoutExpired:
        return True, expected, "TIMEOUT", ["TIMEOUT"]
    except RuntimeError as exc:
        atom = str(exc)
        return True, expected, atom, [atom]
    failure_atoms = list(result.get("failure_atoms", []))
    verdict = result.get("verdict", "FAIL")
    outputs = result.get("outputs", [])
    got = outputs[-1] if outputs else None
    if verdict != "PASS":
        if got is None and failure_atoms:
            got = failure_atoms[0]
        return True, expected, got, failure_atoms
    if got != expected:
        return True, expected, got, failure_atoms
    return False, expected, got, failure_atoms


def _minimize_counterexample(
    task: Task,
    spec: TaskSpec,
    program_path: Path,
    entrypoint: str,
    counterexample: Dict[str, Any],
    step_limit: int = 50,
) -> tuple[Dict[str, Any], int]:
    steps = 0
    inputs = dict(counterexample.get("inputs", {}))
    expected = counterexample.get("expected")
    got = counterexample.get("got")

    for key in sorted(inputs.keys()):
        if steps >= step_limit:
            break
        value = inputs[key]
        if isinstance(value, int):
            low, high = _int_bounds(task, key)
            candidates = _unique([0, 1, -1, low, high])
            for cand in candidates:
                if steps >= step_limit:
                    break
                candidate_inputs = dict(inputs)
                candidate_inputs[key] = cand
                steps += 1
                is_cex, cand_expected, cand_got, _ = _evaluate_candidate(
                    candidate_inputs, spec, program_path, entrypoint
                )
                if is_cex:
                    inputs = candidate_inputs
                    expected = cand_expected
                    got = cand_got
                    value = cand
                    break
            while steps < step_limit and value != 0:
                candidate_value = value - 1 if value > 0 else value + 1
                candidate_inputs = dict(inputs)
                candidate_inputs[key] = candidate_value
                steps += 1
                is_cex, cand_expected, cand_got, _ = _evaluate_candidate(
                    candidate_inputs, spec, program_path, entrypoint
                )
                if is_cex:
                    inputs = candidate_inputs
                    expected = cand_expected
                    got = cand_got
                    value = candidate_value
                else:
                    break
        elif isinstance(value, str):
            while steps < step_limit and len(value) > 0:
                candidate_value = value[: len(value) // 2]
                candidate_inputs = dict(inputs)
                candidate_inputs[key] = candidate_value
                steps += 1
                is_cex, cand_expected, cand_got, _ = _evaluate_candidate(
                    candidate_inputs, spec, program_path, entrypoint
                )
                if is_cex:
                    inputs = candidate_inputs
                    expected = cand_expected
                    got = cand_got
                    value = candidate_value
                else:
                    break

    minimized = {"inputs": inputs, "expected": expected, "got": got}
    return minimized, steps


def run_pyfunc_breaker(
    task: Task, program_path: Path, budget: BreakerBudget
) -> BreakerResult:
    if not program_path.exists():
        kpi = BreakerKPI(
            CDR=0.0,
            TMR=0.0,
            NOVN=0.0,
            WFHR=0.0,
            window={"attempts": 0},
            budget={"attempt_budget": budget.attempt_budget},
        )
        return BreakerResult(
            counterexample=None,
            minimized=None,
            kpi=kpi,
            report={"counterexamples": [], "attempts": 0, "skipped": True},
        )

    code = program_path.read_text()
    program_hash = compute_pyfunc_hash(code)
    seed_value = _seed_for(task, program_hash)
    rng = random.Random(seed_value)
    spec = task_spec(task)
    entrypoint = task.metadata.get("pyfunc", {}).get("entrypoint", "solve")
    commutative = "commutative" in task.metadata.get("pyfunc", {}).get("metamorphic", [])
    attempts = 0
    counterexamples: List[Dict[str, Any]] = []
    found_failure_atoms: List[str] = []
    seen = set()
    max_attempts = max(0, budget.attempt_budget)

    def _try_candidate(candidate: Dict[str, Any]) -> bool:
        nonlocal attempts
        fingerprint = stable_hash(candidate)
        if fingerprint in seen:
            return False
        seen.add(fingerprint)
        if attempts >= max_attempts:
            return True
        attempts += 1
        is_cex, expected, got, failure_atoms = _evaluate_candidate(
            candidate, spec, program_path, entrypoint
        )
        if is_cex:
            counterexamples.append({"inputs": candidate, "expected": expected, "got": got})
            found_failure_atoms.extend(failure_atoms)
            return True
        return False

    boundary_candidates = _candidate_cartesian(task)
    for candidate in boundary_candidates:
        if _try_candidate(candidate):
            break

    if not counterexamples and attempts < max_attempts:
        base_candidates = list(task.examples)
        base_inputs = [example.inputs for example in base_candidates]
        mutation_candidates = _mutate_candidates(
            boundary_candidates + base_inputs, task, commutative
        )
        for candidate in mutation_candidates:
            if _try_candidate(candidate):
                break

    while not counterexamples and attempts < max_attempts:
        candidate = _random_candidate(task, rng)
        if _try_candidate(candidate):
            break

    tmr = float(attempts if counterexamples else budget.attempt_budget)
    kpi = BreakerKPI(
        CDR=(1.0 / max(1, attempts)) if counterexamples else 0.0,
        TMR=tmr,
        NOVN=1.0 if counterexamples else 0.0,
        WFHR=0.0,
        window={"attempts": attempts},
        budget={"attempt_budget": budget.attempt_budget},
    )
    minimized: Optional[Dict[str, Any]] = None
    minimized_steps = 0
    if counterexamples:
        minimized, minimized_steps = _minimize_counterexample(
            task,
            spec,
            program_path,
            entrypoint,
            counterexamples[0],
        )
    report = {
        "counterexamples": counterexamples,
        "minimized": minimized,
        "failure_atoms": found_failure_atoms,
        "attempts": attempts,
        "program_hash": program_hash,
        "minimized_steps": minimized_steps,
    }
    return BreakerResult(
        counterexample=Example(
            inputs=counterexamples[0]["inputs"], output=counterexamples[0]["expected"]
        )
        if counterexamples
        else None,
        minimized=None,
        kpi=kpi,
        report=report,
    )
