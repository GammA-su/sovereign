from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from ..jsonspec.program import JsonSpecProgram, compute_jsonspec_hash
from ..jsonspec.runner import JsonSpecRuntimeError, run_jsonspec_program
from ..jsonspec.validator import JsonSpecValidationError, validate_jsonspec
from ..orchestrator.task import Example, Task
from ..schemas import BreakerKPI
from ..utils import ensure_dir, stable_hash, write_json
from .pyfunc_breaker import BreakerBudget, BreakerResult


def _seed_for(task: Task, program_hash: str) -> int:
    signature = {"task_id": task.task_id, "program_hash": program_hash}
    return int(stable_hash(signature)[:8], 16)


def _extract_root(inputs: Dict[str, Any]) -> Any:
    if len(inputs) == 1:
        return next(iter(inputs.values()))
    return inputs


def _wrap_inputs(task: Task, root: Any) -> Dict[str, Any]:
    if len(task.inputs) == 1:
        key = next(iter(task.inputs))
        return {key: root}
    if isinstance(root, dict):
        return root
    return {"input": root}


def _parse_root(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def _iter_example_roots(task: Task) -> Iterable[Dict[str, Any]]:
    for example in task.examples:
        root = _parse_root(_extract_root(example.inputs))
        if isinstance(root, dict):
            yield root


def _candidate_pool(task: Task, seed_value: int) -> List[Dict[str, Any]]:
    meta = task.metadata.get("jsonspec", {}).get("breaker", {})
    pool_size = int(meta.get("pool_size", 6))
    max_items = int(meta.get("max_items", 2))
    keys = meta.get("keys")
    if not isinstance(keys, list) or not keys:
        key_set: List[str] = []
        for root in _iter_example_roots(task):
            key_set.extend(list(root.keys()))
        keys = sorted({key for key in key_set if isinstance(key, str)}) or ["a", "b"]
    values = meta.get("values")
    if not isinstance(values, list) or not values:
        values = [0, 1, "a", "", True, False, None]
    rng = random.Random(seed_value)
    pool: List[Dict[str, Any]] = []
    for root in _iter_example_roots(task):
        pool.append(dict(root))
    while len(pool) < max(1, pool_size):
        size = rng.randint(0, min(max_items, len(keys)))
        chosen = rng.sample(list(keys), k=size) if keys else []
        candidate: Dict[str, Any] = {}
        for key in chosen:
            candidate[str(key)] = rng.choice(values)
        pool.append(candidate)
    if not pool:
        pool.append({})
    return pool


def _evaluate_candidate(
    task: Task,
    program: JsonSpecProgram,
    oracle: JsonSpecProgram,
    candidate_inputs: Dict[str, Any],
) -> Tuple[bool, Any, Any, List[str]]:
    try:
        expected = run_jsonspec_program(oracle.spec, candidate_inputs)
    except JsonSpecRuntimeError:
        return False, None, None, []
    try:
        got = run_jsonspec_program(program.spec, candidate_inputs)
    except JsonSpecRuntimeError as exc:
        return True, expected, exc.failure_atom, [exc.failure_atom]
    if got != expected:
        return True, expected, got, ["JSONSPEC_OUTPUT_MISMATCH"]
    return False, expected, got, []


def _simplify_values(value: Any) -> List[Any]:
    if isinstance(value, bool):
        return [False]
    if isinstance(value, int):
        return [0, 1, -1]
    if isinstance(value, str):
        return ["", value[: max(0, len(value) // 2)]]
    if isinstance(value, list):
        return [[], value[: max(0, len(value) // 2)]]
    if isinstance(value, dict):
        return [{}]
    return [None]


def _minimize_counterexample(
    task: Task,
    program: JsonSpecProgram,
    oracle: JsonSpecProgram,
    counterexample: Dict[str, Any],
    step_limit: int = 50,
) -> Tuple[Dict[str, Any], int]:
    steps = 0
    inputs = dict(counterexample.get("inputs", {}))
    root = _parse_root(_extract_root(inputs))
    if not isinstance(root, dict):
        return counterexample, steps

    def _is_cex(candidate_root: Dict[str, Any]) -> bool:
        candidate_inputs = _wrap_inputs(task, candidate_root)
        is_cex, _, _, _ = _evaluate_candidate(task, program, oracle, candidate_inputs)
        return is_cex

    keys = sorted(root.keys())
    for key in keys:
        if steps >= step_limit:
            break
        candidate_root = dict(root)
        candidate_root.pop(key, None)
        steps += 1
        if _is_cex(candidate_root):
            root = candidate_root

    for key in sorted(root.keys()):
        if steps >= step_limit:
            break
        value = root[key]
        for simplified in _simplify_values(value):
            if steps >= step_limit:
                break
            candidate_root = dict(root)
            candidate_root[key] = simplified
            steps += 1
            if _is_cex(candidate_root):
                root = candidate_root
                value = simplified
                break

    minimized_inputs = _wrap_inputs(task, root)
    is_cex, expected, got, _ = _evaluate_candidate(task, program, oracle, minimized_inputs)
    if not is_cex:
        return counterexample, steps
    return {"inputs": minimized_inputs, "expected": expected, "got": got}, steps


def run_jsonspec_breaker(
    task: Task,
    program: JsonSpecProgram,
    budget: BreakerBudget,
    run_dir: Path,
) -> BreakerResult:
    try:
        validate_jsonspec(program.spec)
    except JsonSpecValidationError:
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

    oracle_spec = task.metadata.get("jsonspec", {}).get("spec_program")
    if oracle_spec is None:
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
    oracle = JsonSpecProgram(oracle_spec)
    try:
        validate_jsonspec(oracle.spec)
    except JsonSpecValidationError:
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

    program_hash = compute_jsonspec_hash(program.spec)
    seed_value = _seed_for(task, program_hash)
    attempts = 0
    duplicates = 0
    counterexamples: List[Dict[str, Any]] = []
    failure_atoms: List[str] = []
    seen = set()
    max_attempts = max(0, budget.attempt_budget)
    pool = _candidate_pool(task, seed_value)
    pool_len = max(1, len(pool))

    def _try_candidate(root: Dict[str, Any]) -> bool:
        nonlocal attempts, duplicates
        if attempts >= max_attempts:
            return True
        candidate_inputs = _wrap_inputs(task, root)
        fingerprint = stable_hash(candidate_inputs)
        attempts += 1
        if fingerprint in seen:
            duplicates += 1
            return attempts >= max_attempts
        seen.add(fingerprint)
        is_cex, expected, got, found_atoms = _evaluate_candidate(
            task, program, oracle, candidate_inputs
        )
        if is_cex:
            counterexamples.append(
                {"inputs": candidate_inputs, "expected": expected, "got": got}
            )
            for atom in found_atoms:
                if atom not in failure_atoms:
                    failure_atoms.append(atom)
            return True
        return False

    idx = 0
    while attempts < max_attempts and not counterexamples:
        candidate_root = pool[idx % pool_len]
        if _try_candidate(candidate_root):
            break
        idx += 1

    minimized = None
    minimized_steps = 0
    minimized_path = ""
    if counterexamples:
        minimized, minimized_steps = _minimize_counterexample(
            task, program, oracle, counterexamples[0]
        )
        if minimized:
            ensure_dir(run_dir / "artifacts" / "breaker")
            minimized_hash = stable_hash(minimized)
            minimized_path = str(
                run_dir / "artifacts" / "breaker" / f"jsonspec_min_{minimized_hash}.json"
            )
            write_json(Path(minimized_path), minimized)
            if "COUNTEREXAMPLE_MINIMIZED" not in failure_atoms:
                failure_atoms.append("COUNTEREXAMPLE_MINIMIZED")

    tmr = float(attempts if counterexamples else budget.attempt_budget)
    kpi = BreakerKPI(
        CDR=(1.0 / max(1, attempts)) if counterexamples else 0.0,
        TMR=tmr,
        NOVN=1.0 if counterexamples else 0.0,
        WFHR=0.0,
        window={"attempts": attempts},
        budget={"attempt_budget": budget.attempt_budget},
    )
    report = {
        "counterexample": counterexamples[0] if counterexamples else None,
        "counterexamples": counterexamples,
        "minimized": minimized,
        "minimized_path": minimized_path,
        "failure_atoms": failure_atoms,
        "attempts": attempts,
        "duplicate_candidates": duplicates,
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
