from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal

from hypothesis import given
from hypothesis import seed as hypo_seed
from hypothesis import settings as hypo_settings
from hypothesis import strategies as st

from ..bvps.interpreter import eval_program
from ..config import Settings
from ..ledger.ledger import Ledger
from ..orchestrator.specs import TaskSpec
from ..orchestrator.task import Example, Task
from ..schemas import VerifierVerdict


@dataclass
class VerifierContext:
    task: Task
    program: Any
    tests: List[Example]
    trace_hashes: List[str]
    run_dir: str
    settings: Settings
    spec: TaskSpec


def _hypothesis_inputs(task: Task, count: int, seed_value: int) -> List[Dict[str, Any]]:
    if task.task_type == "arith":
        x_min, x_max = task.bounds.get("x", [-5, 5])
        y_min, y_max = task.bounds.get("y", [-5, 5])
        strat = st.tuples(
            st.integers(min_value=int(x_min), max_value=int(x_max)),
            st.integers(min_value=int(y_min), max_value=int(y_max)),
        )
        inputs: List[Dict[str, Any]] = []

        @hypo_settings(derandomize=True, max_examples=count)
        @hypo_seed(seed_value)
        @given(pair=strat)
        def _collect(pair: tuple[int, int]) -> None:
            x_val, y_val = pair
            inputs.append({"x": x_val, "y": y_val})

        _collect()
        return inputs
    if task.task_type == "list":
        len_min, len_max = task.bounds.get("xs_len", [0, 5])
        elem_min, elem_max = task.bounds.get("xs_elem", [-5, 5])
        list_strat = st.lists(
            st.integers(min_value=int(elem_min), max_value=int(elem_max)),
            min_size=int(len_min),
            max_size=int(len_max),
        )
        inputs = []

        @hypo_settings(derandomize=True, max_examples=count)
        @hypo_seed(seed_value)
        @given(xs=list_strat)
        def _collect(xs: List[int]) -> None:
            inputs.append({"xs": xs})

        _collect()
        return inputs
    return []


def recompute_lane(ctx: VerifierContext) -> VerifierVerdict:
    start = time.time_ns()
    failure_atoms: List[str] = []
    for example in ctx.tests:
        value1, trace1 = eval_program(ctx.program, example.inputs, ctx.settings.verify_budget_steps)
        value2, trace2 = eval_program(ctx.program, example.inputs, ctx.settings.verify_budget_steps)
        if value1 != example.output:
            failure_atoms.append("recompute_output_mismatch")
            break
        if trace1 != trace2:
            failure_atoms.append("trace_nondeterministic")
            break
    verdict: Literal["PASS", "FAIL"] = "PASS" if not failure_atoms else "FAIL"
    cost = {"ns": time.time_ns() - start, "tests": len(ctx.tests)}
    return VerifierVerdict(
        verdict=verdict,
        failure_atoms=failure_atoms,
        domain="bvps",
        tier="recompute",
        bounds=ctx.task.bounds,
        soundness_grade="CERT",
        metamorphic_families=[],
        cost=cost,
    )


def consequence_lane(ctx: VerifierContext) -> VerifierVerdict:
    start = time.time_ns()
    failure_atoms: List[str] = []
    inputs = _hypothesis_inputs(ctx.task, count=10, seed_value=ctx.settings.open_seed)
    for inp in inputs:
        expected = ctx.spec.evaluate(inp)
        actual, _ = eval_program(ctx.program, inp, ctx.settings.verify_budget_steps)
        if expected != actual:
            failure_atoms.append("property_violation")
            break
    verdict: Literal["PASS", "FAIL"] = "PASS" if not failure_atoms else "FAIL"
    cost = {"ns": time.time_ns() - start, "samples": len(inputs)}
    return VerifierVerdict(
        verdict=verdict,
        failure_atoms=failure_atoms,
        domain="bvps",
        tier="consequence",
        bounds=ctx.task.bounds,
        soundness_grade="BOUNDED",
        metamorphic_families=[],
        cost=cost,
    )


def _eval_expr_py(expr: Dict[str, Any], env: Dict[str, Any]) -> Any:
    kind = expr["kind"]
    if kind == "int":
        return int(expr["value"])
    if kind == "bool":
        return bool(expr["value"])
    if kind == "var":
        return env[expr["name"]]
    if kind == "binop":
        left = _eval_expr_py(expr["left"], env)
        right = _eval_expr_py(expr["right"], env)
        op = expr["op"]
        if op == "+":
            return int(left) + int(right)
        if op == "-":
            return int(left) - int(right)
        if op == "*":
            return int(left) * int(right)
        if op == "==":
            return left == right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        if op == "and":
            return bool(left) and bool(right)
        if op == "or":
            return bool(left) or bool(right)
        raise ValueError("unknown op")
    if kind == "if":
        cond = _eval_expr_py(expr["cond"], env)
        branch = expr["then"] if cond else expr["else"]
        return _eval_expr_py(branch, env)
    if kind == "let":
        value = _eval_expr_py(expr["value"], env)
        new_env = dict(env)
        new_env[expr["name"]] = value
        return _eval_expr_py(expr["body"], new_env)
    if kind == "list":
        return [_eval_expr_py(item, env) for item in expr["elements"]]
    if kind == "tuple2":
        return (
            _eval_expr_py(expr["left"], env),
            _eval_expr_py(expr["right"], env),
        )
    if kind == "map":
        items = _eval_expr_py(expr["list"], env)
        return [_eval_expr_py(expr["body"], {**env, expr["var"]: item}) for item in items]
    if kind == "fold":
        items = _eval_expr_py(expr["list"], env)
        acc = _eval_expr_py(expr["init"], env)
        for item in items:
            acc = _eval_expr_py(
                expr["body"], {**env, expr["var"]: item, expr["acc"]: acc}
            )
        return acc
    raise ValueError("unknown expr kind")


def translation_lane(ctx: VerifierContext) -> VerifierVerdict:
    start = time.time_ns()
    failure_atoms: List[str] = []
    inputs = _hypothesis_inputs(ctx.task, count=10, seed_value=ctx.settings.open_seed + 1)
    for inp in inputs:
        expected, _ = eval_program(ctx.program, inp, ctx.settings.verify_budget_steps)
        actual = _eval_expr_py(ctx.program.body, inp)
        if expected != actual:
            failure_atoms.append("translation_mismatch")
            break
    verdict: Literal["PASS", "FAIL"] = "PASS" if not failure_atoms else "FAIL"
    cost = {"ns": time.time_ns() - start, "samples": len(inputs)}
    return VerifierVerdict(
        verdict=verdict,
        failure_atoms=failure_atoms,
        domain="bvps",
        tier="translation",
        bounds=ctx.task.bounds,
        soundness_grade="BOUNDED",
        metamorphic_families=[],
        cost=cost,
    )


def anchor_lane(ctx: VerifierContext) -> VerifierVerdict:
    start = time.time_ns()
    failure_atoms: List[str] = []
    ledger_path = Path(ctx.run_dir) / "ledger.jsonl"
    ok, _ = Ledger.verify_chain(ledger_path)
    if not ok:
        failure_atoms.append("ledger_tamper")
    if ctx.settings.verify_budget_steps <= 0:
        failure_atoms.append("invalid_verify_budget")
    verdict: Literal["PASS", "FAIL"] = "PASS" if not failure_atoms else "FAIL"
    cost = {"ns": time.time_ns() - start}
    return VerifierVerdict(
        verdict=verdict,
        failure_atoms=failure_atoms,
        domain="system",
        tier="anchor",
        bounds={},
        soundness_grade="CERT",
        metamorphic_families=[],
        cost=cost,
    )


def _mutations_for_transfer(inp: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "xs" in inp:
        xs = list(inp["xs"])
        return [
            {"xs": list(reversed(xs))},
            {"xs": xs + xs},
        ]
    if "x" in inp and "y" in inp:
        return [
            {"x": inp["x"] + 1, "y": inp["y"]},
            {"x": inp["x"], "y": inp["y"] - 1},
        ]
    return []


def transfer_lane(ctx: VerifierContext) -> VerifierVerdict:
    start = time.time_ns()
    failure_atoms: List[str] = []
    checked = 0
    for example in ctx.tests:
        for mutated in _mutations_for_transfer(example.inputs):
            expected = ctx.spec.evaluate(mutated)
            actual, _ = eval_program(ctx.program, mutated, ctx.settings.verify_budget_steps)
            checked += 1
            if expected != actual:
                failure_atoms.append("transfer_violation")
                break
        if failure_atoms:
            break
    verdict: Literal["PASS", "FAIL"] = "PASS" if not failure_atoms else "FAIL"
    cost = {"ns": time.time_ns() - start, "samples": checked}
    return VerifierVerdict(
        verdict=verdict,
        failure_atoms=failure_atoms,
        domain="bvps",
        tier="transfer",
        bounds=ctx.task.bounds,
        soundness_grade="HEURISTIC",
        metamorphic_families=["reverse", "duplicate"],
        cost=cost,
    )
