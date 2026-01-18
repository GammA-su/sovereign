from __future__ import annotations

import difflib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Literal

from hypothesis import given
from hypothesis import seed as hypo_seed
from hypothesis import settings as hypo_settings
from hypothesis import strategies as st

from ..bvps.interpreter import eval_program
from ..codepatch.applier import apply_patch, parse_unified_diff
from ..codepatch.runner import default_test_command, run_codepatch, run_tests_in_dir
from ..codepatch.validator import extract_patch_paths, validate_patch
from ..config import Settings
from ..jsonspec.program import JsonSpecProgram
from ..jsonspec.runner import JsonSpecRuntimeError, run_jsonspec_program
from ..jsonspec.validator import JsonSpecValidationError, validate_jsonspec
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
    if task.task_type == "bool":
        return _bool_inputs(task)
    return []


def _bool_inputs(task: Task) -> List[Dict[str, Any]]:
    names = list(task.inputs.keys())
    combos = product([False, True], repeat=len(names))
    inputs: List[Dict[str, Any]] = []
    for combo in combos:
        inputs.append({name: bool(value) for name, value in zip(names, combo, strict=True)})
    return inputs


def recompute_lane(ctx: VerifierContext) -> VerifierVerdict:
    start = time.time_ns()
    failure_atoms: List[str] = []
    if ctx.task.has_contradictory_examples():
        failure_atoms.append("contradictory_examples")
        cost = {"ns": time.time_ns() - start, "tests": 0}
        return VerifierVerdict(
            verdict="FAIL",
            failure_atoms=failure_atoms,
            domain="bvps",
            tier="recompute",
            bounds=ctx.task.bounds,
            soundness_grade="CERT",
            metamorphic_families=[],
            cost=cost,
        )
    test_count = len(ctx.tests)
    if ctx.task.task_type == "bool":
        inputs = _bool_inputs(ctx.task)
        test_count = len(inputs)
        for inp in inputs:
            expected = ctx.spec.evaluate(inp)
            value1, trace1 = eval_program(ctx.program, inp, ctx.settings.verify_budget_steps)
            value2, trace2 = eval_program(ctx.program, inp, ctx.settings.verify_budget_steps)
            if value1 != expected:
                failure_atoms.append("recompute_output_mismatch")
                break
            if trace1 != trace2:
                failure_atoms.append("trace_nondeterministic")
                break
    else:
        for example in ctx.tests:
            value1, trace1 = eval_program(
                ctx.program, example.inputs, ctx.settings.verify_budget_steps
            )
            value2, trace2 = eval_program(
                ctx.program, example.inputs, ctx.settings.verify_budget_steps
            )
            if value1 != example.output:
                failure_atoms.append("recompute_output_mismatch")
                break
            if trace1 != trace2:
                failure_atoms.append("trace_nondeterministic")
                break
    verdict: Literal["PASS", "FAIL"] = "PASS" if not failure_atoms else "FAIL"
    cost = {"ns": time.time_ns() - start, "tests": test_count}
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


PYTHON_ROOT = Path(__file__).resolve().parents[2]
_RUNNER_METAMORPHIC = {"commutative", "idempotent"}
_DERIVED_METAMORPHIC = {"reverse_args", "duplicate_inputs", "permute_examples"}
_CODEPATCH_METAMORPHIC = {"whitespace_idempotent", "apply_revert_apply", "commutation_safe"}


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


def _run_pyexec_tests(
    program_path: Path,
    entrypoint: str,
    tests: List[Dict[str, Any]],
    metamorphic: List[str],
) -> tuple[str, List[str], List[str]]:
    failure_atoms: List[str] = []
    metamorphic_families: List[str] = []
    verdict = "FAIL"
    if not program_path.exists():
        failure_atoms.append("EXCEPTION:MISSING_PROGRAM")
        return verdict, failure_atoms, metamorphic_families
    payload = {"tests": tests, "metamorphic": metamorphic}
    script = (
        "import importlib, sys; "
        f"sys.path.insert(0, {str(PYTHON_ROOT)!r}); "
        "import sovereidolon_v1.pyfunc.runner as runner; "
        "runner.main()"
    )
    try:
        with tempfile.TemporaryDirectory() as workdir:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    script,
                    "--program",
                    str(program_path),
                    "--entrypoint",
                    entrypoint,
                ],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=1.0,
                env=_pyexec_env(),
                cwd=workdir,
            )
    except subprocess.TimeoutExpired:
        failure_atoms.append("TIMEOUT")
        return verdict, failure_atoms, metamorphic_families
    if proc.returncode != 0:
        if proc.returncode < 0:
            signal_num = -proc.returncode
            if signal_num in {signal.SIGXCPU, signal.SIGALRM, signal.SIGKILL}:
                failure_atoms.append("TIMEOUT")
            else:
                failure_atoms.append("RESOURCE_LIMIT")
        else:
            failure_atoms.append("EXCEPTION:RUNNER_ERROR")
        return verdict, failure_atoms, metamorphic_families
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        failure_atoms.append("EXCEPTION:BAD_OUTPUT")
        return verdict, failure_atoms, metamorphic_families
    verdict = result.get("verdict", "FAIL")
    failure_atoms.extend(result.get("failure_atoms", []))
    metamorphic_families = result.get("metamorphic_families", [])
    return verdict, failure_atoms, metamorphic_families


def pyexec_lane(ctx: VerifierContext) -> VerifierVerdict:
    start = time.time_ns()
    program_path = Path(ctx.run_dir).resolve() / "artifacts" / "pyfunc" / "program.py"
    entrypoint = ctx.task.metadata.get("pyfunc", {}).get("entrypoint", "solve")
    configured = ctx.task.metadata.get("pyfunc", {}).get("metamorphic", [])
    runner_metamorphic = [name for name in configured if name in _RUNNER_METAMORPHIC]
    verdict, failure_atoms, metamorphic_families = _run_pyexec_tests(
        program_path,
        entrypoint,
        [example.model_dump() for example in ctx.tests],
        runner_metamorphic,
    )
    if not failure_atoms and verdict == "PASS":
        final_verdict: Literal["PASS", "FAIL"] = "PASS"
    else:
        final_verdict = "FAIL"
    cost = {"ns": time.time_ns() - start, "tests": len(ctx.tests) + len(metamorphic_families)}
    return VerifierVerdict(
        verdict=final_verdict,
        failure_atoms=failure_atoms,
        domain="pyfunc",
        tier="pyexec",
        bounds=ctx.task.bounds,
        soundness_grade="BOUNDED",
        metamorphic_families=metamorphic_families,
        cost=cost,
    )


def _pyfunc_metamorphic_tests(
    ctx: VerifierContext, family: str
) -> List[Dict[str, Any]]:
    derived: List[Dict[str, Any]] = []
    if family == "reverse_args":
        if "a" not in ctx.task.inputs or "b" not in ctx.task.inputs:
            return []
        for example in ctx.tests:
            swapped = dict(example.inputs)
            swapped["a"], swapped["b"] = swapped.get("b"), swapped.get("a")
            derived.append({"inputs": swapped, "output": ctx.spec.evaluate(swapped)})
        return derived
    if family == "duplicate_inputs":
        for example in ctx.tests:
            inputs = dict(example.inputs)
            expected = ctx.spec.evaluate(inputs)
            derived.append({"inputs": inputs, "output": expected})
            derived.append({"inputs": inputs, "output": expected})
        return derived
    if family == "permute_examples":
        for example in reversed(ctx.tests):
            inputs = dict(example.inputs)
            derived.append({"inputs": inputs, "output": ctx.spec.evaluate(inputs)})
        return derived
    return []


def pyexec_metamorphic_lane(ctx: VerifierContext) -> VerifierVerdict:
    start = time.time_ns()
    failure_atoms: List[str] = []
    metamorphic_families: List[str] = []
    tests_run = 0
    configured = ctx.task.metadata.get("pyfunc", {}).get("metamorphic", [])
    selected = [name for name in configured if name in _DERIVED_METAMORPHIC]
    if selected:
        program_path = Path(ctx.run_dir).resolve() / "artifacts" / "pyfunc" / "program.py"
        entrypoint = ctx.task.metadata.get("pyfunc", {}).get("entrypoint", "solve")
        for family in selected:
            derived_tests = _pyfunc_metamorphic_tests(ctx, family)
            if not derived_tests:
                continue
            metamorphic_families.append(family)
            tests_run += len(derived_tests)
            verdict, failure_detail, _ = _run_pyexec_tests(
                program_path,
                entrypoint,
                derived_tests,
                [],
            )
            if verdict != "PASS" or failure_detail:
                failure_atoms.append(f"METAMORPHIC_VIOLATION:{family}")
    final_verdict: Literal["PASS", "FAIL"] = "PASS" if not failure_atoms else "FAIL"
    cost = {"ns": time.time_ns() - start, "tests": tests_run}
    return VerifierVerdict(
        verdict=final_verdict,
        failure_atoms=failure_atoms,
        domain="pyfunc",
        tier="metamorphic",
        bounds=ctx.task.bounds,
        soundness_grade="BOUNDED",
        metamorphic_families=metamorphic_families,
        cost=cost,
    )


def jsonspec_lane(ctx: VerifierContext) -> VerifierVerdict:
    start = time.time_ns()
    failure_atoms: List[str] = []
    tests_run = 0
    program = ctx.program if isinstance(ctx.program, JsonSpecProgram) else None
    if program is None:
        candidate = ctx.task.metadata.get("jsonspec", {}).get("candidate_program")
        if candidate:
            program = JsonSpecProgram(candidate)
    if program is None:
        failure_atoms.append("JSONSPEC_MISSING_PROGRAM")
    else:
        try:
            validate_jsonspec(program.spec)
            for example in ctx.tests:
                try:
                    actual = run_jsonspec_program(program.spec, example.inputs)
                except JsonSpecRuntimeError as exc:
                    failure_atoms.append(exc.failure_atom)
                    break
                tests_run += 1
                if actual != example.output:
                    failure_atoms.append("JSONSPEC_OUTPUT_MISMATCH")
                    break
        except JsonSpecValidationError as exc:
            failure_atoms.append(exc.failure_atom)
    verdict: Literal["PASS", "FAIL"] = "PASS" if not failure_atoms else "FAIL"
    cost = {"ns": time.time_ns() - start, "tests": tests_run}
    return VerifierVerdict(
        verdict=verdict,
        failure_atoms=failure_atoms,
        domain="jsonspec",
        tier="jsonspec",
        bounds=ctx.task.bounds,
        soundness_grade="BOUNDED",
        metamorphic_families=[],
        cost=cost,
    )


def _jsonspec_permute(value: Any) -> Any:
    if isinstance(value, dict):
        keys = list(value.keys())
        if len(keys) < 2:
            return value
        permuted = list(reversed(keys))
        return {key: _jsonspec_permute(value[key]) for key in permuted}
    if isinstance(value, list):
        return [_jsonspec_permute(item) for item in value]
    return value


def _jsonspec_json_string(value: Any) -> str:
    return json.dumps(value, indent=2)


def _jsonspec_derive_input(inputs: Dict[str, Any], family: str) -> Dict[str, Any] | None:
    if len(inputs) != 1:
        return None
    key = next(iter(inputs))
    raw = inputs[key]
    try:
        if isinstance(raw, str):
            parsed = json.loads(raw)
        else:
            parsed = raw
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    if family == "key_order_invariance":
        original_keys = list(parsed.keys())
        derived = _jsonspec_permute(parsed)
        if not isinstance(derived, dict):
            return None
        if list(derived.keys()) == original_keys:
            return None
        return {key: derived}
    if family == "whitespace_invariance":
        return {key: _jsonspec_json_string(parsed)}
    return None


def jsonspec_metamorphic_lane(ctx: VerifierContext) -> VerifierVerdict:
    start = time.time_ns()
    failure_atoms: List[str] = []
    metamorphic_families: List[str] = []
    tests_run = 0
    program = ctx.program if isinstance(ctx.program, JsonSpecProgram) else None
    if program is None:
        candidate = ctx.task.metadata.get("jsonspec", {}).get("candidate_program")
        if candidate:
            program = JsonSpecProgram(candidate)
    if program is None:
        cost = {"ns": time.time_ns() - start, "tests": tests_run}
        return VerifierVerdict(
            verdict="PASS",
            failure_atoms=[],
            domain="jsonspec",
            tier="metamorphic",
            bounds=ctx.task.bounds,
            soundness_grade="BOUNDED",
            metamorphic_families=[],
            cost=cost,
        )

    meta = ctx.task.metadata.get("jsonspec", {})
    configured = meta.get("metamorphic", [])
    selected = [
        name for name in configured if name in {"key_order_invariance", "whitespace_invariance"}
    ]
    if not selected:
        cost = {"ns": time.time_ns() - start, "tests": tests_run}
        return VerifierVerdict(
            verdict="PASS",
            failure_atoms=[],
            domain="jsonspec",
            tier="metamorphic",
            bounds=ctx.task.bounds,
            soundness_grade="BOUNDED",
            metamorphic_families=[],
            cost=cost,
        )
    try:
        validate_jsonspec(program.spec)
    except JsonSpecValidationError:
        cost = {"ns": time.time_ns() - start, "tests": tests_run}
        return VerifierVerdict(
            verdict="PASS",
            failure_atoms=[],
            domain="jsonspec",
            tier="metamorphic",
            bounds=ctx.task.bounds,
            soundness_grade="BOUNDED",
            metamorphic_families=[],
            cost=cost,
        )

    for family in selected:
        family_failed = False
        for example in ctx.tests:
            derived = _jsonspec_derive_input(example.inputs, family)
            if derived is None:
                continue
            try:
                base_output = run_jsonspec_program(program.spec, example.inputs)
                derived_output = run_jsonspec_program(program.spec, derived)
            except JsonSpecRuntimeError:
                family_failed = True
                break
            tests_run += 1
            if derived_output != base_output:
                family_failed = True
                break
        if family_failed:
            failure_atoms.append(f"METAMORPHIC_VIOLATION:{family}")
        metamorphic_families.append(family)

    final_verdict: Literal["PASS", "FAIL"] = "PASS" if not failure_atoms else "FAIL"
    cost = {"ns": time.time_ns() - start, "tests": tests_run}
    return VerifierVerdict(
        verdict=final_verdict,
        failure_atoms=failure_atoms,
        domain="jsonspec",
        tier="metamorphic",
        bounds=ctx.task.bounds,
        soundness_grade="BOUNDED",
        metamorphic_families=metamorphic_families,
        cost=cost,
    )


def _codepatch_patch_text(ctx: VerifierContext) -> str:
    patch_text = getattr(ctx.program, "patch", "") if ctx.program is not None else ""
    if not patch_text:
        patch_text = ctx.task.metadata.get("codepatch", {}).get("candidate_patch", "")
    return patch_text


def _strip_trailing_whitespace(paths: List[str], root: Path) -> None:
    for relpath in paths:
        target = root / relpath
        if not target.exists():
            continue
        lines = target.read_text(encoding="utf-8").splitlines()
        trimmed = [line.rstrip() for line in lines]
        target.write_text("\n".join(trimmed) + "\n", encoding="utf-8")


def _diff_patch(paths: List[str], from_root: Path, to_root: Path) -> str:
    diff_lines: List[str] = []
    for relpath in sorted(paths):
        from_path = from_root / relpath
        to_path = to_root / relpath
        from_lines = (
            from_path.read_text(encoding="utf-8").splitlines() if from_path.exists() else []
        )
        to_lines = to_path.read_text(encoding="utf-8").splitlines() if to_path.exists() else []
        diff_lines.extend(
            difflib.unified_diff(
                from_lines,
                to_lines,
                fromfile=f"a/{relpath}",
                tofile=f"b/{relpath}",
                lineterm="",
            )
        )
    if not diff_lines:
        return ""
    return "\n".join(diff_lines) + "\n"


def _find_subsequence(lines: List[str], expected: List[str]) -> int:
    if not expected:
        return 0
    limit = len(lines) - len(expected) + 1
    for idx in range(limit):
        if lines[idx : idx + len(expected)] == expected:
            return idx
    return -1


def _apply_hunks_search(lines: List[str], hunks: List[Any]) -> tuple[List[str], bool]:
    for hunk in hunks:
        expected: List[str] = []
        replacement: List[str] = []
        for raw_line in hunk.lines:
            prefix = raw_line[:1]
            payload = raw_line[1:] if len(raw_line) > 0 else ""
            if prefix in {" ", "-"}:
                expected.append(payload)
            if prefix in {" ", "+"}:
                replacement.append(payload)
        idx = _find_subsequence(lines, expected)
        if idx < 0:
            return lines, False
        lines[idx : idx + len(expected)] = replacement
    return lines, True


def _hunks_non_overlapping(hunks: List[Any]) -> bool:
    ranges: List[tuple[int, int]] = []
    for hunk in hunks:
        start = int(hunk.old_start)
        count = max(1, int(hunk.old_count))
        end = start + count - 1
        ranges.append((start, end))
    ranges.sort()
    last_end = -1
    for start, end in ranges:
        if start <= last_end:
            return False
        last_end = end
    return True


def _apply_patch_commuted(patch: str, root: Path) -> bool:
    patches = parse_unified_diff(patch)
    if not patches:
        return False
    for file_patch in patches:
        if file_patch.is_delete:
            return False
        path = root / file_patch.path
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
        else:
            if not file_patch.is_new:
                return False
            lines = []
        hunks = list(reversed(file_patch.hunks))
        updated, ok = _apply_hunks_search(lines, hunks)
        if not ok:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    return True


def codepatch_metamorphic_lane(ctx: VerifierContext) -> VerifierVerdict:
    start = time.time_ns()
    failure_atoms: List[str] = []
    metamorphic_families: List[str] = []
    tests_run = 0

    meta = ctx.task.metadata.get("codepatch", {})
    configured = meta.get("metamorphic", [])
    selected = [name for name in configured if name in _CODEPATCH_METAMORPHIC]
    patch_text = _codepatch_patch_text(ctx)
    fixture = meta.get("fixture")
    fixture_path = Path(fixture) if fixture else None

    if not selected or not patch_text or not fixture_path or not fixture_path.exists():
        cost = {"ns": time.time_ns() - start, "tests": tests_run}
        return VerifierVerdict(
            verdict="PASS",
            failure_atoms=[],
            domain="codepatch",
            tier="metamorphic",
            bounds=ctx.task.bounds,
            soundness_grade="BOUNDED",
            metamorphic_families=[],
            cost=cost,
        )

    forbidden_targets = meta.get("forbidden_targets", [])
    validation_error = validate_patch(patch_text, forbidden_targets)
    if validation_error:
        cost = {"ns": time.time_ns() - start, "tests": tests_run}
        return VerifierVerdict(
            verdict="PASS",
            failure_atoms=[],
            domain="codepatch",
            tier="metamorphic",
            bounds=ctx.task.bounds,
            soundness_grade="BOUNDED",
            metamorphic_families=[],
            cost=cost,
        )

    test_command = meta.get("test_command", default_test_command())
    timeout_s = float(meta.get("timeout_s", 2.0))
    paths = extract_patch_paths(patch_text)

    for family in selected:
        applicable = True
        family_failed = False
        if family == "whitespace_idempotent":
            with tempfile.TemporaryDirectory() as workdir:
                fixture_root = Path(workdir) / "fixture"
                shutil.copytree(fixture_path, fixture_root)
                applied, _ = apply_patch(patch_text, fixture_root)
                if not applied:
                    applicable = False
                else:
                    _strip_trailing_whitespace(paths, fixture_root)
                    result = run_tests_in_dir(
                        fixture_root, list(test_command), timeout_s=timeout_s
                    )
                    tests_run += result.tests_run
                    family_failed = result.verdict != "PASS" or bool(result.failure_atoms)
        elif family == "apply_revert_apply":
            with tempfile.TemporaryDirectory() as workdir:
                fixture_root = Path(workdir) / "fixture"
                shutil.copytree(fixture_path, fixture_root)
                original_root = Path(workdir) / "original"
                shutil.copytree(fixture_path, original_root)
                applied, _ = apply_patch(patch_text, fixture_root)
                if not applied:
                    applicable = False
                else:
                    revert_patch = _diff_patch(paths, fixture_root, original_root)
                    if revert_patch:
                        reverted, _ = apply_patch(revert_patch, fixture_root)
                    else:
                        reverted = True
                    re_applied, _ = apply_patch(patch_text, fixture_root)
                    if not (reverted and re_applied):
                        family_failed = True
                    else:
                        result = run_tests_in_dir(
                            fixture_root, list(test_command), timeout_s=timeout_s
                        )
                        tests_run += result.tests_run
                        family_failed = result.verdict != "PASS" or bool(
                            result.failure_atoms
                        )
        elif family == "commutation_safe":
            parsed = parse_unified_diff(patch_text)
            if not parsed:
                applicable = False
            else:
                has_multi = False
                for file_patch in parsed:
                    if len(file_patch.hunks) > 1:
                        has_multi = True
                        if not _hunks_non_overlapping(file_patch.hunks):
                            applicable = False
                            break
                if not has_multi:
                    applicable = False
            if applicable:
                with tempfile.TemporaryDirectory() as workdir:
                    fixture_root = Path(workdir) / "fixture"
                    shutil.copytree(fixture_path, fixture_root)
                    applied = _apply_patch_commuted(patch_text, fixture_root)
                    if not applied:
                        family_failed = True
                    else:
                        result = run_tests_in_dir(
                            fixture_root, list(test_command), timeout_s=timeout_s
                        )
                        tests_run += result.tests_run
                        family_failed = result.verdict != "PASS" or bool(
                            result.failure_atoms
                        )

        if not applicable:
            continue
        metamorphic_families.append(family)
        if family_failed:
            failure_atoms.append(f"METAMORPHIC_VIOLATION:{family}")

    final_verdict: Literal["PASS", "FAIL"] = "PASS" if not failure_atoms else "FAIL"
    cost = {"ns": time.time_ns() - start, "tests": tests_run}
    return VerifierVerdict(
        verdict=final_verdict,
        failure_atoms=failure_atoms,
        domain="codepatch",
        tier="metamorphic",
        bounds=ctx.task.bounds,
        soundness_grade="BOUNDED",
        metamorphic_families=metamorphic_families,
        cost=cost,
    )


def codepatch_lane(ctx: VerifierContext) -> VerifierVerdict:
    start = time.time_ns()
    failure_atoms: List[str] = []
    tests_run = 0
    meta = ctx.task.metadata.get("codepatch", {})
    fixture = meta.get("fixture")
    if not fixture:
        failure_atoms.append("EXCEPTION:MISSING_FIXTURE")
    else:
        fixture_path = Path(fixture)
        if not fixture_path.exists():
            failure_atoms.append("EXCEPTION:MISSING_FIXTURE")
        else:
            patch_text = getattr(ctx.program, "patch", "") if ctx.program is not None else ""
            if not patch_text:
                patch_text = meta.get("candidate_patch", "")
            if not patch_text:
                failure_atoms.append("EXCEPTION:MISSING_PATCH")
            else:
                forbidden_targets = meta.get("forbidden_targets", [])
                test_command = meta.get("test_command", default_test_command())
                timeout_s = float(meta.get("timeout_s", 2.0))
                result = run_codepatch(
                    patch_text,
                    fixture_path,
                    list(test_command),
                    forbidden_targets,
                    timeout_s=timeout_s,
                )
                tests_run = result.tests_run
                failure_atoms.extend(result.failure_atoms)
    verdict: Literal["PASS", "FAIL"] = "PASS" if not failure_atoms else "FAIL"
    cost = {"ns": time.time_ns() - start, "tests": tests_run}
    return VerifierVerdict(
        verdict=verdict,
        failure_atoms=failure_atoms,
        domain="codepatch",
        tier="codepatch",
        bounds=ctx.task.bounds,
        soundness_grade="BOUNDED",
        metamorphic_families=[],
        cost=cost,
    )
