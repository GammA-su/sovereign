from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..breaker.pyfunc_breaker import BreakerBudget, BreakerResult
from ..orchestrator.task import Task
from ..schemas import BreakerKPI
from ..utils import ensure_dir, stable_hash, write_json
from .program import CodePatchProgram, compute_codepatch_hash
from .runner import default_test_command, run_codepatch

_BASE_FAMILY_ORDER = [
    "EARLY_RETURN_REMOVAL",
    "OPERATOR_FLIP",
    "CONDITION_NEGATION",
    "OFF_BY_ONE",
    "RETURN_VALUE_FLIP",
    "CONSTANT_TWEAK",
    "HUNK_LINE_SWAP",
]

_EQ_FLIPS = {"==": "!=", "!=": "=="}
_CMP_FLIPS = {">=": "<", "<=": ">", ">": "<=", "<": ">="}
_CONST_FLIPS = {"True": "False", "False": "True", "0": "1", "1": "0"}


@dataclass
class CodepatchCounterexample:
    family: str
    patch: str
    failure_atoms: List[str]
    path: str
    metadata_path: str


@dataclass
class CodepatchBreakerReport:
    counterexamples: List[CodepatchCounterexample]
    minimized: Optional[CodepatchCounterexample]
    attempts: int
    program_hash: str
    minimized_steps: int
    failure_atoms: List[str]
    minimized_path: str
    withheld_trials: int
    withheld_hits: int


def _seed_for(task: Task, program_hash: str) -> int:
    signature = {"task_id": task.task_id, "program_hash": program_hash}
    return int(stable_hash(signature)[:8], 16)


def _is_added_line(line: str) -> bool:
    return line.startswith("+") and not line.startswith("+++")


def _iter_added_lines(lines: List[str]) -> Iterable[tuple[int, str]]:
    for idx, line in enumerate(lines):
        if _is_added_line(line):
            yield idx, line[1:]


def _replace_at(text: str, start: int, end: int, replacement: str) -> str:
    return text[:start] + replacement + text[end:]


def _mutations_off_by_one(line: str) -> List[str]:
    mutations: List[str] = []
    for match in re.finditer(r"\b-?\d+\b", line):
        value = int(match.group(0))
        for delta in (1, -1):
            mutated = str(value + delta)
            mutations.append(_replace_at(line, match.start(), match.end(), mutated))
    return mutations


def _mutations_operator_flip(line: str) -> List[str]:
    mutations: List[str] = []
    for match in re.finditer(r"==|!=", line):
        operator = match.group(0)
        replacement = _EQ_FLIPS[operator]
        mutations.append(_replace_at(line, match.start(), match.end(), replacement))
    return mutations


def _mutations_condition_negation(line: str) -> List[str]:
    mutations: List[str] = []
    for match in re.finditer(r">=|<=|>|<", line):
        operator = match.group(0)
        replacement = _CMP_FLIPS[operator]
        mutations.append(_replace_at(line, match.start(), match.end(), replacement))
    return mutations


def _mutations_early_return(line: str) -> List[str]:
    stripped = line.strip()
    if not stripped.startswith("return"):
        return []
    indent = line[: len(line) - len(line.lstrip())]
    return [f"{indent}pass"]


def _mutations_return_flip(line: str) -> List[str]:
    stripped = line.strip()
    if not stripped.startswith("return "):
        return []
    expr = stripped[len("return ") :].strip()
    if expr in _CONST_FLIPS:
        return [line.replace(expr, _CONST_FLIPS[expr], 1)]
    if re.fullmatch(r"-?\d+", expr):
        value = int(expr)
        return [line.replace(expr, str(-value), 1)]
    return []


def _mutations_constant_tweak(line: str) -> List[str]:
    mutations: List[str] = []
    for match in re.finditer(r"\b(True|False|0|1)\b", line):
        replacement = _CONST_FLIPS[match.group(0)]
        mutations.append(_replace_at(line, match.start(), match.end(), replacement))
    return mutations


def _mutations_line_swap(lines: List[str]) -> List[List[str]]:
    swaps: List[List[str]] = []
    for idx in range(len(lines) - 1):
        if _is_added_line(lines[idx]) and _is_added_line(lines[idx + 1]):
            swapped = list(lines)
            swapped[idx], swapped[idx + 1] = swapped[idx + 1], swapped[idx]
            swaps.append(swapped)
    return swaps


def _candidate_mutations(patch: str) -> Dict[str, List[str]]:
    lines = patch.splitlines()
    families: Dict[str, List[str]] = {family: [] for family in _BASE_FAMILY_ORDER}

    for idx, line in _iter_added_lines(lines):
        for family in _BASE_FAMILY_ORDER:
            if family == "EARLY_RETURN_REMOVAL":
                mutations = _mutations_early_return(line)
            elif family == "OPERATOR_FLIP":
                mutations = _mutations_operator_flip(line)
            elif family == "CONDITION_NEGATION":
                mutations = _mutations_condition_negation(line)
            elif family == "OFF_BY_ONE":
                mutations = _mutations_off_by_one(line)
            elif family == "RETURN_VALUE_FLIP":
                mutations = _mutations_return_flip(line)
            elif family == "CONSTANT_TWEAK":
                mutations = _mutations_constant_tweak(line)
            else:
                mutations = []
            for mutated in mutations:
                new_lines = list(lines)
                new_lines[idx] = f"+{mutated}"
                candidate = "\n".join(new_lines) + "\n"
                families[family].append(candidate)

    if "HUNK_LINE_SWAP" in families:
        for swapped_lines in _mutations_line_swap(lines):
            candidate = "\n".join(swapped_lines) + "\n"
            families["HUNK_LINE_SWAP"].append(candidate)

    return families


def _is_counterexample(failure_atoms: List[str]) -> bool:
    return any(atom in {"TESTS_FAILED", "TIMEOUT", "RESOURCE_LIMIT"} for atom in failure_atoms)


def _hunk_ranges(lines: List[str]) -> List[tuple[int, int]]:
    ranges: List[tuple[int, int]] = []
    idx = 0
    while idx < len(lines):
        if lines[idx].startswith("@@ "):
            start = idx
            idx += 1
            while idx < len(lines) and not lines[idx].startswith(
                "@@ "
            ) and not lines[idx].startswith("--- "):
                idx += 1
            ranges.append((start, idx))
        else:
            idx += 1
    return ranges


def _minimize_patch(
    patch: str,
    eval_fn,
    step_limit: int = 100,
) -> tuple[str, int]:
    steps = 0
    lines = patch.splitlines()

    changed = True
    while changed and steps < step_limit:
        changed = False
        for start, end in _hunk_ranges(lines):
            if steps >= step_limit:
                break
            candidate_lines = lines[:start] + lines[end:]
            candidate_patch = "\n".join(candidate_lines) + "\n"
            steps += 1
            if eval_fn(candidate_patch):
                lines = candidate_lines
                changed = True
                break

    idx = 0
    while idx < len(lines) and steps < step_limit:
        line = lines[idx]
        if line.startswith("@@ ") or line.startswith("--- ") or line.startswith("+++ "):
            idx += 1
            continue
        candidate_lines = lines[:idx] + lines[idx + 1 :]
        candidate_patch = "\n".join(candidate_lines) + "\n"
        steps += 1
        if eval_fn(candidate_patch):
            lines = candidate_lines
            continue
        idx += 1

    idx = 0
    while idx < len(lines) and steps < step_limit:
        line = lines[idx]
        if _is_added_line(line):
            content = line[1:]
            indent = content[: len(content) - len(content.lstrip())]
            if "return " in content:
                simplified = f"{indent}return 0"
            else:
                simplified = f"{indent}pass"
            if simplified != content:
                candidate_lines = list(lines)
                candidate_lines[idx] = f"+{simplified}"
                candidate_patch = "\n".join(candidate_lines) + "\n"
                steps += 1
                if eval_fn(candidate_patch):
                    lines = candidate_lines
                    continue
        idx += 1

    minimized = "\n".join(lines) + "\n"
    return minimized, steps


def _write_counterexample(
    run_dir: Path,
    idx: int,
    family: str,
    patch: str,
    failure_atoms: List[str],
    program_hash: str,
    task_id: str,
) -> tuple[str, str]:
    base_dir = run_dir / "artifacts" / "codepatch"
    ensure_dir(base_dir)
    patch_path = base_dir / f"breaker_counterexample_{idx}.patch"
    meta_path = base_dir / f"breaker_counterexample_{idx}.json"
    patch_path.write_text(patch, encoding="utf-8")
    write_json(
        meta_path,
        {
            "task_id": task_id,
            "program_hash": program_hash,
            "family": family,
            "failure_atoms": failure_atoms,
            "patch_path": str(patch_path),
        },
    )
    return str(patch_path), str(meta_path)


def run_codepatch_breaker(
    task: Task,
    program: CodePatchProgram,
    budget: BreakerBudget,
    run_dir: Path,
) -> BreakerResult:
    patch_text = program.patch
    program_hash = compute_codepatch_hash(patch_text)
    _ = _seed_for(task, program_hash)

    meta = task.metadata.get("codepatch", {})
    fixture = Path(meta.get("fixture", ""))
    forbidden_targets = meta.get("forbidden_targets", [])
    test_command = meta.get("test_command", default_test_command())
    timeout_s = float(meta.get("timeout_s", 2.0))

    attempts = 0
    counterexamples: List[CodepatchCounterexample] = []
    failure_atoms: List[str] = []
    seen: set[str] = set()
    max_attempts = max(0, budget.attempt_budget)
    withheld_trials = 0
    withheld_hits = 0

    def _eval_candidate(candidate: str, family: str) -> Optional[CodepatchCounterexample]:
        result = run_codepatch(candidate, fixture, list(test_command), forbidden_targets, timeout_s)
        if _is_counterexample(result.failure_atoms):
            patch_path, meta_path = _write_counterexample(
                run_dir,
                len(counterexamples),
                family,
                candidate,
                result.failure_atoms,
                program_hash,
                task.task_id,
            )
            return CodepatchCounterexample(
                family=family,
                patch=candidate,
                failure_atoms=result.failure_atoms,
                path=patch_path,
                metadata_path=meta_path,
            )
        return None

    def _try_candidate(candidate: str, family: str) -> bool:
        nonlocal attempts, withheld_trials, withheld_hits
        if attempts >= max_attempts:
            return True
        attempts += 1
        if task.sealed and family in task.sealed.withheld_families:
            withheld_trials += 1
        fingerprint = stable_hash({"patch": candidate})
        if fingerprint in seen:
            return attempts >= max_attempts
        seen.add(fingerprint)
        cex = _eval_candidate(candidate, family)
        if cex:
            counterexamples.append(cex)
            if task.sealed and family in task.sealed.withheld_families:
                withheld_hits += 1
            return True
        return False

    families = _candidate_mutations(patch_text)
    family_order = list(_BASE_FAMILY_ORDER)
    if task.sealed and task.sealed.withheld_families:
        withheld = [f for f in task.sealed.withheld_families if f in family_order]
        remainder = [f for f in family_order if f not in withheld]
        family_order = withheld + remainder

    for family in family_order:
        candidates = families.get(family, [])
        for candidate in candidates:
            if _try_candidate(candidate, family):
                break
        if counterexamples:
            break

    tmr = float(attempts if counterexamples else budget.attempt_budget)
    wfhr = 0.0
    if withheld_trials:
        wfhr = withheld_hits / max(1, withheld_trials)
    kpi = BreakerKPI(
        CDR=(1.0 / max(1, attempts)) if counterexamples else 0.0,
        TMR=tmr,
        NOVN=1.0 if counterexamples else 0.0,
        WFHR=wfhr,
        window={"attempts": attempts},
        budget={"attempt_budget": budget.attempt_budget},
    )

    minimized: Optional[CodepatchCounterexample] = None
    minimized_steps = 0
    minimized_path = ""
    if counterexamples:
        failure_atoms.append(f"BREAKER_FOUND:{counterexamples[0].family}")

        def _still_fails(candidate_patch: str) -> bool:
            result = run_codepatch(
                candidate_patch,
                fixture,
                list(test_command),
                forbidden_targets,
                timeout_s,
            )
            return _is_counterexample(result.failure_atoms)

        minimized_patch, minimized_steps = _minimize_patch(
            counterexamples[0].patch, _still_fails
        )
        minimized_path = str(
            (run_dir / "artifacts" / "codepatch" / "breaker_minimized.patch").resolve()
        )
        ensure_dir(Path(minimized_path).parent)
        Path(minimized_path).write_text(minimized_patch, encoding="utf-8")
        meta_path = str(
            (run_dir / "artifacts" / "codepatch" / "breaker_minimized.json").resolve()
        )
        write_json(
            Path(meta_path),
            {
                "task_id": task.task_id,
                "program_hash": program_hash,
                "family": counterexamples[0].family,
                "failure_atoms": counterexamples[0].failure_atoms,
                "patch_path": minimized_path,
            },
        )
        minimized = CodepatchCounterexample(
            family=counterexamples[0].family,
            patch=minimized_patch,
            failure_atoms=counterexamples[0].failure_atoms,
            path=minimized_path,
            metadata_path=meta_path,
        )
        failure_atoms.append("COUNTEREXAMPLE_MINIMIZED")

    report = CodepatchBreakerReport(
        counterexamples=counterexamples,
        minimized=minimized,
        attempts=attempts,
        program_hash=program_hash,
        minimized_steps=minimized_steps,
        failure_atoms=failure_atoms,
        minimized_path=minimized_path,
        withheld_trials=withheld_trials,
        withheld_hits=withheld_hits,
    )

    return BreakerResult(
        counterexample=None,
        minimized=None,
        kpi=kpi,
        report={
            "counterexamples": [cex.__dict__ for cex in report.counterexamples],
            "minimized": report.minimized.__dict__ if report.minimized else None,
            "attempts": report.attempts,
            "program_hash": report.program_hash,
            "minimized_steps": report.minimized_steps,
            "failure_atoms": report.failure_atoms,
            "minimized_path": report.minimized_path,
            "withheld_trials": report.withheld_trials,
            "withheld_hits": report.withheld_hits,
        },
    )
