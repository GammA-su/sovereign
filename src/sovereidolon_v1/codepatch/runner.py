from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from .applier import apply_patch
from .validator import validate_patch


def _test_env() -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": "",
        "PYTHONHOME": "",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
    }


def _limit_resources() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
    except (ImportError, ValueError):
        pass


@dataclass
class CodePatchRunResult:
    verdict: str
    failure_atoms: List[str]
    tests_run: int
    duration_ns: int


def run_codepatch(
    patch: str,
    fixture_dir: Path,
    test_command: List[str],
    forbidden_targets: Iterable[str],
    timeout_s: float = 2.0,
) -> CodePatchRunResult:
    start = time.time_ns()
    failure_atoms: List[str] = []
    validation_error = validate_patch(patch, forbidden_targets)
    if validation_error:
        return CodePatchRunResult(
            verdict="FAIL",
            failure_atoms=[validation_error],
            tests_run=0,
            duration_ns=time.time_ns() - start,
        )
    with tempfile.TemporaryDirectory() as workdir:
        fixture_root = Path(workdir) / "fixture"
        try:
            shutil.copytree(fixture_dir, fixture_root)
        except Exception as exc:  # noqa: BLE001
            return CodePatchRunResult(
                verdict="FAIL",
                failure_atoms=[f"EXCEPTION:{exc.__class__.__name__}"],
                tests_run=0,
                duration_ns=time.time_ns() - start,
            )
        applied, atom = apply_patch(patch, fixture_root)
        if not applied:
            return CodePatchRunResult(
                verdict="FAIL",
                failure_atoms=[atom],
                tests_run=0,
                duration_ns=time.time_ns() - start,
            )
        failure_atoms, tests_run = _run_tests(fixture_root, test_command, timeout_s)
    verdict = "PASS" if not failure_atoms else "FAIL"
    return CodePatchRunResult(
        verdict=verdict,
        failure_atoms=failure_atoms,
        tests_run=tests_run,
        duration_ns=time.time_ns() - start,
    )


def default_test_command() -> List[str]:
    return [sys.executable, "-m", "pytest", "-q"]


def run_tests_in_dir(
    root: Path,
    test_command: List[str],
    timeout_s: float = 2.0,
) -> CodePatchRunResult:
    start = time.time_ns()
    failure_atoms, tests_run = _run_tests(root, test_command, timeout_s)
    verdict = "PASS" if not failure_atoms else "FAIL"
    return CodePatchRunResult(
        verdict=verdict,
        failure_atoms=failure_atoms,
        tests_run=tests_run,
        duration_ns=time.time_ns() - start,
    )


def _run_tests(
    root: Path, test_command: List[str], timeout_s: float
) -> tuple[List[str], int]:
    failure_atoms: List[str] = []
    tests_run = 0
    try:
        proc = subprocess.run(
            test_command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=_test_env(),
            preexec_fn=_limit_resources,
        )
    except subprocess.TimeoutExpired:
        failure_atoms.append("TIMEOUT")
        tests_run = 1
    except Exception as exc:  # noqa: BLE001
        failure_atoms.append(f"EXCEPTION:{exc.__class__.__name__}")
        tests_run = 1
    else:
        tests_run = 1
        if proc.returncode < 0:
            signal_num = -proc.returncode
            if signal_num in {signal.SIGXCPU, signal.SIGALRM, signal.SIGKILL}:
                failure_atoms.append("TIMEOUT")
            else:
                failure_atoms.append("RESOURCE_LIMIT")
        elif proc.returncode != 0:
            failure_atoms.append("TESTS_FAILED")
    return failure_atoms, tests_run
