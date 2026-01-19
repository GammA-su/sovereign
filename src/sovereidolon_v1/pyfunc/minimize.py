from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .program import canonical_pyfunc_source, compute_pyfunc_hash

PYTHON_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class MinimizedProgram:
    code: str
    program_hash: str
    attempts: int


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
    program_path = program_path.resolve()
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


def _run_pyexec(
    program_path: Path, entrypoint: str, tests: List[Dict[str, Any]]
) -> Dict[str, Any]:
    payload = {"tests": tests, "metamorphic": []}
    with tempfile.TemporaryDirectory() as workdir:
        try:
            proc = subprocess.run(
                _pyfunc_command(program_path, entrypoint),
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=1.0,
                env=_pyexec_env(),
                cwd=workdir,
            )
        except subprocess.TimeoutExpired:
            return {"verdict": "FAIL", "failure_atoms": ["TIMEOUT"]}
    if proc.returncode != 0:
        return {"verdict": "FAIL", "failure_atoms": ["RESOURCE_LIMIT"]}
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {"verdict": "FAIL", "failure_atoms": ["BAD_OUTPUT"]}
    if not isinstance(data, dict):
        return {"verdict": "FAIL", "failure_atoms": ["BAD_OUTPUT"]}
    return data


def _fails_with_atom(
    program_path: Path, entrypoint: str, tests: List[Dict[str, Any]], failure_atom: str
) -> bool:
    result = _run_pyexec(program_path, entrypoint, tests)
    failure_atoms = result.get("failure_atoms", [])
    return failure_atom in failure_atoms


def minimize_pyfunc_failure(
    code: str,
    entrypoint: str,
    tests: List[Dict[str, Any]],
    failure_atom: str,
    budget: int = 50,
) -> MinimizedProgram:
    canonical = canonical_pyfunc_source(code)
    lines = canonical.splitlines()
    attempts = 0
    best = canonical

    with tempfile.TemporaryDirectory() as workdir:
        program_path = Path(workdir) / "program.py"
        program_path.write_text(best, encoding="utf-8")
        if not _fails_with_atom(program_path, entrypoint, tests, failure_atom):
            return MinimizedProgram(
                code=canonical,
                program_hash=compute_pyfunc_hash(canonical),
                attempts=0,
            )

        changed = True
        while changed and attempts < budget:
            changed = False
            for idx in range(len(lines)):
                if attempts >= budget:
                    break
                candidate_lines = [line for i, line in enumerate(lines) if i != idx]
                candidate = "\n".join(candidate_lines) + "\n"
                program_path.write_text(candidate, encoding="utf-8")
                attempts += 1
                if _fails_with_atom(program_path, entrypoint, tests, failure_atom):
                    lines = candidate_lines
                    best = candidate
                    changed = True
                    break

    return MinimizedProgram(
        code=best,
        program_hash=compute_pyfunc_hash(best),
        attempts=attempts,
    )
