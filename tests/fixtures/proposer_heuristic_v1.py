#!/usr/bin/env python3
import json
import sys


def _json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _task_payload(payload: dict) -> dict:
    task = payload.get("task_spec")
    if isinstance(task, dict):
        return task
    task = payload.get("task")
    if isinstance(task, dict):
        return task
    return {}


def _pyfunc_program(task_id: str, metadata: dict) -> str:
    candidate = metadata.get("pyfunc", {}).get("candidate_program", "")
    if "fail" in task_id or "meta_fail" in task_id:
        return "def solve(a, b):\n    return a + b\n"
    return candidate


def _jsonspec_program(task_id: str, metadata: dict) -> str:
    spec = metadata.get("jsonspec", {}).get("spec_program")
    if spec is None:
        spec = metadata.get("jsonspec", {}).get("candidate_program")
    if spec is None:
        return ""
    if isinstance(spec, dict):
        return _json_dumps(spec)
    if isinstance(spec, str):
        return spec
    return ""


def _codepatch_program(task_id: str, metadata: dict) -> str:
    candidate = metadata.get("codepatch", {}).get("candidate_patch", "")
    if "fail" in task_id:
        return (
            "--- a/mini_proj/core.py\n"
            "+++ b/mini_proj/core.py\n"
            "@@ -4,2 +4,2 @@\n"
            "-def add(a: int, b: int) -> int:\n"
            "-    return a - b\n"
            "+def add(a: int, b: int) -> int:\n"
            "+    return a + b\n"
        )
    return candidate


def main() -> int:
    payload = json.load(sys.stdin)
    task_id = payload.get("task_id", "")
    domain = payload.get("domain", "")
    task = _task_payload(payload)
    metadata = task.get("metadata", {}) if isinstance(task, dict) else {}

    program_text = ""
    if domain == "pyfunc":
        program_text = _pyfunc_program(task_id, metadata)
    elif domain == "jsonspec":
        program_text = _jsonspec_program(task_id, metadata)
    elif domain == "codepatch":
        program_text = _codepatch_program(task_id, metadata)

    if not program_text:
        result = {"ok": False, "error_atom": "EXCEPTION:PROPOSER_SUBPROCESS_FAILED"}
        sys.stdout.write(_json_dumps(result))
        return 0

    result = {
        "ok": True,
        "program_text": program_text,
        "metadata": {"kind": "heuristic_v1"},
    }
    sys.stdout.write(_json_dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
