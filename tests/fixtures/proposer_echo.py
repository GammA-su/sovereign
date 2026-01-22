#!/usr/bin/env python3
import json
import sys


def _json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def main() -> int:
    payload = json.load(sys.stdin)
    task_id = payload.get("task_id", "")
    domain = payload.get("domain", "")
    task_spec = payload.get("task_spec", {}) or {}
    if task_id == "pyfunc_fail_01":
        result = {"ok": False, "error_atom": "EXCEPTION:PROPOSER_SUBPROCESS_FAILED"}
        sys.stdout.write(_json_dumps(result))
        return 0
    metadata = task_spec.get("metadata", {}) if isinstance(task_spec, dict) else {}
    candidate = ""
    if domain == "pyfunc":
        candidate = metadata.get("pyfunc", {}).get("candidate_program", "")
    elif domain == "codepatch":
        candidate = metadata.get("codepatch", {}).get("candidate_patch", "")
    elif domain == "jsonspec":
        candidate_spec = metadata.get("jsonspec", {}).get("candidate_program")
        if isinstance(candidate_spec, dict):
            candidate = _json_dumps(candidate_spec)
        elif isinstance(candidate_spec, str):
            candidate = candidate_spec
    if not candidate:
        result = {"ok": False, "error_atom": "EXCEPTION:PROPOSER_SUBPROCESS_FAILED"}
        sys.stdout.write(_json_dumps(result))
        return 0
    result = {"ok": True, "program_text": candidate, "metadata": {}}
    sys.stdout.write(_json_dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
