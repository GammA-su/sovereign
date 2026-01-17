from __future__ import annotations

import argparse
import builtins
import json
import sys
from pathlib import Path
from typing import Any

from .program import ALLOWED_BUILTINS, PyFuncValidationError, validate_pyfunc_code

SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in ALLOWED_BUILTINS
    if hasattr(builtins, name)
}


def _limit_resources() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
        resource.setrlimit(resource.RLIMIT_AS, (128 * 1024 * 1024, 128 * 1024 * 1024))
    except (ImportError, ValueError):
        pass


def _run_function(
    code: str,
    entrypoint: str,
    tests: list[dict[str, Any]],
    metamorphic: list[str],
) -> dict[str, Any]:
    namespace: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
    try:
        exec(compile(code, "<pyfunc>", "exec"), namespace, namespace)
    except MemoryError:
        return {
            "verdict": "FAIL",
            "failure_atoms": ["RESOURCE_LIMIT"],
            "metamorphic_families": [],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "verdict": "FAIL",
            "failure_atoms": [f"EXCEPTION:{exc.__class__.__name__}"],
            "metamorphic_families": [],
        }
    func = namespace.get(entrypoint)
    if not callable(func):
        return {
            "verdict": "FAIL",
            "failure_atoms": ["EXCEPTION:ENTRYPOINT_MISSING"],
            "metamorphic_families": [],
        }
    for example in tests:
        try:
            output = func(**example["inputs"])
        except MemoryError:
            return {
                "verdict": "FAIL",
                "failure_atoms": ["RESOURCE_LIMIT"],
                "metamorphic_families": [],
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "verdict": "FAIL",
                "failure_atoms": [f"EXCEPTION:{exc.__class__.__name__}"],
                "metamorphic_families": [],
            }
        if output != example["output"]:
            return {
                "verdict": "FAIL",
                "failure_atoms": ["EXCEPTION:TEST_MISMATCH"],
                "metamorphic_families": [],
            }
    executed: list[str] = []
    for family in metamorphic:
        if family == "idempotent":
            if not tests or len(tests[0]["inputs"]) != 1:
                continue
            key = next(iter(tests[0]["inputs"]))
            base_value = func(**tests[0]["inputs"])
            try:
                second = func(**{key: base_value})
            except MemoryError:
                return {
                    "verdict": "FAIL",
                    "failure_atoms": ["RESOURCE_LIMIT"],
                    "metamorphic_families": executed,
                }
            except Exception as exc:  # noqa: BLE001
                return {
                    "verdict": "FAIL",
                    "failure_atoms": [f"EXCEPTION:{exc.__class__.__name__}"],
                    "metamorphic_families": executed,
                }
            if second != base_value:
                return {
                    "verdict": "FAIL",
                    "failure_atoms": ["EXCEPTION:METAMORPHIC_VIOLATION"],
                    "metamorphic_families": executed,
                }
            executed.append("idempotent")
        elif family == "commutative":
            if not tests or len(tests[0]["inputs"]) != 2:
                continue
            keys = list(tests[0]["inputs"].keys())
            if len(keys) != 2:
                continue
            swapped = {
                keys[0]: tests[0]["inputs"][keys[1]],
                keys[1]: tests[0]["inputs"][keys[0]],
            }
            first = func(**tests[0]["inputs"])
            second = func(**swapped)
            if first != second:
                return {
                    "verdict": "FAIL",
                    "failure_atoms": ["EXCEPTION:METAMORPHIC_VIOLATION"],
                    "metamorphic_families": executed,
                }
            executed.append("commutative")
    return {
        "verdict": "PASS",
        "failure_atoms": [],
        "metamorphic_families": executed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--entrypoint", type=str, default="solve")
    args = parser.parse_args()
    payload = json.loads(sys.stdin.read())
    tests = payload.get("tests", [])
    metamorphic = payload.get("metamorphic", [])
    code = args.program.read_text()
    try:
        validate_pyfunc_code(code, args.entrypoint)
    except PyFuncValidationError as exc:
        result = {
            "verdict": "FAIL",
            "failure_atoms": [exc.failure_atom],
            "metamorphic_families": [],
        }
        print(json.dumps(result))
        return
    except Exception as exc:  # noqa: BLE001
        result = {
            "verdict": "FAIL",
            "failure_atoms": [f"EXCEPTION:{exc.__class__.__name__}"],
            "metamorphic_families": [],
        }
        print(json.dumps(result))
        return
    _limit_resources()
    try:
        result = _run_function(code, args.entrypoint, tests, metamorphic)
    except Exception:  # noqa: BLE001
        result = {
            "verdict": "FAIL",
            "failure_atoms": ["EXCEPTION:RUNNER_CRASH"],
            "metamorphic_families": [],
        }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
