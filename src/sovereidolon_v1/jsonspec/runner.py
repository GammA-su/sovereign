from __future__ import annotations

import json
from typing import Any, Dict, List


class JsonSpecRuntimeError(Exception):
    def __init__(self, failure_atom: str) -> None:
        super().__init__(failure_atom)
        self.failure_atom = failure_atom


def _is_json_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, str):
        return True
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(val) for key, val in value.items())
    return False


def _extract_root(inputs: Dict[str, Any]) -> Any:
    if len(inputs) == 1:
        return next(iter(inputs.values()))
    return inputs


def _normalize_root(root: Any) -> Dict[str, Any]:
    if isinstance(root, str):
        try:
            root = json.loads(root)
        except json.JSONDecodeError as exc:
            raise JsonSpecRuntimeError("JSONSPEC_BAD_JSON") from exc
    if not isinstance(root, dict):
        raise JsonSpecRuntimeError("JSONSPEC_INPUT_NOT_OBJECT")
    return root


def _resolve_path(root: Dict[str, Any], path: List[str]) -> Any:
    current: Any = root
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            raise JsonSpecRuntimeError("JSONSPEC_MISSING_PATH")
    return current


def _eval_node(node: Dict[str, Any], root: Dict[str, Any]) -> Any:
    op = node["op"]
    if op == "const":
        return node.get("value")
    if op == "get":
        path = node.get("path", [])
        try:
            return _resolve_path(root, path)
        except JsonSpecRuntimeError:
            if "default" in node:
                return node.get("default")
            raise
    if op == "keys":
        path = node.get("path", [])
        target = root
        if path:
            target = _resolve_path(root, path)
        if not isinstance(target, dict):
            raise JsonSpecRuntimeError("JSONSPEC_KEYS_NOT_OBJECT")
        return list(target.keys())
    if op == "object":
        items = node.get("items", {})
        return {key: _eval_node(value, root) for key, value in items.items()}
    if op == "array":
        items = node.get("items", [])
        return [_eval_node(item, root) for item in items]
    raise JsonSpecRuntimeError("JSONSPEC_INVALID_OP")


def run_jsonspec_program(program_spec: Dict[str, Any], inputs: Dict[str, Any]) -> Any:
    root = _normalize_root(_extract_root(inputs))
    output = _eval_node(program_spec, root)
    if not _is_json_value(output):
        raise JsonSpecRuntimeError("JSONSPEC_BAD_OUTPUT")
    return output
