from __future__ import annotations

from typing import Any, Dict, List


class JsonSpecValidationError(Exception):
    def __init__(self, failure_atom: str) -> None:
        super().__init__(failure_atom)
        self.failure_atom = failure_atom


ALLOWED_OPS = {"const", "get", "object", "array", "keys"}
MAX_DEPTH = 20
MAX_NODES = 200


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


def _validate_path(node: Dict[str, Any]) -> List[str]:
    path = node.get("path", [])
    if not isinstance(path, list) or not all(isinstance(item, str) for item in path):
        raise JsonSpecValidationError("JSONSPEC_BAD_PATH")
    return path


def _validate_node(node: Any, depth: int, state: Dict[str, int]) -> None:
    state["nodes"] += 1
    if state["nodes"] > MAX_NODES:
        raise JsonSpecValidationError("JSONSPEC_TOO_LARGE")
    if depth > MAX_DEPTH:
        raise JsonSpecValidationError("JSONSPEC_TOO_DEEP")
    if not isinstance(node, dict):
        raise JsonSpecValidationError("JSONSPEC_INVALID_NODE")
    op = node.get("op")
    if not isinstance(op, str):
        raise JsonSpecValidationError("JSONSPEC_INVALID_NODE")
    if op not in ALLOWED_OPS:
        raise JsonSpecValidationError("JSONSPEC_INVALID_OP")
    if op == "const":
        if "value" not in node or not _is_json_value(node["value"]):
            raise JsonSpecValidationError("JSONSPEC_BAD_CONST")
        return
    if op == "get":
        _validate_path(node)
        if "default" in node and not _is_json_value(node["default"]):
            raise JsonSpecValidationError("JSONSPEC_BAD_CONST")
        return
    if op == "keys":
        _validate_path(node)
        return
    if op == "object":
        items = node.get("items", None)
        if not isinstance(items, dict):
            raise JsonSpecValidationError("JSONSPEC_BAD_ITEMS")
        for key, value in items.items():
            if not isinstance(key, str):
                raise JsonSpecValidationError("JSONSPEC_BAD_ITEMS")
            _validate_node(value, depth + 1, state)
        return
    if op == "array":
        items = node.get("items", None)
        if not isinstance(items, list):
            raise JsonSpecValidationError("JSONSPEC_BAD_ITEMS")
        for item in items:
            _validate_node(item, depth + 1, state)
        return


def validate_jsonspec(spec: Any) -> None:
    state = {"nodes": 0}
    _validate_node(spec, 0, state)
