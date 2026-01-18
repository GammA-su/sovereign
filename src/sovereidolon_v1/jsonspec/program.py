from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import orjson

from ..utils import canonical_dumps, hash_bytes

JSONSPEC_INTERPRETER_HASH = "jsonspec-v1"


def canonical_jsonspec(spec: Any) -> Any:
    return orjson.loads(canonical_dumps(spec))


def jsonspec_bytes(spec: Any) -> bytes:
    return canonical_dumps(spec)


def compute_jsonspec_hash(spec: Any) -> str:
    return hash_bytes(jsonspec_bytes(spec))


@dataclass
class JsonSpecProgram:
    spec: Dict[str, Any]

    def __post_init__(self) -> None:
        canonical = canonical_jsonspec(self.spec)
        object.__setattr__(self, "spec", canonical)

    def to_json(self) -> Dict[str, Any]:
        return self.spec

    def to_bytes(self) -> bytes:
        return jsonspec_bytes(self.spec)
