from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

import orjson
from blake3 import blake3

CANONICALIZATION = "orjson_sort_keys_utf8"
HASH_ALGORITHM = "blake3"


def canonical_dumps(data: Any) -> bytes:
    return orjson.dumps(data, option=orjson.OPT_SORT_KEYS)


def stable_hash(data: Any) -> str:
    return blake3(canonical_dumps(data)).hexdigest()


def hash_bytes(data: bytes) -> str:
    return blake3(data).hexdigest()


def now_ts_ns() -> int:
    return time.time_ns()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_bytes(canonical_dumps(data))


def read_json(path: Path) -> Any:
    return orjson.loads(path.read_bytes())


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_jsonl_line(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    line = canonical_dumps(data) + b"\n"
    with path.open("ab") as handle:
        handle.write(line)


def read_jsonl(path: Path) -> list[Any]:
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines()
    return [orjson.loads(line) for line in lines if line]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if isinstance(value, float):
        return float(f"{value:.6f}")
    return json.loads(json.dumps(value, default=str))


def stable_hash_from_items(items: Iterable[Any]) -> str:
    return stable_hash(list(items))
