from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from ..utils import now_ts_ns, read_jsonl, stable_hash, to_jsonable, write_jsonl_line


class Ledger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._last_hash = ""
        if path.exists():
            entries = read_jsonl(path)
            if entries:
                self._last_hash = entries[-1].get("hash", "")

    def append(self, event_type: str, payload: Dict[str, Any]) -> str:
        payload_json = to_jsonable(payload)
        event = {
            "ts": now_ts_ns(),
            "type": event_type,
            "payload": payload_json,
            "prev_hash": self._last_hash,
        }
        event_hash = stable_hash(event)
        event["hash"] = event_hash
        write_jsonl_line(self.path, event)
        self._last_hash = event_hash
        return event_hash

    @staticmethod
    def verify_chain(path: Path) -> Tuple[bool, str]:
        entries = read_jsonl(path)
        prev_hash = ""
        for idx, entry in enumerate(entries):
            expected_hash = entry.get("hash", "")
            recomputed = stable_hash(
                {
                    "ts": entry.get("ts"),
                    "type": entry.get("type"),
                    "payload": entry.get("payload"),
                    "prev_hash": entry.get("prev_hash"),
                }
            )
            if entry.get("prev_hash") != prev_hash:
                return False, f"prev_hash mismatch at {idx}"
            if recomputed != expected_hash:
                return False, f"hash mismatch at {idx}"
            prev_hash = expected_hash
        return True, "ok"
