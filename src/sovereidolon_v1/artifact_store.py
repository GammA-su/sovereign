from __future__ import annotations

from pathlib import Path
from typing import Any

from .ledger.ledger import Ledger
from .schemas import ArtifactRecord
from .utils import canonical_dumps, ensure_dir, hash_bytes


class ArtifactStore:
    def __init__(self, root: Path, ledger: Ledger) -> None:
        self.root = root
        self.ledger = ledger
        ensure_dir(root)

    def write_json(self, rel_path: str, data: Any, kind: str) -> ArtifactRecord:
        path = self.root / rel_path
        ensure_dir(path.parent)
        payload = canonical_dumps(data)
        path.write_bytes(payload)
        record = ArtifactRecord(
            path=str(path), content_hash=hash_bytes(payload), bytes=len(payload), kind=kind
        )
        self.ledger.append(
            "ARTIFACT_WRITTEN",
            {
                "path": record.path,
                "content_hash": record.content_hash,
                "bytes": record.bytes,
                "kind": record.kind,
            },
        )
        return record

    def write_text(self, rel_path: str, text: str, kind: str) -> ArtifactRecord:
        data = text.encode("utf-8")
        path = self.root / rel_path
        ensure_dir(path.parent)
        path.write_bytes(data)
        record = ArtifactRecord(
            path=str(path), content_hash=hash_bytes(data), bytes=len(data), kind=kind
        )
        self.ledger.append(
            "ARTIFACT_WRITTEN",
            {
                "path": record.path,
                "content_hash": record.content_hash,
                "bytes": record.bytes,
                "kind": record.kind,
            },
        )
        return record

    def write_bytes(self, rel_path: str, data: bytes, kind: str) -> ArtifactRecord:
        path = self.root / rel_path
        ensure_dir(path.parent)
        path.write_bytes(data)
        record = ArtifactRecord(
            path=str(path), content_hash=hash_bytes(data), bytes=len(data), kind=kind
        )
        self.ledger.append(
            "ARTIFACT_WRITTEN",
            {
                "path": record.path,
                "content_hash": record.content_hash,
                "bytes": record.bytes,
                "kind": record.kind,
            },
        )
        return record
