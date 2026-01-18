from __future__ import annotations

from dataclasses import dataclass

from ..utils import hash_bytes

CODEPATCH_INTERPRETER_HASH = "codepatch-v1"


def canonical_patch_source(patch: str) -> str:
    normalized = patch.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.rstrip("\n")
    return normalized + "\n"


def compute_codepatch_hash(patch: str) -> str:
    return hash_bytes(canonical_patch_source(patch).encode("utf-8"))


@dataclass
class CodePatchProgram:
    patch: str

    def __post_init__(self) -> None:
        canonical = canonical_patch_source(self.patch)
        object.__setattr__(self, "patch", canonical)

    def to_json(self) -> dict[str, str]:
        return {"patch": self.patch}

    def to_bytes(self) -> bytes:
        return canonical_patch_source(self.patch).encode("utf-8")
