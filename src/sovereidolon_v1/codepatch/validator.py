from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _normalize_patch_path(path: str) -> str:
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    return path.replace("\\", "/")


def _extract_patch_paths(patch: str) -> list[str]:
    paths: list[str] = []
    for raw_line in patch.splitlines():
        if raw_line.startswith("--- ") or raw_line.startswith("+++ "):
            path = raw_line[4:].split("\t", 1)[0].strip()
            if not path or path == "/dev/null":
                continue
            path = _normalize_patch_path(path)
            paths.append(path)
    seen = set()
    ordered: list[str] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def extract_patch_paths(patch: str) -> list[str]:
    return _extract_patch_paths(patch)


def _is_path_escape(path: str) -> bool:
    probe = Path(path)
    if probe.is_absolute():
        return True
    return ".." in probe.parts


def _is_forbidden(path: str, forbidden_targets: Iterable[str]) -> bool:
    normalized = Path(path).as_posix().lstrip("/")
    for target in forbidden_targets:
        target_norm = Path(target).as_posix().lstrip("/")
        if normalized == target_norm or normalized.startswith(f"{target_norm}/"):
            return True
    return False


def validate_patch(patch: str, forbidden_targets: Iterable[str]) -> str:
    for path in _extract_patch_paths(patch):
        if _is_path_escape(path):
            return "PATCH_PATH_ESCAPE"
        if _is_forbidden(path, forbidden_targets):
            return "PATCH_FORBIDDEN_TARGET"
    return ""
