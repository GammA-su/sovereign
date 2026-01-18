from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .validator import _normalize_patch_path

_HUNK_RE = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[str]


@dataclass
class FilePatch:
    path: str
    hunks: List[Hunk]
    is_new: bool
    is_delete: bool


def _parse_path(line: str) -> str:
    return line[4:].split("\t", 1)[0].strip()


def _parse_unified_diff(patch: str) -> List[FilePatch]:
    lines = patch.splitlines()
    patches: List[FilePatch] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("--- "):
            old_path = _parse_path(line)
            idx += 1
            if idx >= len(lines) or not lines[idx].startswith("+++ "):
                return []
            new_path = _parse_path(lines[idx])
            idx += 1
            target = new_path if new_path != "/dev/null" else old_path
            if target == "/dev/null":
                return []
            target = _normalize_patch_path(target)
            file_patch = FilePatch(
                path=target,
                hunks=[],
                is_new=old_path == "/dev/null",
                is_delete=new_path == "/dev/null",
            )
            while idx < len(lines) and lines[idx].startswith("@@ "):
                match = _HUNK_RE.match(lines[idx])
                if not match:
                    return []
                old_start = int(match.group(1))
                old_count = int(match.group(2) or "1")
                new_start = int(match.group(3))
                new_count = int(match.group(4) or "1")
                idx += 1
                hunk_lines: List[str] = []
                while idx < len(lines):
                    if lines[idx].startswith("@@ ") or lines[idx].startswith("--- "):
                        break
                    if lines[idx].startswith("\\ No newline"):
                        idx += 1
                        continue
                    hunk_lines.append(lines[idx])
                    idx += 1
                file_patch.hunks.append(
                    Hunk(
                        old_start=old_start,
                        old_count=old_count,
                        new_start=new_start,
                        new_count=new_count,
                        lines=hunk_lines,
                    )
                )
            patches.append(file_patch)
            continue
        idx += 1
    return patches


def parse_unified_diff(patch: str) -> List[FilePatch]:
    return _parse_unified_diff(patch)


def _apply_hunks(lines: List[str], hunks: List[Hunk]) -> tuple[List[str], bool]:
    offset = 0
    for hunk in hunks:
        idx = hunk.old_start - 1 + offset
        expected: List[str] = []
        replacement: List[str] = []
        for raw_line in hunk.lines:
            prefix = raw_line[:1]
            payload = raw_line[1:] if len(raw_line) > 0 else ""
            if prefix in {" ", "-"}:
                expected.append(payload)
            if prefix in {" ", "+"}:
                replacement.append(payload)
        if idx < 0 or idx + len(expected) > len(lines):
            return lines, False
        if lines[idx : idx + len(expected)] != expected:
            return lines, False
        lines[idx : idx + len(expected)] = replacement
        offset += len(replacement) - len(expected)
    return lines, True


def apply_patch(patch: str, root: Path) -> tuple[bool, str]:
    patches = _parse_unified_diff(patch)
    if not patches:
        return False, "PATCH_APPLY_FAILED"
    for file_patch in patches:
        if file_patch.is_delete:
            return False, "PATCH_APPLY_FAILED"
        path = root / file_patch.path
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
        else:
            if not file_patch.is_new:
                return False, "PATCH_APPLY_FAILED"
            lines = []
        updated, ok = _apply_hunks(lines, file_patch.hunks)
        if not ok:
            return False, "PATCH_APPLY_FAILED"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    return True, ""
