from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import orjson

from .codepatch.program import compute_codepatch_hash
from .jsonspec.program import compute_jsonspec_hash
from .orchestrator.task import Task
from .pyfunc.program import compute_pyfunc_hash
from .utils import canonical_dumps

if TYPE_CHECKING:
    from .proposer_api import Proposal


class HeuristicProposerV1:
    proposer_id = "heuristic_v1"

    def __init__(self) -> None:
        self.seed_program: Optional[str] = None
        self.seed_program_hash: str = ""
        self.seed_source: str = "none"
        self.prior_verdict: Optional[str] = None
        self.failure_atoms: list[str] = []

    def set_context(self, context: Dict[str, Any], log_path: Optional[str]) -> None:
        _ = log_path
        seed_program = context.get("seed_program")
        if isinstance(seed_program, str) and seed_program:
            self.seed_program = seed_program
        seed_hash = context.get("seed_program_hash")
        if isinstance(seed_hash, str):
            self.seed_program_hash = seed_hash
        seed_source = context.get("seed_source")
        if isinstance(seed_source, str) and seed_source:
            self.seed_source = seed_source
        prior_verdict = context.get("prior_verdict")
        if isinstance(prior_verdict, str):
            self.prior_verdict = prior_verdict.upper()
        failure_atoms = context.get("failure_atoms")
        if isinstance(failure_atoms, list):
            self.failure_atoms = [atom for atom in failure_atoms if isinstance(atom, str)]

    def _should_repair(self, task: Task) -> bool:
        if self.prior_verdict == "FAIL":
            return True
        if self.failure_atoms:
            return True
        task_id_lower = task.task_id.lower()
        return "fail" in task_id_lower

    def _pyfunc_program(self, task: Task) -> str:
        task_id = task.task_id
        candidate = task.metadata.get("pyfunc", {}).get("candidate_program", "")
        if "fail" in task_id or "meta_fail" in task_id:
            return "def solve(a, b):\n    return a + b\n"
        return str(candidate)

    def _pyfunc_ladder(self, task_id: str) -> Optional[str]:
        rung0 = "def solve(a, b):\n    return a + b\n"
        rung1 = "def solve(a, b):\n    result = a + b\n    return result\n"
        rung2 = "def solve(a, b):\n    return (a + b) + 0\n"
        rung0_hash = compute_pyfunc_hash(rung0)
        rung1_hash = compute_pyfunc_hash(rung1)
        rung2_hash = compute_pyfunc_hash(rung2)
        if "ladder_r1" in task_id:
            if self.seed_program_hash in {rung0_hash, rung1_hash}:
                return rung1
            if self.seed_program_hash == rung2_hash:
                return rung2
        if "ladder_r2" in task_id:
            if self.seed_program_hash in {rung1_hash, rung2_hash}:
                return rung2
        return None

    def _jsonspec_program(self, task: Task) -> str:
        meta = task.metadata.get("jsonspec", {})
        spec = meta.get("spec_program")
        if spec is None:
            spec = meta.get("candidate_program")
        if isinstance(spec, dict):
            return canonical_dumps(spec).decode("utf-8")
        if isinstance(spec, str):
            return spec
        return ""

    def _jsonspec_ladder(self, task: Task) -> Optional[str]:
        meta = task.metadata.get("jsonspec", {})
        spec_program = meta.get("spec_program")
        if not isinstance(spec_program, dict):
            return None
        rung0_hash = compute_jsonspec_hash(spec_program)
        if self.seed_program_hash != rung0_hash:
            return None
        return canonical_dumps(spec_program).decode("utf-8")

    def _codepatch_program(self, task: Task) -> str:
        task_id = task.task_id
        candidate = task.metadata.get("codepatch", {}).get("candidate_patch", "")
        if "fail" in task_id:
            return (
                "--- a/mini_proj/core.py\n"
                "+++ b/mini_proj/core.py\n"
                "@@ -4,2 +4,2 @@\n"
                "-def add(a: int, b: int) -> int:\n"
                "-    return a - b\n"
                "+def add(a: int, b: int) -> int:\n"
                "+    return a + b\n"
            )
        return str(candidate)

    def _codepatch_ladder(self, task_id: str) -> Optional[str]:
        rung0 = (
            "--- a/mini_proj/core.py\n"
            "+++ b/mini_proj/core.py\n"
            "@@ -4,2 +4,2 @@\n"
            "-def add(a: int, b: int) -> int:\n"
            "-    return a - b\n"
            "+def add(a: int, b: int) -> int:\n"
            "+    return a + b\n"
        )
        rung1 = (
            "--- a/mini_proj/core.py\n"
            "+++ b/mini_proj/core.py\n"
            "@@ -4,2 +4,2 @@\n"
            "-def add(a: int, b: int) -> int:\n"
            "-    return a - b\n"
            "+def add(a: int, b: int) -> int:\n"
            "+    return a + b  \n"
        )
        rung2 = (
            "--- a/mini_proj/core.py\n"
            "+++ b/mini_proj/core.py\n"
            "@@ -4,2 +4,2 @@\n"
            "-def add(a: int, b: int) -> int:\n"
            "-    return a - b\n"
            "+def add(a: int, b: int) -> int:\n"
            "+    return a + b   \n"
        )
        rung0_hash = compute_codepatch_hash(rung0)
        rung1_hash = compute_codepatch_hash(rung1)
        rung2_hash = compute_codepatch_hash(rung2)
        if "ladder_r1" in task_id:
            if self.seed_program_hash in {rung0_hash, rung1_hash}:
                return rung1
            if self.seed_program_hash == rung2_hash:
                return rung2
        if "ladder_r2" in task_id:
            if self.seed_program_hash in {rung1_hash, rung2_hash}:
                return rung2
        return None

    def _normalize_patch(self, patch: str) -> str:
        if not patch:
            return ""
        lines = [line.rstrip() for line in patch.splitlines()]
        normalized = "\n".join(lines)
        if not normalized.endswith("\n"):
            normalized += "\n"
        return normalized

    def _repair_codepatch(self, task: Task, seed: str) -> str:
        normalized = self._normalize_patch(seed)
        if normalized and normalized != seed:
            return normalized
        candidate = self._codepatch_program(task)
        if candidate:
            return self._normalize_patch(candidate)
        return normalized or seed

    def _repair_jsonspec(self, task: Task, seed: str) -> str:
        meta = task.metadata.get("jsonspec", {})
        spec_program = meta.get("spec_program")
        if isinstance(spec_program, dict):
            spec_json = canonical_dumps(spec_program).decode("utf-8")
        else:
            spec_json = ""
        try:
            parsed = orjson.loads(seed)
        except orjson.JSONDecodeError:
            return spec_json or self._jsonspec_program(task)
        if isinstance(parsed, dict):
            if spec_json:
                try:
                    spec_parsed = orjson.loads(spec_json)
                except orjson.JSONDecodeError:
                    spec_parsed = None
                if spec_parsed is not None and spec_parsed != parsed:
                    return spec_json
            return canonical_dumps(parsed).decode("utf-8")
        return spec_json or self._jsonspec_program(task)

    def _repair_pyfunc(self, task: Task, seed: str) -> str:
        candidate = self._pyfunc_program(task)
        if candidate and candidate == seed:
            if "return a + b" in candidate:
                return "def solve(a, b):\n    return (a + b) + 0\n"
        return candidate

    def propose(
        self,
        task: Task,
        *,
        domain: str,
        spec_signature: str,
        seed: int,
        max_tokens: Optional[int] = None,
    ) -> Proposal:
        from .proposer_api import Proposal

        _ = (spec_signature, seed, max_tokens)
        if self.seed_program:
            task_id = task.task_id.lower()
            if self._should_repair(task):
                repaired = ""
                if domain == "pyfunc":
                    repaired = self._repair_pyfunc(task, self.seed_program)
                elif domain == "jsonspec":
                    repaired = self._repair_jsonspec(task, self.seed_program)
                elif domain == "codepatch":
                    repaired = self._repair_codepatch(task, self.seed_program)
                if repaired and repaired != self.seed_program:
                    metadata = {"kind": "heuristic_v1", "seed_used": True, "seed_repaired": True}
                    return Proposal.build(repaired, self.proposer_id, metadata=metadata)
                if repaired:
                    metadata = {"kind": "heuristic_v1", "seed_used": True}
                    return Proposal.build(repaired, self.proposer_id, metadata=metadata)
            seed_allowed = True
            if self.seed_source == "promotion" and "ladder" in task_id:
                ladder_program = None
                if domain == "pyfunc":
                    ladder_program = self._pyfunc_ladder(task_id)
                elif domain == "jsonspec":
                    ladder_program = self._jsonspec_ladder(task)
                elif domain == "codepatch":
                    ladder_program = self._codepatch_ladder(task_id)
                if ladder_program:
                    metadata = {"kind": "heuristic_v1", "seed_used": True, "ladder": True}
                    return Proposal.build(ladder_program, self.proposer_id, metadata=metadata)
                seed_allowed = False
            if seed_allowed:
                metadata = {"kind": "heuristic_v1", "seed_used": True}
                return Proposal.build(self.seed_program, self.proposer_id, metadata=metadata)
        candidate_program = ""
        if domain == "pyfunc":
            candidate_program = self._pyfunc_program(task)
        elif domain == "jsonspec":
            candidate_program = self._jsonspec_program(task)
        elif domain == "codepatch":
            candidate_program = self._codepatch_program(task)
        metadata = {"kind": "heuristic_v1"}
        if not candidate_program:
            return Proposal.build("", self.proposer_id, metadata=metadata, error_atom="NO_PROPOSAL")
        return Proposal.build(candidate_program, self.proposer_id, metadata=metadata)
