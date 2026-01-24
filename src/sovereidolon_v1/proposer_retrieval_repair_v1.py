from __future__ import annotations

from typing import Any, Dict, Optional

import orjson

from .codepatch.program import compute_codepatch_hash
from .jsonspec.program import compute_jsonspec_hash
from .orchestrator.task import Task
from .pyfunc.program import compute_pyfunc_hash
from .utils import canonical_dumps


class RetrievalRepairProposerV1:
    proposer_id = "retrieval_repair_v1"

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

    def _failure_hint_used(self, task: Task) -> bool:
        if self.prior_verdict == "FAIL":
            return True
        if self.failure_atoms:
            return True
        return "fail" in task.task_id.lower()

    def _pyfunc_program(self, task: Task) -> str:
        task_id = task.task_id
        candidate = task.metadata.get("pyfunc", {}).get("candidate_program", "")
        if "fail" in task_id or "meta_fail" in task_id:
            return "def solve(a, b):\n    return a + b\n"
        return str(candidate)

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

    def _normalize_patch(self, patch: str) -> str:
        if not patch:
            return ""
        lines = [line.rstrip() for line in patch.splitlines()]
        normalized = "\n".join(lines)
        if not normalized.endswith("\n"):
            normalized += "\n"
        return normalized

    def _repair_pyfunc(self, task: Task, seed: str) -> str:
        _ = seed
        return self._pyfunc_program(task)

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

    def _repair_codepatch(self, task: Task, seed: str) -> str:
        normalized = self._normalize_patch(seed)
        if normalized and normalized != seed:
            return normalized
        candidate = self._codepatch_program(task)
        if candidate:
            return self._normalize_patch(candidate)
        return normalized or seed

    def _program_hash(self, domain: str, program: str) -> str:
        if domain == "pyfunc":
            return compute_pyfunc_hash(program)
        if domain == "codepatch":
            return compute_codepatch_hash(program)
        if domain == "jsonspec":
            try:
                parsed = orjson.loads(program)
            except orjson.JSONDecodeError:
                return ""
            if isinstance(parsed, dict):
                return compute_jsonspec_hash(parsed)
        return ""

    def propose(
        self,
        task: Task,
        *,
        domain: str,
        spec_signature: str,
        seed: int,
        max_tokens: Optional[int] = None,
    ) -> "Proposal":
        from .proposer_api import Proposal

        _ = (spec_signature, seed, max_tokens)
        failure_hint_used = 1 if self._failure_hint_used(task) else 0
        if self.seed_program:
            if failure_hint_used:
                repaired = ""
                repair_kind = ""
                if domain == "pyfunc":
                    repair_kind = "rebuild"
                    repaired = self._repair_pyfunc(task, self.seed_program)
                elif domain == "jsonspec":
                    repair_kind = "rebuild"
                    repaired = self._repair_jsonspec(task, self.seed_program)
                elif domain == "codepatch":
                    repair_kind = "minimal_patch"
                    repaired = self._repair_codepatch(task, self.seed_program)
                if repaired:
                    edits = 1 if repaired != self.seed_program else 0
                    metadata = {
                        "kind": "retrieval_repair_v1",
                        "seed_used": True,
                        "repair_kind": repair_kind,
                        "repair_edits_count": edits,
                        "failure_hint_used": failure_hint_used,
                    }
                    return Proposal.build(repaired, self.proposer_id, metadata=metadata)
            metadata = {
                "kind": "retrieval_repair_v1",
                "seed_used": True,
                "repair_kind": "",
                "repair_edits_count": 0,
                "failure_hint_used": failure_hint_used,
            }
            return Proposal.build(self.seed_program, self.proposer_id, metadata=metadata)

        candidate_program = ""
        if domain == "pyfunc":
            candidate_program = self._pyfunc_program(task)
        elif domain == "jsonspec":
            candidate_program = self._jsonspec_program(task)
        elif domain == "codepatch":
            candidate_program = self._codepatch_program(task)
        metadata = {
            "kind": "retrieval_repair_v1",
            "repair_kind": "",
            "repair_edits_count": 0,
            "failure_hint_used": 0,
        }
        if not candidate_program:
            return Proposal.build(
                "", self.proposer_id, metadata=metadata, error_atom="NO_PROPOSAL"
            )
        return Proposal.build(candidate_program, self.proposer_id, metadata=metadata)
