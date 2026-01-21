from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import orjson

from .orchestrator.task import Task
from .utils import canonical_dumps, hash_bytes, read_json, to_jsonable


def _normalize_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    safe = to_jsonable(metadata)
    return safe if isinstance(safe, dict) else {}


def _proposal_payload(
    candidate_program: str,
    proposer_id: str,
    metadata: Dict[str, Any],
    error_atom: Optional[str],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "candidate_program": candidate_program,
        "proposer_id": proposer_id,
        "metadata": metadata,
    }
    if error_atom:
        payload["error_atom"] = error_atom
    return payload


@dataclass(frozen=True)
class Proposal:
    candidate_program: str
    proposer_id: str
    proposal_hash: str
    metadata: Dict[str, Any]
    error_atom: Optional[str] = None

    @classmethod
    def build(
        cls,
        candidate_program: str,
        proposer_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        error_atom: Optional[str] = None,
    ) -> "Proposal":
        safe_metadata = _normalize_metadata(metadata)
        payload = _proposal_payload(candidate_program, proposer_id, safe_metadata, error_atom)
        proposal_hash = hash_bytes(canonical_dumps(payload))
        return cls(
            candidate_program=candidate_program,
            proposer_id=proposer_id,
            proposal_hash=proposal_hash,
            metadata=safe_metadata,
            error_atom=error_atom,
        )

    def with_error(self, error_atom: str) -> "Proposal":
        return Proposal.build(
            candidate_program=self.candidate_program,
            proposer_id=self.proposer_id,
            metadata=self.metadata,
            error_atom=error_atom,
        )

    def to_record(self) -> Dict[str, Any]:
        record = _proposal_payload(
            self.candidate_program, self.proposer_id, self.metadata, self.error_atom
        )
        record["proposal_hash"] = self.proposal_hash
        return record


class BaseProposer(Protocol):
    def propose(
        self,
        task: Task,
        *,
        domain: str,
        spec_signature: str,
        seed: int,
        max_tokens: Optional[int] = None,
    ) -> Proposal:
        ...


class StaticProposer:
    def __init__(self, candidate_program: str, proposer_id: str = "static") -> None:
        self.candidate_program = candidate_program
        self.proposer_id = proposer_id

    def propose(
        self,
        task: Task,
        *,
        domain: str,
        spec_signature: str,
        seed: int,
        max_tokens: Optional[int] = None,
    ) -> Proposal:
        _ = (task, domain, spec_signature, seed, max_tokens)
        return Proposal.build(self.candidate_program, self.proposer_id, metadata={})


class ReplayProposer:
    def __init__(self, replay_path: Path) -> None:
        self.replay_path = Path(replay_path)
        self.records = self._load_records(self.replay_path)

    def _load_records(self, path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
        if not path.exists():
            return {}
        records: List[Dict[str, Any]] = []
        if path.suffix == ".jsonl":
            for line in path.read_bytes().splitlines():
                if not line:
                    continue
                try:
                    record = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
        else:
            data = read_json(path)
            if isinstance(data, list):
                records = [item for item in data if isinstance(item, dict)]
            elif isinstance(data, dict):
                if isinstance(data.get("proposals"), list):
                    records = [item for item in data["proposals"] if isinstance(item, dict)]
                elif isinstance(data.get("items"), list):
                    records = [item for item in data["items"] if isinstance(item, dict)]
                elif "candidate_program" in data:
                    records = [data]
        indexed: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for record in records:
            task_id = record.get("task_id")
            spec_signature = record.get("spec_signature")
            if isinstance(task_id, str) and isinstance(spec_signature, str):
                indexed[(task_id, spec_signature)] = record
        return indexed

    def propose(
        self,
        task: Task,
        *,
        domain: str,
        spec_signature: str,
        seed: int,
        max_tokens: Optional[int] = None,
    ) -> Proposal:
        _ = (domain, seed, max_tokens)
        record = self.records.get((task.task_id, spec_signature))
        if not record:
            return Proposal.build("", "replay", metadata={}, error_atom="PROPOSER_MISSING")
        candidate_program = record.get("candidate_program")
        if not isinstance(candidate_program, str):
            return Proposal.build("", "replay", metadata={}, error_atom="PROPOSER_MISSING")
        proposer_id = record.get("proposer_id", "replay")
        if not isinstance(proposer_id, str):
            proposer_id = "replay"
        metadata = record.get("metadata", {})
        return Proposal.build(candidate_program, proposer_id, metadata=metadata)


class SubprocessProposer:
    def __init__(self, command: List[str], timeout_s: float = 15.0) -> None:
        self.command = list(command)
        self.timeout_s = timeout_s

    def _error(self, reason: str) -> Proposal:
        return Proposal.build(
            "",
            "subprocess",
            metadata={"reason": reason},
            error_atom=f"PROPOSER_ERROR:{reason}",
        )

    def propose(
        self,
        task: Task,
        *,
        domain: str,
        spec_signature: str,
        seed: int,
        max_tokens: Optional[int] = None,
    ) -> Proposal:
        payload = {
            "task": task.model_dump(),
            "domain": domain,
            "spec_signature": spec_signature,
            "seed": seed,
            "max_tokens": max_tokens,
        }
        try:
            result = subprocess.run(
                self.command,
                input=canonical_dumps(payload),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return self._error("timeout")
        except Exception:
            return self._error("spawn_failed")
        if result.returncode != 0:
            return self._error("nonzero")
        try:
            output = orjson.loads(result.stdout or b"{}")
        except orjson.JSONDecodeError:
            return self._error("bad_json")
        if not isinstance(output, dict):
            return self._error("bad_json")
        candidate_program = output.get("candidate_program")
        if not isinstance(candidate_program, str):
            return self._error("missing_candidate")
        metadata = output.get("metadata", {})
        return Proposal.build(candidate_program, "subprocess", metadata=metadata)


class RetrievalProposer:
    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = Path(dataset_path)
        self.dataset_hash = (
            hash_bytes(self.dataset_path.read_bytes()) if self.dataset_path.exists() else ""
        )
        self.index = self._load_index(self.dataset_path)

    def _load_index(self, path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
        if not path.exists():
            return {}
        records: List[Dict[str, Any]] = []
        if path.suffix == ".jsonl":
            for line in path.read_bytes().splitlines():
                if not line:
                    continue
                try:
                    record = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
        else:
            data = read_json(path)
            if isinstance(data, list):
                records = [item for item in data if isinstance(item, dict)]
        index: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for record in records:
            if record.get("verdict") != "PASS":
                continue
            domain = record.get("domain")
            spec_key = record.get("spec_hash") or record.get("spec_signature")
            candidate_program = record.get("candidate_program")
            if not isinstance(domain, str) or not isinstance(spec_key, str):
                continue
            if not isinstance(candidate_program, str) or not candidate_program:
                continue
            score = record.get("controller_score_scaled", 0)
            score_value = int(score) if isinstance(score, (int, float)) else 0
            program_hash = record.get("proposed_program_hash") or record.get("program_hash") or ""
            program_hash = str(program_hash)
            key = (domain, spec_key)
            existing = index.get(key)
            if existing is None:
                index[key] = {
                    "candidate_program": candidate_program,
                    "score_scaled": score_value,
                    "program_hash": program_hash,
                }
                continue
            existing_score = int(existing.get("score_scaled", 0))
            existing_hash = str(existing.get("program_hash", ""))
            if score_value > existing_score:
                index[key] = {
                    "candidate_program": candidate_program,
                    "score_scaled": score_value,
                    "program_hash": program_hash,
                }
            elif score_value == existing_score and program_hash and program_hash < existing_hash:
                index[key] = {
                    "candidate_program": candidate_program,
                    "score_scaled": score_value,
                    "program_hash": program_hash,
                }
        return index

    def _fallback_program(self, task: Task) -> str:
        if task.task_type == "pyfunc":
            return str(task.metadata.get("pyfunc", {}).get("candidate_program", ""))
        if task.task_type == "codepatch":
            return str(task.metadata.get("codepatch", {}).get("candidate_patch", ""))
        if task.task_type == "jsonspec":
            candidate_spec = task.metadata.get("jsonspec", {}).get("candidate_program")
            if isinstance(candidate_spec, dict):
                return canonical_dumps(candidate_spec).decode("utf-8")
            if isinstance(candidate_spec, str):
                return candidate_spec
        return ""

    def propose(
        self,
        task: Task,
        *,
        domain: str,
        spec_signature: str,
        seed: int,
        max_tokens: Optional[int] = None,
    ) -> Proposal:
        _ = (seed, max_tokens)
        spec_key = getattr(task, "spec_hash", None)
        if callable(spec_key):
            spec_key = spec_key()
        if not isinstance(spec_key, str) or not spec_key:
            spec_key = spec_signature
        match = self.index.get((domain, spec_key))
        if match:
            candidate_program = match.get("candidate_program", "")
            metadata = {
                "kind": "retrieval",
                "dataset_hash": self.dataset_hash,
                "match_type": "exact",
            }
            return Proposal.build(
                candidate_program, "retrieval", metadata=metadata, error_atom=None
            )
        candidate_program = self._fallback_program(task)
        metadata = {
            "kind": "retrieval",
            "dataset_hash": self.dataset_hash,
            "match_type": "none",
        }
        if not candidate_program:
            return Proposal.build(
                "", "retrieval", metadata=metadata, error_atom="PROPOSER_MISSING"
            )
        return Proposal.build(candidate_program, "retrieval", metadata=metadata)
