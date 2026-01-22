from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import orjson

from .orchestrator.task import Task
from .utils import canonical_dumps, hash_bytes, read_json, to_jsonable, write_jsonl_line


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
            if not isinstance(task_id, str) or not task_id:
                continue
            spec_hash = record.get("spec_hash")
            spec_signature = record.get("spec_signature")
            candidate_program = record.get("candidate_program")
            proposer_id = record.get("proposer_id", "replay")
            metadata = record.get("metadata", {})
            error_atom = record.get("error_atom")
            if isinstance(error_atom, str) and not error_atom:
                error_atom = None
            output = record.get("output")
            if (not isinstance(candidate_program, str) or not candidate_program) and isinstance(
                output, dict
            ):
                if output.get("ok") is True:
                    candidate_program = output.get("program_text")
                    metadata = output.get("metadata", {})
                    error_atom = None
                else:
                    output_error = output.get("error_atom") or error_atom
                    if not isinstance(output_error, str) or not output_error:
                        output_error = "PROPOSER_MISSING"
                    error_atom = output_error
                    candidate_program = ""
            entry: Dict[str, Any] = {
                "candidate_program": (
                    candidate_program if isinstance(candidate_program, str) else ""
                ),
                "proposer_id": proposer_id if isinstance(proposer_id, str) else "replay",
                "metadata": metadata if isinstance(metadata, dict) else {},
            }
            if isinstance(error_atom, str) and error_atom:
                entry["error_atom"] = error_atom
            if isinstance(spec_hash, str) and spec_hash:
                indexed[(task_id, spec_hash)] = entry
            if isinstance(spec_signature, str) and spec_signature:
                indexed[(task_id, spec_signature)] = entry
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
        spec_hash = None
        if hasattr(task, "spec_hash"):
            try:
                spec_hash = task.spec_hash()
            except Exception:
                spec_hash = None
        record = None
        if isinstance(spec_hash, str) and spec_hash:
            record = self.records.get((task.task_id, spec_hash))
        if record is None:
            record = self.records.get((task.task_id, spec_signature))
        if not record:
            return Proposal.build("", "replay", metadata={}, error_atom="PROPOSER_MISSING")
        candidate_program = record.get("candidate_program")
        error_atom = record.get("error_atom")
        if not isinstance(candidate_program, str) or not candidate_program:
            if not isinstance(error_atom, str) or not error_atom:
                error_atom = "PROPOSER_MISSING"
            return Proposal.build("", "replay", metadata={}, error_atom=error_atom)
        proposer_id = record.get("proposer_id", "replay")
        if not isinstance(proposer_id, str):
            proposer_id = "replay"
        metadata = record.get("metadata", {})
        if isinstance(error_atom, str) and error_atom:
            return Proposal.build("", proposer_id, metadata=metadata, error_atom=error_atom)
        return Proposal.build(candidate_program, proposer_id, metadata=metadata)


class SubprocessProposer:
    def __init__(self, command: List[str], timeout_s: float = 15.0) -> None:
        self.command = list(command)
        self.timeout_s = timeout_s
        self.context: Dict[str, Any] = {}
        self.log_path: Optional[Path] = None

    def set_context(self, context: Dict[str, Any], log_path: Optional[Path]) -> None:
        self.context = dict(context)
        self.log_path = Path(log_path) if log_path else None

    def _error(self) -> Proposal:
        return Proposal.build(
            "",
            "subprocess",
            metadata={},
            error_atom="EXCEPTION:PROPOSER_SUBPROCESS_FAILED",
        )

    def _log_call(
        self,
        *,
        task_id: str,
        domain: str,
        spec_hash: str,
        input_payload: Dict[str, Any],
        output_payload: Dict[str, Any],
        error_atom: Optional[str],
        exit_code: Optional[int] = None,
        stderr_tail: Optional[str] = None,
    ) -> None:
        if not self.log_path:
            return
        if "seed_program" in input_payload:
            input_payload = dict(input_payload)
            input_payload["seed_program_hash"] = hash_bytes(
                input_payload["seed_program"].encode("utf-8")
            )
            input_payload.pop("seed_program", None)
        input_hash = hash_bytes(canonical_dumps(input_payload))
        output_hash = hash_bytes(canonical_dumps(output_payload))
        record: Dict[str, Any] = {
            "task_id": task_id,
            "domain": domain,
            "spec_hash": spec_hash,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "error_atom": error_atom or "",
            "output": output_payload,
        }
        if exit_code is not None:
            record["proposer_exit_code"] = int(exit_code)
        if stderr_tail:
            record["proposer_stderr_tail"] = stderr_tail
        write_jsonl_line(self.log_path, record)

    def propose(
        self,
        task: Task,
        *,
        domain: str,
        spec_signature: str,
        seed: int,
        max_tokens: Optional[int] = None,
    ) -> Proposal:
        _ = (seed, max_tokens, spec_signature)
        suite_id = str(self.context.get("suite_id", ""))
        spec_hash = str(self.context.get("spec_hash") or task.spec_hash())
        task_spec = self.context.get("task_spec") or task.open_view()
        prior_best_hash = str(self.context.get("prior_best_hash", ""))
        dataset_paths = self.context.get("dataset_paths", {})
        payload = {
            "suite_id": suite_id,
            "task_id": task.task_id,
            "domain": domain,
            "spec_hash": spec_hash,
            "task_spec": task_spec,
            "prior_best_hash": prior_best_hash,
        }
        seed_program = self.context.get("seed_program")
        seed_program_hash = self.context.get("seed_program_hash")
        if isinstance(seed_program, str) and seed_program:
            payload["seed_program"] = seed_program
        if isinstance(seed_program_hash, str) and seed_program_hash:
            payload["seed_program_hash"] = seed_program_hash
        if isinstance(dataset_paths, dict) and dataset_paths:
            payload["dataset_paths"] = dict(dataset_paths)
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                env = {
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONHASHSEED": "0",
                    "LC_ALL": "C",
                    "LANG": "C",
                }
                result = subprocess.run(
                    self.command,
                    input=canonical_dumps(payload),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=self.timeout_s,
                    check=False,
                    cwd=tmp_dir,
                    env=env,
                )
        except subprocess.TimeoutExpired as exc:
            error_atom = "EXCEPTION:PROPOSER_SUBPROCESS_FAILED"
            output_payload = {"ok": False, "error_atom": error_atom}
            stderr_tail = (exc.stderr or b"")[-4000:].decode("utf-8", errors="replace")
            if stderr_tail:
                stderr_tail = stderr_tail.replace("\r\n", "\n").replace("\r", "\n")
            self._log_call(
                task_id=task.task_id,
                domain=domain,
                spec_hash=spec_hash,
                input_payload=payload,
                output_payload=output_payload,
                error_atom=error_atom,
                stderr_tail=stderr_tail,
            )
            return self._error()
        except Exception:
            error_atom = "EXCEPTION:PROPOSER_SUBPROCESS_FAILED"
            output_payload = {"ok": False, "error_atom": error_atom}
            self._log_call(
                task_id=task.task_id,
                domain=domain,
                spec_hash=spec_hash,
                input_payload=payload,
                output_payload=output_payload,
                error_atom=error_atom,
            )
            return self._error()
        if result.returncode != 0:
            error_atom = "EXCEPTION:PROPOSER_SUBPROCESS_FAILED"
            output_payload = {"ok": False, "error_atom": error_atom}
            stderr_tail = (result.stderr or b"")[-4000:].decode("utf-8", errors="replace")
            if stderr_tail:
                stderr_tail = stderr_tail.replace("\r\n", "\n").replace("\r", "\n")
            self._log_call(
                task_id=task.task_id,
                domain=domain,
                spec_hash=spec_hash,
                input_payload=payload,
                output_payload=output_payload,
                error_atom=error_atom,
                exit_code=result.returncode,
                stderr_tail=stderr_tail,
            )
            return self._error()
        try:
            output = orjson.loads(result.stdout or b"{}")
        except orjson.JSONDecodeError:
            error_atom = "EXCEPTION:PROPOSER_SUBPROCESS_FAILED"
            output_payload = {"ok": False, "error_atom": error_atom}
            stderr_tail = (result.stderr or b"")[-4000:].decode("utf-8", errors="replace")
            if stderr_tail:
                stderr_tail = stderr_tail.replace("\r\n", "\n").replace("\r", "\n")
            self._log_call(
                task_id=task.task_id,
                domain=domain,
                spec_hash=spec_hash,
                input_payload=payload,
                output_payload=output_payload,
                error_atom=error_atom,
                exit_code=result.returncode,
                stderr_tail=stderr_tail,
            )
            return self._error()
        if not isinstance(output, dict):
            error_atom = "EXCEPTION:PROPOSER_SUBPROCESS_FAILED"
            output_payload = {"ok": False, "error_atom": error_atom}
            stderr_tail = (result.stderr or b"")[-4000:].decode("utf-8", errors="replace")
            if stderr_tail:
                stderr_tail = stderr_tail.replace("\r\n", "\n").replace("\r", "\n")
            self._log_call(
                task_id=task.task_id,
                domain=domain,
                spec_hash=spec_hash,
                input_payload=payload,
                output_payload=output_payload,
                error_atom=error_atom,
                exit_code=result.returncode,
                stderr_tail=stderr_tail,
            )
            return self._error()
        if output.get("ok") is not True:
            raw_error_atom = output.get("error_atom")
            if isinstance(raw_error_atom, str) and raw_error_atom:
                error_atom = raw_error_atom
            else:
                error_atom = "EXCEPTION:PROPOSER_SUBPROCESS_FAILED"
            output_payload = {
                "ok": False,
                "error_atom": error_atom,
            }
            stderr_tail = (result.stderr or b"")[-4000:].decode("utf-8", errors="replace")
            if stderr_tail:
                stderr_tail = stderr_tail.replace("\r\n", "\n").replace("\r", "\n")
            self._log_call(
                task_id=task.task_id,
                domain=domain,
                spec_hash=spec_hash,
                input_payload=payload,
                output_payload=output_payload,
                error_atom=error_atom,
                exit_code=result.returncode,
                stderr_tail=stderr_tail,
            )
            return Proposal.build("", "subprocess", metadata={}, error_atom=error_atom)
        candidate_program = output.get("program_text")
        if not isinstance(candidate_program, str):
            error_atom = "EXCEPTION:PROPOSER_SUBPROCESS_FAILED"
            output_payload = {"ok": False, "error_atom": error_atom}
            stderr_tail = (result.stderr or b"")[-4000:].decode("utf-8", errors="replace")
            if stderr_tail:
                stderr_tail = stderr_tail.replace("\r\n", "\n").replace("\r", "\n")
            self._log_call(
                task_id=task.task_id,
                domain=domain,
                spec_hash=spec_hash,
                input_payload=payload,
                output_payload=output_payload,
                error_atom=error_atom,
                exit_code=result.returncode,
                stderr_tail=stderr_tail,
            )
            return self._error()
        metadata = output.get("metadata", {})
        output_payload = {
            "ok": True,
            "program_text": candidate_program,
            "metadata": to_jsonable(metadata),
        }
        self._log_call(
            task_id=task.task_id,
            domain=domain,
            spec_hash=spec_hash,
            input_payload=payload,
            output_payload=output_payload,
            error_atom=None,
        )
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
