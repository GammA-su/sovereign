from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import orjson
from pydantic import BaseModel, Field, model_validator

from .utils import CANONICALIZATION, HASH_ALGORITHM, stable_hash


class HashableModel(BaseModel):
    schema_version: str = "v1"
    canonicalization: str = CANONICALIZATION
    hash_algorithm: str = HASH_ALGORITHM
    hash_inputs: List[str] = Field(default_factory=list)

    def hash_payload(self) -> Dict[str, Any]:
        data = self.model_dump()
        keys = self.hash_inputs or [key for key in data.keys() if key != "hash_inputs"]
        return {key: data[key] for key in keys if key in data}

    def stable_hash(self) -> str:
        return stable_hash(self.hash_payload())


class VerifierVerdict(HashableModel):
    verdict: Literal["PASS", "FAIL", "BORDERLINE"]
    failure_atoms: List[str] = Field(default_factory=list)
    domain: str
    tier: str
    bounds: Dict[str, Any] = Field(default_factory=dict)
    soundness_grade: Literal["CERT", "BOUNDED", "HEURISTIC"]
    metamorphic_families: List[str] = Field(default_factory=list)
    cost: Dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _set_hash_inputs(self) -> "VerifierVerdict":
        if not self.hash_inputs:
            self.hash_inputs = [
                "schema_version",
                "canonicalization",
                "hash_algorithm",
                "verdict",
                "failure_atoms",
                "domain",
                "tier",
                "bounds",
                "soundness_grade",
                "metamorphic_families",
                "cost",
            ]
        return self


class BreakerKPI(HashableModel):
    CDR: float
    TMR: float
    NOVN: float
    WFHR: float
    window: Dict[str, int] = Field(default_factory=dict)
    budget: Dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _set_hash_inputs(self) -> "BreakerKPI":
        if not self.hash_inputs:
            self.hash_inputs = [
                "schema_version",
                "canonicalization",
                "hash_algorithm",
                "CDR",
                "TMR",
                "NOVN",
                "WFHR",
                "window",
                "budget",
            ]
        return self


class BGRevisionOp(HashableModel):
    op: Literal[
        "ASSERT",
        "SUPERSEDE",
        "PATCH",
        "RETRACT",
        "DECLARE_CONFLICT",
        "RESOLVE_CONFLICT",
    ]
    witness_id: str
    node_id: Optional[str] = None
    old_id: Optional[str] = None
    new_id: Optional[str] = None
    diff: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    conflict_id: Optional[str] = None
    conflict_set: List[str] = Field(default_factory=list)
    chosen_id: Optional[str] = None
    disruption_cost: Optional[int] = None
    payload: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_op(self) -> "BGRevisionOp":
        if self.op == "ASSERT" and not self.node_id:
            raise ValueError("ASSERT requires node_id")
        if self.op == "SUPERSEDE" and not (self.old_id and self.new_id):
            raise ValueError("SUPERSEDE requires old_id and new_id")
        if self.op == "PATCH" and not (self.node_id and self.diff):
            raise ValueError("PATCH requires node_id and diff")
        if self.op == "RETRACT" and not (self.node_id and self.reason):
            raise ValueError("RETRACT requires node_id and reason")
        if self.op == "DECLARE_CONFLICT" and not (self.conflict_id and self.conflict_set):
            raise ValueError("DECLARE_CONFLICT requires conflict_id and conflict_set")
        if self.op == "RESOLVE_CONFLICT" and not (
            self.conflict_id and self.chosen_id is not None and self.disruption_cost is not None
        ):
            raise ValueError("RESOLVE_CONFLICT requires conflict_id, chosen_id, disruption_cost")
        if not self.hash_inputs:
            self.hash_inputs = [
                "schema_version",
                "canonicalization",
                "hash_algorithm",
                "op",
                "witness_id",
                "node_id",
                "old_id",
                "new_id",
                "diff",
                "reason",
                "conflict_id",
                "conflict_set",
                "chosen_id",
                "disruption_cost",
                "payload",
            ]
        return self


class SkillInterface(HashableModel):
    name: str
    version: str
    io_types: Dict[str, str]
    verifier_requirements: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _set_hash_inputs(self) -> "SkillInterface":
        if not self.hash_inputs:
            self.hash_inputs = [
                "schema_version",
                "canonicalization",
                "hash_algorithm",
                "name",
                "version",
                "io_types",
                "verifier_requirements",
            ]
        return self


class LanguagePatch(HashableModel):
    patch_type: str
    scope: str
    conservativity_evidence: str
    rollback_plan: str
    impacted_skills: List[str] = Field(default_factory=list)
    expected_bvps_effort_delta: float = 0.0

    @model_validator(mode="after")
    def _set_hash_inputs(self) -> "LanguagePatch":
        if not self.hash_inputs:
            self.hash_inputs = [
                "schema_version",
                "canonicalization",
                "hash_algorithm",
                "patch_type",
                "scope",
                "conservativity_evidence",
                "rollback_plan",
                "impacted_skills",
                "expected_bvps_effort_delta",
            ]
        return self


class ArtifactRecord(HashableModel):
    path: str
    content_hash: str
    bytes: int
    kind: str

    @model_validator(mode="after")
    def _set_hash_inputs(self) -> "ArtifactRecord":
        if not self.hash_inputs:
            self.hash_inputs = [
                "schema_version",
                "canonicalization",
                "hash_algorithm",
                "path",
                "content_hash",
                "bytes",
                "kind",
            ]
        return self


class WitnessPacket(HashableModel):
    witness_id: str
    overall_verdict: Literal["PASS", "FAIL"]
    failure_reason: str = ""
    artifacts: List[ArtifactRecord] = Field(default_factory=list)
    verifier_verdicts: List[VerifierVerdict] = Field(default_factory=list)
    breaker_evidence: Dict[str, Any] = Field(default_factory=dict)
    hashes: Dict[str, str] = Field(default_factory=dict)
    coverage: Dict[str, Any] = Field(default_factory=dict)
    cost_report: Dict[str, int] = Field(default_factory=dict)
    policy_versions: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _set_hash_inputs(self) -> "WitnessPacket":
        if not self.hash_inputs:
            self.hash_inputs = [
                "schema_version",
                "canonicalization",
                "hash_algorithm",
                "witness_id",
                "overall_verdict",
                "failure_reason",
                "artifacts",
                "verifier_verdicts",
                "breaker_evidence",
                "hashes",
                "coverage",
                "cost_report",
                "policy_versions",
            ]
        return self


class UCR(HashableModel):
    run_id: str
    task_id: str
    inputs: Dict[str, Any]
    interpretations: List[Dict[str, Any]]
    chosen_interpretation: Dict[str, Any]
    solver_trace: List[str] = Field(default_factory=list)
    costs: Dict[str, int] = Field(default_factory=dict)
    hashes: Dict[str, str] = Field(default_factory=dict)
    bg_context: Dict[str, Any] = Field(default_factory=dict)
    active_view_hash: str = ""
    run_metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _set_hash_inputs(self) -> "UCR":
        if not self.hash_inputs:
            self.hash_inputs = [
                "schema_version",
                "canonicalization",
                "hash_algorithm",
                "run_id",
                "task_id",
                "inputs",
                "interpretations",
                "chosen_interpretation",
                "solver_trace",
                "costs",
                "hashes",
                "bg_context",
                "active_view_hash",
                "run_metadata",
            ]
        return self


def export_schemas(output_dir: str) -> None:
    from pathlib import Path

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    models = [
        UCR,
        WitnessPacket,
        VerifierVerdict,
        BreakerKPI,
        BGRevisionOp,
        SkillInterface,
        LanguagePatch,
        ArtifactRecord,
    ]
    for model in models:
        schema = model.model_json_schema()  # type: ignore[attr-defined]
        path = output / f"{model.__name__}.schema.json"
        path.write_bytes(orjson.dumps(schema, option=orjson.OPT_SORT_KEYS))
