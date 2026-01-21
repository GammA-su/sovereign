from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

from .orchestrator.task import Task
from .utils import canonical_dumps


@dataclass
class CoverageLedger:
    per_domain: Dict[str, Dict[str, int]] = field(default_factory=dict)
    totals: Dict[str, int] = field(default_factory=dict)
    families_tried: list[str] = field(default_factory=list)
    metamorphic_families_run: list[str] = field(default_factory=list)
    failures_by_atom: Dict[str, int] = field(default_factory=dict)
    sealed_families_used: Optional[list[str]] = None
    _seen_candidates: Dict[str, set[str]] = field(default_factory=dict, init=False, repr=False)
    _seen_atoms_global: set[str] = field(default_factory=set, init=False, repr=False)
    _seen_atoms_by_domain: Dict[str, set[str]] = field(
        default_factory=dict, init=False, repr=False
    )
    _total_attempts: int = field(default=0, init=False, repr=False)
    _total_new_atoms: int = field(default=0, init=False, repr=False)

    def update_from_episode(
        self,
        *,
        ucr: Mapping[str, Any],
        verifier: Sequence[Mapping[str, Any]],
        breaker: Mapping[str, Any],
        task: Optional[Task] = None,
    ) -> int:
        inputs = ucr.get("inputs", {}) if isinstance(ucr, dict) else {}
        domain = ""
        if isinstance(inputs, dict):
            domain = str(inputs.get("task_type", ""))
            family = inputs.get("family")
            if isinstance(family, str):
                self._add_unique(self.families_tried, family)
        else:
            domain = ""
        if not domain and task is not None:
            domain = task.task_type

        attempts = int(breaker.get("attempts", 0)) if isinstance(breaker, dict) else 0
        self._total_attempts += attempts

        if domain:
            entry = self.per_domain.setdefault(
                domain,
                {
                    "attempts": 0,
                    "atoms_total": 0,
                    "new_atoms": 0,
                },
            )
            entry["attempts"] += attempts

            program_hash = ""
            hashes = ucr.get("hashes", {}) if isinstance(ucr, dict) else {}
            if isinstance(hashes, dict):
                program_hash = str(hashes.get("program_hash", ""))
            if program_hash:
                seen = self._seen_candidates.setdefault(domain, set())
                if program_hash in seen:
                    pass
                else:
                    seen.add(program_hash)

        episode_atoms = self._episode_atoms(domain, verifier, breaker)
        if domain:
            domain_seen = self._seen_atoms_by_domain.setdefault(domain, set())
            new_domain_atoms = episode_atoms - domain_seen
            entry = self.per_domain.get(domain, {})
            if isinstance(entry, dict):
                entry["new_atoms"] = entry.get("new_atoms", 0) + len(new_domain_atoms)
                domain_seen.update(episode_atoms)
                entry["atoms_total"] = len(domain_seen)

        new_atoms = episode_atoms - self._seen_atoms_global
        self._seen_atoms_global.update(episode_atoms)
        self._total_new_atoms += len(new_atoms)

        for verdict in verifier:
            if not isinstance(verdict, dict):
                continue
            for atom in verdict.get("failure_atoms", []) or []:
                if isinstance(atom, str) and atom:
                    self.failures_by_atom[atom] = self.failures_by_atom.get(atom, 0) + 1
            for family in verdict.get("metamorphic_families", []) or []:
                if isinstance(family, str) and family:
                    self._add_unique(self.metamorphic_families_run, family)

        if isinstance(breaker, dict):
            for atom in breaker.get("failure_atoms", []) or []:
                if isinstance(atom, str) and atom:
                    self.failures_by_atom[atom] = self.failures_by_atom.get(atom, 0) + 1

        if task is not None and task.sealed and task.sealed.withheld_families:
            if self.sealed_families_used is None:
                self.sealed_families_used = []
            for family in task.sealed.withheld_families:
                if isinstance(family, str) and family:
                    self._add_unique(self.sealed_families_used, family)
        return len(new_atoms)

    def to_payload(self) -> Dict[str, Any]:
        per_domain_sorted = {
            domain: dict(self.per_domain[domain]) for domain in sorted(self.per_domain.keys())
        }
        payload: Dict[str, Any] = {
            "per_domain": per_domain_sorted,
            "totals": {
                "attempts": self._total_attempts,
                "atoms_total": len(self._seen_atoms_global),
                "new_atoms": self._total_new_atoms,
            },
            "families_tried": sorted(self.families_tried),
            "metamorphic_families_run": sorted(self.metamorphic_families_run),
            "failures_by_atom": {
                atom: self.failures_by_atom[atom] for atom in sorted(self.failures_by_atom.keys())
            },
        }
        if self.sealed_families_used:
            payload["sealed_families_used"] = sorted(self.sealed_families_used)
        return payload

    def to_json(self) -> bytes:
        data = canonical_dumps(self.to_payload())
        if not data.endswith(b"\n"):
            data += b"\n"
        return data

    @staticmethod
    def _add_unique(values: list[str], value: str) -> None:
        if value not in values:
            values.append(value)

    @staticmethod
    def _episode_atoms(
        domain: str,
        verifier: Sequence[Mapping[str, Any]],
        breaker: Mapping[str, Any],
    ) -> set[str]:
        prefix = f"{domain}:" if domain else ""
        atoms: set[str] = set()
        for verdict in verifier:
            if not isinstance(verdict, dict):
                continue
            for atom in verdict.get("failure_atoms", []) or []:
                if isinstance(atom, str) and atom:
                    atoms.add(f"{prefix}fail:{atom}")
            for family in verdict.get("metamorphic_families", []) or []:
                if isinstance(family, str) and family:
                    atoms.add(f"{prefix}meta:{family}")
        if isinstance(breaker, dict):
            for atom in breaker.get("failure_atoms", []) or []:
                if isinstance(atom, str) and atom:
                    atoms.add(f"{prefix}breaker:{atom}")
        return atoms
