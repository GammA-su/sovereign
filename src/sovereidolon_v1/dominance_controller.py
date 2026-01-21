from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

from .utils import stable_hash

KPI_KEYS = ("CDR", "TMR", "NOVN", "WFHR")
SCORE_KPI_KEYS = ("CDR", "NOVN", "WFHR", "TMR")
SCORE_COST_KEYS = ("breaker_attempts", "meta_cases")
SCORE_SCALE = 1_000_000
TIMING_SUFFIXES = ("_ns",)


@dataclass(frozen=True)
class ControllerVerdict:
    decision: str
    reason_atoms: list[str]
    policy_id: str
    policy_hash: str
    snapshots: Dict[str, Dict[str, float | int]]
    score: Dict[str, Any]
    score_key: list[int]
    policy_version: str = "v1"
    score_scaled: int | None = None
    allow_admit: bool | None = None

    def to_record(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "decision": self.decision,
            "reason_atoms": list(self.reason_atoms),
            "policy_id": self.policy_id,
            "policy_hash": self.policy_hash,
            "policy_version": self.policy_version,
            "snapshots": self.snapshots,
            "score": self.score,
            "score_key": list(self.score_key),
        }
        if self.score_scaled is not None:
            record["score_scaled"] = int(self.score_scaled)
        if self.allow_admit is not None:
            record["allow_admit"] = bool(self.allow_admit)
        return record


def _domain_policy_template() -> Dict[str, Any]:
    return {
        "kpi_minima": {
            "CDR": 0.0,
            "TMR": 0.0,
            "NOVN": 0.0,
            "WFHR": 0.0,
        },
        "cost_ceilings": {
            "breaker_attempts": 1_000_000,
            "meta_cases": 1_000_000,
        },
    }


def dominance_v1_policy() -> Dict[str, Any]:
    return {
        "policy_version": "v1",
        "policy_id": "dominance_v1",
        "default": _domain_policy_template(),
        "domains": {
            "pyfunc": _domain_policy_template(),
            "codepatch": _domain_policy_template(),
            "jsonspec": _domain_policy_template(),
        },
    }


def dominance_v2_policy() -> Dict[str, Any]:
    return {
        "policy_version": "v2",
        "domain_cost_ceiling_ns": {
            "pyfunc": 2_000_000,
            "codepatch": 15_000_000,
            "jsonspec": 5_000_000,
        },
        "sealed_weight": 2.0,
        "coverage_reward": {
            "enabled": True,
            "weight": 0.1,
            "atoms_scope": "episode",
        },
    }


def dominance_v3_policy() -> Dict[str, Any]:
    return {
        "policy_version": "v3",
        "domain_cost_ceiling_ns": {
            "pyfunc": 2_000_000,
            "codepatch": 15_000_000,
            "jsonspec": 5_000_000,
        },
        "default_cost_ceiling_ns": 10_000_000,
        "min_coverage_gain": 1,
        "min_score_scaled": 1,
    }


class DominanceController:
    def __init__(self, policy: Mapping[str, Any]) -> None:
        self.policy = dict(policy)
        self.policy_version = str(self.policy.get("policy_version", "v1"))
        self.policy_hash = stable_hash(self.policy)
        if self.policy_version == "v2":
            self.policy_id = self._policy_fingerprint(self.policy, self.policy_version)
        else:
            self.policy_id = str(self.policy.get("policy_id", ""))

    def _policy_for(self, domain: str) -> Dict[str, Any]:
        domains = self.policy.get("domains", {})
        default = self.policy.get("default", {})
        if isinstance(domains, dict):
            domain_policy = domains.get(domain)
            if isinstance(domain_policy, dict):
                return domain_policy
        if isinstance(default, dict):
            return default
        return {}

    def _policy_fingerprint(self, policy: Mapping[str, Any], policy_version: str) -> str:
        payload = {
            "policy_version": policy_version,
            "policy": {
                key: policy[key]
                for key in sorted(policy.keys())
                if key not in {"policy_id", "policy_version"}
            },
        }
        return stable_hash(payload)

    def _kpi_failures(
        self, breaker_kpi: Mapping[str, Any], minima: Mapping[str, Any]
    ) -> list[str]:
        if breaker_kpi.get("status") == "SKIPPED" or breaker_kpi.get("skipped"):
            return []
        failures: list[str] = []
        for key in KPI_KEYS:
            minimum = float(minima.get(key, 0.0))
            value = float(breaker_kpi.get(key, 0.0))
            if value < minimum:
                failures.append(f"KPI_FAIL:{key}")
        return failures

    def _cost_failures(
        self, costs: Mapping[str, Any], ceilings: Mapping[str, Any]
    ) -> list[str]:
        failures: list[str] = []
        for key in sorted(ceilings.keys()):
            value = costs.get(key, 0)
            if not isinstance(value, (int, float)):
                continue
            if float(value) > float(ceilings[key]):
                failures.append(key)
        return failures

    def _snapshot_kpi(self, breaker_kpi: Mapping[str, Any]) -> Dict[str, float]:
        return {key: float(breaker_kpi.get(key, 0.0)) for key in KPI_KEYS}

    def _snapshot_costs(self, costs: Mapping[str, Any]) -> Dict[str, float | int]:
        snapshot: Dict[str, float | int] = {}
        for key, value in costs.items():
            if not isinstance(value, (int, float)):
                continue
            if self._is_timing_key(key):
                continue
            snapshot[key] = value
        return snapshot

    def _is_timing_key(self, key: str) -> bool:
        lowered = key.lower()
        if "time" in lowered:
            return True
        return any(lowered.endswith(suffix) for suffix in TIMING_SUFFIXES)

    def _score_costs(self, costs: Mapping[str, Any]) -> Dict[str, int]:
        return {key: int(costs.get(key, 0)) for key in SCORE_COST_KEYS}

    def _score_key(
        self, verdict_flag: int, kpi: Mapping[str, float], costs: Mapping[str, int]
    ) -> list[int]:
        def _scaled(value: float) -> int:
            return int(round(value * SCORE_SCALE))

        cdr = _scaled(float(kpi.get("CDR", 0.0)))
        novn = _scaled(float(kpi.get("NOVN", 0.0)))
        wfhr = _scaled(float(kpi.get("WFHR", 0.0)))
        tmr = _scaled(float(kpi.get("TMR", 0.0)))
        breaker_attempts = int(costs.get("breaker_attempts", 0))
        meta_cases = int(costs.get("meta_cases", 0))
        return [
            int(verdict_flag),
            cdr,
            novn,
            wfhr,
            -tmr,
            -breaker_attempts,
            -meta_cases,
        ]

    def _coerce_score_key(self, value: Any) -> tuple[int, ...] | None:
        if isinstance(value, dict):
            value = value.get("score_key") or value.get("key")
        if isinstance(value, (list, tuple)):
            try:
                return tuple(int(item) for item in value)
            except (TypeError, ValueError):
                return None
        return None

    def evaluate(
        self,
        task_domain: str,
        lane_results: Mapping[str, Any],
        breaker_kpi: Mapping[str, Any],
        costs: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> ControllerVerdict:
        if self.policy_version == "v3":
            return self._evaluate_v3(
                task_domain=task_domain,
                lane_results=lane_results,
                breaker_kpi=breaker_kpi,
                costs=costs,
                context=context,
            )
        if self.policy_version == "v2":
            return self._evaluate_v2(
                task_domain=task_domain,
                lane_results=lane_results,
                breaker_kpi=breaker_kpi,
                costs=costs,
                context=context,
            )
        return self._evaluate_v1(
            task_domain=task_domain,
            lane_results=lane_results,
            breaker_kpi=breaker_kpi,
            costs=costs,
            context=context,
        )

    def _evaluate_v1(
        self,
        task_domain: str,
        lane_results: Mapping[str, Any],
        breaker_kpi: Mapping[str, Any],
        costs: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> ControllerVerdict:
        policy = self._policy_for(task_domain)
        minima = policy.get("kpi_minima", {})
        ceilings = policy.get("cost_ceilings", {})

        reason_atoms: list[str] = []
        sealed_regression = bool(context.get("sealed_regression", False))
        if sealed_regression:
            reason_atoms.append("SEALED_REGRESSION")

        required_passed = bool(lane_results.get("required_passed", False))
        if not required_passed:
            reason_atoms.append("LANES_FAIL")

        kpi_snapshot = self._snapshot_kpi(breaker_kpi)
        reason_atoms.extend(self._kpi_failures(breaker_kpi, minima))

        if self._cost_failures(costs, ceilings):
            reason_atoms.append("COST_CEILING")

        verdict_flag = 1 if required_passed else 0
        score_costs = self._score_costs(costs)
        score_key = self._score_key(verdict_flag, kpi_snapshot, score_costs)
        best_key = self._coerce_score_key(context.get("best_score"))
        candidate_hash = str(context.get("program_hash", ""))
        best_program_hash = str(context.get("best_program_hash", ""))
        if best_key is not None and len(best_key) == len(score_key):
            if candidate_hash and best_program_hash and candidate_hash == best_program_hash:
                pass
            elif tuple(score_key) <= best_key:
                reason_atoms.append("NOT_BETTER_THAN_BEST")

        decision = "ADMIT" if not reason_atoms else "REJECT"

        snapshots = {
            "kpi": kpi_snapshot,
            "cost": self._snapshot_costs(costs),
        }
        score = {
            "verdict": verdict_flag,
            "kpi": {key: kpi_snapshot.get(key, 0.0) for key in SCORE_KPI_KEYS},
            "cost": score_costs,
        }

        return ControllerVerdict(
            decision=decision,
            reason_atoms=reason_atoms,
            policy_id=self.policy_id,
            policy_hash=self.policy_hash,
            snapshots=snapshots,
            score=score,
            score_key=score_key,
            policy_version=self.policy_version,
            allow_admit=(decision == "ADMIT"),
        )

    def _evaluate_v2(
        self,
        task_domain: str,
        lane_results: Mapping[str, Any],
        breaker_kpi: Mapping[str, Any],
        costs: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> ControllerVerdict:
        reason_atoms: list[str] = []
        allow_admit = True

        required_passed = bool(lane_results.get("required_passed", False))
        families_mode = str(context.get("families_mode", "public"))
        if not required_passed:
            allow_admit = False
            if families_mode == "sealed":
                reason_atoms.append("SEALED_REQUIRED_FAILED")
            else:
                reason_atoms.append("REQUIRED_LANES_FAILED")

        if bool(context.get("metamorphic_violation", False)):
            allow_admit = False
            reason_atoms.append("METAMORPHIC_VIOLATION")

        sealed_regression = bool(context.get("sealed_regression", False))
        if sealed_regression and families_mode == "sealed":
            allow_admit = False
            reason_atoms.append("SEALED_REQUIRED_FAILED")

        domain_ceilings = self.policy.get("domain_cost_ceiling_ns", {})
        synth_ns = int(costs.get("synth_ns", 0))
        ceiling = int(domain_ceilings.get(task_domain, 0))
        cost_penalty = 0.0
        if ceiling > 0:
            cost_penalty = synth_ns / float(ceiling)
            if synth_ns > ceiling:
                allow_admit = False
                reason_atoms.append("COST_CEILING_EXCEEDED")

        coverage_reward = self.policy.get("coverage_reward", {})
        coverage_atoms = context.get("coverage_atoms", [])
        coverage_bonus = 0.0
        if (
            isinstance(coverage_atoms, Sequence)
            and not isinstance(coverage_atoms, (str, bytes))
            and coverage_reward.get("enabled")
            and coverage_reward.get("atoms_scope") == "episode"
        ):
            weight = float(coverage_reward.get("weight", 0.0))
            count = len([atom for atom in coverage_atoms if isinstance(atom, str) and atom])
            coverage_bonus = weight * count
            if coverage_bonus:
                reason_atoms.append("COVERAGE_REWARD_APPLIED")

        base_pass_score = 1.0 if required_passed else 0.0
        base_score = base_pass_score + coverage_bonus
        if families_mode == "sealed":
            sealed_weight = float(self.policy.get("sealed_weight", 1.0))
            base_score *= sealed_weight
        score = base_score - cost_penalty
        score_scaled = int(round(score * SCORE_SCALE))

        score_key = [score_scaled]
        best_key = self._coerce_score_key(context.get("best_score"))
        candidate_hash = str(context.get("program_hash", ""))
        best_program_hash = str(context.get("best_program_hash", ""))
        if best_key is not None and len(best_key) == len(score_key):
            if candidate_hash and best_program_hash and candidate_hash == best_program_hash:
                pass
            elif tuple(score_key) <= best_key:
                allow_admit = False
                reason_atoms.append("NOT_BETTER_THAN_BEST")

        if allow_admit:
            reason_atoms.append("CONTROLLER_OK")

        reason_atoms = sorted(set(reason_atoms))

        snapshots = {
            "kpi": self._snapshot_kpi(breaker_kpi),
            "cost": self._snapshot_costs(costs),
        }
        score_payload: Dict[str, Any] = {
            "base_pass_score": base_pass_score,
            "coverage_bonus": coverage_bonus,
            "cost_penalty": cost_penalty,
            "score": score,
            "score_scaled": score_scaled,
        }
        decision = "ADMIT" if allow_admit else "REJECT"

        return ControllerVerdict(
            decision=decision,
            reason_atoms=reason_atoms,
            policy_id=self.policy_id,
            policy_hash=self.policy_hash,
            snapshots=snapshots,
            score=score_payload,
            score_key=score_key,
            policy_version=self.policy_version,
            score_scaled=score_scaled,
            allow_admit=allow_admit,
        )

    def _evaluate_v3(
        self,
        task_domain: str,
        lane_results: Mapping[str, Any],
        breaker_kpi: Mapping[str, Any],
        costs: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> ControllerVerdict:
        reason_atoms: list[str] = []
        allow_admit = True

        required_passed = bool(lane_results.get("required_passed", False))
        if not required_passed:
            allow_admit = False
            reason_atoms.append("REQUIRED_LANES_FAILED")

        metamorphic_violation = bool(context.get("metamorphic_violation", False))
        if metamorphic_violation:
            reason_atoms.append("METAMORPHIC_VIOLATION")

        metamorphic_pass = bool(context.get("metamorphic_pass", False))
        verified_gain = 100 if required_passed else 0
        if metamorphic_pass:
            verified_gain += 20
        if metamorphic_violation:
            verified_gain -= 30

        coverage_gain_value = context.get("coverage_gain")
        coverage_atoms = context.get("coverage_atoms", [])
        if isinstance(coverage_gain_value, (int, float)):
            coverage_gain = int(coverage_gain_value)
        elif isinstance(coverage_atoms, Sequence) and not isinstance(
            coverage_atoms, (str, bytes)
        ):
            coverage_gain = len([atom for atom in coverage_atoms if isinstance(atom, str)])
        else:
            coverage_gain = 0

        synth_ns = int(costs.get("synth_ns", 0))
        verify_ns = int(costs.get("verify_ns", 0))
        breaker_ns = int(costs.get("breaker_ns", 0))
        cost_ns = synth_ns + verify_ns + breaker_ns

        domain_ceilings = self.policy.get("domain_cost_ceiling_ns", {})
        default_ceiling = int(self.policy.get("default_cost_ceiling_ns", 0))
        ceiling = int(domain_ceilings.get(task_domain, default_ceiling))
        if ceiling > 0 and cost_ns > ceiling:
            allow_admit = False
            reason_atoms.append("COST_CEILING")

        if required_passed:
            min_coverage = int(self.policy.get("min_coverage_gain", 0))
            if coverage_gain < min_coverage:
                allow_admit = False
                reason_atoms.append("INSUFFICIENT_COVERAGE_GAIN")

        score_scaled = int(
            (SCORE_SCALE * (verified_gain + 10 * coverage_gain)) // max(1, cost_ns)
        )
        min_score_scaled = int(self.policy.get("min_score_scaled", 0))
        if score_scaled < min_score_scaled:
            allow_admit = False
            reason_atoms.append("SCORE_TOO_LOW")

        if allow_admit:
            reason_atoms.append("CONTROLLER_OK")

        reason_atoms = sorted(set(reason_atoms))
        snapshots = {
            "kpi": self._snapshot_kpi(breaker_kpi),
            "cost": self._snapshot_costs(costs),
        }
        score_payload: Dict[str, Any] = {
            "verified_gain": verified_gain,
            "coverage_gain": coverage_gain,
            "cost_ns": cost_ns,
            "score_scaled": score_scaled,
        }
        decision = "ADMIT" if allow_admit else "REJECT"

        return ControllerVerdict(
            decision=decision,
            reason_atoms=reason_atoms,
            policy_id=self.policy_id,
            policy_hash=self.policy_hash,
            snapshots=snapshots,
            score=score_payload,
            score_key=[score_scaled],
            policy_version=self.policy_version,
            score_scaled=score_scaled,
            allow_admit=allow_admit,
        )
