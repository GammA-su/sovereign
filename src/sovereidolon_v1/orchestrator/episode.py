from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from ..artifact_store import ArtifactStore
from ..bg.bg_engine import BGEngine, compute_context_hash
from ..breaker.breaker import BreakerLab
from ..bvps.cegis import CEGISResult, run_cegis
from ..bvps.dsl import Program
from ..bvps.interpreter import INTERPRETER_HASH, eval_program
from ..config import Settings
from ..forge.forge import ForgeGate
from ..ledger.ledger import Ledger
from ..schemas import UCR, BGRevisionOp, VerifierVerdict, WitnessPacket
from ..utils import (
    canonical_dumps,
    ensure_dir,
    hash_bytes,
    read_json,
    read_jsonl,
    stable_hash,
    write_json,
)
from ..verify.lanes import VerifierContext
from ..verify.verifier import required_lanes_passed, run_verifiers
from .kernel import KernelStub
from .specs import task_spec
from .task import Task, load_task


class ContradictoryExamplesError(RuntimeError):
    pass


def _init_run_dirs(run_dir: Path) -> None:
    ensure_dir(run_dir)
    for name in ["artifacts", "witnesses", "bg", "reports", "capsules"]:
        ensure_dir(run_dir / name)


def _ledger_has_run_start(ledger_path: Path) -> bool:
    if not ledger_path.exists():
        return False
    entries = read_jsonl(ledger_path)
    return any(entry.get("type") == "RUN_START" for entry in entries)


def _select_run_dir(run_dir: Path) -> Path:
    if not _ledger_has_run_start(run_dir / "ledger.jsonl"):
        return run_dir
    parent = run_dir.parent
    base = run_dir.name
    for idx in range(1, 1000):
        candidate = parent / f"{base}_{idx}"
        if not _ledger_has_run_start(candidate / "ledger.jsonl"):
            return candidate
    raise RuntimeError("Unable to select unique run_dir")


def _load_store_candidate(
    store_dir: Path, spec_signature: str, domain: str
) -> tuple[Program, str] | None:
    manifest_path = store_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    programs: dict[str, dict[str, Any]] = manifest.get("programs", {})
    domain_candidates: list[tuple[Program, str]] = []
    for program_hash, entry in programs.items():
        entry_domain = entry.get("domain", "")
        if not entry_domain:
            continue
        program_path = Path(
            entry.get("store_path", store_dir / entry_domain / f"{program_hash}.json")
        )
        if not program_path.exists():
            continue
        program_data = read_json(program_path)
        program = Program.model_validate(program_data)
        if entry.get("spec_signature") == spec_signature:
            return program, program_hash
        if entry_domain == domain:
            domain_candidates.append((program, program_hash))
    if domain_candidates:
        domain_candidates.sort(key=lambda item: item[1])
        return domain_candidates[0]
    return None


def _synth_failure_verdict(task: Task, reason: str) -> VerifierVerdict:
    return VerifierVerdict(
        verdict="FAIL",
        failure_atoms=[reason],
        domain="bvps",
        tier="synth",
        bounds=task.bounds,
        soundness_grade="HEURISTIC",
        metamorphic_families=[],
        cost={},
    )


def episode_run(task_file: Path, run_dir: Path, settings: Settings) -> Dict[str, Any]:
    run_dir = _select_run_dir(run_dir)
    _init_run_dirs(run_dir)
    ledger = Ledger(run_dir / "ledger.jsonl")
    ledger.append("RUN_START", {"run_dir": str(run_dir), "task_file": str(task_file)})

    context_data = {
        "run_id": run_dir.name,
        "context_name": "default",
        "policy_version": settings.policy_version,
    }

    task: Optional[Task] = None
    interpretations_data: List[Dict[str, Any]] = []
    chosen_data: Dict[str, Any] = {}
    artifacts: List[Any] = []
    verdicts: List[VerifierVerdict] = []
    breaker_report: Dict[str, Any] = {}
    breaker_kpi: Dict[str, Any] = {}
    failure_reason = ""
    stack_summary = ""
    required_pass = False
    overall_verdict: Literal["PASS", "FAIL"] = "FAIL"
    witness_id = ""
    active_view_hash = ""
    synth_cost = 0
    cegis_result: Optional[CEGISResult] = None
    warm_start = False
    warm_start_candidate_hash = ""
    warm_start_candidate_rejected = False
    warm_start_attempted = False
    warm_start_successful = False
    warm_start_provided = bool(settings.warm_start_store)

    artifact_store = ArtifactStore(run_dir / "artifacts", ledger)

    try:
        task = load_task(task_file)
        kernel = KernelStub()
        interpretations = kernel.propose_interpretations(task)
        chosen = kernel.choose_interpretation(task, interpretations)
        interpretations_data = [interp.__dict__ for interp in interpretations]
        chosen_data = chosen.__dict__

        if task.has_contradictory_examples():
            failure_reason = "EXAMPLE_CONTRADICT"
            raise ContradictoryExamplesError("contradictory examples in task")

        spec_signature = task.spec_signature()
        warm_start_program: tuple[Program, str] | None = None
        if settings.warm_start_store:
            store_candidate = _load_store_candidate(
                Path(settings.warm_start_store), spec_signature, task.task_type
            )
            if store_candidate:
                warm_start_program = store_candidate
                warm_start_candidate_hash = store_candidate[1]

        rng_seed = settings.seed_for(run_dir.name)
        if warm_start_program:
            warm_start_attempted = True
            program, program_hash = warm_start_program
            tests = list(task.examples)
            trace_hashes: List[str] = []
            for example in tests:
                _, trace_hash = eval_program(
                    program, example.inputs, step_limit=settings.verify_budget_steps
                )
                trace_hashes.append(trace_hash)
            cegis_result = CEGISResult(
                status="ok",
                program=program,
                tests=tests,
                counterexamples=[],
                ast_hash=program_hash,
                interpreter_hash=INTERPRETER_HASH,
                trace_hashes=trace_hashes,
            )
            synth_cost = 0
        else:
            start_synth = time.time_ns()
            cegis_result = run_cegis(task, settings, rng_seed)
            synth_cost = time.time_ns() - start_synth

        artifacts.append(
            artifact_store.write_json(
                "bvps/tests.json",
                {
                    "tests": [example.model_dump() for example in cegis_result.tests],
                    "counterexamples": [
                        example.model_dump() for example in cegis_result.counterexamples
                    ],
                },
                "bvps_tests",
            )
        )
        artifacts.append(
            artifact_store.write_json(
                "bvps/trace_hashes.json",
                {"trace_hashes": cegis_result.trace_hashes},
                "bvps_traces",
            )
        )

        if cegis_result.status != "ok" or cegis_result.program is None:
            failure_reason = cegis_result.failure_reason or "SYNTH_FAIL"
            verdicts = [_synth_failure_verdict(task, failure_reason)]
            if warm_start_attempted:
                warm_start_candidate_rejected = True
        else:
            artifacts.append(
                artifact_store.write_json(
                    "bvps/program.json", cegis_result.program.to_json(), "bvps_program"
                )
            )

            spec = task_spec(task)
            verifier_ctx = VerifierContext(
                task=task,
                program=cegis_result.program,
                tests=cegis_result.tests,
                trace_hashes=cegis_result.trace_hashes,
                run_dir=str(run_dir),
                settings=settings,
                spec=spec,
            )

            verdicts = run_verifiers(verifier_ctx)
            verdicts_payload = [verdict.model_dump() for verdict in verdicts]
            artifacts.append(
                artifact_store.write_json(
                    "reports/verifier.json", verdicts_payload, "verifier_report"
                )
            )
            for verdict in verdicts:
                ledger.append("VERIFIER_RESULT", verdict.model_dump())

            breaker_lab = BreakerLab(settings, run_dir)
            breaker_result = breaker_lab.run(
                task=task,
                program=cegis_result.program,
                spec=spec,
                tests=cegis_result.tests,
                budget=settings.break_budget_attempts,
                seed=rng_seed + 99,
            )
            artifacts.append(
                artifact_store.write_json(
                    "reports/breaker.json", breaker_result.report, "breaker_report"
                )
            )
            artifacts.append(
                artifact_store.write_json(
                    "reports/breaker_kpi.json", breaker_result.kpi.model_dump(), "breaker_kpi"
                )
            )
            ledger.append("BREAKER_RESULT", breaker_result.report)
            breaker_report = breaker_result.report
            breaker_kpi = breaker_result.kpi.model_dump()

            required_pass = required_lanes_passed(
                verdicts, settings.admission_policy.required_lanes
            )
            overall_verdict = "PASS" if required_pass else "FAIL"
            if not required_pass and not failure_reason:
                failure_reason = "REQUIRED_LANES_FAIL"

            breaker_found = False
            if task.sealed:
                if breaker_result.report.get("counterexample") is not None:
                    breaker_found = True
                if breaker_result.report.get("withheld_hits", 0) > 0:
                    breaker_found = True
            if breaker_found:
                required_pass = False
                overall_verdict = "FAIL"
                if not failure_reason:
                    failure_reason = "BREAKER_WITHHELD_FAIL"

            if warm_start_attempted:
                if required_pass:
                    warm_start_successful = True
                else:
                    warm_start_candidate_rejected = True

            if not witness_id:
                witness_id = stable_hash(
                    {
                        "run_id": run_dir.name,
                        "task_id": task.task_id,
                        "failure_reason": failure_reason,
                        "program_hash": cegis_result.ast_hash,
                    }
                )

            forge = ForgeGate()
            controller_overhead_ratio = 0.0
            withheld_hits = int(breaker_result.report.get("withheld_hits", 0))
            decision = forge.decide(
                task,
                cegis_result.program,
                verdicts,
                required_pass,
                settings.admission_policy,
                controller_overhead_ratio,
                withheld_hits,
            )
            ledger.append(
                "FORGE_DECISION",
                {"decision": decision.decision, "reason": decision.reason},
            )

            decision_payload = {
                "decision": decision.decision,
                "reason": decision.reason,
                "witness_id": witness_id,
                "program_hash": cegis_result.ast_hash,
            }
            decision_path = run_dir / "forge" / "decision.json"
            write_json(decision_path, decision_payload)

            if decision.decision == "ADMIT":
                store_dir = Path("store") / "v1" / task.task_type
                ensure_dir(store_dir)
                program_bytes = canonical_dumps(cegis_result.program.to_json())
                store_path = store_dir / f"{cegis_result.ast_hash}.json"
                store_path.write_bytes(program_bytes)
                content_hash = hash_bytes(program_bytes)
                ledger.append(
                    "FORGE_ADMIT",
                    {
                        "path": str(store_path),
                        "content_hash": content_hash,
                        "bytes": len(program_bytes),
                        "witness_id": witness_id,
                        "domain": task.task_type,
                    },
                )
            elif decision.decision == "REJECT":
                ledger.append(
                    "FORGE_REJECT",
                    {"reason": decision.reason, "witness_id": witness_id},
                )
            elif decision.decision in {"QUARANTINE", "ROLLBACK"}:
                ledger.append(
                    "FORGE_REJECT",
                    {"reason": decision.reason, "witness_id": witness_id},
                )

    except Exception:  # noqa: BLE001
        if not failure_reason:
            failure_reason = "EXCEPTION"
        stack_summary = traceback.format_exc()
        if task is None:
            task = Task(
                task_id="unknown",
                family="unknown",
                task_type="unknown",
                goal="unknown",
                inputs={},
                output="unknown",
                bounds={},
                examples=[],
            )
        verdicts = verdicts or [_synth_failure_verdict(task, failure_reason)]

    task_id = task.task_id if task else "unknown"
    if not witness_id:
        witness_id = stable_hash(
            {
                "run_id": run_dir.name,
                "task_id": task_id,
                "failure_reason": failure_reason,
                "program_hash": cegis_result.ast_hash if cegis_result else "",
            }
        )

    warm_start = warm_start_successful
    witness = WitnessPacket(
        witness_id=witness_id,
        overall_verdict=overall_verdict,
        failure_reason=failure_reason,
        artifacts=artifacts,
        verifier_verdicts=verdicts,
        breaker_evidence={"report": breaker_report, "kpi": breaker_kpi},
        hashes={
            "program": cegis_result.ast_hash if cegis_result else "",
            "interpreter": cegis_result.interpreter_hash if cegis_result else "",
        },
        coverage={"verifier_lanes": [verdict.tier for verdict in verdicts]},
        cost_report={"synth_ns": synth_cost},
        policy_versions={"policy_version": settings.policy_version},
    )

    if task is not None and cegis_result is not None:
        bg_node_payload = {
            "task_id": task_id,
            "interpretation": chosen_data,
            "program_hash": cegis_result.ast_hash,
        }
        bg_node_id = stable_hash(bg_node_payload)
        bg_op = BGRevisionOp(
            op="ASSERT",
            witness_id=witness_id,
            node_id=bg_node_id,
            payload=bg_node_payload,
        )
        bg_engine = BGEngine(run_dir, ledger)
        bg_engine.apply(bg_op, record=True)

    context_hash = compute_context_hash(context_data)
    active_view = BGEngine.replay(
        run_dir / "bg" / "revisions.jsonl",
        context_hash,
        context_data["policy_version"],
    )
    active_view_hash = active_view.active_view_hash

    witness_path = run_dir / "witnesses" / f"{witness_id}.json"
    write_json(witness_path, witness.model_dump())
    ledger.append("WITNESS_WRITTEN", {"path": str(witness_path), "witness_id": witness_id})

    ucr = UCR(
        run_id=run_dir.name,
        task_id=task_id,
        inputs=task.open_view() if task else {},
        interpretations=interpretations_data,
        chosen_interpretation=chosen_data,
        solver_trace=["bvps"],
        costs={"synth_ns": synth_cost},
        hashes={
            "witness_id": witness_id,
            "program_hash": cegis_result.ast_hash if cegis_result else "",
        },
        bg_context=context_data,
        active_view_hash=active_view_hash,
        run_metadata={"policy_version": settings.policy_version},
    )
    ucr_path = run_dir / "ucr.json"
    write_json(ucr_path, ucr.model_dump())
    ledger.append("UCR_WRITTEN", {"path": str(ucr_path), "ucr_hash": ucr.stable_hash()})

    if overall_verdict == "FAIL":
        capsule = {
            "task_id": task_id,
            "run_id": run_dir.name,
            "witness_id": witness_id,
            "failure_reason": failure_reason,
            "stack_summary": stack_summary,
            "examples": [example.model_dump() for example in task.examples] if task else [],
            "counterexamples": [
                example.model_dump() for example in cegis_result.counterexamples
            ]
            if cegis_result
            else [],
            "trace_hashes": cegis_result.trace_hashes if cegis_result else [],
            "attempted_program": cegis_result.program.to_json()
            if cegis_result and cegis_result.program
            else None,
            "artifacts": [record.model_dump() for record in artifacts],
        }
        capsule_path = run_dir / "capsules" / f"failure_{witness_id}.json"
        write_json(capsule_path, capsule)
        ledger.append("CAPSULE_WRITTEN", {"path": str(capsule_path)})

    ledger.append(
        "RUN_END",
        {
            "run_id": run_dir.name,
            "verdict": overall_verdict,
            "active_view_hash": active_view_hash,
            "witness_id": witness_id,
        },
    )

    return {
        "run_id": run_dir.name,
        "task_id": task_id,
        "verdict": overall_verdict,
        "witness_path": str(witness_path),
        "ucr_path": str(ucr_path),
        "active_view_hash": active_view_hash,
        "warm_start_provided": warm_start_provided,
        "warm_start_store": warm_start,
        "warm_start_candidate_hash": warm_start_candidate_hash,
        "warm_start_candidate_rejected": warm_start_candidate_rejected,
        "synth_ns": synth_cost,
        "failure_reason": failure_reason,
    }
