from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

from ..artifact_store import ArtifactStore
from ..bg.bg_engine import BGEngine, compute_context_hash
from ..breaker.breaker import BreakerLab
from ..bvps.cegis import run_cegis
from ..config import Settings
from ..forge.forge import ForgeGate
from ..ledger.ledger import Ledger
from ..schemas import UCR, WitnessPacket
from ..utils import ensure_dir, stable_hash, write_json
from ..verify.lanes import VerifierContext
from ..verify.verifier import required_lanes_passed, run_verifiers
from .kernel import KernelStub
from .specs import task_spec
from .task import load_task


def _init_run_dirs(run_dir: Path) -> None:
    ensure_dir(run_dir)
    for name in ["artifacts", "witnesses", "bg", "reports", "capsules"]:
        ensure_dir(run_dir / name)


def episode_run(task_file: Path, run_dir: Path, settings: Settings) -> Dict[str, Any]:
    _init_run_dirs(run_dir)
    ledger = Ledger(run_dir / "ledger.jsonl")
    ledger.append("RUN_START", {"run_dir": str(run_dir), "task_file": str(task_file)})

    task = load_task(task_file)
    kernel = KernelStub()
    interpretations = kernel.propose_interpretations(task)
    chosen = kernel.choose_interpretation(task, interpretations)

    rng_seed = settings.seed_for(run_dir.name)
    start_synth = time.time_ns()
    cegis_result = run_cegis(task, settings, rng_seed)
    synth_cost = time.time_ns() - start_synth

    artifact_store = ArtifactStore(run_dir / "artifacts", ledger)
    artifacts: List[Any] = []
    artifacts.append(
        artifact_store.write_json(
            "bvps/program.json", cegis_result.program.to_json(), "bvps_program"
        )
    )
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
        artifact_store.write_json("reports/verifier.json", verdicts_payload, "verifier_report")
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

    required_pass = required_lanes_passed(verdicts, settings.required_lanes)
    witness_id = stable_hash(
        {
            "run_id": run_dir.name,
            "task_id": task.task_id,
            "program_hash": cegis_result.ast_hash,
        }
    )

    witness = WitnessPacket(
        witness_id=witness_id,
        artifacts=artifacts,
        verifier_verdicts=verdicts,
        breaker_evidence={
            "report": breaker_result.report,
            "kpi": breaker_result.kpi.model_dump(),
        },
        hashes={"program": cegis_result.ast_hash, "interpreter": cegis_result.interpreter_hash},
        coverage={"verifier_lanes": [verdict.tier for verdict in verdicts]},
        cost_report={"synth_ns": synth_cost},
        policy_versions={"policy_version": settings.policy_version},
    )

    witness_path = run_dir / "witnesses" / f"{witness_id}.json"
    write_json(witness_path, witness.model_dump())
    ledger.append("WITNESS_WRITTEN", {"path": str(witness_path), "witness_id": witness_id})

    ucr = UCR(
        run_id=run_dir.name,
        task_id=task.task_id,
        inputs=task.open_view(),
        interpretations=[interp.__dict__ for interp in interpretations],
        chosen_interpretation=chosen.__dict__,
        solver_trace=["bvps"],
        costs={"synth_ns": synth_cost},
        hashes={"witness_id": witness_id, "program_hash": cegis_result.ast_hash},
        run_metadata={"policy_version": settings.policy_version},
    )
    ucr_path = run_dir / "ucr.json"
    write_json(ucr_path, ucr.model_dump())
    ledger.append("UCR_WRITTEN", {"path": str(ucr_path), "ucr_hash": ucr.stable_hash()})

    forge = ForgeGate()
    decision = forge.decide(task, cegis_result.program, verdicts, required_pass)
    ledger.append("FORGE_DECISION", {"decision": decision.decision, "reason": decision.reason})

    if decision.decision == "PROMOTE":
        bg_engine = BGEngine(run_dir, ledger)
        op = forge.build_promotion_op(cegis_result.program, witness_id)
        bg_engine.apply(op, record=True)
    elif decision.decision in {"QUARANTINE", "ROLLBACK"}:
        ledger.append("FORGE_DECISION", {"decision": decision.decision, "reason": decision.reason})

    context_hash = compute_context_hash(
        {"task": task.task_id, "policy": settings.policy_version}
    )
    active_view = BGEngine.replay(
        run_dir / "bg" / "revisions.jsonl", context_hash, settings.policy_version
    )

    ledger.append(
        "RUN_END",
        {
            "run_id": run_dir.name,
            "verdict": "PASS" if required_pass else "FAIL",
            "active_view_hash": active_view.active_view_hash,
        },
    )

    return {
        "run_id": run_dir.name,
        "task_id": task.task_id,
        "verdict": "PASS" if required_pass else "FAIL",
        "witness_path": str(witness_path),
        "ucr_path": str(ucr_path),
        "active_view_hash": active_view.active_view_hash,
    }
