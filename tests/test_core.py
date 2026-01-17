from __future__ import annotations

import shutil
from pathlib import Path

import orjson
from typer.testing import CliRunner

from sovereidolon_v1.bg.bg_engine import BGEngine, compute_context_hash
from sovereidolon_v1.breaker.breaker import BreakerLab
from sovereidolon_v1.bvps.cegis import run_cegis
from sovereidolon_v1.bvps.dsl import Program, binop, var
from sovereidolon_v1.bvps.interpreter import Interpreter
from sovereidolon_v1.cli import (
    app,
    audit_run,
    bg_replay_cmd,
    migrate_run,
    normalize_suite_report,
    verify_bg_replay,
)
from sovereidolon_v1.config import Settings
from sovereidolon_v1.forge.forge import ForgeGate
from sovereidolon_v1.ledger.ledger import Ledger
from sovereidolon_v1.orchestrator.episode import episode_run
from sovereidolon_v1.orchestrator.specs import task_spec
from sovereidolon_v1.orchestrator.task import Example, Task
from sovereidolon_v1.schemas import BGRevisionOp, VerifierVerdict
from sovereidolon_v1.utils import hash_bytes, read_json, stable_hash


def test_ledger_chain_verification(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = Ledger(ledger_path)
    ledger.append("RUN_START", {"a": 1})
    ledger.append("RUN_END", {"b": 2})
    ok, _ = Ledger.verify_chain(ledger_path)
    assert ok

    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    entry = orjson.loads(lines[0])
    entry["payload"]["a"] = 2
    lines[0] = orjson.dumps(entry, option=orjson.OPT_SORT_KEYS).decode("utf-8")
    ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok, _ = Ledger.verify_chain(ledger_path)
    assert not ok


def test_canonical_hash_stability() -> None:
    payload = {"a": 1, "b": [2, 3]}
    assert stable_hash(payload) == stable_hash(payload)


def test_bg_replay_hash_stability() -> None:
    ops = [
        BGRevisionOp(op="ASSERT", witness_id="w", node_id="n1", payload={"v": 1}),
        BGRevisionOp(op="ASSERT", witness_id="w", node_id="n2", payload={"v": 2}),
        BGRevisionOp(
            op="DECLARE_CONFLICT", witness_id="w", conflict_id="c1", conflict_set=["n1", "n2"]
        ),
        BGRevisionOp(
            op="RESOLVE_CONFLICT",
            witness_id="w",
            conflict_id="c1",
            chosen_id="n1",
            disruption_cost=1,
        ),
    ]
    context_hash = compute_context_hash({"policy": "v1"})
    view1 = BGEngine.replay_from_ops(ops, context_hash, "v1")
    view2 = BGEngine.replay_from_ops(ops, context_hash, "v1")
    assert view1.active_view_hash == view2.active_view_hash


def test_conflict_no_coactive_contradictions() -> None:
    ops = [
        BGRevisionOp(op="ASSERT", witness_id="w", node_id="n1", payload={"v": 1}),
        BGRevisionOp(op="ASSERT", witness_id="w", node_id="n2", payload={"v": 2}),
        BGRevisionOp(
            op="DECLARE_CONFLICT", witness_id="w", conflict_id="c1", conflict_set=["n1", "n2"]
        ),
    ]
    context_hash = compute_context_hash({"policy": "v1"})
    view = BGEngine.replay_from_ops(ops, context_hash, "v1")
    assert not ("n1" in view.active_nodes and "n2" in view.active_nodes)


def test_bvps_interpreter_determinism() -> None:
    program = Program(
        name="add",
        arg_types={"x": "Int", "y": "Int"},
        return_type="Int",
        body=binop("+", var("x"), var("y")),
    )
    interpreter = Interpreter(step_limit=100)
    result1 = interpreter.evaluate(program, {"x": 1, "y": 2})
    result2 = interpreter.evaluate(program, {"x": 1, "y": 2})
    assert result1.trace_hash == result2.trace_hash


def test_cegis_finds_counterexample_then_repairs() -> None:
    task = Task(
        task_id="t1",
        family="arith",
        task_type="arith",
        goal="add",
        inputs={"x": "Int", "y": "Int"},
        output="Int",
        bounds={"x": [-5, 5], "y": [-5, 5]},
        examples=[Example(inputs={"x": 0, "y": 0}, output=0)],
    )
    settings = Settings(verify_budget_steps=200, break_budget_attempts=50)
    result = run_cegis(task, settings, rng_seed=42)
    assert result.status == "ok"
    assert result.counterexamples
    assert result.program is not None
    assert result.program.name == "add"


def test_breaker_kpis_emitted_and_novelty_updates(tmp_path: Path) -> None:
    task = Task(
        task_id="t2",
        family="arith",
        task_type="arith",
        goal="add",
        inputs={"x": "Int", "y": "Int"},
        output="Int",
        bounds={"x": [-3, 3], "y": [-3, 3]},
        examples=[Example(inputs={"x": 0, "y": 0}, output=0)],
    )
    program = Program(
        name="identity",
        arg_types=task.inputs,
        return_type=task.output,
        body=var("x"),
    )
    settings = Settings(verify_budget_steps=100, break_budget_attempts=20)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "capsules").mkdir()
    lab = BreakerLab(settings, run_dir)
    spec = task_spec(task)
    result1 = lab.run(task, program, spec, task.examples, budget=10, seed=1)
    result2 = lab.run(task, program, spec, task.examples, budget=10, seed=2)
    assert 0.0 <= result1.kpi.NOVN <= 1.0
    assert 0.0 <= result2.kpi.NOVN <= 1.0
    history = (run_dir / "capsules" / "novelty.jsonl").read_text(encoding="utf-8")
    assert history.strip()


def test_fail_closed_no_promote_on_heuristic_only() -> None:
    task = Task(
        task_id="t3",
        family="arith",
        task_type="arith",
        goal="add",
        inputs={"x": "Int", "y": "Int"},
        output="Int",
        bounds={},
        examples=[Example(inputs={"x": 1, "y": 2}, output=3)],
    )
    program = Program(
        name="add",
        arg_types=task.inputs,
        return_type=task.output,
        body=binop("+", var("x"), var("y")),
    )
    verdicts = [
        VerifierVerdict(
            verdict="PASS",
            failure_atoms=[],
            domain="bvps",
            tier="transfer",
            bounds={},
            soundness_grade="HEURISTIC",
            metamorphic_families=[],
            cost={},
        )
    ]
    gate = ForgeGate()
    policy = Settings().admission_policy
    decision = gate.decide(
        task,
        program,
        verdicts,
        required_passed=True,
        admission_policy=policy,
        controller_overhead_ratio=0.0,
        withheld_hits=0,
    )
    assert decision.decision == "REJECT"


def test_sealed_canary_quarantine_on_echo() -> None:
    task = Task(
        task_id="t4",
        family="arith",
        task_type="arith",
        goal="add",
        inputs={"x": "Int", "y": "Int"},
        output="Int",
        bounds={},
        examples=[Example(inputs={"x": 1, "y": 2}, output=3)],
        sealed={"canary_token": "SECRET", "sealed_seed": 1, "withheld_families": []},
    )
    program = Program(
        name="SECRET",
        arg_types=task.inputs,
        return_type=task.output,
        body=binop("+", var("x"), var("y")),
    )
    gate = ForgeGate()
    policy = Settings().admission_policy
    decision = gate.decide(
        task,
        program,
        [],
        required_passed=True,
        admission_policy=policy,
        controller_overhead_ratio=0.0,
        withheld_hits=0,
    )
    assert decision.decision == "QUARANTINE"


def test_cli_episode_run_writes_ucr_witness_ledger(tmp_path: Path) -> None:
    runner = CliRunner()
    task_file = Path("examples/tasks/arith_01.json")
    run_dir = tmp_path / "run_cli"
    result = runner.invoke(
        app,
        [
            "episode",
            "run",
            "--task-file",
            str(task_file),
            "--run-dir",
            str(run_dir),
        ],
    )
    assert result.exit_code == 0
    assert (run_dir / "ucr.json").exists()
    assert (run_dir / "ledger.jsonl").exists()
    assert (run_dir / "witnesses").exists()


def test_ucr_active_view_hash_matches_summary(tmp_path: Path) -> None:
    settings = Settings()
    task_file = Path("examples/tasks/arith_01.json")
    run_dir = tmp_path / "run_ucr"
    summary = episode_run(task_file=task_file, run_dir=run_dir, settings=settings)

    ucr_data = orjson.loads((run_dir / "ucr.json").read_bytes())
    assert ucr_data.get("active_view_hash")
    assert summary["active_view_hash"] == ucr_data["active_view_hash"]


def test_ucr_bg_context_replay_matches_hash(tmp_path: Path) -> None:
    settings = Settings()
    task_file = Path("examples/tasks/arith_01.json")
    run_dir = tmp_path / "run_context"
    episode_run(task_file=task_file, run_dir=run_dir, settings=settings)

    ucr_data = orjson.loads((run_dir / "ucr.json").read_bytes())
    bg_context = ucr_data["bg_context"]
    assert bg_context
    context_path = run_dir / "ucr_bg_context.json"
    context_path.write_text(orjson.dumps(bg_context).decode("utf-8"), encoding="utf-8")

    context_hash = compute_context_hash(bg_context)
    policy_version = bg_context.get("policy_version", settings.policy_version)
    active_view = BGEngine.replay(
        run_dir / "bg" / "revisions.jsonl", context_hash, policy_version
    )
    assert active_view.active_view_hash == ucr_data["active_view_hash"]


def test_unsat_ucr_bg_context_written(tmp_path: Path) -> None:
    settings = Settings()
    task_file = Path("examples/tasks/arith_fail_01.json")
    run_dir = tmp_path / "run_fail_context"

    episode_run(task_file=task_file, run_dir=run_dir, settings=settings)
    ucr_data = orjson.loads((run_dir / "ucr.json").read_bytes())
    assert ucr_data.get("bg_context")


def test_ucr_active_view_hash_deterministic(tmp_path: Path) -> None:
    settings = Settings()
    task_file = Path("examples/tasks/arith_01.json")
    run_dir_a = tmp_path / "a" / "run_same"
    run_dir_b = tmp_path / "b" / "run_same"

    summary_a = episode_run(task_file=task_file, run_dir=run_dir_a, settings=settings)
    summary_b = episode_run(task_file=task_file, run_dir=run_dir_b, settings=settings)

    ucr_a = orjson.loads((run_dir_a / "ucr.json").read_bytes())
    ucr_b = orjson.loads((run_dir_b / "ucr.json").read_bytes())

    assert summary_a["active_view_hash"] == summary_b["active_view_hash"]
    assert ucr_a["active_view_hash"] == ucr_b["active_view_hash"]


def test_bg_replay_context_path_and_name_match(tmp_path: Path) -> None:
    settings = Settings()
    task_file = Path("examples/tasks/arith_01.json")
    run_dir = tmp_path / "run_bg"
    episode_run(task_file=task_file, run_dir=run_dir, settings=settings)

    context_path = tmp_path / "context.json"
    context_data = {"run_id": run_dir.name, "context_name": "default"}
    context_path.write_text(orjson.dumps(context_data).decode("utf-8"), encoding="utf-8")

    hash_from_file = bg_replay_cmd(run_dir=run_dir, context=str(context_path), policy_version="v1")
    hash_from_name = bg_replay_cmd(run_dir=run_dir, context="default", policy_version="v1")

    assert hash_from_file == hash_from_name


def test_bg_verify_replay_ok(tmp_path: Path) -> None:
    settings = Settings()
    task_file = Path("examples/tasks/arith_01.json")
    run_dir = tmp_path / "run_verify"

    summary = episode_run(task_file=task_file, run_dir=run_dir, settings=settings)
    episode_dir = Path(summary["ucr_path"]).parent

    report = verify_bg_replay(episode_dir)
    assert report["ok"] is True


def test_duplicate_run_dir_creates_unique_run(tmp_path: Path) -> None:
    settings = Settings()
    task_file = Path("examples/tasks/arith_01.json")
    run_dir = tmp_path / "dup_run"

    summary_a = episode_run(task_file=task_file, run_dir=run_dir, settings=settings)
    summary_b = episode_run(task_file=task_file, run_dir=run_dir, settings=settings)

    assert summary_a["run_id"] != summary_b["run_id"]
    ledger_entries = [
        orjson.loads(line)
        for line in (run_dir / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    run_start_count = sum(1 for entry in ledger_entries if entry.get("type") == "RUN_START")
    assert run_start_count == 1


def test_unsat_episode_writes_witness_and_capsule(tmp_path: Path) -> None:
    settings = Settings()
    task_file = Path("examples/tasks/arith_fail_01.json")
    run_dir = tmp_path / "unsat"

    summary = episode_run(task_file=task_file, run_dir=run_dir, settings=settings)
    assert summary["verdict"] == "FAIL"
    assert Path(summary["witness_path"]).exists()

    ledger_lines = (run_dir / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    entries = [orjson.loads(line) for line in ledger_lines]
    run_start_count = sum(1 for entry in entries if entry.get("type") == "RUN_START")
    run_end_count = sum(1 for entry in entries if entry.get("type") == "RUN_END")
    assert run_start_count == 1
    assert run_end_count == 1

    capsules = list((run_dir / "capsules").glob("failure_*.json"))
    assert capsules

    ok, _ = Ledger.verify_chain(run_dir / "ledger.jsonl")
    assert ok


def test_run_audit_pass_and_fail(tmp_path: Path) -> None:
    settings = Settings()
    pass_dir = tmp_path / "audit_pass"
    fail_dir = tmp_path / "audit_fail"

    summary_pass = episode_run(
        task_file=Path("examples/tasks/arith_01.json"), run_dir=pass_dir, settings=settings
    )
    pass_report = audit_run(Path(summary_pass["ucr_path"]).parent)
    assert pass_report["ok"] is True

    summary_fail = episode_run(
        task_file=Path("examples/tasks/arith_fail_01.json"), run_dir=fail_dir, settings=settings
    )
    fail_report = audit_run(Path(summary_fail["ucr_path"]).parent)
    assert fail_report["ok"] is True
    assert fail_report["checks"].get("capsule") is True


def test_run_audit_detects_tamper(tmp_path: Path) -> None:
    settings = Settings()
    run_dir = tmp_path / "audit_tamper"
    summary = episode_run(
        task_file=Path("examples/tasks/arith_01.json"),
        run_dir=run_dir,
        settings=settings,
    )

    witness_data = orjson.loads(Path(summary["witness_path"]).read_bytes())
    artifact_path = Path(witness_data["artifacts"][0]["path"])
    data = artifact_path.read_bytes()
    if data:
        tampered = bytes([data[0] ^ 0xFF]) + data[1:]
        artifact_path.write_bytes(tampered)

    report = audit_run(Path(summary["ucr_path"]).parent)
    assert report["ok"] is False
    assert report["checks"].get("artifact_hashes") is False


def test_migrate_populates_capsule_and_audit(tmp_path: Path) -> None:
    settings = Settings()
    run_dir = tmp_path / "legacy_fail"
    episode_run(
        task_file=Path("examples/tasks/bool_fail_01.json"),
        run_dir=run_dir,
        settings=settings,
    )

    capsule_dir = run_dir / "capsules"
    capsule_path = next(capsule_dir.glob("failure_*.json"))
    capsule = orjson.loads(capsule_path.read_bytes())
    capsule.pop("witness_id", None)
    capsule_path.write_bytes(orjson.dumps(capsule, option=orjson.OPT_SORT_KEYS))

    assert audit_run(run_dir)["ok"] is False
    report = migrate_run(run_dir, in_place=True)
    assert report["ok"] is True
    updated_capsule = orjson.loads(capsule_path.read_bytes())
    assert updated_capsule.get("witness_id")

    ledger_lines = (run_dir / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ledger_entries = [orjson.loads(line) for line in ledger_lines]
    migration_events = [
        entry
        for entry in ledger_entries
        if entry.get("type") == "RUN_MIGRATED"
        and entry.get("payload", {}).get("changes") == ["capsule_add_witness_id"]
    ]
    assert migration_events


def test_bool_task_pass(tmp_path: Path) -> None:
    settings = Settings()
    run_dir = tmp_path / "bool_pass"
    summary = episode_run(
        task_file=Path("examples/tasks/bool_01.json"),
        run_dir=run_dir,
        settings=settings,
    )
    assert summary["verdict"] == "PASS"


def test_bool_task_fail_with_capsule(tmp_path: Path) -> None:
    settings = Settings()
    run_dir = tmp_path / "bool_fail"
    summary = episode_run(
        task_file=Path("examples/tasks/bool_fail_01.json"),
        run_dir=run_dir,
        settings=settings,
    )
    assert summary["verdict"] == "FAIL"
    capsule_dir = Path(summary["ucr_path"]).parent / "capsules"
    assert list(capsule_dir.glob("failure_*.json"))


def test_pyfunc_episode_pass(tmp_path: Path) -> None:
    settings = Settings()
    run_dir = tmp_path / "pyfunc_pass"
    summary = episode_run(
        task_file=Path("examples/tasks/pyfunc_01.json"),
        run_dir=run_dir,
        settings=settings,
    )
    assert summary["verdict"] == "PASS"
    artifact_dir = Path(summary["ucr_path"]).parent / "artifacts"
    program_path = artifact_dir / "pyfunc" / "program.py"
    assert program_path.exists()


def test_pyfunc_episode_fail(tmp_path: Path) -> None:
    settings = Settings()
    run_dir = tmp_path / "pyfunc_fail"
    summary = episode_run(
        task_file=Path("examples/tasks/pyfunc_fail_01.json"),
        run_dir=run_dir,
        settings=settings,
    )
    assert summary["verdict"] == "FAIL"
    assert summary["failure_reason"] != ""


def _pyexec_failure_atoms(run_dir: Path) -> list[str]:
    report_path = run_dir / "artifacts" / "reports" / "verifier.json"
    assert report_path.exists(), "verifier report missing"
    verdicts = read_json(report_path)
    for entry in verdicts:
        if entry.get("domain") == "pyfunc" and entry.get("tier") == "pyexec":
            return entry.get("failure_atoms", [])
    raise AssertionError("pyexec verdict missing")


def _pybreaker_failure_atoms(run_dir: Path) -> list[str]:
    report_path = run_dir / "artifacts" / "reports" / "verifier.json"
    assert report_path.exists(), "verifier report missing"
    verdicts = read_json(report_path)
    for entry in verdicts:
        if entry.get("domain") == "pyfunc" and entry.get("tier") == "breaker":
            return entry.get("failure_atoms", [])
    raise AssertionError("pyfunc breaker verdict missing")


def test_pyfunc_adversarial_failure_atoms(tmp_path: Path) -> None:
    settings = Settings()
    tasks = [
        ("pyfunc_unsafe_import_01.json", ["AST_FORBIDDEN_NODE"]),
        ("pyfunc_unsafe_open_01.json", ["AST_FORBIDDEN_CALL"]),
        ("pyfunc_unsafe_dunder_01.json", ["AST_FORBIDDEN_CALL"]),
        ("pyfunc_timeout_01.json", ["TIMEOUT"]),
        ("pyfunc_mem_01.json", ["RESOURCE_LIMIT"]),
    ]
    for task_file, expected_atoms in tasks:
        run_dir = tmp_path / task_file
        summary = episode_run(
            task_file=Path(f"examples/tasks/{task_file}"),
            run_dir=run_dir,
            settings=settings,
        )
        assert summary["verdict"] == "FAIL", task_file
        decision_path = run_dir / "forge" / "decision.json"
        if decision_path.exists():
            decision = read_json(decision_path)
            assert decision["decision"] != "ADMIT"
        failure_atoms = _pyexec_failure_atoms(run_dir)
        for atom in expected_atoms:
            assert atom in failure_atoms


def test_pyfunc_breaker_trap_capsule(tmp_path: Path) -> None:
    settings = Settings()
    run_dir = tmp_path / "pyfunc_breaker_trap"
    summary = episode_run(
        task_file=Path("examples/tasks/pyfunc_breaker_trap_01.json"),
        run_dir=run_dir,
        settings=settings,
    )
    assert summary["verdict"] == "FAIL"

    breaker_atoms = _pybreaker_failure_atoms(run_dir)
    assert "COUNTEREXAMPLE_FOUND" in breaker_atoms
    assert "COUNTEREXAMPLE_FOUND:BREAKERV1" in breaker_atoms

    capsule_dir = run_dir / "capsules"
    capsule_path = next(capsule_dir.glob("failure_*.json"))
    capsule = read_json(capsule_path)
    assert "COUNTEREXAMPLE_FOUND" in capsule.get("failure_atoms", [])
    counterexample = capsule.get("counterexample")
    assert counterexample, "expected counterexample in capsule"
    inputs = counterexample.get("inputs", {})
    assert inputs.get("a") == 0
    assert inputs.get("b") == 1


def test_suite_run_reports(tmp_path: Path) -> None:
    suite_file = tmp_path / "suite.json"
    suite_payload = {
        "suite_id": "test_suite",
        "tasks": [
            {"task_file": "examples/tasks/arith_01.json"},
            {"task_file": "examples/tasks/arith_02.json"},
            {"task_file": "examples/tasks/arith_fail_01.json"},
            {"task_file": "examples/tasks/bool_01.json"},
            {"task_file": "examples/tasks/bool_fail_01.json"},
            {"task_file": "examples/tasks/arith_sealed_trap_01.json"},
        ],
    }
    suite_file.write_text(orjson.dumps(suite_payload).decode("utf-8"), encoding="utf-8")

    out_dir = tmp_path / "suite_out"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            str(suite_file),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0

    report = read_json(out_dir / "report.json")
    assert report["suite_id"] == "test_suite"
    totals = report["totals"]
    assert totals["pass"] >= 2
    assert totals["fail"] >= 3
    assert totals["audit_failures"] == 0
    assert report["store_updates"]
    for task in report["per_task"]:
        assert task["audit_ok"] is True
    manifest = read_json(out_dir / "store" / "manifest.json")
    programs = manifest.get("programs", {})
    assert programs
    assert len(report["store_updates"]) == len(programs)
    assert any(entry.get("admitted_count", 0) > 1 for entry in programs.values())


def test_suite_report_norm_written(tmp_path: Path) -> None:
    suite_file = tmp_path / "suite_norm.json"
    suite_payload = {
        "suite_id": "norm_suite",
        "tasks": [
            {"task_file": "examples/tasks/arith_01.json"},
            {"task_file": "examples/tasks/bool_01.json"},
        ],
    }
    suite_file.write_text(orjson.dumps(suite_payload).decode("utf-8"), encoding="utf-8")

    runner = CliRunner()
    out_dir = tmp_path / "suite_norm"
    result = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            str(suite_file),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0
    report = read_json(out_dir / "report.json")
    norm = read_json(out_dir / "report.norm.json")
    assert norm == normalize_suite_report(report)
    text = (out_dir / "report.norm.json").read_text(encoding="utf-8")
    assert text.endswith("\n")


def test_suite_warm_start(tmp_path: Path) -> None:
    suite_file = tmp_path / "suite.json"
    suite_payload = {
        "suite_id": "warm_suite",
        "tasks": [
            {"task_file": "examples/tasks/arith_01.json"},
            {"task_file": "examples/tasks/arith_02.json"},
            {"task_file": "examples/tasks/bool_01.json"},
            {"task_file": "examples/tasks/bool_fail_01.json"},
        ],
    }
    suite_file.write_text(orjson.dumps(suite_payload).decode("utf-8"), encoding="utf-8")

    runner = CliRunner()
    out_a = tmp_path / "suite_a"
    result_a = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            str(suite_file),
            "--out-dir",
            str(out_a),
        ],
    )
    assert result_a.exit_code == 0
    report_a = read_json(out_a / "report.json")

    out_b = tmp_path / "suite_b"
    result_b = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            str(suite_file),
            "--out-dir",
            str(out_b),
            "--warm-start-store",
            str(out_a / "store"),
        ],
    )
    assert result_b.exit_code == 0
    report_b = read_json(out_b / "report.json")

    assert report_a["totals"]["pass"] == report_b["totals"]["pass"]
    assert report_a["totals"]["fail"] == report_b["totals"]["fail"]
    synth_a = sum(task["synth_ns"] for task in report_a["per_task"])
    synth_b = sum(task["synth_ns"] for task in report_b["per_task"])
    assert synth_b <= synth_a
    for task in report_b["per_task"]:
        if task["warm_start_store"]:
            assert task["synth_ns"] == 0
            assert task["warm_start_candidate_hash"]
            assert not task.get("warm_start_candidate_rejected", False)
    fail_task = next(task for task in report_b["per_task"] if task["task_id"] == "bool_fail_01")
    assert fail_task["verdict"] == "FAIL"


def test_domain_warm_start_reuse(tmp_path: Path) -> None:
    suite_file = tmp_path / "warm_domain_suite.json"
    suite_payload = {
        "suite_id": "warm_domain",
        "tasks": [
            {"task_file": "examples/tasks/arith_01.json"},
        ],
    }
    suite_file.write_text(orjson.dumps(suite_payload).decode("utf-8"), encoding="utf-8")

    runner = CliRunner()
    cold_out = tmp_path / "warm_domain_cold"
    result = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            str(suite_file),
            "--out-dir",
            str(cold_out),
        ],
    )
    assert result.exit_code == 0

    manifest = read_json(cold_out / "store" / "manifest.json")
    programs = manifest.get("programs", {})
    arith_hash = next(
        program_hash
        for program_hash, entry in programs.items()
        if entry.get("domain") == "arith"
    )

    settings = Settings(warm_start_store=str(cold_out / "store"))
    run_dir = tmp_path / "arith_02_warm_store"
    summary = episode_run(
        task_file=Path("examples/tasks/arith_02.json"),
        run_dir=run_dir,
        settings=settings,
    )

    assert summary["verdict"] == "PASS"
    assert summary["warm_start_store"]
    assert summary["synth_ns"] == 0
    assert summary["warm_start_candidate_hash"] == arith_hash
    assert not summary["warm_start_candidate_rejected"]


def test_store_audit_cmd(tmp_path: Path) -> None:
    suite_file = tmp_path / "store_suite.json"
    suite_payload = {
        "suite_id": "store_suite",
        "tasks": [
            {"task_file": "examples/tasks/arith_01.json"},
        ],
    }
    suite_file.write_text(orjson.dumps(suite_payload).decode("utf-8"), encoding="utf-8")

    runner = CliRunner()
    out_dir = tmp_path / "store_suite_out"
    result = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            str(suite_file),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0
    store_dir = out_dir / "store"
    audit_result = runner.invoke(
        app,
        ["store", "audit", "--store", str(store_dir)],
    )
    assert audit_result.exit_code == 0


def test_bool_fail_warm_start_unsat(tmp_path: Path) -> None:
    suite_file = tmp_path / "bool_suite.json"
    suite_payload = {
        "suite_id": "bool_suite",
        "tasks": [
            {"task_file": "examples/tasks/bool_01.json"},
        ],
    }
    suite_file.write_text(orjson.dumps(suite_payload).decode("utf-8"), encoding="utf-8")

    runner = CliRunner()
    out_dir = tmp_path / "bool_suite_out"
    result = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            str(suite_file),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0

    store_path = out_dir / "store"
    settings = Settings(warm_start_store=str(store_path))
    run_dir = tmp_path / "bool_fail"
    summary = episode_run(
        task_file=Path("examples/tasks/bool_fail_01.json"),
        run_dir=run_dir,
        settings=settings,
    )
    assert summary["verdict"] == "FAIL"
    assert summary["failure_reason"] == "EXAMPLE_CONTRADICT"
    assert not summary["warm_start_store"]


def test_suite_warm_start_store_persists(tmp_path: Path) -> None:
    suite_a = tmp_path / "suite_a.json"
    suite_a_payload = {
        "suite_id": "suite_warm_store_a",
        "tasks": [
            {"task_file": "examples/tasks/arith_01.json"},
        ],
    }
    suite_a.write_text(orjson.dumps(suite_a_payload).decode("utf-8"), encoding="utf-8")

    runner = CliRunner()
    out_a = tmp_path / "suite_warm_store_a"
    result_a = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            str(suite_a),
            "--out-dir",
            str(out_a),
        ],
    )
    assert result_a.exit_code == 0

    suite_b = tmp_path / "suite_b.json"
    suite_b_payload = {
        "suite_id": "suite_warm_store_b",
        "tasks": [
            {"task_file": "examples/tasks/arith_02.json"},
        ],
    }
    suite_b.write_text(orjson.dumps(suite_b_payload).decode("utf-8"), encoding="utf-8")

    out_b = tmp_path / "suite_warm_store_b"
    result_b = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            str(suite_b),
            "--out-dir",
            str(out_b),
            "--warm-start-store",
            str(out_a / "store"),
        ],
    )
    assert result_b.exit_code == 0
    report_b = read_json(out_b / "report.json")
    arith_entry = next(
        entry for entry in report_b["per_task"] if entry["task_id"] == "arith_02"
    )
    assert arith_entry["warm_start_store"]
    assert arith_entry["warm_start_provided"]
    assert arith_entry["synth_ns"] == 0
    candidate_hash = arith_entry["warm_start_candidate_hash"]
    assert candidate_hash
    assert not arith_entry.get("warm_start_candidate_rejected", False)
    manifest_b = read_json(out_b / "store" / "manifest.json")
    assert candidate_hash in manifest_b.get("programs", {})
    warm_store_file = out_b / "store" / "arith" / f"{candidate_hash}.json"
    assert warm_store_file.exists()
    manifest_entry = manifest_b["programs"][candidate_hash]
    assert Path(manifest_entry["store_path"]) == warm_store_file


def test_suite_pyfunc_warm_start(tmp_path: Path) -> None:
    runner = CliRunner()
    suite_file = Path("examples/suites/suite_v2.json")
    out_a = tmp_path / "suite_pyfunc_a"
    result_a = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            str(suite_file),
            "--out-dir",
            str(out_a),
        ],
    )
    assert result_a.exit_code == 0

    out_b = tmp_path / "suite_pyfunc_b"
    result_b = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            str(suite_file),
            "--out-dir",
            str(out_b),
            "--warm-start-store",
            str(out_a / "store"),
        ],
    )
    assert result_b.exit_code == 0
    report_b = read_json(out_b / "report.json")
    entry = next(task for task in report_b["per_task"] if task["task_id"] == "pyfunc_01")
    assert entry["warm_start_store"]
    assert entry["warm_start_provided"]
    assert entry["synth_ns"] == 0
    candidate_hash = entry["warm_start_candidate_hash"]
    assert candidate_hash
    warm_store_file = out_b / "store" / "pyfunc" / f"{candidate_hash}.py"
    assert warm_store_file.exists()
    manifest_b = read_json(out_b / "store" / "manifest.json")
    manifest_entry = manifest_b.get("programs", {}).get(candidate_hash, {})
    assert manifest_entry.get("store_path") == str(warm_store_file)


def test_suite_v3_baseline(tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "suite_v3"
    result = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            "examples/suites/suite_v3.json",
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0

    report_norm = read_json(out_dir / "report.norm.json")
    baseline = read_json(Path("examples/baselines/suite_v3.report.norm.json"))
    assert report_norm == baseline
    norm_text = (out_dir / "report.norm.json").read_text(encoding="utf-8")
    assert norm_text.endswith("\n")

    audit_result = runner.invoke(
        app,
        ["store", "audit", "--store", str(out_dir / "store")],
    )
    assert audit_result.exit_code == 0


def test_suite_v4_baseline(tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "suite_v4"
    result = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            "examples/suites/suite_v4.json",
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0

    report_norm = read_json(out_dir / "report.norm.json")
    baseline = read_json(Path("examples/baselines/suite_v4.report.norm.json"))
    assert report_norm == baseline
    norm_text = (out_dir / "report.norm.json").read_text(encoding="utf-8")
    assert norm_text.endswith("\n")

    audit_result = runner.invoke(
        app,
        ["store", "audit", "--store", str(out_dir / "store")],
    )
    assert audit_result.exit_code == 0


def test_suite_v3_warm_regression(tmp_path: Path) -> None:
    runner = CliRunner()
    out_a = tmp_path / "suite_v3_warm_a"
    out_b = tmp_path / "suite_v3_warm_b"
    result_a = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            "examples/suites/suite_v3.json",
            "--out-dir",
            str(out_a),
        ],
    )
    assert result_a.exit_code == 0

    result_b = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            "examples/suites/suite_v3.json",
            "--out-dir",
            str(out_b),
            "--warm-start-store",
            str(out_a / "store"),
        ],
    )
    assert result_b.exit_code == 0

    norm_path = out_b / "report.norm.json"
    assert norm_path.exists()
    norm_text = norm_path.read_text(encoding="utf-8")
    assert norm_text.endswith("\n")
    report_norm = read_json(norm_path)
    baseline = read_json(Path("examples/baselines/suite_v3_warm.report.norm.json"))
    assert report_norm == baseline

    audit_result = runner.invoke(
        app,
        ["store", "audit", "--store", str(out_b / "store")],
    )
    assert audit_result.exit_code == 0

    report = read_json(out_b / "report.json")
    entry = next(task for task in report["per_task"] if task["task_id"] == "pyfunc_01")
    assert entry["warm_start_provided"] is True
    assert entry["warm_start_store"] is True
    assert entry["synth_ns"] == 0
    assert entry["warm_start_candidate_hash"]
    assert entry["warm_start_candidate_hash"] == entry["program_hash"]


def test_suite_v2_store_audit_cold(tmp_path: Path) -> None:
    runner = CliRunner()
    suite_file = Path("examples/suites/suite_v2.json")
    out_dir = tmp_path / "suite_v2_cold"
    result = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            str(suite_file),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0

    store_dir = out_dir / "store"
    audit_result = runner.invoke(
        app,
        ["store", "audit", "--store", str(store_dir)],
    )
    assert audit_result.exit_code == 0


def test_suite_v2_store_audit_warm(tmp_path: Path) -> None:
    runner = CliRunner()
    suite_file = Path("examples/suites/suite_v2.json")
    out_a = tmp_path / "suite_v2_cold"
    result_a = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            str(suite_file),
            "--out-dir",
            str(out_a),
        ],
    )
    assert result_a.exit_code == 0

    out_b = tmp_path / "suite_v2_warm"
    result_b = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            str(suite_file),
            "--out-dir",
            str(out_b),
            "--warm-start-store",
            str(out_a / "store"),
        ],
    )
    assert result_b.exit_code == 0

    report_b = read_json(out_b / "report.json")
    entry = next(task for task in report_b["per_task"] if task["task_id"] == "pyfunc_01")
    assert entry["warm_start_store"]
    assert entry["warm_start_provided"]
    assert entry["warm_start_candidate_hash"]

    program_hash = entry["program_hash"]
    manifest = read_json(out_b / "store" / "manifest.json")
    manifest_entry = manifest.get("programs", {}).get(program_hash)
    assert manifest_entry
    store_path = Path(manifest_entry["store_path"])
    if not store_path.is_absolute():
        store_path = out_b / "store" / store_path
    assert store_path.exists()
    assert hash_bytes(store_path.read_bytes()) == program_hash

    audit_result = runner.invoke(
        app,
        ["store", "audit", "--store", str(out_b / "store")],
    )
    assert audit_result.exit_code == 0

def test_forge_admit_and_reject(tmp_path: Path) -> None:
    store_root = Path("store") / "v1" / "arith"
    if store_root.exists():
        shutil.rmtree(store_root)
    store_root.mkdir(parents=True, exist_ok=True)

    settings = Settings()
    pass_dir = tmp_path / "forge_pass"
    summary_pass = episode_run(
        task_file=Path("examples/tasks/arith_01.json"),
        run_dir=pass_dir,
        settings=settings,
    )
    ucr_pass = orjson.loads(Path(summary_pass["ucr_path"]).read_bytes())
    program_hash = ucr_pass["hashes"]["program_hash"]
    admitted_path = store_root / f"{program_hash}.json"
    assert admitted_path.exists()

    before_files = sorted(path.name for path in store_root.glob("*.json"))
    fail_dir = tmp_path / "forge_fail"
    episode_run(
        task_file=Path("examples/tasks/arith_sealed_trap_01.json"),
        run_dir=fail_dir,
        settings=settings,
    )
    after_files = sorted(path.name for path in store_root.glob("*.json"))
    assert before_files == after_files


def test_sealed_withheld_failure_produces_capsule(tmp_path: Path) -> None:
    settings = Settings()
    task_file = Path("examples/tasks/arith_sealed_trap_01.json")
    run_dir = tmp_path / "sealed_trap"

    summary = episode_run(task_file=task_file, run_dir=run_dir, settings=settings)
    assert summary["verdict"] == "FAIL"

    witness_data = orjson.loads(Path(summary["witness_path"]).read_bytes())
    report = witness_data.get("breaker_evidence", {}).get("report", {})
    withheld_hits = report.get("withheld_hits", 0)
    counterexample = report.get("counterexample")
    assert counterexample is not None or withheld_hits > 0

    capsule_dir = Path(summary["ucr_path"]).parent / "capsules"
    assert list(capsule_dir.glob("failure_*.json"))


def test_bg_revision_written_and_active_nodes_nonempty(tmp_path: Path) -> None:
    settings = Settings()
    task_file = Path("examples/tasks/arith_01.json")
    run_dir = tmp_path / "bg_run"

    episode_run(task_file=task_file, run_dir=run_dir, settings=settings)

    revisions_path = run_dir / "bg" / "revisions.jsonl"
    assert revisions_path.exists()
    assert revisions_path.read_text(encoding="utf-8").strip()

    ledger_entries = [
        orjson.loads(line)
        for line in (run_dir / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(entry.get("type") == "BG_OP_APPLIED" for entry in ledger_entries)

    ucr_data = orjson.loads((run_dir / "ucr.json").read_bytes())
    context_hash = compute_context_hash(ucr_data["bg_context"])
    policy_version = ucr_data["bg_context"].get("policy_version", settings.policy_version)
    active_view = BGEngine.replay(
        run_dir / "bg" / "revisions.jsonl", context_hash, policy_version
    )
    assert active_view.active_nodes


def test_demo_reports_have_required_keys() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["demo", "bg"])
    assert result.exit_code == 0
    bg_report = Path("runs/demo_bg/reports/bg_demo.json")
    bg_data = orjson.loads(bg_report.read_bytes())
    for key in [
        "active_view_hash",
        "replay_hash_match",
        "active_nodes_count",
        "ledger_ok",
    ]:
        assert key in bg_data
    assert bg_data["active_nodes_count"] > 0

    result = runner.invoke(app, ["demo", "breaker"])
    assert result.exit_code == 0
    breaker_report = Path("runs/demo_breaker/reports/breaker_demo.json")
    breaker_data = orjson.loads(breaker_report.read_bytes())
    for key in [
        "CDR",
        "TMR",
        "NOVN",
        "WFHR",
        "counterexample_count",
    ]:
        assert key in breaker_data

    result = runner.invoke(app, ["demo", "cost"])
    assert result.exit_code == 0
    cost_report = Path("runs/demo_cost/reports/cost_demo.json")
    cost_data = orjson.loads(cost_report.read_bytes())
    for key in [
        "total_time_ms",
        "total_pass",
        "pass_rate",
        "delta_pass_per_ms",
        "controller_overhead_ratio",
        "kill_switch_triggered",
    ]:
        assert key in cost_data
