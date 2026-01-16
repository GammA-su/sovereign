from __future__ import annotations

from pathlib import Path

import orjson
from typer.testing import CliRunner

from sovereidolon_v1.bg.bg_engine import BGEngine, compute_context_hash
from sovereidolon_v1.breaker.breaker import BreakerLab
from sovereidolon_v1.bvps.cegis import run_cegis
from sovereidolon_v1.bvps.dsl import Program, binop, var
from sovereidolon_v1.bvps.interpreter import Interpreter
from sovereidolon_v1.cli import app, audit_run, bg_replay_cmd, verify_bg_replay
from sovereidolon_v1.config import Settings
from sovereidolon_v1.forge.forge import ForgeGate
from sovereidolon_v1.ledger.ledger import Ledger
from sovereidolon_v1.orchestrator.episode import episode_run
from sovereidolon_v1.orchestrator.specs import task_spec
from sovereidolon_v1.orchestrator.task import Example, Task
from sovereidolon_v1.schemas import BGRevisionOp, VerifierVerdict
from sovereidolon_v1.utils import stable_hash


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
    decision = gate.decide(task, program, verdicts, required_passed=True)
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
    decision = gate.decide(task, program, [], required_passed=True)
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
