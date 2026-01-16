from __future__ import annotations

from pathlib import Path

import orjson
from typer.testing import CliRunner

from sovereidolon_v1.bg.bg_engine import BGEngine, compute_context_hash
from sovereidolon_v1.breaker.breaker import BreakerLab
from sovereidolon_v1.bvps.cegis import run_cegis
from sovereidolon_v1.bvps.dsl import Program, binop, var
from sovereidolon_v1.bvps.interpreter import Interpreter
from sovereidolon_v1.cli import app
from sovereidolon_v1.config import Settings
from sovereidolon_v1.forge.forge import ForgeGate
from sovereidolon_v1.ledger.ledger import Ledger
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
    assert result.counterexamples
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
