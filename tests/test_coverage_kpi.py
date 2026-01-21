from pathlib import Path

from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.config import Settings
from sovereidolon_v1.coverage_ledger import CoverageLedger
from sovereidolon_v1.orchestrator.episode import episode_run
from sovereidolon_v1.orchestrator.task import load_task
from sovereidolon_v1.utils import canonical_dumps, read_json


def _write_replay_records(path: Path, task_files: list[Path]) -> None:
    records = []
    for task_file in task_files:
        task = load_task(task_file)
        candidate_program = (
            task.metadata.get("pyfunc", {}).get("candidate_program", "")
            if task.task_type == "pyfunc"
            else ""
        )
        records.append(
            {
                "task_id": task.task_id,
                "spec_signature": task.spec_signature(),
                "candidate_program": candidate_program,
                "proposer_id": "replay",
            }
        )
    lines = [canonical_dumps(record) for record in records]
    blob = b"\n".join(lines)
    if not blob.endswith(b"\n"):
        blob += b"\n"
    path.write_bytes(blob)


def test_coverage_kpi_deterministic_replay(tmp_path: Path) -> None:
    task_files = [Path("examples/tasks/pyfunc_01.json")]
    replay_path = tmp_path / "proposals.jsonl"
    _write_replay_records(replay_path, task_files)

    suite_file = tmp_path / "suite.json"
    suite_payload = {
        "suite_id": "coverage_replay",
        "tasks": [{"task_file": str(task_files[0])}],
    }
    suite_file.write_bytes(canonical_dumps(suite_payload))

    runner = CliRunner()
    out_a = tmp_path / "suite_a"
    out_b = tmp_path / "suite_b"
    args = [
        "suite",
        "run",
        "--suite-file",
        str(suite_file),
        "--proposer",
        "replay",
        "--replay-file",
        str(replay_path),
    ]
    result_a = runner.invoke(app, [*args, "--out-dir", str(out_a)])
    result_b = runner.invoke(app, [*args, "--out-dir", str(out_b)])
    assert result_a.exit_code == 0
    assert result_b.exit_code == 0

    coverage_a = (out_a / "coverage.json").read_bytes()
    coverage_b = (out_b / "coverage.json").read_bytes()
    assert coverage_a.endswith(b"\n")
    assert coverage_a == coverage_b


def test_controller_v3_coverage_gain_matches_ledger(tmp_path: Path) -> None:
    task_file = Path("examples/tasks/pyfunc_01.json")
    settings = Settings(policy_version="v3")
    summary = episode_run(
        task_file=task_file,
        run_dir=tmp_path / "run_v3",
        settings=settings,
        proposer=None,
    )
    run_dir = Path(summary["ucr_path"]).parent
    verifier_report = read_json(run_dir / "artifacts" / "reports" / "verifier.json")
    breaker_report = read_json(run_dir / "artifacts" / "reports" / "breaker.json")
    task = load_task(task_file)
    ledger = CoverageLedger()
    expected_gain = ledger.update_from_episode(
        ucr={},
        verifier=verifier_report,
        breaker=breaker_report,
        task=task,
    )
    controller = read_json(run_dir / "controller.json")
    score = controller.get("score", {})
    assert score.get("coverage_gain") == expected_gain
