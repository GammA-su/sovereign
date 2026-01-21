from pathlib import Path

from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.config import Settings
from sovereidolon_v1.orchestrator.episode import episode_run
from sovereidolon_v1.orchestrator.task import load_task
from sovereidolon_v1.utils import read_json, write_json


def test_warm_start_rejects_spec_mismatch(tmp_path: Path) -> None:
    suite_file = tmp_path / "suite_pass.json"
    write_json(
        suite_file,
        {
            "suite_id": "spec_mismatch_pass",
            "tasks": [{"task_file": "examples/tasks/codepatch_pass_01.json"}],
        },
    )
    out_dir = tmp_path / "suite_pass"
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

    manifest = read_json(out_dir / "store" / "manifest.json")
    programs = manifest.get("programs", {})
    assert isinstance(programs, dict) and programs
    program_hash = sorted(programs.keys())[0]
    stored_spec_hash = programs[program_hash].get("spec_hash")
    assert stored_spec_hash

    settings = Settings(warm_start_store=str(out_dir / "store"))
    summary = episode_run(
        task_file=Path("examples/tasks/codepatch_fail_tests_01.json"),
        run_dir=tmp_path / "fail_run",
        settings=settings,
    )
    assert summary["warm_start_candidate_rejected"] is True
    assert "SPEC_MISMATCH" in summary.get("warm_start_reject_reason_atoms", [])
    assert summary.get("warm_start_candidate_hash") == program_hash

    fail_task = load_task(Path("examples/tasks/codepatch_fail_tests_01.json"))
    assert stored_spec_hash != fail_task.spec_hash()


def test_promotion_rejects_spec_mismatch(tmp_path: Path) -> None:
    promotion_store = tmp_path / "promotion_store"
    settings_pass = Settings(promotion_store=str(promotion_store))
    summary_pass = episode_run(
        task_file=Path("examples/tasks/codepatch_pass_01.json"),
        run_dir=tmp_path / "pass_run",
        settings=settings_pass,
    )
    assert summary_pass["verdict"] == "PASS"

    settings_fail = Settings(
        promotion_store=str(promotion_store),
        prefer_promotion_store=True,
    )
    summary_fail = episode_run(
        task_file=Path("examples/tasks/codepatch_fail_tests_01.json"),
        run_dir=tmp_path / "fail_run",
        settings=settings_fail,
    )
    assert summary_fail["warm_start_candidate_rejected"] is True
    assert "PROMOTION_SPEC_MISMATCH" in summary_fail.get(
        "promotion_reject_reason_atoms", []
    )
