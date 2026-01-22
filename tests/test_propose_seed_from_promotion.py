from pathlib import Path

from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.utils import read_json, write_json


def test_propose_seed_from_promotion(tmp_path: Path) -> None:
    suite_file = tmp_path / "suite.json"
    overrides = {
        "break_budget_attempts": 0,
        "verify_budget_steps": 5,
        "synth_budget": 1,
        "pyfunc_minimize_budget": 0,
    }
    payload = {
        "suite_id": "seed_from_promotion",
        "tasks": [
            {
                "task_file": "examples/tasks/pyfunc_01.json",
                "overrides": overrides,
            }
        ],
    }
    write_json(suite_file, payload)

    promotion_store = tmp_path / "promo"
    runner = CliRunner()
    out_public = tmp_path / "public"
    result_public = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            str(suite_file),
            "--out-dir",
            str(out_public),
            "--promotion-store",
            str(promotion_store),
        ],
    )
    assert result_public.exit_code == 0, result_public.stdout

    out_propose = tmp_path / "propose"
    result_propose = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            str(suite_file),
            "--out-dir",
            str(out_propose),
            "--promotion-store",
            str(promotion_store),
            "--prefer-promotion-store",
            "--proposer",
            "heuristic_v1",
        ],
    )
    assert result_propose.exit_code == 0, result_propose.stdout

    report = read_json(out_propose / "report.json")
    entry = report.get("per_task", [])[0]
    assert entry.get("proposer_seed_source") == "promotion"
    assert entry.get("proposer_seed_program_hash") == entry.get("program_hash")
    assert entry.get("program_changed") is False
