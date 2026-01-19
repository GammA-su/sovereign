from pathlib import Path

from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.utils import read_json, write_json


def test_sealed_cli_writes_summary(tmp_path: Path) -> None:
    suite_file = tmp_path / "sealed_suite.json"
    suite_payload = {
        "suite_id": "sealed_test",
        "tasks": [{"task_file": "examples/tasks/arith_01.json"}],
    }
    write_json(suite_file, suite_payload)

    runner = CliRunner()
    out_a = tmp_path / "sealed_a"
    result = runner.invoke(
        app,
        [
            "suite",
            "run-sealed",
            "--suite-file",
            str(suite_file),
            "--out-dir",
            str(out_a),
        ],
    )
    assert result.exit_code == 0
    summary_a = read_json(out_a / "sealed_summary.json")
    assert summary_a["suite_id"] == "sealed_test"
    assert summary_a["tasks"][0]["task_id"] == "arith_01"

    out_b = tmp_path / "sealed_b"
    result_b = runner.invoke(
        app,
        [
            "suite",
            "run-sealed",
            "--suite-file",
            str(suite_file),
            "--out-dir",
            str(out_b),
        ],
    )
    assert result_b.exit_code == 0
    summary_b = read_json(out_b / "sealed_summary.json")
    assert summary_a == summary_b
