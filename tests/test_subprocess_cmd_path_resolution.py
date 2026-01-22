import os
import sys
from pathlib import Path

from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.utils import read_json, write_json


def test_subprocess_cmd_path_resolution(tmp_path: Path) -> None:
    suite_file = tmp_path / "suite.json"
    repo_root = Path(__file__).resolve().parents[1]
    task_path = repo_root / "examples" / "tasks" / "pyfunc_01.json"
    suite_payload = {
        "suite_id": "subprocess_path",
        "tasks": [{"task_file": str(task_path)}],
    }
    write_json(suite_file, suite_payload)

    runner = CliRunner()
    fixture_rel = "tests/fixtures/proposer_echo.py"
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        out_dir = tmp_path / "out"
        result = runner.invoke(
            app,
            [
                "suite",
                "run",
                "--suite-file",
                str(suite_file),
                "--out-dir",
                str(out_dir),
                "--proposer",
                "subprocess",
                "--proposer-cmd",
                f"{sys.executable} {fixture_rel}",
            ],
        )
        assert result.exit_code == 0, result.stdout
    finally:
        os.chdir(cwd)

    proposals = (out_dir / "proposals.jsonl").read_text(encoding="utf-8").splitlines()
    assert proposals
    record = read_json(out_dir / "pyfunc_01" / "proposer.json")
    assert record["proposer_id"] == "subprocess"
