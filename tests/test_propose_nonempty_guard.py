import sys
from pathlib import Path

from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.utils import write_json


def _write_suite(path: Path, task_file: str, expected: str | None = None) -> None:
    task = {"task_file": task_file}
    if expected is not None:
        task["expected_verdict"] = expected
    payload = {"suite_id": path.stem, "tasks": [task]}
    write_json(path, payload)


def test_propose_nonempty_guard(tmp_path: Path) -> None:
    suite_file = tmp_path / "suite.json"
    sealed_suite_file = tmp_path / "sealed.json"
    _write_suite(suite_file, "examples/tasks/pyfunc_fail_01.json")
    _write_suite(sealed_suite_file, "examples/tasks/pyfunc_01.json", "PASS")

    fixture = Path(__file__).resolve().parent / "fixtures" / "proposer_echo.py"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "learn",
            "loop",
            "--suite-file",
            str(suite_file),
            "--sealed-suite-file",
            str(sealed_suite_file),
            "--out-dir",
            str(tmp_path / "out"),
            "--promo-store",
            str(tmp_path / "promo"),
            "--proposer",
            "subprocess",
            "--proposer-cmd",
            f"{sys.executable} {fixture}",
        ],
    )
    assert result.exit_code != 0
    assert "zero programs" in result.stdout
