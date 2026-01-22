from pathlib import Path

from typer.testing import CliRunner

from sovereidolon_v1.cli import _is_phase_improved, _no_improve, app
from sovereidolon_v1.utils import read_json, write_json


def test_expected_verdict_kpis(tmp_path: Path) -> None:
    suite_file = tmp_path / "suite.json"
    write_json(
        suite_file,
        {
            "suite_id": "expected_kpis",
            "tasks": [
                {"task_file": "examples/tasks/pyfunc_01.json"},
                {
                    "task_file": "examples/tasks/pyfunc_fail_01.json",
                    "expected_verdict": "FAIL",
                },
            ],
        },
    )
    out_dir = tmp_path / "out"
    result = CliRunner().invoke(
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
    assert result.exit_code == 0, result.stdout
    report = read_json(out_dir / "report.json")
    totals = report.get("totals", {})
    assert totals.get("unexpected_fail") == 0
    assert totals.get("expected_fail") == 1

    public_metrics = {"unexpected_fail": 1, "unexpected_pass": 0}
    propose_metrics = {"unexpected_fail": 0, "unexpected_pass": 0}
    assert _no_improve(public_metrics, propose_metrics) is False
    assert _is_phase_improved(public_metrics, propose_metrics) is True
