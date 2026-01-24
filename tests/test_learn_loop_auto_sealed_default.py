from pathlib import Path

from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.utils import read_json, write_json


def test_learn_loop_auto_sealed_default(tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "loop_auto"
    promo_dir = tmp_path / "promo_auto"
    suite_file = tmp_path / "suite_v16_scale_medium_tiny.json"
    write_json(
        suite_file,
        {
            "suite_id": "suite_v16_scale_medium_tiny",
            "tasks": [{"task_file": "examples/tasks/pyfunc_01.json"}],
        },
    )
    result = runner.invoke(
        app,
        [
            "learn",
            "loop",
            "--suite-file",
            str(suite_file),
            "--out-dir",
            str(out_dir),
            "--promo-store",
            str(promo_dir),
            "--iters",
            "1",
            "--proposer",
            "heuristic_v1",
        ],
    )
    assert result.exit_code == 0, result.stdout
    summary = read_json(out_dir / "loop_summary.json")
    assert summary.get("sealed_suite_id") == "sealed_v9_withheld_scale_medium"
    resolved = summary.get("config", {}).get("resolved_sealed_suite_file", "")
    assert resolved.endswith("examples/sealed/sealed_v9_withheld_scale_medium.json")
    iterations = summary.get("iterations", [])
    assert iterations
    assert iterations[0].get("withheld_survival_ok") is True
