from pathlib import Path

from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.utils import read_json


def test_learn_loop_withheld_gate(tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "loop_withheld"
    promo_dir = tmp_path / "promo_withheld"
    result = runner.invoke(
        app,
        [
            "learn",
            "loop",
            "--suite-file",
            "examples/suites/suite_v12_learn.json",
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
    assert summary.get("sealed_suite_id") == "sealed_v7_withheld_v1"
    iterations = summary.get("iterations", [])
    assert iterations
    first_iter = iterations[0]
    assert first_iter.get("withheld_survival_ok") is True
    assert int(first_iter.get("promotion_writes_sealed", 0)) >= 1
