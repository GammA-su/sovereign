from pathlib import Path

from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.utils import read_json, write_json


def test_reuse_reporting_promotion_attempt(tmp_path: Path) -> None:
    suite_file = tmp_path / "suite.json"
    write_json(
        suite_file,
        {
            "suite_id": "reuse_promo_suite",
            "tasks": [{"task_file": "examples/tasks/pyfunc_01.json"}],
        },
    )
    promo_dir = tmp_path / "promotion_store"
    runner = CliRunner()
    out_a = tmp_path / "suite_a"
    result_a = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            str(suite_file),
            "--out-dir",
            str(out_a),
            "--promotion-store",
            str(promo_dir),
        ],
    )
    assert result_a.exit_code == 0

    out_b = tmp_path / "suite_b"
    result_b = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            str(suite_file),
            "--out-dir",
            str(out_b),
            "--promotion-store",
            str(promo_dir),
            "--prefer-promotion-store",
        ],
    )
    assert result_b.exit_code == 0

    report = read_json(out_b / "report.json")
    entry = report["per_task"][0]
    assert entry.get("promotion_attempted") is True
    promotion_best_hash = entry.get("promotion_best_hash", "")
    if promotion_best_hash:
        assert entry.get("promotion_used") is True
        assert entry.get("reuse_source") == "promotion"
    else:
        reject_atoms = entry.get("promotion_reject_reason_atoms", [])
        assert "PROMOTION_NOT_FOUND" in reject_atoms

    totals = report.get("totals", {})
    reject_counts = totals.get("reuse_reject_counts", {})
    assert reject_counts.get("spec_mismatch_total", 0) == 0
