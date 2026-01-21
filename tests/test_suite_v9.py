from pathlib import Path

from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.store.audit import audit_store
from sovereidolon_v1.utils import read_json


def test_suite_v9_baseline(tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "suite_v9"
    result = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            "examples/suites/suite_v9.json",
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0

    norm_path = out_dir / "report.norm.json"
    assert norm_path.exists()
    assert norm_path.read_text(encoding="utf-8").endswith("\n")
    report_norm = read_json(norm_path)
    baseline = read_json(Path("examples/baselines/suite_v9.report.norm.json"))
    assert report_norm == baseline

    audit_report = audit_store(out_dir / "store")
    assert audit_report["ok"] is True

    per_task = report_norm.get("per_task", [])
    assert any(
        entry.get("controller_score_scaled", 0) != 0
        and entry.get("controller_policy_id", "")
        for entry in per_task
    )
