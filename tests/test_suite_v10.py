from pathlib import Path

from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.store.audit import audit_store
from sovereidolon_v1.utils import read_json


def test_suite_v10_baseline(tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "suite_v10"
    result = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            "examples/suites/suite_v10.json",
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0

    norm_path = out_dir / "report.norm.json"
    assert norm_path.exists()
    assert norm_path.read_text(encoding="utf-8").endswith("\n")
    report_norm = read_json(norm_path)
    baseline = read_json(Path("examples/baselines/suite_v10.report.norm.json"))
    assert report_norm == baseline

    audit_report = audit_store(out_dir / "store")
    assert audit_report["ok"] is True

    report = read_json(out_dir / "report.json")
    for entry in report.get("per_task", []):
        assert entry.get("domain")
