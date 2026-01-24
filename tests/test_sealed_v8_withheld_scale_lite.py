from pathlib import Path

import pytest

from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.store.audit import audit_store
from sovereidolon_v1.utils import read_json

pytestmark = pytest.mark.slow


def test_sealed_v8_withheld_scale_lite_baseline(tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "sealed_v8"
    result = runner.invoke(
        app,
        [
            "suite",
            "run-sealed",
            "--suite-file",
            "examples/sealed/sealed_v8_withheld_scale_lite.json",
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.stdout
    norm_path = out_dir / "report.norm.json"
    assert norm_path.exists()
    assert norm_path.read_text(encoding="utf-8").endswith("\n")
    report_norm = read_json(norm_path)
    baseline = read_json(
        Path("examples/baselines/sealed_v8_withheld_scale_lite.report.norm.json")
    )
    assert report_norm == baseline
    sealed_summary = read_json(out_dir / "sealed_summary.json")
    assert sealed_summary.get("mismatches") == []
    audit_report = audit_store(out_dir / "store")
    assert audit_report["ok"] is True
