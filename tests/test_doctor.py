import json
import os
from pathlib import Path

from typer.testing import CliRunner

from sovereidolon_v1.cli import app


def test_doctor_reports_missing_in_fake_repo(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    ci_all = scripts_dir / "ci_all.sh"
    ci_all.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    os.chmod(ci_all, 0o755)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["doctor", "--repo-root", str(tmp_path), "--json"],
    )
    assert result.exit_code == 1
    data = json.loads(result.stdout)
    missing = data["missing"]
    assert "missing_script:scripts/ci_golden_suite.sh" in missing
    assert "missing_suite:examples/suites/suite_v1.json" in missing
    assert (
        "missing_baseline:examples/baselines/suite_v1.report.norm.json" in missing
    )
    assert "missing_suite:examples/suites/suite_v8.json" in missing
    assert (
        "missing_baseline:examples/baselines/suite_v8.report.norm.json" in missing
    )
    assert "missing_suite:examples/suites/suite_v9.json" in missing
    assert (
        "missing_baseline:examples/baselines/suite_v9.report.norm.json" in missing
    )
    assert "missing_suite:examples/suites/suite_v10.json" in missing
    assert (
        "missing_baseline:examples/baselines/suite_v10.report.norm.json" in missing
    )
    assert (
        "missing_baseline:examples/baselines/suite_v8_replay.report.norm.json" in missing
    )
    assert "missing_script:scripts/ci_golden_suite_v8.sh" in missing
    assert "missing_script:scripts/ci_golden_suite_v9.sh" in missing
    assert "missing_script:scripts/ci_golden_suite_v10.sh" in missing
    assert "missing_script:scripts/ci_golden_suite_v8_replay.sh" in missing
    assert "missing_script:scripts/ci_promo_smoke.sh" in missing
    assert "ci_all_missing_ref:scripts/ci_promo_smoke.sh" in missing
    assert "ci_all_missing_ref:scripts/ci_golden_suite_v9.sh" in missing
    assert "ci_all_missing_ref:scripts/ci_golden_suite_v10.sh" in missing
    assert "missing_sealed_seed" in missing
    assert "missing_gitignore:promoted_store/" in missing


def test_doctor_real_repo_ok() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["doctor", "--repo-root", str(repo_root), "--json"],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["ok"] is True
