from pathlib import Path

import orjson
from typer.testing import CliRunner

from sovereidolon_v1.cli import app


def _run_suite(out_dir: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            "examples/suites/suite_v7.json",
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0


def test_coverage_json_deterministic(tmp_path: Path) -> None:
    out_a = tmp_path / "suite_v7_a"
    out_b = tmp_path / "suite_v7_b"
    _run_suite(out_a)
    _run_suite(out_b)

    coverage_a = (out_a / "coverage.json").read_bytes()
    coverage_b = (out_b / "coverage.json").read_bytes()
    assert coverage_a.endswith(b"\n")
    assert coverage_b.endswith(b"\n")
    assert coverage_a == coverage_b

    payload = orjson.loads(coverage_a)
    per_domain = payload.get("per_domain", {})
    assert per_domain.get("arith", {}).get("attempts", 0) > 0
    assert per_domain.get("jsonspec", {}).get("attempts", 0) > 0
    assert per_domain.get("arith", {}).get("atoms_total", 0) >= 0
    assert payload.get("totals", {}).get("new_atoms", 0) >= 0


def test_coverage_sealed_families_used(tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "suite_v7_sealed"
    result = runner.invoke(
        app,
        [
            "suite",
            "run-sealed",
            "--suite-file",
            "examples/suites/suite_v7.json",
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0
    coverage = orjson.loads((out_dir / "coverage.json").read_bytes())
    sealed_families = coverage.get("sealed_families_used", [])
    assert "permute" in sealed_families
