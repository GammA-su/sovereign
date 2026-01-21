from pathlib import Path

from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.store.audit import audit_store
from sovereidolon_v1.utils import canonical_dumps, read_json


def test_sealed_v3_expected_verdicts(tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "sealed_v3"
    result = runner.invoke(
        app,
        [
            "suite",
            "run-sealed",
            "--suite-file",
            "examples/sealed/sealed_v3.json",
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0

    summary_path = out_dir / "sealed_summary.json"
    assert summary_path.exists()
    summary_text = summary_path.read_text(encoding="utf-8")
    assert summary_text.endswith("\n")
    summary = read_json(summary_path)
    assert summary.get("mismatches") == []
    assert summary.get("families_mode") == "sealed"
    assert "per_domain_kpi_averages" in summary
    assert audit_store(out_dir / "store")["ok"] is True

    canonical = canonical_dumps(summary)
    if not canonical.endswith(b"\n"):
        canonical += b"\n"
    assert summary_text.encode("utf-8") == canonical
