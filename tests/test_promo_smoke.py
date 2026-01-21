from pathlib import Path

from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.store.audit import audit_store
from sovereidolon_v1.utils import read_json


def test_promo_smoke_flow(tmp_path: Path) -> None:
    promo_store = tmp_path / "promoted_store" / "v1"
    out_a = tmp_path / "ci_promo_a"
    out_b = tmp_path / "ci_promo_b"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            "examples/suites/suite_v6.json",
            "--out-dir",
            str(out_a),
            "--promotion-store",
            str(promo_store),
        ],
    )
    assert result.exit_code == 0

    result = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            "examples/suites/suite_v6.json",
            "--out-dir",
            str(out_b),
            "--promotion-store",
            str(promo_store),
            "--prefer-promotion-store",
        ],
    )
    assert result.exit_code == 0

    report = read_json(out_b / "report.json")
    per_task = {entry["task_id"]: entry for entry in report.get("per_task", [])}
    for task_id in ["codepatch_pass_01", "pyfunc_01"]:
        entry = per_task.get(task_id, {})
        assert entry.get("verdict") == "PASS"
        assert entry.get("synth_ns") == 0
        assert entry.get("warm_start_store") is True
        assert entry.get("warm_start_candidate_hash") == entry.get("program_hash")

    assert audit_store(out_a / "store")["ok"] is True
    assert audit_store(out_b / "store")["ok"] is True
