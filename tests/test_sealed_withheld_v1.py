from pathlib import Path

from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.utils import read_json, write_json


def test_sealed_withheld_v1(tmp_path: Path) -> None:
    promo_dir = tmp_path / "promotion_store"
    seed_suite = tmp_path / "seed_suite.json"
    sealed_suite_data = read_json(Path("examples/sealed/sealed_v7_withheld_v1.json"))
    write_json(
        seed_suite,
        {
            "suite_id": "seed_public",
            "tasks": sealed_suite_data.get("tasks", []),
        },
    )

    runner = CliRunner()
    out_public = tmp_path / "public"
    result_public = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            str(seed_suite),
            "--out-dir",
            str(out_public),
            "--promotion-store",
            str(promo_dir),
        ],
    )
    assert result_public.exit_code == 0

    out_sealed = tmp_path / "sealed"
    result_sealed = runner.invoke(
        app,
        [
            "suite",
            "run-sealed",
            "--suite-file",
            "examples/sealed/sealed_v7_withheld_v1.json",
            "--out-dir",
            str(out_sealed),
            "--promotion-store",
            str(promo_dir),
            "--prefer-promotion-store",
            "--prefer-promotion-tier",
            "sealed",
        ],
    )
    assert result_sealed.exit_code == 0

    report = read_json(out_sealed / "report.json")
    assert report.get("suite_id") == "sealed_v7_withheld_v1"
    assert report.get("families_mode") == "withheld_v1"

    totals = report.get("totals", {})
    assert int(totals.get("unexpected_fail", 0)) == 0
    assert int(totals.get("breaker_attempts", 0)) > 0

    per_task = report.get("per_task", [])
    assert any(entry.get("promotion_attempted") for entry in per_task)
    assert any(entry.get("promotion_used") for entry in per_task)
