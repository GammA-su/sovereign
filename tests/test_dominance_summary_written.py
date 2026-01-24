from pathlib import Path

from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.utils import read_json


def _assert_scalar(value: object) -> None:
    if isinstance(value, (str, int, bool)):
        return
    raise AssertionError(f"unexpected value type: {type(value)}")


def test_dominance_summary_written(tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "loop"
    promo_dir = tmp_path / "promo"
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

    dominance_path = out_dir / "dominance_summary.json"
    assert dominance_path.exists()
    assert dominance_path.read_text(encoding="utf-8").endswith("\n")
    data = read_json(dominance_path)
    assert data.get("schema_version") == "v1"
    assert "suite_id" in data
    assert "sealed_suite_id" in data
    assert "promotion_index_tier_counts" in data
    iterations = data.get("iterations", [])
    assert iterations
    for entry in iterations:
        _assert_scalar(entry.get("iter"))
        _assert_scalar(entry.get("withheld_survival_ok"))
        _assert_scalar(entry.get("promotion_upgrades"))
        _assert_scalar(entry.get("promotion_writes_public"))
        _assert_scalar(entry.get("promotion_writes_sealed"))
        for phase in ("public", "propose", "sealed"):
            phase_data = entry.get(phase, {})
            _assert_scalar(phase_data.get("unexpected_fail"))
            _assert_scalar(phase_data.get("controller_score_sum"))
