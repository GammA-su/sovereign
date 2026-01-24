from pathlib import Path

import pytest

from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.utils import read_json

pytestmark = pytest.mark.slow


def _run_loop(out_dir: Path, promo_dir: Path) -> dict:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "learn",
            "loop",
            "--suite-file",
            "examples/suites/suite_v14_ladder.json",
            "--sealed-suite-file",
            "examples/sealed/sealed_v6_ladder.json",
            "--out-dir",
            str(out_dir),
            "--promo-store",
            str(promo_dir),
            "--iters",
            "5",
            "--stop-on-no-improve",
            "--no-improve-patience",
            "2",
            "--write-history",
            "--proposer",
            "heuristic_v1",
        ],
    )
    assert result.exit_code == 0, result.stdout
    summary_path = out_dir / "loop_summary.json"
    assert summary_path.exists()
    return read_json(summary_path)


def test_ladder_compounds_multi_iter(tmp_path: Path) -> None:
    out_a = tmp_path / "loop_a"
    promo_a = tmp_path / "promo_a"
    summary_a = _run_loop(out_a, promo_a)
    iterations = summary_a.get("iterations", [])
    assert len(iterations) >= 2
    iter0 = iterations[0]
    iter1 = iterations[1]
    assert int(iter0.get("promotion_writes_public", 0)) > 0
    assert int(iter1.get("promotion_writes_public", 0)) > 0
    assert summary_a.get("stop_reason") == "NO_IMPROVE"

    last_iter = iterations[-1]
    sealed_metrics = last_iter.get("sealed", {})
    assert int(sealed_metrics.get("unexpected_fail", 0)) == 0

    out_b = tmp_path / "loop_b"
    promo_b = tmp_path / "promo_b"
    summary_b = _run_loop(out_b, promo_b)
    history_a = (out_a / "history.jsonl").read_text(encoding="utf-8")
    history_b = (out_b / "history.jsonl").read_text(encoding="utf-8")
    assert history_a == history_b
    assert summary_a == summary_b
