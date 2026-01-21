from pathlib import Path

from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.store.audit import audit_store
from sovereidolon_v1.utils import read_json


def _run_loop(out_root: Path, promotion_store: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "learn",
            "loop",
            "--suite-file",
            "examples/suites/suite_v3.json",
            "--sealed-suite-file",
            "examples/sealed/sealed_v3.json",
            "--out-root",
            str(out_root),
            "--iters",
            "1",
            "--promotion-store",
            str(promotion_store),
        ],
    )
    assert result.exit_code == 0


def test_learn_loop_deterministic(tmp_path: Path) -> None:
    out_root_a = tmp_path / "loop_a"
    promotion_a = tmp_path / "promotion_a"
    _run_loop(out_root_a, promotion_a)
    summary_path_a = out_root_a / "loop_summary.json"
    assert summary_path_a.exists()
    assert summary_path_a.read_text(encoding="utf-8").endswith("\n")
    summary_a = read_json(summary_path_a)

    sealed_summary = out_root_a / "iter_000" / "sealed" / "sealed_summary.json"
    assert sealed_summary.exists()

    for subdir in ("a", "b", "sealed"):
        audit_report = audit_store(out_root_a / "iter_000" / subdir / "store")
        assert audit_report["ok"] is True

    out_root_b = tmp_path / "loop_b"
    promotion_b = tmp_path / "promotion_b"
    _run_loop(out_root_b, promotion_b)
    summary_b = read_json(out_root_b / "loop_summary.json")
    assert summary_a == summary_b
