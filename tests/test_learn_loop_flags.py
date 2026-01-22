from pathlib import Path

from typer.testing import CliRunner

from sovereidolon_v1.cli import app


def test_learn_loop_flags_parse(tmp_path: Path) -> None:
    out_dir = tmp_path / "loop_flags"
    promo_dir = tmp_path / "promo"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "learn",
            "loop",
            "--suite-file",
            "examples/suites/suite_v12_learn.json",
            "--sealed-suite-file",
            "examples/sealed/sealed_v4_learn.json",
            "--out-dir",
            str(out_dir),
            "--promo-store",
            str(promo_dir),
            "--stop-on-no-improve",
            "--write-history",
            "--no-improve-patience",
            "1",
            "--proposer",
            "heuristic_v1",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (out_dir / "history.jsonl").exists()


def test_learn_loop_flag_rejects_arg(tmp_path: Path) -> None:
    out_dir = tmp_path / "loop_flags_bad"
    promo_dir = tmp_path / "promo_bad"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "learn",
            "loop",
            "--suite-file",
            "examples/suites/suite_v12_learn.json",
            "--sealed-suite-file",
            "examples/sealed/sealed_v4_learn.json",
            "--out-dir",
            str(out_dir),
            "--promo-store",
            str(promo_dir),
            "--stop-on-no-improve",
            "1",
            "--proposer",
            "heuristic_v1",
        ],
    )
    assert result.exit_code != 0
