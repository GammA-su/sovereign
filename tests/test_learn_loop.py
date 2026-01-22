import sys
from pathlib import Path

from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.store.audit import audit_store
from sovereidolon_v1.utils import read_json


def _run_loop(out_root: Path, promotion_store: Path) -> None:
    runner = CliRunner()
    proposer_script = Path(__file__).resolve().parent / "fixtures" / "proposer_echo.py"
    result = runner.invoke(
        app,
        [
            "learn",
            "loop",
            "--out-dir",
            str(out_root),
            "--promo-store",
            str(promotion_store),
            "--proposer",
            "subprocess",
            "--proposer-cmd",
            f"{sys.executable} {proposer_script}",
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

    sealed_summary = out_root_a / "iter000_sealed" / "sealed_summary.json"
    assert sealed_summary.exists()

    for subdir in ("iter000_public", "iter000_propose", "iter000_sealed"):
        audit_report = audit_store(out_root_a / subdir / "store")
        assert audit_report["ok"] is True

    sealed_report = read_json(out_root_a / "iter000_sealed" / "report.json")
    assert sealed_report.get("families_mode") == "sealed"
    assert sealed_report.get("is_sealed_run") is True
    sealed_updates = sealed_report.get("store_updates", [])
    if isinstance(sealed_updates, list) and sealed_updates:
        promo_index = read_json(promotion_a / "index.json")
        tiers = [entry.get("tier") for entry in promo_index.get("entries", {}).values()]
        assert "sealed" in tiers

    out_root_b = tmp_path / "loop_b"
    promotion_b = tmp_path / "promotion_b"
    _run_loop(out_root_b, promotion_b)
    summary_b = read_json(out_root_b / "loop_summary.json")
    assert summary_a == summary_b


def test_learn_loop_flag_alias_parity(tmp_path: Path) -> None:
    runner = CliRunner()
    fixture = Path(__file__).resolve().parent / "fixtures" / "proposer_heuristic_v1.py"
    legacy_dir = tmp_path / "legacy"
    legacy_promo = tmp_path / "legacy_promo"
    legacy_result = runner.invoke(
        app,
        [
            "learn",
            "loop",
            "--suite-file",
            "examples/suites/suite_v12_learn.json",
            "--sealed-suite-file",
            "examples/sealed/sealed_v4_learn.json",
            "--out-dir",
            str(legacy_dir),
            "--promo-store",
            str(legacy_promo),
            "--prefer-tier-public",
            "public",
            "--prefer-tier-sealed",
            "sealed",
            "--proposer",
            "subprocess",
            "--proposer-cmd",
            f"{sys.executable} {fixture}",
        ],
    )
    assert legacy_result.exit_code == 0, legacy_result.stdout

    alias_dir = tmp_path / "alias"
    alias_promo = tmp_path / "alias_promo"
    alias_result = runner.invoke(
        app,
        [
            "learn",
            "loop",
            "--suite-file",
            "examples/suites/suite_v12_learn.json",
            "--sealed-suite-file",
            "examples/sealed/sealed_v4_learn.json",
            "--out-dir",
            str(alias_dir),
            "--promotion-store",
            str(alias_promo),
            "--prefer-promotion-tier",
            "public",
            "--sealed-prefer-promotion-tier",
            "sealed",
            "--proposer",
            "subprocess",
            "--proposer-cmd",
            f"{sys.executable} {fixture}",
        ],
    )
    assert alias_result.exit_code == 0, alias_result.stdout

    legacy_summary = (legacy_dir / "loop_summary.json").read_text(encoding="utf-8")
    alias_summary = (alias_dir / "loop_summary.json").read_text(encoding="utf-8")
    assert legacy_summary == alias_summary

    for phase in ("iter000_public", "iter000_propose", "iter000_sealed"):
        legacy_audit = audit_store(legacy_dir / phase / "store")
        alias_audit = audit_store(alias_dir / phase / "store")
        assert legacy_audit["ok"] is True
        assert alias_audit["ok"] is True
