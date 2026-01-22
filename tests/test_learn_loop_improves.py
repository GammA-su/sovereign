from pathlib import Path

from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.store.audit import audit_store
from sovereidolon_v1.utils import read_json


def _run_loop(out_dir: Path, promo_store: Path, iters: int) -> dict:
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
            str(promo_store),
            "--iters",
            str(iters),
            "--proposer",
            "heuristic_v1",
        ],
    )
    assert result.exit_code == 0, result.stdout
    summary_path = out_dir / "loop_summary.json"
    assert summary_path.exists()
    assert summary_path.read_text(encoding="utf-8").endswith("\n")
    return read_json(summary_path)


def test_learn_loop_improves(tmp_path: Path) -> None:
    out_a = tmp_path / "loop_a"
    promo_a = tmp_path / "promo_a"
    summary_a = _run_loop(out_a, promo_a, iters=2)

    first_iter = summary_a["iterations"][0]
    public_phase = first_iter["public"]
    propose_phase = first_iter["propose"]
    assert propose_phase["fail"] < public_phase["fail"]
    assert propose_phase["pass"] >= public_phase["pass"]

    for phase in ("iter000_public", "iter000_propose", "iter000_sealed"):
        audit_report = audit_store(out_a / phase / "store")
        assert audit_report["ok"] is True

    sealed_report = read_json(out_a / "iter000_sealed" / "report.json")
    assert sealed_report.get("families_mode") == "sealed"
    sealed_updates = sealed_report.get("store_updates", [])
    if isinstance(sealed_updates, list) and sealed_updates:
        index = read_json(promo_a / "index.json")
        tiers = [entry.get("tier") for entry in index.get("entries", {}).values()]
        assert "sealed" in tiers

    out_b = tmp_path / "loop_b"
    promo_b = tmp_path / "promo_b"
    summary_b = _run_loop(out_b, promo_b, iters=2)
    assert summary_a == summary_b

    second_iter = summary_a["iterations"][1]
    propose_phase = second_iter["propose"]
    assert propose_phase.get("propose_changed_tasks", 0) > 0
    propose_report = read_json(out_a / "iter001_propose" / "report.json")
    per_task = propose_report.get("per_task", [])
    assert any(
        entry.get("proposer_seed_source") == "promotion"
        for entry in per_task
        if isinstance(entry, dict)
    )


def test_learn_loop_unexpected_fail_improves(tmp_path: Path) -> None:
    out_dir = tmp_path / "loop_v10"
    promo_dir = tmp_path / "promo_v10"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "learn",
            "loop",
            "--suite-file",
            "examples/suites/suite_v10.json",
            "--sealed-suite-file",
            "examples/sealed/sealed_v3.json",
            "--out-dir",
            str(out_dir),
            "--promo-store",
            str(promo_dir),
            "--iters",
            "2",
            "--proposer",
            "heuristic_v1",
        ],
    )
    assert result.exit_code == 0, result.stdout
    summary = read_json(out_dir / "loop_summary.json")
    first_iter = summary["iterations"][0]
    public_phase = first_iter["public"]
    propose_phase = first_iter["propose"]
    assert "promotion_index_tier_counts" in first_iter
    improved = first_iter.get("improved")
    assert isinstance(improved, bool)
    public_uf = int(public_phase.get("unexpected_fail", 0))
    propose_uf = int(propose_phase.get("unexpected_fail", 0))
    if propose_uf == 0 and public_uf > 0:
        assert improved is True
        assert first_iter.get("stop_reason") == "NONE"
        assert int(first_iter.get("promotion_writes_public", 0)) >= 1
    else:
        expected = False
        if propose_uf < public_uf:
            expected = True
        elif propose_uf == public_uf:
            if int(propose_phase.get("controller_score_sum", 0)) > int(
                public_phase.get("controller_score_sum", 0)
            ):
                expected = True
        assert improved is expected
