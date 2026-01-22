import sys
from pathlib import Path

from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.store.audit import audit_store
from sovereidolon_v1.utils import read_json


def _write_suite(path: Path, tasks: list[str]) -> None:
    overrides = {
        "break_budget_attempts": 0,
        "verify_budget_steps": 5,
        "synth_budget": 1,
        "pyfunc_minimize_budget": 0,
    }
    payload = {
        "suite_id": path.stem,
        "tasks": [{"task_file": task, "overrides": overrides} for task in tasks],
    }
    path.write_text(
        __import__("json").dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _run_loop(
    tmp_path: Path,
    iters: int,
    stop_on_no_improve: int,
    proposer_script: Path,
    suite_file: Path,
    sealed_suite_file: Path,
    prefer_promotion_store: bool = False,
) -> Path:
    out_dir = tmp_path / f"loop_{iters}_{stop_on_no_improve}"
    promo_dir = tmp_path / f"promo_{iters}_{stop_on_no_improve}"
    runner = CliRunner()
    command = [
        "learn",
        "loop",
        "--suite-file",
        str(suite_file),
        "--sealed-suite-file",
        str(sealed_suite_file),
        "--out-dir",
        str(out_dir),
        "--promo-store",
        str(promo_dir),
        "--iters",
        str(iters),
        "--no-improve-patience",
        str(stop_on_no_improve),
        "--proposer",
        "subprocess",
        "--proposer-cmd",
        f"{sys.executable} {proposer_script}",
    ]
    if stop_on_no_improve > 0:
        command.append("--stop-on-no-improve")
    command.append("--write-history")
    if prefer_promotion_store:
        command.append("--prefer-promotion-store")
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.stdout
    return out_dir


def test_learn_loop_multi_iter_history(tmp_path: Path) -> None:
    suite_file = tmp_path / "suite.json"
    sealed_suite_file = tmp_path / "sealed.json"
    _write_suite(suite_file, ["examples/tasks/pyfunc_fail_01.json"])
    _write_suite(sealed_suite_file, ["examples/tasks/pyfunc_fail_01.json"])
    proposer_script = (
        Path(__file__).resolve().parent / "fixtures" / "proposer_heuristic_v1.py"
    )
    out_dir = _run_loop(
        tmp_path,
        iters=3,
        stop_on_no_improve=5,
        proposer_script=proposer_script,
        suite_file=suite_file,
        sealed_suite_file=sealed_suite_file,
        prefer_promotion_store=True,
    )
    history_path = out_dir / "history.jsonl"
    assert history_path.exists()
    history_lines = history_path.read_text(encoding="utf-8").splitlines()
    assert len(history_lines) == 3

    for idx in range(3):
        for phase in ("public", "propose", "sealed"):
            phase_dir = out_dir / f"iter{idx:03d}_{phase}"
            assert (phase_dir / "report.norm.json").exists()
            assert (phase_dir / "dataset_meta.json").exists()
            audit_report = audit_store(phase_dir / "store")
            assert audit_report["ok"] is True
    iter_summary = read_json(out_dir / "iter001_summary.json")
    assert iter_summary["promotion_writes_public"] >= 0
    assert iter_summary["promotion_writes_sealed"] >= 0
    assert iter_summary["promotion_upgrades"] >= 1


def test_learn_loop_stop_on_no_improve(tmp_path: Path) -> None:
    suite_file = tmp_path / "suite.json"
    sealed_suite_file = tmp_path / "sealed.json"
    _write_suite(suite_file, ["examples/tasks/pyfunc_01.json"])
    _write_suite(sealed_suite_file, ["examples/tasks/pyfunc_01.json"])
    proposer_script = Path(__file__).resolve().parent / "fixtures" / "proposer_echo.py"
    out_dir = _run_loop(
        tmp_path,
        iters=5,
        stop_on_no_improve=1,
        proposer_script=proposer_script,
        suite_file=suite_file,
        sealed_suite_file=sealed_suite_file,
    )
    history_path = out_dir / "history.jsonl"
    assert history_path.exists()
    history_lines = history_path.read_text(encoding="utf-8").splitlines()
    assert len(history_lines) == 1
    assert (out_dir / "iter000_public").exists()
    assert not (out_dir / "iter001_public").exists()
    loop_summary = read_json(out_dir / "loop_summary.json")
    assert loop_summary["stopped_early"] is True
    assert loop_summary["stop_reason"] == "NO_IMPROVE"
    iter_summary = read_json(out_dir / "iter000_summary.json")
    assert iter_summary["stopped_early"] is True
    assert iter_summary["stop_reason"] == "NO_IMPROVE"
