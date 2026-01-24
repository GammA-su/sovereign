from pathlib import Path

from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.utils import read_json, write_json


def _write_tiny_suite(path: Path, task_file: str, suite_id: str) -> None:
    payload = {
        "suite_id": suite_id,
        "tasks": [{"task_file": task_file}],
    }
    write_json(path, payload)


def _write_tiny_sealed_suite(path: Path, task_file: str) -> None:
    payload = {
        "suite_id": "sealed_tiny_withheld_v2",
        "families_mode": "withheld_v2",
        "tasks": [{"task_file": task_file, "expected_verdict": "PASS"}],
    }
    write_json(path, payload)


def _run_loop(out_dir: Path, promo_dir: Path, suite_file: Path, sealed_file: Path) -> dict:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "learn",
            "loop",
            "--suite-file",
            str(suite_file),
            "--sealed-suite-file",
            str(sealed_file),
            "--out-dir",
            str(out_dir),
            "--promo-store",
            str(promo_dir),
            "--iters",
            "1",
            "--proposer",
            "retrieval",
        ],
    )
    assert result.exit_code == 0, result.stdout
    return read_json(out_dir / "loop_summary.json")


def test_learn_loop_retrieval_autowire(tmp_path: Path) -> None:
    suite_file = tmp_path / "tiny_suite.json"
    sealed_file = tmp_path / "tiny_sealed.json"
    _write_tiny_suite(suite_file, "examples/tasks/pyfunc_01.json", "tiny_suite")
    _write_tiny_sealed_suite(sealed_file, "examples/tasks/pyfunc_01.json")

    out_a = tmp_path / "loop_a"
    promo_a = tmp_path / "promo_a"
    summary_a = _run_loop(out_a, promo_a, suite_file, sealed_file)
    assert (out_a / "loop_summary.json").read_text(encoding="utf-8").endswith("\n")
    iterations = summary_a.get("iterations", [])
    assert iterations
    first_iter = iterations[0]
    propose = first_iter.get("propose", {})
    assert int(propose.get("propose_proposals_total", 0)) > 0
    resolved = first_iter.get("config", {}).get("resolved_retrieval_dataset", "")
    assert resolved.endswith("iter000_public/dataset.jsonl")
    assert first_iter.get("withheld_survival_ok") is True

    out_b = tmp_path / "loop_b"
    promo_b = tmp_path / "promo_b"
    summary_b = _run_loop(out_b, promo_b, suite_file, sealed_file)
    iter_b = summary_b.get("iterations", [])[0]
    assert int(iter_b.get("propose", {}).get("propose_proposals_total", 0)) == int(
        propose.get("propose_proposals_total", 0)
    )
    assert int(iter_b.get("propose", {}).get("unexpected_fail", 0)) == int(
        propose.get("unexpected_fail", 0)
    )
