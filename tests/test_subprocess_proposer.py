import sys
from pathlib import Path

from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.utils import read_json


def _proposer_cmd() -> str:
    fixture = Path(__file__).resolve().parent / "fixtures" / "proposer_echo.py"
    return f"{sys.executable} {fixture}"


def _run_suite(tmp_path: Path, name: str, proposer: list[str]) -> Path:
    out_dir = tmp_path / name
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            "examples/suites/suite_v11_subprocess.json",
            "--out-dir",
            str(out_dir),
            *proposer,
        ],
    )
    assert result.exit_code == 0, result.stdout
    return out_dir


def test_subprocess_proposer_success_and_failure(tmp_path: Path) -> None:
    out_dir = _run_suite(
        tmp_path,
        "subprocess_run",
        ["--proposer", "subprocess", "--proposer-cmd", _proposer_cmd()],
    )
    report = read_json(out_dir / "report.norm.json")
    per_task = {entry["task_id"]: entry for entry in report["per_task"]}
    assert per_task["pyfunc_01"]["verdict"] == "PASS"
    assert per_task["codepatch_pass_01"]["verdict"] == "PASS"
    assert per_task["jsonspec_pass_01"]["verdict"] == "PASS"
    assert per_task["pyfunc_fail_01"]["verdict"] == "FAIL"

    proposer_record = read_json(out_dir / "pyfunc_fail_01" / "proposer.json")
    assert proposer_record["error_atom"] == "EXCEPTION:PROPOSER_SUBPROCESS_FAILED"


def test_subprocess_proposer_proposals_jsonl_deterministic(tmp_path: Path) -> None:
    out_a = _run_suite(
        tmp_path,
        "subprocess_a",
        ["--proposer", "subprocess", "--proposer-cmd", _proposer_cmd()],
    )
    out_b = _run_suite(
        tmp_path,
        "subprocess_b",
        ["--proposer", "subprocess", "--proposer-cmd", _proposer_cmd()],
    )
    proposals_a = (out_a / "proposals.jsonl").read_bytes()
    proposals_b = (out_b / "proposals.jsonl").read_bytes()
    assert proposals_a == proposals_b


def test_subprocess_proposer_replay_matches_baseline(tmp_path: Path) -> None:
    out_dir = _run_suite(
        tmp_path,
        "subprocess_seed",
        ["--proposer", "subprocess", "--proposer-cmd", _proposer_cmd()],
    )
    proposals = out_dir / "proposals.jsonl"
    replay_dir = _run_suite(
        tmp_path,
        "subprocess_replay",
        ["--proposer", "replay", "--replay-proposals", str(proposals)],
    )
    replay_norm = read_json(replay_dir / "report.norm.json")
    baseline = read_json(Path("examples/baselines/suite_v11_subprocess.report.norm.json"))
    assert replay_norm == baseline
