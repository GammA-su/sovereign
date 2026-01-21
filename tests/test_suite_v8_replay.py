from pathlib import Path

import orjson
from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.store.audit import audit_store
from sovereidolon_v1.utils import read_json


def _run_suite(tmp_path: Path, out_name: str, extra_args: list[str]) -> Path:
    runner = CliRunner()
    out_dir = tmp_path / out_name
    args = [
        "suite",
        "run",
        "--suite-file",
        "examples/suites/suite_v8.json",
        "--out-dir",
        str(out_dir),
    ]
    args.extend(extra_args)
    result = runner.invoke(app, args)
    assert result.exit_code == 0
    return out_dir


def test_suite_v8_record_proposals(tmp_path: Path) -> None:
    record_path = tmp_path / "proposals.jsonl"
    _run_suite(tmp_path, "record_run", ["--record-proposals", str(record_path)])
    assert record_path.exists()
    data = record_path.read_bytes()
    assert data.endswith(b"\n")
    lines = [line for line in data.splitlines() if line]
    assert len(lines) == 3
    required_keys = {
        "task_id",
        "domain",
        "spec_signature",
        "proposer_id",
        "proposal_hash",
        "candidate_program",
    }
    for line in lines:
        entry = orjson.loads(line)
        assert required_keys.issubset(entry.keys())


def test_suite_v8_replay_determinism(tmp_path: Path) -> None:
    record_path = tmp_path / "proposals.jsonl"
    record_out = _run_suite(tmp_path, "record_run", ["--record-proposals", str(record_path)])
    replay_out = _run_suite(
        tmp_path,
        "replay_run",
        ["--proposer", "replay", "--replay-file", str(record_path)],
    )

    baseline = read_json(Path("examples/baselines/suite_v8_replay.report.norm.json"))
    replay_norm = read_json(replay_out / "report.norm.json")
    assert replay_norm == baseline

    record_baseline = read_json(Path("examples/baselines/suite_v8.report.norm.json"))
    record_norm = read_json(record_out / "report.norm.json")
    assert record_norm == record_baseline

    audit_report = audit_store(replay_out / "store")
    assert audit_report["ok"] is True
