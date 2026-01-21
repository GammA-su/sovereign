from pathlib import Path

import orjson
from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.config import Settings
from sovereidolon_v1.orchestrator.episode import episode_run
from sovereidolon_v1.orchestrator.task import load_task
from sovereidolon_v1.proposer_api import ReplayProposer
from sovereidolon_v1.utils import canonical_dumps, read_json, write_json


def _write_replay_records(path: Path, task_files: list[Path]) -> None:
    records = []
    for task_file in task_files:
        task = load_task(task_file)
        candidate_program = (
            task.metadata.get("pyfunc", {}).get("candidate_program", "")
            if task.task_type == "pyfunc"
            else ""
        )
        records.append(
            {
                "task_id": task.task_id,
                "spec_signature": task.spec_signature(),
                "candidate_program": candidate_program,
                "proposer_id": "replay",
            }
        )
    lines = [canonical_dumps(record) for record in records]
    blob = b"\n".join(lines)
    if not blob.endswith(b"\n"):
        blob += b"\n"
    path.write_bytes(blob)


def test_episode_train_record_deterministic(tmp_path: Path) -> None:
    task_file = Path("examples/tasks/pyfunc_01.json")
    replay_path = tmp_path / "proposals.jsonl"
    _write_replay_records(replay_path, [task_file])
    proposer = ReplayProposer(replay_path)

    settings = Settings()
    out_a = tmp_path / "run_a"
    summary_a = episode_run(
        task_file=task_file,
        run_dir=out_a,
        settings=settings,
        proposer=proposer,
    )
    record_a = read_json(Path(summary_a["ucr_path"]).parent / "train_record.json")

    out_b = tmp_path / "run_b"
    proposer_b = ReplayProposer(replay_path)
    summary_b = episode_run(
        task_file=task_file,
        run_dir=out_b,
        settings=settings,
        proposer=proposer_b,
    )
    record_b = read_json(Path(summary_b["ucr_path"]).parent / "train_record.json")

    assert record_a == record_b
    assert record_a["task_id"] == "pyfunc_01"
    assert record_a["domain"] == "pyfunc"
    assert record_a["spec_signature"]
    assert "proposer" in record_a
    assert "verdict" in record_a
    assert "costs" in record_a
    assert "controller_version" in record_a


def test_suite_dataset_jsonl_deterministic(tmp_path: Path) -> None:
    task_files = [
        Path("examples/tasks/pyfunc_01.json"),
        Path("examples/tasks/pyfunc_meta_pass_01.json"),
    ]
    replay_path = tmp_path / "proposals.jsonl"
    _write_replay_records(replay_path, task_files)

    suite_file = tmp_path / "suite.json"
    suite_payload = {
        "suite_id": "train_dataset",
        "tasks": [{"task_file": str(task_file)} for task_file in task_files],
    }
    write_json(suite_file, suite_payload)

    runner = CliRunner()
    out_a = tmp_path / "suite_a"
    result_a = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            str(suite_file),
            "--out-dir",
            str(out_a),
            "--proposer",
            "replay",
            "--replay-file",
            str(replay_path),
        ],
    )
    assert result_a.exit_code == 0

    out_b = tmp_path / "suite_b"
    result_b = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            str(suite_file),
            "--out-dir",
            str(out_b),
            "--proposer",
            "replay",
            "--replay-file",
            str(replay_path),
        ],
    )
    assert result_b.exit_code == 0

    dataset_a = (out_a / "dataset.jsonl").read_bytes()
    dataset_b = (out_b / "dataset.jsonl").read_bytes()
    assert dataset_a.endswith(b"\n")
    assert dataset_a == dataset_b

    parsed = [orjson.loads(line) for line in dataset_a.splitlines() if line]
    ids = [entry.get("task_id", "") for entry in parsed]
    assert ids == sorted(ids)
