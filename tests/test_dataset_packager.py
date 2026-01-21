from pathlib import Path

import orjson
from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.orchestrator.task import load_task
from sovereidolon_v1.utils import canonical_dumps, stable_hash, write_json


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


def _split_bucket(task_id: str) -> int:
    digest = stable_hash(task_id)
    return int(digest[:8], 16) % 10


def test_dataset_meta_train_val_split(tmp_path: Path) -> None:
    task_files = [
        Path("examples/tasks/pyfunc_01.json"),
        Path("examples/tasks/pyfunc_meta_pass_01.json"),
    ]
    replay_path = tmp_path / "proposals.jsonl"
    _write_replay_records(replay_path, task_files)

    suite_file = tmp_path / "suite.json"
    suite_payload = {
        "suite_id": "dataset_packager",
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

    for name in ["dataset.jsonl", "train.jsonl", "val.jsonl", "dataset_meta.json"]:
        path_a = out_a / name
        path_b = out_b / name
        assert path_a.exists()
        assert path_b.exists()
        text_a = path_a.read_text(encoding="utf-8")
        text_b = path_b.read_text(encoding="utf-8")
        assert text_a.endswith("\n")
        assert text_a == text_b

    dataset_lines = [
        orjson.loads(line)
        for line in (out_a / "dataset.jsonl").read_bytes().splitlines()
        if line
    ]
    train_lines = [
        orjson.loads(line)
        for line in (out_a / "train.jsonl").read_bytes().splitlines()
        if line
    ]
    val_lines = [
        orjson.loads(line)
        for line in (out_a / "val.jsonl").read_bytes().splitlines()
        if line
    ]
    task_ids = [entry.get("task_id", "") for entry in dataset_lines]
    assert task_ids == sorted(task_ids)

    expected_val = [
        entry for entry in dataset_lines if _split_bucket(str(entry.get("task_id", ""))) == 0
    ]
    expected_train = [
        entry for entry in dataset_lines if _split_bucket(str(entry.get("task_id", ""))) != 0
    ]
    assert len(val_lines) == len(expected_val)
    assert len(train_lines) == len(expected_train)

    meta = orjson.loads((out_a / "dataset_meta.json").read_bytes())
    assert meta["counts"]["total"] == len(dataset_lines)
    assert meta["counts"]["train"] == len(train_lines)
    assert meta["counts"]["val"] == len(val_lines)
    assert meta["fields_whitelist"] == sorted(meta["fields_whitelist"])
