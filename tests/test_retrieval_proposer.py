from pathlib import Path

import orjson
from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.orchestrator.task import load_task
from sovereidolon_v1.proposer_api import RetrievalProposer
from sovereidolon_v1.utils import canonical_dumps, stable_hash


def _write_dataset(path: Path, records: list[dict[str, object]]) -> None:
    lines = [canonical_dumps(record) for record in records]
    blob = b"\n".join(lines)
    if not blob.endswith(b"\n"):
        blob += b"\n"
    path.write_bytes(blob)


def test_retrieval_proposer_picks_best(tmp_path: Path) -> None:
    task = load_task(Path("examples/tasks/pyfunc_01.json"))
    spec_signature = task.spec_signature()
    spec_hash = task.spec_hash()
    dataset_path = tmp_path / "train.jsonl"
    records = [
        {
            "domain": task.task_type,
            "spec_signature": spec_signature,
            "spec_hash": spec_hash,
            "verdict": "PASS",
            "controller_score_scaled": 5,
            "candidate_program": "def solve(a, b):\n    return a + b\n",
            "proposed_program_hash": "b" * 64,
        },
        {
            "domain": task.task_type,
            "spec_signature": spec_signature,
            "spec_hash": spec_hash,
            "verdict": "PASS",
            "controller_score_scaled": 10,
            "candidate_program": "def solve(a, b):\n    return a + b + 0\n",
            "proposed_program_hash": "a" * 64,
        },
    ]
    _write_dataset(dataset_path, records)
    proposer = RetrievalProposer(dataset_path)
    proposal = proposer.propose(
        task,
        domain=task.task_type,
        spec_signature=spec_signature,
        seed=0,
        max_tokens=None,
    )
    assert proposal.candidate_program == records[1]["candidate_program"]
    assert proposal.metadata.get("match_type") == "exact"


def test_retrieval_proposer_suite_v10_deterministic(tmp_path: Path) -> None:
    suite_file = Path("examples/suites/suite_v10.json")
    suite_data = orjson.loads(suite_file.read_bytes())
    task_files = [Path(entry["task_file"]) for entry in suite_data.get("tasks", [])]
    records = []
    for task_file in task_files:
        task = load_task(task_file)
        if task.task_type == "pyfunc":
            candidate_program = task.metadata.get("pyfunc", {}).get("candidate_program", "")
        elif task.task_type == "codepatch":
            candidate_program = task.metadata.get("codepatch", {}).get("candidate_patch", "")
        elif task.task_type == "jsonspec":
            candidate_spec = task.metadata.get("jsonspec", {}).get("candidate_program")
            if isinstance(candidate_spec, dict):
                candidate_program = canonical_dumps(candidate_spec).decode("utf-8")
            else:
                candidate_program = candidate_spec or ""
        else:
            candidate_program = ""
        if task.task_type in {"pyfunc", "codepatch", "jsonspec"} and candidate_program:
            records.append(
                {
                    "domain": task.task_type,
                    "spec_signature": task.spec_signature(),
                    "spec_hash": task.spec_hash(),
                    "verdict": "PASS",
                    "controller_score_scaled": 1,
                    "candidate_program": candidate_program,
                    "proposed_program_hash": stable_hash(candidate_program),
                }
            )
    dataset_path = tmp_path / "train.jsonl"
    _write_dataset(dataset_path, records)

    runner = CliRunner()
    out_a = tmp_path / "suite_a"
    out_b = tmp_path / "suite_b"
    args = [
        "suite",
        "run",
        "--suite-file",
        str(suite_file),
        "--proposer",
        "retrieval",
        "--retrieval-dataset",
        str(dataset_path),
    ]
    result_a = runner.invoke(app, [*args, "--out-dir", str(out_a)])
    assert result_a.exit_code == 0
    result_b = runner.invoke(app, [*args, "--out-dir", str(out_b)])
    assert result_b.exit_code == 0

    report_a = (out_a / "report.norm.json").read_text(encoding="utf-8")
    report_b = (out_b / "report.norm.json").read_text(encoding="utf-8")
    assert report_a == report_b
