from pathlib import Path

from sovereidolon_v1.config import Settings
from sovereidolon_v1.orchestrator.episode import episode_run
from sovereidolon_v1.orchestrator.task import load_task
from sovereidolon_v1.proposer_api import ReplayProposer, StaticProposer, SubprocessProposer
from sovereidolon_v1.utils import read_json, read_jsonl, write_json


def test_static_proposer_success(tmp_path: Path) -> None:
    task_file = Path("examples/tasks/pyfunc_01.json")
    task_data = read_json(task_file)
    candidate = task_data["metadata"]["pyfunc"]["candidate_program"]
    proposer = StaticProposer(candidate)
    run_dir = tmp_path / "static_run"
    summary = episode_run(
        task_file=task_file,
        run_dir=run_dir,
        settings=Settings(),
        proposer=proposer,
    )
    assert summary["verdict"] == "PASS"
    proposer_record = read_json(run_dir / "proposer.json")
    assert proposer_record["proposer_id"] == "static"


def test_replay_proposer_determinism(tmp_path: Path) -> None:
    task_file = Path("examples/tasks/pyfunc_01.json")
    task = load_task(task_file)
    task_data = read_json(task_file)
    candidate = task_data["metadata"]["pyfunc"]["candidate_program"]
    proposer = StaticProposer(candidate)

    run_a = tmp_path / "run_a"
    summary_a = episode_run(
        task_file=task_file,
        run_dir=run_a,
        settings=Settings(),
        proposer=proposer,
    )
    record = read_json(run_a / "proposer.json")
    record.update({"task_id": task.task_id, "spec_signature": task.spec_signature()})
    replay_path = tmp_path / "replay.json"
    write_json(replay_path, [record])

    replay = ReplayProposer(replay_path)
    run_b = tmp_path / "run_b"
    summary_b = episode_run(
        task_file=task_file,
        run_dir=run_b,
        settings=Settings(),
        proposer=replay,
    )
    assert summary_a["synth_ns"] == summary_b["synth_ns"]
    ucr_a = read_json(run_a / "ucr.json")
    ucr_b = read_json(run_b / "ucr.json")
    assert ucr_a["hashes"]["program_hash"] == ucr_b["hashes"]["program_hash"]
    assert read_json(run_a / "proposer.json") == read_json(run_b / "proposer.json")


def test_proposer_failure(tmp_path: Path) -> None:
    task_file = Path("examples/tasks/pyfunc_01.json")
    proposer = SubprocessProposer(["bash", "-c", "exit 1"], timeout_s=2.0)
    run_dir = tmp_path / "failure_run"
    summary = episode_run(
        task_file=task_file,
        run_dir=run_dir,
        settings=Settings(),
        proposer=proposer,
    )
    assert summary["verdict"] == "FAIL"
    proposer_record = read_json(run_dir / "proposer.json")
    assert proposer_record["error_atom"].startswith("PROPOSER_ERROR:")
    decision = read_json(run_dir / "forge" / "decision.json")
    assert decision["decision"] == "REJECT"
    assert decision["reason"] == "proposer_failed"
    ledger = read_jsonl(run_dir / "ledger.jsonl")
    assert not any(entry.get("type") == "FORGE_ADMIT" for entry in ledger)
    capsule_paths = sorted((run_dir / "capsules").glob("failure_*.json"))
    assert capsule_paths
    capsule = read_json(capsule_paths[0])
    assert any(atom.startswith("PROPOSER_ERROR:") for atom in capsule["failure_atoms"])
