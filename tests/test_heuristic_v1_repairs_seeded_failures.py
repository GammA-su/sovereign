from pathlib import Path

from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.orchestrator.task import load_task
from sovereidolon_v1.promotion_store import promote_artifact
from sovereidolon_v1.pyfunc.program import compute_pyfunc_hash
from sovereidolon_v1.utils import write_json


def test_heuristic_repairs_seeded_failures(tmp_path: Path) -> None:
    task_path = Path("examples/tasks/pyfunc_fail_01.json")
    task = load_task(task_path)
    bad_program = task.metadata.get("pyfunc", {}).get("candidate_program", "")
    program_hash = compute_pyfunc_hash(bad_program)
    promo_dir = tmp_path / "promo"
    artifact_path = tmp_path / "seed.py"
    artifact_path.write_text(bad_program, encoding="utf-8")
    promote_artifact(
        promo_dir,
        domain="pyfunc",
        program_hash=program_hash,
        artifact_path=artifact_path,
        spec_hash=task.spec_hash(),
        lane_id="pyexec",
        families_mode="public",
        meta_families=[],
        score={"score_scaled": 1},
        score_key=[1],
        score_scaled=1,
        admitted_by_run_id="seed",
    )

    suite_file = tmp_path / "suite.json"
    write_json(
        suite_file,
        {
            "suite_id": "seed_repair",
            "tasks": [{"task_file": str(task_path)}],
        },
    )

    out_dir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            str(suite_file),
            "--out-dir",
            str(out_dir),
            "--promotion-store",
            str(promo_dir),
            "--prefer-promotion-store",
            "--proposer",
            "heuristic_v1",
        ],
    )
    assert result.exit_code == 0, result.stdout
    report = (out_dir / "report.json").read_text(encoding="utf-8")
    assert "\"program_changed\":true" in report
    assert "\"verdict\":\"PASS\"" in report
