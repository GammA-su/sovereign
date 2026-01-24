from pathlib import Path

import pytest

from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.orchestrator.task import load_task
from sovereidolon_v1.promotion_store import promote_artifact
from sovereidolon_v1.pyfunc.program import compute_pyfunc_hash
from sovereidolon_v1.codepatch.program import compute_codepatch_hash
from sovereidolon_v1.jsonspec.program import compute_jsonspec_hash
from sovereidolon_v1.utils import canonical_dumps, read_json

pytestmark = pytest.mark.slow


def _seed_promo_store(store_dir: Path) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)
    seeds = [
        Path("examples/tasks/pyfunc_fail_01.json"),
        Path("examples/tasks/jsonspec_fail_01.json"),
        Path("examples/tasks/codepatch_fail_01.json"),
    ]
    for task_file in seeds:
        task = load_task(task_file)
        domain = task.task_type
        spec_hash = task.spec_hash()
        meta_families = task.metadata.get(domain, {}).get("metamorphic", [])
        if not isinstance(meta_families, list):
            meta_families = []
        if domain == "pyfunc":
            program = str(task.metadata.get("pyfunc", {}).get("candidate_program", ""))
            program_hash = compute_pyfunc_hash(program)
            artifact_path = store_dir / f"{task.task_id}.py"
            artifact_path.write_text(program, encoding="utf-8")
            lane_id = "pyexec_meta" if meta_families else "pyexec"
        elif domain == "jsonspec":
            program_spec = task.metadata.get("jsonspec", {}).get("candidate_program", {})
            program_hash = compute_jsonspec_hash(program_spec)
            artifact_path = store_dir / f"{task.task_id}.json"
            artifact_path.write_bytes(canonical_dumps(program_spec))
            lane_id = "jsonspec_meta" if meta_families else "jsonspec_exec"
        else:
            program = str(task.metadata.get("codepatch", {}).get("candidate_patch", ""))
            program_hash = compute_codepatch_hash(program)
            artifact_path = store_dir / f"{task.task_id}.patch"
            artifact_path.write_text(program, encoding="utf-8")
            lane_id = "codepatch_meta" if meta_families else "codepatch_apply"
        promote_artifact(
            store_dir,
            domain=domain,
            program_hash=program_hash,
            artifact_path=artifact_path,
            spec_hash=spec_hash,
            lane_id=lane_id,
            families_mode="public",
            meta_families=meta_families,
            score={"score_scaled": 1},
            score_key=[1],
            score_scaled=1,
            admitted_by_run_id="seed",
        )


def _run_loop(out_dir: Path, promo_dir: Path) -> dict:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "learn",
            "loop",
            "--suite-file",
            "examples/suites/suite_v16_scale_medium.json",
            "--out-dir",
            str(out_dir),
            "--promo-store",
            str(promo_dir),
            "--proposer",
            "retrieval_repair_v1",
            "--iters",
            "2",
            "--write-history",
        ],
    )
    assert result.exit_code == 0, result.stdout
    return read_json(out_dir / "loop_summary.json")


def test_learn_loop_retrieval_repair_v1_smoke(tmp_path: Path) -> None:
    promo_a = tmp_path / "promo_a"
    _seed_promo_store(promo_a)
    out_a = tmp_path / "loop_a"
    summary_a = _run_loop(out_a, promo_a)
    assert summary_a.get("sealed_suite_id") == "sealed_v9_withheld_scale_medium"
    iterations_a = summary_a.get("iterations", [])
    assert iterations_a
    iter0 = iterations_a[0]
    assert iter0.get("withheld_survival_ok") is True
    propose = iter0.get("propose", {})
    assert int(propose.get("propose_repairs_applied_total", 0)) > 0

    promo_b = tmp_path / "promo_b"
    _seed_promo_store(promo_b)
    out_b = tmp_path / "loop_b"
    summary_b = _run_loop(out_b, promo_b)
    assert summary_a == summary_b
