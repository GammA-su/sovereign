from pathlib import Path

import pytest

from sovereidolon_v1.config import Settings
from sovereidolon_v1.orchestrator.episode import episode_run
from sovereidolon_v1.orchestrator.task import load_task
from sovereidolon_v1.proposer_retrieval_repair_v1 import RetrievalRepairProposerV1
from sovereidolon_v1.promotion_store import promote_artifact
from sovereidolon_v1.pyfunc.program import compute_pyfunc_hash
from sovereidolon_v1.codepatch.program import compute_codepatch_hash
from sovereidolon_v1.jsonspec.program import compute_jsonspec_hash
from sovereidolon_v1.utils import canonical_dumps


def _meta_families(task_meta: dict, domain: str) -> list[str]:
    if domain == "pyfunc":
        families = task_meta.get("pyfunc", {}).get("metamorphic", [])
    elif domain == "codepatch":
        families = task_meta.get("codepatch", {}).get("metamorphic", [])
    elif domain == "jsonspec":
        families = task_meta.get("jsonspec", {}).get("metamorphic", [])
    else:
        families = []
    return [name for name in families if isinstance(name, str)]


def _lane_id(domain: str, meta_families: list[str]) -> str:
    meta_required = bool(meta_families)
    if domain == "pyfunc":
        return "pyexec_meta" if meta_required else "pyexec"
    if domain == "codepatch":
        return "codepatch_meta" if meta_required else "codepatch_apply"
    if domain == "jsonspec":
        return "jsonspec_meta" if meta_required else "jsonspec_exec"
    return domain


def _seed_promotion(
    tmp_path: Path, task_file: Path, domain: str, spec_hash: str
) -> Path:
    promo_dir = tmp_path / "promotion_store"
    task = load_task(task_file)
    task_meta = task.metadata
    meta_families = _meta_families(task_meta, domain)
    lane_id = _lane_id(domain, meta_families)

    if domain == "pyfunc":
        program = str(task_meta.get("pyfunc", {}).get("candidate_program", ""))
        program_hash = compute_pyfunc_hash(program)
        artifact_path = tmp_path / f"{task.task_id}.py"
        artifact_path.write_text(program, encoding="utf-8")
    elif domain == "jsonspec":
        program_spec = task_meta.get("jsonspec", {}).get("candidate_program", {})
        program_hash = compute_jsonspec_hash(program_spec)
        artifact_path = tmp_path / f"{task.task_id}.json"
        artifact_path.write_bytes(canonical_dumps(program_spec))
    else:
        program = str(task_meta.get("codepatch", {}).get("candidate_patch", ""))
        program_hash = compute_codepatch_hash(program)
        artifact_path = tmp_path / f"{task.task_id}.patch"
        artifact_path.write_text(program, encoding="utf-8")

    promote_artifact(
        promo_dir,
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
    return promo_dir


@pytest.mark.parametrize(
    ("task_file", "domain"),
    [
        (Path("examples/tasks/pyfunc_fail_01.json"), "pyfunc"),
        (Path("examples/tasks/jsonspec_fail_01.json"), "jsonspec"),
        (Path("examples/tasks/codepatch_fail_01.json"), "codepatch"),
    ],
)
def test_retrieval_repair_v1_repairs_seed(tmp_path: Path, task_file: Path, domain: str) -> None:
    task = load_task(task_file)
    promo_dir = _seed_promotion(tmp_path, task_file, domain, task.spec_hash())
    proposer = RetrievalRepairProposerV1()
    run_dir = tmp_path / f"run_{task.task_id}"
    settings = Settings(
        promotion_store=str(promo_dir),
        prefer_promotion_store=True,
        promotion_write_enabled=False,
        break_budget_attempts=0,
    )
    summary = episode_run(
        task_file=task_file,
        run_dir=run_dir,
        settings=settings,
        proposer=proposer,
    )
    assert summary["verdict"] == "PASS"
    assert summary.get("proposer_kind") == "retrieval_repair_v1"
    assert summary.get("repair_kind")
    assert summary.get("repair_edits_count", 0) >= 1
    assert summary.get("failure_hint_used", 0) == 1
    assert summary.get("program_changed") is True
