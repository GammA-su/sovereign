from pathlib import Path

from sovereidolon_v1.config import Settings
from sovereidolon_v1.orchestrator.episode import episode_run
from sovereidolon_v1.orchestrator.task import load_task
from sovereidolon_v1.promotion_store import get_best_candidate
from sovereidolon_v1.proposer_api import StaticProposer
from sovereidolon_v1.utils import read_json


def _run_episode(
    *,
    task_file: Path,
    run_dir: Path,
    promotion_store: Path,
    break_budget_attempts: int,
    prefer_promotion_store: bool = False,
    proposer: StaticProposer | None = None,
) -> dict[str, object]:
    settings = Settings(
        promotion_store=str(promotion_store),
        break_budget_attempts=break_budget_attempts,
        prefer_promotion_store=prefer_promotion_store,
    )
    return episode_run(
        task_file=task_file,
        run_dir=run_dir,
        settings=settings,
        proposer=proposer,
    )


def test_promotion_store_first_admit_creates_index(tmp_path: Path) -> None:
    promotion_store = tmp_path / "promotion"
    task_file = Path("examples/tasks/arith_01.json")
    summary = _run_episode(
        task_file=task_file,
        run_dir=tmp_path / "run_a",
        promotion_store=promotion_store,
        break_budget_attempts=0,
    )
    assert summary["verdict"] == "PASS"
    task = load_task(task_file)
    entry = get_best_candidate(
        promotion_store,
        domain=task.task_type,
        spec_hash=task.spec_hash(),
        lane_id=task.task_type,
        families_mode="sealed" if task.sealed else "public",
        meta_families=[],
        meta_required=False,
    )
    assert entry is not None
    ucr = read_json(Path(summary["ucr_path"]))
    assert entry["program_hash"] == ucr.get("hashes", {}).get("program_hash")


def test_promotion_store_rejects_not_better(tmp_path: Path) -> None:
    promotion_store = tmp_path / "promotion"
    task_file = Path("examples/tasks/pyfunc_01.json")
    program_a = "def solve(a, b):\n    return a + b\n"
    program_b = "def solve(a, b):\n    return a + b  # warm\n"
    _run_episode(
        task_file=task_file,
        run_dir=tmp_path / "run_b",
        promotion_store=promotion_store,
        break_budget_attempts=0,
        proposer=StaticProposer(program_a),
    )
    summary = _run_episode(
        task_file=task_file,
        run_dir=tmp_path / "run_b_again",
        promotion_store=promotion_store,
        break_budget_attempts=0,
        proposer=StaticProposer(program_b),
    )
    assert summary["verdict"] == "FAIL"
    run_dir = Path(summary["ucr_path"]).parent
    controller = read_json(run_dir / "controller.json")
    assert "NOT_BETTER_THAN_BEST" in controller.get("reason_atoms", [])


def test_promotion_store_replaces_best_on_better_score(tmp_path: Path) -> None:
    promotion_store = tmp_path / "promotion"
    task_file = Path("examples/tasks/pyfunc_01.json")
    program_a = "def solve(a, b):\n    return a + b\n"
    program_b = "def solve(a, b):\n    return a + b  # better\n"
    summary_a = _run_episode(
        task_file=task_file,
        run_dir=tmp_path / "run_c",
        promotion_store=promotion_store,
        break_budget_attempts=10,
        proposer=StaticProposer(program_a),
    )
    run_a = Path(summary_a["ucr_path"]).parent
    controller_a = read_json(run_a / "controller.json")
    score_a = tuple(controller_a.get("score_key", []))

    summary_b = _run_episode(
        task_file=task_file,
        run_dir=tmp_path / "run_c_better",
        promotion_store=promotion_store,
        break_budget_attempts=0,
        proposer=StaticProposer(program_b),
    )
    assert summary_b["verdict"] == "PASS"
    run_b = Path(summary_b["ucr_path"]).parent
    controller_b = read_json(run_b / "controller.json")
    score_b = tuple(controller_b.get("score_key", []))
    assert score_b > score_a

    task = load_task(task_file)
    meta_families = task.metadata.get("pyfunc", {}).get("metamorphic", [])
    meta_families = [name for name in meta_families if isinstance(name, str)]
    entry = get_best_candidate(
        promotion_store,
        domain=task.task_type,
        spec_hash=task.spec_hash(),
        lane_id="pyexec_meta",
        families_mode="sealed" if task.sealed else "public",
        meta_families=meta_families,
        meta_required=True,
    )
    assert entry is not None
    assert tuple(entry.get("score_key", [])) == score_b


def test_promotion_store_warm_start_uses_best(tmp_path: Path) -> None:
    promotion_store = tmp_path / "promotion"
    task_file = Path("examples/tasks/arith_01.json")
    _run_episode(
        task_file=task_file,
        run_dir=tmp_path / "run_d",
        promotion_store=promotion_store,
        break_budget_attempts=10,
    )
    summary = _run_episode(
        task_file=task_file,
        run_dir=tmp_path / "run_d_warm",
        promotion_store=promotion_store,
        break_budget_attempts=0,
        prefer_promotion_store=True,
    )
    assert summary["synth_ns"] == 0
    assert summary["warm_start_store"] is True
