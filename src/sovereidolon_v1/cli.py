from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from .bg.bg_engine import BGEngine, compute_context_hash
from .config import Settings
from .ledger.ledger import Ledger
from .orchestrator.episode import episode_run
from .orchestrator.task import load_task
from .schemas import UCR, export_schemas
from .utils import ensure_dir, hash_bytes, read_json, write_json

app = typer.Typer(help="SOVEREIDOLON v1 CLI")
console = Console()

TASK_FILE_OPTION = typer.Option(..., "--task-file", exists=True)
RUN_DIR_OPTION = typer.Option(None, "--run-dir")
RUN_ID_OPTION = typer.Option(None, "--run-id")
CONFIG_OPTION = typer.Option(None, "--config")
RUN_DIR_REQUIRED_OPTION = typer.Option(..., "--run-dir", exists=True)
CONTEXT_OPTION = typer.Option(
    None,
    "--context",
    help="Context JSON file path or context name string (default if omitted).",
)
POLICY_OPTION = typer.Option("v1", "--policy-version")
SCHEMA_OUT_OPTION = typer.Option(Path("schemas"), "--out-dir")


episode_app = typer.Typer(help="Episode commands")
ledger_app = typer.Typer(help="Ledger commands")
bg_app = typer.Typer(help="Belief graph commands")
demo_app = typer.Typer(help="Demos")
schema_app = typer.Typer(help="Schema utilities")
run_app = typer.Typer(help="Run audits")


@app.callback()
def main() -> None:
    pass


def _load_settings(config: Optional[Path]) -> Settings:
    if config is None:
        return Settings()
    data = read_json(config)
    return Settings(**data)


def _ensure_task_file(task_path: Path) -> None:
    if task_path.exists():
        return
    if task_path.name == "horizon_bg_01.json":
        payload = {
            "task_id": "horizon_bg_01",
            "family": "bg",
            "task_type": "bg",
            "goal": "state_consistency",
            "inputs": {"state": "Map"},
            "output": "Bool",
            "bounds": {},
            "examples": [{"inputs": {"state": {"a": 1}}, "output": True}],
            "metadata": {"description": "BG horizon demo"},
            "sealed": {"canary_token": "CANARY_BG", "sealed_seed": 789, "withheld_families": []},
        }
        write_json(task_path, payload)


@episode_app.command("run")
def episode_run_cmd(
    task_file: Path = TASK_FILE_OPTION,
    run_dir: Optional[Path] = RUN_DIR_OPTION,
    run_id: Optional[str] = RUN_ID_OPTION,
    config: Optional[Path] = CONFIG_OPTION,
) -> None:
    settings = _load_settings(config)
    if run_dir is None:
        run_root = Path("runs")
        ensure_dir(run_root)
        run_name = run_id or f"run_{task_file.stem}"
        run_dir = run_root / run_name
    summary = episode_run(task_file=task_file, run_dir=run_dir, settings=settings)
    table = Table(title="Episode Summary")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in summary.items():
        table.add_row(key, str(value))
    console.print(table)


@ledger_app.command("verify")
def ledger_verify_cmd(run_dir: Path = RUN_DIR_REQUIRED_OPTION) -> None:
    ok, message = Ledger.verify_chain(run_dir / "ledger.jsonl")
    console.print({"ok": ok, "message": message})


@bg_app.command("replay")
def bg_replay_cmd(
    run_dir: Path = RUN_DIR_REQUIRED_OPTION,
    context: Optional[str] = CONTEXT_OPTION,
    policy_version: str = POLICY_OPTION,
) -> str:
    if context is None:
        context_data = {"run_id": run_dir.name, "context_name": "default"}
    else:
        context_path = Path(context)
        if context_path.exists():
            context_data = read_json(context_path)
        else:
            context_data = {"run_id": run_dir.name, "context_name": str(context)}
    if isinstance(context_data, dict):
        policy_version = context_data.get("policy_version", policy_version)
    context_hash = compute_context_hash(context_data)
    active_view = BGEngine.replay(
        run_dir / "bg" / "revisions.jsonl", context_hash, policy_version
    )
    console.print(
        {"active_view_hash": active_view.active_view_hash, "active_nodes": active_view.active_nodes}
    )
    return active_view.active_view_hash


def verify_bg_replay(run_dir: Path) -> dict[str, object]:
    ucr_path = run_dir / "ucr.json"
    if not ucr_path.exists():
        return {
            "ok": False,
            "ucr_hash": "",
            "replay_hash": "",
            "mismatch_reason": "UCR_MISSING",
        }
    ucr_data = read_json(ucr_path)
    bg_context = ucr_data.get("bg_context")
    if not bg_context:
        return {
            "ok": False,
            "ucr_hash": "",
            "replay_hash": "",
            "mismatch_reason": "MISSING_BG_CONTEXT",
        }
    active_view_hash = ucr_data.get("active_view_hash")
    if not active_view_hash:
        return {
            "ok": False,
            "ucr_hash": "",
            "replay_hash": "",
            "mismatch_reason": "MISSING_ACTIVE_VIEW_HASH",
        }
    ucr_hash = UCR(**ucr_data).stable_hash()
    policy_version = bg_context.get("policy_version", Settings().policy_version)
    context_hash = compute_context_hash(bg_context)
    active_view = BGEngine.replay(
        run_dir / "bg" / "revisions.jsonl", context_hash, policy_version
    )
    replay_hash = active_view.active_view_hash
    ok = replay_hash == active_view_hash
    report: dict[str, object] = {
        "ok": ok,
        "ucr_hash": ucr_hash,
        "replay_hash": replay_hash,
    }
    if not ok:
        report["mismatch_reason"] = "HASH_MISMATCH"
    return report


@bg_app.command("verify-replay")
def bg_verify_replay_cmd(run_dir: Path = RUN_DIR_REQUIRED_OPTION) -> None:
    report = verify_bg_replay(run_dir)
    console.print(report)
    if not report["ok"]:
        raise typer.Exit(code=1)


def audit_run(run_dir: Path) -> dict[str, object]:
    checks: dict[str, bool] = {}
    errors: list[str] = []

    ledger_path = run_dir / "ledger.jsonl"
    ucr_path = run_dir / "ucr.json"
    witness_dir = run_dir / "witnesses"

    required_files_ok = (
        ledger_path.exists()
        and ucr_path.exists()
        and witness_dir.exists()
        and any(witness_dir.glob("*.json"))
    )
    checks["required_files"] = required_files_ok
    if not required_files_ok:
        errors.append("required_files_missing")

    ledger_ok = False
    if ledger_path.exists():
        ledger_ok, message = Ledger.verify_chain(ledger_path)
        if not ledger_ok:
            errors.append(f"ledger_invalid:{message}")
    else:
        errors.append("ledger_missing")
    checks["ledger_chain"] = ledger_ok

    ucr_data: dict[str, Any] = {}
    if ucr_path.exists():
        ucr_data = read_json(ucr_path)
    required_keys = [
        "schema_version",
        "run_id",
        "task_id",
        "bg_context",
        "active_view_hash",
        "hashes",
    ]
    ucr_keys_ok = all(key in ucr_data for key in required_keys)
    ucr_keys_ok = ucr_keys_ok and bool(ucr_data.get("bg_context")) and bool(
        ucr_data.get("active_view_hash")
    )
    checks["ucr_required_keys"] = ucr_keys_ok
    if not ucr_keys_ok:
        errors.append("ucr_missing_required_keys")

    witness_id = ""
    if isinstance(ucr_data.get("hashes"), dict):
        witness_id = str(ucr_data["hashes"].get("witness_id", ""))
    witness_path = witness_dir / f"{witness_id}.json" if witness_id else None
    if witness_path is None or not witness_path.exists():
        witness_files = list(witness_dir.glob("*.json"))
        witness_path = witness_files[0] if witness_files else None
        if witness_path is None:
            errors.append("witness_missing")

    artifact_ok = True
    witness_data: dict[str, Any] = {}
    if witness_path is not None and witness_path.exists():
        witness_data = read_json(witness_path)
        artifacts = witness_data.get("artifacts", [])
        for artifact in artifacts:
            path = Path(str(artifact.get("path", "")))
            expected_hash = str(artifact.get("content_hash", ""))
            expected_bytes = int(artifact.get("bytes", 0))
            if not path.exists():
                artifact_ok = False
                errors.append(f"artifact_missing:{path}")
                continue
            data = path.read_bytes()
            if expected_hash and hash_bytes(data) != expected_hash:
                artifact_ok = False
                errors.append(f"artifact_hash_mismatch:{path}")
            if expected_bytes and len(data) != expected_bytes:
                artifact_ok = False
                errors.append(f"artifact_size_mismatch:{path}")
    else:
        artifact_ok = False
        errors.append("witness_missing")
    checks["artifact_hashes"] = artifact_ok

    bg_report = verify_bg_replay(run_dir)
    bg_ok = bool(bg_report.get("ok"))
    checks["bg_replay"] = bg_ok
    if not bg_ok:
        errors.append(str(bg_report.get("mismatch_reason", "bg_replay_failed")))

    capsule_ok = True
    if witness_data.get("overall_verdict") == "FAIL":
        capsule_files = list((run_dir / "capsules").glob("failure_*.json"))
        if not capsule_files:
            capsule_ok = False
            errors.append("capsule_missing")
        else:
            matched = False
            for capsule_path in capsule_files:
                capsule = read_json(capsule_path)
                if witness_id and capsule.get("witness_id") == witness_id:
                    matched = True
                    break
            if witness_id and not matched:
                capsule_ok = False
                errors.append("capsule_missing_witness_id")
    checks["capsule"] = capsule_ok

    ok = all(checks.values()) and not errors
    return {"ok": ok, "checks": checks, "errors": errors}


@run_app.command("audit")
def run_audit_cmd(run_dir: Path = RUN_DIR_REQUIRED_OPTION) -> None:
    report = audit_run(run_dir)
    console.print(report)
    if not report["ok"]:
        raise typer.Exit(code=1)


@demo_app.command("bg")
def demo_bg() -> None:
    run_dir = Path("runs") / "demo_bg"
    ensure_dir(run_dir / "reports")
    task_path = Path("examples/tasks/horizon_bg_01.json")
    _ensure_task_file(task_path)
    settings = Settings()
    summary = episode_run(task_file=task_path, run_dir=run_dir, settings=settings)
    episode_dir = Path("runs") / summary["run_id"]
    ucr_data = read_json(episode_dir / "ucr.json")
    context_data = ucr_data.get(
        "bg_context", {"run_id": episode_dir.name, "context_name": "default"}
    )
    context_hash = compute_context_hash(context_data)
    active_view = BGEngine.replay(
        episode_dir / "bg" / "revisions.jsonl", context_hash, settings.policy_version
    )
    ledger_ok, _ = Ledger.verify_chain(episode_dir / "ledger.jsonl")
    report_path = run_dir / "reports" / "bg_demo.json"
    report = {
        "active_view_hash": active_view.active_view_hash,
        "replay_hash_match": ucr_data.get("active_view_hash") == active_view.active_view_hash,
        "active_nodes_count": len(active_view.active_nodes),
        "ledger_ok": ledger_ok,
    }
    write_json(report_path, report)
    ledger = Ledger(episode_dir / "ledger.jsonl")
    ledger.append("DEMO_REPORT", {"demo": "bg", "path": str(report_path)})
    console.print({"report": str(report_path)})


@demo_app.command("breaker")
def demo_breaker() -> None:
    run_dir = Path("runs") / "demo_breaker"
    ensure_dir(run_dir / "reports")
    task_path = Path("examples/tasks/list_01.json")
    settings = Settings(break_budget_attempts=500)
    summary = episode_run(task_file=task_path, run_dir=run_dir, settings=settings)
    episode_dir = Path("runs") / summary["run_id"]
    kpi = read_json(episode_dir / "artifacts" / "reports" / "breaker_kpi.json")
    breaker_report = read_json(episode_dir / "artifacts" / "reports" / "breaker.json")
    counterexample_count = 1 if breaker_report.get("counterexample") else 0
    minimal_repro = breaker_report.get("minimized")
    task = load_task(task_path)
    if task.sealed and minimal_repro is not None:
        if task.sealed.canary_token in json.dumps(minimal_repro, sort_keys=True):
            minimal_repro = "REDACTED"
    report_path = run_dir / "reports" / "breaker_demo.json"
    report = {
        "CDR": kpi.get("CDR", 0.0),
        "TMR": kpi.get("TMR", 0.0),
        "NOVN": kpi.get("NOVN", 0.0),
        "WFHR": kpi.get("WFHR", 0.0),
        "counterexample_count": counterexample_count,
        "minimal_repro_example": minimal_repro,
    }
    write_json(report_path, report)
    ledger = Ledger(episode_dir / "ledger.jsonl")
    ledger.append("DEMO_REPORT", {"demo": "breaker", "path": str(report_path)})
    console.print({"report": str(report_path)})


@demo_app.command("cost")
def demo_cost() -> None:
    run_dir = Path("runs") / "demo_cost"
    ensure_dir(run_dir / "reports")
    ledger = Ledger(run_dir / "ledger.jsonl")
    ledger.append("RUN_START", {"demo": "cost"})
    tasks = [
        Path("examples/tasks/arith_01.json"),
        Path("examples/tasks/list_01.json"),
    ]
    settings = Settings()
    episode_count = 5
    total_pass = 0
    total_time_ns = 0
    for idx in range(episode_count):
        task_path = tasks[idx % len(tasks)]
        episode_dir = run_dir / f"episode_{idx}"
        start = time.time_ns()
        summary = episode_run(task_file=task_path, run_dir=episode_dir, settings=settings)
        total_time_ns += time.time_ns() - start
        if summary["verdict"] == "PASS":
            total_pass += 1
        ledger.append(
            "DEMO_EPISODE",
            {"episode": idx, "task_file": str(task_path), "verdict": summary["verdict"]},
        )
    total_time_ms = total_time_ns / 1e6
    pass_rate = total_pass / episode_count if episode_count else 0.0
    delta_pass_per_ms = total_pass / total_time_ms if total_time_ms else 0.0
    report_path = run_dir / "reports" / "cost_demo.json"
    report = {
        "total_time_ms": total_time_ms,
        "total_pass": total_pass,
        "pass_rate": pass_rate,
        "delta_pass_per_ms": delta_pass_per_ms,
        "controller_overhead_ratio": 0.0,
        "kill_switch_triggered": False,
    }
    write_json(report_path, report)
    ledger.append("DEMO_REPORT", {"demo": "cost", "path": str(report_path)})
    console.print({"report": str(report_path)})


@schema_app.command("export")
def schema_export_cmd(out_dir: Path = SCHEMA_OUT_OPTION) -> None:
    export_schemas(str(out_dir))
    console.print({"schemas": str(out_dir)})


app.add_typer(episode_app, name="episode")
app.add_typer(ledger_app, name="ledger")
app.add_typer(bg_app, name="bg")
app.add_typer(demo_app, name="demo")
app.add_typer(schema_app, name="schema")
app.add_typer(run_app, name="run")

if __name__ == "__main__":
    app()
