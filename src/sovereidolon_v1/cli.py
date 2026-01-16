from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .bg.bg_engine import BGEngine, compute_context_hash
from .config import Settings
from .ledger.ledger import Ledger
from .orchestrator.episode import episode_run
from .orchestrator.task import load_task
from .schemas import export_schemas
from .utils import ensure_dir, read_json

app = typer.Typer(help="SOVEREIDOLON v1 CLI")
console = Console()

TASK_FILE_OPTION = typer.Option(..., "--task-file", exists=True)
RUN_DIR_OPTION = typer.Option(None, "--run-dir")
RUN_ID_OPTION = typer.Option(None, "--run-id")
CONFIG_OPTION = typer.Option(None, "--config")
RUN_DIR_REQUIRED_OPTION = typer.Option(..., "--run-dir", exists=True)
CONTEXT_OPTION = typer.Option(None, "--context")
POLICY_OPTION = typer.Option("v1", "--policy-version")
SCHEMA_OUT_OPTION = typer.Option(Path("schemas"), "--out-dir")


episode_app = typer.Typer(help="Episode commands")
ledger_app = typer.Typer(help="Ledger commands")
bg_app = typer.Typer(help="Belief graph commands")
demo_app = typer.Typer(help="Demos")
schema_app = typer.Typer(help="Schema utilities")


@app.callback()
def main() -> None:
    pass


def _load_settings(config: Optional[Path]) -> Settings:
    if config is None:
        return Settings()
    data = read_json(config)
    return Settings(**data)


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
    context: Optional[Path] = CONTEXT_OPTION,
    policy_version: str = POLICY_OPTION,
) -> None:
    context_data = {"run_id": run_dir.name}
    if context is not None:
        context_data = read_json(context)
    context_hash = compute_context_hash(context_data)
    active_view = BGEngine.replay(
        run_dir / "bg" / "revisions.jsonl", context_hash, policy_version
    )
    console.print(
        {"active_view_hash": active_view.active_view_hash, "active_nodes": active_view.active_nodes}
    )


@demo_app.command("bg")
def demo_bg() -> None:
    run_dir = Path("runs") / "demo_bg"
    ensure_dir(run_dir / "reports")
    task = load_task(Path("examples/tasks/horizon_bg_01.json"))
    report_path = run_dir / "reports" / "bg_demo.json"
    report = {
        "task_id": task.task_id,
        "note": "BG demo placeholder",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    console.print({"report": str(report_path)})


@demo_app.command("breaker")
def demo_breaker() -> None:
    run_dir = Path("runs") / "demo_breaker"
    ensure_dir(run_dir / "reports")
    report_path = run_dir / "reports" / "breaker_demo.json"
    report_path.write_text(
        json.dumps({"note": "breaker demo placeholder"}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    console.print({"report": str(report_path)})


@demo_app.command("cost")
def demo_cost() -> None:
    run_dir = Path("runs") / "demo_cost"
    ensure_dir(run_dir / "reports")
    report_path = run_dir / "reports" / "cost_demo.json"
    report_path.write_text(
        json.dumps({"note": "cost demo placeholder"}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
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

if __name__ == "__main__":
    app()
