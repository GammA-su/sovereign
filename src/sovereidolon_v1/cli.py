from __future__ import annotations

import copy
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, List, Optional

import typer
from rich.console import Console
from rich.table import Table

from .bg.bg_engine import BGEngine, compute_context_hash
from .config import Settings
from .ledger.ledger import Ledger
from .orchestrator.episode import episode_run
from .orchestrator.task import load_task
from .proposer_api import BaseProposer, ReplayProposer, StaticProposer, SubprocessProposer
from .pyfunc.runner import PYEXEC_VERSION
from .schemas import UCR, export_schemas
from .store.audit import audit_store
from .utils import (
    canonical_dumps,
    ensure_dir,
    hash_bytes,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl_line,
)

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
WARM_START_OPTION = typer.Option(None, "--warm-start-store", exists=True, file_okay=False)
SUITE_WARM_START_OPTION = typer.Option(None, "--warm-start-store", exists=True, file_okay=False)
SUITE_FILE_OPTION = typer.Option(..., "--suite-file", exists=True)
OUT_DIR_OPTION = typer.Option(..., "--out-dir")
STORE_DIR_OPTION = typer.Option(..., "--store", exists=True, file_okay=False)
DOCTOR_REPO_ROOT_OPTION = typer.Option(None, "--repo-root")
DOCTOR_JSON_OPTION = typer.Option(False, "--json")
DOCTOR_SMOKE_OPTION = typer.Option(False, "--smoke")
PROPOSER_OPTION = typer.Option("stub", "--proposer")
STATIC_PROGRAM_OPTION = typer.Option(None, "--static-program")
REPLAY_FILE_OPTION = typer.Option(None, "--replay-file")
CMD_OPTION = typer.Option(None, "--cmd")
RECORD_PROPOSALS_OPTION = typer.Option(None, "--record-proposals")


episode_app = typer.Typer(help="Episode commands")
ledger_app = typer.Typer(help="Ledger commands")
bg_app = typer.Typer(help="Belief graph commands")
demo_app = typer.Typer(help="Demos")
schema_app = typer.Typer(help="Schema utilities")
run_app = typer.Typer(help="Run audits")
store_app = typer.Typer(help="Store commands")
suite_app = typer.Typer(help="Suites")


@app.callback()
def main() -> None:
    pass


def _load_settings(config: Optional[Path]) -> Settings:
    if config is None:
        return Settings()
    data = read_json(config)
    return Settings(**data)


def _load_static_program(value: str) -> str:
    path = Path(value)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return value


def _build_proposer(
    proposer_kind: str,
    static_program: Optional[str],
    replay_file: Optional[Path],
    cmd: Optional[List[str]],
) -> Optional[BaseProposer]:
    if proposer_kind in {"stub", "default"}:
        return None
    if proposer_kind == "static":
        if not static_program:
            raise typer.BadParameter("missing --static-program for static proposer")
        program_text = _load_static_program(static_program)
        return StaticProposer(program_text)
    if proposer_kind == "replay":
        if replay_file is None:
            raise typer.BadParameter("missing --replay-file for replay proposer")
        return ReplayProposer(replay_file)
    if proposer_kind == "subprocess":
        if not cmd:
            raise typer.BadParameter("missing --cmd for subprocess proposer")
        return SubprocessProposer(cmd)
    raise typer.BadParameter(f"unknown proposer: {proposer_kind}")


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


def _find_repo_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return start


@episode_app.command("run")
def episode_run_cmd(
    task_file: Path = TASK_FILE_OPTION,
    run_dir: Optional[Path] = RUN_DIR_OPTION,
    run_id: Optional[str] = RUN_ID_OPTION,
    config: Optional[Path] = CONFIG_OPTION,
    warm_start_store: Optional[Path] = WARM_START_OPTION,
    proposer_kind: str = PROPOSER_OPTION,
    static_program: Optional[str] = STATIC_PROGRAM_OPTION,
    replay_file: Optional[Path] = REPLAY_FILE_OPTION,
    cmd: Optional[List[str]] = CMD_OPTION,
) -> None:
    settings = _load_settings(config)
    if warm_start_store:
        settings = settings.model_copy(update={"warm_start_store": str(warm_start_store)})
    if run_dir is None:
        run_root = Path("runs")
        ensure_dir(run_root)
        run_name = run_id or f"run_{task_file.stem}"
        run_dir = run_root / run_name
    proposer = _build_proposer(proposer_kind, static_program, replay_file, cmd)
    summary = episode_run(
        task_file=task_file,
        run_dir=run_dir,
        settings=settings,
        proposer=proposer,
    )
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


def audit_run(run_dir: Path) -> dict[str, Any]:
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


def _load_first_witness(run_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    witnesses_dir = run_dir / "witnesses"
    if not witnesses_dir.exists():
        return None, {}
    witness_files = sorted(witnesses_dir.glob("*.json"))
    if not witness_files:
        return None, {}
    path = witness_files[0]
    return path, read_json(path)


def _infer_witness_id(run_dir: Path, ucr_data: dict[str, Any]) -> str:
    hashes = ucr_data.get("hashes", {})
    witness_id = str(hashes.get("witness_id") or "")
    if witness_id:
        return witness_id
    witness_file, witness_data = _load_first_witness(run_dir)
    if witness_file and witness_data:
        if witness_data.get("witness_id"):
            return str(witness_data["witness_id"])
        return witness_file.stem
    return ""


def _last_run_end_payload(run_dir: Path) -> dict[str, Any]:
    ledger_path = run_dir / "ledger.jsonl"
    if not ledger_path.exists():
        return {}
    entries = read_jsonl(ledger_path)
    for entry in reversed(entries):
        if entry.get("type") == "RUN_END":
            payload = entry.get("payload", {})
            if isinstance(payload, dict):
                return payload
            return {}
    return {}


def migrate_run(run_dir: Path, in_place: bool) -> dict[str, object]:
    run_dir = Path(run_dir)
    ledger = Ledger(run_dir / "ledger.jsonl")
    capsules_dir = run_dir / "capsules"
    capsule_paths = sorted(capsules_dir.glob("failure_*.json")) if capsules_dir.exists() else []
    ucr_data = read_json(run_dir / "ucr.json") if (run_dir / "ucr.json").exists() else {}
    witness_file, witness_data = _load_first_witness(run_dir)
    overall_verdict = witness_data.get("overall_verdict") if witness_data else ""
    witness_id = _infer_witness_id(run_dir, ucr_data)
    if not witness_id:
        witness_id = str(_last_run_end_payload(run_dir).get("witness_id", ""))
    touched: list[dict[str, Any]] = []

    if overall_verdict == "FAIL" and witness_id:
        for capsule_path in capsule_paths:
            capsule = read_json(capsule_path)
            if capsule.get("witness_id"):
                continue
            before = canonical_dumps(capsule)
            capsule["witness_id"] = witness_id
            after = canonical_dumps(capsule)
            write_json(capsule_path, capsule)
            touched.append(
                {
                    "path": str(capsule_path),
                    "before_hash": hash_bytes(before),
                    "after_hash": hash_bytes(after),
                }
            )

    if touched:
        migration_payload = {
            "migration_version": "v1",
            "from_schema_version": "v1",
            "to_schema_version": "v1",
            "changes": ["capsule_add_witness_id"],
            "capsules": touched,
            "in_place": in_place,
        }
        ledger.append("RUN_MIGRATED", migration_payload)
    report: dict[str, Any] = audit_run(run_dir)
    result: dict[str, object] = {
        "ok": bool(report.get("ok")),
        "files_touched": touched,
        "audit": report,
    }
    return result


@run_app.command("migrate")
def run_migrate_cmd(
    run_dir: Path = RUN_DIR_REQUIRED_OPTION, in_place: bool = typer.Option(False, "--in-place")
) -> None:
    report = migrate_run(run_dir, in_place=in_place)
    console.print(report)
    if not report["ok"]:
        raise typer.Exit(code=1)


@store_app.command("audit")
def store_audit_cmd(
    store_dir: Path = STORE_DIR_OPTION
) -> None:
    report = audit_store(store_dir)
    console.print(report)
    if not report.get("ok"):
        raise typer.Exit(code=1)


@app.command("doctor")
def doctor_cmd(
    repo_root: Optional[Path] = DOCTOR_REPO_ROOT_OPTION,
    json_output: bool = DOCTOR_JSON_OPTION,
    smoke: bool = DOCTOR_SMOKE_OPTION,
) -> None:
    root = repo_root or _find_repo_root(Path(__file__).resolve())
    root = root.resolve()
    missing: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}

    def relpath(path: Path) -> str:
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)

    def check_json(path: Path, label: str, require_newline: bool = False) -> bool:
        ok = True
        if not path.exists():
            missing.append(f"missing_{label}:{relpath(path)}")
            return False
        try:
            read_json(path)
        except Exception:
            missing.append(f"bad_json:{relpath(path)}")
            ok = False
        if require_newline:
            data = path.read_bytes()
            if not data.endswith(b"\n"):
                missing.append(f"missing_newline:{relpath(path)}")
                ok = False
        return ok

    # Store audit import works.
    store_audit_ok = True
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store_dir = Path(tmp_dir) / "store"
            ensure_dir(store_dir / "pyfunc")
            program_bytes = b"def solve(x):\n    return x\n"
            program_hash = hash_bytes(program_bytes)
            program_path = store_dir / "pyfunc" / f"{program_hash}.py"
            program_path.write_bytes(program_bytes)
            manifest = {
                "schema_version": "v2",
                "programs": {program_hash: {"store_path": str(program_path)}},
            }
            write_json(store_dir / "manifest.json", manifest)
            report = audit_store(store_dir)
            store_audit_ok = bool(report.get("ok"))
    except Exception:
        store_audit_ok = False
    if not store_audit_ok:
        missing.append("store_audit_failed")
    checks["store_audit"] = store_audit_ok

    # Scripts existence and executability.
    scripts = [
        "scripts/ci_all.sh",
        "scripts/ci_golden_suite.sh",
        "scripts/ci_golden_suite_v2.sh",
        "scripts/ci_golden_suite_v3.sh",
        "scripts/ci_golden_suite_v3_warm.sh",
        "scripts/ci_golden_suite_v4.sh",
        "scripts/ci_golden_suite_v5.sh",
        "scripts/ci_golden_suite_v6.sh",
        "scripts/ci_golden_suite_v6_warm.sh",
        "scripts/ci_golden_suite_v7.sh",
        "scripts/ci_sealed.sh",
    ]
    scripts_ok = True
    for script in scripts:
        path = root / script
        if not path.exists():
            missing.append(f"missing_script:{script}")
            scripts_ok = False
            continue
        if not os.access(path, os.X_OK):
            missing.append(f"script_not_executable:{script}")
            scripts_ok = False
    checks["scripts"] = scripts_ok

    # Suites and baselines.
    suites_ok = True
    baselines_ok = True
    for version in range(1, 8):
        suite_path = root / "examples" / "suites" / f"suite_v{version}.json"
        if not check_json(suite_path, "suite"):
            suites_ok = False
        baseline_path = (
            root / "examples" / "baselines" / f"suite_v{version}.report.norm.json"
        )
        if not check_json(baseline_path, "baseline", require_newline=True):
            baselines_ok = False
    checks["suites"] = suites_ok
    checks["baselines"] = baselines_ok
    checks["baseline_newlines"] = baselines_ok

    sealed_seed = root / "examples" / "sealed" / "sealed_v1.json"
    sealed_seed_alt = root / "examples" / "sealed_v1.json"
    sealed_ok = sealed_seed.exists() or sealed_seed_alt.exists()
    if not sealed_ok:
        missing.append("missing_sealed_seed")
    checks["sealed_seed"] = sealed_ok

    # ci_all references expected scripts.
    ci_all_path = root / "scripts" / "ci_all.sh"
    ci_all_refs_ok = True
    if ci_all_path.exists():
        content = ci_all_path.read_text(encoding="utf-8")
        for script in scripts[1:]:
            script_name = Path(script).name
            if script_name not in content:
                missing.append(f"ci_all_missing_ref:{script}")
                ci_all_refs_ok = False
    else:
        ci_all_refs_ok = False
    checks["ci_all_refs"] = ci_all_refs_ok

    smoke_ok = True
    if smoke:
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                suite_file = Path(tmp_dir) / "suite.json"
                suite_payload = {
                    "suite_id": "doctor_smoke",
                    "tasks": [
                        {
                            "task_file": str(
                                root / "examples" / "tasks" / "arith_01.json"
                            )
                        }
                    ],
                }
                write_json(suite_file, suite_payload)
                suite_run_cmd(
                    suite_file=suite_file,
                    out_dir=Path(tmp_dir) / "out",
                    policy_version="v1",
                    warm_start_store=None,
                )
        except Exception:
            smoke_ok = False
            missing.append("smoke_failed")
    checks["smoke"] = smoke_ok

    ok = not missing
    if json_output:
        payload = {
            "ok": ok,
            "missing": missing,
            "warnings": warnings,
            "checks": checks,
        }
        print(canonical_dumps(payload).decode("utf-8"))
    else:
        for name in sorted(checks.keys()):
            status = "PASS" if checks[name] else "FAIL"
            print(f"{status} {name}")
        if missing:
            print("SUMMARY FAIL")
            for item in missing:
                print(f"missing: {item}")
        else:
            print("SUMMARY OK")
    if not ok:
        raise typer.Exit(code=1)


def normalize_suite_report(report: dict[str, Any]) -> dict[str, Any]:
    r = copy.deepcopy(report)
    for t in r.get("per_task", []):
        t.pop("synth_ns", None)
        t.pop("active_view_hash", None)
        cost = t.get("cost")
        if isinstance(cost, dict):
            cost.pop("synth_ns", None)
            cost.pop("verify_ns", None)
            cost.pop("breaker_ns", None)
            if not cost:
                t.pop("cost", None)
        fd = t.get("forge_decision")
        if isinstance(fd, dict):
            fd.pop("witness_id", None)
            t["forge_decision"] = fd
    totals = r.get("totals")
    if isinstance(totals, dict):
        totals.pop("verify_ns", None)
    for u in r.get("store_updates", []):
        sp = u.get("store_path")
        if isinstance(sp, str):
            sp = sp.replace("\\", "/")
            marker = "/store/"
            if marker in sp:
                u["store_path"] = "store/" + sp.split(marker, 1)[1]
                continue
            if sp.startswith("store/"):
                u["store_path"] = sp
                continue
            parts = [part for part in sp.split("/") if part]
            if "store" in parts:
                idx = parts.index("store")
                suffix = "/".join(parts[idx + 1 :])
                if suffix:
                    u["store_path"] = "store/" + suffix
                else:
                    u["store_path"] = "store"
                continue
            u["store_path"] = sp
    return r


def ensure_trailing_newline(path: Path) -> None:
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        path.write_bytes(data + b"\n")


def _sealed_summary(suite_file: Path, report: dict[str, Any]) -> dict[str, Any]:
    suite_data = read_json(suite_file)
    suite_id = suite_data.get("suite_id", suite_file.stem)
    tasks = suite_data.get("tasks", [])
    per_task = {
        task.get("task_id"): task for task in report.get("per_task", []) if isinstance(task, dict)
    }
    totals = {"pass": 0, "fail": 0}
    per_domain: dict[str, dict[str, int]] = {}
    tasks_summary: list[dict[str, str]] = []
    for entry in tasks:
        if not isinstance(entry, dict):
            continue
        task_file = Path(entry.get("task_file", ""))
        if not task_file:
            continue
        task = load_task(task_file)
        entry_report = per_task.get(task.task_id, {})
        verdict = str(entry_report.get("verdict", "FAIL"))
        tasks_summary.append({"task_id": task.task_id, "verdict": verdict})
        if verdict == "PASS":
            totals["pass"] += 1
        else:
            totals["fail"] += 1
        domain_counts = per_domain.setdefault(task.task_type, {"pass": 0, "fail": 0})
        if verdict == "PASS":
            domain_counts["pass"] += 1
        else:
            domain_counts["fail"] += 1
    return {
        "suite_id": suite_id,
        "totals": totals,
        "per_domain": per_domain,
        "tasks": tasks_summary,
    }


@suite_app.command("run")
def suite_run_cmd(
    suite_file: Path = SUITE_FILE_OPTION,
    out_dir: Path = OUT_DIR_OPTION,
    policy_version: str = POLICY_OPTION,
    warm_start_store: Optional[Path] = SUITE_WARM_START_OPTION,
    proposer_kind: str = PROPOSER_OPTION,
    static_program: Optional[str] = STATIC_PROGRAM_OPTION,
    replay_file: Optional[Path] = REPLAY_FILE_OPTION,
    cmd: Optional[List[str]] = CMD_OPTION,
    record_proposals: Optional[Path] = RECORD_PROPOSALS_OPTION,
) -> None:
    suite_data = read_json(suite_file)
    suite_id = suite_data.get("suite_id", suite_file.stem)
    tasks = suite_data.get("tasks", [])
    report_dir = out_dir
    ensure_dir(report_dir)
    proposer = _build_proposer(proposer_kind, static_program, replay_file, cmd)
    record_path = Path(record_proposals) if record_proposals else None
    if record_path and record_path.exists():
        record_path.unlink()

    store_dir = report_dir / "store"
    ensure_dir(store_dir)
    manifest_path = store_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
    else:
        manifest = {"schema_version": "v2", "programs": {}}
    if manifest.get("schema_version") != "v2":
        manifest["schema_version"] = "v2"
    programs: dict[str, dict[str, Any]] = manifest.setdefault("programs", {})

    totals = {"pass": 0, "fail": 0, "audit_failures": 0, "verify_ns": 0, "breaker_attempts": 0}
    per_task: List[dict[str, Any]] = []
    store_updates: List[dict[str, str]] = []

    for entry in tasks:
        task_file = Path(entry["task_file"])
        task = load_task(task_file)
        overrides = dict(entry.get("overrides", {}))
        if warm_start_store and not overrides.get("warm_start_store"):
            overrides["warm_start_store"] = str(warm_start_store)
        settings = Settings(**overrides)
        task_run_dir = report_dir / task_file.stem
        summary = episode_run(
            task_file=task_file,
            run_dir=task_run_dir,
            settings=settings,
            proposer=proposer,
        )
        actual_run = Path(summary["ucr_path"]).parent
        audit_report = audit_run(actual_run)
        if record_path is not None:
            proposer_path = actual_run / "proposer.json"
            if proposer_path.exists():
                proposer_record = read_json(proposer_path)
            else:
                proposer_record = {"error_atom": "PROPOSER_RECORD_MISSING"}
            record_entry = {
                "task_id": task.task_id,
                "domain": task.task_type,
                "spec_signature": task.spec_signature(),
                "proposer_id": proposer_record.get("proposer_id", ""),
                "proposal_hash": proposer_record.get("proposal_hash", ""),
                "candidate_program": proposer_record.get("candidate_program", ""),
            }
            error_atom = proposer_record.get("error_atom")
            if error_atom:
                record_entry["error_atom"] = error_atom
            write_jsonl_line(record_path, record_entry)
        ucr_data = read_json(actual_run / "ucr.json")
        program_hash = ucr_data.get("hashes", {}).get("program_hash", "")
        decision_path = actual_run / "forge" / "decision.json"
        decision_data: dict[str, Any] = {}
        if decision_path.exists():
            decision_data = read_json(decision_path)
        if not decision_data:
            decision_data = {"decision": "REJECT", "reason": "unknown"}
        breaker_kpi: dict[str, Any] = {}
        breaker_path = actual_run / "artifacts" / "reports" / "breaker_kpi.json"
        if breaker_path.exists():
            breaker_kpi = read_json(breaker_path)
        else:
            breaker_kpi = {
                "status": "SKIPPED",
                "reason": decision_data.get("reason", "CEGIS_UNSAT"),
            }
        cost = {
            "synth_ns": summary.get("synth_ns", 0),
            "verify_ns": summary.get("verify_ns", 0),
            "breaker_ns": summary.get("breaker_ns", 0),
        }
        attempts = {
            "breaker_attempts": summary.get("breaker_attempts", 0),
            "meta_cases": summary.get("meta_cases", 0),
        }
        per_task.append(
            {
                "task_id": summary["task_id"],
                "verdict": summary["verdict"],
                "active_view_hash": summary["active_view_hash"],
                "program_hash": program_hash,
                "breaker_kpi": breaker_kpi,
                "forge_decision": decision_data,
                "audit_ok": audit_report["ok"],
                "synth_ns": summary.get("synth_ns", 0),
                "cost": cost,
                "attempts": attempts,
                "warm_start_store": summary.get("warm_start_store", False),
                "warm_start_candidate_hash": summary.get("warm_start_candidate_hash", ""),
                "warm_start_candidate_rejected": summary.get(
                    "warm_start_candidate_rejected", False
                ),
                "warm_start_provided": summary.get("warm_start_provided", False),
            }
        )
        if summary["verdict"] == "PASS":
            totals["pass"] += 1
        else:
            totals["fail"] += 1
        if not audit_report["ok"]:
            totals["audit_failures"] += 1
        totals["verify_ns"] += int(cost.get("verify_ns", 0))
        totals["breaker_attempts"] += int(attempts.get("breaker_attempts", 0))
        if decision_data.get("decision") == "ADMIT" and program_hash:
            program_entry = programs.get(program_hash)
            manifest_spec = task.spec_signature()
            io_schema_hash = task.io_schema_hash()
            if task.task_type == "pyfunc":
                ext = ".py"
            elif task.task_type == "codepatch":
                ext = ".patch"
            else:
                ext = ".json"
            store_path = store_dir / task.task_type / f"{program_hash}{ext}"
            program_bytes: bytes | None = None
            artifact_base = actual_run / "artifacts"
            if task.task_type == "pyfunc":
                run_program_path = artifact_base / "pyfunc" / "program.py"
            elif task.task_type == "codepatch":
                run_program_path = artifact_base / "codepatch" / "program.patch"
            elif task.task_type == "jsonspec":
                run_program_path = artifact_base / "jsonspec" / "program.json"
            else:
                run_program_path = artifact_base / "bvps" / "program.json"
            if run_program_path.exists():
                if task.task_type == "pyfunc":
                    program_bytes = run_program_path.read_bytes()
                elif task.task_type == "codepatch":
                    program_bytes = run_program_path.read_bytes()
                else:
                    program_bytes = canonical_dumps(read_json(run_program_path))
            elif settings.warm_start_store:
                warm_manifest_path = Path(settings.warm_start_store) / "manifest.json"
                if warm_manifest_path.exists():
                    warm_manifest = read_json(warm_manifest_path)
                    warm_programs = warm_manifest.get("programs", {})
                    warm_entry = warm_programs.get(program_hash, {})
                    warm_domain = warm_entry.get("domain", task.task_type)
                    if warm_domain == "pyfunc":
                        warm_ext = ".py"
                    elif warm_domain == "codepatch":
                        warm_ext = ".patch"
                    else:
                        warm_ext = ".json"
                    warm_fallback = (
                        Path(settings.warm_start_store) / warm_domain / f"{program_hash}{warm_ext}"
                    )
                    warm_source_path = Path(
                        warm_entry.get("store_path", warm_fallback)
                    )
                    if warm_source_path.exists():
                        program_bytes = warm_source_path.read_bytes()
            if program_bytes is None:
                raise RuntimeError(
                    f"cannot reproduce admitted program {program_hash} for task {task.task_id}"
                )
            ensure_dir(store_path.parent)
            store_path.write_bytes(program_bytes)
            if not program_entry:
                pyexec_version = PYEXEC_VERSION if task.task_type == "pyfunc" else ""
                program_entry = {
                    "program_hash": program_hash,
                    "domain": task.task_type,
                    "spec_signature": manifest_spec,
                    "io_schema_hash": io_schema_hash,
                    "admitted_by_task": summary["task_id"],
                    "admit_count": 1,
                    "pyexec_version": pyexec_version,
                    "first_admitted_by_task": summary["task_id"],
                    "admitted_count": 1,
                    "last_seen_suite": suite_id,
                    "provenance_witness_ids": (
                        [decision_data["witness_id"]]
                        if decision_data.get("witness_id")
                        else []
                    ),
                    "store_path": str(store_path),
                }
                programs[program_hash] = program_entry
                store_updates.append(
                    {"program_hash": program_hash, "store_path": str(store_path)}
                )
            else:
                admit_count = program_entry.get(
                    "admit_count", program_entry.get("admitted_count", 0)
                )
                program_entry["admit_count"] = admit_count + 1
                program_entry["admitted_count"] = program_entry["admit_count"]
                program_entry["spec_signature"] = manifest_spec
                program_entry["io_schema_hash"] = io_schema_hash
                if task.task_type == "pyfunc":
                    program_entry["pyexec_version"] = PYEXEC_VERSION
                program_entry["last_seen_suite"] = suite_id
                witness_id = decision_data.get("witness_id")
                if witness_id:
                    entry_witnesses = program_entry.setdefault("provenance_witness_ids", [])
                    if witness_id not in entry_witnesses:
                        entry_witnesses.append(witness_id)

    write_json(manifest_path, manifest)
    report = {
        "suite_id": suite_id,
        "policy_version": policy_version,
        "per_task": per_task,
        "totals": totals,
        "store_updates": store_updates,
    }
    report_path = report_dir / "report.json"
    write_json(report_path, report)
    ensure_trailing_newline(report_path)
    norm_report = normalize_suite_report(report)
    norm_path = report_dir / "report.norm.json"
    write_json(norm_path, norm_report)
    ensure_trailing_newline(norm_path)
    console.print({"report": str(report_path)})


@suite_app.command("run-sealed")
def suite_run_sealed_cmd(
    suite_file: Path = SUITE_FILE_OPTION,
    out_dir: Path = OUT_DIR_OPTION,
    policy_version: str = POLICY_OPTION,
    warm_start_store: Optional[Path] = SUITE_WARM_START_OPTION,
) -> None:
    suite_run_cmd(
        suite_file=suite_file,
        out_dir=out_dir,
        policy_version=policy_version,
        warm_start_store=warm_start_store,
    )
    report_path = Path(out_dir) / "report.json"
    report = read_json(report_path)
    summary = _sealed_summary(suite_file, report)
    summary_path = Path(out_dir) / "sealed_summary.json"
    write_json(summary_path, summary)
    ensure_trailing_newline(summary_path)
    console.print({"sealed_summary": str(summary_path)})


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
app.add_typer(store_app, name="store")
app.add_typer(suite_app, name="suite")

if __name__ == "__main__":
    app()
