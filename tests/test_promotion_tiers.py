from pathlib import Path

from typer.testing import CliRunner

from sovereidolon_v1.cli import app
from sovereidolon_v1.promotion_store import get_best, make_key, record_best
from sovereidolon_v1.utils import canonical_dumps, read_json, write_json


def _record_entry(
    store_dir: Path,
    *,
    domain: str,
    spec_hash: str,
    lane_id: str,
    families_mode: str,
    program_hash: str,
    score_scaled: int,
) -> None:
    key = make_key(
        domain=domain,
        spec_hash=spec_hash,
        lane_id=lane_id,
        families_mode=families_mode,
        meta_families=[],
    )
    tier = "sealed" if families_mode == "sealed" else "public"
    record_best(
        store_dir,
        key=key,
        program_hash=program_hash,
        domain=domain,
        spec_hash=spec_hash,
        tier=tier,
        score={"score_scaled": score_scaled},
        score_key=[score_scaled],
        score_scaled=score_scaled,
        store_path=f"{domain}/{program_hash}.json",
    )


def test_promotion_tier_prefers_sealed(tmp_path: Path) -> None:
    store_dir = tmp_path / "promo"
    _record_entry(
        store_dir,
        domain="pyfunc",
        spec_hash="spec",
        lane_id="pyexec",
        families_mode="public",
        program_hash="a" * 64,
        score_scaled=1,
    )
    _record_entry(
        store_dir,
        domain="pyfunc",
        spec_hash="spec",
        lane_id="pyexec",
        families_mode="sealed",
        program_hash="b" * 64,
        score_scaled=2,
    )
    entry = get_best(store_dir, "pyfunc", "spec")
    assert entry is not None
    assert entry["program_hash"] == "b" * 64
    assert entry["tier"] == "sealed"


def test_promotion_tier_fallback_public(tmp_path: Path) -> None:
    store_dir = tmp_path / "promo"
    _record_entry(
        store_dir,
        domain="pyfunc",
        spec_hash="spec",
        lane_id="pyexec",
        families_mode="public",
        program_hash="a" * 64,
        score_scaled=1,
    )
    entry = get_best(store_dir, "pyfunc", "spec")
    assert entry is not None
    assert entry["program_hash"] == "a" * 64
    assert entry["tier"] == "public"


def test_promotion_tier_backward_compat_missing_tier(tmp_path: Path) -> None:
    store_dir = tmp_path / "promo"
    index_path = store_dir / "index.json"
    payload = {
        "schema_version": "v2",
        "entries": {
            "k": {
                "program_hash": "a" * 64,
                "domain": "pyfunc",
                "spec_signature": "spec",
                "score_scaled": 3,
                "store_path": "pyfunc/" + "a" * 64 + ".py",
            }
        },
    }
    data = canonical_dumps(payload)
    if not data.endswith(b"\n"):
        data += b"\n"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_bytes(data)

    entry = get_best(store_dir, "pyfunc", "spec")
    assert entry is not None
    assert entry["tier"] == "public"


def test_promotion_tier_integration_prefers_sealed(tmp_path: Path) -> None:
    sealed_task = tmp_path / "pyfunc_sealed.json"
    sealed_payload = read_json(Path("examples/tasks/pyfunc_01.json"))
    sealed_payload["sealed"] = {
        "sealed_seed": 1234,
        "withheld_families": [],
        "canary_token": "CANARY_PROMO",
    }
    write_json(sealed_task, sealed_payload)

    sealed_suite = tmp_path / "sealed_suite.json"
    public_suite = tmp_path / "public_suite.json"
    write_json(
        sealed_suite,
        {"suite_id": "sealed_suite", "tasks": [{"task_file": str(sealed_task)}]},
    )
    write_json(
        public_suite,
        {
            "suite_id": "public_suite",
            "tasks": [{"task_file": "examples/tasks/pyfunc_01.json"}],
        },
    )

    promo_dir = tmp_path / "promotion_store"
    runner = CliRunner()
    sealed_out = tmp_path / "sealed_out"
    result = runner.invoke(
        app,
        [
            "suite",
            "run-sealed",
            "--suite-file",
            str(sealed_suite),
            "--out-dir",
            str(sealed_out),
            "--promotion-store",
            str(promo_dir),
        ],
    )
    assert result.exit_code == 0

    public_out = tmp_path / "public_out"
    result_public = runner.invoke(
        app,
        [
            "suite",
            "run",
            "--suite-file",
            str(public_suite),
            "--out-dir",
            str(public_out),
            "--promotion-store",
            str(promo_dir),
            "--prefer-promotion-store",
        ],
    )
    assert result_public.exit_code == 0

    report = read_json(public_out / "report.json")
    entry = report["per_task"][0]
    assert entry["promotion_best_tier_used"] == "sealed"
    assert entry["promotion_best_hash_used"]


def test_promotion_store_writes_sealed_tier(tmp_path: Path) -> None:
    sealed_suite = tmp_path / "sealed_suite.json"
    write_json(
        sealed_suite,
        {
            "suite_id": "sealed_suite",
            "tasks": [{"task_file": "examples/tasks/pyfunc_01.json"}],
        },
    )

    promo_dir = tmp_path / "promotion_store"
    runner = CliRunner()
    sealed_out = tmp_path / "sealed_out"
    result = runner.invoke(
        app,
        [
            "suite",
            "run-sealed",
            "--suite-file",
            str(sealed_suite),
            "--out-dir",
            str(sealed_out),
            "--promotion-store",
            str(promo_dir),
            "--prefer-promotion-tier",
            "public",
        ],
    )
    assert result.exit_code == 0

    report = read_json(sealed_out / "report.json")
    assert report.get("is_sealed_run") is True
    if report.get("store_updates"):
        index = read_json(promo_dir / "index.json")
        entries = index.get("entries", {})
        tiers = {
            entry.get("tier")
            for entry in entries.values()
            if isinstance(entry, dict)
        }
        assert "sealed" in tiers


def test_promotion_tier_fallback_records_atom(tmp_path: Path) -> None:
    suite_file = tmp_path / "suite.json"
    write_json(
        suite_file,
        {"suite_id": "promo_fallback", "tasks": [{"task_file": "examples/tasks/pyfunc_01.json"}]},
    )
    promo_dir = tmp_path / "promotion_store"
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
            "--promotion-store",
            str(promo_dir),
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
            "--promotion-store",
            str(promo_dir),
            "--prefer-promotion-store",
            "--prefer-promotion-tier",
            "sealed",
        ],
    )
    assert result_b.exit_code == 0

    report = read_json(out_b / "report.json")
    entry = report["per_task"][0]
    assert entry["promotion_attempted"] is True
    assert entry["promotion_used"] is True
    assert entry["promotion_best_tier"] == "public"
    assert "PROMOTION_TIER_FALLBACK" in entry.get("promotion_reject_reason_atoms", [])


def test_promotion_tier_strict_rejects(tmp_path: Path) -> None:
    suite_file = tmp_path / "suite.json"
    write_json(
        suite_file,
        {"suite_id": "promo_strict", "tasks": [{"task_file": "examples/tasks/pyfunc_01.json"}]},
    )
    promo_dir = tmp_path / "promotion_store"
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
            "--promotion-store",
            str(promo_dir),
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
            "--promotion-store",
            str(promo_dir),
            "--prefer-promotion-store",
            "--prefer-promotion-tier",
            "sealed",
            "--promotion-tier-strict",
        ],
    )
    assert result_b.exit_code == 0

    report = read_json(out_b / "report.json")
    entry = report["per_task"][0]
    assert entry["promotion_attempted"] is True
    assert entry["promotion_used"] is False
    assert "PROMOTION_TIER_NOT_FOUND" in entry.get("promotion_reject_reason_atoms", [])
