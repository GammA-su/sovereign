from pathlib import Path

from sovereidolon_v1.cli import app
from sovereidolon_v1.store.audit import audit_store
from sovereidolon_v1.utils import hash_bytes, write_json


def _write_store(tmp_path: Path) -> tuple[Path, str, Path]:
    store_dir = tmp_path / "store"
    program_dir = store_dir / "pyfunc"
    program_dir.mkdir(parents=True)
    program_bytes = b"def solve(x):\n    return x\n"
    program_hash = hash_bytes(program_bytes)
    program_path = program_dir / f"{program_hash}.py"
    program_path.write_bytes(program_bytes)
    manifest = {
        "schema_version": "v2",
        "programs": {program_hash: {"store_path": str(program_path)}},
    }
    write_json(store_dir / "manifest.json", manifest)
    return store_dir, program_hash, program_path


def test_store_audit_ok(tmp_path: Path) -> None:
    assert app
    store_dir, _, _ = _write_store(tmp_path)
    report = audit_store(store_dir)
    assert report["ok"] is True
    assert report["checks"]["manifest_consistency"] is True
    assert report["errors"] == []


def test_store_audit_hash_mismatch(tmp_path: Path) -> None:
    assert app
    store_dir, program_hash, program_path = _write_store(tmp_path)
    bad_hash = "0" * 64
    if bad_hash == program_hash:
        bad_hash = "1" * 64
    manifest = {
        "schema_version": "v2",
        "programs": {bad_hash: {"store_path": str(program_path)}},
    }
    write_json(store_dir / "manifest.json", manifest)
    report = audit_store(store_dir)
    expected_error = f"hash_mismatch:{bad_hash}:{bad_hash}:{program_hash}"
    assert report["ok"] is False
    assert report["checks"]["manifest_consistency"] is False
    assert report["errors"] == [expected_error]
