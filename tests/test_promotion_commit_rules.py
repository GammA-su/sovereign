from pathlib import Path

from sovereidolon_v1.cli import _commit_public_promotions, _commit_sealed_promotions
from sovereidolon_v1.utils import read_json


def _write_program(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_promotion_commit_rules(tmp_path: Path) -> None:
    promotion_store = tmp_path / "promo"
    propose_dir = tmp_path / "propose"
    sealed_dir = tmp_path / "sealed"

    program_hash = "abc123"
    spec_hash = "spec123"
    public_report = {
        "per_task": [
            {
                "task_id": "task_1",
                "domain": "pyfunc",
                "spec_hash": spec_hash,
                "verdict": "FAIL",
                "attempts": {"breaker_attempts": 5},
                "controller_score_scaled": 0,
                "forge_decision": {"decision": "REJECT"},
            }
        ],
        "store_updates": [],
    }

    propose_store_path = propose_dir / "store" / "pyfunc" / f"{program_hash}.py"
    _write_program(propose_store_path, "def solve(a, b):\n    return a + b\n")
    propose_report = {
        "per_task": [
            {
                "task_id": "task_1",
                "domain": "pyfunc",
                "spec_hash": spec_hash,
                "verdict": "PASS",
                "program_hash": program_hash,
                "attempts": {"breaker_attempts": 1},
                "controller_score_scaled": 10,
                "forge_decision": {"decision": "ADMIT"},
            }
        ],
        "store_updates": [
            {"program_hash": program_hash, "store_path": str(propose_store_path)}
        ],
    }

    sealed_store_path = sealed_dir / "store" / "pyfunc" / f"{program_hash}.py"
    _write_program(sealed_store_path, "def solve(a, b):\n    return a + b\n")
    sealed_report = {
        "per_task": [
            {
                "task_id": "task_1",
                "domain": "pyfunc",
                "spec_hash": spec_hash,
                "verdict": "PASS",
                "program_hash": program_hash,
                "attempts": {"breaker_attempts": 1},
                "controller_score_scaled": 10,
                "forge_decision": {"decision": "ADMIT"},
            }
        ],
        "store_updates": [
            {"program_hash": program_hash, "store_path": str(sealed_store_path)}
        ],
    }

    writes_public = _commit_public_promotions(
        public_report, propose_report, propose_dir, promotion_store
    )
    assert writes_public == 1
    index = read_json(promotion_store / "index.json")
    entries = index.get("entries", {})
    assert any(
        entry.get("tier") == "public" and entry.get("spec_hash") == spec_hash
        for entry in entries.values()
        if isinstance(entry, dict)
    )

    writes_sealed, upgrades = _commit_sealed_promotions(
        sealed_report, sealed_dir, promotion_store
    )
    assert writes_sealed == 1
    assert upgrades == 1
