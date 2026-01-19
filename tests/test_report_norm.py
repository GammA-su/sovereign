from pathlib import Path

from sovereidolon_v1.cli import ensure_trailing_newline, normalize_suite_report
from sovereidolon_v1.utils import read_json, write_json


def _write_norm(tmp_path: Path, report: dict[str, object]) -> dict[str, object]:
    norm = normalize_suite_report(report)
    out_path = tmp_path / "report.norm.json"
    write_json(out_path, norm)
    ensure_trailing_newline(out_path)
    text = out_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    return read_json(out_path)


def test_report_norm_preserves_relative_store_path(tmp_path: Path) -> None:
    report = {
        "store_updates": [
            {"program_hash": "a" * 64, "store_path": "store/arith/aaa.json"},
        ]
    }
    norm = _write_norm(tmp_path, report)
    assert norm["store_updates"][0]["store_path"] == "store/arith/aaa.json"


def test_report_norm_normalizes_store_prefix(tmp_path: Path) -> None:
    report = {
        "store_updates": [
            {
                "program_hash": "b" * 64,
                "store_path": "/tmp/runs/out/store/bool/bbb.json",
            },
        ]
    }
    norm = _write_norm(tmp_path, report)
    assert norm["store_updates"][0]["store_path"] == "store/bool/bbb.json"
