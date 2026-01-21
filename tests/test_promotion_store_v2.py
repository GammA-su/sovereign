from pathlib import Path

from sovereidolon_v1.promotion_store import get_best_candidate, load_index, make_key, record_best


def _record(
    store_dir: Path,
    *,
    domain: str,
    spec_hash: str,
    lane_id: str,
    families_mode: str,
    meta_families: list[str],
    program_hash: str,
    score_key: list[int],
) -> None:
    key = make_key(
        domain=domain,
        spec_hash=spec_hash,
        lane_id=lane_id,
        families_mode=families_mode,
        meta_families=meta_families,
    )
    tier = "sealed" if families_mode == "sealed" else "public"
    record_best(
        store_dir,
        key=key,
        program_hash=program_hash,
        domain=domain,
        spec_hash=spec_hash,
        tier=tier,
        score={"verdict": 1, "kpi": {}, "cost": {}},
        score_key=score_key,
        score_scaled=score_key[0] if score_key else 0,
        store_path=f"{domain}/{program_hash}.json",
    )


def test_promotion_v2_key_separation(tmp_path: Path) -> None:
    store_dir = tmp_path / "promo"
    _record(
        store_dir,
        domain="pyfunc",
        spec_hash="spec",
        lane_id="pyexec",
        families_mode="public",
        meta_families=[],
        program_hash="a" * 64,
        score_key=[1, 1],
    )
    _record(
        store_dir,
        domain="pyfunc",
        spec_hash="spec",
        lane_id="pyexec_meta",
        families_mode="public",
        meta_families=["meta_family"],
        program_hash="b" * 64,
        score_key=[2, 1],
    )
    index = load_index(store_dir / "index.json")
    entries = index.get("entries", {})
    assert isinstance(entries, dict)
    assert len(entries) == 1
    entry = next(iter(entries.values()))
    assert entry.get("program_hash") == "b" * 64


def test_promotion_v2_sealed_no_public_fallback(tmp_path: Path) -> None:
    store_dir = tmp_path / "promo"
    _record(
        store_dir,
        domain="arith",
        spec_hash="spec",
        lane_id="arith",
        families_mode="public",
        meta_families=[],
        program_hash="a" * 64,
        score_key=[1, 1],
    )
    entry = get_best_candidate(
        store_dir,
        domain="arith",
        spec_hash="spec",
        lane_id="arith",
        families_mode="sealed",
        meta_families=[],
        meta_required=False,
        strict=True,
    )
    assert entry is None


def test_promotion_v2_spec_hash_mismatch_returns_none(tmp_path: Path) -> None:
    store_dir = tmp_path / "promo"
    _record(
        store_dir,
        domain="jsonspec",
        spec_hash="spec_a",
        lane_id="jsonspec_meta",
        families_mode="public",
        meta_families=[],
        program_hash="a" * 64,
        score_key=[1, 1],
    )
    entry = get_best_candidate(
        store_dir,
        domain="jsonspec",
        spec_hash="spec_b",
        lane_id="jsonspec_meta",
        families_mode="public",
        meta_families=["key_order_invariance"],
        meta_required=True,
    )
    assert entry is None


def test_promotion_v2_determinism_newline_and_sorting(tmp_path: Path) -> None:
    store_a = tmp_path / "promo_a"
    store_b = tmp_path / "promo_b"
    _record(
        store_a,
        domain="pyfunc",
        spec_hash="spec",
        lane_id="pyexec",
        families_mode="public",
        meta_families=[],
        program_hash="a" * 64,
        score_key=[1, 1],
    )
    _record(
        store_a,
        domain="pyfunc",
        spec_hash="spec",
        lane_id="pyexec_meta",
        families_mode="public",
        meta_families=["meta_family"],
        program_hash="b" * 64,
        score_key=[2, 1],
    )
    _record(
        store_b,
        domain="pyfunc",
        spec_hash="spec",
        lane_id="pyexec_meta",
        families_mode="public",
        meta_families=["meta_family"],
        program_hash="b" * 64,
        score_key=[2, 1],
    )
    _record(
        store_b,
        domain="pyfunc",
        spec_hash="spec",
        lane_id="pyexec",
        families_mode="public",
        meta_families=[],
        program_hash="a" * 64,
        score_key=[1, 1],
    )
    data_a = (store_a / "index.json").read_bytes()
    data_b = (store_b / "index.json").read_bytes()
    assert data_a.endswith(b"\n")
    assert data_b.endswith(b"\n")
    assert data_a == data_b
