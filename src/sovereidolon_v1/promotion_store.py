from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .utils import canonical_dumps, ensure_dir, read_json, stable_hash

INDEX_SCHEMA_VERSION = "v2"


def _index_path(path: Path) -> Path:
    if path.suffix == ".json":
        return path
    return path / "index.json"


def load_index(path: Path) -> Dict[str, Any]:
    index_path = _index_path(path)
    if not index_path.exists():
        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "entries": {},
            "legacy_entries": {},
        }
    data = read_json(index_path)
    if not isinstance(data, dict):
        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "entries": {},
            "legacy_entries": {},
        }
    schema_version = str(data.get("schema_version", "v1"))
    entries = data.get("entries")
    if not isinstance(entries, dict):
        entries = {}
    if schema_version == "v1":
        return {
            "schema_version": schema_version,
            "entries": {},
            "legacy_entries": entries,
        }
    legacy = data.get("legacy_entries")
    if not isinstance(legacy, dict):
        legacy = {}
    return {
        "schema_version": schema_version,
        "entries": entries,
        "legacy_entries": legacy,
    }


def save_index(path: Path, index: Mapping[str, Any]) -> None:
    index_path = _index_path(path)
    ensure_dir(index_path.parent)
    payload = dict(index)
    payload.setdefault("schema_version", INDEX_SCHEMA_VERSION)
    payload.setdefault("entries", {})
    legacy = payload.get("legacy_entries")
    if not isinstance(legacy, dict) or not legacy:
        payload.pop("legacy_entries", None)
    data = canonical_dumps(payload)
    if not data.endswith(b"\n"):
        data += b"\n"
    index_path.write_bytes(data)


def make_key(
    *,
    domain: str,
    spec_hash: str,
    lane_id: str,
    families_mode: str,
    meta_families: Sequence[str],
) -> str:
    tier = "sealed" if families_mode == "sealed" else "public"
    return "|".join(
        [
            domain,
            spec_hash,
            tier,
        ]
    )


def compute_meta_profile_hash(families: Sequence[str]) -> str:
    cleaned = sorted({name for name in families if isinstance(name, str) and name})
    if not cleaned:
        return ""
    return stable_hash(cleaned)


def _coerce_score_key(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, dict):
        value = value.get("score_key") or value.get("key")
    if isinstance(value, (list, tuple)):
        try:
            return tuple(int(item) for item in value)
        except (TypeError, ValueError):
            return None
    return None


def _coerce_score_scaled(score: Mapping[str, Any], score_key: Sequence[int]) -> int:
    if isinstance(score, Mapping):
        value = score.get("score_scaled")
        if isinstance(value, (int, float)):
            return int(value)
    if score_key:
        try:
            return int(score_key[0])
        except (TypeError, ValueError):
            return 0
    return 0


def _normalize_tier(value: Any) -> str:
    tier = str(value or "").lower()
    if tier == "sealed":
        return "sealed"
    return "public"


def _entry_spec_hash(entry: Mapping[str, Any]) -> str:
    spec_hash = entry.get("spec_hash")
    if isinstance(spec_hash, str) and spec_hash:
        return spec_hash
    spec_signature = entry.get("spec_signature")
    if isinstance(spec_signature, str):
        return spec_signature
    return ""


def _entry_score_scaled(entry: Mapping[str, Any]) -> int:
    value = entry.get("score_scaled")
    if isinstance(value, (int, float)):
        return int(value)
    key = _coerce_score_key(entry.get("score_key"))
    if key:
        return int(key[0])
    return 0


def record_best(
    store_dir: Path,
    *,
    key: str,
    program_hash: str,
    domain: str,
    spec_hash: str,
    tier: str,
    score: Mapping[str, Any],
    score_key: Sequence[int],
    score_scaled: Optional[int] = None,
    store_path: str,
    admitted_by_run_id: Optional[str] = None,
    admitted_at_iso: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    store_dir = Path(store_dir)
    index = load_index(store_dir / "index.json")
    entries = index.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        index["entries"] = entries
    existing = entries.get(key)
    if isinstance(existing, dict):
        existing_hash = str(existing.get("program_hash", ""))
        if existing_hash == program_hash:
            return existing
        existing_key = _coerce_score_key(existing.get("score_key"))
        new_key = tuple(int(value) for value in score_key)
        if existing_key is not None and new_key <= existing_key:
            return existing
    entry: Dict[str, Any] = {
        "program_hash": program_hash,
        "domain": domain,
        "spec_hash": spec_hash,
        "tier": _normalize_tier(tier),
        "score": dict(score),
        "score_key": [int(value) for value in score_key],
        "score_scaled": int(
            score_scaled
            if isinstance(score_scaled, (int, float))
            else _coerce_score_scaled(score, score_key)
        ),
        "store_path": store_path,
    }
    if admitted_by_run_id:
        entry["admitted_by_run_id"] = admitted_by_run_id
    if admitted_at_iso:
        entry["admitted_at_iso"] = admitted_at_iso
    if metadata:
        entry.update({key: metadata[key] for key in sorted(metadata.keys())})
    entries[key] = entry
    index["schema_version"] = INDEX_SCHEMA_VERSION
    save_index(store_dir / "index.json", index)
    return entry


def _get_best_v1(
    index: Mapping[str, Any], domain: str, spec_hash: str
) -> Optional[Dict[str, Any]]:
    entries = index.get("legacy_entries", {})
    if not isinstance(entries, dict):
        return None
    domain_entries = entries.get(domain, {})
    if not isinstance(domain_entries, dict):
        return None
    entry = domain_entries.get(spec_hash)
    if isinstance(entry, dict):
        return entry
    return None


def _select_best(entries: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not entries:
        return None

    def _key(entry: Mapping[str, Any]) -> tuple[int, str]:
        return (_entry_score_scaled(entry), str(entry.get("program_hash", "")))

    best = max(entries, key=_key)
    return dict(best)


def get_best_for_tier(
    store_dir: Path,
    *,
    domain: str,
    spec_hash: str,
    tier: str,
) -> Optional[Dict[str, Any]]:
    index = load_index(store_dir / "index.json")
    entries = index.get("entries", {})
    candidates: list[Dict[str, Any]] = []
    if isinstance(entries, dict):
        for entry in entries.values():
            if not isinstance(entry, dict):
                continue
            if entry.get("domain") != domain:
                continue
            if _entry_spec_hash(entry) != spec_hash:
                continue
            if _normalize_tier(entry.get("tier")) != _normalize_tier(tier):
                continue
            candidates.append(entry)
    if _normalize_tier(tier) == "public":
        legacy = _get_best_v1(index, domain, spec_hash)
        if isinstance(legacy, dict):
            legacy_entry = dict(legacy)
            legacy_entry.setdefault("tier", "public")
            candidates.append(legacy_entry)
    best = _select_best(candidates)
    if best is not None:
        best["tier"] = _normalize_tier(best.get("tier"))
    return best


def get_best_any(
    store_dir: Path,
    *,
    domain: str,
    spec_hash: str,
) -> Optional[Dict[str, Any]]:
    index = load_index(store_dir / "index.json")
    entries = index.get("entries", {})
    candidates: list[Dict[str, Any]] = []
    if isinstance(entries, dict):
        for entry in entries.values():
            if not isinstance(entry, dict):
                continue
            if entry.get("domain") != domain:
                continue
            if _entry_spec_hash(entry) != spec_hash:
                continue
            candidates.append(entry)
    legacy = _get_best_v1(index, domain, spec_hash)
    if isinstance(legacy, dict):
        legacy_entry = dict(legacy)
        legacy_entry.setdefault("tier", "public")
        candidates.append(legacy_entry)
    best = _select_best(candidates)
    if best is not None:
        best["tier"] = _normalize_tier(best.get("tier"))
    return best


def get_best_candidate(
    store_dir: Path,
    *,
    domain: str,
    spec_hash: str,
    lane_id: str,
    families_mode: str,
    meta_families: Sequence[str],
    meta_required: bool,
    prefer_tier: str = "sealed",
    strict: bool = False,
) -> Optional[Dict[str, Any]]:
    _ = (lane_id, families_mode, meta_families, meta_required)
    prefer = str(prefer_tier or "sealed").lower()
    if prefer == "any":
        return get_best_any(store_dir, domain=domain, spec_hash=spec_hash)
    preferred_entry = get_best_for_tier(
        store_dir, domain=domain, spec_hash=spec_hash, tier=prefer
    )
    if preferred_entry is not None:
        return preferred_entry
    if strict:
        return None
    other_tier = "public" if prefer == "sealed" else "sealed"
    return get_best_for_tier(store_dir, domain=domain, spec_hash=spec_hash, tier=other_tier)


def find_spec_mismatch_candidate(
    store_dir: Path,
    *,
    domain: str,
    spec_hash: str,
    lane_id: str,
    families_mode: str,
    meta_families: Sequence[str],
    prefer_tier: str = "sealed",
) -> Optional[Dict[str, Any]]:
    index = load_index(store_dir / "index.json")
    entries = index.get("entries", {})
    mismatched: list[Dict[str, Any]] = []
    if isinstance(entries, dict):
        for entry in entries.values():
            if not isinstance(entry, dict):
                continue
            if entry.get("domain") != domain:
                continue
            entry_spec = _entry_spec_hash(entry)
            if entry_spec and entry_spec != spec_hash:
                mismatched.append(entry)
    if not mismatched:
        return None
    prefer = str(prefer_tier or "sealed").lower()
    tier_order = ["sealed", "public"]
    if families_mode == "sealed":
        tier_order = ["sealed"]
    elif prefer == "public":
        tier_order = ["public", "sealed"]
    if prefer == "any" and families_mode != "sealed":
        selected = _select_best(mismatched)
        if selected is not None:
            selected["tier"] = _normalize_tier(selected.get("tier"))
        return selected
    for tier in tier_order:
        tier_candidates = [
            entry
            for entry in mismatched
            if _normalize_tier(entry.get("tier")) == tier
        ]
        selected = _select_best(tier_candidates)
        if selected is not None:
            selected["tier"] = _normalize_tier(selected.get("tier"))
            return selected
    return None


def get_best(
    path: Path, domain: str, spec_hash: str, prefer_tier: str = "sealed"
) -> Optional[Dict[str, Any]]:
    index = load_index(path)
    entries = index.get("entries", {})
    candidates: list[Dict[str, Any]] = []
    if isinstance(entries, dict):
        for entry in entries.values():
            if not isinstance(entry, dict):
                continue
            if entry.get("domain") != domain:
                continue
            if _entry_spec_hash(entry) != spec_hash:
                continue
            candidates.append(entry)

    legacy = _get_best_v1(index, domain, spec_hash)
    if isinstance(legacy, dict):
        legacy_entry = dict(legacy)
        legacy_entry.setdefault("tier", "public")
        candidates.append(legacy_entry)

    prefer = str(prefer_tier or "sealed").lower()
    tier_order = ["sealed", "public"]
    if prefer == "public":
        tier_order = ["public", "sealed"]
    if prefer == "any":
        best = _select_best(candidates)
        if best is not None:
            best["tier"] = _normalize_tier(best.get("tier"))
        return best

    for tier in tier_order:
        tier_candidates = [
            entry
            for entry in candidates
            if _normalize_tier(entry.get("tier")) == tier
        ]
        best = _select_best(tier_candidates)
        if best is not None:
            best["tier"] = _normalize_tier(best.get("tier"))
            return best
    return None


def promote_artifact(
    store_dir: Path,
    *,
    domain: str,
    program_hash: str,
    artifact_path: Path,
    spec_hash: str,
    lane_id: str,
    families_mode: str,
    meta_families: Sequence[str],
    score: Mapping[str, Any],
    score_key: Sequence[int],
    score_scaled: Optional[int] = None,
    admitted_by_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    store_dir = Path(store_dir)
    suffix = artifact_path.suffix or ".json"
    dest_dir = store_dir / domain
    ensure_dir(dest_dir)
    dest_path = dest_dir / f"{program_hash}{suffix}"
    dest_path.write_bytes(artifact_path.read_bytes())
    relative_path = dest_path.relative_to(store_dir).as_posix()

    meta_profile_hash = compute_meta_profile_hash(meta_families)
    tier = "sealed" if families_mode == "sealed" else "public"
    entry = {
        "domain": domain,
        "spec_hash": spec_hash,
        "lane_id": lane_id,
        "families_mode": families_mode,
        "meta_profile_hash": meta_profile_hash,
        "tier": tier,
    }
    key = make_key(
        domain=domain,
        spec_hash=spec_hash,
        lane_id=lane_id,
        families_mode=families_mode,
        meta_families=meta_families,
    )
    return record_best(
        store_dir,
        key=key,
        program_hash=program_hash,
        domain=domain,
        spec_hash=spec_hash,
        tier=tier,
        score=score,
        score_key=score_key,
        score_scaled=score_scaled,
        store_path=relative_path,
        admitted_by_run_id=admitted_by_run_id,
        metadata=entry,
    )


def commit_best(
    store_dir: Path,
    *,
    domain: str,
    program_hash: str,
    artifact_path: Path,
    spec_hash: str,
    tier: str,
    lane_id: str,
    meta_families: Sequence[str],
    score_scaled: int,
    breaker_attempts: int,
    admitted_by_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    store_dir = Path(store_dir)
    tier = _normalize_tier(tier)
    families_mode = "sealed" if tier == "sealed" else "public"
    index = load_index(store_dir / "index.json")
    entries = index.get("entries", {})
    if not isinstance(entries, dict):
        entries = {}
    key = make_key(
        domain=domain,
        spec_hash=spec_hash,
        lane_id=lane_id,
        families_mode=families_mode,
        meta_families=meta_families,
    )
    existing_entry = entries.get(key)
    existing_hash = ""
    if isinstance(existing_entry, dict):
        existing_hash = str(existing_entry.get("program_hash", ""))
    if existing_hash == program_hash:
        return {"written": False, "upgraded": False, "entry": existing_entry or {}}
    score_key = [int(score_scaled), -int(breaker_attempts)]
    existing_key = _coerce_score_key(existing_entry) if isinstance(existing_entry, dict) else None
    if existing_key is not None and tuple(score_key) <= existing_key:
        return {"written": False, "upgraded": False, "entry": existing_entry or {}}
    entry = promote_artifact(
        store_dir,
        domain=domain,
        program_hash=program_hash,
        artifact_path=artifact_path,
        spec_hash=spec_hash,
        lane_id=lane_id,
        families_mode=families_mode,
        meta_families=meta_families,
        score={"score_scaled": int(score_scaled)},
        score_key=score_key,
        score_scaled=int(score_scaled),
        admitted_by_run_id=admitted_by_run_id,
    )
    written = entry.get("program_hash") == program_hash
    upgraded = False
    if written and tier == "sealed":
        public_key = make_key(
            domain=domain,
            spec_hash=spec_hash,
            lane_id=lane_id,
            families_mode="public",
            meta_families=meta_families,
        )
        public_entry = entries.get(public_key)
        upgraded = isinstance(public_entry, dict) and bool(public_entry)
    return {"written": bool(written), "upgraded": bool(upgraded), "entry": entry}
